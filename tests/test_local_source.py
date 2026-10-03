"""Tests for scout.sources: Source protocol + LocalSource (T004)."""
from __future__ import annotations

import builtins
import io
import json
import os
import pathlib
import re

import pytest

from scout.bytelog import ByteLog
from scout.errors import HeaderError, ScoutError, WeightReadRefused
from scout.sources import LocalSource, RepoFile, Source
from tests.helpers.st_fixtures import dense_tensors, safetensors_bytes, write_safetensors

CONFIG = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 8}


def _make_repo(root: pathlib.Path) -> dict[str, int]:
    """A small local model folder. Returns {relpath: size} of the visible files."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(CONFIG))
    (root / "README.md").write_text("# hi\n")
    write_safetensors(root / "model.safetensors", dense_tensors(n_layers=1))
    (root / "pytorch_model.bin").write_bytes(b"\x00" * 32)
    (root / "sub").mkdir()
    (root / "sub" / "a.json").write_text("{}")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".hidden.json").write_text("{}")
    return {
        p.relative_to(root).as_posix(): p.stat().st_size
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    }


class _OpenSpy:
    """Records (and optionally refuses) opens under root via builtins.open, io.open and os.open."""

    def __init__(self, root: pathlib.Path, fail: bool) -> None:
        self.root = str(root.resolve())
        self.fail = fail
        self.calls: list[str] = []
        self._real_open = builtins.open
        self._real_os_open = os.open

    def _check(self, file) -> None:
        if isinstance(file, (str, bytes, os.PathLike)):
            p = os.fsdecode(file)
            if os.path.abspath(p).startswith(self.root):
                self.calls.append(p)
                if self.fail:
                    raise AssertionError(f"open() under root: {p}")

    def open(self, file, *args, **kwargs):
        self._check(file)
        return self._real_open(file, *args, **kwargs)

    def os_open(self, path, *args, **kwargs):
        self._check(path)
        return self._real_os_open(path, *args, **kwargs)

    def install(self, monkeypatch) -> "_OpenSpy":
        monkeypatch.setattr(builtins, "open", self.open)
        monkeypatch.setattr(io, "open", self.open)
        monkeypatch.setattr(os, "open", self.os_open)
        return self


def test_init_attributes(tmp_path):
    root = tmp_path / "m"
    _make_repo(root)
    src = LocalSource(root, ByteLog())
    r = str(root.resolve())
    assert (src.kind, src.repo, src.local_path) == ("local", f"local:{r}", r)
    assert src.requested_revision is None and src.endpoint is None
    assert src.revision_kind == "local-stat-hash" and src.revision_sha == ""
    assert isinstance(src.log, ByteLog)
    with pytest.raises(ScoutError):
        LocalSource(root / "config.json", ByteLog())
    with pytest.raises(FileNotFoundError):
        LocalSource(root / "nope", ByteLog())
    with pytest.raises(ScoutError):
        src.files()  # before resolve()


def test_resolve_no_reads(tmp_path, monkeypatch):
    root = tmp_path / "m"
    expected = _make_repo(root)
    log = ByteLog()
    src = LocalSource(root, log)
    spy = _OpenSpy(root, fail=True).install(monkeypatch)

    src.resolve()
    files = src.files()
    assert [f.path for f in files] == sorted(expected)
    assert {f.path: f.size for f in files} == expected
    assert all(isinstance(f, RepoFile) for f in files)
    sha1 = src.revision_sha
    assert re.fullmatch(r"[0-9a-f]{64}", sha1)

    src.resolve()
    assert src.revision_sha == sha1

    ev = [e for e in log.events if e.path == "@local/stat"]
    assert len(ev) == 2
    assert ev[0].event == "fetch" and ev[0].source == "local" and ev[0].bytes == 0
    assert ev[0].note == f"{len(expected)} files"
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}

    st = os.stat(root / "config.json")
    os.utime(root / "config.json", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    src.resolve()
    assert src.revision_sha != sha1
    assert spy.calls == []


def test_read_file_meta(tmp_path):
    root = tmp_path / "m"
    _make_repo(root)
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    raw = (root / "config.json").read_bytes()

    data = src.read_file("config.json")
    assert data == raw
    assert log.totals == {"meta": len(raw), "header": 0, "weight": 0}
    assert log.reserved == 0
    ev = log.events[-1]
    assert (ev.event, ev.source, ev.path, ev.url, ev.bytes) == ("fetch", "local", "config.json", None, len(raw))
    assert ev.range == (0, len(raw))

    assert src.read_file("sub/a.json") == b"{}"
    assert src.read_file("missing.json") is None
    assert src.read_file(".git/HEAD") is None
    assert log.totals["meta"] == len(raw) + 2


def test_read_range_header(tmp_path):
    root = tmp_path / "m"
    _make_repo(root)
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    full = (root / "model.safetensors").read_bytes()

    b = src.read_range("model.safetensors", 0, 8)
    n = int.from_bytes(b, "little")
    log.set_header_len("model.safetensors", n)
    hb = src.read_range("model.safetensors", 8, n)
    assert hb == full[8 : 8 + n]
    json.loads(hb)
    assert log.totals == {"meta": 0, "header": 8 + n, "weight": 0}
    assert log.reserved == 0

    with pytest.raises(ScoutError):
        src.read_range("not-listed.safetensors", 0, 8)


def test_read_range_short_read(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "notes.txt").write_bytes(b"abc")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    with pytest.raises(HeaderError, match="short read notes.txt"):
        src.read_range("notes.txt", 1, 10)
    assert log.totals["meta"] == 2  # the bytes actually read are still logged
    assert log.reserved == 0


def test_weight_refused(tmp_path, monkeypatch):
    root = tmp_path / "m"
    _make_repo(root)
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    n = int.from_bytes((root / "model.safetensors").read_bytes()[:8], "little")
    log.set_header_len("model.safetensors", n)

    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    with pytest.raises(WeightReadRefused):
        src.read_range("model.safetensors", 8 + n, 4)
    with pytest.raises(WeightReadRefused):
        src.read_file("model.safetensors")
    assert spy.calls == []
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}
    assert log.reserved == 0
    assert log.events[-1].event == "refused"


def test_bin_refused(tmp_path, monkeypatch):
    root = tmp_path / "m"
    _make_repo(root)
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    with pytest.raises(WeightReadRefused):
        src.read_file("pytorch_model.bin")
    assert spy.calls == []
    assert log.totals["weight"] == 0 and log.reserved == 0


def test_threshold_releases_reservation(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "big.json").write_bytes(b"x" * 200)
    log = ByteLog(threshold_bytes=100)
    src = LocalSource(root, log)
    src.resolve()
    with pytest.raises(ScoutError):
        src.read_file("big.json")
    assert log.reserved == 0 and log.totals["meta"] == 0


def test_changed_since_resolve(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "c.json").write_bytes(b"{}")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    (root / "c.json").write_bytes(b'{"a": 1}')
    with pytest.raises(ScoutError, match="changed since resolve"):
        src.read_file("c.json")
    assert log.reserved == 0 and log.totals["meta"] == 0


def _hf_cache(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, bytes]:
    """models--Org--Name/{blobs/<hash>, snapshots/<sha>/<file> -> ../../blobs/<hash>}."""
    repo = tmp_path / "hub" / "models--Org--Name"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    (snap / "text_encoder").mkdir(parents=True)
    st = safetensors_bytes(dense_tensors(n_layers=1))
    (blobs / ("1" * 64)).write_bytes(st)
    (blobs / ("2" * 40)).write_text(json.dumps(CONFIG))
    os.symlink(f"../../blobs/{'1' * 64}", snap / "model.safetensors")
    os.symlink(f"../../blobs/{'2' * 40}", snap / "config.json")
    os.symlink(f"../../../blobs/{'2' * 40}", snap / "text_encoder" / "config.json")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    return snap, blobs, st


def test_symlinked_files(tmp_path):
    snap, blobs, st = _hf_cache(tmp_path)
    log = ByteLog()
    src = LocalSource(snap, log)
    src.resolve()
    sizes = {f.path: f.size for f in src.files()}
    assert sizes == {
        "config.json": len(json.dumps(CONFIG)),
        "model.safetensors": len(st),
        "text_encoder/config.json": len(json.dumps(CONFIG)),
    }
    # read_header-style reads
    n = int.from_bytes(src.read_range("model.safetensors", 0, 8), "little")
    log.set_header_len("model.safetensors", n)
    assert src.read_range("model.safetensors", 8, n) == st[8 : 8 + n]
    assert src.read_file("text_encoder/config.json") == json.dumps(CONFIG).encode()
    assert log.totals["header"] == 8 + n and log.totals["weight"] == 0

    # symlinked directories are not followed
    os.symlink(str(blobs), snap / "linked_dir")
    src.resolve()
    assert not any(f.path.startswith("linked_dir") for f in src.files())

    # a broken link raises
    os.symlink("../../blobs/missing", snap / "tokenizer.json")
    with pytest.raises(ScoutError, match="broken symlink tokenizer.json"):
        src.resolve()


def test_hidden_skipped(tmp_path):
    root = tmp_path / "m"
    _make_repo(root)
    (root / "sub" / ".cache").mkdir()
    (root / "sub" / ".cache" / "x.json").write_text("{}")
    src = LocalSource(root, ByteLog())
    src.resolve()
    paths = [f.path for f in src.files()]
    assert not any(p.startswith(".git") for p in paths)
    assert ".hidden.json" not in paths
    assert not any("/." in p or p.startswith(".") for p in paths)
    assert "sub/a.json" in paths


def test_protocol_conformance(tmp_path):
    root = tmp_path / "m"
    _make_repo(root)
    src: Source = LocalSource(root, ByteLog())  # static conformance; runtime attribute check below
    for name in ("kind", "repo", "revision_sha", "revision_kind", "requested_revision",
                 "local_path", "endpoint", "log", "resolve", "files", "read_file", "read_range"):
        assert hasattr(src, name), name


def test_stat_note_lists_symlink_targets_relative(tmp_path):
    """Review r2 item 4: the note is persisted in the Card, so it holds no absolute paths."""
    snap, blobs, _ = _hf_cache(tmp_path)
    log = ByteLog()
    LocalSource(snap, log).resolve()
    note = log.events[-1].note
    assert note.startswith("3 files; symlinks: ")
    assert f"model.safetensors -> ../../blobs/{'1' * 64}" in note
    assert f"text_encoder/config.json -> ../../blobs/{'2' * 40}" in note
    assert str(tmp_path) not in note and str(tmp_path.resolve()) not in note


def test_stat_note_outside_and_capped(tmp_path):
    """Targets outside root/../.. become <outside>/basename; at most 20 entries plus (+K more)."""
    root = tmp_path / "a" / "b" / "m"
    root.mkdir(parents=True)
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "far.json").write_text("{}")
    os.symlink(ext / "far.json", root / "aaa.json")
    for i in range(24):
        (root / f"real{i:02d}.txt").write_text("x")
        os.symlink(f"real{i:02d}.txt", root / f"link{i:02d}.txt")
    log = ByteLog()
    LocalSource(root, log).resolve()
    note = log.events[-1].note
    assert note.startswith("49 files; symlinks: ")
    assert "aaa.json -> <outside>/far.json" in note
    assert "link00.txt -> real00.txt" in note
    assert "link18.txt -> real18.txt" in note
    assert "link19.txt" not in note  # 25 links sorted by rel: aaa.json + link00..link18 shown
    assert note.endswith("(+5 more)")
    assert note.count(" -> ") == 20
    assert str(tmp_path) not in note and str(tmp_path.resolve()) not in note


def test_alias_to_safetensors_refused(tmp_path, monkeypatch):
    """Review repro 1: notes.txt -> model.safetensors must not be readable as meta."""
    root = tmp_path / "m"
    _make_repo(root)
    os.symlink("model.safetensors", root / "notes.txt")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    with pytest.raises(WeightReadRefused):
        src.read_file("notes.txt")
    with pytest.raises(WeightReadRefused):
        src.read_range("notes.txt", 0, 8)  # no header semantics through an alias
    assert spy.calls == []
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}
    assert log.reserved == 0
    assert log.events[-1].event == "refused"


def test_alias_outside_root_bin_refused(tmp_path, monkeypatch):
    """Review repro 2: readme.md -> /elsewhere/outside.bin must not be readable as meta."""
    root = tmp_path / "m"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "outside.bin").write_bytes(b"\x01" * 64)
    os.symlink(elsewhere / "outside.bin", root / "readme.md")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    assert {f.path: f.size for f in src.files()} == {"readme.md": 64}  # outside-root targets stay listed
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    with pytest.raises(WeightReadRefused):
        src.read_file("readme.md")
    with pytest.raises(WeightReadRefused):
        src.read_range("readme.md", 0, 4)
    assert spy.calls == []
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}
    assert log.reserved == 0


def test_inode_alias_refused(tmp_path, monkeypatch):
    """A hard link with a non-weight name sharing an inode with a listed weight file."""
    root = tmp_path / "m"
    root.mkdir()
    (root / "pytorch_model.bin").write_bytes(b"\x02" * 32)
    os.link(root / "pytorch_model.bin", root / "data.txt")
    write_safetensors(root / "model.safetensors", dense_tensors(n_layers=1))
    os.link(root / "model.safetensors", root / "blob")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    for path in ("data.txt", "blob"):
        with pytest.raises(WeightReadRefused):
            src.read_file(path)
        with pytest.raises(WeightReadRefused):
            src.read_range(path, 0, 8)
    assert spy.calls == []
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}
    assert log.reserved == 0


def test_hf_cache_blob_names_are_not_aliases(tmp_path):
    """Extensionless blob targets do not taint a non-weight link name (config.json stays meta)."""
    snap, _, _ = _hf_cache(tmp_path)
    log = ByteLog()
    src = LocalSource(snap, log)
    src.resolve()
    data = src.read_file("config.json")
    assert log.totals == {"meta": len(data), "header": 0, "weight": 0}
    assert log.events[-1].path == "config.json"


def test_walk_error_raises(tmp_path, monkeypatch):
    root = tmp_path / "m"
    _make_repo(root)
    real_scandir = os.scandir
    bad = str(root.resolve() / "sub")

    def scandir(path="."):
        if os.fsdecode(path) == bad:
            raise PermissionError(13, "Permission denied", bad)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    src = LocalSource(root, ByteLog())
    with pytest.raises(ScoutError, match="cannot list .*sub"):
        src.resolve()


def test_symlink_chain_aliases_refused(tmp_path, monkeypatch):
    """Review r2 item 1: readme.md -> ext/model.bin -> ext/<hash>; realpath alone says 'hash'."""
    root = tmp_path / "m"
    root.mkdir()
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / ("9" * 64)).write_bytes(b"\x03" * 64)
    os.symlink(ext / ("9" * 64), ext / "model.bin")
    os.symlink(ext / "model.bin", root / "readme.md")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    assert {f.path: f.size for f in src.files()} == {"readme.md": 64}
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    with pytest.raises(WeightReadRefused):
        src.read_file("readme.md")
    with pytest.raises(WeightReadRefused):
        src.read_range("readme.md", 0, 4)
    assert spy.calls == []
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}
    assert log.reserved == 0


def test_symlink_chain_middle_hop_relative(tmp_path):
    """A relative middle hop (notes.txt -> sub/w.pt -> ../blobhash) is followed hop by hop."""
    root = tmp_path / "m"
    (root / "sub").mkdir(parents=True)
    (root / "blobhash").write_bytes(b"\x04" * 16)
    os.symlink("../blobhash", root / "sub" / "w.pt")
    os.symlink("sub/w.pt", root / "notes.txt")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    with pytest.raises(WeightReadRefused):
        src.read_file("notes.txt")
    # blobhash shares the inode with notes.txt and sub/w.pt, so it is refused too (errs safe)
    with pytest.raises(WeightReadRefused):
        src.read_file("blobhash")
    assert log.totals["weight"] == 0 and log.totals["meta"] == 0


def test_symlink_chain_too_long(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "target.json").write_text("{}")
    prev = "target.json"
    for i in range(41):
        name = f"l{i:02d}.json"
        os.symlink(prev, root / name)
        prev = name
    with pytest.raises(ScoutError, match="symlink chain too long"):
        LocalSource(root, ByteLog()).resolve()


def test_symlink_cycle_raises(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    os.symlink("b.json", root / "a.json")
    os.symlink("a.json", root / "b.json")
    with pytest.raises(ScoutError, match="symlink chain too long"):
        LocalSource(root, ByteLog()).resolve()


def test_hidden_weight_named_hardlink_refused(tmp_path, monkeypatch):
    """Review r2 item 2: .model.bin hardlinked to notes.txt; hidden names still count as aliases."""
    root = tmp_path / "m"
    (root / "sub").mkdir(parents=True)
    (root / "notes.txt").write_bytes(b"\x05" * 32)
    os.link(root / "notes.txt", root / ".model.bin")
    (root / "sub" / "info.md").write_bytes(b"\x06" * 32)
    os.link(root / "sub" / "info.md", root / "sub" / ".w.safetensors")
    (root / "plain.txt").write_text("ok")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    assert [f.path for f in src.files()] == ["notes.txt", "plain.txt", "sub/info.md"]
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    for path in ("notes.txt", "sub/info.md"):
        with pytest.raises(WeightReadRefused):
            src.read_file(path)
        with pytest.raises(WeightReadRefused):
            src.read_range(path, 0, 8)
    assert spy.calls == []
    assert src.read_file("plain.txt") == b"ok"
    assert log.totals == {"meta": 2, "header": 0, "weight": 0}
    assert log.reserved == 0


def test_hidden_broken_link_ignored(tmp_path):
    """Hidden files are only alias sources; a broken hidden link does not fail resolve()."""
    root = tmp_path / "m"
    root.mkdir()
    (root / "config.json").write_text("{}")
    os.symlink("missing", root / ".dangling.bin")
    src = LocalSource(root, ByteLog())
    src.resolve()
    assert src.read_file("config.json") == b"{}"


def test_new_hardlink_after_resolve_detected(tmp_path):
    """A weight-named hard link added after resolve() changes st_nlink, so the read is refused."""
    root = tmp_path / "m"
    root.mkdir()
    (root / "notes.txt").write_bytes(b"abc")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    os.link(root / "notes.txt", root / ".late.bin")
    with pytest.raises(ScoutError, match="changed since resolve"):
        src.read_file("notes.txt")
    assert log.totals["meta"] == 0 and log.reserved == 0


@pytest.mark.parametrize("kind", ["hardlink", "symlink", "chain"])
@pytest.mark.parametrize("other", ["pytorch_model.bin", "x.pt"])
def test_safetensors_name_over_other_weight_format_refused(tmp_path, monkeypatch, kind, other):
    """Review r2 item 3: model.safetensors aliasing a .bin/.pt must not get safetensors header semantics."""
    root = tmp_path / "m"
    root.mkdir()
    ext = tmp_path / "ext"
    ext.mkdir()
    payload = (16).to_bytes(8, "little") + b"{" + b" " * 14 + b"}" + b"\x07" * 64
    if kind == "hardlink":
        (root / other).write_bytes(payload)
        os.link(root / other, root / "model.safetensors")
    elif kind == "symlink":
        (ext / other).write_bytes(payload)
        os.symlink(ext / other, root / "model.safetensors")
    else:
        (ext / ("8" * 64)).write_bytes(payload)
        os.symlink(ext / ("8" * 64), ext / other)
        os.symlink(ext / other, root / "model.safetensors")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    spy = _OpenSpy(root, fail=False).install(monkeypatch)
    with pytest.raises(WeightReadRefused):
        src.read_range("model.safetensors", 0, 8)
    assert spy.calls == []
    assert log.totals == {"meta": 0, "header": 0, "weight": 0}
    assert log.reserved == 0
    assert log.events[-1].event == "refused"


def test_safetensors_over_safetensors_or_blob_allowed(tmp_path):
    """Same-format and extensionless aliases keep normal safetensors header reads."""
    root = tmp_path / "m"
    root.mkdir()
    ext = tmp_path / "ext"
    ext.mkdir()
    st = safetensors_bytes(dense_tensors(n_layers=1))
    (ext / ("7" * 64)).write_bytes(st)
    os.symlink(ext / ("7" * 64), ext / "other.safetensors")
    os.symlink(ext / "other.safetensors", root / "model.safetensors")
    log = ByteLog()
    src = LocalSource(root, log)
    src.resolve()
    n = int.from_bytes(src.read_range("model.safetensors", 0, 8), "little")
    log.set_header_len("model.safetensors", n)
    assert src.read_range("model.safetensors", 8, n) == st[8 : 8 + n]
    assert log.totals == {"meta": 0, "header": 8 + n, "weight": 0}
