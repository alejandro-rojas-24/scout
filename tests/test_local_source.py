"""Tests for scout.sources: Source protocol + LocalSource (T004)."""
from __future__ import annotations

import builtins
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
    """Replaces builtins.open; records (and optionally refuses) opens under root."""

    def __init__(self, root: pathlib.Path, fail: bool) -> None:
        self.root = str(root.resolve())
        self.fail = fail
        self.calls: list[str] = []
        self._real = builtins.open

    def __call__(self, file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            p = os.fsdecode(file)
            if os.path.abspath(p).startswith(self.root):
                self.calls.append(p)
                if self.fail:
                    raise AssertionError(f"open() under root: {p}")
        return self._real(file, *args, **kwargs)


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
    spy = _OpenSpy(root, fail=True)
    monkeypatch.setattr(builtins, "open", spy)

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

    spy = _OpenSpy(root, fail=False)
    monkeypatch.setattr(builtins, "open", spy)
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
    spy = _OpenSpy(root, fail=False)
    monkeypatch.setattr(builtins, "open", spy)
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
