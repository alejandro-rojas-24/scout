"""Tests for scout.safetensors_header (T006)."""
from __future__ import annotations

import json
import pathlib

import pytest

from scout.bytelog import ByteLog
from scout.errors import AmbiguousWeightsError, HeaderError, NoSafetensorsError
from scout.safetensors_header import parse_index, read_header, select_weight_files
from scout.sources import LocalSource, RepoFile
from tests.helpers.st_fixtures import write_safetensors


def _src(root: pathlib.Path) -> tuple[LocalSource, ByteLog]:
    log = ByteLog()
    s = LocalSource(root, log)
    s.resolve()
    return s, log


def _raw(header: dict, data: bytes = b"") -> bytes:
    hb = json.dumps(header).encode()
    return len(hb).to_bytes(8, "little") + hb + data


def test_read_header_ok(tmp_path):
    spec = {"a": ("BF16", (2, 3)), "b": ("F32", ()), "c": ("I8", (0, 4)), "d": ("F32", (5,))}
    n = write_safetensors(tmp_path / "m.safetensors", spec, {"format": "pt", "k": 3})
    s, log = _src(tmp_path)
    size = (tmp_path / "m.safetensors").stat().st_size
    h = read_header(s, "m.safetensors", size)
    assert h.header_len == n and h.path == "m.safetensors"
    assert h.metadata == {"format": "pt", "k": "3"}
    by = {t.name: t for t in h.tensors}
    assert by["a"].shape == (2, 3) and by["a"].numel == 6 and by["a"].nbytes == 12
    assert by["b"].shape == () and by["b"].numel == 1 and by["b"].nbytes == 4
    assert by["c"].numel == 0 and by["c"].nbytes == 0
    assert all(t.file == "m.safetensors" for t in h.tensors)
    begins = [(t.data_begin, t.data_end) for t in h.tensors]
    assert begins == sorted(begins) and begins[0][0] == 0
    assert h.data_bytes == 12 + 4 + 0 + 20
    t = log.totals
    assert t["header"] == 8 + n and t["weight"] == 0


def test_read_header_no_size(tmp_path):
    write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", (2,))})
    s, _ = _src(tmp_path)
    assert read_header(s, "m.safetensors", None).data_bytes == 8


def test_empty_tensors(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(_raw({}))
    s, _ = _src(tmp_path)
    h = read_header(s, "m.safetensors", (tmp_path / "m.safetensors").stat().st_size)
    assert h.tensors == () and h.data_bytes == 0


def test_size_mismatch(tmp_path):
    write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", (2,))})
    s, _ = _src(tmp_path)
    size = (tmp_path / "m.safetensors").stat().st_size
    with pytest.raises(HeaderError, match="size mismatch"):
        read_header(s, "m.safetensors", size + 1)


def test_corrupt_len(tmp_path):
    (tmp_path / "m.safetensors").write_bytes((2**40).to_bytes(8, "little") + b"x" * 16)
    s, log = _src(tmp_path)
    with pytest.raises(HeaderError, match="out of range"):
        read_header(s, "m.safetensors", 24)
    assert log.totals["header"] == 8 and log.totals["weight"] == 0


def test_len_exceeds_file(tmp_path):
    (tmp_path / "m.safetensors").write_bytes((100).to_bytes(8, "little") + b"x" * 16)
    s, _ = _src(tmp_path)
    with pytest.raises(HeaderError):
        read_header(s, "m.safetensors", 24)


def test_noncontiguous(tmp_path):
    hdr = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
           "b": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]}}
    (tmp_path / "m.safetensors").write_bytes(_raw(hdr, b"\0" * 6))
    s, _ = _src(tmp_path)
    with pytest.raises(HeaderError, match="non-contiguous"):
        read_header(s, "m.safetensors", None)


def test_dtype_size_mismatch(tmp_path):
    hdr = {"a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 4]}}
    (tmp_path / "m.safetensors").write_bytes(_raw(hdr, b"\0" * 4))
    s, _ = _src(tmp_path)
    with pytest.raises(HeaderError, match="a"):
        read_header(s, "m.safetensors", None)


def test_unknown_dtype_accepted(tmp_path):
    hdr = {"a": {"dtype": "WEIRD", "shape": [2], "data_offsets": [0, 3]}}
    (tmp_path / "m.safetensors").write_bytes(_raw(hdr, b"\0" * 3))
    s, _ = _src(tmp_path)
    assert read_header(s, "m.safetensors", None).tensors[0].dtype == "WEIRD"


@pytest.mark.parametrize("entry", [
    {"dtype": "U8", "shape": [2]},
    {"dtype": 3, "shape": [2], "data_offsets": [0, 2]},
    {"dtype": "U8", "shape": [-1], "data_offsets": [0, 2]},
    {"dtype": "U8", "shape": [2], "data_offsets": [2, 0]},
    {"dtype": "U8", "shape": [2], "data_offsets": [0]},
    "nope",
])
def test_bad_entries(tmp_path, entry):
    (tmp_path / "m.safetensors").write_bytes(_raw({"a": entry}, b"\0" * 2))
    s, _ = _src(tmp_path)
    with pytest.raises(HeaderError):
        read_header(s, "m.safetensors", None)


def test_bad_json_and_non_dict(tmp_path):
    for body in (b"{not json", b"[1]"):
        (tmp_path / "m.safetensors").write_bytes(len(body).to_bytes(8, "little") + body)
        s, _ = _src(tmp_path)
        with pytest.raises(HeaderError):
            read_header(s, "m.safetensors", None)


def _files(*paths):
    return [RepoFile(p, 1) for p in paths]


def test_select():
    sel = select_weight_files(_files("model.safetensors.index.json", "model.safetensors", "config.json"), "")
    assert (sel.index_path, sel.single_path) == ("model.safetensors.index.json", None)
    sel = select_weight_files(_files("model.safetensors"), "")
    assert (sel.index_path, sel.single_path) == (None, "model.safetensors")
    sel = select_weight_files(_files("vae/diffusion_pytorch_model.safetensors", "vae/config.json"), "vae")
    assert (sel.index_path, sel.single_path) == (None, "vae/diffusion_pytorch_model.safetensors")
    sel = select_weight_files(_files("te/diffusion_pytorch_model.safetensors.index.json"), "te")
    assert sel.index_path == "te/diffusion_pytorch_model.safetensors.index.json"
    sel = select_weight_files(_files("model.fp16.safetensors"), "")
    assert (sel.index_path, sel.single_path) == (None, "model.fp16.safetensors")
    sel = select_weight_files(_files("x.safetensors.index.json", "x-00001.safetensors"), "")
    assert sel.index_path == "x.safetensors.index.json"
    with pytest.raises(AmbiguousWeightsError):
        select_weight_files(_files("model.fp16.safetensors", "model.bf16.safetensors"), "")
    with pytest.raises(NoSafetensorsError, match="pytorch_model.bin"):
        select_weight_files(_files("pytorch_model.bin", "config.json"), "")
    with pytest.raises(NoSafetensorsError):
        select_weight_files(_files("nested/model.safetensors", "a/b/model.safetensors"), "")
    with pytest.raises(NoSafetensorsError):
        select_weight_files(_files("sub/deep/model.safetensors"), "sub")
    with pytest.raises(NoSafetensorsError):
        select_weight_files(_files("model.safetensors"), "sub")


def test_parse_index():
    raw = json.dumps({"metadata": {"total_size": 123},
                      "weight_map": {"a": "s2.safetensors", "b": "s1.safetensors", "c": "s2.safetensors"}}).encode()
    assert parse_index(raw, "te") == (["te/s1.safetensors", "te/s2.safetensors"], 123)
    assert parse_index(raw, "")[0] == ["s1.safetensors", "s2.safetensors"]
    raw2 = json.dumps({"weight_map": {"a": "s.safetensors"}}).encode()
    assert parse_index(raw2, "") == (["s.safetensors"], None)
    with pytest.raises(HeaderError):
        parse_index(b'{"x": 1}', "")
    with pytest.raises(HeaderError):
        parse_index(b"nope", "")
