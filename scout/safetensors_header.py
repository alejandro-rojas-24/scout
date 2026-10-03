"""Safetensors header reader, weight-file selection and index parsing (header-only, no tensor data)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from scout.bytelog import WEIGHT_EXTENSIONS
from scout.errors import AmbiguousWeightsError, HeaderError, NoSafetensorsError
from scout.sources import RepoFile, Source

MAX_HEADER_LEN: int = 100_000_000
DTYPE_BYTES: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}
CANONICAL_INDEX: tuple[str, ...] = (
    "model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json",
)
CANONICAL_SINGLE: tuple[str, ...] = ("model.safetensors", "diffusion_pytorch_model.safetensors")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    file: str
    data_begin: int
    data_end: int


@dataclass(frozen=True)
class SafetensorsHeader:
    path: str
    header_len: int
    metadata: dict[str, str]
    tensors: tuple[TensorInfo, ...]
    data_bytes: int


@dataclass(frozen=True)
class WeightSelection:
    index_path: str | None  # repo-relative
    single_path: str | None  # repo-relative, set when there is no index


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _join(subdir: str, name: str) -> str:
    return name if subdir == "" else f"{subdir}/{name}"


def read_header(source: Source, path: str, file_size: int | None) -> SafetensorsHeader:
    b = source.read_range(path, 0, 8)
    n = int.from_bytes(b, "little")
    if n == 0 or n > MAX_HEADER_LEN:
        raise HeaderError(f"{path}: header length {n} out of range")
    if file_size is not None and 8 + n > file_size:
        raise HeaderError(f"{path}: header length {n} exceeds file size {file_size}")
    source.log.set_header_len(path, n)  # before the next read, or preflight refuses it
    hb = source.read_range(path, 8, n)
    try:
        header = json.loads(hb.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HeaderError(f"{path}: invalid header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise HeaderError(f"{path}: header is not a JSON object")
    meta_raw = header.pop("__metadata__", {}) or {}
    if not isinstance(meta_raw, dict):
        raise HeaderError(f"{path}: __metadata__ is not an object")
    metadata = {str(k): str(v) for k, v in meta_raw.items()}

    tensors: list[TensorInfo] = []
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise HeaderError(f"{path}: tensor {name!r}: entry is not an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offs = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise HeaderError(f"{path}: tensor {name!r}: bad dtype")
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in shape):
            raise HeaderError(f"{path}: tensor {name!r}: bad shape")
        if (not isinstance(offs, list) or len(offs) != 2 or not all(_is_int(o) for o in offs)
                or not 0 <= offs[0] <= offs[1]):
            raise HeaderError(f"{path}: tensor {name!r}: bad data_offsets")
        begin, end = offs
        numel = math.prod(shape)
        nbytes = end - begin
        if dtype in DTYPE_BYTES and nbytes != numel * DTYPE_BYTES[dtype]:
            raise HeaderError(
                f"{path}: tensor {name!r}: {nbytes} bytes != {numel} * {DTYPE_BYTES[dtype]} ({dtype})"
            )
        tensors.append(TensorInfo(name, dtype, tuple(shape), numel, nbytes, path, begin, end))

    tensors.sort(key=lambda t: (t.data_begin, t.data_end, t.name))
    cursor = 0
    for t in tensors:
        if t.data_begin != cursor:
            raise HeaderError(f"{path}: non-contiguous offsets at tensor {t.name!r}")
        cursor = t.data_end
    data_bytes = cursor
    if file_size is not None and 8 + n + data_bytes != file_size:
        raise HeaderError(f"{path}: size mismatch")
    return SafetensorsHeader(path, n, metadata, tuple(tensors), data_bytes)


def select_weight_files(files: list[RepoFile], subdir: str) -> WeightSelection:
    names: set[str] = set()
    for f in files:
        parent, sep, base = f.path.rpartition("/")
        if parent == subdir:
            names.add(base)
    for name in CANONICAL_INDEX:
        if name in names:
            return WeightSelection(_join(subdir, name), None)
    for name in CANONICAL_SINGLE:
        if name in names:
            return WeightSelection(None, _join(subdir, name))
    others_idx = sorted(n for n in names if n.endswith(".safetensors.index.json"))
    others = sorted(n for n in names if n.endswith(".safetensors"))
    if len(others_idx) == 1:
        return WeightSelection(_join(subdir, others_idx[0]), None)
    if len(others) == 1:
        return WeightSelection(None, _join(subdir, others[0]))
    if not others_idx and not others:
        found = sorted(n for n in names if n.lower().endswith(WEIGHT_EXTENSIONS))
        raise NoSafetensorsError(f"{subdir or '.'}: no safetensors; found {found}")
    raise AmbiguousWeightsError(
        f"{subdir or '.'}: ambiguous weight files; index candidates {others_idx}, "
        f"safetensors candidates {others}"
    )


def parse_index(raw: bytes, subdir: str) -> tuple[list[str], int | None]:
    try:
        j = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HeaderError(f"invalid index JSON: {exc}") from exc
    wm = j.get("weight_map") if isinstance(j, dict) else None
    if not isinstance(wm, dict):
        raise HeaderError("index has no weight_map object")
    files = sorted({_join(subdir, v) for v in wm.values()})
    meta = j.get("metadata") or {}
    total = meta.get("total_size") if isinstance(meta, dict) else None
    try:
        return files, (int(total) if total is not None else None)
    except (TypeError, ValueError) as exc:
        raise HeaderError(f"bad total_size {total!r}") from exc
