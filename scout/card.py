"""Card v0: build (in memory), atomic write (JSON + Parquet) and load.

The Card is the only persisted artifact (invariant 1). It holds header-derived
structure, parsed metadata claims and the embedded fetch log; it never holds weights.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

import scout
from scout.bytelog import ByteLog
from scout.errors import ScoutError
from scout.metadata import ConfigInfo, ModelCardInfo, PipelineInfo
from scout.safetensors_header import SafetensorsHeader
from scout.sources import Source
from scout.structure import Structure

SCHEMA_VERSION = "card.v0"
STATS_SLOTS: tuple[str, ...] = (
    "tensor_stats", "spectral_topk", "sigma_curves", "tokenizer_minhash", "attribution",
)
NULL_STAT_COLUMNS: tuple[str, ...] = (
    "stat_mean", "stat_std", "stat_fro_norm", "stat_spectral_topk", "stat_sigma_curve",
)


def _list_of(t: pa.DataType) -> pa.DataType:
    # Child named "element": the Parquet-compliant name pyarrow reads back, so a
    # write/read round trip yields a schema equal to PARQUET_SCHEMA.
    return pa.list_(pa.field("element", t))


# plan.md section 4.3, same order. Columns without "(nullable)" there are non-nullable.
PARQUET_SCHEMA: pa.Schema = pa.schema([
    pa.field("repo", pa.string(), nullable=False),
    pa.field("revision_sha", pa.string(), nullable=False),
    pa.field("component", pa.string(), nullable=False),
    pa.field("name", pa.string(), nullable=False),
    pa.field("collapsed_name", pa.string(), nullable=False),
    pa.field("file", pa.string(), nullable=False),
    pa.field("dtype", pa.string(), nullable=False),
    pa.field("shape", _list_of(pa.int64()), nullable=False),
    pa.field("numel", pa.int64(), nullable=False),
    pa.field("nbytes", pa.int64(), nullable=False),
    pa.field("data_begin", pa.int64(), nullable=False),
    pa.field("data_end", pa.int64(), nullable=False),
    pa.field("stack_prefix", pa.string(), nullable=True),
    pa.field("block_index", pa.int32(), nullable=True),
    pa.field("expert_index", pa.int32(), nullable=True),
    pa.field("stat_mean", pa.float64(), nullable=True),
    pa.field("stat_std", pa.float64(), nullable=True),
    pa.field("stat_fro_norm", pa.float64(), nullable=True),
    pa.field("stat_spectral_topk", _list_of(pa.float64()), nullable=True),
    pa.field("stat_sigma_curve", _list_of(pa.float64()), nullable=True),
])

_LOCAL_PREFIX = "local:"


@dataclass(frozen=True)
class CardPaths:
    json_path: pathlib.Path
    parquet_path: pathlib.Path


@dataclass(frozen=True)
class BuiltCard:
    card: dict
    table: pa.Table
    paths: CardPaths  # target locations (not yet written)


# ---------------------------------------------------------------------- paths


def repo_slug(repo: str) -> str:
    if repo.startswith(_LOCAL_PREFIX):
        abs_path = repo[len(_LOCAL_PREFIX):]
        return "local__" + re.sub(r"[^A-Za-z0-9._-]", "_", abs_path.strip("/"))
    return repo.replace("/", "__")


def component_slug(component: str | None) -> str:
    return "_model" if component is None else component


def _check_segment(seg: str, what: str) -> None:
    # Every path segment comes from repo/model_index data; none may leave out_dir.
    if seg in ("", ".", "..") or "/" in seg or "\\" in seg or "\x00" in seg:
        raise ScoutError(f"unsafe {what} for a Card path: {seg!r}")


def card_paths(out_dir: pathlib.Path, repo: str, revision_sha: str, component: str | None) -> CardPaths:
    rs, cs = repo_slug(repo), component_slug(component)
    _check_segment(rs, "repo slug")
    _check_segment(revision_sha, "revision_sha")
    _check_segment(cs, "component")
    base = pathlib.Path(out_dir) / rs / revision_sha
    return CardPaths(base / f"{cs}.card.json", base / f"{cs}.tensors.parquet")


# ---------------------------------------------------------------------- build


def _build_table(repo: str, revision_sha: str, component: str | None,
                 headers: list[SafetensorsHeader], structure: Structure) -> pa.Table:
    tensors = sorted((t for h in headers for t in h.tensors),
                     key=lambda t: (t.file, t.data_begin, t.data_end, t.name))
    comp = component if component is not None else ""
    cols: dict[str, list] = {f.name: [] for f in PARQUET_SCHEMA}
    for t in tensors:
        pl = structure.places.get(t.name)
        if pl is None:
            raise ScoutError(f"tensor {t.name!r} in {t.file} is missing from the structure analysis")
        cols["repo"].append(repo)
        cols["revision_sha"].append(revision_sha)
        cols["component"].append(comp)
        cols["name"].append(t.name)
        cols["collapsed_name"].append(pl.collapsed_name)
        cols["file"].append(t.file)
        cols["dtype"].append(t.dtype)
        cols["shape"].append(list(t.shape))
        cols["numel"].append(t.numel)
        cols["nbytes"].append(t.nbytes)
        cols["data_begin"].append(t.data_begin)
        cols["data_end"].append(t.data_end)
        cols["stack_prefix"].append(pl.stack_prefix)
        cols["block_index"].append(pl.block_index)
        cols["expert_index"].append(pl.expert_index)
    for c in NULL_STAT_COLUMNS:
        cols[c] = [None] * len(tensors)
    table = pa.Table.from_pydict(cols, schema=PARQUET_SCHEMA)
    return table.replace_schema_metadata({
        b"schema_version": SCHEMA_VERSION.encode(),
        b"repo": repo.encode("utf-8"),
        b"revision_sha": revision_sha.encode("utf-8"),
        b"component": comp.encode("utf-8"),
    })


def build_card(*, source: Source, component: str | None, pipeline: PipelineInfo | None,
               config: ConfigInfo, model_card: ModelCardInfo, headers: list[SafetensorsHeader],
               file_sizes: dict[str, int | None], index_path: str | None, index_total_size: int | None,
               structure: Structure, log: ByteLog, scanned_at: str, elapsed_s: float,
               out_dir: pathlib.Path) -> BuiltCard:
    n_header_tensors = sum(len(h.tensors) for h in headers)
    if n_header_tensors != structure.n_tensors:
        raise ScoutError(
            f"structure has {structure.n_tensors} tensors but headers have {n_header_tensors}")
    paths = card_paths(out_dir, source.repo, source.revision_sha, component)
    files = [
        {
            "path": h.path,
            "size_bytes": file_sizes.get(h.path),
            "header_len": h.header_len,
            "data_bytes": h.data_bytes,
            "n_tensors": len(h.tensors),
            "metadata": dict(h.metadata),
        }
        for h in sorted(headers, key=lambda h: h.path)
    ]
    card = {
        "schema_version": SCHEMA_VERSION,
        "key": {"repo": source.repo, "revision_sha": source.revision_sha, "component": component},
        "source": {
            "kind": source.kind,
            "requested_revision": source.requested_revision,
            "revision_kind": source.revision_kind,
            "local_path": source.local_path,
            "endpoint": source.endpoint,
        },
        "scan": {"scanned_at": scanned_at, "scout_version": scout.__version__,
                 "elapsed_s": float(elapsed_s)},
        "model_card": model_card.to_dict(),
        "config": config.to_dict(),
        "pipeline": pipeline.to_dict() if pipeline is not None else None,
        "weights": {
            "format": "safetensors",
            "index_path": index_path,
            "index_total_size": index_total_size,
            "files": files,
            "n_tensors": structure.n_tensors,
            "params_total": structure.params_total,
            "tensor_bytes_total": sum(t.nbytes for h in headers for t in h.tensors),
            "params_by_dtype": dict(structure.params_by_dtype),
        },
        "structure": structure.to_dict(),
        "stats": {slot: None for slot in STATS_SLOTS},
        "fetch_log": log.to_card_dict(),
        "tensors_parquet": paths.parquet_path.name,
    }
    table = _build_table(source.repo, source.revision_sha, component, headers, structure)
    return BuiltCard(card=card, table=table, paths=paths)


# ---------------------------------------------------------------------- write / load


def _mkstemp_for(target: pathlib.Path) -> tuple[int, str]:
    return tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")


def write_card(built: BuiltCard) -> CardPaths:
    """Parquet first, then JSON; each via its own mkstemp temp file + os.replace."""
    paths = built.paths
    paths.json_path.parent.mkdir(parents=True, exist_ok=True)
    temps: list[str] = []
    try:
        fd, tmp = _mkstemp_for(paths.parquet_path)
        temps.append(tmp)
        with os.fdopen(fd, "wb") as f:
            pq.write_table(built.table, f, compression="zstd")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, paths.parquet_path)
        temps.remove(tmp)

        fd, tmp = _mkstemp_for(paths.json_path)
        temps.append(tmp)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(built.card, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, paths.json_path)
        temps.remove(tmp)
    except BaseException:
        for tmp in temps:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        raise
    return paths


def load_card(json_path: pathlib.Path) -> tuple[dict, pa.Table]:
    json_path = pathlib.Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        card = json.load(f)
    if not isinstance(card, dict) or card.get("schema_version") != SCHEMA_VERSION:
        got = card.get("schema_version") if isinstance(card, dict) else type(card).__name__
        raise ScoutError(f"{json_path}: unsupported Card schema_version {got!r} (want {SCHEMA_VERSION})")
    name = card.get("tensors_parquet")
    if not isinstance(name, str):
        raise ScoutError(f"{json_path}: tensors_parquet is not a string")
    _check_segment(name, "tensors_parquet")
    table = pq.read_table(json_path.parent / name)
    return card, table
