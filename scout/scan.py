"""Scan orchestration: target -> Source -> headers + metadata -> Card(s), with a staged byte log.

Stages: RESOLVE (API/listing, model_index.json, *.safetensors.index.json) > HEADERS (all
safetensors headers, concurrently) > META (README.md, each unit's config.json) > REPORT
(build every Card in memory, then write them). No Card is written until every unit has
succeeded; if a write fails, the Card files and directories this scan newly created are
removed again. Weights are never read.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from scout.bytelog import WEIGHT_EXTENSIONS, ByteLog, Stage
from scout.card import BuiltCard, CardPaths, build_card, card_paths, write_card
from scout.errors import NoSafetensorsError, ScoutError
from scout.hub import REPO_RE, HubSource
from scout.metadata import PipelineInfo, parse_config, parse_model_card, parse_model_index
from scout.safetensors_header import (
    SafetensorsHeader,
    WeightSelection,
    parse_index,
    read_header,
    select_weight_files,
)
from scout.sources import LocalSource, RepoFile, Source
from scout.structure import analyze


@dataclass(frozen=True)
class ScanResult:
    cards: list[CardPaths]
    totals: dict[str, int]
    elapsed_s: float
    revision_sha: str
    repo: str


@dataclass
class _Unit:
    component: str | None
    subdir: str
    index_path: str | None
    shard_files: list[str]
    total: int | None


# ---------------------------------------------------------------------- targets


def parse_target(target: str) -> tuple[str, str, str | None]:
    """("local", abs_path, None) for an existing directory, else ("hub", repo_id, revision_or_None)."""
    if os.path.isdir(target):
        return ("local", str(pathlib.Path(target).resolve()), None)
    repo, sep, rev = target.rpartition("@")
    if not sep:
        repo, rev = target, ""
    if not REPO_RE.match(repo):
        raise ValueError(f"not a directory or owner/name[@rev]: {target}")
    return ("hub", repo, rev or None)


def make_source(target: str, log: ByteLog, *, client: httpx.Client | None = None,
                endpoint: str | None = None, token: str | None = None) -> Source:
    kind, ident, rev = parse_target(target)
    if kind == "local":
        return LocalSource(ident, log)
    return HubSource(ident, rev, log, client=client, endpoint=endpoint, token=token)


# ---------------------------------------------------------------------- helpers


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _has_weight_files(files: list[RepoFile], subdir: str) -> bool:
    for f in files:
        parent, _, base = f.path.rpartition("/")
        if parent == subdir and base.lower().endswith(WEIGHT_EXTENSIONS):
            return True
    return False


def _plan_units(source: Source, files: list[RepoFile], sizes: dict[str, int | None],
                out_dir: pathlib.Path) -> tuple[PipelineInfo | None, list[str], list[tuple[str | None, WeightSelection]]]:
    if "model_index.json" not in sizes:
        return None, [], [(None, select_weight_files(files, ""))]
    try:
        pipeline = parse_model_index(source.read_file("model_index.json"))
    except ValueError as exc:
        raise ScoutError(str(exc)) from exc
    # Component names are hostile input: refuse any that cannot be a single path segment
    # (card_paths' own check) before they are used for anything, including repo lookups.
    for comp in pipeline.components:
        card_paths(out_dir, source.repo, source.revision_sha, comp.name)
    warnings: list[str] = []
    selections: list[tuple[str | None, WeightSelection]] = []
    for comp in pipeline.components:
        try:
            sel = select_weight_files(files, comp.name)
        except NoSafetensorsError:
            comp.has_weights = False
            if _has_weight_files(files, comp.name):
                warnings.append(f"{comp.name}: non-safetensors weights skipped")
            continue
        comp.has_weights = True
        selections.append((comp.name, sel))
    if not selections:
        raise NoSafetensorsError("pipeline has no safetensors components")
    return pipeline, warnings, selections


def _resolve_unit(source: Source, sizes: dict[str, int | None], component: str | None,
                  sel: WeightSelection) -> _Unit:
    subdir = component or ""
    if sel.index_path is None:
        return _Unit(component, subdir, None, [sel.single_path], None)
    raw = source.read_file(sel.index_path)
    if raw is None:  # cannot happen for a listed file; keep the failure explicit
        raise ScoutError(f"index {sel.index_path} could not be read")
    shard_files, total = parse_index(raw, subdir)
    for f in shard_files:
        if f not in sizes:
            raise ScoutError(f"index lists missing file {f}")
    return _Unit(component, subdir, sel.index_path, shard_files, total)


def _read_headers(source: Source, paths: list[str], sizes: dict[str, int | None],
                  max_workers: int) -> dict[str, SafetensorsHeader]:
    out: dict[str, SafetensorsHeader] = {}
    if not paths:
        return out
    ex = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="scout-header")
    try:
        futs = {ex.submit(read_header, source, p, sizes[p]): p for p in paths}
        for fut in as_completed(futs):
            exc = fut.exception()
            if exc is not None:
                for f in futs:
                    f.cancel()  # pending reads never start
                raise exc
            out[futs[fut]] = fut.result()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)  # running reads finish before we leave
    return out


def _missing_dirs(target_dir: pathlib.Path) -> list[pathlib.Path]:
    """Ancestors of (and including) target_dir that do not exist yet, deepest first.

    These are exactly the directories write_card's mkdir(parents=True) would create.
    """
    missing: list[pathlib.Path] = []
    for d in [target_dir, *target_dir.parents]:
        if d.exists():
            break
        missing.append(d)
    return missing


def _write_all(built: list[BuiltCard], log: ByteLog) -> list[CardPaths]:
    """Write every Card; on failure remove what this scan newly created, then re-raise.

    A Card whose JSON already existed before this scan (a rescan of the same key) is never
    removed: os.replace has swapped in a complete new Card, or the old one is still in place.
    """
    written: list[CardPaths] = []
    new_files: list[pathlib.Path] = []
    new_dirs: list[pathlib.Path] = []
    try:
        for b in built:
            existed = b.paths.json_path.exists()
            for d in _missing_dirs(b.paths.json_path.parent):
                if d not in new_dirs:
                    new_dirs.append(d)
            if not existed:
                # Registered before the write so a partial write is also cleaned up.
                new_files.extend([b.paths.json_path, b.paths.parquet_path])
            paths = write_card(b)
            written.append(paths)
            log.note("card_written", str(paths.json_path))
    except BaseException:
        for p in new_files:
            try:
                os.unlink(p)
            except OSError:
                pass
        # Deepest first; only directories this scan created, and only while empty.
        for d in sorted(new_dirs, key=lambda d: len(d.parts), reverse=True):
            try:
                os.rmdir(d)
            except OSError:
                pass
        raise
    return written


# ---------------------------------------------------------------------- scan


def scan(target: str, out_dir: pathlib.Path, log: ByteLog, *, client: httpx.Client | None = None,
         endpoint: str | None = None, token: str | None = None, max_workers: int = 8) -> ScanResult:
    t0 = time.monotonic()
    scanned_at = _utc_now_iso()
    out_dir = pathlib.Path(out_dir)
    source: Source | None = None
    try:
        try:
            # ---- RESOLVE
            log.stage(Stage.RESOLVE)
            source = make_source(target, log, client=client, endpoint=endpoint, token=token)
            source.resolve()
            files = source.files()
            sizes: dict[str, int | None] = {f.path: f.size for f in files}
            pipeline, pipe_warnings, selections = _plan_units(source, files, sizes, out_dir)
            units = [_resolve_unit(source, sizes, comp, sel) for comp, sel in selections]

            # ---- HEADERS
            log.stage(Stage.HEADERS)
            all_paths = list(dict.fromkeys(p for u in units for p in u.shard_files))
            headers = _read_headers(source, all_paths, sizes, max_workers)

            # ---- META
            log.stage(Stage.META)
            model_card = parse_model_card(source.read_file("README.md"))
            configs = []
            for u in units:
                cfg_path = "config.json" if u.subdir == "" else f"{u.subdir}/config.json"
                configs.append(parse_config(source.read_file(cfg_path), cfg_path))

            # ---- structure (pure, no IO)
            structures = []
            for u in units:
                unit_headers = [headers[p] for p in u.shard_files]
                st = analyze(t for h in unit_headers for t in h.tensors)
                warnings = list(st.warnings) + list(pipe_warnings)
                if u.total is not None:
                    s = sum(t.nbytes for h in unit_headers for t in h.tensors)
                    if s != u.total:
                        warnings.append(f"index total_size {u.total} != tensor bytes {s}")
                structures.append(dataclasses.replace(st, warnings=warnings))

            # ---- REPORT: build every Card in memory before writing any
            log.stage(Stage.REPORT)
            built: list[BuiltCard] = []
            for u, cfg, st in zip(units, configs, structures):
                built.append(build_card(
                    source=source, component=u.component, pipeline=pipeline, config=cfg,
                    model_card=model_card, headers=[headers[p] for p in u.shard_files],
                    file_sizes={p: sizes.get(p) for p in u.shard_files},
                    index_path=u.index_path, index_total_size=u.total, structure=st, log=log,
                    scanned_at=scanned_at, elapsed_s=time.monotonic() - t0, out_dir=out_dir,
                ))
            cards = _write_all(built, log)
        except BaseException as exc:
            try:
                log.note("error", f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            raise
        return ScanResult(cards=cards, totals=log.totals, elapsed_s=time.monotonic() - t0,
                          revision_sha=source.revision_sha, repo=source.repo)
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()  # HubSource closes only a client it created itself
