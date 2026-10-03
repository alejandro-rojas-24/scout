"""Build a Card v0 from a synthetic local fixture repo without scan.py (reused by T011)."""
from __future__ import annotations

from pathlib import Path

from scout.bytelog import ByteLog, Stage
from scout.card import BuiltCard, build_card
from scout.errors import NoSafetensorsError
from scout.metadata import PipelineInfo, parse_config, parse_model_card, parse_model_index
from scout.safetensors_header import parse_index, read_header, select_weight_files
from scout.sources import LocalSource
from scout.structure import analyze

FIXTURE_SCANNED_AT = "2026-01-01T00:00:00Z"


def build_fixture_card(root: Path, out_dir: Path, component: str | None = None) -> BuiltCard:
    """Header-only build of one Card for `root` (or one pipeline `component` of it)."""
    log = ByteLog()
    log.stage(Stage.RESOLVE)
    source = LocalSource(root, log)
    source.resolve()
    files = source.files()
    sizes = {f.path: f.size for f in files}
    subdir = component or ""

    pipeline: PipelineInfo | None = None
    if "model_index.json" in sizes:
        pipeline = parse_model_index(source.read_file("model_index.json"))
        for comp in pipeline.components:
            try:
                select_weight_files(files, comp.name)
                comp.has_weights = True
            except NoSafetensorsError:
                comp.has_weights = False

    sel = select_weight_files(files, subdir)
    index_path, index_total = sel.index_path, None
    if index_path is not None:
        shard_paths, index_total = parse_index(source.read_file(index_path), subdir)
    else:
        shard_paths = [sel.single_path]

    log.stage(Stage.HEADERS)
    headers = [read_header(source, p, sizes.get(p)) for p in shard_paths]

    log.stage(Stage.META)
    model_card = parse_model_card(source.read_file("README.md"))
    config_path = f"{subdir}/config.json" if subdir else "config.json"
    config = parse_config(source.read_file(config_path), config_path)

    log.stage(Stage.REPORT)
    structure = analyze(t for h in headers for t in h.tensors)
    return build_card(
        source=source, component=component, pipeline=pipeline, config=config,
        model_card=model_card, headers=headers,
        file_sizes={p: sizes.get(p) for p in shard_paths},
        index_path=index_path, index_total_size=index_total, structure=structure, log=log,
        scanned_at=FIXTURE_SCANNED_AT, elapsed_s=0.0, out_dir=Path(out_dir),
    )
