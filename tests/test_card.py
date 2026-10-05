import json
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scout
from scout import card as card_mod
from scout.card import (
    NULL_STAT_COLUMNS, PARQUET_SCHEMA, SCHEMA_VERSION, STATS_SLOTS, card_paths,
    component_slug, load_card, repo_slug, write_card,
)
from scout.errors import ScoutError
from tests.helpers.cards import build_fixture_card
from tests.helpers.st_fixtures import (
    write_dense_repo, write_moe_repo, write_pipeline_repo, write_sharded,
)

CARD_KEYS = [
    "schema_version", "key", "source", "scan", "model_card", "config", "pipeline",
    "weights", "structure", "stats", "fetch_log", "tensors_parquet",
]
SHA = "0123456789abcdef0123456789abcdef01234567"


def _files_under(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("**/*") if p.is_file())


@pytest.fixture
def dense(tmp_path):
    info = write_dense_repo(tmp_path / "repo", n_shards=2)
    built = build_fixture_card(tmp_path / "repo", tmp_path / "out")
    return info, built, tmp_path / "out"


def test_keys_exact(dense):
    info, built, _ = dense
    c = built.card
    assert set(c) == set(CARD_KEYS)
    assert list(c) == CARD_KEYS
    assert c["schema_version"] == SCHEMA_VERSION == "card.v0"
    assert list(c["key"]) == ["repo", "revision_sha", "component"]
    assert c["key"]["component"] is None
    assert list(c["source"]) == ["kind", "requested_revision", "revision_kind", "local_path", "endpoint"]
    assert list(c["scan"]) == ["scanned_at", "scout_version", "elapsed_s"]
    assert c["scan"]["scout_version"] == scout.__version__
    assert list(c["weights"]) == [
        "format", "index_path", "index_total_size", "files", "n_tensors", "params_total",
        "tensor_bytes_total", "params_by_dtype",
    ]
    for f in c["weights"]["files"]:
        assert list(f) == ["path", "size_bytes", "header_len", "data_bytes", "n_tensors", "metadata"]
    assert [f["path"] for f in c["weights"]["files"]] == sorted(info["files"])
    assert c["weights"]["n_tensors"] == info["n_tensors"]
    assert c["weights"]["params_total"] == info["params_total"]
    assert c["weights"]["tensor_bytes_total"] == info["total_size"]
    assert c["weights"]["index_total_size"] == info["total_size"]
    assert c["weights"]["index_path"] == "model.safetensors.index.json"
    assert set(c["structure"]) == {"stacks", "expert_groups", "warnings"}
    assert list(c["stats"]) == list(STATS_SLOTS)
    assert all(v is None for v in c["stats"].values())
    assert c["pipeline"] is None
    assert c["model_card"]["claimed"] is True
    assert c["tensors_parquet"] == "_model.tensors.parquet"
    json.dumps(c)  # no custom encoder needed


def test_parquet_schema(tmp_path):
    info = write_moe_repo(tmp_path / "repo", n_shards=2)
    t = build_fixture_card(tmp_path / "repo", tmp_path / "out").table
    assert t.schema.equals(PARQUET_SCHEMA, check_metadata=False)
    assert t.schema.names == [
        "repo", "revision_sha", "component", "name", "collapsed_name", "file", "dtype", "shape",
        "numel", "nbytes", "data_begin", "data_end", "stack_prefix", "block_index", "expert_index",
        "stat_mean", "stat_std", "stat_fro_norm", "stat_spectral_topk", "stat_sigma_curve",
    ]
    assert t.schema.field("block_index").type == pa.int32()
    assert t.schema.field("expert_index").type == pa.int32()
    assert t.num_rows == info["n_tensors"]
    for col in NULL_STAT_COLUMNS:
        assert t.column(col).null_count == t.num_rows
    md = t.schema.metadata
    assert md[b"schema_version"] == b"card.v0"
    assert set(md) == {b"schema_version", b"repo", b"revision_sha", b"component"}
    assert md[b"component"] == b""
    rows = t.to_pylist()
    assert [(r["file"], r["data_begin"]) for r in rows] == sorted((r["file"], r["data_begin"]) for r in rows)
    assert all(r["component"] == "" for r in rows)
    assert all(r["nbytes"] == r["data_end"] - r["data_begin"] for r in rows)
    expert_rows = [r for r in rows if r["expert_index"] is not None]
    assert expert_rows and all("*" in r["collapsed_name"] for r in expert_rows)
    assert sum(r["numel"] for r in rows) == info["params_total"]


def test_paths(tmp_path):
    out = tmp_path / "out"
    p = card_paths(out, "Org/M", SHA, None)
    assert p.json_path == out / "Org__M" / SHA / "_model.card.json"
    assert p.parquet_path == out / "Org__M" / SHA / "_model.tensors.parquet"
    assert card_paths(out, "Org/M", SHA, "vae").json_path == out / "Org__M" / SHA / "vae.card.json"
    assert component_slug(None) == "_model" and component_slug("vae") == "vae"
    slug = repo_slug("local:/home/me/my models/x")
    assert "/" not in slug and slug == "local__home_me_my_models_x"
    assert repo_slug("Org/M") == "Org__M"
    with pytest.raises(ScoutError):
        card_paths(out, "Org/M", SHA, "../evil")


def test_pipeline_component(tmp_path):
    info = write_pipeline_repo(tmp_path / "repo")
    built = build_fixture_card(tmp_path / "repo", tmp_path / "out", component="vae")
    assert built.paths.json_path.name == "vae.card.json"
    assert built.card["key"]["component"] == "vae"
    assert built.card["pipeline"]["pipeline_class"] == "TestPipeline"
    assert built.table.schema.metadata[b"component"] == b"vae"
    assert set(built.table.column("component").to_pylist()) == {"vae"}
    assert built.table.num_rows == info["components"]["vae"]["n_tensors"]
    assert built.card["weights"]["index_path"] is None


def test_roundtrip(dense, tmp_path):
    _, built, _ = dense
    paths = write_card(built)
    assert paths == built.paths
    card, table = load_card(paths.json_path)
    assert card == built.card
    assert table.equals(built.table, check_metadata=True)


def test_atomic(dense, monkeypatch):
    _, built, out = dense

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pq, "write_table", boom)
    with pytest.raises(RuntimeError):
        write_card(built)
    assert _files_under(out) == []


def test_atomic_json_failure_leaves_no_temp(dense, monkeypatch):
    _, built, out = dense
    real_replace = card_mod.os.replace

    def replace(src, dst):
        if str(dst).endswith(".card.json"):
            raise OSError("json replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr(card_mod.os, "replace", replace)
    with pytest.raises(OSError):
        write_card(built)
    assert not list(out.glob("**/*.tmp"))
    assert not list(out.glob("**/*.card.json"))
    assert not list(out.glob("**/*.parquet"))
    assert _files_under(out) == []


def test_failed_rewrite_keeps_existing_card(dense, monkeypatch):
    _, built, out = dense
    write_card(built)
    real_replace = card_mod.os.replace

    def replace(src, dst):
        if str(dst).endswith(".card.json"):
            raise OSError("json replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr(card_mod.os, "replace", replace)
    with pytest.raises(OSError):
        write_card(built)
    monkeypatch.setattr(card_mod.os, "replace", real_replace)
    card, table = load_card(built.paths.json_path)
    assert card == built.card
    assert table.equals(built.table, check_metadata=True)
    assert not list(out.glob("**/*.tmp"))
    assert _files_under(out) == sorted([built.paths.json_path, built.paths.parquet_path])


def test_lone_surrogate_in_config_refused(tmp_path):
    write_dense_repo(tmp_path / "repo", n_shards=1)
    (tmp_path / "repo" / "config.json").write_text('{"model_type": "x", "bad": "\\ud800"}')
    out = tmp_path / "out"
    with pytest.raises(ScoutError, match="unencodable"):
        build_fixture_card(tmp_path / "repo", out)
    assert not out.exists() or _files_under(out) == []


def test_lone_surrogate_in_tensor_name_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.json").write_text("{}")
    write_sharded(repo, {"model.\ud800.weight": ("F32", (2,)), "lm_head.weight": ("F32", (2,))}, 1)
    out = tmp_path / "out"
    with pytest.raises(ScoutError):
        build_fixture_card(repo, out)
    assert not out.exists() or _files_under(out) == []


def test_fetch_log_embedded(dense):
    _, built, _ = dense
    fl = built.card["fetch_log"]
    assert fl["counting"] == "http-body-bytes-yielded"
    assert fl["totals"]["weight"] == 0
    assert fl["events"]
    assert all(e["bytes_by_class"]["weight"] == 0 for e in fl["events"])
    assert fl["events"][-1]["event"] == "stage_start" and fl["events"][-1]["stage"] == "REPORT"


def test_concurrent_write(dense):
    _, built, out = dense
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            write_card(built)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    card, table = load_card(built.paths.json_path)
    assert card == built.card
    assert table.equals(built.table, check_metadata=True)
    assert not list(out.glob("**/*.tmp"))
    assert len(_files_under(out)) == 2


def test_only_card_files(dense):
    _, built, out = dense
    write_card(built)
    files = _files_under(out)
    assert files == sorted([built.paths.json_path, built.paths.parquet_path])


def test_load_rejects_wrong_version(dense):
    _, built, _ = dense
    write_card(built)
    data = json.loads(built.paths.json_path.read_text("utf-8"))
    data["schema_version"] = "card.v1"
    built.paths.json_path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ScoutError):
        load_card(built.paths.json_path)
