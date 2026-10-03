"""scan.py: intake orchestration over FakeHub (hub) and tmp dirs (local); all offline."""
from __future__ import annotations

import json
import os
import re
import tempfile
from types import SimpleNamespace

import pytest

import scout.scan
from scout.bytelog import ByteLog, Stage
from scout.card import load_card
from scout.errors import (
    GatedRepoError,
    NetworkError,
    NoSafetensorsError,
    ReadThresholdExceeded,
    ScoutError,
)
from scout.scan import ScanResult, make_source, parse_target, scan
from tests.helpers.fakehub import HUB, FakeHub
from tests.helpers.st_fixtures import (
    write_dense_repo,
    write_moe_repo,
    write_pipeline_repo,
)

REPO = "org/m"
SHA = "b" * 40
_RNG = re.compile(r"^bytes=(\d+)-(\d+)$")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def _hub_scan(hub, out_dir, log=None, target=REPO, **kw):
    log = log if log is not None else ByteLog()
    client = hub.client()
    try:
        res = scan(target, out_dir, log, client=client, endpoint=HUB, **kw)
    finally:
        client.close()
    return res, log


def _hub_with(tmp_path, writer, **writer_kw):
    root = tmp_path / "repo"
    info = writer(root, **writer_kw)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA)
    return SimpleNamespace(hub=hub, root=root, info=info, out=tmp_path / "out")


def _all_files(d):
    if not d.exists():
        return []
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*"))


def _stage_order(log):
    return [e.stage for e in log.events if e.event == "stage_start"]


# ---------------------------------------------------------------- acceptance


def test_scan_dense_sharded_hub(tmp_path):
    env = _hub_with(tmp_path, write_dense_repo, n_shards=3)
    res, log = _hub_scan(env.hub, env.out)
    assert isinstance(res, ScanResult)
    assert len(res.cards) == 1
    assert res.repo == REPO and res.revision_sha == SHA
    card, table = load_card(res.cards[0].json_path)
    assert card["key"] == {"repo": REPO, "revision_sha": SHA, "component": None}
    assert card["weights"]["n_tensors"] == env.info["n_tensors"]
    assert card["weights"]["params_total"] == env.info["params_total"]
    assert card["weights"]["index_path"] == "model.safetensors.index.json"
    assert card["weights"]["index_total_size"] == env.info["total_size"]
    assert table.num_rows == env.info["n_tensors"]
    assert res.totals["weight"] == 0
    hl = env.info["header_lens"]
    assert len(hl) == 3
    assert res.totals["header"] == sum(8 + n for n in hl.values())
    assert env.hub.cdn_reads
    for path, rng, _ in env.hub.cdn_reads:
        m = _RNG.match(rng)
        assert m, rng
        assert int(m.group(2)) + 1 <= 8 + hl[path]
    assert _stage_order(log) == ["RESOLVE", "HEADERS", "META", "REPORT"]
    written = [e for e in log.events if e.event == "card_written"]
    assert [e.note for e in written] == [str(res.cards[0].json_path)]
    assert res.elapsed_s >= card["scan"]["elapsed_s"] >= 0
    assert card["scan"]["scanned_at"].endswith("Z")
    # The Card's fetch_log stops at the REPORT stage_start (no card_written inside).
    assert card["fetch_log"]["events"][-1]["event"] == "stage_start"
    assert card["fetch_log"]["events"][-1]["stage"] == "REPORT"


def test_scan_local_folder(tmp_path):
    root = tmp_path / "repo"
    info = write_dense_repo(root, n_shards=3)
    out = tmp_path / "out"
    log = ByteLog()
    res = scan(str(root), out, log)
    assert len(res.cards) == 1
    card, _ = load_card(res.cards[0].json_path)
    assert card["source"]["revision_kind"] == "local-stat-hash"
    assert card["key"]["repo"].startswith("local:")
    assert res.repo.startswith("local:")
    assert card["weights"]["n_tensors"] == info["n_tensors"]
    assert res.totals["weight"] == 0
    assert _stage_order(log) == ["RESOLVE", "HEADERS", "META", "REPORT"]


def test_scan_moe(tmp_path):
    env = _hub_with(tmp_path, write_moe_repo)
    res, _ = _hub_scan(env.hub, env.out)
    card, _ = load_card(res.cards[0].json_path)
    s = card["structure"]
    assert s["expert_groups"][0]["n_experts"] == 4
    assert all(s["stacks"][0]["block_moe"])
    assert res.totals["weight"] == 0


def test_scan_pipeline(tmp_path):
    env = _hub_with(tmp_path, write_pipeline_repo)
    res, _ = _hub_scan(env.hub, env.out)
    cards = {}
    for cp in res.cards:
        c, _ = load_card(cp.json_path)
        cards[c["key"]["component"]] = c
    assert sorted(cards) == ["text_encoder", "transformer", "vae"]
    assert len(res.cards) == 3
    comps = {c["name"]: c for c in cards["vae"]["pipeline"]["components"]}
    assert comps["scheduler"]["has_weights"] is False
    assert comps["tokenizer"]["has_weights"] is False
    assert comps["text_encoder"]["has_weights"] is True
    te = cards["text_encoder"]["weights"]
    assert te["index_path"] == "text_encoder/model.safetensors.index.json"
    assert te["index_total_size"] == te["tensor_bytes_total"]
    stacks = {s["prefix"]: s for s in cards["transformer"]["structure"]["stacks"]}
    assert stacks["transformer_blocks"]["depth"] == 3
    assert len({c["key"]["revision_sha"] for c in cards.values()}) == 1
    mc = [c["model_card"] for c in cards.values()]
    assert all(m == mc[0] for m in mc) and mc[0]["present"] is True
    assert cards["transformer"]["config"]["path"] == "transformer/config.json"
    assert res.totals["weight"] == 0
    # README is read exactly once.
    assert sum(1 for p, _, _ in env.hub.cdn_reads if p == "README.md") == 0
    readme_reqs = [r for r in env.hub.requests if r.url.path.endswith("/README.md")]
    assert len(readme_reqs) == 1


def test_scan_no_readme(tmp_path):
    env = _hub_with(tmp_path, write_dense_repo, readme=None)
    res, _ = _hub_scan(env.hub, env.out)
    card, _ = load_card(res.cards[0].json_path)
    assert card["model_card"]["present"] is False


def test_scan_gated(tmp_path):
    root = tmp_path / "repo"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA, gated=True, token="tok")
    out = tmp_path / "out"
    with pytest.raises(GatedRepoError):
        _hub_scan(hub, out)
    assert _all_files(out) == []
    res, _ = _hub_scan(hub, out, token="tok")
    assert len(res.cards) == 1


def test_scan_network_retry(tmp_path):
    env = _hub_with(tmp_path, write_dense_repo, n_shards=3)
    shard = env.info["files"][1]
    env.hub.inject(shard, "reset", times=1)
    res, log = _hub_scan(env.hub, env.out)
    assert len(res.cards) == 1
    assert any(e.event == "retry" and e.path == shard for e in log.events)
    assert res.totals["weight"] == 0


def test_scan_network_fail(tmp_path):
    env = _hub_with(tmp_path, write_dense_repo, n_shards=3)
    shard = env.info["files"][1]
    env.hub.inject(shard, "reset", times=10)
    log = ByteLog()
    with pytest.raises(NetworkError):
        _hub_scan(env.hub, env.out, log=log)
    assert _all_files(env.out) == []
    assert any(e.event == "error" for e in log.events)
    assert log.totals["weight"] == 0


def test_scan_bin_only(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "x"}))
    (root / "pytorch_model.bin").write_bytes(b"\x80" * 64)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA)
    log = ByteLog()
    out = tmp_path / "out"
    with pytest.raises(NoSafetensorsError):
        _hub_scan(hub, out, log=log)
    assert log.totals["weight"] == 0
    assert _all_files(out) == []
    # Local folder too.
    log2 = ByteLog()
    with pytest.raises(NoSafetensorsError):
        scan(str(root), out, log2)
    assert log2.totals["weight"] == 0


def test_scan_threshold(tmp_path):
    env = _hub_with(tmp_path, write_dense_repo)
    log = ByteLog(threshold_bytes=200)
    with pytest.raises(ReadThresholdExceeded):
        _hub_scan(env.hub, env.out, log=log)
    assert _all_files(env.out) == []
    assert any(e.event == "error" for e in log.events)


def test_no_extra_files(tmp_path, monkeypatch):
    home, tmpd = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmpd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmpd))
    monkeypatch.setattr(tempfile, "tempdir", None)  # make tempfile re-read TMPDIR
    env = _hub_with(tmp_path, write_pipeline_repo)
    before = _all_files(env.root)
    _hub_scan(env.hub, env.out)
    local_root = tmp_path / "dense"
    write_dense_repo(local_root)
    local_before = _all_files(local_root)
    scan(str(local_root), env.out, ByteLog())
    assert _all_files(home) == []
    assert _all_files(tmpd) == []
    assert _all_files(env.root) == before
    assert _all_files(local_root) == local_before
    for p in env.out.rglob("*"):
        if p.is_file():
            assert p.name.endswith(".card.json") or p.name.endswith(".tensors.parquet"), p


@pytest.mark.parametrize("target,expected", [
    ("Org/M@main", ("hub", "Org/M", "main")),
    ("Org/M", ("hub", "Org/M", None)),
    ("Org/M@", ("hub", "Org/M", None)),
    ("Org/M@" + "c" * 40, ("hub", "Org/M", "c" * 40)),
    ("Org/M@refs/pr/1", ("hub", "Org/M", "refs/pr/1")),
])
def test_parse_target(target, expected):
    assert parse_target(target) == expected


def test_parse_target_dir(tmp_path):
    d = tmp_path / "some dir"
    d.mkdir()
    assert parse_target(str(d)) == ("local", str(d.resolve()), None)


@pytest.mark.parametrize("bad", ["bad", "", "a/b/c", "/nonexistent/dir/x", "@main", "../x@y"])
def test_parse_target_bad(bad):
    with pytest.raises(ValueError, match="not a directory or owner/name"):
        parse_target(bad)


def test_make_source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    log = ByteLog()
    s = make_source(str(root), log)
    assert s.kind == "local"
    hub = FakeHub()
    with hub.client() as c:
        h = make_source("Org/M@main", log, client=c, endpoint=HUB)
        assert (h.kind, h.repo, h.requested_revision, h.endpoint) == ("hub", "Org/M", "main", HUB)


# ---------------------------------------------------------------- failure cleanup


def test_write_failure_removes_cards_of_this_scan(tmp_path, monkeypatch):
    env = _hub_with(tmp_path, write_pipeline_repo)
    real = scout.scan.write_card
    calls = []

    def flaky(built):
        calls.append(built.paths)
        if len(calls) == 3:
            raise OSError("disk full")
        return real(built)

    monkeypatch.setattr(scout.scan, "write_card", flaky)
    log = ByteLog()
    with pytest.raises(OSError, match="disk full"):
        _hub_scan(env.hub, env.out, log=log)
    assert len(calls) == 3
    assert _all_files(env.out) == []  # files and the dirs this scan created are gone
    assert any(e.event == "error" for e in log.events)


def test_write_failure_keeps_preexisting_out_dir(tmp_path, monkeypatch):
    env = _hub_with(tmp_path, write_pipeline_repo)
    env.out.mkdir()
    (env.out / "keep.txt").write_text("x")
    real = scout.scan.write_card
    n = {"i": 0}

    def flaky(built):
        n["i"] += 1
        if n["i"] == 2:
            raise OSError("boom")
        return real(built)

    monkeypatch.setattr(scout.scan, "write_card", flaky)
    with pytest.raises(OSError):
        _hub_scan(env.hub, env.out)
    assert _all_files(env.out) == ["keep.txt"]


def test_rescan_failure_keeps_preexisting_cards(tmp_path, monkeypatch):
    env = _hub_with(tmp_path, write_pipeline_repo)
    first, _ = _hub_scan(env.hub, env.out)
    before = _all_files(env.out)
    assert len(first.cards) == 3

    real = scout.scan.write_card
    n = {"i": 0}

    def flaky(built):
        n["i"] += 1
        if n["i"] == 3:  # fail on the last component, after two cards were rewritten
            raise OSError("boom")
        return real(built)

    monkeypatch.setattr(scout.scan, "write_card", flaky)
    with pytest.raises(OSError):
        _hub_scan(env.hub, env.out)
    assert _all_files(env.out) == before
    for cp in first.cards:
        card, table = load_card(cp.json_path)
        assert card["key"]["revision_sha"] == SHA
        assert table.num_rows == card["weights"]["n_tensors"]


def test_no_card_written_when_later_component_fails(tmp_path, monkeypatch):
    env = _hub_with(tmp_path, write_pipeline_repo)
    # Corrupt the vae header (last unit) so HEADERS fails after other units' reads.
    vae = env.root / "vae" / "diffusion_pytorch_model.safetensors"
    data = bytearray(vae.read_bytes())
    data[8] = ord("[")
    vae.write_bytes(bytes(data))
    calls = []
    monkeypatch.setattr(scout.scan, "write_card", lambda b: calls.append(b))
    with pytest.raises(ScoutError):
        _hub_scan(env.hub, env.out)
    assert calls == []
    assert _all_files(env.out) == []


def test_hostile_component_name_refused(tmp_path):
    root = tmp_path / "repo"
    write_pipeline_repo(root)
    mi = json.loads((root / "model_index.json").read_text())
    mi[".."] = ["diffusers", "Evil"]
    (root / "model_index.json").write_text(json.dumps(mi))
    out = tmp_path / "out"
    log = ByteLog()
    with pytest.raises(ScoutError, match="unsafe component"):
        scan(str(root), out, log)
    assert _all_files(out) == []
    assert sorted(os.listdir(tmp_path)) == ["repo"]
    # Refused in RESOLVE, before any header read.
    assert _stage_order(log) == ["RESOLVE"]


def test_bad_model_index_is_scout_error(tmp_path):
    root = tmp_path / "repo"
    write_pipeline_repo(root)
    (root / "model_index.json").write_text("[1, 2]")
    with pytest.raises(ScoutError):
        scan(str(root), tmp_path / "out", ByteLog())


def test_pipeline_non_safetensors_warning(tmp_path):
    root = tmp_path / "repo"
    write_pipeline_repo(root)
    mi = json.loads((root / "model_index.json").read_text())
    mi["safety_checker"] = ["transformers", "Checker"]
    (root / "model_index.json").write_text(json.dumps(mi))
    (root / "safety_checker").mkdir()
    (root / "safety_checker" / "pytorch_model.bin").write_bytes(b"\x00" * 16)
    res = scan(str(root), tmp_path / "out", ByteLog())
    assert len(res.cards) == 3
    for cp in res.cards:
        card, _ = load_card(cp.json_path)
        assert "safety_checker: non-safetensors weights skipped" in card["structure"]["warnings"]
        comps = {c["name"]: c for c in card["pipeline"]["components"]}
        assert comps["safety_checker"]["has_weights"] is False
    assert res.totals["weight"] == 0


def test_pipeline_without_weights(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"_class_name": "P", "scheduler": ["d", "S"]}))
    with pytest.raises(NoSafetensorsError, match="pipeline has no safetensors components"):
        scan(str(root), tmp_path / "out", ByteLog())


def test_index_missing_shard(tmp_path):
    root = tmp_path / "repo"
    info = write_dense_repo(root, n_shards=3)
    (root / info["files"][2]).unlink()
    out = tmp_path / "out"
    with pytest.raises(ScoutError, match="index lists missing file"):
        scan(str(root), out, ByteLog())
    assert _all_files(out) == []


def test_index_total_size_mismatch_warns(tmp_path):
    root = tmp_path / "repo"
    write_dense_repo(root, n_shards=2)
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    idx["metadata"]["total_size"] += 2
    (root / "model.safetensors.index.json").write_text(json.dumps(idx))
    res = scan(str(root), tmp_path / "out", ByteLog())
    card, _ = load_card(res.cards[0].json_path)
    s = card["weights"]["tensor_bytes_total"]
    assert f"index total_size {s + 2} != tensor bytes {s}" in card["structure"]["warnings"]


def test_scan_closes_owned_client(tmp_path, monkeypatch):
    env = _hub_with(tmp_path, write_dense_repo)
    closed = []
    orig = scout.scan.HubSource.close
    monkeypatch.setattr(scout.scan.HubSource, "close", lambda self: (closed.append(self), orig(self)))
    _hub_scan(env.hub, env.out)
    assert len(closed) == 1
