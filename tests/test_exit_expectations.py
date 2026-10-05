"""Offline tests for scripts/exit_expectations.py: hand-built TargetResults, pass and fail inputs."""

from __future__ import annotations

import copy
import pathlib

import pyarrow as pa
import pytest

from scripts import exit_expectations as ee
from scripts.exit_expectations import (
    DEFAULT_EXPECTATIONS, Check, TargetResult, common_checks, expect_qwen3_8b, expect_qwen3_30b_a3b,
    expect_qwen_image, host_allowed,
)

SHA = "a" * 40
STAT_COLS = ["stat_mean", "stat_std", "stat_fro_norm", "stat_spectral_topk", "stat_sigma_curve"]


# ---------------------------------------------------------------------------
# builders


def _stats() -> dict:
    return {"tensor_stats": None, "spectral_topk": None, "sigma_curves": None,
            "tokenizer_minhash": None, "attribution": None}


def _card(component, *, config=None, model_type=None, archs=None, class_name=None, files=None, n_tensors=2,
          params_total=10, stacks=None, groups=None, index_path=None, index_total_size=None,
          tensor_bytes_total=None, license_="apache-2.0", base_model=None, pipeline=None, sha=SHA, repo="o/n"):
    return {
        "schema_version": "card.v0",
        "key": {"repo": repo, "revision_sha": sha, "component": component},
        "source": {"kind": "hub", "requested_revision": sha, "revision_kind": "git", "local_path": None,
                   "endpoint": "https://huggingface.co"},
        "model_card": {"present": True, "claimed": True, "base_model": base_model or [], "license": license_,
                       "tags": []},
        "config": {"present": True, "path": "config.json", "model_type": model_type,
                   "architectures": archs or [], "class_name": class_name, "raw": config, "parse_error": None},
        "pipeline": pipeline,
        "weights": {"format": "safetensors", "index_path": index_path, "index_total_size": index_total_size,
                    "files": files if files is not None else [], "n_tensors": n_tensors,
                    "params_total": params_total,
                    "tensor_bytes_total": tensor_bytes_total, "params_by_dtype": {}},
        "structure": {"stacks": stacks or [], "expert_groups": groups or [], "warnings": []},
        "stats": _stats(),
        "fetch_log": {"counting": "http-body-bytes-yielded", "threshold_bytes": 1 << 26,
                      "totals": {"meta": 100, "header": 0, "weight": 0}, "events": []},
    }


def _stack(prefix, depth, sigs=None, moe=False, n_experts=None):
    return {"prefix": prefix, "depth": depth, "indices": list(range(depth)), "block_params": [1] * depth,
            "block_signatures": sigs if sigs is not None else ["s"] * depth,
            "block_moe": [moe] * depth, "block_n_experts": [n_experts] * depth}


def _view(component, nodes=None, strips=None):
    return {"card_key": {"repo": "o/n", "revision_sha": SHA, "component": component},
            "nodes": nodes or [], "depth_strips": strips or []}


def _res(cards, views=None, target="t"):
    return TargetResult(target=target, pin=SHA, status="done", elapsed_s=1.0, final={}, events=[], cards=cards,
                        tables=[], views=views if views is not None else [], json_paths=[], parquet_paths=[],
                        wire=[])


def _qwen3_8b():
    cfgraw = {"num_hidden_layers": 36}
    card = _card(None, config=cfgraw, model_type="qwen3", archs=["Qwen3ForCausalLM"], n_tensors=399,
                 params_total=8190735360, stacks=[_stack("model.layers", 36)],
                 base_model=["Qwen/Qwen3-8B-Base"], repo="Qwen/Qwen3-8B")
    card["weights"]["params_by_dtype"] = {"BF16": 8190735360}
    return _res([card])


def _qwen3_30b():
    cfgraw = {"num_hidden_layers": 48, "num_experts": 128, "hidden_size": 2048, "moe_intermediate_size": 768}
    group = {"template": "model.layers.#.mlp.experts.*", "n_experts": 128, "n_instances": 48,
             "params_per_expert": 4718592, "tensors_per_expert": 3, "homogeneous": True}
    card = _card(None, config=cfgraw, model_type="qwen3_moe", n_tensors=18867, params_total=30532122624,
                 stacks=[_stack("model.layers", 48, moe=True, n_experts=128)], groups=[group],
                 base_model=["Qwen/Qwen3-30B-A3B-Base"], repo="Qwen/Qwen3-30B-A3B")
    strip = {"prefix": "model.layers", "depth": 48, "cells": [
        {"index": i, "params": 1, "signature": "s", "moe": True, "n_experts": 128} for i in range(48)]}
    view = _view(None, nodes=[{"id": "x", "kind": "expert_group", "count": 128}, {"id": "y", "kind": "stack",
                                                                                 "count": 48}], strips=[strip])
    return _res([card], [view])


def _image():
    pipe = {"pipeline_class": "QwenImagePipeline", "model_index_path": "model_index.json", "components": [
        {"name": "scheduler", "library": "d", "class_name": "S", "has_weights": False},
        {"name": "text_encoder", "library": "t", "class_name": "Q", "has_weights": True},
        {"name": "tokenizer", "library": "t", "class_name": "T", "has_weights": False},
        {"name": "transformer", "library": "d", "class_name": "X", "has_weights": True},
        {"name": "vae", "library": "d", "class_name": "V", "has_weights": True}]}
    te = _card("text_encoder", config={"num_hidden_layers": 28, "vision_config": {"depth": 32}},
               archs=["Qwen2_5_VLForConditionalGeneration"], pipeline=pipe, repo="Qwen/Qwen-Image",
               stacks=[_stack("model.layers", 28), _stack("visual.blocks", 32)])
    tr = _card("transformer", config={"num_layers": 60}, class_name="QwenImageTransformer2DModel", pipeline=pipe,
               repo="Qwen/Qwen-Image",
               stacks=[_stack("transformer_blocks", 60)])
    vae = _card("vae", class_name="AutoencoderKLQwenImage", pipeline=pipe, repo="Qwen/Qwen-Image",
                files=[{"path": "vae/diffusion_pytorch_model.safetensors"}])
    return _res([te, tr, vae])


def _failing(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.ok]


def _mutate_card(res: TargetResult, fn, component=None):
    res = copy.deepcopy(res)
    for c in res.cards:
        if c["key"]["component"] == component:
            fn(c)
    return res


# ---------------------------------------------------------------------------
# E1..E3 pass


@pytest.mark.parametrize("builder,fn", [(_qwen3_8b, expect_qwen3_8b), (_qwen3_30b, expect_qwen3_30b_a3b),
                                        (_image, expect_qwen_image)])
def test_expectations_pass(builder, fn):
    checks = fn(builder())
    assert checks and all(isinstance(c, Check) for c in checks)
    assert _failing(checks) == []


def test_default_expectations_keys():
    assert set(DEFAULT_EXPECTATIONS) == {"Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen-Image"}


def test_empty_result_fails_without_raising():
    empty = _res([])
    for fn in DEFAULT_EXPECTATIONS.values():
        checks = fn(empty)
        assert checks and _failing(checks)
        assert any(c.actual.startswith("error: ") for c in checks)


# ---------------------------------------------------------------------------
# E1..E3 fail, one input per checked field: (builder, fn, mutator, failing name fragment)


E1_CASES = [
    ("wrong depth", lambda c: c["structure"]["stacks"][0].update(depth=35), "stacks[0].depth"),
    ("depth vs config", lambda c: c["config"]["raw"].update(num_hidden_layers=40), "num_hidden_layers"),
    ("wrong prefix", lambda c: c["structure"]["stacks"][0].update(prefix="layers"), "stacks[0].prefix"),
    ("wrong n_tensors", lambda c: c["weights"].update(n_tensors=398), "n_tensors"),
    ("wrong params", lambda c: c["weights"].update(params_total=1), "params_total"),
    ("wrong dtype", lambda c: c["weights"].update(params_by_dtype={"F16": 8190735360}), "params_by_dtype"),
    ("wrong model_type", lambda c: c["config"].update(model_type="qwen2"), "model_type"),
    ("wrong arch", lambda c: c["config"].update(architectures=["X"]), "architectures"),
    ("two signatures", lambda c: c["structure"]["stacks"][0]["block_signatures"].__setitem__(0, "z"),
     "distinct block_signatures"),
    ("expert groups", lambda c: c["structure"].update(expert_groups=[{"template": "t"}]), "expert_groups"),
    ("wrong base_model", lambda c: c["model_card"].update(base_model=["Qwen/Qwen3-8B"]), "base_model"),
    ("wrong license", lambda c: c["model_card"].update(license="mit"), "license"),
    ("no model card", lambda c: c["model_card"].update(present=False), "model_card.present"),
    ("missing config raw", lambda c: c["config"].update(raw=None), "num_hidden_layers"),
    ("missing stacks key", lambda c: c["structure"].pop("stacks"), "stacks[0]"),
]


@pytest.mark.parametrize("label,mut,frag", E1_CASES, ids=[c[0] for c in E1_CASES])
def test_e1_fail(label, mut, frag):
    checks = expect_qwen3_8b(_mutate_card(_qwen3_8b(), mut))
    assert any(not c.ok and frag in c.name for c in checks), [c for c in _failing(checks)]


def test_e1_missing_component_and_extra_card():
    res = _qwen3_8b()
    res.cards[0]["key"]["component"] = "text_encoder"
    checks = expect_qwen3_8b(res)
    assert _failing(checks)
    res = _qwen3_8b()
    res.cards.append(copy.deepcopy(res.cards[0]))
    assert any(not c.ok and "number of cards" in c.name for c in expect_qwen3_8b(res))


def _e2_view_mut(fn):
    def mut(res):
        fn(res.views[0])
    return mut


E2_CARD_CASES = [
    ("wrong depth", lambda c: c["structure"]["stacks"][0].update(depth=47), "stacks[0].depth"),
    ("wrong n_experts", lambda c: c["structure"]["expert_groups"][0].update(n_experts=64), "n_experts"),
    ("n_experts vs config", lambda c: c["config"]["raw"].update(num_experts=64), "config.num_experts"),
    ("two groups", lambda c: c["structure"]["expert_groups"].append(dict(c["structure"]["expert_groups"][0])),
     "number of expert_groups"),
    ("no groups", lambda c: c["structure"].update(expert_groups=[]), "number of expert_groups"),
    ("wrong template", lambda c: c["structure"]["expert_groups"][0].update(template="a.*"), "template"),
    ("n_instances", lambda c: c["structure"]["expert_groups"][0].update(n_instances=47), "n_instances"),
    ("tensors_per_expert", lambda c: c["structure"]["expert_groups"][0].update(tensors_per_expert=2),
     "tensors_per_expert"),
    ("params_per_expert", lambda c: c["structure"]["expert_groups"][0].update(params_per_expert=1),
     "params_per_expert"),
    ("hidden_size mismatch", lambda c: c["config"]["raw"].update(hidden_size=1024), "hidden_size"),
    ("not homogeneous", lambda c: c["structure"]["expert_groups"][0].update(homogeneous=False), "homogeneous"),
    ("wrong model_type", lambda c: c["config"].update(model_type="qwen3"), "model_type"),
    ("wrong n_tensors", lambda c: c["weights"].update(n_tensors=1), "n_tensors"),
    ("wrong params", lambda c: c["weights"].update(params_total=1), "params_total"),
    ("wrong base_model", lambda c: c["model_card"].update(base_model=[]), "base_model"),
    ("wrong license", lambda c: c["model_card"].update(license=None), "license"),
]


@pytest.mark.parametrize("label,mut,frag", E2_CARD_CASES, ids=[c[0] for c in E2_CARD_CASES])
def test_e2_card_fail(label, mut, frag):
    checks = expect_qwen3_30b_a3b(_mutate_card(_qwen3_30b(), mut))
    assert any(not c.ok and frag in c.name for c in checks), [c for c in _failing(checks)]


E2_VIEW_CASES = [
    ("strip 47 cells", lambda v: v["depth_strips"][0]["cells"].pop(), "depth strip cells"),
    ("strip non-moe cell", lambda v: v["depth_strips"][0]["cells"][3].update(moe=False), "all moe"),
    ("strip wrong n_experts", lambda v: v["depth_strips"][0]["cells"][3].update(n_experts=8), "n_experts == 128"),
    ("strip two signatures", lambda v: v["depth_strips"][0]["cells"][3].update(signature="z"), "signatures"),
    ("no expert node", lambda v: v["nodes"].pop(0), "expert_group nodes"),
    ("wrong node count", lambda v: v["nodes"][0].update(count=64), "node count"),
    ("no strips", lambda v: v.update(depth_strips=[]), "depth strip"),
]


@pytest.mark.parametrize("label,mut,frag", E2_VIEW_CASES, ids=[c[0] for c in E2_VIEW_CASES])
def test_e2_view_fail(label, mut, frag):
    res = copy.deepcopy(_qwen3_30b())
    mut(res.views[0])
    checks = expect_qwen3_30b_a3b(res)
    assert any(not c.ok and frag in c.name for c in checks), [c for c in _failing(checks)]


def test_e2_no_view_fails_without_raising():
    res = _qwen3_30b()
    res.views = []
    assert _failing(expect_qwen3_30b_a3b(res))


E3_CASES = [
    ("wrong text depth", "text_encoder", lambda c: c["structure"]["stacks"][0].update(depth=27),
     "text_encoder stack ending in 'layers'"),
    ("wrong vision depth", "text_encoder", lambda c: c["structure"]["stacks"][1].update(depth=31),
     "stack ending in 'blocks'"),
    ("missing vision_config", "text_encoder", lambda c: c["config"]["raw"].pop("vision_config"),
     "stack ending in 'blocks'"),
    ("no blocks stack", "text_encoder", lambda c: c["structure"]["stacks"].pop(1), "stack with prefix suffix 'blocks'"),
    ("wrong arch", "text_encoder", lambda c: c["config"].update(architectures=["Qwen2ForCausalLM"]),
     "architectures"),
    ("expert groups", "text_encoder", lambda c: c["structure"].update(expert_groups=[{}]), "expert_groups"),
    ("wrong transformer depth", "transformer", lambda c: c["structure"]["stacks"][0].update(depth=59),
     "transformer stack 'transformer_blocks' has expected depth"),
    ("missing num_layers", "transformer", lambda c: c["config"]["raw"].pop("num_layers"),
     "transformer stack 'transformer_blocks' has expected depth"),
    ("wrong transformer class", "transformer", lambda c: c["config"].update(class_name="X"), "transformer config"),
    ("wrong vae class", "vae", lambda c: c["config"].update(class_name="X"), "vae config.class_name"),
    ("vae two files", "vae", lambda c: c["weights"]["files"].append({"path": "b"}), "vae number of weight files"),
    ("vae index path", "vae", lambda c: c["weights"].update(index_path="i.json"), "vae index_path"),
    ("sha differs", "vae", lambda c: c["key"].update(revision_sha="b" * 40), "same revision_sha"),
    ("license differs", "vae", lambda c: c["model_card"].update(license="mit"), "license"),
    ("wrong pipeline class", "vae", lambda c: c["pipeline"].update(pipeline_class="Other"), "pipeline_class"),
    ("scheduler has weights", "vae",
     lambda c: [x.update(has_weights=True) for x in c["pipeline"]["components"] if x["name"] == "scheduler"],
     "scheduler"),
    ("tokenizer missing", "vae",
     lambda c: c["pipeline"].update(components=[x for x in c["pipeline"]["components"] if x["name"] != "tokenizer"]),
     "tokenizer"),
    ("vae no weights", "vae",
     lambda c: [x.update(has_weights=False) for x in c["pipeline"]["components"] if x["name"] == "vae"],
     "vae.has_weights"),
]


@pytest.mark.parametrize("label,comp,mut,frag", E3_CASES, ids=[c[0] for c in E3_CASES])
def test_e3_fail(label, comp, mut, frag):
    res = copy.deepcopy(_image())
    for c in res.cards:
        if c["key"]["component"] == comp:
            mut(c)
    # pipeline mutations apply to the shared pipeline dict in every card
    if label in ("wrong pipeline class", "scheduler has weights", "tokenizer missing", "vae no weights"):
        res = copy.deepcopy(_image())
        for c in res.cards:
            mut(c)
    checks = expect_qwen_image(res)
    assert any(not c.ok and frag in c.name for c in checks), [c for c in _failing(checks)]


def test_e3_missing_component_and_count():
    res = _image()
    res.cards = [c for c in res.cards if c["key"]["component"] != "vae"]
    checks = expect_qwen_image(res)
    assert any(not c.ok and "number of cards" in c.name for c in checks)
    assert any(not c.ok and c.name.startswith("E3 vae") for c in checks)
    res = _image()
    res.cards = [c for c in res.cards if c["key"]["component"] != "text_encoder"]
    assert any(not c.ok and c.name.startswith("E3 text_encoder") for c in expect_qwen_image(res))


def test_e3_text_config_depth_preferred():
    res = _image()
    te = res.cards[0]
    te["config"]["raw"]["text_config"] = {"num_hidden_layers": 40}
    assert any(not c.ok and "text_encoder stack ending in 'layers'" in c.name for c in expect_qwen_image(res))
    te["structure"]["stacks"][0]["depth"] = 40
    assert _failing(expect_qwen_image(res)) == []


def test_stack_suffix_boundary():
    res = _image()
    res.cards[0]["structure"]["stacks"][0]["prefix"] = "model.xlayers"
    checks = expect_qwen_image(res)
    assert any(not c.ok and "stack with prefix suffix 'layers' exists" in c.name for c in checks)
    assert any(not c.ok and "stack ending in 'layers'" in c.name for c in checks)
    # the "." boundary still matches a dotted prefix and a bare prefix
    assert ee._suffix_match("model.layers", "layers") and ee._suffix_match("layers", "layers")
    assert not ee._suffix_match("model.xlayers", "layers")


# ---------------------------------------------------------------------------
# common checks


HEADER_LEN = 100


def _table(component, numels, sha=SHA, repo="o/n"):
    n = len(numels)
    table = pa.table({"numel": pa.array(numels, pa.int64()), **{c: pa.nulls(n, pa.float64()) for c in STAT_COLS}})
    return table.replace_schema_metadata({"schema_version": "card.v0", "repo": repo, "revision_sha": sha,
                                          "component": "" if component is None else component})
DATA = 1000
SIZE = 8 + HEADER_LEN + DATA


def _common(tmp_path: pathlib.Path) -> TargetResult:
    files = [{"path": "model.safetensors", "size_bytes": SIZE, "header_len": HEADER_LEN, "data_bytes": DATA,
              "n_tensors": 2, "metadata": {}}]
    card = _card(None, files=files, n_tensors=2, params_total=15, index_path="model.safetensors.index.json",
                 index_total_size=DATA, tensor_bytes_total=DATA)
    table = _table(None, [5, 10])
    base = tmp_path / "o__n" / SHA
    base.mkdir(parents=True, exist_ok=True)
    jp, pp = base / "_model.card.json", base / "_model.tensors.parquet"
    jp.write_text("{}")
    pp.write_text("x")
    events = [
        {"event": "stage_start", "bytes_by_class": {"meta": 0, "header": 0, "weight": 0}},
        {"event": "fetch", "bytes_by_class": {"meta": 100, "header": 0, "weight": 0}},
        {"event": "fetch", "bytes_by_class": {"meta": 0, "header": 8, "weight": 0}},
        {"event": "fetch", "bytes_by_class": {"meta": 0, "header": HEADER_LEN, "weight": 0}},
    ]
    wire = [
        # T015 records the Hub API call with orig_path None; C10 exempts exactly this path on the endpoint host
        {"method": "GET", "url": "https://huggingface.co/api/models/o/n/revision/%s?blobs=true" % SHA,
         "orig_path": None, "range": None, "status": 200, "location": None, "body_bytes": 100},
        {"method": "GET", "url": "https://cas-bridge.xethub.hf.co/x", "orig_path": "/o/n/resolve/%s/model.safetensors" % SHA,
         "range": "bytes=0-7", "status": 206, "location": None, "body_bytes": 8},
        {"method": "GET", "url": "https://cas-bridge.xethub.hf.co/x", "orig_path": "/o/n/resolve/%s/model.safetensors" % SHA,
         "range": "bytes=8-107", "status": 206, "location": None, "body_bytes": HEADER_LEN},
    ]
    final = {"totals": {"meta": 100, "header": 8 + HEADER_LEN, "weight": 0}}
    return TargetResult(target="t", pin=SHA, status="done", elapsed_s=2.5, final=final, events=events,
                        cards=[card], tables=[table], views=[_view(None, nodes=[{"kind": "module"}])],
                        json_paths=[jp], parquet_paths=[pp], wire=wire)


def _fails(checks, frag):
    return [c for c in checks if not c.ok and frag in c.name]


def test_common_pass(tmp_path):
    checks = common_checks(_common(tmp_path), 10.0)
    assert _failing(checks) == []
    names = " ".join(c.name for c in checks)
    for k in range(1, 13):
        assert f"C{k} " in names


def test_c1(tmp_path):
    r = _common(tmp_path)
    r.elapsed_s = 10.0
    assert _fails(common_checks(r, 10.0), "C1 ")
    r = _common(tmp_path)
    r.status = "error"
    assert _fails(common_checks(r, 10.0), "C1 ")


def test_c2_weight_bytes(tmp_path):
    r = _common(tmp_path)
    r.final["totals"]["weight"] = 1
    assert _fails(common_checks(r, 10.0), "C2 server")
    r = _common(tmp_path)
    r.cards[0]["fetch_log"]["totals"]["weight"] = 1
    assert _fails(common_checks(r, 10.0), "C2 card")
    r = _common(tmp_path)
    r.events[1]["bytes_by_class"]["weight"] = 1
    assert _fails(common_checks(r, 10.0), "C2 sum")


def test_c3_c4_c5(tmp_path):
    r = _common(tmp_path)
    r.events[3]["bytes_by_class"]["header"] += 1
    assert _fails(common_checks(r, 10.0), "C3 ")
    r = _common(tmp_path)
    r.events[3]["event"] = "retry"          # only successful fetch events count
    assert _fails(common_checks(r, 10.0), "C3 ")
    r = _common(tmp_path)
    r.cards[0]["weights"]["files"][0]["data_bytes"] += 1
    assert _fails(common_checks(r, 10.0), "C4 ")
    r = _common(tmp_path)
    r.cards[0]["weights"]["index_total_size"] = DATA + 1
    assert _fails(common_checks(r, 10.0), "C5 ")
    r = _common(tmp_path)
    r.cards[0]["weights"]["index_path"] = None
    r.cards[0]["weights"]["index_total_size"] = 7        # ignored without index_path
    assert not _fails(common_checks(r, 10.0), "C5 ")


def test_c6(tmp_path):
    r = _common(tmp_path)
    r.json_paths[0].unlink()
    assert _fails(common_checks(r, 10.0), "C6 card json")
    r = _common(tmp_path)
    r.cards[0]["key"]["revision_sha"] = "b" * 40
    assert _fails(common_checks(r, 10.0), "C6 card[0]")


def test_c7_c9(tmp_path):
    r = _common(tmp_path)
    r.cards[0]["weights"]["n_tensors"] = 3
    assert _fails(common_checks(r, 10.0), "C7 card[0] _model parquet rows")
    r = _common(tmp_path)
    r.tables[0] = r.tables[0].set_column(2, "stat_std", pa.array([1.0, None], pa.float64()))
    assert _fails(common_checks(r, 10.0), "C7 card[0] _model every stat_")
    r = _common(tmp_path)
    r.cards[0]["stats"]["tensor_stats"] = {}
    assert _fails(common_checks(r, 10.0), "JSON stats")
    r = _common(tmp_path)
    r.cards[0]["weights"]["params_total"] = 16
    assert _fails(common_checks(r, 10.0), "C9 ")


def test_c8_node_limit(tmp_path):
    r = _common(tmp_path)
    r.views[0]["nodes"] = [{"kind": "module"}] * 100 + [{"kind": "tensor"}] * 500
    assert not _fails(common_checks(r, 10.0), "C8 ")
    r.views[0]["nodes"] = [{"kind": "module"}] * 101
    assert _fails(common_checks(r, 10.0), "C8 ")


def test_c10(tmp_path):
    r = _common(tmp_path)
    r.wire[2]["range"] = "bytes=8-108"       # b + 1 = 109 > 108
    assert _fails(common_checks(r, 10.0), "C10 /o/n")
    r = _common(tmp_path)
    r.wire[2]["range"] = None
    assert _fails(common_checks(r, 10.0), "C10 /o/n")
    r = _common(tmp_path)
    r.wire[2]["range"] = "garbage"
    assert _fails(common_checks(r, 10.0), "C10 /o/n")
    r = _common(tmp_path)
    r.wire[2]["orig_path"] = "/o/n/resolve/%s/other.safetensors" % SHA      # unknown file
    assert _fails(common_checks(r, 10.0), "C10 /o/n")
    r = _common(tmp_path)
    r.wire = r.wire[:1]
    r.final["totals"] = {"meta": 100, "header": 0, "weight": 0}
    fails = _fails(common_checks(r, 10.0), "C10 at least one")
    assert fails


def test_c11(tmp_path):
    r = _common(tmp_path)
    r.wire[0]["body_bytes"] += 1
    assert _fails(common_checks(r, 10.0), "C11 ")
    r = _common(tmp_path)
    r.final["totals"]["meta"] += 1
    assert _fails(common_checks(r, 10.0), "C11 ")


def test_c12(tmp_path):
    r = _common(tmp_path)
    assert not _fails(common_checks(r, 10.0), "C12 ")
    for bad in ("mirror.example.com", "evilhf.co", "hf.co", "huggingface.co.evil.com"):
        r = _common(tmp_path)
        r.wire[0]["url"] = f"https://{bad}/api"
        fails = _fails(common_checks(r, 10.0), "C12 every wire host")
        assert fails and bad in fails[0].actual
    r = _common(tmp_path)
    r.cards[0]["source"]["endpoint"] = "https://hub.test"
    assert _fails(common_checks(r, 10.0), "C12 every card source.endpoint")
    assert not _fails(common_checks(r, 10.0, expected_endpoint="https://hub.test"), "C12 every card")
    r = _common(tmp_path)
    r.wire = []
    assert _fails(common_checks(r, 10.0), "C12 at least one")


def test_host_allowed():
    assert host_allowed("huggingface.co", ee.ALLOWED_HOSTS)
    assert host_allowed("cas-bridge.xethub.hf.co", ee.ALLOWED_HOSTS)
    assert host_allowed("cdn-lfs.hf.co", ee.ALLOWED_HOSTS)
    assert not host_allowed("hf.co", ee.ALLOWED_HOSTS)
    assert not host_allowed("mirror.example.com", ee.ALLOWED_HOSTS)
    assert not host_allowed("evilhf.co", ee.ALLOWED_HOSTS)
    assert not host_allowed("", ee.ALLOWED_HOSTS)


def test_common_checks_never_raise_on_garbage():
    r = TargetResult(target="t", pin=SHA, status="done", elapsed_s=1.0, final={}, events=[{}], cards=[{}],
                     tables=[], views=[{}], json_paths=[], parquet_paths=[], wire=[{}])
    checks = common_checks(r, 10.0)
    assert checks and _failing(checks)
    assert any(c.actual.startswith("error: ") for c in checks)
    assert all(isinstance(c, Check) for c in checks)


# ---------------------------------------------------------------------------
# review round 1 fixes (each test fails without its fix)


@pytest.mark.parametrize("elapsed", [0, 0.0, -5, float("nan"), None, float("inf"), True, "1.0"])
def test_c1_elapsed_must_be_finite_positive_number(tmp_path, elapsed):
    r = _common(tmp_path)
    r.elapsed_s = elapsed
    assert _fails(common_checks(r, 10.0), "C1 ")


def _resolve(path, sha=SHA, repo="o/n"):
    return f"/{repo}/resolve/{sha}/{path}"


def _c10_fails(r):
    return _fails(common_checks(r, 10.0), "C10 ")


@pytest.mark.parametrize("orig", [
    _resolve("model.safetensors", repo="x/y"),          # wrong repo
    _resolve("model.safetensors", sha="main"),          # wrong revision
    _resolve("model.safetensors", sha="b" * 40),        # another pinned revision
    _resolve("MODEL.safetensors"),                      # only the extension is case-insensitive
    _resolve("sub/model.safetensors"),                  # suffix match is not a match
], ids=["wrong-repo", "wrong-revision-main", "wrong-revision-sha", "basename-case", "deeper-path"])
def test_c10_wire_must_match_repo_pin_and_card_file(tmp_path, orig):
    r = _common(tmp_path)
    r.wire[2]["orig_path"] = orig                       # Range is still valid: only the mapping is wrong
    fails = _c10_fails(r)
    assert any("Card weight file" in c.name and "matches no Card weight file" in c.actual for c in fails), fails


@pytest.mark.parametrize("orig", [_resolve("model.safetensors", repo="x/y"), _resolve("config.json", sha="main")])
def test_c10_every_record_must_be_under_repo_pin(tmp_path, orig):
    r = _common(tmp_path)
    r.wire.append({"method": "GET", "url": "https://huggingface.co" + orig, "orig_path": orig, "range": None,
                   "status": 200, "location": None, "body_bytes": 0})
    assert _fails(common_checks(r, 10.0), "C10 every wire record maps")


def test_c10_unmapped_redirect_fails(tmp_path):
    r = _common(tmp_path)
    r.wire.append({"method": "GET", "url": "https://cas-bridge.xethub.hf.co/opaque", "orig_path": None,
                   "range": None, "status": 200, "location": None, "body_bytes": 0})
    fails = _fails(common_checks(r, 10.0), "C10 every wire record maps")
    assert fails and "2xx without orig_path" in fails[0].actual
    # a 3xx or 4xx without orig_path carries no file bytes and is not an offender
    r.wire[-1]["status"] = 404
    assert not _c10_fails(r)


@pytest.mark.parametrize("url", [
    "https://huggingface.co/api/models/x/y/revision/%s" % SHA,       # API call for another repo
    "https://huggingface.co/api/models/o/n/revision/main",           # API call for another revision
    "https://cdn-lfs.hf.co/api/models/o/n/revision/%s" % SHA,        # right path, not the endpoint host
])
def test_c10_api_exemption_is_exact(tmp_path, url):
    r = _common(tmp_path)
    r.wire[0]["url"] = url
    assert _fails(common_checks(r, 10.0), "C10 every wire record maps")


@pytest.mark.parametrize("orig", [5, b"/o/n/resolve/x/model.safetensors"])
def test_c10_non_str_orig_path_fails(tmp_path, orig):
    r = _common(tmp_path)
    r.wire[2]["orig_path"] = orig
    assert _fails(common_checks(r, 10.0), "C10 every wire record maps")


def test_c10_uppercase_extension_without_range_fails(tmp_path):
    r = _common(tmp_path)
    r.wire[2]["orig_path"] = _resolve("model.SAFETENSORS")
    assert not _c10_fails(r)                            # maps to the Card file: ext compared case-insensitively
    r.wire[2]["range"] = None
    fails = _c10_fails(r)
    assert any(c.actual == "no Range header" for c in fails), fails


@pytest.mark.parametrize("name", ["other.SAFETENSORS", "other.SafeTensors", "other.safetensors"])
def test_c10_unknown_weight_file_any_case_fails(tmp_path, name):
    r = _common(tmp_path)
    r.wire[2]["orig_path"] = _resolve(name)             # right repo and pin, valid Range, but not a Card file
    fails = _c10_fails(r)
    assert any("matches no Card weight file" in c.actual for c in fails), fails


def test_c10_query_string_is_normalised(tmp_path):
    r = _common(tmp_path)
    r.wire[2]["orig_path"] = _resolve("model.safetensors") + "?download=true"
    assert not _c10_fails(r)
    r.wire[2]["range"] = None
    fails = _c10_fails(r)
    assert any(c.actual == "no Range header" for c in fails), fails
    r.wire[2]["range"] = "bytes=8-200"
    assert _c10_fails(r)


def test_c10_full_url_and_percent_encoding_are_normalised(tmp_path):
    r = _common(tmp_path)
    r.cards[0]["weights"]["files"][0]["path"] = "sub dir/model.safetensors"
    r.wire[1]["orig_path"] = "https://huggingface.co" + _resolve("sub%20dir/model.safetensors")
    r.wire[2]["orig_path"] = _resolve("sub%20dir/model.safetensors")
    assert not _c10_fails(r)


def test_c10_open_or_reversed_range_fails(tmp_path):
    for rng in ("bytes=8-", "bytes=50-8", "bytes=0-7,8-107"):
        r = _common(tmp_path)
        r.wire[2]["range"] = rng
        assert _c10_fails(r), rng


def test_c10_missing_record_for_a_card_file_fails(tmp_path):
    r = _common(tmp_path)
    r.cards[0]["weights"]["files"].append({"path": "model-2.safetensors", "size_bytes": SIZE,
                                           "header_len": HEADER_LEN, "data_bytes": DATA, "n_tensors": 0,
                                           "metadata": {}})
    fails = _fails(common_checks(r, 10.0), "C10 _model model-2.safetensors: >= 1 2xx")
    assert fails and fails[0].actual == "0"
    assert not _fails(common_checks(r, 10.0), "C10 _model model.safetensors: >= 1 2xx")


def test_c10_only_3xx_for_a_card_file_is_not_enough(tmp_path):
    r = _common(tmp_path)
    for w in r.wire[1:]:
        w["status"] = 302
    assert _fails(common_checks(r, 10.0), "C10 _model model.safetensors: >= 1 2xx")


def test_c10_redirect_records_are_range_checked(tmp_path):
    r = _common(tmp_path)
    r.wire.insert(1, {"method": "GET", "url": "https://huggingface.co" + _resolve("model.safetensors"),
                      "orig_path": _resolve("model.safetensors"), "range": None, "status": 302,
                      "location": "https://cas-bridge.xethub.hf.co/x", "body_bytes": 0})
    assert _c10_fails(r)


@pytest.mark.parametrize("attr,bad", [(a, b) for a in ("cards", "views", "wire", "tables", "events", "json_paths",
                                                       "parquet_paths") for b in (None, {}, "x", 3)]
                         + [("final", None), ("final", []), ("cards", [None]), ("wire", ["x"]),
                            ("tables", [{}]), ("views", [3]), ("json_paths", [None])])
def test_common_checks_never_raise_on_wrong_field_types(tmp_path, attr, bad):
    r = _common(tmp_path)
    setattr(r, attr, bad)
    checks = common_checks(r, 10.0)
    assert all(isinstance(c, Check) for c in checks)
    assert _fails(checks, f"TYPE field {attr} ")


@pytest.mark.parametrize("attr,bad", [("cards", None), ("cards", "x"), ("cards", [None, 3])])
def test_expectations_never_raise_on_bad_cards(attr, bad):
    for builder, fn in ((_qwen3_8b, expect_qwen3_8b), (_qwen3_30b, expect_qwen3_30b_a3b), (_image, expect_qwen_image)):
        r = builder()
        setattr(r, attr, bad)
        checks = fn(r)
        assert checks and _failing(checks)


@pytest.mark.parametrize("bad", [None, "x", [None]])
def test_e2_never_raises_on_bad_views(bad):
    r = _qwen3_30b()
    r.views = bad
    checks = expect_qwen3_30b_a3b(r)
    assert any(not c.ok and "depth strip" in c.name for c in checks)


def _move(r, paths_attr, i, new_path):
    old = getattr(r, paths_attr)[i]
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new_path)
    getattr(r, paths_attr)[i] = new_path


@pytest.mark.parametrize("which,rel,frag", [
    ("both", ("o__n", "main"), "C6 path layout"),                         # parent dir is not the pin
    ("both", ("o_n", SHA), "C6 card[0] _model files at"),                 # wrong repo_slug
    ("parquet", ("o__n", SHA, "_model.parquet"), "C6 path layout"),       # wrong suffix
    ("json", ("o__n", SHA, "_model.json"), "C6 path layout"),             # wrong suffix
    ("parquet", ("o__n", SHA, "other.tensors.parquet"), "C6 path layout"),  # stems differ
    ("parquet", ("o__n", "x", SHA, "_model.tensors.parquet"), "C6 path layout"),  # same stem, other dir
    ("both", ("o__n", SHA, "model"), "C6 card[0] _model files at"),       # wrong component_slug
])
def test_c6_layout(tmp_path, which, rel, frag):
    r = _common(tmp_path)
    if which == "both":
        stem = rel[2] if len(rel) == 3 else "_model"
        d = tmp_path.joinpath(*rel[:2])
        _move(r, "json_paths", 0, d / f"{stem}.card.json")
        _move(r, "parquet_paths", 0, d / f"{stem}.tensors.parquet")
    else:
        _move(r, f"{which}_paths", 0, tmp_path.joinpath(*rel))
    assert not _fails(common_checks(r, 10.0), "C6 card json and parquet files exist")
    assert _fails(common_checks(r, 10.0), frag)


def test_c7_c9_pair_tables_by_metadata_not_position(tmp_path):
    r = _common(tmp_path)
    vae = _card("vae", n_tensors=1, params_total=7)
    r.cards.append(vae)
    r.tables = [_table("vae", [7]), r.tables[0]]        # reversed relative to cards
    checks = common_checks(r, 10.0)
    assert not _fails(checks, "C7 ") and not _fails(checks, "C9 ")


@pytest.mark.parametrize("meta", [None, {"component": "vae", "revision_sha": SHA},
                                  {"component": "", "revision_sha": "b" * 40}, {"revision_sha": SHA}])
def test_c7_c9_table_metadata_must_match_card(tmp_path, meta):
    r = _common(tmp_path)
    r.tables[0] = r.tables[0].replace_schema_metadata(meta)
    checks = common_checks(r, 10.0)
    assert _fails(checks, "C7 card[0] _model parquet rows") and _fails(checks, "C9 ")


def test_c7_duplicate_tables_for_one_card_fail(tmp_path):
    r = _common(tmp_path)
    r.tables.append(r.tables[0])
    assert _fails(common_checks(r, 10.0), "C7 card[0] _model parquet rows")


def test_e2_exactly_one_model_layers_strip():
    res = copy.deepcopy(_qwen3_30b())
    res.views[0]["depth_strips"].append(copy.deepcopy(res.views[0]["depth_strips"][0]))
    checks = expect_qwen3_30b_a3b(res)
    assert any(not c.ok and "exactly one model.layers depth strip" in c.name for c in checks)


def test_e3_transformer_blocks_prefix_is_exact():
    res = _image()
    res.cards[1]["structure"]["stacks"][0]["prefix"] = "foo.transformer_blocks"
    checks = expect_qwen_image(res)
    assert any(not c.ok and "stack with prefix 'transformer_blocks' exists" in c.name for c in checks)
    assert any(not c.ok and "stack 'transformer_blocks' has expected depth" in c.name for c in checks)


@pytest.mark.parametrize("archs", ["Qwen2_5_VLForConditionalGeneration", "xQwen2_5_VLForConditionalGenerationx",
                                   ("Qwen2_5_VLForConditionalGeneration",), None])
def test_e3_architectures_must_be_a_list(archs):
    res = _image()
    res.cards[0]["config"]["architectures"] = archs
    checks = expect_qwen_image(res)
    assert any(not c.ok and "text_encoder architectures" in c.name for c in checks)


def test_e3_pipeline_identical_across_cards():
    res = copy.deepcopy(_image())
    vae = res.cards[2]
    vae["pipeline"] = copy.deepcopy(vae["pipeline"])    # _image() shares one pipeline dict across cards
    for comp in vae["pipeline"]["components"]:
        if comp["name"] == "tokenizer":
            comp["has_weights"] = True                  # only the vae card's copy differs
    checks = expect_qwen_image(res)
    assert any(not c.ok and "identical pipeline" in c.name for c in checks)
    res.cards[2]["pipeline"] = None
    assert any(not c.ok and "identical pipeline" in c.name for c in expect_qwen_image(res))


# ---------------------------------------------------------------------------
# follow-ups: target repo per expectation; non-safetensors weight requests


@pytest.mark.parametrize("builder,fn,label", [(_qwen3_8b, expect_qwen3_8b, "E1"),
                                              (_qwen3_30b, expect_qwen3_30b_a3b, "E2"),
                                              (_image, expect_qwen_image, "E3")])
@pytest.mark.parametrize("which", ["all", "one"])
def test_expect_card_repo_is_target_repo(builder, fn, label, which):
    res = copy.deepcopy(builder())
    cards = res.cards if which == "all" else res.cards[-1:]
    for c in cards:
        c["key"]["repo"] = "someone/fork-of-qwen"        # same sha, same content, different repo
    checks = fn(res)
    assert any(not c.ok and c.name.startswith(f"{label} every card key.repo ==") for c in checks)


def test_weight_extensions_match_bytelog():
    from scout.bytelog import WEIGHT_EXTENSIONS
    assert set(ee.WEIGHT_EXTENSIONS) == set(WEIGHT_EXTENSIONS)


@pytest.mark.parametrize("ext", [e for e in ee.WEIGHT_EXTENSIONS if e != ".safetensors"] + [".BIN", ".GGUF"])
@pytest.mark.parametrize("status", [200, 302, 404])
def test_c10_non_safetensors_weight_request_is_offender(tmp_path, ext, status):
    r = _common(tmp_path)
    orig = _resolve("pytorch_model" + ext)              # right repo and pin; any status, even with a Range
    r.wire.append({"method": "GET", "url": "https://huggingface.co" + orig, "orig_path": orig,
                   "range": "bytes=0-7", "status": status, "location": None, "body_bytes": 0})
    fails = _fails(common_checks(r, 10.0), "C10 every wire record maps")
    assert fails and "non-safetensors weight file" in fails[0].actual
