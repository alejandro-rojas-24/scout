import copy
import json

import pytest

from scout.card import load_card, write_card
from scout.errors import ScoutError
from scout.view import DISCLAIMERS, build_view
from tests.helpers.cards import build_fixture_card
from tests.helpers.st_fixtures import (
    DEFAULT_README, moe_tensors, write_dense_repo, write_moe_repo, write_pipeline_repo,
    write_sharded,
)


def _view(tmp_path, root, component=None):
    built = build_fixture_card(root, tmp_path / "out", component)
    return built, build_view(built.card, built.table)


def _by_id(view):
    return {n["id"]: n for n in view["nodes"]}


def test_dense_view(tmp_path):
    write_dense_repo(tmp_path / "r", n_layers=4)
    built, view = _view(tmp_path, tmp_path / "r")
    nodes = _by_id(view)
    n = nodes["model.layers[#]"]
    assert n["kind"] == "stack" and n["count"] == 4 and n["label"] == "layers ×4"
    assert nodes[""]["kind"] == "root" and nodes[""]["parent"] is None
    assert nodes[""]["params"] == built.card["weights"]["params_total"]
    assert view["summary"]["params_total"] == nodes[""]["params"]
    assert not [x for x in view["nodes"] if x["kind"] == "expert_group"]
    assert len(nodes) == len(view["nodes"])
    t = nodes["lm_head.weight"]
    assert t["kind"] == "tensor" and t["shape"] == [32, 8] and t["dtype"] == "BF16"
    assert view["title"].startswith(built.card["key"]["repo"] + "@")


def test_moe_view(tmp_path):
    write_moe_repo(tmp_path / "r", n_layers=3, n_experts=4)
    built, view = _view(tmp_path, tmp_path / "r")
    groups = [x for x in view["nodes"] if x["kind"] == "expert_group"]
    assert len(groups) == 1
    assert groups[0]["count"] == built.card["structure"]["expert_groups"][0]["n_experts"] == 4
    assert groups[0]["label"] == "experts ×4"
    assert len(view["nodes"]) < 40


def test_strips(tmp_path):
    write_moe_repo(tmp_path / "r", n_layers=3, n_experts=4)
    built, view = _view(tmp_path, tmp_path / "r")
    assert len(view["depth_strips"]) == len(built.card["structure"]["stacks"])
    for strip, st in zip(view["depth_strips"], built.card["structure"]["stacks"]):
        assert strip["prefix"] == st["prefix"] and strip["depth"] == st["depth"]
        assert len(strip["cells"]) == strip["depth"]
        assert [c["moe"] for c in strip["cells"]] == st["block_moe"]
        assert [c["n_experts"] for c in strip["cells"]] == st["block_n_experts"]
        assert [c["index"] for c in strip["cells"]] == st["indices"]
        assert [c["params"] for c in strip["cells"]] == st["block_params"]
        assert [c["signature"] for c in strip["cells"]] == st["block_signatures"]
    assert all(c["moe"] for c in view["depth_strips"][0]["cells"])


def test_collapse_scale(tmp_path):
    root = tmp_path / "r"
    write_sharded(root, moe_tensors(n_layers=48, n_experts=128, hidden=8, moe_inter=4), 1)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    (root / "README.md").write_text(DEFAULT_README)
    built, view = _view(tmp_path, root)
    assert len(view["nodes"]) <= 100
    groups = [x for x in view["nodes"] if x["kind"] == "expert_group"]
    assert len(groups) == 1 and groups[0]["count"] == 128
    assert _by_id(view)[""]["params"] == built.card["weights"]["params_total"]


def test_json(tmp_path):
    write_dense_repo(tmp_path / "r")
    _, view = _view(tmp_path, tmp_path / "r")
    assert json.loads(json.dumps(view)) == view


def test_disclaimers(tmp_path):
    write_dense_repo(tmp_path / "r")
    _, view = _view(tmp_path, tmp_path / "r")
    assert view["disclaimers"] == list(DISCLAIMERS)
    assert len(view["disclaimers"]) == 3
    for word, d in zip(("Distillation", "Tokenizer", "Licensing"), view["disclaimers"]):
        assert word in d


def test_wan_vae_bound(tmp_path):
    write_pipeline_repo(tmp_path / "r")
    built, view = _view(tmp_path, tmp_path / "r", "vae")
    nodes = _by_id(view)
    n = nodes["decoder.up_blocks[#].resnets[#]"]
    assert n["kind"] == "stack" and n["count"] == 3
    assert nodes["decoder.up_blocks[#]"]["count"] == 4
    non_tensor = [x for x in view["nodes"] if x["kind"] != "tensor"]
    assert len(non_tensor) <= 100 and len(non_tensor) < 40
    assert nodes[""]["params"] == built.card["weights"]["params_total"]
    assert nodes[""]["label"] == "vae"
    assert view["title"].endswith(" / vae")


# ---------------------------------------------------------------- T011 amendments

def _custom_view(tmp_path, tensors):
    root = tmp_path / "r"
    write_sharded(root, tensors, 1)
    (root / "config.json").write_text(json.dumps({"model_type": "custom"}))
    (root / "README.md").write_text(DEFAULT_README)
    return _view(tmp_path, root)


def _depth(card, prefix):
    return {s["prefix"]: s["depth"] for s in card["structure"]["stacks"]}[prefix]


def test_consecutive_markers_unet_input_blocks(tmp_path):
    # input_blocks.0 has one sub-module, the other 11 have two: 12 x up to 2.
    t = {}
    for i in range(12):
        for j in range(1 if i == 0 else 2):
            t[f"model.diffusion_model.input_blocks.{i}.{j}.weight"] = ("F16", (4, 4))
    built, view = _custom_view(tmp_path, t)
    n = _by_id(view)["model.diffusion_model.input_blocks[#][#]"]
    assert _depth(built.card, "model.diffusion_model.input_blocks") == 12
    assert n["kind"] == "stack" and n["count"] == 12
    assert n["label"] == "input_blocks ×12 ×2"
    assert n["parent"] == "model.diffusion_model"


def test_consecutive_markers_count_is_card_depth(tmp_path):
    t = {f"x.a.{i}.{j}.weight": ("F32", (2,)) for i in range(2) for j in range(3)}
    built, view = _custom_view(tmp_path, t)
    n = _by_id(view)["x.a[#][#]"]
    assert _depth(built.card, "x.a") == 2
    assert n["kind"] == "stack" and n["count"] == 2 and n["label"] == "a ×2 ×3"


def test_root_level_stack(tmp_path):
    t = {f"{i}.weight": ("F32", (3, 3)) for i in range(5)}
    built, view = _custom_view(tmp_path, t)
    nodes = _by_id(view)
    n = nodes["[#]"]
    assert n["kind"] == "stack" and n["count"] == 5 and n["parent"] == "" and n["label"] == "×5"
    leaf = nodes["[#].weight"]
    assert leaf["kind"] == "tensor" and leaf["parent"] == "[#]" and leaf["shape"] == [3, 3]
    assert nodes[""]["params"] == built.card["weights"]["params_total"] == 45
    assert len(view["nodes"]) == 3


def test_mixed_dtypes_same_shape(tmp_path):
    t = {
        "l.0.w": ("BF16", (4, 2)), "l.1.w": ("F16", (4, 2)),     # dtypes differ, shape equal
        "l.0.b": ("F32", (4,)), "l.1.b": ("F32", (2,)),          # shapes differ, dtype equal
        "l.0.c": ("F16", (4,)), "l.1.c": ("BF16", (2,)),         # both differ
    }
    _, view = _custom_view(tmp_path, t)
    nodes = _by_id(view)
    assert nodes["l[#].w"]["dtype"] == "varies" and nodes["l[#].w"]["shape"] == [4, 2]
    assert nodes["l[#].b"]["dtype"] == "F32" and nodes["l[#].b"]["shape"] is None
    assert nodes["l[#].c"]["dtype"] == "varies" and nodes["l[#].c"]["shape"] is None


def test_card_parquet_mismatch_raises(tmp_path):
    write_moe_repo(tmp_path / "r", n_layers=3, n_experts=4)
    built = build_fixture_card(tmp_path / "r", tmp_path / "out")
    bad_stack = copy.deepcopy(built.card)
    bad_stack["structure"]["stacks"][0]["prefix"] = "nope"
    with pytest.raises(ScoutError, match="stack"):
        build_view(bad_stack, built.table)
    bad_group = copy.deepcopy(built.card)
    bad_group["structure"]["expert_groups"][0]["template"] = "nope.*"
    with pytest.raises(ScoutError, match="expert"):
        build_view(bad_group, built.table)


def test_persisted_round_trip(tmp_path):
    write_moe_repo(tmp_path / "r", n_layers=3, n_experts=4)
    built = build_fixture_card(tmp_path / "r", tmp_path / "out")
    paths = write_card(built)
    card, table = load_card(paths.json_path)
    assert build_view(card, table) == build_view(built.card, built.table)


def test_uneven_inner_indices(tmp_path):
    t = {
        "d.blocks.0.resnets.0.w": ("F32", (2,)), "d.blocks.0.resnets.2.w": ("F32", (2,)),
        "d.blocks.1.resnets.0.w": ("F32", (2,)), "d.blocks.1.resnets.1.w": ("F32", (2,)),
    }
    _, view = _custom_view(tmp_path, t)
    nodes = _by_id(view)
    n = nodes["d.blocks[#].resnets[#]"]
    assert n["kind"] == "stack" and n["count"] == 3 and n["label"] == "resnets ×3"
    assert nodes["d.blocks[#]"]["count"] == 2


def test_long_digit_segment_is_not_an_index(tmp_path):
    t = {"a.0.w": ("F32", (2, 2)), "a.1.w": ("F32", (2, 2)), "a.1234567890.w": ("F32", (3,))}
    built, view = _custom_view(tmp_path, t)
    nodes = _by_id(view)
    strip = next(s for s in view["depth_strips"] if s["prefix"] == "a")
    assert strip["depth"] == 2
    assert nodes["a[#]"]["params"] == sum(c["params"] for c in strip["cells"]) == 8
    leaf = nodes["a.1234567890.w"]
    assert leaf["kind"] == "tensor" and leaf["parent"] == "a.1234567890"
    assert not any(n["id"].startswith("a[#]") and "1234567890" in n["id"] for n in view["nodes"])
    assert nodes[""]["params"] == built.card["weights"]["params_total"]
