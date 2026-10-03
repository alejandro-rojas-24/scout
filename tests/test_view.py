import json

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
