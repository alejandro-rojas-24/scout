"""Tests for scout.structure (T008)."""
from __future__ import annotations

import json
import math
import time

import pytest

from scout.errors import DuplicateTensorError
from scout.safetensors_header import DTYPE_BYTES, TensorInfo
from scout.structure import ExpertGroup, Stack, Structure, TensorPlace, analyze, place
from tests.helpers.st_fixtures import dense_tensors, moe_tensors, params_of


def _infos(spec: dict[str, tuple[str, tuple[int, ...]]]) -> list[TensorInfo]:
    out: list[TensorInfo] = []
    cursor = 0
    for name, (dtype, shape) in spec.items():
        numel = math.prod(shape)
        nbytes = numel * DTYPE_BYTES[dtype]
        out.append(TensorInfo(name, dtype, tuple(shape), numel, nbytes, "m.safetensors", cursor, cursor + nbytes))
        cursor += nbytes
    return out


def _w(*shape: int, dtype: str = "F32") -> tuple[str, tuple[int, ...]]:
    return (dtype, tuple(shape))


def test_place():
    p = place("model.layers.3.mlp.experts.17.up_proj.weight")
    assert p == TensorPlace("model.layers.#.mlp.experts.*.up_proj.weight", "model.layers", 3, 17)
    assert (p.collapsed_name, p.stack_prefix, p.block_index, p.expert_index) == (
        "model.layers.#.mlp.experts.*.up_proj.weight", "model.layers", 3, 17)
    q = place("lm_head.weight")
    assert q == TensorPlace("lm_head.weight", None, None, None)
    # expert index without any stack integer
    r = place("mlp.experts.2.w")
    assert r == TensorPlace("mlp.experts.*.w", None, None, 2)
    # only the first integer becomes the stack; deeper ones stay literal
    s = place("decoder.up_blocks.1.resnets.2.conv1.weight")
    assert s == TensorPlace("decoder.up_blocks.#.resnets.2.conv1.weight", "decoder.up_blocks", 1, None)


def test_dense_fixture():
    spec = dense_tensors(n_layers=4)
    st = analyze(_infos(spec))
    assert isinstance(st, Structure)
    assert len(st.stacks) == 1
    s = st.stacks[0]
    assert isinstance(s, Stack)
    assert s.prefix == "model.layers" and s.depth == 4 and s.indices == [0, 1, 2, 3]
    assert len(set(s.block_signatures)) == 1
    assert all(len(sig) == 12 and int(sig, 16) >= 0 for sig in s.block_signatures)
    assert s.block_moe == [False] * 4 and s.block_n_experts == [None] * 4
    assert st.expert_groups == []
    assert st.params_total == params_of(spec)
    assert st.n_tensors == len(spec)
    assert st.params_by_dtype == {"BF16": params_of(spec)}
    assert sum(s.block_params) + 32 * 8 * 2 + 8 == st.params_total  # embed + lm_head + final norm
    assert st.warnings == []
    assert set(st.places) == set(spec)
    assert st.places["model.norm.weight"] == TensorPlace("model.norm.weight", None, None, None)


def test_moe_fixture():
    spec = moe_tensors(n_layers=3, n_experts=4, hidden=8, moe_inter=4)
    st = analyze(_infos(spec))
    assert len(st.stacks) == 1
    s = st.stacks[0]
    assert s.prefix == "model.layers" and s.depth == 3
    assert s.block_moe == [True, True, True]
    assert s.block_n_experts == [4, 4, 4]
    assert len(set(s.block_signatures)) == 1
    assert len(st.expert_groups) == 1
    g = st.expert_groups[0]
    assert isinstance(g, ExpertGroup)
    assert g.template == "model.layers.#.mlp.experts.*"
    assert (g.n_experts, g.n_instances, g.tensors_per_expert) == (4, 3, 3)
    assert g.params_per_expert == 3 * 8 * 4
    assert g.homogeneous is True
    assert st.warnings == []
    assert st.params_total == params_of(spec)


def test_nested_indices():
    spec = {f"encoder.down_blocks.{i}.resnets.{r}.conv1.weight": _w(2, 2) for i in (0, 1) for r in (0, 1)}
    st = analyze(_infos(spec))
    assert [s.prefix for s in st.stacks] == ["encoder.down_blocks"]
    s = st.stacks[0]
    assert s.depth == 2 and s.indices == [0, 1] and s.block_params == [8, 8]
    assert st.places["encoder.down_blocks.1.resnets.0.conv1.weight"].collapsed_name == \
        "encoder.down_blocks.#.resnets.0.conv1.weight"
    assert st.places["encoder.down_blocks.0.resnets.1.conv1.weight"].collapsed_name == \
        "encoder.down_blocks.#.resnets.1.conv1.weight"
    assert {p.collapsed_name for p in st.places.values()} == {
        "encoder.down_blocks.#.resnets.0.conv1.weight", "encoder.down_blocks.#.resnets.1.conv1.weight"}


def test_single_index_not_stack():
    st = analyze(_infos({"visual.merger.mlp.0.weight": _w(4, 4)}))
    assert st.stacks == []
    assert st.places["visual.merger.mlp.0.weight"] == TensorPlace("visual.merger.mlp.0.weight", None, None, None)


def test_single_index_keeps_expert_collapse():
    spec = {f"moe.0.experts.{j}.w": _w(2, 2) for j in range(3)}
    st = analyze(_infos(spec))
    assert st.stacks == []
    assert st.places["moe.0.experts.2.w"] == TensorPlace("moe.0.experts.*.w", None, None, 2)
    [g] = st.expert_groups
    assert (g.template, g.n_experts, g.n_instances, g.params_per_expert) == ("moe.0.experts.*", 3, 1, 4)


def test_noncontiguous_indices():
    spec = {"blocks.0.w": _w(2), "blocks.2.w": _w(2), "blocks.0.b": _w(1)}
    st = analyze(_infos(spec))
    [s] = st.stacks
    assert s.depth == 2 and s.indices == [0, 2] and s.block_params == [3, 2]


def test_heterogeneous_experts():
    spec = moe_tensors(n_layers=2, n_experts=4, hidden=8, moe_inter=4)
    spec["model.layers.1.mlp.experts.2.up_proj.weight"] = ("BF16", (5, 8))
    st = analyze(_infos(spec))
    [g] = st.expert_groups
    assert g.homogeneous is False
    assert any("heterogeneous experts" in w and g.template in w for w in st.warnings)


def test_varying_expert_count():
    spec = {}
    for i, n in ((0, 4), (1, 2)):
        for j in range(n):
            spec[f"layers.{i}.experts.{j}.w"] = _w(2, 2)
    st = analyze(_infos(spec))
    [g] = st.expert_groups
    assert g.n_experts == 4 and g.n_instances == 2
    assert "layers.#.experts.*: expert count varies [2, 4]" in st.warnings
    assert g.params_per_expert == 4  # 24 params / (4 + 2) experts
    assert st.stacks[0].block_n_experts == [4, 2]


def test_params_not_divisible_warns():
    # same tensor set per expert but different dtype-free shapes break exact division
    spec = {"layers.0.experts.0.w": _w(2, 2), "layers.0.experts.1.w": _w(3, 1),
            "layers.1.experts.0.w": _w(2, 2), "layers.1.experts.1.w": _w(2, 2)}
    st = analyze(_infos(spec))
    [g] = st.expert_groups
    assert g.homogeneous is False
    assert g.params_per_expert == 15 // 4
    assert any("not divisible" in w for w in st.warnings)


def test_signature_differs():
    spec = dense_tensors(n_layers=3)
    spec["model.layers.1.self_attn.q_proj.bias"] = ("BF16", (8,))
    st = analyze(_infos(spec))
    sigs = st.stacks[0].block_signatures
    assert sigs[0] == sigs[2] != sigs[1]


def test_signature_depends_on_dtype_and_shape():
    a = analyze(_infos({"l.0.w": _w(2, 2), "l.1.w": _w(2, 2, dtype="BF16"), "l.2.w": _w(4, 1)}))
    assert len(set(a.stacks[0].block_signatures)) == 3


def test_signature_includes_expert_count():
    spec = {}
    for i, n in ((0, 2), (1, 3)):
        for j in range(n):
            spec[f"layers.{i}.experts.{j}.w"] = _w(2, 2)
    sigs = analyze(_infos(spec)).stacks[0].block_signatures
    assert sigs[0] != sigs[1]


def test_stack_order():
    spec = {"small.0.w": _w(1), "small.1.w": _w(1), "big.0.w": _w(9), "big.1.w": _w(9),
            "b2.0.w": _w(1), "b2.1.w": _w(1)}
    st = analyze(_infos(spec))
    assert [s.prefix for s in st.stacks] == ["big", "b2", "small"]


def test_duplicate():
    infos = _infos({"a.w": _w(2)})
    with pytest.raises(DuplicateTensorError, match="a.w"):
        analyze(infos + infos)


def test_fused_experts_warning():
    spec = {"model.layers.0.mlp.experts.gate_up_proj": _w(4, 8, 16),
            "model.layers.0.mlp.experts.foo": _w(8, 16)}
    st = analyze(_infos(spec))
    fused = [w for w in st.warnings if "fused expert" in w]
    assert len(fused) == 1
    assert "leading expert dim 4" in fused[0]
    assert st.expert_groups == []
    # the 2-D experts.foo tensor alone gives no warning
    st2 = analyze(_infos({"model.layers.0.mlp.experts.foo": _w(8, 16)}))
    assert st2.warnings == [] and st2.expert_groups == []


def test_fused_experts_one_warning_per_collapsed_parent():
    spec = {}
    for i in range(3):
        spec[f"model.layers.{i}.mlp.experts.gate_up_proj"] = _w(4, 8, 16)
        spec[f"model.layers.{i}.mlp.experts.down_proj"] = _w(4, 8, 8)
    st = analyze(_infos(spec))
    assert st.warnings == [
        "model.layers.#.mlp.experts: fused expert tensors detected (leading expert dim 4); not collapsed in v0"]
    assert st.stacks[0].block_moe == [False] * 3


def test_to_dict():
    st = analyze(_infos(moe_tensors(n_layers=2)))
    d = st.to_dict()
    assert set(d) == {"stacks", "expert_groups", "warnings"}
    assert set(d["stacks"][0]) == {"prefix", "depth", "indices", "block_params", "block_signatures",
                                   "block_moe", "block_n_experts"}
    assert set(d["expert_groups"][0]) == {"template", "n_experts", "n_instances", "params_per_expert",
                                          "tensors_per_expert", "homogeneous"}
    json.dumps(d)


def test_params_by_dtype_sorted():
    st = analyze(_infos({"b": _w(3, dtype="F32"), "a": _w(2, dtype="BF16")}))
    assert list(st.params_by_dtype) == ["BF16", "F32"]
    assert st.params_by_dtype == {"BF16": 2, "F32": 3}


def test_scale():
    infos = _infos(moe_tensors(n_layers=48, n_experts=128, hidden=8, moe_inter=4))
    assert len(infos) == 18867
    t0 = time.perf_counter()
    st = analyze(infos)
    assert time.perf_counter() - t0 < 2.0
    [g] = st.expert_groups
    assert (g.n_experts, g.n_instances, g.tensors_per_expert, g.homogeneous) == (128, 48, 3, True)
    assert st.stacks[0].depth == 48
