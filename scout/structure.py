"""Structure analysis: repeated-block stacks, MoE expert collapsing, block signatures (pure, no IO)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

from scout.errors import DuplicateTensorError
from scout.safetensors_header import TensorInfo

EXPERT_CONTAINERS: tuple[str, ...] = ("experts",)
STACK_MARK = "#"
EXPERT_MARK = "*"


@dataclass(frozen=True)
class TensorPlace:
    collapsed_name: str
    stack_prefix: str | None
    block_index: int | None
    expert_index: int | None


@dataclass(frozen=True)
class Stack:
    prefix: str
    depth: int
    indices: list[int]
    block_params: list[int]
    block_signatures: list[str]
    block_moe: list[bool]
    block_n_experts: list[int | None]


@dataclass(frozen=True)
class ExpertGroup:
    template: str
    n_experts: int
    n_instances: int
    params_per_expert: int
    tensors_per_expert: int
    homogeneous: bool


@dataclass(frozen=True)
class Structure:
    stacks: list[Stack]
    expert_groups: list[ExpertGroup]
    places: dict[str, TensorPlace]
    params_total: int
    n_tensors: int
    params_by_dtype: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "stacks": [asdict(s) for s in self.stacks],
            "expert_groups": [asdict(g) for g in self.expert_groups],
            "warnings": list(self.warnings),
        }


def _is_index(seg: str) -> bool:
    # ASCII-only so int() never fails on unicode digits such as "²".
    return seg.isascii() and seg.isdigit()


def _split(name: str) -> tuple[list[str], int | None, int | None]:
    """Collapse `name` into segments. Returns (segs, stack_pos, expert_pos); marks already applied."""
    segs = name.split(".")
    expert_pos: int | None = None
    for i in range(len(segs) - 1):
        if segs[i] in EXPERT_CONTAINERS and _is_index(segs[i + 1]):
            expert_pos = i + 1
            break
    stack_pos: int | None = None
    for k, s in enumerate(segs):
        if k != expert_pos and _is_index(s):
            stack_pos = k
            break
    return segs, stack_pos, expert_pos


def _place_from(segs: list[str], stack_pos: int | None, expert_pos: int | None) -> TensorPlace:
    out = list(segs)
    expert_index = None
    if expert_pos is not None:
        expert_index = int(out[expert_pos])
        out[expert_pos] = EXPERT_MARK
    if stack_pos is None:
        return TensorPlace(".".join(out), None, None, expert_index)
    block_index = int(out[stack_pos])
    out[stack_pos] = STACK_MARK
    return TensorPlace(".".join(out), ".".join(out[:stack_pos]), block_index, expert_index)


def place(name: str) -> TensorPlace:
    segs, stack_pos, expert_pos = _split(name)
    return _place_from(segs, stack_pos, expert_pos)


def _signature(lines: Iterable[str]) -> str:
    return hashlib.sha1("\n".join(sorted(set(lines))).encode("utf-8")).hexdigest()[:12]


def analyze(tensors: Iterable[TensorInfo]) -> Structure:
    infos: list[TensorInfo] = []
    seen: set[str] = set()
    for t in tensors:
        if t.name in seen:
            raise DuplicateTensorError(t.name)
        seen.add(t.name)
        infos.append(t)

    # 1. raw placement
    raw: list[tuple[list[str], int | None, int | None]] = [_split(t.name) for t in infos]
    prefix_indices: dict[str, set[int]] = {}
    for segs, sp, _ in raw:
        if sp is not None:
            prefix_indices.setdefault(".".join(segs[:sp]), set()).add(int(segs[sp]))
    stack_prefixes = {p for p, idx in prefix_indices.items() if len(idx) >= 2}

    # 2. final placement (single-index prefixes keep their literal index)
    places: dict[str, TensorPlace] = {}
    final_pos: list[tuple[int | None, int | None]] = []  # (stack_pos, expert_pos) as finally applied
    for t, (segs, sp, ep) in zip(infos, raw):
        if sp is not None and ".".join(segs[:sp]) not in stack_prefixes:
            sp = None
        places[t.name] = _place_from(segs, sp, ep)
        final_pos.append((sp, ep))

    warnings: list[str] = []

    # 3. stacks
    blocks: dict[tuple[str, int], dict] = {}
    for t, (sp, ep) in zip(infos, final_pos):
        if sp is None:
            continue
        pl = places[t.name]
        b = blocks.setdefault((pl.stack_prefix, pl.block_index),
                              {"params": 0, "lines": [], "experts": set(), "tmpl": {}})
        b["params"] += t.numel
        csegs = pl.collapsed_name.split(".")
        rel = ".".join(csegs[sp + 1:])
        b["lines"].append(f"{rel}|{t.dtype}|{list(t.shape)}")
        if pl.expert_index is not None:
            b["experts"].add(pl.expert_index)
            if ep > sp:
                tmpl_rel = ".".join(csegs[sp + 1:ep + 1])
            else:  # expert container sits above the stack: use the absolute template
                tmpl_rel = ".".join(csegs[:ep + 1])
            b["tmpl"].setdefault(tmpl_rel, set()).add(pl.expert_index)

    stacks: list[Stack] = []
    for prefix in stack_prefixes:
        indices = sorted(prefix_indices[prefix])
        bp, sigs, moe, nexp = [], [], [], []
        for i in indices:
            b = blocks[(prefix, i)]
            lines = list(b["lines"])
            lines.extend(f"{tr}|experts={len(ix)}" for tr, ix in b["tmpl"].items())
            bp.append(b["params"])
            sigs.append(_signature(lines))
            is_moe = bool(b["experts"])
            moe.append(is_moe)
            nexp.append(len(b["experts"]) if is_moe else None)
        stacks.append(Stack(prefix, len(indices), indices, bp, sigs, moe, nexp))
    stacks.sort(key=lambda s: (-sum(s.block_params), s.prefix))

    # 4. expert groups
    groups: dict[str, dict] = {}
    for t, (_, ep) in zip(infos, final_pos):
        pl = places[t.name]
        if pl.expert_index is None:
            continue
        csegs = pl.collapsed_name.split(".")
        template = ".".join(csegs[:ep + 1])
        suffix = ".".join(csegs[ep + 1:])
        g = groups.setdefault(template, {"params": 0, "inst": {}, "suffixes": set()})
        g["params"] += t.numel
        g["suffixes"].add(suffix)
        inst = g["inst"].setdefault((pl.stack_prefix, pl.block_index), {})
        inst.setdefault(pl.expert_index, set()).add((suffix, t.dtype, tuple(t.shape)))

    expert_groups: list[ExpertGroup] = []
    for template in sorted(groups):
        g = groups[template]
        counts = [len(experts) for experts in g["inst"].values()]
        if len(set(counts)) > 1:
            warnings.append(f"{template}: expert count varies {sorted(set(counts))}")
        layouts = {frozenset(s) for experts in g["inst"].values() for s in experts.values()}
        homogeneous = len(layouts) == 1
        if not homogeneous:
            warnings.append(f"{template}: heterogeneous experts")
        total_experts = sum(counts)
        ppe, rem = divmod(g["params"], total_experts)
        if rem:
            homogeneous = False
            warnings.append(
                f"{template}: params {g['params']} not divisible by expert count {total_experts}")
        expert_groups.append(ExpertGroup(template, max(counts), len(counts), ppe,
                                         len(g["suffixes"]), homogeneous))

    # 5. fused-expert detection (warning only)
    fused: dict[str, int] = {}
    for t in infos:
        if len(t.shape) < 3:
            continue
        segs = t.name.split(".")
        for i, s in enumerate(segs):
            if s in EXPERT_CONTAINERS and not (i + 1 < len(segs) and _is_index(segs[i + 1])):
                csegs = places[t.name].collapsed_name.split(".")
                fused.setdefault(".".join(csegs[:i + 1]), t.shape[0])
                break
    for parent in sorted(fused):
        warnings.append(
            f"{parent}: fused expert tensors detected (leading expert dim {fused[parent]}); not collapsed in v0")

    # 6. totals
    by_dtype: dict[str, int] = {}
    for t in infos:
        by_dtype[t.dtype] = by_dtype.get(t.dtype, 0) + t.numel
    return Structure(
        stacks=stacks,
        expert_groups=expert_groups,
        places=places,
        params_total=sum(t.numel for t in infos),
        n_tensors=len(infos),
        params_by_dtype={k: by_dtype[k] for k in sorted(by_dtype)},
        warnings=warnings,
    )
