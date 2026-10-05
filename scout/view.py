"""View model: collapsed graph nodes and depth strips built from a persisted Card (pure, no IO).

The view is served to the frontend and never persisted (plan.md section 4.4).
"""

from __future__ import annotations

import copy

import pyarrow as pa

from scout.errors import ScoutError
from scout.structure import _is_index  # same index rule as the Card's structure analysis

DISCLAIMERS: tuple[str, str, str] = (
    "Distillation is invisible to weight forensics: a model trained on another model's outputs leaves no trace in its weights.",
    "Tokenizer reuse alone is not proof of derivation.",
    "Licensing is a human decision: scout shows the license claimed by the model card and does not judge compliance.",
)

_STACK = "[#]"
_EXPERT = "[*]"
_MARK_LEN = len(_STACK)

# A marker origin: ("card", stack_prefix) for "#", ("lit", index) for a literal digit,
# ("exp", template) for "*".
Origin = tuple[str, str | None]


def _segments(collapsed_name: str, stack_prefix: str | None) -> list[tuple[str, list[Origin]]]:
    """Apply the segment rule. Each merged segment carries one origin per marker, in order.

    A marker with no predecessor (a root-level stack) merges into an empty segment, so
    "#.weight" -> ["[#]", "weight"]."""
    orig = collapsed_name.split(".")
    out: list[tuple[str, list[Origin]]] = []
    for k, seg in enumerate(orig):
        if seg == "#":
            mark, info = _STACK, ("card", stack_prefix)
        elif seg == "*":
            mark, info = _EXPERT, ("exp", ".".join(orig[: k + 1]))
        elif _is_index(seg):
            mark, info = _STACK, ("lit", seg.lstrip("0") or "0")
        else:
            out.append((seg, []))
            continue
        if not out:
            out.append(("", []))
        prev, markers = out[-1]
        out[-1] = (prev + mark, markers + [info])
    return out


class _Marker:
    """What one marker position of one node has seen across rows."""

    __slots__ = ("kind", "fixed", "literals")

    def __init__(self, kind: str) -> None:
        self.kind = kind                 # "stack" | "expert_group" (from the first origin seen)
        self.fixed: int | None = None    # Card stack depth / expert group n_experts
        self.literals: set[str] = set()  # distinct literal indices (inner stack)

    def count(self) -> int:
        if self.fixed is not None:       # a Card count is never overridden by a literal
            return self.fixed
        return len(self.literals) or 1


def build_view(card: dict, table: pa.Table) -> dict:
    cols = {n: table.column(n).to_pylist()
            for n in ("collapsed_name", "numel", "shape", "dtype", "stack_prefix")}
    structure = card["structure"]
    stack_depth = {s["prefix"]: s["depth"] for s in structure["stacks"]}
    group_experts = {g["template"]: g["n_experts"] for g in structure["expert_groups"]}

    def resolve(info: Origin, name: str) -> int:
        what, ref = info
        if what == "card":
            if ref not in stack_depth:
                raise ScoutError(
                    f"tensor table row {name!r}: stack_prefix {ref!r} has no matching Card stack")
            return stack_depth[ref]
        if ref not in group_experts:
            raise ScoutError(
                f"tensor table row {name!r}: expert template {ref!r} has no matching Card expert group")
        return group_experts[ref]

    key = card["key"]
    nodes: dict[str, dict] = {
        "": {"id": "", "label": key.get("component") or key["repo"], "kind": "root", "parent": None,
             "params": 0, "count": 1, "shape": None, "dtype": None},
    }
    markers: dict[str, list[_Marker]] = {}   # id -> one entry per marker of its last segment
    base_seg: dict[str, str] = {}            # id -> last segment without its markers
    leaf_rows: dict[str, list[tuple[list[int], str]]] = {}
    strict_prefix: set[str] = set()

    for i in range(len(cols["collapsed_name"])):
        numel = cols["numel"][i]
        name = cols["collapsed_name"][i]
        segs = _segments(name, cols["stack_prefix"][i])
        nodes[""]["params"] += numel
        parent = ""
        for L in range(1, len(segs) + 1):
            seg, infos = segs[L - 1]
            nid = ".".join(s for s, _ in segs[:L])
            if L > 1:
                strict_prefix.add(parent)
            node = nodes.get(nid)
            if node is None:
                node = {"id": nid, "label": "", "kind": "module", "parent": parent, "params": 0,
                        "count": 1, "shape": None, "dtype": None}
                nodes[nid] = node
                base_seg[nid] = seg[: len(seg) - _MARK_LEN * len(infos)]
                markers[nid] = [_Marker("expert_group" if what == "exp" else "stack")
                                for what, _ in infos]
            node["params"] += numel
            for m, info in zip(markers[nid], infos):
                if info[0] == "lit":
                    m.literals.add(info[1])
                else:
                    c = resolve(info, name)
                    if m.fixed is not None and m.fixed != c:
                        raise ScoutError(
                            f"node {nid!r}: conflicting Card counts {m.fixed} and {c} (row {name!r})")
                    m.fixed = c
            parent = nid
        leaf_rows.setdefault(parent, []).append((cols["shape"][i], cols["dtype"][i]))

    for nid, node in nodes.items():
        if nid == "":
            continue
        ms = markers[nid]
        if ms:
            node["kind"] = ms[0].kind
            node["count"] = ms[0].count()
        elif nid in leaf_rows and nid not in strict_prefix:
            node["kind"] = "tensor"
        parts = [base_seg[nid]] if base_seg[nid] else []
        parts.extend(f"×{m.count()}" for m in ms)
        node["label"] = " ".join(parts)
        if node["kind"] == "tensor":
            rows = leaf_rows[nid]
            first_shape, first_dtype = rows[0]
            same_shape = all(s == first_shape for s, _ in rows)
            same_dtype = all(d == first_dtype for _, d in rows)
            node["shape"] = list(first_shape) if same_shape else None
            node["dtype"] = first_dtype if same_dtype else "varies"

    strips = [
        {
            "prefix": s["prefix"],
            "depth": s["depth"],
            "cells": [
                {"index": s["indices"][j], "params": s["block_params"][j],
                 "signature": s["block_signatures"][j], "moe": s["block_moe"][j],
                 "n_experts": s["block_n_experts"][j]}
                for j in range(len(s["indices"]))
            ],
        }
        for s in structure["stacks"]
    ]

    totals = card["fetch_log"]["totals"]
    title = f"{key['repo']}@{key['revision_sha'][:12]}"
    if key.get("component"):
        title += f" / {key['component']}"
    return {
        "card_key": copy.deepcopy(key),
        "title": title,
        "model_card": copy.deepcopy(card["model_card"]),
        "nodes": list(nodes.values()),
        "depth_strips": strips,
        "summary": {
            "params_total": card["weights"]["params_total"],
            "n_tensors": card["weights"]["n_tensors"],
            "n_stacks": len(structure["stacks"]),
            "n_expert_groups": len(structure["expert_groups"]),
            "weight_bytes_read": totals["weight"],
            "header_bytes_read": totals["header"],
            "meta_bytes_read": totals["meta"],
        },
        "disclaimers": list(DISCLAIMERS),
    }
