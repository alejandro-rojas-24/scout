"""View model: collapsed graph nodes and depth strips built from a persisted Card (pure, no IO).

The view is served to the frontend and never persisted (plan.md section 4.4).
"""

from __future__ import annotations

import copy

import pyarrow as pa

DISCLAIMERS: tuple[str, str, str] = (
    "Distillation is invisible to weight forensics: a model trained on another model's outputs leaves no trace in its weights.",
    "Tokenizer reuse alone is not proof of derivation.",
    "Licensing is a human decision: scout shows the license claimed by the model card and does not judge compliance.",
)

_STACK = "[#]"
_EXPERT = "[*]"


def _is_digits(seg: str) -> bool:
    return seg.isascii() and seg.isdigit()


def _segments(collapsed_name: str, stack_prefix: str | None) -> list[tuple[str, tuple | None]]:
    """Apply the segment rule. Each segment carries its origin:
    ("card", stack_prefix) for "#", ("lit", index) for a literal digit, ("exp", template) for "*"."""
    orig = collapsed_name.split(".")
    out: list[tuple[str, tuple | None]] = []
    for k, seg in enumerate(orig):
        if seg == "#":
            mark, info = _STACK, ("card", stack_prefix)
        elif seg == "*":
            mark, info = _EXPERT, ("exp", ".".join(orig[: k + 1]))
        elif _is_digits(seg):
            mark, info = _STACK, ("lit", seg.lstrip("0") or "0")
        else:
            out.append((seg, None))
            continue
        if not out:  # nothing to merge into: keep the segment as it is
            out.append((seg, None))
            continue
        prev, _ = out[-1]
        out[-1] = (prev + mark, info)
    return out


def build_view(card: dict, table: pa.Table) -> dict:
    cols = {n: table.column(n).to_pylist()
            for n in ("collapsed_name", "numel", "shape", "dtype", "stack_prefix")}
    structure = card["structure"]
    stack_depth = {s["prefix"]: s["depth"] for s in structure["stacks"]}
    group_experts = {g["template"]: g["n_experts"] for g in structure["expert_groups"]}

    key = card["key"]
    nodes: dict[str, dict] = {
        "": {"id": "", "label": key.get("component") or key["repo"], "kind": "root", "parent": None,
             "params": 0, "count": 1, "shape": None, "dtype": None},
    }
    card_count: dict[str, int] = {}          # id -> count from a Card stack / expert group
    literals: dict[str, set[str]] = {}       # id -> distinct literal indices (inner stacks)
    last_seg: dict[str, str] = {}
    leaf_rows: dict[str, list[tuple[list[int], str]]] = {}
    strict_prefix: set[str] = set()

    for i in range(len(cols["collapsed_name"])):
        numel = cols["numel"][i]
        segs = _segments(cols["collapsed_name"][i], cols["stack_prefix"][i])
        nodes[""]["params"] += numel
        parent = ""
        for L in range(1, len(segs) + 1):
            nid = ".".join(s for s, _ in segs[:L])
            if L > 1:
                strict_prefix.add(parent)
            node = nodes.get(nid)
            if node is None:
                node = {"id": nid, "label": "", "kind": "module", "parent": parent, "params": 0,
                        "count": 1, "shape": None, "dtype": None}
                nodes[nid] = node
                last_seg[nid] = segs[L - 1][0]
            node["params"] += numel
            info = segs[L - 1][1]
            if info is not None:
                if info[0] == "card":
                    card_count[nid] = stack_depth.get(info[1], 1)
                elif info[0] == "lit":
                    literals.setdefault(nid, set()).add(info[1])
                else:
                    card_count[nid] = group_experts.get(info[1], 1)
            parent = nid
        leaf_rows.setdefault(parent, []).append((cols["shape"][i], cols["dtype"][i]))

    for nid, node in nodes.items():
        if nid == "":
            continue
        last = last_seg[nid]
        if last.endswith(_STACK):
            node["kind"] = "stack"
        elif last.endswith(_EXPERT):
            node["kind"] = "expert_group"
        elif nid in leaf_rows and nid not in strict_prefix:
            node["kind"] = "tensor"
        if nid in card_count:
            node["count"] = card_count[nid]
        elif nid in literals:
            node["count"] = len(literals[nid])
        node["label"] = last.replace(_STACK, f" ×{node['count']}").replace(
            _EXPERT, f" ×{node['count']}")
        if node["kind"] == "tensor":
            rows = leaf_rows[nid]
            first_shape, first_dtype = rows[0]
            if any(s != first_shape for s, _ in rows):
                node["shape"], node["dtype"] = None, "varies"
            else:
                node["shape"], node["dtype"] = list(first_shape), first_dtype

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
