"""Phase 1 exit expectations: the common checks C1..C12 and the per-target tables E1..E3.

Everything here is a pure function over a TargetResult (plan.md section 2). No network, no
server. Every assertion is its own Check, and a Check never raises: an exception inside a
predicate becomes ok=False with actual="error: <Type>: <msg>".
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa


@dataclass(frozen=True)
class Check:
    target: str
    name: str
    expected: str
    actual: str
    ok: bool


@dataclass
class TargetResult:
    """Everything one exit target produced; filled by the exit-check runner (T015)."""

    target: str
    pin: str
    status: str
    elapsed_s: float
    final: dict                      # the last GET /api/scans/{id} response
    events: list[dict]               # all events (since=0) after completion
    cards: list[dict]
    tables: list[pa.Table]
    views: list[dict]
    json_paths: list[pathlib.Path]
    parquet_paths: list[pathlib.Path]
    wire: list[dict]                 # {method,url,orig_path,range,status,location,body_bytes}


HF_ENDPOINT_URL = "https://huggingface.co"
ALLOWED_HOSTS: tuple[str, ...] = ("huggingface.co", "*.hf.co")


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    """Exact match, or "*.x" matching any subdomain of x (not x itself). Case-insensitive."""
    h = (host or "").lower()
    for pat in allowed:
        p = pat.lower()
        if p.startswith("*."):
            suffix = p[1:]                     # ".x"
            if h.endswith(suffix) and len(h) > len(suffix):
                return True
        elif h == p:
            return True
    return False


# ---------------------------------------------------------------------------
# helpers


def _run(target: str, name: str, expected: Any, fn: Callable[[], tuple[Any, bool]]) -> Check:
    """Evaluate fn() -> (actual, ok); never raises."""
    exp = str(expected)
    try:
        actual, ok = fn()
        return Check(target, name, exp, str(actual), bool(ok))
    except Exception as e:  # noqa: BLE001 - a Check must never raise
        return Check(target, name, exp, f"error: {type(e).__name__}: {e}", False)


def _eq(target: str, name: str, expected: Any, getter: Callable[[], Any]) -> Check:
    def fn() -> tuple[Any, bool]:
        a = getter()
        return a, a == expected and type(a) is type(expected)
    return _run(target, name, expected, fn)


def cfg(card: dict, *keys: str) -> Any:
    """Nested lookup in card["config"]["raw"]; raises KeyError when anything is missing."""
    cur = card["config"]["raw"]
    if cur is None:
        raise KeyError("config.raw is null")
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(k)
        cur = cur[k]
    return cur


def _comp(card: Any) -> str:
    try:
        return card["key"]["component"] or "_model"
    except Exception:  # noqa: BLE001
        return "?"


def _card(r: TargetResult, component: str | None) -> dict:
    for c in r.cards:
        if c["key"]["component"] == component:
            return c
    raise KeyError(f"no card for component {component!r}")


def _view(r: TargetResult, component: str | None) -> dict:
    for v in r.views:
        if v["card_key"]["component"] == component:
            return v
    raise KeyError(f"no view for component {component!r}")


def _suffix_match(prefix: str, suffix: str) -> bool:
    return prefix == suffix or prefix.endswith("." + suffix)


def _stacks_with_suffix(card: dict, suffix: str) -> list[dict]:
    return [s for s in card["structure"]["stacks"] if _suffix_match(s["prefix"], suffix)]


def _stack_by_prefix(card: dict, prefix: str) -> dict:
    for s in card["structure"]["stacks"]:
        if s["prefix"] == prefix:
            return s
    raise KeyError(f"no stack with prefix {prefix!r}")


# ---------------------------------------------------------------------------
# common checks C1..C12


def common_checks(r: TargetResult, budget_s: float, *, allowed_hosts: Iterable[str] = ALLOWED_HOSTS,
                  expected_endpoint: str = HF_ENDPOINT_URL) -> list[Check]:
    t = r.target
    allowed = tuple(allowed_hosts)
    out: list[Check] = []
    add = out.append

    # C1
    add(_run(t, "C1 status done and elapsed < budget", f'status == "done" and elapsed_s < {budget_s}',
             lambda: (f"status={r.status!r} elapsed_s={r.elapsed_s}",
                      r.status == "done" and r.elapsed_s < budget_s)))

    # C2
    add(_eq(t, "C2 server totals.weight", 0, lambda: r.final["totals"]["weight"]))
    for i, card in enumerate(r.cards):
        add(_eq(t, f"C2 card[{i}] {_comp(card)} fetch_log.totals.weight", 0,
                lambda card=card: card["fetch_log"]["totals"]["weight"]))
    add(_eq(t, "C2 sum of event bytes_by_class.weight", 0,
            lambda: sum(e["bytes_by_class"]["weight"] for e in r.events)))

    # C3
    def c3() -> tuple[Any, bool]:
        got = sum(e["bytes_by_class"]["header"] for e in r.events if e["event"] == "fetch")
        want = sum(8 + f["header_len"] for c in r.cards for f in c["weights"]["files"])
        return f"{got} (sum of header_len+8 = {want})", got == want
    add(_run(t, "C3 header bytes == sum(8 + header_len)", "equal", c3))

    # C4
    n_files = 0
    for i, card in enumerate(r.cards):
        try:
            files = list(card["weights"]["files"])
        except Exception as e:  # noqa: BLE001
            add(Check(t, f"C4 card[{i}] {_comp(card)} files", "weights.files readable",
                      f"error: {type(e).__name__}: {e}", False))
            continue
        for j, f in enumerate(files):
            n_files += 1
            def c4(f=f) -> tuple[Any, bool]:
                total = 8 + f["header_len"] + f["data_bytes"]
                return total, total == f["size_bytes"]
            try:
                label = f["path"]
                want = f["size_bytes"]
            except Exception:  # noqa: BLE001
                label, want = f"file[{j}]", "size_bytes"
            add(_run(t, f"C4 {_comp(card)} {label}: 8 + header_len + data_bytes == size_bytes", want, c4))
    add(_run(t, "C4 at least one weight file", ">= 1", lambda: (n_files, n_files >= 1)))

    # C5
    for i, card in enumerate(r.cards):
        try:
            has_index = bool(card["weights"]["index_path"])
        except Exception as e:  # noqa: BLE001
            add(Check(t, f"C5 card[{i}] {_comp(card)} index_path", "readable",
                      f"error: {type(e).__name__}: {e}", False))
            continue
        if has_index:
            add(_eq(t, f"C5 card[{i}] {_comp(card)} tensor_bytes_total == index_total_size",
                    card["weights"]["index_total_size"], lambda card=card: card["weights"]["tensor_bytes_total"]))

    # C6 (sanity)
    def c6_counts() -> tuple[Any, bool]:
        ns = (len(r.cards), len(r.tables), len(r.views), len(r.json_paths), len(r.parquet_paths))
        return f"cards/tables/views/json/parquet = {ns}", len(set(ns)) == 1
    add(_run(t, "C6 cards, tables, views and paths have equal counts", "equal", c6_counts))

    def c6_exist() -> tuple[Any, bool]:
        paths = [pathlib.Path(p) for p in list(r.json_paths) + list(r.parquet_paths)]
        missing = [str(p) for p in paths if not p.exists()]
        return f"missing={missing} (checked {len(paths)})", not missing and len(paths) > 0
    add(_run(t, "C6 card json and parquet files exist", "all exist", c6_exist))
    for i, card in enumerate(r.cards):
        add(_eq(t, f"C6 card[{i}] {_comp(card)} key.revision_sha == pin", r.pin,
                lambda card=card: card["key"]["revision_sha"]))

    # C7
    for i, card in enumerate(r.cards):
        lab = f"card[{i}] {_comp(card)}"
        add(_eq(t, f"C7 {lab} parquet rows == n_tensors", card.get("weights", {}).get("n_tensors"),
                lambda i=i: r.tables[i].num_rows))

        def c7_stats(i=i) -> tuple[Any, bool]:
            table = r.tables[i]
            cols = [n for n in table.column_names if n.startswith("stat_")]
            bad = {n: table.column(n).null_count for n in cols if table.column(n).null_count != table.num_rows}
            return f"stat columns={len(cols)} not-all-null={bad}", bool(cols) and not bad
        add(_run(t, f"C7 {lab} every stat_* column is 100% null", "all null, >= 1 column", c7_stats))

        def c7_json(card=card) -> tuple[Any, bool]:
            vals = card["stats"]
            bad = [k for k, v in vals.items() if v is not None]
            return f"non-null stats={bad}", not bad
        add(_run(t, f"C7 {lab} JSON stats all null", "all None", c7_json))

    # C8
    for i, v in enumerate(r.views):
        def c8(v=v) -> tuple[Any, bool]:
            n = sum(1 for node in v["nodes"] if node["kind"] != "tensor")
            return n, n <= 100
        add(_run(t, f"C8 view[{i}] non-tensor nodes <= 100", "<= 100", c8))
    if not r.views:
        add(Check(t, "C8 at least one view", ">= 1", "0", False))

    # C9 (sanity)
    for i, card in enumerate(r.cards):
        add(_eq(t, f"C9 card[{i}] {_comp(card)} params_total == sum(numel)", card.get("weights", {}).get("params_total"),
                lambda i=i: sum(r.tables[i].column("numel").to_pylist())))

    # C10
    def header_len_for(orig_path: str) -> int:
        best: tuple[int, int] | None = None   # (len of matched path, header_len)
        for c in r.cards:
            for f in c["weights"]["files"]:
                p = f["path"]
                if orig_path == p or orig_path.endswith("/" + p):
                    if best is None or len(p) > best[0]:
                        best = (len(p), f["header_len"])
        if best is None:
            raise KeyError(f"no weight file in the cards matches {orig_path!r}")
        return best[1]

    n_st = 0
    for w in r.wire:
        try:
            is_st = str(w["orig_path"]).endswith(".safetensors")
        except Exception:  # noqa: BLE001
            is_st = False
        if not is_st:
            continue
        n_st += 1
        def c10(w=w) -> tuple[Any, bool]:
            rng = w["range"]
            if rng is None:
                return "no Range header", False
            m = re.fullmatch(r"bytes=(\d+)-(\d+)", str(rng))
            if not m:
                return f"unparseable Range {rng!r}", False
            b = int(m.group(2))
            limit = 8 + header_len_for(w["orig_path"])
            return f"{rng} (b+1={b + 1}, 8+header_len={limit})", b + 1 <= limit
        add(_run(t, f"C10 {w.get('orig_path')} Range within header", "Range bytes=a-b with b+1 <= 8+header_len", c10))
    add(_run(t, "C10 at least one .safetensors wire record", ">= 1", lambda: (n_st, n_st >= 1)))

    # C11
    def c11() -> tuple[Any, bool]:
        wire_bytes = sum(w["body_bytes"] for w in r.wire)
        tot = r.final["totals"]
        logged = tot["meta"] + tot["header"] + tot["weight"]
        return f"wire={wire_bytes} bytelog={logged}", wire_bytes == logged
    add(_run(t, "C11 wire body bytes == ByteLog meta+header+weight", "equal", c11))

    # C12
    add(_run(t, "C12 at least one wire record", ">= 1", lambda: (len(r.wire), len(r.wire) >= 1)))

    def c12_hosts() -> tuple[Any, bool]:
        hosts = {(urlsplit(w["url"]).hostname or "").lower() for w in r.wire}
        bad = sorted(h for h in hosts if not host_allowed(h, allowed))
        return f"hosts={sorted(hosts)} offending={bad}", not bad
    add(_run(t, "C12 every wire host allowed", f"hosts in {list(allowed)}", c12_hosts))

    def c12_endpoint() -> tuple[Any, bool]:
        eps = [c["source"]["endpoint"] for c in r.cards]
        return f"endpoints={sorted(set(map(str, eps)))} cards={len(eps)}", bool(eps) and all(e == expected_endpoint for e in eps)
    add(_run(t, "C12 every card source.endpoint", expected_endpoint, c12_endpoint))

    return out


# ---------------------------------------------------------------------------
# E1..E3


def expect_qwen3_8b(r: TargetResult) -> list[Check]:
    t = r.target
    c = lambda: _card(r, None)  # noqa: E731
    out = [
        _eq(t, "E1 number of cards", 1, lambda: len(r.cards)),
        _eq(t, "E1 card component", None, lambda: c()["key"]["component"]),
        _eq(t, "E1 config.model_type", "qwen3", lambda: c()["config"]["model_type"]),
        _eq(t, "E1 config.architectures", ["Qwen3ForCausalLM"], lambda: c()["config"]["architectures"]),
        _eq(t, "E1 n_tensors", 399, lambda: c()["weights"]["n_tensors"]),
        _eq(t, "E1 params_total", 8190735360, lambda: c()["weights"]["params_total"]),
        _eq(t, "E1 params_by_dtype", {"BF16": 8190735360}, lambda: c()["weights"]["params_by_dtype"]),
        _eq(t, "E1 stacks[0].prefix", "model.layers", lambda: c()["structure"]["stacks"][0]["prefix"]),
        _eq(t, "E1 stacks[0].depth", 36, lambda: c()["structure"]["stacks"][0]["depth"]),
        _eq(t, "E1 stacks[0].depth == config.num_hidden_layers", cfg_value(r, None, "num_hidden_layers"),
            lambda: c()["structure"]["stacks"][0]["depth"]),
        _eq(t, "E1 distinct block_signatures in model.layers", 1,
            lambda: len(set(_stack_by_prefix(c(), "model.layers")["block_signatures"]))),
        _eq(t, "E1 expert_groups", [], lambda: c()["structure"]["expert_groups"]),
        _eq(t, "E1 model_card.present", True, lambda: c()["model_card"]["present"]),
        _eq(t, "E1 model_card.license", "apache-2.0", lambda: c()["model_card"]["license"]),
        _eq(t, "E1 model_card.base_model", ["Qwen/Qwen3-8B-Base"], lambda: c()["model_card"]["base_model"]),
    ]
    return out


def cfg_value(r: TargetResult, component: str | None, *keys: str) -> Any:
    """config.raw lookup that returns an error marker instead of raising (used as an `expected`)."""
    try:
        return cfg(_card(r, component), *keys)
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {type(e).__name__}: {e}>"


def expect_qwen3_30b_a3b(r: TargetResult) -> list[Check]:
    t = r.target
    c = lambda: _card(r, None)  # noqa: E731
    eg = lambda: c()["structure"]["expert_groups"][0]  # noqa: E731
    strip = lambda: next(s for s in _view(r, None)["depth_strips"] if s["prefix"] == "model.layers")  # noqa: E731

    def nonexp_cfg(*keys: str) -> Any:
        return cfg_value(r, None, *keys)

    def hidden_x_moe() -> Any:
        try:
            return 3 * cfg(c(), "hidden_size") * cfg(c(), "moe_intermediate_size")
        except Exception as e:  # noqa: BLE001
            return f"<unavailable: {type(e).__name__}: {e}>"

    def expert_nodes() -> list[dict]:
        return [n for n in _view(r, None)["nodes"] if n["kind"] == "expert_group"]

    return [
        _eq(t, "E2 number of cards", 1, lambda: len(r.cards)),
        _eq(t, "E2 config.model_type", "qwen3_moe", lambda: c()["config"]["model_type"]),
        _eq(t, "E2 n_tensors", 18867, lambda: c()["weights"]["n_tensors"]),
        _eq(t, "E2 params_total", 30532122624, lambda: c()["weights"]["params_total"]),
        _eq(t, "E2 stacks[0].prefix", "model.layers", lambda: c()["structure"]["stacks"][0]["prefix"]),
        _eq(t, "E2 stacks[0].depth", 48, lambda: c()["structure"]["stacks"][0]["depth"]),
        _eq(t, "E2 stacks[0].depth == config.num_hidden_layers", nonexp_cfg("num_hidden_layers"),
            lambda: c()["structure"]["stacks"][0]["depth"]),
        _eq(t, "E2 number of expert_groups", 1, lambda: len(c()["structure"]["expert_groups"])),
        _eq(t, "E2 expert_group.template", "model.layers.#.mlp.experts.*", lambda: eg()["template"]),
        _eq(t, "E2 expert_group.n_experts", 128, lambda: eg()["n_experts"]),
        _eq(t, "E2 expert_group.n_experts == config.num_experts", nonexp_cfg("num_experts"), lambda: eg()["n_experts"]),
        _eq(t, "E2 expert_group.n_instances", 48, lambda: eg()["n_instances"]),
        _eq(t, "E2 expert_group.tensors_per_expert", 3, lambda: eg()["tensors_per_expert"]),
        _eq(t, "E2 expert_group.params_per_expert", 4718592, lambda: eg()["params_per_expert"]),
        _eq(t, "E2 expert_group.params_per_expert == 3 x hidden_size x moe_intermediate_size", hidden_x_moe(),
            lambda: eg()["params_per_expert"]),
        _eq(t, "E2 expert_group.homogeneous", True, lambda: eg()["homogeneous"]),
        _eq(t, "E2 depth strip cells", 48, lambda: len(strip()["cells"])),
        _eq(t, "E2 depth strip all moe", True, lambda: all(x["moe"] is True for x in strip()["cells"]) and bool(strip()["cells"])),
        _eq(t, "E2 depth strip all n_experts == 128", True,
            lambda: all(x["n_experts"] == 128 for x in strip()["cells"]) and bool(strip()["cells"])),
        _eq(t, "E2 depth strip distinct signatures", 1, lambda: len({x["signature"] for x in strip()["cells"]})),
        _eq(t, "E2 view expert_group nodes", 1, lambda: len(expert_nodes())),
        _eq(t, "E2 view expert_group node count", 128, lambda: expert_nodes()[0]["count"]),
        _eq(t, "E2 model_card.license", "apache-2.0", lambda: c()["model_card"]["license"]),
        _eq(t, "E2 model_card.base_model", ["Qwen/Qwen3-30B-A3B-Base"], lambda: c()["model_card"]["base_model"]),
    ]


def _suffix_checks(t: str, prefix: str, card_fn: Callable[[], dict], suffix: str,
                   expected_depth_fn: Callable[[], Any]) -> list[Check]:
    """Two Checks: a stack with the suffix exists, and one of them has the expected depth."""
    def exists() -> tuple[Any, bool]:
        prefixes = [s["prefix"] for s in _stacks_with_suffix(card_fn(), suffix)]
        return prefixes, bool(prefixes)

    def depth() -> tuple[Any, bool]:
        want = expected_depth_fn()
        depths = {s["prefix"]: s["depth"] for s in _stacks_with_suffix(card_fn(), suffix)}
        return depths, any(d == want for d in depths.values())

    try:
        want_s = expected_depth_fn()
    except Exception as e:  # noqa: BLE001
        want_s = f"<unavailable: {type(e).__name__}: {e}>"
    return [
        _run(t, f"{prefix} a stack with prefix suffix {suffix!r} exists", f"some stack ending in {suffix!r}", exists),
        _run(t, f"{prefix} stack ending in {suffix!r} has expected depth", f"depth == {want_s}", depth),
    ]


def expect_qwen_image(r: TargetResult) -> list[Check]:
    t = r.target
    te = lambda: _card(r, "text_encoder")  # noqa: E731
    tr = lambda: _card(r, "transformer")  # noqa: E731
    vae = lambda: _card(r, "vae")  # noqa: E731

    def pipeline_components() -> dict:
        for card in r.cards:
            if card.get("pipeline"):
                return {c["name"]: c["has_weights"] for c in card["pipeline"]["components"]}
        raise KeyError("no card has a pipeline")

    def pipeline_classes() -> set:
        return {card["pipeline"]["pipeline_class"] for card in r.cards}

    def text_depth() -> Any:
        raw = te()["config"]["raw"]
        if raw is None:
            raise KeyError("config.raw is null")
        tc = raw.get("text_config")
        if isinstance(tc, dict) and "num_hidden_layers" in tc:
            return tc["num_hidden_layers"]
        return raw["num_hidden_layers"]

    out: list[Check] = [
        _eq(t, "E3 number of cards", 3, lambda: len(r.cards)),
        _eq(t, "E3 components", {"text_encoder", "transformer", "vae"}, lambda: {c["key"]["component"] for c in r.cards}),
        _eq(t, "E3 pipeline.pipeline_class", {"QwenImagePipeline"}, pipeline_classes),
    ]
    for name, want in (("scheduler", False), ("tokenizer", False), ("text_encoder", True),
                       ("transformer", True), ("vae", True)):
        out.append(_eq(t, f"E3 pipeline.components {name}.has_weights", want, lambda name=name: pipeline_components()[name]))

    out.append(_run(t, "E3 text_encoder architectures contain Qwen2_5_VLForConditionalGeneration",
                    "contains", lambda: (te()["config"]["architectures"],
                                         "Qwen2_5_VLForConditionalGeneration" in te()["config"]["architectures"])))
    out += _suffix_checks(t, "E3 text_encoder", te, "layers", text_depth)
    out += _suffix_checks(t, "E3 text_encoder", te, "blocks", lambda: cfg(te(), "vision_config", "depth"))
    out.append(_eq(t, "E3 text_encoder expert_groups", [], lambda: te()["structure"]["expert_groups"]))

    out.append(_eq(t, "E3 transformer config.class_name", "QwenImageTransformer2DModel",
                   lambda: tr()["config"]["class_name"]))
    out += _suffix_checks(t, "E3 transformer", tr, "transformer_blocks", lambda: cfg(tr(), "num_layers"))

    out += [
        _eq(t, "E3 vae config.class_name", "AutoencoderKLQwenImage", lambda: vae()["config"]["class_name"]),
        _eq(t, "E3 vae number of weight files", 1, lambda: len(vae()["weights"]["files"])),
        _eq(t, "E3 vae index_path", None, lambda: vae()["weights"]["index_path"]),
        _eq(t, "E3 all cards same revision_sha", 1, lambda: len({c["key"]["revision_sha"] for c in r.cards})),
        _eq(t, "E3 all cards model_card.license", {"apache-2.0"}, lambda: {c["model_card"]["license"] for c in r.cards}),
    ]
    return out


DEFAULT_EXPECTATIONS: dict[str, Callable[[TargetResult], list[Check]]] = {
    "Qwen/Qwen3-8B": expect_qwen3_8b,
    "Qwen/Qwen3-30B-A3B": expect_qwen3_30b_a3b,
    "Qwen/Qwen-Image": expect_qwen_image,
}
