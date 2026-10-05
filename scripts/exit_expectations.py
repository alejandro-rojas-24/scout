"""Phase 1 exit expectations: the common checks C1..C12 and the per-target tables E1..E3.

Everything here is a pure function over a TargetResult (plan.md section 2). No network, no
server. Every assertion is its own Check, and a Check never raises: an exception inside a
predicate becomes ok=False with actual="error: <Type>: <msg>".

These checks are the phase's independent evidence, so they deliberately do not import the
code under test (scout.*): slugs, path layout and the resolve URL shape are re-derived here
from plan.md section 3.3.
"""

from __future__ import annotations

import math
import os
import pathlib
import posixpath
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

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


def _peek(fn: Callable[[], Any]) -> Any:
    """Value of fn(), or an "<unavailable: ...>" marker (never matches a real expected value)."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {type(e).__name__}: {e}>"


def _comp(card: Any) -> str:
    try:
        return card["key"]["component"] or "_model"
    except Exception:  # noqa: BLE001
        return "?"


def _list_of(r: TargetResult, attr: str) -> list:
    """r.<attr> if it is a list, else [] (never raises; the type itself is checked by _typed)."""
    v = getattr(r, attr, None)
    return v if isinstance(v, list) else []


def _cards(r: TargetResult) -> list[dict]:
    return [c for c in _list_of(r, "cards") if isinstance(c, dict)]


def _card(r: TargetResult, component: str | None) -> dict:
    for c in _cards(r):
        if c["key"]["component"] == component:
            return c
    raise KeyError(f"no card for component {component!r}")


def _view(r: TargetResult, component: str | None) -> dict:
    for v in _list_of(r, "views"):
        if isinstance(v, dict) and v["card_key"]["component"] == component:
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


def _repo_slug(repo: str) -> str:
    """plan.md 3.3 repo_slug for Hub repos (the exit targets are all Hub repos)."""
    return repo.replace("/", "__")


def _component_slug(component: str | None) -> str:
    return "_model" if component is None else component


_JSON_SUFFIX = ".card.json"
_PARQUET_SUFFIX = ".tensors.parquet"


def _split_card_path(p: Any, suffix: str) -> tuple[str, str, str, str] | None:
    """(repo_slug dir, sha dir, stem, parent str) if p ends with suffix, else None."""
    path = pathlib.Path(p)
    if not path.name.endswith(suffix) or len(path.name) == len(suffix):
        return None
    return path.parent.parent.name, path.parent.name, path.name[: -len(suffix)], str(path.parent)


def _table_meta(table: Any) -> dict[str, str]:
    md = table.schema.metadata or {}
    return {k.decode("utf-8", "replace"): v.decode("utf-8", "replace") for k, v in md.items()}


def _table_for(r: TargetResult, card: dict) -> pa.Table:
    """The one table whose Parquet schema metadata names this card's component and revision_sha."""
    comp = card["key"]["component"]
    want = ("" if comp is None else comp, card["key"]["revision_sha"])
    hits = []
    for t in _list_of(r, "tables"):
        if not isinstance(t, pa.Table):
            continue
        md = _table_meta(t)
        if (md.get("component"), md.get("revision_sha")) == want:
            hits.append(t)
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} tables with metadata component={want[0]!r} revision_sha={want[1]!r}")
    return hits[0]


def _norm_orig(orig: Any) -> str:
    """unquote(urlsplit(orig_path).path): drops scheme/host/query/fragment, undoes percent-encoding."""
    if not isinstance(orig, str):
        raise TypeError(f"orig_path is {type(orig).__name__}, not str")
    return unquote(urlsplit(orig).path)


def _ext_key(path: str) -> tuple[str, str]:
    """Match key: the path is compared exactly except its extension, which is compared case-insensitively."""
    base, ext = posixpath.splitext(path)
    return base, ext.lower()


def _is_weight_name(path: str) -> bool:
    return path.lower().endswith(".safetensors")


_RANGE_RE = re.compile(r"bytes=(\d+)-(\d+)", re.IGNORECASE)


def _is_2xx(status: Any) -> bool:
    """Fail closed: anything that is not a readable int outside 200..299 counts as 2xx."""
    if isinstance(status, int) and not isinstance(status, bool):
        return 200 <= status < 300
    return True


# ---------------------------------------------------------------------------
# common checks C1..C12


def _typed(t: str, out: list[Check], r: TargetResult, attr: str, elem: tuple[type, ...] | None,
           container: type = list) -> None:
    """Up-front type Check for one TargetResult field (and its elements)."""
    def fn() -> tuple[Any, bool]:
        v = getattr(r, attr, None)
        if not isinstance(v, container):
            return f"{type(v).__name__}", False
        if elem is None:
            return container.__name__, True
        bad = [i for i, x in enumerate(v) if not isinstance(x, elem)]
        return f"{container.__name__} of {len(v)}, wrong element type at {bad}", not bad
    want = container.__name__ + ("" if elem is None else " of " + "|".join(e.__name__ for e in elem))
    out.append(_run(t, f"TYPE field {attr} has the right type", want, fn))


def common_checks(r: TargetResult, budget_s: float, *, allowed_hosts: Iterable[str] = ALLOWED_HOSTS,
                  expected_endpoint: str = HF_ENDPOINT_URL) -> list[Check]:
    t = r.target
    allowed = tuple(allowed_hosts)
    out: list[Check] = []
    add = out.append

    # field types first; afterwards every list field is coerced (non-list -> [], bad elements dropped)
    _typed(t, out, r, "final", None, dict)
    for attr, elem in (("events", (dict,)), ("cards", (dict,)), ("tables", (pa.Table,)), ("views", (dict,)),
                       ("json_paths", (str, os.PathLike)), ("parquet_paths", (str, os.PathLike)),
                       ("wire", (dict,))):
        _typed(t, out, r, attr, elem)
    events = [e for e in _list_of(r, "events") if isinstance(e, dict)]
    cards = _cards(r)
    views = [v for v in _list_of(r, "views") if isinstance(v, dict)]
    json_paths = [p for p in _list_of(r, "json_paths") if isinstance(p, (str, os.PathLike))]
    parquet_paths = [p for p in _list_of(r, "parquet_paths") if isinstance(p, (str, os.PathLike))]
    wire = [w for w in _list_of(r, "wire") if isinstance(w, dict)]
    n_tables = len(_list_of(r, "tables"))

    # C1
    def c1() -> tuple[Any, bool]:
        e = r.elapsed_s
        num = isinstance(e, (int, float)) and not isinstance(e, bool)
        ok = r.status == "done" and num and math.isfinite(e) and 0 < e < budget_s
        return f"status={r.status!r} elapsed_s={e!r}", ok
    add(_run(t, "C1 status done and 0 < elapsed < budget", f'status == "done" and 0 < elapsed_s < {budget_s}', c1))

    # C2
    add(_eq(t, "C2 server totals.weight", 0, lambda: r.final["totals"]["weight"]))
    for i, card in enumerate(cards):
        add(_eq(t, f"C2 card[{i}] {_comp(card)} fetch_log.totals.weight", 0,
                lambda card=card: card["fetch_log"]["totals"]["weight"]))
    add(_eq(t, "C2 sum of event bytes_by_class.weight", 0,
            lambda: sum(e["bytes_by_class"]["weight"] for e in events)))

    # C3
    def c3() -> tuple[Any, bool]:
        got = sum(e["bytes_by_class"]["header"] for e in events if e["event"] == "fetch")
        want = sum(8 + f["header_len"] for c in cards for f in c["weights"]["files"])
        return f"{got} (sum of header_len+8 = {want})", got == want
    add(_run(t, "C3 header bytes == sum(8 + header_len)", "equal", c3))

    # C4
    n_files = 0
    for i, card in enumerate(cards):
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
    for i, card in enumerate(cards):
        try:
            has_index = bool(card["weights"]["index_path"])
        except Exception as e:  # noqa: BLE001
            add(Check(t, f"C5 card[{i}] {_comp(card)} index_path", "readable",
                      f"error: {type(e).__name__}: {e}", False))
            continue
        if has_index:
            add(_eq(t, f"C5 card[{i}] {_comp(card)} tensor_bytes_total == index_total_size",
                    _peek(lambda card=card: card["weights"]["index_total_size"]),
                    lambda card=card: card["weights"]["tensor_bytes_total"]))

    # C6 (sanity): counts, existence, <out>/<repo_slug>/<sha>/<component_slug>.{card.json,tensors.parquet}
    def c6_counts() -> tuple[Any, bool]:
        ns = (len(_list_of(r, "cards")), n_tables, len(_list_of(r, "views")),
              len(_list_of(r, "json_paths")), len(_list_of(r, "parquet_paths")))
        return f"cards/tables/views/json/parquet = {ns}", len(set(ns)) == 1
    add(_run(t, "C6 cards, tables, views and paths have equal counts", "equal", c6_counts))

    def c6_exist() -> tuple[Any, bool]:
        paths = [pathlib.Path(p) for p in json_paths + parquet_paths]
        missing = [str(p) for p in paths if not p.exists()]
        return f"missing={missing} (checked {len(paths)})", not missing and len(paths) > 0
    add(_run(t, "C6 card json and parquet files exist", "all exist", c6_exist))

    def c6_layout() -> tuple[Any, bool]:
        bad: list[str] = []
        js, ps = set(), set()
        for paths, suffix, acc in ((json_paths, _JSON_SUFFIX, js), (parquet_paths, _PARQUET_SUFFIX, ps)):
            for p in paths:
                parts = _split_card_path(p, suffix)
                if parts is None:
                    bad.append(f"{p}: name does not end in {suffix}")
                    continue
                if parts[1] != r.pin:
                    bad.append(f"{p}: parent dir {parts[1]!r} != pin")
                acc.add((parts[3], parts[2]))
        unpaired = sorted(js ^ ps)
        if unpaired:
            bad.append(f"json/parquet without a same-dir same-stem partner: {unpaired}")
        return f"problems={bad}", not bad and bool(js)
    add(_run(t, "C6 path layout: parent == pin, suffixes, one shared stem per json/parquet pair",
             f"<sha>/<stem>{_JSON_SUFFIX} + <sha>/<stem>{_PARQUET_SUFFIX}", c6_layout))

    def c6_card_paths(card: dict) -> tuple[Any, bool]:
        want = (_repo_slug(card["key"]["repo"]), r.pin, _component_slug(card["key"]["component"]))
        have_j = {_split_card_path(p, _JSON_SUFFIX) for p in json_paths} - {None}
        have_p = {_split_card_path(p, _PARQUET_SUFFIX) for p in parquet_paths} - {None}
        dirs_j = {x[3] for x in have_j if x[:3] == want}
        dirs_p = {x[3] for x in have_p if x[:3] == want}
        return f"json dirs={sorted(dirs_j)} parquet dirs={sorted(dirs_p)}", bool(dirs_j & dirs_p)
    for i, card in enumerate(cards):
        add(_run(t, f"C6 card[{i}] {_comp(card)} files at <repo_slug>/<pin>/<component_slug>",
                 _peek(lambda card=card: "/".join((_repo_slug(card["key"]["repo"]), r.pin,
                                                   _component_slug(card["key"]["component"])))),
                 lambda card=card: c6_card_paths(card)))
        add(_eq(t, f"C6 card[{i}] {_comp(card)} key.revision_sha == pin", r.pin,
                lambda card=card: card["key"]["revision_sha"]))

    # C7 (tables paired with cards by Parquet schema metadata, not by position)
    for i, card in enumerate(cards):
        lab = f"card[{i}] {_comp(card)}"
        add(_eq(t, f"C7 {lab} parquet rows == n_tensors", _peek(lambda card=card: card["weights"]["n_tensors"]),
                lambda card=card: _table_for(r, card).num_rows))

        def c7_stats(card=card) -> tuple[Any, bool]:
            table = _table_for(r, card)
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
    for i, v in enumerate(views):
        def c8(v=v) -> tuple[Any, bool]:
            n = sum(1 for node in v["nodes"] if node["kind"] != "tensor")
            return n, n <= 100
        add(_run(t, f"C8 view[{i}] non-tensor nodes <= 100", "<= 100", c8))
    if not views:
        add(Check(t, "C8 at least one view", ">= 1", "0", False))

    # C9 (sanity)
    for i, card in enumerate(cards):
        add(_eq(t, f"C9 card[{i}] {_comp(card)} params_total == sum(numel)",
                _peek(lambda card=card: card["weights"]["params_total"]),
                lambda card=card: sum(_table_for(r, card).column("numel").to_pylist())))

    out += _c10(t, r.pin, cards, wire, expected_endpoint)

    # C11
    def c11() -> tuple[Any, bool]:
        wire_bytes = sum(w["body_bytes"] for w in wire)
        tot = r.final["totals"]
        logged = tot["meta"] + tot["header"] + tot["weight"]
        return f"wire={wire_bytes} bytelog={logged}", wire_bytes == logged
    add(_run(t, "C11 wire body bytes == ByteLog meta+header+weight", "equal", c11))

    # C12
    add(_run(t, "C12 at least one wire record", ">= 1", lambda: (len(wire), len(wire) >= 1)))

    def c12_hosts() -> tuple[Any, bool]:
        hosts = {(urlsplit(w["url"]).hostname or "").lower() for w in wire}
        bad = sorted(h for h in hosts if not host_allowed(h, allowed))
        return f"hosts={sorted(hosts)} offending={bad}", not bad
    add(_run(t, "C12 every wire host allowed", f"hosts in {list(allowed)}", c12_hosts))

    def c12_endpoint() -> tuple[Any, bool]:
        eps = [c["source"]["endpoint"] for c in cards]
        return (f"endpoints={sorted(set(map(str, eps)))} cards={len(eps)}",
                bool(eps) and all(e == expected_endpoint for e in eps))
    add(_run(t, "C12 every card source.endpoint", expected_endpoint, c12_endpoint))

    return out


def _c10(t: str, pin: str, cards: list[dict], wire: list[dict], expected_endpoint: str) -> list[Check]:
    """C10, wire-level header-only evidence.

    Every record's orig_path is normalised with unquote(urlsplit(orig_path).path) and must lie under
    /{repo}/resolve/{pin}/ for a Card repo; a 2xx record without orig_path fails (the one exception is
    the Hub API revision call /api/models/{repo}/revision/{pin} on the expected endpoint host, which
    T015 records with orig_path None and which never carries file bytes). Every weight request (name
    ends in .safetensors, any case, or maps to a Card file) must map exactly to a Card weight file
    (extension compared case-insensitively) and carry Range bytes=a-b with b + 1 <= 8 + header_len.
    Every Card weight file needs >= 1 2xx record mapped to it.
    """
    out: list[Check] = []
    add = out.append

    file_map: dict[tuple[str, str], tuple[str, int]] = {}   # ext_key(resolve path) -> (label, header_len)
    prefixes: set[str] = set()
    api_paths: set[str] = set()
    labels: list[str] = []
    for i, card in enumerate(cards):
        try:
            repo = card["key"]["repo"]
            if not isinstance(repo, str) or not repo:
                raise TypeError(f"key.repo is {repo!r}")
            prefixes.add(f"/{repo}/resolve/{pin}/")
            api_paths.add(f"/api/models/{repo}/revision/{pin}")
            for f in card["weights"]["files"]:
                path, hl = f["path"], f["header_len"]
                if not isinstance(path, str) or not isinstance(hl, int) or isinstance(hl, bool):
                    raise TypeError(f"file path/header_len {path!r}/{hl!r}")
                label = f"{_comp(card)} {path}"
                file_map[_ext_key(f"/{repo}/resolve/{pin}/{path}")] = (label, hl)
                labels.append(label)
        except Exception as e:  # noqa: BLE001
            add(Check(t, f"C10 card[{i}] {_comp(card)} weight files readable", "key.repo + weights.files",
                      f"error: {type(e).__name__}: {e}", False))
    endpoint_host = (urlsplit(expected_endpoint).hostname or "").lower()

    def api_exempt(w: dict) -> bool:
        try:
            u = urlsplit(w["url"])
            return (u.hostname or "").lower() == endpoint_host and unquote(u.path) in api_paths
        except Exception:  # noqa: BLE001
            return False

    hits: dict[str, int] = {lab: 0 for lab in labels}
    offenders: list[str] = []
    n_weight = 0
    for i, w in enumerate(wire):
        try:
            orig, status = w["orig_path"], w["status"]
        except Exception as e:  # noqa: BLE001
            offenders.append(f"wire[{i}]: unreadable ({type(e).__name__}: {e})")
            continue
        ok2xx = _is_2xx(status)
        if orig is None:
            if ok2xx and not api_exempt(w):
                offenders.append(f"wire[{i}] {w.get('url')} status={status}: 2xx without orig_path")
            continue
        try:
            p = _norm_orig(orig)
        except Exception as e:  # noqa: BLE001
            offenders.append(f"wire[{i}] orig_path={orig!r}: {type(e).__name__}: {e}")
            continue
        if not any(p.startswith(pre) for pre in prefixes):
            offenders.append(f"wire[{i}] {p}: not under {sorted(prefixes)}")
        entry = file_map.get(_ext_key(p))
        if not (_is_weight_name(p) or entry is not None):
            continue
        n_weight += 1
        if entry is not None and ok2xx:
            hits[entry[0]] += 1

        def c10(w=w, p=p, entry=entry) -> tuple[Any, bool]:
            if entry is None:
                return f"{p} matches no Card weight file at /<repo>/resolve/{pin}/", False
            rng = w["range"]
            if rng is None:
                return "no Range header", False
            m = _RANGE_RE.fullmatch(str(rng).strip())
            if not m:
                return f"unparseable Range {rng!r}", False
            a, b = int(m.group(1)), int(m.group(2))
            limit = 8 + entry[1]
            return f"{rng} (b+1={b + 1}, 8+header_len={limit})", a <= b and b + 1 <= limit
        add(_run(t, f"C10 {p} [wire {i}] is a Card weight file with Range within header",
                 "maps to a Card file; Range bytes=a-b with b+1 <= 8+header_len", c10))

    add(_run(t, "C10 every wire record maps to /{repo}/resolve/{pin}/ (2xx needs orig_path)", "no offenders",
             lambda: (f"offenders={offenders} records={len(wire)}", not offenders and bool(prefixes))))
    for lab in labels:
        add(_run(t, f"C10 {lab}: >= 1 2xx wire record", ">= 1", lambda lab=lab: (hits[lab], hits[lab] >= 1)))
    add(_run(t, "C10 at least one weight-file wire record", ">= 1", lambda: (n_weight, n_weight >= 1)))
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
    return _peek(lambda: cfg(_card(r, component), *keys))


def expect_qwen3_30b_a3b(r: TargetResult) -> list[Check]:
    t = r.target
    c = lambda: _card(r, None)  # noqa: E731
    eg = lambda: c()["structure"]["expert_groups"][0]  # noqa: E731

    def strips() -> list[dict]:
        return [s for s in _view(r, None)["depth_strips"] if s["prefix"] == "model.layers"]

    def strip() -> dict:
        found = strips()
        if len(found) != 1:
            raise KeyError(f"{len(found)} depth strips with prefix 'model.layers'")
        return found[0]

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
        _eq(t, "E2 exactly one model.layers depth strip", 1, lambda: len(strips())),
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


def _expected_str(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {type(e).__name__}: {e}>"


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

    return [
        _run(t, f"{prefix} a stack with prefix suffix {suffix!r} exists", f"some stack ending in {suffix!r}", exists),
        _run(t, f"{prefix} stack ending in {suffix!r} has expected depth",
             f"depth == {_expected_str(expected_depth_fn)}", depth),
    ]


def _exact_prefix_checks(t: str, prefix: str, card_fn: Callable[[], dict], stack_prefix: str,
                         expected_depth_fn: Callable[[], Any]) -> list[Check]:
    """Two Checks: a stack with exactly this prefix exists, and it has the expected depth."""
    def exists() -> tuple[Any, bool]:
        prefixes = [s["prefix"] for s in card_fn()["structure"]["stacks"]]
        return prefixes, stack_prefix in prefixes

    def depth() -> tuple[Any, bool]:
        want = expected_depth_fn()
        d = _stack_by_prefix(card_fn(), stack_prefix)["depth"]
        return d, d == want and type(d) is type(want)

    return [
        _run(t, f"{prefix} stack with prefix {stack_prefix!r} exists", f"prefix == {stack_prefix!r}", exists),
        _run(t, f"{prefix} stack {stack_prefix!r} has expected depth",
             f"depth == {_expected_str(expected_depth_fn)}", depth),
    ]


def expect_qwen_image(r: TargetResult) -> list[Check]:
    t = r.target
    te = lambda: _card(r, "text_encoder")  # noqa: E731
    tr = lambda: _card(r, "transformer")  # noqa: E731
    vae = lambda: _card(r, "vae")  # noqa: E731

    def pipeline() -> dict:
        """The pipeline dict, read once from the first card (consistency is a separate Check)."""
        cards = _cards(r)
        if not cards:
            raise KeyError("no cards")
        p = cards[0]["pipeline"]
        if not isinstance(p, dict):
            raise TypeError(f"pipeline is {type(p).__name__}")
        return p

    def pipeline_same() -> tuple[Any, bool]:
        cards = _cards(r)
        distinct = []
        for c in cards:
            if c["pipeline"] not in distinct:
                distinct.append(c["pipeline"])
        return f"{len(distinct)} distinct pipeline dicts over {len(cards)} cards", bool(cards) and len(distinct) == 1

    def pipeline_components() -> dict:
        return {c["name"]: c["has_weights"] for c in pipeline()["components"]}

    def text_depth() -> Any:
        raw = te()["config"]["raw"]
        if raw is None:
            raise KeyError("config.raw is null")
        tc = raw.get("text_config")
        if isinstance(tc, dict) and "num_hidden_layers" in tc:
            return tc["num_hidden_layers"]
        return raw["num_hidden_layers"]

    def te_archs() -> tuple[Any, bool]:
        a = te()["config"]["architectures"]
        return a, isinstance(a, list) and "Qwen2_5_VLForConditionalGeneration" in a

    out: list[Check] = [
        _eq(t, "E3 number of cards", 3, lambda: len(r.cards)),
        _eq(t, "E3 components", {"text_encoder", "transformer", "vae"},
            lambda: {c["key"]["component"] for c in _cards(r)}),
        _run(t, "E3 all cards carry an identical pipeline", "1 distinct pipeline dict", pipeline_same),
        _eq(t, "E3 pipeline.pipeline_class", "QwenImagePipeline", lambda: pipeline()["pipeline_class"]),
    ]
    for name, want in (("scheduler", False), ("tokenizer", False), ("text_encoder", True),
                       ("transformer", True), ("vae", True)):
        out.append(_eq(t, f"E3 pipeline.components {name}.has_weights", want, lambda name=name: pipeline_components()[name]))

    out.append(_run(t, "E3 text_encoder architectures is a list containing Qwen2_5_VLForConditionalGeneration",
                    "list containing it", te_archs))
    out += _suffix_checks(t, "E3 text_encoder", te, "layers", text_depth)
    out += _suffix_checks(t, "E3 text_encoder", te, "blocks", lambda: cfg(te(), "vision_config", "depth"))
    out.append(_eq(t, "E3 text_encoder expert_groups", [], lambda: te()["structure"]["expert_groups"]))

    out.append(_eq(t, "E3 transformer config.class_name", "QwenImageTransformer2DModel",
                   lambda: tr()["config"]["class_name"]))
    out += _exact_prefix_checks(t, "E3 transformer", tr, "transformer_blocks", lambda: cfg(tr(), "num_layers"))

    out += [
        _eq(t, "E3 vae config.class_name", "AutoencoderKLQwenImage", lambda: vae()["config"]["class_name"]),
        _eq(t, "E3 vae number of weight files", 1, lambda: len(vae()["weights"]["files"])),
        _eq(t, "E3 vae index_path", None, lambda: vae()["weights"]["index_path"]),
        _eq(t, "E3 all cards same revision_sha", 1, lambda: len({c["key"]["revision_sha"] for c in _cards(r)})),
        _eq(t, "E3 all cards model_card.license", {"apache-2.0"},
            lambda: {c["model_card"]["license"] for c in _cards(r)}),
    ]
    return out


DEFAULT_EXPECTATIONS: dict[str, Callable[[TargetResult], list[Check]]] = {
    "Qwen/Qwen3-8B": expect_qwen3_8b,
    "Qwen/Qwen3-30B-A3B": expect_qwen3_30b_a3b,
    "Qwen/Qwen-Image": expect_qwen_image,
}
