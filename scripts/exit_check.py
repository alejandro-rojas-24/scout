"""Phase 1 exit check (plan.md section 2): C0 target set, then C1..C12 + E1..E3 per pinned target.

    python scripts/exit_check.py --pins exit/pins.json

Each target is scanned through the real server (make_server, POST /api/scans, poll every 100 ms),
with its own server and its own CountingTransport, strictly one after the other. The
CountingTransport is a transport wrapper (not an event hook) that records every request on the
wire independently of the code under test. The endpoint is fixed at https://huggingface.co and
the C1 budget at 10.0 s; neither can be changed from the command line.

The last stdout line is a JSON summary {"step": "EXIT", ...} for plans/phase-1/log.jsonl.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from scout.card import load_card, repo_slug  # noqa: E402
from scout.server import make_server  # noqa: E402
from scripts.exit_expectations import (  # noqa: E402
    ALLOWED_HOSTS,
    DEFAULT_EXPECTATIONS,
    HF_ENDPOINT_URL,
    Check,
    TargetResult,
    common_checks,
)

BUDGET_S: float = 10.0
POLL_INTERVAL_S = 0.1
SCAN_TIMEOUT_S = 60.0
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RESOLVE_RE = re.compile(r"^/[^/]+/[^/]+/resolve/[^/]+/.+$")

# ---------------------------------------------------------------------------
# wire evidence


class _CountingStream(httpx.SyncByteStream):
    """Yields the inner chunks and counts each one as it is handed to the client."""

    def __init__(self, inner: httpx.SyncByteStream, owner: "CountingTransport", record: dict) -> None:
        self._inner = inner
        self._owner = owner
        self._record = record

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            self._owner._add_bytes(self._record, len(chunk))
            yield chunk

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()


class CountingTransport(httpx.BaseTransport):
    """Transport wrapper recording {method,url,orig_path,range,status,location,body_bytes} per request.

    orig_path is the full unquoted URL path of the originating resolve request
    (/{owner}/{name}/resolve/{sha}/{file}); every redirect hop is mapped back to it through the
    resolved absolute Location of the previous 3xx. The Hub API call keeps orig_path None.
    """

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._location_orig: dict[str, str | None] = {}
        self.records: list[dict] = []

    def _add_bytes(self, record: dict, n: int) -> None:
        with self._lock:
            record["body_bytes"] += n

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.records]

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        path = unquote(urlsplit(url).path)
        with self._lock:
            if _RESOLVE_RE.match(path):
                orig: str | None = path
            else:
                orig = self._location_orig.get(url)
            record = {"method": request.method, "url": url, "orig_path": orig,
                      "range": request.headers.get("range"), "status": None, "location": None,
                      "body_bytes": 0}
            self.records.append(record)
        resp = self._inner.handle_request(request)  # a transport error leaves status None
        location = None
        if 300 <= resp.status_code < 400 and resp.headers.get("location"):
            location = str(request.url.join(resp.headers["location"]))
        with self._lock:
            record["status"] = resp.status_code
            record["location"] = location
            if location is not None:
                self._location_orig[location] = orig
        return httpx.Response(status_code=resp.status_code, headers=resp.headers,
                              stream=_CountingStream(resp.stream, self, record),
                              extensions=resp.extensions)

    def close(self) -> None:
        self._inner.close()


def _env_proxy_for(url: str) -> str | None:
    """The environment proxy for url (HTTPS_PROXY/HTTP_PROXY/ALL_PROXY), or None if NO_PROXY bypasses it."""
    parts = urlsplit(url)
    if urllib.request.proxy_bypass_environment(parts.hostname or ""):
        return None
    proxies = urllib.request.getproxies_environment()
    return proxies.get(parts.scheme) or proxies.get("all") or None


class _EnvProxyTransport(httpx.BaseTransport):
    """Per request host: httpx.HTTPTransport(proxy=<env proxy for that URL, honouring NO_PROXY>).

    Sits UNDER CountingTransport, so wire records always carry the origin URL, never the proxy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._transports: dict[str | None, httpx.HTTPTransport] = {}

    def _for(self, proxy: str | None) -> httpx.HTTPTransport:
        with self._lock:
            t = self._transports.get(proxy)
            if t is None:
                t = self._transports[proxy] = httpx.HTTPTransport(proxy=proxy)
            return t

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._for(_env_proxy_for(str(request.url))).handle_request(request)

    def close(self) -> None:
        with self._lock:
            ts, self._transports = list(self._transports.values()), {}
        for t in ts:
            t.close()


def default_inner_transport() -> httpx.BaseTransport:
    return _EnvProxyTransport()


def _client_for(counting: CountingTransport) -> httpx.Client:
    return httpx.Client(transport=counting, timeout=10.0)


# ---------------------------------------------------------------------------
# one target


def _poll_until_terminal(ctl: httpx.Client, base: str, scan_id: str, t0: float) -> tuple[str, float]:
    since = 0
    while True:
        body = ctl.get(f"{base}/api/scans/{scan_id}", params={"since": since}).json()
        events = body.get("events") or []
        if events:
            since = events[-1]["seq"]
        status = body.get("status")
        if status == "error" or (status == "done" and body.get("views") is not None):
            return status, time.monotonic() - t0
        if time.monotonic() - t0 >= SCAN_TIMEOUT_S:
            return f"timeout after {SCAN_TIMEOUT_S:.0f} s (status {status!r})", time.monotonic() - t0
        time.sleep(POLL_INTERVAL_S)


def _collect_events(ctl: httpx.Client, base: str, scan_id: str) -> tuple[dict, list[dict]]:
    """All events from since=0, paging (POLL_EVENTS_MAX per poll) until a poll returns none."""
    events: list[dict] = []
    since = 0
    while True:
        final = ctl.get(f"{base}/api/scans/{scan_id}", params={"since": since}).json()
        page = final.get("events") or []
        if not page:
            return final, events
        events.extend(page)
        nxt = page[-1]["seq"]
        if nxt <= since:
            return final, events
        since = nxt


def _scan_target(repo: str, sha: str, out_dir: Path, *, endpoint: str,
                 inner_transport_factory: Callable[[], httpx.BaseTransport]) -> TargetResult:
    counting = CountingTransport(inner_transport_factory())
    client_factory = functools.partial(_client_for, counting)
    server = make_server(port=0, out_dir=out_dir, endpoint=endpoint, client_factory=client_factory)
    thread = threading.Thread(target=server.serve_forever, name=f"exit-check-{repo}", daemon=True)
    thread.start()
    target = f"{repo}@{sha}"
    status, elapsed, final, events = "not started", math.nan, {}, []
    views: list[dict] = []
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(timeout=30.0, trust_env=False) as ctl:  # control traffic: not counted
            try:
                t0 = time.monotonic()
                resp = ctl.post(f"{base}/api/scans", json={"target": target})
                if resp.status_code != 202:
                    raise RuntimeError(f"POST /api/scans -> {resp.status_code}: {resp.text[:200]}")
                scan_id = resp.json()["scan_id"]
                status, elapsed = _poll_until_terminal(ctl, base, scan_id, t0)
                final, events = _collect_events(ctl, base, scan_id)
                views = final.get("views") or []
            except Exception as e:  # noqa: BLE001 - surfaces as a failing C1, never a crash
                status = f"runner error: {type(e).__name__}: {e}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)

    cards, tables, json_paths, parquet_paths = [], [], [], []
    card_dir = Path(out_dir) / repo_slug(repo) / sha
    for jp in sorted(card_dir.glob("*.card.json")) if card_dir.is_dir() else []:
        json_paths.append(jp)
        try:
            card, table = load_card(jp)
        except Exception:  # noqa: BLE001 - an unreadable card shows up as a C6 count mismatch
            continue
        cards.append(card)
        tables.append(table)
        parquet_paths.append(jp.parent / card["tensors_parquet"])
    return TargetResult(target=target, pin=sha, status=status, elapsed_s=elapsed, final=final, events=events,
                        cards=cards, tables=tables, views=views, json_paths=json_paths,
                        parquet_paths=parquet_paths, wire=counting.snapshot())


# ---------------------------------------------------------------------------
# run


def check_target_set(pins: dict[str, str], expectations: dict) -> Check:
    return Check(target="*", name="C0 target set", expected=str(sorted(expectations)), actual=str(sorted(pins)),
                 ok=set(pins) == set(expectations))


def _safe_expectation(fn: Callable[[TargetResult], list[Check]], r: TargetResult) -> list[Check]:
    try:
        return list(fn(r))
    except Exception as e:  # noqa: BLE001
        return [Check(r.target, "expectations", "no exception", f"error: {type(e).__name__}: {e}", False)]


def _run_with_results(pins: dict[str, str], out_dir: Path, *, endpoint: str = HF_ENDPOINT_URL,
                      allowed_hosts: Iterable[str] = ALLOWED_HOSTS,
                      inner_transport_factory: Callable[[], httpx.BaseTransport] = default_inner_transport,
                      expectations: dict[str, Callable[[TargetResult], list[Check]]] | None = None,
                      budget_s: float = BUDGET_S) -> tuple[list[Check], list[TargetResult]]:
    expectations = DEFAULT_EXPECTATIONS if expectations is None else expectations
    allowed = tuple(allowed_hosts)
    c0 = check_target_set(pins, expectations)
    if not c0.ok:
        return [c0], []
    checks = [c0]
    results: list[TargetResult] = []
    for repo, sha in pins.items():
        r = _scan_target(repo, sha, Path(out_dir), endpoint=endpoint,
                         inner_transport_factory=inner_transport_factory)
        results.append(r)
        checks += common_checks(r, budget_s, allowed_hosts=allowed, expected_endpoint=endpoint)
        checks += _safe_expectation(expectations[repo], r)
    return checks, results


def run(pins: dict[str, str], out_dir: Path, *, endpoint: str = HF_ENDPOINT_URL,
        allowed_hosts: Iterable[str] = ALLOWED_HOSTS,
        inner_transport_factory: Callable[[], httpx.BaseTransport] = default_inner_transport,
        expectations: dict[str, Callable[[TargetResult], list[Check]]] | None = None,
        budget_s: float = BUDGET_S) -> list[Check]:
    return _run_with_results(pins, out_dir, endpoint=endpoint, allowed_hosts=allowed_hosts,
                             inner_transport_factory=inner_transport_factory, expectations=expectations,
                             budget_s=budget_s)[0]


# ---------------------------------------------------------------------------
# main


def _git_info() -> tuple[str, bool | None]:
    def git(*args: str) -> str | None:
        try:
            p = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        except Exception:  # noqa: BLE001
            return None
        return p.stdout if p.returncode == 0 else None
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return (head.strip() or "unknown") if head is not None else "unknown", \
        (bool(status.strip()) if status is not None else None)


def _row(c: Check) -> str:
    return f"{c.target} | {c.name} | {c.expected} | {c.actual} | {'PASS' if c.ok else 'FAIL'}"


def _load_pins(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError("pins file must be a JSON object mapping repo -> str")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 1 exit check against https://huggingface.co "
                                            f"(fixed endpoint, fixed {BUDGET_S} s budget).")
    p.add_argument("--pins", type=Path, default=ROOT / "exit" / "pins.json",
                   help="JSON {repo: 40-hex sha} (default: <repo>/exit/pins.json)")
    p.add_argument("--out", type=Path, default=None, help="cards dir (default: a fresh temp dir)")
    args = p.parse_args(argv)

    env_ep = os.environ.get("HF_ENDPOINT")
    if env_ep and env_ep.removesuffix("/") != HF_ENDPOINT_URL:
        print(f"HF_ENDPOINT={env_ep} is not allowed for the exit check")
        return 1

    try:
        pins = _load_pins(args.pins)
    except (OSError, ValueError) as e:
        print(f"cannot read pins file {args.pins}: {type(e).__name__}: {e}")
        return 1

    c0 = check_target_set(pins, DEFAULT_EXPECTATIONS)
    if not c0.ok:
        print(_row(c0))
        print(f"EXIT CHECK: FAIL (target set must be exactly: {', '.join(sorted(DEFAULT_EXPECTATIONS))})")
        return 1
    if not all(_SHA_RE.fullmatch(v) for v in pins.values()):
        print("pins not set: run `scout resolve <repo>` and commit exit/pins.json")
        return 2

    out_dir = args.out if args.out is not None else Path(tempfile.mkdtemp(prefix="scout-exit-"))
    print(f"cards dir: {out_dir}", file=sys.stderr)
    checks, results = _run_with_results(pins, out_dir, endpoint=HF_ENDPOINT_URL, allowed_hosts=ALLOWED_HOSTS,
                                        budget_s=BUDGET_S)

    print("target | check | expected | actual | result")
    for c in checks:
        print(_row(c))
    ok = bool(checks) and all(c.ok for c in checks)
    print("EXIT CHECK: PASS" if ok else "EXIT CHECK: FAIL")

    hosts = sorted({(urlsplit(w.get("url", "")).hostname or "").lower()
                    for r in results for w in r.wire if isinstance(w, dict)} - {""})
    proxy = _env_proxy_for(HF_ENDPOINT_URL)
    git_commit, git_dirty = _git_info()
    print(json.dumps({
        "step": "EXIT", "hostname": socket.gethostname(), "pins": pins,
        "passed": sum(1 for c in checks if c.ok), "failed": sum(1 for c in checks if not c.ok),
        "endpoint": HF_ENDPOINT_URL, "budget_s": BUDGET_S, "hosts": hosts,
        "proxy": (urlsplit(proxy).hostname or proxy) if proxy else None,
        "git_commit": git_commit, "git_dirty": git_dirty,
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
