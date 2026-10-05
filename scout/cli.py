"""Command line interface: scout scan / resolve / serve, with a live byte log on stderr."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import httpx

from scout.bytelog import DEFAULT_THRESHOLD_BYTES, ByteLog, LogEvent, Stage
from scout.errors import ScoutError
from scout.hub import HubSource
from scout.scan import parse_target, scan
from scout.view import DISCLAIMERS


def format_event(e: LogEvent) -> str:
    t = e.totals
    line = (f"[{e.stage:<8}] {e.event:<12} {e.path or '-'} +{e.bytes}B  "
            f"meta={t['meta']} header={t['header']} weight={t['weight']}")
    if e.note:
        line += f"  ({e.note})"
    return line


def _stderr_log(threshold_bytes: int) -> ByteLog:
    return ByteLog(threshold_bytes=threshold_bytes,
                   on_event=lambda e: print(format_event(e), file=sys.stderr, flush=True))


def _default_out() -> str:
    return os.environ.get("SCOUT_CARDS_DIR") or "./cards"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scout", description="Model lineage forensics tool")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="header-only scan of a Hub repo or local folder")
    s.add_argument("target")
    s.add_argument("--out", default=_default_out())
    s.add_argument("--max-read-bytes", type=int, default=DEFAULT_THRESHOLD_BYTES)
    s.add_argument("--endpoint", default=None)

    r = sub.add_parser("resolve", help="resolve REPO[@REV] to a commit SHA")
    r.add_argument("target", metavar="REPO[@REV]")
    r.add_argument("--endpoint", default=None)

    v = sub.add_parser("serve", help="serve the web UI")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--out", default=_default_out())
    v.add_argument("--max-read-bytes", type=int, default=DEFAULT_THRESHOLD_BYTES)
    v.add_argument("--endpoint", default=None)
    return p


def _cmd_scan(args: argparse.Namespace, client: httpx.Client | None) -> int:
    try:
        parse_target(args.target)
    except ValueError as e:
        print(f"error: ValueError: {e}", file=sys.stderr)
        return 2
    log = _stderr_log(args.max_read_bytes)
    result = scan(args.target, Path(args.out), log, client=client, endpoint=args.endpoint)
    print(json.dumps({
        "repo": result.repo,
        "revision_sha": result.revision_sha,
        "cards": [str(p.json_path) for p in result.cards],
        "totals": result.totals,
        "elapsed_s": round(result.elapsed_s, 3),
    }))
    for d in DISCLAIMERS:
        print(f"note: {d}", file=sys.stderr)
    return 0


def _cmd_resolve(args: argparse.Namespace, client: httpx.Client | None) -> int:
    try:
        kind, repo, rev = parse_target(args.target)
        if kind != "hub":
            raise ValueError(f"resolve needs a Hub repo (owner/name[@rev]): {args.target}")
    except ValueError as e:
        print(f"error: ValueError: {e}", file=sys.stderr)
        return 2
    log = _stderr_log(DEFAULT_THRESHOLD_BYTES)
    source = HubSource(repo, rev, log, client=client, endpoint=args.endpoint)
    try:
        log.stage(Stage.RESOLVE)
        source.resolve()
        print(source.revision_sha)
    finally:
        source.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from scout.server import serve

    print(f"scout serving on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        serve(args.host, args.port, Path(args.out), args.max_read_bytes, args.endpoint)
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv: list[str] | None = None, *,
         client_factory: Callable[[], httpx.Client] | None = None) -> int:
    args = _build_parser().parse_args(argv)  # argparse errors exit with 2
    if args.command == "serve":
        return _cmd_serve(args)
    client = None
    try:
        client = client_factory() if client_factory else None
        if args.command == "scan":
            return _cmd_scan(args, client)
        return _cmd_resolve(args, client)
    except ScoutError as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return e.exit_code
    except Exception as e:  # noqa: BLE001 - generic handler
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
