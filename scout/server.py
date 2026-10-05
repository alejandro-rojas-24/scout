"""Stdlib HTTP server: background scans, live log polling, views built from persisted Cards.

The server persists nothing itself; scan() writes the Cards (invariant 1) and every byte
fetched is in the scan's ByteLog, which clients poll (invariant 2).
"""

from __future__ import annotations

import http.server
import importlib.resources
import json
import os
import pathlib
import sys
import threading
import time
import urllib.parse
import uuid
from typing import Callable

import httpx

from scout.bytelog import DEFAULT_THRESHOLD_BYTES, ByteLog
from scout.card import load_card
from scout.scan import parse_target, scan
from scout.view import build_view

POLL_EVENTS_MAX = 5000
_MAX_BODY_BYTES = 1 << 20

WEB_DIR = importlib.resources.files("scout").joinpath("web")


class ScanState:
    def __init__(self, scan_id: str, target: str, log: ByteLog) -> None:
        self.scan_id = scan_id
        self.target = target
        self.status = "running"  # "running" | "done" | "error"
        self.log = log
        self.t0 = time.monotonic()
        self.elapsed_s: float | None = None
        self.error: dict | None = None
        self.views: list[dict] | None = None


class ScoutServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    out_dir: pathlib.Path
    threshold_bytes: int
    endpoint: str | None
    client_factory: Callable[[], httpx.Client] | None
    scans: dict[str, ScanState]
    lock: threading.Lock

    def handle_error(self, request, client_address) -> None:
        if isinstance(sys.exc_info()[1], ConnectionError) and os.environ.get("SCOUT_HTTP_LOG") != "1":
            return
        super().handle_error(request, client_address)


def _run_scan(server: ScoutServer, state: ScanState) -> None:
    client: httpx.Client | None = None
    try:
        client = server.client_factory() if server.client_factory is not None else None
        result = scan(state.target, server.out_dir, state.log, client=client, endpoint=server.endpoint)
        views = [build_view(*load_card(p.json_path)) for p in result.cards]
    except BaseException as e:  # noqa: BLE001 - every failure must surface as status "error"
        with server.lock:
            state.status = "error"
            state.error = {"type": type(e).__name__, "message": str(e)}
            state.elapsed_s = time.monotonic() - state.t0
        if not isinstance(e, Exception):
            raise
        return
    finally:
        if client is not None:
            client.close()
    with server.lock:
        state.views = views
        state.status = "done"
        state.elapsed_s = time.monotonic() - state.t0


class _Handler(http.server.BaseHTTPRequestHandler):
    server: ScoutServer
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if os.environ.get("SCOUT_HTTP_LOG") == "1":
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    # ---- responses
    def _send(self, status: int, body: bytes, content_type: str, cache_no_store: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", cache_no_store=True)

    def _static(self, name: str, content_type: str) -> None:
        try:
            f = WEB_DIR.joinpath(name)
            data = f.read_bytes() if f.is_file() else None
        except OSError:
            data = None
        if data is None:
            self._json(404, {"error": "not found"})
            return
        self._send(200, data, content_type)

    # ---- routes
    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if path == "/":
            self._static("index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._static("app.js", "application/javascript; charset=utf-8")
        elif path.startswith("/api/scans/") and path.count("/") == 3:
            self._get_scan(path.rsplit("/", 1)[1], parsed.query)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urllib.parse.urlsplit(self.path).path == "/api/scans":
            self._post_scan()
        else:
            self._json(404, {"error": "not found"})

    def _post_scan(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > _MAX_BODY_BYTES:
            self.close_connection = True
            self._json(400, {"error": "invalid body length"})
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:  # includes UnicodeDecodeError and JSONDecodeError
            self._json(400, {"error": "body is not valid JSON"})
            return
        target = body.get("target") if isinstance(body, dict) else None
        if not isinstance(target, str) or not target.strip():
            self._json(400, {"error": "target must be a non-empty string"})
            return
        try:
            parse_target(target)
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return
        server = self.server
        scan_id = uuid.uuid4().hex[:12]
        state = ScanState(scan_id, target, ByteLog(server.threshold_bytes))
        with server.lock:
            server.scans[scan_id] = state
        threading.Thread(target=_run_scan, args=(server, state), name=f"scout-scan-{scan_id}",
                         daemon=True).start()
        self._json(202, {"scan_id": scan_id})

    def _get_scan(self, scan_id: str, query: str) -> None:
        try:
            raw = urllib.parse.parse_qs(query).get("since", ["0"])[0]
            since = int(raw)
        except ValueError:
            self._json(400, {"error": "since must be an integer"})
            return
        server = self.server
        with server.lock:
            state = server.scans.get(scan_id)
            if state is None:
                snap = None
            else:
                snap = (state.status, state.elapsed_s, state.error,
                        state.views if state.status == "done" else None)
        if state is None or snap is None:
            self._json(404, {"error": "not found"})
            return
        status, elapsed_s, error, views = snap
        if status == "running":
            elapsed_s = time.monotonic() - state.t0
        events = [e.to_dict() for e in state.log.events_since(since)][:POLL_EVENTS_MAX]
        self._json(200, {
            "scan_id": state.scan_id, "target": state.target, "status": status,
            "elapsed_s": elapsed_s, "error": error, "totals": state.log.totals,
            "events": events, "views": views,
        })


def make_server(host: str = "127.0.0.1", port: int = 8765, out_dir: pathlib.Path = pathlib.Path("cards"),
                threshold_bytes: int = DEFAULT_THRESHOLD_BYTES, endpoint: str | None = None,
                client_factory: Callable[[], httpx.Client] | None = None) -> ScoutServer:
    server = ScoutServer((host, port), _Handler)
    server.out_dir = pathlib.Path(out_dir)
    server.threshold_bytes = threshold_bytes
    server.endpoint = endpoint
    server.client_factory = client_factory
    server.scans = {}
    server.lock = threading.Lock()
    return server


def serve(host: str, port: int, out_dir: pathlib.Path, threshold_bytes: int, endpoint: str | None) -> None:
    server = make_server(host, port, out_dir, threshold_bytes, endpoint)
    try:
        server.serve_forever()
    finally:
        server.server_close()
