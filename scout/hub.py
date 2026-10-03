"""HubSource: Hugging Face Hub repo at a pinned revision, read with logged, bounded HTTP.

Every response body byte that the client yields is recorded in the ByteLog:
- 2xx bodies under the requested path (classified meta/header/weight by offset),
- 3xx bodies under REDIRECT_PATH (meta), redirects being followed manually,
- non-2xx bodies under ERROR_PATH (meta), so an error page on a .safetensors URL is
  never counted as header or weight,
- partial 2xx bodies cut off by a transport error as "retry" events.

Every read is preflighted (weight refusal + budget reservation) before any request.
Nothing is written to disk and huggingface_hub is never imported.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable
from urllib.parse import quote

import httpx

from scout import __version__
from scout.bytelog import META_MAX_BYTES, ByteLog
from scout.errors import (
    GatedRepoError,
    HubAPIError,
    NetworkError,
    RangeNotSupported,
    ReadThresholdExceeded,
    RepoNotFoundError,
    RevisionMismatch,
    ScoutError,
)
from scout.sources import RepoFile

DEFAULT_ENDPOINT = "https://huggingface.co"
API_PATH = "@api/revision"  # logged path for the API call (classified meta)
REDIRECT_PATH = "@redirect"  # logged path for 3xx bodies (classified meta)
MAX_REDIRECTS = 5
REDIRECT_MAX_BYTES = 64 * 1024  # per-hop cap and reservation for a 3xx body
ERROR_PATH = "@error"  # logged path for non-2xx response bodies (classified meta)
ERROR_BODY_MAX_BYTES = 64 * 1024  # cap and reservation for one non-2xx body
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_CONTENT_RANGE_RE = re.compile(r"^\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", re.IGNORECASE)
_SOURCE = "hub"

# _get modes
_API = "api"  # any 2xx accepted; non-2xx mapped with the API rules (behavior 2)
_FILE = "file"  # expects 200, body must equal the known size (if any)
_RANGE = "range"  # expects 206 with exactly `cap` bytes; 200 -> RangeNotSupported


class _Retryable(Exception):
    """Internal: a failed attempt that may be retried (429/5xx status or short body)."""

    def __init__(self, message: str, status: int | None) -> None:
        super().__init__(message)
        self.status = status


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return (url.scheme, url.host, url.port)


class HubSource:
    """A Hub model repo, resolved to a 40-hex commit SHA; implements scout.sources.Source."""

    kind = "hub"
    revision_kind = "git"
    local_path = None

    def __init__(
        self,
        repo_id: str,
        revision: str | None,
        log: ByteLog,
        *,
        client: httpx.Client | None = None,
        endpoint: str | None = None,
        token: str | None = None,
        attempts: int = 3,
        backoff_s: tuple[float, ...] = (0.5, 1.0),
        timeout_s: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(repo_id, str) or not REPO_RE.match(repo_id):
            raise ValueError(f"invalid Hub repo id {repo_id!r} (expected owner/name)")
        if attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {attempts}")
        self.repo = repo_id
        self.requested_revision = revision
        self.revision_sha = ""
        self.log = log
        ep = endpoint if endpoint is not None else (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT)
        self.endpoint = ep.rstrip("/")
        self._endpoint_url = httpx.URL(self.endpoint)
        tok = token if token is not None else os.environ.get("HF_TOKEN")
        self._token = tok or None
        self._attempts = attempts
        self._backoff = tuple(backoff_s)
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)
        self._base_headers = {"User-Agent": f"scout/{__version__}", "Accept-Encoding": "identity"}
        self._files: dict[str, RepoFile] | None = None

    # ------------------------------------------------------------------ helpers

    def _token_hint(self) -> str:
        return f"HF_TOKEN {'set' if self._token else 'not set'}"

    def _fail(self, exc: Exception, path: str | None) -> Exception:
        """Note a terminal error in the live log and return it for raising."""
        self.log.note("error", f"{type(exc).__name__}: {exc}", path=path)
        return exc

    def _headers_for(self, url: httpx.URL, extra: dict[str, str] | None) -> dict[str, str]:
        h = dict(self._base_headers)
        if extra:
            h.update(extra)
        if self._token and _origin(url) == _origin(self._endpoint_url):
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _file_url(self, path: str) -> str:
        return f"{self.endpoint}/{self.repo}/resolve/{self.revision_sha}/{quote(path)}"

    def _require_files(self) -> dict[str, RepoFile]:
        if self._files is None:
            raise ScoutError("HubSource used before resolve()")
        return self._files

    # ------------------------------------------------------------------ Source API

    def resolve(self) -> None:
        rev = self.requested_revision or "main"
        url = f"{self.endpoint}/api/models/{self.repo}/revision/{quote(rev, safe='')}?blobs=true"
        r = self.log.preflight(API_PATH, 0, META_MAX_BYTES)
        body = self._get(url, logged_path=API_PATH, start=None, cap=META_MAX_BYTES,
                         reservation=r, headers=None, mode=_API)
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise self._fail(HubAPIError(f"{self.repo}: API returned invalid JSON ({exc})"), API_PATH) from None
        if (not isinstance(data, dict) or not isinstance(data.get("sha"), str)
                or not isinstance(data.get("siblings"), list)):
            raise self._fail(HubAPIError(f"{self.repo}: API response lacks a str 'sha' and a list 'siblings'"),
                             API_PATH)
        sha = data["sha"]
        if not SHA_RE.match(sha):
            raise self._fail(HubAPIError(f"{self.repo}: API returned a non-40-hex sha {sha!r}"), API_PATH)
        req = self.requested_revision
        if req is not None and SHA_RE.match(req) and req != sha:
            raise self._fail(RevisionMismatch(f"{self.repo}: requested {req}, Hub returned {sha}"), API_PATH)
        files: dict[str, RepoFile] = {}
        for s in data["siblings"]:
            name = s.get("rfilename") if isinstance(s, dict) else None
            size = s.get("size") if isinstance(s, dict) else None
            if not isinstance(name, str) or not name:
                raise self._fail(HubAPIError(f"{self.repo}: sibling without a str 'rfilename'"), API_PATH)
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
                raise self._fail(HubAPIError(f"{self.repo}: invalid size for {name}: {size!r}"), API_PATH)
            files[name] = RepoFile(name, size)
        self.revision_sha = sha
        self._files = dict(sorted(files.items()))

    def files(self) -> list[RepoFile]:
        return list(self._require_files().values())

    def read_file(self, path: str) -> bytes | None:
        f = self._require_files().get(path)
        if f is None:
            return None
        n = f.size if f.size is not None else META_MAX_BYTES
        r = self.log.preflight(path, 0, n)
        return self._get(self._file_url(path), logged_path=path, start=0, cap=n, reservation=r,
                         headers=None, mode=_FILE, exact=f.size)

    def read_range(self, path: str, start: int, length: int) -> bytes:
        if start < 0 or length < 1:
            raise ValueError(f"invalid range start={start} length={length}")
        if path not in self._require_files():
            raise ScoutError(f"{path} is not in {self.repo}@{self.revision_sha}")
        r = self.log.preflight(path, start, start + length)
        return self._get(self._file_url(path), logged_path=path, start=start, cap=length, reservation=r,
                         headers={"Range": f"bytes={start}-{start + length - 1}"}, mode=_RANGE, exact=length)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------ HTTP core

    def _get(
        self,
        url: str,
        *,
        logged_path: str,
        start: int | None,
        cap: int,
        reservation: int,
        headers: dict[str, str] | None,
        mode: str = _FILE,
        exact: int | None = None,
    ) -> bytes:
        last: Exception | None = None
        try:
            for attempt in range(1, self._attempts + 1):
                t0 = time.monotonic()
                received = 0
                status: int | None = None
                final_url: str | None = None
                try:
                    resp = self._follow(httpx.URL(url), headers)
                    try:
                        status = resp.status_code
                        final_url = str(resp.request.url)
                        if not 200 <= status < 300:
                            body = self._read_error_body(resp, logged_path)
                            raise self._map_status(mode, status, resp.headers, body, logged_path)
                        enc = resp.headers.get("content-encoding")
                        if enc and enc.strip().lower() != "identity":
                            raise self._fail(NetworkError(
                                f"{logged_path}: unsupported Content-Encoding {enc!r} (identity requested)"),
                                logged_path)
                        if mode == _RANGE and status == 200:
                            self._range_ignored(resp, logged_path, cap, reservation, t0)
                        body_start = start
                        if mode == _RANGE and status == 206:
                            body_start = self._content_range_start(resp, start)
                        chunks: list[bytes] = []
                        for chunk in resp.iter_raw():
                            chunks.append(chunk)
                            received += len(chunk)
                            if received > cap:
                                resp.close()
                                self.log.record(
                                    event="fetch", source=_SOURCE, path=logged_path, url=final_url,
                                    start=body_start, nbytes=received, status=status, attempt=attempt,
                                    elapsed_ms=_ms(t0), note="cap exceeded", release=reservation)
                                raise ReadThresholdExceeded(
                                    received, self._nonweight_total(), self.log.threshold_bytes,
                                    f"{logged_path}: response body exceeded the cap of {cap} bytes "
                                    f"(received {received}); disk=0 reason=bounded read")
                        body = b"".join(chunks)
                        expected_status = {_FILE: 200, _RANGE: 206}.get(mode)
                        if (expected_status is not None and status != expected_status) or body_start != start:
                            self.log.record(
                                event="fetch", source=_SOURCE, path=logged_path, url=final_url,
                                start=body_start, nbytes=received, status=status, attempt=attempt,
                                elapsed_ms=_ms(t0), note="unexpected response", release=reservation)
                            raise self._fail(NetworkError(
                                f"{logged_path}: unexpected response (status {status}, body offset "
                                f"{body_start}, expected {expected_status} at {start})"), logged_path)
                        if exact is not None and received < exact:
                            raise _Retryable(f"short body: {received} of {exact} bytes", status)
                        self.log.record(
                            event="fetch", source=_SOURCE, path=logged_path, url=final_url, start=start,
                            nbytes=received, status=status, attempt=attempt, elapsed_ms=_ms(t0),
                            release=reservation)
                        return body
                    finally:
                        resp.close()
                except (httpx.TransportError, _Retryable) as exc:
                    last = exc
                    if isinstance(exc, _Retryable):
                        status = exc.status
                    self.log.record(
                        event="retry", source=_SOURCE, path=logged_path, url=final_url or url, start=start,
                        nbytes=received, status=status, attempt=attempt, elapsed_ms=_ms(t0),
                        note=f"{type(exc).__name__}: {exc}")
                    if attempt < self._attempts and self._backoff:
                        self._sleep(self._backoff[min(attempt - 1, len(self._backoff) - 1)])
            raise self._fail(NetworkError(f"{logged_path}: {self._attempts} attempts failed: {last}"),
                             logged_path)
        finally:
            self.log.release(reservation)

    def _follow(self, url: httpx.URL, headers: dict[str, str] | None) -> httpx.Response:
        """Send GET with manual redirects; returns the final (non-redirect) response, still streaming."""
        hops = 0
        while True:
            request = self._client.build_request("GET", url, headers=self._headers_for(url, headers))
            resp = self._client.send(request, stream=True, follow_redirects=False)
            if resp.status_code not in _REDIRECT_STATUSES:
                return resp
            try:
                loc = resp.headers.get("location")
                target = request.url.join(loc) if loc else None
                self._log_redirect_body(resp, request.url, target)
            finally:
                resp.close()
            hops += 1
            if target is None:
                raise self._fail(NetworkError(f"{request.url}: {resp.status_code} without Location"), None)
            if hops > MAX_REDIRECTS:
                raise self._fail(NetworkError(f"{url}: more than {MAX_REDIRECTS} redirects"), None)
            url = target

    def _log_redirect_body(self, resp: httpx.Response, req_url: httpx.URL, target: httpx.URL | None) -> None:
        t0 = time.monotonic()
        rr = self.log.preflight(REDIRECT_PATH, 0, REDIRECT_MAX_BYTES)
        note = f"-> {target.host if target is not None else '?'}"
        received = 0
        try:
            try:
                for chunk in resp.iter_raw():
                    received += len(chunk)
                    if received > REDIRECT_MAX_BYTES:
                        resp.close()
                        self.log.record(
                            event="fetch", source=_SOURCE, path=REDIRECT_PATH, url=str(req_url), start=None,
                            nbytes=received, status=resp.status_code, elapsed_ms=_ms(t0),
                            note=f"{note}; cap exceeded", release=rr)
                        raise ReadThresholdExceeded(
                            received, self._nonweight_total(), self.log.threshold_bytes,
                            f"{REDIRECT_PATH}: {resp.status_code} body exceeded {REDIRECT_MAX_BYTES} bytes "
                            f"(received {received}) at {req_url}; disk=0 reason=bounded read")
            except httpx.TransportError as exc:
                self.log.record(
                    event="fetch", source=_SOURCE, path=REDIRECT_PATH, url=str(req_url), start=None,
                    nbytes=received, status=resp.status_code, elapsed_ms=_ms(t0),
                    note=f"{note}; transport error: {exc}", release=rr)
                raise
            self.log.record(
                event="fetch", source=_SOURCE, path=REDIRECT_PATH, url=str(req_url), start=None,
                nbytes=received, status=resp.status_code, elapsed_ms=_ms(t0), note=note, release=rr)
        finally:
            self.log.release(rr)

    def _read_error_body(self, resp: httpx.Response, logged_path: str) -> bytes:
        t0 = time.monotonic()
        re_ = self.log.preflight(ERROR_PATH, 0, ERROR_BODY_MAX_BYTES)
        chunks: list[bytes] = []
        received = 0
        note = f"{resp.status_code} for {logged_path}"
        try:
            try:
                for chunk in resp.iter_raw():
                    chunks.append(chunk)
                    received += len(chunk)
                    if received >= ERROR_BODY_MAX_BYTES:
                        resp.close()
                        note += "; truncated at cap"
                        break
            except httpx.TransportError as exc:
                self.log.record(
                    event="error", source=_SOURCE, path=ERROR_PATH, url=str(resp.request.url), start=None,
                    nbytes=received, status=resp.status_code, elapsed_ms=_ms(t0),
                    note=f"{note}; transport error: {exc}", release=re_)
                raise
            self.log.record(
                event="error", source=_SOURCE, path=ERROR_PATH, url=str(resp.request.url), start=None,
                nbytes=received, status=resp.status_code, elapsed_ms=_ms(t0), note=note, release=re_)
        finally:
            self.log.release(re_)
        return b"".join(chunks)

    def _map_status(self, mode: str, status: int, headers: httpx.Headers, body: bytes,
                    logged_path: str) -> Exception:
        """Map a non-2xx final response to the exception to raise (_Retryable for 429/5xx)."""
        if mode == _API:
            exc = self._map_api_error(status, headers, body)
        elif status in (401, 403):
            exc = GatedRepoError(
                f"{self.repo} is gated or private: accept the terms at {self.endpoint}/{self.repo} "
                f"and set HF_TOKEN ({self._token_hint()})")
        elif status == 404:
            exc = ScoutError(f"missing file {logged_path}")
        elif status == 429 or status >= 500:
            exc = None
        else:
            exc = NetworkError(f"{logged_path}: unexpected HTTP status {status}")
        if exc is None:
            return _Retryable(f"HTTP {status}", status)
        return self._fail(exc, logged_path)

    def _map_api_error(self, status: int, headers: httpx.Headers, body: bytes) -> Exception | None:
        rev = self.requested_revision or "main"
        not_found = RepoNotFoundError(f"{self.repo}: not found or private ({self._token_hint()})")
        rev_not_found = RepoNotFoundError(f"{self.repo}@{rev}: revision not found")
        gated = GatedRepoError(
            f"{self.repo} is gated: accept the terms at {self.endpoint}/{self.repo} "
            f"and set HF_TOKEN ({self._token_hint()})")
        code = (headers.get("x-error-code") or "").strip()
        if code in ("RepoNotFound", "RepositoryNotFound"):
            return not_found
        if code == "RevisionNotFound":
            return rev_not_found
        if code == "GatedRepo":
            return gated
        text = body.decode("utf-8", errors="replace").lower()
        if "repository not found" in text:
            return not_found
        if "revision not found" in text:
            return rev_not_found
        if "gated" in text or "restricted" in text:
            return gated
        if status == 401:
            return not_found
        if status == 403:
            return gated
        if status == 404:
            return not_found
        if status == 429 or status >= 500:
            return None
        return HubAPIError(f"{self.repo}: API status {status}")

    def _range_ignored(self, resp: httpx.Response, logged_path: str, length: int, reservation: int,
                       t0: float) -> None:
        """Server answered 200 to a Range request: count what arrived from offset 0 and raise."""
        received = 0
        note = "Range ignored (200)"
        try:
            for chunk in resp.iter_raw():
                received += len(chunk)
                if received >= length:
                    break
        except httpx.TransportError as exc:
            note += f"; transport error: {exc}"
        finally:
            resp.close()
        self.log.record(
            event="fetch", source=_SOURCE, path=logged_path, url=str(resp.request.url), start=0,
            nbytes=received, status=resp.status_code, elapsed_ms=_ms(t0), note=note, release=reservation)
        raise self._fail(RangeNotSupported(
            f"{logged_path}: server ignored the Range request (200); {received} bytes counted from offset 0"),
            logged_path)

    @staticmethod
    def _content_range_start(resp: httpx.Response, start: int | None) -> int | None:
        m = _CONTENT_RANGE_RE.match(resp.headers.get("content-range") or "")
        return int(m.group(1)) if m else start

    def _nonweight_total(self) -> int:
        t = self.log.totals
        return t["meta"] + t["header"]


def _ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000.0
