"""FakeHub: httpx.MockTransport emulating the HF Hub API, resolve redirects, CDN ranges and failures."""
from __future__ import annotations

import pathlib
import re
import threading

import httpx

HUB = "https://hub.test"
CDN = "https://cdn.test"

_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")


class _ResetStream(httpx.SyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self):
        if self._data:
            yield self._data
        raise httpx.ReadError("connection reset")


class _Repo:
    def __init__(self, repo_id, root, sha, gated, token, meta_redirect):
        self.repo_id = repo_id
        self.root = pathlib.Path(root)
        self.sha = sha
        self.gated = gated
        self.token = token
        self.meta_redirect = meta_redirect

    def files(self) -> dict[str, pathlib.Path]:
        out = {}
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            out[rel.as_posix()] = p
        return out


def _json(status: int, body: dict, code: str | None = None) -> httpx.Response:
    headers = {"X-Error-Code": code} if code else {}
    return httpx.Response(status, json=body, headers=headers)


class FakeHub:
    def __init__(self) -> None:
        self._repos: dict[str, _Repo] = {}
        self._injections: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self.requests: list[httpx.Request] = []
        self.cdn_reads: list[tuple[str, str | None, int]] = []

    def add_repo(self, repo_id: str, root: pathlib.Path, sha: str = "a" * 40,
                 gated: bool = False, token: str | None = None,
                 meta_redirect: str | None = None) -> None:
        self._repos[repo_id] = _Repo(repo_id, root, sha, gated, token, meta_redirect)

    def inject(self, path: str, mode: str, times: int = 1, after_bytes: int = 0) -> None:
        assert mode in {"reset", "503", "ignore_range"}
        self._injections.setdefault(path, []).append(
            {"mode": mode, "times": times, "after_bytes": after_bytes})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport(), follow_redirects=True)

    # ---- internals ----
    def _handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
        host = request.url.host
        segs = [s for s in request.url.path.split("/") if s != ""]
        if request.method != "GET":
            return httpx.Response(405)
        if host == "cdn.test":
            return self._cdn(request, segs)
        if host != "hub.test":
            return httpx.Response(404)
        if len(segs) >= 6 and segs[:2] == ["api", "models"] and segs[4] == "revision":
            return self._api_revision(f"{segs[2]}/{segs[3]}", "/".join(segs[5:]))
        if len(segs) >= 7 and segs[:3] == ["api", "resolve-cache", "models"]:
            return self._resolve_cache(f"{segs[3]}/{segs[4]}", segs[5], "/".join(segs[6:]), request)
        if len(segs) >= 4 and segs[2] == "resolve":
            return self._resolve(f"{segs[0]}/{segs[1]}", segs[3], "/".join(segs[4:]), request)
        return _json(404, {"error": "Not Found"})

    def _api_revision(self, repo_id: str, rev: str) -> httpx.Response:
        repo = self._repos.get(repo_id)
        if repo is None:
            return _json(401, {"error": "Repository Not Found"}, "RepoNotFound")
        if rev not in ("main", repo.sha):
            return _json(404, {"error": "Revision Not Found"}, "RevisionNotFound")
        files = repo.files()
        return _json(200, {
            "id": repo_id, "sha": repo.sha, "gated": "manual" if repo.gated else False,
            "siblings": [{"rfilename": rel, "size": files[rel].stat().st_size} for rel in sorted(files)],
        })

    def _resolve(self, repo_id: str, rev: str, path: str, request: httpx.Request) -> httpx.Response:
        repo = self._repos.get(repo_id)
        if repo is None:
            return _json(401, {"error": "Repository Not Found"}, "RepoNotFound")
        if rev != repo.sha:
            return _json(404, {"error": "Revision Not Found"}, "RevisionNotFound")
        files = repo.files()
        if path not in files:
            return _json(404, {"error": "Entry not found"}, "EntryNotFound")
        if repo.gated and request.headers.get("Authorization") != f"Bearer {repo.token}":
            return _json(401, {"error": "Access to model is restricted"}, "GatedRepo")
        if path.endswith(".safetensors"):
            return httpx.Response(302, headers={"Location": f"{CDN}/{repo.sha}/{path}"},
                                  content=b"Found. Redirecting")
        if repo.meta_redirect == "relative307":
            loc = f"/api/resolve-cache/models/{repo_id}/{repo.sha}/{path}"
            return httpx.Response(307, headers={"Location": loc}, content=b"Temporary Redirect")
        return self._serve(files[path].read_bytes(), request)

    def _resolve_cache(self, repo_id: str, sha: str, path: str, request: httpx.Request) -> httpx.Response:
        repo = self._repos.get(repo_id)
        if repo is None or repo.sha != sha:
            return httpx.Response(404)
        files = repo.files()
        if path not in files:
            return httpx.Response(404)
        return self._serve(files[path].read_bytes(), request)

    def _cdn(self, request: httpx.Request, segs: list[str]) -> httpx.Response:
        if len(segs) < 2:
            return httpx.Response(404)
        sha, path = segs[0], "/".join(segs[1:])
        data = None
        for repo in self._repos.values():
            if repo.sha == sha:
                files = repo.files()
                if path in files:
                    data = files[path].read_bytes()
                    break
        if data is None:
            return httpx.Response(404)
        rng = request.headers.get("Range")
        inj = self._take_injection(path)
        if inj is not None and inj["mode"] == "503":
            with self._lock:
                self.cdn_reads.append((path, rng, 0))
            return httpx.Response(503, content=b"")
        if inj is not None and inj["mode"] == "ignore_range":
            with self._lock:
                self.cdn_reads.append((path, rng, len(data)))
            return httpx.Response(200, content=data)
        status, headers, body = self._range(data, rng)
        if inj is not None and inj["mode"] == "reset":
            sent = body[:inj["after_bytes"]]
            with self._lock:
                self.cdn_reads.append((path, rng, len(sent)))
            headers = dict(headers)
            headers["Content-Length"] = str(len(body))
            return httpx.Response(206, headers=headers, stream=_ResetStream(sent))
        with self._lock:
            self.cdn_reads.append((path, rng, len(body)))
        return httpx.Response(status, headers=headers, content=body)

    def _take_injection(self, path: str) -> dict | None:
        with self._lock:
            lst = self._injections.get(path)
            if not lst:
                return None
            inj = lst[0]
            inj["times"] -= 1
            if inj["times"] <= 0:
                lst.pop(0)
            return dict(inj)

    @staticmethod
    def _range(data: bytes, rng: str | None) -> tuple[int, dict, bytes]:
        m = _RANGE_RE.match(rng) if rng else None
        if not m:
            return 200, {}, data
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else len(data) - 1
        if a >= len(data):
            return 416, {"Content-Range": f"bytes */{len(data)}"}, b""
        body = data[a:b + 1]
        return 206, {"Content-Range": f"bytes {a}-{min(b, len(data) - 1)}/{len(data)}"}, body

    def _serve(self, data: bytes, request: httpx.Request) -> httpx.Response:
        status, headers, body = self._range(data, request.headers.get("Range"))
        return httpx.Response(status, headers=headers, content=body)
