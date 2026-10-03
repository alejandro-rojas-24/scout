"""HubSource against FakeHub: pinned resolve, manual redirects, ranged reads, retries, honest counting."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

import scout.bytelog
import scout.hub
from scout.bytelog import ByteLog
from scout.errors import (
    GatedRepoError,
    HubAPIError,
    NetworkError,
    RangeNotSupported,
    ReadThresholdExceeded,
    RepoNotFoundError,
    RevisionMismatch,
    WeightReadRefused,
)
from scout.hub import API_PATH, ERROR_PATH, REDIRECT_PATH, HubSource
from tests.helpers.fakehub import CDN, HUB, FakeHub
from tests.helpers.st_fixtures import write_dense_repo

REPO = "org/m"
SHA = "a" * 40
REDIRECT_BODY_LEN = len(b"Found. Redirecting")  # 18, same as b"Temporary Redirect"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def _setup(tmp_path, **repo_kw):
    root = tmp_path / "repo"
    info = write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA, **repo_kw)
    shard = info["files"][0]
    return SimpleNamespace(hub=hub, root=root, info=info, shard=shard, N=info["header_lens"][shard])


@pytest.fixture
def env(tmp_path):
    return _setup(tmp_path)


def _src(hub, rev=None, log=None, client=None, **kw):
    log = log if log is not None else ByteLog()
    kw.setdefault("sleep", lambda s: None)
    src = HubSource(REPO, rev, log, client=client if client is not None else hub.client(),
                    endpoint=HUB, **kw)
    return src, log


def _wrap(hub, fn):
    """Client whose transport passes FakeHub responses through fn(request, response) -> response | None."""
    inner = hub.transport()

    def handler(request):
        resp = inner.handle_request(request)
        resp.read()
        out = fn(request, resp)
        if out is not None:
            return out
        return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _is_api(request):
    return request.url.path.startswith("/api/models/")


def _events(log, event=None, path=None, since=0):
    return [e for e in log.events_since(since)
            if (event is None or e.event == event) and (path is None or e.path == path)]


# ---------------------------------------------------------------------------- resolve

def test_resolve_pins_sha(env):
    src, log = _src(env.hub)
    src.resolve()
    assert src.revision_sha == SHA
    assert src.requested_revision is None
    assert src.kind == "hub" and src.revision_kind == "git" and src.local_path is None
    assert src.repo == REPO and src.endpoint == HUB
    expected = sorted(p.relative_to(env.root).as_posix() for p in env.root.rglob("*") if p.is_file())
    files = src.files()
    assert [f.path for f in files] == expected
    for f in files:
        assert f.size == (env.root / f.path).stat().st_size
    api = _events(log, "fetch", API_PATH)
    assert len(api) == 1
    assert api[0].bytes > 0 and api[0].bytes_by_class["meta"] == api[0].bytes
    assert log.totals == {"meta": api[0].bytes, "header": 0, "weight": 0}
    assert log.reserved == 0


def test_unknown_revision(env):
    src, log = _src(env.hub, rev="b" * 40)
    with pytest.raises(RepoNotFoundError, match="revision not found"):
        src.resolve()
    assert log.reserved == 0


def test_revision_mismatch(env):
    def fn(req, resp):
        if _is_api(req) and resp.status_code == 200:
            body = resp.json()
            body["sha"] = "c" * 40
            return httpx.Response(200, json=body)
        return None

    src, log = _src(env.hub, rev=SHA, client=_wrap(env.hub, fn))
    with pytest.raises(RevisionMismatch):
        src.resolve()
    assert log.reserved == 0


def test_repo_not_found(env):
    src = HubSource("org/nope", None, ByteLog(), client=env.hub.client(), endpoint=HUB,
                    sleep=lambda s: None)
    with pytest.raises(RepoNotFoundError, match="HF_TOKEN not set"):
        src.resolve()


def test_invalid_repo_id_no_network(env):
    for bad in ("nope", "/x/y", "a/b/c", "", "-x/y y"):
        with pytest.raises(ValueError):
            HubSource(bad, None, ByteLog(), client=env.hub.client(), endpoint=HUB)
    assert env.hub.requests == []


def test_error_mapping(env):
    def case1(req, resp):  # header kept, body replaced
        if _is_api(req):
            return httpx.Response(401, json={"error": "nope"}, headers={"X-Error-Code": "RepoNotFound"})

    def case2(req, resp):  # header dropped, body lowercased
        if _is_api(req):
            return httpx.Response(404, content=b"repository not found")

    def case3(req, resp):
        if _is_api(req):
            return httpx.Response(400, content=b"")

    def case4(req, resp):
        if _is_api(req):
            return httpx.Response(200, json={"id": "x"})

    def case5(req, resp):  # not JSON
        if _is_api(req):
            return httpx.Response(200, content=b"<html>")

    def case6(req, resp):  # gated via header
        if _is_api(req):
            return httpx.Response(403, content=b"x", headers={"X-Error-Code": "GatedRepo"})

    cases = [(case1, RepoNotFoundError), (case2, RepoNotFoundError), (case3, HubAPIError),
             (case4, HubAPIError), (case5, HubAPIError), (case6, GatedRepoError)]
    for fn, exc in cases:
        src, log = _src(env.hub, client=_wrap(env.hub, fn))
        try:
            src.resolve()
        except KeyError:  # pragma: no cover - the point of the test
            pytest.fail("bare KeyError")
        except exc:
            pass
        else:  # pragma: no cover
            pytest.fail(f"{fn.__name__} did not raise {exc.__name__}")
        assert log.reserved == 0


def test_error_body_logged(env):
    body = env.hub.client().get(f"{HUB}/api/models/org/nope/revision/main").content
    assert body
    log = ByteLog()
    src = HubSource("org/nope", None, log, client=env.hub.client(), endpoint=HUB, sleep=lambda s: None)
    before = log.totals["meta"]
    with pytest.raises(RepoNotFoundError):
        src.resolve()
    errs = _events(log, "error", ERROR_PATH)
    assert len(errs) == 1
    assert errs[0].bytes == len(body) and errs[0].status == 401
    assert log.totals["meta"] - before == len(body)
    assert log.totals["header"] == 0 and log.totals["weight"] == 0
    assert log.reserved == 0

    # A 503 with a body on a shard read: logged under @error (meta), never header.
    state = {"n": 0}

    def fn(req, resp):
        if req.url.host == "cdn.test" and state["n"] == 0:
            state["n"] += 1
            return httpx.Response(503, content=b"Service Unavailable")

    src, log = _src(env.hub, client=_wrap(env.hub, fn))
    src.resolve()
    meta0 = log.totals["meta"]
    assert src.read_range(env.shard, 0, 8) == (env.root / env.shard).read_bytes()[:8]
    errs = _events(log, "error", ERROR_PATH)
    assert len(errs) == 1 and errs[0].bytes == len(b"Service Unavailable")
    assert errs[0].bytes_by_class["meta"] == errs[0].bytes
    assert log.totals["header"] == 8
    assert log.totals["weight"] == 0
    assert log.totals["meta"] - meta0 == len(b"Service Unavailable") + 2 * REDIRECT_BODY_LEN
    assert log.reserved == 0


# ---------------------------------------------------------------------------- ranged reads

def test_header_ranges(env):
    src, log = _src(env.hub)
    src.resolve()
    seq = log.events[-1].seq
    data = (env.root / env.shard).read_bytes()
    assert src.read_range(env.shard, 0, 8) == data[:8]
    assert int.from_bytes(data[:8], "little") == env.N
    log.set_header_len(env.shard, env.N)
    assert src.read_range(env.shard, 8, env.N) == data[8:8 + env.N]
    t = log.totals
    assert t["header"] == 8 + env.N
    assert t["weight"] == 0
    for path, rng, _ in env.hub.cdn_reads:
        assert path == env.shard
        end = int(rng.split("-")[1])
        assert end + 1 <= 8 + env.N
    redirects = _events(log, "fetch", REDIRECT_PATH, since=seq)
    assert len(redirects) == 2
    for e in redirects:
        assert e.bytes == REDIRECT_BODY_LEN and e.bytes_by_class["meta"] == REDIRECT_BODY_LEN
        assert e.note == "-> cdn.test"
    fetches = _events(log, "fetch", env.shard)
    assert [e.status for e in fetches] == [206, 206]
    assert [e.range for e in fetches] == [(0, 8), (8, 8 + env.N)]
    for e in log.events:
        assert e.bytes_by_class["weight"] == 0
    assert log.reserved == 0


def test_manual_redirect_auth(env):
    src, _ = _src(env.hub, token="hf_x")
    src.resolve()
    n0 = len(env.hub.requests)
    src.read_range(env.shard, 0, 8)
    reqs = env.hub.requests
    assert all(r.headers.get("authorization") == "Bearer hf_x" for r in reqs if r.url.host == "hub.test")
    assert all("authorization" not in r.headers for r in reqs if r.url.host == "cdn.test")
    hops = reqs[n0:]
    assert [r.url.host for r in hops] == ["hub.test", "cdn.test"]
    assert all(r.headers.get("range") == "bytes=0-7" for r in hops)
    assert all(r.headers.get("accept-encoding") == "identity" for r in reqs)
    assert all(r.headers.get("user-agent", "").startswith("scout/") for r in reqs)
    assert f"/resolve/{SHA}/" in str(hops[0].url)


def test_weight_refused_no_request(env):
    src, log = _src(env.hub)
    src.resolve()
    log.set_header_len(env.shard, env.N)
    n0 = len(env.hub.requests)
    with pytest.raises(WeightReadRefused):
        src.read_range(env.shard, 8 + env.N, 1)
    assert len(env.hub.requests) == n0
    with pytest.raises(WeightReadRefused):
        src.read_file(env.shard)
    assert len(env.hub.requests) == n0
    assert log.totals["weight"] == 0 and log.reserved == 0


def test_retry_mid_read(env):
    src, log = _src(env.hub)
    src.resolve()
    src.read_range(env.shard, 0, 8)
    log.set_header_len(env.shard, env.N)
    seq = log.events[-1].seq
    env.hub.inject(env.shard, "reset", times=1, after_bytes=3)
    data = src.read_range(env.shard, 8, env.N)
    assert data == (env.root / env.shard).read_bytes()[8:8 + env.N]
    retries = _events(log, "retry", since=seq)
    assert len(retries) == 1
    assert retries[0].bytes == 3 and retries[0].range == (8, 11) and retries[0].attempt == 1
    assert retries[0].bytes_by_class["header"] == 3
    assert len(_events(log, "fetch", env.shard, since=seq)) == 1
    assert log.totals["header"] == 8 + env.N + 3
    assert log.totals["weight"] == 0
    assert log.reserved == 0


def test_retry_exhausted(env):
    sleeps = []
    src, log = _src(env.hub, sleep=sleeps.append)
    src.resolve()
    seq = log.events[-1].seq
    env.hub.inject(env.shard, "reset", times=5, after_bytes=2)
    with pytest.raises(NetworkError, match="3 attempts failed"):
        src.read_range(env.shard, 0, 8)
    assert len(_events(log, "retry", since=seq)) == 3
    assert [e.attempt for e in _events(log, "retry", since=seq)] == [1, 2, 3]
    assert len(_events(log, "error", since=seq)) == 1
    assert sleeps == [0.5, 1.0]
    assert log.totals["header"] == 6
    assert log.reserved == 0


def test_503_then_ok(env):
    src, log = _src(env.hub)
    src.resolve()
    seq = log.events[-1].seq
    env.hub.inject(env.shard, "503", times=2)
    assert src.read_range(env.shard, 0, 8) == (env.root / env.shard).read_bytes()[:8]
    retries = _events(log, "retry", since=seq)
    assert [(e.status, e.attempt) for e in retries] == [(503, 1), (503, 2)]
    fetch = _events(log, "fetch", env.shard, since=seq)
    assert len(fetch) == 1 and fetch[0].attempt == 3
    assert log.reserved == 0


def test_range_ignored(env):
    src, log = _src(env.hub)
    src.resolve()
    src.read_range(env.shard, 0, 8)
    log.set_header_len(env.shard, env.N)
    size = (env.root / env.shard).stat().st_size
    assert size > 8 + env.N
    env.hub.inject(env.shard, "ignore_range")
    n_retry = len(_events(log, "retry"))
    with pytest.raises(RangeNotSupported):
        src.read_range(env.shard, 8, env.N)
    ev = _events(log, "fetch", env.shard)[-1]
    assert ev.range[0] == 0
    assert ev.status == 200
    assert ev.bytes == size
    assert log.totals["weight"] > 0
    assert len(_events(log, "retry")) == n_retry
    assert log.reserved == 0


# ---------------------------------------------------------------------------- whole-file reads

def test_read_file_and_missing(env):
    src, log = _src(env.hub)
    src.resolve()
    assert src.read_file("config.json") == (env.root / "config.json").read_bytes()
    assert src.read_file("nope.json") is None
    assert log.reserved == 0


def test_cap(env, monkeypatch):
    def fn(req, resp):
        if _is_api(req) and resp.status_code == 200:
            body = resp.json()
            for s in body["siblings"]:
                if s["rfilename"] == "config.json":
                    del s["size"]
            return httpx.Response(200, json=body)

    src, log = _src(env.hub, log=ByteLog(threshold_bytes=64 * 1024 * 1024), client=_wrap(env.hub, fn))
    src.resolve()
    assert {f.path: f.size for f in src.files()}["config.json"] is None
    size = (env.root / "config.json").stat().st_size
    assert size > 100
    monkeypatch.setattr(scout.hub, "META_MAX_BYTES", 100)
    monkeypatch.setattr(scout.bytelog, "META_MAX_BYTES", 100)
    meta0 = log.totals["meta"]
    with pytest.raises(ReadThresholdExceeded):
        src.read_file("config.json")
    ev = _events(log, "fetch", "config.json")
    assert len(ev) == 1 and ev[0].bytes == size and ev[0].note == "cap exceeded"
    assert log.totals["meta"] - meta0 == size
    assert log.reserved == 0


def test_gated(tmp_path):
    env = _setup(tmp_path, gated=True, token="hf_x")
    src, log = _src(env.hub)
    src.resolve()
    with pytest.raises(GatedRepoError, match="HF_TOKEN not set"):
        src.read_file("config.json")
    assert len(_events(log, "error", ERROR_PATH)) == 1
    assert log.reserved == 0
    n_req = len(env.hub.requests)
    src2, _ = _src(env.hub, token="hf_x")
    src2.resolve()
    assert src2.read_file("config.json") == (env.root / "config.json").read_bytes()
    # no retry for the 401 earlier: exactly one resolve request was made for it
    assert n_req == 2


def test_relative307_meta(tmp_path):
    env = _setup(tmp_path, meta_redirect="relative307")
    src, log = _src(env.hub, token="hf_x")
    src.resolve()
    api_bytes = _events(log, "fetch", API_PATH)[0].bytes
    n0 = len(env.hub.requests)
    data = src.read_file("config.json")
    assert data == (env.root / "config.json").read_bytes()
    reds = _events(log, "fetch", REDIRECT_PATH)
    assert len(reds) == 1 and reds[0].bytes == REDIRECT_BODY_LEN and reds[0].status == 307
    assert reds[0].note == "-> hub.test"
    hop2 = env.hub.requests[n0 + 1]
    assert hop2.url.host == "hub.test"
    assert hop2.url.path.startswith("/api/resolve-cache/")
    assert hop2.headers.get("authorization") == "Bearer hf_x"  # same host as the endpoint
    assert log.totals["meta"] == REDIRECT_BODY_LEN + len(data) + api_bytes
    assert log.reserved == 0


def test_redirect_body_cap(env):
    big = b"x" * (70 * 1024)

    def fn(req, resp):
        if not _is_api(req):
            return httpx.Response(302, headers={"Location": f"{CDN}/x"}, content=big)

    src, log = _src(env.hub, client=_wrap(env.hub, fn))
    src.resolve()
    with pytest.raises(ReadThresholdExceeded):
        src.read_file("config.json")
    reds = _events(log, "fetch", REDIRECT_PATH)
    assert len(reds) == 1 and reds[0].bytes == len(big)
    assert log.reserved == 0


def test_redirect_loop(env):
    def fn(req, resp):
        if not _is_api(req):
            return httpx.Response(302, headers={"Location": str(req.url)}, content=b"loop")

    src, log = _src(env.hub, client=_wrap(env.hub, fn))
    src.resolve()
    with pytest.raises(NetworkError, match="redirects"):
        src.read_file("config.json")
    reds = _events(log, "fetch", REDIRECT_PATH)
    assert len(reds) == scout.hub.MAX_REDIRECTS + 1
    assert all(e.bytes == 4 for e in reds)
    assert log.reserved == 0


def test_content_encoding_refused(env):
    def fn(req, resp):
        if not _is_api(req):
            return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"abc")

    src, log = _src(env.hub, client=_wrap(env.hub, fn))
    src.resolve()
    with pytest.raises(NetworkError, match="Content-Encoding"):
        src.read_file("config.json")
    assert log.reserved == 0


def test_env_token(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    src, _ = _src(env.hub)
    src.resolve()
    assert env.hub.requests[0].headers.get("authorization") == "Bearer hf_env"
    monkeypatch.setenv("HF_TOKEN", "")
    n0 = len(env.hub.requests)
    src2, _ = _src(env.hub)
    src2.resolve()
    assert "authorization" not in env.hub.requests[n0].headers


def test_env_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.test/")
    src = HubSource(REPO, None, ByteLog())
    try:
        assert src.endpoint == "https://mirror.test"
    finally:
        src.close()
    monkeypatch.delenv("HF_ENDPOINT")
    src = HubSource(REPO, None, ByteLog())
    try:
        assert src.endpoint == scout.hub.DEFAULT_ENDPOINT == "https://huggingface.co"
    finally:
        src.close()


def test_close_only_owned_client(env):
    client = env.hub.client()
    src, _ = _src(env.hub, client=client)
    src.close()
    assert not client.is_closed
    own = HubSource(REPO, None, ByteLog(), endpoint=HUB)
    own.close()
    assert own._client.is_closed


def test_no_disk_writes(env, tmp_path):
    before = sorted(p for p in tmp_path.rglob("*"))
    src, _ = _src(env.hub)
    src.resolve()
    src.read_file("config.json")
    src.read_range(env.shard, 0, 8)
    assert sorted(p for p in tmp_path.rglob("*")) == before
    assert json.loads(src.read_file("config.json"))["model_type"] == "qwen3"
