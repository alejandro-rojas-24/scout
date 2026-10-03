import httpx
import pytest

from tests.helpers.fakehub import CDN, HUB, FakeHub
from tests.helpers.st_fixtures import write_dense_repo

REPO = "Org/Tiny"
SHA = "b" * 40


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "repo"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA)
    return hub, root


def _st(root):
    return sorted(p.name for p in root.iterdir() if p.name.endswith(".safetensors"))[0]


def _url(path, rev=SHA, repo=REPO):
    return f"{HUB}/{repo}/resolve/{rev}/{path}"


def test_api_revision(env):
    hub, root = env
    c = hub.client()
    r = c.get(f"{HUB}/api/models/{REPO}/revision/main?blobs=true")
    assert r.status_code == 200
    j = r.json()
    assert j["id"] == REPO and j["sha"] == SHA and j["gated"] is False
    expected = sorted(p.name for p in root.iterdir() if p.is_file())
    assert [s["rfilename"] for s in j["siblings"]] == expected
    for s in j["siblings"]:
        assert s["size"] == (root / s["rfilename"]).stat().st_size
    assert c.get(f"{HUB}/api/models/{REPO}/revision/{SHA}").status_code == 200
    assert c.get(f"{HUB}/api/models/Org/Nope/revision/main").status_code == 401
    assert c.get(f"{HUB}/api/models/{REPO}/revision/zzz").status_code == 404


def test_range_via_redirect(env):
    hub, root = env
    name = _st(root)
    r = hub.client().get(_url(name), headers={"Range": "bytes=0-7", "Authorization": "Bearer x"})
    assert r.status_code == 206
    assert r.content == (root / name).read_bytes()[:8]
    assert len(hub.requests) == 2
    assert "authorization" not in hub.requests[1].headers
    assert hub.requests[1].url.host == "cdn.test"
    assert hub.cdn_reads == [(name, "bytes=0-7", 8)]


def test_redirect_body(env):
    hub, root = env
    name = _st(root)
    r = hub.client().get(_url(name), follow_redirects=False)
    assert r.status_code == 302
    assert len(r.content) == 18
    assert r.headers["location"] == f"{CDN}/{SHA}/{name}"


def test_gated(tmp_path):
    root = tmp_path / "g"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA, gated=True, token="tok")
    name = _st(root)
    c = hub.client()
    assert c.get(f"{HUB}/api/models/{REPO}/revision/main").json()["gated"] == "manual"
    assert c.get(_url(name), headers={"Range": "bytes=0-7"}).status_code == 401
    r = c.get(_url(name), headers={"Range": "bytes=0-7", "Authorization": "Bearer tok"})
    assert r.status_code == 206


def test_inject_reset(env):
    hub, root = env
    name = _st(root)
    hub.inject(name, "reset", times=1, after_bytes=5)
    c = hub.client()
    got = b""
    with pytest.raises(httpx.ReadError):
        with c.stream("GET", _url(name), headers={"Range": "bytes=0-31"}) as r:
            for chunk in r.iter_bytes():
                got += chunk
    assert len(got) == 5
    assert hub.cdn_reads[0] == (name, "bytes=0-31", 5)
    r2 = c.get(_url(name), headers={"Range": "bytes=0-31"})
    assert r2.status_code == 206 and len(r2.content) == 32


def test_inject_503(env):
    hub, root = env
    name = _st(root)
    hub.inject(name, "503")
    c = hub.client()
    assert c.get(_url(name), headers={"Range": "bytes=0-7"}).status_code == 503
    assert c.get(_url(name), headers={"Range": "bytes=0-7"}).status_code == 206


def test_inject_ignore_range(env):
    hub, root = env
    name = _st(root)
    hub.inject(name, "ignore_range")
    r = hub.client().get(_url(name), headers={"Range": "bytes=0-7"})
    assert r.status_code == 200
    assert len(r.content) == (root / name).stat().st_size


def test_relative307(tmp_path):
    root = tmp_path / "r"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA, meta_redirect="relative307")
    c = hub.client()
    r = c.get(_url("config.json"), follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].startswith("/api/resolve-cache/")
    assert len(r.content) == 18
    r2 = c.get(_url("config.json"))
    assert r2.status_code == 200
    assert r2.content == (root / "config.json").read_bytes()
    assert hub.cdn_reads == []


def test_error_codes(tmp_path):
    root = tmp_path / "e"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA)
    hub.add_repo("Org/Gated", root, sha=SHA, gated=True, token="t")
    c = hub.client()
    code = lambda r: r.headers.get("X-Error-Code")  # noqa: E731
    assert code(c.get(f"{HUB}/api/models/Org/Nope/revision/main")) == "RepoNotFound"
    assert code(c.get(f"{HUB}/api/models/{REPO}/revision/zzz")) == "RevisionNotFound"
    assert code(c.get(_url("config.json", repo="Org/Nope"))) == "RepoNotFound"
    assert code(c.get(_url("config.json", rev="zzz"))) == "RevisionNotFound"
    assert code(c.get(_url("missing.bin"))) == "EntryNotFound"
    r = c.get(_url("config.json", repo="Org/Gated"))
    assert r.status_code == 401 and code(r) == "GatedRepo"
