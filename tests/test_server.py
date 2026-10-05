import threading
import time
from types import SimpleNamespace

import httpx
import pytest

import scout.server as srv
from scout.server import make_server
from tests.helpers.fakehub import HUB, FakeHub
from tests.helpers.st_fixtures import write_dense_repo, write_pipeline_repo

REPO = "acme/tiny"
SHA = "b" * 40


@pytest.fixture
def run_server(tmp_path):
    started = []

    def start(hub=None):
        kw = {}
        if hub is not None:
            kw = {"client_factory": lambda: hub.client(), "endpoint": HUB}
        server = make_server(port=0, out_dir=tmp_path / "out", **kw)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        started.append((server, t))
        return SimpleNamespace(server=server, url=f"http://127.0.0.1:{server.server_port}")

    yield start
    for server, t in started:
        server.shutdown()
        server.server_close()
        t.join(5)


def _post(url, target):
    return httpx.post(f"{url}/api/scans", json={"target": target})


def _wait(url, scan_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = httpx.get(f"{url}/api/scans/{scan_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError("scan did not finish")


def _hub(tmp_path, writer, **kw):
    root = tmp_path / "repo"
    writer(root, **kw)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA)
    return hub


def test_scan_local_roundtrip(tmp_path, run_server):
    root = tmp_path / "local"
    write_dense_repo(root)
    s = run_server()
    r = _post(s.url, str(root))
    assert r.status_code == 202
    assert r.headers["content-type"] == "application/json; charset=utf-8"
    assert r.headers["cache-control"] == "no-store"
    body = _wait(s.url, r.json()["scan_id"])
    assert body["status"] == "done", body["error"]
    assert len(body["views"]) == 1
    assert body["views"][0]["summary"]["weight_bytes_read"] == 0
    assert body["elapsed_s"] is not None
    seqs = [e["seq"] for e in body["events"]]
    assert seqs and seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert body["totals"]["weight"] == 0


def test_since(tmp_path, run_server):
    root = tmp_path / "local"
    write_dense_repo(root)
    s = run_server()
    body = _wait(s.url, _post(s.url, str(root)).json()["scan_id"])
    last = body["events"][-1]["seq"]
    again = httpx.get(f"{s.url}/api/scans/{body['scan_id']}", params={"since": last}).json()
    assert again["events"] == []
    mid = body["events"][1]["seq"]
    part = httpx.get(f"{s.url}/api/scans/{body['scan_id']}", params={"since": mid}).json()
    assert [e["seq"] for e in part["events"]] == [x["seq"] for x in body["events"] if x["seq"] > mid]


def test_hub_pipeline(tmp_path, run_server):
    hub = _hub(tmp_path, write_pipeline_repo)
    s = run_server(hub)
    body = _wait(s.url, _post(s.url, REPO).json()["scan_id"])
    assert body["status"] == "done", body["error"]
    assert len(body["views"]) == 3


def test_error_status(tmp_path, run_server):
    root = tmp_path / "repo"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, sha=SHA, gated=True, token="tok")
    s = run_server(hub)
    body = _wait(s.url, _post(s.url, REPO).json()["scan_id"])
    assert body["status"] == "error"
    assert body["error"]["type"] == "GatedRepoError"
    assert body["views"] is None
    assert body["elapsed_s"] is not None


def test_bad_target(run_server):
    s = run_server()
    r = _post(s.url, "nope")
    assert r.status_code == 400
    assert "error" in r.json()
    assert httpx.post(f"{s.url}/api/scans", json={"target": ""}).status_code == 400
    assert httpx.post(f"{s.url}/api/scans", json={"target": 5}).status_code == 400
    assert httpx.post(f"{s.url}/api/scans", content=b"not json").status_code == 400
    assert httpx.post(f"{s.url}/api/scans", json=["x"]).status_code == 400


def test_static(tmp_path, run_server, monkeypatch):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>hi</html>", encoding="utf-8")
    (web / "app.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(srv, "WEB_DIR", web)
    s = run_server()
    r = httpx.get(f"{s.url}/")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/html; charset=utf-8"
    assert r.text == "<html>hi</html>"
    r = httpx.get(f"{s.url}/app.js")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/javascript; charset=utf-8"


def test_static_missing_and_unknown(tmp_path, run_server, monkeypatch):
    monkeypatch.setattr(srv, "WEB_DIR", tmp_path / "empty")
    s = run_server()
    r = httpx.get(f"{s.url}/")
    assert r.status_code == 404 and r.json() == {"error": "not found"}
    assert httpx.get(f"{s.url}/nope").status_code == 404
    assert httpx.get(f"{s.url}/api/scans/unknown").json() == {"error": "not found"}
    assert httpx.get(f"{s.url}/api/scans/unknown").status_code == 404
    assert httpx.post(f"{s.url}/other", json={}).status_code == 404


class _SlowTransport(httpx.BaseTransport):
    def __init__(self, inner):
        self._inner = inner

    def handle_request(self, request):
        time.sleep(0.03)
        return self._inner.handle_request(request)


def test_live_log_while_running(tmp_path, monkeypatch):
    hub = _hub(tmp_path, write_pipeline_repo)
    clients = []

    def factory():
        c = httpx.Client(transport=_SlowTransport(hub.transport()), follow_redirects=True)
        clients.append(c)
        return c

    loads = []
    real_load = srv.load_card

    def spy(path):
        loads.append(path)
        return real_load(path)

    monkeypatch.setattr(srv, "load_card", spy)
    server = make_server(port=0, out_dir=tmp_path / "out", client_factory=factory, endpoint=HUB)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        scan_id = _post(url, REPO).json()["scan_id"]
        seqs, last, saw_running = [], 0, False
        deadline = time.monotonic() + 20
        while True:
            assert time.monotonic() < deadline
            body = httpx.get(f"{url}/api/scans/{scan_id}", params={"since": last}).json()
            seqs.extend(e["seq"] for e in body["events"])
            if body["events"]:
                last = body["events"][-1]["seq"]
            if body["status"] == "running":
                if body["views"] is None and body["elapsed_s"] is not None and seqs:
                    saw_running = True
            else:
                break
            time.sleep(0.01)
        assert body["status"] == "done", body["error"]
        assert saw_running
        final = httpx.get(f"{url}/api/scans/{scan_id}", params={"since": 0}).json()
        final_seqs = [e["seq"] for e in final["events"]]
        assert seqs == list(range(1, len(seqs) + 1))
        assert seqs == final_seqs
        assert len(loads) == len(final["views"]) == 3
        assert len(clients) == 1
        end = time.monotonic() + 5
        while not clients[0].is_closed and time.monotonic() < end:
            time.sleep(0.01)
        assert clients[0].is_closed
    finally:
        server.shutdown()
        server.server_close()
        t.join(5)
