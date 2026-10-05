"""Static checks for the no-build frontend (T014, D1) and that the server serves it."""

import pathlib
import re
import threading

import httpx
import pytest

from scout.server import make_server

WEB = pathlib.Path(__file__).resolve().parent.parent / "scout" / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP = (WEB / "app.js").read_text(encoding="utf-8")

REQUIRED_IDS = ("target", "scan", "status", "counter-meta", "counter-header", "counter-weight", "log", "cards")
SVG_NS = "http://www.w3.org/2000/svg"


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_index_has_required_ids(element_id):
    assert re.search(r'\bid="%s"' % re.escape(element_id), INDEX), element_id


def test_app_talks_to_scan_api():
    assert "/api/scans" in APP
    assert "since=" in APP


@pytest.mark.parametrize("name,text", [("index.html", INDEX), ("app.js", APP)])
def test_no_external_urls(name, text):
    stripped = text.replace(SVG_NS, "")
    assert "http://" not in stripped, name
    assert "https://" not in stripped, name
    for frag in ('"//', "'//", "(//", "@import"):
        assert frag not in stripped, (name, frag)


def test_app_has_no_inner_html():
    assert "innerHTML" not in APP
    assert "outerHTML" not in APP
    assert "insertAdjacentHTML" not in APP


def test_app_renders_disclaimers():
    assert "disclaimers" in APP


def test_index_loads_app_and_honours_color_scheme():
    assert '<script src="app.js"' in INDEX
    assert "prefers-color-scheme" in INDEX


def test_app_has_no_imports():
    assert not re.search(r"^\s*import\s", APP, re.M)
    assert "require(" not in APP


@pytest.fixture
def served():
    server = make_server(port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    t.join(5)


def test_served_through_make_server(served):
    r = httpx.get(f"{served}/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="cards"' in r.text
    r = httpx.get(f"{served}/app.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/javascript")
    assert "/api/scans" in r.text
