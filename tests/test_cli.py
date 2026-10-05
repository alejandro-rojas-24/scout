"""cli.py: scan / resolve / serve entry points, offline (FakeHub + tmp dirs)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scout.cli import format_event, main
from scout.bytelog import ByteLog, Stage
from scout.view import DISCLAIMERS
from tests.helpers.fakehub import HUB, FakeHub
from tests.helpers.st_fixtures import write_dense_repo

REPO = "org/m"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def _hub(tmp_path, **kw):
    root = tmp_path / "repo"
    write_dense_repo(root)
    hub = FakeHub()
    hub.add_repo(REPO, root, **kw)
    return hub


def test_scan_local(tmp_path, capsys):
    root = tmp_path / "repo"
    write_dense_repo(root)
    out = tmp_path / "out"
    assert main(["scan", str(root), "--out", str(out)]) == 0
    cap = capsys.readouterr()
    data = json.loads(cap.out)
    assert data["totals"]["weight"] == 0
    assert len(data["cards"]) == 1
    import pathlib
    assert pathlib.Path(data["cards"][0]).exists()
    pos = -1
    for stage in ("[RESOLVE", "[HEADERS", "[META", "[REPORT"):
        nxt = cap.err.find(stage, pos + 1)
        assert nxt > pos, f"{stage} missing or out of order"
        pos = nxt
    fetch = [l for l in cap.err.splitlines() if l.startswith("[") and " fetch " in l]
    assert fetch and all("weight=0" in l for l in fetch)


def test_scan_hub_fake(tmp_path, capsys):
    hub = _hub(tmp_path)
    rc = main(["scan", REPO, "--out", str(tmp_path / "out"), "--endpoint", HUB],
              client_factory=hub.client)
    assert rc == 0
    cap = capsys.readouterr()
    err = cap.err
    for d in DISCLAIMERS:
        assert f"note: {d}" in err
    assert json.loads(cap.out)["totals"]["weight"] == 0


def test_resolve_fake(tmp_path, capsys):
    hub = _hub(tmp_path)
    assert main(["resolve", REPO, "--endpoint", HUB], client_factory=hub.client) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == "a" * 40
    assert "[RESOLVE" in cap.err


def test_exit_codes(tmp_path, capsys):
    out = str(tmp_path / "out")

    def err_of():
        return capsys.readouterr().err

    gated = _hub(tmp_path, gated=True, token="tok")
    assert main(["scan", REPO, "--out", out, "--endpoint", HUB], client_factory=gated.client) == 3
    assert "error: GatedRepoError:" in err_of()

    root = tmp_path / "repo"
    shard = sorted(p.name for p in root.glob("*.safetensors"))[0]
    flaky = _hub(tmp_path)
    flaky.inject(shard, "reset", times=100)
    assert main(["scan", REPO, "--out", out, "--endpoint", HUB], client_factory=flaky.client) == 4
    assert "error: NetworkError:" in err_of()

    binonly = tmp_path / "binonly"
    binonly.mkdir()
    (binonly / "pytorch_model.bin").write_bytes(b"x")
    (binonly / "config.json").write_text("{}")
    assert main(["scan", str(binonly), "--out", out]) == 1
    assert "error: NoSafetensorsError:" in err_of()

    assert main(["scan", "not a target", "--out", out]) == 2
    assert "error: ValueError:" in err_of()
    assert main(["resolve", "not a target"]) == 2
    assert "error: ValueError:" in err_of()


def test_serve_defaults(monkeypatch, capsys):
    import pathlib
    import scout.server
    monkeypatch.delenv("SCOUT_CARDS_DIR", raising=False)
    calls = []

    def fake(*a):
        calls.append(a)
        raise KeyboardInterrupt

    monkeypatch.setattr(scout.server, "serve", fake)
    assert main(["serve"]) == 0
    assert calls == [("127.0.0.1", 8765, pathlib.Path("cards"), 67108864, None)]
    assert "scout serving on http://127.0.0.1:8765" in capsys.readouterr().err


def test_format_event():
    seen = []
    log = ByteLog(on_event=seen.append)
    log.stage(Stage.RESOLVE)
    e = seen[0]
    t = e.totals
    expect = (f"[{e.stage:<8}] {e.event:<12} {e.path or '-'} +{e.bytes}B  "
              f"meta={t['meta']} header={t['header']} weight={t['weight']}")
    if e.note:
        expect += f"  ({e.note})"
    assert format_event(e) == expect
    assert format_event(e).startswith("[RESOLVE ] ")
    assert "meta=0 header=0 weight=0" in format_event(e)


def test_python_m(tmp_path):
    root = tmp_path / "repo"
    write_dense_repo(root)
    proc = subprocess.run([sys.executable, "-m", "scout", "scan", str(root), "--out", str(tmp_path / "out")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
