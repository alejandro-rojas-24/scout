"""cli.py: scan / resolve / serve entry points, offline (FakeHub + tmp dirs)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scout.cli import main
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


def test_scan_hub_fake(tmp_path, capsys):
    hub = _hub(tmp_path)
    rc = main(["scan", REPO, "--out", str(tmp_path / "out"), "--endpoint", HUB],
              client_factory=hub.client)
    assert rc == 0
    err = capsys.readouterr().err
    for d in DISCLAIMERS:
        assert d in err


def test_resolve_fake(tmp_path, capsys):
    hub = _hub(tmp_path)
    assert main(["resolve", REPO, "--endpoint", HUB], client_factory=hub.client) == 0
    assert capsys.readouterr().out.strip() == "a" * 40


def test_exit_codes(tmp_path):
    out = str(tmp_path / "out")
    gated = _hub(tmp_path, gated=True, token="tok")
    assert main(["scan", REPO, "--out", out, "--endpoint", HUB], client_factory=gated.client) == 3

    root = tmp_path / "repo"
    shard = sorted(p.name for p in root.glob("*.safetensors"))[0]
    flaky = _hub(tmp_path)
    flaky.inject(shard, "reset", times=100)
    assert main(["scan", REPO, "--out", out, "--endpoint", HUB], client_factory=flaky.client) == 4

    binonly = tmp_path / "binonly"
    binonly.mkdir()
    (binonly / "pytorch_model.bin").write_bytes(b"x")
    (binonly / "config.json").write_text("{}")
    assert main(["scan", str(binonly), "--out", out]) == 1

    assert main(["scan", "not a target", "--out", out]) == 2


def test_python_m(tmp_path):
    root = tmp_path / "repo"
    write_dense_repo(root)
    proc = subprocess.run([sys.executable, "-m", "scout", "scan", str(root), "--out", str(tmp_path / "out")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
