"""Offline tests for scripts/pin_exit.py with a fake run_cmd (no network, no git, no subprocess)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts import pin_exit
from scripts.exit_expectations import DEFAULT_EXPECTATIONS

REPOS = sorted(DEFAULT_EXPECTATIONS)
UNPINNED = json.dumps({r: "UNPINNED" for r in REPOS}, indent=2, sort_keys=True) + "\n"


def _sha(repo: str) -> str:
    return format(abs(hash(repo)) % (16 ** 12), "012x").rjust(40, "0")


def _fake(scout=None, git=None, calls=None, insteadof=None):
    """run_cmd returning per-repo SHAs; scout/git map repo -> str (default: the agreeing _sha)."""
    scout = scout or {}
    git = git or {}

    def run_cmd(cmd: list[str]) -> str:
        if calls is not None:
            calls.append(list(cmd))
        if cmd[:2] == ["git", "config"]:
            assert cmd[2:] == ["--get-urlmatch", "url.insteadOf", "https://huggingface.co"]
            if insteadof is None:
                raise subprocess.CalledProcessError(1, cmd, output="", stderr="")
            return insteadof + "\n"
        if cmd[0] == "git":
            assert cmd[1:2] == ["ls-remote"] and cmd[3] == "refs/heads/main"
            repo = cmd[2][len(pin_exit.HF_GIT) + 1:]
            sha = git.get(repo, _sha(repo))
            return f"{sha}\trefs/heads/main\n" if sha else ""
        assert cmd[:4] == [sys.executable, "-m", "scout", "resolve"]
        return scout.get(cmd[4], _sha(cmd[4])) + "\n"
    return run_cmd


@pytest.fixture
def pins(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    p = tmp_path / "pins.json"
    p.write_text(UNPINNED)
    return p


def test_agree_writes_pins_and_evidence(pins, capsys):
    calls: list = []
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake(calls=calls)) == 0
    data = json.loads(pins.read_text())
    assert data == {r: _sha(r) for r in REPOS}
    assert list(data) == sorted(data)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["step"] == "PIN" and isinstance(line["hostname"], str) and line["hostname"]
    assert line["repos"] == {r: {"scout": _sha(r), "git_ls_remote": _sha(r)} for r in REPOS}
    assert line["git_url_insteadof"] == ""
    # every target was checked by both tools, against the fixed endpoint
    assert [c[2] for c in calls if c[:2] == ["git", "ls-remote"]] == [f"https://huggingface.co/{r}" for r in REPOS]
    assert [c[4] for c in calls if c[0] != "git"] == REPOS


def test_targets_come_from_expectations_not_pins_file(pins):
    pins.write_text(json.dumps({"some/other": "UNPINNED"}))
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake()) == 0
    assert sorted(json.loads(pins.read_text())) == REPOS


def test_disagree_returns_1_and_keeps_file(pins, capsys):
    bad = REPOS[1]
    rc = pin_exit.main(["--pins", str(pins)], run_cmd=_fake(git={bad: "f" * 40}))
    assert rc == 1
    assert pins.read_text() == UNPINNED
    assert f"PIN MISMATCH {bad}: scout={_sha(bad)} git_ls_remote={'f' * 40}" in capsys.readouterr().out


def test_git_empty_returns_1(pins):
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake(git={REPOS[0]: ""})) == 1
    assert pins.read_text() == UNPINNED


def test_non_hex_scout_returns_1(pins):
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake(scout={REPOS[2]: "main"},
                                                              git={REPOS[2]: "main"})) == 1
    assert pins.read_text() == UNPINNED


def test_called_process_error_returns_1(pins):
    def run_cmd(cmd):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")
    assert pin_exit.main(["--pins", str(pins)], run_cmd=run_cmd) == 1
    assert pins.read_text() == UNPINNED


def test_hf_endpoint_refused(pins, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.com")
    calls: list = []
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake(calls=calls)) == 1
    assert calls == [] and pins.read_text() == UNPINNED


def test_hf_endpoint_default_allowed(pins, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co/")
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake()) == 0


def test_git_main_sha_first_field():
    out = "0123456789abcdef0123456789abcdef01234567\trefs/heads/main\nffff\tother\n"
    assert pin_exit.git_main_sha("a/b", lambda cmd: out) == "0123456789abcdef0123456789abcdef01234567"
    assert pin_exit.git_main_sha("a/b", lambda cmd: "") == ""


def test_default_run_cmd_strips_hf_endpoint(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="x\n", stderr="")
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    monkeypatch.setattr(pin_exit.subprocess, "run", fake_run)
    assert pin_exit._default_run_cmd(["echo"]) == "x\n"
    assert "HF_ENDPOINT" not in seen["env"]
    assert seen["check"] is True and seen["capture_output"] is True and seen["text"] is True


def test_insteadof_recorded(pins, capsys):
    rule = "git@mirror.example.com:"
    assert pin_exit.main(["--pins", str(pins)], run_cmd=_fake(insteadof=rule)) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["git_url_insteadof"] == rule
