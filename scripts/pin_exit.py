"""Pin the Phase 1 exit targets: `scout resolve` and plain `git ls-remote` must agree on a 40-hex SHA.

Run once at EXIT on a machine that can reach huggingface.co (plan.md section 2, "Pins"):

    python scripts/pin_exit.py --pins exit/pins.json

Writes the pins file only when both tools agree for every target, then prints one JSON evidence
line {"step": "PIN", ...} that the operator appends to plans/phase-1/log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts.exit_expectations import DEFAULT_EXPECTATIONS, HF_ENDPOINT_URL  # noqa: E402

HF_GIT = "https://huggingface.co"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}
    env["GIT_TERMINAL_PROMPT"] = "0"  # a missing/gated repo must fail, never prompt for credentials
    return env


def _default_run_cmd(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=_clean_env(),
                          cwd=str(ROOT)).stdout


def git_main_sha(repo: str, run_cmd: Callable[[list[str]], str]) -> str:
    out = run_cmd(["git", "ls-remote", f"{HF_GIT}/{repo}", "refs/heads/main"])
    lines = out.splitlines()
    fields = lines[0].split() if lines else []
    return fields[0] if fields else ""


def scout_sha(repo: str, run_cmd: Callable[[list[str]], str]) -> str:
    return run_cmd([sys.executable, "-m", "scout", "resolve", repo]).strip()


def git_insteadof(run_cmd: Callable[[list[str]], str]) -> str:
    """`git config --get-urlmatch url.insteadOf https://huggingface.co` ("" if none: git exits 1)."""
    try:
        return run_cmd(["git", "config", "--get-urlmatch", "url.insteadOf", HF_GIT]).strip()
    except subprocess.CalledProcessError:
        return ""


def _endpoint_refused() -> str | None:
    value = os.environ.get("HF_ENDPOINT")
    if value and value.removesuffix("/") != HF_ENDPOINT_URL:
        return value
    return None


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None, *, run_cmd: Callable[[list[str]], str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pin the Phase 1 exit targets (scout resolve == git ls-remote).")
    p.add_argument("--pins", type=Path, default=ROOT / "exit" / "pins.json")
    args = p.parse_args(argv)
    run_cmd = run_cmd or _default_run_cmd

    bad = _endpoint_refused()
    if bad is not None:
        print(f"HF_ENDPOINT={bad} is not allowed for pinning (expected {HF_ENDPOINT_URL})")
        return 1

    pins: dict[str, str] = {}
    evidence: dict[str, dict[str, str]] = {}
    for repo in sorted(DEFAULT_EXPECTATIONS):
        try:
            s = scout_sha(repo, run_cmd)
            g = git_main_sha(repo, run_cmd)
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", None)
            print(f"PIN ERROR {repo}: {type(exc).__name__}: {exc}" + (f"\n{detail}" if detail else ""))
            return 1
        if not (_SHA_RE.fullmatch(s) and _SHA_RE.fullmatch(g)) or s != g:
            print(f"PIN MISMATCH {repo}: scout={s} git_ls_remote={g}")
            return 1
        pins[repo] = s
        evidence[repo] = {"scout": s, "git_ls_remote": g}

    try:
        insteadof = git_insteadof(run_cmd)
    except OSError as exc:
        print(f"PIN ERROR git config: {type(exc).__name__}: {exc}")
        return 1
    _write_atomic(args.pins, pins)
    print(json.dumps({"step": "PIN", "hostname": socket.gethostname(), "repos": evidence,
                      "git_url_insteadof": insteadof}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
