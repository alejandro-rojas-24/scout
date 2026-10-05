"""Phase 1 exit check E1-E3 against huggingface.co (network; deselected by default via addopts)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.network
def test_exit_check():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "exit_check.py"),
                           "--pins", str(ROOT / "exit" / "pins.json")], cwd=ROOT)
    assert proc.returncode == 0
