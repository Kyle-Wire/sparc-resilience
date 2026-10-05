"""``python -m sparc.studio.selfcheck`` passes on this machine (real server, run, engine, export)."""

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_selfcheck_passes(tmp_path):
    r = subprocess.run([sys.executable, "-m", "sparc.studio.selfcheck", "--dir", str(tmp_path / "sc")],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "OK: 8/8 steps passed" in r.stdout
