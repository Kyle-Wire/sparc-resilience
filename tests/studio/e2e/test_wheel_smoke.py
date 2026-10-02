"""Wheel smoke test (SPEC §14.6 step 7): the built wheel serves the committed SPA with no Node and no checkout.

1. Build the wheel (``pip wheel . --no-deps``), or take ``$SPARC_SMOKE_WHEEL``.
2. It carries ``sparc/studio/static``: ``index.html``, ``BUILD_INFO.json`` and every hashed asset
   that ``index.html`` loads.
3. Install it into a clean virtual environment (no system site-packages). With
   ``SPARC_SMOKE_FULL=1`` (CI) the wheel is installed with its ``studio`` extra and every
   dependency, after ``$SPARC_SMOKE_PREINSTALL`` (pip arguments, e.g. the CPU build of torch);
   otherwise with ``--no-deps`` plus the few packages the server needs to start.
4. Run ``sparc studio --no-browser --port 0`` from outside the checkout: ``GET /`` returns
   ``index.html`` with hashed assets, each asset is served, and the server runs the installed
   package, not this checkout.

Needs network access for pip; marked ``slow`` and ``network`` (CI runs it as its own step).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from .conftest import ROOT, wait_for

pytestmark = [pytest.mark.slow, pytest.mark.network]
MINIMAL = ["fastapi>=0.111,!=0.136.3", "uvicorn>=0.29", "httpx>=0.27", "threadpoolctl>=3", "psutil>=5.9",
           "pyyaml>=6.0", "numpy>=1.24", "pandas>=2.0", "pyarrow>=14.0"]
TOKEN = "smoke"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, **kw)
    assert proc.returncode == 0, f"{' '.join(cmd)} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SPARC_STUDIO", "PYTHON", "VIRTUAL_ENV")) and k not in ("SPARC_PROGRESS",)}
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def test_wheel_serves_the_spa(tmp_path):
    # 1. the wheel
    wheel = os.environ.get("SPARC_SMOKE_WHEEL")
    if wheel:
        wheel_path = Path(wheel)
    else:
        dist = tmp_path / "dist"
        _run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-q", "-w", str(dist)], env=_clean_env())
        wheel_path = next(dist.glob("sparc-*.whl"))

    # 2. its static files
    with zipfile.ZipFile(wheel_path) as z:
        names = set(z.namelist())
        index = z.read("sparc/studio/static/index.html").decode("utf-8")
        info = json.loads(z.read("sparc/studio/static/BUILD_INFO.json"))
    refs = re.findall(r'(?:src|href)="/(assets/[^"]+)"', index)
    assert any(re.match(r"assets/index-[A-Za-z0-9_-]+\.js$", r) for r in refs), refs
    missing = [r for r in refs if f"sparc/studio/static/{r}" not in names]
    assert not missing, f"assets index.html loads but the wheel lacks: {missing}"
    assert set(info) == {"src_sha256", "vite", "react"}
    committed = json.loads((ROOT / "sparc" / "studio" / "static" / "BUILD_INFO.json").read_text("utf-8"))
    assert info == committed

    # 3. a clean environment
    venv = tmp_path / "venv"
    _run([sys.executable, "-m", "venv", str(venv)], env=_clean_env())
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    pip = [str(bindir / "python"), "-m", "pip", "install", "-q"]
    if os.environ.get("SPARC_SMOKE_FULL") == "1":
        pre = shlex.split(os.environ.get("SPARC_SMOKE_PREINSTALL", ""))
        if pre:
            _run(pip + pre, env=_clean_env())
        _run(pip + [f"{wheel_path}[studio]"], env=_clean_env())
    else:
        _run(pip + MINIMAL, env=_clean_env())
        _run(pip + ["--no-deps", str(wheel_path)], env=_clean_env())
    where = _run([str(bindir / "python"), "-c", "import sparc; print(sparc.__file__)"], cwd=tmp_path, env=_clean_env())
    assert str(venv) in where.stdout and str(ROOT) not in where.stdout, where.stdout

    # 4. the installed command serves the SPA
    import httpx

    ws = tmp_path / "ws"
    log = open(tmp_path / "server.log", "wb")
    proc = subprocess.Popen([str(bindir / "sparc"), "studio", "--no-browser", "--port", "0", "--workspace", str(ws),
                             "--token", TOKEN], cwd=tmp_path, env=_clean_env(), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        lock = ws / "studio.lock.json"

        def base():
            if proc.poll() is not None:
                raise AssertionError((tmp_path / "server.log").read_text("utf-8", "replace")[-3000:])
            try:
                info = json.loads(lock.read_text("utf-8"))
                url = f"http://127.0.0.1:{int(info['port'])}"
                return url if httpx.get(url + "/api/health", timeout=2, trust_env=False).status_code == 200 else None
            except (OSError, ValueError, KeyError, httpx.HTTPError):
                return None

        url = wait_for(base, 120, "the installed server")
        with httpx.Client(base_url=url, trust_env=False, timeout=30) as c:
            page = c.get("/")
            assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
            assert page.text == index, "GET / is not the packaged index.html"
            for r in refs:
                asset = c.get("/" + r)
                assert asset.status_code == 200, f"/{r} → {asset.status_code}"
                if r.endswith(".js"):
                    assert "javascript" in asset.headers["content-type"]
            system = c.get("/api/system", headers={"Authorization": f"Bearer {TOKEN}"}).json()
            assert system["web_build"]["src_sha256"] == info["src_sha256"]
            c.post("/api/shutdown", json={"stop_jobs": True}, headers={"Authorization": f"Bearer {TOKEN}"})
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
        time.sleep(0.1)
