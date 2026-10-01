"""The import rule of SPEC §5.5: the modules the Studio server imports never load torch.

Instrumented modules keep ``torch`` imports function-local.  The check runs
in a fresh interpreter, because the test process has torch loaded already.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER_SIDE = ("pipeline", "catalog", "cv", "data", "optimize", "emulator", "planner", "climate", "baselines")


def _import_report(code: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env.update(PYTHONPATH=str(REPO), OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_server_side_core_modules_do_not_import_torch():
    code = f"""
import importlib, importlib.util, json, sys
done, missing = [], []
for name in {SERVER_SIDE!r}:
    mod = "sparc.core." + name
    if importlib.util.find_spec(mod) is None:      # catalog.py arrives with a later work item
        missing.append(name)
        continue
    importlib.import_module(mod)
    done.append(name)
print(json.dumps({{"torch": "torch" in sys.modules, "done": done, "missing": missing}}))
"""
    rep = _import_report(code)
    assert rep["torch"] is False, rep
    assert set(rep["done"]) >= set(SERVER_SIDE) - {"catalog"}


def test_planning_api_works_without_torch(tmp_path):
    """plan_stages, fingerprint_sections and checkpoint_status are the planning half of the API (§11 item 6)."""
    code = f"""
import json, sys
from sparc.core.synthetic import write_demo_project
demo = write_demo_project({str(tmp_path)!r}, n=16, seed=0)
from sparc.core import plan_stages
from sparc.core.config import load_core_config
from sparc.core.pipeline import STAGE_NODES, CHECKPOINT_KEY, checkpoint_status, fingerprint_sections
cfg = load_core_config(demo["config_path"])
nodes = plan_stages(cfg, fast=True)
secs = fingerprint_sections(cfg, True)
st = checkpoint_status({str(tmp_path)!r}, cfg)
print(json.dumps({{"torch": "torch" in sys.modules, "ids": [n["id"] for n in nodes], "secs": sorted(secs),
                  "present": st["present"], "ck": CHECKPOINT_KEY["S2_S3"], "n": len(STAGE_NODES)}}))
"""
    rep = _import_report(code)
    assert rep["torch"] is False, rep
    assert rep["ids"] == ["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"]
    assert rep["secs"] == sorted(["data", "core", "s4", "s5", "climate", "s6", "s7", "code"])
    assert rep["present"] is False and rep["ck"] == "S3" and rep["n"] == 11


def test_core_package_lazy_exports():
    import sparc.core as core
    from sparc.core.pipeline import plan_stages

    assert core.plan_stages is plan_stages
    assert {"plan_stages", "open_run"} <= set(dir(core))
    try:
        fn = core.open_run                          # sparc/core/session.py may not exist yet
    except AttributeError as exc:
        assert "session" in str(exc)
    else:
        assert callable(fn)
