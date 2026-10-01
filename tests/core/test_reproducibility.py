"""Provenance, reproduce, and golden numbers that guard against silent drift.

Golden numbers are regenerated deliberately (after an intended change) with
``SPARC_UPDATE_GOLDEN=1 pytest tests/core/test_reproducibility.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden" / "synthetic_small.json"


def _small_cfg(out_dir=None):
    from sparc.core.config import core_config_from_dict
    from sparc.core.synthetic import synthetic_city_config

    cfg = core_config_from_dict(synthetic_city_config())
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": True}
    cfg.raw["stacker"]["epochs"] = 30
    cfg.raw["stacker"]["tune_lambda"] = [0.0]
    cfg.raw["cv"]["baselines"] = ["idw"]
    if out_dir is not None:
        cfg.raw["output"]["dir"] = str(out_dir)
    return cfg


def test_golden_synthetic_numbers(synthetic_city):
    from sparc.core.pipeline import run_core

    r = run_core(_small_cfg(), stages=("S0", "S1", "S2", "S3"), frame=synthetic_city.frame, write=False)
    now = {k: {"r2": round(v["r2"], 4), "rmse": round(v["rmse"], 4)} for k, v in r.manifest["metrics"].items()
           if "r2" in v}
    now["cv_block_m"] = r.manifest["cv"]["block_m"]
    if os.environ.get("SPARC_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("golden numbers updated")
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert now["cv_block_m"] == pytest.approx(gold["cv_block_m"])
    for k in ("stacker", "ols", "gam", "physics"):
        assert now[k]["r2"] == pytest.approx(gold[k]["r2"], abs=0.02), k


def test_provenance_lock_and_reproduce_check(synthetic_city, tmp_path):
    from sparc.core.pipeline import run_core
    from sparc.core.reproduce import compare_manifests

    r = run_core(_small_cfg(tmp_path), stages=("S0", "S1"), frame=synthetic_city.frame)
    prov = r.manifest["provenance"]
    assert len(prov["input_sha256"]) == 64 and prov["input_kind"] == "frame"
    assert len(prov["code_sha256"]) == 64 and len(prov["config_sha256"]) == 64
    lock = (r.run_dir / "environment.txt").read_text()
    assert "numpy==" in lock and "torch==" in lock

    old = {"cv": {"block_m": 600.0, "test_sizes": [10, 12]}, "metrics": {"stacker": {"r2": 0.80}},
           "scenarios": [{"name": "a", "mean_delta": -0.50}], "provenance": {"input_sha256": "x", "code_sha256": "c1"},
           "versions": {"torch": "2.0"}}
    same = json.loads(json.dumps(old))
    same["metrics"]["stacker"]["r2"] = 0.805
    same["scenarios"][0]["mean_delta"] = -0.51
    same["provenance"]["code_sha256"] = "c2"                    # reported, not failed
    out = compare_manifests(old, same)
    assert out["pass"] and not next(c for c in out["checks"] if c["check"] == "core code")["ok"]
    drift = json.loads(json.dumps(old))
    drift["metrics"]["stacker"]["r2"] = 0.75
    assert not compare_manifests(old, drift)["pass"]
