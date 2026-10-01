"""S6 keeps the per-cell arrays ``causal.json`` drops (SPEC §11 item 21): ``causal_cells.parquet`` and
``treatments[t].model_effects``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def s6_run(tmp_path_factory):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    root = tmp_path_factory.mktemp("causal_cells")
    demo = write_demo_project(root / "project", n=40, seed=0)
    cfg = load_core_config(demo["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": False}
    cfg.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = False
    cfg.raw["climate"]["enabled"] = False
    cfg.raw["causal"]["n_boot"] = 20
    cfg.raw["causal"]["treatments"] = ["canopy", "elevation"]       # elevation is not a lever: no model effects
    cfg.raw["causal"]["confounders"]["elevation"] = ["water_dist"]
    for v, d in (("canopy", [0, 10]), ("impervious", [0, 10]), ("albedo", [0, 0.1])):
        cfg.raw["actionable"][v]["doses"] = d
    return run_core(cfg, stages=("S6",), fast=True, coarse=60.0, run_dir=root / "run")


def test_causal_cells_are_row_aligned_with_predictions(s6_run):
    run_dir = s6_run.run_dir
    cells = pd.read_parquet(run_dir / "causal_cells.parquet")
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    np.testing.assert_array_equal(cells["id"].to_numpy(), pred["id"].to_numpy())
    assert set(cells.columns) == {"id", "cate:canopy", "mslope:canopy", "mslope_own:canopy", "cate:elevation"}
    for c in cells.columns[1:]:
        assert cells[c].dtype == np.float32, c
    tr = s6_run.causal["treatments"]["canopy"]
    np.testing.assert_allclose(cells["cate:canopy"], tr["cate"]["tau_hat"].astype(np.float32))
    me = tr["model_effects"]
    np.testing.assert_allclose(cells["mslope_own:canopy"], np.asarray(me["per_point_own_slope"], np.float32))
    np.testing.assert_allclose(cells["mslope:canopy"], np.asarray(me["per_point_slope"], np.float32), equal_nan=True)
    resp = pd.read_parquet(run_dir / "response_canopy.parquet")
    np.testing.assert_allclose(cells["mslope_own:canopy"], resp["own_effect_per_unit"].astype(np.float32))


def test_model_effects_in_causal_json_without_cell_arrays(s6_run):
    doc = json.loads((s6_run.run_dir / "causal.json").read_text())
    me = doc["treatments"]["canopy"]["model_effects"]
    assert set(me) == {"adoption_slope", "adoption_se", "own_slope", "own_se", "own_pd_curve"}
    curve = me["own_pd_curve"]
    assert set(curve) == {"t", "y", "se"} and len(curve["t"]) == len(curve["y"]) == len(curve["se"]) == 7
    assert np.all(np.diff(curve["t"]) >= 0)
    assert "model_effects" not in doc["treatments"]["elevation"]
    n = s6_run.data.n

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert not str(k).startswith(("tau_hat", "per_point")), path + "/" + str(k)
                walk(v, f"{path}/{k}")
        elif isinstance(obj, list):
            assert len(obj) < n, f"{path} holds a per-cell array ({len(obj)} values)"
            for v in obj:
                walk(v, path)

    walk(doc)
    assert s6_run.manifest["causal"]["canopy"]["audit"]                 # the audit used the model effects
