"""S5 writes ``scenario_detail.npz`` (SPEC §11 item 12, api.md §12.1), and ``ScenarioEngine(base_fold=…)`` skips
the baseline pass without changing a single delta (§11 item 11)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _light(cfg):
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": True, "physics": False}
    cfg.raw["stacker"].update(epochs=20, tune_lambda=[0.0])
    cfg.raw["cv"]["baselines"] = False
    for v, d in (("canopy", [0, 10]), ("impervious", [0, 10]), ("albedo", [0, 0.1])):
        cfg.raw["actionable"][v]["doses"] = d
    return cfg


@pytest.fixture(scope="module")
def s5_run(tmp_path_factory):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    root = tmp_path_factory.mktemp("scenario_detail")
    demo = write_demo_project(root / "project", n=40, seed=0)
    cfg = _light(load_core_config(demo["config_path"]))
    cfg.raw["climate"]["enabled"] = False
    return run_core(cfg, stages=("S0", "S1", "S2", "S3", "S4", "S5"), fast=True, coarse=60.0, run_dir=root / "run")


def test_fold_means_equal_scenario_deltas(s5_run):
    run_dir = s5_run.run_dir
    deltas = pd.read_parquet(run_dir / "scenario_deltas.parquet")
    with np.load(run_dir / "scenario_detail.npz") as z:
        names = json.loads(bytes(z["names"].astype(np.uint8)).decode("utf-8"))
        assert z["names"].dtype == np.uint8
        np.testing.assert_array_equal(z["ids"], deltas["id"].to_numpy())
        np.testing.assert_array_equal(z["ids"], s5_run.data.ids)
        assert names == [c for c in deltas.columns if c != "id"] == [s["name"] for s in s5_run.scenarios]
        K = s5_run.ensemble.folds.n_folds
        for i, name in enumerate(names):
            f, sd, ex = z[f"f{i}"], z[f"sd{i}"], z[f"ex{i}"]
            assert f.dtype == sd.dtype == ex.dtype == np.float32
            assert f.shape == (K, s5_run.data.n) and sd.shape == ex.shape == (s5_run.data.n,)
            np.testing.assert_allclose(f.astype(np.float64).mean(axis=0), deltas[name].to_numpy(), atol=1e-6)
            assert np.all(sd >= 0) and np.all(ex >= 0)
        assert f"f{len(names)}" not in z.files


def test_scenario_detail_can_be_switched_off(tmp_path):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    demo = write_demo_project(tmp_path / "project", n=24, seed=0)
    cfg = _light(load_core_config(demo["config_path"]))
    cfg.raw["climate"]["enabled"] = False
    cfg.raw["scenarios"] = cfg.raw["scenarios"][:1]
    cfg.raw["joint_scenarios"] = []
    assert cfg.raw["output"]["scenario_detail"] is True                 # the default (config.DEFAULTS)
    cfg.raw["output"]["scenario_detail"] = False
    r = run_core(cfg, stages=("S5",), fast=True, run_dir=tmp_path / "run")
    assert (r.run_dir / "scenario_deltas.parquet").exists() and not (r.run_dir / "scenario_detail.npz").exists()


def test_base_fold_passthrough_gives_identical_deltas(s5_run):
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import ScenarioEngine, specs_from_config

    r = s5_run
    med = MediatorChain(r.cfg.mediators).fit(r.data.frame) if r.cfg.mediators else None
    engine = ScenarioEngine(r.data, r.cfg, r.ensemble, dict(r.influence.ranges_m), med)
    assert engine.base_fold.shape == (r.ensemble.folds.n_folds, r.data.n)
    calls = []
    real = r.ensemble.fold_predictions

    def counting(ctx, *a, **k):
        calls.append(1)
        return real(ctx, *a, **k)

    r.ensemble.fold_predictions = counting
    try:
        again = ScenarioEngine(r.data, r.cfg, r.ensemble, dict(r.influence.ranges_m), med, base_fold=engine.base_fold)
        assert calls == []                                              # no baseline pass
        np.testing.assert_array_equal(again.baseline, engine.baseline)
        for spec in specs_from_config(r.cfg):
            np.testing.assert_array_equal(again.run(spec).delta, engine.run(spec).delta)
    finally:
        del r.ensemble.fold_predictions
    with pytest.raises(ValueError, match="base_fold"):
        ScenarioEngine(r.data, r.cfg, r.ensemble, dict(r.influence.ranges_m), med, base_fold=engine.base_fold[:1])
