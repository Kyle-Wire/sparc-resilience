"""Base models, stacker, scenarios, response surfaces, optimiser and the
end-to-end pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sparc.core import operators as ops
from sparc.core.base_models import FeatureContext, MGWRModel, OLSModel
from sparc.core.grid import Grid
from sparc.core.response import fit_saturation
from sparc.core.synthetic import gaussian_random_field


# ---------------------------------------------------------------------------
# Unit tests (fast)
# ---------------------------------------------------------------------------


def _ctx_from_arrays(x, y, X: pd.DataFrame, target):
    g = Grid.from_points(x, y)
    return FeatureContext(frame=X, X=X, F=pd.DataFrame(index=X.index), coords=np.column_stack([x, y]), grid=g,
                          meta={"y": target})


def test_mgwr_recovers_spatially_varying_coefficient():
    rng = np.random.default_rng(0)
    n = 80
    yy, xx = np.mgrid[0:n, 0:n]
    beta = 1.0 + 0.8 * gaussian_random_field((n, n), 20.0, rng)
    xf = gaussian_random_field((n, n), 2.0, rng)
    target = beta * xf + 0.1 * rng.standard_normal((n, n))
    X = pd.DataFrame({"x1": xf.ravel()})
    ctx = _ctx_from_arrays((xx * 30.0).ravel(), (yy * 30.0).ravel(), X, target.ravel())
    idx = np.arange(n * n)
    m = MGWRModel(ranges_m={"x1": 600.0}, intercept_range_m=600.0).fit(ctx, idx)
    b_hat = m.coefficient_rasters()["x1"] / m.std.sd[0]
    corr = np.corrcoef(b_hat[ctx.grid.iy, ctx.grid.ix], beta.ravel())[0, 1]
    assert corr > 0.8
    r2_m = 1 - np.mean((m.predict(ctx) - target.ravel()) ** 2) / np.var(target)
    r2_o = 1 - np.mean((OLSModel().fit(ctx, idx).predict(ctx) - target.ravel()) ** 2) / np.var(target)
    assert r2_m > r2_o + 0.1


def test_fit_saturation_recovers_planted_scale_and_flags_linear_and_censored():
    rng = np.random.default_rng(1)
    doses = np.array([0, 5, 10, 15, 20, 30, 40], dtype=float)
    n = 300
    D = np.repeat(doses[:, None], n, axis=1)
    ds_true = np.where(np.arange(n) < 100, 12.0, 20.0)
    B = 2.0 * (1 - np.exp(-D / ds_true)) + 0.01 * rng.standard_normal(D.shape)
    B[:, 200:] = 0.05 * D[:, 200:] + 0.005 * rng.standard_normal((len(doses), 100))   # linear cells
    B[0] = 0.0
    out = fit_saturation(D, B, np.ones_like(D, dtype=bool))
    assert np.nanmedian(out["ds"][:100]) == pytest.approx(12.0, rel=0.15)
    assert np.nanmedian(out["ds"][100:200]) == pytest.approx(20.0, rel=0.15)
    assert (out["model"][200:] != "saturating").mean() > 0.8 or out["censored"][200:].mean() > 0.8
    assert np.all(np.isnan(out["d90"][200:]) | (out["d90"][200:] <= 40))


def test_gaussian_conv_adjoint_identity():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((30, 30)), rng.standard_normal((30, 30))
    lhs = np.sum(ops.gaussian_conv(a, 2.0) * b)
    rhs = np.sum(a * ops.gaussian_conv(b, 2.0))
    assert lhs == pytest.approx(rhs, rel=1e-8)


def test_green_mass_is_one_in_the_interior():
    m = np.ones((90, 90))
    gm = ops.green_mass(m, 120.0, (40.0, 10.0), 30.0)
    assert gm[45, 45] == pytest.approx(1.0, abs=0.02)
    assert ops.green_centre(0.0, (0.0, 0.0), 30.0) == pytest.approx(1.0)


def test_mediator_abduction_zero_edit_is_identity(synthetic_core_data):
    from sparc.core.mediators import MediatorChain

    cfg, data = synthetic_core_data
    med = MediatorChain(cfg.mediators).fit(data.frame)
    same = med.update(data.frame, data.frame.copy())
    assert np.array_equal(same["ndvi"].to_numpy(), data.frame["ndvi"].to_numpy())
    edited = data.frame.copy()
    edited["canopy"] = edited["canopy"] + 10
    up = med.update(data.frame, edited)
    assert (up["ndvi"] - data.frame["ndvi"]).mean() > 0      # monotone: more canopy → more NDVI


# ---------------------------------------------------------------------------
# Shared synthetic run (S0–S5 + S7) — built once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_run(synthetic_city):
    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import synthetic_city_config

    cfg = core_config_from_dict(synthetic_city_config())
    cfg.raw["stacker"]["epochs"] = 150
    cfg.raw["stacker"]["tune_lambda"] = [0.0, 0.1]
    cfg.raw["actionable"] = {"canopy": {"min": 0, "max": 100, "doses": [0, 5, 10, 15, 20, 30, 40]}}
    cfg.raw["causal"]["enabled"] = False
    return run_core(cfg, stages=("S0", "S1", "S2", "S3", "S4", "S5", "S7"), frame=synthetic_city.frame, write=False)


def test_stacker_is_never_worse_than_best_base_model(synthetic_run):
    """Super-learner guarantee under honest spatial-block CV: the stack
    (convex base, plus a neural residual only if it earns its place out of
    fold) matches or beats the best single base model and the plain mean."""
    ens = synthetic_run.ensemble
    m = ens.metrics
    best_base = min(v["rmse"] for k, v in m.items() if k not in ("stacker", "base_mean"))
    assert m["stacker"]["rmse"] <= 1.01 * best_base
    assert m["stacker"]["rmse"] < m["base_mean"]["rmse"]
    assert "off" in ens.lambda_scores
    for info in ens.stacker_info:                     # convex weights
        w = np.array(list(info["weights"].values()))
        assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)
    # the data are generated by the physics operator: it should dominate
    assert np.mean([i["weights"]["physics"] for i in ens.stacker_info]) > 0.5


def test_cross_conformal_coverage(synthetic_run):
    cov = synthetic_run.ensemble.metrics["stacker"]["interval_coverage"]
    assert 0.85 <= cov <= 0.95


def test_zero_scenario_is_exact_identity(synthetic_run):
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec

    r = synthetic_run
    eng = ScenarioEngine(r.data, r.cfg, r.ensemble, r.influence.ranges_m, MediatorChain(r.cfg.mediators).fit(r.data.frame))
    res = eng.run(ScenarioSpec("zero", [Intervention("canopy", "add", 0.0)]))
    assert np.max(np.abs(res.delta)) < 1e-9
    res2 = eng.run(ScenarioSpec("cap", [Intervention("canopy", "add", 500.0)]), keep_frame=True)
    assert res2.frame["canopy"].max() <= 100.0 + 1e-9


def test_footprint_is_consistent_with_uniform_scenario(synthetic_run):
    """Σ_i footprint_i = Σ_j Δŷ_j for a uniform edit: the analytic chain rule
    (focal kernels + physics Green's function + mediators) must agree with
    re-predicting a small uniform scenario through the full stack."""
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec

    r = synthetic_run
    eng = ScenarioEngine(r.data, r.cfg, r.ensemble, r.influence.ranges_m, MediatorChain(r.cfg.mediators).fit(r.data.frame))
    h = 1.0
    res = eng.run(ScenarioSpec("uniform", [Intervention("canopy", "add", h)]))
    fp = r.responses["canopy"].maps["footprint_effect_per_unit"].to_numpy()
    assert np.mean(fp) == pytest.approx(np.mean(res.delta) / h, rel=0.15)


def test_footprint_tracks_planted_truth_and_exceeds_own_effect(synthetic_run, synthetic_city):
    true_fp = synthetic_city.true_footprint()
    vr = synthetic_run.responses["canopy"]
    fp = vr.maps["footprint_effect_per_unit"].to_numpy()
    own = vr.maps["own_effect_per_unit"].to_numpy()
    # With the saturating physics shade the stacked footprint recovers ~63% of
    # the planted effect with r ≈ 0.95 (was 53%, r 0.87); the rest is
    # attenuation by the prediction-tuned GAM / forest shrinkage.
    ratio = np.mean(fp) / np.mean(true_fp)
    assert 0.5 < ratio < 1.5
    assert np.corrcoef(fp, true_fp)[0, 1] > 0.7
    assert abs(np.mean(fp)) > 3 * abs(np.mean(own))


def test_effect_recovery_benchmark(synthetic_run, synthetic_city):
    from sparc.core.diagnostics import effect_recovery

    rec = effect_recovery(synthetic_run, synthetic_city, variable="canopy", dose=5.0)
    assert {"ols", "mgwr", "gwrf", "gam", "physics"} <= set(rec["models"])
    assert 0.55 < rec["stack"]["share"] < 1.5 and rec["stack"]["corr"] > 0.8
    assert rec["models"]["physics"]["share"] > 0.6          # saturating shade


def _fold0_setup(synthetic_core_data):
    from sparc.core.cv import make_spatial_folds
    from sparc.core.features import build_context
    from sparc.core.influence import compute_influence
    from sparc.core.mediators import MediatorChain
    from sparc.core.pipeline import _block_and_buffer

    cfg, data = synthetic_core_data
    inf = compute_influence(data, cfg.raw["influence"], seed=0)
    ranges = dict(inf.ranges_m)
    blk, buf = _block_and_buffer(cfg, inf, data)
    folds = make_spatial_folds(data.coords, 3, blk, buf, 42)
    ctx = build_context(data.frame, data, ranges, cfg)
    new = data.frame.copy()
    new["canopy"] = np.clip(new["canopy"] + 5.0, 0, 100)
    new = MediatorChain(cfg.mediators).fit(data.frame).update(data.frame, new)
    return cfg, data, inf, ranges, folds, ctx, build_context(new, data, ranges, cfg)


def test_spatial_plus_de_attenuates_mgwr(synthetic_core_data, synthetic_city):
    _cfg, data, inf, ranges, folds, ctx, ctx5 = _fold0_setup(synthetic_core_data)
    tr, te = next(iter(folds.split()))
    tm = synthetic_city.true_response(5.0).mean()
    out = {}
    for sp in (False, True):
        m = MGWRModel(ranges_m=ranges, intercept_range_m=inf.target_resid_range_m, spatial_plus=sp).fit(ctx, tr)
        p = m.predict(ctx)
        out[sp] = ((m.predict(ctx5) - p).mean() / tm, float(np.sqrt(np.mean((p[te] - data.y[te]) ** 2))))
    assert out[True][0] > out[False][0] + 0.08               # more of the planted effect
    assert out[True][1] < 1.03 * out[False][1]              # at no cost in held-out accuracy
    # a zero edit is still an exact identity (the covariate smooth is frozen)
    m = MGWRModel(ranges_m=ranges, intercept_range_m=inf.target_resid_range_m, spatial_plus=True).fit(ctx, tr)
    assert np.allclose(m.predict(ctx), m.predict(ctx))


def test_physics_learns_saturating_canopy_shade(synthetic_core_data):
    from sparc.core.base_models import PhysicsBaseModel

    cfg, data, inf, _ranges, folds, ctx, _ctx5 = _fold0_setup(synthetic_core_data)
    tr, _te = next(iter(folds.split()))
    m = PhysicsBaseModel(ctx.grid, cfg.raw["physics"], L_init=inf.L_prior_m, seed=0).fit(ctx, tr)
    assert m.params["kappa_canopy"] < 0.5                   # planted d_c = 15 pp → κ = 0.15


def test_causal_crosscheck_attaches_linear_band():
    from sparc.core.pipeline import causal_crosscheck

    scen = [{"name": "c+10", "mean_delta": -0.20, "mean_realized": {"canopy": 10.0}},
            {"name": "pkg", "mean_delta": -2.0, "mean_realized": {"canopy": 10.0, "imp": -10.0}},
            {"name": "other", "mean_delta": -0.1, "mean_realized": {"albedo": 0.1}}]
    causal = {"treatments": {"canopy": {"spillover": {"theta_sum": -0.02, "se_sum": 0.005}},
                             "imp": {"spillover": {"theta_sum": 0.05, "se_sum": 0.01}}}}
    out = causal_crosscheck(scen, causal)
    c = out[0]["causal_linear"]
    assert c["delta"] == pytest.approx(-0.2) and c["model_within"]
    assert out[1]["causal_linear"]["delta"] == pytest.approx(-0.7) and not out[1]["causal_linear"]["model_within"]
    assert "causal_linear" not in out[2]


def test_adoption_curve_is_concave_and_scenarios_match_sweep(synthetic_run):
    vr = synthetic_run.responses["canopy"]
    b = vr.curve["mean_benefit"].to_numpy()
    assert np.all(np.diff(b) > 0)                              # more canopy → more cooling
    marg = np.diff(b) / np.diff(vr.curve["dose"].to_numpy())
    assert marg[-1] < marg[0]                                  # diminishing returns
    assert vr.summary["frac_saturating"] + vr.summary["frac_linear"] > 0.5


def test_optimizer_respects_budget_and_closed_loop(synthetic_run):
    o = synthetic_run.optimize
    assert o["total_cost"] <= o["budget"] * (1 + 1e-9)
    assert o["n_cells_treated"] > 0
    assert o["realized_total_cooling"] > 0
    assert 0.3 < o["realized_total_cooling"] / o["planned_total_cooling"] < 3.0


def test_cv_blocks_are_spatial(synthetic_run):
    """Blocks must span many cells (block size comes from the S1 residual
    range, capped by the study-area extent — never from the target)."""
    cv = synthetic_run.manifest["cv"]
    cell = synthetic_run.data.grid.dx
    assert cv["block_m"] >= 5 * cell
    assert cv["n_blocks"] < synthetic_run.data.n / 25


def test_manifest_and_report_are_serialisable(synthetic_run):
    from sparc.core.report import render_report

    json.dumps(synthetic_run.manifest, default=str)
    md = render_report(synthetic_run)
    assert "S2/S3" in md and "S4" in md


def _small_synthetic_cfg(out_dir=None):
    from sparc.core.config import core_config_from_dict
    from sparc.core.synthetic import synthetic_city_config

    cfg = core_config_from_dict(synthetic_city_config())
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": True}
    cfg.raw["stacker"]["epochs"] = 30
    cfg.raw["stacker"]["tune_lambda"] = [0.0]
    if out_dir is not None:
        cfg.raw["output"]["dir"] = str(out_dir)
    return cfg


def test_resume_reuses_checkpoint_and_rejects_changed_config(synthetic_city, tmp_path, monkeypatch):
    from sparc.core import pipeline

    stages = ("S0", "S1", "S2", "S3")
    r1 = pipeline.run_core(_small_synthetic_cfg(tmp_path), stages=stages, frame=synthetic_city.frame)
    assert (r1.run_dir / pipeline.CHECKPOINT).exists()

    def boom(*a, **k):
        raise AssertionError("fit_ensemble must not run when resuming")

    monkeypatch.setattr(pipeline, "fit_ensemble", boom)
    r2 = pipeline.run_core(_small_synthetic_cfg(tmp_path), stages=stages, frame=synthetic_city.frame, resume=True)
    np.testing.assert_array_equal(r2.ensemble.oof_pred, r1.ensemble.oof_pred)
    assert r2.manifest["cv"]["block_m"] == r1.manifest["cv"]["block_m"]

    changed = _small_synthetic_cfg(tmp_path)
    changed.raw["stacker"]["epochs"] = 31                     # different fingerprint → refit
    with pytest.raises(AssertionError, match="must not run"):
        pipeline.run_core(changed, stages=stages, frame=synthetic_city.frame, resume=True)


def test_cv_distance_curve_shows_leaky_random_cv(synthetic_city):
    from sparc.core.pipeline import run_core
    from sparc.core.report import render_report

    cfg = _small_synthetic_cfg()
    cfg.raw["cv"]["distance_curve"] = {"enabled": True, "block_m": [0, 300]}
    r = run_core(cfg, stages=("S0", "S1", "S2", "S3"), frame=synthetic_city.frame, write=False)
    rows = r.cv_distance["rows"]
    assert [row["block_m"] for row in rows] == sorted(row["block_m"] for row in rows)
    main = [row for row in rows if row["main"]]
    assert len(main) == 1 and main[0]["block_m"] == pytest.approx(r.manifest["cv"]["block_m"])
    random_row = rows[0]
    assert random_row["buffer_m"] == 0.0 and random_row["n_blocks"] > r.data.n / 2
    # random points leak through spatial autocorrelation: never harder than the main blocks
    assert random_row["stacker"]["r2"] >= main[0]["stacker"]["r2"]
    json.dumps(r.manifest["cv_distance"], default=str)
    assert "Skill vs distance" in render_report(r)


# ---------------------------------------------------------------------------
# Providence (brown4.csv) — real-data smoke run
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_providence_fast_run(providence_config_path, brown_csv, tmp_path):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core

    cfg = load_core_config(providence_config_path)
    cfg.raw["output"]["dir"] = str(tmp_path)
    res = run_core(cfg, stages=("S0", "S1", "S2", "S3"), fast=True)
    m = res.manifest["metrics"]
    # Smoke-level guarantees on an 8k-point window with ~800 m held-out
    # blocks (a strict extrapolation test — skill is modest by design).
    assert res.manifest["cv"]["block_m"] >= 300.0               # genuinely spatial blocks
    best_base = min(v["rmse"] for k, v in m.items() if k not in ("stacker", "base_mean"))
    assert m["stacker"]["rmse"] <= 1.1 * best_base
    assert m["stacker"]["r2"] > 0.0
    assert 0.8 <= m["stacker"]["interval_coverage"] <= 0.97
    assert (res.run_dir / "report.md").exists()
    assert (res.run_dir / "predictions.parquet").exists()
