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


def test_stacker_beats_best_base_model(synthetic_run):
    m = synthetic_run.ensemble.metrics
    best_base = min(v["rmse"] for k, v in m.items() if k not in ("stacker", "base_mean"))
    assert m["stacker"]["rmse"] <= 0.98 * best_base


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
    t = synthetic_city.truth
    c = synthetic_city.frame["canopy"].to_numpy()
    # True footprint per pp = a·(Gᵀ∗mask)_i·∂q_i/∂c_i, with ∂q/∂c = the saturating
    # canopy term plus the NDVI mediator path (ndvi rises k_ndvi per pp).
    dq = -t["A_c"] * np.exp(-c / t["d_c"]) / t["d_c"] - t["w_ndvi"] * t["k_ndvi"]
    gm = ops.green_mass(synthetic_city.mask.astype(float), t["L"], t["v"], t["dx"])
    true_fp = t["a"] * gm[synthetic_city.fields["_iy"], synthetic_city.fields["_ix"]] * dq
    vr = synthetic_run.responses["canopy"]
    fp = vr.maps["footprint_effect_per_unit"].to_numpy()
    own = vr.maps["own_effect_per_unit"].to_numpy()
    # The fitted stack recovers ~40–45 % of the planted effect: flexible
    # spatial terms and trees attenuate the effect of a spatially smooth,
    # saturating covariate (every base model is below 65 % on this fixture —
    # a known limitation, see CORE_ROADMAP Appendix C).  The band guards the
    # sign and order of magnitude; the pattern and own-vs-footprint checks
    # guard "where it matters most".
    ratio = np.mean(fp) / np.mean(true_fp)
    assert 0.3 < ratio < 1.5
    assert np.corrcoef(fp, true_fp)[0, 1] > 0.3
    assert abs(np.mean(fp)) > 3 * abs(np.mean(own))


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


def test_manifest_and_report_are_serialisable(synthetic_run):
    from sparc.core.report import render_report

    json.dumps(synthetic_run.manifest, default=str)
    md = render_report(synthetic_run)
    assert "S2/S3" in md and "S4" in md


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
    assert m["stacker"]["r2"] > m["base_mean"]["r2"] - 0.01
    assert m["stacker"]["r2"] > 0.5
    assert 0.8 <= m["stacker"]["interval_coverage"] <= 0.97
    assert (res.run_dir / "report.md").exists()
    assert (res.run_dir / "predictions.parquet").exists()
