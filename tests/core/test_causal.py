"""S6 causal validation: planted spillovers, spatial confounding, a known
dose-response curve, and the sensitivity / audit arithmetic."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from sparc.core import causal as C
from sparc.core import operators as ops
from sparc.core.cv import make_spatial_folds
from sparc.core.grid import Grid
from sparc.core.synthetic import gaussian_random_field, make_dose_response_fixture, make_interference_fixture

N_FOLDS = 5
BLOCK_M = 200.0          # 100 blocks on the 1.92 km fixtures; buffer 67 m
BASIS_M = 600.0          # ≈ correlation length of the fixtures' smooth confounder; ≥ 2×radius


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: every estimator below is deterministic)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _single_openmp_thread():
    """HistGradientBoosting on ~4k points gains little from threads.  Under
    parallel test load, OpenMP oversubscription slows it by more than 10×."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="openmp"):
        yield


@pytest.fixture(scope="module")
def interference():
    fx = make_interference_fixture()
    f = fx["frame"]
    coords = f[["x", "y"]].to_numpy(dtype=float)
    grid = Grid.from_points(coords[:, 0], coords[:, 1], cell=30.0)
    folds = make_spatial_folds(coords, n_folds=N_FOLDS, block_m=BLOCK_M, seed=0)
    return {"fx": fx, "frame": f, "coords": coords, "grid": grid, "folds": folds,
            "Y": f["Y"].to_numpy(float), "T": f["T"].to_numpy(float), "W": f[["x1", "x2"]].to_numpy(float)}


@pytest.fixture(scope="module")
def spill(interference):
    d = interference
    return C.spillover_dml(d["Y"], d["T"], d["W"], d["grid"], 150.0, d["folds"], basis_scale_m=BASIS_M,
                           coords=d["coords"])


@pytest.fixture(scope="module")
def dose_response():
    fx = make_dose_response_fixture()
    f = fx["frame"]
    coords = f[["x", "y"]].to_numpy(dtype=float)
    folds = make_spatial_folds(coords, n_folds=N_FOLDS, block_m=BLOCK_M, seed=0)
    dr = C.dr_dose_response(f["Y"].to_numpy(float), f["T"].to_numpy(float), f[["x1", "x2"]].to_numpy(float),
                            folds, n_boot=100, seed=0)
    return fx, dr


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_exposure_mapping_matches_fixture_exposure(interference):
    d = interference
    ref = ops.masked_gaussian(d["T"].reshape(64, 64), np.ones((64, 64), dtype=bool), 150.0 / 30.0 / 2.0,
                              exclude_self=True).ravel()
    got = C.exposure_mapping(d["T"], d["grid"], 150.0)
    assert np.allclose(got, ref, atol=1e-8)


def test_exposure_mapping_is_nan_safe_for_isolated_points():
    x = np.array([0.0, 30.0, 60.0, 3000.0])          # last point is isolated
    y = np.zeros(4)
    grid = Grid.from_points(x, y, cell=30.0)
    out = C.exposure_mapping(np.array([1.0, 2.0, 3.0, 7.0]), grid, 60.0)
    assert np.all(np.isfinite(out))
    # The self-excluded mean is undefined there → falls back to the (self-including) mean ≈ own value.
    # (1e-3 slack: the FFT kernel at σ = 1 cell on a 3-row grid is slightly aliased.)
    assert out[3] == pytest.approx(7.0, abs=1e-3)
    assert out[1] == pytest.approx(2.0, abs=1e-3)     # symmetric neighbours 1 and 3


def test_spatial_basis_centre_count_and_radius_floor(interference):
    coords = interference["coords"]
    B = C.spatial_basis(coords, 300.0)
    # 1920 m × 1920 m / (300 m)² ≈ 41 centres
    assert 30 <= B.shape[1] <= 50
    assert B.shape[0] == len(coords) and np.all((B > 0) & (B <= 1.0))
    # A scale below 2×radius is raised to 2×radius.
    B2 = C.spatial_basis(coords, 100.0, radius_m=150.0)
    assert B2.shape == B.shape and np.allclose(B2, B)
    # Clipping to [4, max_centres].
    assert C.spatial_basis(coords, 5000.0).shape[1] == 4
    assert C.spatial_basis(coords, 60.0, max_centres=25).shape[1] == 25


def test_dml_requires_a_partition_of_test_folds(interference):
    d = interference
    folds = d["folds"]
    bad = type(folds)(fold_id=folds.fold_id, block_id=folds.block_id, n_folds=2, block_m=folds.block_m,
                      buffer_m=folds.buffer_m, train_masks=folds.train_masks[:2], test_masks=folds.test_masks[:2])
    with pytest.raises(ValueError, match="partition"):
        C.dml_plr(d["Y"], d["T"], d["W"], bad)


# ---------------------------------------------------------------------------
# Spillover / interference
# ---------------------------------------------------------------------------


def test_spillover_dml_recovers_own_neighbour_and_total(spill, interference):
    truth = interference["fx"]["truth"]
    th_o, th_n = truth["theta_own"], truth["theta_nbr"]
    th_s = th_o + th_n
    assert abs(spill["theta_own"] - th_o) <= max(3 * spill["se_own"], 0.2 * abs(th_o)), spill
    assert abs(spill["theta_nbr"] - th_n) <= max(3 * spill["se_nbr"], 0.2 * abs(th_n)), spill
    assert abs(spill["theta_sum"] - th_s) <= max(3 * spill["se_sum"], 0.2 * abs(th_s)), spill
    # θ_sum is better determined than θ_nbr (T and T̄ are collinear).
    assert spill["se_sum"] < spill["se_nbr"]
    assert spill["corr_T_Tbar"] > 0.8
    assert spill["basis_scale_m"] >= 2 * spill["radius_m"]
    assert spill["hole_scale_ratio"] < 1.0
    json.dumps(spill)


def test_naive_dml_absorbs_part_of_the_spillover(interference):
    """Leaving T̄ out biases θ̂ from θ_o towards θ_o + θ_n by the omitted-variable
    formula θ_naive = θ_o + θ_n·β, where β = Cov(R_T̄, R_T)/Var(R_T).

    At the fixture's default 150 m radius β ≈ 0.5 analytically.  The residual
    treatment variation is the 120 m-range field z, whose correlation with its
    σ = 75 m self-excluded kernel mean is ≈ 0.55.  So θ_naive sits at the
    θ_o / θ_sum midpoint, and "closer to θ_sum" would test noise here; see
    the short-range variant below for that claim."""
    d = interference
    truth = d["fx"]["truth"]
    th_o, th_s = truth["theta_own"], truth["theta_own"] + truth["theta_nbr"]
    basis = C.spatial_basis(d["coords"], BASIS_M)
    naive = C.dml_plr(d["Y"], d["T"], d["W"], d["folds"], basis=basis)
    full = C.dml_plr(d["Y"], d["T"], d["W"], d["folds"], basis=basis,
                     extra_treatments=C.exposure_mapping(d["T"], d["grid"], 150.0))
    nv = float(naive.theta[0])
    # Biased away from θ_o, towards (and not beyond) θ_sum.
    assert th_s < nv < th_o
    assert abs(nv - th_o) > 3 * float(naive.se_cluster[0])
    # The OVB identity with the estimated β.  It is exact here because both
    # fits share the same (deterministic) Y and T nuisance predictions.
    rt, rtb = full.resid_T[:, 0], full.resid_T[:, 1]
    beta = float(rt @ rtb / (rt @ rt))
    assert 0.3 < beta < 0.8
    assert nv == pytest.approx(full.theta[0] + full.theta[1] * beta, abs=1e-8)


def test_naive_dml_is_closer_to_total_for_short_range_exposure():
    """With a 60 m radius (σ = 1 cell), T̄ is dominated by the immediate
    neighbours and β ≈ 0.87.  The naive estimate then lies closer to
    θ_o + θ_n than to θ_o."""
    fx = make_interference_fixture(radius_m=60.0)
    f = fx["frame"]
    coords = f[["x", "y"]].to_numpy(dtype=float)
    folds = make_spatial_folds(coords, n_folds=N_FOLDS, block_m=BLOCK_M, seed=0)
    naive = C.dml_plr(f["Y"].to_numpy(float), f["T"].to_numpy(float), f[["x1", "x2"]].to_numpy(float), folds,
                      basis=C.spatial_basis(coords, BASIS_M))
    th_o = fx["truth"]["theta_own"]
    th_s = th_o + fx["truth"]["theta_nbr"]
    nv = float(naive.theta[0])
    assert abs(nv - th_s) < abs(nv - th_o)
    assert abs(nv - th_o) > 3 * float(naive.se_cluster[0])


# ---------------------------------------------------------------------------
# Spatial confounding
# ---------------------------------------------------------------------------


def test_spatial_basis_removes_smooth_confounding():
    """Y = θ·T + 0.8 sin(x1) + 0.6·s + ε with T driven by x1 and by an
    unobserved smooth field s.  Adjusting for x1 only is biased; adding the
    RBF basis recovers θ."""
    rng = np.random.default_rng(3)
    n, dx, theta = 64, 30.0, -0.05
    x1 = gaussian_random_field((n, n), 8.0, rng)
    s = gaussian_random_field((n, n), 30.0, rng)
    T = 40.0 + 12.0 * x1 + 6.0 * s + 8.0 * gaussian_random_field((n, n), 4.0, rng)
    Y = theta * T + 0.8 * np.sin(x1) + 0.6 * s + 0.15 * rng.standard_normal((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    coords = np.column_stack([(xx * dx).ravel(), (yy * dx).ravel()]).astype(float)
    folds = make_spatial_folds(coords, n_folds=N_FOLDS, block_m=BLOCK_M, seed=0)
    W = x1.ravel()[:, None]
    without = C.dml_plr(Y.ravel(), T.ravel(), W, folds)
    with_b = C.dml_plr(Y.ravel(), T.ravel(), W, folds, basis=C.spatial_basis(coords, BASIS_M))
    err_without = abs(float(without.theta[0]) - theta)
    err_with = abs(float(with_b.theta[0]) - theta)
    assert err_with < err_without
    assert err_with < 0.2 * abs(theta)
    assert with_b.n_blocks == folds.n_blocks
    assert np.all(with_b.se_cluster > 0) and np.all(with_b.se_iid > 0)
    assert with_b.overlap_weights.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Heterogeneity
# ---------------------------------------------------------------------------


def test_blp_calibration_recovers_true_heterogeneity_and_r_learner_runs():
    rng = np.random.default_rng(5)
    n, dx = 64, 30.0
    x1 = gaussian_random_field((n, n), 6.0, rng).ravel()
    x2 = gaussian_random_field((n, n), 3.0, rng).ravel()
    tau = -0.05 + 0.03 * x2
    T = 20.0 + 5.0 * x1 + 4.0 * rng.standard_normal(n * n)
    Y = tau * T + 0.5 * np.sin(x1) + 0.1 * rng.standard_normal(n * n)
    yy, xx = np.mgrid[0:n, 0:n]
    coords = np.column_stack([(xx * dx).ravel(), (yy * dx).ravel()]).astype(float)
    folds = make_spatial_folds(coords, n_folds=N_FOLDS, block_m=BLOCK_M, seed=0)
    W = np.column_stack([x1, x2])
    res = C.dml_plr(Y, T, W, folds)
    assert float(res.theta[0]) == pytest.approx(-0.05, abs=0.01)
    cal = C.blp_calibration(res.resid_Y, res.resid_T, tau, folds.block_id)
    assert cal["coef"] == pytest.approx(1.0, abs=0.3)
    assert cal["p"] < 1e-3
    noise = C.blp_calibration(res.resid_Y, res.resid_T, rng.standard_normal(n * n), folds.block_id)
    assert abs(noise["coef"]) < 0.1                         # uninformative signal → β₂ ≈ 0
    tau_hat = C.r_learner_cate(Y, T, W, folds, res.resid_Y, res.resid_T, feature_matrix=W)
    assert tau_hat.shape == (n * n,) and np.all(np.isfinite(tau_hat))
    assert np.corrcoef(tau_hat, tau)[0, 1] > 0.7


# ---------------------------------------------------------------------------
# Dose-response
# ---------------------------------------------------------------------------


def test_dr_dose_response_recovers_saturating_curve(dose_response):
    fx, dr = dose_response
    A = fx["truth"]["A"]
    truth = fx["truth"]["curve"](dr.t_grid)
    dev = (dr.theta - dr.theta.mean()) - (truth - truth.mean())
    assert np.max(np.abs(dev)) <= 0.35 * A, dict(zip(np.round(dr.t_grid, 1), np.round(dev, 3)))
    # Monotone decreasing within tolerance, with the right overall drop.
    assert np.all(np.diff(dr.theta) <= 0.1 * A)
    drop_hat, drop_true = dr.theta[-1] - dr.theta[0], truth[-1] - truth[0]
    assert drop_hat == pytest.approx(drop_true, abs=0.35 * A)
    assert dr.ess > 50
    assert dr.hurdle and dr.t_grid[0] == 0.0
    assert np.all(dr.lo <= dr.theta + 1e-12) and np.all(dr.theta <= dr.hi + 1e-12)
    assert 0.0 <= dr.clipped_frac < 0.5
    json.dumps(dr.to_dict())


def test_dr_grid_is_restricted_to_inner_quantiles(dose_response):
    fx, dr = dose_response
    T = fx["frame"]["T"].to_numpy(float)
    q10, q90 = np.quantile(T[T > 0], [0.1, 0.9])
    pos = dr.t_grid[1:]
    assert pos.min() >= q10 - 1e-9 and pos.max() <= q90 + 1e-9


# ---------------------------------------------------------------------------
# Sensitivity and audit arithmetic
# ---------------------------------------------------------------------------


def test_e_value_formula_and_null_crossing():
    ev = C.e_value(-0.05, 10.0, 0.5)
    rr = math.exp(0.91)
    assert ev["d"] == pytest.approx(-1.0)
    assert ev["rr"] == pytest.approx(rr)
    assert ev["e_value"] == pytest.approx(rr + math.sqrt(rr * (rr - 1.0)))
    assert C.e_value(-0.05, 10.0, 0.5, ci=(-0.09, 0.01))["e_value_ci"] == 1.0
    ci = C.e_value(-0.05, 10.0, 0.5, ci=(-0.08, -0.02))
    rr_ci = math.exp(0.91 * 0.4)
    assert ci["e_value_ci"] == pytest.approx(rr_ci + math.sqrt(rr_ci * (rr_ci - 1.0)))
    assert 1.0 < ci["e_value_ci"] < ev["e_value"]
    assert C.e_value(0.0, 10.0, 0.5)["e_value"] == pytest.approx(1.0)


def test_robustness_value_limits_and_monotonicity():
    kw = dict(t_cluster=5.0, se_cluster=0.02, se_iid=0.01, n_blocks=50, n=5000, p=10)
    assert C.robustness_value(0.0, **kw)["rv_q"] == 0.0
    assert C.robustness_value(0.0, **kw)["rv_q_alpha"] == 0.0
    r2s = [0.001, 0.01, 0.05, 0.1, 0.3, 0.6]
    rv = [C.robustness_value(r, **kw)["rv_q"] for r in r2s]
    rva = [C.robustness_value(r, **kw)["rv_q_alpha"] for r in r2s]
    assert np.all(np.diff(rv) > 0) and np.all(np.diff(rva) >= 0)
    assert all(a <= b for a, b in zip(rva, rv))
    # Closed form: f² = R²/(1−R²); RV = ½(√(f⁴+4f²) − f²).
    f2 = 0.1 / 0.9
    assert C.robustness_value(0.1, **kw)["rv_q"] == pytest.approx(0.5 * (math.sqrt(f2**2 + 4 * f2) - f2))
    # Clustering (design effect 2) makes the significance RV smaller.
    iid = C.robustness_value(0.01, **{**kw, "se_cluster": 0.01})
    assert iid["rv_q_alpha"] > C.robustness_value(0.01, **kw)["rv_q_alpha"]


def test_audit_verdicts():
    ok = C.audit(-0.052, 0.004, -0.05, 0.005)
    assert ok["verdict"] == "consistent" and not ok["flag"]
    mag = C.audit(-0.10, 0.004, -0.05, 0.005)
    assert mag["verdict"] == "magnitude differs" and mag["flag"] and mag["sign_agree"]
    sign = C.audit(0.03, 0.004, -0.05, 0.005)
    assert sign["verdict"] == "sign conflict" and sign["flag"] and not sign["sign_agree"]
    # Large relative gap but hopelessly noisy → not flagged.
    noisy = C.audit(-0.10, 0.05, -0.05, 0.05)
    assert noisy["verdict"] == "consistent"
    # Statistically clear but immaterial (< 25 %) → not flagged.
    small = C.audit(-0.055, 0.0001, -0.05, 0.0001)
    assert not small["flag"] and abs(small["z"]) > 2


def test_curve_audit_flags_shape_disagreement():
    dr = {"t_grid": [0, 10, 20, 30], "theta": [0.0, -1.0, -1.5, -1.7],
          "lo": [-0.1, -1.1, -1.6, -1.8], "hi": [0.1, -0.9, -1.4, -1.6]}
    same = C.curve_audit([0, 15, 30], [5.0, 3.75, 3.3], dr)       # level offset only, similar shape
    assert same["n_common"] == 4
    linear = C.curve_audit([0, 30], [0.0, -1.7], dr)             # linear, misses the saturation
    assert linear["flag"] and linear["max_ratio"] > same["max_ratio"]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _drop_tau_hat(obj):
    if isinstance(obj, dict):
        return {k: _drop_tau_hat(v) for k, v in obj.items() if not str(k).startswith("tau_hat")}
    if isinstance(obj, list):
        return [_drop_tau_hat(v) for v in obj]
    return obj


@pytest.fixture(scope="module")
def city_validation(synthetic_core_data):
    cfg, data = synthetic_core_data
    folds = make_spatial_folds(data.coords, n_folds=3, block_m=300.0, seed=0)
    causal_cfg = {**cfg.raw["causal"], "n_boot": 30}
    effects = {"canopy": {"adoption_slope": 0.05, "adoption_se": 0.001, "own_slope": 0.04, "own_se": 0.001,
                          "own_pd_curve": {"t": [0, 20, 40, 60], "y": [0.0, 0.5, 1.0, 1.5]},
                          "per_point_slope": np.linspace(-0.1, 0.1, data.n)}}
    tc = C.run_causal_validation(data, causal_cfg, folds, ranges_m={"canopy": 150.0}, model_effects=effects, seed=0)
    return data, tc


def test_run_causal_validation_end_to_end(city_validation):
    data, tc = city_validation
    r = tc["treatments"]["canopy"]
    assert set(r) >= {"dml", "spillover", "cate", "dr_curve", "sensitivity", "audit"}
    assert "ndvi" not in r["controls"]                       # mediator never adjusted for
    assert r["basis_scale_m"] == 1000.0                        # auto = max(1000, 2·150)
    assert r["dml"]["theta"][0] < 0                           # canopy cools
    assert isinstance(r["cate"]["tau_hat"], np.ndarray) and r["cate"]["tau_hat"].shape == (data.n,)
    assert r["sensitivity"]["e_value"]["contrast"] == 10.0
    assert r["sensitivity"]["e_value"]["e_value"] >= 1.0
    assert 0.0 <= r["sensitivity"]["robustness"]["rv_q_alpha"] <= r["sensitivity"]["robustness"]["rv_q"]
    assert r["spillover"]["basis_scale_m"] >= 300.0
    assert r["dr_curve"]["hurdle"] and r["dr_curve"]["t_grid"][0] == 0.0
    text = json.dumps(_drop_tau_hat(tc))
    assert "theta_sum" in text


def test_run_causal_validation_audits_model_effects(city_validation):
    _, tc = city_validation
    aud = tc["treatments"]["canopy"]["audit"]
    assert set(aud) == {"adoption_vs_theta_sum", "own_vs_theta_own", "own_pd_vs_dr_curve", "blp_calibration"}
    # A warming model slope contradicts the cooling causal estimates.
    assert aud["adoption_vs_theta_sum"]["verdict"] == "sign conflict"
    assert aud["own_pd_vs_dr_curve"]["flag"]                 # rising PD curve vs a falling DR curve
    assert any(f["check"] == "adoption_vs_theta_sum" for f in tc["flags"])


@pytest.mark.slow
def test_dag_audit_runs_on_tiny_problem():
    rng = np.random.default_rng(0)
    n = 900
    a = rng.standard_normal(n)
    b = 0.9 * a + 0.4 * rng.standard_normal(n)
    c = 0.8 * b + 0.4 * rng.standard_normal(n)
    frame = pd.DataFrame({"a": a, "b": b, "c": c})
    block = np.arange(n) // 30
    out = C.dag_audit(frame, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")], block, n_boot=2, n_iter=300)
    if out is None:
        pytest.skip("sparc.causal.mc3 not importable")
    assert set(out) >= {"edge_probs", "expert_low_support", "unexpected_high_support", "node_names"}
    P = np.asarray(out["edge_probs"])
    assert P.shape == (3, 3) and np.all((P >= 0) & (P <= 1))
    json.dumps(out)
