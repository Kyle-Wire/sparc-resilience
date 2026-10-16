"""Physics identifiability, prior sensitivity and prior-data conflict (``sparc.core.identifiability``; PHYSICS_PLAN
W3), and the physics hooks it relies on (``PhysicsModel.fit(fixed=, prior_scale=, init=)``, ``FitTerms``).

Most tests use a small *albedo world*: the physics model's own operator applied to an albedo source with a planted
mixing length L = 90 m on a 60 × 60 grid of 30 m cells, plus independent or spatially correlated noise.  With only
the albedo role the source is q ∝ (1 − EF)·(albedo − mean), so the EF intercept e₀ is exactly confounded with the
gain a: a structurally non-identifiable parameter by construction.  The slow coverage simulation is the method's
validation (module docstring of ``sparc.core.identifiability``)."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sparc.core import identifiability as ID
from sparc.core import operators as ops
from sparc.core.cv import block_ids
from sparc.core.grid import Grid
from sparc.core.physics import NATURAL_LEAF, PhysicsModel
from sparc.core.synthetic import gaussian_random_field

GOLDEN = Path(__file__).parent / "golden" / "physics_fit_reference.json"
L_TRUE = 90.0
BLOCK_M = 300.0
WORLD_CFG = {"roles": {"albedo": "albedo"}, "window": "day", "sw_down": 800.0, "lw_net": -100.0,
             "fit_advection": False, "max_iter": 60}
FAST = {"max_points": 8, "ref_starts": 2, "polish_iter": 100}


def albedo_world(n=60, dx=30.0, L=L_TRUE, noise=0.5, noise_range=None, seed=0, block_m=BLOCK_M, canopy=None,
                 roles=None, canopy_effect=0.0):
    """(grid, frame, y, coords, clusters, cfg): y = signal(L) + noise on an n × n lattice.  The signal is the
    physics model's own operator on the centred albedo source (sd 1); ``noise_range`` (cells) makes the noise a
    Gaussian random field (practical range ≈ 1.73·noise_range), else it is independent.  ``canopy``: a constant
    canopy layer (%) added as a role; ``canopy_effect``: a canopy *warming* planted in the source."""
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(1000)
    alb = 0.15 + 0.05 * gaussian_random_field((n, n), 2.0, srng)
    can = (np.full((n, n), float(canopy)) if canopy is not None and not canopy_effect else
           np.clip(40.0 + 25.0 * gaussian_random_field((n, n), 2.0, srng), 0.0, 100.0))
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx * dx).ravel().astype(float), (yy * dx).ravel().astype(float)
    grid = Grid.from_points(x, y, cell=dx)
    cfg = dict(WORLD_CFG, roles=dict(roles or WORLD_CFG["roles"]))
    pad = PhysicsModel(grid, cfg).pad
    q = -(alb.ravel() - alb.mean()) + canopy_effect * 0.002 * (can.ravel() - can.mean())
    phi = grid.sample(ops.solve(np.nan_to_num(grid.rasterize(q)), L, (0.0, 0.0), dx=dx, pad=pad))
    eps = (gaussian_random_field((n, n), noise_range, rng).ravel() if noise_range else rng.standard_normal(n * n))
    t = phi / phi.std() + noise * eps
    frame = pd.DataFrame({"albedo": alb.ravel(), "canopy": can.ravel()})
    coords = np.column_stack([x, y])
    return grid, frame, t, coords, block_ids(coords, block_m), cfg


def city_world(n=36, dx=30.0, roles=None, seed=5, noise_seed=1, block_m=270.0, **cfg_extra):
    """A small multi-role world (albedo, canopy, impervious, NDVI, elevation, water distance), the physics
    operator on a source with canopy cooling and impervious warming plus an elevation term and independent
    noise: (grid, frame, y, coords, clusters, cfg).  ``roles``: the roles given to the model (default all)."""
    r = np.random.default_rng(seed)

    def g(s):
        return gaussian_random_field((n, n), s, r).ravel()

    alb = 0.15 + 0.04 * g(2.0)
    can = np.clip(35 + 25 * g(2.5), 0, 100)
    imp = np.clip(50 - 0.8 * (can - 35) + 15 * g(3.0), 0, 100)
    ndvi = np.clip(0.4 + 0.004 * (can - 35) + 0.05 * g(2.0), -0.1, 0.95)
    elev = 20 + 10 * g(8.0)
    wd = np.abs(30 * g(6.0)) * 10
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx * dx).ravel().astype(float), (yy * dx).ravel().astype(float)
    grid = Grid.from_points(x, y, cell=dx)
    q = -(alb - alb.mean()) - 0.002 * (can - can.mean()) + 0.001 * (imp - imp.mean())
    phi = grid.sample(ops.solve(np.nan_to_num(grid.rasterize(q)), 90.0, (0.0, 0.0), dx=dx,
                                pad=PhysicsModel(grid, {}).pad))
    t = phi / phi.std() - 0.01 * (elev - elev.mean()) + 0.5 * np.random.default_rng(noise_seed).standard_normal(n * n)
    frame = pd.DataFrame({"albedo": alb, "canopy": can, "impervious": imp, "ndvi": ndvi, "elevation": elev,
                          "water_distance": wd})
    coords = np.column_stack([x, y])
    roles = roles if roles is not None else {k: k for k in frame.columns}
    cfg = {"fit_advection": False, "max_iter": 40, "sw_down": 800.0, "lw_net": -100.0, "window": "day",
           "roles": dict(roles), **cfg_extra}
    return grid, frame, t, coords, block_ids(coords, block_m), cfg


def sink_world(n=60, dx=30.0, L=L_TRUE, noise=1.0, seed=4, frac=0.15, block_m=BLOCK_M):
    """The albedo world plus a water-cell role whose cells sit at the albedo source's local minima (the lowest
    ``frac`` of the source minus its 2L-smoothed self), with no water effect planted: the data-only sink w (≥ 0)
    lands on its bound, and leaves it when L is pushed above its estimate (a sink at the source minima sharpens
    an over-smoothed prediction) — the Providence pattern (s, w at 0 at θ̂, s ≈ 0.2 and w ≈ 10 W/m² at L = 460 m).
    (grid, frame, y, coords, clusters, cfg)."""
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(1000)
    alb = 0.15 + 0.05 * gaussian_random_field((n, n), 2.0, srng)
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx * dx).ravel().astype(float), (yy * dx).ravel().astype(float)
    grid = Grid.from_points(x, y, cell=dx)
    cfg = dict(WORLD_CFG, roles={"albedo": "albedo", "water_distance": "water_distance"})
    pad = PhysicsModel(grid, cfg).pad
    q = -(alb.ravel() - alb.mean())
    hp = q - grid.sample(ops.solve(np.nan_to_num(grid.rasterize(q)), 2 * L, (0.0, 0.0), dx=dx, pad=pad))
    wd = np.where(hp <= np.quantile(hp, frac), 0.0, 300.0)
    phi = grid.sample(ops.solve(np.nan_to_num(grid.rasterize(q)), L, (0.0, 0.0), dx=dx, pad=pad))
    t = phi / phi.std() + noise * rng.standard_normal(n * n)
    coords = np.column_stack([x, y])
    return (grid, pd.DataFrame({"albedo": alb.ravel(), "water_distance": wd}), t, coords, block_ids(coords, block_m),
            cfg)


def _finite_or_none(rep):
    """Every number in the report is finite (or None)."""
    def walk(o):
        if isinstance(o, dict):
            return all(walk(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return all(walk(v) for v in o)
        if isinstance(o, float):
            return math.isfinite(o)
        return True
    return walk(rep)


def _row(rep, key):
    return next(r for r in rep["parameters"] if r["key"] == key)


def _covers(ci, v):
    lo, hi = ci
    return (lo is None or lo <= v) and (hi is None or hi >= v)


# ---------------------------------------------------------------------------
# (g) the physics hooks leave the default fit bit-identical
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def city_fold(synthetic_core_data):
    from sparc.core.cv import make_spatial_folds

    _, data = synthetic_core_data
    folds = make_spatial_folds(data.coords, n_folds=3, block_m=300.0, seed=42)
    train, _ = next(folds.split())
    return data, train


def test_default_fit_is_bit_identical_to_the_pre_hook_reference(city_fold):
    """``tests/core/golden/physics_fit_reference.json`` was written by the code of commit 5b9c6c9 (before the W3
    hooks) on this fold of the synthetic city: every float parameter (repr) and the prediction's SHA-256 must be
    the same.  On another platform or torch/numpy build FFT rounding may differ, so there the parameters are
    compared to 1e-9 instead."""
    from sparc.core.synthetic import synthetic_city_config

    gold = json.loads(GOLDEN.read_text())
    same_env = (gold["torch"] == torch.__version__ and gold["numpy"] == np.__version__
                and gold["machine"] == platform.machine())
    data, train = city_fold
    base = synthetic_city_config()["physics"]
    for name, case in gold["cases"].items():
        m = PhysicsModel(data.grid, {**base, **case["overrides"]}, L_init=3 * data.grid.dx).fit(
            data.frame, data.y, train, data.coords)
        pred = m.predict(data.frame, data.coords)
        got = {k: v for k, v in m.params.items() if isinstance(v, float)}
        assert set(got) == set(case["params"]), name
        if same_env:
            assert {k: repr(v) for k, v in got.items()} == case["params"], name
            assert repr(m.fit_info["loss"]) == case["loss"], name
            assert hashlib.sha256(np.ascontiguousarray(pred, dtype=np.float64).tobytes()).hexdigest() \
                == case["pred_sha256"], name
        else:
            for k, v in got.items():
                assert v == pytest.approx(float(case["params"][k]), rel=1e-9, abs=1e-12), (name, k)
        assert "hooks" not in m.fit_info


def test_fit_hooks_fixed_prior_scale_init_and_fit_terms(city_fold):
    from sparc.core.synthetic import synthetic_city_config

    data, train = city_fold
    cfg = {**synthetic_city_config()["physics"], "fit_advection": False, "max_iter": 30}
    ref = PhysicsModel(data.grid, cfg, L_init=90.0).fit(data.frame, data.y, train, data.coords)
    # prior_scale=1 is the run's objective: the same fit, bit for bit
    one = PhysicsModel(data.grid, cfg, L_init=90.0).fit(data.frame, data.y, train, data.coords, prior_scale=1.0)
    assert one.params["L_m"] == ref.params["L_m"] and one.fit_info["loss"] == ref.fit_info["loss"]
    # FitTerms reproduces the fit's objective and residuals at the fitted parameters
    terms = ref.fit_terms(data.frame, data.y, train, data.coords)
    P = {k: torch.tensor(v, dtype=torch.float64) for k, v in ref.leaves.items()}
    with torch.no_grad():
        loss = float(terms.evaluate(P)[0])
        coef = torch.tensor([ref.a, ref.b, ref.gamma, ref.beta_x, ref.beta_y], dtype=torch.float64)
        r = terms.residuals(P, coef)
    assert loss == pytest.approx(ref.fit_info["loss"], rel=1e-12)
    assert float(r.pow(2).mean().sqrt()) == pytest.approx(ref.train_rmse, rel=1e-10)
    assert terms.linear_names == ["a", "b", "gamma", "beta_x", "beta_y"]
    # the penalty is prior_weight · Σ z² over prior_terms (no drift: advection off)
    T = terms.transform(P)
    z2 = sum(((float(T[k]) if sc == "lin" else math.log(float(T[k]))) - mu) ** 2 / sd ** 2
             for k, sc, mu, sd in ref.prior_terms(terms.tf))
    assert float(terms.penalty(P)) == pytest.approx(cfg.get("prior_weight", 1e-2) * z2, rel=1e-12)
    assert float(terms.penalty(P, prior_scale=0.0)) == 0.0
    # fixed parameters (one on its floor), data-only objective, warm start
    fx = PhysicsModel(data.grid, cfg, L_init=90.0).fit(data.frame, data.y, train, data.coords,
                                                       fixed={"s": 0.3, "e2": 0.0}, prior_scale=0.0, init=ref.leaves)
    assert fx.params["s"] == 0.3 and fx.params["e2"] == 0.0
    assert fx.fit_info["hooks"] == {"fixed": {"s": 0.3, "e2": 0.0}, "prior_scale": 0.0, "init": True}
    assert fx.fit_info["starts"] == [(0.0, 0.0)]
    with pytest.raises(ValueError, match="not fitted parameters"):
        PhysicsModel(data.grid, cfg).fit(data.frame, data.y, train, data.coords, fixed={"vx": 10.0})


@pytest.mark.parametrize("key", list(NATURAL_LEAF))
def test_natural_leaf_transforms_round_trip_and_match_transform(key):
    g = Grid.from_points(np.arange(20.0) * 30.0, np.arange(20.0) * 30.0, cell=30.0)
    m = PhysicsModel(g, {"roles": {}, "fit_advection": True, "wind": [1.0, 0.0]})
    P = {k: v.detach() for k, v in m._init_params((0.0, 0.0)).items()}
    for u in (-3.0, -0.4, 0.0, 0.7, 2.5):
        x = m.to_natural(key, u)
        assert float(m.to_natural(key, torch.tensor(u, dtype=torch.float64))) == pytest.approx(x, rel=1e-14)
        assert float(m._transform({**P, NATURAL_LEAF[key]: torch.tensor(u, dtype=torch.float64)})[key]) \
            == pytest.approx(x, rel=1e-14)
        assert m.to_leaf(key, x) == pytest.approx(u, abs=1e-9)
        h = 1e-6
        fd = (m.to_natural(key, u + h) - m.to_natural(key, u - h)) / (2 * h)
        assert m.natural_slope(key, u) == pytest.approx(fd, rel=1e-6, abs=1e-9)
    lo, hi, kind = m.natural_bounds(key)
    if kind in ("lin", "log"):
        assert math.isfinite(m.to_leaf(key, lo)) and math.isfinite(m.to_leaf(key, hi))
    if kind == "floor":
        assert math.isfinite(m.to_leaf(key, 0.0))


# ---------------------------------------------------------------------------
# (d) power-scaling: the ArviZ port on exact conjugate Gaussian draws
# ---------------------------------------------------------------------------

def _gauss_toy(mu0, tau, x, sigma, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    prec = 1 / tau ** 2 + 1 / sigma ** 2
    m = (mu0 / tau ** 2 + x / sigma ** 2) / prec
    th = m + rng.standard_normal(n) / math.sqrt(prec)
    return ID.power_scaling_sensitivity(th, -0.5 * ((th - mu0) / tau) ** 2, -0.5 * ((x - th) / sigma) ** 2)


def test_power_scaling_on_conjugate_gaussian_toys():
    strong_prior = _gauss_toy(0.0, 0.1, 0.0, 10.0)
    assert strong_prior["prior"] >= 0.05 > strong_prior["likelihood"]
    assert strong_prior["diagnosis"] == "weak_likelihood"
    strong_lik = _gauss_toy(0.0, 100.0, 0.0, 1.0)
    assert strong_lik["likelihood"] >= 0.05 > strong_lik["prior"] and strong_lik["diagnosis"] == "none"
    conflict = _gauss_toy(0.0, 1.0, 6.0, 1.0)
    assert conflict["prior"] >= 0.05 and conflict["likelihood"] >= 0.05
    assert conflict["diagnosis"] == "prior_data_conflict"
    assert conflict["ess_prior"] > 3900 and conflict["ess_likelihood"] > 3900


def test_pareto_smoothed_weights_match_arviz():
    """``psislw`` is ArviZ 0.20.0's ``psislw`` for a 1-D array, and ``power_scaling_sensitivity`` uses it as
    ArviZ's ``psens``/``_powerscale_lw`` do.  The expected numbers were computed with ArviZ 0.20.0's own
    ``_psislw``/``_gpdfit``/``_gpinv``/``_powerscale_sens`` (extracted from the wheel, sha256 5ec4f2ec…; the port
    agreed to the last bit on 200 random vectors)."""
    lw, k = ID.psislw(np.linspace(-3, 2, 101) ** 3 / 10.0)
    assert k == pytest.approx(0.056273356286089496, rel=1e-12)
    assert lw[0] == pytest.approx(-7.200318112436957, rel=1e-12)
    assert lw[50] == pytest.approx(-4.512818112436957, rel=1e-12)
    assert lw[-1] == pytest.approx(-3.700318112436957, rel=1e-12)
    assert np.exp(lw).sum() == pytest.approx(1.0, rel=1e-12)
    x = np.sin(np.arange(300) * 0.37) * 2 + np.arange(300) / 300
    ps = ID.power_scaling_sensitivity(x, log_prior=-0.5 * (x - 0.3) ** 2)
    assert ps["prior"] == pytest.approx(0.12976960390917533, rel=1e-12)
    assert ps["khat_prior"] is not None and ps["ess_prior"] > 0 and ps["likelihood"] is None
    _, k_short = ID.psislw(np.arange(5.0))                     # a tail of ≤ 4 draws: no Pareto fit
    assert k_short == math.inf


# ArviZ 0.20.0 reference values (PHYSICS_PLAN W3 review X19), produced with ArviZ's OWN functions, not the port:
# `pip download arviz==0.20.0 --no-deps` into an empty scratch directory (wheel sha256 5ec4f2ec180a8305ff3d1108c29e1
# 89944ab939663eb5bc3231ff199a1a5dc36), installed with its dependencies in a throwaway venv outside the project
# (numpy 2.4.6 and scipy 1.16.3 as in the project's venv; xarray 2024.9.0, matplotlib 3.9.2), and called as
# ``arviz.stats.stats._cjs_dist(draws, weights)``; ``_powerscale_sens(draws, lower_weights=, upper_weights=, delta=)``
# with the weights made as ``psens`` makes them (``exp(_powerscale_lw(alpha=α, component_draws=c))`` normalised, α =
# 1/(1+δ) and 1+δ); ``_powerscale_lw(alpha=1.01, component_draws=c)``; and ``arviz.psens(idata, component=...)`` on
# the InferenceData of :func:`_az_idata_inputs` (posterior theta, pointwise log_likelihood y, log_prior theta added
# with ``add_groups``).  The inputs below are rebuilt from the same seeds.  This test does not import arviz.
_AZ_CJS = {"x1_exp03": 0.06200516812467475, "neg_x1_exp03": 0.06782218693778053, "x2_w2": 0.004893227164628284,
           "neg_x2_w2": 0.01734523379843964, "x3_w3": 0.014670574893910155}
_AZ_SENS = {"skew_prior": 2.0674894122077245, "skew_heavy": 0.4922570186031529,
            "skew_heavy_delta05": 2.0676975780797067}
_AZ_LW_UPPER = {0: -7.421142838280725, 1: -6.875017879764181, 399: -6.892482558440427, 181: -5.09586202279013,
                390: -9.144800169382657}
_AZ_PSENS_IDATA = {"prior": 0.1737307866157668, "likelihood": 0.24088145766571395}


def _az_cjs_inputs():
    r = np.random.default_rng
    x1 = r(11).standard_normal(400)
    x2 = r(12).gamma(2.0, size=300)                       # skewed: CJS(x) differs from CJS(−x)
    w2 = r(13).uniform(0.5, 1.5, size=300)
    x3 = r(14).standard_t(3, size=250)
    w3 = np.exp(1.5 * r(15).standard_normal(250))       # heavy weights
    return x1, x2, w2, x3, w3


def _az_sens_inputs():
    th = np.random.default_rng(21).gamma(3.0, 0.5, size=800)                  # skewed draws
    lp = -0.5 * ((th - 1.0) / 0.4) ** 2                                       # an informative prior's log density
    ll = 25.0 * np.random.default_rng(22).standard_t(4, size=800)             # heavy-tailed: the Pareto smoothing acts
    return th, lp, ll


def _az_idata_inputs():
    """A conjugate normal model, 2 chains × 300 draws, 20 observations, prior N(0, 0.5²)."""
    rng = np.random.default_rng(31)
    y = rng.normal(1.5, 1.0, size=20)
    mu0, tau, sigma = 0.0, 0.5, 1.0
    prec = 1 / tau ** 2 + len(y) / sigma ** 2
    m = (mu0 / tau ** 2 + y.sum() / sigma ** 2) / prec
    theta = m + rng.standard_normal((2, 300)) / np.sqrt(prec)
    log_lik = -0.5 * ((y[None, None, :] - theta[..., None]) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    log_prior = -0.5 * ((theta - mu0) / tau) ** 2 - np.log(tau * np.sqrt(2 * np.pi))
    return theta, log_lik, log_prior


def test_cjs_distance_matches_arviz_reference_values():
    x1, x2, w2, x3, w3 = _az_cjs_inputs()
    got = {"x1_exp03": ID.cjs_distance(x1, np.exp(0.3 * x1)), "neg_x1_exp03": ID.cjs_distance(-x1, np.exp(0.3 * x1)),
           "x2_w2": ID.cjs_distance(x2, w2), "neg_x2_w2": ID.cjs_distance(-x2, w2), "x3_w3": ID.cjs_distance(x3, w3)}
    for k, v in _AZ_CJS.items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=0), k


def test_power_scaling_matches_arviz_reference_values():
    th, lp, ll = _az_sens_inputs()
    assert ID.power_scaling_sensitivity(th, log_prior=lp)["prior"] == pytest.approx(_AZ_SENS["skew_prior"], rel=1e-12)
    heavy = ID.power_scaling_sensitivity(th, log_lik=ll)
    assert heavy["likelihood"] == pytest.approx(_AZ_SENS["skew_heavy"], rel=1e-12) and heavy["prior"] is None
    assert ID.power_scaling_sensitivity(th, log_lik=ll, delta=0.05)["likelihood"] == \
        pytest.approx(_AZ_SENS["skew_heavy_delta05"], rel=1e-12)
    lw, _ = ID.psislw((1.01 - 1.0) * ll)                 # ArviZ's _powerscale_lw: psislw((α − 1)·component)
    for i, v in _AZ_LW_UPPER.items():
        assert lw[i] == pytest.approx(v, rel=0, abs=1e-12), i
    assert int(np.argmax(ll)) == 181 and int(np.argmin(ll)) == 390        # the indices pinned above
    _, up, _, _ = ID._psens_weights(ll, 0.01)
    assert up == pytest.approx(np.exp(lw) / np.exp(lw).sum(), rel=1e-12)
    # arviz.psens on an InferenceData: the posterior draws of theta (chains stacked), the log prior, and the
    # pointwise log-likelihood summed over the observations
    theta, log_lik, log_prior = _az_idata_inputs()
    ps = ID.power_scaling_sensitivity(theta.ravel(), log_prior=log_prior.ravel(), log_lik=log_lik.sum(-1).ravel())
    assert ps["prior"] == pytest.approx(_AZ_PSENS_IDATA["prior"], rel=1e-12)
    assert ps["likelihood"] == pytest.approx(_AZ_PSENS_IDATA["likelihood"], rel=1e-12)
    assert ps["diagnosis"] == "prior_data_conflict"          # both ≥ 0.05 (Kallioinen et al. 2024)


def test_cjs_distance_port():
    x = np.random.default_rng(0).standard_normal(500)
    assert ID.cjs_distance(x, np.ones_like(x)) == pytest.approx(0.0, abs=1e-6)     # rounding: cumsum vs linspace
    assert ID.cjs_distance(x, np.full_like(x, 7.3)) == pytest.approx(0.0, abs=1e-6)  # weights are normalised
    w = np.exp(0.5 * x)
    d = ID.cjs_distance(x, w)
    assert 0.0 < d < 1.0 and d == pytest.approx(ID.cjs_distance(x[::-1], w[::-1]))
    assert ID.cjs_distance(np.ones(10), np.arange(10.0)) == 0.0           # no spread: no distance
    assert ID.psens_diagnosis(0.1, 0.2) == "prior_data_conflict" and ID.psens_diagnosis(None, 0.1) is None


# ---------------------------------------------------------------------------
# (e) prior-data conflict
# ---------------------------------------------------------------------------

def test_prior_data_conflict_formula_and_not_testable_cases():
    c = ID.prior_data_conflict(2.0, 0.5, 0.0, 1.0)
    z = 2.0 / math.sqrt(1.25)
    assert c["z"] == pytest.approx(z) and c["p"] == pytest.approx(math.erfc(z / math.sqrt(2)))
    assert c["flag"] is False and c["reason"] is None
    assert ID.prior_data_conflict(4.0, 0.5, 0.0, 1.0)["flag"] is True
    assert ID.prior_data_conflict(-4.0, 0.5, 0.0, 1.0)["z"] < 0
    b = ID.prior_data_conflict(0.0, 0.1, 0.6, 0.3, at_bound=True)
    assert b["p"] is None and "fit bound" in b["reason"]
    n = ID.prior_data_conflict(0.3, 0.1, 0.6, 0.3, identified=False)
    assert n["p"] is None and "not identified" in n["reason"]
    assert "standard error" in ID.prior_data_conflict(0.3, None, 0.6, 0.3)["reason"]


# ---------------------------------------------------------------------------
# (a) independent noise; structural non-identifiability by construction
# ---------------------------------------------------------------------------

def test_iid_noise_adjust_factor_near_one_and_profile_covers_L():
    grid, frame, y, coords, _, cfg = albedo_world(noise=0.5, seed=3, block_m=150.0)
    cl = block_ids(coords, 150.0)
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "e0"], posterior=False,
                     settings={**FAST, "block_m": 150.0})
    assert rep["settings"]["score_covariance"] == "blocks"                     # Chandler & Bate's V by default
    L = _row(rep, "L")
    assert rep["n_clusters"] == 144 and L["identifiability"] == "identified"
    assert L["class_by_covariance"] == {"blocks": "identified", "neighbours": "identified"}
    # one profile of l_I, both adjustments: each interval covers L
    for key in ("ci95", "ci95_neighbours", "ci95_naive"):
        assert _covers(L[key], L_TRUE), key
    # independent noise: the block sandwich agrees with the naive information (c ≈ 1) …
    assert 0.6 < L["adjust_factor"] < 1.6
    for r in rep["linear"]:
        assert 0.5 < r["adjust_factor"] < 2.0, r["key"]
        assert r["se_robust_data_neighbours"] >= r["se_robust_data"] - 1e-12, r["key"]
    # … and adding the adjacent blocks' cross terms can only make it more conservative (V_nb ≥ V_blocks)
    assert L["adjust_factor_neighbours"] <= L["adjust_factor"] + 1e-12
    assert L["se_robust_neighbours"] >= L["se_robust"] - 1e-12
    assert L["ci95_neighbours"][0] <= L["ci95"][0] + 1e-9 and L["ci95_neighbours"][1] >= L["ci95"][1] - 1e-9
    assert len(L["profile"]["lambda_adj_neighbours"]) == len(L["profile"]["grid"])
    # e₀ is absorbed exactly by the gain a: held as flat and structurally non-identifiable
    e0 = _row(rep, "e0")
    assert e0["held"] == "flat" and e0["identifiability"] == "structurally_non_identifiable"
    assert "e0" in rep["hessian"]["held_flat"]
    assert max(e0["profile"]["lambda_adj"]) < ID.FLAT_LAMBDA


def _adjacent(G, ncol):
    cen = np.array([[i % ncol, i // ncol] for i in range(G)])
    A = ((np.abs(cen[:, None, 0] - cen[None, :, 0]) <= 1) & (np.abs(cen[:, None, 1] - cen[None, :, 1]) <= 1))
    np.fill_diagonal(A, False)
    return A.astype(float)


@pytest.mark.parametrize("G, p", [(23, 6), (5, 9)])           # V of full rank, and of rank G − 1 < p
def test_neighbours_score_covariance_is_invariant_to_the_parameter_coordinates(G, p):
    """θ → Aθ maps the scores U → UA⁻¹; Chandler & Bate's V and the neighbours V must map to A⁻ᵀVA⁻¹ (so SEs in
    natural units do not depend on the coordinates the study happens to use).  The positive part of the
    adjacent-block cross-products is taken relative to the block V (``_pos_part_rel``); the plain eigenvalue clip
    of the cross-products that it replaces is shown not to be invariant."""
    rng = np.random.default_rng(G)
    U = rng.standard_normal((G, p)) @ rng.standard_normal((p, p))
    st = ID._Study.__new__(ID._Study)
    st.adj = _adjacent(G, 5)
    for A in (np.diag(10.0 ** rng.uniform(-3, 3, p)), rng.standard_normal((p, p)) + 3 * np.eye(p)):
        Ai = np.linalg.inv(A)
        for mode in ("blocks", "neighbours"):
            V, V2 = st.score_cov(U, mode), st.score_cov(U @ Ai, mode)
            assert np.allclose(A.T @ V2 @ A, V, rtol=1e-7, atol=1e-9 * np.abs(V).max()), mode
        Vb, Vn = st.score_cov(U, "blocks"), st.score_cov(U, "neighbours")
        assert np.linalg.eigvalsh(Vn - Vb).min() >= -1e-9 * np.abs(Vb).max()       # never below Chandler & Bate
    # the plain clip (eigenvalues of the cross-products in the given coordinates) is not invariant
    Ub = U - U.mean(axis=0)
    A = np.diag(10.0 ** rng.uniform(-3, 3, p))

    def clip(Ux):
        w, Q = np.linalg.eigh(Ux.T @ st.adj @ Ux)
        return (Q * np.maximum(w, 0)) @ Q.T

    assert not np.allclose(A.T @ clip(Ub @ np.linalg.inv(A)) @ A, clip(Ub), rtol=1e-3)


def test_study_ses_do_not_depend_on_the_scaling_of_its_coordinates(monkeypatch):
    """End to end: rescaling the study's linear-coefficient coordinates (θ_lin × column RMS by default) leaves
    every SE and c in natural units unchanged, with either score covariance."""
    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.5, noise_range=5.8, seed=1)
    kw = dict(L_init=L_TRUE, params=["L"], posterior=False, settings={**QUICK, "max_points": 2, "block_m": BLOCK_M})
    ref = ID.analyse(grid, frame, y, coords, cl, cfg, **kw)
    real_setup = ID._Study.setup

    def setup(self, model):
        real_setup(self, model)
        self.col = self.col * np.array([3.0, 0.01, 40.0, 0.2, 7.0])[: self.col.size]
        self.col_t = ID._t(self.col)

    monkeypatch.setattr(ID._Study, "setup", setup)
    alt = ID.analyse(grid, frame, y, coords, cl, cfg, **kw)
    a, b = _row(ref, "L"), _row(alt, "L")
    for k in ("se_naive", "se_robust", "se_robust_neighbours", "adjust_factor", "adjust_factor_neighbours"):
        assert b[k] == pytest.approx(a[k], rel=1e-6), k
    for ra, rb in zip(ref["linear"], alt["linear"]):
        for k in ("se_naive_data", "se_robust_data", "se_robust_data_neighbours"):
            assert rb[k] == pytest.approx(ra[k], rel=1e-6), (ra["key"], k)


# ---------------------------------------------------------------------------
# (b) spatially correlated noise
# ---------------------------------------------------------------------------

def test_correlated_noise_shrinks_the_information_and_widens_the_interval():
    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.5, noise_range=5.8, seed=0)
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L"], posterior=False,
                     settings={**FAST, "block_m": BLOCK_M})
    L = _row(rep, "L")
    assert rep["n_clusters"] == 36 and L["adjust_factor"] < 0.2
    assert L["se_robust"] > 2.0 * L["se_naive"]
    lo, hi = L["ci95"]
    nlo, nhi = L["ci95_naive"]
    assert lo < nlo and hi > nhi                                   # the adjusted interval contains the naive one
    assert (hi - lo) > 2.0 * (nhi - nlo)
    prof = L["profile"]
    # the adjusted profile is never above the design's l_A at the l_I-profile point (Cauchy–Schwarz)
    for a, ip in zip(prof["lambda_adj"], prof["lambda_adj_ipath"]):
        assert ip is None or a <= ip + 1e-9
    assert all(r["adjust_factor"] < 0.5 for r in rep["linear"] if r["key"] != "b")


def _coverage(R, noise_range):
    """Coverage of the planted L by the blocks, neighbours and naive 95% intervals of one study per replicate,
    and the median c of each adjustment."""
    hits = {"ci95": 0, "ci95_neighbours": 0, "ci95_naive": 0}
    cs = {"blocks": [], "neighbours": []}
    for s in range(R):
        grid, frame, y, coords, cl, cfg = albedo_world(noise=0.5, noise_range=noise_range, seed=s)
        rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L"], posterior=False,
                         settings={**FAST, "block_m": BLOCK_M})
        L = _row(rep, "L")
        for k in hits:
            hits[k] += _covers(L[k], L_TRUE)
        cs["blocks"].append(L["adjust_factor"])
        cs["neighbours"].append(L["adjust_factor_neighbours"])
    return {k: v / R for k, v in hits.items()}, {k: float(np.median(v)) for k, v in cs.items()}


@pytest.mark.slow
@pytest.mark.parametrize("noise", ["correlated", "independent"])
def test_coverage_of_the_adjusted_profile_interval_for_L(noise):
    """Albedo world, 60 replicates per setting (one chunk each, about 5 minutes), 36 blocks of 300 m.  Correlated
    noise with practical range = the block size (Providence's regime) and independent noise: coverage of the
    planted L by the blocks, neighbours and naive 95% intervals.  The numbers are printed (run with -s) and quoted
    in the module docstring.  No assertion compares the two adjustments: the neighbours V is never below the block
    V, so its interval contains the block one by construction."""
    R = 60
    cov, c = _coverage(R, 5.8 if noise == "correlated" else None)
    print(f"\n{noise} noise: blocks {cov['ci95']:.3f}, neighbours {cov['ci95_neighbours']:.3f}, naive "
          f"{cov['ci95_naive']:.3f}; median c blocks {c['blocks']:.3f}, neighbours {c['neighbours']:.3f}")
    if noise == "correlated":
        assert cov["ci95"] >= 0.7 and 0.8 <= cov["ci95_neighbours"] <= 0.99
        assert cov["ci95_naive"] <= cov["ci95"] - 0.4
        assert c["blocks"] < 0.2
    else:
        assert cov["ci95"] >= 0.85 and cov["ci95_neighbours"] >= 0.85 and cov["ci95_naive"] >= 0.85
        assert 0.6 < c["blocks"] < 1.6


#: The multi-parameter world's planted parameters: L, a canopy-shading source (strength s, saturating with κ) and
#: the evaporative fraction's intercept and canopy slope (natural units, ``NATURAL_LEAF`` keys)
SHADE_TRUE = {"L": L_TRUE, "s": 0.4, "kc": 0.15, "e0": math.log(0.35 / 0.65), "e2": 1.0}
SHADE_BLOCK_M = 180.0
SHADE_CHUNKS, SHADE_R = 4, 15


def shade_world(n=36, dx=30.0, noise=0.3, noise_range=SHADE_BLOCK_M / 30.0 / 1.73, seed=0, truth=SHADE_TRUE):
    """A multi-parameter world closer to Providence: albedo and canopy layers, the data the physics model's own
    forward map at the planted parameters (:data:`SHADE_TRUE`: L, canopy shading s with κ, the EF intercept and
    canopy slope; so the model is right and every profiled parameter has a true value) scaled to unit sd, plus
    noise correlated at block scale (practical range ≈ the 180 m blocks; 36 blocks, 9 free parameters).  The
    shade saturates (κ = 0.15), which separates it from the EF's near-linear canopy slope (with κ at its upper
    bound the two are nearly collinear and the data put s on 0, as on Providence).  Only ``seed`` (the noise)
    changes between replicates.  (grid, frame, y, coords, clusters, cfg)."""
    srng = np.random.default_rng(1000)
    alb = 0.15 + 0.05 * gaussian_random_field((n, n), 2.0, srng)
    can = np.clip(40.0 + 25.0 * gaussian_random_field((n, n), 3.0, srng), 0.0, 100.0)
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx * dx).ravel().astype(float), (yy * dx).ravel().astype(float)
    grid = Grid.from_points(x, y, cell=dx)
    cfg = dict(WORLD_CFG, roles={"albedo": "albedo", "canopy": "canopy"})
    frame = pd.DataFrame({"albedo": alb.ravel(), "canopy": can.ravel()})
    coords = np.column_stack([x, y])
    idx = np.arange(n * n)
    # a two-iteration fit only sets the model's feature constants and solver; the leaves are then the planted ones
    m = PhysicsModel(grid, {**cfg, "max_iter": 2}, L_init=truth["L"]).fit(frame, -(alb.ravel() - alb.mean()), idx,
                                                                          coords)
    terms = m.fit_terms(frame, np.zeros(n * n), idx, coords)
    P = {k: torch.tensor(v, dtype=torch.float64) for k, v in m.leaves.items()}
    for k, v in truth.items():
        P[NATURAL_LEAF[k]] = torch.tensor(m.to_leaf(k, v), dtype=torch.float64)
    with torch.no_grad():
        phi = terms.phi(terms.transform(P))[0].numpy()
    rng = np.random.default_rng(seed)
    eps = gaussian_random_field((n, n), noise_range, rng).ravel() if noise_range else rng.standard_normal(n * n)
    return grid, frame, phi / phi.std() + noise * eps, coords, block_ids(coords, SHADE_BLOCK_M), cfg


def _shade_replicate(seed: int) -> dict:
    grid, frame, y, coords, cl, cfg = shade_world(seed=seed)
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "s"], posterior=False,
                     settings={**FAST, "max_points": 6, "block_m": SHADE_BLOCK_M})
    out = {}
    for key, nat in (("L", "L"), ("s", "s")):
        r = _row(rep, key)
        out[key] = {k: bool(r[c] is not None and _covers(r[c], SHADE_TRUE[nat]))     # no interval: not covered
                    for k, c in (("blocks", "ci95"), ("neighbours", "ci95_neighbours"), ("naive", "ci95_naive"))}
        out[key].update(cls=r["identifiability"], c=r["adjust_factor"])
    return out


@pytest.mark.slow
@pytest.mark.parametrize("chunk", range(SHADE_CHUNKS))
def test_coverage_in_a_multi_parameter_shade_world(chunk, tmp_path_factory):
    """Coverage of the planted L and s in :func:`shade_world` by the blocks, neighbours and naive 95% intervals:
    ``SHADE_CHUNKS`` chunks of ``SHADE_R`` replicates (about 5 minutes each), summarised by the next test when they
    run in one session.
    A study without an adjusted interval (s not assessable on its bound) counts as not covering."""
    res = [_shade_replicate(seed) for seed in range(chunk * SHADE_R, (chunk + 1) * SHADE_R)]
    d = tmp_path_factory.getbasetemp() / "shade_coverage"
    d.mkdir(exist_ok=True)
    (d / f"{chunk}.json").write_text(json.dumps(res))
    for key in ("L", "s"):
        cov = {m: np.mean([r[key][m] for r in res]) for m in ("blocks", "neighbours", "naive")}
        print(f"\nchunk {chunk} {key}: " + ", ".join(f"{m} {v:.2f}" for m, v in cov.items()))
        assert cov["neighbours"] >= 0.5                     # a gross failure only: the summary tests the rate


@pytest.mark.slow
def test_coverage_in_a_multi_parameter_shade_world_summary(tmp_path_factory):
    d = tmp_path_factory.getbasetemp() / "shade_coverage"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if len(files) < SHADE_CHUNKS:
        pytest.skip("run every chunk of test_coverage_in_a_multi_parameter_shade_world in this session first")
    res = [r for f in files for r in json.loads(f.read_text())]
    cov = {key: {m: float(np.mean([r[key][m] for r in res])) for m in ("blocks", "neighbours", "naive")}
           for key in ("L", "s")}
    print(f"\nshade world, {len(res)} replicates: " + "; ".join(
        f"{key}: " + ", ".join(f"{m} {v:.3f}" for m, v in c.items()) for key, c in cov.items()))
    assert cov["L"]["neighbours"] >= SHADE_MIN["L"] and cov["s"]["neighbours"] >= SHADE_MIN["s"]
    assert cov["L"]["naive"] <= cov["L"]["blocks"] - 0.3 and cov["s"]["naive"] <= cov["s"]["blocks"] - 0.3


#: coverage floors of the neighbours interval in the shade world, set below the validation run's 97% (L) and 80%
#: (s) (module docstring): s under-covers there because its estimator is biased upward, which no variance
#: adjustment fixes
SHADE_MIN = {"L": 0.85, "s": 0.7}


QUICK = {"max_points": 3, "ref_starts": 1, "polish_iter": 30}


@pytest.mark.parametrize("case", ["sigmoid_shade", "no_elevation", "four_clusters"])
def test_indefinite_or_singular_curvature_does_not_crash(case):
    """H_I indefinite (sigmoid shade form), singular (no elevation role) or V of rank < p (4 clusters for ~12
    parameters): the study completes with a warning instead of raising (it used to overflow in the quadratic
    profile start, ``math.exp`` of a coordinate far outside the fit bounds)."""
    kw = {"sigmoid_shade": dict(shade_form="sigmoid"),
          "no_elevation": dict(roles={k: k for k in ("albedo", "canopy", "impervious", "ndvi")}),
          "four_clusters": dict(block_m=540.0)}[case]
    grid, frame, y, coords, cl, cfg = city_world(**kw)
    block_m = kw.get("block_m", 270.0)
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=90.0, draws=100, units="degC",
                     settings={"max_points": 2, "ref_starts": 1, "polish_iter": 20, "profile_iter": 15,
                               "adj_iter": 15, "refine": 0, "block_m": block_m},
                     context={"block_m": block_m, "physics": cfg, "window": "day"})
    assert rep["status"] == "ok" and rep["parameters"] and _finite_or_none(rep)
    assert rep["warnings"]                                       # the degenerate curvature is reported
    if case == "sigmoid_shade":
        assert any("H_I is not negative definite" in w for w in rep["warnings"])
    if case == "four_clusters":
        assert rep["n_clusters"] == 4 and any("clusters for" in w for w in rep["warnings"])
    for r in rep["parameters"]:
        assert r["identifiability"] is not None, r["key"]


def test_an_estimate_beyond_its_walk_limit_is_not_walked_backwards():
    """Night world: the data-only evaporative fraction saturates and e₀ lands far below its walk limit
    (logit −30).  The lower side must not be "walked" by clamping to −30, a point *above* the estimate (that
    produced an interval excluding its own estimate); it stays open, and every profile point lies on its side."""
    grid, frame, y, coords, cl, cfg = city_world(window="night")
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=90.0, params=["e0", "e3"], posterior=False,
                     settings={**QUICK, "block_m": 270.0})
    e0 = _row(rep, "e0")
    assert e0["estimate_data"] < -ID.EF_WALK_LOGIT                        # the case under test
    assert e0["ci95"][0] is None and "lower" in e0["open_sides"] and "beyond the profile's walk limit on the lower side" in e0["note"]
    for r in rep["parameters"]:
        lo, hi = r["ci95"] if r["ci95"] else (None, None)
        est = r["estimate_data"]
        assert (lo is None or lo <= est + 1e-9) and (hi is None or hi >= est - 1e-9), (r["key"], r["ci95"], est)


def test_advection_drift_profile_runs_on_its_own_coordinates():
    """With advection fitted, |v| is profiled through its angle (the drift's two leaves) and gets a delta-method
    SE; the planted drift lies inside the adjusted interval."""
    n, dx, L, v = 48, 30.0, 90.0, (120.0, 0.0)
    srng = np.random.default_rng(1000)
    alb = 0.15 + 0.05 * gaussian_random_field((n, n), 2.0, srng)
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx * dx).ravel().astype(float), (yy * dx).ravel().astype(float)
    grid = Grid.from_points(x, y, cell=dx)
    cfg = dict(WORLD_CFG, fit_advection=True, wind=[v[0] / 1800.0, 0.0], v_max_m=400.0, L_max_m=600.0)
    pad = PhysicsModel(grid, cfg).pad
    phi = grid.sample(ops.solve(np.nan_to_num(grid.rasterize(-(alb.ravel() - alb.mean()))), L, v, dx=dx, pad=pad))
    t = phi / phi.std() + 0.3 * np.random.default_rng(0).standard_normal(n * n)
    coords = np.column_stack([x, y])
    rep = ID.analyse(grid, pd.DataFrame({"albedo": alb.ravel()}), t, coords, block_ids(coords, 240.0), cfg,
                     L_init=L, params=["v", "L"], posterior=False,
                     settings={**FAST, "max_points": 5, "ref_starts": 1, "block_m": 240.0})
    assert rep["advection"]["fit"] is True
    vr = _row(rep, "v")
    assert vr["estimate_data"] == pytest.approx(math.hypot(*v), rel=0.3)
    assert vr["se_robust"] > 0 and vr["profile"]["sides_explored"] == ["upper", "lower"]
    assert _covers(vr["ci95"], math.hypot(*v)) and vr["identifiability"] in ("identified", "one_sided",
                                                                             "practically_non_identifiable")
    assert {"vx", "vy"} <= set(rep["hessian"]["parameters"])
    # the drift penalty is part of the data-only objective: profile points are compared with the reference on that
    # objective, so none "beats" it by trading penalty (this used to trigger two futile reference refits and a
    # "not a maximum" warning), and Λ is 0 at the estimate (no rebasing)
    assert rep["reference"]["data_only"]["restarts"] == 0
    assert not any("improve" in w for w in rep["warnings"]), rep["warnings"]
    for r in (vr, _row(rep, "L")):
        prof = r["profile"]
        i0 = prof["grid"].index(r["estimate_data"])
        assert prof["lambda_naive"][i0] == 0.0 and min(prof["lambda_naive"]) == 0.0


def test_a_profile_maximises_over_parameters_held_on_a_bound_or_flat():
    """A profile maximises over every other parameter, those held at θ̂ included.  Sink world: w is on its bound
    and e₀ flat (confounded with a while w = 0) at θ̂.  Along L, w leaves its bound above L̂; along w, e₀ is no
    longer confounded.  Each profile's outermost points must be as good as an independent fit with ψ fixed and
    every other parameter free, started with w released inside its bound.  With them frozen (before the fix) L's
    point at 107.6 m had Λ_I = 6.18 against 5.78 from the free fit, and w's profile was too steep likewise."""
    grid, frame, y, coords, cl, cfg = sink_world()
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "w"], posterior=False,
                     settings={**FAST, "max_points": 6, "block_m": BLOCK_M})
    assert rep["hessian"]["held_at_bound"] == ["w"] and rep["hessian"]["held_flat"] == ["e0"]
    idx = np.arange(y.size)
    ref = PhysicsModel(grid, cfg, L_init=L_TRUE).fit(frame, y, idx, coords, prior_scale=0.0)
    l_hat = rep["reference"]["data_only"]["loglik_independence"]
    moved = 0
    for key, nat in (("L", "L"), ("w", "w")):
        prof = _row(rep, key)["profile"]
        for i in (0, len(prof["grid"]) - 1):
            v = prof["grid"][i]
            if v == _row(rep, key)["estimate_data"] or (key == "w" and v <= 0.0):
                continue
            start = {**ref.leaves, "w": -3.0}
            free = PhysicsModel(grid, cfg, L_init=L_TRUE).fit(frame, y, idx, coords, prior_scale=0.0, fixed={nat: v},
                                                             init=start)
            lam_free = 2.0 * (l_hat + 0.5 * y.size * math.log(free.train_rmse ** 2))
            assert prof["lambda_naive"][i] <= lam_free + 0.02, (key, v, prof["lambda_naive"][i], lam_free)
            moved += key == "L" and free.params["w"] > 0.5
    assert moved                                    # the case under test: w leaves its bound along L's profile


def _outward(prof, x0, key):
    """(lower side, upper side) of a profile curve ordered outward from x0."""
    g, lam = prof["grid"], prof[key]
    lo = [lam[i] for i in sorted((i for i in range(len(g)) if g[i] < x0), key=lambda i: -g[i])]
    hi = [lam[i] for i in sorted((i for i in range(len(g)) if g[i] > x0), key=lambda i: g[i])]
    return lo, hi


def test_profiles_are_multi_started_and_monotone():
    """Every profile point is maximised from several starts and the best kept; a point whose Λ exceeds the next
    point outward's is re-maximised from that point.  On the city world (13 parameters, a₁ on its bound) the
    single-start version gave L's adjusted profile 3.94 at 91.99 m and 3.48 at 92.22 m (a failed maximisation:
    the upper 95% end came out at 91.9 m instead of 92.8 m)."""
    grid, frame, y, coords, cl, cfg = city_world()
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L"], posterior=False,
                     settings={**FAST, "max_points": 6, "block_m": 270.0})
    L = _row(rep, "L")
    prof = L["profile"]
    assert prof["monotone"] is True and prof["non_monotone"] == []
    for key in ("lambda_naive", "lambda_adj", "lambda_adj_neighbours"):
        for seq in _outward(prof, L["estimate_data"], key):
            assert all(b >= a - ID.MONO_TOL for a, b in zip(seq, seq[1:])), (key, seq)
    # the adjusted profile is never above the design's value at the l_I-profile point (that point is a start)
    for a, ip in zip(prof["lambda_adj"], prof["lambda_adj_ipath"]):
        assert ip is None or a <= ip + 1e-9


def test_non_monotone_profiles_are_repaired_or_reported(monkeypatch):
    """A point whose Λ exceeds the next point outward's (a failed maximisation) is re-maximised from that point's
    optimum and the better optima kept; what cannot be repaired is recorded and warned about."""
    st = ID._Study.__new__(ID._Study)
    st.modes, st.warnings = ["blocks"], []

    def pt(v, lam_i, lam_a):
        return {"value": v, "l": -lam_i / 2, "lam_i": lam_i, "loss": 0.0, "lam_a": {"blocks": lam_a},
                "lam_a_ipath": {"blocks": lam_a}, "R": {"blocks": lam_a / lam_i}, "leaves": {"v": v, "_adj": {}}}

    seq = [pt(1.0, 1.0, 0.5), pt(2.0, 6.0, 3.0), pt(3.0, 4.0, 2.0)]
    redo = []

    def remax(st_, r, value, warms, dat, cache):
        redo.append((value, warms[0]["v"]))                    # started from the outer neighbour's optimum
        return pt(value, 2.5, 1.2)

    monkeypatch.setattr(ID, "_point", remax)
    assert ID._repair_monotone(st, {"key": "L"}, seq, {}, {}) == []
    assert redo == [(2.0, 3.0)] and seq[1]["lam_i"] == 2.5 and seq[1]["lam_a"]["blocks"] == 1.2
    assert seq[1]["R"]["blocks"] == pytest.approx(1.2 / 2.5) and not st.warnings
    # re-maximisation that does not help: reported (no Λ is ever raised by a merge)
    seq = [pt(1.0, 1.0, 0.5), pt(2.0, 6.0, 3.0), pt(3.0, 4.0, 2.0)]
    monkeypatch.setattr(ID, "_point", lambda *a: pt(2.0, 7.0, 3.5))
    left = ID._repair_monotone(st, {"key": "L"}, seq, {}, {})
    assert seq[1]["lam_i"] == 6.0 and {v["profile"] for v in left} == {"naive", "blocks"}
    assert left[0]["inner"] == 2.0 and left[0]["outer"] == 3.0 and any("not monotone" in w for w in st.warnings)
    # a profile likelihood can be multimodal: the warning does not claim the violation must be a failed maximisation
    assert all("multimodal" in w for w in st.warnings if "not monotone" in w)


def _stub_study(modes=("blocks",)):
    st = ID._Study.__new__(ID._Study)
    st.modes, st.warnings, st.l_hat = list(modes), [], 0.0
    return st


def _stub_row(lams, key="w", x_hat=0.0, at_bound=None, held=None):
    """A row whose upper side has profile points at x = 1, 2, ... with naive and adjusted Λ ``lams`` (R = 1)."""
    pts = [{"value": float(i + 1), "x": float(i + 1), "l": -lam / 2.0, "lam_i": lam, "lam_a": {"blocks": lam},
            "lam_a_ipath": {"blocks": lam}, "R": {"blocks": 1.0}, "leaves": {}} for i, lam in enumerate(lams)]
    return {"key": key, "label": key, "unit": "W/m²", "_nat": key, "_points": pts, "_x_hat": x_hat,
            "_sides": {"upper": {"points": list(pts), "edge": False}}, "_thr": ID.CHI2_95, "at_bound": at_bound,
            "held": held, "estimate_data": x_hat, "boundary_adjustment": None, "notes": []}


def test_an_interval_ends_at_the_outermost_crossing_of_a_profile_that_falls_back():
    """A profile likelihood can be multimodal: a side that crosses 3.841 and falls back below it further out does
    not end at its first crossing.  It ends at the outermost crossing (the hull of {ψ : Λ ≤ 3.841}), or is open when
    the outermost explored point is below the threshold; either way it is recorded and warned about."""
    seq = [{"x": 1.0, "lam": 1.0}, {"x": 2.0, "lam": 4.5}, {"x": 3.0, "lam": 3.0}, {"x": 4.0, "lam": 6.0}]
    first = ID._first_crossing(seq, 0.0, 0.0, "lam")
    hull = ID._crossing(seq, 0.0, 0.0, "lam")
    assert 1.0 < first < 2.0 and 3.0 < hull < 4.0
    assert ID._crossing(seq[:3], 0.0, 0.0, "lam") is None                   # falls back and never crosses again
    mono = [{"x": 1.0, "lam": 1.0}, {"x": 2.0, "lam": 4.5}, {"x": 3.0, "lam": 6.0}]
    assert ID._crossing(mono, 0.0, 0.0, "lam") == ID._first_crossing(mono, 0.0, 0.0, "lam")
    st = _stub_study()
    r = _stub_row([1.0, 4.5, 3.0])
    ID._finish_profile(st, r, 0.0)
    assert r["ci95"][1] is None and "upper" in r["open_sides"] and r["identifiability"] != "identified"
    fb = r["profile"]["fell_back"]
    assert {v["profile"] for v in fb} == {"naive", "blocks"} and all(v["end"] is None for v in fb)
    assert any("falls back below it further out" in w and "is open" in w for w in st.warnings)
    st = _stub_study()
    r = _stub_row([1.0, 4.5, 3.0, 6.0])
    ID._finish_profile(st, r, 0.0)
    assert r["ci95"][1] == pytest.approx(hull) and r["ci95_naive"][1] == pytest.approx(hull)
    assert any("ends at the outermost crossing" in w for w in st.warnings)
    st = _stub_study()
    r = _stub_row([1.0, 4.5, 6.0])
    ID._finish_profile(st, r, 0.0)
    assert r["profile"]["fell_back"] == [] and not st.warnings


# ---------------------------------------------------------------------------
# (c) a canopy layer the model cannot use; (e) a parameter on its fit bound
# ---------------------------------------------------------------------------

def test_constant_canopy_makes_s_non_identifiable_and_prior_dominated():
    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.3, seed=1, canopy=0.0,
                                                   roles={"albedo": "albedo", "canopy": "canopy"})
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "s", "kappa_canopy", "e2"],
                     draws=2000, settings={**FAST, "max_points": 5, "block_m": BLOCK_M})
    s = _row(rep, "s")
    assert s["held"] == "flat" and s["identifiability"] == "structurally_non_identifiable"
    assert max(s["profile"]["lambda_adj"]) < ID.FLAT_LAMBDA and s["se_robust"] is None
    assert s["prior_share_run"] > 0.95
    assert s["posterior"]["method"] == "laplace"
    assert s["psens"]["prior"] >= 0.05 and s["psens"]["likelihood"] < 0.05
    assert s["psens"]["diagnosis"] == "weak_likelihood"
    assert s["conflict"]["p"] is None and "not identified" in s["conflict"]["reason"]
    assert {"s", "kappa_canopy", "e2"} <= set(rep["hessian"]["held_flat"])
    L = _row(rep, "L")
    assert L["identifiability"] == "identified" and _covers(L["ci95"], L_TRUE)
    assert L["prior_share_run"] < 0.05                               # L has no prior
    assert set(rep["posterior"]["flat_prior_parameters"]) >= {"s", "kappa_canopy", "e2"}


def _cross(prof, key, thr, x0):
    """The upper crossing of ``thr`` by a one-sided profile, interpolated linearly in √Λ."""
    pts = sorted((g, l) for g, l in zip(prof["grid"], prof[key]) if g >= x0)
    for (g0, l0), (g1, l1) in zip(pts, pts[1:]):
        if l1 >= thr > l0:
            a, b = math.sqrt(max(l0, 0.0)), math.sqrt(l1)
            return g0 + (math.sqrt(thr) - a) / (b - a) * (g1 - g0)
    return None


def test_a_parameter_on_its_fit_bound_gets_a_one_sided_profile_and_no_conflict_test():
    """A boundary maximum (Self & Liang 1987): the profile is walked from the bound inward, its one-sided 95% bound
    is where Λ reaches χ²₁(0.90) = 2.706 (naive and adjusted), and the adjustment is the scalar c = Σ_naive/Σ_rob of
    the well-conditioned curvature at the bound, Λ_A = c·Λ_I."""
    # canopy *warms* a little in this world, so the data push the shading strength s (≥ 0) onto its lower bound
    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.3, seed=2, canopy_effect=0.1,
                                                   roles={"albedo": "albedo", "canopy": "canopy"})
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "s"], draws=500,
                     settings={**FAST, "max_points": 6, "block_m": BLOCK_M})
    s = _row(rep, "s")
    assert s["at_bound"] == {"side": "lower", "bound": 0.0} and s["held"] == "bound"
    assert s["profile"]["sides_explored"] == ["upper"] and s["ci95"][0] == 0.0
    assert "s" in rep["hessian"]["held_at_bound"]
    assert s["threshold"] == pytest.approx(2.705543454095404) and _row(rep, "L")["threshold"] == ID.CHI2_95
    ba = s["boundary_adjustment"]
    assert ba["computable"] and ba["vif"] <= ID.BOUND_MAX_VIF and ba["min_eig_unit"] >= ID.BOUND_MIN_EIG
    prof = s["profile"]
    for mode, sfx in (("blocks", ""), ("neighbours", "_neighbours")):
        c = ba["c"][mode]
        assert 0 < c and s["adjust_factor" + sfx] == pytest.approx(c, rel=1e-9)
        for lam_a, lam_i in zip(prof["lambda_adj" + sfx], prof["lambda_naive"]):
            assert lam_a == pytest.approx(c * lam_i, rel=1e-9, abs=1e-12)
        assert s["ci95" + sfx][1] == pytest.approx(_cross(prof, "lambda_adj" + sfx, 2.705543454095404, 0.0), rel=1e-9)
    assert s["ci95_naive"][1] == pytest.approx(_cross(prof, "lambda_naive", 2.705543454095404, 0.0), rel=1e-9)
    assert s["ci95_naive"][1] < _cross(prof, "lambda_naive", ID.CHI2_95, 0.0)   # the one-sided bound is tighter
    # one side crosses, the other is open at the physical limit 0 the estimate sits on: one-sided, not identified
    assert s["identifiability"] == "one_sided" and s["open_sides"] == ["lower"] and s["sides_at_limit"] == ["lower"]
    assert s["summary"].startswith(f"s ≤ {ID._g(s['ci95_conservative'][1])}; lower side open at the physical limit 0")
    assert "Self & Liang 1987" in s["summary"] and s["ci95_conservative"][1] >= s["ci95"][1]
    assert s["conflict"]["p"] is None and "fit bound" in s["conflict"]["reason"]
    assert s["posterior"]["method"] == "profile" and s["posterior"]["mean"] > 0.0
    assert s["psens"]["method"] == "profile" and s["psens"]["prior"] is not None
    assert s["estimate_run"] >= s["estimate_data"] - 1e-9
    md = ID.identifiability_markdown(rep)
    assert "at lower bound" in md and "Canopy shading strength" in md


def _side(st_, x=None, why=None):
    return (st_, x, why)


@pytest.mark.parametrize("sides, at_bound, cls, open_, limit, text", [
    ({"lower": ("cross", 0.2, None), "upper": ("cross", 0.5, None)}, None, "identified", [], [], "0.2 ≤ s ≤ 0.5"),
    ({"lower": ("limit", 0.0, "at the physical limit 0"), "upper": ("cross", 0.5, None)}, None, "one_sided",
     ["lower"], ["lower"], "s ≤ 0.5; lower side open at the physical limit 0"),
    ({"lower": ("cross", 0.2, None), "upper": ("open", None, "at the end of the search range (1)")}, None,
     "one_sided", ["upper"], [], "s ≥ 0.2; upper side open at the end of the search range (1)"),
    ({"upper": ("cross", 0.49, None)}, {"side": "lower", "bound": 0.0}, "one_sided", ["lower"], ["lower"],
     "s ≤ 0.49; lower side open at the physical limit 0 (the estimate is on it); one-sided 95% bound"),
    ({"lower": ("limit", 0.0, "at the physical limit 0"), "upper": ("limit", 1.0, "at the physical limit 1")},
     None, "practically_non_identifiable", ["lower", "upper"], ["lower", "upper"], "lower side open"),
])
def test_raue_classes_keep_the_informative_side(sides, at_bound, cls, open_, limit, text):
    """A side that reaches a physical limit without crossing is open (practically non-identifiable on that
    side), not identified; the class says which side crossed and the summary keeps the informative part."""
    r = {"key": "s", "unit": "fraction", "held": "bound" if at_bound else None, "at_bound": at_bound}
    got, open_sides, at_limit = ID._classify(r, sides, flat=False)
    assert (got, open_sides, at_limit) == (cls, open_, limit)
    assert ID._summary(r, got, ID._all_sides(r, sides)).startswith(text)
    assert ID._classify(r, sides, flat=True)[0] == "structurally_non_identifiable"
    # the conservative union of two adjusted sides: open beats a limit beats a crossing; the wider crossing
    assert ID._union_status([("cross", 0.4, None), ("cross", 0.5, None)], "upper")[1] == 0.5
    assert ID._union_status([("cross", 0.2, None), ("limit", 0.0, "x")], "lower")[0] == "limit"
    assert ID._union_status([("limit", 0.0, "x"), ("open", None, "y")], "lower")[0] == "open"


def test_one_sided_parameters_are_not_identified_in_the_register():
    from sparc.core import mechanisms as M

    row = {"key": "s", "label": "Canopy shading strength s", "unit": "fraction of absorbed sunlight",
           "identifiability": "one_sided", "ci95": [0.0, 0.49], "open_sides": ["lower"], "sides_at_limit": ["lower"],
           "summary": "s ≤ 0.49; lower side open at the physical limit 0 (the estimate is on it)",
           "at_bound": {"side": "lower", "bound": 0.0}, "estimate_data": 0.0, "estimate_run": 0.3}
    p = M._ident_param(row, True)
    assert p["class"] == "one_sided" and p["class"] in M.NOT_IDENTIFIED and not p["identified"]
    assert p["reaches_zero"] and "the data alone do not exclude s = 0 (s ≤ 0.49; lower side open" in p["text"]
    p = M._ident_param({**row, "key": "L", "label": "Mixing length L", "unit": "m", "ci95": [200.0, None],
                        "open_sides": ["upper"], "sides_at_limit": [], "at_bound": None,
                        "summary": "L ≥ 200 m; upper side open at the end of the search range (2000)"}, True)
    assert not p["reaches_zero"] and "bounded by the data alone on one side only (L ≥ 200 m" in p["text"]


def test_a_bound_parameter_without_a_well_conditioned_curvature_is_not_assessable():
    """Canopy warms strongly here: at s = 0 l_I is not concave along s (the extended H_I is indefinite), so no block
    adjustment is computable from it.  No adjusted interval is reported (the vertical adjustment from the floored
    curvature used to give one); the naive one-sided bound is, labelled as unadjusted, and s is not assessable,
    with no profile posterior and a register sentence that says so."""
    from sparc.core import mechanisms as M

    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.3, seed=2, canopy_effect=1.0,
                                                   roles={"albedo": "albedo", "canopy": "canopy"})
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["s"], draws=200,
                     settings={**QUICK, "max_points": 4, "ref_starts": 2, "polish_iter": 60, "block_m": BLOCK_M})
    s = _row(rep, "s")
    ba = s["boundary_adjustment"]
    assert s["at_bound"]["side"] == "lower" and ba["computable"] is False and ba["c"] is None and ba["reason"]
    assert s["identifiability"] == "not_assessable" and s["class_by_covariance"]["blocks"] == "not_assessable"
    assert s["ci95"] is None and s["ci95_neighbours"] is None and s["se_robust"] is None
    assert s["ci95_naive"][0] == 0.0 and s["ci95_naive_label"] == "unadjusted (too narrow under spatial dependence)"
    assert s["ci95_naive"][1] == pytest.approx(_cross(s["profile"], "lambda_naive", ID.CHI2_90, 0.0), rel=1e-9)
    assert s["posterior"]["method"] is None and "not assessable" in s["posterior"]["note"] and s["psens"] is None
    assert "unadjusted (too narrow" in ID.identifiability_markdown(rep)
    reg = M._ident_param(s, True)
    assert reg["class"] in M.NOT_IDENTIFIED and "not assessable" in reg["text"] and "too narrow" in reg["text"]


def test_a_poor_laplace_approximation_flags_its_summaries_unreliable(monkeypatch):
    """When the module's own adequacy check calls the Laplace approximation poor, the Laplace rows' posterior and
    power-scaling carry ``reliable: false`` with the reason (and are not shown as numbers in the markdown), while a
    parameter on its bound keeps its 1-D profile posterior, described as such and not caveated as Laplace; the
    Laplace posterior states which parameters it is conditional on.  The check is forced to fail by raising its
    threshold above 1 (an ESS share can never reach it)."""
    monkeypatch.setattr(ID, "LAPLACE_MIN_ESS_FRAC", 1.01)
    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.3, seed=2, canopy_effect=0.1,
                                                   roles={"albedo": "albedo", "canopy": "canopy"})
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "s", "e0"], draws=200,
                     settings={**FAST, "max_points": 4, "block_m": BLOCK_M})
    post = rep["posterior"]
    assert post["laplace_adequate"] is False and post["reliable"] is False and "poor" in post["reason"]
    assert "s" in post["conditional_on"] and post["profile_parameters"] == ["s"] and "L" in post["laplace_parameters"]
    L, s = _row(rep, "L"), _row(rep, "s")
    for part in ("posterior", "psens"):
        assert L[part]["method"] == "laplace" and L[part]["reliable"] is False and "poor" in L[part]["reason"]
        assert s[part]["method"] == "profile" and s[part]["reliable"] is True and s[part]["reason"] is None
        assert "1-D profile posterior" in s[part]["description"] and "Laplace adequacy check does not apply" in \
            s[part]["description"]
    desc = L["posterior"]["description"]
    assert "conditional on" in desc and "held fixed" in desc and " s" in desc and L["posterior"]["conditional_on"] == \
        post["conditional_on"]
    md = ID.identifiability_markdown(rep)
    line = next(ln for ln in md.splitlines() if ln.startswith("| Mixing length"))
    assert line.count("not reliable (Laplace poor)") == 2
    assert "(1-D profile)" in next(ln for ln in md.splitlines() if ln.startswith("| Canopy shading"))
    w = next(x for x in rep["warnings"] if "Laplace approximation" in x)
    assert "1-D profile posteriors of s do not use it" in w and "flagged unreliable" in w
    # the register does not present them (posterior / psens left out, the reason given in words)
    from sparc.core import mechanisms as M

    reg_row = M._ident_param(L, False)
    assert reg_row["posterior"] is None and reg_row["psens"] is None and reg_row["posterior_unreliable"]
    assert "not reported" in reg_row["text"]
    assert M._ident_param(s, False)["psens"]["method"] == "profile"


# ---------------------------------------------------------------------------
# (f) the CLI on a small synthetic run directory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args, msg", [(["--draws", "0"], "draws must be an integer ≥ 100"),
                                       (["--max-points", "0"], "max_points must be an integer ≥ 2"),
                                       (["--max-points", "1"], "max_points must be an integer ≥ 2"),
                                       (["--params", "L,sss,kappa"], "unknown physics parameter(s) 'sss'"),
                                       (["--coarse", "-5"], "coarse must be a positive")])
def test_cli_and_api_reject_bad_arguments_before_any_fit(tmp_path, monkeypatch, capsys, args, msg):
    """--draws 0, --max-points 0 or 1 and unknown --params names fail fast with a clear message: nothing is loaded
    or fitted (the reference fit and the run loader are booby-trapped)."""
    from sparc.core import cli

    def boom(*a, **k):
        raise AssertionError("fitted or loaded before the arguments were checked")

    monkeypatch.setattr(ID._Study, "reference", boom)
    monkeypatch.setattr(ID, "identifiability_report", boom)
    (tmp_path / "manifest.json").write_text("{}")
    assert cli.main(["identifiability", "-r", str(tmp_path), *args]) == 2
    assert msg in capsys.readouterr().err
    grid, frame, y, coords, cl, cfg = albedo_world(n=20)
    kw = {"--draws": "draws", "--max-points": "max_points", "--params": "params"}
    if args[0] in kw:
        val = args[1] if args[0] == "--params" else int(args[1])
        with pytest.raises(ValueError, match=msg.split("(")[0].strip()):
            ID.analyse(grid, frame, y, coords, cl, cfg, **{kw[args[0]]: val})
    assert ID.check_arguments("L,kappa,kc", 2, 100) == ["L", "kappa_canopy"]
    assert ID.check_arguments(None, None, 0, posterior=False) is None      # draws unused without the posterior

@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    out = tmp_path_factory.mktemp("ident_demo")
    info = write_demo_project(out, n=40, seed=0)
    cfg = load_core_config(info["config_path"])
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": True}
    cfg.raw["stacker"]["epochs"] = 30
    cfg.raw["stacker"]["tune_lambda"] = [0.0]
    cfg.raw["cv"]["baselines"] = False
    res = run_core(cfg, stages=("S0", "S1", "S2", "S3"), fast=True)
    return info["config_path"], res.run_dir


def test_cli_writes_identifiability_json_and_the_manifest_section(small_run, capsys):
    from sparc.core import cli
    from sparc.core.pipeline import POST_RUN_KEYS

    config_path, run = small_run
    args = ["identifiability", "-r", str(run), "--params", "L,s,kappa", "--max-points", "3", "--draws", "200"]
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    assert "Profile likelihood per parameter" in printed and "identifiability.json" in printed
    rep = json.loads((run / "identifiability.json").read_text())
    man = json.loads((run / "manifest.json").read_text())
    assert man["identifiability"]["parameters"] == rep["parameters"]
    # the register is refreshed with the study in the same manifest update (PHYSICS_PLAN W3, part 2)
    assert [(e["section"], e["source"]) for e in man["post_run"][-2:]] == [("identifiability", "identifiability"),
                                                                           ("mechanisms", "identifiability")]
    assert "refreshed mechanisms.json and the manifest's mechanisms section (the register uses this study)" in printed
    assert rep["register"] == {"refreshed": True, "identifiability": "used", "note": None}
    reg = json.loads((run / "mechanisms.json").read_text())
    assert reg == man["mechanisms"] and reg["identifiability"]["status"] == "used"
    assert reg["identifiability"]["source"] == "manifest" and set(reg["identifiability"]["parameters"]) <= {
        "L", "s", "kappa_canopy"}
    assert reg["panel"]["identifiability_study"] and "raue2009" in reg["references"]
    profiled = {r["key"] for r in rep["parameters"] if r["estimate_data"] is not None}
    assert profiled and profiled == {r["key"] for r in reg["panel"]["rows"] if r.get("identifiability")}
    assert "identifiability" in POST_RUN_KEYS
    assert [r["key"] for r in rep["parameters"]] == ["L", "s", "kappa_canopy"]
    assert rep["n_clusters"] == man["cv"]["n_blocks"] and rep["cluster"]["block_m"] == man["cv"]["block_m"]
    assert rep["advection"]["fit"] is (man["physics_advection"]["kept"] if man.get("physics_advection") else
                                       rep["advection"]["fit"])
    assert set(rep["references"]) == {"raue2009", "chandler2007", "ribatet2012", "kallioinen2024", "evans2006",
                                      "conley1999", "selfliang1987"}
    for key in ("schema", "n", "n_clusters", "cluster", "coarse_m", "advection", "reference", "parameters",
                "linear", "derived", "settings", "timings", "references"):
        assert key in rep, key
    for r in rep["parameters"]:
        for key in ("key", "label", "unit", "estimate_run", "estimate_data", "at_bound", "se_naive", "se_robust",
                    "adjust_factor", "profile", "ci95", "ci95_naive", "identifiability", "open_sides", "pull",
                    "prior", "prior_share_run", "posterior", "psens", "conflict", "note"):
            assert key in r, key
    assert rep["derived"]["r_a"]["unit"] == "s/m" and "fold_jackknife" in rep["derived"]["r_a"]
    # a second run with the same settings gives the same report (seeded), and its own post_run entry
    assert cli.main(args + ["-p", str(config_path)]) == 0
    again = json.loads((run / "identifiability.json").read_text())
    for a, b in zip(rep["parameters"], again["parameters"]):
        assert a["ci95"] == pytest.approx(b["ci95"], rel=1e-9, abs=1e-12) and a["estimate_data"] == b["estimate_data"]
    man2 = json.loads((run / "manifest.json").read_text())
    assert [e["section"] for e in man2["post_run"]].count("identifiability") == 2
    assert cli.main(["identifiability", "-r", str(run / "missing")]) == 2
    # `sparc core mechanisms` reads the study from the manifest by itself
    (run / "mechanisms.json").unlink()
    assert cli.main(["mechanisms", "-r", str(run)]) == 0
    assert "**Identifiability from the data alone (W3).**" in capsys.readouterr().out
    assert json.loads((run / "mechanisms.json").read_text())["identifiability"]["status"] == "used"


def test_a_rerun_keeps_the_study_but_the_register_ignores_it_once_the_fit_changes(small_run, tmp_path,
                                                                                    monkeypatch):
    """A re-run keeps the post-run ``identifiability`` section (POST_RUN_KEYS); when it refits the physics, the
    register (recomputed with the kept study) ignores it with a note and a ``physics.identifiability_stale``
    warning, and methods.md says it is not used."""
    import shutil

    from sparc.core import cli, progress
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core

    config_path, run0 = small_run
    run = tmp_path / "rerun"
    shutil.copytree(run0, run)
    if not (run / "identifiability.json").exists():
        assert cli.main(["identifiability", "-r", str(run), "--params", "L", "--max-points", "3", "--draws",
                         "200"]) == 0
    study = json.loads((run / "identifiability.json").read_text())
    warned = []
    real_warn = progress.warn
    monkeypatch.setattr(progress, "warn", lambda code, msg, **kw: (warned.append((code, msg)),
                                                                   real_warn(code, msg, **kw)))
    cfg = load_core_config(config_path)
    cfg.raw["models"] = {"ols": True, "mgwr": False, "gwrf": False, "gam": False, "physics": True}
    cfg.raw["stacker"]["epochs"] = 30
    cfg.raw["stacker"]["tune_lambda"] = [0.0]
    cfg.raw["cv"]["baselines"] = False
    cfg.raw["physics"]["max_iter"] = 3                       # a different physics fit on the same cells and folds
    run_core(cfg, stages=("S0", "S1", "S2", "S3"), fast=True, run_dir=run)
    man = json.loads((run / "manifest.json").read_text())
    assert man["identifiability"]["run"] == study["run"]                 # kept (POST_RUN_KEYS)
    ib = man["mechanisms"]["identifiability"]
    assert ib["status"] == "ignored" and "different physics fit" in ib["note"], ib
    assert json.loads((run / "mechanisms.json").read_text())["identifiability"] == ib
    assert [c for c, _ in warned if c == "physics.identifiability_stale"] == ["physics.identifiability_stale"]
    assert "## Identifiability\n\nNot used: identifiability study ignored" in (run / "methods.md").read_text()
