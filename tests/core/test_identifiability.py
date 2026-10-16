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
    out = {}
    for mode in ("blocks", "neighbours"):
        rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "e0"], posterior=False,
                         settings={**FAST, "score_covariance": mode, "block_m": 150.0})
        out[mode] = rep
        L = _row(rep, "L")
        assert rep["n_clusters"] == 144 and L["identifiability"] == "identified"
        assert _covers(L["ci95"], L_TRUE) and _covers(L["ci95_naive"], L_TRUE)
    # independent noise: the block sandwich agrees with the naive information (c ≈ 1) …
    assert 0.6 < _row(out["blocks"], "L")["adjust_factor"] < 1.6
    for r in out["blocks"]["linear"]:
        assert 0.5 < r["adjust_factor"] < 2.0, r["key"]
    # … and adding the adjacent blocks' cross terms can only make it more conservative
    assert _row(out["neighbours"], "L")["adjust_factor"] <= _row(out["blocks"], "L")["adjust_factor"] + 1e-12
    # e₀ is absorbed exactly by the gain a: held as flat and structurally non-identifiable
    e0 = _row(out["blocks"], "e0")
    assert e0["held"] == "flat" and e0["identifiability"] == "structurally_non_identifiable"
    assert "e0" in out["blocks"]["hessian"]["held_flat"]
    assert max(e0["profile"]["lambda_adj"]) < ID.FLAT_LAMBDA


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


def _coverage(R, noise_range, mode):
    hits_a = hits_n = 0
    cs = []
    for s in range(R):
        grid, frame, y, coords, cl, cfg = albedo_world(noise=0.5, noise_range=noise_range, seed=s)
        rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L"], posterior=False,
                         settings={**FAST, "score_covariance": mode, "block_m": BLOCK_M})
        L = _row(rep, "L")
        hits_a += _covers(L["ci95"], L_TRUE)
        hits_n += _covers(L["ci95_naive"], L_TRUE)
        cs.append(L["adjust_factor"])
    return hits_a / R, hits_n / R, float(np.median(cs))


@pytest.mark.slow
def test_coverage_of_the_adjusted_profile_interval_for_L():
    """60 replicates per setting, 36 blocks of 300 m.  Correlated noise with practical range = the block size
    (Providence's regime): the adjusted interval must cover the planted L in 85–99% of replicates and the naive
    one clearly less; independent noise: the adjusted interval stays ≥ 85%.  The numbers are printed (run with
    -s) and quoted in the module docstring."""
    R = 60
    cov_a, cov_n, c_med = _coverage(R, 5.8, "neighbours")
    print(f"\ncorrelated (practical range = block), neighbours V: adjusted {cov_a:.3f}, naive {cov_n:.3f}, "
          f"median c {c_med:.3f}")
    assert 0.85 <= cov_a <= 0.99
    assert cov_n <= cov_a - 0.4
    blk_a, _, blk_c = _coverage(R, 5.8, "blocks")
    print(f"correlated (practical range = block), blocks V (Chandler & Bate): adjusted {blk_a:.3f}, "
          f"median c {blk_c:.3f}")
    assert blk_a <= cov_a                                          # the block-only V under-covers here
    iid_a, iid_n, iid_c = _coverage(R, None, "neighbours")
    print(f"independent noise, neighbours V: adjusted {iid_a:.3f}, naive {iid_n:.3f}, median c {iid_c:.3f}")
    assert iid_a >= 0.85 and iid_n >= 0.85


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
    assert _covers(vr["ci95"], math.hypot(*v)) and vr["identifiability"] in ("identified",
                                                                             "practically_non_identifiable")
    assert {"vx", "vy"} <= set(rep["hessian"]["parameters"])


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


def test_a_parameter_on_its_fit_bound_gets_a_one_sided_profile_and_no_conflict_test():
    # canopy *warms* in this world, so the data push the shading strength s (≥ 0) onto its lower bound
    grid, frame, y, coords, cl, cfg = albedo_world(noise=0.3, seed=2, canopy_effect=1.0,
                                                   roles={"albedo": "albedo", "canopy": "canopy"})
    rep = ID.analyse(grid, frame, y, coords, cl, cfg, L_init=L_TRUE, params=["L", "s"], draws=500,
                     settings={**FAST, "max_points": 6, "block_m": BLOCK_M})
    s = _row(rep, "s")
    assert s["at_bound"] == {"side": "lower", "bound": 0.0} and s["held"] == "bound"
    assert s["profile"]["sides_explored"] == ["upper"] and s["ci95"][0] == 0.0
    assert s["open_sides"] in ([], ["upper"])
    assert "s" in rep["hessian"]["held_at_bound"]
    assert s["conflict"]["p"] is None and "fit bound" in s["conflict"]["reason"]
    assert s["posterior"]["method"] == "profile" and s["posterior"]["mean"] > 0.0
    assert s["psens"]["method"] == "profile" and s["psens"]["prior"] is not None
    assert s["estimate_run"] >= s["estimate_data"] - 1e-9
    md = ID.identifiability_markdown(rep)
    assert "at lower bound" in md and "Canopy shading strength" in md


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
    assert set(rep["references"]) == {"raue2009", "chandler2007", "ribatet2012", "kallioinen2024", "evans2006"}
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
