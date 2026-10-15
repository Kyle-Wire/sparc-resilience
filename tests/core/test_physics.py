"""Physics base model (roadmap §2.1) and the known-source operator fit."""

from __future__ import annotations

import copy
import logging
import math
import pickle
import time

import numpy as np
import pytest
import torch

from sparc.core import operators as ops
from sparc.core.cv import make_spatial_folds
from sparc.core.physics import (PhysicsModel, _SpectralSolver, fit_operator, grid_resolution_report,
                                report_grid_resolution)
from sparc.core.synthetic import make_operator_fixture, synthetic_city_config


def _r2(pred, obs):
    return 1.0 - np.mean((pred - obs) ** 2) / np.var(obs)


@pytest.fixture(scope="module")
def city_fold(synthetic_core_data):
    _, data = synthetic_core_data
    folds = make_spatial_folds(data.coords, n_folds=3, block_m=300.0, seed=42)
    train, test = next(folds.split())
    return data, train, test


@pytest.fixture(scope="module")
def city_model(city_fold):
    data, train, _ = city_fold
    cfg = synthetic_city_config()["physics"]
    return PhysicsModel(data.grid, cfg, L_init=3 * data.grid.dx).fit(data.frame, data.y, train, data.coords)


@pytest.fixture(scope="module")
def scheme_models(city_fold):
    """A short fit (advection on, as configured) of the synthetic city per
    ``physics.advection_scheme``, so the configured scheme is checked all the
    way through ``PhysicsModel`` to its solver and residual."""
    data, train, _ = city_fold
    out = {}
    for scheme in ops.SCHEMES:
        cfg = {**synthetic_city_config()["physics"], "advection_scheme": scheme, "max_iter": 10}
        out[scheme] = PhysicsModel(data.grid, cfg, L_init=3 * data.grid.dx).fit(data.frame, data.y, train,
                                                                              data.coords)
    return out


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ops.SCHEMES)
def test_cached_solver_is_operators_solve_torch(scheme):
    rng = np.random.default_rng(0)
    q = torch.as_tensor(rng.standard_normal((30, 26)))
    pad = ops.pad_cells(200.0, 150.0, 30.0)
    solver = _SpectralSolver(30, 26, 30.0, pad, scheme)
    L, vx, vy = torch.tensor(120.0, dtype=torch.float64), torch.tensor(40.0, dtype=torch.float64), torch.tensor(-25.0, dtype=torch.float64)
    ref = ops.solve_torch(q, L, vx, vy, 30.0, pad=pad, scheme=scheme)
    assert torch.max(torch.abs(solver(q, L, vx, vy) - ref)) < 1e-12
    # strong wind, short L: the schemes differ, and the solver follows its own
    L, vx = torch.tensor(30.0, dtype=torch.float64), torch.tensor(600.0, dtype=torch.float64)
    ref = ops.solve_torch(q, L, vx, vy, 30.0, pad=pad, scheme=scheme)
    assert torch.max(torch.abs(solver(q, L, vx, vy) - ref)) < 1e-12
    other = ops.solve_torch(q, L, vx, vy, 30.0, pad=pad, scheme="central" if scheme != "central" else "hybrid")
    assert torch.max(torch.abs(other - ref)) > 1e-3 * torch.max(torch.abs(ref))


def test_fit_operator_recovers_planted_L_v_a():
    fx = make_operator_fixture()
    t = fx["truth"]
    res = fit_operator(fx["q"], fx["y"], np.ones_like(fx["y"], dtype=bool), t["dx"], L_init=3 * t["dx"], v_max=1000.0)
    assert res["L_m"] == pytest.approx(t["L"], rel=0.25)
    v_err = np.hypot(res["vx_m"] - t["v"][0], res["vy_m"] - t["v"][1])
    assert v_err <= 0.25 * np.hypot(*t["v"])
    assert res["a"] == pytest.approx(t["a"], rel=0.20)
    assert res["b"] == pytest.approx(t["b"], abs=0.1)
    # cell Péclet ≈ 0.06: the grid resolves L, and the result says so
    assert res["L_resolved"] is True and res["resolution_warning"] is None
    assert max(res["cell_peclet_x"], res["cell_peclet_y"]) < 0.2 and res["L_eff_x_m"] >= res["L_m"]


def test_fit_operator_flags_an_L_the_grid_cannot_resolve(caplog):
    """Data with D = L² at L = 40 m under an oblique 1 km wind on 30 m cells
    (planted with central differences, i.e. less diffusion than any positive
    3-point operator can have there: both cell Péclet numbers are 6.6).  The
    central warm start finds L ≈ 40 m; the hybrid operator then has the
    grid's diffusion |v_axis|·dx/2 on both axes whatever L is, and the result
    says L is not identified instead of passing L_m off as the relaxation length."""
    fx = make_operator_fixture(n=48, L=40.0, v=(707.0, 707.0), noise=0.0, scheme="central")
    with caplog.at_level(logging.WARNING, logger="sparc.core.physics"):
        res = fit_operator(fx["q"], fx["y"], np.ones_like(fx["y"], dtype=bool), 30.0, L_init=30.0, v_max=1000.0,
                           v_penalty=0.0, max_iter=30)
    assert res["advection_scheme"] == "hybrid"
    assert res["L_resolved"] is False and min(res["cell_peclet_x"], res["cell_peclet_y"]) > 1.0
    assert (res["L_elasticity_x"], res["L_elasticity_y"]) == (0.0, 0.0)
    # L keeps the warm start's value (the hybrid fit cannot move it), below the Pe = 1 bound at the fitted wind
    assert res["L_m"] == pytest.approx(40.0, rel=0.1)
    v_axis = max(abs(res["vx_m"]), abs(res["vy_m"]))
    assert res["L_m"] < res["L_min_resolved_m"] == pytest.approx(math.sqrt(v_axis * 15.0))
    assert "L is not identified" in res["resolution_warning"] and res["resolution_warning"] in caplog.text


@pytest.mark.parametrize("scheme", ops.SCHEMES)
def test_fit_operator_fits_on_its_own_scheme(scheme, monkeypatch):
    """The fit's solver is built with ``scheme`` (the central warm-start solver
    besides, for a positive scheme with advection) and the result says so."""
    from sparc.core import physics

    built = []
    orig = physics._SpectralSolver.__init__

    def spy(self, ny, nx, dx, pad, scheme=ops.DEFAULT_SCHEME):
        built.append(scheme)
        orig(self, ny, nx, dx, pad, scheme)

    monkeypatch.setattr(physics._SpectralSolver, "__init__", spy)
    fx = make_operator_fixture(n=32, L=60.0, v=(300.0, 0.0), noise=0.0)
    res = fit_operator(fx["q"], fx["y"], np.ones_like(fx["y"], dtype=bool), 30.0, L_init=60.0, v_max=1000.0,
                       max_iter=3)
    assert res["advection_scheme"] == ops.DEFAULT_SCHEME and built[0] == ops.DEFAULT_SCHEME
    built.clear()
    res = fit_operator(fx["q"], fx["y"], np.ones_like(fx["y"], dtype=bool), 30.0, L_init=60.0, v_max=1000.0,
                       max_iter=3, scheme=scheme)
    assert res["advection_scheme"] == scheme and built[0] == scheme
    assert set(built) == ({scheme} if scheme == "central" else {scheme, "central"})


def test_grid_resolution_report_names_what_the_grid_cannot_resolve():
    for scheme in ops.SCHEMES:                                                     # Providence-like: Pe 0.07
        info, msg = grid_resolution_report(256.0, 300.0, 0.0, 30.0, scheme)
        assert info["L_resolved"] is True and msg is None
    # hybrid: L drops out of an axis past Pe = 1, and out of the operator when both axes are past it
    info, msg = grid_resolution_report(30.0, 1000.0, 0.0, 30.0, "hybrid")           # along x only
    assert info["L_resolved"] is False and "along x is the grid's upwind value" in msg
    assert "(√D_x = 122 m)" in msg and "spread along y only" in msg
    _, msg = grid_resolution_report(30.0, 707.1, 707.1, 30.0, "hybrid")              # oblique: both axes
    assert "on both axes" in msg and "L is not identified" in msg and "every L below 103 m" in msg
    assert "for L ≥ 103 m at this wind" in msg
    # exponential: graded by the elasticity 4Pe/sinh(2Pe) of its diffusion
    _, msg = grid_resolution_report(94.0, 707.1, 707.1, 30.0, "exponential")         # Pe 1.2 on both axes
    assert "only weakly identified" in msg and "d ln D/d ln L = 0.88 along x, 0.88 along y" in msg
    _, msg = grid_resolution_report(30.0, 707.1, 707.1, 30.0, "exponential")         # Pe 11.8
    assert "effectively not identified" in msg
    _, msg = grid_resolution_report(30.0, 1000.0, 0.0, 30.0, "exponential")
    assert "along x depends on L only weakly" in msg and "spread along y" in msg
    _, msg = grid_resolution_report(30.0, 1000.0, 0.0, 30.0, "central")
    assert "negative lobes" in msg


# ---------------------------------------------------------------------------
# Physics model on the synthetic city
# ---------------------------------------------------------------------------


def test_physics_model_generalises_to_held_out_blocks(city_fold, city_model, synthetic_city):
    data, train, test = city_fold
    pred = city_model.predict(data.frame, data.coords)
    assert pred.shape == (data.n,)
    assert _r2(pred[test], data.y[test]) > 0.4
    p = city_model.params
    L_true = synthetic_city.truth["L"]
    assert L_true / 2 <= p["L_m"] <= 2 * L_true
    assert p["a"] > 0 and p["fit_warning"] is None
    assert p["L_resolved"] is True and p["resolution_warning"] is None and p["cell_peclet_x"] < 1.0
    assert p["L_eff_x_m"] >= p["L_m"] and p["L_eff_y_m"] >= p["L_m"]
    for key in ("L_m", "vx_m", "vy_m", "v_norm_m", "a", "b", "gamma", "beta_x", "beta_y",
                "s", "a1", "e0", "e1", "e2", "e3", "w", "train_rmse"):
        assert isinstance(p[key], float) and np.isfinite(p[key]), key
    # Penalised source coefficients stay near their literature values.
    assert abs(p["s"] - 0.6) < 0.2 and abs(p["a1"] - 0.3) < 0.2


def test_operator_residual_vanishes_on_own_prediction(city_model, city_fold):
    data, _, _ = city_fold
    pred = city_model.predict(data.frame, data.coords)
    dT, valid = data.grid.rasterize_torch(torch.as_tensor(pred))
    R = city_model.operator_residual_torch(dT, valid)
    assert R.ndim == 1 and R.numel() > 0.5 * data.grid.mask.sum()
    q_scale = np.abs(city_model.source_raster(data.frame)).max()
    assert R.abs().max().item() < 1e-6 * q_scale
    # ... and is differentiable in the ΔT raster.
    dT = dT.clone().requires_grad_(True)
    city_model.operator_residual_torch(dT, valid).pow(2).sum().backward()
    assert dT.grad is not None and torch.isfinite(dT.grad).all()


def test_advection_scheme_is_configured_validated_and_reported(city_model, city_fold):
    from sparc.core.config import DEFAULTS, core_config_from_dict

    data, _, _ = city_fold
    assert city_model.advection_scheme == ops.DEFAULT_SCHEME == "hybrid"
    assert city_model.params["advection_scheme"] == "hybrid" and city_model._solver.scheme == "hybrid"
    assert DEFAULTS["physics"]["advection_scheme"] == ops.DEFAULT_SCHEME
    assert PhysicsModel(data.grid, {"advection_scheme": "Central"}).advection_scheme == "central"
    with pytest.raises(ValueError, match="advection_scheme"):
        PhysicsModel(data.grid, {"advection_scheme": "upwind"})
    raw = synthetic_city_config()
    raw["physics"]["advection_scheme"] = "upwind"
    with pytest.raises(ValueError, match="advection_scheme"):
        core_config_from_dict(raw)


@pytest.mark.parametrize("scheme", ops.SCHEMES)
def test_configured_scheme_reaches_the_fitted_models_solver_and_residual(scheme, scheme_models, city_fold):
    """``physics.advection_scheme`` must reach the solver the fit builds (and
    predicts with), the reported params and the PDE residual, which then
    vanishes on the model's own prediction."""
    data, _, _ = city_fold
    m = scheme_models[scheme]
    assert m.advection_scheme == scheme and m._solver.scheme == scheme and m.params["advection_scheme"] == scheme
    pred = m.predict(data.frame, data.coords)
    dT, valid = data.grid.rasterize_torch(torch.as_tensor(pred))
    R = m.operator_residual_torch(dT, valid)
    assert R.abs().max().item() < 1e-6 * np.abs(m.source_raster(data.frame)).max()
    # its solver is the scheme's operator (operators.solve with the same padding)
    p, q = m.params, m.source_raster(data.frame)
    want = ops.solve(q, p["L_m"], (p["vx_m"], p["vy_m"]), dx=data.grid.dx, pad=m.pad, scheme=scheme)
    got = m._solver(torch.as_tensor(q), *(torch.tensor(p[k], dtype=torch.float64) for k in ("L_m", "vx_m", "vy_m")))
    assert np.max(np.abs(got.numpy() - want)) < 1e-12 * np.max(np.abs(want))


def test_hybrid_fit_is_the_central_fit_below_cell_peclet_one(scheme_models, city_fold):
    """The synthetic city's fit sits at cell Péclet ≈ 0.04: the hybrid fit
    never leaves the central warm start and equals the central fit bit for
    bit (so runs like Providence do not move), while the exponential fit
    continues on its own operator."""
    data, _, _ = city_fold
    h, c, e = (scheme_models[s] for s in ("hybrid", "central", "exponential"))
    assert h.fit_info["continued_on_scheme"] is False and e.fit_info["continued_on_scheme"] is True
    keys = ("L_m", "vx_m", "vy_m", "a", "b", "gamma", "s", "a1", "train_rmse")
    assert [h.params[k] for k in keys] == [c.params[k] for k in keys]
    assert np.array_equal(h.predict(data.frame, data.coords), c.predict(data.frame, data.coords))


def test_pre_scheme_pickle_keeps_the_central_operator(city_model, city_fold):
    """Checkpoints written before ``advection_scheme`` existed hold a model
    without the attribute and a solver with only lam/sx/sy.  They were fitted
    with central differences, so they must predict exactly as the central
    scheme does, and their PDE residual must use the same operator."""
    data, _, _ = city_fold
    g = data.grid
    m = copy.deepcopy(city_model)
    m._T.update(L=45.0, vx=400.0, vy=-250.0)          # cell Péclet > 1: the schemes differ visibly
    m.advection_scheme = "central"
    m._solver = _SpectralSolver(g.ny, g.nx, m.dx, m.pad, "central")
    ref = m.predict(data.frame, data.coords)
    new = {}
    for scheme in ("hybrid", "exponential"):
        new[scheme] = copy.deepcopy(m)
        new[scheme].advection_scheme = scheme
        new[scheme]._solver = _SpectralSolver(g.ny, g.nx, m.dx, m.pad, scheme)
        assert np.max(np.abs(new[scheme].predict(data.frame, data.coords) - ref)) > 1e-3 * np.std(ref)

    old = copy.deepcopy(m)
    del old.advection_scheme
    old.cfg.pop("advection_scheme", None)
    old_solver = _SpectralSolver.__new__(_SpectralSolver)
    old_solver.__dict__.update({k: m._solver.__dict__[k]
                                for k in ("ny", "nx", "dx", "pad", "py", "px", "lam", "sx", "sy")})
    old._solver = old_solver
    loaded = pickle.loads(pickle.dumps(old))
    assert not hasattr(loaded, "advection_scheme") and loaded._solver.scheme == "central"
    assert np.array_equal(loaded.predict(data.frame, data.coords), ref)
    assert loaded.params["advection_scheme"] == "central"
    # an old model reports its grid resolution too (here: central with Pe > 1, i.e. negative lobes)
    assert loaded.params["L_resolved"] is False and "negative lobes" in loaded.params["resolution_warning"]
    assert "L is not identified" in new["hybrid"].params["resolution_warning"]
    assert "L is only weakly identified" in new["exponential"].params["resolution_warning"]
    dT, valid = g.rasterize_torch(torch.as_tensor(ref))
    R = loaded.operator_residual_torch(dT, valid)
    assert R.abs().max().item() < 1e-6 * np.abs(loaded.source_raster(data.frame)).max()
    # a solver that lost its cached symbols rebuilds them for its scheme
    for name in ("lam", "sx", "sy"):
        delattr(loaded._solver, name)
    assert np.array_equal(loaded.predict(data.frame, data.coords), ref)


def test_grid_peclet_warning_is_run_level_and_only_for_delivered_models(city_fold, monkeypatch):
    """A fold fit that lands past Pe = 1 records ``resolution_warning`` in its
    params but emits no event (advection selection may still replace it);
    ``report_grid_resolution`` emits one ``physics.grid_peclet`` for the
    delivered models, with the per-fold numbers.  The synthetic city's fit
    sits at Pe ≈ 0.04, so the grid report is stubbed to say otherwise."""
    from sparc.core import progress

    data, train, _ = city_fold
    unresolved = {"cell_peclet_x": 3.0, "cell_peclet_y": 2.0, "L_eff_x_m": 104.0, "L_eff_y_m": 90.0,
                  "L_elasticity_x": 0.0, "L_elasticity_y": 0.0, "L_min_resolved_m": 104.0, "L_resolved": False}
    seen = []
    monkeypatch.setattr(progress, "warn", lambda code, message, **kw: seen.append((code, message, kw)))
    cfg = {**synthetic_city_config()["physics"], "fit_advection": False, "max_iter": 5}
    resolved = PhysicsModel(data.grid, cfg).fit(data.frame, data.y, train, data.coords).params
    with monkeypatch.context() as mp:
        mp.setattr(ops, "grid_resolution", lambda *a, **k: dict(unresolved))
        m = PhysicsModel(data.grid, cfg).fit(data.frame, data.y, train, data.coords)
        assert m.a > 0 and not [c for c, _, _ in seen if c == "physics.grid_peclet"]
        p = m.params
    assert p["L_resolved"] is False and "L is not identified" in p["resolution_warning"]

    assert report_grid_resolution([resolved, None, resolved]) is None and not seen
    data_ = report_grid_resolution([resolved, p, None, p])
    hits = [(msg, kw) for code, msg, kw in seen if code == "physics.grid_peclet"]
    assert len(hits) == 1
    msg, kw = hits[0]
    assert kw == data_ and kw["folds"] == [1, 3] and kw["n_folds"] == 4 and kw["cell_peclet"] == [3.0, 3.0]
    assert kw["L_min_resolved_m"] == [104.0, 104.0] and kw["scheme"] == "hybrid"
    assert msg.startswith("2 of 4 fold physics fits") and p["resolution_warning"] in msg


def test_source_points_centred_and_rasterised_equals_source_raster(city_model, city_fold):
    data, _, _ = city_fold
    g = data.grid
    q_pts = city_model.source_points(data.frame)
    assert q_pts.shape == (data.n,)
    q_cells = g.rasterize(q_pts)
    expected = q_cells - np.nanmean(q_cells[g.mask])
    got = city_model.source_raster(data.frame)
    assert np.allclose(got[g.mask], expected[g.mask], atol=1e-12)
    assert np.all(got[~g.mask] == 0.0)
    # Pointwise: editing one row changes only that row's q_raw.
    edited = data.frame.copy()
    edited.loc[5, "canopy"] += 20.0
    dq = city_model.source_points(edited) - q_pts
    assert dq[5] != 0 and np.count_nonzero(dq) == 1


def test_uniform_canopy_increase_cools_everywhere(city_model, city_fold):
    """The centring constant is frozen at fit time, so a uniform edit is not
    absorbed by re-centring."""
    data, _, _ = city_fold
    base = city_model.predict(data.frame, data.coords)
    greener = data.frame.copy()
    greener["canopy"] = np.clip(greener["canopy"] + 20.0, 0, 100)
    delta = city_model.predict(greener, data.coords) - base
    assert delta.mean() < 0 and np.mean(delta < 0) > 0.95


def test_held_out_labels_do_not_influence_fit(city_fold):
    data, train, test = city_fold
    cfg = {**synthetic_city_config()["physics"], "fit_advection": False, "max_iter": 15}
    m1 = PhysicsModel(data.grid, cfg).fit(data.frame, data.y, train, data.coords)
    y2 = data.y.copy()
    y2[test] = np.random.default_rng(0).standard_normal(test.size) * 100.0
    m2 = PhysicsModel(data.grid, cfg).fit(data.frame, y2, train, data.coords)
    assert np.allclose(m1.predict(data.frame, data.coords), m2.predict(data.frame, data.coords), atol=1e-10)
    assert m1.params["vx_m"] == 0.0 and m1.params["vy_m"] == 0.0


def test_night_window_and_missing_roles(city_fold):
    data, train, test = city_fold
    roles = dict(synthetic_city_config()["physics"]["roles"])
    roles.pop("albedo")
    roles["ndvi"] = "not_a_column"
    cfg = {"window": "night", "roles": roles, "fit_advection": False, "max_iter": 10}
    m = PhysicsModel(data.grid, cfg).fit(data.frame, data.y, train, data.coords)
    assert "albedo" not in m.params["roles_used"] and "ndvi" not in m.params["roles_used"]
    assert np.isfinite(m.predict(data.frame, data.coords)).all()
    assert m.params["window"] == "night"


# ---------------------------------------------------------------------------
# Real data (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_physics_on_brown_8k_window(brown_csv, providence_config_path):
    from sparc.core.config import load_core_config
    from sparc.core.data import load_core_data

    cfg = load_core_config(providence_config_path)
    cfg.raw["data"]["subsample"] = 8000
    data = load_core_data(cfg)
    folds = make_spatial_folds(data.coords, n_folds=5, block_m=600.0, seed=42)
    train, test = next(folds.split())
    t0 = time.perf_counter()
    model = PhysicsModel(data.grid, cfg.raw["physics"]).fit(data.frame, data.y, train, data.coords)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0
    assert model.params["a"] > 0
    pred = model.predict(data.frame, data.coords)
    assert np.isfinite(pred).all()
