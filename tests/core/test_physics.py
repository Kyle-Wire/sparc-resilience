"""Physics base model (roadmap §2.1) and the known-source operator fit."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from sparc.core import operators as ops
from sparc.core.cv import make_spatial_folds
from sparc.core.physics import PhysicsModel, _SpectralSolver, fit_operator
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


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


def test_cached_solver_is_operators_solve_torch():
    rng = np.random.default_rng(0)
    q = torch.as_tensor(rng.standard_normal((30, 26)))
    pad = ops.pad_cells(200.0, 150.0, 30.0)
    solver = _SpectralSolver(30, 26, 30.0, pad)
    L, vx, vy = torch.tensor(120.0, dtype=torch.float64), torch.tensor(40.0, dtype=torch.float64), torch.tensor(-25.0, dtype=torch.float64)
    ref = ops.solve_torch(q, L, vx, vy, 30.0, pad=pad)
    assert torch.max(torch.abs(solver(q, L, vx, vy) - ref)) < 1e-12


def test_fit_operator_recovers_planted_L_v_a():
    fx = make_operator_fixture()
    t = fx["truth"]
    res = fit_operator(fx["q"], fx["y"], np.ones_like(fx["y"], dtype=bool), t["dx"], L_init=3 * t["dx"], v_max=1000.0)
    assert res["L_m"] == pytest.approx(t["L"], rel=0.25)
    v_err = np.hypot(res["vx_m"] - t["v"][0], res["vy_m"] - t["v"][1])
    assert v_err <= 0.25 * np.hypot(*t["v"])
    assert res["a"] == pytest.approx(t["a"], rel=0.20)
    assert res["b"] == pytest.approx(t["b"], abs=0.1)


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
