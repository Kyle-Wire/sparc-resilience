"""The physics model's optional ``lst`` role (surface–air coupling in the source) and ``assign`` (transfer of
parameters to a domain without labels), PHYSICS_PLAN-style: off by default, recoverable when planted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.core.grid import Grid
from sparc.core.physics import NATURAL_LEAF, PhysicsModel
from sparc.core.synthetic import gaussian_random_field

ROLES = {"canopy": "canopy", "impervious": "impervious", "ndvi": "ndvi", "albedo": "albedo"}


def _world(n=56, dx=30.0, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, n)
    canopy = 50 * (1 + np.tanh(gaussian_random_field(shape, 5.0, rng))) / 2
    imp = 100 * (1 + np.tanh(gaussian_random_field(shape, 7.0, rng) - 0.01 * canopy)) / 2
    ndvi = 0.1 + 0.005 * canopy + 0.02 * gaussian_random_field(shape, 3.0, rng)
    albedo = 0.15 + 0.03 * gaussian_random_field(shape, 4.0, rng)
    hidden = gaussian_random_field(shape, 4.0, rng)                    # a heat source no land-cover layer shows
    lst = 35.0 + 0.08 * imp - 0.05 * canopy + 4.0 * hidden
    iy, ix = np.divmod(np.arange(n * n), n)
    ones = np.ones(shape, dtype=np.int64)
    grid = Grid(x0=0.0, y0=0.0, dx=dx, dy=dx, nx=n, ny=n, ix=ix, iy=iy, mask=ones > 0, counts=ones)
    frame = pd.DataFrame({"canopy": canopy.ravel(), "impervious": imp.ravel(), "ndvi": ndvi.ravel(),
                          "albedo": albedo.ravel(), "lst": lst.ravel()})
    coords = np.column_stack([ix * dx, iy * dx]).astype(float)
    return grid, frame, coords, rng


CFG = {"roles": ROLES, "fit_advection": False, "sw_down": 650.0, "lw_net": -90.0, "max_iter": 80,
       "L_max_m": 600.0}


def test_without_the_role_nothing_about_the_model_changes():
    grid, frame, coords, _ = _world()
    m = PhysicsModel(grid, CFG, L_init=90.0)
    assert "ulst" not in m._init_params((0.0, 0.0)) and "lst" not in m.roles
    P = m._init_params((0.0, 0.0))
    assert "c_lst" not in m._transform(P)
    m2 = PhysicsModel(grid, {**CFG, "roles": {**ROLES, "lst": "lst"}}, L_init=90.0)
    assert "ulst" in m2._init_params((0.0, 0.0)) and NATURAL_LEAF["c_lst"] == "ulst"
    assert m2.natural_bounds("c_lst") == (0.0, None, "floor")
    assert m2.to_natural("c_lst", m2.to_leaf("c_lst", 0.037)) == pytest.approx(0.037)


def _planted(grid, frame, coords, rng, c_lst=0.03, a=5.0, noise=0.05):
    truth = {"L": 150.0, "s": 0.4, "kc": 2.0, "a1": 0.3, "e0": -0.6, "e1": 2.0, "e2": 1.0, "e3": 1.0, "c_lst": c_lst}
    gen = PhysicsModel(grid, {**CFG, "roles": {**ROLES, "lst": "lst"}}, L_init=150.0)
    gen.assign(frame, coords, truth, a=a)
    y = gen.predict(frame, coords)
    return y + noise * y.std() * rng.standard_normal(y.size), truth


def test_a_planted_surface_air_coupling_is_recovered_and_beats_land_cover_alone():
    grid, frame, coords, rng = _world()
    y, truth = _planted(grid, frame, coords, rng)
    n = y.size
    train = np.zeros(n, bool)
    train[rng.choice(n, n // 3, replace=False)] = True
    test = ~train
    with_lst = PhysicsModel(grid, {**CFG, "roles": {**ROLES, "lst": "lst"}}, L_init=90.0).fit(frame, y, train, coords)
    without = PhysicsModel(grid, CFG, L_init=90.0).fit(frame, y, train, coords)
    p1, p0 = with_lst.predict(frame, coords), without.predict(frame, coords)

    def r2(p):
        return 1 - np.mean((p[test] - y[test]) ** 2) / np.var(y[test])

    assert r2(p1) > 0.9 and r2(p1) > r2(p0) + 0.2
    pr = with_lst.params
    assert pr["c_lst"] * pr["a"] == pytest.approx(truth["c_lst"] * 5.0, rel=0.2)
    assert pr["L_m"] == pytest.approx(150.0, rel=0.2)
    assert "c_lst" not in without.params


def test_an_uninformative_lst_gets_a_small_coupling():
    grid, frame, coords, rng = _world(seed=3)
    frame = frame.assign(lst=rng.normal(35.0, 4.0, len(frame)))      # LST unrelated to the air
    y, _ = _planted(grid, frame.assign(lst=35.0), coords, rng)       # truth without any LST signal
    m = PhysicsModel(grid, {**CFG, "roles": {**ROLES, "lst": "lst"}}, L_init=90.0).fit(frame, y, np.arange(len(y)), coords)
    q_land = m.source_raster(frame.assign(lst=35.0)).std()
    assert m.params["c_lst"] * 4.0 < 0.1 * q_land                     # the LST term is < 10% of the source spread


def test_assign_reproduces_a_fitted_model_and_transfers_without_labels():
    grid, frame, coords, rng = _world(seed=1)
    y, _ = _planted(grid, frame, coords, rng)
    cfg = {**CFG, "roles": {**ROLES, "lst": "lst"}}
    fit = PhysicsModel(grid, cfg, L_init=90.0).fit(frame, y, np.arange(len(y)), coords)
    nat = {k: v for k, v in fit._T.items() if k in NATURAL_LEAF}
    pr = fit.params
    copy = PhysicsModel(grid, cfg, L_init=90.0).assign(frame, coords, nat, a=pr["a"], b=pr["b"], gamma=pr["gamma"],
                                                        beta=(pr["beta_x"], pr["beta_y"]))
    assert np.allclose(copy.predict(frame, coords), fit.predict(frame, coords), atol=1e-9)
    assert copy.fit_info["assigned"] and np.isnan(copy.train_rmse)
    # transfer to another city: its own features set every constant; no labels are passed at all
    g2, f2, c2, _ = _world(seed=7)
    other = PhysicsModel(g2, cfg, L_init=90.0).assign(f2, c2, nat, a=pr["a"])
    p = other.predict(f2, c2)
    assert np.isfinite(p).all() and p.std() > 0
    with pytest.raises(ValueError, match="unknown"):
        PhysicsModel(g2, cfg).assign(f2, c2, {"nope": 1.0}, a=1.0)
