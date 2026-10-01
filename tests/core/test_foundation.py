"""Operators, grid and data loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.core import operators as ops
from sparc.core.grid import Grid, spatial_window_subsample


def test_spectral_solve_is_exact_inverse_of_central_operator():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((48, 64))
    L, v = 150.0, (90.0, -45.0)
    phi = ops.solve(q, L, v, dx=30.0)
    resid = ops.apply_operator(phi, L, v, dx=30.0) - q
    assert np.nanmax(np.abs(resid[1:-1, 1:-1])) < 1e-10


def test_solve_torch_matches_numpy_and_is_differentiable():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    q = rng.standard_normal((32, 40))
    pad = ops.pad_cells(120.0, np.hypot(60.0, 30.0), 30.0)
    ref = ops.solve(q, 120.0, (60.0, 30.0), dx=30.0, pad=pad)
    L = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
    out = ops.solve_torch(torch.tensor(q), L, torch.tensor(60.0, dtype=torch.float64),
                          torch.tensor(30.0, dtype=torch.float64), 30.0, pad=pad)
    assert np.max(np.abs(out.detach().numpy() - ref)) < 1e-9
    out.pow(2).sum().backward()
    assert L.grad is not None and torch.isfinite(L.grad)


def test_apply_operator_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((20, 24))
    ref = ops.apply_operator(phi, 90.0, (30.0, -10.0), dx=30.0)[1:-1, 1:-1]
    got = ops.apply_operator_torch(torch.tensor(phi), 90.0, 30.0, -10.0, 30.0).numpy()
    assert np.allclose(ref, got)


def test_masked_gaussian_preserves_constants_and_excludes_self():
    mask = np.ones((40, 40), dtype=bool)
    mask[:, :5] = False
    f = np.full((40, 40), 3.0)
    out = ops.masked_gaussian(f, mask, 2.0)
    assert np.allclose(out[mask], 3.0)
    spike = np.zeros((40, 40))
    spike[20, 20] = 1.0
    excl = ops.masked_gaussian(spike, np.ones_like(mask), 2.0, exclude_self=True)
    assert excl[20, 20] == pytest.approx(0.0, abs=1e-9)
    assert excl[20, 21] > 0


def test_grid_recovers_lattice_and_round_trips():
    rng = np.random.default_rng(4)
    iy, ix = np.nonzero(rng.random((30, 50)) < 0.6)
    x = 1234.5 + ix * 30.0
    y = 987.25 + iy * 30.0
    g = Grid.from_points(x, y)
    assert g.dx == pytest.approx(30.0, rel=1e-6)
    assert g.collision_fraction() == 0.0
    vals = rng.standard_normal(len(x))
    assert np.allclose(g.sample(g.rasterize(vals)), vals)


def test_spatial_window_subsample_is_contiguous():
    rng = np.random.default_rng(5)
    x, y = rng.random(5000), rng.random(5000)
    idx = spatial_window_subsample(x, y, 1000)
    assert 1000 <= len(idx) <= 1100
    assert x[idx].max() - x[idx].min() < 0.6


def test_prepare_frame_synthetic(synthetic_core_data, synthetic_city):
    cfg, data = synthetic_core_data
    assert data.n == len(synthetic_city.frame)
    assert data.grid.dx == pytest.approx(30.0)
    assert abs(np.median(data.y)) < 1e-9          # background = median removed
    assert list(data.X.columns) == cfg.predictors


def test_encodings_one_hot_and_circular():
    from sparc.core.data import encode_features

    frame = pd.DataFrame({"lc": [11, 21, 21], "aspect": [0.0, 90.0, 180.0], "z": [1.0, 2.0, 3.0]})
    X = encode_features(frame, {"categorical": ["lc"], "circular_degrees": ["aspect"]})
    assert {"lc__11", "lc__21", "aspect__sin", "aspect__cos", "z"} == set(X.columns)
    assert np.allclose(X["aspect__cos"], [1.0, 0.0, -1.0], atol=1e-12)


def test_brown_loads_on_its_30m_lattice(brown_csv, providence_config_path):
    from sparc.core.config import load_core_config
    from sparc.core.data import load_core_data

    cfg = load_core_config(providence_config_path)
    data = load_core_data(cfg)
    assert data.n == 54701
    assert data.grid.dx == pytest.approx(30.0, abs=0.05)
    assert data.qa["cell_collisions"] < 0.001
    assert data.qa["clipped"].get("Albedo", 0) >= 1          # albedo > 0.9 exists in the file
    assert data.target_raw.min() > 70 and data.target_raw.max() < 100   # °F, not a z-score


def test_coarse_mode_averages_full_extent_onto_coarser_cells(synthetic_city):
    from sparc.core.config import core_config_from_dict
    from sparc.core.data import prepare_frame
    from sparc.core.synthetic import synthetic_city_config

    raw = synthetic_city_config()
    fine = prepare_frame(synthetic_city.frame, core_config_from_dict(raw))
    raw["data"]["coarse_m"] = 60.0
    df = synthetic_city.frame.assign(zone=np.where(synthetic_city.frame["x"] < synthetic_city.frame["x"].median(), 1, 2))
    raw["data"]["zone"] = "zone"
    coarse = prepare_frame(df, core_config_from_dict(raw))
    assert coarse.grid.dx == pytest.approx(60.0)
    assert coarse.n == pytest.approx(fine.n / 4, rel=0.05)
    assert coarse.qa["coarse"]["n_fine"] == fine.n
    # same extent (cell centres sit half a coarse cell inside the fine envelope)
    assert np.ptp(coarse.x) == pytest.approx(np.ptp(fine.x), abs=60.0)
    # cell means ≈ preserve the area mean (exactly, up to partial edge cells)
    assert coarse.target_raw.mean() == pytest.approx(fine.target_raw.mean(), abs=0.05)
    assert coarse.frame["canopy"].mean() == pytest.approx(fine.frame["canopy"].mean(), abs=0.5)
    assert set(np.unique(coarse.zones)) == {1, 2}
    assert coarse.qa["grid_fill_fraction"] >= fine.qa["grid_fill_fraction"] - 0.02
    assert coarse.qa["cell_collisions"] == 0.0


def test_qa_flags_classed_target_albedo_scale_and_dose_context():
    from sparc.core.config import core_config_from_dict
    from sparc.core.data import prepare_frame

    rng = np.random.default_rng(0)
    xx, yy = np.meshgrid(np.arange(40) * 30.0, np.arange(40) * 30.0)
    n = xx.size
    t = 85.0 + rng.normal(0, 2, n)
    t[: int(0.7 * n)] = np.round(t[: int(0.7 * n)])
    df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "T": t, "alb": rng.normal(0.40, 0.05, n),
                       "can": rng.uniform(0, 60, n), "imp": rng.uniform(30, 90, n)})
    raw = {"name": "qa", "data": {"target": "T", "x": "x", "y": "y"}, "predictors": ["alb", "can", "imp"],
           "actionable": {"alb": {"min": 0, "max": 1, "doses": [0, 0.05, 0.3]}},
           "physics": {"roles": {"albedo": "alb", "canopy": "can", "impervious": "imp"}}}
    data = prepare_frame(df, core_config_from_dict(raw))
    codes = {f["code"] for f in data.qa["flags"]}
    assert {"classed_target", "albedo_scale", "cover_overlap", "dose_scale_alb"} <= codes
    assert data.qa["target_fraction_integer_valued"] == pytest.approx(0.7, abs=0.02)
    hist = data.qa["target_fractional_histogram"]
    assert hist[0] > 0.7 and sum(hist) == pytest.approx(1.0)
    ds = data.qa["dose_scale"]["alb"]
    assert ds["doses_in_sd"][-1] == pytest.approx(0.3 / 0.05, rel=0.1)
    assert 50 < ds["median_cell_to_percentile"][0] < 95 and ds["median_cell_to_percentile"][-1] > 99


def test_physics_albedo_map_rescales_to_broadband():
    from sparc.core.grid import Grid
    from sparc.core.physics import PhysicsModel

    xx, yy = np.meshgrid(np.arange(20) * 30.0, np.arange(20) * 30.0)
    g = Grid.from_points(xx.ravel(), yy.ravel(), cell=30.0)
    a = np.linspace(0.3, 0.5, xx.size)
    pm = PhysicsModel(g, {"roles": {"albedo": "alb"}, "albedo_map": {"from": "auto", "to": [0.1, 0.2]}})
    v = pm._raw_features(pd.DataFrame({"alb": a}))["albedo"]
    lo, hi = np.percentile(a, [2, 98])
    assert np.interp(lo, a, v) == pytest.approx(0.1) and np.interp(hi, a, v) == pytest.approx(0.2)
    pm2 = PhysicsModel(g, {"roles": {"albedo": "alb"}, "albedo_map": {"from": [0.3, 0.5], "to": [0.1, 0.2]}})
    assert pm2._raw_features(pd.DataFrame({"alb": a}))["albedo"][[0, -1]] == pytest.approx([0.1, 0.2])


def test_brown_coarse_mode_keeps_extent_and_flags_product(brown_csv, providence_config_path):
    from sparc.core.config import load_core_config
    from sparc.core.data import load_core_data

    cfg = load_core_config(providence_config_path)
    cfg.raw["data"]["coarse_m"] = 60.0
    data = load_core_data(cfg)
    assert 13000 < data.n < 14500 and data.grid.dx == pytest.approx(60.0)
    codes = {f["code"] for f in data.qa["flags"]}
    assert {"classed_target", "albedo_scale", "coarse"} <= codes
    assert data.qa["target_fraction_integer_valued"] > 0.6          # source product, measured before averaging
    assert set(np.unique(data.zones)) >= {0, 1, 2, 3, 4}
