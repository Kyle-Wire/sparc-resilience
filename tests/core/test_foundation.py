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
