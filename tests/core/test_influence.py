"""S1 influence: correlograms, anisotropy, ring ranges and focal features."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from sparc.core import influence as inf
from sparc.core.grid import Grid

DX = 30.0


def _anisotropic_field(n: int, theta_deg: float, sigma_major: float, sigma_minor: float, seed: int) -> np.ndarray:
    """White noise filtered with an elliptical Gaussian (major axis at θ,
    counter-clockwise from +x; axis 1 of the array is x)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n))
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    th = np.deg2rad(theta_deg)
    k_major = kx * np.cos(th) + ky * np.sin(th)
    k_minor = -kx * np.sin(th) + ky * np.cos(th)
    H = np.exp(-2.0 * np.pi**2 * (sigma_major**2 * k_major**2 + sigma_minor**2 * k_minor**2))
    z = np.real(np.fft.ifft2(np.fft.fft2(w) * H))
    return (z - z.mean()) / z.std()


# ---------------------------------------------------------------------------
# Correlogram engine
# ---------------------------------------------------------------------------


def test_fft_pair_sums_match_brute_force_on_masked_raster():
    rng = np.random.default_rng(0)
    ny, nx = 9, 11
    z = rng.standard_normal((ny, nx))
    mask = rng.random((ny, nx)) < 0.7
    zc = np.where(mask, z, 0.0)
    eng = inf._MaskedCorrelogram(mask, DX, 5 * DX, 5, 4, full_plane=False)
    got = eng._xcorr(zc)
    for k, (hy, hx) in enumerate(zip(eng.geo.hy, eng.geo.hx)):
        s = n = 0.0
        for iy in range(ny):
            for ix in range(nx):
                jy, jx = iy + hy, ix + hx
                if 0 <= jy < ny and 0 <= jx < nx and mask[iy, ix] and mask[jy, jx]:
                    s += z[iy, ix] * z[jy, jx]
                    n += 1
        assert got[k] == pytest.approx(s, abs=1e-9)
        assert eng.pairs[k] == n
    # half-plane: no lag appears together with its negative
    lags = set(zip(eng.geo.hy.tolist(), eng.geo.hx.tolist()))
    assert all((-a, -b) not in lags for a, b in lags)
    assert any(b < 0 for _, b in lags)            # signed x offsets are present


def test_ccf_of_a_field_with_itself_equals_its_acf():
    z = _anisotropic_field(48, 30.0, 4.0, 2.0, seed=1)
    mask = np.ones_like(z, dtype=bool)
    mask[:10, :7] = False
    acf = inf.fft_acf(z, mask, DX, 600.0, n_bins=10)
    ccf = inf.fft_ccf(z, z, mask, DX, 600.0, n_bins=10)
    assert ccf["ccf0"] == pytest.approx(1.0)
    assert np.allclose(acf["acf"], ccf["ccf"], equal_nan=True)
    for ang in acf["directional"]:
        assert np.allclose(acf["directional"][ang], ccf["directional"][ang], equal_nan=True)


# ---------------------------------------------------------------------------
# Anisotropy (full-plane ACF distinguishes 45° from 135°)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [135.0, 45.0])
def test_anisotropy_recovers_major_axis(theta):
    z = _anisotropic_field(128, theta, sigma_major=6.0, sigma_minor=2.0, seed=0)
    acf = inf.fft_acf(z, np.ones_like(z, dtype=bool), DX, 900.0, n_bins=20)
    est = inf.anisotropy_from_directional(acf["directional"], acf["lags_m"])
    assert abs(inf._axial_diff_deg(est["theta_deg"], theta)) <= 25.0
    assert est["ratio"] < 0.6


def test_isotropic_field_has_ratio_near_one_and_is_unreliable():
    z = _anisotropic_field(128, 0.0, sigma_major=4.0, sigma_minor=4.0, seed=0)
    mask = np.ones_like(z, dtype=bool)
    acf = inf.fft_acf(z, mask, DX, 900.0, n_bins=20)
    assert inf.anisotropy_from_directional(acf["directional"], acf["lags_m"])["ratio"] > 0.8
    boot = inf.anisotropy_bootstrap(z, mask, DX, 900.0, n_bins=20, n_boot=12, seed=0)
    assert boot["reliable"] is False


def test_anisotropy_bootstrap_marks_strong_anisotropy_reliable():
    z = _anisotropic_field(128, 135.0, sigma_major=6.0, sigma_minor=2.0, seed=0)
    boot = inf.anisotropy_bootstrap(z, np.ones_like(z, dtype=bool), DX, 900.0, n_bins=20, n_boot=12, seed=0)
    assert boot["reliable"] is True
    assert boot["ci_width_deg"] < 30.0
    assert boot["n_boot"] >= 10


# ---------------------------------------------------------------------------
# Permutation band / practical range
# ---------------------------------------------------------------------------


def test_iid_noise_is_rarely_significant():
    mask = np.ones((96, 96), dtype=bool)
    fracs = []
    for seed in range(4):
        z = np.random.default_rng(100 + seed).standard_normal(mask.shape)
        acf = inf.fft_acf(z, mask, DX, 900.0, n_bins=20)
        mean, sd = inf.permutation_band(z, mask, DX, 900.0, n_bins=20, n_perm=19, seed=seed)
        fracs.append(np.mean(np.abs(acf["acf"] - mean) > 2.0 * sd))
    assert np.mean(fracs) <= 0.15


def test_fit_practical_range_recovers_exponential_model():
    h = np.linspace(30.0, 2000.0, 25)
    acf = 0.9 * np.exp(-3.0 * h / 600.0)
    assert inf.fit_practical_range(h, acf) == pytest.approx(600.0, rel=1e-3)
    # White noise → fallback: the first lag below 0.05
    assert inf.fit_practical_range(h, np.full_like(h, 0.01)) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Ring (distributed-lag) influence
# ---------------------------------------------------------------------------


def _gaussian_mass_radius(sigma_cells: float, mass: float = 0.9) -> float:
    K = int(8 * sigma_cells)
    yy, xx = np.mgrid[-K:K + 1, -K:K + 1]
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_cells**2))
    r = np.hypot(xx, yy).ravel() * DX
    order = np.argsort(r, kind="stable")
    cum = np.cumsum(k.ravel()[order]) / k.sum()
    return float(r[order][np.searchsorted(cum, mass)])


@pytest.mark.parametrize("x_range_cells", [0.0, 2.0, 5.0])
def test_ring_influence_recovers_kernel_mass_radius(x_range_cells):
    from sparc.core.synthetic import gaussian_random_field

    n = 96
    truth = _gaussian_mass_radius(3.0)
    rng = np.random.default_rng(0)
    # Draw on a larger domain and crop: the kernel then also sees X outside
    # the study area, as in real data.
    big = n + 40
    X = rng.standard_normal((big, big)) if x_range_cells == 0 else gaussian_random_field((big, big), x_range_cells, rng)
    signal = ndimage.gaussian_filter(X, 3.0, mode="wrap")
    X, signal = X[20:-20, 20:-20], signal[20:-20, 20:-20]
    resid = signal + 0.3 * signal.std() * rng.standard_normal((n, n))
    mask = np.ones((n, n), dtype=bool)
    r, betas, edges = inf.ring_influence(resid, X, mask, DX, n_rings=10, max_lag_m=900.0)
    assert r == pytest.approx(truth, rel=0.30)
    assert len(betas) == len(edges) - 1
    assert edges[0] == 0.0 and edges[1] == DX / 2 and edges[-1] == 900.0
    assert np.all(betas >= 0)                                  # one-signed profile, positive effect
    # A free-ring fit with a tiny ridge is what the regularised profile replaces.
    fit = inf.fit_ring_profile(resid, X, mask, DX, 10, 900.0)
    assert fit.significant and fit.family == "gauss"
    assert fit.betas_free.shape == betas.shape


def test_ring_influence_green_family_matches_operator_kernel():
    """A target produced by the physics operator is best described by the
    operator's own Green's function family."""
    from sparc.core import operators as ops
    from sparc.core.synthetic import kernel_mass_radius

    n, L = 96, 90.0
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n + 40, n + 40))
    signal = ops.solve(X, L, (0.0, 0.0), dx=DX)[20:-20, 20:-20]
    X = X[20:-20, 20:-20]
    resid = signal + 0.2 * signal.std() * rng.standard_normal((n, n))
    fit = inf.fit_ring_profile(resid, X, np.ones((n, n), dtype=bool), DX, 10, 900.0)
    assert fit.family == "green"
    assert fit.scale_m == pytest.approx(L, rel=0.25)
    assert fit.range_m == pytest.approx(kernel_mass_radius(L, (0.0, 0.0), DX), rel=0.30)


def test_ring_influence_pure_noise_predictor_has_no_range():
    n = 96
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, n))
    resid = ndimage.gaussian_filter(rng.standard_normal((n, n)), 4.0)   # structure unrelated to X
    fit = inf.fit_ring_profile(resid, X, np.ones((n, n), dtype=bool), DX, 10, 900.0)
    assert not fit.significant
    assert fit.range_m == pytest.approx(DX / 2)


def test_ring_edges_are_nonempty_annuli():
    edges = inf.ring_edges(DX, 10, 2000.0)
    assert np.all(np.diff(edges) > 0)
    K = int(2000 / DX)
    yy, xx = np.mgrid[-K:K + 1, -K:K + 1]
    d = DX * np.hypot(xx, yy)
    for lo, hi in zip(edges[:-1], edges[1:]):
        assert ((d >= lo) & (d <= hi)).any()


# ---------------------------------------------------------------------------
# Focal features
# ---------------------------------------------------------------------------


def _lattice_frame(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, Grid]:
    rng = np.random.default_rng(seed)
    iy, ix = np.nonzero(rng.random((n, n)) < 0.8)
    x = 500.0 + ix * DX
    y = 700.0 + iy * DX
    frame = pd.DataFrame({"a": rng.standard_normal(ix.size), "b": np.full(ix.size, 3.5), "c": rng.random(ix.size)})
    return frame, Grid.from_points(x, y)


def test_focal_features_naming_and_constant_preservation():
    frame, grid = _lattice_frame()
    ranges = {"a": 90.0, "b": 150.0}
    F = inf.focal_features(frame, grid, ranges)
    assert list(F.columns) == ["a__f0.5", "a__f1", "a__f2", "b__f0.5", "b__f1", "b__f2"]
    assert F.index.equals(frame.index)
    assert np.allclose(F[["b__f0.5", "b__f1", "b__f2"]].to_numpy(), 3.5)
    F2 = inf.focal_features(frame, grid, ranges, scales=(1.0,), columns=["a"])
    assert list(F2.columns) == ["a__f1"]
    assert np.allclose(F2["a__f1"], F["a__f1"])


def test_focal_features_spill_to_neighbours_only():
    frame, grid = _lattice_frame()
    ranges = {"a": 90.0}
    base = inf.focal_features(frame, grid, ranges, scales=(2.0,))
    pts = grid.point_coords()
    i0 = int(np.argmin(np.hypot(pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean())))
    edited = frame.copy()
    edited.loc[i0, "a"] += 10.0
    new = inf.focal_features(edited, grid, ranges, scales=(2.0,))
    delta = (new - base)["a__f2"].to_numpy()
    d = np.hypot(*(pts - pts[i0]).T)
    sigma_m = 2.0 * 90.0 / 2.0
    assert delta[i0] > 0
    near = (d > 0) & (d <= DX * 1.5)
    assert np.all(delta[near] > 0)                         # spillover to the 8 neighbours
    assert np.all(np.abs(delta[d > 8 * sigma_m]) < 1e-9)   # nothing far away
    # own-cell change ≈ w0·Δ / Σ(valid weights) ≥ w0·Δ
    w0 = inf.focal_self_weight(ranges, grid, scales=(2.0,))["a__f2"]
    assert delta[i0] >= w0 * 10.0 - 1e-9


# ---------------------------------------------------------------------------
# Driver on the synthetic city
# ---------------------------------------------------------------------------


def test_compute_influence_on_synthetic_city(synthetic_core_data):
    cfg, data = synthetic_core_data
    res = inf.compute_influence(data, {**cfg.raw["influence"], "n_boot": 6, "n_perm": 9})
    dx = data.grid.dx
    r_hi = min(res.max_lag_m, 0.25 * min(data.grid.nx, data.grid.ny) * dx)
    assert set(res.ranges_m) == set(cfg.predictors)
    for r in res.ranges_m.values():
        assert math.isfinite(r) and 2 * dx - 1e-9 <= r <= r_hi + 1e-9
    assert res.max_lag_m <= cfg.raw["influence"]["max_lag_m"]
    assert dx <= res.target_resid_range_m <= res.max_lag_m
    assert res.L_prior_m == pytest.approx(res.target_resid_range_m / math.sqrt(8.0))
    assert res.block_size_m >= 3 * dx
    assert set(res.anisotropy) == {"target", *cfg.predictors}
    for a in res.anisotropy.values():
        assert {"ratio", "theta_deg", "ci_width_deg", "reliable"} <= set(a)
    assert len(res.target_acf["acf"]) == len(res.target_acf["band_sd"]) == len(res.target_acf["lags"])
    d = res.to_dict()
    json.loads(json.dumps(d, allow_nan=False))
    # Focal features follow straight from the result.
    F = inf.focal_features(data.frame, data.grid, res.ranges_m)
    assert F.shape == (data.n, 3 * len(cfg.predictors))
    assert np.isfinite(F.to_numpy()).all()


def test_compute_influence_recovers_planted_canopy_range(synthetic_core_data, synthetic_city):
    """Canopy enters ΔT only through the physics source spread by the
    operator kernel (L = 150 m, v = (60, 0) m), whose 90 %-mass radius is the
    planted area of influence."""
    cfg, data = synthetic_core_data
    res = inf.compute_influence(data, {**cfg.raw["influence"], "n_boot": 2, "n_perm": 2})
    truth = synthetic_city.truth["influence_radius_90"]
    assert truth / 2 <= res.ranges_m["canopy"] <= 2 * truth
    fits = res.diagnostics["ring_fits"]
    assert fits["canopy"]["significant"] and fits["canopy"]["amplitude"] < 0      # canopy cools
    assert fits["impervious"]["significant"] and fits["impervious"]["amplitude"] > 0
    assert all(r >= 2 * data.grid.dx for r in res.ranges_m.values())
