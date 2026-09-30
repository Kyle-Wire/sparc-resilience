"""Regression tests for the FFT correlogram (A1 / A2).

* A1 — ``irfft2(...)[:ny, :nx]`` kept only offsets with dx ≥ 0, dy ≥ 0 (one
  quadrant), so a field elongated along 45° looked far more autocorrelated
  than the same field elongated along 135°.
* A2 — significance used SE = 1/√n_pairs; it is now a permutation null.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import pdist

from sparc.run.spatial_autocorr_comprehensive import (
    SpatialAutocorrelationAnalyzer,
    fft_correlogram,
)

_KEYS = {
    "lag_distances", "morans_i_values", "z_scores", "p_values",
    "optimal_block_size", "first_zero_crossing", "correlogram_results",
}


def _lattice(n: int, spacing: float) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    return np.column_stack([xx.ravel() * spacing, yy.ravel() * spacing]).astype(float)


def _elliptical_kernel(sig_major: float, sig_minor: float, angle_deg: float) -> np.ndarray:
    r = int(4 * sig_major)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    th = np.deg2rad(angle_deg)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    return np.exp(-0.5 * ((u / sig_major) ** 2 + (v / sig_minor) ** 2))


def _aniso_field(noise: np.ndarray, angle_deg: float) -> np.ndarray:
    f = ndimage.convolve(noise, _elliptical_kernel(6.0, 1.0, angle_deg), mode="wrap")
    return (f - f.mean()) / f.std()


def _quadrant_only_radial_acf(grid: np.ndarray, spacing: float, edges: np.ndarray) -> np.ndarray:
    """Reference of the *old* behaviour: offsets with dx ≥ 0, dy ≥ 0 only."""
    ny, nx = grid.shape
    z = grid - grid.mean()
    F = np.fft.rfft2(z, s=(2 * ny, 2 * nx))
    raw = np.fft.irfft2(F * np.conj(F), s=(2 * ny, 2 * nx))[:ny, :nx]
    Fm = np.fft.rfft2(np.ones_like(z), s=(2 * ny, 2 * nx))
    cnt = np.fft.irfft2(Fm * np.conj(Fm), s=(2 * ny, 2 * nx))[:ny, :nx]
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    d = np.hypot(I, J) * spacing
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi) & ((I > 0) | (J > 0))
        out.append(raw[m].sum() / (cnt[m].sum() * z.var()))
    return np.array(out)


# ---------------------------------------------------------------------------
# A1 — direction coverage
# ---------------------------------------------------------------------------

def test_mirrored_anisotropy_gives_same_radial_correlogram():
    n, spacing = 96, 10.0
    coords = _lattice(n, spacing)
    noise = np.random.default_rng(0).normal(size=(n, n))
    f45 = _aniso_field(noise, 45.0)
    f135 = f45[:, ::-1]  # exact mirror image → elongated along 135°

    r45 = fft_correlogram(coords, f45.ravel(), 150.0, 10, n_permutations=0)
    r135 = fft_correlogram(coords, f135.ravel(), 150.0, 10, n_permutations=0)
    np.testing.assert_allclose(r45["morans_i_values"], r135["morans_i_values"], atol=1e-6)

    # Sanity: the test field is discriminating — a one-quadrant estimator
    # (the pre-fix behaviour) sees very different correlograms.
    edges = np.linspace(0.0, 150.0, 11)
    q45 = _quadrant_only_radial_acf(f45, spacing, edges)
    q135 = _quadrant_only_radial_acf(f135, spacing, edges)
    assert np.max(np.abs(q45 - q135)) > 0.25


def test_45_vs_135_filtered_noise_agree_within_tolerance():
    n, spacing = 96, 10.0
    coords = _lattice(n, spacing)
    noise = np.random.default_rng(0).normal(size=(n, n))
    f45 = _aniso_field(noise, 45.0)
    f135 = _aniso_field(noise, 135.0)

    a = fft_correlogram(coords, f45.ravel(), 150.0, 10, n_permutations=0)["morans_i_values"]
    b = fft_correlogram(coords, f135.ravel(), 150.0, 10, n_permutations=0)["morans_i_values"]
    assert np.max(np.abs(a - b)) < 0.08


def test_pair_counts_match_brute_force_unordered_pairs():
    rng = np.random.default_rng(0)
    pts = _lattice(30, 7.0)
    pts = pts[rng.random(len(pts)) < 0.5]
    vals = rng.normal(size=len(pts))
    md, nl = 60.0, 6
    res = fft_correlogram(pts, vals, md, nl, n_permutations=0)

    # Replicate the rasterisation to get the occupied cells' grid positions.
    cell = max(md / (nl * 2.0), 5.0)
    x, y = pts[:, 0], pts[:, 1]
    xs, ys = np.ptp(x), np.ptp(y)
    nx, ny = int(np.ceil(xs / cell)) + 2, int(np.ceil(ys / cell)) + 2
    dx, dy = xs / (nx - 1), ys / (ny - 1)
    ix = np.round((x - x.min()) / dx).astype(int)
    iy = np.round((y - y.min()) / dy).astype(int)
    cells = np.unique(np.column_stack([iy, ix]), axis=0)
    d = pdist(cells * np.array([dy, dx]))
    expected = np.histogram(d[d < md], bins=np.linspace(0.0, md, nl + 1))[0]

    got = [r["n_pairs"] for r in res["correlogram_results"]]
    assert got == expected.tolist()


# ---------------------------------------------------------------------------
# Agreement with the O(n²) Moran's-I correlogram
# ---------------------------------------------------------------------------

def test_fft_matches_direct_morans_i_on_smooth_field():
    m, spacing = 24, 10.0
    coords = _lattice(m, spacing)
    rng = np.random.default_rng(1)
    field = ndimage.gaussian_filter(rng.normal(size=(m, m)), 3.0, mode="reflect")
    field = ((field - field.mean()) / field.std()).ravel()

    ref = SpatialAutocorrelationAnalyzer(coords, max_distance=100.0, n_lags=5).compute_correlogram(
        field, plot=False,
    )
    fft = fft_correlogram(coords, field, 100.0, 5, n_permutations=19)

    np.testing.assert_allclose(fft["lag_distances"], ref["lag_distances"])
    assert np.max(np.abs(fft["morans_i_values"] - np.asarray(ref["morans_i_values"]))) < 0.1
    # Smooth field: strongly positive, significant short-range autocorrelation.
    assert fft["correlogram_results"][0]["significant"]


# ---------------------------------------------------------------------------
# A2 — permutation null
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_iid_noise_rarely_significant(seed):
    rng = np.random.default_rng(seed)
    coords = _lattice(40, 10.0)
    vals = rng.normal(size=len(coords))
    res = fft_correlogram(coords, vals, 200.0, 20, n_permutations=99, random_state=seed)

    frac = np.mean([r["significant"] for r in res["correlogram_results"]])
    assert frac <= 0.15
    assert np.all(np.isfinite(res["z_scores"]))
    assert np.all((res["p_values"] >= 0) & (res["p_values"] <= 1))


def test_output_contract_and_permutation_determinism():
    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 500, size=(400, 2))
    vals = np.sin(coords[:, 0] / 80.0) + 0.3 * rng.normal(size=400)

    a = fft_correlogram(coords, vals, 150.0, 8, n_permutations=19, random_state=3)
    b = fft_correlogram(coords, vals, 150.0, 8, n_permutations=19, random_state=3)
    legacy = fft_correlogram(coords, vals, 150.0, 8, n_permutations=0)

    for res in (a, legacy):
        assert set(res) == _KEYS
        assert len(res["lag_distances"]) == 8
        assert len(res["correlogram_results"]) == 8
        assert isinstance(res["optimal_block_size"], float)
        assert isinstance(res["first_zero_crossing"], float)
    np.testing.assert_array_equal(a["z_scores"], b["z_scores"])
    # The ACF itself does not depend on the significance method.
    np.testing.assert_allclose(a["morans_i_values"], legacy["morans_i_values"])
    # Legacy analytic path: z = acf · √n_pairs.
    r0 = legacy["correlogram_results"][0]
    assert r0["z_score"] == pytest.approx(r0["morans_i"] * np.sqrt(r0["n_pairs"]))
