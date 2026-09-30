"""S1 — area of influence: correlograms, influence ranges, anisotropy and
focal features.

Correlograms
------------
For a raster z with validity mask m, the masked, mean-centred empirical
auto-correlation at lag h = (hx, hy) (cells) is

    ρ(h) = [Σ_s m(s)m(s+h)·z̃(s)z̃(s+h) / N(h)] / σ²,     z̃ = z − z̄  (valid cells)

with pair counts N(h) = Σ_s m(s)m(s+h).  Both sums are circular
cross-correlations, computed with one zero-padded FFT each
(``irfft2(F(a)·conj(F(b)))`` = Σ_s a(s+h)·b(s)); padding every axis by the
largest lag K makes the circular result exact for |h| ≤ K.

ρ(h) = ρ(−h), so the *full plane* of distinct lags is the half-plane
hy ∈ [0, K], hx ∈ [−K, K] minus the half-line {hy = 0, hx ≤ 0}.  Keeping
signed hx is what separates 45° from 135° structure; a one-quadrant
correlogram (hx, hy ≥ 0) folds the two diagonals onto each other.

Radial bins pool pairs (Σ numerator / Σ N), i.e. they are pair-weighted.
Directional bins use axial sectors of ±90°/n_dirs around 0/45/90/135°
(angle counter-clockwise from +x; θ and θ + 180° are the same direction).
Cross-correlograms keep the full plane (ρ_ab(h) ≠ ρ_ab(−h)).

Influence range of a predictor (distributed-lag rings)
-----------------------------------------------------
The CCF between the target residual and X_j is the true influence kernel
convolved with X_j's own autocorrelation, so its range is biased high.
Instead we regress the (partial) residual on ring means of the
(normal-scored) predictor,

    r(s) = Σ_m β_m · X̄_m(s) + ε,      X̄_m = mask-normalised mean of X over annulus m

where ring 0 is the cell itself and the annuli have geometrically spaced
outer edges.  If r = k ⊛ X then β_m ≈ Σ_{h∈ring m} k(h) (the kernel mass in
the ring), whatever X's autocorrelation, so the radius at which the
cumulative Σ|β_m| reaches ``mass`` of the total is the kernel's
``mass``-radius — the same definition as
:func:`sparc.core.synthetic.kernel_mass_radius`.  Free β_m are unstable
(collinear rings, confounded outer rings, alternating signs), so the
profile is constrained to β_m = A·w_m(family, scale) — the ring masses of
a one-signed kernel (the physics operator's Green's function or a
Gaussian) whose family and scale are chosen by leave-one-block-out CV, and
gated by a block-jackknife test on A (:func:`ring_influence`).

Anisotropy
----------
Per-direction practical ranges R(θ) are fitted to the directional
correlograms and an ellipse is fitted in the linear form
1/R(θ)² = A + B·cos 2θ + C·sin 2θ, giving the major-axis direction
θ₀ = ½·atan2(−C, −B) and axis ratio b/a = √((A − ρ)/(A + ρ)), ρ = √(B² + C²).
Reliability comes from block half-sampling (balanced repeated replication):
the spread of θ over random halves of spatial blocks estimates the
sampling variance of the full-data θ.

Focal features
--------------
Kernel-weighted (Gaussian, σ = s·r_j/2) mask-normalised means of each
predictor at scales s ∈ {½, 1, 2}·r_j, recomputed from any (edited) frame
so scenarios propagate spillover through the neighbourhood.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import fft as sfft

from sparc.core import operators as ops
from sparc.core.grid import Grid

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "max_lag_m": 2000.0,
    "n_rings": 10,
    "n_perm": 19,
    "scales": [0.5, 1.0, 2.0],
    "mass": 0.9,
    "n_bins": 20,
    "n_dirs": 4,
    "n_boot": 20,
    "ridge": 1e-3,
    "min_range_cells": 2.0,
    # ring-profile estimator (see ring_influence)
    "families": ("green", "gauss"),
    "transform": "nscore",
    "n_scales": 25,
    "nuisance_scale_m": None,
    "z_thresh": 2.0,
    "jack_side": 5,
}

_MIN_PAIRS = 10          # radial / directional bins with fewer pairs are NaN
_WORKERS = -1            # scipy.fft: use all cores


# ---------------------------------------------------------------------------
# Lag geometry and the masked correlogram engine
# ---------------------------------------------------------------------------


@dataclass
class _LagGeometry:
    """Distinct lags within ``max_lag_m`` and their radial / sector bins."""

    K: int
    hy: np.ndarray             # signed cell offsets (int)
    hx: np.ndarray
    dist: np.ndarray           # metres
    rbin: np.ndarray           # radial bin index per lag
    sectors: dict              # angle_deg -> bool array over lags
    edges: np.ndarray          # radial bin edges (m)


def _lag_geometry(dx: float, max_lag_m: float, n_bins: int, n_dirs: int, shape: tuple[int, int],
                  full_plane: bool) -> _LagGeometry:
    ny, nx = shape
    K = int(max(1, min(math.floor(max_lag_m / dx + 1e-9), max(ny, nx) - 1)))
    hy, hx = np.mgrid[-K:K + 1, -K:K + 1]
    hy, hx = hy.ravel(), hx.ravel()
    if full_plane:
        keep = ~((hy == 0) & (hx == 0))
    else:  # distinct half-plane: hy > 0, or hy == 0 and hx > 0
        keep = (hy > 0) | ((hy == 0) & (hx > 0))
    dist = dx * np.hypot(hx, hy)
    keep &= dist <= max_lag_m + 1e-9
    hy, hx, dist = hy[keep], hx[keep], dist[keep]
    edges = np.linspace(0.0, max_lag_m, n_bins + 1)
    rbin = np.minimum((dist / max_lag_m * n_bins).astype(np.int64), n_bins - 1)
    ang = np.mod(np.degrees(np.arctan2(hy, hx)), 180.0)       # axial, [0, 180)
    half = 90.0 / n_dirs
    sectors = {}
    for k in range(n_dirs):
        a0 = 180.0 * k / n_dirs
        d = np.abs(np.mod(ang - a0 + 90.0, 180.0) - 90.0)     # axial angular distance
        sectors[a0] = d <= half + 1e-9
    return _LagGeometry(K=K, hy=hy, hx=hx, dist=dist, rbin=rbin, sectors=sectors, edges=edges)


class _MaskedCorrelogram:
    """Pair counts for a fixed mask + lag geometry; evaluates correlograms of
    any fields on that mask with two FFTs each."""

    def __init__(self, mask: np.ndarray, dx: float, max_lag_m: float, n_bins: int, n_dirs: int,
                 full_plane: bool = False):
        self.mask = np.asarray(mask, dtype=bool)
        ny, nx = self.mask.shape
        self.geo = _lag_geometry(dx, max_lag_m, n_bins, n_dirs, (ny, nx), full_plane)
        K = self.geo.K
        self.P = (sfft.next_fast_len(ny + K + 1, real=True), sfft.next_fast_len(nx + K + 1, real=True))
        self._iy = np.mod(self.geo.hy, self.P[0])
        self._ix = np.mod(self.geo.hx, self.P[1])
        self.n_bins = n_bins
        m = self.mask.astype(np.float64)
        self.pairs = np.rint(self._xcorr(m, m)).clip(min=0.0)
        self.rad_pairs = np.bincount(self.geo.rbin, weights=self.pairs, minlength=n_bins)
        self.rad_lag = np.bincount(self.geo.rbin, weights=self.pairs * self.geo.dist, minlength=n_bins)
        self.dir_pairs = {a: np.bincount(self.geo.rbin[s], weights=self.pairs[s], minlength=n_bins)
                          for a, s in self.geo.sectors.items()}

    def _xcorr(self, a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        """Σ_s a(s+h)·b(s) at every lag of the geometry."""
        Fa = sfft.rfft2(a, s=self.P, workers=_WORKERS)
        Fb = Fa if b is None or b is a else sfft.rfft2(b, s=self.P, workers=_WORKERS)
        c = sfft.irfft2(Fa * np.conj(Fb), s=self.P, workers=_WORKERS)
        return c[self._iy, self._ix]

    def lags(self) -> np.ndarray:
        """Pair-weighted mean lag of each radial bin (geometric centre if empty)."""
        centre = 0.5 * (self.geo.edges[:-1] + self.geo.edges[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            lag = self.rad_lag / self.rad_pairs
        return np.where(self.rad_pairs > 0, lag, centre)

    def correlate(self, a: np.ndarray, b: np.ndarray | None = None, directional: bool = True) -> dict:
        """Radial (and directional) correlogram of centred, masked fields."""
        num = self._xcorr(a, b)
        rad = np.bincount(self.geo.rbin, weights=num, minlength=self.n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = rad / self.rad_pairs
        r[self.rad_pairs < _MIN_PAIRS] = np.nan
        out = {"radial": r}
        if directional:
            dirs = {}
            for ang, sel in self.geo.sectors.items():
                d = np.bincount(self.geo.rbin[sel], weights=num[sel], minlength=self.n_bins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = d / self.dir_pairs[ang]
                v[self.dir_pairs[ang] < _MIN_PAIRS] = np.nan
                dirs[ang] = v
            out["directional"] = dirs
        return out


def _centred(raster: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(z̃·m, effective mask, variance) with mean/variance over valid cells."""
    r = np.asarray(raster, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(r)
    if m.sum() < 2:
        raise ValueError("correlogram needs at least two valid cells")
    vals = r[m]
    z = np.zeros_like(r)
    z[m] = vals - vals.mean()
    return z, m, float(np.mean(z[m] ** 2))


def fft_acf(raster: np.ndarray, mask: np.ndarray, dx: float, max_lag_m: float, n_bins: int = 20,
            n_dirs: int = 4) -> dict:
    """Masked, mean-centred, full-plane autocorrelogram (see module docstring).

    Returns ``lags_m`` (pair-weighted mean lag of each radial bin — the bin's
    centre of mass), ``edges_m``, ``acf`` (radial), ``pairs`` (distinct pairs
    per radial bin), ``directional`` {angle_deg: acf}, ``directional_pairs``
    and ``variance``.  Bins with fewer than 10 pairs are NaN.
    """
    z, m, var = _centred(raster, mask)
    eng = _MaskedCorrelogram(m, dx, max_lag_m, n_bins, n_dirs, full_plane=False)
    res = eng.correlate(z)
    scale = 1.0 / var if var > 0 else np.nan
    return {
        "lags_m": eng.lags(),
        "edges_m": eng.geo.edges.copy(),
        "acf": res["radial"] * scale,
        "pairs": eng.rad_pairs.copy(),
        "directional": {a: v * scale for a, v in res["directional"].items()},
        "directional_pairs": {a: p.copy() for a, p in eng.dir_pairs.items()},
        "variance": var,
        "n_valid": int(m.sum()),
    }


def fft_ccf(a: np.ndarray, b: np.ndarray, mask: np.ndarray, dx: float, max_lag_m: float, n_bins: int = 20,
            n_dirs: int = 4) -> dict:
    """Masked cross-correlogram ρ_ab(h) = corr(a(s + h), b(s)) over the full
    plane of lags (both signs of h), binned like :func:`fft_acf`.

    ``ccf0`` is the zero-lag (co-located) correlation.  Directional sectors
    are axial, so h and −h are pooled within a direction.
    """
    m = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    za, m, va = _centred(a, m)
    zb, _, vb = _centred(b, m)
    eng = _MaskedCorrelogram(m, dx, max_lag_m, n_bins, n_dirs, full_plane=True)
    res = eng.correlate(za, zb)
    scale = 1.0 / math.sqrt(va * vb) if va > 0 and vb > 0 else np.nan
    return {
        "lags_m": eng.lags(),
        "edges_m": eng.geo.edges.copy(),
        "ccf": res["radial"] * scale,
        "ccf0": float(np.sum(za * zb) / m.sum() * scale),
        "pairs": eng.rad_pairs.copy(),
        "directional": {k: v * scale for k, v in res["directional"].items()},
        "n_valid": int(m.sum()),
    }


def permutation_band(raster: np.ndarray, mask: np.ndarray, dx: float, max_lag_m: float, n_bins: int = 20,
                     n_perm: int = 19, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Null band of the radial ACF: shuffle values among valid cells
    (destroying spatial structure, keeping the mask and the marginal) and
    recompute.  Returns per-bin (mean, sd); a lag is significant when
    |acf − mean| > 2·sd."""
    z, m, var = _centred(raster, mask)
    eng = _MaskedCorrelogram(m, dx, max_lag_m, n_bins, 1, full_plane=False)
    rng = np.random.default_rng(seed)
    vals = z[m]
    sims = np.empty((max(1, n_perm), n_bins))
    zp = np.zeros_like(z)
    for i in range(sims.shape[0]):
        zp[m] = rng.permutation(vals)
        sims[i] = eng.correlate(zp, directional=False)["radial"] / var
    return np.nanmean(sims, axis=0), np.nanstd(sims, axis=0, ddof=1) if sims.shape[0] > 1 else np.zeros(n_bins)


# ---------------------------------------------------------------------------
# Practical range
# ---------------------------------------------------------------------------


def fit_practical_range(lags_m: np.ndarray, acf: np.ndarray) -> float:
    """Practical range R (m) of ρ(h) ≈ c·exp(−3h/R) (ρ(R) = 5 % of c).

    Least squares (scipy) over the leading run of positive, finite lags —
    later positive lobes come from large-scale structure, not the local
    decay.  Falls back to the first lag where ρ < 0.05 (or the largest lag
    if it never decays) when fewer than two positive lags exist, when the
    first lag is already below 0.05, or when the fit fails.
    """
    from scipy.optimize import least_squares

    h = np.asarray(lags_m, dtype=np.float64)
    r = np.asarray(acf, dtype=np.float64)
    ok = np.isfinite(h) & np.isfinite(r)
    h, r = h[ok], r[ok]
    if h.size == 0:
        return float("nan")
    below = np.flatnonzero(r < 0.05)
    fallback = float(h[below[0]] if below.size else h[-1])
    nonpos = np.flatnonzero(r <= 0.0)
    stop = int(nonpos[0]) if nonpos.size else r.size
    hh, rr = h[:stop], r[:stop]
    if hh.size < 2 or rr[0] < 0.05:
        return fallback

    def resid(p):
        c, logR = p
        return c * np.exp(-3.0 * hh / np.exp(logR)) - rr

    x0 = [min(max(rr[0] * math.exp(3.0 * hh[0] / max(fallback, hh[0])), 1e-3), 1.9),
          math.log(max(fallback, hh[0]))]
    lo_R, hi_R = math.log(max(hh[0] * 1e-2, 1e-6)), math.log(hh[-1] * 1e3)
    x0[1] = min(max(x0[1], lo_R + 1e-6), hi_R - 1e-6)
    try:
        sol = least_squares(resid, x0, bounds=([1e-6, lo_R], [2.0, hi_R]), method="trf")
    except Exception:  # pragma: no cover - defensive
        return fallback
    c, logR = sol.x
    R = float(math.exp(logR))
    if not sol.success or not math.isfinite(R) or c < 0.05:
        return fallback
    return R


# ---------------------------------------------------------------------------
# Ring (distributed-lag) influence
# ---------------------------------------------------------------------------


def _lattice_radii(dx: float, max_lag_m: float) -> np.ndarray:
    K = int(math.floor(max_lag_m / dx + 1e-9))
    i, j = np.mgrid[0:K + 1, 0:K + 1]
    r = np.unique(np.round(dx * np.hypot(i, j), 9))
    return r[r <= max_lag_m + 1e-9]


def ring_edges(dx: float, n_rings: int, max_lag_m: float) -> np.ndarray:
    """Ring edges [0, dx/2, e_1, …, e_M] (metres).

    Ring 0 (d < dx/2) is the cell itself.  The annuli's outer edges are
    geometrically spaced from 1.5·dx (so ring 1 holds the 8 neighbours) to
    ``max_lag_m`` and snapped to midpoints between consecutive distinct
    lattice radii, so no annulus is empty; duplicates after snapping are
    merged, so M ≤ ``n_rings``.
    """
    radii = _lattice_radii(dx, max_lag_m)          # includes 0
    if radii.size < 3:
        return np.array([0.0, dx / 2.0, max(max_lag_m, dx * 1.5)])
    mids = 0.5 * (radii[1:-1] + radii[2:])          # midpoints above the first neighbour ring
    r_min = min(1.5 * dx, max_lag_m)
    n = max(1, int(n_rings))
    targets = r_min * (max_lag_m / r_min) ** (np.arange(n) / max(1, n - 1)) if n > 1 else np.array([max_lag_m])
    snapped = []
    for t in targets[:-1]:
        snapped.append(float(mids[np.argmin(np.abs(mids - t))]) if mids.size else t)
    snapped.append(float(max_lag_m))
    outer = np.unique(np.round(snapped, 9))
    outer = outer[outer > dx / 2.0]
    return np.concatenate([[0.0, dx / 2.0], outer])


def ring_features(x_raster: np.ndarray, mask: np.ndarray, dx: float, edges: np.ndarray,
                  min_count: float = 0.5) -> np.ndarray:
    """Mask-normalised mean of ``x_raster`` over each ring → (n_rings, ny, nx).

    Ring m covers lattice offsets with edges[m] ≤ d < edges[m+1] (ring 0 is
    the cell itself).  FFT convolution of m·x and of m with the annulus
    indicator; NaN where the annulus holds no valid cell.
    """
    x = np.asarray(x_raster, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    ny, nx = m.shape
    K = int(math.ceil(edges[-1] / dx))
    P = (sfft.next_fast_len(ny + K + 1, real=True), sfft.next_fast_len(nx + K + 1, real=True))
    hy, hx = np.mgrid[-K:K + 1, -K:K + 1]
    d = dx * np.hypot(hx, hy)
    mf = m.astype(np.float64)
    F_num = sfft.rfft2(np.where(m, x, 0.0), s=P, workers=_WORKERS)
    F_den = sfft.rfft2(mf, s=P, workers=_WORKERS)
    out = np.full((len(edges) - 1, ny, nx), np.nan)
    out[0] = np.where(m, x, np.nan)
    for k in range(1, len(edges) - 1):
        sel = (d >= edges[k]) & (d < edges[k + 1]) if k < len(edges) - 2 else (d >= edges[k]) & (d <= edges[k + 1] + 1e-9)
        if not sel.any():
            continue
        ker = np.zeros(P)
        ker[np.mod(hy[sel], P[0]), np.mod(hx[sel], P[1])] = 1.0
        Kh = sfft.rfft2(ker, workers=_WORKERS)
        num = sfft.irfft2(F_num * Kh, s=P, workers=_WORKERS)[:ny, :nx]
        den = sfft.irfft2(F_den * Kh, s=P, workers=_WORKERS)[:ny, :nx]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(den >= min_count, num / den, np.nan)
    return out


def _standardise_raster(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    out = np.full_like(x, np.nan)
    if m.sum() == 0:
        return out
    sd = x[m].std()
    out[m] = (x[m] - x[m].mean()) / (sd if sd > 0 else 1.0)
    return out


def rbf_controls(shape: tuple[int, int], dx: float, scale_m: float) -> np.ndarray:
    """Coarse Gaussian RBF basis (K, ny, nx): centres on a regular lattice
    spaced ≤ ``scale_m`` spanning the raster, σ = 0.6·scale_m.  Used as
    nuisance controls that absorb smooth (≥ scale_m) confounding."""
    ny, nx = shape
    ext_x, ext_y = (nx - 1) * dx, (ny - 1) * dx
    cx = np.linspace(0.0, ext_x, max(2, int(math.ceil(ext_x / scale_m)) + 1))
    cy = np.linspace(0.0, ext_y, max(2, int(math.ceil(ext_y / scale_m)) + 1))
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
    s2 = 2.0 * (0.6 * scale_m) ** 2
    return np.stack([np.exp(-((X - a) ** 2 + (Y - b) ** 2) / s2) for b in cy for a in cx])


#: 90 %-mass radius / scale parameter of each kernel family (2-D, continuous).
_FAMILY_R90 = {"gauss": 2.146, "exp": 3.89, "green": 3.9}


@functools.lru_cache(maxsize=4096)
def _kernel_ring_weights_cached(edges: tuple, dx: float, family: str, scale_m: float) -> np.ndarray:
    K = int(math.ceil(edges[-1] / dx))
    hy, hx = np.mgrid[-K:K + 1, -K:K + 1]
    d = dx * np.hypot(hx, hy)
    if family == "gauss":
        k = np.exp(-(d**2) / (2.0 * scale_m**2))
    elif family == "exp":
        k = np.exp(-d / scale_m)
    elif family == "green":
        q = np.zeros(d.shape)
        q[K, K] = 1.0
        k = ops.solve(q, scale_m, (0.0, 0.0), dx=dx, pad=int(math.ceil(6.0 * scale_m / dx)) + 4)
    else:
        raise ValueError(f"unknown kernel family {family!r}")
    w = np.array([k[_ring_selector(d, edges, i)].sum() for i in range(len(edges) - 1)])
    return w / w.sum()


def _ring_selector(d: np.ndarray, edges, i: int) -> np.ndarray:
    """Offsets of ring i (the last ring includes its outer edge)."""
    if i < len(edges) - 2:
        return (d >= edges[i]) & (d < edges[i + 1])
    return (d >= edges[i]) & (d <= edges[i + 1] + 1e-9)


def kernel_ring_weights(edges: np.ndarray, dx: float, family: str, scale_m: float) -> np.ndarray:
    """Normalised mass of a kernel in each ring (offsets up to edges[-1]).

    Families: ``"green"`` — the discrete Green's function of (1 − ℓ²∇²)
    (the S2 physics operator without advection; K₀-like, sharp centre and
    heavy tail), ``"gauss"`` — exp(−d²/2σ²), ``"exp"`` — exp(−d/ℓ).
    """
    return _kernel_ring_weights_cached(tuple(float(e) for e in edges), float(dx), family, float(scale_m)).copy()


def _normal_scores(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata

    out = np.full(np.shape(x), np.nan)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    if m.sum() > 1:
        out[m] = norm.ppf((rankdata(x[m]) - 0.5) / m.sum())
    return out


def _mass_radius(beta: np.ndarray, edges: np.ndarray, mass: float, interpolate: bool) -> float:
    w = np.abs(beta)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return float(edges[1])
    cum = np.cumsum(w) / tot
    k = min(int(np.searchsorted(cum, mass - 1e-12)), len(w) - 1)
    if not interpolate:
        return float(edges[k + 1])
    prev = cum[k - 1] if k > 0 else 0.0
    frac = (mass - prev) / max(cum[k] - prev, 1e-12)
    return float(edges[k] + float(np.clip(frac, 0.0, 1.0)) * (edges[k + 1] - edges[k]))


@dataclass
class RingFit:
    """Distributed-lag kernel estimate for one predictor."""

    range_m: float
    betas: np.ndarray            # fitted kernel profile A·w_m (one-signed, smooth)
    betas_free: np.ndarray       # unconstrained small-ridge ring coefficients (diagnostic)
    edges: np.ndarray
    family: str                  # selected kernel family
    scale_m: float               # its scale parameter (σ, ℓ or L)
    amplitude: float             # A (total kernel mass, per SD of the transformed predictor)
    amplitude_se: float          # block-jackknife SE of A
    significant: bool            # |A| > z·SE
    cv_error: float              # leave-one-block-out squared error at the optimum
    n_controls: int

    def to_dict(self) -> dict:
        return _jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})


def fit_ring_profile(resid_raster: np.ndarray, x_raster: np.ndarray, mask: np.ndarray, dx: float, n_rings: int,
                     max_lag_m: float, mass: float = 0.9, ridge: float = 1e-3, interpolate: bool = True,
                     families: tuple = ("green", "gauss"), n_scales: int = 25, transform: str = "nscore",
                     nuisance_scale_m: float | None = None, z_thresh: float = 2.0, jack_side: int = 5) -> RingFit:
    """Robust distributed-lag kernel of a predictor (see :func:`ring_influence`)."""
    edges = ring_edges(dx, n_rings, max_lag_m)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x_raster)
    xt = _normal_scores(x_raster, m) if transform == "nscore" else _standardise_raster(x_raster, m)
    F = ring_features(xt, m, dx, edges)
    y = np.asarray(resid_raster, dtype=np.float64)
    M = F.shape[0]
    C = rbf_controls(m.shape, dx, nuisance_scale_m) if nuisance_scale_m else np.zeros((0,) + m.shape)
    K = C.shape[0]
    rows = np.asarray(mask, dtype=bool) & np.isfinite(y) & np.all(np.isfinite(F), axis=0)
    own = float(edges[1])
    zeros = np.zeros(M)
    if rows.sum() < M + K + 10:
        log.warning("ring_influence: only %d usable cells; returning the own-cell range", int(rows.sum()))
        return RingFit(own, zeros, zeros, edges, "none", 0.0, 0.0, float("inf"), False, float("nan"), K)

    Z = np.concatenate([F[:, rows], C[:, rows]], axis=0).T
    Z = Z - Z.mean(axis=0)
    yy = y[rows] - y[rows].mean()
    H0 = Z.T @ Z
    g0 = Z.T @ yy
    nb = M + K
    eps = np.zeros(nb)
    eps[:M] = ridge * float(np.mean(np.diag(H0)[:M]))
    if K:
        eps[M:] = 1e-8 * float(np.mean(np.diag(H0)[M:]))
    beta_free = np.linalg.solve(H0 + np.diag(eps), g0)[:M]

    # Per-block sufficient statistics: leave-one-block-out CV and jackknife.
    ny, nx = rows.shape
    iy, ix = np.nonzero(rows)
    js = max(1, int(jack_side))
    blk = (iy * js // ny) * js + (ix * js // nx)
    stats = []
    for b in np.unique(blk):
        sel = blk == b
        if sel.sum() == yy.size:
            continue
        Zb, yb = Z[sel], yy[sel]
        stats.append((Zb.T @ Zb, Zb.T @ yb, float(yb @ yb)))

    best = None
    r_grid = np.geomspace(dx, max_lag_m, max(2, int(n_scales)))
    for fam in families:
        for r_t in r_grid:
            scale = float(r_t / _FAMILY_R90[fam])
            w = kernel_ring_weights(edges, dx, fam, scale)
            T = np.zeros((nb, 1 + K))
            T[:M, 0] = w
            T[M:, 1:] = np.eye(K)
            H = T.T @ H0 @ T
            H[np.diag_indices_from(H)] += 1e-10 * np.trace(H) / H.shape[0]
            g = T.T @ g0
            err = 0.0
            for Hb, gb, yyb in stats:
                Hb_t, gb_t = T.T @ Hb @ T, T.T @ gb
                th = np.linalg.solve(H - Hb_t, g - gb_t)
                err += yyb - 2.0 * th @ gb_t + th @ Hb_t @ th
            if not stats:
                th = np.linalg.solve(H, g)
                err = float(yy @ yy - 2.0 * th @ g + th @ H @ th)
            if best is None or err < best[0]:
                best = (float(err), fam, scale, w, T, H, g)

    err, fam, scale, w, T, H, g = best
    A = float(np.linalg.solve(H, g)[0])
    reps = [float(np.linalg.solve(H - T.T @ Hb @ T, g - T.T @ gb)[0]) for Hb, gb, _ in stats]
    B = len(reps)
    se = float(math.sqrt((B - 1) / B * np.sum((np.asarray(reps) - np.mean(reps)) ** 2))) if B >= 4 else 0.0
    significant = bool(abs(A) > z_thresh * se)
    profile = A * w
    r = _mass_radius(profile, edges, mass, interpolate) if significant else own
    return RingFit(r, profile, beta_free, edges, fam, scale, A, se, significant, err, K)


def ring_influence(resid_raster: np.ndarray, x_raster: np.ndarray, mask: np.ndarray, dx: float, n_rings: int,
                   max_lag_m: float, mass: float = 0.9, ridge: float = 1e-3, interpolate: bool = True,
                   **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
    """Distributed-lag influence range of a predictor.

    Regresses ``resid_raster`` (valid cells) on the mask-normalised ring
    means of the transformed predictor; β_m estimates the influence-kernel
    mass in ring m.  Free ring coefficients are unstable — adjacent rings of
    a smooth predictor are collinear, wide outer rings (whose means barely
    vary) pick up smooth confounding, and alternating signs make "90 % of
    Σ|β|" meaningless — so the profile is regularised to be smooth and
    one-signed by construction:

    1. **Kernel-family profile** — β_m = A·w_m(family, scale), where w_m is
       the ring mass of a unit kernel: the discrete Green's function of
       (1 − ℓ²∇²) (``"green"``, the physics operator's shape: sharp centre,
       heavy tail) or a Gaussian (``"gauss"``); ``"exp"`` is also
       available.  Family and scale (``n_scales`` values with 90 %-radius
       from dx to ``max_lag_m``) are chosen by leave-one-block-out CV over a
       ``jack_side``² tiling — spatially honest, unlike GCV.
    2. **Transform** — the predictor is replaced by its normal scores
       (``transform="nscore"``, default; ``"standardise"`` for z-scores),
       which linearises monotone saturating responses and zero-inflated
       predictors (e.g. canopy) that otherwise distort the scale.
    3. **Significance gate** — if |A| ≤ ``z_thresh``·SE (block jackknife)
       the predictor has no detectable neighbourhood influence and the
       own-cell edge dx/2 is returned.
    4. Optional **nuisance controls** (``nuisance_scale_m``): a coarse RBF
       basis entering unpenalised to absorb smooth confounding.  Off by
       default: on the synthetic city, RBFs at scale ≥ max_lag were
       collinear with the outer rings and destabilised the estimates.

    r_j is where the cumulative |β| of the fitted profile first reaches
    ``mass`` of its total (within ``max_lag_m``): linearly interpolated in
    radius inside that ring (``interpolate=True``, default) or its outer
    edge.  ``ridge`` only affects the diagnostic free coefficients
    (:attr:`RingFit.betas_free`).

    Returns (r_j [m], profile β (n_rings+1,), edges [0, dx/2, e_1, …]).
    Use :func:`fit_ring_profile` for the full :class:`RingFit`.
    """
    fit = fit_ring_profile(resid_raster, x_raster, mask, dx, n_rings, max_lag_m, mass, ridge, interpolate, **kwargs)
    return fit.range_m, fit.betas, fit.edges


# ---------------------------------------------------------------------------
# Anisotropy
# ---------------------------------------------------------------------------


def _axial_diff_deg(a: np.ndarray | float, b: float) -> np.ndarray:
    """Signed axial difference a − b wrapped to [−90, 90)."""
    return np.mod(np.asarray(a, dtype=float) - b + 90.0, 180.0) - 90.0


def anisotropy_from_directional(directional: dict, lags: np.ndarray) -> dict:
    """Ellipse fit to per-direction practical ranges.

    Returns ``ratio`` (minor/major, in (0, 1]), ``theta_deg`` (major-axis
    direction, axial, [0, 180)) and ``ranges_m`` {angle: R}.  NaN ratio/θ if
    fewer than three directions yield a range or the fit is degenerate.
    """
    angles, ranges = [], {}
    for ang in sorted(directional):
        R = fit_practical_range(lags, directional[ang])
        ranges[float(ang)] = float(R)
        if math.isfinite(R) and R > 0:
            angles.append((float(ang), R))
    out = {"ratio": float("nan"), "theta_deg": float("nan"), "ranges_m": ranges}
    if len(angles) < 3:
        return out
    th = np.deg2rad([a for a, _ in angles])
    w = np.array([1.0 / R**2 for _, R in angles])
    A = np.column_stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th)])
    (A0, B, C), *_ = np.linalg.lstsq(A, w, rcond=None)
    rho = math.hypot(B, C)
    inv_minor2 = A0 + rho                  # 1/b² (short axis → large curvature)
    inv_major2 = max(A0 - rho, 0.0)        # 1/a²
    if inv_minor2 <= 0:
        return out
    out["ratio"] = float(math.sqrt(inv_major2 / inv_minor2))
    out["theta_deg"] = float(np.mod(0.5 * np.degrees(math.atan2(-C, -B)), 180.0)) if rho > 0 else 0.0
    return out


def anisotropy_bootstrap(raster: np.ndarray, mask: np.ndarray, dx: float, max_lag_m: float, n_bins: int = 20,
                         n_dirs: int = 4, n_boot: int = 20, block_m: float | None = None, seed: int = 0,
                         max_ratio: float = 0.87, max_ci_deg: float = 30.0) -> dict:
    """Anisotropy of a raster with a block half-sampling reliability check.

    Each replicate keeps a random half of square spatial blocks (side
    ``block_m``, default max(8·dx, max_lag/4)), recomputes the directional
    ACF and the ellipse.  ``ci_width_deg`` = 2·1.96·sd of the axial θ
    deviations from the full-data θ (half-sample deviations estimate the
    full-sample variance).  ``reliable = ratio ≤ max_ratio and
    ci_width_deg < max_ci_deg``.
    """
    acf = fft_acf(raster, mask, dx, max_lag_m, n_bins, n_dirs)
    est = anisotropy_from_directional(acf["directional"], acf["lags_m"])
    m = np.asarray(mask, dtype=bool) & np.isfinite(raster)
    block_m = float(block_m) if block_m else max(8.0 * dx, max_lag_m / 4.0)
    B = max(2, int(round(block_m / dx)))
    ny, nx = m.shape
    by, bx = np.arange(ny)[:, None] // B, np.arange(nx)[None, :] // B
    nby, nbx = int(by.max()) + 1, int(bx.max()) + 1
    rng = np.random.default_rng(seed)
    thetas, ratios = [], []
    for _ in range(int(n_boot)):
        keep = rng.random((nby, nbx)) < 0.5
        mb = m & keep[by, bx]
        if mb.sum() < max(50, 0.1 * m.sum()):
            continue
        try:
            a = fft_acf(raster, mb, dx, max_lag_m, n_bins, n_dirs)
        except ValueError:
            continue
        e = anisotropy_from_directional(a["directional"], a["lags_m"])
        if math.isfinite(e["theta_deg"]):
            thetas.append(e["theta_deg"])
            ratios.append(e["ratio"])
    ci = float("nan")
    if len(thetas) >= 3 and math.isfinite(est["theta_deg"]):
        dev = _axial_diff_deg(np.array(thetas), est["theta_deg"])
        ci = float(2.0 * 1.96 * math.sqrt(np.mean(dev**2)))
    ratio = est["ratio"]
    reliable = bool(math.isfinite(ratio) and math.isfinite(ci) and ratio <= max_ratio and ci < max_ci_deg)
    return {
        "ratio": ratio,
        "theta_deg": est["theta_deg"],
        "ci_width_deg": ci,
        "reliable": reliable,
        "ranges_m": est["ranges_m"],
        "ratio_boot_sd": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else float("nan"),
        "n_boot": len(thetas),
    }


# ---------------------------------------------------------------------------
# Result + driver
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


@dataclass
class InfluenceResult:
    """S1 output: per-predictor influence ranges, the target-residual
    correlogram (→ physics length-scale prior and CV block size) and
    anisotropy diagnostics."""

    ranges_m: dict[str, float]
    ring_betas: dict[str, list]
    ring_edges_m: list
    target_acf: dict
    target_resid_range_m: float
    L_prior_m: float
    anisotropy: dict[str, dict]
    block_size_m: float
    cell_m: float
    max_lag_m: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonable({
            "ranges_m": self.ranges_m,
            "ring_betas": self.ring_betas,
            "ring_edges_m": self.ring_edges_m,
            "target_acf": self.target_acf,
            "target_resid_range_m": self.target_resid_range_m,
            "L_prior_m": self.L_prior_m,
            "anisotropy": self.anisotropy,
            "block_size_m": self.block_size_m,
            "cell_m": self.cell_m,
            "max_lag_m": self.max_lag_m,
            "diagnostics": self.diagnostics,
        })


def _numeric_predictors(data) -> list[str]:
    return [c for c in data.frame.columns if c in data.X.columns]


def compute_influence(data, cfg_influence: dict | None, residual: np.ndarray | None = None,
                      predictors: list[str] | None = None, seed: int = 0) -> InfluenceResult:
    """S1: influence ranges, target-residual correlogram and anisotropy.

    ``residual`` defaults to y minus a Ridge(α=1) fit on the standardised
    encoded features, which removes the predictors' co-located effect so the
    target ACF reflects unexplained spatial structure.  For predictor j the
    ring regression then uses the *partial* residual (residual + b_j·X_j):
    subtracting X_j's own co-located fit would otherwise leave a large
    negative own-cell β that swamps the kernel.  A caller-supplied residual
    (e.g. from S2 models) is used as is.  Ring estimation options come from
    ``cfg_influence`` (``families``, ``transform``, ``n_scales``,
    ``nuisance_scale_m``, ``z_thresh``, ``jack_side``; see
    :func:`ring_influence`).

    The effective maximum lag is min(max_lag_m, ½·largest domain side).
    Predictor ranges are clipped to [``min_range_cells``·dx,
    min(max_lag, extent/4)] with extent the shorter grid side in metres, so
    the widest focal kernel (σ = 2·r/2 = r) still varies across the study
    area — a near-constant focal feature would make uniform-adoption
    scenarios extrapolate.  The target-residual range is clipped to
    [dx, max_lag].
    """
    from sklearn.linear_model import Ridge

    cfg = {**DEFAULTS, **(cfg_influence or {})}
    grid: Grid = data.grid
    dx = float(grid.dx)
    max_lag = float(min(float(cfg["max_lag_m"]), 0.5 * max(grid.nx, grid.ny) * dx))
    max_lag = max(max_lag, 2.0 * dx)
    n_bins, n_dirs = int(cfg["n_bins"]), int(cfg["n_dirs"])
    r_lo = float(cfg["min_range_cells"]) * dx
    r_hi = max(r_lo, min(max_lag, 0.25 * min(grid.nx, grid.ny) * dx))
    ring_kw = {k: cfg[k] for k in ("families", "transform", "n_scales", "nuisance_scale_m", "z_thresh", "jack_side")}
    ring_kw["families"] = tuple(ring_kw["families"])
    preds = list(predictors) if predictors is not None else _numeric_predictors(data)

    partial: dict[str, np.ndarray] = {}
    if residual is None:
        X = data.X.to_numpy(dtype=np.float64)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        model = Ridge(alpha=1.0).fit(Xs, data.y)
        resid = np.asarray(data.y, dtype=np.float64) - model.predict(Xs)
        cols = list(data.X.columns)
        for c in preds:
            if c in cols:
                j = cols.index(c)
                partial[c] = resid + model.coef_[j] * Xs[:, j]
    else:
        resid = np.asarray(residual, dtype=np.float64)
        if resid.shape != (data.n,):
            raise ValueError(f"residual must have shape ({data.n},), got {resid.shape}")

    r_rast = grid.rasterize(resid)
    mask = grid.mask & np.isfinite(r_rast)

    acf = fft_acf(r_rast, mask, dx, max_lag, n_bins, n_dirs)
    band_mean, band_sd = permutation_band(r_rast, mask, dx, max_lag, n_bins, int(cfg["n_perm"]), seed)
    with np.errstate(invalid="ignore"):
        significant = np.abs(acf["acf"] - band_mean) > 2.0 * band_sd
    R_t = fit_practical_range(acf["lags_m"], acf["acf"])
    R_t = float(np.clip(R_t if math.isfinite(R_t) else dx, dx, max_lag))
    block = max(R_t, 3.0 * dx)
    boot_block = max(block, 8.0 * dx)

    anis = {"target": anisotropy_bootstrap(r_rast, mask, dx, max_lag, n_bins, n_dirs,
                                           int(cfg["n_boot"]), boot_block, seed)}
    ranges, betas, ring_info = {}, {}, {}
    edges = ring_edges(dx, int(cfg["n_rings"]), max_lag)
    for c in preds:
        x_rast = grid.rasterize(data.frame[c].to_numpy(dtype=np.float64))
        xm = mask & np.isfinite(x_rast)
        if xm.sum() < 10 or np.nanstd(x_rast[xm]) == 0:
            log.warning("influence: predictor %r is constant or empty; range set to the minimum", c)
            ranges[c], betas[c] = r_lo, []
            continue
        tgt = grid.rasterize(partial[c]) if c in partial else r_rast
        fit = fit_ring_profile(tgt, x_rast, xm, dx, int(cfg["n_rings"]), max_lag, float(cfg["mass"]),
                               float(cfg["ridge"]), **ring_kw)
        edges = fit.edges
        ranges[c] = float(np.clip(fit.range_m, r_lo, r_hi))
        betas[c] = [float(v) for v in fit.betas]
        ring_info[c] = {"raw_range_m": fit.range_m, "family": fit.family, "scale_m": fit.scale_m,
                        "amplitude": fit.amplitude, "amplitude_se": fit.amplitude_se,
                        "significant": fit.significant, "betas_free": fit.betas_free}
        anis[c] = anisotropy_bootstrap(x_rast, xm, dx, max_lag, n_bins, n_dirs,
                                       int(cfg["n_boot"]), boot_block, seed)
        log.info("influence: %s range %.0f m (anisotropy ratio %.2f, θ %.0f°)", c, ranges[c],
                 anis[c]["ratio"], anis[c]["theta_deg"])

    target_acf = {
        "lags": acf["lags_m"], "acf": acf["acf"], "band_mean": band_mean, "band_sd": band_sd,
        "significant": significant, "pairs": acf["pairs"],
        "directional": {str(int(a)): v for a, v in acf["directional"].items()},
    }
    log.info("influence: target residual practical range %.0f m → L prior %.0f m", R_t, R_t / math.sqrt(8.0))
    return InfluenceResult(
        ranges_m=ranges,
        ring_betas=betas,
        ring_edges_m=[float(e) for e in edges],
        target_acf=target_acf,
        target_resid_range_m=R_t,
        L_prior_m=R_t / math.sqrt(8.0),
        anisotropy=anis,
        block_size_m=float(block),
        cell_m=dx,
        max_lag_m=max_lag,
        diagnostics={"residual_source": "ridge" if residual is None else "caller",
                     "n_valid_cells": int(mask.sum()), "range_bounds_m": [r_lo, r_hi], "ring_fits": ring_info},
    )


# ---------------------------------------------------------------------------
# Focal features
# ---------------------------------------------------------------------------


def _focal_sigma_cells(r_m: float, scale: float, dx: float) -> float:
    return float(scale) * float(r_m) / 2.0 / dx


def focal_features(frame: pd.DataFrame, grid: Grid, ranges_m: dict[str, float], scales=(0.5, 1.0, 2.0),
                   columns: list[str] | None = None) -> pd.DataFrame:
    """Gaussian mask-normalised neighbourhood means of each predictor.

    For column c with range r_c and scale s the kernel has σ = s·r_c/2
    (≈ 90 % of a 2-D Gaussian's mass lies within 2.15σ ≈ s·r_c).  Values
    are rasterised from ``frame`` (which may be an edited copy — focal
    values of neighbours then change, which is the spillover channel),
    smoothed over valid cells and sampled back at every point.  Rows align
    with ``frame``; columns are named ``f"{c}__f{s:g}"``.
    """
    if len(frame) != grid.n_points:
        raise ValueError(f"frame has {len(frame)} rows but the grid maps {grid.n_points} points")
    cols = [c for c in (columns if columns is not None else list(frame.columns)) if c in ranges_m and c in frame]
    out: dict[str, np.ndarray] = {}
    for c in cols:
        rast = grid.rasterize(frame[c].to_numpy(dtype=np.float64))
        m = np.isfinite(rast)
        for s in scales:
            sm = ops.masked_gaussian(rast, m, _focal_sigma_cells(ranges_m[c], s, grid.dx))
            out[f"{c}__f{float(s):g}"] = grid.sample(sm)
    return pd.DataFrame(out, index=frame.index)


def focal_self_weight(ranges_m: dict[str, float], grid: Grid, scales=(0.5, 1.0, 2.0)) -> dict[str, float]:
    """Centre-cell weight w0 of each focal kernel ({focal column: w0}).

    Editing only the own cell by Δ moves its focal value by ≈ w0·Δ (exactly
    w0/Σ_valid-weights near mask edges) — used for own-only marginal effects.
    """
    return {f"{c}__f{float(s):g}": ops.self_weight(_focal_sigma_cells(r, s, grid.dx))
            for c, r in ranges_m.items() for s in scales}
