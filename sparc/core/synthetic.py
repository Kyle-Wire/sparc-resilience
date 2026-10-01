"""Synthetic fixtures with planted, analytically known truths.

These drive the core unit tests: every stage must recover what was planted.

* :func:`make_operator_fixture` — a known source ``q`` pushed through the
  advection–diffusion–relaxation operator with known (L, v).
* :func:`make_synthetic_city` — a small city with canopy / impervious /
  albedo / NDVI (mediator) / elevation / water, a smooth confounder, a
  physics-shaped target and an exactly saturating canopy response.
* :func:`make_interference_fixture` and :func:`make_dose_response_fixture`
  — small causal fixtures with known own/neighbour effects and a known
  dose-response curve.

Target units are "°F-like" (arbitrary but realistic magnitudes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from sparc.core import operators as ops


def gaussian_random_field(shape: tuple[int, int], range_cells: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance smooth field (Gaussian-filtered white noise)."""
    z = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=max(range_cells / 2.0, 0.5), mode="wrap")
    return (z - z.mean()) / (z.std() + 1e-12)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def kernel_mass_radius(L: float, v: tuple[float, float], dx: float, mass: float = 0.9, n: int = 257) -> float:
    """Radius (m) around the source containing ``mass`` of |G| for the
    discrete operator Green's function — the true 'area of influence'."""
    q = np.zeros((n, n))
    c = n // 2
    q[c, c] = 1.0
    g = np.abs(ops.solve(q, L, v, dx=dx))
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c).ravel() * dx
    w = g.ravel()
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / w.sum()
    return float(r[order][np.searchsorted(cum, mass)])


# ---------------------------------------------------------------------------
# Operator fixture
# ---------------------------------------------------------------------------


def make_operator_fixture(
    n: int = 128,
    dx: float = 30.0,
    L: float = 150.0,
    v: tuple[float, float] = (90.0, -45.0),
    a: float = 2.0,
    b: float = 0.5,
    noise: float = 0.05,
    seed: int = 0,
) -> dict:
    """y = a·solve(q; L, v) + b + ε on a full n×n grid with a short-range source."""
    rng = np.random.default_rng(seed)
    q = gaussian_random_field((n, n), range_cells=2.0, rng=rng)
    phi = ops.solve(q, L, v, dx=dx)
    y = a * phi + b
    y = y + noise * y.std() * rng.standard_normal(y.shape)
    yy, xx = np.mgrid[0:n, 0:n]
    return {
        "q": q,
        "phi": phi,
        "y": y,
        "x": (xx * dx).ravel().astype(float),
        "y_coord": (yy * dx).ravel().astype(float),
        "values": y.ravel(),
        "truth": {"L": L, "v": v, "a": a, "b": b, "dx": dx},
    }


# ---------------------------------------------------------------------------
# Synthetic city
# ---------------------------------------------------------------------------


@dataclass
class SyntheticCity:
    frame: pd.DataFrame                      # x, y (m), T, predictors, U (hidden confounder)
    truth: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)  # full rasters (ny, nx), NaN outside mask
    mask: np.ndarray | None = None

    def true_footprint(self) -> np.ndarray:
        """Exact footprint per +1 pp canopy at every point: Σ_j ∂ΔT_j/∂c_i =
        a·(Gᵀ∗mask)_i·∂q_i/∂c_i, including the NDVI mediator path."""
        t = self.truth
        c = self.frame["canopy"].to_numpy(float)
        dq = -t["A_c"] * np.exp(-c / t["d_c"]) / t["d_c"] - t["w_ndvi"] * t["k_ndvi"]
        gm = ops.green_mass(self.mask.astype(float), t["L"], t["v"], t["dx"])
        return t["a"] * gm[self.fields["_iy"], self.fields["_ix"]] * dq

    def true_response(self, canopy_increment: float) -> np.ndarray:
        """Exact ΔT change at every point for a uniform canopy increment
        (NDVI updated through its known mediator law)."""
        t = self.truth
        c = self.fields["canopy"]
        dq = -t["A_c"] * np.exp(-np.nan_to_num(c) / t["d_c"]) * (1.0 - np.exp(-canopy_increment / t["d_c"]))
        dq = np.where(self.mask, dq, 0.0)
        # NDVI mediator: ndvi += k_ndvi·Δc, entering q with weight -w_ndvi.
        dq = dq - t["w_ndvi"] * t["k_ndvi"] * canopy_increment * self.mask
        dphi = ops.solve(dq, t["L"], t["v"], dx=t["dx"])
        rows, cols = self.fields["_iy"], self.fields["_ix"]
        return t["a"] * dphi[rows, cols]


def make_synthetic_city(
    n: int = 96,
    dx: float = 30.0,
    mask_fraction: float = 0.7,
    L: float = 150.0,
    v: tuple[float, float] = (60.0, 0.0),
    seed: int = 0,
    noise_frac: float = 0.05,
) -> SyntheticCity:
    """Synthetic city on an n×n lattice (dx metres) with a ragged footprint.

    Planted truths (returned in ``truth``):
      * physics: ΔT_phys = a·solve(q; L, v) with source q built from
        impervious, albedo, NDVI and a saturating canopy term
        −A_c·(1 − exp(−canopy/d_c)); water cells are a heat sink.
      * saturation: a uniform canopy increment d changes every cell by
        (…)·(1 − exp(−d/d_c)) exactly, so the per-cell saturation scale is d_c.
      * a local threshold effect of albedo (> 0.3 cools by 0.6) that only
        tree models capture;
      * an elevation lapse term and a smooth hidden confounder U that
        drives both canopy and temperature.
    """
    rng = np.random.default_rng(seed)
    shape = (n, n)

    # Ragged footprint: threshold a smooth field, keep the largest blob.
    foot = gaussian_random_field(shape, 12.0, rng)
    thr = np.quantile(foot, 1.0 - mask_fraction)
    mask = foot > thr
    lab, nlab = ndimage.label(mask)
    if nlab > 1:
        sizes = ndimage.sum(mask, lab, range(1, nlab + 1))
        mask = lab == (1 + int(np.argmax(sizes)))

    U = gaussian_random_field(shape, 50.0, rng)                       # ~1.5 km confounder
    z_c = gaussian_random_field(shape, 6.0, rng)
    z_1 = gaussian_random_field(shape, 8.0, rng)
    z_2 = gaussian_random_field(shape, 4.0, rng)
    elev = 20.0 + 10.0 * gaussian_random_field(shape, 30.0, rng)

    canopy = 60.0 * _sigmoid(1.2 * z_c + 0.8 * U)
    canopy[rng.random(shape) < 0.2] = 0.0                              # 20% zeros (hurdle)
    impervious = 100.0 * _sigmoid(-0.7 * z_1 + 0.7 * z_2 - 0.5 * U)
    albedo = np.clip(0.18 + 0.06 * gaussian_random_field(shape, 5.0, rng) - 0.03 * (impervious / 100.0 - 0.5), 0.05, 0.6)
    k_ndvi = 0.004
    ndvi = 0.1 + k_ndvi * canopy - 0.001 * impervious + 0.02 * rng.standard_normal(shape)

    # Water: a meandering river (cells within ~1.5 cells of a sine line).
    yy, xx = np.mgrid[0:n, 0:n]
    river_y = n * 0.35 + 6.0 * np.sin(xx / 9.0)
    water = np.abs(yy - river_y) < 1.5
    water_dist = ndimage.distance_transform_edt(~water) * dx

    A_c, d_c = 1.6, 15.0
    w_ndvi = 1.0
    w_water = 1.2
    q = (
        (1.0 - albedo) * (0.4 + 0.6 * impervious / 100.0)
        - A_c * (1.0 - np.exp(-canopy / d_c))
        - w_ndvi * ndvi
        - w_water * water
    )
    q = np.where(mask, q, 0.0)
    q = np.where(mask, q - q[mask].mean(), 0.0)
    a = 6.0
    phi = ops.solve(q, L, v, dx=dx)
    gamma = -0.02
    thresh = -0.6 * (albedo > 0.3)
    T_clean = a * phi + gamma * (elev - elev[mask].mean()) + thresh + 0.4 * U
    sd = T_clean[mask].std()
    T = T_clean + noise_frac * sd * rng.standard_normal(shape) + 88.0

    iy, ix = np.nonzero(mask)
    frame = pd.DataFrame(
        {
            "x": ix * dx + 1000.0,
            "y": iy * dx + 2000.0,
            "T": T[iy, ix],
            "canopy": canopy[iy, ix],
            "impervious": impervious[iy, ix],
            "albedo": albedo[iy, ix],
            "ndvi": ndvi[iy, ix],
            "elevation": elev[iy, ix],
            "water_dist": water_dist[iy, ix],
            "U": U[iy, ix],
        }
    )
    fields = {
        "canopy": np.where(mask, canopy, np.nan),
        "impervious": np.where(mask, impervious, np.nan),
        "albedo": np.where(mask, albedo, np.nan),
        "ndvi": np.where(mask, ndvi, np.nan),
        "q": q,
        "phi": phi,
        "_iy": iy,
        "_ix": ix,
    }
    truth = {
        "L": L, "v": v, "a": a, "dx": dx, "A_c": A_c, "d_c": d_c, "gamma": gamma,
        "k_ndvi": k_ndvi, "w_ndvi": w_ndvi, "w_water": w_water,
        "influence_radius_90": kernel_mass_radius(L, v, dx, 0.9),
        "noise_sd": noise_frac * sd,
    }
    return SyntheticCity(frame=frame, truth=truth, fields=fields, mask=mask)


def synthetic_city_config(out_dir: str = "output/core/synthetic") -> dict:
    """A core config dict matching :func:`make_synthetic_city`'s columns."""
    return {
        "name": "synthetic_city",
        "data": {"path": None, "target": "T", "x": "x", "y": "y", "coord_unit": "m",
                 "target_units": "degF", "background": "median"},
        "predictors": ["canopy", "impervious", "albedo", "ndvi", "elevation", "water_dist"],
        "qa": {"clip": {"canopy": [0, 100], "impervious": [0, 100], "albedo": [0.02, 0.9]}},
        "actionable": {
            "canopy": {"min": 0, "max": 100, "doses": [0, 5, 10, 15, 20, 30, 40], "cost_per_unit": 1.0},
            "albedo": {"min": 0.02, "max": 0.9, "doses": [0, 0.05, 0.1, 0.15, 0.2]},
        },
        "mediators": {"ndvi": {"parents": ["canopy", "impervious"], "context": ["elevation"],
                               "monotone": {"canopy": 1, "impervious": -1}}},
        # The fixture's advection v = (60, 0) m corresponds to a light breeze
        # u = v/τ with τ = 1800 s — supplied as the "wind record" so advection
        # is fitted (as it would be for a real city with ERA5 wind).
        "physics": {"roles": {"albedo": "albedo", "canopy": "canopy", "impervious": "impervious",
                              "ndvi": "ndvi", "elevation": "elevation", "water_distance": "water_dist"},
                    "wind": [60.0 / 1800.0, 0.0], "tau_s": 1800.0},
        "cv": {"n_folds": 3},
        "stacker": {"epochs": 200, "tune_lambda": [0.0, 1.0]},
        "causal": {"treatments": ["canopy"], "confounders": {"canopy": ["impervious", "albedo", "elevation", "water_dist"]},
                   "contrast": {"canopy": 10}},
        "optimize": {"variable": "canopy", "budget": 2000.0},
        "output": {"dir": out_dir},
    }


# ---------------------------------------------------------------------------
# Causal fixtures
# ---------------------------------------------------------------------------


def make_interference_fixture(
    n: int = 64,
    dx: float = 30.0,
    theta_own: float = -0.05,
    theta_nbr: float = -0.03,
    radius_m: float = 150.0,
    seed: int = 0,
) -> dict:
    """Y = θ_o·T + θ_n·T̄ + g(X, s) + ε with confounding through X and space.

    T̄ is the self-excluded Gaussian neighbourhood mean of T with
    σ = radius/2 (the same exposure mapping the core uses).
    """
    rng = np.random.default_rng(seed)
    shape = (n, n)
    x1 = gaussian_random_field(shape, 8.0, rng)
    x2 = gaussian_random_field(shape, 3.0, rng)
    s = gaussian_random_field(shape, 30.0, rng)                      # spatial confounder
    T = 40.0 + 12.0 * x1 + 6.0 * s + 8.0 * gaussian_random_field(shape, 4.0, rng)
    mask = np.ones(shape, dtype=bool)
    sigma_cells = radius_m / dx / 2.0
    Tbar = ops.masked_gaussian(T, mask, sigma_cells, exclude_self=True)
    g = 0.8 * np.sin(x1) + 0.5 * x2**2 + 0.6 * s
    Y = theta_own * T + theta_nbr * Tbar + g + 0.15 * rng.standard_normal(shape)
    yy, xx = np.mgrid[0:n, 0:n]
    frame = pd.DataFrame({"x": (xx * dx).ravel(), "y": (yy * dx).ravel(), "Y": Y.ravel(), "T": T.ravel(),
                          "x1": x1.ravel(), "x2": x2.ravel()})
    return {"frame": frame, "truth": {"theta_own": theta_own, "theta_nbr": theta_nbr, "radius_m": radius_m, "dx": dx}}


def make_dose_response_fixture(n: int = 64, dx: float = 30.0, A: float = 2.0, d: float = 15.0, seed: int = 0) -> dict:
    """No interference: Y = −A·(1 − exp(−T/d)) + g(X) + ε, T confounded by X,
    with a 20% point mass at T = 0."""
    rng = np.random.default_rng(seed)
    shape = (n, n)
    x1 = gaussian_random_field(shape, 6.0, rng)
    x2 = gaussian_random_field(shape, 3.0, rng)
    T = 60.0 * _sigmoid(1.0 * x1 + 0.5 * rng.standard_normal(shape))
    T[rng.random(shape) < 0.2] = 0.0
    f = -A * (1.0 - np.exp(-T / d))
    Y = f + 0.7 * x1 + 0.3 * x2**2 + 0.1 * rng.standard_normal(shape)
    yy, xx = np.mgrid[0:n, 0:n]
    frame = pd.DataFrame({"x": (xx * dx).ravel(), "y": (yy * dx).ravel(), "Y": Y.ravel(), "T": T.ravel(),
                          "x1": x1.ravel(), "x2": x2.ravel()})

    def true_curve(t):
        return -A * (1.0 - np.exp(-np.asarray(t, dtype=float) / d))

    return {"frame": frame, "truth": {"A": A, "d": d, "curve": true_curve}}
