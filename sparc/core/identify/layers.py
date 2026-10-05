"""Per-cell design layers.

* **Footprints**: canopy and impervious as Gaussian neighbourhood means (100 m, 300 m, 1 km) next to the
  cell's own value (``_0``), for the levels and map designs.
* **Rings**: means over distance rings around every cell, (0, 100], (100, 300] and (300, 1000] m, for the
  street designs.  Raising canopy by +10 pp in every ring out to R is exactly a disk edit of radius R
  around the cell, so ring coefficients add up to "the effect of canopy within R".
* **A flexible canopy response**: in the cell and the 100 m ring canopy enters through a piecewise-linear
  basis (knots at 15% and 40%), so shade that saturates is not forced onto a straight line; the wider
  rings stay linear.  A linear fit to a saturating
  response is dominated by the largest canopy contrasts and understates the effect of the next 10 pp at
  low canopy, where most streets are.  ``edit_`` layers hold how much each basis term changes in each
  ring under a uniform +10 pp edit, so an effect is the fitted model applied to that exact edit.
* **Wind sectors**: canopy upwind and downwind (300 m and 1 km, 90° wide), for the signature test.
* **Geography**: albedo, elevation and distance to water, with piecewise-linear terms (water effects
  saturate within a few hundred metres of the shore).
* A smooth spatial basis (waves of 2 km and longer) and 1 km spatial blocks for clustered inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RADII = (100.0, 300.0, 1000.0)          # footprint radii (m) next to the cell itself ("_0")
RINGS = (100.0, 300.0, 1000.0)          # outer edges of the distance rings (m)
KNOTS = (15.0, 40.0)                    # canopy knots (%) of the piecewise-linear response
FLEX = ("0", "r100")                    # where canopy gets the flexible response; linear beyond (a wide
                                        # ring averages many cells, and three terms there cost precision)
SECTOR_M = (300.0, 1000.0)              # reach of the upwind / downwind canopy sectors
SECTOR_DEG = 90.0                       # their opening angle
WATER_KNOTS = (100.0, 300.0, 1000.0)    # distance-to-water knots (m)


def focal(values: np.ndarray, grid, radius_m: float) -> np.ndarray:
    """Gaussian neighbourhood mean (σ = radius / 2) over the valid cells, at every point."""
    from sparc.core import operators as ops

    r = grid.rasterize(np.asarray(values, float))
    m = np.isfinite(r)
    return grid.sample(ops.masked_gaussian(r, m, max(radius_m / 2.0 / grid.dx, 0.5)))


def _kernel_mean(values: np.ndarray, grid, k: np.ndarray) -> np.ndarray:
    """Mean of ``values`` over the cells the 0/1 kernel ``k`` (centred, offsets in grid axes) picks
    around every cell, valid cells only; cells with no valid neighbour get the field mean."""
    from scipy.signal import fftconvolve

    r = grid.rasterize(np.asarray(values, float))
    m = np.isfinite(r)
    kf = k[::-1, ::-1].astype(float)                  # correlate: value at (cell + offset) weighted by k
    num = fftconvolve(np.where(m, r, 0.0), kf, mode="same")
    den = fftconvolve(m.astype(float), kf, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
    v = grid.sample(out)
    return np.where(np.isfinite(v), v, float(np.nanmean(values)))


def _offsets(grid, reach_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = max(int(math.ceil(reach_m / grid.dx)), 1)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    return xx, yy, np.hypot(xx, yy) * grid.dx


def ring_mean(values: np.ndarray, grid, inner_m: float, outer_m: float) -> np.ndarray:
    """Mean over the ring inner < distance ≤ outer around every cell (the cell itself excluded)."""
    xx, yy, d = _offsets(grid, outer_m)
    return _kernel_mean(values, grid, (d > max(inner_m, 0.0)) & (d <= outer_m) & (d > 0))


def sector_mean(values: np.ndarray, grid, toward: tuple[float, float], radius_m: float = 300.0,
                width_deg: float = SECTOR_DEG) -> np.ndarray:
    """Mean of ``values`` over the sector reaching ``radius_m`` from each cell in direction ``toward``
    (a unit vector in grid axes: +x east, +y north), the cell itself excluded."""
    xx, yy, d = _offsets(grid, radius_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (xx * toward[0] + yy * toward[1]) / np.maximum(np.hypot(xx, yy), 1e-9)
    k = (d <= radius_m) & (d > 0) & (cosang >= math.cos(math.radians(width_deg / 2.0)))
    return _kernel_mean(values, grid, k)


def canopy_basis(c: np.ndarray, knots=KNOTS) -> list[np.ndarray]:
    """Piecewise-linear basis of canopy (%): c, (c − k)+ for every knot."""
    c = np.clip(np.asarray(c, float), 0.0, 100.0)
    return [c] + [np.maximum(c - k, 0.0) for k in knots]


def spatial_basis(grid, scale_m: float = 2000.0, max_terms: int = 120) -> np.ndarray:
    """A smooth 2-D cosine basis (no constant) of every wave whose wavelength is at least ``scale_m``: it
    absorbs confounding structure at the scale of neighbourhoods and larger, never at the scale of a street."""
    lx = max(float(grid.ix.max() - grid.ix.min()) * grid.dx, grid.dx)
    ly = max(float(grid.iy.max() - grid.iy.min()) * grid.dx, grid.dx)
    x = (grid.ix - grid.ix.min()) * grid.dx / lx
    y = (grid.iy - grid.iy.min()) * grid.dx / ly
    kx, ky = int(2 * lx / scale_m), int(2 * ly / scale_m)
    waves = [(p, q) for p in range(kx + 1) for q in range(ky + 1)
             if (p, q) != (0, 0) and (p * scale_m / (2 * lx)) ** 2 + (q * scale_m / (2 * ly)) ** 2 <= 1.0]
    waves.sort(key=lambda pq: (pq[0] / lx) ** 2 + (pq[1] / ly) ** 2)          # longest waves first
    cols = [np.cos(math.pi * p * x) * np.cos(math.pi * q * y) for p, q in waves[:max_terms]]
    return np.column_stack(cols) if cols else np.zeros((grid.ix.size, 0))


def block_ids(grid, block_m: float = 1000.0) -> np.ndarray:
    """Spatial block of every point (squares of ``block_m``): the clusters of the robust standard errors."""
    k = max(int(round(block_m / grid.dx)), 1)
    bx, by = grid.ix // k, grid.iy // k
    return (by * (int(bx.max()) + 1) + bx).astype(np.int64)


def ring_names(prefix: str, rings=RINGS) -> list[str]:
    """Own cell first, then the rings outward: ``prefix_0``, ``prefix_r100``, …"""
    return [f"{prefix}_0"] + [f"{prefix}_r{int(r)}" for r in rings]


@dataclass
class DesignLayers:
    """Every per-cell layer the designs use (row order of the layout)."""

    names: list[str]
    values: dict[str, np.ndarray]
    basis: np.ndarray
    blocks: np.ndarray
    wind: tuple[float, float] | None
    meta: dict = field(default_factory=dict)

    def mat(self, names) -> np.ndarray:
        return np.column_stack([self.values[n] for n in names])

    def has(self, names) -> list[str]:
        return [n for n in names if n in self.values]

    def canopy_terms(self, reach: float | None = None) -> list[str]:
        """Canopy terms of the cell and of every ring out to ``reach`` m (all rings when None)."""
        nk = len(self.meta.get("knots", KNOTS)) + 1
        flex = self.meta.get("flex", FLEX)
        rings = [r for r in self.meta.get("rings", RINGS) if reach is None or r <= reach]
        return [f"cb{k}_{p}" for p in ["0"] + [f"r{int(r)}" for r in rings] for k in range(nk if p in flex else 1)]

    def geography(self) -> list[str]:
        return self.has(["albedo", "elevation", "water_distance"]
                        + [f"water_{int(k)}" for k in self.meta.get("water_knots", WATER_KNOTS)])


def wind_vector(cfg) -> tuple[float, float] | None:
    """Unit vector the air moves toward (grid axes), from the config's physics wind (the forcing file)."""
    w = ((cfg.raw.get("physics") or {}).get("wind") or [0.0, 0.0])[:2]
    n = math.hypot(*w)
    return (w[0] / n, w[1] / n) if n > 0 else None


def add_canopy_rings(vals: dict, canopy: np.ndarray, grid, rings=RINGS, knots=KNOTS, dose: float = 10.0) -> None:
    """``cb{k}_{ring}`` (basis term k averaged over the ring) and ``edit_cb{k}_{ring}`` (its change under a
    uniform +``dose`` pp edit, clipped at 100%)."""
    c0 = np.clip(np.asarray(canopy, float), 0.0, 100.0)
    b0 = canopy_basis(c0, knots)
    b1 = canopy_basis(np.minimum(c0 + dose, 100.0), knots)
    edges = [0.0] + list(rings)
    for k, (a, b) in enumerate(zip(b0, b1)):
        vals[f"cb{k}_0"] = a
        vals[f"edit_cb{k}_0"] = b - a
        for lo, hi in zip(edges[:-1], edges[1:]):
            vals[f"cb{k}_r{int(hi)}"] = ring_mean(a, grid, lo, hi)
            vals[f"edit_cb{k}_r{int(hi)}"] = ring_mean(b - a, grid, lo, hi)


def build_layers(layout, radii=RADII, rings=RINGS, knots=KNOTS, basis_scale_m: float = 2000.0,
                 block_m: float = 1000.0, dose: float = 10.0) -> DesignLayers:
    g = layout.grid
    vals: dict[str, np.ndarray] = {}
    edges = [0.0] + list(rings)
    for role in ("canopy", "impervious"):
        v = layout.col(role)
        vals[f"{role}_0"] = v
        for r in radii:
            vals[f"{role}_{int(r)}"] = focal(v, g, r)
    imp = layout.col("impervious")
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals[f"imp_r{int(hi)}"] = ring_mean(imp, g, lo, hi)
    add_canopy_rings(vals, layout.col("canopy"), g, rings, knots, dose)
    for role in ("albedo", "elevation", "water_distance"):
        v = layout.col(role)
        if v is not None:
            vals[role] = v
    if "water_distance" in vals:
        for k in WATER_KNOTS:
            vals[f"water_{int(k)}"] = np.maximum(vals["water_distance"] - k, 0.0)
    wind = wind_vector(layout.cfg)
    if wind is not None:
        c = layout.col("canopy")
        for R in SECTOR_M:
            vals[f"canopy_up{int(R)}"] = sector_mean(c, g, (-wind[0], -wind[1]), R)
            vals[f"canopy_down{int(R)}"] = sector_mean(c, g, wind, R)
    return DesignLayers(names=list(vals), values=vals, basis=spatial_basis(g, basis_scale_m), blocks=block_ids(g, block_m),
                        wind=wind, meta={"radii": list(radii), "rings": list(rings), "knots": list(knots), "flex": list(FLEX),
                                         "water_knots": list(WATER_KNOTS), "basis_scale_m": basis_scale_m,
                                         "block_m": block_m, "dose": dose})
