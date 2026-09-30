"""Regular raster support for point data.

Most UHI inputs (CAPA rasters sampled at fishnet centroids, brown4.csv) are
points on a regular lattice.  :class:`Grid` recovers that lattice (cell size
and origin) from the coordinates so every point maps to a raster cell, which
lets the core use FFT convolutions and finite-difference operators.

For genuinely irregular data the same mapping bins points into cells
(collisions are averaged).  Coordinates must already be in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def estimate_lattice_spacing(x: np.ndarray, y: np.ndarray, sample: int = 20000, seed: int = 0) -> float:
    """Median nearest-neighbour distance (metres) — the lattice spacing for gridded data."""
    from scipy.spatial import cKDTree

    pts = np.column_stack([x, y]).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=min(sample, len(pts)), replace=False)
    tree = cKDTree(pts)
    d, _ = tree.query(pts[idx], k=2)
    nn = d[:, 1]
    nn = nn[nn > 1e-9]
    if nn.size == 0:
        raise ValueError("All points coincide; cannot infer a grid spacing.")
    return float(np.median(nn))


def _lattice_offset(v: np.ndarray, cell: float) -> float:
    """Circular mean of ``v mod cell`` — the lattice phase along one axis."""
    ang = 2.0 * np.pi * np.mod(v, cell) / cell
    phase = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float(np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi) * cell)


@dataclass
class Grid:
    """Point ↔ raster mapping.

    Cell (iy, ix) is centred at (x0 + ix·dx, y0 + iy·dy).  Row index grows
    with y (no image flip), so axis 1 is x and axis 0 is y everywhere in the
    core.
    """

    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    ix: np.ndarray
    iy: np.ndarray
    mask: np.ndarray = field(repr=False)
    counts: np.ndarray = field(repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_points(cls, x: np.ndarray, y: np.ndarray, cell: float | None = None, margin: int = 1) -> "Grid":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if cell is None:
            cell = estimate_lattice_spacing(x, y)
        ox = _lattice_offset(x, cell)
        oy = _lattice_offset(y, cell)
        # Snap the origin onto the lattice phase, one margin cell below the data.
        x0 = (np.floor((x.min() - ox) / cell) - margin) * cell + ox
        y0 = (np.floor((y.min() - oy) / cell) - margin) * cell + oy
        ix = np.rint((x - x0) / cell).astype(np.int64)
        iy = np.rint((y - y0) / cell).astype(np.int64)
        nx = int(ix.max()) + 1 + margin
        ny = int(iy.max()) + 1 + margin
        counts = np.zeros((ny, nx), dtype=np.int64)
        np.add.at(counts, (iy, ix), 1)
        return cls(x0=float(x0), y0=float(y0), dx=float(cell), dy=float(cell), nx=nx, ny=ny,
                   ix=ix, iy=iy, mask=counts > 0, counts=counts)

    # --------------------------------------------------------------- queries
    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def n_points(self) -> int:
        return int(self.ix.size)

    @property
    def cell_index(self) -> np.ndarray:
        """Flat raster index of each point."""
        return self.iy * self.nx + self.ix

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + np.arange(self.nx) * self.dx
        ys = self.y0 + np.arange(self.ny) * self.dy
        return np.meshgrid(xs, ys)

    def point_coords(self) -> np.ndarray:
        """Snapped (cell-centre) coordinates of every point, shape (n, 2)."""
        return np.column_stack([self.x0 + self.ix * self.dx, self.y0 + self.iy * self.dy])

    def collision_fraction(self) -> float:
        return float((self.counts > 1).sum() / max(1, self.mask.sum()))

    # ------------------------------------------------------- raster <-> point
    def rasterize(self, values: np.ndarray, subset: np.ndarray | None = None, fill: float = np.nan) -> np.ndarray:
        """Mean of point ``values`` per cell (optionally only ``subset`` points)."""
        values = np.asarray(values, dtype=np.float64)
        sel = np.ones(self.n_points, dtype=bool) if subset is None else np.asarray(subset)
        if sel.dtype != bool:
            b = np.zeros(self.n_points, dtype=bool)
            b[sel] = True
            sel = b
        sel = sel & np.isfinite(values)
        s = np.zeros(self.shape)
        c = np.zeros(self.shape)
        np.add.at(s, (self.iy[sel], self.ix[sel]), values[sel])
        np.add.at(c, (self.iy[sel], self.ix[sel]), 1.0)
        out = np.full(self.shape, fill, dtype=np.float64)
        ok = c > 0
        out[ok] = s[ok] / c[ok]
        return out

    def raster_mask(self, subset: np.ndarray | None = None) -> np.ndarray:
        if subset is None:
            return self.mask.copy()
        return np.isfinite(self.rasterize(np.ones(self.n_points), subset=subset))

    def sample(self, raster: np.ndarray) -> np.ndarray:
        """Value of ``raster`` at every point (exact cell lookup)."""
        return np.asarray(raster)[self.iy, self.ix]

    def fill_nan(self, raster: np.ndarray, value: float | None = None) -> np.ndarray:
        """Replace NaNs (no-data cells) with ``value`` (default: mean of valid cells)."""
        r = np.asarray(raster, dtype=np.float64).copy()
        bad = ~np.isfinite(r)
        if bad.any():
            r[bad] = np.nanmean(r) if value is None else value
        return r

    # ------------------------------------------------------------- torch
    def rasterize_torch(self, values, subset_idx=None):
        """Differentiable per-cell mean of a point tensor (collisions averaged).

        Returns (raster, valid_mask) as torch tensors of shape (ny, nx).
        """
        import torch

        idx = torch.as_tensor(self.cell_index, device=values.device)
        if subset_idx is not None:
            idx = idx[subset_idx]
            values = values[subset_idx]
        flat = values.new_zeros(self.ny * self.nx)
        cnt = values.new_zeros(self.ny * self.nx)
        flat = flat.index_add(0, idx, values)
        cnt = cnt.index_add(0, idx, torch.ones_like(values))
        valid = cnt > 0
        out = torch.where(valid, flat / cnt.clamp(min=1.0), torch.zeros_like(flat))
        return out.view(self.ny, self.nx), valid.view(self.ny, self.nx)


def spatial_window_subsample(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Indices of ~``n`` points inside a centred rectangle (keeps the lattice
    contiguous, unlike random thinning which would break focal features)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if n >= len(x):
        return np.arange(len(x))
    cx, cy = np.median(x), np.median(y)
    wx, wy = x.max() - x.min(), y.max() - y.min()
    lo, hi = 0.0, 1.0
    for _ in range(40):  # bisection on the rectangle scale
        mid = 0.5 * (lo + hi)
        inside = (np.abs(x - cx) <= mid * wx / 2) & (np.abs(y - cy) <= mid * wy / 2)
        if inside.sum() < n:
            lo = mid
        else:
            hi = mid
    inside = (np.abs(x - cx) <= hi * wx / 2) & (np.abs(y - cy) <= hi * wy / 2)
    return np.flatnonzero(inside)
