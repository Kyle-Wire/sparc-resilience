"""The run grid on the wire (SPEC §6.2–6.3, api.md §6.2): geometry, ids, lon/lat, zones and ``GridMeta``.

The grid is ``sparc.core.grid.Grid.from_points(x_m, y_m, cell=cell_m)`` over the run's points in row order
(the order of ``predictions.parquet`` / ``data.ids``).  Coordinates come from ``predictions.parquet``
(``x_m``, ``y_m``, plus ``id`` and ``zone``) once S2_S3 has written it, else from the S0 data rebuilt from
the launch snapshot.  ``cell_m`` comes from ``run_state.meta.cell_m``, ``manifest.qa.cell_m`` or
``influence.json`` ``cell_m``.

Conventions (api.md §1, §6.2):

* ``x0_m`` / ``y0_m``: run-frame metres of the centre of cell ``(ix=0, iy=0)``; ``iy`` grows north.
* ``corners``: ``[lat, lon]`` of the corner cell centres ``sw = (0, 0)``, ``se = (nx−1, 0)``,
  ``nw = (0, ny−1)``, ``ne = (nx−1, ny−1)`` (clients interpolate bilinearly between them).
* ``zones``: the distinct zone codes; ``grid.bin``'s ``zone`` (int16) indexes this list, ``-1`` = no zone.
* lon/lat: pyproj ``crs → EPSG:4326`` on ``x_m / coord_scale`` (``reproject_to`` when set, scale 1);
  NaN without a CRS.

The result is cached in ``<studio_dir>/cache/grid.npz`` (``ix``, ``iy`` int32, ``ids``, ``lon``, ``lat``
float32, ``zone`` int16 plus the zone list and the frame) keyed by ``manifest.created_utc`` or the
``run_state`` start time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["RunGrid", "build_grid", "load_cached", "save_cached", "grid_meta", "pack_arrays", "grid_bin",
           "lonlat_transformer", "to_run_xy"]

GRID_CACHE = "cache/grid.npz"
_CACHE_VERSION = 2


@dataclass
class RunGrid:
    """Row-order geometry of a run."""

    ix: np.ndarray                 # int32 (n,)
    iy: np.ndarray                 # int32 (n,)
    x0: float
    y0: float
    dx: float
    nx: int
    ny: int
    ids: np.ndarray                # int64 or str (n,)
    lon: np.ndarray                # float32 (n,), NaN without a CRS
    lat: np.ndarray                # float32 (n,)
    zone: np.ndarray               # int16 (n,), index into ``zones``; -1 = none
    zones: list = field(default_factory=list)
    crs: str | None = None         # the CRS of the run frame (``reproject_to`` or ``crs``)
    coord_scale: float = 1.0       # run metres per CRS unit
    key: str = ""

    @property
    def n(self) -> int:
        return int(self.ix.size)

    @property
    def x(self) -> np.ndarray:
        """Cell-centre eastings (run metres)."""
        return self.x0 + self.ix.astype(np.float64) * self.dx

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.iy.astype(np.float64) * self.dx

    @property
    def has_lonlat(self) -> bool:
        return self.crs is not None and bool(np.isfinite(self.lon).any())

    @property
    def ids_kind(self) -> str:
        return "int" if np.issubdtype(np.asarray(self.ids).dtype, np.integer) else "str"

    @property
    def zone_codes(self) -> np.ndarray | None:
        """Zone code per row (object array), or None without zones."""
        if not self.zones:
            return None
        table = np.array(list(self.zones) + [None], dtype=object)
        return table[np.where(self.zone >= 0, self.zone, len(self.zones))]

    def cell_index(self) -> np.ndarray:
        return self.iy.astype(np.int64) * self.nx + self.ix.astype(np.int64)

    def raster(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """(ny, nx) raster of per-row values (mean of collisions)."""
        v = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(v)
        s = np.zeros(self.ny * self.nx)
        c = np.zeros(self.ny * self.nx)
        idx = self.cell_index()
        np.add.at(s, idx[ok], v[ok])
        np.add.at(c, idx[ok], 1.0)
        out = np.full(self.ny * self.nx, fill)
        m = c > 0
        out[m] = s[m] / c[m]
        return out.reshape(self.ny, self.nx)

    def mask_raster(self) -> np.ndarray:
        m = np.zeros(self.ny * self.nx, dtype=bool)
        m[self.cell_index()] = True
        return m.reshape(self.ny, self.nx)

    def core_grid(self):
        """The equivalent ``sparc.core.grid.Grid`` (for ``planner.export_geotiffs`` and the correlogram)."""
        from sparc.core.grid import Grid

        counts = np.zeros((self.ny, self.nx), dtype=np.int64)
        np.add.at(counts, (self.iy, self.ix), 1)
        return Grid(x0=float(self.x0), y0=float(self.y0), dx=float(self.dx), dy=float(self.dx), nx=int(self.nx),
                    ny=int(self.ny), ix=self.ix.astype(np.int64), iy=self.iy.astype(np.int64), mask=counts > 0,
                    counts=counts)

    @property
    def nbytes(self) -> int:
        return int(sum(getattr(a, "nbytes", 0) for a in (self.ix, self.iy, self.ids, self.lon, self.lat, self.zone)))


# ---------------------------------------------------------------------------
# lon/lat
# ---------------------------------------------------------------------------

def run_crs(cfg_raw: dict | None) -> tuple[str | None, float]:
    """``(crs, coord_scale)`` of a run frame: ``reproject_to`` with scale 1, else ``crs`` with the coord unit."""
    from sparc.core.config import UNIT_TO_METRES

    d = (cfg_raw or {}).get("data") or {}
    if d.get("reproject_to"):
        return str(d["reproject_to"]), 1.0
    crs = d.get("crs")
    unit = str(d.get("coord_unit") or "m").lower()
    return (str(crs) if crs else None), float(UNIT_TO_METRES.get(unit, 1.0))


def lonlat_transformer(crs: str):
    from pyproj import Transformer

    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def transform_xy(tr, x, y) -> tuple[np.ndarray, np.ndarray]:
    """``tr.transform`` of 1-d float64 arrays of any length (a one-element array goes through pyproj's scalar
    path, which NumPy deprecates for arrays)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 1:
        a, b = tr.transform(float(x[0]), float(y[0]))
        return np.array([a], dtype=np.float64), np.array([b], dtype=np.float64)
    a, b = tr.transform(x, y)
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def to_run_xy(crs: str, coord_scale: float, lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:4326 lon/lat → run-frame metres."""
    from pyproj import Transformer

    x, y = transform_xy(Transformer.from_crs("EPSG:4326", crs, always_xy=True), lon, lat)
    return x * coord_scale, y * coord_scale


def _lonlat(x_m: np.ndarray, y_m: np.ndarray, crs: str | None, scale: float) -> tuple[np.ndarray, np.ndarray]:
    n = x_m.size
    if not crs:
        return np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    try:
        lon, lat = transform_xy(lonlat_transformer(crs), x_m / scale, y_m / scale)
    except Exception:                    # an unknown CRS string: no lon/lat rather than no grid
        return np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    return np.asarray(lon, np.float32), np.asarray(lat, np.float32)


# ---------------------------------------------------------------------------
# build and cache
# ---------------------------------------------------------------------------

def _zone_index(zones_raw) -> tuple[np.ndarray, list]:
    if zones_raw is None:
        return None, []
    z = np.asarray(zones_raw, dtype=object)
    present = np.array([v is not None and not (isinstance(v, float) and not np.isfinite(v)) for v in z])
    vals = [v.item() if hasattr(v, "item") else v for v in z[present]]
    try:
        codes = sorted(set(vals))
    except TypeError:                    # mixed types: sort by their text
        codes = sorted(set(vals), key=str)
    codes = [int(c) if isinstance(c, float) and float(c).is_integer() else c for c in codes]
    lookup = {(_zkey(c)): i for i, c in enumerate(codes)}
    out = np.full(z.size, -1, dtype=np.int16)
    if len(codes) > np.iinfo(np.int16).max:
        return out, []
    for i, (ok, v) in enumerate(zip(present, z)):
        if ok:
            out[i] = lookup[_zkey(v.item() if hasattr(v, "item") else v)]
    return out, codes


def _zkey(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return ("n", int(v))
    if isinstance(v, (float, np.floating)):
        return ("n", float(v))
    return ("s", str(v))


def build_grid(x_m, y_m, cell_m: float | None, ids, zones_raw=None, cfg_raw: dict | None = None,
               key: str = "") -> RunGrid:
    """A :class:`RunGrid` from row-order coordinates (``Grid.from_points`` with the run's cell size)."""
    from sparc.core.grid import Grid

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    g = Grid.from_points(x, y, cell=float(cell_m) if cell_m else None)
    crs, scale = run_crs(cfg_raw)
    lon, lat = _lonlat(x, y, crs, scale)
    zone, codes = _zone_index(zones_raw)
    if zone is None:
        zone = np.full(x.size, -1, dtype=np.int16)
    ids = np.asarray(ids)
    if ids.dtype == object:
        try:
            ids = ids.astype(np.int64)
        except (TypeError, ValueError):
            ids = ids.astype(str)
    elif np.issubdtype(ids.dtype, np.floating) and np.all(np.isfinite(ids)) and np.all(ids == np.round(ids)):
        ids = ids.astype(np.int64)
    elif np.issubdtype(ids.dtype, np.integer):
        ids = ids.astype(np.int64)
    return RunGrid(ix=g.ix.astype(np.int32), iy=g.iy.astype(np.int32), x0=float(g.x0), y0=float(g.y0),
                   dx=float(g.dx), nx=int(g.nx), ny=int(g.ny), ids=ids, lon=lon, lat=lat, zone=zone, zones=codes,
                   crs=crs, coord_scale=scale, key=key)


def load_cached(studio_dir: str | os.PathLike | None, key: str) -> RunGrid | None:
    """The cached grid when its key matches, else None."""
    if not studio_dir:
        return None
    path = Path(studio_dir) / GRID_CACHE
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            if meta.get("key") != key or meta.get("version") != _CACHE_VERSION:
                return None
            return RunGrid(ix=z["ix"], iy=z["iy"], x0=meta["x0"], y0=meta["y0"], dx=meta["dx"], nx=meta["nx"],
                           ny=meta["ny"], ids=z["ids"], lon=z["lon"], lat=z["lat"], zone=z["zone"],
                           zones=meta.get("zones") or [], crs=meta.get("crs"),
                           coord_scale=float(meta.get("coord_scale") or 1.0), key=key)
    except (OSError, KeyError, ValueError):
        return None


def save_cached(studio_dir: str | os.PathLike | None, grid: RunGrid) -> None:
    """Write ``<studio_dir>/cache/grid.npz`` atomically (best effort: a read-only folder only loses the cache)."""
    if not studio_dir:
        return
    from sparc.core import runio

    meta = {"version": _CACHE_VERSION, "key": grid.key, "x0": grid.x0, "y0": grid.y0, "dx": grid.dx,
            "nx": grid.nx, "ny": grid.ny, "zones": grid.zones, "crs": grid.crs, "coord_scale": grid.coord_scale}
    try:
        runio.write_npz_atomic(Path(studio_dir) / GRID_CACHE, ix=grid.ix, iy=grid.iy, ids=grid.ids, lon=grid.lon,
                               lat=grid.lat, zone=grid.zone,
                               meta=np.frombuffer(json.dumps(meta, default=str).encode("utf-8"), dtype=np.uint8))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# wire formats
# ---------------------------------------------------------------------------

def grid_etag(run_id: str, grid: RunGrid) -> str:
    return '"' + hashlib.sha1(f"{run_id}|grid|{grid.key}|{grid.n}|{grid.nx}x{grid.ny}".encode()).hexdigest() + '"'


def grid_meta(run_id: str, grid: RunGrid, *, n_folds: int | None, target_units: str,
              background: float | None) -> dict:
    """``GridMeta`` (api.md §1)."""
    has_ll = grid.has_lonlat
    corners = bounds = None
    if has_ll:
        x1 = grid.x0 + (grid.nx - 1) * grid.dx
        y1 = grid.y0 + (grid.ny - 1) * grid.dx
        xs = np.array([grid.x0, x1, grid.x0, x1])
        ys = np.array([grid.y0, grid.y0, y1, y1])
        lon, lat = transform_xy(lonlat_transformer(grid.crs), xs / grid.coord_scale, ys / grid.coord_scale)
        corners = {k: [float(lat[i]), float(lon[i])] for i, k in enumerate(("sw", "se", "nw", "ne"))}
        ok = np.isfinite(grid.lon) & np.isfinite(grid.lat)
        bounds = [float(np.min(grid.lon[ok])), float(np.min(grid.lat[ok])), float(np.max(grid.lon[ok])),
                  float(np.max(grid.lat[ok]))]
    from sparc.core.catalog import unit_label

    return {
        "n": grid.n, "nx": grid.nx, "ny": grid.ny, "dx_m": grid.dx, "x0_m": grid.x0, "y0_m": grid.y0,
        "crs": grid.crs, "coord_scale": grid.coord_scale, "has_lonlat": has_ll, "bounds_lonlat": bounds,
        "corners": corners, "ids_kind": grid.ids_kind, "zones": list(grid.zones), "n_folds": n_folds,
        "units": {"target": unit_label(target_units) or target_units}, "background": background,
        "etag": grid_etag(run_id, grid).strip('"'),
    }


def pack_arrays(arrays: list[tuple[str, np.ndarray]]) -> tuple[bytes, list[dict]]:
    """Concatenate little-endian arrays, each 8-byte aligned (api.md §0.4); returns ``(body, offsets)``."""
    names = {np.dtype("int32"): "int32", np.dtype("float32"): "float32", np.dtype("int16"): "int16",
             np.dtype("uint8"): "uint8", np.dtype("int64"): "int64", np.dtype("float64"): "float64"}
    parts = []
    offsets = []
    pos = 0
    for name, arr in arrays:
        a = np.ascontiguousarray(arr)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        raw = a.tobytes()
        offsets.append({"name": name, "dtype": names.get(np.dtype(a.dtype.str.replace(">", "<")), str(a.dtype)),
                        "offset": pos, "length": int(a.size)})
        parts.append(raw)
        pos += len(raw)
        pad = (-pos) % 8
        if pad:
            parts.append(b"\0" * pad)
            pos += pad
    return b"".join(parts), offsets


def grid_bin(grid: RunGrid) -> tuple[bytes, list[dict]]:
    """Packed ``ix:i32, iy:i32, lon:f32, lat:f32, zone:i16``."""
    return pack_arrays([("ix", grid.ix.astype(np.int32)), ("iy", grid.iy.astype(np.int32)),
                        ("lon", grid.lon.astype(np.float32)), ("lat", grid.lat.astype(np.float32)),
                        ("zone", grid.zone.astype(np.int16))])
