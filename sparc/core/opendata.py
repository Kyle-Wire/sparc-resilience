"""Open raster data aggregated onto the study grid (COG windows over HTTPS).

Sources (public AWS buckets, read with GDAL ``/vsicurl/`` — only the window
over the study area is fetched):

* **HRSL** (Meta / CIESIN High Resolution Settlement Layer, ~30 m,
  ``s3://dataforgood-fb-data/hrsl-cogs``): people per pixel, total and by
  group (``elderly_60_plus``, ``children_under_five``, …).  Residential
  population (census-based, circa 2010s), not daytime presence.
* **ESA WorldCover 2021** (10 m, ``s3://esa-worldcover``): land-cover classes
  → per-cell fractions (tree, grass, built-up, bare, water, …).

:func:`raster_to_points` assigns raster pixels to grid cells by pixel centre,
after splitting each pixel into ``sub × sub`` sub-pixels so counts are
conserved and spread evenly at similar resolutions.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HRSL = "https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/{layer}/{layer}-latest.vrt"
HRSL_LAYERS = {"people": "hrsl_general", "people_60_plus": "hrsl_elderly_60_plus",
               "people_under_5": "hrsl_children_under_five"}
WORLDCOVER = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}


def gdal_env():
    import rasterio

    ca = os.environ.get("CURL_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    opts = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
            "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2", "VSI_CACHE": "TRUE"}
    if ca:
        opts["CURL_CA_BUNDLE"] = ca
    return rasterio.Env(**opts)


def points_lonlat(data, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Longitude/latitude of every point (from the config's CRS)."""
    from pyproj import Transformer

    d = cfg.data
    crs = d.get("reproject_to") or d.get("crs")
    if not crs:
        raise ValueError("open-data layers need data.crs (or reproject_to)")
    s = 1.0 if d.get("reproject_to") else cfg.coord_scale
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(data.x / s, data.y_coord / s)
    return np.asarray(lon), np.asarray(lat)


def read_window(url: str, bounds: tuple[float, float, float, float]):
    """(array, affine transform) of a lon/lat window of a COG/VRT."""
    import rasterio
    from rasterio.windows import from_bounds

    with gdal_env(), rasterio.open("/vsicurl/" + url) as src:
        w = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=w, boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
        return arr, src.window_transform(w), src.nodata


def raster_to_points(arr: np.ndarray, transform, data, cfg, mode: str = "sum", classes: dict | None = None,
                     sub: int = 3, nodata=None) -> dict[str, np.ndarray]:
    """Aggregate a lon/lat raster onto the study grid and sample at the points.

    ``mode="sum"``: total of the pixel values falling in each cell (counts);
    ``mode="fractions"``: share of each class code in ``classes`` per cell."""
    from pyproj import Transformer

    g = data.grid
    d = cfg.data
    crs = d.get("reproject_to") or d.get("crs")
    s = 1.0 if d.get("reproject_to") else cfg.coord_scale
    ny, nx = arr.shape
    k = max(int(sub), 1)
    jj, ii = np.meshgrid((np.arange(nx * k) + 0.5) / k, (np.arange(ny * k) + 0.5) / k)
    lon = transform.c + jj * transform.a + ii * transform.b
    lat = transform.f + jj * transform.d + ii * transform.e
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lon.ravel(), lat.ravel())
    ix = np.rint((np.asarray(x) * s - g.x0) / g.dx).astype(np.int64)
    iy = np.rint((np.asarray(y) * s - g.y0) / g.dy).astype(np.int64)
    vals = np.repeat(np.repeat(arr, k, axis=0), k, axis=1).ravel().astype(float)
    ok = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
    if nodata is not None and np.isfinite(nodata):
        ok &= vals != nodata
    ok &= np.isfinite(vals)
    flat = iy[ok] * g.nx + ix[ok]
    v = vals[ok]
    cell = g.iy * g.nx + g.ix
    size = g.nx * g.ny
    if mode == "sum":
        tot = np.bincount(flat, weights=v / (k * k), minlength=size)
        return {"sum": tot[cell]}
    if mode == "fractions":
        n = np.bincount(flat, minlength=size).astype(float)
        out = {}
        for code, name in (classes or {}).items():
            c = np.bincount(flat, weights=(v == code).astype(float), minlength=size)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name] = np.where(n > 0, c / n, np.nan)[cell]
        return out
    raise ValueError(f"unknown mode {mode!r}")


def _bounds(lon, lat, margin: float = 0.003) -> tuple[float, float, float, float]:
    return (float(lon.min()) - margin, float(lat.min()) - margin, float(lon.max()) + margin, float(lat.max()) + margin)


def worldcover_tiles(bounds) -> list[str]:
    w, s, e, n = bounds
    out = []
    for la in range(int(math.floor(s / 3.0) * 3), int(math.floor(n / 3.0) * 3) + 1, 3):
        for lo in range(int(math.floor(w / 3.0) * 3), int(math.floor(e / 3.0) * 3) + 1, 3):
            out.append(f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
    return out


def fetch_layers(data, cfg, hrsl=HRSL_LAYERS, worldcover: bool = True) -> pd.DataFrame:
    """People (HRSL groups) and WorldCover fractions at every point."""
    lon, lat = points_lonlat(data, cfg)
    b = _bounds(lon, lat)
    out = {"id": data.ids, "x_m": data.x, "y_m": data.y_coord}
    for name, layer in (hrsl or {}).items():
        arr, tr, nod = read_window(HRSL.format(layer=layer), b)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        out[name] = raster_to_points(arr, tr, data, cfg, mode="sum", sub=3)["sum"]
        log.info("layers: %s total %.0f", name, out[name].sum())
    if worldcover:
        tiles = worldcover_tiles(b)
        if len(tiles) != 1:
            raise NotImplementedError(f"study area spans several WorldCover tiles {tiles}")
        arr, tr, nod = read_window(WORLDCOVER.format(tile=tiles[0]), b)
        fr = raster_to_points(arr, tr, data, cfg, mode="fractions", classes=WC_CLASSES, sub=1, nodata=0)
        for k, v in fr.items():
            out[f"lc_{k}"] = v
        log.info("layers: WorldCover %s, mean built-up %.2f, tree %.2f", tiles[0], np.nanmean(fr["built"]),
                 np.nanmean(fr["tree"]))
    return pd.DataFrame(out)


def load_layers(cfg, data) -> pd.DataFrame | None:
    """The config's precomputed layers table (``planner.layers``) aligned with
    ``data`` (re-aggregated for coarse runs: people summed, fractions averaged)."""
    path = (cfg.raw.get("planner") or {}).get("layers")
    if not path:
        return None
    p = cfg.resolve_path(path)
    if not Path(p).exists():
        log.warning("planner layers %s not found (run `sparc core layers`)", p)
        return None
    lay = pd.read_parquet(p)
    co = (data.qa or {}).get("coarse")
    if co:
        from sparc.core.data import _fine_to_coarse

        inv, members, _, _ = _fine_to_coarse(lay["x_m"].to_numpy(float), lay["y_m"].to_numpy(float),
                                             float(co["fine_cell_m"]), float(co["cell_m"]))
        out = {"id": data.ids}
        for c in lay.columns:
            if c in ("id", "x_m", "y_m"):
                continue
            v = np.nan_to_num(lay[c].to_numpy(float))
            agg = np.bincount(inv, weights=v, minlength=members.size)
            out[c] = agg if c.startswith("people") else agg / members
        lay = pd.DataFrame(out)
        if len(lay) != data.n:
            raise ValueError("coarse layers do not match the run's cells")
        return lay
    lay = lay.set_index("id").reindex(data.ids)
    if lay.isna().all(axis=1).any():
        log.warning("planner layers: %d points have no layer values", int(lay.isna().all(axis=1).sum()))
    return lay.reset_index()
