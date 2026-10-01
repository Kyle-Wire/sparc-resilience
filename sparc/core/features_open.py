"""Any-city predictors from open data (no local GIS needed).

Builds the six predictors the core pipeline uses from public AWS data, on
any study grid:

=====================  ===========================================================
canopy (%)             ESA WorldCover 2021 tree-cover fraction × 100 (10 m)
impervious (%)         ESA WorldCover 2021 built-up fraction × 100
ndvi                   Sentinel-2 L2A summer median (B08 − B04)/(B08 + B04),
                       cloud/shadow/cirrus masked with the scene classification
albedo                 Sentinel-2 broadband albedo, Bonafoni & Sekertekin (2020):
                       0.2453·B2 + 0.0508·B3 + 0.1804·B4 + 0.3081·B8 + 0.1332·B11
                       + 0.0521·B12 + 0.0011 (surface reflectance; B8 vs B8A to verify)
elevation (m)          Copernicus DEM GLO-30
water distance (m)     distance to the nearest WorldCover water pixel
=====================  ===========================================================

Sentinel-2 scenes are found by listing ``s3://sentinel-cogs`` for the MGRS
tiles covering the area (the STAC API is not required), filtered by the
scene's cloud cover, and read as COG windows warped onto one 10 m UTM grid.
:func:`compare_features` scores each open layer against the city's own
layers (Pearson r, linear R², Spearman ρ), the check before trusting a city
built from open data alone.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

from sparc.core.opendata import (
    WC_CLASSES,
    WORLDCOVER,
    _bounds,
    gdal_env,
    points_lonlat,
    raster_to_points,
    read_window,
    worldcover_tiles,
)

log = logging.getLogger(__name__)

S2 = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
ALBEDO_COEF = {"B02": 0.2453, "B03": 0.0508, "B04": 0.1804, "B08": 0.3081, "B11": 0.1332, "B12": 0.0521}
ALBEDO_OFFSET = 0.0011
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)          # no data, saturated, cloud shadow, cloud (med/high), cirrus, snow


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def mgrs_tiles(bounds, zone: int) -> list[str]:
    """MGRS 100 km tile ids (zone + band + square) covering lon/lat ``bounds``."""
    try:
        import mgrs  # optional dependency
        m = mgrs.MGRS()
        out = set()
        w, s, e, n = bounds
        for lo in np.linspace(w, e, 5):
            for la in np.linspace(s, n, 5):
                out.add(m.toMGRS(la, lo, MGRSPrecision=0))
        return sorted(out)
    except ImportError:
        return []


def _s2_tile_prefixes(bounds, tiles: list[str] | None) -> list[str]:
    if tiles:
        return tiles
    t = mgrs_tiles(bounds, 0)
    if not t:
        raise ValueError("pass features.s2_tiles (e.g. ['19TCG', '19TBG']) or install the 'mgrs' package")
    return t


def s2_scenes(tiles: list[str], months: list[tuple[int, int]], max_cloud: float = 20.0,
              fetch=None) -> list[dict]:
    """Scenes (one per tile/date, lowest processing number) under ``max_cloud`` %."""
    from sparc.core.climate import http_fetch
    from sparc.core.forcing import s3_keys

    fetch = fetch or http_fetch
    out = []
    for tile in tiles:
        z, band, sq = tile[:2], tile[2], tile[3:]
        for y, mth in months:
            prefix = f"sentinel-s2-l2a-cogs/{int(z)}/{band}/{sq}/{y}/{mth}/"
            names = sorted({k.split("/")[6] for k in s3_keys(S2, prefix, fetch)})
            seen = set()
            for nm in names:
                date = nm.split("_")[2]
                if (tile, date) in seen:
                    continue
                meta = json.loads(fetch(S2 + prefix + f"{nm}/{nm}.json") or b"{}")
                cc = (meta.get("properties") or {}).get("eo:cloud_cover")
                if cc is not None and cc <= max_cloud:
                    out.append({"tile": tile, "name": nm, "date": date, "cloud": cc, "prefix": prefix + nm + "/"})
                    seen.add((tile, date))
    return out


def s2_composite(scenes: list[dict], bounds_lonlat, res: float = 10.0) -> dict:
    """Median composite of the S2 bands on one UTM grid (cloud-masked)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    w, s, e, n = bounds_lonlat
    epsg = utm_epsg(0.5 * (w + e), 0.5 * (s + n))
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([w, e, w, e], [s, s, n, n])
    x0, x1 = math.floor(min(xs) / res) * res, math.ceil(max(xs) / res) * res
    y0, y1 = math.floor(min(ys) / res) * res, math.ceil(max(ys) / res) * res
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    T = from_origin(x0, y1, res, res)
    stack = {b: [] for b in S2_BANDS}
    used = []
    with gdal_env():
        for sc in scenes:
            def warp(band, resampling):
                with rasterio.open("/vsicurl/" + S2 + sc["prefix"] + f"{band}.tif") as src, \
                        WarpedVRT(src, crs=f"EPSG:{epsg}", transform=T, width=W, height=H,
                                  resampling=resampling, nodata=0) as vrt:
                    return vrt.read(1).astype("float32")
            try:
                scl = warp("SCL", Resampling.nearest)
                bad = np.isin(scl, SCL_BAD)
                if bad.mean() > 0.95:
                    continue
                for b in S2_BANDS:
                    a = warp(b, Resampling.bilinear) / 10000.0
                    a[bad | (a <= 0)] = np.nan
                    stack[b].append(a)
                used.append(sc["name"])
                log.info("s2: %s (%.1f%% cloud) — %.0f%% usable", sc["name"], sc["cloud"], 100 * (1 - bad.mean()))
            except Exception as exc:                          # noqa: BLE001 - skip a broken scene
                log.warning("s2: skipping %s (%s)", sc["name"], exc)
    if not used:
        raise RuntimeError("no usable Sentinel-2 scenes")
    with np.errstate(all="ignore"):
        med = {b: np.nanmedian(np.stack(v), axis=0) for b, v in stack.items()}
    ndvi = (med["B08"] - med["B04"]) / (med["B08"] + med["B04"])
    albedo = sum(ALBEDO_COEF[b] * med[b] for b in S2_BANDS) + ALBEDO_OFFSET
    return {"ndvi": ndvi, "albedo": albedo, "transform": T, "crs": f"EPSG:{epsg}", "scenes": used}


def dem_tiles(bounds) -> list[str]:
    w, s, e, n = bounds
    out = []
    for la in range(int(math.floor(s)), int(math.floor(n)) + 1):
        for lo in range(int(math.floor(w)), int(math.floor(e)) + 1):
            out.append(f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
                       f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")
    return out


def build_open_features(data, cfg, months=((2020, 6), (2020, 7), (2020, 8)), max_cloud: float = 20.0,
                        s2_tiles: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """The six predictors at every point from open data, plus provenance."""
    from scipy import ndimage

    lon, lat = points_lonlat(data, cfg)
    b = _bounds(lon, lat)
    out = {"id": data.ids}
    prov: dict = {}
    # WorldCover: tree / built fractions and water distance
    tiles = worldcover_tiles(b)
    if len(tiles) != 1:
        raise NotImplementedError(f"study area spans several WorldCover tiles {tiles}")
    wc, tr, _ = read_window(WORLDCOVER.format(tile=tiles[0]), _bounds(lon, lat, margin=0.02))
    fr = raster_to_points(wc, tr, data, cfg, mode="fractions", classes=WC_CLASSES, sub=1, nodata=0)
    out["canopy"] = 100.0 * np.nan_to_num(fr["tree"])
    out["impervious"] = 100.0 * np.nan_to_num(fr["built"])
    water = wc == 80
    px_m = abs(tr.a) * 111320.0 * math.cos(math.radians(float(np.mean(lat))))
    py_m = abs(tr.e) * 111320.0
    dist = ndimage.distance_transform_edt(~water, sampling=(py_m, px_m)).astype("float32")
    out["water_distance"] = raster_to_points(dist, tr, data, cfg, mode="mean", sub=1)["mean"]
    prov["worldcover"] = tiles[0]
    # Copernicus DEM
    dts = dem_tiles(b)
    if len(dts) != 1:
        raise NotImplementedError(f"study area spans several DEM tiles {dts}")
    dem, trd, nod = read_window(DEM.format(name=dts[0]), b)
    out["elevation"] = raster_to_points(dem.astype(float), trd, data, cfg, mode="mean", sub=3, nodata=nod)["mean"]
    prov["dem"] = dts[0]
    # Sentinel-2 composite
    scenes = s2_scenes(_s2_tile_prefixes(b, s2_tiles), list(months), max_cloud)
    comp = s2_composite(scenes, b)
    for k in ("ndvi", "albedo"):
        out[k] = raster_to_points(comp[k], comp["transform"], data, cfg, mode="mean", sub=1,
                                  src_crs=comp["crs"])["mean"]
    prov["sentinel2"] = {"scenes": comp["scenes"], "max_cloud": max_cloud, "months": [list(m) for m in months],
                         "albedo": "Bonafoni & Sekertekin 2020 narrow-to-broadband coefficients"}
    df = pd.DataFrame(out)
    for c in df.columns:
        if c != "id" and df[c].isna().any():
            df[c] = df[c].fillna(float(np.nanmedian(df[c])))
    return df, prov


def compare_features(open_df: pd.DataFrame, data, roles: dict) -> list[dict]:
    """Agreement of each open layer with the city's own layer (same role)."""
    from scipy.stats import spearmanr

    rows = []
    for role in ("canopy", "impervious", "ndvi", "albedo", "elevation", "water_distance"):
        col = roles.get(role)
        if not col or col not in data.frame or role not in open_df:
            continue
        a = data.frame[col].to_numpy(float)
        o = open_df[role].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(o)
        r = float(np.corrcoef(a[ok], o[ok])[0, 1])
        rows.append({"role": role, "city_column": col, "pearson_r": r, "r2_linear": r * r,
                     "spearman": float(spearmanr(a[ok], o[ok]).correlation),
                     "city_mean": float(np.mean(a[ok])), "open_mean": float(np.mean(o[ok]))})
    return rows
