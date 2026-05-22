"""
equity.py — HOLC redlining, CDC SVI, and EJScreen joins to analysis grid.

Fetches three equity layers and joins them to the 30m fishnet:
  holc_grade     — HOLC redlining grade A=4, B=3, C=2, D=1; NaN outside coverage
  cdc_svi        — CDC Social Vulnerability Index composite score (0–1)
  ejscreen_score — EPA EJScreen composite percentile (0–100, normalised to 0–1)

All three are model co-variates joined spatially to the grid.  HOLC polygons
are also returned separately as a GeoDataFrame for the desktop boundary overlay.
Layers gracefully degrade to NaN / skipped when the study area has no coverage
(e.g. non-US locations).
"""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

HTTP_TIMEOUT = 30.0

# Mapping Inequality HOLC API
HOLC_API = "https://dsl.richmond.edu/panorama/redlining/static/fullDownload.geojson"

# EPA EJScreen REST API (no key)
EJSCREEN_API = (
    "https://ejscreen.epa.gov/mapper/ejscreenRESTbroker.aspx"
)

# CDC SVI 2022 national CSV download
CDC_SVI_URL = (
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI2022_US.csv"
)

HOLC_GRADE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_equity(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Optional[Path] = None,
) -> tuple[object, Optional[object]]:
    """Join HOLC, CDC SVI, and EJScreen equity layers onto the analysis fishnet.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.  Returns with three new columns.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.

    Returns
    -------
    (gdf, holc_gdf)
        ``gdf`` — fishnet with equity columns appended.
        ``holc_gdf`` — HOLC polygon GeoDataFrame for map overlay (or None).
    """
    cache_dir = cache_dir or _default_cache_dir()

    gdf = fishnet_gdf
    holc_overlay: Optional[object] = None

    # --- HOLC ---
    try:
        holc_gdf = _fetch_holc(bbox, cache_dir)
        if holc_gdf is not None and not holc_gdf.empty:  # type: ignore[union-attr]
            gdf = _join_holc(gdf, holc_gdf)
            holc_overlay = holc_gdf
        else:
            gdf = _add_null_column(gdf, "holc_grade")
    except Exception:
        gdf = _add_null_column(gdf, "holc_grade")

    # --- CDC SVI ---
    try:
        gdf = _join_cdc_svi(gdf, bbox, cache_dir)
    except Exception:
        gdf = _add_null_column(gdf, "cdc_svi")

    # --- EJScreen ---
    try:
        gdf = _join_ejscreen(gdf, bbox)
    except Exception:
        gdf = _add_null_column(gdf, "ejscreen_score")

    return gdf, holc_overlay


# ---------------------------------------------------------------------------
# HOLC
# ---------------------------------------------------------------------------

def _fetch_holc(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Download the full HOLC GeoJSON and clip to bbox."""
    try:
        import geopandas as gpd
    except ImportError:
        return None

    cache_path = cache_dir / "holc_full.geojson"
    if not cache_path.exists():
        req = urllib.request.Request(
            HOLC_API, headers={"User-Agent": "SPARC-DataCollection/1.0"}
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            cache_path.write_bytes(resp.read())

    gdf = gpd.read_file(cache_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs.to_epsg()) != "4326":
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = bbox
    clipped = gdf.cx[minx:maxx, miny:maxy]
    return clipped if not clipped.empty else None


def _join_holc(fishnet_gdf: object, holc_gdf: object) -> object:
    """Spatial join HOLC grade to fishnet cell centroids."""
    import geopandas as gpd

    # Normalise grade column name
    grade_col = next(
        (c for c in ["holc_grade", "grade", "Grade", "GRADE"] if c in holc_gdf.columns),  # type: ignore[union-attr]
        None,
    )
    if grade_col is None:
        return _add_null_column(fishnet_gdf, "holc_grade")

    holc = holc_gdf.copy()  # type: ignore[union-attr]
    holc["holc_grade"] = holc[grade_col].map(HOLC_GRADE_MAP)

    gdf_4326 = fishnet_gdf.to_crs("EPSG:4326")  # type: ignore[union-attr]
    centroids = gdf_4326.copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin(centroids, holc[["holc_grade", "geometry"]], how="left", predicate="within")
    result = fishnet_gdf.copy()  # type: ignore[union-attr]
    result["holc_grade"] = joined["holc_grade"].values  # type: ignore[index]
    return result


# ---------------------------------------------------------------------------
# CDC SVI
# ---------------------------------------------------------------------------

def _join_cdc_svi(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> object:
    """Download CDC SVI CSV and spatially join RPL_THEMES to fishnet."""
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        return _add_null_column(fishnet_gdf, "cdc_svi")

    cache_path = cache_dir / "svi2022_us.csv"
    if not cache_path.exists():
        req = urllib.request.Request(
            CDC_SVI_URL, headers={"User-Agent": "SPARC-DataCollection/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            cache_path.write_bytes(resp.read())

    svi = pd.read_csv(cache_path, usecols=["FIPS", "RPL_THEMES", "LATITUDE", "LONGITUDE"])
    svi = svi[svi["RPL_THEMES"] >= 0]  # -999 = no data sentinel

    minx, miny, maxx, maxy = bbox
    svi = svi[
        (svi["LONGITUDE"] >= minx) & (svi["LONGITUDE"] <= maxx) &
        (svi["LATITUDE"] >= miny)  & (svi["LATITUDE"] <= maxy)
    ]

    if svi.empty:
        return _add_null_column(fishnet_gdf, "cdc_svi")

    svi_gdf = gpd.GeoDataFrame(
        svi,
        geometry=gpd.points_from_xy(svi["LONGITUDE"], svi["LATITUDE"]),
        crs="EPSG:4326",
    )

    # Spatial join — nearest tract centroid to each fishnet cell centroid
    gdf_4326 = fishnet_gdf.to_crs("EPSG:4326")  # type: ignore[union-attr]
    centroids = gdf_4326.copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin_nearest(centroids, svi_gdf[["RPL_THEMES", "geometry"]], how="left")
    result = fishnet_gdf.copy()  # type: ignore[union-attr]
    result["cdc_svi"] = joined["RPL_THEMES"].values  # type: ignore[index]
    return result


# ---------------------------------------------------------------------------
# EJScreen
# ---------------------------------------------------------------------------

def _join_ejscreen(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
) -> object:
    """Fetch EJScreen composite score for the study area via REST API."""
    try:
        import geopandas as gpd
    except ImportError:
        return _add_null_column(fishnet_gdf, "ejscreen_score")

    minx, miny, maxx, maxy = bbox
    # EJScreen GIS REST service — query block groups within bbox
    params = urllib.parse.urlencode({
        "f": "geojson",
        "geometry": json.dumps({
            "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
            "spatialReference": {"wkid": 4326},
        }),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "P_VULEOPCT",   # EJScreen composite vulnerability percentile
        "returnGeometry": "true",
        "outSR": "4326",
    })
    url = (
        "https://ejscreen.epa.gov/mapper/arcgis/rest/services/ejscreenrest/"
        "ejscreen_2023/MapServer/0/query?" + params
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return _add_null_column(fishnet_gdf, "ejscreen_score")

    features = data.get("features", [])
    if not features:
        return _add_null_column(fishnet_gdf, "ejscreen_score")

    ejs_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    ejs_gdf = ejs_gdf.rename(columns={"P_VULEOPCT": "ejscreen_score"})
    # Normalise 0–100 percentile → 0–1
    ejs_gdf["ejscreen_score"] = ejs_gdf["ejscreen_score"].clip(0, 100) / 100.0

    gdf_4326 = fishnet_gdf.to_crs("EPSG:4326")  # type: ignore[union-attr]
    centroids = gdf_4326.copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin(centroids, ejs_gdf[["ejscreen_score", "geometry"]], how="left", predicate="within")
    result = fishnet_gdf.copy()  # type: ignore[union-attr]
    result["ejscreen_score"] = joined["ejscreen_score"].values  # type: ignore[index]
    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _add_null_column(gdf: object, col_name: str) -> object:
    result = gdf.copy()  # type: ignore[union-attr]
    result[col_name] = float("nan")  # type: ignore[index]
    return result


def _default_cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "equity"
    p.mkdir(parents=True, exist_ok=True)
    return p
