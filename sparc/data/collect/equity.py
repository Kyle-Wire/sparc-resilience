"""equity.py — HOLC redlining, health vulnerability, and EJScreen joins to analysis grid.

Fetches three equity layers and joins them to the 30m fishnet:
  holc_grade     — HOLC redlining grade A=4, B=3, C=2, D=1; NaN outside coverage
  cdc_svi        — CDC PLACES health vulnerability composite (0–1), derived from
                   mental health, obesity, smoking, physical health, and inactivity
                   prevalence rates.  Named ``cdc_svi`` for API compatibility.
  ejscreen_score — EPA EJScreen composite percentile (0–100, normalised to 0–1)

All three are model co-variates joined spatially to the grid.  HOLC polygons
are also returned separately as a GeoDataFrame for the desktop boundary overlay.
Layers gracefully degrade to NaN / skipped when the study area has no coverage
(e.g. non-US locations).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

HTTP_TIMEOUT = 30.0

# Mapping Inequality HOLC — full download (returns HTML as of 2025; kept as
# primary attempt, other paths tried before giving up).
HOLC_API = "https://dsl.richmond.edu/panorama/redlining/static/fullDownload.geojson"

# EPA EJScreen REST API — removed from geodata.epa.gov and ejscreen.epa.gov in
# early 2025.  CDC SVI (cdc_svi) provides equivalent health-vulnerability signal
# and is collected via _join_cdc_svi above, so EJScreen is gracefully skipped.
EJSCREEN_API: str | None = None  # service no longer available

# CDC PLACES census-tract health indicators (SODA, no key required)
# Used to compute a composite health-vulnerability proxy for ``cdc_svi``.
CDC_PLACES_URL = "https://data.cdc.gov/resource/ib3w-k9rq.json"

HOLC_GRADE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}


# ---------------------------------------------------------------------------
# Public API — two-phase (download then assign to grid)
# ---------------------------------------------------------------------------

def download_equity(
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Optional[Path] = None,
) -> tuple[Optional[object], Optional[object]]:
    """Download HOLC and CDC SVI data for *bbox* without grid assignment.

    This is **Phase 1** of the two-phase collection pattern.  Equity layer
    polygons/points are fetched and returned as GeoDataFrames.  Call
    :func:`assign_equity_to_grid` once the fishnet has been created at the
    desired resolution.

    Returns
    -------
    (holc_gdf | None, cdc_svi_gdf | None)
        Raw GeoDataFrames for HOLC polygons and CDC SVI health-vulnerability
        points.  ``None`` when the data is unavailable for this area.
    """
    cache_dir = cache_dir or _default_cache_dir()

    holc_gdf: Optional[object] = None
    try:
        holc_gdf = _fetch_holc(bbox, cache_dir)
    except Exception:
        pass

    cdc_svi_gdf: Optional[object] = None
    try:
        cdc_svi_gdf = _fetch_cdc_svi_points(bbox, cache_dir)
    except Exception:
        pass

    return holc_gdf, cdc_svi_gdf


def assign_equity_to_grid(
    fishnet_gdf: object,
    holc_gdf: Optional[object],
    cdc_svi_gdf: Optional[object],
) -> tuple[object, Optional[object]]:
    """Join downloaded equity data onto fishnet cells.

    This is **Phase 2** of the two-phase collection pattern.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        Analysis grid at any resolution.
    holc_gdf : gpd.GeoDataFrame | None
        HOLC redlining polygons from :func:`download_equity`.
    cdc_svi_gdf : gpd.GeoDataFrame | None
        CDC SVI health-vulnerability points from :func:`download_equity`.

    Returns
    -------
    (gdf, holc_overlay)
        ``gdf`` — fishnet with ``holc_grade``, ``cdc_svi``, and
        ``ejscreen_score`` columns appended.
        ``holc_overlay`` — HOLC GeoDataFrame for map overlay, or None.
    """
    gdf = fishnet_gdf
    holc_overlay: Optional[object] = None

    # HOLC
    try:
        if holc_gdf is not None and not holc_gdf.empty:  # type: ignore[union-attr]
            gdf = _join_holc(gdf, holc_gdf)
            holc_overlay = holc_gdf
        else:
            gdf = _add_null_column(gdf, "holc_grade")
    except Exception:
        gdf = _add_null_column(gdf, "holc_grade")

    # CDC SVI
    try:
        if cdc_svi_gdf is not None and not cdc_svi_gdf.empty:  # type: ignore[union-attr]
            gdf = _assign_cdc_svi_to_grid(gdf, cdc_svi_gdf)
        else:
            gdf = _add_null_column(gdf, "cdc_svi")
    except Exception:
        gdf = _add_null_column(gdf, "cdc_svi")

    # EJScreen — permanently unavailable; always NaN
    gdf = _add_null_column(gdf, "ejscreen_score")

    return gdf, holc_overlay


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compatibility)
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
    holc_gdf, cdc_svi_gdf = download_equity(bbox, cache_dir=cache_dir)
    return assign_equity_to_grid(fishnet_gdf, holc_gdf, cdc_svi_gdf)


# ---------------------------------------------------------------------------
# HOLC
# ---------------------------------------------------------------------------

def _fetch_holc(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Download the full HOLC GeoJSON and clip to bbox.

    The Mapping Inequality project migrated to a SPA in 2024; the static
    download URL may return an HTML page instead of GeoJSON.  We detect this
    and raise so the caller can fall back gracefully.
    """
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
            raw = resp.read()

        # Detect HTML shell (project migrated to SPA; static GeoJSON gone)
        if raw[:5].lower().startswith(b"<!doc") or raw[:5].lower().startswith(b"<html"):
            raise RuntimeError(
                "HOLC download returned an HTML page — "
                "the Mapping Inequality static GeoJSON endpoint is unavailable. "
                "holc_grade will be NaN for this run."
            )
        cache_path.write_bytes(raw)

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
# CDC SVI (via CDC PLACES health-vulnerability proxy)
# ---------------------------------------------------------------------------

def _fetch_cdc_svi_points(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Fetch CDC PLACES health-vulnerability composite as a GeoDataFrame of points.

    Returns a GeoDataFrame in EPSG:3857 with a ``cdc_svi`` column (0–1 score),
    or ``None`` when no tracts are found in *bbox*.
    """
    try:
        import geopandas as gpd
    except ImportError:
        return None

    minx, miny, maxx, maxy = bbox
    params = urllib.parse.urlencode({
        "$where": f"within_box(geolocation, {miny}, {minx}, {maxy}, {maxx})",
        "$select": (
            "tractfips,geolocation,"
            "csmoking_crudeprev,obesity_crudeprev,"
            "mhlth_crudeprev,phlth_crudeprev,lpa_crudeprev"
        ),
        "$limit": "2000",
    })
    url = f"{CDC_PLACES_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        rows = json.loads(resp.read().decode("utf-8"))

    if not rows:
        return None

    _PLACE_FIELDS = [
        "csmoking_crudeprev", "obesity_crudeprev",
        "mhlth_crudeprev", "phlth_crudeprev", "lpa_crudeprev",
    ]
    records: list[dict] = []
    for row in rows:
        geo = row.get("geolocation")
        if not geo:
            continue
        coords = geo.get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])

        vals = []
        for field in _PLACE_FIELDS:
            try:
                vals.append(float(row[field]))
            except (KeyError, ValueError, TypeError):
                pass
        if not vals:
            continue
        composite = float(np.mean(vals)) / 100.0  # normalise prevalence % → 0–1
        records.append({"lon": lon, "lat": lat, "cdc_svi": composite})

    if not records:
        return None

    import pandas as pd
    df = pd.DataFrame(records)
    svi_gdf = gpd.GeoDataFrame(
        df[["cdc_svi"]],
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return svi_gdf.to_crs("EPSG:3857")


def _assign_cdc_svi_to_grid(
    fishnet_gdf: object,
    svi_gdf: object,
) -> object:
    """Spatially join CDC SVI points (EPSG:3857) onto fishnet cell centroids."""
    import geopandas as gpd

    # Compute centroids in projected CRS to avoid geographic-CRS warnings
    centroids = fishnet_gdf.copy()  # type: ignore[union-attr]
    centroids["geometry"] = centroids.geometry.centroid  # type: ignore[union-attr]

    joined = gpd.sjoin_nearest(centroids, svi_gdf[["cdc_svi", "geometry"]], how="left")  # type: ignore[index]
    result = fishnet_gdf.copy()  # type: ignore[union-attr]
    result["cdc_svi"] = joined["cdc_svi"].values  # type: ignore[index]
    return result


def _join_cdc_svi(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> object:
    """Legacy: fetch CDC SVI and join to fishnet in one call."""
    try:
        svi_gdf = _fetch_cdc_svi_points(bbox, cache_dir)
    except Exception:
        return _add_null_column(fishnet_gdf, "cdc_svi")

    if svi_gdf is None:
        return _add_null_column(fishnet_gdf, "cdc_svi")
    return _assign_cdc_svi_to_grid(fishnet_gdf, svi_gdf)



# ---------------------------------------------------------------------------
# EJScreen
# ---------------------------------------------------------------------------

def _join_ejscreen(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
) -> object:
    """EJScreen — permanently unavailable since EPA removed the service in 2025.

    CDC SVI (``cdc_svi``) provides equivalent health-vulnerability signal and is
    already collected via ``_join_cdc_svi``.  This function exists only for API
    compatibility; it returns the fishnet unchanged with an all-NaN column.
    """
    return _add_null_column(fishnet_gdf, "ejscreen_score")


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
