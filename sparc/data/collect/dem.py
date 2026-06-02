"""
dem.py — Digital Elevation Model (DEM) collector for the SPARC pipeline.

Downloads terrain data and derives:
  - elevation_m     : metres above sea level (NAVD88 / EGM2008)
  - slope_deg       : slope in degrees (0–90)
  - aspect_deg      : aspect in degrees from North (0–360)
  - distance_water_m: Euclidean distance in metres to the nearest water body

Data sources
------------
US (bbox centroid longitude −180–−60, latitude 18–72):
  USGS 3DEP 1/3 arc-second (≈10m) via WCS endpoint:
  https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer

Global (all other bboxes):
  Copernicus DEM GLO-30 (≈30m) via Microsoft Planetary Computer STAC:
  https://planetarycomputer.microsoft.com/api/stac/v1/search
  Collection: cop-dem-glo-30

Distance to water
-----------------
Derived via OSM Overpass API — queries water polygon features within the bbox
(natural=water, waterway=*, landuse=reservoir) and computes per-cell
centroid distance.  Falls back to a fixed NaN column if Overpass is
unavailable.

Public API
----------
  download_dem(bbox, *, is_us=None, cache_dir=None) -> tuple[dict|None, list[str]]
  assign_dem_to_grid(fishnet_gdf, dem_data) -> GeoDataFrame
"""

from __future__ import annotations

import io
import json
import logging
import math
import struct
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 90.0

# USGS 3DEP WCS endpoint for 1/3 arc-second National Elevation Dataset (NED)
_3DEP_WCS = (
    "https://elevation.nationalmap.gov/arcgis/services/"
    "3DEPElevation/ImageServer/WCSServer"
)

# Planetary Computer STAC for Copernicus GLO-30
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_COP_DEM_COLLECTION = "cop-dem-glo-30"

# OSM Overpass API
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box defining the contiguous US + AK + HI + territories
_US_LON_MIN, _US_LON_MAX = -180.0, -60.0
_US_LAT_MIN, _US_LAT_MAX = 18.0, 72.0


def _is_us_bbox(bbox: tuple[float, float, float, float]) -> bool:
    """Return True if the bbox centroid is within the US extent."""
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    return (
        _US_LON_MIN <= cx <= _US_LON_MAX
        and _US_LAT_MIN <= cy <= _US_LAT_MAX
    )


# ---------------------------------------------------------------------------
# DEM download helpers
# ---------------------------------------------------------------------------

def _download_3dep(bbox: tuple[float, float, float, float]) -> bytes:
    """Download a GeoTIFF from USGS 3DEP WCS for the given WGS84 bbox."""
    minx, miny, maxx, maxy = bbox
    # WCS 1.0.0 GetCoverage request — returns GeoTIFF
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": "DEP3Elevation",
        "CRS": "EPSG:4326",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": "512",
        "HEIGHT": "512",
        "FORMAT": "GeoTiff",
    }
    url = _3DEP_WCS + "?" + urllib.parse.urlencode(params)
    log.info("dem: downloading 3DEP elevation from %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


def _download_srtm_opentopo(bbox: tuple[float, float, float, float]) -> bytes:
    """Download SRTM GL1 (30m) elevation GeoTIFF from OpenTopography public API.

    Uses the public demo API key.  No authentication beyond that is required.
    """
    minx, miny, maxx, maxy = bbox
    params = urllib.parse.urlencode({
        "demtype": "SRTMGL1",
        "south": str(miny),
        "north": str(maxy),
        "west": str(minx),
        "east": str(maxx),
        "outputFormat": "GTiff",
        "API_Key": "demoapikeyot",
    })
    url = f"https://portal.opentopography.org/API/globaldem?{params}"
    log.info("dem: downloading SRTM from OpenTopography: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = resp.read()
    # OpenTopography returns an error XML/JSON on failure
    if len(data) < 100 or data[:4] not in (b"II\x2a\x00", b"MM\x00\x2a"):  # TIFF magic bytes
        raise RuntimeError(f"OpenTopography returned non-TIFF response: {data[:200]}")
    return data


def _download_cop_dem(bbox: tuple[float, float, float, float]) -> bytes:
    """Download Copernicus GLO-30 DEM tiles for the given WGS84 bbox.

    Queries Planetary Computer STAC, signs the download URLs, and mosaics
    the returned tiles into a single in-memory GeoTIFF.
    """
    try:
        import rasterio
        from rasterio.merge import merge as rasterio_merge
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError(f"rasterio is required for Copernicus DEM download: {exc}")

    minx, miny, maxx, maxy = bbox
    payload = json.dumps({
        "collections": [_COP_DEM_COLLECTION],
        "bbox": [minx, miny, maxx, maxy],
        "limit": 100,
    }).encode()
    req = urllib.request.Request(
        PC_STAC,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        stac = json.loads(resp.read())

    features = stac.get("features", [])
    if not features:
        raise FileNotFoundError(f"No Copernicus DEM tiles found for bbox {bbox}")

    log.info("dem: found %d Copernicus DEM tiles via PC STAC", len(features))

    # Download each tile, collect opened rasterio datasets, then merge
    tile_files: list[io.BytesIO] = []
    datasets: list = []
    for feat in features:
        href = (feat.get("assets", {}).get("data") or {}).get("href")
        if not href:
            continue
        # Strip any expired/empty SAS tokens from Azure Blob URLs — the
        # public Copernicus DEM files do not require auth.
        if "blob.core.windows.net" in href and "?" in href:
            href = href.split("?")[0]
        try:
            tile_req = urllib.request.Request(href, headers={"User-Agent": "SPARC/1.0"})
            with urllib.request.urlopen(tile_req, timeout=HTTP_TIMEOUT) as tresp:
                tile_bytes = tresp.read()
            buf = io.BytesIO(tile_bytes)
            tile_files.append(buf)
            ds = rasterio.open(buf)
            datasets.append(ds)
        except Exception as exc:
            log.warning("dem: failed to download tile %s: %s", href, exc)

    if not datasets:
        raise RuntimeError("No Copernicus DEM tiles could be downloaded")

    mosaic, mosaic_transform = rasterio_merge(datasets)
    for ds in datasets:
        ds.close()

    out_buf = io.BytesIO()
    with rasterio.open(
        out_buf,
        "w",
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype=mosaic.dtype,
        crs="EPSG:4326",
        transform=mosaic_transform,
    ) as dst:
        dst.write(mosaic)
    out_buf.seek(0)
    return out_buf.read()


# ---------------------------------------------------------------------------
# Terrain derivative computation
# ---------------------------------------------------------------------------

def _compute_slope_aspect(
    elevation: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (degrees from North) from a 2-D array.

    Uses the Horn (1981) 8-neighbour finite-difference method.
    """
    # Pad with edge values to handle borders
    padded = np.pad(elevation, 1, mode="edge")

    dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * cell_size_m)
    dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * cell_size_m)

    # Slope in degrees
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.degrees(np.arctan(rise_run))

    # Aspect: 0° = North, clockwise
    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx))
    aspect = (90.0 - aspect) % 360.0

    return slope, aspect


def _read_geotiff_to_array(tiff_bytes: bytes) -> tuple[np.ndarray, object, tuple]:
    """Read a GeoTIFF from raw bytes and return (array, transform, crs)."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(f"rasterio is required for DEM processing: {exc}")

    buf = io.BytesIO(tiff_bytes)
    with rasterio.open(buf) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


# ---------------------------------------------------------------------------
# Water body distance
# ---------------------------------------------------------------------------

def _fetch_water_polygons_osm(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """Query OSM Overpass for water polygons within bbox.

    Returns a list of GeoJSON-compatible geometry dicts.
    """
    minx, miny, maxx, maxy = bbox
    query = (
        f"[out:json][timeout:30];"
        f"("
        f'way["natural"="water"]({miny},{minx},{maxy},{maxx});'
        f'relation["natural"="water"]({miny},{minx},{maxy},{maxx});'
        f'way["waterway"~"river|canal|stream"]({miny},{minx},{maxy},{maxx});'
        f'way["landuse"="reservoir"]({miny},{minx},{maxy},{maxx});'
        f");"
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        _OVERPASS_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        log.warning("dem: OSM water query failed: %s", exc)
        return []

    geometries: list[dict] = []
    for element in result.get("elements", []):
        geom = element.get("geometry", [])
        if geom and element.get("type") == "way":
            coords = [[pt["lon"], pt["lat"]] for pt in geom]
            if len(coords) >= 3:
                geometries.append({"type": "Polygon", "coordinates": [coords]})
    return geometries


def _compute_distance_to_water(
    cell_centroids_xy: np.ndarray,
    water_geometries: list[dict],
    crs_epsg: str = "EPSG:4326",
) -> np.ndarray:
    """Compute per-cell Euclidean distance to the nearest water polygon.

    Parameters
    ----------
    cell_centroids_xy : ndarray shape (N, 2)  — [lon, lat] in the cell CRS
    water_geometries  : list of GeoJSON geometry dicts
    crs_epsg          : EPSG string for the cell CRS (affects units of output)

    Returns
    -------
    ndarray of floats, length N — distances in metres (or degrees if
    geographic CRS; the caller should project first for accuracy).
    """
    if not water_geometries:
        return np.full(len(cell_centroids_xy), np.nan, dtype=np.float32)

    try:
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, shape
    except ImportError:
        return np.full(len(cell_centroids_xy), np.nan, dtype=np.float32)

    try:
        water_shapes = [shape(g) for g in water_geometries if g]
        if not water_shapes:
            return np.full(len(cell_centroids_xy), np.nan, dtype=np.float32)

        water_union = MultiPolygon(
            [s for s in water_shapes if s.is_valid and not s.is_empty]
        )
        if water_union.is_empty:
            return np.full(len(cell_centroids_xy), np.nan, dtype=np.float32)

        from shapely.geometry import Point

        distances = np.array(
            [Point(xy).distance(water_union) for xy in cell_centroids_xy],
            dtype=np.float32,
        )

        # If CRS is geographic (degrees), convert to approximate metres
        if "4326" in crs_epsg or "WGS" in crs_epsg.upper():
            # 1 degree ≈ 111_320 m at equator
            distances = distances * 111_320.0

        return distances
    except Exception as exc:
        log.warning("dem: distance_to_water computation failed: %s", exc)
        return np.full(len(cell_centroids_xy), np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dem(
    bbox: tuple[float, float, float, float],
    *,
    is_us: Optional[bool] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[dict | None, list[str]]:
    """Download DEM data for the given WGS84 bounding box.

    Parameters
    ----------
    bbox : (minx, miny, maxx, maxy) in WGS84 degrees
    is_us : Override US detection.  If None, auto-detected from bbox centroid.
    cache_dir : If provided, cache downloaded bytes to disk and read from
                cache on subsequent calls.

    Returns
    -------
    (dem_data, warnings)
      dem_data is a dict with keys:
        "tiff_bytes" : bytes  — raw GeoTIFF
        "source"     : str    — "3dep" or "cop_dem_glo30"
        "bbox"       : tuple  — the requested bbox
        "is_us"      : bool
      warnings is a list of strings describing non-fatal issues.
    """
    warnings: list[str] = []

    if is_us is None:
        is_us = _is_us_bbox(bbox)

    # Check cache
    cache_key: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        bbox_str = "_".join(f"{v:.4f}" for v in bbox)
        source_tag = "3dep" if is_us else "cop30"
        cache_key = cache_dir / f"dem_{source_tag}_{bbox_str}.tif"
        if cache_key.exists():
            log.info("dem: using cached DEM at %s", cache_key)
            return {
                "tiff_bytes": cache_key.read_bytes(),
                "source": source_tag,
                "bbox": bbox,
                "is_us": is_us,
            }, warnings

    tiff_bytes: Optional[bytes] = None

    if is_us:
        try:
            tiff_bytes = _download_3dep(bbox)
            source = "3dep"
        except Exception as exc:
            warnings.append(f"3DEP download failed ({exc}); trying OpenTopography SRTM")
            log.warning("dem: 3DEP failed: %s", exc)

    if tiff_bytes is None:
        try:
            tiff_bytes = _download_srtm_opentopo(bbox)
            source = "srtm_opentopo"
        except Exception as exc:
            warnings.append(f"OpenTopography SRTM also failed ({exc}); trying Copernicus GLO-30")
            log.warning("dem: OpenTopography failed: %s", exc)

    if tiff_bytes is None:
        try:
            tiff_bytes = _download_cop_dem(bbox)
            source = "cop_dem_glo30"
        except Exception as exc:
            warnings.append(f"Copernicus GLO-30 download also failed: {exc}")
            log.error("dem: cop-dem failed: %s", exc)
            return None, warnings

    if cache_key is not None and tiff_bytes is not None:
        try:
            cache_key.write_bytes(tiff_bytes)
        except Exception:
            pass

    return {
        "tiff_bytes": tiff_bytes,
        "source": source,
        "bbox": bbox,
        "is_us": is_us,
    }, warnings


def assign_dem_to_grid(
    fishnet_gdf,
    dem_data: dict,
) -> object:
    """Compute terrain variables for each cell in the fishnet and attach them.

    Derives ``elevation_m``, ``slope_deg``, ``aspect_deg``, and
    ``distance_water_m`` from the DEM GeoTIFF and spatial-joins them to
    the fishnet using zonal statistics (cell centroid sampling).

    Parameters
    ----------
    fishnet_gdf : GeoDataFrame with cell polygons (any CRS)
    dem_data    : dict returned by ``download_dem``

    Returns
    -------
    GeoDataFrame with new columns added in-place.
    """
    try:
        import geopandas as gpd
        import rasterio
        import rasterio.transform
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError as exc:
        log.error("dem: rasterio/geopandas required: %s", exc)
        for col in ("elevation_m", "slope_deg", "aspect_deg", "distance_water_m"):
            fishnet_gdf[col] = np.nan
        return fishnet_gdf

    if dem_data is None or "tiff_bytes" not in dem_data:
        log.warning("dem: no dem_data; filling columns with NaN")
        for col in ("elevation_m", "slope_deg", "aspect_deg", "distance_water_m"):
            fishnet_gdf[col] = np.nan
        return fishnet_gdf

    tiff_bytes: bytes = dem_data["tiff_bytes"]
    bbox: tuple = dem_data.get("bbox", (0, 0, 0, 0))

    # Read raw GeoTIFF
    arr, transform, src_crs = _read_geotiff_to_array(tiff_bytes)

    # Compute slope and aspect (approximate cell size in metres at centroid)
    miny, maxy = bbox[1], bbox[3]
    lat_mid = (miny + maxy) / 2.0
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = 111_320.0 * math.cos(math.radians(lat_mid))
    # Use transform pixel size
    pixel_lon = abs(transform.a)
    pixel_lat = abs(transform.e)
    cell_size_m = (pixel_lon * metres_per_degree_lon + pixel_lat * metres_per_degree_lat) / 2.0
    slope_arr, aspect_arr = _compute_slope_aspect(arr, cell_size_m)

    # Reproject fishnet centroids to match DEM CRS for sampling.
    # Compute centroids in UTM (projected) for accuracy, then reproject to src_crs.
    gdf = fishnet_gdf.copy()
    orig_crs = gdf.crs
    centroids_proj = gdf.geometry.centroid          # accurate in UTM
    centroids_gdf = (
        centroids_proj.to_frame("geometry")
        .set_crs(orig_crs)
        .to_crs(src_crs)
    )
    centroids_wgs84 = centroids_gdf.geometry

    # Sample arrays at each centroid using rasterio transform — vectorized
    def _sample_array(arr2d: np.ndarray, xcoords, ycoords, nodata_val=np.nan) -> np.ndarray:
        rows, cols = rasterio.transform.rowcol(transform, xcoords, ycoords)
        nrows, ncols = arr2d.shape
        r_idx = np.clip(np.array(rows, dtype=int), 0, nrows - 1)
        c_idx = np.clip(np.array(cols, dtype=int), 0, ncols - 1)
        out = arr2d[r_idx, c_idx].astype(np.float32)
        out[~np.isfinite(out)] = nodata_val
        return out

    xs = centroids_wgs84.x.to_numpy()
    ys = centroids_wgs84.y.to_numpy()

    gdf["elevation_m"] = _sample_array(arr, xs, ys)
    gdf["slope_deg"] = _sample_array(slope_arr, xs, ys)
    gdf["aspect_deg"] = _sample_array(aspect_arr, xs, ys)

    # Distance to water via OSM
    water_geoms = _fetch_water_polygons_osm(bbox)
    centroids_xy = np.column_stack([xs, ys])
    crs_str = str(src_crs) if src_crs else "EPSG:4326"
    gdf["distance_water_m"] = _compute_distance_to_water(centroids_xy, water_geoms, crs_str)

    return gdf
