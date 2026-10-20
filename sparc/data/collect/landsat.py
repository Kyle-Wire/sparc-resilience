"""
landsat.py — USGS Landsat C2 scene query, index computation, and LST.

Queries the USGS Landsat Collection 2 Level-2 STAC catalog for scenes
covering the study boundary within a configurable date range and cloud cover
threshold.  Downloads COG windows for the required bands, masks clouds with
the QA_PIXEL band, reprojects every scene onto one target grid, and computes
the spectral indices plus land surface temperature (LST).

Temporal modes
--------------
  "single"     — pick the one scene with the lowest cloud cover in the window
  "composite"  — pixel-wise median across all valid scenes → one row per cell
  "panel"      — each scene becomes a dated row set; returns multi-date panel
  campaign     — :func:`campaign_lst`: the clear scene(s) nearest a campaign
                 date within ±N days, with each scene's date, its offset in
                 days from the campaign and its clear fraction over the
                 study area (the rows of one overpass mosaicked; LST masked
                 150 m around cloud)

Spectral indices computed
-------------------------
  ndvi, ndbi, mndwi, ndwi, savi, evi, ndbai, nbr, ndmi, ui, bsi, albedo, lst
  (°C) and lst_kelvin

Method (USGS Landsat Collection 2 Level-2 Science Product Guide)
----------------------------------------------------------------
* **Scaling.**  Surface reflectance SR_B* = DN·0.0000275 − 0.2; surface
  temperature ST_B10 = DN·0.00341802 + 149.0 K (°C = K − 273.15).  DN 0 is
  the fill value of both.  Indices are computed from reflectance, not DN
  (a normalised difference of DNs is not the index: the −0.2 offset does not
  cancel).
* **QA_PIXEL bits** (Collection 2, Landsat 8/9): 0 fill, 1 dilated cloud,
  2 cirrus (high confidence), 3 cloud, 4 cloud shadow, 5 snow, 6 clear,
  7 water; 8–15 are two-bit confidences.  Pixels with the fill, dilated
  cloud, cirrus, cloud or cloud-shadow bit set are masked
  (:data:`QA_MASKED`; the dilated-cloud bit is the product's buffer around
  detected clouds, kept as part of "cloud").  Campaign LST is also masked
  within 150 m of the cloud bits (:data:`MASK_BUFFER_M`; not of fill): clear
  LST next to the mask is cool-biased.
* **Geometry.**  Each band window keeps its own affine transform and CRS
  (Landsat COGs are in the scene's UTM zone; scenes of different WRS-2
  paths/rows do not share a pixel grid).  Every scene is reprojected onto
  one *target grid* (:func:`target_grid`: a UTM grid over the study bbox,
  30 m by default) — bilinear for continuous bands, nearest for QA — before
  any composite, and grid cells are sampled from that grid by projecting
  their centroids into its CRS (:func:`sample_raster`; NaN outside it).
  Before this, ``_assign_arrays_to_fishnet`` stretched each array over the
  fishnet's lon/lat bounds, which misplaced every pixel unless the window
  happened to span exactly those bounds.
* **Thermal resolution.**  TIRS band 10 is acquired at 100 m and delivered
  resampled to 30 m, so LST varies over ~100 m even on the 30 m grid.
  Landsat 8/9 cross the equator at about 10:00 local time; the LST is a
  mid-morning surface temperature, not the afternoon's.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 120.0

# Primary: Microsoft Planetary Computer — signed Azure Blob Storage URLs (no
# user credentials required; the planetary-computer SDK handles token exchange).
PC_STAC        = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_PC_COLLECTION = "landsat-c2-l2"

# Fallback: AWS Element84 Earth-Search (for STAC metadata only; data is on
# the usgs-landsat Requester Pays S3 bucket, which requires AWS credentials).
ELEMENT84_STAC    = "https://earth-search.aws.element84.com/v1/search"
_E84_COLLECTION   = "landsat-c2-l2"

TemporalMode = Literal["single", "composite", "panel"]

# Landsat 8/9 OLI / Landsat 7 ETM+ band mapping (Planetary Computer and
# Element84 Earth-Search use the same human-readable asset keys for L7–L9).
BAND_ASSETS = {
    "blue":    "blue",    # Band 2 (OLI) / Band 1 (ETM+)   ~0.45–0.52 µm
    "green":   "green",   # Band 3 (OLI) / Band 2 (ETM+)   ~0.52–0.60 µm
    "red":     "red",     # Band 4 (OLI) / Band 3 (ETM+)   ~0.63–0.69 µm
    "nir":     "nir08",   # Band 5 (OLI) / Band 4 (ETM+)   ~0.85–0.88 µm
    "swir1":   "swir16",  # Band 6 (OLI) / Band 5 (ETM+)   ~1.57–1.65 µm
    "swir2":   "swir22",  # Band 7 (OLI) / Band 7 (ETM+)   ~2.11–2.29 µm
    "thermal": "lwir11",  # Band 10 (OLI) / Band 6 (ETM+)  ~10.6–11.2 µm (ST)
    "qa":      "qa_pixel",
}

# Collection 2 Level-2 scale factors (USGS Landsat Collection 2 Level-2 Science Product Guide)
SR_SCALE   = 0.0000275
SR_OFFSET  = -0.2
LST_SCALE  = 0.00341802
LST_OFFSET = 149.0          # K; subtract 273.15 for °C
KELVIN     = 273.15

#: QA_PIXEL bit positions (Collection 2 Level-2, Landsat 8/9 OLI/TIRS).
QA_BITS = {"fill": 0, "dilated_cloud": 1, "cirrus": 2, "cloud": 3, "cloud_shadow": 4, "snow": 5,
           "clear": 6, "water": 7}
#: Bits whose pixels are masked by default.
QA_MASKED = ("fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow")
#: Masked bits around which :func:`campaign_lst` also masks LST (:data:`MASK_BUFFER_M`): every one but fill, so the
#: edge of a scene's footprint is not eaten.
CLOUD_BITS = ("dilated_cloud", "cirrus", "cloud", "cloud_shadow")
#: Distance (m) from a cloud-masked pixel within which campaign LST is masked too: 5 Landsat 30 m pixels.  Clear LST
#: next to the QA mask is cool-biased (TIRS's 100 m pixels and cloud edges just outside the mask): the land-cover
#: residual of LST, within 2 km blocks, is −0.75 to −0.95 °C at 0–60 m from masked pixels, −0.32 at 60–120 m and about
#: 0 at 120–240 m (Charleston 2021; Virginia Beach −0.70 to −0.87 at 60–120 m, −0.16 to −0.27 at 120–240 m), while
#: dTa shows no such cooling.  150 m keeps 80–89% of their LST cells (300 m: 62–76%).
MASK_BUFFER_M = 150.0
#: When no scene is ``prefer_clear``, scenes within this much clear fraction of the clearest count as clear as it and
#: the nearest of them is taken (:func:`campaign_lst`).
CLEAR_TOLERANCE = 0.10
#: Platforms used by :func:`campaign_lst` (Landsat 7's scan-line corrector failed in 2003).
LST_PLATFORMS = ("landsat-8", "landsat-9")

Reader = Callable[[str, tuple], tuple]          # (href, lon/lat bbox) → (array, affine transform, crs)
Search = Callable[[tuple, date, date], list]    # (lon/lat bbox, start, end) → scene dicts

#: Planetary Computer SAS token for the Landsat Collection 2 container (what ``planetary_computer.sign`` fetches).
PC_SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsateuwest/landsat-c2"
PC_BLOB = "https://landsateuwest.blob.core.windows.net/landsat-c2"

_BLOCKED_TEXT = ("tunnel connection failed", "could not resolve host", "name or service not known",
                 "temporary failure in name resolution", "connection refused", "network is unreachable",
                 "from proxy after connect")
_BLOCKED_URLERROR = _BLOCKED_TEXT + ("proxy", "403", "401", "407")
_HTTP_CODE = re.compile(r"(?:http(?: response code| error)?[: ]*|status[: ]*)(\d{3})", re.IGNORECASE)


def io_error_kind(exc: BaseException) -> str | None:
    """How a read failed: ``"blocked"`` (the host refused or cannot be reached: HTTP 401/403/407, a proxy's
    tunnel refusal, DNS or connection refused), ``"io"`` (any other network or file I/O failure: 5xx, 404,
    timeouts, cut-off transfers, GDAL/rasterio I/O errors) or None (not an I/O error: a data problem).  The
    exception's cause chain is inspected too (``RuntimeError(...) from URLError``)."""
    import http.client
    import urllib.error

    seen = 0
    e: BaseException | None = exc
    kind = None
    while e is not None and seen < 6:
        seen += 1
        msg = str(e).lower()
        if isinstance(e, urllib.error.HTTPError):
            return "blocked" if e.code in (401, 403, 407) else "io"
        if isinstance(e, urllib.error.URLError):
            if isinstance(e.reason, TimeoutError) or "timed out" in msg:
                return "io"
            return "blocked" if any(t in msg for t in _BLOCKED_URLERROR) or isinstance(e.reason, OSError) else "io"
        name = type(e).__name__
        if name in ("RasterioIOError", "CPLE_HttpResponseError", "CPLE_OpenFailedError", "CPLE_FileIOError",
                    "CPLE_AWSError", "CPLE_AWSAccessDeniedError", "CPLE_AWSObjectNotFoundError"):
            m = _HTTP_CODE.search(msg)
            if (m and m.group(1) in ("401", "403", "407")) or any(t in msg for t in _BLOCKED_TEXT):
                return "blocked"
            kind = "io"
        elif isinstance(e, (TimeoutError, ConnectionError, http.client.HTTPException)):
            kind = kind or "io"
        elif isinstance(e, OSError):
            kind = kind or ("blocked" if any(t in msg for t in _BLOCKED_TEXT) else "io")
        e = e.__cause__ or e.__context__
    return kind


def href_host(href: str | None) -> str | None:
    from urllib.parse import urlparse

    if not href:
        return None
    h = str(href)
    for pre in ("/vsicurl/", "/vsis3/", "/vsiaz/"):
        if h.startswith(pre):
            h = h[len(pre):]
    return urlparse(h).hostname


# ---------------------------------------------------------------------------
# Scaling, QA and geometry helpers
# ---------------------------------------------------------------------------

def qa_mask(qa: np.ndarray, bits=QA_MASKED) -> np.ndarray:
    """True where a QA_PIXEL value has any of ``bits`` set (names of :data:`QA_BITS`)."""
    q = np.asarray(qa).astype(np.uint32)
    flag = 0
    for b in bits:
        flag |= 1 << QA_BITS[b]
    return (q & flag) != 0


def buffer_mask(mask: np.ndarray, distance_m: float, pixel_m: tuple[float, float] = (30.0, 30.0)) -> np.ndarray:
    """True within ``distance_m`` of a True pixel of ``mask`` (Euclidean, centre to centre; ``pixel_m`` = the
    pixel's (height, width) in m).  At 30 m, 150 m reaches 5 pixels along a row or column."""
    from scipy.ndimage import binary_dilation

    m = np.asarray(mask, dtype=bool)
    if distance_m <= 0 or not m.any():
        return m.copy()
    dy, dx = (float(abs(v)) for v in pixel_m)
    ry, rx = int(distance_m // dy), int(distance_m // dx)
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    disk = (yy * dy) ** 2 + (xx * dx) ** 2 <= distance_m ** 2 * (1.0 + 1e-9)
    return binary_dilation(m, structure=disk)


_WRS_ID = re.compile(r"^L[A-Z]\d{2}_[A-Z0-9]{4}_(\d{3})(\d{3})_\d{8}")


def wrs_path_row(scene: dict) -> tuple[int, int] | None:
    """WRS-2 (path, row) of a scene: its ``wrs_path`` / ``wrs_row`` when given, else read from the product id
    (``LC08_L2SP_PPPRRR_YYYYMMDD_…``, as on the Planetary Computer and Earth-Search); None when neither has them."""
    p, r = scene.get("wrs_path"), scene.get("wrs_row")
    if p is not None and r is not None:
        return int(p), int(r)
    m = _WRS_ID.match(str(scene.get("id") or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def st_to_kelvin(dn: np.ndarray) -> np.ndarray:
    """ST_B10 digital numbers → surface temperature (K); the fill value 0 → NaN."""
    d = np.asarray(dn, dtype=np.float64)
    out = d * LST_SCALE + LST_OFFSET
    out[~np.isfinite(d) | (d <= 0)] = np.nan
    return out


def sr_to_reflectance(dn: np.ndarray) -> np.ndarray:
    """SR_B* digital numbers → surface reflectance; the fill value 0 → NaN."""
    d = np.asarray(dn, dtype=np.float64)
    out = d * SR_SCALE + SR_OFFSET
    out[~np.isfinite(d) | (d <= 0)] = np.nan
    return out


def utm_crs(lon: float, lat: float) -> str:
    """The WGS-84 UTM zone CRS of a lon/lat point (standard zones; no Norway/Svalbard exceptions)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def target_grid(bbox: tuple[float, float, float, float], res: float = 30.0, crs: str | None = None) -> dict:
    """A north-up grid covering the lon/lat ``bbox``: ``{"transform", "crs", "shape"}``.

    The grid is in ``crs`` (default: the UTM zone of the bbox centre), its edges are multiples of ``res``
    and it covers the projected envelope of the bbox."""
    from pyproj import Transformer
    from rasterio.transform import from_origin

    w, s, e, n = (float(v) for v in bbox)
    crs = crs or utm_crs(0.5 * (w + e), 0.5 * (s + n))
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lo = np.linspace(w, e, 9)
    la = np.linspace(s, n, 9)
    LO, LA = np.meshgrid(lo, la)
    xs, ys = tr.transform(LO.ravel(), LA.ravel())
    x0, x1 = math.floor(min(xs) / res) * res, math.ceil(max(xs) / res) * res
    y0, y1 = math.floor(min(ys) / res) * res, math.ceil(max(ys) / res) * res
    width, height = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return {"transform": from_origin(x0, y1, res, res), "crs": crs, "shape": (height, width)}


def reproject_array(arr: np.ndarray, src_transform, src_crs, dst: dict, resampling: str = "bilinear",
                    nodata: float = np.nan) -> np.ndarray:
    """``arr`` (georeferenced by ``src_transform`` / ``src_crs``) resampled onto the ``dst`` grid
    (:func:`target_grid`).  Float arrays use NaN as no-data; cells the source does not cover get ``nodata``."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    rs = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest, "average": Resampling.average}[resampling]
    src = np.asarray(arr)
    if np.issubdtype(src.dtype, np.floating):
        src = src.astype(np.float64)
        out = np.full(dst["shape"], np.nan, dtype=np.float64)
        reproject(src, out, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=dst["transform"], dst_crs=dst["crs"], dst_nodata=np.nan, resampling=rs)
        if not (isinstance(nodata, float) and math.isnan(nodata)):
            out[np.isnan(out)] = nodata
        return out
    out = np.full(dst["shape"], nodata, dtype=src.dtype)
    reproject(src, out, src_transform=src_transform, src_crs=src_crs, dst_transform=dst["transform"],
              dst_crs=dst["crs"], dst_nodata=nodata, resampling=rs)
    return out


def sample_raster(arr: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Value of the pixel containing each (x, y) (coordinates in the raster's CRS); NaN outside the raster.

    Unlike clipping row/column indices into range, a point beyond the raster's edge gets no value."""
    inv = ~transform
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    cols = inv.a * xs + inv.b * ys + inv.c
    rows = inv.d * xs + inv.e * ys + inv.f
    c = np.floor(cols).astype(np.int64)
    r = np.floor(rows).astype(np.int64)
    h, w = arr.shape
    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w) & np.isfinite(cols) & np.isfinite(rows)
    out = np.full(xs.shape, np.nan)
    vals = np.asarray(arr)[r[ok], c[ok]].astype(np.float64)
    out[ok] = vals
    out[~np.isfinite(out)] = np.nan
    return out


def bbox_mask(dst: dict, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """True for the ``dst`` grid cells whose centre lies inside the lon/lat ``bbox``."""
    from pyproj import Transformer

    h, w = dst["shape"]
    T = dst["transform"]
    jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = T.c + jj * T.a + ii * T.b
    y = T.f + jj * T.d + ii * T.e
    lon, lat = Transformer.from_crs(dst["crs"], "EPSG:4326", always_xy=True).transform(x.ravel(), y.ravel())
    lon = np.asarray(lon).reshape(h, w)
    lat = np.asarray(lat).reshape(h, w)
    W, S, E, N = bbox
    return (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)


def read_window(href: str, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, object, str]:
    """(array, window transform, CRS) of the window of a COG covering the lon/lat ``bbox``.

    The window is cut in the COG's own CRS (Landsat: the scene's UTM zone), clipped to the raster and
    returned with *its own* transform, so the pixels stay where they are on the ground."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    minx, miny, maxx, maxy = bbox
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            sb = transform_bounds("EPSG:4326", src.crs, minx, miny, maxx, maxy, densify_pts=21)
            f = from_bounds(*sb, transform=src.transform)
            c0, r0 = math.floor(f.col_off), math.floor(f.row_off)
            c1, r1 = math.ceil(f.col_off + f.width), math.ceil(f.row_off + f.height)
            win = Window(c0, r0, c1 - c0, r1 - r0).intersection(Window(0, 0, src.width, src.height))
            arr = src.read(1, window=win)
            return arr, src.window_transform(win), src.crs.to_string()


# ---------------------------------------------------------------------------
# Public API — two-phase (download then assign to grid)
# ---------------------------------------------------------------------------

def download_landsat(
    bbox: tuple[float, float, float, float],
    window: "TemporalWindow",
    *,
    cloud_cover_max: float = 20.0,
    temporal_mode: TemporalMode = "composite",
    enabled_indices: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    target: Optional[dict] = None,
    reader: Optional[Reader] = None,
) -> tuple[Optional[object], list[date]]:
    """Download Landsat scenes and compute composite arrays without grid assignment.

    This is **Phase 1** of the two-phase collection pattern.  Raw COG bands are
    fetched from the Planetary Computer STAC, cloud-masked with QA_PIXEL,
    reprojected onto ``target`` (default :func:`target_grid` of ``bbox`` at
    30 m), spectral indices are computed, and the composite is returned as a
    dict of arrays keyed by index name plus ``"_georef"`` (the target grid).
    No fishnet is required at this stage.

    Call :func:`assign_landsat_to_grid` afterwards to apply the composite to
    a fishnet created at the desired resolution.

    Parameters
    ----------
    bbox : (minx, miny, maxx, maxy) in EPSG:4326
    window : TemporalWindow
    cloud_cover_max : float
    temporal_mode : "single" | "composite" | "panel"
    enabled_indices : list[str] | None
    cache_dir : Path, optional
    target : dict, optional
        The grid every scene is reprojected onto (:func:`target_grid`).
    reader : callable, optional
        ``(href, bbox) → (array, transform, crs)``; default :func:`read_window`.

    Returns
    -------
    (composite_arrays | scene_list | None, scene_dates)
        ``composite_arrays`` is a ``dict[str, np.ndarray]`` (plus ``"_georef"``) for
        composite/single mode.  ``scene_list`` is a list of such dicts for
        panel mode.  ``None`` when no suitable scenes are found.
    """
    from ._temporal import TemporalWindow as _TemporalWindow
    if not isinstance(window, _TemporalWindow):
        raise TypeError(
            f"download_landsat() expects a TemporalWindow, got {type(window).__name__}."
        )
    cache_dir = cache_dir or _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    enabled = set(enabled_indices or list(SPECTRAL_INDEX_FORMULAS.keys()) + ["lst"])
    target = target or target_grid(bbox)

    if window.anchor_dates:
        scene_arrays, scene_dates = _fetch_anchored_scenes(
            bbox, window, enabled, cache_dir, cloud_cover_max, temporal_mode, target, reader
        )
    else:
        scene_arrays, scene_dates = _fetch_full_range_scenes(
            bbox, window.date_start, window.date_end, enabled, cache_dir,
            cloud_cover_max, temporal_mode, target, reader
        )

    if not scene_arrays:
        return None, []

    if temporal_mode == "panel":
        return scene_arrays, scene_dates  # list of per-scene dicts (each with "_georef")

    composite = _median_composite(scene_arrays)
    composite["_georef"] = target
    return composite, scene_dates


def assign_landsat_to_grid(
    fishnet_gdf: object,
    composite_or_panels: Optional[object],
    scene_dates: list[date],
    *,
    enabled_indices: Optional[list[str]] = None,
    temporal_mode: TemporalMode = "composite",
) -> object:
    """Apply Landsat composite arrays to fishnet cells by centroid sampling.

    This is **Phase 2** of the two-phase collection pattern.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        Analysis grid at any resolution.
    composite_or_panels : dict[str, np.ndarray] | list | None
        Composite dict or panel scene list from :func:`download_landsat`
        (each carrying ``"_georef"``).  NaN-fills all requested indices when ``None``.
    scene_dates : list[date]
    enabled_indices : list[str] | None
    temporal_mode : str

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet with spectral index columns appended.
    """
    enabled = set(enabled_indices or list(SPECTRAL_INDEX_FORMULAS.keys()) + ["lst"])

    if composite_or_panels is None:
        return _nan_fishnet(fishnet_gdf, enabled)

    if temporal_mode == "panel":
        return _build_panel(fishnet_gdf, composite_or_panels, scene_dates)

    return _assign_arrays_to_fishnet(fishnet_gdf, composite_or_panels)


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fetch_landsat(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    window: "TemporalWindow",
    *,
    cloud_cover_max: float = 20.0,
    temporal_mode: TemporalMode = "composite",
    enabled_indices: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
) -> object:
    """Fetch Landsat data aligned to CAPA anchor dates.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.
    window : TemporalWindow
        Temporal search contract.  ``window.anchor_dates`` are the CAPA
        measurement dates; Landsat searches within ``±window.tolerance_days``
        of each anchor.  If ``anchor_dates`` is empty, falls back to searching
        the full ``window.date_start`` → ``window.date_end`` range.
    cloud_cover_max : float
        Maximum cloud cover percentage (0–100).  Default 20%.
    temporal_mode : "single" | "composite" | "panel"
        How multiple scenes are collapsed.
    enabled_indices : list[str] | None
        Subset of index names to compute; None = all 13.
    cache_dir : Path, optional
        Cache directory for COG downloads.

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet (or panel) with spectral columns.  NaN-filled when no scenes
        are available — never raises on empty STAC results.

    Raises
    ------
    TypeError
        If ``window`` is not a ``TemporalWindow`` instance.  The old bare
        ``date_start / date_end`` positional interface has been removed.
    """
    from ._temporal import TemporalWindow as _TemporalWindow
    if not isinstance(window, _TemporalWindow):
        raise TypeError(
            f"fetch_landsat() expects a TemporalWindow as the third argument, "
            f"got {type(window).__name__}. The bare date_start/date_end interface "
            f"has been replaced — use TemporalWindow.from_capa_dates()."
        )

    composite, scene_dates = download_landsat(
        bbox, window,
        cloud_cover_max=cloud_cover_max,
        temporal_mode=temporal_mode,
        enabled_indices=enabled_indices,
        cache_dir=cache_dir,
    )
    return assign_landsat_to_grid(
        fishnet_gdf, composite, scene_dates,
        enabled_indices=enabled_indices,
        temporal_mode=temporal_mode,
    )


def _nan_fishnet(fishnet_gdf: object, enabled: set) -> object:
    """Return fishnet with NaN columns for all requested indices."""
    try:
        gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    except Exception:
        gdf = fishnet_gdf
    for idx in list(enabled) + ["lst"]:
        gdf[idx] = float("nan")  # type: ignore[index]
    return gdf


def _fetch_anchored_scenes(
    bbox: tuple,
    window: "TemporalWindow",
    enabled: set,
    cache_dir: Path,
    cloud_max: float,
    temporal_mode: str,
    target: dict,
    reader: Optional[Reader] = None,
) -> tuple[list, list]:
    """Fetch one best-match Landsat scene per CAPA anchor date."""
    scene_arrays: list[dict] = []
    scene_dates: list[date] = []

    for anchor in window.anchor_dates:
        start, end = window.search_range(anchor)
        try:
            scenes = _query_stac(bbox, start, end, cloud_max)
        except RuntimeError:
            scenes = []

        if not scenes:
            continue

        # Report date offset via TemporalWindow warnings
        available_dates = [_parse_date(s["datetime"]) for s in scenes]
        closest_date, _offset = window.closest_match(available_dates, anchor)

        # Pick the scene with the closest acquisition date to the anchor
        target_scene = min(scenes, key=lambda s: abs((_parse_date(s["datetime"]) - anchor).days))
        res = _download_bands(target_scene, bbox, enabled, cache_dir, target=target, reader=reader)
        if res is None:
            continue

        scene_arrays.append(_scene_indices(res, enabled))
        scene_dates.append(closest_date or anchor)

    return scene_arrays, scene_dates


def _fetch_full_range_scenes(
    bbox: tuple,
    date_start: date,
    date_end: date,
    enabled: set,
    cache_dir: Path,
    cloud_max: float,
    temporal_mode: str,
    target: dict,
    reader: Optional[Reader] = None,
) -> tuple[list, list]:
    """Fallback: fetch all Landsat scenes in the full date range."""
    try:
        scenes = _query_stac(bbox, date_start, date_end, cloud_max)
    except RuntimeError:
        return [], []

    if not scenes:
        return [], []

    if temporal_mode == "single":
        scenes = [min(scenes, key=lambda s: s["cloud_cover"])]

    scene_arrays: list[dict] = []
    scene_dates: list[date] = []
    for scene in scenes:
        acq_date = _parse_date(scene["datetime"])
        res = _download_bands(scene, bbox, enabled, cache_dir, target=target, reader=reader)
        if res is None:
            continue
        scene_arrays.append(_scene_indices(res, enabled))
        scene_dates.append(acq_date)

    return scene_arrays, scene_dates


def _scene_indices(res: dict, enabled: set) -> dict:
    out = _compute_indices(res["bands"], enabled)
    out["_georef"] = res["georef"]
    return out


# ---------------------------------------------------------------------------
# Campaign-matched LST
# ---------------------------------------------------------------------------

def campaign_lst(
    bbox: tuple[float, float, float, float],
    campaign_date: date,
    *,
    window_days: int = 16,
    min_clear: float = 0.5,
    n_scenes: int = 1,
    prefer_clear: float = 0.8,
    clear_tolerance: float = CLEAR_TOLERANCE,
    mask_buffer_m: float = MASK_BUFFER_M,
    platforms: tuple[str, ...] = LST_PLATFORMS,
    cloud_cover_max: float = 100.0,
    target: Optional[dict] = None,
    study_mask: Optional[np.ndarray] = None,
    qa_bits: tuple[str, ...] = QA_MASKED,
    search: Optional[Search] = None,
    reader: Optional[Reader] = None,
    cache_dir: Optional[Path] = None,
) -> dict:
    """LST of the clear Landsat overpass(es) nearest a campaign date.

    Method.  Scenes of ``platforms`` acquired within ``campaign_date ± window_days`` are grouped into
    **overpasses**: the rows of one WRS-2 path on one date (:func:`wrs_path_row`), acquired seconds apart.  A city
    can straddle two rows, and judging each row alone loses part of it: Albuquerque 2021-07-05's row 35 ends inside
    the grid (8.7% of the driven cells without LST, all in the south) and row 36 of the same overpass was never
    inspected; Atlanta 2021-08-29's rows 36 and 37 are 0.60 and 0.63 clear each.  Overpasses are ordered by their
    distance in days from the campaign (ties broken by scene id).  For each, in that order, every row's QA_PIXEL is
    read over the study area and reprojected onto ``target`` (nearest); each row is masked on its own (``qa_bits``)
    and the overpass's mosaic is clear wherever any row is.  Its **clear fraction** is the share of study-area
    cells clear in the mosaic, and its **footprint fraction** the share inside any row's footprint (QA fill not
    set), so a scene that misses part of the city is not mistaken for a cloudy one (Seattle 2020-07-29: the east of
    the King County grid lies outside path 047; row 26 holds none of the study area and row 27 is 0.76 clear).  A
    row with no clear pixel on the grid adds no LST, so its thermal band is not read (Seattle's row 26 has 115 clear
    cells in a corner of the grid, outside the study area, so it is read).

    Choice.  With one scene (default) the kept overpass is the **nearest with a clear fraction ≥ ``prefer_clear``**
    (0.8): the nearest-date rule kept a 60%-clear scene over a fully clear one 8 days further (LA 2019: 38% of the
    driven cells lost LST), and a scene a week further is a smaller error than a third of the city missing.  When
    none is that clear it is the **nearest whose clear fraction is within ``clear_tolerance`` (0.10) of the
    clearest's** and ≥ ``min_clear`` (ties: the clearer): the clearest alone kept Arlington's 2021-06-30 (−15 d,
    0.72 clear) over 2021-07-16 (+1 d, 0.64), two more weeks of surface change for 8 points of cover.  With
    ``n_scenes`` > 1 the first ``n_scenes`` overpasses with ≥ ``min_clear`` in date order are kept and their median
    fills each other's gaps.  ``selection`` states the rule that chose.  Scene-level ``eo:cloud_cover`` describes
    the whole 185 km scene, so it only pre-filters (``cloud_cover_max``, default no filter); the study-area clear
    fraction decides.

    LST.  Each kept row's ST_B10 is scaled to kelvin, masked with its own QA (on its own grid and on ``target``) and
    reprojected (bilinear); where rows overlap, any clear value is taken.  LST is also masked within
    ``mask_buffer_m`` (:data:`MASK_BUFFER_M`, 150 m) of cloud (the :data:`CLOUD_BITS` among ``qa_bits``): clear
    LST next to the mask is cool-biased (Charleston, Virginia Beach).  Each row's values are masked within that
    distance of its own cloud, before the overlap takes any clear value (each row's cloud mask is computed on its
    own scene and they differ at cloud edges: in Atlanta 2021-08-29's overlap 0.9% and 1.8% of the cells are cloud
    in one row and clear in the other, and 17 driven cells within 150 m of each row's own cloud but of no cell
    cloudy in both kept LST with a land-cover residual of −2.2 K), and the mosaic within that distance of its
    cloud (where no row is clear: a row's cloud next to the edge of another's footprint).
    Fill is not buffered, so the edges of a footprint keep their pixels, and the clear fraction that chose the
    scene uses the unbuffered mask; a kept overpass's ``clear_fraction_lst`` is the share of study-area cells with
    LST after the buffer.  With more than one overpass the pixel-wise median of the masked overpasses is the LST.

    ``study_mask`` (bool, ``target`` shape) defines the study area; default the target cells inside
    ``bbox``.  ``search`` ``(bbox, start, end) → scenes`` and ``reader`` ``(href, bbox) → (array,
    transform, crs)`` default to the Planetary Computer STAC search with signed assets
    (:func:`search_pc`; no Element84 fallback: its hrefs sit on a requester-pays bucket) and
    :func:`read_window`; tests pass mocks.

    **Failures are not "no clear scene".**  A scene whose read raises is recorded with ``error`` and
    ``error_kind`` (:func:`io_error_kind`: ``"blocked"``, ``"io"``, or ``"data"`` for anything else), and
    ``s3://`` hrefs are refused (``"unreadable"``: a requester-pays bucket needs credentials).  A row whose QA read
    fails is left out of its overpass's mosaic; a chosen overpass whose thermal read fails for any row gives way to
    the next.  ``status`` says what happened: ``"ok"`` (a scene kept), ``"no_scene"`` (none in the window),
    ``"no_clear_scene"`` (every inspected scene read and was too cloudy) or ``"io_error"`` (none kept and at least
    one inspected scene failed on I/O, so whether a clear scene exists is unknown); ``io_errors`` lists those
    failures with the host of each.  Callers must not treat ``"io_error"`` as a finished result.

    Returns ``{"lst_k", "lst_c", "georef", "scenes", "candidates", "overpasses", "campaign_date", "window_days",
    "min_clear", "prefer_clear", "clear_tolerance", "qa_masked", "mask_buffer_m", "mask_buffer_bits", ...}``.
    ``candidates`` are every scene (row) in the window, each with ``id``, ``date``, ``datetime``, ``platform``,
    ``wrs_path``, ``wrs_row``, ``overpass`` (its overpass's label), ``offset_days`` (scene − campaign),
    ``cloud_cover`` (scene-level %), ``clear_fraction`` and ``footprint_fraction`` (study area, this row alone; None
    when the scene was not inspected because enough nearer overpasses were kept), ``inspected`` and ``selected``.
    ``overpasses`` hold the same per overpass, with the mosaic's ``clear_fraction`` and ``footprint_fraction``,
    ``ids`` (its rows; once inspected, those whose QA read and that hold a clear pixel on the grid: the rows the
    mosaic is made of), ``id`` (those ids joined by "+": the scene id when there is one row), ``wrs_rows`` and
    ``cloud_cover`` (the largest of its rows').  ``scenes`` are the kept overpasses (with ``clear_fraction_lst``), so
    a scene's ``id``, ``date`` and ``clear_fraction`` read as before.
    ``lst_k`` / ``lst_c`` are None when no overpass is clear enough.
    """
    from datetime import timedelta

    target = target or target_grid(bbox)
    if study_mask is None:
        study_mask = bbox_mask(target, bbox)
    study_mask = np.asarray(study_mask, dtype=bool)
    if study_mask.shape != tuple(target["shape"]):
        raise ValueError(f"study_mask has shape {study_mask.shape}, the target grid {tuple(target['shape'])}")
    if not study_mask.any():
        raise ValueError("the study area has no cells on the target grid")
    start, end = campaign_date - timedelta(days=int(window_days)), campaign_date + timedelta(days=int(window_days))
    search = search or (lambda b, d0, d1: search_pc(b, d0, d1))
    reader = reader or read_window
    scenes = [s for s in search(bbox, start, end) or []
              if (not platforms or str(s.get("platform") or "").lower() in platforms)
              and float(s.get("cloud_cover", 0.0) if s.get("cloud_cover") is not None else 0.0) <= cloud_cover_max]
    for s in scenes:
        s["_date"] = _parse_date(s["datetime"])
        s["_offset"] = (s["_date"] - campaign_date).days
        s["_wrs"] = wrs_path_row(s)
    scenes = [s for s in scenes if abs(s["_offset"]) <= window_days]
    scenes.sort(key=lambda s: (abs(s["_offset"]), str(s.get("id"))))
    groups: dict = {}                                # an overpass: the rows of one path on one date
    for s in scenes:
        key = (str(s.get("platform") or "").lower(), s["_wrs"][0], s["_date"]) if s["_wrs"] else ("id", s.get("id"))
        groups.setdefault(key, []).append(s)
    passes = list(groups.values())                   # nearest first (the scenes' order)
    T = target["transform"]
    pixel_m = (math.hypot(T.b, T.e), math.hypot(T.a, T.d))
    buffer_bits = tuple(b for b in CLOUD_BITS if b in qa_bits)

    candidates, overpasses, kept, stack, ok = [], [], [], [], []
    single = int(n_scenes) == 1

    def enough() -> bool:
        if single:
            return any(op["clear_fraction"] >= prefer_clear for op, _ in ok)
        return len(ok) >= int(n_scenes)

    def failed(rec: dict, exc: Exception, href) -> None:
        rec["error"] = str(exc).splitlines()[0][:200] if str(exc) else repr(exc)[:200]
        rec["error_kind"] = io_error_kind(exc) or "data"
        rec["host"] = href_host(href)
        log.warning("landsat: skipping %s (%s: %s)", rec["id"], rec["error_kind"], rec["error"])

    for g in passes:                                 # QA of each overpass, nearest first, until the rule is met
        g.sort(key=lambda s: ((s["_wrs"] or (0, 0))[1], str(s.get("id"))))
        s0 = g[0]
        label = (f"{s0.get('platform')} {s0['_wrs'][0]:03d} {s0['_date'].isoformat()}" if s0["_wrs"]
                 else str(s0.get("id")))
        recs = [{"id": s.get("id"), "date": s["_date"].isoformat(), "datetime": s.get("datetime"),
                 "platform": s.get("platform"), "wrs_path": s["_wrs"][0] if s["_wrs"] else None,
                 "wrs_row": s["_wrs"][1] if s["_wrs"] else None, "overpass": label, "offset_days": int(s["_offset"]),
                 "cloud_cover": s.get("cloud_cover"), "clear_fraction": None, "footprint_fraction": None,
                 "inspected": False, "selected": False} for s in g]
        candidates.extend(recs)
        cc = [r["cloud_cover"] for r in recs if r["cloud_cover"] is not None]
        op = {"id": "+".join(str(r["id"]) for r in recs), "ids": [r["id"] for r in recs], "overpass": label,
              "date": recs[0]["date"], "datetime": recs[0]["datetime"], "platform": recs[0]["platform"],
              "wrs_path": recs[0]["wrs_path"], "wrs_rows": [r["wrs_row"] for r in recs],
              "offset_days": recs[0]["offset_days"], "cloud_cover": max(cc) if cc else None, "clear_fraction": None,
              "footprint_fraction": None, "inspected": False, "selected": False}
        overpasses.append(op)
        if enough():
            continue
        op["inspected"] = True
        rows = []
        for s, rec in zip(g, recs):                  # each row QA-masked on its own
            rec["inspected"] = True
            qa_href = _asset_href(s, "qa")
            st_href = _asset_href(s, "thermal")
            if not qa_href or not st_href:
                rec["error"], rec["error_kind"] = "scene has no qa_pixel or lwir11 asset", "data"
                continue
            if any(str(h).lower().startswith("s3://") for h in (qa_href, st_href)):
                rec["error"] = "s3:// href (requester-pays bucket, e.g. the Element84 fallback): not readable here"
                rec["error_kind"] = "unreadable"
                rec["host"] = str(qa_href).split("/")[2] if "//" in str(qa_href) else None
                continue
            try:
                qa_arr, qa_tr, qa_crs = _cached_read(s, "qa_pixel", qa_href, bbox, reader, cache_dir)
                qa_t = reproject_array(np.asarray(qa_arr).astype(np.uint16), qa_tr, qa_crs, target, "nearest",
                                       nodata=1)
                bad = qa_mask(qa_t, qa_bits)
            except Exception as exc:                 # noqa: BLE001 - a broken scene is recorded, not fatal
                failed(rec, exc, qa_href)
                continue
            inside = ~qa_mask(qa_t, ("fill",))
            rec["clear_fraction"] = float(np.mean(~bad[study_mask]))
            rec["footprint_fraction"] = float(np.mean(inside[study_mask]))
            rows.append({"scene": s, "rec": rec, "st_href": st_href, "qa": (qa_arr, qa_tr, qa_crs), "bad": bad,
                         "inside": inside, "cloud": qa_mask(qa_t, buffer_bits)})
        if not rows:
            continue
        clear = np.logical_or.reduce([~r["bad"] for r in rows])          # overlaps: clear in any row
        inside = np.logical_or.reduce([r.pop("inside") for r in rows])
        cloud = np.logical_or.reduce([r["cloud"] for r in rows]) & ~clear       # the mosaic's cloud, to buffer
        used = [r for r in rows if not r["bad"].all()] or rows             # a row with no clear pixel adds nothing
        op["ids"] = [r["rec"]["id"] for r in used]
        op["id"] = "+".join(str(i) for i in op["ids"])
        op["wrs_rows"] = [r["rec"]["wrs_row"] for r in used]
        op["clear_fraction"] = float(np.mean(clear[study_mask]))
        op["footprint_fraction"] = float(np.mean(inside[study_mask]))
        if op["clear_fraction"] >= min_clear:
            ok.append((op, {"rows": used, "clear": clear, "cloud": cloud}))
    if single:
        good = [x for x in ok if x[0]["clear_fraction"] >= prefer_clear]
        if good:
            first = [good[0]]
            selection = f"the nearest scene ≥ {prefer_clear:.0%} clear"
        else:                                        # the nearest of those about as clear as the clearest
            best = max((x[0]["clear_fraction"] for x in ok), default=1.0)
            first = sorted([x for x in ok if x[0]["clear_fraction"] >= best - clear_tolerance - 1e-9],
                           key=lambda x: (abs(x[0]["offset_days"]), -x[0]["clear_fraction"]))
            selection = (f"no scene ≥ {prefer_clear:.0%} clear: the nearest ≥ {min_clear:.0%} clear within "
                         f"{100 * clear_tolerance:.0f} points of the clearest ({best:.0%})")
        taken = {id(x) for x in first}
        order = first + sorted([x for x in ok if id(x) not in taken],
                               key=lambda x: (-x[0]["clear_fraction"], abs(x[0]["offset_days"])))
    else:
        order = ok
        selection = f"the {int(n_scenes)} nearest scenes ≥ {min_clear:.0%} clear (median)"
    for op, d in order:                              # thermal of the chosen overpass(es); the next if one fails
        if len(kept) >= int(n_scenes):
            break
        k_t, r = None, None
        try:
            for r in d["rows"]:
                qa_arr, qa_tr, qa_crs = r["qa"]
                st_arr, st_tr, st_crs = _cached_read(r["scene"], "lwir11", r["st_href"], bbox, reader, cache_dir)
                k_native = st_to_kelvin(st_arr)
                qa_native = _qa_on(st_arr, st_tr, st_crs, qa_arr, qa_tr, qa_crs)
                k_native[qa_mask(qa_native, qa_bits)] = np.nan
                k_row = reproject_array(k_native, st_tr, st_crs, target, "bilinear")
                k_row[r["bad"] | buffer_mask(r["cloud"], mask_buffer_m, pixel_m)] = np.nan   # its own cloud edges
                k_t = k_row if k_t is None else np.where(np.isfinite(k_t), k_t, k_row)
        except Exception as exc:                     # noqa: BLE001 - a broken scene is recorded, not fatal
            failed(r["rec"], exc, r["st_href"])
            op["error"] = f"{r['rec']['id']}: {r['rec']['error']}"
            continue
        k_t[buffer_mask(d["cloud"], mask_buffer_m, pixel_m)] = np.nan          # and a row's cloud past another's edge
        op["selected"] = True
        for r in d["rows"]:
            r["rec"]["selected"] = True
        op["clear_fraction_lst"] = float(np.mean(np.isfinite(k_t[study_mask])))
        kept.append(op)
        stack.append(k_t)
    lst_k = None
    if stack:
        with np.errstate(all="ignore"):
            lst_k = stack[0] if len(stack) == 1 else np.nanmedian(np.stack(stack), axis=0)
    io_errors = [{"id": c["id"], "kind": c["error_kind"], "host": c.get("host"), "error": c["error"]}
                 for c in candidates if c.get("error_kind") in ("blocked", "io", "unreadable")]
    if kept:
        status = "ok"
    elif not candidates:
        status = "no_scene"
    elif io_errors:
        status = "io_error"
    else:
        status = "no_clear_scene"
    return {"lst_k": lst_k, "lst_c": None if lst_k is None else lst_k - KELVIN, "georef": target,
            "scenes": kept, "candidates": candidates, "overpasses": overpasses,
            "campaign_date": campaign_date.isoformat(), "window_days": int(window_days), "min_clear": float(min_clear),
            "qa_masked": list(qa_bits), "mask_buffer_m": float(mask_buffer_m), "mask_buffer_bits": list(buffer_bits),
            "platforms": list(platforms), "status": status, "io_errors": io_errors,
            "selection": selection if kept else None, "prefer_clear": float(prefer_clear),
            "clear_tolerance": float(clear_tolerance)}


def _qa_on(arr, tr, crs, qa_arr, qa_tr, qa_crs) -> np.ndarray:
    """QA_PIXEL on the pixel grid of ``arr`` (the same grid in a real scene; reprojected otherwise)."""
    qa = np.asarray(qa_arr)
    if qa.shape == np.shape(arr) and tuple(qa_tr) == tuple(tr) and str(qa_crs) == str(crs):
        return qa
    dst = {"transform": tr, "crs": crs, "shape": np.shape(arr)}
    return reproject_array(qa.astype(np.uint16), qa_tr, qa_crs, dst, "nearest", nodata=1)


def _asset_href(scene: dict, band: str) -> str | None:
    assets = scene.get("assets") or {}
    key = BAND_ASSETS[band]
    asset = (assets.get(key) or assets.get(key.lower())
             # thermal band fallback: lwir11 (E84) ↔ lwir (older metadata)
             or (assets.get("lwir11") if key in ("lwir", "lwir11") else None)
             or (assets.get("lwir") if key in ("lwir", "lwir11") else None)
             or assets.get(f"ST_{key.upper()}") or (assets.get("QA_PIXEL") if band == "qa" else None))
    if asset is None:
        return None
    return asset.get("href", asset.get("url", "")) or None


def _cached_read(scene: dict, asset_key: str, href: str, bbox, reader: Reader, cache_dir: Optional[Path]):
    """``reader(href, bbox)`` with an on-disk cache keyed by scene, asset and bbox (array + transform + CRS)."""
    if cache_dir is None or reader is not read_window:
        return reader(href, tuple(bbox))
    from affine import Affine

    tag = hashlib.sha1(json.dumps([round(float(v), 6) for v in bbox]).encode()).hexdigest()[:10]
    path = Path(cache_dir) / f"{scene.get('id')}_{asset_key}_{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["arr"], Affine(*z["transform"][:6]), str(z["crs"])
    arr, tr, crs = reader(href, tuple(bbox))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, arr=arr, transform=np.asarray(tuple(tr)[:6], dtype=np.float64), crs=np.asarray(crs))
    return arr, tr, crs


# ---------------------------------------------------------------------------
# STAC query
# ---------------------------------------------------------------------------

def _query_stac(
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
    cloud_max: float,
) -> list[dict]:
    """Search for Landsat Collection 2 Level-2 scenes.

    Tries Microsoft Planetary Computer first (signed Azure Blob URLs), then
    the AWS Element84 Earth-Search endpoint (metadata only: its data are on a
    requester-pays bucket).  Cloud-cover filtering is done client-side after
    the response to avoid relying on optional STAC API query extensions that
    may not be enabled on all servers.
    """
    endpoints = [
        (PC_STAC,        _PC_COLLECTION,  True),   # (url, collection, needs_signing)
        (ELEMENT84_STAC, _E84_COLLECTION, False),
    ]

    last_error: Exception | None = None
    for url, collection, sign in endpoints:
        try:
            scenes = _query_stac_endpoint(url, collection, bbox, date_start, date_end,
                                          sign_assets=sign)
            return [s for s in scenes if s["cloud_cover"] <= cloud_max]
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"STAC query failed on all endpoints: {last_error}"
    ) from last_error


def search_pc(bbox: tuple[float, float, float, float], date_start: date, date_end: date) -> list[dict]:
    """Landsat C2 L2 scenes from the Planetary Computer STAC API with **signed** asset hrefs (SAS token from
    planetarycomputer.microsoft.com; the COGs are on landsateuwest.blob.core.windows.net).  Unlike
    :func:`_query_stac` there is no fallback: a failed search or a failed signing raises."""
    return _query_stac_endpoint(PC_STAC, _PC_COLLECTION, bbox, date_start, date_end, sign_assets=True,
                                require_signing=True)


def _query_stac_endpoint(
    url: str,
    collection: str,
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
    *,
    sign_assets: bool = False,
    require_signing: bool = False,
) -> list[dict]:
    """Send a minimal STAC search (no query/fields extensions) to one endpoint (following ``next`` links)."""
    minx, miny, maxx, maxy = bbox
    payload = {
        "collections": [collection],
        "bbox": [minx, miny, maxx, maxy],
        "datetime": f"{date_start.isoformat()}T00:00:00Z/{date_end.isoformat()}T23:59:59Z",
        "limit": 200,
    }
    items: list = []
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SPARC-DataCollection/1.0",
        },
    )
    for _page in range(20):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"STAC query failed ({url}): {exc}") from exc
        items.extend(data.get("features", []))
        nxt = next((ln for ln in data.get("links", []) if ln.get("rel") == "next"), None)
        if not nxt or not nxt.get("href"):
            break
        body = nxt.get("body")
        req = urllib.request.Request(
            nxt["href"],
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json", "User-Agent": "SPARC-DataCollection/1.0"},
            method=str(nxt.get("method", "POST" if body is not None else "GET")).upper(),
        )

    # Sign Planetary Computer items so Azure Blob URLs include SAS tokens.
    # This is a no-op if planetary_computer is not installed.
    if sign_assets:
        try:
            import planetary_computer as pc  # type: ignore[import]
        except ImportError as exc:
            if require_signing:
                raise RuntimeError("Planetary Computer assets need signing: install planetary-computer") from exc
            pc = None  # Fall through; download will likely fail but won't crash here
        if pc is not None:
            try:
                items = [pc.sign(item) for item in items]
            except Exception as exc:
                raise RuntimeError(f"Planetary Computer SAS signing failed ({PC_SAS_TOKEN}): {exc}") from exc

    scenes = []
    for item in items:
        props = item.get("properties", {})
        scenes.append({
            "id": item["id"],
            "datetime": props.get("datetime", ""),
            "cloud_cover": props.get("eo:cloud_cover", 100),
            "platform": props.get("platform"),
            "assets": item.get("assets", {}),
        })
    return scenes


# ---------------------------------------------------------------------------
# Band download
# ---------------------------------------------------------------------------

def _download_bands(
    scene: dict,
    bbox: tuple[float, float, float, float],
    enabled: set[str],
    cache_dir: Path,
    *,
    target: Optional[dict] = None,
    reader: Optional[Reader] = None,
    qa_bits: tuple[str, ...] = QA_MASKED,
) -> Optional[dict]:
    """Bands of one scene, cloud-masked and on the ``target`` grid.

    Returns ``{"bands": {name: array}, "georef": target, "clear_fraction": f}`` (reflectance for the
    SR bands, ``lst_kelvin`` / ``lst`` (°C) for the thermal band), or None when no band could be read."""
    if reader is None:
        try:
            import rasterio  # noqa: F401
        except ImportError:
            raise ImportError("rasterio is required for Landsat band download: pip install rasterio")
        reader = read_window
    target = target or target_grid(bbox)

    qa_t = None
    qa_href = _asset_href(scene, "qa")
    if qa_href:
        try:
            qa_arr, qa_tr, qa_crs = _cached_read(scene, "qa_pixel", qa_href, bbox, reader, cache_dir)
            qa_t = reproject_array(np.asarray(qa_arr).astype(np.uint16), qa_tr, qa_crs, target, "nearest", nodata=1)
        except Exception as exc:                     # noqa: BLE001
            log.warning("landsat: QA_PIXEL unavailable for %s (%s); scene not cloud-masked", scene.get("id"), exc)
    bad = qa_mask(qa_t, qa_bits) if qa_t is not None else None

    bands: dict[str, np.ndarray] = {}
    for band_name, asset_key in BAND_ASSETS.items():
        if band_name == "qa" or not _band_needed(band_name, enabled):
            continue
        href = _asset_href(scene, band_name)
        if not href:
            continue
        try:
            arr, tr, crs = _cached_read(scene, asset_key, href, bbox, reader, cache_dir)
        except Exception as _dl_exc:
            import warnings
            warnings.warn(f"Landsat band download failed for {band_name}: {_dl_exc}")
            continue
        phys = st_to_kelvin(arr) if band_name == "thermal" else sr_to_reflectance(arr)
        v = reproject_array(phys, tr, crs, target, "bilinear")
        if bad is not None:
            v[bad] = np.nan
        bands[band_name] = v

    if "thermal" in bands:
        bands["lst_kelvin"] = bands.pop("thermal")
        bands["lst"] = bands["lst_kelvin"] - KELVIN

    if not bands:
        return None
    clear = float(np.mean(~bad)) if bad is not None else None
    return {"bands": bands, "georef": target, "clear_fraction": clear}


# ---------------------------------------------------------------------------
# Spectral index formulas
# ---------------------------------------------------------------------------

def _safe_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with NaN for zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where((a + b) != 0, (a - b) / (a + b), np.nan)
    return result.astype(np.float32)


SPECTRAL_INDEX_FORMULAS: dict = {
    "ndvi":  lambda b: _safe_norm(b["nir"],   b["red"]),
    "ndbi":  lambda b: _safe_norm(b["swir1"], b["nir"]),
    "mndwi": lambda b: _safe_norm(b["green"], b["swir1"]),
    "ndwi":  lambda b: _safe_norm(b["green"], b["nir"]),
    "savi":  lambda b: ((b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5)) * 1.5,
    "evi":   lambda b: 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6*b["red"] - 7.5*b["blue"] + 1),
    "ndbai": lambda b: _safe_norm(b["swir1"], b["swir2"]),
    "nbr":   lambda b: _safe_norm(b["nir"],   b["swir2"]),
    "ndmi":  lambda b: _safe_norm(b["nir"],   b["swir1"]),
    "ui":    lambda b: _safe_norm(b["swir2"], b["nir"]),
    "bsi":   lambda b: _safe_norm(b["red"] + b["swir1"], b["nir"] + b["blue"]),
    "albedo": lambda b: (
        0.356*b["blue"] + 0.130*b["red"] + 0.373*b["nir"] +
        0.085*b["swir1"] + 0.072*b["swir2"] - 0.0018
    ),
}


def _band_needed(band_name: str, enabled: set[str]) -> bool:
    """Return True if *band_name* is used by any enabled index."""
    usage = {
        "blue":    {"bsi", "evi", "albedo"},
        "green":   {"mndwi", "ndwi"},
        "red":     {"ndvi", "savi", "evi", "bsi", "albedo"},
        "nir":     {"ndvi", "ndbi", "mndwi", "ndwi", "savi", "evi", "nbr", "ndmi", "ui", "bsi", "albedo"},
        "swir1":   {"ndbi", "mndwi", "ndbai", "ndmi", "bsi", "albedo"},
        "swir2":   {"ndbai", "nbr", "ui", "albedo"},
        "thermal": {"lst", "lst_kelvin"},
    }
    return bool(enabled & usage.get(band_name, set()))


def _compute_indices(
    bands: dict[str, np.ndarray],
    enabled: set[str],
) -> dict[str, np.ndarray]:
    """Indices from surface-reflectance bands; ``lst`` (°C) and ``lst_kelvin`` passed through."""
    indices: dict[str, np.ndarray] = {}
    for name, formula in SPECTRAL_INDEX_FORMULAS.items():
        if name not in enabled:
            continue
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                indices[name] = formula(bands).astype(np.float32)
        except KeyError:
            pass  # required band not available for this scene
    if "lst" in enabled and "lst" in bands:
        indices["lst"] = bands["lst"]
        if "lst_kelvin" in bands:
            indices["lst_kelvin"] = bands["lst_kelvin"]
    return indices


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def _median_composite(scene_arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pixel-wise median across scenes for each index.

    Every scene must already be on one grid (``download_landsat`` reprojects each onto the target
    grid), so pixels of different scenes are the same place on the ground."""
    composite: dict[str, np.ndarray] = {}
    all_keys = {k for s in scene_arrays for k in s if not str(k).startswith("_")}
    georefs = [s.get("_georef") for s in scene_arrays if isinstance(s, dict) and s.get("_georef")]
    if georefs and any(_georef_key(g) != _georef_key(georefs[0]) for g in georefs):
        raise ValueError("scenes are on different grids: reproject them onto one target grid first")
    for key in all_keys:
        arrays = [s[key] for s in scene_arrays if key in s and s[key] is not None]
        if not arrays:
            continue
        shapes = {a.shape for a in arrays}
        if len(shapes) > 1:
            raise ValueError(f"index {key!r}: scenes have different shapes {sorted(shapes)} "
                             "(they must share the target grid)")
        stack = np.stack(arrays, axis=0).astype(np.float32)
        with np.errstate(all="ignore"):
            composite[key] = np.nanmedian(stack, axis=0)
    if georefs:
        composite["_georef"] = georefs[0]
    return composite


def _georef_key(g: dict) -> tuple:
    return (tuple(round(float(v), 6) for v in tuple(g["transform"])[:6]), str(g["crs"]), tuple(g["shape"]))


def _assign_arrays_to_fishnet(
    fishnet_gdf: object,
    index_arrays: dict[str, np.ndarray],
    georef: Optional[dict] = None,
) -> object:
    """Assign index arrays to fishnet cells by sampling the pixel under each cell centroid.

    The arrays must carry their georeference (``index_arrays["_georef"]`` or ``georef``: ``{"transform",
    "crs"}``).  Centroids are computed in the fishnet's projected CRS, projected into the array's CRS and
    looked up through its affine transform; cells outside the array get NaN (no clipping to the edge).
    """
    from pyproj import Transformer

    georef = georef or index_arrays.get("_georef")
    if georef is None:
        raise ValueError("Landsat arrays carry no georeference (_georef: transform + CRS); placing them by the "
                         "fishnet's bounds would misplace every pixel")
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    centroids = gdf.geometry.centroid  # type: ignore[union-attr]
    tr = Transformer.from_crs(gdf.crs, georef["crs"], always_xy=True)  # type: ignore[union-attr]
    xs, ys = tr.transform(centroids.x.to_numpy(float), centroids.y.to_numpy(float))
    for name, arr in index_arrays.items():
        if str(name).startswith("_"):
            continue
        if arr is None or np.size(arr) == 0:
            gdf[name] = float("nan")  # type: ignore[index]
            continue
        gdf[name] = sample_raster(np.asarray(arr), georef["transform"], np.asarray(xs), np.asarray(ys))  # type: ignore[index]
    return gdf


def _build_panel(
    fishnet_gdf: object,
    scene_arrays: list[dict[str, np.ndarray]],
    scene_dates: list[date],
) -> object:
    """Build a multi-date panel GeoDataFrame (one copy per scene date)."""
    import pandas as pd
    import geopandas as gpd

    frames = []
    for arr_dict, acq_date in zip(scene_arrays, scene_dates):
        gdf = _assign_arrays_to_fishnet(fishnet_gdf, arr_dict)
        gdf["scene_date"] = acq_date.isoformat()  # type: ignore[index]
        frames.append(gdf)

    if not frames:
        return fishnet_gdf

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=fishnet_gdf.crs,  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_date(dt_str: str) -> date:
    return date.fromisoformat(dt_str[:10])


def _default_cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "landsat"
    p.mkdir(parents=True, exist_ok=True)
    return p
