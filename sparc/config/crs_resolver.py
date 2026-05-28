"""sparc.config.crs_resolver — CRS detection and working-CRS resolution.

Public API
----------
detect_crs_from_file(path)
    Read the embedded CRS from a spatial file (shapefile, GeoPackage,
    GeoJSON, GeoTIFF).  Returns an ``"EPSG:NNNN"`` string or ``None`` for
    formats with no embedded CRS (e.g. CSV).

resolve_working_crs(input_epsg, sample_xy=None)
    Given the input CRS, return the projected CRS to use for all spatial
    operations (block sizes, buffers, grid creation, etc.).

    Rules
    -----
    * Metric projected (metres)  → use as-is.
    * Feet-based projected       → use as-is (feet are valid working units).
    * Geographic (degrees)       → auto-select best UTM zone.

    Pass ``sample_xy=(lon, lat)`` (EPSG:4326) to improve UTM zone selection
    when the input CRS is geographic.

crs_unit(epsg)
    Return ``"m"`` (metres), ``"ft"`` (feet), or ``"deg"`` (degrees).

crs_unit_label(epsg)
    Return a human-readable label: ``"meters"``, ``"feet"``, or ``"degrees"``.

best_utm_zone(lon, lat)
    Return the EPSG code for the best UTM zone for the given WGS-84
    longitude/latitude centroid.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# pyproj is a hard dependency for CRS introspection.
# ---------------------------------------------------------------------------
try:
    from pyproj import CRS as _ProjCRS
    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover
    _HAS_PYPROJ = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEOGRAPHIC_UNITS = {"degree", "degrees", "degree (supplier to define representation)"}
_FEET_UNITS = {"us survey foot", "foot", "feet", "foot_us", "foot_survey_us"}
_METRE_UNITS = {"metre", "meter", "m"}


def _unit_for_crs(crs: "_ProjCRS") -> str:
    """Return 'm', 'ft', or 'deg' for a pyproj CRS object."""
    if crs.is_geographic:
        return "deg"
    for ax in crs.axis_info:
        unit = ax.unit_name.lower().strip()
        if any(u in unit for u in _FEET_UNITS):
            return "ft"
        if any(u in unit for u in _METRE_UNITS):
            return "m"
    # Fallback: inspect the linear unit directly
    try:
        lu = crs.linear_units.lower()
        if any(u in lu for u in _FEET_UNITS):
            return "ft"
    except Exception:
        pass
    return "m"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def best_utm_zone(lon: float, lat: float) -> str:
    """Return the EPSG code for the best UTM zone for a WGS-84 centroid.

    Parameters
    ----------
    lon, lat:
        WGS-84 longitude and latitude of the study-area centroid.

    Returns
    -------
    str
        EPSG code string, e.g. ``"EPSG:32619"`` (UTM zone 19N).
    """
    zone = int((lon + 180) / 6) + 1
    # Clamp to valid range 1–60
    zone = max(1, min(60, zone))
    if lat >= 0:
        epsg = 32600 + zone  # Northern hemisphere
    else:
        epsg = 32700 + zone  # Southern hemisphere
    return f"EPSG:{epsg}"


def crs_unit(epsg: str) -> str:
    """Return the linear unit of *epsg*: ``'m'``, ``'ft'``, or ``'deg'``.

    Parameters
    ----------
    epsg:
        An EPSG code string such as ``"EPSG:4326"`` or ``"EPSG:3438"``.

    Raises
    ------
    ImportError
        If pyproj is not installed.
    ValueError
        If the EPSG code cannot be parsed.
    """
    if not _HAS_PYPROJ:  # pragma: no cover
        raise ImportError("pyproj is required for CRS introspection: pip install pyproj")
    crs = _ProjCRS.from_user_input(epsg)
    return _unit_for_crs(crs)


def crs_unit_label(epsg: str) -> str:
    """Return a human-readable unit label for *epsg*: ``'meters'``, ``'feet'``, or ``'degrees'``."""
    unit = crs_unit(epsg)
    return {"m": "meters", "ft": "feet", "deg": "degrees"}.get(unit, "meters")


def detect_crs_from_file(path: str) -> Optional[str]:
    """Read the embedded CRS from a spatial file and return an EPSG string.

    Supported formats (those with embedded CRS metadata):
    - GeoPackage (.gpkg)
    - Shapefile (.shp)
    - GeoJSON (.geojson, .json)
    - GeoTIFF (.tif, .tiff)
    - FlatGeobuf (.fgb)

    Parameters
    ----------
    path:
        Path to the spatial file.

    Returns
    -------
    str or None
        An ``"EPSG:NNNN"`` string if the CRS can be determined, otherwise
        ``None`` (e.g. for CSV files or files with no embedded CRS).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # Raster formats — use rasterio
    if suffix in {".tif", ".tiff", ".img", ".nc"}:
        return _detect_crs_raster(path)

    # Vector / tabular formats — use geopandas
    if suffix in {".shp", ".gpkg", ".geojson", ".json", ".fgb", ".gdb"}:
        return _detect_crs_vector(path)

    # CSV and other non-spatial formats have no embedded CRS
    return None


def resolve_working_crs(
    input_epsg: str,
    sample_xy: Optional[Tuple[float, float]] = None,
) -> str:
    """Determine the projected working CRS for spatial operations.

    Parameters
    ----------
    input_epsg:
        The CRS of the input data (e.g. ``"EPSG:4326"`` or ``"EPSG:3438"``).
    sample_xy:
        Optional ``(lon, lat)`` tuple in WGS-84 used to pick the best UTM
        zone when *input_epsg* is geographic.  When omitted and the input
        is geographic, UTM zone 1N (``EPSG:32601``) is returned as a safe
        placeholder — callers should always supply coordinates if possible.

    Returns
    -------
    str
        A projected EPSG code string (never geographic).  The returned CRS
        has linear units of either metres or feet.

    Raises
    ------
    ImportError
        If pyproj is not installed.
    """
    if not _HAS_PYPROJ:  # pragma: no cover
        raise ImportError("pyproj is required for CRS resolution: pip install pyproj")

    crs = _ProjCRS.from_user_input(input_epsg)

    if crs.is_projected:
        # Already a projected CRS (metres or feet) — use it as-is.
        return input_epsg

    if crs.is_geographic:
        # Need a projected CRS.  Auto-select best UTM zone.
        if sample_xy is not None:
            lon, lat = sample_xy
            return best_utm_zone(lon, lat)
        # No sample coords — warn and return a generic UTM placeholder.
        warnings.warn(
            f"Input CRS {input_epsg!r} is geographic (degrees). "
            "Provide sample_xy=(lon, lat) for accurate UTM zone selection. "
            "Defaulting to EPSG:32601 (UTM zone 1N) as a placeholder.",
            UserWarning,
            stacklevel=2,
        )
        return "EPSG:32601"

    # Geocentric or other unusual CRS — pass through and let downstream tools
    # raise a meaningful error if needed.
    return input_epsg


def validate_crs_input(epsg: str) -> str:
    """Validate that *epsg* is a recognised EPSG code.

    Parameters
    ----------
    epsg:
        Input string, e.g. ``"EPSG:4326"`` or ``"4326"``.

    Returns
    -------
    str
        Normalised ``"EPSG:NNNN"`` string.

    Raises
    ------
    ValueError
        If the code cannot be parsed by pyproj.
    """
    if not _HAS_PYPROJ:  # pragma: no cover
        raise ImportError("pyproj is required: pip install pyproj")
    # Normalise bare integers like "4326" → "EPSG:4326"
    stripped = epsg.strip()
    if stripped.isdigit():
        stripped = f"EPSG:{stripped}"
    try:
        _ProjCRS.from_user_input(stripped)
    except Exception as exc:
        raise ValueError(f"Unrecognised CRS {epsg!r}: {exc}") from exc
    return stripped


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_crs_vector(path: str) -> Optional[str]:
    """Attempt to read CRS from a vector file using geopandas."""
    try:
        import geopandas as gpd  # type: ignore[import]
        gdf = gpd.read_file(path, rows=0)  # read header only
        if gdf.crs is not None:
            # Prefer EPSG authority code; fall back to WKT-derived string.
            epsg = gdf.crs.to_epsg()
            if epsg is not None:
                return f"EPSG:{epsg}"
            # Try authority code from the CRS object
            auth = gdf.crs.to_authority()
            if auth:
                return f"{auth[0]}:{auth[1]}"
    except Exception:
        pass
    return None


def _detect_crs_raster(path: str) -> Optional[str]:
    """Attempt to read CRS from a raster file using rasterio."""
    try:
        import rasterio  # type: ignore[import]
        with rasterio.open(path) as ds:
            if ds.crs is not None:
                epsg = ds.crs.to_epsg()
                if epsg is not None:
                    return f"EPSG:{epsg}"
                # Fallback: use proj_crs
                try:
                    auth = ds.crs.to_authority()
                    if auth:
                        return f"{auth[0]}:{auth[1]}"
                except Exception:
                    pass
    except Exception:
        pass
    return None
