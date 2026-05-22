"""
boundary.py — Study-area boundary resolution for SPARC data collection.

Accepts three input modes:
  1. Place name string  → Nominatim (OpenStreetMap) returns the polygon
  2. Local file path    → GeoJSON / Shapefile / GeoPackage read via geopandas
  3. Drawn GeoJSON dict → constructed directly into a GeoDataFrame

All paths return a ``BoundaryResult`` with the boundary GeoDataFrame in
EPSG:4326 and a bounding-box tuple for downstream API clipping.

No other collect sub-modules are imported here.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Union

HTTP_TIMEOUT = 15.0
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SPARC-DataCollection/1.0 (github.com/sparc-resilience)"


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class BoundaryResult:
    """Resolved study-area boundary.

    Attributes
    ----------
    gdf : GeoDataFrame
        Single-row GeoDataFrame (EPSG:4326) containing the boundary polygon.
    bbox : tuple[float, float, float, float]
        (minx, miny, maxx, maxy) in EPSG:4326 — used by all fetch calls.
    source : str
        Human-readable description of where the boundary came from.
    place_name : str | None
        Canonical place name returned by Nominatim, or None for file/draw inputs.
    """

    gdf: object          # gpd.GeoDataFrame — typed as object to avoid hard import
    bbox: tuple[float, float, float, float]
    source: str
    place_name: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_boundary(
    place_name: str | None = None,
    file_path: str | Path | None = None,
    geojson: dict | None = None,
) -> BoundaryResult:
    """Resolve a study-area boundary from exactly one of three inputs.

    Parameters
    ----------
    place_name : str, optional
        Free-text place name, e.g. ``"Providence, RI"`` or ``"Chicago"`` or
        ``"Los Angeles County, CA"``.  Resolved via Nominatim.
    file_path : str | Path, optional
        Path to a local vector file (.geojson, .shp, .gpkg, .json).
    geojson : dict, optional
        A GeoJSON FeatureCollection or Feature dict drawn by the user on the
        desktop map.

    Returns
    -------
    BoundaryResult

    Raises
    ------
    ValueError
        If none or more than one input is provided, or if the place name
        returns no results.
    ImportError
        If geopandas is not installed.
    """
    provided = sum(x is not None for x in (place_name, file_path, geojson))
    if provided == 0:
        raise ValueError("Provide exactly one of: place_name, file_path, or geojson")
    if provided > 1:
        raise ValueError("Provide exactly one of: place_name, file_path, or geojson — not multiple")

    if place_name is not None:
        return _from_place_name(place_name)
    if file_path is not None:
        return _from_file(Path(file_path))
    return _from_geojson(geojson)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _from_place_name(name: str) -> BoundaryResult:
    """Query Nominatim for the best-match polygon boundary."""
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except ImportError as exc:
        raise ImportError("geopandas and shapely are required for boundary resolution") from exc

    params = urllib.parse.urlencode({
        "q": name,
        "format": "geojson",
        "polygon_geojson": "1",
        "limit": "1",
        "dedupe": "1",
    })
    url = f"{NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Nominatim request failed for '{name}': {exc}") from exc

    features = data.get("features", [])
    if not features:
        raise ValueError(
            f"No boundary found for '{name}'. Try a more specific name, "
            "e.g. 'Providence, Rhode Island, USA'."
        )

    feat = features[0]
    geom = shape(feat["geometry"])
    canonical_name = (
        feat.get("properties", {}).get("display_name") or name
    )

    gdf = gpd.GeoDataFrame(
        {"place_name": [canonical_name]},
        geometry=[geom],
        crs="EPSG:4326",
    )
    bbox = _bbox_from_gdf(gdf)
    return BoundaryResult(
        gdf=gdf,
        bbox=bbox,
        source=f"Nominatim: {canonical_name}",
        place_name=canonical_name,
    )


def _from_file(path: Path) -> BoundaryResult:
    """Load a local vector file as the boundary."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to load boundary files") from exc

    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Boundary file has no CRS: {path}")
    if str(gdf.crs.to_epsg()) != "4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Dissolve to a single boundary polygon
    dissolved = gdf.dissolve().reset_index(drop=True)
    bbox = _bbox_from_gdf(dissolved)
    return BoundaryResult(
        gdf=dissolved,
        bbox=bbox,
        source=f"Local file: {path.name}",
        place_name=None,
    )


def _from_geojson(geojson: dict) -> BoundaryResult:
    """Construct boundary from a drawn GeoJSON dict."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required for GeoJSON boundary") from exc

    gdf = gpd.GeoDataFrame.from_features(
        geojson.get("features", [geojson]),
        crs="EPSG:4326",
    )
    if gdf.empty:
        raise ValueError("Drawn GeoJSON contains no features")
    dissolved = gdf.dissolve().reset_index(drop=True)
    bbox = _bbox_from_gdf(dissolved)
    return BoundaryResult(
        gdf=dissolved,
        bbox=bbox,
        source="Drawn on map",
        place_name=None,
    )


def _bbox_from_gdf(gdf: object) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) from a GeoDataFrame's total bounds."""
    bounds = gdf.total_bounds  # type: ignore[union-attr]
    return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
