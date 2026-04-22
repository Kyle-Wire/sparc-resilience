"""
Context-layer catalog (Phase 18)
================================

Catalog of public, no-key XYZ tile layers + WMS overlays the desktop
can pop on top of the basemap. Each entry is annotated with the
domains it's most relevant to so the UI can suggest defaults.

We deliberately stick to layers with stable XYZ endpoints — anything
that needs an API key, paid hosting, or per-bbox export gymnastics is
left out for now (and noted in the README of this module's docstring).
"""

from __future__ import annotations

from typing import Any


# ------------------------------------------------------------------
# Catalog
# ------------------------------------------------------------------
# kind: "basemap" (replaces base) | "overlay" (drawn on top, transparent)
# url_template: maplibre-compatible {z}/{x}/{y} XYZ URL
# domains: list of project domains where this layer is recommended
# default_for: domains where the layer should auto-toggle on
# opacity_default: 0..1
# tile_size: 256 or 512
# max_zoom: int

CONTEXT_LAYERS: list[dict[str, Any]] = [
    # ---------------- basemaps ----------------
    {
        "id": "carto-positron",
        "name": "Carto Positron (light)",
        "kind": "basemap",
        "url_template": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap, © CARTO",
        "description": "Default warm-paper-friendly basemap",
        "domains": ["*"],
        "default_for": ["*"],
        "opacity_default": 1.0,
        "tile_size": 256,
        "max_zoom": 19,
    },
    {
        "id": "osm",
        "name": "OpenStreetMap",
        "kind": "basemap",
        "url_template": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "description": "Standard OSM raster tiles",
        "domains": ["*"],
        "default_for": [],
        "opacity_default": 1.0,
        "tile_size": 256,
        "max_zoom": 19,
    },
    {
        "id": "usgs-topo",
        "name": "USGS Topographic",
        "kind": "basemap",
        "url_template": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        "attribution": "USGS The National Map",
        "description": "USGS topographic basemap (US-focused)",
        "domains": ["uhi", "drought", "geotechnical", "groundwater", "noise", "*"],
        "default_for": [],
        "opacity_default": 1.0,
        "tile_size": 256,
        "max_zoom": 16,
    },
    {
        "id": "usgs-imagery",
        "name": "USGS Imagery",
        "kind": "basemap",
        "url_template": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        "attribution": "USGS / NAIP",
        "description": "USGS aerial imagery (Y/X order)",
        "domains": ["*"],
        "default_for": [],
        "opacity_default": 1.0,
        "tile_size": 256,
        "max_zoom": 17,
    },
    # ---------------- overlays ----------------
    {
        "id": "usgs-hillshade",
        "name": "Hillshade",
        "kind": "overlay",
        "url_template": "https://services.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri / USGS",
        "description": "Multi-directional hillshade for terrain context",
        "domains": ["geotechnical", "drought", "groundwater", "uhi"],
        "default_for": ["geotechnical"],
        "opacity_default": 0.6,
        "tile_size": 256,
        "max_zoom": 16,
    },
    {
        "id": "esri-natgeo",
        "name": "Nat Geo style overlay",
        "kind": "overlay",
        "url_template": "https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri / National Geographic",
        "description": "Topo+labels overlay for reference",
        "domains": ["*"],
        "default_for": [],
        "opacity_default": 0.5,
        "tile_size": 256,
        "max_zoom": 16,
    },
    {
        "id": "carto-labels",
        "name": "Place labels",
        "kind": "overlay",
        "url_template": "https://basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
        "attribution": "© CARTO",
        "description": "Transparent place labels (city/road names)",
        "domains": ["*"],
        "default_for": [],
        "opacity_default": 0.85,
        "tile_size": 256,
        "max_zoom": 19,
    },
    {
        "id": "esri-light-gray-ref",
        "name": "Esri reference (boundaries)",
        "kind": "overlay",
        "url_template": "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri",
        "description": "Boundary + label overlay (admin lines)",
        "domains": ["*"],
        "default_for": [],
        "opacity_default": 0.7,
        "tile_size": 256,
        "max_zoom": 16,
    },
    {
        "id": "esri-ocean-ref",
        "name": "Ocean reference",
        "kind": "overlay",
        "url_template": "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri / NOAA",
        "description": "Ocean place + bathymetry labels",
        "domains": ["coastal", "forcesmip"],
        "default_for": ["coastal"],
        "opacity_default": 0.8,
        "tile_size": 256,
        "max_zoom": 13,
    },
    {
        "id": "openseamap",
        "name": "OpenSeaMap (nautical)",
        "kind": "overlay",
        "url_template": "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        "attribution": "© OpenSeaMap",
        "description": "Nautical seamarks (buoys, depths, harbors)",
        "domains": ["coastal"],
        "default_for": [],
        "opacity_default": 0.85,
        "tile_size": 256,
        "max_zoom": 18,
    },
]


# Domains where roadmap layers (no public XYZ available) would belong;
# we surface these in the catalog as a hint for future work, but mark
# `available=False` so the UI can render them disabled.
ROADMAP_LAYERS: list[dict[str, Any]] = [
    {
        "id": "fema-nfhl",
        "name": "FEMA flood zones (NFHL)",
        "kind": "overlay",
        "url_template": None,
        "attribution": "FEMA",
        "description": "Roadmap: requires ArcGIS export proxy (no public XYZ).",
        "domains": ["coastal", "geotechnical", "*"],
        "default_for": [],
        "available": False,
    },
    {
        "id": "nlcd-landcover",
        "name": "NLCD land cover",
        "kind": "overlay",
        "url_template": None,
        "attribution": "MRLC / USGS",
        "description": "Roadmap: served via WMS only; needs proxy for tile fetch.",
        "domains": ["uhi", "drought", "*"],
        "default_for": [],
        "available": False,
    },
    {
        "id": "acs-tracts",
        "name": "ACS demographics by tract",
        "kind": "overlay",
        "url_template": None,
        "attribution": "US Census",
        "description": "Roadmap: needs vector tiles + ACS join (we already have area-aggregate via /equity/census).",
        "domains": ["*"],
        "default_for": [],
        "available": False,
    },
]


def _filter_for_domain(domain: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in CONTEXT_LAYERS:
        doms = entry.get("domains") or []
        if domain is None or "*" in doms or domain in doms:
            e = dict(entry)
            e["available"] = True
            e["recommended"] = bool(domain) and domain in (entry.get("default_for") or [])
            out.append(e)
    return out


def get_catalog(domain: str | None = None) -> dict[str, Any]:
    """Return the layer catalog, optionally filtered + annotated for *domain*."""
    layers = _filter_for_domain(domain)
    roadmap = [dict(r, available=False) for r in ROADMAP_LAYERS]
    defaults = [layer["id"] for layer in layers if layer.get("recommended")]
    return {
        "domain": domain,
        "layers": layers,
        "roadmap": roadmap,
        "defaults": defaults,
        "count": len(layers),
    }


__all__ = ["CONTEXT_LAYERS", "ROADMAP_LAYERS", "get_catalog"]
