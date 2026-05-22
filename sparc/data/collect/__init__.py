"""
sparc.data.collect
==================

Data collection pipeline for SPARC spatial datasets.

Each sub-module is a deep module with a simple, testable interface.
No sub-module imports another; the assembler composes them.

Modules
-------
boundary    — resolve study-area boundary from place name, file, or GeoJSON
manifest    — variable availability tracking and manifest JSON output
nlcd        — MRLC WCS fetch for NLCD imperviousness, canopy, land cover
era5        — Open-Meteo ERA5 background temperature fetch and downscaling
capa        — NOAA CAPA air temperature aligned to Landsat overpass dates
buildings   — building footprints + heights with fallback chain; H:W SVF
equity      — HOLC redlining, CDC SVI, EJScreen joins to the analysis grid
landsat     — USGS STAC Landsat scenes, spectral indices, LST computation
assembler   — compose all layers onto the 30m fishnet; write GeoParquet output
"""

from .boundary import resolve_boundary, BoundaryResult
from .manifest import VariableManifest, VariableStatus, ManifestEntry

__all__ = [
    "resolve_boundary",
    "BoundaryResult",
    "VariableManifest",
    "VariableStatus",
    "ManifestEntry",
]
