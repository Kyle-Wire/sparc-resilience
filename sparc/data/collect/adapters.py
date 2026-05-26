"""
adapters.py — DataSource adapter classes for every collection source.

Each adapter wraps one ``fetch_*()`` function, normalises the parameter dict
from the portal or assembler, computes coverage, and returns a
:class:`~sparc.data.collect.sources.FetchResult`.

Side-effect on import: all standard adapters are registered in
:data:`~sparc.data.collect.sources.SOURCE_REGISTRY`.

No heavy imports at module level — each adapter imports its service module
lazily inside ``fetch()``.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from ._temporal import TemporalWindow
from .sources import FetchResult, SOURCE_REGISTRY


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _coverage(gdf: object, col: str) -> float:
    """Fraction of rows with a non-null value for *col*."""
    if not hasattr(gdf, "columns") or col not in gdf.columns:  # type: ignore[union-attr]
        return 0.0
    series = gdf[col]  # type: ignore[index]
    return float(series.notna().sum()) / max(len(gdf), 1)  # type: ignore[arg-type]


def _parse_window(params: dict) -> TemporalWindow:
    """Build or extract a :class:`TemporalWindow` from fetch params.

    Accepts either a pre-built ``TemporalWindow`` under the ``"window"`` key,
    or ISO-format ``"date_start"`` / ``"date_end"`` strings (legacy portal format).
    """
    from datetime import date as _date

    window = params.get("window")
    if isinstance(window, TemporalWindow):
        return window

    d_start = params.get("date_start", "2022-06-01")
    d_end = params.get("date_end", "2022-08-31")
    if isinstance(d_start, str):
        d_start = _date.fromisoformat(d_start)
    if isinstance(d_end, str):
        d_end = _date.fromisoformat(d_end)
    return TemporalWindow.from_range(d_start, d_end)


# ---------------------------------------------------------------------------
# ERA5 adapter
# ---------------------------------------------------------------------------

class ERA5Adapter:
    """Wraps :func:`sparc.data.collect.era5.fetch_era5`."""

    @property
    def group_name(self) -> str:
        return "era5"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        from .era5 import fetch_era5
        window = _parse_window(params)
        try:
            gdf = fetch_era5(fishnet, bbox, window)
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        col = "era5_t2m"
        return FetchResult(
            gdf=gdf,
            columns={col: "Open-Meteo ERA5"},
            coverage={col: _coverage(gdf, col)},
        )


# ---------------------------------------------------------------------------
# CAPA adapter
# ---------------------------------------------------------------------------

class CAPAAdapter:
    """Wraps :func:`sparc.data.collect.capa.fetch_capa`.

    CAPA is the label source; ``fetch()`` stores the discovered anchor dates
    in ``FetchResult.metadata["anchor_dates"]`` so the session can propagate
    them to subsequent adapters.
    """

    @property
    def group_name(self) -> str:
        return "capa"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        from .capa import fetch_capa
        window = _parse_window(params)
        osf_node_id   = params.get("capa_osf_node")
        osf_folder    = params.get("capa_osf_folder")
        try:
            gdf, anchor_dates = fetch_capa(fishnet, bbox, window, osf_node_id=osf_node_id, osf_folder_hint=osf_folder)
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        cols = {
            "aat_morning":  "NOAA/NIHHIS Heat Watch (OSF)",
            "aat_midday":   "NOAA/NIHHIS Heat Watch (OSF)",
            "aat_night":    "NOAA/NIHHIS Heat Watch (OSF)",
            "diurnal_aat":  "Derived: CAPA midday − night",
        }
        present = {k: v for k, v in cols.items() if k in gdf.columns}  # type: ignore[operator]
        return FetchResult(
            gdf=gdf,
            columns=present,
            coverage={col: _coverage(gdf, col) for col in present},
            metadata={"anchor_dates": anchor_dates},
        )


# ---------------------------------------------------------------------------
# Landsat adapter
# ---------------------------------------------------------------------------

_DEFAULT_INDICES = [
    "lst", "ndvi", "ndbi", "mndwi", "albedo",
    "ndwi", "savi", "evi", "ndbai", "nbr", "ndmi", "ui", "bsi",
]


class LandsatAdapter:
    """Wraps :func:`sparc.data.collect.landsat.fetch_landsat`."""

    @property
    def group_name(self) -> str:
        return "landsat"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        from .landsat import fetch_landsat
        window = _parse_window(params)
        enabled = params.get("enabled_indices") or _DEFAULT_INDICES
        try:
            gdf = fetch_landsat(
                fishnet, bbox, window,
                cloud_cover_max=float(params.get("cloud_cover_max", 20)),
                temporal_mode=params.get("temporal_mode", "composite"),
                enabled_indices=enabled,
            )
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        cols = {idx: "USGS STAC Landsat C2" for idx in enabled if idx in gdf.columns}  # type: ignore[operator]
        return FetchResult(
            gdf=gdf,
            columns=cols,
            coverage={col: _coverage(gdf, col) for col in cols},
        )


# ---------------------------------------------------------------------------
# NLCD adapter
# ---------------------------------------------------------------------------

class NLCDAdapter:
    """Wraps :func:`sparc.data.collect.nlcd.fetch_nlcd`."""

    @property
    def group_name(self) -> str:
        return "nlcd"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        from .nlcd import fetch_nlcd
        try:
            gdf = fetch_nlcd(fishnet, bbox)
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        cols = {
            "pct_impervious": "MRLC WCS NLCD 2021",
            "pct_canopy":     "MRLC WCS NLCD 2021",
            "land_cover":     "MRLC WCS NLCD 2021",
        }
        present = {k: v for k, v in cols.items() if k in gdf.columns}  # type: ignore[operator]
        return FetchResult(
            gdf=gdf,
            columns=present,
            coverage={col: _coverage(gdf, col) for col in present},
        )


# ---------------------------------------------------------------------------
# Buildings adapter
# ---------------------------------------------------------------------------

class BuildingsAdapter:
    """Wraps :func:`sparc.data.collect.buildings.fetch_buildings`."""

    _TIER_NAMES = {
        1: "LiDAR (local)",
        2: "Microsoft MLBuildings",
        3: "OSHB (Planetary Computer)",
        4: "Constant default",
    }

    @property
    def group_name(self) -> str:
        return "buildings"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        from .buildings import fetch_buildings
        lidar = Path(params["lidar_path"]) if params.get("lidar_path") else None
        dsm   = Path(params["dsm_path"])   if params.get("dsm_path")   else None
        try:
            gdf, tier = fetch_buildings(
                fishnet, bbox,
                lidar_path=lidar,
                dsm_path=dsm,
                default_height_m=float(params.get("default_height_m", 5.0)),
            )
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        src = self._TIER_NAMES.get(tier, "Unknown")
        cols = {
            "bldg_height_mean": src,
            "bldg_coverage":    src,
            "svf":              src,
        }
        present = {k: v for k, v in cols.items() if k in gdf.columns}  # type: ignore[operator]
        return FetchResult(
            gdf=gdf,
            columns=present,
            coverage={col: _coverage(gdf, col) for col in present},
            metadata={"tier": tier},
        )


# ---------------------------------------------------------------------------
# Equity adapter
# ---------------------------------------------------------------------------

class EquityAdapter:
    """Wraps :func:`sparc.data.collect.equity.fetch_equity`."""

    @property
    def group_name(self) -> str:
        return "equity"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        from .equity import fetch_equity
        try:
            gdf, _ = fetch_equity(fishnet, bbox)
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        cols = {
            "holc_grade":      "Mapping Inequality HOLC",
            "cdc_svi":         "CDC SVI 2022",
            "ejscreen_score":  "EPA EJScreen 2023",
        }
        present = {k: v for k, v in cols.items() if k in gdf.columns}  # type: ignore[operator]
        return FetchResult(
            gdf=gdf,
            columns=present,
            coverage={col: _coverage(gdf, col) for col in present},
        )


# ---------------------------------------------------------------------------
# Sentinel-2 adapter
# ---------------------------------------------------------------------------

class Sentinel2Adapter:
    """Wraps :func:`sparc.data.collect.sentinel2.fetch_sentinel2`."""

    @property
    def group_name(self) -> str:
        return "sentinel2"

    def fetch(self, fishnet: object, bbox: tuple, params: dict) -> FetchResult:
        window = _parse_window(params)
        try:
            from .sentinel2 import fetch_sentinel2
            gdf = fetch_sentinel2(
                fishnet, bbox,
                window.date_start, window.date_end,
                cloud_cover_max=float(params.get("cloud_cover_max", 20)),
                temporal_mode=params.get("temporal_mode", "composite"),
            )
        except ImportError:
            # Graceful degradation — sentinel2 optional dependency not installed
            import copy
            gdf = copy.copy(fishnet)
            for col in ("ndvi_s2", "ndbi_s2", "mndwi_s2"):
                gdf[col] = float("nan")  # type: ignore[index]
        except Exception as exc:
            return FetchResult(gdf=fishnet, error=str(exc))
        cols = {
            "ndvi_s2":  "Sentinel-2 L2A (Planetary Computer)",
            "ndbi_s2":  "Sentinel-2 L2A (Planetary Computer)",
            "mndwi_s2": "Sentinel-2 L2A (Planetary Computer)",
        }
        present = {k: v for k, v in cols.items() if k in gdf.columns}  # type: ignore[operator]
        return FetchResult(
            gdf=gdf,
            columns=present,
            coverage={col: _coverage(gdf, col) for col in present},
        )


# ---------------------------------------------------------------------------
# Registration — importing this module populates SOURCE_REGISTRY
# ---------------------------------------------------------------------------

def _register_all() -> None:
    SOURCE_REGISTRY.register(ERA5Adapter())
    SOURCE_REGISTRY.register(CAPAAdapter())
    SOURCE_REGISTRY.register(LandsatAdapter())
    SOURCE_REGISTRY.register(NLCDAdapter())
    SOURCE_REGISTRY.register(BuildingsAdapter())
    SOURCE_REGISTRY.register(EquityAdapter())
    SOURCE_REGISTRY.register(Sentinel2Adapter())


_register_all()
