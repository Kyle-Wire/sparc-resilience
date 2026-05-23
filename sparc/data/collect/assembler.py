"""
assembler.py — Compose all fetched layers onto the 30m fishnet and write GeoParquet.

Orchestrates the full data collection pipeline:
  1. Build the 30m fishnet clipped to the study boundary
  2. CAPA — fetch ground-truth air temperature; discover anchor dates
  3. Landsat — search for scenes within ±tolerance of each CAPA anchor date
  4. NLCD — land cover, impervious surface, canopy
  5. ERA5 — background air temperature
  6. Compute AAT_residual = CAPA midday − ERA5 background
  7. Buildings + SVF
  8. Equity layers
  9. Write output GeoParquet named by temporal mode
  10. Write data_manifest.json sidecar
  11. Update project.yml data.file_path and data.target_column in-place

CAPA runs first because its measurement dates are the prediction targets.
All covariates (Landsat spectral state, ERA5 background) are aligned to
those dates via a TemporalWindow, not the other way around.

The assembler is the only module that imports the other collect sub-modules.
It accepts a CollectConfig dataclass to keep the caller interface stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from .boundary import BoundaryResult
from .manifest import VariableManifest, VariableStatus

TemporalMode = Literal["single", "composite", "panel"]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class CollectConfig:
    """All parameters for one data collection run.

    Attributes
    ----------
    boundary : BoundaryResult
        Resolved study boundary.
    date_start, date_end : date
        Temporal window for Landsat / CAPA / ERA5 fetches.
    cloud_cover_max : float
        Maximum Landsat cloud cover percentage (0–100).
    temporal_mode : TemporalMode
        "single", "composite", or "panel".
    enabled_indices : list[str]
        Spectral indices to compute.  Defaults to all 13.
    output_dir : Path
        Where to write GeoParquet and manifest JSON.
    project_yml : Path | None
        If provided, auto-update data.file_path and data.target_column.
    lidar_path : Path | None
        Optional LiDAR building height raster.
    dsm_path : Path | None
        Optional DSM for true SVF computation.
    default_building_height_m : float
        Constant fallback height when all building sources fail.
    """

    boundary: BoundaryResult
    date_start: date
    date_end: date
    cloud_cover_max: float = 20.0
    temporal_mode: TemporalMode = "composite"
    enabled_indices: list[str] = field(default_factory=lambda: [
        "lst", "ndvi", "ndbi", "mndwi", "albedo",
        "ndwi", "savi", "evi", "ndbai", "nbr", "ndmi", "ui", "bsi",
    ])
    output_dir: Path = Path(".")
    project_yml: Optional[Path] = None
    lidar_path: Optional[Path] = None
    dsm_path: Optional[Path] = None
    default_building_height_m: float = 5.0


@dataclass
class AssemblerResult:
    """Output of a completed assembler run."""
    geoparquet_path: Path
    manifest_path: Path
    stations_path: Optional[Path]
    n_cells: int
    n_scenes: int
    scene_dates: list[date]
    manifest: VariableManifest
    building_tier: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(config: CollectConfig) -> AssemblerResult:
    """Execute the full data collection pipeline.

    Parameters
    ----------
    config : CollectConfig

    Returns
    -------
    AssemblerResult

    Raises
    ------
    RuntimeError
        If any required variable has zero coverage after all fetches.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required: pip install geopandas") from exc

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = VariableManifest.for_uhi()
    manifest.boundary_description = config.boundary.source
    manifest.temporal_description = (
        f"{config.temporal_mode} | {config.date_start} → {config.date_end} | "
        f"cloud ≤ {config.cloud_cover_max}%"
    )

    # --- Step 1: Build 30m fishnet ---
    fishnet = _build_fishnet(config.boundary)

    # --- Step 2: CAPA — fetch label data first, discover anchor dates ---
    # CAPA ground-truth air temperature measurements define the prediction
    # targets.  Their acquisition dates become the canonical temporal anchors
    # for Landsat and ERA5 alignment.
    for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
        manifest.fetching(col)
    from .capa import fetch_capa
    from ._temporal import TemporalWindow
    try:
        fishnet, capa_dates = fetch_capa(
            fishnet, config.boundary.bbox, config.date_start, config.date_end
        )
        for col, src in [
            ("aat_morning",  "NOAA CAPA / Open-Meteo"),
            ("aat_midday",   "NOAA CAPA / Open-Meteo"),
            ("aat_night",    "NOAA CAPA / Open-Meteo"),
            ("diurnal_aat",  "Derived: CAPA midday − night"),
        ]:
            manifest.update(col, coverage_pct=_coverage(fishnet, col), source_name=src)
    except Exception as exc:
        for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
            manifest.error(col, str(exc))
        capa_dates = []

    # Build the temporal search contract from CAPA anchor dates.
    # Landsat will search within ±tolerance_days of each anchor.
    # If CAPA failed (empty list), Landsat falls back to the full date range.
    window = TemporalWindow.from_capa_dates(
        capa_dates,
        date_start=config.date_start,
        date_end=config.date_end,
    )

    # --- Step 3: Landsat (spectral indices + LST) — aligned to CAPA anchors ---
    manifest.fetching("lst")
    from .landsat import fetch_landsat
    try:
        fishnet = fetch_landsat(
            fishnet,
            config.boundary.bbox,
            window,
            cloud_cover_max=config.cloud_cover_max,
            temporal_mode=config.temporal_mode,
            enabled_indices=config.enabled_indices,
        )
        for idx in config.enabled_indices:
            if idx in fishnet.columns:
                cov = _coverage(fishnet, idx)
                manifest.update(idx, coverage_pct=cov, source_name="USGS STAC Landsat C2")
    except Exception as exc:
        for idx in config.enabled_indices:
            manifest.error(idx, str(exc))

    # --- Step 4: NLCD ---
    for col in ("pct_impervious", "pct_canopy", "land_cover"):
        manifest.fetching(col)
    from .nlcd import fetch_nlcd
    try:
        fishnet = fetch_nlcd(fishnet, config.boundary.bbox)
        for col, src in [
            ("pct_impervious", "MRLC WCS NLCD 2021"),
            ("pct_canopy",     "MRLC WCS NLCD 2021"),
            ("land_cover",     "MRLC WCS NLCD 2021"),
        ]:
            manifest.update(col, coverage_pct=_coverage(fishnet, col), source_name=src)
    except Exception as exc:
        for col in ("pct_impervious", "pct_canopy", "land_cover"):
            manifest.error(col, str(exc))

    # --- Step 5: ERA5 ---
    manifest.fetching("era5_t2m")
    from .era5 import fetch_era5
    try:
        fishnet = fetch_era5(fishnet, config.boundary.bbox, config.date_start, config.date_end)
        manifest.update("era5_t2m", coverage_pct=_coverage(fishnet, "era5_t2m"),
                        source_name="Open-Meteo ERA5")
    except Exception as exc:
        manifest.error("era5_t2m", str(exc))

    # --- Step 6: AAT residual ---
    manifest.fetching("aat_residual")
    if "aat_midday" in fishnet.columns and "era5_t2m" in fishnet.columns:
        fishnet["aat_residual"] = fishnet["aat_midday"] - fishnet["era5_t2m"]
        manifest.update("aat_residual", coverage_pct=_coverage(fishnet, "aat_residual"),
                        source_name="Derived: CAPA midday − ERA5 background")
    else:
        manifest.error("aat_residual", "aat_midday or era5_t2m unavailable")

    # --- Step 7: Buildings + SVF ---
    for col in ("bldg_height_mean", "bldg_coverage", "svf"):
        manifest.fetching(col)
    from .buildings import fetch_buildings
    try:
        fishnet, bldg_tier = fetch_buildings(
            fishnet,
            config.boundary.bbox,
            lidar_path=config.lidar_path,
            dsm_path=config.dsm_path,
            default_height_m=config.default_building_height_m,
        )
        tier_names = {1: "LiDAR (local)", 2: "Microsoft MLBuildings",
                      3: "OSHB (Planetary Computer)", 4: "Constant default"}
        bldg_src = tier_names.get(bldg_tier, "Unknown")
        for col in ("bldg_height_mean", "bldg_coverage", "svf"):
            manifest.update(col, coverage_pct=_coverage(fishnet, col),
                            source_name=bldg_src, tier=bldg_tier)
    except Exception as exc:
        for col in ("bldg_height_mean", "bldg_coverage", "svf"):
            manifest.error(col, str(exc))
        bldg_tier = -1

    # --- Step 8: Equity ---
    for col in ("holc_grade", "cdc_svi", "ejscreen_score"):
        manifest.fetching(col)
    from .equity import fetch_equity
    try:
        fishnet, holc_overlay = fetch_equity(fishnet, config.boundary.bbox)
        holc_cov = _coverage(fishnet, "holc_grade")
        if holc_cov == 0.0:
            manifest.skip("holc_grade", "No HOLC coverage for this study area")
        else:
            manifest.update("holc_grade", coverage_pct=holc_cov,
                            source_name="Mapping Inequality HOLC API")
        for col, src in [
            ("cdc_svi",        "CDC SVI 2022"),
            ("ejscreen_score", "EPA EJScreen 2023"),
        ]:
            cov = _coverage(fishnet, col)
            if cov == 0.0:
                manifest.skip(col, f"No {src} coverage for this area")
            else:
                manifest.update(col, coverage_pct=cov, source_name=src)
    except Exception as exc:
        for col in ("holc_grade", "cdc_svi", "ejscreen_score"):
            manifest.error(col, str(exc))

    # --- Step 9: has_gap flag ---
    required_cols = [e.name for e in manifest.entries.values() if e.required]
    existing_cols = [c for c in required_cols if c in fishnet.columns]
    if existing_cols:
        fishnet["has_gap"] = fishnet[existing_cols].isnull().any(axis=1)
    else:
        fishnet["has_gap"] = False

    # --- Blocking check ---
    if not manifest.can_build:
        blocking = manifest.blocking_variables
        raise RuntimeError(
            f"Cannot build dataset — {len(blocking)} required variable(s) failed: "
            + ", ".join(blocking)
        )

    # --- Write outputs ---
    suffix = f"_{config.temporal_mode}"
    geoparquet_path = config.output_dir / f"dataset{suffix}.parquet"
    fishnet.to_parquet(str(geoparquet_path), index=False)

    manifest_path = config.output_dir / "data_manifest.json"
    manifest.save(manifest_path)

    # Stations sidecar (placeholder — populated by capa if station obs available)
    stations_path: Optional[Path] = None
    stations_file = config.output_dir / "stations.parquet"
    if stations_file.exists():
        stations_path = stations_file

    # --- Update project.yml ---
    if config.project_yml and config.project_yml.exists():
        _update_project_yml(
            config.project_yml,
            data_file_path=str(geoparquet_path),
            target_column="aat_residual",
        )

    return AssemblerResult(
        geoparquet_path=geoparquet_path,
        manifest_path=manifest_path,
        stations_path=stations_path,
        n_cells=len(fishnet),
        n_scenes=len(capa_dates),
        scene_dates=capa_dates,
        manifest=manifest,
        building_tier=bldg_tier,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_fishnet(boundary: BoundaryResult) -> object:
    """Create a 30m fishnet clipped to the boundary polygon."""
    from sparc.data.processing import create_fishnet, clip_to_boundary
    import geopandas as gpd

    boundary_proj = boundary.gdf.to_crs("EPSG:3857")  # type: ignore[union-attr]
    bounds = boundary_proj.total_bounds  # type: ignore[union-attr]
    fishnet = create_fishnet(tuple(bounds), resolution=30.0, crs="EPSG:3857")
    fishnet = clip_to_boundary(fishnet, boundary_proj)
    fishnet["cell_id"] = range(len(fishnet))
    fishnet["cell_x"] = fishnet.geometry.centroid.x
    fishnet["cell_y"] = fishnet.geometry.centroid.y
    return fishnet


def _coverage(gdf: object, col: str) -> float:
    """Return fraction of non-null values in *col*."""
    if col not in gdf.columns:  # type: ignore[operator]
        return 0.0
    series = gdf[col]  # type: ignore[index]
    n = len(series)
    if n == 0:
        return 0.0
    return float(series.notna().sum()) / n


def _update_project_yml(yml_path: Path, data_file_path: str, target_column: str) -> None:
    """Update only data.file_path and data.target_column in project.yml."""
    text = yml_path.read_text(encoding="utf-8")

    text = re.sub(
        r"(file_path\s*:\s*).*",
        lambda m: m.group(1) + f'"{data_file_path}"',
        text,
    )
    text = re.sub(
        r"(target_column\s*:\s*).*",
        lambda m: m.group(1) + f'"{target_column}"',
        text,
    )
    yml_path.write_text(text, encoding="utf-8")
