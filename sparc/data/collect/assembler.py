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
from typing import Callable, Literal, Optional

from .boundary import BoundaryResult
from .manifest import VariableManifest, VariableStatus
from sparc.data.spatial_grid import SpatialGrid
from .capa import download_capa, assign_capa_to_grid, fetch_capa
from ._temporal import TemporalWindow
from .landsat import download_landsat, assign_landsat_to_grid, fetch_landsat
from .nlcd import download_nlcd, assign_nlcd_to_grid, fetch_nlcd
from .era5 import download_era5, assign_era5_to_grid, fetch_era5
from .buildings import download_buildings, assign_buildings_to_grid, fetch_buildings
from .equity import download_equity, assign_equity_to_grid, fetch_equity

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
    capa_osf_node: Optional[str] = None   # OSF project node ID, e.g. "rk75w" for Brockton MA
    capa_osf_folder: Optional[str] = None  # Folder hint for multi-city nodes, e.g. "Boston" in tdsy7
    resolution_m: float = 30.0            # Analysis grid cell size in metres (user-defined)
    working_crs: str = "EPSG:3857"        # Projected CRS for spatial operations; defaults to Web Mercator


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

def run(config: CollectConfig, *, on_step: Optional[Callable[[str, bool], None]] = None) -> AssemblerResult:
    """Execute the full data collection pipeline.

    The pipeline is split into two explicit phases:

    **Phase 1 — Raw data collection (no fishnet required)**
        Downloads all source data for the bounding box.  CAPA traverse rasters
        are fetched to discover anchor dates; Landsat COG bands are downloaded
        and composited; NLCD tiles are fetched from MRLC WCS; ERA5 temperatures
        are fetched at 0.25° grid points; OSM building polygons are downloaded;
        HOLC and CDC SVI equity data are downloaded.

    **Phase 2 — Grid creation and assignment**
        A spatial fishnet is built at ``config.resolution_m`` (default 30 m).
        All Phase 1 data is then assigned to the grid via zonal statistics and
        spatial joins, producing the final per-cell dataset.

    Parameters
    ----------
    config : CollectConfig
    on_step : callable(group_name: str, success: bool) | None
        Optional progress callback.  Called after key pipeline milestones.
        Useful for streaming live progress to a desktop wizard.
        Step names: ``"capa_download"``, ``"raw_download"``, ``"grid_created"``,
        ``"landsat"``, ``"nlcd"``, ``"era5"``, ``"buildings"``, ``"equity"``.

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
        f"cloud ≤ {config.cloud_cover_max}% | grid {config.resolution_m:.0f} m"
    )

    bbox = config.boundary.bbox

    # =========================================================================
    # PHASE 1: Download raw data for the bounding box (no fishnet yet)
    # =========================================================================

    # --- 1a: CAPA — discover anchor dates (defines temporal alignment for all other sources) ---
    for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
        manifest.fetching(col)
    capa_raster_data = None
    capa_dates: list[date] = []
    try:
        capa_raster_data, capa_dates = download_capa(
            bbox, config.date_start, config.date_end,
            osf_node_id=config.capa_osf_node,
            osf_folder_hint=config.capa_osf_folder,
        )
    except Exception as exc:
        for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
            manifest.error(col, str(exc))

    if on_step is not None:
        on_step("capa", bool(capa_dates))

    # Landsat will search within ±tolerance_days of each CAPA anchor date.
    # If CAPA failed (empty list), Landsat falls back to the full date range.
    window = TemporalWindow.from_capa_dates(
        capa_dates,
        date_start=config.date_start,
        date_end=config.date_end,
    )

    # --- 1b: Landsat — download COG bands and compute composite arrays ---
    manifest.fetching("lst")
    landsat_composite = None
    landsat_scene_dates: list[date] = []
    try:
        landsat_composite, landsat_scene_dates = download_landsat(
            bbox, window,
            cloud_cover_max=config.cloud_cover_max,
            temporal_mode=config.temporal_mode,
            enabled_indices=config.enabled_indices,
        )
    except Exception as exc:
        for idx in config.enabled_indices:
            manifest.error(idx, str(exc))

    # --- 1c: NLCD — download WCS tiles ---
    for col in ("pct_impervious", "pct_canopy", "land_cover"):
        manifest.fetching(col)
    nlcd_tif_paths: dict = {}
    try:
        nlcd_tif_paths = download_nlcd(bbox)
    except Exception as exc:
        for col in ("pct_impervious", "pct_canopy", "land_cover"):
            manifest.error(col, str(exc))

    # --- 1d: ERA5 — fetch temperature at 0.25° grid points ---
    manifest.fetching("era5_t2m")
    era5_grid = None
    try:
        era5_grid = download_era5(bbox, window)
    except Exception as exc:
        manifest.error("era5_t2m", str(exc))

    # --- 1e: Buildings — download polygons/determine source ---
    for col in ("bldg_height_mean", "bldg_coverage", "svf"):
        manifest.fetching(col)
    buildings_raw: dict = {"type": "constant", "default_height_m": config.default_building_height_m}
    bldg_tier = 4
    try:
        buildings_raw, bldg_tier = download_buildings(
            bbox,
            lidar_path=config.lidar_path,
            dsm_path=config.dsm_path,
            default_height_m=config.default_building_height_m,
        )
    except Exception as exc:
        for col in ("bldg_height_mean", "bldg_coverage", "svf"):
            manifest.error(col, str(exc))

    # --- 1f: Equity — download HOLC + CDC SVI ---
    for col in ("holc_grade", "cdc_svi", "ejscreen_score"):
        manifest.fetching(col)
    equity_holc_gdf = None
    equity_cdc_svi_gdf = None
    try:
        equity_holc_gdf, equity_cdc_svi_gdf = download_equity(bbox)
    except Exception as exc:
        for col in ("holc_grade", "cdc_svi", "ejscreen_score"):
            manifest.error(col, str(exc))

    if on_step is not None:
        on_step("raw_download", True)

    # =========================================================================
    # PHASE 2: Create fishnet at user-specified resolution, then assign data
    # =========================================================================

    # --- 2a: Build fishnet at config.resolution_m in the project working CRS ---
    grid = SpatialGrid.from_boundary(
        config.boundary,
        resolution_m=config.resolution_m,
        crs=config.working_crs,
    )
    fishnet = grid.cells_3857.copy()
    fishnet["cell_id"] = range(len(fishnet))
    fishnet["cell_x"] = fishnet.geometry.centroid.x
    fishnet["cell_y"] = fishnet.geometry.centroid.y

    if on_step is not None:
        on_step("grid_created", True)

    # --- 2b: Assign CAPA to fishnet ---
    try:
        fishnet = assign_capa_to_grid(fishnet, capa_raster_data)
        for col, src in [
            ("aat_morning",  "NOAA/NIHHIS Heat Watch (OSF)"),
            ("aat_midday",   "NOAA/NIHHIS Heat Watch (OSF)"),
            ("aat_night",    "NOAA/NIHHIS Heat Watch (OSF)"),
            ("diurnal_aat",  "Derived: CAPA midday − night"),
        ]:
            if "error" not in manifest.entries.get(col, {}) if hasattr(manifest.entries.get(col, {}), "get") else True:
                manifest.update(col, coverage_pct=_coverage(fishnet, col), source_name=src)
    except Exception as exc:
        for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
            manifest.error(col, str(exc))

    # --- 2c: Assign Landsat to fishnet ---
    try:
        fishnet = assign_landsat_to_grid(
            fishnet, landsat_composite, landsat_scene_dates,
            enabled_indices=config.enabled_indices,
            temporal_mode=config.temporal_mode,
        )
        for idx in config.enabled_indices:
            if idx in fishnet.columns:
                cov = _coverage(fishnet, idx)
                manifest.update(idx, coverage_pct=cov, source_name="USGS STAC Landsat C2")
    except Exception as exc:
        for idx in config.enabled_indices:
            manifest.error(idx, str(exc))

    if on_step is not None:
        on_step("landsat", "lst" in fishnet.columns)

    # --- 2d: Assign NLCD to fishnet ---
    try:
        fishnet = assign_nlcd_to_grid(fishnet, nlcd_tif_paths)
        for col, src in [
            ("pct_impervious", "MRLC WCS NLCD 2021"),
            ("pct_canopy",     "MRLC WCS NLCD 2021"),
            ("land_cover",     "MRLC WCS NLCD 2021"),
        ]:
            manifest.update(col, coverage_pct=_coverage(fishnet, col), source_name=src)
    except Exception as exc:
        for col in ("pct_impervious", "pct_canopy", "land_cover"):
            manifest.error(col, str(exc))

    if on_step is not None:
        on_step("nlcd", "pct_impervious" in fishnet.columns)

    # --- 2e: Assign ERA5 to fishnet ---
    try:
        if era5_grid is not None:
            fishnet = assign_era5_to_grid(fishnet, *era5_grid)
        manifest.update("era5_t2m", coverage_pct=_coverage(fishnet, "era5_t2m"),
                        source_name="Open-Meteo ERA5")
    except Exception as exc:
        manifest.error("era5_t2m", str(exc))

    if on_step is not None:
        on_step("era5", "era5_t2m" in fishnet.columns)

    # --- 2f: Derived: AAT residual ---
    manifest.fetching("aat_residual")
    if "aat_midday" in fishnet.columns and "era5_t2m" in fishnet.columns:
        fishnet["aat_residual"] = fishnet["aat_midday"] - fishnet["era5_t2m"]
        manifest.update("aat_residual", coverage_pct=_coverage(fishnet, "aat_residual"),
                        source_name="Derived: CAPA midday − ERA5 background")
    else:
        manifest.error("aat_residual", "aat_midday or era5_t2m unavailable")

    # --- 2g: Assign buildings to fishnet ---
    try:
        fishnet, bldg_tier = assign_buildings_to_grid(
            fishnet, buildings_raw, bldg_tier,
            default_height_m=config.default_building_height_m,
        )
        tier_names = {1: "LiDAR (local)", 2: "OSM Overpass",
                      3: "OSHB (Planetary Computer)", 4: "Constant default"}
        bldg_src = tier_names.get(bldg_tier, "Unknown")
        for col in ("bldg_height_mean", "bldg_coverage", "svf"):
            manifest.update(col, coverage_pct=_coverage(fishnet, col),
                            source_name=bldg_src, tier=bldg_tier)
    except Exception as exc:
        for col in ("bldg_height_mean", "bldg_coverage", "svf"):
            manifest.error(col, str(exc))
        bldg_tier = -1

    if on_step is not None:
        on_step("buildings", bldg_tier > 0)

    # --- 2h: Assign equity to fishnet ---
    try:
        fishnet, holc_overlay = assign_equity_to_grid(fishnet, equity_holc_gdf, equity_cdc_svi_gdf)
        holc_cov = _coverage(fishnet, "holc_grade")
        if holc_cov == 0.0:
            manifest.skip("holc_grade", "No HOLC coverage for this study area")
        else:
            manifest.update("holc_grade", coverage_pct=holc_cov,
                            source_name="Mapping Inequality HOLC API")
        for col, src in [
            ("cdc_svi",        "CDC SVI 2022"),
            ("ejscreen_score", "EPA EJScreen 2023 (unavailable — NaN)"),
        ]:
            cov = _coverage(fishnet, col)
            if cov == 0.0:
                manifest.skip(col, f"No {src} coverage for this area")
            else:
                manifest.update(col, coverage_pct=cov, source_name=src)
    except Exception as exc:
        for col in ("holc_grade", "cdc_svi", "ejscreen_score"):
            manifest.error(col, str(exc))

    if on_step is not None:
        on_step("equity", "cdc_svi" in fishnet.columns)

    # =========================================================================
    # Phase 3: QA, export
    # =========================================================================

    # --- 3a: has_gap flag ---
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
