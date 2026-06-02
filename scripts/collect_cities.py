"""
collect_cities.py -- Resumable multi-city data collection for SPARC JEPA pipeline.

Reads configs/multicity_pilot.yml, collects all data layers for each pilot city,
assembles a canonical flat GeoParquet, and writes a per-city project.yml.

Usage
-----
    python scripts/collect_cities.py
    python scripts/collect_cities.py --config configs/multicity_pilot.yml --resume
    python scripts/collect_cities.py --cities chicago_il seattle_wa
    python scripts/collect_cities.py --resume --cities philadelphia_pa
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap -- allow running from repo root or scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_cities")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/multicity_pilot.yml",
                   help="Path to multicity pilot config (default: configs/multicity_pilot.yml)")
    p.add_argument("--resume", action="store_true",
                   help="Skip cities that already have data.geoparquet")
    p.add_argument("--cities", nargs="+", metavar="SLUG",
                   help="Only collect these city slugs (space-separated)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fishnet builder
# ---------------------------------------------------------------------------

def build_fishnet(
    bbox: tuple[float, float, float, float],
    utm_epsg: str,
    resolution_m: int = 30,
) -> object:
    """Build a 30m fishnet GeoDataFrame in *utm_epsg* CRS over *bbox* (WGS-84)."""
    import geopandas as gpd
    from shapely.geometry import box

    # Reproject bbox corners to UTM
    bbox_poly = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")
    bbox_utm = bbox_gdf.to_crs(utm_epsg)
    minx, miny, maxx, maxy = bbox_utm.total_bounds

    import numpy as np
    xs = np.arange(minx, maxx, resolution_m)
    ys = np.arange(miny, maxy, resolution_m)

    cells = []
    for x in xs:
        for y in ys:
            cells.append(box(x, y, x + resolution_m, y + resolution_m))

    fishnet = gpd.GeoDataFrame(geometry=cells, crs=utm_epsg)
    fishnet = fishnet.reset_index(drop=True)
    log.info("fishnet: %d cells at %dm in %s", len(fishnet), resolution_m, utm_epsg)
    return fishnet


# ---------------------------------------------------------------------------
# Layer collectors (each wrapped in try/except)
# ---------------------------------------------------------------------------

def _safe(fn, *args, default=None, label="layer", **kwargs):
    """Call *fn* with args; log warning and return *default* on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("%s failed: %s -- filling NaN", label, exc)
        return default


def collect_nlcd(fishnet_gdf, bbox):
    from sparc.data.collect.nlcd import fetch_nlcd
    return _safe(fetch_nlcd, fishnet_gdf, bbox, label="NLCD")


def collect_landsat(fishnet_gdf, bbox, campaign_date):
    from sparc.data.collect.landsat import fetch_landsat
    from sparc.data.collect._temporal import TemporalWindow
    d_start = campaign_date - timedelta(days=45)
    d_end   = campaign_date + timedelta(days=45)
    window  = TemporalWindow(d_start, d_end)
    return _safe(fetch_landsat, fishnet_gdf, bbox, window, temporal_mode="composite",
                 label="Landsat")


def collect_sentinel2(fishnet_gdf, bbox, campaign_date):
    from sparc.data.collect.sentinel2 import fetch_sentinel2
    d_start = campaign_date - timedelta(days=45)
    d_end   = campaign_date + timedelta(days=45)
    return _safe(fetch_sentinel2, fishnet_gdf, bbox, d_start, d_end,
                 temporal_mode="composite", label="Sentinel-2")


def collect_buildings(fishnet_gdf, bbox):
    from sparc.data.collect.buildings import fetch_buildings
    result = _safe(fetch_buildings, fishnet_gdf, bbox, label="Buildings")
    if isinstance(result, tuple):
        return result[0]   # fetch_buildings returns (gdf, tier) -- extract gdf
    return result


def collect_dem(fishnet_gdf, bbox):
    from sparc.data.collect.dem import download_dem, assign_dem_to_grid
    dem_data, _ = _safe(download_dem, bbox, default=(None, []), label="DEM download")
    if dem_data is None:
        return None
    return _safe(assign_dem_to_grid, fishnet_gdf, dem_data, label="DEM assign")


def collect_equity(fishnet_gdf, bbox):
    from sparc.data.collect.equity import download_equity, assign_equity_to_grid
    holc_gdf, cdc_svi_gdf = _safe(
        download_equity, bbox, default=(None, None), label="Equity download"
    )
    result = _safe(
        assign_equity_to_grid, fishnet_gdf, holc_gdf, cdc_svi_gdf,
        default=(None, None), label="Equity assign",
    )
    if result is None or (isinstance(result, tuple) and result[0] is None):
        return None
    if isinstance(result, tuple):
        return result[0]   # (gdf, holc_overlay) → return gdf
    return result


def collect_capa(fishnet_gdf, bbox, osf_node_id, osf_folder_hint=None):
    from sparc.data.collect.capa import download_capa, assign_capa_to_grid
    from sparc.data.collect._temporal import TemporalWindow
    window = TemporalWindow(date(2010, 1, 1), date.today())
    raster_data, capa_dates = _safe(
        download_capa, bbox, window,
        osf_node_id=osf_node_id, osf_folder_hint=osf_folder_hint,
        default=(None, []), label=f"CAPA download (node={osf_node_id})",
    )
    capa_gdf = _safe(assign_capa_to_grid, fishnet_gdf, raster_data, label="CAPA assign")
    campaign_date = capa_dates[0] if capa_dates else None
    return capa_gdf, campaign_date


def collect_era5_boundary(fishnet_gdf, bbox, campaign_date):
    from sparc.data.collect.era5 import download_era5_boundary, assign_era5_boundary_to_grid
    grid_lons, grid_lats, boundary_data = _safe(
        download_era5_boundary, bbox, campaign_date,
        default=([], [], {}), label="ERA5 boundary download",
    )
    return _safe(
        assign_era5_boundary_to_grid, fishnet_gdf, grid_lons, grid_lats, boundary_data,
        label="ERA5 boundary assign",
    )


# ---------------------------------------------------------------------------
# project.yml generation
# ---------------------------------------------------------------------------

def generate_city_project_yml(
    city_cfg: dict,
    output_dir: Path,
    geoparquet_path: Path,
    pilot_cfg: dict,
) -> Path:
    """Write a project.yml for one city that the SPARC pipeline can consume.

    The generated YAML is readable by ``sparc.config.config.load_config()``.
    """
    mc = pilot_cfg["multicity"]
    feature_cols = mc["feature_columns"]
    land_surface_cols = feature_cols.get("land_surface", [])
    era5_midday_cols  = feature_cols.get("era5_midday", [])
    predictors = land_surface_cols + era5_midday_cols

    physics_block = {
        "priors_file":  "physics/priors.yml",
        "caps_file":    "physics/caps.yml",
        "monotone_constraints": {
            "pct_canopy":     -1,
            "pct_impervious":  1,
            "ndvi":           -1,
            "albedo":         -1,
            "elevation_m":     0,
            "distance_water_m": 0,
        },
    }

    yml_content = {
        "project": {
            "name": f"SPARC Multi-City -- {city_cfg['city']}",
            "description": (
                f"Auto-generated multi-city JEPA project for {city_cfg['city']}. "
                "Do not edit -- regenerate via collect_cities.py."
            ),
            "domain": "uhi",
            "version": "2.1.0",
            "author":  "SPARC collect_cities (auto-generated)",
        },
        "collect": {
            "capa_osf_node": "",  # already collected; placeholder for pipeline compat
        },
        "data": {
            "file_path":        str(geoparquet_path.resolve()),
            "target_column":    "aat_midday",
            "identifier_column":"cell_id",
            "coord_columns":    ["centroid_x", "centroid_y"],
        },
        "crs": {
            "input":   city_cfg["utm_epsg"],
            "working": city_cfg["utm_epsg"],
        },
        "predictors": predictors,
        "physics": physics_block,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    yml_path = output_dir / "project.yml"
    header = (
        "# Auto-generated by scripts/collect_cities.py\n"
        f"# City: {city_cfg['city']}  slug: {city_cfg['city_slug']}\n"
        "# Edit configs/multicity_pilot.yml to change features, then re-run.\n\n"
    )
    with open(yml_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(yml_content, fh, default_flow_style=False, allow_unicode=True)
    log.info("project.yml → %s", yml_path)
    return yml_path


# ---------------------------------------------------------------------------
# Per-city collection
# ---------------------------------------------------------------------------

def collect_one_city(
    city_cfg: dict,
    pilot_cfg: dict,
    output_root: Path,
    resume: bool = False,
) -> dict:
    """Collect all layers for a single city.  Returns a result summary dict."""
    from sparc.data.collect.assembler_multicity import assemble_city_geoparquet
    from sparc.data.collect.capa_catalog import CAPA_CATALOG

    city_slug = city_cfg["city_slug"]
    city_name = city_cfg["city"]
    bbox      = tuple(city_cfg["bbox"])
    utm_epsg  = city_cfg["utm_epsg"]
    mc        = pilot_cfg["multicity"]
    res_m     = mc.get("fishnet_resolution_m", 30)

    city_out = output_root / city_slug
    city_out.mkdir(parents=True, exist_ok=True)
    parquet_path = city_out / "data.geoparquet"

    if resume and parquet_path.exists():
        log.info("⏭  [%s] resume: already collected -- skipping", city_slug)
        return {"city_slug": city_slug, "status": "skipped", "path": str(parquet_path)}

    log.info("")
    log.info("═" * 60)
    log.info("  Collecting: %s  (%s)", city_name, city_slug)
    log.info("  bbox=%s  utm=%s", bbox, utm_epsg)
    log.info("═" * 60)
    t0 = time.time()

    # Look up OSF node from catalog
    catalog_entry = CAPA_CATALOG.get(city_name, {})
    osf_node_id   = catalog_entry.get("osf_node")
    osf_folder    = catalog_entry.get("folder_hint")

    # Build fishnet
    fishnet = build_fishnet(bbox, utm_epsg, resolution_m=res_m)

    # ── Phase 1: CAPA -- needed first to get campaign_date ──────────────────
    log.info("[%s] Collecting CAPA ...", city_slug)
    capa_gdf, campaign_date = collect_capa(fishnet, bbox, osf_node_id, osf_folder)
    # Allow config-level override (e.g. when OSF filename dates are upload dates, not campaign dates)
    override_str = city_cfg.get("campaign_date_override")
    if override_str:
        from datetime import datetime as _dt
        campaign_date = _dt.strptime(override_str, "%Y-%m-%d").date()
        log.info("[%s] campaign_date_override applied: %s", city_slug, campaign_date)
    elif campaign_date is None:
        campaign_date = date(2019, 8, 1)
        log.warning("[%s] CAPA campaign date unknown -- defaulting to %s", city_slug, campaign_date)

    log.info("[%s] Campaign date: %s", city_slug, campaign_date)

    # ── Phase 2: Remaining layers ──────────────────────────────────────────
    log.info("[%s] Collecting NLCD ...", city_slug)
    nlcd_gdf = collect_nlcd(fishnet, bbox)

    log.info("[%s] Collecting Landsat ...", city_slug)
    landsat_gdf = collect_landsat(fishnet, bbox, campaign_date)

    log.info("[%s] Collecting Sentinel-2 ...", city_slug)
    s2_gdf = collect_sentinel2(fishnet, bbox, campaign_date)

    log.info("[%s] Collecting Buildings ...", city_slug)
    bldg_gdf = collect_buildings(fishnet, bbox)

    log.info("[%s] Collecting DEM ...", city_slug)
    dem_gdf = collect_dem(fishnet, bbox)

    log.info("[%s] Collecting Equity ...", city_slug)
    equity_gdf = collect_equity(fishnet, bbox)

    log.info("[%s] Collecting ERA5 boundary ...", city_slug)
    era5_boundary_gdf = collect_era5_boundary(fishnet, bbox, campaign_date)

    # ── Phase 3: Assemble ──────────────────────────────────────────────────
    log.info("[%s] Assembling GeoParquet ...", city_slug)
    feature_cols_flat = (
        mc["feature_columns"].get("land_surface", [])
        + mc["feature_columns"].get("era5_morning", [])
        + mc["feature_columns"].get("era5_midday", [])
        + mc["feature_columns"].get("era5_evening", [])
    )
    label_cols = mc["label_columns"]

    gdf = assemble_city_geoparquet(
        fishnet_gdf=fishnet,
        layers={
            "nlcd":          nlcd_gdf,
            "landsat":       landsat_gdf,
            "sentinel2":     s2_gdf,
            "buildings":     bldg_gdf,
            "dem":           dem_gdf,
            "equity":        equity_gdf,
            "capa":          capa_gdf,
            "era5_boundary": era5_boundary_gdf,
        },
        city_slug=city_slug,
        utm_epsg=utm_epsg,
        campaign_date=campaign_date,
        feature_columns=feature_cols_flat,
        label_columns=label_cols,
    )

    # Add cell_id column as integer index
    gdf["cell_id"] = range(len(gdf))

    # Save GeoParquet
    log.info("[%s] Saving → %s", city_slug, parquet_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf.to_parquet(str(parquet_path), index=False)

    # ── Phase 4: Generate project.yml ─────────────────────────────────────
    generate_city_project_yml(city_cfg, city_out, parquet_path, pilot_cfg)

    elapsed = time.time() - t0
    n_rows = len(gdf)
    nan_rate = float(gdf.drop(columns=["geometry"], errors="ignore").isna().mean().mean())

    log.info(
        "[%s] ✓ Done in %.1fs  |  %d cells  |  NaN rate %.1f%%",
        city_slug, elapsed, n_rows, nan_rate * 100,
    )
    return {
        "city_slug": city_slug,
        "status": "ok",
        "path": str(parquet_path),
        "n_rows": n_rows,
        "nan_rate": nan_rate,
        "campaign_date": str(campaign_date),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        pilot_cfg = yaml.safe_load(fh)

    mc = pilot_cfg["multicity"]
    output_root = Path(mc["output_dir"])
    cities = mc["pilot_cities"]

    # Filter to requested slugs if --cities given
    if args.cities:
        requested = set(args.cities)
        cities = [c for c in cities if c["city_slug"] in requested]
        if not cities:
            log.error("No matching cities found for: %s", args.cities)
            sys.exit(1)

    # Group cities by OSF node to avoid hammering shared nodes sequentially
    from sparc.data.collect.capa_catalog import CAPA_CATALOG
    node_groups: dict[str, list[dict]] = defaultdict(list)
    for city_cfg in cities:
        catalog_entry = CAPA_CATALOG.get(city_cfg["city"], {})
        osf_node = catalog_entry.get("osf_node", "unknown")
        node_groups[osf_node].append(city_cfg)

    log.info("Pilot config: %s", config_path)
    log.info("Output root:  %s", output_root)
    log.info("Cities to process: %d (in %d OSF-node groups)",
             len(cities), len(node_groups))

    results = []
    for osf_node, group in node_groups.items():
        log.info("── OSF node group '%s' (%d cities) ──", osf_node, len(group))
        for city_cfg in group:
            result = collect_one_city(city_cfg, pilot_cfg, output_root, resume=args.resume)
            results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print(f"  {'CITY SLUG':<22} {'STATUS':<10} {'ROWS':>8} {'NaN%':>7}  {'DATE':<12}")
    print("─" * 72)
    for r in results:
        slug   = r.get("city_slug", "?")
        status = r.get("status", "?")
        rows   = r.get("n_rows", "─")
        nan_pct = f"{r['nan_rate']*100:.1f}%" if "nan_rate" in r else "─"
        cdate  = r.get("campaign_date", "─")
        print(f"  {slug:<22} {status:<10} {str(rows):>8} {nan_pct:>7}  {cdate:<12}")
    print("═" * 72)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_sk = sum(1 for r in results if r.get("status") == "skipped")
    print(f"  {n_ok} collected, {n_sk} skipped  |  total {len(results)} cities")
    print()


if __name__ == "__main__":
    main()
