"""
recollect_era5.py — Re-fetch ERA5 boundary data for cities where the
campaign_date was wrong (OSF filename dates were upload dates, not
measurement dates).

Usage:
    python scripts/recollect_era5.py

Only updates the 12 ERA5 boundary columns and campaign_date in each
city's data.geoparquet.  All other columns are preserved.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ERA5_COLS = [
    "era5_morning_t2m", "era5_morning_windspeed", "era5_morning_winddir", "era5_morning_rh",
    "era5_midday_t2m",  "era5_midday_windspeed",  "era5_midday_winddir",  "era5_midday_rh",
    "era5_evening_t2m", "era5_evening_windspeed", "era5_evening_winddir", "era5_evening_rh",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "multicity_pilot.yml"

    with open(config_path) as f:
        pilot_cfg = yaml.safe_load(f)

    mc = pilot_cfg["multicity"]
    output_root = repo_root / mc["output_dir"]

    from sparc.data.collect.era5 import download_era5_boundary, assign_era5_boundary_to_grid

    for city_cfg in mc["pilot_cities"]:
        override_str = city_cfg.get("campaign_date_override")
        if not override_str:
            log.info("⏭  %s — no campaign_date_override, skipping", city_cfg["city_slug"])
            continue

        slug = city_cfg["city_slug"]
        bbox = tuple(city_cfg["bbox"])
        campaign_date = datetime.strptime(override_str, "%Y-%m-%d").date()

        parquet_path = output_root / slug / "data.geoparquet"
        if not parquet_path.exists():
            log.warning("⚠  %s — data.geoparquet not found, skipping", slug)
            continue

        log.info("━" * 60)
        log.info("  %s  →  campaign_date=%s", slug, campaign_date)

        log.info("  Fetching ERA5 boundary …")
        try:
            grid_lons, grid_lats, boundary_data = download_era5_boundary(bbox, campaign_date)
        except Exception as exc:
            log.error("  ERA5 download failed: %s", exc)
            continue

        log.info("  Loading parquet …")
        import geopandas as gpd
        gdf = gpd.read_parquet(parquet_path)

        log.info("  Assigning ERA5 to grid (%d rows) …", len(gdf))
        try:
            updated_gdf = assign_era5_boundary_to_grid(gdf, grid_lons, grid_lats, boundary_data)
        except Exception as exc:
            log.error("  ERA5 assign failed: %s", exc)
            continue

        # Drop old ERA5 columns and replace with updated ones
        drop_cols = [c for c in ERA5_COLS if c in gdf.columns]
        gdf = gdf.drop(columns=drop_cols)
        for col in ERA5_COLS:
            if col in updated_gdf.columns:
                gdf[col] = updated_gdf[col].values

        # Update campaign_date column
        if "campaign_date" in gdf.columns:
            gdf["campaign_date"] = str(campaign_date)

        log.info("  Saving updated parquet …")
        gdf.to_parquet(parquet_path, index=False)
        log.info("  ✓  Saved  era5_morning_t2m mean=%.2f°C", gdf["era5_morning_t2m"].mean())

    log.info("━" * 60)
    log.info("Done.")


if __name__ == "__main__":
    main()
