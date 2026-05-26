"""
test_collect_download.py — Visual smoke test for data collection layers.

Runs NLCD, Landsat (STAC query + NDVI composite), buildings, and equity
fetchers against a small Providence, RI study area and prints a summary
table the user can inspect to confirm real data was retrieved.

Run with:
    python scripts/test_collect_download.py
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from pathlib import Path

# ── Study area ─────────────────────────────────────────────────────────────
# Small Providence, RI bounding box in WGS-84
BBOX = (-71.46, 41.80, -71.38, 41.86)   # (minx, miny, maxx, maxy)
MINX, MINY, MAXX, MAXY = BBOX

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "collect_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def summarise(gdf, label: str, cols: list[str]) -> None:
    """Print per-column descriptive stats for collected columns."""
    import pandas as pd
    print(f"\n{label}  ({len(gdf)} cells, CRS={gdf.crs})")
    rows = []
    for col in cols:
        if col not in gdf.columns:
            rows.append({"column": col, "non-null": "—", "min": "—", "mean": "—", "max": "—"})
            continue
        s = gdf[col].dropna()
        rows.append({
            "column": col,
            "non-null": len(s),
            "min": f"{s.min():.3f}" if len(s) else "—",
            "mean": f"{s.mean():.3f}" if len(s) else "—",
            "max": f"{s.max():.3f}" if len(s) else "—",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


# ── Build a tiny fishnet ────────────────────────────────────────────────────
sep("STEP 0 — Build fishnet (30 m grid, ~50 cells)")
import geopandas as gpd
import numpy as np
from shapely.geometry import box as shapely_box
from pyproj import Transformer

t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
t_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

xmin_m, ymin_m = t_fwd.transform(MINX, MINY)
xmax_m, ymax_m = t_fwd.transform(MAXX, MAXY)

CELL_SIZE = 300.0  # 300 m cells → manageable grid size for quick smoke test
xs = np.arange(xmin_m, xmax_m, CELL_SIZE)
ys = np.arange(ymin_m, ymax_m, CELL_SIZE)
cells = [shapely_box(x, y, x + CELL_SIZE, y + CELL_SIZE)
         for y in ys for x in xs]
fishnet = gpd.GeoDataFrame(
    {"cell_id": range(len(cells)), "geometry": cells},
    crs="EPSG:3857",
)
print(f"  Created {len(fishnet)} fishnet cells over Providence, RI")

# ── NLCD ────────────────────────────────────────────────────────────────────
sep("STEP 1 — NLCD (impervious, canopy, land_cover)")
try:
    from sparc.data.collect.nlcd import fetch_nlcd
    nlcd_gdf = fetch_nlcd(fishnet, BBOX, cache_dir=OUTPUT_DIR / "nlcd_cache")
    summarise(nlcd_gdf, "NLCD", ["pct_impervious", "pct_canopy", "land_cover"])
    nlcd_out = OUTPUT_DIR / "nlcd_result.gpkg"
    nlcd_gdf.to_file(nlcd_out, driver="GPKG")
    print(f"\n  Saved: {nlcd_out}")
except Exception as exc:
    print(f"  ERROR: {exc}")

# ── Landsat ─────────────────────────────────────────────────────────────────
sep("STEP 2 — Landsat STAC query + NDVI composite")
try:
    from sparc.data.collect.landsat import fetch_landsat
    from sparc.data.collect._temporal import TemporalWindow
    window = TemporalWindow(
        date_start=date(2023, 6, 1),
        date_end=date(2023, 8, 31),
        anchor_dates=[],
        tolerance_days=30,
    )
    ls_gdf = fetch_landsat(
        fishnet, BBOX, window,
        cloud_cover_max=30.0,
        temporal_mode="composite",
        enabled_indices=["ndvi", "ndbi", "lst"],
        cache_dir=OUTPUT_DIR / "landsat_cache",
    )
    summarise(ls_gdf, "Landsat", ["ndvi", "ndbi", "lst"])
    ls_out = OUTPUT_DIR / "landsat_result.gpkg"
    ls_gdf.to_file(ls_out, driver="GPKG")
    print(f"\n  Saved: {ls_out}")
except Exception as exc:
    print(f"  ERROR: {exc}")

# ── Buildings ────────────────────────────────────────────────────────────────
sep("STEP 3 — Buildings (OSM Overpass → tier fallback)")
try:
    from sparc.data.collect.buildings import fetch_buildings
    bldg_gdf, tier = fetch_buildings(
        fishnet, BBOX,
        cache_dir=OUTPUT_DIR / "buildings_cache",
    )
    tier_labels = {1: "LiDAR", 2: "OSM Overpass", 3: "OSHB", 4: "constant-fallback"}
    print(f"  Data source tier used: {tier} ({tier_labels.get(tier, '?')})")
    summarise(bldg_gdf, "Buildings", ["bldg_height_mean", "bldg_coverage", "svf"])
    bldg_out = OUTPUT_DIR / "buildings_result.gpkg"
    bldg_gdf.to_file(bldg_out, driver="GPKG")
    print(f"\n  Saved: {bldg_out}")
except Exception as exc:
    print(f"  ERROR: {exc}")

# ── Equity ───────────────────────────────────────────────────────────────────
sep("STEP 4 — Equity (HOLC, CDC SVI, EJScreen)")
try:
    from sparc.data.collect.equity import fetch_equity
    eq_gdf, holc_overlay = fetch_equity(fishnet, BBOX, cache_dir=OUTPUT_DIR / "equity_cache")
    summarise(eq_gdf, "Equity", ["holc_grade", "cdc_svi", "ejscreen_score"])
    eq_out = OUTPUT_DIR / "equity_result.gpkg"
    eq_gdf.to_file(eq_out, driver="GPKG")
    print(f"\n  Saved: {eq_out}")
    if holc_overlay is not None and not holc_overlay.empty:
        holc_out = OUTPUT_DIR / "holc_polygons.gpkg"
        holc_overlay.to_file(holc_out, driver="GPKG")
        print(f"  HOLC polygons saved: {holc_out}")
    else:
        print("  HOLC: no coverage for this bounding box")
except Exception as exc:
    print(f"  ERROR: {exc}")

# ── Sentinel-2 ───────────────────────────────────────────────────────────────
sep("STEP 5 — Sentinel-2 (NDVI, NDBI, MNDWI)")
try:
    from sparc.data.collect.sentinel2 import fetch_sentinel2
    s2_gdf = fetch_sentinel2(
        fishnet, BBOX,
        date_start=date(2023, 6, 1),
        date_end=date(2023, 8, 31),
        cloud_cover_max=30.0,
        temporal_mode="composite",
    )
    summarise(s2_gdf, "Sentinel-2", ["ndvi_s2", "ndbi_s2", "mndwi_s2"])
    s2_out = OUTPUT_DIR / "sentinel2_result.gpkg"
    s2_gdf.to_file(s2_out, driver="GPKG")
    print(f"\n  Saved: {s2_out}")
except Exception as exc:
    print(f"  ERROR: {exc}")

# ── CAPA Heat Watch ───────────────────────────────────────────────────────────
sep("STEP 6 — CAPA Heat Watch traverse data (OSF)")
# Philadelphia 2022 campaign: https://osf.io/3xts6/files/
# Note: Philadelphia's bbox is far from Providence; the traverses cover Philadelphia.
# We pass the fishnet but the spatial interpolation will extrapolate via nearest-
# neighbour for cells outside the traverse extent — useful as a function test.
CAPA_OSF_NODE = "3xts6"  # Philadelphia 2022
try:
    from sparc.data.collect.capa import fetch_capa
    from sparc.data.collect._temporal import TemporalWindow
    window = TemporalWindow(
        date_start=date(2022, 9, 21),
        date_end=date(2022, 9, 21),
        anchor_dates=[],
        tolerance_days=3,
    )
    capa_gdf, capa_dates = fetch_capa(
        fishnet, BBOX, window, osf_node_id=CAPA_OSF_NODE
    )
    summarise(capa_gdf, "CAPA", ["aat_morning", "aat_midday", "aat_night", "diurnal_aat"])
    print(f"  Anchor dates: {capa_dates}")
    capa_out = OUTPUT_DIR / "capa_result.gpkg"
    capa_gdf.to_file(capa_out, driver="GPKG")
    print(f"\n  Saved: {capa_out}")
except Exception as exc:
    print(f"  ERROR: {exc}")

# ── ERA5 background temperature ───────────────────────────────────────────────
sep("STEP 7 — ERA5 background temperature (Open-Meteo archive)")
try:
    from sparc.data.collect.era5 import fetch_era5
    from sparc.data.collect._temporal import TemporalWindow
    window = TemporalWindow(
        date_start=date(2023, 6, 1),
        date_end=date(2023, 8, 31),
        anchor_dates=[],
        tolerance_days=30,
    )
    era5_gdf = fetch_era5(fishnet, BBOX, window)
    summarise(era5_gdf, "ERA5", ["era5_t2m"])
    era5_out = OUTPUT_DIR / "era5_result.gpkg"
    era5_gdf.to_file(era5_out, driver="GPKG")
    print(f"\n  Saved: {era5_out}")
except Exception as exc:
    print(f"  ERROR: {exc}")

sep("DONE — check output/collect_test/ for GeoPackage files")
print(f"  Output directory: {OUTPUT_DIR}")