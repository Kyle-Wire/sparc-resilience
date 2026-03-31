#!/usr/bin/env python3
"""
preprocess_forcesmip.py — Convert ForceSMIP Tier 1 NetCDF data to SPARC CSV format.

Reads ForceSMIP Evaluation and Ensemble-Mean NetCDF files, computes per-grid-cell
OLS trends, merges forcing indices and spatial covariates, and outputs one CSV
per target variable × evaluation member ready for the SPARC pipeline.

Usage
-----
    python preprocess_forcesmip.py \
        --eval-dir  "path/to/Evaluation-Tier1" \
        --ensmean-dir "path/to/ensmeans-Tier1" \
        --output-dir "data" \
        --trend-start 1980 --trend-end 2022 \
        --variables tos tas pr psl

Outputs
-------
    data/forcesmip_{var}_{member}.csv          — SPARC input (one row per grid cell)
    data/forcesmip_ensmean_{var}_{member}.csv  — truth/forced response for validation
    data/global_2p5deg_grid.gpkg               — AOI grid polygons

Dependencies: xarray, numpy, pandas, geopandas, shapely, scipy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time helpers (handles both numpy datetime64 and cftime calendars)
# ---------------------------------------------------------------------------

def _time_to_fractional_years(times) -> np.ndarray:
    """Convert xarray time values to fractional year floats.
    
    Works with numpy datetime64 *and* cftime (noleap, 360_day, etc.).
    """
    try:
        _dt = pd.to_datetime(times)
        return np.array(_dt.year + (_dt.month - 0.5) / 12.0, dtype=float)
    except (TypeError, ValueError):
        # cftime objects — extract .year / .month manually
        return np.array([t.year + (t.month - 0.5) / 12.0 for t in times], dtype=float)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def build_grid_dataframe(lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """Create a dataframe with one row per grid cell, with spatial covariates."""
    lon2d, lat2d = np.meshgrid(lons, lats)
    lon_flat = lon2d.ravel()
    lat_flat = lat2d.ravel()

    # Grid cell IDs: row-major order
    n_cells = len(lon_flat)
    grid_cell_id = np.arange(n_cells)

    # Spatial covariates
    lat_abs = np.abs(lat_flat)

    # Simple land fraction proxy: NaN in tos marks ocean;
    # we'll fill this from actual data later.
    # Ocean basin encoding (rough):
    #   0 = land, 1 = Atlantic, 2 = Pacific, 3 = Indian, 4 = Southern, 5 = Arctic
    ocean_basin = _assign_ocean_basins(lat_flat, lon_flat)

    return pd.DataFrame({
        "grid_cell_id": grid_cell_id,
        "lon": lon_flat,
        "lat": lat_flat,
        "lat_abs": lat_abs,
        "land_fraction": np.nan,  # filled per-variable from data mask
        "ocean_basin": ocean_basin,
    })


def _assign_ocean_basins(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Rough ocean basin assignment from lat/lon. Returns integer codes."""
    # Normalize lon to 0-360
    lon360 = lon % 360
    basin = np.zeros(len(lat), dtype=int)  # default: land (0)

    # Southern Ocean: lat < -60
    basin[lat < -60] = 4
    # Arctic: lat > 66
    basin[lat > 66] = 5
    # Atlantic: roughly 80W-0E (280-360) and 0-30E (0-30), lat -60 to 66
    atlantic_mask = (
        ((lon360 >= 280) | (lon360 <= 30))
        & (lat >= -60) & (lat <= 66)
    )
    basin[atlantic_mask] = 1
    # Indian: roughly 30E-120E, lat -60 to 30
    indian_mask = (
        (lon360 >= 30) & (lon360 <= 120)
        & (lat >= -60) & (lat <= 30)
    )
    basin[indian_mask] = 3
    # Pacific: everything else that's ocean (we'll mask land later)
    pacific_mask = (basin == 0) & (lat >= -60) & (lat <= 66)
    # Fill remaining ocean cells as Pacific
    basin[pacific_mask] = 2

    return basin


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

def compute_trend_field(
    da,  # xarray.DataArray (time, lat, lon)
    start_year: int = 1980,
    end_year: int = 2022,
) -> np.ndarray:
    """
    Compute per-grid-cell OLS linear trend over the specified period.

    Returns a 1D array of trend slopes (units per year × n_years), i.e.
    the total change over the period, matching ForceSMIP convention.
    """
    from scipy.stats import linregress

    # Select time window
    da_sel = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    n_time = da_sel.sizes["time"]
    n_years = end_year - start_year

    # Flatten spatial dims
    data = da_sel.values.reshape(n_time, -1)  # (time, n_cells)
    n_cells = data.shape[1]

    # Time axis in fractional years
    times = da_sel["time"].values
    t_numeric = _time_to_fractional_years(times)
    t_centered = t_numeric - t_numeric.mean()

    trends = np.full(n_cells, np.nan)
    for j in range(n_cells):
        col = data[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < max(24, n_time // 3):
            # Skip cells with too much missing data
            continue
        result = linregress(t_centered[valid], col[valid])
        trends[j] = result.slope * n_years  # total trend over period

    return trends


def compute_gmst_trend(
    ds_tas,  # xarray.Dataset with 'tas'
    start_year: int = 1980,
    end_year: int = 2022,
) -> float:
    """Compute global-mean surface temperature trend as a scalar."""
    from scipy.stats import linregress

    da = ds_tas["tas"]
    da_sel = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Area-weighted global mean (cos-lat weighting)
    weights = np.cos(np.deg2rad(da_sel.lat))
    gmst = da_sel.weighted(weights).mean(dim=["lat", "lon"]).values

    times = da_sel["time"].values
    t_numeric = _time_to_fractional_years(times)
    n_years = end_year - start_year
    t_centered = t_numeric - t_numeric.mean()

    valid = ~np.isnan(gmst)
    result = linregress(t_centered[valid], gmst[valid])
    return result.slope * n_years


# ---------------------------------------------------------------------------
# Forcing index computation
# ---------------------------------------------------------------------------

def compute_forcing_indices(
    ds_eval,  # xarray.Dataset for tas (to derive GMST)
    lats: np.ndarray,
    lons: np.ndarray,
    start_year: int = 1980,
    end_year: int = 2022,
) -> dict:
    """
    Compute forcing and climate mode indices.

    For this fingerprinting approach, forcing indices are scalars replicated
    across all grid cells (the spatial pattern comes from the regression,
    not from spatially varying forcings).

    Returns dict of {column_name: scalar_or_array}.
    """
    n_cells = len(lats) * len(lons)

    # GMST trend
    gmst_trend = compute_gmst_trend(ds_eval, start_year, end_year)

    # For forcing indices, we use simple placeholders based on known
    # historical forcing trajectories.  Users should replace these with
    # actual CMIP6 forcing timeseries for production runs.
    # These are approximate 1980-2022 trends from IPCC AR6 Fig 2.10:
    ghg_forcing_index = 1.2      # ~1.2 W/m2 increase 1980-2022
    aerosol_forcing_index = -0.1 # net change small (peaked ~2005)
    volcanic_forcing_index = 0.0 # no major eruption trend
    solar_forcing_index = 0.0    # ~11yr cycle, no net trend

    # Climate mode indices: compute from the data
    da_tas = ds_eval["tas"] if "tas" in ds_eval else None

    enso = _compute_enso_index(ds_eval, start_year, end_year, lats, lons)
    amv = _compute_amv_index(ds_eval, start_year, end_year, lats, lons)
    ipo = _compute_ipo_index(ds_eval, start_year, end_year, lats, lons)

    return {
        "GMST_trend": np.full(n_cells, gmst_trend),
        "GHG_forcing_index": np.full(n_cells, ghg_forcing_index),
        "aerosol_forcing_index": np.full(n_cells, aerosol_forcing_index),
        "volcanic_forcing_index": np.full(n_cells, volcanic_forcing_index),
        "solar_forcing_index": np.full(n_cells, solar_forcing_index),
        "ENSO_detrended": np.full(n_cells, enso),
        "AMV_index": np.full(n_cells, amv),
        "IPO_index": np.full(n_cells, ipo),
    }


def _compute_enso_index(ds, start_year, end_year, lats, lons):
    """Compute detrended Nino3.4 trend (simplified)."""
    from scipy.stats import linregress

    var_name = _find_sst_var(ds)
    if var_name is None:
        return 0.0

    da = ds[var_name].sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
        lat=slice(-5, 5),
        lon=slice(190, 240),
    )
    if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
        # Try 0-360 vs -180-180
        da = ds[var_name].sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
            lat=slice(-5, 5),
        )
        lon_vals = da.lon.values % 360
        mask = (lon_vals >= 190) & (lon_vals <= 240)
        if mask.any():
            da = da.isel(lon=mask)
        else:
            return 0.0

    nino34 = da.mean(dim=["lat", "lon"]).values
    valid = ~np.isnan(nino34)
    if valid.sum() < 24:
        return 0.0

    t = np.arange(len(nino34), dtype=float)
    # Detrend
    result = linregress(t[valid], nino34[valid])
    detrended = nino34 - (result.slope * t + result.intercept)
    # Return trend of detrended (should be ~0, but captures residual)
    n_years = end_year - start_year
    t_centered = t - t.mean()
    result2 = linregress(t_centered[valid], detrended[valid])
    return result2.slope * n_years


def _compute_amv_index(ds, start_year, end_year, lats, lons):
    """Compute AMV index: N. Atlantic SST minus GMST."""
    from scipy.stats import linregress

    var_name = _find_sst_var(ds)
    if var_name is None:
        return 0.0

    da = ds[var_name].sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
    )

    # North Atlantic: lat 0-60, lon 280-360 (in 0-360)
    lon_vals = da.lon.values % 360
    lat_vals = da.lat.values
    lat_mask = (lat_vals >= 0) & (lat_vals <= 60)
    lon_mask = (lon_vals >= 280) | (lon_vals <= 0)

    if lat_mask.any() and lon_mask.any():
        na_sst = da.isel(lat=lat_mask, lon=lon_mask).mean(dim=["lat", "lon"]).values
    else:
        return 0.0

    # Global mean SST
    gmst = da.mean(dim=["lat", "lon"]).values

    amv = na_sst - gmst
    valid = ~np.isnan(amv)
    if valid.sum() < 24:
        return 0.0

    t = np.arange(len(amv), dtype=float)
    n_years = end_year - start_year
    t_centered = t - t.mean()
    result = linregress(t_centered[valid], amv[valid])
    return result.slope * n_years


def _compute_ipo_index(ds, start_year, end_year, lats, lons):
    """Compute simplified IPO tripole index."""
    from scipy.stats import linregress

    var_name = _find_sst_var(ds)
    if var_name is None:
        return 0.0

    da = ds[var_name].sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
    )
    lon_vals = da.lon.values % 360

    # Tropical Pacific (10S-10N, 170E-90W → 170-270)
    lat_mask_trop = (da.lat.values >= -10) & (da.lat.values <= 10)
    lon_mask_trop = (lon_vals >= 170) & (lon_vals <= 270)

    # North Pacific (25-45N, 140E-145W → 140-215)
    lat_mask_np = (da.lat.values >= 25) & (da.lat.values <= 45)
    lon_mask_np = (lon_vals >= 140) & (lon_vals <= 215)

    if not (lat_mask_trop.any() and lon_mask_trop.any() and
            lat_mask_np.any() and lon_mask_np.any()):
        return 0.0

    trop = da.isel(lat=lat_mask_trop, lon=lon_mask_trop).mean(dim=["lat", "lon"]).values
    npac = da.isel(lat=lat_mask_np, lon=lon_mask_np).mean(dim=["lat", "lon"]).values

    ipo = trop - npac
    valid = ~np.isnan(ipo)
    if valid.sum() < 24:
        return 0.0

    t = np.arange(len(ipo), dtype=float)
    n_years = end_year - start_year
    t_centered = t - t.mean()
    result = linregress(t_centered[valid], ipo[valid])
    return result.slope * n_years


def _find_sst_var(ds) -> Optional[str]:
    """Find the SST/tas variable name in a dataset."""
    for name in ("tos", "tas", "arr_EM"):
        if name in ds:
            return name
    return None


# ---------------------------------------------------------------------------
# Land fraction from data masks
# ---------------------------------------------------------------------------

def compute_land_fraction(
    eval_dir: Path,
    member: str,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Estimate land fraction per grid cell from tos (SST) NaN mask.
    Cells where tos is always NaN are land.
    """
    import xarray as xr

    tos_file = eval_dir / f"tos_mon_{member}.195001-202212.nc"
    if not tos_file.exists():
        return np.full(len(lats) * len(lons), 0.5)

    ds = xr.open_dataset(tos_file)
    tos = ds["tos"]
    # Fraction of time the cell has data (ocean) vs NaN (land)
    frac_valid = tos.count(dim="time").values / tos.sizes["time"]
    land_frac = 1.0 - frac_valid.ravel()
    ds.close()
    return land_frac


# ---------------------------------------------------------------------------
# AOI GeoPackage generation
# ---------------------------------------------------------------------------

def create_grid_geopackage(
    lats: np.ndarray,
    lons: np.ndarray,
    output_path: Path,
) -> None:
    """Create a GeoPackage with 2.5° grid cell polygons."""
    import geopandas as gpd
    from shapely.geometry import box

    dlat = abs(lats[1] - lats[0]) / 2 if len(lats) > 1 else 1.25
    dlon = abs(lons[1] - lons[0]) / 2 if len(lons) > 1 else 1.25

    records = []
    cell_id = 0
    for lat in lats:
        for lon in lons:
            # Normalize lon to -180..180 for GeoPackage
            lon_center = lon if lon <= 180 else lon - 360
            geom = box(
                lon_center - dlon, lat - dlat,
                lon_center + dlon, lat + dlat,
            )
            records.append({"grid_cell_id": cell_id, "geometry": geom})
            cell_id += 1

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(output_path, driver="GPKG")
    print(f"  Wrote AOI grid: {output_path} ({len(gdf)} cells)")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

# File naming patterns
EVAL_PATTERNS = {
    "tos": "tos_mon_{member}.195001-202212.nc",
    "tas": "tas_mon_{member}.195001-202212.nc",
    "pr": "pr_mon_{member}.195001-202212.nc",
    "psl": "psl_mon_{member}.195001-202212.nc",
    "monmaxtasmax": "monmaxtasmax_day_{member}.195001-202212.nc",
    "monmintasmin": "monmintasmin_day_{member}.195001-202212.nc",
    "monmaxpr": "monmaxpr_day_{member}.195001-202212.nc",
    "zmta": "zmta_mon_{member}.195001-202212.nc",
}

ENSMEAN_PATTERN = "{member}.{var}.EM.1950-2022.nc"

ALL_MEMBERS = ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1H", "1I", "1J"]
MODEL_MEMBERS = [m for m in ALL_MEMBERS if m != "1I"]  # 1I = reanalysis/obs


def process_member_variable(
    eval_dir: Path,
    ensmean_dir: Optional[Path],
    output_dir: Path,
    var: str,
    member: str,
    start_year: int,
    end_year: int,
    grid_df: pd.DataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    forcing_indices: Optional[dict] = None,
) -> Path:
    """Process one evaluation member × variable → CSV."""
    import xarray as xr

    eval_file = eval_dir / EVAL_PATTERNS[var].format(member=member)
    if not eval_file.exists():
        print(f"  SKIP: {eval_file.name} not found")
        return None

    print(f"  Processing {var} × {member}...")
    ds = xr.open_dataset(eval_file)

    # Find the data variable name (usually matches var name)
    data_vars = [v for v in ds.data_vars if v not in ("time_bnds",)]
    da = ds[data_vars[0]]

    # Compute trend field
    trends = compute_trend_field(da, start_year, end_year)
    ds.close()

    # Build output row per grid cell
    out = grid_df.copy()
    out[f"{var}_trend"] = trends

    # Merge forcing indices (same for all cells within one member)
    if forcing_indices is not None:
        for col, vals in forcing_indices.items():
            out[col] = vals

    # Drop cells with NaN target (e.g. land cells for tos)
    out = out.dropna(subset=[f"{var}_trend"])
    out = out.reset_index(drop=True)
    out["grid_cell_id"] = out.index

    csv_path = output_dir / f"forcesmip_{var}_{member}.csv"
    out.to_csv(csv_path, index=False)
    print(f"    → {csv_path.name} ({len(out)} cells)")

    # Process truth (ensemble mean) if available
    if ensmean_dir and member in MODEL_MEMBERS:
        em_file = ensmean_dir / ENSMEAN_PATTERN.format(member=member, var=var)
        if em_file.exists():
            ds_em = xr.open_dataset(em_file)
            # Ensemble mean uses 'arr_EM' as the variable name
            em_da = ds_em["arr_EM"]
            em_trends = compute_trend_field(em_da, start_year, end_year)
            ds_em.close()

            truth = grid_df.copy()
            truth[f"{var}_trend_forced"] = em_trends
            truth = truth.dropna(subset=[f"{var}_trend_forced"])
            truth = truth.reset_index(drop=True)
            truth["grid_cell_id"] = truth.index

            truth_path = output_dir / f"forcesmip_ensmean_{var}_{member}.csv"
            truth.to_csv(truth_path, index=False)
            print(f"    → {truth_path.name} (truth)")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert ForceSMIP Tier 1 NetCDF to SPARC CSV format"
    )
    parser.add_argument(
        "--eval-dir", required=True,
        help="Path to Evaluation-Tier1 directory",
    )
    parser.add_argument(
        "--ensmean-dir", default=None,
        help="Path to ensmeans-Tier1 directory (optional, for truth data)",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory for CSVs (default: data/)",
    )
    parser.add_argument(
        "--variables", nargs="+",
        default=["tos", "tas", "pr", "psl"],
        help="Target variables to process (default: tos tas pr psl)",
    )
    parser.add_argument(
        "--members", nargs="+", default=None,
        help="Evaluation members to process (default: all 10)",
    )
    parser.add_argument(
        "--trend-start", type=int, default=1980,
        help="Start year for trend computation (default: 1980)",
    )
    parser.add_argument(
        "--trend-end", type=int, default=2022,
        help="End year for trend computation (default: 2022)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    ensmean_dir = Path(args.ensmean_dir) if args.ensmean_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    members = args.members or ALL_MEMBERS
    variables = args.variables

    import xarray as xr

    # Get grid coordinates from first available file
    print("Reading grid coordinates...")
    sample_file = next(eval_dir.glob("*.nc"))
    ds_sample = xr.open_dataset(sample_file)
    lats = ds_sample["lat"].values
    lons = ds_sample["lon"].values
    ds_sample.close()
    print(f"  Grid: {len(lats)} lat × {len(lons)} lon = {len(lats)*len(lons)} cells")

    # Build grid dataframe with spatial covariates
    grid_df = build_grid_dataframe(lats, lons)

    # Create AOI GeoPackage
    gpkg_path = output_dir / "global_2p5deg_grid.gpkg"
    if not gpkg_path.exists():
        create_grid_geopackage(lats, lons, gpkg_path)

    # Process each member × variable
    total = len(members) * len(variables)
    done = 0
    for member in members:
        print(f"\n{'='*60}")
        print(f"Member: {member}")
        print(f"{'='*60}")

        # Compute land fraction from tos mask
        land_frac = compute_land_fraction(eval_dir, member, lats, lons)
        grid_df["land_fraction"] = land_frac

        # Update ocean basin: set land cells (land_frac > 0.5) to basin=0
        grid_df.loc[grid_df["land_fraction"] > 0.5, "ocean_basin"] = 0

        # Compute forcing indices using tas data for this member
        tas_file = eval_dir / EVAL_PATTERNS["tas"].format(member=member)
        forcing_indices = None
        if tas_file.exists():
            print("  Computing forcing indices...")
            ds_tas = xr.open_dataset(tas_file)
            forcing_indices = compute_forcing_indices(
                ds_tas, lats, lons, args.trend_start, args.trend_end,
            )
            ds_tas.close()

        for var in variables:
            process_member_variable(
                eval_dir, ensmean_dir, output_dir,
                var, member,
                args.trend_start, args.trend_end,
                grid_df, lats, lons,
                forcing_indices,
            )
            done += 1
            print(f"  Progress: {done}/{total}")

    print(f"\nDone. Output in: {output_dir}")


if __name__ == "__main__":
    main()
