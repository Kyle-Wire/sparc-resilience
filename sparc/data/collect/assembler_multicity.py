"""
assembler_multicity.py — Assemble per-city collected layers into a canonical GeoParquet.

Merges individual layer GeoDataFrames (fishnet + NLCD, Landsat, Sentinel-2,
Buildings, DEM, Equity, CAPA, ERA5 boundary) into a single flat GeoParquet
with a canonical column order defined by the pilot config schema.

The schema is intentionally flat and schema-checked: missing columns are
filled with NaN, extra columns are dropped. This guarantees all city
GeoParquets have identical schemas and can be concat'd for JEPA training.

Usage
-----
    from sparc.data.collect.assembler_multicity import assemble_city_geoparquet

    gdf = assemble_city_geoparquet(
        fishnet_gdf=fishnet,
        layers={
            "nlcd": nlcd_gdf,
            "landsat": landsat_gdf,
            "sentinel2": s2_gdf,
            "buildings": bldg_gdf,
            "dem": dem_gdf,
            "equity": equity_gdf,
            "capa": capa_gdf,
            "era5_boundary": era5_boundary_gdf,
        },
        city_slug="philadelphia_pa",
        utm_epsg="EPSG:32618",
        campaign_date=date(2022, 9, 21),
        feature_columns=None,   # None = use CANONICAL_SCHEMA default
        label_columns=None,
    )
    gdf.to_parquet(output_path, index=False)
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

# Ordered list of every expected column (excluding geometry which leads).
# Scripts and training code may rely on this order.
CANONICAL_SCHEMA: list[str] = [
    # ── Metadata ───────────────────────────────────────────────────────────
    "city_slug",
    "utm_epsg",
    "campaign_date",
    "centroid_x",
    "centroid_y",
    # ── Land surface (NLCD) ────────────────────────────────────────────────
    "pct_impervious",
    "pct_canopy",
    "land_cover",
    # ── Landsat spectral / thermal ─────────────────────────────────────────
    "ndvi",
    "ndbi",
    "lst",
    "albedo",
    # ── Sentinel-2 spectral ────────────────────────────────────────────────
    "ndvi_s2",
    "ndbi_s2",
    "mndwi_s2",
    # ── Buildings ─────────────────────────────────────────────────────────
    "bldg_height_mean",
    "bldg_coverage",
    "svf",
    # ── Equity ────────────────────────────────────────────────────────────
    "cdc_svi",
    # ── DEM / terrain ─────────────────────────────────────────────────────
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "distance_water_m",
    # ── ERA5 boundary — morning ───────────────────────────────────────────
    "era5_morning_t2m",
    "era5_morning_windspeed",
    "era5_morning_winddir",
    "era5_morning_rh",
    # ── ERA5 boundary — midday ────────────────────────────────────────────
    "era5_midday_t2m",
    "era5_midday_windspeed",
    "era5_midday_winddir",
    "era5_midday_rh",
    # ── ERA5 boundary — evening ───────────────────────────────────────────
    "era5_evening_t2m",
    "era5_evening_windspeed",
    "era5_evening_winddir",
    "era5_evening_rh",
    # ── Labels (CAPA raster) ──────────────────────────────────────────────
    "aat_morning",
    "aat_midday",
    "aat_night",
    "diurnal_aat",
]

# Maps layer name → columns that layer GDF contributes.
LAYER_COLUMN_MAP: dict[str, list[str]] = {
    "nlcd":         ["pct_impervious", "pct_canopy", "land_cover"],
    "landsat":      ["ndvi", "ndbi", "lst", "albedo"],
    "sentinel2":    ["ndvi_s2", "ndbi_s2", "mndwi_s2"],
    "buildings":    ["bldg_height_mean", "bldg_coverage", "svf"],
    "dem":          ["elevation_m", "slope_deg", "aspect_deg", "distance_water_m"],
    "equity":       ["cdc_svi"],
    "capa":         ["aat_morning", "aat_midday", "aat_night", "diurnal_aat"],
    "era5_boundary": [
        "era5_morning_t2m", "era5_morning_windspeed",
        "era5_morning_winddir", "era5_morning_rh",
        "era5_midday_t2m",  "era5_midday_windspeed",
        "era5_midday_winddir",  "era5_midday_rh",
        "era5_evening_t2m", "era5_evening_windspeed",
        "era5_evening_winddir", "era5_evening_rh",
    ],
}

_LABEL_COLUMNS = ["aat_morning", "aat_midday", "aat_night", "diurnal_aat"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_city_geoparquet(
    fishnet_gdf: object,
    layers: dict[str, object],
    city_slug: str,
    utm_epsg: str,
    campaign_date: date,
    feature_columns: Optional[list[str]] = None,
    label_columns: Optional[list[str]] = None,
) -> object:
    """Assemble per-layer GeoDataFrames into a canonical flat GeoParquet.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        The base fishnet grid.  All layer GDFs must share the same row order
        and index as this fishnet.
    layers : dict[str, gpd.GeoDataFrame]
        Mapping from layer name (see :data:`LAYER_COLUMN_MAP`) to the
        corresponding GeoDataFrame returned by the collect functions.
    city_slug : str
        Short lowercase identifier, e.g. ``"philadelphia_pa"``.
    utm_epsg : str
        Projected CRS for this city, e.g. ``"EPSG:32618"``.
    campaign_date : date
        CAPA campaign date used to populate the ``campaign_date`` column.
    feature_columns : list[str] | None
        If given, only these feature columns are retained (plus labels and
        metadata).  *None* uses the full :data:`CANONICAL_SCHEMA`.
    label_columns : list[str] | None
        If given, validate that these label columns are present and non-all-NaN.
        *None* defaults to ``["aat_morning", "aat_midday", "aat_night", "diurnal_aat"]``.

    Returns
    -------
    gpd.GeoDataFrame
        Flat GeoParquet in *utm_epsg* CRS with canonical column order.
    """
    import geopandas as gpd
    import pandas as pd

    if label_columns is None:
        label_columns = _LABEL_COLUMNS

    # Start from the fishnet geometry, reset index for safe column assignment
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    gdf = gdf.reset_index(drop=True)

    # Ensure we're in the correct projected CRS
    if str(gdf.crs) != utm_epsg:
        try:
            gdf = gdf.to_crs(utm_epsg)
        except Exception as exc:
            log.warning("assemble: could not reproject fishnet to %s: %s", utm_epsg, exc)

    # Attach columns from each layer by index alignment
    for layer_name, col_names in LAYER_COLUMN_MAP.items():
        layer_gdf = layers.get(layer_name)
        if layer_gdf is None:
            log.warning("assemble: layer '%s' not provided — filling NaN", layer_name)
            for col in col_names:
                gdf[col] = float("nan")  # type: ignore[index]
            continue

        try:
            layer_df = _coerce_to_dataframe(layer_gdf)
            layer_df = layer_df.reset_index(drop=True)

            # Handle equity: may have holc_grade / ejscreen_score — we only want cdc_svi
            if layer_name == "equity":
                _attach_equity_columns(gdf, layer_df, col_names)
            else:
                for col in col_names:
                    if col in layer_df.columns:
                        gdf[col] = layer_df[col].values  # type: ignore[index]
                    else:
                        # Try alternative column names for landsat albedo
                        alt = _find_alt_column(col, layer_df)
                        if alt is not None:
                            gdf[col] = layer_df[alt].values  # type: ignore[index]
                            log.debug(
                                "assemble: mapped '%s' → '%s' in layer '%s'",
                                alt, col, layer_name,
                            )
                        else:
                            log.warning(
                                "assemble: column '%s' missing from layer '%s' — filling NaN",
                                col, layer_name,
                            )
                            gdf[col] = float("nan")  # type: ignore[index]
        except Exception as exc:
            log.warning(
                "assemble: error processing layer '%s' (%s) — filling NaN",
                layer_name, exc,
            )
            for col in col_names:
                if col not in gdf.columns:  # type: ignore[union-attr]
                    gdf[col] = float("nan")  # type: ignore[index]

    # Add metadata columns
    gdf["city_slug"]     = city_slug  # type: ignore[index]
    gdf["utm_epsg"]      = utm_epsg  # type: ignore[index]
    gdf["campaign_date"] = str(campaign_date)  # type: ignore[index]

    # Compute centroid_x / centroid_y in the projected CRS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centroids = gdf.geometry.centroid  # type: ignore[union-attr]
        gdf["centroid_x"] = centroids.x  # type: ignore[index]
        gdf["centroid_y"] = centroids.y  # type: ignore[index]
    except Exception as exc:
        log.warning("assemble: centroid computation failed (%s)", exc)
        gdf["centroid_x"] = float("nan")  # type: ignore[index]
        gdf["centroid_y"] = float("nan")  # type: ignore[index]

    # Enforce canonical column order
    # Build ordered list: geometry first, then schema columns present in gdf
    all_canonical = CANONICAL_SCHEMA if feature_columns is None else _filter_schema(
        feature_columns, label_columns,
    )
    ordered_cols = [c for c in all_canonical if c in gdf.columns]  # type: ignore[union-attr]
    missing_cols = [c for c in all_canonical if c not in gdf.columns]  # type: ignore[union-attr]
    for col in missing_cols:
        gdf[col] = float("nan")  # type: ignore[index]
        ordered_cols.append(col)

    gdf = gdf[["geometry"] + all_canonical]  # type: ignore[index]

    # Validate labels
    _validate_labels(gdf, label_columns, city_slug)

    return gdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_to_dataframe(layer_gdf: object) -> object:
    """Return a plain DataFrame (strip geometry if needed)."""
    import pandas as pd
    try:
        import geopandas as gpd
        if isinstance(layer_gdf, gpd.GeoDataFrame):
            return pd.DataFrame(layer_gdf.drop(columns=["geometry"], errors="ignore"))
    except ImportError:
        pass
    return layer_gdf  # type: ignore[return-value]


def _attach_equity_columns(gdf: object, layer_df: object, col_names: list[str]) -> None:
    """Attach equity columns, filtering to only what's in col_names (i.e. cdc_svi)."""
    for col in col_names:
        if col in layer_df.columns:  # type: ignore[union-attr]
            gdf[col] = layer_df[col].values  # type: ignore[index,union-attr]
        else:
            gdf[col] = float("nan")  # type: ignore[index]


def _find_alt_column(col: str, df: object) -> Optional[str]:
    """Try to find an alternative column name in *df* for a missing *col*."""
    _ALTERNATIVES: dict[str, list[str]] = {
        "albedo": ["surface_reflectance", "albedo_mean", "broad_albedo"],
        "lst":    ["land_surface_temp", "lst_celsius", "land_surface_temperature"],
        "ndvi":   ["NDVI"],
        "ndbi":   ["NDBI"],
    }
    candidates = _ALTERNATIVES.get(col, [])
    for alt in candidates:
        if alt in df.columns:  # type: ignore[union-attr]
            return alt
    return None


def _filter_schema(feature_columns: list[str], label_columns: list[str]) -> list[str]:
    """Build an ordered schema list from custom feature + label columns."""
    metadata = ["city_slug", "utm_epsg", "campaign_date", "centroid_x", "centroid_y"]
    return metadata + feature_columns + label_columns


def _validate_labels(gdf: object, label_columns: list[str], city_slug: str) -> None:
    """Warn if any label column is missing or all-NaN."""
    for col in label_columns:
        if col not in gdf.columns:  # type: ignore[union-attr]
            log.warning(
                "assemble [%s]: label column '%s' missing from output", city_slug, col,
            )
            continue
        series = gdf[col]  # type: ignore[index]
        try:
            if series.isna().all():
                log.warning(
                    "assemble [%s]: label column '%s' is all-NaN — "
                    "CAPA download may have failed",
                    city_slug, col,
                )
        except Exception:
            pass
