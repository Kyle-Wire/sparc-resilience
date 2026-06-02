"""
This module provides standardized functions for data loading, cleaning,
and preprocessing for the SPARC pipeline. It ensures that data is consistently
prepared for all subsequent modeling stages.

Includes GeoParquet caching for 3-5x faster inter-stage data transfer in
the subprocess-based pipeline orchestrator.
"""
import hashlib
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from pathlib import Path

from sparc.data.temporal import is_temporal, prepare_temporal_data

# ---------------------------------------------------------------------------
# GeoParquet caching — avoids re-parsing CSV across subprocess stages
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "_sparc_prepared_data.parquet"


def _cache_key(raw_data_path: str, predictor_cols: list | None = None) -> str:
    """Deterministic hash of the source CSV (path + mtime + size + predictors)."""
    p = Path(raw_data_path)
    try:
        stat = p.stat()
        token = f"{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        # File doesn't exist yet; use path string only so caller can still
        # attempt a cache lookup (which will miss) without raising early.
        token = str(p.resolve())
    if predictor_cols:
        token += "|" + ",".join(sorted(predictor_cols))
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _cache_path(output_dir: str, raw_data_path: str,
                predictor_cols: list | None = None) -> Path:
    """Return the full path to the cached parquet file."""
    return Path(output_dir) / f"{_cache_key(raw_data_path, predictor_cols)}_{_CACHE_FILENAME}"


def save_prepared_geodataframe(gdf: gpd.GeoDataFrame, output_dir: str,
                               raw_data_path: str,
                               predictor_cols: list | None = None) -> Path:
    """
    Cache a prepared GeoDataFrame as GeoParquet for fast reloading.

    Parameters
    ----------
    gdf : GeoDataFrame
        The prepared data to cache.
    output_dir : str
        Directory to write the cache file into.
    raw_data_path : str
        Path to the original CSV (used to build the cache key).
    predictor_cols : list, optional
        Predictor column names (included in cache key for invalidation).

    Returns
    -------
    Path to the written parquet file.
    """
    if len(gdf) == 0:
        raise ValueError(
            "Refusing to cache an empty GeoDataFrame. "
            "Fix the CRS mismatch or data issue before re-running."
        )
    out = _cache_path(output_dir, raw_data_path, predictor_cols)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out, index=False)
    print(f"    Cached prepared data → {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def load_cached_geodataframe(output_dir: str, raw_data_path: str,
                             predictor_cols: list | None = None):
    """
    Load a previously cached GeoDataFrame if it exists and is still valid.

    Returns
    -------
    GeoDataFrame or None
        The cached data, or None if no valid cache exists.
    """
    cache = _cache_path(output_dir, raw_data_path, predictor_cols)
    if cache.exists():
        try:
            gdf = gpd.read_parquet(cache)
            print(f"    Loaded cached data ← {cache.name} ({len(gdf)} rows)")
            # Guard against non-finite projected coords baked into an old cache
            # before this check was added (e.g. from a stale pre-fix parquet).
            if 'projected_X' in gdf.columns and 'projected_Y' in gdf.columns:
                finite_mask = (
                    np.isfinite(gdf['projected_X'].values)
                    & np.isfinite(gdf['projected_Y'].values)
                )
                n_inf = int((~finite_mask).sum())
                if n_inf > 0:
                    print(f"    Removed {n_inf} rows with non-finite projected coords from cache")
                    gdf = gdf[finite_mask].copy()
            # If filtering emptied the cache entirely, fall back to CSV so the
            # data is re-projected from scratch (the cache was fully corrupt).
            if len(gdf) == 0:
                print(f"    Cache is empty after filtering — falling back to CSV")
                try:
                    cache.unlink()
                except Exception:
                    pass
                return None
            return gdf
        except Exception as exc:
            print(f"    Cache read failed ({exc}), falling back to CSV")
    return None


# ---------------------------------------------------------------------------
# Dask-aware CSV loading for large datasets (>50 K rows)
# ---------------------------------------------------------------------------

_DASK_ROW_THRESHOLD = 50_000


def _read_csv_smart(path: str, low_memory: bool = False) -> pd.DataFrame:
    """
    Read a CSV, using dask for chunked I/O when the file is large.

    Falls back to plain pandas if dask is not installed or the file is
    small enough to fit comfortably in memory.
    """
    # Quick row-count heuristic: check file size (assume ~200 bytes/row)
    file_size_mb = os.path.getsize(path) / 1e6
    estimated_rows = int(file_size_mb * 1e6 / 200)

    if estimated_rows > _DASK_ROW_THRESHOLD:
        try:
            import dask.dataframe as dd
            print(f"    Large file detected (~{estimated_rows:,} rows est.) — using dask for chunked read")
            ddf = dd.read_csv(path, low_memory=low_memory, blocksize="64MB")
            return ddf.compute()
        except ImportError:
            pass  # dask not installed — fall through to pandas
        except Exception as exc:
            print(f"    Dask read failed ({exc}), falling back to pandas")

    # Try common encodings in order.  Many real-world CSVs exported from
    # Excel or legacy tools use Latin-1 / CP-1252 rather than UTF-8.
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, low_memory=low_memory, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"Cannot decode '{path}' with any supported encoding "
        "(UTF-8, UTF-8-BOM, Latin-1, CP-1252). "
        "Please save the file as UTF-8 and retry."
    )

def load_and_preprocess_data(
    raw_data_path: str,
    identifier_col: str,
    target_col: str,
    coords_cols: list,
    predictor_cols: list,
    initial_crs: str,
    target_crs: str,
    output_dir: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Loads raw data from a CSV, performs cleaning, type conversion,
    and projects it to the target Coordinate Reference System (CRS).

    This function consolidates the initial data preparation steps found
    across the original analysis scripts into a single, reusable utility.

    Args:
        raw_data_path (str): The full path to the raw input CSV file.
        identifier_col (str): The name of the unique identifier column.
        target_col (str): The name of the target variable column.
        coords_cols (list): A list containing the names of the X and Y coordinate columns.
        predictor_cols (list): A list of predictor column names to be used.
        initial_crs (str): The Coordinate Reference System of the source data (e.g., "EPSG:3438").
        target_crs (str): The target CRS for projection (e.g., "EPSG:26919").

    Returns:
        gpd.GeoDataFrame: A cleaned, projected GeoDataFrame with 'projected_X' and
                          'projected_Y' columns, ready for feature engineering and modeling.
                          
    Raises:
        FileNotFoundError: If the raw_data_path does not exist.
        ValueError: If essential columns are missing from the raw data.
    """
    print(f"--- Starting Data Loading and Preprocessing ---")
    print(f"    Loading data from: {raw_data_path}")

    # Validate the data file exists before any cache operations.  Doing this
    # first surfaces a clear FileNotFoundError rather than an opaque OS error
    # from _cache_key's stat() call (e.g. [WinError 3] when parent dir is missing).
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Data file not found: '{raw_data_path}'\n"
            "Check the 'data.file_path' setting in your project.yml — "
            "paths are resolved relative to the project.yml directory."
        )

    # GeoParquet fast-path: already preprocessed, skip CSV pipeline entirely
    if raw_data_path.endswith(('.geoparquet', '.parquet')):
        import geopandas as _gpd
        gdf = _gpd.read_parquet(raw_data_path)
        # Ensure projected_X / projected_Y exist (multi-city pipeline uses centroid_x/y)
        if 'projected_X' not in gdf.columns and 'centroid_x' in gdf.columns:
            gdf['projected_X'] = gdf['centroid_x']
            gdf['projected_Y'] = gdf['centroid_y']
        elif 'projected_X' not in gdf.columns and gdf.geometry is not None:
            gdf['projected_X'] = gdf.geometry.x
            gdf['projected_Y'] = gdf.geometry.y
        # Drop rows missing target or predictors
        essential = [c for c in [identifier_col, target_col] + coords_cols if c in gdf.columns]
        gdf = gdf.dropna(subset=essential)
        print(f"    Loaded GeoParquet. Shape: {gdf.shape}")
        print(f"--- Finished Data Loading (GeoParquet) ---")
        return gdf

    # Check for cached GeoParquet first (3-5x faster than CSV re-parse)
    if output_dir is not None:
        cached = load_cached_geodataframe(output_dir, raw_data_path, predictor_cols)
        if cached is not None:
            print(f"--- Finished Data Loading (from cache) ---")
            return cached

    data = _read_csv_smart(raw_data_path)
    print(f"    Successfully loaded raw data. Initial shape: {data.shape}")

    essential_cols = [identifier_col, target_col] + coords_cols + predictor_cols
    
    # --- Data Cleaning and Type Conversion ---
    missing_cols = [col for col in essential_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"ERROR: Essential columns missing from data: {missing_cols}")

    for col in predictor_cols + [target_col] + coords_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col].dtype):
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    original_rows = len(data)
    data.dropna(subset=essential_cols, inplace=True)
    print(f"    Shape after dropping NaNs in essential columns: {data.shape} (dropped {original_rows - len(data)} rows)")

    if data.empty:
        raise ValueError("ERROR: DataFrame is empty after dropping NaNs. Cannot proceed.")

    # --- Geospatial Transformation ---
    cols_to_keep = list(set([identifier_col] + coords_cols + predictor_cols + [target_col]))
    gdf = gpd.GeoDataFrame(
        data[cols_to_keep].copy(),
        geometry=gpd.points_from_xy(data[coords_cols[0]], data[coords_cols[1]]),
        crs=initial_crs
    )

    if str(gdf.crs).upper() != str(target_crs).upper():
        print(f"    Projecting GeoDataFrame from {gdf.crs} to {target_crs}...")
        gdf = gdf.to_crs(target_crs)

    gdf['projected_X'] = gdf.geometry.x
    gdf['projected_Y'] = gdf.geometry.y

    # CRS reprojection can produce ±inf for points outside the projection's valid
    # domain.  Drop those rows now so no downstream stage sees non-finite coords.
    finite_mask = np.isfinite(gdf['projected_X'].values) & np.isfinite(gdf['projected_Y'].values)
    n_inf = int((~finite_mask).sum())
    if n_inf > 0:
        print(f"    Dropped {n_inf} rows with non-finite projected coordinates after CRS transform")
        gdf = gdf[finite_mask].copy()

    if gdf.empty:
        raise ValueError(
            f"All {original_rows} rows were dropped because their projected coordinates "
            f"are non-finite (inf/NaN) after reprojecting from '{initial_crs}' to "
            f"'{target_crs}'.\n"
            "This almost always means the 'crs.input' setting in your project.yml does "
            "not match the actual coordinate system of the source CSV.\n"
            f"  • Your CSV has coordinates like X={data[coords_cols[0]].iloc[0]:.2f}, "
            f"Y={data[coords_cols[1]].iloc[0]:.2f}\n"
            f"  • Your project says they are in {initial_crs}\n"
            "Tip: Rhode Island State Plane (feet) data uses EPSG:3438.  "
            "WGS84 lat/lon data uses EPSG:4326."
        )

    print(f"    Data preparation complete. Final shape: {gdf.shape}")

    # Write cache for downstream stages
    if output_dir is not None:
        try:
            save_prepared_geodataframe(gdf, output_dir, raw_data_path,
                                       predictor_cols)
        except Exception as exc:
            print(f"    Warning: could not write cache ({exc})")

    print(f"--- Finished Data Loading and Preprocessing ---")
    
    return gdf


def load_and_preprocess_panel(config: dict) -> tuple[gpd.GeoDataFrame, list[str]]:
    """
    Config-driven loader that handles both cross-sectional and panel data.

    When ``config['temporal']['enabled']`` is True, this:
    1. Loads the CSV preserving the time column
    2. Runs temporal preparation (lag/rolling features, optional aggregation)
    3. Projects to GeoDataFrame

    Returns
    -------
    (gdf, temporal_feature_names)
        temporal_feature_names is empty for cross-sectional data.
    """
    data_cfg = config['data']
    file_path = data_cfg['file_path']
    target_col = data_cfg['target_column']
    id_col = data_cfg.get('identifier_column', 'OBJECTID')
    coord_cols = list(data_cfg['coord_columns'])
    predictor_cols = list(config['predictors']['base_model'])
    input_crs = config['crs']['initial']
    proj_crs = config['crs']['target_projected']

    if not is_temporal(config):
        gdf = load_and_preprocess_data(
            file_path, id_col, target_col, coord_cols,
            predictor_cols, input_crs, proj_crs,
        )
        return gdf, []

    # --- Temporal path ---
    time_col = config['temporal']['time_column']
    print(f"--- Loading Panel Data (temporal mode) ---")
    print(f"    Time column: {time_col}")

    df = _read_csv_smart(file_path)
    print(f"    Raw shape: {df.shape}")

    # Ensure essential columns exist
    essential = [id_col, target_col, time_col] + coord_cols + predictor_cols
    missing = [c for c in essential if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Numeric conversion (skip time column)
    for col in predictor_cols + [target_col] + coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    original = len(df)
    df.dropna(subset=[id_col, target_col] + coord_cols, inplace=True)
    print(f"    Dropped {original - len(df)} rows with NaN in core columns")

    # Temporal preparation (lags, rolling, aggregation)
    df, new_features = prepare_temporal_data(df, config)
    print(f"    After temporal processing: {df.shape}, {len(new_features)} new features")

    # Project to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[coord_cols[0]], df[coord_cols[1]]),
        crs=input_crs,
    )
    if str(gdf.crs).upper() != str(proj_crs).upper():
        gdf = gdf.to_crs(proj_crs)
    gdf['projected_X'] = gdf.geometry.x
    gdf['projected_Y'] = gdf.geometry.y

    # Drop rows where projection produced ±inf (e.g. points outside CRS domain).
    finite_mask = np.isfinite(gdf['projected_X'].values) & np.isfinite(gdf['projected_Y'].values)
    n_inf = int((~finite_mask).sum())
    if n_inf > 0:
        print(f"    Dropped {n_inf} rows with non-finite projected coordinates after CRS transform")
        gdf = gdf[finite_mask].copy()

    print(f"    Final panel GeoDataFrame: {gdf.shape}")
    return gdf, new_features


def load_data(file_path):
    """Simple data loading function for pipeline stages."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    data = _read_csv_smart(file_path)
    print(f"Loaded data with shape: {data.shape}")
    return data


def prepare_data(df, config):
    """
    Prepare data for modeling stages.
    
    Returns:
        X: Feature matrix
        y: Target vector  
        coords: Coordinate matrix
        feature_names: List of feature names
    """
    # Get configuration values
    target_col = config['data']['target_column']
    coord_cols = config['data']['coord_columns'] 
    predictor_cols = config['predictors']['gwen']  # Use GWEN predictors for variable selection
    
    # Handle missing values
    essential_cols = [target_col] + coord_cols + predictor_cols
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Convert to numeric and handle inf values
    for col in predictor_cols + [target_col] + coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing values in essential columns
    df_clean = df.dropna(subset=essential_cols).copy()
    print(f"Data shape after cleaning: {df_clean.shape}")
    
    if df_clean.empty:
        raise ValueError("No data remaining after cleaning")
    
    # Extract features, target, and coordinates
    X = df_clean[predictor_cols].values
    y = df_clean[target_col].values
    coords = df_clean[coord_cols].values
    feature_names = predictor_cols
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Coordinates shape: {coords.shape}")
    
    return X, y, coords, feature_names
