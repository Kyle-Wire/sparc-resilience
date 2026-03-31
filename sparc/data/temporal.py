"""
Temporal data management for the SPARC spatiotemporal engine.

Handles panel-data loading, lag/rolling feature generation,
temporal train/test splitting, and aggregation for location×time
observations.
"""

import numpy as np
import pandas as pd


def is_temporal(config: dict) -> bool:
    """Return True if the project has temporal mode enabled."""
    return config.get('temporal', {}).get('enabled', False)


def parse_time_column(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Parse a time column into a pandas DatetimeIndex-compatible Series.

    Accepts ISO-8601 strings, Unix timestamps (int/float), or integer
    period indices (left as-is for ordering).
    """
    sample = df[time_col].dropna().iloc[:10]
    if pd.api.types.is_integer_dtype(sample):
        return df[time_col].astype(int)
    return pd.to_datetime(df[time_col])


def sort_panel(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """Sort a panel DataFrame by location then time."""
    return df.sort_values([id_col, time_col]).reset_index(drop=True)


def build_lag_features(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    variables: list[str],
    lags: list[int],
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate lagged and rolling-window features per location.

    Parameters
    ----------
    df : DataFrame
        Panel data sorted by (id_col, time_col).
    id_col : str
        Column identifying unique spatial locations.
    time_col : str
        Column identifying the time index.
    variables : list[str]
        Predictor columns to lag.
    lags : list[int]
        Lag steps (e.g. [1, 2, 3]).
    rolling_windows : list[int] | None
        Rolling window sizes for mean/std features.

    Returns
    -------
    DataFrame
        Original df with new lag/rolling columns appended.
        Rows with NaN from lagging are **not** dropped (caller decides).
    """
    df = df.copy()
    grouped = df.groupby(id_col)

    for var in variables:
        for lag in lags:
            col_name = f"{var}_lag{lag}"
            df[col_name] = grouped[var].shift(lag)

        if rolling_windows:
            for win in rolling_windows:
                df[f"{var}_rmean{win}"] = grouped[var].transform(
                    lambda s: s.shift(1).rolling(win, min_periods=1).mean()
                )
                df[f"{var}_rstd{win}"] = grouped[var].transform(
                    lambda s: s.shift(1).rolling(win, min_periods=1).std()
                )

    return df


def apply_lag_config(
    df: pd.DataFrame,
    config: dict,
    id_col: str,
    time_col: str,
    predictor_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Read the ``temporal.lag_features`` config and apply lag/rolling generation.

    Returns
    -------
    (df_lagged, new_feature_names)
        The augmented DataFrame and the list of newly created column names.
    """
    lag_cfg = config.get('temporal', {}).get('lag_features', {})
    if not lag_cfg.get('enabled', True):
        return df, []

    lags = lag_cfg.get('lags', [1, 2, 3])
    rolling_windows = lag_cfg.get('rolling_windows', [])
    variables = lag_cfg.get('variables') or predictor_cols

    # Only lag variables actually present in df
    variables = [v for v in variables if v in df.columns]
    if not variables:
        return df, []

    before_cols = set(df.columns)
    df = build_lag_features(df, id_col, time_col, variables, lags, rolling_windows)
    new_cols = sorted(set(df.columns) - before_cols)

    return df, new_cols


def temporal_train_test_split(
    df: pd.DataFrame,
    time_col: str,
    training_window: int | None = None,
    validation_window: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Split panel data into training and validation sets by time.

    Parameters
    ----------
    df : DataFrame
        Panel data with a parsed time column.
    time_col : str
        Name of time column.
    training_window : int | None
        Use only the last *training_window* time steps for training.
    validation_window : int | None
        Hold out the last *validation_window* time steps for validation.

    Returns
    -------
    (train_df, val_df)
        val_df is None when no validation_window is specified.
    """
    unique_times = np.sort(df[time_col].unique())
    n = len(unique_times)

    if validation_window and validation_window < n:
        cutoff = unique_times[-(validation_window)]
        val_df = df[df[time_col] >= cutoff].copy()
        train_df = df[df[time_col] < cutoff].copy()
    else:
        train_df = df.copy()
        val_df = None

    if training_window and training_window < len(np.sort(train_df[time_col].unique())):
        train_times = np.sort(train_df[time_col].unique())
        keep_after = train_times[-(training_window)]
        train_df = train_df[train_df[time_col] >= keep_after].copy()

    return train_df, val_df


def aggregate_temporal(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    method: str = "mean",
    coord_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collapse multi-temporal observations to one row per location.

    Parameters
    ----------
    df : DataFrame
        Panel data.
    id_col : str
        Location identifier.
    time_col : str
        Time column (excluded from aggregation).
    method : str
        Aggregation method: mean, median, max, min, trend_slope.
    coord_cols : list[str] | None
        Coordinate columns to preserve via first-value.

    Returns
    -------
    DataFrame
        One row per location.
    """
    if method == "none":
        return df

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col and c != time_col]
    skip_cols = set(coord_cols or [])

    if method == "trend_slope":
        # Per-location OLS slope for each numeric predictor
        def _slope(group):
            t = np.arange(len(group), dtype=float)
            result = {}
            for col in numeric_cols:
                if col in skip_cols:
                    result[col] = group[col].iloc[0]
                    continue
                vals = group[col].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.sum() < 2:
                    result[col] = np.nan
                else:
                    result[col] = np.polyfit(t[mask], vals[mask], 1)[0]
            return pd.Series(result)

        agg = df.groupby(id_col).apply(_slope, include_groups=False).reset_index()
    else:
        agg_func = {"mean": "mean", "median": "median", "max": "max", "min": "min"}[method]
        agg_dict = {c: "first" if c in skip_cols else agg_func for c in numeric_cols}
        agg = df.groupby(id_col).agg(agg_dict).reset_index()

    # Ensure identifier column preserved
    if id_col not in agg.columns:
        agg[id_col] = df.groupby(id_col).first().index

    return agg


def prepare_temporal_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    High-level entry point: prepare a raw DataFrame for spatiotemporal modeling.

    Steps:
    1. Parse and sort by time
    2. Generate lag/rolling features
    3. Optionally aggregate to cross-sectional
    4. Drop rows with NaN from lagging

    Returns
    -------
    (prepared_df, new_feature_names)
    """
    tcfg = config['temporal']
    time_col = tcfg['time_column']
    id_col = config['data'].get('identifier_column', 'OBJECTID')
    predictor_cols = config['predictors']['base_model']

    # Parse time
    df[time_col] = parse_time_column(df, time_col)
    df = sort_panel(df, id_col, time_col)

    # Lag features
    df, new_features = apply_lag_config(df, config, id_col, time_col, predictor_cols)

    # Aggregation
    agg_cfg = tcfg.get('aggregation', {})
    agg_method = agg_cfg.get('method', 'none')
    if agg_method != 'none':
        coord_cols = config['data'].get('coord_columns', [])
        df = aggregate_temporal(df, id_col, time_col, agg_method, coord_cols)
        # After aggregation, lag features are baked in — clear new_features since panel collapsed
        new_features = []

    # Drop NaN rows created by lagging (only in panel mode)
    if agg_method == 'none' and new_features:
        df = df.dropna(subset=new_features).reset_index(drop=True)

    return df, new_features
