"""Pre-flight data validation for the SPARC pipeline.

Checks for common data quality issues before processing begins,
surfacing them as a structured checklist the user can resolve in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class ValidationItem:
    """Single checklist item."""
    id: str
    category: str  # "missing", "crs", "duplicate", "distribution", "type"
    severity: str  # "critical", "warning", "info"
    column: str | None = None
    message: str = ""
    detail: str = ""
    value: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Full pre-flight validation report."""
    passed: bool = True
    n_critical: int = 0
    n_warning: int = 0
    n_info: int = 0
    items: list[ValidationItem] = field(default_factory=list)
    summary: str = ""

    def add(self, item: ValidationItem) -> None:
        self.items.append(item)
        if item.severity == "critical":
            self.n_critical += 1
            self.passed = False
        elif item.severity == "warning":
            self.n_warning += 1
        else:
            self.n_info += 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = (
            f"{self.n_critical} critical, {self.n_warning} warnings, "
            f"{self.n_info} info — {'PASS' if self.passed else 'NEEDS REVIEW'}"
        )
        return d


# ------------------------------------------------------------------
# Validation functions
# ------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    predictor_cols: list[str] | None = None,
    coord_cols: list[str] | None = None,
    expected_crs: str | None = None,
    missing_threshold: float = 0.05,
) -> ValidationReport:
    """Run all pre-flight checks and return a structured report.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        The dataset to validate.
    target_col : str, optional
        Name of the target/outcome column.
    predictor_cols : list[str], optional
        Predictor column names to validate.
    coord_cols : list[str], optional
        Coordinate column names (x, y).
    expected_crs : str, optional
        Expected CRS string (e.g. "EPSG:4326").
    missing_threshold : float
        Flag columns with more than this fraction missing (default 5%).
    """
    report = ValidationReport()

    all_cols = []
    if target_col:
        all_cols.append(target_col)
    if predictor_cols:
        all_cols.extend(predictor_cols)

    # 1. Check for required columns existing
    _check_columns_exist(df, all_cols, coord_cols, report)

    # 2. Missing values
    _check_missing_values(df, all_cols, missing_threshold, report)

    # 3. Coordinate system consistency
    _check_crs(df, expected_crs, report)

    # 4. Duplicate records
    _check_duplicates(df, coord_cols, report)

    # 5. Distribution anomalies
    _check_distributions(df, predictor_cols or [], report)

    # 6. Column type validation
    _check_numeric_types(df, all_cols, report)

    return report


def _check_columns_exist(
    df: pd.DataFrame,
    value_cols: list[str],
    coord_cols: list[str] | None,
    report: ValidationReport,
) -> None:
    """Verify all expected columns exist in the dataframe."""
    all_expected = list(value_cols)
    if coord_cols:
        all_expected.extend(coord_cols)

    for col in all_expected:
        if col not in df.columns:
            report.add(ValidationItem(
                id=f"missing_col_{col}",
                category="column",
                severity="critical",
                column=col,
                message=f"Required column '{col}' not found in dataset",
                detail=f"Available columns: {', '.join(df.columns[:20])}",
            ))


def _check_missing_values(
    df: pd.DataFrame,
    cols: list[str],
    threshold: float,
    report: ValidationReport,
) -> None:
    """Flag columns with missing values above threshold."""
    n_rows = len(df)
    if n_rows == 0:
        report.add(ValidationItem(
            id="empty_dataset",
            category="missing",
            severity="critical",
            message="Dataset has 0 rows",
        ))
        return

    for col in cols:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        pct = n_missing / n_rows
        if pct > threshold:
            report.add(ValidationItem(
                id=f"missing_{col}",
                category="missing",
                severity="critical" if pct > 0.20 else "warning",
                column=col,
                message=f"'{col}' has {pct:.1%} missing values ({n_missing:,} / {n_rows:,})",
                detail=f"Threshold is {threshold:.0%}. Consider imputation or dropping rows.",
                value=round(pct, 4),
            ))
        elif n_missing > 0:
            report.add(ValidationItem(
                id=f"missing_{col}",
                category="missing",
                severity="info",
                column=col,
                message=f"'{col}' has {n_missing:,} missing values ({pct:.1%})",
                value=round(pct, 4),
            ))


def _check_crs(
    df: pd.DataFrame,
    expected_crs: str | None,
    report: ValidationReport,
) -> None:
    """Check CRS consistency."""
    if not hasattr(df, "crs"):
        return  # not a GeoDataFrame

    if df.crs is None:
        report.add(ValidationItem(
            id="crs_none",
            category="crs",
            severity="warning",
            message="GeoDataFrame has no CRS set",
            detail="Coordinate reference system is undefined. Set it before processing.",
        ))
    elif expected_crs:
        import pyproj
        try:
            actual = pyproj.CRS(df.crs)
            expected = pyproj.CRS(expected_crs)
            if not actual.equals(expected):
                report.add(ValidationItem(
                    id="crs_mismatch",
                    category="crs",
                    severity="warning",
                    message=f"CRS mismatch: data is {df.crs}, expected {expected_crs}",
                    detail="Data will be reprojected during processing.",
                ))
            else:
                report.add(ValidationItem(
                    id="crs_ok",
                    category="crs",
                    severity="info",
                    message=f"CRS matches expected: {expected_crs}",
                ))
        except Exception:
            pass


def _check_duplicates(
    df: pd.DataFrame,
    coord_cols: list[str] | None,
    report: ValidationReport,
) -> None:
    """Detect duplicate records by coordinate pairs."""
    if not coord_cols or len(coord_cols) < 2:
        return

    existing = [c for c in coord_cols if c in df.columns]
    if len(existing) < 2:
        return

    n_dupes = int(df.duplicated(subset=existing, keep="first").sum())
    if n_dupes > 0:
        pct = n_dupes / len(df)
        report.add(ValidationItem(
            id="duplicate_coords",
            category="duplicate",
            severity="warning" if pct < 0.05 else "critical",
            message=f"{n_dupes:,} duplicate coordinate pairs ({pct:.1%} of rows)",
            detail="Duplicate locations may bias spatial models. Consider deduplication.",
            value=n_dupes,
        ))
    else:
        report.add(ValidationItem(
            id="duplicate_coords",
            category="duplicate",
            severity="info",
            message="No duplicate coordinate pairs found",
        ))


# Known variable bounds for distribution anomaly checking
_KNOWN_BOUNDS: dict[str, tuple[float, float, str]] = {
    "albedo":           (0.0, 1.0, "Albedo must be between 0 and 1"),
    "ndvi":             (-1.0, 1.0, "NDVI must be between -1 and 1"),
    "ndbi":             (-1.0, 1.0, "NDBI must be between -1 and 1"),
    "pct_canopy":       (0.0, 100.0, "Canopy percentage must be 0-100"),
    "pct_impervious":   (0.0, 100.0, "Impervious percentage must be 0-100"),
    "pct_tree":         (0.0, 100.0, "Tree percentage must be 0-100"),
    "pct_grass":        (0.0, 100.0, "Grass percentage must be 0-100"),
    "pct_water":        (0.0, 100.0, "Water percentage must be 0-100"),
    "pct_bare":         (0.0, 100.0, "Bare percentage must be 0-100"),
}


def _check_distributions(
    df: pd.DataFrame,
    predictor_cols: list[str],
    report: ValidationReport,
) -> None:
    """Flag values that look like data entry errors or sensor artifacts."""
    for col in predictor_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Check known bounds by matching column name patterns
        col_lower = col.lower()
        for pattern, (lo, hi, desc) in _KNOWN_BOUNDS.items():
            if pattern in col_lower:
                n_below = int((series < lo).sum())
                n_above = int((series > hi).sum())
                if n_below > 0 or n_above > 0:
                    total_bad = n_below + n_above
                    report.add(ValidationItem(
                        id=f"bounds_{col}",
                        category="distribution",
                        severity="critical" if total_bad > 10 else "warning",
                        column=col,
                        message=f"'{col}' has {total_bad} values outside expected range [{lo}, {hi}]",
                        detail=f"{desc}. {n_below} below {lo}, {n_above} above {hi}.",
                        value={"n_below": n_below, "n_above": n_above},
                    ))
                break

        # Check for extreme outliers (> 6σ from mean)
        mean = series.mean()
        std = series.std()
        if std > 0:
            z_scores = np.abs((series - mean) / std)
            n_extreme = int((z_scores > 6).sum())
            if n_extreme > 0:
                report.add(ValidationItem(
                    id=f"outlier_{col}",
                    category="distribution",
                    severity="warning",
                    column=col,
                    message=f"'{col}' has {n_extreme} extreme outliers (>6σ from mean)",
                    detail=f"Mean={mean:.4g}, Std={std:.4g}. Review for data entry errors.",
                    value=n_extreme,
                ))

        # Check for zero variance
        if std == 0 or np.isnan(std):
            report.add(ValidationItem(
                id=f"zero_var_{col}",
                category="distribution",
                severity="warning",
                column=col,
                message=f"'{col}' has zero variance (constant value: {mean:.4g})",
                detail="Constant predictors provide no information to models.",
            ))

        # Check for negative values in percentage-like columns
        if "pct" in col_lower or "percent" in col_lower:
            n_neg = int((series < 0).sum())
            if n_neg > 0:
                report.add(ValidationItem(
                    id=f"neg_pct_{col}",
                    category="distribution",
                    severity="critical",
                    column=col,
                    message=f"'{col}' has {n_neg} negative values in percentage column",
                    detail="Percentage columns should not contain negative values.",
                    value=n_neg,
                ))


def _check_numeric_types(
    df: pd.DataFrame,
    cols: list[str],
    report: ValidationReport,
) -> None:
    """Ensure expected numeric columns actually parse as numeric."""
    for col in cols:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try coercing to see how many fail
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_failed = int(coerced.isna().sum() - df[col].isna().sum())
            report.add(ValidationItem(
                id=f"type_{col}",
                category="type",
                severity="critical" if n_failed > 0 else "warning",
                column=col,
                message=f"'{col}' is not numeric (dtype: {df[col].dtype})",
                detail=f"{n_failed} values cannot be converted to numbers." if n_failed > 0
                       else "Column dtype is object but values may be convertible.",
                value=n_failed,
            ))
