"""Processed data versioning for the SPARC pipeline.

When a user reprocesses data with different settings, this module
saves numbered versions (data_v1.parquet, data_v2.parquet, ...) with
a changelog so users can compare results across processing runs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CHANGELOG_FILENAME = "data_changelog.json"


def _load_changelog(data_dir: Path) -> list[dict]:
    """Load existing changelog or return empty list."""
    path = data_dir / CHANGELOG_FILENAME
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_changelog(data_dir: Path, entries: list[dict]) -> None:
    """Write changelog to disk."""
    path = data_dir / CHANGELOG_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, default=str)


def next_version(data_dir: Path) -> int:
    """Return the next version number (1-based)."""
    changelog = _load_changelog(data_dir)
    if not changelog:
        return 1
    return max(e.get("version", 0) for e in changelog) + 1


def save_versioned(
    df: pd.DataFrame,
    data_dir: Path,
    *,
    settings: dict[str, Any] | None = None,
    description: str = "",
) -> dict:
    """Save a processed dataset as a versioned parquet file.

    Parameters
    ----------
    df : DataFrame
        Processed data to save.
    data_dir : Path
        Project data directory.
    settings : dict, optional
        Processing settings (CRS, resolution, variables, etc.) for the changelog.
    description : str
        Human-readable description of what changed.

    Returns
    -------
    dict with version info: version, filename, path, timestamp, n_rows, n_cols.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    version = next_version(data_dir)
    filename = f"data_v{version}.parquet"
    filepath = data_dir / filename

    # Save parquet
    export_df = df.copy()
    if hasattr(export_df, "geometry"):
        export_df = pd.DataFrame(export_df.drop(columns="geometry"))
    export_df.to_parquet(filepath, index=False)

    # Also save CSV for backwards compatibility
    csv_path = data_dir / f"data_v{version}.csv"
    export_df.to_csv(csv_path, index=False)

    # Build changelog entry
    entry = {
        "version": version,
        "filename": filename,
        "csv_filename": csv_path.name,
        "path": str(filepath),
        "csv_path": str(csv_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "description": description,
        "settings": settings or {},
    }

    # Update changelog
    changelog = _load_changelog(data_dir)
    changelog.append(entry)
    _save_changelog(data_dir, changelog)

    return entry


def list_versions(data_dir: Path) -> list[dict]:
    """Return all data versions from the changelog.

    Returns
    -------
    List of version entries, sorted by version number.
    """
    data_dir = Path(data_dir)
    changelog = _load_changelog(data_dir)
    # Filter to versions whose files still exist
    result = []
    for entry in changelog:
        filepath = data_dir / entry.get("filename", "")
        entry["file_exists"] = filepath.exists()
        result.append(entry)
    return sorted(result, key=lambda e: e.get("version", 0))


def get_version_path(data_dir: Path, version: int) -> Path | None:
    """Return the CSV path for a specific version, or None if not found."""
    data_dir = Path(data_dir)
    changelog = _load_changelog(data_dir)
    for entry in changelog:
        if entry.get("version") == version:
            csv_path = entry.get("csv_path")
            if csv_path and Path(csv_path).exists():
                return Path(csv_path)
            # Fallback to parquet
            parquet_path = data_dir / entry.get("filename", "")
            if parquet_path.exists():
                return parquet_path
    return None
