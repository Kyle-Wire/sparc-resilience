"""sparc.server.routes.data — /data/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.
Routes use ``sparc.server.deps.state`` for shared server state.

Currently migrated:
  - /data/summary
  - /data/preview
  - /data/histogram
  - /data/geojson
  - /data/files
  - /data/select

NOTE: POST /data/upload is intentionally NOT registered here.
The full upload handler in app.py (which saves the file, loads it into
state, and handles rasters/shapefiles) must remain the canonical handler.
A previous incomplete migration of this route caused it to shadow the full
handler and silently discard uploads. See tests/test_upload_encoding.py.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sparc.server.deps import session

router = APIRouter(tags=["data"])


@router.get("/data/summary")
async def data_summary():
    """Return a full statistical summary of the loaded dataset."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded. Load a project first.")

    import pandas as pd  # noqa: F401 (may already be imported transitively)

    df = session.data
    numeric = df.select_dtypes(include="number")

    summary: dict = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "numeric_summary": {
            c: {
                "mean": float(numeric[c].mean()),
                "median": float(numeric[c].median()),
                "std": float(numeric[c].std()),
                "min": float(numeric[c].min()),
                "max": float(numeric[c].max()),
                "count": int(numeric[c].count()),
            }
            for c in numeric.columns
        },
    }

    if hasattr(df, "crs") and df.crs is not None:
        summary["crs"] = str(df.crs)

    if hasattr(df, "total_bounds"):
        bounds = df.total_bounds
        summary["bbox"] = {
            "minx": float(bounds[0]),
            "miny": float(bounds[1]),
            "maxx": float(bounds[2]),
            "maxy": float(bounds[3]),
        }

    return summary


@router.get("/data/preview")
async def data_preview(
    n: int = Query(20, ge=1, le=1000, description="Number of rows to return"),
):
    """Return the first N rows of the loaded dataset."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded.")
    return {"rows": session.data.head(n).to_dict(orient="records")}


@router.get("/data/histogram")
async def data_histogram(
    variable: str = Query(..., description="Column name"),
    bins: int = Query(40, ge=4, le=200, description="Number of histogram bins"),
):
    """Return histogram bin counts for a single column (numpy-backed, single pass)."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded.")
    if variable not in session.data.columns:
        raise HTTPException(404, f"Column '{variable}' not found.")

    import numpy as np
    import pandas as pd

    col = session.data[variable]
    if not pd.api.types.is_numeric_dtype(col):
        raise HTTPException(400, f"Column '{variable}' is not numeric.")

    arr = col.to_numpy(dtype="float64", copy=False)
    finite = arr[np.isfinite(arr)]
    n_total = int(arr.size)
    n_finite = int(finite.size)
    n_missing = n_total - n_finite

    if n_finite == 0:
        return {
            "variable": variable,
            "bins": [],
            "edges": [],
            "n_total": n_total,
            "n_finite": 0,
            "n_missing": n_missing,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    counts, edges = np.histogram(finite, bins=bins)
    return {
        "variable": variable,
        "bins": counts.tolist(),
        "edges": edges.tolist(),
        "n_total": n_total,
        "n_finite": n_finite,
        "n_missing": n_missing,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if n_finite > 1 else 0.0,
    }


_LAT_NAMES = {"lat", "latitude", "y", "point_y", "lat_dd", "ycoord", "ylat"}
_LON_NAMES = {"lon", "long", "longitude", "x", "point_x", "lon_dd", "xcoord", "xlon"}


def _autodetect_latlon(df) -> tuple:
    """Return (lat_col, lon_col) by matching common column-name patterns."""
    cols_lower = {c.lower(): c for c in df.columns}
    lat = next((cols_lower[k] for k in _LAT_NAMES if k in cols_lower), None)
    lon = next((cols_lower[k] for k in _LON_NAMES if k in cols_lower), None)
    return lat, lon


@router.get("/data/geojson")
async def data_geojson(
    lat_col: Optional[str] = Query(None),
    lon_col: Optional[str] = Query(None),
    max_points: int = Query(5000, ge=1, le=50000),
):
    """Return the dataset as a GeoJSON FeatureCollection."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded.")

    # 1. Project config takes priority
    if (lat_col is None or lon_col is None) and session.project_config is not None:
        coords = session.project_config.get("variables", {}).get("coordinates", [])
        if len(coords) >= 2:
            lat_col = lat_col or coords[0]
            lon_col = lon_col or coords[1]

    # 2. Fall back to common-name auto-detection
    if not lat_col or not lon_col:
        lat_col, lon_col = _autodetect_latlon(session.data)

    if not lat_col or not lon_col:
        raise HTTPException(
            400,
            "Cannot determine coordinate columns. "
            "Pass lat_col and lon_col, or name your columns lat/lon, y/x, etc.",
        )

    if lat_col not in session.data.columns:
        raise HTTPException(400, f"lat_col '{lat_col}' not found in dataset.")
    if lon_col not in session.data.columns:
        raise HTTPException(400, f"lon_col '{lon_col}' not found in dataset.")

    df = session.data.head(max_points)
    features = []
    for _, row in df.iterrows():
        try:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row[lon_col]), float(row[lat_col])],
                },
                "properties": {k: v for k, v in row.items() if k not in (lat_col, lon_col)},
            }
            features.append(feat)
        except (ValueError, TypeError):
            continue  # skip rows with non-numeric coords
    return {"type": "FeatureCollection", "features": features}


@router.get("/data/files")
async def data_files(
    directory: Optional[str] = Query(None, description="Directory to scan; defaults to project output"),
):
    """List CSV/Parquet files available for loading."""
    from pathlib import Path

    if directory:
        scan_dir = Path(directory).resolve()
    elif session.project_config:
        scan_dir = Path(session.project_config.get("data", {}).get("file_path", ".")).parent
    else:
        raise HTTPException(400, "No project loaded and no directory specified.")

    if not scan_dir.exists():
        return {"files": []}

    files = [
        {"name": p.name, "path": str(p), "size_bytes": p.stat().st_size}
        for p in sorted(scan_dir.iterdir())
        if p.suffix in (".csv", ".parquet")
    ]
    return {"files": files}


class _SelectBody(BaseModel):
    path: str


@router.post("/data/select")
async def data_select(body: _SelectBody):
    """Set the active data file by path (passed as a JSON request body)."""
    path = body.path

    from pathlib import Path

    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(404, f"File not found: {p}")

    try:
        import pandas as pd

        if p.suffix == ".parquet":
            session.data = pd.read_parquet(str(p))
        else:
            session.data = pd.read_csv(str(p))
    except Exception as exc:
        raise HTTPException(422, f"Could not load file: {exc}")

    return {"status": "loaded", "row_count": len(session.data), "columns": list(session.data.columns)}
