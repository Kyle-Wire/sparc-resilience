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

from sparc.server.deps import session

router = APIRouter(tags=["data"])


@router.get("/data/summary")
async def data_summary():
    """Return a statistical summary of the loaded dataset."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded. Load a project first.")
    return {
        "row_count": len(session.data),
        "columns": list(session.data.columns),
    }


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
    column: str = Query(..., description="Column name"),
    bins: int = Query(20, ge=2, le=200, description="Number of histogram bins"),
):
    """Return histogram bin counts for a single column."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded.")
    if column not in session.data.columns:
        raise HTTPException(404, f"Column '{column}' not found.")

    col = session.data[column].dropna()
    if len(col) == 0:
        return {"column": column, "bins": [], "counts": []}

    col_min = float(col.min())
    col_max = float(col.max())
    if col_min == col_max:
        return {"column": column, "bins": [col_min], "counts": [int(len(col))]}

    width = (col_max - col_min) / bins
    edges = [col_min + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in col:
        idx = min(int((float(v) - col_min) / width), bins - 1)
        counts[idx] += 1

    return {"column": column, "bins": edges, "counts": counts}


@router.get("/data/geojson")
async def data_geojson(
    lat_col: Optional[str] = Query(None),
    lon_col: Optional[str] = Query(None),
    max_points: int = Query(5000, ge=1, le=50000),
):
    """Return the dataset as a GeoJSON FeatureCollection."""
    if session.data is None:
        raise HTTPException(400, "No dataset loaded.")
    if session.project_config is None:
        raise HTTPException(400, "No project loaded.")

    coords = session.project_config.get("variables", {}).get("coordinates", [])
    if lat_col is None and len(coords) >= 2:
        lat_col, lon_col = coords[0], coords[1]
    if not lat_col or not lon_col:
        raise HTTPException(400, "lat_col and lon_col are required.")

    df = session.data.head(max_points)
    features = []
    for _, row in df.iterrows():
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row[lon_col]), float(row[lat_col])]},
            "properties": {k: v for k, v in row.items() if k not in (lat_col, lon_col)},
        }
        features.append(feat)
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


@router.post("/data/select")
async def data_select(path: str = Query(..., description="Absolute path to CSV/Parquet file")):
    """Set the active data file by path (passed as a query parameter)."""

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
