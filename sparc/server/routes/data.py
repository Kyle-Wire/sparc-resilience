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

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from sparc.server.deps import session, state

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


# ---------------------------------------------------------------------------
# B15 — remaining data routes migrated from app.py
# ---------------------------------------------------------------------------

@router.get("/crs/distortion")
async def crs_distortion(
    input_epsg: str = Query("4326"),
    projected_epsg: str = Query(""),
):
    if not projected_epsg:
        raise HTTPException(400, "projected_epsg is required")

    try:
        from pyproj import Transformer, CRS
        import numpy as np

        cx, cy = 0.0, 45.0
        if state.data is not None and hasattr(state.data, "geometry"):
            try:
                import geopandas as gpd
                gdf = state.data
                if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                    gdf = gdf.to_crs(epsg=4326)
                bounds = gdf.total_bounds
                cx = float((bounds[0] + bounds[2]) / 2)
                cy = float((bounds[1] + bounds[3]) / 2)
            except Exception:
                pass
        elif state.data_summary and "bbox" in (state.data_summary or {}):
            bb = state.data_summary["bbox"]
            if isinstance(bb, dict):
                cx = (bb["minx"] + bb["maxx"]) / 2
                cy = (bb["miny"] + bb["maxy"]) / 2

        norm_in = input_epsg.replace("EPSG:", "").strip()
        norm_proj = projected_epsg.replace("EPSG:", "").strip()
        if norm_in == norm_proj:
            src_crs = CRS.from_epsg(int(norm_in))
            return {
                "center_lon": cx,
                "center_lat": cy,
                "k_x": 1.0,
                "k_y": 1.0,
                "k_mean": 1.0,
                "area_distortion_pct": 0.0,
                "input_crs_name": src_crs.name,
                "projected_crs_name": src_crs.name,
                "assessment": "acceptable",
            }

        delta_deg = 0.001
        try:
            t = Transformer.from_crs(
                f"EPSG:{input_epsg.replace('EPSG:', '')}",
                f"EPSG:{projected_epsg.replace('EPSG:', '')}",
                always_xy=True,
            )
            x0, y0 = t.transform(cx, cy)
            x1, y1 = t.transform(cx + delta_deg, cy)
            x2, y2 = t.transform(cx, cy + delta_deg)

            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            _, _, dist_x = geod.inv(cx, cy, cx + delta_deg, cy)
            _, _, dist_y = geod.inv(cx, cy, cx, cy + delta_deg)

            tgt_crs_obj = CRS.from_epsg(int(norm_proj))
            tgt_is_angular = tgt_crs_obj.axis_info[0].unit_name in ("degree", "grad")
            if tgt_is_angular:
                k_x = 1.0
                k_y = 1.0
            else:
                k_x = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dist_x) if dist_x else 1.0
                k_y = float(np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2) / dist_y) if dist_y else 1.0
            k_mean = (k_x + k_y) / 2
            area_distortion_pct = float(abs(k_mean ** 2 - 1) * 100)

            src_crs = CRS.from_epsg(int(input_epsg.replace("EPSG:", "")))
            tgt_crs = CRS.from_epsg(int(projected_epsg.replace("EPSG:", "")))

            return {
                "center_lon": cx,
                "center_lat": cy,
                "k_x": k_x,
                "k_y": k_y,
                "k_mean": k_mean,
                "area_distortion_pct": area_distortion_pct,
                "input_crs_name": src_crs.name,
                "projected_crs_name": tgt_crs.name,
                "assessment": "acceptable" if area_distortion_pct < 0.5 else "high",
            }
        except Exception as e:
            raise HTTPException(400, f"Projection error: {e}")

    except ImportError:
        raise HTTPException(500, "pyproj not available")


@router.post("/data/validate")
async def validate_data():
    if state.data is None:
        raise HTTPException(400, "No data loaded. Load a project first.")
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    from sparc.data.validation import validate_dataset

    config = state.project_config
    report = validate_dataset(
        state.data,
        target_col=config.get("variables", {}).get("target"),
        predictor_cols=config.get("predictors", {}).get("base_model", []),
        coord_cols=config.get("variables", {}).get("coordinates", []),
        expected_crs=config.get("crs", {}).get("initial"),
    )
    return report.to_dict()


@router.get("/data/versions")
async def list_data_versions():
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    from pathlib import Path
    from sparc.data.versioning import list_versions

    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    versions = list_versions(data_dir)
    return {"versions": versions}


@router.post("/data/select_version")
async def select_data_version(version: int = Query(..., description="Version number to activate")):
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    from pathlib import Path
    from sparc.data.versioning import get_version_path
    from sparc.server.app import _set_data_path, _load_data_into_state

    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    path = get_version_path(data_dir, version)

    if path is None:
        raise HTTPException(404, f"Version {version} not found or file missing.")

    _set_data_path(str(path))
    _load_data_into_state(state.project_config)

    return {
        "status": "selected",
        "version": version,
        "path": str(path),
        "columns": list(state.data.columns) if state.data is not None else [],
        "row_count": len(state.data) if state.data is not None else 0,
    }


@router.post("/data/preprocess")
async def preprocess_data():
    import asyncio as _asyncio
    import json as _json
    from fastapi.responses import StreamingResponse
    from pathlib import Path as _Path

    if state.data is None:
        raise HTTPException(400, "No data loaded. Load a project first.")
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    config = state.project_config
    project_dir = _Path(config["paths"]["project_root"])

    async def _generate():
        from sparc.data.preprocessing import run_pipeline

        result_ref: dict = {}
        pipeline = run_pipeline(state.data, config, _result_ref=result_ref)

        # Emit change-detection notice before the first (Ingest CSV) step
        first_step = next(pipeline)
        try:
            from sparc.data.versioning import get_last_hash as _get_last_hash
            _prior_sha = _get_last_hash(project_dir, "ingest_csv")
        except Exception:
            _prior_sha = None
        if _prior_sha is not None and _prior_sha != first_step["sha"]:
            yield "data: " + _json.dumps({
                "step": "Ingest CSV", "changed": True,
                "message": "Raw data modified since last run",
                "sha": first_step["sha"],
            }) + "\n\n"
            await _asyncio.sleep(0)
        yield "data: " + _json.dumps(first_step) + "\n\n"
        await _asyncio.sleep(0)

        for step_dict in pipeline:
            yield "data: " + _json.dumps(step_dict) + "\n\n"
            await _asyncio.sleep(0)

        final_df = result_ref.get("df")
        if final_df is not None:
            state.data = final_df
            state.data_summary = None

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/data/fishnet")
async def create_fishnet_endpoint(body: dict = Body(...)):
    from pathlib import Path as _Path
    from sparc.data.processing import create_fishnet, clip_to_boundary

    bounds = body.get("bounds")
    resolution = body.get("resolution", 100)
    crs = body.get("crs", "EPSG:4326")

    if not bounds or len(bounds) != 4:
        raise HTTPException(400, "bounds must be [minx, miny, maxx, maxy]")

    gdf = create_fishnet(tuple(bounds), resolution, crs)

    boundary_path = body.get("boundary_path")
    if boundary_path and _Path(boundary_path).exists():
        import geopandas as _gpd
        boundary = _gpd.read_file(boundary_path)
        if boundary.crs and boundary.crs.to_string() != crs:
            boundary = boundary.to_crs(crs)
        gdf = clip_to_boundary(gdf, boundary)

    if state.project_config:
        out_dir = _Path(state.project_config.get("paths", {}).get("project_dir", "."))
        out_path = out_dir / "fishnet.gpkg"
        gdf.to_file(out_path, driver="GPKG")

    return {"n_cells": len(gdf), "columns": list(gdf.columns)}


@router.post("/data/zonal_stats")
async def zonal_stats_endpoint(body: dict = Body(...)):
    from pathlib import Path as _Path
    from sparc.data.processing import run_zonal_stats
    import geopandas as _gpd

    fishnet_path = body.get("fishnet_path")
    raster_paths = body.get("raster_paths", [])
    stats = body.get("stats", "mean")

    if not fishnet_path or not _Path(fishnet_path).exists():
        raise HTTPException(400, "fishnet_path not found")
    if not raster_paths:
        raise HTTPException(400, "raster_paths required")

    gdf = _gpd.read_file(fishnet_path)
    gdf = run_zonal_stats(gdf, raster_paths, stats=stats)

    out_path = _Path(fishnet_path).with_name("fishnet_with_stats.gpkg")
    gdf.to_file(out_path, driver="GPKG")

    csv_path = out_path.with_suffix(".csv")
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    return {"n_cells": len(gdf), "columns": list(gdf.columns), "csv_path": str(csv_path)}


@router.post("/data/prepare")
async def prepare_data_pipeline(body: dict = Body(...)):
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    from pathlib import Path as _Path
    from sparc.data.processing import create_fishnet, clip_to_boundary, run_zonal_stats
    import geopandas as _gpd

    boundary_path = body.get("boundary_path")
    raster_paths = body.get("raster_paths", [])
    resolution = body.get("resolution", 100)
    crs = body.get("crs", "EPSG:4326")
    stats = body.get("stats", "mean")
    set_as_data = body.get("set_as_data", True)

    if not raster_paths:
        raise HTTPException(400, "At least one raster_path is required")

    boundary_gdf = None
    if boundary_path and _Path(boundary_path).exists():
        boundary_gdf = _gpd.read_file(boundary_path)
        if boundary_gdf.crs and boundary_gdf.crs.to_string() != crs:
            boundary_gdf = boundary_gdf.to_crs(crs)
        bounds = tuple(boundary_gdf.total_bounds)
    else:
        try:
            import rasterio
            with rasterio.open(raster_paths[0]) as src:
                bounds = tuple(src.bounds)
                if not crs:
                    crs = str(src.crs)
        except Exception as exc:
            raise HTTPException(400, f"Cannot determine bounds: {exc}")

    gdf = create_fishnet(bounds, resolution, crs)

    if boundary_gdf is not None:
        gdf = clip_to_boundary(gdf, boundary_gdf)

    if len(gdf) == 0:
        raise HTTPException(400, "Fishnet has 0 cells after clipping — check CRS/resolution")

    valid_rasters = [r for r in raster_paths if _Path(r).exists()]
    if not valid_rasters:
        raise HTTPException(400, "None of the raster paths exist")
    gdf = run_zonal_stats(gdf, valid_rasters, stats=stats)

    project_dir = _Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = data_dir / "fishnet_with_stats.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    export_df = gdf.copy()
    export_df["centroid_x"] = gdf.geometry.centroid.x
    export_df["centroid_y"] = gdf.geometry.centroid.y
    export_flat = export_df.drop(columns=["geometry"])

    from sparc.data.versioning import save_versioned
    version_info = save_versioned(
        export_flat,
        data_dir,
        settings={"resolution": resolution, "crs": crs, "stats": stats, "n_rasters": len(valid_rasters), "boundary": boundary_path},
        description=f"Fishnet processing: {len(valid_rasters)} rasters, resolution={resolution}, CRS={crs}",
    )
    csv_path = version_info["csv_path"]

    canonical_csv = data_dir / "fishnet_with_stats.csv"
    export_flat.to_csv(canonical_csv, index=False)

    if set_as_data:
        from sparc.server.app import _set_data_path, _load_data_into_state
        _set_data_path(str(csv_path))
        _load_data_into_state(state.project_config)

    return {
        "status": "prepared",
        "n_cells": len(gdf),
        "columns": list(gdf.columns),
        "csv_path": str(csv_path),
        "gpkg_path": str(gpkg_path),
        "set_as_data": set_as_data,
        "version": version_info.get("version"),
    }

