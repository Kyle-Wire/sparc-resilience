"""
SPARC FastAPI Server
====================

Thin HTTP / WebSocket layer over the existing ``sparc/`` pipeline.
Launched by the Tauri desktop app as a sidecar on ``localhost:8008``,
or manually via ``sparc server --port 8008``.

The server holds loaded project data and model objects in memory so that
the React frontend can query results without disk round-trips, and
pipeline stages stream structured JSON events over a WebSocket instead
of dumping to stdout.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from sparc.server.state import ServerState
from sparc.server.stream import stream_stage

# ------------------------------------------------------------------
# Application & shared state
# ------------------------------------------------------------------

app = FastAPI(
    title="SPARC Server",
    version="1.0.0",
    docs_url="/docs",
)

# Allow the Tauri webview (and dev server) to reach us
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "http://localhost:1420",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = ServerState()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


# ------------------------------------------------------------------
# Physics helpers
# ------------------------------------------------------------------

def _resolve_expected_sign(cfg: dict, variable: str) -> str | None:
    """Return ``'negative'``, ``'positive'``, or ``None`` for *variable*.

    Checks three sources in order:
      1. ``config["physics"]["monotone_constraints"]``  (inline in project.yml)
      2. ``config["physics"]["caps_file"]`` → ``monotonicity.{var}.expected_sign``
      3. Built-in PhysicsPriors defaults (Canopy → negative, etc.)
    """
    physics = cfg.get("physics", {})
    if not isinstance(physics, dict):
        return None

    # 1. monotone_constraints  e.g. {"Pct_Canopy": -1, "NDVI": -1}
    mc = physics.get("monotone_constraints", {})
    if isinstance(mc, dict) and variable in mc:
        val = mc[variable]
        if val == -1 or val == "-1":
            return "negative"
        if val == 1 or val == "1":
            return "positive"
        return None  # 0 = unconstrained

    # 2. caps_file → monotonicity section
    caps_path = physics.get("caps_file")
    if caps_path and os.path.exists(caps_path):
        try:
            import yaml
            with open(caps_path, "r", encoding="utf-8") as fh:
                caps = yaml.safe_load(fh) or {}
            mono = caps.get("monotonicity", {})
            if isinstance(mono, dict) and variable in mono:
                return mono[variable].get("expected_sign") if isinstance(mono[variable], dict) else None
        except Exception:
            pass

    # 3. Built-in PhysicsPriors class (literature defaults)
    try:
        from sparc.interventions.physics_priors import PhysicsPriors
        pp = PhysicsPriors()
        coef = pp.coefficients.get(variable)
        if coef is not None:
            return "negative" if coef.coefficient < 0 else "positive" if coef.coefficient > 0 else None
    except Exception:
        pass

    return None


# ------------------------------------------------------------------
# Scenario GPKG cache (avoids re-reading multi-MB file per slider tick)
# ------------------------------------------------------------------

_scenario_gpkg_cache: dict[str, Any] = {}  # {path_str: GeoDataFrame}


def _load_scenario_gpkg(paths) -> Any:
    """Return the scenario GeoDataFrame (WGS84), cached after first load."""
    for gpkg_name in ("scenario_results.gpkg", "scenario_results_dag.gpkg",
                      "scenario_results_hybrid.gpkg", "scenario_results_reprediction.gpkg"):
        candidate = paths.stage4_dir / gpkg_name
        if candidate.exists():
            key = str(candidate)
            if key in _scenario_gpkg_cache:
                return _scenario_gpkg_cache[key]
            import geopandas as gpd
            gdf = gpd.read_file(candidate)
            _scenario_gpkg_cache[key] = gdf
            return gdf
    return None


# ------------------------------------------------------------------
# Startup: auto-load project if SPARC_SERVER_PROJECT env var is set
# ------------------------------------------------------------------

@app.on_event("startup")
async def _auto_load_project():
    project_env = os.environ.get("SPARC_SERVER_PROJECT")
    if not project_env:
        return
    resolved = Path(project_env).resolve()
    if not resolved.exists():
        print(f"Warning: SPARC_SERVER_PROJECT file not found: {resolved}")
        return
    try:
        from sparc.config.config import load_config
        import yaml
        with open(resolved, 'r', encoding='utf-8') as fh:
            raw_yaml = yaml.safe_load(fh) or {}
        config = load_config(str(resolved))
        state.project_path = str(resolved)
        state.project_config = config
        state.raw_project_yaml = raw_yaml
        _load_data_into_state(config)
        print(f"Auto-loaded project: {resolved}")
    except Exception as exc:
        print(f"Warning: auto-load failed for {resolved}: {exc}")


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "project_loaded": state.project_config is not None,
        "project_path": state.project_path,
        "is_running": state.is_running,
        "current_stage": state.current_stage,
    }


@app.get("/debug/paths")
async def debug_paths():
    """Diagnostic endpoint: show resolved output paths and whether expected files exist."""
    if state.project_config is None:
        return {"error": "No project loaded"}

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        return {"error": f"Cannot resolve paths: {exc}"}

    expected_files = {
        "stage1_gwen_csv": str(paths.stage1_dir / "gwen_variable_importance.csv"),
        "stage1_gwen_json": str(paths.stage1_dir / "gwen_results.json"),
        "stage2_predictions_gpkg": str(paths.stage2_dir / "spatial_cv_predictions.gpkg"),
        "stage3_coefficients": str(paths.stage3_dir / "scenario_coefficients.json"),
        "stage3_dose_response": str(paths.stage3_dir / "dose_response_curves.json"),
        "stage4_scenario_gpkg": str(paths.stage4_dir / "scenario_results.gpkg"),
    }

    return {
        "output_dir": str(paths.output_dir),
        "stage1_dir": str(paths.stage1_dir),
        "stage2_dir": str(paths.stage2_dir),
        "stage3_dir": str(paths.stage3_dir),
        "stage4_dir": str(paths.stage4_dir),
        "final_dir": str(paths.final_dir),
        "files": {k: {"path": v, "exists": Path(v).exists()} for k, v in expected_files.items()},
        "config_output_base_dir": state.project_config.get("output", {}).get("base_dir"),
        "config_paths_output_dir": state.project_config.get("paths", {}).get("output_dir"),
    }


# ------------------------------------------------------------------
# Project endpoints
# ------------------------------------------------------------------

@app.post("/project/load")
async def load_project(path: str = Query(..., description="Absolute path to project.yml")):
    """Load a project.yml and return its metadata."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(404, f"Project file not found: {resolved}")

    from sparc.config.config import load_config

    # Also keep the raw YAML for the config editor
    import yaml
    with open(resolved, 'r', encoding='utf-8') as fh:
        raw_yaml = yaml.safe_load(fh) or {}

    try:
        config = load_config(str(resolved))
    except Exception as exc:
        raise HTTPException(422, f"Invalid project configuration: {exc}")

    state.project_path = str(resolved)
    state.project_config = config
    state.raw_project_yaml = raw_yaml

    # Pre-load data summary if the CSV exists
    data_path = config["data"]["file_path"]
    if os.path.exists(data_path):
        _load_data_into_state(config)

    return {
        "status": "loaded",
        "project": raw_yaml.get("project", {}),
        "columns": list(state.data.columns) if state.data is not None else [],
        "row_count": len(state.data) if state.data is not None else 0,
    }


@app.post("/project/validate")
async def validate_project(path: str = Query(..., description="Absolute path to project.yml")):
    """Validate a project.yml without loading it into state."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(404, f"Project file not found: {resolved}")

    from sparc.config.config import load_config

    warnings: list[str] = []
    try:
        config = load_config(str(resolved))
    except Exception as exc:
        return {"valid": False, "error": str(exc), "warnings": []}

    # Check data file
    data_path = config["data"]["file_path"]
    if not os.path.exists(data_path):
        warnings.append(f"Data file not found: {data_path}")
    else:
        import pandas as pd
        df = pd.read_csv(data_path, nrows=5)
        expected = (
            [config["variables"]["target"]]
            + list(config["variables"]["coordinates"])
            + list(config["predictors"]["base_model"])
        )
        missing = [c for c in expected if c not in df.columns]
        if missing:
            warnings.append(f"Missing columns: {missing}")

    # Check physics files
    for key in ("priors_file", "caps_file"):
        fpath = config.get("physics", {}).get(key)
        if fpath and not os.path.exists(fpath):
            warnings.append(f"{key} not found: {fpath}")

    # Check DAG file
    dag_file = config.get("causal", {}).get("dag_file")
    if dag_file and not os.path.exists(dag_file):
        warnings.append(f"dag_file not found: {dag_file}")

    return {"valid": True, "warnings": warnings}


@app.post("/project/init")
async def init_project(
    template: str = Query("blank", description="Template name"),
    output: str = Query(..., description="Output directory path"),
):
    """Scaffold a new project from a domain template."""
    source = TEMPLATES_DIR / template
    if not source.exists():
        available = [d.name for d in TEMPLATES_DIR.iterdir() if d.is_dir()]
        raise HTTPException(404, f"Template '{template}' not found. Available: {available}")

    dest = Path(output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)

    return {
        "status": "created",
        "template": template,
        "path": str(dest),
        "project_yml": str(dest / "project.yml"),
    }


@app.get("/project/templates")
async def list_templates():
    """List available domain templates."""
    templates = []
    for d in sorted(TEMPLATES_DIR.iterdir()):
        if d.is_dir():
            yml = d / "project.yml"
            templates.append({
                "name": d.name,
                "has_project_yml": yml.exists(),
            })
    return {"templates": templates}


@app.get("/project/config")
async def get_project_config():
    """Return the current project configuration as JSON (project.yml structure)."""
    if state.raw_project_yaml is None:
        raise HTTPException(400, "No project loaded.")
    return state.raw_project_yaml


@app.put("/project/config")
async def update_project_config(body: dict[str, Any]):
    """Update the project configuration and persist to disk."""
    if state.raw_project_yaml is None:
        raise HTTPException(400, "No project loaded.")
    state.raw_project_yaml.update(body)
    # Persist to the project YAML
    if state.project_path:
        import yaml
        yml_path = Path(state.project_path)
        if yml_path.is_dir():
            yml_path = yml_path / "project.yml"
        with open(yml_path, "w", encoding="utf-8") as fh:
            yaml.dump(state.raw_project_yaml, fh, default_flow_style=False, sort_keys=False)
        # Reload internal config
        from sparc.config.config import load_config
        state.project_config = load_config(str(yml_path))
    return {"status": "updated"}


# ------------------------------------------------------------------
# Data endpoints
# ------------------------------------------------------------------

@app.get("/data/summary")
async def data_summary():
    """Column statistics for LLM context injection and UI display."""
    if state.data is None:
        raise HTTPException(400, "No data loaded. Load a project first.")
    return state.data_summary or _compute_summary(state)


@app.get("/data/preview")
async def data_preview(n: int = Query(50, ge=1, le=500)):
    """Return the first N rows as JSON records."""
    if state.data is None:
        raise HTTPException(400, "No data loaded.")
    import pandas as pd

    df = state.data.head(n)
    # Drop geometry column for JSON serialization
    if hasattr(df, "geometry"):
        df = pd.DataFrame(df.drop(columns="geometry"))
    return {"rows": df.to_dict(orient="records"), "total": len(state.data)}


@app.get("/data/geojson")
async def data_geojson(variable: str | None = Query(None)):
    """Return raw data as GeoJSON, optionally filtered to a single variable for map coloring."""
    if state.data is None:
        raise HTTPException(400, "No data loaded.")

    df = state.data
    if not hasattr(df, "geometry") or df.geometry is None:
        raise HTTPException(400, "Loaded data has no geometry column.")

    if variable:
        if variable not in df.columns:
            raise HTTPException(400, f"Column '{variable}' not found.")
        subset = df[["geometry", variable]].copy()
    else:
        # Return just geometry + first 5 numeric cols to keep payload small
        numeric_cols = list(df.select_dtypes(include="number").columns[:5])
        subset = df[["geometry"] + numeric_cols].copy()

    # Reproject to WGS84 for web display (deck.gl / maplibre expect EPSG:4326)
    if hasattr(subset, "crs") and subset.crs is not None and str(subset.crs) != "EPSG:4326":
        subset = subset.to_crs(epsg=4326)

    return subset.__geo_interface__


@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Accept a CSV, raster (.tif/.tiff), or spatial file (.shp/.gpkg/.geojson) upload."""
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name  # strip any directory components
    dest = data_dir / safe_name
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    suffix = dest.suffix.lower()
    result: dict = {"status": "uploaded", "path": str(dest), "file_type": suffix}

    # CSV / Parquet — load as primary data
    if suffix in (".csv", ".parquet"):
        _set_data_path(str(dest))
        _load_data_into_state(state.project_config)
        result.update({
            "columns": list(state.data.columns) if state.data is not None else [],
            "row_count": len(state.data) if state.data is not None else 0,
        })

    # Raster — store metadata for zonal-stats pipeline
    elif suffix in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(dest) as src:
                result.update({
                    "crs": str(src.crs) if src.crs else None,
                    "bounds": list(src.bounds),
                    "shape": list(src.shape),
                    "band_count": src.count,
                })
        except ImportError:
            result["crs"] = None
            result["note"] = "rasterio not installed; metadata unavailable"
        except Exception as exc:
            result["note"] = f"Could not read raster metadata: {exc}"

    # Shapefile / GeoPackage / GeoJSON — boundary or vector layer
    elif suffix in (".shp", ".gpkg", ".geojson"):
        try:
            import geopandas as _gpd
            gdf = _gpd.read_file(dest)
            result.update({
                "crs": str(gdf.crs) if gdf.crs else None,
                "n_features": len(gdf),
                "columns": [c for c in gdf.columns if c != "geometry"],
                "bounds": list(gdf.total_bounds),
            })
        except Exception as exc:
            result["note"] = f"Could not read spatial file: {exc}"

    # Shapefile sidecar files (.shx, .dbf, .prj, .cpg) — just store
    elif suffix in (".shx", ".dbf", ".prj", ".cpg"):
        result["note"] = "Shapefile sidecar stored"

    else:
        result["note"] = f"Unknown file type {suffix}; stored as-is"

    return result


@app.get("/data/files")
async def list_data_files():
    """List CSV/Parquet files inside the project's working directory."""
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    project_dir = Path(state.project_config["paths"]["project_root"])
    files: list[dict] = []
    for ext in ("*.csv", "*.parquet", "*.CSV", "*.tif", "*.tiff", "*.shp", "*.gpkg", "*.geojson"):
        for p in project_dir.rglob(ext):
            try:
                rel = p.relative_to(project_dir)
            except ValueError:
                rel = p
            files.append({
                "name": p.name,
                "path": str(p),
                "relative": str(rel),
                "size": p.stat().st_size,
            })
    files.sort(key=lambda f: f["relative"])
    return {"project_dir": str(project_dir), "files": files}


@app.post("/data/select")
async def select_data_file(path: str = Query(..., description="Absolute path to a data file")):
    """Select an existing data file (must already be on disk)."""
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(404, f"File not found: {resolved}")
    if resolved.suffix.lower() not in (".csv", ".parquet"):
        raise HTTPException(400, "Only .csv and .parquet files are supported.")

    # Update config to point to this file and persist to disk
    _set_data_path(str(resolved))
    _load_data_into_state(state.project_config)

    return {
        "status": "selected",
        "path": str(resolved),
        "columns": list(state.data.columns) if state.data is not None else [],
        "row_count": len(state.data) if state.data is not None else 0,
    }


# ------------------------------------------------------------------
# Data validation endpoint
# ------------------------------------------------------------------

@app.post("/data/validate")
async def validate_data():
    """Run pre-flight validation checks on the currently loaded dataset.

    Returns a structured checklist of issues (missing values, CRS,
    duplicates, distribution anomalies, type errors) with severity
    levels: critical / warning / info.
    """
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


# ------------------------------------------------------------------
# Data versioning endpoints
# ------------------------------------------------------------------

@app.get("/data/versions")
async def list_data_versions():
    """Return all versioned data snapshots for the current project."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    from sparc.data.versioning import list_versions

    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    versions = list_versions(data_dir)
    return {"versions": versions}


@app.post("/data/select_version")
async def select_data_version(version: int = Query(..., description="Version number to activate")):
    """Switch the active dataset to a specific versioned snapshot."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    from sparc.data.versioning import get_version_path

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


# ------------------------------------------------------------------
# Session log endpoint
# ------------------------------------------------------------------

@app.get("/run/log")
async def get_run_log():
    """Return the persistent session log for the current project.

    Each entry is a JSON object with timestamp, stage, type, message, and data.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    project_dir = Path(state.project_config["paths"]["project_root"])
    log_path = project_dir / "session_log.jsonl"

    if not log_path.exists():
        return {"entries": [], "path": str(log_path)}

    import json
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"entries": entries, "path": str(log_path)}


# ------------------------------------------------------------------
# Pipeline streaming (WebSocket)
# ------------------------------------------------------------------

@app.websocket("/run/stream")
async def run_stream(ws: WebSocket):
    """Stream structured pipeline events over a WebSocket.

    The client sends a JSON message to start:
        ``{"stage": 2, "fast": false, "skip_gwen": false}``

    The server pushes events until the stage completes or errors:
        ``{"type": "metric", "stage": 2, "fold": 3, "metric": "r2", "value": 0.891}``
        ``{"type": "complete", "stage": 2}``
    """
    await ws.accept()

    try:
        init_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    stage = int(init_msg.get("stage", 0))
    fast = bool(init_msg.get("fast", False))
    skip_gwen = bool(init_msg.get("skip_gwen", False))

    if state.project_config is None:
        await ws.send_json({"type": "error", "message": "No project loaded"})
        await ws.close()
        return

    if state.is_running:
        await ws.send_json({"type": "error", "message": f"Stage {state.current_stage} already running"})
        await ws.close()
        return

    try:
        async for event in stream_stage(state, stage, fast=fast, skip_gwen=skip_gwen):
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        if ws.client_state.name != "DISCONNECTED":
            await ws.close()


@app.get("/run/events")
async def get_run_events():
    """Return buffered events for the current (or most recent) pipeline run.

    Allows a client that reconnects (e.g. navigated away) to catch up on
    events it missed without needing to re-open the WebSocket.
    """
    return {
        "is_running": state.is_running,
        "current_stage": state.current_stage,
        "events": state.get_buffered_events(),
    }


# ------------------------------------------------------------------
# Structured results endpoints (MUST be defined before /results/{stage}
# so FastAPI matches exact paths before the parameterized catch-all)
# ------------------------------------------------------------------

@app.get("/results/correlogram")
async def get_correlogram_data():
    """Return correlogram analysis results with per-variable lag/Moran's I data."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import json as _json

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    candidates = [
        paths.stage0_dir / "correlogram_analysis_results.json",
        paths.stage0_dir / "correlogram_results.json",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise HTTPException(404, "Correlogram results not found. Run Stage 0 first.")

    with open(found, "r", encoding="utf-8") as fh:
        return _json.load(fh)


@app.get("/results/gwen")
async def get_gwen_data():
    """Return GWEN variable importance as a row-oriented table."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import json as _json
    import pandas as pd

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    # Try stage1_dir (GWEN dir) first, then fall back to output_dir for legacy
    for search_dir in [paths.stage1_dir, paths.output_dir]:
        csv_path = search_dir / "gwen_variable_importance.csv"
        json_path = search_dir / "gwen_results.json"

        print(f"[SPARC] GWEN lookup: csv={csv_path} exists={csv_path.exists()}, json={json_path} exists={json_path.exists()}")

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return {"rows": df.to_dict(orient="records")}
        elif json_path.exists():
            with open(json_path, "r", encoding="utf-8") as fh:
                return _json.load(fh)

    # List what IS in the stage directory to help diagnose
    found = list(paths.stage1_dir.glob("*")) if paths.stage1_dir.exists() else []
    found_str = ", ".join(f.name for f in found[:20]) if found else "(directory empty or missing)"
    raise HTTPException(
        404,
        f"GWEN results not found. Looked in: {paths.stage1_dir}. "
        f"Files present: {found_str}",
    )


@app.get("/results/spatial_cv/predictions")
async def get_spatial_cv_predictions():
    """Return spatial CV predictions as GeoJSON (from gpkg with geometry)."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    gpkg_path = paths.stage2_dir / "spatial_cv_predictions.gpkg"
    print(f"[SPARC] Spatial CV lookup: {gpkg_path} exists={gpkg_path.exists()}")

    gdf = None

    if gpkg_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_path)
    else:
        # ---- Fallback: reconstruct from CSV + source geometry ----
        csv_path = paths.stage2_dir / "final_ensemble_predictions.csv"
        oof_path = paths.stage2_dir / "optimized_oof_predictions.csv"
        source_csv = csv_path if csv_path.exists() else (oof_path if oof_path.exists() else None)

        if source_csv is not None and state.data is not None:
            import pandas as pd, geopandas as gpd
            pred_df = pd.read_csv(source_csv)
            src = state.data
            if len(pred_df) == len(src):
                gdf = gpd.GeoDataFrame(pred_df, geometry=src.geometry.values, crs=src.crs)
                print(f"[SPARC] Reconstructed GPKG from {source_csv.name} + source geometry")
                # Persist for next time
                try:
                    gdf.to_file(gpkg_path, driver="GPKG")
                except Exception as exc:
                    print(f"[SPARC] Could not persist reconstructed gpkg: {exc}")

    if gdf is None:
        found = list(paths.stage2_dir.glob("*")) if paths.stage2_dir.exists() else []
        found_str = ", ".join(f.name for f in found[:20]) if found else "(directory empty or missing)"
        raise HTTPException(
            404,
            f"Spatial CV predictions gpkg not found. Looked in: {paths.stage2_dir}. "
            f"Files present: {found_str}",
        )

    # Reproject to WGS84 for web display
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf.__geo_interface__


@app.get("/results/causal")
async def get_causal_results():
    """Return causal validation results (coefficients, effects, diagnostics)."""
    from datetime import datetime as _dt
    import json as _json
    import traceback as _tb

    log_lines: list[str] = [f"=== /results/causal diagnostic log  {_dt.now().isoformat()} ==="]
    stage3_dir = None  # will be set once paths resolve

    def _write_log():
        """Best-effort: dump log_lines into stage3_dir/_causal_endpoint_log.txt"""
        try:
            target = stage3_dir or Path(".")
            target.mkdir(parents=True, exist_ok=True)
            log_path = target / "_causal_endpoint_log.txt"
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(log_lines) + "\n")
            print(f"[SPARC] Diagnostic log written to {log_path}")
        except Exception as exc:
            print(f"[SPARC] Could not write diagnostic log: {exc}")

    # ── 1. Check project config ─────────────────────────────────────
    if state.project_config is None:
        log_lines.append("FAIL: state.project_config is None — no project loaded")
        _write_log()
        raise HTTPException(400, "No project loaded")

    log_lines.append(f"project_config keys: {list(state.project_config.keys())}")
    output_cfg = state.project_config.get("output", {})
    log_lines.append(f"output config: base_dir={output_cfg.get('base_dir')}")
    log_lines.append(f"output config: stage_dirs={output_cfg.get('stage_dirs')}")

    # ── 2. Resolve PipelinePaths ────────────────────────────────────
    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        log_lines.append(f"FAIL: PipelinePaths.from_config raised {type(exc).__name__}: {exc}")
        log_lines.append(_tb.format_exc())
        _write_log()
        raise HTTPException(404, "Cannot resolve output paths")

    stage3_dir = paths.stage3_dir
    log_lines.append(f"output_dir  = {paths.output_dir}  exists={paths.output_dir.exists()}")
    log_lines.append(f"stage3_dir  = {stage3_dir}  exists={stage3_dir.exists()}")

    # ── 3. List everything in stage3_dir ────────────────────────────
    if stage3_dir.exists():
        children = sorted(stage3_dir.iterdir())
        log_lines.append(f"stage3_dir contents ({len(children)} items):")
        for ch in children:
            log_lines.append(f"  {ch.name}  size={ch.stat().st_size if ch.is_file() else 'DIR'}")
    else:
        log_lines.append("stage3_dir DOES NOT EXIST on disk")

    # ── 4. Try primary file ─────────────────────────────────────────
    coeff_path = stage3_dir / "scenario_coefficients.json"
    log_lines.append(f"primary file: {coeff_path}  exists={coeff_path.exists()}")

    if coeff_path.exists():
        try:
            with open(coeff_path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            log_lines.append(f"OK: loaded scenario_coefficients.json  top-keys={list(data.keys())}")
            _write_log()
            return data
        except Exception as exc:
            log_lines.append(f"FAIL: could not parse scenario_coefficients.json: {exc}")

    # ── 5. Fallback: causal_diagnostics.json ────────────────────────
    diag_path = stage3_dir / "causal_diagnostics.json"
    log_lines.append(f"fallback file: {diag_path}  exists={diag_path.exists()}")
    if diag_path.exists():
        try:
            with open(diag_path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            log_lines.append(f"OK: loaded causal_diagnostics.json  top-keys={list(data.keys())}")
            _write_log()
            return data
        except Exception as exc:
            log_lines.append(f"FAIL: could not parse causal_diagnostics.json: {exc}")

    # ── 6. Fallback: in-memory result ───────────────────────────────
    mem_result = state.get_result(3)
    log_lines.append(f"in-memory state.get_result(3): {type(mem_result).__name__}, truthy={bool(mem_result)}")
    if mem_result is not None:
        log_lines.append("OK: returning in-memory result")
        _write_log()
        return mem_result

    # ── 7. Nothing found ────────────────────────────────────────────
    found = list(stage3_dir.glob("*")) if stage3_dir.exists() else []
    found_str = ", ".join(f.name for f in found[:20]) if found else "(directory empty or missing)"
    log_lines.append(f"ALL LOOKUPS FAILED — returning 404. Files present: {found_str}")
    _write_log()
    raise HTTPException(
        404,
        f"Causal results not found. Looked in: {stage3_dir}. "
        f"Files present: {found_str}",
    )


@app.get("/results/causal/dose_response")
async def get_dose_response():
    """Return dose-response curves if available."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import json as _json

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    dr_path = paths.stage3_dir / "dose_response_curves.json"
    if not dr_path.exists():
        raise HTTPException(404, "Dose-response data not available")

    with open(dr_path, "r", encoding="utf-8") as fh:
        return _json.load(fh)


@app.get("/results/causal/diagnostics")
async def get_causal_diagnostics():
    """Return CATE diagnostics (calibration, cumulative effects, RATE)."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import json as _json

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    diag_path = paths.stage3_dir / "causal_diagnostics.json"
    if not diag_path.exists():
        raise HTTPException(404, "Causal diagnostics not available")

    with open(diag_path, "r", encoding="utf-8") as fh:
        return _json.load(fh)


@app.get("/results/pdp_curves")
async def get_pdp_curves():
    """Return GWRF partial dependence / condition curves."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import json as _json

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    # Check all possible locations (canonical first, then legacy)
    candidates = [
        paths.stage2_dir / "gwrf_pdp" / "gwrf_condition_curves.json",
        paths.output_dir / "spatial_intelligence" / "gwrf_pdp" / "gwrf_condition_curves.json",
        paths.spatial_analysis_dir / "gwrf_pdp" / "gwrf_condition_curves.json",
        paths.stage2_dir / "spatial_intelligence" / "gwrf_pdp" / "gwrf_condition_curves.json",
        paths.stage3_dir / "gwrf_condition_curves.json",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise HTTPException(404, "PDP curve data not found. Run Stage 2 first.")

    with open(found, "r", encoding="utf-8") as fh:
        return _json.load(fh)


@app.get("/results/scenarios/detail")
async def get_scenario_detail():
    """Return scenario results as GeoJSON with delta columns + summary table."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import pandas as pd
    import numpy as np

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    # Load scenario spatial results (check canonical name + mode-specific variants)
    gpkg_path = None
    for gpkg_name in (
        "scenario_results.gpkg",
        "scenario_results_dag.gpkg",
        "scenario_results_hybrid.gpkg",
        "scenario_results_reprediction.gpkg",
    ):
        candidate = paths.stage4_dir / gpkg_name
        if candidate.exists():
            gpkg_path = candidate
            break
    print(f"[SPARC] Scenario gpkg lookup: {gpkg_path} found={gpkg_path is not None}")
    geojson_data = None
    if gpkg_path is not None:
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_path)

        # Compute delta columns (scenario prediction - baseline)
        baseline_col = None
        for col in gdf.columns:
            if "baseline" in col.lower() and "pred" in col.lower():
                baseline_col = col
                break
        if baseline_col is None:
            baseline_col = next((c for c in gdf.columns if c == "pred_baseline"), None)

        if baseline_col is not None:
            pred_cols = [c for c in gdf.columns if c.startswith("pred_") and c != baseline_col]
            for pc in pred_cols:
                delta_col = pc.replace("pred_", "delta_")
                try:
                    gdf[delta_col] = gdf[pc].astype(float) - gdf[baseline_col].astype(float)
                except Exception:
                    pass

        # Reproject to WGS84 for web display
        if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        geojson_data = gdf.__geo_interface__
    else:
        # Fall back to in-memory result
        result = state.get_result(4)
        if result is not None:
            spatial = result.get("spatial", result) if isinstance(result, dict) else result
            import geopandas as gpd
            if isinstance(spatial, gpd.GeoDataFrame):
                if spatial.crs is not None and str(spatial.crs) != "EPSG:4326":
                    spatial = spatial.to_crs(epsg=4326)
                geojson_data = spatial.__geo_interface__

        # If still no GeoJSON, reconstruct from CSV data + coordinates
        if geojson_data is None:
            try:
                import geopandas as gpd
                from shapely.geometry import Point

                cfg = state.project_config
                coord_cols = cfg.get("data", {}).get("coord_columns",
                             cfg.get("variables", {}).get("coordinates", []))
                data_file = cfg.get("data", {}).get("file_path",
                            cfg.get("paths", {}).get("raw_csv_path"))
                crs = cfg.get("crs", {}).get("projected",
                      cfg.get("crs", {}).get("input", "EPSG:4326"))

                # Look for per-point MC consensus CSV (has spatial predictions)
                mc_csv = paths.stage4_dir / "scenario_mc_consensus.csv"
                if mc_csv.exists() and data_file and len(coord_cols) == 2:
                    base_df = pd.read_csv(data_file)
                    mc_df = pd.read_csv(mc_csv)
                    x_col, y_col = coord_cols
                    if x_col in base_df.columns and y_col in base_df.columns:
                        # Align lengths (mc_df may match base_df row-for-row)
                        n = min(len(base_df), len(mc_df))
                        geom = [Point(xy) for xy in zip(
                            base_df[x_col].iloc[:n], base_df[y_col].iloc[:n])]
                        gdf = gpd.GeoDataFrame(mc_df.iloc[:n], geometry=geom, crs=crs)
                        if str(gdf.crs) != "EPSG:4326":
                            gdf = gdf.to_crs(epsg=4326)
                        geojson_data = gdf.__geo_interface__
                        print(f"[SPARC] Reconstructed GeoJSON from MC consensus CSV ({n} features)")
            except Exception as exc:
                print(f"[SPARC] CSV→GeoJSON fallback failed: {exc}")

    # Load summary CSV (check all mode variants + MC consensus)
    summary_data = []
    for summary_name in (
        "scenario_summary.csv",
        "scenario_summary_dag.csv",
        "scenario_summary_hybrid.csv",
        "scenario_summary_reprediction.csv",
        "scenario_mc_consensus_summary.csv",
        "scenario_mc_consensus.csv",
    ):
        summary_path = paths.stage4_dir / summary_name
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_data = summary_df.to_dict(orient="records")
            break

    return {"geojson": geojson_data, "summary": summary_data}


@app.get("/results/report")
async def get_report_data():
    """Compile all stage results into a structured payload for the report view."""
    import pandas as pd
    import json as _json

    cfg = state.project_config or {}

    report: dict[str, Any] = {}

    # Project info
    raw = state.raw_project_yaml or {}
    report["project"] = raw.get("project", {})
    report["data_summary"] = state.data_summary or {}

    # Predictors
    predictors = cfg.get("predictors", {})
    if isinstance(predictors, list):
        report["predictors"] = predictors
    elif isinstance(predictors, dict):
        report["predictors"] = predictors.get("base_model", [])
    else:
        report["predictors"] = []

    # Causal + physics + pipeline config
    report["causal"] = cfg.get("causal", {})
    report["physics"] = cfg.get("physics", {})
    report["pipeline"] = cfg.get("pipeline", {})

    # Stage-specific structured data
    from sparc.run.pipeline_paths import PipelinePaths
    try:
        paths = PipelinePaths.from_config(cfg)
    except Exception as exc:
        print(f"[SPARC] Report: PipelinePaths error: {exc}")
        paths = None

    if paths:
        print(f"[SPARC] Report: output_dir={paths.output_dir}")
        print(f"[SPARC] Report: stage3_dir={paths.stage3_dir} exists={paths.stage3_dir.exists()}")
        print(f"[SPARC] Report: stage4_dir={paths.stage4_dir} exists={paths.stage4_dir.exists()}")
        coeff_check = paths.stage3_dir / "scenario_coefficients.json"
        print(f"[SPARC] Report: coeff_path={coeff_check} exists={coeff_check.exists()}")
        if paths.stage4_dir.exists():
            s4_files = list(paths.stage4_dir.glob("*"))
            print(f"[SPARC] Report: stage4 files={[f.name for f in s4_files]}")

    # Correlogram summary (Stage 0)
    if paths:
        corr_path = paths.stage0_dir / "correlogram_analysis_results.json"
        if not corr_path.exists():
            corr_path = paths.stage0_dir / "correlogram_results.json"
        if corr_path.exists():
            with open(corr_path, "r", encoding="utf-8") as fh:
                corr_data = _json.load(fh)
            # Extract just the summary metrics per variable
            individual = corr_data.get("individual_results", {})
            report["correlogram"] = {
                var: {
                    "optimal_bandwidth": info.get("optimal_bandwidth"),
                    "effective_range": info.get("effective_range"),
                    "max_moran_i": info.get("max_moran_i"),
                }
                for var, info in individual.items()
            }

    # GWEN summary (Stage 1)
    if paths:
        csv_path = paths.stage1_dir / "gwen_variable_importance.csv"
        if not csv_path.exists():
            csv_path = paths.output_dir / "gwen_variable_importance.csv"  # legacy fallback
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            report["gwen"] = df.to_dict(orient="records")

    # Spatial CV performance
    if paths:
        oof_path = paths.stage2_dir / "optimized_oof_predictions.csv"
        if oof_path.exists():
            oof_df = pd.read_csv(oof_path)
            report["spatial_cv_models"] = list(oof_df.columns)

    # Causal coefficients (Stage 3)
    if paths:
        coeff_path = paths.stage3_dir / "scenario_coefficients.json"
        if coeff_path.exists():
            with open(coeff_path, "r", encoding="utf-8") as fh:
                report["causal_results"] = _json.load(fh)
        # Also load dose-response curves if available
        dr_path = paths.stage3_dir / "dose_response_curves.json"
        if dr_path.exists():
            with open(dr_path, "r", encoding="utf-8") as fh:
                report["dose_response"] = _json.load(fh)
        # Propensity diagnostics
        prop_path = paths.stage3_dir / "propensity_diagnostics.json"
        if prop_path.exists():
            with open(prop_path, "r", encoding="utf-8") as fh:
                report["propensity_diagnostics"] = _json.load(fh)

    # Scenario summary (check all mode variants + MC consensus)
    if paths:
        for scenario_name in (
            "scenario_summary.csv",
            "scenario_summary_dag.csv",
            "scenario_summary_hybrid.csv",
            "scenario_summary_reprediction.csv",
            "scenario_mc_consensus_summary.csv",
            "scenario_mc_consensus.csv",
        ):
            summary_path = paths.stage4_dir / scenario_name
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                report["scenario_summary"] = summary_df.to_dict(orient="records")
                break

    # Plot URLs per stage
    plot_stages: dict[str, list] = {}
    for stage_num, stage_dir in [(0, paths.stage0_dir if paths else None),
                                  (1, paths.stage1_dir if paths else None),
                                  (2, paths.stage2_dir if paths else None),
                                  (3, paths.stage3_dir if paths else None),
                                  (4, paths.stage4_dir if paths else None)]:
        if stage_dir and stage_dir.exists():
            plots = []
            for ext in ("*.png", "*.svg"):
                for f in sorted(stage_dir.rglob(ext)):
                    plots.append({
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f.relative_to(stage_dir)),
                        "stage": stage_num,
                    })
            if plots:
                plot_stages[f"stage_{stage_num}"] = plots
    report["plots"] = plot_stages

    return report


# ------------------------------------------------------------------
# CATE map, local coefficients, & increment endpoints (Phase 2)
# ------------------------------------------------------------------

@app.get("/results/causal/cate_map")
async def get_cate_map(variable: str = Query(...)):
    """Return spatial CATE multiplier for *variable* as GeoJSON."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import numpy as np

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    # Find the .npy CATE multiplier file
    npy_name = f"spatial_cate_multiplier_{variable}.npy"
    npy_path = paths.stage3_dir / npy_name
    if not npy_path.exists():
        raise HTTPException(404, f"No CATE map for variable '{variable}'")

    multiplier = np.load(npy_path, allow_pickle=False)

    # Also try to load spatial_cate_maps.gpkg for geometry
    gpkg_path = paths.stage3_dir / "spatial_cate_maps.gpkg"
    if gpkg_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_path)
        cate_col = f"cate_{variable}"
        if cate_col not in gdf.columns and len(multiplier) == len(gdf):
            gdf[cate_col] = multiplier
        if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        return gdf.__geo_interface__

    # Fallback: reconstruct from data coordinates + multiplier array
    try:
        cfg = state.project_config
        coord_cols = cfg.get("data", {}).get("coord_columns",
                     cfg.get("variables", {}).get("coordinates", []))
        data_file = cfg.get("data", {}).get("file_path",
                    cfg.get("paths", {}).get("raw_csv_path"))
        crs = cfg.get("crs", {}).get("projected",
              cfg.get("crs", {}).get("target_projected", "EPSG:4326"))

        if data_file and len(coord_cols) == 2:
            import pandas as pd
            import geopandas as gpd
            from shapely.geometry import Point

            base_df = pd.read_csv(data_file)
            x_col, y_col = coord_cols
            n = min(len(base_df), len(multiplier))
            geom = [Point(xy) for xy in zip(
                base_df[x_col].iloc[:n], base_df[y_col].iloc[:n])]
            gdf = gpd.GeoDataFrame(
                {f"cate_{variable}": multiplier[:n]},
                geometry=geom, crs=crs,
            )
            if str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs(epsg=4326)
            return gdf.__geo_interface__
    except Exception as exc:
        raise HTTPException(500, f"Could not build CATE GeoJSON: {exc}")

    raise HTTPException(404, "Cannot reconstruct CATE map geometry")


@app.get("/results/causal/cate_map/variables")
async def get_cate_variables():
    """Return list of variables that have spatial CATE multiplier maps."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    variables = []
    # 1. Check for per-variable .npy files
    for f in paths.stage3_dir.glob("spatial_cate_multiplier_*.npy"):
        var = f.stem.replace("spatial_cate_multiplier_", "")
        variables.append(var)
    # 2. Fallback: check spatial_cate_maps.gpkg column names
    if not variables:
        gpkg_path = paths.stage3_dir / "spatial_cate_maps.gpkg"
        if gpkg_path.exists():
            try:
                import geopandas as gpd
                gdf = gpd.read_file(gpkg_path, rows=0)  # schema only
                for col in gdf.columns:
                    if col.startswith("cate_"):
                        variables.append(col.replace("cate_", "", 1))
            except Exception:
                pass
    return {"variables": sorted(set(variables))}


@app.get("/results/local_coefficients")
async def get_local_coefficients(variable: str = Query(...)):
    """Return MGWR/GWR spatially-varying coefficients for *variable* as GeoJSON."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import pandas as pd
    import numpy as np

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    cfg = state.project_config

    # Check for pre-extracted local coefficient CSV
    coef_csv = paths.stage2_dir / "base_models_full" / "mgwr_local_coefficients.csv"
    if not coef_csv.exists():
        coef_csv = paths.stage2_dir / "base_models_full" / "gwr_local_coefficients.csv"

    if not coef_csv.exists():
        raise HTTPException(404, "Local coefficient data not found. Run Stage 2b first.")

    coef_df = pd.read_csv(coef_csv)
    if variable not in coef_df.columns:
        available = [c for c in coef_df.columns if not c.startswith("_") and c not in ("Intercept", "geometry")]
        raise HTTPException(404, f"Variable '{variable}' not in coefficient data. Available: {available}")

    # Build GeoJSON
    coord_cols = cfg.get("data", {}).get("coord_columns",
                 cfg.get("variables", {}).get("coordinates", []))
    data_file = cfg.get("data", {}).get("file_path",
                cfg.get("paths", {}).get("raw_csv_path"))
    crs = cfg.get("crs", {}).get("projected",
          cfg.get("crs", {}).get("target_projected", "EPSG:4326"))

    if not data_file or len(coord_cols) != 2:
        raise HTTPException(500, "Cannot resolve coordinate columns")

    import geopandas as gpd
    from shapely.geometry import Point

    base_df = pd.read_csv(data_file)
    x_col, y_col = coord_cols
    n = min(len(base_df), len(coef_df))
    geom = [Point(xy) for xy in zip(
        base_df[x_col].iloc[:n], base_df[y_col].iloc[:n])]

    # Include physics constraint metadata
    expected_sign = _resolve_expected_sign(cfg, variable)

    coef_values = coef_df[variable].iloc[:n].values.astype(float)
    sign_ok = (
        (coef_values < 0) if expected_sign == "negative"
        else (coef_values > 0) if expected_sign == "positive"
        else np.ones(n, dtype=bool)
    )
    result = {
        "coefficient": coef_values,
        "sign_correct": sign_ok,
        "expected_sign": np.full(n, expected_sign or "none", dtype=object),
    }
    gdf = gpd.GeoDataFrame(result, geometry=geom, crs=crs)
    if str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf.__geo_interface__


@app.get("/results/local_coefficients/variables")
async def get_local_coef_variables():
    """Return list of variables with local coefficient data."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import pandas as pd

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    coef_csv = paths.stage2_dir / "base_models_full" / "mgwr_local_coefficients.csv"
    if not coef_csv.exists():
        coef_csv = paths.stage2_dir / "base_models_full" / "gwr_local_coefficients.csv"
    if not coef_csv.exists():
        return {"variables": []}

    df = pd.read_csv(coef_csv, nrows=0)
    skip = {"Intercept", "geometry", "_intercept"}
    variables = [c for c in df.columns if c not in skip and not c.startswith("_")]
    return {"variables": variables}


@app.get("/results/scenarios/variables")
async def get_scenario_variables():
    """Return the list of scenario variables and their available increments.

    Response shape::

        {
            "variables": {
                "Pct_Canopy": { "increments": [5, 10, 15, ...], "sign": "plus" },
                "Pct_Impervious": { "increments": [-5, -10, ...], "sign": "minus" }
            }
        }
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import pandas as pd

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    # Load summary CSV
    summary_path = None
    for name in ("scenario_summary.csv", "scenario_summary_dag.csv",
                 "scenario_summary_hybrid.csv", "scenario_summary_reprediction.csv",
                 "scenario_mc_consensus_summary.csv"):
        candidate = paths.stage4_dir / name
        if candidate.exists():
            summary_path = candidate
            break
    if summary_path is None:
        raise HTTPException(404, "No scenario summary found")

    summary_df = pd.read_csv(summary_path)
    variables: dict[str, dict] = {}
    for _, row in summary_df.iterrows():
        var = str(row.get("Variable", ""))
        inc = row.get("Increment")
        if not var or inc is None:
            continue
        inc = float(inc)
        if var not in variables:
            variables[var] = {"increments": [], "sign": "plus" if inc >= 0 else "minus"}
        if inc not in variables[var]["increments"]:
            variables[var]["increments"].append(inc)

    for info in variables.values():
        info["increments"].sort(key=lambda x: abs(x))

    return {"variables": variables}


@app.get("/results/scenarios/increment")
async def get_scenario_increment(variable: str = Query(...), increment: float = Query(...)):
    """Return GeoJSON filtered to a specific variable+increment scenario."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import pandas as pd

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    # Load summary CSV to find matching scenario label
    summary_path = None
    for name in ("scenario_summary.csv", "scenario_summary_dag.csv",
                 "scenario_summary_hybrid.csv", "scenario_summary_reprediction.csv",
                 "scenario_mc_consensus_summary.csv"):
        candidate = paths.stage4_dir / name
        if candidate.exists():
            summary_path = candidate
            break
    if summary_path is None:
        raise HTTPException(404, "No scenario summary found")

    summary_df = pd.read_csv(summary_path)
    # Filter to matching variable + increment
    mask = (
        (summary_df["Variable"].str.lower() == variable.lower()) &
        (summary_df["Increment"].astype(float).round(6) == round(float(increment), 6))
    )
    matched = summary_df[mask]
    if matched.empty:
        raise HTTPException(404, f"No scenario found for {variable} at increment {increment}")

    # Load spatial results (cached in server state)
    gdf = _load_scenario_gpkg(paths)
    if gdf is None:
        raise HTTPException(404, "No spatial results found")

    # Find the scenario column — convention: pred_{Scenario_Label}
    scenario_label = matched.iloc[0].get("Scenario", "")
    pred_col = f"pred_{scenario_label}"
    delta_col = f"delta_{scenario_label}"
    baseline_col = next((c for c in gdf.columns if c == "pred_baseline"), None)

    cols_to_keep = ["geometry"]
    if baseline_col and baseline_col in gdf.columns:
        cols_to_keep.append(baseline_col)
    if pred_col in gdf.columns:
        cols_to_keep.append(pred_col)
    if delta_col in gdf.columns:
        cols_to_keep.append(delta_col)
    elif pred_col in gdf.columns and baseline_col:
        gdf[delta_col] = gdf[pred_col].astype(float) - gdf[baseline_col].astype(float)
        cols_to_keep.append(delta_col)

    result_gdf = gdf[cols_to_keep].copy()
    if result_gdf.crs is not None and str(result_gdf.crs) != "EPSG:4326":
        result_gdf = result_gdf.to_crs(epsg=4326)

    return {
        "geojson": result_gdf.__geo_interface__,
        "summary": matched.to_dict(orient="records"),
    }


# ------------------------------------------------------------------
# Results endpoints (parameterized — MUST come after named routes above)
# ------------------------------------------------------------------

@app.get("/results/{stage}")
async def get_results(stage: int, format: str = Query("json", regex="^(json|geojson)$")):
    """Return results for a completed pipeline stage."""
    result = state.get_result(stage)

    if result is None:
        # Try loading from disk if the stage directory exists
        result = _try_load_from_disk(stage)
        if result is None:
            raise HTTPException(404, f"No results for stage {stage}")

    if format == "geojson":
        return _to_geojson(result)
    return _to_json(result)


@app.get("/results/{stage}/predictions")
async def get_predictions(stage: int, format: str = Query("geojson", regex="^(json|geojson)$")):
    """Return spatial predictions for a stage (optimized for deck.gl)."""
    result = state.get_result(stage)
    if result is None:
        result = _try_load_from_disk(stage)
        if result is None:
            raise HTTPException(404, f"No predictions for stage {stage}")

    if isinstance(result, dict) and "spatial" in result:
        gdf = result["spatial"]
    else:
        gdf = result

    if format == "geojson":
        return _to_geojson(gdf)
    return _to_json(gdf)


@app.get("/results/{stage}/plots")
async def list_stage_plots(stage: int):
    """List available plot images for a completed pipeline stage."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    stage_map = {0: paths.stage0_dir, 1: paths.stage1_dir, 2: paths.stage2_dir, 3: paths.stage3_dir, 4: paths.stage4_dir}
    stage_dir = stage_map.get(stage)
    if stage_dir is None or not stage_dir.exists():
        return {"plots": []}

    # Collect PNG/SVG files recursively
    plots = []
    for ext in ("*.png", "*.svg"):
        for f in sorted(stage_dir.rglob(ext)):
            plots.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f.relative_to(stage_dir)),
            })
    return {"plots": plots}


@app.get("/results/{stage}/plots/{file_path:path}")
async def get_stage_plot(stage: int, file_path: str):
    """Serve a specific plot image file from a stage output directory."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    stage_map = {0: paths.stage0_dir, 1: paths.stage1_dir, 2: paths.stage2_dir, 3: paths.stage3_dir, 4: paths.stage4_dir}
    stage_dir = stage_map.get(stage)
    if stage_dir is None:
        raise HTTPException(404, "Invalid stage")

    full_path = (stage_dir / file_path).resolve()
    # Security: ensure the resolved path is within the stage directory
    if not str(full_path).startswith(str(stage_dir.resolve())):
        raise HTTPException(403, "Access denied")
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, f"Plot not found: {file_path}")

    media = "image/png" if full_path.suffix == ".png" else "image/svg+xml"
    return FileResponse(full_path, media_type=media)


# ------------------------------------------------------------------
# Scenario endpoints
# ------------------------------------------------------------------

@app.post("/scenarios/run")
async def run_scenarios():
    """Execute all scenarios defined in the loaded project."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    scenarios = state.project_config.get("scenarios", [])
    interaction_scenarios = state.project_config.get("interaction_scenarios", [])
    if not scenarios and not interaction_scenarios:
        raise HTTPException(400, "No scenarios defined in project.yml")

    # Run synchronously for now; a production version would use
    # background tasks or the WebSocket streaming endpoint.
    from sparc.interventions.scenario_simulator import ScenarioSimulator
    import pandas as pd

    config = state.project_config
    sim = ScenarioSimulator(config)
    sim.load_models()

    csv_path = config["paths"]["raw_csv_path"]
    data = pd.read_csv(csv_path)

    dag_file = config.get("causal", {}).get("dag_file")
    has_dag = dag_file and Path(dag_file).exists()
    scenario_mode = config.get("pipeline", {}).get("scenario_mode", "auto")

    if scenario_mode == "auto":
        scenario_mode = "hybrid" if has_dag else "physics"

    if scenario_mode == "hybrid":
        summary_df, results_gdf = sim.run_with_hybrid_reprediction(data, verbose=True)
    elif scenario_mode == "model_reprediction":
        summary_df, results_gdf = sim.run_with_model_reprediction(data, verbose=True)
    elif scenario_mode == "dag_coefficient":
        if has_dag:
            summary_df, results_gdf = sim.run_with_causal_dag(data, verbose=True)
        else:
            summary_df, results_gdf = sim.run(verbose=True)
    else:
        summary_df, results_gdf = sim.run(verbose=True)

    # --- Conservation checks on scenario results ---------------------
    conservation_violations = []
    try:
        import numpy as np
        from sparc.interventions.physics_priors import ConservationChecker
        checker = ConservationChecker()
        for scenario in scenarios:
            var = scenario['variable']
            for inc in scenario.get('increments', []):
                direction = scenario.get('direction', 'increase')
                delta_signed = -inc if direction == 'decrease' else inc
                col_label = f"total_{var}_{'minus' if delta_signed < 0 else 'plus'}_{str(inc).replace('.', 'p')}"
                if hasattr(results_gdf, 'columns') and col_label in results_gdf.columns:
                    deltas = {var: np.full(len(data), delta_signed)}
                    target_deltas = results_gdf[col_label].values
                    violations = checker.check(data, deltas, target_deltas=target_deltas, verbose=True)
                    conservation_violations.extend(violations)
    except Exception as e:
        print(f"  [CONSERVATION] Check skipped ({e})")

    state.store_result(4, {"summary": summary_df, "spatial": results_gdf})

    return {
        "status": "complete",
        "n_scenarios": len(scenarios) + len(interaction_scenarios),
        "summary_rows": len(summary_df),
        "scenario_mode": scenario_mode,
        "conservation_violations": len(conservation_violations),
    }


@app.get("/scenarios/results")
async def scenario_results(format: str = Query("geojson", regex="^(json|geojson)$")):
    """Return scenario simulation results."""
    result = state.get_result(4)
    if result is None:
        raise HTTPException(404, "No scenario results. Run /scenarios/run first.")

    if format == "geojson" and "spatial" in result:
        return _to_geojson(result["spatial"])

    if "summary" in result:
        return _to_json(result["summary"])

    return _to_json(result)


# ------------------------------------------------------------------
# Physics defaults endpoint
# ------------------------------------------------------------------

@app.get("/physics/defaults")
async def physics_defaults():
    """Return default literature-based physics priors."""
    try:
        from sparc.interventions.physics_priors import PhysicsPriors
        pp = PhysicsPriors()
        return {
            k: c.to_dict() for k, c in pp.coefficients.items()
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------------------
# Data Processing endpoints
# ------------------------------------------------------------------

@app.post("/data/fishnet")
async def create_fishnet_endpoint(body: dict = Body(...)):
    """Create a fishnet grid from bounds + resolution, optionally clip to boundary."""
    from sparc.data.processing import create_fishnet, clip_to_boundary

    bounds = body.get("bounds")  # [minx, miny, maxx, maxy]
    resolution = body.get("resolution", 100)
    crs = body.get("crs", "EPSG:4326")

    if not bounds or len(bounds) != 4:
        raise HTTPException(400, "bounds must be [minx, miny, maxx, maxy]")

    gdf = create_fishnet(tuple(bounds), resolution, crs)

    boundary_path = body.get("boundary_path")
    if boundary_path and Path(boundary_path).exists():
        import geopandas as _gpd
        boundary = _gpd.read_file(boundary_path)
        if boundary.crs and boundary.crs.to_string() != crs:
            boundary = boundary.to_crs(crs)
        gdf = clip_to_boundary(gdf, boundary)

    # Persist to project directory
    if state.project_config:
        out_dir = Path(state.project_config.get("paths", {}).get("project_dir", "."))
        out_path = out_dir / "fishnet.gpkg"
        gdf.to_file(out_path, driver="GPKG")

    return {"n_cells": len(gdf), "columns": list(gdf.columns)}


@app.post("/data/zonal_stats")
async def zonal_stats_endpoint(body: dict = Body(...)):
    """Compute zonal statistics for raster layers on a fishnet."""
    from sparc.data.processing import run_zonal_stats
    import geopandas as _gpd

    fishnet_path = body.get("fishnet_path")
    raster_paths = body.get("raster_paths", [])
    stats = body.get("stats", "mean")

    if not fishnet_path or not Path(fishnet_path).exists():
        raise HTTPException(400, "fishnet_path not found")
    if not raster_paths:
        raise HTTPException(400, "raster_paths required")

    gdf = _gpd.read_file(fishnet_path)
    gdf = run_zonal_stats(gdf, raster_paths, stats=stats)

    out_path = Path(fishnet_path).with_name("fishnet_with_stats.gpkg")
    gdf.to_file(out_path, driver="GPKG")

    # Also export CSV for pipeline consumption
    csv_path = out_path.with_suffix(".csv")
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    return {"n_cells": len(gdf), "columns": list(gdf.columns), "csv_path": str(csv_path)}


@app.post("/data/prepare")
async def prepare_data_pipeline(body: dict = Body(...)):
    """Unified pipeline: boundary + rasters → fishnet → zonal stats → CSV.

    Body fields:
        boundary_path: str — path to boundary shapefile/GPKG/GeoJSON
        raster_paths: list[str] — paths to GeoTIFF raster layers
        resolution: float — fishnet cell size (in CRS units, default 100)
        crs: str — target CRS (default EPSG:4326)
        stats: str — zonal stat types, space-separated (default "mean")
        set_as_data: bool — if true, set the result CSV as the project data file (default true)
    """
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

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

    # Step 1: Determine bounds — from boundary or from rasters
    boundary_gdf = None
    if boundary_path and Path(boundary_path).exists():
        boundary_gdf = _gpd.read_file(boundary_path)
        if boundary_gdf.crs and boundary_gdf.crs.to_string() != crs:
            boundary_gdf = boundary_gdf.to_crs(crs)
        bounds = tuple(boundary_gdf.total_bounds)
    else:
        # Derive bounds from first raster
        try:
            import rasterio
            with rasterio.open(raster_paths[0]) as src:
                bounds = tuple(src.bounds)
                if not crs:
                    crs = str(src.crs)
        except Exception as exc:
            raise HTTPException(400, f"Cannot determine bounds: {exc}")

    # Step 2: Generate fishnet
    gdf = create_fishnet(bounds, resolution, crs)

    # Step 3: Clip to boundary
    if boundary_gdf is not None:
        gdf = clip_to_boundary(gdf, boundary_gdf)

    if len(gdf) == 0:
        raise HTTPException(400, "Fishnet has 0 cells after clipping — check CRS/resolution")

    # Step 4: Run zonal statistics
    valid_rasters = [r for r in raster_paths if Path(r).exists()]
    if not valid_rasters:
        raise HTTPException(400, "None of the raster paths exist")
    gdf = run_zonal_stats(gdf, valid_rasters, stats=stats)

    # Step 5: Save outputs (versioned)
    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = data_dir / "fishnet_with_stats.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    # Add coordinate columns before saving
    export_df = gdf.copy()
    export_df["centroid_x"] = gdf.geometry.centroid.x
    export_df["centroid_y"] = gdf.geometry.centroid.y
    export_flat = export_df.drop(columns=["geometry"])

    # Save versioned snapshot
    from sparc.data.versioning import save_versioned
    version_info = save_versioned(
        export_flat,
        data_dir,
        settings={
            "resolution": resolution,
            "crs": crs,
            "stats": stats,
            "n_rasters": len(valid_rasters),
            "boundary": boundary_path,
        },
        description=f"Fishnet processing: {len(valid_rasters)} rasters, "
                    f"resolution={resolution}, CRS={crs}",
    )
    csv_path = version_info["csv_path"]

    # Also save as canonical name for backwards compatibility
    canonical_csv = data_dir / "fishnet_with_stats.csv"
    export_flat.to_csv(canonical_csv, index=False)

    # Step 6: Optionally set as active data
    if set_as_data:
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


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

@app.post("/report/generate")
async def generate_report(format: str = Query("markdown", regex="^(markdown|json)$")):
    """Generate a pipeline report summarising config, data, and results.

    Returns Markdown by default; JSON structure when ``format=json``.
    """
    import pandas as pd

    sections: list[str] = []
    report_data: dict[str, Any] = {}

    # --- Project info ---
    cfg = state.project_config or {}
    proj = cfg.get("project", {})
    sections.append(f"# SPARC Pipeline Report\n")
    sections.append(f"**Project:** {proj.get('name', 'Untitled')}\n")
    sections.append(f"**Domain:** {proj.get('domain', 'N/A')}\n")
    if proj.get("description"):
        sections.append(f"**Description:** {proj['description']}\n")
    report_data["project"] = proj

    # --- Data overview ---
    if state.data_summary:
        s = state.data_summary
        sections.append("## Data Overview\n")
        sections.append(f"- **Rows:** {s.get('row_count', '?')}")
        sections.append(f"- **Columns:** {s.get('column_count', '?')}")
        if s.get("crs"):
            sections.append(f"- **CRS:** {s['crs']}")
        sections.append("")
        report_data["data_summary"] = s

    # --- Predictors ---
    predictors = cfg.get("predictors", {})
    if isinstance(predictors, list):
        sections.append(f"## Predictors ({len(predictors)})\n")
        sections.append(", ".join(f"`{p}`" for p in predictors))
        sections.append("")
        report_data["predictors"] = predictors
    elif isinstance(predictors, dict):
        base = predictors.get("base_model", [])
        sections.append(f"## Predictors ({len(base)})\n")
        sections.append(", ".join(f"`{p}`" for p in base))
        sections.append("")
        report_data["predictors"] = base

    # --- Causal config ---
    causal = cfg.get("causal", {})
    if causal:
        sections.append("## Causal Configuration\n")
        sections.append(f"- **Estimator:** {causal.get('estimator', 'N/A')}")
        sections.append(f"- **DAG blend weight:** {causal.get('dag_blend_weight', 'N/A')}")
        av = causal.get("actionable_variables", [])
        if av:
            sections.append(f"- **Actionable:** {', '.join(av)}")
        sections.append("")
        report_data["causal"] = causal

    # --- Physics priors ---
    physics = cfg.get("physics", {})
    if physics:
        mc = physics.get("monotone_constraints", {})
        if mc:
            sections.append("## Physics Constraints\n")
            for k, v in mc.items():
                direction = "increasing" if v > 0 else "decreasing" if v < 0 else "none"
                sections.append(f"- `{k}`: {direction}")
            sections.append("")
            report_data["physics"] = physics

    # --- Stage results summary ---
    sections.append("## Results Summary\n")
    sections.append("| Stage | Rows | Metrics |")
    sections.append("|-------|------|---------|")

    for stage_num in [0, 2, 3, 4]:
        result = state.get_result(stage_num)
        if result is None:
            continue
        if isinstance(result, dict) and "summary" in result:
            df = result["summary"]
        elif isinstance(result, pd.DataFrame):
            df = result
        elif isinstance(result, dict) and "rows" in result:
            df = pd.DataFrame(result["rows"])
        else:
            continue

        if hasattr(df, "__len__"):
            n_rows = len(df)
            n_metrics = len(df.columns) if hasattr(df, "columns") else 0
            sections.append(f"| Stage {stage_num} | {n_rows} | {n_metrics} |")
            report_data[f"stage_{stage_num}"] = {"rows": n_rows, "metrics": n_metrics}

    sections.append("")

    # --- Pipeline config ---
    pipeline = cfg.get("pipeline", {})
    if pipeline:
        sections.append("## Pipeline Settings\n")
        sections.append(f"- **Random seed:** {pipeline.get('random_seed', 'N/A')}")
        sections.append(f"- **Spatial folds:** {pipeline.get('n_spatial_folds', 'N/A')}")
        sections.append(f"- **Fast mode:** {pipeline.get('fast_mode', False)}")
        sections.append("")

    md_text = "\n".join(sections)

    if format == "json":
        return {"report": report_data, "markdown": md_text}

    return JSONResponse(content={"markdown": md_text}, media_type="application/json")


@app.post("/report/pdf")
async def generate_pdf_report():
    """Generate a PDF report with embedded plots and return it as a file download."""
    from starlette.responses import Response as StarletteResponse

    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    cfg = state.project_config
    output_dir = cfg.get("paths", {}).get("output_dir")

    # Collect causal results from state
    causal_results = None
    causal_state = state.get_result(3)
    if isinstance(causal_state, dict):
        causal_results = causal_state

    # Collect scenario summary
    scenario_summary = None
    scenario_state = state.get_result(4)
    if isinstance(scenario_state, dict) and "summary" in scenario_state:
        import pandas as pd
        s = scenario_state["summary"]
        if isinstance(s, pd.DataFrame):
            scenario_summary = s.to_dict(orient="records")
        elif isinstance(s, list):
            scenario_summary = s

    try:
        from sparc.report import generate_report_pdf

        pdf_bytes = generate_report_pdf(
            config=cfg,
            data_summary=state.data_summary,
            causal_results=causal_results,
            scenario_summary=scenario_summary,
            output_dir=output_dir,
        )

        # Also save a copy to the project directory
        project_dir = Path(cfg["paths"]["project_root"])
        dest = project_dir / "sparc_report.pdf"
        if isinstance(pdf_bytes, Path):
            pdf_data = pdf_bytes.read_bytes()
        else:
            pdf_data = pdf_bytes
            dest.write_bytes(pdf_data)

        return StarletteResponse(
            content=pdf_data,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=sparc_report.pdf"},
        )
    except RuntimeError as exc:
        # weasyprint not installed — fall back to HTML
        from sparc.report import generate_report_html

        html_str = generate_report_html(
            config=cfg,
            data_summary=state.data_summary,
            causal_results=causal_results,
            scenario_summary=scenario_summary,
            output_dir=output_dir,
        )
        # Save HTML to disk
        project_dir = Path(cfg["paths"]["project_root"])
        dest = project_dir / "sparc_report.html"
        dest.write_text(html_str, encoding="utf-8")

        return {
            "status": "html_fallback",
            "message": str(exc),
            "html_path": str(dest),
        }


# ------------------------------------------------------------------
# DAG endpoints
# ------------------------------------------------------------------

@app.get("/dag")
async def get_dag():
    """Return the project's DAG definition."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    dag_file = state.project_config.get("causal", {}).get("dag_file")
    if not dag_file or not Path(dag_file).exists():
        return {"nodes": [], "edges": []}

    from sparc.causal.dag_definition import load_dag
    dag = load_dag(dag_file)
    return dag


@app.post("/dag/validate")
async def validate_dag(dag: dict):
    """Validate a DAG definition without saving it."""
    from sparc.causal.dag_definition import dag_to_networkx
    import networkx as nx

    try:
        G = dag_to_networkx(dag)
        is_dag = nx.is_directed_acyclic_graph(G)
        return {
            "valid": is_dag,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "error": None if is_dag else "Graph contains cycles",
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@app.get("/dag/mc3_result")
async def get_mc3_result():
    """Return pending MC³ edge-inclusion probabilities for DAG approval."""
    if state.pending_mc3 is None:
        raise HTTPException(404, "No MC³ result pending")
    return state.pending_mc3


@app.post("/dag/approve")
async def approve_dag():
    """Approve the discovered DAG and unblock the pipeline."""
    if state.pending_mc3 is None:
        raise HTTPException(400, "No MC³ result pending approval")
    state.dag_approved.set()
    return {"status": "approved"}


@app.post("/dag/reject")
async def reject_dag():
    """Reject the discovered DAG and cancel the pipeline."""
    if state.pending_mc3 is None:
        raise HTTPException(400, "No MC³ result pending approval")
    # Unblock the gate — the pipeline will continue (we clear pending_mc3
    # but the pipeline thread checks is_running which cancel sets to false)
    state.pending_mc3 = None
    state.dag_approved.set()
    return {"status": "rejected"}


# ------------------------------------------------------------------
# Artifact download
# ------------------------------------------------------------------

@app.get("/results/artifacts")
async def list_artifacts():
    """List all downloadable output files across all stages."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        raise HTTPException(400, f"Cannot resolve output paths: {exc}")

    artifacts = []
    stage_dirs = {
        0: ("Stage 0 — Correlogram", paths.stage0_dir),
        1: ("Stage 1 — GWEN", paths.stage1_dir),
        2: ("Stage 2 — Spatial CV", paths.stage2_dir),
        3: ("Stage 3 — Causal", paths.stage3_dir),
        4: ("Stage 4 — Scenarios", paths.stage4_dir),
    }

    for stage_num, (stage_label, stage_dir) in stage_dirs.items():
        if stage_dir is None or not stage_dir.exists():
            continue
        for f in sorted(stage_dir.rglob("*")):
            if f.is_file() and f.suffix in (
                ".csv", ".parquet", ".geojson", ".gpkg", ".png", ".html",
                ".json", ".pdf", ".svg", ".yml", ".yaml",
            ):
                rel = f.relative_to(stage_dir)
                artifacts.append({
                    "stage": stage_num,
                    "stage_label": stage_label,
                    "filename": f.name,
                    "relative_path": str(rel),
                    "absolute_path": str(f),
                    "size_bytes": f.stat().st_size,
                    "extension": f.suffix,
                })

    return {"artifacts": artifacts, "total": len(artifacts)}


@app.get("/results/download/{stage}/{file_path:path}")
async def download_artifact(stage: int, file_path: str):
    """Download a single output file by stage and relative path."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        raise HTTPException(400, f"Cannot resolve output paths: {exc}")

    stage_map = {
        0: paths.stage0_dir,
        1: paths.stage1_dir,
        2: paths.stage2_dir,
        3: paths.stage3_dir,
        4: paths.stage4_dir,
    }
    stage_dir = stage_map.get(stage)
    if stage_dir is None or not stage_dir.exists():
        raise HTTPException(404, f"Stage {stage} output directory not found")

    target = (stage_dir / file_path).resolve()
    # Security: ensure target is within stage directory
    if not str(target).startswith(str(stage_dir.resolve())):
        raise HTTPException(403, "Path traversal not allowed")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, f"File not found: {file_path}")

    return FileResponse(
        target,
        filename=target.name,
        media_type="application/octet-stream",
    )


@app.get("/results/geopackage")
async def download_geopackage():
    """Merge all spatial stage outputs into a single GeoPackage download."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    import geopandas as gpd
    from sparc.run.pipeline_paths import PipelinePaths
    import tempfile

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        raise HTTPException(400, f"Cannot resolve output paths: {exc}")

    # Collect spatial layers from stage directories
    layers: dict[str, gpd.GeoDataFrame] = {}

    layer_sources = [
        (paths.stage2_dir, "predictions", ["predictions*.parquet", "predictions*.geojson", "predictions*.csv"]),
        (paths.stage3_dir, "causal_effects", ["*cate*.parquet", "*cate*.geojson", "*causal*.parquet"]),
        (paths.stage4_dir, "scenario_deltas", ["*scenario*.parquet", "*scenario*.geojson", "*delta*.parquet"]),
    ]

    for stage_dir, layer_name, patterns in layer_sources:
        if stage_dir is None or not stage_dir.exists():
            continue
        for pat in patterns:
            files = list(stage_dir.glob(pat))
            if files:
                f = files[0]
                try:
                    if f.suffix == ".parquet":
                        gdf = gpd.read_parquet(f)
                    elif f.suffix == ".geojson":
                        gdf = gpd.read_file(f)
                    else:
                        import pandas as pd
                        gdf = pd.read_csv(f)
                        # If it has coordinate columns, try to make spatial
                        coord_cols = state.project_config.get("variables", {}).get("coordinates", [])
                        if len(coord_cols) >= 2 and all(c in gdf.columns for c in coord_cols):
                            from shapely.geometry import Point
                            gdf = gpd.GeoDataFrame(
                                gdf,
                                geometry=[Point(x, y) for x, y in zip(gdf[coord_cols[0]], gdf[coord_cols[1]])],
                                crs=state.project_config.get("crs", {}).get("target_projected", "EPSG:4326"),
                            )
                        else:
                            continue
                    if isinstance(gdf, gpd.GeoDataFrame) and len(gdf) > 0:
                        layers[layer_name] = gdf
                        break
                except Exception:
                    continue

    if not layers:
        raise HTTPException(404, "No spatial outputs found to package")

    # Write to a temp GeoPackage
    project_name = state.project_config.get("project", {}).get("name", "sparc_results")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".gpkg", prefix=f"{project_name}_", delete=False
    )
    tmp_path = tmp.name
    tmp.close()

    try:
        for layer_name, gdf in layers.items():
            gdf.to_file(tmp_path, layer=layer_name, driver="GPKG")
    except Exception as exc:
        raise HTTPException(500, f"GeoPackage creation failed: {exc}")

    return FileResponse(
        tmp_path,
        filename=f"{project_name}_results.gpkg",
        media_type="application/geopackage+sqlite3",
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_data_path(abs_path: str) -> None:
    """Update the data file path in memory *and* persist to project.yml on disk.

    This ensures that pipeline stages that re-read project.yml from disk
    (e.g. correlogram_analysis via ``load_config()``) see the correct path.
    """
    # 1. In-memory config
    state.project_config["data"]["file_path"] = abs_path
    state.project_config["paths"]["raw_csv_path"] = abs_path

    # 2. Raw YAML dict (for /project/config GET)
    if state.raw_project_yaml is not None:
        state.raw_project_yaml.setdefault("data", {})["file_path"] = abs_path

    # 3. Persist to disk so load_config() in subprocesses picks it up
    if state.project_path:
        import yaml
        yml_path = Path(state.project_path)
        if yml_path.is_dir():
            yml_path = yml_path / "project.yml"
        if yml_path.exists() and state.raw_project_yaml is not None:
            with open(yml_path, "w", encoding="utf-8") as fh:
                yaml.dump(state.raw_project_yaml, fh,
                          default_flow_style=False, sort_keys=False)


def _load_data_into_state(config: dict) -> None:
    """Load and preprocess the project's CSV into state.data."""
    from sparc.data.data_utils import load_and_preprocess_data

    data_path = config["data"]["file_path"]
    if not os.path.exists(data_path):
        return

    try:
        gdf = load_and_preprocess_data(
            raw_data_path=data_path,
            identifier_col=config["variables"]["identifier"],
            target_col=config["variables"]["target"],
            coords_cols=config["variables"]["coordinates"],
            predictor_cols=config["predictors"]["base_model"],
            initial_crs=config["crs"]["initial"],
            target_crs=config["crs"]["target_projected"],
            output_dir=config.get("output", {}).get("base_dir"),
        )
        state.data = gdf
        _compute_summary(state)
    except Exception as exc:
        print(f"Warning: data pre-load failed: {exc}")


def _compute_summary(st: ServerState) -> dict:
    """Build a compact data summary for the LLM and the UI."""
    if st.data is None:
        return {}
    import pandas as pd

    df = st.data
    numeric = df.select_dtypes(include="number")

    summary = {
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
            }
            for c in numeric.columns
        },
    }

    # CRS info if available
    if hasattr(df, "crs") and df.crs is not None:
        summary["crs"] = str(df.crs)

    # Bounding box
    if hasattr(df, "total_bounds"):
        bounds = df.total_bounds
        summary["bbox"] = {
            "minx": float(bounds[0]),
            "miny": float(bounds[1]),
            "maxx": float(bounds[2]),
            "maxy": float(bounds[3]),
        }

    st.data_summary = summary
    return summary


def _try_load_from_disk(stage: int) -> Any:
    """Attempt to load stage results from the output directory."""
    if state.project_config is None:
        return None

    from sparc.run.pipeline_paths import PipelinePaths
    import pandas as pd

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        return None

    stage_map = {
        0: paths.stage0_dir,
        1: paths.stage1_dir,
        2: paths.stage2_dir,
        3: paths.stage3_dir,
        4: paths.stage4_dir,
    }
    stage_dir = stage_map.get(stage)
    if stage_dir is None or not stage_dir.exists():
        return None

    # Look for common output files
    for pattern in ["*.csv", "*.parquet", "*.geojson"]:
        files = list(stage_dir.glob(pattern))
        if files:
            f = files[0]
            if f.suffix == ".parquet":
                import geopandas as gpd
                return gpd.read_parquet(f)
            elif f.suffix == ".geojson":
                import geopandas as gpd
                return gpd.read_file(f)
            else:
                return pd.read_csv(f)
    return None


def _to_geojson(data: Any) -> dict:
    """Convert a GeoDataFrame or DataFrame to GeoJSON."""
    import geopandas as gpd

    if isinstance(data, gpd.GeoDataFrame):
        return data.__geo_interface__
    if isinstance(data, dict) and "spatial" in data:
        return data["spatial"].__geo_interface__
    raise HTTPException(400, "Data is not spatial; use format=json")


def _to_json(data: Any) -> Any:
    """Convert a DataFrame or dict to JSON-serializable form."""
    import pandas as pd
    import geopandas as gpd

    if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        # Drop geometry for JSON serialization
        if isinstance(data, gpd.GeoDataFrame):
            data = pd.DataFrame(data.drop(columns="geometry"))
        return {"rows": data.to_dict(orient="records")}
    if isinstance(data, dict):
        return data
    return {"data": str(data)}
