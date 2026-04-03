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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query, HTTPException
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


@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Accept a CSV upload, store it, and load into state."""
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    # Save to a temp location inside the project directory
    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    # Update config to point to new file and persist to disk
    _set_data_path(str(dest))
    _load_data_into_state(state.project_config)

    return {
        "status": "uploaded",
        "path": str(dest),
        "columns": list(state.data.columns) if state.data is not None else [],
        "row_count": len(state.data) if state.data is not None else 0,
    }


@app.get("/data/files")
async def list_data_files():
    """List CSV/Parquet files inside the project's working directory."""
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    project_dir = Path(state.project_config["paths"]["project_root"])
    files: list[dict] = []
    for ext in ("*.csv", "*.parquet", "*.CSV"):
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


# ------------------------------------------------------------------
# Results endpoints
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

    stage_map = {0: paths.stage1_dir, 1: paths.stage1_dir, 2: paths.stage2_dir, 3: paths.stage3_dir, 4: paths.stage4_dir}
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

    stage_map = {0: paths.stage1_dir, 1: paths.stage1_dir, 2: paths.stage2_dir, 3: paths.stage3_dir, 4: paths.stage4_dir}
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
    if not scenarios:
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
        "n_scenarios": len(scenarios),
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
        0: paths.stage1_dir,
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


# ------------------------------------------------------------------
# Structured results endpoints
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
        paths.stage1_dir / "correlogram_analysis_results.json",
        paths.stage1_dir / "correlogram_results.json",
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

    # Try CSV first, then JSON
    csv_path = paths.stage1_dir / "gwen_variable_importance.csv"
    json_path = paths.stage1_dir / "gwen_results.json"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return {"rows": df.to_dict(orient="records")}
    elif json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fh:
            return _json.load(fh)
    else:
        raise HTTPException(404, "GWEN results not found. Run Stage 1 first.")


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
    if not gpkg_path.exists():
        raise HTTPException(404, "Spatial CV predictions gpkg not found. Run Stage 2 first.")

    import geopandas as gpd
    gdf = gpd.read_file(gpkg_path)
    # Reproject to WGS84 for web display
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf.__geo_interface__


@app.get("/results/causal")
async def get_causal_results():
    """Return causal validation results (coefficients, effects, diagnostics)."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths
    import json as _json

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception:
        raise HTTPException(404, "Cannot resolve output paths")

    coeff_path = paths.stage3_dir / "scenario_coefficients.json"
    if not coeff_path.exists():
        raise HTTPException(404, "Causal results not found. Run Stage 3 first.")

    with open(coeff_path, "r", encoding="utf-8") as fh:
        return _json.load(fh)


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

    # Check both possible locations
    candidates = [
        paths.spatial_analysis_dir / "gwrf_pdp" / "gwrf_condition_curves.json",
        paths.stage2_dir / "spatial_intelligence" / "gwrf_pdp" / "gwrf_condition_curves.json",
        paths.stage2_dir / "gwrf_pdp" / "gwrf_condition_curves.json",
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

    # Load scenario spatial results
    gpkg_path = paths.stage4_dir / "scenario_results.gpkg"
    geojson_data = None
    if gpkg_path.exists():
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
        if result is None:
            raise HTTPException(404, "No scenario results found. Run scenarios first.")
        spatial = result.get("spatial", result) if isinstance(result, dict) else result
        import geopandas as gpd
        if isinstance(spatial, gpd.GeoDataFrame):
            if spatial.crs is not None and str(spatial.crs) != "EPSG:4326":
                spatial = spatial.to_crs(epsg=4326)
            geojson_data = spatial.__geo_interface__
        else:
            raise HTTPException(404, "No spatial scenario results available")

    # Load summary CSV
    summary_data = []
    summary_path = paths.stage4_dir / "scenario_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_data = summary_df.to_dict(orient="records")

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
    except Exception:
        paths = None

    # Correlogram summary
    if paths:
        corr_path = paths.stage1_dir / "correlogram_analysis_results.json"
        if not corr_path.exists():
            corr_path = paths.stage1_dir / "correlogram_results.json"
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

    # GWEN summary
    if paths:
        csv_path = paths.stage1_dir / "gwen_variable_importance.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            report["gwen"] = df.to_dict(orient="records")

    # Spatial CV performance
    if paths:
        oof_path = paths.stage2_dir / "optimized_oof_predictions.csv"
        if oof_path.exists():
            oof_df = pd.read_csv(oof_path)
            report["spatial_cv_models"] = list(oof_df.columns)

    # Causal coefficients
    if paths:
        coeff_path = paths.stage3_dir / "scenario_coefficients.json"
        if coeff_path.exists():
            with open(coeff_path, "r", encoding="utf-8") as fh:
                report["causal_results"] = _json.load(fh)

    # Scenario summary
    if paths:
        summary_path = paths.stage4_dir / "scenario_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            report["scenario_summary"] = summary_df.to_dict(orient="records")

    # Plot URLs per stage
    plot_stages: dict[str, list] = {}
    for stage_num, stage_dir in [(0, paths.stage1_dir if paths else None),
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
