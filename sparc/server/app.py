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

import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from sparc.server.stream import stream_stage
from sparc.server.artifact_reader import read_batch, prewarm_ids
from sparc.server.deps import state, session, get_open_store  # single shared instance; do NOT create another below

# ------------------------------------------------------------------
# Application & shared state
# ------------------------------------------------------------------

app = FastAPI(
    title="SPARC Server",
    version="1.0.7",
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

# ------------------------------------------------------------------
# Startup token — locks the sidecar to its hosting Tauri process.
# The token is injected via the SPARC_SERVER_TOKEN env var at spawn
# time. Every non-health request must supply it in X-SPARC-Token.
# When the env var is absent (e.g. plain CLI launch), auth is skipped
# so developers can still call the server manually.
# ------------------------------------------------------------------

_SERVER_TOKEN: str | None = os.environ.get("SPARC_SERVER_TOKEN") or None

# ------------------------------------------------------------------
# JWT / JWKS support (optional — enabled when SUPABASE_URL is set)
# ------------------------------------------------------------------

_SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL") or None
_REQUIRE_JWT: bool = os.environ.get("SPARC_REQUIRE_JWT", "").lower() in ("1", "true", "yes")

_jwks_lock = asyncio.Lock()
_jwks_cache: dict | None = None
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 86_400  # 24 h


async def _get_jwks() -> dict | None:
    """Fetch and cache Supabase JWKS. Returns None if SUPABASE_URL not set."""
    global _jwks_cache, _jwks_fetched_at
    if not _SUPABASE_URL:
        return None
    async with _jwks_lock:
        if _jwks_cache is not None and time.monotonic() - _jwks_fetched_at < _JWKS_TTL:
            return _jwks_cache
        try:
            url = f"{_SUPABASE_URL}/auth/v1/keys"
            raw = await asyncio.to_thread(
                lambda: urllib.request.urlopen(url, timeout=5).read()  # noqa: S310
            )
            _jwks_cache = json.loads(raw)
            _jwks_fetched_at = time.monotonic()
        except Exception:
            pass  # Return stale cache on error
        return _jwks_cache


async def _verify_jwt(token: str) -> bool:
    """Verify a Supabase-issued JWT against cached JWKS.

    Returns True if valid (or if SUPABASE_URL is unconfigured so verification
    is not possible).  Returns False on any crypto / expiry error.
    """
    try:
        from jose import jwt as jose_jwt  # lazy — only imported when JWT auth is active
        jwks = await _get_jwks()
        if jwks is None:
            return True  # SUPABASE_URL not configured — skip
        header = jose_jwt.get_unverified_header(token)
        kid = header.get("kid")
        keys = jwks.get("keys", [])
        key = next((k for k in keys if k.get("kid") == kid), None) or (keys[0] if keys else None)
        if key is None:
            return False
        jose_jwt.decode(token, key, algorithms=["RS256"], options={"verify_aud": False})
        return True
    except Exception:
        return False


class _TokenMiddleware(BaseHTTPMiddleware):
    """Reject requests that are missing or present the wrong X-SPARC-Token.

    Skipped for:
    - OPTIONS (CORS preflight — browser sends no custom headers)
    - /health  (UI polls this before the webview has the token)
    - /shutdown (Tauri calls this from Rust, not the webview)
    """

    # Paths exempt from token checking.
    _EXEMPT = {"/health", "/shutdown"}

    async def dispatch(self, request: Request, call_next):
        if _SERVER_TOKEN is None:
            # Running outside Tauri (dev / CLI) — skip auth.
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in self._EXEMPT:
            return await call_next(request)
        token = request.headers.get("x-sparc-token", "")
        if token != _SERVER_TOKEN:
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        # Optional JWT verification — only active when SPARC_REQUIRE_JWT=1
        if _REQUIRE_JWT:
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse({"detail": "JWT required"}, status_code=401)
            if not await _verify_jwt(auth[7:]):
                return JSONResponse({"detail": "invalid token"}, status_code=401)
        return await call_next(request)


app.add_middleware(_TokenMiddleware)

# ------------------------------------------------------------------
# Path-containment guard
# ------------------------------------------------------------------
# All endpoints that accept filesystem paths from the client call this
# helper.  It resolves symlinks and checks that the result sits inside
# the user's home directory, preventing path-traversal attacks even if
# an attacker somehow reaches the (token-protected) sidecar.
_HOME = Path.home().resolve()

def _resolve_safe(raw: str, *, allow_create: bool = False) -> Path:
    """Resolve *raw* to an absolute path and assert it is inside the
    user's home directory.

    Parameters
    ----------
    raw:
        The path string supplied by the client.
    allow_create:
        When *True* the resolved path need not exist yet (used for new
        project scaffolding).  The *parent* directory is still checked
        so that ``../../etc`` style attacks are blocked.

    Raises
    ------
    HTTPException(400)
        If the resolved path escapes the home directory.
    """
    resolved = Path(raw).resolve()
    # For paths that don't exist yet, check their closest existing ancestor.
    check = resolved if resolved.exists() else resolved.parent.resolve()
    try:
        check.relative_to(_HOME)
    except ValueError:
        raise HTTPException(
            400,
            "Path must be inside the user home directory.",
        )
    return resolved


TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


# ------------------------------------------------------------------
# Physics helpers
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Run registry helpers
# ------------------------------------------------------------------

def _attach_registry(config: dict) -> None:
    """Attach a RunRegistry to ``state``; runs migrate_from_disk for legacy runs.

    Failures here are non-fatal — endpoints fall back to disk paths.
    """
    try:
        from sparc.registry import RunRegistry
        from sparc.run.pipeline_paths import PipelinePaths
        paths = PipelinePaths.from_config(config)
        reg = RunRegistry(paths.output_dir, autoload=True)
        # If the manifest is empty (legacy / freshly-loaded run), import what's
        # already on disk so the frontend sees correct availability.
        try:
            reg.migrate_from_disk(paths)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: registry migration failed: {exc}")
        # Detach previous listener (if any) before swapping registries.
        prev = state.registry
        if prev is not None:
            try:
                prev.remove_register_listener(_on_artifact_registered)
            except Exception:  # noqa: BLE001
                pass
        try:
            reg.add_register_listener(_on_artifact_registered)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not attach artifact listener: {exc}")
        state.registry = reg
        # Clear stale cached results from the previous project.
        state.result_cache.clear()
        # Make this registry the process-global "active" one so pipeline
        # writers (`get_active_store()`) and any in-process consumers see it
        # for the lifetime of the loaded project. Mirrors `__main__.py`.
        try:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(reg)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not set active registry: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not attach RunRegistry: {exc}")
        state.registry = None
        try:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(None, force=True)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------
# Live artifact-event broadcasting
# ------------------------------------------------------------------

from sparc.server.deps import broadcaster as _broadcaster


def _on_artifact_registered(entry: Any) -> None:
    """Listener attached to ``RunRegistry.register_artifact``.

    Buffers an ``artifact_written`` event for ``/run/events`` polling and
    fans it out to any live ``/run/stream`` subscribers.
    """
    event = {
        "type": "artifact_written",
        "stage": str(getattr(entry, "stage", "")),
        "artifact_id": getattr(entry, "id", None),
        "kind": getattr(entry, "storage_kind", None),
        "format": getattr(entry, "format", None),
        "size_bytes": getattr(entry, "size_bytes", 0),
        "row_count": getattr(entry, "row_count", None),
        "content_hash": getattr(entry, "sha256", None) or getattr(entry, "blob_sha256", None),
        "written_at": getattr(entry, "written_at", None),
    }
    try:
        state.buffer_event(event)
    except Exception:  # noqa: BLE001
        pass
    # Invalidate cached results for this stage so the next read re-fetches.
    try:
        stage_str = str(getattr(entry, "stage", ""))
        if stage_str:
            state.result_cache.invalidate_stage(stage_str)
    except Exception:  # noqa: BLE001
        pass
    # Fan out to live ws subscribers via the broadcaster.
    _broadcaster.broadcast(event)



# ------------------------------------------------------------------
# Background result pre-warm
# ------------------------------------------------------------------

import threading as _threading

# Cancellation event for the currently-running pre-warm thread.
# Replaced each time a new project is loaded.
_prewarm_cancel: _threading.Event | None = None


def _prewarm_results(cancel: _threading.Event) -> None:
    """Populate the server-side ResultCache with all complete artifacts.

    Runs in a daemon thread so it doesn't block the /project/load response.
    Reads every complete (non-partial) artifact from the live manifest so
    that any frontend fetch hits the cache instead of blocking a request thread.
    """
    try:
        reg = state.registry
        if reg is None:
            return

        # Drive from the live manifest — no static catalog or consumer-tag filter.
        frontend_ids = prewarm_ids(reg.manifest)

        store = get_open_store()
        for stage, artifact_id in frontend_ids:
            if cancel.is_set():
                return
            # Skip if already cached.
            if state.result_cache.get(stage, artifact_id) is not None:
                continue
            try:
                if store.has(stage, artifact_id):
                    result = store.read_any(stage, artifact_id)
                    state.result_cache.set(stage, artifact_id, result)
            except Exception:
                # Missing or unreadable artifact — not an error during pre-warm.
                pass

    except Exception as exc:
        print(f"[prewarm] failed: {exc}")


def _start_prewarm() -> None:
    """Cancel any running pre-warm and start a fresh one for the loaded project."""
    global _prewarm_cancel
    if _prewarm_cancel is not None:
        _prewarm_cancel.set()
    cancel = _threading.Event()
    _prewarm_cancel = cancel
    t = _threading.Thread(
        target=_prewarm_results,
        args=(cancel,),
        daemon=True,
        name="sparc-prewarm",
    )
    t.start()


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
        _attach_registry(config)
        _load_data_into_state(config)
        _start_prewarm()
        print(f"Auto-loaded project: {resolved}")
    except Exception as exc:
        print(f"Warning: auto-load failed for {resolved}: {exc}")


# ------------------------------------------------------------------
# Route modules — imported and mounted here so they can be tested
# independently.  The inline routes below remain for features not yet
# migrated to a router module.
# ------------------------------------------------------------------
from sparc.server.routes.health import router as _health_router
from sparc.server.routes.project import router as _project_router
from sparc.server.routes.data import router as _data_router
from sparc.server.routes.results import router as _results_router
from sparc.server.routes.causal import router as _causal_router
from sparc.server.routes.physics import router as _physics_router
from sparc.server.routes.inference import router as _inference_router
from sparc.server.routes.api import router as _api_router
from sparc.server.routes.ai import router as _ai_router
from sparc.server.routes.reproduce import router as _reproduce_router
from sparc.server.routes.dag import router as _dag_router
from sparc.server.routes.report import router as _report_router
from sparc.server.routes.run import router as _run_router
from sparc.server.routes.scenarios import router as _scenarios_router
from sparc.server.routes.decision import router as _decision_router
from sparc.server.routes.artifacts import router as _artifacts_router
from sparc.server.routes.results_extended import router as _results_extended_router
from sparc.server.routes.misc import router as _misc_router
from sparc.server.routes.collect import router as _collect_router

app.include_router(_health_router)
app.include_router(_project_router)
app.include_router(_data_router)
app.include_router(_results_router)
app.include_router(_causal_router)
app.include_router(_physics_router)
app.include_router(_inference_router)
app.include_router(_api_router)
app.include_router(_ai_router)
app.include_router(_reproduce_router)
app.include_router(_dag_router)
app.include_router(_report_router)
app.include_router(_run_router)
app.include_router(_scenarios_router)
app.include_router(_decision_router)
app.include_router(_artifacts_router)
app.include_router(_results_extended_router)
app.include_router(_misc_router)
app.include_router(_collect_router)

# ------------------------------------------------------------------
# Health  (REMOVED — now served by routes.health)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------------
#
# Called by the Tauri host (Rust `kill_server`) before it hard-kills the
# sidecar so uvicorn can finish flushing in-flight requests and the
# `@app.on_event("shutdown")` handler can close DB connections / WebSocket
# subscribers cleanly. Without this, abruptly killing the process during a
# Stage write can corrupt artifacts.db.
#
# Localhost-only by design: the sidecar binds to 127.0.0.1 already, but we
# double-check the client host as a safety net in case the bind address is
# ever changed.

# ------------------------------------------------------------------
# Run control
# (/run/cancel is now served by routes.run via include_router above)
# ------------------------------------------------------------------

# (POST /shutdown is now served by routes.misc via include_router above)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Clean up shared resources before the process exits.

    Best-effort: anything that fails here is logged and swallowed so we
    don't block uvicorn's exit path.
    """
    # Broadcast server_shutdown so all WebSocket consumers can clean up.
    try:
        _broadcaster.broadcast({"type": "server_shutdown"})
    except Exception as exc:
        print(f"Shutdown: broadcaster drain failed: {exc}")

    # Release the cached scenario GeoDataFrame(s) so the file handle is
    # closed before PyInstaller's `_MEI` cleanup runs.
    try:
        _scenario_gpkg_cache.clear()
    except Exception:
        pass

    # Close any artifact-store / SQLite connection the registry may hold.
    try:
        registry = getattr(state, "registry", None)
        close = getattr(registry, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        print(f"Shutdown: registry close failed: {exc}")

    print("SPARC server shutdown complete")


# (GET /api/hardware, GET+PUT /api/preferences, POST /api/hardware/validate
# are now served by routes.api via include_router above)


# (GET /debug/paths is now served by routes.misc via include_router above)


# ------------------------------------------------------------------
# AI proxy
# (/ai/key GET+PUT+DELETE and /ai/chat are now served by routes.ai via
# include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Data endpoints
# (NOTE: /project/validate, /project/init, /project/create,
# /project/templates, and /project/config GET+PUT are served by
# _project_router (sparc.server.routes.project), registered above.
# /data/summary, /data/preview, /data/histogram are served by
# _data_router (sparc.server.routes.data), registered above.)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Data validation, versioning, preprocessing, processing
# (/data/validate, /data/versions, /data/select_version, /data/preprocess,
# /crs/distortion, /data/fishnet, /data/zonal_stats, /data/prepare are now
# served by routes.data via include_router above)
# ------------------------------------------------------------------

# (GET /data/geojson is now served by routes.data via include_router above)

# Extensions accepted by /data/upload.Anything else is rejected before
# touching the filesystem.
_UPLOAD_ALLOWED_SUFFIXES: frozenset[str] = frozenset({
    ".csv", ".parquet",                      # tabular data
    ".tif", ".tiff",                         # GeoTIFF rasters
    ".shp", ".shx", ".dbf", ".prj", ".cpg", # Shapefile family
    ".gpkg",                                 # GeoPackage
    ".geojson",                              # GeoJSON
})

# Hard cap: 500 MB per upload.  Rasters can be large; we keep this generous
# but not unbounded.
_UPLOAD_MAX_BYTES: int = 500 * 1024 * 1024  # 500 MB


@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Accept a CSV, raster (.tif/.tiff), or spatial file (.shp/.gpkg/.geojson) upload."""
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    # ── Extension allowlist ──────────────────────────────────────────────────
    safe_name = Path(file.filename).name  # strip any directory components
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"File type '{suffix or '(none)'}' is not accepted. "
            f"Allowed: {', '.join(sorted(_UPLOAD_ALLOWED_SUFFIXES))}",
        )

    # ── Size cap ─────────────────────────────────────────────────────────────
    content = await file.read(_UPLOAD_MAX_BYTES + 1)
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            413,
            f"Upload exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB limit.",
        )

    project_dir = Path(state.project_config["paths"]["project_root"])
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / safe_name
    with open(dest, "wb") as f:
        f.write(content)

    result: dict = {"status": "uploaded", "path": str(dest), "file_type": suffix}

    # CSV / Parquet — load as primary data
    if suffix in (".csv", ".parquet"):
        _set_data_path(str(dest))
        _load_data_into_state(state.project_config)
        # Sync into session so routes/data.py endpoints see the new data.
        session.data = state.data
        session.data_summary = state.data_summary
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


# (GET /data/files, POST /data/select are now served by routes.data via include_router above)


# ------------------------------------------------------------------
# Session log, pipeline streaming WebSocket routes
# (/run/log, /run/execute, /run/artifacts, /run/stream are now served
# by routes.run via include_router above)
# (/data/preprocess is now served by routes.data via include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Run control, run events, run comparison
# (/run/cancel, /run/log, /run/events, /run/execute WS,
# /run/artifacts WS, /run/stream WS, /runs/discover, /runs/diff
# are now served by routes.run via include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Reproducibility
# (/reproduce/provenance, /reproduce/freeze, /reproduce/load,
# /reproduce/verify are now served by routes.reproduce via include_router)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Scenario library + chain rollout
# (moved to routes.scenarios via include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Standalone snapshot HTML and audience reports
# (/report/standalone and /report/audience are now served by
# routes.report via include_router above)
# ------------------------------------------------------------------


# (GET /context/layers, GET /gwen/status, POST /gwen/approve, DELETE /gwen/approve,
#  GET /insights/headline are now served by routes.misc via include_router above)

# (GET /panels/availability and its helpers _PANEL_SPECS, _STAGE_HINTS,
# _STAGE_LABELS, _spec_matches are now served by routes.misc)
# (/data/fishnet, /data/zonal_stats, /data/prepare are now served by
# routes.data via include_router above)


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
        # Invalidate the geojson cache so the next request re-reprojects with fresh data.
        state.result_cache.invalidate_stage("data_geojson")
        _compute_summary(state)
        # Keep session in sync so routes/data.py endpoints see the loaded data.
        session.data = state.data
        session.data_summary = state.data_summary
    except Exception as exc:
        print(f"Warning: data pre-load failed: {exc}")


def _compute_summary(st) -> dict:
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


# (Artifact Phase-5 download/export/list routes moved to routes.artifacts via include_router above)


# ===========================================================================
# Data Collection endpoints  (/collect/*)
# (Moved to routes.collect via include_router above)
# ===========================================================================


# ---------------------------------------------------------------------------
# SpatialANP inference endpoints
# (/inference/zero-shot and /inference/few-shot are now served by
# routes.inference via include_router above)
# ---------------------------------------------------------------------------
