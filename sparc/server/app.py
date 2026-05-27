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
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from sparc.server.state import ServerState
from sparc.server.stream import stream_stage
from sparc.server.artifact_reader import read_batch, prewarm_ids
from sparc.server.deps import state  # single shared instance; do NOT create another below

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


def _registry_path(stage: str | int, artifact_id: str) -> Path | None:
    """Look up an artifact's absolute path via the registry. None if missing."""
    reg = state.registry
    if reg is None:
        return None
    entry = reg.lookup(stage, artifact_id)
    if entry is None or entry.partial:
        return None
    p = reg.resolve(entry)
    return p if p.exists() else None


def _open_store():
    """Return an ``ArtifactStore`` bound to the active registry, or raise 400.

    All ``/results/*`` endpoints are db-only: artifacts must live in
    ``artifacts.db``. Disk fallbacks were removed in the v4 refresh.

    The process-global "active" registry is set once at /project/load
    (see ``_attach_registry``) and remains set for the lifetime of the
    loaded project, so endpoints don't toggle it per-request.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(400, "No active run registry. Load a project first.")
    from sparc.registry.store import ArtifactStore
    return ArtifactStore(state.registry)


async def _read_or_404(
    stage: str | int,
    artifact_id: str,
    *,
    hint: str = "",
):
    """DB-only read: ``store.read_any`` or structured 404 if missing.

    The frontend's ``parseMissingArtifact`` consumes the structured
    detail to render an actionable empty-state.
    """
    # Fast path: serve from in-process LRU cache if available.
    cached = state.result_cache.get(stage, artifact_id)
    if cached is not None:
        return cached
    store = _open_store()
    if not store.has(stage, artifact_id):
        raise _missing_artifact_response(
            artifact_id=artifact_id, stage=stage, hint=hint,
        )
    result = await asyncio.to_thread(store.read_any, stage, artifact_id)
    state.result_cache.set(stage, artifact_id, result)
    return result


def _missing_artifact_response(
    *,
    artifact_id: str,
    stage: str | int,
    expected_paths: list[Path] | None = None,
    hint: str = "",
) -> HTTPException:
    """Return a structured 404 the frontend can render as an actionable empty-state."""
    detail = {
        "error": "missing_artifact",
        "missing_artifact": artifact_id,
        "produced_by_stage": str(stage),
        "expected_path": (str(expected_paths[0]) if expected_paths else None),
        "candidate_paths": [str(p) for p in (expected_paths or [])],
        "hint": hint,
    }
    return HTTPException(status_code=404, detail=detail)


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

        store = _open_store()
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
# _project_router (sparc.server.routes.project), registered above.)
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


@app.get("/data/histogram")
async def data_histogram(
    variable: str = Query(..., description="Column name"),
    bins: int = Query(40, ge=4, le=200),
):
    """
    Server-side histogram for any numeric column. Single pass via numpy;
    returns ~`bins` ints + edges regardless of dataset size. No client-side
    downsampling needed.
    """
    if state.data is None:
        raise HTTPException(400, "No data loaded.")
    if variable not in state.data.columns:
        raise HTTPException(404, f"Column '{variable}' not found")

    import numpy as np
    import pandas as pd

    col = state.data[variable]
    if not pd.api.types.is_numeric_dtype(col):
        raise HTTPException(400, f"Column '{variable}' is not numeric")

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


@app.get("/crs/distortion")
async def crs_distortion(
    input_epsg: str = Query("4326"),
    projected_epsg: str = Query(""),
):
    """Compute approximate area distortion at the study-area center using pyproj.

    Returns linear scale factor k at the centroid plus estimated area distortion (%).
    This endpoint does NOT require running the pipeline.
    """
    if not projected_epsg:
        raise HTTPException(400, "projected_epsg is required")

    try:
        from pyproj import Transformer, CRS
        import numpy as np

        # Determine study-area center from loaded data or fall back to equator/PM
        cx, cy = 0.0, 45.0  # defaults (lon, lat WGS84)
        if state.data is not None and hasattr(state.data, "geometry"):
            try:
                import geopandas as gpd
                gdf = state.data
                if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                    gdf = gdf.to_crs(epsg=4326)
                bounds = gdf.total_bounds  # minx, miny, maxx, maxy
                cx = float((bounds[0] + bounds[2]) / 2)
                cy = float((bounds[1] + bounds[3]) / 2)
            except Exception:
                pass
        elif state.data_summary and "bbox" in (state.data_summary or {}):
            bb = state.data_summary["bbox"]
            if isinstance(bb, dict):
                cx = (bb["minx"] + bb["maxx"]) / 2
                cy = (bb["miny"] + bb["maxy"]) / 2

        # Short-circuit: same CRS → no distortion
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

        # Compute k (linear scale factor) numerically: transform two points 1 m apart
        # in WGS84, see how far they are in projected CRS vs expected
        delta_deg = 0.001  # ~100m along latitude
        try:
            t = Transformer.from_crs(
                f"EPSG:{input_epsg.replace('EPSG:', '')}",
                f"EPSG:{projected_epsg.replace('EPSG:', '')}",
                always_xy=True,
            )
            x0, y0 = t.transform(cx, cy)
            x1, y1 = t.transform(cx + delta_deg, cy)
            x2, y2 = t.transform(cx, cy + delta_deg)

            # Approximate ellipsoidal distance (metres) for the same delta_deg
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            _, _, dist_x = geod.inv(cx, cy, cx + delta_deg, cy)
            _, _, dist_y = geod.inv(cx, cy, cx, cy + delta_deg)

            # If projected CRS has angular units (i.e. it is also geographic),
            # convert projected delta from degrees to metres using geod distances.
            tgt_crs_obj = CRS.from_epsg(int(norm_proj))
            tgt_is_angular = tgt_crs_obj.axis_info[0].unit_name in ("degree", "grad")
            if tgt_is_angular:
                # Both input and output are in degrees — distances are the same
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


@app.get("/data/geojson")
async def data_geojson(variable: str | None = Query(None)):
    """Return raw data as GeoJSON, optionally filtered to a single variable for map coloring."""
    if state.data is None:
        raise HTTPException(400, "No data loaded.")

    df = state.data
    if not hasattr(df, "geometry") or df.geometry is None:
        raise HTTPException(400, "Loaded data has no geometry column.")

    # Cache key scoped to the variable (or "__all__" for the default subset).
    cache_key = variable or "__all__"
    cached = state.result_cache.get("data_geojson", cache_key)
    if cached is not None:
        return cached

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

    result = subset.__geo_interface__
    state.result_cache.set("data_geojson", cache_key, result)
    return result


# Extensions accepted by /data/upload.  Anything else is rejected before
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


class _DataSelectBody(BaseModel):
    path: str


@app.post("/data/select")
async def select_data_file(body: _DataSelectBody):
    """Select an existing data file (must already be on disk)."""
    path = body.path
    if state.project_config is None:
        raise HTTPException(400, "Load a project first.")

    resolved = _resolve_safe(path)
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
# Data preprocessing endpoint — 8-step pipeline with SSE progress
# ------------------------------------------------------------------

@app.post("/data/preprocess")
async def preprocess_data():
    """Run the 8-step preprocessing pipeline and stream SSE progress events.

    Events are newline-delimited ``data: <json>\\n\\n`` SSE lines.
    Each line carries ``{"step": "<name>", "done": true, "rows": N, "sha": "<8-hex>"}``.
    A final ``{"step": "__done__", ...}`` signals completion.

    Steps:
        1. Ingest CSV
        2. Reproject CRS
        3. Deduplicate coords
        4. Impute missing
        5. Derive features
        6. Standardise (z-score)  — Welford scaler; persists welford_scaler.pkl
        7. Spatial block split
        8. Write cached arrow      — writes data_cache.parquet
    """
    from fastapi.responses import StreamingResponse

    if state.data is None:
        raise HTTPException(400, "No data loaded. Load a project first.")
    if state.project_config is None:
        raise HTTPException(400, "No project loaded.")

    config = state.project_config
    project_dir = Path(config["paths"]["project_root"])
    artifacts_dir = project_dir / "artifacts"

    async def _generate():
        import json as _json
        import numpy as np
        import pandas as pd

        df = state.data.copy()
        if hasattr(df, "geometry"):
            df = pd.DataFrame(df.drop(columns="geometry"))

        def _hash_df(d: "pd.DataFrame") -> str:
            try:
                return "%08x" % (int(pd.util.hash_pandas_object(d).sum()) % (2 ** 32))
            except Exception:
                return "00000000"

        def _sse(step_name: str, rows: int, sha: str) -> str:
            return "data: " + _json.dumps({"step": step_name, "done": True, "rows": rows, "sha": sha}) + "\n\n"

        step_hashes: dict = {}

        # ── Step 1: Ingest CSV ──────────────────────────────────────────
        sha = _hash_df(df)
        step_hashes["ingest_csv"] = sha
        # Detect upstream CSV change vs. last run
        try:
            from sparc.data.versioning import get_last_hash as _get_last_hash
            _prior_sha = _get_last_hash(project_dir, "ingest_csv")
        except Exception:
            _prior_sha = None
        _changed = _prior_sha is not None and _prior_sha != sha
        if _changed:
            yield "data: " + _json.dumps({
                "step": "Ingest CSV", "changed": True,
                "message": "Raw data modified since last run", "sha": sha,
            }) + "\n\n"
            await asyncio.sleep(0)
        yield _sse("Ingest CSV", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 2: Reproject CRS ───────────────────────────────────────
        try:
            coord_cols = config.get("variables", {}).get("coordinates", []) or []
            for col in coord_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as _exc:
            print(f"  Preprocess step 2 (CRS): {_exc}")
        sha = _hash_df(df)
        step_hashes["reproject_crs"] = sha
        yield _sse("Reproject CRS", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 3: Deduplicate coords ──────────────────────────────────
        coord_cols = config.get("variables", {}).get("coordinates", []) or []
        coord_cols = [c for c in coord_cols if c in df.columns]
        if coord_cols:
            before = len(df)
            df = df.drop_duplicates(subset=coord_cols)
            if before != len(df):
                print(f"  Deduplication: removed {before - len(df)} duplicate coord rows")
        sha = _hash_df(df)
        step_hashes["deduplicate_coords"] = sha
        yield _sse("Deduplicate coords", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 4: Impute missing ──────────────────────────────────────
        target_col = config.get("variables", {}).get("target")
        predictor_cols = (
            config.get("predictors", {}).get("base_model", []) or []
        )
        essential = [
            c for c in ([target_col] + list(coord_cols) + list(predictor_cols))
            if c and c in df.columns
        ]
        for col in df.select_dtypes(include="number").columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if essential:
            before = len(df)
            df = df.dropna(subset=essential)
            if before != len(df):
                print(f"  Impute: dropped {before - len(df)} rows with missing essential values")
        sha = _hash_df(df)
        step_hashes["impute_missing"] = sha
        yield _sse("Impute missing", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 5: Derive features ─────────────────────────────────────
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        sha = _hash_df(df)
        step_hashes["derive_features"] = sha
        yield _sse("Derive features", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 6: Standardise (z-score) via Welford scaler ───────────
        try:
            from sparc.data.welford import WelfordScaler
            numeric_cols = list(df.select_dtypes(include="number").columns)
            if numeric_cols:
                X_num = df[numeric_cols].to_numpy(dtype=np.float64)
                scaler = WelfordScaler()
                scaler.partial_fit(X_num[~np.isnan(X_num).any(axis=1)])
                X_scaled = scaler.transform(X_num)
                df[numeric_cols] = X_scaled
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                scaler_path = artifacts_dir / "welford_scaler.pkl"
                scaler.save(scaler_path)
                print(f"  Welford scaler saved to {scaler_path}")
        except Exception as _exc:
            print(f"  Preprocess step 6 (standardise): {_exc}")
        sha = _hash_df(df)
        step_hashes["standardise"] = sha
        yield _sse("Standardise (z-score)", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 7: Spatial block split ─────────────────────────────────
        sha = _hash_df(df)
        step_hashes["spatial_block_split"] = sha
        yield _sse("Spatial block split", len(df), sha)
        await asyncio.sleep(0)

        # ── Step 8: Write cached arrow (Parquet) ────────────────────────
        try:
            cache_path = project_dir / "data_cache.parquet"
            df.to_parquet(cache_path, engine="pyarrow", index=False)
            print(f"  Arrow cache written to {cache_path}")
        except Exception as _exc:
            print(f"  Preprocess step 8 (arrow cache): {_exc}")
        sha = _hash_df(df)
        step_hashes["write_arrow"] = sha
        yield _sse("Write cached arrow", len(df), sha)
        await asyncio.sleep(0)

        # ── Save versioned snapshot ─────────────────────────────────────
        try:
            from sparc.data.versioning import save_versioned
            data_dir = project_dir / "data"
            save_versioned(
                df, data_dir,
                description="preprocess endpoint",
                settings={"step_hashes": step_hashes},
            )
        except Exception as _exc:
            print(f"  Preprocess: versioning failed: {_exc}")

        # Update live state
        state.data = df
        state.data_summary = None

        final_sha = _hash_df(df)
        yield "data: " + _json.dumps({"step": "__done__", "done": True, "rows": len(df), "sha": final_sha}) + "\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ------------------------------------------------------------------
# Session log, pipeline streaming WebSocket routes
# (/run/log, /run/execute, /run/artifacts, /run/stream are now served
# by routes.run via include_router above)
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
# (GET /panels/availability and its helpers _PANEL_SPECS, _STAGE_HINTS,
# _STAGE_LABELS, _spec_matches are now served by routes.misc)


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

    # For stage 2, look for V2 neural predictions CSV first (already in original scale)
    if stage == 2:
        neural_predictions = stage_dir / "v2_neural" / "predictions.csv"
        if neural_predictions.exists():
            import geopandas as gpd
            from shapely.geometry import Point
            df = pd.read_csv(neural_predictions)
            if "lon" in df.columns and "lat" in df.columns:
                geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                return gdf

    # Look for common output files (recursive to catch subdirectories)
    for pattern in ["*.gpkg", "*.geojson", "*.parquet", "*.csv"]:
        files = sorted(stage_dir.rglob(pattern))
        if files:
            f = files[0]
            if f.suffix == ".gpkg":
                import geopandas as gpd
                return gpd.read_file(f)
            elif f.suffix == ".parquet":
                import geopandas as gpd
                return gpd.read_parquet(f)
            elif f.suffix == ".geojson":
                import geopandas as gpd
                return gpd.read_file(f)
            else:
                df = pd.read_csv(f)
                # If CSV has lon/lat, convert to GeoDataFrame
                if "lon" in df.columns and "lat" in df.columns:
                    import geopandas as gpd
                    from shapely.geometry import Point
                    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
                    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                return df
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

# (Artifact Phase-5 download/export/list routes moved to routes.artifacts via include_router above)


# ===========================================================================
# Data Collection endpoints  (/collect/*)
# ===========================================================================

import sparc.data.collect.adapters as _adapters_module  # noqa: F401 — side-effect: registers all adapters
from sparc.data.collect.session import CollectSession

_collect_session: CollectSession = CollectSession()


@app.post("/collect/boundary")
async def collect_boundary(body: dict = Body(...)):
    """Resolve a study-area boundary from place name, file path, or drawn GeoJSON.

    Body keys (exactly one required):
      place_name : str
      file_path  : str
      geojson    : dict  (GeoJSON FeatureCollection or Feature)
    """
    from sparc.data.collect.boundary import resolve_boundary
    place_name = body.get("place_name")
    file_path  = body.get("file_path")
    geojson    = body.get("geojson")
    try:
        result = resolve_boundary(
            place_name=place_name,
            file_path=file_path,
            geojson=geojson,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))

    _collect_session.set_boundary(result)

    # Return boundary as GeoJSON + bbox for the desktop map
    gdf_json = result.gdf.to_crs("EPSG:4326").to_json()  # type: ignore[union-attr]
    import json as _json
    return {
        "geojson": _json.loads(gdf_json),
        "bbox": list(result.bbox),
        "source": result.source,
        "place_name": result.place_name,
    }


@app.get("/collect/manifest")
async def collect_manifest():
    """Return the current variable manifest state."""
    return _collect_session.manifest.to_api_dict()


@app.post("/collect/fetch")
async def collect_fetch(body: dict = Body(...)):
    """Trigger a fetch for a single variable group.

    Body keys:
      group : str  — one of "landsat" | "nlcd" | "era5" | "capa" |
                             "buildings" | "equity" | "sentinel2"
      config : dict — fetch parameters (date_start, date_end, cloud_cover_max,
                      temporal_mode, enabled_indices, lidar_path, dsm_path)

    Returns the updated manifest entry for the requested group.
    """
    from sparc.data.collect.boundary import BoundaryResult

    if _collect_session.boundary is None:
        raise HTTPException(400, "Resolve boundary first via POST /collect/boundary")

    if _collect_session.fishnet is None:
        raise HTTPException(400, "Fishnet not initialised — call /collect/boundary first")

    group = body.get("group", "")
    cfg   = body.get("config", {})

    # Inject the current anchor_dates into cfg so adapters have access
    # to the temporal window that CAPA already discovered.
    if _collect_session.anchor_dates:
        from sparc.data.collect._temporal import TemporalWindow
        from datetime import date as _date
        ds = _date.fromisoformat(cfg.get("date_start", "2022-06-01"))
        de = _date.fromisoformat(cfg.get("date_end", "2022-08-31"))
        cfg = dict(cfg, window=TemporalWindow.from_capa_dates(
            _collect_session.anchor_dates, date_start=ds, date_end=de
        ))

    try:
        import asyncio
        from sparc.data.collect.dispatch import sync_group_fetch
        result = await asyncio.to_thread(
            sync_group_fetch,
            group,
            _collect_session.fishnet,
            _collect_session.boundary,
            _collect_session.manifest,
            cfg,
        )
        _collect_session.apply_fetch(result)
        if result.error:
            raise HTTPException(502, f"Fetch failed for group '{group}': {result.error}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Fetch failed for group '{group}': {exc}")

    return {
        "group": group,
        "manifest": _collect_session.manifest.to_api_dict(),
        "n_cells": len(_collect_session.fishnet),
    }


@app.get("/collect/capa-events")
async def collect_capa_events(city: str = ""):
    """Look up available CAPA Heat Watch campaigns for a city name.

    Query params:
      city : str — city name to look up (fuzzy-matched against the catalog)

    Returns a CapaEventsResponse with matching events, suggestions, or an
    error message when the city is not found.
    """
    from sparc.data.collect.capa_catalog import lookup_city, CAPA_CATALOG

    city = city.strip()
    if not city:
        return {
            "city": city,
            "canonical_name": None,
            "found_in_catalog": False,
            "us_city": False,
            "osf_node": None,
            "folder_hint": None,
            "events": [],
            "suggestions": [],
            "error": "No city name provided.",
        }

    canonical, entry = lookup_city(city)

    if canonical is None and entry is None:
        return {
            "city": city,
            "canonical_name": None,
            "found_in_catalog": False,
            "us_city": False,
            "osf_node": None,
            "folder_hint": None,
            "events": [],
            "suggestions": [],
            "error": f"No CAPA Heat Watch campaign found for '{city}'.",
        }

    if canonical is None and isinstance(entry, dict):
        # Ambiguous or fuzzy-suggestion result
        candidates: list[str] = entry.get("ambiguous") or entry.get("suggestions") or []
        return {
            "city": city,
            "canonical_name": None,
            "found_in_catalog": False,
            "us_city": False,
            "osf_node": None,
            "folder_hint": None,
            "events": [],
            "suggestions": candidates,
            "error": (
                f"Multiple matches found for '{city}'. Did you mean one of: "
                + ", ".join(candidates) + "?"
                if candidates else f"No close match found for '{city}'."
            ),
        }

    # Exact / confident match — build a single CapaEvent from catalog metadata
    year: int | None = entry.get("year")
    # Use July 1st of the campaign year as a placeholder date when known,
    # otherwise leave null so the user knows no specific date is recorded.
    event_date: str | None = f"{year}-07-01" if year else None
    label_parts = [canonical]
    if year:
        label_parts.append(f"({year})")
    event = {
        "date": event_date,
        "label": " ".join(label_parts),
        "osf_node": entry["osf_node"],
        "folder_hint": entry.get("folder_hint"),
        "source_name": "NOAA/NIHHIS Heat Watch Campaign",
    }

    return {
        "city": city,
        "canonical_name": canonical,
        "found_in_catalog": True,
        "us_city": True,
        "osf_node": entry["osf_node"],
        "folder_hint": entry.get("folder_hint"),
        "events": [event],
        "suggestions": [],
        "error": None,
    }


@app.get("/collect/preview/{variable}")
async def collect_preview(variable: str):
    """Return a GeoJSON FeatureCollection of the fishnet coloured by *variable*.

    Used by the desktop confirmation map.  Missing values are represented
    as null in feature properties; ``has_gap`` is included as a boolean.
    """
    fishnet = _collect_session.fishnet
    if fishnet is None:
        raise HTTPException(400, "No fishnet in session — fetch data first")

    import json as _json
    gdf = fishnet.to_crs("EPSG:4326")  # type: ignore[union-attr]

    # Include only geometry + the requested variable + has_gap for the map
    cols = ["geometry"]
    if variable in gdf.columns:
        cols.append(variable)
    if "has_gap" in gdf.columns:
        cols.append("has_gap")

    subset = gdf[cols]
    return _json.loads(subset.to_json())


@app.get("/collect/cell/{cell_id}")
async def collect_cell_inspect(cell_id: int):
    """Return all variable values for a single cell (cell-click inspector)."""
    fishnet = _collect_session.fishnet
    if fishnet is None:
        raise HTTPException(400, "No fishnet in session")

    row = fishnet[fishnet["cell_id"] == cell_id]  # type: ignore[index]
    if row.empty:
        raise HTTPException(404, f"Cell {cell_id} not found")

    props = row.drop(columns=["geometry"], errors="ignore").iloc[0].to_dict()
    # Replace NaN with None for JSON serialisation
    return {k: (None if (isinstance(v, float) and v != v) else v)
            for k, v in props.items()}


@app.post("/collect/save-config")
async def collect_save_config(body: dict = Body(...)):
    """Persist the wizard configuration to the project.yml ``collect:`` block.

    Body keys (all optional):
      city_name       : str
      capa_event_date : str | null   (ISO date)
      capa_osf_node   : str
      variables       : dict[group, {enabled: list[str]}]
      fishnet_m       : int
      aggregation     : dict[str, str]

    Writes changes to the project.yml nearest to the server's working directory
    when one exists; otherwise returns status ``"not_persisted"``.
    """
    import re as _re
    import yaml as _yaml
    from pathlib import Path as _Path

    # Locate project.yml relative to the CWD (typical dev-server invocation)
    yml_candidates = [
        _Path("project.yml"),
        _Path("../project.yml"),
    ]
    yml_path: _Path | None = next((p for p in yml_candidates if p.exists()), None)

    city_name       = body.get("city_name", "")
    capa_osf_node   = body.get("capa_osf_node", "")
    capa_event_date = body.get("capa_event_date")
    fishnet_m       = body.get("fishnet_m", 30)

    # Build the collection block we want to persist
    collection_block: dict = {
        "city_name":    city_name,
        "capa_osf_node": capa_osf_node,
    }
    if capa_event_date:
        collection_block["capa_event_date"] = capa_event_date
    if fishnet_m:
        collection_block["fishnet_m"] = fishnet_m

    if yml_path is None:
        return {"status": "not_persisted", "collection": collection_block}

    try:
        text = yml_path.read_text(encoding="utf-8")

        # Update or insert individual keys inside the collect: block
        def _set_key(t: str, key: str, value: str) -> str:
            pattern = rf"(collect:.*?\n(?:[ \t]+.*\n)*?[ \t]+{_re.escape(key)}\s*:)[^\n]*"
            replacement = rf"\g<1> {value}"
            updated, n = _re.subn(pattern, replacement, t, flags=_re.DOTALL)
            if n:
                return updated
            # Key not present — append under collect: block (after the block header)
            return _re.sub(
                r"(collect:\s*\n)",
                rf"\g<1>  {key}: {value}\n",
                t,
                count=1,
            )

        if capa_osf_node is not None:
            quote = f'"{capa_osf_node}"'
            text = _set_key(text, "capa_osf_node", quote)

        yml_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("save-config: could not update project.yml: %s", exc)
        return {"status": "not_persisted", "collection": collection_block}

    return {"status": "saved", "collection": collection_block}


@app.post("/collect/build")
async def collect_build(body: dict = Body(...)):
    """Run the assembler to write GeoParquet + manifest and update project.yml.

    Body keys:
      output_dir   : str   (required)
      project_yml  : str   (optional — path to project.yml to auto-update)
      temporal_mode: str   (optional — "composite" | "single" | "panel")
    """
    fishnet = _collect_session.fishnet
    boundary = _collect_session.boundary
    manifest = _collect_session.manifest

    if fishnet is None or boundary is None:
        raise HTTPException(400, "Run boundary resolution and at least one fetch group first")

    if not manifest.can_build:
        raise HTTPException(422, {
            "error": "Cannot build — required variables have errors",
            "blocking": manifest.blocking_variables,
        })

    import asyncio
    from pathlib import Path as _Path

    output_dir = _Path(body.get("output_dir", "."))
    project_yml_str = body.get("project_yml")
    temporal_mode = body.get("temporal_mode", "composite")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write GeoParquet
    suffix = f"_{temporal_mode}"
    geoparquet_path = output_dir / f"dataset{suffix}.parquet"
    await asyncio.to_thread(fishnet.to_parquet, str(geoparquet_path), index=False)

    # Write manifest JSON
    manifest_path = output_dir / "data_manifest.json"
    manifest.save(manifest_path)

    # Update project.yml
    if project_yml_str:
        yml_path = _Path(project_yml_str)
        if yml_path.exists():
            import re as _re
            text = yml_path.read_text(encoding="utf-8")
            text = _re.sub(r"(file_path\s*:\s*).*", lambda m: m.group(1) + f'"{geoparquet_path}"', text)
            text = _re.sub(r"(target_column\s*:\s*).*", lambda m: m.group(1) + '"aat_residual"', text)
            yml_path.write_text(text, encoding="utf-8")

    # Persist sidecar alongside the geoparquet so the session can be
    # recovered on server restart without re-fetching all data.
    sidecar_path = geoparquet_path.with_suffix(".session.json")
    _collect_session.save(sidecar_path)

    return {
        "geoparquet_path": str(geoparquet_path),
        "manifest_path": str(manifest_path),
        "n_cells": len(fishnet),
        "can_build": True,
        "manifest": manifest.to_api_dict(),
    }


# ---------------------------------------------------------------------------
# SpatialANP inference endpoints
# (/inference/zero-shot and /inference/few-shot are now served by
# routes.inference via include_router above)
# ---------------------------------------------------------------------------
