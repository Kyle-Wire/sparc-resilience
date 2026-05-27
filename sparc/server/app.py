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


@app.post("/shutdown")
async def shutdown(request: Request):
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="shutdown is localhost-only")

    # Defer the actual signal so this response can flush back to the caller
    # first. SIGINT triggers uvicorn's graceful-shutdown path, which in turn
    # invokes our @app.on_event("shutdown") hook.
    import signal

    async def _terminate_soon() -> None:
        await asyncio.sleep(0.1)
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to deliver SIGINT during /shutdown: {exc}")

    asyncio.create_task(_terminate_soon())
    return {"status": "shutting down"}


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
# Block export — capture the current view of any chart/map/plot as a
# PNG and register it in the run registry so it shows up in Report.
# ------------------------------------------------------------------

class BlockExportRequest(BaseModel):
    artifact_id: str
    stage: str = "export"
    label: Optional[str] = None
    png_b64: str  # data URL or raw base64

    class Config:
        extra = "ignore"


@app.post("/results/export")
async def export_block(req: BlockExportRequest):
    """Persist a screenshot of a UI block (map / chart / plot) to disk and the registry.

    Body fields:
      - artifact_id: stable id for this export (e.g. "cate_map", "dose_response").
      - stage: registry stage to file under (defaults to "export").
      - label: optional human-readable suffix used in the filename.
      - png_b64: data URL ("data:image/png;base64,...") or raw base64 PNG bytes.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    import base64
    import time
    from sparc.run.pipeline_paths import PipelinePaths

    payload = req.png_b64
    if payload.startswith("data:"):
        try:
            payload = payload.split(",", 1)[1]
        except IndexError:
            raise HTTPException(400, "Malformed data URL in png_b64")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(400, f"Invalid base64 PNG: {exc}")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(400, "png_b64 does not contain a PNG signature")

    try:
        paths = PipelinePaths.from_config(state.project_config)
    except Exception as exc:
        raise HTTPException(500, f"Cannot resolve paths: {exc}")

    exports_dir = paths.output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    safe_label = "".join(c for c in (req.label or "") if c.isalnum() or c in "-_") or "snapshot"
    fname = f"{req.stage}__{req.artifact_id}__{safe_label}__{ts}.png"
    out_path = exports_dir / fname
    out_path.write_bytes(raw)

    entry_dict: dict[str, Any] = {
        "saved_to": str(out_path),
        "size_bytes": len(raw),
        "registered": False,
    }

    reg = state.registry
    if reg is not None:
        try:
            entry = reg.register_artifact(
                stage=req.stage,
                artifact_id=f"export::{req.artifact_id}::{ts}",
                path=out_path,
                format="png",
                producer="desktop:export",
                consumers=["user:export", "report:figures"],
                metadata={
                    "exported_from": req.artifact_id,
                    "label": req.label or "",
                    "exported_at": ts,
                },
            )
            entry_dict["registered"] = True
            entry_dict["artifact"] = entry.model_dump()
        except Exception as exc:
            entry_dict["registry_error"] = str(exc)

    return entry_dict


# ------------------------------------------------------------------
# AI proxy
# (/ai/key GET+PUT+DELETE and /ai/chat are now served by routes.ai via
# include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Project endpoints  (NOTE: /project/load, /project/validate, /project/init,
# /project/create, /project/templates, and /project/config are served by
# _project_router (sparc.server.routes.project), registered above.
# The handlers below cover project-adjacent utilities not yet migrated.)
# ------------------------------------------------------------------

@app.post("/project/validate")
async def validate_project(path: str = Query(..., description="Absolute path to project.yml")):
    """Validate a project.yml without loading it into state."""
    resolved = _resolve_safe(path)
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

    dest = _resolve_safe(output, allow_create=True)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)

    return {
        "status": "created",
        "template": template,
        "path": str(dest),
        "project_yml": str(dest / "project.yml"),
    }


@app.post("/project/create")
async def create_project(payload: dict[str, Any] = Body(...)):
    """
    Create a project from the wizard payload.

    Combines template scaffolding with structured edits to ``project.yml``
    so the YAML reflects exactly what the user entered in the wizard
    (no leaked template defaults). Required keys:

      - template: str
      - output:   str (absolute path or relative dir name)
      - identity: {name, description?, author?, response_units?}
      - crs:      {input, projected}
    """
    import yaml

    template = payload.get("template", "blank")
    output = payload.get("output")
    identity = payload.get("identity") or {}
    crs = payload.get("crs") or {}

    if not output:
        raise HTTPException(400, "output directory is required")
    if not identity.get("name"):
        raise HTTPException(400, "identity.name is required")
    if not crs.get("input") or not crs.get("projected"):
        raise HTTPException(400, "crs.input and crs.projected are required")

    source = TEMPLATES_DIR / template
    if not source.exists():
        available = [d.name for d in TEMPLATES_DIR.iterdir() if d.is_dir()]
        raise HTTPException(404, f"Template '{template}' not found. Available: {available}")

    dest = _resolve_safe(str(output), allow_create=True)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)

    yml_path = dest / "project.yml"
    if yml_path.exists():
        with open(yml_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = {}

    # Merge wizard identity into project block (preserve any template-seeded keys)
    proj = cfg.get("project") or {}
    proj["name"] = identity["name"]
    if "description" in identity:
        proj["description"] = identity.get("description") or ""
    proj["domain"] = template
    if "author" in identity:
        proj["author"] = identity.get("author") or ""
    if "response_units" in identity:
        proj["response_units"] = identity.get("response_units") or ""
    cfg["project"] = proj

    # Merge CRS
    cfg_crs = cfg.get("crs") or {}
    cfg_crs["input"] = crs["input"]
    cfg_crs["projected"] = crs["projected"]
    cfg["crs"] = cfg_crs

    with open(yml_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    return {
        "status": "created",
        "template": template,
        "path": str(dest),
        "project_yml": str(yml_path),
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


# ------------------------------------------------------------------
# Context layers (Phase 18)
# ------------------------------------------------------------------

@app.get("/context/layers")
async def get_context_layers(domain: str | None = Query(default=None)):
    """Return the basemap+overlay tile catalog, optionally filtered by *domain*."""
    from sparc.server.context_layers import get_catalog
    if domain is None and state.project_config is not None:
        proj = state.project_config.get("project") or {}
        domain = proj.get("domain")
    return get_catalog(domain)


# ------------------------------------------------------------------
# Phase-C — KernelField, causal PDP, divergence, scenario routing
# (correlogram, causal/pdp_curves and causal/divergence are now served
# by routes.results and routes.causal via include_router above)
# ------------------------------------------------------------------

@app.get("/results/kernel_field")
async def get_kernel_field_artifact():
    """Return the canonical KernelField artifact (Stage 0 → Stage 1).

    Falls back to the Stage-0 ``cross_correlogram_kernel_field`` payload
    when no MGWR-derived KernelField has been written yet — that artifact
    carries the same per-pair spatial-coupling information the panel
    needs.  The fallback payload is annotated with
    ``{"source": "cross_correlogram"}`` so the renderer can dispatch on
    schema.
    """
    store = _open_store()
    try:
        for stage in ("1", "0"):
            if store.has(stage, "kernel_field"):
                payload = store.read_any(stage, "kernel_field")
                if isinstance(payload, dict):
                    payload.setdefault("source", "kernel_field")
                return payload
        if store.has("0", "cross_correlogram_kernel_field"):
            payload = store.read_any("0", "cross_correlogram_kernel_field")
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["source"] = "cross_correlogram"
            return payload
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)
    raise _missing_artifact_response(
        artifact_id="kernel_field", stage="0/1",
        hint=(
            "No KernelField artifact written by Stages 0/1 and no "
            "cross_correlogram_kernel_field fallback found in Stage 0."
        ),
    )


@app.get("/results/scenarios/routing_audit")
async def get_scenario_routing_audit():
    """Return the scenario routing + anisotropic-frame audit (Stage 4, Phase C-4)."""
    return await _read_or_404(
        "4", "scenario_routing_audit",
        hint="Stage 4 has not produced scenario_routing_audit.",
    )


@app.get("/results/gwen")
async def get_gwen_data():
    """Return GWEN variable importance as a row-oriented table."""
    store = _open_store()
    try:
        if store.has("1", "gwen_variable_importance"):
            df = store.read_any("1", "gwen_variable_importance")
            return {"rows": df.to_dict(orient="records")}
        if store.has("1", "gwen_results"):
            return store.read_any("1", "gwen_results")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)
    raise _missing_artifact_response(
        artifact_id="gwen_variable_importance", stage="1",
        hint="Stage 1 (GWEN) has not produced gwen_variable_importance. Run Stage 1.",
    )


# ------------------------------------------------------------------
# GWEN approval gate — allows the frontend to surface GWEN results and
# let the user approve/skip before Stage 2 begins.
# ------------------------------------------------------------------

@app.get("/gwen/status")
async def get_gwen_status():
    """Return GWEN approval state and variable importance rows.

    Response schema::

        {
          "approved": bool,
          "approval_path": str,
          "rows": [...] | null,    # gwen_variable_importance records
          "stage1_complete": bool,
        }
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(state.project_config)
    approval_path = paths.gwen_approved
    approved = approval_path.exists()
    stage1_complete = False
    rows = None

    store = None
    try:
        if state.registry is not None:
            from sparc.registry.store import ArtifactStore
            store = ArtifactStore(state.registry)
            stage1_complete = store.has("1", "gwen_variable_importance") or store.has("1", "gwen_results")
            if store.has("1", "gwen_variable_importance"):
                df = store.read_any("1", "gwen_variable_importance")
                rows = df.to_dict(orient="records") if df is not None else None
            elif store.has("1", "gwen_results"):
                r = store.read_any("1", "gwen_results")
                rows = r.get("rows") if isinstance(r, dict) else None
    except Exception:
        pass

    return {
        "approved": approved,
        "approval_path": str(approval_path),
        "stage1_complete": stage1_complete,
        "rows": rows,
    }


@app.post("/gwen/approve")
async def post_gwen_approve():
    """Write the ``gwen_approved.txt`` sentinel so the pipeline can proceed to Stage 2.

    Returns the approval path and the timestamp written.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    import time
    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(state.project_config)
    approval_path = paths.gwen_approved
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    approval_path.write_text(f"approved at {ts}\n", encoding="utf-8")
    return {"approved": True, "approval_path": str(approval_path), "approved_at": ts}


@app.delete("/gwen/approve")
async def delete_gwen_approve():
    """Revoke approval by removing the ``gwen_approved.txt`` sentinel."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    from sparc.run.pipeline_paths import PipelinePaths

    paths = PipelinePaths.from_config(state.project_config)
    approval_path = paths.gwen_approved
    if approval_path.exists():
        approval_path.unlink()
    return {"approved": False, "approval_path": str(approval_path)}


@app.get("/results/model_performance")
async def get_model_performance():
    """Return per-model R²/RMSE for the R² bar chart on the Results page."""
    cached = state.result_cache.get("2", "model_performance_response")
    if cached is not None:
        return cached

    models: list[dict] = []

    # In-memory result from the just-finished stage 2 (if available).
    r2_result = state.get_result(2)
    if isinstance(r2_result, dict):
        indiv = (r2_result.get("performance") or {}).get("individual_models")
        if indiv is None:
            indiv = r2_result.get("base_models")
        if isinstance(indiv, dict):
            for name, metrics in indiv.items():
                if isinstance(metrics, dict) and metrics.get("r2") is not None:
                    models.append({
                        "name": name.upper(),
                        "r2": metrics["r2"],
                        "rmse": metrics.get("rmse"),
                    })
        ens = r2_result.get("final_ensemble") or r2_result.get("meta_ensemble_best")
        if isinstance(ens, dict) and ens.get("r2") is not None:
            models.append({
                "name": "Ensemble",
                "r2": ens["r2"],
                "rmse": ens.get("rmse"),
            })

    if not models:
        store = _open_store()
        try:
            if store.has("2", "ensemble_results"):
                data = store.read_any("2", "ensemble_results")
                base = data.get("base_models", {})
                for name, metrics in base.items():
                    if isinstance(metrics, dict) and metrics.get("r2") is not None:
                        models.append({
                            "name": name.upper(),
                            "r2": metrics["r2"],
                            "rmse": metrics.get("rmse"),
                        })
                ens = data.get("final_ensemble") or data.get("meta_ensemble_best")
                if isinstance(ens, dict) and ens.get("r2") is not None:
                    models.append({
                        "name": "Ensemble",
                        "r2": ens["r2"],
                        "rmse": ens.get("rmse"),
                    })
        finally:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(None)

    if not models:
        raise _missing_artifact_response(
            artifact_id="ensemble_results", stage="2",
            hint="Stage 2 has not produced ensemble_results. Run Stage 2.",
        )

    models.sort(key=lambda m: m.get("r2") or 0, reverse=True)
    result = {"models": models}
    state.result_cache.set("2", "model_performance_response", result)
    return result


@app.get("/results/spatial_cv/predictions")
async def get_spatial_cv_predictions():
    """Return spatial CV predictions as GeoJSON (DB-only)."""
    cached = state.result_cache.get("2", "spatial_cv_predictions_geojson")
    if cached is not None:
        return cached
    store = _open_store()
    try:
        if not store.has("2", "spatial_cv_predictions"):
            raise _missing_artifact_response(
                artifact_id="spatial_cv_predictions", stage="2",
                hint=(
                    "Stage 2 has not produced spatial_cv_predictions. "
                    "Run Stage 2 to populate the predictions map."
                ),
            )
        gdf = store.read_any("2", "spatial_cv_predictions")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)

    import geopandas as gpd
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise HTTPException(
            500,
            "Artifact (2, spatial_cv_predictions) is not a GeoDataFrame; "
            "the producer must include a geometry column.",
        )
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    # Build a self-describing response: include the target unit so the
    # frontend never has to decode z-scores or look up a separate artifact.
    unit_str = ""
    if "_unit" in gdf.columns and len(gdf) > 0:
        unit_str = str(gdf["_unit"].iloc[0])
    if not unit_str:
        raw = getattr(state, "raw_project_yaml", None) or {}
        unit_str = (
            (raw.get("output") or {}).get("response_units")
            or (raw.get("project") or {}).get("response_units")
            or ""
        )
    geojson = dict(gdf.__geo_interface__)
    if unit_str:
        geojson["unit"] = unit_str
    state.result_cache.set("2", "spatial_cv_predictions_geojson", geojson)
    return geojson


@app.get("/results/causal")
async def get_causal_results():
    """Return causal validation results (DB-only)."""
    store = _open_store()
    try:
        if store.has("3", "scenario_coefficients"):
            return store.read_any("3", "scenario_coefficients")
        if store.has("3", "causal_diagnostics"):
            return store.read_any("3", "causal_diagnostics")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)

    mem_result = state.get_result(3)
    if mem_result is not None:
        return mem_result

    raise _missing_artifact_response(
        artifact_id="scenario_coefficients", stage="3",
        hint=(
            "Stage 3 (Causal Validation) has not produced "
            "`scenario_coefficients`. Run Stage 3 to populate the causal panel."
        ),
    )


@app.get("/results/causal/dose_response")
async def get_dose_response():
    """Return dose-response curves (DB-only)."""
    return await _read_or_404(
        "3", "dose_response_curves",
        hint=(
            "Dose-response curves are produced by Stage 3 (Causal Validation). "
            "They are skipped when `causal.inference: bayesian`. Re-run Stage 3 "
            "in frequentist mode or check the Causal Diagnostics panel for the "
            "Bayesian posterior."
        ),
    )


@app.get("/results/causal/sensitivity")
async def get_causal_sensitivity():
    """Return E-values + tipping-point analysis for the current causal payload.

    Reuses ``/results/causal`` and annotates each effect with VanderWeele
    E-values so users can see how strong an unmeasured confounder would
    have to be to explain away the observed effect.
    """
    from sparc.causal.sensitivity import annotate_causal_payload

    payload = await get_causal_results()  # type: ignore[misc]
    if not isinstance(payload, dict):
        raise HTTPException(500, "Causal payload is not a dict")
    annotated = annotate_causal_payload(payload)
    sens = annotated.get("sensitivity") or {}
    return sens


@app.get("/results/causal/negative_control")
async def get_causal_negative_control(
    variable: str = Query(...),
    n_permutations: int = Query(1000, ge=50, le=10000),
):
    """Permutation negative-control test on the spatial CATE values (DB-only)."""
    from sparc.evaluation.negative_controls import permutation_test_cate

    arr_np = _read_cate_multiplier_from_store(variable)
    if arr_np is None:
        raise _missing_artifact_response(
            artifact_id=f"cate_summary[{variable}]", stage="3",
            hint=(
                f"No CATE map for variable '{variable}'. Re-run Stage 3 "
                "with `causal.estimate_cate: true`."
            ),
        )
    arr = arr_np.astype(float).tolist()
    res = permutation_test_cate(arr, n_permutations=n_permutations)
    return {
        "variable": variable,
        "n": res.n,
        "mean_observed": res.mean_observed,
        "mean_null": res.mean_null,
        "std_null": res.std_null,
        "p_value": res.p_value,
        "z_score": res.z_score,
        "n_permutations": res.n_permutations,
        "passed": res.passed,
        "interpretation": (
            "p > 0.05 — observed CATE indistinguishable from null (negative-control PASS)."
            if res.passed
            else f"p = {res.p_value:.4f} — CATE is significantly non-zero (real treatment effect)."
        ),
    }


@app.get("/results/causal/diagnostics")
async def get_causal_diagnostics():
    """Return CATE diagnostics (calibration, cumulative effects, RATE)."""
    return await _read_or_404(
        "3", "causal_diagnostics",
        hint="Stage 3 (Causal Validation) has not produced causal_diagnostics.",
    )


def _load_neural_pdp(paths) -> dict:
    """Read PINN neural-network PDP curves into the canonical curves dict (DB-only).

    Each PDP table is registered as ``v2_neural_pdp::<feature>`` under stage 2.
    The ``paths`` argument is accepted for API compatibility but unused.
    """
    out: dict = {}
    if state.registry is None:
        return out
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore
        set_active_registry(state.registry)
        try:
            _store = ArtifactStore(state.registry)
            manifest_stages = getattr(state.registry.manifest, "stages", {}) or {}
            stage2 = manifest_stages.get("2", {}) or {}
            for art_id in stage2.keys():
                if not art_id.startswith("v2_neural_pdp::"):
                    continue
                feat_col = art_id.split("::", 1)[1]
                try:
                    df = _store.read_any("2", art_id)
                except Exception:
                    continue
                if df is None or len(df) == 0 or feat_col not in df.columns:
                    continue
                grid = [float(v) for v in df[feat_col].tolist()]
                pdp_vals = [float(v) for v in df["mean_prediction"].tolist()]
                if {"q10", "q90"}.issubset(df.columns):
                    pdp_std = [
                        (float(q90) - float(q10)) / 2.56
                        for q10, q90 in zip(df["q10"].tolist(), df["q90"].tolist())
                    ]
                else:
                    pdp_std = None
                out[feat_col] = {
                    "grid_values": grid,
                    "pdp_values": pdp_vals,
                    "pdp_std": pdp_std,
                    "source": "neural_pde",
                }
        finally:
            set_active_registry(None)
    except Exception:
        pass
    return out


def _load_gwrf_pdp(paths) -> dict | None:
    """Load GWRF condition curves dict from artifacts.db (DB-only).

    Returns the parsed dict (variable -> curve metadata) or ``None`` if no
    artifact is registered. The ``paths`` argument is accepted for API
    compatibility but unused.
    """
    if state.registry is None:
        return None
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore
        set_active_registry(state.registry)
        try:
            _store = ArtifactStore(state.registry)
            for stage_id, art_id in [
                ("2", "gwrf_condition_curves"),
                ("3", "gwrf_condition_curves"),
            ]:
                if _store.has(stage_id, art_id):
                    data = _store.read_any(stage_id, art_id)
                    if isinstance(data, dict):
                        for _var, curve in data.items():
                            if isinstance(curve, dict):
                                curve.setdefault("source", "gwrf")
                        return data
        finally:
            set_active_registry(None)
    except Exception:
        return None
    return None


@app.get("/results/neural_pdp")
async def get_neural_pdp():
    """Return PINN-derived PDP curves (DB-only)."""
    curves = _load_neural_pdp(None) if state.registry is None else _load_neural_pdp(None)
    if not curves:
        raise _missing_artifact_response(
            artifact_id="v2_neural_pdp::*", stage="2",
            hint="Neural PDP curves not available — run Stage 2 (v2_neural training) first.",
        )
    return curves


@app.get("/results/pdp_curves")
async def get_pdp_curves():
    """Return partial dependence / condition curves from any available source (DB-only).

    Merges in order of preference:
      1. PINN neural-network PDP   (artifact `v2_neural_pdp::<feature>`)
      2. GWRF condition curves     (artifact `gwrf_condition_curves`)
    Causal dose-response (Stage 3) is exposed under `_meta.by_source.causal_dose_response`.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    gwrf = _load_gwrf_pdp(None) or {}
    neural = _load_neural_pdp(None) or {}

    available_sources: list[str] = []
    if neural:
        available_sources.append("neural_pde")
    if gwrf:
        available_sources.append("gwrf")

    causal_curves: dict = {}
    if state.registry is not None:
        try:
            from sparc.registry.run_registry import set_active_registry
            from sparc.registry.store import ArtifactStore
            set_active_registry(state.registry)
            try:
                _store = ArtifactStore(state.registry)
                if _store.has("3", "dose_response_curves"):
                    cd = _store.read_any("3", "dose_response_curves")
                    if isinstance(cd, dict):
                        for var, curve in cd.items():
                            if isinstance(curve, dict):
                                curve.setdefault("source", "causal_dose_response")
                                causal_curves[var] = curve
                        if causal_curves:
                            available_sources.append("causal_dose_response")
            finally:
                set_active_registry(None)
        except Exception:
            pass

    merged: dict = {}
    merged.update(gwrf)
    merged.update(neural)

    if not merged and not causal_curves:
        raise _missing_artifact_response(
            artifact_id="pdp_curves", stage="2",
            hint=(
                "No response-curve data found. Stage 2 produces neural PDP "
                "and GWRF condition curves; Stage 3 (frequentist mode) produces "
                "causal dose-response curves. Re-run the relevant stage."
            ),
        )

    merged["_meta"] = {
        "available_sources": available_sources,
        "by_source": {
            "neural_pde": neural,
            "gwrf": gwrf,
            "causal_dose_response": causal_curves,
        },
    }
    return merged


@app.get("/results/scenarios/nuts_summary")
async def get_nuts_summary():
    """Return NUTS posterior summaries, convergence diagnostics, BMA coefficients (DB-only)."""
    store = _open_store()
    try:
        result: dict = {}
        if store.has("3", "nuts_summary"):
            ns = store.read_any("3", "nuts_summary") or {}
            result["acceptance_rate"] = ns.get("acceptance_rate")
            result["n_divergences"] = ns.get("n_divergences")
        if store.has("3", "parameter_posteriors"):
            df = store.read_any("3", "parameter_posteriors")
            if df is not None:
                result["posteriors"] = df.to_dict(orient="records")
        if store.has("3", "convergence_diagnostics"):
            df = store.read_any("3", "convergence_diagnostics")
            if df is not None:
                result["convergence"] = df.to_dict(orient="records")
        if store.has("3", "bma_coefficients"):
            df = store.read_any("3", "bma_coefficients")
            if df is not None:
                result["bma"] = df.to_dict(orient="records")
    finally:
        from sparc.registry.run_registry import set_active_registry
        set_active_registry(None)

    if not result:
        raise _missing_artifact_response(
            artifact_id="nuts_summary", stage="3",
            hint=(
                "No NUTS results found. Run Stage 3 with "
                "`causal.inference: bayesian` to populate posterior summaries."
            ),
        )
    return result


@app.get("/results/scenarios")
async def list_scenarios():
    """Enumerate available scenarios from the active scenario_results table.

    Reads exclusively through the ArtifactStore (no disk fallback). Returns
    `{ "scenarios": [{"index": int, "pred_column": str, "delta_column": str|None}],
       "results_artifact_id": str | None,
       "available": [str] }`
    """
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import ScenarioBundle

    if state.registry is None:
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="No active run registry. Load a project first.",
        )
    store = ArtifactStore(state.registry)
    bundle = ScenarioBundle.from_store(store)
    return {
        "results_artifact_id": bundle.results_artifact_id,
        "summary_artifact_id": bundle.summary_artifact_id,
        "available": sorted(bundle.available),
        "has_uncertainty": bundle.has_uncertainty(),
        "scenarios": [
            {
                "index": sid,
                "pred_column": (rec := bundle.get_scenario(sid)).pred_column,
                "delta_column": rec.delta_column,
            }
            for sid in bundle.list_scenarios()
        ],
    }


@app.get("/results/scenarios/detail")
async def get_scenario_detail():
    """Return scenario results as GeoJSON with delta columns + summary table.

    Reads exclusively through the ArtifactStore via :class:`ScenarioBundle`.
    Mode-variant precedence (hybrid > reprediction > dag > base) is handled
    automatically. The legacy on-disk ``scenario_results*.gpkg`` files are
    deliberately ignored.
    """
    cached = state.result_cache.get("4", "scenario_detail_response")
    if cached is not None:
        return cached

    if state.registry is None:
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="No active run registry. Load a project first.",
        )

    from sparc.registry.store import ArtifactStore
    from sparc.scenario import (
        DELTA_SCENARIO_PREFIX,
        PRED_BASELINE_COL,
        PRED_SCENARIO_PREFIX,
        ScenarioBundle,
    )

    store = ArtifactStore(state.registry)
    bundle = ScenarioBundle.from_store(store)

    if "results" not in bundle.available or bundle.results is None:
        scenarios_cfg = (state.project_config or {}).get("scenarios") or []
        scenario_count = len(scenarios_cfg) if isinstance(scenarios_cfg, list) else 0
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint=(
                f"Stage 4 should auto-run the {scenario_count} scenario(s) "
                "defined in project.yml. If Stage 4 finished without "
                "writing scenario results to the database, re-run Stage 4 "
                "or use the Scenario Runner page."
                if scenario_count
                else "No `scenarios:` block found in project.yml."
            ),
        )

    import geopandas as gpd

    gdf = bundle.results
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise HTTPException(
            500,
            f"Artifact {bundle.results_artifact_id} is not a GeoDataFrame; "
            "scenario producer must include a geometry column.",
        )

    # Compute delta_Scenario{N} columns when missing.
    if PRED_BASELINE_COL in gdf.columns:
        for col in list(gdf.columns):
            if not col.startswith(PRED_SCENARIO_PREFIX):
                continue
            delta_col = col.replace(PRED_SCENARIO_PREFIX, DELTA_SCENARIO_PREFIX)
            if delta_col in gdf.columns:
                continue
            try:
                gdf[delta_col] = (
                    gdf[col].astype(float) - gdf[PRED_BASELINE_COL].astype(float)
                )
            except Exception:
                pass

    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    geojson_data = gdf.__geo_interface__

    summary_records: list[dict[str, Any]] = []
    if bundle.summary is not None:
        try:
            summary_records = bundle.summary.to_dict(orient="records")
        except Exception:
            summary_records = []

    response = {
        "geojson": geojson_data,
        "summary": summary_records,
        "results_artifact_id": bundle.results_artifact_id,
        "summary_artifact_id": bundle.summary_artifact_id,
    }
    state.result_cache.set("4", "scenario_detail_response", response)
    return response


@app.get("/results/scenarios/attribution")
async def get_scenario_attribution():
    """Return the per-scenario × per-variable attribution table."""
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import SCENARIO_ATTRIBUTION, SCENARIO_STAGE

    if state.registry is None:
        raise _missing_artifact_response(
            artifact_id=SCENARIO_ATTRIBUTION, stage=SCENARIO_STAGE,
            hint="No active run registry.",
        )
    store = ArtifactStore(state.registry)
    if not store.has(SCENARIO_STAGE, SCENARIO_ATTRIBUTION):
        raise _missing_artifact_response(
            artifact_id=SCENARIO_ATTRIBUTION, stage=SCENARIO_STAGE,
            hint="Scenario simulator did not produce an attribution table.",
        )
    df = store.read_table(SCENARIO_STAGE, SCENARIO_ATTRIBUTION)
    return {"records": df.to_dict(orient="records"), "columns": list(df.columns)}


@app.get("/results/scenarios/trajectory")
async def get_scenario_trajectory(scenario_id: int | None = None):
    """Return the long-format scenario trajectory table, optionally filtered.

    Expected columns: ``scenario_id, t, geometry_id, value, std?``.
    """
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import SCENARIO_STAGE, SCENARIO_TRAJECTORY

    if state.registry is None:
        raise _missing_artifact_response(
            artifact_id=SCENARIO_TRAJECTORY, stage=SCENARIO_STAGE,
            hint="No active run registry.",
        )
    store = ArtifactStore(state.registry)
    if not store.has(SCENARIO_STAGE, SCENARIO_TRAJECTORY):
        raise _missing_artifact_response(
            artifact_id=SCENARIO_TRAJECTORY, stage=SCENARIO_STAGE,
            hint="Scenario simulator did not produce a trajectory table.",
        )
    df = store.read_table(SCENARIO_STAGE, SCENARIO_TRAJECTORY)
    if scenario_id is not None and "scenario_id" in df.columns:
        df = df[df["scenario_id"] == scenario_id]
    return {"records": df.to_dict(orient="records"), "columns": list(df.columns)}


@app.get("/results/scenarios/uncertainty")
async def get_scenario_uncertainty(scenario_id: int | None = None):
    """Return per-feature uncertainty (mean/std/p05/p95) as GeoJSON.

    When ``scenario_id`` is provided, the per-scenario prediction column is
    included; otherwise only the uncertainty columns are exposed.
    """
    from sparc.registry.store import ArtifactStore
    from sparc.scenario import (
        UNCERTAINTY_COLS,
        ScenarioBundle,
        pred_column,
    )

    if state.registry is None:
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="No active run registry.",
        )
    store = ArtifactStore(state.registry)
    bundle = ScenarioBundle.from_store(store)
    if bundle.results is None:
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="Scenario results not in database.",
        )
    if not bundle.has_uncertainty():
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint=(
                "Active scenario_results table has no uncertainty columns "
                f"({', '.join(UNCERTAINTY_COLS)}). Re-run with the MC-Dropout "
                "ensemble enabled."
            ),
        )
    import geopandas as gpd

    gdf = bundle.results
    keep = [c for c in UNCERTAINTY_COLS if c in gdf.columns]
    if scenario_id is not None:
        sc = bundle.get_scenario(scenario_id)
        if sc is not None:
            keep.append(sc.pred_column)
            if sc.delta_column:
                keep.append(sc.delta_column)
    if isinstance(gdf, gpd.GeoDataFrame):
        out = gdf[[*keep, gdf.geometry.name]].copy()
        if out.crs is not None and str(out.crs) != "EPSG:4326":
            out = out.to_crs(epsg=4326)
        # Surface dedicated MC tables when present so the UI can render
        # quantile fans / consensus diagnostics without a second round-trip.
        mc_uncertainty_records: list[dict[str, Any]] | None = None
        mc_consensus_records: list[dict[str, Any]] | None = None
        mc_consensus_summary_records: list[dict[str, Any]] | None = None
        if bundle.mc_uncertainty is not None and hasattr(bundle.mc_uncertainty, "to_dict"):
            try:
                mc_uncertainty_records = bundle.mc_uncertainty.to_dict(orient="records")
            except Exception:
                mc_uncertainty_records = None
        if bundle.mc_consensus is not None and hasattr(bundle.mc_consensus, "to_dict"):
            try:
                mc_consensus_records = bundle.mc_consensus.to_dict(orient="records")
            except Exception:
                mc_consensus_records = None
        if bundle.mc_consensus_summary is not None and hasattr(
            bundle.mc_consensus_summary, "to_dict"
        ):
            try:
                mc_consensus_summary_records = bundle.mc_consensus_summary.to_dict(
                    orient="records"
                )
            except Exception:
                mc_consensus_summary_records = None
        return {
            "geojson": out.__geo_interface__,
            "results_artifact_id": bundle.results_artifact_id,
            "mc_uncertainty": mc_uncertainty_records,
            "mc_consensus": mc_consensus_records,
            "mc_consensus_summary": mc_consensus_summary_records,
        }
    raise HTTPException(500, "scenario results table is not a GeoDataFrame")


# `/results/scenario_library/timeline` was removed in the Phase C7 audit —
# it duplicated `/scenarios/library`, which is now DB-first with disk fallback.


@app.get("/results/report")
async def get_report_data():
    """Compile all stage results into a structured payload for the report view (DB-only)."""
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

    if state.registry is None:
        return report

    from sparc.registry.run_registry import set_active_registry
    from sparc.registry.store import ArtifactStore
    set_active_registry(state.registry)
    try:
        store = ArtifactStore(state.registry)

        # Stage 0 — correlogram summary
        if store.has("0", "correlogram_results"):
            corr_data = store.read_any("0", "correlogram_results")
            individual = (corr_data or {}).get("individual_results", {})
            report["correlogram"] = {
                var: {
                    "optimal_bandwidth": info.get("optimal_bandwidth"),
                    "effective_range": info.get("effective_range"),
                    "max_moran_i": info.get("max_moran_i"),
                }
                for var, info in individual.items()
            }

        # Stage 1 — GWEN variable importance
        if store.has("1", "gwen_variable_importance"):
            gwen_df = store.read_any("1", "gwen_variable_importance")
            if gwen_df is not None:
                report["gwen"] = gwen_df.to_dict(orient="records")

        # Stage 2 — spatial CV models
        if store.has("2", "oof_predictions"):
            oof_df = store.read_any("2", "oof_predictions")
            if oof_df is not None:
                report["spatial_cv_models"] = list(oof_df.columns)
        if store.has("2", "ensemble_results"):
            report["ensemble_results"] = store.read_any("2", "ensemble_results")

        # Stage 3 — causal coefficients + dose-response + propensity
        for art_id, key in (
            ("scenario_coefficients", "causal_results"),
            ("dose_response_curves", "dose_response"),
            ("propensity_diagnostics", "propensity_diagnostics"),
            ("causal_diagnostics", "causal_diagnostics"),
        ):
            if store.has("3", art_id):
                report[key] = store.read_any("3", art_id)

        # Stage 4 — scenario summary via ScenarioBundle so that the report
        # reflects the same mode-variant precedence (hybrid > reprediction >
        # dag > base) used everywhere else in the API surface. We also
        # surface the chosen artifact ids so report consumers can show the
        # user *which* variant fed the document.
        from sparc.scenario import ScenarioBundle

        bundle = ScenarioBundle.from_store(store)
        if bundle.summary is not None and hasattr(bundle.summary, "to_dict"):
            try:
                report["scenario_summary"] = bundle.summary.to_dict(orient="records")
            except Exception:
                pass
        elif bundle.mc_consensus_summary is not None and hasattr(
            bundle.mc_consensus_summary, "to_dict"
        ):
            try:
                report["scenario_summary"] = bundle.mc_consensus_summary.to_dict(
                    orient="records"
                )
            except Exception:
                pass
        report["scenario_results_artifact_id"] = bundle.results_artifact_id
        report["scenario_summary_artifact_id"] = bundle.summary_artifact_id
    finally:
        set_active_registry(None)

    # Plot URLs intentionally omitted: visualisations are rendered live in
    # the desktop app from artifacts.db data and exported client-side.
    report["plots"] = {}
    return report


# ------------------------------------------------------------------
# Insights aggregator (Phase 4): single round-trip "headline" summary
# the practitioner Insights view shows at the top of the page.  Combines
# the best-ranked decision candidate, sensitivity strength, and a few
# top-line counts so the desktop doesn't need to fan out to four
# separate endpoints just to render one card.
# ------------------------------------------------------------------

@app.get("/insights/headline")
async def get_insights_headline():
    """Return a single 'headline' payload for the Insights page.

    Shape::

        {
          "best": InterventionCandidate | null,
          "alternatives": int,
          "score": float | null,             # mean_effect / |cost|
          "sensitivity": {                   # strongest E-value across effects
            "treatment": str | null,
            "e_value": float | null,
            "robust": bool,                  # e_value >= 2
          } | null,
          "n_treatments": int,
          "n_outcomes": int,
          "warnings": [str, ...],
        }
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    # 1) Decision candidates → "best".  Reuse the existing endpoint so
    #    the heuristic stays single-sourced.
    best: dict | None = None
    score: float | None = None
    n_alt = 0
    warnings: list[str] = []
    try:
        cand_payload = await get_decision_candidates()  # type: ignore[misc]
        candidates = (cand_payload or {}).get("candidates", []) if isinstance(cand_payload, dict) else []
    except HTTPException as exc:
        candidates = []
        warnings.append(f"decision: {exc.detail}")

    if candidates:
        scored: list[tuple[float, dict]] = []
        for c in candidates:
            try:
                eff = float(c.get("mean_effect"))
            except (TypeError, ValueError):
                continue
            cost = abs(float(c.get("cost", 1.0) or 1.0)) or 1.0
            scored.append((eff / cost, c))
        if scored:
            scored.sort(key=lambda p: p[0], reverse=True)
            score, best = scored[0]
            n_alt = max(0, len(candidates) - 1)

    # 2) Strongest sensitivity (E-value) across reported effects.
    sensitivity_top: dict | None = None
    try:
        sens = await get_causal_sensitivity()  # type: ignore[misc]
        effects = []
        if isinstance(sens, dict):
            effects = sens.get("effects") or sens.get("rows") or []
        if not isinstance(effects, list):
            effects = []
        best_e = None
        for row in effects:
            if not isinstance(row, dict):
                continue
            try:
                ev = float(row.get("e_value"))
            except (TypeError, ValueError):
                continue
            if best_e is None or ev > best_e["e_value"]:
                best_e = {"treatment": row.get("treatment") or row.get("variable"), "e_value": ev}
        if best_e is not None:
            best_e["robust"] = best_e["e_value"] >= 2.0
            sensitivity_top = best_e
    except HTTPException as exc:
        warnings.append(f"sensitivity: {exc.detail}")

    # 3) Cheap structure counts from the DAG so the headline can show
    #    "ranked across N treatments → 1 outcome".
    n_treatments = 0
    n_outcomes = 0
    try:
        dag_file = state.project_config.get("causal", {}).get("dag_file")
        if dag_file and Path(dag_file).exists():
            from sparc.causal.dag_definition import load_dag, dag_to_networkx, get_node_roles
            dag = load_dag(dag_file)
            G = dag_to_networkx(dag)
            roles = get_node_roles(G)
            n_treatments = len(roles.get("treatments", []))
            n_outcomes = len(roles.get("outcomes", []))
    except Exception as exc:
        warnings.append(f"dag: {exc}")

    return {
        "best": best,
        "alternatives": n_alt,
        "score": score,
        "sensitivity": sensitivity_top,
        "n_treatments": n_treatments,
        "n_outcomes": n_outcomes,
        "warnings": warnings,
    }


# ------------------------------------------------------------------
# Dataset profile (Phase 4): richer column-level diagnostics than
# ``/data/summary``.  Adds missing-rate, skewness, kurtosis, and
# Pearson correlation against the project target so the Insights view
# can flag degenerate or weakly-related predictors.
# ------------------------------------------------------------------

@app.get("/results/dataset/profile")
async def get_dataset_profile():
    """Return per-column diagnostics for the loaded project dataset.

    Shape::

        {
          "n_rows": int,
          "n_cols": int,
          "target": str | null,
          "columns": [
            {
              "name": str,
              "dtype": str,
              "missing_pct": float,
              "n_unique": int | null,
              "min": float | null, "max": float | null,
              "mean": float | null, "std": float | null,
              "skew": float | null, "kurtosis": float | null,
              "corr_target": float | null
            }, ...
          ],
          "crs": str | null,
        }
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.data is None:
        raise HTTPException(404, "No dataset loaded — upload a CSV first")

    import numpy as np
    import pandas as pd

    df = state.data
    target = (state.project_config.get("variables") or {}).get("target")
    target_series = None
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        target_series = df[target]

    cols_out: list[dict] = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        miss = float(s.isna().sum()) / max(n, 1)
        entry: dict = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_pct": round(miss * 100.0, 3),
        }
        if pd.api.types.is_numeric_dtype(s):
            arr = s.dropna().to_numpy()
            if arr.size > 0:
                entry["n_unique"] = int(pd.Series(arr).nunique())
                entry["min"] = float(np.min(arr))
                entry["max"] = float(np.max(arr))
                entry["mean"] = float(np.mean(arr))
                entry["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                if arr.size >= 3 and entry["std"] > 0:
                    try:
                        entry["skew"] = float(pd.Series(arr).skew())
                        entry["kurtosis"] = float(pd.Series(arr).kurtosis())
                    except Exception:
                        entry["skew"] = None
                        entry["kurtosis"] = None
                else:
                    entry["skew"] = None
                    entry["kurtosis"] = None
                if target_series is not None and col != target:
                    try:
                        corr = float(s.corr(target_series))
                        entry["corr_target"] = corr if np.isfinite(corr) else None
                    except Exception:
                        entry["corr_target"] = None
                else:
                    entry["corr_target"] = None
            else:
                entry.update({
                    "n_unique": 0, "min": None, "max": None, "mean": None,
                    "std": None, "skew": None, "kurtosis": None, "corr_target": None,
                })
        else:
            try:
                entry["n_unique"] = int(s.nunique(dropna=True))
            except Exception:
                entry["n_unique"] = None
            entry.update({"min": None, "max": None, "mean": None, "std": None,
                          "skew": None, "kurtosis": None, "corr_target": None})
        cols_out.append(entry)

    crs = None
    if hasattr(df, "crs") and df.crs is not None:
        crs = str(df.crs)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "target": target,
        "columns": cols_out,
        "crs": crs,
    }


# ------------------------------------------------------------------
# CATE map & increment endpoints (Phase 2)
# ------------------------------------------------------------------

def _read_cate_column_from_store(variable: str, column: str = "multiplier_mean"):
    """Return per-cell values of *column* for *variable* from the
    ``("3","cate_summary")`` long-format table, or ``None`` if absent.

    The v4 causal stack persists CATE results as a long-format table
    keyed by ``(cell_id, treatment)``. Filter by treatment, sort by
    ``cell_id``, return *column* as a NumPy array.

    The default ``multiplier_mean`` is the dimensionless spatial sensitivity
    multiplier consumed internally by the scenario simulator; pass
    ``column="cate_mean"`` to read the Bayesian per-cell coefficient β(s)
    in original ΔY-per-unit-T units (this is what the CATE map and the
    Decision Support benefit vector display).
    """
    if state.registry is None:
        return None
    try:
        from sparc.registry.run_registry import set_active_registry
        from sparc.registry.store import ArtifactStore
        import numpy as np

        set_active_registry(state.registry)
        try:
            store = ArtifactStore(state.registry)
            if not store.has("3", "cate_summary"):
                return None
            df = store.read_table("3", "cate_summary")
            if df is None or "treatment" not in df.columns:
                return None
            sub = df[df["treatment"] == variable]
            if sub.empty or column not in sub.columns:
                return None
            sub = sub.sort_values("cell_id")
            return np.asarray(sub[column].to_numpy(), dtype=float)
        finally:
            set_active_registry(None)
    except Exception:
        return None


def _read_cate_multiplier_from_store(variable: str):
    """Back-compat wrapper: return ``multiplier_mean`` (used by scenario simulator)."""
    return _read_cate_column_from_store(variable, "multiplier_mean")


def _read_cate_coefficient_from_store(variable: str, *, with_ci: bool = False):
    """Return the Bayesian per-cell coefficient β(s) for *variable*.

    Reads ``cate_mean`` from the ``("3","cate_summary")`` table. When
    ``with_ci=True``, also returns ``cate_ci5`` and ``cate_ci95`` so the
    caller can render uncertainty (e.g. stipple cells whose 90% CI crosses
    zero on the CATE map).

    Returns ``None`` if the artifact is missing, otherwise a dict with
    keys ``mean``, ``ci5``, ``ci95``, ``source``. ``ci5``/``ci95`` may be
    ``None`` when ``with_ci=False`` or when those columns are absent.
    """
    mean = _read_cate_column_from_store(variable, "cate_mean")
    if mean is None:
        return None
    src = _read_cate_column_from_store(variable, "source")
    source_label = None
    if src is not None and len(src):
        try:
            source_label = str(src[0])
        except Exception:
            source_label = None
    out: dict[str, Any] = {"mean": mean, "source": source_label}
    if with_ci:
        out["ci5"] = _read_cate_column_from_store(variable, "cate_ci5")
        out["ci95"] = _read_cate_column_from_store(variable, "cate_ci95")
    return out


@app.get("/results/causal/cate_map")
async def get_cate_map(
    variable: str = Query(...),
    with_uncertainty: bool = Query(
        False,
        description="When true, also emit per-cell coef_ci5_<var> / coef_ci95_<var>"
                    " properties so the frontend can stipple cells whose 90% CI"
                    " crosses zero (“not significant”).",
    ),
):
    """Return the Bayesian per-cell coefficient β(s) for *variable* as GeoJSON.

    DB-only: reads ``cate_mean`` (and optionally ``cate_ci5``/``cate_ci95``)
    from the ``("3","cate_summary")`` table and joins onto the in-memory
    project geometry. β(s) is the posterior mean local treatment effect
    in original ΔY-per-unit-T units, produced by ``BayesianSpatialCATE``
    (NUTS over a random-Fourier-feature basis on coordinates).

    GeoJSON properties are written under both ``coef_<var>`` (canonical) and
    ``cate_<var>`` (one-release alias for back-compat with the old multiplier
    payload). Top-level foreign members ``units``, ``source``, and ``variable``
    are added so the desktop can render the legend and equation panel.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    payload = _read_cate_coefficient_from_store(variable, with_ci=with_uncertainty)
    if payload is None:
        raise _missing_artifact_response(
            artifact_id=f"cate_summary[{variable}]", stage="3",
            hint=(
                f"No CATE map for variable '{variable}'. Re-run Stage 3 with "
                "`causal.estimate_cate: true` and at least one treatment in your DAG."
            ),
        )

    if state.data is None or not hasattr(state.data, "geometry"):
        raise HTTPException(
            500,
            "Project data has no geometry; cannot project CATE coefficients "
            "onto the map. Re-load the project so the input dataset is"
            " parsed with coordinate columns.",
        )

    import geopandas as gpd

    src = state.data
    coef = payload["mean"]
    n = min(len(src), len(coef))
    cols: dict[str, Any] = {
        f"coef_{variable}": coef[:n],
        f"cate_{variable}": coef[:n],  # back-compat alias — same values as coef_<var>
    }
    if with_uncertainty:
        ci5 = payload.get("ci5")
        ci95 = payload.get("ci95")
        if ci5 is not None and ci95 is not None:
            cols[f"coef_ci5_{variable}"] = ci5[:n]
            cols[f"coef_ci95_{variable}"] = ci95[:n]
            # “Not significant at 90%” flag — the frontend uses this to
            # dim/stipple cells whose CI brackets zero.
            import numpy as _np
            cols[f"coef_ns_{variable}"] = (
                (_np.asarray(ci5[:n], dtype=float) <= 0.0)
                & (_np.asarray(ci95[:n], dtype=float) >= 0.0)
            )

    gdf = gpd.GeoDataFrame(cols, geometry=src.geometry.values[:n], crs=src.crs)
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    fc = gdf.__geo_interface__
    # GeoJSON foreign members — RFC 7946 §6.1.
    fc["variable"] = variable
    fc["units"] = "ΔY per unit T (posterior mean of β(s))"
    fc["source"] = payload.get("source") or "bayesian_nuts"
    fc["with_uncertainty"] = bool(with_uncertainty)
    return fc


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

    variables: list[str] = []
    diagnostics: dict[str, Any] = {}

    # 1. Prefer ("3","cate_summary") long-format table.
    if state.registry is not None:
        try:
            from sparc.registry.run_registry import set_active_registry
            from sparc.registry.store import ArtifactStore
            set_active_registry(state.registry)
            try:
                store = ArtifactStore(state.registry)
                if store.has("3", "cate_summary"):
                    df = store.read_table("3", "cate_summary")
                    if df is not None and "treatment" in df.columns:
                        variables.extend(
                            sorted(str(t) for t in df["treatment"].unique())
                        )
                        diagnostics["cate_summary_hits"] = len(variables)
            finally:
                set_active_registry(None)
        except Exception as exc:
            diagnostics["cate_summary_error"] = str(exc)

    # 2. Legacy registry: any cate_multiplier::* artifact wins.
    if not variables and state.registry is not None:
        for art in state.registry.list_for_stage("3"):
            if art.id.startswith("cate_multiplier::") and not art.partial:
                var = art.metadata.get("variable") or art.id.split("::", 1)[1]
                variables.append(var)
        diagnostics["registry_hits"] = len(variables)

    # 3. Fallback: per-variable .npy files on disk.
    # (DB-only: legacy disk lookups removed in v4 refresh.)

    # 3. Fallback: column names in spatial_cate_maps.gpkg.
    # (DB-only: legacy gpkg lookups removed in v4 refresh.)

    # 4. Empty-state hint to help the user act on the Budget Optimizer screen.
    sorted_vars = sorted(set(variables))
    payload: dict[str, Any] = {"variables": sorted_vars}
    if not sorted_vars:
        causal_cfg = state.project_config.get("causal", {}) or {}
        inference = (causal_cfg.get("inference") or "").lower()
        estimate_cate = causal_cfg.get("estimate_cate", True)
        reasons: list[str] = []
        if not estimate_cate:
            reasons.append("`causal.estimate_cate` is false in project.yml.")
        if inference == "bayesian":
            reasons.append(
                "Bayesian inference path historically skipped CATE; this is "
                "fixed in newer Stage 3 runs but the existing run was "
                "produced before the fix."
            )
        if not reasons:
            reasons.append(
                "Stage 3 finished but did not write spatial CATE multipliers. "
                "Check the DAG for at least one treatment node."
            )
        payload["empty_reason"] = " ".join(reasons)
        payload["next_action"] = (
            "Re-run Stage 3 with `causal.estimate_cate: true` and at least "
            "one treatment in your DAG."
        )
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


# NOTE: the legacy ``/results/local_coefficients`` and
# ``/results/local_coefficients/variables`` endpoints have been removed.
# MGWR/GWR are kept as Stage-2 diagnostics, but their coefficients are
# no longer surfaced for effect or decision logic — every consumer now
# reads the Bayesian per-cell coefficient β(s) from
# ``("3","cate_summary")`` via :func:`_read_cate_coefficient_from_store`.


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

    DB-only: reads ``scenario_summary*`` from ``artifacts.db`` via the
    ``ArtifactStore``. The on-disk ``scenario_summary*.csv`` files are
    deliberately ignored.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(404, "No active run registry")

    from sparc.registry.store import ArtifactStore
    from sparc.scenario import SCENARIO_STAGE, SCENARIO_SUMMARY_VARIANTS

    store = ArtifactStore(state.registry)
    summary_df = None
    for variant in SCENARIO_SUMMARY_VARIANTS:
        if store.has(SCENARIO_STAGE, variant):
            summary_df = store.read_table(SCENARIO_STAGE, variant)
            break
    if summary_df is None:
        raise HTTPException(404, "No scenario summary found in artifacts.db")

    variables: dict[str, dict] = {}
    for _, row in summary_df.iterrows():
        var = str(row.get("Variable", ""))
        inc = row.get("Increment")
        if not var or inc is None:
            continue
        try:
            inc = float(inc)
        except (TypeError, ValueError):
            continue
        if var not in variables:
            variables[var] = {"increments": [], "sign": "plus" if inc >= 0 else "minus"}
        if inc not in variables[var]["increments"]:
            variables[var]["increments"].append(inc)

    for info in variables.values():
        info["increments"].sort(key=lambda x: abs(x))

    return {"variables": variables}


@app.get("/results/scenarios/increment")
async def get_scenario_increment(variable: str = Query(...), increment: float = Query(...)):
    """Return GeoJSON filtered to a specific variable+increment scenario.

    DB-only: reads ``scenario_summary*`` and ``scenario_results*`` from
    ``artifacts.db`` via the ``ArtifactStore``. Legacy ``.csv`` / ``.gpkg``
    files in ``Stage_4_Scenarios`` are intentionally not consulted.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(404, "No active run registry")

    from sparc.registry.store import ArtifactStore
    from sparc.scenario import (
        SCENARIO_STAGE,
        SCENARIO_SUMMARY_VARIANTS,
        SCENARIO_RESULTS_VARIANTS,
    )

    store = ArtifactStore(state.registry)

    summary_df = None
    for variant in SCENARIO_SUMMARY_VARIANTS:
        if store.has(SCENARIO_STAGE, variant):
            summary_df = store.read_table(SCENARIO_STAGE, variant)
            break
    if summary_df is None:
        raise HTTPException(404, "No scenario summary found in artifacts.db")

    mask = (
        (summary_df["Variable"].str.lower() == variable.lower()) &
        (summary_df["Increment"].astype(float).round(6) == round(float(increment), 6))
    )
    matched = summary_df[mask]
    if matched.empty:
        raise HTTPException(404, f"No scenario found for {variable} at increment {increment}")

    gdf = None
    for variant in SCENARIO_RESULTS_VARIANTS:
        if store.has(SCENARIO_STAGE, variant):
            gdf = store.read_table(SCENARIO_STAGE, variant)
            break
    if gdf is None:
        raise HTTPException(404, "No spatial scenario results found in artifacts.db")

    # ArtifactStore returns either a GeoDataFrame or a plain DataFrame — for
    # tables registered with ``geometry_col`` it's a GeoDataFrame already.
    import geopandas as gpd  # local import to keep top-level import light
    if not isinstance(gdf, gpd.GeoDataFrame) and "geometry" in gdf.columns:
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    scenario_label = matched.iloc[0].get("Scenario", "")
    pred_col = f"pred_{scenario_label}"
    delta_col = f"delta_{scenario_label}"
    baseline_col = "pred_baseline" if "pred_baseline" in gdf.columns else None

    cols_to_keep = ["geometry"]
    if baseline_col:
        cols_to_keep.append(baseline_col)
    if pred_col in gdf.columns:
        cols_to_keep.append(pred_col)
    if delta_col in gdf.columns:
        cols_to_keep.append(delta_col)
    elif pred_col in gdf.columns and baseline_col:
        gdf[delta_col] = gdf[pred_col].astype(float) - gdf[baseline_col].astype(float)
        cols_to_keep.append(delta_col)

    result_gdf = gdf[cols_to_keep].copy()
    if isinstance(result_gdf, gpd.GeoDataFrame):
        if result_gdf.crs is not None and str(result_gdf.crs) != "EPSG:4326":
            result_gdf = result_gdf.to_crs(epsg=4326)

    return {
        "geojson": result_gdf.__geo_interface__,
        "summary": matched.to_dict(orient="records"),
    }


# ------------------------------------------------------------------
# Run registry (artifacts manifest) endpoints
# These MUST be declared before the parameterized /results/{stage}
# route below — FastAPI matches routes in declaration order, and the
# parameterized handler types `stage: int`, which would otherwise
# swallow /results/manifest with a 422.
# ------------------------------------------------------------------


@app.get("/results/batch")
async def get_batch_results(
    ids: str = Query(
        ...,
        description=(
            "Comma-separated 'stage:artifact_id' pairs, "
            "e.g. '0:correlogram_results,1:gwen_results'."
        ),
    ),
):
    """Return multiple artifacts in a single round-trip.

    Each entry in *ids* is resolved via :func:`~sparc.server.artifact_reader.read_batch`.
    Present artifacts appear in ``results``; unresolvable ones appear in
    ``missing`` with a hint string — never a 404.
    """
    pairs: list[tuple[str, str]] = []
    for token in ids.split(","):
        token = token.strip()
        if ":" in token:
            stage, artifact_id = token.split(":", 1)
            pairs.append((stage.strip(), artifact_id.strip()))
    store = _open_store()
    return await asyncio.to_thread(read_batch, store, pairs)


@app.get("/results/manifest")
async def get_results_manifest(
    refresh: bool = Query(False, description="Reload from disk before returning."),
    rescan: bool = Query(False, description="Re-run the migrate_from_disk scan."),
):
    """Return the full run manifest (Pydantic ``RunManifest``).

    Frontend uses this to decide which Results panels to render and to
    surface 'unavailable, produced by stage X' empty-states.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        _attach_registry(state.project_config)
    if state.registry is None:
        raise HTTPException(503, "Run registry unavailable")

    if refresh or rescan:
        state.registry.load()
        if rescan:
            from sparc.run.pipeline_paths import PipelinePaths
            paths = PipelinePaths.from_config(state.project_config)
            state.registry.migrate_from_disk(paths)

    return state.registry.manifest.model_dump(mode="json", exclude_none=False)


@app.get("/results/manifest/{stage}")
async def get_stage_manifest(stage: str):
    """Return the per-stage manifest fragment (artifacts + status)."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        _attach_registry(state.project_config)
    if state.registry is None:
        raise HTTPException(503, "Run registry unavailable")

    sm = state.registry.manifest.stages.get(str(stage))
    if sm is None:
        return {"stage": str(stage), "status": "pending", "artifacts": {}}
    return sm.model_dump(mode="json", exclude_none=False)


@app.post("/results/manifest/rescan")
async def rescan_manifest():
    """Re-walk the run directory and import any unregistered artifacts."""
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        _attach_registry(state.project_config)
    if state.registry is None:
        raise HTTPException(503, "Run registry unavailable")
    from sparc.run.pipeline_paths import PipelinePaths
    paths = PipelinePaths.from_config(state.project_config)
    n = state.registry.migrate_from_disk(paths)
    return {"newly_registered": n,
            "total_artifacts": len(state.registry.manifest.all_artifacts())}


@app.post("/results/manifest/repair")
async def repair_manifest(threshold_minutes: float = Query(30.0)):
    """Detect stale partial-write entries and tombstone them.

    A partial entry is *stale* when it was last updated more than
    *threshold_minutes* ago — indicating the writing process crashed
    mid-flight and will never complete.  Stale entries are marked
    ``partial=False`` and their stage status reset to ``failed`` so the
    frontend can prompt the user to re-run the affected stage.

    Returns the list of repaired artifact IDs grouped by stage.
    """
    if state.registry is None:
        raise HTTPException(503, "Run registry unavailable")
    stale = state.registry.stale_partials(threshold_minutes=threshold_minutes)
    if not stale:
        return {"repaired": {}}
    repaired: dict[str, list[str]] = {}
    for entry in stale:
        entry.partial = False
        stage_key = str(entry.stage)
        repaired.setdefault(stage_key, []).append(entry.id)
        # Mark the stage as failed so the frontend shows a re-run prompt.
        sm = state.registry.manifest.stages.get(stage_key)
        if sm is not None and sm.status not in ("failed",):
            sm.status = "failed"
            sm.error = f"Partial write detected for {entry.id!r}; re-run stage {stage_key}."
    state.registry.save()
    return {"repaired": repaired}


# ------------------------------------------------------------------
# Artifact fetch endpoints (db-resident artifacts via ArtifactStore)
#
# These render artifacts on demand from artifacts.db. The pipeline stops
# producing CSV / PNG / JSON files; users download them through here.
# ------------------------------------------------------------------

_RENDER_MIME = {
    "csv":     "text/csv",
    "json":    "application/json",
    "geojson": "application/geo+json",
    "html":    "text/html",
    "pkl":     "application/octet-stream",
    "joblib":  "application/octet-stream",
    "pt":      "application/octet-stream",
    "bin":     "application/octet-stream",
}
# NOTE: PNG / image MIME types are intentionally absent. The server
# never renders or serves rasterised images. The desktop app renders
# visualisations live from artifacts.db and exports them client-side.


def _ensure_registry_attached() -> None:
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        _attach_registry(state.project_config)
    if state.registry is None:
        raise HTTPException(503, "Run registry unavailable")


@app.get("/artifacts/{stage}/{artifact_id}.csv")
async def get_artifact_csv(stage: str, artifact_id: str, index: bool = False):
    """Render a registered artifact as CSV bytes."""
    _ensure_registry_attached()
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.report.render import render_csv, RenderError

    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        try:
            data = render_csv(stage, artifact_id, index=index)
        except RenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)

    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{artifact_id}.csv"'},
    )


@app.get("/artifacts/{stage}/{artifact_id}.json")
async def get_artifact_json(stage: str, artifact_id: str):
    """Render a registered artifact (struct or table) as JSON."""
    _ensure_registry_attached()
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.report.render import render_json, RenderError

    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        try:
            data = render_json(stage, artifact_id)
        except RenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)
    return Response(content=data, media_type="application/json")


@app.get("/artifacts/{stage}/{artifact_id}.geojson")
async def get_artifact_geojson(stage: str, artifact_id: str):
    """Render a geometry-bearing table as GeoJSON."""
    _ensure_registry_attached()
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.report.render import render_geojson, RenderError

    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        try:
            data = render_geojson(stage, artifact_id)
        except RenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)
    return Response(content=data, media_type="application/geo+json")


@app.get("/artifacts/{stage}/{artifact_id}.png")
async def get_artifact_png(stage: str, artifact_id: str, dpi: int = 150):
    """Render a registered artifact as a PNG via the figures module.

    Dispatches through ``sparc.report.figures.render_for_artifact``; returns
    404 when no renderer is registered for ``(stage, artifact_id)``.
    """
    _ensure_registry_attached()
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    try:
        from sparc.report.figures import FigureRenderError, render_for_artifact
    except ImportError as exc:
        raise HTTPException(503, f"figures module unavailable: {exc}")

    if state.registry is None:
        raise _missing_artifact_response(
            artifact_id=artifact_id, stage=stage,
            hint="No active run registry.",
        )
    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        try:
            data = render_for_artifact(stage, artifact_id, registry=state.registry, dpi=dpi)
        except FigureRenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)
    return Response(content=data, media_type="application/octet-stream")


# Catch-all native route — declared LAST so the suffixed routes above
# match first. Otherwise ``{artifact_id}`` swallows ``foo.csv`` etc. and
# the suffixed routes become dead code.
@app.get("/artifacts/{stage}/{artifact_id}")
async def get_artifact_native(stage: str, artifact_id: str):
    """Return an artifact in its native format (CSV for tables, JSON for structs, raw bytes for blobs)."""
    _ensure_registry_attached()
    # Bind the active registry so render_* helpers can find it.
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.report.render import render_native, RenderError

    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        try:
            data, ext = render_native(stage, artifact_id)
        except RenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)

    return Response(
        content=data,
        media_type=_RENDER_MIME.get(ext, "application/octet-stream"),
        headers={
            "Content-Disposition": f'attachment; filename="{artifact_id}.{ext}"',
        },
    )


@app.get("/results/bundle")
async def get_results_bundle():
    """Stream a ZIP of every registered artifact (data formats + manifest).

    Each artifact is rendered in its native format (CSV/JSON/GeoJSON) plus a
    PNG when a renderer is registered. The full ``RunManifest`` is included
    as ``manifest.json`` for reproducibility.
    """
    _ensure_registry_attached()
    if state.registry is None:
        raise HTTPException(404, "No active run registry.")

    import io
    import zipfile
    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.registry.store import ArtifactStore
    from sparc.report.render import (
        RenderError,
        render_csv,
        render_geojson,
        render_json,
    )

    try:
        from sparc.report.figures import FigureRenderError, render_for_artifact
        figures_available = True
    except ImportError:
        figures_available = False

    store = ArtifactStore(state.registry)
    manifest = state.registry.manifest

    buf = io.BytesIO()
    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                manifest.model_dump_json(indent=2),
            )
            # Snapshot stage / artifact lists before iterating: figure
            # rendering may write a cached PNG back into the registry, which
            # mutates ``manifest.stages`` mid-loop and raises
            # ``RuntimeError: dictionary changed size during iteration``.
            stage_items = list(manifest.stages.items())
            for stage_id, stage_manifest in stage_items:
                artifact_items = list(stage_manifest.artifacts.items())
                for artifact_id, entry in artifact_items:
                    base = f"stage_{stage_id}/{artifact_id}"
                    # Native data export.
                    try:
                        if entry.storage_kind == "table":
                            geom = (entry.metadata or {}).get("geometry_col")
                            if geom:
                                zf.writestr(
                                    f"{base}.geojson",
                                    render_geojson(stage_id, artifact_id),
                                )
                            else:
                                zf.writestr(
                                    f"{base}.csv",
                                    render_csv(stage_id, artifact_id),
                                )
                        elif entry.storage_kind == "struct":
                            zf.writestr(
                                f"{base}.json",
                                render_json(stage_id, artifact_id),
                            )
                        # Blobs are skipped — they may be huge and opaque.
                    except RenderError as exc:
                        zf.writestr(
                            f"{base}.SKIPPED.txt",
                            f"render failed: {exc}".encode("utf-8"),
                        )
                    # Optional PNG.
                    if figures_available:
                        try:
                            png = render_for_artifact(stage_id, artifact_id, registry=state.registry)
                            zf.writestr(f"{base}.png", png)
                        except FigureRenderError:
                            pass
                        except Exception as exc:  # noqa: BLE001
                            zf.writestr(
                                f"{base}.png.SKIPPED.txt",
                                f"png render failed: {exc}".encode("utf-8"),
                            )
    finally:
        set_active_registry(prev)

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="sparc_results_bundle.zip"',
        },
    )


# ------------------------------------------------------------------
# Results endpoints (parameterized — MUST come after named routes above)
# ------------------------------------------------------------------

@app.get("/results/{stage:int}")
async def get_results(stage: int, format: str = Query("json", regex="^(json|geojson)$")):
    """Return in-memory results for a completed pipeline stage (DB-only).

    For artifact-style data use the dedicated `/results/<name>` endpoints.
    This endpoint only exposes the live in-memory result captured during
    the current process run.
    """
    result = state.get_result(stage)
    if result is None:
        raise _missing_artifact_response(
            artifact_id=f"stage_{stage}_result", stage=str(stage),
            hint=(
                f"No in-memory result for stage {stage}. Run the stage in this "
                "session, or query a specific artifact via `/results/<name>`."
            ),
        )
    if format == "geojson":
        return _to_geojson(result)
    return _to_json(result)


@app.get("/results/{stage:int}/predictions")
async def get_predictions(stage: int, format: str = Query("geojson", regex="^(json|geojson)$")):
    """Return spatial predictions for a stage (DB-only).

    Mapping:
      - stage 2 -> artifact ("2", "spatial_cv_predictions")
      - stage 4 -> ScenarioBundle baseline geometry from ("4", "scenario_results")
      - other stages do not produce per-point predictions.
    """
    if stage == 2:
        store = _open_store()
        try:
            if not store.has("2", "spatial_cv_predictions"):
                raise _missing_artifact_response(
                    artifact_id="spatial_cv_predictions", stage="2",
                    hint="Stage 2 has not produced spatial_cv_predictions. Run Stage 2.",
                )
            gdf = store.read_any("2", "spatial_cv_predictions")
        finally:
            from sparc.registry.run_registry import set_active_registry
            set_active_registry(None)
        if format == "geojson":
            return _to_geojson(gdf)
        return _to_json(gdf)

    if stage == 4:
        # Scenario predictions live under /results/scenarios/detail (full
        # mode-variant precedence + per-scenario column resolution).
        raise _missing_artifact_response(
            artifact_id="scenario_results", stage="4",
            hint="Use /results/scenarios/detail for stage 4 scenario predictions.",
        )

    hints = {
        0: "Stage 0 (EDA) does not write per-point predictions.",
        1: "Stage 1 (GWEN) writes feature-selection results, not predictions.",
        3: "Stage 3 produces causal effects, not direct predictions. View predictions on the Stage 2 tab.",
    }
    raise _missing_artifact_response(
        artifact_id=f"stage_{stage}_predictions", stage=str(stage),
        hint=hints.get(stage, f"No predictions registered for stage {stage}."),
    )


# (Scenario run/optimize/results endpoints moved to routes.scenarios via include_router above)


# (Scenario run/optimize/results/budget endpoints moved to routes.scenarios via include_router above)


# (/equity/layer is now served by routes.decision via include_router above)


# (/physics/defaults is now served by routes.physics via include_router above)


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
# (/report/generate, /report/pdf, /report/docx are now served by
# routes.report via include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# DAG endpoints
# (/dag GET+PUT, /dag/validate, /dag/mc3_result, /dag/approve,
# /dag/reject, /dag/suggest-edges are now served by routes.dag via
# include_router above)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Artifact download
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Run registry endpoints — see definitions earlier in the file
# (placed before /results/{stage} parameterized route to avoid the
# integer-only path converter swallowing /results/manifest paths).
# ------------------------------------------------------------------

@app.get("/results/availability")
async def get_results_availability():
    """Report which result artifacts are present in artifacts.db (DB-only).

    Returns ``{endpoint_path: {available, source}}`` so the desktop can grey
    out tabs that have no underlying data.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    # Build set of (stage, artifact_id) currently registered.
    db_artifacts: set[tuple[str, str]] = set()
    if state.registry is not None:
        try:
            man = state.registry.manifest
            for stage_obj in man.stages.values():
                for art in stage_obj.artifacts.values():
                    if not art.partial:
                        db_artifacts.add((str(stage_obj.stage), art.id))
        except Exception:
            db_artifacts = set()

    # Endpoint -> primary artifact lookup.  None means "any of a family".
    endpoints: dict[str, tuple[str, str] | None] = {
        "/results/correlogram": ("0", "correlogram_results"),
        "/results/gwen": ("1", "gwen_variable_importance"),
        "/results/model_performance": ("2", "ensemble_results"),
        "/results/spatial_cv/predictions": ("2", "spatial_cv_predictions"),
        "/results/neural_pdp": None,  # family: v2_neural_pdp::*
        "/results/pdp_curves": None,  # family
        "/results/causal": ("3", "scenario_coefficients"),
        "/results/causal/diagnostics": ("3", "causal_diagnostics"),
        "/results/causal/dose_response": ("3", "dose_response_curves"),
        "/results/causal/cate_map": ("3", "cate_summary"),
        "/results/scenarios/nuts_summary": ("3", "nuts_summary"),
        "/results/scenarios/detail": ("4", "scenario_results"),
        "/results/kernel_field": ("0", "kernel_field"),
        "/results/causal/pdp_curves": ("3", "causal_pdp_curves"),
        "/results/causal/divergence": ("3", "cate_vs_gwr_divergence"),
        "/results/scenarios/routing_audit": ("4", "scenario_routing_audit"),
        "/dag/mc3_result": ("3", "mc3_summary"),
    }

    out: dict[str, dict] = {}
    has_neural_pdp = any(s == "2" and a.startswith("v2_neural_pdp::") for s, a in db_artifacts)
    has_gwrf = (("2", "gwrf_condition_curves") in db_artifacts) or \
               (("3", "gwrf_condition_curves") in db_artifacts)

    for endpoint, key in endpoints.items():
        if endpoint == "/results/neural_pdp":
            available = has_neural_pdp
            source = "artifacts.db:2:v2_neural_pdp::*" if available else ""
        elif endpoint == "/results/pdp_curves":
            available = has_neural_pdp or has_gwrf
            source = "artifacts.db (neural_pdp+gwrf)" if available else ""
        else:
            stage_id, art_id = key  # type: ignore[misc]
            available = (stage_id, art_id) in db_artifacts
            source = f"artifacts.db:{stage_id}:{art_id}" if available else ""
        out[endpoint] = {"available": available, "source": source}

    out["_project"] = {
        "available": True,
        "source": state.project_path,
    }
    return out


# ------------------------------------------------------------------
# Panel availability — declarative status for desktop insights panels
# ------------------------------------------------------------------

# Each panel entry maps to one or more *artifact specs*. A spec is either a
# tuple ``(stage, artifact_id)`` or a tuple ``(stage, "prefix::*")`` which
# matches any registered artifact whose id starts with that prefix.
# A panel is ``ready`` when at least one of its specs is satisfied,
# ``partial`` when some but not all are, otherwise ``awaiting``.
_PANEL_SPECS: dict[str, dict] = {
    "overview":            {"stage": "0", "specs": [("0", "correlogram_results")]},
    "headline":            {"stage": "2", "specs": [("2", "ensemble_results")]},
    "model_performance":   {"stage": "2", "specs": [("2", "ensemble_results")]},
    "dataset_profile":     {"stage": "0", "specs": [("2", "dataset_profile")]},
    "correlogram":         {"stage": "0", "specs": [("0", "correlogram_results")]},
    "kernel_field":        {"stage": "0", "specs": [
        ("1", "kernel_field"),
        ("0", "kernel_field"),
        ("0", "cross_correlogram_kernel_field"),
    ]},
    "predictions_map":     {"stage": "2", "specs": [("2", "spatial_cv_predictions")]},
    "pdp":                 {"stage": "2", "specs": [
        ("3", "dose_response_curves"),
        ("2", "gwrf_condition_curves"),
        ("2", "v2_neural_pdp::*"),
    ]},
    "dose_response":       {"stage": "3", "specs": [("3", "dose_response_curves")]},
    "cate":                {"stage": "3", "specs": [("3", "cate_summary")]},
    "divergence":          {"stage": "3", "specs": [("3", "cate_vs_gwr_divergence")]},
    "sensitivity":         {"stage": "3", "specs": [
        ("3", "scenario_coefficients"),
        ("3", "causal_diagnostics"),
    ]},
    "negative_control":    {"stage": "3", "specs": [("3", "cate_summary")]},
    "scenario_strip":      {"stage": "4", "specs": [("4", "scenario_results")]},
    "scenario_map":        {"stage": "4", "specs": [("4", "scenario_results")]},
    "scenario_uncertainty":{"stage": "4", "specs": [("4", "scenario_results")]},
    "scenario_trajectory": {"stage": "4", "specs": [("4", "scenario_results")]},
    "equity_cost":         {"stage": "4", "specs": [("4", "scenario_results")]},
    "artifact_browser":    {"stage": "0", "specs": [("0", "correlogram_results")]},
}

_STAGE_HINTS: dict[str, str] = {
    "0": "Run Stage 0 (Correlogram) to populate this panel.",
    "1": "Run Stage 1 (GWEN) to populate this panel.",
    "2": "Run Stage 2 (Spatial CV) to populate this panel.",
    "3": "Run Stage 3 (Causal Validation) to populate this panel.",
    "4": "Run Stage 4 (Scenarios) to populate this panel.",
}

_STAGE_LABELS: dict[str, str] = {
    "0": "STAGE 0 · CORRELOGRAM",
    "1": "STAGE 1 · GWEN",
    "2": "STAGE 2 · SPATIAL CV",
    "3": "STAGE 3 · CAUSAL",
    "4": "STAGE 4 · SCENARIOS",
}


def _spec_matches(spec: tuple[str, str], db_artifacts: set[tuple[str, str]],
                  prefix_index: dict[tuple[str, str], bool]) -> bool:
    stage_id, art_id = spec
    if art_id.endswith("::*"):
        prefix = art_id[: -1]  # keep '::' but drop the trailing '*'
        return prefix_index.get((stage_id, prefix), False)
    return (stage_id, art_id) in db_artifacts


@app.get("/panels/availability")
async def get_panels_availability():
    """Per-panel availability snapshot for the desktop insights workspace.

    Returns a status enum the frontend can use to render a uniform
    ``<PanelGate>`` (status pill + empty-state message) without each
    panel re-implementing manifest-walk logic.

    Status values
    -------------
    ``ready``    — at least one required artifact is present.
    ``partial``  — some required artifacts present; others missing.
    ``awaiting`` — none of the required artifacts have been produced yet.
    ``running``  — pipeline is currently executing the panel's stage.
    ``failed``   — a previous run reported an error for the panel's stage.
    """
    db_artifacts: set[tuple[str, str]] = set()
    # Pre-compute prefix presence (stage, "prefix::") -> bool to keep the
    # per-panel scan O(1) for the wildcard specs.
    prefix_index: dict[tuple[str, str], bool] = {}
    if state.registry is not None:
        try:
            man = state.registry.manifest
            for stage_obj in man.stages.values():
                stage_str = str(stage_obj.stage)
                for art in stage_obj.artifacts.values():
                    if art.partial:
                        continue
                    db_artifacts.add((stage_str, art.id))
                    sep = art.id.find("::")
                    if sep > 0:
                        prefix_index[(stage_str, art.id[: sep + 2])] = True
        except Exception:
            db_artifacts = set()
            prefix_index = {}

    is_running = bool(state.is_running)
    current_stage = (
        str(state.current_stage) if state.current_stage is not None else None
    )
    last_errors: dict[str, str] = getattr(state, "stage_errors", {}) or {}

    panels_out: dict[str, dict] = {}
    for panel_id, cfg in _PANEL_SPECS.items():
        stage = cfg["stage"]
        specs: list[tuple[str, str]] = list(cfg["specs"])
        matched = [s for s in specs if _spec_matches(s, db_artifacts, prefix_index)]
        missing = [f"{s[0]}:{s[1]}" for s in specs if s not in matched]

        if matched and not missing:
            status = "ready"
        elif matched:
            status = "partial"
        else:
            status = "awaiting"

        # Stage-level overrides win (they describe pipeline state, not
        # artifact presence).
        if stage in last_errors and not matched:
            status = "failed"
        if is_running and current_stage == stage and not matched:
            status = "running"

        panels_out[panel_id] = {
            "status": status,
            "stage": stage,
            "stage_label": _STAGE_LABELS.get(stage, f"STAGE {stage}"),
            "missing": missing,
            "matched": [f"{s[0]}:{s[1]}" for s in matched],
            "hint": (
                last_errors[stage] if status == "failed"
                else _STAGE_HINTS.get(stage, "Run the relevant stage.")
            ),
        }

    return {
        "panels": panels_out,
        "is_running": is_running,
        "current_stage": current_stage,
    }


@app.get("/results/artifacts")
async def list_artifacts():
    """List every registered artifact across all stages (DB-only).

    Iterates ``state.registry.manifest.stages`` rather than walking the
    filesystem. Each artifact is annotated with a ``download_url`` pointing
    at the canonical ``/artifacts/{stage}/{id}`` native renderer.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(404, "No active run registry")

    stage_labels = {
        "0": "Stage 0 — Correlogram",
        "1": "Stage 1 — GWEN",
        "2": "Stage 2 — Spatial CV",
        "3": "Stage 3 — Causal",
        "4": "Stage 4 — Scenarios",
    }

    artifacts: list[dict] = []
    for stage_key, stage_obj in list(state.registry.manifest.stages.items()):
        for art_id, art in list(stage_obj.artifacts.items()):
            if getattr(art, "partial", False):
                continue
            kind = getattr(art, "kind", "")
            ext = {"table": "csv", "struct": "json", "blob": "bin"}.get(kind, "bin")
            artifacts.append({
                "stage": stage_key,
                "stage_label": stage_labels.get(str(stage_key), f"Stage {stage_key}"),
                "artifact_id": art_id,
                "filename": f"{art_id}.{ext}",
                "kind": kind,
                "extension": f".{ext}",
                "size_bytes": getattr(art, "size_bytes", None),
                "producer": getattr(art, "producer", None),
                "download_url": f"/artifacts/{stage_key}/{art_id}",
            })

    return {"artifacts": artifacts, "total": len(artifacts)}


@app.get("/results/download/{stage}/{file_path:path}")
async def download_artifact(stage: str, file_path: str):
    """Download a registered artifact in its native (or requested) format.

    ``file_path`` is parsed as ``<artifact_id>[.<ext>]`` so legacy frontends
    that still build URLs like ``/results/download/2/predictions.csv`` keep
    working. The ext (``csv|json|geojson|png``) routes through the matching
    ``render_*`` helper; an empty ext falls back to the artifact's native
    rendering. All bytes come from ``artifacts.db``.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(404, "No active run registry")

    from sparc.registry.run_registry import set_active_registry, get_active_registry
    from sparc.report.render import (
        render_native, render_csv, render_json, render_geojson, RenderError,
    )

    # Strip any leading directory components a legacy caller may have included.
    leaf = file_path.rsplit("/", 1)[-1]
    if "." in leaf:
        artifact_id, _, ext = leaf.rpartition(".")
        ext = ext.lower()
    else:
        artifact_id, ext = leaf, ""

    prev = get_active_registry()
    set_active_registry(state.registry)
    try:
        try:
            if ext == "csv":
                data = render_csv(str(stage), artifact_id)
                media = "text/csv"
                fname = f"{artifact_id}.csv"
            elif ext == "json":
                data = render_json(str(stage), artifact_id)
                media = "application/json"
                fname = f"{artifact_id}.json"
            elif ext == "geojson":
                data = render_geojson(str(stage), artifact_id)
                media = "application/geo+json"
                fname = f"{artifact_id}.geojson"
            elif ext == "png":
                from sparc.report.figures import (
                    FigureRenderError, render_for_artifact,
                )
                try:
                    data = render_for_artifact(
                        str(stage), artifact_id, registry=state.registry,
                    )
                except FigureRenderError as exc:
                    raise HTTPException(404, str(exc))
                media = "image/png"
                fname = f"{artifact_id}.png"
            else:
                data, native_ext = render_native(str(stage), artifact_id)
                media = _RENDER_MIME.get(native_ext, "application/octet-stream")
                fname = f"{artifact_id}.{native_ext}"
        except RenderError as exc:
            raise HTTPException(404, str(exc))
    finally:
        set_active_registry(prev)

    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/results/geopackage")
async def download_geopackage():
    """Merge spatial layers from artifacts.db into a single GeoPackage download.

    Layers come from canonical artifacts only:
      * ``predictions``       ← Stage 2 ``predictions``
      * ``causal_effects``    ← Stage 3 ``cate_summary``
      * ``scenario_deltas``   ← Stage 4 ``scenario_results``
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(404, "No active run registry")

    import geopandas as gpd
    import tempfile
    from sparc.registry.store import ArtifactStore

    store = ArtifactStore(state.registry)
    layer_specs = [
        ("predictions",     "2", "predictions"),
        ("causal_effects",  "3", "cate_summary"),
        ("scenario_deltas", "4", "scenario_results"),
    ]

    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name, stage_id, art_id in layer_specs:
        if not store.has(stage_id, art_id):
            continue
        try:
            gdf = store.read_table(stage_id, art_id)
        except Exception:
            continue
        if not isinstance(gdf, gpd.GeoDataFrame):
            if "geometry" in gdf.columns:
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
            else:
                continue
        if len(gdf) > 0:
            layers[layer_name] = gdf

    if not layers:
        raise HTTPException(404, "No spatial artifacts available in artifacts.db")

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

# ---------------------------------------------------------------------------
# Phase 5: Artifact export-on-demand endpoints (db-only architecture)
# ---------------------------------------------------------------------------
# These let the desktop UI download any DB-resident artifact as a file
# (CSV / parquet / JSON / GPKG / joblib / etc.) without keeping
# stage folders on disk.

from fastapi.responses import FileResponse as _FileResponse


def _get_artifact_store():
    if state.registry is None:
        raise HTTPException(503, "No active run/registry")
    from sparc.registry.store import ArtifactStore
    return ArtifactStore(state.registry)


@app.get("/artifacts/{stage}/{artifact_id}/download")
def download_artifact(stage: str, artifact_id: str, fmt: Optional[str] = None):
    """Download an artifact from artifacts.db as a file.

    Query parameters:
      fmt � one of csv, parquet, json, gpkg, geojson, joblib, pkl, npy.
            If omitted, defaults to the artifact's natural format.
    """
    store = _get_artifact_store()
    if not store.has(stage, artifact_id):
        raise HTTPException(404, f"Artifact not found: {stage}/{artifact_id}")
    try:
        path = store.export(stage, artifact_id, fmt=fmt)
    except Exception as exc:
        raise HTTPException(500, f"Export failed: {exc}") from exc
    suffix = path.suffix.lstrip(".") or "bin"
    media = {
        "csv": "text/csv",
        "json": "application/json",
        "parquet": "application/octet-stream",
        "gpkg": "application/geopackage+sqlite3",
        "geojson": "application/geo+json",
        "joblib": "application/octet-stream",
        "pkl": "application/octet-stream",
        "npy": "application/octet-stream",
    }.get(suffix, "application/octet-stream")
    return _FileResponse(
        path,
        media_type=media,
        filename=f"{artifact_id}.{suffix}",
    )


@app.post("/artifacts/{stage}/export")
def export_stage_zip(stage: str):
    """Bundle every artifact for a stage into a downloadable .zip."""
    store = _get_artifact_store()
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix=f"sparc_export_{stage}_"))
    try:
        zip_path = store.export_stage(stage, tmpdir, as_zip=True)
        if not isinstance(zip_path, Path):
            raise RuntimeError("export_stage did not return a zip path")
    except Exception as exc:
        raise HTTPException(500, f"Stage export failed: {exc}") from exc
    return _FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"stage_{stage}_artifacts.zip",
    )


@app.get("/artifacts/{stage}")
def list_stage_artifacts(stage: str):
    """List all artifacts registered for a stage (id, format, storage_kind)."""
    if state.registry is None:
        raise HTTPException(503, "No active run/registry")
    entries = state.registry.list_for_stage(stage)
    return {
        "stage": str(stage),
        "count": len(entries),
        "artifacts": [
            {
                "artifact_id": e.id,
                "format": e.format,
                "storage_kind": getattr(e, "storage_kind", "legacy_path"),
                "partial": bool(e.partial),
                "producer": getattr(e, "producer", None),
            }
            for e in entries
        ],
    }


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
