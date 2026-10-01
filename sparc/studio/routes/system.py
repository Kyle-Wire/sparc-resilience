"""Auth and system endpoints (api.md §2): ``/auth``, health, meta, event schema, settings, system, netcheck,
storage and shutdown."""

from __future__ import annotations

import asyncio
import hmac
import importlib.metadata
import os
import platform
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import sparc
from sparc.studio import __version__
from sparc.studio.app import STATIC_DIR, StudioContext, get_ctx
from sparc.studio.errors import ApiError
from sparc.studio.events import event_json_schema
from sparc.studio.jobs.eta import _cpu_model, host_id
from sparc.studio.jobs.kinds import KINDS
from sparc.studio.meta import build_meta
from sparc.studio.schemas.common import (
    FreedBytes,
    Health,
    Meta,
    NetcheckRequest,
    NetcheckResult,
    Ok,
    ShutdownRequest,
    StorageInfo,
    SystemInfo,
)
from sparc.studio.security import COOKIE_NAME, safe_next, safe_path, session_value
from sparc.studio.settings import Settings, SettingsPatch, save_settings
from sparc.studio.workspace import dir_size, read_json, utc_iso

router = APIRouter(tags=["system"])
root_router = APIRouter(tags=["auth"])

_UNAUTHORIZED_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SPARC Studio</title></head>
<body style="font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>Link expired</h1><p>This sign-in link is not valid for the running server. Open the link that
<code>sparc studio</code> printed when it started (it changes on every launch).</p></body></html>"""


@root_router.get("/auth", response_class=RedirectResponse, status_code=302,
                 responses={302: {"description": "Cookie set; redirect to `next`"},
                            401: {"description": "Wrong token (HTML)", "content": {"text/html": {}}}})
async def auth(request: Request, t: str = Query(..., description="launch token"),
               next: str | None = Query(None, description="same-origin path to open")):
    """Exchange the launch token for the HttpOnly session cookie (api.md §0.2)."""
    sctx: StudioContext = request.app.state.studio
    if not hmac.compare_digest(t.encode(), sctx.token.encode()):
        return HTMLResponse(_UNAUTHORIZED_HTML, status_code=401)
    resp = RedirectResponse(safe_next(next), status_code=302)
    resp.set_cookie(COOKIE_NAME, session_value(sctx.token), httponly=True, samesite="strict", path="/",
                    secure=request.url.scheme == "https")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@router.get("/health", response_model=Health)
async def health(sctx: StudioContext = Depends(get_ctx)):
    """Liveness (no auth)."""
    active = sctx.jobs.active_count() if sctx.jobs is not None else 0
    return {"ok": True, "version": __version__, "workspace": str(sctx.workspace.root), "pid": sctx.pid,
            "started_utc": sctx.started_utc, "active_jobs": active, "engine": {"state": sctx.engine_state()}}


@router.get("/meta", response_model=Meta)
async def meta(sctx: StudioContext = Depends(get_ctx)):
    """Stage, job-kind, output, palette and unit-cost catalogs."""
    return await asyncio.to_thread(build_meta, include_test_kinds=sctx.test_kinds())


@router.get("/meta/event-schema", response_model=dict)
async def event_schema():
    """JSON Schema of the per-job event union (api.md §17)."""
    return event_json_schema()


@router.get("/settings", response_model=Settings)
async def get_settings(sctx: StudioContext = Depends(get_ctx)):
    return sctx.reload_settings()


@router.put("/settings", response_model=Settings)
async def put_settings(patch: SettingsPatch, sctx: StudioContext = Depends(get_ctx)):
    """Partial update; returns the full settings.  ``422 validation`` on an invalid combination."""
    new = await asyncio.to_thread(save_settings, sctx.db, patch.model_dump(exclude_unset=True))
    sctx.reload_settings()
    if sctx.jobs is not None:
        sctx.jobs.wake()
    return new


_WS_BYTES: dict[str, tuple[float, int]] = {}


def _workspace_bytes(root: Path, ttl: float = 60.0) -> int:
    key = str(root)
    hit = _WS_BYTES.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    n = dir_size(root)
    _WS_BYTES[key] = (now, n)
    return n


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _system(sctx: StudioContext) -> dict:
    import psutil

    vm = psutil.virtual_memory()
    free = shutil.disk_usage(sctx.workspace.root).free
    build = read_json(STATIC_DIR / "BUILD_INFO.json") if sctx.config.static_dir is None else read_json(
        Path(sctx.config.static_dir) / "BUILD_INFO.json")
    web = None
    if isinstance(build, dict) and build.get("src_sha256"):
        web = {"src_sha256": str(build["src_sha256"]), "vite": build.get("vite"), "react": build.get("react")}
    return {
        "cpu_count": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        "cpu_model": _cpu_model(),
        "mem_total_gb": round(vm.total / 2 ** 30, 2), "mem_available_gb": round(vm.available / 2 ** 30, 2),
        "disk_free_gb": round(free / 2 ** 30, 2), "workspace": str(sctx.workspace.root),
        "workspace_bytes": _workspace_bytes(sctx.workspace.root), "host_id": host_id(),
        "versions": {"python": platform.python_version(), "sparc": getattr(sparc, "__version__", "?"),
                     "numpy": _version("numpy"), "pandas": _version("pandas"),
                     "torch": _version("torch"),           # package metadata only: never import torch here
                     "fastapi": _version("fastapi")},
        "web_build": web,
    }


@router.get("/system", response_model=SystemInfo)
async def system(sctx: StudioContext = Depends(get_ctx)):
    """Machine, workspace and version facts (``torch`` from package metadata, never imported)."""
    return await asyncio.to_thread(_system, sctx)


def _default_hosts(sctx: StudioContext) -> list[str]:
    hosts: list[str] = []
    for k in KINDS.values():
        for h in k.network_hosts:
            if h not in hosts:
                hosts.append(h)
    url = sctx.settings().basemap_url
    if url:
        netloc = urlsplit(url.replace("{s}", "a")).netloc
        if netloc and netloc not in hosts:
            hosts.append(netloc)
    return hosts


@router.post("/system/netcheck", response_model=NetcheckResult)
async def netcheck(body: NetcheckRequest | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """TCP reachability of the network hosts job kinds use (cached 10 min per host)."""
    hosts = (body.hosts if body and body.hosts else None) or _default_hosts(sctx)
    results = await asyncio.to_thread(sctx.jobs.netcheck, hosts)
    return {"results": results}


def _storage(sctx: StudioContext) -> dict:
    ws = sctx.workspace
    cache = []
    if ws.cache_dir.is_dir():
        for p in sorted(ws.cache_dir.iterdir()):
            try:
                mtime = utc_iso(p.stat().st_mtime)
            except OSError:
                mtime = None
            cache.append({"name": p.name, "bytes": dir_size(p), "mtime": mtime})
    runs = []
    for r in sctx.db.fetchall("SELECT id, label, project_id, run_dir FROM runs ORDER BY created_utc DESC"):
        d = Path(r["run_dir"])
        ck = d / "checkpoint.pkl"
        runs.append({"run_id": r["id"], "label": r.get("label"), "project_id": r.get("project_id"),
                     "outputs_bytes": dir_size(d, exclude={"checkpoint.pkl"}),
                     "checkpoint_bytes": ck.stat().st_size if ck.is_file() else 0})
    studies = [{"study_id": s["id"], "bytes": dir_size(s["out_dir"]) if s.get("out_dir") else 0}
               for s in sctx.db.fetchall("SELECT id, out_dir FROM studies ORDER BY created_utc DESC")]
    return {"workspace_bytes": _workspace_bytes(ws.root, ttl=0.0), "free_bytes": shutil.disk_usage(ws.root).free,
            "cache": cache, "runs": runs, "studies": studies, "jobs_bytes": dir_size(ws.jobs_dir)}


@router.get("/storage", response_model=StorageInfo)
async def storage(sctx: StudioContext = Depends(get_ctx)):
    """Disk use of the workspace: cache entries, runs (outputs vs checkpoint), studies, job logs."""
    return await asyncio.to_thread(_storage, sctx)


@router.delete("/storage/cache/{name}", response_model=FreedBytes)
async def delete_cache(name: str, sctx: StudioContext = Depends(get_ctx)):
    """Delete one entry of ``<workspace>/cache``."""
    cache = sctx.workspace.cache_dir
    target = safe_path(name, cache)
    if target == cache.resolve() or target.parent != cache.resolve() or not target.exists():
        raise ApiError("not_found", f"no cache entry {name!r}")

    def _rm() -> int:
        n = dir_size(target)
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        return n

    return {"freed_bytes": await asyncio.to_thread(_rm)}


@router.post("/shutdown", response_model=Ok, status_code=202)
async def shutdown(body: ShutdownRequest | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Stop the server (jobs keep running unless ``stop_jobs``); broadcasts ``server_shutdown``."""
    sctx.stop_jobs = bool(body.stop_jobs) if body else False
    sctx.announce_shutdown()
    if sctx.request_shutdown is not None:
        asyncio.get_running_loop().call_later(0.2, sctx.request_shutdown)
    return {"ok": True}
