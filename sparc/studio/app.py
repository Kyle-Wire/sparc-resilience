"""The FastAPI application: ``create_app(config)`` (SPEC §4.1, §10.1–10.2, §10.9).

``create_app`` has no side effects on disk - everything that touches the
workspace happens in the lifespan, so ``--dump-openapi`` can build an app
just to render its schema.  Startup (in order):

1. create the workspace dirs, write the token file, migrate the database
   (``--reindex``: rebuild the index from disk), load the user settings;
2. ``scan`` hooks (the runs registry scans the workspace and watch roots);
3. ``JobManager.start()``: reattach live jobs, start the ResourceSampler and
   the scheduler;
4. ``ready`` hooks (the engine item reconnects to a live engine host).

Shutdown broadcasts ``server_shutdown``, runs the shutdown hooks (engine
host), flushes every tracker projection and leaves jobs running in their own
process groups unless ``--stop-jobs-on-exit`` / ``{stop_jobs: true}``.

Feature items plug in through registries only: route modules
(:mod:`sparc.studio.routes`), job-kind modules
(:mod:`sparc.studio.jobs.kinds`), executors, and the lifecycle hooks
:func:`startup_hook` / :func:`shutdown_hook` defined here.  Route code
reaches the server context with ``Depends(get_ctx)`` or
``request.app.state.studio``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse

from sparc.studio import __version__
from sparc.studio.db import Database
from sparc.studio.errors import ApiError, install_error_handlers
from sparc.studio.events import EventHub
from sparc.studio.jobs.kinds import load_kind_modules
from sparc.studio.routes import load_router_modules
from sparc.studio.security import (
    PathGuard,
    SecurityMiddleware,
    allowed_hosts,
    allowed_origins,
    generate_token,
    safe_path,
)
from sparc.studio.settings import Settings, StudioSettings, load_settings
from sparc.studio.workspace import utc_now, write_private

log = logging.getLogger("sparc.studio")

__all__ = ["StudioContext", "create_app", "get_ctx", "startup_hook", "shutdown_hook", "STATIC_DIR",
           "NOT_BUILT_HTML"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
NOT_BUILT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPARC Studio</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem;color:#1b1b1a}
code{background:#eeede8;padding:.1rem .3rem;border-radius:3px}</style></head>
<body><h1>SPARC Studio</h1>
<p><strong>Frontend not built</strong> &mdash; run <code>npm --prefix studio-web run build</code>.</p>
<p>The API is running: <a href="/api/health">/api/health</a>.</p></body></html>
"""

_STARTUP: list[tuple[str, int, str, Callable]] = []
_SHUTDOWN: list[tuple[int, str, Callable]] = []


def startup_hook(name: str, *, phase: str = "scan", order: int = 100) -> Callable[[Callable], Callable]:
    """Register ``fn(sctx)`` (sync or async) to run at startup.

    ``phase="scan"`` runs before the job manager starts (registry scans);
    ``phase="ready"`` after jobs are reattached (engine host reconnect).
    A failing hook is logged and never stops the server.
    """
    if phase not in ("scan", "ready"):
        raise ValueError("phase must be 'scan' or 'ready'")

    def deco(fn: Callable) -> Callable:
        _STARTUP[:] = [h for h in _STARTUP if h[2] != name]
        _STARTUP.append((phase, order, name, fn))
        _STARTUP.sort(key=lambda h: (h[1], h[2]))
        return fn
    return deco


def shutdown_hook(name: str, *, order: int = 100) -> Callable[[Callable], Callable]:
    """Register ``fn(sctx)`` (sync or async) to run at shutdown, before projections are flushed."""
    def deco(fn: Callable) -> Callable:
        _SHUTDOWN[:] = [h for h in _SHUTDOWN if h[1] != name]
        _SHUTDOWN.append((order, name, fn))
        _SHUTDOWN.sort(key=lambda h: (h[0], h[1]))
        return fn
    return deco


async def _call(fn: Callable, *args) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    res = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(res):
        return await res
    return res


class StudioContext:
    """Everything a request, job hook or feature service needs from the running server.

    ``db`` (:class:`~sparc.studio.db.Database`), ``hub``
    (:class:`~sparc.studio.events.EventHub`), ``jobs``
    (:class:`~sparc.studio.jobs.manager.JobManager`, after startup),
    ``workspace``, ``paths`` (:class:`~sparc.studio.security.PathGuard`),
    ``config`` (launch :class:`~sparc.studio.settings.StudioSettings`),
    :meth:`settings` (user settings) and ``services`` - a dict where feature
    items keep their singletons (``services["registry"]``,
    ``services["engine"]``).
    """

    def __init__(self, config: StudioSettings):
        self.config = config
        self.workspace = config.workspace
        self.token = config.token or generate_token()
        self.hosts = allowed_hosts(config.port, bind_host=config.host, public_hosts=config.public_hosts,
                                   dev_origins=config.dev_origins)
        self.origins = allowed_origins(self.hosts, config.dev_origins)
        self.hub = EventHub()
        self.db = Database(self.workspace.db_path)
        self.paths = PathGuard(self.workspace.root)
        self.services: dict[str, Any] = {}
        self.jobs = None
        self.pid = os.getpid()
        self.started_utc = utc_now()
        self.started = False
        self.stop_jobs = False
        self.request_shutdown: Callable[[], None] | None = None
        self._settings: Settings | None = None
        self._shutdown_begun = False
        self._announced = False

    # -- settings ---------------------------------------------------------------------------------

    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self.db)
        return self._settings

    def reload_settings(self) -> Settings:
        self._settings = load_settings(self.db)
        return self._settings

    @property
    def registry(self):
        """The runs registry service (backend-runs), when present."""
        return self.services.get("registry")

    def engine_state(self) -> str:
        eng = self.services.get("engine")
        state = getattr(eng, "state", None)
        if callable(state):
            try:
                state = state()
            except Exception:
                state = "error"
        return state if isinstance(state, str) else "absent"

    def test_kinds(self) -> bool:
        return bool(self.config.test_kinds)

    # -- lifecycle --------------------------------------------------------------------------------

    async def startup(self) -> None:
        t0 = time.perf_counter()
        ws = self.workspace
        await asyncio.to_thread(ws.ensure)
        await asyncio.to_thread(write_private, ws.token_path, self.token + "\n")
        self.db.reopen()
        if self.hub.closed:                       # a second lifespan of the same app
            self.hub = EventHub()
        self._shutdown_begun = self._announced = False
        await asyncio.to_thread(self.db.migrate)
        if self.config.reindex:
            from sparc.studio.db import reindex

            summary = await asyncio.to_thread(reindex, self.db, ws)
            log.warning("reindexed %s: %s", ws.root, summary)
        self.reload_settings()
        self.hub.bind()
        for phase, _o, name, fn in list(_STARTUP):
            if phase == "scan":
                await self._hook(name, fn)
        from sparc.studio.jobs.manager import JobManager

        self.jobs = JobManager(self)
        await self.jobs.start(reattach=True)
        for phase, _o, name, fn in list(_STARTUP):
            if phase == "ready":
                await self._hook(name, fn)
        self.jobs.start_sampler()
        self.started = True
        log.info("SPARC Studio ready in %.2f s (workspace %s)", time.perf_counter() - t0, ws.root)

    async def _hook(self, name: str, fn: Callable) -> None:
        try:
            await _call(fn, self)
        except Exception:
            log.exception("hook %s failed", name)

    def announce_shutdown(self) -> None:
        """Broadcast ``server_shutdown`` once."""
        if self._announced:
            return
        self._announced = True
        try:
            self.hub.publish("server_shutdown", {"stop_jobs": bool(self.stop_jobs or self.config.stop_jobs_on_exit)})
        except Exception:
            pass

    def begin_shutdown(self) -> None:
        """Tell every SSE client the server is going away and end the streams (before connections drain)."""
        if self._shutdown_begun:
            return
        self._shutdown_begun = True
        self.announce_shutdown()
        loop = getattr(self.hub, "_loop", None)
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.hub.close)
        else:
            self.hub.close()

    async def shutdown(self) -> None:
        if not self._shutdown_begun:
            self.begin_shutdown()
            await asyncio.sleep(0)
        for _o, name, fn in list(_SHUTDOWN):
            await self._hook(name, fn)
        if self.jobs is not None:
            await self.jobs.stop(stop_jobs=bool(self.stop_jobs or self.config.stop_jobs_on_exit))
        self.hub.close()
        await asyncio.to_thread(self.db.close)
        self.started = False


def get_ctx(request: Request) -> StudioContext:
    """FastAPI dependency: the server's :class:`StudioContext`."""
    return request.app.state.studio


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

def create_app(config: StudioSettings | None = None) -> FastAPI:
    """The Studio ASGI app for ``config`` (default: the default workspace and port)."""
    config = config or StudioSettings()
    sctx = StudioContext(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await sctx.startup()
        try:
            yield
        finally:
            await sctx.shutdown()

    app = FastAPI(title="SPARC Studio", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url="/openapi.json", lifespan=lifespan,
                  description="Local web app for the SPARC urban-heat pipeline (docs/studio/api.md).")
    app.state.studio = sctx
    install_error_handlers(app)
    load_kind_modules(test_kinds=bool(config.test_kinds))
    for mod in load_router_modules():
        router = getattr(mod, "router", None)
        if router is not None:
            app.include_router(router, prefix="/api")
        root = getattr(mod, "root_router", None)
        if root is not None:
            app.include_router(root)
    _mount_static(app, config.static_dir or STATIC_DIR)
    app.add_middleware(SecurityMiddleware, config=lambda: sctx)
    _install_openapi(app)
    return app


def _install_openapi(app: FastAPI) -> None:
    """OpenAPI with Studio's error envelope (api.md §0.3) as every operation's error response."""
    from fastapi.openapi.utils import get_openapi

    from sparc.studio.schemas.common import ErrorEnvelope

    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        comps = schema.setdefault("components", {}).setdefault("schemas", {})
        env = ErrorEnvelope.model_json_schema(ref_template="#/components/schemas/{model}")
        comps.update(env.pop("$defs", {}))
        comps["ErrorEnvelope"] = env
        error = {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}}
        for item in schema.get("paths", {}).values():
            for op in item.values():
                responses = op.setdefault("responses", {})
                if "422" in responses:
                    responses["422"] = {"description": "Validation error (`detail.errors`)", "content": error}
                responses.setdefault("default", {"description": "Error envelope (api.md §0.3)", "content": error})
        for name in ("HTTPValidationError", "ValidationError"):
            comps.pop(name, None)
        app.openapi_schema = schema
        return schema

    app.openapi = openapi


def _mount_static(app: FastAPI, static_dir: Path) -> None:
    static_dir = Path(static_dir)
    immutable = {"Cache-Control": "public, max-age=31536000, immutable"}

    @app.get("/assets/{path:path}", include_in_schema=False)
    async def asset(path: str):
        assets = static_dir / "assets"
        try:
            target = safe_path(path, assets)
        except ApiError:
            raise ApiError("not_found", "no such asset")
        if not target.is_file():
            raise ApiError("not_found", f"no such asset: {path}")
        return FileResponse(target, headers=immutable)

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise ApiError("not_found", f"no such endpoint: /{path}")
        if path and "." in path.rsplit("/", 1)[-1]:
            try:
                target = safe_path(path, static_dir)
            except ApiError:
                target = None
            if target is not None and target.is_file() and target.name != "index.html":
                return FileResponse(target, headers={"Cache-Control": "no-cache"})
        index = static_dir / "index.html"
        if index.is_file():
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-store"})
        return HTMLResponse(NOT_BUILT_HTML, headers={"Cache-Control": "no-store"})
