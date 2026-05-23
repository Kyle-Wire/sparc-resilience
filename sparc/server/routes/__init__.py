"""sparc.server.routes — FastAPI route modules.

Each sub-module exposes a ``router: APIRouter`` that ``app.py`` mounts
with :meth:`~fastapi.FastAPI.include_router`.

Example::

    from sparc.server.routes.results import router as results_router
    app.include_router(results_router)
"""
