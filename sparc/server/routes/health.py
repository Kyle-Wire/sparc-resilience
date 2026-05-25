"""sparc.server.routes.health — /health status endpoint.

Migrated from app.py.  Importable independently of the full FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter

from sparc.server.deps import session, execution

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Return server liveness and project-load status."""
    return {
        "status": "ok",
        "project_loaded": session.project_config is not None,
        "is_running": execution.is_running,
        "current_stage": execution.current_stage,
        "manifest_loaded": session.registry is not None,
    }
