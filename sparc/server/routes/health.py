"""sparc.server.routes.health — /health status endpoint.

Migrated from app.py.  Importable independently of the full FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter

from sparc.server.deps import state

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Return server liveness and project-load status."""
    return {
        "status": "ok",
        "project_loaded": state.project_config is not None,
        "is_running": state.is_running,
        "current_stage": state.current_stage,
        "manifest_loaded": state.registry is not None,
    }
