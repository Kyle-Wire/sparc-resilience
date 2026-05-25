"""sparc.server.deps — shared server-side state and helpers.

This module owns the single ``state`` instance and common helpers that
route modules need.  Routes should import from here rather than from
``app.py`` so they can be imported independently for testing.

Architecture note
-----------------
``app.py`` previously held ``state = ServerState()`` at module level.
Route functions closed over ``state`` and helpers defined in the same
file, making independent testing impossible.

With ``deps.py`` as the single source of truth for ``state``, any route
module can do::

    from sparc.server.deps import state, get_open_store

without triggering the full FastAPI + middleware initialisation in
``app.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from fastapi import HTTPException

from sparc.server.state import ServerState
from sparc.server.project_session import ProjectSession
from sparc.server.execution_context import ExecutionContext

# ---------------------------------------------------------------------------
# Process-global server state — THE single instance
# ---------------------------------------------------------------------------

state: ServerState = ServerState()

# Structured replacements for ServerState — new routes should prefer these.
session: ProjectSession = ProjectSession()
execution: ExecutionContext = ExecutionContext()

# ---------------------------------------------------------------------------
# Store accessor — raises 400 when nothing is loaded
# ---------------------------------------------------------------------------


def get_open_store():
    """Return an ``ArtifactStore`` bound to the active registry.

    Raises :exc:`~fastapi.HTTPException` (400) when no project is loaded or
    the registry is unavailable.
    """
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        raise HTTPException(400, "No active run registry. Load a project first.")
    from sparc.registry.store import ArtifactStore
    return ArtifactStore(state.registry)


# ---------------------------------------------------------------------------
# Missing-artifact helper
# ---------------------------------------------------------------------------


def missing_artifact_response(
    *,
    artifact_id: str,
    stage: str | int,
    expected_paths: "list[Path] | None" = None,
    hint: str = "",
) -> HTTPException:
    """Return a structured 404 that the frontend can render as an actionable
    empty-state."""
    detail = {
        "error": "missing_artifact",
        "missing_artifact": artifact_id,
        "produced_by_stage": str(stage),
        "expected_path": (str(expected_paths[0]) if expected_paths else None),
        "candidate_paths": [str(p) for p in (expected_paths or [])],
        "hint": hint,
    }
    return HTTPException(status_code=404, detail=detail)


# ---------------------------------------------------------------------------
# Cached read helper
# ---------------------------------------------------------------------------


async def read_or_404(
    stage: str | int,
    artifact_id: str,
    *,
    hint: str = "",
    store: Optional[Any] = None,
) -> Any:
    """DB-only read: ``store.read_any`` or structured 404 if missing.

    Results are served from the in-process LRU cache when available.
    If *store* is not provided, the active registry store is used.
    """
    cached = state.result_cache.get(stage, artifact_id)
    if cached is not None:
        return cached
    _store = store if store is not None else get_open_store()
    if not _store.has(stage, artifact_id):
        raise missing_artifact_response(
            artifact_id=artifact_id, stage=stage, hint=hint,
        )
    result = await asyncio.to_thread(_store.read_any, stage, artifact_id)
    state.result_cache.set(stage, artifact_id, result)
    return result


# ---------------------------------------------------------------------------
# Registry path helper
# ---------------------------------------------------------------------------


def registry_path(stage: str | int, artifact_id: str) -> "Path | None":
    """Look up an artifact's absolute path via the registry. None if missing."""
    reg = state.registry
    if reg is None:
        return None
    entry = reg.lookup(stage, artifact_id)
    if entry is None or entry.partial:
        return None
    p = reg.resolve(entry)
    return p if p.exists() else None
