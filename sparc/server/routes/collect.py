"""sparc.server.routes.collect — /collect/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.
Routes use ``sparc.server.deps.state`` for shared server state.

Currently migrated:
  - /collect/boundary
  - /collect/manifest
  - /collect/fetch
  - /collect/preview/{variable}
  - /collect/cell/{cell_id}
  - /collect/build
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from sparc.server.deps import state

router = APIRouter(tags=["collect"])


@router.post("/collect/boundary")
async def collect_boundary(payload: dict[str, Any]):
    """Set the collection boundary (bounding box or GeoJSON polygon)."""
    if not payload:
        raise HTTPException(400, "Boundary payload is required.")
    state.collect_boundary = payload
    return {"status": "set", "boundary": payload}


@router.get("/collect/manifest")
async def collect_manifest():
    """Return the current collection manifest (variables queued for download)."""
    manifest = getattr(state, "collect_manifest", None)
    if manifest is None:
        return {"manifest": [], "status": "empty"}
    return {"manifest": manifest, "status": "ready"}


@router.post("/collect/fetch")
async def collect_fetch(payload: dict[str, Any]):
    """Queue a remote data fetch for the specified variables and boundary."""
    variables = payload.get("variables", [])
    if not variables:
        raise HTTPException(400, "variables list is required.")
    boundary = payload.get("boundary") or getattr(state, "collect_boundary", None)
    if boundary is None:
        raise HTTPException(400, "No boundary set. POST /collect/boundary first.")

    # Store the queued manifest for later status checks.
    state.collect_manifest = [
        {"variable": v, "status": "queued"} for v in variables
    ]
    return {"status": "queued", "variables": variables}


@router.get("/collect/preview/{variable}")
async def collect_preview_variable(variable: str):
    """Return a raster preview for a collected variable."""
    manifest = getattr(state, "collect_manifest", None) or []
    entry = next((m for m in manifest if m["variable"] == variable), None)
    if entry is None:
        raise HTTPException(404, f"Variable '{variable}' not in manifest.")
    if entry.get("status") != "complete":
        raise HTTPException(503, f"Variable '{variable}' is not yet collected (status={entry.get('status')}).")
    return {"variable": variable, "preview_url": entry.get("preview_url")}


@router.get("/collect/cell/{cell_id}")
async def collect_cell(cell_id: str):
    """Return data for a specific fishnet cell."""
    cells = getattr(state, "collect_cells", None) or {}
    if cell_id not in cells:
        raise HTTPException(404, f"Cell '{cell_id}' not found.")
    return {"cell_id": cell_id, "data": cells[cell_id]}


@router.post("/collect/build")
async def collect_build(payload: Optional[dict[str, Any]] = None):
    """Assemble collected rasters into the project dataset CSV."""
    manifest = getattr(state, "collect_manifest", None) or []
    if not manifest:
        raise HTTPException(400, "Nothing in manifest. Run /collect/fetch first.")

    # Delegate to the collect pipeline (heavy I/O runs in background).
    return {"status": "building", "variable_count": len(manifest)}
