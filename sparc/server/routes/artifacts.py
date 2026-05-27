"""Artifact download and export routes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from sparc.server import deps

router = APIRouter(tags=["artifacts"])

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


def _ensure_registry() -> None:
    """Raise HTTPException if project or registry is unavailable.

    Attempts to auto-attach the registry from disk when a project is loaded
    but no registry has been set yet.
    """
    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")
    if state.registry is None:
        try:
            from sparc.registry import RunRegistry
            from sparc.run.pipeline_paths import PipelinePaths
            paths = PipelinePaths.from_config(state.project_config)
            reg = RunRegistry(paths.output_dir, autoload=True)
            try:
                reg.migrate_from_disk(paths)
            except Exception:
                pass
            state.registry = reg
        except Exception:
            pass
    if state.registry is None:
        raise HTTPException(503, "Run registry unavailable")


def _get_artifact_store():
    state = deps.state
    if state.registry is None:
        raise HTTPException(503, "No active run/registry")
    from sparc.registry.store import ArtifactStore
    return ArtifactStore(state.registry)


# ---------------------------------------------------------------------------
# Extension-specific routes (declared before the catch-all {artifact_id})
# ---------------------------------------------------------------------------

@router.get("/artifacts/{stage}/{artifact_id}.csv")
async def get_artifact_csv(stage: str, artifact_id: str, index: bool = False):
    """Render a registered artifact as CSV bytes."""
    _ensure_registry()
    from sparc.registry.run_registry import get_active_registry, set_active_registry
    from sparc.report.render import RenderError, render_csv

    state = deps.state
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


@router.get("/artifacts/{stage}/{artifact_id}.json")
async def get_artifact_json(stage: str, artifact_id: str):
    """Render a registered artifact (struct or table) as JSON."""
    _ensure_registry()
    from sparc.registry.run_registry import get_active_registry, set_active_registry
    from sparc.report.render import RenderError, render_json

    state = deps.state
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


@router.get("/artifacts/{stage}/{artifact_id}.geojson")
async def get_artifact_geojson(stage: str, artifact_id: str):
    """Render a geometry-bearing table as GeoJSON."""
    _ensure_registry()
    from sparc.registry.run_registry import get_active_registry, set_active_registry
    from sparc.report.render import RenderError, render_geojson

    state = deps.state
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


@router.get("/artifacts/{stage}/{artifact_id}.png")
async def get_artifact_png(stage: str, artifact_id: str, dpi: int = 150):
    """Render a registered artifact as a PNG via the figures module."""
    _ensure_registry()
    from sparc.registry.run_registry import get_active_registry, set_active_registry
    try:
        from sparc.report.figures import FigureRenderError, render_for_artifact
    except ImportError as exc:
        raise HTTPException(503, f"figures module unavailable: {exc}")

    state = deps.state
    if state.registry is None:
        raise deps.missing_artifact_response(
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


# ---------------------------------------------------------------------------
# DB-backed artifact download / export (Phase 5 architecture)
# ---------------------------------------------------------------------------

@router.get("/artifacts/{stage}/{artifact_id}/download")
def download_artifact(stage: str, artifact_id: str, fmt: Optional[str] = None):
    """Download an artifact from artifacts.db as a file.

    Query parameters:
      fmt — one of csv, parquet, json, gpkg, geojson, joblib, pkl, npy.
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
    return FileResponse(
        path,
        media_type=media,
        filename=f"{artifact_id}.{suffix}",
    )


@router.post("/artifacts/{stage}/export")
def export_stage_zip(stage: str):
    """Bundle every artifact for a stage into a downloadable .zip."""
    store = _get_artifact_store()
    tmpdir = Path(tempfile.mkdtemp(prefix=f"sparc_export_{stage}_"))
    try:
        zip_path = store.export_stage(stage, tmpdir, as_zip=True)
        if not isinstance(zip_path, Path):
            raise RuntimeError("export_stage did not return a zip path")
    except Exception as exc:
        raise HTTPException(500, f"Stage export failed: {exc}") from exc
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"stage_{stage}_artifacts.zip",
    )


@router.get("/artifacts/{stage}")
def list_stage_artifacts(stage: str):
    """List all artifacts registered for a stage (id, format, storage_kind)."""
    state = deps.state
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


# NOTE: Declared LAST so extension-specific routes above match first.
# Otherwise {artifact_id} would swallow "foo.csv" etc. and the suffixed
# routes would become dead code.
@router.get("/artifacts/{stage}/{artifact_id}")
async def get_artifact_native(stage: str, artifact_id: str):
    """Return an artifact in its native format (CSV for tables, JSON for structs, raw bytes for blobs)."""
    _ensure_registry()
    from sparc.registry.run_registry import get_active_registry, set_active_registry
    from sparc.report.render import RenderError, render_native

    state = deps.state
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
