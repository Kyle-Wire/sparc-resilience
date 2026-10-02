"""Exports and the report preview (api.md §10, SPEC §6.7): ``POST /api/exports`` creates the row and queues
``export.<kind>`` by name (the engine's pack kinds included), listings, downloads (streamed) and deletes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError, validation_error
from sparc.studio.exports import store
from sparc.studio.exports.kinds import ReportSection
from sparc.studio.schemas.common import Job, Ok

router = APIRouter(tags=["exports"])


class Export(BaseModel):
    id: str
    project_id: str
    run_id: str | None = None
    kind: str
    ref: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    job_id: str
    status: Literal["running", "ready", "failed"]
    path: str | None = None
    bytes: int | None = None
    draft: bool = False
    created_utc: str


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["bundle", "gis", "page", "report", "findings", "decision_pack", "plan_pack", "compare_pack"]
    project_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExportCreated(BaseModel):
    export: Export
    job: Job


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    sections: list[ReportSection] = Field(min_length=1)
    result_ids: list[str] | None = None
    plan_ids: list[str] | None = None
    finding_ids: list[str] | None = None


class PreviewResponse(BaseModel):
    html: str


def _jobs(sctx: StudioContext):
    if sctx.jobs is None:
        raise ApiError("not_ready", "the job manager is still starting", status=503)
    return sctx.jobs


@router.post("/exports", status_code=202, response_model=ExportCreated)
async def create_export(body: ExportRequest, sctx: StudioContext = Depends(get_ctx)):
    _jobs(sctx)
    return await store.create_export(sctx, body.kind, body.project_id, body.params)


@router.get("/projects/{pid}/exports", response_model=list[Export])
async def project_exports(pid: str, sctx: StudioContext = Depends(get_ctx)):
    if sctx.db.fetchone("SELECT id FROM projects WHERE id = ?", (pid,)) is None:
        raise ApiError("not_found", f"no project {pid!r}")
    return await asyncio.to_thread(store.list_exports, sctx.db, pid)


@router.get("/exports/{eid}", response_model=Export)
async def get_export(eid: str, sctx: StudioContext = Depends(get_ctx)):
    row = await asyncio.to_thread(lambda: store.reconcile(sctx.db, store.get_row(sctx.db, eid)))
    return store.export_out(row)


@router.get("/exports/{eid}/download")
async def download_export(eid: str, sctx: StudioContext = Depends(get_ctx)):
    row = await asyncio.to_thread(lambda: store.reconcile(sctx.db, store.get_row(sctx.db, eid)))
    path = Path(row["path"]) if row.get("path") else None
    if row.get("status") != "ready" or path is None or not path.is_file():
        raise ApiError("not_ready", "the export is not ready yet" if row.get("status") == "running" else
                       "the export has no file", detail={"status": row.get("status"), "job_id": row.get("job_id")})
    d = store.export_dir(sctx.db, row)
    if d is None or d.resolve() not in path.resolve().parents:
        raise ApiError("not_found", "the export's file is outside its folder")
    media = {".zip": "application/zip", ".html": "text/html; charset=utf-8", ".md": "text/markdown; charset=utf-8",
             ".json": "application/json"}.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name,
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.delete("/exports/{eid}", response_model=Ok)
async def delete_export(eid: str, sctx: StudioContext = Depends(get_ctx)):
    await asyncio.to_thread(store.delete_export, sctx.db, eid)
    return {"ok": True}


@router.post("/projects/{pid}/report/preview", response_model=PreviewResponse)
async def report_preview(pid: str, body: PreviewRequest, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.exports.report import build_report

    project = sctx.db.fetchone("SELECT * FROM projects WHERE id = ?", (pid,))
    if project is None:
        raise ApiError("not_found", f"no project {pid!r}")
    reader = sctx.services.get("reader")
    if reader is None:
        raise ApiError("not_ready", "the runs registry is still starting", status=503)
    ctx = reader.get(body.run_id)
    if ctx.project_id and ctx.project_id != pid:
        raise ApiError("validation", f"run {body.run_id} belongs to another project",
                       detail={"errors": [{"path": "run_id", "message": "other project", "code": "mismatch"}]})
    for key, table in (("result_ids", "results"), ("plan_ids", "plans"), ("finding_ids", "findings")):
        bad = await asyncio.to_thread(store.unknown_ids, sctx.db, table, getattr(body, key), pid)
        if bad:                             # as POST /api/exports refuses them for export.report
            raise validation_error([{"path": key, "code": "unknown_id", "message": f"not in this project: {bad}"}],
                                   f"unknown {table}")
    html = await asyncio.to_thread(build_report, ctx, sctx.db, body.sections, result_ids=body.result_ids or [],
                                   plan_ids=body.plan_ids or [], finding_ids=body.finding_ids, fmt="html",
                                   project=project)
    return {"html": html}
