"""Findings notebook endpoints (api.md §11, SPEC §6.10): CRUD, ordering and the raw image upload."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from sparc.core import runio
from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError
from sparc.studio.exports import findings as F
from sparc.studio.schemas.common import Ok
from sparc.studio.security import stream_upload

router = APIRouter(tags=["findings"])


class Finding(BaseModel):
    id: str
    project_id: str
    run_id: str | None = None
    view: str
    url_state: str
    title: str
    note_md: str
    snapshot: dict[str, Any] = Field(default_factory=dict)
    image_url: str | None = None
    position: float
    created_utc: str
    updated_utc: str


class FindingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    run_id: str | None = None
    view: str
    url_state: str
    title: str
    note_md: str = ""
    snapshot: dict[str, Any] = Field(default_factory=dict)


class FindingPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    note_md: str | None = None
    position: float | None = None


class ImageUrl(BaseModel):
    image_url: str


@router.get("/findings", response_model=list[Finding])
async def list_findings(project: str | None = Query(None), run: str | None = Query(None),
                        sctx: StudioContext = Depends(get_ctx)):
    return await asyncio.to_thread(F.list_findings, sctx.db, project, run)


@router.post("/findings", status_code=201, response_model=Finding)
async def create_finding(body: FindingCreate, sctx: StudioContext = Depends(get_ctx)):
    return await asyncio.to_thread(F.create_finding, sctx.db, body.model_dump())


@router.patch("/findings/{fid}", response_model=Finding)
async def patch_finding(fid: str, body: FindingPatch, sctx: StudioContext = Depends(get_ctx)):
    return await asyncio.to_thread(F.patch_finding, sctx.db, fid, body.model_dump(exclude_none=True))


@router.delete("/findings/{fid}", response_model=Ok)
async def delete_finding(fid: str, sctx: StudioContext = Depends(get_ctx)):
    await asyncio.to_thread(F.delete_finding, sctx.db, fid)
    return {"ok": True}


@router.put("/findings/{fid}/image", response_model=ImageUrl)
async def put_image(fid: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    """Raw ``image/png`` or ``image/svg+xml`` body (≤ 10 MB), validated by content type, size and content (the
    PNG signature, an ``<svg`` root) before it replaces the finding's image."""
    row = F.get_row(sctx.db, fid)
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    suffix = F.IMAGE_TYPES.get(ctype)
    if suffix is None:
        raise ApiError("bad_suffix", f"content type {ctype or '(none)'!r} is not an accepted image",
                       detail={"content_type": ctype, "allowed": sorted(F.IMAGE_TYPES)})
    dest = F.findings_dir(sctx.db, row["project_id"]) / f"{fid}{suffix}"
    staged = dest.with_name(f".{fid}.incoming{suffix}")      # the current image stays until the new one checks out
    try:
        await stream_upload(request, staged, max_bytes=F.IMAGE_MAX_BYTES, allowed_suffixes=(suffix,),
                            content_types=(ctype,))
        if not await asyncio.to_thread(F.image_ok, staged, suffix):
            raise ApiError("bad_suffix", f"the body is not {'a PNG' if suffix == '.png' else 'an SVG'} image",
                           detail={"content_type": ctype})
        runio.replace(staged, dest)
    finally:
        staged.unlink(missing_ok=True)
    out = await asyncio.to_thread(F.set_image, sctx.db, fid, dest)
    return {"image_url": out["image_url"]}


@router.get("/findings/{fid}/image")
async def get_image(fid: str, sctx: StudioContext = Depends(get_ctx)):
    path = F.image_path(sctx.db, fid)
    if path is None:
        raise ApiError("not_found", f"finding {fid} has no image")
    svg = path.suffix == ".svg"
    headers = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}
    if svg:
        headers["Content-Security-Policy"] = "sandbox"
    return FileResponse(path, media_type="image/svg+xml" if svg else "image/png", headers=headers)
