"""Budget plan endpoints (api.md §7.9): planned mode, stored plans, verification, layers, plan → scenario and
the field kit (SPEC §7.10)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.engine.service import get_service
from sparc.studio.errors import ApiError
from sparc.studio.scenarios import library, plans
from sparc.studio.scenarios import schemas as S
from sparc.studio.schemas.common import Job, Ok
from sparc.studio.workspace import read_json

router = APIRouter(tags=["plans"])

PLAN_FIELDS = ("dose", "planned_benefit", "closed_loop_delta")


def _reader(sctx: StudioContext):
    reader = sctx.services.get("reader")
    if reader is None:
        raise ApiError("not_ready", "the runs registry is still starting", status=503)
    return reader


def _ctx(sctx: StudioContext, rid: str):
    return _reader(sctx).get(rid)


def _project_dir(sctx: StudioContext, pid: str | None) -> Path | None:
    if not pid:
        return None
    row = sctx.db.fetchone("SELECT dir FROM projects WHERE id = ?", (pid,))
    return Path(row["dir"]) if row and row.get("dir") else None


def _jobs(sctx: StudioContext):
    if sctx.jobs is None:
        raise ApiError("not_ready", "the job manager is still starting", status=503)
    return sctx.jobs


async def _verify(sctx: StudioContext, row: dict, frontier: bool) -> dict:
    rid = row["run_id"]
    run = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (rid,))
    if run is None or not (Path(run["run_dir"]) / "checkpoint.pkl").is_file():
        raise ApiError("no_checkpoint", "verifying a plan needs the run's checkpoint.pkl", detail={"run_id": rid})
    await asyncio.to_thread(get_service(sctx).preflight, rid)
    kind = "engine.plan_frontier" if frontier else "engine.plan_verify"
    return await _jobs(sctx).submit(kind, {"plan_id": row["id"]}, run_id=rid, project_id=row.get("project_id"),
                                    label=f"{'Verify frontier' if frontier else 'Verify'}: {row.get('name') or row['id']}")


@router.post("/runs/{rid}/plans/preview", response_model=S.PlanPreview)
async def plan_preview(rid: str, body: S.PlanParams, sctx: StudioContext = Depends(get_ctx)):
    ctx = _ctx(sctx, rid)
    return await asyncio.to_thread(plans.preview, ctx, body, db=sctx.db, project_dir=_project_dir(sctx, ctx.project_id))


@router.post("/runs/{rid}/plans", status_code=201, response_model=S.PlanCreated)
async def create_plan(rid: str, body: S.PlanCreate, sctx: StudioContext = Depends(get_ctx)):
    ctx = _ctx(sctx, rid)
    row = await asyncio.to_thread(plans.create, sctx.db, ctx, body.params, body.name,
                                  project_dir=_project_dir(sctx, ctx.project_id))
    job = None
    if body.verify and (ctx.run_dir / "checkpoint.pkl").is_file():
        try:
            job = await _verify(sctx, row, False)
        except ApiError as exc:                 # the plan stands; Verify can be started later
            if exc.code not in ("untrusted_pickle", "engine_memory", "no_checkpoint"):
                raise
    return {"plan": plans.plan_out(row), "job": job}


@router.get("/runs/{rid}/plans", response_model=list[S.Plan])
async def list_plans(rid: str, sctx: StudioContext = Depends(get_ctx)):
    _ctx(sctx, rid)
    return plans.list_plans(sctx.db, rid)


@router.get("/plans/{plid}", response_model=S.Plan)
async def get_plan(plid: str, sctx: StudioContext = Depends(get_ctx)):
    return plans.plan_out(plans.plan_row(sctx.db, plid))


@router.post("/plans/{plid}/verify", status_code=202, response_model=Job)
async def verify_plan(plid: str, body: S.VerifyRequest | None = None, sctx: StudioContext = Depends(get_ctx)):
    body = body or S.VerifyRequest()
    return await _verify(sctx, plans.plan_row(sctx.db, plid), bool(body.frontier))


@router.get("/plans/{plid}/layers/{field}.bin", response_class=Response,
            responses={200: {"content": {"application/octet-stream": {}}}})
async def plan_layer(plid: str, field: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.engine import store

    row = plans.plan_row(sctx.db, plid)
    if field not in PLAN_FIELDS:
        raise ApiError("unknown_layer", f"plan layers are {', '.join(PLAN_FIELDS)}")
    pdir = Path(row["dir"])
    if field == "closed_loop_delta":
        rj = read_json(pdir / "realised.json") or {}
        if not rj.get("result_id"):
            raise ApiError("not_found", "the plan has not been verified yet (POST /plans/{plid}/verify)")
        rrow = store.result_row(sctx.db, rj["result_id"])
        arr = await asyncio.to_thread(store.result_array, rrow, "delta")
        tag = rj["result_id"]
    else:
        p = pdir / f"{field}.npy"
        if not p.exists():
            raise ApiError("not_found", f"plan {plid} has no {field}")
        arr = np.load(p, allow_pickle=False)
        tag = p.stat().st_mtime_ns
    etag = '"' + hashlib.sha1(f"{plid}|{field}|{tag}".encode()).hexdigest() + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    return Response(a.tobytes(), media_type="application/octet-stream",
                    headers={"X-SPARC-Dtype": "float32", "X-SPARC-Length": str(a.size), "ETag": etag,
                             "Cache-Control": "no-cache"})


@router.post("/plans/{plid}/to-scenario", status_code=201, response_model=S.Scenario)
async def plan_to_scenario(plid: str, sctx: StudioContext = Depends(get_ctx)):
    row = plans.plan_row(sctx.db, plid)
    if not row.get("project_id"):
        raise ApiError("validation", "the plan's run belongs to no project",
                       detail={"errors": [{"path": "plid", "message": "no project", "code": "project"}]})
    srow = await asyncio.to_thread(library.create, sctx.db, sctx.workspace, row["project_id"],
                                   plans.to_scenario_doc(row))
    return library.scenario_out(sctx.db, srow)


@router.post("/plans/{plid}/field-kit", response_model=S.FieldKit)
async def plan_field_kit(plid: str, body: S.FieldKitRequest | None = None, sctx: StudioContext = Depends(get_ctx)):
    body = body or S.FieldKitRequest()
    row = plans.plan_row(sctx.db, plid)
    ctx = _ctx(sctx, row["run_id"])
    return await asyncio.to_thread(plans.field_kit, sctx.db, ctx, row, n_sites=body.n_sites,
                                   min_spacing_m=body.min_spacing_m, n_pairs=body.n_pairs,
                                   min_distance_m=body.min_distance_m)


@router.delete("/plans/{plid}", response_model=Ok)
async def delete_plan(plid: str, sctx: StudioContext = Depends(get_ctx)):
    await asyncio.to_thread(plans.delete, sctx.db, plid)
    return {"ok": True}
