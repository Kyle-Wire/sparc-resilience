"""Scenario Lab endpoints (api.md §7.1–7.8): the engine, levers, emulator, preview and compile, templates and
the scenario library, results, comparisons, climate × adaptation and sweeps.  Budget plans are in
:mod:`sparc.studio.routes.plans`.

At server start (``ready`` phase) the engine service reconnects to a live engine host (``pid`` +
``create_time`` match) or removes its stale files.  At a clean shutdown the host is stopped - unless an engine
job is still running and jobs are left running, so it can be reattached on the next start.  When jobs stop
with the server, the running engine jobs are cancelled first and end ``cancelled``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response

from sparc.studio.app import StudioContext, get_ctx, shutdown_hook, startup_hook
from sparc.studio.engine import store
from sparc.studio.engine.service import get_service
from sparc.studio.errors import ApiError
from sparc.studio.jobs.manager import job_out
from sparc.studio.scenarios import library
from sparc.studio.scenarios import schemas as S
from sparc.studio.schemas.common import ACTIVE_STATUSES, LIVE_STATUSES, Job, Ok

router = APIRouter(tags=["lab"])
log = logging.getLogger("sparc.studio.engine")

DESIGN_MAX_BYTES = 256 * 1024 ** 2


# ---------------------------------------------------------------------------
# services
# ---------------------------------------------------------------------------

@startup_hook("engine.reconnect", phase="ready", order=60)
def _reconnect(sctx: StudioContext) -> None:
    get_service(sctx).reconnect()


@shutdown_hook("engine.host", order=60)
async def _stop_host(sctx: StudioContext) -> None:
    svc = sctx.services.get("engine")
    if svc is None:
        svc = get_service(sctx)
    stop_jobs = bool(sctx.stop_jobs or sctx.config.stop_jobs_on_exit)
    marks = ",".join("?" for _ in LIVE_STATUSES)
    live = sctx.db.fetchall(f"SELECT id, job_dir FROM jobs WHERE executor = 'engine' AND status IN ({marks})",
                            LIVE_STATUSES)
    if live and not stop_jobs:
        return                                  # an engine job keeps running: the host survives the restart
    if live:
        await _cancel_engine_jobs(sctx, live)
    await asyncio.to_thread(svc.shutdown)


async def _cancel_engine_jobs(sctx: StudioContext, rows: list[dict], wait_s: float = 10.0) -> None:
    """Shutdown with ``stop_jobs``: the running engine jobs are cancelled before the host stops (the host stops
    them between folds and they end ``cancelled``), and are marked killed, so one the host stop interrupts still
    ends ``cancelled``, never ``failed: engine host exited``."""
    import time

    from sparc.studio.engine.executor import ENGINE_EXECUTOR

    executor = (getattr(sctx.jobs, "executors", None) or {}).get("engine") or ENGINE_EXECUTOR
    for r in rows:
        executor.killed.add(r["id"])
        if sctx.jobs is not None:
            try:
                await sctx.jobs.cancel(r["id"], by="shutdown")
            except ApiError:
                pass
    deadline = time.monotonic() + min(wait_s, float(sctx.config.shutdown_grace_s))

    def settled(r: dict) -> bool:
        if r.get("job_dir") and (Path(r["job_dir"]) / "result.json").exists():
            return True
        row = sctx.db.fetchone("SELECT status FROM jobs WHERE id = ?", (r["id"],))
        return row is None or row["status"] not in LIVE_STATUSES

    while time.monotonic() < deadline and not all(settled(r) for r in rows):
        await asyncio.sleep(0.1)


def _reader(sctx: StudioContext):
    reader = sctx.services.get("reader")
    if reader is None:
        raise ApiError("not_ready", "the runs registry is still starting", status=503)
    return reader


def _ctx(sctx: StudioContext, rid: str):
    return _reader(sctx).get(rid)


def _jobs(sctx: StudioContext):
    if sctx.jobs is None:
        raise ApiError("not_ready", "the job manager is still starting", status=503)
    return sctx.jobs


def _project_dir(sctx: StudioContext, pid: str | None) -> Path | None:
    if not pid:
        return None
    row = sctx.db.fetchone("SELECT dir FROM projects WHERE id = ?", (pid,))
    return Path(row["dir"]) if row and row.get("dir") else None


def _unit(ctx) -> str:
    return ctx.units.get("target", "°F")


def _f32(arr, etag: str, *, immutable: bool = True) -> Response:
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    return Response(a.tobytes(), media_type="application/octet-stream",
                    headers={"X-SPARC-Dtype": "float32", "X-SPARC-Length": str(a.size), "ETag": etag,
                             "Cache-Control": "private, max-age=31536000, immutable" if immutable else "no-store"})


def _etag(*parts) -> str:
    return '"' + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest() + '"'


def _has_checkpoint(sctx, rid: str) -> None:
    row = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (rid,))
    if row is None:
        raise ApiError("not_found", f"no run {rid!r}")
    if not (Path(row["run_dir"]) / "checkpoint.pkl").is_file():
        raise ApiError("no_checkpoint", "exact scenarios need the run's checkpoint.pkl (the run did not reach S3)",
                       detail={"run_id": rid})


# ---------------------------------------------------------------------------
# engine (api.md §7.2)
# ---------------------------------------------------------------------------

@router.get("/engine", response_model=S.EngineHost)
async def get_engine(sctx: StudioContext = Depends(get_ctx)):
    return await asyncio.to_thread(get_service(sctx).host_info)


@router.post("/engine/restart", status_code=202, response_model=Ok)
async def restart_engine(sctx: StudioContext = Depends(get_ctx)):
    await asyncio.to_thread(get_service(sctx).restart)
    return {"ok": True}


@router.get("/runs/{rid}/engine", response_model=S.RunEngine)
async def get_run_engine(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return await asyncio.to_thread(get_service(sctx).run_status, rid)


@router.post("/runs/{rid}/engine/open", response_model=None, status_code=202,
             responses={202: {"model": Job}, 200: {"model": S.RunEngine}})
async def open_engine(rid: str, sctx: StudioContext = Depends(get_ctx)):
    svc = get_service(sctx)
    jobs = _jobs(sctx)
    await asyncio.to_thread(svc.preflight, rid)
    if await asyncio.to_thread(svc.loaded, rid) is not None:
        return JSONResponse(await asyncio.to_thread(svc.run_status, rid), status_code=200)
    job = await asyncio.to_thread(svc.open_job, rid)
    if job is not None:
        return JSONResponse(job_out(job), status_code=202)
    out = await jobs.submit("engine.open", {"run_id": rid}, run_id=rid)
    svc.publish(rid, "queued")
    return JSONResponse(out, status_code=202)


async def _evict(rid: str, sctx: StudioContext) -> dict:
    svc = get_service(sctx)
    if sctx.db.fetchone("SELECT id FROM runs WHERE id = ?", (rid,)) is None:
        raise ApiError("not_found", f"no run {rid!r}")
    await asyncio.to_thread(svc.evict, rid)
    return await asyncio.to_thread(svc.run_status, rid)


@router.delete("/runs/{rid}/engine", response_model=S.RunEngine)
async def delete_engine(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return await _evict(rid, sctx)


@router.post("/runs/{rid}/engine/evict", response_model=S.RunEngine,
             description="As built: DELETE /runs/{rid}/engine as a POST, so an Action (POST/GET only) can evict.")
async def evict_engine(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return await _evict(rid, sctx)


# ---------------------------------------------------------------------------
# levers, emulator, preview, compile (api.md §7.3)
# ---------------------------------------------------------------------------

@router.get("/runs/{rid}/levers", response_model=list[S.Lever])
async def get_levers(rid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.engine.preview import levers

    ctx = _ctx(sctx, rid)
    return await asyncio.to_thread(levers, ctx)


@router.get("/runs/{rid}/emulator", response_model=S.EmulatorInfo)
async def get_emulator(rid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.engine.preview import emulator_info

    ctx = _ctx(sctx, rid)
    return await asyncio.to_thread(emulator_info, ctx)


@router.post("/runs/{rid}/preview", response_class=Response,
             responses={200: {"content": {"application/octet-stream": {}},
                              "description": "packed delta (float32[n]) + edited bitset (api.md §7.3)"}})
async def post_preview(rid: str, body: S.PreviewRequest, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.engine.preview import preview

    ctx = _ctx(sctx, rid)
    payload, headers = await asyncio.to_thread(preview, ctx, body, db=sctx.db, reader=_reader(sctx),
                                               project_dir=_project_dir(sctx, ctx.project_id))
    if body.scenario_id:
        shown = {"name": "preview", "edits": [e.model_dump(mode="json") for e in body.edits],
                 "options": body.options.model_dump(mode="json") if body.options is not None else {}}
        await asyncio.to_thread(library.mark_previewed, sctx.db, body.scenario_id, shown)
    return Response(payload, media_type="application/octet-stream", headers=headers)


def _compile(sctx, ctx, doc, *, threads: int = 2):
    from sparc.studio.engine.compile import compile_scenario

    svc = get_service(sctx)
    loaded = svc.loaded(ctx.run_id) is not None if svc.client.alive() else False
    return compile_scenario(ctx, doc, db=sctx.db, reader=_reader(sctx), project_dir=_project_dir(sctx, ctx.project_id),
                            cost_model=getattr(sctx.jobs, "cost_model", None), threads=threads, engine_loaded=loaded)


@router.post("/runs/{rid}/compile", response_model=S.CompileResponse)
async def post_compile(rid: str, body: S.CompileRequest, sctx: StudioContext = Depends(get_ctx)):
    from threadpoolctl import threadpool_limits

    from sparc.studio.engine.preview import emulator_status

    ctx = _ctx(sctx, rid)

    def run():
        with threadpool_limits(1):
            comp = _compile(sctx, ctx, body.scenario, threads=int(sctx.settings().engine_threads))
            return comp.response(emulator_status(ctx, comp))

    return await asyncio.to_thread(run)


# ---------------------------------------------------------------------------
# templates and library (api.md §7.4)
# ---------------------------------------------------------------------------

@router.get("/scenario-templates", response_model=list[S.Template])
async def get_templates():
    from sparc.studio.scenarios import templates

    return templates.listing()


def _scenario(sctx, row: dict, unit: str = "°F") -> dict:
    return library.scenario_out(sctx.db, row, unit)


@router.post("/projects/{pid}/scenarios/from-template", status_code=201, response_model=S.Scenario)
async def from_template(pid: str, body: S.FromTemplate, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import templates

    library.project_row(sctx.db, pid)
    ctx = _ctx(sctx, body.run_id)
    if ctx.project_id and ctx.project_id != pid:
        raise ApiError("validation", f"run {body.run_id} belongs to another project",
                       detail={"errors": [{"path": "run_id", "message": "another project", "code": "project"}]})
    doc = await asyncio.to_thread(templates.build, ctx, body.template, body.params)
    row = await asyncio.to_thread(library.create, sctx.db, sctx.workspace, pid, doc)
    return _scenario(sctx, row, _unit(ctx))


@router.get("/projects/{pid}/scenarios", response_model=list[S.ScenarioSummary])
async def list_scenarios(pid: str, run: str | None = None, tag: str | None = None, status: str | None = None,
                         q: str | None = None, archived: bool = False, sctx: StudioContext = Depends(get_ctx)):
    library.project_row(sctx.db, pid)
    return await asyncio.to_thread(library.list_scenarios, sctx.db, pid, run=run, tag=tag, status=status, q=q,
                                   archived=archived)


@router.post("/projects/{pid}/scenarios", status_code=201, response_model=S.Scenario)
async def create_scenario(pid: str, body: S.ScenarioCreate, sctx: StudioContext = Depends(get_ctx)):
    row = await asyncio.to_thread(library.create, sctx.db, sctx.workspace, pid, body.doc)
    return _scenario(sctx, row)


@router.get("/scenarios/{sid}", response_model=S.Scenario)
async def get_scenario(sid: str, sctx: StudioContext = Depends(get_ctx)):
    return _scenario(sctx, library.get_row(sctx.db, sid))


@router.patch("/scenarios/{sid}", response_model=S.Scenario)
async def patch_scenario(sid: str, body: S.ScenarioPatch, sctx: StudioContext = Depends(get_ctx)):
    data = body.model_dump(exclude_unset=True)
    if body.doc is not None:
        data["doc"] = body.doc
    row = await asyncio.to_thread(library.patch, sctx.db, sctx.workspace, sid, data)
    return _scenario(sctx, row)


@router.post("/scenarios/{sid}/fork", status_code=201, response_model=S.Scenario)
async def fork_scenario(sid: str, body: S.ForkRequest | None = None, sctx: StudioContext = Depends(get_ctx)):
    body = body or S.ForkRequest()
    row = await asyncio.to_thread(library.fork, sctx.db, sctx.workspace, sid, name=body.name, doc=body.doc)
    return _scenario(sctx, row)


@router.delete("/scenarios/{sid}", response_model=Ok)
async def delete_scenario(sid: str, results: bool = False, force: bool = False,
                          sctx: StudioContext = Depends(get_ctx)):
    await asyncio.to_thread(library.delete, sctx.db, sctx.workspace, sid, results=results, force=force)
    return {"ok": True}


async def _run_scenario(sctx: StudioContext, sid: str, run_id: str, force: bool) -> JSONResponse:
    row = library.get_row(sctx.db, sid)
    _has_checkpoint(sctx, run_id)
    svc = get_service(sctx)
    await asyncio.to_thread(svc.preflight, run_id, memory=False)
    ctx = _ctx(sctx, run_id)
    if ctx.project_id and row.get("project_id") and ctx.project_id != row["project_id"]:
        raise ApiError("validation", f"run {run_id} belongs to another project",
                       detail={"errors": [{"path": "run_id", "message": "another project", "code": "project"}]})

    def check():
        store.refresh_stale(sctx.db, run_id, ctx.run_dir)
        comp = _compile(sctx, ctx, library._doc(row))
        if comp.blocking:
            raise ApiError("validation", "; ".join(w["message"] for w in comp.blocking),
                           detail={"warnings": comp.blocking, "errors": [
                               {"path": f"edits.{w.get('edit_index')}", "message": w["message"], "code": w["code"]}
                               for w in comp.blocking]})
        hit = None if force else store.find_cached(sctx.db, run_id, comp.content_hash,
                                                   store.checkpoint_key(ctx.run_dir), store.code_sha(),
                                                   prefer_scenario=sid)
        if hit is not None and hit.get("scenario_id") != sid:
            # computed for another scenario with the same content: this scenario gets its own copy
            doc = library._doc(row)
            hit = store.adopt_result(sctx.db, hit, {"id": sid, "revision": row.get("revision"),
                                                    "name": doc.get("name") or row.get("name")})
        return hit

    hit = await asyncio.to_thread(check)
    if hit is not None:
        return JSONResponse({"cached": store.summary_out(hit, _unit(ctx)), "job": None}, status_code=200)
    await asyncio.to_thread(svc.preflight, run_id)          # the engine is needed: memory, unless loaded
    job = await _jobs(sctx).submit("engine.scenario", {"run_id": run_id, "scenario_id": sid,
                                                       "revision": int(row.get("revision") or 1)},
                                   run_id=run_id, scenario_id=sid, project_id=row.get("project_id"),
                                   label=f"Exact: {library._doc(row).get('name') or sid}")
    return JSONResponse({"cached": None, "job": job}, status_code=202)


@router.post("/scenarios/{sid}/run", response_model=S.RunScenarioResponse, status_code=202,
             responses={200: {"model": S.RunScenarioResponse, "description": "a cache hit"}})
async def run_scenario(sid: str, body: S.RunScenarioRequest, sctx: StudioContext = Depends(get_ctx)):
    return await _run_scenario(sctx, sid, body.run_id, body.force)


@router.post("/scenarios/run-batch", status_code=202, response_model=S.JobOut)
async def run_batch(body: S.BatchRequest, sctx: StudioContext = Depends(get_ctx)):
    _has_checkpoint(sctx, body.run_id)
    await asyncio.to_thread(get_service(sctx).preflight, body.run_id)
    rows = [library.get_row(sctx.db, s) for s in body.scenario_ids]
    job = await _jobs(sctx).submit("engine.batch", {"run_id": body.run_id, "scenario_ids": body.scenario_ids},
                                   run_id=body.run_id, project_id=rows[0].get("project_id"),
                                   label=f"Exact: {len(rows)} scenarios")
    return {"job": job}


@router.post("/scenarios/{sid}/ladder", status_code=201, response_model=S.LadderResponse)
async def make_ladder(sid: str, body: S.LadderRequest, sctx: StudioContext = Depends(get_ctx)):
    row = library.get_row(sctx.db, sid)
    _lid, docs = library.ladder_docs(row, body.edit_index, body.amounts)
    run_id = body.run_id or library._doc(row).get("anchor_run_id")
    run = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (run_id,)) if run_id else None
    if body.run_id and run is None:
        raise ApiError("not_found", f"no run {body.run_id!r}")
    exact = run is not None and (Path(run["run_dir"]) / "checkpoint.pkl").is_file()
    if exact:                                    # refuse (untrusted pickle, memory) before forking anything
        await asyncio.to_thread(get_service(sctx).preflight, run_id)
    made = []
    for d in docs:
        made.append(await asyncio.to_thread(library.fork, sctx.db, sctx.workspace, sid, doc=d))
    job = None
    if exact:
        job = await _jobs(sctx).submit("engine.batch", {"run_id": run_id, "scenario_ids": [m["id"] for m in made]},
                                       run_id=run_id, project_id=row.get("project_id"),
                                       label=f"Ladder: {len(made)} doses")
    return {"scenarios": [_scenario(sctx, m) for m in made], "job": job}


@router.post("/scenarios/{sid}/across-runs/estimate", response_model=S.AcrossRunsEstimate)
async def across_estimate(sid: str, body: S.AcrossRunsRequest, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.engine.kinds import across_runs_estimate

    return await asyncio.to_thread(across_runs_estimate, sctx, sid, body.run_ids)


@router.post("/scenarios/{sid}/across-runs", status_code=202, response_model=Job)
async def across_runs(sid: str, body: S.AcrossRunsRequest, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.engine.kinds import across_runs_estimate

    row = library.get_row(sctx.db, sid)
    est = await asyncio.to_thread(across_runs_estimate, sctx, sid, body.run_ids)
    if not any(r["ok"] for r in est["runs"]):
        raise ApiError("validation", "none of these runs can evaluate the scenario",
                       detail={"errors": [{"path": "run_ids", "message": r["reason"] or "unusable", "code": "run"}
                                          for r in est["runs"]], "runs": est["runs"]})
    return await _jobs(sctx).submit("scenario.across_runs", {"scenario_id": sid, "run_ids": body.run_ids},
                                    project_id=row.get("project_id"), scenario_id=sid,
                                    label=f"Check across runs: {library._doc(row).get('name') or sid}")


@router.post("/scenarios/{sid}/promote", response_model=S.PromoteResponse)
async def promote(sid: str, body: S.PromoteRequest | None = None, sctx: StudioContext = Depends(get_ctx)):
    body = body or S.PromoteRequest()
    return await asyncio.to_thread(library.promote, sctx.db, sid, apply=body.apply)


@router.get("/scenarios/{sid}/design.csv", response_class=Response,
            responses={200: {"content": {"text/csv": {}}}})
async def design_csv(sid: str, run_id: str = Query(...), sctx: StudioContext = Depends(get_ctx)):
    ctx = _ctx(sctx, run_id)
    text = await asyncio.to_thread(library.design_csv, sctx.db, ctx, sid, reader=_reader(sctx),
                                   project_dir=_project_dir(sctx, ctx.project_id))
    return Response(text, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{sid}_{run_id}_design.csv"', "Cache-Control": "no-store"})


@router.post("/runs/{rid}/designs/import", status_code=201, response_model=S.DesignImport,
             openapi_extra={"requestBody": {"content": {"text/csv": {"schema": {"type": "string"}}},
                                            "required": True}})
async def import_design(rid: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    ctx = _ctx(sctx, rid)
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > DESIGN_MAX_BYTES:
            raise ApiError("too_large", "the design CSV is too large", detail={"max_bytes": DESIGN_MAX_BYTES})
        chunks.append(chunk)
    return await asyncio.to_thread(library.import_design, sctx.db, ctx, b"".join(chunks))


# ---------------------------------------------------------------------------
# results (api.md §7.5)
# ---------------------------------------------------------------------------

@router.get("/runs/{rid}/scenarios", response_model=S.RunScenarios)
async def run_scenarios(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = _ctx(sctx, rid)

    def build():
        store.refresh_stale(sctx.db, rid, ctx.run_dir)
        return {"configured": library.configured_list(ctx), "results": store.run_results(sctx.db, rid, _unit(ctx))}

    return await asyncio.to_thread(build)


@router.post("/runs/{rid}/configured/{slug}/rerun-exact", status_code=202, response_model=Job)
async def rerun_configured(rid: str, slug: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = _ctx(sctx, rid)
    _has_checkpoint(sctx, rid)
    await asyncio.to_thread(get_service(sctx).preflight, rid)
    match = next((s for s in ctx.configured_scenarios() if s["slug"] == slug), None)
    if match is None:
        raise ApiError("not_found", f"no configured scenario {slug!r} on this run")
    return await _jobs(sctx).submit("engine.rerun_configured", {"run_id": rid, "slug": slug}, run_id=rid,
                                    project_id=ctx.project_id, label=f"Re-run exactly: {match['name']}")


def _result_ctx(sctx, res_id: str):
    row = store.result_row(sctx.db, res_id)
    return row, _ctx(sctx, row["run_id"])


@router.get("/results/{res_id}", response_model=S.Result)
async def get_result(res_id: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios.impacts import impacts_for_result

    row, ctx = _result_ctx(sctx, res_id)

    def build():
        store.refresh_stale(sctx.db, ctx.run_id, ctx.run_dir)
        cur = store.result_row(sctx.db, res_id)
        res = store.load_result(sctx.db, cur)
        if res.get("impacts") is None:
            try:
                res["impacts"] = impacts_for_result(ctx, cur, cache_dir=sctx.workspace.cache_dir)
            except ApiError:                     # no people layers: the result has no impacts section
                res["impacts"] = None
            except Exception:                    # optional section: never fail the result itself
                log.exception("impacts of %s failed", res_id)
                res["impacts"] = None
        return res

    return await asyncio.to_thread(build)


@router.get("/results/{res_id}/layers/{field}.bin", response_class=Response,
            responses={200: {"content": {"application/octet-stream": {}}}})
async def result_layer(res_id: str, field: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    row, ctx = _result_ctx(sctx, res_id)
    if field not in store.RESULT_FIELDS and not field.startswith("realized_"):
        raise ApiError("unknown_layer", f"unknown result field {field!r}")
    st = (Path(row["dir"]) / "cells.parquet").stat().st_mtime_ns if (Path(row["dir"]) / "cells.parquet").exists() \
        else 0
    etag = _etag(res_id, field, st)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})

    def build():
        obs = None
        if field == "abs":
            pred = ctx.predictions
            obs = pred["target"].to_numpy(np.float64) if pred is not None and "target" in pred else None
        return store.result_array(row, field, obs)

    return _f32(await asyncio.to_thread(build), etag)


@router.post("/results/{res_id}/impacts", response_model=S.Impacts)
async def result_impacts(res_id: str, body: S.ImpactsRequest | None = None, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios.impacts import impacts_for_result

    body = body or S.ImpactsRequest()
    row, ctx = _result_ctx(sctx, res_id)
    return await asyncio.to_thread(impacts_for_result, ctx, row, thresholds=body.thresholds, futures=body.futures,
                                   cache_dir=sctx.workspace.cache_dir)


@router.delete("/results/{res_id}", response_model=Ok)
async def delete_result(res_id: str, sctx: StudioContext = Depends(get_ctx)):
    row = store.result_row(sctx.db, res_id)
    if row.get("job_id"):
        j = sctx.db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
        if j is not None and j["status"] in ACTIVE_STATUSES:
            raise ApiError("active", "the job that writes this result is still running")
    await asyncio.to_thread(store.delete_result, sctx.db, res_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# compare (api.md §7.6)
# ---------------------------------------------------------------------------

@router.post("/runs/{rid}/compare", status_code=201, response_model=S.Comparison)
async def post_compare(rid: str, body: S.CompareRequest, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import compare

    ctx = _ctx(sctx, rid)
    return await asyncio.to_thread(compare.compare, sctx.db, ctx, body.items, regions=body.regions,
                                   thresholds=body.thresholds)


@router.get("/comparisons/{cid}", response_model=S.Comparison)
async def get_comparison(cid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import compare

    return await asyncio.to_thread(compare.comparison_out, sctx.db, cid)


@router.get("/runs/{rid}/comparisons", response_model=list[S.ComparisonRow])
async def list_comparisons(rid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import compare

    _ctx(sctx, rid)
    return compare.list_comparisons(sctx.db, rid)


@router.delete("/comparisons/{cid}", response_model=Ok)
async def delete_comparison(cid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import compare

    await asyncio.to_thread(compare.delete, sctx.db, cid)
    return {"ok": True}


# ---------------------------------------------------------------------------
# climate × adaptation (api.md §7.7)
# ---------------------------------------------------------------------------

@router.get("/runs/{rid}/climate/factors")
async def climate_factors(rid: str, sctx: StudioContext = Depends(get_ctx)) -> dict[str, Any]:
    from sparc.studio.scenarios import climate

    ctx = _ctx(sctx, rid)
    return await asyncio.to_thread(climate.factors_info, ctx, sctx.workspace.cache_dir)


@router.post("/runs/{rid}/climate/explore")
async def climate_explore(rid: str, body: S.ClimateExplore, sctx: StudioContext = Depends(get_ctx)) -> dict[str, Any]:
    from sparc.studio.scenarios import climate

    ctx = _ctx(sctx, rid)
    return await asyncio.to_thread(climate.explore, sctx.db, ctx, body.model_dump(mode="json", exclude_none=True),
                                   sctx.workspace.cache_dir)


# ---------------------------------------------------------------------------
# sweeps (api.md §7.8)
# ---------------------------------------------------------------------------

@router.post("/runs/{rid}/sweeps", status_code=202, response_model=S.SweepCreated)
async def post_sweep(rid: str, body: S.SweepRequest, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import sweeps

    ctx = _ctx(sctx, rid)
    _has_checkpoint(sctx, rid)
    await asyncio.to_thread(get_service(sctx).preflight, rid)
    sel = body.model_dump(mode="json", exclude_none=True).get("selection")
    params = await asyncio.to_thread(sweeps.create, sctx.db, ctx, body.lever, body.doses, sel)
    job = await _jobs(sctx).submit("engine.sweep", {"run_id": rid, "sweep_id": params["id"]}, run_id=rid,
                                   project_id=ctx.project_id, label=f"Sweep: {body.lever}")
    from sparc.studio import db as dbmod
    from sparc.studio.workspace import write_json_atomic

    params = {**params, "job_id": job["id"]}
    await asyncio.to_thread(sctx.db.update, "sweeps", {"id": params["id"]},
                            {"job_id": job["id"], "params_json": dbmod.dumps(params)})
    write_json_atomic(Path(sweeps.sweep_row(sctx.db, params["id"])["dir"]) / "params.json", params)
    return {"sweep_id": params["id"], "job": job}


@router.get("/sweeps/{swid}", response_model=S.Sweep)
async def get_sweep(swid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import sweeps

    row = sweeps.sweep_row(sctx.db, swid)
    ctx = _ctx(sctx, row["run_id"])
    return await asyncio.to_thread(sweeps.sweep_out, sctx.db, ctx, row)


@router.get("/runs/{rid}/sweeps", response_model=list[S.SweepRow])
async def list_sweeps(rid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import sweeps

    _ctx(sctx, rid)
    return sweeps.list_sweeps(sctx.db, rid)


@router.delete("/sweeps/{swid}", response_model=Ok)
async def delete_sweep(swid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.scenarios import sweeps

    await asyncio.to_thread(sweeps.delete, sctx.db, swid)
    return {"ok": True}
