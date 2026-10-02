"""Studies and post-run actions endpoints (api.md §9, SPEC §8): status rows, launches, estimates, the
demo's truth card, study listings and views, resume, attach / detach, the inline simcheck merge and
deletes."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError, validation_error
from sparc.studio.runs.schemas import RunSummary
from sparc.studio.schemas.common import Action, Job, Ok
from sparc.studio.studies import kinds as K
from sparc.studio.studies import service, views

router = APIRouter(tags=["studies"])


# ---------------------------------------------------------------------------
# wire models (api.md §9)
# ---------------------------------------------------------------------------

class Requirements(BaseModel):
    ok: bool
    missing: list[str] = Field(default_factory=list)


class Estimate(BaseModel):
    est_s: float
    est_lo: float
    est_hi: float


class StudyStatusRow(BaseModel):
    kind: Literal["baselines", "planner", "emulator", "uncertainty", "writeup", "placebo", "simcheck", "multiverse",
                  "reproduce", "literature", "benchmark"]
    state: Literal["not_run", "queued", "running", "done", "stale", "failed"]
    study_id: str | None = None
    job_id: str | None = None
    updated_utc: str | None = None
    headline: str | None = None
    estimate: Estimate | None = None
    attached: bool | None = None
    action: Action | None = None
    requirements: Requirements


class Study(BaseModel):
    id: str
    project_id: str
    kind: str
    target_run_id: str | None = None
    job_id: str | None = None
    out_dir: str
    status: str
    params: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] | None = None
    origin: Literal["studio", "imported"]
    created_utc: str
    updated_utc: str
    children: list[RunSummary] = Field(default_factory=list)
    attached_runs: list[str] = Field(default_factory=list)
    stale_vs: list[str] = Field(default_factory=list)


class StudyLaunch(BaseModel):
    study: Study
    job: Job


class StudyLink(BaseModel):
    study: Study
    job: Job | None = None


class EstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    run_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class StudyCost(BaseModel):
    est_s: float
    est_lo: float
    est_hi: float
    est_peak_rss_gb: float
    est_disk_gb: float
    n_children: int


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    study_ids: list[str] = Field(min_length=1)


class MergeResponse(BaseModel):
    summary: dict[str, Any]
    markdown: str


class TruthRow(BaseModel):
    quantity: Literal["canopy_scenario", "footprint_mean", "L_m", "influence_radius_m", "noise_sd"]
    label: str
    truth: float
    recovered: float | None = None
    se: float | None = None
    share: float | None = None
    unit: str
    scenario: str | None = None


class TruthResponse(BaseModel):
    rows: list[TruthRow]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reader(sctx: StudioContext):
    reader = sctx.services.get("reader")
    if reader is None:
        raise ApiError("not_ready", "the runs registry is still starting", status=503)
    return reader


def _jobs(sctx: StudioContext):
    if sctx.jobs is None:
        raise ApiError("not_ready", "the job manager is still starting", status=503)
    return sctx.jobs


def _requirements_error(kind: str, req: dict, rid: str) -> ApiError:
    what = ", ".join(req["missing"])
    return ApiError("requirements", f"{kind} needs {what} in run {rid}", detail={"missing": req["missing"],
                                                                                    "kind": kind})


def _validate(kind: str, params: dict | None):
    model = K.PARAMS[kind]
    from pydantic import ValidationError

    from sparc.studio.errors import pydantic_errors

    try:
        return model.model_validate(dict(params or {}))
    except ValidationError as exc:
        rows = pydantic_errors(exc.errors())
        for r in rows:
            r["path"] = f"params.{r['path']}" if r["path"] else "params"
        raise validation_error(rows, f"invalid params for {kind}")


def _pool_check(sctx: StudioContext, p) -> None:
    """``workers × threads`` must fit the heavy slot (``threads_heavy``), SPEC §8."""
    cap = int(sctx.settings().threads_heavy)
    if int(p.workers) * int(p.threads) > cap:
        raise validation_error([{"path": "params.workers", "code": "threads_heavy",
                                 "message": f"workers × threads = {int(p.workers) * int(p.threads)} exceeds "
                                            f"threads_heavy ({cap})"}],
                               f"workers × threads must not exceed threads_heavy ({cap})")


def _check_multiverse(p) -> None:
    from sparc.core.multiverse import VARIANTS

    bad = [v for v in (p.variants or []) if v not in VARIANTS and v not in (p.custom_variants or {})]
    clash = sorted(set(p.custom_variants or {}) & set(VARIANTS))
    errs = []
    if bad:
        errs.append({"path": "params.variants", "code": "unknown_variant",
                     "message": f"unknown variants {bad}; built-in: {sorted(VARIANTS)}"})
    if clash:
        errs.append({"path": "params.custom_variants", "code": "clash",
                     "message": f"custom variants {clash} reuse built-in names"})
    if errs:
        raise validation_error(errs, "invalid multiverse variants")


async def _check_action_params(sctx: StudioContext, ctx, kind: str, p) -> None:
    if kind == "planner" and p.package:
        try:
            await asyncio.to_thread(K._package_name, ctx.run_dir, p.package)
        except ValueError as exc:
            raise validation_error([{"path": "params.package", "code": "unknown_scenario", "message": str(exc)}],
                                   str(exc))
    if kind == "uncertainty":
        want = [("multiverse", p.multiverse_study)] + [("simcheck", s) for s in p.simcheck_studies or []] + \
            [("placebo", p.placebo_study)]
        errs = []
        for k, sid in want:
            if not sid:
                continue
            row = sctx.db.fetchone("SELECT kind FROM studies WHERE id = ?", (sid,))
            if row is None or row["kind"] != k:
                errs.append({"path": f"params.{k}_stud{'ies' if k == 'simcheck' else 'y'}", "code": "study",
                             "message": f"{sid} is not a {k} study"})
        if errs:
            raise validation_error(errs, "unknown studies for the uncertainty report")


async def _submit_study(sctx: StudioContext, row: dict, params: dict, *, run_id: str | None,
                        label: str) -> dict:
    """Queue the study's job and record it on the row."""
    job = await _jobs(sctx).submit(f"study.{row['kind']}", params, run_id=run_id, project_id=row.get("project_id"),
                                   study_id=row["id"], label=label)
    await asyncio.to_thread(service.update_study, sctx.db, row["id"], sctx.hub, job_id=job["id"], status="queued")
    return job


def _drop_new_study(sctx: StudioContext, row: dict) -> None:
    def fn(conn):
        conn.execute("DELETE FROM studies WHERE id = ?", (row["id"],))
        conn.execute("DELETE FROM study_links WHERE study_id = ?", (row["id"],))

    sctx.db.run(fn)
    if row.get("origin") == "studio":
        shutil.rmtree(row["out_dir"], ignore_errors=True)


def _label(kind: str, ctx) -> str:
    return f"{views_label(kind)} · {ctx.row.get('label') or ctx.run_id}"


def views_label(kind: str) -> str:
    return {"placebo": "Placebo study", "simcheck": "Simulation check", "multiverse": "Multiverse",
            "reproduce": "Reproduction", "benchmark": "Effect benchmark", "baselines": "Reference baselines",
            "planner": "Planner pack", "emulator": "Scenario emulator", "uncertainty": "Uncertainty report",
            "writeup": "Methods & model card"}.get(kind, kind)


# ---------------------------------------------------------------------------
# run status and launches
# ---------------------------------------------------------------------------

@router.get("/runs/{rid}/studies", response_model=list[StudyStatusRow])
async def run_studies(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = _reader(sctx).get(rid)
    return await asyncio.to_thread(views.status_rows, sctx, ctx)


@router.post("/runs/{rid}/actions/{kind}", status_code=202, response_model=Job)
async def run_action(rid: str, kind: str, body: dict[str, Any] | None = Body(None),
                     sctx: StudioContext = Depends(get_ctx)):
    if kind not in service.POST_KINDS:
        raise ApiError("unknown_kind", f"unknown post-run action {kind!r}", detail={"kinds": list(service.POST_KINDS)})
    ctx = _reader(sctx).get(rid)
    req = await asyncio.to_thread(views.requirements, ctx, kind)
    if "checkpoint" in req["missing"]:
        raise ApiError("no_checkpoint", f"{views_label(kind)} needs the run's checkpoint.pkl",
                       detail={"run_id": rid, "missing": req["missing"]})
    if not req["ok"]:
        raise _requirements_error(kind, req, rid)
    p = _validate(kind, body)
    await _check_action_params(sctx, ctx, kind, p)
    return await _jobs(sctx).submit(f"post.{kind}", p, run_id=rid, project_id=ctx.project_id,
                                    label=_label(kind, ctx))


@router.post("/runs/{rid}/studies/{kind}", status_code=202, response_model=StudyLaunch)
async def run_study(rid: str, kind: str, body: dict[str, Any] | None = Body(None),
                    sctx: StudioContext = Depends(get_ctx)):
    if kind not in service.RUN_STUDY_KINDS:
        raise ApiError("unknown_kind", f"unknown study kind {kind!r}", detail={"kinds": list(service.RUN_STUDY_KINDS)})
    ctx = _reader(sctx).get(rid)
    if not ctx.project_id:
        raise ApiError("validation", "a study needs a run that belongs to a project",
                       detail={"errors": [{"path": "run_id", "message": "no project", "code": "project"}]})
    req = await asyncio.to_thread(views.requirements, ctx, kind)
    if not req["ok"]:
        raise _requirements_error(kind, req, rid)
    p = _validate(kind, body)
    if kind in ("simcheck", "multiverse"):
        _pool_check(sctx, p)
    if kind == "multiverse":
        _check_multiverse(p)
    params = p.model_dump(mode="json")
    if kind == "simcheck" and p.continue_study_id:
        old = sctx.db.fetchone("SELECT * FROM studies WHERE id = ?", (p.continue_study_id,))
        if old is None or old["kind"] != "simcheck" or old.get("project_id") != ctx.project_id:
            raise validation_error([{"path": "params.continue_study_id", "code": "study",
                                     "message": f"{p.continue_study_id} is not a simcheck study of this project"}],
                                   "cannot continue that study")
        if service.effective_status(sctx.db, old) in service.ACTIVE_JOB:
            raise ApiError("active", "that simulation check is still running", detail={"study_id": old["id"]})
        if old.get("origin") != "studio":
            raise ApiError("imported_in_place", "an imported simulation check is read-only; start a new one",
                           detail={"study_id": old["id"]})
        row = await asyncio.to_thread(service.update_study, sctx.db, old["id"], sctx.hub, params=params,
                                      status="queued")
        job = await _submit_study(sctx, row, params, run_id=rid, label=_label(kind, ctx))
    else:
        row = await asyncio.to_thread(lambda: service.create_study(
            sctx.db, project_id=ctx.project_id, kind=kind, target_run_id=rid, params=params, hub=sctx.hub))
        try:
            job = await _submit_study(sctx, row, params, run_id=rid, label=_label(kind, ctx))
        except BaseException:
            await asyncio.to_thread(_drop_new_study, sctx, row)
            raise
    row = await asyncio.to_thread(service.get_row, sctx.db, row["id"])
    return {"study": await asyncio.to_thread(service.study_out, sctx.db, row), "job": job}


@router.post("/projects/{pid}/studies/benchmark", status_code=202, response_model=StudyLaunch)
async def project_benchmark(pid: str, body: dict[str, Any] | None = Body(None),
                            sctx: StudioContext = Depends(get_ctx)):
    if sctx.db.fetchone("SELECT id FROM projects WHERE id = ?", (pid,)) is None:
        raise ApiError("not_found", f"no project {pid!r}")
    p = _validate("benchmark", body)
    params = p.model_dump(mode="json")
    row = await asyncio.to_thread(lambda: service.create_study(
        sctx.db, project_id=pid, kind="benchmark", target_run_id=None, params=params, hub=sctx.hub))
    try:
        job = await _submit_study(sctx, row, params, run_id=None, label=f"Effect benchmark (n={p.n}, seed {p.seed})")
    except BaseException:
        await asyncio.to_thread(_drop_new_study, sctx, row)
        raise
    row = await asyncio.to_thread(service.get_row, sctx.db, row["id"])
    return {"study": await asyncio.to_thread(service.study_out, sctx.db, row), "job": job}


@router.post("/studies/estimate", response_model=StudyCost)
async def estimate(body: EstimateRequest, sctx: StudioContext = Depends(get_ctx)):
    if body.kind not in K.PARAMS:
        raise ApiError("unknown_kind", f"no estimate for {body.kind!r}", detail={"kinds": sorted(K.PARAMS)})
    p = _validate(body.kind, body.params)
    run_row = None
    if body.run_id:
        run_row = sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (body.run_id,))
        if run_row is None:
            raise ApiError("not_found", f"no run {body.run_id!r}")
    e = await asyncio.to_thread(K.estimate_kind, sctx.db, body.kind, run_row, p)
    return {"est_s": e["est_s"], "est_lo": e["est_lo"], "est_hi": e["est_hi"],
            "est_peak_rss_gb": e["peak_ram_gb"], "est_disk_gb": round(e["disk_bytes"] / 1e9, 3),
            "n_children": e["n_children"]}


@router.get("/runs/{rid}/truth", response_model=TruthResponse)
async def run_truth(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = _reader(sctx).get(rid)
    return await asyncio.to_thread(views.truth_rows, sctx, ctx)


# ---------------------------------------------------------------------------
# reading and managing studies
# ---------------------------------------------------------------------------

@router.post("/studies/simcheck/merge", response_model=MergeResponse)
async def simcheck_merge(body: MergeRequest, sctx: StudioContext = Depends(get_ctx)):
    dirs = []
    for sid in body.study_ids:
        row = sctx.db.fetchone("SELECT kind, out_dir FROM studies WHERE id = ?", (sid,))
        if row is None:
            raise ApiError("not_found", f"no study {sid!r}")
        if row["kind"] != "simcheck":
            raise validation_error([{"path": "study_ids", "code": "kind", "message": f"{sid} is not a simcheck study"}],
                                   "only simulation checks can be merged")
        dirs.append(Path(row["out_dir"]))

    def merge():
        from sparc.core.simcheck import merge_results, simcheck_markdown, summarize

        rows = merge_results(dirs)
        summ = summarize(rows) if rows else {"generators": {}, "bias_correction": {}, "n_rows": 0, "n_errors": 0}
        from sparc.studio.runs.common import clean

        return {"summary": clean(summ), "markdown": simcheck_markdown(summ) if rows else ""}

    return await asyncio.to_thread(merge)


@router.get("/projects/{pid}/studies", response_model=list[Study])
async def project_studies(pid: str, kind: str | None = Query(None), sctx: StudioContext = Depends(get_ctx)):
    if sctx.db.fetchone("SELECT id FROM projects WHERE id = ?", (pid,)) is None:
        raise ApiError("not_found", f"no project {pid!r}")
    return await asyncio.to_thread(service.list_studies, sctx.db, pid, kind)


@router.get("/studies/{stid}", response_model=Study)
async def get_study(stid: str, sctx: StudioContext = Depends(get_ctx)):
    row = service.get_row(sctx.db, stid)
    return await asyncio.to_thread(service.study_out, sctx.db, row)


@router.get("/studies/{stid}/view")
async def study_view(stid: str, sctx: StudioContext = Depends(get_ctx)) -> dict:
    from sparc.studio.runs.common import clean

    row = service.get_row(sctx.db, stid)
    return clean(await asyncio.to_thread(views.study_view, sctx.db, row))


@router.post("/studies/{stid}/resume", status_code=202, response_model=Job)
async def resume_study(stid: str, sctx: StudioContext = Depends(get_ctx)):
    row = service.get_row(sctx.db, stid)
    if row.get("origin") != "studio":
        raise ApiError("not_resumable", "an imported study folder cannot be resumed from Studio",
                       detail={"study_id": stid, "origin": row.get("origin")})
    if service.effective_status(sctx.db, row) in service.ACTIVE_JOB:
        raise ApiError("active", "the study's job is still running", detail={"job_id": row.get("job_id")})
    rid = row.get("target_run_id")
    if row["kind"] != "benchmark":
        if not rid or sctx.db.fetchone("SELECT id FROM runs WHERE id = ?", (rid,)) is None:
            raise ApiError("not_resumable", "the study's run no longer exists", detail={"run_id": rid})
    from sparc.studio import db as dbmod

    params = dbmod.loads(row.get("params_json"), {}) or {}
    p = _validate(row["kind"], params)
    if row["kind"] in ("simcheck", "multiverse"):
        _pool_check(sctx, p)
    label = f"{views_label(row['kind'])} (resumed)"
    return await _submit_study(sctx, row, p.model_dump(mode="json"), run_id=rid, label=label)


async def _link(sctx: StudioContext, stid: str, run_id: str, attached: bool) -> dict:
    row = await asyncio.to_thread(service.set_links, sctx.db, stid, run_id, attached, sctx.hub)
    job = None
    finished = service.effective_status(sctx.db, row) in ("succeeded", "done")
    run = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (run_id,))
    has_report = run is not None and (Path(run["run_dir"]) / "uncertainty.json").is_file()
    if finished and (attached or has_report) and sctx.settings().auto_uncertainty:
        job = await K.submit_uncertainty(sctx, run_id)
    return {"study": await asyncio.to_thread(service.study_out, sctx.db, row), "job": job}


@router.post("/studies/{stid}/attach", response_model=StudyLink)
async def attach_study(stid: str, body: LinkRequest, sctx: StudioContext = Depends(get_ctx)):
    return await _link(sctx, stid, body.run_id, True)


@router.post("/studies/{stid}/detach", response_model=StudyLink)
async def detach_study(stid: str, body: LinkRequest, sctx: StudioContext = Depends(get_ctx)):
    return await _link(sctx, stid, body.run_id, False)


@router.delete("/studies/{stid}", response_model=Ok)
async def delete_study(stid: str, files: bool = Query(False), sctx: StudioContext = Depends(get_ctx)):
    row = service.get_row(sctx.db, stid)
    await asyncio.to_thread(service.delete_study, sctx.db, stid, files=files)
    service.publish(sctx.hub, row, status="deleted")
    return {"ok": True}
