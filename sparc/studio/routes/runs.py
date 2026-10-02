"""Runs endpoints (api.md §6, §6.1): listing, plan and launch, import, detail, resume and rerun, the Status
Board, timeline, manifest, config, provenance, environment, outputs, views, docs, files and the dictionary.

This module also starts the runs services at server start (SPEC §10.9): the
:class:`~sparc.studio.runs.reader.RunReader` (``sctx.services["reader"]``) and the
:class:`~sparc.studio.runs.registry.Registry` (``sctx.services["registry"]``), which scans the workspace,
the imported runs and the watch roots, then follows live CLI runs every 10 s.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import FileResponse, Response

from sparc.studio.app import StudioContext, get_ctx, shutdown_hook, startup_hook
from sparc.studio.errors import ApiError
from sparc.studio.jobs.manager import job_out
from sparc.studio.runs import launch as launchmod
from sparc.studio.runs import outputs as outmod
from sparc.studio.runs import registry as regmod
from sparc.studio.runs import statusboard as sb
from sparc.studio.runs.common import clean, file_stat
from sparc.studio.runs.reader import RunReader, load_config_raw
from sparc.studio.runs.schemas import (
    DictionaryRow,
    DocContent,
    DocEntry,
    FileEntry,
    FileTable,
    FreedBytes,
    ImportRequest,
    LaunchRequest,
    LaunchResponse,
    OutputsBatch,
    OutputsResponse,
    PlanRequest,
    RerunRequest,
    RerunResponse,
    ResumeRequest,
    RunConfig,
    RunDetail,
    RunEnvironment,
    RunPage,
    RunPatch,
    RunPlan,
    RunProvenance,
    RunSummary,
    StatusBoard,
    Timeline,
    ViewModel,
)
from sparc.studio.schemas.common import ACTIVE_STATUSES, Job

router = APIRouter(tags=["runs"])


# ---------------------------------------------------------------------------
# services
# ---------------------------------------------------------------------------

@startup_hook("runs.registry", phase="scan", order=10)
def _start_registry(sctx: StudioContext) -> None:
    reader = RunReader(sctx.db)
    reg = regmod.Registry(sctx)
    sctx.services["reader"] = reader
    sctx.services["registry"] = reg
    regmod._ACTIVE = reg
    reg.scan()


@startup_hook("runs.watch", phase="ready", order=50)
async def _start_watch(sctx: StudioContext) -> None:
    reg = sctx.services.get("registry")
    if reg is not None:
        reg.start()


@shutdown_hook("runs.watch", order=50)
async def _stop_watch(sctx: StudioContext) -> None:
    reg = sctx.services.get("registry")
    if reg is not None:
        await reg.stop()


def services(sctx: StudioContext) -> tuple[RunReader, regmod.Registry]:
    reader, reg = sctx.services.get("reader"), sctx.services.get("registry")
    if reader is None or reg is None:
        raise ApiError("not_ready", "the runs registry is still starting", status=503)
    return reader, reg


def run_context(sctx: StudioContext, rid: str):
    reader, _ = services(sctx)
    return reader.get(rid)


def active_job(sctx: StudioContext, rid: str) -> dict | None:
    marks = ",".join("?" for _ in ACTIVE_STATUSES)
    return sctx.db.fetchone(f"SELECT * FROM jobs WHERE run_id = ? AND status IN ({marks}) "
                            f"ORDER BY created_utc DESC", (rid, *ACTIVE_STATUSES))


def _last_run_job(sctx: StudioContext, rid: str) -> dict | None:
    return sctx.db.fetchone("SELECT * FROM jobs WHERE run_id = ? AND kind IN ('run.core','run.external') "
                            "ORDER BY created_utc DESC, rowid DESC", (rid,))


#: ``{job_id: (events file stat, {"stages": …})}`` of finished run jobs: the Status Board and the run pages read
#: the stage states of every run, and replaying a long finished events file on each request does not scale
_STAGES: "OrderedDict[str, tuple[tuple | None, dict]]" = OrderedDict()
_STAGES_MAX = 4096


async def projection(sctx: StudioContext, rid: str) -> tuple[dict | None, bool]:
    """``({"stages": …} of the run's latest run job's tracker state, job active?)``."""
    job = _last_run_job(sctx, rid)
    if job is None or sctx.jobs is None:
        return None, False
    active = job["status"] in ACTIVE_STATUSES
    st = None
    if not active and job["id"] not in sctx.jobs.tailers:
        st = file_stat(sctx.jobs.events_path(job))
        hit = _STAGES.get(job["id"])
        if hit is not None and hit[0] == st:
            _STAGES.move_to_end(job["id"])
            return hit[1], False
    try:
        state, _eta = await sctx.jobs.projection(job["id"])
    except Exception:
        return None, False
    out = {"stages": state.get("stages")}
    if not active and st is not None:
        _STAGES[job["id"]] = (st, out)
        while len(_STAGES) > _STAGES_MAX:
            _STAGES.popitem(last=False)
    return out, active


def live_stages(proj: dict | None, active: bool) -> dict | None:
    if not active or not proj or not proj.get("stages"):
        return None
    return {sid: (st or {}).get("state") for sid, st in proj["stages"].items()}


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------

def _encode(n: int) -> str:
    return base64.urlsafe_b64encode(str(n).encode()).decode().rstrip("=")


def _decode(c: str | None) -> int:
    if not c:
        return 0
    try:
        return max(0, int(base64.urlsafe_b64decode(c + "=" * (-len(c) % 4)).decode()))
    except Exception:
        raise ApiError("validation", "bad cursor", detail={"errors": [{"path": "cursor", "message": "invalid",
                                                                       "code": "cursor"}]})


def list_runs(sctx: StudioContext, *, project=None, status=None, mode=None, origin=None, q=None, sort="created_desc",
              limit=50, cursor=None) -> dict:
    _, reg = services(sctx)
    where, args = [], []
    for col, val in (("project_id", project), ("mode", mode), ("origin", origin)):
        if val:
            vals = [v for v in str(val).split(",") if v]
            where.append(f"{col} IN ({','.join('?' for _ in vals)})")
            args += vals
    if status:
        vals = [v for v in str(status).split(",") if v]
        where.append(f"status IN ({','.join('?' for _ in vals)})")
        args += vals
    if q:
        where.append("(id LIKE ? OR label LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where) if where else "")
    order = {"created_desc": " ORDER BY created_utc DESC, id DESC", "r2": " ORDER BY r2 IS NULL, r2 DESC, id",
             "duration": " ORDER BY created_utc DESC, id DESC"}.get(sort or "created_desc")
    if order is None:
        raise ApiError("validation", f"unknown sort {sort!r}",
                       detail={"errors": [{"path": "sort", "message": "created_desc, duration or r2", "code": "sort"}]})
    rows = sctx.db.fetchall(sql + order, args)
    items = reg.summaries(rows)
    if sort == "duration":
        items.sort(key=lambda s: (s["duration_s"] is None, -(s["duration_s"] or 0)))
    limit = max(1, min(int(limit or 50), 500))
    off = _decode(cursor)
    page = items[off:off + limit]
    nxt = _encode(off + limit) if off + limit < len(items) else None
    return {"items": page, "next_cursor": nxt}


@router.get("/runs", response_model=RunPage)
def get_runs(project: str | None = None, status: str | None = None, mode: str | None = None,
             origin: str | None = None, q: str | None = None, sort: str = "created_desc",
             limit: int = Query(50, ge=1, le=500), cursor: str | None = None,
             sctx: StudioContext = Depends(get_ctx)):
    return list_runs(sctx, project=project, status=status, mode=mode, origin=origin, q=q, sort=sort, limit=limit,
                     cursor=cursor)


@router.get("/projects/{pid}/runs", response_model=RunPage)
def get_project_runs(pid: str, status: str | None = None, mode: str | None = None, origin: str | None = None,
                     q: str | None = None, sort: str = "created_desc", limit: int = Query(50, ge=1, le=500),
                     cursor: str | None = None, sctx: StudioContext = Depends(get_ctx)):
    launchmod.project_row(sctx.db, pid)
    return list_runs(sctx, project=pid, status=status, mode=mode, origin=origin, q=q, sort=sort, limit=limit,
                     cursor=cursor)


# ---------------------------------------------------------------------------
# plan, launch, import
# ---------------------------------------------------------------------------

@router.post("/projects/{pid}/runs/plan", response_model=RunPlan)
async def post_plan(pid: str, body: PlanRequest, sctx: StudioContext = Depends(get_ctx)):
    project = launchmod.project_row(sctx.db, pid)
    resumable = None
    done: frozenset = frozenset()
    if body.resume_run_id:
        reader, _ = services(sctx)
        ctx = reader.get(body.resume_run_id)
        if ctx.project_id != pid:
            raise ApiError("validation", "resume_run_id is not a run of this project",
                           detail={"errors": [{"path": "resume_run_id", "message": "other project", "code": "run"}]})
        resumable = await asyncio.to_thread(launchmod.checkpoint_info, ctx, sctx.db,
                                            active=active_job(sctx, ctx.run_id) is not None)
        if not ctx.launch:
            raise ApiError("not_resumable", "this run has no launch snapshot", detail={"checkpoint": resumable})
        raw = ctx.launch["config_raw"]
        cdir = Path(ctx.launch.get("config_dir") or project["dir"])
        a = dict(ctx.args)
        if body.threads:
            a["threads"] = body.threads
        if resumable["present"] and resumable["reuses"]:
            done = frozenset(resumable["done"])
        args = a
    else:
        raw, cdir = launchmod._project_raw(project)
        args = launchmod.mode_args(body.mode, body.coarse_m, body.stages, body.cv_curve, body.threads)
    plan = await asyncio.to_thread(launchmod.compute_plan, sctx, raw, cdir, args, done=done, resumable=resumable)
    return clean({k: v for k, v in plan.items() if not k.startswith("_")})


@router.post("/projects/{pid}/runs", status_code=202, response_model=LaunchResponse)
async def post_launch(pid: str, body: LaunchRequest, sctx: StudioContext = Depends(get_ctx)):
    services(sctx)
    return await launchmod.launch_run(sctx, pid, body.model_dump())


@router.post("/runs/import", status_code=201, response_model=RunSummary)
async def post_import(body: ImportRequest, sctx: StudioContext = Depends(get_ctx)):
    _, reg = services(sctx)
    return await asyncio.to_thread(reg.import_run, body.dir, project_id=body.project_id, config_path=body.config_path,
                                   trust_pickles=body.trust_pickles)


# ---------------------------------------------------------------------------
# detail
# ---------------------------------------------------------------------------

def _run_flags(ctx) -> list[dict]:
    flags = []
    if ctx.data_error:
        code = "inputs.not_reproducible" if "not reproducible" in ctx.data_error else "data.unavailable"
        flags.append({"code": code, "severity": "warn", "message": ctx.data_error})
    if ctx.manifest_raw and ctx.folds_error and "not known" not in ctx.folds_error:
        flags.append({"code": "folds.unavailable", "severity": "warn", "message": ctx.folds_error})
    for sec, tag in ctx.sections.items():
        if tag.get("stale"):
            flags.append({"code": "output.stale", "severity": "warn",
                          "message": f"{sec} comes from a file older than the manifest (an earlier run in this "
                                     "folder); it may not match this run"})
        if tag.get("older_code"):
            flags.append({"code": "output.older_code", "severity": "info",
                          "message": f"{sec}: written by older core code (some details are not in this run)"})
    info = sb.stage_info(ctx.row)
    if info.get("link") == "inferred":
        flags.append({"code": "lineage.inferred", "severity": "info",
                      "message": "linked to its parent run by its folder name (inferred, no run_meta)"})
    if ctx.cfg_source == "import":
        flags.append({"code": "config.import", "severity": "info",
                      "message": "the run has no usable provenance config; Studio uses the config given at import"})
    return flags


def build_detail(sctx: StudioContext, ctx, proj: dict | None, active: bool) -> dict:
    _, reg = services(sctx)
    row = ctx.row
    m = ctx.manifest_raw or {}
    prov = m.get("provenance") or {}
    g = None
    try:
        g = ctx.grid
    except Exception:
        g = None
    qa = m.get("qa") or {}
    shape = qa.get("grid_shape") or ctx.meta.get("grid_shape") or ([g.ny, g.nx] if g is not None else None)
    a = ctx.args
    header = {"name": ctx.name, "created_utc": row.get("created_utc"), "git_commit": row.get("git_commit"),
              "git_dirty": None if row.get("git_dirty") is None else bool(row["git_dirty"]),
              "versions": m.get("versions"), "n_points": row.get("n_points") or (g.n if g is not None else None),
              "grid_shape": list(shape) if shape else None, "cell_m": ctx.cell_m, "fast": bool(a.get("fast")),
              "coarse_m": a.get("coarse"), "run_dir": str(ctx.run_dir), "demo": bool(row.get("demo"))}
    stages = [{k: r[k] for k in ("id", "state", "seconds", "source", "reason")} for r in sb.stage_rows(ctx, proj)]
    entries = outmod.output_entries(ctx, live=live_stages(proj, active), resumable=sb.can_resume(ctx, active))
    summ = {k: sum(1 for e in entries if e["state"] == k) for k in ("present", "missing", "stale", "writing")}
    summ["present"] += sum(1 for e in entries if e["state"] == "partial")
    summ["missing"] = sum(1 for e in entries if e["state"] == "missing" and e["_expected"])
    children = sctx.db.fetchall("SELECT * FROM runs WHERE parent_run_id = ? ORDER BY created_utc", (ctx.run_id,))
    jobs = sctx.db.fetchall("SELECT * FROM jobs WHERE run_id = ? ORDER BY created_utc DESC LIMIT 50", (ctx.run_id,))
    wc = sctx.db.fetchval("SELECT COALESCE(SUM(count), 0) FROM warnings WHERE job_id IN "
                          "(SELECT id FROM jobs WHERE run_id = ?)", (ctx.run_id,), default=0)
    launch = dict(ctx.launch) if ctx.launch else None
    if launch and "config_raw" in launch:
        launch["config_raw"] = None
    return {"run": reg.summary(row), "header": header, "state": ctx.run_state, "launch": launch, "stages": stages,
            "checkpoint": launchmod.checkpoint_info(ctx, sctx.db, active=active), "outputs_summary": summ,
            "sections": ctx.sections, "flags": [*([{"code": "qa." + str(f.get("code")), "severity":
                                                   str(f.get("severity") or "info"), "message": str(f.get("message"))}
                                                  for f in qa.get("flags") or [] if isinstance(f, dict)]),
                                                *_run_flags(ctx)],
            "children": reg.summaries(children), "jobs": [job_out(j) for j in jobs], "warnings_count": int(wc or 0),
            "_prov": prov}


@router.get("/runs/{rid}", response_model=RunDetail)
async def get_run(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    proj, active = await projection(sctx, rid)
    out = await asyncio.to_thread(build_detail, sctx, ctx, proj, active)
    out.pop("_prov", None)
    return clean(out)


@router.patch("/runs/{rid}", response_model=RunSummary)
def patch_run(rid: str, body: RunPatch, sctx: StudioContext = Depends(get_ctx)):
    reader, reg = services(sctx)
    reader.row(rid)
    vals: dict[str, Any] = {}
    data = body.model_dump(exclude_unset=True)
    if "label" in data:
        vals["label"] = data["label"]
    if "notes" in data:
        vals["notes"] = data["notes"]
    if "pinned" in data:
        vals["pinned"] = int(bool(data["pinned"]))
    if vals:
        sctx.db.update("runs", {"id": rid}, vals)
        sctx.hub.publish("run.updated", {"run_id": rid, "project_id": reader.row(rid).get("project_id"),
                                         "status": reader.row(rid)["status"], "fields": sorted(vals)})
    return reg.summary(reader.row(rid))


def _rm_checkpoint(ctx) -> int:
    freed = 0
    for name in ("checkpoint.pkl", "checkpoint.json"):
        p = ctx.run_dir / name
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    return freed


async def _evict_engine(sctx: StudioContext, rid: str) -> None:
    """Close ``rid`` in the engine host.  ``EngineService.evict`` blocks until the host's work lock is free (up to
    120 s while it serves another run), so it runs in a thread, never on the event loop."""
    eng = sctx.services.get("engine")
    fn = getattr(eng, "evict", None) if eng is not None else None
    if fn is None:
        return
    try:
        res = await asyncio.to_thread(fn, rid)
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        pass


@router.delete("/runs/{rid}", response_model=FreedBytes)
async def delete_run(rid: str, what: str = "all", force_files: bool = False, sctx: StudioContext = Depends(get_ctx)):
    reader, reg = services(sctx)
    row = reader.row(rid)
    if what not in ("checkpoint", "outputs", "all"):
        raise ApiError("validation", "what must be checkpoint, outputs or all",
                       detail={"errors": [{"path": "what", "message": "invalid", "code": "what"}]})
    act = active_job(sctx, rid)
    if act is not None and act["kind"] != "run.external":
        raise ApiError("active", "a job is running on this run", detail={"job_id": act["id"]})
    if act is not None:
        raise ApiError("active", "the run is live (a CLI run): it cannot be deleted while it runs",
                       detail={"job_id": act["id"]})
    ctx = reader.for_row(row)
    root = sctx.workspace.root.resolve()
    rd = ctx.run_dir.resolve()
    in_ws = rd == root or root in rd.parents
    freed = 0
    if what in ("checkpoint", "outputs") and not in_ws and not force_files:
        raise ApiError("imported_in_place", "this run was imported in place: Studio deletes files outside the "
                       "workspace only with force_files=true", detail={"run_dir": str(rd)})
    if what == "checkpoint":
        freed = await asyncio.to_thread(_rm_checkpoint, ctx)
        await _evict_engine(sctx, rid)
    elif what == "outputs":
        def rm_outputs() -> int:
            n = 0
            for p in rd.iterdir():
                if p.name == "studio":
                    continue
                n += p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            return n

        freed = await asyncio.to_thread(rm_outputs)
        await _evict_engine(sctx, rid)
    else:
        def rm_all() -> int:
            n = 0
            if in_ws or force_files:
                n += sum(f.stat().st_size for f in rd.rglob("*") if f.is_file()) if rd.exists() else 0
                shutil.rmtree(rd, ignore_errors=True)
            studio = Path(row["studio_dir"]).resolve()
            if studio.exists() and (root in studio.parents) and not (rd in studio.parents and not rd.exists()):
                n += sum(f.stat().st_size for f in studio.rglob("*") if f.is_file())
                imp = studio.parent if studio.parent.parent == sctx.workspace.imports_dir.resolve() else studio
                shutil.rmtree(imp, ignore_errors=True)
            return n

        freed = await asyncio.to_thread(rm_all)
        await _evict_engine(sctx, rid)
        await asyncio.to_thread(reg.forget, rid)
        reader.forget(rid)
        pid = row.get("project_id")
        if pid and sctx.db.fetchval("SELECT active_run_id FROM projects WHERE id = ?", (pid,)) == rid:
            await asyncio.to_thread(regmod.set_active_run, sctx.db, pid, None)
        sctx.hub.publish("run.updated", {"run_id": rid, "project_id": row.get("project_id"), "status": "deleted",
                                         "fields": ["deleted"]})
        return {"freed_bytes": int(freed)}
    reader.forget(rid)
    await asyncio.to_thread(reg.refresh, rid)
    return {"freed_bytes": int(freed)}


@router.delete("/runs/{rid}/checkpoint", response_model=FreedBytes)
async def delete_checkpoint(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return await delete_run(rid, what="checkpoint", force_files=True, sctx=sctx)


@router.post("/runs/{rid}/resume", status_code=202, response_model=Job)
async def post_resume(rid: str, body: ResumeRequest = Body(default_factory=ResumeRequest),
                      sctx: StudioContext = Depends(get_ctx)):
    services(sctx)
    return await launchmod.resume_run(sctx, rid, body.model_dump())


@router.post("/runs/{rid}/rerun", status_code=202, response_model=RerunResponse)
async def post_rerun(rid: str, body: RerunRequest = Body(default_factory=RerunRequest),
                     sctx: StudioContext = Depends(get_ctx)):
    services(sctx)
    return await launchmod.rerun_run(sctx, rid, body.model_dump())


# ---------------------------------------------------------------------------
# status board, timeline, manifest, config, provenance, environment
# ---------------------------------------------------------------------------

@router.get("/projects/{pid}/status-board", response_model=StatusBoard)
async def get_status_board(pid: str, sctx: StudioContext = Depends(get_ctx)):
    reader, reg = services(sctx)
    launchmod.project_row(sctx.db, pid)
    rows = sctx.db.fetchall("SELECT * FROM runs WHERE project_id = ? ORDER BY created_utc DESC, id DESC", (pid,))
    summaries = reg.summaries(rows)
    out = []
    for row, summ in zip(rows, summaries):
        ctx = reader.for_row(row)
        proj, active = await projection(sctx, row["id"])
        resumable = sb.can_resume(ctx, active)
        cells = await asyncio.to_thread(sb.board_row, ctx, sctx.db, sb.stage_rows(ctx, proj), resumable=resumable)
        out.append({"run": summ, "cells": cells})
    return clean({"columns": sb.COLUMNS, "rows": out})


@router.get("/runs/{rid}/timeline", response_model=Timeline)
async def get_timeline(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    proj, _ = await projection(sctx, rid)
    return clean(sb.timeline(ctx, sctx.db, proj))


@router.get("/runs/{rid}/manifest")
def get_manifest(rid: str, sctx: StudioContext = Depends(get_ctx)) -> dict:
    ctx = run_context(sctx, rid)
    if not ctx.manifest and not ctx.sections:
        raise ApiError("output_missing", "the run has no manifest yet (it is written when the run finishes)",
                       detail={"output": "manifest", "produced_by": "stage:finish", "expected_path": "manifest.json"})
    return clean({**ctx.manifest, "_sections": ctx.sections})


def _project_raw_for(sctx: StudioContext, ctx, *, snapshot: bool) -> dict | None:
    """The project's current raw config; with ``snapshot`` in its launch-snapshot form (paths absolute, the
    workspace cache, the project's ``runs``: SPEC §4.3), so a diff against a Studio run's ``launch.json``
    shows real edits rather than the path rewriting every launch does."""
    if not ctx.project_id:
        return None
    p = sctx.db.fetchone("SELECT config_path, dir FROM projects WHERE id = ?", (ctx.project_id,))
    if p is None or not Path(p["config_path"]).is_file():
        return None
    try:
        raw = load_config_raw(p["config_path"])
        if snapshot:
            raw = launchmod.absolutise(raw, Path(p["config_path"]).resolve().parent,
                                       cache_dir=sctx.workspace.cache_dir, runs_dir=Path(p["dir"]) / "runs")
        return raw
    except Exception:
        return None


@router.get("/runs/{rid}/config", response_model=RunConfig)
def get_config(rid: str, sctx: StudioContext = Depends(get_ctx)):
    import yaml

    from sparc.core.config import DEFAULTS, _deep_merge
    from sparc.studio.runs.compare import config_diff, flatten

    ctx = run_context(sctx, rid)
    imp = (ctx.import_rec or {}).get("config_path")
    from_launch = bool(ctx.launch and ctx.launch.get("config_raw") is not None)
    if from_launch:
        raw = ctx.launch["config_raw"]
    elif isinstance((ctx.manifest_raw or {}).get("config"), dict):
        raw = ctx.manifest_raw["config"]
    elif imp and Path(imp).is_file():
        raw = load_config_raw(imp)
    else:
        raise ApiError("needs_config", "this run has no usable config: import it again with a config path",
                       detail={"run_id": rid})
    raw = raw.get("core", raw)
    effective = ctx.cfg.raw if ctx.cfg is not None else _deep_merge(DEFAULTS, raw)
    proj = _project_raw_for(sctx, ctx, snapshot=from_launch)
    vs_project = config_diff(raw, proj, a_key="run", b_key="project") if proj is not None else []
    fe, fd = flatten(effective), flatten(DEFAULTS)
    vs_def = [{"path": p, "value": clean(v), "default": clean(fd.get(p))} for p, v in fe.items()
              if json.dumps(v, sort_keys=True, default=str) != json.dumps(fd.get(p), sort_keys=True, default=str)]
    return clean({"effective": effective, "raw": raw,
                  "yaml": yaml.safe_dump({"core": raw}, sort_keys=False, allow_unicode=True),
                  "source": ctx.cfg_source or ("launch" if ctx.launch else "manifest"),
                  "config_dir": ctx.config_dir or (ctx.launch or {}).get("config_dir") or str(ctx.run_dir),
                  "vs_project_diff": vs_project, "vs_defaults": vs_def})


@router.get("/runs/{rid}/provenance", response_model=RunProvenance)
def get_provenance(rid: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.compare import environment_packages

    ctx = run_context(sctx, rid)
    m = ctx.manifest_raw or {}
    prov = m.get("provenance")
    ck = ctx.checkpoint_json or {}
    hashes = {}
    for k, v in (("input_sha256", (prov or {}).get("input_sha256")), ("config_sha256", (prov or {}).get("config_sha256")),
                 ("code_sha256", (prov or {}).get("code_sha256")), ("checkpoint_code_sha256", ck.get("code_sha256")),
                 ("fingerprint", ck.get("fingerprint") or (ctx.run_state or {}).get("fingerprint"))):
        if v:
            hashes[k] = str(v)
    for k, v in (ck.get("sections") or {}).items():
        hashes[f"section:{k}"] = str(v)
    git = dict((prov or {}).get("git") or {})
    if not git and m.get("git_commit"):
        git = {"commit": m["git_commit"]}
    launch = dict(ctx.launch) if ctx.launch else None
    return clean({"provenance": prov, "git": git or None, "platform": (prov or {}).get("platform"), "hashes": hashes,
                  "environment": environment_packages(ctx), "launch": launch})


@router.get("/runs/{rid}/environment", response_model=RunEnvironment)
def get_environment(rid: str, diff_with: str | None = None, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.compare import env_diff, environment_packages

    ctx = run_context(sctx, rid)
    pk = environment_packages(ctx)
    out: dict[str, Any] = {"packages": pk}
    if diff_with:
        other = run_context(sctx, diff_with)
        out["diff"] = env_diff(environment_packages(other), pk)
    return out


# ---------------------------------------------------------------------------
# outputs and views
# ---------------------------------------------------------------------------

async def outputs_env(sctx: StudioContext, ctx) -> dict:
    proj, active = await projection(sctx, ctx.run_id)
    any_active = active_job(sctx, ctx.run_id) is not None
    live = live_stages(proj, active)
    entries = await asyncio.to_thread(outmod.output_entries, ctx, live=live,
                                      resumable=sb.can_resume(ctx, any_active))
    tabs = outmod.tab_availability(ctx, entries, job_active=any_active)
    return {"proj": proj, "active": active, "entries": entries, "tabs": tabs}


def _public_entry(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}


@router.get("/runs/{rid}/outputs", response_model=OutputsResponse)
async def get_outputs(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    env = await outputs_env(sctx, ctx)
    return clean({"outputs": [_public_entry(e) for e in env["entries"]],
                  "tabs": [{"id": t["id"], "availability": t["availability"],
                            "missing": [{k: m.get(k) for k in ("output", "produced_by", "action")} for m in t["missing"]]}
                           for t in env["tabs"]]})


@router.get("/runs/{rid}/outputs/batch", response_model=OutputsBatch)
def get_outputs_batch(rid: str, ids: str = Query(...), sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    results, missing = {}, []
    for oid in [i for i in ids.split(",") if i]:
        try:
            results[oid] = outmod.output_content(ctx, oid)
        except ApiError as exc:
            if exc.code not in ("output_missing", "not_found"):
                raise
            d = exc.detail or {}
            missing.append({"id": oid, "produced_by": d.get("produced_by"),
                            "action": exc.action if isinstance(exc.action, dict) or exc.action is None else None})
    return clean({"results": results, "missing": missing})


@router.get("/runs/{rid}/outputs/{oid}")
def get_output(rid: str, oid: str, sctx: StudioContext = Depends(get_ctx)) -> dict:
    ctx = run_context(sctx, rid)
    return outmod.output_content(ctx, oid)


def study_index(sctx: StudioContext, ctx) -> list[dict]:
    """The studies of the run's project (or targeting the run), newest first, each with whether it is attached
    to this run: ``{id, kind, status, out_dir, summary, target_run_id, attached}`` (``attached`` is None when
    no ``study_links`` row exists).  Read through the DB contract of SPEC §10.2 (the studies item owns the
    rows)."""
    try:
        rows = sctx.db.fetchall(
            "SELECT s.id, s.kind, s.status, s.out_dir, s.summary_json, s.target_run_id, s.updated_utc, "
            "l.attached AS attached FROM studies s "
            "LEFT JOIN study_links l ON l.study_id = s.id AND l.run_id = ? "
            "WHERE s.target_run_id = ? OR l.run_id IS NOT NULL OR (s.project_id IS NOT NULL AND s.project_id = ?) "
            "ORDER BY s.updated_utc DESC, s.created_utc DESC", (ctx.run_id, ctx.run_id, ctx.project_id))
    except Exception:
        rows = []
    out = []
    for r in rows:
        try:
            summ = json.loads(r.get("summary_json") or "null")
        except ValueError:
            summ = None
        out.append({"id": r["id"], "kind": r["kind"], "status": r.get("status"), "out_dir": r.get("out_dir"),
                    "summary": summ if isinstance(summ, dict) else None, "target_run_id": r.get("target_run_id"),
                    "attached": None if r.get("attached") is None else bool(r["attached"])})
    return out


def _run_study_rows(index: list[dict], rid: str) -> list[dict]:
    """The study of each kind that belongs to this run (targets it or is attached), newest first."""
    out, seen = [], set()
    for r in index:
        if r["kind"] in seen or not (r["target_run_id"] == rid or r["attached"]):
            continue
        seen.add(r["kind"])
        out.append(r)
    return out


def _findings(sctx: StudioContext, rid: str) -> list[dict]:
    try:
        rows = sctx.db.fetchall("SELECT id, title, note_md, view, url_state, created_utc FROM findings WHERE run_id = ? "
                                "ORDER BY position, created_utc", (rid,))
    except Exception:
        rows = []
    return [{k: r.get(k) or "" for k in ("id", "title", "note_md", "view", "url_state", "created_utc")} for r in rows]


def _headline(sctx: StudioContext, ctx) -> str | None:
    if not ctx.project_id:
        return None
    p = sctx.db.fetchone("SELECT meta_json FROM projects WHERE id = ?", (ctx.project_id,))
    try:
        meta = json.loads((p or {}).get("meta_json") or "{}") or {}
    except ValueError:
        meta = {}
    return meta.get("headline_scenario")


@router.get("/runs/{rid}/views/{view}", response_model=ViewModel)
async def get_view(rid: str, view: str, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.views import VIEWS, build_view

    if view not in VIEWS:
        raise ApiError("unknown_view", f"unknown view {view!r}", detail={"views": list(VIEWS)})
    ctx = run_context(sctx, rid)
    oenv = await outputs_env(sctx, ctx)
    rows = sb.stage_rows(ctx, oenv["proj"])
    env = {"outputs": oenv["entries"], "tabs": oenv["tabs"], "stage_rows": rows,
           "labels": sb.STAGE_LABELS, "demo": bool(ctx.row.get("demo")), "headline_scenario": _headline(sctx, ctx)}
    if view in ("overview", "uncertainty"):
        env["studies_index"] = study_index(sctx, ctx)
    if view == "overview":
        env["study_rows"] = _run_study_rows(env["studies_index"], rid)
        env["findings"] = _findings(sctx, rid)
        env["run_flags"] = _run_flags(ctx)

    def build():
        from threadpoolctl import threadpool_limits

        with threadpool_limits(1):
            if view == "overview":
                env["board"] = sb.board_row(ctx, sctx.db, rows, resumable=sb.can_resume(ctx, oenv["active"]))
            return build_view(ctx, view, env)

    return await asyncio.to_thread(build)


# ---------------------------------------------------------------------------
# docs, files, dictionary
# ---------------------------------------------------------------------------

@router.get("/runs/{rid}/docs", response_model=list[DocEntry])
def get_docs(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return outmod.docs_list(run_context(sctx, rid), sctx.db)


@router.get("/runs/{rid}/docs/{doc}", response_model=DocContent)
def get_doc(rid: str, doc: str, sctx: StudioContext = Depends(get_ctx)):
    return outmod.doc_content(run_context(sctx, rid), sctx.db, doc)


@router.get("/runs/{rid}/files", response_model=list[FileEntry])
def get_files(rid: str, path: str | None = None, sctx: StudioContext = Depends(get_ctx)):
    return outmod.list_files(run_context(sctx, rid), path)


@router.get("/runs/{rid}/files/raw")
def get_file_raw(rid: str, path: str, request: Request, as_: str | None = Query(None, alias="as"),
                 sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    if as_:
        if as_ not in ("csv", "json", "html", "geojson"):
            raise ApiError("no_conversion", f"unknown conversion {as_!r}")
        body, media, name = outmod.convert_file(ctx, path, as_)
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"',
                                                         "Cache-Control": "no-store"})
    p = outmod._safe(ctx, path)
    if not p.is_file():
        raise ApiError("not_found", f"no file {path!r} in this run")
    return FileResponse(p, filename=p.name, headers={"Cache-Control": "no-store"})


@router.get("/runs/{rid}/files/table", response_model=FileTable)
def get_file_table(rid: str, path: str, limit: int = Query(200, ge=1, le=5000), columns: str | None = None,
                   sctx: StudioContext = Depends(get_ctx)):
    cols = [c for c in columns.split(",") if c] if columns else None
    return outmod.file_table(run_context(sctx, rid), path, limit, cols)


@router.get("/runs/{rid}/dictionary", response_model=list[DictionaryRow])
def get_dictionary(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return outmod.dictionary(run_context(sctx, rid))
