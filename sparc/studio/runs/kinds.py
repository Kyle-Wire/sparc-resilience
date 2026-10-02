"""Job kinds of the runs item (SPEC §10.3, api.md §8): ``run.core`` (launch and resume) and the read-only
``run.external`` pseudo-job of a live CLI run (SPEC §5.14).

``run.core`` runs in the worker process: it builds the config from the run's launch snapshot
(``<studio_dir>/launch.json``: ``core_config_from_dict(config_raw, base_dir=config_dir)``) and calls
``run_core(cfg, run_dir=…, run_meta=…, **args)``.  Resume reuses the snapshot byte for byte; only
``threads`` may differ (job params).  The server-side hooks keep the run row current: ``on_event``
(``run.start``, ``stage.end``, ``checkpoint``, ``run.end``) and ``on_finish`` re-derive it through the
registry.  ``retry`` of a ``run.core`` job is a resume when the run has a checkpoint, with the guards of a
resume (:func:`~sparc.studio.runs.launch.retry_run`); a resume that reaches the scheduler once its run is
complete fails its start-time preflight (``not_resumable``) instead of re-running the finished run.

This module is imported by the server (to list kinds) and by workers; heavy imports stay inside the job
function.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sparc.studio.jobs.kinds import job_kind

log = logging.getLogger("sparc.studio.runs")

__all__ = ["RunCoreParams", "RunExternalParams", "run_core_job", "run_external_job", "run_core_estimate"]


class RunCoreParams(BaseModel):
    """``run.core`` params (api.md §8); every other argument comes from ``launch.json``."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    resume: bool = False
    use_current_config: bool | None = None
    threads: int | None = Field(None, ge=1)


class RunExternalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str


def _registry(sctx):
    reg = getattr(sctx, "services", {}).get("registry") if sctx is not None else None
    return reg


def _refresh(sctx, job: dict) -> None:
    reg = _registry(sctx)
    if reg is not None and job.get("run_id"):
        try:
            reg.refresh(job["run_id"])
        except Exception:
            log.exception("refreshing run %s failed", job.get("run_id"))


def run_core_on_event(sctx, job: dict, event: dict) -> None:
    if event.get("type") == "run.start" or event.get("type") in ("checkpoint", "run.end", "stage.end"):
        _refresh(sctx, job)


def run_core_on_finish(sctx, job: dict, result) -> None:
    _sweep_tmp(sctx, job)
    _refresh(sctx, job)
    reader = getattr(sctx, "services", {}).get("reader") if sctx is not None else None
    if reader is not None and job.get("run_id"):
        reader.forget(job["run_id"])
    _adopt_as_active(sctx, job)


def _sweep_tmp(sctx, job: dict) -> None:
    """A worker killed while it wrote a file (Force stop or the OOM killer during a checkpoint save) leaves the
    hidden temporary - up to a whole checkpoint - in the run folder: the worker is gone now, so remove it."""
    if sctx is None or not job.get("run_id"):
        return
    from sparc.core import runio

    try:
        row = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (job["run_id"],))
        if row is not None:
            runio.remove_stale_tmp(row["run_dir"])
    except Exception:
        log.exception("sweeping the temporaries of run %s failed", job.get("run_id"))


def _adopt_as_active(sctx, job: dict) -> None:
    """A run that completes becomes its project's active run when the project has none (or its active run
    was deleted), so the Lab and the project overview open on it."""
    if sctx is None or not job.get("run_id") or not job.get("project_id"):
        return
    from sparc.studio.runs.registry import set_active_run

    try:
        run = sctx.db.fetchone("SELECT status, project_id FROM runs WHERE id = ?", (job["run_id"],))
        proj = sctx.db.fetchone("SELECT active_run_id FROM projects WHERE id = ?", (job["project_id"],))
        if run is None or proj is None or run["status"] != "complete" or run.get("project_id") != job["project_id"]:
            return
        cur = proj.get("active_run_id")
        if cur and sctx.db.fetchone("SELECT id FROM runs WHERE id = ?", (cur,)) is not None:
            return
        set_active_run(sctx.db, job["project_id"], job["run_id"])
    except Exception:
        log.exception("setting the active run of %s failed", job.get("project_id"))


def run_core_estimate(sctx, job: dict, params) -> dict | None:
    """Planned units of the run (stored at launch), minus the stages a resume restores from the checkpoint."""
    from sparc.studio import db as dbmod
    from sparc.studio.jobs.eta import checkpoint_bytes, peak_ram_gb

    run_id = job.get("run_id")
    if not run_id or sctx is None:
        return None
    row = sctx.db.fetchone("SELECT run_dir, stages_json, n_points FROM runs WHERE id = ?", (run_id,))
    if row is None:
        return None
    info = dbmod.loads(row.get("stages_json"), {}) or {}
    plan = info.get("plan") or []
    n = info.get("n_points") or row.get("n_points")
    done: set = set()
    resume = bool(params.resume if hasattr(params, "resume") else (params or {}).get("resume"))
    if resume:
        try:
            side = json.loads((Path(row["run_dir"]) / "checkpoint.json").read_text("utf-8"))
            done = set(side.get("done") or [])
        except (OSError, ValueError):
            done = set()
    units: dict[str, float] = {}
    for node in plan:
        if node.get("state", "will_run") != "will_run":
            continue
        if node.get("checkpoint_key") and node["checkpoint_key"] in done:
            continue
        for u, c in (node.get("units") or {}).items():
            units[u] = units.get(u, 0) + c
    out: dict = {"units": units, "n_cells": n}
    if n:
        k = int(info.get("n_folds") or 5)
        out["peak_ram_gb"] = round(peak_ram_gb(n, k), 2)
        out["disk_bytes"] = checkpoint_bytes(n)
    return out


def run_core_retry_params(sctx, job: dict) -> dict:
    """A retry is a resume when the run has a checkpoint (api.md §3 ``POST /jobs/{jid}/retry``)."""
    params = dict(job.get("params") or {})
    row = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (job.get("run_id"),)) if sctx else None
    if row is not None:
        rd = Path(row["run_dir"])
        params["resume"] = (rd / "checkpoint.pkl").exists() or (rd / "checkpoint.json").exists()
    params.pop("use_current_config", None)
    return params


async def run_core_retry(sctx, job: dict) -> dict:
    """``POST /jobs/{jid}/retry``: the resume guards apply (``409 active``, ``409 not_resumable``)."""
    from sparc.studio.runs.launch import retry_run

    return await retry_run(sctx, job)


def run_core_preflight(sctx, job: dict, params) -> list[dict]:
    """A resume whose run is complete by the time it would start (queued behind the job that finished it,
    or through ``POST /api/jobs``) is refused: it would re-run S0, S7 and finish over the finished run."""
    resume = params.get("resume") if isinstance(params, dict) else getattr(params, "resume", False)
    if not resume or sctx is None:
        return []
    row = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (job.get("run_id"),))
    if row is None:
        return []
    rd = Path(row["run_dir"])
    try:
        state = json.loads((rd / "run_state.json").read_text("utf-8"))
    except (OSError, ValueError):
        state = None
    done = (state.get("status") == "succeeded") if isinstance(state, dict) else (rd / "manifest.json").is_file()
    if done:
        return [{"reason": "the run is complete: nothing to resume", "fatal": True, "code": "not_resumable"}]
    return []


@job_kind("run.core", lane="heavy", executor="process", label="Pipeline run", params=RunCoreParams, needs_run=True,
          locks_run=True, long=True, network_hosts=(), estimate=run_core_estimate, on_event=run_core_on_event,
          on_event_types=("run.start", "stage.end", "checkpoint", "run.end"), on_finish=run_core_on_finish,
          retry_params=run_core_retry_params, retry=run_core_retry, preflight=run_core_preflight)
def run_core_job(ctx, params: RunCoreParams) -> dict:
    """Run (or resume) the pipeline from the run's launch snapshot (SPEC §4.3)."""
    from sparc.core import progress
    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import run_core

    launch = ctx.launch
    if not launch or launch.get("config_raw") is None:
        raise FileNotFoundError(f"run {params.run_id}: no studio/launch.json snapshot to run from")
    cfg = core_config_from_dict(launch["config_raw"], base_dir=launch.get("config_dir") or ctx.project_dir)
    args = dict(launch.get("args") or {})
    # run_core takes no thread count: the threads the scheduler gave this job (params.threads or the launch
    # args, capped by the budget; job.json) go through progress.set_threads, as the worker already did
    args.pop("threads", None)
    progress.set_threads(int(ctx.threads))
    meta = {"studio_run_id": params.run_id, "project_id": launch.get("project_id"),
            "origin": launch.get("origin") or "studio"}
    for k in ("study_id", "parent_run_id", "role"):
        if launch.get(k):
            meta[k] = launch[k]
    progress.emit("log", logger="sparc.studio.runs", level="INFO",
                  msg=f"{'resuming' if params.resume else 'starting'} run {params.run_id} "
                      f"({'fast' if args.get('fast') else 'coarse ' + str(args['coarse']) if args.get('coarse') else 'full'})")
    res = run_core(cfg, stages=args.get("stages") or ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"),
                   fast=bool(args.get("fast")), coarse=args.get("coarse"), cv_curve=args.get("cv_curve"),
                   resume=bool(params.resume), run_dir=Path(ctx.run_dir), run_meta=meta)
    manifest = getattr(res, "manifest", None) or {}
    st = (manifest.get("metrics") or {}).get("stacker") or {}
    done = []
    try:
        done = list(json.loads((Path(ctx.run_dir) / "run_state.json").read_text("utf-8")).get("done") or [])
    except (OSError, ValueError):
        pass
    return {"run_id": params.run_id, "status": "succeeded", "timings_s": manifest.get("timings_s") or {},
            "metrics": {"r2": st.get("r2"), "rmse": st.get("rmse"), "coverage": st.get("interval_coverage")}
            if st else None, "done": done}


def run_external_on_finish(sctx, job: dict, result) -> None:
    _refresh(sctx, job)


@job_kind("run.external", lane="none", executor="external", label="CLI run (watched)", params=RunExternalParams,
          needs_run=True, on_finish=run_external_on_finish)
def run_external_job(ctx, params: RunExternalParams) -> dict:
    """Never executed: ``run.external`` rows are created by the registry and only tailed (SPEC §5.14)."""
    raise RuntimeError("run.external pseudo-jobs are tailed, never run")
