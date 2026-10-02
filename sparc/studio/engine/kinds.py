"""Job kinds of the Scenario Lab (SPEC §10.3, api.md §8).

=======================================  ===========  ========  =============================================
Kind                                     Lane         Executor  Does
=======================================  ===========  ========  =============================================
``engine.open``                          engine       engine    load the run into the host (progress: unpickle
                                                                MB, mediators, the baseline pass)
``engine.scenario``                      engine       engine    one exact scenario → ``results/<id>/``
``engine.batch``                         engine       engine    several scenarios, one tick each
``engine.rerun_configured``              engine       engine    a configured S5 scenario with folds
``engine.sweep``                         engine       engine    one exact run per dose, fitted curve
``engine.plan_verify`` / ``…_frontier``  engine       engine    a plan's closed loop / closed-loop Pareto
``scenario.across_runs``                 heavy        process   one engine at a time in the job process
``export.decision_pack`` / ``plan_pack``  medium      process   the packs (:mod:`sparc.studio.scenarios.packs`)
/ ``compare_pack``
=======================================  ===========  ========  =============================================

Engine kinds never run in a worker: :class:`~sparc.studio.engine.executor.EngineExecutor` calls
:func:`prepare_request` in the server (compile, masks, result ids) and sends the request to the host.  The
``on_finish`` hooks (server side, fast, idempotent) index the result directories the host wrote, move the
scenario statuses, record sweeps and verified plans, publish ``scenario.result`` and ``engine.status``, and
complete the ``exports`` row of a pack.

Heavy libraries are imported inside functions: the server imports this module to list kinds.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sparc.studio import db as dbmod
from sparc.studio.engine import executor as _executor  # noqa: F401 - registers the "engine" executor
from sparc.studio.errors import ApiError
from sparc.studio.jobs.kinds import job_kind
from sparc.studio.workspace import new_id, read_json

log = logging.getLogger("sparc.studio.engine")

__all__ = ["prepare_request", "EngineOpenParams", "EngineScenarioParams", "EngineBatchParams",
           "RerunConfiguredParams", "EngineSweepParams", "PlanVerifyParams", "AcrossRunsParams",
           "DecisionPackParams", "PlanPackParams", "ComparePackParams", "across_runs_estimate", "run_row_payload",
           "index_job_results"]


class EngineOpenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str


class EngineScenarioParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    scenario_id: str
    revision: int | None = None


class EngineBatchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    scenario_ids: list[str] = Field(min_length=1)


class RerunConfiguredParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    slug: str


class EngineSweepParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    sweep_id: str


class PlanVerifyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str


class AcrossRunsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    run_ids: list[str] = Field(min_length=1)


class DecisionPackParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    result_id: str
    thresholds: list[float] | None = None


class PlanPackParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    plan_id: str


class ComparePackParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    comparison_id: str


# ---------------------------------------------------------------------------
# server-side preparation of engine requests
# ---------------------------------------------------------------------------

def _host_only(ctx, params) -> dict:
    raise RuntimeError("engine jobs run in the engine host (executor 'engine'), not in a worker")


def _services(sctx):
    reader = sctx.services.get("reader")
    if reader is None:
        raise ApiError("not_ready", "the runs registry is still starting", status=503)
    return reader


def run_row_payload(sctx, run_id: str) -> tuple[dict, dict]:
    """``(run_row, common payload)``: what every engine request carries about its run."""
    from sparc.studio.engine.service import get_service

    row = sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise ApiError("not_found", f"no run {run_id!r}")
    svc = get_service(sctx)
    sdir = svc.studio_dir(row)
    if not svc.trusted(run_id):
        raise ApiError("untrusted_pickle", "this run was imported without trusting its checkpoint",
                       action=svc.trust_action(row))
    rec = read_json(sdir / "import.json") or {}
    proj = sctx.db.fetchone("SELECT demo FROM projects WHERE id = ?", (row.get("project_id"),)) \
        if row.get("project_id") else None
    run_row = {k: row.get(k) for k in ("id", "run_dir", "studio_dir", "project_id", "status", "origin", "label")}
    return run_row, {"run_row": run_row, "trusted": True, "studio_dir": str(sdir),
                     "config_path": rec.get("config_path"),
                     "demo": bool(row.get("demo") or (proj or {}).get("demo"))}


def _project_dir(sctx, project_id: str | None) -> Path | None:
    if not project_id:
        return None
    p = sctx.db.fetchone("SELECT dir FROM projects WHERE id = ?", (project_id,))
    return Path(p["dir"]) if p and p.get("dir") else None


def _new_result(studio_dir: str) -> dict:
    from sparc.studio.engine.store import result_dir

    rid = new_id("result")
    return {"id": rid, "dir": str(result_dir(studio_dir, rid))}


def _specification(sctx, scenario_id: str) -> list[float] | None:
    """City means of the latest successful "check across runs" of the scenario (its specification band)."""
    row = sctx.db.fetchone("SELECT result_json FROM jobs WHERE kind = 'scenario.across_runs' AND scenario_id = ? "
                           "AND status = 'succeeded' ORDER BY finished_utc DESC", (scenario_id,))
    res = dbmod.loads((row or {}).get("result_json"), {}) or {}
    vals = [r["city"]["estimate"] for r in res.get("rows") or [] if r.get("ok") and r.get("city")]
    return [float(min(vals)), float(max(vals))] if len(vals) >= 2 else None


def compiled_item(sctx, ctx, srow: dict, common: dict, *, threads: int = 1) -> dict:
    """The host payload of one stored scenario on the run of ``ctx`` (``422`` on blocking warnings)."""
    from threadpoolctl import threadpool_limits

    from sparc.studio.engine.compile import compile_scenario
    from sparc.studio.engine.preview import compute_delta, emulator_status, load_emulator

    reader = _services(sctx)
    doc = dbmod.loads(srow.get("doc_json"), {}) or {}
    with threadpool_limits(1):
        comp = compile_scenario(ctx, doc, db=sctx.db, reader=reader, project_dir=_project_dir(sctx, ctx.project_id),
                                cost_model=getattr(sctx.jobs, "cost_model", None), threads=threads)
    if comp.blocking:
        raise ApiError("validation", "; ".join(w["message"] for w in comp.blocking),
                       detail={"warnings": comp.blocking, "errors": [
                           {"path": f"edits.{w.get('edit_index')}", "message": w["message"], "code": w["code"]}
                           for w in comp.blocking]})
    preview_delta = None
    try:
        em = load_emulator(ctx)
        if em is not None and emulator_status(ctx, comp, em)["usable"]:
            with threadpool_limits(1):
                preview_delta = compute_delta(em, comp.dx(), comp.n)
    except Exception as exc:                    # the emulator comparison is optional
        log.info("preview-vs-exact skipped: %s", exc)
    return {**common, "result": _new_result(common["studio_dir"]),
            "scenario": {"id": srow["id"], "revision": int(srow.get("revision") or 1), "name": doc.get("name")},
            "name": doc.get("name") or srow["id"], "content_hash": comp.content_hash, "options": comp.options,
            "interventions": comp.interventions(), "regions": comp.regions, "per_unit": comp.costs,
            "requested": comp.requested, "lever_cells": comp.lever_cells,
            "warnings": [w for w in comp.warnings if not w.get("blocking")], "compiled": comp.lever_summary(),
            "preview_delta": preview_delta, "specification": _specification(sctx, srow["id"]), "kind": "exact"}


def prepare_request(sctx, job: dict) -> tuple[str, dict]:
    """``(op, payload)`` of an engine job (runs in a server thread, see the executor)."""
    params = job.get("params") if isinstance(job.get("params"), dict) else dbmod.loads(job.get("params_json"), {})
    params = params or {}
    kind = job["kind"]
    if kind == "engine.open":
        _row, common = run_row_payload(sctx, params["run_id"])
        return "open", common
    if kind == "engine.scenario":
        from sparc.studio.scenarios import library

        _row, common = run_row_payload(sctx, params["run_id"])
        ctx = _services(sctx).get(params["run_id"])
        srow = library.get_row(sctx.db, params["scenario_id"])
        return "scenario", compiled_item(sctx, ctx, srow, common, threads=int(job.get("threads") or 1))
    if kind == "engine.batch":
        from sparc.studio.scenarios import library

        _row, common = run_row_payload(sctx, params["run_id"])
        ctx = _services(sctx).get(params["run_id"])
        items, failed = [], []
        for sid in params["scenario_ids"]:
            try:
                items.append(compiled_item(sctx, ctx, library.get_row(sctx.db, sid), common))
            except ApiError as exc:
                failed.append({"scenario_id": sid, "error": exc.message})
        return "batch", {**common, "items": items, "failed": failed}
    if kind == "engine.rerun_configured":
        _row, common = run_row_payload(sctx, params["run_id"])
        ctx = _services(sctx).get(params["run_id"])
        match = next((s for s in ctx.configured_scenarios() if s["slug"] == params["slug"]), None)
        if match is None:
            raise ApiError("not_found", f"no configured scenario {params['slug']!r} on this run")
        return "rerun_configured", {**common, "result": _new_result(common["studio_dir"]), "name": match["name"],
                                    "slug": params["slug"], "scenario": None}
    if kind == "engine.sweep":
        from sparc.studio.runs.selection import RunSource, resolve
        from sparc.studio.scenarios import sweeps

        _row, common = run_row_payload(sctx, params["run_id"])
        ctx = _services(sctx).get(params["run_id"])
        srow = sweeps.sweep_row(sctx.db, params["sweep_id"])
        p = dbmod.loads(srow.get("params_json"), {}) or {}
        where = resolve(RunSource(ctx, sctx.db), p["selection"]) if p.get("selection") else None
        sign = -1.0 if p.get("direction") == "decrease" else 1.0
        return "sweep", {**common, "sweep_id": srow["id"], "sweep_dir": srow["dir"], "lever": p["lever"],
                         "sign": sign, "where": where,
                         "items": [{"dose": float(d), "result": _new_result(common["studio_dir"])}
                                   for d in p.get("doses") or []]}
    if kind in ("engine.plan_verify", "engine.plan_frontier"):
        from sparc.studio.scenarios import plans

        prow = plans.plan_row(sctx.db, params["plan_id"])
        _row, common = run_row_payload(sctx, prow["run_id"])
        pp = read_json(Path(prow["dir"]) / "params.json") or {}
        sign = -1.0 if pp.get("direction") == "decrease" else 1.0
        lever = (pp.get("params") or {}).get("lever")
        base = {**common, "plan_id": prow["id"], "plan_dir": prow["dir"], "lever": lever, "sign": sign,
                "name": f"Plan: {prow.get('name') or prow['id']}"}
        if kind == "engine.plan_verify":
            dose = np.asarray(np.load(Path(prow["dir"]) / "dose.npy", allow_pickle=False), dtype=np.float64)
            return "plan_verify", {**base, "dose": dose, "result": _new_result(common["studio_dir"])}
        ctx = _services(sctx).get(prow["run_id"])
        pts = plans.frontier_points(ctx, prow, db=sctx.db, project_dir=_project_dir(sctx, ctx.project_id))
        return "plan_frontier", {**base, "points": pts}
    raise ValueError(f"{kind} is not an engine job")


# ---------------------------------------------------------------------------
# estimates and hooks
# ---------------------------------------------------------------------------

def _n_cells(sctx, run_id: str | None) -> int | None:
    if not run_id:
        return None
    r = sctx.db.fetchone("SELECT n_points FROM runs WHERE id = ?", (run_id,))
    return int(r["n_points"]) if r and r.get("n_points") else None


def _engine_loaded(sctx, run_id: str | None) -> bool:
    from sparc.studio.engine.service import get_service

    try:
        return bool(run_id) and get_service(sctx).loaded(run_id) is not None
    except Exception:
        return False


def open_estimate(sctx, job: dict, params) -> dict | None:
    from sparc.studio.engine.service import estimate_rss_gb

    rid = job.get("run_id")
    row = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (rid,)) if rid else None
    if row is None:
        return None
    try:
        nbytes = (Path(row["run_dir"]) / "checkpoint.pkl").stat().st_size
    except OSError:
        return None
    return {"units": {"unpickle": 1, "mediator_fit": 1, "engine_pass": 1}, "n_cells": _n_cells(sctx, rid),
            "peak_ram_gb": round(estimate_rss_gb(nbytes), 2)}


def _passes_estimate(n_passes):
    def est(sctx, job: dict, params) -> dict | None:
        rid = job.get("run_id")
        p = params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        k = n_passes(sctx, p) if callable(n_passes) else n_passes
        units = {"engine_pass": max(1, int(k))}
        if not _engine_loaded(sctx, rid):
            units.update(unpickle=1, mediator_fit=1, engine_pass=units["engine_pass"] + 1)
        return {"units": units, "n_cells": _n_cells(sctx, rid)}
    return est


def _sweep_passes(sctx, p: dict) -> int:
    row = sctx.db.fetchone("SELECT params_json FROM sweeps WHERE id = ?", (p.get("sweep_id"),))
    return len((dbmod.loads((row or {}).get("params_json"), {}) or {}).get("doses") or [1])


def _open_preflight(sctx, job: dict, params) -> list[dict]:
    from sparc.studio.engine.service import get_service

    rid = job.get("run_id")
    try:
        get_service(sctx).preflight(rid, memory=False)
    except ApiError as exc:
        fatal = exc.code in ("no_checkpoint", "untrusted_pickle")
        return [{"reason": exc.message, "fatal": fatal, "code": exc.code,
                 "actions": [exc.action] if exc.action else []}]
    return []


def _job_params(job: dict) -> dict:
    p = job.get("params")
    if not isinstance(p, dict):
        p = dbmod.loads(job.get("params_json"), {}) or {}
    return p


def index_job_results(sctx, job: dict, result: dict | None) -> list[dict]:
    """Index what a succeeded engine job wrote (idempotent): its result directories, the statuses and mirrors
    of their scenarios, a sweep's curve and a plan's verified result.

    The executor calls this before the job turns ``succeeded`` (so a client that sees the job end already finds
    the results and the new scenario status); ``on_finish`` calls it again and publishes ``scenario.result``.
    ``job`` is a ``jobs`` row or a ``Job``."""
    from sparc.studio.engine import store
    from sparc.studio.scenarios import library

    rows = []
    synced: set[str] = set()
    for rid in _result_ids(result):
        row = sctx.db.fetchone("SELECT * FROM results WHERE id = ?", (rid,))
        if row is None:
            for d in _result_dirs(sctx, job, rid):
                row = store.insert_result_row(sctx.db, d)
                if row is not None:
                    break
        if row is None:
            continue
        rows.append(row)
        sid = row.get("scenario_id")
        if sid and sid not in synced:
            synced.add(sid)
            library.sync_status(sctx.db, sid)
            library.write_mirror(sctx.db, sctx.workspace, sid)
    params = _job_params(job)
    if job.get("kind") == "engine.sweep" and params.get("sweep_id"):
        srow = sctx.db.fetchone("SELECT dir FROM sweeps WHERE id = ?", (params["sweep_id"],))
        if srow is not None:
            curve = read_json(Path(srow["dir"]) / "curve.json")
            sctx.db.update("sweeps", {"id": params["sweep_id"]}, {"summary_json": dbmod.dumps(curve),
                                                                  "job_id": job["id"]})
    if job.get("kind") == "engine.plan_verify" and params.get("plan_id") and (result or {}).get("result_id"):
        sctx.db.update("plans", {"id": params["plan_id"]}, {"verified_result_id": result["result_id"]})
    return rows


def _publish_results(sctx, job: dict, result: dict | None) -> None:
    for row in index_job_results(sctx, job, result):
        sctx.hub.publish("scenario.result", {"scenario_id": row.get("scenario_id"), "result_id": row["id"],
                                             "run_id": row["run_id"], "kind": row["kind"]})


def _result_dirs(sctx, job: dict, rid: str) -> list[Path]:
    from sparc.studio.engine.store import result_dir

    run_id = job.get("run_id")
    row = sctx.db.fetchone("SELECT studio_dir, run_dir FROM runs WHERE id = ?", (run_id,)) if run_id else None
    if row is None:
        return []
    sd = row.get("studio_dir") or str(Path(row["run_dir"]) / "studio")
    return [result_dir(sd, rid)]


def _result_ids(result: dict | None) -> list[str]:
    r = result or {}
    ids = list(r.get("results") or [])
    for k in ("result_id",):
        if r.get(k) and r[k] not in ids:
            ids.append(r[k])
    for k in ("result_ids", "points"):
        for v in r.get(k) or []:
            if v not in ids:
                ids.append(v)
    return ids


def engine_on_finish(sctx, job: dict, result) -> None:
    """Index the results the host wrote and tell the clients (idempotent)."""
    from sparc.studio.engine.service import get_service

    svc = get_service(sctx)
    svc.invalidate()
    if job["kind"] == "engine.open":
        st = "ready" if job["status"] == "succeeded" else "incompatible" if (job.get("error") or {}).get(
            "type") == "Incompatible" else "error"
        svc.publish(job.get("run_id"), st, rss_mb=(result or {}).get("rss_mb"))
        return
    if job["status"] == "succeeded":
        _publish_results(sctx, job, result)
    elif job["kind"] == "engine.sweep" and _job_params(job).get("sweep_id"):
        sctx.db.update("sweeps", {"id": _job_params(job)["sweep_id"]}, {"job_id": job["id"]})
    # a failed scenario leaves the engine loaded: publish the run's actual engine state, not "error"
    rid = job.get("run_id")
    if rid:
        try:
            st = svc.run_status(rid)
            svc.publish(rid, st["state"], progress=st.get("progress"), rss_mb=st.get("rss_mb"))
        except Exception:                       # the run went away meanwhile
            log.debug("engine status of %s unavailable", rid, exc_info=True)


_ENGINE = dict(lane="engine", executor="engine", needs_run=True, needs_checkpoint=True, long=False,
               on_finish=engine_on_finish)


@job_kind("engine.open", label="Open engine", params=EngineOpenParams, estimate=open_estimate,
          preflight=_open_preflight, **_ENGINE)
def engine_open_job(ctx, params: EngineOpenParams) -> dict:
    return _host_only(ctx, params)


@job_kind("engine.scenario", label="Exact scenario", params=EngineScenarioParams, estimate=_passes_estimate(1),
          preflight=_open_preflight, **_ENGINE)
def engine_scenario_job(ctx, params: EngineScenarioParams) -> dict:
    return _host_only(ctx, params)


@job_kind("engine.batch", label="Exact scenarios (batch)", params=EngineBatchParams,
          estimate=_passes_estimate(lambda s, p: len(p.get("scenario_ids") or [1])), preflight=_open_preflight,
          **_ENGINE)
def engine_batch_job(ctx, params: EngineBatchParams) -> dict:
    return _host_only(ctx, params)


@job_kind("engine.rerun_configured", label="Re-run configured scenario", params=RerunConfiguredParams,
          estimate=_passes_estimate(1), preflight=_open_preflight, **_ENGINE)
def engine_rerun_job(ctx, params: RerunConfiguredParams) -> dict:
    return _host_only(ctx, params)


@job_kind("engine.sweep", label="Dose sweep", params=EngineSweepParams, estimate=_passes_estimate(_sweep_passes),
          preflight=_open_preflight, **_ENGINE)
def engine_sweep_job(ctx, params: EngineSweepParams) -> dict:
    return _host_only(ctx, params)


@job_kind("engine.plan_verify", label="Verify plan", params=PlanVerifyParams, estimate=_passes_estimate(1),
          preflight=_open_preflight, **_ENGINE)
def engine_plan_verify_job(ctx, params: PlanVerifyParams) -> dict:
    return _host_only(ctx, params)


@job_kind("engine.plan_frontier", label="Verify plan frontier", params=PlanVerifyParams,
          estimate=_passes_estimate(4), preflight=_open_preflight, **_ENGINE)
def engine_plan_frontier_job(ctx, params: PlanVerifyParams) -> dict:
    return _host_only(ctx, params)


# ---------------------------------------------------------------------------
# check across runs (heavy, in the job process)
# ---------------------------------------------------------------------------

def _trusted_row(db, workspace_root: Path, row: dict) -> bool:
    rd = Path(row["run_dir"]).resolve()
    root = Path(workspace_root).resolve()
    if rd == root or root in rd.parents:
        return True
    sd = Path(row["studio_dir"]) if row.get("studio_dir") else rd / "studio"
    return bool((read_json(sd / "import.json") or {}).get("trust_pickles"))


def _run_check(db, workspace_root: Path, run_id: str, doc: dict) -> tuple[dict | None, str | None]:
    """``(run row, reason it cannot be used)``."""
    from sparc.studio.runs.selection import is_portable

    row = db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        return None, "no such run"
    if not (Path(row["run_dir"]) / "checkpoint.pkl").is_file():
        return row, "no checkpoint (the run did not reach S3)"
    if not _trusted_row(db, workspace_root, row):
        return row, "imported without trusting its checkpoint"
    for e in doc.get("edits") or []:
        if e.get("mode") == "per_cell" and not str(e.get("per_cell_ref") or "").startswith("csv:"):
            return row, "the scenario uses run-specific per-cell edits (plans or brushes)"
        if e.get("where") and not is_portable(e["where"], db):
            return row, "the scenario uses run-specific selections"
    return row, None


def across_runs_estimate(sctx, scenario_id: str, run_ids: list[str]) -> dict:
    """``POST /api/scenarios/{sid}/across-runs/estimate``: load + exact seconds and RSS per run."""
    from sparc.studio.engine.compile import estimate_exact_s, load_seconds
    from sparc.studio.engine.service import estimate_rss_gb
    from sparc.studio.scenarios import library

    srow = library.get_row(sctx.db, scenario_id)
    doc = dbmod.loads(srow.get("doc_json"), {}) or {}
    cm = getattr(sctx.jobs, "cost_model", None)
    threads = int(sctx.settings().threads_heavy)
    rows, total, peak = [], 0.0, 0.0
    for rid in run_ids:
        row, reason = _run_check(sctx.db, sctx.workspace.root, rid, doc)
        n = int((row or {}).get("n_points") or 0) or 54701
        nbytes = 0
        if row is not None:
            try:
                nbytes = (Path(row["run_dir"]) / "checkpoint.pkl").stat().st_size
            except OSError:
                nbytes = 0
        load = load_seconds(n, cost_model=cm, threads=threads, checkpoint_bytes=nbytes or None)
        exact = estimate_exact_s(n, cost_model=cm, threads=threads)
        rss = estimate_rss_gb(nbytes) if nbytes else 0.0
        ok = reason is None
        rows.append({"run_id": rid, "ok": ok, "reason": reason, "load_s": round(load, 1), "exact_s": round(exact, 1),
                     "rss_gb": round(rss, 2)})
        if ok:
            total += load + exact
            peak = max(peak, rss)
    return {"runs": rows, "total_s": round(total, 1), "peak_rss_gb": round(peak, 2)}


def _across_estimate(sctx, job: dict, params) -> dict | None:
    p = params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
    try:
        e = across_runs_estimate(sctx, p["scenario_id"], p["run_ids"])
    except ApiError:
        return None
    return {"est_s": e["total_s"], "est_lo": 0.75 * e["total_s"], "est_hi": 1.25 * e["total_s"],
            "peak_ram_gb": e["peak_rss_gb"]}


def _open_for_check(row: dict, threads: int):
    """``open_run`` as the engine host does it: the import-time config as the fallback and the cached baseline
    pass when it still matches the checkpoint and code."""
    from sparc.core.session import checkpoint_key, config_for_run, open_run
    from sparc.studio.engine.host import Host

    sd = Path(row["studio_dir"]) if row.get("studio_dir") else Path(row["run_dir"]) / "studio"
    fb = (read_json(sd / "import.json") or {}).get("config_path")
    cfg = config_for_run(row["run_dir"], fallback=fb if fb and Path(fb).is_file() else None, studio_dir=sd)
    base, _meta = Host._base_fold(str(sd), checkpoint_key(row["run_dir"]))
    return open_run(row["run_dir"], cfg, threads=threads, base_fold=base, studio_dir=sd)


@job_kind("scenario.across_runs", lane="heavy", executor="process", label="Check across runs",
          params=AcrossRunsParams, long=True, estimate=_across_estimate)
def across_runs_job(ctx, params: AcrossRunsParams) -> dict:
    """Evaluate a portable scenario on several runs, one engine at a time in this process (SPEC §7.12)."""
    from sparc.core import progress
    from sparc.studio.engine import ops
    from sparc.studio.engine.compile import compile_scenario
    from sparc.studio.runs.common import likely
    from sparc.studio.runs.reader import RunContext

    db = ctx.db
    srow = db.fetchone("SELECT * FROM scenarios WHERE id = ?", (params.scenario_id,))
    if srow is None:
        raise LookupError(f"no scenario {params.scenario_id!r}")
    doc = dbmod.loads(srow.get("doc_json"), {}) or {}
    rows = []
    n = len(params.run_ids)
    for k, rid in enumerate(params.run_ids, start=1):
        progress.check_cancel()
        row, reason = _run_check(db, ctx.workspace.root, rid, doc)
        out = {"run_id": rid, "city": None, "ok": False, "error": reason}
        if reason is None:
            with progress.task("variant", k=k, n=n, key=rid) as sp:
                try:
                    rctx = RunContext(row, ("across", time.time()))
                    comp = compile_scenario(rctx, doc, db=db)
                    if comp.blocking:
                        raise ValueError("; ".join(w["message"] for w in comp.blocking))
                    session = _open_for_check(row, int(ctx.threads))
                    res, _med = ops.evaluate(session, comp.interventions(), comp.options, doc.get("name") or "scenario")
                    from sparc.studio.engine import stats as S

                    unit = rctx.units.get("target", "°F")
                    out.update(ok=True, error=None, city=likely(float(np.mean(res.delta)), S.masked_se(res.delta_folds),
                                                                unit, what="the city"))
                    sp.metrics["mean_delta"] = float(np.mean(res.delta))
                    del session, res
                except progress.Cancelled:
                    raise
                except Exception as exc:          # noqa: BLE001 - one run never stops the check
                    out["error"] = f"{type(exc).__name__}: {exc}"[:500]
                finally:
                    gc.collect()
        rows.append(out)
        progress.tick(k, n, unit="run", label=rid)
    vals = [r["city"]["estimate"] for r in rows if r["ok"] and r["city"]]
    signs = [np.sign(v) for v in vals]
    majority = max(set(signs), key=signs.count) if signs else 0
    return {"rows": rows, "sign_stability": (signs.count(majority) / len(signs)) if signs else None,
            "spread": (float(max(vals) - min(vals)) if vals else None)}


# ---------------------------------------------------------------------------
# packs (medium, in the job process)
# ---------------------------------------------------------------------------

def pack_on_finish(sctx, job: dict, result) -> None:
    """Complete the ``exports`` row of a pack (``[S]`` creates it; nothing happens without it)."""
    eid = (job.get("params") or {}).get("export_id")
    if not eid:
        return
    row = sctx.db.fetchone("SELECT options_json FROM exports WHERE id = ?", (eid,))
    if row is None:
        return
    opts = dbmod.loads(row.get("options_json"), {}) or {}
    if job["status"] == "succeeded" and result:
        opts["draft"] = bool(result.get("draft"))
        sctx.db.update("exports", {"id": eid}, {"status": "ready", "path": result.get("path"),
                                                 "bytes": result.get("bytes"), "options_json": dbmod.dumps(opts)})
    else:
        sctx.db.update("exports", {"id": eid}, {"status": "failed"})


def _pack_estimate(seconds: float):
    def est(sctx, job, params):
        return {"est_s": seconds, "est_lo": 0.5 * seconds, "est_hi": 2.0 * seconds}
    return est


@job_kind("export.decision_pack", lane="medium", executor="process", label="Decision pack", params=DecisionPackParams,
          on_finish=pack_on_finish, estimate=_pack_estimate(10.0))
def decision_pack_job(ctx, params: DecisionPackParams) -> dict:
    from sparc.studio.scenarios.packs import decision_pack

    return decision_pack(ctx, params.export_id, params.result_id, params.thresholds)


@job_kind("export.plan_pack", lane="medium", executor="process", label="Plan pack", params=PlanPackParams,
          on_finish=pack_on_finish, estimate=_pack_estimate(15.0))
def plan_pack_job(ctx, params: PlanPackParams) -> dict:
    from sparc.studio.scenarios.packs import plan_pack

    return plan_pack(ctx, params.export_id, params.plan_id)


@job_kind("export.compare_pack", lane="medium", executor="process", label="Compare pack", params=ComparePackParams,
          on_finish=pack_on_finish, estimate=_pack_estimate(10.0))
def compare_pack_job(ctx, params: ComparePackParams) -> dict:
    from sparc.studio.scenarios.packs import compare_pack

    return compare_pack(ctx, params.export_id, params.comparison_id)
