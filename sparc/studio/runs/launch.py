"""Plan, preflight, launch, resume and rerun of pipeline runs (SPEC §4.3, §5.4, §5.9, §5.13; api.md §6).

**Plan.**  ``core.plan_stages`` on the project's config with the mode arguments (one S0 load gives the
point count, extent and cell size), estimates per node from the foundation cost model
(``jobs.eta.estimate_nodes``), peak RAM (55 kB × n × K^0.5 + 0.8 GB) and checkpoint disk (9.6 kB × n),
plus preflight rows: memory, disk, input files, network hosts and threads.

**Launch** writes ``<run_dir>/studio/launch.json`` (SPEC §4.3) - the raw config with every path made
absolute, ``climate.cache`` set to the workspace cache and ``output.dir`` to the project's ``runs/`` -
creates the run row and the ``run.core`` job, and chains the "then" post-run jobs with ``after_job_id``.

**Resume** always uses ``launch.json``; only ``threads`` may differ.  ``use_current_config`` renames it
to ``launch.<n>.json`` and snapshots the current project config (the deliberate refit, shown with its
impact first).  **Rerun** creates a new run with the same mode and args.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from sparc.studio import __version__
from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.runs.common import fnum, read_json_cached
from sparc.studio.runs.reader import load_config_raw
from sparc.studio.workspace import new_run_id, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.runs")

__all__ = ["ALL_STAGES", "THEN_KINDS", "project_row", "absolutise", "mode_args", "mode_tag", "compute_plan",
           "launch_run", "resume_run", "retry_run", "rerun_run", "checkpoint_info", "impact"]

ALL_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
THEN_KINDS = ("post.planner", "post.emulator", "post.uncertainty", "post.writeup", "post.baselines")
CMIP6_HOSTS = ("cmip6-pds.s3.amazonaws.com",)
_PATH_KEYS = (("data", "path"), ("physics", "forcing"), ("climate", "table"), ("planner", "layers"))


def project_row(db, pid: str) -> dict:
    row = db.fetchone("SELECT * FROM projects WHERE id = ?", (pid,))
    if row is None:
        raise ApiError("not_found", f"no project {pid!r}")
    return row


def _abs(p, base: Path):
    """A relative path string resolved against ``base`` (``CoreConfig.resolve_path``: no ``~`` expansion);
    absolute paths and non-strings unchanged."""
    if not isinstance(p, str) or not p.strip() or Path(p).is_absolute():
        return p
    return str((Path(base) / p).resolve())


def absolutise(raw: dict, config_dir: Path, *, cache_dir: Path, runs_dir: Path) -> dict:
    """The launch snapshot's ``config_raw`` (SPEC §4.3): path keys absolute, cache and output dir set.

    The same rule as the project config service's ``launch_raw`` (``data.path``, ``data.join[].path``,
    ``physics.forcing``, ``climate.table``, ``planner.layers`` resolved against the project folder;
    ``climate.cache`` = the workspace cache; ``output.dir`` = the project's ``runs``), so the config impact
    preview compares Studio-launched runs in the form they were launched with."""
    out = copy.deepcopy(raw.get("core", raw) if isinstance(raw, dict) else {})
    for sec, key in _PATH_KEYS:
        s = out.get(sec)
        if isinstance(s, dict) and key in s:
            s[key] = _abs(s[key], config_dir)
    data = out.get("data")
    if isinstance(data, dict):
        for j in data.get("join") or []:
            if isinstance(j, dict) and "path" in j:
                j["path"] = _abs(j["path"], config_dir)
    out.setdefault("climate", {})
    if isinstance(out["climate"], dict):
        out["climate"]["cache"] = str(Path(cache_dir).resolve())
    out.setdefault("output", {})
    if isinstance(out["output"], dict):
        out["output"]["dir"] = str(Path(runs_dir).resolve())
    return out


def mode_args(mode: str, coarse_m: float | None = None, stages=None, cv_curve: bool | None = None,
              threads: int | None = None) -> dict:
    if mode not in ("fast", "coarse", "full"):
        raise ApiError("validation", f"mode must be fast, coarse or full, not {mode!r}",
                       detail={"errors": [{"path": "mode", "message": "invalid mode", "code": "mode"}]})
    st = [s for s in ALL_STAGES if s in set(stages)] if stages else list(ALL_STAGES)
    bad = [s for s in (stages or []) if s not in ALL_STAGES]
    if bad:
        raise ApiError("validation", f"unknown stages {bad}",
                       detail={"errors": [{"path": "stages", "message": f"unknown {bad}", "code": "stages"}]})
    return {"stages": st, "fast": mode == "fast",
            "coarse": float(coarse_m or 60.0) if mode == "coarse" else None,
            "cv_curve": cv_curve, "threads": threads}


def mode_tag(args: dict) -> str:
    if args.get("coarse"):
        return f"coarse{int(round(float(args['coarse'])))}"
    if args.get("fast"):
        return "fast"
    if set(args.get("stages") or ALL_STAGES) != set(ALL_STAGES):
        return "custom"
    return "full"


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _issue(level, path, code, message) -> dict:
    return {"level": level, "path": path, "code": code, "message": message}


def _load_cfg(raw: dict, base_dir) -> tuple[Any, list[dict]]:
    from sparc.studio.runs.reader import core_cfg

    try:
        return core_cfg(raw, base_dir), []
    except Exception as exc:                 # invalid config: surfaced as issues, never a 500
        return None, [_issue("error", "", "config_invalid", f"{type(exc).__name__}: {exc}")]


def _file_issues(cfg) -> list[dict]:
    out = []
    try:
        if not Path(cfg.data_path).is_file():
            out.append(_issue("error", "data.path", "file_missing", f"data file not found: {cfg.data_path}"))
    except ValueError as exc:
        out.append(_issue("error", "data.path", "file_missing", str(exc)))
    for j in cfg.data.get("join") or []:
        p = cfg.resolve_path(j.get("path"))
        if p is not None and not Path(p).is_file():
            out.append(_issue("error", "data.join", "file_missing", f"join table not found: {p}"))
    raw = cfg.raw
    cl = raw.get("climate") or {}
    if cl.get("enabled") and cl.get("source", "table") == "table":
        p = cfg.resolve_path(cl.get("table")) if cl.get("table") else None
        if p is None or not Path(p).is_file():
            out.append(_issue("warn", "climate.table", "file_missing",
                              "the climate stage needs climate.table (fetch CMIP6 change factors first)"))
    lay = (raw.get("planner") or {}).get("layers")
    if lay and not Path(cfg.resolve_path(lay)).is_file():
        out.append(_issue("warn", "planner.layers", "file_missing",
                          f"planner layers not found: {cfg.resolve_path(lay)} (people objective and planner pack need them)"))
    return out


def compute_plan(sctx, cfg_raw: dict, config_dir, args: dict, *, done=frozenset(), resumable=None,
                 for_launch: bool = False) -> dict:
    """``RunPlan`` (api.md §6) of running ``cfg_raw`` with ``args``; ``done`` = a checkpoint done set."""
    import numpy as np

    from sparc.core.data import load_core_data
    from sparc.core.pipeline import apply_mode_overrides, plan_stages
    from sparc.studio.jobs.eta import CostModel, checkpoint_bytes, estimate_nodes, peak_ram_gb

    cfg, issues = _load_cfg(cfg_raw, config_dir)
    if cfg is None:
        raise ApiError("validation", "the project config has errors", detail={"issues": issues, "errors": [
            {"path": i["path"], "message": i["message"], "code": i["code"]} for i in issues]})
    issues += _file_issues(cfg)
    errors = [i for i in issues if i["level"] == "error"]
    if errors:
        raise ApiError("validation", "the project config has errors", detail={"issues": issues, "errors": [
            {"path": i["path"], "message": i["message"], "code": i["code"]} for i in errors]})
    eff = apply_mode_overrides(copy.deepcopy(cfg), fast=bool(args.get("fast")), coarse=args.get("coarse"),
                               cv_curve=args.get("cv_curve"))
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(1):
            data = load_core_data(eff)
    except Exception as exc:
        raise ApiError("validation", f"the data cannot be loaded: {exc}",
                       detail={"issues": issues + [_issue("error", "data", "data_unreadable", str(exc))],
                               "errors": [{"path": "data", "message": str(exc), "code": "data_unreadable"}]})
    n = int(data.n)
    dx = float(data.grid.dx)
    extent = float(min(np.ptp(data.x), np.ptp(data.y_coord)))
    nodes = plan_stages(cfg, stages=args.get("stages") or ALL_STAGES, fast=bool(args.get("fast")),
                        coarse=args.get("coarse"), cv_curve=args.get("cv_curve"), done=frozenset(done),
                        n_points=n, extent=extent, dx=dx)
    settings = sctx.settings() if sctx is not None else None
    threads = int(args.get("threads") or (settings.threads_heavy if settings else 1))
    model = getattr(getattr(sctx, "jobs", None), "cost_model", None) or CostModel()
    nodes = estimate_nodes(nodes, n, threads, model)
    total = sum(fnum(nd.get("est_s"), 0.0) for nd in nodes)
    lo = sum(fnum(nd.get("est_lo"), 0.0) for nd in nodes)
    hi = sum(fnum(nd.get("est_hi"), 0.0) for nd in nodes)
    k = int(eff.raw["cv"]["n_folds"])
    ram = peak_ram_gb(n, k)
    ckpt = checkpoint_bytes(n)
    disk_gb = ckpt * 1.15 / 1e9
    cl = eff.raw.get("climate") or {}
    hosts = list(CMIP6_HOSTS) if cl.get("enabled") and cl.get("source") == "cmip6" else []
    pre = preflight_rows(sctx, eff, ram=ram, disk_bytes=ckpt * 1.15, hosts=hosts, threads=threads, issues=issues)
    return {"nodes": nodes, "total_est_s": round(total, 3), "est_lo": round(lo, 3), "est_hi": round(hi, 3),
            "est_peak_rss_gb": round(ram, 3), "est_disk_gb": round(disk_gb, 4), "network_hosts": hosts,
            "preflight": pre, "issues": issues, "resumable": resumable, "threads": threads,
            "_n_points": n, "_n_folds": k}


def preflight_rows(sctx, cfg, *, ram: float, disk_bytes: float, hosts: list[str], threads: int,
                   issues: list[dict]) -> list[dict]:
    """Launch preflight (SPEC §5.13, §10.4): memory, disk, files, network hosts, threads."""
    from sparc.studio.jobs.resources import memory_available_gb, preflight_disk, preflight_memory

    rows = []
    jobs = getattr(sctx, "jobs", None)
    others = 0.0
    if jobs is not None:
        others = sum(float((jobs.samples.get(j) or {}).get("rss_mb") or 0) for j in jobs.live_pids())
    mem = preflight_memory(ram, others)
    avail = memory_available_gb(others)
    act = None
    if mem and jobs is not None:
        live = jobs._rows_by_status(("starting", "running", "cancelling"))
        if live:
            act = {"kind": "open", "label": f"Stop {live[0].get('label') or live[0]['kind']}", "method": "POST",
                   "path": f"/api/jobs/{live[0]['id']}/cancel"}
    rows.append({"check": "memory", "ok": mem is None, "severity": "error" if mem else "info",
                 "message": mem["reason"] if mem else f"needs about {ram:.1f} GB at peak; {avail:.1f} GB available",
                 "action": act})
    ws_root = sctx.workspace.root if sctx is not None else Path.cwd()
    disk = preflight_disk(ws_root, disk_bytes)
    rows.append({"check": "disk", "ok": disk is None, "severity": "error" if disk else "info",
                 "message": disk["reason"] if disk else f"checkpoint and outputs need about {disk_bytes / 1e9:.2f} GB",
                 "action": None})
    files = [i for i in issues if i["code"] == "file_missing"]
    rows.append({"check": "files", "ok": not files, "severity": "warn" if files else "info",
                 "message": "; ".join(i["message"] for i in files) if files else "every input file is present",
                 "action": None})
    if hosts:
        settings = sctx.settings() if sctx is not None else None
        if settings is not None and settings.offline:
            rows.append({"check": "network", "ok": False, "severity": "error",
                         "message": "offline mode is on (Settings) and this run fetches CMIP6 data", "action": None})
        elif jobs is not None:
            res = jobs.netcheck(hosts, timeout=3.0)
            bad = [r for r in res if not r["ok"]]
            rows.append({"check": "network", "ok": not bad, "severity": "warn" if bad else "info",
                         "message": ("cannot reach " + ", ".join(r["host"] for r in bad)) if bad else
                         "network hosts reachable: " + ", ".join(hosts), "action": None})
    settings = sctx.settings() if sctx is not None else None
    if settings is not None:
        over = threads > settings.threads_heavy
        rows.append({"check": "threads", "ok": not over, "severity": "warn" if over else "info",
                     "message": (f"{threads} threads exceed the heavy-job budget of {settings.threads_heavy}" if over
                                 else f"{threads} threads (heavy-job budget {settings.threads_heavy})"),
                     "action": None})
    if jobs is not None:
        busy = [r for r in jobs._rows_by_status(("starting", "running", "cancelling", "queued", "blocked"))
                if r["lane"] == "heavy"]
        if busy:
            rows.append({"check": "queue", "ok": True, "severity": "info",
                         "message": f"queued behind {len(busy)} heavy job(s): {busy[0].get('label') or busy[0]['kind']}",
                         "action": None})
    return rows


# ---------------------------------------------------------------------------
# checkpoint card
# ---------------------------------------------------------------------------

def checkpoint_info(ctx, db, *, active: bool) -> dict:
    """``CheckpointInfo`` (api.md §6): never unpickles (``core.checkpoint_status``)."""
    from sparc.studio.runs.statusboard import recorded_seconds, reused_stages

    side = ctx.checkpoint_json or {}
    pkl = ctx.run_dir / "checkpoint.pkl"
    present = pkl.exists()
    st: dict = {}
    cfg = None
    try:
        if ctx.launch and ctx.launch.get("config_raw") is not None:
            from sparc.studio.runs.reader import core_cfg

            cfg = core_cfg(ctx.launch["config_raw"], ctx.launch.get("config_dir") or ctx.run_dir)
        else:
            cfg = ctx.cfg
        from sparc.core.pipeline import checkpoint_status

        st = checkpoint_status(ctx.run_dir, cfg) if present else {}
    except Exception as exc:
        log.warning("%s: checkpoint status unavailable: %s", ctx.run_id, exc)
        st = {}
    done = list(side.get("done") or st.get("done") or [])
    matches = st.get("matches") if present else None
    fp_ok = st.get("fingerprint_match")
    reason = None
    resumable = True
    if not ctx.launch:
        resumable, reason = False, "no launch snapshot (studio/launch.json): use Rerun"
    elif active:
        resumable, reason = False, "a job is running on this run"
    elif ctx.status == "complete":
        resumable, reason = False, "the run is complete"
    reuses = reused_stages(done) if present and fp_ok is not False else []
    if present and fp_ok is False:
        ch = ", ".join(st.get("changed_sections") or []) or "code"
        reason = reason or f"the checkpoint does not match the launch snapshot ({ch} changed): a resume refits"
    saves = None
    if reuses:
        secs = [recorded_seconds(db, ctx, s) for s in reuses]
        known = [s for s in secs if s is not None]
        saves = round(sum(known), 1) if known else None
    return {"present": present, "bytes": int(pkl.stat().st_size) if present else side.get("bytes"),
            "done": done, "saved_utc": side.get("saved_utc") or st.get("saved_utc"),
            "fingerprint": side.get("fingerprint") or st.get("fingerprint"),
            "matches_snapshot": matches if present else None,
            "changed_sections": list(st.get("changed_sections") or []), "resumable": resumable, "reuses": reuses,
            "saves_s": saves, "reason": reason}


#: first stage a changed fingerprint section invalidates (S0 always runs; the config service's mapping)
_SECTION_STAGE = (("data", "S1"), ("code", "S1"), ("core", "S1"), ("s4", "S4"), ("s5", "S5"),
                  ("climate", "climate"), ("s6", "S6"), ("s7", "S7"))


def impact(ctx, project: dict, workspace=None) -> dict:
    """What a resume with the current project config would refit: the fingerprint sections of the current
    config - in its launch-snapshot form, as a Studio launch would write it - that differ from the run's
    ``checkpoint.json`` sections (the pickle is never opened)."""
    import pandas as pd

    from sparc.core.pipeline import fingerprint_sections
    from sparc.studio.runs.reader import core_cfg

    raw, cdir = _project_raw(project)
    cache = Path(workspace.cache_dir) if workspace is not None else cdir / "cache"
    snap = absolutise(raw, cdir, cache_dir=cache, runs_dir=Path(project["dir"]) / "runs")
    cfg = core_cfg(snap, cdir)
    frame = None
    if ctx.exists("input_frame.parquet"):
        try:
            frame = pd.read_parquet(ctx.path("input_frame.parquet"))
        except Exception:
            frame = None
    a = ctx.args
    cur = fingerprint_sections(cfg, bool(a.get("fast")), frame, coarse=a.get("coarse"), cv_curve=a.get("cv_curve"))
    old = (ctx.checkpoint_json or {}).get("sections") or {}
    if not old:
        return {"changed_sections": list(cur), "refit_from": "S1",
                "phrase": "the run has no checkpoint: a resume with the current config runs every stage"}
    changed = [k for k in cur if old.get(k) != cur.get(k)]
    first = next((stage for sec, stage in _SECTION_STAGE if sec in changed), None)
    phrase = (f"a resume with the current config would refit from {first}" if first else
              "the current config matches the checkpoint: nothing would be refitted")
    return {"changed_sections": changed, "refit_from": first, "phrase": phrase}


# ---------------------------------------------------------------------------
# launch, resume, rerun
# ---------------------------------------------------------------------------

def _project_raw(project: dict) -> tuple[dict, Path]:
    path = Path(project["config_path"])
    if not path.is_file():
        raise ApiError("not_found", f"the project config {path} is missing")
    return load_config_raw(path), path.resolve().parent


def _check_then(then: list[str] | None) -> list[str]:
    from sparc.studio.jobs.kinds import get_kind

    out = []
    for k in then or []:
        if k not in THEN_KINDS:
            raise ApiError("validation", f"{k} cannot follow a run",
                           detail={"errors": [{"path": "then", "message": f"unknown {k}", "code": "then"}]})
        if get_kind(k) is None:
            raise ApiError("validation", f"{k} is not available in this build",
                           detail={"errors": [{"path": "then", "message": f"{k} unavailable", "code": "unavailable"}]})
        out.append(k)
    return out


async def launch_run(sctx, pid: str, body: dict) -> dict:
    """``POST /api/projects/{pid}/runs`` → ``{run, job, chain}``."""
    project = project_row(sctx.db, pid)
    then = _check_then(body.get("then"))
    raw, cdir = _project_raw(project)
    args = mode_args(body.get("mode", "fast"), body.get("coarse_m"), body.get("stages"), body.get("cv_curve"),
                     body.get("threads"))
    plan = await asyncio.to_thread(compute_plan, sctx, raw, cdir, args)
    bad = [p for p in plan["preflight"] if not p["ok"] and p["severity"] == "error"]
    if bad:
        raise ApiError("preflight_failed", bad[0]["message"], detail={"preflight": plan["preflight"]},
                       action=bad[0].get("action"))
    return await _create_run(sctx, project, raw, cdir, args, plan, label=body.get("label"), notes=body.get("notes"),
                             then=then)


async def _create_run(sctx, project: dict, raw: dict, cdir: Path, args: dict, plan: dict, *, label=None,
                      notes=None, then=()) -> dict:
    reg = sctx.services["registry"]
    pdir = Path(project["dir"])
    run_id = new_run_id(mode_tag(args))
    run_dir = pdir / "runs" / run_id
    studio = run_dir / "studio"
    studio.mkdir(parents=True, exist_ok=False)
    launch = {"studio_version": __version__, "run_id": run_id, "project_id": project["id"],
              "config_raw": absolutise(raw, cdir, cache_dir=sctx.workspace.cache_dir, runs_dir=pdir / "runs"),
              "config_dir": str(cdir), "args": {k: args.get(k) for k in ("stages", "fast", "coarse", "cv_curve",
                                                                         "threads")},
              "created_utc": utc_now(), "job_id": None, "origin": "studio"}
    write_json_atomic(studio / "launch.json", launch)
    compact = [{"id": n["id"], "state": n["state"], "units": n.get("units") or {},
                "checkpoint_key": n.get("checkpoint_key")} for n in plan["nodes"]]
    info = {"requested": args["stages"], "plan": compact, "n_points": plan.get("_n_points"),
            "n_folds": plan.get("_n_folds"), "est_s": plan.get("total_est_s")}
    row = {"id": run_id, "project_id": project["id"], "run_dir": str(run_dir.resolve()),
           "studio_dir": str(studio.resolve()), "origin": "studio", "label": label or None, "notes": notes,
           "status": "queued", "created_utc": launch["created_utc"], "stages_json": dbmod.dumps(info),
           "mode": "coarse" if args.get("coarse") else "fast" if args.get("fast") else mode_tag(args),
           "coarse_m": args.get("coarse"), "demo": int(bool(project.get("demo"))), "n_points": plan.get("_n_points")}
    await sctx.db.ainsert("runs", row)
    try:
        job = await sctx.jobs.submit("run.core", {"run_id": run_id, "resume": False, "threads": args.get("threads")},
                                     run_id=run_id, project_id=project["id"],
                                     label=f"Run {label or run_id} ({mode_tag(args)})")
    except Exception:
        await sctx.db.aexecute("DELETE FROM runs WHERE id = ?", (run_id,))
        shutil.rmtree(run_dir, ignore_errors=True)   # only launch.json: a rescan must not find a run never queued
        raise
    launch["job_id"] = job["id"]
    write_json_atomic(studio / "launch.json", launch)
    await sctx.db.aupdate("runs", {"id": run_id}, {"last_job_id": job["id"]})
    chain = []
    prev = job["id"]
    for kind in then:
        j = await sctx.jobs.submit(kind, {}, run_id=run_id, project_id=project["id"], after_job_id=prev)
        chain.append(j)
        prev = j["id"]
    new = await asyncio.to_thread(reg.refresh, run_id, publish=False)
    sctx.hub.publish("run.indexed", {"run_id": run_id, "project_id": project["id"], "origin": "studio"})
    return {"run": reg.summary(new or row), "job": sctx.jobs.get(job["id"]), "chain": chain}


def _run_row_without_jobs(sctx, rid: str) -> dict:
    """The run's row; ``409 active`` while any job is queued or running on it."""
    row = sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (rid,))
    if row is None:
        raise ApiError("not_found", f"no run {rid!r}")
    active = sctx.db.fetchone("SELECT id FROM jobs WHERE run_id = ? AND status IN "
                              "('queued','blocked','starting','running','cancelling')", (rid,))
    if active is not None:
        raise ApiError("active", "a job is already running on this run", detail={"job_id": active["id"]})
    return row


async def resume_run(sctx, rid: str, body: dict) -> dict:
    """``POST /api/runs/{rid}/resume`` → ``Job``.

    Resume replays ``launch.json`` (only ``threads`` may differ).  ``use_current_config`` first renames it to
    ``launch.<n>.json`` and snapshots the project's current config with its impact (SPEC §4.3); a refused
    resume carries ``detail.checkpoint`` and, with ``use_current_config``, ``detail.impact``.  The checks and
    the queueing hold the run's submit lock, so of two resumes at once one is queued and the other is
    ``409 active``."""
    async with sctx.jobs.run_submit_lock(rid):
        return await _resume_locked(sctx, rid, body)


async def _resume_locked(sctx, rid: str, body: dict) -> dict:
    reg = sctx.services["registry"]
    reader = sctx.services["reader"]
    row = _run_row_without_jobs(sctx, rid)
    ctx = reader.for_row(row)
    info = await asyncio.to_thread(checkpoint_info, ctx, sctx.db, active=False)
    detail: dict[str, Any] = {"checkpoint": info}
    current = bool(body.get("use_current_config"))
    project = imp = None
    if current:
        if not row.get("project_id"):
            raise ApiError("not_resumable", "the run belongs to no project, so there is no current config",
                           detail=detail)
        project = project_row(sctx.db, row["project_id"])
        try:
            imp = await asyncio.to_thread(impact, ctx, project, sctx.workspace)
        except ApiError:
            raise
        except Exception as exc:              # the current config does not load: say so, never a 500
            raise ApiError("not_resumable", f"the current project config cannot be used: {exc}",
                           detail=detail) from exc
        detail["impact"] = imp
    if not ctx.launch:
        raise ApiError("not_resumable", "this run has no launch snapshot (it was not launched by Studio): rerun it",
                       detail=detail)
    if row["status"] == "complete":
        raise ApiError("not_resumable", "the run is complete: nothing to resume", detail=detail)
    params: dict[str, Any] = {"run_id": rid, "resume": True}
    threads = body.get("threads") or (ctx.launch.get("args") or {}).get("threads")
    if threads:                                  # the snapshot's threads unless the resume overrides them
        params["threads"] = int(threads)
    studio = Path(row["studio_dir"])
    previous = None
    if current:
        raw, cdir = _project_raw(project)
        n = 1
        while (studio / f"launch.{n}.json").exists():
            n += 1
        old = read_json_cached(studio / "launch.json") or {}
        previous = studio / f"launch.{n}.json"
        os.replace(studio / "launch.json", previous)
        new_launch = {**old, "config_raw": absolutise(raw, cdir, cache_dir=sctx.workspace.cache_dir,
                                                      runs_dir=Path(project["dir"]) / "runs"),
                      "config_dir": str(cdir), "created_utc": utc_now(), "previous": previous.name,
                      "impact": imp}
        write_json_atomic(studio / "launch.json", new_launch)
        params["use_current_config"] = True
    try:
        job = await sctx.jobs.submit("run.core", params, run_id=rid, project_id=row.get("project_id"),
                                     parent_job_id=row.get("last_job_id"),
                                     label=f"Resume {row.get('label') or rid}")
    except Exception:
        if previous is not None:              # the resume never started: the run keeps its snapshot
            os.replace(previous, studio / "launch.json")
        raise
    await sctx.db.aupdate("runs", {"id": rid}, {"last_job_id": job["id"]})
    await asyncio.to_thread(reg.refresh, rid)
    return job


async def retry_run(sctx, job: dict) -> dict:
    """``POST /api/jobs/{jid}/retry`` of a finished ``run.core`` job → ``Job`` (the kind's ``retry`` hook).

    The retry runs the run again from its launch snapshot - a resume when it has a checkpoint
    (:func:`~sparc.studio.runs.kinds.run_core_retry_params`) - under the guards of :func:`resume_run`:
    ``409 active`` while a job is queued or running on the run, ``409 not_resumable`` for a run that is
    complete or has no launch snapshot.  The new job becomes the run's last job."""
    from sparc.studio.runs.kinds import run_core_retry_params

    rid = job.get("run_id")
    if not rid:
        raise ApiError("not_found", "the job names no run")
    async with sctx.jobs.run_submit_lock(rid):
        row = _run_row_without_jobs(sctx, rid)
        ctx = sctx.services["reader"].for_row(row)
        if not ctx.launch:
            raise ApiError("not_resumable", "this run has no launch snapshot (it was not launched by Studio): "
                           "rerun it", detail={"run_id": rid})
        if row["status"] == "complete":
            raise ApiError("not_resumable", "the run is complete: nothing to retry (rerun it for a new run)",
                           detail={"run_id": rid})
        params = await asyncio.to_thread(run_core_retry_params, sctx, job)
        new = await sctx.jobs.submit("run.core", params, project_id=job.get("project_id") or row.get("project_id"),
                                     run_id=rid, study_id=job.get("study_id"), priority=int(job.get("priority") or 0),
                                     parent_job_id=job["id"], label=job.get("label"))
        await sctx.db.aupdate("runs", {"id": rid}, {"last_job_id": new["id"]})
    await asyncio.to_thread(sctx.services["registry"].refresh, rid)
    return new


async def rerun_run(sctx, rid: str, body: dict) -> dict:
    """``POST /api/runs/{rid}/rerun`` → ``{run, job}`` (a new run with the same mode and args)."""
    row = sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (rid,))
    if row is None:
        raise ApiError("not_found", f"no run {rid!r}")
    if not row.get("project_id"):
        raise ApiError("validation", "only runs of a project can be rerun",
                       detail={"errors": [{"path": "run", "message": "no project", "code": "no_project"}]})
    project = project_row(sctx.db, row["project_id"])
    ctx = sctx.services["reader"].for_row(row)
    a = ctx.args
    args = {"stages": a.get("stages") or list(ALL_STAGES), "fast": bool(a.get("fast")), "coarse": a.get("coarse"),
            "cv_curve": a.get("cv_curve"), "threads": a.get("threads")}
    if body.get("use_current_config", True) or not ctx.launch:
        raw, cdir = _project_raw(project)
    else:
        raw = ctx.launch["config_raw"]
        cdir = Path(ctx.launch.get("config_dir") or project["dir"])
    plan = await asyncio.to_thread(compute_plan, sctx, raw, cdir, args)
    bad = [p for p in plan["preflight"] if not p["ok"] and p["severity"] == "error"]
    if bad:
        raise ApiError("preflight_failed", bad[0]["message"], detail={"preflight": plan["preflight"]},
                       action=bad[0].get("action"))
    label = body.get("label") or (f"{row['label']} (rerun)" if row.get("label") else None)
    out = await _create_run(sctx, project, raw, cdir, args, plan, label=label)
    return {"run": out["run"], "job": out["job"]}
