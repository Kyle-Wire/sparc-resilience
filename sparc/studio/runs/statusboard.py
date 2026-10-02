"""Stage rows of a run, the run timeline and the Pipeline Status Board (SPEC §5.12, api.md §6).

Stage states come from the latest ``run.core`` (or ``run.external``) job's tracker projection when one
exists (``source: "events"``), else from the manifest (``timings_s``, ``stages_run``) or ``run_state.json``
(``done`` set, current stage).  So imported CLI runs get a board row too.

Board columns: the eleven stages minus ``finish`` | ``planner, emulator, uncertainty, writeup`` |
``placebo, multiverse, simcheck, reproduce``.  Post and study cells come from the run's files, the
``jobs`` table and the ``studies`` / ``study_links`` tables (DB contract with the studies item).
"""

from __future__ import annotations

from sparc.studio import db as dbmod
from sparc.studio.runs.common import fnum
from sparc.studio.workspace import parse_utc

__all__ = ["STAGES", "STAGE_LABELS", "COLUMNS", "stage_rows", "board_row", "timeline", "CHECKPOINT_STAGES",
           "reused_stages", "plan_reasons", "relaunch_action", "can_resume"]

STAGES = ("S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish")
STAGE_LABELS = {"S0": "Data and QA", "S1": "Area of influence", "S2_S3": "Base models and stacker",
                "baselines": "Reference baselines", "cv_curve": "Skill vs distance", "S4": "Response surfaces",
                "S5": "Scenarios", "climate": "Climate and adaptation", "S6": "Causal validation",
                "S7": "Budget optimisation", "finish": "Manifest and report"}
#: stage → checkpoint done-set entry restoring it (sparc.core.pipeline.CHECKPOINT_KEY)
CHECKPOINT_STAGES = {"S1": "S3", "S2_S3": "S3", "baselines": "baselines", "cv_curve": "cv_curve", "S4": "S4",
                     "S5": "S5", "climate": "climate", "S6": "S6"}
COLUMNS = ([{"id": s, "label": STAGE_LABELS[s], "group": "stage"} for s in STAGES if s != "finish"]
           + [{"id": k, "label": lab, "group": "post"} for k, lab in (("planner", "Planner pack"),
                                                                     ("emulator", "Emulator"),
                                                                     ("uncertainty", "Uncertainty"),
                                                                     ("writeup", "Methods & model card"))]
           + [{"id": k, "label": lab, "group": "study"} for k, lab in (("placebo", "Placebo"),
                                                                      ("multiverse", "Multiverse"),
                                                                      ("simcheck", "Simulation check"),
                                                                      ("reproduce", "Reproduction"))])
_STALE_OUTPUTS = {"S6": "causal.json", "S7": "optimize.json", "cv_curve": "cv_distance.json",
                  "baselines": "baselines.json", "climate": "climate.json"}
_STALE_KEYS = {"S6": "causal", "S7": "optimize", "cv_curve": "cv_distance", "baselines": "baselines",
               "climate": "climate"}


#: "Re-run with …" wording per stage
_RERUN_WITH = {"S0": "the data stage", "S1": "the influence stage", "S2_S3": "the models", "baselines": "the baselines",
               "cv_curve": "the CV curve", "S4": "the response surfaces", "S5": "the scenarios",
               "climate": "the climate stage", "S6": "the causal audit", "S7": "the budget optimisation",
               "finish": "the run report"}


def relaunch_action(ctx, stage: str) -> dict | None:
    """The remedy for a stage a finished run did not compute: the project's Launch page, prefilled from this
    run (``?from=<run_id>``), where the stage can be enabled before a new run starts.  A resume could not
    add it: resume replays the run's own launch snapshot."""
    if not ctx.project_id:
        return None
    from urllib.parse import quote

    return {"kind": "open", "label": f"Re-run with {_RERUN_WITH.get(stage, stage)}", "method": "GET",
            "path": f"/p/{quote(ctx.project_id)}/launch?from={quote(ctx.run_id)}"}


def _ckpt_map() -> dict:
    try:
        from sparc.core.pipeline import CHECKPOINT_KEY

        return dict(CHECKPOINT_KEY)
    except Exception:
        return dict(CHECKPOINT_STAGES)


def reused_stages(done) -> list[str]:
    done = set(done or ())
    return [s for s, k in _ckpt_map().items() if k in done]


def can_resume(ctx, active: bool) -> bool:
    """Whether ``POST /runs/{rid}/resume`` would accept the run: Studio launched it (``launch.json``), no job is
    active on it and it is not complete (SPEC §5.8: partial, failed, cancelled and interrupted runs resume)."""
    return bool(ctx.launch) and not active and ctx.status not in ("complete", "queued", "running", "external_live")


def plan_reasons(ctx) -> dict[str, str | None]:
    """``{stage: skip reason}`` of the run's own plan (``core.plan_stages`` on its config and mode arguments,
    with the geometry of its grid, so no data is loaded); ``{}`` when the config is unusable."""
    def build() -> dict:
        cfg = ctx.cfg
        if cfg is None:
            return {}
        try:
            import numpy as np

            from sparc.core.pipeline import ALL_STAGES, plan_stages

            a = ctx.args
            g = ctx.grid
            geo = {}
            if g is not None and g.n:
                geo = {"n_points": g.n, "dx": float(g.dx),
                       "extent": float(min(np.ptp(g.x), np.ptp(g.y)))}
            nodes = plan_stages(cfg, stages=a.get("stages") or ALL_STAGES, fast=bool(a.get("fast")),
                                coarse=a.get("coarse"), cv_curve=a.get("cv_curve"), **geo)
        except Exception:                    # an older config the planner cannot read: no reasons
            return {}
        return {n["id"]: n.get("reason") for n in nodes if n.get("state") == "skipped"}

    return ctx._get("plan_reasons", build)


def stage_rows(ctx, proj: dict | None) -> list[dict]:
    """``[{id, state, seconds, source, reason, started_ts, ended_ts, progress}]`` for the eleven stages."""
    if proj and proj.get("stages"):
        out = []
        for sid in STAGES:
            st = proj["stages"].get(sid) or {}
            out.append({"id": sid, "state": st.get("state") or "planned", "seconds": fnum(st.get("elapsed_s")),
                        "source": "events", "reason": st.get("reason"), "started_ts": st.get("started_ts"),
                        "ended_ts": st.get("ended_ts"), "progress": fnum(st.get("progress"))})
        return out
    m = ctx.manifest_raw or {}
    if m:
        timings = dict(m.get("timings_s") or {})
        detail = ((m.get("timings_detail") or {}).get("stages") or {})
        ran = set(m.get("stages_run") or []) if isinstance(m.get("stages_run"), list) else None
        out = []
        for sid in STAGES:
            secs = fnum(detail.get(sid, timings.get(sid)))
            if sid == "finish":
                state, reason = "done", None
            elif secs is not None or (ran is not None and sid in ran) or (sid == "climate" and m.get("climate")):
                state, reason = "done", None
            elif ran is not None:
                state, reason = "skipped", plan_reasons(ctx).get(sid) or "not run"
            elif _section_for(m, sid):
                state, reason = "done", None
            else:
                state, reason = "skipped", plan_reasons(ctx).get(sid) or "not in this run"
            out.append({"id": sid, "state": state, "seconds": secs, "source": "manifest", "reason": reason,
                        "started_ts": None, "ended_ts": None, "progress": None})
        return out
    rs = ctx.run_state or {}
    done = set(rs.get("done") or [])
    cached = set(reused_stages(done))
    cur = rs.get("stage")
    status = rs.get("status")
    out = []
    seen_cur = False
    for sid in STAGES:
        if sid == cur:
            seen_cur = True
            state = "running" if status == "running" else ("failed" if status == "failed" else
                                                           "cancelled" if status == "cancelled" else "done")
        elif sid in cached or (not seen_cur and cur and sid not in ("finish",)):
            state = "done"
        else:
            state = "planned" if status == "running" else "not_reached"
        out.append({"id": sid, "state": state, "seconds": None, "source": "run_state", "reason": None,
                    "started_ts": None, "ended_ts": None, "progress": None})
    return out


def _section_for(m: dict, sid: str) -> bool:
    keys = {"S0": "qa", "S1": "influence", "S2_S3": "metrics", "baselines": "baselines", "cv_curve": "cv_distance",
            "S4": "response", "S5": "scenarios", "climate": "climate", "S6": "causal", "S7": "optimize"}
    return bool(m.get(keys.get(sid, ""), None))


def _jobs_by_kind(db, run_id: str) -> dict[str, dict]:
    rows = db.fetchall("SELECT id, kind, status, progress, started_utc, finished_utc FROM jobs WHERE run_id = ? "
                       "ORDER BY created_utc DESC, rowid DESC", (run_id,))
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["kind"], r)
    return out


def _job_seconds(j: dict) -> float | None:
    a, b = parse_utc(j.get("started_utc")), parse_utc(j.get("finished_utc"))
    return (b - a) if a and b and b >= a else None


def _job_cell(j: dict, base: dict) -> dict:
    st = j["status"]
    if st in ("queued", "blocked", "starting", "running", "cancelling"):
        # BoardCell has no queued state: a job still waiting (e.g. in a launch's "then" chain) is a running cell
        # whose reason says so
        return {**base, "state": "running", "progress": fnum(j.get("progress")), "job_id": j["id"],
                "reason": st if st in ("queued", "blocked") else None}
    if st in ("failed", "interrupted", "cancelled"):
        return {**base, "state": "failed", "reason": st, "job_id": j["id"]}
    return {**base, "state": "done", "seconds": _job_seconds(j), "job_id": j["id"]}


def board_row(ctx, db, rows: list[dict], *, resumable: bool) -> dict[str, dict]:
    """The status cells of one run (``{column id: cell}``)."""
    rid = ctx.run_id
    cells: dict[str, dict] = {}

    def cell(**kw) -> dict:
        base = {"state": "not_run", "seconds": None, "progress": None, "reason": None, "job_id": None,
                "study_id": None, "action": None}
        base.update(kw)
        return base

    for r in rows:
        sid = r["id"]
        if sid == "finish":
            continue
        st = r["state"]
        if st == "done":
            c = cell(state="done", seconds=r["seconds"])
        elif st == "cached":
            c = cell(state="cached", reason=r["reason"] or "checkpoint")
        elif st == "running":
            c = cell(state="running", progress=r.get("progress"))
        elif st in ("failed", "cancelled"):
            c = cell(state="failed", reason=st)
        elif st == "disabled" or (st == "skipped" and str(r["reason"] or "").startswith("disabled_by_config")):
            c = cell(state="disabled", reason=r["reason"])
        elif st in ("skipped", "not_requested"):
            c = cell(state="skipped", reason=r["reason"] or st)
        else:
            c = cell(state="not_run", reason=r["reason"])
            if resumable:
                c["action"] = {"kind": "resume", "label": f"Resume to compute {sid}", "method": "POST",
                               "path": f"/api/runs/{rid}/resume", "body": {}}
            elif ctx.project_id:
                c["action"] = relaunch_action(ctx, sid)
        rel = _STALE_OUTPUTS.get(sid)
        if c["state"] == "done" and rel and ctx.stale_file(rel, _STALE_KEYS.get(sid)):
            c["state"] = "stale"
            c["reason"] = f"{rel} is older than the manifest"
        cells[sid] = c
    jobs = _jobs_by_kind(db, rid)
    files = {"planner": ("planner/planner.json", "planner"), "emulator": ("emulator.json", "emulator"),
             "uncertainty": ("uncertainty.json", "uncertainty"), "writeup": ("methods.md", None)}
    for k, (rel, key) in files.items():
        j = jobs.get(f"post.{k}")
        action = {"kind": "build_emulator" if k == "emulator" else "run_job",
                  "label": {"planner": "Run planner pack", "emulator": "Build emulator",
                            "uncertainty": "Compute uncertainty", "writeup": "Regenerate methods & model card"}[k],
                  "method": "POST", "path": f"/api/runs/{rid}/actions/{k}", "body": {}}
        if j is not None and j["status"] in ("queued", "blocked", "starting", "running", "cancelling"):
            cells[k] = _job_cell(j, cell())
        elif (ctx.run_dir / rel).exists():
            c = _job_cell(j, cell()) if j is not None and j["status"] == "succeeded" else cell(state="done")
            if key and ctx.stale_file(rel, key):
                c.update(state="stale", reason=f"{rel} is older than the manifest")
            cells[k] = c
        elif j is not None:
            cells[k] = {**_job_cell(j, cell()), "action": action}
        else:
            cells[k] = cell(action=action)
    studies = _studies(db, ctx)
    fp = ctx.row.get("fingerprint")
    m = ctx.manifest_raw or {}
    for k in ("placebo", "multiverse", "simcheck", "reproduce"):
        s = studies.get(k)
        action = {"kind": "run_job", "label": f"Run the {k} study", "method": "POST",
                  "path": f"/api/runs/{rid}/studies/{k}", "body": {}}
        if s is not None:
            st = str(s.get("status") or "")
            state = ("running" if st in ("queued", "running", "starting") else
                     "failed" if st in ("failed", "interrupted", "cancelled") else "done")
            c = cell(state=state, study_id=s["id"], job_id=s.get("job_id"), reason="queued" if st == "queued" else None)
            if state == "done" and fp and s.get("run_fingerprint") and s["run_fingerprint"] != fp:
                c.update(state="stale", reason="the run was refitted after this study")
            cells[k] = c
        elif k == "reproduce":
            child = db.fetchone("SELECT id, status FROM runs WHERE parent_run_id = ? AND stages_json LIKE ? "
                                "ORDER BY created_utc DESC", (rid, '%"role":"reproduction"%'))
            if child is not None:
                cells[k] = cell(state="running" if child["status"] in ("queued", "running") else
                                "done" if child["status"] == "complete" else "failed",
                                reason=f"child run {child['id']}")
            else:
                cells[k] = cell(action=action)
        elif m.get(k):
            cells[k] = cell(state="done", reason="recorded in the manifest")
        else:
            cells[k] = cell(action=action)
    return cells


def _studies(db, ctx) -> dict[str, dict]:
    try:
        rows = db.fetchall("SELECT DISTINCT s.id, s.kind, s.status, s.job_id, s.run_fingerprint, s.updated_utc "
                           "FROM studies s LEFT JOIN study_links l ON l.study_id = s.id "
                           "WHERE s.target_run_id = ? OR (l.run_id = ? AND l.attached = 1) "
                           "ORDER BY s.updated_utc DESC", (ctx.run_id, ctx.run_id))
    except Exception:
        rows = []
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["kind"], r)
    return out


def timeline(ctx, db, proj: dict | None) -> dict:
    """``GET /runs/{rid}/timeline``."""
    rows = stage_rows(ctx, proj)
    src = "events" if proj and proj.get("stages") else "manifest"
    stages = [{"id": r["id"], "start_ts": r.get("started_ts"), "end_ts": r.get("ended_ts"), "seconds": r["seconds"],
               "state": r["state"]} for r in rows]
    jobs = []
    for j in db.fetchall("SELECT id, kind, status, started_utc, finished_utc FROM jobs WHERE run_id = ? "
                         "ORDER BY created_utc ASC, rowid ASC", (ctx.run_id,)):
        jobs.append({"job_id": j["id"], "kind": j["kind"], "start_ts": parse_utc(j.get("started_utc")),
                     "end_ts": parse_utc(j.get("finished_utc")), "status": j["status"]})
    return {"source": src, "stages": stages, "jobs": jobs}


def stage_info(row: dict) -> dict:
    info = dbmod.loads(row.get("stages_json"), {}) or {}
    return info if isinstance(info, dict) else {}


def recorded_seconds(db, ctx, stage: str) -> float | None:
    r = db.fetchone("SELECT seconds FROM stage_timings WHERE run_id = ? AND stage = ?", (ctx.run_id, stage))
    if r and r.get("seconds") is not None:
        return float(r["seconds"])
    t = ((ctx.manifest_raw or {}).get("timings_s") or {}).get(stage)
    return fnum(t)
