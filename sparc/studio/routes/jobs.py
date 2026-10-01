"""Jobs and tracking endpoints (api.md §3).

Cursors are byte offsets of event lines.  ``after=X`` always means "events
whose cursor is greater than X"; without ``after`` reading starts at the
first line.  ``next_cursor`` is the cursor of the last line examined, so it
can be passed straight back as ``after``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response

from sparc.core.progress import LEVELS
from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError
from sparc.studio.jobs import tracker
from sparc.studio.jobs.eta import CostModel, host_id, scale
from sparc.studio.jobs.manager import job_out
from sparc.studio.jobs.tailer import LOG_CAP_BYTES, read_events
from sparc.studio.schemas.common import (
    ACTIVE_STATUSES,
    EventsPage,
    Job,
    JobCreate,
    JobPatch,
    KillRequest,
    LogsPage,
    MetricPoint,
    Ok,
    Page,
    QueueState,
    ResourceRow,
    Span,
    Timings,
    TrackerSnapshot,
    WarningRow,
)
from sparc.studio.workspace import utc_iso

router = APIRouter(tags=["jobs"])

RESOURCE_WINDOW_S = 15 * 60


def _manager(sctx: StudioContext):
    if sctx.jobs is None:
        raise ApiError("not_ready", "the server is still starting", status=503)
    return sctx.jobs


def _row(sctx: StudioContext, jid: str) -> dict:
    row = sctx.db.fetchone("SELECT * FROM jobs WHERE id = ?", (jid,))
    if row is None:
        raise ApiError("not_found", f"no job {jid!r}")
    return row


def _events_path(row: dict, sctx: StudioContext | None = None) -> Path:
    if sctx is not None and sctx.jobs is not None:
        return sctx.jobs.events_path(row)
    return Path(row["job_dir"]) / "events.jsonl"


def _encode_cursor(created: str, rowid: int) -> str:
    return base64.urlsafe_b64encode(json.dumps([created, rowid]).encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, int]:
    try:
        pad = "=" * (-len(cursor) % 4)
        created, rowid = json.loads(base64.urlsafe_b64decode(cursor + pad))
        return str(created), int(rowid)
    except Exception:
        raise ApiError("validation", "bad cursor",
                       detail={"errors": [{"path": "cursor", "message": "not a cursor", "code": "bad_cursor"}]})


def _split(v: str | None) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


@router.get("/jobs", response_model=Page[Job])
async def list_jobs(status: str | None = Query(None, description="comma list, or 'active'"),
                    kind: str | None = Query(None, description="comma list"), project: str | None = None,
                    run: str | None = None, study: str | None = None, parent: str | None = None,
                    limit: int = Query(50, ge=1, le=500), cursor: str | None = None,
                    sctx: StudioContext = Depends(get_ctx)):
    """Jobs, newest first."""
    where, args = [], []
    statuses = []
    for s in _split(status):
        statuses.extend(ACTIVE_STATUSES if s == "active" else [s])
    if statuses:
        where.append(f"status IN ({','.join('?' for _ in statuses)})")
        args.extend(statuses)
    kinds = _split(kind)
    if kinds:
        where.append(f"kind IN ({','.join('?' for _ in kinds)})")
        args.extend(kinds)
    for col, val in (("project_id", project), ("run_id", run), ("study_id", study), ("parent_job_id", parent)):
        if val:
            where.append(f"{col} = ?")
            args.append(val)
    if cursor:
        c, r = _decode_cursor(cursor)
        where.append("(created_utc < ? OR (created_utc = ? AND rowid < ?))")
        args.extend([c, c, r])
    sql = "SELECT rowid AS _rowid, * FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_utc DESC, rowid DESC LIMIT ?"
    rows = sctx.db.fetchall(sql, (*args, limit + 1))
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = _encode_cursor(rows[-1]["created_utc"] or "", rows[-1]["_rowid"]) if more and rows else None
    return {"items": [job_out(r) for r in rows], "next_cursor": nxt}


@router.post("/jobs", response_model=Job, status_code=202)
async def create_job(body: JobCreate, sctx: StudioContext = Depends(get_ctx)):
    """Generic job creator: params are validated against the kind's schema.  A run lock never fails the
    request - the job waits as ``blocked``."""
    if body.kind.startswith("test.") and not sctx.test_kinds():
        raise ApiError("unknown_kind", f"unknown job kind {body.kind!r} (test kinds are disabled)")
    return await _manager(sctx).submit(body.kind, body.params, project_id=body.project_id, run_id=body.run_id,
                                       study_id=body.study_id, priority=body.priority,
                                       after_job_id=body.after_job_id)


@router.get("/jobs/{jid}", response_model=Job)
async def get_job(jid: str, sctx: StudioContext = Depends(get_ctx)):
    return job_out(_row(sctx, jid))


def _resources(sctx: StudioContext, jid: str, since: float | None = None) -> list[dict]:
    since = since if since is not None else time.time() - RESOURCE_WINDOW_S
    rows = sctx.db.fetchall("SELECT ts, rss_mb, cpu_pct, n_procs, threads FROM resource_samples WHERE job_id = ? "
                            "AND ts >= ? ORDER BY ts", (jid, since))
    live = (sctx.jobs.samples.get(jid) if sctx.jobs is not None else None)
    if live and (not rows or live["ts"] > rows[-1]["ts"]) and live["ts"] >= since:
        rows.append({k: live.get(k) for k in ("ts", "rss_mb", "cpu_pct", "n_procs", "threads")})
    return rows


@router.get("/jobs/{jid}/tracker", response_model=TrackerSnapshot)
async def tracker_snapshot(jid: str, sctx: StudioContext = Depends(get_ctx)):
    """The projection plus ``cursor`` (open the job stream with ``after=cursor``)."""
    mgr = _manager(sctx)
    row = _row(sctx, jid)
    state, eta = await mgr.projection(jid)
    parts = tracker.snapshot_parts(state, eta=eta)
    job = job_out(row)
    if row["status"] in ("starting", "running", "cancelling"):
        job["progress"] = state.get("progress")
        job.update(eta_s=eta.get("eta_s"), eta_lo=eta.get("eta_lo"), eta_hi=eta.get("eta_hi"),
                   stage=state.get("stage"), current_path=state.get("current_path"))
    try:
        capped = _events_path(row, sctx).stat().st_size > LOG_CAP_BYTES
    except OSError:
        capped = False
    return {"job": job, **parts, "resources": _resources(sctx, jid), "log_capped": capped}


@router.get("/jobs/{jid}/spans", response_model=list[Span])
async def spans(jid: str, under: str | None = None, max_depth: int = Query(4, ge=1, le=64),
                sctx: StudioContext = Depends(get_ctx)):
    _row(sctx, jid)
    state, _eta = await _manager(sctx).projection(jid)
    return tracker.spans_list(state, under=under, max_depth=max_depth, limit=50_000)


@router.get("/jobs/{jid}/events", response_model=EventsPage)
async def events(jid: str, after: int | None = Query(None, description="cursor; omit to start at the first line"),
                 limit: int = Query(1000, ge=1, le=5000), types: str | None = None,
                 min_lvl: str | None = Query(None, pattern="^(debug|info|warning|error)$"),
                 sctx: StudioContext = Depends(get_ctx)):
    """Raw event lines (envelope + fields, each with ``cursor``) - the polling fallback of the job stream."""
    row = _row(sctx, jid)
    evs, last, eof = await asyncio.to_thread(read_events, _events_path(row, sctx), after, limit=limit,
                                             types=set(_split(types)) or None, min_lvl=min_lvl)
    return {"events": evs, "next_cursor": last, "eof": eof}


def _log_lines(path: Path, after: int | None, level: str | None, logger: str | None, stage: str | None,
               q: str | None, limit: int) -> tuple[list[dict], int]:
    floor = LEVELS.get((level or "debug").lower(), 10)
    needle = (q or "").lower()
    out: list[dict] = []
    evs, last, _eof = read_events(path, after, types={"log", "warning"})
    for ev in evs:
        if ev["type"] == "log":
            lvl = str(ev.get("level") or ev.get("lvl") or "info").lower()
            name = str(ev.get("logger") or "")
            msg = str(ev.get("msg") or "")
        else:
            lvl = str(ev.get("lvl") or "warning").lower()
            name = f"warning:{ev.get('code')}"
            msg = str(ev.get("message") or "")
        if LEVELS.get(lvl, 20) < floor:
            continue
        if logger and not name.startswith(logger):
            continue
        if stage and tracker.stage_of(ev.get("path")) != stage:
            continue
        if needle and needle not in msg.lower():
            continue
        out.append({"cursor": ev["cursor"], "ts": ev.get("ts") or 0.0, "level": lvl, "logger": name, "msg": msg,
                    "path": [str(p) for p in ev.get("path") or []]})
        if len(out) >= limit:
            last = ev["cursor"]
            break
    return out, last


@router.get("/jobs/{jid}/logs", response_model=LogsPage)
async def logs(jid: str, after: int | None = None, level: str | None = None, logger: str | None = None,
               stage: str | None = None, q: str | None = None, limit: int = Query(500, ge=1, le=5000),
               sctx: StudioContext = Depends(get_ctx)):
    """``log`` and ``warning`` events, filtered."""
    row = _row(sctx, jid)
    lines, last = await asyncio.to_thread(_log_lines, _events_path(row, sctx), after, level, logger, stage, q, limit)
    return {"lines": lines, "next_cursor": last}


def _render_text(path: Path) -> str:
    out = []
    evs, _last, _eof = read_events(path, None)
    for ev in evs:
        t = ev.get("type")
        ts = utc_iso(ev.get("ts")) or "?"
        if t == "log":
            out.append(f"{ts} {str(ev.get('level') or 'INFO').upper():7s} {ev.get('logger')}: {ev.get('msg')}")
        elif t == "warning":
            out.append(f"{ts} WARNING {ev.get('code')}: {ev.get('message')}")
        elif t in ("stage.start", "stage.end", "stage.skip"):
            extra = ev.get("status") or ev.get("reason") or ""
            out.append(f"{ts} STAGE   {ev.get('stage')} {t.split('.')[1]} {extra}".rstrip())
        elif t in ("job.status", "run.end", "cancel.requested"):
            what = ev.get("status") or ev.get("by") or ""
            out.append(f"{ts} {t.upper():7s} {what}")
    return "\n".join(out) + ("\n" if out else "")


@router.get("/jobs/{jid}/logs/raw", response_class=Response,
            responses={200: {"content": {"text/plain": {}, "application/x-ndjson": {}}}})
async def logs_raw(jid: str, stream: str = Query("events", pattern="^(events|stdout|stderr|text)$"),
                   sctx: StudioContext = Depends(get_ctx)):
    """Download ``events.jsonl``, ``stdout.log``, ``stderr.log`` or a rendered ``log.txt``."""
    row = _row(sctx, jid)
    job_dir = Path(row["job_dir"])
    if stream == "text":
        text = await asyncio.to_thread(_render_text, _events_path(row, sctx))
        return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{jid}-log.txt"'})
    name, media = {"events": ("events.jsonl", "application/x-ndjson"), "stdout": ("stdout.log", "text/plain"),
                   "stderr": ("stderr.log", "text/plain")}[stream]
    path = _events_path(row, sctx) if stream == "events" else job_dir / name
    if not path.is_file():
        raise ApiError("not_found", f"{jid} has no {name}")
    return FileResponse(path, media_type=media, filename=f"{jid}-{name}", headers={"Cache-Control": "no-store"})


def _metric_points(path: Path, names: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {n: [] for n in names}
    evs, _last, _eof = read_events(path, None, types={"metric"})
    for ev in evs:
        name = ev.get("name")
        if name not in names:
            continue
        v = ev.get("value")
        num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        text = None if num is not None or v is None else (("true" if v else "false") if isinstance(v, bool)
                                                          else str(v))
        out[name].append({"cursor": ev["cursor"], "ts": ev.get("ts") or 0.0, "value": num, "value_text": text,
                          "tags": ev.get("tags") or {}})
    return out


@router.get("/jobs/{jid}/metrics", response_model=dict[str, list[MetricPoint]])
async def metrics(jid: str, names: str = Query(..., description="comma list of metric names"),
                  sctx: StudioContext = Depends(get_ctx)):
    row = _row(sctx, jid)
    return await asyncio.to_thread(_metric_points, _events_path(row, sctx), set(_split(names)))


@router.get("/jobs/{jid}/warnings", response_model=list[WarningRow])
async def warnings(jid: str, sctx: StudioContext = Depends(get_ctx)):
    _row(sctx, jid)
    state, _eta = await _manager(sctx).projection(jid)
    return tracker.snapshot_parts(state)["warnings"]


@router.get("/jobs/{jid}/resources", response_model=list[ResourceRow])
async def resources(jid: str, since_ts: float | None = None, sctx: StudioContext = Depends(get_ctx)):
    _row(sctx, jid)
    return _resources(sctx, jid, since_ts if since_ts is not None else 0.0)


@router.post("/jobs/{jid}/cancel", response_model=Job, status_code=202)
async def cancel(jid: str, sctx: StudioContext = Depends(get_ctx)):
    """Request a cancel (``cancelling``; ``cancelled`` at once when still queued)."""
    return await _manager(sctx).cancel(jid)


@router.post("/jobs/{jid}/kill", response_model=Job, status_code=202)
async def kill(jid: str, body: KillRequest | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Force stop: SIGKILL to the process group (after the grace period, or with ``force_now``)."""
    return await _manager(sctx).kill(jid, force_now=bool(body and body.force_now))


@router.post("/jobs/{jid}/retry", response_model=Job, status_code=202)
async def retry(jid: str, sctx: StudioContext = Depends(get_ctx)):
    return await _manager(sctx).retry(jid)


@router.patch("/jobs/{jid}", response_model=Job)
async def patch_job(jid: str, body: JobPatch, sctx: StudioContext = Depends(get_ctx)):
    return await _manager(sctx).set_priority(jid, body.priority)


@router.delete("/jobs/{jid}", response_model=Ok)
async def delete_job(jid: str, files: bool = False, sctx: StudioContext = Depends(get_ctx)):
    await _manager(sctx).delete(jid, files=files)
    return {"ok": True}


@router.get("/queue", response_model=QueueState)
async def queue(sctx: StudioContext = Depends(get_ctx)):
    return _manager(sctx).queue_state()


@router.post("/queue/pause", response_model=QueueState)
async def queue_pause(sctx: StudioContext = Depends(get_ctx)):
    mgr = _manager(sctx)
    await mgr.set_paused(True)
    return mgr.queue_state()


@router.post("/queue/resume", response_model=QueueState)
async def queue_resume(sctx: StudioContext = Depends(get_ctx)):
    mgr = _manager(sctx)
    await mgr.set_paused(False)
    return mgr.queue_state()


def _timings(sctx: StudioContext, unit: str | None, stage: str | None, project: str | None, limit: int) -> dict:
    rows = sctx.db.fetchall("SELECT unit, seconds, n_cells, threads FROM unit_timings WHERE host_id = ? ORDER BY ts",
                            (host_id(),))
    per: dict[str, list[float]] = {}
    for r in rows:
        u = r["unit"]
        if unit and not (u == unit or u.startswith(unit.rstrip("*").rstrip(":") + ":") or u.startswith(unit)):
            continue
        if not r["seconds"] or r["seconds"] <= 0:
            continue
        s = scale(u, r["n_cells"], r["threads"])
        per.setdefault(u, []).append(r["seconds"] / s if s > 0 else r["seconds"])
    rates = []
    for u, xs in sorted(per.items()):
        model = CostModel({u: xs})
        lo, hi = model.rate_range(u)
        rates.append({"unit": u, "median_s": round(model.rate(u), 6), "p25_s": round(lo, 6), "p75_s": round(hi, 6),
                      "n": len(xs)})
    where, args = [], []
    if stage:
        where.append("t.stage = ?")
        args.append(stage)
    if project:
        where.append("r.project_id = ?")
        args.append(project)
    sql = ("SELECT t.run_id, r.label, COALESCE(t.git_commit, r.git_commit) AS git_commit, t.stage, t.seconds, "
           "t.n_points, COALESCE(t.mode, r.mode) AS mode, t.threads FROM stage_timings t "
           "LEFT JOIN runs r ON r.id = t.run_id")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.created_utc DESC, t.run_id DESC LIMIT ?"
    hist = sctx.db.fetchall(sql, (*args, limit))
    return {"unit_rates": rates, "stage_history": hist}


@router.get("/timings", response_model=Timings)
async def timings(unit: str | None = None, stage: str | None = None, project: str | None = None,
                  limit: int = Query(500, ge=1, le=10_000), sctx: StudioContext = Depends(get_ctx)):
    """Per-host unit rates (normalised to 54,701 cells and 4 threads) and the stage-duration history."""
    return await asyncio.to_thread(_timings, sctx, unit, stage, project, limit)
