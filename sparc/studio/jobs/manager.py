"""JobManager: queue, lanes, chains, per-run locks, thread budget, preflight, spawn, cancel, kill, reattach.

State machine (SPEC §5.8)::

    queued → starting → running → succeeded | failed | cancelled
                           └→ cancelling → cancelled | (grace 90 s) → killed → cancelled
    starting --(no worker event within 30 s)--> failed("worker did not start")
    running --(pid vanished, no run.end / result.json)--> interrupted
    queued --(dependency / lock / preflight)--> blocked → queued

Scheduling (SPEC §10.4–10.5): queued and blocked jobs are visited by priority
(desc), then creation order.  A job waits as ``blocked`` (with a reason and
actions) behind an unfinished ``after_job_id`` (it is cancelled when that job
fails), behind another job holding its run's write lock (``run_locks``), or
when a preflight check fails; it waits as ``queued`` for a lane slot (heavy 1,
medium 1, network 2, engine 1 - user settings) or for the thread budget
(the running jobs' threads plus its own may exceed ``thread_budget`` by at
most one).  The queue can be paused.

Every transition is written to SQLite, to ``state.json`` and as a
``job.status`` line in the job's own ``events.jsonl`` (one ordered log), and
broadcast as a global ``job.status`` event.  Feature items create jobs with
``await sctx.jobs.submit(kind, params, run_id=…, …)``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.events import append_event
from sparc.studio.jobs import kinds as kindsmod
from sparc.studio.jobs import tracker
from sparc.studio.jobs.eta import CostModel, estimate_units, host_id, projection_eta
from sparc.studio.jobs.executors import ExitInfo, ProcessExecutor, registered_executors
from sparc.studio.jobs.resources import ResourceSampler, preflight_disk, preflight_memory
from sparc.studio.jobs.tailer import JobTailer, iter_lines, parse_line
from sparc.studio.schemas.common import ACTIVE_STATUSES, FINAL_STATUSES, LIVE_STATUSES
from sparc.studio.workspace import new_id, read_json, utc_iso, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.jobs")

__all__ = ["JobManager", "job_out", "LANE_SLOT_KEYS"]

LANE_SLOT_KEYS = {"heavy": "heavy_slots", "medium": "medium_slots", "network": "network_slots"}
OOM_FRACTION = 0.8
NETCHECK_TTL_S = 600.0
RETENTION_EVERY_S = 1800.0
PENDING = ("queued", "blocked")


def job_out(row: dict) -> dict:
    """A ``jobs`` row as the api.md ``Job`` object."""
    return {
        "id": row["id"], "kind": row["kind"], "lane": row["lane"], "executor": row["executor"],
        "label": row.get("label") or row["kind"], "status": row["status"],
        "project_id": row.get("project_id"), "run_id": row.get("run_id"), "study_id": row.get("study_id"),
        "scenario_id": row.get("scenario_id"), "parent_job_id": row.get("parent_job_id"),
        "after_job_id": row.get("after_job_id"), "priority": int(row.get("priority") or 0),
        "params": dbmod.loads(row.get("params_json"), {}) or {},
        "created_utc": row.get("created_utc") or "", "started_utc": row.get("started_utc"),
        "finished_utc": row.get("finished_utc"), "progress": row.get("progress"),
        "eta_s": row.get("eta_s"), "eta_lo": row.get("eta_lo"), "eta_hi": row.get("eta_hi"),
        "stage": row.get("stage"), "current_path": dbmod.loads(row.get("current_path")),
        "exit_code": row.get("exit_code"), "error": _error_out(dbmod.loads(row.get("error_json"))),
        "blocked": dbmod.loads(row.get("blocked_json")), "result": dbmod.loads(row.get("result_json")),
        "peak_rss_mb": row.get("peak_rss_mb"), "threads": row.get("threads"),
    }


def _error_out(err: Any) -> dict | None:
    """A stored error in the ``Job.error`` shape ``{type, message, ...}`` (a kind's ``result.json`` may carry
    a bare string or omit a key; one such job must not break ``GET /api/jobs``)."""
    if err is None or err == {}:
        return None
    if not isinstance(err, dict):
        return {"type": "Error", "message": str(err)}
    return {**err, "type": str(err.get("type") or "Error"), "message": str(err.get("message") or "")}


def _valid_actions(actions, kind: str) -> list[dict]:
    """The ``Action``-shaped entries of a blocked reason (api.md §0.3); others are dropped and logged."""
    from pydantic import ValidationError

    from sparc.studio.schemas.common import Action

    out = []
    for a in actions or []:
        try:
            out.append(Action.model_validate(a).model_dump(exclude_none=True))
        except ValidationError:
            log.warning("%s: dropping a blocked action that is not an Action: %r", kind, a)
    return out


class JobManager:
    """The scheduler and lifecycle owner of every job (see the module docstring)."""

    def __init__(self, sctx):
        self.sctx = sctx
        self.db = sctx.db
        self.hub = sctx.hub
        self.ws = sctx.workspace
        self.cfg = sctx.config
        self.executors: dict[str, Any] = {}
        self.tailers: dict[str, JobTailer] = {}
        self.watchers: dict[str, asyncio.Task] = {}
        self.samples: dict[str, dict] = {}
        self.killed: set[str] = set()
        self.start_failed: set[str] = set()
        self.cost_model = CostModel()
        self.sampler: ResourceSampler | None = None
        self._wake = asyncio.Event()
        self._sched_task: asyncio.Task | None = None
        self._sched_lock = asyncio.Lock()
        self._status_lock = asyncio.Lock()
        self._stopping = False
        self._netcheck: dict[str, tuple[float, dict]] = {}
        self._replays: dict[str, tuple[tuple, dict]] = {}
        self._starting: set[str] = set()
        self._run_submit: dict[str, asyncio.Lock] = {}
        self.paused = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self, *, reattach: bool = True) -> None:
        self.hub.bind()
        self.executors = {"process": ProcessExecutor(test_kinds=bool(self.cfg.test_kinds),
                                                     poll_s=self.cfg.exit_poll_s)}
        for name, ex in registered_executors().items():
            self.executors.setdefault(name, ex)
        for ex in self.executors.values():
            bind = getattr(ex, "bind", None)
            if callable(bind):
                try:
                    bind(self.sctx)
                except Exception:
                    log.exception("executor bind failed")
        self.paused = bool(self.db.get_setting("_queue_paused", False))
        try:
            self.cost_model = await asyncio.to_thread(CostModel.from_db, self.db)
        except Exception:
            self.cost_model = CostModel()
        await self.db.aexecute("DELETE FROM run_locks WHERE job_id NOT IN (SELECT id FROM jobs WHERE status IN "
                               "('starting','running','cancelling'))")
        if reattach:
            await self.reattach_all()
        self._sched_task = asyncio.create_task(self._schedule_loop(), name="job-scheduler")
        self.wake()

    def start_sampler(self) -> None:
        """Start the ResourceSampler (the last startup step, SPEC §10.9)."""
        if self.cfg.sampler and self.sampler is None:
            self.sampler = ResourceSampler(db=self.db, hub=self.hub, workspace_root=self.ws.root,
                                           live_jobs=self.live_pids, on_sample=self._on_sample,
                                           interval=self.cfg.sample_interval_s).start()

    async def stop(self, *, stop_jobs: bool = False) -> None:
        """Stop scheduling and tailing.  Jobs keep running in their own process groups unless ``stop_jobs``."""
        self._stopping = True
        if self._sched_task is not None:
            self._sched_task.cancel()
            try:
                await self._sched_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sched_task = None
        if self.sampler is not None:
            await self.sampler.stop()
            self.sampler = None
        if stop_jobs:
            live = [r for r in self._rows_by_status(LIVE_STATUSES) if r["lane"] != "none"]
            for row in live:
                try:
                    await self.cancel(row["id"], by="shutdown")
                except ApiError:
                    pass
            pending = [self.watchers[r["id"]] for r in live if r["id"] in self.watchers]
            if pending:
                await asyncio.wait(pending, timeout=self.cfg.shutdown_grace_s)
            for row in live:
                if row["id"] in self.watchers and not self.watchers[row["id"]].done():
                    try:
                        await self.kill(row["id"], force_now=True, by="shutdown")
                    except ApiError:
                        pass
            pending = [t for t in self.watchers.values() if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=10)
        for jid, t in list(self.watchers.items()):
            if not t.done():
                t.cancel()
        for t in list(self.watchers.values()):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self.watchers.clear()
        for jid, tailer in list(self.tailers.items()):
            try:
                await tailer.stop(drain=True)
            except Exception:
                log.exception("final flush of %s failed", jid)
        self.tailers.clear()

    def wake(self) -> None:
        self._wake.set()

    async def _schedule_loop(self) -> None:
        last_retention = 0.0
        while not self._stopping:
            try:
                await self.schedule()
                if time.monotonic() - last_retention > RETENTION_EVERY_S:
                    last_retention = time.monotonic()
                    await self.apply_retention()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduling failed")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), self.cfg.schedule_interval_s)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------ reads

    def get_row(self, job_id: str) -> dict | None:
        return self.db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))

    def get(self, job_id: str) -> dict:
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        return job_out(row)

    def _rows_by_status(self, statuses) -> list[dict]:
        marks = ",".join("?" for _ in statuses)
        return self.db.fetchall(f"SELECT * FROM jobs WHERE status IN ({marks}) "
                                f"ORDER BY priority DESC, created_utc ASC, rowid ASC", tuple(statuses))

    def live_pids(self) -> dict[str, int]:
        return {r["id"]: r["pid"] for r in self._rows_by_status(LIVE_STATUSES)
                if r.get("pid") and r["executor"] == "process"}

    def active_count(self) -> int:
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        return int(self.db.fetchval(f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})", ACTIVE_STATUSES) or 0)

    def settings(self):
        return self.sctx.settings()

    # ------------------------------------------------------------------ submit

    async def submit(self, kind: str, params: dict | BaseModel | None = None, *, project_id: str | None = None,
                     run_id: str | None = None, study_id: str | None = None, scenario_id: str | None = None,
                     priority: int = 0, after_job_id: str | None = None, parent_job_id: str | None = None,
                     label: str | None = None, threads: int | None = None) -> dict:
        """Validate and queue a job; returns the ``Job``.  Errors: ``404 unknown_kind``, ``422 validation``,
        ``409 no_checkpoint``, ``404 not_found`` (``after_job_id`` or run)."""
        k = kindsmod.get_kind(kind)
        if k is None:
            raise ApiError("unknown_kind", f"unknown job kind {kind!r}")
        if k.lane == "none" or k.executor == "external":
            raise ApiError("validation", f"{kind} jobs are created by Studio itself",
                           detail={"errors": [{"path": "kind", "message": "not creatable", "code": "not_creatable"}]})
        p = k.validate_params(params.model_dump(mode="json") if isinstance(params, BaseModel) else params)
        params_d = k.dump_params(p)
        if k.needs_run:
            if not run_id:
                raise ApiError("validation", f"{kind} needs a run",
                               detail={"errors": [{"path": "run_id", "message": "required", "code": "missing"}]})
            run = self.db.fetchone("SELECT id, run_dir, project_id FROM runs WHERE id = ?", (run_id,))
            if run is None:
                raise ApiError("not_found", f"no run {run_id!r}")
            project_id = project_id or run.get("project_id")
            if k.needs_checkpoint and not (Path(run["run_dir"]) / "checkpoint.pkl").is_file():
                raise ApiError("no_checkpoint", f"{k.label} needs the run's checkpoint.pkl",
                               detail={"run_id": run_id})
        if after_job_id and self.get_row(after_job_id) is None:
            raise ApiError("not_found", f"no job {after_job_id!r} to run after")
        jid = new_id("j")
        job_dir = self.ws.job_dir(jid)
        job_dir.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        row = {"id": jid, "kind": kind, "lane": k.lane, "executor": k.executor, "label": label or k.label,
               "project_id": project_id, "run_id": run_id, "study_id": study_id, "scenario_id": scenario_id,
               "parent_job_id": parent_job_id, "after_job_id": after_job_id, "params_json": dbmod.dumps(params_d),
               "priority": int(priority or 0), "status": "queued", "job_dir": str(job_dir), "threads": threads,
               "created_utc": now, "last_cursor": -1, "host_id": host_id()}
        est = await self._estimate(k, row, p)
        if est:
            row.update({"eta_s": est.get("est_s"), "eta_lo": est.get("est_lo"), "eta_hi": est.get("est_hi")})
        self._write_job_json(row, k)
        append_event(job_dir / "events.jsonl", "job.status", job_id=jid, status="queued", exit_code=None,
                     error=None)
        await self.db.ainsert("jobs", row)
        job = self.get(jid)
        self.hub.publish("job.created", {"job": job})
        self.wake()
        return job

    def _write_job_json(self, row: dict, k, context: dict | None = None) -> None:
        job_dir = Path(row["job_dir"])
        data = {"id": row["id"], "kind": row["kind"], "lane": row["lane"], "executor": row["executor"],
                "label": row.get("label"), "params": dbmod.loads(row.get("params_json"), {}) or {},
                "project_id": row.get("project_id"), "run_id": row.get("run_id"), "study_id": row.get("study_id"),
                "scenario_id": row.get("scenario_id"), "parent_job_id": row.get("parent_job_id"),
                "after_job_id": row.get("after_job_id"), "priority": row.get("priority") or 0,
                "threads": row.get("threads"), "created_utc": row.get("created_utc"),
                "module": getattr(k, "module", None) if k is not None else None}
        if context:
            data["context"] = context
        write_json_atomic(job_dir / "job.json", data)

    async def _estimate(self, k, row: dict, params) -> dict | None:
        if k.estimate is None:
            return None
        try:
            est = await asyncio.to_thread(k.estimate, self.sctx, job_out({**row, "status": row.get("status")}), params)
        except Exception:
            log.exception("estimate for %s failed", k.kind)
            return None
        if not isinstance(est, dict):
            return None
        est = dict(est)
        if est.get("units") and est.get("est_s") is None:
            e, lo, hi = estimate_units(est["units"], est.get("n_cells"), est.get("threads") or row.get("threads"),
                                       self.cost_model)
            est.update(est_s=round(e, 3), est_lo=round(lo, 3), est_hi=round(hi, 3))
        return est

    # ------------------------------------------------------------------ status

    async def _set_status(self, job_id: str, status: str, *, publish: bool = True,
                          only_from: tuple[str, ...] | None = None, **fields) -> dict | None:
        """Move a job to ``status`` (plus ``fields``); returns the new row, or None when the job is gone.

        Transitions are serialised by one lock, so a check-then-set is atomic: with ``only_from`` the
        transition happens only when the current status is one of those (else None) - a cancel that
        lands while the scheduler is starting the same job can never be undone by it, and vice versa.
        The ``job.status`` line is appended before the row changes, so whoever sees the new status in
        SQLite also finds its line in ``events.jsonl``.
        """
        async with self._status_lock:
            row = self.get_row(job_id)
            if row is None:
                return None
            prev = row["status"]
            if only_from is not None and prev not in only_from:
                return None
            values = {"status": status, **fields}
            if status in FINAL_STATUSES and "finished_utc" not in values:
                values["finished_utc"] = utc_now()
            if status not in ("blocked",) and "blocked_json" not in values:
                values["blocked_json"] = None
            error = dbmod.loads(values.get("error_json")) if "error_json" in values else dbmod.loads(
                row.get("error_json"))
            exit_code = values.get("exit_code", row.get("exit_code"))
            if prev != status:
                try:
                    append_event(Path(row["job_dir"]) / "events.jsonl", "job.status", job_id=job_id, status=status,
                                 exit_code=exit_code, error=error)
                except OSError:
                    log.exception("cannot append job.status for %s", job_id)
            await self.db.aupdate("jobs", {"id": job_id}, values)
        if prev != status:
            self._write_state(row, status=status)
            if publish:
                self.hub.publish("job.status", {"job_id": job_id, "status": status, "prev_status": prev,
                                                "exit_code": exit_code, "error": error, "run_id": row.get("run_id"),
                                                "project_id": row.get("project_id"), "kind": row["kind"],
                                                "label": row.get("label") or row["kind"]})
        return self.get_row(job_id)

    def _write_state(self, row: dict, **updates) -> None:
        path = Path(row["job_dir"]) / "state.json"
        state = read_json(path, {}) or {}
        state.setdefault("executor", row.get("executor"))
        state.update(updates)
        try:
            write_json_atomic(path, state)
        except OSError:
            log.exception("cannot write %s", path)

    async def _block(self, row: dict, reason: str, actions: list | None = None) -> None:
        blocked = {"reason": str(reason), "actions": _valid_actions(actions, row["kind"])}
        if row["status"] == "blocked" and dbmod.loads(row.get("blocked_json")) == blocked:
            return
        if row["status"] == "blocked":         # same status, new reason: no transition, just the details
            await self.db.aexecute("UPDATE jobs SET blocked_json = ? WHERE id = ? AND status = 'blocked'",
                                   (dbmod.dumps(blocked), row["id"]))
            return
        await self._set_status(row["id"], "blocked", blocked_json=dbmod.dumps(blocked), only_from=PENDING)

    async def _unblock(self, row: dict) -> None:
        if row["status"] == "blocked":
            await self._set_status(row["id"], "queued", only_from=("blocked",))

    # ------------------------------------------------------------------ scheduling

    def _threads_for(self, row: dict, k, settings, heavy_running: bool) -> int:
        params = dbmod.loads(row.get("params_json"), {}) or {}
        requested = params.get("threads")
        if row.get("threads"):
            n = int(row["threads"])
        elif k.threads is not None:
            n = int(k.threads(settings, params))
        elif row["lane"] == "heavy":
            n = requested if isinstance(requested, int) and requested > 0 else settings.threads_heavy
        elif row["lane"] == "engine":
            n = 1 if heavy_running else settings.engine_threads
        else:
            n = 1
        return max(1, min(n, settings.thread_budget))

    async def schedule(self) -> None:
        """One pass over the queue (see the module docstring)."""
        async with self._sched_lock:
            if self._stopping:
                return
            settings = self.settings()
            live = self._rows_by_status(LIVE_STATUSES)
            running = {lane: 0 for lane in ("heavy", "medium", "network", "engine")}
            threads_used = 0
            heavy_running = False
            for r in live:
                if r["lane"] in running:
                    running[r["lane"]] += 1
                    threads_used += int(r.get("threads") or 1)
                heavy_running = heavy_running or r["lane"] == "heavy"
            for pending in self._rows_by_status(PENDING):
                row = self.get_row(pending["id"])      # earlier steps of this pass awaited: re-read it
                if row is None or row["status"] not in PENDING:
                    continue
                k = kindsmod.get_kind(row["kind"])
                if k is None:
                    await self._fail(row, "UnknownKind", f"job kind {row['kind']!r} is not available",
                                     only_from=PENDING)
                    continue
                if row.get("after_job_id"):
                    dep = self.get_row(row["after_job_id"])
                    if dep is None or dep["status"] in ("failed", "cancelled", "interrupted"):
                        what = "is missing" if dep is None else dep["status"]
                        if await self._set_status(row["id"], "cancelled", only_from=PENDING, error_json=dbmod.dumps(
                                {"type": "DependencyFailed", "message": f"job {row['after_job_id']} {what}"})):
                            await self._call_on_finish(row["id"])
                        continue
                    if dep["status"] != "succeeded":
                        await self._block(row, f"waiting for {dep.get('label') or dep['kind']} ({dep['id']})")
                        continue
                if k.locks_run and row.get("run_id"):
                    holder = self.db.fetchone(
                        "SELECT l.job_id, j.kind, j.label FROM run_locks l LEFT JOIN jobs j ON j.id = l.job_id "
                        "WHERE l.run_id = ?", (row["run_id"],))
                    if holder and holder["job_id"] != row["id"]:
                        await self._block(row, f"waiting for {holder.get('kind') or 'a job'} on this run",
                                          [{"kind": "open", "label": "Show that job", "method": "GET",
                                            "path": f"/jobs/{holder['job_id']}"}])
                        continue
                if self.paused:
                    await self._unblock(row)
                    continue
                lane = row["lane"]
                slots = int(getattr(settings, LANE_SLOT_KEYS[lane])) if lane in LANE_SLOT_KEYS else 1
                if lane not in running or running[lane] >= slots:
                    await self._unblock(row)
                    continue
                threads = self._threads_for(row, k, settings, heavy_running)
                if live and threads_used + threads > settings.thread_budget + 1:
                    await self._unblock(row)
                    continue
                failures = await self._preflight(row, k, settings)
                fatal = [f for f in failures if f.get("fatal")]
                if fatal:
                    await self._fail(row, "PreflightFailed", fatal[0]["reason"], detail=fatal[0], only_from=PENDING)
                    continue
                if failures:
                    await self._block(row, failures[0]["reason"], failures[0].get("actions"))
                    continue
                started = await self._start(row, k, threads)
                if started:
                    running[lane] += 1
                    threads_used += threads
                    heavy_running = heavy_running or lane == "heavy"
                    live = live + [row]

    async def _preflight(self, row: dict, k, settings) -> list[dict]:
        """Checks before a heavy or ``engine.open`` job starts (SPEC §10.4); ``[]`` when it may start."""
        out: list[dict] = []
        params = dbmod.loads(row.get("params_json"), {}) or {}
        if k.needs_checkpoint and row.get("run_id"):
            run = self.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (row["run_id"],))
            if run is None or not (Path(run["run_dir"]) / "checkpoint.pkl").is_file():
                return [{"reason": "the run has no checkpoint.pkl", "fatal": True, "code": "no_checkpoint"}]
        if settings.offline and k.network_hosts:
            out.append({"reason": "offline mode is on (Settings); this job needs the network", "actions": []})
        if row["lane"] == "heavy" or row["kind"] == "engine.open":
            est = await self._estimate(k, row, params) or {}
            others = sum(float((self.samples.get(j) or {}).get("rss_mb") or 0) for j in self.live_pids())
            if est.get("peak_ram_gb"):
                mem = preflight_memory(float(est["peak_ram_gb"]), others)
                if mem:
                    actions = [{"kind": "open", "label": f"Stop {r.get('label') or r['kind']}", "method": "POST",
                                "path": f"/api/jobs/{r['id']}/cancel"} for r in self._rows_by_status(LIVE_STATUSES)]
                    out.append({**mem, "actions": actions})
            if est.get("disk_bytes"):
                disk = preflight_disk(self.ws.root, float(est["disk_bytes"]))
                if disk:
                    out.append({**disk, "actions": []})
        if k.preflight is not None:
            try:
                extra = await asyncio.to_thread(k.preflight, self.sctx, job_out(row), params)
                out.extend(dict(x) for x in (extra or []))
            except Exception as exc:
                log.exception("preflight of %s failed", row["kind"])
                out.append({"reason": f"preflight error: {exc}", "fatal": True})
        return out

    async def _start(self, row: dict, k, threads: int) -> bool:
        executor = self.executors.get(row["executor"])
        if executor is None:
            await self._fail(row, "NoExecutor", f"no {row['executor']!r} executor is available", only_from=PENDING)
            return False
        jid = row["id"]
        started = utc_now()
        # claim the job: a cancel that landed since the scheduler read it wins
        if await self._set_status(jid, "starting", only_from=PENDING, threads=threads, started_utc=started,
                                  host_id=host_id()) is None:
            return False
        self._starting.add(jid)
        if k.locks_run and row.get("run_id"):
            await self.db.aexecute("INSERT OR REPLACE INTO run_locks (run_id, job_id, acquired_utc) VALUES (?,?,?)",
                                   (row["run_id"], jid, utc_now()))
        context = self._context(row)
        row = {**row, "threads": threads}
        self._write_job_json(row, k, context)
        self._write_state(row, status="starting", started_utc=started, executor=row["executor"])
        tailer = self._make_tailer(row, flushed_cursor=int(row.get("last_cursor") if row.get("last_cursor")
                                                             is not None else -1))
        self.tailers[jid] = tailer.start()
        job = {**row, "job_dir": row["job_dir"], "status": "starting"}
        try:
            info = await executor.submit(job)
        except Exception as exc:
            log.exception("spawning %s failed", jid)
            await self._finalize(jid, ExitInfo(None), forced=("failed", {"type": type(exc).__name__,
                                                                          "message": f"could not start: {exc}"}))
            return False
        info = info or {}
        self._write_state(row, status="starting", started_utc=started, **{
            k2: info.get(k2) for k2 in ("pid", "pgid", "proc_create_time", "cmdline_token", "executor")
            if info.get(k2) is not None})
        await self.db.aupdate("jobs", {"id": jid}, {"pid": info.get("pid"), "pgid": info.get("pgid"),
                                                     "proc_create_time": info.get("proc_create_time")})
        self.watchers[jid] = asyncio.create_task(self._watch(jid, executor, reattached=False), name=f"watch-{jid}")
        return True

    def _context(self, row: dict) -> dict:
        ctx: dict[str, Any] = {"workspace": str(self.ws.root), "cache_dir": str(self.ws.cache_dir)}
        if row.get("project_id"):
            p = self.db.fetchone("SELECT dir, config_path FROM projects WHERE id = ?", (row["project_id"],))
            if p:
                ctx["project_dir"] = p["dir"]
                ctx["config_path"] = p["config_path"]
        if row.get("run_id"):
            r = self.db.fetchone("SELECT run_dir, studio_dir FROM runs WHERE id = ?", (row["run_id"],))
            if r:
                ctx["run_dir"], ctx["studio_dir"] = r["run_dir"], r["studio_dir"]
        if row.get("study_id"):
            s = self.db.fetchone("SELECT out_dir FROM studies WHERE id = ?", (row["study_id"],))
            if s and s.get("out_dir"):
                ctx["study_dir"] = s["out_dir"]
        return ctx

    def _make_tailer(self, row: dict, *, flushed_cursor: int) -> JobTailer:
        mode = None
        if row.get("run_id"):
            mode = self.db.fetchval("SELECT mode FROM runs WHERE id = ?", (row["run_id"],))
        tailer = JobTailer(db=self.db, hub=self.hub, job_id=row["id"],
                           events_path=Path(row["job_dir"]) / "events.jsonl", run_id=row.get("run_id"),
                           threads=row.get("threads"), mode=mode or row["kind"], flushed_cursor=flushed_cursor,
                           interval=self.cfg.tail_interval_s, flush_interval=self.cfg.flush_interval_s,
                           cost_model=self.cost_model, on_event=self._on_event, on_progress=self._on_progress)
        tailer.kind = row["kind"]
        return tailer

    async def _watch(self, jid: str, executor, *, reattached: bool) -> None:
        row = self.get_row(jid) or {}
        wait = asyncio.create_task(executor.wait({**row, "id": jid}))
        deadline = time.monotonic() + self.cfg.start_timeout_s
        try:
            while not wait.done():
                tailer = self.tailers.get(jid)
                if (not reattached and tailer is not None and not tailer.worker_seen
                        and time.monotonic() > deadline and jid not in self.start_failed):
                    await tailer.poll()
                    if not tailer.worker_seen:
                        self.start_failed.add(jid)
                        log.warning("job %s: worker did not start within %.0f s", jid, self.cfg.start_timeout_s)
                        await self._set_status(jid, "failed", only_from=LIVE_STATUSES, error_json=dbmod.dumps(
                            {"type": "StartTimeout", "message": "worker did not start"}))
                        await executor.kill(self.get_row(jid) or row)
                await asyncio.wait({wait}, timeout=0.5)
            exit_info = wait.result()
        except asyncio.CancelledError:
            wait.cancel()
            raise
        await self._finalize(jid, exit_info)

    # ------------------------------------------------------------------ events and samples

    async def _on_event(self, jid: str, cursor: int, ev: dict) -> None:
        t = ev.get("type")
        if jid in self._starting and t not in ("job.status", "cancel.requested"):
            self._starting.discard(jid)
            await self._set_status(jid, "running", only_from=("starting",))
        if t == "artifact":
            self._output_written(jid, ev)
        tailer = self.tailers.get(jid)
        k = kindsmod.get_kind(tailer.kind) if tailer is not None and getattr(tailer, "kind", None) else None
        if k is not None and k.on_event is not None and t in k.on_event_types:
            row = self.get_row(jid)
            try:
                await asyncio.to_thread(k.on_event, self.sctx, job_out(row), ev)
            except Exception:
                log.exception("on_event hook of %s failed", k.kind)

    def _output_written(self, jid: str, ev: dict) -> None:
        tailer = self.tailers.get(jid)
        run_id = getattr(tailer, "run_id", None)
        if not run_id or tracker.is_nested(ev) or not isinstance(ev.get("path"), str):
            return
        self.hub.publish("output.written", {"run_id": run_id, "relpath": ev["path"], "role": ev.get("role"),
                                            "output_id": _output_id(ev["path"]), "stage": ev.get("stage"),
                                            "bytes": int(ev.get("bytes") or 0)})

    def _on_progress(self, jid: str, payload: dict) -> None:
        self.hub.publish("job.progress", payload)

    def _on_sample(self, jid: str, sample: dict) -> None:
        try:
            import psutil

            sample = {**sample, "avail_mb": psutil.virtual_memory().available / 2 ** 20}
        except Exception:
            pass
        self.samples[jid] = sample

    # ------------------------------------------------------------------ finishing

    async def _fail(self, row: dict, etype: str, message: str, *, detail: dict | None = None,
                    only_from: tuple[str, ...] | None = None) -> None:
        err = {"type": etype, "message": message}
        if detail:
            err["detail"] = {k: v for k, v in detail.items() if k not in ("actions",)}
        if await self._set_status(row["id"], "failed", only_from=only_from, error_json=dbmod.dumps(err)):
            await self._call_on_finish(row["id"])

    async def _finalize(self, jid: str, exit_info: ExitInfo, *, forced: tuple[str, dict] | None = None) -> None:
        """Work has ended: drain the events, decide the final status, release locks, run ``on_finish``."""
        tailer = self.tailers.pop(jid, None)
        if tailer is not None:
            await tailer.stop(drain=True)
        self.watchers.pop(jid, None)
        row = self.get_row(jid)
        if row is None:
            return
        state = tailer.state if tailer is not None else None
        result_json = read_json(Path(row["job_dir"]) / "result.json")
        status, error, result = self._final_status(row, exit_info, result_json, state)
        if forced is not None:
            status, error = forced
        if jid in self.start_failed:
            status = "failed"
            error = dbmod.loads(row.get("error_json")) or {"type": "StartTimeout", "message": "worker did not start"}
        exit_code = exit_info.code if exit_info.code is not None else (result_json or {}).get("exit_code")
        values: dict[str, Any] = {"exit_code": exit_code, "error_json": dbmod.dumps(error)}
        if result is not None:
            values["result_json"] = dbmod.dumps(result)
        if status == "succeeded":
            values.update(progress=1.0, eta_s=0.0, eta_lo=0.0, eta_hi=0.0)
        if row["status"] == status:          # already final (start timeout): just record the details
            await self.db.aupdate("jobs", {"id": jid}, values)
        else:
            await self._set_status(jid, status, **values)
        await self.db.aexecute("DELETE FROM run_locks WHERE job_id = ?", (jid,))
        self.hub.publish_job(jid, ("end", status))
        self.killed.discard(jid)
        self.start_failed.discard(jid)
        self._starting.discard(jid)
        self.samples.pop(jid, None)
        await self._call_on_finish(jid)
        self.wake()

    def _final_status(self, row: dict, exit_info: ExitInfo, result_json: dict | None,
                      state: dict | None) -> tuple[str, dict | None, dict | None]:
        jid = row["id"]
        code = exit_info.code
        run_status = (state or {}).get("run_status")
        if isinstance(result_json, dict) and result_json.get("status") in ("succeeded", "failed", "cancelled"):
            return result_json["status"], result_json.get("error"), result_json.get("result")
        if jid in self.killed:
            return "cancelled", None, None
        if code == 130:
            return "cancelled", None, None
        if code in (-15, 143) and self._cancel_was_requested(row):
            return "cancelled", None, None       # SIGTERM landed before the worker installed its handlers
        if code == 0:
            return "succeeded", None, (state or {}).get("result")
        if code is None:
            if run_status in ("succeeded", "failed", "cancelled"):
                err = ((state or {}).get("run") or {}).get("error") if run_status == "failed" else None
                return run_status, err, (state or {}).get("result")
            msg = "process vanished"
            last = self.db.fetchone("SELECT rss_mb FROM resource_samples WHERE job_id = ? ORDER BY ts DESC LIMIT 1",
                                    (jid,))
            if last and last.get("rss_mb") and _high_memory(float(last["rss_mb"])):
                msg = "process vanished; possibly out of memory"
            return "interrupted", {"type": "Interrupted", "message": msg}, None
        if code in (-9, 137):
            sample = self.samples.get(jid)
            if sample and sample.get("rss_mb"):
                rss = float(sample["rss_mb"])
                avail = float(sample.get("avail_mb") or 0.0)
                if rss > OOM_FRACTION * (avail + rss):
                    try:
                        import psutil

                        total = psutil.virtual_memory().total / 2 ** 30
                    except Exception:
                        total = (avail + rss) / 1024
                    peak = max(rss, float(row.get("peak_rss_mb") or 0.0)) / 1024
                    return "failed", {"type": "OutOfMemory",
                                      "message": f"likely out of memory (peak {peak:.1f} GB of {total:.1f} GB)"}, None
            return "failed", {"type": "Killed", "message": "the worker was killed (signal 9)"}, None
        if run_status == "failed":
            err = ((state or {}).get("run") or {}).get("error")
            if err:
                return "failed", err, None
        what = f"signal {-code}" if code < 0 else f"code {code}"
        return "failed", {"type": "WorkerExit", "message": f"the worker exited with {what}"}, None

    async def _call_on_finish(self, jid: str) -> None:
        row = self.get_row(jid)
        if row is None:
            return
        k = kindsmod.get_kind(row["kind"])
        if k is None or k.on_finish is None:
            return
        job = job_out(row)
        try:
            await asyncio.to_thread(k.on_finish, self.sctx, job, job.get("result"))
        except Exception:
            log.exception("on_finish hook of %s failed", k.kind)

    # ------------------------------------------------------------------ actions

    async def cancel(self, job_id: str, *, by: str = "user") -> dict:
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        status = row["status"]
        if row["lane"] == "none" or row["executor"] == "external":
            raise ApiError("not_cancellable", "external runs cannot be cancelled from Studio",
                           detail={"pid": row.get("pid"), "hint": "stop the process where it was started"})
        if status in PENDING:
            if await self._set_status(job_id, "cancelled", only_from=PENDING, error_json=None):
                await self._call_on_finish(job_id)
                self.wake()
                return self.get(job_id)
            row = self.get_row(job_id) or row        # the scheduler started it meanwhile: cancel the live job
            status = row["status"]
        if status in FINAL_STATUSES:
            raise ApiError("not_cancellable", f"the job already {status}", detail={"status": status})
        if status == "cancelling" and by == "user":
            return self.get(job_id)
        job_dir = Path(row["job_dir"])
        append_event(job_dir / "events.jsonl", "cancel.requested", job_id=job_id, by=by)
        (job_dir / "cancel").touch()
        if status != "cancelling":
            await self._set_status(job_id, "cancelling", only_from=("starting", "running"))
        row = self.get_row(job_id)
        if row is None or row["status"] != "cancelling":
            return self.get(job_id)              # it ended meanwhile: never signal a pid that is not ours any more
        executor = self.executors.get(row["executor"])
        if executor is not None:
            await executor.cancel(row)
        return self.get(job_id)

    def _cancel_was_requested(self, row: dict) -> bool:
        return row.get("status") == "cancelling" or (bool(row.get("job_dir")) and
                                                     self.cancel_requested_at(row) is not None)

    def cancel_requested_at(self, row: dict) -> float | None:
        try:
            return (Path(row["job_dir"]) / "cancel").stat().st_mtime
        except OSError:
            return None

    async def kill(self, job_id: str, *, force_now: bool = False, by: str = "kill") -> dict:
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        if row["lane"] == "none" or row["executor"] == "external":
            raise ApiError("not_cancellable", "external runs cannot be stopped from Studio")
        if row["status"] not in ("starting", "running", "cancelling"):
            raise ApiError("not_cancellable", f"the job is {row['status']}", detail={"status": row["status"]})
        if not force_now:
            t = self.cancel_requested_at(row) if row["status"] == "cancelling" else None
            grace = self.cfg.kill_grace_s
            if t is None or time.time() - t < grace:
                left = grace if t is None else max(0.0, grace - (time.time() - t))
                raise ApiError("not_ready", f"Force stop is available {grace:.0f} s after Cancel",
                               detail={"grace_s": grace, "remaining_s": round(left, 1)})
        job_dir = Path(row["job_dir"])
        append_event(job_dir / "events.jsonl", "cancel.requested", job_id=job_id, by="shutdown" if by == "shutdown"
                     else "kill")
        (job_dir / "cancel").touch()
        self.killed.add(job_id)
        if row["status"] != "cancelling":
            await self._set_status(job_id, "cancelling", only_from=("starting", "running"))
        row = self.get_row(job_id)
        if row is None or row["status"] != "cancelling":    # it ended meanwhile: nothing left to kill
            self.killed.discard(job_id)
            status = row["status"] if row else "gone"
            raise ApiError("not_cancellable", f"the job is {status}", detail={"status": status})
        executor = self.executors.get(row["executor"])
        if executor is not None:
            await executor.kill(row)
        return self.get(job_id)

    def run_submit_lock(self, run_id: str) -> asyncio.Lock:
        """The lock a caller holds from checking a run's jobs to queueing its own (resume, retry of
        ``run.core``), so two requests at once cannot both pass an "already active" check."""
        lock = self._run_submit.get(run_id)
        if lock is None:
            lock = self._run_submit[run_id] = asyncio.Lock()
        return lock

    async def retry(self, job_id: str) -> dict:
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        if row["status"] in ACTIVE_STATUSES:
            raise ApiError("active", "the job is still active", detail={"status": row["status"]})
        k = kindsmod.get_kind(row["kind"])
        if k is None:
            raise ApiError("unknown_kind", f"unknown job kind {row['kind']!r}")
        if k.retry is not None:
            return await k.retry(self.sctx, job_out(row))
        params = dbmod.loads(row.get("params_json"), {}) or {}
        if k.retry_params is not None:
            params = await asyncio.to_thread(k.retry_params, self.sctx, job_out(row)) or params
        return await self.submit(row["kind"], params, project_id=row.get("project_id"), run_id=row.get("run_id"),
                                 study_id=row.get("study_id"), scenario_id=row.get("scenario_id"),
                                 priority=int(row.get("priority") or 0), parent_job_id=job_id,
                                 label=row.get("label"))

    async def set_priority(self, job_id: str, priority: int) -> dict:
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        if row["status"] not in ("queued", "blocked"):
            raise ApiError("not_ready", "only queued or blocked jobs can be reprioritised",
                           detail={"status": row["status"]})
        await self.db.aupdate("jobs", {"id": job_id}, {"priority": int(priority)})
        self.wake()
        return self.get(job_id)

    async def delete(self, job_id: str, *, files: bool = False) -> None:
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        if row["status"] in ACTIVE_STATUSES:
            raise ApiError("active", "finished jobs only: cancel it first", detail={"status": row["status"]})

        def _rm(conn):
            if row["status"] == "succeeded":     # its dependency is met: queued dependents no longer wait on it
                conn.execute(f"UPDATE jobs SET after_job_id = NULL WHERE after_job_id = ? AND status IN "
                             f"({','.join('?' for _ in PENDING)})", (job_id, *PENDING))
            for table in ("spans", "metrics", "warnings", "artifacts", "checkpoints", "resource_samples"):
                conn.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

        await self.db.atransaction(_rm)
        if files:
            job_dir = Path(row["job_dir"]).resolve()
            if self.ws.jobs_dir.resolve() in job_dir.parents:
                await asyncio.to_thread(shutil.rmtree, job_dir, True)

    async def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        await asyncio.to_thread(self.db.set_setting, "_queue_paused", self.paused)
        self.wake()

    def queue_state(self) -> dict:
        settings = self.settings()
        lanes = []
        for lane in ("heavy", "medium", "network", "engine"):
            slots = int(getattr(settings, LANE_SLOT_KEYS[lane])) if lane in LANE_SLOT_KEYS else 1
            rows = self.db.fetchall("SELECT id, status FROM jobs WHERE lane = ? AND status IN "
                                    "('queued','blocked','starting','running','cancelling') "
                                    "ORDER BY priority DESC, created_utc ASC, rowid ASC", (lane,))
            lanes.append({"lane": lane, "slots": slots,
                          "running": [r["id"] for r in rows if r["status"] in LIVE_STATUSES],
                          "queued": [r["id"] for r in rows if r["status"] in ("queued", "blocked")]})
        return {"paused": self.paused, "lanes": lanes}

    # ------------------------------------------------------------------ reattach

    async def reattach_all(self) -> dict:
        """Server start: reattach live jobs whose process still matches, mark the rest ``interrupted``."""
        out = {"reattached": [], "finished": []}
        for row in self._rows_by_status(LIVE_STATUSES):
            if row["lane"] == "none" or row["executor"] == "external":
                continue
            st = read_json(Path(row["job_dir"]) / "state.json", {}) or {}
            job = {**row, **{k: st.get(k) for k in ("pid", "pgid", "proc_create_time", "cmdline_token")
                             if st.get(k) is not None}}
            executor = self.executors.get(row["executor"])
            ok = False
            if executor is not None:
                try:
                    ok = await executor.reattach(job)
                except Exception:
                    log.exception("reattach of %s failed", row["id"])
            flushed = int(row["last_cursor"]) if row.get("last_cursor") is not None else -1
            tailer = self._make_tailer(row, flushed_cursor=flushed)
            self.tailers[row["id"]] = tailer
            if ok:
                k = kindsmod.get_kind(row["kind"])
                if k is not None and k.locks_run and row.get("run_id"):
                    # the lock row may be gone (--reindex empties run_locks); the live job still holds the run
                    await self.db.aexecute("INSERT OR IGNORE INTO run_locks (run_id, job_id, acquired_utc) "
                                           "VALUES (?,?,?)", (row["run_id"], row["id"], utc_now()))
                if row["status"] == "starting":
                    self._starting.add(row["id"])
                tailer.start()
                await tailer.poll()
                if row["status"] == "starting" and tailer.worker_seen:
                    # its first events were flushed before the restart, so no new one may arrive soon
                    self._starting.discard(row["id"])
                    await self._set_status(row["id"], "running", only_from=("starting",))
                self.watchers[row["id"]] = asyncio.create_task(self._watch(row["id"], executor, reattached=True),
                                                               name=f"watch-{row['id']}")
                out["reattached"].append(row["id"])
            else:
                await self._finalize(row["id"], ExitInfo(None, reattached=True))
                out["finished"].append(row["id"])
        return out

    # ------------------------------------------------------------------ external (CLI) runs

    def events_path(self, row: dict) -> Path:
        """The event log of a job: ``<job_dir>/events.jsonl``, or ``run_state.events_path`` of the run for a
        ``run.external`` pseudo-job (SPEC §5.14)."""
        tailer = self.tailers.get(row["id"])
        if tailer is not None:
            return tailer.path
        if row.get("executor") == "external" and row.get("run_id"):
            run = self.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (row["run_id"],))
            if run:
                st = read_json(Path(run["run_dir"]) / "run_state.json", {}) or {}
                if st.get("events_path"):
                    return Path(st["events_path"])
        return Path(row["job_dir"]) / "events.jsonl"

    async def create_external(self, kind: str, *, run_id: str, events_path: str | os.PathLike | None,
                              project_id: str | None = None, label: str | None = None, params: dict | None = None,
                              pid: int | None = None) -> dict:
        """Create and start tailing a read-only pseudo-job for a live CLI run (``run.external``, SPEC §5.14).

        The row has lane ``none`` and executor ``external`` (never queued, never cancellable); its
        ``job_dir`` is the directory of the events file.  When ``events_path`` is None nothing is tailed
        (the registry synthesises stage state from ``run_state.json``).  End it with :meth:`detach_external`.
        """
        jid = new_id("j")
        events = Path(events_path) if events_path else None
        row = {"id": jid, "kind": kind, "lane": "none", "executor": "external", "label": label or kind,
               "project_id": project_id, "run_id": run_id, "params_json": dbmod.dumps(params or {"run_id": run_id}),
               "status": "running", "job_dir": str(events.parent if events else ""), "pid": pid,
               "created_utc": utc_now(), "started_utc": utc_now(), "last_cursor": -1, "host_id": host_id()}
        await self.db.ainsert("jobs", row)
        if events is not None:
            self.attach_external(jid, events)
        job = self.get(jid)
        self.hub.publish("job.created", {"job": job})
        return job

    def attach_external(self, job_id: str, events_path: str | os.PathLike) -> JobTailer:
        """Tail the events file of a ``run.external`` pseudo-job the runs registry created (lane ``none``,
        executor ``external``); its tracker, events and stream endpoints then work like any job's."""
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        if job_id in self.tailers:
            return self.tailers[job_id]
        tailer = self._make_tailer(row, flushed_cursor=int(row.get("last_cursor") if row.get("last_cursor")
                                                             is not None else -1))
        tailer.path = Path(events_path)
        self.tailers[job_id] = tailer.start()
        return tailer

    async def detach_external(self, job_id: str, status: str, *, error: dict | None = None) -> None:
        """End a pseudo-job: drain its tailer and record the final status (``succeeded``, ``failed``,
        ``cancelled`` or ``interrupted``)."""
        tailer = self.tailers.pop(job_id, None)
        if tailer is not None:
            await tailer.stop(drain=True)
        row = self.get_row(job_id)
        if row is None or row["status"] in FINAL_STATUSES:
            return
        await self.db.aupdate("jobs", {"id": job_id}, {"status": status, "finished_utc": utc_now(),
                                                       "error_json": dbmod.dumps(error)})
        self.hub.publish("job.status", {"job_id": job_id, "status": status, "prev_status": row["status"],
                                        "exit_code": None, "error": error, "run_id": row.get("run_id"),
                                        "project_id": row.get("project_id"), "kind": row["kind"],
                                        "label": row.get("label") or row["kind"]})
        self.hub.publish_job(job_id, ("end", status))
        await self._call_on_finish(job_id)

    # ------------------------------------------------------------------ retention

    async def apply_retention(self) -> int:
        """Delete finished jobs (rows and directories) older than ``keep_job_logs_days``; returns the count."""
        days = self.settings().keep_job_logs_days
        if days is None:
            return 0
        cutoff = utc_iso(time.time() - float(days) * 86400.0)
        marks = ",".join("?" for _ in FINAL_STATUSES)
        old = self.db.fetchall(f"SELECT id FROM jobs WHERE status IN ({marks}) AND finished_utc IS NOT NULL "
                               f"AND finished_utc < ?", (*FINAL_STATUSES, cutoff))
        for r in old:
            try:
                await self.delete(r["id"], files=True)
            except ApiError:
                pass
        return len(old)

    # ------------------------------------------------------------------ tracker

    async def projection(self, job_id: str) -> tuple[dict, dict]:
        """``(state, eta)`` of a job: the live tailer's projection, else a fresh replay of its events file."""
        tailer = self.tailers.get(job_id)
        if tailer is not None:
            async with tailer.lock:
                return _copy(tailer.state), dict(tailer.eta)
        row = self.get_row(job_id)
        if row is None:
            raise ApiError("not_found", f"no job {job_id!r}")
        path = self.events_path(row)
        state = await asyncio.to_thread(self._replay_cached, str(path))
        eta = projection_eta(state, self.cost_model, threads=row.get("threads"))
        return state, eta

    def _replay_cached(self, path: str) -> dict:
        try:
            st = os.stat(path)
            key = (path, st.st_size, st.st_mtime_ns)
        except OSError:
            return tracker.new_state()
        hit = self._replays.get(path)
        if hit is not None and hit[0] == key:
            return _copy(hit[1])
        state = tracker.new_state()
        for cursor, raw in iter_lines(path, 0):
            tracker.reduce(state, parse_line(raw), cursor)
        if len(self._replays) > 16:
            self._replays.pop(next(iter(self._replays)), None)
        self._replays[path] = (key, state)
        return _copy(state)

    # ------------------------------------------------------------------ network check

    def netcheck(self, hosts: list[str], *, timeout: float = 3.0) -> list[dict]:
        """TCP-connect check of ``host[:port]`` (port 443 by default), cached 10 minutes per host."""
        out = []
        now = time.time()
        for host in hosts:
            cached = self._netcheck.get(host)
            if cached is not None and now - cached[0] < NETCHECK_TTL_S:
                out.append(cached[1])
                continue
            name, _, port = host.partition(":")
            t0 = time.perf_counter()
            try:
                with socket.create_connection((name, int(port or 443)), timeout=timeout):
                    pass
                res = {"host": host, "ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": None}
            except (OSError, ValueError) as exc:
                res = {"host": host, "ok": False, "ms": None, "error": f"{type(exc).__name__}: {exc}"[:200]}
            self._netcheck[host] = (now, res)
            out.append(res)
        return out


_CATALOG: list | None = None


def _output_id(relpath: str) -> str | None:
    """The catalog output (``sparc.core.catalog.OUTPUTS``) whose file globs match ``relpath``, if any."""
    global _CATALOG
    if _CATALOG is None:
        try:
            import importlib

            _CATALOG = list(getattr(importlib.import_module("sparc.core.catalog"), "OUTPUTS", []) or [])
        except Exception:
            _CATALOG = []
    import fnmatch

    for spec in _CATALOG:
        root = getattr(spec, "root", "run")
        if root != "run":
            continue
        for pattern in getattr(spec, "files", ()) or ():
            if fnmatch.fnmatch(relpath, pattern):
                return getattr(spec, "id", None)
    return None


def _copy(state: dict) -> dict:
    import copy

    return copy.deepcopy(state)


def _high_memory(rss_mb: float) -> bool:
    try:
        import psutil

        return rss_mb > OOM_FRACTION * psutil.virtual_memory().total / 2 ** 20
    except Exception:
        return False
