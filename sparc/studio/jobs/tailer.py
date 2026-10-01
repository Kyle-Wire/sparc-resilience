"""JobTailer: one asyncio task per live job over its ``events.jsonl`` (SPEC §5.6).

The cursor of an event is the byte offset of its line's first byte: monotonic,
unique, coordination-free, and the SSE ``id``.  The tailer polls the file
size every 250 ms and parses only complete (``\\n``-terminated) lines.  Each
line is validated (:func:`~sparc.studio.events.validate_event`), folded into
the projection (:mod:`.tracker`), published to the job's SSE subscribers and
passed to the kind's server-side ``on_event`` hook.  Every 2 s and at the end
the projection is flushed to SQLite: the ``jobs`` row (progress, ETA,
current path, stage, ``last_cursor``) plus the ``spans``, ``metrics``,
``artifacts``, ``warnings``, ``checkpoints``, ``unit_timings`` and
``stage_timings`` tables.  Append-only rows are written once: after a
restart the tailer re-reads the file from byte 0 to rebuild the projection
but inserts rows only for events past the flushed ``last_cursor``.

Past 200 MB of events, debug-level lines stay on disk but no longer reach
SQLite (the worker's emitter drops to info on its own, SPEC §5.6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from sparc.studio import db as dbmod
from sparc.studio.events import SERVER_EVENT_TYPES, validate_event
from sparc.studio.jobs import tracker
from sparc.studio.jobs.eta import CostModel, host_id, projection_eta

log = logging.getLogger("sparc.studio.jobs")

__all__ = ["JobTailer", "RowCollector", "read_events", "iter_lines", "rebuild_job_rows", "LOG_CAP_BYTES"]

LOG_CAP_BYTES = 200 * 1024 ** 2
READ_CHUNK = 4 * 1024 ** 2
BATCH_LINES = 20_000


def iter_lines(path: str | os.PathLike, start: int = 0, *, max_bytes: int | None = None):
    """Yield ``(cursor, raw_line_bytes)`` for complete lines from byte ``start`` (stops at a partial line)."""
    try:
        f = open(path, "rb")
    except OSError:
        return
    with f:
        f.seek(start)
        offset = start
        buf = b""
        read = 0
        while True:
            chunk = f.read(READ_CHUNK)
            if not chunk:
                break
            read += len(chunk)
            buf += chunk
            pos = 0
            while True:
                nl = buf.find(b"\n", pos)
                if nl < 0:
                    break
                yield offset + pos, buf[pos:nl]
                pos = nl + 1
            offset += pos
            buf = buf[pos:]
            if max_bytes is not None and read >= max_bytes:
                break


def parse_line(raw: bytes) -> dict:
    """A parsed and validated event; undecodable lines become ``log`` events."""
    try:
        ev = json.loads(raw)
    except ValueError:
        ev = {"type": "log", "v": 1, "ts": time.time(), "lvl": "warning", "path": [], "ctx": {},
              "logger": "studio.event", "level": "WARNING",
              "msg": "undecodable event line: " + raw[:200].decode("utf-8", "replace")}
    return validate_event(ev)


def read_events(path: str | os.PathLike, after: int | None = None, *, limit: int | None = None,
                types: set[str] | None = None, min_lvl: str | None = None) -> tuple[list[dict], int, bool]:
    """Events with ``cursor > after`` (all when ``after`` is None), each with its ``cursor``.

    Returns ``(events, next_cursor, eof)``: ``next_cursor`` is the cursor of
    the last event examined (or ``after``/-1 when none), ``eof`` whether the
    end of the file was reached.
    """
    from sparc.core.progress import LEVELS

    floor = LEVELS.get(min_lvl, 0) if min_lvl else 0
    out: list[dict] = []
    last = -1 if after is None else int(after)
    start = 0 if after is None or after < 0 else int(after)
    eof = True
    for cursor, raw in iter_lines(path, start):
        if after is not None and cursor <= after:
            continue
        last = cursor
        ev = parse_line(raw)
        if types and ev.get("type") not in types:
            continue
        if floor and LEVELS.get(ev.get("lvl", "info"), 20) < floor:
            continue
        ev = dict(ev)
        ev["cursor"] = cursor
        out.append(ev)
        if limit is not None and len(out) >= limit:
            eof = False
            break
    return out, last, eof


class RowCollector:
    """SQLite rows produced by events (shared by the live tailer and ``--reindex``)."""

    def __init__(self, job_id: str, *, run_id: str | None = None, threads: int | None = None,
                 mode: str | None = None, host: str | None = None, git_commit: str | None = None):
        self.job_id = job_id
        self.run_id = run_id
        self.threads = threads
        self.mode = mode
        self.host = host or host_id()
        self.git_commit = git_commit
        self.metrics: list[tuple] = []
        self.artifacts: list[tuple] = []
        self.checkpoints: list[tuple] = []
        self.unit_timings: list[tuple] = []
        self.stage_timings: list[tuple] = []
        self.dirty_spans: set[str] = set()
        self.capped = False

    def add(self, cursor: int, ev: dict, state: dict) -> None:
        t = ev.get("type")
        sid = ev.get("span")
        if t in ("run.start", "run.end", "stage.start", "stage.end", "task.start", "task.end") and sid:
            self.dirty_spans.add(sid)
        if self.capped and ev.get("lvl") == "debug":
            return
        ts = ev.get("ts")
        nested = tracker.is_nested(ev)
        if t == "metric":
            value = ev.get("value")
            num = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            text = None if num is not None or value is None else (
                ("true" if value else "false") if isinstance(value, bool) else str(value))
            self.metrics.append((self.job_id, cursor, sid, ev.get("name"), num, text, ev.get("unit"),
                                 dbmod.dumps(ev.get("tags") or {}), ts))
        if nested:
            return
        if t == "artifact" and isinstance(ev.get("path"), str):
            self.artifacts.append((self.job_id, self.run_id, ev["path"], ev.get("role"),
                                   ev.get("stage") or tracker.stage_of(ev.get("span_path") or []),
                                   int(ev.get("bytes") or 0), ts))
        elif t == "checkpoint":
            self.checkpoints.append((self.job_id, self.run_id, ev.get("action"), dbmod.dumps(ev.get("done") or []),
                                     ev.get("bytes"), ts))
        comp = tracker.unit_completion(ev)
        if comp is not None and comp[1] is not None and comp[1] > 0:
            self.unit_timings.append((self.host, comp[0], comp[1], state.get("n_points"), self.threads, self.mode,
                                      self.job_id, ts))
        if t == "stage.end" and ev.get("status") == "ok" and self.run_id and ev.get("elapsed_s") is not None:
            self.stage_timings.append((self.run_id, ev.get("stage"), float(ev["elapsed_s"]), state.get("n_points"),
                                       self.mode, self.threads, self.host, self.git_commit, "events"))

    def write(self, conn, state: dict, *, job_values: dict | None = None) -> None:
        """Insert the collected rows and upsert spans/warnings (inside the caller's transaction)."""
        jid = self.job_id
        if self.metrics:
            conn.executemany("INSERT INTO metrics (job_id, cursor, span_id, name, value, value_text, unit, tags_json, "
                             "ts) VALUES (?,?,?,?,?,?,?,?,?)", self.metrics)
        if self.artifacts:
            conn.executemany("INSERT INTO artifacts (job_id, run_id, relpath, role, stage, bytes, ts) "
                             "VALUES (?,?,?,?,?,?,?)", self.artifacts)
        if self.checkpoints:
            conn.executemany("INSERT INTO checkpoints (job_id, run_id, action, done_json, bytes, ts) "
                             "VALUES (?,?,?,?,?,?)", self.checkpoints)
        if self.unit_timings:
            conn.executemany("INSERT INTO unit_timings (host_id, unit, seconds, n_cells, threads, mode, job_id, ts) "
                             "VALUES (?,?,?,?,?,?,?,?)", self.unit_timings)
        if self.stage_timings:
            conn.executemany("INSERT OR REPLACE INTO stage_timings (run_id, stage, seconds, n_points, mode, threads, "
                             "host_id, git_commit, source) VALUES (?,?,?,?,?,?,?,?,?)", self.stage_timings)
        spans = state["spans"]
        rows = []
        for sid in self.dirty_spans:
            sp = spans.get(sid)
            if sp is None:
                continue
            rows.append((jid, sid, sp.get("parent_id"), sp.get("kind"), sp.get("name"), sp.get("key"), sp.get("k"),
                         sp.get("n"), sp.get("unit"), sp.get("status"), sp.get("started_ts"), sp.get("ended_ts"),
                         sp.get("elapsed_s"), dbmod.dumps(sp.get("ctx") or {}), dbmod.dumps(sp.get("metrics") or {})))
        if rows:
            conn.executemany("INSERT OR REPLACE INTO spans (job_id, span_id, parent_id, kind, name, key, k, n, unit, "
                             "status, started_ts, ended_ts, elapsed_s, ctx_json, metrics_json) "
                             "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        wrows = [(jid, w["code"], w["msg_hash"], w["lvl"], w["message"], dbmod.dumps(w.get("data") or {}),
                  w.get("stage"), w["first_cursor"], w["count"]) for w in state["warnings"].values()]
        if wrows:
            conn.executemany("INSERT INTO warnings (job_id, code, msg_hash, lvl, message, data_json, stage, "
                             "first_cursor, count) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(job_id, code, msg_hash) "
                             "DO UPDATE SET count = excluded.count", wrows)
        if job_values:
            sets = ", ".join(f"{k} = ?" for k in job_values)
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*job_values.values(), jid])
        self.metrics, self.artifacts, self.checkpoints = [], [], []
        self.unit_timings, self.stage_timings = [], []
        self.dirty_spans = set()


class JobTailer:
    """Tails one job's ``events.jsonl`` (see the module docstring).

    ``on_event(job_id, cursor, ev)`` is awaited for each new event (the
    manager uses it for the starting → running transition and the kind
    hooks); ``on_progress(job_id, payload)`` gets the throttled
    ``job.progress`` payload.
    """

    def __init__(self, *, db, hub, job_id: str, events_path: str | os.PathLike, run_id: str | None = None,
                 threads: int | None = None, mode: str | None = None, flushed_cursor: int = -1,
                 interval: float = 0.25, flush_interval: float = 2.0, cost_model: CostModel | None = None,
                 on_event: Callable[..., Any] | None = None, on_progress: Callable[..., Any] | None = None):
        self.db = db
        self.hub = hub
        self.job_id = job_id
        self.path = Path(events_path)
        self.run_id = run_id
        self.threads = threads
        self.interval = interval
        self.flush_interval = flush_interval
        self.flushed_cursor = flushed_cursor
        self.model = cost_model or CostModel()
        self.on_event = on_event
        self.on_progress = on_progress
        self.kind: str | None = None
        self.state = tracker.new_state()
        self.offset = 0
        self.rows = RowCollector(job_id, run_id=run_id, threads=threads, mode=mode)
        self.eta: dict = {"eta_s": None, "eta_lo": None, "eta_hi": None, "stages": {}}
        self.worker_seen = False
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_flush = 0.0
        self._last_progress = 0.0
        self._last_payload: tuple | None = None
        self._lock = asyncio.Lock()
        self.n_lines = 0

    # -- lifecycle --------------------------------------------------------------------------------

    def start(self) -> "JobTailer":
        self._task = asyncio.create_task(self._run(), name=f"tailer-{self.job_id}")
        return self

    async def stop(self, *, drain: bool = True) -> None:
        """Stop polling; with ``drain`` read to the end of the file and flush first."""
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("tailer %s failed", self.job_id)
            self._task = None
        if drain:
            await self.poll()
            await self.flush(force=True)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.poll()
                if time.monotonic() - self._last_flush >= self.flush_interval:
                    await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("tailer %s poll failed", self.job_id)
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
            except asyncio.TimeoutError:
                pass

    # -- reading ----------------------------------------------------------------------------------

    async def poll(self) -> int:
        """Process every complete line past the current offset; returns the number of events.

        Reading, parsing and reducing run in a worker thread in batches of
        :data:`BATCH_LINES` (so a large backlog never stalls the event loop);
        publishing and hooks run on the loop, in cursor order.  Readers of
        the projection take :attr:`lock`.
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return 0
        if size <= self.offset:
            return 0
        total = 0
        async with self._lock:
            while True:
                n_read, batch = await asyncio.to_thread(self._read_batch)
                if not n_read:
                    break
                total += n_read
                if self.hub is not None and self.hub.has_job_subscribers(self.job_id):
                    for cursor, ev in batch:
                        self.hub.publish_job(self.job_id, ("event", cursor, ev))
                if self.on_event is not None:
                    for cursor, ev in batch:
                        try:
                            res = self.on_event(self.job_id, cursor, ev)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception:
                            log.exception("on_event for %s failed", self.job_id)
                if n_read < BATCH_LINES:
                    break
            if total:
                self._progress()
        return total

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _read_batch(self) -> tuple[int, list[tuple[int, dict]]]:
        """Reduce up to :data:`BATCH_LINES` lines: ``(lines read, events past the flushed cursor)``."""
        out: list[tuple[int, dict]] = []
        n = 0
        for cursor, raw in iter_lines(self.path, self.offset):
            self.offset = cursor + len(raw) + 1
            ev = parse_line(raw)
            tracker.reduce(self.state, ev, cursor)
            n += 1
            self.n_lines += 1
            if not self.rows.capped and self.offset > LOG_CAP_BYTES:
                self.rows.capped = True
                self.state["log_capped"] = True
            if not self.worker_seen and ev.get("type") not in SERVER_EVENT_TYPES:
                self.worker_seen = True
            if cursor > self.flushed_cursor:
                self.rows.add(cursor, ev, self.state)
                out.append((cursor, ev))
            if n >= BATCH_LINES:
                break
        return n, out

    def _progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < 1.0:
            return
        self.eta = projection_eta(self.state, self.model, n_cells=self.state.get("n_points"), threads=self.threads)
        tail = (self.state.get("current_path") or [])[-4:]
        payload = (self.state.get("progress"), self.eta.get("eta_s"), self.eta.get("eta_lo"), self.eta.get("eta_hi"),
                   self.state.get("stage"), tuple(tail))
        if payload == self._last_payload and not force:
            return
        self._last_progress = now
        self._last_payload = payload
        if self.on_progress is not None:
            self.on_progress(self.job_id, {"job_id": self.job_id, "frac": payload[0], "eta_s": payload[1],
                                           "eta_lo": payload[2], "eta_hi": payload[3], "stage": payload[4],
                                           "path_tail": list(tail)})

    # -- SQLite -----------------------------------------------------------------------------------

    def job_values(self) -> dict:
        return {"progress": self.state.get("progress"), "eta_s": self.eta.get("eta_s"),
                "eta_lo": self.eta.get("eta_lo"), "eta_hi": self.eta.get("eta_hi"),
                "current_path": dbmod.dumps(self.state.get("current_path")), "stage": self.state.get("stage"),
                "last_cursor": self.state["cursor"]}

    async def flush(self, *, force: bool = False) -> None:
        """Write the projection and the collected rows to SQLite (one transaction)."""
        self._last_flush = time.monotonic()
        async with self._lock:
            if force:
                self._progress(force=True)
            values = self.job_values()
            state = self.state
            rows = self.rows

            def _write(conn):
                rows.write(conn, state, job_values=values)

            try:
                await self.db.atransaction(_write)
                self.flushed_cursor = max(self.flushed_cursor, state["cursor"])
            except Exception:
                log.exception("flush of %s failed", self.job_id)


def rebuild_job_rows(db, job_dir: str | os.PathLike) -> bool:
    """Rebuild one job's rows from its directory (``--reindex``); returns True when a row was written."""
    from sparc.studio.workspace import read_json

    job_dir = Path(job_dir)
    job = read_json(job_dir / "job.json")
    if not isinstance(job, dict) or not job.get("id"):
        return False
    state_json = read_json(job_dir / "state.json", {}) or {}
    result = read_json(job_dir / "result.json")
    events = job_dir / "events.jsonl"
    st = tracker.new_state()
    rows = RowCollector(job["id"], run_id=job.get("run_id"), threads=job.get("threads"), mode=job.get("kind"))
    for cursor, raw in iter_lines(events, 0):
        ev = parse_line(raw)
        tracker.reduce(st, ev, cursor)
        rows.add(cursor, ev, st)
    rows.dirty_spans = set(st["spans"])
    status = (result or {}).get("status") or st.get("status") or state_json.get("status") or "interrupted"
    if status in ("queued", "blocked", "starting", "running", "cancelling"):
        status = "interrupted"                 # nothing is live during a reindex
    eta = projection_eta(st)
    row = {
        "id": job["id"], "kind": job.get("kind", "?"), "lane": job.get("lane", "none"),
        "executor": job.get("executor", "process"), "label": job.get("label") or job.get("kind"),
        "project_id": job.get("project_id"), "run_id": job.get("run_id"), "study_id": job.get("study_id"),
        "scenario_id": job.get("scenario_id"), "parent_job_id": job.get("parent_job_id"),
        "after_job_id": job.get("after_job_id"), "params_json": dbmod.dumps(job.get("params") or {}),
        "priority": int(job.get("priority") or 0), "status": status, "pid": state_json.get("pid"),
        "pgid": state_json.get("pgid"), "proc_create_time": state_json.get("proc_create_time"),
        "job_dir": str(job_dir), "threads": job.get("threads"), "created_utc": job.get("created_utc"),
        "started_utc": state_json.get("started_utc"), "finished_utc": (result or {}).get("finished_utc"),
        "exit_code": (result or {}).get("exit_code", st.get("exit_code")),
        "error_json": dbmod.dumps((result or {}).get("error") or st.get("error")),
        "result_json": dbmod.dumps((result or {}).get("result") or st.get("result")),
        "progress": st.get("progress"), "eta_s": eta.get("eta_s"), "eta_lo": eta.get("eta_lo"),
        "eta_hi": eta.get("eta_hi"), "current_path": dbmod.dumps(st.get("current_path")), "stage": st.get("stage"),
        "last_cursor": st["cursor"], "host_id": host_id(),
    }

    def _write(conn):
        cols = list(row)
        conn.execute(f"INSERT OR REPLACE INTO jobs ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                     [row[c] for c in cols])
        rows.write(conn, st)

    db.run(_write)
    return True
