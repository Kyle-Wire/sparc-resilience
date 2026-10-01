"""SQLite index of the workspace (SPEC §10.6): schema, migrations, one writer, many readers, reindex.

``studio.sqlite`` runs in WAL mode with ``synchronous=NORMAL``.  The server
process is the only writer (job workers never write SQLite): one writer
connection guarded by a lock, used from coroutines through
:func:`asyncio.to_thread` (``await db.aexecute(...)``) and directly from
worker threads (``db.execute(...)``).  Readers get one autocommit connection
per thread, so reads never wait for a write.

Schema versions are tracked with ``PRAGMA user_version``; :data:`MIGRATIONS`
maps each version to the SQL that reaches it.  Version 1 is the full DDL of
SPEC §10.6 - every table of every work item; feature items only use them.

The tables are an index: ``sparc studio --reindex`` (:func:`reindex`) rebuilds
the job tables from the job directories and lets feature modules rebuild
theirs through :func:`reindex_hook`.  Only ``settings``, ``config_versions``,
``regions`` and ``blobs`` cannot be rebuilt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

log = logging.getLogger("sparc.studio.db")

__all__ = ["SCHEMA_V1", "MIGRATIONS", "SCHEMA_VERSION", "REBUILDABLE_TABLES", "Database", "dumps", "loads",
           "reindex_hook", "reindex"]

SCHEMA_V1 = """
CREATE TABLE projects (id TEXT PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, dir TEXT NOT NULL,
  config_path TEXT NOT NULL, template TEXT, demo INTEGER DEFAULT 0, active_run_id TEXT, archived INTEGER DEFAULT 0,
  meta_json TEXT DEFAULT '{}', created_utc TEXT, updated_utc TEXT);
CREATE TABLE config_versions (project_id TEXT, version INTEGER, yaml TEXT NOT NULL, note TEXT, saved_utc TEXT,
  sections_json TEXT, PRIMARY KEY (project_id, version));
CREATE TABLE runs (id TEXT PRIMARY KEY, project_id TEXT, run_dir TEXT UNIQUE NOT NULL, studio_dir TEXT NOT NULL,
  origin TEXT NOT NULL, parent_run_id TEXT, study_id TEXT, label TEXT, mode TEXT, coarse_m REAL, stages_json TEXT,
  status TEXT NOT NULL, created_utc TEXT, finished_utc TEXT, n_points INTEGER, r2 REAL, rmse REAL, coverage REAL,
  checkpoint_bytes INTEGER, checkpoint_done_json TEXT, fingerprint TEXT, code_sha TEXT, config_sha TEXT, input_sha TEXT,
  git_commit TEXT, git_dirty INTEGER, has_emulator INTEGER DEFAULT 0, manifest_mtime REAL, last_job_id TEXT,
  pinned INTEGER DEFAULT 0, notes TEXT, demo INTEGER DEFAULT 0);
CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, lane TEXT NOT NULL, executor TEXT NOT NULL, label TEXT,
  project_id TEXT, run_id TEXT, study_id TEXT, scenario_id TEXT, parent_job_id TEXT, after_job_id TEXT,
  params_json TEXT NOT NULL, priority INTEGER DEFAULT 0, status TEXT NOT NULL, blocked_json TEXT,
  pid INTEGER, pgid INTEGER, proc_create_time REAL, job_dir TEXT NOT NULL, threads INTEGER,
  created_utc TEXT, started_utc TEXT, finished_utc TEXT, exit_code INTEGER, error_json TEXT, result_json TEXT,
  progress REAL, eta_s REAL, eta_lo REAL, eta_hi REAL, current_path TEXT, stage TEXT, last_cursor INTEGER DEFAULT 0,
  peak_rss_mb REAL, host_id TEXT);
CREATE INDEX jobs_status ON jobs(status); CREATE INDEX jobs_run ON jobs(run_id);
CREATE TABLE spans (job_id TEXT, span_id TEXT, parent_id TEXT, kind TEXT, name TEXT, key TEXT, k INTEGER, n INTEGER,
  unit TEXT, status TEXT, started_ts REAL, ended_ts REAL, elapsed_s REAL, ctx_json TEXT, metrics_json TEXT,
  PRIMARY KEY (job_id, span_id));
CREATE TABLE metrics (job_id TEXT, cursor INTEGER, span_id TEXT, name TEXT, value REAL, value_text TEXT, unit TEXT,
  tags_json TEXT, ts REAL);
CREATE INDEX metrics_job_name ON metrics(job_id, name);
CREATE TABLE artifacts (job_id TEXT, run_id TEXT, relpath TEXT, role TEXT, stage TEXT, bytes INTEGER, ts REAL);
CREATE TABLE warnings (job_id TEXT, code TEXT, msg_hash TEXT, lvl TEXT, message TEXT, data_json TEXT, stage TEXT,
  first_cursor INTEGER, count INTEGER DEFAULT 1, PRIMARY KEY (job_id, code, msg_hash));
CREATE TABLE checkpoints (job_id TEXT, run_id TEXT, action TEXT, done_json TEXT, bytes INTEGER, ts REAL);
CREATE TABLE unit_timings (host_id TEXT, unit TEXT, seconds REAL, n_cells INTEGER, threads INTEGER, mode TEXT,
  job_id TEXT, ts REAL);
CREATE INDEX unit_timings_hu ON unit_timings(host_id, unit);
CREATE TABLE stage_timings (run_id TEXT, stage TEXT, seconds REAL, n_points INTEGER, mode TEXT, threads INTEGER,
  host_id TEXT, git_commit TEXT, source TEXT, PRIMARY KEY (run_id, stage));
CREATE TABLE resource_samples (job_id TEXT, ts REAL, rss_mb REAL, cpu_pct REAL, n_procs INTEGER, threads INTEGER);
CREATE TABLE run_locks (run_id TEXT PRIMARY KEY, job_id TEXT, acquired_utc TEXT);
CREATE TABLE studies (id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, target_run_id TEXT, job_id TEXT, out_dir TEXT,
  status TEXT, params_json TEXT, summary_json TEXT, run_fingerprint TEXT, origin TEXT DEFAULT 'studio',
  created_utc TEXT, updated_utc TEXT);
CREATE TABLE study_links (run_id TEXT, study_id TEXT, attached INTEGER DEFAULT 1, PRIMARY KEY (run_id, study_id));
CREATE TABLE scenarios (id TEXT PRIMARY KEY, project_id TEXT, revision INTEGER, parent_id TEXT, name TEXT,
  doc_json TEXT NOT NULL, content_hash TEXT NOT NULL, tags_json TEXT DEFAULT '[]', status TEXT, anchor_run_id TEXT,
  archived INTEGER DEFAULT 0, created_utc TEXT, updated_utc TEXT);
CREATE TABLE results (id TEXT PRIMARY KEY, scenario_id TEXT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
  content_hash TEXT, code_sha TEXT, ckpt_key TEXT, dir TEXT, summary_json TEXT, has_folds INTEGER, stale INTEGER DEFAULT 0,
  job_id TEXT, created_utc TEXT);
CREATE INDEX results_lookup ON results(run_id, content_hash, ckpt_key, code_sha);
CREATE TABLE plans (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, name TEXT, params_json TEXT, summary_json TEXT,
  dir TEXT, verified_result_id TEXT, created_utc TEXT);
CREATE TABLE sweeps (id TEXT PRIMARY KEY, run_id TEXT, params_json TEXT, dir TEXT, job_id TEXT, summary_json TEXT, created_utc TEXT);
CREATE TABLE comparisons (id TEXT PRIMARY KEY, run_id TEXT, items_json TEXT, dir TEXT, summary_json TEXT, created_utc TEXT);
CREATE TABLE regions (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, name TEXT, spec_json TEXT, n_cells INTEGER, created_utc TEXT);
CREATE TABLE blobs (id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, path TEXT, bytes INTEGER, created_utc TEXT);
CREATE TABLE exports (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, kind TEXT, ref TEXT, options_json TEXT,
  job_id TEXT, path TEXT, bytes INTEGER, status TEXT, created_utc TEXT);
CREATE TABLE findings (id TEXT PRIMARY KEY, project_id TEXT, run_id TEXT, view TEXT, url_state TEXT, title TEXT,
  note_md TEXT, snapshot_json TEXT, image_path TEXT, position REAL, created_utc TEXT, updated_utc TEXT);
CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT);
"""

#: version → SQL that upgrades the previous version to it
MIGRATIONS: dict[int, str] = {1: SCHEMA_V1}
SCHEMA_VERSION = max(MIGRATIONS)

#: tables ``--reindex`` empties and rebuilds from disk (SPEC §10.6)
REBUILDABLE_TABLES = ("runs", "studies", "study_links", "stage_timings", "artifacts", "checkpoints", "jobs", "spans",
                      "metrics", "warnings", "resource_samples", "unit_timings", "run_locks", "results", "plans",
                      "sweeps", "comparisons", "scenarios", "findings", "exports")


def dumps(obj: Any) -> str | None:
    """JSON text for a ``*_json`` column (NaN/Inf → null); None stays NULL."""
    if obj is None:
        return None
    return json.dumps(_finite(obj), separators=(",", ":"), allow_nan=False, default=str)


def loads(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


def _finite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _row_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {d[0]: v for d, v in zip(cursor.description, row)}


class Database:
    """The workspace database: one writer (thread-safe), one reader connection per thread.

    Reads: :meth:`fetchone`, :meth:`fetchall`, :meth:`fetchval` (rows are
    dicts).  Writes from threads: :meth:`execute`, :meth:`executemany`,
    :meth:`transaction`; from coroutines: :meth:`aexecute`,
    :meth:`aexecutemany`, :meth:`atransaction` (they run the same calls in a
    worker thread so the event loop never waits on SQLite).
    """

    def __init__(self, path: str | Path, *, timeout: float = 30.0):
        self.path = Path(path)
        self.timeout = timeout
        self._wlock = threading.RLock()
        self._writer: sqlite3.Connection | None = None
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return f"Database({str(self.path)!r})"

    # -- connections --------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("database is closed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=self.timeout, isolation_level=None, check_same_thread=False)
        conn.row_factory = _row_factory
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    def _reader(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _writer_conn(self) -> sqlite3.Connection:
        if self._writer is None:
            self._writer = self._connect()
        return self._writer

    def reopen(self) -> None:
        """Allow new connections again after :meth:`close` (an app whose lifespan runs a second time)."""
        self._closed = False
        self._local = threading.local()

    def close(self) -> None:
        with self._wlock:
            self._closed = True
            with self._conns_lock:
                conns, self._conns = self._conns, []
            for c in conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._writer = None

    # -- schema ---------------------------------------------------------------------------------

    def user_version(self) -> int:
        return int(self.fetchval("PRAGMA user_version", default=0) or 0)

    def migrate(self) -> int:
        """Apply every migration above ``PRAGMA user_version``; returns the resulting version."""
        with self._wlock:
            conn = self._writer_conn()
            current = int(conn.execute("PRAGMA user_version").fetchone()["user_version"] or 0)
            for version in sorted(v for v in MIGRATIONS if v > current):
                log.info("migrating %s to schema v%d", self.path, version)
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in _statements(MIGRATIONS[version]):
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version={int(version)}")
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                current = version
            return current

    def tables(self) -> list[str]:
        return [r["name"] for r in self.fetchall("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    # -- reads ----------------------------------------------------------------------------------

    def fetchall(self, sql: str, params: Sequence | dict = ()) -> list[dict]:
        return self._reader().execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: Sequence | dict = ()) -> dict | None:
        return self._reader().execute(sql, params).fetchone()

    def fetchval(self, sql: str, params: Sequence | dict = (), default: Any = None) -> Any:
        row = self.fetchone(sql, params)
        if not row:
            return default
        return next(iter(row.values()))

    # -- writes (threads) -------------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``with db.transaction() as conn:`` - the writer connection inside ``BEGIN IMMEDIATE`` … ``COMMIT``."""
        with self._wlock:
            conn = self._writer_conn()
            if conn.in_transaction:            # nested use from the same thread: join the outer transaction
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence | dict = ()) -> int:
        """Run one write statement; returns the affected row count."""
        with self.transaction() as conn:
            return conn.execute(sql, params).rowcount

    def executemany(self, sql: str, rows: Iterable[Sequence | dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.transaction() as conn:
            return conn.executemany(sql, rows).rowcount

    def run(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """``fn(conn)`` inside one write transaction; returns its result."""
        with self.transaction() as conn:
            return fn(conn)

    # -- writes (coroutines) ----------------------------------------------------------------------

    async def aexecute(self, sql: str, params: Sequence | dict = ()) -> int:
        return await asyncio.to_thread(self.execute, sql, params)

    async def aexecutemany(self, sql: str, rows: Iterable[Sequence | dict]) -> int:
        return await asyncio.to_thread(self.executemany, sql, list(rows))

    async def atransaction(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        return await asyncio.to_thread(self.run, fn)

    # -- small DAO helpers ------------------------------------------------------------------------

    def insert(self, table: str, row: dict, *, replace: bool = False) -> None:
        cols = list(row)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self.execute(f"{verb} INTO {_ident(table)} ({', '.join(map(_ident, cols))}) "
                     f"VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])

    def update(self, table: str, key: dict, values: dict) -> int:
        if not values:
            return 0
        sets = ", ".join(f"{_ident(c)} = ?" for c in values)
        where = " AND ".join(f"{_ident(c)} = ?" for c in key)
        return self.execute(f"UPDATE {_ident(table)} SET {sets} WHERE {where}", [*values.values(), *key.values()])

    async def ainsert(self, table: str, row: dict, *, replace: bool = False) -> None:
        await asyncio.to_thread(self.insert, table, row, replace=replace)

    async def aupdate(self, table: str, key: dict, values: dict) -> int:
        return await asyncio.to_thread(self.update, table, key, values)

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value_json FROM settings WHERE key = ?", (key,))
        return default if row is None else loads(row["value_json"], default)

    def set_setting(self, key: str, value: Any) -> None:
        self.execute("INSERT OR REPLACE INTO settings (key, value_json) VALUES (?, ?)", (key, dumps(value)))


def _ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"bad SQL identifier {name!r}")
    return name


def _statements(script: str) -> list[str]:
    """Split a DDL script into statements (the DDL has no string literals containing ';')."""
    return [s.strip() for s in script.split(";") if s.strip()]


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------

_REINDEX_HOOKS: list[tuple[int, str, Callable]] = []


def reindex_hook(name: str, *, order: int = 100) -> Callable[[Callable], Callable]:
    """Register ``fn(db, workspace) -> dict`` to rebuild a feature's tables during ``--reindex``.

    Feature modules register at import time (their route or kind modules are
    imported by :func:`reindex`).  Hooks run in ``order`` after the job
    tables are rebuilt; each returns a small summary dict (counts).  Lower
    ``order`` runs first: the runs registry (``order=10``) before studies,
    scenarios and findings.
    """
    def deco(fn: Callable) -> Callable:
        _REINDEX_HOOKS[:] = [h for h in _REINDEX_HOOKS if h[1] != name]
        _REINDEX_HOOKS.append((order, name, fn))
        _REINDEX_HOOKS.sort(key=lambda h: (h[0], h[1]))
        return fn
    return deco


def reindex(db: Database, workspace, *, import_features: bool = True) -> dict:
    """Empty the rebuildable tables and rebuild them from disk (``sparc studio --reindex``).

    Jobs (and their spans, metrics, warnings, artifacts, checkpoints and unit
    timings) come from ``jobs/*/job.json`` + ``state.json`` + ``result.json``
    + ``events.jsonl``; every other table is rebuilt by the registered
    :func:`reindex_hook` functions.  Returns ``{table_or_hook: count}``.
    """
    from sparc.studio.jobs import kinds as _kinds
    from sparc.studio.jobs.tailer import rebuild_job_rows

    if import_features:
        _import_feature_modules()
        _kinds.load_kind_modules()
    db.migrate()
    with db.transaction() as conn:
        for table in REBUILDABLE_TABLES:
            conn.execute(f"DELETE FROM {_ident(table)}")
    summary: dict[str, Any] = {"jobs": 0}
    jobs_dir = Path(workspace.jobs_dir)
    if jobs_dir.is_dir():
        for job_dir in sorted(p for p in jobs_dir.iterdir() if (p / "job.json").is_file()):
            try:
                if rebuild_job_rows(db, job_dir):
                    summary["jobs"] += 1
            except Exception:                # one bad job dir never stops the rebuild
                log.exception("reindex: skipping %s", job_dir)
    for _order, name, fn in list(_REINDEX_HOOKS):
        try:
            summary[name] = fn(db, workspace)
        except Exception as exc:
            log.exception("reindex hook %s failed", name)
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return summary


def _import_feature_modules() -> None:
    import importlib

    from sparc.studio.routes import ROUTER_MODULES

    for name in ROUTER_MODULES:
        mod = f"sparc.studio.routes.{name}"
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            if exc.name != mod:
                raise
