"""Job-kind registry (SPEC §10.2–10.3).

Feature items register kinds in their own modules::

    @job_kind("post.planner", lane="medium", executor="process", label="Planner pack", params=PlannerParams,
              needs_run=True, locks_run=True, network_hosts=("noaa-ghcn-pds.s3.amazonaws.com",), long=True,
              estimate=planner_estimate, on_finish=record_planner)
    def run_planner(ctx: JobContext, params: PlannerParams) -> dict: ...

The server imports every module in :data:`KIND_MODULES` to list kinds and
their params schemas, skipping modules that do not exist yet (a
``ModuleNotFoundError`` naming exactly that module); any other import error
is raised.  Kind modules therefore import heavy libraries (anything that may
pull torch) inside the job function, never at module level.

**Hooks** run in the server process (workers never write SQLite):

* ``on_event(sctx, job, event)`` for the event types in ``on_event_types``
  (default ``run.dir``, ``run.start``, ``artifact``);
* ``on_finish(sctx, job, result)`` after the final status;
* ``estimate(sctx, job, params) -> {units?, n_cells?, est_s?, est_lo?, est_hi?, peak_ram_gb?, disk_bytes?}``;
* ``preflight(sctx, job, params) -> [{reason, actions?, fatal?}]`` (extra checks before start);
* ``retry_params(sctx, job) -> params`` (``POST /jobs/{jid}/retry``; default: the same params);
* ``threads(settings, params) -> int`` (default: by lane, SPEC §10.5).

``sctx`` is the server's :class:`~sparc.studio.app.StudioContext` (``db``,
``hub``, ``workspace``, ``jobs``, ``services``).  Hooks run in a worker
thread, must be fast (< 50 ms) and idempotent: reattach may replay them.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

log = logging.getLogger("sparc.studio.jobs")

__all__ = ["LANES", "EXECUTORS", "KIND_MODULES", "TEST_KIND_MODULE", "JobKind", "KINDS", "job_kind",
           "load_kind_modules", "get_kind", "kinds_listing", "JobContext", "ReadOnlyDB"]

LANES = ("heavy", "medium", "network", "engine", "none")
EXECUTORS = ("process", "engine", "external")
KIND_MODULES = ["sparc.studio.projects.kinds", "sparc.studio.runs.kinds", "sparc.studio.engine.kinds",
                "sparc.studio.studies.kinds", "sparc.studio.exports.kinds"]
TEST_KIND_MODULE = "sparc.studio.jobs.testkinds"
DEFAULT_HOOK_TYPES = ("run.dir", "run.start", "artifact")


@dataclass
class JobKind:
    kind: str
    fn: Callable | None
    lane: str
    executor: str = "process"
    label: str = ""
    params: type[BaseModel] | None = None
    needs_run: bool = False
    needs_checkpoint: bool = False
    locks_run: bool = False
    network_hosts: tuple[str, ...] = ()
    long: bool = False
    estimate: Callable | None = None
    on_event: Callable | None = None
    on_event_types: tuple[str, ...] = DEFAULT_HOOK_TYPES
    on_finish: Callable | None = None
    preflight: Callable | None = None
    retry_params: Callable | None = None
    threads: Callable | None = None
    module: str = ""
    extra: dict = field(default_factory=dict)

    def params_schema(self) -> dict:
        if self.params is None:
            return {"type": "object"}
        try:
            return self.params.model_json_schema()
        except Exception:                    # a schema pydantic cannot render must not break /api/meta
            return {"type": "object"}

    def validate_params(self, raw: dict | None) -> BaseModel | dict:
        """Params as the kind's model (``additionalProperties: false``); ``422 validation`` when invalid."""
        raw = dict(raw or {})
        if self.params is None:
            return raw
        try:
            return self.params.model_validate(raw)
        except ValidationError as exc:
            from sparc.studio.errors import pydantic_errors, validation_error

            rows = pydantic_errors(exc.errors())
            for r in rows:
                r["path"] = f"params.{r['path']}" if r["path"] else "params"
            raise validation_error(rows, f"invalid params for {self.kind}")

    def dump_params(self, params: BaseModel | dict) -> dict:
        if isinstance(params, BaseModel):
            return params.model_dump(mode="json")
        return dict(params)

    def listing(self) -> dict:
        return {"kind": self.kind, "label": self.label, "lane": self.lane, "executor": self.executor,
                "needs_run": self.needs_run, "needs_checkpoint": self.needs_checkpoint, "locks_run": self.locks_run,
                "network_hosts": list(self.network_hosts), "long": self.long, "params_schema": self.params_schema()}


KINDS: dict[str, JobKind] = {}


def job_kind(kind: str, *, lane: str, executor: str = "process", label: str | None = None,
             params: type[BaseModel] | None = None, needs_run: bool = False, needs_checkpoint: bool = False,
             locks_run: bool = False, network_hosts: tuple[str, ...] | list[str] = (), long: bool = False,
             estimate: Callable | None = None, on_event: Callable | None = None,
             on_event_types: tuple[str, ...] | list[str] = DEFAULT_HOOK_TYPES, on_finish: Callable | None = None,
             preflight: Callable | None = None, retry_params: Callable | None = None,
             threads: Callable | None = None, **extra) -> Callable[[Callable], Callable]:
    """Register the decorated function ``fn(ctx: JobContext, params) -> dict | None`` as job kind ``kind``."""
    if lane not in LANES:
        raise ValueError(f"lane must be one of {LANES}, not {lane!r}")
    if executor not in EXECUTORS:
        raise ValueError(f"executor must be one of {EXECUTORS}, not {executor!r}")

    def deco(fn: Callable) -> Callable:
        KINDS[kind] = JobKind(
            kind=kind, fn=fn, lane=lane, executor=executor, label=label or kind, params=params, needs_run=needs_run,
            needs_checkpoint=needs_checkpoint, locks_run=locks_run, network_hosts=tuple(network_hosts), long=long,
            estimate=estimate, on_event=on_event, on_event_types=tuple(on_event_types), on_finish=on_finish,
            preflight=preflight, retry_params=retry_params, threads=threads,
            module=getattr(fn, "__module__", "") or "", extra=dict(extra))
        return fn

    return deco


def _import_tolerant(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ModuleNotFoundError as exc:
        if exc.name == name or (exc.name and name.startswith(exc.name + ".")):
            log.debug("job kind module %s not present; skipped", name)
            return False
        raise


def load_kind_modules(*, test_kinds: bool | None = None, only: str | None = None) -> list[str]:
    """Import the kind modules (tolerantly); returns the ones that loaded.

    ``only``: import just that module (the worker imports the module that
    registered its kind).  ``test_kinds``: also import :mod:`.testkinds`
    (default: ``$SPARC_STUDIO_TEST_KINDS``).
    """
    from sparc.studio.settings import testkinds_enabled

    if test_kinds is None:
        test_kinds = testkinds_enabled()
    names = [only] if only else list(KIND_MODULES)
    if test_kinds and TEST_KIND_MODULE not in names:
        names.append(TEST_KIND_MODULE)
    return [n for n in names if _import_tolerant(n)]


def get_kind(kind: str) -> JobKind | None:
    return KINDS.get(kind)


def kinds_listing(*, include_test: bool = True) -> list[dict]:
    return [k.listing() for name, k in sorted(KINDS.items())
            if include_test or not name.startswith("test.")]


# ---------------------------------------------------------------------------
# JobContext (worker side)
# ---------------------------------------------------------------------------

class ReadOnlyDB:
    """Read-only access to ``studio.sqlite`` for job code (workers never write it)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
            self._conn.row_factory = lambda cur, row: {d[0]: v for d, v in zip(cur.description, row)}
        return self._conn

    def fetchone(self, sql: str, params=()) -> dict | None:
        try:
            return self._c().execute(sql, params).fetchone()
        except sqlite3.OperationalError:     # no database (or table) yet
            return None

    def fetchall(self, sql: str, params=()) -> list[dict]:
        try:
            return self._c().execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class JobContext:
    """What a job function gets (SPEC §10.2): ids, directories, threads, the read-only DB and the run config.

    Built by the worker from ``job.json``, whose ``context`` block the server
    fills when it spawns the job (project, run, studio and study dirs); the
    database is the fallback for anything missing.
    """

    def __init__(self, job_dir: str | Path, job: dict | None = None):
        from sparc.studio.workspace import Workspace

        self.job_dir = Path(job_dir).resolve()
        self.job = job if job is not None else json.loads((self.job_dir / "job.json").read_text("utf-8"))
        c = self.job.get("context") or {}
        self.job_id: str = self.job["id"]
        self.kind: str = self.job.get("kind", "")
        self.params: dict = dict(self.job.get("params") or {})
        self.workspace = Workspace(c.get("workspace") or self.job_dir.parent.parent)
        self.project_id: str | None = self.job.get("project_id")
        self.run_id: str | None = self.job.get("run_id")
        self.study_id: str | None = self.job.get("study_id")
        self.scenario_id: str | None = self.job.get("scenario_id")
        self.threads: int = int(self.job.get("threads") or 1)
        self.cache_dir = Path(c.get("cache_dir") or self.workspace.cache_dir)
        self.db = ReadOnlyDB(self.workspace.db_path)
        self.project_dir = _path(c.get("project_dir")) or self._lookup("projects", "dir", self.project_id)
        self.run_dir = _path(c.get("run_dir")) or self._lookup("runs", "run_dir", self.run_id)
        self.studio_dir = _path(c.get("studio_dir")) or self._lookup("runs", "studio_dir", self.run_id) or (
            self.run_dir / "studio" if self.run_dir else None)
        self.study_dir = _path(c.get("study_dir")) or self._lookup("studies", "out_dir", self.study_id)
        self.config_path = _path(c.get("config_path"))
        self.result: dict = {}

    def _lookup(self, table: str, column: str, key: str | None) -> Path | None:
        if not key:
            return None
        row = self.db.fetchone(f"SELECT {column} AS v FROM {table} WHERE id = ?", (key,))
        return _path(row["v"]) if row and row.get("v") else None

    def emit_result(self, result: dict) -> None:
        """Merge ``result`` into the job's result (used when the function itself returns None)."""
        self.result.update(result or {})

    @property
    def launch(self) -> dict | None:
        """``<studio_dir>/launch.json`` (SPEC §4.3), or None."""
        if not self.studio_dir:
            return None
        p = Path(self.studio_dir) / "launch.json"
        try:
            return json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def run_config(self):
        """The run's launch-snapshot ``CoreConfig``, resolved as in SPEC §4.3.

        1. ``<studio_dir>/launch.json`` (``config_raw`` + ``config_dir``);
        2. ``manifest.config`` + ``manifest.provenance.config_dir``;
        3. an explicit config path (``job.json`` context ``config_path`` or params ``config_path``).
        """
        from sparc.core.config import core_config_from_dict, load_core_config

        launch = self.launch
        if launch and launch.get("config_raw") is not None:
            return core_config_from_dict(launch["config_raw"], base_dir=launch.get("config_dir") or self.run_dir)
        if self.run_dir:
            try:
                manifest = json.loads((Path(self.run_dir) / "manifest.json").read_text("utf-8"))
            except (OSError, ValueError):
                manifest = None
            if manifest and isinstance(manifest.get("config"), dict):
                base = (manifest.get("provenance") or {}).get("config_dir") or self.run_dir
                return core_config_from_dict(manifest["config"], base_dir=base)
        explicit = self.config_path or _path(self.params.get("config_path"))
        if explicit and Path(explicit).is_file():
            return load_core_config(explicit)
        raise FileNotFoundError(f"no config for run {self.run_id!r}: no launch.json, manifest config or config path")


def _path(v: Any) -> Path | None:
    if not v:
        return None
    return Path(os.fspath(v))
