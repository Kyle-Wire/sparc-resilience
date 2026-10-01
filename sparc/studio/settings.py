"""Launch settings (``StudioSettings``, from the CLI) and user settings (``Settings``, SPEC §10.10).

``StudioSettings`` is what one server process was started with: workspace,
bind address, token, security options and a few internal knobs tests use
(timeouts, poll intervals).  It never changes while the server runs.

``Settings`` is the user-editable table behind ``GET/PUT /api/settings``.
Defaults depend on the machine (CPU count, RAM); only explicitly saved keys
are stored, one row per key in the ``settings`` table, so changing hardware
moves the defaults with it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sparc.studio.workspace import Workspace, resolve_workspace

__all__ = ["StudioSettings", "Settings", "SettingsPatch", "default_settings", "load_settings", "save_settings",
           "SETTINGS_KEYS", "ENV_TEST_KINDS", "ENV_RUNNER", "testkinds_enabled", "machine"]

ENV_TEST_KINDS = "SPARC_STUDIO_TEST_KINDS"
ENV_RUNNER = "SPARC_STUDIO_RUNNER"
DEFAULT_PORT = 8765


def testkinds_enabled() -> bool:
    return os.environ.get(ENV_TEST_KINDS, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class StudioSettings:
    """How this server process was launched (``sparc studio`` flags, SPEC §10.9)."""

    workspace: Workspace = field(default_factory=resolve_workspace)
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str | None = None                   # None: generated per launch
    allow_remote: bool = False
    public_hosts: list[str] = field(default_factory=list)
    dev_origins: list[str] = field(default_factory=list)
    reindex: bool = False
    stop_jobs_on_exit: bool = False
    log_level: str = "info"
    static_dir: Path | None = None             # None: the packaged sparc/studio/static
    test_kinds: bool | None = None             # None: from $SPARC_STUDIO_TEST_KINDS
    # internal knobs (tests shorten them)
    kill_grace_s: float = 90.0                 # cancelling → Force stop allowed (SPEC §5.9)
    shutdown_grace_s: float = 30.0             # stop_jobs on exit: cancel, then kill after this
    start_timeout_s: float = 30.0              # starting with no worker event → failed
    tail_interval_s: float = 0.25              # JobTailer poll
    flush_interval_s: float = 2.0              # projection → SQLite
    sample_interval_s: float = 2.0             # ResourceSampler
    schedule_interval_s: float = 1.0           # scheduler re-check of blocked/queued jobs
    exit_poll_s: float = 0.25                  # exit watcher poll
    ping_interval_s: float = 15.0              # SSE keep-alive comment
    sampler: bool = True                       # run the ResourceSampler

    def __post_init__(self):
        if not isinstance(self.workspace, Workspace):
            self.workspace = Workspace(self.workspace)
        if self.test_kinds is None:
            self.test_kinds = testkinds_enabled()

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"


# ---------------------------------------------------------------------------
# user settings
# ---------------------------------------------------------------------------

def machine() -> dict:
    """``{cpu_count, mem_total_gb}`` of this machine (psutil when available)."""
    cpu = os.cpu_count() or 1
    try:
        import psutil

        cpu = psutil.cpu_count(logical=True) or cpu
        mem = psutil.virtual_memory().total / 2 ** 30
    except Exception:  # pragma: no cover - psutil is a core dependency
        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        except (ValueError, OSError, AttributeError):
            mem = 8.0
    return {"cpu_count": int(cpu), "mem_total_gb": float(mem)}


class Settings(BaseModel):
    """User settings (SPEC §10.10, api.md §2).  Machine-dependent defaults come from :func:`default_settings`."""

    model_config = ConfigDict(extra="forbid")

    thread_budget: int = Field(ge=1)
    threads_heavy: int = Field(ge=1)
    engine_threads: int = Field(2, ge=1)
    heavy_slots: int = Field(1, ge=1, le=16)
    medium_slots: int = Field(1, ge=1, le=16)
    network_slots: int = Field(2, ge=1, le=16)
    engine_max_runs: int = Field(2, ge=1, le=16)
    engine_mem_budget_gb: float = Field(gt=0)
    engine_idle_min: float = Field(30, ge=0)
    auto_uncertainty: bool = True
    watch_roots: list[str] = Field(default_factory=list)
    upload_max_gb: float = Field(2, gt=0)
    keep_job_logs_days: float | None = Field(None, ge=0)
    offline: bool = False
    notifications: bool = False
    basemap_url: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.threads_heavy + self.engine_threads > self.thread_budget + 1:
            raise ValueError(f"threads_heavy + engine_threads ({self.threads_heavy} + {self.engine_threads}) "
                             f"exceeds thread_budget + 1 ({self.thread_budget + 1})")
        return self


class SettingsPatch(BaseModel):
    """``PUT /api/settings`` body: any subset of :class:`Settings`."""

    model_config = ConfigDict(extra="forbid")

    thread_budget: int | None = None
    threads_heavy: int | None = None
    engine_threads: int | None = None
    heavy_slots: int | None = None
    medium_slots: int | None = None
    network_slots: int | None = None
    engine_max_runs: int | None = None
    engine_mem_budget_gb: float | None = None
    engine_idle_min: float | None = None
    auto_uncertainty: bool | None = None
    watch_roots: list[str] | None = None
    upload_max_gb: float | None = None
    keep_job_logs_days: float | None = None
    offline: bool | None = None
    notifications: bool | None = None
    basemap_url: str | None = None


SETTINGS_KEYS: tuple[str, ...] = tuple(Settings.model_fields)


def default_settings() -> dict:
    m = machine()
    cpu = m["cpu_count"]
    heavy = max(1, cpu - 1)
    return {
        "thread_budget": cpu,
        "threads_heavy": heavy,
        # 2, but never so many that the defaults break their own budget rule (a 1-CPU machine gets 1)
        "engine_threads": max(1, min(2, cpu + 1 - heavy)),
        "heavy_slots": 1,
        "medium_slots": 1,
        "network_slots": 2,
        "engine_max_runs": 2,
        "engine_mem_budget_gb": round(min(0.4 * m["mem_total_gb"], 6.0), 2),
        "engine_idle_min": 30,
        "auto_uncertainty": True,
        "watch_roots": [],
        "upload_max_gb": 2,
        "keep_job_logs_days": None,
        "offline": False,
        "notifications": False,
        "basemap_url": None,
    }


def load_settings(db) -> Settings:
    """Defaults overlaid with the stored keys; a stored set that no longer validates falls back key by key."""
    values = default_settings()
    stored: dict[str, Any] = {}
    for row in db.fetchall("SELECT key, value_json FROM settings"):
        if row["key"] in SETTINGS_KEYS:
            from sparc.studio.db import loads

            stored[row["key"]] = loads(row["value_json"])
    try:
        return Settings(**{**values, **stored})
    except ValidationError:
        good = {}
        for k, v in stored.items():
            try:
                Settings(**{**values, **good, k: v})
                good[k] = v
            except ValidationError:
                pass
        return Settings(**{**values, **good})


def save_settings(db, patch: dict) -> Settings:
    """Validate ``patch`` against the current settings and store the changed keys.

    Raises :class:`~sparc.studio.errors.ApiError` ``422 validation`` with one
    row per problem (including a watch root that is not a directory).
    """
    from sparc.studio.db import dumps
    from sparc.studio.errors import pydantic_errors, validation_error

    patch = {k: v for k, v in patch.items() if k in SETTINGS_KEYS}
    current = load_settings(db).model_dump()
    merged = {**current, **patch}
    try:
        new = Settings(**merged)
    except ValidationError as exc:
        rows = pydantic_errors(exc.errors())
        for r in rows:
            if not r["path"]:
                r["path"] = "threads_heavy" if "threads_heavy" in r["message"] else ""
        raise validation_error(rows, "invalid settings")
    bad = [{"path": f"watch_roots.{i}", "message": f"not a directory: {p}", "code": "not_a_directory"}
           for i, p in enumerate(new.watch_roots) if not Path(os.path.expanduser(p)).is_dir()]
    if bad:
        raise validation_error(bad, "invalid settings")
    rows = [(k, dumps(v) if v is not None else "null") for k, v in patch.items()]
    db.executemany("INSERT OR REPLACE INTO settings (key, value_json) VALUES (?, ?)", rows)
    return new
