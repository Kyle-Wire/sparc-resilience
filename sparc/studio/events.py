"""Event schema v1 (SPEC §5.3, api.md §17) and the in-process EventHub.

**Per-job events** are the lines of ``jobs/<jid>/events.jsonl``: written by
``sparc.core.progress`` in the worker and, for ``job.status`` /
``cancel.requested``, appended by the server.  :func:`validate_event` checks
a parsed line against the pydantic model of its type (unknown fields are
kept) and turns unknown or malformed types into ``log`` events, so readers
can rely on the documented fields.  :func:`append_event` is the server's
single-``os.write`` appender.

**The EventHub** fans events out to SSE subscribers:

* global events (``job.created``, ``job.status``, ``job.progress``,
  ``run.*``, ``engine.status``, ``storage.low``, …) get a monotonic
  ``gseq`` and live in a 10,000-event ring buffer, so a reconnecting client
  resumes from ``Last-Event-ID: g:<gseq>`` or gets ``resync {reason:
  "expired"}``;
* per-job items (event lines with their byte cursor, transient ``resource``
  samples, ``end``) go to subscribers of that job.

Every subscriber has a bounded queue (1,000 global, 2,000 per job).  On
overflow the queue is emptied and the subscriber is flagged; the SSE layer
then sends ``resync``.  ``publish*`` may be called from any thread.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import threading
import time
from typing import Any, Iterable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "EVENT_SCHEMA_VERSION", "EVENT_TYPES", "SERVER_EVENT_TYPES", "GLOBAL_TOPICS", "Envelope", "JobEvent",
    "validate_event", "event_json_schema", "append_event", "encode_line", "EventHub", "Subscription",
]

EVENT_SCHEMA_VERSION = 1
MAX_LINE = 4096


class Envelope(BaseModel):
    """Fields present on every line (SPEC §5.3).  Unknown fields are kept."""

    model_config = ConfigDict(extra="allow")

    v: int = 1
    type: str
    seq: int = 0
    ts: float
    t_rel: float = 0.0
    pid: int = 0
    job: str = ""
    lvl: Literal["debug", "info", "warning", "error"] = "info"
    span: str | None = None
    parent: str | None = None
    path: list[Any] = Field(default_factory=list)
    ctx: dict[str, Any] = Field(default_factory=dict)


class PlanNodeEv(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str | None = None
    state: str = "will_run"
    reason: str | None = None
    units: dict[str, float] = Field(default_factory=dict)
    checkpoint_key: str | None = None


class RunStart(Envelope):
    type: Literal["run.start"]
    name: str
    stages: list[str] = Field(default_factory=list)
    fast: bool | None = None
    coarse: float | None = None
    resume: bool | None = None
    cv_curve: bool | None = None
    config_sha256: str | None = None
    code_sha256: str | None = None
    run_meta: dict[str, Any] = Field(default_factory=dict)


class RunDir(Envelope):
    type: Literal["run.dir"]
    run_dir: str
    fingerprint: str | None = None


class RunPlan(Envelope):
    type: Literal["run.plan"]
    nodes: list[PlanNodeEv]
    total_units: dict[str, float] = Field(default_factory=dict)
    n_points: int | None = None


class StageStart(Envelope):
    type: Literal["stage.start"]
    stage: str
    label: str | None = None
    est_s: float | None = None


class StageEnd(Envelope):
    type: Literal["stage.end"]
    stage: str
    status: Literal["ok", "error", "cancelled"]
    elapsed_s: float | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class StageSkip(Envelope):
    type: Literal["stage.skip"]
    stage: str
    reason: str


class TaskStart(Envelope):
    type: Literal["task.start"]
    name: str
    key: str | None = None
    k: int | None = None
    n: int | None = None
    unit: str | None = None


class TaskEnd(Envelope):
    type: Literal["task.end"]
    name: str
    key: str | None = None
    k: int | None = None
    n: int | None = None
    unit: str | None = None
    status: Literal["ok", "error", "cancelled"]
    elapsed_s: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class Tick(Envelope):
    type: Literal["tick"]
    k: float
    n: float
    unit: str
    frac: float | None = None
    label: str = ""


class Metric(Envelope):
    type: Literal["metric"]
    name: str
    value: float | int | str | bool | None = None
    unit: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


class Artifact(Envelope):
    """``path`` is the run-relative file; the span ancestry moves to ``span_path``."""
    type: Literal["artifact"]
    path: str  # type: ignore[assignment]
    role: str
    bytes: int = 0
    stage: str | None = None
    span_path: list[Any] = Field(default_factory=list)


class Checkpoint(Envelope):
    type: Literal["checkpoint"]
    action: Literal["saved", "loaded", "mismatch"]
    done: list[str] = Field(default_factory=list)
    bytes: int | None = None
    elapsed_s: float | None = None
    fingerprint: str | None = None
    changed_sections: list[str] | None = None


class Warning_(Envelope):
    type: Literal["warning"]
    code: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Log(Envelope):
    type: Literal["log"]
    logger: str = ""
    level: str = "INFO"
    msg: str = ""


class Heartbeat(Envelope):
    type: Literal["heartbeat"]
    rss_mb: float | None = None
    cpu_s: float | None = None
    threads: int | None = None


class CancelRequested(Envelope):
    type: Literal["cancel.requested"]
    by: Literal["user", "kill", "shutdown"]


class CancelAck(Envelope):
    type: Literal["cancel.ack"]
    at_path: list[Any] = Field(default_factory=list)


class RunEnd(Envelope):
    type: Literal["run.end"]
    status: Literal["succeeded", "failed", "cancelled"]
    elapsed_s: float | None = None
    timings_s: dict[str, Any] = Field(default_factory=dict)
    done: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None


class JobStatusEv(Envelope):
    type: Literal["job.status"]
    status: Literal["queued", "blocked", "starting", "running", "cancelling", "succeeded", "failed", "cancelled",
                    "interrupted"]
    exit_code: int | None = None
    error: dict[str, Any] | None = None


class JobResult(Envelope):
    type: Literal["job.result"]
    result: dict[str, Any] = Field(default_factory=dict)


_MODELS: tuple[type[Envelope], ...] = (
    RunStart, RunDir, RunPlan, StageStart, StageEnd, StageSkip, TaskStart, TaskEnd, Tick, Metric, Artifact,
    Checkpoint, Warning_, Log, Heartbeat, CancelRequested, CancelAck, RunEnd, JobStatusEv, JobResult,
)
_BY_TYPE: dict[str, type[Envelope]] = {m.model_fields["type"].annotation.__args__[0]: m for m in _MODELS}
EVENT_TYPES: tuple[str, ...] = tuple(_BY_TYPE)
SERVER_EVENT_TYPES = frozenset({"job.status", "cancel.requested"})

JobEvent = Union[_MODELS]  # type: ignore[valid-type]


def event_json_schema() -> dict:
    """JSON Schema of the per-job event union (``GET /api/meta/event-schema``)."""
    schema = TypeAdapter(JobEvent).json_schema(ref_template="#/$defs/{model}")
    schema["title"] = "SPARC Studio job event (schema v1)"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def validate_event(ev: Any) -> dict:
    """``ev`` checked against its type's model.  Known and valid: returned as is (unknown fields kept).

    Unknown types and invalid lines become ``log`` events that keep the
    envelope and say what arrived (``orig_type``), so nothing is silently lost.
    """
    if not isinstance(ev, dict):
        return _as_log({}, "studio.event", f"not an event object: {str(ev)[:200]}")
    model = _BY_TYPE.get(ev.get("type"))
    if model is None:
        return _as_log(ev, "studio.event", f"unknown event type {ev.get('type')!r}")
    try:
        model.model_validate(ev)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ()))
        return _as_log(ev, "studio.event", f"invalid {ev.get('type')} event ({where}: {first.get('msg', 'invalid')})")
    return ev


def _as_log(ev: dict, logger: str, msg: str) -> dict:
    out = {k: ev[k] for k in ("v", "seq", "ts", "t_rel", "pid", "job", "span", "parent", "path", "ctx") if k in ev}
    out.setdefault("v", 1)
    out.setdefault("ts", time.time())
    out.setdefault("path", [])
    out.setdefault("ctx", {})
    out.update({"type": "log", "lvl": "warning", "logger": logger, "level": "WARNING", "msg": msg,
                "orig_type": ev.get("type")})
    return out


# ---------------------------------------------------------------------------
# server-side appends
# ---------------------------------------------------------------------------

_SERVER_SEQ = 0
_SEQ_LOCK = threading.Lock()


def encode_line(ev: dict) -> bytes:
    """One JSON line ≤ 4,096 bytes (long strings in ``error`` and ``msg`` are shortened first)."""
    line = json.dumps(ev, separators=(",", ":"), allow_nan=False, ensure_ascii=False, default=str) + "\n"
    data = line.encode("utf-8", "replace")
    limit = 1024
    while len(data) > MAX_LINE and limit >= 16:
        ev = _shorten(ev, limit)
        data = (json.dumps(ev, separators=(",", ":"), allow_nan=False, ensure_ascii=False, default=str)
                + "\n").encode("utf-8", "replace")
        limit //= 4
    return data


def _shorten(v, limit: int):
    if isinstance(v, str):
        return v if len(v) <= limit else v[: limit - 1] + "…"
    if isinstance(v, dict):
        return {k: (x if k in ("v", "type", "seq", "ts", "pid", "job", "lvl") else _shorten(x, limit))
                for k, x in v.items()}
    if isinstance(v, list):
        return [_shorten(x, limit) for x in v[:64]]
    return v


def append_event(events_path, type_: str, *, job_id: str, t0: float | None = None, lvl: str = "info",
                 **fields) -> dict:
    """Append one server event (``job.status``, ``cancel.requested``) with a single ``os.write`` (O_APPEND).

    The envelope matches core's: ``pid`` is the server's, ``span``/``parent``
    are null and ``path`` is empty.  Returns the event dict.
    """
    global _SERVER_SEQ
    now = time.time()
    with _SEQ_LOCK:
        _SERVER_SEQ += 1
        seq = _SERVER_SEQ
    ev = {"v": EVENT_SCHEMA_VERSION, "type": type_, "seq": seq, "ts": round(now, 3),
          "t_rel": round(now - t0, 3) if t0 else 0.0, "pid": os.getpid(), "job": job_id, "lvl": lvl,
          "span": None, "parent": None, "path": [], "ctx": {}}
    for k, v in fields.items():
        ev[k + "_" if k in ev else k] = v
    data = encode_line(ev)
    path = os.fspath(events_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return ev


# ---------------------------------------------------------------------------
# hub
# ---------------------------------------------------------------------------

#: global event type → topic (None: delivered on every topic)
GLOBAL_TOPICS: dict[str, str | None] = {
    "job.created": "jobs", "job.status": "jobs", "job.progress": "jobs",
    "run.updated": "runs", "run.indexed": "runs", "output.written": "runs", "scenario.result": "runs",
    "engine.status": "engine", "study.updated": "studies", "storage.low": "storage",
    "resync": None, "server_shutdown": None,
}
ALL_TOPICS = ("jobs", "runs", "engine", "studies", "storage")


class Subscription:
    """A bounded queue of hub items for one SSE connection."""

    def __init__(self, hub: "EventHub", maxsize: int, *, job_id: str | None = None,
                 topics: Iterable[str] | None = None):
        self.hub = hub
        self.maxsize = maxsize
        self.job_id = job_id
        self.topics = None if topics is None else frozenset(topics)
        self.items: collections.deque = collections.deque()
        self.overflowed = False
        self.closed = False
        self._ready = asyncio.Event()

    def wants(self, type_: str) -> bool:
        topic = GLOBAL_TOPICS.get(type_, "jobs")
        return self.topics is None or topic is None or topic in self.topics

    def put(self, item) -> None:
        if self.closed:
            return
        if len(self.items) >= self.maxsize:
            self.items.clear()
            self.overflowed = True
        else:
            self.items.append(item)
        self._ready.set()

    async def get(self, timeout: float | None = None):
        """Next item; ``None`` on timeout.  Check :attr:`overflowed` / :attr:`closed` after each call."""
        while not self.items:
            if self.closed or self.overflowed:
                return None
            self._ready.clear()
            try:
                await asyncio.wait_for(self._ready.wait(), timeout)
            except asyncio.TimeoutError:
                return None
        return self.items.popleft()

    def close(self) -> None:
        self.closed = True
        self._ready.set()
        self.hub._unsubscribe(self)


class EventHub:
    """Fan-out of global and per-job events to SSE subscribers (see the module docstring)."""

    def __init__(self, *, ring_size: int = 10_000, global_queue: int = 1_000, job_queue: int = 2_000):
        self.ring: collections.deque = collections.deque(maxlen=ring_size)
        self.gseq = 0
        self.global_queue = global_queue
        self.job_queue = job_queue
        self._global: set[Subscription] = set()
        self._jobs: dict[str, set[Subscription]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: int | None = None
        self.closed = False

    def bind(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Remember the event loop that owns the subscribers (publishes from other threads hop onto it)."""
        self._loop = loop or asyncio.get_running_loop()
        self._thread = threading.get_ident()

    def _call(self, fn, *args):
        if self._loop is not None and threading.get_ident() != self._thread and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(fn, *args)
        else:
            fn(*args)

    # -- global -----------------------------------------------------------------------------

    def publish(self, type_: str, data: dict | None = None) -> None:
        """Publish a global event (``data`` gets ``gseq`` and ``ts``); safe from any thread."""
        self._call(self._publish, type_, dict(data or {}))

    def _publish(self, type_: str, data: dict) -> dict:
        self.gseq += 1
        rec = {**data, "type": type_, "gseq": self.gseq, "ts": round(time.time(), 3)}
        self.ring.append(rec)
        for sub in list(self._global):
            if sub.wants(type_):
                sub.put(rec)
        return rec

    def subscribe_global(self, topics: Iterable[str] | None = None) -> Subscription:
        sub = Subscription(self, self.global_queue, topics=topics)
        self._global.add(sub)
        return sub

    def replay(self, after: int) -> tuple[bool, list[dict]]:
        """``(expired, records)``: the buffered events after ``gseq == after``.

        ``expired`` when events after ``after`` were already evicted, or when
        ``after`` is beyond the current ``gseq`` (a client of an earlier
        server process).
        """
        if after > self.gseq:
            return True, []
        oldest = self.ring[0]["gseq"] if self.ring else self.gseq + 1
        if after < oldest - 1:
            return True, []
        return False, [r for r in self.ring if r["gseq"] > after]

    # -- per job ----------------------------------------------------------------------------

    def subscribe_job(self, job_id: str) -> Subscription:
        sub = Subscription(self, self.job_queue, job_id=job_id)
        self._jobs.setdefault(job_id, set()).add(sub)
        return sub

    def has_job_subscribers(self, job_id: str) -> bool:
        return bool(self._jobs.get(job_id))

    def publish_job(self, job_id: str, item: tuple) -> None:
        """``item`` is ``("event", cursor, ev)``, ``("resource", data)`` or ``("end", status)``."""
        self._call(self._publish_job, job_id, item)

    def _publish_job(self, job_id: str, item: tuple) -> None:
        for sub in list(self._jobs.get(job_id, ())):
            sub.put(item)

    def _unsubscribe(self, sub: Subscription) -> None:
        if sub.job_id is None:
            self._global.discard(sub)
        else:
            subs = self._jobs.get(sub.job_id)
            if subs is not None:
                subs.discard(sub)
                if not subs:
                    self._jobs.pop(sub.job_id, None)

    def close(self) -> None:
        """End every subscription (server shutdown)."""
        self.closed = True
        for sub in list(self._global) + [s for subs in self._jobs.values() for s in subs]:
            sub.closed = True
            sub._ready.set()
