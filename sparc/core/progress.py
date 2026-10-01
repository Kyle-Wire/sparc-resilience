"""Structured progress, cancellation and thread control for the core pipeline.

Long-running core code reports what it is doing through this module: spans
(run, stages, tasks) with timings, throttled ticks, metrics, artifacts,
checkpoints and warnings.  Each event is one JSON line (schema v1, see
``docs/studio/SPEC.md`` §5.3) appended to a sink: a file path (the
``SPARC_PROGRESS`` environment variable), ``"stderr"``, ``"fd:N"`` or a
callable (tests and in-process use).

* **Zero cost when unconfigured.**  Every reporting call starts with one
  module-level bool check and returns when no sink is configured.
* **Write discipline.**  A line is at most 4,096 bytes (newline included) and
  is written by a single ``os.write`` on an ``O_APPEND`` descriptor, so lines
  from pool workers never interleave (4 KB is also ``PIPE_BUF`` on Linux).
  NaN/Inf become ``null``; oversized lines lose string length first, then
  their largest fields (``"truncated": true``).
* **Cancellation** works with or without a sink.  ``check_cancel()`` raises
  :class:`Cancelled` - a ``BaseException``, so ``except Exception`` blocks do
  not swallow it - once a signal arrived, ``request_cancel()`` was called or
  the cancel file exists (stat at most every 0.5 s).  The first SIGTERM /
  SIGINT only sets the flag; a second one raises at once.
* **Spans nest** through ``contextvars``: across threads with
  :func:`wrap_context` and across process pools with
  ``ProcessPoolExecutor(initializer=progress.init_worker,
  initargs=(progress.worker_env(),))``.

The module is stdlib-only: ``psutil`` is used for heartbeat memory when it is
importable, and ``threadpoolctl`` / an already-imported ``torch`` only inside
:func:`limit_threads` and :func:`set_threads`.  It is excluded from the resume
fingerprint, so instrumentation edits never invalidate checkpoints.
"""

from __future__ import annotations

import contextvars
import functools
import itertools
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Iterator

__all__ = [
    "SCHEMA", "ENV_SINK", "ENV_LEVEL", "ENV_JOB", "ENV_CANCEL", "Cancelled", "LogBridge",
    "configure", "configure_from_env", "reset", "enabled", "sink_path", "emit",
    "span", "stage", "task", "skip", "tick", "metric", "artifact", "checkpoint", "warn",
    "context", "run_dir_scope", "job_scope",
    "check_cancel", "cancel_requested", "request_cancel", "install_signal_handlers",
    "worker_env", "init_worker", "wrap_context", "limit_threads", "set_threads",
]

SCHEMA = 1
ENV_SINK, ENV_LEVEL, ENV_JOB, ENV_CANCEL = "SPARC_PROGRESS", "SPARC_PROGRESS_LEVEL", "SPARC_JOB_ID", "SPARC_CANCEL_FILE"

MAX_LINE = 4096                  # bytes per event line, newline included
TICK_INTERVAL_S = 1.0            # at most one tick per second per span (k == 1 and k == n always pass)
CANCEL_STAT_S = 0.5              # cancel-file stat interval
LOG_CAP_BYTES = 200 * 1024 ** 2  # past this, a debug-level emitter drops to info (SPEC §5.6)
LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

_LEVEL_NAMES = {v: k for k, v in LEVELS.items()}
_ENVELOPE = ("v", "type", "seq", "ts", "t_rel", "pid", "job", "lvl", "span", "parent", "path", "ctx")
_FIXED = frozenset(_ENVELOPE) - {"path", "ctx"}     # never truncated
_RUN_STATUS = {"ok": "succeeded", "error": "failed", "cancelled": "cancelled"}


class Cancelled(BaseException):
    """Raised at a safe point after a cancel request.

    Derives from ``BaseException`` so that ``except Exception`` handlers in
    library code let it through; the CLI and the Studio worker map it to exit
    code 130.
    """


class _Span:
    """An open span; the object ``span()`` yields.

    Callers fill ``metrics`` (task.end), ``summary`` (stage.end) or call
    ``set(**fields)`` for any other closing field (run.end ``timings_s``,
    ``done``) before the block exits.
    """

    __slots__ = ("id", "parent", "path", "kind", "name", "stage", "t0", "last_tick", "metrics", "summary", "fields")

    def __init__(self, id: str | None, parent: str | None, path: tuple, kind: str = "", name: str = "",
                 stage: str | None = None):
        self.id, self.parent, self.path, self.kind, self.name, self.stage = id, parent, path, kind, name, stage
        self.t0 = time.perf_counter()
        self.last_tick = -math.inf
        self.metrics: dict = {}
        self.summary: dict = {}
        self.fields: dict = {}

    def set(self, **fields) -> None:
        self.fields.update(fields)


class _State:
    """Everything a configuration owns; ``job_scope`` swaps whole states."""

    __slots__ = ("sink", "own_fd", "spec", "level", "job", "t0", "heartbeat", "cancel_file", "cancel", "acked",
                 "last_stat", "errors")

    def __init__(self):
        self.sink: int | Callable[[dict], Any] | None = None
        self.own_fd = False
        self.spec: str | None = None          # "stderr", "fd:N" or an absolute path; None for callables
        self.level = LEVELS["info"]
        self.job = ""
        self.t0 = time.time()
        self.heartbeat: _Heartbeat | None = None
        self.cancel_file: str | None = None
        self.cancel = False
        self.acked = False
        self.last_stat = -math.inf
        self.errors = 0


_ENABLED = False                 # the one check every reporting call makes
_S = _State()
_ENV_KEY: tuple | None = None    # env values configure_from_env last applied

_SIGNALED = False                # process-wide: survives job_scope swaps
_SIGNALS = 0
_PREV_HANDLERS: dict[int, Any] = {}
_QUIET_SIGNALS = False           # pool workers do not print the cancel notice

_SEQ = itertools.count(1)
_SPAN_IDS = itertools.count(1)
# Taken while a line gets its seq and is written, so seq increases in file order within a process.
# Re-entrant: a signal handler may emit cancel.ack while the main thread is inside _write.
_WRITE_LOCK = threading.RLock()
_CUR: contextvars.ContextVar[_Span | None] = contextvars.ContextVar("sparc_progress_span", default=None)
_CTX: contextvars.ContextVar[dict | None] = contextvars.ContextVar("sparc_progress_ctx", default=None)
_RUN_DIR: contextvars.ContextVar[str | None] = contextvars.ContextVar("sparc_progress_run_dir", default=None)
_ROOT: _Span | None = None       # the parent process's span, in a pool worker
_ROOT_CTX: dict = {}
_ROOT_RUN_DIR: str | None = None
_TOP = _Span(None, None, ())     # tick throttle for events outside any span

_BRIDGE: LogBridge | None = None
_BRIDGE_LOGGERS = ("sparc", "py.warnings")
_CAPTURED_WARNINGS = False
_PSUTIL: Any = None


def _after_fork_in_child() -> None:
    """A fork may copy the write lock while another thread holds it; the child starts with a free one."""
    global _WRITE_LOCK
    _WRITE_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def configure(sink: str | Callable[[dict], None] | None, *, level: str = "info", job_id: str | None = None,
              cancel_file: str | None = None, heartbeat_s: float = 15.0) -> None:
    """Point the emitter at ``sink`` (a path, ``"stderr"``, ``"fd:N"`` or a callable taking the event dict).

    ``sink=None`` disables emission but still arms ``cancel_file``.  A path is
    opened ``O_WRONLY | O_APPEND | O_CREAT``.  A heartbeat thread emits every
    ``heartbeat_s`` seconds while a sink is configured (0 disables it).
    """
    global _ENABLED, _S
    _ENABLED = False
    _close(_S)
    s = _State()
    s.level = LEVELS.get(str(level or "info").lower(), LEVELS["info"])
    s.job = str(job_id) if job_id else ""
    s.cancel_file = os.path.abspath(os.fspath(cancel_file)) if cancel_file else None   # pool workers may chdir
    if sink is not None and sink != "":
        s.sink, s.own_fd, s.spec = _open_sink(sink)
    _S = s
    _ENABLED = s.sink is not None
    _bridge(_ENABLED)
    if _ENABLED and heartbeat_s and heartbeat_s > 0:
        s.heartbeat = _Heartbeat(s, float(heartbeat_s))
        s.heartbeat.start()


def configure_from_env() -> None:
    """Configure from ``SPARC_PROGRESS``, ``SPARC_PROGRESS_LEVEL``, ``SPARC_JOB_ID`` and ``SPARC_CANCEL_FILE``.

    Idempotent: a second call with unchanged variables does nothing, and with
    neither a sink nor a cancel file set any existing configuration is kept.
    """
    global _ENV_KEY
    key = tuple(os.environ.get(k) or None for k in (ENV_SINK, ENV_LEVEL, ENV_JOB, ENV_CANCEL))
    if key == _ENV_KEY or (key[0] is None and key[3] is None):
        return
    configure(key[0], level=key[1] or "info", job_id=key[2], cancel_file=key[3])
    _ENV_KEY = key


def reset() -> None:
    """Back to the unconfigured state (tests): closes the sink, clears cancel flags, restores signal handlers."""
    global _ENABLED, _S, _ENV_KEY, _SIGNALED, _SIGNALS, _ROOT, _ROOT_CTX, _ROOT_RUN_DIR, _QUIET_SIGNALS
    _ENABLED = False
    _close(_S)
    _S = _State()
    _ENV_KEY = None
    _SIGNALED, _SIGNALS, _QUIET_SIGNALS = False, 0, False
    for sig, handler in list(_PREV_HANDLERS.items()):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, TypeError):
            pass
    _PREV_HANDLERS.clear()
    _ROOT, _ROOT_CTX, _ROOT_RUN_DIR = None, {}, None
    _TOP.last_tick = -math.inf
    _bridge(False)


def enabled() -> bool:
    return _ENABLED


def sink_path() -> str | None:
    """Absolute path of the file sink, if any (recorded as ``run_state.events_path``)."""
    spec = _S.spec if _ENABLED else None
    return spec if spec and spec != "stderr" and not spec.startswith("fd:") else None


def _open_sink(sink) -> tuple[int | Callable, bool, str | None]:
    if callable(sink):
        return sink, False, None
    spec = os.fspath(sink)
    if spec == "stderr":
        return 2, False, spec
    if spec.startswith("fd:"):
        return int(spec[3:]), False, spec
    path = os.path.abspath(spec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    return fd, True, path


def _close(s: _State) -> None:
    if s.heartbeat is not None:
        s.heartbeat.stop()
        s.heartbeat = None
    if s.own_fd and isinstance(s.sink, int):
        try:
            os.close(s.sink)
        except OSError:
            pass
    s.sink, s.own_fd = None, False


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def emit(type: str, /, *, lvl: str = "info", **fields) -> None:
    """Emit one event of ``type`` with the envelope plus ``fields``."""
    if not _ENABLED:
        return
    if LEVELS.get(lvl, 20) < _S.level:
        return
    _write(_event(type, lvl, fields, None))


def _event(type_: str, lvl: str, fields: dict, sp: _Span | None, s: _State | None = None) -> dict:
    """The event dict: envelope plus ``fields`` (``seq`` is assigned by ``_write``)."""
    if sp is None:
        sp = _CUR.get() or _ROOT
    if s is None:
        s = _S
    ctx = _CTX.get()
    now = time.time()
    ev = {"v": SCHEMA, "type": type_, "seq": 0, "ts": round(now, 3), "t_rel": round(now - s.t0, 3),
          "pid": os.getpid(), "job": s.job, "lvl": lvl,
          "span": sp.id if sp is not None else None, "parent": sp.parent if sp is not None else None,
          "path": list(sp.path) if sp is not None else [], "ctx": _clean(_ROOT_CTX if ctx is None else ctx)}
    for k, v in fields.items():
        ev[k + "_" if k in ev else k] = _clean(v)     # a field never overwrites the envelope
    return ev


def _clean(v):
    """JSON-safe copy: NaN/Inf → None, numpy/pandas → Python, paths → str, anything else → str."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_clean(x) for x in v), key=str)
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    if hasattr(v, "tolist") and type(v).__module__.split(".")[0] in ("numpy", "pandas"):
        return _clean(v.tolist())
    return str(v)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, ensure_ascii=False, default=str)


def _line(ev: dict) -> bytes:
    return (_dumps(ev) + "\n").encode("utf-8", "replace")


def _trunc(v, limit: int):
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit - 1] + "…"
    if isinstance(v, list):
        return [_trunc(x, limit) for x in v]
    if isinstance(v, dict):
        return {k: _trunc(x, limit) for k, x in v.items()}
    return v


def _encode(ev: dict) -> tuple[dict, bytes]:
    """The event and its line, shrunk to ``MAX_LINE`` bytes: long strings first, then the largest fields."""
    data = _line(ev)
    if len(data) <= MAX_LINE:
        return ev, data
    ev["truncated"] = True
    for limit in (1024, 256, 64):
        ev = {k: (v if k in _FIXED else _trunc(v, limit)) for k, v in ev.items()}
        data = _line(ev)
        if len(data) <= MAX_LINE:
            return ev, data
    while len(data) > MAX_LINE:
        extra = [k for k in ev if k not in _ENVELOPE and k != "truncated"]
        if not extra:
            break
        del ev[max(extra, key=lambda k: len(_dumps(ev[k])))]
        data = _line(ev)
    if len(data) > MAX_LINE:             # the envelope alone is too long: keep the innermost path
        ev["ctx"] = {}
        ev["path"] = _trunc(ev["path"][-3:], 64) if isinstance(ev["path"], list) else _trunc(ev["path"], 256)
        data = _line(ev)
    if len(data) > MAX_LINE:             # last resort: bound every free-text envelope string
        ev["path"], ev["job"] = [], _trunc(ev["job"], 64)
        ev["type"], ev["lvl"] = _trunc(ev["type"], 128), _trunc(ev["lvl"], 16)
        data = _line(ev)
    return ev, data


def _write(ev: dict, s: _State | None = None) -> None:
    """Number ``ev`` and write it to the sink of ``s`` (default: the current configuration).

    A file descriptor gets the line under ``_WRITE_LOCK``, so ``seq`` grows in
    file order even with several threads reporting.  A callable sink is
    called outside the lock (it may log, and logging takes its own locks).
    """
    if s is None:
        s = _S
    try:
        sink = s.sink
        with _WRITE_LOCK:
            ev["seq"] = next(_SEQ)
            ev, data = _encode(ev)
            if isinstance(sink, int):
                os.write(sink, data)
                return
        if sink is not None:
            sink(ev)
    except Exception:                  # reporting must never break the computation it reports on
        s.errors += 1


# ---------------------------------------------------------------------------
# spans and typed events
# ---------------------------------------------------------------------------

@contextmanager
def span(kind: str, name: str, *, k: int | None = None, n: int | None = None, unit: str | None = None,
         est_s: float | None = None, **fields) -> Iterator[_Span]:
    """Emit ``<kind>.start`` and, on exit, ``<kind>.end`` with ``status`` ok / error / cancelled and ``elapsed_s``.

    ``kind`` is ``"stage"`` or ``"task"``; ``"run"`` is also accepted for
    ``run_core`` (``run.start`` / ``run.end`` with status succeeded / failed /
    cancelled, plus whatever the block passed to ``handle.set``).  Pass
    ``key=`` for a named task and ``label=`` for a stage.  The path element is
    ``<kind>:<name>[k/n]`` (or ``[key]``).
    """
    if not _ENABLED:
        yield _Span(None, None, ())
        return
    up = _CUR.get() or _ROOT
    key = fields.pop("key", None)
    key = None if key is None else str(key)
    el = f"{kind}:{name}"
    if k is not None and n is not None:
        el += f"[{k}/{n}]"
    elif key is not None:
        el += f"[{key}]"
    elif k is not None:
        el += f"[{k}]"
    stage_id = name if kind == "stage" else (up.stage if up is not None else None)
    sp = _Span(f"{os.getpid()}:{next(_SPAN_IDS)}", up.id if up is not None else None,
               (up.path if up is not None else ()) + (el,), kind, name, stage_id)
    if kind == "stage":
        start = {"stage": name, "label": fields.pop("label", name), "est_s": est_s}
        start.update({f: v for f, v in (("k", k), ("n", n), ("unit", unit)) if v is not None})
    elif kind == "run":
        start = {"name": name}
    else:
        start = {"name": name, "key": key, "k": k, "n": n, "unit": unit}
        if est_s is not None:
            start["est_s"] = est_s
    start.update(fields)
    _write(_event(f"{kind}.start", "info", start, sp))
    token = _CUR.set(sp)
    status, exc = "ok", None
    try:
        yield sp
    except BaseException as e:
        status = "cancelled" if isinstance(e, (Cancelled, KeyboardInterrupt)) else "error"
        exc = e
        raise
    finally:
        try:
            _CUR.reset(token)
        except ValueError:             # exited from another context
            _CUR.set(up)
        if _ENABLED:
            elapsed = round(time.perf_counter() - sp.t0, 4)
            err = None if exc is None else {"type": type(exc).__name__, "message": str(exc)[:1000]}
            if kind == "stage":
                end = {"stage": name, "status": status, "elapsed_s": elapsed, "summary": sp.summary}
            elif kind == "run":
                end = {"status": _RUN_STATUS[status], "elapsed_s": elapsed, "timings_s": {}, "done": []}
                if err is not None:
                    err["traceback_tail"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)
                                                    [-12:])[-1500:]
                end["error"] = err
            else:
                end = {"name": name, "key": key, "k": k, "n": n, "unit": unit, "status": status,
                       "elapsed_s": elapsed, "metrics": sp.metrics}
            if err is not None and kind != "run":
                end["error"] = err
            end.update(sp.fields)
            _write(_event(f"{kind}.end", "error" if status == "error" else "info", end, sp))


def stage(name: str, **f):
    """``span("stage", name, ...)``; ``name`` is a stage id (S0, S1, S2_S3, baselines, cv_curve, S4, …)."""
    return span("stage", name, **f)


def task(name: str, **f):
    """``span("task", name, ...)``."""
    return span("task", name, **f)


def skip(stage: str, reason: str, **fields) -> None:
    """``stage.skip``: the stage will not run (``reason`` from SPEC §5.3)."""
    if not _ENABLED:
        return
    emit("stage.skip", stage=stage, reason=reason, **fields)


def tick(k: int, n: int, *, unit: str, label: str = "", lvl: str = "info", **metrics) -> None:
    """Progress ``k`` of ``n`` in ``unit``; at most one per second per span, but ``k == 1`` and ``k == n`` always pass."""
    if not _ENABLED:
        return
    if LEVELS.get(lvl, 20) < _S.level:
        return
    sp = _CUR.get() or _ROOT
    holder = _TOP if sp is None else sp
    now = time.monotonic()
    if k != 1 and k != n and now - holder.last_tick < TICK_INTERVAL_S:
        return
    holder.last_tick = now
    fields = {"k": k, "n": n, "unit": unit, "frac": round(k / n, 6) if n else None, "label": label}
    fields.update(metrics)
    _write(_event("tick", lvl, fields, sp))


def metric(name: str, value: float | int | str | bool | None, *, unit: str | None = None, lvl: str = "info",
           **tags) -> None:
    """A named scalar (``tags`` say what it belongs to, e.g. ``scenario=…``)."""
    if not _ENABLED:
        return
    emit("metric", lvl=lvl, name=name, value=value, unit=unit, tags=tags)


def artifact(path, *, role: str, stage: str | None = None) -> None:
    """A file (or directory) was written: ``path`` relative to the run dir, its size, and the stage it belongs to.

    ``path`` replaces the envelope's span path on this event (the span id is
    kept; the ancestry moves to ``span_path``).
    """
    if not _ENABLED:
        return
    p = os.fspath(path)
    rel = p
    base = _RUN_DIR.get() or _ROOT_RUN_DIR
    if base:
        try:
            r = os.path.relpath(os.path.abspath(p), base)
            if r != ".." and not r.startswith(".." + os.sep):
                rel = r
        except ValueError:             # another drive (Windows)
            pass
    sp = _CUR.get() or _ROOT
    if stage is None and sp is not None:
        stage = sp.stage
    ev = _event("artifact", "info", {"role": role, "bytes": _size(p), "stage": stage}, sp)
    ev["span_path"], ev["path"] = ev["path"], rel.replace(os.sep, "/")
    _write(ev)


def _size(p: str) -> int:
    try:
        if os.path.isdir(p):
            return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(p) for f in files)
        return os.path.getsize(p)
    except OSError:
        return 0


def checkpoint(action: str, *, done, bytes: int | None = None, elapsed_s: float | None = None,
               fingerprint: str | None = None, changed_sections: list[str] | None = None) -> None:
    """``checkpoint`` event: ``action`` is saved, loaded or mismatch."""
    if not _ENABLED:
        return
    done = sorted(done) if isinstance(done, (set, frozenset)) else list(done or [])
    emit("checkpoint", action=action, done=done, bytes=bytes, elapsed_s=elapsed_s, fingerprint=fingerprint,
         changed_sections=None if changed_sections is None else list(changed_sections))


def warn(code: str, message: str, **data) -> None:
    """``warning`` event with a registered ``code`` (SPEC §5.3) and structured ``data``."""
    if not _ENABLED:
        return
    emit("warning", lvl="warning", code=code, message=str(message), data=data)


@contextmanager
def context(**kv) -> Iterator[None]:
    """Merge ``kv`` into ``ctx`` of every event emitted inside the block (partition, variant, generator, …)."""
    cur = _CTX.get()
    token = _CTX.set({**(_ROOT_CTX if cur is None else cur), **kv})
    try:
        yield
    finally:
        _reset_var(_CTX, token, cur)


@contextmanager
def run_dir_scope(run_dir) -> Iterator[None]:
    """``artifact()`` paths inside the block are made relative to ``run_dir``."""
    prev = _RUN_DIR.get()
    token = _RUN_DIR.set(os.path.abspath(os.fspath(run_dir)))
    try:
        yield
    finally:
        _reset_var(_RUN_DIR, token, prev)


def _reset_var(var: contextvars.ContextVar, token, prev) -> None:
    try:
        var.reset(token)
    except ValueError:
        var.set(prev)


@contextmanager
def job_scope(job_id: str, *, sink: str, cancel_file: str | None = None) -> Iterator[None]:
    """Report to another job for the duration of the block (the engine host serves one job per request).

    The previous configuration is restored afterwards, and a cancel requested
    inside the scope does not leak into the next one.
    """
    global _ENABLED, _S, _ENV_KEY
    saved = (_ENABLED, _S, _ENV_KEY)
    level = _LEVEL_NAMES.get(_S.level, "info") if _ENABLED else (os.environ.get(ENV_LEVEL) or "info")
    _ENABLED, _S = False, _State()     # detach without closing the saved sink
    try:
        configure(sink, level=level, job_id=job_id, cancel_file=cancel_file)
        yield
    finally:
        _ENABLED = False
        _close(_S)
        _ENABLED, _S, _ENV_KEY = saved
        _bridge(_ENABLED)


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------

def check_cancel() -> None:
    """Raise :class:`Cancelled` if a cancel was requested (signal, ``request_cancel`` or the cancel file)."""
    if _SIGNALED or _S.cancel:
        _raise_cancelled()
    if _S.cancel_file is not None and _poll_cancel_file():
        _raise_cancelled()


def cancel_requested() -> bool:
    return _SIGNALED or _S.cancel or (_S.cancel_file is not None and _poll_cancel_file())


def request_cancel() -> None:
    """Ask the current job to stop at its next ``check_cancel()`` (in-process callers and tests)."""
    _S.cancel = True


def _poll_cancel_file() -> bool:
    s = _S
    now = time.monotonic()
    if now - s.last_stat < CANCEL_STAT_S:
        return False
    s.last_stat = now
    _check_log_cap(s)
    if os.path.exists(s.cancel_file):
        s.cancel = True
        return True
    return False


def _raise_cancelled():
    s = _S
    if not s.acked:
        s.acked = True
        if _ENABLED:
            sp = _CUR.get() or _ROOT
            _write(_event("cancel.ack", "info", {"at_path": list(sp.path) if sp is not None else []}, None))
    raise Cancelled("cancelled")


def install_signal_handlers() -> None:
    """SIGTERM / SIGINT (and SIGBREAK on Windows) set the cancel flag; a second signal raises :class:`Cancelled`.

    Only possible from the main thread; elsewhere this is a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    sigs = [signal.SIGTERM, signal.SIGINT] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])
    for sig in sigs:
        if sig in _PREV_HANDLERS:
            continue
        try:
            _PREV_HANDLERS[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


def _on_signal(signum, frame) -> None:
    global _SIGNALED, _SIGNALS
    _SIGNALS += 1
    _SIGNALED = True
    if _SIGNALS == 1:
        if not _QUIET_SIGNALS:
            try:
                os.write(2, f"sparc: cancel requested (signal {signum}); stopping at the next safe point "
                            f"- send it again to stop now\n".encode())
            except OSError:
                pass
        return
    _raise_cancelled()


# ---------------------------------------------------------------------------
# pools and threads
# ---------------------------------------------------------------------------

def worker_env() -> dict:
    """What a pool worker needs to report into the same job, under the current span and context."""
    s = _S
    env: dict[str, Any] = {ENV_LEVEL: _LEVEL_NAMES.get(s.level, "info")}
    if _ENABLED and s.spec is not None and not s.spec.startswith("fd:"):
        env[ENV_SINK] = s.spec
    if s.job:
        env[ENV_JOB] = s.job
    if s.cancel_file:
        env[ENV_CANCEL] = s.cancel_file
    sp = _CUR.get() or _ROOT
    ctx = _CTX.get()
    env["span"] = None if sp is None else [sp.id, sp.parent, list(sp.path), sp.stage]
    env["ctx"] = _clean(_ROOT_CTX if ctx is None else ctx)
    env["run_dir"] = _RUN_DIR.get() or _ROOT_RUN_DIR
    env["t0"] = s.t0
    return env


def init_worker(env: dict) -> None:
    """Pool initializer: ``ProcessPoolExecutor(initializer=progress.init_worker, initargs=(progress.worker_env(),))``."""
    global _ROOT, _ROOT_CTX, _ROOT_RUN_DIR, _QUIET_SIGNALS, _ENV_KEY
    env = dict(env or {})
    for k in (ENV_SINK, ENV_LEVEL, ENV_JOB, ENV_CANCEL):
        if env.get(k):
            os.environ[k] = str(env[k])
        else:
            os.environ.pop(k, None)
    configure(env.get(ENV_SINK), level=env.get(ENV_LEVEL) or "info", job_id=env.get(ENV_JOB),
              cancel_file=env.get(ENV_CANCEL), heartbeat_s=0)
    _ENV_KEY = tuple(os.environ.get(k) or None for k in (ENV_SINK, ENV_LEVEL, ENV_JOB, ENV_CANCEL))
    if env.get("t0") is not None:
        _S.t0 = float(env["t0"])
    sp = env.get("span")
    _ROOT = None if not sp else _Span(sp[0], sp[1], tuple(sp[2]), stage=sp[3])
    _ROOT_CTX = dict(env.get("ctx") or {})
    _ROOT_RUN_DIR = env.get("run_dir")
    _QUIET_SIGNALS = True
    install_signal_handlers()


def wrap_context(fn: Callable) -> Callable:
    """Run ``fn`` in (a copy of) the caller's context: ``executor.submit(progress.wrap_context(fn), …)``.

    Each call runs in its own copy, so one wrapper may be mapped over many
    items concurrently.
    """
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def run(*args, **kwargs):
        return ctx.copy().run(fn, *args, **kwargs)

    return run


_LIMIT_LOCK = threading.Lock()
_LIMITS: list[list[int]] = []    # open limit_threads blocks, oldest first; the newest one applies
_LIMIT_SAVED: tuple | None = None  # (env, torch threads, threadpoolctl limiter) from before the first block


@contextmanager
def limit_threads(n: int) -> Iterator[None]:
    """Cap BLAS / OpenMP / torch threads at ``n`` inside the block (all restored on exit).

    Sets ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``OPENBLAS_NUM_THREADS``
    (read by libraries loaded later, torch included), applies
    ``threadpoolctl`` limits to the loaded ones when it is installed, and
    calls ``torch.set_num_threads`` when torch is already imported.  torch is
    never imported here, so a torch-free process stays torch-free.  The
    limits are process-wide: blocks open in several threads share them (the
    newest applies) and the original state returns when the last one exits.
    """
    global _LIMIT_SAVED
    entry = [max(1, int(n))]
    with _LIMIT_LOCK:
        if not _LIMITS:
            torch = sys.modules.get("torch")
            _LIMIT_SAVED = ({k: os.environ.get(k) for k in THREAD_ENV},
                            torch.get_num_threads() if torch is not None else None, _threadpool_limits(entry[0]))
        _LIMITS.append(entry)
        _apply_threads(entry[0], tpc=len(_LIMITS) > 1)
    try:
        yield
    finally:
        with _LIMIT_LOCK:
            del _LIMITS[next(i for i, e in enumerate(_LIMITS) if e is entry)]
            if _LIMITS:
                _apply_threads(_LIMITS[-1][0])
            else:
                env, torch_threads, limiter = _LIMIT_SAVED
                _LIMIT_SAVED = None
                if limiter is not None:
                    limiter.restore_original_limits()
                torch = sys.modules.get("torch")
                if torch is not None and torch_threads is not None:
                    torch.set_num_threads(torch_threads)
                for k, v in env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


def set_threads(n: int) -> None:
    """Permanent form of :func:`limit_threads` for a worker process."""
    _apply_threads(max(1, int(n)))


def _apply_threads(n: int, tpc: bool = True) -> None:
    for k in THREAD_ENV:
        os.environ[k] = str(n)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(n)
    if tpc:
        _threadpool_limits(n)


def _threadpool_limits(n: int):
    """Apply ``threadpoolctl`` limits (the returned limiter can restore the previous ones), or None."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=n)
    except Exception:                  # not installed, or a library it cannot drive
        return None


# ---------------------------------------------------------------------------
# heartbeat and logging
# ---------------------------------------------------------------------------

class _Heartbeat(threading.Thread):
    def __init__(self, state: _State, interval: float):
        super().__init__(name="sparc-progress-heartbeat", daemon=True)
        self.state, self.interval = state, interval
        self.stopped = threading.Event()

    def run(self) -> None:
        s = self.state
        while not self.stopped.wait(self.interval):
            if _ENABLED and _S is s:               # paused while a job_scope reports elsewhere
                if LEVELS["info"] >= s.level:
                    # written through its own state: a job_scope starting meanwhile never gets this beat
                    _write(_event("heartbeat", "info", _resources(), None, s), s)
                _check_log_cap(s)

    def stop(self) -> None:
        self.stopped.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=2.0)


def _resources() -> dict:
    global _PSUTIL
    rss = cpu = threads = None
    if _PSUTIL is None:
        try:
            import psutil

            _PSUTIL = psutil
        except ImportError:
            _PSUTIL = False
    if _PSUTIL:
        try:
            p = _PSUTIL.Process()
            with p.oneshot():
                rss = p.memory_info().rss
                t = p.cpu_times()
                cpu, threads = t.user + t.system, p.num_threads()
        except Exception:
            pass
    if rss is None:
        try:
            with open("/proc/self/statm") as f:
                rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            try:
                import resource

                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss = peak if sys.platform == "darwin" else peak * 1024
            except Exception:
                rss = 0
    if cpu is None:
        t = os.times()
        cpu = t.user + t.system
    if threads is None:
        try:
            threads = len(os.listdir("/proc/self/task"))
        except OSError:
            threads = threading.active_count()
    return {"rss_mb": round(rss / 2 ** 20, 1), "cpu_s": round(cpu, 2), "threads": threads}


def _check_log_cap(s: _State) -> None:
    if s.level < LEVELS["info"] and isinstance(s.sink, int) and s.own_fd:
        try:
            if os.fstat(s.sink).st_size > LOG_CAP_BYTES:
                s.level = LEVELS["info"]
        except OSError:
            pass


class LogBridge(logging.Handler):
    """Forwards log records verbatim as ``log`` events; WARNING and above also as ``warning`` (code ``log.<logger>``).

    Attached to the ``sparc`` and ``py.warnings`` loggers while a sink is
    configured (with ``logging.captureWarnings(True)``).  Records below a
    logger's effective level never reach it: the host process decides how
    verbose logging is (the CLI's ``logging.basicConfig(level=INFO)``).
    """

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self._busy = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if not _ENABLED or getattr(self._busy, "on", False):
            return
        self._busy.on = True
        try:
            msg = record.getMessage()
            lvl = ("error" if record.levelno >= logging.ERROR else "warning" if record.levelno >= logging.WARNING
                   else "info" if record.levelno >= logging.INFO else "debug")
            fields = {"logger": record.name, "level": record.levelname, "msg": msg}
            if record.exc_info and record.exc_info[0] is not None:
                fields["exc"] = "".join(traceback.format_exception(*record.exc_info))[-1500:]
            _emit_event("log", lvl=lvl, **fields)
            if record.levelno >= logging.WARNING:
                _emit_event("warning", lvl=lvl, code=f"log.{record.name}", message=msg, data={})
        except Exception:
            pass
        finally:
            self._busy.on = False


_emit_event = emit


def _bridge(on: bool) -> None:
    global _BRIDGE, _CAPTURED_WARNINGS
    if on and _BRIDGE is None:
        _BRIDGE = LogBridge()
        for name in _BRIDGE_LOGGERS:
            logging.getLogger(name).addHandler(_BRIDGE)
        if getattr(logging, "_warnings_showwarning", None) is None:
            logging.captureWarnings(True)
            _CAPTURED_WARNINGS = True
    elif not on and _BRIDGE is not None:
        for name in _BRIDGE_LOGGERS:
            logging.getLogger(name).removeHandler(_BRIDGE)
        _BRIDGE = None
        if _CAPTURED_WARNINGS:
            logging.captureWarnings(False)
            _CAPTURED_WARNINGS = False
