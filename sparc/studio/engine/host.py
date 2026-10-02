"""The engine host: ``python -m sparc.studio.engine.host --sock <ws>/engine/host.sock`` (SPEC §7.6, api.md §13).

A long-lived process in its own session that keeps loaded runs in memory and serves exact scenario
requests one at a time.

**Transport.**  ``multiprocessing.connection.Listener`` on ``AF_UNIX`` (``AF_PIPE`` on Windows) with a random
``authkey``.  Once listening the host writes ``engine/host.json`` ``{pid, create_time, sock, family,
authkey_hex, started_utc}`` (0600); the server reads it to connect, and on a restart reconnects when ``pid``
and ``create_time`` still match.  Logs go to ``engine/host.log``; stdout is never used for protocol.  Every
connection carries one pickled request dict and gets one reply dict; requests are unpickled by a restricted
unpickler that admits only builtins and numpy arrays.

**Requests** ``{op, request_id, job_id, job_dir, run_id, run_dir, threads, payload}``; ops ``status``,
``open``, ``close``, ``scenario``, ``batch``, ``rerun_configured``, ``sweep``, ``plan_verify``,
``plan_frontier`` and ``shutdown``.  ``status`` is answered at once by the connection's thread, even while a
request runs; every other op waits for the single work lock.  A request with a job runs inside
``progress.job_scope(job_id, sink=<job_dir>/events.jsonl, cancel_file=<job_dir>/cancel)`` and
``progress.limit_threads(threads)``, gets one tick per fold of every engine pass, and its outcome is
written to ``<job_dir>/result.json`` (``{status, exit_code, result, error}``) before the reply - so a server
that restarted meanwhile still finds it.  A cancel raises ``Cancelled`` between folds; the host stays up.

**LRU** of :class:`~sparc.core.session.RunSession` objects bounded by count (``engine_max_runs``) and
memory (``engine_mem_budget_gb``; the limits ride on every request).  Eviction drops the least recently used
session and runs ``gc.collect()``.  When the RSS still exceeds the budget + 1 GB (allocator fragmentation)
the host finishes the request and exits ("recycle"); the server restarts it lazily and re-opens the most
recently used run.  It also exits after ``engine_idle_min`` minutes without a loaded run.

**Opening a run** uses ``<studio_dir>/engine/base_fold.npy`` (float64 K × n; ``base_fold.json`` keys it by
checkpoint mtime+size and code sha) and writes it after a fresh baseline pass.  An ``AttributeError`` /
``ImportError`` while unpickling or in the first evaluation is ``Incompatible``: recorded in
``<studio_dir>/engine/status.json`` until the checkpoint changes.
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import os
import pickle
import secrets
import sys
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("sparc.studio.engine.host")

__all__ = ["main", "Host", "safe_loads", "HOST_JSON", "SOCK_NAME", "HOST_LOG", "RECYCLE_JSON", "address_for",
           "unix_bind_path"]

HOST_JSON = "host.json"
SOCK_NAME = "host.sock"
HOST_LOG = "host.log"
RECYCLE_JSON = "recycle.json"
_IS_WIN = os.name == "nt"
WORK_OPS = ("open", "close", "scenario", "batch", "rerun_configured", "sweep", "plan_verify", "plan_frontier")
SLACK_ENV = "SPARC_STUDIO_ENGINE_SLACK_GB"


# ---------------------------------------------------------------------------
# restricted unpickling
# ---------------------------------------------------------------------------

_ALLOWED = {
    ("builtins", n) for n in ("dict", "list", "tuple", "set", "frozenset", "str", "bytes", "bytearray", "int",
                              "float", "bool", "complex", "slice")
} | {
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("_codecs", "encode"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        if module == "numpy" and name.startswith(("float", "int", "uint", "bool")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"the engine host refuses {module}.{name} in a request")


def safe_loads(raw: bytes) -> Any:
    """Unpickle a request: builtins and numpy arrays only (api.md §13)."""
    return _SafeUnpickler(io.BytesIO(raw)).load()


def address_for(engine_dir: Path) -> tuple[str, str]:
    """``(address, family)`` of the host's listener for a workspace's ``engine/`` directory."""
    if _IS_WIN:
        import hashlib

        h = hashlib.sha1(str(Path(engine_dir).resolve()).encode()).hexdigest()[:12]
        return rf"\\.\pipe\sparc-studio-engine-{h}", "AF_PIPE"
    return str(Path(engine_dir) / SOCK_NAME), "AF_UNIX"


# sun_path holds 108 bytes on Linux and 104 on macOS, including the terminating NUL.
_SUN_PATH_MAX = 100


def unix_bind_path(sock: Path) -> str:
    """The path the host binds for the nominal ``<ws>/engine/host.sock``.

    A workspace deep in the file system gives a socket path longer than ``AF_UNIX`` allows; the host
    then binds a short per-workspace path in the temp directory instead (owner-only, like the
    nominal one).  ``host.json`` records the path actually bound, which is what clients connect to.
    """
    import hashlib
    import tempfile

    path = str(sock)
    if len(os.fsencode(path)) <= _SUN_PATH_MAX:
        return path
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return os.path.join(tempfile.gettempdir(), f"sparc-engine-{uid}-{h}.sock")


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return 0.0


def _now() -> str:
    from sparc.studio.workspace import utc_now

    return utc_now()


# ---------------------------------------------------------------------------
# the host
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    session: Any
    run_dir: str
    studio_dir: str | None
    loaded_utc: str
    last_used_utc: str
    est_rss_mb: float
    code_match: bool | None
    load_seconds: dict = field(default_factory=dict)
    evaluated: bool = False              # an engine pass ran on this session (the load-time one counts)


class Host:
    """The request loop, the LRU and the lifecycle of the engine host (see the module docstring)."""

    def __init__(self, listener=None, *, engine_dir: Path | None = None, idle_min: float = 30.0,
                 watch_s: float = 5.0):
        self.listener = listener
        self.engine_dir = Path(engine_dir) if engine_dir else None
        self.sessions: "OrderedDict[str, Entry]" = OrderedDict()
        self.max_runs = 2
        self.budget_gb = 6.0
        self.idle_min = float(idle_min)
        self.watch_s = float(watch_s)
        self.slack_gb = float(os.environ.get(SLACK_ENV) or 1.0)
        self.work = threading.Lock()
        self.state_lock = threading.Lock()
        self.busy: dict | None = None
        self.last_activity = time.monotonic()
        self.stopping = threading.Event()
        self.recycle = False
        self.started_utc = _now()
        self.incompatible: dict[str, dict] = {}

    # ------------------------------------------------------------------ serving

    def serve(self) -> None:
        threading.Thread(target=self._watchdog, name="engine-idle", daemon=True).start()
        while not self.stopping.is_set():
            try:
                conn = self.listener.accept()
            except (OSError, EOFError):
                if self.stopping.is_set():
                    break
                continue
            except Exception:                    # an authentication failure: drop that client only
                log.warning("rejected a connection: %s", traceback.format_exc(limit=1).strip())
                continue
            threading.Thread(target=self._handle, args=(conn,), name="engine-conn", daemon=True).start()

    def stop(self) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        try:
            self.listener.close()
        except Exception:
            pass

    def _watchdog(self) -> None:
        """Stop the host after ``idle_min`` minutes with no loaded run and no request in progress."""
        while not self.stopping.wait(self.watch_s):
            with self.state_lock:
                idle = (not self.sessions and self.busy is None and
                        time.monotonic() - self.last_activity > self.idle_min * 60.0)
            if idle:
                log.info("no loaded run for %.0f min: stopping", self.idle_min)
                self.stop()

    def _handle(self, conn) -> None:
        try:
            try:
                raw = conn.recv_bytes()
            except (EOFError, OSError):
                return
            try:
                req = safe_loads(raw)
                if not isinstance(req, dict) or not isinstance(req.get("op"), str):
                    raise ValueError("a request is a dict with an op")
            except Exception as exc:
                self._send(conn, {"request_id": None, "ok": False,
                                  "error": {"type": "BadRequest", "message": str(exc)[:500], "traceback_tail": ""}})
                return
            op = req["op"]
            if op == "status":
                self._send(conn, {"request_id": req.get("request_id"), "ok": True, "result": self.status()})
                return
            if op == "shutdown":
                self._send(conn, {"request_id": req.get("request_id"), "ok": True, "result": {"stopping": True}})
                self.stop()
                return
            if op not in WORK_OPS:
                self._send(conn, {"request_id": req.get("request_id"), "ok": False,
                                  "error": {"type": "BadRequest", "message": f"unknown op {op!r}",
                                            "traceback_tail": ""}})
                return
            with self.work:
                reply = self.execute(req)
                if self.recycle:
                    log.warning("host RSS %.0f MB exceeds the budget + %.1f GB after eviction: recycling",
                                _rss_mb(), self.slack_gb)
                    self._write_recycle()          # before the reply: the server looks for it right after
            self._send(conn, reply)
            if self.recycle:
                self.stop()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _write_recycle(self) -> None:
        """``engine/recycle.json``: the most recently used run, which the server re-opens in the next host."""
        if self.engine_dir is None:
            return
        from sparc.studio.workspace import write_json_atomic

        with self.state_lock:
            mru = next(reversed(self.sessions), None) if self.sessions else None
        try:
            write_json_atomic(self.engine_dir / RECYCLE_JSON, {"pid": os.getpid(), "mru_run_id": mru,
                                                               "rss_mb": round(_rss_mb(), 1), "at": _now()})
        except OSError:
            log.exception("cannot write %s", RECYCLE_JSON)

    @staticmethod
    def _send(conn, reply: dict) -> None:
        try:
            conn.send(reply)
        except (OSError, EOFError, BrokenPipeError):
            log.info("the server went away before the reply of %s", reply.get("request_id"))

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        with self.state_lock:
            runs = [{"run_id": rid, "est_rss_mb": round(e.est_rss_mb, 1), "loaded_utc": e.loaded_utc,
                     "last_used_utc": e.last_used_utc, "code_match": e.code_match, "load_seconds": e.load_seconds}
                    for rid, e in self.sessions.items()]
            busy = dict(self.busy) if self.busy else None
        return {"pid": os.getpid(), "rss_mb": round(_rss_mb(), 1), "runs": runs, "busy": busy,
                "max_runs": self.max_runs, "budget_gb": self.budget_gb, "started_utc": self.started_utc,
                "incompatible": dict(self.incompatible), "recycle": self.recycle}

    # ------------------------------------------------------------------ requests

    def execute(self, req: dict) -> dict:
        from sparc.core import progress

        rid = req.get("request_id")
        job_id, job_dir = req.get("job_id"), req.get("job_dir")
        limits = (req.get("payload") or {}).get("limits") or {}
        self._limits(limits)
        with self.state_lock:
            self.busy = {"request_id": rid, "op": req["op"], "job_id": job_id, "run_id": req.get("run_id"),
                         "started_utc": _now()}
            self.last_activity = time.monotonic()
        scope = progress.job_scope(job_id, sink=str(Path(job_dir) / "events.jsonl"),
                                   cancel_file=str(Path(job_dir) / "cancel")) if job_id and job_dir else None
        status, code, result, error = "failed", 1, None, None
        saved_tick = progress.TICK_INTERVAL_S
        try:
            if scope is not None:
                scope.__enter__()
            # an engine pass is K fold ticks; on small runs folds take milliseconds, so the 1/s throttle
            # would hide all but the first and last (on real runs a fold takes seconds and nothing changes)
            progress.TICK_INTERVAL_S = 0.0
            try:
                with progress.limit_threads(max(1, int(req.get("threads") or 1))):
                    result = self._dispatch(req)
                status, code = "succeeded", 0
                if scope is not None:
                    progress.emit("job.result", result=result or {})
            except progress.Cancelled:
                status, code, error = "cancelled", 130, {"type": "Cancelled", "message": "cancelled",
                                                         "traceback_tail": ""}
            except BaseException as exc:          # noqa: BLE001 - every failure is reported, the host lives on
                error = _error(exc)
                log.error("%s failed: %s", req["op"], error["message"])
            finally:
                progress.TICK_INTERVAL_S = saved_tick
        finally:
            if scope is not None:
                try:
                    scope.__exit__(None, None, None)
                except Exception:
                    log.exception("closing the job scope failed")
            with self.state_lock:
                self.busy = None
                self.last_activity = time.monotonic()
        if job_dir:
            self._write_result(Path(job_dir), status, code, result, error)
        if status == "succeeded":
            return {"request_id": rid, "ok": True, "result": result}
        return {"request_id": rid, "ok": False, "error": error}

    @staticmethod
    def _write_result(job_dir: Path, status: str, code: int, result, error) -> None:
        from sparc.studio.workspace import write_json_atomic

        try:
            write_json_atomic(job_dir / "result.json", {"status": status, "exit_code": code, "result": result,
                                                        "error": error, "finished_utc": _now()})
        except OSError:
            log.exception("cannot write %s/result.json", job_dir)

    def _limits(self, limits: dict) -> None:
        if limits.get("max_runs"):
            self.max_runs = max(1, int(limits["max_runs"]))
        if limits.get("budget_gb"):
            self.budget_gb = float(limits["budget_gb"])
        if limits.get("idle_min") is not None:
            self.idle_min = float(limits["idle_min"])

    def _dispatch(self, req: dict) -> dict:
        from sparc.studio.engine import ops

        op = req["op"]
        payload = dict(req.get("payload") or {})
        run_id = req.get("run_id")
        if op == "close":
            return {"closed": self.evict(run_id) if run_id else False}
        if not run_id or not req.get("run_dir"):
            raise ValueError(f"{op} needs run_id and run_dir")
        entry = self.ensure(run_id, req["run_dir"], payload, int(req.get("threads") or 1))
        if op == "open":
            return {"load_seconds": entry.load_seconds, "rss_mb": round(_rss_mb(), 1),
                    "code_match": entry.code_match, "est_rss_mb": round(entry.est_rss_mb, 1)}
        entry.session.set_threads(int(req.get("threads") or 1))
        from sparc.core.session import IncompatibleCheckpoint

        try:
            out = ops.run_op(op, entry.session, payload, job_id=req.get("job_id"))
        except IncompatibleCheckpoint as exc:
            if entry.evaluated:                  # it evaluated before: not drift, report the original error
                raise (exc.__cause__ or exc) from None
            self._incompatible_at_first_use(run_id, entry, exc)
            raise
        with self.state_lock:
            entry.evaluated = True
            entry.last_used_utc = _now()
        return out

    def _incompatible_at_first_use(self, run_id: str, entry: "Entry", exc: BaseException) -> None:
        """The first evaluation of a session opened from a cached baseline pass failed with pickle drift: record
        ``incompatible`` (until the checkpoint changes), evict the session and drop the cached pass so the next
        open evaluates at load time."""
        from sparc.core.session import checkpoint_key

        ck = checkpoint_key(entry.run_dir)
        info = {"type": "Incompatible", "message": str(exc)[:1000], "ckpt_key": ck}
        self.incompatible[run_id] = info
        self._write_status(entry.studio_dir, {"state": "incompatible", "error": info, "ckpt_key": ck,
                                              "updated_utc": _now()})
        if entry.studio_dir:
            for name in ("base_fold.json", "base_fold.npy"):
                try:
                    (Path(entry.studio_dir) / "engine" / name).unlink()
                except OSError:
                    pass
        self.evict(run_id)

    # ------------------------------------------------------------------ the LRU

    def evict(self, run_id: str) -> bool:
        with self.state_lock:
            e = self.sessions.pop(run_id, None)
        if e is None:
            return False
        del e
        gc.collect()
        log.info("evicted %s", run_id)
        return True

    def _evict_lru(self, keep: str | None = None) -> str | None:
        with self.state_lock:
            victim = next((rid for rid in self.sessions if rid != keep), None)
        if victim is not None:
            self.evict(victim)
        return victim

    def ensure(self, run_id: str, run_dir: str, payload: dict, threads: int) -> Entry:
        """The loaded session of ``run_id``, loading it when needed.  Before a load, least recently used sessions
        are evicted to stay within ``max_runs`` and to leave room in the memory budget for the new run's
        estimated RSS (≈ 3.5 × checkpoint bytes + 0.3 GB); after it, by the measured RSS."""
        from sparc.core import progress
        from sparc.core.session import IncompatibleCheckpoint, checkpoint_key, open_run

        with self.state_lock:
            e = self.sessions.get(run_id)
            if e is not None:
                self.sessions.move_to_end(run_id)
                e.last_used_utc = _now()
        if e is not None:
            return e
        if payload.get("trusted") is not True:
            raise PermissionError("the engine host only loads checkpoints the registry marked trusted")
        studio_dir = payload.get("studio_dir")
        ck = checkpoint_key(run_dir)
        while len(self.sessions) >= self.max_runs:
            self._evict_lru()
        from sparc.studio.engine.service import estimate_rss_gb

        try:
            need_mb = estimate_rss_gb(os.path.getsize(Path(run_dir) / "checkpoint.pkl")) * 1024.0
        except OSError:
            need_mb = 0.0
        while self.sessions and _rss_mb() + need_mb > self.budget_gb * 1024.0:
            self._evict_lru()
        base, base_meta = self._base_fold(studio_dir, ck)
        before = _rss_mb()
        t0 = time.perf_counter()
        try:
            fallback = payload.get("config_path")
            from sparc.core.session import config_for_run

            cfg = config_for_run(run_dir, fallback=fallback if fallback and Path(fallback).is_file() else None,
                                 studio_dir=studio_dir)
            session = open_run(run_dir, cfg, threads=threads, base_fold=base, studio_dir=studio_dir)
        except IncompatibleCheckpoint as exc:
            info = {"type": "Incompatible", "message": str(exc)[:1000], "ckpt_key": ck}
            self.incompatible[run_id] = info
            self._write_status(studio_dir, {"state": "incompatible", "error": info, "ckpt_key": ck,
                                            "updated_utc": _now()})
            raise
        self.incompatible.pop(run_id, None)
        self._write_status(studio_dir, None)
        if base is None and studio_dir:
            self._save_base_fold(studio_dir, session, ck)
        rss = _rss_mb()
        now = _now()
        entry = Entry(session=session, run_dir=str(run_dir), studio_dir=studio_dir, loaded_utc=now, last_used_utc=now,
                      est_rss_mb=max(0.0, rss - before), code_match=session.code_match,
                      load_seconds={**session.load_seconds, "total": round(time.perf_counter() - t0, 3)},
                      evaluated=base is None)
        with self.state_lock:
            self.sessions[run_id] = entry
            self.sessions.move_to_end(run_id)
        progress.metric("engine.rss_mb", round(rss, 1), unit="MB", run=run_id)
        # memory budget: drop least recently used sessions, then recycle if fragmentation keeps RSS high
        budget_mb = self.budget_gb * 1024.0
        while _rss_mb() > budget_mb and len(self.sessions) > 1:
            self._evict_lru(keep=run_id)
        if _rss_mb() > budget_mb + self.slack_gb * 1024.0:
            self.recycle = True
        return entry

    # ------------------------------------------------------------------ files

    @staticmethod
    def _base_fold(studio_dir: str | None, ck: str | None) -> tuple[np.ndarray | None, dict | None]:
        if not studio_dir or not ck:
            return None, None
        from sparc.studio.engine.store import code_sha
        from sparc.studio.workspace import read_json

        d = Path(studio_dir) / "engine"
        meta = read_json(d / "base_fold.json")
        if not isinstance(meta, dict) or meta.get("ckpt_key") != ck or meta.get("code_sha") != code_sha():
            return None, None
        try:
            return np.load(d / "base_fold.npy", allow_pickle=False), meta
        except (OSError, ValueError):
            return None, None

    @staticmethod
    def _save_base_fold(studio_dir: str, session, ck: str | None) -> None:
        from sparc.core import runio
        from sparc.studio.engine.store import code_sha
        from sparc.studio.workspace import write_json_atomic

        d = Path(studio_dir) / "engine"
        try:
            d.mkdir(parents=True, exist_ok=True)
            with runio.atomic_open(d / "base_fold.npy", "wb") as fh:
                np.save(fh, np.asarray(session.base_fold, dtype=np.float64), allow_pickle=False)
            write_json_atomic(d / "base_fold.json", {"ckpt_key": ck, "code_sha": code_sha(),
                                                     "shape": list(np.shape(session.base_fold)),
                                                     "written_utc": _now()})
        except OSError:
            log.warning("cannot cache base_fold in %s", d)

    @staticmethod
    def _write_status(studio_dir: str | None, status: dict | None) -> None:
        if not studio_dir:
            return
        from sparc.studio.workspace import write_json_atomic

        p = Path(studio_dir) / "engine" / "status.json"
        try:
            if status is None:
                if p.exists():
                    p.unlink()
            else:
                write_json_atomic(p, status)
        except OSError:
            log.warning("cannot write %s", p)


def _error(exc: BaseException) -> dict:
    from sparc.core.session import IncompatibleCheckpoint

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-12:])[-3000:]
    etype = "Incompatible" if isinstance(exc, IncompatibleCheckpoint) else type(exc).__name__
    return {"type": etype, "message": str(exc)[:2000], "traceback_tail": tb}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from multiprocessing.connection import Listener

    from sparc.studio.workspace import write_json_atomic

    ap = argparse.ArgumentParser(prog="python -m sparc.studio.engine.host")
    ap.add_argument("--sock", required=True, help="<workspace>/engine/host.sock")
    ap.add_argument("--idle-min", type=float, default=30.0)
    args = ap.parse_args(argv)
    engine_dir = Path(args.sock).resolve().parent
    engine_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(engine_dir / HOST_LOG, encoding="utf-8")], force=True)
    address, family = address_for(engine_dir)
    if family == "AF_UNIX":
        address = unix_bind_path(Path(args.sock).resolve())
        try:
            os.unlink(address)
        except FileNotFoundError:
            pass
    authkey = secrets.token_bytes(32)
    old = os.umask(0o077)
    try:
        listener = Listener(address, family=family, backlog=16, authkey=authkey)
    finally:
        os.umask(old)
    sock = getattr(getattr(listener, "_listener", None), "_socket", None)
    if sock is not None:                     # accept() wakes every second to notice a stop
        sock.settimeout(1.0)
    try:
        import psutil

        ctime = psutil.Process().create_time()
    except Exception:
        ctime = None
    write_json_atomic(engine_dir / HOST_JSON, {"pid": os.getpid(), "create_time": ctime, "sock": address,
                                               "family": family, "authkey_hex": authkey.hex(),
                                               "started_utc": _now()}, private=True)
    log.info("engine host %d listening on %s", os.getpid(), address)
    host = Host(listener, engine_dir=engine_dir, idle_min=args.idle_min)
    try:
        host.serve()
    finally:
        try:
            listener.close()
        except Exception:
            pass
        try:
            from sparc.studio.workspace import read_json

            cur = read_json(engine_dir / HOST_JSON) or {}
            if cur.get("pid") == os.getpid():
                (engine_dir / HOST_JSON).unlink()
            if family == "AF_UNIX" and os.path.exists(address):
                os.unlink(address)
        except OSError:
            pass
        log.info("engine host %d stopped%s", os.getpid(), " (recycle)" if host.recycle else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
