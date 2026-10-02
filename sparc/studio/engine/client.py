"""The server's side of the engine host protocol (api.md §13): start, find, ask, stop and kill the host.

The host is started on first need with ``python -m sparc.studio.engine.host --sock <ws>/engine/host.sock`` in
its own session (process group), stdout and stderr appended to ``engine/host.log``.  It survives server
restarts: :meth:`EngineClient.info` accepts a running host only when ``host.json``'s ``pid`` is alive with
the same ``create_time``; otherwise the stale ``host.json`` and socket are removed and the next request starts
a new host.

Each request is its own connection: connect (``authkey`` from ``host.json``), send one dict, wait for one
reply.  A host that dies mid-request raises :class:`HostGone`; a failed request raises :class:`EngineError`
with the host's ``{type, message, traceback_tail}``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sparc.studio.engine.host import HOST_JSON, HOST_LOG, address_for
from sparc.studio.workspace import read_json

log = logging.getLogger("sparc.studio.engine")

__all__ = ["EngineClient", "EngineError", "HostGone", "START_TIMEOUT_S"]

START_TIMEOUT_S = 60.0
_IS_WIN = os.name == "nt"


class HostGone(RuntimeError):
    """The host is not running, or exited before replying."""


class EngineError(RuntimeError):
    """A request the host answered with ``ok: false``."""

    def __init__(self, error: dict):
        self.type = str((error or {}).get("type") or "Error")
        self.message = str((error or {}).get("message") or "")
        self.traceback_tail = str((error or {}).get("traceback_tail") or "")
        super().__init__(f"{self.type}: {self.message}")

    def as_dict(self) -> dict:
        return {"type": self.type, "message": self.message, "traceback_tail": self.traceback_tail}


def _alive(pid: int | None, create_time: float | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(int(pid))
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        return create_time is None or abs(p.create_time() - float(create_time)) < 0.01
    except Exception:
        return False


class EngineClient:
    """Connection to the engine host of one workspace (thread-safe)."""

    def __init__(self, workspace, *, idle_min: float = 30.0, threads: int = 2):
        self.ws = workspace
        self.engine_dir = Path(workspace.engine_dir)
        self.idle_min = float(idle_min)
        self.threads = int(threads)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self.starting = False
        self.last_error: str | None = None

    # ------------------------------------------------------------------ discovery

    @property
    def host_json(self) -> Path:
        return self.engine_dir / HOST_JSON

    def info(self) -> dict | None:
        """``host.json`` of a live host (pid + create_time match), else None."""
        info = read_json(self.host_json)
        if not isinstance(info, dict) or not info.get("authkey_hex"):
            return None
        if not _alive(info.get("pid"), info.get("create_time")):
            return None
        return info

    def alive(self) -> bool:
        return self.info() is not None

    def cleanup_stale(self) -> None:
        """Remove ``host.json`` and the socket of a host that is gone."""
        if self.info() is not None:
            return
        for p in (self.host_json, self.engine_dir / "host.sock"):
            try:
                p.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ lifecycle

    def start(self, timeout: float = START_TIMEOUT_S) -> dict:
        """Start a host (unless one is running) and wait for its ``host.json``."""
        with self._lock:
            info = self.info()
            if info is not None:
                return info
            self.cleanup_stale()
            self.engine_dir.mkdir(parents=True, exist_ok=True)
            from sparc.studio.jobs.executors import _sparc_root

            env = dict(os.environ)
            root = _sparc_root()
            paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
            env["PYTHONPATH"] = os.pathsep.join([root, *paths])
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[k] = str(max(1, self.threads))
            for k in ("SPARC_PROGRESS", "SPARC_JOB_ID", "SPARC_CANCEL_FILE", "SPARC_PROGRESS_LEVEL"):
                env.pop(k, None)
            address, _family = address_for(self.engine_dir)
            sock = address if not _IS_WIN else str(self.engine_dir / "host.sock")
            cmd = [sys.executable, "-m", "sparc.studio.engine.host", "--sock", sock, "--idle-min", str(self.idle_min)]
            log_f = open(self.engine_dir / HOST_LOG, "ab")
            kw: dict[str, Any] = {}
            if _IS_WIN:
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            else:
                kw["start_new_session"] = True
            self.starting = True
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log_f, stderr=log_f, env=env,
                                        cwd=str(self.engine_dir), close_fds=True, **kw)
            finally:
                log_f.close()
            self._proc = proc
            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    info = read_json(self.host_json)
                    if isinstance(info, dict) and info.get("pid") == proc.pid and info.get("authkey_hex"):
                        self.last_error = None
                        return info
                    if proc.poll() is not None:
                        self.last_error = f"the engine host exited with code {proc.returncode} while starting " \
                                          f"(see {self.engine_dir / HOST_LOG})"
                        raise HostGone(self.last_error)
                    time.sleep(0.05)
                self.last_error = "the engine host did not start in time"
                proc.kill()
                raise HostGone(self.last_error)
            finally:
                self.starting = False

    def ensure(self) -> dict:
        info = self.info()
        return info if info is not None else self.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the host to exit (clean shutdown); kill it after ``timeout``."""
        info = self.info()
        if info is None:
            return
        try:
            self.request("shutdown", start=False, timeout=5.0)
        except (HostGone, EngineError, OSError):
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _alive(info.get("pid"), info.get("create_time")):
            self._reap()
            time.sleep(0.05)
        if _alive(info.get("pid"), info.get("create_time")):
            self.kill()
        self._reap()
        self.cleanup_stale()

    def kill(self) -> bool:
        """SIGKILL the host's process group (Force stop / restart); True when one was running."""
        info = read_json(self.host_json) or {}
        pid = info.get("pid")
        if not pid or not _alive(pid, info.get("create_time")):
            self.cleanup_stale()
            return False
        try:
            if _IS_WIN:
                os.kill(int(pid), signal.SIGTERM)
            else:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _alive(pid, info.get("create_time")):
            self._reap()
            time.sleep(0.02)
        self._reap()
        self.cleanup_stale()
        return True

    def _reap(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None

    # ------------------------------------------------------------------ requests

    def request(self, op: str, *, job_id: str | None = None, job_dir: str | None = None, run_id: str | None = None,
                run_dir: str | None = None, threads: int = 1, payload: dict | None = None, start: bool = True,
                timeout: float | None = None) -> dict:
        """Send one request; returns the host's ``result``.  ``timeout`` bounds the wait for the reply."""
        from multiprocessing.connection import Client

        info = self.ensure() if start else self.info()
        if info is None:
            raise HostGone("the engine host is not running")
        try:
            conn = Client(info["sock"], family=info.get("family") or "AF_UNIX",
                          authkey=bytes.fromhex(info["authkey_hex"]))
        except (OSError, EOFError) as exc:
            raise HostGone(f"cannot reach the engine host: {exc}") from exc
        rid = uuid.uuid4().hex
        try:
            conn.send({"op": op, "request_id": rid, "job_id": job_id, "job_dir": job_dir, "run_id": run_id,
                       "run_dir": run_dir, "threads": int(threads or 1), "payload": payload or {}})
            if timeout is not None and not conn.poll(timeout):
                raise HostGone(f"the engine host did not answer {op} within {timeout:.0f} s")
            reply = conn.recv()
        except (OSError, EOFError) as exc:
            raise HostGone(f"the engine host exited during {op}: {exc}") from exc
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if not isinstance(reply, dict):
            raise EngineError({"type": "BadReply", "message": f"unexpected reply {type(reply).__name__}"})
        if not reply.get("ok"):
            raise EngineError(reply.get("error") or {})
        return reply.get("result") or {}

    def status(self, timeout: float = 5.0) -> dict | None:
        """The host's status, or None when no host runs (never starts one)."""
        try:
            return self.request("status", start=False, timeout=timeout)
        except (HostGone, EngineError):
            return None
