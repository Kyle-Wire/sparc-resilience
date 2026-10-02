"""``EngineExecutor``: the job executor of the ``engine`` lane (SPEC §7.6, §10.2).

Registered at import with ``register_executor("engine", ENGINE_EXECUTOR)``; the job manager binds it to the
server context at start.  For each ``engine.*`` job:

* **submit** appends a ``log`` line to the job's ``events.jsonl`` (so the job shows as running at once) and
  starts a thread that prepares the request on the server (the kind's ``prepare``: compile the scenario,
  resolve masks, allocate result ids - :mod:`sparc.studio.engine.kinds`) and sends it to the host;
* the **host** reports into the job's ``events.jsonl`` and writes ``<job_dir>/result.json``; **wait**
  resolves once that file exists (exit code 0 / 1 / 130);
* **cancel**: the manager has already touched the job's ``cancel`` file; the host raises ``Cancelled`` between
  folds and stays alive (a request not sent yet is not sent);
* **kill** = host restart: the host's process group is killed, which evicts every engine; this job ends
  ``cancelled`` and any other in-flight engine job ``failed: engine host exited``; the host starts again on
  the next request;
* **reattach** (server restart): a job whose ``result.json`` exists, or that the live host is still serving,
  is followed again; anything else ends ``interrupted``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

from sparc.studio.engine.client import EngineError, HostGone
from sparc.studio.events import append_event
from sparc.studio.jobs.executors import ExitInfo, register_executor
from sparc.studio.workspace import read_json

log = logging.getLogger("sparc.studio.engine")

__all__ = ["EngineExecutor", "ENGINE_EXECUTOR", "write_job_result", "RECYCLE_REOPEN_S"]

RECYCLE_REOPEN_S = 300.0


def write_job_result(job_dir: Path, status: str, code: int, result=None, error: dict | None = None) -> None:
    """``result.json`` as the worker writes it (api.md §12.3) - only when the host did not write one."""
    from sparc.studio.jobs.worker import write_result

    if (Path(job_dir) / "result.json").exists():
        return
    write_result(Path(job_dir), status, code, result, error)


class EngineExecutor:
    name = "engine"

    def __init__(self, poll_s: float = 0.2):
        self.sctx = None
        self.poll_s = poll_s
        self.threads: dict[str, threading.Thread] = {}
        self.killed: set[str] = set()
        self.reopened: dict[str, float] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind(self, sctx) -> None:
        self.sctx = sctx

    def service(self):
        from sparc.studio.engine.service import get_service

        return get_service(self.sctx)

    # ------------------------------------------------------------------ submit

    async def submit(self, job: dict) -> dict:
        self.loop = asyncio.get_running_loop()
        job_dir = Path(job["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        append_event(job_dir / "events.jsonl", "log", job_id=job["id"], logger="sparc.studio.engine", level="INFO",
                     msg=f"engine request queued: {job['kind']}")
        th = threading.Thread(target=self._run, args=(dict(job),), name=f"engine-{job['id']}", daemon=True)
        with self._lock:
            self.threads[job["id"]] = th
        th.start()
        info = self.service().client.info() or {}
        return {"pid": info.get("pid"), "pgid": info.get("pid"), "proc_create_time": info.get("create_time"),
                "cmdline_token": str(job_dir), "executor": self.name}

    def _run(self, job: dict) -> None:
        from sparc.studio.engine.kinds import prepare_request

        job_dir = Path(job["job_dir"])
        jid = job["id"]
        try:
            if (job_dir / "cancel").exists():
                write_job_result(job_dir, "cancelled", 130)
                return
            op, payload = prepare_request(self.sctx, job)
            svc = self.service()
            payload["limits"] = svc.limits()
            if (job_dir / "cancel").exists():
                write_job_result(job_dir, "cancelled", 130)
                return
            run = self.sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (job.get("run_id"),)) \
                if job.get("run_id") else None
            svc.publish(job.get("run_id"), "loading" if op == "open" else "busy")
            svc.client.request(op, job_id=jid, job_dir=str(job_dir), run_id=job.get("run_id"),
                               run_dir=run["run_dir"] if run else None, threads=int(job.get("threads") or 1),
                               payload=payload)
        except HostGone as exc:
            if jid in self.killed:
                write_job_result(job_dir, "cancelled", 130)
            else:
                write_job_result(job_dir, "failed", 1, error={"type": "EngineExit",
                                                               "message": f"engine host exited ({exc})"})
        except EngineError as exc:
            write_job_result(job_dir, "cancelled" if exc.type == "Cancelled" else "failed",
                             130 if exc.type == "Cancelled" else 1, error=exc.as_dict())
        except BaseException as exc:            # noqa: BLE001 - a preparation error fails the job, not the server
            import traceback

            from sparc.studio.errors import ApiError

            msg = exc.message if isinstance(exc, ApiError) else str(exc)
            write_job_result(job_dir, "failed", 1, error={
                "type": exc.code if isinstance(exc, ApiError) else type(exc).__name__, "message": msg[:2000],
                "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-8:])[-2000:]})
        finally:
            try:
                self.service().invalidate()
                self._after_recycle()
            except Exception:
                log.exception("engine bookkeeping after %s failed", jid)

    def _after_recycle(self) -> None:
        """A host that recycled itself left ``engine/recycle.json``: re-open its most recently used run in a new
        host (once per run within :data:`RECYCLE_REOPEN_S`, so a run too big for the budget cannot thrash)."""
        from sparc.studio.engine.host import RECYCLE_JSON

        svc = self.service()
        path = Path(svc.client.engine_dir) / RECYCLE_JSON
        info = read_json(path)
        if not isinstance(info, dict):
            return
        # the host writes the marker just before it exits: let that host go first
        _wait_for(lambda: (svc.client.info() or {}).get("pid") != info.get("pid"), 10.0)
        try:
            path.unlink()
        except OSError:
            pass
        mru = info.get("mru_run_id")
        now = time.monotonic()
        if not mru or now - self.reopened.get(mru, -1e9) < RECYCLE_REOPEN_S or self.loop is None:
            return
        self.reopened[mru] = now
        log.warning("the engine host recycled (RSS %s MB); re-opening %s", info.get("rss_mb"), mru)
        asyncio.run_coroutine_threadsafe(self._reopen(mru), self.loop)

    async def _reopen(self, run_id: str) -> None:
        if self.sctx is None or self.sctx.jobs is None:
            return
        svc = self.service()
        if svc.open_job(run_id) is not None:
            return
        try:
            await self.sctx.jobs.submit("engine.open", {"run_id": run_id}, run_id=run_id,
                                        label="Re-open engine after recycle")
        except Exception:
            log.exception("re-opening %s after a recycle failed", run_id)

    # ------------------------------------------------------------------ wait

    async def wait(self, job: dict) -> ExitInfo:
        job_dir = Path(job["job_dir"])
        jid = job["id"]
        res_path = job_dir / "result.json"
        while True:
            if res_path.exists():
                res = read_json(res_path) or {}
                with self._lock:
                    self.threads.pop(jid, None)
                    self.killed.discard(jid)
                code = res.get("exit_code")
                return ExitInfo(int(code) if code is not None else (0 if res.get("status") == "succeeded" else 1))
            with self._lock:
                th = self.threads.get(jid)
            if th is None:
                # a reattached job: the host serves it; it is lost when the host is gone
                if not self.service().client.alive():
                    await asyncio.sleep(self.poll_s)
                    if not res_path.exists():
                        write_job_result(job_dir, "cancelled" if jid in self.killed else "failed",
                                         130 if jid in self.killed else 1,
                                         error=None if jid in self.killed else
                                         {"type": "EngineExit", "message": "engine host exited"})
                    continue
            elif not th.is_alive() and not res_path.exists():
                write_job_result(job_dir, "failed", 1, error={"type": "EngineError",
                                                              "message": "the engine request ended without a result"})
                continue
            await asyncio.sleep(self.poll_s)

    # ------------------------------------------------------------------ cancel, kill, reattach

    async def cancel(self, job: dict) -> None:
        """The cancel file is already there: the host stops between folds."""
        return None

    async def kill(self, job: dict) -> None:
        self.killed.add(job["id"])
        await asyncio.to_thread(self.service().restart)

    async def reattach(self, job: dict) -> bool:
        job_dir = Path(job["job_dir"])
        if (job_dir / "result.json").exists():
            return True
        st = await asyncio.to_thread(self.service().host_status, fresh=True)
        if not st:
            return False
        busy = st.get("busy") or {}
        if busy.get("job_id") == job["id"]:
            return True
        # the request may be waiting for the host's work lock right behind another one
        return False


ENGINE_EXECUTOR = EngineExecutor()
register_executor("engine", ENGINE_EXECUTOR)


def _wait_for(pred, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())
