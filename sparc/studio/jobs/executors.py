"""Executors: how a job's work actually runs (SPEC §5.6, §10.2).

An executor implements four coroutines - ``submit(job)``, ``cancel(job)``,
``kill(job)``, ``reattach(job)`` - plus ``wait(job)``, which resolves when the
job's work has ended.  ``job`` is the ``jobs`` row as a dict (``job_dir``,
``threads``, ``pid`` …).  Two implementations exist:

* :class:`ProcessExecutor` (here): ``python -m sparc.studio.jobs.worker
  <job_dir>`` in its own session / process group, with the progress sink,
  job id, cancel file and thread env of SPEC §5.6; or the replay runner for
  ``run.core`` when ``SPARC_STUDIO_RUNNER=replay:<dir>``.
* ``EngineExecutor`` (backend-engine), registered at import with
  :func:`register_executor`.

Executors may define ``bind(sctx)``; the job manager calls it at start-up
with the server context.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sparc.core import progress

log = logging.getLogger("sparc.studio.jobs")

__all__ = ["ExitInfo", "Executor", "ProcessExecutor", "register_executor", "registered_executors",
           "worker_env", "REPLAY_SPEED_ENV", "spawn_command"]

REPLAY_SPEED_ENV = "SPARC_STUDIO_REPLAY_SPEED"
_IS_WIN = os.name == "nt"


@dataclass
class ExitInfo:
    """How a job's work ended: ``code`` is the exit code (negative = killed by that signal; None = unknown,
    e.g. a reattached process that is not our child)."""
    code: int | None
    reattached: bool = False


@runtime_checkable
class Executor(Protocol):
    async def submit(self, job: dict) -> dict: ...

    async def cancel(self, job: dict) -> None: ...

    async def kill(self, job: dict) -> None: ...

    async def reattach(self, job: dict) -> bool: ...

    async def wait(self, job: dict) -> ExitInfo: ...


_REGISTRY: dict[str, Any] = {}


def register_executor(name: str, executor: Any) -> None:
    """Make ``executor`` available for kinds registered with ``executor=name``."""
    _REGISTRY[name] = executor


def registered_executors() -> dict[str, Any]:
    return dict(_REGISTRY)


def _sparc_root() -> str:
    """The directory holding the ``sparc`` package this server imported (so workers import the same code)."""
    import sparc

    return str(Path(sparc.__file__).resolve().parent.parent)


def worker_env(job: dict, *, test_kinds: bool = False, base: dict | None = None) -> dict:
    """Environment of a job process (SPEC §5.6)."""
    env = dict(os.environ if base is None else base)
    job_dir = Path(job["job_dir"])
    threads = str(max(1, int(job.get("threads") or 1)))
    env.pop(progress.ENV_LEVEL, None)
    env.update({
        progress.ENV_SINK: str(job_dir / "events.jsonl"),
        progress.ENV_JOB: job["id"],
        progress.ENV_CANCEL: str(job_dir / "cancel"),
        "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
        "PYTHONUNBUFFERED": "1",
    })
    if test_kinds:
        env["SPARC_STUDIO_TEST_KINDS"] = "1"
    root = _sparc_root()
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join([root, *paths])
    return env


def spawn_command(job: dict) -> list[str]:
    """``python -m sparc.studio.jobs.worker <job_dir>``, or the replay runner for ``run.core`` in replay mode."""
    job_dir = str(Path(job["job_dir"]))
    runner = os.environ.get("SPARC_STUDIO_RUNNER", "")
    if runner.startswith("replay:") and job.get("kind") == "run.core":
        fixture = runner.split(":", 1)[1]
        speed = os.environ.get(REPLAY_SPEED_ENV, "20")
        return [sys.executable, "-m", "sparc.studio.jobs.replay", os.path.abspath(fixture), job_dir, "--speed", speed]
    return [sys.executable, "-m", "sparc.studio.jobs.worker", job_dir]


class ProcessExecutor:
    """Runs each job as a subprocess in its own session (POSIX) or process group (Windows).

    ``command(job)`` builds the argv (default :func:`spawn_command`; tests
    replace it).  The job directory path is the *cmdline token*: reattach
    accepts a process only if its pid, ``create_time`` and command line all
    match ``state.json``.
    """

    name = "process"

    def __init__(self, *, test_kinds: bool = False, poll_s: float = 0.25, command=None):
        self.test_kinds = test_kinds
        self.poll_s = poll_s
        self.command = command or spawn_command
        self.procs: dict[str, Any] = {}

    async def submit(self, job: dict) -> dict:
        job_dir = Path(job["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.command(job)
        env = worker_env(job, test_kinds=self.test_kinds)
        out = open(job_dir / "stdout.log", "ab")
        err = open(job_dir / "stderr.log", "ab")
        try:
            kwargs: dict[str, Any] = {}
            if _IS_WIN:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            else:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, cwd=str(job_dir), env=env,
                                    close_fds=True, **kwargs)
        finally:
            out.close()
            err.close()
        self.procs[job["id"]] = proc
        info = {"pid": proc.pid, "pgid": proc.pid, "proc_create_time": None, "cmdline_token": str(job_dir),
                "executor": self.name}
        try:
            if not _IS_WIN:
                info["pgid"] = os.getpgid(proc.pid)
            import psutil

            info["proc_create_time"] = psutil.Process(proc.pid).create_time()
        except Exception:                    # the process may already have exited
            pass
        return info

    async def wait(self, job: dict) -> ExitInfo:
        proc = self.procs.get(job["id"])
        if proc is None:
            return ExitInfo(None)
        if isinstance(proc, subprocess.Popen):
            while proc.poll() is None:
                await asyncio.sleep(self.poll_s)
            self.procs.pop(job["id"], None)
            return ExitInfo(proc.returncode)
        # a reattached process: not our child (or a child of an earlier app in this process)
        while _alive(proc):
            await asyncio.sleep(max(self.poll_s, 1.0) if not _child_of_us(proc.pid) else self.poll_s)
        code = _reap(proc.pid)
        self.procs.pop(job["id"], None)
        return ExitInfo(code, reattached=True)

    async def cancel(self, job: dict) -> None:
        self._signal(job, signal.SIGTERM if not _IS_WIN else getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))

    async def kill(self, job: dict) -> None:
        pid = job.get("pid")
        tree: list = []
        try:
            import psutil

            if pid:
                tree = psutil.Process(int(pid)).children(recursive=True)
        except Exception:
            tree = []
        self._signal(job, signal.SIGKILL if not _IS_WIN else signal.SIGTERM, group_kill=True)
        for p in tree:                       # descendants that left the group (setsid in a library) die too
            try:
                p.kill()
            except Exception:
                pass

    def _signal(self, job: dict, sig, group_kill: bool = False) -> None:
        pid = job.get("pid")
        pgid = job.get("pgid") or pid
        if not pid:
            return
        try:
            if _IS_WIN:
                if group_kill:
                    proc = self.procs.get(job["id"])
                    if isinstance(proc, subprocess.Popen):
                        proc.kill()
                    else:
                        os.kill(int(pid), signal.SIGTERM)
                else:
                    os.kill(int(pid), sig)
            else:
                os.killpg(int(pgid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    async def reattach(self, job: dict) -> bool:
        """True (and remember the process) when ``job``'s pid is alive with the same create_time and token."""
        import psutil

        pid, ctime = job.get("pid"), job.get("proc_create_time")
        token = str(job.get("cmdline_token") or job.get("job_dir") or "")
        if not pid or ctime is None or not token:
            return False
        try:
            p = psutil.Process(int(pid))
            if abs(p.create_time() - float(ctime)) > 0.01:
                return False
            if p.status() == psutil.STATUS_ZOMBIE:
                return False
            if not any(token in part for part in p.cmdline()):
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
        self.procs[job["id"]] = p
        return True


def _alive(p) -> bool:
    import psutil

    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return True


def _child_of_us(pid: int) -> bool:
    try:
        import psutil

        return psutil.Process(pid).ppid() == os.getpid()
    except Exception:
        return False


def _reap(pid: int) -> int | None:
    """Collect the exit status when ``pid`` is our (zombie) child; None otherwise."""
    if _IS_WIN:
        return None
    try:
        wpid, status = os.waitpid(int(pid), os.WNOHANG)
    except ChildProcessError:
        return None
    except OSError:
        return None
    if wpid == 0:
        return None
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return None
