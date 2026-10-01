"""Job worker: ``python -m sparc.studio.jobs.worker <job_dir>`` (api.md §14).

1. Read ``job.json``.
2. ``progress.configure_from_env()``, ``install_signal_handlers()`` and
   ``set_threads(job.threads)`` (env + threadpoolctl + torch when loaded).
3. Import the kind's module (``job.json`` ``module``; else every module of
   ``KIND_MODULES``).
4. Call the registered library function with ``(JobContext, params)``.
5. Emit ``job.result`` and write ``result.json``
   ``{status, exit_code, result, error, finished_utc}`` atomically.

Exit codes: 0 succeeded, 1 failed (``error`` filled), 130 cancelled.
Workers never write SQLite; the server turns ``events.jsonl`` and
``result.json`` into rows.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_CANCELLED = 0, 1, 130


def _error(exc: BaseException) -> dict:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-12:])
    return {"type": type(exc).__name__, "message": str(exc)[:2000], "traceback_tail": tb[-3000:]}


def write_result(job_dir: Path, status: str, exit_code: int, result: dict | None, error: dict | None) -> None:
    from sparc.studio.workspace import utc_now, write_json_atomic

    write_json_atomic(job_dir / "result.json", {"status": status, "exit_code": exit_code, "result": result,
                                                "error": error, "finished_utc": utc_now()})


def run(job_dir: str | os.PathLike) -> int:
    from sparc.core import progress
    from sparc.studio.jobs import kinds

    job_dir = Path(job_dir).resolve()
    job = json.loads((job_dir / "job.json").read_text("utf-8"))
    progress.configure_from_env()
    progress.install_signal_handlers()
    progress.set_threads(int(job.get("threads") or 1))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    progress.emit("log", logger="sparc.studio.worker", level="INFO",
                  msg=f"worker started: {job.get('kind')} (pid {os.getpid()}, threads {job.get('threads') or 1})")
    status, code, result, error = "failed", EXIT_FAILED, None, None
    try:
        kind = kinds.get_kind(job["kind"])
        if kind is None and job.get("module"):
            kinds.load_kind_modules(only=job["module"])
            kind = kinds.get_kind(job["kind"])
        if kind is None:
            kinds.load_kind_modules()
            kind = kinds.get_kind(job["kind"])
        if kind is None or kind.fn is None:
            raise LookupError(f"unknown job kind {job['kind']!r}")
        params = kind.validate_params(job.get("params") or {})
        ctx = kinds.JobContext(job_dir, job)
        scope = progress.run_dir_scope(ctx.run_dir) if ctx.run_dir else nullcontext()
        with scope:
            out = kind.fn(ctx, params)
        result = out if isinstance(out, dict) else (ctx.result or {})
        status, code = "succeeded", EXIT_OK
    except progress.Cancelled:
        status, code = "cancelled", EXIT_CANCELLED
    except KeyboardInterrupt:
        status, code = "cancelled", EXIT_CANCELLED
    except BaseException as exc:            # noqa: BLE001 - the worker reports every failure
        error = _error(exc)
        status, code = "failed", EXIT_FAILED
        logging.getLogger("sparc.studio.worker").error("job failed: %s", error["message"])
    if status == "succeeded":
        progress.emit("job.result", result=result or {})
    try:
        write_result(job_dir, status, code, result, error)
    except Exception:                       # result.json is best effort; the exit code still tells the story
        traceback.print_exc()
    progress.reset()
    return code


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("usage: python -m sparc.studio.jobs.worker <job_dir>", file=sys.stderr)
        return 2
    t0 = time.time()
    try:
        return run(argv[0])
    finally:
        sys.stderr.write(f"worker exit after {time.time() - t0:.1f} s\n")


if __name__ == "__main__":
    sys.exit(main())
