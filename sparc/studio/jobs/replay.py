"""Replay runner (api.md §14; tests and e2e only).

``python -m sparc.studio.jobs.replay <fixture_dir> <job_dir> [--speed 20]``
stands in for the worker of ``run.core`` jobs when the server runs with
``SPARC_STUDIO_RUNNER=replay:<fixture_dir>``.  The fixture is a recorded run
(``tests/studio/fixtures/synth_run/``: the run directory's files plus its
``events.jsonl`` and ``FIXTURE.json``).  The runner:

* writes the fixture events to the job's ``events.jsonl`` with fresh ``job``,
  ``pid`` and ``ts`` values, keeping their relative timing compressed by
  ``--speed`` (gaps are capped at 2 s);
* copies each output into the new run directory just before its
  ``artifact`` event, rewrites ``run.dir.run_dir``, and keeps
  ``run_state.json`` (``events_path``, ``pid``, ``job``, stage, done) and
  ``checkpoint.json`` (on every ``checkpoint{saved}``) current;
* on cancel (SIGTERM or the cancel file) stops at the next event boundary,
  writes ``run_state.status = "cancelled"``, emits ``cancel.ack`` and
  ``run.end{cancelled}`` and exits 130;
* on resume (``params.resume``) reads the done set of the last saved
  checkpoint, marks those stages cached in ``run.plan``, emits
  ``stage.skip{checkpoint}`` for them and replays the remaining events.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from sparc.studio.events import encode_line

__all__ = ["replay", "main", "CHECKPOINT_KEY", "cached_stages"]

#: stage id → the checkpoint done-set entry that restores it (``sparc.core.pipeline.CHECKPOINT_KEY``; this copy
#: is the fallback when core cannot be imported).  S1 and S2_S3 both come back with "S3".
CHECKPOINT_KEY = {"S1": "S3", "S2_S3": "S3", "baselines": "baselines", "cv_curve": "cv_curve", "S4": "S4",
                  "S5": "S5", "climate": "climate", "S6": "S6"}
MAX_GAP_S = 2.0
EXIT_OK, EXIT_FAILED, EXIT_CANCELLED = 0, 1, 130

_SIGNALED = False


def _on_term(signum, frame):
    global _SIGNALED
    _SIGNALED = True


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj) -> None:
    from sparc.studio.workspace import write_json_atomic

    write_json_atomic(path, obj)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def cached_stages(done) -> set[str]:
    """Stages a resume restores from a checkpoint whose done set is ``done`` (core's own mapping)."""
    try:
        from sparc.core.pipeline import CHECKPOINT_KEY as mapping
    except Exception:                        # core unavailable: the documented mapping
        mapping = CHECKPOINT_KEY
    done = set(done or ())
    return {sid for sid, key in dict(mapping).items() if key in done}


def _stage_of(ev: dict) -> str | None:
    if ev.get("type") in ("stage.start", "stage.end", "stage.skip"):
        return ev.get("stage")
    path = ev.get("span_path") if ev.get("type") == "artifact" else ev.get("path")
    for el in path or ():
        if isinstance(el, str) and el.startswith("stage:"):
            return el[6:].split("[", 1)[0]
    return None


def _load_events(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("type") not in ("job.status", "cancel.requested"):
                out.append(ev)
    return out


class _Writer:
    def __init__(self, events_path: Path, job_id: str):
        self.path = events_path
        self.job_id = job_id
        self.fd = os.open(events_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        self.t0 = time.time()

    def write(self, ev: dict) -> None:
        ev = dict(ev)
        now = time.time()
        ev["job"] = self.job_id
        ev["pid"] = os.getpid()
        ev["ts"] = round(now, 3)
        ev["t_rel"] = round(now - self.t0, 3)
        os.write(self.fd, encode_line(ev))

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def _cancel_requested(cancel_file: Path) -> bool:
    return _SIGNALED or cancel_file.exists()


def _metrics(root: Path) -> dict | None:
    try:
        manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    m = manifest.get("metrics") if isinstance(manifest, dict) else None
    if not isinstance(m, dict):
        return None
    pick = {}
    for key, names in (("r2", ("r2", "stacker_r2", "oof_r2")), ("rmse", ("rmse", "stacker_rmse", "oof_rmse")),
                       ("coverage", ("coverage", "interval_coverage"))):
        for n in names:
            v = m.get(n)
            if isinstance(v, (int, float)):
                pick[key] = v
                break
        else:
            pick[key] = None
    return pick


def replay(fixture_dir: str | os.PathLike, job_dir: str | os.PathLike, *, speed: float = 20.0) -> int:
    """Replay the fixture into the job (see the module docstring); returns the exit code."""
    from sparc.studio.jobs.kinds import JobContext
    from sparc.studio.jobs.worker import write_result

    fixture_dir = Path(fixture_dir).resolve()
    job_dir = Path(job_dir).resolve()
    root = fixture_dir / "run" if (fixture_dir / "run").is_dir() else fixture_dir
    ctx = JobContext(job_dir)
    params = ctx.params
    if ctx.run_dir is None:
        write_result(job_dir, "failed", EXIT_FAILED, None,
                     {"type": "ReplayError", "message": "the job has no run directory"})
        return EXIT_FAILED
    run_dir = Path(ctx.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events = _load_events(fixture_dir / "events.jsonl")
    events_path = job_dir / "events.jsonl"
    cancel_file = job_dir / "cancel"
    out = _Writer(events_path, ctx.job_id)
    resume = bool(params.get("resume"))
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    fixture_state = {}
    try:
        fixture_state = json.loads((root / "run_state.json").read_text("utf-8"))
    except (OSError, ValueError):
        pass
    fixture_ckpt = {}
    try:
        fixture_ckpt = json.loads((root / "checkpoint.json").read_text("utf-8"))
    except (OSError, ValueError):
        pass

    done: list[str] = []
    if resume:
        try:
            done = list(json.loads((run_dir / "checkpoint.json").read_text("utf-8")).get("done") or [])
        except (OSError, ValueError):
            done = []
    cached = cached_stages(done)
    run_state = {**{k: v for k, v in fixture_state.items() if k in ("schema", "host", "meta", "fingerprint")},
                 "schema": fixture_state.get("schema", 1), "status": "running", "pid": os.getpid(),
                 "job": ctx.job_id, "started_utc": _utc(), "updated_utc": _utc(), "stage": None, "done": list(done),
                 "events_path": str(events_path), "error": None}
    _write_json(run_dir / "run_state.json", run_state)

    run_span = run_path = None
    name = "run"
    t_first = next((e.get("ts") for e in events if isinstance(e.get("ts"), (int, float))), time.time())
    wall0 = time.monotonic()
    last_ts = t_first
    sim = 0.0
    timings: dict = {}
    loaded_sent = False

    def finish_cancelled(at_path) -> int:
        out.write({"v": 1, "type": "cancel.ack", "seq": 0, "lvl": "info", "span": run_span, "parent": None,
                   "path": at_path or [], "ctx": {}, "at_path": at_path or []})
        out.write({"v": 1, "type": "run.end", "seq": 0, "lvl": "info", "span": run_span, "parent": None,
                   "path": run_path or [], "ctx": {}, "status": "cancelled", "elapsed_s": round(time.time() - out.t0, 3),
                   "timings_s": timings, "done": list(done), "error": None})
        run_state.update(status="cancelled", updated_utc=_utc(), done=list(done))
        _write_json(run_dir / "run_state.json", run_state)
        write_result(job_dir, "cancelled", EXIT_CANCELLED, None, None)
        out.close()
        return EXIT_CANCELLED

    current_path: list = []
    for ev in events:
        if _cancel_requested(cancel_file):
            return finish_cancelled(current_path)
        ts = ev.get("ts")
        if isinstance(ts, (int, float)):
            gap = max(0.0, ts - last_ts) / max(speed, 1e-6)
            sim += min(gap, MAX_GAP_S)
            last_ts = ts
            wait = sim - (time.monotonic() - wall0)
            while wait > 0:
                if _cancel_requested(cancel_file):
                    return finish_cancelled(current_path)
                time.sleep(min(wait, 0.05))
                wait = sim - (time.monotonic() - wall0)
        t = ev.get("type")
        stage = _stage_of(ev)
        if t == "run.start" and run_span is None:
            run_span, run_path, name = ev.get("span"), ev.get("path"), ev.get("name") or name
            ev = {**ev, "resume": resume}
        if stage in cached:
            if t == "stage.start":
                out.write({**{k: ev[k] for k in ("v", "seq", "lvl", "span", "parent", "ctx") if k in ev},
                           "type": "stage.skip", "path": run_path or [], "stage": stage, "reason": "checkpoint",
                           "span": run_span, "parent": None})
            continue
        if t == "run.plan" and cached:
            nodes = []
            for n in ev.get("nodes") or []:
                if n.get("id") in cached:
                    n = {**n, "state": "cached", "reason": "checkpoint"}
                nodes.append(n)
            totals: dict = {}
            for n in nodes:
                if n.get("state", "will_run") == "will_run":
                    for u, c in (n.get("units") or {}).items():
                        totals[u] = totals.get(u, 0) + c
            ev = {**ev, "nodes": nodes, "total_units": totals}
        elif t == "run.dir":
            ev = {**ev, "run_dir": str(run_dir)}
        elif t == "artifact" and isinstance(ev.get("path"), str):
            src = root / ev["path"]
            if src.exists() and ev["path"] not in ("run_state.json", "checkpoint.json"):
                _copy(src, run_dir / ev["path"])
        elif t == "checkpoint" and ev.get("action") in ("loaded", "mismatch"):
            if resume:
                continue                  # the replay emits its own "loaded" for the resumed checkpoint
        if resume and cached and not loaded_sent and t == "stage.start":
            out.write({"v": 1, "type": "checkpoint", "seq": 0, "lvl": "info", "span": run_span, "parent": None,
                       "path": run_path or [], "ctx": {}, "action": "loaded", "done": sorted(done), "bytes": None,
                       "elapsed_s": 0.0, "fingerprint": fixture_ckpt.get("fingerprint"), "changed_sections": []})
            loaded_sent = True
        if t == "run.end":
            break
        out.write(ev)
        if t in ("stage.start", "task.start"):
            current_path = list(ev.get("path") or [])
        if t == "stage.start":
            run_state.update(stage=ev.get("stage"), updated_utc=_utc())
            _write_json(run_dir / "run_state.json", run_state)
        elif t == "stage.end":
            if ev.get("elapsed_s") is not None:
                timings[ev.get("stage")] = ev["elapsed_s"]
            run_state.update(updated_utc=_utc(), done=list(done))
            _write_json(run_dir / "run_state.json", run_state)
        elif t == "checkpoint" and ev.get("action") == "saved":
            done = sorted(set(done) | set(ev.get("done") or []))
            ckpt = {**fixture_ckpt, "schema": fixture_ckpt.get("schema", 1), "done": list(done),
                    "saved_utc": _utc(), "bytes": ev.get("bytes") or fixture_ckpt.get("bytes")}
            _write_json(run_dir / "checkpoint.json", ckpt)
            run_state.update(done=list(done), updated_utc=_utc())
            _write_json(run_dir / "run_state.json", run_state)

    if _cancel_requested(cancel_file):
        return finish_cancelled(current_path)
    final = next((e for e in reversed(events) if e.get("type") == "run.end"), None)
    status = (final or {}).get("status", "succeeded")
    end = {**(final or {"v": 1, "type": "run.end", "seq": 0, "lvl": "info", "span": run_span, "parent": None,
                        "path": run_path or [], "ctx": {}, "error": None}),
           "status": status, "elapsed_s": round(time.time() - out.t0, 3),
           "timings_s": (final or {}).get("timings_s") or timings, "done": (final or {}).get("done") or list(done)}
    out.write(end)
    run_state.update(status=status, updated_utc=_utc(), stage=fixture_state.get("stage", "finish"),
                     done=end["done"], error=(final or {}).get("error"))
    _write_json(run_dir / "run_state.json", run_state)
    result = {"run_id": ctx.run_id, "status": status, "timings_s": end["timings_s"], "metrics": _metrics(root),
              "done": end["done"]}
    code = EXIT_OK if status == "succeeded" else EXIT_FAILED
    out.write({"v": 1, "type": "job.result", "seq": 0, "lvl": "info", "span": None, "parent": None, "path": [],
               "ctx": {}, "result": result})
    out.close()
    write_result(job_dir, "succeeded" if code == EXIT_OK else "failed", code, result,
                 None if code == EXIT_OK else (final or {}).get("error"))
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sparc.studio.jobs.replay")
    ap.add_argument("fixture_dir")
    ap.add_argument("job_dir")
    ap.add_argument("--speed", type=float, default=20.0)
    args = ap.parse_args(argv)
    try:
        return replay(args.fixture_dir, args.job_dir, speed=args.speed)
    except Exception as exc:                 # report like the worker does
        from sparc.studio.jobs.worker import write_result

        write_result(Path(args.job_dir), "failed", EXIT_FAILED, None,
                     {"type": type(exc).__name__, "message": str(exc)[:2000]})
        raise


if __name__ == "__main__":
    sys.exit(main())
