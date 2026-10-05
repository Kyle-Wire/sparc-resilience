"""Self-test job kinds (SPEC §10.3; enabled by ``SPARC_STUDIO_TEST_KINDS=1``).

* ``test.sleep`` - sleeps ``seconds`` in small steps, ticking and honouring cancel
  (with ``n_stages``: as that many stage spans).
* ``test.events`` - a schema-complete synthetic run: run/stage/task spans,
  ``run.plan`` with units, ticks (engine passes with ``pass_s``), tagged and
  untagged metrics, artifacts, checkpoints, deduplicated warnings, a bridged
  log warning, cached / disabled / not-requested stages and ``run.end``.
  ``tests/studio/fixtures/selftest_events.jsonl`` was written by it
  (``python -m sparc.studio.jobs.testkinds --write-fixture PATH``).
* ``test.fail`` - emits a little, then raises.
* ``test.ignore_sigterm`` - ignores SIGTERM/SIGINT (and so do its ``workers``
  child processes): only Force stop ends it.
* ``test.pool`` - a ``ProcessPoolExecutor`` with ``progress.init_worker``
  whose tasks tick and check cancel.
* ``test.lock`` - ``test.sleep`` on the medium lane that takes its run's write lock.

Params for all: ``{seconds?, n_stages?, message?, workers?}``; result ``{ok: true}``.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sparc.core import progress
from sparc.studio.jobs.kinds import JobContext, job_kind

log = logging.getLogger("sparc.selftest")

STEP_S = 0.05


class TestParams(BaseModel):
    """Params of every ``test.*`` kind."""

    __test__ = False                      # not a pytest class
    model_config = ConfigDict(extra="forbid")

    seconds: float = Field(1.0, ge=0, le=86_400)
    n_stages: int | None = Field(None, ge=0, le=11)
    message: str | None = None
    workers: int | None = Field(None, ge=0, le=16)


def _sleep(seconds: float, *, unit: str = "sleep_step", check: bool = True) -> None:
    steps = max(1, int(round(seconds / STEP_S)))
    dt = seconds / steps
    for k in range(1, steps + 1):
        if check:
            progress.check_cancel()
        time.sleep(dt)
        progress.tick(k, steps, unit=unit, label="sleeping")
    if check:
        progress.check_cancel()


_STAGES = ("S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish")


@job_kind("test.sleep", lane="heavy", label="Self-test: sleep", params=TestParams)
def sleep_kind(ctx: JobContext, params: TestParams) -> dict:
    if params.n_stages:
        per = params.seconds / params.n_stages
        for sid in _STAGES[: params.n_stages]:
            with progress.stage(sid, label=f"sleep {sid}"):
                _sleep(per)
    else:
        with progress.task("sleep", n=1, k=1):
            _sleep(params.seconds)
    return {"ok": True, "message": params.message}


@job_kind("test.lock", lane="medium", label="Self-test: run lock", params=TestParams, needs_run=True, locks_run=True)
def lock_kind(ctx: JobContext, params: TestParams) -> dict:
    with progress.task("locked_sleep", key=ctx.run_id):
        _sleep(params.seconds)
    return {"ok": True, "run_id": ctx.run_id}


@job_kind("test.fail", lane="heavy", label="Self-test: failure", params=TestParams)
def fail_kind(ctx: JobContext, params: TestParams) -> dict:
    with progress.stage("S0", label="Load & QA"):
        progress.warn("selftest.about_to_fail", "the self-test is about to fail")
        _sleep(min(params.seconds, 0.2))
        raise RuntimeError(params.message or "test.fail: deliberate failure")


def _ignore_cancel_signals() -> None:
    """Ignore what Studio's cancel sends: SIGTERM / SIGINT, and CTRL_BREAK (SIGBREAK) on Windows."""
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)


def _ignore_and_sleep(seconds: float) -> None:
    _ignore_cancel_signals()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(0.05)


@job_kind("test.ignore_sigterm", lane="heavy", label="Self-test: ignores SIGTERM", params=TestParams)
def ignore_sigterm_kind(ctx: JobContext, params: TestParams) -> dict:
    import multiprocessing as mp

    _ignore_cancel_signals()
    children = []
    mpctx = mp.get_context("spawn")
    for _ in range(params.workers or 0):
        p = mpctx.Process(target=_ignore_and_sleep, args=(params.seconds,), daemon=False)
        p.start()
        children.append(p)
    progress.emit("log", logger="sparc.selftest", level="INFO",
                  msg=f"ignoring SIGTERM; children {[c.pid for c in children]}")
    with progress.task("stubborn", n=1, k=1):
        _sleep(params.seconds, check=False)
    for p in children:
        p.kill()
        p.join(5)
    return {"ok": True, "children": [c.pid for c in children]}


def _pool_task(i: int, seconds: float) -> int:
    with progress.task("pool_task", key=str(i), unit="pool_task"):
        _sleep(seconds, unit="pool_step")
    return os.getpid()


@job_kind("test.pool", lane="heavy", label="Self-test: process pool", params=TestParams)
def pool_kind(ctx: JobContext, params: TestParams) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    workers = params.workers or 2
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=progress.init_worker, initargs=(progress.worker_env(),))
    pids: set[int] = set()
    try:
        futures = [ex.submit(_pool_task, i, params.seconds) for i in range(workers * 2)]
        done = 0
        for fut in as_completed(futures):
            progress.check_cancel()
            pids.add(fut.result())
            done += 1
            progress.tick(done, len(futures), unit="pool_tasks")
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return {"ok": True, "workers": sorted(pids)}


# ---------------------------------------------------------------------------
# test.events
# ---------------------------------------------------------------------------

K_FOLDS = 3
MODELS = ("ols", "gam")
CANDIDATES = ("mean", "nnls")


def _plan(n_stages: int | None) -> list[dict]:
    nodes = [
        {"id": "S0", "label": "Load & QA", "state": "will_run", "reason": None, "units": {"s0_load": 1}},
        {"id": "S1", "label": "Influence ranges", "state": "will_run", "reason": None, "units": {"s1_influence": 1}},
        {"id": "S2_S3", "label": "Base models & stacker", "state": "will_run", "reason": None,
         "units": {**{f"base_fit:{m}": K_FOLDS for m in MODELS}, **{f"stacker_fit:{c}": K_FOLDS for c in CANDIDATES},
                   "checkpoint_save": 1}, "checkpoint_key": "S3"},
        {"id": "baselines", "label": "Baselines", "state": "skipped", "reason": "disabled_by_config:baselines.enabled",
         "units": {}, "checkpoint_key": "baselines"},
        {"id": "cv_curve", "label": "CV distance curve", "state": "skipped", "reason": "not_requested", "units": {},
         "checkpoint_key": "cv_curve"},
        {"id": "S4", "label": "Response curves", "state": "will_run", "reason": None, "units": {"engine_pass": 3},
         "checkpoint_key": "S4"},
        {"id": "S5", "label": "Scenarios", "state": "will_run", "reason": None, "units": {"engine_pass": 2},
         "checkpoint_key": "S5"},
        {"id": "climate", "label": "Climate", "state": "skipped", "reason": "disabled_by_config:climate.source",
         "units": {}, "checkpoint_key": "climate"},
        {"id": "S6", "label": "Causal validation", "state": "skipped", "reason": "no_treatments", "units": {},
         "checkpoint_key": "S6"},
        {"id": "S7", "label": "Budget optimisation", "state": "will_run", "reason": None,
         "units": {"engine_pass": 1, "pareto": 1}},
        {"id": "finish", "label": "Documents", "state": "will_run", "reason": None, "units": {}},
    ]
    if n_stages is not None:
        will = [n for n in nodes if n["state"] == "will_run"]
        for n in will[n_stages:]:
            n.update(state="skipped", reason="not_requested")
    return nodes


def _write(run_dir: Path, rel: str, text: str, *, role: str) -> None:
    p = run_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    progress.artifact(p, role=role)


def _engine_pass(pause: float) -> None:
    t0 = time.perf_counter()
    for k in range(1, K_FOLDS + 1):
        progress.check_cancel()
        time.sleep(pause)
        if k == K_FOLDS:
            progress.tick(k, K_FOLDS, unit="engine_pass", pass_s=round(time.perf_counter() - t0, 4))
        else:
            progress.tick(k, K_FOLDS, unit="engine_pass")


def emit_selftest_run(run_dir: Path, seconds: float = 1.0, n_stages: int | None = None) -> dict:
    """The ``test.events`` body (usable in-process with any progress sink)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan(n_stages)
    will = {n["id"] for n in plan if n["state"] == "will_run"}
    pause = max(seconds, 0.0) / 60.0
    timings: dict[str, float] = {}
    total_units: dict[str, float] = {}
    for n in plan:
        if n["state"] == "will_run":
            for u, c in n["units"].items():
                total_units[u] = total_units.get(u, 0) + c
    with progress.run_dir_scope(run_dir), progress.span(
            "run", "selftest", stages=list(_STAGES[:-1]), fast=True, coarse=None, resume=False, cv_curve=False,
            config_sha256="0" * 64, code_sha256="1" * 64,
            run_meta={"studio_run_id": None, "origin": "selftest"}) as run:
        progress.emit("run.plan", nodes=plan, total_units=total_units, n_points=40)
        progress.emit("run.dir", run_dir=str(run_dir), fingerprint="0123456789abcdef")
        for node in plan:
            sid = node["id"]
            if sid not in will:
                progress.skip(sid, node["reason"])
                continue
            t0 = time.perf_counter()
            with progress.stage(sid, label=node["label"]) as st:
                progress.check_cancel()
                if sid == "S0":
                    time.sleep(pause)
                    progress.warn("qa.coarse", "coarse cells: 40 points on a 3 x 3 km demo grid", cell_m=90)
                    progress.warn("qa.coarse", "coarse cells: 40 points on a 3 x 3 km demo grid", cell_m=90)
                    _write(run_dir, "qa.json", '{"n_points": 40}\n', role="qa")
                    st.summary = {"n_points": 40, "n_input": 42, "n_dropped": 2, "grid_shape": [7, 6], "cell_m": 90.0}
                elif sid == "S1":
                    for pred, rng in (("Pct_Canopy", 310.0), ("Pct_Impervious", 455.0)):
                        progress.metric("influence.range_m", rng, unit="m", predictor=pred)
                        progress.metric("influence.anisotropy_ratio", 1.2, predictor=pred)
                    progress.warn("influence.few_cells", "few cells for Pct_Albedo; range set to the prior",
                                  predictor="Pct_Albedo")
                    _write(run_dir, "influence.json", '{"ranges": {}}\n', role="influence")
                    time.sleep(pause)
                    st.summary = {"block_size_m": 900.0, "L_prior_m": 300.0}
                elif sid == "S2_S3":
                    for k in range(1, K_FOLDS + 1):
                        with progress.task("fold", k=k, n=K_FOLDS):
                            for j, m in enumerate(MODELS):
                                with progress.task("base_model", key=m, unit=f"base_fit:{m}") as t:
                                    progress.check_cancel()
                                    time.sleep(pause)
                                    t.metrics = {"fit_s": round(pause, 4), "heldout_rmse": round(1.0 + 0.1 * j
                                                                                                 + 0.01 * k, 4),
                                                 "heldout_r2": round(0.6 - 0.05 * j, 4)}
                    for c, cand in enumerate(CANDIDATES, start=1):
                        with progress.task("stacker_candidate", k=c, n=len(CANDIDATES), key=cand):
                            for k in range(1, K_FOLDS + 1):
                                with progress.task("stacker_fold", k=k, n=K_FOLDS, unit=f"stacker_fit:{cand}"):
                                    progress.check_cancel()
                                    time.sleep(pause / 2)
                            progress.metric("candidate_rmse", round(0.9 + 0.05 * c, 4), candidate=cand)
                    progress.metric("stacker_rmse", 0.95, unit="F")
                    progress.metric("stacker_r2", 0.66)
                    progress.metric("interval_coverage", 0.9)
                    _write(run_dir, "predictions.csv", "id,pred\n1,80.1\n", role="predictions")
                    progress.checkpoint("saved", done=["S3"], bytes=1024, elapsed_s=0.01, fingerprint="0123456789abcdef")
                    log.warning("stacker weights are close to uniform")
                elif sid == "S4":
                    with progress.task("engine_init"):
                        _engine_pass(pause / K_FOLDS)
                    with progress.task("variable", k=1, n=1, key="Pct_Canopy"):
                        for d in (1, 2):
                            with progress.task("dose", k=d, n=2) as t:
                                _engine_pass(pause / K_FOLDS)
                                t.metrics = {"mean_benefit": round(0.1 * d, 4), "mean_se": 0.02,
                                             "frac_extrapolated": 0.0}
                    _write(run_dir, "response_curves.json", '{"Pct_Canopy": {}}\n', role="response")
                    progress.checkpoint("saved", done=["S3", "S4"], bytes=2048, elapsed_s=0.01,
                                        fingerprint="0123456789abcdef")
                elif sid == "S5":
                    for s, name in enumerate(("Canopy +10", "Albedo +0.1"), start=1):
                        with progress.task("scenario", k=s, n=2, key=name):
                            _engine_pass(pause / K_FOLDS)
                        progress.metric("scenario.mean_delta", round(-0.2 * s, 4), unit="F", scenario=name)
                        progress.metric("scenario.se", 0.05, unit="F", scenario=name)
                    _write(run_dir, "scenarios.json", '{"scenarios": []}\n', role="scenarios")
                elif sid == "S7":
                    with progress.task("allocate"):
                        _engine_pass(pause / K_FOLDS)
                    with progress.task("pareto", unit="pareto"):
                        time.sleep(pause / 2)
                    _write(run_dir, "optimize.json", '{"pareto": []}\n', role="optimize")
                elif sid == "finish":
                    _write(run_dir, "report.md", "# Self-test\n", role="docs")
                    _write(run_dir, "manifest.json", '{"schema_version": 2}\n', role="manifest")
            timings[sid] = round(time.perf_counter() - t0, 4)
        run.set(timings_s=timings, done=["S3", "S4"])
    return {"ok": True, "timings_s": timings}


@job_kind("test.events", lane="heavy", label="Self-test: event stream", params=TestParams)
def events_kind(ctx: JobContext, params: TestParams) -> dict:
    return emit_selftest_run(ctx.job_dir / "run", params.seconds, params.n_stages)


# ---------------------------------------------------------------------------
# fixture generator
# ---------------------------------------------------------------------------

def write_fixture(path: str | os.PathLike, seconds: float = 1.0) -> Path:
    """Write ``selftest_events.jsonl``: a ``test.events`` run plus the server's ``job.status`` lines."""
    import tempfile

    from sparc.studio.events import append_event

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    jid = "j_selftest"
    t0 = time.time()
    for status in ("queued", "starting"):
        append_event(path, "job.status", job_id=jid, t0=t0, status=status, exit_code=None, error=None)
    progress.reset()
    logging.basicConfig(level=logging.INFO)
    progress.configure(str(path), job_id=jid, heartbeat_s=max(seconds / 4, 0.05))
    with tempfile.TemporaryDirectory() as tmp:
        progress.emit("log", logger="sparc.studio.worker", level="INFO", msg="worker started: test.events")
        append_event(path, "job.status", job_id=jid, t0=t0, status="running", exit_code=None, error=None)
        result = emit_selftest_run(Path(tmp) / "run", seconds)
        progress.emit("job.result", result=result)
    progress.reset()
    append_event(path, "job.status", job_id=jid, t0=t0, status="succeeded", exit_code=0, error=None)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sparc.studio.jobs.testkinds")
    ap.add_argument("--write-fixture", metavar="PATH", required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args(argv)
    print(write_fixture(args.write_fixture, args.seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
