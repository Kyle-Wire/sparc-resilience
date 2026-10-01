"""sparc.core.progress: envelope, write discipline, span nesting, throttling, cancellation, bridges.

The spawn ProcessPool test imports this module in the workers, so module-level
code stays light (stdlib + pytest only).
"""

from __future__ import annotations

import ast
import json
import logging
import multiprocessing
import os
import signal
import statistics
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from sparc.core import progress

ENVELOPE = ("v", "type", "seq", "ts", "t_rel", "pid", "job", "lvl", "span", "parent", "path", "ctx")
ENV_KEYS = (progress.ENV_SINK, progress.ENV_LEVEL, progress.ENV_JOB, progress.ENV_CANCEL) + progress.THREAD_ENV


@pytest.fixture(autouse=True)
def clean_progress():
    """Each test starts and ends unconfigured, with the progress/thread env vars as they were."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS[:4]:
        os.environ.pop(k, None)
    progress.reset()
    yield
    progress.reset()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def read_events(path) -> list[dict]:
    """Parse every line strictly (no NaN/Infinity) and check the size limit and the envelope."""
    events = []
    raw = Path(path).read_bytes()
    assert raw == b"" or raw.endswith(b"\n")
    for line in raw.splitlines(keepends=True):
        assert len(line) <= progress.MAX_LINE, len(line)
        ev = json.loads(line, parse_constant=_reject_constant)
        missing = [k for k in ENVELOPE if k not in ev]
        assert not missing, (missing, ev)
        assert ev["v"] == progress.SCHEMA
        events.append(ev)
    return events


class Collect:
    """Callable sink."""

    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, ev):
        self.events.append(ev)

    def of(self, type_):
        return [e for e in self.events if e["type"] == type_]


# ---------------------------------------------------------------------------
# unconfigured: no-ops and cost
# ---------------------------------------------------------------------------

def test_unconfigured_calls_write_nothing(tmp_path):
    sink = Collect()
    progress.configure(sink, heartbeat_s=0)
    progress.reset()
    assert not progress.enabled()
    with progress.stage("S0") as st, progress.context(partition="p"), progress.run_dir_scope(tmp_path):
        st.summary["n"] = 1
        with progress.task("fold", k=1, n=2) as t:
            t.metrics["x"] = 1
            progress.tick(1, 2, unit="u")
            progress.metric("m", 1.0)
            progress.emit("custom", a=1)
            progress.artifact(tmp_path / "x", role="r")
            progress.checkpoint("saved", done=["S0"])
            progress.warn("qa.coarse", "w")
            progress.skip("S6", "not_requested")
            progress.check_cancel()
    logging.getLogger("sparc.core.x").warning("not bridged")
    assert sink.events == []
    assert not progress.cancel_requested()
    assert progress.sink_path() is None


def _noop_tick(k, n, *, unit, label="", lvl="info", **metrics):
    return None


def _noop_metric(name, value, *, unit=None, lvl="info", **tags):
    return None


def _noop_emit(type, /, *, lvl="info", **fields):
    return None


def _time_tick(fn, n):
    t0 = time.perf_counter()
    for i in range(n):
        fn(i, n, unit="engine_pass", label="fold", pass_s=1.5)
    return time.perf_counter() - t0


def _time_metric(fn, n):
    t0 = time.perf_counter()
    for i in range(n):
        fn("heldout_rmse", 0.5, unit="K", model="mgwr")
    return time.perf_counter() - t0


def _time_emit(fn, n):
    t0 = time.perf_counter()
    for i in range(n):
        fn("custom", lvl="debug", a=i, b="x")
    return time.perf_counter() - t0


def test_disabled_cost_is_at_most_twice_a_noop_call(tmp_path):
    """1e6 unconfigured calls of tick/metric/emit cost ≤ 2× a no-op of the same signature and write nothing."""
    sink = Collect()
    events = tmp_path / "events.jsonl"
    progress.configure(sink, heartbeat_s=0)
    progress.configure(events, heartbeat_s=0)
    progress.reset()
    size0 = events.stat().st_size
    n = 1_000_000
    for timer, ours, noop in ((_time_tick, progress.tick, _noop_tick),
                              (_time_metric, progress.metric, _noop_metric),
                              (_time_emit, progress.emit, _noop_emit)):
        t_ours, t_noop = [], []
        for _ in range(5):                  # interleaved, so CPU contention hits both alike
            t_ours.append(timer(ours, n))
            t_noop.append(timer(noop, n))
        ratio = statistics.median(t_ours) / statistics.median(t_noop)
        assert ratio <= 2.0, f"{ours.__name__}: {ratio:.2f}× a no-op call"
    assert sink.events == []
    assert events.stat().st_size == size0 == 0


def test_module_is_stdlib_only():
    tree = ast.parse(Path(progress.__file__).read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:                  # module-level imports only; psutil/threadpoolctl are function-local
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add(node.module.split(".")[0])
    assert top - {"__future__"} <= set(sys.stdlib_module_names), top - set(sys.stdlib_module_names)


# ---------------------------------------------------------------------------
# envelope and write discipline
# ---------------------------------------------------------------------------

def test_envelope_types_and_line_limit(tmp_path):
    path = tmp_path / "events.jsonl"
    progress.configure(path, job_id="j_test", heartbeat_s=0)
    with progress.span("run", "demo", stages=["S0", "S1"]) as run:
        with progress.stage("S0", label="Data", est_s=1.5) as st:
            st.summary.update(n_points=40, nan=float("nan"))
            with progress.task("fold", k=2, n=5, unit="base_fit:ols") as t:
                t.metrics.update(rmse=float("inf"), r2=0.5)
                progress.tick(1, 3, unit="engine_pass", pass_s=float("nan"))
                progress.metric("scenario.mean_delta", float("-inf"), unit="K", scenario="a")
                progress.metric("label", "text")
            progress.checkpoint("saved", done={"S3", "S0"}, bytes=10, fingerprint="abc")
        progress.skip("S6", "disabled_by_config:causal.enabled")
        progress.warn("qa.coarse", "coarse cells", n=float("nan"))
        progress.emit("custom", big="é" * 5000, long=list(range(3000)), small=1)
        progress.emit("custom2", **{f"f{i}": "x" * 300 for i in range(40)})
        run.set(done=["S0"], timings_s={"S0": 0.1})
    ev = read_events(path)
    types = [e["type"] for e in ev]
    assert types == ["run.start", "stage.start", "task.start", "tick", "metric", "metric", "task.end", "checkpoint",
                     "stage.end", "stage.skip", "warning", "custom", "custom2", "run.end"]
    assert [e["seq"] for e in ev] == sorted(e["seq"] for e in ev)
    assert all(e["job"] == "j_test" and e["pid"] == os.getpid() for e in ev)
    by = {e["type"]: e for e in ev}
    assert by["stage.start"]["stage"] == "S0" and by["stage.start"]["label"] == "Data"
    assert by["stage.start"]["est_s"] == 1.5
    assert by["stage.end"]["summary"] == {"n_points": 40, "nan": None} and by["stage.end"]["status"] == "ok"
    assert {k: by["task.start"][k] for k in ("name", "key", "k", "n", "unit")} == \
        {"name": "fold", "key": None, "k": 2, "n": 5, "unit": "base_fit:ols"}
    assert by["task.end"]["metrics"] == {"rmse": None, "r2": 0.5} and by["task.end"]["elapsed_s"] >= 0
    assert by["tick"]["frac"] == pytest.approx(1 / 3, abs=1e-6) and by["tick"]["pass_s"] is None
    assert ev[4]["value"] is None and ev[4]["tags"] == {"scenario": "a"} and ev[4]["unit"] == "K"
    assert by["checkpoint"]["done"] == ["S0", "S3"] and by["checkpoint"]["changed_sections"] is None
    assert by["stage.skip"] == {**by["stage.skip"], "stage": "S6", "reason": "disabled_by_config:causal.enabled"}
    assert by["warning"]["lvl"] == "warning" and by["warning"]["data"] == {"n": None}
    assert by["run.end"]["status"] == "succeeded" and by["run.end"]["done"] == ["S0"] and by["run.end"]["error"] is None
    # oversized lines: strings shortened with an ellipsis first, then the largest field dropped
    big = by["custom"]
    assert big["truncated"] is True and big["small"] == 1 and "long" not in big and big["big"].endswith("…")
    assert by["custom2"]["truncated"] is True and by["custom2"]["f0"].endswith("…")
    # spans: ids "<pid>:<n>", parents chain, path names the ancestry
    run_id, st_id, t_id = by["run.start"]["span"], by["stage.start"]["span"], by["task.start"]["span"]
    assert run_id.startswith(f"{os.getpid()}:")
    assert by["stage.start"]["parent"] == run_id and by["task.start"]["parent"] == st_id
    assert by["tick"]["span"] == t_id and by["tick"]["parent"] == st_id
    assert by["task.end"]["path"] == ["run:demo", "stage:S0", "task:fold[2/5]"]
    assert by["stage.skip"]["path"] == ["run:demo"]


def test_span_status_on_error_and_cancel():
    sink = Collect()
    progress.configure(sink, heartbeat_s=0)
    with pytest.raises(ValueError):
        with progress.span("run", "r"):
            with progress.task("base_model", key="mgwr", unit="base_fit:mgwr"):
                raise ValueError("boom")
    with pytest.raises(progress.Cancelled):
        with progress.stage("S4"):
            progress.request_cancel()
            progress.check_cancel()
    ends = {e["type"]: e for e in sink.events if e["type"].endswith(".end")}
    assert ends["task.end"]["status"] == "error" and ends["task.end"]["error"] == {"type": "ValueError",
                                                                                   "message": "boom"}
    assert ends["task.end"]["key"] == "mgwr" and ends["task.end"]["path"] == ["run:r", "task:base_model[mgwr]"]
    assert ends["run.end"]["status"] == "failed" and ends["run.end"]["error"]["type"] == "ValueError"
    assert "boom" in ends["run.end"]["error"]["traceback_tail"]
    assert ends["stage.end"]["status"] == "cancelled"


def test_artifact_paths_are_run_relative(tmp_path):
    sink = Collect()
    progress.configure(sink, heartbeat_s=0)
    run_dir = tmp_path / "run"
    (run_dir / "sub").mkdir(parents=True)
    (run_dir / "a.json").write_text("{}")
    (run_dir / "sub" / "b.bin").write_bytes(b"x" * 10)
    with progress.run_dir_scope(run_dir), progress.stage("S2_S3"):
        progress.artifact(run_dir / "a.json", role="manifest")
        with progress.task("fold", k=1, n=1):
            progress.artifact(str(run_dir / "sub"), role="dir")
        progress.artifact(tmp_path / "elsewhere.txt", role="x", stage="S7")
    a, d, other = sink.of("artifact")
    assert (a["path"], a["bytes"], a["stage"], a["role"]) == ("a.json", 2, "S2_S3", "manifest")
    assert (d["path"], d["bytes"], d["stage"]) == ("sub", 10, "S2_S3")
    assert d["span_path"] == ["stage:S2_S3", "task:fold[1/1]"]
    assert other["path"] == (tmp_path / "elsewhere.txt").as_posix() and other["stage"] == "S7"


# ---------------------------------------------------------------------------
# nesting across contextvars, threads and processes
# ---------------------------------------------------------------------------

def _thread_task(i):
    with progress.task("remote_object", key=f"m{i}", unit="climate_model"):
        progress.metric("i", i)
    return threading.get_ident()


def test_spans_nest_across_contextvars_and_threads(tmp_path):
    path = tmp_path / "events.jsonl"
    progress.configure(path, heartbeat_s=0)
    with progress.stage("climate"):
        with progress.context(partition="p1"):
            with progress.context(variant="v"):
                progress.metric("inner", 1)
            progress.metric("outer", 1)
            wrapped = progress.wrap_context(_thread_task)
            with ThreadPoolExecutor(4) as ex:
                idents = set(ex.map(wrapped, range(8)))          # one wrapper, many concurrent calls
                list(ex.map(_thread_task, range(8, 10)))           # unwrapped: no inherited span
        progress.metric("after", 1)
    assert len(idents) >= 1
    ev = read_events(path)
    st_id = next(e["span"] for e in ev if e["type"] == "stage.start")
    m = {e["name"]: e for e in ev if e["type"] == "metric" and e["name"] in ("inner", "outer", "after")}
    assert m["inner"]["ctx"] == {"partition": "p1", "variant": "v"}
    assert m["outer"]["ctx"] == {"partition": "p1"} and m["after"]["ctx"] == {}
    starts = [e for e in ev if e["type"] == "task.start"]
    wrapped_starts = [e for e in starts if int(e["key"][1:]) < 8]
    assert len(wrapped_starts) == 8
    for e in wrapped_starts:
        assert e["parent"] == st_id and e["ctx"] == {"partition": "p1"}
        assert e["path"] == ["stage:climate", f"task:remote_object[{e['key']}]"]
    for e in starts:
        if int(e["key"][1:]) >= 8:
            assert e["parent"] is None and e["path"] == [f"task:remote_object[{e['key']}]"] and e["ctx"] == {}
    task_ids = {e["span"] for e in starts}
    inner = [e for e in ev if e["type"] == "metric" and e["name"] == "i"]
    assert all(e["span"] in task_ids for e in inner) and len(task_ids) == 10


def _pool_task(i):
    with progress.task("replicate", key=f"g/{i}", unit="replicate:g") as t:
        with progress.context(seed=i):
            progress.tick(1, 2, unit="replicate:g")
            progress.metric("share", 0.25 * i, generator="g")
            progress.tick(2, 2, unit="replicate:g")
        t.metrics["i"] = i
    return os.getpid()


def test_spans_nest_across_spawn_process_pool(tmp_path):
    path = tmp_path / "events.jsonl"
    cancel = tmp_path / "cancel"
    progress.configure(path, job_id="j_pool", cancel_file=cancel, heartbeat_s=0)
    with progress.run_dir_scope(tmp_path), progress.stage("S5"), progress.context(placebo="grf"):
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(2, mp_context=ctx, initializer=progress.init_worker,
                                 initargs=(progress.worker_env(),)) as ex:
            pids = set(ex.map(_pool_task, range(6)))
    assert os.getpid() not in pids
    ev = read_events(path)
    st_id = next(e["span"] for e in ev if e["type"] == "stage.start")
    worker = [e for e in ev if e["pid"] in pids]
    assert {e["type"] for e in worker} == {"task.start", "tick", "metric", "task.end"}
    assert all(e["job"] == "j_pool" for e in worker)
    starts = [e for e in worker if e["type"] == "task.start"]
    assert len(starts) == 6
    for e in starts:
        assert e["parent"] == st_id and e["span"].split(":")[0] == str(e["pid"])
        assert e["path"] == ["stage:S5", f"task:replicate[{e['key']}]"] and e["ctx"] == {"placebo": "grf"}
    metrics = [e for e in worker if e["type"] == "metric"]
    assert sorted(e["ctx"]["seed"] for e in metrics) == list(range(6))
    assert all(e["ctx"]["placebo"] == "grf" and e["tags"] == {"generator": "g"} for e in metrics)
    ends = [e for e in worker if e["type"] == "task.end"]
    assert sorted(e["metrics"]["i"] for e in ends) == list(range(6)) and all(e["status"] == "ok" for e in ends)
    assert sum(1 for e in worker if e["type"] == "tick") == 12     # k == 1 and k == n both kept
    assert all(e["t_rel"] >= 0 for e in worker)


def test_worker_env_round_trip():
    progress.configure(Collect(), heartbeat_s=0)
    assert progress.ENV_SINK not in progress.worker_env()      # a callable cannot cross processes
    progress.reset()
    progress.configure("stderr", level="debug", job_id="j_x", cancel_file="/tmp/none", heartbeat_s=0)
    with progress.stage("S1"), progress.context(a=1):
        env = progress.worker_env()
    assert env[progress.ENV_SINK] == "stderr" and env[progress.ENV_LEVEL] == "debug"
    assert env[progress.ENV_JOB] == "j_x" and env[progress.ENV_CANCEL] == "/tmp/none"
    assert env["span"][2] == ["stage:S1"] and env["span"][3] == "S1" and env["ctx"] == {"a": 1}
    json.dumps(env)                                            # plain data: picklable for initargs


# ---------------------------------------------------------------------------
# ticks
# ---------------------------------------------------------------------------

def test_tick_throttle_keeps_first_and_last(monkeypatch):
    sink = Collect()
    progress.configure(sink, heartbeat_s=0)
    clock = [1000.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    with progress.task("a"):
        for k in range(1, 101):                      # all within one second: only k == 1 and k == n
            progress.tick(k, 100, unit="engine_pass")
        with progress.task("b"):                     # each span throttles on its own
            progress.tick(1, 3, unit="u")
            progress.tick(2, 3, unit="u")
            progress.tick(3, 3, unit="u")
        for k in range(1, 11):                       # 0.4 s apart: one tick per second plus first and last
            clock[0] += 0.4
            progress.tick(k, 10, unit="replicates")
    ticks = sink.of("tick")
    assert [(t["unit"], t["k"]) for t in ticks[:2]] == [("engine_pass", 1), ("engine_pass", 100)]
    assert [t["k"] for t in ticks if t["unit"] == "u"] == [1, 3]
    seen = [t["k"] for t in ticks if t["unit"] == "replicates"]
    assert seen[0] == 1 and seen[-1] == 10 and 4 <= len(seen) <= 6
    assert all(t["frac"] == pytest.approx(t["k"] / t["n"]) for t in ticks)


def test_debug_ticks_follow_the_level():
    sink = Collect()
    progress.configure(sink, level="info", heartbeat_s=0)
    progress.tick(1, 2, unit="u", lvl="debug")
    progress.metric("val_mse", 1.0, lvl="debug")
    progress.emit("x", lvl="debug")
    assert sink.events == []
    progress.configure(sink, level="debug", heartbeat_s=0)
    progress.tick(1, 2, unit="u", lvl="debug")
    assert sink.of("tick")[0]["lvl"] == "debug"


def test_log_cap_downgrades_debug_level(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "LOG_CAP_BYTES", 2000)
    monkeypatch.setattr(progress, "CANCEL_STAT_S", 0.0)
    path = tmp_path / "events.jsonl"
    progress.configure(path, level="debug", cancel_file=tmp_path / "cancel", heartbeat_s=0)
    for i in range(20):
        progress.emit("chatter", lvl="debug", i=i)
    progress.check_cancel()                          # the next stat notices the size
    n = len(read_events(path))
    progress.emit("chatter", lvl="debug", i=99)
    progress.emit("kept", lvl="info")
    ev = read_events(path)
    assert len(ev) == n + 1 and ev[-1]["type"] == "kept"


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------

def _swallowing_worker():
    """Library-style code that catches every Exception around a cancel point."""
    try:
        progress.check_cancel()
    except Exception:  # noqa: BLE001 - the point of the test
        return "swallowed"
    return "finished"


def test_cancelled_passes_through_except_exception():
    sink = Collect()
    progress.configure(sink, heartbeat_s=0)
    assert _swallowing_worker() == "finished"
    progress.request_cancel()
    assert progress.cancel_requested()
    with progress.stage("S6"), progress.task("treatment", k=1, n=3):
        with pytest.raises(progress.Cancelled):
            _swallowing_worker()
        with pytest.raises(progress.Cancelled):          # stays requested
            progress.check_cancel()
    assert issubclass(progress.Cancelled, BaseException) and not issubclass(progress.Cancelled, Exception)
    acks = sink.of("cancel.ack")
    assert len(acks) == 1 and acks[0]["at_path"] == ["stage:S6", "task:treatment[1/3]"]


def test_cancel_works_without_a_sink(tmp_path, monkeypatch):
    cancel = tmp_path / "cancel"
    progress.configure(None, cancel_file=cancel)
    assert not progress.enabled()
    progress.check_cancel()
    cancel.touch()
    calls = []
    real_exists = os.path.exists
    monkeypatch.setattr(progress.os.path, "exists", lambda p: calls.append(p) or real_exists(p))
    deadline = time.monotonic() + 3
    with pytest.raises(progress.Cancelled):
        while time.monotonic() < deadline:
            progress.check_cancel()
    assert 1 <= len(calls) <= 8                      # a hot loop stats at most every 0.5 s


def test_signal_handler_sets_flag_then_second_signal_raises():
    sink = Collect()
    progress.configure(sink, heartbeat_s=0)
    previous = signal.getsignal(signal.SIGTERM)
    progress.install_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) is not previous
    with pytest.raises(progress.Cancelled), progress.task("sleeping"):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)                             # handler ran: flag only, nothing raised
        assert progress.cancel_requested()
        os.kill(os.getpid(), signal.SIGINT)          # the second signal raises inside the sleep
        for _ in range(200):
            time.sleep(0.01)
    assert len(sink.of("cancel.ack")) == 1 and sink.of("task.end")[0]["status"] == "cancelled"
    progress.reset()
    assert signal.getsignal(signal.SIGTERM) is previous
    assert not progress.cancel_requested()


def test_job_scope_reconfigures_and_restores(tmp_path):
    outer = tmp_path / "outer.jsonl"
    inner = tmp_path / "inner.jsonl"
    progress.configure(outer, job_id="j_host", heartbeat_s=0)
    progress.emit("before")
    with progress.job_scope("j_req", sink=str(inner), cancel_file=str(tmp_path / "cancel")):
        progress.emit("during")
        progress.request_cancel()
        with pytest.raises(progress.Cancelled):
            progress.check_cancel()
    progress.check_cancel()                          # the request's cancel does not leak
    progress.emit("after")
    o, i = read_events(outer), read_events(inner)
    assert [e["type"] for e in o] == ["before", "after"] and {e["job"] for e in o} == {"j_host"}
    assert [e["type"] for e in i] == ["during", "cancel.ack"] and {e["job"] for e in i} == {"j_req"}


def test_configure_from_env_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "e.jsonl"
    monkeypatch.setenv(progress.ENV_SINK, str(path))
    monkeypatch.setenv(progress.ENV_JOB, "j_env")
    monkeypatch.setenv(progress.ENV_LEVEL, "debug")
    progress.configure_from_env()
    fd = progress._S.sink
    progress.configure_from_env()
    assert progress._S.sink == fd and progress.enabled() and progress.sink_path() == str(path)
    progress.emit("x", lvl="debug")
    assert read_events(path)[0]["job"] == "j_env"
    progress.reset()
    monkeypatch.delenv(progress.ENV_SINK)
    progress.configure_from_env()
    assert not progress.enabled()


# ---------------------------------------------------------------------------
# logging bridge, heartbeat, threads
# ---------------------------------------------------------------------------

def test_log_bridge_forwards_verbatim():
    sink = Collect()
    log = logging.getLogger("sparc.core.bridge_test")
    old = log.level
    log.setLevel(logging.DEBUG)
    try:
        progress.configure(sink, level="debug", heartbeat_s=0)
        log.info("fold %d/%d: R² %.3f — 100%% done", 2, 5, 0.81234)
        log.warning("few cells for %s", "canopy")
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("deprecated thing", UserWarning)
        try:
            raise RuntimeError("bad")
        except RuntimeError:
            log.exception("failed")
        logging.getLogger("other.lib").warning("not ours")
        progress.reset()
        log.warning("after reset")
    finally:
        log.setLevel(old)
    logs = sink.of("log")
    assert logs[0]["msg"] == "fold 2/5: R² 0.812 — 100% done" and logs[0]["lvl"] == "info"
    assert logs[0]["logger"] == "sparc.core.bridge_test" and logs[0]["level"] == "INFO"
    assert logs[1]["msg"] == "few cells for canopy" and logs[1]["lvl"] == "warning"
    assert any(e["logger"] == "py.warnings" and "deprecated thing" in e["msg"] for e in logs)
    assert logs[-1]["msg"] == "failed" and "RuntimeError: bad" in logs[-1]["exc"] and logs[-1]["lvl"] == "error"
    codes = [w["code"] for w in sink.of("warning")]
    assert codes == ["log.sparc.core.bridge_test", "log.py.warnings", "log.sparc.core.bridge_test"]
    assert not any("not ours" in e.get("msg", "") or "after reset" in e.get("msg", "") for e in sink.events)
    assert progress._BRIDGE is None and not any(isinstance(h, progress.LogBridge)
                                                for h in logging.getLogger("sparc").handlers)


def test_heartbeat_thread_starts_and_stops():
    sink = Collect()
    progress.configure(sink, heartbeat_s=0.05)
    thread = progress._S.heartbeat
    assert thread is not None and thread.is_alive() and thread.daemon
    deadline = time.monotonic() + 5
    while len(sink.of("heartbeat")) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    beats = sink.of("heartbeat")
    assert len(beats) >= 2
    assert beats[0]["rss_mb"] > 0 and beats[0]["cpu_s"] >= 0 and beats[0]["threads"] >= 2
    assert beats[0]["span"] is None and beats[0]["path"] == []
    progress.reset()
    assert not thread.is_alive()
    n = len(sink.events)
    time.sleep(0.15)
    assert len(sink.events) == n
    progress.configure(sink, heartbeat_s=0)
    assert progress._S.heartbeat is None


def test_limit_threads_applies_and_restores():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy  # noqa: F401 - loads a BLAS for threadpoolctl to limit

    before_env = {k: os.environ.get(k) for k in progress.THREAD_ENV}
    before = {i["internal_api"]: i["num_threads"] for i in threadpoolctl.threadpool_info()}
    torch = sys.modules.get("torch")
    torch_before = torch.get_num_threads() if torch is not None else None
    with progress.limit_threads(1):
        assert all(os.environ[k] == "1" for k in progress.THREAD_ENV)
        assert all(i["num_threads"] == 1 for i in threadpoolctl.threadpool_info())
        if torch is not None:
            assert torch.get_num_threads() == 1
    assert {k: os.environ.get(k) for k in progress.THREAD_ENV} == before_env
    assert {i["internal_api"]: i["num_threads"] for i in threadpoolctl.threadpool_info()} == before
    if torch is not None:
        assert torch.get_num_threads() == torch_before


def test_overlapping_limit_threads_blocks_restore_the_original():
    """Blocks opened in several server threads may exit in any order; the newest open one applies."""
    before = {k: os.environ.get(k) for k in progress.THREAD_ENV}

    def omp():
        return os.environ["OMP_NUM_THREADS"]

    a, b, c = progress.limit_threads(1), progress.limit_threads(3), progress.limit_threads(1)
    a.__enter__()
    assert omp() == "1"
    b.__enter__()
    assert omp() == "3"
    a.__exit__(None, None, None)
    assert omp() == "3"
    c.__enter__()
    assert omp() == "1"
    b.__exit__(None, None, None)
    assert omp() == "1"
    c.__exit__(None, None, None)
    assert {k: os.environ.get(k) for k in progress.THREAD_ENV} == before
    assert progress._LIMITS == [] and progress._LIMIT_SAVED is None


def test_thread_helpers_never_import_torch(tmp_path):
    import subprocess

    code = ("import sys\nfrom sparc.core import progress\n"
            "with progress.limit_threads(2):\n    pass\n"
            "progress.set_threads(1)\n"
            "import os\nassert os.environ['OMP_NUM_THREADS'] == '1'\n"
            "assert 'torch' not in sys.modules, 'torch imported'\nprint('ok')\n")
    root = Path(progress.__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": str(root)})
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
