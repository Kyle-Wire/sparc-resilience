"""The event-invariant test (SPEC §14.1): a fast run of the synthetic demo (n = 40) with every stage and
``SPARC_PROGRESS`` set must describe itself completely through progress events."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
ENABLED = ["S0", "S1", "S2_S3", "baselines", "S4", "S5", "climate", "S6", "S7", "finish"]
# run_state / checkpoint files are covered by checkpoint events; IGNORED as in sparc.core.catalog (SPEC §6.1)
NOT_ARTIFACTS = ("run_state.json", "checkpoint.pkl", "checkpoint.json")
IGNORED = (".sparc.lock", "*.tmp", "studio/**", "__pycache__/**")
# planned unit kinds (SPEC §5.4); other tick units only add fractional progress
PLANNED = ("s0_load", "s1_influence", "base_fit:", "adv_refit", "stacker_fit:", "checkpoint_save", "baseline_fit:",
           "engine_pass", "causal_step:", "climate_model", "remote_object", "pareto", "replicate:", "variant:")


def _ignored() -> tuple[str, ...]:
    try:                                                              # backend-runs adds the catalog
        from sparc.core.catalog import IGNORED as catalog_ignored
    except ImportError:
        return IGNORED
    return tuple(catalog_ignored)


def completed_units(events: list[dict]) -> Counter:
    """SPEC §5.4 unit accounting: ok task.end with a unit, k == n ticks, S0/S1 stage ends, saved checkpoints."""
    done: Counter = Counter()
    for e in events:
        t = e["type"]
        if t == "task.end" and e.get("status") == "ok" and e.get("unit"):
            done[e["unit"]] += 1
        elif t == "tick" and e.get("unit") and e.get("k") == e.get("n"):
            done[e["unit"]] += 1
        elif t == "stage.end" and e.get("status") == "ok" and e["stage"] in ("S0", "S1"):
            done["s0_load" if e["stage"] == "S0" else "s1_influence"] += 1
        elif t == "checkpoint" and e.get("action") == "saved":
            done["checkpoint_save"] += 1
    return done


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    from sparc.core import progress
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    root = tmp_path_factory.mktemp("progress_pipeline")
    demo = write_demo_project(root / "project", n=40, seed=0)
    sink = root / "events.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv(progress.ENV_SINK, str(sink))
    mp.setenv(progress.ENV_JOB, "j_invariant")
    mp.setenv(progress.ENV_LEVEL, "info")
    mp.delenv(progress.ENV_CANCEL, raising=False)
    progress.reset()
    progress.configure_from_env()
    try:
        res = run_core(load_core_config(demo["config_path"]), fast=True, run_dir=root / "run",
                       run_meta={"studio_run_id": "r_demo", "origin": "studio"})
    finally:
        progress.reset()
        mp.undo()
    events = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    return SimpleNamespace(res=res, events=events, run_dir=root / "run", sink=sink)


def test_every_stage_starts_and_ends_or_is_skipped_with_its_reason(demo_run):
    ev = demo_run.events
    starts = [e["stage"] for e in ev if e["type"] == "stage.start"]
    ends = [(e["stage"], e["status"]) for e in ev if e["type"] == "stage.end"]
    skips = {e["stage"]: e["reason"] for e in ev if e["type"] == "stage.skip"}
    assert starts == ENABLED
    assert ends == [(s, "ok") for s in ENABLED]
    assert skips == {"cv_curve": "disabled_by_config:cv.distance_curve.enabled"}
    for e in ev:                                                     # starts and ends pair up, in order
        if e["type"] in ("stage.start", "stage.end"):
            assert e["path"][-1] == f"stage:{e['stage']}" and e["path"][0] == "run:synthetic_demo"
    s0 = next(e for e in ev if e["type"] == "stage.end" and e["stage"] == "S0")["summary"]
    assert {"n_points", "n_input", "n_dropped", "clipped", "grid_shape", "cell_m", "fill_fraction", "background",
            "noise_floor"} <= set(s0) and s0["n_points"] == demo_run.res.data.n
    s1 = next(e for e in ev if e["type"] == "stage.end" and e["stage"] == "S1")["summary"]
    assert set(s1) == {"block_size_m", "L_prior_m", "target_resid_range_m"}
    metrics = Counter(e["name"] for e in ev if e["type"] == "metric")
    n_pred = len(demo_run.res.cfg.predictors)
    assert metrics["influence.range_m"] == metrics["influence.anisotropy_ratio"] == n_pred
    assert metrics["scenario.mean_delta"] == metrics["scenario.se"] == len(demo_run.res.scenarios) == 8
    assert {"candidate_rmse", "stacker_rmse", "stacker_r2", "interval_coverage", "interval_halfwidth", "theta",
            "planned_total", "realised_total"} <= set(metrics)
    base = [e for e in ev if e["type"] == "task.end" and e["name"] == "base_model"]
    assert base and all({"fit_s", "heldout_rmse", "heldout_r2"} <= set(e["metrics"]) for e in base)


def test_run_envelope_events(demo_run):
    ev = demo_run.events
    types = [e["type"] for e in ev if e["type"] not in ("heartbeat", "log", "warning")]
    assert types[:3] == ["run.start", "run.plan", "run.dir"] and types[-1] == "run.end"
    first_stage = types.index("stage.start")
    assert first_stage > types.index("run.dir")
    start = ev[[e["type"] for e in ev].index("run.start")]
    assert start["name"] == "synthetic_demo" and start["stages"] == ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    assert start["fast"] is True and start["resume"] is False and start["run_meta"]["studio_run_id"] == "r_demo"
    assert len(start["config_sha256"]) == 64 and len(start["code_sha256"]) == 64
    rdir = next(e for e in ev if e["type"] == "run.dir")
    assert Path(rdir["run_dir"]) == demo_run.run_dir.resolve() and len(rdir["fingerprint"]) == 16
    end = ev[-1] if ev[-1]["type"] == "run.end" else next(e for e in reversed(ev) if e["type"] == "run.end")
    assert end["status"] == "succeeded" and end["error"] is None
    assert set(end["timings_s"]) == {"S0", "S1", "S2_S3", "baselines", "S4", "S5", "S6", "S7"}
    assert end["done"] == ["S3", "S4", "S5", "S6", "baselines", "climate"]
    assert all(e["job"] == "j_invariant" for e in ev)
    lines = demo_run.sink.read_bytes().splitlines(keepends=True)
    assert max(len(line) for line in lines) <= 4096
    assert not any(e.get("truncated") for e in ev)
    for p in (e for e in ev if e["type"] == "run.plan"):
        assert [n["id"] for n in p["nodes"]] == ENABLED[:4] + ["cv_curve"] + ENABLED[4:]
        assert p["n_points"] in (None, demo_run.res.data.n)
    assert [e for e in ev if e["type"] == "run.plan"][-1]["n_points"] == demo_run.res.data.n


def test_every_file_in_the_run_dir_has_an_artifact_event(demo_run):
    arts = [e for e in demo_run.events if e["type"] == "artifact"]
    paths = {e["path"] for e in arts}
    files = {p.relative_to(demo_run.run_dir).as_posix() for p in demo_run.run_dir.rglob("*") if p.is_file()}
    ign = _ignored()
    expected = {f for f in files if f not in NOT_ARTIFACTS and not any(fnmatch.fnmatch(f, g) for g in ign)}
    assert expected <= paths, sorted(expected - paths)
    assert paths <= files                                             # nothing reported that is not there
    assert {"scenario_detail.npz", "causal_cells.parquet", "manifest.json", "report.md", "climate.json",
            "allocation.parquet"} <= paths
    for e in arts:
        assert e["bytes"] > 0 and e["stage"] in ENABLED, e
    assert {e["path"]: e["stage"] for e in arts}["predictions.parquet"] == "S2_S3"


def test_checkpoint_events_match_checkpoint_json(demo_run):
    ck = [e for e in demo_run.events if e["type"] == "checkpoint"]
    side = json.loads((demo_run.run_dir / "checkpoint.json").read_text())
    assert ck and all(e["action"] == "saved" for e in ck)
    assert ck[-1]["done"] == side["done"] and ck[-1]["bytes"] == side["bytes"]
    assert ck[-1]["fingerprint"] == side["fingerprint"]
    assert side["bytes"] == (demo_run.run_dir / "checkpoint.pkl").stat().st_size
    assert [len(e["done"]) for e in ck] == list(range(1, len(ck) + 1))      # one more stage per save


def test_run_state_ends_succeeded(demo_run):
    rs = json.loads((demo_run.run_dir / "run_state.json").read_text())
    assert rs["status"] == "succeeded" and rs["stage"] == "finish" and rs["error"] is None
    assert rs["job"] == "j_invariant" and rs["events_path"] == str(demo_run.sink.resolve())
    assert rs["done"] == ["S3", "S4", "S5", "S6", "baselines", "climate"]
    assert rs["fingerprint"] == json.loads((demo_run.run_dir / "checkpoint.json").read_text())["fingerprint"]
    meta = rs["meta"]
    data = demo_run.res.data
    assert meta["n_points"] == data.n and meta["cell_m"] == data.grid.dx and meta["grid_shape"] == list(data.grid.shape)
    assert meta["coarse_m"] is None and meta["subsample_window_n"] == data.n
    assert meta["cv"] == {"n_folds": 3, "block_m": demo_run.res.manifest["cv"]["block_m"],
                          "buffer_m": demo_run.res.manifest["cv"]["buffer_m"], "seed": 42}


def test_no_info_level_gap_exceeds_15_s(demo_run):
    info = [e for e in demo_run.events if e["lvl"] in ("info", "warning", "error") and e["type"] != "heartbeat"]
    gaps = [b["ts"] - a["ts"] for a, b in zip(info, info[1:])]
    assert max(gaps) <= 15.0


def test_unit_accounting_matches_the_plan(demo_run):
    plans = [e for e in demo_run.events if e["type"] == "run.plan"]
    planned = plans[-1]["total_units"]
    done = completed_units(demo_run.events)
    observed = {k: v for k, v in done.items() if k.startswith(PLANNED)}
    assert observed == planned
    assert planned["engine_pass"] > 40 and planned["checkpoint_save"] == 6
    assert all(n["state"] == "will_run" or n["id"] == "cv_curve" for n in plans[-1]["nodes"])


def test_qa_flags_become_warnings(demo_run):
    warns = [e for e in demo_run.events if e["type"] == "warning"]
    codes = Counter(e["code"] for e in warns)
    for f in demo_run.res.data.qa["flags"]:
        assert codes[f"qa.{f['code']}"] == 1
    assert {"qa.window", "qa.cover_overlap"} <= set(codes)
    for e in warns:
        assert e["lvl"] == "warning" and isinstance(e["data"], dict) and e["message"]


@pytest.mark.slow
def test_fixture_script_is_deterministic(tmp_path):
    """``scripts/make_studio_fixtures.py`` twice in temp dirs: same files, arrays, JSON (but volatile keys) and
    event sequence (SPEC §14.1)."""
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "make_studio_fixtures.py"), "--out", str(out),
                            "--work", str(tmp_path / "work")], cwd=REPO, capture_output=True, text=True,
                           timeout=1800, env=_clean_env())
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(out)
    a, b = outs
    files_a = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert files_a == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    assert "checkpoint.pkl" not in files_a and {"events.jsonl", "FIXTURE.json", "manifest.json"} <= set(files_a)
    assert sum((a / f).stat().st_size for f in files_a) <= 3 * 2 ** 20
    assert json.loads((a / "FIXTURE.json").read_text()) == {"n": 40, "seed": 0, "mode": "fast",
                                                            "generator": "write_demo_project"}
    for f in files_a:
        pa, pb = a / f, b / f
        if f.endswith(".parquet"):
            da, db = pd.read_parquet(pa), pd.read_parquet(pb)
            assert list(da.columns) == list(db.columns), f
            for c in da.columns:
                assert _same(da[c].to_numpy(), db[c].to_numpy()), (f, c)
        elif f.endswith(".npz"):
            with np.load(pa) as za, np.load(pb) as zb:
                assert za.files == zb.files and all(_same(za[k], zb[k]) for k in za.files), f
        elif f.endswith(".json"):
            assert _strip_volatile(json.loads(pa.read_text())) == _strip_volatile(json.loads(pb.read_text())), f
        elif f == "events.jsonl":
            assert _event_sequence(pa) == _event_sequence(pb)
    assert _event_sequence(a / "events.jsonl") and not (tmp_path / "work").exists()


VOLATILE = {"created_utc", "timings_s", "timings_detail", "git_commit", "pid", "ts", "elapsed_s", "seconds", "fit_s",
            "pass_s", "saved_utc", "updated_utc", "started_utc", "bytes"}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-identical arrays (NaN where NaN)."""
    return np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")


def _clean_env() -> dict:
    import os

    return {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")} | {"OMP_NUM_THREADS": "1",
                                                                                   "MKL_NUM_THREADS": "1"}


def _strip_volatile(obj, key: str = ""):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v, k) for k, v in obj.items()
                if k not in VOLATILE and not (key == "provenance" and k == "git")}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _event_sequence(path: Path) -> list[tuple]:
    seq = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["type"] in ("heartbeat", "log"):
            continue
        if e["type"] == "tick" and e.get("k") != e.get("n") and e.get("k") != 1:
            continue                                                  # throttled by wall time
        seq.append((e["type"], e.get("name"), e.get("key"), e.get("stage"), e.get("k"), e.get("n"), e.get("unit")))
    return seq


# ---------------------------------------------------------------------------
# network fetchers (SPEC §5.5): one remote_object task per object, retries reported
# ---------------------------------------------------------------------------


def test_cmip6_models_are_remote_object_tasks_on_the_thread_pool(monkeypatch, tmp_path):
    from sparc.core import climate as C
    from sparc.core import progress

    names = ["A", "B", "BROKEN", "C"]
    monkeypatch.setattr(C, "load_catalog", lambda cache, fetch: pd.DataFrame({"source_id": names}))
    monkeypatch.setattr(C, "select_runs", lambda cat, *a, **k: cat)

    def fake_model(m, runs, *a, **k):
        if m == "BROKEN":
            raise OSError("store unreachable")
        return [{"model": m, "experiment": "ssp245", "period": "2041-2060", "delta_K": 2.0}]

    monkeypatch.setattr(C, "_model_factors", fake_model)
    events: list[dict] = []
    counts: list[int] = []
    progress.configure(events.append, heartbeat_s=0)
    try:
        with progress.task("climate_fetch"):
            df = C.cmip6_change_factors(41.8, -71.4, tmp_path, experiments=("ssp245",), max_workers=3,
                                        on_models=counts.append)
        progress.request_cancel()
        with pytest.raises(progress.Cancelled):
            C.cmip6_change_factors(41.8, -71.4, tmp_path, experiments=("ssp245",), max_workers=2)
    finally:
        progress.reset()
    assert counts == [4] and sorted(df["model"]) == ["A", "B", "C"]
    ends = [e for e in events if e["type"] == "task.end" and e["name"] == "remote_object" and e["unit"]]
    assert sorted(e["key"] for e in ends[:4]) == sorted(names)
    assert all(e["unit"] == "climate_model" and e["status"] == "ok" for e in ends[:4])
    assert all(e["path"][0] == "task:climate_fetch" for e in ends)      # thread-pool tasks keep their parent span
    skipped = [e for e in events if e["type"] == "warning" and e["code"] == "climate.model_skipped"]
    assert len(skipped) == 1 and skipped[0]["data"]["model"] == "BROKEN"
    assert len(ends) == 4                                              # after the cancel no model ran


def test_network_retries_are_reported_and_cancellable(monkeypatch):
    import urllib.error

    from sparc.core import climate as C
    from sparc.core import progress

    events: list[dict] = []
    progress.configure(events.append, heartbeat_s=0)
    try:
        C.retry_wait("https://cmip6-pds.s3.amazonaws.com/x", 0, 0.01, "reset")
        calls = []

        def flaky(url, timeout=None):
            calls.append(url)
            raise urllib.error.URLError("connection reset")

        waits = []
        monkeypatch.setattr("urllib.request.urlopen", flaky)
        monkeypatch.setattr(C, "retry_wait", lambda url, attempt, wait_s, error="": waits.append((attempt, wait_s)))
        with pytest.raises(urllib.error.URLError):
            C.http_fetch("https://example.org/obj", retries=2)
        assert len(calls) == 3 and waits == [(0, 2.0), (1, 4.0)]
        monkeypatch.undo()
        progress.request_cancel()
        with pytest.raises(progress.Cancelled):
            C.retry_wait("https://example.org/obj", 1, 30.0)                # a long back-off stops on cancel
    finally:
        progress.reset()
    warns = [e for e in events if e["type"] == "warning" and e["code"] == "network.retry"]
    assert warns[0]["data"] == {"host": "cmip6-pds.s3.amazonaws.com", "attempt": 1, "wait_s": 0.01, "error": "reset"}
    assert warns[-1]["data"]["host"] == "example.org" and warns[-1]["data"]["attempt"] == 2
