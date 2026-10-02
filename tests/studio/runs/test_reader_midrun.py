"""RunReader on unfinished and older runs (SPEC §6.2): live metrics from ``predictions.parquet``, the grid and
the CV folds from ``run_state.meta`` + ``launch.json`` before the manifest exists, older-code sections."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studio.runs.conftest import RUN_ID

AFTER_S2 = ("manifest.json", "response_albedo.parquet", "response_canopy.parquet", "response_impervious.parquet",
            "response_curves.json", "scenarios.json", "scenario_deltas.parquet", "scenario_detail.npz",
            "climate.json", "causal.json", "causal_cells.parquet", "optimize.json", "allocation.parquet",
            "report.md", "methods.md", "model_card.md", "environment.txt")
AFTER_S0 = AFTER_S2 + ("predictions.parquet", "physics.json", "baselines.json", "influence.json",
                       "checkpoint.json")


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _running_state(done: list[str], stage: str):
    def edit(rd: Path) -> None:
        st = json.loads((rd / "run_state.json").read_text())
        st.update(status="running", stage=stage, done=done, pid=_dead_pid(), host=socket.gethostname(),
                  updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (rd / "run_state.json").write_text(json.dumps(st))
        if (rd / "checkpoint.json").exists():
            ck = json.loads((rd / "checkpoint.json").read_text())
            ck["done"] = done
            (rd / "checkpoint.json").write_text(json.dumps(ck))

    return edit


def test_midrun_without_manifest(client, demo, place_run, synth):
    place_run(demo, drop=AFTER_S2, edit=_running_state(["S3"], "S4"))
    rid = RUN_ID
    run = client.get(f"/api/runs/{rid}").json()
    assert run["run"]["status"] == "partial"                 # its process is gone, S3 is checkpointed

    acc = client.get(f"/api/runs/{rid}/views/accuracy").json()
    assert acc["sections"]["live"] is True
    pred = pd.read_parquet(synth / "predictions.parquet")
    y, p = pred["target"].to_numpy(float), pred["pred"].to_numpy(float)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    stack = next(m for m in acc["sections"]["models"] if m["kind"] == "stack")
    assert stack["r2"] == pytest.approx(r2)
    assert {m["model"] for m in acc["sections"]["models"]} >= {"ols", "mgwr", "gwrf", "gam", "physics"}

    grid = client.get(f"/api/runs/{rid}/grid").json()
    assert grid["n"] == 1120 and (grid["ny"], grid["nx"]) == (42, 42) and grid["dx_m"] == 30.0
    assert grid["n_folds"] == 3
    for k in range(3):
        r = client.get(f"/api/runs/{rid}/folds/{k}.bin")
        assert r.status_code == 200 and r.headers["x-sparc-dtype"] == "uint8"
        cls = np.frombuffer(r.content, dtype=np.uint8)
        assert cls.size == 1120
        np.testing.assert_array_equal(cls == 1, pred["fold"].to_numpy() == k)
        assert set(np.unique(cls)) <= {0, 1, 2}
        assert "immutable" not in r.headers["cache-control"]  # an unfinished run is never cached
    assert client.get(f"/api/runs/{rid}/folds/3.bin").status_code == 404

    tabs = {t["id"]: t["availability"] for t in client.get(f"/api/runs/{rid}/outputs").json()["tabs"]}
    assert tabs["accuracy"] == "ready" and tabs["scenarios"] == "missing" and tabs["map"] == "ready"
    sc = client.get(f"/api/runs/{rid}/views/scenarios").json()
    assert sc["availability"] == "missing" and sc["missing"][0]["produced_by"] == "stage:S5"


def test_s0_only_run_rebuilds_grid_and_folds_from_the_data(client, demo, place_run, synth):
    place_run(demo, drop=AFTER_S0, edit=_running_state([], "S1"))
    rid = RUN_ID
    assert client.get(f"/api/runs/{rid}").json()["run"]["status"] == "interrupted"
    grid = client.get(f"/api/runs/{rid}/grid").json()
    assert grid["n"] == 1120 and (grid["ny"], grid["nx"]) == (42, 42)
    raw = client.get(f"/api/runs/{rid}/grid.bin")
    offs = {o["name"]: o for o in json.loads(raw.headers["x-sparc-offsets"])}
    ix = np.frombuffer(raw.content, "<i4", 1120, offs["ix"]["offset"])
    iy = np.frombuffer(raw.content, "<i4", 1120, offs["iy"]["offset"])
    pred = pd.read_parquet(synth / "predictions.parquet")
    dx = grid["dx_m"]
    np.testing.assert_array_equal(ix, np.round((pred["x_m"] - grid["x0_m"]) / dx).astype(int))
    np.testing.assert_array_equal(iy, np.round((pred["y_m"] - grid["y0_m"]) / dx).astype(int))
    ids = client.get(f"/api/runs/{rid}/grid/ids.json").json()
    assert ids == pred["id"].tolist()
    cls = np.frombuffer(client.get(f"/api/runs/{rid}/folds/1.bin").content, dtype=np.uint8)
    np.testing.assert_array_equal(cls == 1, pred["fold"].to_numpy() == 1)
    acc = client.get(f"/api/runs/{rid}/views/accuracy").json()
    assert acc["availability"] == "missing" and acc["sections"]["live"] is False


def test_older_code_sections(client, demo, place_run):
    def older(rd: Path) -> None:
        m = json.loads((rd / "manifest.json").read_text())
        for s in m["scenarios"]:
            s.pop("mean_delta_se", None)
        m["metrics"]["stacker"].pop("interval_diagnostics", None)
        m.pop("provenance")
        (rd / "manifest.json").write_text(json.dumps(m))

    place_run(demo, edit=older)
    sec = client.get(f"/api/runs/{RUN_ID}").json()["sections"]
    assert sec["scenarios"]["older_code"] is True and sec["metrics"]["older_code"] is True
    assert sec["provenance"]["older_code"] is True and sec["causal"]["older_code"] is False


def test_reader_cache_follows_the_files(ctx, demo, place_run):
    rd = place_run(demo)
    reader = ctx.services["reader"]
    a = reader.get(RUN_ID)
    assert reader.get(RUN_ID) is a                            # unchanged files → the same context
    st = json.loads((rd / "run_state.json").read_text())
    st["updated_utc"] = "2026-10-01T22:00:00Z"
    time.sleep(0.01)
    (rd / "run_state.json").write_text(json.dumps(st))
    b = reader.get(RUN_ID)
    assert b is not a and b.run_state["updated_utc"] == "2026-10-01T22:00:00Z"
