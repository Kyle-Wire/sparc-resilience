"""Performance budgets (SPEC §6.2, §13): a warm layer fetch at Providence size (54,701 cells) under 50 ms and the
run list of 200 runs under 100 ms, through the HTTP stack."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studio.runs.conftest import f32

pytestmark = pytest.mark.slow

N_CELLS = 54_701


def _best_ms(fn, repeat: int = 7) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def _big_run(root: Path) -> Path:
    """A run folder with 54,701 cells scattered over a 287 × 334-cell extent of 30 m cells."""
    rng = np.random.default_rng(0)
    iy, ix = np.mgrid[0:334, 0:287]
    keep = rng.permutation(ix.size)[:N_CELLS]
    keep.sort()
    x = 290000.0 + ix.ravel()[keep] * 30.0
    y = 4630000.0 + iy.ravel()[keep] * 30.0
    target = 85 + rng.normal(0, 2, N_CELLS)
    pred = target + rng.normal(0, 0.5, N_CELLS)
    rd = root / "big_run"
    rd.mkdir(parents=True)
    pd.DataFrame({"id": np.arange(N_CELLS), "x_m": x, "y_m": y, "fold": rng.integers(0, 5, N_CELLS),
                  "target": target, "pred": pred, "pi_lo": pred - 1, "pi_hi": pred + 1}).to_parquet(
        rd / "predictions.parquet")
    (rd / "manifest.json").write_text(json.dumps({
        "name": "big", "created_utc": "2026-10-01T12:00:00+00:00", "fast_mode": False,
        "config": {"data": {"path": "absent.csv", "target": "t", "crs": "EPSG:32619", "target_units": "degF"}},
        "qa": {"cell_m": 30.0}, "metrics": {"stacker": {"r2": 0.8, "rmse": 0.5}}}))
    return rd


def test_warm_layer_fetch_at_providence_size(client, ctx, tmp_path):
    row = ctx.services["registry"].index_run_dir(_big_run(tmp_path), origin="imported")
    rid = row["id"]
    meta = client.get(f"/api/runs/{rid}/grid").json()
    assert meta["n"] == N_CELLS and (meta["ny"], meta["nx"]) == (336, 289)          # one padding cell per side
    first = client.get(f"/api/runs/{rid}/layers/pred.bin")
    assert first.status_code == 200 and f32(first.content).size == N_CELLS
    assert first.headers["cache-control"].endswith("immutable")
    ms = _best_ms(lambda: client.get(f"/api/runs/{rid}/layers/pred.bin"))
    assert ms < 50, f"warm layer fetch took {ms:.1f} ms"
    ms304 = _best_ms(lambda: client.get(f"/api/runs/{rid}/layers/pred.bin",
                                        headers={"If-None-Match": first.headers["etag"]}))
    assert ms304 < 50
    ms_grid = _best_ms(lambda: client.get(f"/api/runs/{rid}/grid.bin"))
    assert ms_grid < 100, f"warm grid.bin took {ms_grid:.1f} ms"


def test_run_list_of_200_runs(client, ctx, tmp_path):
    reg = ctx.services["registry"]
    root = tmp_path / "many"
    for i in range(200):
        rd = root / f"run_{i:03d}"
        rd.mkdir(parents=True)
        (rd / "manifest.json").write_text(json.dumps({
            "name": f"run {i}", "created_utc": f"2026-09-{1 + i % 28:02d}T{i % 24:02d}:00:00+00:00",
            "fast_mode": bool(i % 2), "timings_s": {"S0": 1.0, "S1": 2.0},
            "metrics": {"stacker": {"r2": 0.5 + i / 1000, "rmse": 1.0}}}))
        reg.index_run_dir(rd, origin="imported", publish=False)
    assert ctx.db.fetchval("SELECT COUNT(*) FROM runs") == 200
    page = client.get("/api/runs", params={"limit": 200}).json()
    assert len(page["items"]) == 200 and page["next_cursor"] is None
    ms = _best_ms(lambda: client.get("/api/runs", params={"limit": 200}), repeat=5)
    assert ms < 100, f"listing 200 runs took {ms:.1f} ms"
