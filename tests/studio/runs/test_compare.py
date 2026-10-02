"""Compare runs (api.md §6.5): config diff and chips, the b − a layer, and priority agreement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from tests.studio.runs.conftest import RUN_ID, f32

RUN_B = "20261001-212119-fast-b2c3"


@pytest.fixture
def pair(client, demo, place_run, synth):
    """The fixture run (a) and a copy (b) with another stacker coverage and perturbed footprint slopes."""
    rng = np.random.default_rng(3)

    def perturb(rd: Path) -> None:
        r = pd.read_parquet(rd / "response_canopy.parquet")
        r["footprint_effect_per_unit"] = r["footprint_effect_per_unit"] + rng.normal(0, 0.02, len(r))
        r.to_parquet(rd / "response_canopy.parquet")
        p = pd.read_parquet(rd / "predictions.parquet")
        p["pred"] = p["pred"] + 0.25
        p.to_parquet(rd / "predictions.parquet")
        (rd / "climate.json").unlink()

    def coverage(raw: dict) -> None:
        raw.setdefault("stacker", {})["coverage"] = 0.8

    place_run(demo)
    place_run(demo, RUN_B, edit=perturb, edit_config=coverage)
    return RUN_ID, RUN_B


def test_compare_runs(client, pair):
    a, b = pair
    r = client.get("/api/compare/runs", params={"a": a, "b": b})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["a"]["id"] == a and c["b"]["id"] == b
    assert c["same"]["grid"] is True and c["same"]["code"] is True and c["same"]["data"] is True
    diff = {d["path"]: d for d in c["config_diff"]}
    assert diff["stacker.coverage"]["a"] == 0.9 and diff["stacker.coverage"]["b"] == 0.8
    assert len(diff) == 1
    assert "climate" in c["outputs"]["a_only"] and c["outputs"]["b_only"] == []
    r2 = next(m for m in c["metrics"] if m["key"] == "r2")
    assert r2["a"] == r2["b"] and r2["delta"] == 0.0                   # both from the manifest metrics
    assert {s["name"] for s in c["scenarios"]} >= {"Canopy Increase +10", "Cooling package"}
    assert c["environment"] == {"added": [], "removed": [], "changed": []}
    same = client.get("/api/compare/runs", params={"a": a, "b": a}).json()
    assert same["config_diff"] == [] and same["same"]["config"] is True


def test_compare_layer_is_b_minus_a(client, pair):
    a, b = pair
    r = client.get("/api/compare/layer.bin", params={"a": a, "b": b, "key": "pred"})
    assert r.status_code == 200 and r.headers["x-sparc-dtype"] == "float32"
    d = f32(r.content)
    pa = f32(client.get(f"/api/runs/{a}/layers/pred.bin").content)
    pb = f32(client.get(f"/api/runs/{b}/layers/pred.bin").content)
    np.testing.assert_array_equal(d, pb - pa)
    np.testing.assert_allclose(d, 0.25, atol=1e-4)
    assert r.headers["cache-control"].endswith("immutable")
    assert client.get("/api/compare/layer.bin", params={"a": a, "b": b, "key": "pred"},
                      headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_priority_agreement(client, pair):
    a, b = pair
    same = client.post("/api/compare/priority", json={"a": a, "b": a, "layer": "fp_canopy"}).json()
    assert same["kendall_tau"] == pytest.approx(1.0) and same["top_decile_jaccard"] == 1.0 and same["n"] == 1120
    res = client.post("/api/compare/priority", json={"a": a, "b": b, "layer": "fp_canopy"}).json()
    va = f32(client.get(f"/api/runs/{a}/layers/fp_canopy.bin").content).astype(np.float64)
    vb = f32(client.get(f"/api/runs/{b}/layers/fp_canopy.bin").content).astype(np.float64)
    ok = np.isfinite(va) & np.isfinite(vb)
    assert res["n"] == int(ok.sum())
    assert res["kendall_tau"] == pytest.approx(kendalltau(va[ok], vb[ok]).statistic)
    k = round(0.1 * ok.sum())
    ta = set(np.argsort(va[ok], kind="stable")[:k])            # negative = cooler: most negative first
    tb = set(np.argsort(vb[ok], kind="stable")[:k])
    assert res["top_decile_jaccard"] == pytest.approx(len(ta & tb) / len(ta | tb))
    assert 0 < res["kendall_tau"] < 1 and 0 < res["top_decile_jaccard"] < 1


def test_grid_mismatch(client, demo, place_run):
    def fewer(rd: Path) -> None:
        for name in ("predictions.parquet", "response_canopy.parquet"):
            p = pd.read_parquet(rd / name)
            p.iloc[:-40].to_parquet(rd / name)

    place_run(demo)
    place_run(demo, RUN_B, edit=fewer)
    r = client.get("/api/compare/layer.bin", params={"a": RUN_ID, "b": RUN_B, "key": "pred"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "grid_mismatch"
    assert client.post("/api/compare/priority", json={"a": RUN_ID, "b": RUN_B,
                                                      "layer": "fp_canopy"}).status_code == 409
    c = client.get("/api/compare/runs", params={"a": RUN_ID, "b": RUN_B}).json()
    assert c["same"]["grid"] is False
