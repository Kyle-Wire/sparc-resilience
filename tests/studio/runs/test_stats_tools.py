"""Analysis tools (api.md §6.4): region stats with the fold jackknife, breakdown, hexbin and the correlogram."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from tests.studio.runs.conftest import f32

SLUG, NAME = "canopy-increase-plus-10", "Canopy Increase +10"


def _detail_folds(synth, name: str) -> np.ndarray:
    with np.load(synth / "scenario_detail.npz") as z:
        names = json.loads(bytes(z["names"]).decode()) if z["names"].dtype == np.uint8 else list(z["names"])
        return np.asarray(z[f"f{list(names).index(name)}"], np.float64)


def _jackknife(folds: np.ndarray, mask: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        m = np.array([f[mask].mean() for f in folds])
    else:
        m = np.array([np.sum(f[mask] * w[mask]) / np.sum(w[mask]) for f in folds])
    return float(m.std() * np.sqrt(len(m) - 1))


def test_region_stats_jackknife(client, fixture_run, synth):
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    obs = pred["target"].to_numpy()
    folds = _detail_folds(synth, NAME)
    assert folds.shape == (3, 1120)
    delta = pd.read_parquet(synth / "scenario_deltas.parquet")[NAME].to_numpy(np.float32).astype(np.float64)
    spec = {"kind": "filter", "column": "obs", "op": ">", "value": float(np.median(obs))}
    mask = obs.astype(np.float32).astype(np.float64) > float(np.median(obs))
    r = client.post(f"/api/runs/{rid}/stats/region", json={"selection": spec, "layers": ["obs", "canopy"],
                                                          "scenarios": [f"configured:{SLUG}"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_cells"] == int(mask.sum()) and body["area_km2"] == pytest.approx(mask.sum() * 900 / 1e6)
    lay = body["layers"]["obs"]
    o32 = obs.astype(np.float32).astype(np.float64)
    assert lay["mean"] == pytest.approx(o32[mask].mean()) and lay["mean_outside"] == pytest.approx(o32[~mask].mean())
    assert lay["p50"] == pytest.approx(np.percentile(o32[mask], 50)) and lay["sd"] == pytest.approx(o32[mask].std())
    sc = body["scenarios"][f"configured:{SLUG}"]
    assert sc["has_folds"] is True
    assert sc["inside"]["estimate"] == pytest.approx(delta[mask].mean(), rel=1e-6)
    assert sc["inside"]["se"] == pytest.approx(_jackknife(folds, mask), rel=1e-6)
    assert sc["outside"]["se"] == pytest.approx(_jackknife(folds, ~mask), rel=1e-6)
    lo, hi = sc["inside"]["lo"], sc["inside"]["hi"]
    assert lo == pytest.approx(sc["inside"]["estimate"] - 1.96 * sc["inside"]["se"], abs=1e-9)
    assert hi == pytest.approx(sc["inside"]["estimate"] + 1.96 * sc["inside"]["se"], abs=1e-9)
    assert sc["inside"]["confidence"] in ("confident_cools", "confident_warms", "could_be_zero")


def test_city_wide_jackknife_matches_core_summary(client, fixture_run, synth):
    rid, _ = fixture_run
    manifest = json.loads((synth / "manifest.json").read_text())
    row = next(s for s in manifest["scenarios"] if s["name"] == NAME)
    body = client.post(f"/api/runs/{rid}/stats/region", json={"selection": {"kind": "all"}, "layers": [],
                                                              "scenarios": [f"configured:{SLUG}"]}).json()
    inside = body["scenarios"][f"configured:{SLUG}"]["inside"]
    assert inside["se"] == pytest.approx(row["mean_delta_se"], rel=1e-4)
    assert inside["estimate"] == pytest.approx(row["mean_delta"], rel=1e-4)


def test_people_weighted_region_stats(client, fixture_run, synth):
    rid, _ = fixture_run
    people = f32(client.get(f"/api/runs/{rid}/layers/people.bin").content).astype(np.float64)
    folds = _detail_folds(synth, NAME)
    pred = pd.read_parquet(synth / "predictions.parquet")
    mask = pred["fold"].to_numpy() == 0
    body = client.post(f"/api/runs/{rid}/stats/region", json={
        "selection": {"kind": "filter", "column": "fold", "op": "==", "value": 0}, "layers": ["obs"],
        "weights": "people", "scenarios": [f"configured:{SLUG}"]}).json()
    w = np.where(np.isfinite(people), people, 0.0)
    ok = mask & (w > 0)
    assert body["people"] == pytest.approx(float(np.nansum(people[mask])), rel=1e-6)
    o32 = pred["target"].to_numpy(np.float32).astype(np.float64)
    assert body["layers"]["obs"]["mean"] == pytest.approx(np.sum(o32[ok] * w[ok]) / np.sum(w[ok]), rel=1e-9)
    assert body["scenarios"][f"configured:{SLUG}"]["inside"]["se"] == pytest.approx(_jackknife(folds, ok, w),
                                                                                     rel=1e-6)


def test_region_stats_errors(client, fixture_run):
    rid, _ = fixture_run
    r = client.post(f"/api/runs/{rid}/stats/region", json={"selection": {"kind": "all"}, "layers": [],
                                                          "scenarios": ["configured:nope"]})
    assert r.status_code == 422
    r = client.post(f"/api/runs/{rid}/stats/region", json={"selection": {"kind": "all"}, "layers": ["nope"]})
    assert r.status_code == 422


def test_breakdown(client, fixture_run, synth):
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    o32 = pred["target"].to_numpy(np.float32).astype(np.float64)
    fold = pred["fold"].to_numpy()
    res = client.post(f"/api/runs/{rid}/stats/breakdown", json={"value": "obs", "by": {"kind": "fold"},
                                                                "stat": "box"}).json()
    assert [g["label"] for g in res["groups"]] == ["Fold 1", "Fold 2", "Fold 3"]
    for k, g in enumerate(res["groups"]):
        assert g["n"] == int((fold == k).sum())
        assert g["q"] == pytest.approx(np.percentile(o32[fold == k], [10, 25, 50, 75, 90]).tolist())
        assert g["mean"] == pytest.approx(o32[fold == k].mean())
    canopy = f32(client.get(f"/api/runs/{rid}/layers/canopy.bin").content).astype(np.float64)
    res = client.post(f"/api/runs/{rid}/stats/breakdown", json={
        "value": "obs", "by": {"kind": "quantile", "layer": "canopy", "q": 4}, "stat": "mean"}).json()
    assert sum(g["n"] for g in res["groups"]) == int(np.isfinite(canopy).sum())
    assert all(g["q"] is None and g["label"].startswith("Q") for g in res["groups"])
    cats = client.post(f"/api/runs/{rid}/stats/breakdown", json={
        "value": "obs", "by": {"kind": "category", "layer": "cls_canopy"}}).json()["groups"]
    assert cats and all(not g["label"][0].isdigit() for g in cats)
    hexes = client.post(f"/api/runs/{rid}/stats/breakdown", json={
        "value": "pred", "by": {"kind": "hex", "size_m": 500}}).json()["groups"]
    assert sum(g["n"] for g in hexes) == 1120
    assert client.post(f"/api/runs/{rid}/stats/breakdown", json={
        "value": "obs", "by": {"kind": "zone"}}).status_code == 422


def test_hexbin(client, fixture_run, synth):
    rid, _ = fixture_run
    canopy = f32(client.get(f"/api/runs/{rid}/layers/canopy.bin").content).astype(np.float64)
    obs = pd.read_parquet(synth / "predictions.parquet")["target"].to_numpy(np.float32).astype(np.float64)
    sel = {"kind": "filter", "column": "fold", "op": "==", "value": 2}
    res = client.post(f"/api/runs/{rid}/stats/hexbin", json={"x": "canopy", "y": "obs", "bins": 20,
                                                             "selection": sel}).json()
    ok = np.isfinite(canopy) & np.isfinite(obs)
    counts, xe, ye = np.histogram2d(canopy[ok], obs[ok], bins=20,
                                    range=[[canopy[ok].min(), canopy[ok].max()], [obs[ok].min(), obs[ok].max()]])
    np.testing.assert_allclose(res["x_edges"], xe)
    np.testing.assert_allclose(res["y_edges"], ye)
    np.testing.assert_array_equal(np.array(res["counts"]), counts.astype(int))
    fold = pd.read_parquet(synth / "predictions.parquet")["fold"].to_numpy()
    assert np.array(res["sel_counts"]).sum() == int((ok & (fold == 2)).sum())
    assert res["spearman"] == pytest.approx(spearmanr(canopy[ok], obs[ok]).statistic)
    assert len(res["binned_mean"]) <= 20


def test_acf_matches_core_fft_acf(client, fixture_run, synth):
    from sparc.core.influence import fft_acf, permutation_band

    rid, _ = fixture_run
    meta = client.get(f"/api/runs/{rid}/grid").json()
    g = client.get(f"/api/runs/{rid}/grid.bin")
    offs = {o["name"]: o for o in json.loads(g.headers["x-sparc-offsets"])}
    ix = np.frombuffer(g.content, "<i4", 1120, offs["ix"]["offset"])
    iy = np.frombuffer(g.content, "<i4", 1120, offs["iy"]["offset"])
    obs = f32(client.get(f"/api/runs/{rid}/layers/obs.bin").content).astype(np.float64)
    raster = np.full((meta["ny"], meta["nx"]), np.nan)
    raster[iy, ix] = obs
    mask = np.isfinite(raster)
    res = client.post(f"/api/runs/{rid}/stats/acf", json={"layer": "obs", "max_lag_m": 600, "n_perm": 5}).json()
    want = fft_acf(raster, mask, meta["dx_m"], 600.0, n_bins=20)
    np.testing.assert_allclose(np.array(res["acf"], dtype=float), want["acf"], rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(np.array(res["lags_m"], dtype=float), want["lags_m"], rtol=1e-9, equal_nan=True)
    bm, _bs = permutation_band(raster, mask, meta["dx_m"], 600.0, n_bins=20, n_perm=5)
    np.testing.assert_allclose(np.array(res["band_mean"], dtype=float), bm, rtol=1e-9, equal_nan=True)
    # default lag: min(influence.max_lag_m, half the shorter side of the grid)
    dflt = client.post(f"/api/runs/{rid}/stats/acf", json={"layer": "obs", "n_perm": 2}).json()
    lags = [x for x in dflt["lags_m"] if x is not None]
    assert lags and max(lags) <= min(meta["nx"], meta["ny"]) * meta["dx_m"] / 2
