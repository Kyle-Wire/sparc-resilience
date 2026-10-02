"""Selections (SPEC §6.6, api.md §1, §6.3): every SelectionSpec kind against a brute-force reference, the
combinator laws, the bitset codec, blobs and saved regions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.studio.runs import selection as S
from tests.studio.runs.conftest import RUN_ID, f32


@pytest.fixture
def run(client, fixture_run, synth):
    """``(rid, resolve(spec) -> bool[n], predictions, grid meta)``."""
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    meta = client.get(f"/api/runs/{rid}/grid").json()

    def resolve(spec) -> np.ndarray:
        r = client.post(f"/api/runs/{rid}/selection/resolve", json={"selection": spec})
        assert r.status_code == 200, r.text
        body = r.json()
        mask = S.decode_bitset(body["mask"], 1120)
        assert body["n_cells"] == int(mask.sum())
        assert body["area_km2"] == pytest.approx(mask.sum() * 900 / 1e6)
        return mask

    return rid, resolve, pred, meta


def _to_lonlat(x, y):
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True)
    pts = [tr.transform(float(a), float(b)) for a, b in zip(x, y)]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def test_polygon_in_lonlat(run):
    rid, resolve, pred, meta = run
    x0, y0, dx = meta["x0_m"], meta["y0_m"], meta["dx_m"]
    ix = np.round((pred["x_m"] - x0) / dx).astype(int).to_numpy()
    iy = np.round((pred["y_m"] - y0) / dx).astype(int).to_numpy()
    # an outline whose edges run between cell centres (half a cell ≫ the projection's curvature at city scale)
    xs = x0 + np.array([8.5, 30.5, 30.5, 8.5]) * dx
    ys = y0 + np.array([5.5, 5.5, 25.5, 25.5]) * dx
    hx = x0 + np.array([14.5, 20.5, 20.5, 14.5]) * dx
    hy = y0 + np.array([10.5, 10.5, 15.5, 15.5]) * dx
    lon, lat = _to_lonlat(xs, ys)
    hlon, hlat = _to_lonlat(hx, hy)
    outer = [[float(a), float(b)] for a, b in zip(lon, lat)]
    hole = [[float(a), float(b)] for a, b in zip(hlon, hlat)]
    got = resolve({"kind": "polygon", "crs": "EPSG:4326", "rings": [outer]})
    want = (ix >= 9) & (ix <= 30) & (iy >= 6) & (iy <= 25)
    np.testing.assert_array_equal(got, want)
    got = resolve({"kind": "polygon", "crs": "EPSG:4326", "rings": [outer, hole]})
    np.testing.assert_array_equal(got, want & ~((ix >= 15) & (ix <= 20) & (iy >= 11) & (iy <= 15)))
    rect = resolve({"kind": "rect", "crs": "run_xy_m", "min": [float(xs[0]), float(ys[0])],
                    "max": [float(xs[2]), float(ys[2])]})
    np.testing.assert_array_equal(rect, want)
    rect_ll = resolve({"kind": "rect", "crs": "EPSG:4326", "min": [float(lon.min()), float(lat.min())],
                       "max": [float(lon.max()), float(lat.max())]})
    assert rect_ll.sum() >= want.sum() and (rect_ll & want).sum() == want.sum()   # a lon/lat box is wider


def test_circle_brute_force(run):
    rid, resolve, pred, meta = run
    x, y = pred["x_m"].to_numpy(), pred["y_m"].to_numpy()
    cx, cy = float(np.median(x)) + 7.0, float(np.median(y)) - 3.0
    for r in (0.0, 45.0, 200.0):
        want = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        np.testing.assert_array_equal(resolve({"kind": "circle", "crs": "run_xy_m", "center": [cx, cy],
                                               "radius_m": r}), want)
    lon, lat = _to_lonlat([cx], [cy])
    got = resolve({"kind": "circle", "crs": "EPSG:4326", "center": [float(lon[0]), float(lat[0])], "radius_m": 200.0})
    np.testing.assert_array_equal(got, (x - cx) ** 2 + (y - cy) ** 2 <= 200.0 ** 2)


class _DenseSource:
    """A full 300 × 300 grid of 10 m cells (no run needed)."""

    def __init__(self):
        from sparc.studio.runs.grid import build_grid

        iy, ix = np.mgrid[0:300, 0:300]
        x, y = 300000.0 + ix.ravel() * 10.0, 4600000.0 + iy.ravel() * 10.0
        self.grid = build_grid(x, y, 10.0, np.arange(x.size), None, {"data": {"crs": "EPSG:32619"}})

    @property
    def n(self):
        return self.grid.n


@pytest.mark.parametrize("radius", [150.0, 400.0, 1000.0])
def test_circle_area_within_3_percent(radius):
    src = _DenseSource()
    lon, lat = _to_lonlat([301500.0], [4601500.0])
    mask = S.resolve(src, {"kind": "circle", "crs": "EPSG:4326", "center": [float(lon[0]), float(lat[0])],
                           "radius_m": radius})
    area = mask.sum() * 100.0
    assert area == pytest.approx(math.pi * radius ** 2, rel=0.03)


def test_buffer_brute_force(run):
    rid, resolve, pred, meta = run
    x, y = pred["x_m"].to_numpy(), pred["y_m"].to_numpy()
    base_spec = {"kind": "top", "column": "obs", "k": 5, "direction": "highest"}
    base = resolve(base_spec)
    assert base.sum() == 5
    for r in (0.0, 30.0, 75.0, 160.0):
        d = np.sqrt((x[:, None] - x[base][None, :]) ** 2 + (y[:, None] - y[base][None, :]) ** 2).min(axis=1)
        np.testing.assert_array_equal(resolve({"kind": "buffer", "of": base_spec, "radius_m": r}), d <= r + 1e-6)
    rng = pd.read_json(Path(__file__).resolve().parents[1] / "fixtures" / "synth_run" / "influence.json",
                       typ="series")["ranges_m"]["canopy"]
    d = np.sqrt((x[:, None] - x[base][None, :]) ** 2 + (y[:, None] - y[base][None, :]) ** 2).min(axis=1)
    np.testing.assert_array_equal(resolve({"kind": "buffer", "of": base_spec, "lever_range": "canopy"}),
                                  d <= rng + 1e-6)


def test_hex_keys(run):
    from sparc.core.planner import hex_ids

    rid, resolve, pred, meta = run
    for size in (250, 500):
        keys = hex_ids(pred["x_m"].to_numpy(), pred["y_m"].to_numpy(), float(size))[0]
        pick = sorted(set(keys.tolist()))[1:4]
        np.testing.assert_array_equal(resolve({"kind": "hex", "size_m": size, "keys": pick}), np.isin(keys, pick))


def test_zones(client, demo, place_run):
    def zoned(rd: Path) -> None:
        p = pd.read_parquet(rd / "predictions.parquet")
        p["zone"] = (np.arange(len(p)) % 3 + 1).astype(float)
        p.loc[[4, 5], "zone"] = np.nan
        p.to_parquet(rd / "predictions.parquet")

    place_run(demo, edit=zoned)
    zone = (np.arange(1120) % 3 + 1).astype(float)
    zone[[4, 5]] = np.nan
    r = client.post(f"/api/runs/{RUN_ID}/selection/resolve",
                    json={"selection": {"kind": "zones", "values": [1, "3"]}})
    got = S.decode_bitset(r.json()["mask"], 1120)
    np.testing.assert_array_equal(got, (zone == 1) | (zone == 3))
    assert client.get(f"/api/runs/{RUN_ID}/grid").json()["zones"] == [1, 2, 3]


def test_zones_without_zones_is_422(run, client):
    rid = run[0]
    r = client.post(f"/api/runs/{rid}/selection/resolve", json={"selection": {"kind": "zones", "values": [1]}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"


def test_filters(run, client):
    rid, resolve, pred, meta = run
    obs = pred["target"].to_numpy()
    q = float(np.median(obs))
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "obs", "op": ">", "value": q}), obs > q)
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "pred:target", "op": "<=", "value": q}),
                                  obs <= q)
    lo, hi = np.quantile(obs, [0.2, 0.6])
    v = obs.astype(np.float32).astype(np.float64)              # layers travel as float32
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "layer:obs", "op": "between",
                                           "value": [float(hi), float(lo)]}), (v >= lo) & (v <= hi))
    fold = pred["fold"].to_numpy()
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "fold", "op": "in", "value": [0, 2]}),
                                  np.isin(fold, [0, 2]))
    canopy = f32(client.get(f"/api/runs/{rid}/layers/canopy.bin").content)
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "predictor:canopy", "op": ">=", "value": 20}),
                                  canopy >= 20)
    resp = pd.read_parquet(Path(__file__).resolve().parents[1] / "fixtures" / "synth_run" / "response_canopy.parquet")
    d90 = resp["d90"].to_numpy()
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "response:canopy:d90", "op": "<", "value": 30}),
                                  np.isfinite(d90) & (d90 < 30))
    sc = pd.read_parquet(Path(__file__).resolve().parents[1] / "fixtures" / "synth_run" / "scenario_deltas.parquet")
    v = sc["Canopy Increase +10"].to_numpy(np.float32)
    np.testing.assert_array_equal(resolve({"kind": "filter", "column": "configured:canopy-increase-plus-10",
                                           "op": "<", "value": -0.5}), v < -0.5)
    bad = client.post(f"/api/runs/{rid}/selection/resolve",
                      json={"selection": {"kind": "filter", "column": "nope:x", "op": ">", "value": 1}})
    assert bad.status_code == 422


def test_top_k_within(run):
    rid, resolve, pred, meta = run
    p = pred["pred"].to_numpy()
    fold = pred["fold"].to_numpy()
    within = {"kind": "filter", "column": "fold", "op": "==", "value": 1}
    got = resolve({"kind": "top", "column": "pred", "k": 10, "direction": "highest", "within": within})
    cand = np.flatnonzero(fold == 1)
    want = np.zeros(1120, bool)
    want[cand[np.argsort(-p[cand].astype(np.float32), kind="stable")[:10]]] = True
    np.testing.assert_array_equal(got, want)
    low = resolve({"kind": "top", "column": "pred", "frac": 0.1, "direction": "lowest"})
    assert low.sum() == 112 and p[low].max() <= p[~low].min()


def test_combinator_laws(run):
    rid, resolve, pred, meta = run
    A = {"kind": "filter", "column": "obs", "op": ">", "value": float(pred["target"].median())}
    B = {"kind": "filter", "column": "fold", "op": "==", "value": 0}
    C = {"kind": "top", "column": "pred", "frac": 0.3, "direction": "highest"}
    a, b, c = resolve(A), resolve(B), resolve(C)
    assert 0 < a.sum() < 1120 and 0 < b.sum() < 1120 and 0 < c.sum() < 1120
    np.testing.assert_array_equal(resolve({"op": "and", "args": [A, B]}), a & b)
    np.testing.assert_array_equal(resolve({"op": "or", "args": [A, B, C]}), a | b | c)
    np.testing.assert_array_equal(resolve({"op": "minus", "args": [A, B, C]}), a & ~(b | c))
    np.testing.assert_array_equal(resolve({"op": "not", "arg": A}), ~a)
    np.testing.assert_array_equal(resolve({"op": "not", "arg": {"op": "not", "arg": A}}), a)
    # De Morgan, commutativity, distributivity
    np.testing.assert_array_equal(resolve({"op": "not", "arg": {"op": "or", "args": [A, B]}}),
                                  resolve({"op": "and", "args": [{"op": "not", "arg": A}, {"op": "not", "arg": B}]}))
    np.testing.assert_array_equal(resolve({"op": "and", "args": [A, B]}), resolve({"op": "and", "args": [B, A]}))
    np.testing.assert_array_equal(resolve({"op": "and", "args": [A, {"op": "or", "args": [B, C]}]}),
                                  resolve({"op": "or", "args": [{"op": "and", "args": [A, B]},
                                                                {"op": "and", "args": [A, C]}]}))
    np.testing.assert_array_equal(resolve({"op": "minus", "args": [A, B]}),
                                  resolve({"op": "and", "args": [A, {"op": "not", "arg": B}]}))
    np.testing.assert_array_equal(resolve({"kind": "all"}), np.ones(1120, bool))


def test_cells_by_id(run):
    rid, resolve, pred, meta = run
    ids = pred["id"].to_numpy()
    np.testing.assert_array_equal(resolve({"kind": "cells", "ids": [int(ids[3]), str(ids[10]), 10 ** 9]}),
                                  np.isin(np.arange(1120), [3, 10]))


@pytest.mark.parametrize("n", [1, 7, 8, 9, 1120, 54701])
def test_bitset_codec_round_trip(n):
    rng = np.random.default_rng(n)
    mask = rng.random(n) < 0.3
    b64 = S.encode_bitset(mask)
    np.testing.assert_array_equal(S.decode_bitset(b64, n), mask)
    raw = S.mask_bytes(mask)
    assert len(raw) == (n + 7) // 8
    one = np.zeros(n, bool)
    one[0] = True
    assert S.mask_bytes(one)[0] == 1                          # LSB-first: row 0 is bit 0 of byte 0
    if n > 9:
        one[:] = False
        one[9] = True
        assert S.mask_bytes(one)[1] == 2
    with pytest.raises(Exception):
        S.decode_bitset(S.encode_bitset(np.zeros(n + 8, bool)), n)


def test_blobs_and_regions(run, client):
    rid, resolve, pred, meta = run
    mask = np.zeros(1120, bool)
    mask[[1, 2, 3, 500, 1119]] = True
    r = client.put(f"/api/runs/{rid}/blobs", params={"kind": "mask"}, content=S.mask_bytes(mask))
    assert r.status_code == 201 and r.json()["bytes"] == 140
    bid = r.json()["blob_id"]
    np.testing.assert_array_equal(resolve({"kind": "blob", "blob_id": bid}), mask)
    res = client.post(f"/api/runs/{rid}/selection/resolve", json={"selection": {"kind": "blob", "blob_id": bid}})
    assert res.json()["portable"] is False and res.json()["warnings"]
    assert client.put(f"/api/runs/{rid}/blobs", params={"kind": "mask"}, content=b"\x00" * 3).status_code == 422
    edit = np.array([1, 2], "<i4").tobytes() + np.array([0.5, 1.5], "<f4").tobytes()
    assert client.put(f"/api/runs/{rid}/blobs", params={"kind": "edit"}, content=edit,
                      headers={"X-SPARC-Count": "2"}).status_code == 201

    spec = {"kind": "filter", "column": "obs", "op": ">", "value": 88.5}
    reg = client.post(f"/api/runs/{rid}/regions", json={"name": "Hot cells", "spec": spec})
    assert reg.status_code == 201
    rg = reg.json()
    assert rg["n_cells"] == int((pred["target"].to_numpy() > 88.5).sum()) and rg["portable"] is True
    listed = client.get(f"/api/runs/{rid}/regions").json()
    assert [x["id"] for x in listed] == [rg["id"]]
    np.testing.assert_array_equal(resolve({"kind": "region", "id": rg["id"]}), resolve(spec))
    np.testing.assert_array_equal(resolve({"op": "and", "args": [{"kind": "region", "id": rg["id"]},
                                                                 {"kind": "blob", "blob_id": bid}]}),
                                  resolve(spec) & mask)
    assert client.delete(f"/api/runs/{rid}/regions/{rg['id']}").json() == {"ok": True}
    gone = client.post(f"/api/runs/{rid}/selection/resolve", json={"selection": {"kind": "region", "id": rg["id"]}})
    assert gone.status_code == 422


def test_resolve_summary(run, client):
    rid, resolve, pred, meta = run
    body = client.post(f"/api/runs/{rid}/selection/resolve",
                       json={"selection": {"kind": "filter", "column": "fold", "op": "==", "value": 2}}).json()
    sel = pred["fold"].to_numpy() == 2
    people = f32(client.get(f"/api/runs/{rid}/layers/people.bin").content)
    assert body["people"] == pytest.approx(float(np.nansum(people[sel])), rel=1e-5)
    assert body["medians"]["obs"] == pytest.approx(float(np.median(pred["target"].to_numpy(np.float32)[sel])),
                                                   rel=1e-6)
    assert body["portable"] is True and body["warnings"] == []
