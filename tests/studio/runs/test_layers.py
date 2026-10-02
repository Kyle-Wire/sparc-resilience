"""Grid, layers, folds, cells and hexagons on the wire (api.md §0.4–0.5, §6.2–6.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studio.runs.conftest import RUN_ID, f32

IMMUTABLE = "private, max-age=31536000, immutable"


def _offsets(r) -> dict:
    return {o["name"]: o for o in json.loads(r.headers["x-sparc-offsets"])}


def _unpack(r, name: str) -> np.ndarray:
    o = _offsets(r)[name]
    dt = {"int32": "<i4", "float32": "<f4", "int16": "<i2", "uint8": "u1"}[o["dtype"]]
    return np.frombuffer(r.content, dt, o["length"], o["offset"])


def test_layer_catalog(client, fixture_run):
    rid, _ = fixture_run
    groups = {g["id"]: g for g in client.get(f"/api/runs/{rid}/layers").json()["groups"]}
    assert {"temperature", "inputs", "cv", "effects", "causal", "scenarios", "budget", "planner"} <= set(groups)
    keys = {m["key"]: m for g in groups.values() for m in g["layers"]}
    for k in ("obs", "pred", "resid", "halfwidth", "canopy", "fold", "fp_canopy", "cls_canopy", "cate_canopy",
              "sc:canopy-increase-plus-10", "sc_sd:canopy-increase-plus-10", "alloc_dose", "people"):
        assert k in keys, k
    assert keys["fold"]["dtype"] == "uint8" and keys["cls_canopy"]["dtype"] == "uint8"
    assert keys["cls_canopy"]["labels"] and keys["cls_canopy"]["scale"] == "cat"
    assert keys["sc:canopy-increase-plus-10"]["scale"] == "div" and keys["sc:canopy-increase-plus-10"]["center"] == 0
    assert keys["obs"]["unit"] == "°F" and keys["obs"]["stats"]["n"] == 1120
    pred = pd.read_parquet(Path(fixture_run[1]) / "predictions.parquet")
    assert keys["obs"]["stats"]["p50"] == pytest.approx(float(np.median(pred["target"])), rel=1e-5)


@pytest.mark.parametrize("key,column", [("obs", "target"), ("pred", "pred"), ("dist_train_m", "dist_train_m")])
def test_float_layers_match_their_source(client, fixture_run, synth, key, column):
    rid, _ = fixture_run
    r = client.get(f"/api/runs/{rid}/layers/{key}.bin")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-sparc-dtype"] == "float32" and r.headers["x-sparc-length"] == "1120"
    v = f32(r.content)
    assert v.size == 1120
    src = pd.read_parquet(synth / "predictions.parquet")[column].to_numpy(np.float32)
    np.testing.assert_array_equal(v, src)


def test_derived_and_scenario_layers(client, fixture_run, synth):
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    resid = f32(client.get(f"/api/runs/{rid}/layers/resid.bin").content)
    np.testing.assert_allclose(resid, (pred["target"] - pred["pred"]).to_numpy(np.float32), atol=1e-4)
    sc = pd.read_parquet(synth / "scenario_deltas.parquet")
    v = f32(client.get(f"/api/runs/{rid}/layers/sc:canopy-increase-plus-10.bin").content)
    np.testing.assert_array_equal(v, sc["Canopy Increase +10"].to_numpy(np.float32))
    with np.load(synth / "scenario_detail.npz") as z:
        names = json.loads(bytes(z["names"]).decode()) if z["names"].dtype == np.uint8 else list(z["names"])
        i = list(names).index("Canopy Increase +10")
        sd = np.asarray(z[f"sd{i}"], np.float32)
    np.testing.assert_array_equal(f32(client.get(f"/api/runs/{rid}/layers/sc_sd:canopy-increase-plus-10.bin")
                                      .content), sd)
    fold = np.frombuffer(client.get(f"/api/runs/{rid}/layers/fold.bin").content, np.uint8)
    np.testing.assert_array_equal(fold, pred["fold"].to_numpy())
    cls = np.frombuffer(client.get(f"/api/runs/{rid}/layers/cls_canopy.bin").content, np.uint8)
    assert cls.size == 1120 and set(np.unique(cls)) <= {0, 1, 2, 3, 255}
    assert client.get(f"/api/runs/{rid}/layers/nope.bin").status_code == 404


def test_nan_round_trip(client, demo, place_run, synth):
    def holes(rd: Path) -> None:
        p = pd.read_parquet(rd / "predictions.parquet")
        p.loc[[0, 7, 1119], "pred"] = np.nan
        p.to_parquet(rd / "predictions.parquet")

    place_run(demo, edit=holes)
    v = f32(client.get(f"/api/runs/{RUN_ID}/layers/pred.bin").content)
    assert v.size == 1120 and np.flatnonzero(np.isnan(v)).tolist() == [0, 7, 1119]
    src = pd.read_parquet(synth / "predictions.parquet")["pred"].to_numpy(np.float32)
    ok = ~np.isnan(v)
    np.testing.assert_array_equal(v[ok], src[ok])


def test_caching_headers_and_304(client, fixture_run):
    rid, _ = fixture_run
    r = client.get(f"/api/runs/{rid}/layers/obs.bin")
    etag = r.headers["etag"]
    assert etag.startswith('"') and r.headers["cache-control"] == IMMUTABLE
    again = client.get(f"/api/runs/{rid}/layers/obs.bin", headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == etag
    other = client.get(f"/api/runs/{rid}/layers/pred.bin", headers={"If-None-Match": etag})
    assert other.status_code == 200 and other.headers["etag"] != etag
    for path in ("grid.bin", "folds/0.bin", "grid/ids.bin"):
        r = client.get(f"/api/runs/{rid}/{path}")
        assert r.status_code == 200 and r.headers["cache-control"] == IMMUTABLE, path
        assert client.get(f"/api/runs/{rid}/{path}", headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_layers_that_can_change_are_revalidated(client, ctx, fixture_run, synth, tmp_path):
    """A finished run's own outputs are immutable; layers rebuilt from project files (the data, planner
    layers) and every layer of a run imported in place are revalidated (``no-cache`` + ETag → 304)."""
    rid, _ = fixture_run
    for key, cc in (("pred", IMMUTABLE), ("fp_canopy", IMMUTABLE), ("canopy", "private, no-cache"),
                    ("people", "private, no-cache")):
        r = client.get(f"/api/runs/{rid}/layers/{key}.bin")
        assert r.status_code == 200 and r.headers["cache-control"] == cc, key
        again = client.get(f"/api/runs/{rid}/layers/{key}.bin", headers={"If-None-Match": r.headers["etag"]})
        assert again.status_code == 304 and again.headers["cache-control"] == cc, key
    import shutil

    src = tmp_path / "cli" / "run"
    shutil.copytree(synth, src, ignore=shutil.ignore_patterns("events.jsonl", "FIXTURE.json"))
    row = ctx.services["registry"].index_run_dir(src, origin="imported")
    for path in ("layers/pred.bin", "grid.bin"):
        r = client.get(f"/api/runs/{row['id']}/{path}")
        assert r.status_code == 200 and r.headers["cache-control"] == "private, no-cache", path
        assert client.get(f"/api/runs/{row['id']}/{path}",
                          headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_input_layers_follow_the_data_file(client, demo, fixture_run):
    """Input layers are rebuilt from the snapshot's data file: replacing it (same ids) changes the values and the
    ETag, so an immutable cached copy is never served for the new data."""
    rid, _ = fixture_run
    first = client.get(f"/api/runs/{rid}/layers/canopy.bin")
    assert first.status_code == 200
    data = Path(demo["dir"]) / "data" / "city.csv"
    df = pd.read_csv(data)
    df["canopy"] = df["canopy"] * 0.5
    df.to_csv(data, index=False)
    second = client.get(f"/api/runs/{rid}/layers/canopy.bin", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 200 and second.headers["etag"] != first.headers["etag"]
    np.testing.assert_allclose(f32(second.content), f32(first.content) * 0.5, rtol=1e-5, equal_nan=True)


def test_grid_meta_and_packed_grid(client, fixture_run, synth):
    from pyproj import Transformer

    rid, _ = fixture_run
    meta = client.get(f"/api/runs/{rid}/grid").json()
    assert meta["n"] == 1120 and (meta["ny"], meta["nx"]) == (42, 42) and meta["dx_m"] == 30.0
    assert meta["crs"] == "EPSG:32619" and meta["has_lonlat"] and meta["ids_kind"] == "int"
    assert meta["zones"] == [] and meta["n_folds"] == 3 and meta["units"]["target"] == "°F"
    r = client.get(f"/api/runs/{rid}/grid.bin")
    offs = _offsets(r)
    assert [o["name"] for o in offs.values()] == ["ix", "iy", "lon", "lat", "zone"]
    assert all(o["offset"] % 8 == 0 and o["length"] == 1120 for o in offs.values())
    assert offs["zone"]["dtype"] == "int16" and (_unpack(r, "zone") == -1).all()
    pred = pd.read_parquet(synth / "predictions.parquet")
    tr = Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(pred["x_m"].to_numpy(), pred["y_m"].to_numpy())
    np.testing.assert_allclose(_unpack(r, "lon"), lon, atol=1e-5)
    np.testing.assert_allclose(_unpack(r, "lat"), lat, atol=1e-5)
    ix, iy = _unpack(r, "ix"), _unpack(r, "iy")
    np.testing.assert_allclose(meta["x0_m"] + ix * meta["dx_m"], pred["x_m"], atol=1e-6)
    np.testing.assert_allclose(meta["y0_m"] + iy * meta["dx_m"], pred["y_m"], atol=1e-6)
    # corners are [lat, lon] of the corner cell centres (sw = cell (0, 0), ne = (nx-1, ny-1))
    x1, y1 = meta["x0_m"] + (meta["nx"] - 1) * 30.0, meta["y0_m"] + (meta["ny"] - 1) * 30.0
    for name, (x, y) in {"sw": (meta["x0_m"], meta["y0_m"]), "ne": (x1, y1), "nw": (meta["x0_m"], y1),
                         "se": (x1, meta["y0_m"])}.items():
        lo, la = tr.transform(x, y)
        assert meta["corners"][name] == pytest.approx([la, lo], abs=1e-6)
    west, south, east, north = meta["bounds_lonlat"]
    assert west == pytest.approx(lon.min(), abs=1e-5) and north == pytest.approx(lat.max(), abs=1e-5)
    ids = np.frombuffer(client.get(f"/api/runs/{rid}/grid/ids.bin").content, "<i8")
    np.testing.assert_array_equal(ids, pred["id"].to_numpy())
    assert meta["etag"] and client.get(f"/api/runs/{rid}/grid").json()["etag"] == meta["etag"]


def test_zones_and_string_ids(client, demo, place_run):
    codes = np.array(["Downtown", "Fox Point", None, "Elmhurst"], dtype=object)

    def zoned(rd: Path) -> None:
        p = pd.read_parquet(rd / "predictions.parquet")
        p["zone"] = codes[np.arange(len(p)) % 4]
        p["id"] = ["c" + str(i) for i in p["id"]]
        p.to_parquet(rd / "predictions.parquet")

    place_run(demo, edit=zoned)
    meta = client.get(f"/api/runs/{RUN_ID}/grid").json()
    assert meta["zones"] == ["Downtown", "Elmhurst", "Fox Point"] and meta["ids_kind"] == "str"
    r = client.get(f"/api/runs/{RUN_ID}/grid.bin")
    zone = _unpack(r, "zone")
    want = np.array([0, 2, -1, 1] * 280, dtype=np.int16)
    np.testing.assert_array_equal(zone, want)
    assert client.get(f"/api/runs/{RUN_ID}/grid/ids.bin").status_code == 409
    assert client.get(f"/api/runs/{RUN_ID}/grid/ids.json").json()[:2] == ["c0", "c1"]
    cell = client.get(f"/api/runs/{RUN_ID}/cells/1").json()
    assert cell["zone"] == "Fox Point" and cell["id"] == "c1"


def test_folds_match_the_core_design(client, fixture_run, synth):
    from sparc.core.cv import make_spatial_folds

    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    st = json.loads((synth / "run_state.json").read_text())["meta"]["cv"]
    folds = make_spatial_folds(pred[["x_m", "y_m"]].to_numpy(), n_folds=st["n_folds"], block_m=st["block_m"],
                               buffer_m=st["buffer_m"], seed=st["seed"])
    for k in range(3):
        cls = np.frombuffer(client.get(f"/api/runs/{rid}/folds/{k}.bin").content, np.uint8)
        np.testing.assert_array_equal(cls == 0, folds.train_masks[k])
        np.testing.assert_array_equal(cls == 1, folds.test_masks[k])
        assert (cls == 2).sum() == (~folds.train_masks[k] & ~folds.test_masks[k]).sum() > 0
    assert client.get(f"/api/runs/{rid}/folds/x.bin").status_code == 404


def test_cell_inspector(client, fixture_run, synth):
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    resp = pd.read_parquet(synth / "response_canopy.parquet")
    i = 5
    cell = client.get(f"/api/runs/{rid}/cells/{i}", params={"scenarios": "configured:cooling-package"}).json()
    assert cell["index"] == i and cell["id"] == int(pred["id"][i])
    assert cell["values"]["obs"] == pytest.approx(float(pred["target"][i]), rel=1e-6)
    assert cell["values"]["canopy"] is not None and cell["values"]["fold"] == float(pred["fold"][i])
    sc = pd.read_parquet(synth / "scenario_deltas.parquet")
    assert cell["scenarios"]["configured:canopy-increase-plus-10"] == pytest.approx(
        float(sc["Canopy Increase +10"][i]), rel=1e-6)
    assert cell["scenarios"]["configured:cooling-package"] == pytest.approx(float(sc["Cooling package"][i]),
                                                                           rel=1e-6)
    curve = cell["curves"]["canopy"]
    row = resp.iloc[i]
    assert curve["model"] == row["curve_model"] == "saturating"
    d = np.asarray(curve["dose"])
    want = row["max_cooling_A"] * (1 - np.exp(-d / row["saturation_scale_ds"]))
    np.testing.assert_allclose(curve["benefit"], want, rtol=1e-6)
    assert curve["benefit"][0] == 0.0 and d[-1] == 40.0
    # the stored d90 is where the fitted curve reaches 90 % of A
    assert row["max_cooling_A"] * (1 - np.exp(-row["d90"] / row["saturation_scale_ds"])) == pytest.approx(
        0.9 * row["max_cooling_A"], rel=1e-6)
    assert client.get(f"/api/runs/{rid}/cells/1120").status_code == 404


def test_hex_aggregates(client, fixture_run, synth):
    from sparc.core.planner import hex_ids

    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    res = client.get(f"/api/runs/{rid}/hex", params={"size": 250, "layers": "obs,pred", "sums": "people"}).json()
    key, _, _ = hex_ids(pred["x_m"].to_numpy(), pred["y_m"].to_numpy(), 250.0)
    df = pd.DataFrame({"key": key, "obs": pred["target"]}).groupby("key")
    want_n, want_obs = df.size(), df["obs"].mean()
    got = {h["key"]: h for h in res["hex"]}
    assert set(got) == set(want_n.index)
    for k, h in got.items():
        assert h["n_cells"] == want_n[k]
        assert h["values"]["obs"] == pytest.approx(want_obs[k], rel=1e-5)
        assert h["values"]["people"] is not None and h["lon"] is not None
    people = f32(client.get(f"/api/runs/{rid}/layers/people.bin").content)
    assert sum(h["values"]["people"] for h in res["hex"]) == pytest.approx(float(np.nansum(people)), rel=1e-5)

    csv = client.get(f"/api/runs/{rid}/hex", params={"size": 500, "layers": "obs", "fmt": "csv"})
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("key,")
    gj = client.get(f"/api/runs/{rid}/hex", params={"size": 250, "layers": "obs", "fmt": "geojson"}).json()
    assert len(gj["features"]) == len(got) and gj["features"][0]["geometry"]["type"] == "Polygon"
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert len(ring) == 7 and ring[0] == ring[-1]
    gpkg = client.get(f"/api/runs/{rid}/hex", params={"size": 250, "layers": "obs", "fmt": "gpkg"})
    assert gpkg.status_code == 200 and gpkg.content[:16] == b"SQLite format 3\x00"
    assert client.get(f"/api/runs/{rid}/hex", params={"size": 300}).status_code == 422


def test_providence_grids(client, ctx, providence_runs, tmp_path):
    """The recorded Providence runs: the fast run imports in place (causal.json and allocation.parquet are older
    than its manifest, which lacks their sections, so both are stale); grid shapes (ny, nx) are (100, 86) fast and
    (334, 287) full."""
    import shutil

    fast = providence_runs / "providence_uhi_fast"
    r = client.post("/api/runs/import", json={"dir": str(fast)})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    states = {o["id"]: o["state"] for o in client.get(f"/api/runs/{rid}/outputs").json()["outputs"]}
    assert states["causal"] == "stale" and states["allocation"] == "stale"
    assert states["predictions"] == "present"
    meta = client.get(f"/api/runs/{rid}/grid").json()
    assert (meta["ny"], meta["nx"]) == (100, 86)
    v = f32(client.get(f"/api/runs/{rid}/layers/pred.bin").content)
    assert v.size == meta["n"]

    full = providence_runs / "providence_uhi"
    if not (full / "manifest.json").is_file():
        pytest.skip("no full Providence run")
    # a light copy (the full run's provenance points at another checkout's configs)
    light = tmp_path / "providence_uhi"
    light.mkdir()
    for name in ("predictions.parquet", "influence.json", "run_state.json"):
        if (full / name).is_file():
            shutil.copy2(full / name, light / name)
    m = json.loads((full / "manifest.json").read_text())
    m["provenance"]["config_dir"] = str(tmp_path / "absent")
    (light / "manifest.json").write_text(json.dumps(m))
    row = ctx.services["registry"].index_run_dir(light, origin="imported")
    meta = client.get(f"/api/runs/{row['id']}/grid").json()
    assert (meta["ny"], meta["nx"]) == (334, 287)
