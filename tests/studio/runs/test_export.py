"""Single-layer export (api.md §6.4 ``/export/layer/{key}``): GeoTIFF in the run's CRS with the grid's transform,
CSV, Parquet and GeoJSON; runs without a CRS get ``422 needs_crs`` for the formats that need one."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from tests.studio.runs.conftest import RUN_ID, f32


def test_geotiff_opens_with_crs_and_transform(client, fixture_run, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    rid, _ = fixture_run
    r = client.get(f"/api/runs/{rid}/export/layer/obs", params={"fmt": "tif"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/tiff"
    assert f'filename="{rid}_obs.tif"' in r.headers["content-disposition"]
    path = tmp_path / "obs.tif"
    path.write_bytes(r.content)
    meta = client.get(f"/api/runs/{rid}/grid").json()
    g = client.get(f"/api/runs/{rid}/grid.bin")
    offs = {o["name"]: o for o in json.loads(g.headers["x-sparc-offsets"])}
    ix = np.frombuffer(g.content, "<i4", 1120, offs["ix"]["offset"])
    iy = np.frombuffer(g.content, "<i4", 1120, offs["iy"]["offset"])
    obs = f32(client.get(f"/api/runs/{rid}/layers/obs.bin").content)
    dx = meta["dx_m"]
    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg() == 32619
        assert (ds.height, ds.width) == (meta["ny"], meta["nx"])
        t = ds.transform
        assert t.a == pytest.approx(dx) and t.e == pytest.approx(-dx) and t.b == 0 and t.d == 0
        assert t.c == pytest.approx(meta["x0_m"] - dx / 2)
        assert t.f == pytest.approx(meta["y0_m"] + (meta["ny"] - 1) * dx + dx / 2)
        band = ds.read(1)
        xs = meta["x0_m"] + ix * dx
        ys = meta["y0_m"] + iy * dx
        rows, cols = zip(*(ds.index(x, y) for x, y in zip(xs, ys)))
        np.testing.assert_array_equal(band[np.array(rows), np.array(cols)], obs)
        filled = np.zeros(band.shape, bool)
        filled[np.array(rows), np.array(cols)] = True
        assert np.isnan(band[~filled]).all()                     # cells outside the run are nodata


def test_csv_parquet_geojson(client, fixture_run, synth):
    rid, _ = fixture_run
    pred = pd.read_parquet(synth / "predictions.parquet")
    csv = pd.read_csv(io.StringIO(client.get(f"/api/runs/{rid}/export/layer/pred",
                                             params={"fmt": "csv"}).text))
    assert list(csv.columns) == ["id", "lon", "lat", "pred"]
    np.testing.assert_array_equal(csv["id"], pred["id"])
    np.testing.assert_allclose(csv["pred"], pred["pred"].to_numpy(np.float32), rtol=1e-6)
    pq = pd.read_parquet(io.BytesIO(client.get(f"/api/runs/{rid}/export/layer/pred",
                                               params={"fmt": "parquet"}).content))
    assert list(pq.columns) == ["id", "x_m", "y_m", "lon", "lat", "pred"]
    np.testing.assert_allclose(pq["x_m"], pred["x_m"])
    gj = client.get(f"/api/runs/{rid}/export/layer/fold", params={"fmt": "geojson"}).json()
    assert len(gj["features"]) == 1120 and gj["features"][0]["properties"]["fold"] == float(pred["fold"][0])
    assert client.get(f"/api/runs/{rid}/export/layer/pred", params={"fmt": "xlsx"}).status_code == 422
    assert client.get(f"/api/runs/{rid}/export/layer/nope", params={"fmt": "csv"}).status_code == 404


def test_run_without_crs_needs_crs(client, demo, place_run):
    def no_crs(raw: dict) -> None:
        raw["data"].pop("crs", None)
        raw["data"].pop("reproject_to", None)

    place_run(demo, edit_config=no_crs)
    meta = client.get(f"/api/runs/{RUN_ID}/grid").json()
    assert meta["crs"] is None and meta["has_lonlat"] is False and meta["corners"] is None
    for fmt in ("tif", "geojson"):
        r = client.get(f"/api/runs/{RUN_ID}/export/layer/obs", params={"fmt": fmt})
        assert r.status_code == 422 and r.json()["error"]["code"] == "needs_crs", fmt
    csv = client.get(f"/api/runs/{RUN_ID}/export/layer/obs", params={"fmt": "csv"})
    assert csv.status_code == 200 and csv.text.splitlines()[0] == "id,obs"
    r = client.get(f"/api/runs/{RUN_ID}/hex", params={"size": 250, "layers": "obs", "fmt": "gpkg"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_crs"
    poly = {"kind": "polygon", "crs": "EPSG:4326", "rings": [[[-71.4, 41.8], [-71.3, 41.8], [-71.3, 41.9]]]}
    r = client.post(f"/api/runs/{RUN_ID}/selection/resolve", json={"selection": poly})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_crs"
