"""Open-data feature helpers (offline): UTM/MGRS bookkeeping, tiles, joins, comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.core import features_open as FO


def test_utm_and_tile_names():
    assert FO.utm_epsg(-71.4, 41.8) == 32619 and FO.utm_epsg(151.2, -33.9) == 32756
    assert FO.dem_tiles((-71.46, 41.80, -71.36, 41.86)) == ["Copernicus_DSM_COG_10_N41_00_W072_00_DEM"]
    from sparc.core.opendata import worldcover_tiles

    assert worldcover_tiles((-71.46, 41.80, -71.36, 41.86)) == ["N39W072"]


def test_albedo_coefficients_sum_to_a_broadband_weighting():
    assert sum(FO.ALBEDO_COEF.values()) == pytest.approx(0.9699, abs=1e-4)
    assert set(FO.ALBEDO_COEF) == set(FO.S2_BANDS)


def test_compare_features_and_data_join(tmp_path, synthetic_city):
    from sparc.core.config import core_config_from_dict
    from sparc.core.data import load_core_data
    from sparc.core.synthetic import synthetic_city_config

    df = synthetic_city.frame.assign(oid=np.arange(len(synthetic_city.frame)))
    df.to_csv(tmp_path / "city.csv", index=False)
    rng = np.random.default_rng(0)
    pd.DataFrame({"id": df["oid"], "open_canopy": df["canopy"] + rng.normal(0, 5, len(df))}).to_parquet(tmp_path / "f.parquet")
    raw = synthetic_city_config()
    raw["data"].update(path="city.csv", id="oid", join=[{"path": "f.parquet", "key": "oid", "right_key": "id"}])
    raw["predictors"] = raw["predictors"] + ["open_canopy"]
    cfg = core_config_from_dict(raw, base_dir=tmp_path)
    data = load_core_data(cfg)
    assert "open_canopy" in data.frame and np.isfinite(data.frame["open_canopy"]).all()
    cmp = FO.compare_features(pd.DataFrame({"canopy": data.frame["open_canopy"]}), data, {"canopy": "canopy"})
    assert cmp[0]["pearson_r"] > 0.8 and cmp[0]["role"] == "canopy"


def test_mgrs_square_without_the_mgrs_package():
    # 100 km squares of known places (Sentinel-2 tile names): Providence's two, New York, Los Angeles, Chicago
    assert FO.mgrs_square(-71.0, 41.8) == "19TCG" and FO.mgrs_square(-71.6, 41.8) == "19TBG"
    assert FO.mgrs_square(-74.0, 40.7) == "18TWL" and FO.mgrs_square(-118.24, 34.05) == "11SLT"
    assert FO.mgrs_square(-87.63, 41.88) == "16TDM"
    tiles = FO.mgrs_tiles((-71.5, 41.75, -71.35, 41.9), 0)
    assert set(tiles) >= {"19TBG", "19TCG"} and all(len(t) == 5 for t in tiles)
    with pytest.raises(ValueError):
        FO.mgrs_square(0.0, 85.0)


def test_s2_composite_in_row_blocks_equals_the_whole_stack(tmp_path, monkeypatch):
    """R15: the Sentinel-2 median is taken in blocks of rows (memory bounded by the block, not area × scenes);
    the result must equal the composite of the whole stack at once and the plain per-pixel median."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    from sparc.core import features_open as F

    bounds = (-71.43, 41.80, -71.40, 41.83)
    epsg = F.utm_epsg(-71.415, 41.815)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([bounds[0], bounds[2], bounds[0], bounds[2]], [bounds[1], bounds[1], bounds[3], bounds[3]])
    res = 10.0
    x0, y1 = np.floor(min(xs) / res) * res, np.ceil(max(ys) / res) * res
    W = int(round((np.ceil(max(xs) / res) * res - x0) / res))
    H = int(round((y1 - np.floor(min(ys) / res) * res) / res))
    T = from_origin(x0, y1, res, res)
    rng = np.random.default_rng(0)
    scenes, ref = [], {b: [] for b in F.S2_BANDS}
    for k in range(4):
        d = tmp_path / f"s{k}"
        d.mkdir()
        scl = np.full((H, W), 4, np.uint8)
        scl[rng.random((H, W)) < 0.2] = 9                                      # 20% cloud per scene
        if k == 3:
            scl[:] = 9                                                          # a fully cloudy scene: dropped
        for band in F.S2_BANDS + ("SCL",):
            arr = scl if band == "SCL" else rng.integers(200, 4000, (H, W)).astype(np.uint16)
            with rasterio.open(d / f"{band}.tif", "w", driver="GTiff", width=W, height=H, count=1,
                               dtype=arr.dtype, crs=f"EPSG:{epsg}", transform=T) as dst:
                dst.write(arr, 1)
            if band != "SCL" and k < 3:
                a = arr.astype(np.float32) / 10000.0
                a[np.isin(scl, F.SCL_BAD)] = np.nan
                ref[band].append(a)
        scenes.append({"name": f"s{k}", "cloud": 5.0, "prefix": str(d) + "/"})
    monkeypatch.setattr(F, "_s2_href", lambda sc, band: sc["prefix"] + f"{band}.tif")
    whole = F.s2_composite(scenes, bounds, memory_mb=1e4)
    blocks = F.s2_composite(scenes, bounds, memory_mb=0.05)
    assert whole["scenes"] == blocks["scenes"] == ["s0", "s1", "s2"]
    assert blocks["block_rows"] < H <= whole["block_rows"]
    for k in ("ndvi", "albedo"):
        assert np.array_equal(whole[k], blocks[k], equal_nan=True), k
    with np.errstate(all="ignore"):
        med = {b: np.nanmedian(np.stack(v), axis=0) for b, v in ref.items()}
    ndvi = (med["B08"] - med["B04"]) / (med["B08"] + med["B04"])
    assert np.allclose(blocks["ndvi"], ndvi, atol=1e-5, equal_nan=True)


def _tile(path, west, north, values, res=0.25, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    arr = np.asarray(values, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(west, north, res, res),
                       nodata=nodata) as dst:
        dst.write(arr, 1)
    return str(path)


def test_read_mosaic_joins_tiles_on_their_own_pixel_grid(tmp_path):
    """A window across a tile edge reads both tiles without resampling; pixels outside every tile are the fill."""
    from sparc.core.opendata import read_mosaic

    west = _tile(tmp_path / "w.tif", 0.0, 1.0, np.arange(16).reshape(4, 4))
    east = _tile(tmp_path / "e.tif", 1.0, 1.0, 100 + np.arange(16).reshape(4, 4))
    arr, tr, nod = read_mosaic([west, east], (0.6, 0.1, 1.4, 0.9), fill=float("nan"))
    assert (tr.a, tr.e, tr.c, tr.f) == (0.25, -0.25, 0.5, 1.0)           # bounds snapped outwards to the grid
    assert arr.shape == (4, 4)
    np.testing.assert_array_equal(arr[:, :2], np.arange(16).reshape(4, 4)[:, 2:])
    np.testing.assert_array_equal(arr[:, 2:], 100 + np.arange(16).reshape(4, 4)[:, :2])
    arr, tr, nod = read_mosaic([west, east], (0.6, -0.4, 1.4, 0.9), fill=float("nan"))
    assert np.isnan(nod) and np.isnan(arr[-2:]).all() and np.isfinite(arr[:-2]).all()


def test_read_mosaic_skips_missing_tiles_only_for_http_404(tmp_path, monkeypatch):
    import rasterio
    from rasterio.errors import RasterioIOError

    from sparc.core.opendata import read_mosaic

    west = _tile(tmp_path / "w.tif", 0.0, 1.0, np.ones((4, 4)))
    real = rasterio.open

    def fake(path, *a, **k):
        if "ocean" in str(path):
            raise RasterioIOError("HTTP response code: 404")
        if "blocked" in str(path):
            raise RasterioIOError("HTTP response code: 403")
        return real(path, *a, **k)

    monkeypatch.setattr(rasterio, "open", fake)
    arr, _, _ = read_mosaic([west, "https://example.org/ocean.tif"], (0.6, 0.1, 1.4, 0.9), fill=float("nan"))
    assert np.isfinite(arr[:, :2]).all() and np.isnan(arr[:, 2:]).all()
    with pytest.raises(RasterioIOError, match="403"):
        read_mosaic([west, "https://example.org/blocked.tif"], (0.6, 0.1, 1.4, 0.9))
    with pytest.raises(ValueError, match="pixel sizes"):
        read_mosaic([west, _tile(tmp_path / "fine.tif", 1.0, 1.0, np.ones((8, 8)), res=0.125)], (0.6, 0.1, 1.4, 0.9))
