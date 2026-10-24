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


def test_gdal_env_hands_the_ca_bundle_to_gdal_3_10(monkeypatch):
    """Behind a TLS-inspecting proxy GDAL must trust the system bundle; the GDAL 3.10 in rasterio 1.4 wheels reads
    GDAL_CURL_CA_BUNDLE and ignores CURL_CA_BUNDLE (the steward sandbox: 'self-signed certificate in chain')."""
    pytest.importorskip("rasterio")
    from sparc.core.opendata import gdal_env

    for v in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    opts = gdal_env().options
    assert opts["GDAL_CURL_CA_BUNDLE"] == opts["CURL_CA_BUNDLE"] == "/etc/ssl/certs/ca-certificates.crt"


def test_read_mosaic_joins_tiles_on_their_common_grid(tmp_path):
    """Multi-tile support (owner decision 13): a window spanning two WorldCover-style tiles is read from both,
    on their shared pixel grid; a missing tile (open sea) is skipped."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from sparc.core.opendata import read_mosaic

    px = 0.01
    paths = []
    for k, (west, val) in enumerate(((-72.0, 10), (-71.0, 50))):           # two 1°×1° tiles side by side
        p = tmp_path / f"tile{k}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(west, 42.0, px, px), nodata=0) as dst:
            dst.write(np.full((1, 100, 100), val, dtype="uint8"))
        paths.append(str(p))
    arr, tr, nod = read_mosaic(paths + [str(tmp_path / "missing.tif")], (-71.2, 41.5, -70.8, 41.7))
    assert arr.shape == (20, 40) and tr.a == pytest.approx(px) and tr.c == pytest.approx(-71.2)
    assert (arr[:, :20] == 10).all() and (arr[:, 20:] == 50).all() and nod == 0
