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
