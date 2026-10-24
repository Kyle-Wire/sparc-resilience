"""scripts/setup_core_city.py: the pieces that turn a pilot city and its traverses into a core config."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def setup_mod():
    spec = importlib.util.spec_from_file_location("setup_core_city", REPO / "scripts" / "setup_core_city.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pts():
    t = pd.to_datetime(["2021-07-09 06:00:24", "2021-07-09 06:53:29", "2021-07-09 15:00:00", "2021-07-09 16:00:00"])
    return pd.DataFrame({"time": t, "lat": [35.0, 35.2, 35.1, 35.15], "lon": [-106.7, -106.5, -106.6, -106.55],
                         "window": ["morning", "morning", "midday", "midday"],
                         "run": ["2021-07-09 morning"] * 2 + ["2021-07-09 midday"] * 2})


def test_pilot_entry_and_campaign_from_the_traverse_clock(setup_mod):
    e = setup_mod.pilot_entry("albuquerque_nm", REPO / "configs" / "multicity_pilot.yml")
    assert e["city"] == "Albuquerque, NM" and e["utm_epsg"] == "EPSG:32613" and e["timezone"] == "America/Denver"
    with pytest.raises(SystemExit):
        setup_mod.pilot_entry("atlantis", REPO / "configs" / "multicity_pilot.yml")
    camp = setup_mod.campaign_from_traverses(_pts())
    assert camp["date"] == "2021-07-09" and camp["hours"] == [15, 16]
    assert camp["runs"] == ["2021-07-09 midday", "2021-07-09 morning"]
    with pytest.raises(ValueError, match="no afternoon run"):
        setup_mod.campaign_from_traverses(_pts().query("window == 'morning'"))
    with pytest.raises(ValueError, match="no timestamps"):      # Scranton 2023: t_f, rh, hi_f and positions only
        setup_mod.campaign_from_traverses(_pts().assign(time=pd.NaT))


def test_study_bbox_is_the_pilot_bbox_clipped_to_the_traverses(setup_mod):
    dx = 1 / (111.32 * np.cos(np.radians(35.125)))
    b = setup_mod.study_bbox([-106.78, 35.04, -106.52, 35.22], _pts(), margin_km=1.0)
    assert b == pytest.approx([-106.7 - dx, 35.04, -106.52, 35.2 + 1 / 111.32], abs=1e-4)
    with pytest.raises(ValueError):
        setup_mod.study_bbox([-90.0, 30.0, -89.0, 31.0], _pts())


def test_nearest_isd_station_needs_a_wban_and_an_active_record(setup_mod):
    hist = pd.DataFrame({
        "USAF": ["723653", "723650", "999999", "720000"],
        "WBAN": ["99999", "23050", "23050", "12345"],
        "STATION NAME": ["RADAR", "ALBUQUERQUE INTL SUNPORT AIRPORT", "MUNICIPAL", "CLOSED FIELD"],
        "ICAO": ["KZAB", "KABQ", "KABQ", "KXXX"],
        "LAT": ["+35.167", "+35.042", "+35.042", "+35.10"], "LON": ["-106.567", "-106.615", "-106.616", "-106.60"],
        "BEGIN": ["20100512", "19410701", "19710101", "19500101"], "END": ["20251231", "20250827", "19721231", "19991231"]})
    st = setup_mod.nearest_isd_station(hist, 35.1, -106.6, "2021-07-09")
    assert st["station"] == "72365023050" and st["ghcn"] == "USW00023050" and st["icao"] == "KABQ"


def test_grid_table_keeps_cells_on_capas_map(setup_mod):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    net = gpd.GeoDataFrame({"aat_midday": [99.0, np.nan, 100.5], "aat_morning": [70.0, 71.0, np.nan]},
                           geometry=[box(0, 0, 30, 30), box(30, 0, 60, 30), box(60, 0, 90, 30)], crs="EPSG:32613")
    df = setup_mod.grid_table(net)
    assert df["id"].tolist() == [1, 2] and df["x"].tolist() == [15.0, 75.0] and df["y"].tolist() == [15.0, 15.0]
    assert df["aat_midday"].tolist() == [99.0, 100.5] and (df["placeholder"] == 0.0).all()


def test_core_config_validates_and_points_identify_at_the_city(setup_mod, tmp_path):
    from sparc.core.config import core_config_from_dict

    camp = setup_mod.campaign_from_traverses(_pts())
    st = {"station": "72365023050", "ghcn": "USW00023050"}
    core = setup_mod.core_config("albuquerque_nm", "Albuquerque, NM", "EPSG:32613", camp, None, st)
    text = setup_mod.config_text(core, "albuquerque_nm", camp, [-106.78, 35.04, -106.52, 35.22])
    assert text.startswith("# SPARC core config for Albuquerque, NM")
    cfg = core_config_from_dict(yaml.safe_load(text), base_dir=tmp_path / "configs")
    assert cfg.name == "albuquerque_nm_uhi" and cfg.data["target"] == "aat_midday" and cfg.coord_scale == 1.0
    assert cfg.physics_role("canopy") == "open_canopy" and cfg.physics_role("impervious") == "open_impervious"
    assert cfg.output_dir == (tmp_path / "output" / "core" / "albuquerque_nm").resolve()
    assert cfg.raw["planner"]["ghcn_station"] == "USW00023050"
