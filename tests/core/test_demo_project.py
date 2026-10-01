"""The synthetic demo project (SPEC §9.2, §11 item 20): one generator for the Studio template, the committed
fixture and the tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def demo_pair(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    a = write_demo_project(tmp_path_factory.mktemp("demo_a"), n=24, seed=3)
    b = write_demo_project(tmp_path_factory.mktemp("demo_b"), n=24, seed=3)
    return a, b


def test_demo_project_is_byte_deterministic(demo_pair, tmp_path):
    from sparc.core.synthetic import write_demo_project

    a, b = demo_pair
    assert set(a["files"]) == {"data", "truth", "layers", "climate", "config"}
    for key in a["files"]:
        assert a["files"][key].read_bytes() == b["files"][key].read_bytes(), key
    other = write_demo_project(tmp_path, n=24, seed=4)
    assert other["files"]["data"].read_bytes() != a["files"]["data"].read_bytes()


def test_demo_project_has_crs_and_loads(demo_pair):
    from sparc.core.config import load_core_config
    from sparc.core.data import load_core_data
    from sparc.core.synthetic import DEMO_CRS

    a, _ = demo_pair
    cfg = load_core_config(a["config_path"])
    assert cfg.data["crs"] == DEMO_CRS == "EPSG:32619"
    assert cfg.data["path"] == "data/city.csv"                       # relative to the project directory
    assert cfg.raw["climate"]["enabled"] and cfg.raw["climate"]["source"] == "table"
    assert cfg.raw["causal"]["treatments"] == ["canopy"] and cfg.raw["optimize"]["budget"]
    assert set(cfg.actionable) == {"canopy", "impervious", "albedo"}
    assert cfg.resolve_path(cfg.raw["planner"]["layers"]).exists()
    data = load_core_data(cfg)
    df = pd.read_csv(a["files"]["data"])
    np.testing.assert_array_equal(data.ids, df["id"].to_numpy())
    assert df["x"].min() > 300_000 and df["y"].min() > 4_630_000      # offset to the fictional site
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(DEMO_CRS, "EPSG:4326", always_xy=True).transform(df["x"].mean(), df["y"].mean())
    clim = pd.read_csv(a["files"]["climate"])
    assert clim["site_lat"].iloc[0] == pytest.approx(lat, abs=1e-3)
    assert clim["site_lon"].iloc[0] == pytest.approx(lon, abs=1e-3)
    assert clim["model"].nunique() == 6 and clim["experiment"].nunique() == 4 and clim["period"].nunique() == 3
    lay = pd.read_parquet(a["files"]["layers"])
    np.testing.assert_array_equal(lay["id"].to_numpy(), df["id"].to_numpy())
    lc = lay[[c for c in lay.columns if c.startswith("lc_")]].sum(axis=1)
    np.testing.assert_allclose(lc, 1.0, atol=2e-3)
    assert (lay["people_60_plus"] <= lay["people"] + 1e-9).all() and lay["people"].sum() > 0


def test_truth_matches_true_response(demo_pair):
    from sparc.core.synthetic import make_synthetic_city

    a, _ = demo_pair
    truth = json.loads(a["files"]["truth"].read_text())
    derived = truth["derived"]
    assert derived["n"] == 24 and derived["seed"] == 3
    city = make_synthetic_city(n=24, seed=3)
    ts = derived["true_scenarios"]
    assert set(ts) == {"Canopy Increase +5", "Canopy Increase +10", "Canopy Increase +20"}
    for name, value in ts.items():
        d = float(name.rsplit("+", 1)[1])
        assert value == pytest.approx(float(np.mean(city.true_response(d))), rel=1e-12)
    assert derived["true_footprint_mean"] == pytest.approx(float(np.mean(city.true_footprint())), rel=1e-12)
    assert truth["L"] == city.truth["L"] and truth["v"] == list(city.truth["v"])
    assert a["truth"]["derived"] == derived
