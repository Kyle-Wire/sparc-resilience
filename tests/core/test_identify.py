"""Canopy identification: design layers, the street-differences estimator, the lab's scoring, and the
real-traverse path (projection, passes, CLI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sparc.core import simcheck as S
from sparc.core.identify import estimators as E
from sparc.core.identify import layers as LY
from sparc.core.identify import validate as V


@pytest.fixture(scope="module")
def layout(synthetic_core_data, synthetic_city):
    cfg, data = synthetic_core_data
    return S.Layout(cfg=cfg, df=synthetic_city.frame, data=data, roles=cfg.raw["physics"]["roles"])


@pytest.fixture(scope="module")
def L(layout):
    return LY.build_layers(layout)


# --------------------------------------------------------------------------- #
# Layers                                                                       #
# --------------------------------------------------------------------------- #
def test_ring_mean_matches_brute_force(layout):
    g = layout.grid
    v = np.random.default_rng(0).uniform(0, 100, g.iy.size)
    got = LY.ring_mean(v, g, 100.0, 300.0)
    x, y = g.ix * g.dx, g.iy * g.dy
    for i in np.random.default_rng(1).choice(v.size, 25, replace=False):
        d = np.hypot(x - x[i], y - y[i])
        sel = (d > 100.0 + 1e-6) & (d <= 300.0 + 1e-6)
        assert got[i] == pytest.approx(v[sel].mean(), rel=1e-6)


def test_sector_mean_points_the_right_way(layout):
    g = layout.grid
    north = (g.iy * g.dy).astype(float)                  # a field that grows northward
    up = LY.sector_mean(north, g, (0.0, 1.0), 300.0)     # the sector to the north of each cell
    down = LY.sector_mean(north, g, (0.0, -1.0), 300.0)
    inner = (g.iy > g.iy.min() + 12) & (g.iy < g.iy.max() - 12)          # (an empty sector takes the mean)
    assert np.mean(up[inner] > north[inner]) > 0.95 and np.mean(down[inner] < north[inner]) > 0.95
    assert np.median(up[inner] - north[inner]) == pytest.approx(np.median(north[inner] - down[inner]), rel=0.1)


def test_canopy_terms_are_flexible_near_and_linear_far(L):
    terms = L.canopy_terms()
    assert terms[:6] == ["cb0_0", "cb1_0", "cb2_0", "cb0_r100", "cb1_r100", "cb2_r100"]
    assert terms[6:] == ["cb0_r300", "cb0_r1000"]
    assert L.canopy_terms(100) == terms[:6]


def test_edit_layers_hold_the_uniform_dose(L, layout):
    c = np.clip(layout.col("canopy"), 0, 100)
    assert np.allclose(L.values["edit_cb0_0"], np.minimum(c + 10, 100) - c)
    # basis term (c − 15)+ moves by the part of the +10 pp above the knot
    assert np.allclose(L.values["edit_cb1_0"], np.maximum(np.minimum(c + 10, 100) - 15, 0) - np.maximum(c - 15, 0))
    assert np.allclose(L.values["edit_cb0_r300"], LY.ring_mean(np.minimum(c + 10, 100) - c, layout.grid, 100, 300))


# --------------------------------------------------------------------------- #
# Estimators                                                                   #
# --------------------------------------------------------------------------- #
def test_cluster_ols_recovers_coefficients_and_scales_errors():
    rng = np.random.default_rng(0)
    n, G = 4000, 40
    g = rng.integers(0, G, n)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n) * 50])
    y = X @ np.array([1.0, -2.0, 0.03]) + rng.normal(size=G)[g] + rng.normal(size=n)
    b, V_ = E.cluster_ols(y, X, g)
    assert b == pytest.approx([1.0, -2.0, 0.03], abs=0.4)
    assert np.sqrt(V_[0, 0]) > 0.1                       # the shared cluster shock widens the intercept's error


def test_pairs_stay_within_one_pass():
    df = pd.DataFrame({"seg": [0, 0, 0, 1, 1, 0], "vehicle": [0, 0, 0, 0, 0, 1], "t_s": [0, 3, 6, 9, 12, 0]})
    i, j = E.sfd_pairs(df, lags=(1, 2))
    pairs = sorted(zip(i.tolist(), j.tolist()))
    assert pairs == [(0, 1), (0, 2), (1, 2), (3, 4)]


def _noise_free(kind, layout, seed=0):
    from sparc.core.identify.campaign import simulate

    rng = np.random.default_rng(seed)
    gen = S.Generator(kind, layout, rng)
    T = 88.0 + gen.signal(gen.C0, gen.I0)
    camp = simulate(layout, T, rng, S._product_features(layout), street_m=150.0, product=False,
                    drift_f_per_h=(1.0, 1.0), vehicle_sd=0.3, sensor_sd=0.0)
    return gen, camp


def test_street_design_recovers_a_local_effect_and_ignores_drift(layout, L):
    gen, camp = _noise_free("own_only", layout)
    pts = np.unique(camp.samples["cell"].to_numpy())
    pts = np.random.default_rng(0).choice(pts, min(300, pts.size), replace=False)
    truth = V.reach_profile(gen, layout, pts, reaches=(100,))
    rows = {r["estimator"]: r for r in E.street_effects(camp.samples, L)}
    assert rows["street_100"]["estimate"] == pytest.approx(truth["within_100"], rel=0.15)
    # warming drift and vehicle offsets alone (no canopy effect) give nothing
    gen0, camp0 = _noise_free("null", layout)
    r0 = {r["estimator"]: r for r in E.street_effects(camp0.samples, L)}
    assert abs(r0["street_100"]["estimate"]) < 0.02


def test_run_all_returns_every_design_with_its_estimand(layout, L):
    gen, camp = _noise_free("additive", layout)
    rows = E.run_all(camp.samples, gen.signal(gen.C0, gen.I0) + 88.0, L, layout.grid)
    got = {r["estimator"]: r for r in rows}
    assert set(got) == set(E.ESTIMATORS)
    for name, r in got.items():
        assert r["estimand"] == E.ESTIMAND[name]
        assert np.isfinite(r["estimate"]) and r["se"] >= 0 and r["lo"] <= r["estimate"] <= r["hi"]
    assert got["updown"]["kind"] == "contrast"


# --------------------------------------------------------------------------- #
# The lab's scoring                                                            #
# --------------------------------------------------------------------------- #
def _per(mean, truth, cov=1.0, excl=0.0, n=8):
    return {"n": n, "mean": mean, "truth": truth, "bias": mean - truth, "coverage": cov, "excludes_zero": excl}


def test_verdicts():
    clean = _per(0.01, 0.0)
    ok = V.design_verdict({"null": clean, "additive": _per(-0.25, -0.26), "physics": _per(-0.10, -0.11)}, "effect")
    assert ok["status"] == "trustworthy" and ok["trustworthy"]
    low = V.design_verdict({"null": clean, "additive": _per(-0.15, -0.27, cov=0.5)}, "effect")
    assert low["status"] == "conservative" and low["understates"] == ["additive"]
    over = V.design_verdict({"null": clean, "additive": _per(-0.26, -0.27), "coarse_scale": _per(-0.60, -0.22, cov=0.2)},
                            "effect")
    assert over["status"] == "partial" and over["misses"] == ["coarse_scale"]
    sign = V.design_verdict({"null": clean, "additive": _per(-0.26, -0.27), "physics": _per(-0.06, -0.11, cov=0.6),
                             "coarse_scale": _per(-0.09, -0.047, cov=0.5)}, "effect")
    assert sign["status"] == "direction only"
    biased = V.design_verdict({"null": _per(-0.2, 0.0, excl=0.6), "additive": _per(-0.25, -0.26)}, "effect")
    assert biased["status"] == "not trustworthy" and not biased["clean_null"]
    sig = V.design_verdict({"null": {"excludes_zero": 0.0, "mean": 0.0, "advects": False},
                            "physics": {"excludes_zero": 0.1, "mean": -0.05, "advects": True}}, "contrast")
    assert sig["status"] == "underpowered"


def test_replicate_and_summary(layout, L):
    feats = S._product_features(layout)
    real = {"mean": 88.0, "sd": 1.5, "residual_range_m": 400.0}
    rows = []
    for kind in ("null", "own_only"):
        rows += V.replicate(kind, 0, layout, L, feats, real, which=("street_100", "updown", "map_street_300"))
    assert {r["estimator"] for r in rows} == {"street_100", "updown", "map_street_300"}
    own = [r for r in rows if r["generator"] == "own_only" and r["estimator"] == "street_100"][0]
    assert own["truth"] < 0 and own["truth_city"] < 0 and "city_estimate" in own
    assert all(r["truth"] == 0.0 for r in rows if r["generator"] == "null" and r["kind"] == "effect")
    summ = V.summarize(rows)
    assert set(summ["designs"]) == {"street_100", "updown", "map_street_300"}
    md = V.lab_markdown(summ)
    assert "Canopy identification lab" in md and "Street differences" in md


# --------------------------------------------------------------------------- #
# Real traverses (Providence frame)                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def providence(brown_csv, providence_config_path):
    from sparc.core.config import load_core_config

    cfg = load_core_config(providence_config_path)
    return cfg, S.load_layout(cfg)


def test_campaign_csv_round_trips_through_projection_and_passes(providence, tmp_path):
    from sparc.core.identify.campaign import simulate
    from sparc.core.identify.traverses import campaign_csv, read_traverses, to_campaign

    cfg, lay = providence
    rng = np.random.default_rng(0)
    T = 88.0 + rng.normal(0, 1, lay.grid.iy.size)
    camp = simulate(lay, T, rng, np.zeros((T.size, 1)), product=False)
    path = tmp_path / "traverses.csv"
    path.write_bytes(campaign_csv(camp.samples, cfg, lay))
    pts = read_traverses(path)
    assert {"time", "lat", "lon", "temp_f", "vehicle_key"} <= set(pts.columns)
    assert pts["vehicle_key"].nunique() == camp.samples["vehicle"].nunique()      # the car column
    df, qa = to_campaign(pts, cfg, lay, window=None)
    assert qa["share_on_grid"] == 1.0
    assert np.array_equal(np.sort(df["cell"].to_numpy()), np.sort(camp.samples["cell"].to_numpy()))
    assert abs(qa["n_passes"] - camp.samples["seg"].nunique()) <= 0.05 * camp.samples["seg"].nunique()
    assert qa["median_step_m"] == pytest.approx(lay.grid.dx, rel=0.05)


def test_one_hertz_samples_collapse_to_one_visit_per_cell(providence):
    from sparc.core.identify.traverses import grid_to_lonlat, to_campaign

    cfg, lay = providence
    g = lay.grid
    i0 = int(np.flatnonzero((g.iy == np.median(g.iy).astype(int)))[10])
    cells = [i0] * 3 + [i0 + 1] * 3                      # three samples in each of two neighbouring cells
    lon, lat = grid_to_lonlat(cfg, g.x0 + g.ix[cells] * g.dx, g.y0 + g.iy[cells] * g.dy)
    pts = pd.DataFrame({"time": pd.date_range("2020-07-29 15:00", periods=6, freq="1s"), "lat": lat, "lon": lon,
                        "temp_f": [88, 88.2, 88.4, 89, 89, 89], "window": "midday", "vehicle_key": "a"})
    df, qa = to_campaign(pts, cfg, lay)
    assert df["cell"].tolist() == [i0, i0 + 1] and df["temp"].tolist() == pytest.approx([88.2, 89.0])
    assert qa["n_passes"] == 1


def test_cli_simulate_then_estimate(providence_config_path, brown_csv, tmp_path):
    from sparc.core.identify.__main__ import main

    csv = tmp_path / "sim.csv"
    assert main(["simulate", "-p", str(providence_config_path), "--world", "own_only", "--out", str(csv)]) == 0
    assert main(["estimate", "-p", str(providence_config_path), "--traverses", str(csv), "--window", "all",
                 "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "identify_estimate.json").read_text())
    rows = {r["estimator"]: r for r in res["designs"]}
    assert rows["street_100"]["hi"] < 0                   # the planted own-cell cooling is found
    assert res["headline"]["main"] == "street_100" and res["headline"]["street_100"]["excludes_zero"]
    assert (tmp_path / "identify_estimate.md").read_text().startswith("# Canopy effect from the traverses")


def test_results_page_reads_the_identification_outputs(tmp_path):
    from sparc.core.results_page import _identify_section

    rows = []
    for gen, truth in (("null", 0.0), ("additive", -0.2)):
        for rep in range(3):
            rows.append({"estimator": "street_100", "kind": "effect", "where": "street", "level": "traverse",
                         "generator": gen, "rep": rep, "estimate": truth + 0.01 * (rep - 1), "se": 0.03,
                         "truth": truth, "truth_total": truth, "advects": False})
    run = tmp_path / "city" / "run"
    run.mkdir(parents=True)
    assert _identify_section(run) is None
    out = tmp_path / "city" / "identify"
    out.mkdir()
    (out / "identify_lab.json").write_text(json.dumps(V.summarize(rows)))
    sec = _identify_section(run)
    d = sec["designs"]["street_100"]
    assert d["status"] == "trustworthy" and set(d["worlds"]) == {"null", "additive"}
    assert sec["n_reps"] == {"null": 3, "additive": 3} and sec["estimate"] is None


# --------------------------------------------------------------------------- #
# Real archives, runs, the kilometre design and the floor                      #
# --------------------------------------------------------------------------- #
def _points_csv(cfg, lay, cells, start, temp=88.0, car=1):
    from sparc.core.identify.traverses import grid_to_lonlat

    g = lay.grid
    lon, lat = grid_to_lonlat(cfg, g.x0 + g.ix[cells] * g.dx, g.y0 + g.iy[cells] * g.dy)
    t = pd.Timestamp(start) + pd.to_timedelta(np.arange(len(cells)) * 3, unit="s")
    return pd.DataFrame({"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "lat": lat, "lon": lon,
                         "T_F": temp + 0.01 * np.arange(len(cells)), "car": car})


def test_scan_reads_an_osf_style_folder_and_reports_every_file(providence, tmp_path):
    import zipfile

    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    from sparc.core.identify.traverses import inspect_markdown, scan_traverses

    cfg, lay = providence
    row = np.flatnonzero(lay.grid.iy == int(np.median(lay.grid.iy)))[:40]
    root = tmp_path / "wu9v7-osfstorage-archive" / "Providence"
    (root / "traverses").mkdir(parents=True)
    am = _points_csv(cfg, lay, row, "2020-07-29 06:10:00")
    am.to_csv(tmp_path / "am_trav.csv", index=False)
    with zipfile.ZipFile(root / "traverses" / "am_traverses.zip", "w") as z:
        z.write(tmp_path / "am_trav.csv", "am_trav.csv")
    af = _points_csv(cfg, lay, row, "2020-07-29 15:05:00", temp=91.0, car=2)
    gdf = gpd.GeoDataFrame({"TempF": af["T_F"], "date": af["datetime"].str[:10], "time": af["datetime"].str[11:],
                            "route": [1, 2] * 20, "X": 0.0, "Y": 0.0},
                           geometry=[Point(xy) for xy in zip(af["lon"], af["lat"])], crs="EPSG:4326").to_crs("EPSG:3438")
    gdf["X"], gdf["Y"] = gdf.geometry.x, gdf.geometry.y          # projected feet, which must not be read as degrees
    gdf.to_file(root / "traverses" / "af_trav.shp")
    night = _points_csv(cfg, lay, row, "2020-07-30 00:20:00", temp=84.0)
    night.rename(columns={"T_F": "temp_c"}).assign(temp_c=lambda d: (d["temp_c"] - 32) * 5 / 9).to_csv(
        root / "traverses" / "late.csv", index=False)
    gpd.GeoDataFrame({"n": [1]}, geometry=[Polygon([(-71.5, 41.7), (-71.3, 41.7), (-71.3, 41.9)])],
                     crs="EPSG:4326").to_file(root / "study_area.shp")
    (root / "af_t_f.tif").write_bytes(b"II*\x00")
    pd.DataFrame({"zone": [1], "mean_t": [88.0]}).to_csv(root / "summary.csv", index=False)

    pts, report = scan_traverses(tmp_path / "wu9v7-osfstorage-archive")
    status = {r["file"].split("/")[-1]: r["status"] for r in report}
    assert status == {"am_trav.csv": "read", "af_trav.shp": "read", "late.csv": "read", "study_area.shp": "skipped",
                      "af_t_f.tif": "skipped", "summary.csv": "skipped"}
    assert sorted(pts["window"].unique()) == ["midday", "morning", "night"]
    assert pts.loc[pts["window"] == "night", "temp_f"].mean() == pytest.approx(84.2, abs=0.05)   # °C converted
    assert pts.loc[pts["window"] == "midday", "vehicle_key"].nunique() == 2                       # the route column
    assert pts["lat"].between(41.7, 41.95).all()
    md = inspect_markdown(pts, report, cfg, lay)
    assert "On the project grid: 100%" in md and "2020-07-30 night" in md and "not a point layer" in md


def test_windshift_is_quiet_without_advection_and_finds_it_with(layout, L):
    from sparc.core.identify.windshift import sector_table, wind_shift

    table = sector_table(layout, reach_m=600.0, step_deg=30)
    g = layout.grid
    rng = np.random.default_rng(0)
    winds = {"am": 270, "pm": 180, "ev": 90}
    cells = np.flatnonzero(layout.col("impervious") > 30)
    def runs(advect):
        frames = []
        for k, (name, d) in enumerate(winds.items()):
            a = table["contrasts"][("canopy", d)][0]
            T = 88.0 + 0.3 * rng.standard_normal(g.iy.size) - (0.02 * a if advect else 0.0)
            frames.append(pd.DataFrame({"cell": cells, "run": name, "t_s": rng.uniform(0, 3600, cells.size),
                                        "vehicle": 10 * k + rng.integers(0, 3, cells.size), "temp": T[cells]}))
        return pd.concat(frames, ignore_index=True)
    quiet = wind_shift(runs(False), L, table, winds, g)
    loud = wind_shift(runs(True), L, table, winds, g)
    assert quiet["p_rotation"] > 0.05 and quiet["n_rotations"] == 11
    assert loud["estimate"] < -0.1 and loud["p_cooling"] <= 1 / 12 + 1e-9
    assert loud["spread_deg"] == 180.0


def test_floor_stats_validate_the_bound():
    rows = []
    for gen, total, est in (("null", 0.0, 0.0), ("additive", -0.27, -0.18)):
        for rep in range(8):
            rows.append({"estimator": "street_300", "generator": gen, "rep": rep, "estimate": est + 0.01 * (rep - 4),
                         "se": 0.05, "truth_street_total": total})
    fs = V.floor_stats(rows, "street_300")
    assert fs["valid"] and fs["worlds"]["additive"]["holds"] == 1.0
    bad = [dict(r, estimate=-0.6) for r in rows if r["generator"] == "additive"]
    assert not V.floor_stats(bad, "street_300")["valid"]


def test_kmlab_summary_verdict():
    from sparc.core.identify.kmlab import km_markdown, summarize_km

    rows = [{"generator": g, "rep": r, "estimate": e, "se": 0.02, "p_rotation": p, "p_cooling": pc, "null_sd": 0.02,
             "winds": [[290, 2.0], [170, 7.7]]}
            for g, e, p, pc in (("null", 0.0, 0.5, 0.5), ("physics", -0.08, 0.02, 0.01)) for r in range(8)]
    summ = summarize_km(rows)
    d = summ["designs"]["wind_shift"]
    assert d["verdict"]["status"] == "trustworthy" and d["generators"]["physics"]["finds_cooling"] == 1.0
    assert "Kilometre lab" in km_markdown(summ)


def test_estimate_compares_runs_and_reads_the_kilometre_scale(providence, tmp_path):
    from sparc.core.identify.campaign import simulate
    from sparc.core.identify.traverses import campaign_csv, estimate, estimate_markdown

    cfg, lay = providence
    rng = np.random.default_rng(1)
    T = 88.0 + rng.normal(0, 0.5, lay.grid.iy.size)
    for start in ("2020-07-29 06:00:00", "2020-07-29 15:00:00"):
        camp = simulate(lay, T, rng, np.zeros((T.size, 1)), product=False)
        (tmp_path / f"run_{start[11:13]}.csv").write_bytes(campaign_csv(camp.samples, cfg, lay, start=start))
    res = estimate(cfg, tmp_path, layout=lay, winds={"morning": (290, 2.0), "midday": (170, 7.7)}, fetch_winds=False)
    assert res["window"] == "midday" and len(res["runs"]) == 2
    assert all(r["street"]["street_100"] for r in res["runs"])
    km = res["kilometre"]
    assert km["status"] == "estimated" and km["result"]["n_rotations"] == 35 and km["result"]["spread_deg"] == 120.0
    assert "floor" in res["headline"]
    md = estimate_markdown(res)
    assert "## By time of day" in md and "Kilometre scale" in md and "City-wide" in md
    no_wind = estimate(cfg, tmp_path, layout=lay, fetch_winds=False)
    assert no_wind["kilometre"]["status"] == "needs winds"


def test_utc_timestamps_are_moved_to_local_time(tmp_path):
    from sparc.core.identify.traverses import scan_traverses

    t = pd.date_range("2020-07-29 10:10", periods=5, freq="3s")          # 06:10 EDT, written as UTC
    pd.DataFrame({"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "lat": 41.82, "lon": -71.41, "T_F": 80.0}).to_csv(
        tmp_path / "t.csv", index=False)
    raw, _ = scan_traverses(tmp_path)
    loc, _ = scan_traverses(tmp_path, utc_to="America/New_York")
    assert raw["window"].iloc[0] == "morning" and raw["time"].iloc[0].hour == 10
    assert loc["time"].iloc[0].hour == 6 and loc["run"].iloc[0] == "2020-07-29 morning"
