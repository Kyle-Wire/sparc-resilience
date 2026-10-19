"""LST pilot harness (L4): design, paired verdicts, leave-one-city-out without held-out labels, the physics
transfer rule and the CLI, on small synthetic cities in the dataset's cache format."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sparc.core.lstair import dataset as D
from sparc.core.lstair import evaluate as E
from sparc.core.synthetic import gaussian_random_field


def toy_city(k: int, informative: bool = True, n: int = 48, seed: int = 0) -> D.CityData:
    rng = np.random.default_rng(1000 * seed + k)
    shape = (n, n)
    grid = D.CityGrid(crs="EPSG:32619", x0=300015.0 + 40000 * k, y0=4600015.0, dx=30.0, nx=n, ny=n)
    canopy = 60 * (1 + np.tanh(gaussian_random_field(shape, 4.0, rng))) / 2
    imp = 100 * (1 + np.tanh(gaussian_random_field(shape, 6.0, rng) - 0.02 * (canopy - 30))) / 2
    feats = {"canopy": canopy, "impervious": imp, "ndvi": 0.1 + 0.005 * canopy, "albedo": 0.15 + 0.02 * gaussian_random_field(shape, 3.0, rng),
             "elevation": 20 + 5 * gaussian_random_field(shape, 20.0, rng), "water_distance": 300 + 200 * np.abs(gaussian_random_field(shape, 10.0, rng))}
    hidden = gaussian_random_field(shape, 3.0, rng)
    lst = 35 + 0.1 * imp - 0.08 * canopy + (3.0 * hidden if informative else 0.0) + 0.3 * rng.standard_normal(shape)
    R = D.derived_rasters(feats, lst, grid.dx)
    air = -0.03 * R["canopy_f100"] + 0.02 * imp + (0.8 * hidden if informative else 0.0) + 0.1 * rng.standard_normal(shape)
    streets = np.zeros(shape, bool)
    streets[2::4, 3:-3] = True
    streets[3:-3, 1::4] = True
    cell = np.flatnonzero(streets.ravel())
    iy, ix = np.divmod(cell, n)
    Ta = 28 + air[iy, ix]
    t = pd.DataFrame({"city": f"Toy {k}", "slug": f"toy_{k}", "campaign_date": "2021-07-20", "run": "af",
                      "run_id": "2021-07-20 midday", "run_date": "2021-07-20", "cell": cell})
    x, y = grid.cell_xy(cell)
    t["x"], t["y"] = x, y
    t["lat"], t["lon"] = 41.5, -71.0
    t["n_readings"], t["n_visits"] = 3, 1
    t["Ta"] = Ta
    t["ta_background"] = float(np.median(Ta))
    t["dTa"] = Ta - t["ta_background"]
    t["ta_era5"] = 27.0
    for name, r in R.items():
        t[name] = r[iy, ix]
    t["lst_scene_date"], t["lst_offset_days"], t["lst_clear_fraction"] = "2021-07-18", -2 - k, 0.9
    meta = {"slug": f"toy_{k}", "status": "ok", "built_utc": "x", "grid": grid.as_dict(),
            "runs": [{"run": "af", "era5": {"sw_down": 600.0 + 20 * k, "lw_net": -90.0, "ta_era5": 27.0}}]}
    driven = np.zeros(shape, bool)
    driven[iy, ix] = True
    return D.CityData(slug=f"toy_{k}", meta=meta, table=t, grid=grid, rasters={**R, "driven": driven})


def write_city(c: D.CityData, root):
    d = root / "cities" / c.slug
    d.mkdir(parents=True)
    c.table.to_parquet(d / "table.parquet", index=False)
    np.savez_compressed(d / "rasters.npz", **c.rasters)
    (d / "meta.json").write_text(json.dumps(c.meta))


FAST = E.Settings(boot=500, physics_iter=25, gbm=False, ridge_alphas=(3.0, 300.0), within_city=True)


def test_design_centres_within_runs_and_keeps_targets_out():
    t = pd.concat([toy_city(0).table, toy_city(1).table])
    d = E.design(t)
    for _, g in d.groupby(["slug", "run_id"]):
        assert abs(g["canopy_c"].median()) < 1e-9 and abs(g["lst_c"].median()) < 1e-9
    assert np.allclose(d["lst_f300_c"], d["lst_f300"] - d.groupby("slug")["lst"].transform("median"))
    forbidden = {"Ta", "dTa", "ta_background", "ta_era5", "n_readings", "n_visits", "x", "y", "lat", "lon"}
    assert not forbidden & set(E.B_COLS) and set(E.A_COLS) < set(E.B_COLS)
    assert set(E.B_COLS) - set(E.A_COLS) == {"lst_c", "lst_f100_c", "lst_f300_c"}


def test_paired_verdicts():
    a = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 0.95, 1.15, 1.05, 1.25])
    yes = E.paired(a - 0.2 + 0.01 * np.arange(10), a)
    assert yes["verdict"] == "LST adds information beyond land cover" and yes["ci95"][1] < 0
    assert yes["n_better"] == 10 and yes["p_sign_flip"] == pytest.approx(1 / 1024)
    assert yes["sign_flip"].startswith("exact") and yes["ci_method"] == "t-interval, 9 df"
    mixed = E.paired(a + np.array([-0.2, 0.2, -0.1, 0.15, -0.05, 0.1, -0.1, 0.05, 0.1, -0.15]), a)
    assert mixed["verdict"] == "not shown" and mixed["ci95"][0] < 0 < mixed["ci95"][1]
    worse = E.paired(a + 0.3, a)
    assert worse["verdict"] == "not shown" and "worse" in worse["reason"]
    tiny = E.paired(a - 0.02 + 0.001 * np.arange(10), a)                   # significant but a 2% change
    assert tiny["verdict"] == "not shown" and "only" in tiny["reason"] and tiny["p_sign_flip"] <= E.ALPHA
    few = E.paired(a[:7] - 0.2, a[:7])
    assert few["verdict"] == "not testable" and E.MIN_VERDICT_CITIES == 8
    assert E.paired(a - 0.2, a, claim="x")["verdict"] == "x"
    # five cities: even when every one improves, the smallest sign-flip p is 1/32 > 0.025
    assert E.sign_flip(-np.ones(5))["p_less"] == pytest.approx(1 / 32)


def test_the_paired_test_holds_its_level_under_a_true_null_at_the_city_counts_of_the_pilot():
    """R1: the percentile bootstrap claimed 'LST adds information' for 7.7% of null draws at 5–6 cities and 6.2% at 8
    (nominal 2.5%).  The sign-flip p and the verdict must hold the level at 8 and 10 cities."""
    rng = np.random.default_rng(1)
    N = 1500
    for n in (8, 10):
        for draw in (lambda: rng.standard_normal(n), lambda: rng.standard_t(3, n)):
            p_hits = claims = 0
            for i in range(N):
                d = 0.05 * draw()
                p_hits += E.sign_flip(d)["p_less"] <= E.ALPHA
                claims += E.paired(1.0 + d, np.ones(n), seed=i)["verdict"] == "LST adds information beyond land cover"
            tol = 3 * np.sqrt(E.ALPHA * (1 - E.ALPHA) / N)
            assert p_hits / N <= E.ALPHA + tol and claims / N <= E.ALPHA + tol, (n, p_hits / N, claims / N)


def test_city_metrics_and_centring():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = E.city_metrics(y, y + 0.5)
    assert m["rmse"] == pytest.approx(0.5) and m["r"] == pytest.approx(1.0) and m["r2"] == pytest.approx(1 - 1.0 / 5.0)
    p = E.centre(np.array([1.0, 2.0, 3.0, 10.0, 20.0]), np.array(["a", "a", "a", "b", "b"]))
    assert np.allclose(p, [-1, 0, 1, -5, 5])


@pytest.fixture(scope="module")
def informative():
    cities = [toy_city(k) for k in range(8)]
    res, pred = E.evaluate_window(cities, "af", FAST)
    return cities, res, pred


def test_lst_that_carries_a_hidden_source_beats_land_cover_out_of_city(informative):
    _, res, pred = informative
    assert res["status"] == "done" and len(res["per_city"]) == 8
    assert res["verdict"] == "LST adds information beyond land cover"
    assert res["paired"]["C-A"]["n_cities"] == 8 and res["paired"]["C-A"]["mean_diff_mse"] < 0
    assert {"pred_A", "pred_B", "pred_C", "pred_C0"} <= set(pred.columns)
    for r in res["per_city"]:
        assert r["B"]["r"] > r["A"]["r"]
    wc = res["within_city"]
    assert "interpolation-prone" in wc["label"] and len(wc["cities"]) == 8
    fits = {f["slug"]: f for f in res["physics_fits"]["C"]}
    assert all(f["c_lst"] > 0 and f["h_lst"] == pytest.approx(f["c_lst"] * f["norm"]) for f in fits.values())
    assert all(f["forcing"][2] == "ERA5" for f in fits.values())


def test_held_out_labels_never_reach_their_own_predictions(informative):
    cities, res, pred = informative
    h = cities[2]
    rng = np.random.default_rng(5)
    shuffled = h.table.copy()
    perm = rng.permutation(len(shuffled))
    shuffled["dTa"] = shuffled["dTa"].to_numpy()[perm]
    shuffled["Ta"] = shuffled["Ta"].to_numpy()[perm]                       # the labels the target is centred from
    alt = D.CityData(slug=h.slug, meta=h.meta, table=shuffled, grid=h.grid, rasters=h.rasters)
    s = E.Settings(**{**FAST.__dict__, "within_city": False})
    _, pred2 = E.evaluate_window(cities[:2] + [alt] + cities[3:], "af", s)
    a = pred[pred["slug"] == h.slug].sort_values("cell")
    b = pred2[pred2["slug"] == h.slug].sort_values("cell")
    for m in ("pred_A", "pred_B", "pred_C", "pred_C0"):
        assert np.allclose(a[m].to_numpy(), b[m].to_numpy(), atol=1e-10), m


def test_transfer_rule_uses_medians_in_physical_units():
    fits = [{"slug": f"c{i}", "natural": {k: float(i + 1) for k in E.TRANSFER_KEYS}, "a_phys": 0.01 * (i + 1),
             "gamma": -0.001 * i, "c_lst": 0.02, "norm": 500.0 + 100 * i, "h_lst": (0.02) * (500.0 + 100 * i)}
            for i in range(3)]
    tp = E.transfer_params(fits, True)
    assert tp["natural"]["L"] == 2.0 and tp["a_phys"] == pytest.approx(0.02) and tp["gamma"] == pytest.approx(-0.001)
    assert tp["h_lst"] == pytest.approx(0.02 * 600.0) and tp["cities"] == ["c0", "c1", "c2"]
    assert "h_lst" not in E.transfer_params(fits, False)
    assert "medians" in E.TRANSFER_RULE and "No held-out" in E.TRANSFER_RULE


def test_physics_fits_are_cached_per_city(tmp_path):
    cities = [toy_city(k, n=32) for k in range(3)]
    s = E.Settings(physics_iter=10)
    f1 = E.physics_fits(cities, "af", True, s, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"toy_{k}__af__lst.json" for k in range(3)]
    calls = []
    orig = E.fit_city_physics
    try:
        E.fit_city_physics = lambda *a, **k: calls.append(a) or orig(*a, **k)
        f2 = E.physics_fits(cities, "af", True, s, tmp_path)
    finally:
        E.fit_city_physics = orig
    assert calls == [] and f1["toy_0"]["natural"] == f2["toy_0"]["natural"]


def test_too_few_cities_is_reported_not_run():
    res, pred = E.evaluate_window([toy_city(0), toy_city(1)], "af", FAST)
    assert res["status"] == "not run" and "need 3" in res["reason"] and pred.empty


def test_cli_evaluate_writes_results(tmp_path):
    from sparc.core.cli import main

    for k in range(4):
        write_city(toy_city(k, n=36), tmp_path / "data")
    rc = main(["lstair", "evaluate", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "res"), "--no-gbm",
               "--physics-iter", "10", "--boot", "200", "--no-within-city"])
    assert rc == 0
    res = json.loads((tmp_path / "res" / "results.json").read_text())
    w = res["windows"]["af"]
    assert w["status"] == "done" and len(w["per_city"]) == 4 and w["paired"]["B-A"]["verdict"] == "not testable"
    md = (tmp_path / "res" / "results.md").read_text()
    assert md.startswith("# LST → air temperature") and "Transfer rule" in md and "| Toy 0 |" in md
    assert (tmp_path / "res" / "predictions.parquet").exists() and (tmp_path / "res" / "physics").is_dir()


def _renamed(c: D.CityData, slug: str, dx_m: float = 0.0, meta: dict | None = None) -> D.CityData:
    g = D.CityGrid(**{**c.grid.as_dict(), "x0": c.grid.x0 + dx_m})
    t = c.table.assign(slug=slug, city=slug, x=c.table["x"] + dx_m)
    return D.CityData(slug=slug, meta={**c.meta, "slug": slug, **(meta or {})}, table=t, grid=g, rasters=c.rasters)


def test_campaign_groups_link_one_campaign_and_neighbouring_grids():
    """R2: Bronx and Manhattan are folders of one OSF node and campaign; New York sits beside its boroughs."""
    a = toy_city(0, n=24)
    far = toy_city(3, n=24)                                                 # 120 km east
    same = _renamed(toy_city(5, n=24), "bronx", meta={"catalog": {"node": "j6eqr"}, "campaign_date": "2021-07-20"})
    man = _renamed(toy_city(6, n=24), "manhattan", meta={"catalog": {"node": "j6eqr"}, "campaign_date": "2021-07-20"})
    near = _renamed(a, "next_door", dx_m=1500.0)                            # overlapping grid
    g = E.campaign_groups([a, far, same, man, near])
    assert g["toy_0"] == g["next_door"] == "next_door+toy_0"
    assert g["bronx"] == g["manhattan"] == "bronx+manhattan" and g["toy_3"] == "toy_3"
    other_day = _renamed(toy_city(6, n=24), "manhattan", meta={"catalog": {"node": "j6eqr"}, "campaign_date": "2022-07-20"})
    assert E.campaign_groups([same, other_day])["bronx"] == "bronx"


def test_a_campaign_group_is_held_out_whole_and_counted_once():
    cities = [toy_city(k, n=32) for k in range(4)]
    other = toy_city(4, n=32)                                               # another city's data, moved next to toy_0
    shift = cities[0].grid.x0 + 600.0 - other.grid.x0
    cities.append(_renamed(other, "toy_0_twin", dx_m=shift))
    s = E.Settings(**{**FAST.__dict__, "within_city": False, "physics": False})
    res, pred = E.evaluate_window(cities, "af", s)
    assert res["groups"]["toy_0+toy_0_twin"] == ["toy_0", "toy_0_twin"]
    assert res["paired"]["B-A"]["n_cities"] == 4                            # 4 independent units, not 5 cities
    # the twin's labels never reach toy_0's predictions (held out together, never in each other's training)
    tw = cities[-1]
    perm = np.random.default_rng(0).permutation(len(tw.table))
    alt = D.CityData(slug=tw.slug, meta=tw.meta, grid=tw.grid, rasters=tw.rasters,
                     table=tw.table.assign(dTa=tw.table["dTa"].to_numpy()[perm], Ta=tw.table["Ta"].to_numpy()[perm]))
    _, pred2 = E.evaluate_window(cities[:-1] + [alt], "af", s)
    a = pred[pred["slug"] == "toy_0"].sort_values("cell")
    b = pred2[pred2["slug"] == "toy_0"].sort_values("cell")
    assert np.allclose(a["pred_A"], b["pred_A"]) and np.allclose(a["pred_B"], b["pred_B"])


def test_the_target_is_recentred_on_the_cells_scored():
    """R6: an LST cloud over the cooler cells left the scored target with a non-zero median (an offset in every
    model's error); the target is now centred on the scored cells, as the predictions are."""
    cities = [toy_city(k, n=32) for k in range(4)]
    for c in cities:                                                        # cloud over the coolest third
        t = c.table
        cool = t["dTa"] < t["dTa"].quantile(0.33)
        for k in ("lst", "lst_f100", "lst_f300"):
            t.loc[cool, k] = np.nan
    s = E.Settings(**{**FAST.__dict__, "within_city": False, "physics": False})
    res, pred = E.evaluate_window(cities, "af", s)
    for _, g in pred.groupby(["slug", "run_id"]):
        assert abs(g["dTa"].median()) < 1e-9
    rows = res["target_recentred"]
    assert len(rows) == 4 and all(r["share_dropped"] > 0.3 and r["shift_c"] > 0 for r in rows)
    shifted, _ = E.recentre_target(E.design(cities[0].table).dropna(subset=E.B_COLS))
    assert np.allclose(shifted["dTa"] + shifted["Ta"].median(), shifted["Ta"])


def _no_lst(c: D.CityData) -> D.CityData:
    t = c.table.copy()
    for k in ("lst", "lst_f100", "lst_f300"):
        t[k] = np.nan
    return D.CityData(slug=c.slug, meta=c.meta, table=t, grid=c.grid,
                      rasters={k: v for k, v in c.rasters.items() if not k.startswith("lst")})


def test_without_lst_the_land_cover_and_physics_baselines_still_run():
    """R22: a dataset built with --no-lst gave 'not run' for every model; a city without a scene vanished unnamed."""
    cities = [_no_lst(toy_city(k, n=32)) for k in range(4)]
    s = E.Settings(**{**FAST.__dict__, "within_city": True, "physics_iter": 10})
    res, pred = E.evaluate_window(cities, "af", s)
    assert res["status"] == "done" and res["lst"].startswith("none")
    assert set(res["summary"]) == {"A", "C0"} and "pred_A" in pred and "pred_C0" in pred and "pred_B" not in pred
    assert res["paired"]["B-A"]["verdict"] == "not testable" and "no LST" in res["paired"]["B-A"]["reason"]
    assert "LST: none" in E.results_markdown({"target": "t", "n_cities": 4, "transfer_rule": "r",
                                              "windows": {"af": res}})
    # one city without a scene among LST cities: named as excluded, the rest evaluated
    mixed = [toy_city(k, n=32) for k in range(4)] + [_no_lst(toy_city(4, n=32))]
    res2, _ = E.evaluate_window(mixed, "af", E.Settings(**{**s.__dict__, "physics": False}))
    assert [e["slug"] for e in res2["excluded"]] == ["toy_4"] and "no clear Landsat scene" in res2["excluded"][0]["reason"]
    assert {r["slug"] for r in res2["per_city"]} == {"toy_0", "toy_1", "toy_2", "toy_3"}


def test_physics_uses_one_run_with_its_own_forcing_and_predicts_every_run(tmp_path):
    """R18: a city with two afternoon runs was fitted on a blend of the days with the first run's forcing."""
    c = toy_city(0, n=32)
    t1 = c.table
    t2 = t1.iloc[: len(t1) // 2].assign(run_id="2021-07-21 midday", run_date="2021-07-21", Ta=t1["Ta"].iloc[: len(t1) // 2] + 3,
                                        ta_background=float(t1["ta_background"].iloc[0]) + 3)
    meta = {**c.meta, "runs": [{"run": "af", "run_id": "2021-07-20 midday", "era5": {"sw_down": 700.0, "lw_net": -90.0}},
                               {"run": "af", "run_id": "2021-07-21 midday", "era5": {"sw_down": 300.0, "lw_net": -70.0}}]}
    city = D.CityData(slug=c.slug, meta=meta, table=pd.concat([t1, t2], ignore_index=True), grid=c.grid, rasters=c.rasters)
    assert E.physics_run_id(city, "af") == "2021-07-20 midday"                 # the run with the most cells
    a = E.physics_inputs(city, "af", False, 60.0)
    b = E.physics_inputs(city, "af", False, 60.0, run_id="2021-07-21 midday")
    assert (a["sw"], a["run_id"]) == (700.0, "2021-07-20 midday") and b["sw"] == 300.0
    assert b["background"] == pytest.approx(a["background"] + 3) and len(b["table"]) == len(t2)
    s = E.Settings(physics_iter=10)
    rec = E.fit_city_physics(city, "af", False, s)
    assert rec["run_id"] == "2021-07-20 midday" and rec["forcing"][0] == 700.0
    tp = E.transfer_params([rec, rec], False)
    pc = E.predict_physics(city, "af", False, tp, s)
    assert set(pc.index.get_level_values(0)) == {"2021-07-20 midday", "2021-07-21 midday"}
    for _, part in pc.groupby(level=0):
        assert abs(part.median()) < 1e-9


def test_the_lst_gain_is_transferred_as_the_median_of_the_per_city_products():
    """R4: a and c_lst trade off within a fit (anti-correlated across cities); the product of the separate medians
    misstated the transferred LST response.  The median of a_phys·h_lst is what reaches the held-out city."""
    fits = [{"slug": f"c{i}", "natural": {k: 1.0 for k in E.TRANSFER_KEYS}, "a_phys": a, "gamma": 0.0, "h_lst": h,
             "g_lst": a * h} for i, (a, h) in enumerate([(1.0, 3.0), (2.0, 1.0), (3.0, 2.0)])]
    tp = E.transfer_params(fits, True)
    assert tp["g_lst"] == pytest.approx(3.0) and tp["a_phys"] == pytest.approx(2.0)
    assert tp["a_phys"] * tp["h_lst"] == pytest.approx(3.0)                       # not median(a)·median(h) = 4
    zero = E.transfer_params([{**f, "a_phys": 0.0} for f in fits], True)
    assert zero["h_lst"] == 0.0 and "cannot" in zero["note"]


def test_the_physics_regime_is_fixed_per_window_and_transfer_never_mixes_regimes():
    """R5: am/pm runs straddle SW ≈ 50 W m⁻²; per-city regimes mixed the day and night source formulas in the
    transfer and handed a night city starting values of parameters no night fit uses."""
    base = {"grid": None, "roles": [], "lw": -80.0, "background": 0.0}
    assert E.physics_cfg({**base, "run": "am", "sw": 30.0}, FAST)["window"] == "day"
    assert E.physics_cfg({**base, "run": "pm", "sw": 80.0}, FAST)["window"] == "night"
    assert E.physics_cfg({**base, "run": "af", "sw": 650.0}, FAST)["window"] == "day"
    nat = {k: 1.0 for k in E.TRANSFER_KEYS}
    fits = [{"slug": "n1", "window": "night", "natural": {**nat, "a1n": 0.4}, "a_phys": 1.0, "gamma": 0.0},
            {"slug": "n2", "window": "night", "natural": {**nat, "a1n": 0.6}, "a_phys": 2.0, "gamma": 0.0},
            {"slug": "d1", "window": "day", "natural": {**nat, "a1n": 0.5}, "a_phys": 99.0, "gamma": 0.0}]
    tp = E.transfer_params(fits, False)
    assert tp["regime"] == "night" and tp["cities"] == ["n1", "n2"] and tp["a_phys"] == pytest.approx(1.5)
    assert tp["natural"]["a1n"] == pytest.approx(0.5) and {"s", "kc", "a1"} <= set(tp["unused_keys"])
