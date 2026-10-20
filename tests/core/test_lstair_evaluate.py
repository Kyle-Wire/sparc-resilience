"""LST pilot harness (L4): design, the registered skill comparisons and claim rule (Holm, influence), leave-one-
campaign-out without held-out labels, the physics transfer rules, the sensitivities and the CLI, on small synthetic
cities in the dataset's cache format."""

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


def test_paired_verdicts_of_the_secondary_absolute_mse():
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
    assert res["status"] == "done" and len(res["per_city"]) == 8 and len(res["per_unit"]) == 8
    assert res["verdict"] == "LST adds information beyond land cover"
    ba, ca = res["paired"]["B-A"], res["paired"]["C-A"]
    assert ca["n_units"] == 8 and ca["mean_dss"] > E.MIN_DSS and ca["verdict"] == "physics with LST beats land cover"
    # one window alone: Holm over the registered family with am and pm untested (p = 1), m = 6
    assert ba["p_holm"] == pytest.approx(6 * ba["p_sign_flip"]) and ba["holm"]["m"] == 6
    assert ba["ci95"][0] > 0 and ba["mean_ss_x"] > 0 and res["absolute"]["B-A"]["mean_diff_mse"] < 0
    assert res["paired"]["C-C0"]["p_holm"] is None                          # outside the family
    assert {"pred_A", "pred_B", "pred_C", "pred_C0"} <= set(pred.columns)
    for r in res["per_city"]:
        assert r["B"]["r"] > r["A"]["r"] and r["B"]["ss"] > r["A"]["ss"]
    for u in res["per_unit"]:
        assert u["N"]["ss"] == 0.0 and u["B"]["ss"] > u["A"]["ss"]
    assert res["units_r_A_negative"]["n"] == 0 and res["units_r_A_negative"]["of"] == 8
    lb = res["physics_L_bounds"]["C"]
    assert lb["n_fits"] == 8 and lb["n_lower"] + lb["n_upper"] <= 8
    # 8 units: without any one of them the window is not testable, and the influence says so
    inf = ba["influence"]
    assert inf["n"] == 8 and set(inf["flipped_to"].values()) == {"not testable"}
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
    assert res["primary"]["window"] == "af" and res["primary"]["B-A"]["verdict"] == "not testable"
    md = (tmp_path / "res" / "results.md").read_text()
    assert md.startswith("# LST → air temperature") and "Transfer rule" in md and "| toy_0 | 1 | 1 |" in md
    # the primary verdict first, then the window with its per-unit table, influence, and the secondary ΔMSE last
    order = [md.index(h) for h in ("## Primary verdict (af, as registered)", "## Run af (primary)", "| SS_N |",
                                   "Leave-one-unit-out influence", "## Secondary: absolute ΔMSE")]
    assert order == sorted(order)
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


def test_one_node_and_date_links_only_grids_within_50_km():
    """Review 2026-10-20: VFIC's node 3xvmg holds eight city campaigns 19–505 km apart on 15 July 2021, and
    Boise and Sacramento (642 km) share NIHHIS's tdsy7 on one date; the node + date link merged them into two
    units.  It now needs the grids within 50 km; ``same_node_km=None`` restores it (Virginia as one group)."""
    vfic = {"catalog": {"node": "3xvmg"}, "campaign_date": "2021-07-15"}
    lyn = _renamed(toy_city(0, n=24), "lynchburg", meta=vfic)
    far = _renamed(toy_city(3, n=24), "richmond", meta=vfic)                # 120 km east of lynchburg
    close = _renamed(toy_city(1, n=24), "petersburg", meta=vfic)            # 40 km east: a 39 km gap
    g = E.campaign_groups([lyn, far, close])
    assert g["lynchburg"] == g["petersburg"] == "lynchburg+petersburg" and g["richmond"] == "richmond"
    one = E.campaign_groups([lyn, far, close], same_node_km=None)
    assert len(set(one.values())) == 1
    assert E.SAME_NODE_KM == 50.0 and "conservative, not anti-conservative" in E.campaign_groups.__doc__


def test_a_campaign_group_is_held_out_whole_and_counted_once():
    cities = [toy_city(k, n=32) for k in range(4)]
    other = toy_city(4, n=32)                                               # another city's data, moved next to toy_0
    shift = cities[0].grid.x0 + 600.0 - other.grid.x0
    cities.append(_renamed(other, "toy_0_twin", dx_m=shift))
    s = E.Settings(**{**FAST.__dict__, "within_city": False, "physics": False})
    res, pred = E.evaluate_window(cities, "af", s)
    assert res["groups"]["toy_0+toy_0_twin"] == ["toy_0", "toy_0_twin"]
    assert res["paired"]["B-A"]["n_units"] == 4                             # 4 independent units, not 5 cities
    twin = next(u for u in res["per_unit"] if u["unit"] == "toy_0+toy_0_twin")
    runs = [r for r in res["per_run"] if r["group"] == twin["unit"]]
    assert twin["n_runs"] == 2 and twin["A"]["ss"] == pytest.approx(np.mean([r["A"]["ss"] for r in runs]))
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
    assert set(res["summary"]) == {"N", "A", "C0"} and "pred_A" in pred and "pred_C0" in pred and "pred_B" not in pred
    assert res["paired"]["B-A"]["verdict"] == "not testable" and "no LST" in res["paired"]["B-A"]["reason"]
    assert "LST: none" in E.results_markdown({"target": "t", "n_cities": 4, "transfer_rule": "r", "settings": {},
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


def _strict(text: str):
    def bad(c):
        raise ValueError(f"non-standard JSON token {c}")
    return json.loads(text, parse_constant=bad)


def test_every_json_file_is_strict_json(tmp_path, monkeypatch):
    """R24: city_metrics returns NaN r / R² by design, and json.dumps wrote bare NaN (invalid JSON for browsers,
    jq --strict and Studio); the files are now written as the core's runio does (NaN → null)."""
    from sparc.core.lstair import to_json

    assert _strict(to_json({"r": float("nan"), "x": np.float64("inf"), "a": np.array([1.0, np.nan])})) == \
        {"r": None, "x": None, "a": [1.0, None]}
    orig = E.city_metrics
    monkeypatch.setattr(E, "city_metrics", lambda y, p: {**orig(y, p), "r": float("nan")})
    for k in range(4):
        write_city(toy_city(k, n=32), tmp_path / "data")
    E.run_evaluation(tmp_path / "data", tmp_path / "res", E.Settings(**{**FAST.__dict__, "physics_iter": 5,
                                                                         "within_city": False}))
    res = _strict((tmp_path / "res" / "results.json").read_text())
    assert res["windows"]["af"]["per_city"][0]["A"]["r"] is None
    for f in (tmp_path / "res" / "physics").glob("*.json"):
        _strict(f.read_text())


# --------------------------------------------------------------------------- #
# The registered analysis (docs/results/lst_air_pilot.md)                      #
# --------------------------------------------------------------------------- #
def test_skill_is_scored_against_model_n_per_run_and_averaged_over_a_units_runs():
    """SS = 1 − MSE/MSE_N with N the zero prediction (so SS_N = 0); a unit's skill is the mean over its runs, so
    a run with a large spread (Los Angeles) weighs no more than a quiet one."""
    y = np.array([-1.0, 0.0, 1.0, 2.0])
    assert E.skill(y, np.zeros(4)) == 0.0 and E.skill(y, y) == 1.0
    assert E.skill(y, y + 1.0) == pytest.approx(1 - 1.0 / np.mean(y ** 2))
    df = pd.DataFrame({"slug": ["a"] * 4 + ["a"] * 4 + ["b"] * 4, "run_id": ["r1"] * 4 + ["r2"] * 4 + ["r1"] * 4,
                       "group": ["a"] * 8 + ["b"] * 4, "dTa": np.r_[10 * y, y, y]})
    pa = np.r_[10 * y + 1.0, y + 1.0, y]                                   # the same 1 °C error in both runs of a
    runs = E.score_runs(df, {"N": np.zeros(12), "A": pa})
    assert [r["n"] for r in runs] == [4, 4, 4] and all(r["N"]["ss"] == 0.0 for r in runs)
    ss = [r["A"]["ss"] for r in runs]
    assert ss[0] == pytest.approx(1 - 1 / np.mean((10 * y) ** 2)) and ss[1] == pytest.approx(1 - 1 / np.mean(y ** 2))
    units = E.score_units(runs, {"a": ["a"], "b": ["b"]})
    assert units[0]["A"]["ss"] == pytest.approx(np.mean(ss[:2])) and units[1]["A"]["ss"] == pytest.approx(1.0)
    assert units[0]["N"]["ss"] == 0.0 and units[0]["n_runs"] == 2


def _skill_rec(d, base=0.3, seed=0, key="B-A"):
    d = np.asarray(d, float)
    return E.paired_skill(base + d, np.full(d.size, base), [f"u{i}" for i in range(d.size)], seed=seed, key=key)


def test_the_claim_rule_needs_holm_p_an_interval_above_zero_a_005_gain_and_beating_n():
    rng = np.random.default_rng(0)
    yes = _skill_rec(0.1 + 0.01 * rng.standard_normal(10))
    assert yes["verdict"] == "LST adds information beyond land cover" and yes["n_better"] == 10
    assert yes["p_sign_flip"] == pytest.approx(1 / 1024) and yes["ci95"][0] > 0 and yes["sign_flip"].startswith("exact")
    small = _skill_rec(0.03 + 0.003 * rng.standard_normal(10))             # significant, but ΔSS < 0.05
    assert small["verdict"] == "not shown" and "only" in small["reason"]
    below_n = E.paired_skill(np.full(10, -0.2) + 0.1 + 0.01 * rng.standard_normal(10), np.full(10, -0.2))
    assert below_n["verdict"] == "not shown" and "does not beat N" in below_n["reason"]
    worse = _skill_rec(-0.1 + 0.01 * rng.standard_normal(10))
    assert worse["verdict"] == "not shown" and "worse" in worse["reason"]
    mixed = _skill_rec(np.array([0.1, -0.1] * 5) + 0.001 * np.arange(10))
    assert mixed["verdict"] == "not shown" and mixed["ci95"][0] < 0
    assert _skill_rec(np.full(7, 0.1))["verdict"] == "not testable"
    # the Holm-adjusted p decides: 10 units all better give p = 1/1024; × 6 still ≤ 0.025, but not × 30
    v = E.skill_verdict(yes, 30 / 1024, adjusted=True)
    assert v["verdict"] == "not shown" and "Holm-adjusted" in v["reason"]
    c = E.paired_skill(np.full(10, 0.5), np.full(10, 0.4) + 0.001 * np.arange(10), key="C-A")
    assert c["claim"] == "physics with LST beats land cover" and c["x"] == "C" and c["verdict"] == c["claim"]


def test_holm_over_the_registered_family_of_three_windows_and_two_claims():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adj = E.holm(p)
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    rng = np.random.default_rng(3)

    def window(d):
        return {"status": "done", "paired": {k: _skill_rec(d, key=k) for k in ("B-A", "C-A", "C-C0")}}
    strong = 0.1 + 0.01 * rng.standard_normal(10)                          # p = 1/1024 each
    windows = {"af": window(strong), "am": window(strong), "pm": window(strong)}
    E.finalise(windows)
    for w in ("af", "am", "pm"):
        assert windows[w]["paired"]["B-A"]["p_holm"] == pytest.approx(6 / 1024)
        assert windows[w]["paired"]["B-A"]["holm"]["m"] == 6 and windows[w]["verdict"] == E.CLAIMS["B-A"]
        assert windows[w]["paired"]["C-A"]["verdict"] == E.CLAIMS["C-A"]
        assert windows[w]["paired"]["C-C0"]["p_holm"] is None
    # a window with 7 units is not testable and counts as p = 1; a missing window too
    windows = {"af": window(strong), "am": window(strong[:7])}
    E.finalise(windows)
    h = windows["af"]["paired"]["B-A"]["holm"]
    assert set(h["untested_as_p1"]) == {"am B-A", "am C-A", "pm B-A", "pm C-A"}
    assert windows["am"]["paired"]["B-A"]["verdict"] == "not testable"


def test_leave_one_unit_out_names_the_unit_that_carries_the_claim():
    """One unit (0.15) lifts mean ΔSS over 0.05; without it the mean is 0.04: the claim rests on it alone."""
    d = np.r_[0.15, np.full(9, 0.04) + 0.001 * np.arange(9)]
    res = {"status": "done", "paired": {"B-A": _skill_rec(d)}}
    windows = {"af": res}
    E.finalise(windows)
    rec = res["paired"]["B-A"]
    assert rec["verdict"] == E.CLAIMS["B-A"]
    inf = rec["influence"]
    assert inf["n"] == 10 and inf["flips"] == ["u0"] and inf["any_flip"] and inf["flipped_to"] == {"u0": "not shown"}
    assert inf["mean_dss_range"][0] == pytest.approx(np.mean(d[1:])) and inf["verdicts"][E.CLAIMS["B-A"]] == 9
    assert all(r["p"] == pytest.approx(6 * r["p_sign_flip"]) for r in inf["rows"])


def test_variance_weights_make_each_city_weigh_equally_relative_to_its_own_variance():
    rng = np.random.default_rng(0)
    slugs = np.array(["la"] * 300 + ["sea"] * 100)
    y = np.r_[3.5 * rng.standard_normal(300), 0.4 * rng.standard_normal(100)]
    w = E._weights(slugs, y, "variance")
    assert w.mean() == pytest.approx(1.0)
    tot = {c: w[slugs == c].sum() * np.var(y[slugs == c]) for c in ("la", "sea")}
    assert tot["la"] == pytest.approx(tot["sea"])                          # Σw · var equal per city
    wn = E._weights(slugs, y, "n")
    assert wn[slugs == "la"].sum() == pytest.approx(wn[slugs == "sea"].sum()) and wn.mean() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        E.Settings(weights="bogus")


def test_splines_are_clamped_outside_the_training_range():
    """Los Angeles's elevation anomaly (SD 140 m against 6–46 m elsewhere) ran along a linear spline tail."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (400, 1))
    y = x[:, 0] + 0.05 * rng.standard_normal(400)
    p = E._fit_predict("ridge_spline", {"alpha": 1.0}, x, y, np.ones(400), np.array([[1.0], [5.0], [50.0]]), 0)
    assert p[1] == pytest.approx(p[2], abs=1e-9) and abs(p[1] - p[0]) < 0.2


def test_anomalies_and_target_are_centred_on_the_scored_cells():
    """Seattle: LST covers the urban west only, so its scored cells had a median impervious of 69% against 53% over
    all driven cells; centred before the predictor filter, every anomaly carried that offset."""
    cities = [toy_city(k, n=32) for k in range(4)]
    for c in cities:                                                        # no LST over the less paved half
        t = c.table
        low = t["impervious"] < t["impervious"].median()
        t.loc[low, ["lst", "lst_f100", "lst_f300"]] = np.nan
    s = E.Settings(**{**FAST.__dict__, "within_city": False, "physics": False})
    df, info = E.window_table(cities, "af", s)
    assert info["n_cells_dropped_missing_predictor"] > 0
    for _, g in df.groupby(["slug", "run_id"]):
        for k in ("impervious_c", "canopy_c", "elevation_c", "lst_c", "dTa"):
            assert abs(g[k].median()) < 1e-9, k
    whole = E.design(pd.concat([c.table for c in cities], ignore_index=True))
    scored = whole.dropna(subset=["lst"])
    assert scored.groupby("slug")["impervious_c"].median().min() > 1.0      # the offset the old order left


def _flag(c: D.CityData, run_id: str, why: str) -> D.CityData:
    t = c.table.assign(qa_exclude=np.where(c.table["run_id"] == run_id, why, ""))
    return D.CityData(slug=c.slug, meta=c.meta, table=t, grid=c.grid, rasters=c.rasters)


def test_flagged_runs_are_dropped_by_default_listed_and_kept_on_request():
    base = [toy_city(k, n=32) for k in range(4)]
    c0 = base[0]
    second = c0.table.assign(run_id="2021-07-21 midday", run_date="2021-07-21")
    two = D.CityData(slug=c0.slug, meta=c0.meta, table=pd.concat([c0.table, second], ignore_index=True), grid=c0.grid,
                     rasters=c0.rasters)
    cities = [_flag(two, "2021-07-21 midday", "single-vehicle run: drift not identifiable")] + base[1:]
    s = E.Settings(**{**FAST.__dict__, "within_city": False, "physics": False})
    prepared, flagged = E.prepare(cities, s)
    assert flagged == [{"slug": "toy_0", "run": "af", "run_id": "2021-07-21 midday", "n_cells": len(second),
                        "reason": "single-vehicle run: drift not identifiable", "included": False}]
    assert set(prepared[0].table["run_id"]) == {"2021-07-20 midday"}
    assert E.physics_run_id(prepared[0], "af") == "2021-07-20 midday"     # the physics never sees the run either
    res, pred = E.evaluate_window(cities, "af", s)
    assert res["qa_flagged"] == flagged and set(pred.loc[pred["slug"] == "toy_0", "run_id"]) == {"2021-07-20 midday"}
    res2, pred2 = E.evaluate_window(cities, "af", E.Settings(**{**s.__dict__, "include_flagged": True}))
    assert res2["qa_flagged"][0]["included"] and pred2.loc[pred2["slug"] == "toy_0", "run_id"].nunique() == 2
    # an old table without the column, or an empty reason, is usable
    assert E.prepare(base, s)[1] == [] and E.prepare([_flag(base[1], "none", "")], s)[1] == []


def test_include_flagged_takes_back_walked_and_single_car_runs_only():
    """The registered sensitivity includes walking campaigns and single-car runs; the QA registry's verified faults
    (Fort Lauderdale's rain over 4 of 7 routes: "run excluded", no sensitivity) stay out."""
    base = toy_city(0, n=32)
    rid = "2021-07-20 midday"
    reasons = {"walk": "pedestrian platform (median 4.9 km/h)", "car": "single-vehicle run: drift not identifiable",
               "rain": "qa registry: rain over 4 of 7 routes (51% of readings at RH ≥ 99%)",
               "both": "qa registry: rain over 4 of 7 routes; pedestrian platform (median 4.9 km/h)"}
    cities = [_renamed(_flag(base, rid, why), name) for name, why in reasons.items()]
    s = E.Settings(include_flagged=True, physics=False)
    prepared, flagged = E.prepare(cities, s)
    kept = {c.slug: len(c.table) for c in prepared}
    assert kept == {"walk": len(base.table), "car": len(base.table), "rain": 0, "both": 0}
    assert {f["slug"]: f["included"] for f in flagged} == {"walk": True, "car": True, "rain": False, "both": False}
    assert not any(f["included"] for f in E.prepare(cities, E.Settings(physics=False))[1])
    assert E.includable(reasons["walk"]) and not E.includable("") and not E.includable(reasons["rain"])
    md = E.results_markdown({"target": "t", "n_cities": 4, "transfer_rule": "r", "settings": {"include_flagged": True},
                             "windows": {}, "qa_flagged": flagged})
    assert "rain over 4 of 7 routes (51% of readings at RH ≥ 99%) (excluded)" in md and "km/h) (included)" in md


def test_lst_local_and_regional_columns_come_from_the_rasters():
    c = toy_city(0, n=40)
    s = E.Settings(lst_mode="local", regional=True)
    (p,), _ = E.prepare([c], s)
    cells = c.table["cell"].to_numpy()
    for col, src in (("lst_f3000", "lst"), ("elevation_f3000", "elevation"), ("impervious_f3000", "impervious")):
        assert np.allclose(p.table[col], D.focal_mean(c.rasters[src], 3000, c.grid.dx).ravel()[cells])
    a, b = E.model_columns(s)
    assert {"elevation_f3000_c", "impervious_f3000_c"} <= set(a) and set(b) - set(a) == set(E.LST_COLS)
    assert E.model_columns() == (E.A_COLS, E.B_COLS)
    d = E.design(p.table, s)
    loc = p.table["lst"] - p.table["lst_f3000"]
    assert np.allclose(d["lst_c"], loc - loc.median()) and np.allclose(d["lst_f300_c"], p.table["lst_f300"] -
                                                                      p.table["lst_f3000"] - loc.median())
    assert abs(d["impervious_f3000_c"].median()) < 1e-9
    full = E.design(p.table)
    assert np.allclose(full["lst_c"], p.table["lst"] - p.table["lst"].median())
    # a dataset loaded without rasters cannot give the 3 km columns: an error, not every cell silently dropped
    bare = D.CityData(slug=c.slug, meta=c.meta, table=c.table, grid=c.grid, rasters={})
    with pytest.raises(ValueError, match="no elevation raster"):
        E.prepare([bare], E.Settings(regional=True))
    (nl,), _ = E.prepare([_no_lst(c)], E.Settings(lst_mode="local"))           # no LST: NaN, as the LST itself
    assert nl.table["lst_f3000"].isna().all()


def _with_nlcd(c: D.CityData, shift: float = 5.0) -> D.CityData:
    t = c.table.copy()
    R = dict(c.rasters)
    for base, new in E.NLCD_SWAP.items():
        t[new] = t[base] + shift
        R[new] = c.rasters[base] + shift
    return D.CityData(slug=c.slug, meta=c.meta, table=t, grid=c.grid, rasters=R)


def test_nlcd_land_cover_swaps_canopy_and_impervious_or_refuses():
    cities = [_with_nlcd(toy_city(k, n=32)) for k in range(3)]
    s = E.Settings(land_cover="nlcd")
    prepared, _ = E.prepare(cities, s)
    for c, p in zip(cities, prepared):
        assert np.allclose(p.table["canopy"], c.table["tcc"]) and np.allclose(p.table["impervious_f300"],
                                                                             c.table["nlcd_imp_f300"])
        assert np.allclose(p.rasters["impervious"], c.rasters["nlcd_imp"])          # the physics too
        assert np.allclose(c.table["canopy"] + 5.0, p.table["canopy"])              # the input untouched
    with pytest.raises(ValueError, match="toy_3 lack the NLCD columns"):
        E.prepare(cities + [toy_city(3, n=32)], s)
    # a city that is not scored anyway (no LST over its cells) does not block the variant
    out, _ = E.prepare(cities + [_no_lst(toy_city(3, n=32))], s)
    assert out[-1].table["canopy"].isna().all()


def test_the_medoid_transfer_uses_one_training_city_whole():
    nat = {k: 1.0 for k in E.TRANSFER_KEYS}
    fits = [{"slug": s, "window": "day", "natural": {**nat, "L": L, "e0": e0}, "a_phys": a, "gamma": -0.002,
             "h_lst": h, "g_lst": a * h}
            for s, L, e0, a, h in (("lo", 100.0, -1.0, 0.001, 30.0), ("mid", 300.0, 0.0, 0.002, 20.0),
                                   ("hi", 900.0, 1.0, 0.004, 10.0), ("near_mid", 330.0, 0.1, 0.0021, 19.0))]
    tp = E.transfer_params(fits, True, rule="medoid")
    assert tp["rule"] == "medoid" and tp["medoid"] in ("mid", "near_mid")
    f = next(x for x in fits if x["slug"] == tp["medoid"])
    assert tp["natural"] == f["natural"] and tp["a_phys"] == f["a_phys"] and tp["h_lst"] == f["h_lst"]
    assert tp["a_phys"] * tp["h_lst"] == pytest.approx(f["g_lst"]) and "L" in tp["log"] and "e0" not in tp["log"]
    assert E.transfer_params(fits, True)["rule"] == "median"
    with pytest.raises(ValueError):
        E.Settings(physics_transfer="mean")


def test_physics_fits_on_an_l_bound_are_counted():
    fits = {"C": {"a": {"slug": "a", "natural": {"L": 61.0}, "L_bounds_m": [60.0, 2000.0]},
                  "b": {"slug": "b", "natural": {"L": 1950.0}, "L_bounds_m": [60.0, 2000.0]},
                  "c": {"slug": "c", "natural": {"L": 400.0}, "cell_m": 60.0},
                  "d": {"slug": "d", "error": "failed"}}}
    r = E.l_bounds_report(fits, E.Settings())["C"]
    assert (r["n_fits"], r["n_lower"], r["n_upper"], r["lower"], r["upper"]) == (3, 1, 1, ["a"], ["b"])


def test_sensitivity_runs_each_variant_and_reuses_the_physics_fits(tmp_path, monkeypatch):
    from sparc.core.cli import main

    for k in range(4):
        write_city(toy_city(k, n=32), tmp_path / "data")
    calls = []
    orig = E.fit_city_physics
    monkeypatch.setattr(E, "fit_city_physics", lambda *a, **k: calls.append(a[0].slug) or orig(*a, **k))
    rc = main(["lstair", "sensitivity", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "sens"),
               "--variants", "weights_n,nlcd,physics_medoid", "--no-gbm", "--physics-iter", "5", "--boot", "200",
               "--windows", "af"])
    assert rc == 0
    sens = json.loads((tmp_path / "sens" / "sensitivity.json").read_text())
    rows = {r["variant"]: r for r in sens["rows"]}
    assert list(rows) == ["primary", "absolute_dmse", "leave_one_unit_out", "weights_n", "nlcd", "physics_medoid"]
    assert rows["nlcd"]["status"] == "not run" and "NLCD" in rows["nlcd"]["reason"]
    assert rows["weights_n"]["windows"]["af"]["B-A"]["verdict"] == "not testable"          # 4 units
    assert rows["absolute_dmse"]["windows"]["af"]["B-A"]["mean_diff_mse"] is not None
    assert json.loads((tmp_path / "sens" / "weights_n" / "results.json").read_text())["settings"]["weights"] == "n"
    # the variants that leave the fits alone reuse the primary run's cache; none of them refits
    assert len(list((tmp_path / "sens" / "primary" / "physics").glob("*.json"))) == 8
    assert not (tmp_path / "sens" / "weights_n" / "physics").exists() and len(calls) == 8
    md = (tmp_path / "sens" / "sensitivity.md").read_text()
    assert "## B − A" in md and "## C − A" in md and "| weights_n (" in md and "not run:" in md
    assert main(["lstair", "sensitivity", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "x"),
                 "--variants", "nope"]) == 1
