"""Study S1 (calibrating held-out maps with local stations), offline on synthetic cities: simple kriging recovers a
smooth field, calibration helps when residuals are spatially correlated and not when they are white noise, a
unit's variogram never sees its own cells, the station draws are seeded and keep their separation and buffer,
and the real-station (ISD) path with a mocked reader and station list."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sparc.core.lstair import calibrate as K
from sparc.core.lstair import dataset as D
from sparc.core.synthetic import gaussian_random_field

N, DX = 60, 100.0                               # 6 km square cities of 100 m cells; streets every 3rd row/column
RUN = "2021-07-15 midday"


def _grid(u: int) -> D.CityGrid:
    return D.CityGrid(crs="EPSG:32619", x0=300050.0 + 20000 * u, y0=4600050.0, dx=DX, nx=N, ny=N)


def toy_pred(n_units: int = 10, smooth: bool = True, seed: int = 0, amp: float = 1.0) -> pd.DataFrame:
    """A prediction table like ``lstair evaluate``'s: per unit one af run on street cells; dTa = pred_A + residual
    (a smooth 2 km field, or white noise) + a little noise; dTa and pred median-centred."""
    rng = np.random.default_rng(seed)
    iy, ix = np.divmod(np.arange(N * N), N)
    street = (iy % 3 == 0) | (ix % 3 == 0)
    rows = []
    for u in range(n_units):
        g = _grid(u)
        lc = gaussian_random_field((N, N), 6.0, rng).ravel()
        res = (gaussian_random_field((N, N), 20.0, rng).ravel() if smooth else rng.standard_normal(N * N)) * amp
        cells = np.flatnonzero(street)
        x, y = g.cell_xy(cells)
        pred = 0.8 * lc[cells]
        dta = pred + res[cells] + 0.05 * rng.standard_normal(cells.size)
        rows.append(pd.DataFrame({"city": f"Toy {u}", "slug": f"toy_{u}", "group": f"toy_{u}", "run": "af",
                                  "run_id": RUN, "cell": cells, "x": x, "y": y, "dTa": dta - np.median(dta),
                                  "pred_A": pred - np.median(pred)}))
    return pd.concat(rows, ignore_index=True)


FAST = dict(ks=(3, 10), draws=5, windows=("af",), bases=("A",), stations=False, boot=1000)


def test_kriging_recovers_a_smooth_field():
    rng = np.random.default_rng(1)
    f = lambda xy: np.sin(2 * np.pi * xy[:, 0] / 8000.0) + np.cos(2 * np.pi * xy[:, 1] / 10000.0)  # noqa: E731
    xs, xt = rng.uniform(0, 10000, (80, 2)), rng.uniform(1000, 9000, (500, 2))
    r = f(xs)
    est = K.simple_krige(xs, r - r.mean(), xt, {"status": "ok", "nugget": 0.0, "psill": 1.0, "range_m": 4000.0})
    truth = f(xt) - r.mean()
    assert np.corrcoef(est, truth)[0, 1] > 0.98
    assert np.sqrt(np.mean((est - truth) ** 2)) < 0.25 * truth.std()
    # no structured part → nothing is spread
    assert not K.simple_krige(xs, r, xt, {"status": "ok", "nugget": 1.0, "psill": 0.0, "range_m": 100.0}).any()


def test_variogram_fit_tells_smooth_from_white():
    for smooth, share in ((True, (0.0, 0.3)), (False, (0.7, 1.0))):
        df = toy_pred(4, smooth=smooth)
        s = K.CalSettings(**FAST)
        v = K.unit_variograms(K.window_pair_stats(df, ("A",), s), ("A",))["A"]["_all"]
        assert v["status"] == "ok"
        assert share[0] <= v["nugget_share"] <= share[1] or (not smooth and v["range_m"] < 50), v
        if smooth:
            assert 300 < v["range_m"] < 10000


def test_calibration_helps_when_residuals_are_correlated_and_not_when_white():
    s = K.CalSettings(**FAST)
    res = K.calibrate(toy_pred(10, smooth=True), s)
    rec = res["windows"]["af"]["bases"]["A"]["k"]
    assert rec[10]["mean_dss"] > 0.1 and rec[10]["n_better"] == 10
    assert rec[10]["mean_dss"] > rec[3]["mean_dss"] > 0
    assert rec[10]["verdict"] == K.CLAIM and rec[10]["p_holm"] <= K.ALPHA
    assert rec[10]["holm"]["m"] == 2
    white = K.calibrate(toy_pred(10, smooth=False, seed=3), s)["windows"]["af"]["bases"]["A"]["k"]
    for k in (3, 10):
        assert abs(white[k]["mean_dss"]) < 0.01
        assert white[k]["verdict"] != K.CLAIM
    md = K.calibrate_markdown(res)
    assert "## Primary: A calibrated − A, af" in md and "| 10 | 10 |" in md


def test_unit_variogram_excludes_the_units_own_cells():
    s = K.CalSettings(**FAST)
    df = toy_pred(5, smooth=True)
    v = K.unit_variograms(K.window_pair_stats(df, ("A",), s), ("A",))["A"]
    u = "toy_2"
    alone = K.unit_variograms(K.window_pair_stats(df[df["group"] != u], ("A",), s), ("A",))["A"]["_all"]
    for key in ("nugget", "psill", "range_m", "n_pairs"):
        assert v[u][key] == pytest.approx(alone[key], rel=1e-9)
    # scrambling u's labels changes every other unit's covariance but never u's own
    df2 = df.copy()
    m = df2["group"] == u
    df2.loc[m, "dTa"] = np.random.default_rng(9).permutation(df2.loc[m, "dTa"].to_numpy()) * 3.0
    v2 = K.unit_variograms(K.window_pair_stats(df2, ("A",), s), ("A",))["A"]
    assert v2[u]["nugget"] == pytest.approx(v[u]["nugget"], rel=1e-12)
    assert v2[u]["range_m"] == pytest.approx(v[u]["range_m"], rel=1e-12)
    assert v2["toy_0"]["nugget_share"] != pytest.approx(v["toy_0"]["nugget_share"], rel=1e-3)


def test_draws_are_seeded_and_keep_separation_and_buffer():
    df = toy_pred(1)
    xy = df[["x", "y"]].to_numpy(float)
    a = K.draw_stations(xy, 10, 1000.0, np.random.default_rng([0, 10, 7]))
    b = K.draw_stations(xy, 10, 1000.0, np.random.default_rng([0, 10, 7]))
    c = K.draw_stations(xy, 10, 1000.0, np.random.default_rng([1, 10, 7]))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    d = np.hypot(*(xy[a][:, None, :] - xy[a][None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(10, 1)].min() >= 1000.0
    far = K.far_from(xy, xy[a], 500.0)
    dmin = np.hypot(*(xy[:, None, :] - xy[a][None, :, :]).transpose(2, 0, 1)).min(axis=1)
    assert (dmin[far] > 500.0).all() and (dmin[~far] <= 500.0).all()
    assert K.draw_stations(xy[:50], 30, 1000.0, np.random.default_rng(0)) is None       # no room for 30
    # the whole run: identical twice, identical stations for every base, skipped when nothing is left to score
    s = K.CalSettings(**{**FAST, "bases": ("A", "B")})
    preds = {"A": df["pred_A"].to_numpy(), "B": 0.5 * df["pred_A"].to_numpy()}
    vg = {"status": "ok", "nugget": 0.1, "psill": 0.9, "range_m": 800.0}
    r1 = K.calibrate_run(xy, df["dTa"].to_numpy(), preds, {"A": vg, "B": vg}, s, key=7)
    r2 = K.calibrate_run(xy, df["dTa"].to_numpy(), preds, {"A": vg, "B": vg}, s, key=7)
    assert json.dumps(r1) == json.dumps(r2) and r1[10]["n_valid"] == 5
    assert r1[10]["n_test"] == K.calibrate_run(xy, df["dTa"].to_numpy(), {"A": preds["A"]}, {"A": vg}, s, 7)[10]["n_test"]
    big = K.CalSettings(**{**FAST, "ks": (3,), "buffer_m": 5000.0})
    r3 = K.calibrate_run(xy, df["dTa"].to_numpy(), {"A": preds["A"]}, {"A": vg}, big, key=7)
    assert r3[3]["n_valid"] == 0 and r3[3]["skipped_few_cells"] == 5


def _toy_city(u: int) -> D.CityData:
    meta = {"slug": f"toy_{u}", "city": f"Toy {u}", "tz": "America/New_York", "campaign_date": "2021-07-15",
            "runs": [{"run_id": RUN, "run": "af", "start": "2021-07-15 15:00:00", "end": "2021-07-15 16:00:59"}]}
    return D.CityData(slug=f"toy_{u}", meta=meta, table=pd.DataFrame(), grid=_grid(u))


def test_real_station_path_with_a_mocked_reader():
    df = toy_pred(10, smooth=True, seed=4)
    cities = {f"toy_{u}": _toy_city(u) for u in range(10)}
    hist, truth = [], {}
    for u in range(10):
        part = df[df["slug"] == f"toy_{u}"]
        g = cities[f"toy_{u}"].grid
        pick = part.sample(6, random_state=u)
        if u == 9:
            pick = pick.iloc[:1]                                       # one usable station: not scored
        for i, r in enumerate(pick.itertuples()):
            sid = f"72{u:02d}{i:02d}99999"
            lon, lat = g.to_lonlat([r.x + 20.0], [r.y + 20.0])
            hist.append({"station": sid, "name": f"S{u}{i}", "icao": "", "lat": lat[0], "lon": lon[0],
                         "begin": 20000101, "end": 20251231})
            truth[sid] = 30.0 + r.dTa
        lon, lat = g.to_lonlat([g.x0 + 30.0], [g.y0 + 30.0])
        hist.append({"station": f"73{u:02d}0099999", "name": "OLD", "icao": "", "lat": lat[0], "lon": lon[0],
                     "begin": 19500101, "end": 19991231})                # closed before the run: not in the grid
        lon, lat = g.to_lonlat([g.x0 - 5000.0], [g.y0])
        hist.append({"station": f"74{u:02d}0099999", "name": "AWAY", "icao": "", "lat": lat[0], "lon": lon[0],
                     "begin": 20000101, "end": 20251231})                # outside the grid
    hist = pd.DataFrame(hist)
    # toy_0's first station listed again under a second id at the same place (one site); toy_1's second station
    # has no Global Hourly file for the year
    dup = hist.iloc[0].copy()
    dup["station"] = "72000099998"
    truth["72000099998"] = truth[hist.iloc[0]["station"]]
    hist = pd.concat([hist, dup.to_frame().T.astype(hist.dtypes.to_dict())], ignore_index=True)
    calls = []

    def reader(station, start_utc, end_utc):
        calls.append((station, start_utc, end_utc))
        if station == "72010199999":
            raise FileNotFoundError("no Global Hourly file")
        h = hist.set_index("station").loc[station]
        return {"station": station, "lat": float(h["lat"]), "lon": float(h["lon"]), "t_C": truth[station],
                "n_reports": 5 if station != "72000099998" else 3}

    s = K.CalSettings(**{**FAST, "stations": True})
    res = K.calibrate(df, s, cities=cities, history=hist, reader=reader)
    rs = res["real_stations"]
    assert rs["status"] == "done"
    assert calls[0][1].isoformat() == "2021-07-15T19:00:00+00:00"       # 15 h EDT in UTC
    by = {c["slug"]: c["runs"][0] for c in rs["cities"]}
    assert by["toy_0"]["n_in_grid"] == 7 and by["toy_0"]["n_usable"] == 6   # the duplicate is one site
    assert any(q["status"] == "duplicate site" for q in by["toy_0"]["readings"])
    assert by["toy_1"]["n_in_grid"] == 6 and by["toy_1"]["n_with_obs"] == 5 and by["toy_1"]["n_usable"] == 5
    assert by["toy_9"]["n_usable"] == 1 and not by["toy_9"]["scored"]
    rec = rs["paired"]["A"]["af"]["all"]
    assert rec["n_units"] == 9 and rec["mean_dss"] > 0 and rec["n_better"] >= 8
    assert rs["paired"]["A"]["af"]["far"]["n_units"] <= 9
    assert "| Toy 9 | 1 | af |" in K.calibrate_markdown(res)


def test_isd_history_parse_and_unreachable(monkeypatch):
    raw = (b'"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
           b'"725070","14765","PROVIDENCE T F GREEN","US","RI","KPVD","+41.723","-071.433","+0016.8","19420801","20250819"\n'
           b'"007018","99999","WXPOD 7018","","","","+00.000","+000.000","+7018.0","20110309","20130730"\n'
           b'"999999","00001","NO POSITION","","","","","","","20000101","20010101"\n')
    h = K.parse_isd_history(raw)
    assert h["station"].tolist() == ["72507014765"] and h["icao"].iat[0] == "KPVD"

    def down(url):
        raise OSError("no route to host")

    with pytest.raises(ConnectionError, match="no route"):
        K.isd_history(fetch=down)

    def unreachable(fetch=None, timeout=30.0):
        raise ConnectionError("noaa-isd-pds: timed out")

    monkeypatch.setattr(K, "isd_history", unreachable)
    res = K.calibrate(toy_pred(3), K.CalSettings(**{**FAST, "stations": True}), cities={})
    assert res["real_stations"] == {"status": "ISD unreachable", "reason": "noaa-isd-pds: timed out"}
    assert "ISD unreachable: noaa-isd-pds: timed out" in K.calibrate_markdown(res)


def test_cli_calibrate_writes_results(tmp_path):
    from sparc.core.cli import main

    ev = tmp_path / "eval"
    ev.mkdir()
    toy_pred(4).to_parquet(ev / "predictions.parquet", index=False)
    rc = main(["lstair", "calibrate", "--data", str(tmp_path / "data"), "--eval", str(ev), "--out",
               str(tmp_path / "cal"), "--k", "3", "--draws", "2", "--windows", "af", "--bases", "A", "--no-stations"])
    assert rc == 0
    out = json.loads((tmp_path / "cal" / "calibrate.json").read_text())
    assert out["settings"]["ks"] == [3] and out["real_stations"]["status"].startswith("skipped")
    assert out["windows"]["af"]["bases"]["A"]["k"]["3"]["verdict"] == "not testable"     # 4 units < 8
    assert (tmp_path / "cal" / "calibrate.md").exists()
    assert main(["lstair", "calibrate", "--data", "x", "--eval", str(tmp_path / "none"), "--out",
                 str(tmp_path / "c2")]) == 1
