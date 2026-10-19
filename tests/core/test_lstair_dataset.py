"""LST pilot dataset (L3): synthetic traverse folders in the CAPA layout + synthetic feature/LST rasters → the
long table (rows, units, backgrounds, drift and vehicle offsets, no target leakage), the per-city cache, the
fail-fast network check and the CLI."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import urllib.error

import numpy as np
import pandas as pd
import pytest

from sparc.core.lstair import dataset as D

CENTRES = {"Providence, RI": (-71.41, 41.82), "Boston, MA": (-71.06, 42.36)}
UTM19 = "EPSG:32619"


def _truth(x, y, x0, y0):
    """Planted afternoon air temperature (°C): a 1 °C/km west–east gradient plus a warm blob."""
    return 28.0 + 0.001 * (x - x0) + 1.5 * np.exp(-((x - x0 - 600) ** 2 + (y - y0 - 300) ** 2) / (2 * 250.0 ** 2))


def _routes(x0, y0, size=2400.0, spacing=300.0, step=10.0):
    """Street lines every ``spacing`` m (east–west and north–south), sampled every ``step`` m."""
    segs = []
    s = np.arange(-size / 2, size / 2 + 1e-9, step)
    for off in np.arange(-size / 2, size / 2 + 1e-9, spacing):
        segs.append((x0 + s, np.full(s.size, y0 + off)))
        segs.append((np.full(s.size, x0 + off), y0 + s))
    return segs


def write_campaign(folder, centre, date="2020-07-29", drift_c_per_h=1.2, offsets=(0.4, -0.3, 0.0), seed=0,
                   temp_shift=None, capa_layout_run="am"):
    """A campaign folder as CAPA publishes one: one table per run; the afternoon as ``datetime, lat, lon, T_F,
    car`` and the morning as CAPA's logger table (``file`` with the sensor id, ``dttm_lc`` local clock, ``t_c``)."""
    from pyproj import Transformer

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    to_xy = Transformer.from_crs("EPSG:4326", UTM19, always_xy=True)
    to_ll = Transformer.from_crs(UTM19, "EPSG:4326", always_xy=True)
    x0, y0 = to_xy.transform(*centre)
    segs = _routes(x0, y0)
    nv = len(offsets)
    # each vehicle drives its streets, then its first two again ~25 min later: the repeat visits that identify the
    # drift within cell fixed effects (crossings a few minutes apart do not; see adjust_visits)
    plan = [(k % nv, xs, ys) for k, (xs, ys) in enumerate(segs)] + [(v, *segs[v + nv * j]) for j in range(2)
                                                                     for v in range(nv)]
    for run, start, base in (("af", "15:00:00", 0.0), ("am", "06:05:00", -6.0)):
        rows = []
        t_veh = [0.0] * nv
        for v, xs, ys in plan:
            n = xs.size
            ts = t_veh[v] + np.arange(n) * 1.0                                   # 1 Hz at 10 m/s
            t_veh[v] = ts[-1] + 20.0
            T = _truth(xs, ys, x0, y0) + base
            if temp_shift is not None:
                T = T + temp_shift(xs, ys)
            T = T + drift_c_per_h * (ts - 1800.0) / 3600.0 + offsets[v] + 0.05 * rng.standard_normal(n)
            lon, lat = to_ll.transform(xs, ys)
            rows.append(pd.DataFrame({"t": ts, "lat": lat, "lon": lon, "T": T, "car": v + 1}))
        df = pd.concat(rows, ignore_index=True)
        df = df[df["t"] < 3500.0]                                                # one hour of driving
        when = pd.Timestamp(f"{date} {start}") + pd.to_timedelta(df["t"], unit="s")
        if run == capa_layout_run:
            sensor = df["car"].map({1: "CAPA1061", 2: "CAPA1079", 3: "CAPA1085"})
            out = pd.DataFrame({"file": sensor + "_290720_060000_0.csv", "dttm_lc": when.dt.strftime("%Y-%m-%d %H:%M:%S"),
                                "t_c": df["T"].round(3), "lat": df["lat"].round(7), "lon": df["lon"].round(7)})
        else:
            out = pd.DataFrame({"datetime": when.dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": df["lat"].round(7),
                                "lon": df["lon"].round(7), "T_F": (df["T"] * 9 / 5 + 32).round(3), "car": df["car"]})
        out.to_csv(folder / f"{run}_trav.csv", index=False)
    (folder / "af_t_f.tif").write_bytes(b"II*\x00")                                 # a raster: skipped
    return x0, y0


def synthetic_features(ctx, campaign_date):
    X, Y = ctx.centres()
    xr, yr = X - X.mean(), Y - Y.mean()
    feats = {"canopy": np.clip(30 + 20 * np.sin(xr / 500.0), 0, 100), "impervious": np.clip(50 + 30 * np.cos(yr / 700.0), 0, 100),
             "ndvi": 0.3 + 0.1 * np.sin(xr / 500.0), "albedo": 0.15 + 0.02 * np.cos(yr / 300.0),
             "elevation": 10.0 + 0.002 * yr, "water_distance": np.abs(xr - 1500.0)}
    return feats, {"source": "synthetic"}


def synthetic_lst(ctx, campaign_date, study):
    X, Y = ctx.centres()
    lst = 38.0 + 0.002 * (X - X.mean()) + 0.0 * Y
    lst[:5] = np.nan                                                                # a cloudy strip
    scene = {"id": "LC08_test", "date": (campaign_date - dt.timedelta(days=3)).isoformat(), "offset_days": -3,
             "clear_fraction": 0.97, "selected": True}
    assert study.shape == ctx.shape and study.any()
    return {"lst_c": lst, "scenes": [scene], "candidates": [scene]}


def fake_era5(lon, lat, tz, start, end):
    assert tz == "America/New_York"
    return {"ta_era5": 27.0 if start.hour >= 12 else 21.0, "sw_down": 650.0 if start.hour >= 12 else 150.0,
            "lw_net": -90.0, "hours_local": [start.hour, start.hour + 1]}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("lstair")
    trav = root / "traverses"
    for name, c in CENTRES.items():
        write_campaign(trav / D.slugify(name), c, seed=len(name))
    out = root / "data"
    summary = D.build_dataset(list(CENTRES), out, traverses_dir=trav, features=synthetic_features, lst=synthetic_lst,
                              era5=fake_era5, probe_opener=_no_network)
    return root, out, summary


def _no_network(req, timeout=None):
    raise AssertionError(f"no host should be probed when every source is local or synthetic: {req.full_url}")


def test_the_table_has_one_row_per_run_and_cell_in_celsius_with_run_backgrounds(built):
    root, out, summary = built
    assert {v["status"] for v in summary["cities"].values()} == {"ok"}
    cities = D.load_dataset(out)
    assert [c.slug for c in cities] == ["boston_ma", "providence_ri"]
    for c in cities:
        t = c.table
        assert set(t["run"]) == {"am", "af"} and not t.duplicated(["run_id", "cell"]).any()
        assert {"city", "campaign_date", "run", "cell", "x", "y", "lat", "lon", "n_readings", "Ta", "dTa",
                "ta_era5"} <= set(t.columns)
        assert t["campaign_date"].unique().tolist() == ["2020-07-29"]
        af = t[t["run"] == "af"]
        assert 20 < af["Ta"].median() < 35                                       # °C, not °F
        for _, g in t.groupby("run_id"):
            assert g["dTa"].median() == pytest.approx(0.0, abs=1e-9)
            assert np.allclose(g["Ta"] - g["dTa"], g["ta_background"])
        assert (t["n_readings"] >= t["n_visits"]).all() and t["n_readings"].median() >= 2
        assert set(t.loc[t["run"] == "af", "ta_era5"]) == {27.0} and set(t.loc[t["run"] == "am", "ta_era5"]) == {21.0}
        # the positions sit on the 30 m grid in UTM 19N, inside the lat/lon of the city
        assert c.grid.crs == UTM19 and c.grid.dx == 30.0
        lon0, lat0 = CENTRES[c.meta["city"]]
        assert abs(t["lon"].median() - lon0) < 0.02 and abs(t["lat"].median() - lat0) < 0.02


def test_drift_and_vehicle_offsets_are_removed_and_the_pattern_recovered(built):
    from pyproj import Transformer

    _, out, _ = built
    c = D.load_city(out / "cities" / "providence_ri")
    af = next(r for r in c.meta["runs"] if r["run"] == "af")
    assert af["drift"]["method"] == "fe" and af["drift"]["drift_per_h"] == pytest.approx(1.2, abs=0.08)
    offs = np.array(sorted(af["vehicle_offsets_c"].values()))
    assert np.allclose(offs, np.sort(np.array([0.4, -0.3, 0.0]) - np.mean([0.4, -0.3, 0.0])), atol=0.06)
    t = c.table[c.table["run"] == "af"]
    x0, y0 = Transformer.from_crs("EPSG:4326", UTM19, always_xy=True).transform(*CENTRES["Providence, RI"])
    truth = _truth(t["x"].to_numpy(), t["y"].to_numpy(), x0, y0)
    resid = t["Ta"].to_numpy() - truth
    assert np.corrcoef(t["Ta"], truth)[0, 1] > 0.99 and resid.std() < 0.08


def test_predictors_are_the_rasters_at_the_cells_and_the_lst_is_matched(built):
    _, out, _ = built
    c = D.load_city(out / "cities" / "providence_ri")
    t = c.table
    feats, _ = synthetic_features(c.grid, None)
    iy, ix = np.divmod(t["cell"].to_numpy(), c.grid.nx)
    for k in D.FEATURES:
        assert np.allclose(t[k], feats[k][iy, ix], atol=1e-4 * max(1.0, np.abs(feats[k]).max())), k
    for k in ("canopy_f100", "canopy_f300", "canopy_f1000", "albedo_f1000", "lst", "lst_f100", "lst_f300"):
        assert k in t and np.isfinite(t[k]).mean() > 0.9, k
    # a 1000 m Gaussian focal mean is smoother than the 100 m one
    assert t["canopy_f1000"].std() < t["canopy_f100"].std()
    assert set(t["lst_scene_date"]) == {"2020-07-26"} and set(t["lst_offset_days"]) == {-3}
    assert c.meta["lst"]["scenes"][0]["id"] == "LC08_test"
    assert c.rasters["canopy"].shape == c.grid.shape and c.rasters["driven"].sum() == t["cell"].nunique()


def test_no_target_enters_the_predictors(built, tmp_path):
    root, out, _ = built
    trav = tmp_path / "shifted"
    write_campaign(trav / "providence_ri", CENTRES["Providence, RI"], seed=len("Providence, RI"),
                   temp_shift=lambda x, y: 3.0 * np.sin(x / 200.0))
    out2 = tmp_path / "data"
    D.build_dataset(["Providence, RI"], out2, traverses_dir=trav, features=synthetic_features, lst=synthetic_lst,
                    era5=fake_era5, probe_opener=_no_network)
    a = D.load_city(out / "cities" / "providence_ri").table.sort_values(["run_id", "cell"]).reset_index(drop=True)
    b = D.load_city(out2 / "cities" / "providence_ri").table.sort_values(["run_id", "cell"]).reset_index(drop=True)
    assert a["cell"].tolist() == b["cell"].tolist()
    targets = set(D.TARGET_COLUMNS) | {"ta_era5"}
    preds = [k for k in a.columns if k not in targets and a[k].dtype.kind == "f" and k not in ("x", "y", "lat", "lon")]
    assert {"canopy", "lst", "lst_f300", "impervious_f1000"} <= set(preds)
    for k in preds:
        assert np.array_equal(a[k].to_numpy(), b[k].to_numpy(), equal_nan=True), k
    assert not np.allclose(a["dTa"], b["dTa"])


def test_reruns_skip_finished_cities_and_retry_failed_ones(built, tmp_path):
    root, out, _ = built
    calls = []

    def counting_features(ctx, d):
        calls.append(ctx)
        return synthetic_features(ctx, d)

    s = D.build_dataset(list(CENTRES), out, traverses_dir=root / "traverses", features=counting_features,
                        lst=synthetic_lst, era5=fake_era5, probe_opener=_no_network)
    assert calls == [] and {v.get("skipped") for v in s["cities"].values()} == {"cached"}
    bad = tmp_path / "data"
    (bad / "cities" / "boston_ma").mkdir(parents=True)
    (bad / "cities" / "boston_ma" / "meta.json").write_text(json.dumps({"status": "failed"}))
    s2 = D.build_dataset(["Boston, MA"], bad, traverses_dir=root / "traverses", features=counting_features,
                         lst=False, era5=False, probe_opener=_no_network)
    assert s2["cities"]["Boston, MA"]["status"] == "ok" and len(calls) == 1
    t = D.load_city(bad / "cities" / "boston_ma").table
    assert t["lst"].isna().all() and t["ta_era5"].isna().all()


def test_a_broken_city_is_recorded_and_the_others_still_build(built, tmp_path):
    root, _, _ = built
    trav = tmp_path / "trav"
    shutil.copytree(root / "traverses" / "boston_ma", trav / "boston_ma")
    (trav / "providence_ri").mkdir(parents=True)
    (trav / "providence_ri" / "notes.txt").write_text("nothing here")
    s = D.build_dataset(list(CENTRES), tmp_path / "out", traverses_dir=trav, features=synthetic_features,
                        lst=False, era5=False, probe_opener=_no_network)
    assert s["cities"]["Boston, MA"]["status"] == "ok"
    assert s["cities"]["Providence, RI"]["status"] == "failed" and "no traverse points" in s["cities"]["Providence, RI"]["error"]


def test_blocked_hosts_fail_fast_with_their_name(tmp_path):
    calls = []

    def blocked(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    with pytest.raises(D.NetworkBlocked) as exc:
        D.build_dataset(["Providence, RI"], tmp_path / "out", features=synthetic_features, lst=False, era5=False,
                        probe_opener=blocked)
    assert exc.value.host == "api.osf.io" and "api.osf.io" in str(exc.value) and "--traverses-dir" in str(exc.value)
    assert calls == ["https://api.osf.io/v2/"]
    assert not (tmp_path / "out" / "raw").exists()

    def http403(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    with pytest.raises(D.NetworkBlocked, match="planetarycomputer.microsoft.com"):
        D.check_hosts(["landsat"], opener=http403)
    assert set(D.hosts_needed(True, True, True, True)) == set(D.HOSTS)
    req = D.required_host_names()
    assert {"api.osf.io", "files.osf.io", "planetarycomputer.microsoft.com", "nsf-ncar-era5.s3.amazonaws.com"} <= set(req)


def test_traverses_are_downloaded_when_no_local_folder_is_given(built, tmp_path):
    root, _, _ = built
    seen = []

    def ok(req, timeout=None):
        seen.append(req.full_url)
        return _osf_opener(req, timeout)

    def download(spec, dest):
        assert spec.node == "wu9v7" and spec.folder_hint is None
        shutil.copytree(root / "traverses" / "providence_ri", dest)

    s = D.build_dataset(["Providence, RI"], tmp_path / "out", features=synthetic_features, lst=False, era5=False,
                        probe_opener=ok, download=download)
    assert s["cities"]["Providence, RI"]["status"] == "ok" and seen[0] == "https://api.osf.io/v2/"
    assert any(u.startswith("https://files.osf.io/v1/resources/wu9v7/") for u in seen)      # a real download probed
    assert (tmp_path / "out" / "raw" / "providence_ri" / "af_trav.csv").exists()


def test_adjust_visits_recovers_drift_and_offsets_and_falls_back_without_repeats():
    rng = np.random.default_rng(1)
    n_cells, n = 400, 6000
    cells = rng.integers(0, n_cells, n)
    veh = rng.integers(0, 4, n)
    t = rng.uniform(0, 3600, n)
    mu = rng.normal(30, 1, n_cells)
    off = np.array([0.5, -0.2, 0.1, -0.4])
    temp = mu[cells] + 1.5 * (t - np.median(t)) / 3600 + off[veh] + 0.02 * rng.standard_normal(n)
    v = pd.DataFrame({"cell": cells, "vehicle": veh, "t_s": t, "temp": temp})
    adj, info = D.adjust_visits(v)
    assert info["method"] == "fe" and info["drift_per_h"] == pytest.approx(1.5, abs=0.02)
    got = np.array([info["vehicle_offsets"][str(k)] for k in range(4)])
    assert np.allclose(got, off - np.average(off, weights=np.bincount(veh)), atol=0.01)
    per_cell = pd.Series(adj).groupby(cells).mean()
    assert np.allclose(per_cell - per_cell.mean(), mu[per_cell.index] - mu[per_cell.index].mean(), atol=0.03)
    # one visit per cell: drift is not identified against cell effects → CAPA's protocol line through the hour
    v1 = pd.DataFrame({"cell": np.arange(500), "vehicle": 0, "t_s": np.linspace(0, 3600, 500),
                       "temp": 30 + 1.0 * np.linspace(-0.5, 0.5, 500)})
    adj1, info1 = D.adjust_visits(v1)
    assert info1["method"] == "protocol" and "fallback" in info1 and info1["drift_per_h"] == pytest.approx(1.0)
    assert np.allclose(adj1, 30.0)


def test_city_grid_orientation_and_target_georeference():
    g = D.make_city_grid(np.array([-71.42, -71.40]), np.array([41.81, 41.83]), UTM19, 30.0, 300.0)
    assert g.x0 % 30.0 == pytest.approx(15.0) and g.y0 % 30.0 == pytest.approx(15.0)
    north = np.arange(g.ny * g.nx, dtype=float).reshape(g.ny, g.nx)[::-1]       # row 0 = north
    arr = g.from_northup(north)
    T = g.target["transform"]
    X, Y = g.centres()
    # the north-up pixel containing each cell centre is that cell
    cols = ((X - T.c) / T.a).astype(int)
    rows = ((Y - T.f) / T.e).astype(int)
    assert np.array_equal(north[rows, cols], arr)
    w, s, e, n = g.bounds_lonlat()
    assert w < -71.42 and e > -71.40 and s < 41.81 and n > 41.83


def test_cli_fails_fast_naming_the_blocked_host(tmp_path, monkeypatch, capsys):
    from sparc.core.cli import main

    monkeypatch.setattr(D, "probe", lambda url, timeout=10.0, opener=None: "URLError: Tunnel connection failed: 403")
    rc = main(["lstair", "dataset", "--cities", "Providence, RI", "--out", str(tmp_path / "o")])
    assert rc == 2 and "api.osf.io" in capsys.readouterr().err


def test_fe_drift_falls_back_when_revisits_are_only_minutes_apart():
    """Real Heat Watch routes (R7): each vehicle drives its own route and comes back to a cell within minutes, so
    cell fixed effects cannot see the drift over the hour; the run must fall back to CAPA's protocol line."""
    rng = np.random.default_rng(3)
    rows = []
    for veh in range(3):                                         # three vehicles, disjoint routes
        t = 0.0
        for leg in range(12):                                    # 12 legs of 40 cells, each driven out and back
            cells = 10_000 * veh + 40 * leg + np.arange(40)
            for c in list(cells) + list(cells[::-1]):            # back along the same street within ~1.5 min
                rows.append((c, veh, t))
                t += 1.2
            t += 120.0
    v = pd.DataFrame(rows, columns=["cell", "vehicle", "t_s"])
    mu = rng.normal(30.0, 0.8, v["cell"].max() + 1)
    v["temp"] = (mu[v["cell"]] + 1.5 * (v["t_s"] - v["t_s"].median()) / 3600.0 + 0.4 * v["vehicle"]
                 + 0.05 * rng.standard_normal(len(v)))
    adj, info = D.adjust_visits(v)
    assert info["n_repeat_cells"] > 1000 and info["n_repeat_cells_spread"] == 0
    assert info["method"] == "protocol" and "min apart" in info["fallback"] and info["vehicle_offsets"] == {}
    assert np.isfinite(info["drift_se_per_h"])
    # revisits spread over the run: the fixed-effect drift is identified and its SE is recorded
    rows2 = [(c, k % 3, t) for k, (c, t) in enumerate(zip(np.tile(np.arange(300), 4), np.linspace(0, 3600, 1200)))]
    v2 = pd.DataFrame(rows2, columns=["cell", "vehicle", "t_s"])
    v2["temp"] = mu[v2["cell"]] + 1.5 * (v2["t_s"] - v2["t_s"].median()) / 3600.0 + 0.02 * rng.standard_normal(len(v2))
    _, info2 = D.adjust_visits(v2)
    assert info2["method"] == "fe" and info2["drift_per_h"] == pytest.approx(1.5, abs=0.05)
    assert 0 < info2["drift_se_per_h"] < D.MAX_DRIFT_SE
    # a noisy fixed-effect drift (SE above the limit) also falls back
    v3 = v2.assign(temp=v2["temp"] + 2.0 * rng.standard_normal(len(v2)))
    _, info3 = D.adjust_visits(v3, max_drift_se=0.05)
    assert info3["method"] == "protocol" and "SE above" in info3["fallback"] and info3["fe_drift_se_per_h"] > 0.05


def test_repeated_traverse_files_are_read_once_and_appendix_archives_skipped(built, tmp_path):
    """R8: CAPA's Providence node ships the afternoon run in its main archive and again in an appendix archive
    (plus extra sensors); reading both doubled readings and visits and changed the drift and the target."""
    import zipfile

    root, out, _ = built
    trav = tmp_path / "trav" / "providence_ri"
    shutil.copytree(root / "traverses" / "providence_ri", trav)
    af = pd.read_csv(trav / "af_trav.csv")
    extra = af[af["car"] == 1].assign(car=9, lat=af["lat"] + 0.3)               # a sensor far outside the study area
    with zipfile.ZipFile(trav / "Appendix Data_Heat Watch RI.zip", "w") as z:
        z.writestr("appendix_af_trav.csv", pd.concat([af, extra]).to_csv(index=False))
    kw = dict(features=synthetic_features, lst=False, era5=False, probe_opener=_no_network)
    D.build_dataset(["Providence, RI"], tmp_path / "o1", traverses_dir=tmp_path / "trav", **kw)
    c1 = D.load_city(tmp_path / "o1" / "cities" / "providence_ri")
    ref = D.load_city(out / "cities" / "providence_ri")
    skipped = [f for f in c1.meta["files"] if f["file"].startswith("Appendix")]
    assert skipped and skipped[0]["status"] == "skipped" and "excluded" in skipped[0]["reason"]
    assert c1.grid.as_dict() == ref.grid.as_dict() and c1.meta["n_duplicate_readings"] == 0
    # a second copy of a file under another name: its readings are read once
    shutil.copy(trav / "af_trav.csv", trav / "af_trav_copy.csv")
    D.build_dataset(["Providence, RI"], tmp_path / "o2", traverses_dir=tmp_path / "trav", **kw)
    c2 = D.load_city(tmp_path / "o2" / "cities" / "providence_ri")
    assert c2.meta["n_duplicate_readings"] == len(af)
    assert any("read once" in w for w in c2.meta["warnings"])
    a = ref.table.sort_values(["run_id", "cell"]).reset_index(drop=True)
    b = c2.table.sort_values(["run_id", "cell"]).reset_index(drop=True)
    assert a["cell"].tolist() == b["cell"].tolist() and (a["n_readings"] == b["n_readings"]).all()
    assert np.allclose(a["Ta"], b["Ta"]) and np.allclose(a["dTa"], b["dTa"])
    # read with the appendix: its copies of the afternoon readings are dropped, its extra sensor kept
    D.build_dataset(["Providence, RI"], tmp_path / "o3", traverses_dir=tmp_path / "trav", skip_files=(), **kw)
    c3 = D.load_city(tmp_path / "o3" / "cities" / "providence_ri")
    assert c3.meta["n_duplicate_readings"] == 2 * len(af)
    run = next(r for r in c3.meta["runs"] if r["run"] == "af")
    assert run["vehicles"] == 4


class _Resp:
    def __init__(self, body=b""):
        self.status, self._b = 200, body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _ok_opener(req, timeout=None):
    return _Resp(b'{"token": "st=x&sig=y"}' if "/sas/" in req.full_url else b"")


def test_a_blocked_or_failing_landsat_host_is_never_cached_as_a_finished_city(built, tmp_path, monkeypatch):
    """R9/R20: STAC answers but every COG read fails (proxy block, 5xx): the build names the blocked host, or
    records the city failed and retries it; it never writes status 'ok' with NaN LST."""
    import urllib.error

    from sparc.data.collect import landsat as L

    root, _, _ = built
    scene = {"id": "LC08_L2SP_012031_20200724_02_T1", "datetime": "2020-07-24T15:30:00Z", "cloud_cover": 5.0,
             "platform": "landsat-8",
             "assets": {"qa_pixel": {"href": "https://landsateuwest.blob.core.windows.net/landsat-c2/x_QA.TIF?sig=1"},
                        "lwir11": {"href": "https://landsateuwest.blob.core.windows.net/landsat-c2/x_ST.TIF?sig=1"}}}
    monkeypatch.setattr(L, "search_pc", lambda b, d0, d1: [dict(scene)])
    reads = []

    def blocked(href, bbox):
        reads.append(href)
        raise urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))

    monkeypatch.setattr(L, "read_window", blocked)
    kw = dict(traverses_dir=root / "traverses", features=synthetic_features, era5=False, probe_opener=_ok_opener)
    with pytest.raises(D.NetworkBlocked) as exc:
        D.build_dataset(["Providence, RI"], tmp_path / "o", **kw)
    assert exc.value.host == "landsateuwest.blob.core.windows.net" and reads
    assert D.city_status(tmp_path / "o" / "cities" / "providence_ri") is None

    def unavailable(href, bbox):
        reads.append(href)
        raise urllib.error.HTTPError(href, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(L, "read_window", unavailable)
    s = D.build_dataset(["Providence, RI"], tmp_path / "o", **kw)
    rec = s["cities"]["Providence, RI"]
    assert rec["status"] == "failed" and "retried" in rec["error"]
    n = len(reads)
    D.build_dataset(["Providence, RI"], tmp_path / "o", **kw)                   # the rerun tries again
    assert len(reads) > n


def test_the_landsat_probe_signs_and_lists_the_blob_container():
    import urllib.error

    seen = []

    def opener(req, timeout=None):
        seen.append(req.full_url)
        if "blob.core.windows.net" in req.full_url:
            raise urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))
        return _ok_opener(req, timeout)

    with pytest.raises(D.NetworkBlocked) as exc:
        D.check_hosts(["landsat"], opener=opener)
    assert exc.value.host == "landsateuwest.blob.core.windows.net"
    assert seen[0].startswith("https://planetarycomputer.microsoft.com/api/stac/v1/")
    assert any("/api/sas/v1/token/landsateuwest/landsat-c2" in u for u in seen)
    assert any("restype=container&comp=list" in u and "sig=y" in u for u in seen)
    assert D.check_hosts(["landsat"], opener=_ok_opener) == {"landsat": "ok"}


_LISTING = json.dumps({"data": [{"id": "f1", "attributes": {"kind": "file", "name": "Read Me.txt", "size": 309}},
                                {"id": "f2", "attributes": {"kind": "file", "name": "All Data.zip", "size": 9e7}}],
                       "links": {}}).encode()
_ORACLE = "https://idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com/cos-prod-osf-1-files-ash/abc"


def _osf_opener(req, timeout=None, oracle_blocked=False):
    """OSF as it answers: the API listing, a 302 from files.osf.io to the Oracle object store, the file there."""
    u = req.full_url
    if u.startswith("https://api.osf.io/v2/nodes/"):
        return _Resp(_LISTING)
    if u.startswith("https://files.osf.io/"):
        raise urllib.error.HTTPError(u, 302, "Found", {"Location": _ORACLE}, None)
    if u.startswith(_ORACLE) and oracle_blocked:
        raise urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))
    return _Resp(b"x")


def test_osf_downloads_redirect_to_oracle_storage_and_the_probe_names_that_host():
    """R10: files.osf.io redirects every download to OSF's Oracle Cloud object store; with only the OSF hosts
    allowed the probe must name the storage host, which is also in the hosts list."""
    assert D.check_hosts(["osf"], opener=_osf_opener) == {"osf": "ok"}
    with pytest.raises(D.NetworkBlocked) as exc:
        D.check_hosts(["osf"], opener=lambda r, timeout=None: _osf_opener(r, timeout, oracle_blocked=True))
    host = "idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com"
    assert exc.value.host == host and host in str(exc.value)
    assert host in D.required_host_names() and host in D.HOSTS["osf"][1]


def test_an_osf_http_error_fails_one_city_and_a_blocked_files_host_is_named(built, tmp_path, monkeypatch):
    """R21: a 404 for one city's node is that city's failure, not a blocked api.osf.io; a connection refused at
    files.osf.io names files.osf.io."""
    from sparc.data.collect import capa

    root, _, _ = built

    def scan(node, top_level_filter=None, **kw):
        if node == "tdsy7":
            raise urllib.error.HTTPError("https://api.osf.io/v2/nodes/tdsy7/files/osfstorage/", 404, "NOT FOUND",
                                         {}, None)
        return [{"name": "af_trav.csv", "id": "9", "path": "/af_trav.csv", "size": None}]

    monkeypatch.setattr(capa, "_scan_storage", scan)
    src = (root / "traverses" / "providence_ri" / "af_trav.csv").read_bytes()
    monkeypatch.setattr(capa, "_http_get", lambda url: src)
    monkeypatch.setattr(capa, "_fetch_file", lambda url, part, size=None: part.write_bytes(src), raising=False)
    kw = dict(features=synthetic_features, lst=False, era5=False, probe_opener=_osf_opener)
    s = D.build_dataset(["Boston, MA", "Providence, RI"], tmp_path / "o", **kw)
    assert s["cities"]["Boston, MA"]["status"] == "failed" and "HTTP 404" in s["cities"]["Boston, MA"]["error"]
    assert s["cities"]["Providence, RI"]["status"] == "ok"

    def refused(url):
        exc = urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))
        exc.sparc_url = url
        raise exc

    monkeypatch.setattr(capa, "_http_get", refused)
    monkeypatch.setattr(capa, "_fetch_file", lambda url, part, size=None: refused(url), raising=False)
    with pytest.raises(D.NetworkBlocked) as exc:
        D.build_dataset(["Providence, RI"], tmp_path / "o2", **kw)
    assert exc.value.host == "files.osf.io"


def write_2019_campaign(folder, centre=(-116.20, 43.61), day="2019-07-24", tz="America/Boise"):
    """The 2019 NIHHIS layout (tdsy7 Boise/Boston): one table per run named AM/PM/Eve, fields lat, long, time_n
    (UTC HHMMSS, no date) and T_F; PM is the 15–16 h afternoon run, Eve the 19–20 h one."""
    from pyproj import Transformer

    folder.mkdir(parents=True, exist_ok=True)
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
    x0, y0 = to_xy.transform(*centre)
    segs = _routes(x0, y0)
    xs = np.concatenate([s[0] for s in segs])
    ys = np.concatenate([s[1] for s in segs])
    lon, lat = to_ll.transform(xs, ys)
    for name, local_start, base in (("AM", "06:00:00", 20.0), ("PM", "15:00:00", 33.0), ("Eve", "19:00:00", 31.0)):
        when = pd.Timestamp(f"{day} {local_start}") + pd.to_timedelta(np.arange(xs.size) % 3500, unit="s")
        utc = when.tz_localize(tz).tz_convert("UTC")
        T = base + _truth(xs, ys, x0, y0) - 28.0
        pd.DataFrame({"lat": lat, "long": lon, "time_n": utc.strftime("%H%M%S").astype(int), "speed": 10.0,
                      "T_F": T * 9 / 5 + 32, "case": 1}).to_csv(folder / f"{name}_TraversePoints_Boise.csv", index=False)


def test_the_2019_nihhis_layout_is_dated_from_the_campaign_and_says_what_it_needs(tmp_path):
    """R11: tdsy7's Boise/Boston tables carry time_n (UTC HHMMSS, no date); without a campaign date the build
    must say so (it used to fail with an empty reason), and with one the runs land at 6, 15 and 19 h local."""
    write_2019_campaign(tmp_path / "trav" / "boise_id")
    kw = dict(traverses_dir=tmp_path / "trav", features=synthetic_features, lst=False, era5=False,
              probe_opener=_no_network)
    s = D.build_dataset(["Boise, ID"], tmp_path / "o", **kw)
    err = s["cities"]["Boise, ID"]["error"]
    assert s["cities"]["Boise, ID"]["status"] == "failed"
    assert "time_n" in err and "--campaign-date" in err and "usable timestamp" in err
    s = D.build_dataset(["Boise, ID"], tmp_path / "o", campaign_dates={"Boise, ID": "2019-07-24"}, **kw)
    assert s["cities"]["Boise, ID"]["status"] == "ok", s
    c = D.load_city(tmp_path / "o" / "cities" / "boise_id")
    starts = {r["run"]: r["start"] for r in c.meta["runs"]}
    assert set(starts) == {"am", "af", "pm"}
    assert starts["am"].startswith("2019-07-24 06:0") and starts["af"].startswith("2019-07-24 15:0")
    assert starts["pm"].startswith("2019-07-24 19:0")                       # 01:00 UTC of the next day
    assert c.meta["campaign_date"] == "2019-07-24" and "time_n" in c.meta["campaign_date_source"]
    assert all("time_n" in f["clock"] for f in c.meta["files"] if f["status"] == "read")


def test_no_run_reason_is_never_empty():
    spec = D.CitySpec(name="X, ID", slug="x_id", tz="America/Boise")
    msg = D._no_run_reason([], [{"file": "a.csv", "status": "skipped", "reason": "no time column"}], spec)
    assert "no file could be read" in msg and "a.csv" in msg
    msg = D._no_run_reason([{"run_id": "r1", "error": "no cells"}], [], spec)
    assert msg.endswith("r1: no cells")


def test_catalog_cities_without_traverses_are_reported_not_downloaded(built, tmp_path):
    """R12: New York's tdsy7 folder does not exist; the city is reported unavailable, nothing is fetched."""
    root, _, _ = built
    s = D.build_dataset(["New York, NY", "Providence, RI"], tmp_path / "o", traverses_dir=root / "traverses",
                        features=synthetic_features, lst=False, era5=False, probe_opener=_no_network,
                        download=lambda spec, dest: pytest.fail("no download for an unavailable city"))
    assert s["cities"]["New York, NY"]["status"] == "unavailable" and "no traverse points" in s["cities"]["New York, NY"]["error"]
    assert s["cities"]["Providence, RI"]["status"] == "ok"


def test_the_download_record_is_not_read_as_a_traverse_file(tmp_path):
    from sparc.core.identify.traverses import scan_traverses

    folder = tmp_path / "raw"
    write_campaign(folder, CENTRES["Providence, RI"])
    (folder / "_download.json").write_text(json.dumps({"node": "x", "files": [], "complete": True}))
    (folder / "af_trav.csv.part").write_text("partial")
    _, report = scan_traverses(folder)
    assert not any(r["file"].startswith("_download") or r["file"].endswith(".part") for r in report)


def _to_utc_clock(folder, hours):
    """Rewrite a write_campaign folder's clocks as UTC (``hours`` ahead) under a plain ``datetime`` column."""
    for f in folder.glob("*_trav.csv"):
        df = pd.read_csv(f)
        col = "dttm_lc" if "dttm_lc" in df else "datetime"
        df["datetime"] = (pd.to_datetime(df[col]) + pd.Timedelta(hours=hours)).dt.strftime("%Y-%m-%d %H:%M:%S")
        df.drop(columns=[c for c in ("dttm_lc",) if c in df]).to_csv(f, index=False)


@pytest.mark.parametrize("city,centre,tz_hours", [("Providence, RI", (-71.41, 41.82), 4),
                                                   ("Los Angeles, CA", (-118.24, 34.05), 7)])
def test_utc_clocks_are_detected_from_the_protocol_windows(tmp_path, city, centre, tz_hours):
    """R13: the old check only caught Eastern/Central UTC stamps; a Pacific campaign stamped in UTC put the
    06–07 morning run at 13 h and was labelled the afternoon run."""
    folder = tmp_path / "trav" / D.slugify(city)
    write_campaign(folder, centre)
    _to_utc_clock(folder, tz_hours)
    s = D.build_dataset([city], tmp_path / "o", traverses_dir=tmp_path / "trav", features=synthetic_features,
                        lst=False, era5=False, probe_opener=_no_network)
    assert s["cities"][city]["status"] == "ok", s
    c = D.load_city(tmp_path / "o" / "cities" / D.slugify(city))
    starts = {r["run"]: r["start"] for r in c.meta["runs"]}
    assert set(starts) == {"am", "af"}
    assert starts["af"][11:13] == "15" and starts["am"][11:13] == "06"
    assert any("timestamps are UTC" in w for w in c.meta["warnings"])


def test_a_clock_outside_every_protocol_window_is_refused():
    pts = pd.DataFrame({"time": pd.date_range("2020-07-29 12:00", periods=600, freq="6s"), "clock": None})
    with pytest.raises(ValueError, match="protocol windows"):
        D.check_clock(pts, "America/New_York")
    local = pd.DataFrame({"time": pd.date_range("2020-07-29 15:00", periods=600, freq="6s"), "clock": None})
    out, note = D.check_clock(local, "America/New_York")
    assert note is None and out is local


def test_a_utc_only_capa_table_is_converted_with_its_own_tz_column(tmp_path):
    from sparc.core.identify.traverses import scan_traverses

    df = pd.DataFrame({"dttm_tc": pd.date_range("2020-07-29 19:00", periods=50, freq="10s").strftime("%Y-%m-%d %H:%M:%S"),
                       "tz": "America/New_York", "lat": 41.82, "lon": np.linspace(-71.42, -71.40, 50), "t_c": 30.0})
    df.to_csv(tmp_path / "af_trav.csv", index=False)
    pts, rep = scan_traverses(tmp_path)
    assert pts["time"].min() == pd.Timestamp("2020-07-29 15:00:00") and "America/New_York" in rep[0]["clock"]


def test_cli_utc_builds_every_city_in_one_call_and_never_from_another_citys_folder(built, tmp_path, monkeypatch):
    """R14/R23: with --utc each city used to be built alone, so a city without a subfolder got the whole
    --traverses-dir (another city's traverses) and dataset.json kept only the last city."""
    from sparc.core.cli import main

    root, _, _ = built
    trav = tmp_path / "trav"
    shutil.copytree(root / "traverses" / "providence_ri", trav / "providence_ri")
    _to_utc_clock(trav / "providence_ri", 4)
    calls = []
    orig = D.build_city

    def spy(spec, folder, *a, **k):
        calls.append((spec.slug, str(folder), k.get("utc_to")))
        return orig(spec, folder, *a, **{**k, "features": synthetic_features})

    monkeypatch.setattr(D, "build_city", spy)
    monkeypatch.setattr(D, "run_probe", lambda pr, opener=None, timeout=10.0: ("host", None))
    monkeypatch.setattr(D, "_download_traverses", lambda spec, dest: (_ for _ in ()).throw(RuntimeError("offline")))
    rc = main(["lstair", "dataset", "--cities", "Providence, RI", "Brockton, MA", "--out", str(tmp_path / "o"),
               "--traverses-dir", str(trav), "--utc", "--no-lst", "--no-era5"])
    assert rc == 0 and calls == [("providence_ri", str(trav / "providence_ri"), "America/New_York")]
    summary = json.loads((tmp_path / "o" / "dataset.json").read_text())
    assert set(summary["cities"]) == {"Providence, RI", "Brockton, MA"}
    assert summary["cities"]["Providence, RI"]["status"] == "ok"
    assert summary["cities"]["Brockton, MA"]["status"] == "failed" and "offline" in summary["cities"]["Brockton, MA"]["error"]
    c = D.load_city(tmp_path / "o" / "cities" / "providence_ri")
    assert {r["run"]: r["start"][11:13] for r in c.meta["runs"]} == {"af": "15", "am": "06"}


def test_a_city_cut_off_mid_build_resumes_from_the_sources_already_done(built, tmp_path):
    """R15: features, LST and ERA5 are kept per city as each finishes, so a restart (or an out-of-memory kill)
    during a later source does not throw the finished ones away."""
    root, _, _ = built
    calls = {"features": 0, "lst": 0, "era5": 0}

    def feats(ctx, d):
        calls["features"] += 1
        return synthetic_features(ctx, d)

    def era5(*a):
        calls["era5"] += 1
        return fake_era5(*a)

    def lst_killed(ctx, d, study):
        calls["lst"] += 1
        raise MemoryError("killed")

    def lst_ok(ctx, d, study):
        calls["lst"] += 1
        return synthetic_lst(ctx, d, study)

    kw = dict(traverses_dir=root / "traverses", probe_opener=_no_network)
    s = D.build_dataset(["Providence, RI"], tmp_path / "o", features=feats, lst=lst_killed, era5=era5, **kw)
    assert s["cities"]["Providence, RI"]["status"] == "failed" and calls == {"features": 1, "lst": 1, "era5": 0}
    part = tmp_path / "o" / "cities" / "providence_ri" / "partial"
    assert (part / "features.json").exists() and (part / "features.npz").exists()
    s = D.build_dataset(["Providence, RI"], tmp_path / "o", features=feats, lst=lst_ok, era5=era5, **kw)
    assert s["cities"]["Providence, RI"]["status"] == "ok" and calls == {"features": 1, "lst": 2, "era5": 2}
    assert not part.exists()
    a = D.load_city(tmp_path / "o" / "cities" / "providence_ri").table
    b = D.load_city(built[1] / "cities" / "providence_ri").table
    assert np.allclose(a["canopy_f300"], b["canopy_f300"]) and np.allclose(a["lst"], b["lst"], equal_nan=True)
    m = D.load_city(tmp_path / "o" / "cities" / "providence_ri").meta
    assert not any("large grid" in w for w in m["warnings"])
