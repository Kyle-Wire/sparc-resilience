"""Campaign forcing: ERA5 HDF5 range reader (fake files, no network), the
accumulation bookkeeping, station parsing and the meteorology helpers."""

from __future__ import annotations

import datetime as dt
import io
import json
import math

import numpy as np
import pytest

from sparc.core import forcing as F

h5py = pytest.importorskip("h5py")

LAT = np.array([42.5, 42.25, 42.0, 41.75, 41.5])          # ERA5 order: north → south
LON = np.array([288.0, 288.25, 288.5, 288.75, 289.0])
OCEAN_COL = 3                                               # 288.75 E is water in the fake land mask
UTC = dt.timezone.utc


def _h5(build) -> bytes:
    bio = io.BytesIO()
    with h5py.File(bio, "w") as h:
        h["latitude"] = LAT
        h["longitude"] = LON
        build(h)
    return bio.getvalue()


def _hours(t: dt.datetime) -> int:
    return int((t - F.ERA5_EPOCH).total_seconds() // 3600)


def _field(value: float) -> np.ndarray:
    f = np.full((LAT.size, LON.size), value, dtype="f4")
    f[:, OCEAN_COL] = value + 100.0                         # must get zero weight
    return f


def _analysis_file(var: str, per_hour) -> bytes:
    t0 = dt.datetime(2020, 7, 1, tzinfo=UTC)
    times = [t0 + dt.timedelta(hours=k) for k in range(31 * 24)]

    def build(h):
        h["time"] = np.array([_hours(t) for t in times], dtype="i4")
        h.create_dataset(var, data=np.stack([_field(per_hour(t)) for t in times]), chunks=(24, 5, 5),
                         compression="gzip")
    return _h5(build)


def _accum_file(var: str, w_m2: float) -> bytes:
    inits = [dt.datetime(2020, 7, 16, 6, tzinfo=UTC) + dt.timedelta(hours=12 * k) for k in range(32)]

    def build(h):
        h["forecast_initial_time"] = np.array([_hours(t) for t in inits], dtype="i4")
        h["forecast_hour"] = np.arange(1, 13, dtype="i4")
        data = np.zeros((len(inits), 12, LAT.size, LON.size), dtype="f4")
        for i, t in enumerate(inits):
            for s in range(12):
                valid = t + dt.timedelta(hours=s + 1)
                data[i, s] = _field(w_m2 * 3600.0 * (1.0 + 0.01 * valid.hour))   # J/m² over the hour ending at valid
        h.create_dataset(var, data=data, chunks=(1, 12, 5, 5), compression="gzip")
    return _h5(build)


def _fake_bucket(objects: dict[str, bytes], station_csv: str | None = None):
    def fetch(url):
        if url.startswith(F.ERA5 + "?"):
            import urllib.parse

            prefix = urllib.parse.parse_qs(url.split("?", 1)[1])["prefix"][0]
            keys = "".join(f"<Key>{k}</Key>" for k in objects if k.startswith(prefix))
            return f"<ListBucketResult>{keys}</ListBucketResult>".encode()
        if url.startswith(F.GLOBAL_HOURLY):
            return station_csv.encode() if station_csv else None
        return None

    def fetch_range(url, a, b):
        blob = objects[url[len(F.ERA5):]]
        return blob[a:b + 1], len(blob)

    return fetch, fetch_range


AN = "e5.oper.an.sfc/202007/e5.oper.an.sfc.{c}.ll025sc.2020070100_2020073123.nc"
ACC = "e5.oper.fc.sfc.accumu/202007/e5.oper.fc.sfc.accumu.{c}.ll025sc.2020071606_2020080106.nc"


def _lsm() -> bytes:
    land = np.ones((1, LAT.size, LON.size), dtype="f4")
    land[0, :, OCEAN_COL] = 0.0
    return _h5(lambda h: h.create_dataset("LSM", data=land))


def test_accumulation_init_picks_the_forecast_containing_the_hour():
    def chk(hour, init_day, init_hour, step):
        t = dt.datetime(2020, 7, 18, hour, tzinfo=UTC)
        init = F.accumulation_init(t)
        assert (init.day, init.hour) == (init_day, init_hour)
        assert (t - init) == dt.timedelta(hours=step)

    chk(20, 18, 18, 2)
    chk(19, 18, 18, 1)
    chk(18, 18, 6, 12)
    chk(7, 18, 6, 1)
    chk(6, 17, 18, 12)
    chk(0, 17, 18, 6)


def test_era5_site_reads_land_weighted_window_values():
    objs = {F.LSM_KEY: _lsm(),
            AN.format(c="128_167_2t"): _analysis_file("VAR_2T", lambda t: 280.0 + t.hour),
            ACC.format(c="128_169_ssrd"): _accum_file("SSRD", 700.0)}
    fetch, fetch_range = _fake_bucket(objs)
    times = F.window_utc("2020-07-18", (15, 16), "America/New_York")
    assert [t.hour for t in times] == [19, 20]                          # EDT = UTC − 4
    out = F.era5_site(41.826, -71.403, times, fetch=fetch, fetch_range=fetch_range,
                      analyses={"t2m": "128_167_2t"}, accumulations={"ssrd": "128_169_ssrd"})
    assert out["series"]["t2m"] == pytest.approx([299.0, 300.0])         # ocean column (+100) excluded
    assert out["values"]["t2m"] == pytest.approx(299.5)
    assert out["series"]["ssrd"] == pytest.approx([700.0 * 1.20])        # the hour ending 20Z only
    assert all(abs(lon - 288.75) > 1e-9 for _, lon, _ in out["cells"])


def test_station_obs_window_wind_vector_and_heat_index():
    csv = "\n".join([
        "STATION,DATE,LATITUDE,LONGITUDE,NAME,REPORT_TYPE,TMP,DEW,WND",
        '72507014765,2020-07-18T17:51:00,41.72,-71.43,"PROVIDENCE, RI US",FM-15,"+0350,5","+0100,5","270,5,N,0100,5"',
        '72507014765,2020-07-18T18:51:00,41.72,-71.43,"PROVIDENCE, RI US",FM-15,"+0300,5","+0200,5","180,5,N,0050,5"',
        '72507014765,2020-07-18T19:20:00,41.72,-71.43,"PROVIDENCE, RI US",FM-16,"+9999,9","+9999,9","999,9,V,0030,5"',
        '72507014765,2020-07-18T19:51:00,41.72,-71.43,"PROVIDENCE, RI US",FM-15,"+0290,5","+0200,5","180,5,N,0070,5"',
        '72507014765,2020-07-18T20:51:00,41.72,-71.43,"PROVIDENCE, RI US",FM-15,"+0250,5","+0100,5","090,5,N,0100,5"',
    ])
    fetch, _ = _fake_bucket({}, csv)
    st = F.station_obs("72507014765", dt.datetime(2020, 7, 18, 19, tzinfo=UTC), dt.datetime(2020, 7, 18, 20, tzinfo=UTC),
                       fetch=fetch)
    assert st["n_reports"] == 3                                          # 18:51–19:51 (± 30 min pad)
    assert st["t_C"] == pytest.approx(29.5) and st["td_C"] == pytest.approx(20.0)
    assert st["wind_from_deg"] == pytest.approx(180.0)                   # variable-direction report has no vector
    assert st["v"] == pytest.approx(6.0) and st["u"] == pytest.approx(0.0, abs=1e-9)
    assert st["wind_speed"] == pytest.approx((5.0 + 3.0 + 7.0) / 3)


def test_meteorology_helpers():
    assert F.heat_index_F(90.0, 60.0) == pytest.approx(100.0, abs=1.0)   # NWS table
    assert F.heat_index_F(96.0, 50.0) == pytest.approx(108.0, abs=1.0)
    assert F.heat_index_F(70.0, 50.0) < 71.0                             # simple formula below 80 °F
    assert F.relative_humidity(30.0, 30.0) == pytest.approx(100.0)
    assert F.wind_from(0.0, -5.0) == pytest.approx(0.0)                  # blowing south = from north
    assert F.wind_from(-3.0, 0.0) == pytest.approx(90.0)                 # blowing west = from east
    jan = F.window_utc("2020-01-18", (15, 16), "America/New_York")
    assert [t.hour for t in jan] == [20, 21]                             # EST = UTC − 5


def test_campaign_forcing_and_config_round_trip(tmp_path):
    values = {"128_167_2t": ("VAR_2T", 303.15), "128_168_2d": ("VAR_2D", 293.15), "128_165_10u": ("VAR_10U", 3.0),
              "128_166_10v": ("VAR_10V", 4.0), "128_159_blh": ("BLH", 1200.0), "128_164_tcc": ("TCC", 0.1),
              "128_235_skt": ("SKT", 305.0)}
    objs = {F.LSM_KEY: _lsm()}
    for code, (var, v) in values.items():
        objs[AN.format(c=code)] = _analysis_file(var, lambda t, v=v: v)
    for code, var, w in (("128_169_ssrd", "SSRD", 700.0), ("228_129_ssrdc", "SSRDC", 760.0), ("128_175_strd", "STRD", 400.0)):
        objs[ACC.format(c=code)] = _accum_file(var, w / 1.20)                # 20Z hour → exactly w
    fetch, fetch_range = _fake_bucket(objs)
    res = F.campaign_forcing(41.826, -71.403, "2020-07-18", (15, 16), "America/New_York",
                             fetch=fetch, fetch_range=fetch_range)
    e = res["era5"]
    assert e["t2m_F"] == pytest.approx(86.0, abs=1e-3)
    assert e["wind_speed"] == pytest.approx(5.0) and e["wind_from_deg"] == pytest.approx(216.87, abs=0.01)
    assert e["sw_down"] == pytest.approx(700.0) and e["clear_sky_index"] == pytest.approx(700.0 / 760.0)
    assert e["lw_net"] == pytest.approx(400.0 - 0.95 * 5.670374419e-8 * 305.0 ** 4)
    assert res["wind_source"] == "era5" and res["physics"]["wind"] == [3.0, 4.0]
    assert res["physics"]["window"] == "day"

    path = tmp_path / "forcing.json"
    path.write_text(json.dumps(res, default=str))
    from sparc.core.config import core_config_from_dict
    from sparc.core.synthetic import synthetic_city_config

    raw = synthetic_city_config()
    raw["physics"]["forcing"] = "forcing.json"
    cfg = core_config_from_dict(raw, base_dir=tmp_path)
    ph = cfg.raw["physics"]
    assert ph["sw_down"] == pytest.approx(700.0, abs=0.1) and ph["wind"] == [3.0, 4.0]
    assert ph["forcing_info"]["file"] == "forcing.json" and ph["forcing_info"]["date"] == "2020-07-18"
    assert math.isclose(ph["lw_net"], round(e["lw_net"], 1))
