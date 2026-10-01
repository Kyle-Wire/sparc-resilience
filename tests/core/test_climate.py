"""Climate projections: CMIP6 point reader (fake zarr store, no network),
calendar decoding, land weighting and the projection × adaptation summary."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from sparc.core import climate as C

numcodecs = pytest.importorskip("numcodecs")


def test_decode_year_month_across_calendars():
    for cal, dpy in (("noleap", 365.0), ("360_day", 360.0), ("proleptic_gregorian", 365.2425)):
        months = np.arange(24)
        values = (months * dpy / 12.0) + 15.0                    # mid-month stamps from 2015-01-01
        y, m = C.decode_year_month(values, "days since 2015-01-01", cal)
        assert list(y[:13]) == [2015] * 12 + [2016]
        assert list(m[:13]) == list(range(1, 13)) + [1]
    y, m = C.decode_year_month(np.array([24 * (365 * 165 + 40 + 15)]), "hours since 1850-01-01 00:00:00", "standard")
    assert y[0] == 2015 and m[0] == 1


def _zarray(shape, chunks, dtype="<f4", compressor=True):
    return {"shape": list(shape), "chunks": list(chunks), "dtype": dtype, "order": "C", "fill_value": 1e20,
            "filters": None, "zarr_format": 2,
            "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0} if compressor else None}


def _put(objs, url, name, arr, chunks):
    codec = numcodecs.get_codec({"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0})
    for idx in np.ndindex(*[int(np.ceil(s / c)) for s, c in zip(arr.shape, chunks)]):
        block = np.full(chunks, 1e20, dtype=arr.dtype)            # zarr v2 pads edge chunks
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, arr.shape))
        block[tuple(slice(0, s.stop - s.start) for s in sl)] = arr[sl]
        objs[f"{url}{name}/{'.'.join(map(str, idx))}"] = codec.encode(np.ascontiguousarray(block).tobytes())


def _fake_store(objs, url, start_year, n_years, lat, lon, signal, calendar="noleap"):
    """Monthly tasmax on a small grid; ``signal(year, month)`` K added to 290."""
    nt = 12 * n_years
    t = np.arange(nt) * 365.0 / 12.0 + 15.0
    years, months = start_year + np.arange(nt) // 12, np.arange(nt) % 12 + 1
    field = np.empty((nt, lat.size, lon.size), dtype="<f4")
    for k in range(nt):
        field[k] = 290.0 + signal(years[k], months[k])
    meta = {
        "time/.zarray": _zarray([nt], [nt], "<f8"), "time/.zattrs": {"units": f"days since {start_year}-01-01",
                                                                      "calendar": calendar},
        "lat/.zarray": _zarray([lat.size], [lat.size], "<f8"), "lon/.zarray": _zarray([lon.size], [lon.size], "<f8"),
        "tasmax/.zarray": _zarray([nt, lat.size, lon.size], [120, lat.size, lon.size]),
    }
    objs[url + ".zmetadata"] = json.dumps({"metadata": meta}).encode()
    _put(objs, url, "time", t, [nt])
    _put(objs, url, "lat", lat.astype("<f8"), [lat.size])
    _put(objs, url, "lon", lon.astype("<f8"), [lon.size])
    _put(objs, url, "tasmax", field, [120, lat.size, lon.size])


def test_cmip6_change_factors_recover_planted_warming():
    objs: dict[str, bytes] = {}
    lat, lon = np.array([38.0, 40.0, 42.0, 44.0]), np.arange(280.0, 296.0, 2.0)
    root = "s3://cmip6-pds/CMIP6/"
    hist, fut = root + "CMIP/X/FAKE/historical/r1i1p1f1/Amon/tasmax/gn/v1/", root + "ScenarioMIP/X/FAKE/ssp245/r1i1p1f1/Amon/tasmax/gn/v1/"
    summer = lambda y, m: 5.0 if m in (6, 7, 8) else 0.0                                  # noqa: E731
    _fake_store(objs, C._store_url(hist), 1980, 35, lat, lon, summer)
    # +2 K in JJA from 2041, +4 K from 2081 (and +1 K outside summer, which must be ignored)
    _fake_store(objs, C._store_url(fut), 2015, 86, lat, lon,
                lambda y, m: summer(y, m) + ((2.0 if y >= 2041 else 0.0) + (2.0 if y >= 2081 else 0.0)
                                             if m in (6, 7, 8) else 1.0))
    cat = pd.DataFrame([{"activity_id": "CMIP", "institution_id": "X", "source_id": "FAKE", "experiment_id": e,
                         "member_id": "r1i1p1f1", "table_id": "Amon", "variable_id": "tasmax", "grid_label": "gn",
                         "zstore": z, "dcpp_init_year": None, "version": 1} for e, z in (("historical", hist), ("ssp245", fut))])
    buf = io.StringIO()
    cat.to_csv(buf, index=False)
    objs[C.CATALOG_URL] = buf.getvalue().encode()

    def fetch(url):
        return objs.get(url)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        df = C.cmip6_change_factors(41.8, -71.4, d, experiments=("ssp245",), fetch=fetch, max_workers=1)
    got = df.set_index("period")["delta_K"]
    assert got["2021-2040"] == pytest.approx(0.0, abs=1e-5)
    assert got["2041-2060"] == pytest.approx(2.0, abs=1e-5)
    assert got["2081-2100"] == pytest.approx(4.0, abs=1e-5)
    assert np.allclose(df["baseline_K"], 295.0)


def test_site_weights_prefer_land_cells():
    lat, lon = np.array([40.0, 42.0]), np.array([288.0, 290.0])
    w = {(i, j): v for i, j, v in C.site_weights(lat, lon, 41.0, -71.0)}
    assert sum(w.values()) == pytest.approx(1.0) and all(v == pytest.approx(0.25) for v in w.values())
    land = np.array([[1.0, 0.0], [1.0, 0.0]])                     # eastern column is ocean
    w = {(i, j): v for i, j, v in C.site_weights(lat, lon, 41.0, -71.0, land)}
    assert set(w) == {(0, 0), (1, 0)} and sum(w.values()) == pytest.approx(1.0)


def test_summarize_projections_thresholds_and_offsets():
    obs = np.array([86.0, 88.0, 90.0, 94.0])
    factors = pd.DataFrame({"experiment": ["ssp245"] * 3, "period": ["2041-2060"] * 3, "model": list("abc"),
                            "delta_K": [1.0, 2.0, 3.0]})
    adapt = {"trees": np.array([-1.0, -1.0, -1.0, -1.0])}
    out = C.summarize_projections(obs, factors, adapt, thresholds=[95], to_units=1.8)
    assert out["present"]["share_at_or_above"]["95.0"] == 0.0
    p = out["projections"][0]
    assert p["warming"]["median"] == pytest.approx(3.6) and p["n_models"] == 3
    none, trees = p["variants"]
    assert none["share_at_or_above"]["95.0"]["median"] == pytest.approx(0.25)    # 94 + 3.6 ≥ 95
    assert trees["share_at_or_above"]["95.0"]["median"] == pytest.approx(0.25)   # 94 + 3.6 − 1 = 96.6
    assert trees["offset_share_of_median_warming"] == pytest.approx(1.0 / 3.6)


def test_default_adaptation_picks_packages_and_largest_in_support_dose():
    from sparc.core.pipeline import default_adaptation

    sc = [{"name": "c+10", "frac_extrapolated": 0.05, "mean_realized": {"c": 10.0}},
          {"name": "c+30", "frac_extrapolated": 0.10, "mean_realized": {"c": 28.0}},
          {"name": "a+0.2", "frac_extrapolated": 0.90, "mean_realized": {"a": 0.2}},
          {"name": "pkg", "frac_extrapolated": 0.15, "mean_realized": {"c": 15.0, "i": -15.0}}]
    assert default_adaptation(sc) == ["pkg", "c+30"]
