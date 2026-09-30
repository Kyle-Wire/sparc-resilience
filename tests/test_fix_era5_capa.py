"""Regression tests for the G9 data-layer fixes (roadmap Appendix A22–A24).

All network access is mocked: ``urllib.request.urlopen`` is replaced for
every test (api.osf.io and Open-Meteo must never be contacted).

A24 — ERA5 boundary windows are requested in LOCAL time (IANA timezone or
      "auto"), follow the CAPA traverse protocol, and add ssrd / blh / u10 / v10.
A23 — every pilot city carries ``timezone`` and ``campaign_date_source``.
A22 — raw CAPA traverse discovery + parsing.
"""
from __future__ import annotations

import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sparc.data.collect import era5
from sparc.data.collect import capa
from sparc.data.collect.assembler_multicity import CANONICAL_SCHEMA, LAYER_COLUMN_MAP

ROOT = Path(__file__).resolve().parents[1]
NEW_ERA5_VARS = ("ssrd", "blh", "u10", "v10")
WINDOWS = ("morning", "midday", "evening")


class _FakeResponse:
    def __init__(self, payload):
        self._data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*a, **k):
        raise RuntimeError("network access is disabled in tests")
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


def _url(req):
    return req.full_url if hasattr(req, "full_url") else str(req)


def _query(req) -> dict:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(_url(req)).query))


def _hourly_payload(day: str, hours=range(24), ws=None, wd=None, extended=True):
    hours = list(hours)
    hourly = {
        "time": [f"{day}T{h:02d}:00" for h in hours],
        "temperature_2m": [float(h) for h in hours],          # value == local hour
        "relativehumidity_2m": [50.0 + h for h in hours],
        "windspeed_10m": ws if ws is not None else [2.0 for _ in hours],
        "winddirection_10m": wd if wd is not None else [180.0 for _ in hours],
    }
    if extended:
        hourly["shortwave_radiation"] = [100.0 * h for h in hours]
        hourly["boundary_layer_height"] = [1000.0 + h for h in hours]
    return {"hourly": hourly}


# ---------------------------------------------------------------------------
# ERA5 (A24)
# ---------------------------------------------------------------------------


class TestEra5LocalWindows:
    PHILLY = (-75.28, 39.86, -74.96, 40.14)

    def test_requests_local_timezone_and_averages_protocol_hours(self, monkeypatch):
        seen = []

        def fake(req, timeout=None):
            seen.append(_query(req))
            return _FakeResponse(_hourly_payload("2022-07-20"))

        monkeypatch.setattr(urllib.request, "urlopen", fake)
        _, _, data = era5.download_era5_boundary(
            self.PHILLY, date(2022, 7, 20), timezone="America/New_York")
        assert seen and all(q["timezone"] == "America/New_York" for q in seen)
        assert all(q["start_date"] == q["end_date"] == "2022-07-20" for q in seen)
        assert all("shortwave_radiation" in q["hourly"] and
                   "boundary_layer_height" in q["hourly"] for q in seen)
        point = next(iter(data.values()))
        # t2m == local hour → window mean == mean of the protocol hours
        assert point["morning"]["t2m"] == pytest.approx(6.5)
        assert point["midday"]["t2m"] == pytest.approx(15.5)
        assert point["evening"]["t2m"] == pytest.approx(19.5)
        assert point["evening"]["ssrd"] == pytest.approx(1950.0)
        assert point["evening"]["blh"] == pytest.approx(1019.5)
        for w in WINDOWS:
            assert set(point[w]) == set(era5.ERA5_BOUNDARY_VARIABLES)

    def test_no_longitude_utc_arithmetic_default_auto(self, monkeypatch):
        seen = []

        def fake(req, timeout=None):
            seen.append(_query(req))
            return _FakeResponse(_hourly_payload("2022-07-20"))

        monkeypatch.setattr(urllib.request, "urlopen", fake)
        _, _, data = era5.download_era5_boundary(self.PHILLY, date(2022, 7, 20))
        assert all(q["timezone"] == "auto" for q in seen)
        # old code: evening = (20 − round(−75/15)) % 24 = 01 UTC of the SAME UTC
        # date = the previous local evening.  Now the evening is local 19–20 h.
        assert next(iter(data.values()))["evening"]["t2m"] == pytest.approx(19.5)

    def test_custom_windows_and_afternoon_alias(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=None: _FakeResponse(_hourly_payload("2022-07-20")))
        _, _, data = era5.download_era5_boundary(
            self.PHILLY, date(2022, 7, 20), timezone="America/New_York",
            windows={"afternoon": (13,), "evening": (21, 22, 23)})
        p = next(iter(data.values()))
        assert p["morning"]["t2m"] == pytest.approx(6.5)       # default kept
        assert p["midday"]["t2m"] == pytest.approx(13.0)
        assert p["evening"]["t2m"] == pytest.approx(22.0)
        with pytest.raises(ValueError):
            era5.download_era5_boundary(self.PHILLY, date(2022, 7, 20), windows={"night": (1,)})

    def test_default_windows_follow_capa_protocol(self):
        assert era5.DEFAULT_BOUNDARY_WINDOWS == {
            "morning": (6, 7), "midday": (15, 16), "evening": (19, 20)}

    def test_dst_day_uses_time_array(self, monkeypatch):
        # Spring-forward day: local 02:00 does not exist → 23 hourly entries.
        hours = [h for h in range(24) if h != 2]
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: _FakeResponse(_hourly_payload("2021-03-14", hours)))
        p = era5._fetch_boundary_point(-75.0, 40.0, date(2021, 3, 14), timezone="America/New_York")
        assert p["morning"]["t2m"] == pytest.approx(6.5)
        assert p["evening"]["t2m"] == pytest.approx(19.5)

    def test_wind_components_and_vector_mean_direction(self, monkeypatch):
        ws = [3.0] * 24
        wd = [0.0] * 24
        wd[6], wd[7] = 350.0, 10.0          # morning: northerly either side of 0°
        wd[15] = wd[16] = 90.0              # midday: easterly wind
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: _FakeResponse(_hourly_payload("2022-07-20", ws=ws, wd=wd)))
        p = era5._fetch_boundary_point(-75.0, 40.0, date(2022, 7, 20), timezone="America/New_York")
        # from the east: u = −ws·sin(90°) = −3, v = −ws·cos(90°) = 0
        assert p["midday"]["u10"] == pytest.approx(-3.0)
        assert p["midday"]["v10"] == pytest.approx(0.0, abs=1e-9)
        assert p["midday"]["winddir"] == pytest.approx(90.0)
        # circular mean of 350° and 10° is 0° (arithmetic mean would say 180°)
        wd_m = p["morning"]["winddir"]
        assert min(wd_m, 360.0 - wd_m) == pytest.approx(0.0, abs=1e-6)
        assert p["morning"]["v10"] == pytest.approx(-3.0 * math.cos(math.radians(10.0)))
        assert p["morning"]["windspeed"] == pytest.approx(3.0)

    def test_extended_variables_rejected_falls_back_to_base(self, monkeypatch):
        calls = []

        def fake(req, timeout=None):
            q = _query(req)
            calls.append(q["hourly"])
            if "boundary_layer_height" in q["hourly"]:
                raise urllib.error.HTTPError(_url(req), 400, "Bad Request", {}, None)
            return _FakeResponse(_hourly_payload("2022-07-20", extended=False))

        monkeypatch.setattr(urllib.request, "urlopen", fake)
        p = era5._fetch_boundary_point(-75.0, 40.0, date(2022, 7, 20), timezone="auto")
        assert len(calls) == 2
        assert p["morning"]["t2m"] == pytest.approx(6.5)
        assert math.isnan(p["morning"]["ssrd"]) and math.isnan(p["morning"]["blh"])

    def test_network_failure_gives_nan_window_dicts(self):
        p = era5._fetch_boundary_point(-75.0, 40.0, date(2022, 7, 20))
        for w in WINDOWS:
            assert all(math.isnan(v) for v in p[w].values())

    def test_assign_to_grid_flattens_all_variables(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point
        pts = gpd.GeoSeries([Point(-75.10, 40.00), Point(-75.05, 40.02)], crs="EPSG:4326")
        fishnet = gpd.GeoDataFrame(geometry=pts.to_crs("EPSG:32618"))
        grid = {}
        for lon in (-75.25, -75.0):
            for lat in (39.75, 40.25):
                grid[(lon, lat)] = {w: {v: 1.0 + i for i, v in enumerate(era5.ERA5_BOUNDARY_VARIABLES)}
                                    for w in WINDOWS}
        out = era5.assign_era5_boundary_to_grid(fishnet, [-75.25, -75.0], [39.75, 40.25], grid)
        for w in WINDOWS:
            for i, v in enumerate(era5.ERA5_BOUNDARY_VARIABLES):
                col = f"era5_{w}_{v}"
                assert col in out.columns
                np.testing.assert_allclose(out[col].to_numpy(), 1.0 + i)


class TestAssemblerSchema:
    def test_new_columns_in_schema_and_layer_map(self):
        for w in WINDOWS:
            for v in NEW_ERA5_VARS:
                col = f"era5_{w}_{v}"
                assert col in CANONICAL_SCHEMA
                assert col in LAYER_COLUMN_MAP["era5_boundary"]
        # appended after the labels → existing column positions unchanged
        assert CANONICAL_SCHEMA.index("era5_morning_ssrd") > CANONICAL_SCHEMA.index("diurnal_aat")

    def test_new_columns_opt_in_for_pilot_config(self):
        cfg = yaml.safe_load((ROOT / "configs" / "multicity_pilot.yml").read_text(encoding="utf-8"))
        fc = cfg["multicity"]["feature_columns"]
        listed = {c for cols in fc.values() for c in cols}
        assert not any(f"era5_{w}_{v}" in listed for w in WINDOWS for v in NEW_ERA5_VARS)

    def test_assemble_keeps_new_columns_by_default(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box
        from sparc.data.collect.assembler_multicity import assemble_city_geoparquet
        fishnet = gpd.GeoDataFrame(geometry=[box(0, 0, 30, 30), box(30, 0, 60, 30)],
                                   crs="EPSG:32618")
        era5_layer = pd.DataFrame({c: [1.0, 2.0] for c in LAYER_COLUMN_MAP["era5_boundary"]})
        gdf = assemble_city_geoparquet(fishnet, {"era5_boundary": era5_layer}, "x", "EPSG:32618",
                                       date(2022, 7, 20))
        np.testing.assert_allclose(gdf["era5_evening_v10"].to_numpy(), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Pilot config (A23)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cities():
    cfg = yaml.safe_load((ROOT / "configs" / "multicity_pilot.yml").read_text(encoding="utf-8"))
    return cfg["multicity"]["pilot_cities"]


class TestPilotConfigMetadata:
    def test_every_city_has_timezone_and_date_source(self, cities):
        import zoneinfo
        for c in cities:
            zoneinfo.ZoneInfo(c["timezone"])
            assert c["campaign_date_source"] in {"era5_matched", "filename_guess"}

    def test_date_source_consistent_with_overrides(self, cities):
        text = (ROOT / "configs" / "multicity_pilot.yml").read_text(encoding="utf-8")
        for c in cities:
            if not c.get("campaign_date_override"):
                assert c["campaign_date_source"] == "filename_guess", c["city_slug"]
        by_slug = {c["city_slug"]: c for c in cities}
        for slug in ("raleigh_nc", "chicago_il", "boston_ma", "las_vegas_nv", "iowa_city_ia"):
            assert by_slug[slug]["campaign_date_source"] == "era5_matched"
        for slug in ("scranton_pa", "omaha_ne", "san_francisco_ca"):
            assert by_slug[slug]["campaign_date_source"] == "filename_guess"
        assert "circular" in text.lower().split("multicity:")[0].lower()


# ---------------------------------------------------------------------------
# CAPA traverse points (A22)
# ---------------------------------------------------------------------------


class TestTraverseDiscovery:
    def test_find_traverse_sources(self):
        files = [
            {"name": "traverses_chw_philadelphia_072522.zip", "id": "a"},
            {"name": "rasters_chw_philadelphia_072522.zip", "id": "b"},
            {"name": "Philly_traverses_am.csv", "id": "c"},
            {"name": "Traverse_points.geojson", "id": "d"},
            {"name": "traverses_report.pdf", "id": "e"},
            {"name": "am_trav.shp", "id": "f"},
            "folder/traverse_pm.shp",
        ]
        got = capa.find_traverse_sources(files)
        assert [g["id"] if isinstance(g, dict) else g for g in got] == [
            "a", "c", "d", "folder/traverse_pm.shp"]


def _df(**cols):
    return pd.DataFrame(cols)


class TestParseTraverseTable:
    @pytest.mark.parametrize("frame", [
        _df(DateTime=["2022-07-25 06:30:00", "2022-07-25 15:10:00", "2022-07-25 19:40:00"],
            Latitude=[39.95, 39.96, 39.97], Longitude=[-75.16, -75.17, -75.18], T_F=[76.0, 92.0, 85.0]),
        _df(timestamp=["2022-07-25T06:30:00", "2022-07-25T15:10:00", "2022-07-25T19:40:00"],
            lat=[39.95, 39.96, 39.97], long=[-75.16, -75.17, -75.18], temp_f=[76.0, 92.0, 85.0]),
        _df(Time=["2022-07-25 06:30", "2022-07-25 15:10", "2022-07-25 19:40"],
            Y=[39.95, 39.96, 39.97], X=[-75.16, -75.17, -75.18], temperature=[76.0, 92.0, 85.0]),
    ])
    def test_varied_headers(self, frame):
        out = capa.parse_traverse_table(frame)
        assert list(out.columns) == capa.TRAVERSE_COLUMNS
        assert list(out["window"]) == ["morning", "midday", "evening"]
        np.testing.assert_allclose(out["temp_f"], [76.0, 92.0, 85.0])
        np.testing.assert_allclose(out["lat"], [39.95, 39.96, 39.97])
        np.testing.assert_allclose(out["lon"], [-75.16, -75.17, -75.18])
        assert pd.api.types.is_datetime64_any_dtype(out["time"])

    def test_celsius_column_converted_and_date_time_combined(self):
        frame = _df(**{"date": ["2022-07-25", "2022-07-25"], "time": ["06:05:00", "16:59:00"],
                       "lat": [39.9, 39.9], "lon": [-75.1, -75.1],
                       "Temperature (°C)": [25.0, 35.0]})
        out = capa.parse_traverse_table(frame)
        np.testing.assert_allclose(out["temp_f"], [77.0, 95.0])
        assert list(out["window"]) == ["morning", "midday"]
        assert out["time"].iloc[0] == pd.Timestamp("2022-07-25 06:05:00")

    def test_window_boundaries(self):
        frame = _df(datetime=["2022-07-25 09:59", "2022-07-25 10:00",
                              "2022-07-25 16:59", "2022-07-25 17:00"],
                    lat=[40.0] * 4, lon=[-75.0] * 4, t_f=[80.0] * 4)
        out = capa.parse_traverse_table(frame)
        assert list(out["window"]) == ["morning", "midday", "midday", "evening"]

    def test_utc_timestamps_converted_to_local(self):
        frame = _df(timestamp=["2022-07-25T10:30:00Z", "2022-07-25T23:30:00Z"],
                    lat=[40.0, 40.0], lon=[-75.0, -75.0], temp_f=[80.0, 85.0])
        out = capa.parse_traverse_table(frame, timezone="America/New_York")
        assert list(out["window"]) == ["morning", "evening"]     # 06:30 / 19:30 EDT

    def test_rows_missing_values_dropped_and_errors(self):
        frame = _df(datetime=["2022-07-25 06:00", "2022-07-25 06:01"],
                    lat=[40.0, None], lon=[-75.0, -75.0], t_f=[80.0, 81.0])
        assert len(capa.parse_traverse_table(frame)) == 1
        with pytest.raises(ValueError):
            capa.parse_traverse_table(_df(datetime=["2022-07-25 06:00"], lat=[40.0], lon=[-75.0]))
        with pytest.raises(ValueError):   # projected coordinates are not lat/lon
            capa.parse_traverse_table(_df(datetime=["2022-07-25 06:00"], y=[4.4e6],
                                          x=[4.9e5], t_f=[80.0]))


def _traverse_zip() -> bytes:
    am = _df(DateTime=["2022-07-25 06:10:00", "2022-07-25 06:40:00"],
             Latitude=[39.95, 39.96], Longitude=[-75.16, -75.17], T_F=[75.0, 76.0])
    pm = _df(timestamp=["2022-07-25 19:10:00"], lat=[39.97], lon=[-75.18], temp_c=[30.0])
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("pm_traverse.csv", pm.to_csv(index=False))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("traverses/am_traverse.csv", am.to_csv(index=False))
        z.writestr("traverses/nested_pm.zip", inner.getvalue())
        z.writestr("README.txt", "not a table")
    return outer.getvalue()


class TestTraverseZip:
    def test_in_memory_zip_with_nested_zip(self):
        out = capa.parse_traverse_bytes("traverses_chw_philadelphia_072522.zip", _traverse_zip())
        assert len(out) == 3
        assert sorted(out["window"]) == ["evening", "morning", "morning"]
        ev = out[out["window"] == "evening"].iloc[0]
        assert ev["temp_f"] == pytest.approx(86.0)
        assert set(out["source_file"]) == {"am_traverse.csv", "pm_traverse.csv"}

    def test_download_capa_traverses_with_mocked_osf(self, monkeypatch):
        listing = {"data": [
            {"id": "f1", "attributes": {"kind": "file",
                                        "name": "traverses_chw_philadelphia_072522.zip"}},
            {"id": "f2", "attributes": {"kind": "file",
                                        "name": "rasters_chw_philadelphia_072522.zip"}},
        ], "links": {"next": None}}
        payload = _traverse_zip()
        requested = []

        def fake(req, timeout=None):
            url = _url(req)
            requested.append(url)
            if "api.osf.io" in url:
                return _FakeResponse(listing)
            if url.endswith("/f1"):
                return _FakeResponse(payload)
            raise AssertionError(f"unexpected download {url}")

        monkeypatch.setattr(urllib.request, "urlopen", fake)
        out = capa.download_capa_traverses("abc12", timezone="America/New_York")
        assert len(out) == 3
        assert not any(u.endswith("/f2") for u in requested)

    def test_download_capa_traverses_never_raises(self):
        out = capa.download_capa_traverses("abc12")      # network blocked by fixture
        assert len(out) == 0
        assert list(out.columns) == capa.TRAVERSE_COLUMNS + ["source_file"]
