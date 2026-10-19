"""Landsat Collection 2 Level-2 fixes (LST pilot L1): georeferencing, QA_PIXEL masks, campaign-matched scene
selection and the ST_B10 scale.  No network: synthetic georeferenced arrays and a mocked STAC search / reader."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_bounds as tr_from_bounds  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.transform import rowcol  # noqa: E402

from sparc.data.collect import landsat as L  # noqa: E402

UTM = "EPSG:32619"                     # Providence's UTM zone
X0, Y1 = 290_000.0, 4_640_000.0        # upper-left corner of the synthetic scene window (m)


def _scene_raster(n=200, res=30.0, x0=X0, y1=Y1):
    """A 6 km scene window in UTM whose value is the easting in km (a field with a known location)."""
    T = from_origin(x0, y1, res, res)
    cols = np.arange(n) + 0.5
    east_km = (x0 + cols * res) / 1000.0
    return np.tile(east_km, (n, 1)), T


def _fishnet(xs, ys, crs=UTM, half=15.0):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return gpd.GeoDataFrame({"cell": np.arange(len(xs))},
                            geometry=[box(x - half, y - half, x + half, y + half) for x, y in zip(xs, ys)], crs=crs)


# --------------------------------------------------------------------------- #
# (a) georeferencing                                                           #
# --------------------------------------------------------------------------- #
def test_legacy_bounds_assumption_misplaces_pixels_and_the_georeferenced_path_does_not():
    arr, T = _scene_raster()
    # a small fishnet in the middle of the scene window (the window is larger than the fishnet)
    xs = X0 + np.array([2000.0, 2500.0, 3100.0, 4000.0]) + 15.0
    ys = Y1 - np.array([2000.0, 2600.0, 3000.0, 3500.0]) - 15.0
    fish = _fishnet(xs, ys)
    truth = xs / 1000.0
    # what the old code did: stretch the array over the fishnet's lon/lat bounds
    f4326 = fish.to_crs("EPSG:4326")
    cent = fish.geometry.centroid.to_frame("geometry").set_crs(fish.crs).to_crs("EPSG:4326").geometry
    T_old = tr_from_bounds(*f4326.total_bounds, arr.shape[1], arr.shape[0])
    r, c = rowcol(T_old, cent.x.values, cent.y.values)
    old = arr[np.clip(np.asarray(r), 0, arr.shape[0] - 1), np.clip(np.asarray(c), 0, arr.shape[1] - 1)]
    assert np.max(np.abs(old - truth)) > 1.0                      # off by more than a kilometre
    new = L._assign_arrays_to_fishnet(fish, {"lst": arr, "_georef": {"transform": T, "crs": UTM}})
    assert np.max(np.abs(new["lst"].to_numpy() - truth)) <= 0.0151   # within half a 30 m pixel


def test_cells_outside_the_raster_are_nan_not_edge_values():
    arr, T = _scene_raster(n=50)
    xs = np.array([X0 + 100.0, X0 - 500.0, X0 + 50 * 30.0 + 400.0])
    ys = np.array([Y1 - 100.0, Y1 - 100.0, Y1 - 100.0])
    out = L.sample_raster(arr, T, xs, ys)
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])


def test_fishnet_in_another_crs_is_sampled_in_the_raster_crs():
    arr, T = _scene_raster()
    from pyproj import Transformer

    xs_u = X0 + np.array([1015.0, 3015.0])
    ys_u = Y1 - np.array([1015.0, 4015.0])
    x3857, y3857 = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True).transform(xs_u, ys_u)
    fish = _fishnet(np.asarray(x3857), np.asarray(y3857), crs="EPSG:3857", half=20.0)
    out = L._assign_arrays_to_fishnet(fish, {"v": arr}, georef={"transform": T, "crs": UTM})
    assert np.allclose(out["v"].to_numpy(), xs_u / 1000.0, atol=0.0151)


def test_arrays_without_georeference_are_refused():
    arr, _ = _scene_raster(n=10)
    fish = _fishnet([X0 + 15.0], [Y1 - 15.0])
    with pytest.raises(ValueError, match="georeference"):
        L._assign_arrays_to_fishnet(fish, {"lst": arr})


def test_reproject_onto_a_shifted_target_grid_keeps_a_linear_field_exact():
    arr, T = _scene_raster()
    tgt = {"transform": from_origin(X0 + 607.0, Y1 - 593.0, 30.0, 30.0), "crs": UTM, "shape": (60, 70)}
    out = L.reproject_array(arr, T, UTM, tgt, "bilinear")
    cols = np.arange(70) + 0.5
    want = (X0 + 607.0 + cols * 30.0) / 1000.0
    assert np.allclose(out[5:-5, 5:-5], want[None, 5:-5], atol=1e-6)


def test_target_grid_covers_the_bbox_in_its_utm_zone():
    bbox = (-71.45, 41.80, -71.38, 41.85)
    g = L.target_grid(bbox, 30.0)
    assert g["crs"] == UTM and g["transform"].a == 30.0 and g["transform"].e == -30.0
    m = L.bbox_mask(g, bbox)
    assert m.mean() > 0.85 and m.any(axis=0).mean() > 0.97 and m.any(axis=1).mean() > 0.97


def test_median_composite_refuses_scenes_on_different_grids():
    a = {"lst": np.zeros((3, 3)), "_georef": {"transform": from_origin(0, 90, 30, 30), "crs": UTM, "shape": (3, 3)}}
    b = {"lst": np.ones((3, 3)), "_georef": {"transform": from_origin(30, 90, 30, 30), "crs": UTM, "shape": (3, 3)}}
    with pytest.raises(ValueError, match="different grids"):
        L._median_composite([a, b])
    c = L._median_composite([a, {**a, "lst": np.full((3, 3), 2.0)}, {**a, "lst": np.full((3, 3), np.nan)}])
    assert np.allclose(c["lst"], 1.0) and c["_georef"] is a["_georef"]


# --------------------------------------------------------------------------- #
# (b) QA_PIXEL and (d) scaling                                                 #
# --------------------------------------------------------------------------- #
def test_qa_pixel_bits():
    clear = 1 << 6
    vals = np.array([clear, 1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 7, clear | (1 << 8)])
    m = L.qa_mask(vals)
    #            clear  fill  dilated cirrus cloud shadow snow   water  clear+conf
    assert m.tolist() == [False, True, True, True, True, True, False, False, False]
    assert L.qa_mask(np.array([1 << 5]), bits=("snow",)).tolist() == [True]


def test_surface_temperature_and_reflectance_scales():
    k = 300.0
    dn = (k - 149.0) / 0.00341802
    out = L.st_to_kelvin(np.array([dn, 0.0]))
    assert out[0] == pytest.approx(k, abs=1e-9) and np.isnan(out[1])
    assert (out[0] - L.KELVIN) == pytest.approx(26.85, abs=1e-9)
    ref = L.sr_to_reflectance(np.array([(0.1 + 0.2) / 0.0000275, 0]))
    assert ref[0] == pytest.approx(0.1) and np.isnan(ref[1])


def test_indices_come_from_reflectance_not_dn():
    nir, red = 0.30, 0.05
    dn = {"nir": np.array([(nir + 0.2) / 2.75e-5]), "red": np.array([(red + 0.2) / 2.75e-5])}
    ndvi_dn = float(L._safe_norm(dn["nir"], dn["red"])[0])
    bands = {k: L.sr_to_reflectance(v) for k, v in dn.items()}
    ndvi = float(L._compute_indices(bands, {"ndvi"})["ndvi"][0])
    assert ndvi == pytest.approx((nir - red) / (nir + red), rel=1e-5) and abs(ndvi_dn - ndvi) > 0.2


# --------------------------------------------------------------------------- #
# (c) campaign-matched selection with a mocked STAC client                     #
# --------------------------------------------------------------------------- #
def _mock_world(n=40):
    """Scenes in UTM 19N: each has a QA array (cloudy rows as configured) and an ST array (300 K + easting)."""
    T = from_origin(X0, Y1, 30.0, 30.0)
    cols = np.arange(n) + 0.5
    k = 300.0 + (cols * 30.0) / 1000.0                              # 300.0–301.2 K across the window
    st = np.tile((k - 149.0) / 0.00341802, (n, 1)).astype(np.uint16)
    clear_qa = np.full((n, n), 1 << 6, dtype=np.uint16)
    data = {}

    def add(sid, day, cloudy_rows=0, platform="landsat-8", st_shift=0.0, cc=10.0):
        qa = clear_qa.copy()
        qa[:cloudy_rows] = 1 << 3
        s = st.astype(float) + st_shift / 0.00341802
        data[f"mem://{sid}/qa"] = qa
        data[f"mem://{sid}/st"] = np.round(s).astype(np.uint16)
        return {"id": sid, "datetime": f"{day}T15:30:00Z", "cloud_cover": cc, "platform": platform,
                "assets": {"qa_pixel": {"href": f"mem://{sid}/qa"}, "lwir11": {"href": f"mem://{sid}/st"}}}

    scenes = [add("near_cloudy", "2020-07-31", cloudy_rows=36),       # +2 d, 90% cloudy
              add("l7_nearest", "2020-07-29", platform="landsat-7"),  # same day, but Landsat 7
              add("clear_minus7", "2020-07-22"),                       # −7 d, clear
              add("clear_plus9", "2020-08-07", st_shift=2.0),          # +9 d, clear, 2 K warmer
              add("outside", "2020-09-20")]                             # beyond the window
    calls = []

    def search(bbox, start, end):
        calls.append((start, end))
        return [dict(s) for s in scenes]

    def reader(href, bbox):
        return data[href], T, UTM

    tgt = {"transform": T, "crs": UTM, "shape": (n, n)}
    return search, reader, tgt, calls, k


def test_campaign_lst_picks_the_nearest_clear_scene_and_records_every_candidate():
    search, reader, tgt, calls, k = _mock_world()
    res = L.campaign_lst((-71.5, 41.8, -71.4, 41.9), date(2020, 7, 29), window_days=16, min_clear=0.5,
                         target=tgt, study_mask=np.ones(tgt["shape"], bool), search=search, reader=reader)
    assert calls == [(date(2020, 7, 13), date(2020, 8, 14))]
    assert [s["id"] for s in res["scenes"]] == ["clear_minus7"]
    sel = res["scenes"][0]
    assert sel["offset_days"] == -7 and sel["date"] == "2020-07-22" and sel["clear_fraction"] == 1.0
    cand = {c["id"]: c for c in res["candidates"]}
    assert set(cand) == {"near_cloudy", "clear_minus7", "clear_plus9"}      # Landsat 7 and out-of-window dropped
    assert cand["near_cloudy"]["clear_fraction"] == pytest.approx(0.1) and not cand["near_cloudy"]["selected"]
    assert cand["clear_plus9"]["inspected"] is False and cand["clear_plus9"]["clear_fraction"] is None
    assert np.allclose(res["lst_k"], k[None, :], atol=0.004)               # DN rounding: 0.0034 K
    assert np.allclose(res["lst_c"], res["lst_k"] - 273.15)


def test_campaign_lst_two_scenes_is_their_median_and_clouds_are_masked():
    search, reader, tgt, _, k = _mock_world()
    res = L.campaign_lst((-71.5, 41.8, -71.4, 41.9), date(2020, 7, 29), window_days=16, min_clear=0.05,
                         n_scenes=2, target=tgt, study_mask=np.ones(tgt["shape"], bool), search=search,
                         reader=reader)
    assert [s["id"] for s in res["scenes"]] == ["near_cloudy", "clear_minus7"]
    lst = res["lst_k"]
    # cloudy rows of the nearest scene: only the clear scene remains (a median of one value)
    assert np.allclose(lst[:36], k[None, :], atol=0.004)
    assert np.allclose(lst[36:], k[None, :], atol=0.004)                    # both scenes agree there


def test_campaign_lst_reports_no_scene_when_none_is_clear_enough():
    search, reader, tgt, _, _ = _mock_world()
    res = L.campaign_lst((-71.5, 41.8, -71.4, 41.9), date(2020, 7, 31), window_days=3, min_clear=0.5,
                         target=tgt, study_mask=np.ones(tgt["shape"], bool), search=search, reader=reader)
    assert res["lst_k"] is None and res["scenes"] == []
    assert [c["id"] for c in res["candidates"]] == ["near_cloudy"]


def test_download_bands_masks_qa_and_reprojects(tmp_path):
    search, reader, tgt, _, k = _mock_world()
    scene = search(None, None, None)[0]                                      # near_cloudy
    res = L._download_bands(scene, (-71.5, 41.8, -71.4, 41.9), {"lst"}, tmp_path, target=tgt, reader=reader)
    lst = res["bands"]["lst_kelvin"]
    assert np.isnan(lst[:36]).all() and np.allclose(lst[36:], k[None, :], atol=0.004)
    assert res["clear_fraction"] == pytest.approx(0.1)
    assert np.allclose(res["bands"]["lst"], lst - 273.15, equal_nan=True)


def test_stac_items_keep_their_platform(monkeypatch):
    import io
    import json

    page = {"features": [{"id": "LC09_x", "properties": {"datetime": "2020-07-22T15:30:00Z", "eo:cloud_cover": 5,
                                                         "platform": "landsat-9"}, "assets": {}}], "links": []}

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps(page).encode())

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    out = L._query_stac_endpoint("https://example.invalid/search", "landsat-c2-l2", (0, 0, 1, 1),
                                 date(2020, 7, 1), date(2020, 7, 31))
    assert out == [{"id": "LC09_x", "datetime": "2020-07-22T15:30:00Z", "cloud_cover": 5, "platform": "landsat-9",
                    "assets": {}}]


def test_read_window_returns_the_window_transform_of_a_real_cog(tmp_path):
    """The rasterio reader on a local GeoTIFF: the bbox window keeps its own transform (pixels stay put)."""
    from pyproj import Transformer

    arr, T = _scene_raster(n=200)
    path = tmp_path / "st.tif"
    with rasterio.open(path, "w", driver="GTiff", width=200, height=200, count=1, dtype="float64", crs=UTM,
                       transform=T) as dst:
        dst.write(arr, 1)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    (w, e), (s, n) = to_ll.transform([X0 + 1500, X0 + 2700], [Y1 - 2700, Y1 - 1500])
    win, wt, crs = L.read_window(str(path), (min(w, e), min(s, n), max(w, e), max(s, n)))
    assert crs == UTM and win.shape[0] < 200 and win.shape[1] < 200
    xs = np.array([X0 + 2000.0 + 15.0])
    out = L.sample_raster(win, wt, xs, np.array([Y1 - 2000.0 - 15.0]))
    assert out[0] == pytest.approx(xs[0] / 1000.0, abs=0.0151)


def test_campaign_lst_tells_read_failures_apart_from_no_clear_scene():
    """R9/R20: a blocked or failing COG host must not look like 'no clear scene'."""
    import urllib.error

    search, reader, tgt, _, _ = _mock_world()
    kw = dict(window_days=16, min_clear=0.5, target=tgt, study_mask=np.ones(tgt["shape"], bool), search=search)
    bbox, day = (-71.5, 41.8, -71.4, 41.9), date(2020, 7, 29)
    ok = L.campaign_lst(bbox, day, reader=reader, **kw)
    assert ok["status"] == "ok" and ok["io_errors"] == []
    cloudy = L.campaign_lst(bbox, date(2020, 7, 31), reader=reader, **{**kw, "window_days": 3})
    assert cloudy["status"] == "no_clear_scene" and cloudy["io_errors"] == []
    none = L.campaign_lst(bbox, date(2021, 7, 31), reader=reader, **kw)
    assert none["status"] == "no_scene"

    def blocked(href, b):
        raise urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))

    res = L.campaign_lst(bbox, day, reader=blocked, **kw)
    assert res["lst_c"] is None and res["status"] == "io_error"
    assert {e["kind"] for e in res["io_errors"]} == {"blocked"}
    assert all(c.get("error_kind") == "blocked" for c in res["candidates"] if c["inspected"])

    def flaky(href, b):
        raise urllib.error.HTTPError(href, 503, "Service Unavailable", {}, None)

    assert L.campaign_lst(bbox, day, reader=flaky, **kw)["io_errors"][0]["kind"] == "io"

    def s3(b, d0, d1):
        return [{**sc, "assets": {"qa_pixel": {"href": "s3://usgs-landsat/x_QA.TIF"},
                                  "lwir11": {"href": "s3://usgs-landsat/x_ST.TIF"}}} for sc in search(b, d0, d1)]

    r3 = L.campaign_lst(bbox, day, reader=reader, **{**kw, "search": s3})
    assert r3["status"] == "io_error" and r3["io_errors"][0]["kind"] == "unreadable"


def test_io_error_kind():
    import urllib.error

    assert L.io_error_kind(urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))) == "blocked"
    assert L.io_error_kind(urllib.error.HTTPError("u", 403, "Forbidden", {}, None)) == "blocked"
    assert L.io_error_kind(urllib.error.HTTPError("u", 500, "x", {}, None)) == "io"
    assert L.io_error_kind(TimeoutError("timed out")) == "io"
    assert L.io_error_kind(RuntimeError("STAC query failed")) is None
    try:
        try:
            raise urllib.error.URLError(OSError("Tunnel connection failed: 403"))
        except urllib.error.URLError as exc:
            raise RuntimeError("STAC query failed (https://planetarycomputer.microsoft.com/api/stac/v1/search)") from exc
    except RuntimeError as e:
        assert L.io_error_kind(e) == "blocked"
    assert L.io_error_kind(ValueError("bad shape")) is None


def test_campaign_lst_default_search_is_planetary_computer_with_required_signing(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "_query_stac_endpoint", lambda *a, **k: calls.append((a, k)) or [])
    L.search_pc((0, 0, 1, 1), date(2020, 7, 1), date(2020, 7, 31))
    (args, kw), = calls
    assert args[0] == L.PC_STAC and kw == {"sign_assets": True, "require_signing": True}


def test_a_fully_clear_scene_a_week_further_beats_a_partly_cloudy_nearer_one():
    """R19 (LA 2019): the nearest-date rule kept LC08 2019-09-19 (+4 d, 60% clear) over 2019-09-03 (−12 d, 99.9%
    clear), and 38% of the driven cells lost their LST."""
    T = from_origin(X0, Y1, 30.0, 30.0)
    n = 40
    data = {}

    def add(sid, day, cloudy_rows):
        qa = np.full((n, n), 1 << 6, dtype=np.uint16)
        qa[:cloudy_rows] = 1 << 3
        data[f"mem://{sid}/qa"] = qa
        data[f"mem://{sid}/st"] = np.full((n, n), int((300.0 - 149.0) / 0.00341802), dtype=np.uint16)
        return {"id": sid, "datetime": f"{day}T18:30:00Z", "cloud_cover": 10.0, "platform": "landsat-8",
                "assets": {"qa_pixel": {"href": f"mem://{sid}/qa"}, "lwir11": {"href": f"mem://{sid}/st"}}}

    tgt = {"transform": T, "crs": UTM, "shape": (n, n)}
    kw = dict(window_days=16, min_clear=0.5, target=tgt, study_mask=np.ones((n, n), bool),
              reader=lambda h, b: (data[h], T, UTM))
    near, far = add("p_plus4", "2019-09-19", 16), add("c_minus12", "2019-09-03", 0)       # 60% vs 100% clear
    res = L.campaign_lst((-118.3, 34.0, -118.2, 34.1), date(2019, 9, 15), search=lambda b, d0, d1: [near, far], **kw)
    assert [s["id"] for s in res["scenes"]] == ["c_minus12"] and "nearest scene ≥ 80% clear" in res["selection"]
    assert np.isfinite(res["lst_c"]).all()
    # no scene 80% clear: the clearest one ≥ min_clear, not the nearest
    a, b = add("a_plus2", "2019-09-17", 16), add("b_minus5", "2019-09-10", 12)             # 60% vs 70%
    res2 = L.campaign_lst((-118.3, 34.0, -118.2, 34.1), date(2019, 9, 15), search=lambda b_, d0, d1: [a, b], **kw)
    assert [s["id"] for s in res2["scenes"]] == ["b_minus5"] and "clearest" in res2["selection"]
