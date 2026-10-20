"""Continuous US land cover (NLCD tree canopy, percent impervious) added post hoc to built cities: a synthetic MRLC
WCS (GeoTIFF windows on the NLCD Albers pixel grid, its fill codes, nodata declared 0) → the layers on the city's
UTM grid (alignment, orientation, units, no data), focal means, the files' schema, idempotent reruns, failures and
the CLI.  Offline: every request goes to the synthetic server."""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.error
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from sparc.core.lstair import landcover_us as LC
from sparc.core.lstair.dataset import CityGrid, NetworkBlocked, focal_mean, load_city

UTM17 = "EPSG:32617"
GRID = {"crs": UTM17, "x0": 726585.0, "y0": 4124895.0, "dx": 30.0, "nx": 60, "ny": 50}      # near Farmville, VA
XW, YS = GRID["x0"] - 15.0, GRID["y0"] - 15.0                                              # the grid's SW corner
TCC_PERIOD, IMP_PERIOD = 600.0, 450.0                                                      # sawtooth periods (m)


def tcc_codes(x, y):
    """Planted TCC (uint8 codes) at UTM positions: a 0–100% sawtooth along x (1/6 % per m: a one-cell shift is a 5%
    error), a 0% strip in the south, a 254 (non-processing) patch and a 255 (background) NE corner."""
    u, v = np.asarray(x) - XW, np.asarray(y) - YS
    out = (np.mod(u, TCC_PERIOD) / TCC_PERIOD * 100.0)
    out = np.where(v < 150.0, 0.0, out)
    out = np.where((u >= 300) & (u < 600) & (v >= 900) & (v < 1200), 254.0, out)
    return np.where((u >= 1500) & (v >= 1200), 255.0, out)


def imp_codes(x, y):
    """Planted impervious: a sawtooth along y (rows must grow north), a 0% strip in the west, 127 in the NE corner."""
    u, v = np.asarray(x) - XW, np.asarray(y) - YS
    out = np.mod(v, IMP_PERIOD) / IMP_PERIOD * 100.0
    out = np.where(u < 150.0, 0.0, out)
    return np.where((u >= 1500) & (v >= 1200), 127.0, out)


def tcc_region(x, y):
    u, v = np.asarray(x) - XW, np.asarray(y) - YS
    r = np.floor(u / TCC_PERIOD) + 10
    r = np.where(v < 150.0, 0, r)
    r = np.where((u >= 300) & (u < 600) & (v >= 900) & (v < 1200), 1, r)
    return np.where((u >= 1500) & (v >= 1200), 2, r)


def imp_region(x, y):
    u, v = np.asarray(x) - XW, np.asarray(y) - YS
    r = np.floor(v / IMP_PERIOD) + 10
    r = np.where(u < 150.0, 0, r)
    return np.where((u >= 1500) & (v >= 1200), 2, r)


def interior(region, X, Y, r=45.0):
    """Cells whose centre and the points ±r around it lie in one region (no discontinuity under the cell)."""
    c = region(X, Y)
    ok = np.ones(X.shape, bool)
    for dx in (-r, 0.0, r):
        for dy in (-r, 0.0, r):
            ok &= region(X + dx, Y + dy) == c
    return ok, c


class FakeWCS:
    """A synthetic MRLC GetCoverage: the pixels of the NLCD 30 m grid whose centres lie in the EPSG:5070 subset (both
    bounds inclusive), valued by each centre's position in UTM 17N, as a GeoTIFF with nodata declared 0 (as the real
    server does).  ``shift_m`` moves the returned georeference off the NLCD grid (a broken server)."""

    def __init__(self, layers=None, shift_m=0.0):
        self.layers = layers or {"tcc": tcc_codes, "nlcd_imp": imp_codes}
        self.shift_m = shift_m
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        import rasterio
        from pyproj import Transformer
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        self.calls.append(url)
        q = parse_qs(urlparse(url).query)
        cov = q["coverageid"][0]
        layer = "tcc" if "_tcc_" in cov else "nlcd_imp"
        if layer not in self.layers:
            return (b'<?xml version="1.0"?><ows:ExceptionReport><ows:Exception exceptionCode="NoSuchCoverage">'
                    b"<ows:ExceptionText>Could not locate coverage " + cov.encode() + b"</ows:ExceptionText>"
                    b"</ows:Exception></ows:ExceptionReport>")
        sub = {s[0]: tuple(float(v) for v in s[2:-1].split(",")) for s in q["subset"]}
        (xa, xb), (ya, yb) = sub["X"], sub["Y"]
        ox, oy = LC.NLCD_ORIGIN
        i0, i1 = int(np.ceil((xa - ox) / 30.0 - 0.5)), int(np.floor((xb - ox) / 30.0 - 0.5))
        j0, j1 = int(np.ceil((oy - yb) / 30.0 - 0.5)), int(np.floor((oy - ya) / 30.0 - 0.5))
        CX, CY = np.meshgrid(ox + 15.0 + 30.0 * np.arange(i0, i1 + 1), oy - 15.0 - 30.0 * np.arange(j0, j1 + 1))
        xu, yu = Transformer.from_crs("EPSG:5070", UTM17, always_xy=True).transform(CX, CY)
        arr = np.rint(self.layers[layer](np.asarray(xu), np.asarray(yu))).astype(np.uint8)
        buf = MemoryFile()
        with buf.open(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="uint8",
                      crs="EPSG:5070", nodata=0,
                      transform=from_origin(ox + 30.0 * i0 + self.shift_m, oy - 30.0 * j0, 30.0, 30.0)) as dst:
            dst.write(arr, 1)
        assert rasterio.__version__
        return bytes(buf.read())


def write_city(root, slug="testville_va", date="2020-07-29", status="ok", grid=GRID):
    """A finished city in the layout ``lstair dataset`` writes: meta.json, table.parquet (two runs, one flagged
    ``qa_exclude``), rasters.npz (WorldCover-like canopy/impervious and their focal means, the driven mask)."""
    d = root / "cities" / slug
    d.mkdir(parents=True)
    ctx = CityGrid(**grid)
    X, Y = ctx.centres()
    canopy = np.clip(40.0 + 45.0 * np.sin((X - XW) / 150.0), 0.0, 100.0)
    canopy[Y - YS < 150.0] = 0.0
    impervious = np.clip(50.0 + 45.0 * np.cos((Y - YS) / 120.0), 0.0, 100.0)
    rasters = {"canopy": canopy, "impervious": impervious}
    for k in ("canopy", "impervious"):
        for r in (100, 300, 1000):
            rasters[f"{k}_f{r}"] = focal_mean(rasters[k], r, ctx.dx)
    nx, ny = ctx.nx, ctx.ny
    af = np.unique(np.r_[10 * nx + np.arange(nx), np.arange(ny) * nx + 20, np.arange(ny) * nx + 55])
    frames = []
    for run, cells in (("af", af), ("am", af[::2])):
        iy, ix = np.divmod(cells, nx)
        x, y = ctx.cell_xy(cells)
        frames.append(pd.DataFrame({
            "city": "Testville, VA", "slug": slug, "campaign_date": date, "run": run, "run_id": f"{date} {run}",
            "run_date": date, "cell": cells, "x": x, "y": y, "Ta": 30.0 + 0.001 * (x - XW),
            "dTa": 0.001 * (x - x.mean()),
            "canopy": rasters["canopy"][iy, ix], "impervious": rasters["impervious"][iy, ix],
            "qa_exclude": "" if run == "af" else "single-vehicle run: drift not identifiable"}))
    pd.concat(frames, ignore_index=True).to_parquet(d / "table.parquet", index=False)
    driven = np.zeros(ctx.shape, bool)
    driven.ravel()[af] = True
    np.savez_compressed(d / "rasters.npz", **{k: v.astype(np.float32) for k, v in rasters.items()}, driven=driven)
    meta = {"schema": 1, "status": status, "city": "Testville, VA", "slug": slug, "campaign_date": date, "grid": grid,
            "runs": [{"run_id": f"{date} af", "run": "af"}], "rasters": sorted(rasters), "warnings": [],
            "built_utc": "2026-10-20T00:00:00Z"}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


@pytest.fixture()
def city(tmp_path):
    return write_city(tmp_path)


def _files(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


def test_the_layers_land_on_the_grid_in_percent_with_rows_growing_north():
    ctx = CityGrid(**GRID)
    wcs = FakeWCS()
    R, prov = LC.landcover_rasters(ctx, dt.date(2020, 7, 29), fetch=wcs)
    X, Y = ctx.centres()
    for k, codes, region in (("tcc", tcc_codes, tcc_region), ("nlcd_imp", imp_codes, imp_region)):
        a = R[k]
        assert a.shape == ctx.shape and a.dtype == np.float64
        assert np.nanmin(a) >= 0.0 and np.nanmax(a) <= 100.0                       # percent, fill codes gone
        ok, reg = interior(region, X, Y)
        saw = ok & (reg >= 10)
        truth = codes(X, Y)
        # the sawtooth rises 1/6 (tcc, along x) and 2/9 (impervious, along y) % per m: a cell off by one, or
        # rows flipped, would be ≥ 5% wrong; the average of the rounded Albers pixels is within ~1%
        err = np.abs(a[saw] - truth[saw])
        assert saw.sum() > 500 and err.max() < 1.5 and err.mean() < 0.5, (k, err.max(), err.mean())
    assert prov["layers"]["tcc"]["year"] == 2020 and prov["layers"]["nlcd_imp"]["year"] == 2021
    assert prov["layers"]["tcc"]["coverage_id"] == "mrlc_download__nlcd_tcc_conus_2020_v2021-4"
    assert prov["layers"]["nlcd_imp"]["coverage_id"] == "mrlc_download__NLCD_2021_Impervious_L48"
    # every request is a GetCoverage of the WCS, bounds a quarter pixel inside the wanted pixels' edges
    ox, oy = LC.NLCD_ORIGIN
    for url in wcs.calls:
        assert url.startswith(LC.WCS + "?") and "request=GetCoverage" in url and "format=image/tiff" in url
        sub = {s[0]: [float(v) for v in s[2:-1].split(",")] for s in parse_qs(urlparse(url).query)["subset"]}
        assert sorted(np.mod(np.array(sub["X"]) - ox, 30.0).round(3)) == [7.5, 22.5]
        assert sorted(np.mod(oy - np.array(sub["Y"]), 30.0).round(3)) == [7.5, 22.5]


def test_fill_codes_are_no_data_and_zero_stays_zero_although_the_tiff_declares_it_nodata():
    ctx = CityGrid(**GRID)
    R, prov = LC.landcover_rasters(ctx, dt.date(2020, 7, 29), fetch=FakeWCS())
    X, Y = ctx.centres()
    ok, reg = interior(tcc_region, X, Y)
    assert np.isnan(R["tcc"][ok & (reg == 1)]).all() and (ok & (reg == 1)).sum() > 20      # 254
    assert np.isnan(R["tcc"][ok & (reg == 2)]).all() and (ok & (reg == 2)).sum() > 20      # 255
    assert (R["tcc"][ok & (reg == 0)] == 0.0).all() and (ok & (reg == 0)).sum() > 100      # 0 is 0%, not nodata
    ok, reg = interior(imp_region, X, Y)
    assert np.isnan(R["nlcd_imp"][ok & (reg == 2)]).all()                                  # 127
    assert (R["nlcd_imp"][ok & (reg == 0)] == 0.0).all() and (ok & (reg == 0)).sum() > 100
    # never filled: the no-data cells stay NaN, and the share is recorded
    assert 0.9 < prov["layers"]["tcc"]["grid_share"] < 0.98
    # focal means average the finite neighbours (masked kernel), as the WorldCover ones
    assert np.isfinite(R["tcc_f100"][ok & (reg == 2)]).any()


def test_tiled_requests_give_the_window_of_one_request():
    ctx = CityGrid(**GRID)
    one, many = FakeWCS(), FakeWCS()
    A, _ = LC.landcover_rasters(ctx, dt.date(2020, 7, 29), fetch=one)
    B, prov = LC.landcover_rasters(ctx, dt.date(2020, 7, 29), fetch=many, tile_px=16)
    assert len(one.calls) == 2 and len(many.calls) > 2 * 9
    for k in LC.COLUMNS:
        np.testing.assert_array_equal(np.isnan(A[k]), np.isnan(B[k]))
        np.testing.assert_allclose(A[k], B[k], equal_nan=True, atol=1e-9)
    assert len(prov["layers"]["tcc"]["urls"]) == len(many.calls) // 2


def test_a_tile_off_the_nlcd_grid_or_an_exception_report_raises():
    ctx = CityGrid(**GRID)
    with pytest.raises(RuntimeError, match="not on the NLCD 30 m pixel grid"):
        LC.landcover_rasters(ctx, dt.date(2020, 7, 29), fetch=FakeWCS(shift_m=10.0))
    with pytest.raises(RuntimeError, match="Could not locate coverage mrlc_download__NLCD_2021"):
        LC.landcover_rasters(ctx, dt.date(2020, 7, 29), fetch=FakeWCS(layers={"tcc": tcc_codes}))


@pytest.mark.parametrize("day, tcc, imp", [("2019-08-14", 2019, 2019), ("2020-07-29", 2020, 2021),
                                           ("2021-07-15", 2021, 2021), ("2023-07-24", 2021, 2021),
                                           ("2017-09-01", 2017, 2016), ("2014-07-01", 2014, 2013)])
def test_each_layer_takes_its_year_nearest_the_campaign(day, tcc, imp):
    ids = LC.coverage_ids(dt.date.fromisoformat(day))
    assert ids["tcc"] == f"mrlc_download__nlcd_tcc_conus_{tcc}_v2021-4"
    assert ids["nlcd_imp"] == f"mrlc_download__NLCD_{imp}_Impervious_L48"
    assert LC.nearest_year(dt.date(2017, 7, 1), (2018, 2016)) == 2016                    # a tie: the earlier year


def test_add_landcover_samples_the_rasters_at_the_cells_and_keeps_everything_else(city):
    before = pd.read_parquet(city / "table.parquet")
    with np.load(city / "rasters.npz") as z:
        old = {k: z[k] for k in z.files}
    meta0 = json.loads((city / "meta.json").read_text())
    rec = LC.add_landcover(city, fetch=FakeWCS())
    assert rec["status"] == "added" and rec["years"] == {"tcc": 2020, "nlcd_imp": 2021}
    t = pd.read_parquet(city / "table.parquet")
    assert list(t.columns) == list(before.columns) + list(LC.COLUMNS)
    pd.testing.assert_frame_equal(t[before.columns], before)                     # qa_exclude and all else kept
    assert all(t[c].dtype == np.float64 for c in LC.COLUMNS)
    with np.load(city / "rasters.npz") as z:
        new = {k: z[k] for k in z.files}
    assert set(new) == set(old) | set(LC.COLUMNS)
    for k, v in old.items():
        np.testing.assert_array_equal(new[k], v)
        assert new[k].dtype == v.dtype                                           # driven stays bool
    assert all(new[c].dtype == np.float32 for c in LC.COLUMNS)
    nx = GRID["nx"]
    iy, ix = np.divmod(t["cell"].to_numpy(), nx)
    for c in LC.COLUMNS:
        np.testing.assert_allclose(t[c].to_numpy(), new[c][iy, ix], rtol=1e-6, equal_nan=True)
    # the focal means are the dataset's kernel at the dataset's radii, on the stored raster
    for k in ("tcc", "nlcd_imp"):
        for r in (100, 300, 1000):
            ref = focal_mean(new[k].astype(np.float64), r, 30.0)
            np.testing.assert_allclose(new[f"{k}_f{r}"], ref, rtol=1e-5, atol=1e-4, equal_nan=True)
    assert np.nanstd(new["tcc_f1000"]) < np.nanstd(new["tcc_f100"]) < np.nanstd(new["tcc"])
    # some driven cells lie in the NE no-data corner: NaN there, recorded as coverage
    meta = json.loads((city / "meta.json").read_text())
    lc = meta["landcover_us"]
    assert t["tcc"].isna().any() and t["tcc"].notna().mean() > 0.8
    assert lc["driven"]["tcc"]["share_covered"] == pytest.approx(float(t.drop_duplicates("cell")["tcc"].notna().mean()))
    assert set(lc["driven"]["tcc"]) >= {"share_zero", "canopy_share_zero", "r_with_canopy"}
    assert lc["layers"]["tcc"]["urls"][0].startswith(LC.WCS) and lc["layers"]["nlcd_imp"]["year"] == 2021
    assert lc["columns"] == list(LC.COLUMNS) and lc["focal_radii_m"] == [100, 300, 1000] and lc["key"]
    assert set(LC.COLUMNS) <= set(meta["rasters"])
    assert {k: v for k, v in meta.items() if k not in ("landcover_us", "rasters")} == \
        {k: v for k, v in meta0.items() if k != "rasters"}
    assert not list(city.glob("*.part"))
    c = load_city(city)                                                         # still a dataset city
    assert set(LC.COLUMNS) <= set(c.table.columns) and set(LC.COLUMNS) <= set(c.rasters)


def test_rerunning_is_idempotent_and_a_rebuild_gets_the_columns_back(city):
    LC.add_landcover(city, fetch=FakeWCS())
    first = _files(city)
    again = FakeWCS()
    assert LC.add_landcover(city, fetch=again)["status"] == "present" and not again.calls
    assert _files(city) == first
    assert LC.has_landcover(city)
    forced = FakeWCS()
    assert LC.add_landcover(city, fetch=forced, force=True)["status"] == "added" and forced.calls
    t1 = pd.read_parquet(city / "table.parquet")
    # a rebuild by ``lstair dataset`` writes the table, rasters and meta without the columns
    meta = json.loads((city / "meta.json").read_text())
    meta.pop("landcover_us")
    (city / "meta.json").write_text(json.dumps(meta))
    t1.drop(columns=list(LC.COLUMNS)).to_parquet(city / "table.parquet", index=False)
    assert not LC.has_landcover(city)
    assert LC.add_landcover(city, fetch=FakeWCS())["status"] == "added"
    pd.testing.assert_frame_equal(pd.read_parquet(city / "table.parquet"), t1)
    # a city rebuilt on another grid is not taken as done
    meta = json.loads((city / "meta.json").read_text())
    meta["grid"] = {**meta["grid"], "x0": meta["grid"]["x0"] + 30.0}
    (city / "meta.json").write_text(json.dumps(meta))
    assert not LC.has_landcover(city)


def test_add_dataset_records_failures_leaves_them_untouched_and_skips_unbuilt_cities(tmp_path):
    ok = write_city(tmp_path, "testville_va")
    bad = write_city(tmp_path, "otherton_va", date="2019-08-14")
    unbuilt = write_city(tmp_path, "nobuild_va", status="failed")
    bad_files = _files(bad)

    def fetch(url):
        if "2019" in url:                                                      # otherton's years
            raise RuntimeError("www.mrlc.gov answered HTTP 404 for ...: Could not locate coverage")
        return FakeWCS()(url)

    s = LC.add_dataset(tmp_path, fetch=fetch)
    assert s["cities"]["testville_va"]["status"] == "added"
    assert s["cities"]["otherton_va"]["status"] == "failed" and "404" in s["cities"]["otherton_va"]["error"]
    assert s["cities"]["nobuild_va"]["status"] == "not built"
    assert _files(bad) == bad_files and "landcover_us" not in json.loads((unbuilt / "meta.json").read_text())
    assert json.loads((tmp_path / "landcover_us.json").read_text())["cities"]["testville_va"]["status"] == "added"
    assert LC.has_landcover(ok)
    # by name: only that city, and a name that matches no folder says so
    s = LC.add_dataset(tmp_path, ["Testville, VA"], fetch=fetch)
    assert list(s["cities"]) == ["testville_va"] and s["cities"]["testville_va"]["status"] == "present"
    with pytest.raises(ValueError, match="no city folder"):
        LC.add_dataset(tmp_path, ["Nowhere"], fetch=fetch)


def test_a_city_whose_augmented_files_are_unreadable_fails_alone(tmp_path):
    """A city done earlier whose rasters.npz was later truncated: its skip check must not abort the command (it
    raised BadZipFile before any city was processed); the city is recorded as failed, the others go on."""
    done, other = write_city(tmp_path, "testville_va"), write_city(tmp_path, "otherton_va")
    LC.add_landcover(done, fetch=FakeWCS())
    raw = (done / "rasters.npz").read_bytes()
    (done / "rasters.npz").write_bytes(raw[: len(raw) // 2])
    assert not LC.has_landcover(done)
    s = LC.add_dataset(tmp_path, fetch=FakeWCS())
    assert s["cities"]["testville_va"]["status"] == "failed" and s["cities"]["otherton_va"]["status"] == "added"
    assert LC.has_landcover(other)


def test_a_blocked_wcs_host_stops_before_any_fetch(city, monkeypatch):
    calls = []
    monkeypatch.setattr(LC, "probe", lambda url, timeout=10.0, opener=None: "URLError: Tunnel connection failed: 403")
    monkeypatch.setattr(LC, "http_get", lambda url, **k: calls.append(url))
    with pytest.raises(NetworkBlocked, match="www.mrlc.gov"):
        LC.add_dataset(city.parent.parent)
    assert not calls and "landcover_us" not in json.loads((city / "meta.json").read_text())


def test_http_get_names_a_refusal_retries_5xx_and_reports_the_exception_text(monkeypatch):
    import sparc.core.climate as climate

    monkeypatch.setattr(climate, "retry_wait", lambda *a, **k: None)
    body = (b'<ows:ExceptionReport><ows:Exception><ows:ExceptionText>Could not locate coverage x'
            b'</ows:ExceptionText></ows:Exception></ows:ExceptionReport>')

    def raising(code, payload=b""):
        def opener(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(payload))
        return opener

    with pytest.raises(NetworkBlocked, match="www.mrlc.gov"):
        LC.http_get(LC.WCS, opener=raising(403))
    with pytest.raises(NetworkBlocked):
        LC.http_get(LC.WCS, opener=lambda req, timeout=None: (_ for _ in ()).throw(
            urllib.error.URLError("Tunnel connection failed: 403 Forbidden")))
    with pytest.raises(RuntimeError, match="HTTP 404 .*Could not locate coverage x"):
        LC.http_get(LC.WCS, opener=raising(404, body))
    answers = [raising(503), raising(502)]

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"II*\x00tiff"

    def flaky(req, timeout=None):
        return answers.pop(0)(req, timeout) if answers else _Resp()

    assert LC.http_get(LC.WCS, opener=flaky) == b"II*\x00tiff"
    with pytest.raises(RuntimeError, match="failed 4 times"):
        LC.http_get(LC.WCS, opener=raising(503))


def test_cli_adds_the_columns_and_names_a_blocked_host(city, monkeypatch, capsys):
    from sparc.core.cli import main

    data = city.parent.parent
    monkeypatch.setattr(LC, "probe", lambda url, timeout=10.0, opener=None: None)
    monkeypatch.setattr(LC, "http_get", FakeWCS())
    assert main(["lstair", "add-landcover", "--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "testville_va: added (tcc 2020, nlcd_imp 2021)" in out and "1 of 1 cities" in out
    assert main(["lstair", "add-landcover", "--data", str(data), "--cities", "testville_va"]) == 0
    assert "testville_va: present" in capsys.readouterr().out
    monkeypatch.setattr(LC, "probe", lambda url, timeout=10.0, opener=None: "URLError: Tunnel connection failed: 403")
    assert main(["lstair", "add-landcover", "--data", str(data), "--force"]) == 2
    assert "www.mrlc.gov" in capsys.readouterr().err
