"""Continuous US land cover for the LST pilot: NLCD tree canopy and percent impervious on each city grid.

WorldCover gives every 10 m pixel one class, so the dataset's ``canopy`` and ``impervious`` are class fractions: a
street tree over a road inside a built pixel counts as built, and canopy is exactly 0 on more than half of the
driven cells in 16 of the 31 pilot cities (54–98%: Las Cruces 98%, Victorville 97%, Albuquerque 92%, San Diego 82%;
review of 2026-10-20, recounted from the built tables).  An "LST beyond land cover" gain could then be LST standing
in for the canopy the classes miss.  The pre-registered sensitivity (``docs/results/lst_air_pilot.md``: continuous land
cover in place of WorldCover classes, if obtainable) uses continuous layers instead; every pilot city lies in CONUS:

==================  ===============================================================================================
``tcc`` (%)         USFS Tree Canopy Cover v2021-4 (the NLCD tree canopy product), CONUS, annual 2011–2021, 30 m
``nlcd_imp`` (%)    NLCD Percent Developed Imperviousness, L48: 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019,
                    2021, 30 m
==================  ===============================================================================================

with their Gaussian focal means at 100, 300 and 1000 m (``tcc_f100`` … ``nlcd_imp_f1000``), the radii and kernel of
the WorldCover ones (:func:`sparc.core.lstair.dataset.focal_mean`).  ``evaluate --land-cover nlcd`` swaps them in
for canopy / impervious (:data:`REPLACES`).

Source
------
Probed 2026-10-20.  **MRLC's GeoServer WCS 2.0.1** answers at ``https://www.mrlc.gov/geoserver/mrlc_download/wcs``:
``request=GetCoverage&coverageid=<id>&subset=X(x0,x1)&subset=Y(y0,y1)&format=image/tiff``, the subset in EPSG:5070
(CONUS Albers, the layers' native CRS; 30 m pixels on the grid whose upper-left corner is −2,493,045 / 3,310,005 m,
from DescribeCoverage).  It returns the native uint8 values of the window only, on the native pixel grid: Farmville's
10 × 12 km in about 1.2 s, a 75 × 90 km window (larger than Seattle's grid) in 2.8 s.  Coverage ids:
``mrlc_download__nlcd_tcc_conus_{year}_v2021-4`` and ``mrlc_download__NLCD_{year}_Impervious_L48``; an unknown id
answers HTTP 404 with an OWS ExceptionReport.  Values 0–100 are percent (open water inside the mapped area is 0);
above 100 is no data: impervious 127 and TCC 255 beyond the mapped area (offshore), TCC 254 its non-processing mask.
The GeoTIFF declares nodata 0, a valid percent here, so the declaration is ignored.  Not usable from here: MRLC's S3
bucket (``s3-us-west-2.amazonaws.com/mrlc``, whole-CONUS zips: the listing and every key tried answered 403, so a
wrong name cannot be told from a refusal), the USFS ArcGIS ImageServer (``apps.fs.usda.gov/fsgisx01``: 403) and
Planetary Computer (no NLCD collection).  The Annual NLCD (Collection 1) fractional impervious layers are not on the
WCS (only its land-cover change summaries are).

Method
------
1. **Window.**  The city grid's outline, densified, is transformed to EPSG:5070 and snapped outwards to the NLCD
   pixel grid with a :data:`MARGIN_PX` margin.  A window wider than :data:`MAX_TILE_PX` is fetched in tiles, each
   placed by the georeference it comes back with (which must lie on the NLCD grid).  Subset bounds sit a quarter
   pixel inside the wanted pixels' outer edges, so which pixels the server includes never hangs on a pixel centre
   lying exactly on a bound (the server includes such a centre on one side of the window and not the other).
2. **Values.**  Values above 100 become NaN; the window is reprojected onto the city grid with GDAL's area-weighted
   average (:func:`sparc.data.collect.landsat.reproject_array`, ``average``): each 30 m UTM cell is the
   overlap-weighted mean of the Albers pixels it touches, the percent of the cell's area, as WorldCover's class
   fractions × 100 are.  A cell that no valid pixel touches is NaN (never filled).
3. **Year.**  For each layer, its year nearest the campaign date (a year's nominal date is 1 July; ties: the earlier
   year): the campaign year where the layer has it, else the nearest (2021 for the 2022–23 campaigns; impervious
   2021 for the 2020 ones, 2019 for 2019).
4. **Post hoc.**  :func:`add_landcover` augments a finished city in place (``sparc core lstair add-landcover``):
   the columns are sampled at the driven cells into ``table.parquet``, the rasters added to ``rasters.npz`` and the
   provenance (URLs, coverage ids, years, coverage share, agreement with WorldCover on the driven cells) kept in
   ``meta["landcover_us"]``.  Every file is replaced atomically, ``meta.json`` last, and only after both layers were
   read, so a failed fetch leaves the city as it was.  It is idempotent: a city whose meta carries the same key
   (grid, coverage ids, method) and whose files hold every column is skipped.  ``lstair dataset`` does not know
   these columns, so a rebuild drops them (its meta has no ``landcover_us``) and rerunning the command adds them back.

References: Coulston, J.W. et al. (2012), Modeling percent tree canopy cover: a pilot study, *Photogramm. Eng.
Remote Sens.* 78(7), 715–727; Dewitz, J. (2023), *National Land Cover Database (NLCD) 2021 Products*, USGS data
release, doi:10.5066/P9JZ7AO3; USDA Forest Service (2023), *USFS Tree Canopy Cover* v2021-4,
https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sparc.core.lstair import to_json
from sparc.core.lstair.dataset import FOCAL_RADII, CityGrid, NetworkBlocked, focal_mean, probe, slugify

log = logging.getLogger(__name__)

WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
HOST = "www.mrlc.gov"
PURPOSE = "NLCD tree canopy and percent impervious (MRLC GeoServer WCS)"
REMEDY = "Allow www.mrlc.gov (only add-landcover needs it)."
NLCD_CRS = "EPSG:5070"
NLCD_ORIGIN = (-2493045.0, 3310005.0)   # upper-left corner of the CONUS 30 m pixel grid (DescribeCoverage)
NLCD_RES = 30.0
VALID_MAX = 100                         # above: no data (impervious 127; TCC 254 non-processing, 255 background)
MARGIN_PX = 2                           # NLCD pixels beyond the grid's outline (the average kernel at the edges)
MAX_TILE_PX = 2048                      # NLCD pixels per side of one GetCoverage (≈ 4 MB of uint8)
METHOD = 1                              # part of the key: a change of method refetches every city
#: layer → (coverage id pattern, years the WCS serves, product)
LAYERS = {
    "tcc": ("mrlc_download__nlcd_tcc_conus_{year}_v2021-4", tuple(range(2011, 2022)),
            "USFS Tree Canopy Cover v2021-4 (NLCD tree canopy), CONUS, percent of the pixel"),
    "nlcd_imp": ("mrlc_download__NLCD_{year}_Impervious_L48", (2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021),
                 "NLCD Percent Developed Imperviousness, L48, percent of the pixel"),
}
#: the WorldCover predictor each layer stands in for (``evaluate --land-cover nlcd``)
REPLACES = {"tcc": "canopy", "nlcd_imp": "impervious"}
COLUMNS = tuple(c for k in LAYERS for c in (k, *(f"{k}_f{r}" for r in FOCAL_RADII)))


# --------------------------------------------------------------------------- #
# The WCS                                                                      #
# --------------------------------------------------------------------------- #
def nearest_year(day: dt.date, years) -> int:
    """The year of ``years`` nearest ``day`` (a year's nominal date is 1 July; ties: the earlier year)."""
    return min(sorted(int(y) for y in years), key=lambda y: abs((dt.date(y, 7, 1) - day).days))


def coverage_ids(day: dt.date) -> dict[str, str]:
    """Layer → the coverage id of its year nearest ``day``."""
    return {k: pat.format(year=nearest_year(day, years)) for k, (pat, years, _) in LAYERS.items()}


def wcs_url(coverage: str, x0: float, x1: float, y0: float, y1: float) -> str:
    """GetCoverage of the EPSG:5070 window [x0, x1] × [y0, y1] (m) of ``coverage`` as a GeoTIFF."""
    return (f"{WCS}?service=WCS&version=2.0.1&request=GetCoverage&coverageid={coverage}"
            f"&subset=X({x0:.1f},{x1:.1f})&subset=Y({y0:.1f},{y1:.1f})&format=image/tiff")


def probe_url() -> str:
    """A small WCS answer (DescribeCoverage of one layer, ~3 kB) that shows the host serves the layers."""
    return (f"{WCS}?service=WCS&version=2.0.1&request=DescribeCoverage&coverageid="
            + LAYERS["nlcd_imp"][0].format(year=LAYERS["nlcd_imp"][1][-1]))


def check_host(opener: Callable | None = None, timeout: float = 10.0) -> None:
    """Raise :class:`~sparc.core.lstair.dataset.NetworkBlocked` naming :data:`HOST` when the WCS does not answer."""
    why = probe(probe_url(), timeout=timeout, opener=opener)
    if why:
        raise NetworkBlocked(HOST, why, PURPOSE, REMEDY)


def _exception_text(body: bytes | None) -> str:
    """The OWS ExceptionText of an error answer, else its first characters."""
    text = (body or b"")[:4000].decode("utf-8", "replace")
    m = re.search(r"<(?:ows:)?ExceptionText>(.*?)</(?:ows:)?ExceptionText>", text, re.S)
    return (m.group(1) if m else text).strip()[:200] or "empty answer"


#: a URLError reason that means the host is refused or unknown (not a transient failure: never retried)
_REFUSED = ("tunnel connection failed", "proxy", "403", "407", "name or service not known",
            "temporary failure in name resolution", "nodename nor servname")


def _refused(exc: BaseException) -> bool:
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (401, 403, 407)
    return isinstance(exc, urllib.error.URLError) and any(t in str(exc.reason).lower() for t in _REFUSED)


def http_get(url: str, timeout: float = 120.0, retries: int = 3, opener: Callable | None = None) -> bytes:
    """GET ``url``.  A refusal (HTTP 401/403/407, a proxy's tunnel refusal, an unknown host) raises NetworkBlocked
    at once; 5xx answers, timeouts and dropped connections are retried with backoff, then raise RuntimeError; any
    other HTTP error raises RuntimeError with the server's ExceptionText."""
    import http.client
    import urllib.error
    import urllib.request

    from sparc.core.climate import retry_wait

    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-lstair/1.0"})
    for attempt in range(retries + 1):
        try:
            with opener(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as exc:
            if _refused(exc):
                raise NetworkBlocked(HOST, str(getattr(exc, "reason", exc))[:160], PURPOSE, REMEDY) from exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                try:
                    body = exc.read()
                except Exception:                     # noqa: BLE001 - the status alone is the reason
                    body = b""
                raise RuntimeError(f"{HOST} answered HTTP {exc.code} for {url}: {_exception_text(body)}") from exc
            if attempt == retries:
                raise RuntimeError(f"{HOST}: {url} failed {retries + 1} times ({exc!r:.160})") from exc
            retry_wait(url, attempt, 2.0 * 2 ** attempt, repr(exc)[:200])
    raise AssertionError("unreachable")


def decode_geotiff(data: bytes, url: str = "") -> tuple[np.ndarray, object]:
    """(percent as float, NaN where no data; affine transform) of a WCS GeoTIFF answer in EPSG:5070.

    Values above :data:`VALID_MAX` are NaN.  The declared nodata (0 on the MRLC server: a valid percent) is ignored.
    An answer that is not a GeoTIFF (an OWS ExceptionReport) raises RuntimeError with the server's text."""
    from rasterio.io import MemoryFile

    if not data or data[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        raise RuntimeError(f"{HOST} sent no GeoTIFF for {url}: {_exception_text(data)}")
    with MemoryFile(data) as mf, mf.open() as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            raise RuntimeError(f"{url}: the GeoTIFF is in {src.crs}, not {NLCD_CRS}")
        a = src.read(1).astype(np.float64)
        T = src.transform
    a[(a < 0) | (a > VALID_MAX)] = np.nan
    return a, T


def nlcd_window(ctx: CityGrid, margin_px: int = MARGIN_PX) -> tuple[int, int, int, int]:
    """(col, row, width, height) of the NLCD pixels covering the city grid plus ``margin_px``, on the CONUS grid
    (row 0 at the top).  The grid's outline is densified before it is transformed: the image of a rectangle's
    edges bounds the image of the rectangle."""
    from pyproj import Transformer

    h = 0.5 * ctx.dx
    xs = ctx.x0 - h + np.linspace(0.0, ctx.nx * ctx.dx, 41)
    ys = ctx.y0 - h + np.linspace(0.0, ctx.ny * ctx.dx, 41)
    ex = np.concatenate([xs, np.full(ys.size, xs[-1]), xs[::-1], np.full(ys.size, xs[0])])
    ey = np.concatenate([np.full(xs.size, ys[0]), ys, np.full(xs.size, ys[-1]), ys[::-1]])
    X, Y = Transformer.from_crs(ctx.crs, NLCD_CRS, always_xy=True).transform(ex, ey)
    ox, oy = NLCD_ORIGIN
    c0 = math.floor((float(np.min(X)) - ox) / NLCD_RES) - margin_px
    c1 = math.ceil((float(np.max(X)) - ox) / NLCD_RES) + margin_px
    r0 = math.floor((oy - float(np.max(Y))) / NLCD_RES) - margin_px
    r1 = math.ceil((oy - float(np.min(Y))) / NLCD_RES) + margin_px
    return c0, r0, c1 - c0, r1 - r0


def read_layer(coverage: str, window: tuple[int, int, int, int], fetch: Callable[[str], bytes] | None = None,
               tile_px: int = MAX_TILE_PX) -> tuple[np.ndarray, object, list[str]]:
    """The NLCD pixels of ``window`` (:func:`nlcd_window`) of ``coverage``: (percent, NaN where no data; its
    EPSG:5070 transform; the URLs read).  ``fetch(url) → bytes`` is the HTTP GET (default :func:`http_get`).
    Each tile is placed by the transform it came back with; one off the NLCD pixel grid raises."""
    from affine import Affine

    fetch = fetch or http_get
    c0, r0, w, h = window
    ox, oy = NLCD_ORIGIN
    out = np.full((h, w), np.nan)
    q = 0.25 * NLCD_RES
    urls = []
    for tr in range(0, h, tile_px):
        for tc in range(0, w, tile_px):
            tw, th = min(tile_px, w - tc), min(tile_px, h - tr)
            x0 = ox + (c0 + tc) * NLCD_RES
            y1 = oy - (r0 + tr) * NLCD_RES
            url = wcs_url(coverage, x0 + q, x0 + tw * NLCD_RES - q, y1 - th * NLCD_RES + q, y1 - q)
            a, T = decode_geotiff(fetch(url), url)
            col, row = (T.c - ox) / NLCD_RES - c0, (oy - T.f) / NLCD_RES - r0
            if (abs(T.a - NLCD_RES) > 1e-6 or abs(T.e + NLCD_RES) > 1e-6 or abs(T.b) > 1e-9 or abs(T.d) > 1e-9
                    or abs(col - round(col)) > 1e-3 or abs(row - round(row)) > 1e-3):
                raise RuntimeError(f"{url}: the tile is not on the NLCD 30 m pixel grid ({tuple(T)[:6]})")
            col, row = int(round(col)), int(round(row))
            ra, rb = max(row, 0), min(row + a.shape[0], h)
            ca, cb = max(col, 0), min(col + a.shape[1], w)
            if rb > ra and cb > ca:
                out[ra:rb, ca:cb] = a[ra - row:rb - row, ca - col:cb - col]
            urls.append(url)
    return out, Affine(NLCD_RES, 0.0, ox + c0 * NLCD_RES, 0.0, -NLCD_RES, oy - r0 * NLCD_RES), urls


# --------------------------------------------------------------------------- #
# One city                                                                     #
# --------------------------------------------------------------------------- #
def landcover_rasters(ctx: CityGrid, campaign_date: dt.date, fetch: Callable[[str], bytes] | None = None,
                      tile_px: int = MAX_TILE_PX) -> tuple[dict[str, np.ndarray], dict]:
    """``tcc`` and ``nlcd_imp`` (%) on the city grid (grid orientation: row = iy) and their focal means at
    :data:`~sparc.core.lstair.dataset.FOCAL_RADII`, with provenance (module docstring, Method 1–3)."""
    from sparc.data.collect.landsat import reproject_array

    win = nlcd_window(ctx)
    out: dict[str, np.ndarray] = {}
    prov: dict = {"source": "MRLC GeoServer WCS 2.0.1 GetCoverage (GeoTIFF, native values)", "wcs": WCS,
                  "source_crs": NLCD_CRS, "window_px": list(win),
                  "resampling": "area-weighted average of the 30 m Albers pixels onto the 30 m city grid",
                  "no_data": f"values above {VALID_MAX} (impervious 127; TCC 254, 255) are NaN, never filled",
                  "year_rule": "the layer's year nearest the campaign date (1 July nominal; ties: the earlier)",
                  "layers": {}}
    for k, (pattern, years, product) in LAYERS.items():
        year = nearest_year(campaign_date, years)
        cov = pattern.format(year=year)
        arr, T, urls = read_layer(cov, win, fetch, tile_px)
        on_grid = ctx.from_northup(reproject_array(arr, T, NLCD_CRS, ctx.target, "average"))
        out[k] = on_grid
        for r in FOCAL_RADII:
            out[f"{k}_f{r}"] = focal_mean(on_grid, r, ctx.dx)
        prov["layers"][k] = {"product": product, "coverage_id": cov, "year": year, "years_served": list(years),
                             "replaces": REPLACES[k], "urls": urls,
                             "grid_share": float(np.isfinite(on_grid).mean()),
                             "window_no_data_share": float(np.isnan(arr).mean())}
    return out, prov


def landcover_key(grid: dict, campaign_date: dt.date) -> str:
    """What a city's land-cover columns depend on: the method, the source, the grid and the coverage ids."""
    parts = [METHOD, WCS, grid, coverage_ids(campaign_date), list(FOCAL_RADII), MARGIN_PX]
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def has_landcover(city_dir: str | Path, meta: dict | None = None) -> bool:
    """Whether the city's files hold every land-cover column from the current key (the command would skip it)."""
    import pyarrow.parquet as pq

    d = Path(city_dir)
    meta = meta if meta is not None else json.loads((d / "meta.json").read_text(encoding="utf-8"))
    lc = meta.get("landcover_us") or {}
    try:
        if lc.get("key") != landcover_key(meta["grid"], dt.date.fromisoformat(meta["campaign_date"])):
            return False
        names = set(pq.read_schema(d / "table.parquet").names)
        with np.load(d / "rasters.npz", allow_pickle=False) as z:
            keys = set(z.files)
    except (OSError, ValueError, KeyError):
        return False
    return all(c in names and c in keys for c in COLUMNS)


def _atomic(path: Path, write: Callable) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        write(f)
    tmp.replace(path)


def _driven_stats(table: pd.DataFrame) -> dict:
    """Per layer, on the driven cells (each cell once): coverage share, share exactly 0 (of the layer and of the
    WorldCover predictor it replaces) and the Pearson r between them."""
    u = table.drop_duplicates("cell")
    out = {"n_cells": int(len(u))}
    for k, wc in REPLACES.items():
        a = u[k].to_numpy(float)
        rec = {"share_covered": float(np.isfinite(a).mean()) if a.size else float("nan"),
               "share_zero": float(np.mean(a[np.isfinite(a)] == 0.0)) if np.isfinite(a).any() else None,
               "mean": float(np.nanmean(a)) if np.isfinite(a).any() else None}
        if wc in u:
            b = u[wc].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            rec[f"{wc}_share_zero"] = float(np.mean(b[np.isfinite(b)] == 0.0)) if np.isfinite(b).any() else None
            rec[f"r_with_{wc}"] = (float(np.corrcoef(a[ok], b[ok])[0, 1])
                                   if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else None)
        out[k] = rec
    return out


def add_landcover(city_dir: str | Path, fetch: Callable[[str], bytes] | None = None, force: bool = False,
                  tile_px: int = MAX_TILE_PX) -> dict:
    """Add the continuous US land cover to one finished city in place; returns ``{"status": "added" | "present",
    ...}`` (``present``: the files already hold it under the current key, and ``force`` is off).

    ``table.parquet`` gains (or has replaced) the :data:`COLUMNS`, sampled at each row's cell; ``rasters.npz`` the
    same keys (float32, as the other rasters), every other column and array kept as it was; ``meta.json``
    ``landcover_us`` (provenance and the key) and the new raster names.  Nothing is written before both layers are
    read; ``meta.json`` is written last."""
    d = Path(city_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        raise ValueError(f"{d.name}: not a finished city (meta.json status {meta.get('status')!r})")
    day = dt.date.fromisoformat(meta["campaign_date"])
    if not force and has_landcover(d, meta):
        lc = meta["landcover_us"]
        return {"status": "present", "years": {k: v["year"] for k, v in lc["layers"].items()}}
    t0 = time.perf_counter()
    ctx = CityGrid(**meta["grid"])
    rasters, prov = landcover_rasters(ctx, day, fetch, tile_px)
    table = pd.read_parquet(d / "table.parquet")
    iy, ix = np.divmod(table["cell"].to_numpy(np.int64), ctx.nx)
    for c in COLUMNS:
        table[c] = rasters[c][iy, ix].astype(np.float64)
    arrays = {}
    if (d / "rasters.npz").exists():
        with np.load(d / "rasters.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    arrays.update({c: rasters[c].astype(np.float32) for c in COLUMNS})
    meta["landcover_us"] = {**prov, "key": landcover_key(meta["grid"], day), "campaign_date": day.isoformat(),
                            "columns": list(COLUMNS), "focal_radii_m": list(FOCAL_RADII),
                            "driven": _driven_stats(table),
                            "seconds": round(time.perf_counter() - t0, 1),
                            "added_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    meta["rasters"] = sorted(set(meta.get("rasters") or []) | set(COLUMNS))
    _atomic(d / "table.parquet", lambda f: table.to_parquet(f, index=False))
    _atomic(d / "rasters.npz", lambda f: np.savez_compressed(f, **arrays))
    _atomic(d / "meta.json", lambda f: f.write(to_json(meta).encode()))
    return {"status": "added", "years": {k: v["year"] for k, v in prov["layers"].items()},
            "driven": meta["landcover_us"]["driven"], "seconds": meta["landcover_us"]["seconds"]}


# --------------------------------------------------------------------------- #
# Every city                                                                   #
# --------------------------------------------------------------------------- #
def _select(dirs: list[Path], names: list[str]) -> list[Path]:
    """The city folders ``names`` designate: a slug, a catalog name ("Farmville, VA") or the name before the state
    ("Farmville"); an unknown or ambiguous name raises ValueError."""
    out = []
    for n in names:
        full, short = slugify(n), slugify(n.split(",")[0])
        hit = [p for p in dirs if p.name == full] or [p for p in dirs if p.name.rsplit("_", 1)[0] == short]
        if len(hit) != 1:
            raise ValueError(f"city {n!r}: " + ("no city folder of that name" if not hit else
                                                "ambiguous (" + ", ".join(p.name for p in hit) + ")"))
        if hit[0] not in out:
            out.append(hit[0])
    return out


def add_dataset(data_dir: str | Path, cities: list[str] | None = None, *, force: bool = False,
                fetch: Callable[[str], bytes] | None = None, probe_opener: Callable | None = None,
                tile_px: int = MAX_TILE_PX) -> dict:
    """:func:`add_landcover` on every finished city under ``data_dir`` (the ``--out`` of ``lstair dataset``, or its
    ``cities`` folder), or on those of ``cities``; returns the summary also written to ``data_dir/landcover_us.json``.

    Before anything is fetched, the WCS host is probed (when ``fetch`` is not given and some city needs it): a
    refusal raises NetworkBlocked naming it.  A city that fails for another reason is recorded with its error and
    left as it was; the others go on.  Cities not finished by ``lstair dataset`` are listed, never touched."""
    root = Path(data_dir)
    cdir = root / "cities" if (root / "cities").is_dir() else root
    dirs = sorted(p for p in cdir.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if cities:
        dirs = _select(dirs, cities)
    metas = {}
    for p in dirs:
        try:
            metas[p.name] = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            metas[p.name] = {"status": f"unreadable meta.json ({exc})"}
    todo = [p for p in dirs if metas[p.name].get("status") == "ok" and (force or not has_landcover(p, metas[p.name]))]
    summary: dict = {"schema": 1, "source": WCS, "cities": {}}
    if todo and fetch is None:
        check_host(probe_opener)
        summary["probe"] = "ok"
    for p in dirs:
        m = metas[p.name]
        if m.get("status") != "ok":
            summary["cities"][p.name] = {"status": "not built", "error": f"meta.json status {m.get('status')!r}"}
            continue
        try:
            rec = add_landcover(p, fetch=fetch, force=force, tile_px=tile_px)
        except NetworkBlocked:
            raise
        except Exception as exc:                      # noqa: BLE001 - one city's failure is recorded
            msg = str(exc).splitlines()[0][:300] if str(exc) else repr(exc)[:300]
            log.warning("lstair add-landcover: %s failed: %s", p.name, msg)
            rec = {"status": "failed", "error": msg}
        else:
            log.info("lstair add-landcover: %s %s (%s)", p.name, rec["status"],
                     ", ".join(f"{k} {y}" for k, y in rec["years"].items()))
        summary["cities"][p.name] = rec
    _atomic(root / "landcover_us.json", lambda f: f.write(to_json(summary).encode()))
    return summary


__all__ = ["LAYERS", "COLUMNS", "REPLACES", "WCS", "HOST", "nearest_year", "coverage_ids", "wcs_url", "check_host",
           "http_get", "decode_geotiff", "nlcd_window", "read_layer", "landcover_rasters", "landcover_key",
           "has_landcover", "add_landcover", "add_dataset"]
