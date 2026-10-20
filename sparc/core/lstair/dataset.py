"""The multi-city traverse dataset: one row per (city, run, 30 m cell) a Heat Watch vehicle drove.

For every city (a CAPA catalog name, or a local folder given by the user):

1. **Traverses.**  The campaign folder is read as published (:func:`sparc.core.identify.traverses.scan_traverses`:
   CSV / shapefile / GeoJSON / zip, nested archives, CAPA's local clock and sensor ids), split into runs (am, af,
   pm, ni by local clock) and, per run, into passes collapsed to one sample per visit to a 30 m cell
   (:func:`~sparc.core.identify.traverses.to_campaign`).  The folder comes from ``--traverses-dir`` or is
   downloaded from OSF (:func:`sparc.data.collect.capa.download_capa_files`) into the cache.
2. **Grid.**  A 30 m grid in the city's UTM zone (catalog ``utm_epsg``, else the zone of the points' centre),
   cell edges on multiples of 30 m, covering the traverses plus a margin (default 2 km: the 1 km focal means
   and the physics model's source domain).
3. **Drift and vehicles.**  Within a run the air warms or cools and each vehicle's sensor has an offset.  As
   :mod:`sparc.core.identify`'s levels design does, a time polynomial is fitted with a level per vehicle, here
   against **cell fixed effects** instead of land-cover terms, so no predictor enters the target:
   ``T_visit = μ_cell + β·τ (+ β₂·τ²) + o_vehicle + ε`` (τ in hours from the run's median time).  Drift is
   identified only from cells revisited at different times.  Real Heat Watch routes rarely give that: each
   vehicle drives its own route and comes back to a cell within minutes (Providence: every repeat within 4 min).
   So the fixed-effect drift is used only when at least 20 cells were revisited 15 min or more apart, in at
   least 10 blocks of 300 m and, with two or more vehicles, not more than 80% of them one vehicle's, and its
   standard error (clustered by block) is below 0.3 °C/h (:func:`adjust_visits`); otherwise the run takes CAPA's
   protocol line, a polynomial in time fitted within vehicles (a level each, so that routes ending early do not
   pass for warming), with the reason and the drift estimates in ``meta.json``.  Vehicle offsets come only from
   cells two vehicles visited within 10 min of each other (at least 5 such cells; :func:`co_visit_offsets`) and
   are centred (visit-weighted mean 0) within each group so linked; a vehicle without such a link keeps 0.
   Vehicles are sensors: one sensor's logger files of a run are one vehicle (:func:`sensor_name`).
4. **Target.**  ``Ta`` (°C) is the cell mean of the adjusted visits of the run; ``dTa = Ta − background`` with the
   background the run's median ``Ta`` over its cells.  ``ta_era5`` is ERA5's 2 m temperature at the city centre,
   time-weighted over the run's span (:func:`era5_run`, :func:`sparc.core.forcing.era5_site`), when available.
   **QA** (the review of the 31 built cities, 2026-10-20): before the grid is laid, the readings of bicycle and
   walking sensors and those the QA registry lists (:mod:`~sparc.core.lstair.exclusions`) are dropped; each run's
   ``qa_exclude`` ("" when usable; a ``table.parquet`` column and ``meta["runs"]``) says why it is left out by
   default: the registry, a walked traverse (median speed below 10 km/h) or a single vehicle with the protocol
   line (:func:`build_run`).
5. **Predictors** (rasters on the city grid, sampled at the cells): canopy and impervious (%), NDVI, albedo,
   elevation (m) and distance to water (m) from :mod:`sparc.core.features_open` (WorldCover, Sentinel-2,
   Copernicus DEM), with Gaussian focal means (σ = radius/2, as the core's focal features) at 100, 300 and
   1000 m; LST (°C) of the Landsat 8/9 scene nearest the campaign date that is ≥ 80% clear over the study
   area (else the clearest ≥ 50%)
   (:func:`sparc.data.collect.landsat.campaign_lst`) with focal means at 100 and 300 m, the scene date and its
   offset in days.  Predictors are computed from rasters only; the traverse temperatures never enter them.

While a city builds, each source's result (features, LST, ERA5 per run) is kept in ``OUT/cities/<slug>/partial/``
as it finishes, so a build cut off by a container restart resumes there.  Each city is cached in
``OUT/cities/<slug>/``: ``table.parquet``, ``rasters.npz`` (every raster on the full
grid: the physics model needs features beyond the driven cells) and ``meta.json`` (provenance, runs, drift,
scenes), written last, so a city with ``meta.json`` status ``ok`` is skipped on a rerun unless its settings, the
QA rules (:data:`DATASET_VERSION`) or its QA-registry entries changed; a rebuild on the same grid reuses the
previous build's feature rasters (:func:`reusable_features`).  Hosts are probed
before anything is fetched (for Landsat: the STAC API, the SAS token and a signed listing of the COG container);
a blocked host stops the build at once with its name (:class:`NetworkBlocked`).  A Landsat read that fails for
another reason fails the city (retried on the next run): NaN LST in a finished city means only that every
inspected scene was read and none was clear enough.

References: Shandas et al. (2019), *Climate* 7(1), 5, doi:10.3390/cli7010005 (the Heat Watch method: traverses
plus satellite predictors and machine learning); Voogt & Oke (2003), *RSE* 86(3), 370–384,
doi:10.1016/S0034-4257(03)00079-8.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import math
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from sparc.core.lstair import RUN_CODE, to_json

log = logging.getLogger(__name__)

CELL_M = 30.0
MARGIN_M = 2000.0
FEATURES = ("canopy", "impervious", "ndvi", "albedo", "elevation", "water_distance")
FOCAL_FEATURES = ("canopy", "impervious", "ndvi", "albedo")
FOCAL_RADII = (100, 300, 1000)
LST_RADII = (100, 300)
MAX_CELLS = 6_000_000                  # ~70 km square at 30 m: larger grids are refused
WARN_CELLS = 1_500_000                 # ~37 km square: the open features take long (a warning in meta.json)
STUDY_BUFFER_M = 300.0                 # LST clear fraction is measured within this distance of the traverses
TARGET_COLUMNS = ("Ta", "dTa", "ta_background", "n_readings", "n_visits")
#: File-name patterns not read by default: CAPA's appendix archives repeat the main traverses (Providence wu9v7:
#: the afternoon and evening runs again, plus three more sensors far outside the study area); stationary-sensor
#: tables (Columbia SC, Clark County NV: fixed loggers, a time series at a few points, not traverses); mobile
#: air-quality tables (the Oklahoma City node also holds Bloomington IN's, another city and instrument).
SKIP_FILES = ("appendix", "stationary", "air quality")
SCHEMA = 1

#: IANA zone by state (for converting the traverses' local clock to UTC hours for ERA5), with city overrides.
STATE_TZ = {**{s: "America/New_York" for s in ("VT", "MA", "RI", "NY", "NJ", "PA", "DE", "MD", "VA", "NC", "SC",
                                                "GA", "FL", "KY", "CT", "NH", "ME", "DC", "OH", "MI", "IN", "WV")},
            **{s: "America/Chicago" for s in ("MS", "LA", "TX", "OK", "IL", "WI", "MO", "KS", "IA", "NE", "MN",
                                               "AR", "AL", "TN")},
            **{s: "America/Denver" for s in ("CO", "UT", "NM", "MT", "WY")},
            "AZ": "America/Phoenix", "ID": "America/Boise", "AK": "America/Anchorage",
            **{s: "America/Los_Angeles" for s in ("NV", "CA", "OR", "WA")}}
CITY_TZ = {"El Paso, TX": "America/Denver"}


# --------------------------------------------------------------------------- #
# Network                                                                      #
# --------------------------------------------------------------------------- #
class NetworkBlocked(RuntimeError):
    """A host the build needs cannot be reached; ``host`` names it."""

    def __init__(self, host: str, reason: str, purpose: str, remedy: str = ""):
        self.host, self.reason, self.purpose = host, reason, purpose
        if re.match(r"^HTTP 5\d\d", str(reason)):          # the host answered: a server error, not a block
            super().__init__(f"{host} answered {reason} (a server error, not a network block: try again later): "
                             f"it serves {purpose}.")
            return
        super().__init__(f"cannot reach {host} ({reason}): it serves {purpose}." + (f" {remedy}" if remedy else ""))


def _probe_landsat_blob(opener: Callable | None = None, timeout: float = 10.0) -> tuple[str, str | None]:
    """A signed listing of the Landsat container: the SAS token from Planetary Computer, then one blob listed
    on Azure (what reading a COG needs); (host that failed or the blob host, reason or None)."""
    import urllib.request

    from sparc.data.collect.landsat import PC_BLOB, PC_SAS_TOKEN

    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(PC_SAS_TOKEN, headers={"User-Agent": "SPARC-lstair/1.0"})
    try:
        with opener(req, timeout=timeout) as r:
            token = json.loads(r.read().decode("utf-8"))["token"]
    except Exception as exc:                          # noqa: BLE001 - any failure is the reason
        return "planetarycomputer.microsoft.com", probe_reason(exc)
    why = probe(f"{PC_BLOB}?restype=container&comp=list&maxresults=1&{token}", timeout=timeout, opener=opener)
    return "landsateuwest.blob.core.windows.net", why


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):              # a 3xx surfaces as HTTPError: the probe follows it itself
        return None


def _probe_osf_download(opener: Callable | None = None, timeout: float = 10.0, node: str = "wu9v7"
                        ) -> tuple[str, str | None]:
    """A real OSF file download: the smallest file of ``node`` listed on api.osf.io, fetched from files.osf.io
    with its redirect to OSF's object storage followed hop by hop, so the host that fails is the one named."""
    import urllib.error
    import urllib.request
    from urllib.parse import urljoin, urlparse

    from sparc.data.collect.capa import OSF_API, OSF_WATERBUTLER

    ua = {"User-Agent": "SPARC-lstair/1.0"}
    get = opener or urllib.request.build_opener(_NoRedirect()).open
    try:
        with get(urllib.request.Request(OSF_API.format(node_id=node), headers=ua), timeout=timeout) as r:
            items = json.loads(r.read().decode("utf-8")).get("data", [])
        files = [i for i in items if i["attributes"].get("kind") == "file"]
        fid = min(files, key=lambda i: i["attributes"].get("size") or 0)["id"]
    except Exception as exc:                          # noqa: BLE001 - any failure is the reason
        return "api.osf.io", probe_reason(exc)
    url = OSF_WATERBUTLER.format(node_id=node, file_id=fid)
    for _ in range(6):
        host = urlparse(url).hostname or url
        try:
            with get(urllib.request.Request(url, headers={**ua, "Range": "bytes=0-0"}), timeout=timeout) as r:
                code = int(getattr(r, "status", None) or 200)
                return host, None if code < 400 else f"HTTP {code}"
        except urllib.error.HTTPError as exc:
            loc = exc.headers.get("Location") if exc.headers is not None else None
            if exc.code in (301, 302, 303, 307, 308) and loc:
                url = urljoin(url, loc)
                continue
            return host, probe_reason(exc)
        except Exception as exc:                      # noqa: BLE001
            return host, probe_reason(exc)
    return host, "too many redirects"


#: name → (probes, host(s) the real run needs, what they serve, remedy).  A probe is a URL that answers HTTP < 400
#: when reachable (an API root or a public-bucket listing; the failing URL's host is named) or a callable
#: ``(opener, timeout) → (host, reason or None)``.
HOSTS = {
    "osf": (("https://api.osf.io/v2/", _probe_osf_download),
            ("api.osf.io", "files.osf.io", "idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com"),
            "the CAPA Heat Watch traverse files (OSF API listing; downloads from files.osf.io, which redirects them "
            "to OSF's object storage on Oracle Cloud)",
            "Allow api.osf.io, files.osf.io and idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com (or "
            "*.objectstorage.us-ashburn-1.oraclecloud.com), or download the campaign folders and pass "
            "--traverses-dir."),
    "landsat": (("https://planetarycomputer.microsoft.com/api/stac/v1/", _probe_landsat_blob),
                ("planetarycomputer.microsoft.com", "landsateuwest.blob.core.windows.net"),
                "Landsat Collection 2 Level-2 scenes (STAC search and SAS-token signing; COGs on Azure blob storage)",
                "Allow planetarycomputer.microsoft.com and landsateuwest.blob.core.windows.net, or pass --no-lst."),
    "worldcover": (("https://esa-worldcover.s3.amazonaws.com/?list-type=2&max-keys=1",), ("esa-worldcover.s3.amazonaws.com",),
                   "ESA WorldCover 2021 (canopy, impervious, water)", ""),
    "dem": (("https://copernicus-dem-30m.s3.amazonaws.com/?list-type=2&max-keys=1",),
            ("copernicus-dem-30m.s3.amazonaws.com",), "the Copernicus GLO-30 DEM (elevation)", ""),
    "sentinel2": (("https://sentinel-cogs.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=1",),
                  ("sentinel-cogs.s3.us-west-2.amazonaws.com",), "Sentinel-2 L2A COGs (NDVI, albedo)", ""),
    "era5": (("https://nsf-ncar-era5.s3.amazonaws.com/?list-type=2&max-keys=1",), ("nsf-ncar-era5.s3.amazonaws.com",),
             "ERA5 2 m temperature and radiation (NSF NCAR mirror)", "Or pass --no-era5."),
}


def probe(url: str, timeout: float = 10.0, opener: Callable | None = None) -> str | None:
    """None when ``url`` answers with HTTP < 400, else the reason (an HTTP status or the connection error)."""
    import urllib.error
    import urllib.request

    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-lstair/1.0", "Range": "bytes=0-1023"})
    try:
        with opener(req, timeout=timeout) as r:
            code = int(getattr(r, "status", None) or r.getcode() or 200)
            return None if code < 400 else f"HTTP {code}"
    except Exception as exc:                          # noqa: BLE001 - URLError, timeouts, TLS, resets
        return probe_reason(exc)


def probe_reason(exc: BaseException) -> str:
    """A short reason for a failed request: ``HTTP <code>`` or the connection error."""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    reason = getattr(exc, "reason", None) or exc
    return f"{type(exc).__name__}: {str(reason)[:160]}"


def hosts_needed(download_traverses: bool, features: bool, lst: bool, era5: bool) -> list[str]:
    """The :data:`HOSTS` keys a build with these sources needs."""
    out = []
    if download_traverses:
        out.append("osf")
    if features:
        out += ["worldcover", "dem", "sentinel2"]
    if lst:
        out.append("landsat")
    if era5:
        out.append("era5")
    return out


def run_probe(pr, opener: Callable | None = None, timeout: float = 10.0) -> tuple[str, str | None]:
    """(host, reason or None) of one :data:`HOSTS` probe."""
    from urllib.parse import urlparse

    if callable(pr):
        return pr(opener, timeout)
    return urlparse(pr).hostname or str(pr), probe(pr, timeout=timeout, opener=opener)


def check_hosts(names: list[str], opener: Callable | None = None, timeout: float = 10.0) -> dict:
    """Run every probe of every source in ``names``; raise :class:`NetworkBlocked` naming the host of the first
    probe that fails."""
    out = {}
    for name in names:
        probes, hosts, purpose, remedy = HOSTS[name]
        for pr in probes:
            host, why = run_probe(pr, opener, timeout)
            if why:
                out[name] = why
                raise NetworkBlocked(host, why, purpose, remedy)
        out[name] = "ok"
    return out


def probe_label(pr) -> str:
    return pr if isinstance(pr, str) else (pr.__doc__ or pr.__name__).strip().splitlines()[0]


def required_host_names() -> list[str]:
    """Every host a full real run needs (traverses from OSF, open-data features, Landsat, ERA5)."""
    return [h for k in ("osf", "worldcover", "dem", "sentinel2", "landsat", "era5") for h in HOSTS[k][1]]


# --------------------------------------------------------------------------- #
# Cities                                                                       #
# --------------------------------------------------------------------------- #
def slugify(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", name.lower())).strip("_")


@dataclass
class CitySpec:
    name: str
    slug: str
    node: str | None = None
    folder_hint: str | None = None
    crs: str | None = None
    tz: str | None = None
    catalog_date: str | None = None
    catalog_date_source: str | None = None
    availability: dict = field(default_factory=dict)


def city_spec(name: str) -> CitySpec:
    """The catalog entry of ``name`` (fuzzy match), or a bare spec for a city not in the catalog."""
    from sparc.data.collect import capa_catalog as C

    canonical, entry = C.lookup_city(name)
    if canonical is None:
        if isinstance(entry, dict) and (entry.get("ambiguous") or entry.get("suggestions")):
            raise ValueError(f"city {name!r} is ambiguous: {entry.get('ambiguous') or entry.get('suggestions')}")
        return CitySpec(name=name, slug=slugify(name))
    node, hint = C.traverse_node(entry)
    d, src = C.campaign_date(entry)
    return CitySpec(name=canonical, slug=slugify(canonical), node=node, folder_hint=hint, crs=entry.get("utm_epsg"),
                    tz=CITY_TZ.get(canonical) or STATE_TZ.get(entry.get("state", "")), catalog_date=d,
                    catalog_date_source=src, availability=C.availability(entry))


def all_catalog_cities() -> list[str]:
    from sparc.data.collect.capa_catalog import list_cities

    return list_cities()


# --------------------------------------------------------------------------- #
# The city grid                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class CityGrid:
    """A full rectangle of ``cell_m`` cells in ``crs``; cell (iy, ix) is centred at (x0 + ix·dx, y0 + iy·dx) and
    has id iy·nx + ix (:class:`sparc.core.grid.Grid` orientation: row index grows with y)."""

    crs: str
    x0: float
    y0: float
    dx: float
    nx: int
    ny: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def grid(self):
        from sparc.core.grid import Grid

        if getattr(self, "_grid", None) is None:
            iy, ix = np.divmod(np.arange(self.nx * self.ny, dtype=np.int64), self.nx)
            ones = np.ones(self.shape, dtype=np.int64)
            self._grid = Grid(x0=self.x0, y0=self.y0, dx=self.dx, dy=self.dx, nx=self.nx, ny=self.ny, ix=ix, iy=iy,
                              mask=ones > 0, counts=ones)
        return self._grid

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates (ny, nx) in ``crs``."""
        return np.meshgrid(self.x0 + np.arange(self.nx) * self.dx, self.y0 + np.arange(self.ny) * self.dx)

    def cell_xy(self, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        iy, ix = np.divmod(np.asarray(cell, dtype=np.int64), self.nx)
        return self.x0 + ix * self.dx, self.y0 + iy * self.dx

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        from pyproj import Transformer

        lon, lat = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(np.asarray(x, float),
                                                                                          np.asarray(y, float))
        return np.asarray(lon, float), np.asarray(lat, float)

    @property
    def target(self) -> dict:
        """The grid as a north-up raster georeference (``transform``, ``crs``, ``shape``) for reprojection."""
        from affine import Affine

        top = self.y0 + (self.ny - 0.5) * self.dx
        return {"transform": Affine(self.dx, 0.0, self.x0 - 0.5 * self.dx, 0.0, -self.dx, top), "crs": self.crs,
                "shape": self.shape}

    @staticmethod
    def from_northup(arr: np.ndarray) -> np.ndarray:
        """A north-up raster on :attr:`target` → grid orientation (row = iy)."""
        return np.asarray(arr)[::-1]

    to_northup = from_northup

    def bounds_lonlat(self, margin_deg: float = 0.0) -> tuple[float, float, float, float]:
        xs = np.array([self.x0, self.x0 + (self.nx - 1) * self.dx])
        ys = np.array([self.y0, self.y0 + (self.ny - 1) * self.dx])
        X, Y = np.meshgrid(np.linspace(xs[0], xs[1], 7), np.linspace(ys[0], ys[1], 7))
        lon, lat = self.to_lonlat(X.ravel(), Y.ravel())
        return (float(lon.min()) - margin_deg, float(lat.min()) - margin_deg, float(lon.max()) + margin_deg,
                float(lat.max()) + margin_deg)

    def as_dict(self) -> dict:
        return {"crs": self.crs, "x0": self.x0, "y0": self.y0, "dx": self.dx, "nx": self.nx, "ny": self.ny}

    # identify's to_campaign / features_open take a (cfg, data/layout) pair: the minimal stand-ins
    def identify_cfg(self):
        return SimpleNamespace(data={"reproject_to": self.crs}, coord_scale=1.0)

    def identify_layout(self):
        return SimpleNamespace(grid=self.grid)

    def points_data(self):
        """A ``data`` stand-in whose points are every cell (for :mod:`sparc.core.features_open`)."""
        X, Y = self.centres()
        return SimpleNamespace(ids=np.arange(self.nx * self.ny), x=X.ravel(), y_coord=Y.ravel(), grid=self.grid)


def utm_of(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def make_city_grid(lon: np.ndarray, lat: np.ndarray, crs: str | None = None, cell_m: float = CELL_M,
                   margin_m: float = MARGIN_M) -> CityGrid:
    """The grid over traverse positions (WGS-84 degrees) plus ``margin_m``, edges on multiples of ``cell_m``."""
    from pyproj import Transformer

    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        raise ValueError("no traverse positions")
    crs = crs or utm_of(float(np.median(lon[ok])), float(np.median(lat[ok])))
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon[ok], lat[ok])
    x, y = np.asarray(x), np.asarray(y)
    c = float(cell_m)
    x_lo = math.floor((x.min() - margin_m) / c) * c + 0.5 * c
    y_lo = math.floor((y.min() - margin_m) / c) * c + 0.5 * c
    nx = int(math.ceil((x.max() + margin_m - x_lo) / c)) + 1
    ny = int(math.ceil((y.max() + margin_m - y_lo) / c)) + 1
    if nx * ny > MAX_CELLS:
        raise ValueError(f"the traverses span {nx * c / 1000:.0f} × {ny * c / 1000:.0f} km: more than {MAX_CELLS:,} "
                         "cells at this resolution (wrong positions, or several cities in one folder?)")
    return CityGrid(crs=crs, x0=x_lo, y0=y_lo, dx=c, nx=nx, ny=ny)


# --------------------------------------------------------------------------- #
# Drift and vehicle offsets                                                    #
# --------------------------------------------------------------------------- #
def _components(cells: np.ndarray, vehicles: np.ndarray) -> np.ndarray:
    """Group id of every vehicle: vehicles are linked when they visited a common cell (the rank of the cell and
    vehicle effects)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    uv, vi = np.unique(vehicles, return_inverse=True)
    uc, ci = np.unique(cells, return_inverse=True)
    n = uv.size + uc.size
    A = coo_matrix((np.ones(vi.size), (vi, uv.size + ci)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    return lab[:uv.size]


MIN_REPEAT_SPAN_S = 900.0             # a repeat visit identifies drift only when it comes back this much later
MAX_DRIFT_SE = 0.3                     # °C/h: a cell-fixed-effect drift less certain than this is not used
#: ... nor one whose spread repeats fall in fewer blocks of REPEAT_BLOCK_M than this (Fort Lauderdale 2019-09-18
#: midday: all 22 in 8 blocks of one route, fe −2.16 °C/h against +0.20 for the protocol line, a local cooling on
#: CAPA80's western route that put a +1.3 to +2.9 °C/h ramp along every other route) ...
MIN_REPEAT_BLOCKS = 10
REPEAT_BLOCK_M = 300.0
#: ... nor, with two or more vehicles, one whose spread repeats are more than this share one vehicle's (Fort
#: Lauderdale: 22 of 22 CAPA80's; New Orleans 2020-08-18 midday: 34 of 35 CAPA1050's, fe −0.97 against −0.13 °C/h
#: while every sensor's 10-minute medians stay flat)
MAX_REPEAT_SHARE = 0.8
#: °C/h: a fixed-effect drift that passes every guard but differs this much from the protocol line is kept, with a
#: warning: the protocol line has faults of its own (route composition, :func:`_protocol_line`)
DRIFT_GAP_WARN = 0.75
#: Two vehicles are linked (their offsets estimated against each other) only by cells both visited within this
#: many seconds of each other, at least MIN_CO_VISIT_CELLS such cells (:func:`co_visit_offsets`)
CO_VISIT_S = 600.0
MIN_CO_VISIT_CELLS = 5


def _protocol_line(temp: np.ndarray, P: np.ndarray, vi: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """CAPA's detrending as a line within vehicles: ``temp ~ vehicle levels + P(t)``; (the coefficients of ``P``,
    the SE of the linear one).  ``vi``: each visit's vehicle index (0 … k−1); None fits one level, the line through
    every reading.

    The vehicle levels keep route composition out of the time axis.  In Virginia Beach's afternoon run
    (2021-07-15) four eastern, oceanfront routes stop at 15:27 and the warmest inland route starts at 15:18: the
    line through every reading climbs +2.29 °C/h while the vehicles' own readings warm +1.06 °C/h, and removing
    +2.29 damped the coast–inland contrast (r(dTa, lon) −0.33 against −0.41 within vehicles).  With one vehicle,
    or vehicles that each drive the whole run, it is the line through every reading."""
    n = temp.size
    vi = np.zeros(n, dtype=np.int64) if vi is None else np.asarray(vi, dtype=np.int64)
    k = int(vi.max()) + 1 if n else 1
    D = np.zeros((n, k))
    D[np.arange(n), vi] = 1.0
    X = np.column_stack([D, P])
    b, *_ = np.linalg.lstsq(X, temp, rcond=None)
    dof = n - int(np.linalg.matrix_rank(X))
    se = float("inf")
    if dof > 0:
        s2 = float(np.sum((temp - X @ b) ** 2)) / dof
        with np.errstate(all="ignore"):
            cov = s2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[k, k], 0.0)))
    return b[k:], se


def co_visit_offsets(cells: np.ndarray, vi: np.ndarray, t: np.ndarray, resid: np.ndarray, weights: np.ndarray,
                     max_dt_s: float = CO_VISIT_S, min_cells: int = MIN_CO_VISIT_CELLS
                     ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Vehicle offsets from co-located, near-simultaneous visits only.

    Two vehicles are linked when at least ``min_cells`` cells were visited by both within ``max_dt_s`` (default
    10 min) of each other; each such cell gives one difference of the two vehicles' drift-removed temperatures
    (``resid``; the mean over the cell's visit pairs within ``max_dt_s``).  The offsets solve those differences by
    least squares within each connected group of linked vehicles and are centred there (visit-weighted mean 0,
    ``weights``: visits per vehicle index); a vehicle without such a link keeps 0.

    Offsets fitted against cell fixed effects rest on whatever cells two vehicles share, however far apart in
    time, so a link driven an hour apart reads the change of the air (or of the street) over that hour as a sensor
    bias.  Los Angeles 2019-09-15 morning: CAPA34 +1.37 and CAPA39 −1.39 °C from 19 cells of one street 52 min
    apart, which near-simultaneous visits to nearby cells contradict (the morning's offsets raised their mean
    squared mismatch within 1.5 km and 30 min from 0.31 to 1.86 °C²); Richmond 2021-07-15 evening: 1102 −1.04 °C
    from 7 cells 57 min apart.  Co-visits
    minutes apart do show real biases: Fort Lauderdale 2019-09-18 midday, CAPA70 reads 1.2 °C above CAPA67 at 14
    visit pairs within 10 min in the same cell.

    Returns (offset per vehicle index, linked-group label per vehicle index, the links: the two vehicle indices,
    the cells and the mean difference first − second)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    k = int(weights.size)
    off, lab = np.zeros(k), np.arange(k)
    df = pd.DataFrame({"cell": cells, "v": vi, "t": t, "r": resid})
    m = df[(df.groupby("cell")["v"].transform("nunique") >= 2).to_numpy()]
    pr = m.merge(m, on="cell", suffixes=("_a", "_b"))
    pr = pr[((pr["v_a"] < pr["v_b"]) & ((pr["t_a"] - pr["t_b"]).abs() <= max_dt_s)).to_numpy()]
    per = pr.assign(d=pr["r_a"] - pr["r_b"]).groupby(["v_a", "v_b", "cell"], sort=True)["d"].mean().reset_index()
    per = per[(per.groupby(["v_a", "v_b"])["d"].transform("size") >= min_cells).to_numpy()]
    if not len(per):
        return off, lab, []
    a, b, d = (per[c].to_numpy() for c in ("v_a", "v_b", "d"))
    _, lab = connected_components(coo_matrix((np.ones(a.size), (a, b)), shape=(k, k)), directed=False)
    for g in np.unique(lab):
        sel = np.flatnonzero(lab == g)
        if sel.size < 2:
            continue
        rows = np.flatnonzero(np.isin(a, sel))
        A = np.zeros((rows.size, sel.size))
        A[np.arange(rows.size), np.searchsorted(sel, a[rows])] = 1.0
        A[np.arange(rows.size), np.searchsorted(sel, b[rows])] = -1.0
        o, *_ = np.linalg.lstsq(A, d[rows], rcond=None)
        off[sel] = o - np.average(o, weights=weights[sel])
    agg = per.groupby(["v_a", "v_b"])["d"].agg(["size", "mean"])
    links = [{"vehicles": [int(x), int(y)], "n_cells": int(r["size"]), "diff_c": float(r["mean"])}
             for (x, y), r in agg.iterrows()]
    return off, lab, links


def adjust_visits(v: pd.DataFrame, method: str = "fe", degree: int = 1, min_repeat_cells: int = 20,
                  min_span_s: float = MIN_REPEAT_SPAN_S, max_drift_se: float = MAX_DRIFT_SE,
                  min_repeat_blocks: int = MIN_REPEAT_BLOCKS, max_repeat_share: float = MAX_REPEAT_SHARE,
                  co_visit_s: float = CO_VISIT_S, min_co_visit_cells: int = MIN_CO_VISIT_CELLS
                  ) -> tuple[np.ndarray, dict]:
    """Visit temperatures with the run's drift and the vehicles' offsets removed.

    ``v`` has ``cell, vehicle, t_s, temp`` (one row per visit) and, for the block guard and the clustered SE,
    ``x, y`` (the cell centre, m).  ``method``:

    * ``"fe"`` (default): the drift is a time polynomial fitted against cell fixed effects, with a level per
      vehicle so that vehicles' levels never pass for drift (module docstring), used only when the repeat visits
      identify it (below), else the protocol line; the vehicle offsets come from co-located, near-simultaneous
      visits only (:func:`co_visit_offsets`), whichever drift is used.
    * ``"protocol"``: CAPA's detrending, a polynomial in time fitted within vehicles (:func:`_protocol_line`); no
      vehicle offsets.
    * ``"none"``.

    Within cell fixed effects the drift is identified only by cells visited twice **at different times**: two
    passes a minute apart say almost nothing about the warming over an hour, and extrapolating from them puts
    an error along each route's time axis into the target (spatially structured, since time runs along the
    route).  So the fixed-effect drift is used only when (1) at least ``min_repeat_cells`` cells were revisited
    ``min_span_s`` or more apart (default 15 min), (2) those cells lie in at least ``min_repeat_blocks`` blocks of
    300 m (when ``x, y`` are given), (3) with two or more vehicles, no vehicle holds more than
    ``max_repeat_share`` of them (a cell shared among its vehicles by their visits there) and (4) the drift's
    standard error is below ``max_drift_se`` (default 0.3 °C/h): a drift seen on a few streets, or on one route,
    is that place's change, not the run's.  Otherwise the run takes the protocol line, with the reason in
    ``info["fallback"]``.  A fixed-effect drift that passes but differs from the protocol line by more than
    :data:`DRIFT_GAP_WARN` is kept with ``info["warning"]``.  The SE is the larger of the OLS one of the linear
    coefficient (Frisch–Waugh: σ̂²/‖M τ‖², with M the projection off the cell and vehicle effects and the other time
    terms; σ̂² from the fit's residuals and its degrees of freedom) and that one clustered by 300 m block (by
    vehicle without ``x, y``), √(G/(G−1) · Σ_g s_g²) / ‖M τ‖² with s_g the block's sum of the visits' scores
    (M τ)_i û_i: the repeats of one pass along one street are not independent, and each cell lies in one block
    (the larger of the conventional and the robust SE: Angrist & Pischke 2009, *Mostly Harmless Econometrics*,
    §8.1; Cameron & Miller 2015, *J. Hum. Resour.* 50(2), 317–372).

    Returns (adjusted temperatures, info) with ``info["method"]`` the method used, ``drift_per_h`` (the linear
    coefficient, temperature units per hour at the run's median time), ``drift_se_per_h``, ``drift_coef``,
    ``protocol_drift_per_h`` (the protocol line) and ``pooled_drift_per_h`` (the line through every reading),
    ``fe_drift_per_h`` when the fixed effects were fitted, the guards' evidence (``n_repeat_cells``,
    ``n_repeat_cells_spread``, ``n_repeat_blocks``, ``repeat_vehicle_share``, ``repeat_vehicle``),
    ``vehicle_offsets`` (every vehicle; 0 without a co-visit link), ``vehicle_links`` and ``n_vehicle_groups``.
    The adjustment is zero at the run's median time (τ = 0), so levels are those of mid-run."""
    temp = v["temp"].to_numpy(float)
    t = v["t_s"].to_numpy(float)
    tau = (t - np.median(t)) / 3600.0
    cells = v["cell"].to_numpy()
    veh = v["vehicle"].to_numpy()
    uv, vi = np.unique(veh, return_inverse=True)
    w = np.bincount(vi, minlength=uv.size).astype(float)
    counts = pd.Series(cells).value_counts()
    n_rep = int((counts >= 2).sum())
    gt = pd.Series(t).groupby(cells)
    span = gt.max() - gt.min()
    spread = span.index[(span >= float(min_span_s)).to_numpy()]
    n_spread = int(spread.size)
    blk = None
    if "x" in v and "y" in v:
        bx = np.floor(v["x"].to_numpy(float) / REPEAT_BLOCK_M).astype(np.int64)
        by = np.floor(v["y"].to_numpy(float) / REPEAT_BLOCK_M).astype(np.int64)
        blk = pd.factorize(pd.MultiIndex.from_arrays([bx, by]))[0] if len(v) else np.zeros(0, np.int64)
    in_spread = np.isin(cells, spread)
    n_blocks = None if blk is None else int(np.unique(blk[in_spread]).size)
    share, top = 0.0, None
    if n_spread:
        nv_c = pd.DataFrame({"c": cells[in_spread], "v": vi[in_spread]}).groupby(["c", "v"]).size()
        by_v = (nv_c / nv_c.groupby(level="c").transform("sum")).groupby(level="v").sum() / n_spread
        share, top = float(by_v.max()), str(uv[int(by_v.idxmax())])
    info: dict = {"requested": method, "degree": int(degree), "n_visits": int(temp.size), "n_cells": int(counts.size),
                  "n_repeat_cells": n_rep, "n_repeat_cells_spread": n_spread, "n_repeat_blocks": n_blocks,
                  "repeat_vehicle_share": share, "repeat_vehicle": top, "min_span_s": float(min_span_s),
                  "max_drift_se": float(max_drift_se), "n_vehicles": int(uv.size),
                  "max_vehicle_visit_share": float(w.max() / w.sum()) if w.size else 0.0}
    if method == "none" or temp.size < 3:
        info.update(method="none", drift_per_h=0.0, vehicle_offsets={})
        return temp.copy(), info
    if method not in ("fe", "protocol"):
        raise ValueError(f"unknown drift method {method!r} (fe, protocol, none)")
    P = np.column_stack([tau ** k for k in range(1, degree + 1)])
    line, line_se = _protocol_line(temp, P, vi)
    info.update(protocol_drift_per_h=float(line[0]), protocol_drift_se_per_h=line_se,
                pooled_drift_per_h=float(_protocol_line(temp, P)[0][0]))

    def finish(used: str, coef: np.ndarray, se: float, reason: str | None = None) -> tuple[np.ndarray, dict]:
        if reason:
            info["fallback"] = reason
        adj = temp - P @ coef
        info.update(method=used, drift_per_h=float(coef[0]), drift_se_per_h=se, drift_coef=[float(x) for x in coef],
                    vehicle_offsets={})
        if method == "fe":
            off, lab, links = co_visit_offsets(cells, vi, t, adj, w, co_visit_s, min_co_visit_cells)
            adj = adj - off[vi]
            info.update(vehicle_offsets={str(k): float(o) for k, o in zip(uv, off)},
                        n_vehicle_groups=int(np.unique(lab).size), offsets=f"co-visits ≤ {co_visit_s / 60:.0f} min",
                        vehicle_links=[{**ln, "vehicles": [str(uv[i]) for i in ln["vehicles"]]} for ln in links])
        return adj, info

    protocol = "the protocol line (CAPA's detrending, fitted within vehicles)"
    if method == "protocol":
        return finish("protocol", line, line_se)
    if n_spread < min_repeat_cells:
        return finish("protocol", line, line_se,
                      f"only {n_spread} cells revisited ≥ {min_span_s / 60:.0f} min apart (< {min_repeat_cells}; "
                      f"{n_rep} revisited at all): repeat visits minutes apart do not identify the drift over the "
                      f"run, {protocol}")
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsqr

    uc, ci = np.unique(cells, return_inverse=True)
    n = temp.size
    C = csr_matrix((np.ones(n), (np.arange(n), ci)), shape=(n, uc.size))
    V = csr_matrix((np.ones(n), (np.arange(n), vi)), shape=(n, uv.size))
    X = hstack([C, csr_matrix(P), V]).tocsr()
    lim = 20 * X.shape[1] + 1000
    y0 = temp - temp.mean()
    sol = lsqr(X, y0, atol=1e-12, btol=1e-12, iter_lim=lim)[0]
    beta = sol[uc.size:uc.size + degree]
    n_groups = int(np.unique(_components(cells, veh)).size)
    # SE of the linear drift coefficient (Frisch–Waugh), and clustered by block (by vehicle without positions)
    dof = n - (uc.size + degree + uv.size - n_groups)
    Z = hstack([C, csr_matrix(P[:, 1:]), V]).tocsr() if degree > 1 else hstack([C, V]).tocsr()
    e = P[:, 0] - Z @ lsqr(Z, P[:, 0], atol=1e-12, btol=1e-12, iter_lim=20 * Z.shape[1] + 1000)[0]
    ee = float(e @ e)
    se_ols = se_cl = float("inf")
    if dof > 0 and ee > 1e-12:
        u = y0 - X @ sol
        se_ols = math.sqrt(float(np.sum(u ** 2)) / dof / ee)
        s = e * u
        g = blk if blk is not None else vi
        S = np.bincount(g, weights=s)
        G = int(np.count_nonzero(np.bincount(g, weights=np.abs(e)) > 1e-9))
        se_cl = math.sqrt(G / (G - 1) * float(np.sum(S ** 2))) / ee if G >= 2 else float("nan")
    se = max(se_ols, se_cl) if np.isfinite(se_cl) else se_ols
    info.update(fe_drift_per_h=float(beta[0]), fe_drift_se_per_h=se, fe_drift_se_ols_per_h=se_ols,
                fe_drift_se_cluster_per_h=se_cl, se_clusters="300 m blocks" if blk is not None else "vehicles")
    fe = f"cell-fixed-effect drift {beta[0]:+.2f} ± {se:.2f} °C/h"
    if n_blocks is not None and n_blocks < min_repeat_blocks:
        return finish("protocol", line, line_se,
                      f"{fe} rests on {n_spread} cells revisited ≥ {min_span_s / 60:.0f} min apart in only {n_blocks} "
                      f"blocks of {REPEAT_BLOCK_M:.0f} m (< {min_repeat_blocks}): a change on a few streets, not "
                      f"the run's drift; {protocol}")
    if uv.size >= 2 and share > max_repeat_share:
        return finish("protocol", line, line_se,
                      f"{fe} rests on vehicle {top}'s repeats ({share:.0%} of the {n_spread} cells revisited ≥ "
                      f"{min_span_s / 60:.0f} min apart, > {max_repeat_share:.0%}): one route's change, not the "
                      f"run's drift; {protocol}")
    if not se <= max_drift_se:
        return finish("protocol", line, line_se, f"{fe} (SE above {max_drift_se} °C/h): {protocol}")
    if abs(beta[0] - line[0]) > DRIFT_GAP_WARN:
        info["warning"] = (f"{fe} differs from the protocol line ({line[0]:+.2f} °C/h) by more than "
                           f"{DRIFT_GAP_WARN} °C/h; kept, since its repeat visits pass every guard")
    return finish("fe", beta, se)


# --------------------------------------------------------------------------- #
# Rasters: focal means and providers                                           #
# --------------------------------------------------------------------------- #
def focal_mean(raster: np.ndarray, radius_m: float, dx: float) -> np.ndarray:
    """Gaussian neighbourhood mean (σ = radius/2) over the finite cells, as the core's focal features."""
    from sparc.core import operators as ops

    r = np.asarray(raster, dtype=np.float64)
    m = np.isfinite(r)
    if not m.any():
        return np.full(r.shape, np.nan)
    out = ops.masked_gaussian(np.where(m, r, 0.0), m, max(radius_m / 2.0 / dx, 0.5))
    out = np.asarray(out, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def derived_rasters(feats: dict[str, np.ndarray], lst: np.ndarray | None, dx: float) -> dict[str, np.ndarray]:
    """Feature rasters, their focal means (``<name>_f<r>``) and the LST with its focal means."""
    out = {k: np.asarray(feats[k], dtype=np.float64) for k in FEATURES if k in feats}
    for k in FOCAL_FEATURES:
        if k in out:
            for r in FOCAL_RADII:
                out[f"{k}_f{r}"] = focal_mean(out[k], r, dx)
    if lst is not None:
        out["lst"] = np.asarray(lst, dtype=np.float64)
        for r in LST_RADII:
            out[f"lst_f{r}"] = focal_mean(out["lst"], r, dx)
    return out


def open_features(ctx: CityGrid, campaign_date: dt.date, max_cloud: float = 20.0,
                  s2_tiles: list[str] | None = None) -> tuple[dict, dict]:
    """The six predictors on the city grid from open data (:func:`sparc.core.features_open.build_open_features`;
    Sentinel-2 months: the campaign month and the one on each side)."""
    from sparc.core.features_open import build_open_features

    months = []
    for k in (-1, 0, 1):
        m = campaign_date.month + k
        y = campaign_date.year + (m - 1) // 12
        months.append((y, (m - 1) % 12 + 1))
    df, prov = build_open_features(ctx.points_data(), ctx.identify_cfg(), months=tuple(months), max_cloud=max_cloud,
                                   s2_tiles=s2_tiles)
    return {k: df[k].to_numpy(float).reshape(ctx.shape) for k in FEATURES}, prov


def landsat_lst(ctx: CityGrid, campaign_date: dt.date, study: np.ndarray, window_days: int = 16,
                min_clear: float = 0.5, cache_dir: Path | None = None, prefer_clear: float = 0.8) -> dict:
    """Campaign-matched Landsat LST (°C) on the city grid (:func:`sparc.data.collect.landsat.campaign_lst`).

    A search or read that a host refused raises :class:`NetworkBlocked` naming that host; when no scene could be
    kept because reads failed for another reason, RuntimeError (the city is recorded failed and retried): only
    "every inspected scene was read and too cloudy" gives NaN LST in a finished city."""
    from sparc.data.collect.landsat import campaign_lst, io_error_kind

    try:
        res = campaign_lst(ctx.bounds_lonlat(0.005), campaign_date, window_days=window_days, min_clear=min_clear,
                           prefer_clear=prefer_clear, target=ctx.target, study_mask=ctx.to_northup(study),
                           cache_dir=cache_dir)
    except Exception as exc:                          # noqa: BLE001 - name a blocked host, else fail the city
        _raise_lst_failure([{"kind": io_error_kind(exc), "host": _failed_host(exc), "error": str(exc)[:200]}])
        raise
    check_lst_result(res)
    lst = None if res["lst_c"] is None else ctx.from_northup(res["lst_c"])
    return {"lst_c": lst, "scenes": res["scenes"], "candidates": res["candidates"], "window_days": window_days,
            "min_clear": min_clear, "prefer_clear": prefer_clear, "selection": res.get("selection"),
            "qa_masked": res["qa_masked"], "status": res.get("status")}


def _failed_host(exc: BaseException) -> str | None:
    from sparc.data.collect.landsat import href_host

    e, n = exc, 0
    while e is not None and n < 6:
        for a in ("sparc_url", "url", "filename"):
            h = href_host(getattr(e, a, None)) if isinstance(getattr(e, a, None), str) else None
            if h:
                return h
        m = re.search(r"https?://[^\s'\")]+", str(e))
        if m:
            return href_host(m.group(0))
        e, n = e.__cause__ or e.__context__, n + 1
    return None


def _raise_lst_failure(errors: list[dict]) -> None:
    """NetworkBlocked when a Landsat host refused us; RuntimeError (the city is recorded failed and retried) when
    reads failed otherwise; nothing when ``errors`` holds no I/O failure."""
    _, hosts, purpose, remedy = HOSTS["landsat"]
    blocked = [e for e in errors if e.get("kind") == "blocked"]
    if blocked:
        e = blocked[0]
        raise NetworkBlocked(e.get("host") or hosts[-1], str(e.get("error"))[:160], purpose, remedy)
    io = [e for e in errors if e.get("kind") in ("io", "unreadable")]
    if io:
        raise RuntimeError(f"Landsat reads failed for every inspected scene ({io[0].get('host')}: "
                           f"{str(io[0].get('error'))[:160]}); whether a clear scene exists is unknown, so the city "
                           "is not cached and is retried on the next run")


def check_lst_result(res: dict) -> None:
    """Refuse a :func:`~sparc.data.collect.landsat.campaign_lst` result whose scenes failed to read (status
    ``io_error``): such a city must not be cached as finished with NaN LST."""
    if res.get("status") == "io_error":
        _raise_lst_failure(res.get("io_errors") or [{"kind": "io", "error": "unknown read failure"}])


def span_weights(n_hours: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights of a time-weighted mean over [``a``, ``b``] (hours after the first of ``n_hours + 1`` whole hours):
    (for the ``n_hours + 1`` instantaneous values at the hours, interpolated linearly between them; for the
    ``n_hours`` hourly means, by each hour's overlap).  Each sums to 1; a span of zero length weighs the value
    at ``a`` (and the hour holding it)."""
    a, b = float(min(a, b)), float(max(a, b))
    wa, wh = np.zeros(n_hours + 1), np.zeros(n_hours)
    if b - a <= 1e-12:
        i = min(max(int(math.floor(a)), 0), n_hours - 1)
        u = min(max(a - i, 0.0), 1.0)
        wa[i], wa[i + 1], wh[i] = 1.0 - u, u, 1.0
        return wa, wh
    for i in range(n_hours):
        lo, hi = max(a, i), min(b, i + 1)
        if hi <= lo:
            continue
        ulo, uhi = lo - i, hi - i                  # the trapezoid of the line between hours i and i + 1
        wa[i] += (hi - lo) * ((1 - ulo) + (1 - uhi)) / 2
        wa[i + 1] += (hi - lo) * (ulo + uhi) / 2
        wh[i] += hi - lo
    return wa / (b - a), wh / (b - a)


#: How :func:`era5_run` averages (part of the ERA5 partial-cache key: a result of the earlier hourly mean is not
#: reused)
ERA5_AVERAGING = ("time-weighted over the run's span: analyses (t2m, skt) interpolated linearly between the hours, "
                  "hourly accumulations (ssrd, strd) by their overlap with the run")


def era5_run(lon: float, lat: float, tz: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """ERA5 at the city centre over the run: 2 m temperature (°C) and the physics forcing (SW↓ and LW net,
    W m⁻²; :func:`sparc.core.forcing.era5_site`), each the time-weighted mean over the run's span from ``start``
    to ``end`` (local clock of ``tz``).

    The whole hours bracketing the run are requested (floor(start) … ceil(end)); the analyses (t2m, skt) are
    instantaneous at the hours and interpolated linearly between them, the accumulations (ssrd, strd) are hourly
    means weighted by their overlap with the run (:func:`span_weights`).  The plain mean of the bracketing hours
    put Sedona's evening run (19:00–19:27 MST) at the mean of 19 h (25.96 °C) and 20 h (20.78 °C), so
    its background ran +5.4 °C above ERA5 against +2.7 in the afternoon."""
    from zoneinfo import ZoneInfo

    from sparc.core import forcing as F

    s, e = sorted((pd.Timestamp(start), pd.Timestamp(end)))
    h0 = s.floor("h")
    h1 = max(e.ceil("h"), h0 + pd.Timedelta(hours=1))
    hours = list(pd.date_range(h0, h1, freq="h"))
    z = ZoneInfo(tz)
    times = [dt.datetime(h.year, h.month, h.day, h.hour, tzinfo=z).astimezone(dt.timezone.utc) for h in hours]
    wa, wh = span_weights(len(hours) - 1, (s - h0) / pd.Timedelta(hours=1), (e - h0) / pd.Timedelta(hours=1))
    out = F.era5_site(lat, lon, times, analyses={k: F.ANALYSES[k] for k in ("t2m", "skt")},
                      accumulations={k: F.ACCUMULATIONS[k] for k in ("ssrd", "strd")})
    ser = out["series"]
    t2m, skt = (float(np.dot(wa, ser[k])) for k in ("t2m", "skt"))
    ssrd, strd = (float(np.dot(wh, ser[k])) for k in ("ssrd", "strd"))
    lw_up = F.EMISSIVITY * F.SIGMA * skt ** 4
    return {"ta_era5": t2m - 273.15, "sw_down": ssrd, "lw_net": strd - lw_up, "hours_local": [h0.hour, h1.hour],
            "span_local": [str(s), str(e)], "times_utc": out["times_utc"], "weights_hours": [float(x) for x in wa],
            "weights_hourly_means": [float(x) for x in wh], "averaging": ERA5_AVERAGING,
            "source": "ERA5 (NSF NCAR mirror), city centre"}


# --------------------------------------------------------------------------- #
# One city                                                                     #
# --------------------------------------------------------------------------- #
def _run_code(window: str) -> str:
    return RUN_CODE.get(str(window), str(window))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


class _Partial:
    """Per-source results of a city build kept as each finishes (``OUT/cities/<slug>/partial/<name>.npz|json``),
    so a build cut off by a restart or an out-of-memory kill resumes from the sources already done.  Each entry
    carries a key (a hash of the grid, the date, the provider and its settings); a different key recomputes.
    Removed when the city is finished (its rasters are then in ``rasters.npz``)."""

    def __init__(self, city_dir: Path):
        self.dir = Path(city_dir) / "partial"

    @staticmethod
    def key(*parts) -> str:
        import hashlib

        return hashlib.sha256(json.dumps(parts, default=_json_default, sort_keys=True).encode()).hexdigest()[:20]

    def load(self, name: str, key: str):
        j, z = self.dir / f"{name}.json", self.dir / f"{name}.npz"
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
            if meta.get("key") != key:
                return None
            arrays = {}
            if z.exists():
                with np.load(z, allow_pickle=False) as f:
                    arrays = {k: f[k] for k in f.files}
            return arrays, meta.get("value")
        except (OSError, ValueError, KeyError):
            return None

    def save(self, name: str, key: str, arrays: dict | None, value) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if arrays:
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
            _write_atomic(self.dir / f"{name}.npz", buf.getvalue())
        _write_atomic(self.dir / f"{name}.json", to_json({"key": key, "value": value}, indent=None).encode())

    def clear(self) -> None:
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


#: The dataset's QA version, part of every city's settings fingerprint, so a cache built under earlier rules is
#: reported stale and rebuilt.  1: the first build of the 31 cities; 2: the rules of their review (2026-10-20):
#: protocol line within vehicles, vehicle offsets from co-visits only, the drift guards, the single-vehicle,
#: platform, non-vehicle-sensor and QA-registry checks, ERA5 averaged over the run's span.
DATASET_VERSION = 2

#: build_city settings that change a city's table (the fingerprint a cached city is compared with).
#: ``qa_registry``: a key of the city's entries in the QA registry (:mod:`~sparc.core.lstair.exclusions`), so
#: adding or changing one rebuilds that city only.
SETTING_DEFAULTS = {"cell_m": CELL_M, "margin_m": MARGIN_M, "drift": "fe", "drift_degree": 1, "utc_to": None,
                    "lst_window_days": 16, "lst_min_clear": 0.5, "lst_prefer_clear": 0.8, "skip_files": list(SKIP_FILES),
                    "campaign_date": None, "lst": True, "era5": True, "qa_registry": None}


def build_settings(**kw) -> dict:
    """The fingerprinted settings of a city build (``SETTING_DEFAULTS`` updated with ``kw``; ``lst`` / ``era5`` are
    whether the source is on) and their key."""
    out = dict(SETTING_DEFAULTS)
    for k, v in kw.items():
        if k in out:
            out[k] = (v is not False) if k in ("lst", "era5") else (list(v) if isinstance(v, tuple) else v)
    out["cell_m"], out["margin_m"] = float(out["cell_m"]), float(out["margin_m"])
    out["lst_window_days"], out["lst_min_clear"] = int(out["lst_window_days"]), float(out["lst_min_clear"])
    out["lst_prefer_clear"] = float(out["lst_prefer_clear"])
    out["schema"] = SCHEMA
    out["dataset_version"] = DATASET_VERSION
    return out


def settings_key(settings: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]


def registry_key(slug: str, registry: list[dict] | None = None) -> str | None:
    """A key of ``slug``'s QA-registry entries (None without any), for :data:`SETTING_DEFAULTS`."""
    from sparc.core.lstair import exclusions

    mine = exclusions.entries(slug, registry)
    return _Partial.key(mine)[:12] if mine else None


def _provider_id(fn, default: str) -> str:
    return default if fn is None else f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}"


def city_status(city_dir: Path) -> str | None:
    p = Path(city_dir) / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


def _features_provider(meta: dict) -> str | None:
    """The feature provider of a finished build: as recorded, or, for builds before it was recorded, the open-data
    provider when the provenance names its sources (WorldCover and Sentinel-2)."""
    if meta.get("features_provider"):
        return meta["features_provider"]
    prov = meta.get("features")
    return "open_features" if isinstance(prov, dict) and {"worldcover", "sentinel2"} <= set(prov) else None


def reusable_features(city_dir: str | Path, ctx: CityGrid, campaign: dt.date, provider: str
                      ) -> tuple[dict, dict, str] | None:
    """The feature rasters of the city's previous finished build, when it has the same grid (crs, x0, y0, dx, nx,
    ny), campaign date and feature provider: (the :data:`FEATURES` rasters from its ``rasters.npz``, float32 as
    stored, its feature provenance, its ``built_utc``), else None.  Read before the rebuild overwrites them; the
    Sentinel-2 composite alone takes 5–25 min per city, and none of these inputs depends on the traverse QA."""
    d = Path(city_dir)
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    g, want = meta.get("grid") or {}, ctx.as_dict()
    try:
        same = (g.get("crs") == want["crs"] and all(int(g[k]) == want[k] for k in ("nx", "ny"))
                and all(abs(float(g[k]) - want[k]) < 1e-6 for k in ("x0", "y0", "dx")))
    except (KeyError, TypeError, ValueError):
        return None
    if (meta.get("status") != "ok" or not same or meta.get("campaign_date") != campaign.isoformat()
            or _features_provider(meta) != provider):
        return None
    try:
        with np.load(d / "rasters.npz", allow_pickle=False) as z:
            if not all(k in z.files for k in FEATURES):
                return None
            feats = {k: z[k].astype(np.float64) for k in FEATURES}
    except (OSError, ValueError):
        return None
    if any(r.shape != ctx.shape for r in feats.values()):
        return None
    return feats, meta.get("features") or {}, str(meta.get("built_utc"))


# --------------------------------------------------------------------------- #
# Run QA: platform, merged loggers, a single vehicle                           #
# --------------------------------------------------------------------------- #
PEDESTRIAN_KMH = 10.0                  # a run whose readings move slower than this (median) was walked, not driven
SINGLE_VEHICLE_SHARE = 0.9             # one sensor with this share of a run's visits drives it alone
SINGLE_VEHICLE = "single-vehicle run: drift not identifiable"
#: One vehicle key holds several loggers when this share of its readings share their second with a reading of the
#: key more than MERGED_KEY_M away (Boston 2019: about ten cars per run file, no sensor id)
MERGED_KEY_SHARE = 0.5
MERGED_KEY_M = 2000.0


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def platform_speed(points: pd.DataFrame, max_dt_s: float = 30.0) -> dict:
    """The speed of a run's raw readings: the median over every step between a vehicle's consecutive readings at
    most ``max_dt_s`` apart (haversine distance over the time step, km/h), per sensor and for the run, and the
    run's ``platform``: ``"pedestrian"`` below :data:`PEDESTRIAN_KMH`, else ``"vehicle"`` (also when no step
    can be measured).

    Consecutive is the order the loggers wrote them (as read): Boston 2019's run files hold about ten cars one
    after another without a sensor id, which in time order interleave (a 6 km 'step' per second).  Cars drive at
    ≥ ~30 km/h (Lynchburg 2021-07-15: 32–37 km/h, 9–10 m steps at 1 Hz); Farmville's runs move at 4.7–5.7 km/h
    and Abingdon's at 4.9–5.0 (1.3–1.6 m steps): walked campaigns, whose sensor height and aspiration are not the
    car-mounted protocol's."""
    from sparc.core.identify.traverses import sensor_ids

    out: dict = {"median_speed_kmh": None, "platform": "vehicle", "n_steps": 0, "speed_kmh_by_sensor": {}}
    if len(points) < 2:
        return out
    key = points["vehicle_key"].astype(str).to_numpy()
    t = pd.to_datetime(points["time"], errors="coerce")
    ts = (t - t.min()).dt.total_seconds().to_numpy(float)
    lat, lon = points["lat"].to_numpy(float), points["lon"].to_numpy(float)
    step_t = np.diff(ts)
    d = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    ok = (key[1:] == key[:-1]) & (step_t > 0) & (step_t <= max_dt_s) & np.isfinite(d)
    if not ok.any():
        return out
    kmh = d[ok] / step_t[ok] * 3.6
    by = pd.Series(kmh).groupby(sensor_ids(points).map(sensor_name).to_numpy()[1:][ok]).median()
    med = float(np.median(kmh))
    out.update(median_speed_kmh=med, median_step_m=float(np.median(d[ok])), n_steps=int(ok.sum()),
               platform="pedestrian" if med < PEDESTRIAN_KMH else "vehicle",
               speed_kmh_by_sensor={str(k): round(float(x), 1) for k, x in by.items()})
    return out


def merged_logger_share(points: pd.DataFrame) -> float:
    """Share of a run's readings that share their vehicle key and timestamp with a reading of that key more than
    :data:`MERGED_KEY_M` away: several loggers read as one vehicle.  Boston 2019 (one key per run file: about ten
    cars one after another, 4–6 readings per second, 'case' constant) is the case; a car's own readings, or a clock
    cut to the minute (readings of one minute within about a kilometre), are not."""
    if not len(points):
        return 0.0
    df = pd.DataFrame({"k": points["vehicle_key"].astype(str).to_numpy(),
                       "t": pd.to_datetime(points["time"], errors="coerce").to_numpy(),
                       "la": points["lat"].to_numpy(float), "lo": points["lon"].to_numpy(float)})
    g = df.groupby(["k", "t"])
    n = g["la"].transform("size").to_numpy(float)
    dla = (g["la"].transform("max") - g["la"].transform("min")).to_numpy(float) * 111_320.0
    dlo = ((g["lo"].transform("max") - g["lo"].transform("min")).to_numpy(float) * 111_320.0
           * np.cos(np.radians(df["la"].to_numpy(float))))
    return float(np.mean((n >= 2) & (np.hypot(dla, dlo) > MERGED_KEY_M)))


#: A CAPA logger file name, "<sensor>_<DDMMYY>_<HHMMSS[.sss]>_<n>.csv", whose sensor the reader does not split off
#: when it has no letters (the 2021 campaigns: 1057_150721_095302.305_0.csv, 1013B_150721_185303.000_0.csv, and
#: names cut short in the tables, 1072_1_095722.000_0.csv, 1085_150721_1849_0.csv)
_LOGGER_FILE = re.compile(r"^([A-Za-z]*\d+[A-Za-z]?)_\d")


def sensor_name(sid) -> str:
    """The sensor of a parsed sensor id (:func:`~sparc.core.identify.traverses.sensor_ids`): the id without a
    logger file's date and time, so that one sensor's files of a run (a logger restarted) are one vehicle."""
    m = _LOGGER_FILE.match(str(sid))
    return m.group(1) if m else str(sid)


def _vehicle_labels(sub: pd.DataFrame, n: int) -> list[str]:
    """The sensor of each of :func:`~sparc.core.identify.traverses.to_campaign`'s vehicle indices (its vehicle
    keys in sorted order; :func:`sensor_name`): the vehicles of :func:`adjust_visits`, so one sensor's keys in
    several files (Los Angeles 2019: CAPA34 in the north and the south route files) are one vehicle; the indices
    themselves when the keys do not match."""
    from sparc.core.identify.traverses import sensor_ids

    pts = sub.dropna(subset=["time", "lat", "lon", "temp_f"]).drop_duplicates("vehicle_key")
    sid = dict(zip(pts["vehicle_key"].astype(str), sensor_ids(pts).astype(str)))
    keys = sorted(sid)
    return [sensor_name(sid[k]) for k in keys] if len(keys) == n else [str(i) for i in range(n)]


def build_run(sub: pd.DataFrame, info: dict, ctx: CityGrid, drift: str = "fe", drift_degree: int = 1,
              registry_reason: str | None = None) -> tuple[dict, pd.DataFrame | None, list[str]]:
    """One run of a city: its visits (:func:`~sparc.core.identify.traverses.to_campaign`), drift and vehicle
    offsets (:func:`adjust_visits`), cell means and QA; (the run's ``meta["runs"]`` record, its cells or None
    when it cannot be placed on the grid, warnings).

    The record's ``qa_exclude`` is "" for a usable run, else why it is excluded by default (reasons joined with
    "; "; :mod:`~sparc.core.lstair.evaluate` drops such runs unless asked to include flagged ones):

    * ``registry_reason``: the QA registry's (:func:`~sparc.core.lstair.exclusions.run_exclusions`), as
      "qa registry: " + reason;
    * "pedestrian platform (median X km/h)" when the readings move below :data:`PEDESTRIAN_KMH`
      (:func:`platform_speed`; ``platform`` and ``qa["median_speed_kmh"]`` record it);
    * :data:`SINGLE_VEHICLE` when one sensor holds :data:`SINGLE_VEHICLE_SHARE` of the visits and the drift is the
      protocol line: time and route position are then one axis, so the line is the route's gradient (Lynchburg
      2021-07-15 morning and midday: car 1057 with 971 of 1,076 and 938 of 948 visits, protocol −0.63 / +1.43
      °C/h, against −0.31 / +1.36 for 1057 alone and +0.09 / +0.53 in ERA5; re-detrending moved r(dTa, LST) from
      0.57 to 0.44).  Sensors are counted by their id; a key holding several loggers (:func:`merged_logger_share`)
      is not one vehicle and gets a warning instead."""
    from sparc.core.identify.traverses import to_campaign
    from sparc.core.lstair.exclusions import PREFIX

    run_id = info["run"]
    rec = {"run_id": run_id, "run": _run_code(info["window"]), "window": info["window"], "start": info["start"],
           "end": info["end"], "n_readings": int(info["n"]), "vehicles": info["vehicles"]}
    warnings: list[str] = []
    try:
        df, qa = to_campaign(sub, ctx.identify_cfg(), ctx.identify_layout(), window=None)
    except ValueError as exc:
        rec.update(error=str(exc), qa_exclude=f"not built: {exc}")
        return rec, None, warnings
    df = df.assign(temp=(df["temp"].to_numpy(float) - 32.0) * 5.0 / 9.0)        # °F → °C
    sensor = np.asarray(_vehicle_labels(sub, int(df["vehicle"].max()) + 1), dtype=object)[df["vehicle"].to_numpy()]
    x, y = ctx.cell_xy(df["cell"].to_numpy())
    adj, dinfo = adjust_visits(df.assign(vehicle=sensor, x=x, y=y), drift, drift_degree)
    g = pd.DataFrame({"cell": df["cell"].to_numpy(), "Ta": adj, "n_raw": df["n_raw"].to_numpy()}).groupby("cell")
    cells = pd.DataFrame({"cell": g.size().index.to_numpy(), "Ta": g["Ta"].mean().to_numpy(),
                          "n_readings": g["n_raw"].sum().to_numpy(), "n_visits": g.size().to_numpy()})
    bg = float(np.median(cells["Ta"]))
    cells["dTa"] = cells["Ta"] - bg
    cells["ta_background"] = bg
    by_sensor = pd.Series(sensor).value_counts()
    top, top_share = str(by_sensor.index[0]), float(by_sensor.iloc[0] / by_sensor.sum())
    speed = platform_speed(sub)
    merged = merged_logger_share(sub)
    qa.update(median_speed_kmh=speed["median_speed_kmh"], platform=speed["platform"], speed_n_steps=speed["n_steps"],
              speed_kmh_by_sensor=speed["speed_kmh_by_sensor"], n_sensors=int(by_sensor.size), top_sensor=top,
              top_sensor_visit_share=top_share, merged_logger_share=merged)
    reasons = [PREFIX + registry_reason] if registry_reason else []
    if speed["platform"] == "pedestrian":
        reasons.append(f"pedestrian platform (median {speed['median_speed_kmh']:.1f} km/h)")
        warnings.append(f"{run_id}: the readings move at a median {speed['median_speed_kmh']:.1f} km/h (below "
                        f"{PEDESTRIAN_KMH:.0f}): a walked traverse, platform 'pedestrian', excluded by default")
    if merged >= MERGED_KEY_SHARE:
        warnings.append(f"{run_id}: {merged:.0%} of the readings share their vehicle key and second with one over "
                        f"{MERGED_KEY_M / 1000:.0f} km away: several loggers under one key without a sensor id, so "
                        "their offsets cannot be estimated and it is not a single-vehicle run")
    elif top_share >= SINGLE_VEHICLE_SHARE and dinfo.get("method") == "protocol":
        reasons.append(SINGLE_VEHICLE)
        warnings.append(f"{run_id}: {top} holds {top_share:.0%} of the visits and the drift is the protocol line: "
                        "a single-vehicle run, excluded by default")
    if dinfo.get("warning"):
        warnings.append(f"{run_id}: {dinfo['warning']}")
    rec.update(qa=qa, drift={k: v for k, v in dinfo.items() if k != "vehicle_offsets"},
               vehicle_offsets_c=dinfo.get("vehicle_offsets", {}), background_c=bg,
               n_cells=int(len(cells)), run_date=str(pd.Timestamp(info["start"]).date()), platform=speed["platform"],
               qa_exclude="; ".join(reasons))
    cells["run"], cells["run_id"], cells["run_date"] = rec["run"], rec["run_id"], rec["run_date"]
    cells["qa_exclude"] = rec["qa_exclude"]
    return rec, cells, warnings


def _drop_non_vehicles(points: pd.DataFrame) -> tuple[pd.DataFrame, dict, str | None]:
    """The points without the readings of bicycle or walking sensors
    (:func:`~sparc.core.identify.traverses.non_vehicle_sensors`); (points, {sensor id: readings dropped}, a
    warning naming them).  Sacramento 2019's Bicycle_CAPA56 rides the American River Parkway inside the car files:
    680–760 cells per run that the cars almost never share, and 98% of the evening cells with |dTa| > 5 °C."""
    from sparc.core.identify.traverses import non_vehicle_sensors, sensor_ids

    bad = non_vehicle_sensors(points)
    if not bad.any():
        return points, {}, None
    counts = {str(k): int(v) for k, v in sensor_ids(points[bad]).value_counts().sort_index().items()}
    note = (f"{int(bad.sum()):,} readings of non-vehicle sensors dropped: "
            + ", ".join(f"{k} ({v:,})" for k, v in counts.items()))
    return points[~bad].reset_index(drop=True), counts, note


def _apply_registry(points: pd.DataFrame, slug: str, registry: list[dict] | None
                    ) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """The points without the readings the QA registry drops for this city
    (:func:`~sparc.core.lstair.exclusions.reading_mask`); (points, its records, warnings with the counts: an entry
    that matches no reading says so)."""
    from sparc.core.lstair import exclusions

    keep, records = exclusions.reading_mask(points, slug, registry)
    notes = []
    for r in records:
        what = f"qa registry: {r['sensor']} in {r['run_id']}" + (
            f" from {r['from_local']}" if r.get("from_local") else "") + (
            f" until {r['until_local']}" if r.get("until_local") else "")
        notes.append(f"{what}: {r['n_readings']:,} readings dropped ({r['reason']})" if r["n_readings"] else
                     f"{what} matches no reading (is the sensor id parsed as {r['sensor']!r}?)")
    return (points[keep].reset_index(drop=True) if not keep.all() else points), records, notes


def build_city(spec: CitySpec, traverses: str | Path, out_dir: str | Path, *, features: Callable | None = None,
               lst: Callable | None = None, era5: Callable | None = None, cell_m: float = CELL_M,
               margin_m: float = MARGIN_M, drift: str = "fe", drift_degree: int = 1, utc_to: str | None = None,
               lst_window_days: int = 16, lst_min_clear: float = 0.5, cache_dir: Path | None = None,
               skip_files: tuple[str, ...] = SKIP_FILES, campaign_date: str | None = None,
               lst_prefer_clear: float = 0.8, registry: list[dict] | None = None,
               reuse_features: bool = True) -> dict:
    """Build one city's table and rasters from a local traverse folder; returns its meta (also written).

    ``features(ctx, campaign_date) → (rasters, provenance)``, ``lst(ctx, campaign_date, study) → {"lst_c",
    "scenes", ...}`` and ``era5(lon, lat, tz, start, end) → {"ta_era5", "sw_down", "lw_net", ...}`` are the data
    providers (defaults: :func:`open_features`, :func:`landsat_lst`, :func:`era5_run`; ``lst=False`` /
    ``era5=False`` switch a source off).  Tests pass synthetic ones.  Files whose path contains a pattern of
    ``skip_files`` (default :data:`SKIP_FILES`) are not read; readings repeated across files are read once
    (:func:`sparc.core.identify.traverses.scan_traverses`), and the count is ``meta["n_duplicate_readings"]``.

    Timestamps are read as the local clock of ``spec.tz`` (tz-aware ones converted to it).  Tables whose only clock
    is the 2019 NIHHIS ``time_n`` (UTC ``HHMMSS``, no date; tdsy7's Boise and Boston) are dated with
    ``campaign_date`` (local ISO date; default the catalog's ``campaign_date``); without one the build stops
    and says so.

    Before the grid is laid, the readings of bicycle and walking sensors are dropped
    (``meta["non_vehicle_sensors"]``) and so are those the QA ``registry`` lists for the city (default: the
    shipped one, :mod:`~sparc.core.lstair.exclusions`; ``meta["qa_registry"]``); each run is built by
    :func:`build_run`, and its ``qa_exclude`` (also a ``table.parquet`` column) says why it is excluded by default
    ("" when usable).  With ``reuse_features`` the feature rasters of the city's previous finished build are reused
    when its grid, campaign date and feature provider are the same (:func:`reusable_features`;
    ``meta["features_reused_from"]`` names that build); LST and ERA5 are always recomputed."""
    from sparc.core.identify.traverses import runs_summary, scan_traverses
    from sparc.core.lstair import exclusions

    t0 = time.perf_counter()
    settings = build_settings(cell_m=cell_m, margin_m=margin_m, drift=drift, drift_degree=drift_degree, utc_to=utc_to,
                              lst_window_days=lst_window_days, lst_min_clear=lst_min_clear,
                              lst_prefer_clear=lst_prefer_clear, skip_files=skip_files,
                              campaign_date=campaign_date, lst=lst, era5=era5,
                              qa_registry=registry_key(spec.slug, registry))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    clock_date = campaign_date or spec.catalog_date
    points, report = scan_traverses(traverses, timezone=spec.tz, utc_to=utc_to, exclude=tuple(skip_files or ()),
                                    campaign_date=clock_date)
    hms = [r for r in report if str(r.get("clock") or "").startswith("time_n")]
    n_dup = int(sum(r.get("duplicates", 0) for r in report))
    if n_dup:
        warnings.append(f"{n_dup:,} readings repeat another file's (same sensor, time and position): read once")
    if not len(points):
        skipped = "; ".join(f"{r['file']}: {r.get('reason')}" for r in report[:6])
        raise ValueError(f"no traverse points under {traverses}" + (f" (skipped {skipped})" if skipped else ""))
    points = points.dropna(subset=["lat", "lon", "temp_f"])
    points, clock_note = check_clock(points, spec.tz, utc_given=bool(utc_to))
    if clock_note:
        warnings.append(clock_note)
    points, non_vehicle, note = _drop_non_vehicles(points)
    if note:
        warnings.append(note)
    points, qa_records, notes = _apply_registry(points, spec.slug, registry)
    warnings += notes
    run_excl = exclusions.run_exclusions(spec.slug, registry)
    ctx = make_city_grid(points["lon"].to_numpy(float), points["lat"].to_numpy(float), spec.crs, cell_m, margin_m)
    if ctx.nx * ctx.ny > WARN_CELLS:
        warnings.append(f"large grid: {ctx.nx * ctx.dx / 1000:.0f} × {ctx.ny * ctx.dx / 1000:.0f} km "
                        f"({ctx.nx * ctx.ny:,} cells); the open features take long (several readings far apart?)")
    partial = _Partial(out)

    runs, frames = [], []
    for info in runs_summary(points):
        rec, cells, notes = build_run(points[points["run"] == info["run"]], info, ctx, drift, drift_degree,
                                      run_excl.get(info["run"]))
        runs.append(rec)
        warnings += notes
        if cells is not None:
            frames.append(cells)
    for rid in sorted(set(run_excl) - {r["run_id"] for r in runs}):
        warnings.append(f"qa registry: run {rid} is not among the city's runs (nothing excluded)")
    if not frames:
        raise ValueError(_no_run_reason(runs, report, spec))
    table = pd.concat(frames, ignore_index=True)
    ok_runs = [r for r in runs if "error" not in r]
    af = [r for r in ok_runs if r["run"] == "af"]
    campaign = dt.date.fromisoformat((af[0] if af else ok_runs[0])["run_date"])
    date_source = "traverse timestamps"
    if hms:
        date_source = (("--campaign-date" if campaign_date else f"catalog campaign_date ({spec.catalog_date_source})")
                       + ": the traverse clock (time_n) is UTC HHMMSS without a date")
    if spec.catalog_date and spec.catalog_date != campaign.isoformat():
        warnings.append(f"catalog campaign date {spec.catalog_date} ({spec.catalog_date_source}) differs from the "
                        f"traverse timestamps ({campaign.isoformat()}); the timestamps are used")

    # rasters: features (reused from the previous build when nothing they depend on changed), focal means, LST
    driven = np.zeros(ctx.nx * ctx.ny, dtype=bool)
    driven[table["cell"].to_numpy()] = True
    from scipy import ndimage

    study = ndimage.distance_transform_edt(~driven.reshape(ctx.shape)) * ctx.dx <= STUDY_BUFFER_M
    provider = _provider_id(features, "open_features")
    fkey = partial.key("features", ctx.as_dict(), campaign.isoformat(), provider)
    got = partial.load("features", fkey)
    reused_from = None
    if got is not None:
        feats, fprov = {k: np.asarray(v, dtype=np.float64) for k, v in got[0].items()}, got[1]
        reused_from = (partial.load("features_reuse", fkey) or ({}, None))[1]
    else:
        old = reusable_features(out, ctx, campaign, provider) if reuse_features else None
        if old is not None:
            feats, fprov, reused_from = old
            partial.save("features_reuse", fkey, None, reused_from)
        else:
            feats, fprov = (features or open_features)(ctx, campaign)
        partial.save("features", fkey, {k: feats[k] for k in FEATURES if k in feats}, fprov)
    lst_res = None
    if lst is not False:
        lkey = partial.key("lst", ctx.as_dict(), campaign.isoformat(), _provider_id(lst, "landsat_lst"),
                           lst_window_days, lst_min_clear, lst_prefer_clear, np.packbits(study).tobytes().hex())
        got = partial.load("lst", lkey)
        if got is not None:
            lst_res = {**got[1], "lst_c": got[0].get("lst_c")}
        else:
            lst_res = (lst or (lambda c, d, s: landsat_lst(c, d, s, lst_window_days, lst_min_clear, cache_dir,
                                                           lst_prefer_clear)))(ctx, campaign, study)
            check_lst_result(lst_res)                 # a provider's read failures never pass as "no clear scene"
            partial.save("lst", lkey, None if lst_res.get("lst_c") is None else {"lst_c": lst_res["lst_c"]},
                         {k: v for k, v in lst_res.items() if k != "lst_c"})
        if lst_res.get("lst_c") is None:
            why = ("no Landsat scene within the window" if lst_res.get("status") == "no_scene"
                   else "no Landsat scene clear enough within the window")
            warnings.append(f"{why}: LST columns are NaN")
    rasters = derived_rasters(feats, None if lst_res is None else lst_res.get("lst_c"), ctx.dx)

    # sample at the driven cells
    cell = table["cell"].to_numpy()
    iy, ix = np.divmod(cell, ctx.nx)
    x, y = ctx.cell_xy(cell)
    lon, lat = ctx.to_lonlat(x, y)
    table["x"], table["y"], table["lon"], table["lat"] = x, y, lon, lat
    for k, r in rasters.items():
        table[k] = r[iy, ix]
    if lst_res is not None and lst_res.get("scenes"):
        sc = lst_res["scenes"]
        scene_date = sorted(s["date"] for s in sc)[len(sc) // 2]
        table["lst_scene_date"] = scene_date
        table["lst_offset_days"] = [(dt.date.fromisoformat(scene_date) - dt.date.fromisoformat(d)).days
                                    for d in table["run_date"]]
        table["lst_clear_fraction"] = float(np.mean([s["clear_fraction"] for s in sc]))
    else:
        table["lst_scene_date"], table["lst_offset_days"], table["lst_clear_fraction"] = None, np.nan, np.nan
        if "lst" not in table:
            table["lst"] = np.nan
            for r in LST_RADII:
                table[f"lst_f{r}"] = np.nan

    # ERA5 per run
    clon, clat = ctx.to_lonlat(ctx.x0 + 0.5 * (ctx.nx - 1) * ctx.dx, ctx.y0 + 0.5 * (ctx.ny - 1) * ctx.dx)
    table["ta_era5"] = np.nan
    for rec in ok_runs:
        if era5 is False or not spec.tz:
            rec["era5"] = {"error": "ERA5 off" if era5 is False else "no time zone for this city"}
            continue
        try:
            ekey = partial.key("era5", float(clon), float(clat), spec.tz, rec["start"], rec["end"],
                               _provider_id(era5, "era5_run"), ERA5_AVERAGING)
            got = partial.load(f"era5_{_Partial.key(rec['run_id'])}", ekey)
            if got is not None:
                e = got[1]
            else:
                e = (era5 or era5_run)(float(clon), float(clat), spec.tz, pd.Timestamp(rec["start"]),
                                       pd.Timestamp(rec["end"]))
                partial.save(f"era5_{_Partial.key(rec['run_id'])}", ekey, None, e)
            rec["era5"] = e
            table.loc[table["run_id"] == rec["run_id"], "ta_era5"] = e["ta_era5"]
        except Exception as exc:                      # noqa: BLE001 - ERA5 is optional
            rec["era5"] = {"error": str(exc).splitlines()[0][:200] if str(exc) else repr(exc)[:200]}
            warnings.append(f"ERA5 unavailable for {rec['run_id']}: {rec['era5']['error']}")

    table.insert(0, "city", spec.name)
    table.insert(1, "slug", spec.slug)
    table.insert(2, "campaign_date", campaign.isoformat())
    lead = ["city", "slug", "campaign_date", "run", "run_id", "run_date", "qa_exclude", "cell", "x", "y", "lat", "lon",
            "n_readings", "n_visits", "Ta", "dTa", "ta_background", "ta_era5"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    buf = io.BytesIO()
    table.to_parquet(buf, index=False)
    _write_atomic(out / "table.parquet", buf.getvalue())
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v.astype(np.float32) for k, v in rasters.items()},
                        driven=driven.reshape(ctx.shape))
    _write_atomic(out / "rasters.npz", buf.getvalue())
    meta = {"schema": SCHEMA, "dataset_version": DATASET_VERSION, "status": "ok", "city": spec.name,
            "slug": spec.slug, "traverses": str(traverses), "campaign_date": campaign.isoformat(),
            "campaign_date_source": date_source,
            "catalog": {"node": spec.node, "folder_hint": spec.folder_hint, "campaign_date": spec.catalog_date,
                        "campaign_date_source": spec.catalog_date_source, "availability": spec.availability},
            "grid": ctx.as_dict(), "tz": spec.tz, "drift_method": drift, "settings": settings,
            "settings_key": settings_key(settings), "runs": runs,
            "files": report, "skip_files": list(skip_files or ()), "n_duplicate_readings": n_dup,
            "non_vehicle_sensors": non_vehicle, "qa_registry": {"readings": qa_records, "runs": run_excl},
            "features": fprov, "features_provider": provider, "features_reused_from": reused_from,
            "lst": None if lst_res is None else {k: v for k, v in lst_res.items() if k != "lst_c"},
            "n_rows": int(len(table)), "rasters": sorted(rasters), "warnings": warnings,
            "seconds": round(time.perf_counter() - t0, 1),
            "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    _write_atomic(out / "meta.json", to_json(meta).encode())
    partial.clear()
    return meta


#: CAPA Heat Watch drives 6–7, 15–16 and 19–20 local time (Providence also 0–1): windows in local hours with a
#: half-hour to an hour of slack, the night one wrapping midnight.
PROTOCOL_HOURS = ((5.5, 8.0), (14.5, 17.0), (18.5, 21.0), (23.5, 26.0))


def protocol_share(times: pd.Series) -> float:
    """Share of the readings whose local clock time falls in a CAPA protocol window (:data:`PROTOCOL_HOURS`)."""
    t = pd.to_datetime(times, errors="coerce").dropna()
    if not len(t):
        return float("nan")
    h = (t.dt.hour + t.dt.minute / 60.0).to_numpy(float)
    ok = np.zeros(h.size, bool)
    for lo, hi in PROTOCOL_HOURS:
        ok |= ((h >= lo) & (h < hi)) | ((h + 24.0 >= lo) & (h + 24.0 < hi))
    return float(ok.mean())


def check_clock(points: pd.DataFrame, tz: str | None, utc_given: bool = False) -> tuple[pd.DataFrame, str | None]:
    """Is the traverse clock local?  The readings' hours are compared with the CAPA protocol windows (6–7, 15–16,
    19–20 h, and 0–1 h) as read and after reading them as UTC converted to ``tz``; the clock that puts more
    readings in the windows is used (ties: as read).  Readings whose clock is known (a ``clock`` note from
    :func:`~sparc.core.identify.traverses.scan_traverses`: time_n, or a UTC column converted with the file's
    zone) are left alone.  Returns (points, a note when converted); ValueError when neither clock puts half the
    readings in the windows.  Without ``tz`` only the old heuristic warning is given.  With ``utc_given``
    (``--utc``) nothing is checked."""
    if utc_given:
        return points, None
    t = pd.to_datetime(points["time"], errors="coerce")
    free = t.notna() & (points["clock"].isna() if "clock" in points else True)
    if not free.any():
        return points, None
    if not tz:
        h = t[free].dt.hour
        if ((h >= 9) & (h < 12)).mean() > 0.25 and not h.between(14, 17).any():
            return points, "no readings at 14–17 h but many at 9–12 h: the times may be UTC (rerun with --utc)"
        return points, None
    as_read = protocol_share(t[free])
    conv = t[free].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    as_utc = protocol_share(conv)
    if max(as_read, as_utc) < 0.5:
        raise ValueError(f"the traverse clock fits neither reading of the CAPA protocol windows (6–7, 15–16, 19–20 h "
                         f"local): {as_read:.0%} of the readings fall in them as recorded and {as_utc:.0%} read as "
                         f"UTC in {tz}; check the files' time column (or pass --utc if it is UTC)")
    if as_utc > as_read:
        from sparc.core.identify.traverses import assign_runs

        pts = points.copy()
        pts.loc[free, "time"] = conv
        pts = assign_runs(pts.drop(columns=["run"], errors="ignore").reset_index(drop=True))
        return pts, (f"the timestamps are UTC: {as_utc:.0%} of the readings fall in the CAPA protocol windows after "
                     f"conversion to {tz} against {as_read:.0%} as recorded; converted")
    return points, None


def _no_run_reason(runs: list[dict], report: list[dict], spec: CitySpec) -> str:
    """Why no run could be built, file by file (the reason never ends empty)."""
    if runs:
        return "no run could be placed on the city grid: " + "; ".join(
            f"{r['run_id']}: {r.get('error') or 'no cells'}" for r in runs)
    read = [r for r in report if r.get("status") == "read"]
    parts = [f"{r['file']}: skipped ({r.get('reason')})" for r in report if r.get("status") != "read"][:6]
    parts += [f"{r['file']}: {r['rows']:,} readings" + (f", clock {r['clock']}" if r.get("clock") else "")
              for r in read][:6]
    msg = (f"no reading has a usable timestamp, so no run could be formed ({len(read)} files read)"
           if read else "no file could be read")
    if any("no date" in str(r.get("clock")) for r in read):
        msg += (". These are the 2019 NIHHIS tables whose clock is time_n (UTC HHMMSS without a date): give the "
                f"campaign's local date with --campaign-date \"{spec.name}=YYYY-MM-DD\" (from the city's report) or "
                "campaign_date in its catalog entry" + ("" if spec.tz else ", and a time zone for the city"))
    return msg + (": " + "; ".join(parts) if parts else "")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (dt.date, dt.datetime, pd.Timestamp)):
        return o.isoformat()
    return str(o)


# --------------------------------------------------------------------------- #
# Many cities                                                                  #
# --------------------------------------------------------------------------- #
def local_traverses(spec: CitySpec, traverses_dir: str | Path | None, single: bool) -> Path | None:
    """The city's folder under ``traverses_dir`` (``<slug>``, the catalog name, or the name before the comma);
    with one city, ``traverses_dir`` itself when it holds the files directly."""
    if not traverses_dir:
        return None
    root = Path(traverses_dir)
    if not root.exists():
        raise FileNotFoundError(f"--traverses-dir {root} does not exist")
    if root.is_file():
        return root if single else None
    for cand in (spec.slug, spec.name, spec.name.split(",")[0].strip(), slugify(spec.name.split(",")[0])):
        p = root / cand
        if p.exists():
            return p
    low = {q.name.lower(): q for q in root.iterdir()}
    for cand in (spec.slug, spec.name.lower(), spec.name.split(",")[0].strip().lower()):
        if cand in low:
            return low[cand]
    return root if single else None


def build_dataset(cities: list[str], out_dir: str | Path, *, traverses_dir: str | Path | None = None,
                  features: Callable | None = None, lst: Callable | bool | None = None,
                  era5: Callable | bool | None = None, force: bool = False, probe_opener: Callable | None = None,
                  download: Callable | None = None, campaign_dates: dict[str, str] | None = None, utc: bool = False,
                  **kw) -> dict:
    """Build (or resume) the dataset for ``cities``; returns the summary also written to ``OUT/dataset.json``.

    Cities with a finished cache are skipped unless ``force`` or the cache is stale (built with other settings,
    under an earlier :data:`DATASET_VERSION`, or before a change of the city's QA-registry entries; ``registry``
    in ``kw`` replaces the shipped registry, as in :func:`build_city`).  Before anything is fetched, every host the
    remaining cities need is probed (:func:`check_hosts`): a blocked host raises :class:`NetworkBlocked` at once.
    A city that fails for another reason is recorded with its error and the build moves on.  ``campaign_dates``
    maps a city (name as given, catalog name or slug) to its local campaign date, for traverse tables without
    one (:func:`build_city`).  ``utc``: every city's traverse clocks are UTC; each is converted to its own time zone
    (a city without one fails with that reason)."""
    out = Path(out_dir)
    (out / "cities").mkdir(parents=True, exist_ok=True)
    specs = [city_spec(c) for c in cities]
    single = len(specs) == 1
    wanted = {s.slug: settings_key(build_settings(**{**kw, "lst": lst, "era5": era5,
                                                     "campaign_date": _lookup(campaign_dates, s),
                                                     "qa_registry": registry_key(s.slug, kw.get("registry")),
                                                     **({"utc_to": s.tz} if utc else {})})) for s in specs}
    stale = {s.slug: _stale(out / "cities" / s.slug, wanted[s.slug]) for s in specs}
    todo = [s for s in specs if force or city_status(out / "cities" / s.slug) != "ok" or stale[s.slug]]
    for s in specs:
        if stale[s.slug] and not force and city_status(out / "cities" / s.slug) == "ok":
            log.info("lstair: %s is rebuilt: %s", s.name, stale[s.slug])
    local = {s.slug: local_traverses(s, traverses_dir, single) for s in todo}
    unavailable = {s.slug: _unavailable(s) for s in todo if local[s.slug] is None and _unavailable(s)}
    need_dl = [s for s in todo if local[s.slug] is None and s.slug not in unavailable and not _downloaded(out, s)]
    for s in need_dl:
        if not s.node:
            raise ValueError(f"{s.name}: not in the CAPA catalog and no local folder under --traverses-dir")
    todo = [s for s in todo if s.slug not in unavailable]
    names = hosts_needed(bool(need_dl), features is None and bool(todo), lst is None and bool(todo),
                         era5 is None and bool(todo))
    probes = check_hosts(names, opener=probe_opener) if names else {}
    summary = {"schema": SCHEMA, "cities": {}, "probes": probes}
    for s in specs:
        cdir = out / "cities" / s.slug
        if s.slug in unavailable:
            summary["cities"][s.name] = {"slug": s.slug, "status": "unavailable", "error": unavailable[s.slug]}
            continue
        if s not in todo:
            summary["cities"][s.name] = {"slug": s.slug, "status": "ok", "skipped": "cached"}
            continue
        try:
            ckw = dict(kw)
            if utc:
                if not s.tz:
                    raise ValueError("no time zone known for this city: its UTC clock (--utc) cannot be converted")
                ckw["utc_to"] = s.tz
            folder = local[s.slug]
            if folder is None:
                folder = out / "raw" / s.slug
                (download or _download_traverses)(s, folder)
            meta = build_city(s, folder, cdir, features=features, lst=lst, era5=era5,
                              campaign_date=_lookup(campaign_dates, s), **ckw)
            summary["cities"][s.name] = {"slug": s.slug, "status": "ok", "rows": meta["n_rows"],
                                         "runs": [r["run"] for r in meta["runs"] if "error" not in r],
                                         "excluded_runs": {r["run_id"]: r["qa_exclude"] for r in meta["runs"]
                                                           if r.get("qa_exclude")},
                                         "warnings": meta["warnings"]}
            log.info("lstair: %s — %d rows (%s)", s.name, meta["n_rows"], ", ".join(summary["cities"][s.name]["runs"]))
        except NetworkBlocked:
            raise
        except Exception as exc:                      # noqa: BLE001 - one city's failure is recorded
            msg = str(exc).splitlines()[0][:300] if str(exc) else repr(exc)[:300]
            log.warning("lstair: %s failed: %s", s.name, msg)
            cdir.mkdir(parents=True, exist_ok=True)
            _write_atomic(cdir / "meta.json", to_json({"schema": SCHEMA, "status": "failed", "city": s.name,
                                                          "slug": s.slug, "error": msg}).encode())
            summary["cities"][s.name] = {"slug": s.slug, "status": "failed", "error": msg}
    _write_atomic(out / "dataset.json", to_json(summary).encode())
    return summary


def _lookup(mapping: dict | None, spec: CitySpec):
    """``mapping``'s value for a city keyed by its catalog name, slug or the name before the comma."""
    if not mapping:
        return None
    keys = {spec.name.lower(), spec.slug, spec.name.split(",")[0].strip().lower()}
    return next((v for k, v in mapping.items() if str(k).strip().lower() in keys or slugify(str(k)) == spec.slug),
                None)


def _stale(city_dir: Path, key: str) -> str | None:
    """Why a finished city's cache does not match the requested settings (None when it does or is absent)."""
    p = Path(city_dir) / "meta.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("status") != "ok" or meta.get("settings_key") == key:
        return None
    if "settings_key" not in meta:
        return "built before settings were recorded"
    version = (meta.get("settings") or {}).get("dataset_version") or 1
    if version < DATASET_VERSION:
        return (f"built under dataset version {version}, before the QA rules of version {DATASET_VERSION} "
                "(see DATASET_VERSION)")
    return f"built with other settings ({meta.get('settings')})"


def _unavailable(spec: CitySpec) -> str | None:
    """Why the catalog says this city has no traverse points to download, else None."""
    if (spec.availability or {}).get("traverse_points") is False:
        return (f"the CAPA catalog lists no traverse points for {spec.name} (OSF node {spec.node}"
                + (f", folder {spec.folder_hint!r}" if spec.folder_hint else "") + "); pass --traverses-dir with "
                "its campaign folder if you have it")
    return None


def _downloaded(out: Path, spec: CitySpec) -> bool:
    """A finished download for this city's (node, folder hint) (a record for another node or hint is stale)."""
    p = out / "raw" / spec.slug / "_download.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(rec.get("complete")) and rec.get("node") == spec.node and rec.get("folder_hint") == spec.folder_hint


def _download_traverses(spec: CitySpec, dest: Path) -> dict:
    """The city's OSF files into ``dest``.  An HTTP answer (404 for a moved node, 410, 5xx) fails this city only;
    a connection that cannot be made (proxy refusal, DNS, timeout) raises :class:`NetworkBlocked` naming the host
    of the URL that failed."""
    import urllib.error
    from urllib.parse import urlparse

    from sparc.data.collect.capa import RASTER_ARCHIVE_PATTERNS, download_capa_files

    try:
        return download_capa_files(spec.node, dest, folder_hint=spec.folder_hint,
                                   skip_patterns=RASTER_ARCHIVE_PATTERNS + SKIP_FILES)
    except urllib.error.HTTPError as exc:
        url = getattr(exc, "sparc_url", None) or getattr(exc, "url", None) or ""
        raise RuntimeError(f"OSF answered HTTP {exc.code} ({exc.reason}) for {url or 'a request'} "
                           f"(node {spec.node})") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        _, hosts, purpose, remedy = HOSTS["osf"]
        url = getattr(exc, "sparc_url", None) or getattr(exc, "filename", None) or ""
        host = (urlparse(url).hostname if isinstance(url, str) and url else None) or hosts[0]
        raise NetworkBlocked(host, str(getattr(exc, "reason", exc))[:160], purpose, remedy) from exc


# --------------------------------------------------------------------------- #
# Reading the dataset back                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class CityData:
    slug: str
    meta: dict
    table: pd.DataFrame
    grid: CityGrid
    rasters: dict = field(default_factory=dict)


def load_city(city_dir: str | Path, rasters: bool = True) -> CityData:
    d = Path(city_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(d / "table.parquet")
    grid = CityGrid(**meta["grid"])
    r = {}
    if rasters and (d / "rasters.npz").exists():
        with np.load(d / "rasters.npz", allow_pickle=False) as z:
            r = {k: (z[k].astype(np.float64) if k != "driven" else z[k]) for k in z.files}
    return CityData(slug=meta["slug"], meta=meta, table=table, grid=grid, rasters=r)


def load_dataset(data_dir: str | Path, rasters: bool = True) -> list[CityData]:
    """Every finished city under ``data_dir`` (the ``--out`` of ``lstair dataset``, or its ``cities`` folder)."""
    root = Path(data_dir)
    root = root / "cities" if (root / "cities").is_dir() else root
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if city_status(d) == "ok" and (d / "table.parquet").exists():
            out.append(load_city(d, rasters))
    return out


__all__ = ["NetworkBlocked", "HOSTS", "probe", "check_hosts", "hosts_needed", "required_host_names", "CitySpec",
           "city_spec", "CityGrid", "make_city_grid", "adjust_visits", "co_visit_offsets", "focal_mean",
           "derived_rasters", "open_features", "landsat_lst", "era5_run", "span_weights", "build_run", "build_city",
           "build_dataset", "load_city", "load_dataset", "reusable_features", "platform_speed", "merged_logger_share",
           "sensor_name", "CityData", "FEATURES", "FOCAL_RADII", "LST_RADII", "TARGET_COLUMNS", "DATASET_VERSION",
           "SINGLE_VEHICLE"]
