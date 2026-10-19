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
   :mod:`sparc.core.identify`'s levels design does, a time polynomial and vehicle offsets are fitted jointly,
   here against **cell fixed effects** instead of land-cover terms, so no predictor enters the target:
   ``T_visit = μ_cell + β·τ (+ β₂·τ²) + o_vehicle + ε`` (τ in hours from the run's median time).  Drift is
   identified only from cells revisited at different times; vehicle offsets only between vehicles linked by
   shared cells, so they are centred (visit-weighted mean 0) within each linked group.  Real Heat Watch routes
   rarely give that: each vehicle drives its own route and comes back to a cell within minutes (Providence:
   every repeat within 4 min).  So the fixed-effect drift is used only when at least 20 cells were revisited
   15 min or more apart and its standard error is below 0.3 °C/h (:func:`adjust_visits`); otherwise the run
   falls back to CAPA's protocol: a straight line through the hour fitted to all readings (vehicle offsets 0),
   with the reason and both drift estimates in ``meta.json``.
4. **Target.**  ``Ta`` (°C) is the cell mean of the adjusted visits of the run; ``dTa = Ta − background`` with the
   background the run's median ``Ta`` over its cells.  ``ta_era5`` is ERA5's 2 m temperature over the run's
   hours at the city centre (:func:`sparc.core.forcing.era5_site`), when available.
5. **Predictors** (rasters on the city grid, sampled at the cells): canopy and impervious (%), NDVI, albedo,
   elevation (m) and distance to water (m) from :mod:`sparc.core.features_open` (WorldCover, Sentinel-2,
   Copernicus DEM), with Gaussian focal means (σ = radius/2, as the core's focal features) at 100, 300 and
   1000 m; LST (°C) of the clear Landsat 8/9 scene nearest the campaign date
   (:func:`sparc.data.collect.landsat.campaign_lst`) with focal means at 100 and 300 m, the scene date and its
   offset in days.  Predictors are computed from rasters only; the traverse temperatures never enter them.

Each city is cached in ``OUT/cities/<slug>/``: ``table.parquet``, ``rasters.npz`` (every raster on the full
grid: the physics model needs features beyond the driven cells) and ``meta.json`` (provenance, runs, drift,
scenes), written last, so a city with ``meta.json`` status ``ok`` is skipped on a rerun.  Hosts are probed
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

from sparc.core.lstair import RUN_CODE

log = logging.getLogger(__name__)

CELL_M = 30.0
MARGIN_M = 2000.0
FEATURES = ("canopy", "impervious", "ndvi", "albedo", "elevation", "water_distance")
FOCAL_FEATURES = ("canopy", "impervious", "ndvi", "albedo")
FOCAL_RADII = (100, 300, 1000)
LST_RADII = (100, 300)
MAX_CELLS = 6_000_000                  # ~70 km square at 30 m: larger grids are refused
STUDY_BUFFER_M = 300.0                 # LST clear fraction is measured within this distance of the traverses
TARGET_COLUMNS = ("Ta", "dTa", "ta_background", "n_readings", "n_visits")
#: File-name patterns not read by default: CAPA's appendix archives repeat the main traverses (Providence wu9v7:
#: the afternoon and evening runs again, plus three more sensors far outside the study area).
SKIP_FILES = ("appendix",)
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
    """Group id of every vehicle: vehicles are linked when they visited a common cell."""
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


def _protocol_line(temp: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, float]:
    """CAPA's detrending: a polynomial in time through every reading; (coefficients, SE of the linear one)."""
    X = np.column_stack([np.ones(P.shape[0]), P])
    b, *_ = np.linalg.lstsq(X, temp, rcond=None)
    dof = temp.size - X.shape[1]
    se = float("inf")
    if dof > 0:
        s2 = float(np.sum((temp - X @ b) ** 2)) / dof
        with np.errstate(all="ignore"):
            cov = s2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return b, se


def adjust_visits(v: pd.DataFrame, method: str = "fe", degree: int = 1, min_repeat_cells: int = 20,
                  min_span_s: float = MIN_REPEAT_SPAN_S, max_drift_se: float = MAX_DRIFT_SE) -> tuple[np.ndarray, dict]:
    """Visit temperatures with the run's drift and the vehicles' offsets removed.

    ``v`` has ``cell, vehicle, t_s, temp`` (one row per visit).  ``method``: ``"fe"`` (time polynomial and
    vehicle offsets fitted against cell fixed effects; module docstring), ``"protocol"`` (a polynomial in time
    fitted to all readings, CAPA's detrending; no vehicle offsets) or ``"none"``.

    Within cell fixed effects the drift is identified only by cells visited twice **at different times**: two
    passes a minute apart say almost nothing about the warming over an hour, and extrapolating from them puts
    an error along each route's time axis into the target (spatially structured, since time runs along the
    route).  So ``"fe"`` is used only when at least ``min_repeat_cells`` cells were revisited ``min_span_s`` or
    more apart (default 15 min) *and* the drift's standard error is below ``max_drift_se`` (default 0.3 °C/h);
    otherwise it falls back to ``"protocol"`` with the reason in ``info["fallback"]``.  The SE is the OLS one
    of the linear coefficient (Frisch–Waugh: σ̂²/‖M τ‖², with M the projection off the cell and vehicle effects
    and the other time terms; σ̂² from the fit's residuals and its degrees of freedom).

    Returns (adjusted temperatures, info) with ``info["method"]`` the method used, ``drift_per_h`` (the linear
    coefficient, temperature units per hour at the run's median time), ``drift_se_per_h``, ``vehicle_offsets``,
    ``n_repeat_cells`` and ``n_repeat_cells_spread`` (revisited at least ``min_span_s`` apart).  The adjustment
    is zero at the run's median time (τ = 0), so levels are those of mid-run."""
    temp = v["temp"].to_numpy(float)
    t = v["t_s"].to_numpy(float)
    tau = (t - np.median(t)) / 3600.0
    cells = v["cell"].to_numpy()
    veh = v["vehicle"].to_numpy()
    counts = pd.Series(cells).value_counts()
    n_rep = int((counts >= 2).sum())
    gt = pd.Series(t).groupby(cells)
    span = gt.max() - gt.min()
    n_spread = int((span >= float(min_span_s)).sum())
    info: dict = {"requested": method, "degree": int(degree), "n_visits": int(temp.size), "n_cells": int(counts.size),
                  "n_repeat_cells": n_rep, "n_repeat_cells_spread": n_spread, "min_span_s": float(min_span_s),
                  "max_drift_se": float(max_drift_se), "n_vehicles": int(np.unique(veh).size)}
    if method == "none" or temp.size < 3:
        info.update(method="none", drift_per_h=0.0, vehicle_offsets={})
        return temp.copy(), info
    if method not in ("fe", "protocol"):
        raise ValueError(f"unknown drift method {method!r} (fe, protocol, none)")
    P = np.column_stack([tau ** k for k in range(1, degree + 1)])

    def protocol(reason: str | None) -> tuple[np.ndarray, dict]:
        b, se = _protocol_line(temp, P)
        if reason:
            info["fallback"] = reason
        info.update(method="protocol", drift_per_h=float(b[1]), drift_se_per_h=se, drift_coef=[float(x) for x in b[1:]],
                    vehicle_offsets={})
        return temp - P @ b[1:], info

    if method == "protocol":
        return protocol(None)
    if n_spread < min_repeat_cells:
        return protocol(f"only {n_spread} cells revisited ≥ {min_span_s / 60:.0f} min apart (< {min_repeat_cells}; "
                        f"{n_rep} revisited at all): repeat visits minutes apart do not identify the drift over the "
                        "run, CAPA protocol detrending")
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsqr

    uc, ci = np.unique(cells, return_inverse=True)
    uv, vi = np.unique(veh, return_inverse=True)
    n = temp.size
    C = csr_matrix((np.ones(n), (np.arange(n), ci)), shape=(n, uc.size))
    V = csr_matrix((np.ones(n), (np.arange(n), vi)), shape=(n, uv.size))
    X = hstack([C, csr_matrix(P), V]).tocsr()
    lim = 20 * X.shape[1] + 1000
    y0 = temp - temp.mean()
    sol = lsqr(X, y0, atol=1e-12, btol=1e-12, iter_lim=lim)[0]
    beta = sol[uc.size:uc.size + degree]
    off = sol[uc.size + degree:].copy()
    comp = _components(cells, veh)
    n_groups = int(np.unique(comp).size)
    # SE of the linear drift coefficient (Frisch–Waugh)
    dof = n - (uc.size + degree + uv.size - n_groups)
    Z = hstack([C, csr_matrix(P[:, 1:]), V]).tocsr() if degree > 1 else hstack([C, V]).tocsr()
    e = P[:, 0] - Z @ lsqr(Z, P[:, 0], atol=1e-12, btol=1e-12, iter_lim=20 * Z.shape[1] + 1000)[0]
    ee = float(e @ e)
    se = float("inf")
    if dof > 0 and ee > 1e-12:
        s2 = float(np.sum((y0 - X @ sol) ** 2)) / dof
        se = math.sqrt(s2 / ee)
    info["fe_drift_per_h"], info["fe_drift_se_per_h"] = float(beta[0]), se
    if not se <= max_drift_se:
        return protocol(f"cell-fixed-effect drift {beta[0]:+.2f} ± {se:.2f} °C/h (SE above {max_drift_se} °C/h): "
                        "CAPA protocol detrending")
    w = np.bincount(vi, minlength=uv.size).astype(float)
    for g in np.unique(comp):
        sel = comp == g
        off[sel] -= np.average(off[sel], weights=w[sel])
    adj = temp - P @ beta - off[vi]
    info.update(method="fe", drift_per_h=float(beta[0]), drift_se_per_h=se, drift_coef=[float(x) for x in beta],
                vehicle_offsets={str(k): float(o) for k, o in zip(uv, off)}, n_vehicle_groups=n_groups)
    return adj, info


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
                min_clear: float = 0.5, cache_dir: Path | None = None) -> dict:
    """Campaign-matched Landsat LST (°C) on the city grid (:func:`sparc.data.collect.landsat.campaign_lst`).

    A search or read that a host refused raises :class:`NetworkBlocked` naming that host; when no scene could be
    kept because reads failed for another reason, RuntimeError (the city is recorded failed and retried): only
    "every inspected scene was read and too cloudy" gives NaN LST in a finished city."""
    from sparc.data.collect.landsat import campaign_lst, io_error_kind

    try:
        res = campaign_lst(ctx.bounds_lonlat(0.005), campaign_date, window_days=window_days, min_clear=min_clear,
                           target=ctx.target, study_mask=ctx.to_northup(study), cache_dir=cache_dir)
    except Exception as exc:                          # noqa: BLE001 - name a blocked host, else fail the city
        _raise_lst_failure([{"kind": io_error_kind(exc), "host": _failed_host(exc), "error": str(exc)[:200]}])
        raise
    check_lst_result(res)
    lst = None if res["lst_c"] is None else ctx.from_northup(res["lst_c"])
    return {"lst_c": lst, "scenes": res["scenes"], "candidates": res["candidates"], "window_days": window_days,
            "min_clear": min_clear, "qa_masked": res["qa_masked"], "status": res.get("status")}


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


def era5_run(lon: float, lat: float, tz: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """ERA5 at the city centre over the run's whole local hours: 2 m temperature (°C) and the physics forcing
    (SW↓ and LW net, W m⁻²; :func:`sparc.core.forcing.era5_site`)."""
    from sparc.core import forcing as F

    h0 = int(start.hour)
    h1 = int(end.hour) + (1 if (end.minute or end.second) else 0)
    h1 = max(h1, h0 + 1)
    if h1 > 23:
        h0, h1 = 22, 23
    times = F.window_utc(start.date().isoformat(), (h0, h1), tz)
    out = F.era5_site(lat, lon, times, analyses={k: F.ANALYSES[k] for k in ("t2m", "skt")},
                      accumulations={k: F.ACCUMULATIONS[k] for k in ("ssrd", "strd")})
    v = out["values"]
    lw_up = F.EMISSIVITY * F.SIGMA * v["skt"] ** 4
    return {"ta_era5": float(v["t2m"] - 273.15), "sw_down": float(v["ssrd"]), "lw_net": float(v["strd"] - lw_up),
            "hours_local": [h0, h1], "times_utc": out["times_utc"], "source": "ERA5 (NSF NCAR mirror), city centre"}


# --------------------------------------------------------------------------- #
# One city                                                                     #
# --------------------------------------------------------------------------- #
def _run_code(window: str) -> str:
    return RUN_CODE.get(str(window), str(window))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def city_status(city_dir: Path) -> str | None:
    p = Path(city_dir) / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


def build_city(spec: CitySpec, traverses: str | Path, out_dir: str | Path, *, features: Callable | None = None,
               lst: Callable | None = None, era5: Callable | None = None, cell_m: float = CELL_M,
               margin_m: float = MARGIN_M, drift: str = "fe", drift_degree: int = 1, utc_to: str | None = None,
               lst_window_days: int = 16, lst_min_clear: float = 0.5, cache_dir: Path | None = None,
               skip_files: tuple[str, ...] = SKIP_FILES, campaign_date: str | None = None) -> dict:
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
    and says so."""
    from sparc.core.identify.traverses import runs_summary, scan_traverses, to_campaign

    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    clock_date = campaign_date or spec.catalog_date
    points, report = scan_traverses(traverses, timezone=spec.tz, utc_to=utc_to, exclude=tuple(skip_files or ()),
                                    campaign_date=clock_date)
    hms = [r for r in report if r.get("clock")]
    n_dup = int(sum(r.get("duplicates", 0) for r in report))
    if n_dup:
        warnings.append(f"{n_dup:,} readings repeat another file's (same sensor, time and position): read once")
    if not len(points):
        skipped = "; ".join(f"{r['file']}: {r.get('reason')}" for r in report[:6])
        raise ValueError(f"no traverse points under {traverses}" + (f" (skipped {skipped})" if skipped else ""))
    points = points.dropna(subset=["lat", "lon", "temp_f"])
    ctx = make_city_grid(points["lon"].to_numpy(float), points["lat"].to_numpy(float), spec.crs, cell_m, margin_m)
    cfg, layout = ctx.identify_cfg(), ctx.identify_layout()
    hours = pd.to_datetime(points["time"]).dt.hour
    if hours.notna().any() and ((hours >= 9) & (hours < 12)).mean() > 0.25 and not hours.between(14, 17).any():
        warnings.append("no readings at 14–17 h but many at 9–12 h: the times may be UTC (rerun with --utc)")

    runs, frames = [], []
    for info in runs_summary(points):
        sub = points[points["run"] == info["run"]]
        rec = {"run_id": info["run"], "run": _run_code(info["window"]), "window": info["window"],
               "start": info["start"], "end": info["end"], "n_readings": int(info["n"]), "vehicles": info["vehicles"]}
        try:
            df, qa = to_campaign(sub, cfg, layout, window=None)
        except ValueError as exc:
            rec["error"] = str(exc)
            runs.append(rec)
            continue
        df = df.assign(temp=(df["temp"].to_numpy(float) - 32.0) * 5.0 / 9.0)        # °F → °C
        adj, dinfo = adjust_visits(df, drift, drift_degree)
        g = pd.DataFrame({"cell": df["cell"].to_numpy(), "Ta": adj, "n_raw": df["n_raw"].to_numpy()}).groupby("cell")
        cells = pd.DataFrame({"cell": g.size().index.to_numpy(), "Ta": g["Ta"].mean().to_numpy(),
                              "n_readings": g["n_raw"].sum().to_numpy(), "n_visits": g.size().to_numpy()})
        bg = float(np.median(cells["Ta"]))
        cells["dTa"] = cells["Ta"] - bg
        cells["ta_background"] = bg
        rec.update(qa=qa, drift={k: v for k, v in dinfo.items() if k != "vehicle_offsets"},
                   vehicle_offsets_c=dinfo.get("vehicle_offsets", {}), background_c=bg, n_cells=int(len(cells)),
                   run_date=str(pd.Timestamp(info["start"]).date()))
        cells["run"], cells["run_id"], cells["run_date"] = rec["run"], rec["run_id"], rec["run_date"]
        runs.append(rec)
        frames.append(cells)
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

    # rasters: features, focal means, LST
    driven = np.zeros(ctx.nx * ctx.ny, dtype=bool)
    driven[table["cell"].to_numpy()] = True
    from scipy import ndimage

    study = ndimage.distance_transform_edt(~driven.reshape(ctx.shape)) * ctx.dx <= STUDY_BUFFER_M
    feats, fprov = (features or open_features)(ctx, campaign)
    lst_res = None
    if lst is not False:
        lst_res = (lst or (lambda c, d, s: landsat_lst(c, d, s, lst_window_days, lst_min_clear, cache_dir)))(
            ctx, campaign, study)
        check_lst_result(lst_res)                     # a provider's read failures never pass as "no clear scene"
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
            e = (era5 or era5_run)(float(clon), float(clat), spec.tz, pd.Timestamp(rec["start"]), pd.Timestamp(rec["end"]))
            rec["era5"] = e
            table.loc[table["run_id"] == rec["run_id"], "ta_era5"] = e["ta_era5"]
        except Exception as exc:                      # noqa: BLE001 - ERA5 is optional
            rec["era5"] = {"error": str(exc).splitlines()[0][:200] if str(exc) else repr(exc)[:200]}
            warnings.append(f"ERA5 unavailable for {rec['run_id']}: {rec['era5']['error']}")

    table.insert(0, "city", spec.name)
    table.insert(1, "slug", spec.slug)
    table.insert(2, "campaign_date", campaign.isoformat())
    lead = ["city", "slug", "campaign_date", "run", "run_id", "run_date", "cell", "x", "y", "lat", "lon",
            "n_readings", "n_visits", "Ta", "dTa", "ta_background", "ta_era5"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    buf = io.BytesIO()
    table.to_parquet(buf, index=False)
    _write_atomic(out / "table.parquet", buf.getvalue())
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v.astype(np.float32) for k, v in rasters.items()},
                        driven=driven.reshape(ctx.shape))
    _write_atomic(out / "rasters.npz", buf.getvalue())
    meta = {"schema": SCHEMA, "status": "ok", "city": spec.name, "slug": spec.slug, "traverses": str(traverses),
            "campaign_date": campaign.isoformat(), "campaign_date_source": date_source,
            "catalog": {"node": spec.node, "folder_hint": spec.folder_hint, "campaign_date": spec.catalog_date,
                        "campaign_date_source": spec.catalog_date_source, "availability": spec.availability},
            "grid": ctx.as_dict(), "tz": spec.tz, "drift_method": drift, "runs": runs,
            "files": report, "skip_files": list(skip_files or ()), "n_duplicate_readings": n_dup, "features": fprov,
            "lst": None if lst_res is None else {k: v for k, v in lst_res.items() if k != "lst_c"},
            "n_rows": int(len(table)), "rasters": sorted(rasters), "warnings": warnings,
            "seconds": round(time.perf_counter() - t0, 1),
            "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    _write_atomic(out / "meta.json", json.dumps(meta, indent=1, default=_json_default).encode())
    return meta


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
                  download: Callable | None = None, campaign_dates: dict[str, str] | None = None, **kw) -> dict:
    """Build (or resume) the dataset for ``cities``; returns the summary also written to ``OUT/dataset.json``.

    Cities with a finished cache are skipped unless ``force``.  Before anything is fetched, every host the
    remaining cities need is probed (:func:`check_hosts`): a blocked host raises :class:`NetworkBlocked` at once.
    A city that fails for another reason is recorded with its error and the build moves on.  ``campaign_dates``
    maps a city (name as given, catalog name or slug) to its local campaign date, for traverse tables without
    one (:func:`build_city`)."""
    out = Path(out_dir)
    (out / "cities").mkdir(parents=True, exist_ok=True)
    specs = [city_spec(c) for c in cities]
    single = len(specs) == 1
    todo = [s for s in specs if force or city_status(out / "cities" / s.slug) != "ok"]
    local = {s.slug: local_traverses(s, traverses_dir, single) for s in todo}
    need_dl = [s for s in todo if local[s.slug] is None and not _downloaded(out, s)]
    for s in need_dl:
        if not s.node:
            raise ValueError(f"{s.name}: not in the CAPA catalog and no local folder under --traverses-dir")
    names = hosts_needed(bool(need_dl), features is None and bool(todo), lst is None and bool(todo),
                         era5 is None and bool(todo))
    probes = check_hosts(names, opener=probe_opener) if names else {}
    summary = {"schema": SCHEMA, "cities": {}, "probes": probes}
    for s in specs:
        cdir = out / "cities" / s.slug
        if s not in todo:
            summary["cities"][s.name] = {"slug": s.slug, "status": "ok", "skipped": "cached"}
            continue
        try:
            folder = local[s.slug]
            if folder is None:
                folder = out / "raw" / s.slug
                (download or _download_traverses)(s, folder)
            meta = build_city(s, folder, cdir, features=features, lst=lst, era5=era5,
                              campaign_date=_lookup(campaign_dates, s), **kw)
            summary["cities"][s.name] = {"slug": s.slug, "status": "ok", "rows": meta["n_rows"],
                                         "runs": [r["run"] for r in meta["runs"] if "error" not in r],
                                         "warnings": meta["warnings"]}
            log.info("lstair: %s — %d rows (%s)", s.name, meta["n_rows"], ", ".join(summary["cities"][s.name]["runs"]))
        except NetworkBlocked:
            raise
        except Exception as exc:                      # noqa: BLE001 - one city's failure is recorded
            msg = str(exc).splitlines()[0][:300] if str(exc) else repr(exc)[:300]
            log.warning("lstair: %s failed: %s", s.name, msg)
            cdir.mkdir(parents=True, exist_ok=True)
            _write_atomic(cdir / "meta.json", json.dumps({"schema": SCHEMA, "status": "failed", "city": s.name,
                                                          "slug": s.slug, "error": msg}, indent=1).encode())
            summary["cities"][s.name] = {"slug": s.slug, "status": "failed", "error": msg}
    _write_atomic(out / "dataset.json", json.dumps(summary, indent=1, default=_json_default).encode())
    return summary


def _lookup(mapping: dict | None, spec: CitySpec):
    """``mapping``'s value for a city keyed by its catalog name, slug or the name before the comma."""
    if not mapping:
        return None
    keys = {spec.name.lower(), spec.slug, spec.name.split(",")[0].strip().lower()}
    return next((v for k, v in mapping.items() if str(k).strip().lower() in keys or slugify(str(k)) == spec.slug),
                None)


def _downloaded(out: Path, spec: CitySpec) -> bool:
    return (out / "raw" / spec.slug / "_download.json").exists()


def _download_traverses(spec: CitySpec, dest: Path) -> dict:
    """The city's OSF files into ``dest``.  An HTTP answer (404 for a moved node, 410, 5xx) fails this city only;
    a connection that cannot be made (proxy refusal, DNS, timeout) raises :class:`NetworkBlocked` naming the host
    of the URL that failed."""
    import urllib.error
    from urllib.parse import urlparse

    from sparc.data.collect.capa import download_capa_files

    try:
        return download_capa_files(spec.node, dest, folder_hint=spec.folder_hint)
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
           "city_spec", "CityGrid", "make_city_grid", "adjust_visits", "focal_mean", "derived_rasters",
           "open_features", "landsat_lst", "era5_run", "build_city", "build_dataset", "load_city", "load_dataset",
           "CityData", "FEATURES", "FOCAL_RADII", "LST_RADII", "TARGET_COLUMNS"]
