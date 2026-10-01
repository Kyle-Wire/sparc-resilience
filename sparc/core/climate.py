"""Climate projections: CMIP6 change factors combined with the fitted model.

Goal 4 asks what air temperature would look like under new conditions.  This
module supplies the *new climate*; the scenario engine supplies the
*adaptation*:

    T_future(x) = T_observed(x) + ΔT_GCM(model, SSP, period) [+ Δ_adaptation(x)]

* ``ΔT_GCM`` is the change in summer (default June–August) mean daily maximum
  near-surface air temperature (CMIP6 ``Amon/tasmax``) at the study area,
  between the 1995–2014 baseline and an IPCC AR6 period (2021–2040,
  2041–2060, 2081–2100), for every model that provides the historical run and
  all requested SSPs.  One ensemble member per model (r1i1p1f1 where
  available) so each model gets one vote.
* The site value is a bilinear interpolation of the four surrounding GCM
  cells, weighted by land fraction (``fx/sftlf``) when available, so a coastal
  city is not represented by ocean cells that warm more slowly.
* ``Δ_adaptation`` is the model's re-predicted change for an adaptation
  scenario (S5).  It is applied unchanged under each future climate — the
  delta method assumes the local land-cover effect does not itself depend on
  the background warming.

Data come from the public CMIP6 archive on AWS (Pangeo zarr stores,
``s3://cmip6-pds``), read over plain HTTPS: only the time chunks that cover
the requested periods are downloaded (~50 MB each), decoded with
``numcodecs``, and reduced to the site at once.  Results are cached as a small
CSV so a run is reproducible offline (``climate.source: table``).
"""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BUCKET = "https://cmip6-pds.s3.amazonaws.com/"
CATALOG_URL = BUCKET + "pangeo-cmip6.csv"
EXPERIMENTS = ("ssp126", "ssp245", "ssp370", "ssp585")
SSP_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
PERIODS = {"2021-2040": (2021, 2040), "2041-2060": (2041, 2060), "2081-2100": (2081, 2100)}
BASELINE = (1995, 2014)


# --------------------------------------------------------------------------- #
# HTTP + zarr (v2) point reader                                                #
# --------------------------------------------------------------------------- #
def http_fetch(url: str, timeout: float = 180.0, retries: int = 4) -> bytes | None:
    """GET ``url``; None for a missing object (zarr: chunk of fill values).
    Transient failures (connection resets, truncated reads) are retried with
    exponential backoff."""
    import http.client
    import time
    import urllib.error
    import urllib.request

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return None
            if exc.code < 500 or attempt == retries:
                raise
        except (urllib.error.URLError, http.client.IncompleteRead, ConnectionError, TimeoutError, OSError):
            if attempt == retries:
                raise
        time.sleep(2.0 * 2 ** attempt)
    return None


def _store_url(zstore: str) -> str:
    return zstore.replace("s3://cmip6-pds/", BUCKET).rstrip("/") + "/"


class ZarrStore:
    """Minimal reader for a consolidated zarr v2 store over HTTP."""

    def __init__(self, url: str, fetch: Callable[[str], bytes | None] = http_fetch):
        self.url = url
        self.fetch = fetch
        raw = fetch(url + ".zmetadata")
        if raw is None:
            raise FileNotFoundError(url + ".zmetadata")
        self.meta = json.loads(raw)["metadata"]
        self._cache: dict[str, np.ndarray] = {}

    def array_meta(self, name: str) -> dict:
        return self.meta[f"{name}/.zarray"]

    def attrs(self, name: str) -> dict:
        return self.meta.get(f"{name}/.zattrs", {})

    def _decode(self, name: str, key: str) -> np.ndarray:
        import numcodecs

        am = self.array_meta(name)
        shape = tuple(am["chunks"])
        raw = self.fetch(f"{self.url}{name}/{key}")
        if raw is None:
            fill = am.get("fill_value")
            return np.full(shape, np.nan if fill is None else fill, dtype=np.dtype(am["dtype"]))
        buf = raw
        if am.get("compressor"):
            buf = numcodecs.get_codec(am["compressor"]).decode(buf)
        for f in reversed(am.get("filters") or []):
            buf = numcodecs.get_codec(f).decode(buf)
        arr = np.frombuffer(buf, dtype=np.dtype(am["dtype"]))
        return arr.reshape(shape, order=am.get("order", "C"))

    def chunk(self, name: str, idx: tuple[int, ...]) -> np.ndarray:
        key = ".".join(str(i) for i in idx)
        ck = f"{name}/{key}"
        if ck not in self._cache:
            self._cache[ck] = self._decode(name, key)
        return self._cache[ck]

    def read_all(self, name: str) -> np.ndarray:
        """A whole (small) array — coordinates."""
        am = self.array_meta(name)
        shape, chunks = am["shape"], am["chunks"]
        grid = [range(math.ceil(s / c)) for s, c in zip(shape, chunks)]
        out = np.empty(shape, dtype=np.dtype(am["dtype"]))
        for idx in np.ndindex(*[len(g) for g in grid]):
            block = self.chunk(name, idx)
            sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
            out[sl] = block[tuple(slice(0, s.stop - s.start) for s in sl)]
        return out

    def series(self, name: str, iy: int, ix: int, t0: int, t1: int) -> np.ndarray:
        """``name[t0:t1, iy, ix]`` for a (time, lat, lon) array, fetching only
        the chunks that cover it.  Chunks are released afterwards."""
        am = self.array_meta(name)
        ct, cy, cx = am["chunks"]
        out = np.empty(t1 - t0)
        for kt in range(t0 // ct, (t1 - 1) // ct + 1):
            block = self.chunk(name, (kt, iy // cy, ix // cx))
            a, b = max(t0, kt * ct), min(t1, (kt + 1) * ct)
            out[a - t0:b - t0] = block[a - kt * ct:b - kt * ct, iy % cy, ix % cx]
        fill = am.get("fill_value")
        if fill is not None:
            out[out == fill] = np.nan
        out[np.abs(out) > 1e19] = np.nan
        return out

    def drop_cache(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------------- #
# Time decoding (CF "<unit> since <date>", any CMIP6 calendar)                 #
# --------------------------------------------------------------------------- #
_DPY = {"noleap": 365.0, "365_day": 365.0, "360_day": 360.0, "all_leap": 366.0, "366_day": 366.0}


def decode_year_month(values: np.ndarray, units: str, calendar: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Year and month of monthly time stamps.  Exact enough for monthly means
    stamped mid-month in every CMIP6 calendar."""
    m = re.match(r"\s*(days|hours|minutes|seconds)\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})", units)
    if not m:
        raise ValueError(f"unsupported time units {units!r}")
    scale = {"days": 1.0, "hours": 1 / 24, "minutes": 1 / 1440, "seconds": 1 / 86400}[m.group(1)]
    y0, mo0, d0 = int(m.group(2)), int(m.group(3)), int(m.group(4))
    dpy = _DPY.get((calendar or "standard").lower(), 365.2425)
    start = y0 + ((mo0 - 1) * (dpy / 12.0) + (d0 - 1)) / dpy
    frac_year = start + np.asarray(values, dtype=float) * scale / dpy
    year = np.floor(frac_year).astype(int)
    month = np.clip(np.floor((frac_year - year) * 12.0).astype(int) + 1, 1, 12)
    return year, month


# --------------------------------------------------------------------------- #
# Site value from a GCM grid                                                   #
# --------------------------------------------------------------------------- #
def site_weights(lat: np.ndarray, lon: np.ndarray, site_lat: float, site_lon: float,
                 land: np.ndarray | None = None) -> list[tuple[int, int, float]]:
    """Bilinear weights of the four GCM cells around the site, re-weighted by
    land fraction when given (a coastal site must not be ocean-dominated)."""
    lon360 = np.mod(lon, 360.0)
    x = site_lon % 360.0
    lorder = np.argsort(lat)
    k1 = int(np.clip(np.searchsorted(lat[lorder], site_lat), 1, lat.size - 1))
    iy0, iy1 = int(lorder[k1 - 1]), int(lorder[k1])
    order = np.argsort(lon360)
    ls = lon360[order]
    j = int(np.searchsorted(ls, x)) % ls.size
    ix0, ix1 = int(order[j - 1]), int(order[j])
    dx = (lon360[ix1] - lon360[ix0]) % 360.0 or 1.0
    fx = ((x - lon360[ix0]) % 360.0) / dx
    fy = (site_lat - lat[iy0]) / ((lat[iy1] - lat[iy0]) or 1.0)
    fx, fy = float(np.clip(fx, 0, 1)), float(np.clip(fy, 0, 1))
    cells = [(iy0, ix0, (1 - fx) * (1 - fy)), (iy0, ix1, fx * (1 - fy)), (iy1, ix0, (1 - fx) * fy), (iy1, ix1, fx * fy)]
    if land is not None:
        lw = [(iy, ix, w * float(land[iy, ix])) for iy, ix, w in cells]
        if sum(w for *_, w in lw) > 1e-6:
            cells = lw
    tot = sum(w for *_, w in cells)
    return [(iy, ix, w / tot) for iy, ix, w in cells if w > 0]


# --------------------------------------------------------------------------- #
# Catalogue                                                                    #
# --------------------------------------------------------------------------- #
def load_catalog(cache_dir: Path, fetch: Callable[[str], bytes | None] = http_fetch) -> pd.DataFrame:
    """Pangeo CMIP6 catalogue, cached as a compact CSV of the rows we use."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    small = cache_dir / "pangeo-cmip6-tasmax-sftlf.csv"
    if small.exists():
        return pd.read_csv(small)
    raw = fetch(CATALOG_URL)
    if raw is None:
        raise FileNotFoundError(CATALOG_URL)
    import io

    df = pd.read_csv(io.BytesIO(raw))
    keep = ((df.table_id == "Amon") & (df.variable_id.isin(["tasmax", "tas"]))) | \
           ((df.table_id == "fx") & (df.variable_id == "sftlf"))
    df = df[keep & df.experiment_id.isin(("historical",) + EXPERIMENTS)]
    df.to_csv(small, index=False)
    return df


def _member_rank(m: str) -> tuple:
    nums = [int(v) for v in re.findall(r"\d+", m)]
    return (m != "r1i1p1f1", nums)


def select_runs(cat: pd.DataFrame, experiments=EXPERIMENTS, variable: str = "tasmax",
                models: list[str] | None = None) -> pd.DataFrame:
    """One member per model that has ``historical`` and every experiment."""
    d = cat[(cat.table_id == "Amon") & (cat.variable_id == variable)]
    need = {"historical", *experiments}
    rows = []
    for src, g in d.groupby("source_id"):
        if models and src not in models:
            continue
        ok = [mem for mem, gm in g.groupby("member_id") if need <= set(gm.experiment_id)]
        if not ok:
            continue
        mem = sorted(ok, key=_member_rank)[0]
        sel = g[(g.member_id == mem) & g.experiment_id.isin(need)].sort_values("version")
        rows.append(sel.groupby("experiment_id").tail(1))
    return pd.concat(rows, ignore_index=True) if rows else d.iloc[:0]


# --------------------------------------------------------------------------- #
# Change factors                                                               #
# --------------------------------------------------------------------------- #
def _seasonal_means(store: ZarrStore, var: str, cells, months: tuple[int, ...], y0: int, y1: int) -> float:
    t = store.read_all("time")
    ta = store.attrs("time")
    year, month = decode_year_month(t, ta.get("units", "days since 1850-01-01"), ta.get("calendar"))
    sel = np.flatnonzero((year >= y0) & (year <= y1) & np.isin(month, months))
    if sel.size == 0:
        return float("nan")
    t0, t1 = int(sel.min()), int(sel.max()) + 1
    acc = np.zeros(t1 - t0)
    for iy, ix, w in cells:
        acc += w * store.series(var, iy, ix, t0, t1)
    keep = sel - t0
    return float(np.nanmean(acc[keep]))


def _model_factors(model: str, runs: pd.DataFrame, cat: pd.DataFrame, site_lat: float, site_lon: float,
                   experiments, periods: dict, baseline: tuple[int, int], months, variable: str,
                   fetch) -> list[dict]:
    rows = []
    hist = runs[runs.experiment_id == "historical"].iloc[0]
    hs = ZarrStore(_store_url(hist.zstore), fetch)
    lat, lon = hs.read_all("lat").astype(float), hs.read_all("lon").astype(float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("curvilinear grid")
    land = None
    fxr = cat[(cat.source_id == model) & (cat.table_id == "fx") & (cat.variable_id == "sftlf")]
    if len(fxr):
        try:
            fs = ZarrStore(_store_url(fxr.iloc[0].zstore), fetch)
            land = fs.read_all("sftlf").astype(float)
            land = land / 100.0 if np.nanmax(land) > 1.5 else land
            land = np.nan_to_num(land) if land.shape == (lat.size, lon.size) else None
        except Exception as exc:  # land mask is a refinement, never a blocker
            log.info("climate: %s land fraction unavailable (%s)", model, exc)
    cells = site_weights(lat, lon, site_lat, site_lon, land)
    base = _seasonal_means(hs, variable, cells, months, *baseline)
    hs.drop_cache()
    for exp in experiments:
        r = runs[runs.experiment_id == exp].iloc[0]
        st = ZarrStore(_store_url(r.zstore), fetch)
        for pname, (y0, y1) in periods.items():
            fut = _seasonal_means(st, variable, cells, months, y0, y1)
            rows.append({"model": model, "member": r.member_id, "experiment": exp, "period": pname,
                         "baseline_K": base, "future_K": fut, "delta_K": fut - base,
                         "land_weighted": land is not None})
        st.drop_cache()
    log.info("climate: %s done (%s)", model, ", ".join(f"{x['experiment']} {x['period']} {x['delta_K']:+.2f} K"
                                                     for x in rows if x["period"] == list(periods)[-1]))
    return rows


def cmip6_change_factors(site_lat: float, site_lon: float, cache_dir: str | Path, experiments=EXPERIMENTS,
                         periods: dict | None = None, baseline: tuple[int, int] = BASELINE,
                         months: tuple[int, ...] = (6, 7, 8), variable: str = "tasmax",
                         models: list[str] | None = None, max_workers: int = 4,
                         fetch: Callable[[str], bytes | None] = http_fetch) -> pd.DataFrame:
    """Per-model change factors at a site (K), one row per model × SSP × period."""
    periods = dict(periods or PERIODS)
    cat = load_catalog(Path(cache_dir), fetch)
    runs = select_runs(cat, experiments, variable, models)
    names = sorted(runs.source_id.unique())
    log.info("climate: %d CMIP6 models with historical + %s", len(names), ", ".join(experiments))
    out: list[dict] = []

    def work(m):
        try:
            return _model_factors(m, runs[runs.source_id == m], cat, site_lat, site_lon, experiments, periods,
                                  baseline, months, variable, fetch)
        except Exception as exc:
            log.warning("climate: skipping %s (%s)", m, exc)
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rows in ex.map(work, names):
            out.extend(rows)
    df = pd.DataFrame(out)
    if len(df):
        df.insert(0, "site_lat", site_lat)
        df.insert(1, "site_lon", site_lon)
        df["months"] = "-".join(str(m) for m in months)
        df["variable"] = variable
        df["baseline"] = f"{baseline[0]}-{baseline[1]}"
    return df


# --------------------------------------------------------------------------- #
# Combine with observations and adaptation scenarios                           #
# --------------------------------------------------------------------------- #
def _pct(a, q):
    return float(np.percentile(np.asarray(a, float), q)) if len(a) else float("nan")


def summarize_projections(observed: np.ndarray, factors: pd.DataFrame, adaptation: dict[str, np.ndarray],
                          thresholds: list[float], to_units: float = 1.0) -> dict:
    """Future temperature, threshold exposure and adaptation offsets.

    ``factors``: rows with experiment, period, model, delta_K.  ``to_units``
    converts K to the target unit (1.8 for °F).  ``adaptation`` maps scenario
    name → per-point Δ from S5 (target units)."""
    obs = np.asarray(observed, float)
    thresholds = [float(t) for t in thresholds]
    present = {"mean": float(np.mean(obs)), "share_at_or_above": {str(t): float(np.mean(obs >= t)) for t in thresholds}}
    out = []
    for (exp, per), g in factors.groupby(["experiment", "period"], sort=True):
        d = g["delta_K"].to_numpy(float) * to_units
        d = d[np.isfinite(d)]
        if not d.size:
            continue
        row = {"experiment": exp, "label": SSP_LABELS.get(exp, exp), "period": per, "n_models": int(d.size),
               "warming": {"median": float(np.median(d)), "p10": _pct(d, 10), "p90": _pct(d, 90),
                           "min": float(d.min()), "max": float(d.max()),
                           "by_model": dict(zip(g.loc[np.isfinite(g["delta_K"]), "model"], d.tolist()))},
               "variants": []}
        for name, delta in [("no adaptation", None)] + list(adaptation.items()):
            add = 0.0 if delta is None else np.asarray(delta, float)
            shares = {str(t): [float(np.mean(obs + di + add >= t)) for di in d] for t in thresholds}
            mean_adapt = 0.0 if delta is None else float(np.mean(delta))
            row["variants"].append({
                "name": name,
                "mean": float(np.mean(obs) + np.median(d) + mean_adapt),
                "adaptation_mean_delta": mean_adapt,
                "offset_share_of_median_warming": (-mean_adapt / float(np.median(d))) if delta is not None
                and np.median(d) > 0 else None,
                "share_at_or_above": {t: {"median": float(np.median(v)), "p10": _pct(v, 10), "p90": _pct(v, 90)}
                                      for t, v in shares.items()},
            })
        out.append(row)
    return {"present": present, "thresholds": [float(t) for t in thresholds], "projections": out,
            "adaptation": list(adaptation)}
