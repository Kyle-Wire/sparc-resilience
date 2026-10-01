"""Campaign-day forcing: ERA5 reanalysis and the nearest weather station.

The physics source term needs the radiation and wind of the hour the target
was measured (``physics.sw_down``, ``lw_net``, ``wind``).  Generic values
(800 W m⁻², −100 W m⁻², no wind) are a placeholder; this module replaces them
with the conditions of the campaign window:

* **ERA5** (``s3://nsf-ncar-era5``, NetCDF4/HDF5 read over HTTPS range
  requests: only the compressed chunks around the site are fetched, ~10–20 MB
  per variable):

  - instantaneous analyses — 2 m temperature and dewpoint, 10 m wind, boundary
    layer height, total cloud cover, skin temperature;
  - hourly accumulations — surface downward shortwave (all-sky and clear-sky)
    and longwave, averaged over the window → W m⁻².

  ``lw_net = LW↓ − ε σ T_skin⁴`` (ε = 0.95).  The site value is the bilinear
  interpolation of the four 0.25° cells, weighted by ERA5 land fraction.

* **Station** (NOAA Global Hourly / ISD CSV, ``s3://noaa-global-hourly-pds``):
  temperature, dewpoint, wind and the NWS heat index in the window — a check
  on ERA5 at the site and the wind actually measured at the city.

The result is a small JSON file; point ``physics.forcing`` at it and the
config takes ``sw_down``, ``lw_net`` and ``wind`` from it (and records where
they came from).  Advection is then fitted from that wind, and the pipeline's
out-of-fold test (``physics.select_advection``) still decides whether it is
kept.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sparc.core.climate import http_fetch, site_weights

log = logging.getLogger(__name__)

ERA5 = "https://nsf-ncar-era5.s3.amazonaws.com/"
GLOBAL_HOURLY = "https://noaa-global-hourly-pds.s3.amazonaws.com/"
ERA5_EPOCH = dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)
SIGMA = 5.670374419e-8
EMISSIVITY = 0.95

# ERA5 parameter codes (NCAR file naming) for the analyses and accumulations used
ANALYSES = {"t2m": "128_167_2t", "d2m": "128_168_2d", "u10": "128_165_10u", "v10": "128_166_10v",
            "blh": "128_159_blh", "tcc": "128_164_tcc", "skt": "128_235_skt"}
ACCUMULATIONS = {"ssrd": "128_169_ssrd", "ssrdc": "228_129_ssrdc", "strd": "128_175_strd"}
LSM_KEY = "e5.oper.invariant/197901/e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
_COORDS = {"latitude", "longitude", "time", "utc_date", "forecast_initial_time", "forecast_hour"}


# --------------------------------------------------------------------------- #
# HTTP access                                                                  #
# --------------------------------------------------------------------------- #
def http_range(url: str, start: int, end: int, retries: int = 4) -> tuple[bytes, int]:
    """Bytes ``start..end`` (inclusive) of ``url`` and the object's total size."""
    import time
    import urllib.request

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                size = int(r.headers["Content-Range"].split("/")[1])
                return r.read(), size
        except Exception:                       # noqa: BLE001 - retried, then re-raised
            if attempt == retries:
                raise
            time.sleep(2.0 * 2 ** attempt)
    raise RuntimeError("unreachable")


class RangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests with a block cache
    (h5py accepts any such Python file object)."""

    def __init__(self, url: str, fetch_range: Callable[[str, int, int], tuple[bytes, int]] = http_range,
                 block: int = 1 << 20):
        self.url, self.fetch_range, self.block = url, fetch_range, int(block)
        self._cache: dict[int, bytes] = {}
        first, self.size = fetch_range(url, 0, self.block - 1)
        self._cache[0] = first
        self.pos = 0
        self.n_bytes = len(first)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            a = i * self.block
            data, _ = self.fetch_range(self.url, a, min(a + self.block, self.size) - 1)
            self._cache[i] = data
            self.n_bytes += len(data)
        return self._cache[i]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        p = self.pos
        while len(out) < n:
            i = p // self.block
            piece = self._block(i)[p - i * self.block:p - i * self.block + n - len(out)]
            if not piece:
                break
            out += piece
            p += len(piece)
        buf[:len(out)] = out
        self.pos += len(out)
        return len(out)


def s3_keys(bucket_url: str, prefix: str, fetch: Callable[[str], bytes | None] = http_fetch) -> list[str]:
    """Object keys under ``prefix`` (S3 ListObjectsV2 over HTTPS, paginated)."""
    import urllib.parse

    keys: list[str] = []
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        raw = fetch(bucket_url + "?" + urllib.parse.urlencode(q))
        if raw is None:
            break
        text = raw.decode("utf-8", "replace")
        keys += re.findall(r"<Key>([^<]+)</Key>", text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not m:
            break
        token = m.group(1)
    return keys


# --------------------------------------------------------------------------- #
# ERA5                                                                         #
# --------------------------------------------------------------------------- #
def _hours_since_epoch(t: dt.datetime) -> int:
    return int(round((t - ERA5_EPOCH).total_seconds() / 3600.0))


def _span(key: str) -> tuple[dt.datetime, dt.datetime]:
    a, b = re.search(r"\.(\d{10})_(\d{10})\.nc$", key).groups()
    f = "%Y%m%d%H"
    return (dt.datetime.strptime(a, f).replace(tzinfo=dt.timezone.utc),
            dt.datetime.strptime(b, f).replace(tzinfo=dt.timezone.utc))


def _era5_key(keys: list[str], code: str, t: dt.datetime) -> str:
    for k in keys:
        if f".{code}." in k:
            a, b = _span(k)
            if a <= t <= b:
                return k
    raise FileNotFoundError(f"no ERA5 file for {code} covering {t:%Y-%m-%d %H}Z")


def _open_h5(url: str, fetch_range):
    import h5py

    f = RangeFile(url, fetch_range)
    return h5py.File(f, "r"), f


def _data_var(h) -> str:
    names = [k for k in h if k not in _COORDS]
    if len(names) != 1:
        raise ValueError(f"expected one data variable, found {names}")
    return names[0]


def accumulation_init(t: dt.datetime) -> dt.datetime:
    """06Z/18Z forecast initialisation whose hourly steps 1–12 include the hour ending at ``t``."""
    p = t - dt.timedelta(hours=1)
    init = p.replace(hour=6 if 6 <= p.hour < 18 else 18, minute=0, second=0, microsecond=0)
    return init - dt.timedelta(days=1) if p.hour < 6 else init


def _box(cells) -> tuple[slice, slice]:
    iy = [c[0] for c in cells]
    ix = [c[1] for c in cells]
    return slice(min(iy), max(iy) + 1), slice(min(ix), max(ix) + 1)


def _weighted(field: np.ndarray, cells, box) -> np.ndarray:
    """Weighted site value of ``field[..., y, x]`` cropped to ``box``."""
    return sum(w * field[..., iy - box[0].start, ix - box[1].start] for iy, ix, w in cells)


def era5_site(site_lat: float, site_lon: float, times_utc: list[dt.datetime],
              fetch: Callable[[str], bytes | None] = http_fetch,
              fetch_range: Callable[[str, int, int], tuple[bytes, int]] = http_range,
              analyses: dict | None = None, accumulations: dict | None = None) -> dict:
    """ERA5 values at the site.  Analyses are averaged over ``times_utc``;
    accumulations over the hours *ending* at ``times_utc[1:]`` (the window's
    hours), converted to W m⁻²."""
    analyses = ANALYSES if analyses is None else analyses
    accumulations = ACCUMULATIONS if accumulations is None else accumulations
    try:
        h, _ = _open_h5(ERA5 + LSM_KEY, fetch_range)
        land = h[_data_var(h)][0]
        h.close()
    except Exception as exc:                    # noqa: BLE001 - land weighting is optional
        log.warning("ERA5 land-sea mask unavailable (%s); unweighted bilinear", exc)
        land = None
    out: dict = {"times_utc": [t.strftime("%Y-%m-%dT%H:%MZ") for t in times_utc], "values": {}, "series": {}}
    listings: dict[str, list[str]] = {}
    opened: dict[str, tuple] = {}

    def keys(prefix):
        if prefix not in listings:
            listings[prefix] = s3_keys(ERA5, prefix, fetch)
        return listings[prefix]

    def site_field(key):
        """(h5 file, variable, weighted-cell list, crop box) — opened once per file."""
        if key not in opened:
            h, f = _open_h5(ERA5 + key, fetch_range)
            cells = site_weights(h["latitude"][:], h["longitude"][:], site_lat, site_lon, land)
            opened[key] = (h, f, _data_var(h), cells, _box(cells))
            out.setdefault("cells", [[float(h["latitude"][iy]), float(h["longitude"][ix]), float(w)]
                                     for iy, ix, w in cells])
        return opened[key]

    try:
        for name, code in analyses.items():
            vals = []
            for t in times_utc:
                h, _, var, cells, box = site_field(_era5_key(keys(f"e5.oper.an.sfc/{t:%Y%m}/"), code, t))
                ti = int(np.flatnonzero(h["time"][:] == _hours_since_epoch(t))[0])
                vals.append(float(_weighted(h[var][ti, box[0], box[1]], cells, box)))
            out["series"][name] = vals
            out["values"][name] = float(np.mean(vals))
        # accumulation over the hour ending at t comes from the 06Z/18Z forecast
        # initialised before it (steps 1–12 are hourly, not cumulative)
        ends = times_utc[1:] if len(times_utc) > 1 else times_utc
        for name, code in accumulations.items():
            vals = []
            for t in ends:
                init = accumulation_init(t)
                h, _, var, cells, box = site_field(_era5_key(keys(f"e5.oper.fc.sfc.accumu/{init:%Y%m}/"), code, init))
                ii = int(np.flatnonzero(h["forecast_initial_time"][:] == _hours_since_epoch(init))[0])
                step = int(round((t - init).total_seconds() / 3600.0))
                si = int(np.flatnonzero(h["forecast_hour"][:] == step)[0])
                vals.append(float(_weighted(h[var][ii, si, box[0], box[1]], cells, box)) / 3600.0)
            out["series"][name] = vals
            out["values"][name] = float(np.mean(vals))
    finally:
        out["bytes_read"] = int(sum(f.n_bytes for _, f, *_ in opened.values()))
        for h, *_ in opened.values():
            h.close()
    return out


# --------------------------------------------------------------------------- #
# Station (NOAA Global Hourly)                                                  #
# --------------------------------------------------------------------------- #
def _isd_value(s: pd.Series, scale: float, missing: str) -> pd.Series:
    parts = s.astype(str).str.split(",", expand=True)
    v = pd.to_numeric(parts[0], errors="coerce")
    q = parts[1] if parts.shape[1] > 1 else pd.Series("1", index=s.index)
    bad = (parts[0].str.lstrip("+-") == missing.lstrip("+-")) | ~q.isin(["0", "1", "4", "5", "9", "A", "C", "I", "M",
                                                                          "P", "R", "U"])
    return (v / scale).where(~bad)


def station_obs(station: str, start_utc: dt.datetime, end_utc: dt.datetime,
                fetch: Callable[[str], bytes | None] = http_fetch, cache_dir: str | Path | None = None) -> dict:
    """Observations of a NOAA Global Hourly station (USAF+WBAN id, e.g.
    ``72507014765`` = Providence T.F. Green) between two UTC times."""
    year = start_utc.year
    raw = None
    cache = Path(cache_dir) / f"global_hourly_{station}_{year}.csv" if cache_dir else None
    if cache is not None and cache.exists():
        raw = cache.read_bytes()
    if raw is None:
        raw = fetch(f"{GLOBAL_HOURLY}{year}/{station}.csv")
        if raw is None:
            raise FileNotFoundError(f"no Global Hourly file for station {station} in {year}")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
    t = pd.to_datetime(df["DATE"], utc=True)
    pad = pd.Timedelta(minutes=30)              # hourly METARs are issued at ~:51
    df = df.loc[(t >= pd.Timestamp(start_utc) - pad) & (t <= pd.Timestamp(end_utc) + pad)].copy()
    if df.empty:
        raise ValueError(f"station {station}: no reports between {start_utc} and {end_utc}")
    T = _isd_value(df["TMP"], 10.0, "+9999")
    Td = _isd_value(df["DEW"], 10.0, "+9999")
    w = df["WND"].astype(str).str.split(",", expand=True)
    wdir = pd.to_numeric(w[0], errors="coerce").where(lambda x: x != 999)
    wspd = (pd.to_numeric(w[3], errors="coerce") / 10.0).where(lambda x: x < 999)
    calm = w[2] == "C"
    wdir = wdir.where(~calm)
    wspd = wspd.where(~calm, 0.0)
    ok = wdir.notna() | calm                                    # variable / missing directions carry no vector
    u = -(wspd * np.sin(np.deg2rad(wdir.fillna(0.0))))[ok]       # wind blowing *towards* east / north
    v = -(wspd * np.cos(np.deg2rad(wdir.fillna(0.0))))[ok]
    out = {
        "station": station, "name": str(df["NAME"].iloc[0]) if "NAME" in df else None,
        "lat": float(pd.to_numeric(df["LATITUDE"]).iloc[0]) if "LATITUDE" in df else None,
        "lon": float(pd.to_numeric(df["LONGITUDE"]).iloc[0]) if "LONGITUDE" in df else None,
        "n_reports": int(len(df)),
        "t_C": float(T.mean()), "td_C": float(Td.mean()), "t_max_C": float(T.max()),
        "wind_speed": float(wspd.mean()), "u": float(u.mean()), "v": float(v.mean()),
    }
    out["wind_from_deg"] = wind_from(out["u"], out["v"])
    out["rh"] = relative_humidity(out["t_C"], out["td_C"])
    out["heat_index_F"] = heat_index_F(c_to_f(out["t_C"]), out["rh"])
    return out


# --------------------------------------------------------------------------- #
# Meteorology helpers                                                          #
# --------------------------------------------------------------------------- #
def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def relative_humidity(t_C: float, td_C: float) -> float:
    """Magnus formula (Alduchov & Eskridge 1996 constants), percent."""
    a, b = 17.625, 243.04
    return float(100.0 * math.exp(a * td_C / (b + td_C) - a * t_C / (b + t_C)))


def heat_index_F(t_F: float, rh: float) -> float:
    """NWS heat index (Rothfusz regression with the NWS adjustments)."""
    simple = 0.5 * (t_F + 61.0 + (t_F - 68.0) * 1.2 + rh * 0.094)
    if (simple + t_F) / 2.0 < 80.0:
        return float(simple)
    hi = (-42.379 + 2.04901523 * t_F + 10.14333127 * rh - 0.22475541 * t_F * rh - 6.83783e-3 * t_F ** 2
          - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t_F ** 2 * rh + 8.5282e-4 * t_F * rh ** 2 - 1.99e-6 * t_F ** 2 * rh ** 2)
    if rh < 13 and 80 <= t_F <= 112:
        hi -= ((13 - rh) / 4.0) * math.sqrt((17 - abs(t_F - 95.0)) / 17.0)
    elif rh > 85 and 80 <= t_F <= 87:
        hi += ((rh - 85) / 10.0) * ((87 - t_F) / 5.0)
    return float(hi)


def wind_from(u: float, v: float) -> float:
    """Meteorological direction the wind blows *from* (degrees clockwise from north)."""
    return float((math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0)


def window_utc(date: str, hours: tuple[int, int], tz: str) -> list[dt.datetime]:
    """Whole UTC hours from local ``hours[0]`` to ``hours[1]`` on ``date``."""
    from zoneinfo import ZoneInfo

    d = dt.date.fromisoformat(date)
    z = ZoneInfo(tz)
    h0, h1 = int(hours[0]), int(hours[1])
    if h1 <= h0:
        raise ValueError("hours must be an increasing local range, e.g. 15-16")
    return [dt.datetime(d.year, d.month, d.day, h, tzinfo=z).astimezone(dt.timezone.utc) for h in range(h0, h1 + 1)]


# --------------------------------------------------------------------------- #
# Campaign forcing                                                             #
# --------------------------------------------------------------------------- #
def campaign_forcing(site_lat: float, site_lon: float, date: str, hours: tuple[int, int], tz: str,
                     station: str | None = None, wind_source: str = "auto",
                     fetch: Callable[[str], bytes | None] = http_fetch,
                     fetch_range: Callable[[str, int, int], tuple[bytes, int]] = http_range,
                     cache_dir: str | Path | None = None) -> dict:
    """ERA5 + station conditions of the campaign window and the physics
    forcing derived from them (``physics`` block of the result)."""
    times = window_utc(date, hours, tz)
    era = era5_site(site_lat, site_lon, times, fetch=fetch, fetch_range=fetch_range)
    e = era["values"]
    t_C, td_C = e["t2m"] - 273.15, e["d2m"] - 273.15
    lw_up = EMISSIVITY * SIGMA * e["skt"] ** 4
    era_out = {
        **era,
        "t2m_C": t_C, "t2m_F": c_to_f(t_C), "d2m_C": td_C, "rh": relative_humidity(t_C, td_C),
        "heat_index_F": heat_index_F(c_to_f(t_C), relative_humidity(t_C, td_C)),
        "u10": e["u10"], "v10": e["v10"], "wind_speed": float(math.hypot(e["u10"], e["v10"])),
        "wind_from_deg": wind_from(e["u10"], e["v10"]),
        "blh_m": e["blh"], "tcc": e["tcc"], "skt_C": e["skt"] - 273.15,
        "sw_down": e["ssrd"], "sw_down_clear": e.get("ssrdc"), "lw_down": e["strd"], "lw_up": lw_up,
        "lw_net": e["strd"] - lw_up,
        "clear_sky_index": (e["ssrd"] / e["ssrdc"]) if e.get("ssrdc") else None,
    }
    st = None
    if station:
        try:
            st = station_obs(station, times[0], times[-1], fetch=fetch, cache_dir=cache_dir)
        except Exception as exc:                # noqa: BLE001 - the station is a cross-check
            log.warning("station %s unavailable: %s", station, exc)
    src = wind_source if wind_source != "auto" else ("station" if st else "era5")
    if src == "station" and not st:
        raise ValueError("wind_source=station but no station observations")
    wind = [st["u"], st["v"]] if src == "station" else [e["u10"], e["v10"]]
    checks = []
    if st:
        dT = c_to_f(st["t_C"]) - era_out["t2m_F"]
        checks.append(f"station − ERA5 2 m temperature: {dT:+.1f} °F (ERA5 cells are 0.25°, partly rural/water)")
        if st["wind_speed"] > 1.0 and era_out["wind_speed"] > 1.0:
            dd = abs((st["wind_from_deg"] - era_out["wind_from_deg"] + 180.0) % 360.0 - 180.0)
            checks.append(f"station vs ERA5 wind direction differ by {dd:.0f}°"
                          + (" — local flow (e.g. a sea breeze) not resolved by ERA5" if dd > 45 else ""))
    csi = era_out["clear_sky_index"]
    if csi is not None:
        checks.append(f"clear-sky index {csi:.2f}" + (" (clear)" if csi > 0.85 else " (cloud reduced sunshine)"))
    return {
        "date": date, "hours_local": [int(hours[0]), int(hours[1])], "tz": tz, "site": [site_lat, site_lon],
        "era5": era_out, "station": st, "wind_source": src, "checks": checks,
        "physics": {"window": "day" if era_out["sw_down"] > 50.0 else "night",
                    "sw_down": round(float(era_out["sw_down"]), 1), "lw_net": round(float(era_out["lw_net"]), 1),
                    "wind": [round(float(wind[0]), 3), round(float(wind[1]), 3)]},
    }


def apply_forcing_file(physics: dict, path: Path, label: str | None = None) -> dict:
    """Merge a forcing JSON's ``physics`` values into a physics config block
    (``label``: the path as written in the config, recorded for provenance)."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    out = dict(physics)
    for k in ("window", "sw_down", "lw_net", "wind"):
        if k in blob.get("physics", {}):
            out[k] = blob["physics"][k]
    out["forcing_info"] = {"file": label or Path(path).name, "date": blob.get("date"), "hours_local": blob.get("hours_local"),
                           "tz": blob.get("tz"), "wind_source": blob.get("wind_source"),
                           "station": (blob.get("station") or {}).get("station"),
                           "checks": blob.get("checks", [])}
    return out
