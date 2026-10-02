"""Deterministic stand-ins for the network fetchers the input kinds call (no network in tests).

:func:`install` patches the library functions in place; the job workers of
``test_inputs.py`` run :mod:`fake_worker`, which calls it before the real
worker, and in-process tests use it through ``monkeypatch``.  The fakes have
the real functions' signatures and return values of the real shapes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEMO_SITE = (41.828, -71.378)            # the synthetic demo's fictional site (EPSG:32619 offset)
ISD_ROWS = [
    ("725070", "14765", "PROVIDENCE T F GREEN STATE AIRPORT", "+41.722", "-071.433", "19420101", "20260930"),
    ("725074", "54752", "PAWTUCKET NORTH CENTRAL AIRPORT", "+41.921", "-071.491", "19980101", "20260930"),
    ("725088", "94746", "NORWOOD MEMORIAL AIRPORT", "+42.191", "-071.174", "19730101", "20260930"),
    ("744904", "04780", "BLOCK ISLAND STATE AIRPORT", "+41.168", "-071.578", "19890101", "20260930"),
    ("999999", "99999", "UNKNOWN PLACEHOLDER", "+00.000", "+000.000", "20110309", "20130730"),
]


def fake_campaign_forcing(site_lat: float, site_lon: float, date: str, hours, tz: str, station: str | None = None,
                          wind_source: str = "auto", fetch=None, fetch_range=None, cache_dir=None) -> dict:
    era = {"t2m_C": 31.0, "t2m_F": 87.8, "d2m_C": 18.0, "rh": 46.0, "heat_index_F": 89.5, "u10": 2.1, "v10": 3.4,
           "wind_speed": 4.0, "wind_from_deg": 211.7, "sw_down": 636.0, "lw_net": -71.0, "clear_sky_index": 0.87}
    st = None
    if station:
        st = {"station": station, "name": "FAKE STATION", "t_C": 30.8, "td_C": 18.2, "rh": 47.0,
              "heat_index_F": 89.0, "wind_speed": 7.7, "u": 0.4, "v": 7.6, "wind_from_deg": 183.0, "n_reports": 3}
    src = wind_source if wind_source != "auto" else ("station" if st else "era5")
    wind = [st["u"], st["v"]] if src == "station" and st else [era["u10"], era["v10"]]
    return {"date": date, "hours_local": [int(hours[0]), int(hours[1])], "tz": tz, "site": [site_lat, site_lon],
            "era5": era, "station": st, "wind_source": src, "checks": ["clear-sky index 0.87 (clear)"],
            "physics": {"window": "day", "sw_down": 636.0, "lw_net": -71.0, "wind": [round(w, 3) for w in wind]}}


def fake_fetch_layers(data, cfg, hrsl=None, worldcover: bool = True) -> pd.DataFrame:
    g = data.grid
    n = data.n
    built = ((g.ix % 5) / 5.0).astype(float)
    tree = np.clip(1.0 - built, 0.0, 1.0) * 0.5
    people = 20.0 * built + 1.0
    return pd.DataFrame({"id": data.ids, "x_m": data.x, "y_m": data.y_coord, "people": people,
                         "people_60_plus": people * 0.15, "people_under_5": people * 0.06, "lc_tree": tree,
                         "lc_shrub": np.zeros(n), "lc_grass": 1.0 - built - tree, "lc_crop": np.zeros(n),
                         "lc_built": built, "lc_bare": np.zeros(n), "lc_water": np.zeros(n), "lc_wetland": np.zeros(n)})


def fake_build_open_features(data, cfg, months=((2020, 6),), max_cloud: float = 20.0,
                             s2_tiles=None) -> tuple[pd.DataFrame, dict]:
    g = data.grid
    rng = np.random.default_rng(0)
    roles = (cfg.raw.get("physics") or {}).get("roles") or {}

    def base(role, lo, hi):
        col = roles.get(role)
        if col and col in data.frame:
            v = data.frame[col].to_numpy(float)
            return v + rng.normal(0, 0.05 * (np.std(v) + 1e-9), len(v))
        return lo + (hi - lo) * ((g.ix * 7 + g.iy * 3) % 11) / 10.0

    df = pd.DataFrame({"id": data.ids, "canopy": base("canopy", 0, 80), "impervious": base("impervious", 0, 90),
                       "water_distance": base("water_distance", 0, 900), "elevation": base("elevation", 5, 60),
                       "ndvi": base("ndvi", 0.05, 0.7), "albedo": base("albedo", 0.08, 0.25)})
    prov = {"worldcover": "FAKE", "dem": "FAKE", "sentinel2": {"scenes": ["FAKE_SCENE"], "max_cloud": max_cloud,
                                                               "months": [list(m) for m in months]}}
    return df, prov


def fake_cmip6_change_factors(site_lat: float, site_lon: float, cache_dir, experiments=("ssp245",), periods=None,
                              baseline=(1995, 2014), months=(6, 7, 8), variable: str = "tasmax", models=None,
                              max_workers: int = 4, fetch=None, on_models=None) -> pd.DataFrame:
    periods = dict(periods or {"2041-2060": (2041, 2060)})
    names = list(models or ["FAKE-A", "FAKE-B", "FAKE-C"])
    if on_models is not None:
        on_models(len(names))
    rows = []
    for i, m in enumerate(names):
        for e_i, e in enumerate(experiments):
            for p_i, (pname, _years) in enumerate(periods.items()):
                d = round(0.8 + 0.3 * i + 0.5 * e_i + 0.6 * p_i, 4)
                rows.append({"model": m, "member": "r1i1p1f1", "experiment": e, "period": pname,
                             "baseline_K": 300.0, "future_K": 300.0 + d, "delta_K": d, "land_weighted": True})
    df = pd.DataFrame(rows)
    df.insert(0, "site_lat", site_lat)
    df.insert(1, "site_lon", site_lon)
    df["months"] = "-".join(str(m) for m in months)
    df["variable"] = variable
    df["baseline"] = f"{baseline[0]}-{baseline[1]}"
    return df


def _ghcn_csv(station: str) -> bytes:
    rows = ["ID,DATE,ELEMENT,DATA_VALUE,M_FLAG,Q_FLAG,S_FLAG,OBS_TIME"]
    for y in range(1995, 2015):
        for d in pd.date_range(f"{y}-06-01", f"{y}-08-31", freq="D"):
            tenths = 270 + (d.dayofyear * 7 + y) % 90
            rows.append(f"{station},{d:%Y%m%d},TMAX,{tenths},,,7,")
            rows.append(f"{station},{d:%Y%m%d},TMIN,{tenths - 100},,,7,")
    return ("\n".join(rows) + "\n").encode()


def _isd_csv() -> bytes:
    head = '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"'
    lines = [head] + [f'"{u}","{w}","{n}","US","RI","","{la}","{lo}","+10.0","{b}","{e}"'
                      for u, w, n, la, lo, b, e in ISD_ROWS]
    return ("\n".join(lines) + "\n").encode()


def fake_http_fetch(url: str, timeout: float = 180.0, retries: int = 4) -> bytes | None:
    if "noaa-ghcn-pds" in url:
        station = url.rsplit("/", 1)[-1].removesuffix(".csv")
        return _ghcn_csv(station)
    if url.endswith("isd-history.csv"):
        return _isd_csv()
    return None


def install(target: Any = None) -> None:
    """Patch the fetchers (``target``: a pytest ``monkeypatch``, or None to assign directly)."""
    import sparc.core.climate as climate
    import sparc.core.features_open as features_open
    import sparc.core.forcing as forcing
    import sparc.core.opendata as opendata

    pairs = [(forcing, "campaign_forcing", fake_campaign_forcing), (opendata, "fetch_layers", fake_fetch_layers),
             (features_open, "build_open_features", fake_build_open_features),
             (climate, "cmip6_change_factors", fake_cmip6_change_factors), (climate, "http_fetch", fake_http_fetch)]
    for mod, name, fn in pairs:
        if target is None:
            setattr(mod, name, fn)
        else:
            target.setattr(mod, name, fn)
