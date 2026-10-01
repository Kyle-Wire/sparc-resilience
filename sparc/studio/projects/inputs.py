"""Project inputs (SPEC §9.3 step 4, api.md §5.4): where fetched files live, how they link into the config,
their state and chart views, and the ISD station lookup.

Files (project-relative)::

    inputs/forcing/forcing_<date>_<h0>-<h1>.json      → physics.forcing
    inputs/climate/cmip6_<variable>_<months>.csv      → climate.table (+ source, enabled, periods, experiments)
    inputs/layers/layers.parquet                       → planner.layers
    inputs/features/open_features.parquet (+ .json)    → data.join + open_* predictors (this project or <name>_open)
    inputs/ghcn.json                                   the fetched GHCN station (series cached in <ws>/cache)

The link edits here are pure functions of the raw config; job workers apply
them with :func:`~sparc.studio.projects.config_service.edit_config_file`
and the server with ``save_config`` (``POST /link``).  The station lookup
only reads ``<ws>/cache/isd-history.csv``; it never downloads.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sparc.studio.errors import ApiError
from sparc.studio.projects.config_schema import ROLE_NAMES
from sparc.studio.projects.config_service import absolutize_paths, get_dotted, set_dotted
from sparc.studio.workspace import read_json

__all__ = ["FORCING_DIR", "CLIMATE_DIR", "LAYERS_DIR", "FEATURES_DIR", "FEATURES_FILE", "GHCN_RECORD",
           "ISD_FILE", "ISD_URLS", "forcing_name", "cmip6_name", "link_forcing", "link_climate", "link_layers",
           "link_features", "open_project_raw", "inputs_state", "input_view", "nearest_stations",
           "LINK_KINDS", "site_of", "OPEN_CLIP", "open_rename_map"]

FORCING_DIR = "inputs/forcing"
CLIMATE_DIR = "inputs/climate"
LAYERS_DIR = "inputs/layers"
FEATURES_DIR = "inputs/features"
FEATURES_FILE = f"{FEATURES_DIR}/open_features.parquet"
LAYERS_FILE = f"{LAYERS_DIR}/layers.parquet"
GHCN_RECORD = "inputs/ghcn.json"
ISD_FILE = "isd-history.csv"
ISD_URLS = ("https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv",
            "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv")
LINK_KINDS = ("forcing", "climate", "layers", "features_join", "features_new_project")
OPEN_CLIP = {"open_canopy": [0, 100], "open_impervious": [0, 100], "open_albedo": [0.02, 0.9]}
DEFAULT_PERIODS = {"2021-2040": [2021, 2040], "2041-2060": [2041, 2060], "2081-2100": [2081, 2100]}
DEFAULT_EXPERIMENTS = ["ssp126", "ssp245", "ssp370", "ssp585"]


def forcing_name(date: str, hours) -> str:
    return f"{FORCING_DIR}/forcing_{date}_{int(hours[0]):02d}-{int(hours[1]):02d}.json"


def cmip6_name(variable: str, months) -> str:
    return f"{CLIMATE_DIR}/cmip6_{variable}_{'-'.join(str(int(m)) for m in months)}.csv"


# ---------------------------------------------------------------------------
# link edits (pure; raw is the core: block, edited in place)
# ---------------------------------------------------------------------------

def link_forcing(path: str) -> Callable[[dict], None]:
    def edit(raw: dict) -> None:
        set_dotted(raw, "physics.forcing", path)
    return edit


def link_climate(path: str, *, periods: dict | None = None, experiments: list[str] | None = None) \
        -> Callable[[dict], None]:
    """``climate.table`` (+ ``source: table``, ``enabled``), and ``periods``/``experiments`` when they differ
    from what the config would otherwise use."""
    def edit(raw: dict) -> None:
        set_dotted(raw, "climate.table", path)
        set_dotted(raw, "climate.source", "table")
        set_dotted(raw, "climate.enabled", True)
        if periods:
            want = {str(k): [int(v[0]), int(v[1])] for k, v in periods.items()}
            have = get_dotted(raw, "climate.periods") or DEFAULT_PERIODS
            if {str(k): [int(x) for x in v] for k, v in dict(have).items()} != want:
                set_dotted(raw, "climate.periods", want)
        if experiments:
            have = list(get_dotted(raw, "climate.experiments") or DEFAULT_EXPERIMENTS)
            if sorted(have) != sorted(experiments):
                set_dotted(raw, "climate.experiments", list(experiments))
    return edit


def link_layers(path: str) -> Callable[[dict], None]:
    def edit(raw: dict) -> None:
        set_dotted(raw, "planner.layers", path)
    return edit


def _join_entry(path: str, key: str) -> dict:
    return {"path": path, "key": key, "right_key": "id"}


def link_features(path: str) -> Callable[[dict], None]:
    """Join the open features into this project.  A project without predictors (bootstrap mode) takes the six
    ``open_*`` predictors and roles; otherwise the ``open_*`` predictors are added and only unmapped roles
    are pointed at them."""
    def edit(raw: dict) -> None:
        data = raw.setdefault("data", {}) if isinstance(raw.get("data"), dict) else raw.setdefault("data", {})
        key = data.get("id") or "id"
        joins = [j for j in (data.get("join") or []) if not (isinstance(j, dict) and j.get("path") == path)]
        joins.append(_join_entry(path, key))
        data["join"] = joins
        preds = list(raw.get("predictors") or [])
        opens = [f"open_{r}" for r in ROLE_NAMES]
        physics = raw.get("physics") if isinstance(raw.get("physics"), dict) else {}
        roles = dict(physics.get("roles") or {})
        if not preds:
            raw["predictors"] = opens
            roles = {r: f"open_{r}" for r in ROLE_NAMES}
        else:
            raw["predictors"] = preds + [c for c in opens if c not in preds]
            for r in ROLE_NAMES:
                if not roles.get(r):
                    roles[r] = f"open_{r}"
        physics["roles"] = roles
        raw["physics"] = physics
        qa = raw.get("qa") if isinstance(raw.get("qa"), dict) else {}
        clip = dict(qa.get("clip") or {})
        for c, b in OPEN_CLIP.items():
            clip.setdefault(c, list(b))
        qa["clip"] = clip
        raw["qa"] = qa
    return edit


def open_rename_map(raw: dict) -> dict[str, str]:
    roles = ((raw.get("physics") or {}).get("roles") or {}) if isinstance(raw.get("physics"), dict) else {}
    return {str(c): f"open_{r}" for r, c in roles.items() if c and r in ROLE_NAMES}


def open_project_raw(raw: dict, project_dir: str | Path, features_path: str) -> dict:
    """The config of ``<name>_open``: this project's config with every path absolute, the open features joined
    (``features_path`` is the new project's own copy) and every column reference moved from the city's
    layer to the open layer of the same role (references to columns without one are dropped)."""
    out = absolutize_paths(raw, project_dir)
    ren = open_rename_map(raw)
    opens = [f"open_{r}" for r in ROLE_NAMES]
    keep = set(opens)

    def col(c):
        c = ren.get(str(c), str(c))
        return c if c in keep else None

    out["name"] = f"{raw.get('name') or 'core_run'}_open"
    data = out.setdefault("data", {})
    key = data.get("id") or "id"
    data["join"] = list(data.get("join") or []) + [_join_entry(features_path, key)]
    out["predictors"] = opens
    phys = out.get("physics") if isinstance(out.get("physics"), dict) else {}
    phys["roles"] = {r: f"open_{r}" for r in ROLE_NAMES}
    out["physics"] = phys
    qa = out.get("qa") if isinstance(out.get("qa"), dict) else {}
    clip = {col(k): v for k, v in (qa.get("clip") or {}).items() if col(k)}
    for c, b in OPEN_CLIP.items():
        clip.setdefault(c, list(b))
    qa["clip"] = clip
    out["qa"] = qa
    enc = out.get("encodings") if isinstance(out.get("encodings"), dict) else None
    if enc:
        out["encodings"] = {k: [col(c) for c in (v or []) if col(c)] for k, v in enc.items()}
    if isinstance(out.get("actionable"), dict):
        out["actionable"] = {col(k): v for k, v in out["actionable"].items() if col(k)}
    if isinstance(out.get("mediators"), dict):
        meds = {}
        for k, v in out["mediators"].items():
            if not col(k) or not isinstance(v, dict):
                continue
            v = dict(v)
            v["parents"] = [col(c) for c in v.get("parents") or [] if col(c)]
            v["context"] = [col(c) for c in v.get("context") or [] if col(c)]
            v["monotone"] = {col(c): s for c, s in (v.get("monotone") or {}).items() if col(c)}
            meds[col(k)] = v
        out["mediators"] = meds
    if isinstance(out.get("coupling"), list):
        out["coupling"] = [{**c, "sum": [col(x) for x in c.get("sum") or [] if col(x)]} for c in out["coupling"]
                           if isinstance(c, dict) and all(col(x) for x in c.get("sum") or [])]
    if isinstance(out.get("scenarios"), list):
        out["scenarios"] = [{**s, "variable": col(s.get("variable"))} for s in out["scenarios"]
                            if isinstance(s, dict) and col(s.get("variable"))]
    if isinstance(out.get("joint_scenarios"), list):
        js = []
        for j in out["joint_scenarios"]:
            if not isinstance(j, dict):
                continue
            ivs = [{**iv, "variable": col(iv.get("variable"))} for iv in j.get("interventions") or []
                   if isinstance(iv, dict) and col(iv.get("variable"))]
            if ivs:
                js.append({**j, "interventions": ivs})
        out["joint_scenarios"] = js
    c = out.get("causal") if isinstance(out.get("causal"), dict) else None
    if c:
        c["treatments"] = [col(t) for t in c.get("treatments") or [] if col(t)]
        for k in ("confounders", "exclude_controls"):
            if isinstance(c.get(k), dict):
                c[k] = {col(t): [col(x) for x in v or [] if col(x)] for t, v in c[k].items() if col(t)}
        if isinstance(c.get("contrast"), dict):
            c["contrast"] = {col(t): v for t, v in c["contrast"].items() if col(t)}
    o = out.get("optimize") if isinstance(out.get("optimize"), dict) else None
    if o and o.get("variable"):
        o["variable"] = col(o["variable"])
    return out


# ---------------------------------------------------------------------------
# site
# ---------------------------------------------------------------------------

def site_of(raw: dict, project_dir: str | Path, lat: float | None = None, lon: float | None = None) \
        -> tuple[float, float]:
    """The input site: explicit lat/lon, else ``climate.site``, else the data centroid (needs a CRS; reads S0's
    coordinates only).  ``422 needs_crs`` when none is available."""
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    site = get_dotted(raw, "climate.site")
    if isinstance(site, (list, tuple)) and len(site) == 2:
        return float(site[0]), float(site[1])
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    crs = data.get("crs")                      # the CRS of the raw x/y (reproject_to only changes the grid frame)
    if not crs or not data.get("path"):
        raise ApiError("needs_crs", "Set data.crs (or give lat/lon): the site defaults to the data centroid")
    from pyproj import Transformer

    from sparc.studio.projects.files import read_table_head

    p = Path(data["path"])
    p = p if p.is_absolute() else Path(project_dir) / p
    x, y = data.get("x") or "x", data.get("y") or "y"
    df = read_table_head(p, rows=None, columns=[x, y])
    xs = pd.to_numeric(df[x], errors="coerce").to_numpy(float)
    ys = pd.to_numeric(df[y], errors="coerce").to_numpy(float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        raise ApiError("needs_crs", "the data has no finite coordinates to place the site")
    lo, la = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(float(np.mean(xs[ok])),
                                                                            float(np.mean(ys[ok])))
    return float(la), float(lo)


# ---------------------------------------------------------------------------
# state and views
# ---------------------------------------------------------------------------

def _resolve(pdir: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else pdir / q


def _rel(pdir: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(pdir.resolve()).as_posix()
    except ValueError:
        return str(p)


def _latest(d: Path, pattern: str) -> Path | None:
    files = [p for p in d.glob(pattern) if p.is_file() and not p.name.startswith(".")] if d.is_dir() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _pick(pdir: Path, configured: str | None, folder: str, pattern: str) -> tuple[Path | None, bool]:
    if configured:
        p = _resolve(pdir, str(configured))
        if p.is_file():
            return p, True
    return _latest(pdir / folder, pattern), False


def _read_table(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, encoding="utf-8-sig")


def inputs_state(raw: dict, project_dir: str | Path, workspace=None) -> dict:
    """``GET /api/projects/{pid}/inputs``."""
    pdir = Path(project_dir)
    out: dict[str, Any] = {"forcing": None, "climate": None, "layers": None, "features": None, "ghcn": None}
    p, linked = _pick(pdir, get_dotted(raw, "physics.forcing"), FORCING_DIR, "*.json")
    if p is not None:
        blob = read_json(p, {}) or {}
        out["forcing"] = {"path": _rel(pdir, p), "date": str(blob.get("date") or ""),
                          "physics": blob.get("physics") or {}, "checks": list(blob.get("checks") or []),
                          "linked": linked}
    p, linked = _pick(pdir, get_dotted(raw, "climate.table"), CLIMATE_DIR, "*.csv")
    if p is not None:
        try:
            df = _read_table(p)
            out["climate"] = {"path": _rel(pdir, p), "n_models": int(df["model"].nunique()) if "model" in df else 0,
                              "experiments": sorted(map(str, df["experiment"].unique())) if "experiment" in df else [],
                              "periods": sorted(map(str, df["period"].unique())) if "period" in df else [],
                              "linked": linked}
        except Exception:                            # noqa: BLE001 - an unreadable file shows as present
            out["climate"] = {"path": _rel(pdir, p), "n_models": 0, "experiments": [], "periods": [],
                              "linked": linked}
    p, linked = _pick(pdir, get_dotted(raw, "planner.layers"), LAYERS_DIR, "*.parquet")
    if p is not None:
        try:
            import pyarrow.parquet as pq

            n = int(pq.ParquetFile(p).metadata.num_rows)
            people = float(pd.read_parquet(p, columns=["people"])["people"].sum()) if "people" in \
                pq.ParquetFile(p).schema_arrow.names else 0.0
        except Exception:                            # noqa: BLE001
            n, people = 0, 0.0
        out["layers"] = {"path": _rel(pdir, p), "n": n, "people_total": people, "linked": linked}
    joins = [str(j.get("path")) for j in (get_dotted(raw, "data.join") or []) if isinstance(j, dict)
             and j.get("path")]
    joined = [j for j in joins if FEATURES_DIR in j.replace("\\", "/") or Path(j).name.startswith("open_")]
    p, linked = _pick(pdir, joined[0] if joined else None, FEATURES_DIR, "*.parquet")
    if p is not None:
        side = read_json(p.with_suffix(".json"), {}) or {}
        out["features"] = {"path": _rel(pdir, p), "agreement": list(side.get("agreement") or []), "linked": linked}
    rec = read_json(pdir / GHCN_RECORD)
    station = get_dotted(raw, "planner.ghcn_station")
    if isinstance(rec, dict) and rec.get("station"):
        out["ghcn"] = {"station": str(rec["station"]), "years": list(rec.get("years") or [None, None])}
    elif station and workspace is not None and (Path(workspace.cache_dir) / f"ghcn_{station}.csv").is_file():
        out["ghcn"] = {"station": str(station), "years": _ghcn_years(Path(workspace.cache_dir) / f"ghcn_{station}.csv")}
    return out


def _ghcn_years(path: Path) -> list[int | None]:
    try:
        d = pd.read_csv(path, usecols=["DATE"], dtype={"DATE": str})["DATE"].str[:4].astype(int)
        return [int(d.min()), int(d.max())]
    except Exception:                                # noqa: BLE001
        return [None, None]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def input_view(kind: str, raw: dict, project_dir: str | Path) -> dict:
    """``GET /inputs/{kind}/view``: chart-ready models (``unknown_view`` for other kinds)."""
    pdir = Path(project_dir)
    if kind == "forcing":
        p, _ = _pick(pdir, get_dotted(raw, "physics.forcing"), FORCING_DIR, "*.json")
        if p is None:
            raise ApiError("not_found", "no forcing file yet")
        blob = read_json(p, {}) or {}
        era, st = blob.get("era5") or {}, blob.get("station")
        rows = []
        for name, ek, sk in (("air temperature (°F)", "t2m_F", None), ("dewpoint (°C)", "d2m_C", "td_C"),
                             ("relative humidity (%)", "rh", "rh"), ("heat index (°F)", "heat_index_F", "heat_index_F"),
                             ("wind speed (m/s)", "wind_speed", "wind_speed"),
                             ("wind from (°)", "wind_from_deg", "wind_from_deg"),
                             ("wind u (m/s)", "u10", "u"), ("wind v (m/s)", "v10", "v")):
            sv = None
            if st:
                sv = (st.get("t_C") * 9 / 5 + 32) if (ek == "t2m_F" and st.get("t_C") is not None) else \
                    (st.get(sk) if sk else None)
            rows.append({"name": name, "era5": _num(era.get(ek)), "station": _num(sv)})
        return {"era5": era, "station": st, "compare": rows, "checks": list(blob.get("checks") or []),
                "physics": blob.get("physics") or {}, "path": _rel(pdir, p)}
    if kind == "climate":
        p, _ = _pick(pdir, get_dotted(raw, "climate.table"), CLIMATE_DIR, "*.csv")
        if p is None:
            raise ApiError("not_found", "no climate table yet")
        df = _read_table(p)
        need = {"model", "experiment", "period", "delta_K"}
        if not need <= set(df.columns):
            raise ApiError("validation", f"{p.name} lacks {', '.join(sorted(need - set(df.columns)))}",
                           detail={"errors": [{"path": "climate.table", "message": "missing columns",
                                               "code": "climate_table_columns"}]})
        rows = [{"model": str(r.model), "experiment": str(r.experiment), "period": str(r.period),
                 "delta_K": _num(r.delta_K)} for r in df.itertuples(index=False)]
        summ = []
        for (e, per), g in df.groupby(["experiment", "period"], sort=True):
            v = g["delta_K"].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            summ.append({"experiment": str(e), "period": str(per), "median": float(np.median(v)),
                         "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": int(len(v))})
        return {"rows": rows, "summary": summ, "path": _rel(pdir, p)}
    if kind == "layers":
        p, _ = _pick(pdir, get_dotted(raw, "planner.layers"), LAYERS_DIR, "*.parquet")
        if p is None:
            raise ApiError("not_found", "no people/land-cover layers yet")
        df = pd.read_parquet(p)
        totals, cols = {}, []
        for c in df.columns:
            if c in ("id", "x_m", "y_m") or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            v = df[c].to_numpy(float)
            fin = v[np.isfinite(v)]
            agg = "sum" if c.startswith("people") else "mean"
            val = float(fin.sum()) if agg == "sum" else (float(fin.mean()) if len(fin) else None)
            totals[c] = val
            cols.append({"name": c, "agg": agg, "value": val, "min": float(fin.min()) if len(fin) else None,
                         "max": float(fin.max()) if len(fin) else None, "n_null": int((~np.isfinite(v)).sum())})
        return {"totals": totals, "columns": cols, "n": int(len(df)), "path": _rel(pdir, p)}
    if kind == "features":
        joins = [str(j.get("path")) for j in (get_dotted(raw, "data.join") or []) if isinstance(j, dict)]
        joined = [j for j in joins if FEATURES_DIR in j.replace("\\", "/")]
        p, _ = _pick(pdir, joined[0] if joined else None, FEATURES_DIR, "*.parquet")
        if p is None:
            raise ApiError("not_found", "no open features yet")
        side = read_json(p.with_suffix(".json"), {}) or {}
        return {"agreement": list(side.get("agreement") or []), "scatter_bins": _scatter_bins(raw, pdir, p),
                "provenance": side.get("provenance"), "path": _rel(pdir, p)}
    raise ApiError("unknown_view", f"no input view {kind!r} (forcing, climate, layers, features)")


def _scatter_bins(raw: dict, pdir: Path, features: Path, bins: int = 24) -> list[dict]:
    """2-D histograms of each role's city layer against its open layer (joined by id)."""
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    roles = ((raw.get("physics") or {}).get("roles") or {}) if isinstance(raw.get("physics"), dict) else {}
    ident = data.get("id")
    if not data.get("path") or not ident or not roles:
        return []
    dp = _resolve(pdir, str(data["path"]))
    try:
        from sparc.studio.projects.files import read_table_head

        city_cols = [c for r, c in roles.items() if c and not str(c).startswith("open_")]
        have = set(read_table_head(dp, rows=0).columns)
        city_cols = [c for c in city_cols if c in have]
        if not city_cols or ident not in have:
            return []
        city = read_table_head(dp, rows=None, columns=[ident] + city_cols)
        opn = pd.read_parquet(features)
    except Exception:                                # noqa: BLE001
        return []
    m = city.merge(opn, left_on=ident, right_on="id", how="inner")
    out = []
    for role, c in roles.items():
        oc = f"open_{role}"
        if c not in m or oc not in m or c == oc:
            continue
        a, b = m[c].to_numpy(float), m[oc].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 2:
            continue
        h, xe, ye = np.histogram2d(a[ok], b[ok], bins=bins)
        out.append({"role": role, "city_column": c, "open_column": oc, "x_edges": [float(v) for v in xe],
                    "y_edges": [float(v) for v in ye], "counts": h.astype(int).tolist()})
    return out


# ---------------------------------------------------------------------------
# ISD stations
# ---------------------------------------------------------------------------

_ISD_CACHE: dict[tuple, pd.DataFrame] = {}
_ISD_LOCK = threading.Lock()


def _isd_table(path: Path) -> pd.DataFrame:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _ISD_LOCK:
        if key in _ISD_CACHE:
            return _ISD_CACHE[key]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")
    ok = lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0)) & lat.between(-90, 90) & lon.between(-180, 180)
    out = pd.DataFrame({"usaf_wban": (df["USAF"].str.strip() + df["WBAN"].str.strip())[ok],
                        "name": df["STATION NAME"].str.strip()[ok], "lat": lat[ok], "lon": lon[ok],
                        "begin": df["BEGIN"][ok], "end": df["END"][ok]}).reset_index(drop=True)
    with _ISD_LOCK:
        _ISD_CACHE.clear()
        _ISD_CACHE[key] = out
    return out


def _iso_day(v: str) -> str | None:
    v = str(v or "").strip()
    return f"{v[:4]}-{v[4:6]}-{v[6:8]}" if len(v) == 8 and v.isdigit() else None


def nearest_stations(cache_dir: str | Path, lat: float, lon: float, limit: int = 10) -> list[dict]:
    """The ``limit`` ISD stations nearest to (lat, lon) from the cached index; ``FileNotFoundError`` when the
    index is not cached."""
    path = Path(cache_dir) / ISD_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    t = _isd_table(path)
    la1, lo1 = np.radians(float(lat)), np.radians(float(lon))
    la2, lo2 = np.radians(t["lat"].to_numpy(float)), np.radians(t["lon"].to_numpy(float))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    dist = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    order = np.argsort(dist, kind="stable")[:max(1, int(limit))]
    return [{"usaf_wban": str(t["usaf_wban"].iat[i]), "name": str(t["name"].iat[i]), "lat": float(t["lat"].iat[i]),
             "lon": float(t["lon"].iat[i]), "dist_km": round(float(dist[i]), 2), "begin": _iso_day(t["begin"].iat[i]),
             "end": _iso_day(t["end"].iat[i])} for i in order]
