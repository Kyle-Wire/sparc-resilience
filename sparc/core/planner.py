"""Planner decision pack: people, hot days, plantable space, districts, exports.

Everything here post-processes a finished run (``sparc core planner <run
dir>``) using the open-data layers table (``sparc core layers``):

* **Exposure** — residents (HRSL) by afternoon temperature: person-weighted
  mean, people in cells at or above each threshold, today and in each
  climate future (median CMIP6 warming), with and without the adaptation
  package.
* **Who benefits** — mean cooling of a scenario by quintile of population
  density and of the share of residents aged 60+ (and under 5), and a
  concentration index of cooling benefit over that vulnerability ranking
  (> 0: benefits concentrate where vulnerable residents live).
* **Hot days** — afternoons per summer at or above each threshold in every
  cell: the airport's daily maximum (GHCN, 1995–2014) plus the cell's
  campaign-afternoon offset from the airport, with the offset scaled by 1
  (campaign-like clear, calm afternoons) or 0.5 (a lower bound for ordinary
  days).  Futures add each CMIP6 model's summer-mean change (delta method).
* **Plantable space** — the canopy a cell can still gain: open green and
  bare land (WorldCover) plus a configurable share of paved area (street
  trees, parking), never above 100 − canopy.  S7 can cap allocations by it.
* **Districts and hexagons** — zone (``data.zone``) tables and 250 m / 500 m
  hexagon summaries.
* **Next campaign** — logger sites that would most sharpen the canopy effect
  (canopy contrasts at matched impervious cover, where the effect's
  fold-to-fold spread is largest), and matched control cells for a
  before/after evaluation of the budget plan.
"""

from __future__ import annotations

import io
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GHCN = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{station}.csv"


# --------------------------------------------------------------------------- #
# Plantable space                                                              #
# --------------------------------------------------------------------------- #
def plantable_headroom(canopy: np.ndarray, layers: pd.DataFrame, paved_share: float = 0.2) -> np.ndarray:
    """Canopy (pp) a cell can still gain from open land plus a share of paving."""
    open_land = sum(np.nan_to_num(layers[c].to_numpy(float)) for c in ("lc_grass", "lc_bare", "lc_crop", "lc_shrub")
                    if c in layers)
    built = np.nan_to_num(layers["lc_built"].to_numpy(float)) if "lc_built" in layers else 0.0
    cap = 100.0 * (open_land + paved_share * built)
    return np.clip(np.minimum(cap, 100.0 - np.asarray(canopy, float)), 0.0, None)


# --------------------------------------------------------------------------- #
# Exposure and equity                                                          #
# --------------------------------------------------------------------------- #
def exposure_table(temp: np.ndarray, people: np.ndarray, thresholds, futures: dict[str, float],
                   adaptation: np.ndarray | None = None) -> list[dict]:
    """People at or above each threshold (and person-weighted mean temperature)."""
    p = np.nan_to_num(people)
    rows = []

    def row(label, t, adapted):
        r = {"case": label, "adapted": adapted, "person_mean_temp": float(np.sum(p * t) / max(p.sum(), 1e-9))}
        for th in thresholds:
            r[f"people_ge_{th:g}"] = float(p[t >= th].sum())
            r[f"share_people_ge_{th:g}"] = float(p[t >= th].sum() / max(p.sum(), 1e-9))
        return r

    rows.append(row("today", temp, False))
    if adaptation is not None:
        rows.append(row("today", temp + adaptation, True))
    for label, w in futures.items():
        rows.append(row(label, temp + w, False))
        if adaptation is not None:
            rows.append(row(label, temp + w + adaptation, True))
    return rows


def concentration_index(benefit: np.ndarray, rank_by: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted concentration index of ``benefit`` over the ranking ``rank_by``
    (+1: all benefit at the top of the ranking, 0: even, −1: at the bottom)."""
    w = np.ones_like(benefit) if weights is None else np.nan_to_num(weights)
    order = np.argsort(rank_by, kind="stable")
    b, w = benefit[order], w[order]
    cw = np.cumsum(w)
    r = (cw - 0.5 * w) / cw[-1]                                    # fractional rank
    mu = np.sum(w * b) / np.sum(w)
    if mu == 0:
        return 0.0
    return float(2.0 * np.sum(w * (b - mu) * (r - np.sum(w * r) / np.sum(w))) / (np.sum(w) * mu))


def benefit_by_group(cooling: np.ndarray, layers: pd.DataFrame, n_q: int = 5) -> dict:
    """Mean cooling (positive = cooler) per quintile of density and vulnerability."""
    out = {}
    p = np.nan_to_num(layers["people"].to_numpy(float))
    groups = {"population density": p}
    for c, label in (("people_60_plus", "share aged 60+"), ("people_under_5", "share under 5")):
        if c in layers:
            with np.errstate(invalid="ignore", divide="ignore"):
                groups[label] = np.where(p > 0.5, np.nan_to_num(layers[c].to_numpy(float)) / p, np.nan)
    for label, v in groups.items():
        ok = np.isfinite(v) & (p > 0) if label != "population density" else np.ones_like(v, bool)
        q = pd.qcut(pd.Series(v[ok]).rank(method="first"), n_q, labels=False)
        rows = []
        for k in range(n_q):
            sel = np.flatnonzero(ok)[q.to_numpy() == k]
            rows.append({"quintile": k + 1, "mean_cooling": float(np.mean(cooling[sel])),
                         "people": float(p[sel].sum()), "value_range": [float(np.min(v[sel])), float(np.max(v[sel]))]})
        out[label] = {"quintiles": rows,
                      "concentration_index": concentration_index(cooling[ok], v[ok], weights=p[ok] + 1e-9)}
    return out


# --------------------------------------------------------------------------- #
# Hot days                                                                     #
# --------------------------------------------------------------------------- #
def ghcn_tmax(station: str, cache_dir: str | Path | None = None, fetch=None) -> pd.Series:
    """Daily maximum temperature (°F) of a GHCN-Daily station, indexed by date."""
    from sparc.core.climate import http_fetch

    fetch = fetch or http_fetch
    cache = Path(cache_dir) / f"ghcn_{station}.csv" if cache_dir else None
    raw = cache.read_bytes() if cache is not None and cache.exists() else None
    if raw is None:
        raw = fetch(GHCN.format(station=station))
        if raw is None:
            raise FileNotFoundError(f"GHCN station {station} not found")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw), dtype={"DATE": str, "Q_FLAG": str}, low_memory=False)
    df = df[(df["ELEMENT"] == "TMAX") & (df["Q_FLAG"].isna())]
    s = pd.Series(df["DATA_VALUE"].to_numpy(float) / 10.0 * 9.0 / 5.0 + 32.0,
                  index=pd.to_datetime(df["DATE"], format="%Y%m%d"))
    return s.sort_index()


def hot_days(tmax: pd.Series, offsets: np.ndarray, thresholds, years=(1995, 2014), months=(6, 7, 8),
             shift: float = 0.0, offset_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Mean days per summer with Tmax_station + shift + scale·offset ≥ threshold, per cell."""
    t = tmax[(tmax.index.year >= years[0]) & (tmax.index.year <= years[1]) & tmax.index.month.isin(months)]
    n_years = max(len(set(t.index.year)), 1)
    vals = np.sort(t.to_numpy(float) + shift)
    out = {}
    for th in thresholds:
        need = th - offset_scale * np.asarray(offsets, float)          # station value needed in each cell
        out[f"{th:g}"] = (vals.size - np.searchsorted(vals, need, side="left")) / n_years
    return out


# --------------------------------------------------------------------------- #
# Hexagons and districts                                                       #
# --------------------------------------------------------------------------- #
def hex_ids(x: np.ndarray, y: np.ndarray, size_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointy-top hexagon (axial q, r) of each point; size = centre-to-centre spacing."""
    R = size_m / math.sqrt(3.0)
    q = (math.sqrt(3.0) / 3.0 * x - y / 3.0) / R
    r = (2.0 / 3.0 * y) / R
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    ry = np.where(fix_y, -rx - rz, ry)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    hq, hr = rx.astype(np.int64), rz.astype(np.int64)
    centre_x = R * math.sqrt(3.0) * (hq + hr / 2.0)
    centre_y = R * 1.5 * hr
    key = (hq + 100000) * 1000000 + (hr + 100000)
    return key, centre_x, centre_y


def summarize_hex(x, y, values: dict[str, np.ndarray], size_m: float, sums=("people",)) -> pd.DataFrame:
    key, cx, cy = hex_ids(np.asarray(x, float), np.asarray(y, float), size_m)
    df = pd.DataFrame({"hex": key, "cx": cx, "cy": cy, **{k: np.asarray(v, float) for k, v in values.items()}})
    agg = {"cx": "first", "cy": "first"}
    agg.update({k: ("sum" if k in sums else "mean") for k in values})
    out = df.groupby("hex").agg(agg)
    out["n_cells"] = df.groupby("hex").size()
    return out.reset_index()


def zone_table(zones: np.ndarray, values: dict[str, np.ndarray], sums=("people",)) -> list[dict]:
    rows = []
    for z in sorted(pd.unique(zones)):
        sel = zones == z
        r = {"zone": z.item() if hasattr(z, "item") else z, "n_cells": int(sel.sum())}
        for k, v in values.items():
            v = np.asarray(v, float)[sel]
            r[k] = float(np.nansum(v)) if k in sums else float(np.nanmean(v))
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# Next campaign                                                                #
# --------------------------------------------------------------------------- #
def logger_sites(canopy: np.ndarray, impervious: np.ndarray, effect_sd: np.ndarray, x: np.ndarray, y: np.ndarray,
                 n: int = 30, min_spacing_m: float = 400.0, n_strata: int = 4) -> pd.DataFrame:
    """Sites that sharpen the canopy effect: within impervious strata, pick the
    lowest- and highest-canopy cells (a contrast at matched paving), favouring
    cells where the effect's fold-to-fold spread is large, spaced apart."""
    imp_q = pd.qcut(pd.Series(impervious).rank(method="first"), n_strata, labels=False).to_numpy()
    score = np.nan_to_num(effect_sd) / (np.nanmax(effect_sd) or 1.0)
    picks: list[int] = []
    per = max(n // (2 * n_strata), 1)
    for s in range(n_strata):
        idx = np.flatnonzero(imp_q == s)
        cq = pd.qcut(pd.Series(canopy[idx]).rank(method="first"), 5, labels=False).to_numpy()
        for end in (0, 4):
            cand = idx[cq == end]
            cand = cand[np.argsort(-score[cand])]
            took = 0
            for c in cand:
                if all(math.hypot(x[c] - x[p], y[c] - y[p]) >= min_spacing_m for p in picks):
                    picks.append(int(c))
                    took += 1
                    if took >= per:
                        break
    return pd.DataFrame({"cell": picks, "x_m": x[picks], "y_m": y[picks], "canopy": canopy[picks],
                         "impervious": impervious[picks], "effect_sd": effect_sd[picks],
                         "role": ["low canopy" if canopy[p] < np.median(canopy[picks]) else "high canopy"
                                  for p in picks]})


def matched_controls(treated: np.ndarray, covars: np.ndarray, x: np.ndarray, y: np.ndarray,
                     min_distance_m: float = 1000.0, n_pairs: int = 30, seed: int = 0) -> pd.DataFrame:
    """Before/after design: treated cells (largest planned doses) paired with
    untreated cells of similar covariates at least ``min_distance_m`` from any
    treated cell (nearest standardised-covariate match, without replacement)."""
    from scipy.spatial import cKDTree

    t_idx = np.flatnonzero(treated)
    if t_idx.size == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    tree = cKDTree(np.column_stack([x[t_idx], y[t_idx]]))
    far = tree.query(np.column_stack([x, y]))[0] >= min_distance_m
    pool = np.flatnonzero(far & ~treated)
    Z = (covars - covars.mean(0)) / (covars.std(0) + 1e-12)
    chosen_t = rng.choice(t_idx, min(n_pairs, t_idx.size), replace=False)
    used: set[int] = set()
    rows = []
    for t in chosen_t:
        d = np.linalg.norm(Z[pool] - Z[t], axis=1)
        for j in np.argsort(d):
            c = int(pool[j])
            if c not in used:
                used.add(c)
                rows.append({"treated": int(t), "control": c, "covariate_distance": float(d[j])})
                break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #
def export_geotiffs(data, cfg, layers: dict[str, np.ndarray], out_dir: str | Path) -> list[str]:
    """One float32 GeoTIFF per layer on the study grid, in the data's CRS."""
    import rasterio
    from rasterio.transform import from_origin

    g = data.grid
    d = cfg.data
    crs = d.get("reproject_to") or d.get("crs")
    s = 1.0 if d.get("reproject_to") else cfg.coord_scale
    tr = from_origin((g.x0 - g.dx / 2.0) / s, (g.y0 + (g.ny - 0.5) * g.dy) / s, g.dx / s, g.dy / s)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, v in layers.items():
        r = g.rasterize(np.asarray(v, float))[::-1].astype("float32")
        p = out_dir / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=g.ny, width=g.nx, count=1, dtype="float32", crs=crs,
                           transform=tr, nodata=np.nan, compress="deflate") as dst:
            dst.write(r, 1)
        paths.append(str(p))
    return paths


def export_hex_gpkg(hexdf: pd.DataFrame, size_m: float, cfg, path: str | Path) -> str:
    import geopandas as gpd
    from shapely.geometry import Polygon

    d = cfg.data
    s = 1.0 if d.get("reproject_to") else cfg.coord_scale
    R = size_m / math.sqrt(3.0)
    ang = np.deg2rad(np.arange(6) * 60.0 + 30.0)
    polys = [Polygon(np.column_stack([(cx + R * np.cos(ang)) / s, (cy + R * np.sin(ang)) / s]))
             for cx, cy in zip(hexdf["cx"], hexdf["cy"])]
    gdf = gpd.GeoDataFrame(hexdf.drop(columns=["cx", "cy"]), geometry=polys, crs=d.get("reproject_to") or d.get("crs"))
    gdf.to_file(path, layer=f"hex_{int(size_m)}m", driver="GPKG")
    return str(path)


# --------------------------------------------------------------------------- #
# The pack                                                                     #
# --------------------------------------------------------------------------- #
def _station_offsets(m: dict, target: np.ndarray) -> tuple[np.ndarray, float | None, str | None]:
    """Cell minus station temperature on the campaign afternoon (°F)."""
    ph = (m.get("config") or {}).get("physics") or {}
    fi = ph.get("forcing_info") or {}
    path = fi.get("file")
    t_station = None
    if path:
        cfg_dir = (m.get("provenance") or {}).get("config_dir")
        p = Path(cfg_dir) / path if cfg_dir else Path(path)
        if p.exists():
            st = json.loads(p.read_text(encoding="utf-8")).get("station") or {}
            if st.get("t_C") is not None:
                t_station = st["t_C"] * 9.0 / 5.0 + 32.0
    station = fi.get("station")
    ghcn = f"USW000{str(station)[-5:]}" if station else None
    if t_station is None:
        t_station = float(np.median(target))
    return np.asarray(target, float) - t_station, t_station, ghcn


def planner_pack(run_dir, cfg, out_dir=None, thresholds=None, package: str | None = None,
                 hex_sizes=(250.0, 500.0), export: bool = True, cache_dir="output/core/cache") -> dict:
    from sparc.core.baselines import load_run
    from sparc.core.opendata import load_layers

    run_dir = Path(run_dir)
    out_dir = Path(out_dir or run_dir / "planner")
    out_dir.mkdir(parents=True, exist_ok=True)
    data, folds, m, pred = load_run(run_dir, cfg)
    layers = load_layers(cfg, data)
    if layers is None:
        raise ValueError("planner.layers is not set or missing — run `sparc core layers` first")
    units = data.target_units
    pcfg = cfg.raw.get("planner") or {}
    thresholds = list(thresholds or (cfg.raw.get("climate") or {}).get("thresholds") or [90, 95])
    target = data.target_raw
    people = np.nan_to_num(layers["people"].to_numpy(float))
    out: dict = {"units": units, "thresholds": thresholds, "n_cells": int(data.n), "people_total": float(people.sum())}

    # adaptation package and futures
    deltas = pd.read_parquet(run_dir / "scenario_deltas.parquet") if (run_dir / "scenario_deltas.parquet").exists() else None
    clim = json.loads((run_dir / "climate.json").read_text(encoding="utf-8")) if (run_dir / "climate.json").exists() else {}
    joint = [j["name"] for j in (cfg.raw.get("joint_scenarios") or [])]
    pkg = package or (joint[0] if joint else (clim.get("adaptation") or [None])[0])
    adapt = deltas[pkg].to_numpy(float) if deltas is not None and pkg in deltas else None
    futures = {}                                   # warming is stored in target units
    by_model = {}
    for p in clim.get("projections") or []:
        if p["period"] in ("2041-2060", "2081-2100") and p["experiment"] in ("ssp245", "ssp585"):
            label = f"{p.get('label')} {p['period']}"
            futures[label] = float(p["warming"]["median"])
            by_model[label] = p["warming"].get("by_model") or {}
    out["package"] = pkg
    out["exposure"] = exposure_table(target, people, thresholds, futures, adapt)

    # who benefits
    if adapt is not None:
        out["equity"] = benefit_by_group(-adapt, layers)

    # hot days
    offsets, t_station, ghcn = _station_offsets(m, target)
    station = pcfg.get("ghcn_station") or ghcn
    if station:
        try:
            tmax = ghcn_tmax(station, cache_dir)
            hd = {"station": station, "station_campaign_temp": t_station, "baseline": [1995, 2014], "cases": []}
            maps = {}
            for scale, tag in ((1.0, "campaign-like"), (0.5, "lower bound")):
                today = hot_days(tmax, offsets, thresholds, offset_scale=scale)
                case = {"case": "today", "offset_scale": scale, "label": tag,
                        **{f"days_ge_{k}": float(np.mean(v)) for k, v in today.items()},
                        **{f"person_days_ge_{k}": float(np.sum(people * v)) for k, v in today.items()}}
                hd["cases"].append(case)
                if scale == 1.0:
                    maps.update({f"hot_days_ge_{k}_today": v for k, v in today.items()})
                for label, models in by_model.items():
                    per_model = [hot_days(tmax, offsets, thresholds, shift=w, offset_scale=scale)
                                 for w in models.values()]
                    med = {k: np.median([pm[k] for pm in per_model], axis=0) for k in today}
                    case = {"case": label, "offset_scale": scale, "label": tag,
                            **{f"days_ge_{k}": float(np.mean(v)) for k, v in med.items()},
                            **{f"person_days_ge_{k}": float(np.sum(people * v)) for k, v in med.items()}}
                    if adapt is not None:
                        pa = [hot_days(tmax, offsets + adapt / max(scale, 1e-9), thresholds,
                                       shift=w, offset_scale=scale) for w in models.values()]
                        meda = {k: np.median([x[k] for x in pa], axis=0) for k in today}
                        case.update({f"days_ge_{k}_adapted": float(np.mean(v)) for k, v in meda.items()})
                        case.update({f"person_days_ge_{k}_adapted": float(np.sum(people * v)) for k, v in meda.items()})
                    hd["cases"].append(case)
                    if scale == 1.0 and label.endswith("2041-2060") and "SSP2-4.5" in label:
                        maps.update({f"hot_days_ge_{k}_ssp245_mid": v for k, v in med.items()})
            out["hot_days"] = hd
        except Exception as exc:                # noqa: BLE001 - optional component
            log.warning("hot days skipped: %s", exc)
            maps = {}
    else:
        maps = {}

    # plantable space
    roles = (cfg.raw.get("physics") or {}).get("roles") or {}
    can, imp = roles.get("canopy"), roles.get("impervious")
    head = plantable_headroom(data.frame[can].to_numpy(float), layers, float(pcfg.get("paved_plantable_share", 0.2)))
    out["plantable"] = {"paved_share": float(pcfg.get("paved_plantable_share", 0.2)),
                        "mean_headroom_pp": float(head.mean()), "total_pp_cells": float(head.sum()),
                        "share_cells_no_room": float(np.mean(head < 1.0))}

    # districts and hexagons
    vals = {"temperature": target, "people": people, "plantable_pp": head}
    if adapt is not None:
        vals["package_cooling"] = -adapt
    vals.update({k: v for k, v in maps.items() if k.endswith("_today")})
    if data.zones is not None:
        out["zones"] = zone_table(data.zones, vals)
    hexes = {}
    for size in hex_sizes:
        h = summarize_hex(data.x, data.y_coord, vals, size)
        hexes[size] = h
        h.to_csv(out_dir / f"hex_{int(size)}m.csv", index=False)
    out["hex_files"] = [f"hex_{int(s)}m.csv" for s in hex_sizes]

    # next campaign
    if (run_dir / f"response_{can}.parquet").exists():
        r = pd.read_parquet(run_dir / f"response_{can}.parquet")
        sd_col = "footprint_effect_sd" if "footprint_effect_sd" in r else None
        eff_sd = r[sd_col].to_numpy(float) if sd_col else np.abs(r["footprint_effect_per_unit"].to_numpy(float))
        sites = logger_sites(data.frame[can].to_numpy(float), data.frame[imp].to_numpy(float), eff_sd,
                             data.x, data.y_coord)
        sites.to_csv(out_dir / "logger_sites.csv", index=False)
        out["logger_sites"] = {"n": int(len(sites)), "file": "logger_sites.csv"}
    if (run_dir / "allocation.parquet").exists():
        a = pd.read_parquet(run_dir / "allocation.parquet")
        dose = a["dose"].to_numpy(float)
        top = dose >= np.quantile(dose[dose > 0], 0.5) if (dose > 0).any() else dose > 0
        cov = data.frame[[c for c in (can, imp, roles.get("elevation"), roles.get("water_distance")) if c]].to_numpy(float)
        pairs = matched_controls(top, cov, data.x, data.y_coord)
        pairs.to_csv(out_dir / "before_after_pairs.csv", index=False)
        out["before_after"] = {"n_pairs": int(len(pairs)), "file": "before_after_pairs.csv",
                               "median_covariate_distance": float(pairs["covariate_distance"].median())
                               if len(pairs) else None}

    if export:
        lay = {"temperature_observed": target, "temperature_model": pred["pred"].to_numpy(float),
               "people": people, "plantable_canopy_pp": head, **maps}
        if adapt is not None:
            lay["package_cooling"] = -adapt
        out["geotiffs"] = [Path(p).name for p in export_geotiffs(data, cfg, lay, out_dir / "geotiff")]
        try:
            for size, h in hexes.items():
                export_hex_gpkg(h, size, cfg, out_dir / "hexagons.gpkg")
            out["geopackage"] = "hexagons.gpkg"
        except Exception as exc:                # noqa: BLE001 - geopandas optional
            log.warning("GeoPackage export skipped: %s", exc)
    pd.DataFrame({"id": data.ids, "people": people, "plantable_canopy_pp": head,
                  **{k: v for k, v in maps.items()}}).to_parquet(out_dir / "planner_cells.parquet", index=False)
    (out_dir / "planner.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    m["planner"] = out
    (run_dir / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return out
