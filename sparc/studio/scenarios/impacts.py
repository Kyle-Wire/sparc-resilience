"""Impacts of a scenario result (SPEC §7.7, api.md §7.5 ``Impacts``); people layers are required.

* **exposure** - ``planner.exposure_table``: residents at or above each threshold and the person-weighted
  temperature today and under the CMIP6 median warming of each future, with and without the scenario;
* **equity** - ``planner.benefit_by_group``: mean cooling by quintile of density and of the shares aged 60+ and
  under 5, with the concentration index;
* **hot days** - ``planner.hot_days`` person-days at or above each threshold today and in each future, with and
  without the scenario, from the GHCN station's cached daily maxima (``<ws>/cache/ghcn_<station>.csv``); when
  the series is not cached, ``hot_days`` is null and ``hot_days_action`` fetches it (an ``input.ghcn`` job) -
  nothing is downloaded inside a request;
* **zones** and **hexes** (250 m and 500 m) - cooling, residents and temperature per zone / hexagon;
* **climate offset** - the share of each future's median warming the city-mean cooling cancels;
* **heat** - NWS heat-index risk with the campaign dewpoint: residents at Extreme caution or worse and at Danger
  or worse before and after the design, today and in each future under both humidity assumptions (null without
  campaign humidity; :func:`sparc.studio.runs.heat.design_heat`).

Results are cached in ``results/<res_id>/impacts_<hash>.json`` keyed by the parameters (the defaults also as
``impacts_default.json``, which ``GET /api/results/{id}`` includes).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from sparc.studio.errors import ApiError

log = logging.getLogger("sparc.studio.scenarios")

__all__ = ["compute_impacts", "impacts_for_result", "default_futures", "hot_days_section", "params_hash"]

FUTURES = (("ssp245", "2041-2060"), ("ssp245", "2081-2100"), ("ssp585", "2041-2060"), ("ssp585", "2081-2100"))


#: bumped when the impacts gain a section, so results cached before it are recomputed (2: heat risk)
IMPACTS_VERSION = 2


def params_hash(thresholds, futures) -> str:
    text = json.dumps({"thresholds": thresholds, "futures": futures, "v": IMPACTS_VERSION}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def default_futures(ctx, cache_dir=None) -> list[dict]:
    from sparc.studio.scenarios.climate import load_factors

    f, _ = load_factors(ctx, cache_dir)
    if f is None:
        return []
    have = {(str(e), str(p)) for e, p in zip(f["experiment"], f["period"])}
    return [{"experiment": e, "period": p} for e, p in FUTURES if (e, p) in have]


def _warming(ctx, futures: list[dict], cache_dir=None) -> list[tuple[str, str, str, float]]:
    """``[(experiment, period, label, median warming in target units)]``."""
    from sparc.core.climate import SSP_LABELS
    from sparc.studio.scenarios.climate import load_factors, to_units

    f, _ = load_factors(ctx, cache_dir)
    if f is None or not futures:
        return []
    k = to_units(ctx)
    out = []
    for fu in futures:
        e, p = str(fu.get("experiment")), str(fu.get("period"))
        g = f[(f["experiment"].astype(str) == e) & (f["period"].astype(str) == p)]
        d = g["delta_K"].to_numpy(float) * k
        d = d[np.isfinite(d)]
        if d.size:
            out.append((e, p, f"{SSP_LABELS.get(e, e)} {p}", float(np.median(d))))
    return out


def hot_days_section(ctx, delta: np.ndarray, people: np.ndarray, thresholds: list[float],
                     warming: list[tuple], cache_dir) -> tuple[dict | None, dict | None]:
    """``(hot_days, hot_days_action)``: person-days at or above each threshold, with and without the scenario."""
    from sparc.core.planner import _station_offsets, ghcn_tmax, hot_days

    obs = np.asarray(ctx.data.target_raw, dtype=np.float64)
    offsets, t_station, ghcn = _station_offsets(ctx.manifest, obs)
    station = (ctx.cfg_raw.get("planner") or {}).get("ghcn_station") or ghcn
    if not station:
        return None, None
    cache = Path(cache_dir) / f"ghcn_{station}.csv" if cache_dir else None
    if cache is None or not cache.exists():
        action = {"kind": "fetch_input", "label": f"Fetch GHCN daily maxima ({station})", "method": "POST",
                  "path": f"/api/projects/{ctx.project_id}/inputs/ghcn", "body": {"station": station}} \
            if ctx.project_id else None
        return None, action
    tmax = ghcn_tmax(station, cache_dir)
    p = np.nan_to_num(people)
    cases = []
    for label, w in [("today", 0.0)] + [(lab, ww) for _e, _p, lab, ww in warming]:
        for adapted in (False, True):
            off = offsets + (delta if adapted else 0.0)
            days = hot_days(tmax, off, thresholds, shift=w)
            cases.append({"case": label, "adapted": adapted, **{f"days_ge_{k}": float(np.mean(v))
                                                               for k, v in days.items()},
                          **{f"person_days_ge_{k}": float(np.sum(p * v)) for k, v in days.items()}})
    avoided = {}
    for c in cases:
        if c["adapted"]:
            base = next(x for x in cases if x["case"] == c["case"] and not x["adapted"])
            avoided[c["case"]] = {f"{t:g}": base[f"person_days_ge_{t:g}"] - c[f"person_days_ge_{t:g}"]
                                  for t in thresholds}
    return {"station": station, "station_campaign_temp": t_station, "baseline": [1995, 2014], "cases": cases,
            "person_days_avoided": avoided}, None


def compute_impacts(ctx, delta, *, thresholds=None, futures=None, cache_dir=None) -> dict:
    """``Impacts`` of a per-cell Δ on the run of ``ctx``; ``422 needs_layers`` without people layers."""
    from sparc.core.planner import benefit_by_group, exposure_table, summarize_hex, zone_table
    from sparc.studio.runs import layers as L
    from sparc.studio.runs.common import clean
    from sparc.studio.runs.grid import lonlat_transformer, transform_xy
    from sparc.studio.scenarios.climate import _thresholds

    lay = L.people_layers(ctx) if ctx.data is not None else None
    if lay is None or "people" not in lay:
        action = {"kind": "fetch_input", "label": "Fetch people and land cover", "method": "POST",
                  "path": f"/api/projects/{ctx.project_id}/inputs/layers", "body": {}} if ctx.project_id else None
        raise ApiError("needs_layers", "impacts need the planner layers (planner.layers: residents and land cover)",
                       action=action)
    delta = np.asarray(delta, dtype=np.float64)
    obs = np.asarray(ctx.data.target_raw, dtype=np.float64)
    people = np.nan_to_num(lay["people"].to_numpy(float))
    thr = _thresholds(ctx, thresholds)
    futs = futures if futures is not None else default_futures(ctx, cache_dir)
    warming = _warming(ctx, futs, cache_dir)
    rows = exposure_table(obs, people, thr, {lab: w for _e, _p, lab, w in warming}, delta)
    exposure = []
    for r in rows:
        exposure.append({"case": r["case"], "adapted": bool(r["adapted"]), "person_mean_temp": r["person_mean_temp"],
                         "people_ge": {f"{t:g}": r[f"people_ge_{t:g}"] for t in thr},
                         "share_people_ge": {f"{t:g}": r[f"share_people_ge_{t:g}"] for t in thr}})
    try:
        equity = benefit_by_group(-delta, lay)
    except Exception as exc:                       # too few residents for quintiles
        log.info("equity skipped: %s", exc)
        equity = {}
    hd, hd_action = None, None
    try:
        hd, hd_action = hot_days_section(ctx, delta, people, thr, warming, cache_dir)
    except Exception as exc:                       # an unreadable cache: no hot days, but the rest stands
        log.warning("hot days skipped: %s", exc)
    vals = {"cooling": -delta, "people": people, "temperature": obs}
    zones = []
    g = ctx.grid
    if g.zones:
        codes = g.zone_codes
        ok = np.array([c is not None for c in codes])
        if ok.any():
            zones = zone_table(np.asarray([str(c) for c in codes[ok]], dtype=object),
                               {k: np.asarray(v)[ok] for k, v in vals.items()})
    hexes = {}
    for size in (250, 500):
        h = summarize_hex(g.x, g.y, vals, float(size))
        if g.crs:
            lon, lat = transform_xy(lonlat_transformer(g.crs), h["cx"].to_numpy() / g.coord_scale,
                                    h["cy"].to_numpy() / g.coord_scale)
            h["lon"], h["lat"] = lon, lat
        hexes[str(size)] = h.to_dict(orient="records")
    offset = [{"experiment": e, "period": p, "label": lab,
               "offset_share": (-float(np.mean(delta)) / w) if w > 0 else None} for e, p, lab, w in warming]
    heat = None
    try:
        from sparc.studio.runs.heat import design_heat

        heat = design_heat(ctx, obs, people, delta, {lab: w for _e, _p, lab, w in warming})
    except Exception as exc:                       # the rest of the impacts stand without it
        log.warning("heat risk skipped: %s", exc)
    return clean({"thresholds": thr, "exposure": exposure, "equity": equity, "hot_days": hd,
                  "hot_days_action": hd_action, "zones": zones, "hexes": hexes, "climate_offset": offset,
                  "heat": heat})


def impacts_for_result(ctx, row: dict, *, thresholds=None, futures=None, cache_dir=None) -> dict:
    """Cached impacts of a stored result (``impacts_<hash>.json``)."""
    from sparc.studio.engine import store
    from sparc.studio.workspace import read_json, write_json_atomic

    rdir = Path(row["dir"])
    key = params_hash(thresholds, futures)
    path = rdir / f"impacts_{key}.json"
    hit = read_json(path)
    if isinstance(hit, dict) and hit.get("hot_days") is not None:
        return hit
    if isinstance(hit, dict) and hit.get("hot_days_action") is None:
        return hit
    delta = store.read_cells(rdir)["delta"].to_numpy(np.float64)
    out = compute_impacts(ctx, delta, thresholds=thresholds, futures=futures, cache_dir=cache_dir)
    try:
        write_json_atomic(path, out)
        if thresholds is None and futures is None:
            write_json_atomic(rdir / "impacts_default.json", out)
    except OSError:
        log.warning("cannot cache impacts in %s", rdir)
    return out
