"""Build a core config for a CAPA Heat Watch city from its pilot entry and its OSF files (Phase 2 of
docs/agent/ROADMAP.md).

    python scripts/setup_core_city.py albuquerque_nm

Steps, each skipped when its output already exists (``--force`` redoes them):

1. **traverses**: ``capa.fetch_capa_raw`` into ``output/core/<slug>/identify/raw``.  The afternoon run's
   date and hours, read from the traverse timestamps, date the campaign (not the pilot file's guess).
2. **grid**: a 30 m UTM fishnet (``scripts/collect_cities.build_fishnet``) over the pilot bbox, clipped to
   the traverses' extent plus ``--margin-km``, sampled from CAPA's rasters
   (``capa._sample_rasters_to_fishnet``) → ``<slug>_grid.csv`` with ``id, x, y, aat_morning, aat_midday,
   aat_night`` and a constant ``placeholder`` predictor, which only lets the open-data step load the grid.
3. **forcing**: ``sparc core forcing`` for the afternoon window, at the nearest NOAA Global Hourly station
   with a WBAN number (``isd-history.csv``) → ``configs/forcing/<slug>_<date>.json``.
4. **features**: ``sparc core features`` (WorldCover, Sentinel-2, Copernicus DEM) → ``<slug>_open_features.parquet``.
5. **config**: ``configs/core_<slug>.yml`` in the ``core_providence_open.yml`` pattern (open roles, target
   ``aat_midday``, ``output.dir: ../output/core/<slug>``).
6. **layers**: ``sparc core layers`` (HRSL residents, WorldCover) for the planner.  Optional: a failure is
   logged and the config still works for ``python -m sparc.core.identify``.

Every step logs its outcome to ``output/core/<slug>/setup.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("setup_core_city")
ISD_HISTORY = "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"


# --------------------------------------------------------------------------- #
# Pieces (pure, tested in tests/core/test_setup_core_city.py)                  #
# --------------------------------------------------------------------------- #
def pilot_entry(slug: str, pilot_path: Path) -> dict:
    """The city's entry in ``configs/multicity_pilot.yml``."""
    def walk(o):
        if isinstance(o, dict):
            if o.get("city_slug") == slug:
                yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    found = next(walk(yaml.safe_load(Path(pilot_path).read_text(encoding="utf-8"))), None)
    if found is None:
        raise SystemExit(f"{slug!r} is not a city_slug in {pilot_path}")
    return found


def campaign_from_traverses(pts: pd.DataFrame) -> dict:
    """Date, local hours and centre of the afternoon run (the run CAPA's afternoon map is made from)."""
    if not pts["time"].notna().any():
        raise ValueError("the traverse files carry no timestamps (runs and passes cannot be formed)")
    mid = pts[pts["window"] == "midday"].dropna(subset=["time"])
    if not len(mid):
        raise ValueError("no afternoon run among the traverses")
    t0, t1 = mid["time"].min(), mid["time"].max()
    return {"date": f"{t0:%Y-%m-%d}", "hours": [int(t0.hour), int(t1.hour) + (1 if t1.minute or t1.second else 0)],
            "lat": float(mid["lat"].median()), "lon": float(mid["lon"].median()),
            "runs": sorted(str(r) for r in pts["run"].dropna().unique())}


def study_bbox(pilot_bbox, pts: pd.DataFrame, margin_km: float = 1.0) -> list[float]:
    """The pilot bbox clipped to the traverses' extent plus ``margin_km``: cells far from every street the
    vehicles drove add nothing to the identification and slow every step."""
    w, s, e, n = (float(v) for v in pilot_bbox)
    lat0 = float(pts["lat"].median())
    dy = margin_km / 111.32
    dx = margin_km / (111.32 * math.cos(math.radians(lat0)))
    out = [max(w, pts["lon"].min() - dx), max(s, pts["lat"].min() - dy),
           min(e, pts["lon"].max() + dx), min(n, pts["lat"].max() + dy)]
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError(f"the traverses do not overlap the pilot bbox {pilot_bbox}")
    return [round(v, 5) for v in out]


def nearest_isd_station(history: pd.DataFrame, lat: float, lon: float, date: str) -> dict:
    """Nearest NOAA Global Hourly station with a WBAN number, active on ``date`` (``isd-history.csv``)."""
    h = history.copy()
    h.columns = [c.strip().upper() for c in h.columns]
    day = int(date.replace("-", ""))
    ok = ((h["WBAN"].astype(str) != "99999") & (h["USAF"].astype(str) != "999999")
          & (pd.to_numeric(h["BEGIN"], errors="coerce") <= day) & (pd.to_numeric(h["END"], errors="coerce") >= day))
    h = h[ok].copy()
    la, lo = pd.to_numeric(h["LAT"], errors="coerce"), pd.to_numeric(h["LON"], errors="coerce")
    d = np.hypot(la - lat, (lo - lon) * math.cos(math.radians(lat))) * 111.32
    if not d.notna().any():
        raise ValueError("no active WBAN station in isd-history")
    r = h.loc[d.idxmin()]
    wban = str(r["WBAN"]).zfill(5)
    return {"station": f"{str(r['USAF']).zfill(6)}{wban}", "name": str(r["STATION NAME"]), "km": float(d.min()),
            "ghcn": f"USW000{wban}", "icao": str(r.get("ICAO", "") or "")}


def grid_table(fishnet) -> pd.DataFrame:
    """``id, x, y`` (UTM cell centres) and the CAPA raster columns; cells off CAPA's afternoon map dropped."""
    c = fishnet.geometry.centroid
    df = pd.DataFrame({"x": np.round(c.x.to_numpy(float), 2), "y": np.round(c.y.to_numpy(float), 2)})
    for col in ("aat_morning", "aat_midday", "aat_night"):
        if col in fishnet:
            df[col] = pd.to_numeric(fishnet[col], errors="coerce").to_numpy(float)
    df = df[np.isfinite(df["aat_midday"])].reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["placeholder"] = 0.0
    return df


OPEN = ("open_canopy", "open_impervious", "open_ndvi", "open_albedo", "open_elevation", "open_water_distance")


def core_config(slug: str, place: str, utm_epsg: str, camp: dict, forcing_rel: str | None,
                station: dict | None) -> dict:
    """The core YAML (as a dict) in the ``core_providence_open.yml`` pattern."""
    c, i, a = "open_canopy", "open_impervious", "open_albedo"
    conf = {c: [i, a, "open_elevation", "open_water_distance"], i: [c, a, "open_elevation", "open_water_distance"],
            a: [c, i, "open_elevation", "open_water_distance"]}
    pp = {"min": 0, "max": 100, "unit": "pp", "doses": [0, 5, 10, 15, 20, 30, 40, 50], "cost_per_unit": 1.0}
    core = {
        "name": f"{slug}_uhi",
        "data": {"path": f"../output/core/{slug}/{slug}_grid.csv", "target": "aat_midday", "target_units": "degF",
                 "id": "id", "x": "x", "y": "y", "crs": utm_epsg, "coord_unit": "m", "background": "median",
                 "join": [{"path": f"../output/core/{slug}/{slug}_open_features.parquet", "key": "id",
                           "right_key": "id"}]},
        "predictors": list(OPEN),
        "qa": {"clip": {a: [0.02, 0.9], c: [0, 100], i: [0, 100]}},
        "actionable": {c: dict(pp), i: {**pp, "direction": "decrease"},
                       a: {"min": 0.02, "max": 0.9, "unit": "reflectance", "doses": [0, 0.02, 0.05, 0.10, 0.15],
                           "cost_per_unit": 100.0}},
        "coupling": [],
        "mediators": {"open_ndvi": {"parents": [c, i], "context": ["open_elevation", "open_water_distance"],
                                    "monotone": {c: 1, i: -1}}},
        "physics": {"enabled": True, **({"forcing": forcing_rel} if forcing_rel else {}),
                    "roles": {"albedo": a, "canopy": c, "impervious": i, "ndvi": "open_ndvi",
                              "elevation": "open_elevation", "water_distance": "open_water_distance"}},
        "influence": {"max_lag_m": 2000, "n_rings": 10, "n_perm": 19},
        "cv": {"n_folds": 5, "block_m": "auto", "buffer_m": "auto", "seed": 42},
        "models": {"ols": True, "mgwr": True, "gwrf": True, "gam": True, "physics": True},
        "stacker": {"epochs": 400, "tune_lambda": [0.0, 0.1, 1.0]},
        "scenarios": [{"name": "Canopy Increase", "variable": c, "direction": "increase", "increments": [5, 10, 20, 30]},
                      {"name": "Impervious Decrease", "variable": i, "direction": "decrease",
                       "increments": [5, 10, 20, 30]},
                      {"name": "Albedo Increase", "variable": a, "direction": "increase", "increments": [0.02, 0.05, 0.10]}],
        "causal": {"enabled": True, "treatments": [c, i, a], "confounders": conf,
                   "exclude_controls": {c: ["open_ndvi"], i: ["open_ndvi"]},
                   "contrast": {c: 10, i: -10, a: 0.05}, "dag_audit": False},
        "climate": {"enabled": False, "site": [round(camp["lat"], 4), round(camp["lon"], 4)], "thresholds": [90, 95]},
        "optimize": {"enabled": False, "variable": c, "budget": 20000, "cost_per_unit": 1.0, "equity_column": None},
        "planner": {"layers": f"../output/core/{slug}/{slug}_layers.parquet", "paved_plantable_share": 0.2,
                    **({"ghcn_station": station["ghcn"]} if station else {})},
        "output": {"dir": f"../output/core/{slug}"},
        "report": {"title": f"{place} Heat Model", "place": place, "area": f"the {place} Heat Watch area",
                   "limitations": [f"The target is CAPA's area-wide afternoon temperature surface from the {camp['date']} "
                                   "campaign, extended to every cell by a model on land-cover predictors; canopy "
                                   "effects come from the raw traverses (python -m sparc.core.identify), not this map."]},
    }
    return core


def config_text(core: dict, slug: str, camp: dict, bbox: list[float]) -> str:
    head = (f"# SPARC core config for {core['report']['place']}: written by scripts/setup_core_city.py {slug}\n"
            f"# Campaign {camp['date']}, afternoon run {camp['hours'][0]}-{camp['hours'][1]} h local (from the traverse "
            f"timestamps); grid: 30 m UTM over {bbox}.\n"
            "# Data and predictors are regenerated by the script into output/ (not committed).\n")
    return head + yaml.safe_dump({"core": core}, sort_keys=False, allow_unicode=True, width=120)


# --------------------------------------------------------------------------- #
# Steps (network)                                                              #
# --------------------------------------------------------------------------- #
def _build_fishnet():
    spec = importlib.util.spec_from_file_location("collect_cities", REPO / "scripts" / "collect_cities.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_fishnet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", help="city_slug in configs/multicity_pilot.yml, e.g. albuquerque_nm")
    ap.add_argument("--pilot", default=str(REPO / "configs" / "multicity_pilot.yml"))
    ap.add_argument("--margin-km", type=float, default=1.0, help="grid margin around the traverses (default 1 km)")
    ap.add_argument("--cell-m", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    ap.add_argument("--skip-layers", action="store_true", help="skip the planner layers (HRSL)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from sparc.data.collect import capa
    from sparc.data.collect.capa_catalog import CAPA_CATALOG
    from sparc.core.identify.traverses import scan_traverses

    slug = args.slug
    entry = pilot_entry(slug, Path(args.pilot))
    cat = CAPA_CATALOG.get(entry["city"]) or {}
    node = cat.get("traverse_node") or cat.get("osf_node") or entry.get("capa_osf_node")
    hint = None if cat.get("traverse_node") else cat.get("folder_hint")      # rasters and traverses: one campaign
    if not node:
        raise SystemExit(f"no OSF node for {entry['city']} in capa_catalog")
    utm = entry.get("utm_epsg") or cat.get("utm_epsg")
    out = REPO / "output" / "core" / slug
    raw = out / "identify" / "raw"
    status: dict = {"slug": slug, "city": entry["city"], "osf_node": node, "steps": {}}

    def step(name, fn):
        t = time.time()
        try:
            res = fn()
            status["steps"][name] = {"status": "ok", "seconds": round(time.time() - t, 1), **(res or {})}
        except Exception as exc:  # noqa: BLE001 - every step's outcome is recorded
            status["steps"][name] = {"status": "failed", "seconds": round(time.time() - t, 1),
                                     "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"}
            log.error("%s failed: %s", name, exc)
        out.mkdir(parents=True, exist_ok=True)
        (out / "setup.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
        return status["steps"][name]["status"] == "ok"

    state: dict = {}

    def traverses():
        files = capa.fetch_capa_raw(node, raw, folder_hint=hint)
        pts, _ = scan_traverses(raw)
        state["pts"] = pts
        state["camp"] = campaign_from_traverses(pts)
        state["bbox"] = study_bbox(entry["bbox"], pts, args.margin_km)
        return {"files": [p.name for p in files], "readings": int(len(pts)), **state["camp"], "bbox": state["bbox"]}

    if not step("traverses", traverses):
        print(json.dumps(status, indent=1, default=str))
        return 1
    camp, bbox = state["camp"], state["bbox"]
    grid_csv = out / f"{slug}_grid.csv"

    def grid():
        if grid_csv.is_file() and not args.force:
            return {"cells": int(sum(1 for _ in grid_csv.open()) - 1), "reused": True}
        net = _build_fishnet()(tuple(bbox), utm, args.cell_m)
        src = capa._find_raster_source(node, folder_hint=hint)
        sampled = capa._sample_rasters_to_fishnet(net, capa._load_raster_data(src))
        df = grid_table(sampled)
        df.to_csv(grid_csv, index=False)
        return {"cells": int(len(df)), "fishnet_cells": int(len(net)), "raster_source": src.get("name")}

    if not step("grid", grid):
        print(json.dumps(status, indent=1, default=str))
        return 1

    forcing_rel = f"forcing/{slug}_{camp['date']}.json"
    forcing_path = REPO / "configs" / forcing_rel

    def forcing():
        import io

        from sparc.core.forcing import campaign_forcing, http_fetch

        hist = pd.read_csv(io.BytesIO(http_fetch(ISD_HISTORY)), dtype=str)
        st = nearest_isd_station(hist, camp["lat"], camp["lon"], camp["date"])
        state["station"] = st
        if forcing_path.is_file() and not args.force:
            return {"station": st, "reused": True}
        res = campaign_forcing(camp["lat"], camp["lon"], camp["date"], tuple(camp["hours"]), entry["timezone"],
                               station=st["station"], cache_dir=str(REPO / "output" / "core" / "cache"))
        forcing_path.parent.mkdir(parents=True, exist_ok=True)
        forcing_path.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
        return {"station": st, "wind": res["physics"].get("wind")}

    have_forcing = step("forcing", forcing)
    feats = out / f"{slug}_open_features.parquet"

    def features():
        if feats.is_file() and not args.force:
            return {"reused": True}
        from sparc.core.config import core_config_from_dict
        from sparc.core.data import load_core_data
        from sparc.core.features_open import build_open_features

        boot = core_config_from_dict({"name": f"{slug}_setup", "data": {
            "path": str(grid_csv), "target": "aat_midday", "id": "id", "x": "x", "y": "y", "crs": utm,
            "coord_unit": "m"}, "predictors": ["placeholder"], "physics": {"enabled": False},
            "output": {"dir": str(out)}}, base_dir=REPO)
        y, m = int(camp["date"][:4]), int(camp["date"][5:7])
        df, prov = build_open_features(load_core_data(boot), boot, months=[(y, m)])
        df.rename(columns={c: f"open_{c}" for c in df.columns if c != "id"}).to_parquet(feats, index=False)
        return {"provenance": prov}

    have_features = step("features", features)
    cfg_path = REPO / "configs" / f"core_{slug}.yml"

    def config():
        core = core_config(slug, entry["city"], utm, camp, forcing_rel if have_forcing else None, state.get("station"))
        cfg_path.write_text(config_text(core, slug, camp, bbox), encoding="utf-8")
        if have_features:
            from sparc.core.config import load_core_config
            from sparc.core.data import load_core_data

            d = load_core_data(load_core_config(cfg_path))
            return {"config": str(cfg_path.relative_to(REPO)), "cells_loaded": int(len(d.ids))}
        return {"config": str(cfg_path.relative_to(REPO)), "note": "features missing: the config will not load yet"}

    ok_cfg = step("config", config)
    if not args.skip_layers and ok_cfg and have_features:
        def layers():
            lay_path = out / f"{slug}_layers.parquet"
            if lay_path.is_file() and not args.force:
                return {"reused": True}
            from sparc.core.config import load_core_config
            from sparc.core.data import load_core_data
            from sparc.core.opendata import fetch_layers

            cfg = load_core_config(cfg_path)
            lay = fetch_layers(load_core_data(cfg), cfg)
            lay.to_parquet(lay_path, index=False)
            return {"residents": float(lay["people"].sum())}

        step("layers", layers)
    print(json.dumps(status, indent=1, default=str))
    return 0 if ok_cfg and have_features else 1


if __name__ == "__main__":
    sys.exit(main())
