"""A real campaign's traverse points: read, place on the model grid, split into passes, estimate.

CAPA Heat Watch campaigns publish the raw vehicle traverses next to the interpolated maps (for
Providence, OSF project ``tdsy7``: ``traverses`` archives of CSV/shapefiles for the morning, afternoon
and evening runs).  The designs need, per sample: its grid cell, its time, which vehicle took it and
which continuous pass it belongs to.

* Files are parsed by :func:`sparc.data.collect.capa.parse_traverse_bytes` (CSV, GeoJSON, shapefile
  archives, nested zips), keeping the window (morning / midday / evening) from the local time.
* A vehicle is the value of an id column when the file has one (``car``, ``vehicle``, ``route``,
  ``sensor``, ``trip``, ``device``, ``id``), otherwise the source file.
* Coordinates are projected to the project's frame exactly as the input table's are (``data.crs`` and
  ``coord_unit``, or ``data.reproject_to``) and snapped to the nearest cell; samples farther than
  ``max_offset`` cells from a valid cell are dropped.
* A **pass** is a run of samples of one vehicle with no gap longer than ``gap_s`` seconds and no jump
  farther than ``jump_m``.  Consecutive samples in the same cell are averaged (a 1 Hz logger takes about
  three per 30 m cell).
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ID_COLUMNS = ("car", "car_id", "vehicle", "vehicle_id", "route", "route_id", "sensor", "sensor_id", "trip",
              "trip_id", "device", "device_id", "id")
WINDOWS = ("morning", "midday", "evening")


# --------------------------------------------------------------------------- #
# Reading                                                                      #
# --------------------------------------------------------------------------- #
def _norm(c) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(c).lower())).strip("_")


def _id_column(df: pd.DataFrame) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    for n in ID_COLUMNS:
        if n in cols and 1 < df[cols[n]].nunique() <= max(200, len(df) // 50):
            return cols[n]
    return None


def _parse_frame(df: pd.DataFrame, source: str, timezone: str | None) -> pd.DataFrame:
    from sparc.data.collect.capa import _window_hint_from_name, parse_traverse_table

    idc = _id_column(df)
    groups = [(None, df)] if idc is None else list(df.groupby(idc, sort=False))
    out = []
    for key, part in groups:
        t = parse_traverse_table(part.reset_index(drop=True), timezone=timezone,
                                 window_hint=_window_hint_from_name(source))
        t["source_file"] = source
        t["vehicle_key"] = f"{source}:{key}" if key is not None else source
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def read_traverses(path: str | Path, timezone: str | None = None) -> pd.DataFrame:
    """Every traverse point under ``path`` (a file, an archive, or a folder of them):
    ``time, lat, lon, temp_f, window, source_file, vehicle_key``."""
    from sparc.data.collect.capa import parse_traverse_bytes

    p = Path(path)
    files = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    frames = []
    for f in files:
        low = f.name.lower()
        if f.name.startswith(".") or "__macosx" in str(f).lower():
            continue
        if low.endswith(".csv"):
            frames.append(_parse_frame(pd.read_csv(f), f.name, timezone))
        elif low.endswith((".geojson", ".shp", ".gpkg")):
            import geopandas as gpd

            frames.append(_parse_frame(gpd.read_file(f), f.name, timezone))
        elif low.endswith(".zip"):
            t = parse_traverse_bytes(f.name, f.read_bytes(), timezone=timezone)
            if len(t):
                t["vehicle_key"] = t["source_file"]
                frames.append(t)
    frames = [x for x in frames if len(x)]
    if not frames:
        raise ValueError(f"no traverse points found under {p}")
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# Placing on the grid                                                          #
# --------------------------------------------------------------------------- #
def lonlat_to_grid(cfg, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS-84 degrees → the project's metric frame (as :func:`sparc.core.data.prepare_frame` builds it)."""
    from pyproj import Transformer

    d = cfg.data
    if d.get("reproject_to"):
        tr = Transformer.from_crs("EPSG:4326", d["reproject_to"], always_xy=True)
        x, y = tr.transform(lon, lat)
        return np.asarray(x, float), np.asarray(y, float)
    if not d.get("crs"):
        raise ValueError("placing traverses needs data.crs (the frame of the input table)")
    tr = Transformer.from_crs("EPSG:4326", d["crs"], always_xy=True)
    x, y = tr.transform(lon, lat)
    return np.asarray(x, float) * cfg.coord_scale, np.asarray(y, float) * cfg.coord_scale


def grid_to_lonlat(cfg, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of :func:`lonlat_to_grid`."""
    from pyproj import Transformer

    d = cfg.data
    if d.get("reproject_to"):
        tr = Transformer.from_crs(d["reproject_to"], "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x, y)
    else:
        tr = Transformer.from_crs(d["crs"], "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(np.asarray(x) / cfg.coord_scale, np.asarray(y) / cfg.coord_scale)
    return np.asarray(lon, float), np.asarray(lat, float)


def snap(grid, x: np.ndarray, y: np.ndarray, max_offset: int = 1) -> np.ndarray:
    """Point index of the valid cell nearest each (x, y) within ``max_offset`` cells; −1 when none."""
    idx = np.full(grid.shape, -1, dtype=np.int64)
    idx[grid.iy, grid.ix] = np.arange(grid.iy.size)
    fx, fy = (np.asarray(x) - grid.x0) / grid.dx, (np.asarray(y) - grid.y0) / grid.dy
    ix, iy = np.rint(fx).astype(np.int64), np.rint(fy).astype(np.int64)
    best = np.full(ix.size, -1, dtype=np.int64)
    bestd = np.full(ix.size, np.inf)
    for oy in range(-max_offset, max_offset + 1):
        for ox in range(-max_offset, max_offset + 1):
            jx, jy = ix + ox, iy + oy
            ok = (jx >= 0) & (jy >= 0) & (jx < grid.shape[1]) & (jy < grid.shape[0])
            cell = np.full(ix.size, -1, dtype=np.int64)
            cell[ok] = idx[jy[ok], jx[ok]]
            d = np.hypot(fx - jx, fy - jy)
            take = (cell >= 0) & (d < bestd)
            best[take], bestd[take] = cell[take], d[take]
    return best


# --------------------------------------------------------------------------- #
# Passes                                                                       #
# --------------------------------------------------------------------------- #
def to_campaign(points: pd.DataFrame, cfg, layout, window: str | None = "midday", gap_s: float = 30.0,
                jump_m: float = 200.0, max_offset: int = 1) -> tuple[pd.DataFrame, dict]:
    """The designs' input (``cell, seg, vehicle, order, t_s, temp``) from traverse points, and a QA dict."""
    qa = {"n_points": int(len(points))}
    pts = points
    if window and "window" in pts and pts["window"].notna().any():
        qa["windows"] = {str(k): int(v) for k, v in pts["window"].value_counts().items()}
        pts = pts[pts["window"] == window]
        if not len(pts):
            raise ValueError(f"no traverse points in the {window!r} window (have {qa['windows']})")
    pts = pts.dropna(subset=["time", "lat", "lon", "temp_f"]).copy()
    if not len(pts):
        raise ValueError("no traverse points with a time, a position and a temperature")
    x, y = lonlat_to_grid(cfg, pts["lon"].to_numpy(float), pts["lat"].to_numpy(float))
    pts["x"], pts["y"] = x, y
    pts["cell"] = snap(layout.grid, x, y, max_offset)
    qa["n_window"] = int(len(pts))
    qa["share_on_grid"] = float(np.mean(pts["cell"] >= 0))
    pts = pts[pts["cell"] >= 0]
    if not len(pts):
        raise ValueError("no traverse point falls on the project's grid (wrong city or CRS?)")
    t0 = pts["time"].min()
    pts["t_s"] = (pts["time"] - t0).dt.total_seconds()
    vkeys = {k: i for i, k in enumerate(sorted(pts["vehicle_key"].astype(str).unique()))}
    pts["vehicle"] = pts["vehicle_key"].astype(str).map(vkeys)
    pts = pts.sort_values(["vehicle", "t_s"]).reset_index(drop=True)
    v = pts["vehicle"].to_numpy()
    t = pts["t_s"].to_numpy()
    step = np.r_[np.inf, np.hypot(np.diff(pts["x"].to_numpy()), np.diff(pts["y"].to_numpy()))]
    gap = np.r_[np.inf, np.diff(t)]
    new = (np.r_[True, v[1:] != v[:-1]]) | (gap > gap_s) | (step > jump_m)
    pts["seg"] = np.cumsum(new) - 1
    qa["median_step_m"] = float(np.nanmedian(step[~new])) if (~new).any() else None
    if qa["median_step_m"] is not None and qa["median_step_m"] > jump_m / 2:
        log.warning("traverse steps are long (median %.0f m): are several vehicles mixed in one file?",
                    qa["median_step_m"])
    # one sample per visit to a cell: consecutive samples of a pass in the same cell are averaged
    c = pts["cell"].to_numpy()
    s = pts["seg"].to_numpy()
    run = np.cumsum(np.r_[True, (c[1:] != c[:-1]) | (s[1:] != s[:-1])]) - 1
    pts["run"] = run
    g = pts.groupby("run", sort=True)
    df = pd.DataFrame({"cell": g["cell"].first().to_numpy(), "seg": g["seg"].first().to_numpy(),
                       "vehicle": g["vehicle"].first().to_numpy(), "t_s": g["t_s"].mean().to_numpy(),
                       "temp": g["temp_f"].mean().to_numpy(), "n_raw": g.size().to_numpy()})
    df["order"] = df.groupby("seg").cumcount()
    lens = df.groupby("seg").size()
    qa.update(n_samples=int(len(df)), n_passes=int(lens.size), n_vehicles=int(df["vehicle"].nunique()),
              n_cells=int(df["cell"].nunique()), median_pass_cells=float(lens.median()),
              duration_min=float((df["t_s"].max() - df["t_s"].min()) / 60.0),
              share_of_city=float(df["cell"].nunique() / layout.grid.iy.size))
    return df, qa


def campaign_csv(samples: pd.DataFrame, cfg, layout, start: str = "2020-07-29 15:00:00") -> bytes:
    """A simulated campaign written the way CAPA publishes one (``datetime, lat, lon, T_F, car``): for
    trying the real-data path end to end."""
    g = layout.grid
    cells = samples["cell"].to_numpy()
    lon, lat = grid_to_lonlat(cfg, g.x0 + g.ix[cells] * g.dx, g.y0 + g.iy[cells] * g.dy)
    t = pd.Timestamp(start) + pd.to_timedelta(samples["t_s"].to_numpy(), unit="s")
    out = pd.DataFrame({"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "lat": np.round(lat, 7),
                        "lon": np.round(lon, 7), "T_F": np.round(samples["temp"].to_numpy(), 2),
                        "car": samples["vehicle"].to_numpy() + 1})
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue().encode()


# --------------------------------------------------------------------------- #
# Estimating                                                                   #
# --------------------------------------------------------------------------- #
TRAVERSE_DESIGNS = ("street_100", "street_300", "reach_1000", "updown", "levels")


def estimate(cfg, path: str | Path, window: str | None = "midday", timezone: str | None = None, layout=None,
             L=None, lab: dict | None = None) -> dict:
    """Every traverse design on a real campaign, with the lab's verdict on each when ``lab`` (a
    :func:`~sparc.core.identify.validate.summarize` result for this city) is given."""
    from sparc.core.identify.estimators import run_all
    from sparc.core.identify.layers import build_layers
    from sparc.core.simcheck import load_layout

    layout = layout or load_layout(cfg)
    L = L or build_layers(layout)
    points = read_traverses(path, timezone=timezone)
    df, qa = to_campaign(points, cfg, layout, window=window)
    rows = run_all(df, None, L, layout.grid, TRAVERSE_DESIGNS)
    verdicts = (lab or {}).get("designs", {})
    for r in rows:
        v = verdicts.get(r["estimator"], {}).get("verdict")
        r["lab_status"] = v["status"] if v else "not validated"
        r["lab_reasons"] = v["reasons"] if v else []
    return {"source": str(path), "window": window, "qa": qa, "designs": rows,
            "headline": headline(rows), "wind": list(L.wind) if L.wind else None}


def headline(rows: list[dict]) -> dict:
    """The answer by reach: each street reading with its lab verdict, the wind signature, and which one is
    the headline (the widest reach the lab trusts)."""
    by = {r["estimator"]: r for r in rows}
    out: dict = {}
    for name in ("street_100", "street_300", "reach_1000"):
        r = by.get(name)
        if r:
            out[name] = {"estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"], "status": r.get("lab_status"),
                         "excludes_zero": bool(r["hi"] < 0 or r["lo"] > 0),
                         "city_estimate": r.get("city_estimate"), "city_se": r.get("city_se")}
    trusted = [n for n in ("reach_1000", "street_300", "street_100") if out.get(n, {}).get("status") == "trustworthy"]
    out["main"] = trusted[0] if trusted else ("street_100" if "street_100" in out else None)
    sig = by.get("updown")
    if sig:
        out["signature"] = {"contrast": sig["estimate"], "lo": sig["lo"], "hi": sig["hi"],
                            "flagged": bool(sig["hi"] < 0), "status": sig.get("lab_status")}
    return out


def fmt_ci(r: dict) -> str:
    return f"{r['estimate']:+.2f} °F (95% CI {r['lo']:+.2f} to {r['hi']:+.2f})"


REACH_TEXT = {"street_100": "within 100 m (about a city block)", "street_300": "within 300 m",
              "reach_1000": "within 1 km"}
STATUS_NOTE = {
    "trustworthy": "the lab trusts this design on this layout: it recovers the planted effect in every simulated "
                   "world and stays at zero when there is none",
    "direction only": "the lab finds the sign reliable but the size dependent on how far cooling spreads",
    "conservative": "the lab finds it a lower bound on the cooling",
    "partial": "the lab finds it right in all but one simulated world",
    "not trustworthy": "the lab does not trust this reading on this layout",
    "not validated": "run the identification lab to validate it on this layout",
}


def estimate_markdown(res: dict) -> str:
    qa, h = res["qa"], res["headline"]
    lines = ["# Canopy effect from the traverses", "",
             f"Source: `{res['source']}`, {res['window'] or 'all'} window.  {qa['n_samples']:,} cell visits on "
             f"{qa['n_passes']:,} passes by {qa['n_vehicles']} vehicles over {qa['duration_min']:.0f} min; "
             f"{qa['n_cells']:,} cells ({qa['share_of_city']:.0%} of the grid).", "", "## Answer", ""]
    main = h.get("main")
    for name in ("street_100", "street_300", "reach_1000"):
        r = h.get(name)
        if not r:
            continue
        lead = "**" if name == main else ""
        lines.append(f"* Raising canopy by 10 percentage points {REACH_TEXT[name]} of a street point: "
                     f"{lead}{fmt_ci(r)}{lead} — {STATUS_NOTE.get(r['status'] or 'not validated', r['status'])}."
                     + ("" if r["excludes_zero"] else " The interval includes zero."))
    g = h.get("signature")
    if g:
        lines.append(f"* Wind signature (upwind − downwind canopy, 1 km): "
                     f"{fmt_ci({'estimate': g['contrast'], 'lo': g['lo'], 'hi': g['hi']})}"
                     + (" — upwind canopy matters more, as advection predicts" if g["flagged"] else
                        " — no detectable direction") + f" ({STATUS_NOTE.get(g['status'] or 'not validated', g['status'])}).")
    lines += ["", "City-wide effects of a uniform edit are not identified by one campaign: a kilometre-scale canopy "
              "effect and a kilometre-scale confounder that tracks canopy leave the same traces in the data.", "",
              "## Every design", "", "| design | estimate | 95% CI | lab verdict |", "|---|---|---|---|"]
    for d in res["designs"]:
        lines.append(f"| {d['label']} | {d['estimate']:+.3f} | {d['lo']:+.3f} to {d['hi']:+.3f} | {d['lab_status']} |")
    return "\n".join(lines) + "\n"


__all__ = ["read_traverses", "lonlat_to_grid", "grid_to_lonlat", "snap", "to_campaign", "campaign_csv", "estimate",
           "headline", "estimate_markdown", "WINDOWS"]
