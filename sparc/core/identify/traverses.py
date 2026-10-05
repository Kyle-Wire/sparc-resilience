"""A real campaign's traverse points: read, place on the model grid, split into runs and passes, estimate.

CAPA Heat Watch campaigns publish the raw vehicle traverses next to the interpolated maps (for
Providence, OSF project ``wu9v7``: CSV / shapefile archives of the morning, afternoon and evening
runs).  The designs need, per sample: its grid cell, its time, which vehicle took it, which run of the
day it belongs to and which continuous pass.

* **Reading** walks a file, an archive or a whole downloaded folder (zips are unpacked, nested ones
  too).  Every file is either read or listed as skipped with the reason (rasters, boundary polygons,
  tables without a temperature or a position), so ``python -m sparc.core.identify inspect`` shows what
  was found before anything is estimated.  Columns are recognised by name: time (``datetime``,
  ``timestamp``, ``date`` + ``time``, …), position (``lat``/``lon``, or the geometry of a point layer in
  any coordinate system), temperature in °F (``T_F``, ``temp_f``, ``TempF``, …) or °C (converted).
* A **vehicle** is the value of an id column when the file has one (``car``, ``vehicle``, ``route``,
  ``sensor``, ``trip``, ``device``, ``id``), otherwise the source file.
* A **run** (window) is a cluster of readings with no gap longer than ``run_gap_h`` hours, named by its
  local clock time: morning, midday (afternoon), evening or night.  Heat Watch drives the same routes
  in each run, so the runs are what the kilometre-scale design compares.
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
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ID_COLUMNS = ("car", "car_id", "vehicle", "vehicle_id", "route", "route_id", "sensor", "sensor_id", "trip",
              "trip_id", "device", "device_id", "id")
WINDOWS = ("morning", "midday", "evening", "night")
WINDOW_ALIASES = {"afternoon": "midday", "am": "morning", "af": "midday", "pm": "evening"}
TABLES = (".csv", ".txt", ".tsv")
VECTORS = (".geojson", ".json", ".shp", ".gpkg", ".kml")
_TEMP_F = re.compile(r"^(t|temp|temperature|air_?temp|air_?temperature|ta)_?(deg_?)?(f|fahrenheit|degf)$")
_TEMP_C = re.compile(r"^(t|temp|temperature|air_?temp|air_?temperature|ta)_?(deg_?)?(c|celsius|degc)$")
_COORD_NAMES = {"lat", "latitude", "y", "lon", "long", "longitude", "lng", "x"}


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


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature columns under a name the CAPA parser knows (``temp_f`` / ``temp_c``), and lat/lon
    from a point geometry (reprojected to WGS-84) in place of projected x/y columns."""
    ren = {}
    for c in df.columns:
        n = _norm(c)
        if _TEMP_F.match(n) and "temp_f" not in ren.values():
            ren[c] = "temp_f"
        elif _TEMP_C.match(n) and "temp_c" not in ren.values():
            ren[c] = "temp_c"
    out = df.rename(columns=ren)
    geom = getattr(out, "geometry", None) if "geometry" in out.columns else None
    if geom is not None:
        if not len(geom) or not (geom.geom_type == "Point").mean() > 0.9:
            raise ValueError("not a point layer")
        if getattr(out, "crs", None) is not None:
            geom = geom.to_crs("EPSG:4326")
        keep = [c for c in out.columns if c != "geometry" and _norm(c) not in _COORD_NAMES]
        out = pd.DataFrame({"lat": geom.y.to_numpy(float), "lon": geom.x.to_numpy(float),
                            **{c: out[c].to_numpy() for c in keep}})
    return out


def _parse_frame(df: pd.DataFrame, source: str, timezone: str | None) -> pd.DataFrame:
    from sparc.data.collect.capa import _window_hint_from_name, parse_traverse_table

    df = _standardise(df)
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


def _read_any(f: Path) -> pd.DataFrame:
    low = f.name.lower()
    if low.endswith(TABLES):
        sep = "\t" if low.endswith(".tsv") else None
        return pd.read_csv(f, sep=sep, engine="python" if sep is None else "c")
    import geopandas as gpd

    return gpd.read_file(f)


def _walk(root: Path, tmp: Path, depth: int = 0):
    """Every readable file under ``root`` (zips unpacked into ``tmp``), as (path, display name)."""
    files = sorted(q for q in root.rglob("*") if q.is_file()) if root.is_dir() else [root]
    for f in files:
        name = f.relative_to(root).as_posix() if root.is_dir() else f.name
        if f.name.startswith(".") or "__macosx" in name.lower():
            continue
        if f.suffix.lower() == ".zip" and depth < 4:
            dest = tmp / f"z{depth}_{abs(hash(str(f))) % 10**8}"
            try:
                with zipfile.ZipFile(f) as z:
                    z.extractall(dest)
            except (zipfile.BadZipFile, OSError) as exc:
                yield f, name, f"unreadable zip ({exc})"
                continue
            for sub, sub_name, err in _walk(dest, tmp, depth + 1):
                yield sub, f"{name}/{sub_name}", err
        else:
            yield f, name, None


def scan_traverses(path: str | Path, timezone: str | None = None, utc_to: str | None = None
                   ) -> tuple[pd.DataFrame, list[dict]]:
    """Every traverse point under ``path`` (a file, an archive, or a folder of them) and a report of each
    file: ``{"file", "status": "read"|"skipped", "rows"|"reason", "columns"}``.  ``utc_to``: the
    timestamps are UTC without saying so; convert them to this local time zone."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    frames, report = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f, name, err in _walk(p, Path(tmp)):
            low = f.name.lower()
            if err:
                report.append({"file": name, "status": "skipped", "reason": err})
                continue
            if not low.endswith(TABLES + VECTORS):
                if not low.endswith((".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".xml", ".qmd", ".qix")):
                    report.append({"file": name, "status": "skipped", "reason": f"not a table or vector layer ({f.suffix or 'no extension'})"})
                continue
            try:
                raw = _read_any(f)
                t = _parse_frame(raw, name, timezone)
                if not len(t):
                    raise ValueError("no rows with a position and a temperature")
            except Exception as exc:  # noqa: BLE001 - every file is reported, none stops the scan
                report.append({"file": name, "status": "skipped", "reason": str(exc).splitlines()[0][:200]})
                continue
            report.append({"file": name, "status": "read", "rows": int(len(t)),
                           "columns": [str(c) for c in list(raw.columns)[:30]]})
            frames.append(t)
    if not frames:
        return pd.DataFrame(columns=["time", "lat", "lon", "temp_f", "window", "source_file", "vehicle_key"]), report
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    if utc_to:
        t = out["time"]
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize("UTC")
        out["time"] = t.dt.tz_convert(utc_to).dt.tz_localize(None)
    return assign_runs(out), report


def read_traverses(path: str | Path, timezone: str | None = None, utc_to: str | None = None) -> pd.DataFrame:
    """Every traverse point under ``path``: ``time, lat, lon, temp_f, window, run, source_file,
    vehicle_key``."""
    pts, report = scan_traverses(path, timezone, utc_to)
    if not len(pts):
        skipped = "; ".join(f"{r['file']}: {r.get('reason')}" for r in report[:8])
        raise ValueError(f"no traverse points found under {path}" + (f" (skipped {skipped})" if skipped else ""))
    return pts


def _window_name(hour: float) -> str:
    if 4 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "midday"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def assign_runs(pts: pd.DataFrame, run_gap_h: float = 1.5) -> pd.DataFrame:
    """``run``: clusters of readings separated by more than ``run_gap_h`` hours; ``window``: each run's
    name from its median local hour (readings without a time keep the window their file name gives)."""
    pts = pts.copy()
    t = pts["time"]
    have = t.notna().to_numpy()
    pts["run"] = pd.Series([None] * len(pts), dtype=object)
    if have.any():
        order = np.argsort(t[have].to_numpy())
        ts = t[have].to_numpy()[order]
        gap = np.r_[np.inf, np.diff(ts).astype("timedelta64[s]").astype(float)]
        rid = np.cumsum(gap > run_gap_h * 3600.0) - 1
        runs = np.empty(have.sum(), dtype=object)
        names, windows = {}, np.empty(have.sum(), dtype=object)
        for r in np.unique(rid):
            sel = rid == r
            mid = pd.Timestamp(ts[sel][len(ts[sel]) // 2])
            win = _window_name(mid.hour + mid.minute / 60.0)
            start = pd.Timestamp(ts[sel][0])
            names[r] = f"{start:%Y-%m-%d} {win}"
            windows[sel] = win
            runs[sel] = names[r]
        idx = np.flatnonzero(have)[order]
        pts.loc[pts.index[idx], "run"] = runs
        pts.loc[pts.index[idx], "window"] = windows
    return pts


def runs_summary(pts: pd.DataFrame) -> list[dict]:
    """Per run: start and end (local clock as recorded), readings, vehicles, files."""
    out = []
    for run, part in pts.groupby("run", sort=False):
        out.append({"run": run, "window": str(part["window"].iloc[0]), "start": str(part["time"].min()),
                    "end": str(part["time"].max()), "n": int(len(part)),
                    "vehicles": int(part["vehicle_key"].nunique()), "files": sorted(part["source_file"].unique())[:6],
                    "temp_f": [float(part["temp_f"].quantile(0.05)), float(part["temp_f"].quantile(0.95))]})
    return sorted(out, key=lambda r: r["start"])


def inspect_markdown(pts: pd.DataFrame, report: list[dict], cfg=None, layout=None, show_all: bool = False) -> str:
    """What a campaign folder holds, in plain text: files read and skipped, runs, and grid overlap."""
    read = [r for r in report if r["status"] == "read"]
    skipped = [r for r in report if r["status"] == "skipped"]
    lines = ["# Traverse files", "", f"{len(read)} file(s) read, {len(skipped)} skipped; {len(pts):,} readings.", ""]
    if read:
        lines += ["## Read", "", "| file | readings | columns |", "|---|---|---|"]
        lines += [f"| {r['file']} | {r['rows']:,} | {', '.join(r['columns'][:12])} |" for r in read]
        lines.append("")
    if skipped:
        shown = skipped if show_all else skipped[:25]
        lines += ["## Skipped", "", "| file | why |", "|---|---|"]
        lines += [f"| {r['file']} | {r['reason']} |" for r in shown]
        if len(shown) < len(skipped):
            lines.append(f"| … | {len(skipped) - len(shown)} more (use --all) |")
        lines.append("")
    if len(pts):
        lines += ["## Runs", "", "| run | from | to | readings | vehicles | °F (5–95%) |", "|---|---|---|---|---|---|"]
        for r in runs_summary(pts):
            lines.append(f"| {r['run']} | {r['start'][11:19]} | {r['end'][11:19]} | {r['n']:,} | {r['vehicles']} | "
                         f"{r['temp_f'][0]:.1f}–{r['temp_f'][1]:.1f} |")
        lines.append("")
        hours = pd.to_datetime(pts["time"]).dt.hour
        if hours.notna().any() and ((hours >= 9) & (hours < 12)).mean() > 0.25 and not (hours.between(14, 17)).any():
            lines.append("Note: no readings at 14–17 h but many at 9–12 h: the times may be UTC (add --utc).")
        if cfg is not None and layout is not None:
            x, y = lonlat_to_grid(cfg, pts["lon"].to_numpy(float), pts["lat"].to_numpy(float))
            on = snap(layout.grid, x, y) >= 0
            lines += [f"On the project grid: {on.mean():.0%} of readings "
                      f"(lat {pts['lat'].min():.3f}–{pts['lat'].max():.3f}, lon {pts['lon'].min():.3f}–{pts['lon'].max():.3f})."]
            if on.mean() < 0.2:
                lines.append("Most readings fall outside the grid: is this the right city, or are the positions projected?")
    return "\n".join(lines) + "\n"


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
        want = WINDOW_ALIASES.get(window, window)
        sel = pts["window"] == want
        if "run" in pts:
            sel |= pts["run"] == window
        pts = pts[sel]
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


def run_winds(cfg, points: pd.DataFrame, overrides: dict | None = None, cache_dir=None, fetch: bool = True) -> dict:
    """Wind of every run: ``run → {"from_deg", "speed", "source"}`` (or ``{"error"}``).  ``overrides`` maps a
    run name or a window name (morning / midday / evening / night) to ``(from_deg, speed)``; otherwise the
    station named in the project's forcing file is read for the run's hours (NOAA Global Hourly)."""
    import datetime as dt
    import json as _json

    overrides = {WINDOW_ALIASES.get(k, k): v for k, v in (overrides or {}).items()}
    forcing = None
    fpath = ((cfg.raw.get("physics") or {}).get("forcing"))
    if fpath:
        p = Path(fpath)
        p = p if p.is_absolute() else Path(cfg.base_dir) / p
        if p.is_file():
            forcing = _json.loads(p.read_text(encoding="utf-8"))
    station = ((forcing or {}).get("station") or {}).get("station")
    tz = (forcing or {}).get("tz")
    out = {}
    for run, part in points.groupby("run", sort=False):
        win = str(part["window"].iloc[0])
        if run in overrides or win in overrides:
            d, spd = overrides.get(run, overrides.get(win))
            out[run] = {"from_deg": float(d) % 360.0, "speed": float(spd), "source": "given"}
            continue
        if not (fetch and station and tz):
            out[run] = {"error": "no wind given and no station in the forcing file"}
            continue
        try:
            from zoneinfo import ZoneInfo

            from sparc.core.forcing import station_obs

            t0, t1 = pd.Timestamp(part["time"].min()), pd.Timestamp(part["time"].max())
            if t0.tzinfo is None:
                t0, t1 = t0.tz_localize(ZoneInfo(tz)), t1.tz_localize(ZoneInfo(tz))
            obs = station_obs(station, t0.tz_convert("UTC").to_pydatetime().replace(tzinfo=dt.timezone.utc),
                              t1.tz_convert("UTC").to_pydatetime().replace(tzinfo=dt.timezone.utc), cache_dir=cache_dir)
            out[run] = {"from_deg": float(obs["wind_from_deg"]), "speed": float(obs["wind_speed"]),
                        "source": f"station {station} ({obs['n_reports']} reports)"}
        except Exception as exc:  # noqa: BLE001 - a missing wind only disables the kilometre design
            out[run] = {"error": f"station {station}: {str(exc).splitlines()[0][:160]}"}
    return out


def _attach_verdicts(rows: list[dict], lab: dict | None) -> list[dict]:
    verdicts = (lab or {}).get("designs", {})
    for r in rows:
        d = verdicts.get(r["estimator"], {})
        v = d.get("verdict")
        r["lab_status"] = v["status"] if v else "not validated"
        r["lab_reasons"] = v["reasons"] if v else []
        if "floor" in d:
            r["lab_floor_valid"] = bool(d["floor"]["valid"])
    return rows


def estimate(cfg, path: str | Path, window: str | None = "midday", timezone: str | None = None, layout=None,
             L=None, lab: dict | None = None, winds: dict | None = None, fetch_winds: bool = True,
             cache_dir=None, kilometre: bool = True, utc_to: str | None = None) -> dict:
    """Every traverse design on a real campaign.

    * Each run of the day (morning, midday, evening, night) gets the street designs: canopy's
      block-scale effect at each time of day.
    * The headline is the run named by ``window`` (``"all"`` pools every run).
    * With two or more runs under known winds, the wind-shift design reads advected kilometre-scale
      cooling (:mod:`~sparc.core.identify.windshift`).

    ``lab`` (the identification lab's summary for this city) gives each design its verdict."""
    from sparc.core.identify.estimators import run_all
    from sparc.core.identify.layers import build_layers
    from sparc.core.simcheck import load_layout

    layout = layout or load_layout(cfg)
    L = L or build_layers(layout)
    points, files = scan_traverses(path, timezone, utc_to)
    if not len(points):
        skipped = "; ".join(f"{r['file']}: {r.get('reason')}" for r in files[:8])
        raise ValueError(f"no traverse points found under {path}" + (f" (skipped {skipped})" if skipped else ""))
    designs = ("street_100", "street_300", "reach_1000", "levels")
    runs, frames = [], []
    for k, info in enumerate(runs_summary(points)):
        sub = points[points["run"] == info["run"]]
        try:
            df, qa = to_campaign(sub, cfg, layout, window=None)
        except ValueError as exc:
            runs.append({**info, "error": str(exc)})
            continue
        rows = _attach_verdicts(run_all(df, None, L, layout.grid, designs + (("updown",) if info["window"] == "midday" else ())), lab)
        runs.append({**info, "qa": qa, "designs": rows, "headline": headline(rows)})
        frames.append(df.assign(run=info["run"], vehicle=df["vehicle"] + 1000 * k, seg=df["seg"] + 1_000_000 * k))
    ok = [r for r in runs if "designs" in r]
    if not ok:
        raise ValueError("no run could be placed on the project grid: " + "; ".join(r.get("error", "") for r in runs))
    if window == "all" and len(ok) > 1:
        df, qa = to_campaign(points, cfg, layout, window=None)
        main_rows = _attach_verdicts(run_all(df, None, L, layout.grid, designs), lab)
        main = {"run": "all runs", "window": "all", "qa": qa, "designs": main_rows, "headline": headline(main_rows)}
    else:
        want = WINDOW_ALIASES.get(window or "midday", window or "midday")
        cand = [r for r in ok if r["window"] == want or r["run"] == window]
        main = cand[0] if cand else max(ok, key=lambda r: r["qa"]["n_samples"])
    out = {"source": str(path), "window": main["window"], "run": main["run"], "qa": main["qa"],
           "designs": main["designs"], "headline": main["headline"], "wind": list(L.wind) if L.wind else None,
           "runs": [{k: v for k, v in r.items() if k != "designs"} | ({"street": {n: r["headline"].get(n) for n in
                     ("street_100", "street_300")}} if "headline" in r else {}) for r in runs],
           "files": files}
    if kilometre and len(ok) >= 2:
        out["kilometre"] = kilometre_reading(pd.concat(frames, ignore_index=True), points, cfg, layout, L, winds,
                                             fetch_winds, cache_dir, lab)
    elif kilometre:
        out["kilometre"] = {"status": "needs runs", "reason": "one run only: the kilometre design compares runs under different winds"}
    return out


def kilometre_reading(df: pd.DataFrame, points: pd.DataFrame, cfg, layout, L, winds, fetch_winds, cache_dir, lab) -> dict:
    from sparc.core.identify.windshift import sector_table, wind_shift

    wmap = run_winds(cfg, points, winds, cache_dir=cache_dir, fetch=fetch_winds)
    known = {r: w["from_deg"] for r, w in wmap.items() if "from_deg" in w and r in set(df["run"])}
    res: dict = {"winds": wmap}
    if len(known) < 2:
        res.update(status="needs winds", reason="fewer than two runs with a known wind (give --wind, e.g. "
                   "--wind morning=290:2.0 midday=170:7.7)")
        return res
    try:
        r = wind_shift(df, L, sector_table(layout), known, layout.grid)
    except ValueError as exc:
        res.update(status="failed", reason=str(exc))
        return res
    v = ((lab or {}).get("designs", {}).get("wind_shift") or {}).get("verdict")
    res.update(status="estimated", result=r, lab_status=v["status"] if v else "not validated",
               lab_reasons=v["reasons"] if v else [])
    return res


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
    # the floor on city-wide cooling: the widest near-field reading the lab validates as a floor
    for name in ("street_300", "street_100"):
        r = by.get(name)
        if r and r.get("lab_floor_valid", True):
            fl = r["estimate"] + 1.6448536269514722 * r["se"]
            out["floor"] = {"design": name, "floor": fl, "validated": bool(r.get("lab_floor_valid", False)),
                            "binding": bool(fl < 0)}
            break
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
             f"Source: `{res['source']}`.  Headline run: {res.get('run', res['window'])} — {qa['n_samples']:,} cell "
             f"visits on {qa['n_passes']:,} passes by {qa['n_vehicles']} vehicles over {qa['duration_min']:.0f} min; "
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
    fl = h.get("floor")
    if fl:
        if fl["binding"]:
            lines.append(f"* **City-wide** (+10 pp everywhere): street air cools by **at least {-fl['floor']:.2f} °F** "
                         f"(one-sided 95% floor from the {REACH_TEXT[fl['design']].split(' (')[0]} reading; it assumes canopy "
                         "never warms the air at a distance" + ("; the lab validates it as a floor on this layout)." if
                                                                fl["validated"] else ")."))
        else:
            lines.append("* **City-wide** (+10 pp everywhere): this run sets no floor above zero on the cooling.")
    g = h.get("signature")
    if g:
        lines.append(f"* Wind signature within the run (upwind − downwind canopy, 1 km): "
                     f"{fmt_ci({'estimate': g['contrast'], 'lo': g['lo'], 'hi': g['hi']})}"
                     + (" — upwind canopy matters more, as advection predicts" if g["flagged"] else
                        " — no detectable direction") + f" ({STATUS_NOTE.get(g['status'] or 'not validated', g['status'])}).")
    if len(res.get("runs", [])) > 1:
        lines += ["", "## By time of day", "",
                  "Canopy within 100 m of a street point, +10 pp, in each run of the campaign.", "",
                  "| run | readings | within 100 m | within 300 m |", "|---|---|---|---|"]
        for r in res["runs"]:
            if not r.get("street"):
                lines.append(f"| {r['run']} | {r['n']:,} | {r.get('error', '—')} | |")
                continue
            a, b = r["street"].get("street_100"), r["street"].get("street_300")
            lines.append(f"| {r['run']} | {r['qa']['n_samples']:,} | {fmt_ci(a) if a else '—'} | {fmt_ci(b) if b else '—'} |")
    km = res.get("kilometre")
    if km:
        lines += ["", "## Kilometre scale: the same streets under different winds", ""]
        for run, w in (km.get("winds") or {}).items():
            lines.append(f"* {run}: " + (f"wind from {w['from_deg']:.0f}° at {w['speed']:.1f} m/s ({w['source']})"
                                         if "from_deg" in w else f"no wind ({w['error']})"))
        if km.get("status") == "estimated":
            r = km["result"]
            lines += ["", f"Raising canopy by 10 pp over the kilometre **upwind** of a street rather than the kilometre "
                      f"downwind changes it by **{fmt_ci(r)}** (rotation test: p = {r.get('p_rotation', float('nan')):.2f}; "
                      f"one-sided for cooling p = {r.get('p_cooling', float('nan')):.2f}; {r['n_cells']:,} cells seen in "
                      f"{len(r['runs'])} runs, winds {r['spread_deg']:.0f}° apart at most).",
                      "", "Each street is compared with itself across runs, so nothing fixed about it (its neighbourhood, "
                      "its distance to the bay, how leafy and wealthy its district is) can produce this contrast; what "
                      "changes is which canopy the wind crossed before reaching it.  It measures the part of canopy's "
                      "kilometre-scale effect that travels with the wind. "
                      + STATUS_NOTE.get(km.get("lab_status") or "not validated", "") .capitalize() + "."]
            if r.get("p_cooling", 1.0) >= 0.05:
                lines.append("Not detected in this campaign. The identification lab shows how many runs under varied "
                             "winds it takes to detect advected cooling of a given size.")
        else:
            lines += ["", f"Not estimated: {km.get('reason')}"]
    lines += ["", "Cooling that spreads the same way in every direction over a kilometre is not identified by one "
              "campaign: it and a kilometre-scale factor that tracks canopy leave the same traces. Runs before and "
              "after a canopy change would identify it.", "",
              "## Every design (headline run)", "", "| design | estimate | 95% CI | lab verdict |", "|---|---|---|---|"]
    for d in res["designs"]:
        lines.append(f"| {d['label']} | {d['estimate']:+.3f} | {d['lo']:+.3f} to {d['hi']:+.3f} | {d['lab_status']} |")
    return "\n".join(lines) + "\n"


__all__ = ["scan_traverses", "inspect_markdown", "run_winds", "read_traverses", "lonlat_to_grid", "grid_to_lonlat",
           "snap", "to_campaign", "campaign_csv", "estimate", "headline", "estimate_markdown", "WINDOWS"]
