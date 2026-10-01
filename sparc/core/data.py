"""S0 — data loading, QA and encoding for the core pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sparc.core.config import CoreConfig
from sparc.core.grid import Grid, spatial_window_subsample


@dataclass
class CoreData:
    """Everything downstream stages need about the study area."""

    frame: pd.DataFrame            # raw predictor columns (QA-clipped), one row per point
    X: pd.DataFrame                # encoded model features (raw units)
    y: np.ndarray                  # ΔT target (target units, background removed)
    target_raw: np.ndarray         # original target values
    background: float
    x: np.ndarray                  # metres, grid frame
    y_coord: np.ndarray            # metres, grid frame
    grid: Grid
    ids: np.ndarray
    qa: dict = field(default_factory=dict)
    target_units: str = "degF"
    zones: np.ndarray | None = None   # optional zone / neighbourhood code per point (reporting only)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def coords(self) -> np.ndarray:
        return np.column_stack([self.x, self.y_coord])

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def raster(self, column: str) -> np.ndarray:
        return self.grid.rasterize(self.frame[column].to_numpy(dtype=float))

    def with_frame(self, frame: pd.DataFrame, encodings: dict) -> "CoreData":
        """Copy with modified raw predictors (used by scenarios); target unchanged."""
        return CoreData(
            frame=frame,
            X=encode_features(frame, encodings),
            y=self.y,
            target_raw=self.target_raw,
            background=self.background,
            x=self.x,
            y_coord=self.y_coord,
            grid=self.grid,
            ids=self.ids,
            qa=self.qa,
            target_units=self.target_units,
            zones=self.zones,
        )


def _read_csv(path) -> pd.DataFrame:
    # utf-8-sig strips the BOM that Excel/ArcGIS exports put on the first header.
    return pd.read_csv(path, encoding="utf-8-sig")


def encode_features(frame: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    """One-hot categorical columns and sin/cos circular (degree) columns."""
    enc = encodings or {}
    cats = list(enc.get("categorical") or [])
    circ = list(enc.get("circular_degrees") or [])
    out = {}
    for c in frame.columns:
        if c in cats:
            for level in sorted(pd.unique(frame[c].dropna())):
                out[f"{c}__{level}"] = (frame[c] == level).astype(float).to_numpy()
        elif c in circ:
            rad = np.deg2rad(frame[c].to_numpy(dtype=float))
            out[f"{c}__sin"] = np.sin(rad)
            out[f"{c}__cos"] = np.cos(rad)
        else:
            out[c] = frame[c].to_numpy(dtype=float)
    return pd.DataFrame(out, index=frame.index)


def _project(x: np.ndarray, y: np.ndarray, src: str, dst: str) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    tr = Transformer.from_crs(src, dst, always_xy=True)
    return tr.transform(x, y)


def target_qa(t: np.ndarray) -> dict:
    """Is the target a classed / rounded product?  Continuous measurements put
    ~10 % of values in each tenth of a unit; a classed raster piles them on
    whole units, which caps the useful resolution of every error metric."""
    t = np.asarray(t, float)
    frac = t - np.floor(t)
    on_int = np.isclose(t, np.round(t))
    hist = np.histogram(np.where(np.isclose(frac, 1.0), 0.0, frac), bins=10, range=(0.0, 1.0))[0] / max(t.size, 1)
    return {
        "target_fraction_integer_valued": float(on_int.mean()),
        "target_fraction_half_valued": float(np.isclose(2 * t, np.round(2 * t)).mean()),
        "target_fractional_histogram": [float(h) for h in hist],
        "target_n_unique": int(np.unique(t).size),
        # rounding error is uniform on ±0.5 for the whole-unit share
        "target_rounding_noise_sd": float(np.sqrt(on_int.mean() / 12.0)),
    }


def _role_column(cfg: CoreConfig, role: str) -> str | None:
    return ((cfg.raw.get("physics") or {}).get("roles") or {}).get(role)


def qa_flags(qa: dict, frame: pd.DataFrame, cfg: CoreConfig) -> list[dict]:
    """Data findings a reader must see next to the results (report + model card)."""
    flags: list[dict] = []
    units = cfg.data.get("target_units", "")
    fi = qa.get("target_fraction_integer_valued", 0.0)
    if fi > 0.5:
        flags.append({"code": "classed_target", "severity": "warn", "message": (
            f"{fi:.0%} of target values are whole {units or 'units'} (continuous measurements would give ~10%): "
            f"the target looks like a classed or rounded raster product, not raw measurements. Errors below "
            f"~{qa.get('target_rounding_noise_sd', 0.29):.2f} {units} are rounding noise, and fitted spatial scales "
            "partly describe the product's interpolator.")})
    col = _role_column(cfg, "albedo")
    if col and col in frame:
        a = frame[col].to_numpy(float)
        mapped = bool((cfg.raw.get("physics") or {}).get("albedo_map"))
        if not 0.03 <= float(np.mean(a)) <= 0.30:
            flags.append({"code": "albedo_scale", "severity": "warn", "message": (
                f"Mean {col} is {np.mean(a):.3f} (sd {np.std(a):.3f}); broadband urban albedo is typically "
                "0.10–0.20, so this layer is probably a single band or an index, not broadband albedo. Albedo "
                "effects are relative (per unit of this layer), not reflectance changes. "
                + ("physics.albedo_map rescales it for the physics term." if mapped else
                   "Set physics.albedo_map to rescale it for the physics term."))})
    for role in ("canopy", "impervious"):
        col = _role_column(cfg, role)
        if col and col in frame and float(frame[col].max()) <= 1.0:
            flags.append({"code": f"{role}_scale", "severity": "warn", "message": (
                f"{col} never exceeds 1 — it looks like a 0–1 fraction, but the physics expects percent (0–100).")})
    c, i = _role_column(cfg, "canopy"), _role_column(cfg, "impervious")
    if c in frame and i in frame:
        over = float(np.mean(frame[c].to_numpy(float) + frame[i].to_numpy(float) > 100.0 + 1e-9))
        if over > 0.05:
            flags.append({"code": "cover_overlap", "severity": "info", "message": (
                f"{c} + {i} exceeds 100% on {over:.0%} of cells (canopy overhanging pavement), so no sum "
                "constraint is imposed between them.")})
    dose = {}
    for var, spec in (cfg.actionable or {}).items():
        if var not in frame or not spec.get("doses"):
            continue
        v = frame[var].to_numpy(float)
        sd = float(np.std(v))
        med = float(np.median(v))
        sign = -1.0 if str(spec.get("direction", "increase")) == "decrease" else 1.0
        doses = [float(x) for x in spec["doses"] if float(x) > 0]
        dose[var] = {"sd": sd, "doses": doses, "doses_in_sd": [x / sd if sd > 0 else None for x in doses],
                     "median_cell_to_percentile": [float(np.mean(v <= med + sign * x) * 100.0) if sign > 0
                                                   else float(np.mean(v < med - x) * 100.0) for x in doses]}
        big = [x for x in doses if sd > 0 and x / sd > 3.0]
        if big:
            flags.append({"code": f"dose_scale_{var}", "severity": "info", "message": (
                f"{var} doses {', '.join(f'{x:g}' for x in big)} exceed 3 sd of the layer (sd {sd:.3g}); "
                "their effects lean on extrapolation (see the scenarios' extrapolated share).")})
    qa["dose_scale"] = dose
    if qa.get("coarse"):
        co = qa["coarse"]
        flags.append({"code": "coarse", "severity": "info", "message": (
            f"Coarse mode: {co['n_fine']:,} input cells averaged onto {co['n_cells']:,} {co['cell_m']:g} m cells "
            "(validation-study resolution; effects are for cell-mean changes at that scale).")})
    if qa.get("subsample_window_n"):
        flags.append({"code": "window", "severity": "info", "message": (
            f"Fast window: only {qa['subsample_window_n']:,} points around the centre — a smoke-test extent, "
            "too small for the spatial-block CV to mean what it does on the full area.")})
    return flags


def _fine_to_coarse(xs: np.ndarray, ys: np.ndarray, dx_fine: float, cell_m: float):
    x0, y0 = xs.min() - 0.5 * dx_fine, ys.min() - 0.5 * dx_fine
    ix = np.floor((xs - x0) / cell_m).astype(np.int64)
    iy = np.floor((ys - y0) / cell_m).astype(np.int64)
    w = int(ix.max()) + 1
    uk, inv, members = np.unique(iy * w + ix, return_inverse=True, return_counts=True)
    cx = x0 + (uk % w + 0.5) * cell_m
    cy = y0 + (uk // w + 0.5) * cell_m
    return inv.ravel(), members, cx, cy


def coarsen(df: pd.DataFrame, frame: pd.DataFrame, xs: np.ndarray, ys: np.ndarray, dx_fine: float, cell_m: float,
            cfg: CoreConfig):
    """Average fine lattice cells onto ``cell_m`` squares over the full extent.

    Continuous columns (target, predictors, a background column) are cell
    means, circular columns circular means, categorical columns and the zone
    the most common value; the id is the first member's.  Coordinates become
    the coarse cell centres.  Returns ``(df, frame, x, y, members)``."""
    if cell_m < 1.5 * dx_fine:
        raise ValueError(f"data.coarse_m={cell_m:g} must be at least 1.5× the input cell ({dx_fine:g} m)")
    inv, members, cx, cy = _fine_to_coarse(xs, ys, dx_fine, cell_m)
    k = members.size
    enc = cfg.raw.get("encodings") or {}
    cats, circ = set(enc.get("categorical") or []), set(enc.get("circular_degrees") or [])

    def mean(v):
        return np.bincount(inv, weights=np.asarray(v, float), minlength=k) / members

    def mode(v):
        codes, uniq = pd.factorize(pd.Series(v), use_na_sentinel=False)
        cnt = np.zeros((k, len(uniq)))
        np.add.at(cnt, (inv, codes), 1.0)
        return np.asarray(uniq)[cnt.argmax(axis=1)]

    def circular(v):
        r = np.deg2rad(np.asarray(v, float))
        return np.rad2deg(np.arctan2(mean(np.sin(r)), mean(np.cos(r)))) % 360.0

    out_frame = pd.DataFrame({c: (mode(frame[c]) if c in cats else circular(frame[c]) if c in circ
                                  else mean(frame[c])) for c in frame.columns})
    d = cfg.data
    out = {d["target"]: mean(df[d["target"]])}
    bg = d.get("background")
    if isinstance(bg, str) and bg in df.columns and bg not in out:
        out[bg] = mean(df[bg])
    order = np.argsort(inv, kind="stable")
    first = order[np.r_[0, np.cumsum(members)[:-1]]]
    if d.get("id") and d["id"] in df.columns:
        out[d["id"]] = df[d["id"]].to_numpy()[first]
    if d.get("zone") and d["zone"] in df.columns:
        out[d["zone"]] = mode(df[d["zone"]].to_numpy())
    return pd.DataFrame(out), out_frame, cx, cy, members


def prepare_frame(df: pd.DataFrame, cfg: CoreConfig) -> CoreData:
    """Build :class:`CoreData` from an in-memory table (used by tests too)."""
    d = cfg.data
    target, xcol, ycol = d["target"], d["x"], d["y"]
    preds = cfg.predictors
    needed = [target, xcol, ycol] + preds
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"data is missing columns: {missing}")

    qa: dict[str, Any] = {"n_input": int(len(df))}
    keep = np.isfinite(df[needed].to_numpy(dtype=float)).all(axis=1)
    qa["n_dropped_nonfinite"] = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)

    sub = d.get("subsample")
    if sub:
        idx = spatial_window_subsample(df[xcol].to_numpy(float), df[ycol].to_numpy(float), int(sub))
        df = df.iloc[idx].reset_index(drop=True)
        qa["subsample_window_n"] = int(len(df))

    frame = df[preds].astype(float).copy()
    clipped = {}
    for col, (lo, hi) in (cfg.raw["qa"].get("clip") or {}).items():
        if col in frame:
            before = frame[col].to_numpy()
            n_out = int(((before < lo) | (before > hi)).sum())
            frame[col] = np.clip(before, lo, hi)
            if n_out:
                clipped[col] = n_out
    qa["clipped"] = clipped

    # Coordinates → metres.  Default: keep the source CRS frame (scaled to
    # metres) so lattice data stays axis-aligned; optional pyproj reprojection.
    xs = df[xcol].to_numpy(float)
    ys = df[ycol].to_numpy(float)
    if d.get("reproject_to"):
        if not d.get("crs"):
            raise ValueError("data.reproject_to requires data.crs")
        xs, ys = _project(xs, ys, d["crs"], d["reproject_to"])
        xs, ys = np.asarray(xs), np.asarray(ys)
    else:
        xs = xs * cfg.coord_scale
        ys = ys * cfg.coord_scale

    # Source-product QA runs on the input values, before any aggregation.
    qa.update(target_qa(df[target].to_numpy(float)))

    grid = Grid.from_points(xs, ys, cell=d.get("cell_m"))
    cm = d.get("coarse_m")
    if cm:
        fine_dx = grid.dx
        df, frame, xs, ys, members = coarsen(df, frame, xs, ys, fine_dx, float(cm), cfg)
        grid = Grid.from_points(xs, ys, cell=float(cm))
        full = int(round(float(cm) / fine_dx)) ** 2
        qa["coarse"] = {"cell_m": float(cm), "fine_cell_m": float(fine_dx), "n_fine": int(members.sum()),
                        "n_cells": int(len(df)), "members_mean": float(members.mean()),
                        "frac_partial": float(np.mean(members < full)),
                        "target_fraction_integer_valued": float(np.mean(np.isclose(df[target], np.round(df[target]))))}
    qa["cell_m"] = grid.dx
    qa["grid_shape"] = list(grid.shape)
    qa["grid_fill_fraction"] = float(grid.mask.mean())
    qa["cell_collisions"] = float(grid.collision_fraction())

    t_raw = df[target].to_numpy(float)
    bg = d.get("background", "median")
    if isinstance(bg, (int, float)):
        background = float(bg)
        qa["background_source"] = "config"
    elif isinstance(bg, str) and bg in df.columns:
        background = float(np.nanmean(df[bg].to_numpy(float)))
        qa["background_source"] = f"column:{bg}"
    else:
        background = float(np.median(t_raw))
        qa["background_source"] = "field_median"
    qa["background"] = background
    qa["flags"] = qa_flags(qa, frame, cfg)

    ids = df[d["id"]].to_numpy() if d.get("id") and d["id"] in df.columns else np.arange(len(df))
    zones = df[d["zone"]].to_numpy() if d.get("zone") and d["zone"] in df.columns else None
    return CoreData(
        frame=frame,
        X=encode_features(frame, cfg.raw.get("encodings")),
        y=t_raw - background,
        target_raw=t_raw,
        background=background,
        x=xs,
        y_coord=ys,
        grid=grid,
        ids=ids,
        qa=qa,
        target_units=str(d.get("target_units", "")),
        zones=zones,
    )


def read_input(cfg: CoreConfig) -> pd.DataFrame:
    """The configured table, with ``data.join`` tables merged by id (e.g. open-data
    features from ``sparc core features``: ``join: {path: ..., on: OBJECTID}``)."""
    df = _read_csv(cfg.data_path)
    for j in cfg.data.get("join") or []:
        path = cfg.resolve_path(j["path"])
        extra = pd.read_parquet(path) if str(path).endswith(".parquet") else _read_csv(path)
        key = j.get("on") or cfg.data.get("id")
        right = j.get("right_on", "id" if "id" in extra.columns else key)
        extra = extra.rename(columns={right: key}) if right != key else extra
        cols = [c for c in extra.columns if c != key and c not in df.columns]
        df = df.merge(extra[[key] + cols], on=key, how="left")
    return df


def load_core_data(cfg: CoreConfig) -> CoreData:
    """S0: read the configured table and return QA'd, encoded :class:`CoreData`."""
    return prepare_frame(read_input(cfg), cfg)
