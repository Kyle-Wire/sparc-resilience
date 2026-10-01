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

    grid = Grid.from_points(xs, ys, cell=d.get("cell_m"))
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

    # Quantisation diagnostic: many raster products are rounded to whole units.
    frac_integer = float(np.mean(np.isclose(t_raw, np.round(t_raw))))
    qa["target_fraction_integer_valued"] = frac_integer

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


def load_core_data(cfg: CoreConfig) -> CoreData:
    """S0: read the configured table and return QA'd, encoded :class:`CoreData`."""
    return prepare_frame(_read_csv(cfg.data_path), cfg)
