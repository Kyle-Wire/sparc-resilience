"""Data check: S0 inline (api.md §5.2) and the preview binaries it leaves behind.

``data_check`` loads the table exactly as a run's S0 would (``read_input``
with joins, ``prepare_frame`` with QA clipping, subsample/coarse, grid,
background and QA flags) on the saved config deep-merged with an optional
``config_patch`` (unsaved form values).  Predictors missing from the table
are reported in ``columns_missing`` and left out, so the check also works
for a project that has no predictors yet.

Each check stores a preview under a token (30 min, LRU): the packed grid
(``ix, iy: int32, lon, lat: float32``, api.md §0.4) with its ``GridMeta`` and
every numeric column as Float32 in row order (predictors as QA-clipped,
coarse cells as their first member's value except predictors and the
target, which are cell means).
"""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sparc.studio.errors import ApiError
from sparc.studio.projects.config_service import build_core_config

__all__ = ["data_check", "PreviewStore", "pack_arrays", "grid_meta", "deep_merge", "INLINE_MAX_ROWS",
           "PREVIEW_TTL_S", "points_lonlat"]

INLINE_MAX_ROWS = 2_000_000
PREVIEW_TTL_S = 30 * 60.0
_ROW = "__sparc_row__"


def deep_merge(base: dict, patch: dict | None) -> dict:
    out = copy.deepcopy(base or {})
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# preview store
# ---------------------------------------------------------------------------

class PreviewStore:
    """Thread-safe LRU of data-check previews, bounded by count, bytes and age."""

    def __init__(self, max_items: int = 12, max_bytes: int = 256 * 1024 ** 2, ttl_s: float = PREVIEW_TTL_S):
        self.max_items, self.max_bytes, self.ttl_s = max_items, max_bytes, ttl_s
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, project_id: str, payload: dict) -> str:
        token = secrets.token_urlsafe(12)
        payload = {**payload, "project_id": project_id, "created": time.monotonic()}
        with self._lock:
            self._items[token] = payload
            self._evict()
        return token

    def _evict(self) -> None:
        now = time.monotonic()
        for t in [t for t, p in self._items.items() if now - p["created"] > self.ttl_s]:
            self._items.pop(t, None)
        while len(self._items) > self.max_items or (
                len(self._items) > 1 and sum(p.get("nbytes", 0) for p in self._items.values()) > self.max_bytes):
            self._items.popitem(last=False)

    def get(self, project_id: str, token: str) -> dict:
        with self._lock:
            self._evict()
            p = self._items.get(token)
            if p is None or p["project_id"] != project_id:
                raise ApiError("not_found", "preview expired or unknown; run the data check again",
                               detail={"token": token})
            self._items.move_to_end(token)
            return p


# ---------------------------------------------------------------------------
# binaries
# ---------------------------------------------------------------------------

def pack_arrays(arrays: list[tuple[str, np.ndarray]]) -> tuple[bytes, list[dict]]:
    """Concatenate little-endian arrays, each 8-byte aligned; returns ``(body, X-SPARC-Offsets entries)``."""
    names = {"<i4": "int32", "<f4": "float32", "|u1": "uint8", "<i8": "int64", "<i2": "int16"}
    out = bytearray()
    offsets = []
    for name, a in arrays:
        a = np.ascontiguousarray(a)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        pad = (-len(out)) % 8
        out += b"\0" * pad
        offsets.append({"name": name, "dtype": names[a.dtype.str], "offset": len(out), "length": int(a.size)})
        out += a.tobytes()
    return bytes(out), offsets


def points_lonlat(x_m: np.ndarray, y_m: np.ndarray, data_cfg: dict, coord_scale: float) \
        -> tuple[np.ndarray, np.ndarray] | None:
    """Lon/lat of grid-frame metres (as ``sparc.core.opendata.points_lonlat``); None without a CRS."""
    crs = data_cfg.get("reproject_to") or data_cfg.get("crs")
    if not crs:
        return None
    from pyproj import Transformer

    s = 1.0 if data_cfg.get("reproject_to") else coord_scale
    try:
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(np.asarray(x_m, float) / s, np.asarray(y_m, float) / s)
    except Exception:                                 # noqa: BLE001 - an unknown CRS: no lon/lat
        return None
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if not (np.isfinite(lon).any() and np.isfinite(lat).any()):
        return None
    return lon, lat


def grid_meta(data, data_cfg: dict, coord_scale: float, *, units: dict, zones: list, ids_kind: str,
              etag: str, n_folds: int | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """``GridMeta`` (api.md §1) of a ``CoreData`` plus the per-row lon/lat (NaN without a CRS).

    ``corners`` are the [lat, lon] of the grid's corner cell centres (cell
    (0, 0) is ``sw``; rows grow north), the frame ``studio-web``'s map
    interpolates between.
    """
    g = data.grid
    n = int(data.n)
    ll = points_lonlat(data.x, data.y_coord, data_cfg, coord_scale)
    corners = bounds = None
    if ll is not None:
        lon, lat = ll
        cx = np.array([g.x0, g.x0 + (g.nx - 1) * g.dx, g.x0, g.x0 + (g.nx - 1) * g.dx])
        cy = np.array([g.y0, g.y0, g.y0 + (g.ny - 1) * g.dy, g.y0 + (g.ny - 1) * g.dy])
        c = points_lonlat(cx, cy, data_cfg, coord_scale)
        if c is not None:
            clon, clat = c
            corners = {k: [float(clat[i]), float(clon[i])] for i, k in enumerate(("sw", "se", "nw", "ne"))}
        bounds = [float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat))]
    else:
        lon = lat = np.full(n, np.nan)
    meta = {"n": n, "nx": int(g.nx), "ny": int(g.ny), "dx_m": float(g.dx), "x0_m": float(g.x0),
            "y0_m": float(g.y0), "crs": data_cfg.get("reproject_to") or data_cfg.get("crs"),
            "coord_scale": float(coord_scale), "has_lonlat": ll is not None, "bounds_lonlat": bounds,
            "corners": corners, "ids_kind": ids_kind, "zones": zones, "n_folds": n_folds, "units": units,
            "background": float(data.background) if np.isfinite(data.background) else None, "etag": etag}
    return meta, np.asarray(lon, np.float32), np.asarray(lat, np.float32)


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------

def _bad(errors: list[dict], message: str) -> ApiError:
    return ApiError("validation", message, detail={"errors": errors})


def data_check(raw: dict, project_dir: str | Path, patch: dict | None = None) -> tuple[dict, dict]:
    """S0 on the project's data: ``(response, preview payload)`` (see the module docstring)."""
    from threadpoolctl import threadpool_limits

    from sparc.core.data import prepare_frame, read_input
    from sparc.studio.projects.files import count_rows

    merged = deep_merge(raw, patch)
    cfg = build_core_config(merged, project_dir)
    d = cfg.data
    missing = [{"path": f"data.{k}", "message": f"data.{k} is required", "code": "required"}
               for k in ("target", "x", "y") if not d.get(k)]
    if missing:
        raise _bad(missing, "map the target and the x/y columns first")
    if not d.get("path"):
        raise _bad([{"path": "data.path", "message": "data.path is required", "code": "required"}],
                   "add a data file first")
    path = cfg.data_path
    if not Path(path).is_file():
        raise ApiError("not_found", f"data file not found: {d.get('path')}", detail={"path": str(d.get("path"))})
    try:
        n_rows, _exact = count_rows(path)
    except ApiError:
        raise
    except OSError as exc:
        raise _bad([{"path": "data.path", "message": str(exc), "code": "unreadable"}], "the data file is unreadable")
    if n_rows > INLINE_MAX_ROWS:
        raise ApiError("too_large_inline", f"{n_rows:,} rows is too many for an inline check "
                       f"(limit {INLINE_MAX_ROWS:,}); launch a fast run instead",
                       detail={"n_rows": n_rows, "limit": INLINE_MAX_ROWS})
    t0 = time.perf_counter()
    with threadpool_limits(1):
        try:
            df = read_input(cfg)
        except (ValueError, KeyError, OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise _bad([{"path": "data.path", "message": str(exc)[:300], "code": "unreadable"}],
                       f"cannot read the data: {exc}")
        absent = [{"path": f"data.{k}", "message": f"column {d[k]!r} is not in the data", "code": "missing_column"}
                  for k in ("target", "x", "y") if d[k] not in df.columns]
        if absent:
            raise _bad(absent, "the target or coordinate columns are missing")
        flags: list[dict] = []
        for k in ("target", "x", "y"):
            if not pd.api.types.is_numeric_dtype(df[d[k]]):
                raise _bad([{"path": f"data.{k}", "message": f"column {d[k]!r} is not numeric",
                             "code": "not_numeric"}], f"column {d[k]!r} is not numeric")
        preds = list(cfg.predictors)
        columns_missing = [p for p in preds if p not in df.columns]
        cats = set((cfg.raw.get("encodings") or {}).get("categorical") or [])
        usable = []
        for p in preds:
            if p in columns_missing:
                continue
            if not pd.api.types.is_numeric_dtype(df[p]) and p not in cats:
                flags.append({"code": f"not_numeric_{p}", "severity": "warn",
                              "message": f"predictor {p} is not numeric; it was left out of the check"})
                continue
            usable.append(p)
        ids_col = d.get("id") if d.get("id") in df.columns else None
        ids_kind = "int" if ids_col is None or pd.api.types.is_integer_dtype(df[ids_col]) else "str"
        cfg.raw["predictors"] = usable
        cfg.raw["data"]["id"] = _ROW
        df = df.copy()
        df[_ROW] = np.arange(len(df), dtype=np.int64)
        try:
            data = prepare_frame(df, cfg)
        except (ValueError, KeyError) as exc:
            raise _bad([{"path": "data", "message": str(exc)[:300], "code": "s0_failed"}], f"S0 failed: {exc}")
    elapsed = time.perf_counter() - t0
    qa = data.qa
    rows = np.asarray(data.ids, dtype=np.int64)
    zones: list = []
    if data.zones is not None:
        uz = [z.item() if hasattr(z, "item") else z for z in pd.unique(pd.Series(data.zones).dropna())]
        try:
            uz = sorted(uz)
        except TypeError:                            # mixed kinds: keep first-seen order
            pass
        zones = uz[:500]
    token_seed = hashlib.sha1(f"{project_dir}:{time.time()}".encode()).hexdigest()[:16]
    meta, lon, lat = grid_meta(data, d, cfg.coord_scale, units={"target": str(d.get("target_units") or "")},
                               zones=zones, ids_kind=ids_kind, etag=f"preview-{token_seed}")
    body, offsets = pack_arrays([("ix", data.grid.ix.astype(np.int32)), ("iy", data.grid.iy.astype(np.int32)),
                                 ("lon", lon), ("lat", lat)])
    columns: dict[str, np.ndarray] = {}
    for c in df.columns:
        if c == _ROW or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if c in data.frame.columns:
            v = data.frame[c].to_numpy(float)
        elif c == d["target"]:
            v = data.target_raw
        else:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)[rows]
        columns[str(c)] = np.asarray(v, dtype=np.float32)
    dose = {}
    for var, s in (qa.get("dose_scale") or {}).items():
        dose[var] = {"sd": s.get("sd"), "doses": s.get("doses") or [], "doses_in_sd": s.get("doses_in_sd") or [],
                     "percentile_reached": s.get("median_cell_to_percentile") or []}
    fi = float(qa.get("target_fraction_integer_valued") or 0.0)
    resp = {
        "n_points": int(data.n), "n_input": int(qa.get("n_input", len(df))),
        "n_dropped": int(qa.get("n_dropped_nonfinite", 0)), "clipped": {k: int(v) for k, v in
                                                                        (qa.get("clipped") or {}).items()},
        "grid": {"nx": int(data.grid.nx), "ny": int(data.grid.ny), "cell_m": float(data.grid.dx),
                 "fill_fraction": float(qa.get("grid_fill_fraction", 0.0)),
                 "collisions": float(qa.get("cell_collisions", 0.0))},
        "background": {"value": float(data.background), "source": str(qa.get("background_source", ""))},
        "noise_floor": float(qa["target_rounding_noise_sd"]) if fi > 0.5 and "target_rounding_noise_sd" in qa else None,
        "flags": [{"code": f["code"], "severity": f["severity"], "message": f["message"]}
                  for f in (qa.get("flags") or [])] + flags,
        "dose_scale": dose, "coarse": qa.get("coarse") or None,
        "extent_m": [float(np.ptp(data.x)), float(np.ptp(data.y_coord))],
        "columns_missing": columns_missing, "preview_token": "",
        "preview_columns": sorted(columns), "elapsed_s": round(elapsed, 3),
    }
    payload = {"grid_body": body, "grid_offsets": offsets, "grid_meta": meta, "columns": columns,
               "nbytes": len(body) + sum(a.nbytes for a in columns.values())}
    return resp, payload


def preview_column(payload: dict, name: str) -> np.ndarray:
    cols = payload["columns"]
    if name not in cols:
        raise ApiError("not_found", f"no preview column {name!r}", detail={"columns": sorted(cols)})
    return cols[name]


def dumps_header(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str)
