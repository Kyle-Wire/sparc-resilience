"""Selections (SPEC §6.6, api.md §1, §6.3): ``resolve(source, spec) -> bool[n]`` for every SelectionSpec kind.

Pure numpy over the run grid's cell centres (a few ms on 54,701 cells):

* ``polygon`` / ``rect`` / ``circle``: pyproj into the run frame when the crs is ``EPSG:4326``, then
  ``matplotlib.path.Path.contains_points`` (polygons: the first ring is the outline, later rings are holes);
  ``radius_m`` is metres in the run frame;
* ``buffer``: ``scipy.ndimage.distance_transform_edt`` on the raster (cells within ``radius_m`` of the
  selected cells, or within the S1 influence range of a lever with ``lever_range``);
* ``hex``: ``sparc.core.planner.hex_ids`` of the cell centres (the web client computes the same keys);
* ``zones``: the zone codes of ``GridMeta.zones`` (numbers and strings compare by value);
* ``filter`` / ``top``: a namespaced column (``predictor:``, ``layer:``, ``pred:``, ``response:<var>:``,
  ``planner:``, ``configured:<slug>``, ``result:<res_id>:delta``) or a layer key;
* ``cells`` (ids), ``blob`` (an uploaded bitset), ``region`` (a saved region) and the combinators
  ``and`` / ``or`` / ``minus`` (the first argument minus the union of the rest) / ``not``.

Masks travel as :func:`encode_bitset` strings: base64 of a little-endian, LSB-first bit array (byte ``i``
holds rows ``8i … 8i+7``).
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError

__all__ = ["encode_bitset", "decode_bitset", "mask_bytes", "mask_from_bytes", "resolve", "is_portable",
           "RunSource", "column_values", "summarize", "spec_dict", "MAX_DEPTH"]

MAX_DEPTH = 16
_PRED_COLS = {"target": "obs", "pred": "pred", "resid": "resid", "halfwidth": "halfwidth",
              "dist_train_m": "dist_train_m"}


# ---------------------------------------------------------------------------
# bitsets
# ---------------------------------------------------------------------------

def mask_bytes(mask) -> bytes:
    """Little-endian, LSB-first packed bits of a boolean row mask."""
    return np.packbits(np.asarray(mask, dtype=bool), bitorder="little").tobytes()


def mask_from_bytes(raw: bytes, n: int) -> np.ndarray:
    need = (n + 7) // 8
    if len(raw) != need:
        raise ApiError("validation", f"a bitset for {n} rows has {need} bytes, not {len(raw)}",
                       detail={"errors": [{"path": "mask", "message": "wrong length", "code": "bitset_length"}]})
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:n].astype(bool)


def encode_bitset(mask) -> str:
    return base64.b64encode(mask_bytes(mask)).decode("ascii")


def decode_bitset(b64: str, n: int) -> np.ndarray:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError):
        raise ApiError("validation", "the bitset is not valid base64",
                       detail={"errors": [{"path": "mask", "message": "not base64", "code": "bitset"}]})
    return mask_from_bytes(raw, n)


def spec_dict(spec: Any) -> dict:
    """A SelectionSpec (pydantic model or dict) as a plain dict."""
    if hasattr(spec, "model_dump"):
        return spec.model_dump(mode="json", exclude_none=True)
    return dict(spec)


def _bad(msg: str, path: str = "selection", code: str = "selection") -> ApiError:
    return ApiError("validation", msg, detail={"errors": [{"path": path, "message": msg, "code": code}]})


# ---------------------------------------------------------------------------
# a run as a selection source
# ---------------------------------------------------------------------------

class RunSource:
    """What :func:`resolve` needs from a run: the grid, columns, saved regions, blobs and lever ranges."""

    def __init__(self, ctx, db=None):
        self.ctx = ctx
        self.db = db
        self.grid = ctx.grid
        if self.grid is None:
            raise ApiError("output_missing", "the run's grid is not known yet (S0 has not finished)",
                           detail={"output": "grid", "produced_by": "stage:S0", "expected_path": None})

    @property
    def n(self) -> int:
        return self.grid.n

    def column(self, name: str) -> np.ndarray:
        return column_values(self.ctx, name)

    def region(self, rid: str) -> dict:
        row = self.db.fetchone("SELECT spec_json FROM regions WHERE id = ?", (rid,)) if self.db is not None else None
        if row is None:
            raise _bad(f"no region {rid!r}", "selection.id", "unknown_region")
        return json.loads(row["spec_json"])

    def blob(self, bid: str) -> np.ndarray:
        row = self.db.fetchone("SELECT run_id, kind, path FROM blobs WHERE id = ?", (bid,)) if self.db is not None \
            else None
        if row is None or row.get("kind") != "mask":
            raise _bad(f"no mask blob {bid!r}", "selection.blob_id", "unknown_blob")
        if row.get("run_id") != self.ctx.run_id:
            raise _bad(f"blob {bid!r} belongs to another run (blobs are not portable)", "selection.blob_id",
                       "foreign_blob")
        try:
            raw = Path(row["path"]).read_bytes()
        except OSError:
            raise _bad(f"blob {bid!r} is gone", "selection.blob_id", "unknown_blob")
        return mask_from_bytes(raw, self.n)

    def lever_range(self, var: str) -> float:
        ranges = ((self.ctx.manifest.get("influence") or {}).get("ranges_m")
                  or (self.ctx.json("influence.json") or {}).get("ranges_m") or {})
        if var not in ranges or ranges[var] is None:
            raise _bad(f"no influence range for {var!r} (S1 has not run, or it is not a predictor)",
                       "selection.lever_range", "no_range")
        return float(ranges[var])


def column_values(ctx, column: str) -> np.ndarray:
    """A namespaced column (api.md §1) or a layer key as float64 in row order (``422 validation`` if unknown)."""
    from sparc.studio.runs import layers as L

    def layer(key):
        try:
            return np.asarray(L.layer_array(ctx, key), dtype=np.float64)
        except ApiError as exc:
            if exc.code in ("unknown_layer", "output_missing"):
                raise _bad(f"unknown column {column!r}: {exc.message}", "column", "unknown_column")
            raise

    ns, _, rest = column.partition(":")
    if not rest:
        return layer(column)
    if ns == "predictor":
        data = ctx.data
        if data is None:
            raise _bad(f"predictor columns are unavailable for this run ({ctx.data_error})", "column", "no_data")
        if rest not in data.frame.columns:
            raise _bad(f"unknown predictor {rest!r}", "column", "unknown_column")
        return data.frame[rest].to_numpy(dtype=np.float64)
    if ns == "layer":
        return layer(rest)
    if ns == "pred":
        if rest not in _PRED_COLS:
            raise _bad(f"pred: column must be one of {sorted(_PRED_COLS)}", "column", "unknown_column")
        return layer(_PRED_COLS[rest])
    if ns == "response":
        var, _, col = rest.partition(":")
        df = ctx.parquet(f"response_{var}.parquet")
        if df is None or col not in df.columns:
            raise _bad(f"unknown response column {column!r}", "column", "unknown_column")
        return df[col].to_numpy(dtype=np.float64)
    if ns == "planner":
        df = ctx.parquet("planner/planner_cells.parquet")
        if df is None or rest not in df.columns:
            raise _bad(f"unknown planner column {column!r}", "column", "unknown_column")
        return np.asarray(L._aligned(ctx, df, rest), dtype=np.float64)
    if ns == "configured":
        return layer(f"sc:{rest}")
    if ns == "result":
        rid, _, field_ = rest.partition(":")
        return layer(f"res:{rid}:{field_ or 'delta'}")
    if ns in ("sc", "sc_sd", "sc_ex", "res", "plan", "cmp"):
        return layer(column)
    raise _bad(f"unknown column namespace {ns!r}", "column", "unknown_namespace")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _centres(grid) -> np.ndarray:
    return np.column_stack([grid.x, grid.y])


def _to_run(src, crs: str, pts) -> np.ndarray:
    from sparc.studio.runs.grid import to_run_xy

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if crs == "run_xy_m":
        return pts
    if crs != "EPSG:4326":
        raise _bad(f"unknown crs {crs!r}", "selection.crs", "crs")
    g = src.grid
    if not g.crs:
        raise ApiError("needs_crs", "this run has no CRS: use run_xy_m geometry",
                       detail={"errors": [{"path": "selection.crs", "message": "no CRS", "code": "needs_crs"}]})
    x, y = to_run_xy(g.crs, g.coord_scale, pts[:, 0], pts[:, 1])
    return np.column_stack([x, y])


def _densify(ring: np.ndarray, steps: int = 16) -> np.ndarray:
    out = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        for t in np.linspace(0, 1, steps, endpoint=False):
            out.append(a + (b - a) * t)
    return np.array(out)


def _in_polygon(points: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    from matplotlib.path import Path as MPath

    if not rings or len(rings[0]) < 3:
        raise _bad("a polygon needs an outline of at least 3 points", "selection.rings", "geometry")
    inside = MPath(rings[0]).contains_points(points)
    for hole in rings[1:]:
        if len(hole) >= 3:
            inside &= ~MPath(hole).contains_points(points)
    return inside


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def resolve(src, spec, _depth: int = 0) -> np.ndarray:
    """Boolean row mask of ``spec`` on ``src`` (a :class:`RunSource` or any object with the same members)."""
    if _depth > MAX_DEPTH:
        raise _bad("the selection nests too deep (a region that refers to itself?)", "selection", "depth")
    s = spec_dict(spec)
    g = src.grid
    n = g.n
    op = s.get("op")
    if op in ("and", "or", "minus"):
        args = [resolve(src, a, _depth + 1) for a in (s.get("args") or [])]
        if not args:
            raise _bad(f"{op} needs at least one argument", "selection.args", "combinator")
        out = args[0].copy()
        if op == "and":
            for a in args[1:]:
                out &= a
        elif op == "or":
            for a in args[1:]:
                out |= a
        else:
            for a in args[1:]:
                out &= ~a
        return out
    if op == "not":
        return ~resolve(src, s.get("arg") or {"kind": "all"}, _depth + 1)
    kind = s.get("kind")
    if kind == "all":
        return np.ones(n, dtype=bool)
    if kind == "zones":
        codes = g.zone_codes
        if codes is None:
            raise _bad("this run has no zones", "selection.values", "no_zones")
        want = {_zkey(v) for v in s.get("values") or []}
        return np.array([c is not None and _zkey(c) in want for c in codes], dtype=bool)
    if kind == "polygon":
        rings = [_to_run(src, s["crs"], r) for r in s.get("rings") or []]
        return _in_polygon(_centres(g), rings)
    if kind == "rect":
        lo, hi = np.asarray(s["min"], float), np.asarray(s["max"], float)
        if s["crs"] == "run_xy_m":
            c = _centres(g)
            return (c[:, 0] >= lo[0]) & (c[:, 0] <= hi[0]) & (c[:, 1] >= lo[1]) & (c[:, 1] <= hi[1])
        ring = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        return _in_polygon(_centres(g), [_to_run(src, s["crs"], _densify(ring))])
    if kind == "circle":
        r = float(s.get("radius_m") or 0.0)
        if not math.isfinite(r) or r < 0:
            raise _bad("radius_m must be ≥ 0", "selection.radius_m", "geometry")
        cx, cy = _to_run(src, s["crs"], [s["center"]])[0]
        c = _centres(g)
        return (c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 <= r * r + 1e-9
    if kind == "cells":
        ids = np.asarray(g.ids).astype(str)
        want = {str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v) for v in s.get("ids") or []}
        return np.isin(ids, list(want))
    if kind == "blob":
        return src.blob(str(s.get("blob_id")))
    if kind == "hex":
        from sparc.core.planner import hex_ids

        size = float(s.get("size_m") or 250)
        keys = hex_ids(g.x, g.y, size)[0]
        return np.isin(keys, np.asarray(s.get("keys") or [], dtype=np.int64))
    if kind == "filter":
        return _filter(src, s)
    if kind == "top":
        return _top(src, s, _depth)
    if kind == "buffer":
        return _buffer(src, s, _depth)
    if kind == "region":
        return resolve(src, src.region(str(s.get("id"))), _depth + 1)
    raise _bad(f"unknown selection kind {kind or op!r}", "selection.kind", "kind")


def _zkey(v):
    if isinstance(v, bool):
        return ("s", str(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return ("n", int(f)) if f.is_integer() else ("n", f)
    try:
        f = float(str(v))
        return ("n", int(f)) if f.is_integer() else ("n", f)
    except ValueError:
        return ("s", str(v))


def _filter(src, s: dict) -> np.ndarray:
    v = src.column(str(s.get("column")))
    op = s.get("op")
    val = s.get("value")
    with np.errstate(invalid="ignore"):
        if op in ("<", "<=", ">", ">=", "=="):
            try:
                x = float(val)
            except (TypeError, ValueError):
                raise _bad(f"filter {op} needs a number", "selection.value", "filter")
            return {"<": v < x, "<=": v <= x, ">": v > x, ">=": v >= x, "==": np.isclose(v, x)}[op] & np.isfinite(v)
        if op == "between":
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise _bad("between needs [lo, hi]", "selection.value", "filter")
            lo, hi = sorted((float(val[0]), float(val[1])))
            return (v >= lo) & (v <= hi)
        if op == "in":
            vals = val if isinstance(val, (list, tuple)) else [val]
            try:
                nums = np.asarray([float(x) for x in vals], dtype=np.float64)
            except (TypeError, ValueError):
                raise _bad("in needs numbers for a numeric column", "selection.value", "filter")
            return np.isin(v, nums)
    raise _bad(f"unknown filter op {op!r}", "selection.op", "filter")


def _top(src, s: dict, depth: int) -> np.ndarray:
    v = src.column(str(s.get("column")))
    within = resolve(src, s["within"], depth + 1) if s.get("within") else np.ones(src.grid.n, dtype=bool)
    cand = np.flatnonzero(within & np.isfinite(v))
    if s.get("k") is not None:
        k = int(s["k"])
    elif s.get("frac") is not None:
        f = float(s["frac"])
        if not 0 <= f <= 1:
            raise _bad("frac must be within [0, 1]", "selection.frac", "top")
        k = int(round(f * cand.size))
    else:
        raise _bad("top needs k or frac", "selection", "top")
    k = max(0, min(k, cand.size))
    vals = v[cand]
    order = np.argsort(-vals if s.get("direction", "highest") == "highest" else vals, kind="stable")
    out = np.zeros(src.grid.n, dtype=bool)
    out[cand[order[:k]]] = True
    return out


def _buffer(src, s: dict, depth: int) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    base = resolve(src, s["of"], depth + 1)
    if s.get("radius_m") is not None:
        r = float(s["radius_m"])
    elif s.get("lever_range"):
        r = src.lever_range(str(s["lever_range"]))
    else:
        raise _bad("buffer needs radius_m or lever_range", "selection", "buffer")
    g = src.grid
    if not base.any():
        return base.copy()
    ras = np.zeros((g.ny, g.nx), dtype=bool)
    ras[g.iy[base], g.ix[base]] = True
    dist = distance_transform_edt(~ras, sampling=g.dx)
    return dist[g.iy, g.ix] <= r + 1e-6


def is_portable(spec, db=None, _depth: int = 0) -> bool:
    """Whether ``spec`` means the same cells on other runs of the project (lon/lat geometry, attributes, ids)."""
    s = spec_dict(spec)
    if _depth > MAX_DEPTH:
        return False
    if s.get("op") in ("and", "or", "minus"):
        return all(is_portable(a, db, _depth + 1) for a in s.get("args") or [])
    if s.get("op") == "not":
        return is_portable(s.get("arg") or {}, db, _depth + 1)
    kind = s.get("kind")
    if kind in ("polygon", "rect", "circle"):
        return s.get("crs") == "EPSG:4326"
    if kind == "blob":
        return False
    if kind in ("filter", "top"):
        col = str(s.get("column") or "")
        if col.startswith(("result:", "res:", "plan:", "cmp:")):
            return False
        if kind == "top" and s.get("within"):
            return is_portable(s["within"], db, _depth + 1)
        return True
    if kind == "buffer":
        return is_portable(s.get("of") or {}, db, _depth + 1)
    if kind == "region" and db is not None:
        row = db.fetchone("SELECT spec_json FROM regions WHERE id = ?", (s.get("id"),))
        return bool(row) and is_portable(json.loads(row["spec_json"]), db, _depth + 1)
    return True


def summarize(ctx, src, mask: np.ndarray) -> dict:
    """``{n_cells, area_km2, people, medians}`` of a mask (people from the planner layers when present)."""
    from sparc.studio.runs import layers as L

    g = src.grid
    n = int(mask.sum())
    out: dict[str, Any] = {"n_cells": n, "area_km2": n * g.dx * g.dx / 1e6, "people": None, "medians": {}}
    defs = L.layer_defs(ctx)
    if "people" in defs:
        try:
            p = np.asarray(L.layer_array(ctx, "people"), dtype=np.float64)
            out["people"] = float(np.nansum(p[mask]))
        except ApiError:
            pass
    keys = [k for k in L.lever_info(ctx) if k in defs] + (["obs"] if "obs" in defs else [])
    for k in keys:
        try:
            v = np.asarray(L.layer_array(ctx, k), dtype=np.float64)[mask]
            v = v[np.isfinite(v)]
            out["medians"][k] = float(np.median(v)) if v.size else None
        except ApiError:
            out["medians"][k] = None
    return out
