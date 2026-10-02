"""Single-layer and hexagon exports (SPEC §6.7, api.md §6.2 ``/hex?fmt=``, §6.4 ``/export/layer/{key}``).

* GeoTIFF: float32 on the run grid in the data's CRS (deflate, nodata NaN) through
  ``sparc.core.planner.export_geotiffs``; written to a temporary folder and streamed, never stored.
* CSV: ``id, lon, lat`` (with a CRS) and the value; Parquet: ``id, x_m, y_m, lon, lat`` and the value;
  GeoJSON: points in EPSG:4326 (≤ 60,000 features, else ``413 too_many_features``).
* Hexagons: CSV, GeoJSON polygons (EPSG:4326) or a GeoPackage in the data CRS.

Formats that need coordinates on the earth (GeoTIFF, GeoJSON, GPKG) refuse runs without a CRS with
``422 needs_crs``.
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sparc.studio.errors import ApiError

__all__ = ["export_layer", "export_hex", "MAX_FEATURES"]

MAX_FEATURES = 60_000


def _needs_crs(ctx, what: str) -> None:
    g = ctx.grid
    if g is None or not g.crs:
        raise ApiError("needs_crs", f"{what} needs a run with a CRS (set data.crs in the config)",
                       detail={"errors": [{"path": "fmt", "message": "no CRS", "code": "needs_crs"}]})


def _safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "layer"


def _cfg_for_export(ctx):
    cfg = ctx.cfg
    if cfg is not None:
        return cfg
    g = ctx.grid
    return SimpleNamespace(data={"crs": g.crs, "reproject_to": None}, coord_scale=g.coord_scale)


def export_layer(ctx, key: str, fmt: str) -> tuple[bytes | Path, str, str, Path | None]:
    """``(body or file, media type, file name, temp dir to remove)`` of one layer."""
    import pandas as pd

    from sparc.studio.runs.layers import layer_array, layer_defs

    vals = np.asarray(layer_array(ctx, key), dtype=np.float64)
    d = layer_defs(ctx).get(key)
    if d is not None and d.dtype == "uint8":
        vals = np.where(vals == 255, np.nan, vals)
    g = ctx.grid
    name = _safe_name(key)
    if fmt == "tif":
        _needs_crs(ctx, "GeoTIFF")
        from sparc.core.planner import export_geotiffs

        tmp = Path(tempfile.mkdtemp(prefix="sparc-layer-"))
        try:
            paths = export_geotiffs(SimpleNamespace(grid=g.core_grid()), _cfg_for_export(ctx), {name: vals}, tmp)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return Path(paths[0]), "image/tiff", f"{ctx.run_id}_{name}.tif", tmp
    df = pd.DataFrame({"id": g.ids})
    if fmt == "parquet":
        df["x_m"], df["y_m"] = g.x, g.y
    if g.crs:
        df["lon"], df["lat"] = np.round(g.lon.astype(np.float64), 7), np.round(g.lat.astype(np.float64), 7)
    df[key] = vals
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv; charset=utf-8", f"{ctx.run_id}_{name}.csv", None
    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue(), "application/vnd.apache.parquet", f"{ctx.run_id}_{name}.parquet", None
    if fmt == "geojson":
        _needs_crs(ctx, "GeoJSON")
        if g.n > MAX_FEATURES:
            raise ApiError("too_many_features", f"{g.n:,} cells exceed the {MAX_FEATURES:,}-feature GeoJSON limit; "
                           "export GeoTIFF or CSV instead", detail={"n": g.n, "max": MAX_FEATURES})
        from sparc.studio.runs.outputs import geojson_points

        body = geojson_points(df[["id", key]], g.lon.astype(np.float64), g.lat.astype(np.float64))
        return body.encode("utf-8"), "application/geo+json", f"{ctx.run_id}_{name}.geojson", None
    raise ApiError("validation", f"unknown format {fmt!r}",
                   detail={"errors": [{"path": "fmt", "message": "tif, csv, geojson or parquet", "code": "fmt"}]})


def _hex_polygon(cx: float, cy: float, size_m: float) -> np.ndarray:
    R = size_m / math.sqrt(3.0)
    ang = np.deg2rad(np.arange(7) * 60.0 + 30.0)
    return np.column_stack([cx + R * np.cos(ang), cy + R * np.sin(ang)])


def export_hex(ctx, df, size_m: float, fmt: str) -> tuple[bytes | Path, str, str, Path | None]:
    """A hexagon table (:func:`sparc.studio.runs.stats.hex_table`) as CSV, GeoJSON or GPKG."""
    from sparc.studio.runs.common import clean

    name = f"{ctx.run_id}_hex_{int(size_m)}m"
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv; charset=utf-8", f"{name}.csv", None
    if fmt == "geojson":
        _needs_crs(ctx, "GeoJSON")
        from sparc.studio.runs.grid import lonlat_transformer, transform_xy

        g = ctx.grid
        tr = lonlat_transformer(g.crs)
        feats = []
        props = [c for c in df.columns if c not in ("cx", "cy")]
        for rec in df.to_dict(orient="records"):
            ring = _hex_polygon(rec["cx"], rec["cy"], size_m)
            lon, lat = transform_xy(tr, ring[:, 0] / g.coord_scale, ring[:, 1] / g.coord_scale)
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
                [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon, lat)]]},
                "properties": clean({k: rec[k] for k in props})})
        body = json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"))
        return body.encode("utf-8"), "application/geo+json", f"{name}.geojson", None
    if fmt == "gpkg":
        _needs_crs(ctx, "GeoPackage")
        from sparc.core.planner import export_hex_gpkg

        tmp = Path(tempfile.mkdtemp(prefix="sparc-hex-"))
        out = tmp / f"{name}.gpkg"
        try:
            export_hex_gpkg(df.drop(columns=["lon", "lat"]).rename(columns={"key": "hex"}), size_m,
                            _cfg_for_export(ctx), out)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ApiError("no_conversion", f"GeoPackage export failed: {exc}") from exc
        return out, "application/geopackage+sqlite3", out.name, tmp
    raise ApiError("validation", f"unknown format {fmt!r}",
                   detail={"errors": [{"path": "fmt", "message": "json, csv, geojson or gpkg", "code": "fmt"}]})
