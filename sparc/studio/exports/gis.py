"""The GIS pack (``export.gis``, SPEC §6.7): every map layer of a run as a GeoTIFF, the hexagon summaries as one
GeoPackage and the planner's logger sites and before/after pairs with lon/lat, zipped.

GeoTIFFs are float32 on the run grid in the data's CRS (deflate, nodata NaN), written through
``sparc.core.planner.export_geotiffs`` (categorical layers keep their class codes, 255 → nodata).
``hexagons.gpkg`` holds one layer per hexagon size (``hex_250m``, ``hex_500m``) with the observed and
predicted temperature, the model error and - with a planner pack - the residents.  Runs without a CRS are
refused (``needs_crs``).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sparc.core import runio

__all__ = ["build_gis", "gis_layers", "HEX_SIZES"]

HEX_SIZES = (250.0, 500.0)
_HEX_COLUMNS = (("obs", "temperature"), ("pred", "predicted"), ("err", "error"), ("hw", "interval_halfwidth"))


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "layer"


def _cfg_for(ctx):
    cfg = ctx.cfg
    if cfg is not None:
        return cfg
    g = ctx.grid
    return SimpleNamespace(data={"crs": g.crs, "reproject_to": None}, coord_scale=g.coord_scale)


def gis_layers(ctx, keys: list[str] | None = None) -> list[str]:
    """The layer keys of the pack: ``keys`` (each must exist), else every layer of the run."""
    from sparc.studio.runs.layers import layer_defs

    defs = layer_defs(ctx)
    if keys:
        bad = [k for k in keys if k not in defs]
        if bad:
            raise KeyError(f"unknown layers {bad}")
        return list(keys)
    return list(defs)


def _with_lonlat(df, g, cols: list[tuple[str, str]]):
    """Add ``<prefix>lon``/``<prefix>lat`` (and run metres) for each row-index column of ``cols``."""
    for col, prefix in cols:
        if col not in df.columns:
            continue
        idx = df[col].to_numpy(np.int64)
        ok = (idx >= 0) & (idx < g.n)
        lon = np.full(idx.size, np.nan)
        lat = np.full(idx.size, np.nan)
        lon[ok] = np.asarray(g.lon, np.float64)[idx[ok]]
        lat[ok] = np.asarray(g.lat, np.float64)[idx[ok]]
        df[f"{prefix}lon"] = np.round(lon, 7)
        df[f"{prefix}lat"] = np.round(lat, 7)
        df[f"{prefix}id"] = np.asarray(g.ids)[np.where(ok, idx, 0)]
    return df


def build_gis(ctx, out_zip: Path, layers: list[str] | None = None, progress_cb=None) -> dict:
    """Write the GIS pack of the run behind ``ctx`` (a ``RunContext``) to ``out_zip``; ``{path, bytes, files}``."""
    import pandas as pd

    from sparc.core.catalog import unit_label
    from sparc.core.planner import export_geotiffs, export_hex_gpkg
    from sparc.studio.errors import ApiError
    from sparc.studio.runs.layers import layer_array, layer_defs
    from sparc.studio.runs.stats import hex_table

    g = ctx.grid
    if g is None or not g.crs:
        raise ValueError("a GIS pack needs a run with a CRS (set data.crs in the config)")
    cfg = _cfg_for(ctx)
    defs = layer_defs(ctx)
    keys = gis_layers(ctx, layers)
    top = f"{ctx.run_id}_gis"
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="sparc-gis-"))
    root = stage / top
    readme = [f"GIS pack of run {ctx.run_id}", "=" * 40, "", f"CRS {g.crs}; cells {g.dx:g} m; nodata = NaN.",
              f"Temperatures in {unit_label(ctx.target_units)}; ΔT layers: negative = cooler.", "", "GeoTIFFs",
              "--------"]
    skipped = []
    try:
        arrays: dict[str, np.ndarray] = {}
        for i, key in enumerate(keys):
            try:
                v = np.asarray(layer_array(ctx, key), dtype=np.float64)
            except ApiError as exc:          # a layer whose output is missing: listed in the README
                skipped.append(f"{key}: {exc.message}")
                continue
            d = defs[key]
            if d.dtype == "uint8":
                v = np.where(v == 255, np.nan, v)
            arrays[_safe(key)] = v
            unit = f" [{d.unit}]" if d.unit else ""
            labels = f"; classes {', '.join(f'{j}={lab}' for j, lab in enumerate(d.labels))}" if d.labels else ""
            readme.append(f"geotiff/{_safe(key)}.tif  {d.label}{unit}{labels}")
            if progress_cb is not None:
                progress_cb(i + 1, len(keys), key)
        if not arrays:
            raise ValueError("none of the requested layers could be read")
        export_geotiffs(SimpleNamespace(grid=g.core_grid()), cfg, arrays, root / "geotiff")
        # hexagons: one GeoPackage layer per size
        cols = [k for k, _ in _HEX_COLUMNS if k in defs]
        names = dict(_HEX_COLUMNS)
        sums = []
        if (ctx.run_dir / "planner" / "planner_cells.parquet").exists():
            cols.append("planner:people")
            names["planner:people"] = "people"
            sums.append("planner:people")
        gpkg = root / "hexagons.gpkg"
        for size in HEX_SIZES:
            h = hex_table(ctx, ctx, size, cols, sums)
            h = h.drop(columns=["lon", "lat"]).rename(columns={"key": "hex", **names})
            export_hex_gpkg(h, size, cfg, gpkg)
        readme += ["", "hexagons.gpkg  layers " + ", ".join(f"hex_{int(s)}m" for s in HEX_SIZES)
                   + f": mean of {', '.join(names[c] for c in cols if c not in sums)}"
                   + (", sum of people" if sums else "") + " per hexagon (n_cells cells)."]
        # logger sites and before/after pairs with lon/lat
        pdir = ctx.run_dir / "planner"
        if (pdir / "logger_sites.csv").exists():
            df = _with_lonlat(pd.read_csv(pdir / "logger_sites.csv"), g, [("cell", "")])
            df.to_csv(root / "logger_sites.csv", index=False)
            readme.append("logger_sites.csv  suggested logger sites (cell = row index; lon/lat in EPSG:4326)")
        if (pdir / "before_after_pairs.csv").exists():
            df = _with_lonlat(pd.read_csv(pdir / "before_after_pairs.csv"), g,
                              [("treated", "treated_"), ("control", "control_")])
            df.to_csv(root / "before_after_pairs.csv", index=False)
            readme.append("before_after_pairs.csv  treated/control cells with lon/lat (EPSG:4326)")
        if skipped:
            readme += ["", "Not included", "------------", *skipped]
        (root / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
        files = []
        tmp = out_zip.with_name(f".{out_zip.name}.{os.getpid()}.tmp")
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        arc = p.relative_to(stage).as_posix()
                        comp = zipfile.ZIP_STORED if p.suffix in (".tif", ".gpkg") else zipfile.ZIP_DEFLATED
                        zf.write(p, arc, compress_type=comp)
                        files.append(arc)
            runio.replace(tmp, out_zip)
        finally:
            if tmp.exists():
                tmp.unlink()
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"path": str(out_zip), "bytes": out_zip.stat().st_size, "files": files}
