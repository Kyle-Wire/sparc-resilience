"""Grid, layers, cells, selections, regions, blobs, analysis tools and layer exports (api.md §6.2–6.4).

Binary arrays are little-endian with ``X-SPARC-Dtype`` / ``X-SPARC-Length`` (packed bodies add
``X-SPARC-Offsets``).  Finished runs send a strong ``ETag`` plus ``Cache-Control: private,
max-age=31536000, immutable`` and answer a matching ``If-None-Match`` with ``304``; running runs send
``no-store``.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError
from sparc.studio.routes.runs import run_context
from sparc.studio.runs import layers as L
from sparc.studio.runs import selection as S
from sparc.studio.runs import stats as T
from sparc.studio.runs.common import clean, fnum
from sparc.studio.runs.schemas import (
    AcfRequest,
    AcfResponse,
    BlobOut,
    BreakdownRequest,
    BreakdownResponse,
    CellInfo,
    HexbinRequest,
    HexbinResponse,
    HexResponse,
    LayersResponse,
    RegionCreate,
    RegionOut,
    RegionStats,
    RegionStatsRequest,
    ResolveRequest,
    ResolveResponse,
)
from sparc.studio.schemas.common import GridMeta, Ok
from sparc.studio.workspace import new_id, utc_now

router = APIRouter(tags=["layers"])

IMMUTABLE = "private, max-age=31536000, immutable"
BLOB_MAX_BYTES = 64 * 1024 ** 2


def _grid_or_404(ctx):
    g = ctx.grid
    if g is None:
        raise ApiError("output_missing", "the run's grid is not known yet (S0 has not written its points)",
                       detail={"output": "grid", "produced_by": "stage:S0", "expected_path": "predictions.parquet"})
    return g


def binary(request: Request, ctx, body: bytes, *, dtype: str | None, length: int | None, etag: str,
           extra: dict | None = None) -> Response:
    """A binary response with the api.md §0.4–0.5 headers (304 on a matching ``If-None-Match``)."""
    headers = {"ETag": etag}
    if dtype:
        headers["X-SPARC-Dtype"] = dtype
    if length is not None:
        headers["X-SPARC-Length"] = str(length)
    headers.update(extra or {})
    headers["Cache-Control"] = IMMUTABLE if ctx.finished else "no-store"
    inm = request.headers.get("if-none-match")
    if ctx.finished and inm and etag in [t.strip() for t in inm.split(",")]:
        return Response(status_code=304, headers={k: v for k, v in headers.items()
                                                  if k in ("ETag", "Cache-Control")})
    return Response(body, media_type="application/octet-stream", headers=headers)


def _source(sctx, ctx) -> S.RunSource:
    return S.RunSource(ctx, sctx.db)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def _grid_meta(ctx) -> dict:
    from sparc.studio.runs.grid import grid_meta

    g = _grid_or_404(ctx)
    cv = ctx.cv_meta
    qa = (ctx.manifest_raw or {}).get("qa") or {}
    bg = fnum(qa.get("background"))
    if bg is None and ctx.data is not None:
        bg = float(ctx.data.background)
    return grid_meta(ctx.run_id, g, n_folds=cv["n_folds"] if cv else None, target_units=ctx.target_units,
                     background=bg)


@router.get("/runs/{rid}/grid", response_model=GridMeta)
def get_grid(rid: str, sctx: StudioContext = Depends(get_ctx)):
    return clean(_grid_meta(run_context(sctx, rid)))


@router.get("/runs/{rid}/grid.bin")
def get_grid_bin(rid: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.grid import grid_bin, grid_etag

    ctx = run_context(sctx, rid)
    g = _grid_or_404(ctx)
    body, offsets = grid_bin(g)
    return binary(request, ctx, body, dtype=None, length=g.n, etag=grid_etag(rid, g),
                  extra={"X-SPARC-Offsets": json.dumps(offsets, separators=(",", ":"))})


@router.get("/runs/{rid}/grid/ids.bin")
def get_ids_bin(rid: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.grid import grid_etag

    ctx = run_context(sctx, rid)
    g = _grid_or_404(ctx)
    if g.ids_kind != "int":
        raise ApiError("string_ids", "this run's ids are strings: use grid/ids.json")
    body = np.ascontiguousarray(g.ids.astype("<i8")).tobytes()
    return binary(request, ctx, body, dtype="int64", length=g.n, etag=grid_etag(rid, g).rstrip('"') + ':ids"')


@router.get("/runs/{rid}/grid/ids.json")
def get_ids_json(rid: str, sctx: StudioContext = Depends(get_ctx)) -> list:
    g = _grid_or_404(run_context(sctx, rid))
    return [str(x) for x in g.ids] if g.ids_kind == "str" else [int(x) for x in g.ids]


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

@router.get("/runs/{rid}/layers", response_model=LayersResponse)
async def get_layers(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    _grid_or_404(ctx)
    return await asyncio.to_thread(L.layer_catalog, ctx)


@router.get("/runs/{rid}/layers/{key}.bin")
def get_layer_bin(rid: str, key: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    _grid_or_404(ctx)
    etag = L.layer_etag(ctx, key)
    if ctx.finished and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": IMMUTABLE})
    arr = L.layer_array(ctx, key)
    dtype = "uint8" if arr.dtype == np.uint8 else "float32"
    body = arr.astype("<f4" if dtype == "float32" else np.uint8, copy=False).tobytes()
    return binary(request, ctx, body, dtype=dtype, length=int(arr.size), etag=etag)


@router.get("/runs/{rid}/folds/{k}.bin")
def get_fold_bin(rid: str, k: str, request: Request, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    try:
        kk = int(k)
    except ValueError:
        raise ApiError("not_found", f"no fold {k!r}")
    arr = L.fold_classes(ctx, kk)
    cv = ctx.cv_meta or {}
    etag = '"' + f"folds:{rid}:{kk}:{cv.get('n_folds')}:{cv.get('block_m')}:{cv.get('seed')}:{ctx.n}" + '"'
    return binary(request, ctx, arr.tobytes(), dtype="uint8", length=int(arr.size), etag=etag)


# ---------------------------------------------------------------------------
# cells and hexagons
# ---------------------------------------------------------------------------

def _curves(ctx, i: int) -> dict:
    """Per-lever dose-response curve of one cell, rebuilt from its fitted parameters (SPEC §6.4 Response)."""
    curves = {}
    rc = ctx.json("response_curves.json") or {}
    for var in ctx.response_vars():
        df = ctx.parquet(f"response_{var}.parquet")
        if df is None or i >= len(df):
            continue
        r = df.iloc[i]
        model = str(r.get("curve_model")) if "curve_model" in df.columns else "unknown"
        A = fnum(r.get("max_cooling_A"))
        ds = fnum(r.get("saturation_scale_ds"))
        infl = fnum(r.get("inflection_dose"))
        slope = fnum(r.get("marginal_benefit_per_unit"))
        doses = [float(d) for d in ((rc.get(var) or {}).get("curve") or {}).get("dose") or []]
        if not doses:
            spec = (ctx.cfg_raw.get("actionable") or {}).get(var) or {}
            doses = [float(d) for d in spec.get("doses") or [0, 5, 10, 20, 30]]
        dmax = max(doses) if doses else None
        grid = list(np.linspace(0.0, dmax, 25)) if dmax else []
        benefit = _curve_values(model, A, ds, infl, slope, grid, dmax, bool(r.get("censored")))
        curves[var] = {"model": model, "A": A, "ds": ds, "inflection": infl, "d90": fnum(r.get("d90")), "dmax": dmax,
                       "dose": [float(x) for x in grid] if benefit else [], "benefit": benefit}
    return curves


def _curve_values(model: str, A, ds, infl, slope, dose: list[float], dmax, censored: bool) -> list[float]:
    """Benefit at each neighbourhood dose for the fitted shape (``sparc.core.response.fit_saturation``)."""
    d = np.asarray(dose, dtype=float)
    if model == "linear" and slope is not None:
        return [float(x) for x in slope * d]
    if model == "saturating" and A is not None:
        scale = ds
        if scale is None and slope:                       # censored: d_s = A / (marginal benefit at 0)
            scale = A / slope if slope else None
        if scale and scale > 0:
            return [float(x) for x in A * (1.0 - np.exp(-d / scale))]
        return []
    if model == "sigmoid" and A is not None and infl is not None and dmax:
        w = _sigmoid_width(A, infl, slope, dmax)
        if w is None:
            return []

        def sig(z):
            return 1.0 / (1.0 + np.exp(-z))

        s0 = sig(-infl / w)
        return [float(x) for x in A * (sig((d - infl) / w) - s0) / (1.0 - s0)]
    return []


def _sigmoid_width(A: float, d0: float, slope, top: float) -> float | None:
    """The width ``w`` of a fitted S-curve, which the response parquet does not store.

    ``fit_saturation`` searches ``D0 = c·top`` (c in linspace(0.15, 0.85, 8)) and ``w = c'·top``
    (c' in 0.04, 0.08, 0.15), so ``w / D0`` is one of 24 ratios whatever ``top`` was; the stored slope at
    0 (``A·σ(−D0/w)/w``) picks the one that was fitted."""
    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    ratios = np.array([cw / cd for cd in np.linspace(0.15, 0.85, 8) for cw in (0.04, 0.08, 0.15)])
    cands = np.concatenate([d0 * ratios, np.array([0.04, 0.08, 0.15]) * top]) if d0 > 0 else \
        np.array([0.04, 0.08, 0.15]) * top
    cands = cands[cands > 0]
    if cands.size == 0:
        return None
    if slope is None or not np.isfinite(slope) or slope == 0:
        return float(0.08 * top) if top else None
    with np.errstate(over="ignore"):
        fit = np.abs(A * sig(-d0 / cands) / cands - slope) / abs(slope)
    return float(cands[int(np.argmin(fit))])


@router.get("/runs/{rid}/cells/{index}", response_model=CellInfo)
def get_cell(rid: str, index: int, scenarios: str | None = None, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    g = _grid_or_404(ctx)
    if index < 0 or index >= g.n:
        raise ApiError("not_found", f"row {index} is outside the run (n = {g.n})")
    values = {}
    for key, d in L.layer_defs(ctx).items():
        try:
            v = L.layer_array(ctx, key)[index]
        except ApiError:
            continue
        values[key] = None if (d.dtype == "uint8" and int(v) == 255) else fnum(float(v))
    scen: dict[str, float | None] = {}
    for s in ctx.configured_scenarios():
        k = f"sc:{s['slug']}"
        if k in values:
            scen[f"configured:{s['slug']}"] = values[k]
    for ref in [r for r in (scenarios or "").split(",") if r]:
        if ref in scen:
            continue
        try:
            delta, _f, _l = T.scenario_arrays(ctx, ref)
            scen[ref] = fnum(delta[index])
        except ApiError:
            scen[ref] = None
    zone = g.zones[g.zone[index]] if g.zones and g.zone[index] >= 0 else None
    return clean({"index": index, "id": g.ids[index], "lon": fnum(g.lon[index]), "lat": fnum(g.lat[index]),
                  "zone": zone, "values": values, "curves": _curves(ctx, index), "scenarios": scen})


def _cleanup(tmp: Path | None) -> BackgroundTask | None:
    return BackgroundTask(shutil.rmtree, tmp, True) if tmp is not None else None


@router.get("/runs/{rid}/hex", response_model=HexResponse)
def get_hex(rid: str, size: int = Query(250), layers: str = "", sums: str | None = None,
            fmt: str = Query("json"), sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.export import export_hex

    if size not in (250, 500):
        raise ApiError("validation", "size must be 250 or 500",
                       detail={"errors": [{"path": "size", "message": "250 or 500", "code": "size"}]})
    ctx = run_context(sctx, rid)
    src = _source(sctx, ctx)
    keys = [k for k in layers.split(",") if k]
    sm = [k for k in (sums or "").split(",") if k]
    for k in sm:
        if k not in keys:
            keys.append(k)
    df = T.hex_table(ctx, src, float(size), keys, sm)
    if fmt == "json":
        rows = []
        for rec in df.to_dict(orient="records"):
            rows.append({"key": int(rec["key"]), "cx": float(rec["cx"]), "cy": float(rec["cy"]),
                         "lon": fnum(rec["lon"]), "lat": fnum(rec["lat"]), "n_cells": int(rec["n_cells"]),
                         "values": {k: fnum(rec[k]) for k in keys}})
        return {"hex": rows}
    if fmt not in ("csv", "geojson", "gpkg"):
        raise ApiError("validation", "fmt must be json, csv, geojson or gpkg",
                       detail={"errors": [{"path": "fmt", "message": "unknown format", "code": "fmt"}]})
    body, media, name, tmp = export_hex(ctx, df, float(size), fmt)
    headers = {"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"}
    if isinstance(body, Path):
        return FileResponse(body, media_type=media, filename=name, headers=headers, background=_cleanup(tmp))
    return Response(body, media_type=media, headers=headers)


# ---------------------------------------------------------------------------
# selections, regions, blobs
# ---------------------------------------------------------------------------

@router.post("/runs/{rid}/selection/resolve", response_model=ResolveResponse)
async def post_resolve(rid: str, body: ResolveRequest, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)

    def work():
        from threadpoolctl import threadpool_limits

        with threadpool_limits(1):
            src = _source(sctx, ctx)
            spec = S.spec_dict(body.selection)
            mask = S.resolve(src, spec)
            summ = S.summarize(ctx, src, mask)
            warnings = []
            if not mask.any():
                warnings.append("the selection is empty on this run")
            portable = S.is_portable(spec, sctx.db)
            if not portable:
                warnings.append("this selection only applies to this run (run-frame geometry, a brushed blob or a "
                                "result column)")
            return clean({**summ, "mask": S.encode_bitset(mask), "portable": portable, "warnings": warnings})

    return await asyncio.to_thread(work)


def _region_rows(sctx: StudioContext, ctx) -> list[dict]:
    if ctx.project_id:
        return sctx.db.fetchall("SELECT * FROM regions WHERE project_id = ? ORDER BY created_utc", (ctx.project_id,))
    return sctx.db.fetchall("SELECT * FROM regions WHERE run_id = ? ORDER BY created_utc", (ctx.run_id,))


@router.get("/runs/{rid}/regions", response_model=list[RegionOut])
def get_regions(rid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    src = _source(sctx, ctx) if ctx.grid is not None else None
    out = []
    for r in _region_rows(sctx, ctx):
        spec = json.loads(r["spec_json"])
        n = r.get("n_cells") or 0
        if src is not None and r.get("run_id") != rid:
            try:
                n = int(S.resolve(src, spec).sum())
            except ApiError:
                n = 0
        out.append({"id": r["id"], "name": r["name"], "spec": spec, "n_cells": int(n), "created_utc": r["created_utc"],
                    "portable": S.is_portable(spec, sctx.db)})
    return out


@router.post("/runs/{rid}/regions", status_code=201, response_model=RegionOut)
def post_region(rid: str, body: RegionCreate, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    spec = S.spec_dict(body.spec)
    mask = S.resolve(_source(sctx, ctx), spec)
    rgid = new_id("rg")
    row = {"id": rgid, "project_id": ctx.project_id, "run_id": rid, "name": body.name,
           "spec_json": json.dumps(spec, separators=(",", ":")), "n_cells": int(mask.sum()), "created_utc": utc_now()}
    sctx.db.insert("regions", row)
    return {"id": rgid, "name": body.name, "spec": spec, "n_cells": int(mask.sum()), "created_utc": row["created_utc"],
            "portable": S.is_portable(spec, sctx.db)}


@router.delete("/runs/{rid}/regions/{rgid}", response_model=Ok)
def delete_region(rid: str, rgid: str, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    row = sctx.db.fetchone("SELECT project_id, run_id FROM regions WHERE id = ?", (rgid,))
    if row is None or (row.get("project_id") != ctx.project_id and row.get("run_id") != rid):
        raise ApiError("not_found", f"no region {rgid!r} on this run")
    sctx.db.execute("DELETE FROM regions WHERE id = ?", (rgid,))
    return {"ok": True}


@router.put("/runs/{rid}/blobs", status_code=201, response_model=BlobOut)
async def put_blob(rid: str, request: Request, kind: str = Query(...), sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    g = _grid_or_404(ctx)
    if kind not in ("mask", "edit"):
        raise ApiError("validation", "kind must be mask or edit",
                       detail={"errors": [{"path": "kind", "message": "mask or edit", "code": "kind"}]})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > BLOB_MAX_BYTES:
        raise ApiError("too_large", "blob too large", detail={"max_bytes": BLOB_MAX_BYTES})
    chunks, total = [], 0
    async for part in request.stream():
        total += len(part)
        if total > BLOB_MAX_BYTES:
            raise ApiError("too_large", "blob too large", detail={"max_bytes": BLOB_MAX_BYTES})
        chunks.append(part)
    raw = b"".join(chunks)
    if kind == "mask":
        S.mask_from_bytes(raw, g.n)                      # validates the length
    else:
        cnt = request.headers.get("x-sparc-count")
        if cnt is None or not cnt.isdigit():
            raise ApiError("validation", "edit blobs need X-SPARC-Count",
                           detail={"errors": [{"path": "X-SPARC-Count", "message": "required", "code": "count"}]})
        m = int(cnt)
        if len(raw) != 8 * m:
            raise ApiError("validation", f"an edit blob of {m} rows has {8 * m} bytes, not {len(raw)}",
                           detail={"errors": [{"path": "body", "message": "wrong length", "code": "length"}]})
        idx = np.frombuffer(raw[:4 * m], dtype="<i4")
        if m and (idx.min() < 0 or idx.max() >= g.n):
            raise ApiError("validation", "edit indices outside the run's rows",
                           detail={"errors": [{"path": "body", "message": "index out of range", "code": "index"}]})
    bid = new_id("bl")
    path = ctx.studio_dir / "blobs" / f"{bid}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    await sctx.db.ainsert("blobs", {"id": bid, "run_id": rid, "kind": kind, "path": str(path), "bytes": len(raw),
                                    "created_utc": utc_now()})
    return {"blob_id": bid, "bytes": len(raw)}


# ---------------------------------------------------------------------------
# analysis tools
# ---------------------------------------------------------------------------

def _limited(fn):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        return fn()


@router.post("/runs/{rid}/stats/region", response_model=RegionStats)
async def post_region_stats(rid: str, body: RegionStatsRequest, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)

    def work():
        src = _source(sctx, ctx)
        mask = S.resolve(src, S.spec_dict(body.selection))
        return T.region_stats(ctx, src, mask, body.layers, body.weights, body.scenarios)

    return await asyncio.to_thread(_limited, work)


@router.post("/runs/{rid}/stats/breakdown", response_model=BreakdownResponse)
async def post_breakdown(rid: str, body: BreakdownRequest, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    return await asyncio.to_thread(_limited, lambda: T.breakdown(ctx, _source(sctx, ctx), body.value,
                                                                 body.by.model_dump(), body.weights, body.stat))


@router.post("/runs/{rid}/stats/hexbin", response_model=HexbinResponse)
async def post_hexbin(rid: str, body: HexbinRequest, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)

    def work():
        src = _source(sctx, ctx)
        mask = S.resolve(src, S.spec_dict(body.selection)) if body.selection is not None else None
        return T.hexbin(ctx, src, body.x, body.y, body.bins, mask)

    return await asyncio.to_thread(_limited, work)


@router.post("/runs/{rid}/stats/acf", response_model=AcfResponse)
async def post_acf(rid: str, body: AcfRequest, sctx: StudioContext = Depends(get_ctx)):
    ctx = run_context(sctx, rid)
    return await asyncio.to_thread(_limited, lambda: T.acf(ctx, _source(sctx, ctx), body.layer, body.max_lag_m,
                                                           body.n_perm))


@router.get("/runs/{rid}/export/layer/{key}")
async def get_export_layer(rid: str, key: str, fmt: str = Query(...), sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.export import export_layer

    if fmt not in ("tif", "csv", "geojson", "parquet"):
        raise ApiError("validation", "fmt must be tif, csv, geojson or parquet",
                       detail={"errors": [{"path": "fmt", "message": "unknown format", "code": "fmt"}]})
    ctx = run_context(sctx, rid)
    _grid_or_404(ctx)
    body, media, name, tmp = await asyncio.to_thread(_limited, lambda: export_layer(ctx, key, fmt))
    headers = {"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"}
    if isinstance(body, Path):
        return FileResponse(body, media_type=media, filename=name, headers=headers, background=_cleanup(tmp))
    return Response(body, media_type=media, headers=headers)
