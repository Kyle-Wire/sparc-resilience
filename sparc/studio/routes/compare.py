"""Compare runs (api.md §6.5): ``GET /compare/runs``, ``GET /compare/layer.bin`` (b − a) and
``POST /compare/priority`` (Kendall τ and top-decile Jaccard of a priority layer)."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

from fastapi import APIRouter, Depends, Query, Request

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.routes.layers import binary
from sparc.studio.routes.runs import run_context, services
from sparc.studio.runs import compare as C
from sparc.studio.runs.schemas import CompareRuns, PriorityRequest, PriorityResponse

router = APIRouter(tags=["compare"])


@router.get("/compare/runs", response_model=CompareRuns)
async def get_compare(a: str = Query(...), b: str = Query(...), sctx: StudioContext = Depends(get_ctx)):
    _, reg = services(sctx)
    ca, cb = run_context(sctx, a), run_context(sctx, b)
    return await asyncio.to_thread(C.compare_runs, ca, cb, reg.summary(ca.row), reg.summary(cb.row))


@router.get("/compare/layer.bin")
def get_compare_layer(request: Request, a: str = Query(...), b: str = Query(...), key: str = Query(...),
                      sctx: StudioContext = Depends(get_ctx)):
    from sparc.studio.runs.layers import layer_etag

    ca, cb = run_context(sctx, a), run_context(sctx, b)
    arr = C.layer_diff(ca, cb, key)
    etag = '"' + hashlib.sha1(f"{layer_etag(ca, key)}|{layer_etag(cb, key)}".encode()).hexdigest() + '"'
    both = SimpleNamespace(finished=ca.finished and cb.finished)      # immutable only when both runs are finished
    return binary(request, both, arr.astype("<f4").tobytes(), dtype="float32", length=int(arr.size), etag=etag)


@router.post("/compare/priority", response_model=PriorityResponse)
async def post_priority(body: PriorityRequest, sctx: StudioContext = Depends(get_ctx)):
    ca, cb = run_context(sctx, body.a), run_context(sctx, body.b)
    return await asyncio.to_thread(C.priority_agreement, ca, cb, body.layer)
