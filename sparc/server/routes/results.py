"""sparc.server.routes.results — /results/* read endpoints.

This module contains the extracted read-only result endpoints previously
defined in ``app.py``.  Each route imports shared state and helpers from
:mod:`sparc.server.deps` rather than closing over module-level names in
``app.py``.

Migration status
----------------
* ``router`` is registered on the FastAPI app via ``include_router``.
* Routes are migrated incrementally; while in-progress the routes still
  live in ``app.py`` — they will be removed from there and added here
  as each one is fully tested.

Currently migrated:
  - /results/correlogram  (read-only, db-only)

Pending migration (still in app.py):
  - /results/kernel_field
  - /results/causal
  - /results/causal/pdp_curves
  - /results/causal/divergence
  - /results/gwen
  - /results/model_performance
  - /results/spatial_cv/predictions
  - /results/causal/dose_response
  - /results/causal/sensitivity
  - /results/causal/negative_control
  - /results/causal/diagnostics
  - /results/neural_pdp
  - /results/pdp_curves
  - /results/scenarios/*
  - /results/report
  - /results/dataset/profile
"""

from __future__ import annotations

from fastapi import APIRouter

from sparc.server.deps import read_or_404

router = APIRouter(tags=["results"])


@router.get("/results/correlogram")
async def results_correlogram():
    """Return the Stage 0 correlogram analysis payload."""
    return await read_or_404("0", "correlogram_results", hint="Run Stage 0 first")
