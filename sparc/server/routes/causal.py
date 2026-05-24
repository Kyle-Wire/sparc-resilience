"""sparc.server.routes.causal — /results/causal/* read endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.

Currently migrated:
  - /results/causal/pdp_curves
  - /results/causal/divergence

Pending migration (still in app.py):
  - /results/causal
  - /results/causal/dose_response
  - /results/causal/sensitivity
  - /results/causal/negative_control
  - /results/causal/diagnostics
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from sparc.registry.store import ArtifactStore
from sparc.server.deps import get_open_store, read_or_404

router = APIRouter(tags=["causal"])


@router.get("/results/causal/pdp_curves")
async def results_causal_pdp_curves(
    store: Annotated[ArtifactStore, Depends(get_open_store)],
):
    """Return the Bayesian causal PDP curves (Stage 3, Phase C-2)."""
    return await read_or_404(
        store, "3", "causal_pdp_curves",
        hint="Stage 3 has not produced causal_pdp_curves. Run Stage 3 with Bayesian-CATE enabled.",
    )


@router.get("/results/causal/divergence")
async def results_causal_divergence(
    store: Annotated[ArtifactStore, Depends(get_open_store)],
):
    """Return the CATE-vs-GWR divergence audit (Stage 3, Phase C-3)."""
    return await read_or_404(
        store, "3", "cate_vs_gwr_divergence",
        hint="Stage 3 has not produced cate_vs_gwr_divergence.",
    )
