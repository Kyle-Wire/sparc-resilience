"""sparc.server.routes.physics — /physics/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.

Currently migrated:
  - /physics/defaults
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["physics"])


@router.get("/physics/defaults")
async def physics_defaults():
    """Return default literature-based physics priors."""
    try:
        from sparc.interventions.physics_priors import PhysicsPriors
        pp = PhysicsPriors()
        return {k: c.to_dict() for k, c in pp.coefficients.items()}
    except Exception as exc:
        raise HTTPException(500, str(exc))
