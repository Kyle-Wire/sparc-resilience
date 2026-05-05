"""Phase C-4 + anisotropic-frame metadata — scenario routing audit.

Produces a JSON-friendly summary of *how* the scenario simulator routes
its decision-grade per-cell effect estimates.  Two questions are answered:

1. **Causal vs correlational routing** — for each scenario variable,
   does the simulator use a Stage-3 ``SpatialCATE`` per-cell multiplier
   (causal) or fall back to the heuristic disagreement-based multiplier
   (correlational)?  This is C-4: turn the existing CATE-multiplier
   preference logic into a first-class artifact so the report and the
   desktop can surface it.

2. **Anisotropic-frame metadata** — when a ``KernelField`` is attached
   to the simulator (or any caller), summarise per-predictor anisotropy
   (κ_x, κ_y, θ, eccentricity, bandwidth) so downstream visualisers
   can draw the correct ellipses + principal-axis arrows.

Persist as ``stage="4", artifact_id="scenario_routing_audit"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Anisotropic frame metadata
# ---------------------------------------------------------------------------


def kernel_field_anisotropy_metadata(kernel_field: Any) -> List[Dict[str, Any]]:
    """Per-predictor anisotropic-frame summary suitable for a desktop view.

    Returns a list of dicts (one per predictor) carrying the geometric
    parameters needed to draw an anisotropy ellipse:

    - ``name``: predictor name
    - ``is_anisotropic``: True when κ_x ≠ κ_y by > 1e-9
    - ``kappa_x, kappa_y``: per-axis decay rates (1/length)
    - ``theta_rad, theta_deg``: principal-axis orientation
    - ``range_x, range_y``: 1/κ along each axis (length units)
    - ``eccentricity``: sqrt(1 − (min/max)²) of the ellipse
    - ``aspect_ratio``: max(κ)/min(κ)
    - ``bandwidth_to_outcome``: per-pair effective range
    - ``nu``, ``sigma2``
    """
    if kernel_field is None or not hasattr(kernel_field, "predictors"):
        return []
    out: List[Dict[str, Any]] = []
    for p in kernel_field.predictors:
        kx, ky = p.kappa_x, p.kappa_y
        is_aniso = bool(getattr(p, "is_anisotropic", False))
        record: Dict[str, Any] = {
            "name": str(p.name),
            "is_anisotropic": is_aniso,
            "kappa": (None if p.kappa is None else float(p.kappa)),
            "kappa_x": (None if kx is None else float(kx)),
            "kappa_y": (None if ky is None else float(ky)),
            "theta_rad": (None if p.theta_rad is None else float(p.theta_rad)),
            "theta_deg": (
                None if p.theta_rad is None
                else float(math.degrees(float(p.theta_rad)))
            ),
            "range_x": (
                None if kx is None or float(kx) <= 0 else 1.0 / float(kx)
            ),
            "range_y": (
                None if ky is None or float(ky) <= 0 else 1.0 / float(ky)
            ),
            "nu": (None if p.nu is None else float(p.nu)),
            "sigma2": (None if p.sigma2 is None else float(p.sigma2)),
            "bandwidth_to_outcome": (
                None if p.bandwidth_to_outcome is None
                else float(p.bandwidth_to_outcome)
            ),
        }
        if is_aniso and kx and ky:
            kmax, kmin = max(float(kx), float(ky)), min(float(kx), float(ky))
            record["aspect_ratio"] = float(kmax / kmin)
            ratio2 = (kmin / kmax) ** 2
            record["eccentricity"] = float(math.sqrt(max(0.0, 1.0 - ratio2)))
        else:
            record["aspect_ratio"] = 1.0
            record["eccentricity"] = 0.0
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Routing audit
# ---------------------------------------------------------------------------


@dataclass
class ScenarioRoutingAudit:
    variables: List[str]
    routing: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anisotropy: List[Dict[str, Any]] = field(default_factory=list)
    kernel_field_outcome: Optional[str] = None

    def to_payload(self) -> dict:
        return {
            "variables": list(self.variables),
            "routing": {k: dict(v) for k, v in self.routing.items()},
            "anisotropy": list(self.anisotropy),
            "kernel_field_outcome": self.kernel_field_outcome,
            "summary": {
                "total_variables": len(self.variables),
                "causal_routed": sum(
                    1 for v in self.routing.values()
                    if v.get("source") == "causal_cate"
                ),
                "correlational_routed": sum(
                    1 for v in self.routing.values()
                    if v.get("source") == "heuristic_disagreement"
                ),
                "missing": sum(
                    1 for v in self.routing.values()
                    if v.get("source") == "missing"
                ),
                "anisotropic_predictor_count": sum(
                    1 for r in self.anisotropy if r.get("is_anisotropic")
                ),
            },
        }


def build_scenario_routing_audit(
    *,
    scenario_variables: Sequence[str],
    cate_multipliers: Optional[Dict[str, np.ndarray]] = None,
    n_cells: Optional[int] = None,
    kernel_field: Any = None,
) -> ScenarioRoutingAudit:
    """Assemble the routing audit payload for the scenario simulator.

    Parameters
    ----------
    scenario_variables : list of variable names that the simulator will
        actually intervene on.
    cate_multipliers : per-variable CATE multiplier arrays, exactly as
        ``ScenarioSimulator._load_cate_multipliers`` returns them.
    n_cells : expected per-cell array length.  When provided, multipliers
        whose length does not match are routed as ``"length_mismatch"``
        instead of ``"causal_cate"``.
    kernel_field : optional ``KernelField`` to summarise.
    """
    cate_multipliers = cate_multipliers or {}
    routing: Dict[str, Dict[str, Any]] = {}
    for var in scenario_variables:
        if var in cate_multipliers:
            arr = np.asarray(cate_multipliers[var])
            if n_cells is not None and arr.size != n_cells:
                routing[var] = {
                    "source": "length_mismatch",
                    "n_cells_expected": int(n_cells),
                    "n_cells_actual": int(arr.size),
                }
            else:
                routing[var] = {
                    "source": "causal_cate",
                    "n_cells": int(arr.size),
                    "mean_multiplier": float(np.nanmean(arr)),
                    "std_multiplier": float(np.nanstd(arr)),
                    "min_multiplier": float(np.nanmin(arr)),
                    "max_multiplier": float(np.nanmax(arr)),
                }
        else:
            routing[var] = {"source": "heuristic_disagreement"}

    audit = ScenarioRoutingAudit(
        variables=list(scenario_variables),
        routing=routing,
        anisotropy=kernel_field_anisotropy_metadata(kernel_field),
        kernel_field_outcome=(
            getattr(kernel_field, "outcome_name", None)
            if kernel_field is not None else None
        ),
    )
    return audit


def persist_scenario_routing_audit(
    audit: ScenarioRoutingAudit,
    *,
    producer: str = "stage4_scenario_simulator",
) -> Optional[dict]:
    """Persist via the active artifact store, if available."""
    payload = audit.to_payload()
    try:
        from sparc.registry.store import get_active_store
        store = get_active_store()
        if store is None:
            return None
        store.write_struct(
            "4", "scenario_routing_audit", payload,
            producer=producer,
            consumers=[
                "report:/stage4/routing_audit",
                "desktop:scenario_routing",
                "desktop:kernel_field_view",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        payload["persist_error"] = str(exc)
    return payload
