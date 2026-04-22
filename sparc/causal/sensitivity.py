"""Causal robustness sensitivity utilities.

E-values (VanderWeele & Ding 2017) measure the minimum strength of
association — on the risk-ratio scale — that an unmeasured confounder
would need to have with both the treatment *and* the outcome (above and
beyond the measured confounders) to fully explain away an observed
exposure–outcome association.

For a continuous (standardized) effect ``d`` we use VanderWeele's
recommended approximation::

    RR  ≈  exp(0.91 · d)

then::

    E-value(RR)  =  RR + sqrt(RR · (RR − 1))     for RR ≥ 1
    E-value(RR)  =  1/RR + sqrt((1/RR) · (1/RR − 1))   for RR < 1

The same formula applied to the lower CI bound gives the **tipping
point** — how strong a confounder must be to push the CI through 1
(i.e. nullify the effect).

This module is intentionally pure-Python and stateless so the FastAPI
layer can apply it to any causal payload without re-running stage 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


_EVAL_RR_COEFF = 0.91  # VanderWeele continuous-outcome conversion


@dataclass
class SensitivityResult:
    effect_label: str
    point_estimate: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    rr_equivalent: float
    e_value_point: float
    e_value_ci: Optional[float]   # tipping-point: confounder strength to push CI past 1
    interpretation: str


def _rr_from_continuous(d: float) -> float:
    """VanderWeele's exp(0.91 d) approximation."""
    return math.exp(_EVAL_RR_COEFF * float(d))


def _e_value_from_rr(rr: float) -> float:
    """E-value formula. Returns 1.0 for RR == 1 (no association)."""
    if not math.isfinite(rr) or rr <= 0:
        return float("nan")
    if abs(rr - 1.0) < 1e-12:
        return 1.0
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def _interpret(e_point: float, e_ci: Optional[float]) -> str:
    """Return a one-line plain-English interpretation."""
    if e_ci is not None and e_ci <= 1.05:
        return "Fragile: the CI already crosses null; unmeasured confounding could explain it."
    if e_point < 1.5:
        return "Weak: a modest unmeasured confounder could explain the effect."
    if e_point < 2.5:
        return "Moderate: confounder must have non-trivial association with both treatment and outcome."
    if e_point < 4.0:
        return "Strong: only a substantial unmeasured confounder could nullify the effect."
    return "Very strong: implausibly large unmeasured confounding required."


def e_value_continuous(
    effect: float,
    *,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    label: str = "effect",
) -> SensitivityResult:
    """Compute the E-value for a continuous (standardized) effect.

    Pass ``ci_lower``/``ci_upper`` to also compute the tipping-point
    E-value at the CI bound nearest the null.
    """
    rr = _rr_from_continuous(effect)
    e_point = _e_value_from_rr(rr)

    e_ci: Optional[float] = None
    if ci_lower is not None and ci_upper is not None:
        # Pick the CI bound nearest to null (1 on RR scale, 0 on continuous).
        if effect >= 0:
            bound = ci_lower
        else:
            bound = ci_upper
        # If the CI already crosses 0 the bound is on the wrong side ⇒ tipping = 1.
        if (effect >= 0 and bound <= 0) or (effect < 0 and bound >= 0):
            e_ci = 1.0
        else:
            e_ci = _e_value_from_rr(_rr_from_continuous(bound))

    return SensitivityResult(
        effect_label=label,
        point_estimate=float(effect),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        rr_equivalent=rr,
        e_value_point=e_point,
        e_value_ci=e_ci,
        interpretation=_interpret(e_point, e_ci),
    )


def annotate_causal_payload(payload: dict) -> dict:
    """Walk a causal-results dict and attach per-effect E-values.

    Recognises common shapes::

        {"effects": [{"name": ..., "estimate": ..., "ci_low": ..., "ci_high": ...}, ...]}
        {"direct_effect": x, "indirect_effect": y, "total_effect": z, ...}

    Adds a ``sensitivity`` block at the top level::

        "sensitivity": {
            "method": "VanderWeele E-value (continuous, RR≈exp(0.91·d))",
            "results": [SensitivityResult-as-dict, ...],
        }
    """
    if not isinstance(payload, dict):
        return payload

    results: list[dict] = []

    # Shape A: explicit list of effects.
    effects = payload.get("effects")
    if isinstance(effects, list):
        for e in effects:
            if not isinstance(e, dict):
                continue
            est = e.get("estimate") or e.get("effect") or e.get("mean")
            if est is None:
                continue
            sr = e_value_continuous(
                float(est),
                ci_lower=_safe_float(e.get("ci_low") or e.get("ci_lower")),
                ci_upper=_safe_float(e.get("ci_high") or e.get("ci_upper")),
                label=str(e.get("name") or e.get("label") or "effect"),
            )
            results.append(_sr_to_dict(sr))

    # Shape B: scalar direct/indirect/total fields.
    for key in ("direct_effect", "indirect_effect", "total_effect"):
        v = payload.get(key)
        if isinstance(v, (int, float)):
            sr = e_value_continuous(float(v), label=key)
            results.append(_sr_to_dict(sr))
        elif isinstance(v, dict):
            est = v.get("mean") or v.get("estimate")
            if est is not None:
                sr = e_value_continuous(
                    float(est),
                    ci_lower=_safe_float(v.get("ci_low") or v.get("ci_lower")),
                    ci_upper=_safe_float(v.get("ci_high") or v.get("ci_upper")),
                    label=key,
                )
                results.append(_sr_to_dict(sr))

    payload = {**payload, "sensitivity": {
        "method": "VanderWeele E-value (continuous, RR≈exp(0.91·d))",
        "results": results,
    }}
    return payload


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _sr_to_dict(sr: SensitivityResult) -> dict:
    return {
        "effect_label": sr.effect_label,
        "point_estimate": sr.point_estimate,
        "ci_lower": sr.ci_lower,
        "ci_upper": sr.ci_upper,
        "rr_equivalent": sr.rr_equivalent,
        "e_value_point": sr.e_value_point,
        "e_value_ci": sr.e_value_ci,
        "interpretation": sr.interpretation,
    }
