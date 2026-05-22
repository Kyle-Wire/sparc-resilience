"""Physics priors registry — single seam for loading and merging physics priors.

All Stage-4 consumers (ScenarioSimulator, ScenarioEngineV4, CausalEffectResolver)
should call ``build_physics_priors_map(config)`` rather than implementing their
own load/merge logic.

The returned ``PhysicsPriorsMap`` is a plain ``Dict[str, dict]`` where each
variable entry has the schema::

    {
        "coefficient":  float,  # blended (lit × w + ols × (1-w))
        "unit_increment": float,
        "direction":    str,    # "cooling" | "warming"
        "ols_coef":     float,
        "lit_coef":     float,
        "uncertainty":  float,
    }

The raw YAML content (as returned by ``load_physics_priors``) is also available
via ``load_physics_priors_yaml(config)`` for callers that need the original
nested structure (e.g. ``CausalEffectResolver._literature_beta``).
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "build_physics_priors_map",
    "load_physics_priors_yaml",
]


def load_physics_priors_yaml(config: dict) -> dict:
    """Return the raw ``priors.yml`` content for *config*.

    Delegates to ``sparc.config.config.load_physics_priors``; exposed here
    so callers only need to import from one place.
    """
    from sparc.config.config import load_physics_priors
    return load_physics_priors(config)


def build_physics_priors_map(
    config: dict,
    *,
    literature_weight: float | None = None,
) -> Dict[str, dict]:
    """Load and merge physics priors into a per-variable coefficient map.

    Parameters
    ----------
    config:
        Full pipeline config (``load_config(project_path)`` output).
    literature_weight:
        Blend weight for the literature coefficient (0 = pure OLS,
        1 = pure literature).  Defaults to
        ``config["physics"]["literature_weight"]`` → 0.5.

    Returns
    -------
    Dict[str, dict]
        Keyed by variable name.  Each value has the schema documented in
        the module docstring.
    """
    physics = config.get("physics", {})
    if literature_weight is None:
        literature_weight = float(physics.get("literature_weight", 0.5))

    priors_data = load_physics_priors_yaml(config)
    coefficients_section: Dict[str, Any] = priors_data.get("coefficients", {})

    # Accept legacy flat ``ols_coefficients`` block from either the priors
    # YAML or the physics config section.
    ols_coeffs: Dict[str, float] = {
        **priors_data.get("ols_coefficients", {}),
        **physics.get("ols_coefficients", {}),
    }

    result: Dict[str, dict] = {}
    all_vars = set(list(coefficients_section.keys()) + list(ols_coeffs.keys()))

    for var in all_vars:
        lit_entry = coefficients_section.get(var, {})
        if isinstance(lit_entry, dict):
            lit_c = float(lit_entry.get("value", 0.0))
        else:
            lit_c = float(lit_entry)

        ols_c = float(ols_coeffs.get(var, 0.0))

        # When no OLS prior exists, use the literature value directly so the
        # blend weight doesn't halve it.
        if ols_c == 0.0 and var not in ols_coeffs:
            blended = lit_c
        else:
            blended = (1.0 - literature_weight) * ols_c + literature_weight * lit_c

        direction = "cooling" if blended < 0 else "warming"

        # Infer unit increment from the variable name or explicit YAML field,
        # falling back to the ``unit_increments`` block or 1.0.
        if "Pct" in var or "pct" in var:
            unit_inc = 10.0  # literature reports per +10 pp
        elif var in ("NDVI", "Albedo"):
            unit_inc = 0.1   # literature reports per +0.1
        elif isinstance(lit_entry, dict) and "unit_increment" in lit_entry:
            unit_inc = float(lit_entry["unit_increment"])
        else:
            unit_inc = float(
                priors_data.get("unit_increments", {}).get(var, 1.0)
            )

        result[var] = {
            "coefficient":    blended,
            "unit_increment": unit_inc,
            "direction":      direction,
            "ols_coef":       ols_c,
            "lit_coef":       lit_c,
            "uncertainty":    (
                lit_entry.get("uncertainty", 0.2)
                if isinstance(lit_entry, dict) else 0.2
            ),
        }

    return result
