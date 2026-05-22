"""Direct-effect computation strategies for Stage-4 scenario simulation.

Provides a ``DirectEffectChain`` — a priority-ordered list of
``DirectEffectStrategy`` adapters.  Each adapter encapsulates one tier of
evidence for the per-cell causal direct effect::

    Tier 0  AlphaFieldStrategy  — PDE-learned α field × causal β_global
    Tier 1  NUTSBetaStrategy    — NUTS posterior-mean β(s) × PDP × physics guard
    Tier 2  SaturationStrategy  — dose-response curve only (β(s) unavailable)
    Tier 3  PhysicsLitStrategy  — pure literature prior fallback

The chain interface is::

    chain = DirectEffectChain.from_simulator(sim)
    delta, label = chain.compute(variable, effective_change,
                                 baseline_values, modified_values)

``compute`` tries each adapter in order and returns the first non-``None``
result.  Each adapter is independently testable by instantiating it
directly and calling ``.compute()``.

Design note: strategies receive dependencies via constructor injection
(callables + immutable dicts), not via direct ``ScenarioSimulator``
reference.  This keeps them stateless and mockable.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "DirectEffectChain",
    "AlphaFieldStrategy",
    "NUTSBetaStrategy",
    "SaturationStrategy",
    "PhysicsLitStrategy",
]

# Type alias
_EffectResult = Optional[Tuple[np.ndarray, str]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _spatial_gini(delta: np.ndarray) -> float:
    """Return the Gini coefficient of *delta* as a spatial inequality measure."""
    arr = np.abs(delta).ravel()
    if arr.sum() < 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def _apply_physics_guard(
    delta: np.ndarray,
    variable: str,
    physics_priors: Dict[str, dict],
) -> Tuple[np.ndarray, bool]:
    """Sign check + ±3σ magnitude cap from physics literature.

    Returns ``(guarded_delta, was_applied)``.
    """
    prior = physics_priors.get(variable, {})
    expected_dir = prior.get("direction")

    if expected_dir in ("negative", "decrease", "cooling"):
        wrong_sign = delta > 0
    elif expected_dir in ("positive", "increase", "warming"):
        wrong_sign = delta < 0
    else:
        wrong_sign = np.zeros(len(delta), dtype=bool)

    applied = False
    if wrong_sign.any():
        delta = np.where(wrong_sign, 0.0, delta)
        warnings.warn(
            f"Physics guard: {int(wrong_sign.sum())} cells for '{variable}' "
            "zeroed (wrong sign vs. literature prior).",
            RuntimeWarning,
            stacklevel=4,
        )
        applied = True

    abs_d = np.abs(delta)
    d_mean = float(np.mean(abs_d))
    d_std = float(np.std(abs_d))
    cap = d_mean + 3.0 * d_std
    if cap > 1e-10:
        clipped = np.clip(delta, -cap, cap)
        if not np.array_equal(clipped, delta):
            n_clipped = int(np.sum(np.abs(delta) > cap))
            warnings.warn(
                f"Physics guard: {n_clipped} cells for '{variable}' "
                "clipped at ±3σ magnitude cap.",
                RuntimeWarning,
                stacklevel=4,
            )
            delta = clipped
            applied = True

    return delta, applied


def _pdp_shape_modifier(
    variable: str,
    baseline_values: np.ndarray,
    modified_values: np.ndarray,
    effective_change: np.ndarray,
    curve: dict,
) -> Optional[np.ndarray]:
    """Compute dimensionless PDP saturation shape modifier f_PDP ∈ [0.1, 10].

    Returns ``None`` when the extrapolation guard is unavailable or the PDP
    slope mean is too small to normalise.
    """
    try:
        from sparc.interventions.extrapolation_guard import (
            predict_scenario_with_saturation,
        )
    except ImportError:
        return None

    try:
        pdp_delta = predict_scenario_with_saturation(
            variable_name=variable,
            baseline_values=baseline_values,
            modified_values=modified_values,
            condition_curve={
                "grid_values": curve["grid_values"],
                "pdp_values": curve["pdp_values"],
            },
            spatial_multiplier=None,
        )
    except Exception as exc:
        warnings.warn(
            f"PDP shape modifier failed for {variable}: {exc}; using f_PDP=1.",
            RuntimeWarning,
            stacklevel=4,
        )
        return None

    active = np.abs(effective_change) > 1e-8
    raw_slope = np.where(active, np.abs(pdp_delta) / np.abs(effective_change), 0.0)
    slope_mean = float(np.mean(raw_slope[active])) if active.any() else 0.0
    if slope_mean <= 1e-10:
        return np.ones(len(effective_change), dtype=float)
    return np.clip(raw_slope / slope_mean, 0.1, 10.0)


# ---------------------------------------------------------------------------
# Strategy adapters
# ---------------------------------------------------------------------------

class AlphaFieldStrategy:
    """Tier 0: PDE-learned spatial α field × causal β_global.

    Uses the physics-constrained α field from Stage 2 (GWEN + PDE loss) to
    spatially modulate the global causal coefficient.  ``beta_global`` comes
    from ``get_causal_coeff_fn`` (DAG-adjusted structural coefficient) with a
    fallback to the literature prior per unit increment.

    Optionally further shapes the effect with the PDP saturation modifier when
    a condition curve with R² ≥ ``min_r2`` is available.
    """

    def __init__(
        self,
        *,
        alpha_field: Optional[np.ndarray],
        get_causal_coeff_fn: Callable[[str], Optional[float]],
        condition_curves: Dict[str, dict],
        min_r2: float,
        physics_priors: Dict[str, dict],
    ) -> None:
        self._alpha_field = alpha_field
        self._get_causal_coeff = get_causal_coeff_fn
        self._curves = condition_curves
        self._min_r2 = min_r2
        self._physics_priors = physics_priors

    def compute(
        self,
        variable: str,
        effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray],
        modified_values: Optional[np.ndarray],
    ) -> _EffectResult:
        if self._alpha_field is None:
            return None

        n = min(len(effective_change), len(self._alpha_field))
        alpha_s = self._alpha_field[:n]
        alpha_mean = float(np.mean(alpha_s))
        if alpha_mean <= 1e-12:
            return None

        alpha_norm = alpha_s / alpha_mean  # centred at 1.0

        # β_global from the DAG-adjusted structural coefficient or lit prior
        beta_global = self._get_causal_coeff(variable)
        if beta_global is None:
            prior = self._physics_priors.get(variable, {})
            unit_inc = prior.get("unit_increment", 1.0)
            beta_global = prior.get("lit_coef", 0.0) / unit_inc if unit_inc else 0.0

        curve = self._curves.get(variable)
        if (
            curve is not None
            and curve.get("r2", 0.0) >= self._min_r2
            and baseline_values is not None
            and modified_values is not None
        ):
            eff_n = effective_change[:n]
            f_pdp = _pdp_shape_modifier(
                variable,
                baseline_values[:n],
                modified_values[:n],
                eff_n,
                curve,
            )
            if f_pdp is not None:
                delta = beta_global * eff_n * f_pdp * alpha_norm
                gini = _spatial_gini(delta)
                print(
                    f"   [Tier 0] alpha+PDP delta for {variable}: "
                    f"Gini={gini:.4f}, mean={float(np.mean(delta)):.4f}  "
                    f"beta_global={beta_global:.6f}"
                )
                return delta, "pde_alpha_field_pdp"

        # Without PDP: β_global × alpha_norm × effective_change
        delta = beta_global * alpha_norm * effective_change[:n]
        gini = _spatial_gini(delta)
        print(
            f"   [Tier 0] alpha delta for {variable}: "
            f"Gini={gini:.4f}, mean={float(np.mean(delta)):.4f}"
        )
        return delta, "pde_alpha_field"


class NUTSBetaStrategy:
    """Tier 1+2: NUTS posterior-mean per-cell β(s) × PDP modifier × physics guard.

    The primary causal estimator for Stage 4.  β(s) is the spatial CATE
    posterior mean from ``("3", "cate_summary").cate_mean`` — a causal quantity
    unlike the MGWR local coefficients it replaced.

    Optionally multiplied by the dimensionless PDP saturation shape modifier
    when a condition curve with R² ≥ ``min_r2`` is available (``combined_beta_pdp``)
    or applied directly when not (``combined_beta``).  A physics guard
    (sign check + ±3σ cap) is applied after assembly.
    """

    def __init__(
        self,
        *,
        get_bayesian_beta_fn: Callable[[str], Optional[np.ndarray]],
        condition_curves: Dict[str, dict],
        min_r2: float,
        physics_priors: Dict[str, dict],
    ) -> None:
        self._get_beta = get_bayesian_beta_fn
        self._curves = condition_curves
        self._min_r2 = min_r2
        self._physics_priors = physics_priors

    def compute(
        self,
        variable: str,
        effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray],
        modified_values: Optional[np.ndarray],
    ) -> _EffectResult:
        beta_s = self._get_beta(variable)
        if beta_s is None or beta_s.size == 0:
            return None

        n = len(effective_change)
        nb = min(n, len(beta_s))

        # Attempt PDP shape modifier
        f_pdp: Optional[np.ndarray] = None
        curve = self._curves.get(variable)
        if (
            curve is not None
            and curve.get("r2", 0.0) >= self._min_r2
            and baseline_values is not None
            and modified_values is not None
        ):
            f_pdp = _pdp_shape_modifier(
                variable,
                baseline_values[:nb],
                modified_values[:nb],
                effective_change[:nb],
                curve,
            )

        # Assemble δ = β(s) × [f_PDP] × Δx
        if f_pdp is not None:
            nf = min(nb, len(f_pdp))
            delta = beta_s[:nf] * f_pdp[:nf] * effective_change[:nf]
            source = "combined_beta_pdp"
        else:
            delta = beta_s[:nb] * effective_change[:nb]
            source = "combined_beta"

        delta, guard_applied = _apply_physics_guard(delta, variable, self._physics_priors)

        gini = _spatial_gini(delta)
        print(
            f"   [{source}] delta for {variable}: "
            f"Gini={gini:.4f}, mean={float(np.mean(delta)):+.4f}, "
            f"f_pdp={'yes' if f_pdp is not None else 'no'}, "
            f"guard={'applied' if guard_applied else 'ok'}"
        )
        return delta, source


class SaturationStrategy:
    """Tier 2 standalone: dose-response saturation curve when β(s) is unavailable.

    Uses the GWRF PDP condition curve to predict per-cell response.  Applied
    only when a curve with R² ≥ ``min_r2`` exists and β(s) is not available
    (β(s) available → handled by ``NUTSBetaStrategy`` which incorporates the
    curve as a shape modifier).
    """

    def __init__(
        self,
        *,
        condition_curves: Dict[str, dict],
        min_r2: float,
    ) -> None:
        self._curves = condition_curves
        self._min_r2 = min_r2

    def compute(
        self,
        variable: str,
        effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray],
        modified_values: Optional[np.ndarray],
    ) -> _EffectResult:
        curve = self._curves.get(variable)
        if (
            curve is None
            or curve.get("r2", 0.0) < self._min_r2
            or baseline_values is None
            or modified_values is None
        ):
            return None

        try:
            from sparc.interventions.extrapolation_guard import (
                predict_scenario_with_saturation,
            )
            delta = predict_scenario_with_saturation(
                variable_name=variable,
                baseline_values=baseline_values,
                modified_values=modified_values,
                condition_curve={
                    "grid_values": curve["grid_values"],
                    "pdp_values": curve["pdp_values"],
                },
                spatial_multiplier=None,
            )
            return delta, "saturation_curve"
        except Exception as exc:
            warnings.warn(
                f"Saturation curve fallback failed for {variable}: {exc}; "
                "deferring to physics literal.",
                RuntimeWarning,
                stacklevel=4,
            )
            return None


class PhysicsLitStrategy:
    """Tier 3: Pure physics literature prior — always applicable, never None.

    Produces a spatially uniform effect ``lit_per_unit × Δx``.  Used when
    no causal estimator (β(s), α field, saturation curve) is available.
    """

    def __init__(self, *, physics_priors: Dict[str, dict]) -> None:
        self._physics_priors = physics_priors

    def compute(
        self,
        variable: str,
        effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray],  # noqa: ARG002
        modified_values: Optional[np.ndarray],  # noqa: ARG002
    ) -> _EffectResult:
        prior = self._physics_priors.get(variable, {})
        unit_inc = prior.get("unit_increment", 1.0)
        lit_per_unit = prior.get("lit_coef", 0.0) / unit_inc if unit_inc else 0.0
        return lit_per_unit * effective_change, "physics_lit"


# ---------------------------------------------------------------------------
# Chain combinator
# ---------------------------------------------------------------------------

class DirectEffectChain:
    """Priority-ordered chain of ``DirectEffectStrategy`` adapters.

    ``compute`` tries each adapter in registration order and returns the first
    non-``None`` result.  ``PhysicsLitStrategy`` should always be the last
    adapter in the list so the chain never returns ``None``.

    Build via :meth:`from_simulator` to get the standard four-tier chain::

        chain = DirectEffectChain.from_simulator(sim)
        delta, label = chain.compute(
            "Pct_Canopy", effective_change,
            baseline_values=baseline, modified_values=modified,
        )

    Or inject custom adapters for testing::

        chain = DirectEffectChain([
            NUTSBetaStrategy(get_bayesian_beta_fn=mock_beta, ...),
            PhysicsLitStrategy(physics_priors={}),
        ])
    """

    def __init__(
        self,
        strategies: list,
    ) -> None:
        self._strategies = strategies

    @classmethod
    def from_simulator(cls, sim: "ScenarioSimulator") -> "DirectEffectChain":  # type: ignore[name-defined]
        """Build the standard four-tier chain from a live *ScenarioSimulator*."""
        return cls([
            AlphaFieldStrategy(
                alpha_field=sim._alpha_field,
                get_causal_coeff_fn=sim.get_causal_coefficient,
                condition_curves=sim._condition_curves,
                min_r2=sim._condition_curve_min_r2,
                physics_priors=sim.physics_priors,
            ),
            NUTSBetaStrategy(
                get_bayesian_beta_fn=sim._get_bayesian_beta,
                condition_curves=sim._condition_curves,
                min_r2=sim._condition_curve_min_r2,
                physics_priors=sim.physics_priors,
            ),
            SaturationStrategy(
                condition_curves=sim._condition_curves,
                min_r2=sim._condition_curve_min_r2,
            ),
            PhysicsLitStrategy(
                physics_priors=sim.physics_priors,
            ),
        ])

    def compute(
        self,
        variable: str,
        effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray] = None,
        modified_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str]:
        """Return *(delta, method_label)* from the first applicable adapter."""
        for strategy in self._strategies:
            result = strategy.compute(
                variable, effective_change, baseline_values, modified_values
            )
            if result is not None:
                return result
        # Unreachable when PhysicsLitStrategy is the last adapter, but safe fallback.
        return np.zeros_like(effective_change), "no_effect"
