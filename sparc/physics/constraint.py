"""
sparc/physics/constraint.py — PhysicsConstraint protocol: the seam between
training-time PDE regularisation and inference-time coefficient shrinkage.

Two concrete adapters satisfy this protocol:

``PDEAdapter``
    Training-time.  Wraps ``pde_loss.PDELoss`` and ``pde_operators``.
    Regularises a neural field during gradient descent.

``PriorsAdapter``
    Inference-time.  Wraps ``PhysicsPriors`` / ``physics_priors_map``.
    Shrinks posterior treatment effects toward literature coefficients.

``NullPhysicsConstraint``
    No-op adapter for testing and ablation.

Two adapters justify a real seam (not hypothetical).  A third adapter
(e.g. graph-based soft constraints) can be added without touching callers.

Interface
---------
::

    class MyConstraint(PhysicsConstraint):
        def regularise(self, field, coords) -> float: ...
        def shrink(self, posterior, treatment) -> np.ndarray: ...

    adapter: PhysicsConstraint = PDEAdapter(pde_loss_instance)
    loss    = adapter.regularise(predicted_field, coords)
    effects = adapter.shrink(posterior_samples, "tree_canopy")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Protocol — the seam
# ---------------------------------------------------------------------------


@runtime_checkable
class PhysicsConstraint(Protocol):
    """Protocol satisfied by all physics constraint adapters.

    Implementations must provide both methods; callers cross this seam
    and never import a concrete adapter directly.
    """

    def regularise(self, field: Any, coords: Any) -> float:
        """Compute a physics regularisation loss for a predicted spatial field.

        Parameters
        ----------
        field:
            Predicted values, shape ``(N,)`` or ``(N, D)``.
        coords:
            Spatial coordinates, shape ``(N, 2)``.

        Returns
        -------
        float
            Scalar loss term (0.0 when constraints are inactive).
        """
        ...

    def shrink(self, posterior: np.ndarray, treatment: str) -> np.ndarray:
        """Shrink posterior treatment-effect samples toward a physics prior.

        Parameters
        ----------
        posterior:
            MCMC/variational posterior samples, shape ``(D,)`` or ``(D, N)``.
        treatment:
            Name of the treatment variable (e.g. ``"tree_canopy"``).

        Returns
        -------
        np.ndarray
            Shrunk posterior, same shape as input.
        """
        ...


# ---------------------------------------------------------------------------
# PDEAdapter — training-time (wraps pde_loss / pde_operators)
# ---------------------------------------------------------------------------


class PDEAdapter:
    """Training-time adapter.  Wraps a ``PDELoss`` instance.

    Parameters
    ----------
    pde_loss:
        Instance of :class:`sparc.physics.pde_loss.PDELoss`.
        Pass ``None`` to treat this adapter as a no-op (equivalent to
        ``NullPhysicsConstraint`` but preserves the PDEAdapter type).
    weight:
        Scalar multiplier applied to the PDE loss term.  Defaults to 1.0.
    """

    def __init__(self, pde_loss: Any = None, *, weight: float = 1.0) -> None:
        self._pde_loss = pde_loss
        self._weight = weight

    def regularise(self, field: Any, coords: Any) -> float:
        if self._pde_loss is None:
            return 0.0
        try:
            loss_val = self._pde_loss(field, coords)
            # pde_loss may return a torch.Tensor; extract scalar safely
            if hasattr(loss_val, "item"):
                return float(loss_val.item()) * self._weight
            return float(loss_val) * self._weight
        except Exception:  # noqa: BLE001
            return 0.0

    def shrink(self, posterior: np.ndarray, treatment: str) -> np.ndarray:
        # Training-time adapter has no shrinkage responsibility.
        return posterior

    def __repr__(self) -> str:
        return f"PDEAdapter(weight={self._weight}, active={self._pde_loss is not None})"


# ---------------------------------------------------------------------------
# PriorsAdapter — inference-time (wraps physics_priors_map)
# ---------------------------------------------------------------------------


class PriorsAdapter:
    """Inference-time adapter.  Wraps a pre-loaded ``physics_priors_map``.

    Parameters
    ----------
    priors_map:
        Dict produced by
        :func:`~sparc.interventions.physics_priors_registry.build_physics_priors_map`.
        Keyed by variable name; each value has ``coefficient``,
        ``direction``, ``uncertainty``, etc.
    shrinkage_weight:
        Blend weight: 0 = no shrinkage (pure posterior), 1 = full
        literature shrinkage.  Defaults to the ``calibration.shrinkage_weight``
        in the priors map if present, else 0.3.
    """

    def __init__(
        self,
        priors_map: Dict[str, Dict[str, Any]],
        *,
        shrinkage_weight: Optional[float] = None,
    ) -> None:
        self._priors_map = priors_map or {}
        # Allow per-map override from calibration block
        cal = self._priors_map.get("calibration") or {}
        if shrinkage_weight is None:
            shrinkage_weight = float(cal.get("shrinkage_weight", 0.3))
        self._w = float(shrinkage_weight)

    def regularise(self, field: Any, coords: Any) -> float:
        # Inference-time adapter has no training loss responsibility.
        return 0.0

    def shrink(self, posterior: np.ndarray, treatment: str) -> np.ndarray:
        entry = self._priors_map.get(treatment)
        if entry is None:
            return posterior
        lit_coef = entry.get("lit_coef") or entry.get("coefficient")
        if lit_coef is None:
            return posterior
        lit_val = float(lit_coef)
        arr = np.asarray(posterior, dtype=np.float64)
        return (1.0 - self._w) * arr + self._w * lit_val

    @classmethod
    def from_project_config(cls, config: Dict[str, Any]) -> "PriorsAdapter":
        """Convenience factory from a raw project config dict."""
        try:
            from sparc.interventions.physics_priors_registry import build_physics_priors_map
            priors_map = build_physics_priors_map(config)
        except Exception:
            priors_map = {}
        w = float(config.get("physics", {}).get("shrinkage_weight", 0.3))
        return cls(priors_map, shrinkage_weight=w)

    def __repr__(self) -> str:
        return f"PriorsAdapter(treatments={list(self._priors_map)!r}, w={self._w})"


# ---------------------------------------------------------------------------
# NullPhysicsConstraint — no-op (testing, ablation)
# ---------------------------------------------------------------------------


class NullPhysicsConstraint:
    """No-op adapter.  Passes through without modification.

    Use in tests when physics constraints should be disabled::

        model.train(physics_constraint=NullPhysicsConstraint())
    """

    def regularise(self, field: Any, coords: Any) -> float:
        return 0.0

    def shrink(self, posterior: np.ndarray, treatment: str) -> np.ndarray:
        return np.asarray(posterior, dtype=np.float64)

    def __repr__(self) -> str:
        return "NullPhysicsConstraint()"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_physics_constraint(
    config: Dict[str, Any],
    *,
    mode: str = "priors",
    pde_loss: Any = None,
) -> PhysicsConstraint:
    """Build the right adapter from a project config dict.

    Parameters
    ----------
    config:
        Full pipeline config dict.
    mode:
        ``"pde"`` → :class:`PDEAdapter`,
        ``"priors"`` → :class:`PriorsAdapter`,
        ``"null"`` → :class:`NullPhysicsConstraint`.
    pde_loss:
        Required when ``mode="pde"``.
    """
    if mode == "pde":
        w = float(config.get("training", {}).get("lambda_physics", 0.1))
        return PDEAdapter(pde_loss, weight=w)
    if mode == "priors":
        return PriorsAdapter.from_project_config(config)
    if mode == "null":
        return NullPhysicsConstraint()
    raise ValueError(f"Unknown mode {mode!r}. Choose from: pde, priors, null")
