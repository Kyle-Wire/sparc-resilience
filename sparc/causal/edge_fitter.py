"""EdgeFitter — thin orchestrator for per-edge coefficient estimation.

Extracts the shrinkage-blending, monotone-constraint warning, and
edge-iteration logic from ``CausalValidator.fit()`` (lines 527–651) into
a focused module with a testable interface.

``EdgeFitter`` does not know about estimator internals — it accepts any
object satisfying the ``EdgeEstimator`` protocol (OLS, HGB, DML, or a test
stub).  This is C1 in the architecture review: the first extraction step
that breaks the god-object without requiring a full rewrite of Stage 3.

Design
------
* **No side effects on DAG state** — the graph is read, never mutated.
  When a :class:`ResidualContext` is provided, a remapped graph is created
  internally and discarded after use.
* **Failure isolation** — estimator exceptions are caught per-edge, logged
  at WARNING, and produce ``Coefficient(value=0.0)`` so the rest of the
  fit continues.
* **Monotone-constraint violations** emit a ``UserWarning`` and are stored
  unchanged — the caller (report / scenario stage) decides what to do.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sparc.causal.edge_estimator import Coefficient, EdgeEstimator
from sparc.causal.residual_context import ResidualContext


class EdgeFitter:
    """Orchestrate per-edge estimation with shrinkage and constraint checks.

    Parameters
    ----------
    estimator : EdgeEstimator
        Any object with ``fit(parent, child, data, confounders) -> Coefficient``.
    priors : dict
        Physics-prior coefficients keyed by parent variable name.
        Expected shape: ``{"Pct_Canopy": {"value": -0.028, "confidence": "high"}, ...}``.
    monotone : dict
        Sign constraints keyed by parent variable name (+1, -1, or 0).
    shrinkage : float
        Blending weight toward the physics prior (0 = all data, 1 = all prior).
        Applied only for edges leading directly to *target*.
    target : str or None
        Name of the outcome variable.  Shrinkage and monotone constraints are
        only applied for edges ``parent → target``.
    """

    def __init__(
        self,
        estimator: EdgeEstimator,
        priors: dict,
        monotone: dict,
        shrinkage: float = 0.7,
        target: Optional[str] = None,
    ) -> None:
        self.estimator = estimator
        self.priors = priors
        self.monotone = monotone
        self.shrinkage = shrinkage
        self.target = target

    # ------------------------------------------------------------------
    # Single-edge fit
    # ------------------------------------------------------------------

    def fit_edge(
        self,
        parent: str,
        child: str,
        data: pd.DataFrame,
        confounders: List[str],
    ) -> Coefficient:
        """Estimate, shrink, and check one edge.

        Returns
        -------
        Coefficient
            The (blended) coefficient.  On estimator failure returns a
            ``Coefficient`` with ``value=0.0``.
        """
        try:
            coeff = self.estimator.fit(parent, child, data, confounders)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "EdgeFitter: estimator failed for %s→%s (%s); using 0.0",
                parent, child, exc,
            )
            return Coefficient(edge=(parent, child), value=0.0)

        # Shrinkage toward physics prior — only applied for edges → outcome
        if child == self.target:
            prior_info = self.priors.get(parent, {})
            if prior_info:
                coeff_prior = float(prior_info.get("value", coeff.value))
                blended = (1.0 - self.shrinkage) * coeff.value + self.shrinkage * coeff_prior
                coeff = Coefficient(
                    edge=coeff.edge,
                    value=blended,
                    se=coeff.se,
                    model=coeff.model,
                    metadata={**coeff.metadata, "prior_value": coeff_prior, "raw_value": coeff.value},
                )

        # Monotone-constraint check — warn but never correct
        if child == self.target and parent in self.monotone:
            expected_sign = self.monotone[parent]
            if expected_sign != 0 and coeff.value != 0:
                actual_sign = 1 if coeff.value > 0 else -1
                if actual_sign != expected_sign:
                    warnings.warn(
                        f"EdgeFitter: monotone violation for {parent}→{child}: "
                        f"coeff={coeff.value:.4f} but expected sign={expected_sign}",
                        UserWarning,
                        stacklevel=2,
                    )

        return coeff

    # ------------------------------------------------------------------
    # All-edges fit
    # ------------------------------------------------------------------

    def fit_all(
        self,
        graph,
        data: pd.DataFrame,
        resid_context: Optional[ResidualContext] = None,
    ) -> Dict[Tuple[str, str], Coefficient]:
        """Fit all edges in *graph*, skipping those with missing columns.

        Parameters
        ----------
        graph : networkx.DiGraph
            DAG to iterate over.
        data : pd.DataFrame
            Observational data.  Columns must include all present node names.
        resid_context : ResidualContext or None
            When provided, edges in the remapped graph are used for fitting.
            The original *graph* is not mutated.

        Returns
        -------
        dict mapping (parent, child) → Coefficient
            Includes all edges where both columns exist in *data*.
            Failed estimations produce ``Coefficient(value=0.0)``.
        """
        from sparc.causal.dag_definition import compute_backdoor_set

        working_graph = (
            resid_context.remap_graph(graph)
            if resid_context and not resid_context.is_empty()
            else graph
        )

        results: Dict[Tuple[str, str], Coefficient] = {}

        for parent, child in working_graph.edges():
            if parent not in data.columns or child not in data.columns:
                continue

            confounders = [
                n for n in compute_backdoor_set(working_graph, parent, child)
                if n in data.columns
            ]

            coeff = self.fit_edge(parent, child, data, confounders)
            results[(parent, child)] = coeff

        return results
