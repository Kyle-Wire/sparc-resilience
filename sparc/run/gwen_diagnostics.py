"""
GWENDiagnosticBuilder — pure extraction of GWEN model diagnostics.

Extracted from ``save_gwen_diagnostics()`` in ``gwen_variable_selection.py``.
This class contains no I/O — it converts a fitted GWENModel into a structured
diagnostics dict.  Persistence is the caller's responsibility.

Design
------
- All methods are static: the class is a namespace, not a stateful object.
- No filesystem reads or writes.
- No artifact store access.
- Calling ``build()`` twice with the same inputs returns identical dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from sparc.models.gwen import GWENModel


class GWENDiagnosticBuilder:
    """Build a GWEN diagnostics dict from a fitted model.  Pure — no I/O."""

    @staticmethod
    def build(
        model: "GWENModel",
        feature_names: List[str],
        selection_threshold: float,
    ) -> dict:
        """Build and return the canonical GWEN diagnostics structure.

        Parameters
        ----------
        model:
            A fitted :class:`~sparc.models.gwen.GWENModel`.  Must have
            ``global_model`` and ``local_models`` attributes populated.
        feature_names:
            Feature names in the same order as columns in the training matrix.
        selection_threshold:
            The selection-frequency threshold used to call a feature "selected."

        Returns
        -------
        dict
            Three top-level sections: ``tuning_parameters``, ``stability_metrics``,
            ``selection_rule``.  The dict is JSON-safe (no numpy scalars).
        """
        local_coefs = np.array([m.coef_ for m in model.local_models])

        tuning = GWENDiagnosticBuilder._tuning_parameters(model)
        stability = GWENDiagnosticBuilder._stability_metrics(local_coefs, feature_names)
        rule = GWENDiagnosticBuilder._selection_rule(selection_threshold)

        return {
            "tuning_parameters": tuning,
            "stability_metrics": stability,
            "selection_rule": rule,
        }

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _tuning_parameters(model: "GWENModel") -> dict:
        gm = model.global_model
        alphas = gm.alphas_
        return {
            "bandwidth_k": model.k_neighbors,
            "alpha_range": f"{float(alphas.min()):.4f} - {float(alphas.max()):.4f}",
            "selected_alpha": float(gm.alpha_),
            "selected_l1_ratio": float(gm.l1_ratio_),
            "cv_folds": model.cv_folds,
            "sample_size": model.sample_size,
            "n_alphas": model.n_alphas,
            "l1_ratio_range": list(model.l1_ratios),
        }

    @staticmethod
    def _stability_metrics(local_coefs: np.ndarray, feature_names: List[str]) -> dict:
        """Per-feature stability statistics derived from local model coefficients."""
        metrics: dict = {}
        for i, name in enumerate(feature_names):
            coefs = local_coefs[:, i]
            nonzero = coefs[coefs != 0]
            if len(nonzero) > 0:
                sign_stability = float(
                    np.mean(np.sign(nonzero) == np.sign(nonzero[0]))
                )
                metrics[name] = {
                    "selection_frequency": float(np.mean(coefs != 0)),
                    "mean_abs_coefficient": float(np.mean(np.abs(nonzero))),
                    "coefficient_sign_stability": sign_stability,
                    "spatial_variability": float(np.std(np.abs(coefs))),
                    "coefficient_range": (
                        f"{float(nonzero.min()):.4f} to {float(nonzero.max()):.4f}"
                    ),
                }
            else:
                metrics[name] = {
                    "selection_frequency": 0.0,
                    "mean_abs_coefficient": 0.0,
                    "coefficient_sign_stability": 0.0,
                    "spatial_variability": 0.0,
                    "coefficient_range": "Not selected",
                }
        return metrics

    @staticmethod
    def _selection_rule(selection_threshold: float) -> dict:
        pct = selection_threshold * 100
        return {
            "threshold": float(selection_threshold),
            "criterion": "selection_frequency >= threshold",
            "rationale": (
                f"Variables selected in ≥{pct:.0f}% of local models "
                "considered spatially stable"
            ),
        }
