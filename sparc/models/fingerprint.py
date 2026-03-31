"""
Fingerprinting hybrid layer — project SPARC predictions onto a known
forced spatial pattern (ensemble-mean or leading EOF) to improve
climate signal detection.

Usage
-----
    from sparc.models.fingerprint import FingerprintProjector

    fp = FingerprintProjector()
    fp.from_ensemble_mean(truth_field)        # or fp.from_eof(member_matrix, n_modes=3)
    blended = fp.blend(raw_predictions, alpha=0.3)
    scores  = fp.score(raw_predictions, truth_field)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


class FingerprintProjector:
    """Project spatial predictions onto a known forced pattern.

    Two pattern sources are supported:

    * **ensemble_mean** — the multi-model-mean truth field (e.g.
      ``{var}_trend_forced``).  Simple and robust.
    * **eof** — leading EOF(s) extracted via SVD from the member × grid
      matrix of predictions or residuals.  Captures dominant spatial
      modes of inter-model spread.

    After fitting a pattern, call :meth:`project` to obtain the
    projection coefficient and projected field, :meth:`blend` to merge
    with raw predictions, and :meth:`score` for evaluation metrics.
    """

    def __init__(self) -> None:
        self.pattern: Optional[np.ndarray] = None        # (n_grid,) or (n_modes, n_grid)
        self.pattern_source: Optional[str] = None        # "ensemble_mean" | "eof"
        self.eof_singular_values: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Pattern fitting
    # ------------------------------------------------------------------

    def from_ensemble_mean(self, truth_field: np.ndarray) -> "FingerprintProjector":
        """Set the target pattern to the ensemble-mean forced field.

        Parameters
        ----------
        truth_field : array-like, shape (n_grid,)
            The "truth" forced-response field (e.g. from the ensemble mean
            of CMIP models).
        """
        truth = np.asarray(truth_field, dtype=float).ravel()
        norm = np.linalg.norm(truth)
        if norm == 0:
            raise ValueError("truth_field is all zeros — cannot normalise")
        self.pattern = truth / norm
        self.pattern_source = "ensemble_mean"
        return self

    def from_eof(
        self,
        member_fields: np.ndarray,
        n_modes: int = 3,
    ) -> "FingerprintProjector":
        """Extract leading EOF(s) from a member × grid matrix.

        Parameters
        ----------
        member_fields : array-like, shape (n_members, n_grid)
            Each row is one ensemble member's spatial field (e.g. trend
            predictions or residuals from the forced truth).
        n_modes : int
            Number of EOF modes to retain.
        """
        M = np.asarray(member_fields, dtype=float)
        if M.ndim != 2:
            raise ValueError(f"member_fields must be 2-D; got shape {M.shape}")

        # Centre across members (remove the multi-model mean)
        M_centred = M - M.mean(axis=0, keepdims=True)

        # Thin SVD: U (n_members, k), S (k,), Vt (k, n_grid)
        U, S, Vt = np.linalg.svd(M_centred, full_matrices=False)
        n_modes = min(n_modes, len(S))

        self.pattern = Vt[:n_modes]                          # (n_modes, n_grid)
        self.eof_singular_values = S[:n_modes]
        total_var = np.sum(S ** 2)
        self.explained_variance_ratio = (S[:n_modes] ** 2) / total_var if total_var > 0 else S[:n_modes] * 0
        self.pattern_source = "eof"
        return self

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project predictions onto the stored pattern(s).

        Parameters
        ----------
        predictions : array-like, shape (n_grid,)

        Returns
        -------
        coefficients : ndarray, shape (1,) or (n_modes,)
            Dot-product projection coefficient(s).
        projected_field : ndarray, shape (n_grid,)
            Reconstruction from the pattern(s): sum_k coeff_k * pattern_k.
        """
        if self.pattern is None:
            raise RuntimeError("No pattern fitted — call from_ensemble_mean() or from_eof() first")
        preds = np.asarray(predictions, dtype=float).ravel()

        if self.pattern.ndim == 1:
            # Single pattern (ensemble-mean mode)
            coeff = np.array([np.dot(preds, self.pattern)])
            projected = coeff[0] * self.pattern
        else:
            # Multiple EOF modes
            coeff = self.pattern @ preds                      # (n_modes,)
            projected = coeff @ self.pattern                  # (n_grid,)

        return coeff, projected

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    def blend(
        self,
        raw_predictions: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Blend raw predictions with the fingerprint-projected field.

        ``final = alpha * projected + (1 - alpha) * raw``

        Parameters
        ----------
        raw_predictions : array-like, shape (n_grid,)
        alpha : float in [0, 1]
            0 = pure raw, 1 = pure fingerprint projection.
        """
        _, projected = self.project(raw_predictions)
        return alpha * projected + (1.0 - alpha) * np.asarray(raw_predictions, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _pattern_correlation(a: np.ndarray, b: np.ndarray) -> float:
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 3:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _amplitude_ratio(pred: np.ndarray, truth: np.ndarray) -> float:
        mask = ~(np.isnan(pred) | np.isnan(truth))
        p, t = pred[mask], truth[mask]
        if len(t) < 2 or np.std(t) == 0:
            return np.nan
        return float(np.std(p) / np.std(t))

    def score(
        self,
        predictions: np.ndarray,
        truth: np.ndarray,
    ) -> Dict[str, float]:
        """Score predictions against a truth field.

        Returns dict with ``pattern_correlation``, ``amplitude_ratio``,
        ``pattern_correlation_projected``, ``amplitude_ratio_projected``,
        and projection ``coefficients``.
        """
        preds = np.asarray(predictions, dtype=float).ravel()
        truth = np.asarray(truth, dtype=float).ravel()
        _, projected = self.project(preds)

        return {
            "pattern_correlation_raw": self._pattern_correlation(preds, truth),
            "amplitude_ratio_raw": self._amplitude_ratio(preds, truth),
            "pattern_correlation_projected": self._pattern_correlation(projected, truth),
            "amplitude_ratio_projected": self._amplitude_ratio(projected, truth),
        }
