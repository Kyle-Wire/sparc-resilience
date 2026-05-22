"""
Single source-of-truth ``KernelField`` dataclass shared by the classical
MGWR (`sparc.models.gwr.GWRModel`, `sparc.models.gwrf.GWRFModel`) and the
neural surrogate (`sparc.models.surrogates.DifferentiableGWR`,
`DifferentiableGWRF`).

Phases 1–4 of the correlogram roadmap produce per-variable Matérn
parameters, anisotropy ellipses, and a V×V effective-range matrix.  This
module bundles those into a single artifact-shaped object that the GWR
stack consumes through one consistent API.

Design rules
------------
* **Additive only**: legacy GWR call sites that pass ``bandwidth`` /
  ``variable_bandwidths`` continue to work unchanged.  When a
  ``KernelField`` is supplied it overrides the legacy triple.
* **Plain-Python serialisation**: the dataclass round-trips to / from the
  Stage-0 artifacts (`correlogram_matern_fit`, `correlogram_anisotropy`,
  `effective_range_matrix`) via :meth:`from_artifacts` and
  :meth:`to_artifact_payload`.  No NumPy arrays in the canonical form.
* **Per-predictor (κ_x, κ_y, θ, ν)** drive the *kernel shape*; per-pair
  ``bandwidth_to_outcome`` (Phase 4) drives the *spatial weighting scale*
  for that predictor's local regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import math
import numpy as np


# ---------------------------------------------------------------------------
# Per-predictor kernel record
# ---------------------------------------------------------------------------


@dataclass
class PredictorKernel:
    """Kernel parameters for a single predictor variable.

    All fields except ``name`` are optional; a missing field falls back to
    the isotropic / single-bandwidth default so the surrogate can still
    consume an incomplete record.
    """

    name: str
    # Phase 1 (Matérn fit)
    kappa: Optional[float] = None              # isotropic decay rate (1/length)
    nu: Optional[float] = None                 # smoothness ∈ {0.5, 1.5, 2.5}
    sigma2: Optional[float] = None             # marginal variance
    # Phase 2 (anisotropy ellipse)
    kappa_x: Optional[float] = None            # along-major decay rate
    kappa_y: Optional[float] = None            # along-minor decay rate
    theta_rad: Optional[float] = None          # major-axis orientation [0, π)
    eccentricity: Optional[float] = None       # 0 (isotropic) → 1 (degenerate)
    # Phase 4 (per-pair effective range to the outcome variable)
    bandwidth_to_outcome: Optional[float] = None
    # Phase 1 posterior support (used by Bayesian-MGWR mode)
    kappa_posterior_samples: Optional[list[float]] = None

    @property
    def is_anisotropic(self) -> bool:
        return (self.kappa_x is not None and self.kappa_y is not None
                and self.theta_rad is not None
                and abs(self.kappa_x - self.kappa_y) > 1e-9)

    def effective_kappa(self, phi_rad: float) -> float:
        """Direction-dependent effective κ at world-frame angle ``phi``.

        Uses the Phase-2 model ``κ_eff(φ)² = (κ_x cos(φ−θ))² + (κ_y sin(φ−θ))²``.
        Falls back to the isotropic ``kappa`` when anisotropy is absent.
        """
        if self.is_anisotropic:
            d = phi_rad - (self.theta_rad or 0.0)
            return float(math.sqrt((self.kappa_x * math.cos(d)) ** 2
                                   + (self.kappa_y * math.sin(d)) ** 2))
        if self.kappa is not None:
            return float(self.kappa)
        # Last-resort fallback: treat bandwidth_to_outcome as 1/κ
        if self.bandwidth_to_outcome and self.bandwidth_to_outcome > 0:
            return 1.0 / float(self.bandwidth_to_outcome)
        return 0.0

    @staticmethod
    def anisotropy_from_terrain(
        slope_rad: float,
        aspect_rad: float,
        base_kappa: float,
    ) -> dict:
        """Derive anisotropic kernel parameters from local terrain gradient.

        The major axis (smaller κ → larger range) aligns with the downhill
        direction.  Eccentricity scales proportionally to the normalised slope
        ``slope_rad / (π/2)`` so that flat terrain is perfectly isotropic.

        Parameters
        ----------
        slope_rad : slope angle in radians [0, π/2]
        aspect_rad : downhill azimuth clockwise from north, in [0, 2π).
        base_kappa : isotropic decay rate (1/length) at zero slope.

        Returns
        -------
        dict with keys ``kappa_x`` (along-slope), ``kappa_y`` (cross-slope),
        and ``theta_rad`` (major-axis orientation in [0, π)).
        """
        eccentricity = float(slope_rad) / (math.pi / 2.0)
        eccentricity = max(0.0, min(eccentricity, 1.0))
        kappa_x = float(base_kappa) / (1.0 + eccentricity)  # along-slope: larger range
        kappa_y = float(base_kappa) * (1.0 + eccentricity)  # cross-slope: smaller range
        theta_rad = float(aspect_rad) % math.pi              # major axis in [0, π)
        return {"kappa_x": kappa_x, "kappa_y": kappa_y, "theta_rad": theta_rad}


# ---------------------------------------------------------------------------
# KernelField (one per outcome variable)
# ---------------------------------------------------------------------------


@dataclass
class KernelField:
    """Matrix-aware kernel-field bundle consumed by base + surrogate MGWR.

    A single ``KernelField`` describes the kernel geometry for **one outcome
    variable** plus all of its predictors.  Stage-2 fits one ``GWRModel`` /
    ``DifferentiableGWR`` per outcome, so this granularity matches the
    fit-time API.
    """

    outcome_name: str
    predictors: list[PredictorKernel]
    # The Phase-4 V×V matrix (variable_names × variable_names) — full matrix
    # is retained so callers can look up any (i, j) cross-range.
    variable_names: list[str] = field(default_factory=list)
    range_matrix: list[list[Optional[float]]] = field(default_factory=list)
    bandwidth_mismatch_warnings: list[dict] = field(default_factory=list)
    # Metadata
    matern_default_nu: float = 1.5
    source: str = "stage0_artifacts"

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def predictor(self, name: str) -> Optional[PredictorKernel]:
        for p in self.predictors:
            if p.name == name:
                return p
        return None

    def cross_range(self, var_a: str, var_b: str) -> Optional[float]:
        if var_a not in self.variable_names or var_b not in self.variable_names:
            return None
        i = self.variable_names.index(var_a)
        j = self.variable_names.index(var_b)
        v = self.range_matrix[i][j]
        return float(v) if v is not None else None

    def bandwidth_for(self, predictor_name: str,
                      fallback: Optional[float] = None) -> Optional[float]:
        """Phase-4 per-pair (predictor → outcome) effective range."""
        p = self.predictor(predictor_name)
        if p is not None and p.bandwidth_to_outcome is not None:
            return float(p.bandwidth_to_outcome)
        # Fallback to the V×V matrix lookup
        v = self.cross_range(self.outcome_name, predictor_name)
        if v is not None:
            return v
        return fallback

    # ------------------------------------------------------------------
    # Anisotropic distance helper (used by the base MGWR)
    # ------------------------------------------------------------------

    @staticmethod
    def anisotropic_distance(dx: np.ndarray, dy: np.ndarray,
                             kappa_x: float, kappa_y: float,
                             theta_rad: float) -> np.ndarray:
        """Distances measured in a predictor's own anisotropic frame.

        Rotate the displacement ``(dx, dy)`` into the (κ_x, κ_y, θ) frame
        and scale each axis by its κ.  The returned array is a
        dimensionless metric — multiply by an effective length scale (or
        compare against ``1`` for a Matérn ``κ‖r‖`` kernel) before
        weighting.
        """
        dx = np.asarray(dx, dtype=np.float64)
        dy = np.asarray(dy, dtype=np.float64)
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        # Rotate into the principal-axis frame
        u = dx * c + dy * s
        v = -dx * s + dy * c
        # Scaled metric: (κ_x · u)² + (κ_y · v)²  (κ has units of 1/length)
        return np.sqrt((kappa_x * u) ** 2 + (kappa_y * v) ** 2)

    # ------------------------------------------------------------------
    # Construction from Stage-0 artifacts
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        outcome_name: str,
        predictor_names: list[str],
        *,
        matern_artifact: Optional[dict] = None,
        anisotropy_artifact: Optional[dict] = None,
        effective_range_matrix: Optional[dict] = None,
        default_nu: float = 1.5,
    ) -> "KernelField":
        """Assemble a ``KernelField`` from the Stage-0 artifact dicts.

        All inputs are optional; missing artifacts fall back to ``None``
        fields on each ``PredictorKernel``, which the base / surrogate
        models interpret as "use the legacy isotropic kernel".
        """
        matern_per_var = (matern_artifact or {}).get("per_variable_fits", {}) \
            if matern_artifact else {}
        aniso_per_var = (anisotropy_artifact or {}).get("per_variable_fits", {}) \
            if anisotropy_artifact else {}

        var_names = (effective_range_matrix or {}).get("variable_names", []) \
            if effective_range_matrix else []
        M = (effective_range_matrix or {}).get("range_matrix", []) \
            if effective_range_matrix else []
        warnings_ = (effective_range_matrix or {}).get("mismatch_warnings", []) \
            if effective_range_matrix else []

        # Per-pair bandwidth: row of the matrix corresponding to outcome.
        outcome_row: dict[str, float] = {}
        if effective_range_matrix and outcome_name in var_names:
            ti = var_names.index(outcome_name)
            for j, vname in enumerate(var_names):
                v = M[ti][j] if (ti < len(M) and j < len(M[ti])) else None
                if v is not None:
                    outcome_row[vname] = float(v)

        def _scalar(v):
            """Return v['mean'] if v is a posterior-stats dict, else v itself."""
            if isinstance(v, dict):
                return v.get("mean")
            return v

        predictors: list[PredictorKernel] = []
        for pname in predictor_names:
            mfit = matern_per_var.get(pname) or {}
            kappa_mean = (
                ((mfit.get("posterior") or {}).get("kappa") or {}).get("mean")
                if isinstance(mfit, dict) else None
            ) or _scalar(mfit.get("kappa"))
            nu_mean = (
                ((mfit.get("posterior") or {}).get("nu") or {}).get("mean")
                if isinstance(mfit, dict) else None
            ) or _scalar(mfit.get("nu"))
            sigma2_mean = (
                ((mfit.get("posterior") or {}).get("sigma2") or {}).get("mean")
                if isinstance(mfit, dict) else None
            ) or _scalar(mfit.get("sigma2"))
            # kappa samples may live inside the kappa posterior dict
            _kappa_raw = mfit.get("kappa") if isinstance(mfit, dict) else None
            kappa_samples = (
                mfit.get("kappa_samples")
                or (_kappa_raw.get("samples") if isinstance(_kappa_raw, dict) else None)
            )

            afit = aniso_per_var.get(pname) or {}
            kx = (
                ((afit.get("posterior") or {}).get("kappa_x") or {}).get("mean")
                if isinstance(afit, dict) else None
            ) or _scalar(afit.get("kappa_x"))
            ky = (
                ((afit.get("posterior") or {}).get("kappa_y") or {}).get("mean")
                if isinstance(afit, dict) else None
            ) or _scalar(afit.get("kappa_y"))
            theta = (
                ((afit.get("posterior") or {}).get("theta_major_rad") or {}).get("mean")
                if isinstance(afit, dict) else None
            ) or _scalar(afit.get("theta_major_rad")) or _scalar(afit.get("theta_rad"))
            ecc = (((afit.get("ellipse") or {}).get("eccentricity") or {}).get("mean")
                   if isinstance(afit, dict) else None)

            predictors.append(PredictorKernel(
                name=pname,
                kappa=float(kappa_mean) if kappa_mean is not None else None,
                nu=float(nu_mean) if nu_mean is not None else None,
                sigma2=float(sigma2_mean) if sigma2_mean is not None else None,
                kappa_x=float(kx) if kx is not None else None,
                kappa_y=float(ky) if ky is not None else None,
                theta_rad=float(theta) if theta is not None else None,
                eccentricity=float(ecc) if ecc is not None else None,
                bandwidth_to_outcome=outcome_row.get(pname),
                kappa_posterior_samples=(
                    list(map(float, kappa_samples))
                    if kappa_samples is not None else None
                ),
            ))

        return cls(
            outcome_name=outcome_name,
            predictors=predictors,
            variable_names=list(var_names),
            range_matrix=[list(row) for row in M] if M else [],
            bandwidth_mismatch_warnings=list(warnings_),
            matern_default_nu=float(default_nu),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_artifact_payload(self) -> dict[str, Any]:
        return {
            "outcome_name": self.outcome_name,
            "predictors": [asdict(p) for p in self.predictors],
            "variable_names": list(self.variable_names),
            "range_matrix": [list(row) for row in self.range_matrix],
            "bandwidth_mismatch_warnings": list(self.bandwidth_mismatch_warnings),
            "matern_default_nu": float(self.matern_default_nu),
            "source": self.source,
        }

    @classmethod
    def from_artifact_payload(cls, payload: dict[str, Any]) -> "KernelField":
        return cls(
            outcome_name=payload["outcome_name"],
            predictors=[PredictorKernel(**p) for p in payload.get("predictors", [])],
            variable_names=list(payload.get("variable_names", [])),
            range_matrix=[list(r) for r in payload.get("range_matrix", [])],
            bandwidth_mismatch_warnings=list(payload.get("bandwidth_mismatch_warnings", [])),
            matern_default_nu=float(payload.get("matern_default_nu", 1.5)),
            source=payload.get("source", "deserialised"),
        )


# ---------------------------------------------------------------------------
# Matérn kernel (closed-form for the standard ν values)
# ---------------------------------------------------------------------------


def matern_kernel_weights(distances: np.ndarray, kappa: float,
                           nu: float = 1.5) -> np.ndarray:
    """Matérn kernel ``M_ν(κ‖r‖)`` for ν ∈ {0.5, 1.5, 2.5}.

    The kernel is normalised to ``1`` at zero lag.  Returned weights are
    suitable as direct GWR sample weights (replacing the Gaussian /
    exponential / bisquare weights in the legacy path).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if kappa <= 0 or not np.isfinite(kappa):
        return np.ones_like(distances)
    h = kappa * distances
    if nu <= 0.5 + 1e-6:
        return np.exp(-h)
    if nu <= 1.5 + 1e-6:
        return (1.0 + math.sqrt(3.0) * h) * np.exp(-math.sqrt(3.0) * h)
    # ν = 2.5
    return (1.0 + math.sqrt(5.0) * h + (5.0 / 3.0) * h ** 2) \
        * np.exp(-math.sqrt(5.0) * h)
