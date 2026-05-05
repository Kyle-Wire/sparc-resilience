"""Phase C-3 — CATE-vs-GWR divergence audit.

When the **causal** per-cell CATE (Stage 3) and the **correlational**
per-cell GWR β (Stage 2) disagree by more than what kernel uncertainty
would predict, the most likely explanations are:

1. Hidden confounding inflating GWR β (CATE corrects → genuine signal).
2. Treatment effect modification by a covariate the surrogate cannot
   represent.
3. Severe model mis-specification on either side.

This module computes per-cell signed divergence ``d_i = β_i − τ_i`` and
flags cells where ``|d_i|`` exceeds a posterior-aware threshold.  The
output is a JSON-friendly payload ready to persist as
``stage="3", artifact_id="cate_vs_gwr_divergence"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DivergenceReport:
    treatment: str
    cell_count: int
    flagged_count: int
    flagged_fraction: float
    mean_abs_divergence: float
    median_abs_divergence: float
    max_abs_divergence: float
    pearson_correlation: float
    sign_agreement_rate: float
    threshold_used: float
    per_cell_divergence: Optional[np.ndarray] = None
    flagged_mask: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self, include_per_cell: bool = True) -> dict:
        out = {
            "treatment": self.treatment,
            "cell_count": int(self.cell_count),
            "flagged_count": int(self.flagged_count),
            "flagged_fraction": float(self.flagged_fraction),
            "mean_abs_divergence": float(self.mean_abs_divergence),
            "median_abs_divergence": float(self.median_abs_divergence),
            "max_abs_divergence": float(self.max_abs_divergence),
            "pearson_correlation": float(self.pearson_correlation),
            "sign_agreement_rate": float(self.sign_agreement_rate),
            "threshold_used": float(self.threshold_used),
            "diagnostics": dict(self.diagnostics),
        }
        if include_per_cell and self.per_cell_divergence is not None:
            out["per_cell_divergence"] = self.per_cell_divergence.tolist()
        if include_per_cell and self.flagged_mask is not None:
            out["flagged_mask"] = self.flagged_mask.astype(int).tolist()
        return out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cate_vs_gwr_divergence(
    gwr_beta: np.ndarray,
    cate: np.ndarray,
    *,
    treatment: str = "T",
    cate_std: Optional[np.ndarray] = None,
    gwr_std: Optional[np.ndarray] = None,
    threshold_sigma: float = 2.0,
    absolute_threshold: Optional[float] = None,
) -> DivergenceReport:
    """Per-cell divergence between GWR β and Stage-3 CATE.

    Threshold logic:

    - If ``absolute_threshold`` is provided, flag cells with
      ``|β − τ| > absolute_threshold``.
    - Else if posterior std arrays are provided, flag cells with
      ``|β − τ| > threshold_sigma · √(σ_β² + σ_τ²)``.
    - Else flag cells with ``|β − τ| > threshold_sigma · std(β − τ)``.
    """
    gwr_beta = np.asarray(gwr_beta, dtype=np.float64).ravel()
    cate = np.asarray(cate, dtype=np.float64).ravel()
    if gwr_beta.shape != cate.shape:
        raise ValueError(
            f"GWR β shape {gwr_beta.shape} != CATE shape {cate.shape}"
        )
    n = gwr_beta.size
    finite = np.isfinite(gwr_beta) & np.isfinite(cate)
    if not finite.any():
        raise ValueError("All cells are non-finite in either GWR β or CATE")
    diff = gwr_beta - cate
    abs_diff = np.abs(diff)

    if absolute_threshold is not None:
        thr = float(absolute_threshold)
    elif cate_std is not None and gwr_std is not None:
        cs = np.asarray(cate_std, dtype=np.float64).ravel()
        gs = np.asarray(gwr_std, dtype=np.float64).ravel()
        per_cell_thr = threshold_sigma * np.sqrt(gs ** 2 + cs ** 2)
        # Use a scalar summary thr but flag per-cell against per_cell_thr
        thr = float(np.nanmedian(per_cell_thr))
        flagged = abs_diff > per_cell_thr
        flagged[~finite] = False
        report_thr = thr
        return _build_report(
            treatment, n, gwr_beta, cate, diff, abs_diff, flagged,
            report_thr, finite, sigma_aware=True,
        )
    else:
        thr = float(threshold_sigma * np.nanstd(diff[finite]))

    flagged = (abs_diff > thr) & finite
    return _build_report(
        treatment, n, gwr_beta, cate, diff, abs_diff, flagged, thr, finite,
        sigma_aware=False,
    )


def _build_report(
    treatment: str,
    n: int,
    gwr_beta: np.ndarray,
    cate: np.ndarray,
    diff: np.ndarray,
    abs_diff: np.ndarray,
    flagged: np.ndarray,
    thr: float,
    finite: np.ndarray,
    sigma_aware: bool,
) -> DivergenceReport:
    finite_n = int(finite.sum())
    sign_agree = (
        (np.sign(gwr_beta[finite]) == np.sign(cate[finite])).mean()
        if finite_n else float("nan")
    )
    pearson = _safe_corr(gwr_beta[finite], cate[finite])
    flagged_count = int(flagged.sum())
    return DivergenceReport(
        treatment=treatment,
        cell_count=n,
        flagged_count=flagged_count,
        flagged_fraction=float(flagged_count / max(1, n)),
        mean_abs_divergence=float(np.nanmean(abs_diff[finite])),
        median_abs_divergence=float(np.nanmedian(abs_diff[finite])),
        max_abs_divergence=float(np.nanmax(abs_diff[finite])),
        pearson_correlation=float(pearson),
        sign_agreement_rate=float(sign_agree),
        threshold_used=float(thr),
        per_cell_divergence=diff,
        flagged_mask=flagged,
        diagnostics={
            "finite_cell_count": finite_n,
            "non_finite_cell_count": int(n - finite_n),
            "threshold_mode": "per_cell_sigma" if sigma_aware else "global_sigma",
        },
    )


def divergence_audit_for_all(
    gwr_betas: Dict[str, np.ndarray],
    cates: Dict[str, np.ndarray],
    *,
    threshold_sigma: float = 2.0,
    cate_stds: Optional[Dict[str, np.ndarray]] = None,
    gwr_stds: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, DivergenceReport]:
    """Run :func:`cate_vs_gwr_divergence` for every treatment present in both.

    Treatments missing from either dict are skipped silently (caller can
    diff the keys to surface coverage gaps).
    """
    out: Dict[str, DivergenceReport] = {}
    common = sorted(set(gwr_betas) & set(cates))
    for tr in common:
        out[tr] = cate_vs_gwr_divergence(
            gwr_betas[tr], cates[tr],
            treatment=tr,
            cate_std=(cate_stds or {}).get(tr),
            gwr_std=(gwr_stds or {}).get(tr),
            threshold_sigma=threshold_sigma,
        )
    return out
