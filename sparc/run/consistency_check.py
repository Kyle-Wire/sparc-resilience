"""
Phase 5c — Base / surrogate consistency check.

After Stage 2 fits both the classical MGWR (`GWRModel`) and the
neural surrogate (`DifferentiableGWR`), we need a single number per
predictor that says *how faithfully the surrogate is approximating
the base model's local-coefficient surface*.  When that number is
low the surrogate's downstream causal δ extraction will silently
diverge from the base model.

This module exposes one entry point:

    compute_base_vs_surrogate_alignment(
        base_beta=…,        # (N, n_vars)  numpy
        surrogate_beta=…,   # (N, n_vars)  numpy
        feature_names=…,    # list[str], length n_vars
        warn_threshold=0.7,
    )

Returns the artifact-shaped payload (also persisted via
``ArtifactStore.write_struct(2, "gwr_base_vs_surrogate_alignment", …)``
when ``persist=True``).  Pure-NumPy, no torch dependency at the
audit site.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _cosine_similarity_columnwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column cosine similarity for two (N, K) matrices.

    Returns shape (K,).  Columns that are identically zero in either
    matrix yield 0.0 (no signal to align).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: base={a.shape} vs surrogate={b.shape}"
        )
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    out = np.zeros(a.shape[1], dtype=np.float64)
    mask = den > 1e-12
    out[mask] = num[mask] / den[mask]
    return out


def compute_base_vs_surrogate_alignment(
    base_beta: np.ndarray,
    surrogate_beta: np.ndarray,
    feature_names: list[str],
    *,
    warn_threshold: float = 0.7,
    persist: bool = False,
    producer: str = "stage2_consistency_check",
) -> dict:
    """Compute the per-predictor cosine alignment between the classical
    GWR β surface and the differentiable surrogate β surface.

    Parameters
    ----------
    base_beta, surrogate_beta : (N, n_vars)
        Per-location local coefficients from the two fits.  Must be
        aligned (same N, same predictor ordering).
    feature_names : list[str], length n_vars
        Predictor names — used to label the per-predictor scores.
    warn_threshold : float, default 0.7
        Predictors with cosine similarity below this value are flagged
        in ``warnings``.
    persist : bool, default False
        When True, writes the payload to the active ArtifactStore as
        ``stage=2, artifact_id="gwr_base_vs_surrogate_alignment"``.
    """
    if base_beta.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"base_beta n_features ({base_beta.shape[1]})"
        )

    cos = _cosine_similarity_columnwise(base_beta, surrogate_beta)

    per_predictor: dict[str, dict] = {}
    warnings_: list[dict] = []
    for i, name in enumerate(feature_names):
        score = float(cos[i])
        per_predictor[name] = {
            "cosine_similarity": score,
            "below_threshold": bool(score < warn_threshold),
            "base_beta_l2": float(np.linalg.norm(base_beta[:, i])),
            "surrogate_beta_l2": float(np.linalg.norm(surrogate_beta[:, i])),
        }
        if score < warn_threshold:
            warnings_.append({
                "predictor": name,
                "cosine_similarity": score,
                "threshold": float(warn_threshold),
                "message": (
                    f"Surrogate β surface for '{name}' diverges from "
                    f"classical GWR (cos={score:.3f} < {warn_threshold})"
                ),
            })

    summary = {
        "n_samples": int(base_beta.shape[0]),
        "n_predictors": int(base_beta.shape[1]),
        "warn_threshold": float(warn_threshold),
        "mean_cosine_similarity": float(np.mean(cos)),
        "min_cosine_similarity": float(np.min(cos)),
        "predictors_below_threshold": [w["predictor"] for w in warnings_],
        "n_warnings": len(warnings_),
    }

    payload = {
        "per_predictor": per_predictor,
        "summary": summary,
        "warnings": warnings_,
    }

    if persist:
        try:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is not None:
                store.write_struct(
                    "2", "gwr_base_vs_surrogate_alignment", payload,
                    producer=producer,
                    consumers=[
                        "report:/stage2/consistency",
                        "audit:/stage2/surrogate_alignment",
                    ],
                )
        except Exception as e:  # noqa: BLE001
            payload["persist_error"] = str(e)

    return payload


def collect_surrogate_beta(
    surrogate, spatial_features, n_vars: int,
) -> np.ndarray:
    """Helper: extract a ``(N, n_vars)`` β matrix from a
    ``DifferentiableGWR`` surrogate by calling ``coefficient_map`` for
    each predictor.

    The intercept column (index ``n_vars`` in ``coeff_head``) is
    excluded so the result is shape-aligned with
    ``GWRModel.coefficients_``.
    """
    import torch  # local import — keeps this module torch-optional otherwise

    surrogate.eval()
    cols: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(n_vars):
            cols.append(surrogate.coefficient_map(spatial_features, k))
    return np.stack(cols, axis=1)
