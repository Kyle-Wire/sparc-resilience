"""
GWENConfig — validated configuration dataclass for GWENModel.

Replaces the 8-parameter constructor with a typed, boundary-checked contract.
Callers construct a GWENConfig, pass it to GWENModel.from_config(), and never
need to know which individual kwargs GWENModel accepts internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GWENConfig:
    """Validated configuration for a GWENModel run.

    All validation happens in ``__post_init__`` so a ``GWENConfig`` object is
    always in a usable state — callers never receive silent silently-clamped
    values.

    Parameters
    ----------
    k_neighbors:
        Number of nearest-neighbour locations used for each local ElasticNet.
        Must be > 0.
    cv_folds:
        Cross-validation folds for the global ElasticNetCV.  Must be >= 2.
    n_alphas:
        Grid size for the ElasticNetCV alpha sweep.  Must be > 0.
    l1_ratios:
        L1/L2 mixing ratios to try during CV.
    sample_size:
        Optional cap on training samples.  ``None`` means use all data.
    quick_mode:
        When ``True`` the global alpha is reused for every local model
        (skips per-location CV — much faster, slightly less accurate).
    n_jobs:
        Parallelism for ElasticNetCV.  ``-1`` means all cores.
    selection_threshold:
        Fraction of local models that must select a feature for it to be
        included in the output.  Must be in ``[0.0, 1.0]``.
    spatial_cv_folds:
        Optional pre-built list of ``(train_idx, test_idx)`` tuples from
        Stage 0.  When supplied, overrides ``cv_folds`` for the global model.
    local_cv:
        When ``True`` run a full ElasticNetCV at each location (slow but
        maximally adaptive).  When ``False`` reuse global alpha.
    """

    k_neighbors: int = 50
    cv_folds: int = 5
    n_alphas: int = 100
    l1_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
    )
    sample_size: Optional[int] = None
    quick_mode: bool = False
    n_jobs: int = 1
    selection_threshold: float = 0.1
    spatial_cv_folds: Optional[list] = None
    local_cv: bool = False

    def __post_init__(self) -> None:
        if self.k_neighbors <= 0:
            raise ValueError(
                f"k_neighbors must be > 0, got {self.k_neighbors}"
            )
        if not (0.0 <= self.selection_threshold <= 1.0):
            raise ValueError(
                f"selection_threshold must be in [0, 1], got {self.selection_threshold}"
            )
        if self.n_alphas <= 0:
            raise ValueError(f"n_alphas must be > 0, got {self.n_alphas}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
