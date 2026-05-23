"""
GWRFConfig — validated configuration dataclass for GWRFModel.

Mirrors the GWENConfig pattern.  Validation in __post_init__ ensures
a GWRFConfig is always in a usable state before it reaches the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GWRFConfig:
    """Validated configuration for a GWRFModel run.

    Parameters
    ----------
    n_estimators:
        Number of trees in each local random forest.  Must be > 0.
    k_neighbors:
        Neighbourhood size for adaptive bandwidth.  Must be > 0.
    min_samples_leaf:
        Minimum samples per leaf node.  Must be > 0.
    n_jobs:
        Parallelism for local RF fitting.  -1 means all cores.
    subsample_fraction:
        Fraction of locations at which to fit local models (0, 1].
        Mutually exclusive with subsample_n.
    subsample_n:
        Absolute count of fit locations.  Mutually exclusive with
        subsample_fraction.
    kernel_field:
        Optional KernelField for Phase-5a matrix-aware geometry.
    """

    n_estimators: int = 100
    k_neighbors: int = 100
    min_samples_leaf: int = 5
    n_jobs: int = -1
    subsample_fraction: Optional[float] = None
    subsample_n: Optional[int] = None
    kernel_field: Optional[object] = None

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be > 0, got {self.n_estimators}"
            )
        if self.k_neighbors <= 0:
            raise ValueError(
                f"k_neighbors must be > 0, got {self.k_neighbors}"
            )
        if self.min_samples_leaf <= 0:
            raise ValueError(
                f"min_samples_leaf must be > 0, got {self.min_samples_leaf}"
            )
        if self.subsample_fraction is not None:
            if not (0 < self.subsample_fraction <= 1.0):
                raise ValueError(
                    f"subsample_fraction must be in (0, 1], got {self.subsample_fraction}"
                )
        if self.subsample_fraction is not None and self.subsample_n is not None:
            raise ValueError(
                "subsample_fraction and subsample_n are mutually exclusive; "
                "set at most one of them."
            )
