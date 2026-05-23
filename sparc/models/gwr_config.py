"""
GWRConfig — validated configuration dataclass for GWRModel.

Mirrors the GWENConfig pattern: a caller constructs a GWRConfig,
passes it to GWRModel.from_config(), and never needs to know which
individual kwargs GWRModel accepts internally.  Validation happens
in __post_init__ so a GWRConfig is always in a usable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

_VALID_KERNELS = ("gaussian", "exponential", "bisquare")


@dataclass
class GWRConfig:
    """Validated configuration for a GWRModel run.

    Parameters
    ----------
    bandwidth:
        Number of nearest neighbours for adaptive bandwidth.
        ``None`` means the caller will supply variable_bandwidths.
    variable_bandwidths:
        Per-predictor bandwidth dict (MGWR mode).  Mutually
        exclusive with a scalar bandwidth.
    kernel:
        Kernel function.  One of 'gaussian', 'exponential', 'bisquare'.
    coords_cols:
        Column names for the coordinate pair [x, y].
    alpha:
        L2 regularisation strength.  Must be >= 0.
    min_points:
        Minimum neighbourhood size.  Must be > 0.
    use_constrained_regression:
        Apply physics sign constraints during fitting.
    physics_priors:
        Dict mapping predictor names to prior coefficient values.
    sign_constraints:
        Dict mapping predictor names to -1 | 0 | +1.  Replaces the
        module-level defaults; ``None`` delegates to those defaults.
    kernel_field:
        Optional KernelField for Phase-5a matrix-aware geometry.
    """

    bandwidth: Optional[int] = None
    variable_bandwidths: Optional[Dict] = None
    kernel: str = "gaussian"
    coords_cols: Optional[List[str]] = None
    alpha: float = 0.1
    min_points: int = 50
    use_constrained_regression: bool = True
    physics_priors: Optional[Dict] = None
    sign_constraints: Optional[Dict] = None
    kernel_field: Optional[object] = None

    def __post_init__(self) -> None:
        if self.kernel not in _VALID_KERNELS:
            raise ValueError(
                f"kernel must be one of {_VALID_KERNELS}, got {self.kernel!r}"
            )
        if self.alpha < 0:
            raise ValueError(
                f"alpha must be >= 0, got {self.alpha}"
            )
        if self.min_points <= 0:
            raise ValueError(
                f"min_points must be > 0, got {self.min_points}"
            )
