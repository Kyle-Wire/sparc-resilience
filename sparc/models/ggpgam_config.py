"""
GGPGAMConfig — validated configuration dataclass for GGPGAM_SVC.

Mirrors the GWENConfig pattern.  physics_signs moves from a post-instantiation
attribute assignment into the constructor, closing the split-interface gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GGPGAMConfig:
    """Validated configuration for a GGPGAM_SVC run.

    Parameters
    ----------
    n_splines:
        Spline grid size for feature smoothers.  Must be > 0.
    n_spatial_bases:
        Basis dimension for spatial smoothers s(x) + s(y) [+ s(z)].
        Must be > 0.
    lam:
        GAM smoothing parameter.  Must be > 0.
    clamp_features:
        When True, feature values are clamped to [min, max] seen at
        fit time before prediction.
    physics_signs:
        Dict mapping predictor names to -1 | 0 | +1 sign constraints.
        ``None`` means no constraints are applied.
    """

    n_splines: int = 5
    n_spatial_bases: int = 10
    lam: float = 0.6
    clamp_features: bool = False
    physics_signs: Optional[Dict] = None

    def __post_init__(self) -> None:
        if self.n_splines <= 0:
            raise ValueError(
                f"n_splines must be > 0, got {self.n_splines}"
            )
        if self.n_spatial_bases <= 0:
            raise ValueError(
                f"n_spatial_bases must be > 0, got {self.n_spatial_bases}"
            )
        if self.lam <= 0:
            raise ValueError(
                f"lam must be > 0, got {self.lam}"
            )
