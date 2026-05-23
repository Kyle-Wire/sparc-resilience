"""
ProcessRateDomainConfig — validated configuration dataclass for ProcessRateNet.

Replaces the implicit dict contract {bounds, prior_mean, ...} with a typed,
boundary-checked object.  ProcessRateNet accepts either a
ProcessRateDomainConfig or a legacy plain dict (backward compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ProcessRateDomainConfig:
    """Validated domain configuration for ProcessRateNet.

    Parameters
    ----------
    bounds:
        (lo, hi) physical bounds for the process rate.  lo must be < hi
        and both must be > 0 for physical validity.
    prior_mean:
        Prior mean for the process rate.  Must satisfy lo < prior_mean < hi.
    name:
        Human-readable name for the process rate (e.g. 'spatial_responsiveness').
    units:
        Physical units string (for logging / artifacts).
    material_priors:
        Optional per-material prior coefficients dict.
    """

    bounds: Tuple[float, float]
    prior_mean: float
    name: str = "process_rate"
    units: str = "unknown"
    material_priors: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if lo >= hi:
            raise ValueError(
                f"bounds must satisfy lo < hi, got bounds=({lo}, {hi})"
            )
        if not (lo < self.prior_mean < hi):
            raise ValueError(
                f"prior_mean must satisfy bounds[0] < prior_mean < bounds[1]; "
                f"got prior_mean={self.prior_mean}, bounds=({lo}, {hi})"
            )

    def to_dict(self) -> dict:
        """Convert back to the legacy dict format ProcessRateNet accepts."""
        return {
            "bounds": list(self.bounds),
            "prior_mean": self.prior_mean,
            "name": self.name,
            "units": self.units,
            "material_priors": self.material_priors or {},
        }
