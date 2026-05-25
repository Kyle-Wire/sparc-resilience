"""
NeuralMetaConfig — validated configuration dataclass for SPARCMetaLearner.

Mirrors the GWENConfig pattern.  Collects all 11 constructor parameters
into one validated object, closing the 3-way config scatter (config dict,
_ArchConfig, bundle.py hard-coded assembly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NeuralMetaConfig:
    """Validated configuration for a SPARCMetaLearner.

    Parameters
    ----------
    n_base_models:
        Number of differentiable surrogate outputs (e.g. 3 for GWR/GWRF/GGPGAM).
    n_physics_features:
        Dimension of the physics input tensor.
    d_spatial:
        Dimension of the sinusoidal spatial encoding.
        Must be divisible by n_heads.
    hidden_dim:
        Hidden dimension for all MLP streams.  Must be > 0.
    dropout:
        Dropout rate.  Must be in [0, 1).
    thresholds:
        Exceedance probability thresholds for the exceedance heads.
    n_heads:
        Number of sparse spatial attention heads.  Must be > 0.
    max_neighbors:
        KNN neighbourhood size for sparse spatial attention.  Must be > 0.
    siren_omega:
        SIREN frequency parameter for the physics encoder.  Must be > 0.
    time_embed_dim:
        Temporal embedding dimension (0 = disabled).  Must be >= 0.
    init_bandwidth:
        Initial spatial attention decay scale.  Must be > 0.
    """

    n_base_models: int = 3
    n_physics_features: int = 4
    d_spatial: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1
    thresholds: Optional[List[float]] = None
    n_heads: int = 4
    max_neighbors: int = 128
    siren_omega: float = 30.0
    time_embed_dim: int = 0
    init_bandwidth: float = 1000.0
    geo_pe_dim: int = 0  # 0 = disabled; must be divisible by 4 when > 0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {self.hidden_dim}"
            )
        if self.d_spatial % self.n_heads != 0:
            raise ValueError(
                f"d_spatial ({self.d_spatial}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )
        if self.time_embed_dim < 0:
            raise ValueError(
                f"time_embed_dim must be >= 0, got {self.time_embed_dim}"
            )
        if self.n_heads <= 0:
            raise ValueError(
                f"n_heads must be > 0, got {self.n_heads}"
            )
        if self.siren_omega <= 0:
            raise ValueError(
                f"siren_omega must be > 0, got {self.siren_omega}"
            )
        if self.geo_pe_dim != 0 and self.geo_pe_dim % 4 != 0:
            raise ValueError(
                f"geo_pe_dim must be 0 (disabled) or divisible by 4, "
                f"got {self.geo_pe_dim}"
            )
