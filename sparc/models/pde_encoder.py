"""
PDE-informed physics encoder for SPARC V3 neural meta-learner.

Decomposes the physics representation into two sub-networks following
PDE solution structure:

  T = w(s) · T_p  +  (1 − w(s)) · T_h

  T_p  (source-driven):  Particular solution from a GELU MLP.
       Inputs: solar forcing proxies (albedo, impervious, anthropogenic).

  T_h  (harmonic):        Homogeneous solution from a SIREN network.
       Inputs: spatial coordinates and morphological features.
       SIREN's sinusoidal activations naturally represent solutions
       to Laplace's equation (∇²T_h = 0).

  w(s) (blend weight):    Spatially-varying (N, 1) field from a small
       coordinate MLP.  Near-water/suburban → w ≈ 0 (boundary-dominated).
       Dense urban → w ≈ 1 (source-dominated).  NOT a global scalar.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SIRENLayer(nn.Module):
    """Sinusoidal activation with SIREN-specific initialisation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        self.layer_norm = nn.LayerNorm(in_dim)

        if is_first:
            nn.init.uniform_(self.linear.weight, -1 / in_dim, 1 / in_dim)
        else:
            bound = math.sqrt(6 / in_dim) / omega
            nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return torch.sin(self.omega * self.linear(x))


class SourceDrivenEncoder(nn.Module):
    """
    GELU MLP for the particular solution T_p.

    Inputs are source-term proxies: albedo, impervious fraction,
    anthropogenic heat, solar-forcing-related features.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HarmonicEncoder(nn.Module):
    """
    SIREN network for the homogeneous solution T_h.

    SIREN activations naturally represent smooth harmonic functions
    (solutions to ∇²T_h = 0), making the architecture structurally
    biased toward physically valid boundary-dominated solutions.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega: float = 30.0,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        layers = [SIRENLayer(in_dim, out_dim, omega=omega, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SIRENLayer(out_dim, out_dim, omega=omega))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BlendWeightNet(nn.Module):
    """
    Small MLP producing spatially-varying blend weight w(s) ∈ (0, 1).

    Input: all physics features (the network learns to identify
    source-dominated vs boundary-dominated regions from the same
    feature set).

    Output: (N, 1) sigmoid-bounded per-point weight.
      w(s) → 1 : source-dominated (dense urban core, high impervious)
      w(s) → 0 : boundary-dominated (near water, suburban fringe)
    """

    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Initialise with small random weights for spatial variation
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.1)
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))  # (N, 1)


class PDEInformedPhysicsEncoder(nn.Module):
    """
    PDE-informed physics encoder combining source-driven and harmonic
    sub-encoders with a spatially-varying blend weight.

    Parameters
    ----------
    n_physics_features : int
        Total number of physics input features (original + derivatives).
    out_dim : int
        Output dimension (matches hidden_dim of meta-learner).
    omega : float
        SIREN frequency parameter for harmonic encoder.
    """

    def __init__(
        self,
        n_physics_features: int,
        out_dim: int,
        omega: float = 30.0,
    ) -> None:
        super().__init__()

        self.source_enc = SourceDrivenEncoder(
            in_dim=n_physics_features, out_dim=out_dim,
        )
        self.harmonic_enc = HarmonicEncoder(
            in_dim=n_physics_features, out_dim=out_dim, omega=omega,
        )
        self.blend_net = BlendWeightNet(
            in_dim=n_physics_features,
        )

    def forward(
        self,
        physics_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        physics_feats : (N, n_physics_features) — all physics inputs

        Returns
        -------
        encoded : (N, out_dim) — blended physics representation
        w_source : (N, 1) — spatial blend weight field (for diagnostics)
        """
        T_p = self.source_enc(physics_feats)        # (N, out_dim)
        T_h = self.harmonic_enc(physics_feats)       # (N, out_dim)
        w = self.blend_net(physics_feats)             # (N, 1)

        encoded = w * T_p + (1.0 - w) * T_h          # (N, out_dim)

        return encoded, w
