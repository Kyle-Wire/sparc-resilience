"""
Process rate auxiliary network for SPARC V2.

Domain-agnostic network that learns the spatially-varying physical process
rate from land-cover composition.  The same architecture serves all domains:

  UHI          → spatial responsiveness α (sensitivity to thermal gradients)
  Coastal      → hydraulic diffusivity κ (m²/s)
  Wildfire     → fire spread rate λ (m/s)
  Air quality  → turbulent diffusivity K (m²/s)

For UHI, α(s) represents the local sensitivity of temperature to spatial
thermal gradients — higher where the temperature field is actively adjusting
to boundary influences (transition zones, waterfront), lower where it has
reached local equilibrium (dense urban core).  In scenario simulation,
alpha_norm(s) modulates treatment effect magnitude: locations with higher
spatial responsiveness experience stronger intervention effects because
they are in active thermal adjustment zones.

Physical bounds are hard-enforced via sigmoid + log-space scaling — no
clamping, no gradient killing, always physically valid output.

The network is initialized near the domain prior mean and deviates where
data supports it.  Systematic deviation from the prior is a scientific
finding, not a training artifact.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class ProcessRateNet(nn.Module):
    """
    Learn the mapping:  land-cover composition → local process rate.

    Parameters
    ----------
    n_inputs : int
        Number of input features (land-cover fractions / surface props).
    domain_config : dict
        Must include ``bounds`` (2-list), ``prior_mean`` (float), and
        optionally ``material_priors`` (dict str→float).

    Output
    ------
    ``(N, 1)`` tensor — always positive, always within ``[bounds[0], bounds[1]]``.
    """

    def __init__(self, n_inputs: int, domain_config: dict, n_temporal_inputs: int = 0) -> None:
        super().__init__()

        self.bounds_lo = max(float(domain_config["bounds"][0]), 1e-8)
        self.bounds_hi = float(domain_config["bounds"][1])
        self.prior_mean = float(domain_config["prior_mean"])
        self.material_priors = domain_config.get("material_priors", {})
        self.domain_name = domain_config.get("name", "process_rate")
        self.units = domain_config.get("units", "unknown")
        self.n_temporal_inputs = n_temporal_inputs

        total_inputs = n_inputs + n_temporal_inputs
        hidden = 64

        self.network = nn.Sequential(
            nn.Linear(total_inputs, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Small init so initial output ≈ 0 → sigmoid ≈ 0.5
        # → rate ≈ geometric mean of bounds (close to prior_mean for
        # symmetric-in-log-space bounds).
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, land_cover_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        land_cover_features : (N, n_inputs)

        Returns
        -------
        rate : (N, 1)  always in ``[bounds_lo, bounds_hi]``
        """
        raw = self.network(land_cover_features)  # (N, 1) unbounded

        # Map through sigmoid → [0, 1] → log-linear interpolation in
        # physical-rate space so that the output is always valid.
        t = torch.sigmoid(raw)  # (N, 1) in [0, 1]
        log_lo = np.log(self.bounds_lo)
        log_hi = np.log(self.bounds_hi)
        log_rate = log_lo + t * (log_hi - log_lo)
        return torch.exp(log_rate)  # (N, 1) always positive, in bounds

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_mixture_prior(
        self,
        land_cover: torch.Tensor,
        material_table: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """
        Compute a physics-based mixture prior for the process rate.

        Each point's prior is a weighted sum of material diffusivities,
        where the weights come from land-cover fractions.

        Parameters
        ----------
        land_cover : (N, n_materials) — fractional coverage per material
        material_table : dict mapping material name → process rate value.
            Defaults to ``self.material_priors``.

        Returns
        -------
        prior : (N, 1)
        """
        if material_table is None:
            material_table = self.material_priors

        if not material_table:
            return torch.full(
                (land_cover.shape[0], 1),
                self.prior_mean,
                device=land_cover.device,
            )

        values = torch.tensor(
            list(material_table.values()),
            dtype=land_cover.dtype,
            device=land_cover.device,
        )

        n_materials = min(land_cover.shape[1], len(values))
        weights = land_cover[:, :n_materials]
        vals = values[:n_materials]

        # Weighted sum (linear mixing in physical units)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        prior = (weights * vals.unsqueeze(0)).sum(dim=1, keepdim=True) / weight_sum

        # Clamp to physical bounds
        prior = prior.clamp(min=self.bounds_lo, max=self.bounds_hi)
        return prior

    # ------------------------------------------------------------------
    @torch.no_grad()
    def material_validation_report(
        self,
        land_cover: torch.Tensor,
        land_cover_classes: list[str],
        literature_values: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Compare learned per-class mean α against literature values.

        Parameters
        ----------
        land_cover : (N, n_classes) fractional coverage
        land_cover_classes : names for each column
        literature_values : expected α per class from literature

        Returns
        -------
        List of dicts with ``class_name``, ``learned_mean``, ``literature``,
        ``ratio``, ``status``.
        """
        self.eval()
        # Get current predictions
        rates = self.forward(land_cover).squeeze(-1)  # (N,)

        report = []
        for idx, cls_name in enumerate(land_cover_classes):
            if idx >= land_cover.shape[1]:
                break
            col = land_cover[:, idx]
            # Weighted mean of rate by this class's fraction
            weight_sum = col.sum()
            if weight_sum < 1e-8:
                continue

            learned_mean = (rates * col).sum() / weight_sum
            lit = literature_values.get(cls_name)
            if lit is None:
                continue

            ratio = learned_mean.item() / lit
            status = "OK" if 0.5 <= ratio <= 2.0 else "DEVIATION"

            report.append({
                "class_name": cls_name,
                "learned_mean": learned_mean.item(),
                "literature": lit,
                "ratio": ratio,
                "status": status,
            })

        return report


class SourceTermNet(nn.Module):
    """
    Learn the mapping: physics features → local source/sink term S(x).

    The source term captures spatially-varying forcing in the PDE
    ``α∇²T = S``.  Unlike ProcessRateNet, S can be positive (heat
    source, e.g. impervious surfaces) or negative (heat sink, e.g.
    tree canopy evapotranspiration).

    Initialized near zero so the physics loss starts as before (∇²T ≈ 0)
    and the network learns deviations where data supports them.

    Parameters
    ----------
    n_inputs : int
        Number of physics input features.
    scale : float
        Soft output scale (tanh × scale). Controls the initial range
        of learnable source terms.  Set to a value consistent with
        α × ∇²T magnitudes in the domain (default 1.0).
    """

    def __init__(self, n_inputs: int, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        hidden = 64

        self.network = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Near-zero init so initial source ≈ 0 (matches prior behavior)
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, physics_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        physics_features : (N, n_inputs)

        Returns
        -------
        source : (N, 1) — unbounded source/sink term
        """
        raw = self.network(physics_features)
        return self.scale * torch.tanh(raw)
