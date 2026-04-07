"""
Differentiable surrogate models for SPARC V2.

Neural approximations of the V1 base models (GWR, GWRF, GGPGAM) that
participate in end-to-end gradient-based training under the joint loss.

Each surrogate preserves the mathematical structure of the original:
  - DifferentiableGWR   → spatially-varying coefficients + local intercept
                           (matches GWR: y = Σ β_k(loc) · X_k + β_0(loc))
  - DifferentiableGWRF  → FiLM-modulated nonlinear network
                           (matches GWRF: per-location nonlinear transform
                            via spatial gating, analogous to blending local
                            random forests with kernel weights)
  - DifferentiableGGPGAM → spatial smoothers + feature smoothers +
                            spatial×feature interactions
                           (matches GGPGAM: s(x) + s(y) + Σ s(f_k) +
                            Σ te(x,f_k) + Σ te(y,f_k))

Surrogates are validated against the true V1 base model outputs via
``validate_surrogates()`` — required R² > 0.95 before joint training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable GWR
# ---------------------------------------------------------------------------

class DifferentiableGWR(nn.Module):
    """
    Approximates GWR by learning spatially-varying coefficients AND a
    spatially-varying intercept as a function of location.

    Original GWR solves per-location weighted least squares:
        y_i = β_0(loc_i) + Σ_k β_k(loc_i) · X_ik
    with kernel-weighted neighbors.  This surrogate learns β(loc)
    through a deeper network with residual connections for smooth
    spatial variation.

    Interpretability
    ----------------
    ``beta (N, n_vars)`` maps directly to GWR coefficient surfaces.
    ``intercept (N,)`` maps to the local intercept surface.
    """

    def __init__(
        self,
        n_vars: int,
        n_spatial_features: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars

        # Deeper network with residual connection for spatial smoothness
        self.spatial_embed = nn.Sequential(
            nn.Linear(n_spatial_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Output: n_vars coefficients + 1 intercept
        self.coeff_head = nn.Linear(hidden_dim, n_vars + 1)

        # Small init so initial coefficients ≈ 0
        nn.init.xavier_uniform_(self.coeff_head.weight, gain=0.01)
        nn.init.zeros_(self.coeff_head.bias)

    def forward(
        self, X: torch.Tensor, spatial_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X : (N, n_vars) — predictor values
        spatial_features : (N, n_spatial_features) — sinusoidal encoding

        Returns
        -------
        y_pred : (N,)
        beta   : (N, n_vars) — local coefficients (excludes intercept)
        """
        h = self.spatial_embed(spatial_features)          # (N, H)
        h = h + self.res_block(h)                         # residual
        coeff_all = self.coeff_head(h)                    # (N, n_vars + 1)
        beta = coeff_all[:, :self.n_vars]                 # (N, n_vars)
        intercept = coeff_all[:, self.n_vars]             # (N,)
        y_pred = (X * beta).sum(dim=-1) + intercept       # (N,)
        return y_pred, beta

    @torch.no_grad()
    def coefficient_map(
        self, spatial_features: torch.Tensor, predictor_idx: int
    ) -> np.ndarray:
        """Return the spatially-varying coefficient for one predictor."""
        self.eval()
        h = self.spatial_embed(spatial_features)
        h = h + self.res_block(h)
        coeff_all = self.coeff_head(h)
        return coeff_all[:, predictor_idx].cpu().numpy()


# ---------------------------------------------------------------------------
# Differentiable GWRF
# ---------------------------------------------------------------------------

class DifferentiableGWRF(nn.Module):
    """
    Approximates GWRF using FiLM (Feature-wise Linear Modulation).

    Original GWRF fits separate random forests at each location and
    blends predictions from k nearest local models using kernel
    weights — each location effectively gets its own nonlinear
    feature transform.

    This surrogate uses FiLM conditioning: the spatial context
    generates per-neuron scale (γ) and shift (β) parameters that
    *modulate* the feature representation at each layer, so every
    location gets a different effective nonlinear transform.
    This is closer to "local expert blending" than simple
    concatenation.
    """

    def __init__(
        self,
        n_vars: int,
        n_spatial_features: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        # Spatial conditioning — generates FiLM parameters
        self.film_generator = nn.Sequential(
            nn.Linear(n_spatial_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # γ, β for each of 2 FiLM layers → 4 * hidden_dim params
        self.film_params = nn.Linear(hidden_dim, hidden_dim * 4)

        # Feature processing layers (modulated by spatial FiLM)
        self.feat_layer1 = nn.Linear(n_vars, hidden_dim)
        self.feat_ln1 = nn.LayerNorm(hidden_dim)
        self.feat_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.feat_ln2 = nn.LayerNorm(hidden_dim)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, X: torch.Tensor, spatial_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (N, n_vars)
        spatial_features : (N, n_spatial_features)

        Returns
        -------
        y_pred : (N,)
        """
        # Generate FiLM parameters from spatial context
        spatial_ctx = self.film_generator(spatial_features)   # (N, H)
        film = self.film_params(spatial_ctx)                  # (N, 4H)
        H = film.shape[1] // 4
        gamma1, beta1, gamma2, beta2 = film.split(H, dim=1)

        # Layer 1: feature encoding + FiLM modulation
        h = self.feat_layer1(X)                               # (N, H)
        h = self.feat_ln1(h)
        h = (1 + gamma1) * h + beta1                          # FiLM
        h = F.gelu(h)

        # Layer 2: deeper features + FiLM modulation
        h = self.feat_layer2(h)                               # (N, H)
        h = self.feat_ln2(h)
        h = (1 + gamma2) * h + beta2                          # FiLM
        h = F.gelu(h)

        return self.predictor(h).squeeze(-1)                  # (N,)


# ---------------------------------------------------------------------------
# Differentiable GGPGAM
# ---------------------------------------------------------------------------

class DifferentiableGGPGAM(nn.Module):
    """
    Approximates GGPGAM with the correct additive structure including
    spatial terms and spatial×feature interactions.

    Original GGPGAM (PyGAM) formula:
        y = s(x_coord) + s(y_coord)
          + Σ_k s(feature_k)
          + Σ_k te(x_coord, feature_k) + Σ_k te(y_coord, feature_k)

    This surrogate mirrors that structure:
      - 2 spatial smoothers: f_x(spatial) + f_y(spatial)
      - n_vars feature smoothers: Σ f_k(X_k)
      - 2 * n_vars interaction nets: Σ g_xk(x_coord, X_k) + g_yk(y_coord, X_k)
      - learnable intercept

    The spatial features input is the sinusoidal encoding; we extract
    the first 2 raw coordinate dimensions for interaction terms.
    """

    def __init__(
        self,
        n_vars: int,
        n_spatial_features: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_spatial_features = n_spatial_features

        # --- Spatial main effect smoothers: s(x_coord), s(y_coord) ---
        # Use first few dims of spatial encoding as proxy for s(x), s(y)
        spatial_input_dim = min(n_spatial_features, 16)  # first 16 frequencies
        self.spatial_smoother = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # --- Feature main effect smoothers: s(f_k) per variable ---
        self.shape_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_vars)
        ])

        # --- Spatial × feature interaction terms: te(coord, f_k) ---
        # Each interaction takes (spatial_encoding_subset, feature_k) as input
        interaction_input_dim = spatial_input_dim + 1
        self.interactions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(interaction_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_vars)
        ])

        self.intercept = nn.Parameter(torch.zeros(1))

        # Small-init output layers so all additive terms start near zero
        for net in [self.spatial_smoother]:
            nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
            nn.init.zeros_(net[-1].bias)
        for net in list(self.shape_functions) + list(self.interactions):
            nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
            nn.init.zeros_(net[-1].bias)

    def forward(
        self, X: torch.Tensor, spatial_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (N, n_vars) — predictor features
        spatial_features : (N, n_spatial_features) — sinusoidal encoding

        Returns
        -------
        y_pred : (N,)
        """
        # Spatial main effects: s(x) + s(y) via spatial encoding
        spatial_input_dim = min(self.n_spatial_features, 16)
        spatial_sub = spatial_features[:, :spatial_input_dim]
        spatial_effect = self.spatial_smoother(spatial_sub).squeeze(-1)  # (N,)

        # Feature main effects: Σ s(f_k)
        feature_effects = [
            f(X[:, k : k + 1]).squeeze(-1)
            for k, f in enumerate(self.shape_functions)
        ]

        # Spatial × feature interactions: Σ te(spatial, f_k)
        interaction_effects = []
        for k, g in enumerate(self.interactions):
            # Concatenate spatial subset with feature k
            interaction_input = torch.cat(
                [spatial_sub, X[:, k : k + 1]], dim=-1
            )
            interaction_effects.append(g(interaction_input).squeeze(-1))

        # Additive combination (preserves GAM structure)
        y_pred = (
            self.intercept
            + spatial_effect
            + sum(feature_effects)
            + sum(interaction_effects)
        )
        return y_pred  # (N,)

    @torch.no_grad()
    def partial_effect(
        self, X: torch.Tensor, predictor_idx: int, n_grid: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the partial effect curve for one predictor.

        Parameters
        ----------
        X : (N, n_vars) — used only to determine the range of predictor_idx
        predictor_idx : int
        n_grid : int — number of grid points

        Returns
        -------
        x_range : (n_grid,) ndarray
        effect  : (n_grid,) ndarray
        """
        self.eval()
        col = X[:, predictor_idx]
        x_min, x_max = col.min().item(), col.max().item()
        x_range = torch.linspace(x_min, x_max, n_grid, device=X.device).unsqueeze(-1)
        effect = self.shape_functions[predictor_idx](x_range).squeeze(-1)
        return x_range.squeeze(-1).cpu().numpy(), effect.cpu().numpy()


# ---------------------------------------------------------------------------
# Surrogate validation
# ---------------------------------------------------------------------------

def validate_surrogates(
    surrogates: dict[str, nn.Module],
    true_predictions: dict[str, torch.Tensor],
    X: torch.Tensor,
    spatial_features: torch.Tensor,
    threshold: float = 0.95,
) -> dict[str, dict]:
    """
    Validate each surrogate against its true base model predictions.

    Parameters
    ----------
    surrogates : dict
        ``{'gwr': DifferentiableGWR, 'gwrf': DifferentiableGWRF,
           'ggpgam': DifferentiableGGPGAM}``
    true_predictions : dict
        ``{'gwr': Tensor(N,), 'gwrf': ..., 'ggpgam': ...}``
        from the fitted V1 base models.
    X : (N, n_vars) predictor tensor
    spatial_features : (N, n_spatial_features) spatial encoding tensor
    threshold : float — minimum R² to pass.

    Returns
    -------
    dict per model with ``r2``, ``passed`` (bool), ``status`` string.
    """
    results = {}

    for name, surrogate in surrogates.items():
        surrogate.eval()
        with torch.no_grad():
            if name == "ggpgam":
                pred = surrogate(X, spatial_features)
            else:
                out = surrogate(X, spatial_features)
                pred = out[0] if isinstance(out, tuple) else out

        y_true = true_predictions[name]
        ss_res = ((y_true - pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()
        passed = r2 >= threshold

        status = "PASS" if passed else "FAIL"
        results[name] = {"r2": r2, "passed": passed, "status": status}
        print(f"Surrogate {name}: R²={r2:.4f} vs true model [{status}]")

    return results
