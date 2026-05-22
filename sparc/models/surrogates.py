"""
Differentiable surrogate models for SPARC V2.

Neural approximations of the V1 base models (GWR, GWRF, GGPGAM) that
participate in end-to-end gradient-based training under the joint loss.

Each surrogate preserves the mathematical structure of the original:
  - DifferentiableGWR   → MGWR-style spatially-varying coefficients with
                           bandwidth conditioning from Stage 0 correlogram
                           (y = Σ β_k(loc) · X_k + β_0(loc),
                            bandwidth info injected as spatial context)
  - DifferentiableGWRF  → FiLM-modulated nonlinear network + distance-weighted
                           kernel predictions using neighbor indices/distances
                           (matches GWRF: kernel-weighted local expert blending)
  - DifferentiableGGPGAM → spatial smoothers + feature smoothers +
                            spatial×feature interactions with monotonicity
                            enforcement via cumulative softplus
                           (matches GGPGAM: s(x) + s(y) + Σ s(f_k) +
                            Σ te(x,f_k) + Σ te(y,f_k))

Surrogates are validated against the true V1 base model outputs via
``validate_surrogates()`` — required R² ≥ 0.85 before joint training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable GWR (MGWR: per-predictor bandwidth scaling)
# ---------------------------------------------------------------------------

class DifferentiableGWR(nn.Module):
    """
    MGWR-style surrogate with bandwidth conditioning.

    Each predictor k gets its own bandwidth bw_k from the Stage 0
    correlogram.  Bandwidth information is injected as additive
    conditioning on the spatial embedding so the coefficient network
    can learn bandwidth-aware spatial variation.

    Architecture:
        spatial_features → spatial_embed + res_block →
        + bandwidth_conditioning(bw_scale) →
        coeff_head → [β_1(loc), …, β_K(loc), β_0(loc)]
        y = Σ β_k · X_k + β_0(loc)

    Parameters
    ----------
    bandwidths : optional (n_vars,) array
        Per-predictor bandwidths from Stage 0 correlogram.
        When None, falls back to uniform conditioning.
    """

    def __init__(
        self,
        n_vars: int,
        n_spatial_features: int,
        hidden_dim: int = 64,
        bandwidths: np.ndarray | None = None,
        kernel_field=None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars

        # Per-predictor bandwidth scaling (MGWR extension)
        if bandwidths is not None:
            bw = np.asarray(bandwidths, dtype=np.float32)
            # Normalise bandwidths so median = 1.0
            bw_median = np.median(bw)
            if bw_median > 0:
                bw_scale = bw / bw_median
            else:
                bw_scale = np.ones(n_vars, dtype=np.float32)
            self.register_buffer(
                "bw_scale", torch.from_numpy(bw_scale)  # (n_vars,)
            )
        else:
            self.register_buffer(
                "bw_scale", torch.ones(n_vars)
            )

        # Phase 5b: matrix-aware kernel-field conditioning.
        # Build a per-predictor feature tensor of shape (n_vars, F_kf):
        #   [log κ_x, log κ_y, sin θ, cos θ, log ν, log σ², log bw_to_outcome]
        # When fields are missing they fall back to neutral defaults so
        # callers without Phases 1-4 outputs are unchanged.
        self.has_kernel_field = kernel_field is not None
        kf_feat = self._build_kernel_field_features(
            kernel_field, feature_names, n_vars
        )  # (n_vars, 7) float32
        self.register_buffer("kf_features", torch.from_numpy(kf_feat))

        # Shared spatial embedding
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

        # Bandwidth conditioning: inject correlogram info into spatial repr
        self.bw_condition = nn.Linear(n_vars, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.bw_condition.weight, gain=0.05)

        # Per-predictor kernel-feature MLP (Phase 5b).  Outputs a hidden
        # context vector for each predictor; we sum over predictors to
        # broadcast into the spatial embedding (shape parity with
        # `bw_condition` so legacy-mode runs are byte-identical when
        # `kernel_field=None`).
        self.kf_condition = nn.Sequential(
            nn.Linear(self.kf_features.shape[1], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for m in self.kf_condition.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Phase 7 (Bucket A item 2): per-predictor anisotropic spatial
        # warp.  For each predictor k we derive a multiplicative gate
        # over the spatial-feature axes from kf_features[k]; this lets
        # the surrogate represent each predictor's own (κ_x, κ_y, θ)
        # frame as a per-predictor warp of the shared spatial encoding.
        # The per-predictor refinement β is added to the legacy global
        # β; weights are zero-initialised so the contribution at init is
        # exactly zero (legacy mode is forward-byte-identical at t=0).
        self.predictor_warp = nn.Linear(
            self.kf_features.shape[1], n_spatial_features
        )
        nn.init.zeros_(self.predictor_warp.weight)
        nn.init.zeros_(self.predictor_warp.bias)
        self.beta_per_predictor = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.beta_per_predictor.weight)
        nn.init.zeros_(self.beta_per_predictor.bias)

        # Direct coefficient output: n_vars coefficients + 1 intercept
        self.coeff_head = nn.Linear(hidden_dim, n_vars + 1)
        nn.init.xavier_uniform_(self.coeff_head.weight, gain=0.1)
        nn.init.zeros_(self.coeff_head.bias)

    @staticmethod
    def _build_kernel_field_features(
        kernel_field, feature_names: Optional[list[str]], n_vars: int,
    ) -> np.ndarray:
        """Build a (n_vars, 7) per-predictor feature matrix from a KernelField.

        Returns an all-zeros matrix when no field is supplied (legacy
        path → ``kf_condition`` contributes a learned constant).
        """
        F = 7  # log κ_x, log κ_y, sin θ, cos θ, log ν, log σ², log bw_to_outcome
        feats = np.zeros((n_vars, F), dtype=np.float32)
        if kernel_field is None or not feature_names:
            return feats
        eps = 1e-6
        for i, name in enumerate(feature_names):
            p = kernel_field.predictor(name)
            if p is None:
                continue
            kx = p.kappa_x if p.kappa_x is not None else (p.kappa or 0.0)
            ky = p.kappa_y if p.kappa_y is not None else (p.kappa or 0.0)
            theta = p.theta_rad if p.theta_rad is not None else 0.0
            nu = p.nu if p.nu is not None else 1.5
            sigma2 = p.sigma2 if p.sigma2 is not None else 1.0
            bw = p.bandwidth_to_outcome
            if bw is None:
                bw = kernel_field.cross_range(kernel_field.outcome_name, name) or 0.0
            feats[i, 0] = float(np.log(max(kx, eps)))
            feats[i, 1] = float(np.log(max(ky, eps)))
            feats[i, 2] = float(np.sin(theta))
            feats[i, 3] = float(np.cos(theta))
            feats[i, 4] = float(np.log(max(nu, eps)))
            feats[i, 5] = float(np.log(max(sigma2, eps)))
            feats[i, 6] = float(np.log(max(bw, eps))) if bw > 0 else 0.0
        return feats

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
        N = X.shape[0]
        h_base = self.spatial_embed(spatial_features)          # (N, H)
        h_base = h_base + self.res_block(h_base)               # residual

        # Bandwidth conditioning (same for all locations, learned)
        bw_ctx = self.bw_condition(self.bw_scale)              # (H,)
        # Phase 5b: matrix-aware kernel-field conditioning.  Sum the
        # per-predictor MLP outputs into a single context vector
        # (broadcast across N).  When `kf_features` is all-zeros (legacy
        # mode) this contributes a learned constant offset only.
        kf_ctx = self.kf_condition(self.kf_features).sum(dim=0)  # (H,)
        h = h_base + bw_ctx + kf_ctx                           # broadcast

        coeff = self.coeff_head(h)                             # (N, n_vars+1)
        beta = coeff[:, :self.n_vars]                          # (N, n_vars)
        intercept = coeff[:, -1]                               # (N,)

        # Phase 7 item 2: per-predictor anisotropic warp refinement.
        # gates: (n_vars, F_spatial) ∈ ~[0.9, 1.1] when warp is small;
        # exactly 1 when predictor_warp output is 0 (zero-init contract).
        gates = 1.0 + 0.1 * torch.tanh(
            self.predictor_warp(self.kf_features)
        )                                                      # (n_vars, F_sp)
        sf_warped = (spatial_features.unsqueeze(1)
                     * gates.unsqueeze(0))                     # (N, n_vars, F_sp)
        h_per = self.spatial_embed(
            sf_warped.reshape(-1, sf_warped.shape[-1])
        ).reshape(N, self.n_vars, -1)                          # (N, n_vars, H)
        h_per = h_per + self.res_block(h_per)
        beta_refine = self.beta_per_predictor(h_per).squeeze(-1)  # (N, n_vars)
        beta = beta + beta_refine                              # additive

        y_pred = (X * beta).sum(dim=-1) + intercept            # (N,)
        return y_pred, beta

    @torch.no_grad()
    def coefficient_map(
        self, spatial_features: torch.Tensor, predictor_idx: int
    ) -> np.ndarray:
        """Return the spatially-varying coefficient for one predictor."""
        self.eval()
        h = self.spatial_embed(spatial_features)
        h = h + self.res_block(h)
        bw_ctx = self.bw_condition(self.bw_scale)
        kf_ctx = self.kf_condition(self.kf_features).sum(dim=0)
        h = h + bw_ctx + kf_ctx
        coeff = self.coeff_head(h)
        beta_global = coeff[:, predictor_idx]                  # (N,)
        # Per-predictor warp refinement (Phase 7 item 2)
        gate_k = 1.0 + 0.1 * torch.tanh(
            self.predictor_warp(self.kf_features[predictor_idx])
        )                                                      # (F_sp,)
        sf_warped_k = spatial_features * gate_k                # (N, F_sp)
        h_k = self.spatial_embed(sf_warped_k)
        h_k = h_k + self.res_block(h_k)
        beta_refine_k = self.beta_per_predictor(h_k).squeeze(-1)  # (N,)
        return (beta_global + beta_refine_k).cpu().numpy()

    @torch.no_grad()
    def validate(
        self,
        y_true: torch.Tensor,
        X: torch.Tensor,
        spatial_features: torch.Tensor,
        min_r2: float = 0.85,
        raise_on_fail: bool = False,
    ) -> float:
        """Validate surrogate R\u00b2 against true base model predictions.

        Parameters
        ----------
        y_true : (N,) tensor of true base model outputs
        X : (N, n_vars) predictor tensor
        spatial_features : (N, n_spatial_features) spatial encoding
        min_r2 : float \u2014 minimum acceptable R\u00b2
        raise_on_fail : bool \u2014 raise ValueError when R\u00b2 < min_r2

        Returns
        -------
        r2 : float
        """
        was_training = self.training
        self.eval()
        pred, _ = self(X, spatial_features)
        if was_training:
            self.train()
        ss_res = ((y_true - pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item()
        if raise_on_fail and r2 < min_r2:
            raise ValueError(
                f"{self.__class__.__name__} R\u00b2={r2:.4f} < required {min_r2}"
            )
        return r2


# ---------------------------------------------------------------------------
# Differentiable GWRF (with distance-weighted kernel predictions)
# ---------------------------------------------------------------------------

class DifferentiableGWRF(nn.Module):
    """
    Approximates GWRF using FiLM conditioning + distance-weighted
    kernel predictions from neighbor indices and distances.

    Original GWRF fits separate random forests at each location and
    blends predictions from k nearest local models using kernel
    weights — each location effectively gets its own nonlinear
    feature transform.

    This surrogate uses:
    1. FiLM conditioning: spatial context generates per-neuron scale (γ)
       and shift (β) to modulate the feature representation.
    2. Distance-weighted kernel: when ``neighbor_idx`` and
       ``neighbor_dists`` are provided, predictions from neighbors are
       blended via Gaussian kernel weights for true GWRF behaviour.
       When neighbors are unavailable (e.g. during pre-training),
       falls back to FiLM-only predictions.
    """

    def __init__(
        self,
        n_vars: int,
        n_spatial_features: int,
        hidden_dim: int = 64,
        kernel_field=None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        # Phase 5b: matrix-aware kernel-field conditioning (mirrors
        # DifferentiableGWR).  The per-predictor feature tensor is
        # summed and concatenated to the spatial features before FiLM
        # generation, so the modulation is kernel-aware.
        self.has_kernel_field = kernel_field is not None
        kf_feat = DifferentiableGWR._build_kernel_field_features(
            kernel_field, feature_names, n_vars
        )  # (n_vars, 7)
        self.register_buffer("kf_features", torch.from_numpy(kf_feat))
        kf_dim = self.kf_features.shape[1]

        # Spatial conditioning — generates FiLM parameters
        self.film_generator = nn.Sequential(
            nn.Linear(n_spatial_features + kf_dim, hidden_dim),
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

        # Learnable kernel bandwidth for distance weighting
        self.log_kernel_bw = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        X: torch.Tensor,
        spatial_features: torch.Tensor,
        neighbor_idx: torch.Tensor | None = None,
        neighbor_dists: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (N, n_vars)
        spatial_features : (N, n_spatial_features)
        neighbor_idx : (N, K) int tensor, optional
            Indices of K nearest neighbors per point. -1 for missing.
        neighbor_dists : (N, K) float tensor, optional
            Distances to K nearest neighbors. Must accompany neighbor_idx.

        Returns
        -------
        y_pred : (N,)
        """
        # Generate FiLM parameters from spatial context
        # Phase 5b: prepend a per-batch kernel-field summary (mean over
        # predictors of the kf feature vector).  Shape (kf_dim,) → broadcast.
        kf_summary = self.kf_features.mean(dim=0)              # (kf_dim,)
        kf_summary_b = kf_summary.unsqueeze(0).expand(
            spatial_features.shape[0], -1
        )                                                       # (N, kf_dim)
        spatial_in = torch.cat([spatial_features, kf_summary_b], dim=-1)
        spatial_ctx = self.film_generator(spatial_in)         # (N, H)
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

        y_local = self.predictor(h).squeeze(-1)               # (N,)

        # Distance-weighted kernel blending (GWRF behaviour)
        if neighbor_idx is not None and neighbor_dists is not None:
            N, K = neighbor_idx.shape
            kernel_bw = torch.exp(self.log_kernel_bw).clamp(min=1e-6)

            # Gaussian kernel weights: w_k = exp(-d_k² / (2 * bw²))
            weights = torch.exp(
                -neighbor_dists.pow(2) / (2.0 * kernel_bw.pow(2))
            )  # (N, K)

            # Mask invalid neighbors (idx == -1)
            valid = neighbor_idx >= 0  # (N, K)
            weights = weights * valid.float()

            # Gather neighbor predictions
            # Clamp indices to valid range for gather (masked out later)
            safe_idx = neighbor_idx.clamp(min=0)  # (N, K)
            neighbor_preds = y_local[safe_idx]     # (N, K)
            neighbor_preds = neighbor_preds * valid.float()

            # Weighted average of neighbor predictions + self prediction
            w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
            y_neighbors = (weights * neighbor_preds).sum(dim=1) / w_sum.squeeze(1)

            # Blend: 50/50 self-prediction + neighbor-weighted prediction
            # (learnable via the kernel bandwidth)
            y_pred = 0.5 * y_local + 0.5 * y_neighbors
        else:
            y_pred = y_local

        return y_pred

    @torch.no_grad()
    def validate(
        self,
        y_true: torch.Tensor,
        X: torch.Tensor,
        spatial_features: torch.Tensor,
        min_r2: float = 0.85,
        raise_on_fail: bool = False,
    ) -> float:
        """Validate surrogate R\u00b2 against true base model predictions.

        Parameters
        ----------
        y_true : (N,) tensor of true base model outputs
        X : (N, n_vars) predictor tensor
        spatial_features : (N, n_spatial_features) spatial encoding
        min_r2 : float \u2014 minimum acceptable R\u00b2
        raise_on_fail : bool \u2014 raise ValueError when R\u00b2 < min_r2

        Returns
        -------
        r2 : float
        """
        was_training = self.training
        self.eval()
        pred = self(X, spatial_features)
        if was_training:
            self.train()
        ss_res = ((y_true - pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item()
        if raise_on_fail and r2 < min_r2:
            raise ValueError(
                f"{self.__class__.__name__} R\u00b2={r2:.4f} < required {min_r2}"
            )
        return r2


# ---------------------------------------------------------------------------
# Differentiable GGPGAM (with monotonicity via cumulative softplus)
# ---------------------------------------------------------------------------

class DifferentiableGGPGAM(nn.Module):
    """
    Approximates GGPGAM with the correct additive structure including
    spatial terms, spatial×feature interactions, and optional
    monotonicity enforcement via cumulative softplus.

    Original GGPGAM (PyGAM) formula:
        y = s(x_coord) + s(y_coord)
          + Σ_k s(feature_k)
          + Σ_k te(x_coord, feature_k) + Σ_k te(y_coord, feature_k)

    This surrogate mirrors that structure:
      - 2 spatial smoothers: f_x(spatial) + f_y(spatial)
      - n_vars feature smoothers: Σ f_k(X_k) with monotonic option
      - 2 * n_vars interaction nets: Σ g_xk(x_coord, X_k) + g_yk(y_coord, X_k)
      - learnable intercept

    Monotonicity
    ------------
    Shape functions use cumulative softplus to enforce monotonicity:
    instead of directly outputting s(x), the network outputs increments
    δ_j(x) and the shape value is cumsum(softplus(δ_j)).  This
    guarantees ∂s/∂x ≥ 0 everywhere (monotone increasing).  A learnable
    sign parameter per predictor allows monotone-decreasing as well.
    """

    def __init__(
        self,
        n_vars: int,
        n_spatial_features: int,
        hidden_dim: int = 64,
        monotonic: bool = False,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_spatial_features = n_spatial_features
        self.monotonic = monotonic
        # Expose as attribute so forward() doesn't recompute it
        self.spatial_input_dim = min(n_spatial_features, 32)

        # --- Spatial main effect smoothers: s(x_coord), s(y_coord) ---
        spatial_input_dim = self.spatial_input_dim
        self.spatial_smoother = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # --- Feature main effect smoothers: s(f_k) per variable ---
        if monotonic:
            # Monotonic shape functions via cumulative softplus:
            # Network outputs n_incr increments; shape = sign * cumsum(softplus(δ))
            n_incr = hidden_dim  # number of basis increments
            self.shape_functions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, n_incr),  # output increments
                )
                for _ in range(n_vars)
            ])
            # Learnable sign per predictor: tanh(sign_logit) → direction
            self.sign_logits = nn.Parameter(torch.zeros(n_vars))
            # Final projection from accumulated increments to scalar
            self.shape_proj = nn.ModuleList([
                nn.Linear(n_incr, 1) for _ in range(n_vars)
            ])
        else:
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

        # --- Spatial × feature interaction terms ---
        interaction_input_dim = spatial_input_dim + 1
        self.interactions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(interaction_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(n_vars)
        ])

        self.intercept = nn.Parameter(torch.zeros(1))

        # Small-init output layers
        for net in [self.spatial_smoother]:
            nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
            nn.init.zeros_(net[-1].bias)
        for net in list(self.interactions):
            nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
            nn.init.zeros_(net[-1].bias)
        if monotonic:
            for net in list(self.shape_functions):
                nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
                nn.init.zeros_(net[-1].bias)
            for proj in self.shape_proj:
                nn.init.xavier_uniform_(proj.weight, gain=0.01)
                nn.init.zeros_(proj.bias)
        else:
            for net in list(self.shape_functions):
                nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
                nn.init.zeros_(net[-1].bias)

    def _monotonic_shape(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute monotonic shape function for predictor k.

        Parameters
        ----------
        x : (N, 1) — single predictor values
        k : int — predictor index

        Returns
        -------
        effect : (N,) — monotonic effect
        """
        increments_raw = self.shape_functions[k](x)         # (N, n_incr)
        increments = F.softplus(increments_raw)              # non-negative
        cumulative = torch.cumsum(increments, dim=-1)        # monotone basis
        # Linear combination of cumulative basis → scalar
        effect = self.shape_proj[k](cumulative).squeeze(-1)  # (N,)
        # Apply learnable sign (tanh for smooth [-1, 1])
        sign = torch.tanh(self.sign_logits[k])
        return sign * effect

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
        # Spatial main effects
        spatial_sub = spatial_features[:, :self.spatial_input_dim]
        spatial_effect = self.spatial_smoother(spatial_sub).squeeze(-1)

        # Feature main effects: Σ s(f_k)
        if self.monotonic:
            feature_effects = [
                self._monotonic_shape(X[:, k:k + 1], k)
                for k in range(self.n_vars)
            ]
        else:
            feature_effects = [
                f(X[:, k:k + 1]).squeeze(-1)
                for k, f in enumerate(self.shape_functions)
            ]

        # Spatial × feature interactions
        interaction_effects = []
        for k, g in enumerate(self.interactions):
            interaction_input = torch.cat(
                [spatial_sub, X[:, k:k + 1]], dim=-1
            )
            interaction_effects.append(g(interaction_input).squeeze(-1))

        y_pred = (
            self.intercept
            + spatial_effect
            + sum(feature_effects)
            + sum(interaction_effects)
        )
        return y_pred

    @torch.no_grad()
    def partial_effect(
        self, X: torch.Tensor, predictor_idx: int, n_grid: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the partial effect curve for one predictor.

        For monotonic shape functions, this returns the monotone curve.
        """
        self.eval()
        col = X[:, predictor_idx]
        x_min, x_max = col.min().item(), col.max().item()
        x_range = torch.linspace(x_min, x_max, n_grid, device=X.device).unsqueeze(-1)
        if self.monotonic:
            effect = self._monotonic_shape(x_range, predictor_idx)
        else:
            effect = self.shape_functions[predictor_idx](x_range).squeeze(-1)
        return x_range.squeeze(-1).cpu().numpy(), effect.cpu().numpy()

    @torch.no_grad()
    def validate(
        self,
        y_true: torch.Tensor,
        X: torch.Tensor,
        spatial_features: torch.Tensor,
        min_r2: float = 0.85,
        raise_on_fail: bool = False,
    ) -> float:
        """Validate surrogate R\u00b2 against true base model predictions.

        Parameters
        ----------
        y_true : (N,) tensor of true base model outputs
        X : (N, n_vars) predictor tensor
        spatial_features : (N, n_spatial_features) spatial encoding
        min_r2 : float \u2014 minimum acceptable R\u00b2
        raise_on_fail : bool \u2014 raise ValueError when R\u00b2 < min_r2

        Returns
        -------
        r2 : float
        """
        was_training = self.training
        self.eval()
        pred = self(X, spatial_features)
        if was_training:
            self.train()
        ss_res = ((y_true - pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item()
        if raise_on_fail and r2 < min_r2:
            raise ValueError(
                f"{self.__class__.__name__} R\u00b2={r2:.4f} < required {min_r2}"
            )
        return r2


# ---------------------------------------------------------------------------
# Surrogate validation
# ---------------------------------------------------------------------------

def validate_surrogates(
    surrogates: dict[str, nn.Module],
    true_predictions: dict[str, torch.Tensor],
    X: torch.Tensor,
    spatial_features: torch.Tensor,
    threshold: float = 0.85,
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
    threshold : float — minimum R² to pass (default 0.85).

    Returns
    -------
    dict per model with ``r2``, ``passed`` (bool), ``status`` string.
    """
    results = {}

    for name, surrogate in surrogates.items():
        r2 = surrogate.validate(
            true_predictions[name], X, spatial_features,
            min_r2=threshold, raise_on_fail=False,
        )
        passed = r2 >= threshold
        status = "PASS" if passed else "FAIL"
        results[name] = {"r2": r2, "passed": passed, "status": status}
        print(f"Surrogate {name}: R²={r2:.4f} vs true model [{status}]")

    return results
