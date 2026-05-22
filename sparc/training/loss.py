"""
Joint loss function for SPARC V2 neural training.

Seven-term unified objective that combines data fidelity, physics
constraints, and model consistency into a single differentiable loss.

All terms back-propagate end-to-end through the meta-learner and
differentiable surrogates to the raw input features.

Terms
-----
1. MSE — data fidelity (regression)
2. Cross-entropy — exceedance probability calibration
3. Physics residual — PDE satisfaction (normalized)
4. Prediction smoothness — penalize implausible spatial spikes
5. Alpha field smoothness — process rate spatial continuity
6. Alpha prior regularization — curriculum-decayed
7. Spatial neighborhood consistency
(8–11)  V3 multi-term PDE / BC / IC losses (optional)
12. JEPA self-supervised loss (optional, Phase 1)

(Surrogate fidelity is accepted but inactive — surrogates learn
end-to-end via the main loss terms, not by matching V1 outputs.)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Physics residual computation
# ---------------------------------------------------------------------------

def compute_physics_residual(
    T_pred: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_idx: torch.Tensor,
    source_term: torch.Tensor,
    resolution: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finite-difference Laplacian with normalized residual.

    Implements the steady-state heat equation in the standard form:

        α∇²T − S ≈ 0

    where α is the process rate (spatial responsiveness — sensitivity
    of the temperature field to spatial thermal gradients) and S is the
    source/sink term.  Alpha is normalized by its detached mean so the
    residual stays O(1) in z-score space.

    Parameters
    ----------
    T_pred : (N,) — predicted outcome field (z-scored)
    alpha : (N,) — process rate per point (physical units)
    neighbor_idx : (N, 4) long — indices [N, S, E, W]; -1 = missing
    source_term : (N,) — forcing / source term (z-scored)
    resolution : float — grid spacing in meters

    Returns
    -------
    normalized_residual : (M,) — physics residual / running std
    laplacian : (M,) — raw finite-difference Laplacian
    valid : (N,) bool mask — points with all 4 neighbors present
    """
    valid = (neighbor_idx != -1).all(dim=1)
    T = T_pred

    # Single batched gather for all 4 neighbours — (M, 4) — replaces 4 separate index ops
    T_nb = T[neighbor_idx[valid]]  # (M, 4)
    laplacian = (T_nb.sum(dim=-1) - 4 * T[valid]) / (resolution ** 2)

    # Normalize alpha by its detached mean so the PDE residual is O(1).
    # This avoids mixing physical units (m²/s) with z-score units and
    # prevents gradient explosions when alpha → bounds_lo.
    alpha_mean = alpha.detach().mean().clamp(min=1e-8)
    alpha_norm = alpha[valid] / alpha_mean
    physics_residual = alpha_norm * laplacian - source_term[valid]

    # Normalize by residual magnitude — keeps physics loss on same scale as MSE
    residual_scale = physics_residual.detach().std().clamp(min=1e-6)
    normalized_residual = physics_residual / residual_scale

    return normalized_residual, laplacian, valid


# ---------------------------------------------------------------------------
# Prior weight decay
# ---------------------------------------------------------------------------

def get_prior_weight(epoch: int, lambda_prior_base: float) -> float:
    """
    Decay prior regularization over training.

    Early:  prior guides alpha near mixture model.
    Late:   data trusted to pull alpha away from prior.

    Schedule:
      epoch < 10:  full weight
      epoch < 30:  full weight
      epoch >= 30: exponential decay toward 10% of base
    """
    if epoch < 30:
        return lambda_prior_base
    decay = 0.1 + 0.9 * math.exp(-0.05 * (epoch - 30))
    return lambda_prior_base * decay


# ---------------------------------------------------------------------------
# Joint loss
# ---------------------------------------------------------------------------

def sparc_joint_loss(
    # Predictions
    T_pred: torch.Tensor,
    exceedance_preds: list[torch.Tensor],
    y_true: torch.Tensor,
    thresholds: list[float],
    # Process rate
    alpha: torch.Tensor,
    alpha_prior: torch.Tensor,
    # Spatial structure
    neighbor_idx: torch.Tensor,
    source_term: torch.Tensor,
    resolution: float,
    # Surrogate alignment
    surrogate_preds: list[torch.Tensor],
    surrogate_targets: list[torch.Tensor],
    # Lambda weights (from CMA-ES or config)
    lambda_physics: float,
    lambda_smooth: float,
    lambda_alpha_smooth: float,
    lambda_prior: float,
    lambda_base: float,
    lambda_neighbor: float,
    # Training phase (for curriculum)
    epoch: int,
    # V3 PDE loss (optional — backward compatible)
    h_field: torch.Tensor | float | None = None,
    pde_loss_weights=None,
    alpha_prior_field: torch.Tensor | None = None,
    energy_residual: torch.Tensor | None = None,
    bc_specs: dict | None = None,
    water_mask: torch.Tensor | None = None,
    T_water: float | None = None,
    lambda_pde: float = 0.0,
    lambda_bc: float = 0.0,
    # V3 Initial condition (optional)
    T0: torch.Tensor | None = None,
    lambda_ic: float = 0.0,
    # V3 Temporal / transient PDE (optional)
    T_prev: torch.Tensor | None = None,
    dt_hours: float | None = None,
    dT_dt_observed: torch.Tensor | None = None,
    nocturnal_dT_dt: torch.Tensor | None = None,
    T_night: torch.Tensor | None = None,
    # JEPA self-supervised term (optional — Phase 1)
    jepa_h_pred: torch.Tensor | None = None,
    jepa_h_target: torch.Tensor | None = None,
    jepa_weights=None,
    lambda_jepa: float = 0.0,
    # JEPA Phase 2: scenario-distillation pair (predictor under action)
    jepa_scenario_h_pred: torch.Tensor | None = None,
    jepa_scenario_h_target: torch.Tensor | None = None,
    lambda_jepa_scenario: float = 0.0,
    # JEPA Phase 2: latent PDE consistency (Laplacian smoothness)
    jepa_latent_h: torch.Tensor | None = None,
    jepa_latent_neighbor_idx: torch.Tensor | None = None,
    jepa_latent_alpha: torch.Tensor | None = None,
    lambda_jepa_latent_pde: float = 0.0,
    # Precomputed sparse Laplacian (optional — accelerates full-N PDE eval)
    sparse_laplacian: torch.Tensor | None = None,
    valid_laplacian_mask: torch.Tensor | None = None,
    # Static-batch padding mask (optional — CU-9 CUDA Graph support)
    valid_mask: torch.Tensor | None = None,
    # Sheaf Laplacian (optional — topological regularisation)
    sheaf_delta: torch.Tensor | None = None,
    # Candidate A: per-surrogate fidelity weights from gate EMA feedback
    surr_fidelity_weights: list[float] | None = None,
    # Candidate C: alpha field classification scaffold
    alpha_class_targets: torch.Tensor | None = None,
    lambda_alpha_class: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the 8-term joint loss.

    Returns
    -------
    total : scalar Tensor — total loss for backward()
    loss_components : dict — each term's scalar value for logging
    """
    loss_components: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Data fidelity — regression MSE
    # ------------------------------------------------------------------
    _t = T_pred.squeeze()
    if valid_mask is not None:
        mse = F.mse_loss(_t[valid_mask], y_true[valid_mask])
    else:
        mse = F.mse_loss(_t, y_true)
    loss_components["mse"] = mse.item()

    # ------------------------------------------------------------------
    # 2. Exceedance probabilities — cross-entropy per threshold
    # ------------------------------------------------------------------
    ce = torch.tensor(0.0, device=T_pred.device)
    if exceedance_preds and thresholds:
        for exc_pred, thresh in zip(exceedance_preds, thresholds):
            exc_clamped = exc_pred.squeeze().clamp(1e-6, 1 - 1e-6)
            if not torch.isfinite(exc_clamped).all():
                continue  # skip if NaN/inf leaked through
            target = (y_true > thresh).to(dtype=exc_clamped.dtype)
            if valid_mask is not None:
                exc_clamped = exc_clamped[valid_mask]
                target = target[valid_mask]
            ce = ce + F.binary_cross_entropy(exc_clamped, target)
        ce = ce * 0.5
    loss_components["cross_entropy"] = ce.item()

    # ------------------------------------------------------------------
    # 3. Physics residual — heat diffusion PDE (or domain equivalent)
    # ------------------------------------------------------------------
    physics = torch.tensor(0.0, device=T_pred.device)
    smooth = torch.tensor(0.0, device=T_pred.device)
    alpha_smooth = torch.tensor(0.0, device=T_pred.device)

    has_neighbors = neighbor_idx is not None and neighbor_idx.numel() > 0

    # When V3 PDE loss is active, compute_pde_loss() already computes:
    #   • heat_diffusion  (≡ V2 Term 3, physics residual)
    #   • alpha_smooth    (≡ V2 Term 5)
    #   • alpha_prior     (≡ V2 Term 6)
    # Running those V2 terms simultaneously doubles the physics gradient.
    # They are skipped here and delegated to compute_pde_loss() (Term 9).
    _v3_active = lambda_pde > 0 and h_field is not None and has_neighbors

    if has_neighbors and lambda_physics > 0 and not _v3_active:
        normalized_residual, laplacian, valid = compute_physics_residual(
            T_pred.squeeze(), alpha.squeeze(), neighbor_idx, source_term, resolution,
        )
        if valid.sum() > 0:
            physics = lambda_physics * normalized_residual.pow(2).mean()

            # 4. Prediction smoothness
            smooth = lambda_smooth * laplacian.pow(2).mean()

            # 5. Alpha field smoothness — batched gather: one (M,4) op replaces 4 separate gathers
            a = alpha.squeeze()
            a_nb = a[neighbor_idx[valid]]  # (M, 4)
            alpha_lap = (a_nb.sum(dim=-1) - 4 * a[valid]) / (resolution ** 2)
            alpha_smooth = lambda_alpha_smooth * alpha_lap.pow(2).mean()

    loss_components["physics"] = physics.item()
    loss_components["smooth"] = smooth.item()
    loss_components["alpha_smooth"] = alpha_smooth.item()

    # ------------------------------------------------------------------
    # 6. Alpha prior regularization — decayed by curriculum
    #    Skipped in V3 mode: compute_pde_loss() Term 8 (alpha_prior) covers this.
    # ------------------------------------------------------------------
    prior_reg = torch.tensor(0.0, device=T_pred.device)
    if not _v3_active:
        prior_weight = get_prior_weight(epoch, lambda_prior)
        if valid_mask is not None:
            prior_reg = prior_weight * (alpha[valid_mask] - alpha_prior[valid_mask]).pow(2).mean()
        else:
            prior_reg = prior_weight * (alpha - alpha_prior).pow(2).mean()
    loss_components["alpha_prior"] = prior_reg.item()

    # ------------------------------------------------------------------
    # 7. Surrogate fidelity (Candidate A: gate-weight-scaled per surrogate)
    # ------------------------------------------------------------------
    surrogate_loss = torch.tensor(0.0, device=T_pred.device)
    if surrogate_preds and surrogate_targets:
        if surr_fidelity_weights is not None and len(surr_fidelity_weights) == len(surrogate_preds):
            surrogate_loss = lambda_base * sum(
                surr_fidelity_weights[k] * F.mse_loss(pred, target)
                for k, (pred, target) in enumerate(zip(surrogate_preds, surrogate_targets))
            ) / max(len(surrogate_preds), 1)
        else:
            surrogate_loss = lambda_base * sum(
                F.mse_loss(pred, target)
                for pred, target in zip(surrogate_preds, surrogate_targets)
            )
    loss_components["surrogate"] = surrogate_loss.item()

    # ------------------------------------------------------------------
    # 7b. Alpha classification scaffold (Candidate C: ProcessRateNet curriculum)
    # Penalises alpha deviating from the land-cover class centroids computed
    # during ProcessRateNet pre-training.  Decays via the alpha_class schedule
    # so it acts as a soft anchor in Stage A/B and releases in Stage C.
    # ------------------------------------------------------------------
    alpha_class_loss = torch.tensor(0.0, device=T_pred.device)
    if alpha_class_targets is not None and lambda_alpha_class > 0.0:
        _act = alpha_class_targets.to(T_pred.device).squeeze()
        _alp = alpha.squeeze()
        if valid_mask is not None:
            alpha_class_loss = lambda_alpha_class * F.mse_loss(
                _alp[valid_mask], _act[valid_mask]
            )
        else:
            alpha_class_loss = lambda_alpha_class * F.mse_loss(_alp, _act)
    loss_components["alpha_class"] = alpha_class_loss.item()

    # ------------------------------------------------------------------
    # 8. Spatial neighborhood consistency
    # ------------------------------------------------------------------
    neighborhood = torch.tensor(0.0, device=T_pred.device)
    if has_neighbors and lambda_neighbor > 0:
        point_loss = (T_pred.squeeze() - y_true).pow(2)
        neighbor_loss = torch.zeros_like(point_loss)
        for k in range(4):
            valid_k = neighbor_idx[:, k] != -1
            neighbor_loss[valid_k] += 0.25 * point_loss[neighbor_idx[valid_k, k]]
        combined = point_loss + 0.3 * neighbor_loss
        if valid_mask is not None:
            n_valid = valid_mask.float().sum().clamp(min=1)
            neighborhood = lambda_neighbor * (combined * valid_mask.float()).sum() / n_valid
        else:
            neighborhood = lambda_neighbor * combined.mean()
    loss_components["neighborhood"] = neighborhood.item()

    # ------------------------------------------------------------------
    # 9. V3 PDE loss (multi-term, staged curriculum)
    # ------------------------------------------------------------------
    pde_total = torch.tensor(0.0, device=T_pred.device)
    if lambda_pde > 0 and h_field is not None and has_neighbors:
        from sparc.physics.pde_loss import compute_pde_loss
        pde_total, pde_dict = compute_pde_loss(
            T_pred=T_pred.squeeze(),
            alpha=alpha,
            source_term=source_term,
            neighbor_idx=neighbor_idx,
            h=h_field,
            weights=pde_loss_weights,
            epoch=epoch,
            alpha_prior_field=alpha_prior_field,
            energy_residual=energy_residual,
            T_prev=T_prev,
            dt_hours=dt_hours,
            dT_dt_observed=dT_dt_observed,
            nocturnal_dT_dt=nocturnal_dT_dt,
            T_night=T_night,
            sparse_laplacian=sparse_laplacian,
            valid_laplacian_mask=valid_laplacian_mask,
        )
        pde_total = lambda_pde * pde_total
        for k, v in pde_dict.items():
            loss_components[k] = v

    loss_components["pde_total"] = pde_total.item()

    # ------------------------------------------------------------------
    # 10. V3 Boundary condition loss
    # ------------------------------------------------------------------
    bc_total = torch.tensor(0.0, device=T_pred.device)
    if lambda_bc > 0 and h_field is not None and has_neighbors:
        from sparc.physics.boundary_conditions import compute_bc_loss
        bc_total, bc_dict = compute_bc_loss(
            T=T_pred.squeeze(),
            neighbor_idx=neighbor_idx,
            h=h_field,
            bc_specs=bc_specs,
            water_mask=water_mask,
            T_water=T_water,
        )
        bc_total = lambda_bc * bc_total
        for k, v in bc_dict.items():
            loss_components[k] = v

    loss_components["bc_total"] = bc_total.item()

    # ------------------------------------------------------------------
    # 11. V3 Initial condition consistency
    # ------------------------------------------------------------------
    ic_total = torch.tensor(0.0, device=T_pred.device)
    if lambda_ic > 0 and T0 is not None:
        from sparc.physics.initial_conditions import (
            ic_consistency_loss, warmup_ic_schedule,
        )
        ic_raw = ic_consistency_loss(T_pred.squeeze(), T0)
        ic_total = warmup_ic_schedule(epoch, ic_raw)
        ic_total = lambda_ic * ic_total

    loss_components["ic_total"] = ic_total.item()

    # ------------------------------------------------------------------
    # 12. JEPA self-supervised loss (Phase 1, optional)
    # ------------------------------------------------------------------
    jepa_total = torch.tensor(0.0, device=T_pred.device)
    if (
        lambda_jepa > 0
        and jepa_h_pred is not None
        and jepa_h_target is not None
    ):
        from sparc.training.jepa_loss import jepa_loss as _jepa_loss
        jepa_raw, jepa_dict = _jepa_loss(
            h_pred=jepa_h_pred,
            h_target=jepa_h_target,
            weights=jepa_weights,
        )
        jepa_total = lambda_jepa * jepa_raw
        for k, v in jepa_dict.items():
            loss_components[k] = v

    loss_components["jepa_weighted"] = jepa_total.item()

    # ------------------------------------------------------------------
    # 13. JEPA Phase 2 — scenario-distillation (action-conditioned)
    # ------------------------------------------------------------------
    jepa_scenario_total = torch.tensor(0.0, device=T_pred.device)
    if (
        lambda_jepa_scenario > 0
        and jepa_scenario_h_pred is not None
        and jepa_scenario_h_target is not None
    ):
        from sparc.training.jepa_loss import scenario_distillation_loss
        sd_raw, sd_dict = scenario_distillation_loss(
            h_pred_perturbed=jepa_scenario_h_pred,
            h_target_perturbed=jepa_scenario_h_target,
            weights=jepa_weights,
        )
        jepa_scenario_total = lambda_jepa_scenario * sd_raw
        for k, v in sd_dict.items():
            loss_components[k] = v

    loss_components["jepa_scenario_weighted"] = jepa_scenario_total.item()

    # ------------------------------------------------------------------
    # 14. JEPA Phase 2 — latent PDE consistency (Laplacian smoothness)
    # ------------------------------------------------------------------
    jepa_latent_pde_total = torch.tensor(0.0, device=T_pred.device)
    if (
        lambda_jepa_latent_pde > 0
        and jepa_latent_h is not None
        and jepa_latent_neighbor_idx is not None
    ):
        from sparc.training.jepa_loss import latent_pde_consistency_loss
        lp_raw, lp_dict = latent_pde_consistency_loss(
            h=jepa_latent_h,
            neighbor_idx=jepa_latent_neighbor_idx,
            alpha=jepa_latent_alpha,
        )
        jepa_latent_pde_total = lambda_jepa_latent_pde * lp_raw
        for k, v in lp_dict.items():
            loss_components[k] = v

    loss_components["jepa_latent_pde_weighted"] = jepa_latent_pde_total.item()

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------
    total = (
        mse + ce + physics + smooth + alpha_smooth
        + prior_reg + surrogate_loss + alpha_class_loss + neighborhood
        + pde_total + bc_total + ic_total
        + jepa_total + jepa_scenario_total + jepa_latent_pde_total
    )
    loss_components["total"] = total.item()

    return total, loss_components


# ---------------------------------------------------------------------------
# Adaptive loss weight balancing — ReLoBRaLo
# ---------------------------------------------------------------------------

class AdaptiveLossWeighter:
    """Relative Loss Balancing with Residual Rescaling (ReLoBRaLo).

    Adjusts per-term lambda weights each training step so that every loss
    term contributes a comparable gradient signal — preventing any single
    term (typically the PDE residual in high-κ domains) from dominating
    the gradient flow and suppressing data-fidelity learning.

    The update rule tracks a running mean of each term's loss value and
    rescales its lambda so that all terms have equal *relative* contribution
    to the total loss:

        λ_i ← λ̄_i · (L̄_ref / L_i)

    where L̄_ref is the running mean of the reference term (default: "mse")
    and L_i is the current running mean of term i.

    Parameters
    ----------
    base_lambdas : dict[str, float]
        Initial lambda values keyed by loss component name.  These values
        act as scale anchors — the adaptive weights cannot drift more than
        ``max_scale_factor`` away from them.
    reference_term : str
        The loss term whose magnitude defines the target scale.  All other
        terms are rescaled to match this term's running mean.
        Defaults to "mse".
    ema_alpha : float
        Exponential moving average coefficient for tracking loss means.
        Smaller = smoother but slower to adapt.  Default 0.01.
    warmup_steps : int
        Number of steps to use base_lambdas unchanged while the EMA warms
        up.  Default 100.
    max_scale_factor : float
        Maximum ratio by which any lambda can deviate from its base value.
        Prevents runaway weights.  Default 10.0.
    enabled_terms : set[str] | None
        If provided, only adapt weights for these terms; others stay at
        their base value.  Defaults to all terms in base_lambdas.

    Usage
    -----
    >>> weighter = AdaptiveLossWeighter(
    ...     base_lambdas={"lambda_physics": 0.3, "lambda_smooth": 0.1, ...}
    ... )
    >>> # inside training loop:
    >>> total, components = sparc_joint_loss(**fixed_kwargs)
    >>> adapted = weighter.step(components)
    >>> # adapted["lambda_physics"] etc. are now adaptively scaled
    >>> # use them on the NEXT forward pass
    """

    def __init__(
        self,
        base_lambdas: dict[str, float],
        *,
        reference_term: str = "mse",
        ema_alpha: float = 0.01,
        warmup_steps: int = 100,
        max_scale_factor: float = 10.0,
        enabled_terms: set | None = None,
    ) -> None:
        self.base_lambdas = dict(base_lambdas)
        self.reference_term = reference_term
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.max_scale_factor = max_scale_factor
        self.enabled_terms = enabled_terms

        # Running EMA of each loss component (keyed by component name)
        self._ema: dict[str, float] = {}
        self._step = 0

        # Current adapted lambdas — starts equal to base
        self.lambdas: dict[str, float] = dict(base_lambdas)

    # Map from lambda parameter name → loss_components key
    # e.g. "lambda_physics" → "physics", "lambda_smooth" → "smooth"
    _LAMBDA_TO_COMPONENT: dict[str, str] = {
        "lambda_physics":      "physics",
        "lambda_smooth":       "smooth",
        "lambda_alpha_smooth": "alpha_smooth",
        "lambda_prior":        "alpha_prior",
        "lambda_base":         "surrogate",
        "lambda_neighbor":     "neighborhood",
        "lambda_pde":          "pde_total",
        "lambda_bc":           "bc_total",
        "lambda_jepa":         "jepa_weighted",
    }

    def step(self, loss_components: dict[str, float]) -> dict[str, float]:
        """Update EMA statistics and return adapted lambda dict.

        Parameters
        ----------
        loss_components : dict
            The ``loss_components`` dict returned by ``sparc_joint_loss``
            from the *previous* step (or the current step — one step
            behind is fine for the adaptive signal).

        Returns
        -------
        dict[str, float]
            Updated lambda values (same keys as ``base_lambdas``).
            Pass these as keyword arguments to ``sparc_joint_loss`` on the
            next forward pass.
        """
        self._step += 1

        # Update EMA for each tracked component
        for component_name, value in loss_components.items():
            if not math.isfinite(value) or value < 0:
                continue
            prev = self._ema.get(component_name, value)
            self._ema[component_name] = (
                (1.0 - self.ema_alpha) * prev + self.ema_alpha * value
            )

        # During warmup: keep base lambdas unchanged
        if self._step < self.warmup_steps:
            return dict(self.lambdas)

        # Reference term mean (must be positive and finite)
        ref_mean = self._ema.get(self.reference_term, 0.0)
        if not (math.isfinite(ref_mean) and ref_mean > 1e-12):
            return dict(self.lambdas)

        # Rescale each lambda so its component mean ≈ ref_mean
        updated: dict[str, float] = {}
        for lambda_key, base_val in self.base_lambdas.items():
            component = self._LAMBDA_TO_COMPONENT.get(lambda_key)
            should_adapt = (
                component is not None
                and (self.enabled_terms is None or lambda_key in self.enabled_terms)
            )
            if not should_adapt:
                updated[lambda_key] = base_val
                continue

            comp_mean = self._ema.get(component, 0.0)
            if not (math.isfinite(comp_mean) and comp_mean > 1e-12):
                updated[lambda_key] = base_val
                continue

            # ReLoBRaLo: scale = ref_mean / comp_mean
            scale = ref_mean / comp_mean
            # Clamp to max_scale_factor around base
            scale = max(1.0 / self.max_scale_factor, min(scale, self.max_scale_factor))
            updated[lambda_key] = base_val * scale

        self.lambdas = updated
        return dict(updated)

    def state_dict(self) -> dict:
        """Serializable state for checkpoint saving."""
        return {
            "ema": dict(self._ema),
            "step": self._step,
            "lambdas": dict(self.lambdas),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a checkpoint."""
        self._ema = dict(state.get("ema", {}))
        self._step = int(state.get("step", 0))
        self.lambdas = dict(state.get("lambdas", self.base_lambdas))
