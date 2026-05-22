"""
JEPA loss for SPARC: cosine alignment + VICReg anti-collapse regularizers.

Components
----------
1. Alignment   :  1 − cos(h_pred, h_target)               (predictor learns target)
2. Variance    :  hinge(γ − std(h))                       (per-feature std ≥ γ)
3. Covariance  :  off-diagonal Frobenius of cov(h)        (decorrelate features)

Targets are assumed to come from an EMA-tracked encoder and are
already detached (no gradient flows through them); this module does not
re-detach so the caller is responsible for ``stop-grad`` on
``h_target`` (typically by computing it under ``torch.no_grad()``).

Curriculum
----------
``jepa_curriculum_weight`` returns a 0→1 multiplier following the
two-stage schedule from the integration plan:

  epochs 1..warmup_start          : weight = 0      (off)
  epochs warmup_start..warmup_end : linear ramp    (warming up)
  epochs >= warmup_end            : weight = 1      (full)

This lets supervised signals stabilize before the JEPA term begins
shaping the embedding geometry.

References
----------
- Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance
  Regularization for Self-Supervised Learning.* ICLR 2022.
- Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable
  Understanding, Prediction and Planning.* arXiv:2506.09985, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JEPALossWeights:
    """VICReg-style component weights for the JEPA loss."""
    alignment: float = 1.0
    variance: float = 1.0
    covariance: float = 0.04
    variance_gamma: float = 1.0  # target per-feature std (hinge threshold)


# ---------------------------------------------------------------------------
# VICReg components
# ---------------------------------------------------------------------------

def variance_loss(h: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Hinge penalty: encourage per-feature std ≥ ``gamma``.

    ``h`` : (N, D) embedding.  Standard deviation is taken across the N
    sample dimension; the hinge ``relu(γ − std)`` is averaged over D.
    A small ``eps`` is added under the sqrt to avoid div-by-zero
    gradients at perfect collapse (std = 0).
    """
    if h.dim() != 2:
        raise ValueError(f"variance_loss expects (N, D); got {tuple(h.shape)}")
    std = torch.sqrt(h.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(h: torch.Tensor) -> torch.Tensor:
    """
    Off-diagonal Frobenius norm of the empirical covariance, scaled by 1/D.

    ``h`` : (N, D) embedding.  Centers ``h``, computes Cᵢⱼ, sums squared
    off-diagonal entries and divides by D (per VICReg's normalization).
    """
    if h.dim() != 2:
        raise ValueError(f"covariance_loss expects (N, D); got {tuple(h.shape)}")
    n, d = h.shape
    if n < 2:
        return h.new_zeros(())

    h_centered = h - h.mean(dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / (n - 1)        # (D, D)

    # Off-diagonal mask: zero on the diagonal, one elsewhere.
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def alignment_loss(h_pred: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
    """
    Mean (1 − cosine_similarity) between predicted and target embeddings.

    Cosine is taken along the feature dimension so the loss is invariant
    to embedding magnitude (which the variance term separately controls).
    """
    if h_pred.shape != h_target.shape:
        raise ValueError(
            f"shape mismatch: h_pred {tuple(h_pred.shape)} vs "
            f"h_target {tuple(h_target.shape)}"
        )
    cos = F.cosine_similarity(h_pred, h_target, dim=-1)  # (N,)
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Combined JEPA loss
# ---------------------------------------------------------------------------

def jepa_loss(
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
    weights: JEPALossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined alignment + VICReg loss.

    Variance and covariance terms are computed on **both** ``h_pred``
    and ``h_target`` and averaged — symmetric regularization, as in
    VICReg.  This prevents collapse on either branch.

    Returns
    -------
    total : scalar Tensor
    components : dict of detached scalar values for logging
    """
    if weights is None:
        weights = JEPALossWeights()

    align = alignment_loss(h_pred, h_target)

    var_p = variance_loss(h_pred, gamma=weights.variance_gamma)
    var_t = variance_loss(h_target, gamma=weights.variance_gamma)
    var = 0.5 * (var_p + var_t)

    cov_p = covariance_loss(h_pred)
    cov_t = covariance_loss(h_target)
    cov = 0.5 * (cov_p + cov_t)

    total = (
        weights.alignment * align
        + weights.variance * var
        + weights.covariance * cov
    )

    components = {
        "jepa_align": align.item(),
        "jepa_variance": var.item(),
        "jepa_covariance": cov.item(),
        "jepa_total": total.item(),
    }
    return total, components


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def jepa_curriculum_weight(
    epoch: int, warmup_start: int = 5, warmup_end: int = 15,
) -> float:
    """
    Two-stage schedule for the JEPA loss multiplier.

    epoch < warmup_start            -> 0.0
    warmup_start <= epoch < warmup_end -> linear ramp 0..1
    epoch >= warmup_end             -> 1.0
    """
    if warmup_end <= warmup_start:
        raise ValueError("warmup_end must be > warmup_start.")
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return 1.0
    span = warmup_end - warmup_start
    return float(epoch - warmup_start) / float(span)


# ---------------------------------------------------------------------------
# Phase 2: scenario-distillation + latent PDE consistency
# ---------------------------------------------------------------------------

def scenario_distillation_loss(
    h_pred_perturbed: torch.Tensor,
    h_target_perturbed: torch.Tensor,
    weights: JEPALossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """JEPA loss specialised for the scenario-distillation regime.

    Identical math to the symmetric ``jepa_loss`` (alignment + VICReg
    variance + covariance), but reported under distinct component names
    so the two regimes (Phase 1 masked-context vs Phase 2 perturbed)
    can be tracked independently in training logs.

    Parameters
    ----------
    h_pred_perturbed : (N, D)
        Output of ``predictor(h_state, action_embed)``.
    h_target_perturbed : (N, D)
        Output of ``ema_trunk.encode_target(physics + Δx, alpha)``;
        must already be detached.
    """
    if weights is None:
        weights = JEPALossWeights()

    align = alignment_loss(h_pred_perturbed, h_target_perturbed)
    var_p = variance_loss(h_pred_perturbed, gamma=weights.variance_gamma)
    var_t = variance_loss(h_target_perturbed, gamma=weights.variance_gamma)
    cov_p = covariance_loss(h_pred_perturbed)
    cov_t = covariance_loss(h_target_perturbed)
    var = 0.5 * (var_p + var_t)
    cov = 0.5 * (cov_p + cov_t)

    total = (
        weights.alignment * align
        + weights.variance * var
        + weights.covariance * cov
    )

    return total, {
        "scenario_align": align.item(),
        "scenario_variance": var.item(),
        "scenario_covariance": cov.item(),
        "scenario_total": total.item(),
    }


def latent_pde_consistency_loss(
    h: torch.Tensor,
    neighbor_idx: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Spatial smoothness in latent space — Phase 2 cheap stand-in for
    JVP-based latent-PDE consistency.

    Encourages the trunk embedding to vary smoothly across spatial
    neighbours — a discrete analogue of ``α · ∇²ĥ ≈ 0`` once the
    process-rate gating is applied.  Implementation:

        Δh_i = h_i − mean_{j ∈ N(i)} h_j
        loss = ||α_i ⊙ Δh_i||² / (N · D)

    This is a lightweight first-order Laplacian penalty; the full JVP
    formulation (project the existing finite-difference ``∇²`` operator
    into latent space) lands in Phase 4 alongside the hierarchical
    encoders where it composes naturally.

    Parameters
    ----------
    h : (N, D) — latent embedding (typically ``predictor(...)`` output).
    neighbor_idx : (N, K) long — KNN indices into the same batch.
    alpha : (N, 1) or None — optional process-rate gating in [0, 1].

    Returns
    -------
    loss : scalar
    components : dict
    """
    if h.dim() != 2:
        raise ValueError(f"h must be (N, D); got {tuple(h.shape)}")
    if neighbor_idx.dim() != 2 or neighbor_idx.shape[0] != h.shape[0]:
        raise ValueError(
            f"neighbor_idx must be (N, K) with N={h.shape[0]}; "
            f"got {tuple(neighbor_idx.shape)}",
        )

    n, d = h.shape
    # h_neighbors : (N, K, D)
    h_neighbors = h[neighbor_idx]
    # Discrete Laplacian: h - mean(neighbors)
    laplacian = h - h_neighbors.mean(dim=1)  # (N, D)

    if alpha is not None:
        # Broadcast (N, 1) over D
        laplacian = alpha * laplacian

    loss = laplacian.pow(2).mean()
    return loss, {"latent_pde": loss.item()}


# ---------------------------------------------------------------------------
# Spatial patch masking
# ---------------------------------------------------------------------------

def spatial_patch_mask(
    coords: torch.Tensor,
    mask_ratio: float = 0.4,
    n_patches: int = 4,
    min_patch_radius: float | None = None,
) -> torch.Tensor:
    """Build a per-point boolean mask from spatial disk patches.

    Rather than masking random feature channels (the old approach), this
    function selects ``n_patches`` random centre points from the batch and
    masks every point within a radius of each centre.  The result is a
    geographically contiguous hidden region — consistent with the original
    I-JEPA intuition that the model must predict a spatially coherent
    neighbourhood, not just reconstruct isolated missing features.

    Parameters
    ----------
    coords : (N, 2) float tensor
        2-D spatial coordinates of the points in the current batch.
    mask_ratio : float
        Target fraction of points to mask overall.  The patch radius is
        derived so that (on average) ``mask_ratio × N`` points fall inside
        the union of the ``n_patches`` disks.
    n_patches : int
        Number of non-overlapping (best-effort) disk centres to sample.
    min_patch_radius : float or None
        Minimum radius.  If None, derived from the coordinate range so that
        each patch covers approximately ``mask_ratio / n_patches`` of the
        bounding box area.

    Returns
    -------
    mask : (N,) bool tensor
        ``True`` for points that are **masked** (hidden from the context
        encoder).  Callers convert to a ``channel_mask`` or point-level
        weight as appropriate.
    """
    N = coords.shape[0]
    device = coords.device

    if N < 4:
        # Too few points — fall back to random point masking
        return torch.rand(N, device=device) < mask_ratio

    # Coordinate range
    coord_min = coords.min(dim=0).values   # (2,)
    coord_max = coords.max(dim=0).values   # (2,)
    span = (coord_max - coord_min).clamp(min=1e-6)  # (2,)

    if min_patch_radius is None:
        # Each patch should cover mask_ratio/n_patches of the bounding box
        # Area of bbox = span[0]*span[1]; target area per patch = (mask_ratio/n_patches)*bbox_area
        # π*r² = target_area  →  r = sqrt(target_area / π)
        bbox_area = span[0] * span[1]
        target_area = (mask_ratio / max(n_patches, 1)) * bbox_area
        radius = float((target_area / 3.14159).sqrt().clamp(min=span.min() * 0.05))
    else:
        radius = float(min_patch_radius)

    masked = torch.zeros(N, dtype=torch.bool, device=device)
    centres_used: list[torch.Tensor] = []

    rng_idx = torch.randperm(N, device=device)

    for candidate in rng_idx:
        if len(centres_used) >= n_patches:
            break
        centre = coords[candidate]  # (2,)

        # Soft non-overlap: skip if too close to an existing centre
        too_close = any(
            float((centre - c).norm()) < radius * 0.5
            for c in centres_used
        )
        if too_close:
            continue

        dist = (coords - centre).norm(dim=1)  # (N,)
        masked |= dist <= radius
        centres_used.append(centre)

    # Safety: if mask is empty or coverage is way off, supplement with random
    n_masked = int(masked.sum())
    target_n = int(mask_ratio * N)
    if n_masked == 0:
        # No patches landed — full random fallback
        return torch.rand(N, device=device) < mask_ratio
    if n_masked < target_n // 2:
        # Patches covered very little — supplement randomly
        extra = torch.rand(N, device=device) < (mask_ratio - n_masked / N)
        masked = masked | extra

    return masked

