"""Latent-space scenario rollout (JEPA Phase 2.e).

Encodes inputs once via the SharedTrunk, applies an action-conditioned
update in latent space via ``LatentPredictor``, and decodes back through
the CityHead to a continuous outcome:

    h_state = model.encode(physics, alpha)               # ~256-d
    h_next  = predictor(h_state, action_embed(treatment, Δx, Δt))
    T_pred  = model.decode(h_next, base_preds, X_spatial, coords, knn)

This is an A/B alternative to ``ScenarioSimulator.run_with_model_reprediction``
that bypasses base-model re-prediction and runs the intervention purely
through the learned trunk + predictor — the V-JEPA 2-AC pattern.

Pure helper; does not modify global state and is config-flag isolated
from the legacy ScenarioSimulator path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from sparc.models.action_embedding import ActionEmbedding
from sparc.models.latent_predictor import LatentPredictor
from sparc.models.neural_meta import SPARCMetaLearner


@dataclass
class LatentRolloutResult:
    """Output of ``latent_rollout`` for a single (treatment, magnitude) pair."""

    treatment: str
    delta_x: float
    T_pred: np.ndarray            # (N,) — denormalised outcome
    T_baseline: np.ndarray        # (N,) — denormalised baseline outcome
    delta: np.ndarray             # T_pred - T_baseline


@torch.no_grad()
def latent_rollout(
    *,
    model: SPARCMetaLearner,
    predictor: LatentPredictor,
    action_embed: ActionEmbedding,
    physics_feats: torch.Tensor,
    base_preds: torch.Tensor,
    X_spatial: torch.Tensor,
    coords: torch.Tensor,
    knn_index: torch.Tensor,
    alpha: torch.Tensor,
    treatment: str,
    delta_x: float,
    delta_t: float = 0.0,
    y_mean: float = 0.0,
    y_std: float = 1.0,
) -> LatentRolloutResult:
    """Run a single-shot latent-space scenario rollout.

    Parameters
    ----------
    model : trained SPARCMetaLearner.
    predictor : action-conditioned LatentPredictor (Phase 2 form).
    action_embed : ActionEmbedding mapping (treatment, Δx, Δt) → R^D.
    physics_feats, base_preds, X_spatial, coords, knn_index, alpha :
        Tensors as produced by the V2 training tensors dict.  All on
        the same device as ``model``.
    treatment : str
        Name of the perturbed feature.  Unknown names fall back to the
        embedding's "__none__" slot (no-op intervention).
    delta_x : float
        Signed magnitude of the intervention in feature units.
    delta_t : float
        Time delta (normalised); 0.0 for pure scenario rollout.
    y_mean, y_std : float
        Normalisation stats used to denormalise the model outputs.

    Returns
    -------
    LatentRolloutResult
    """
    model.eval()
    predictor.eval()
    action_embed.eval()

    n = physics_feats.shape[0]
    device = physics_feats.device

    # Baseline encode → decode (no action).
    h_state = model.encode(physics_feats, alpha)
    T_baseline, _ = model.decode(
        h_state, base_preds, X_spatial, coords, knn_index,
    )

    # Encode the action and apply the predictor in latent space.
    treat_idx = torch.full(
        (n,), action_embed.token_index(treatment),
        dtype=torch.long, device=device,
    )
    dx = torch.full((n,), float(delta_x), dtype=physics_feats.dtype, device=device)
    dt = torch.full((n,), float(delta_t), dtype=physics_feats.dtype, device=device)
    a = action_embed(treat_idx, dx, dt)

    h_next = predictor(h_state, a)
    T_pred, _ = model.decode(
        h_next, base_preds, X_spatial, coords, knn_index,
    )

    # Denormalise.
    T_pred_np = (T_pred.cpu().numpy() * y_std) + y_mean
    T_base_np = (T_baseline.cpu().numpy() * y_std) + y_mean

    return LatentRolloutResult(
        treatment=treatment,
        delta_x=float(delta_x),
        T_pred=T_pred_np,
        T_baseline=T_base_np,
        delta=T_pred_np - T_base_np,
    )


@torch.no_grad()
def latent_rollout_grid(
    *,
    model: SPARCMetaLearner,
    predictor: LatentPredictor,
    action_embed: ActionEmbedding,
    physics_feats: torch.Tensor,
    base_preds: torch.Tensor,
    X_spatial: torch.Tensor,
    coords: torch.Tensor,
    knn_index: torch.Tensor,
    alpha: torch.Tensor,
    treatments: Sequence[str],
    magnitudes: Sequence[float],
    y_mean: float = 0.0,
    y_std: float = 1.0,
) -> list[LatentRolloutResult]:
    """Sweep ``latent_rollout`` over a (treatment × magnitude) grid."""
    out: list[LatentRolloutResult] = []
    for treatment in treatments:
        for delta_x in magnitudes:
            out.append(
                latent_rollout(
                    model=model, predictor=predictor, action_embed=action_embed,
                    physics_feats=physics_feats, base_preds=base_preds,
                    X_spatial=X_spatial, coords=coords, knn_index=knn_index,
                    alpha=alpha,
                    treatment=treatment, delta_x=float(delta_x),
                    y_mean=y_mean, y_std=y_std,
                )
            )
    return out
