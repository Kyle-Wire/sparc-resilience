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


# ---------------------------------------------------------------------------
# JD-4: Multi-step latent rollout
# ---------------------------------------------------------------------------

@dataclass
class MultiStepRolloutResult:
    """Output of ``multi_step_latent_rollout``.

    Attributes
    ----------
    steps : list[LatentRolloutResult]
        One result per action in the sequence (in order).
    final : LatentRolloutResult
        The last element of ``steps`` — the result after all actions have
        been applied.
    n_steps : int
        Total number of actions applied.
    """

    steps: list           # list[LatentRolloutResult]
    final: LatentRolloutResult
    n_steps: int


@torch.no_grad()
def multi_step_latent_rollout(
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
    actions: list,           # list[(treatment: str, delta_x: float, delta_t: float)]
    mode: str = "latent",    # "latent" | "reencode"
    max_steps: int = 5,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    cascade=None,            # PhysicsCascade | None (required for mode="reencode")
    physics_feat_cols: list | None = None,  # column names matching physics_feats dim
) -> MultiStepRolloutResult:
    """Chain multiple latent-space interventions sequentially.

    Parameters
    ----------
    actions : list of (treatment, delta_x, delta_t) tuples
        Sequence of interventions to apply.  Each step uses the latent
        state left by the previous step as its starting point.
    mode : str
        ``"latent"`` (default) — propagate purely through the predictor in
        latent space.  ``"reencode"`` — re-encode perturbed physics features
        via the trunk between each step (requires *cascade*).
    max_steps : int
        Guard against excessively long chains.  Raises ``ValueError`` when
        ``len(actions) > max_steps``.
    cascade : PhysicsCascade | None
        Physics cascade model required for ``mode="reencode"``.
    physics_feat_cols : list[str] | None
        Column names corresponding to the ``physics_feats`` tensor columns.
        Required when ``mode="reencode"`` so that the cascade can address
        features by name.

    Returns
    -------
    MultiStepRolloutResult
    """
    if len(actions) > max_steps:
        raise ValueError(
            f"multi_step_latent_rollout: len(actions)={len(actions)} exceeds "
            f"max_steps={max_steps}.  Increase max_steps or shorten the action chain."
        )
    if mode == "reencode" and cascade is None:
        raise ValueError(
            "multi_step_latent_rollout: mode='reencode' requires a cascade object "
            "but cascade=None was supplied.  Pass a PhysicsCascade instance or "
            "use mode='latent'."
        )

    model.eval()
    predictor.eval()
    action_embed.eval()

    n = physics_feats.shape[0]
    device = physics_feats.device

    steps: list[LatentRolloutResult] = []
    h_state = model.encode(physics_feats, alpha)
    current_physics = physics_feats  # will be updated per step in reencode mode

    # Baseline decode (before any action)
    T_baseline_t, _ = model.decode(h_state, base_preds, X_spatial, coords, knn_index)
    T_baseline_np = (T_baseline_t.cpu().numpy() * y_std) + y_mean

    for treatment, delta_x, delta_t in actions:
        if mode == "reencode" and cascade is not None:
            # Apply physics cascade to the current feature tensor
            import pandas as _pd
            cols = physics_feat_cols or [str(i) for i in range(current_physics.shape[1])]
            df_phys = _pd.DataFrame(
                current_physics.cpu().numpy(), columns=cols
            )
            df_perturbed = cascade.apply(df_phys, treatment=treatment, delta_x=float(delta_x))
            current_physics = torch.tensor(
                df_perturbed.values, dtype=physics_feats.dtype, device=device
            )
            h_state = model.encode(current_physics, alpha)
        else:
            treat_idx = torch.full(
                (n,), action_embed.token_index(treatment),
                dtype=torch.long, device=device,
            )
            dx = torch.full((n,), float(delta_x), dtype=physics_feats.dtype, device=device)
            dt = torch.full((n,), float(delta_t), dtype=physics_feats.dtype, device=device)
            a = action_embed(treat_idx, dx, dt)
            h_state = predictor(h_state, a)

        T_pred_t, _ = model.decode(h_state, base_preds, X_spatial, coords, knn_index)
        T_pred_np = (T_pred_t.cpu().numpy() * y_std) + y_mean

        step_result = LatentRolloutResult(
            treatment=treatment,
            delta_x=float(delta_x),
            T_pred=T_pred_np,
            T_baseline=T_baseline_np,
            delta=T_pred_np - T_baseline_np,
        )
        steps.append(step_result)

    return MultiStepRolloutResult(
        steps=steps,
        final=steps[-1] if steps else LatentRolloutResult(
            treatment="", delta_x=0.0,
            T_pred=T_baseline_np, T_baseline=T_baseline_np,
            delta=np.zeros(n),
        ),
        n_steps=len(steps),
    )
