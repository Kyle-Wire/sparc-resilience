"""
FoldState — typed container for the shared state passed to _exec_cv_fold.

Extracted from v2_neural_training to break the circular dependency:
  fold_trainer.py  →  v2_neural_training.py  (FoldState was defined here)
  v2_neural_training.py  →  fold_trainer.py  (FoldTrainer uses it)

With FoldState in its own module both consumers can import it directly
without a TYPE_CHECKING guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ArchConfig:
    hidden_dim: int
    dropout: float
    n_heads: int
    max_neighbors: int
    siren_omega: float
    thresholds: list
    thresholds_norm: list
    time_embed_dim: int
    geo_pe_dim: int
    n_base: int
    n_physics: int
    n_physics_extended: int
    d_spatial: int
    resolution: float
    pr_input_dim: int
    pr_col_idxs: list
    prior_mean: float
    pr_cfg: dict
    n_treatments: int
    material_priors: dict
    gwrf_k: int
    predictor_bandwidths: Any
    predictor_kernel_field: Any
    feature_names: list


@dataclass
class _TrainingConfig:
    n_epochs: int
    pretrain_epochs: int
    pr_pretrain_epochs: int
    lr: float
    clip_norm: float
    warmup_epochs: int
    ramp_epochs: int
    batch_size: int
    target_lambdas: dict
    y_mean: float
    y_std: float
    n_folds: int
    use_amp: bool
    use_cuda_graphs: bool
    head_weight_decay: float = 1e-4
    seed: int | None = None


@dataclass
class _JEPAConfig:
    enable: bool
    weights: Any
    mask_ratio: float
    spatial_patch: bool
    n_patches: int
    lam: float
    ema_tau_start: float
    ema_tau_end: float
    warmup_steps: int
    curric_start: int
    curric_end: int
    lambda_scenario: float
    lambda_latent_pde: float
    scenario_perturb_std: float
    predictor_blocks: int
    predictor_film: bool
    action_dim: int


@dataclass
class _ContinualConfig:
    ewc_active: bool
    ewc_lambda: float
    cont_fisher: list
    cont_optimal: list
    ot_active: bool
    ot_lambda: float
    coreset_acts: Any
    # Candidate B: experience replay
    replay_lambda: float = 0.0
    previous_coresets: list = None  # list of dicts with keys X, y, coords

    def __post_init__(self):
        if self.previous_coresets is None:
            self.previous_coresets = []


@dataclass
class FoldState:
    """Typed container for all state shared with :func:`_exec_cv_fold`.

    Grouping avoids the ~50-key opaque dict pattern and makes the
    interface between the outer training loop and each CV fold explicit.
    """
    tensors: dict
    coords_np: Any
    alpha_targets: Any
    jepa_pretrained_trunk_state: Any
    base_oof_predictions: Any
    arch: _ArchConfig
    training: _TrainingConfig
    jepa: _JEPAConfig
    continual: _ContinualConfig
