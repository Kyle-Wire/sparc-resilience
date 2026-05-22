"""
Training phase protocol and phase classes for SPARC V2.

Defines the ``TrainingPhase`` interface and four concrete phases that
the ``_exec_cv_fold`` training loop currently implements inline.  This
module establishes the architectural boundary for a future incremental
migration of ``v2_neural_training._exec_cv_fold``.

Current state
-------------
All phase logic lives in ``_exec_cv_fold`` (sparc/run/v2_neural_training.py).
The four phases are identifiable by their natural sequencing:

  1. **SurrogatePhase** — independent MSE pre-training of the three
     differentiable surrogates against V1 OOF predictions.
     (current: ``_pretrain_surrogates()`` called at ~line 820 of fold body)

  2. **JEPAPhase** — trunk-level self-supervised pre-training using
     masked physics context prediction (I-JEPA style).
     (current: ``jepa_pretrain`` block, guarded by ``jepa_enable``)

  3. **MainFoldPhase** — joint training of all models through the
     4-stage curriculum (Warmup → Physics Activation → Joint → Final).
     (current: ``for epoch in range(n_epochs)`` at line ~1573 of fold body)

  4. **SWAPhase** — Stochastic Weight Averaging over the trained models
     to improve generalisation.
     (current: ``swa_model`` block after main fold loop)

Migration path
--------------
1. Extract each phase's logic into the corresponding class below.
2. In ``_exec_cv_fold``, replace the inline block with::

       phase = SurrogatePhase(surrogates, train_physics, train_spatial, ...)
       phase.run(n_epochs=pretrain_epochs)

3. Repeat for each remaining phase in order.
4. After all four phases are extracted, ``_exec_cv_fold`` becomes a thin
   coordinator that instantiates and sequences phases.

Usage (post-migration, illustrative)::

    bundle   = ModelBundle.create(ss.arch, device)
    jl       = JointLoss.from_target_lambdas(target_lambdas, thresholds, resolution)
    trainer  = FoldTrainer(bundle, jl, ss, device)
    result   = trainer.run(train_idx, test_idx)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Protocol / base class
# ---------------------------------------------------------------------------

class TrainingPhase(ABC):
    """Base class for a discrete SPARC training phase.

    Each phase encapsulates the data and hyper-parameters it needs;
    the calling code only invokes :meth:`run` (or :meth:`run_epoch`
    in the mini-batch case) and consumes the returned metrics.
    """

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """Execute the full phase and return a metrics dict."""

    # Optional — phases that expose per-epoch streaming implement this.
    def run_epoch(self, epoch: int, **kwargs) -> dict[str, float]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support per-epoch stepping"
        )


# ---------------------------------------------------------------------------
# Phase 1 — Surrogate pre-training
# ---------------------------------------------------------------------------

@dataclass
class SurrogatePhase(TrainingPhase):
    """Independently pre-train each differentiable surrogate with MSE.

    Wraps ``_pretrain_surrogates`` in ``v2_neural_training``.

    Parameters
    ----------
    surrogates : dict[str, nn.Module]
        ``{"gwr": ..., "gwrf": ..., "ggpgam": ...}``
    physics_feats : torch.Tensor
        (N_train, n_physics) — physics feature inputs for the training fold.
    spatial_feats : torch.Tensor
        (N_train, d_spatial) — sinusoidal spatial encoding.
    y : torch.Tensor
        (N_train,) — normalised outcome (fallback when V1 targets unavailable).
    base_targets : dict[str, torch.Tensor] | None
        V1 OOF predictions keyed by surrogate name.  When provided, each
        surrogate trains against the corresponding V1 output; otherwise
        falls back to ``y``.
    n_epochs : int
    lr : float
    """

    surrogates: dict[str, nn.Module]
    physics_feats: torch.Tensor
    spatial_feats: torch.Tensor
    y: torch.Tensor
    base_targets: "dict[str, torch.Tensor] | None" = None
    n_epochs: int = 50
    lr: float = 1e-3

    def run(self, **kwargs) -> dict[str, Any]:
        """Run the surrogate pre-training phase.

        Delegates to ``_pretrain_surrogates`` in the training orchestrator.
        The inline implementation is reproduced here for standalone use.

        Returns
        -------
        dict mapping surrogate name to final R².
        """
        import torch.nn.functional as F

        results: dict[str, float] = {}
        for name, surrogate in self.surrogates.items():
            target = (
                self.base_targets[name]
                if self.base_targets and name in self.base_targets
                else self.y
            )
            surrogate.train()
            optimizer = torch.optim.AdamW(surrogate.parameters(), lr=self.lr)

            for _ep in range(self.n_epochs):
                if name == "gwr":
                    pred, _ = surrogate(self.physics_feats, self.spatial_feats)
                else:
                    pred = surrogate(self.physics_feats, self.spatial_feats)
                loss = F.mse_loss(pred.squeeze(), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                if name == "gwr":
                    pred, _ = surrogate(self.physics_feats, self.spatial_feats)
                else:
                    pred = surrogate(self.physics_feats, self.spatial_feats)
                pred = pred.squeeze()
                ss_res = ((pred - target) ** 2).sum()
                ss_tot = ((target - target.mean()) ** 2).sum()
                r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
            results[name] = r2

        return results


# ---------------------------------------------------------------------------
# Phase 2 — JEPA trunk pre-training
# ---------------------------------------------------------------------------

@dataclass
class JEPAPhase(TrainingPhase):
    """I-JEPA-style self-supervised trunk pre-training.

    Wraps the ``jepa_enable`` block inside ``_exec_cv_fold``.

    Parameters
    ----------
    model : SPARCMetaLearner
        Online encoder (SharedTrunk + CityHead).
    ema_trunk : EMATrunk
        Momentum-updated target encoder.
    latent_predictor : LatentPredictor
        Trunk-level latent prediction head.
    optimizer : torch.optim.Optimizer
        Dedicated JEPA optimizer (AdamW over predictor + action_embed).
    config : dict
        JEPA hyper-parameters from ``_JEPAConfig`` fields.
    tensors : dict[str, torch.Tensor]
        Full-dataset tensors (training rows are sliced inside ``run``).
    train_idx : np.ndarray
        Fold training indices into *tensors*.
    """

    model: nn.Module
    ema_trunk: nn.Module
    latent_predictor: nn.Module
    optimizer: torch.optim.Optimizer
    config: dict  # mirrors _JEPAConfig fields as plain dict
    tensors: dict[str, torch.Tensor]
    train_idx: "Any"  # np.ndarray

    # action_embed is optional (only used when lambda_scenario > 0)
    action_embed: "nn.Module | None" = None

    def run(self, **kwargs) -> dict[str, Any]:
        """Execute the JEPA pre-training phase.

        .. note::
            Full implementation is still inline in ``_exec_cv_fold``.
            This stub documents the intended interface; it will be
            filled in during the migration from ``v2_neural_training``.
        """
        raise NotImplementedError(
            "JEPAPhase.run() is not yet migrated from _exec_cv_fold. "
            "Set jepa_enable=True in your config and the inline code in "
            "v2_neural_training._exec_cv_fold will execute it."
        )


# ---------------------------------------------------------------------------
# Phase 3 — Main fold training loop
# ---------------------------------------------------------------------------

@dataclass
class MainFoldPhase(TrainingPhase):
    """4-stage curriculum joint training.

    Wraps the ``for epoch in range(n_epochs)`` loop inside
    ``_exec_cv_fold``, which progresses through:

      Stage A (Warmup)          — representation learning, low physics weight
      Stage B (Physics Act.)    — physics loss ramps up
      Stage C (Joint Optim.)    — all terms active at target lambdas
      Stage D (Final)           — SWA accumulation starts

    Parameters
    ----------
    bundle : ModelBundle
        All models for this fold.
    joint_loss : JointLoss
        Configured joint loss instance (owns lambda weights + curriculum).
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    tensors : dict[str, torch.Tensor]
        Training fold tensors (pre-sliced to training rows).
    fold_knn : torch.Tensor
        (N_train, max_neighbors) KNN indices for spatial attention.
    fold_cardinal : torch.Tensor
        (N_train, 4) cardinal-neighbor indices for PDE residual.
    config : dict
        Training hyper-params from ``_TrainingConfig`` fields.
    """

    bundle: "Any"  # ModelBundle — avoid circular import at module level
    joint_loss: "Any"  # JointLoss
    optimizer: torch.optim.Optimizer
    scheduler: "Any"  # torch.optim.lr_scheduler._LRScheduler
    tensors: dict[str, torch.Tensor]
    fold_knn: torch.Tensor
    fold_cardinal: torch.Tensor
    config: dict  # mirrors _TrainingConfig fields as plain dict

    def run(self, **kwargs) -> dict[str, Any]:
        """Execute the main fold training loop.

        .. note::
            Full implementation is still inline in ``_exec_cv_fold``.
            This stub documents the intended interface; it will be
            filled in during the migration.
        """
        raise NotImplementedError(
            "MainFoldPhase.run() is not yet migrated from _exec_cv_fold. "
            "The inline training loop at v2_neural_training.py ~line 1573 "
            "is the reference implementation."
        )

    def run_epoch(self, epoch: int, **kwargs) -> dict[str, float]:
        """Execute one training epoch (for streaming / callback use).

        .. note::
            This will delegate to a per-batch loop after migration.
        """
        raise NotImplementedError("run_epoch() will be available after migration")


# ---------------------------------------------------------------------------
# Phase 4 — Stochastic Weight Averaging
# ---------------------------------------------------------------------------

@dataclass
class SWAPhase(TrainingPhase):
    """Stochastic Weight Averaging over the trained model.

    Wraps the ``swa_model`` block inside ``_exec_cv_fold``.

    Parameters
    ----------
    model : nn.Module
        Trained ``SPARCMetaLearner`` from :class:`MainFoldPhase`.
    swa_lr : float
        SWA learning rate.
    swa_epochs : int
        Number of SWA averaging steps.
    tensors : dict
        Full training fold tensors.
    optimizer_state : dict
        State dict from the main optimizer (for SWALR).
    """

    model: nn.Module
    swa_lr: float = 5e-4
    swa_epochs: int = 10
    tensors: "dict[str, torch.Tensor]" = field(default_factory=dict)
    optimizer_state: "dict | None" = None

    def run(self, **kwargs) -> dict[str, Any]:
        """Execute SWA over *swa_epochs* steps.

        .. note::
            Full implementation is still inline in ``_exec_cv_fold``.
            This stub documents the intended interface.
        """
        raise NotImplementedError(
            "SWAPhase.run() is not yet migrated from _exec_cv_fold. "
            "The inline SWA block in v2_neural_training._exec_cv_fold "
            "is the reference implementation."
        )


# ---------------------------------------------------------------------------
# FoldTrainer — thin coordinator (post-migration target state)
# ---------------------------------------------------------------------------

@dataclass
class FoldTrainer:
    """Thin coordinator that sequences the four training phases.

    This is the intended **post-migration** interface.  It will be
    functional once all four phase classes are fully implemented.

    Parameters
    ----------
    bundle : ModelBundle
    joint_loss : JointLoss
    fold_state : FoldState
        Full :class:`FoldState` from ``train_neural_meta``.
    device : torch.device

    Usage (post-migration)::

        trainer = FoldTrainer(bundle, jl, ss, device)
        result  = trainer.run(train_idx, test_idx)
    """

    bundle: "Any"   # ModelBundle
    joint_loss: "Any"  # JointLoss
    fold_state: "Any"  # FoldState
    device: torch.device

    def run(
        self,
        train_idx: "Any",
        test_idx: "Any",
    ) -> "tuple[Any, Any, Any]":
        """Train one fold and return ``(test_idx, oof_preds, oof_std)``.

        .. note::
            Delegates to ``_exec_cv_fold`` until all phase stubs are
            implemented.  The phases in this module form the migration
            target; once each ``run()`` method is filled in,
            ``_exec_cv_fold`` can be replaced by this body.
        """
        from sparc.run.v2_neural_training import _exec_cv_fold  # noqa: F401
        raise NotImplementedError(
            "FoldTrainer.run() will be operational after SurrogatePhase, "
            "JEPAPhase, MainFoldPhase, and SWAPhase are fully migrated. "
            "Use _exec_cv_fold directly until then."
        )
