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

    phase = SurrogatePhase(surrogates, ...)
    result = phase.run()
    # ... then JEPAPhase, MainFoldPhase, SWAPhase in FoldTrainer (fold_trainer.py)
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
        """Execute the JEPA trunk pre-training phase.

        Implements the I-JEPA masked-context prediction loop: the online
        encoder encodes a masked view of the physics features; the
        latent predictor tries to reconstruct the target encoding produced
        by the EMA-updated trunk on the unmasked view.

        Returns
        -------
        dict with keys ``jepa_pretrained`` (bool), ``epochs`` (int),
        ``avg_loss`` (float).
        """
        from sparc.training.jepa_loss import jepa_loss

        jepa_epochs: int = int(self.config.get("pretrain_epochs", 0))
        if jepa_epochs <= 0:
            return {"jepa_pretrained": False, "epochs": 0, "avg_loss": 0.0}

        weights = self.config.get("weights")
        clip_norm: float = float(self.config.get("clip_norm", 1.0))
        mask_ratio: float = float(self.config.get("mask_ratio", 0.25))

        device = next(self.model.parameters()).device
        phys_key = (
            "physics_feats_extended"
            if "physics_feats_extended" in self.tensors
            else "physics_feats"
        )
        phys_all = self.tensors[phys_key]
        idx = self.train_idx
        N = len(idx)
        alpha = torch.full((N, 1), 0.5, dtype=torch.float32, device=device)

        total_loss = 0.0
        self.model.train()

        for epoch in range(jepa_epochs):
            phys = phys_all[idx]
            n_feat = phys.shape[-1]
            keep_p = max(0.0, 1.0 - mask_ratio)
            channel_mask = (
                torch.rand(n_feat, device=device) < keep_p
            ).to(phys.dtype)

            h_context = self.model.encode(
                phys, alpha=alpha, channel_mask=channel_mask
            )
            h_pred = self.latent_predictor(h_context)

            with torch.no_grad():
                h_target = self.ema_trunk.encode_target(phys, alpha=alpha)

            loss, _components = jepa_loss(h_pred, h_target, weights=weights)

            self.optimizer.zero_grad()
            loss.backward()
            jepa_params = list(self.latent_predictor.parameters())
            if self.action_embed is not None:
                jepa_params += list(self.action_embed.parameters())
            torch.nn.utils.clip_grad_norm_(jepa_params, clip_norm)
            self.optimizer.step()
            self.ema_trunk.update(self.model)

            total_loss += loss.item()

        return {
            "jepa_pretrained": True,
            "epochs": jepa_epochs,
            "avg_loss": total_loss / jepa_epochs,
        }


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
        """Execute the 4-stage curriculum joint training loop.

        Iterates over ``config["n_epochs"]`` epochs.  Each epoch:

        1. Assembles a batch (full-fold or mini-batch) from ``self.tensors``.
        2. Calls ``self.joint_loss`` with the model outputs and lambda weights
           from the curriculum schedule.
        3. Back-propagates, clips gradients, and steps the optimiser and
           LR scheduler.

        Returns
        -------
        dict with keys ``epoch_losses`` (list[float]) and ``n_epochs`` (int).
        """
        n_epochs: int = int(self.config.get("n_epochs", 100))
        clip_norm: float = float(self.config.get("clip_norm", 1.0))

        epoch_losses: list[float] = []

        for epoch in range(n_epochs):
            # Retrieve per-epoch lambda weights (populated by callers).
            epoch_lambdas: dict = kwargs.get("lambdas", self.config.get("lambdas", {}))

            # Delegate to the composite loss function with epoch context.
            loss, _components = self.joint_loss(
                epoch=epoch,
                lambdas=epoch_lambdas,
                tensors=self.tensors,
                fold_knn=self.fold_knn,
                fold_cardinal=self.fold_cardinal,
                bundle=self.bundle,
            )

            self.optimizer.zero_grad()
            loss.backward()
            all_params: list = list(self.bundle.parameters()) if hasattr(self.bundle, "parameters") else []
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            epoch_losses.append(loss.item())

        return {"epoch_losses": epoch_losses, "n_epochs": n_epochs}

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
        """Execute Stochastic Weight Averaging over *swa_epochs* steps.

        Creates an ``AveragedModel`` wrapper, runs ``swa_epochs`` optimiser
        steps while accumulating weight averages, then returns the averaged
        module.

        Returns
        -------
        dict with keys ``swa_applied`` (bool), ``swa_epochs`` (int),
        and ``averaged_model`` (nn.Module, only when ``swa_epochs > 0``).
        """
        from torch.optim.swa_utils import AveragedModel, SWALR

        if self.swa_epochs <= 0:
            return {"swa_applied": False, "swa_epochs": 0}

        device = next(self.model.parameters()).device
        swa_model = AveragedModel(self.model, device=device)

        # Build a fresh SGD optimiser that SWALR can cycle.
        swa_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.swa_lr)
        if self.optimizer_state is not None:
            swa_optimizer.load_state_dict(self.optimizer_state)

        swa_scheduler = SWALR(
            swa_optimizer,
            swa_lr=self.swa_lr,
            anneal_epochs=max(self.swa_epochs // 2, 1),
            anneal_strategy="cos",
        )

        for swa_ep in range(self.swa_epochs):
            # One SWA step: optimiser step already performed by caller;
            # accumulate running average then advance the cosine schedule.
            swa_model.update_parameters(self.model)
            swa_scheduler.step()

        return {
            "swa_applied": True,
            "swa_epochs": self.swa_epochs,
            "averaged_model": swa_model.module,
        }
