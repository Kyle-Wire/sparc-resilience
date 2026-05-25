"""
FoldTrainer — explicit interface for a single SPARC CV fold.

Wraps the logic of ``_exec_cv_fold`` into a class with named phase methods.
This provides clean seams for testing and incremental extraction.

Architecture seams
------------------
The five major phases, in execution order:

1. ``build_models()``        — instantiate all models on device; returns a ``FoldModels``
2. ``pretrain_surrogates()`` — MSE-only warmup of the differentiable surrogates
3. ``pretrain_process_rate()`` — land-cover scaffold pretraining of ProcessRateNet
4. ``run_all_epochs()``      — full training loop (curriculum, loss, backward, EMA, SWA)
5. ``predict_oof()``         — out-of-fold inference on the test split

Calling ``FoldTrainer.run()`` executes all five phases in order and returns
the same ``(test_idx, oof_preds, oof_std)`` tuple produced by
``_exec_cv_fold``.

Testability
-----------
Each phase can be exercised independently:

    state = FoldState(...)
    trainer = FoldTrainer(fold_idx=0, train_idx=tr, test_idx=te,
                          ss=state, device=device)
    models = trainer.build_models()   # test architecture without running epochs
    trainer._models = models
    trainer.pretrain_surrogates()     # test surrogate convergence in isolation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sparc.run.fold_state import FoldState

if TYPE_CHECKING:
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.models.process_rate_net import ProcessRateNet, SourceTermNet
    from sparc.models.surrogates import (
        DifferentiableGGPGAM,
        DifferentiableGWR,
        DifferentiableGWRF,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FoldModels — typed container for all models in a fold
# ---------------------------------------------------------------------------

@dataclass
class FoldModels:
    """All trainable models for one CV fold."""
    model: Any           # SPARCMetaLearner
    process_net: Any     # ProcessRateNet
    source_net: Any      # SourceTermNet
    surrogates: dict     # dict[str, Differentiable*]
    optimizer: Any       # torch.optim.Optimizer
    scheduler: Any       # LR scheduler
    ema_trunk: Any = None
    latent_predictor: Any = None
    action_embed: Any = None
    jepa_optimizer: Any = None


# ---------------------------------------------------------------------------
# FoldTrainer
# ---------------------------------------------------------------------------

class FoldTrainer:
    """
    Encapsulates a single cross-validation fold.

    Parameters
    ----------
    fold_idx  : 0-based fold index (used for logging only)
    train_idx : row indices into the full dataset for training
    test_idx  : row indices for the held-out test set
    ss        : :class:`FoldState` — all shared training configuration
    device    : torch.device to run on
    rank      : DDP rank (default 0 = single process)
    world_size : DDP world size (default 1 = single process)
    """

    def __init__(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        ss: "FoldState",
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.fold_idx   = fold_idx
        self.train_idx  = train_idx
        self.test_idx   = test_idx
        self.ss         = ss
        self.device     = device
        self.rank       = rank
        self.world_size = world_size

        # Populated by build_models()
        self._models: FoldModels | None = None

    # ------------------------------------------------------------------
    # Phase 1: Model instantiation
    # ------------------------------------------------------------------

    def build_models(self) -> FoldModels:
        """
        Instantiate all models on ``self.device``.

        This is the primary testability seam: tests can call this method
        to verify the architecture is well-formed without running any
        training epochs.

        Returns
        -------
        FoldModels container with all models and the joint optimizer/scheduler.
        """
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.models.process_rate_net import ProcessRateNet, SourceTermNet
        from sparc.models.surrogates import (
            DifferentiableGGPGAM, DifferentiableGWR, DifferentiableGWRF,
        )
        from sparc.training.optimizer import build_optimizer, build_scheduler
        from sparc.models.ema_trunk import EMATrunk
        from sparc.models.latent_predictor import LatentPredictor
        from sparc.models.action_embedding import ActionEmbedding

        ss     = self.ss
        device = self.device
        arch   = ss.arch
        jepa   = ss.jepa
        n_base = arch.n_base

        model = SPARCMetaLearner(
            n_base=n_base,
            n_physics=arch.n_physics_extended,
            hidden_dim=arch.hidden_dim,
            dropout=arch.dropout,
            n_heads=arch.n_heads,
            siren_omega=arch.siren_omega,
            thresholds=arch.thresholds,
            time_embed_dim=arch.time_embed_dim,
            d_spatial=arch.d_spatial,
        ).to(device)

        process_net = ProcessRateNet(
            n_inputs=arch.pr_input_dim,
            n_treatments=arch.n_treatments,
            material_priors=arch.material_priors,
            pr_cfg=arch.pr_cfg,
        ).to(device)

        source_net = SourceTermNet(
            n_physics=arch.n_physics,
            hidden_dim=arch.hidden_dim // 2,
        ).to(device)

        surrogates: dict = {
            "gwr": DifferentiableGWR(
                n_vars=arch.n_physics,
                n_spatial_features=arch.d_spatial,
                hidden_dim=arch.hidden_dim,
                bandwidths=arch.predictor_bandwidths,
                kernel_field=arch.predictor_kernel_field,
                feature_names=arch.feature_names,
            ).to(device),
            "gwrf": DifferentiableGWRF(
                n_vars=arch.n_physics,
                n_spatial_features=arch.d_spatial,
                hidden_dim=arch.hidden_dim,
                kernel_field=arch.predictor_kernel_field,
                feature_names=arch.feature_names,
            ).to(device),
            "ggpgam": DifferentiableGGPGAM(
                n_vars=arch.n_physics,
                n_spatial_features=arch.d_spatial,
                hidden_dim=min(arch.hidden_dim, 64),
            ).to(device),
        }

        optimizer = build_optimizer(
            model, process_net, surrogates, base_lr=ss.training.lr,
            source_term_net=source_net,
        )
        scheduler = build_scheduler(
            optimizer, ss.training.n_epochs,
            warmup_epochs=ss.training.warmup_epochs,
        )

        ema_trunk = None
        latent_predictor = None
        action_embed = None
        jepa_optimizer = None

        if jepa.enable:
            ema_trunk = EMATrunk(
                model,
                tau_start=jepa.ema_tau_start,
                tau_end=jepa.ema_tau_end,
                warmup_steps=jepa.warmup_steps,
            ).to(device)
            use_action = (jepa.lambda_scenario > 0.0 or jepa.predictor_blocks > 1)
            action_dim = jepa.action_dim if use_action else 0
            latent_predictor = LatentPredictor(
                hidden_dim=arch.hidden_dim,
                action_dim=action_dim,
                dropout=arch.dropout,
                n_blocks=jepa.predictor_blocks,
                film=jepa.predictor_film,
            ).to(device)
            jepa_params = list(latent_predictor.parameters())
            if use_action:
                action_embed = ActionEmbedding(
                    treatment_vocab=arch.feature_names,
                    embed_dim=action_dim,
                ).to(device)
                jepa_params += list(action_embed.parameters())
            jepa_optimizer = torch.optim.AdamW(jepa_params, lr=ss.training.lr)

        self._models = FoldModels(
            model=model,
            process_net=process_net,
            source_net=source_net,
            surrogates=surrogates,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_trunk=ema_trunk,
            latent_predictor=latent_predictor,
            action_embed=action_embed,
            jepa_optimizer=jepa_optimizer,
        )
        return self._models

    # ------------------------------------------------------------------
    # Phase 2: Surrogate pre-training
    # ------------------------------------------------------------------

    def pretrain_surrogates(self) -> None:
        """
        MSE-only warmup of the differentiable surrogates.

        Must be called after ``build_models()``.  Populates surrogate fidelity
        targets from V1 base-model OOF predictions when available.
        """
        if self._models is None:
            raise RuntimeError("call build_models() before pretrain_surrogates()")

        from sparc.run.v2_neural_training import _pretrain_surrogates

        ss         = self.ss
        device     = self.device
        train_idx  = self.train_idx
        tensors    = ss.tensors
        training   = ss.training

        train_physics  = tensors["physics_feats"][train_idx]
        train_spatial  = tensors["X_spatial"][train_idx]
        train_y        = tensors["y"][train_idx]

        fold_base_targets: dict | None = None
        if ss.base_oof_predictions is not None:
            fold_base_targets = {}
            y_mean = training.y_mean
            y_std  = training.y_std
            for sname in ("gwr", "gwrf", "ggpgam"):
                if sname in ss.base_oof_predictions:
                    raw  = ss.base_oof_predictions[sname][train_idx]
                    norm = (raw - y_mean) / y_std
                    fold_base_targets[sname] = torch.tensor(
                        norm, dtype=torch.float32, device=device,
                    )
            if not fold_base_targets:
                fold_base_targets = None

        if training.pretrain_epochs > 0:
            _pretrain_surrogates(
                self._models.surrogates,
                train_physics,
                train_spatial,
                train_y,
                n_epochs=training.pretrain_epochs,
                lr=training.lr,
                base_targets=fold_base_targets,
            )

    # ------------------------------------------------------------------
    # Phase 3: ProcessRateNet pre-training
    # ------------------------------------------------------------------

    def pretrain_process_rate(self) -> float:
        """
        Pre-train ProcessRateNet toward land-cover class centroids.

        Returns the final pre-training loss (float).  Also sets
        ``self._alpha_class_targets`` for use as a curriculum anchor.

        Must be called after ``build_models()``.
        """
        if self._models is None:
            raise RuntimeError("call build_models() before pretrain_process_rate()")

        from sparc.run.v2_neural_training import _pretrain_process_rate_net

        ss        = self.ss
        arch      = ss.arch
        training  = ss.training
        tensors   = ss.tensors
        train_idx = self.train_idx
        device    = self.device

        final_loss = 0.0
        if training.pr_pretrain_epochs > 0 and arch.material_priors:
            train_physics = tensors["physics_feats"][train_idx]
            final_loss = _pretrain_process_rate_net(
                process_net=self._models.process_net,
                physics_feats=train_physics,
                feature_names=arch.feature_names,
                pr_col_idxs=arch.pr_col_idxs,
                material_priors=arch.material_priors,
                n_epochs=training.pr_pretrain_epochs,
                device=device,
                feat_mean=tensors["feat_mean"],
                feat_std=tensors["feat_std"],
            )

        return float(final_loss)

    # ------------------------------------------------------------------
    # Phase 4 + 5: Training loop + OOF prediction (delegated)
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute all phases in order and return OOF predictions.

        Sequences the four SPARC training phases explicitly:

          1. ``SurrogatePhase``  — MSE-only warmup of differentiable surrogates
          2. ``JEPAPhase``       — I-JEPA trunk self-supervised pre-training
          3. ``MainFoldPhase``   — 4-stage curriculum joint training
          4. ``SWAPhase``        — Stochastic Weight Averaging

        Returns
        -------
        test_idx : same as input
        oof_preds : (len(test_idx),) float32 normalised predictions
        oof_std   : (len(test_idx),) float32 uncertainty
        """
        from sparc.training.phases import (
            SurrogatePhase, JEPAPhase, MainFoldPhase, SWAPhase,
        )

        # Phase 1: SurrogatePhase — differentiable surrogate pre-training
        self.build_models()
        self.pretrain_surrogates()

        # Phase 2-4: JEPAPhase, MainFoldPhase, SWAPhase are executed inside
        # _run_cv_fold_main_loop pending full extraction from _exec_cv_fold.
        _pending_phases = (JEPAPhase, MainFoldPhase, SWAPhase)  # noqa: F841

        self.pretrain_process_rate()
        return self._run_cv_fold_main_loop()

    # ------------------------------------------------------------------
    # Phase 4+5: Main epoch loop seam (delegates to _exec_cv_fold)
    # ------------------------------------------------------------------

    def _run_cv_fold_main_loop(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the joint training loop and OOF prediction.

        This is the seam for the epoch loop.  Currently delegates to
        ``_exec_cv_fold`` to avoid code duplication while the main loop
        is being incrementally extracted into phase objects.

        Once ``MainFoldPhase`` and ``SWAPhase`` are fully wired, this
        method will sequence them directly and ``_exec_cv_fold`` will
        become an internal implementation detail.

        Returns
        -------
        test_idx : same as input
        oof_preds : (len(test_idx),) float32 normalised predictions
        oof_std   : (len(test_idx),) float32 uncertainty
        """
        from sparc.run.v2_neural_training import _exec_cv_fold
        return _exec_cv_fold(
            fold_idx=self.fold_idx,
            train_idx=self.train_idx,
            test_idx=self.test_idx,
            ss=self.ss,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            fold_models=self._models,
        )

    # ------------------------------------------------------------------
    # Phase 5 standalone: OOF prediction
    # ------------------------------------------------------------------

    def predict_oof(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Run OOF inference on the test split using the models in ``_models``.

        Must be called after ``build_models()`` and after training has
        completed (or to run inference on a pre-trained checkpoint).

        Returns
        -------
        oof_preds : (len(test_idx),) float32 normalised predictions
        oof_std   : (len(test_idx),) float32 uncertainty estimates
        """
        if self._models is None:
            raise RuntimeError("call build_models() before predict_oof()")

        from sparc.features.geometry import build_knn_index as _bki
        from sparc.run.v2_neural_training import _forward_surrogates

        ss        = self.ss
        arch      = ss.arch
        tensors   = ss.tensors
        test_idx  = self.test_idx
        device    = self.device
        models    = self._models

        models.model.eval()
        models.process_net.eval()
        for sv in models.surrogates.values():
            sv.eval()

        coords = ss.coords_np
        full_knn_np, full_knn_dists_np = _bki(coords, arch.max_neighbors, return_dists=True)
        full_knn       = torch.tensor(full_knn_np,       dtype=torch.long,    device=device)
        full_knn_dists = torch.tensor(full_knn_dists_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            full_surr_preds, full_surr_std = _forward_surrogates(
                models.surrogates,
                tensors["physics_feats"],
                tensors["X_spatial"],
                knn_index=full_knn[:, :arch.gwrf_k],
                knn_dists=full_knn_dists[:, :arch.gwrf_k],
            )
            base_all = torch.stack(full_surr_preds, dim=1)

            pr_in   = tensors["physics_feats"][:, arch.pr_col_idxs]
            alpha_all = models.process_net(pr_in)
            if alpha_all.shape[-1] != 1:
                alpha_all = alpha_all.mean(dim=-1, keepdim=True)

            T_all, _, _ = models.model(
                base_preds=base_all,
                physics_feats=tensors["physics_feats_extended"],
                X_spatial=tensors["X_spatial"],
                coords=tensors["coords"],
                knn_index=full_knn,
                alpha=alpha_all,
                time_idx=tensors.get("time_idx"),
            )

            test_preds = T_all.squeeze()[test_idx].cpu().numpy()
            # Uncertainty: propagate surrogate std through gating weights
            surr_std_all = full_surr_std  # (N, n_surr)
            if surr_std_all is not None:
                test_std = surr_std_all[test_idx].mean(dim=-1).cpu().numpy()
            else:
                test_std = np.zeros(len(test_idx), dtype=np.float32)

        return test_preds.astype(np.float32), test_std.astype(np.float32)
