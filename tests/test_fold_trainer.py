"""
Tests for C4: FoldTrainer double-init fix.

Verifies that when FoldTrainer provides pre-built models via fold_models,
_exec_cv_fold does NOT re-construct model instances.

The observable behavior: SPARCMetaLearner.__init__ is never called when
fold_models is provided — no new model instance is created inside the fold.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("torch")
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_fold_state(device):
    """Build the smallest FoldState that gets past the KNN-building block."""
    from sparc.run.fold_state import (
        FoldState, _ArchConfig, _TrainingConfig, _JEPAConfig, _ContinualConfig,
    )

    N = 6
    n_phys = 4
    d_sp = 16

    coords_np = np.random.rand(N, 2).astype("float32")
    tensors = {
        "physics_feats":          torch.randn(N, n_phys, device=device),
        "physics_feats_extended": torch.randn(N, n_phys, device=device),
        "X_spatial":              torch.randn(N, d_sp, device=device),
        "coords":                 torch.tensor(coords_np, device=device),
        "y":                      torch.randn(N, device=device),
        "h_field":                None,
        "water_mask":             None,
        "time_idx":               None,
        "feat_mean":              np.zeros(n_phys, dtype="float32"),
        "feat_std":               np.ones(n_phys, dtype="float32"),
    }

    arch = _ArchConfig(
        hidden_dim=16, dropout=0.0, n_heads=2, max_neighbors=4,
        siren_omega=30.0, thresholds=[0.5], thresholds_norm=[0.0],
        time_embed_dim=0, geo_pe_dim=0, n_base=2, n_physics=n_phys,
        n_physics_extended=n_phys, d_spatial=d_sp, resolution=30.0,
        pr_input_dim=n_phys, pr_col_idxs=list(range(n_phys)),
        prior_mean=0.5, pr_cfg={}, n_treatments=1,
        material_priors={}, gwrf_k=2,
        predictor_bandwidths=None, predictor_kernel_field=None,
        feature_names=[f"f{i}" for i in range(n_phys)],
    )
    training = _TrainingConfig(
        n_epochs=0,  # no training epochs — only test the guard
        pretrain_epochs=0, pr_pretrain_epochs=0,
        lr=1e-3, clip_norm=1.0, warmup_epochs=0, ramp_epochs=0,
        batch_size=N, target_lambdas={},
        y_mean=0.0, y_std=1.0, n_folds=1,
        use_amp=False, use_cuda_graphs=False,
    )
    jepa = _JEPAConfig(
        enable=False, weights=None, mask_ratio=0.25,
        spatial_patch=False, n_patches=4, lam=0.0,
        ema_tau_start=0.99, ema_tau_end=0.9999, warmup_steps=0,
        curric_start=0, curric_end=0, lambda_scenario=0.0,
        lambda_latent_pde=0.0, scenario_perturb_std=0.0,
        predictor_blocks=1, predictor_film=False, action_dim=0,
    )
    continual = _ContinualConfig(
        ewc_active=False, ewc_lambda=0.0, cont_fisher=[],
        cont_optimal=[], ot_active=False, ot_lambda=0.0,
        coreset_acts=None,
    )

    return FoldState(
        tensors=tensors, coords_np=coords_np,
        alpha_targets=None, jepa_pretrained_trunk_state=None,
        base_oof_predictions=None,
        arch=arch, training=training, jepa=jepa, continual=continual,
    )


def _make_stub_fold_models(device):
    """Minimal FoldModels with a real tiny SPARCMetaLearner."""
    from sparc.models.neural_meta_config import NeuralMetaConfig
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.run.fold_trainer import FoldModels

    cfg = NeuralMetaConfig(
        n_base_models=2, n_physics_features=4, d_spatial=16,
        hidden_dim=16, n_heads=2, max_neighbors=4,
    )
    model = SPARCMetaLearner.from_config(cfg).to(device)

    def _mock_surrogate(return_tuple=True):
        m = MagicMock()
        m.parameters.return_value = iter([])
        m.eval.return_value = m
        m.train.return_value = m
        m.state_dict.return_value = {}
        if return_tuple:
            m.return_value = (torch.rand(6), None)
        else:
            m.return_value = torch.rand(6)
        return m

    surrogates = {
        "gwr":    _mock_surrogate(return_tuple=True),
        "gwrf":   _mock_surrogate(return_tuple=False),
        "ggpgam": _mock_surrogate(return_tuple=False),
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    return FoldModels(
        model=model, process_net=MagicMock(), source_net=MagicMock(),
        surrogates=surrogates, optimizer=optimizer, scheduler=scheduler,
    )


class TestFoldTrainerNoDoubleInit:
    """
    _exec_cv_fold called with fold_models must not instantiate SPARCMetaLearner.

    We spy on SPARCMetaLearner.__init__ to detect any new instantiation.
    The spy is installed AFTER fold_models is built, so only new instances
    created inside _exec_cv_fold are counted.
    """

    def test_no_model_construction_when_fold_models_provided(self):
        """
        SPARCMetaLearner.__init__ must never be called when fold_models is
        provided — the early-assignment guard must fire before construction.

        If the guard is AFTER construction (C4 double-init bug), __init__
        is called and the assertion fails.
        """
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.run.v2_neural_training import _exec_cv_fold

        device = torch.device("cpu")
        ss = _make_minimal_fold_state(device)
        fold_models = _make_stub_fold_models(device)

        N = ss.tensors["y"].shape[0]
        train_idx = np.arange(N - 1)
        test_idx  = np.array([N - 1])

        init_call_count = []
        original_init = SPARCMetaLearner.__init__

        def counting_init(self_inner, *args, **kwargs):
            init_call_count.append(1)
            return original_init(self_inner, *args, **kwargs)

        # Activate spy AFTER fold_models is created — only counts new inits
        with patch.object(SPARCMetaLearner, "__init__", counting_init):
            try:
                _exec_cv_fold(
                    fold_idx=0,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    ss=ss,
                    device=device,
                    fold_models=fold_models,
                )
            except Exception:
                pass  # only care about __init__ call count, not later errors

        assert init_call_count == [], (
            f"SPARCMetaLearner.__init__ was called {len(init_call_count)} time(s) "
            "inside _exec_cv_fold even though fold_models was provided — "
            "C4 double-init bug: move the fold_models guard before model construction."
        )
