"""TDD — JEPACurriculum (Front 4).

Concentrates the JEPA Phase-1 warm-up logic — EMA update schedule,
loss computation, predictor step — into a single testable module.

Cycles
------
1  JEPACurriculum.step(batch) returns a finite float loss.
2  step() increments an internal step counter.
3  checkpoint() returns a dict that includes 'step', 'online', 'predictor'.
4  JEPACurriculum can be constructed from_model (factory shortcut).
"""

from __future__ import annotations
import pytest
pytest.importorskip("torch")

import math
import torch
import pytest


# ---------------------------------------------------------------------------
# Minimal model factory
# ---------------------------------------------------------------------------

def _tiny_online():
    """Build the smallest valid SPARCMetaLearner for tests."""
    from sparc.models.neural_meta_config import NeuralMetaConfig
    from sparc.models.neural_meta import SPARCMetaLearner

    cfg = NeuralMetaConfig(
        n_base_models=2,
        n_physics_features=4,
        d_spatial=16,
        hidden_dim=16,
        n_heads=4,
    )
    return SPARCMetaLearner.from_config(cfg)


def _make_batch(n_physics: int = 4, n_base: int = 2, batch_size: int = 8):
    torch.manual_seed(0)
    return {
        "physics_feats": torch.randn(batch_size, n_physics),
        "alpha": torch.randn(batch_size, n_base),
    }


@pytest.fixture(scope="module")
def online():
    return _tiny_online()


# ---------------------------------------------------------------------------
# Cycle 1 — step returns a finite float loss
# ---------------------------------------------------------------------------

def test_curriculum_step_returns_finite_loss(online):
    """A single training step must return a non-NaN finite loss."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    curriculum = JEPACurriculum.from_model(online)
    batch = _make_batch()

    loss = curriculum.step(batch)

    assert isinstance(loss, float), f"expected float, got {type(loss)}"
    assert math.isfinite(loss), f"loss is not finite: {loss}"


# ---------------------------------------------------------------------------
# Cycle 2 — step increments step counter
# ---------------------------------------------------------------------------

def test_curriculum_step_increments_counter(online):
    """step_count must go up by 1 after each call to step()."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    curriculum = JEPACurriculum.from_model(online)
    batch = _make_batch()

    assert curriculum.step_count == 0
    curriculum.step(batch)
    assert curriculum.step_count == 1
    curriculum.step(batch)
    assert curriculum.step_count == 2


# ---------------------------------------------------------------------------
# Cycle 3 — checkpoint returns serialisable dict
# ---------------------------------------------------------------------------

def test_curriculum_checkpoint_has_required_keys(online):
    """checkpoint() must contain keys needed to resume training."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    curriculum = JEPACurriculum.from_model(online)
    curriculum.step(_make_batch())

    ckpt = curriculum.checkpoint()

    assert "step" in ckpt, "checkpoint must record step count"
    assert "online" in ckpt, "checkpoint must include online model state"
    assert "predictor" in ckpt, "checkpoint must include predictor state"


def test_curriculum_checkpoint_step_matches_step_count(online):
    from sparc.training.jepa_curriculum import JEPACurriculum

    curriculum = JEPACurriculum.from_model(online)
    for _ in range(3):
        curriculum.step(_make_batch())

    assert curriculum.checkpoint()["step"] == 3


# ---------------------------------------------------------------------------
# Cycle 4 — from_model factory
# ---------------------------------------------------------------------------

def test_curriculum_from_model_produces_curriculum():
    """from_model should return a JEPACurriculum, not crash."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    assert isinstance(curriculum, JEPACurriculum)


# ---------------------------------------------------------------------------
# Cycle 5 — step_scenario() Phase 2 scenario distillation (C1 deepening)
# ---------------------------------------------------------------------------

def _make_batch_with_coords(n_physics: int = 4, batch_size: int = 8):
    """Batch including coords, needed for scenario + latent-PDE steps."""
    torch.manual_seed(0)
    return {
        "physics_feats": torch.randn(batch_size, n_physics),
        "alpha": torch.rand(batch_size, 1) * 0.8 + 0.1,
        "coords": torch.rand(batch_size, 2),
    }


def test_step_scenario_returns_finite_loss():
    """step_scenario(batch, action_embed) must return a finite float.

    This tests the Phase 2 scenario-distillation path: perturb physics,
    ask the predictor to predict the target's EMA encoding of the perturbed view.
    """
    from sparc.training.jepa_curriculum import JEPACurriculum
    from sparc.models.action_embedding import ActionEmbedding

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    action_embed = ActionEmbedding(
        treatment_vocab=["feat_0", "feat_1", "feat_2", "feat_3"],
        embed_dim=16,
    )
    batch = _make_batch_with_coords()
    loss = curriculum.step_scenario(batch, action_embed, perturb_std=0.1)

    assert isinstance(loss, float), f"expected float, got {type(loss)}"
    assert math.isfinite(loss), f"step_scenario returned non-finite: {loss}"


def test_step_scenario_loss_is_non_negative():
    """VICReg-aligned JEPA loss is a sum of non-negative terms."""
    from sparc.training.jepa_curriculum import JEPACurriculum
    from sparc.models.action_embedding import ActionEmbedding

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    action_embed = ActionEmbedding(treatment_vocab=["feat_0"], embed_dim=16)
    loss = curriculum.step_scenario(_make_batch_with_coords(), action_embed)
    assert loss >= 0.0, f"loss should be non-negative, got {loss}"


# ---------------------------------------------------------------------------
# Cycle 6 — step_latent_pde() latent PDE consistency (C1 deepening)
# ---------------------------------------------------------------------------

def _make_neighbor_idx(n: int = 8, k: int = 3) -> "torch.Tensor":
    """Build valid KNN indices — no self-loops, indices within [0, n)."""
    idx = torch.zeros(n, k, dtype=torch.long)
    for i in range(n):
        candidates = [j for j in range(n) if j != i][:k]
        idx[i] = torch.tensor(candidates)
    return idx


def test_step_latent_pde_returns_finite_loss():
    """step_latent_pde(batch, neighbor_idx) must return a finite float."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    batch = _make_batch_with_coords()
    neighbor_idx = _make_neighbor_idx(n=batch["physics_feats"].shape[0])

    loss = curriculum.step_latent_pde(batch, neighbor_idx)

    assert isinstance(loss, float)
    assert math.isfinite(loss), f"step_latent_pde returned non-finite: {loss}"


def test_step_latent_pde_loss_is_non_negative():
    """Laplacian smoothness loss must be >= 0."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    batch = _make_batch_with_coords()
    neighbor_idx = _make_neighbor_idx(n=batch["physics_feats"].shape[0])
    loss = curriculum.step_latent_pde(batch, neighbor_idx)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# Cycle 7 — warm_up() multi-epoch convenience loop (C1 deepening)
# ---------------------------------------------------------------------------

def test_warm_up_returns_metrics_dict():
    """warm_up(batches, epochs) must return a dict with 'avg_loss'."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    batches = [_make_batch() for _ in range(3)]
    metrics = curriculum.warm_up(batches, epochs=2)

    assert isinstance(metrics, dict), "warm_up must return a dict"
    assert "avg_loss" in metrics, "metrics must contain 'avg_loss'"
    assert math.isfinite(metrics["avg_loss"])


def test_warm_up_updates_step_count():
    """warm_up(batches, epochs=N) must call step() len(batches)*N times."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    batches = [_make_batch() for _ in range(4)]
    curriculum.warm_up(batches, epochs=3)

    assert curriculum.step_count == 4 * 3


def test_warm_up_zero_epochs_is_no_op():
    """warm_up with epochs=0 must not modify step_count and return avg_loss=0."""
    from sparc.training.jepa_curriculum import JEPACurriculum

    online = _tiny_online()
    curriculum = JEPACurriculum.from_model(online)
    metrics = curriculum.warm_up([_make_batch()], epochs=0)

    assert curriculum.step_count == 0
    assert metrics["avg_loss"] == 0.0
