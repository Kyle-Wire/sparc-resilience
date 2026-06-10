"""
Tests for C3: SpatialTrunk protocol and TrunkLoader for SPARCMetaLearner.

Verifies:
  1. SPARCMetaLearner satisfies SpatialTrunk at runtime (has save/load_checkpoint).
  2. save_checkpoint writes a loadable file; load_checkpoint restores identical weights.
  3. load_checkpoint raises FileNotFoundError for missing paths.
  4. TrunkLoader.from_state_dict(model_type='sparc_meta') returns SPARCMetaLearner.
  5. TrunkLoader round-trip preserves weights.
  6. Regression: default TrunkLoader behaviour (SpatialANP) is unchanged.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_meta(hidden_dim: int = 16):
    from sparc.models.neural_meta_config import NeuralMetaConfig
    from sparc.models.neural_meta import SPARCMetaLearner

    cfg = NeuralMetaConfig(
        n_base_models=2,
        n_physics_features=4,
        d_spatial=16,
        hidden_dim=hidden_dim,
        n_heads=2,
        max_neighbors=4,
    )
    return SPARCMetaLearner.from_config(cfg)


# ---------------------------------------------------------------------------
# C3a — SpatialTrunk protocol
# ---------------------------------------------------------------------------

class TestSpatialTrunkProtocol:

    def test_sparc_meta_satisfies_protocol(self):
        """isinstance(SPARCMetaLearner(...), SpatialTrunk) must be True."""
        from sparc.inference.trunk import SpatialTrunk
        model = _tiny_meta()
        assert isinstance(model, SpatialTrunk), (
            "SPARCMetaLearner does not satisfy SpatialTrunk — "
            "add save_checkpoint / load_checkpoint methods"
        )

    def test_spatial_anp_regression_still_satisfies_protocol(self):
        """Regression: SpatialANP must still satisfy SpatialTrunk."""
        from sparc.inference.trunk import SpatialTrunk
        from sparc.inference.anp import SpatialANP
        model = SpatialANP(x_dim=4, hidden_dim=16, n_heads=2)
        assert isinstance(model, SpatialTrunk)

    def test_save_checkpoint_creates_file(self):
        """save_checkpoint must write a file at the given path."""
        model = _tiny_meta()
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "trunk.pt")
            model.save_checkpoint(path)
            assert Path(path).exists()

    def test_load_checkpoint_restores_weights(self):
        """load_checkpoint must exactly restore the saved parameters."""
        model_a = _tiny_meta()
        model_b = _tiny_meta()

        with torch.no_grad():
            for p in model_b.parameters():
                p.fill_(0.0)

        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "trunk.pt")
            model_a.save_checkpoint(path)
            model_b.load_checkpoint(path)

        for k, v_a in model_a.state_dict().items():
            v_b = model_b.state_dict()[k]
            assert torch.allclose(v_a, v_b), f"'{k}' mismatch after round-trip"

    def test_load_checkpoint_raises_for_missing_file(self):
        """load_checkpoint must raise FileNotFoundError when path does not exist."""
        model = _tiny_meta()
        with pytest.raises(FileNotFoundError):
            model.load_checkpoint("/nonexistent/trunk.pt")


# ---------------------------------------------------------------------------
# C3b — TrunkLoader with SPARCMetaLearner
# ---------------------------------------------------------------------------

class TestTrunkLoaderSPARCMeta:

    def test_from_state_dict_sparc_meta_returns_sparc_meta_instance(self):
        """TrunkLoader.from_state_dict with model_type='sparc_meta' returns SPARCMetaLearner."""
        from sparc.inference.trunk import TrunkLoader
        from sparc.models.neural_meta import SPARCMetaLearner

        src = _tiny_meta()
        restored = TrunkLoader.from_state_dict(
            src.state_dict(),
            model_type="sparc_meta",
            n_base_models=2,
            n_physics_features=4,
            d_spatial=16,
            hidden_dim=16,
        )
        assert isinstance(restored, SPARCMetaLearner)

    def test_from_state_dict_sparc_meta_preserves_weights(self):
        """All parameters must be identical after the round-trip."""
        from sparc.inference.trunk import TrunkLoader

        src = _tiny_meta()
        state = src.state_dict()

        restored = TrunkLoader.from_state_dict(
            state,
            model_type="sparc_meta",
            n_base_models=2,
            n_physics_features=4,
            d_spatial=16,
            hidden_dim=16,
        )

        for k, v in state.items():
            assert k in restored.state_dict(), f"key '{k}' missing after load"
            assert torch.allclose(v, restored.state_dict()[k]), f"'{k}' differs"

    def test_from_state_dict_default_anp_unchanged(self):
        """Regression: omitting model_type still returns SpatialANP."""
        from sparc.inference.trunk import TrunkLoader
        from sparc.inference.anp import SpatialANP

        # Use default SpatialANP dims so the round-trip succeeds without
        # passing explicit hidden_dim / n_heads to from_state_dict.
        src = SpatialANP(x_dim=4)
        restored = TrunkLoader.from_state_dict(src.state_dict(), x_dim=4)
        assert isinstance(restored, SpatialANP)
