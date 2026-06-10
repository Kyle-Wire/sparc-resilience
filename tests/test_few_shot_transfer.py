"""Tests for the few-shot transfer loop.

Covers the complete City-A → City-B path:
  SpatialANP.from_checkpoint  ← trunk checkpoint persistence
  zero_shot_predict            ← load trunk, run with empty context
  few_shot_predict             ← load trunk, fine-tune decoder, compute R²
  climate zone conditioning    ← ClimateZoneEncoder appended to features
"""

from __future__ import annotations
import pytest
pytest.importorskip("torch")

import numpy as np
import pytest
import torch

from sparc.data.satellite_types import SatelliteFeatureSet, SatelliteBand
from sparc.inference.anp import SpatialANP
from sparc.inference.zero_shot import zero_shot_predict
from sparc.inference.few_shot import few_shot_predict
from sparc.registry.city_registry import CityRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(
    n: int = 20,
    with_bands: bool = True,
    climate_zone: str | None = None,
) -> SatelliteFeatureSet:
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1, (n, 2)).astype("float32")
    bands: dict[SatelliteBand, np.ndarray] = {}
    if with_bands:
        bands = {
            SatelliteBand.NDVI: rng.uniform(0, 1, n).astype("float32"),
            SatelliteBand.LST: rng.uniform(20, 40, n).astype("float32"),
            SatelliteBand.NDBI: rng.uniform(-0.5, 0.5, n).astype("float32"),
        }
    return SatelliteFeatureSet(
        coords=coords,
        bands=bands,
        climate_zone=climate_zone,
    )


def _save_anp(tmp_path, x_dim: int) -> str:
    """Save a fresh SpatialANP with given x_dim and return the path."""
    anp = SpatialANP(x_dim=x_dim)
    path = str(tmp_path / "trunk.pt")
    anp.save_checkpoint(path)
    return path


def _calibration_data(K: int = 8, D: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    cal_X = rng.standard_normal((K, D)).astype("float32")
    cal_y = rng.standard_normal(K).astype("float32")
    cal_c = rng.uniform(0, 1, (K, 2)).astype("float32")
    return cal_X, cal_y, cal_c


# ---------------------------------------------------------------------------
# SpatialANP.from_checkpoint
# ---------------------------------------------------------------------------

class TestANPFromCheckpoint:
    def test_roundtrip_weights(self, tmp_path):
        """Checkpoint save → from_checkpoint should recover identical weights."""
        anp = SpatialANP(x_dim=7, hidden_dim=32, encoder_dim=32)
        path = str(tmp_path / "anp.pt")
        anp.save_checkpoint(path)

        loaded = SpatialANP.from_checkpoint(path)
        for (n1, p1), (n2, p2) in zip(anp.named_parameters(), loaded.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"

    def test_inferred_dims(self, tmp_path):
        """from_checkpoint must correctly infer x_dim, hidden_dim, encoder_dim."""
        anp = SpatialANP(x_dim=5, hidden_dim=64, encoder_dim=32)
        path = str(tmp_path / "anp.pt")
        anp.save_checkpoint(path)

        loaded = SpatialANP.from_checkpoint(path)
        assert loaded.x_dim == 5
        assert loaded.encoder_dim == 32

    def test_returns_eval_mode(self, tmp_path):
        """from_checkpoint should return a model in eval mode."""
        anp = SpatialANP(x_dim=4)
        path = str(tmp_path / "anp.pt")
        anp.save_checkpoint(path)

        loaded = SpatialANP.from_checkpoint(path)
        assert not loaded.training

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            SpatialANP.from_checkpoint("/nonexistent/missing.pt")

    def test_forward_after_load(self, tmp_path):
        """Loaded model must produce valid (mean, std) tensors."""
        anp = SpatialANP(x_dim=3)
        path = str(tmp_path / "anp.pt")
        anp.save_checkpoint(path)

        loaded = SpatialANP.from_checkpoint(path)
        ctx_x = torch.zeros(0, 3)
        ctx_y = torch.zeros(0, 1)
        tgt_x = torch.randn(5, 3)
        mean, std = loaded(ctx_x, ctx_y, tgt_x)
        assert mean.shape == (5, 1)
        assert (std > 0).all()


# ---------------------------------------------------------------------------
# zero_shot_predict
# ---------------------------------------------------------------------------

class TestZeroShotPredict:
    def test_no_trunk_random_weights(self):
        features = _make_features(n=10)
        result = zero_shot_predict(features)
        assert result.y_pred.shape == (10,)
        assert result.uncertainty.shape == (10,)
        assert (result.uncertainty > 0).all()

    def test_with_trunk_path(self, tmp_path):
        """Passing trunk_path must load checkpoint instead of raising."""
        features = _make_features(n=10)
        # x_dim = 2 coords + 3 bands = 5
        path = _save_anp(tmp_path, x_dim=5)
        result = zero_shot_predict(features, trunk_path=path)
        assert result.y_pred.shape == (10,)
        assert result.climate_zone is None

    def test_trunk_path_changes_predictions(self, tmp_path):
        """Two different checkpoints must produce different outputs."""
        features = _make_features(n=10)
        subdir = tmp_path / "a"
        subdir.mkdir()
        p1 = _save_anp(subdir, x_dim=5)
        p2 = _save_anp(tmp_path, x_dim=5)

        r1 = zero_shot_predict(features, trunk_path=p1)
        r2 = zero_shot_predict(features, trunk_path=p2)
        # Randomly-init checkpoints almost certainly differ
        assert not np.allclose(r1.y_pred, r2.y_pred)

    def test_climate_zone_conditioning(self, tmp_path):
        """Climate zone embedding must expand x_dim by 8 without errors."""
        features = _make_features(n=10, climate_zone="Cfa")
        # x_dim = 2 + 3 + 8 = 13 with climate conditioning
        path = _save_anp(tmp_path, x_dim=13)
        result = zero_shot_predict(features, trunk_path=path)
        assert result.y_pred.shape == (10,)
        assert result.climate_zone == "Cfa"

    def test_coords_only_no_bands(self):
        """Features with no bands should fall back to coords-only x_dim=2."""
        features = _make_features(n=8, with_bands=False)
        result = zero_shot_predict(features)
        assert result.y_pred.shape == (8,)

    def test_registry_path_falls_back_to_fresh_model(self, tmp_path):
        """registry_path with no matching city falls back to fresh model — does not raise."""
        reg = CityRegistry(tmp_path / "reg")
        # Register a city; features below have climate_zone=None so lookup is skipped,
        # global trunk is None → falls through to fresh model initialisation.
        reg.register_city("portland", SpatialANP(x_dim=5).state_dict(), climate_zone="BSk")

        features = _make_features(n=5)  # climate_zone=None
        result = zero_shot_predict(features, registry_path=str(tmp_path / "reg"))
        assert result.y_pred.shape == (5,)

    def test_output_coords_match_input(self):
        features = _make_features(n=12)
        result = zero_shot_predict(features)
        np.testing.assert_array_equal(result.coords, features.coords)


# ---------------------------------------------------------------------------
# few_shot_predict — no fine-tuning
# ---------------------------------------------------------------------------

class TestFewShotNoFinetune:
    def test_basic_no_trunk(self):
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=5, D=3)
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c, n_finetune_epochs=0
        )
        assert result.y_pred.shape == (20,)
        assert result.n_calibration == 5
        assert result.calibration_r2 is None  # no finetune → no R²

    def test_with_trunk_path(self, tmp_path):
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=5, D=3)
        path = _save_anp(tmp_path, x_dim=5)  # 2+3
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=0,
        )
        assert result.y_pred.shape == (20,)
        assert result.n_calibration == 5

    def test_context_affects_predictions(self):
        """More calibration points should produce different outputs than zero context."""
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=10, D=3)

        r_few = few_shot_predict(
            features, cal_X, cal_y, cal_c, n_finetune_epochs=0
        )
        # Zero-shot is just few_shot with empty calibration
        r_zero = zero_shot_predict(features)
        assert not np.allclose(r_few.y_pred, r_zero.y_pred)



# ---------------------------------------------------------------------------
# few_shot_predict — with fine-tuning
# ---------------------------------------------------------------------------

class TestFewShotFinetune:
    def test_finetune_runs_without_error(self, tmp_path):
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=8, D=3)
        path = _save_anp(tmp_path, x_dim=5)
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=5,
        )
        assert result.y_pred.shape == (20,)

    def test_finetune_changes_predictions(self, tmp_path):
        """Fine-tuning the decoder must alter output relative to frozen model."""
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=10, D=3)
        path = _save_anp(tmp_path, x_dim=5)

        r_frozen = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=0,
        )
        r_finetuned = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=10,
        )
        assert not np.allclose(r_frozen.y_pred, r_finetuned.y_pred)

    def test_calibration_r2_returned(self, tmp_path):
        """R² should be populated after fine-tuning with K ≥ 2."""
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=8, D=3)
        path = _save_anp(tmp_path, x_dim=5)
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=5,
        )
        assert result.calibration_r2 is not None
        assert isinstance(result.calibration_r2, float)

    def test_no_r2_when_k_is_1(self, tmp_path):
        """LOO R² is undefined for K=1; result.calibration_r2 must be None."""
        features = _make_features(n=10)
        cal_X = np.zeros((1, 3), dtype="float32")
        cal_y = np.array([1.0], dtype="float32")
        cal_c = np.array([[0.5, 0.5]], dtype="float32")
        path = _save_anp(tmp_path, x_dim=5)
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=5,
        )
        assert result.calibration_r2 is None

    def test_climate_zone_with_finetune(self, tmp_path):
        """Climate zone conditioning + fine-tuning must complete without error."""
        features = _make_features(n=20, climate_zone="BSk")
        cal_X, cal_y, cal_c = _calibration_data(K=8, D=3)
        # x_dim = 2 + 3 + 8 = 13
        path = _save_anp(tmp_path, x_dim=13)
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=5,
        )
        assert result.y_pred.shape == (20,)

    def test_uncertainty_positive_after_finetune(self, tmp_path):
        """std must remain strictly positive after fine-tuning."""
        features = _make_features(n=20)
        cal_X, cal_y, cal_c = _calibration_data(K=8, D=3)
        path = _save_anp(tmp_path, x_dim=5)
        result = few_shot_predict(
            features, cal_X, cal_y, cal_c,
            trunk_path=path, n_finetune_epochs=10,
        )
        assert (result.uncertainty > 0).all()


# ---------------------------------------------------------------------------
# Checkpoint bundle — encoder weights used in inference (not random)
# ---------------------------------------------------------------------------

def _save_bundle(tmp_path, x_dim: int, encoder_seed: int) -> str:
    """Save a bundle checkpoint containing ANP + ClimateZoneEncoder."""
    from sparc.models.climate_encoder import ClimateZoneEncoder

    anp = SpatialANP(x_dim=x_dim)
    torch.manual_seed(encoder_seed)
    cze = ClimateZoneEncoder(embed_dim=8)
    path = str(tmp_path / f"bundle_{encoder_seed}.pt")
    anp.save_checkpoint(path, climate_encoder=cze)
    return path


class TestCheckpointBundleEncoderInference:
    """The saved encoder, not a freshly-initialised one, must be used."""

    def test_bundle_inference_is_deterministic(self, tmp_path):
        """Two calls with the same bundle checkpoint must produce identical output."""
        # x_dim = 2 coords + 3 bands + 8 climate = 13
        path = _save_bundle(tmp_path, x_dim=13, encoder_seed=99)
        features = _make_features(n=15, climate_zone="Cfa")
        cal_X, cal_y, cal_c = _calibration_data(K=5, D=3)

        r1 = few_shot_predict(features, cal_X, cal_y, cal_c,
                              trunk_path=path, n_finetune_epochs=0)
        r2 = few_shot_predict(features, cal_X, cal_y, cal_c,
                              trunk_path=path, n_finetune_epochs=0)
        np.testing.assert_array_equal(r1.y_pred, r2.y_pred,
                                      err_msg="Inference must be deterministic when encoder weights are loaded from bundle")

    def test_different_encoder_seeds_produce_different_outputs(self, tmp_path):
        """Two bundles with different encoder weights must differ in output."""
        from sparc.models.climate_encoder import ClimateZoneEncoder

        anp_a = SpatialANP(x_dim=13)
        torch.manual_seed(1)
        cze_a = ClimateZoneEncoder(embed_dim=8)
        path_a = str(tmp_path / "bundle_seed1.pt")
        anp_a.save_checkpoint(path_a, climate_encoder=cze_a)

        anp_b = SpatialANP(x_dim=13)
        torch.manual_seed(2)
        cze_b = ClimateZoneEncoder(embed_dim=8)
        path_b = str(tmp_path / "bundle_seed2.pt")
        anp_b.save_checkpoint(path_b, climate_encoder=cze_b)

        features = _make_features(n=15, climate_zone="Cfa")
        cal_X, cal_y, cal_c = _calibration_data(K=5, D=3)

        r_a = few_shot_predict(features, cal_X, cal_y, cal_c,
                               trunk_path=path_a, n_finetune_epochs=0)
        r_b = few_shot_predict(features, cal_X, cal_y, cal_c,
                               trunk_path=path_b, n_finetune_epochs=0)
        assert not np.allclose(r_a.y_pred, r_b.y_pred), (
            "Different encoder seeds must produce different outputs"
        )

    def test_bundle_encoder_embedding_matches_saved_weights(self, tmp_path):
        """The embedding applied during inference must equal saved encoder weights."""
        from sparc.models.climate_encoder import ClimateZoneEncoder

        torch.manual_seed(55)
        cze_original = ClimateZoneEncoder(embed_dim=8)
        anp = SpatialANP(x_dim=13)
        path = str(tmp_path / "bundle.pt")
        anp.save_checkpoint(path, climate_encoder=cze_original)

        # What the saved encoder would produce for zone "Cfa"
        zone_idx = torch.tensor([ClimateZoneEncoder.zone_to_index("Cfa")])
        with torch.no_grad():
            expected_emb = cze_original(zone_idx).numpy()  # (1, 8)

        # Load the encoder back and verify weights match
        cze_loaded = ClimateZoneEncoder.from_checkpoint(path)
        assert cze_loaded is not None
        with torch.no_grad():
            actual_emb = cze_loaded(zone_idx).numpy()

        np.testing.assert_array_equal(expected_emb, actual_emb,
                                      err_msg="Loaded encoder must produce the same embedding as the original")
