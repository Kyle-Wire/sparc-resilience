"""TDD — C4: sparc.models public interface only exposes SPARCMetaLearner, not internals.

Verifies that:
1. Internal neural sub-modules are NOT in ``sparc.models.__all__``
2. ``SPARCMetaLearner`` remains the single neural export in ``__all__``
3. Model classes (GWRModel, GWENModel, etc.) remain accessible
4. Internal sub-modules are still importable from their own paths (no breakage)
"""

from __future__ import annotations

import pytest


_INTERNAL_NEURAL_NAMES = [
    "SIRENLayer",
    "SparseSpatialAttention",
    "ProcessRateNet",
    "DifferentiableGWR",
    "DifferentiableGWRF",
    "DifferentiableGGPGAM",
]

_PUBLIC_BASE_MODELS = [
    "GWENModel",
    "OLSModel",
    "GWRModel",
    "GWRFModel",
    "GGPGAM_SVC",
    "ModelSpec",
]


class TestModelsPublicInterface:
    def test_internal_neural_not_in_all(self):
        import sparc.models as models
        all_names = set(getattr(models, "__all__", []))
        for name in _INTERNAL_NEURAL_NAMES:
            assert name not in all_names, (
                f"{name!r} is an internal neural sub-module and should not be "
                f"in sparc.models.__all__. Access via sparc.models.<submodule> instead."
            )

    def test_sparc_meta_learner_in_all(self):
        """SPARCMetaLearner is the one intended neural export."""
        import sparc.models as models
        all_names = set(getattr(models, "__all__", []))
        assert "SPARCMetaLearner" in all_names

    def test_base_models_still_in_all(self):
        import sparc.models as models
        all_names = set(getattr(models, "__all__", []))
        for name in _PUBLIC_BASE_MODELS:
            assert name in all_names, f"Public base model {name!r} missing from __all__"

    def test_internal_neural_not_top_level_attrs(self):
        """Internal sub-modules should not pollute the top-level namespace."""
        import sparc.models as models
        for name in _INTERNAL_NEURAL_NAMES:
            assert not hasattr(models, name), (
                f"{name!r} should not be a top-level attribute of sparc.models. "
                "Import from its own sub-module (e.g. sparc.models.spatial_attention)."
            )

    def test_siren_layer_importable_from_submodule(self):
        """SIRENLayer can still be imported directly from its sub-module."""
        pytest.importorskip("torch")
        from sparc.models.spatial_attention import SIRENLayer
        assert SIRENLayer is not None

    def test_sparse_spatial_attention_importable_from_submodule(self):
        pytest.importorskip("torch")
        from sparc.models.spatial_attention import SparseSpatialAttention
        assert SparseSpatialAttention is not None

    def test_process_rate_net_importable_from_submodule(self):
        pytest.importorskip("torch")
        from sparc.models.process_rate_net import ProcessRateNet
        assert ProcessRateNet is not None

    def test_differentiable_gwr_importable_from_submodule(self):
        pytest.importorskip("torch")
        from sparc.models.surrogates import DifferentiableGWR
        assert DifferentiableGWR is not None
