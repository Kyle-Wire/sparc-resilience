"""Smoke-test: verify all public sparc sub-packages are importable."""
import importlib
import pytest

SUBPACKAGES = [
    "sparc",
    "sparc.config",
    "sparc.config.config",
    "sparc.data",
    "sparc.data.data_utils",
    "sparc.models",
    "sparc.models.gwen",
    "sparc.models.ols",
    "sparc.models.gwr",
    "sparc.models.gwrf",
    "sparc.models.ggpgam",
    "sparc.models.meta_ensemble",
    "sparc.models.deep_kriging_v2",
    "sparc.causal",
    "sparc.causal.dag_definition",
    "sparc.evaluation",
    "sparc.features",
    "sparc.features.laplacian",
    "sparc.interventions",
    "sparc.run",
    "sparc.run.pipeline_paths",
]


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_import(module: str) -> None:
    importlib.import_module(module)
