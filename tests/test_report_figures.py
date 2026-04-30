"""Tests for sparc.report.figures.

Keys registered in :data:`FIGURES_REGISTRY` must match what the
pipeline actually writes via :class:`ArtifactStore`. These tests lock
in the current contract; if a producer changes its (stage, artifact_id)
this file will fail and force the registry to be re-keyed in lock-step.
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.report.figures import (
    FIGURES_REGISTRY,
    FIGURES_REGISTRY_PREFIX,
    FigureRenderError,
    render_for_artifact,
)
from sparc.report.render import render_png


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    return ArtifactStore(reg)


def _is_png(b: bytes) -> bool:
    return len(b) > 8 and b[:8] == b"\x89PNG\r\n\x1a\n"


def test_registry_includes_expected_keys():
    """The registry must mirror real pipeline producers (stage / artifact_id)."""
    expected_subset = {
        # Stage 0 — EDA
        ("0", "correlogram_results"),
        # Stage 2 — Spatial CV
        ("2", "spatial_cv_predictions"),
        ("2", "ensemble_predictions"),
        # Stage 3 — Causal validation
        ("3", "median_probability_dag"),
        ("3", "edge_inclusion_probs"),
        ("3", "cate_summary"),
        ("3", "spatial_cate_maps"),
        ("3", "dose_response_curves"),
        ("3", "parameter_posteriors"),
        # Stage 4 — Scenario contract
        ("4", "scenario_results"),
        ("4", "scenario_results_dag"),
        ("4", "scenario_results_reprediction"),
        ("4", "scenario_results_hybrid"),
    }
    assert expected_subset.issubset(set(FIGURES_REGISTRY.keys()))


def test_registry_excludes_keys_with_no_producer():
    """Keys whose producers were dropped must not silently shadow real artifacts."""
    stale_keys = {
        ("0", "data"),
        ("0", "data_summary"),
        ("0", "data_histogram_bins"),
        ("1", "dag"),
        ("1", "edge_inclusion_probs"),
        ("1", "correlogram"),
        ("2", "predictions"),
        ("2", "cate"),
        ("2", "spatial_cate"),
        ("2", "pdp"),
        ("3", "sensitivity_tornado"),
        ("3", "dose_response"),
        ("3", "posteriors"),
        ("3", "posterior_trace"),
        ("5", "decision_allocation"),
    }
    assert stale_keys.isdisjoint(set(FIGURES_REGISTRY.keys()))


def test_render_correlogram_modern_schema(store):
    """Stage 0 ``correlogram_results`` uses the per-variable individual_results layout."""
    payload = {
        "individual_results": {
            "var_a": {
                "lag_distances": [100.0, 200.0, 300.0, 400.0],
                "morans_i_values": [0.45, 0.30, 0.10, -0.05],
                "p_values": [0.001, 0.01, 0.05, 0.5],
            },
            "var_b": {
                "lag_distances": [100.0, 200.0, 300.0, 400.0],
                "morans_i_values": [0.20, 0.05, -0.02, -0.10],
                "p_values": [0.02, 0.1, 0.4, 0.6],
            },
        }
    }
    store.write_struct("0", "correlogram_results", payload)
    png = render_for_artifact(
        "0", "correlogram_results", registry=store.registry, use_cache=False,
    )
    assert _is_png(png)


def test_pdp_prefix_renderer_resolves(store):
    """Per-feature PDP artifacts resolve via FIGURES_REGISTRY_PREFIX."""
    assert ("2", "v2_neural_pdp::") in FIGURES_REGISTRY_PREFIX
    df = pd.DataFrame({"x": np.linspace(0, 1, 10), "pdp": np.linspace(0, 0.5, 10)})
    store.write_table("2", "v2_neural_pdp::feature_alpha", df)
    png = render_for_artifact(
        "2", "v2_neural_pdp::feature_alpha", registry=store.registry, use_cache=False,
    )
    assert _is_png(png)


def test_render_for_artifact_caches_correlogram(store):
    payload = {
        "individual_results": {
            "v": {"lag_distances": [1.0, 2.0], "morans_i_values": [0.3, 0.1]}
        }
    }
    store.write_struct("0", "correlogram_results", payload)

    png1 = render_for_artifact("0", "correlogram_results", registry=store.registry)
    assert _is_png(png1)
    assert store.has("report_figures", "0__correlogram_results")

    png2 = render_for_artifact("0", "correlogram_results", registry=store.registry)
    assert png1 == png2


def test_render_for_artifact_unknown_renderer_raises(store):
    df = pd.DataFrame({"a": [1, 2]})
    store.write_table("99", "no_renderer", df)
    with pytest.raises(FigureRenderError):
        render_for_artifact("99", "no_renderer", registry=store.registry)


def test_render_png_delegates_to_figures(store):
    payload = {
        "individual_results": {
            "v": {"lag_distances": [1.0, 2.0], "morans_i_values": [0.3, 0.1]}
        }
    }
    store.write_struct("0", "correlogram_results", payload)
    png = render_png(
        "0", "correlogram_results", registry=store.registry, use_cache=False,
    )
    assert _is_png(png)
