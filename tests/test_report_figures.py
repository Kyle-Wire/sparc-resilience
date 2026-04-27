"""Phase 3 tests for sparc.report.figures."""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.report.figures import (
    FIGURES_REGISTRY,
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
    expected_subset = {
        ("0", "data"),
        ("1", "dag"),
        ("1", "correlogram"),
        ("2", "predictions"),
        ("2", "cate"),
        ("3", "sensitivity_tornado"),
        ("3", "posteriors"),
        ("3", "posterior_trace"),
        ("4", "scenario_results"),
        ("4", "scenario_attribution"),
        ("4", "scenario_trajectory"),
        ("5", "decision_allocation"),
    }
    assert expected_subset.issubset(set(FIGURES_REGISTRY.keys()))


def test_render_data_histogram(store):
    df = pd.DataFrame({"target": np.random.RandomState(0).normal(size=200)})
    store.write_table("0", "data", df)
    png = render_for_artifact("0", "data", registry=store.registry, use_cache=False)
    assert _is_png(png)


def test_render_correlogram(store):
    payload = {
        "matrix": [[1.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 1.0]],
        "labels": ["a", "b", "c"],
    }
    store.write_struct("1", "correlogram", payload)
    png = render_for_artifact("1", "correlogram", registry=store.registry, use_cache=False)
    assert _is_png(png)


def test_render_sensitivity_tornado(store):
    payload = {
        "items": [
            {"name": "x1", "effect": 0.6, "low": 0.4, "high": 0.8},
            {"name": "x2", "effect": -0.3, "low": -0.5, "high": -0.1},
            {"name": "x3", "effect": 0.1, "low": 0.0, "high": 0.2},
        ]
    }
    store.write_struct("3", "sensitivity_tornado", payload)
    png = render_for_artifact("3", "sensitivity_tornado", registry=store.registry, use_cache=False)
    assert _is_png(png)


def test_render_posterior_trace(store):
    rng = np.random.RandomState(0)
    payload = {"chains": {"alpha": rng.normal(size=200).tolist(), "beta": rng.normal(size=200).tolist()}}
    store.write_struct("3", "posterior_trace", payload)
    png = render_for_artifact("3", "posterior_trace", registry=store.registry, use_cache=False)
    assert _is_png(png)


def test_render_attribution_waterfall(store):
    df = pd.DataFrame(
        {"driver": ["A", "B", "C", "D"], "contribution": [0.4, -0.2, 0.1, -0.05]}
    )
    store.write_table("4", "scenario_attribution", df)
    png = render_for_artifact(
        "4", "scenario_attribution", registry=store.registry, use_cache=False
    )
    assert _is_png(png)


def test_render_scenario_trajectory(store):
    df = pd.DataFrame(
        {
            "t": np.arange(20),
            "pred_Scenario1": np.linspace(1.0, 1.5, 20),
            "pred_Scenario2": np.linspace(1.0, 1.2, 20),
        }
    )
    store.write_table("4", "scenario_trajectory", df)
    png = render_for_artifact(
        "4", "scenario_trajectory", registry=store.registry, use_cache=False
    )
    assert _is_png(png)


def test_render_scenario_library_timeline(store):
    payload = {
        "events": [
            {"name": "baseline", "t": 2025, "kind": "ref"},
            {"name": "intervention", "t": 2030, "kind": "scenario"},
        ]
    }
    store.write_struct("4", "scenario_library", payload)
    png = render_for_artifact(
        "4", "scenario_library", registry=store.registry, use_cache=False
    )
    assert _is_png(png)


def test_render_decision_allocation(store):
    payload = {
        "allocations": [
            {"name": "site A", "value": 12.5},
            {"name": "site B", "value": 7.2},
            {"name": "site C", "value": 3.1},
        ]
    }
    store.write_struct("5", "decision_allocation", payload)
    png = render_for_artifact("5", "decision_allocation", registry=store.registry, use_cache=False)
    assert _is_png(png)


def test_render_for_artifact_caches_to_report_figures(store):
    df = pd.DataFrame({"y": np.arange(10, dtype=float)})
    store.write_table("0", "data", df)

    # First render writes to cache.
    png1 = render_for_artifact("0", "data", registry=store.registry)
    assert _is_png(png1)
    assert store.has("report_figures", "0__data")

    # Second render hits cache (bytes-identical).
    png2 = render_for_artifact("0", "data", registry=store.registry)
    assert png1 == png2


def test_render_for_artifact_unknown_renderer_raises(store):
    df = pd.DataFrame({"a": [1, 2]})
    store.write_table("99", "no_renderer", df)
    with pytest.raises(FigureRenderError):
        render_for_artifact("99", "no_renderer", registry=store.registry)


def test_render_png_delegates_to_figures(store):
    payload = {"matrix": [[1.0, 0.0], [0.0, 1.0]], "labels": ["x", "y"]}
    store.write_struct("1", "correlogram", payload)
    # render_png lives on render.py and forwards to figures.render_for_artifact.
    png = render_png("1", "correlogram", registry=store.registry, use_cache=False)
    assert _is_png(png)
