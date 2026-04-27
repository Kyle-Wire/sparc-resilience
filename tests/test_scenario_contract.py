"""Phase 0 contract tests for the scenario artifact bundle.

These tests pin the names, column conventions, and mode-variant precedence
that downstream consumers (server endpoints, report figures, desktop UI)
rely on.  Any change here is a contract change and must be versioned.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.scenario import (
    ScenarioBundle,
    SCENARIO_RESULTS_VARIANTS,
    SCENARIO_STAGE,
    delta_column,
    parse_scenario_index,
    pred_column,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(RunRegistry(tmp_path))


def _scenario_frame(n_scenarios: int = 2, n_rows: int = 3) -> pd.DataFrame:
    """Build a synthetic scenario_results-shaped DataFrame."""
    cols: dict[str, list[float]] = {
        "id": list(range(n_rows)),
        "pred_baseline": [10.0 + i for i in range(n_rows)],
    }
    for s in range(1, n_scenarios + 1):
        cols[pred_column(s)] = [10.0 + s + i for i in range(n_rows)]
        cols[delta_column(s)] = [float(s) for _ in range(n_rows)]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Contract: names + column convention
# ---------------------------------------------------------------------------


def test_results_variants_precedence_order():
    # Most-specific -> least-specific.
    assert SCENARIO_RESULTS_VARIANTS == (
        "scenario_results_hybrid",
        "scenario_results_reprediction",
        "scenario_results_dag",
        "scenario_results",
    )


def test_pred_column_naming_round_trips():
    for s in (1, 2, 17):
        col = pred_column(s)
        assert col == f"pred_Scenario{s}"
        assert parse_scenario_index(col) == s


def test_delta_column_naming():
    assert delta_column(3) == "delta_Scenario3"


def test_parse_scenario_index_rejects_unrelated_columns():
    for col in ("pred_baseline", "delta_Scenario1", "scenario", "pred_other", ""):
        assert parse_scenario_index(col) is None


# ---------------------------------------------------------------------------
# Bundle: empty, basic, multi-scenario
# ---------------------------------------------------------------------------


def test_bundle_empty_store_is_safe(store):
    bundle = ScenarioBundle.from_store(store)
    assert bundle.available == set()
    assert bundle.list_scenarios() == []
    assert bundle.get_scenario(1) is None
    assert bundle.has_uncertainty() is False
    assert bundle.baseline_column() is None


def test_bundle_loads_base_results(store):
    df = _scenario_frame(n_scenarios=2)
    store.write_table(SCENARIO_STAGE, "scenario_results", df)

    bundle = ScenarioBundle.from_store(store)
    assert "results" in bundle.available
    assert bundle.results_artifact_id == "scenario_results"
    assert bundle.list_scenarios() == [1, 2]
    sc = bundle.get_scenario(2)
    assert sc is not None
    assert sc.pred_column == "pred_Scenario2"
    assert sc.delta_column == "delta_Scenario2"
    assert bundle.baseline_column() == "pred_baseline"


def test_bundle_get_scenario_missing_returns_none(store):
    df = _scenario_frame(n_scenarios=1)
    store.write_table(SCENARIO_STAGE, "scenario_results", df)
    bundle = ScenarioBundle.from_store(store)
    assert bundle.get_scenario(999) is None


def test_bundle_uncertainty_columns_detected(store):
    df = _scenario_frame(n_scenarios=1)
    df["pred_std"] = 0.5
    store.write_table(SCENARIO_STAGE, "scenario_results", df)
    bundle = ScenarioBundle.from_store(store)
    assert bundle.has_uncertainty() is True


# ---------------------------------------------------------------------------
# Mode-variant precedence
# ---------------------------------------------------------------------------


def test_hybrid_wins_over_all_other_variants(store):
    base = _scenario_frame(n_scenarios=1)
    hybrid = _scenario_frame(n_scenarios=3)  # distinguishable shape
    store.write_table(SCENARIO_STAGE, "scenario_results", base)
    store.write_table(SCENARIO_STAGE, "scenario_results_dag", base)
    store.write_table(SCENARIO_STAGE, "scenario_results_reprediction", base)
    store.write_table(SCENARIO_STAGE, "scenario_results_hybrid", hybrid)

    bundle = ScenarioBundle.from_store(store)
    assert bundle.results_artifact_id == "scenario_results_hybrid"
    assert bundle.list_scenarios() == [1, 2, 3]


def test_reprediction_wins_when_no_hybrid(store):
    df = _scenario_frame()
    store.write_table(SCENARIO_STAGE, "scenario_results", df)
    store.write_table(SCENARIO_STAGE, "scenario_results_dag", df)
    store.write_table(SCENARIO_STAGE, "scenario_results_reprediction", df)
    bundle = ScenarioBundle.from_store(store)
    assert bundle.results_artifact_id == "scenario_results_reprediction"


def test_dag_wins_over_base(store):
    df = _scenario_frame()
    store.write_table(SCENARIO_STAGE, "scenario_results", df)
    store.write_table(SCENARIO_STAGE, "scenario_results_dag", df)
    bundle = ScenarioBundle.from_store(store)
    assert bundle.results_artifact_id == "scenario_results_dag"


# ---------------------------------------------------------------------------
# Regression: no consumer reads the legacy on-disk geopackage
# ---------------------------------------------------------------------------


def test_bundle_module_does_not_reference_gpkg():
    """Source-level guard: the bundle reader must not open .gpkg files."""
    src = inspect.getsource(__import__("sparc.scenario.bundle", fromlist=["*"]))
    # Strip docstrings so the contract docs (which mention .gpkg deliberately)
    # don't trigger this guard.  Look for real call-site patterns instead.
    forbidden = ('read_file(', '.gpkg")', ".gpkg')", "to_file(")
    for pattern in forbidden:
        assert pattern not in src, f"bundle.py must not contain {pattern!r}"


def test_bundle_does_not_open_disk_files(tmp_path, monkeypatch, store):
    """If a stale .gpkg sits on disk, the bundle ignores it."""
    df = _scenario_frame(n_scenarios=1)
    store.write_table(SCENARIO_STAGE, "scenario_results", df)

    stray = store.registry.output_dir / "scenario_results.gpkg"
    stray.write_bytes(b"not a real gpkg")

    opened: list[Path] = []
    real_open = Path.open

    def tracking_open(self, *a, **kw):  # type: ignore[override]
        opened.append(self)
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", tracking_open)
    bundle = ScenarioBundle.from_store(store)

    assert bundle.results_artifact_id == "scenario_results"
    assert all(".gpkg" not in str(p) for p in opened)
