"""End-to-end tests for the centralized data layer (PR1 of the database refactor).

These tests exercise the new ``ResultStore.save_dataset`` /
``ResultStore.load_dataset`` API together with ``RunRegistry``'s DuckDB query
layer and the auto-written ``provenance.json`` sidecars.

Scope of PR1: foundation only. Stage code is not yet converted; we just
validate the contract works round-trip in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path):
    """A ResultStore wired to a fresh tmp output directory."""
    from sparc.run.pipeline_paths import PipelinePaths
    from sparc.run.result_store import ResultStore

    cfg = {
        "paths": {"project_root": str(tmp_path)},
        "output": {"base_dir": str(tmp_path / "output")},
    }
    paths = PipelinePaths.from_config(cfg)
    return ResultStore(paths)


@pytest.fixture
def sample_config() -> dict:
    """Minimal config dict suitable for provenance hashing."""
    return {
        "paths": {"input_csv": "tests/_nonexistent.csv"},
        "model": {"seed": 42, "kind": "stub"},
    }


# ---------------------------------------------------------------------------
# Dataset / DatasetSchema models
# ---------------------------------------------------------------------------


def test_dataset_default_format_dispatch():
    from sparc.registry.manifest import Dataset

    assert Dataset.default_format_for("table") == "parquet"
    assert Dataset.default_format_for("geotable") == "parquet"
    assert Dataset.default_format_for("array") == "npy"
    assert Dataset.default_format_for("checkpoint") == "pt"
    assert Dataset.default_format_for("metadata") == "json"


def test_dataset_qualified_name_numeric_stage():
    from sparc.registry.manifest import Dataset, DatasetSchema

    ds = Dataset(
        stage="2",
        name="cv_predictions",
        kind="table",
        data_schema=DatasetSchema(kind="table"),
    )
    assert ds.qualified_name == "stage2.cv_predictions"


def test_dataset_qualified_name_named_stage():
    from sparc.registry.manifest import Dataset, DatasetSchema

    ds = Dataset(
        stage="final",
        name="summary",
        kind="metadata",
        data_schema=DatasetSchema(kind="metadata"),
        storage_format="json",
    )
    assert ds.qualified_name == "final.summary"


def test_dataset_resolved_artifact_id_defaults_to_name():
    from sparc.registry.manifest import Dataset, DatasetSchema

    ds = Dataset(
        stage="1",
        name="features",
        data_schema=DatasetSchema(kind="metadata"),
        storage_format="json",
    )
    assert ds.resolved_artifact_id() == "features"

    ds2 = Dataset(
        stage="1",
        name="features",
        artifact_id="custom_id",
        data_schema=DatasetSchema(kind="metadata"),
        storage_format="json",
    )
    assert ds2.resolved_artifact_id() == "custom_id"


# ---------------------------------------------------------------------------
# Round-trip: tabular DataFrame
# ---------------------------------------------------------------------------


def test_save_load_table_dataset_round_trips(store, sample_config):
    from sparc.registry.manifest import ColumnSpec, Dataset, DatasetSchema

    df = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.1, 1.9, 3.2, 3.8],
        }
    )
    schema = DatasetSchema(
        kind="table",
        columns=[
            ColumnSpec(name="fold", dtype="int64"),
            ColumnSpec(name="y_true", dtype="float64"),
            ColumnSpec(name="y_pred", dtype="float64"),
        ],
        primary_key=["fold"],
    )
    ds = Dataset(
        stage="2",
        name="cv_predictions",
        kind="table",
        data_schema=schema,
        description="Cross-validation predictions per fold.",
    )

    dest = store.save_dataset(ds, df, config=sample_config)

    assert dest.exists() and dest.suffix == ".parquet"
    sidecar = dest.with_suffix(dest.suffix + ".provenance.json")
    assert sidecar.exists(), "provenance sidecar should be written when config is provided"

    loaded = store.load_dataset("2", "cv_predictions")
    pd.testing.assert_frame_equal(loaded, df)


def test_save_dataset_registers_in_run_registry(store):
    from sparc.registry.manifest import Dataset, DatasetSchema

    ds = Dataset(
        stage="1",
        name="selected_features",
        kind="metadata",
        data_schema=DatasetSchema(kind="metadata"),
        storage_format="json",
    )
    store.save_dataset(ds, {"features": ["lst", "ndvi", "elev"]})

    entry = store.registry.lookup("1", "selected_features")
    assert entry is not None
    assert entry.format == "json"
    assert entry.partial is False
    assert entry.metadata["dataset_kind"] == "metadata"
    assert entry.metadata["qualified_name"] == "stage1.selected_features"
    assert entry.metadata["dataset_schema"]["kind"] == "metadata"


def test_save_dataset_skips_provenance_when_no_config(store):
    from sparc.registry.manifest import Dataset, DatasetSchema

    ds = Dataset(
        stage="1",
        name="features",
        kind="metadata",
        data_schema=DatasetSchema(kind="metadata"),
        storage_format="json",
    )
    dest = store.save_dataset(ds, {"a": 1})
    sidecar = dest.with_suffix(dest.suffix + ".provenance.json")
    assert not sidecar.exists()


# ---------------------------------------------------------------------------
# Round-trip: numpy array dataset
# ---------------------------------------------------------------------------


def test_save_load_array_dataset(store):
    from sparc.registry.manifest import Dataset, DatasetSchema

    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    schema = DatasetSchema(kind="array", array_shape=[4, 5], array_dtype="float32")
    ds = Dataset(
        stage="2",
        name="alpha_field",
        kind="array",
        data_schema=schema,
        storage_format="npy",
    )
    store.save_dataset(ds, arr)

    loaded = store.load_dataset("2", "alpha_field")
    np.testing.assert_array_equal(loaded, arr)
    assert loaded.dtype == np.float32


# ---------------------------------------------------------------------------
# Round-trip: GeoDataFrame as GeoParquet
# ---------------------------------------------------------------------------


def test_save_load_geotable_dataset(store):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from sparc.registry.manifest import ColumnSpec, Dataset, DatasetSchema

    gdf = gpd.GeoDataFrame(
        {
            "tract_id": ["a", "b", "c"],
            "score": [0.1, 0.5, 0.9],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    schema = DatasetSchema(
        kind="geotable",
        columns=[
            ColumnSpec(name="tract_id", dtype="string"),
            ColumnSpec(name="score", dtype="float64"),
            ColumnSpec(name="geometry", dtype="geometry"),
        ],
        geometry_column="geometry",
        crs="EPSG:4326",
    )
    ds = Dataset(
        stage="4",
        name="scenario_results_baseline",
        kind="geotable",
        data_schema=schema,
    )

    store.save_dataset(ds, gdf)
    loaded = store.load_dataset("4", "scenario_results_baseline")

    assert isinstance(loaded, gpd.GeoDataFrame)
    assert str(loaded.crs).upper().endswith("4326")
    assert list(loaded.columns) == list(gdf.columns)
    assert len(loaded) == len(gdf)


# ---------------------------------------------------------------------------
# DuckDB query layer
# ---------------------------------------------------------------------------


def test_duckdb_view_exposes_parquet_dataset(store):
    pytest.importorskip("duckdb")
    from sparc.registry.manifest import Dataset, DatasetSchema

    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    ds = Dataset(
        stage="2",
        name="cv_predictions",
        kind="table",
        data_schema=DatasetSchema(kind="table"),
    )
    store.save_dataset(ds, df)

    con = store.registry.duckdb()
    rows = con.execute("SELECT count(*) AS n, sum(x) AS sx FROM stage2_cv_predictions").fetchone()
    assert rows == (4, 10)


def test_list_datasets_reports_dataset_metadata(store):
    from sparc.registry.manifest import Dataset, DatasetSchema

    store.save_dataset(
        Dataset(
            stage="1",
            name="gwen_results",
            kind="table",
            data_schema=DatasetSchema(kind="table"),
        ),
        pd.DataFrame({"feature": ["a", "b"], "score": [0.1, 0.2]}),
    )
    store.save_dataset(
        Dataset(
            stage="1",
            name="diagnostics",
            kind="metadata",
            data_schema=DatasetSchema(kind="metadata"),
            storage_format="json",
        ),
        {"r2": 0.91},
    )

    listed = store.registry.list_datasets()
    by_q = {d["qualified_name"]: d for d in listed}
    assert "stage1.gwen_results" in by_q
    assert by_q["stage1.gwen_results"]["kind"] == "table"
    assert by_q["stage1.gwen_results"]["view"] == "stage1_gwen_results"
    assert by_q["stage1.diagnostics"]["kind"] == "metadata"
    # JSON datasets are not exposed as DuckDB views.
    assert by_q["stage1.diagnostics"]["view"] is None


def test_duckdb_view_name_sanitizes_special_chars():
    from sparc.registry.run_registry import RunRegistry

    assert RunRegistry._duckdb_view_name("2", "neural_pdp::lst") == "stage2_neural_pdp__lst"
    assert RunRegistry._duckdb_view_name("final", "summary-v2") == "final_summary_v2"


# ---------------------------------------------------------------------------
# Provenance sidecar contents
# ---------------------------------------------------------------------------


def test_provenance_sidecar_includes_artifact_hash(store, sample_config):
    from sparc.registry.manifest import Dataset, DatasetSchema

    ds = Dataset(
        stage="0",
        name="dataset_profile",
        kind="metadata",
        data_schema=DatasetSchema(kind="metadata"),
        storage_format="json",
    )
    dest = store.save_dataset(ds, {"n_rows": 1234}, config=sample_config)
    sidecar = dest.with_suffix(dest.suffix + ".provenance.json")
    payload = json.loads(sidecar.read_text())

    assert payload["schema_version"] == 1
    assert payload["config_hash"], "config_hash must be populated"
    assert payload["artifact"]["sha256"], "artifact sha256 must be recorded"
    assert payload["artifact"]["path"].endswith("dataset_profile.json")
