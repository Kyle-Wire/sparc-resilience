"""Roundtrip tests for ``ArtifactStore.export`` and ``export_stage``.

Validates that artifacts persisted to ``artifacts.db`` can be exported back
to disk in their original or alternate formats and re-read correctly.  These
tests cover the user-facing "Download" path of the db-only architecture.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore


@pytest.fixture()
def store(tmp_path: Path) -> ArtifactStore:
    reg = RunRegistry(tmp_path, autoload=False)
    return ArtifactStore(reg)


def test_export_table_csv(store: ArtifactStore, tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    store.write_table("0", "tbl", df, producer="test")

    out = store.export("0", "tbl", fmt="csv", dest=tmp_path / "tbl.csv")
    assert out.exists()
    rt = pd.read_csv(out)
    pd.testing.assert_frame_equal(rt, df)


def test_export_table_parquet(store: ArtifactStore, tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [10, 20]})
    store.write_table("1", "tbl", df, producer="test")

    out = store.export("1", "tbl", fmt="parquet", dest=tmp_path / "tbl.parquet")
    assert out.exists()
    rt = pd.read_parquet(out)
    pd.testing.assert_frame_equal(rt, df)


def test_export_struct_json(store: ArtifactStore, tmp_path: Path) -> None:
    payload = {"alpha": 0.1, "betas": [1, 2, 3], "label": "hello"}
    store.write_struct("2", "cfg", payload, producer="test")

    out = store.export("2", "cfg", fmt="json", dest=tmp_path / "cfg.json")
    assert out.exists()
    with open(out) as f:
        rt = json.load(f)
    assert rt == payload


def test_export_blob_joblib(store: ArtifactStore, tmp_path: Path) -> None:
    payload = {"weights": np.arange(10), "name": "model"}
    store.write_blob("2", "model", payload, serializer="joblib", producer="test")

    out = store.export("2", "model", fmt="joblib", dest=tmp_path / "model.joblib")
    assert out.exists()
    import joblib
    rt = joblib.load(out)
    assert rt["name"] == "model"
    assert np.array_equal(rt["weights"], payload["weights"])


def test_export_default_fmt_chosen_by_kind(store: ArtifactStore, tmp_path: Path) -> None:
    store.write_struct("0", "s", {"x": 1}, producer="test")
    out = store.export("0", "s", dest=tmp_path / "s.json")
    with open(out) as f:
        assert json.load(f) == {"x": 1}


def test_export_unknown_artifact_raises(store: ArtifactStore, tmp_path: Path) -> None:
    from sparc.registry.store import ArtifactStoreError

    with pytest.raises(ArtifactStoreError, match="Unknown artifact"):
        store.export("0", "missing", fmt="json", dest=tmp_path / "x.json")


def test_export_stage_writes_all_artifacts(store: ArtifactStore, tmp_path: Path) -> None:
    df = pd.DataFrame({"v": [1, 2]})
    store.write_table("3", "tbl", df, producer="test")
    store.write_struct("3", "meta", {"k": "v"}, producer="test")
    store.write_blob("3", "blob", {"a": 1}, serializer="joblib", producer="test")
    # Add an artifact for a different stage; it must NOT appear in the export.
    store.write_struct("4", "other", {"x": 1}, producer="test")

    out_dir = tmp_path / "exported"
    written = store.export_stage("3", out_dir)
    assert isinstance(written, list)
    names = sorted(p.name for p in written)
    assert names == ["blob.joblib", "meta.json", "tbl.csv"]
    assert all(p.exists() for p in written)


def test_export_stage_as_zip(store: ArtifactStore, tmp_path: Path) -> None:
    store.write_struct("0", "a", {"x": 1}, producer="test")
    store.write_struct("0", "b", {"y": 2}, producer="test")

    out_dir = tmp_path / "exported"
    zip_path = store.export_stage("0", out_dir, as_zip=True)
    assert isinstance(zip_path, Path)
    assert zip_path.suffix == ".zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        assert names == ["a.json", "b.json"]
