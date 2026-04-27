"""Tests for the ArtifactStore db-resident artifact API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import (
    ArtifactStore,
    ArtifactStoreError,
    BLOB_INLINE_THRESHOLD_BYTES,
    GEOMETRY_COLUMN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    return ArtifactStore(reg)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_write_read_table_roundtrip(store):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [0.1, 0.2, 0.3]})
    store.write_table(stage="2", artifact_id="my_table", df=df)

    out = store.read_table("2", "my_table")
    pd.testing.assert_frame_equal(out, df, check_dtype=False)


def test_table_registers_artifact_metadata(store):
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    store.write_table(stage="3", artifact_id="metrics", df=df,
                      producer="test", consumers=["server:/results/metrics"])

    entry = store.registry.lookup("3", "metrics")
    assert entry is not None
    assert entry.storage_kind == "table"
    assert entry.format == "table"
    assert entry.data_table == "data__3__metrics"
    assert entry.row_count == 5
    assert entry.producer == "test"
    assert "server:/results/metrics" in entry.consumers
    assert set(entry.metadata["columns"]) == {"x", "y"}


def test_table_overwrite_replaces_data(store):
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [9, 8, 7]})
    store.write_table("0", "t", df1)
    store.write_table("0", "t", df2)

    out = store.read_table("0", "t")
    assert out["a"].tolist() == [9, 8, 7]


def test_read_table_unknown_artifact_raises(store):
    with pytest.raises(ArtifactStoreError):
        store.read_table("0", "nope")


def test_read_table_wrong_kind_raises(store):
    store.write_struct("0", "s", {"a": 1})
    with pytest.raises(ArtifactStoreError):
        store.read_table("0", "s")


def test_table_with_geometry_roundtrip(store):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = geopandas.GeoDataFrame(
        {"id": [1, 2], "val": [10.0, 20.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    store.write_table("4", "scenarios", gdf)

    out = store.read_table("4", "scenarios")
    assert isinstance(out, geopandas.GeoDataFrame)
    assert out.crs is not None and out.crs.to_string() == "EPSG:4326"
    assert out.geometry.iloc[0].equals(Point(0, 0))
    assert out.geometry.iloc[1].equals(Point(1, 1))
    assert out["id"].tolist() == [1, 2]


def test_table_with_geometry_no_geopandas_returns_wkb(store, monkeypatch):
    """Without geopandas, geometry column is exposed as raw WKB bytes."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = geopandas.GeoDataFrame(
        {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    store.write_table("4", "g", gdf)

    # Simulate missing geopandas at read time.
    import sys

    monkeypatch.setitem(sys.modules, "geopandas", None)
    out = store.read_table("4", "g")
    assert GEOMETRY_COLUMN in out.columns
    assert isinstance(out[GEOMETRY_COLUMN].iloc[0], (bytes, bytearray))


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


def test_write_read_struct_roundtrip(store):
    payload = {"score": 0.92, "config": {"k": 5, "alpha": 0.1}, "names": ["a", "b"]}
    store.write_struct("1", "summary", payload)

    out = store.read_struct("1", "summary")
    assert out == payload


def test_struct_overwrite_replaces(store):
    store.write_struct("1", "s", {"v": 1})
    store.write_struct("1", "s", {"v": 2})
    assert store.read_struct("1", "s") == {"v": 2}


def test_struct_registers_size(store):
    store.write_struct("0", "ds", {"hello": "world"})
    entry = store.registry.lookup("0", "ds")
    assert entry is not None
    assert entry.storage_kind == "struct"
    assert entry.format == "struct"
    assert entry.size_bytes > 0


def test_read_struct_unknown_raises(store):
    with pytest.raises(ArtifactStoreError):
        store.read_struct("0", "nope")


# ---------------------------------------------------------------------------
# Blobs
# ---------------------------------------------------------------------------


def test_inline_blob_pickle_roundtrip(store):
    obj = {"weights": np.arange(10).tolist(), "label": "hello"}
    store.write_blob("2", "scaler", obj, serializer="pickle")

    entry = store.registry.lookup("2", "scaler")
    assert entry is not None
    assert entry.storage_kind == "blob_inline"
    assert entry.format == "blob"
    assert entry.blob_id is not None

    out = store.read_blob("2", "scaler")
    assert out == obj


def test_inline_blob_json_roundtrip(store):
    obj = {"a": [1, 2, 3], "b": "x"}
    store.write_blob("0", "cfg", obj, serializer="json")
    assert store.read_blob("0", "cfg") == obj


def test_inline_blob_raw_roundtrip(store):
    payload = b"\x00\x01\x02hello\xff"
    store.write_blob("0", "raw", payload, serializer="raw")
    assert store.read_blob("0", "raw") == payload


def test_external_blob_above_threshold(store):
    # Force external storage with a tiny threshold.
    big = np.zeros(1024, dtype=np.uint8).tobytes()
    store.write_blob("2", "big", big, serializer="raw", threshold_bytes=512)

    entry = store.registry.lookup("2", "big")
    assert entry is not None
    assert entry.storage_kind == "blob_external"
    assert entry.blob_sha256 is not None
    blob_path = store.blobs_dir / entry.blob_sha256[:2] / entry.blob_sha256
    assert blob_path.exists()
    assert blob_path.read_bytes() == big

    # Round-trip read.
    assert store.read_blob("2", "big") == big


def test_external_blob_dedupes_by_sha(store):
    """Writing the same bytes under two artifact ids reuses the CAS file."""
    payload = b"x" * 2048
    store.write_blob("2", "a", payload, serializer="raw", threshold_bytes=512)
    store.write_blob("2", "b", payload, serializer="raw", threshold_bytes=512)

    a = store.registry.lookup("2", "a")
    b = store.registry.lookup("2", "b")
    assert a.blob_sha256 == b.blob_sha256
    # Only one file exists in the CAS dir for this sha.
    matches = list(store.blobs_dir.rglob(a.blob_sha256))
    assert len(matches) == 1


def test_blob_unknown_raises(store):
    with pytest.raises(ArtifactStoreError):
        store.read_blob("0", "nope")


def test_blob_wrong_kind_raises(store):
    store.write_struct("0", "s", {"v": 1})
    with pytest.raises(ArtifactStoreError):
        store.read_blob("0", "s")


def test_unknown_serializer_raises(store):
    with pytest.raises(ArtifactStoreError):
        store.write_blob("0", "x", object(), serializer="bogus")


def test_threshold_default_is_10mb():
    assert BLOB_INLINE_THRESHOLD_BYTES == 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


def test_prune_unreferenced_blobs(store):
    payload = b"a" * 2048
    store.write_blob("2", "keep", payload, serializer="raw", threshold_bytes=512)

    # Manually drop in an orphan file.
    orphan_dir = store.blobs_dir / "ff"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    orphan = orphan_dir / ("f" * 64)
    orphan.write_bytes(b"orphan")

    removed = store.prune_unreferenced_blobs()
    assert removed == 1
    assert not orphan.exists()
    # Referenced blob untouched.
    keep_entry = store.registry.lookup("2", "keep")
    keep_path = store.blobs_dir / keep_entry.blob_sha256[:2] / keep_entry.blob_sha256
    assert keep_path.exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_register_legacy_path_still_works(tmp_path):
    """Back-compat: register_artifact with path= still produces legacy entries."""
    reg = RunRegistry(tmp_path)
    f = tmp_path / "out.csv"
    f.write_text("a,b\n1,2\n")
    entry = reg.register_artifact(
        stage="2", artifact_id="legacy", path=f, format="csv",
    )
    assert entry.storage_kind == "legacy_path"
    assert entry.path == "out.csv"
    assert entry.size_bytes > 0
