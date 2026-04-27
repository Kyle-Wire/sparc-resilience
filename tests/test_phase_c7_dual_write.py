"""Phase C7 — ResultStore + versioning ArtifactStore dual-write tests.

Verifies:
- ResultStore.save_dataframe / save_geodataframe / save_json / save_numpy /
  save_pickle dual-write to the active ArtifactStore.
- ResultStore no longer exposes ``save_figure`` and refuses to write PNGs.
- data.versioning.save_versioned mirrors the changelog into artifacts.db
  and ``list_versions`` reads from the store first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore


@pytest.fixture()
def store_env(tmp_path):
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    store = ArtifactStore(reg)
    yield reg, store
    set_active_registry(None)


def _make_result_store(tmp_path, reg):
    from sparc.run.pipeline_paths import PipelinePaths
    from sparc.run.result_store import ResultStore

    paths = PipelinePaths.from_config({
        "output": {"base_dir": str(tmp_path)},
    })
    return ResultStore(paths, registry=reg)


def test_result_store_no_save_figure():
    from sparc.run.result_store import ResultStore
    assert not hasattr(ResultStore, "save_figure")


def test_result_store_save_dataframe_dual_writes(tmp_path, store_env):
    reg, store = store_env
    rs = _make_result_store(tmp_path, reg)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rs.save_dataframe(0, "demo_table.csv", df, fmt="csv")
    assert store.has("0", "demo_table")
    out = store.read_any("0", "demo_table")
    assert list(out["a"]) == [1, 2, 3]


def test_result_store_save_json_dual_writes(tmp_path, store_env):
    reg, store = store_env
    rs = _make_result_store(tmp_path, reg)
    payload = {"alpha": 1, "names": ["a", "b"]}
    rs.save_json(1, "demo.json", payload)
    assert store.has("1", "demo")
    out = store.read_any("1", "demo")
    assert out["alpha"] == 1


def test_result_store_save_numpy_dual_writes(tmp_path, store_env):
    reg, store = store_env
    rs = _make_result_store(tmp_path, reg)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    rs.save_numpy(2, "matrix.npy", arr)
    assert store.has("2", "matrix")
    out = store.read_any("2", "matrix")
    np.testing.assert_array_equal(out, arr)


def test_result_store_save_pickle_dual_writes(tmp_path, store_env):
    reg, store = store_env
    rs = _make_result_store(tmp_path, reg)
    obj = {"weights": [0.1, 0.2, 0.7], "tag": "v1"}
    rs.save_pickle(2, "obj.pkl", obj)
    assert store.has("2", "obj")
    out = store.read_any("2", "obj")
    assert out["tag"] == "v1"


def test_result_store_save_geodataframe_dual_writes(tmp_path, store_env):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    reg, store = store_env
    rs = _make_result_store(tmp_path, reg)
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"], "value": [1.0, 2.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    rs.save_geodataframe(4, "geo_demo.gpkg", gdf)
    assert store.has("4", "geo_demo")
    out = store.read_any("4", "geo_demo")
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 2


def test_versioning_writes_changelog_to_store(tmp_path, store_env):
    from sparc.data.versioning import save_versioned, list_versions
    reg, store = store_env
    df = pd.DataFrame({"x": [1, 2, 3]})
    entry = save_versioned(df, tmp_path, description="initial")
    assert entry["version"] == 1
    assert store.has("data", "data_changelog")
    cl = store.read_any("data", "data_changelog")
    assert isinstance(cl, list) and len(cl) == 1
    # list_versions should pick it up from the store
    versions = list_versions(tmp_path)
    assert len(versions) == 1
    assert versions[0]["version"] == 1
