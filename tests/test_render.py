"""Tests for the on-demand render layer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore
from sparc.report.render import (
    RenderError,
    render_csv,
    render_geojson,
    render_json,
    render_native,
)


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    yield ArtifactStore(reg)
    set_active_registry(None)


def test_render_csv_from_table(store):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    store.write_table("0", "t", df)

    out = render_csv("0", "t")
    text = out.decode("utf-8")
    assert "a,b" in text
    assert "1,3" in text


def test_render_json_from_struct(store):
    payload = {"score": 0.9, "names": ["a", "b"]}
    store.write_struct("0", "s", payload)

    out = render_json("0", "s")
    assert json.loads(out) == payload


def test_render_json_from_table(store):
    df = pd.DataFrame({"a": [1, 2]})
    store.write_table("0", "t", df)

    out = render_json("0", "t")
    parsed = json.loads(out)
    assert parsed == [{"a": 1}, {"a": 2}]


def test_render_csv_from_struct_dict_of_lists(store):
    store.write_struct("0", "s", {"x": [1, 2, 3], "y": [10, 20, 30]})
    out = render_csv("0", "s").decode("utf-8")
    assert "x,y" in out
    assert "1,10" in out


def test_render_native_table_returns_csv(store):
    df = pd.DataFrame({"a": [1]})
    store.write_table("0", "t", df)
    data, ext = render_native("0", "t")
    assert ext == "csv"
    assert b"a" in data


def test_render_native_struct_returns_json(store):
    store.write_struct("0", "s", {"k": "v"})
    data, ext = render_native("0", "s")
    assert ext == "json"
    assert json.loads(data) == {"k": "v"}


def test_render_native_blob_returns_raw(store):
    store.write_blob("0", "b", b"hello", serializer="raw")
    data, ext = render_native("0", "b")
    assert data == b"hello"
    assert ext == "bin"


def test_render_geojson_from_geodataframe(store):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = geopandas.GeoDataFrame(
        {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    store.write_table("4", "g", gdf)

    out = render_geojson("4", "g").decode("utf-8")
    parsed = json.loads(out)
    assert parsed["type"] == "FeatureCollection"
    assert parsed["features"][0]["geometry"]["type"] == "Point"


def test_render_native_geometry_table_returns_geojson(store):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = geopandas.GeoDataFrame(
        {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    store.write_table("4", "g", gdf)

    data, ext = render_native("4", "g")
    assert ext == "geojson"
    assert json.loads(data)["type"] == "FeatureCollection"


def test_render_unknown_artifact_raises(store):
    with pytest.raises(RenderError):
        render_csv("0", "missing")


def test_render_without_active_registry(tmp_path):
    set_active_registry(None)
    with pytest.raises(RenderError):
        render_csv("0", "anything")
