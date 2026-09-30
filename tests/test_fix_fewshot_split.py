"""Regression tests for the JEPA few-shot spatial split (roadmap Appendix A21).

``scripts/train_multicity_jepa.py`` imports torch and reconfigures stdout at
module import, so the pure split helpers are loaded WITHOUT executing the
script: importlib locates the script and returns its source, and only the
two helper functions are compiled from the AST.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_multicity_jepa.py"
_FUNCS = ("_fewshot_block_partition", "_spatial_fewshot_split")


def _load_split_helpers() -> dict:
    spec = importlib.util.spec_from_file_location("train_multicity_jepa", SCRIPT)
    source = spec.loader.get_source("train_multicity_jepa")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _FUNCS]
    assert {n.name for n in nodes} == set(_FUNCS)
    module = ast.Module(body=nodes, type_ignores=[])
    ns = {"np": np, "Optional": Optional, "__name__": "fewshot_split_helpers"}
    exec(compile(module, str(SCRIPT), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def helpers():
    return _load_split_helpers()


def _grid_xy(n_side=80, spacing=30.0):
    xs, ys = np.meshgrid(np.arange(n_side) * spacing, np.arange(n_side) * spacing)
    return np.column_stack([xs.ravel(), ys.ravel()]) + np.array([500_000.0, 4_400_000.0])


def _setup(seed=0, n_train=40):
    xy = _grid_xy()                                   # 2.4 km × 2.4 km at 30 m
    rng = np.random.default_rng(seed)
    labeled = np.sort(rng.choice(len(xy), size=4000, replace=False))
    cand = rng.choice(labeled, size=n_train, replace=False)
    return xy, labeled, cand


def _min_test_train_dist(xy, train, test):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy[train]).query(xy[test], k=1)
    return float(d.min())


def test_random_mode_reproduces_legacy_split(helpers):
    xy, labeled, cand = _setup()
    train, test = helpers["_spatial_fewshot_split"](
        xy, cand, labeled, mode="random", buffer_m=500.0, block_m=2000.0,
        rng=np.random.default_rng(1))
    np.testing.assert_array_equal(train, cand)
    np.testing.assert_array_equal(test, np.setdiff1d(labeled, cand))


def test_buffer_mode_min_distance_exceeds_buffer(helpers):
    xy, labeled, cand = _setup(n_train=10)
    train, test = helpers["_spatial_fewshot_split"](
        xy, cand, labeled, mode="buffer", buffer_m=500.0, block_m=2000.0,
        rng=np.random.default_rng(1))
    np.testing.assert_array_equal(train, cand)
    assert len(test) > 0
    assert set(test).issubset(set(labeled) - set(cand))
    assert _min_test_train_dist(xy, train, test) > 500.0
    # everything farther than the buffer is kept
    legacy = np.setdiff1d(labeled, cand)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy[train]).query(xy[legacy], k=1)
    assert len(test) == int((d > 500.0).sum())


def test_block_mode_uses_disjoint_blocks(helpers):
    xy = _grid_xy(n_side=200)                         # 6 km × 6 km → 9 blocks of 2 km
    labeled = np.arange(len(xy))
    block_m = 2000.0
    pool, _ = helpers["_fewshot_block_partition"](
        xy, labeled, block_m, np.random.default_rng(7))
    cand = np.random.default_rng(3).choice(pool, size=50, replace=False)
    train, test = helpers["_spatial_fewshot_split"](
        xy, cand, labeled, mode="block", buffer_m=300.0, block_m=block_m,
        rng=np.random.default_rng(7))
    np.testing.assert_array_equal(np.sort(train), np.sort(cand))   # drawn from train half
    assert len(test) > 0

    def blocks(idx):
        b = np.floor(xy[idx] / block_m).astype(int)
        return {tuple(r) for r in b}

    assert blocks(train).isdisjoint(blocks(test))
    assert _min_test_train_dist(xy, train, test) > 300.0


def test_block_mode_drops_candidates_in_test_blocks(helpers):
    xy = _grid_xy(n_side=200)
    labeled = np.arange(len(xy))
    cand = np.random.default_rng(5).choice(labeled, size=200, replace=False)
    train, test = helpers["_spatial_fewshot_split"](
        xy, cand, labeled, mode="block", buffer_m=0.0, block_m=2000.0,
        rng=np.random.default_rng(11))
    pool, test_pool = helpers["_fewshot_block_partition"](
        xy, labeled, 2000.0, np.random.default_rng(11))
    assert set(train).issubset(set(pool))
    assert 0 < len(train) < len(cand)
    np.testing.assert_array_equal(test, test_pool)


def test_block_partition_is_deterministic_and_complete(helpers):
    xy = _grid_xy(n_side=200)
    labeled = np.arange(0, len(xy), 3)
    a = helpers["_fewshot_block_partition"](xy, labeled, 2000.0, np.random.default_rng(0))
    b = helpers["_fewshot_block_partition"](xy, labeled, 2000.0, np.random.default_rng(0))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), labeled)
    assert len(np.intersect1d(a[0], a[1])) == 0


def test_unknown_mode_rejected(helpers):
    xy, labeled, cand = _setup()
    with pytest.raises(ValueError):
        helpers["_spatial_fewshot_split"](xy, cand, labeled, mode="kfold")


def test_cli_flags_present_with_defaults():
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    flags = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            kw = {k.arg: k.value for k in node.keywords}
            flags[node.args[0].value] = kw
    assert ast.literal_eval(flags["--fewshot-buffer-m"]["default"]) == 500.0
    assert ast.literal_eval(flags["--fewshot-split"]["default"]) == "buffer"
    assert ast.literal_eval(flags["--fewshot-split"]["choices"]) == ["buffer", "block", "random"]
