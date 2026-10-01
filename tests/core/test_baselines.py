"""Reference baselines and the paired block-clustered comparison."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.core.baselines import baseline_oof, compare_baselines, paired_comparison
from sparc.core.cv import make_spatial_folds


def _field(n=40, seed=0):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(n) * 30.0, np.arange(n) * 30.0)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    cov = rng.normal(size=(coords.shape[0], 2))
    y = 1.5 * cov[:, 0] - 0.5 * cov[:, 1] + np.sin(coords[:, 0] / 300.0) + 0.1 * rng.normal(size=coords.shape[0])
    return cov, coords, y


def test_paired_comparison_is_zero_for_identical_predictions_and_signed():
    y = np.arange(100.0)
    b = np.repeat(np.arange(10), 10)
    p = y + np.tile([1.0, -1.0], 50)
    same = paired_comparison(y, p, p.copy(), b)
    assert same["delta_mse"] == 0.0 and same["delta_mse_se"] == 0.0 and not same["stack_better"]
    worse = paired_comparison(y, p, y + 3.0 + 0.1 * np.sin(y), b)
    assert worse["delta_mse"] > 0 and worse["stack_better"] and worse["frac_blocks_stack_better"] == 1.0
    assert worse["delta_rmse"] == pytest.approx(worse["rmse"] - worse["stack_rmse"])


def test_baselines_predict_every_test_point_and_covariates_matter():
    cov, coords, y = _field()
    folds = make_spatial_folds(coords, n_folds=3, block_m=300.0, buffer_m=100.0, seed=1)
    oof = baseline_oof(cov, coords, y, folds, models=("regression_kriging", "hgb", "idw"), seed=0)
    tested = np.zeros(y.size, bool)
    for te in folds.test_masks:
        tested |= te
    for m, p in oof.items():
        assert np.isfinite(p[tested]).all(), m
    rmse = {m: float(np.sqrt(np.nanmean((p - y) ** 2))) for m, p in oof.items()}
    assert rmse["regression_kriging"] < rmse["idw"]           # covariates carry most of the signal
    # a near-oracle "stack" beats every baseline; the verdict says so
    res = compare_baselines(cov, coords, y, y + 0.05 * np.random.default_rng(3).normal(size=y.size), folds,
                            models=("hgb", "idw"))
    assert res["verdict"].startswith("stack better than every baseline")
    # focal ablation is skipped without the focal matrix
    assert "hgb_focal" not in compare_baselines(cov, coords, y, y.copy(), folds, models=("idw", "hgb_focal"))["rows"]
