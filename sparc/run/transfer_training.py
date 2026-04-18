"""
Transfer learning utilities for SPARC V3.

Provides cold-start, warm-start, and warm-start-with-finetune training
modes to validate trunk transfer from a source city to a target city.

Usage:
    from sparc.run.transfer_training import train_cold_start, train_warm_start

    # Train source city (Providence) — saves trunk checkpoint
    result_src = train_cold_start(config_src, data_src, output_dir_src)

    # Train target city (Boston) warm-start from Providence trunk
    result_tgt = train_warm_start(config_tgt, data_tgt, output_dir_tgt,
                                   trunk_path=output_dir_src / "shared_trunk.pt")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _load_project(project_path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load a project config and its data.

    Returns (config, y, coords, feature_matrix, feature_names).
    """
    from sparc.config.config import load_config
    from sparc.data.data_utils import load_and_preprocess_data

    project_path = Path(project_path)
    config = load_config(str(project_path))

    df = load_and_preprocess_data(config)
    target_col = config["data"]["target_column"]
    coord_cols = config["data"]["coord_columns"]
    predictor_cols = config.get("predictors", {})
    if isinstance(predictor_cols, dict):
        predictor_cols = predictor_cols.get("base_model", [])
    elif isinstance(predictor_cols, list):
        pass
    else:
        predictor_cols = []

    # Fallback: use all numeric columns except target and coords
    if not predictor_cols:
        predictor_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != target_col and c not in coord_cols
        ]

    y = df[target_col].values
    coords = df[coord_cols].values
    feature_matrix = df[predictor_cols].values
    feature_names = list(predictor_cols)

    return config, y, coords, feature_matrix, feature_names


def _build_folds(y: np.ndarray, coords: np.ndarray, config: dict) -> list:
    """Build spatial CV folds matching the training pipeline."""
    from sparc.training.v2_neural_training import build_spatial_folds

    n_folds = config.get("spatial_cv", {}).get("n_folds", 5)
    return build_spatial_folds(coords, n_folds=n_folds)


# ---- Training modes ----


def train_cold_start(
    config: dict,
    y: np.ndarray,
    coords: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    folds: list,
    save_trunk: bool = True,
) -> dict[str, Any]:
    """
    Standard training from random initialization (baseline).

    Returns dict with metrics, training_time, and optionally trunk_path.
    """
    from sparc.training.v2_neural_training import train_neural_meta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cold-start training → %s", output_dir)
    t0 = time.time()

    result = train_neural_meta(
        config=config,
        y=y,
        coords=coords,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        output_dir=output_dir,
        folds=folds,
    )

    elapsed = time.time() - t0
    result["training_time"] = elapsed

    if save_trunk:
        trunk_path = output_dir / "shared_trunk.pt"
        if "model" in result and hasattr(result["model"], "save_trunk"):
            result["model"].save_trunk(str(trunk_path))
            result["trunk_path"] = str(trunk_path)
            logger.info("Trunk saved → %s", trunk_path)

    return result


def train_warm_start(
    config: dict,
    y: np.ndarray,
    coords: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    trunk_path: str | Path,
    folds: list,
    freeze_trunk: bool = True,
    save_trunk: bool = False,
) -> dict[str, Any]:
    """
    Warm-start from a pre-trained trunk checkpoint.

    The trunk is loaded and optionally frozen so only the CityHead trains.
    """
    from sparc.training.v2_neural_training import train_neural_meta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Warm-start training (trunk=%s, freeze=%s) → %s",
                trunk_path, freeze_trunk, output_dir)

    # Inject transfer config for the training pipeline to pick up
    config = dict(config)
    config["_transfer"] = {
        "trunk_path": str(trunk_path),
        "freeze_trunk": freeze_trunk,
    }

    t0 = time.time()

    result = train_neural_meta(
        config=config,
        y=y,
        coords=coords,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        output_dir=output_dir,
        folds=folds,
    )

    elapsed = time.time() - t0
    result["training_time"] = elapsed

    if save_trunk:
        out_trunk = output_dir / "shared_trunk.pt"
        if "model" in result and hasattr(result["model"], "save_trunk"):
            result["model"].save_trunk(str(out_trunk))
            result["trunk_path"] = str(out_trunk)

    return result


def train_warm_start_finetune(
    config: dict,
    y: np.ndarray,
    coords: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    trunk_path: str | Path,
    folds: list,
    unfreeze_after: int = 20,
    trunk_lr_factor: float = 0.1,
    save_trunk: bool = True,
) -> dict[str, Any]:
    """
    Warm-start with fine-tuning: freeze trunk for N epochs, then unfreeze
    at a reduced learning rate.
    """
    from sparc.training.v2_neural_training import train_neural_meta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Warm-start+finetune (unfreeze_after=%d, lr_factor=%.3f) → %s",
                unfreeze_after, trunk_lr_factor, output_dir)

    config = dict(config)
    config["_transfer"] = {
        "trunk_path": str(trunk_path),
        "freeze_trunk": True,
        "unfreeze_after_epoch": unfreeze_after,
        "trunk_lr_factor": trunk_lr_factor,
    }

    t0 = time.time()

    result = train_neural_meta(
        config=config,
        y=y,
        coords=coords,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        output_dir=output_dir,
        folds=folds,
    )

    elapsed = time.time() - t0
    result["training_time"] = elapsed

    if save_trunk:
        out_trunk = output_dir / "shared_trunk.pt"
        if "model" in result and hasattr(result["model"], "save_trunk"):
            result["model"].save_trunk(str(out_trunk))
            result["trunk_path"] = str(out_trunk)

    return result


def compare_transfer_results(
    cold_result: dict[str, Any],
    warm_result: dict[str, Any],
    finetune_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare cold-start vs warm-start (and optionally fine-tune) metrics.

    Returns a summary dict with per-mode metrics and improvement stats.
    """
    def _extract(result: dict) -> dict:
        m = result.get("metrics", {})
        return {
            "r2": m.get("r2"),
            "rmse": m.get("rmse"),
            "training_time": result.get("training_time"),
        }

    comparison = {
        "cold_start": _extract(cold_result),
        "warm_start": _extract(warm_result),
    }

    # Compute improvement
    cold_r2 = comparison["cold_start"].get("r2")
    warm_r2 = comparison["warm_start"].get("r2")
    if cold_r2 is not None and warm_r2 is not None:
        comparison["r2_improvement"] = warm_r2 - cold_r2

    cold_time = comparison["cold_start"].get("training_time")
    warm_time = comparison["warm_start"].get("training_time")
    if cold_time and warm_time and warm_time > 0:
        comparison["speedup_factor"] = cold_time / warm_time

    if finetune_result is not None:
        comparison["warm_start_finetune"] = _extract(finetune_result)

    return comparison
