"""
Transfer learning validation for SPARC V3.

Orchestrates a full A/B comparison between cold-start and warm-start
training across two cities (source → target).

Usage:
    sparc transfer --source-project providence.yml --target-project boston.yml

Produces:
    - Cold-start training on target city
    - Warm-start training on target city (frozen trunk from source)
    - Warm-start with fine-tuning (optional)
    - Comparison metrics JSON
    - Convergence comparison summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sparc.run.transfer_training import (
    _build_folds,
    _load_project,
    compare_transfer_results,
    train_cold_start,
    train_warm_start,
    train_warm_start_finetune,
)

logger = logging.getLogger(__name__)


def run_transfer_validation(
    source_project: str | Path,
    target_project: str | Path,
    output_dir: str | Path,
    run_finetune: bool = True,
    unfreeze_after: int = 20,
    trunk_lr_factor: float = 0.1,
) -> dict[str, Any]:
    """
    Full transfer learning validation pipeline.

    Steps:
        1. Train source city (Providence) → save trunk
        2. Train target city cold-start (random init)
        3. Train target city warm-start (frozen trunk from source)
        4. Optionally train target city warm-start with fine-tuning
        5. Compare metrics and save report

    Parameters
    ----------
    source_project : path to source city project.yml
    target_project : path to target city project.yml
    output_dir : base directory for all transfer outputs
    run_finetune : also run warm-start with finetune mode
    unfreeze_after : epoch to unfreeze trunk in finetune mode
    trunk_lr_factor : LR multiplier for trunk in finetune mode

    Returns
    -------
    dict with comparison metrics and per-mode results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load and train source city ----
    logger.info("=" * 60)
    logger.info("Step 1: Training source city")
    logger.info("=" * 60)

    src_config, src_y, src_coords, src_features, src_names = _load_project(source_project)
    src_folds = _build_folds(src_y, src_coords, src_config)
    src_dir = output_dir / "source"

    src_result = train_cold_start(
        config=src_config, y=src_y, coords=src_coords,
        feature_matrix=src_features, feature_names=src_names,
        output_dir=src_dir, folds=src_folds, save_trunk=True,
    )

    trunk_path = src_dir / "shared_trunk.pt"
    if not trunk_path.exists():
        raise FileNotFoundError(
            f"Source city training did not produce trunk checkpoint at {trunk_path}"
        )

    logger.info("Source city R²: %.4f, training time: %.1fs",
                src_result.get("metrics", {}).get("r2", float("nan")),
                src_result.get("training_time", 0))

    # ---- Step 2: Load target city data ----
    logger.info("=" * 60)
    logger.info("Step 2: Loading target city data")
    logger.info("=" * 60)

    tgt_config, tgt_y, tgt_coords, tgt_features, tgt_names = _load_project(target_project)
    tgt_folds = _build_folds(tgt_y, tgt_coords, tgt_config)

    # ---- Step 3: Cold-start on target ----
    logger.info("=" * 60)
    logger.info("Step 3: Cold-start training on target city")
    logger.info("=" * 60)

    cold_dir = output_dir / "target_cold"
    cold_result = train_cold_start(
        config=tgt_config.copy(), y=tgt_y, coords=tgt_coords,
        feature_matrix=tgt_features, feature_names=tgt_names,
        output_dir=cold_dir, folds=tgt_folds, save_trunk=False,
    )
    logger.info("Cold-start R²: %.4f", cold_result.get("metrics", {}).get("r2", float("nan")))

    # ---- Step 4: Warm-start on target ----
    logger.info("=" * 60)
    logger.info("Step 4: Warm-start training on target city (frozen trunk)")
    logger.info("=" * 60)

    warm_dir = output_dir / "target_warm"
    warm_result = train_warm_start(
        config=tgt_config.copy(), y=tgt_y, coords=tgt_coords,
        feature_matrix=tgt_features, feature_names=tgt_names,
        output_dir=warm_dir, trunk_path=trunk_path, folds=tgt_folds,
        freeze_trunk=True, save_trunk=True,
    )
    logger.info("Warm-start R²: %.4f", warm_result.get("metrics", {}).get("r2", float("nan")))

    # ---- Step 5: Fine-tune on target (optional) ----
    finetune_result = None
    if run_finetune:
        logger.info("=" * 60)
        logger.info("Step 5: Warm-start + fine-tune on target city")
        logger.info("=" * 60)

        ft_dir = output_dir / "target_finetune"
        finetune_result = train_warm_start_finetune(
            config=tgt_config.copy(), y=tgt_y, coords=tgt_coords,
            feature_matrix=tgt_features, feature_names=tgt_names,
            output_dir=ft_dir, trunk_path=trunk_path, folds=tgt_folds,
            unfreeze_after=unfreeze_after, trunk_lr_factor=trunk_lr_factor,
            save_trunk=True,
        )
        logger.info("Fine-tune R²: %.4f", finetune_result.get("metrics", {}).get("r2", float("nan")))

    # ---- Step 6: Compare and report ----
    comparison = compare_transfer_results(cold_result, warm_result, finetune_result)
    comparison["source_project"] = str(source_project)
    comparison["target_project"] = str(target_project)
    comparison["source_metrics"] = {
        "r2": src_result.get("metrics", {}).get("r2"),
        "rmse": src_result.get("metrics", {}).get("rmse"),
    }

    report_path = output_dir / "transfer_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Transfer comparison saved to %s", report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING RESULTS")
    logger.info("=" * 60)
    logger.info("Cold-start R²:   %.4f", comparison["cold_start"].get("r2", float("nan")))
    logger.info("Warm-start R²:   %.4f", comparison["warm_start"].get("r2", float("nan")))
    if finetune_result:
        logger.info("Fine-tune R²:    %.4f", comparison.get("warm_start_finetune", {}).get("r2", float("nan")))
    r2_imp = comparison.get("r2_improvement")
    if r2_imp is not None:
        logger.info("R² improvement:  %+.4f", r2_imp)
    speedup = comparison.get("speedup_factor")
    if speedup is not None:
        logger.info("Speedup factor:  %.1f×", speedup)

    return comparison
