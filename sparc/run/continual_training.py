"""
Continual training orchestrator for SPARC V3.

Trains the shared trunk sequentially across multiple cities, using:
  - Elastic Weight Consolidation (EWC) to preserve past knowledge
  - Experience replay on coreset data from previous cities
  - Welford online scaling for consistent feature normalisation
  - City registry for artifact management

Usage:
    sparc continual --cities providence.yml,boston.yml --registry ./sparc_registry/
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from sparc.data.welford import WelfordScaler
from sparc.registry.city_registry import CityRegistry
from sparc.run.transfer_training import _build_folds, _load_project
from sparc.run.v2_neural_training import train_neural_meta
from sparc.training.ewc import compute_fisher_matrix, extract_trunk_params
from sparc.training.replay import CoresetSelector

logger = logging.getLogger(__name__)

# Trunk parameter name prefixes (must match neural_meta._TRUNK_KEYS)
_TRUNK_KEYS = {"physics_enc", "alpha_emb", "trunk_fusion", "time_embed"}


def train_continual(
    city_configs: list[str | Path],
    registry_path: str | Path,
    output_dir: str | Path,
    ewc_lambda: float = 1.0,
    replay_lambda: float = 0.5,
    coreset_size: int = 400,
) -> dict[str, Any]:
    """
    Orchestrate sequential continual training across multiple cities.

    For each city in order:
      1. Load data + config
      2. Load global trunk from registry (if exists)
      3. Train with EWC penalty on previous Fisher matrices
         + replay loss on previous coresets
      4. Compute Fisher matrix for this city
      5. Select coreset from this city
      6. Update Welford scaler with this city's features
      7. Register city in the registry
      8. Update global trunk

    Parameters
    ----------
    city_configs : ordered list of project.yml paths
    registry_path : path to the city registry directory
    output_dir : base directory for per-city outputs
    ewc_lambda : weight for EWC penalty (default: 1.0)
    replay_lambda : weight for replay loss (default: 0.5)
    coreset_size : points per city for experience replay

    Returns
    -------
    dict with per_city_metrics, training_order, registry_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = CityRegistry(registry_path)
    coreset_selector = CoresetSelector(coreset_size=coreset_size)
    welford = _load_or_init_welford(registry)

    results: dict[str, Any] = {
        "per_city_metrics": {},
        "training_order": [],
        "registry_path": str(registry_path),
    }

    for city_idx, config_path in enumerate(city_configs):
        config_path = Path(config_path)
        logger.info("=" * 60)
        logger.info("City %d/%d: %s", city_idx + 1, len(city_configs), config_path.name)
        logger.info("=" * 60)

        # ---- 1. Load data ----
        config, y, coords, features, feature_names = _load_project(config_path)
        city_name = config.get("registry", {}).get("city_name",
                     config_path.stem.replace("_", "-"))

        # ---- 2. Update Welford scaler ----
        welford.partial_fit(features)

        # ---- 3. Apply Welford scaling ----
        features_scaled = welford.transform(features)

        # ---- 4. Load previous continual learning artifacts ----
        # Get EWC artifacts from all previous cities
        previous_fishers = registry.load_all_fisher_matrices()
        previous_coresets = registry.load_all_coresets()

        # Load global trunk if available
        global_trunk = registry.load_global_trunk()

        # ---- 5. Configure transfer and continual params ----
        train_config = copy.deepcopy(config)
        train_config["_transfer"] = {}

        if global_trunk is not None:
            # Warm start from global trunk
            trunk_path = output_dir / f"{city_name}_loaded_trunk.pt"
            import torch
            torch.save(global_trunk, trunk_path)
            train_config["_transfer"]["trunk_path"] = str(trunk_path)
            train_config["_transfer"]["freeze_trunk"] = False  # allow updates

        # Add EWC + replay config
        # Load optimal params (θ*) from trunk checkpoints of previous cities
        previous_optimal_params = []
        for city_name, _ in previous_fishers:
            try:
                record = registry.load_city(city_name)
                if record.trunk_checkpoint is not None:
                    previous_optimal_params.append(record.trunk_checkpoint)
            except Exception:
                pass

        train_config["_continual"] = {
            "ewc_lambda": ewc_lambda if previous_fishers else 0.0,
            "replay_lambda": replay_lambda if previous_coresets else 0.0,
            "fisher_matrices": [fm for _, fm in previous_fishers],
            "optimal_params_list": previous_optimal_params,
            "previous_coresets": previous_coresets,
        }

        # ---- 6. Train ----
        city_output = output_dir / city_name
        folds = _build_folds(y, coords, config)

        city_output.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        train_result = train_neural_meta(
            config=train_config,
            y=y,
            coords=coords,
            feature_matrix=features_scaled,
            feature_names=feature_names,
            output_dir=city_output,
            folds=folds,
        )
        elapsed = time.time() - t0

        metrics = train_result.get("metrics", {})
        metrics["training_time"] = elapsed
        logger.info("City %s: R²=%.4f, time=%.1fs",
                    city_name, metrics.get("r2", float("nan")), elapsed)

        # ---- 7. Post-training: compute Fisher, select coreset ----
        fisher_matrix = train_result.get("fisher_matrix")
        if fisher_matrix is None:
            logger.warning("No Fisher matrix returned for %s; EWC will be weaker", city_name)
            fisher_matrix = {}

        coreset = coreset_selector.select(features, y, coords)

        # ---- 8. Register city ----
        trunk_state = train_result.get("trunk_state_dict", {})
        registry.register_city(
            city_name=city_name,
            trunk_state_dict=trunk_state,
            fisher_matrix=fisher_matrix,
            coreset_X=coreset.get("X"),
            coreset_coords=coreset.get("coords"),
            coreset_y=coreset.get("y"),
            welford_state={
                "n_features": welford.n_features,
                "count": welford.count,
                "mean": welford.mean,
                "M2": welford.M2,
            },
            metrics=metrics,
        )

        # ---- 9. Update global trunk ----
        if trunk_state:
            registry.update_global_trunk(trunk_state)

        results["per_city_metrics"][city_name] = metrics
        results["training_order"].append(city_name)

    # Save summary
    summary_path = output_dir / "continual_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Continual training summary saved to %s", summary_path)

    return results



def _load_or_init_welford(registry: CityRegistry) -> WelfordScaler:
    """
    Load the most recent Welford scaler state from the registry,
    or initialise a fresh one.
    """
    cities = registry.list_cities()
    if not cities:
        return WelfordScaler()

    # Load from last registered city
    last_city = cities[-1]
    try:
        record = registry.load_city(last_city)
        if record.welford_state is not None:
            scaler = WelfordScaler()
            scaler.n_features = record.welford_state["n_features"]
            scaler.count = record.welford_state["count"]
            scaler.mean = record.welford_state["mean"]
            scaler.M2 = record.welford_state["M2"]
            logger.info("Loaded Welford scaler from city '%s' (count=%d)",
                        last_city, scaler.count)
            return scaler
    except Exception as e:
        logger.warning("Could not load Welford state from %s: %s", last_city, e)

    return WelfordScaler()
