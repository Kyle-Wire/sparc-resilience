"""
City Registry for SPARC V3 continual spatial learning.

Manages cross-city artifacts: shared trunk checkpoints, Fisher matrices
(EWC), experience replay coresets, Welford scaler states, and per-city
metrics.  The registry enables sequential training across cities without
catastrophic forgetting.

Directory structure:
    sparc_registry/
        registry.json             # manifest: city list, training order
        global_trunk.pt           # best shared trunk across all cities
        providence/
            trunk_checkpoint.pt   # trunk state after this city
            fisher_matrix.pt      # diagonal Fisher for EWC
            coreset.npz           # representative points for replay
            welford_state.pkl     # online scaler state
            metrics.json          # R², RMSE, Moran's I
        boston/
            ...
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CityRecord:
    """In-memory representation of a registered city's artifacts."""

    name: str
    trunk_checkpoint: dict[str, torch.Tensor] | None = None
    fisher_matrix: dict[str, torch.Tensor] | None = None
    coreset_X: np.ndarray | None = None
    coreset_coords: np.ndarray | None = None
    coreset_y: np.ndarray | None = None
    welford_state: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    trained_at: str | None = None


class CityRegistry:
    """
    Persistent registry of city-specific training artifacts.

    Parameters
    ----------
    registry_path : str | Path
        Root directory for the registry.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.root = Path(registry_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "registry.json"
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest management
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {
            "cities": [],
            "training_order": [],
            "city_index": {},
            "created": datetime.now(timezone.utc).isoformat(),
        }

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # City registration
    # ------------------------------------------------------------------

    def register_city(
        self,
        city_name: str,
        trunk_state_dict: dict[str, torch.Tensor],
        fisher_matrix: dict[str, torch.Tensor] | None = None,
        coreset_X: np.ndarray | None = None,
        coreset_coords: np.ndarray | None = None,
        coreset_y: np.ndarray | None = None,
        welford_state: Any | None = None,
        metrics: dict[str, Any] | None = None,
        climate_zone: str | None = None,
    ) -> None:
        """
        Register a city's training artifacts in the registry.

        Overwrites any existing registration for the same city.
        """
        city_dir = self.root / city_name
        city_dir.mkdir(parents=True, exist_ok=True)

        # Save trunk checkpoint
        torch.save(trunk_state_dict, city_dir / "trunk_checkpoint.pt")

        # Save Fisher matrix (EWC)
        if fisher_matrix is not None:
            torch.save(fisher_matrix, city_dir / "fisher_matrix.pt")

        # Save coreset (experience replay)
        if coreset_X is not None:
            np.savez(
                city_dir / "coreset.npz",
                X=coreset_X,
                coords=coreset_coords if coreset_coords is not None else np.array([]),
                y=coreset_y if coreset_y is not None else np.array([]),
            )

        # Save Welford scaler state
        if welford_state is not None:
            with open(city_dir / "welford_state.pkl", "wb") as f:
                pickle.dump(welford_state, f)

        # Save metrics
        if metrics is not None:
            with open(city_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)

        # Update manifest
        if city_name not in self._manifest["cities"]:
            self._manifest["cities"].append(city_name)
        self._manifest["training_order"].append({
            "city": city_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # city_index: keyed by city name, stores metadata including climate_zone
        if "city_index" not in self._manifest:
            self._manifest["city_index"] = {}
        self._manifest["city_index"][city_name] = {
            "name": city_name,
            "climate_zone": climate_zone,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_manifest()

        logger.info("Registered city '%s' (zone=%s) in registry at %s",
                    city_name, climate_zone, self.root)

    # ------------------------------------------------------------------
    # City loading
    # ------------------------------------------------------------------

    def load_city(self, city_name: str) -> CityRecord:
        """Load all artifacts for a registered city."""
        city_dir = self.root / city_name
        if not city_dir.exists():
            raise FileNotFoundError(f"City '{city_name}' not found in registry at {self.root}")

        record = CityRecord(name=city_name)

        # Trunk
        trunk_path = city_dir / "trunk_checkpoint.pt"
        if trunk_path.exists():
            record.trunk_checkpoint = torch.load(trunk_path, map_location="cpu", weights_only=True)

        # Fisher
        fisher_path = city_dir / "fisher_matrix.pt"
        if fisher_path.exists():
            record.fisher_matrix = torch.load(fisher_path, map_location="cpu", weights_only=True)

        # Coreset
        coreset_path = city_dir / "coreset.npz"
        if coreset_path.exists():
            data = np.load(coreset_path)
            record.coreset_X = data["X"]
            if "coords" in data and data["coords"].size > 0:
                record.coreset_coords = data["coords"]
            if "y" in data and data["y"].size > 0:
                record.coreset_y = data["y"]

        # Welford
        welford_path = city_dir / "welford_state.pkl"
        if welford_path.exists():
            with open(welford_path, "rb") as f:
                record.welford_state = pickle.load(f)

        # Metrics
        metrics_path = city_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                record.metrics = json.load(f)

        return record

    def load_by_climate(self, climate_zone: str) -> "CityRecord | None":
        """Load artifacts for the most-recently registered city with *climate_zone*.

        Parameters
        ----------
        climate_zone :
            Köppen-Geiger zone string (e.g. ``"Cfa"``).

        Returns
        -------
        CityRecord or None
            The city record, or ``None`` when no city with that zone is
            registered.  Multiple cities sharing a zone return the most
            recently registered one (last write wins).
        """
        city_index = self._manifest.get("city_index", {})
        # Collect cities matching the zone, preserving registration order
        matching = [
            entry["name"]
            for entry in city_index.values()
            if entry.get("climate_zone") == climate_zone
        ]
        if not matching:
            return None
        # Most-recently registered city is last in the list
        return self.load_city(matching[-1])

    # ------------------------------------------------------------------
    # Global trunk management
    # ------------------------------------------------------------------

    def load_global_trunk(self) -> dict[str, torch.Tensor] | None:
        """Load the global (best) shared trunk state dict."""
        path = self.root / "global_trunk.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def update_global_trunk(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Save a new global trunk checkpoint."""
        torch.save(state_dict, self.root / "global_trunk.pt")
        logger.info("Updated global trunk at %s", self.root / "global_trunk.pt")

    # ------------------------------------------------------------------
    # Coreset loading (for replay)
    # ------------------------------------------------------------------

    def load_all_coresets(self) -> list[dict[str, np.ndarray]]:
        """
        Load coresets from all registered cities.

        Returns list of dicts with keys 'X', 'coords', 'y', 'city'.
        """
        coresets = []
        for city_name in self._manifest["cities"]:
            coreset_path = self.root / city_name / "coreset.npz"
            if coreset_path.exists():
                data = np.load(coreset_path)
                entry = {"X": data["X"], "city": city_name}
                if "coords" in data and data["coords"].size > 0:
                    entry["coords"] = data["coords"]
                if "y" in data and data["y"].size > 0:
                    entry["y"] = data["y"]
                coresets.append(entry)
        return coresets

    def load_all_fisher_matrices(self) -> list[tuple[str, dict[str, torch.Tensor]]]:
        """Load Fisher matrices from all registered cities."""
        fishers = []
        for city_name in self._manifest["cities"]:
            fisher_path = self.root / city_name / "fisher_matrix.pt"
            if fisher_path.exists():
                fm = torch.load(fisher_path, map_location="cpu", weights_only=True)
                fishers.append((city_name, fm))
        return fishers

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_cities(self) -> list[str]:
        """Return list of registered city names."""
        return list(self._manifest["cities"])

    def n_cities(self) -> int:
        """Number of registered cities."""
        return len(self._manifest["cities"])

    def has_city(self, city_name: str) -> bool:
        """Check if a city is registered."""
        return city_name in self._manifest["cities"]

    def query_coresets_by_bandwidth_cluster(
        self,
        threshold: float = 0.3,
    ) -> list[dict]:
        """
        Return all city coresets annotated with their log-bandwidth label.

        Bandwidth is read from city metrics key ``"median_log_bandwidth"``.
        Cities without this key are assigned ``log_bw = 0.0`` (neutral prior).

        Parameters
        ----------
        threshold : float
            Reserved for caller use; not applied here (returned for informational
            parity with :func:`~sparc.training.spatial_contrastive.mine_positive_pairs`).

        Returns
        -------
        List of dicts with keys ``"X"``, ``"coords"``, ``"log_bw"``, ``"city"``.
        """
        from sparc.training.spatial_contrastive import query_coresets_by_bandwidth_cluster
        return query_coresets_by_bandwidth_cluster(self, threshold=threshold)

    def __repr__(self) -> str:
        return f"CityRegistry({self.root}, cities={self.list_cities()})"
