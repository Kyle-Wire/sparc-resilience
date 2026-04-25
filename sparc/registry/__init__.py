"""SPARC registry: city registry (continual learning) + run-wide artifact registry."""

from sparc.registry.manifest import (
    ArtifactEntry,
    ArtifactFormat,
    RunManifest,
    StageManifest,
    StageStatus,
)
from sparc.registry.run_registry import RunRegistry

# CityRegistry has a heavy ML dependency chain (torch, etc.).
# Import lazily so that lightweight consumers (tests, server) can use
# RunRegistry without requiring the full ML stack.
try:
    from sparc.registry.city_registry import CityRegistry
except ImportError:  # torch not installed
    CityRegistry = None  # type: ignore[assignment,misc]

__all__ = [
    "CityRegistry",
    "RunRegistry",
    "RunManifest",
    "StageManifest",
    "ArtifactEntry",
    "ArtifactFormat",
    "StageStatus",
]
