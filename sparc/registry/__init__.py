"""SPARC registry: city registry (continual learning) + run-wide artifact registry."""

from sparc.registry.city_registry import CityRegistry
from sparc.registry.manifest import (
    ArtifactEntry,
    ArtifactFormat,
    RunManifest,
    StageManifest,
    StageStatus,
    StorageKind,
)
from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore, ArtifactStoreError, get_active_store
from sparc.registry.backend import StorageBackend
from sparc.registry.memory_store import MemoryStore

__all__ = [
    "CityRegistry",
    "RunRegistry",
    "RunManifest",
    "StageManifest",
    "ArtifactEntry",
    "ArtifactFormat",
    "StageStatus",
    "StorageKind",
    "ArtifactStore",
    "ArtifactStoreError",
    "get_active_store",
    # Storage seam (C1)
    "StorageBackend",
    "MemoryStore",
]
