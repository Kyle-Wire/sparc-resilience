"""SPARC registry: city registry (continual learning) + run-wide artifact registry.

``CityRegistry`` is imported lazily so that lightweight callers (server,
exports, tests) don't pay the cost of pulling in torch when they only need
``RunRegistry`` or the manifest models.
"""

from sparc.registry.manifest import (
    ArtifactEntry,
    ArtifactFormat,
    ColumnSpec,
    Dataset,
    DatasetKind,
    DatasetSchema,
    RunManifest,
    StageManifest,
    StageStatus,
)
from sparc.registry.run_registry import RunRegistry

__all__ = [
    "CityRegistry",
    "RunRegistry",
    "RunManifest",
    "StageManifest",
    "ArtifactEntry",
    "ArtifactFormat",
    "ColumnSpec",
    "Dataset",
    "DatasetKind",
    "DatasetSchema",
    "StageStatus",
]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "CityRegistry":
        from sparc.registry.city_registry import CityRegistry as _CityRegistry
        return _CityRegistry
    raise AttributeError(f"module 'sparc.registry' has no attribute {name!r}")
