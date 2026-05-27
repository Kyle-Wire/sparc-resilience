"""StorageBackend protocol — the formal seam between RunRegistry/ArtifactStore and callers.

Both :class:`~sparc.registry.store.ArtifactStore` and
:class:`~sparc.registry.memory_store.MemoryStore` satisfy this protocol, so
any pipeline stage, server route, or test can accept a ``StorageBackend`` and
receive either a full SQLite-backed store or an in-memory stub.

Usage
-----
::

    from sparc.registry.backend import StorageBackend
    from sparc.registry.memory_store import MemoryStore

    def run_stage(store: StorageBackend) -> None:
        store.write_struct("1", "summary", {"rmse": 0.12})

    run_stage(MemoryStore())          # tests / isolated runs
    run_stage(ArtifactStore(registry))  # production
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Structural protocol satisfied by both ``ArtifactStore`` and ``MemoryStore``.

    All methods mirror those of :class:`~sparc.registry.store.ArtifactStore`
    exactly so that either implementation is a drop-in replacement.
    """

    def write_struct(
        self,
        stage: str | int,
        artifact_id: str,
        payload: dict,
        *,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None: ...

    def read_struct(self, stage: str | int, artifact_id: str) -> dict: ...

    def write_blob(
        self,
        stage: str | int,
        artifact_id: str,
        obj: Any,
        *,
        serializer: str = "pickle",
        mime: Optional[str] = None,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        metadata: Optional[dict] = None,
        threshold_bytes: int = 0,
    ) -> None: ...

    def read_blob(
        self, stage: str | int, artifact_id: str, *, trusted: bool = False
    ) -> Any: ...

    def write_table(
        self,
        stage: str | int,
        artifact_id: str,
        df: Any,
        *,
        geometry_col: Optional[str] = None,
        crs: Optional[str] = None,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        metadata: Optional[dict] = None,
        if_exists: str = "replace",
    ) -> str: ...

    def read_table(self, stage: str | int, artifact_id: str) -> Any: ...

    def read_any(
        self, stage: str | int, artifact_id: str, *, trusted: bool = False
    ) -> Any: ...

    def list_artifacts(
        self, stage: str | int | None = None
    ) -> list[tuple[str, str]]: ...
