"""In-memory ArtifactStore adapter for tests and stage isolation.

``MemoryStore`` implements the same write/read interface as
:class:`~sparc.registry.store.ArtifactStore` but keeps all data in a
plain dict. No database, no filesystem. Intended for:

- Unit/integration tests that need to verify artifacts are written
- Running pipeline stages without a database configured
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class MemoryStore:
    """In-memory drop-in for :class:`~sparc.registry.store.ArtifactStore`.

    Stores everything in ``self._data``, keyed by ``(str(stage), artifact_id)``.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], Any] = {}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_struct(
        self,
        stage: str | int,
        artifact_id: str,
        payload: dict,
        *,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store *payload* under ``(stage, artifact_id)``."""
        self._data[(str(stage), artifact_id)] = payload

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
    ) -> None:
        """Store *obj* under ``(stage, artifact_id)``."""
        self._data[(str(stage), artifact_id)] = obj

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
    ) -> str:
        """Store *df* under ``(stage, artifact_id)``; returns a table name token."""
        self._data[(str(stage), artifact_id)] = df
        return f"{stage}__{artifact_id}"

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_struct(self, stage: str | int, artifact_id: str) -> dict:
        """Return the stored struct; raises :exc:`KeyError` if not found."""
        key = (str(stage), artifact_id)
        if key not in self._data:
            raise KeyError(f"No struct stored for stage={stage!r} artifact_id={artifact_id!r}")
        return self._data[key]

    def read_blob(self, stage: str | int, artifact_id: str, *, trusted: bool = False) -> Any:
        """Return the stored blob; raises :exc:`KeyError` if not found."""
        key = (str(stage), artifact_id)
        if key not in self._data:
            raise KeyError(f"No blob stored for stage={stage!r} artifact_id={artifact_id!r}")
        return self._data[key]

    def read_table(self, stage: str | int, artifact_id: str) -> Any:
        """Return the stored table; raises :exc:`KeyError` if not found."""
        key = (str(stage), artifact_id)
        if key not in self._data:
            raise KeyError(f"No table stored for stage={stage!r} artifact_id={artifact_id!r}")
        return self._data[key]

    def read_any(self, stage: str | int, artifact_id: str, *, trusted: bool = False) -> Any:
        """Return whatever was stored under ``(stage, artifact_id)``."""
        key = (str(stage), artifact_id)
        if key not in self._data:
            raise KeyError(f"No artifact stored for stage={stage!r} artifact_id={artifact_id!r}")
        return self._data[key]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_artifacts(self, stage: str | int | None = None) -> list[tuple[str, str]]:
        """Return ``[(stage, artifact_id), ...]`` for all stored entries, optionally filtered."""
        if stage is None:
            return list(self._data.keys())
        s = str(stage)
        return [(sk, aid) for (sk, aid) in self._data if sk == s]

    def __len__(self) -> int:
        return len(self._data)
