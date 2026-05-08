"""Thread-safe LRU result cache for the SPARC FastAPI server.

Serialised endpoint responses are stored in-memory so repeated requests
for the same artifact (e.g. after re-navigating to the Results page) are
served without touching SQLite or deserialising geospatial data.

Key design choices
------------------
- Keys are ``(stage: str, artifact_id: str)`` tuples, mirroring the
  ArtifactStore's own namespace.
- The cache is *intentionally* untyped (stores ``Any``): the endpoints
  already know what they expect, and adding generics here would force
  re-serialisation on every read.
- Max size defaults to 50 entries (configurable via ``SPARC_RESULT_CACHE_SIZE``
  env var).  This is conservative — most result payloads are JSON dicts or
  GeoJSON FeatureCollections whose total in-process size is a few MB each.
- Invalidation is stage-scoped: when stage N completes, all keys with that
  stage are evicted.  Mirrors the frontend's ``invalidateStage`` call.
- Thread-safe: a single ``threading.Lock`` guards all mutations.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

_DEFAULT_MAX_SIZE = int(os.environ.get("SPARC_RESULT_CACHE_SIZE", "256"))


class ResultCache:
    """In-memory LRU cache for serialised endpoint results.

    Usage::

        cache = ResultCache()
        hit = cache.get("0", "correlogram_results")
        if hit is None:
            result = expensive_db_read()
            cache.set("0", "correlogram_results", result)
        else:
            result = hit
    """

    def __init__(self, maxsize: int = _DEFAULT_MAX_SIZE) -> None:
        self._maxsize = max(1, maxsize)
        self._store: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, stage: str | int, artifact_id: str) -> Any | None:
        """Return the cached value or ``None`` if not cached."""
        key = (str(stage), artifact_id)
        with self._lock:
            if key not in self._store:
                return None
            # Move to end (most recently used).
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, stage: str | int, artifact_id: str, value: Any) -> None:
        """Store *value* under *(stage, artifact_id)*, evicting LRU if needed."""
        key = (str(stage), artifact_id)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def invalidate(self, stage: str | int, artifact_id: str) -> None:
        """Evict a single entry (no-op if absent)."""
        key = (str(stage), artifact_id)
        with self._lock:
            self._store.pop(key, None)

    def invalidate_stage(self, stage: str | int) -> None:
        """Evict all entries for a given stage (called when a stage writes new data)."""
        stage_str = str(stage)
        with self._lock:
            to_remove = [k for k in self._store if k[0] == stage_str]
            for k in to_remove:
                del self._store[k]

    def clear(self) -> None:
        """Drop the entire cache (used on project load/unload)."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
