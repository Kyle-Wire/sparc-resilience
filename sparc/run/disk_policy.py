"""
Disk-write policy for the SPARC pipeline.

Single source of truth controlling whether the pipeline writes per-stage
artifacts to disk in addition to the canonical SQLite-backed
``ArtifactStore``.  When disabled (the default), the run directory contains
only:

* the session log
* the project file (``*.sparcproj``)
* the artifact database (``artifacts.db``) plus its manifest sidecar
  (``artifacts_manifest.json`` / ``run_manifest.json``)
* the input data parquet/CSV used to seed the run

All other intermediate stage outputs (CSVs, JSONs, GPKGs, joblib dumps,
``.npy`` arrays, model checkpoints, …) are written exclusively into the
artifact DB and read back via :class:`sparc.registry.store.ArtifactStore`.

Configuration order of precedence (highest first):

1. Explicit call to :func:`set_disk_writes`.
2. Environment variable ``SPARC_DISK_WRITES`` (``"1"``/``"true"`` enables;
   ``"0"``/``"false"`` disables).
3. Default: ``False`` (disk writes OFF).

Callers should treat the returned flag as read-only state.  To produce a
human-readable file from a stored artifact at runtime, use
``ArtifactStore.export(...)`` rather than re-enabling disk writes.
"""

from __future__ import annotations

import os
import threading

_DEFAULT: bool = False
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_lock = threading.Lock()
_explicit: bool | None = None  # set via set_disk_writes(); overrides env


def _env_value() -> bool | None:
    raw = os.environ.get("SPARC_DISK_WRITES")
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return None


def disk_writes_enabled() -> bool:
    """Return ``True`` when the pipeline is allowed to create per-stage files."""
    with _lock:
        if _explicit is not None:
            return _explicit
    env = _env_value()
    if env is not None:
        return env
    return _DEFAULT


def set_disk_writes(enabled: bool | None) -> None:
    """Override the disk-write flag for the current process.

    Pass ``None`` to clear the override and fall back to env / default.
    """
    global _explicit
    with _lock:
        _explicit = None if enabled is None else bool(enabled)


__all__ = ["disk_writes_enabled", "set_disk_writes"]
