"""
``artifact_io``: tiny façade over :class:`ArtifactStore` for legacy stage
modules that previously wrote files directly with ``df.to_csv``,
``json.dump``, ``joblib.dump``, ``np.save``, or ``gdf.to_file(...)``.

The functions here always route through the active
:class:`~sparc.registry.store.ArtifactStore` (i.e. into ``artifacts.db``)
and never touch disk.  Stage modules can therefore replace direct-disk
writes one-for-one without plumbing a ``ResultStore`` instance through
their call graphs.

Reads use :meth:`ArtifactStore.read_any` which transparently handles
db-resident artifacts and on-disk legacy-path artifacts; consumers do
not need to know where the data lives.

A small set of *path-style* compatibility helpers (``stage_dir``,
``artifact_id_for``) let legacy code that hard-codes file names compute
the equivalent ``(stage, artifact_id)`` tuple deterministically.
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Union

# ---------------------------------------------------------------------------
# Active store accessor
# ---------------------------------------------------------------------------

_injected_store = None  # set via set_store() for tests / isolated runs


def set_store(store) -> None:
    """Inject a store instance (e.g. :class:`~sparc.registry.memory_store.MemoryStore`).

    All subsequent ``save_*`` calls in this process will route to *store*.
    Call :func:`clear_store` to remove the override.
    """
    global _injected_store
    _injected_store = store


def clear_store() -> None:
    """Remove any previously injected store; revert to the active-store discovery."""
    global _injected_store
    _injected_store = None


def _store():
    """Return the active :class:`ArtifactStore`, or ``None`` when unavailable."""
    if _injected_store is not None:
        return _injected_store
    try:
        from sparc.registry.store import get_active_store
        return get_active_store()
    except Exception:  # noqa: BLE001 — never break a writer because of registry init
        return None


def _stage_key(stage: Union[int, str]) -> str:
    return str(stage)


def _id_from_filename(name: str, *, subdir: Optional[str] = None) -> str:
    """Derive a deterministic artifact_id from a legacy filename + optional subdir."""
    base = Path(name).stem  # strip extension
    if subdir:
        # Embed the subdir so artifacts in different subfolders don't collide.
        sub = subdir.strip("/").replace("/", "__").replace("\\", "__")
        return f"{sub}__{base}"
    return base


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def save_struct(
    stage: Union[int, str],
    name: str,
    payload: Any,
    *,
    subdir: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Persist a JSON-serializable dict/list to ``result_structs``.

    Returns the resolved ``artifact_id`` (useful for callers that want to
    log it).  Silently no-ops when no active store is available.
    """
    store = _store()
    artifact_id = _id_from_filename(name, subdir=subdir)
    if store is None:
        warnings.warn(
            f"artifact_io.save_struct({stage}/{artifact_id}): no active store; "
            "payload not persisted.",
            RuntimeWarning,
            stacklevel=2,
        )
        return artifact_id
    # Coerce non-dict payloads (lists, primitives) into a dict so the
    # ArtifactStore JSON schema is satisfied without changing call sites.
    if not isinstance(payload, dict):
        payload = {"value": payload}
    store.write_struct(
        _stage_key(stage), artifact_id, payload,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    return artifact_id


def save_table(
    stage: Union[int, str],
    name: str,
    df,
    *,
    subdir: Optional[str] = None,
    geometry_col: Optional[str] = None,
    crs: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Persist a DataFrame (or GeoDataFrame) as a SQLite table."""
    store = _store()
    artifact_id = _id_from_filename(name, subdir=subdir)
    if store is None:
        warnings.warn(
            f"artifact_io.save_table({stage}/{artifact_id}): no active store; "
            "DataFrame not persisted.",
            RuntimeWarning,
            stacklevel=2,
        )
        return artifact_id
    store.write_table(
        _stage_key(stage), artifact_id, df,
        geometry_col=geometry_col, crs=crs,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    return artifact_id


def save_blob(
    stage: Union[int, str],
    name: str,
    obj: Any,
    *,
    serializer: str = "joblib",
    subdir: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Persist a binary object (sklearn estimators, scalers, fold lists, …)."""
    store = _store()
    artifact_id = _id_from_filename(name, subdir=subdir)
    if store is None:
        warnings.warn(
            f"artifact_io.save_blob({stage}/{artifact_id}): no active store; "
            "object not persisted.",
            RuntimeWarning,
            stacklevel=2,
        )
        return artifact_id
    store.write_blob(
        _stage_key(stage), artifact_id, obj,
        serializer=serializer,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    return artifact_id


def save_array(
    stage: Union[int, str],
    name: str,
    arr,
    *,
    subdir: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Persist a numpy array (saved as a pickled blob with serializer='pickle')."""
    return save_blob(
        stage, name, arr, serializer="pickle", subdir=subdir,
        producer=producer, consumers=consumers, metadata=metadata,
    )


def save_text(
    stage: Union[int, str],
    name: str,
    text: str,
    *,
    subdir: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Persist a plain-text payload as a struct (``{"text": …}``)."""
    return save_struct(
        stage, name, {"text": text}, subdir=subdir,
        producer=producer, consumers=consumers, metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Readers — all return ``None`` when artifact missing (callers handle fallback)
# ---------------------------------------------------------------------------


def load_struct(
    stage: Union[int, str], name: str, *, subdir: Optional[str] = None,
) -> Optional[Any]:
    store = _store()
    if store is None:
        return None
    artifact_id = _id_from_filename(name, subdir=subdir)
    if not store.has(_stage_key(stage), artifact_id):
        return None
    try:
        payload = store.read_struct(_stage_key(stage), artifact_id)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(payload, dict) and set(payload.keys()) == {"value"}:
        return payload["value"]
    if isinstance(payload, dict) and set(payload.keys()) == {"text"}:
        return payload["text"]
    return payload


def load_table(
    stage: Union[int, str], name: str, *, subdir: Optional[str] = None,
):
    store = _store()
    if store is None:
        return None
    artifact_id = _id_from_filename(name, subdir=subdir)
    if not store.has(_stage_key(stage), artifact_id):
        return None
    try:
        return store.read_table(_stage_key(stage), artifact_id)
    except Exception:  # noqa: BLE001
        return None


def load_blob(
    stage: Union[int, str], name: str, *, subdir: Optional[str] = None,
) -> Optional[Any]:
    store = _store()
    if store is None:
        return None
    artifact_id = _id_from_filename(name, subdir=subdir)
    if not store.has(_stage_key(stage), artifact_id):
        return None
    try:
        return store.read_blob(_stage_key(stage), artifact_id)
    except Exception:  # noqa: BLE001
        return None


def load_array(
    stage: Union[int, str], name: str, *, subdir: Optional[str] = None,
):
    warnings.warn(
        "load_array() is a deprecated pass-through for load_blob(); "
        "use load_blob() directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_blob(stage, name, subdir=subdir)


def load_text(
    stage: Union[int, str], name: str, *, subdir: Optional[str] = None,
) -> Optional[str]:
    val = load_struct(stage, name, subdir=subdir)
    if isinstance(val, str):
        return val
    if isinstance(val, dict) and "text" in val:
        return val["text"]
    return None


def has(
    stage: Union[int, str], name: str, *, subdir: Optional[str] = None,
) -> bool:
    store = _store()
    if store is None:
        return False
    artifact_id = _id_from_filename(name, subdir=subdir)
    try:
        return store.has(_stage_key(stage), artifact_id)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Disk-aware helpers for path-based legacy call sites
# ---------------------------------------------------------------------------
# These are designed for near-1:1 replacement of ``joblib.dump(obj, path)``,
# ``df.to_csv(path)``, ``json.dump(obj, fp)``, ``np.save(path, arr)``, and
# ``gdf.to_file(path)`` calls in stage modules.  They:
#
#   1. Always write to the active :class:`ArtifactStore` (DB-resident).
#   2. Optionally write to ``path`` when :func:`disk_writes_enabled` is True,
#      creating parent directories as needed.
#
# ``stage`` and ``artifact_id`` should be passed explicitly for stable
# ``(stage, id)`` keys; when omitted, ``artifact_id`` is derived from
# ``Path(path).stem``.
# ---------------------------------------------------------------------------


def _disk_on() -> bool:
    try:
        from sparc.run.disk_policy import disk_writes_enabled
        return disk_writes_enabled()
    except Exception:  # noqa: BLE001
        return True


def _ensure_parent(path: Union[str, Path]) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def save_blob_path(
    obj: Any,
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    serializer: str = "joblib",
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[Path]:
    """Drop-in replacement for ``joblib.dump(obj, path)``.

    Always writes to the artifact store; writes to ``path`` only when
    disk writes are enabled.  Returns the on-disk path when written,
    else ``None``.
    """
    aid = artifact_id or Path(path).stem
    save_blob(
        stage, aid, obj, serializer=serializer,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    if _disk_on():
        _ensure_parent(path)
        if serializer == "joblib":
            import joblib  # type: ignore[import-not-found]
            joblib.dump(obj, path)
        elif serializer == "pickle":
            import pickle as _pkl
            with open(path, "wb") as f:
                _pkl.dump(obj, f, protocol=_pkl.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"save_blob_path: unsupported serializer={serializer!r}")
        return Path(path)
    return None


def load_blob_path(
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    serializer: str = "joblib",
) -> Optional[Any]:
    """Drop-in replacement for ``joblib.load(path)``.

    Tries the artifact store first; falls back to on-disk ``path`` when
    the store has no entry but the file exists.  Returns ``None`` when
    neither is available.
    """
    aid = artifact_id or Path(path).stem
    obj = load_blob(stage, aid)
    if obj is not None:
        return obj
    p = Path(path)
    if p.exists():
        if serializer == "joblib":
            import joblib  # type: ignore[import-not-found]
            return joblib.load(p)
        if serializer == "pickle":
            import pickle as _pkl
            with open(p, "rb") as f:
                return _pkl.load(f)
    return None


def save_struct_path(
    payload: Any,
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[Path]:
    """Drop-in replacement for ``json.dump(obj, open(path, 'w'))``."""
    aid = artifact_id or Path(path).stem
    save_struct(
        stage, aid, payload,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    if _disk_on():
        _ensure_parent(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return Path(path)
    return None


def save_table_path(
    df,
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    fmt: str = "csv",
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[Path]:
    """Drop-in replacement for ``df.to_csv(path)`` / ``df.to_parquet(path)``."""
    aid = artifact_id or Path(path).stem
    save_table(
        stage, aid, df,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    if _disk_on():
        _ensure_parent(path)
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"save_table_path: unsupported fmt={fmt!r}")
        return Path(path)
    return None


def save_geo_path(
    gdf,
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    driver: str = "GPKG",
    layer: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[Path]:
    """Drop-in replacement for ``gdf.to_file(path, driver='GPKG')``."""
    aid = artifact_id or Path(path).stem
    save_table(
        stage, aid, gdf,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    if _disk_on():
        _ensure_parent(path)
        if layer is not None:
            gdf.to_file(path, driver=driver, layer=layer)
        else:
            gdf.to_file(path, driver=driver)
        return Path(path)
    return None


def save_array_path(
    arr,
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[Path]:
    """Drop-in replacement for ``np.save(path, arr)``."""
    aid = artifact_id or Path(path).stem
    save_array(
        stage, aid, arr,
        producer=producer, consumers=consumers, metadata=metadata,
    )
    if _disk_on():
        import numpy as _np
        _ensure_parent(path)
        _np.save(path, arr)
        return Path(path)
    return None


def ensure_dir(path: Union[str, Path]) -> Optional[Path]:
    """Replacement for ``os.makedirs(d, exist_ok=True)`` that no-ops when
    disk writes are disabled.  Returns the path when created, else None.
    """
    if not _disk_on():
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Disk-aware READ helpers (counterparts to save_*_path)
# ---------------------------------------------------------------------------
# These mirror ``load_blob_path``: try the active ArtifactStore first via
# ``(stage, artifact_id)``; fall back to disk only when the store has no
# entry AND the legacy file exists.  Each raises FileNotFoundError naming
# both lookup strategies if neither succeeds, so DB-only regressions are
# obvious.
#
# ``artifact_id`` defaults to ``Path(path).stem``, matching the
# ``save_*_path`` convention.
# ---------------------------------------------------------------------------


def _try_store_read_any(stage: Union[int, str], artifact_id: str) -> Any:
    """Return the artifact via ``store.read_any`` or ``None`` if absent."""
    store = _store()
    if store is None:
        return None
    try:
        if not store.has(_stage_key(stage), artifact_id):
            return None
        return store.read_any(_stage_key(stage), artifact_id)
    except Exception:  # noqa: BLE001
        return None


def load_table_path(
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    fmt: str = "csv",
):
    """Drop-in replacement for ``pd.read_csv(path)`` / ``pd.read_parquet(path)``.

    Tries the artifact store first; falls back to on-disk ``path`` when
    the store has no entry but the file exists.  Raises
    :class:`FileNotFoundError` when neither is available.
    """
    aid = artifact_id or Path(path).stem
    val = _try_store_read_any(stage, aid)
    if val is not None:
        return val
    p = Path(path)
    if p.exists():
        import pandas as _pd
        if fmt == "csv":
            return _pd.read_csv(p)
        if fmt == "parquet":
            return _pd.read_parquet(p)
        raise ValueError(f"load_table_path: unsupported fmt={fmt!r}")
    raise FileNotFoundError(
        f"Artifact not found: stage={stage} id={aid} (and no file at {p})"
    )


def load_struct_path(
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
) -> Any:
    """Drop-in replacement for ``json.load(open(path))``.

    Tries the artifact store first; falls back to on-disk JSON at
    ``path``.  Raises :class:`FileNotFoundError` when neither is
    available.
    """
    aid = artifact_id or Path(path).stem
    val = _try_store_read_any(stage, aid)
    if val is not None:
        return val
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"Artifact not found: stage={stage} id={aid} (and no file at {p})"
    )


def load_geo_path(
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
    layer: Optional[str] = None,
):
    """Drop-in replacement for ``gpd.read_file(path)``.

    Tries the artifact store first; falls back to on-disk GeoPackage at
    ``path``.  Raises :class:`FileNotFoundError` when neither is
    available.
    """
    aid = artifact_id or Path(path).stem
    val = _try_store_read_any(stage, aid)
    if val is not None:
        return val
    p = Path(path)
    if p.exists():
        import geopandas as _gpd  # type: ignore[import-not-found]
        if layer is not None:
            return _gpd.read_file(p, layer=layer)
        return _gpd.read_file(p)
    raise FileNotFoundError(
        f"Artifact not found: stage={stage} id={aid} (and no file at {p})"
    )


def exists_path(
    path: Union[str, Path],
    *,
    stage: Union[int, str],
    artifact_id: Optional[str] = None,
) -> bool:
    """Replacement for ``os.path.exists(path)`` for catalogued artifacts.

    Returns True if either the artifact store has the entry OR the file
    exists on disk.
    """
    aid = artifact_id or Path(path).stem
    store = _store()
    if store is not None:
        try:
            if store.has(_stage_key(stage), aid):
                return True
        except Exception:  # noqa: BLE001
            pass
    return Path(path).exists()


__all__ = [
    "save_struct", "save_table", "save_blob", "save_array", "save_text",
    "load_struct", "load_table", "load_blob", "load_array", "load_text",
    "has",
    # Disk-aware helpers
    "save_blob_path", "load_blob_path", "save_struct_path", "save_table_path",
    "save_geo_path", "save_array_path", "ensure_dir",
    "load_table_path", "load_struct_path", "load_geo_path", "exists_path",
]
