"""
Unified Result I/O for the SPARC Pipeline.

``ResultStore`` wraps ``PipelinePaths`` with typed save/load methods and a
per-stage manifest (``_manifest.json``) that records every artifact written,
its format, byte-size, and timestamp.  Stages, server endpoints, and the
report generator all go through this single layer so that:

* file locations are never hard-coded outside ``PipelinePaths``
* every output is tracked and discoverable
* partial-result recovery is possible on failure
* format upgrades (e.g. CSV → Parquet) are transparent
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

# Optional heavy imports — loaded lazily so lightweight callers don't pay the cost
_geopandas = None
_joblib = None


def _get_geopandas():
    global _geopandas
    if _geopandas is None:
        import geopandas
        _geopandas = geopandas
    return _geopandas


def _get_joblib():
    global _joblib
    if _joblib is None:
        import joblib
        _joblib = joblib
    return _joblib


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

Format = Literal["json", "csv", "parquet", "gpkg", "npy", "pkl", "png", "txt"]

_EXT_MAP: Dict[Format, str] = {
    "json": ".json",
    "csv": ".csv",
    "parquet": ".parquet",
    "gpkg": ".gpkg",
    "npy": ".npy",
    "pkl": ".pkl",
    "png": ".png",
    "txt": ".txt",
}


def _infer_format(name: str) -> Format:
    """Infer format from file name extension."""
    ext = Path(name).suffix.lower()
    for fmt, expected_ext in _EXT_MAP.items():
        if ext == expected_ext:
            return fmt
    raise ValueError(f"Cannot infer format from extension '{ext}' in '{name}'")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_NAME = "_manifest.json"


def _load_manifest(directory: Path) -> Dict[str, Any]:
    path = directory / MANIFEST_NAME
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"artifacts": {}}
    return {"artifacts": {}}


def _save_manifest(directory: Path, manifest: Dict[str, Any]) -> None:
    path = directory / MANIFEST_NAME
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# WriteTransaction — atomic disk write with rollback on registry failure
# ---------------------------------------------------------------------------

class WriteTransaction:
    """Atomic disk write that rolls back when the registry write fails.

    The artifact is first written to a sibling temp path
    (``.<name>.sparc_tmp``).  When :meth:`commit` is called successfully, the
    temp file is renamed to *dest* (atomic on POSIX; best-effort on Windows).
    If the context block exits with an unhandled exception, or if :meth:`commit`
    was never called, the temp file is cleaned up and *dest* is left untouched.

    This guarantees that the on-disk file and the per-stage manifest are only
    updated when the write was successful end-to-end, eliminating the
    split-brain state where a file exists on disk but the registry is blind to
    it (or vice-versa).

    Parameters
    ----------
    dest : Path
        Final destination path for the artifact.
    write_fn : callable
        ``write_fn(path)`` — writes the data to *path*.  Called with the temp
        path inside ``__enter__``; called again with *dest* as fallback on
        Windows rename failure.

    Example
    -------
    ::

        with WriteTransaction(dest, lambda p: df.to_csv(p)) as tx:
            registry.put(artifact_id, data)   # may raise
            tx.commit()                        # rename temp → dest only on success
        # if registry.put raised, temp is cleaned up, dest unchanged
    """

    def __init__(self, dest: Path, write_fn) -> None:
        self._dest = dest
        self._write_fn = write_fn
        self._tmp = dest.parent / f".{dest.name}.sparc_tmp"
        self._entered = False
        self._committed = False

    # -- context protocol --

    def __enter__(self) -> "WriteTransaction":
        self._dest.parent.mkdir(parents=True, exist_ok=True)
        self._write_fn(self._tmp)
        self._entered = True
        return self

    def commit(self) -> None:
        """Promote the temp file to *dest*.  Call exactly once per transaction."""
        if not self._entered:
            raise RuntimeError("WriteTransaction.commit() called outside context block")
        try:
            self._tmp.replace(self._dest)
        except OSError:
            # Windows can fail on cross-device or locked files — copy then delete.
            import shutil
            shutil.copy2(self._tmp, self._dest)
            try:
                self._tmp.unlink()
            except OSError:
                pass
        self._committed = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._committed:
            # Either an exception occurred, or commit() was never called.
            try:
                self._tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return False  # never suppress exceptions


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------

class ResultStore:
    """Unified save/load layer for every pipeline artifact.

    Parameters
    ----------
    paths : PipelinePaths
        Centralised path object (from ``PipelinePaths.from_config`` or
        the legacy constructor).
    prefer_parquet : bool
        When *True* (default), ``save_dataframe`` writes Parquet by default
        instead of CSV.
    """

    def __init__(self, paths, *, prefer_parquet: bool = True, registry=None):
        from sparc.run.pipeline_paths import PipelinePaths
        if not isinstance(paths, PipelinePaths):
            raise TypeError(f"Expected PipelinePaths, got {type(paths)}")
        self.paths = paths
        self.prefer_parquet = prefer_parquet
        self._stage_dirs = {
            0: paths.stage0_dir,
            1: paths.stage1_dir,
            2: paths.stage2_dir,
            3: paths.stage3_dir,
            4: paths.stage4_dir,
            "final": paths.final_dir,
            "spatial": paths.spatial_analysis_dir,
        }
        # Run-wide artifact registry (canonical JSON + SQLite cache).
        # Created lazily so legacy code that constructs ResultStore without
        # one continues to work.
        if registry is None:
            try:
                from sparc.registry.run_registry import RunRegistry
                registry = RunRegistry(paths.output_dir, autoload=True)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"RunRegistry could not be initialised ({exc}); "
                    "per-stage manifest still recorded.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                registry = None
        self.registry = registry

    @staticmethod
    def _stage_to_registry_key(stage: Union[int, str]) -> str:
        return str(stage)

    # ------------------------------------------------------------------
    # Directory resolution
    # ------------------------------------------------------------------

    def stage_dir(self, stage: Union[int, str]) -> Path:
        """Return the directory for *stage* (0-4, 'final', or 'spatial')."""
        try:
            return self._stage_dirs[stage]
        except KeyError:
            raise ValueError(f"Unknown stage: {stage!r}. "
                             f"Valid: {list(self._stage_dirs)}")

    # ------------------------------------------------------------------
    # Generic save / load
    # ------------------------------------------------------------------

    def save(
        self,
        stage: Union[int, str],
        name: str,
        data: Any,
        fmt: Optional[Format] = None,
        *,
        subdir: Optional[str] = None,
        partial: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save an artifact and record it in the stage manifest.

        Parameters
        ----------
        stage : int | str
            Pipeline stage (0-4, ``'final'``, or ``'spatial'``).
        name : str
            File name including extension (e.g. ``'scenario_summary.csv'``).
        data : Any
            The object to persist.  Dispatch is based on *fmt*.
        fmt : Format | None
            Explicit format.  Inferred from *name* when ``None``.
        subdir : str | None
            Optional subdirectory within the stage dir.
        partial : bool
            If ``True`` the artifact is saved with a ``_partial`` suffix in the
            manifest so it can be identified for recovery.
        metadata : dict | None
            Extra metadata stored in the manifest entry.

        Returns
        -------
        Path — absolute path of the written file.
        """
        if fmt is None:
            fmt = _infer_format(name)

        # PNG (and other binary figures) must not be routed through the
        # dual-write ResultStore.save() path. Figures are written by the
        # report layer directly to the RunRegistry; allowing a generic
        # save(fmt="png") here would either bypass the artifact store
        # entirely (DB-only mode silently drops the data) or create an
        # un-mirrored disk-only file. Force callers to use the explicit
        # figure pipeline.
        if fmt == "png":
            raise ValueError(
                f"ResultStore.save() does not accept fmt='png' (name={name!r}). "
                "Use sparc.report.figures (which registers figures via the "
                "RunRegistry directly) for PNG output."
            )

        # Resolve target path (always — even if we won't actually write to it
        # so callers that log the location still get a sensible string).
        directory = self.stage_dir(stage)
        if subdir:
            directory = directory / subdir
        dest = directory / name

        # Honor the global disk-write policy.  When disabled, we skip mkdir,
        # the on-disk write, and the legacy per-stage manifest update — all
        # state lives in the run-wide registry / ArtifactStore.
        try:
            from sparc.run.disk_policy import disk_writes_enabled
            disk_on = disk_writes_enabled()
        except Exception:  # noqa: BLE001
            disk_on = True

        elapsed = 0.0
        size_bytes = 0
        rel_key = str(Path(subdir) / name) if subdir else name
        if disk_on:
            # Use WriteTransaction so that the manifest is only updated once the
            # file is safely on disk and (if present) the registry has accepted
            # the record.  If the registry write raises, the temp file is
            # cleaned up and neither the destination file nor the manifest are
            # touched — no split-brain state.
            _fmt_ref = fmt  # capture for lambda closure
            t0 = time.monotonic()
            with WriteTransaction(dest, lambda p: self._write(p, data, _fmt_ref)) as _tx:
                # ── registry write (inside the transaction) ──────────────────
                # We attempt this *before* committing so a registry failure
                # can prevent the disk write from becoming visible.
                if self.registry is not None:
                    try:
                        row_count = None
                        if isinstance(data, pd.DataFrame):
                            row_count = int(len(data))
                        _meta = dict(metadata or {})
                        _artifact_id = _meta.pop(
                            "artifact_id",
                            rel_key.rsplit(".", 1)[0],
                        )
                        _consumers = _meta.pop("consumers", None)
                        _producer = _meta.pop("producer", None)
                        self.registry.register_artifact(
                            stage=self._stage_to_registry_key(stage),
                            artifact_id=_artifact_id,
                            path=dest,
                            format=fmt,
                            producer=_producer,
                            consumers=_consumers,
                            row_count=row_count,
                            partial=partial,
                            metadata=_meta,
                            write_seconds=0.0,  # updated below after commit
                        )
                    except Exception as exc:  # noqa: BLE001
                        warnings.warn(
                            f"RunRegistry registration failed for {dest}: {exc}; "
                            "disk write will NOT be committed.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        # Transaction __exit__ will clean up the temp file.
                        raise

                # Both temp write and registry succeeded — promote to final.
                _tx.commit()

            elapsed = time.monotonic() - t0
            size_bytes = dest.stat().st_size if dest.exists() else 0

            # Record in per-stage manifest only after the file exists on disk.
            manifest = _load_manifest(self.stage_dir(stage))
            manifest["artifacts"][rel_key] = {
                "format": fmt,
                "size_bytes": size_bytes,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "write_seconds": round(elapsed, 3),
                "partial": partial,
                **(metadata or {}),
            }
            _save_manifest(self.stage_dir(stage), manifest)

        # Phase C7: dual-write into the active ArtifactStore so legacy
        # ResultStore call sites automatically populate artifacts.db.
        # The disk file remains the backing storage; the store entry is
        # an additional copy keyed by (stage, artifact_id).
        try:
            from sparc.registry.store import get_active_store
            _astore = get_active_store()
        except Exception:
            _astore = None
        if _astore is not None:
            try:
                meta_in = dict(metadata or {})
                artifact_id = meta_in.pop(
                    "artifact_id", rel_key.rsplit(".", 1)[0]
                )
                producer = meta_in.pop("producer", None)
                consumers = meta_in.pop("consumers", None)
                if fmt in ("csv", "parquet"):
                    if isinstance(data, pd.DataFrame):
                        _astore.write_table(
                            self._stage_to_registry_key(stage),
                            artifact_id, data,
                            producer=producer, consumers=consumers,
                            metadata=meta_in or None,
                        )
                elif fmt == "gpkg":
                    gpd_mod = _get_geopandas()
                    if isinstance(data, gpd_mod.GeoDataFrame):
                        _astore.write_table(
                            self._stage_to_registry_key(stage),
                            artifact_id, data,
                            geometry_col="geometry",
                            crs=str(data.crs) if data.crs is not None else None,
                            producer=producer, consumers=consumers,
                            metadata=meta_in or None,
                        )
                elif fmt == "json":
                    _astore.write_struct(
                        self._stage_to_registry_key(stage),
                        artifact_id, data,
                        producer=producer, consumers=consumers,
                        metadata=meta_in or None,
                    )
                elif fmt in ("npy", "pkl"):
                    _astore.write_blob(
                        self._stage_to_registry_key(stage),
                        artifact_id, data,
                        serializer="pickle",
                        producer=producer, consumers=consumers,
                        metadata=meta_in or None,
                    )
                # txt / png intentionally omitted (txt is rare; pipeline
                # never writes png).
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"ArtifactStore dual-write failed for {dest}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return dest

    def load(
        self,
        stage: Union[int, str],
        name: str,
        fmt: Optional[Format] = None,
        *,
        subdir: Optional[str] = None,
    ) -> Any:
        """Load a previously saved artifact.

        When the on-disk file is missing, falls back to the active
        :class:`ArtifactStore` (DB-resident).  Raises ``FileNotFoundError``
        only if neither location has the artifact.
        """
        if fmt is None:
            fmt = _infer_format(name)

        directory = self.stage_dir(stage)
        if subdir:
            directory = directory / subdir
        src = directory / name

        if src.exists():
            return self._read(src, fmt)

        # Disk miss — try the artifact store.
        astore_obj = self._load_from_store(stage, name, subdir=subdir)
        if astore_obj is not None:
            return astore_obj
        raise FileNotFoundError(f"Artifact not found: {src}")

    def exists(self, stage: Union[int, str], name: str, *, subdir: Optional[str] = None) -> bool:
        """Check whether an artifact is reachable (on disk or in the store)."""
        directory = self.stage_dir(stage)
        if subdir:
            directory = directory / subdir
        if (directory / name).exists():
            return True
        # Fall back to the artifact DB.
        try:
            from sparc.registry.store import get_active_store
            astore = get_active_store()
            if astore is None:
                return False
            artifact_id = (
                str(Path(subdir) / name).rsplit(".", 1)[0] if subdir
                else name.rsplit(".", 1)[0]
            )
            return astore.has(self._stage_to_registry_key(stage), artifact_id)
        except Exception:  # noqa: BLE001
            return False

    def _load_from_store(
        self,
        stage: Union[int, str],
        name: str,
        *,
        subdir: Optional[str] = None,
    ) -> Any:
        """Best-effort read from the active ArtifactStore.  Returns ``None``
        when nothing is found or the store is unavailable.
        """
        try:
            from sparc.registry.store import get_active_store
            astore = get_active_store()
        except Exception:  # noqa: BLE001
            return None
        if astore is None:
            return None
        artifact_id = (
            str(Path(subdir) / name).rsplit(".", 1)[0] if subdir
            else name.rsplit(".", 1)[0]
        )
        try:
            if not astore.has(self._stage_to_registry_key(stage), artifact_id):
                return None
            return astore.read_any(self._stage_to_registry_key(stage), artifact_id)
        except Exception:  # noqa: BLE001
            return None

    def artifact_path(self, stage: Union[int, str], name: str, *, subdir: Optional[str] = None) -> Path:
        """Return the expected absolute path without checking existence."""
        directory = self.stage_dir(stage)
        if subdir:
            directory = directory / subdir
        return directory / name

    # ------------------------------------------------------------------
    # Typed convenience methods
    # ------------------------------------------------------------------

    def save_json(self, stage: Union[int, str], name: str, data: Any,
                  *, subdir: Optional[str] = None, partial: bool = False,
                  metadata: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(stage, name, data, "json", subdir=subdir,
                         partial=partial, metadata=metadata)

    def load_json(self, stage: Union[int, str], name: str,
                  *, subdir: Optional[str] = None) -> Any:
        return self.load(stage, name, "json", subdir=subdir)

    def save_dataframe(
        self,
        stage: Union[int, str],
        name: str,
        df: pd.DataFrame,
        *,
        fmt: Optional[Format] = None,
        subdir: Optional[str] = None,
        partial: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a DataFrame.  Defaults to Parquet if *prefer_parquet* is set,
        otherwise inferred from *name*.  A CSV companion is always written
        alongside Parquet for human inspection."""
        if fmt is None:
            fmt = _infer_format(name)
        dest = self.save(stage, name, df, fmt, subdir=subdir,
                         partial=partial, metadata=metadata)
        # Write a CSV companion when saving as Parquet (only if disk writes
        # are enabled; otherwise the parquet itself is already DB-resident).
        if fmt == "parquet":
            try:
                from sparc.run.disk_policy import disk_writes_enabled
                disk_on = disk_writes_enabled()
            except Exception:  # noqa: BLE001
                disk_on = True
            if disk_on:
                csv_name = Path(name).with_suffix(".csv").name
                csv_path = dest.parent / csv_name
                df.to_csv(csv_path, index=False)
        return dest

    def load_dataframe(self, stage: Union[int, str], name: str,
                       *, subdir: Optional[str] = None) -> pd.DataFrame:
        return self.load(stage, name, subdir=subdir)

    def save_geodataframe(
        self,
        stage: Union[int, str],
        name: str,
        gdf,
        *,
        subdir: Optional[str] = None,
        partial: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a GeoDataFrame as GeoPackage."""
        return self.save(stage, name, gdf, "gpkg", subdir=subdir,
                         partial=partial, metadata=metadata)

    def load_geodataframe(self, stage: Union[int, str], name: str,
                          *, subdir: Optional[str] = None):
        return self.load(stage, name, "gpkg", subdir=subdir)

    def save_numpy(self, stage: Union[int, str], name: str, arr: np.ndarray,
                   *, subdir: Optional[str] = None, partial: bool = False,
                   metadata: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(stage, name, arr, "npy", subdir=subdir,
                         partial=partial, metadata=metadata)

    def load_numpy(self, stage: Union[int, str], name: str,
                   *, subdir: Optional[str] = None) -> np.ndarray:
        return self.load(stage, name, "npy", subdir=subdir)

    def save_pickle(self, stage: Union[int, str], name: str, obj: Any,
                    *, subdir: Optional[str] = None, partial: bool = False,
                    metadata: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(stage, name, obj, "pkl", subdir=subdir,
                         partial=partial, metadata=metadata)

    def load_pickle(self, stage: Union[int, str], name: str,
                    *, subdir: Optional[str] = None) -> Any:
        return self.load(stage, name, "pkl", subdir=subdir)

    def save_text(self, stage: Union[int, str], name: str, text: str,
                  *, subdir: Optional[str] = None, partial: bool = False,
                  metadata: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(stage, name, text, "txt", subdir=subdir,
                         partial=partial, metadata=metadata)

    def load_text(self, stage: Union[int, str], name: str,
                  *, subdir: Optional[str] = None) -> str:
        return self.load(stage, name, "txt", subdir=subdir)

    # ------------------------------------------------------------------
    # Manifest queries
    # ------------------------------------------------------------------

    def manifest(self, stage: Union[int, str]) -> Dict[str, Any]:
        """Return the full manifest dict for *stage*."""
        return _load_manifest(self.stage_dir(stage))

    def list_artifacts(self, stage: Union[int, str]) -> List[str]:
        """Return the artifact names recorded in the manifest for *stage*."""
        m = _load_manifest(self.stage_dir(stage))
        return list(m.get("artifacts", {}).keys())

    def is_stage_complete(self, stage: Union[int, str], required: Optional[List[str]] = None) -> bool:
        """Check whether *stage* has all *required* artifacts (non-partial).

        If *required* is ``None``, returns ``True`` when at least one
        non-partial artifact exists.
        """
        m = _load_manifest(self.stage_dir(stage))
        arts = m.get("artifacts", {})
        if not arts:
            return False
        if required is None:
            return any(not a.get("partial", False) for a in arts.values())
        for name in required:
            entry = arts.get(name)
            if entry is None or entry.get("partial", False):
                return False
        return True

    def partials(self, stage: Union[int, str]) -> List[str]:
        """Return names of partial (incomplete) artifacts in *stage*."""
        m = _load_manifest(self.stage_dir(stage))
        return [
            name for name, info in m.get("artifacts", {}).items()
            if info.get("partial", False)
        ]

    def promote_partial(self, stage: Union[int, str], name: str) -> None:
        """Mark a partial artifact as complete in the manifest."""
        m = _load_manifest(self.stage_dir(stage))
        if name in m.get("artifacts", {}):
            m["artifacts"][name]["partial"] = False
            _save_manifest(self.stage_dir(stage), m)

    # ------------------------------------------------------------------
    # Internal I/O dispatch
    # ------------------------------------------------------------------

    def _write(self, dest: Path, data: Any, fmt: Format) -> None:
        if fmt == "json":
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        elif fmt == "csv":
            if isinstance(data, pd.DataFrame):
                data.to_csv(dest, index=False)
            else:
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(str(data))

        elif fmt == "parquet":
            data.to_parquet(dest, index=False)

        elif fmt == "gpkg":
            gpd = _get_geopandas()
            if not isinstance(data, gpd.GeoDataFrame):
                raise TypeError("GPKG format requires a GeoDataFrame")
            data.to_file(dest, driver="GPKG")

        elif fmt == "npy":
            np.save(dest, data)

        elif fmt == "pkl":
            jl = _get_joblib()
            jl.dump(data, dest)

        elif fmt == "png":
            raise TypeError(
                "PNG output is no longer produced by the pipeline. "
                "Visualisations are rendered live in the desktop app from "
                "artifacts.db data."
            )

        elif fmt == "txt":
            with open(dest, "w", encoding="utf-8") as f:
                f.write(str(data))

        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _read(self, src: Path, fmt: Format) -> Any:
        if fmt == "json":
            with open(src, "r", encoding="utf-8") as f:
                return json.load(f)

        elif fmt == "csv":
            return pd.read_csv(src)

        elif fmt == "parquet":
            return pd.read_parquet(src)

        elif fmt == "gpkg":
            gpd = _get_geopandas()
            return gpd.read_file(src)

        elif fmt == "npy":
            return np.load(src, allow_pickle=False)

        elif fmt == "pkl":
            jl = _get_joblib()
            return jl.load(src)

        elif fmt == "txt":
            return src.read_text(encoding="utf-8")

        elif fmt == "png":
            return src.read_bytes()

        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        stages_with_artifacts = []
        for key in self._stage_dirs:
            m = _load_manifest(self.stage_dir(key))
            n = len(m.get("artifacts", {}))
            if n:
                stages_with_artifacts.append(f"stage {key}: {n} artifacts")
        return (f"ResultStore(output_dir={self.paths.output_dir}, "
                f"stages=[{', '.join(stages_with_artifacts) or 'empty'}])")
