"""
ArtifactStore: db-resident artifact API.

This is the single entry point pipeline modules should use instead of
``df.to_csv()``, ``json.dump()``, ``fig.savefig()``, ``pickle.dump()``,
``torch.save()``, etc.

Three storage tiers, all anchored on ``(stage, artifact_id)``:

* **Tables** (`write_table` / `read_table`)
  Tabular results land in a SQLite table named
  ``data__<stage>__<artifact_id>``.  GeoDataFrames are supported: the
  geometry column is serialized to WKB BLOB and the CRS stored alongside
  in ``result_structs`` so ``read_table`` can rehydrate a GeoDataFrame.

* **Structs** (`write_struct` / `read_struct`)
  Arbitrary JSON-serializable dicts (config snapshots, summaries,
  metric blobs).  Stored verbatim in ``result_structs``.

* **Blobs** (`write_blob` / `read_blob`)
  Binary payloads (pickles, joblib dumps, torch checkpoints, raw bytes).
  Below ``BLOB_INLINE_THRESHOLD_BYTES`` they live in ``internal_blobs``;
  above that threshold they go to a content-addressed external store at
  ``<output_dir>/blobs/<sha[:2]>/<sha>`` so SQLite stays small.

The store is a thin layer on top of :class:`RunRegistry`:  every write
also registers an :class:`ArtifactEntry` so the registry remains the
queryable index.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import pickle
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from sparc.registry.run_registry import RunRegistry

# Blobs >= this size go to the external CAS dir instead of inlined in SQLite.
BLOB_INLINE_THRESHOLD_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Reserved column name for vector geometry (WKB BLOB).
GEOMETRY_COLUMN: str = "_geometry_wkb"

_log = logging.getLogger(__name__)

# Error message helper for torch unpickle failures under the safe path.
_TORCH_UNTRUSTED_HINT = (
    "torch.load(weights_only=True) refused to deserialize the artifact. "
    "This usually means the checkpoint contains pickled Python objects "
    "(e.g. a full nn.Module instance) rather than a plain state_dict. "
    "If the artifact source is trusted, retry the read with trusted=True. "
    "Otherwise treat the file as untrusted and do not load it."
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_name(stage: str | int, artifact_id: str) -> str:
    """Deterministic SQLite table name for a (stage, artifact_id) pair."""
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(artifact_id))
    return f"data__{stage}__{safe}"


def _struct_metadata_id(artifact_id: str) -> str:
    """Sidecar struct id used to store CRS / geometry-column info for tables."""
    return f"__table_meta__{artifact_id}"


class ArtifactStoreError(RuntimeError):
    """Raised on store-layer failures (e.g. unknown artifact, decode error)."""


class ArtifactStore:
    """High-level db-resident artifact API.

    Bound to a :class:`RunRegistry`; all writes also register an
    :class:`ArtifactEntry` so consumers can ``query()`` the registry for
    metadata.
    """

    def __init__(self, registry: "RunRegistry") -> None:
        self.registry = registry
        # Co-located CAS dir for large blobs.
        self.blobs_dir: Path = registry.output_dir / "blobs"

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def write_table(
        self,
        stage: str | int,
        artifact_id: str,
        df: "pd.DataFrame",
        *,
        geometry_col: Optional[str] = None,
        crs: Optional[str] = None,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        if_exists: str = "replace",
    ) -> str:
        """Persist a DataFrame as a SQLite table; register the artifact.

        ``geometry_col``: name of a geometry column to be serialized to WKB
        (auto-detected for GeoDataFrames). ``crs``: CRS spec string (e.g.
        ``"EPSG:4326"``); auto-extracted from GeoDataFrames.

        Returns the underlying SQLite table name.
        """
        import pandas as pd

        t0 = time.perf_counter()
        table = _table_name(stage, artifact_id)
        df_to_write, geom_col_resolved, crs_resolved = self._prepare_table_frame(
            df, geometry_col=geometry_col, crs=crs,
        )

        with self.registry.sqlite_connection() as conn:
            # Use pandas' to_sql for column-type inference.
            df_to_write.to_sql(table, conn, if_exists=if_exists, index=False)

        # Sidecar metadata for tables that have geometry / non-trivial CRS.
        if geom_col_resolved or crs_resolved:
            self._write_struct_raw(
                stage=stage,
                artifact_id=_struct_metadata_id(artifact_id),
                payload={
                    "geometry_col": geom_col_resolved,
                    "crs": crs_resolved,
                    "wkb_column": GEOMETRY_COLUMN if geom_col_resolved else None,
                },
            )

        meta = dict(metadata or {})
        meta.setdefault("columns", list(df_to_write.columns))
        if geom_col_resolved:
            meta["geometry_col"] = geom_col_resolved
            meta["crs"] = crs_resolved

        self.registry.register_artifact(
            stage=stage,
            artifact_id=artifact_id,
            format="table",
            storage_kind="table",
            data_table=table,
            producer=producer,
            consumers=consumers,
            row_count=len(df_to_write),
            size_bytes=int(df_to_write.memory_usage(deep=True).sum()),
            metadata=meta,
            write_seconds=time.perf_counter() - t0,
            name=artifact_id,
        )
        return table

    def read_table(
        self,
        stage: str | int,
        artifact_id: str,
    ) -> "pd.DataFrame":
        """Load a registered table; returns a GeoDataFrame if geometry present."""
        import pandas as pd

        entry = self.registry.lookup(stage, artifact_id)
        if entry is None:
            raise ArtifactStoreError(f"Unknown artifact: stage={stage} id={artifact_id}")
        if entry.storage_kind != "table":
            raise ArtifactStoreError(
                f"Artifact {artifact_id} is not a table (storage_kind={entry.storage_kind})"
            )
        table = entry.data_table or _table_name(stage, artifact_id)

        with self.registry.sqlite_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)

        # Rehydrate geometry if sidecar metadata declares one.
        meta = self._read_struct_raw(stage, _struct_metadata_id(artifact_id))
        if meta and meta.get("geometry_col") and GEOMETRY_COLUMN in df.columns:
            try:
                import geopandas as gpd
                from shapely import wkb

                geom = df[GEOMETRY_COLUMN].apply(
                    lambda b: wkb.loads(bytes(b)) if b is not None else None
                )
                df = df.drop(columns=[GEOMETRY_COLUMN])
                df = df.rename(columns={})  # noqa: PLW1505 — explicit copy
                gdf = gpd.GeoDataFrame(df, geometry=geom, crs=meta.get("crs"))
                # Restore original geometry column name if it differed.
                orig = meta.get("geometry_col")
                if orig and orig != "geometry":
                    gdf = gdf.rename_geometry(orig)
                return gdf
            except ImportError:
                # No geopandas — leave WKB bytes in place; caller can decode.
                return df
        return df

    # ------------------------------------------------------------------
    # Structs
    # ------------------------------------------------------------------

    def write_struct(
        self,
        stage: str | int,
        artifact_id: str,
        payload: dict[str, Any],
        *,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a JSON-serializable dict; register the artifact."""
        t0 = time.perf_counter()
        encoded = self._write_struct_raw(stage=stage, artifact_id=artifact_id, payload=payload)
        self.registry.register_artifact(
            stage=stage,
            artifact_id=artifact_id,
            format="struct",
            storage_kind="struct",
            producer=producer,
            consumers=consumers,
            size_bytes=len(encoded),
            metadata=metadata,
            write_seconds=time.perf_counter() - t0,
            name=artifact_id,
        )

    def read_struct(self, stage: str | int, artifact_id: str) -> dict[str, Any]:
        entry = self.registry.lookup(stage, artifact_id)
        if entry is None:
            raise ArtifactStoreError(f"Unknown artifact: stage={stage} id={artifact_id}")
        if entry.storage_kind != "struct":
            raise ArtifactStoreError(
                f"Artifact {artifact_id} is not a struct (storage_kind={entry.storage_kind})"
            )
        result = self._read_struct_raw(stage, artifact_id)
        if result is None:
            raise ArtifactStoreError(f"Struct payload missing for {stage}/{artifact_id}")
        return result

    # ------------------------------------------------------------------
    # Blobs
    # ------------------------------------------------------------------

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
        metadata: Optional[dict[str, Any]] = None,
        threshold_bytes: int = BLOB_INLINE_THRESHOLD_BYTES,
    ) -> None:
        """Persist a binary payload.

        ``serializer`` selects encoding:
        - ``"pickle"``  — ``pickle.dumps(obj)``
        - ``"joblib"``  — ``joblib.dump(obj, buf)`` (sklearn estimators, scalers)
        - ``"torch"``   — ``torch.save(obj, buf)`` (model checkpoints, tensors)
        - ``"json"``    — ``json.dumps(obj).encode()``
        - ``"raw"``     — ``obj`` is already ``bytes`` / ``bytearray`` / ``memoryview``

        Small payloads (<threshold) are inlined in ``internal_blobs``; large
        payloads go to ``<output_dir>/blobs/<sha[:2]>/<sha>``.
        """
        t0 = time.perf_counter()
        data = self._serialize(obj, serializer=serializer)
        sha = hashlib.sha256(data).hexdigest()
        size = len(data)

        if size < threshold_bytes:
            blob_id = self._write_inline_blob(
                stage=stage, artifact_id=artifact_id, data=data,
                serializer=serializer, mime=mime, sha=sha,
            )
            self.registry.register_artifact(
                stage=stage,
                artifact_id=artifact_id,
                format="blob",
                storage_kind="blob_inline",
                blob_id=blob_id,
                producer=producer,
                consumers=consumers,
                size_bytes=size,
                metadata={**(metadata or {}), "serializer": serializer, "mime": mime},
                write_seconds=time.perf_counter() - t0,
                name=artifact_id,
            )
        else:
            self._write_external_blob(sha, data)
            rel_path = f"blobs/{sha[:2]}/{sha}"
            self.registry.register_artifact(
                stage=stage,
                artifact_id=artifact_id,
                path=self.registry.output_dir / rel_path,
                format="blob",
                storage_kind="blob_external",
                blob_sha256=sha,
                producer=producer,
                consumers=consumers,
                size_bytes=size,
                metadata={**(metadata or {}), "serializer": serializer, "mime": mime},
                write_seconds=time.perf_counter() - t0,
                name=artifact_id,
            )

    def read_blob(
        self,
        stage: str | int,
        artifact_id: str,
        *,
        trusted: bool = False,
    ) -> Any:
        """Read a blob artifact.

        ``trusted=False`` (default) loads torch checkpoints with
        ``weights_only=True`` (restricted unpickler — safe against arbitrary
        code execution). Set ``trusted=True`` only when the artifact was
        produced by code you control; doing so falls back to full pickle
        deserialization and emits a WARNING for audit.
        """
        entry = self.registry.lookup(stage, artifact_id)
        if entry is None:
            raise ArtifactStoreError(f"Unknown artifact: stage={stage} id={artifact_id}")
        if entry.storage_kind not in ("blob_inline", "blob_external"):
            raise ArtifactStoreError(
                f"Artifact {artifact_id} is not a blob (storage_kind={entry.storage_kind})"
            )
        serializer = (entry.metadata or {}).get("serializer", "pickle")

        if entry.storage_kind == "blob_inline":
            with self.registry.sqlite_connection() as conn:
                row = conn.execute(
                    "SELECT bytes FROM internal_blobs WHERE stage=? AND artifact_id=?",
                    (str(stage), artifact_id),
                ).fetchone()
            if row is None:
                raise ArtifactStoreError(
                    f"Inline blob payload missing for {stage}/{artifact_id}"
                )
            data = bytes(row[0])
        else:  # blob_external
            sha = entry.blob_sha256
            if not sha:
                raise ArtifactStoreError(
                    f"External blob {stage}/{artifact_id} has no sha256"
                )
            blob_path = self.blobs_dir / sha[:2] / sha
            if not blob_path.exists():
                raise ArtifactStoreError(f"External blob file missing: {blob_path}")
            data = blob_path.read_bytes()

        return self._deserialize(
            data,
            serializer=serializer,
            trusted=trusted,
            artifact_key=f"{stage}/{artifact_id}",
        )

    # ------------------------------------------------------------------
    # Unified read (handles all storage_kinds, incl. legacy_path)
    # ------------------------------------------------------------------

    def read_any(
        self,
        stage: str | int,
        artifact_id: str,
        *,
        trusted: bool = False,
    ) -> Any:
        """Read an artifact regardless of where it lives.

        See :meth:`read_blob` for the meaning of ``trusted``.


        Resolves by ``storage_kind``:
        - ``table``        -> :meth:`read_table`
        - ``struct``       -> :meth:`read_struct`
        - ``blob_inline`` / ``blob_external`` -> :meth:`read_blob`
        - ``legacy_path``  -> deserialized from disk based on ``format``
          (json, csv, parquet, gpkg, npy, pkl, joblib, pt, txt, html).

        This is the single read API consumer code should use during the
        migration: it works whether the artifact is db-resident or still
        on disk from a legacy run.
        """
        entry = self.registry.lookup(stage, artifact_id)
        if entry is None:
            raise ArtifactStoreError(
                f"Unknown artifact: stage={stage} id={artifact_id}"
            )

        kind = entry.storage_kind
        if kind == "table":
            return self.read_table(stage, artifact_id)
        if kind == "struct":
            return self.read_struct(stage, artifact_id)
        if kind in ("blob_inline", "blob_external"):
            return self.read_blob(stage, artifact_id, trusted=trusted)
        if kind == "legacy_path":
            return self._read_legacy_path(entry, trusted=trusted)
        raise ArtifactStoreError(
            f"Unsupported storage_kind: {kind}"
        )

    def _read_legacy_path(self, entry: Any, *, trusted: bool = False) -> Any:
        """Best-effort deserialize for on-disk legacy artifacts.

        See :meth:`read_blob` for the meaning of ``trusted``.
        """
        path = self.registry.resolve(entry)
        if not path.exists():
            raise ArtifactStoreError(f"Legacy file missing: {path}")
        fmt = entry.format
        if fmt in ("json", "geojson"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if fmt == "csv":
            import pandas as pd
            return pd.read_csv(path)
        if fmt == "parquet":
            import pandas as pd
            return pd.read_parquet(path)
        if fmt == "gpkg":
            import geopandas as gpd
            return gpd.read_file(path)
        if fmt == "npy":
            import numpy as np
            return np.load(path, allow_pickle=True)
        if fmt == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        if fmt == "joblib":
            import joblib  # type: ignore[import-not-found]
            return joblib.load(path)
        if fmt == "pt":
            import torch  # type: ignore[import-not-found]

            if trusted:
                _log.warning(
                    "Loading torch artifact %s with trusted=True "
                    "(weights_only=False — full pickle deserialization).",
                    path,
                )
                return torch.load(path, weights_only=False)
            try:
                return torch.load(path, weights_only=True)
            except pickle.UnpicklingError as exc:
                raise ArtifactStoreError(
                    f"{_TORCH_UNTRUSTED_HINT} (path={path})"
                ) from exc
        if fmt in ("txt", "html"):
            return path.read_text(encoding="utf-8")
        # Unknown format: return raw bytes.
        return path.read_bytes()

    def has(self, stage: str | int, artifact_id: str) -> bool:
        """Return True if the artifact exists and is readable."""
        entry = self.registry.lookup(stage, artifact_id)
        if entry is None or entry.partial:
            return False
        return self.registry.is_available(stage, artifact_id)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_unreferenced_blobs(self) -> int:
        """Delete external CAS files no longer referenced by any artifact.

        Returns the number of files removed.
        """
        if not self.blobs_dir.exists():
            return 0
        referenced: set[str] = set()
        with self.registry.sqlite_connection() as conn:
            for (sha,) in conn.execute(
                "SELECT blob_sha256 FROM artifacts WHERE blob_sha256 IS NOT NULL"
            ).fetchall():
                if sha:
                    referenced.add(sha)
        removed = 0
        for blob in self.blobs_dir.rglob("*"):
            if blob.is_file() and blob.name not in referenced:
                try:
                    blob.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    # ------------------------------------------------------------------
    # Export-on-demand (download from artifacts.db)
    # ------------------------------------------------------------------

    # Map storage_kind -> default export format when caller passes fmt=None.
    _DEFAULT_EXPORT_FMT: dict[str, str] = {
        "table": "csv",
        "struct": "json",
        "blob_inline": "joblib",
        "blob_external": "joblib",
        "legacy_path": "native",
    }

    def export(
        self,
        stage: str | int,
        artifact_id: str,
        *,
        fmt: Optional[str] = None,
        dest: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Materialize a single artifact to disk and return the path.

        Parameters
        ----------
        stage, artifact_id : artifact key.
        fmt : output format. ``None`` chooses a sensible default based
            on storage kind (``table -> csv``, ``struct -> json``,
            ``blob -> joblib``, ``legacy_path -> native``). Supported:
            tables: ``csv``, ``parquet``, ``gpkg`` (when geometry present),
                    ``geojson``;
            structs: ``json``;
            blobs: ``joblib``, ``pkl``, ``npy`` (when payload is ndarray).
        dest : destination file path. If ``None``, a tempfile is created.

        Returns
        -------
        pathlib.Path
            Path to the written file.
        """
        entry = self.registry.lookup(stage, artifact_id)
        if entry is None:
            raise ArtifactStoreError(
                f"Unknown artifact: stage={stage} id={artifact_id}"
            )
        kind = entry.storage_kind or "legacy_path"
        chosen_fmt = (fmt or self._DEFAULT_EXPORT_FMT.get(kind, "json")).lower()

        # Resolve destination — create tempfile if not given.
        if dest is None:
            import tempfile
            suffix = "." + chosen_fmt if chosen_fmt != "native" else ""
            tmp = tempfile.NamedTemporaryFile(
                prefix=f"{stage}_{artifact_id}_", suffix=suffix, delete=False,
            )
            tmp.close()
            dest_path = Path(tmp.name)
        else:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Legacy path: if user asked for native fmt, just copy.
        if kind == "legacy_path":
            src = self.registry.resolve(entry)
            if not src.exists():
                raise ArtifactStoreError(f"Legacy file missing: {src}")
            if chosen_fmt in ("native", entry.format):
                import shutil
                shutil.copyfile(src, dest_path)
                return dest_path
            # Otherwise, fall through and re-serialize via read_any.

        payload = self.read_any(stage, artifact_id)

        if kind == "table" or chosen_fmt in ("csv", "parquet", "gpkg", "geojson"):
            import pandas as pd
            df = payload
            if not hasattr(df, "to_csv"):
                raise ArtifactStoreError(
                    f"Cannot export non-table payload as {chosen_fmt}"
                )
            if chosen_fmt == "csv":
                df.to_csv(dest_path, index=False)
            elif chosen_fmt == "parquet":
                df.to_parquet(dest_path, index=False)
            elif chosen_fmt in ("gpkg", "geojson"):
                try:
                    import geopandas as gpd
                except ImportError as e:
                    raise ArtifactStoreError(
                        f"geopandas required for {chosen_fmt} export"
                    ) from e
                if not isinstance(df, gpd.GeoDataFrame):
                    raise ArtifactStoreError(
                        f"Artifact has no geometry column; cannot export as {chosen_fmt}"
                    )
                driver = "GPKG" if chosen_fmt == "gpkg" else "GeoJSON"
                df.to_file(dest_path, driver=driver)
            return dest_path

        if kind == "struct" or chosen_fmt == "json":
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            return dest_path

        # Blob: choose serializer.
        if chosen_fmt == "joblib":
            import joblib  # type: ignore[import-not-found]
            joblib.dump(payload, dest_path)
            return dest_path
        if chosen_fmt == "pkl":
            with open(dest_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            return dest_path
        if chosen_fmt == "npy":
            try:
                import numpy as np
            except ImportError as e:
                raise ArtifactStoreError("numpy required for .npy export") from e
            np.save(dest_path, payload)
            return dest_path

        raise ArtifactStoreError(
            f"Unsupported export fmt={chosen_fmt!r} for storage_kind={kind!r}"
        )

    def export_stage(
        self,
        stage: str | int,
        dest_dir: Union[str, Path],
        *,
        as_zip: bool = False,
    ) -> list[Path] | Path:
        """Dump every artifact for ``stage`` into ``dest_dir``.

        Parameters
        ----------
        stage : stage label.
        dest_dir : directory to write files into (will be created).
        as_zip : if True, instead returns a single .zip at
            ``dest_dir.with_suffix('.zip')`` containing all files.

        Returns
        -------
        list[pathlib.Path] | pathlib.Path
            List of written files, or path to the zip when ``as_zip`` is True.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for entry in self.registry.list_for_stage(stage):
            if entry.partial:
                continue
            try:
                # Pick a name + sensible extension.
                kind = entry.storage_kind or "legacy_path"
                ext = self._DEFAULT_EXPORT_FMT.get(kind, "bin")
                if ext == "native" and entry.format:
                    ext = entry.format
                name = f"{entry.id}.{ext}"
                target = dest_dir / name
                self.export(stage, entry.id, dest=target)
                written.append(target)
            except Exception as exc:  # noqa: BLE001
                # Don't let one bad artifact abort the whole stage export.
                warnings.warn(
                    f"export_stage: failed to export {entry.id}: {exc}",
                    RuntimeWarning,
                )
        if as_zip:
            import zipfile
            zip_path = dest_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in written:
                    zf.write(p, arcname=p.name)
            return zip_path
        return written

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_table_frame(
        self,
        df: "pd.DataFrame",
        *,
        geometry_col: Optional[str],
        crs: Optional[str],
    ) -> tuple["pd.DataFrame", Optional[str], Optional[str]]:
        """Return (frame_for_sqlite, resolved_geometry_col, resolved_crs)."""
        # Detect GeoDataFrame.
        try:
            import geopandas as gpd
        except ImportError:
            gpd = None  # type: ignore[assignment]

        resolved_geom = geometry_col
        resolved_crs = crs

        if gpd is not None and isinstance(df, gpd.GeoDataFrame):
            if resolved_geom is None:
                resolved_geom = df.geometry.name
            if resolved_crs is None and df.crs is not None:
                resolved_crs = df.crs.to_string() if hasattr(df.crs, "to_string") else str(df.crs)

        if resolved_geom and resolved_geom in df.columns:
            # Convert geometry column to WKB BLOB column.
            try:
                from shapely.geometry.base import BaseGeometry
            except ImportError:
                BaseGeometry = object  # type: ignore[misc,assignment]

            def _to_wkb(g: Any) -> Optional[bytes]:
                if g is None:
                    return None
                if isinstance(g, (bytes, bytearray)):
                    return bytes(g)
                if isinstance(g, BaseGeometry):
                    return g.wkb
                # Fall back to str repr; not ideal but avoids crashes.
                return str(g).encode("utf-8")

            wkb_series = df[resolved_geom].map(_to_wkb)
            out = df.drop(columns=[resolved_geom]).copy()
            out[GEOMETRY_COLUMN] = wkb_series
            return out, resolved_geom, resolved_crs

        return df, None, resolved_crs

    def _write_struct_raw(
        self,
        *,
        stage: str | int,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> bytes:
        encoded = json.dumps(payload, default=str).encode("utf-8")
        with self.registry.sqlite_connection() as conn:
            conn.execute(
                "INSERT INTO result_structs (stage, artifact_id, json, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(stage, artifact_id) DO UPDATE SET "
                "json=excluded.json, created_at=excluded.created_at",
                (str(stage), artifact_id, encoded.decode("utf-8"), _utcnow()),
            )
        return encoded

    def _read_struct_raw(
        self, stage: str | int, artifact_id: str,
    ) -> Optional[dict[str, Any]]:
        with self.registry.sqlite_connection() as conn:
            row = conn.execute(
                "SELECT json FROM result_structs WHERE stage=? AND artifact_id=?",
                (str(stage), artifact_id),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def _write_inline_blob(
        self,
        *,
        stage: str | int,
        artifact_id: str,
        data: bytes,
        serializer: str,
        mime: Optional[str],
        sha: str,
    ) -> int:
        with self.registry.sqlite_connection() as conn:
            cur = conn.execute(
                "INSERT INTO internal_blobs "
                "(stage, artifact_id, mime, serializer, sha256, size_bytes, "
                " bytes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(stage, artifact_id) DO UPDATE SET "
                "mime=excluded.mime, serializer=excluded.serializer, "
                "sha256=excluded.sha256, size_bytes=excluded.size_bytes, "
                "bytes=excluded.bytes, created_at=excluded.created_at",
                (str(stage), artifact_id, mime, serializer, sha, len(data),
                 sqlite3.Binary(data), _utcnow()),
            )
            blob_id = cur.lastrowid
            if not blob_id:
                row = conn.execute(
                    "SELECT id FROM internal_blobs WHERE stage=? AND artifact_id=?",
                    (str(stage), artifact_id),
                ).fetchone()
                blob_id = int(row[0]) if row else 0
        return int(blob_id)

    def _write_external_blob(self, sha: str, data: bytes) -> Path:
        target_dir = self.blobs_dir / sha[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / sha
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        return target

    @staticmethod
    def _serialize(obj: Any, *, serializer: str) -> bytes:
        if serializer == "pickle":
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if serializer == "joblib":
            import joblib  # type: ignore[import-not-found]

            buf = io.BytesIO()
            joblib.dump(obj, buf)
            return buf.getvalue()
        if serializer == "torch":
            import torch  # type: ignore[import-not-found]

            buf = io.BytesIO()
            torch.save(obj, buf)
            return buf.getvalue()
        if serializer == "json":
            return json.dumps(obj, default=str).encode("utf-8")
        if serializer == "raw":
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return bytes(obj)
            raise ArtifactStoreError(
                f"serializer='raw' requires bytes-like, got {type(obj).__name__}"
            )
        raise ArtifactStoreError(f"Unknown serializer: {serializer!r}")

    @staticmethod
    def _deserialize(
        data: bytes,
        *,
        serializer: str,
        trusted: bool = False,
        artifact_key: str | None = None,
    ) -> Any:
        if serializer == "pickle":
            return pickle.loads(data)
        if serializer == "joblib":
            import joblib  # type: ignore[import-not-found]

            return joblib.load(io.BytesIO(data))
        if serializer == "torch":
            import torch  # type: ignore[import-not-found]

            if trusted:
                _log.warning(
                    "Loading torch artifact %s with trusted=True "
                    "(weights_only=False \u2014 full pickle deserialization).",
                    artifact_key or "<unknown>",
                )
                return torch.load(io.BytesIO(data), weights_only=False)
            try:
                return torch.load(io.BytesIO(data), weights_only=True)
            except pickle.UnpicklingError as exc:
                key = artifact_key or "<unknown>"
                raise ArtifactStoreError(
                    f"{_TORCH_UNTRUSTED_HINT} (artifact={key})"
                ) from exc
        if serializer == "json":
            return json.loads(data.decode("utf-8"))
        if serializer == "raw":
            return data
        raise ArtifactStoreError(f"Unknown serializer: {serializer!r}")


# ---------------------------------------------------------------------------
# Module-level helpers — convenience for writers using the active registry.
# ---------------------------------------------------------------------------

def get_active_store() -> Optional[ArtifactStore]:
    """Return an ArtifactStore bound to the active registry, if any."""
    from sparc.registry.run_registry import get_active_registry

    reg = get_active_registry()
    if reg is None:
        return None
    return ArtifactStore(reg)


__all__ = [
    "ArtifactStore",
    "ArtifactStoreError",
    "BLOB_INLINE_THRESHOLD_BYTES",
    "GEOMETRY_COLUMN",
    "get_active_store",
]
