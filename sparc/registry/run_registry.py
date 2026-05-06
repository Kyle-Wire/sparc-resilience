"""
RunRegistry: single-source-of-truth artifact tracker for a SPARC pipeline run.

Storage model
-------------
* Canonical: ``output_dir/artifacts_manifest.json`` (Pydantic ``RunManifest``)
* Cache:      ``output_dir/artifacts.db`` (SQLite mirror, fully rebuildable)

Both files are kept in sync inside ``register_artifact`` / ``complete_stage``;
either can be deleted and rebuilt from the other (``rebuild_cache()`` /
``rebuild_from_cache()``).

Concurrency
-----------
A simple lock-file (``artifacts_manifest.json.lock``) guards JSON writes so
multiple stages in the same run can register concurrently.  SQLite has its
own WAL-backed locking.

Migration
---------
Legacy runs (no manifest, just per-stage ``_manifest.json`` files or raw
artifacts on disk) can be imported via :meth:`migrate_from_disk`, which
walks the run directory and registers everything matching a known catalog.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from sparc.registry.manifest import (
    ArtifactEntry,
    ArtifactFormat,
    RunManifest,
    StageManifest,
    StageStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "artifacts_manifest.json"
SQLITE_FILENAME = "artifacts.db"
LOCK_FILENAME = "artifacts_manifest.json.lock"

# Catalog of artifact ids → expected (stage, relative path, format) used by the
# disk-scan migration.  The migration is best-effort; unknown files are
# ignored.  ``{stage}`` placeholders are resolved against PipelinePaths.

_KNOWN_CATALOG: list[dict[str, Any]] = [
    # Stage 0
    {"id": "correlogram_results", "stage": "0",
     "patterns": ["correlogram_analysis_results.json", "correlogram_results.json"],
     "format": "json", "consumers": ["server:/results/correlogram"]},
    {"id": "correlogram_summary", "stage": "0",
     "patterns": ["correlogram_summary.csv"], "format": "csv"},
    {"id": "dataset_profile", "stage": "0",
     "patterns": ["dataset_profile.json"], "format": "json"},

    # Stage 1
    {"id": "gwen_results", "stage": "1",
     "patterns": ["gwen_results.json"], "format": "json",
     "consumers": ["server:/results/gwen"]},
    {"id": "gwen_variable_importance", "stage": "1",
     "patterns": ["gwen_variable_importance.csv"], "format": "csv"},
    {"id": "gwen_diagnostics", "stage": "1",
     "patterns": ["gwen_diagnostics.json"], "format": "json"},
    {"id": "selected_features", "stage": "1",
     "patterns": ["selected_features.txt"], "format": "txt"},

    # Stage 2
    {"id": "ensemble_results", "stage": "2",
     "patterns": ["final_ensemble_results.json"], "format": "json",
     "consumers": ["server:/results/model_performance"]},
    {"id": "ensemble_predictions", "stage": "2",
     "patterns": ["final_ensemble_predictions.csv"], "format": "csv"},
    {"id": "oof_predictions", "stage": "2",
     "patterns": ["optimized_oof_predictions.csv"], "format": "csv"},
    {"id": "spatial_cv_predictions", "stage": "2",
     "patterns": ["spatial_cv_predictions.gpkg"], "format": "gpkg",
     "consumers": ["server:/results/2/predictions",
                   "server:/results/spatial_cv/predictions"]},
    {"id": "feature_info", "stage": "2",
     "patterns": ["feature_info.json"], "format": "json"},
    {"id": "feature_scaler", "stage": "2",
     "patterns": ["feature_scaler.pkl"], "format": "pkl"},
    {"id": "neural_meta_weights", "stage": "2",
     "patterns": ["v2_neural/neural_meta.pt"], "format": "pt"},
    {"id": "neural_meta_info", "stage": "2",
     "patterns": ["v2_neural/meta_info.json"], "format": "json"},
    {"id": "neural_predictions", "stage": "2",
     "patterns": ["v2_neural/predictions.csv"], "format": "csv",
     "consumers": ["server:/results/2/predictions"]},
    {"id": "neural_local_coefficients", "stage": "2",
     "patterns": ["v2_neural/local_coefficients_gwr.csv"], "format": "csv"},
    {"id": "neural_process_rate_map", "stage": "2",
     "patterns": ["v2_neural/process_rate_map.csv"], "format": "csv"},
    {"id": "neural_attention_summary", "stage": "2",
     "patterns": ["v2_neural/spatial_attention_summary.csv"], "format": "csv"},
    {"id": "neural_feature_importance", "stage": "2",
     "patterns": ["v2_neural/feature_importance.csv"], "format": "csv"},
    {"id": "neural_alpha_field", "stage": "2",
     "patterns": ["v2_neural/alpha_field.npy"], "format": "npy"},
    {"id": "neural_process_rate_net", "stage": "2",
     "patterns": ["v2_neural/process_rate_net.pt"], "format": "pt"},
    {"id": "neural_source_term_net", "stage": "2",
     "patterns": ["v2_neural/source_term_net.pt"], "format": "pt"},
    {"id": "sinusoidal_encoder", "stage": "2",
     "patterns": ["v2_neural/sinusoidal_encoder.pkl"], "format": "pkl"},
    {"id": "gwrf_pdp_curves", "stage": "2",
     "patterns": ["gwrf_pdp/gwrf_condition_curves.json"], "format": "json",
     "consumers": ["server:/results/pdp_curves"]},
    {"id": "gwrf_models", "stage": "2",
     "patterns": ["gwrf_models.joblib", "gwrf/local_models.joblib"],
     "format": "joblib"},
    {"id": "gwr_constrained_coeffs", "stage": "2",
     "patterns": ["gwr_constrained_coefficients.csv"], "format": "csv"},
    {"id": "gwr_unconstrained_coeffs", "stage": "2",
     "patterns": ["gwr_unconstrained_coefficients.csv"], "format": "csv"},
    {"id": "gwr_modifier_csv", "stage": "2",
     "patterns": ["gwr_modifiers.csv"], "format": "csv"},
    {"id": "ggpgam_uncertainty", "stage": "2",
     "patterns": ["ggpgam_uncertainty.csv"], "format": "csv"},
    {"id": "ggpgam_uncertainty_map", "stage": "2",
     "patterns": ["ggpgam_uncertainty_map.gpkg"], "format": "gpkg"},
    {"id": "final_predictions", "stage": "2",
     "patterns": ["final_predictions.csv"], "format": "csv",
     "consumers": ["server:/results/2/predictions"]},
    {"id": "sensitivity_package", "stage": "2",
     "patterns": ["sensitivity_package.json"], "format": "json"},
    # Spatial folds + retrained full base models (consumed by scenario_simulator
    # and the Stage 2 resume path).
    {"id": "folds", "stage": "2",
     "patterns": ["folds.pkl"], "format": "pkl"},
    {"id": "ols_model_full", "stage": "2",
     "patterns": ["base_models_full/ols_model_full.pkl"], "format": "pkl"},
    {"id": "gwr_model_full", "stage": "2",
     "patterns": ["base_models_full/gwr_model_full.pkl"], "format": "pkl"},
    {"id": "gwrf_model_full", "stage": "2",
     "patterns": ["base_models_full/gwrf_model_full.pkl"], "format": "pkl"},
    {"id": "ggpgam_model_full", "stage": "2",
     "patterns": ["base_models_full/ggpgam_model_full.pkl"], "format": "pkl"},
    {"id": "standard_meta_ensemble", "stage": "2",
     "patterns": ["standard_meta_ensemble.pkl"], "format": "pkl"},
    {"id": "pipeline_config", "stage": "2",
     "patterns": ["pipeline_config.json"], "format": "json"},

    # Stage 3
    {"id": "scenario_coefficients", "stage": "3",
     "patterns": ["scenario_coefficients.json"], "format": "json",
     "consumers": ["server:/results/causal"]},
    {"id": "dose_response_curves", "stage": "3",
     "patterns": ["dose_response_curves.json"], "format": "json",
     "consumers": ["server:/results/causal/dose_response"]},
    {"id": "spatial_cate_maps", "stage": "3",
     "patterns": ["spatial_cate_maps.gpkg"], "format": "gpkg",
     "consumers": ["server:/results/causal/cate_map"]},
    {"id": "propensity_diagnostics", "stage": "3",
     "patterns": ["propensity_diagnostics.json"], "format": "json"},
    {"id": "dag_discovery_report", "stage": "3",
     "patterns": ["dag_discovery_report.json"], "format": "json"},
    {"id": "edge_models", "stage": "3",
     "patterns": ["edge_models.joblib"], "format": "joblib"},
    {"id": "causal_diagnostics", "stage": "3",
     "patterns": ["causal_diagnostics.json"], "format": "json"},
    {"id": "nuts_summary", "stage": "3",
     "patterns": ["bayesian/nuts_summary.json", "nuts_summary.json"],
     "format": "json", "consumers": ["server:/results/scenarios/nuts_summary"]},
    {"id": "mc3_summary", "stage": "3",
     "patterns": ["bayesian/mc3_summary.json"], "format": "json"},
    {"id": "parameter_posteriors", "stage": "3",
     "patterns": ["bayesian/parameter_posteriors.csv"], "format": "csv",
     "consumers": ["server:/results/scenarios/nuts_summary"]},
    {"id": "convergence_diagnostics", "stage": "3",
     "patterns": ["bayesian/convergence_diagnostics.csv"], "format": "csv",
     "consumers": ["server:/results/scenarios/nuts_summary"]},
    {"id": "bma_coefficients", "stage": "3",
     "patterns": ["bayesian/bma_coefficients.csv"], "format": "csv",
     "consumers": ["server:/results/scenarios/nuts_summary"]},
    {"id": "edge_inclusion_probs", "stage": "3",
     "patterns": ["bayesian/edge_inclusion_probs.csv",
                  "bayesian/posterior_edge_probs.csv"], "format": "csv"},
    {"id": "edge_inclusion_probs_npy", "stage": "3",
     "patterns": ["bayesian/edge_inclusion_probs.npy"], "format": "npy"},
    {"id": "edge_confidence", "stage": "3",
     "patterns": ["bayesian/edge_confidence.csv"], "format": "csv"},
    {"id": "median_dag", "stage": "3",
     "patterns": ["bayesian/median_dag.json"], "format": "json"},
    # nuts_beta_chain (single-edge legacy back-compat) removed in SPARC v4
    # — superseded by ("3","nuts_edge_summary")+("3","nuts_edge_samples").
    {"id": "causal_validation_summary", "stage": "3",
     "patterns": ["causal_validation_summary.txt"], "format": "txt"},

    # Stage 4
    {"id": "scenario_results", "stage": "4",
     "patterns": ["scenario_results.gpkg", "scenario_results_dag.gpkg",
                  "scenario_results_hybrid.gpkg",
                  "scenario_results_reprediction.gpkg"],
     "format": "gpkg",
     "consumers": ["server:/results/scenarios/detail"]},
    {"id": "scenario_summary", "stage": "4",
     "patterns": ["scenario_summary.csv", "scenario_summary_dag.csv",
                  "scenario_summary_hybrid.csv",
                  "scenario_summary_reprediction.csv",
                  "scenario_mc_consensus_summary.csv"],
     "format": "csv",
     "consumers": ["server:/results/scenarios/variables"]},
    {"id": "scenario_mc_consensus", "stage": "4",
     "patterns": ["scenario_mc_consensus.csv"], "format": "csv"},
]


# ---------------------------------------------------------------------------
# File lock (cross-platform best-effort)
# ---------------------------------------------------------------------------

class _FileLock:
    """Tiny cross-platform file lock used for JSON writes.

    Falls back to a per-process threading.Lock when no other process is
    expected to write (the common case during a single pipeline run).
    """

    _process_locks: dict[str, threading.Lock] = {}

    def __init__(self, path: Path):
        self.path = path
        self._tlock = self._process_locks.setdefault(str(path), threading.Lock())
        self._fh = None

    def __enter__(self) -> "_FileLock":
        self._tlock.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch a sentinel file so external observers can see writes are in flight.
        self._fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                # Lock 1 byte; blocks (LK_LOCK) until available
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass  # best-effort
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._fh is not None:
                if os.name == "nt":
                    try:
                        import msvcrt
                        self._fh.seek(0)
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    try:
                        import fcntl
                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                self._fh.close()
                self._fh = None
        finally:
            self._tlock.release()


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------

class RunRegistry:
    """Run-wide artifact registry backed by JSON canonical + SQLite cache."""

    def __init__(self, output_dir: str | Path, *, autoload: bool = True):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / MANIFEST_FILENAME
        self.sqlite_path = self.output_dir / SQLITE_FILENAME
        self.lock_path = self.output_dir / LOCK_FILENAME
        self._manifest: RunManifest = RunManifest()
        # Listeners notified after every successful register_artifact.
        # Signature: ``listener(entry: ArtifactEntry) -> None``.
        # Listener exceptions are caught and logged so a single bad
        # subscriber cannot break the write path.
        self._on_register_listeners: list[Any] = []
        if autoload:
            self.load()
        self._ensure_sqlite_schema()

    # ------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------

    def add_register_listener(self, listener) -> None:  # type: ignore[no-untyped-def]
        """Subscribe ``listener(entry)`` to artifact registration events."""
        if listener not in self._on_register_listeners:
            self._on_register_listeners.append(listener)

    def remove_register_listener(self, listener) -> None:  # type: ignore[no-untyped-def]
        try:
            self._on_register_listeners.remove(listener)
        except ValueError:
            pass

    def _emit_registered(self, entry: ArtifactEntry) -> None:
        for listener in list(self._on_register_listeners):
            try:
                listener(entry)
            except Exception as exc:  # noqa: BLE001
                # Listeners must never break a write.
                print(f"[RunRegistry] register listener {listener!r} failed: {exc}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self) -> RunManifest:
        """Load the canonical JSON manifest from disk (if present)."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._manifest = RunManifest.model_validate(raw)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                # Corrupt manifest — back it up and start fresh.
                backup = self.manifest_path.with_suffix(
                    f".corrupt-{int(time.time())}.json"
                )
                try:
                    self.manifest_path.replace(backup)
                except OSError:
                    pass
                self._manifest = RunManifest()
                self._manifest.metadata = {  # type: ignore[attr-defined]
                    "load_error": str(exc),
                    "backup": str(backup),
                }
        else:
            self._manifest = RunManifest()
        return self._manifest

    def _save_manifest(self) -> None:
        self._manifest.touch()
        # Atomic write: temp + replace.
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                self._manifest.model_dump(mode="json", exclude_none=False),
                f,
                indent=2,
                default=str,
            )
        os.replace(tmp, self.manifest_path)

    def save(self) -> None:
        """Persist the in-memory manifest to JSON + SQLite."""
        with _FileLock(self.lock_path):
            self._save_manifest()
            self._sync_sqlite()

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _sqlite(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextlib.contextmanager
    def sqlite_connection(self) -> Iterator[sqlite3.Connection]:
        """Public access to the registry's SQLite connection for ArtifactStore."""
        with self._sqlite() as conn:
            yield conn

    def _ensure_sqlite_schema(self) -> None:
        with self._sqlite() as conn:
            # Phase 1: base tables (no references to columns added by migration).
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stages (
                    stage TEXT PRIMARY KEY,
                    status TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_seconds REAL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    format TEXT NOT NULL,
                    storage_kind TEXT NOT NULL DEFAULT 'legacy_path',
                    data_table TEXT,
                    blob_id INTEGER,
                    blob_sha256 TEXT,
                    producer TEXT,
                    consumers TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    sha256 TEXT,
                    row_count INTEGER,
                    written_at TEXT,
                    write_seconds REAL,
                    partial INTEGER DEFAULT 0,
                    metadata TEXT,
                    PRIMARY KEY (stage, id)
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_format
                    ON artifacts(format);
                CREATE INDEX IF NOT EXISTS idx_artifacts_consumers
                    ON artifacts(consumers);

                -- Inline binary payloads (<10 MB by default). Larger blobs
                -- live in the content-addressed external store referenced by
                -- artifacts.blob_sha256.
                CREATE TABLE IF NOT EXISTS internal_blobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    mime TEXT,
                    serializer TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    bytes BLOB NOT NULL,
                    created_at TEXT,
                    UNIQUE (stage, artifact_id)
                );

                -- Non-tabular structured results (dicts / JSON).
                CREATE TABLE IF NOT EXISTS result_structs (
                    stage TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    json TEXT NOT NULL,
                    created_at TEXT,
                    PRIMARY KEY (stage, artifact_id)
                );
                """
            )
            # Phase 2: idempotent column additions for upgrades from v1 schema.
            # Must run BEFORE any index that references the migrated columns.
            self._migrate_artifact_columns(conn)
            # Phase 3: indexes that depend on migrated columns.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_storage_kind "
                "ON artifacts(storage_kind)"
            )

    def _migrate_artifact_columns(self, conn: sqlite3.Connection) -> None:
        """Add new columns to artifacts table if upgrading from older schema."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(artifacts)").fetchall()}
        for col, ddl in (
            ("storage_kind", "ALTER TABLE artifacts ADD COLUMN storage_kind TEXT NOT NULL DEFAULT 'legacy_path'"),
            ("data_table",   "ALTER TABLE artifacts ADD COLUMN data_table TEXT"),
            ("blob_id",      "ALTER TABLE artifacts ADD COLUMN blob_id INTEGER"),
            ("blob_sha256",  "ALTER TABLE artifacts ADD COLUMN blob_sha256 TEXT"),
        ):
            if col not in cols:
                conn.execute(ddl)

    def _sync_sqlite(self) -> None:
        """Wipe and rewrite SQLite from in-memory manifest.

        Note: ``internal_blobs`` and ``result_structs`` are NOT wiped here
        — they are the canonical store for db-resident artifacts and are
        managed directly by ``ArtifactStore`` writes.
        """
        with self._sqlite() as conn:
            conn.execute("DELETE FROM stages")
            conn.execute("DELETE FROM artifacts")
            for stage_id, sm in self._manifest.stages.items():
                conn.execute(
                    "INSERT INTO stages "
                    "(stage, status, started_at, finished_at, "
                    " duration_seconds, error) VALUES (?, ?, ?, ?, ?, ?)",
                    (stage_id, sm.status, sm.started_at, sm.finished_at,
                     sm.duration_seconds, sm.error),
                )
                for art in sm.artifacts.values():
                    conn.execute(
                        "INSERT INTO artifacts "
                        "(id, stage, name, path, format, storage_kind, "
                        " data_table, blob_id, blob_sha256, producer, "
                        " consumers, size_bytes, sha256, row_count, "
                        " written_at, write_seconds, partial, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            art.id, stage_id, art.name, art.path, art.format,
                            getattr(art, "storage_kind", "legacy_path"),
                            getattr(art, "data_table", None),
                            getattr(art, "blob_id", None),
                            getattr(art, "blob_sha256", None),
                            art.producer,
                            json.dumps(list(art.consumers)),
                            int(art.size_bytes or 0),
                            art.sha256, art.row_count,
                            art.written_at, art.write_seconds,
                            1 if art.partial else 0,
                            json.dumps(art.metadata or {}, default=str),
                        ),
                    )

    def rebuild_cache(self) -> None:
        """Recreate the SQLite cache from the JSON manifest."""
        try:
            self.sqlite_path.unlink()
        except FileNotFoundError:
            pass
        self._ensure_sqlite_schema()
        with _FileLock(self.lock_path):
            self._sync_sqlite()

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def start_stage(self, stage: str | int) -> StageManifest:
        sm = self._manifest.get_stage(str(stage))
        sm.status = "running"
        sm.started_at = datetime.now(timezone.utc).isoformat()
        sm.error = None
        self.save()
        return sm

    def complete_stage(
        self,
        stage: str | int,
        *,
        status: StageStatus = "complete",
        error: Optional[str] = None,
    ) -> StageManifest:
        sm = self._manifest.get_stage(str(stage))
        sm.status = status
        sm.error = error
        sm.finished_at = datetime.now(timezone.utc).isoformat()
        if sm.started_at:
            try:
                t0 = datetime.fromisoformat(sm.started_at)
                t1 = datetime.fromisoformat(sm.finished_at)
                sm.duration_seconds = (t1 - t0).total_seconds()
            except ValueError:
                pass
        self.save()
        return sm

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        *,
        stage: str | int,
        artifact_id: str,
        path: str | Path | None = None,
        format: ArtifactFormat,
        storage_kind: str = "legacy_path",
        data_table: Optional[str] = None,
        blob_id: Optional[int] = None,
        blob_sha256: Optional[str] = None,
        producer: Optional[str] = None,
        consumers: Optional[Iterable[str]] = None,
        row_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
        partial: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        compute_hash: bool = False,
        write_seconds: Optional[float] = None,
        name: Optional[str] = None,
    ) -> ArtifactEntry:
        """Register (or update) an artifact in the manifest.

        For ``storage_kind='legacy_path'`` (default) ``path`` must be set and
        will be stored relative to ``output_dir``. For db-resident artifacts
        (``table``/``struct``/``blob_inline``/``blob_external``) ``path`` may
        be omitted; size/row-count come from the caller.
        """
        rel_path = ""
        art_name = name or artifact_id
        size = size_bytes or 0
        sha = None

        if path is not None:
            abs_path = Path(path)
            if not abs_path.is_absolute():
                abs_path = (self.output_dir / abs_path).resolve()
            try:
                rel_path = abs_path.relative_to(self.output_dir).as_posix()
            except ValueError:
                rel_path = abs_path.as_posix()  # outside output_dir; absolute
            if size_bytes is None:
                size = abs_path.stat().st_size if abs_path.exists() else 0
            sha = _sha256_file(abs_path) if (compute_hash and abs_path.exists()) else None
            if name is None:
                art_name = abs_path.name

        entry = ArtifactEntry(
            id=artifact_id,
            stage=str(stage),
            name=art_name,
            path=rel_path,
            format=format,
            storage_kind=storage_kind,  # type: ignore[arg-type]
            data_table=data_table,
            blob_id=blob_id,
            blob_sha256=blob_sha256,
            producer=producer,
            consumers=list(consumers or []),
            size_bytes=size,
            sha256=sha,
            row_count=row_count,
            partial=partial,
            metadata=dict(metadata or {}),
            write_seconds=write_seconds,
        )

        sm = self._manifest.get_stage(str(stage))
        sm.artifacts[artifact_id] = entry
        self.save()
        self._emit_registered(entry)
        return entry

    def mark_complete(self, stage: str | int, artifact_id: str) -> None:
        """Flip a previously-partial artifact to ``partial=False``."""
        sm = self._manifest.get_stage(str(stage), create=False)
        art = sm.artifacts.get(artifact_id)
        if art is None:
            return
        art.partial = False
        # Refresh size in case file grew between open and close.
        if getattr(art, "storage_kind", "legacy_path") == "legacy_path" and art.path:
            abs_path = self.resolve(art)
            if abs_path.exists():
                art.size_bytes = abs_path.stat().st_size
        self.save()

    def remove_artifact(self, stage: str | int, artifact_id: str) -> None:
        sm = self._manifest.stages.get(str(stage))
        if sm and artifact_id in sm.artifacts:
            del sm.artifacts[artifact_id]
            self.save()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, stage: str | int, artifact_id: str) -> Optional[ArtifactEntry]:
        return self._manifest.lookup(str(stage), artifact_id)

    def resolve(self, entry: ArtifactEntry) -> Path:
        """Return an absolute path for the artifact entry."""
        p = Path(entry.path)
        if p.is_absolute():
            return p
        return (self.output_dir / p).resolve()

    def is_available(self, stage: str | int, artifact_id: str) -> bool:
        entry = self.lookup(stage, artifact_id)
        if entry is None or entry.partial:
            return False
        kind = getattr(entry, "storage_kind", "legacy_path")
        if kind == "legacy_path":
            return self.resolve(entry).exists()
        if kind == "blob_external":
            if not entry.blob_sha256:
                return False
            return (self.output_dir / "blobs" / entry.blob_sha256[:2] / entry.blob_sha256).exists()
        # table / struct / blob_inline live inside artifacts.db
        return self.sqlite_path.exists()

    def list_for_stage(self, stage: str | int) -> list[ArtifactEntry]:
        sm = self._manifest.stages.get(str(stage))
        return list(sm.artifacts.values()) if sm else []

    def query(
        self,
        *,
        stage: Optional[str | int] = None,
        format: Optional[ArtifactFormat] = None,
        consumer: Optional[str] = None,
        partial: Optional[bool] = None,
    ) -> list[ArtifactEntry]:
        """SQLite-backed query over registered artifacts."""
        clauses: list[str] = []
        params: list[Any] = []
        if stage is not None:
            clauses.append("stage = ?")
            params.append(str(stage))
        if format is not None:
            clauses.append("format = ?")
            params.append(format)
        if partial is not None:
            clauses.append("partial = ?")
            params.append(1 if partial else 0)
        if consumer is not None:
            clauses.append("consumers LIKE ?")
            params.append(f'%"{consumer}"%')
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._sqlite() as conn:
            rows = conn.execute(
                f"SELECT id, stage, name, path, format, storage_kind, "
                f"data_table, blob_id, blob_sha256, producer, consumers, "
                f"size_bytes, sha256, row_count, written_at, write_seconds, "
                f"partial, metadata FROM artifacts{where}",
                params,
            ).fetchall()

        out: list[ArtifactEntry] = []
        for row in rows:
            (a_id, a_stage, name, path, fmt, storage_kind, data_table,
             blob_id, blob_sha256, producer, consumers,
             size_bytes, sha256, row_count, written_at, write_seconds,
             partial_flag, metadata) = row
            out.append(ArtifactEntry(
                id=a_id, stage=a_stage, name=name, path=path or "", format=fmt,
                storage_kind=storage_kind or "legacy_path",
                data_table=data_table,
                blob_id=blob_id,
                blob_sha256=blob_sha256,
                producer=producer,
                consumers=json.loads(consumers) if consumers else [],
                size_bytes=size_bytes or 0, sha256=sha256, row_count=row_count,
                written_at=written_at or "", write_seconds=write_seconds,
                partial=bool(partial_flag),
                metadata=json.loads(metadata) if metadata else {},
            ))
        return out

    # ------------------------------------------------------------------
    # Disk-scan migration (legacy runs)
    # ------------------------------------------------------------------

    def migrate_from_disk(
        self,
        paths: Any,
        *,
        force: bool = False,
    ) -> int:
        """Walk the run directory and register any known artifacts.

        ``paths`` is a ``PipelinePaths`` instance.  Returns the number of
        artifacts newly registered.  Existing manifest entries are left
        untouched unless ``force=True``.
        """
        from sparc.run.pipeline_paths import PipelinePaths
        if not isinstance(paths, PipelinePaths):
            raise TypeError(f"Expected PipelinePaths, got {type(paths)}")

        stage_dir_map = {
            "0": paths.stage0_dir,
            "1": paths.stage1_dir,
            "2": paths.stage2_dir,
            "3": paths.stage3_dir,
            "4": paths.stage4_dir,
            "final": paths.final_dir,
            "spatial": paths.spatial_analysis_dir,
        }

        registered = 0
        for entry_def in _KNOWN_CATALOG:
            stage_id = entry_def["stage"]
            base = stage_dir_map.get(stage_id)
            if base is None:
                continue
            existing = self.lookup(stage_id, entry_def["id"])
            if existing and not force:
                continue
            for pattern in entry_def["patterns"]:
                candidate = base / pattern
                if candidate.exists():
                    self.register_artifact(
                        stage=stage_id,
                        artifact_id=entry_def["id"],
                        path=candidate,
                        format=entry_def["format"],
                        producer="migrate_from_disk",
                        consumers=entry_def.get("consumers", []),
                        metadata={"migrated": True},
                    )
                    registered += 1
                    break

        # Glob-based pickups for variable-suffixed CATE multipliers.
        cate_dir = stage_dir_map.get("3")
        if cate_dir and cate_dir.exists():
            for cate_npy in cate_dir.glob("spatial_cate_multiplier_*.npy"):
                var = cate_npy.stem.replace("spatial_cate_multiplier_", "")
                art_id = f"cate_multiplier::{var}"
                if not force and self.lookup("3", art_id):
                    continue
                self.register_artifact(
                    stage="3",
                    artifact_id=art_id,
                    path=cate_npy,
                    format="npy",
                    producer="migrate_from_disk",
                    consumers=["server:/results/causal/cate_map",
                               "server:/results/causal/cate_map/variables"],
                    metadata={"variable": var, "migrated": True},
                )
                registered += 1

        # Per-variable PDP CSVs from neural meta-learner.
        if base := stage_dir_map.get("2"):
            pdp_dir = base / "v2_neural" / "pdp"
            if pdp_dir.exists():
                for pdp_csv in pdp_dir.glob("pdp_*.csv"):
                    var = pdp_csv.stem.replace("pdp_", "")
                    art_id = f"neural_pdp::{var}"
                    if not force and self.lookup("2", art_id):
                        continue
                    self.register_artifact(
                        stage="2",
                        artifact_id=art_id,
                        path=pdp_csv,
                        format="csv",
                        producer="migrate_from_disk",
                        consumers=["server:/results/pdp_curves"],
                        metadata={"variable": var, "source": "neural_pde",
                                  "migrated": True},
                    )
                    registered += 1

        # Per-variable scenario GPKGs + CSVs in Stage 4 (Bayesian MC mode
        # writes one file per intervention).
        s4 = stage_dir_map.get("4")
        if s4 and s4.exists():
            for gpkg in s4.glob("scenario_*.gpkg"):
                art_id = f"scenario_result::{gpkg.stem}"
                if not force and self.lookup("4", art_id):
                    continue
                self.register_artifact(
                    stage="4", artifact_id=art_id, path=gpkg, format="gpkg",
                    producer="migrate_from_disk",
                    consumers=["server:/results/scenarios/detail"],
                    metadata={"migrated": True},
                )
                registered += 1
            for csv in s4.glob("scenario_mc_*.csv"):
                art_id = f"scenario_mc::{csv.stem}"
                if not force and self.lookup("4", art_id):
                    continue
                self.register_artifact(
                    stage="4", artifact_id=art_id, path=csv, format="csv",
                    producer="migrate_from_disk",
                    metadata={"migrated": True},
                )
                registered += 1
            # All figures under Stage 4 (figs/, maps/)
            for png in s4.rglob("*.png"):
                rel = png.relative_to(s4).as_posix()
                art_id = f"figure::stage4::{rel}"
                if not force and self.lookup("4", art_id):
                    continue
                self.register_artifact(
                    stage="4", artifact_id=art_id, path=png, format="png",
                    producer="migrate_from_disk",
                    metadata={"migrated": True, "figure": True},
                )
                registered += 1

        # Figures for stages 0-3 — generic pickup so the manifest surfaces
        # every diagnostic plot the pipeline produced.
        for sid in ("0", "1", "2", "3"):
            base_dir = stage_dir_map.get(sid)
            if not base_dir or not base_dir.exists():
                continue
            for png in base_dir.rglob("*.png"):
                rel = png.relative_to(base_dir).as_posix()
                art_id = f"figure::stage{sid}::{rel}"
                if not force and self.lookup(sid, art_id):
                    continue
                self.register_artifact(
                    stage=sid, artifact_id=art_id, path=png, format="png",
                    producer="migrate_from_disk",
                    metadata={"migrated": True, "figure": True},
                )
                registered += 1

        return registered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Active-registry singleton (Phase D)
#
# Writers across the pipeline (scenario_simulator, gwr, ggpgam, …) drop
# files on disk inside their own modules. Threading a registry parameter
# through every call site is invasive, so we expose a process-wide
# "active" registry. ``__main__.py`` calls :func:`set_active_registry`
# right after constructing the run registry; writers call
# :func:`register_path` which is a no-op when no registry is set
# (e.g. ad-hoc unit tests / library usage).
# ---------------------------------------------------------------------------

_ACTIVE_REGISTRY: Optional["RunRegistry"] = None

_FORMAT_FROM_SUFFIX: dict[str, ArtifactFormat] = {
    ".json": "json",
    ".csv": "csv",
    ".gpkg": "gpkg",
    ".png": "png",
    ".pkl": "pkl",
    ".pt": "pt",
    ".npy": "npy",
    ".txt": "txt",
    ".parquet": "parquet",
    ".html": "html",
    ".pdf": "pdf",
    ".tif": "tif",
    ".tiff": "tif",
}


def set_active_registry(
    registry: Optional["RunRegistry"], *, force: bool = False
) -> None:
    """Install (or clear) the process-wide active registry.

    The active registry is normally installed once per loaded project
    and remains set for the lifetime of that project so concurrent
    pipeline writers can persist artifacts. To prevent stray request
    handlers from accidentally clearing it via ``set_active_registry(None)``
    while a real registry is attached, ``None`` is treated as a no-op
    unless ``force=True`` is passed (used by load-failure / shutdown
    paths that genuinely need to detach).
    """
    global _ACTIVE_REGISTRY
    if registry is None and not force and _ACTIVE_REGISTRY is not None:
        return
    _ACTIVE_REGISTRY = registry


def get_active_registry() -> Optional["RunRegistry"]:
    """Return the currently-installed registry, if any."""
    return _ACTIVE_REGISTRY


def register_path(
    path: str | Path,
    *,
    stage: str | int,
    artifact_id: str,
    format: Optional[ArtifactFormat] = None,
    producer: Optional[str] = None,
    consumers: Optional[Iterable[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    partial: bool = False,
) -> Optional[ArtifactEntry]:
    """Register ``path`` with the active registry (best-effort).

    Returns ``None`` when no active registry is installed; never raises so
    individual writers stay decoupled from registry availability.
    """
    reg = _ACTIVE_REGISTRY
    if reg is None:
        return None
    p = Path(path)
    fmt = format
    if fmt is None:
        fmt = _FORMAT_FROM_SUFFIX.get(p.suffix.lower(), "binary")  # type: ignore[assignment]
    try:
        return reg.register_artifact(
            stage=stage,
            artifact_id=artifact_id,
            path=p,
            format=fmt,  # type: ignore[arg-type]
            producer=producer,
            consumers=consumers,
            metadata=metadata,
            partial=partial,
        )
    except Exception as exc:  # noqa: BLE001
        # Writers must keep working even if the registry barfs.
        print(f"  [registry] register_path({artifact_id}) failed: {exc}")
        return None


__all__ = [
    "RunRegistry",
    "set_active_registry",
    "get_active_registry",
    "register_path",
]

