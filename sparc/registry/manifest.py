"""
Pydantic models for the SPARC Run Registry.

A *Run Registry* is the single source of truth for what artifacts a pipeline
run has produced on disk. Every writer registers its output, every server
endpoint resolves paths through it, every frontend panel queries it to
decide what to render.

The on-disk canonical form is ``output_dir/artifacts_manifest.json``. A
SQLite mirror (``output_dir/artifacts.db``) is maintained for fast queries
but is fully rebuildable from the JSON.

Two data abstractions live here:

* ``ArtifactEntry`` — one materialized file on disk (path + format + bytes).
  Multiple artifacts may belong to one dataset (e.g. a Parquet canonical
  store + a CSV export + a PNG plot).
* ``Dataset`` — the typed contract between stages: a named bundle of data
  with a schema, a stage owner, and a canonical storage format. Stages save
  *datasets*; users request *artifacts* on demand from those datasets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ArtifactFormat = Literal[
    "json", "csv", "parquet", "gpkg", "npy", "pkl", "pt",
    "png", "txt", "html", "joblib", "yaml", "geojson", "other",
]

StageStatus = Literal["pending", "running", "complete", "failed", "partial"]

# Stage identifier — int 0..4 or string for non-numeric stages
StageId = str  # we serialize all stage keys as strings ("0", "1", ..., "final", "spatial")

# Kind of a Dataset — drives storage format selection and load dispatch.
DatasetKind = Literal[
    "table",      # tabular pandas DataFrame → Parquet
    "geotable",   # GeoDataFrame → GeoParquet (or GPKG fallback)
    "array",      # numpy ndarray → npy/npz
    "checkpoint", # torch state_dict / model weights → pt/joblib
    "metadata",   # dict / list / scalar config → json
    "blob",       # opaque bytes → other
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactEntry(BaseModel):
    """A single registered artifact."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Stable lookup id (e.g. 'scenario_coefficients').")
    stage: StageId = Field(..., description="Producing stage id ('0'..'4', 'final', 'spatial', 'cross').")
    name: str = Field(..., description="On-disk file name including extension.")
    path: str = Field(..., description="Path relative to the run output_dir.")
    format: ArtifactFormat = Field(..., description="Serialization format.")
    producer: Optional[str] = Field(None, description="Module:function that wrote this artifact.")
    consumers: list[str] = Field(
        default_factory=list,
        description="Known consumers (e.g. 'server:/results/causal').",
    )
    size_bytes: int = Field(0, description="File size at registration time.")
    sha256: Optional[str] = Field(None, description="Optional content hash.")
    row_count: Optional[int] = Field(None, description="For tabular artifacts.")
    written_at: str = Field(default_factory=_utcnow, description="ISO-8601 UTC.")
    write_seconds: Optional[float] = Field(None, description="Time spent writing.")
    partial: bool = Field(
        False,
        description=(
            "True while a write is in flight. A partial=True artifact existing "
            "at registry-load time signals an unclean shutdown during this write."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageManifest(BaseModel):
    """Per-stage manifest fragment."""

    model_config = ConfigDict(extra="allow")

    stage: StageId
    status: StageStatus = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    artifacts: dict[str, ArtifactEntry] = Field(
        default_factory=dict,
        description="Keyed by ArtifactEntry.id.",
    )


class RunManifest(BaseModel):
    """Run-wide manifest persisted as ``artifacts_manifest.json``."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    run_id: Optional[str] = None
    project_name: Optional[str] = None
    config_hash: Optional[str] = None
    pipeline_version: Optional[str] = None
    started_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)
    stages: dict[StageId, StageManifest] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    # -- ergonomic helpers --------------------------------------------------

    def get_stage(self, stage: StageId, *, create: bool = True) -> StageManifest:
        s = str(stage)
        sm = self.stages.get(s)
        if sm is None:
            if not create:
                raise KeyError(f"Unknown stage {stage!r}")
            sm = StageManifest(stage=s)
            self.stages[s] = sm
        return sm

    def lookup(self, stage: StageId, artifact_id: str) -> Optional[ArtifactEntry]:
        sm = self.stages.get(str(stage))
        if sm is None:
            return None
        return sm.artifacts.get(artifact_id)

    def all_artifacts(self) -> list[ArtifactEntry]:
        out: list[ArtifactEntry] = []
        for sm in self.stages.values():
            out.extend(sm.artifacts.values())
        return out


# ---------------------------------------------------------------------------
# Dataset abstraction (typed contract between stages)
# ---------------------------------------------------------------------------


class ColumnSpec(BaseModel):
    """One column of a tabular DatasetSchema."""

    model_config = ConfigDict(extra="allow")

    name: str
    dtype: str = Field(..., description="Pandas/Arrow dtype string (e.g. 'int64', 'float64', 'string', 'datetime64[ns]', 'geometry').")
    nullable: bool = True
    description: Optional[str] = None


class DatasetSchema(BaseModel):
    """Typed schema for a Dataset.

    For ``kind='table'`` and ``kind='geotable'``, ``columns`` enumerates the
    expected columns. For non-tabular kinds, ``columns`` may be empty and
    ``array_shape`` / ``array_dtype`` (or free-form metadata) describe the
    structure instead.
    """

    model_config = ConfigDict(extra="allow")

    kind: DatasetKind = "table"
    columns: list[ColumnSpec] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)
    partition_keys: list[str] = Field(default_factory=list)
    geometry_column: Optional[str] = Field(
        None, description="Geometry column name for kind='geotable'."
    )
    crs: Optional[str] = Field(
        None, description="CRS string for kind='geotable' (e.g. 'EPSG:4326')."
    )
    array_shape: Optional[list[Optional[int]]] = Field(
        None, description="Expected ndarray shape (None = any) for kind='array'."
    )
    array_dtype: Optional[str] = Field(
        None, description="Expected ndarray dtype for kind='array'."
    )
    description: Optional[str] = None

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class Dataset(BaseModel):
    """A typed, named bundle of data produced by a stage.

    The dataset is the *contract* between stages. Its on-disk representation
    (Parquet, GeoParquet, npz, pt, json) is a storage detail; consumers
    address it via ``(stage, name)`` and the registry resolves the rest.

    Storage format defaults are chosen by ``kind``:

    * ``table``      → ``parquet``
    * ``geotable``   → ``parquet`` (GeoParquet)
    * ``array``      → ``npy``
    * ``checkpoint`` → ``pt``
    * ``metadata``   → ``json``
    * ``blob``       → ``other``
    """

    model_config = ConfigDict(extra="allow")

    stage: StageId = Field(..., description="Producing stage id ('0'..'4', 'final', 'spatial').")
    name: str = Field(..., description="Stage-local dataset name (e.g. 'cv_predictions').")
    kind: DatasetKind = "table"
    data_schema: DatasetSchema = Field(
        ..., description="Schema describing the dataset's structure."
    )
    storage_format: ArtifactFormat = Field(
        "parquet", description="Canonical on-disk format."
    )
    artifact_id: Optional[str] = Field(
        None,
        description="Registered artifact id; defaults to ``name`` when None.",
    )
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Stable cross-stage identifier: ``'stage1.cv_predictions'``."""
        s = self.stage
        prefix = f"stage{s}" if s.isdigit() else s
        return f"{prefix}.{self.name}"

    def resolved_artifact_id(self) -> str:
        return self.artifact_id or self.name

    @staticmethod
    def default_format_for(kind: DatasetKind) -> ArtifactFormat:
        return {
            "table": "parquet",
            "geotable": "parquet",
            "array": "npy",
            "checkpoint": "pt",
            "metadata": "json",
            "blob": "other",
        }[kind]


__all__ = [
    "ArtifactEntry",
    "ArtifactFormat",
    "ColumnSpec",
    "Dataset",
    "DatasetKind",
    "DatasetSchema",
    "StageManifest",
    "StageStatus",
    "StageId",
    "RunManifest",
]
