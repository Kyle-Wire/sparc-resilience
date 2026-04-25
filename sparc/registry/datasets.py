"""Canonical Dataset specs for every pipeline stage.

Every stage that produces data declares its outputs here as ``Dataset``
instances. Stages then call ``ResultStore.save_dataset(spec, data)`` to
write them — schema, format, and registry id all flow from one place.

Naming convention (PR2):
    Directories   ``output/stageN_topic/``  (e.g. ``stage0_correlogram``)
    Tabular data  ``*.parquet``  (canonical)
    Metadata      ``*.json``
    Arrays        ``*.npy``
    Checkpoints   ``*.pt``
    Files inside a stage dir drop redundant prefixes:
        ``stage1_gwen/results.json``  not  ``gwen_results.json``.
"""

from __future__ import annotations

from sparc.registry.manifest import ColumnSpec, Dataset, DatasetSchema


# ---------------------------------------------------------------------------
# Stage 0 — Correlogram Analysis  (output/stage0_correlogram/)
# ---------------------------------------------------------------------------

STAGE0_DATASET_PROFILE = Dataset(
    stage="0",
    name="profile",
    kind="metadata",
    storage_format="json",
    data_schema=DatasetSchema(
        kind="metadata",
        description=(
            "DatasetProfiler output: size tier, spatial extent, "
            "recommended correlogram parameters."
        ),
    ),
    description="Profiler-driven dataset characterization consumed by Stage 0b.",
    metadata={"producer": "sparc.run.correlogram_analysis:main"},
)

STAGE0_CORRELOGRAM_RESULTS = Dataset(
    stage="0",
    name="results",
    kind="metadata",
    storage_format="json",
    data_schema=DatasetSchema(
        kind="metadata",
        description=(
            "Comprehensive correlogram results: per-variable bandwidths, "
            "model bandwidths, spatial CV configuration, summary stats."
        ),
    ),
    description=(
        "Single source of truth for correlogram-derived spatial parameters. "
        "Read by Stage 2 for bandwidth selection and the desktop UI."
    ),
    metadata={"producer": "sparc.run.correlogram_analysis:main"},
)

STAGE0_CORRELOGRAM_SUMMARY = Dataset(
    stage="0",
    name="summary",
    kind="table",
    storage_format="parquet",
    data_schema=DatasetSchema(
        kind="table",
        columns=[
            ColumnSpec(name="Variable", dtype="string"),
            ColumnSpec(name="Optimal_Bandwidth", dtype="float64"),
            ColumnSpec(name="Effective_Range", dtype="float64"),
            ColumnSpec(name="Block_Size", dtype="float64"),
            ColumnSpec(name="Best_Kernel", dtype="string"),
            ColumnSpec(name="Max_Moran_I", dtype="float64"),
            ColumnSpec(name="Significant_Lags", dtype="int64"),
        ],
        primary_key=["Variable"],
    ),
    description="Per-variable correlogram summary (one row per analyzed variable).",
    metadata={"producer": "sparc.run.correlogram_analysis:main"},
)

STAGE0_SPATIAL_CV_CONFIG = Dataset(
    stage="0",
    name="spatial_cv_config",
    kind="metadata",
    storage_format="json",
    data_schema=DatasetSchema(
        kind="metadata",
        description=(
            "Optimal CV block size, model bandwidths, and per-block "
            "validation results derived from correlogram analysis."
        ),
    ),
    description=(
        "Structured CV configuration. The legacy ``spatial_cv_config.txt`` "
        "human-readable companion is still written for backward compatibility."
    ),
    metadata={"producer": "sparc.run.correlogram_analysis:main"},
)


# ---------------------------------------------------------------------------
# Stage 1 — GWEN Variable Selection  (output/stage1_gwen/)
# ---------------------------------------------------------------------------

STAGE1_GWEN_RESULTS = Dataset(
    stage="1",
    name="results",
    kind="metadata",
    storage_format="json",
    data_schema=DatasetSchema(
        kind="metadata",
        description=(
            "GWEN selection summary: selected features, model params, "
            "feature importance records."
        ),
    ),
    description="Top-level GWEN output consumed by server endpoints and reports.",
    metadata={"producer": "sparc.run.gwen_variable_selection:save_gwen_results"},
)

STAGE1_GWEN_DIAGNOSTICS = Dataset(
    stage="1",
    name="diagnostics",
    kind="metadata",
    storage_format="json",
    data_schema=DatasetSchema(
        kind="metadata",
        description=(
            "GWEN diagnostics: tuning params, per-feature stability metrics."
        ),
    ),
    description="Diagnostic metadata for the GWEN run (no per-row tabular data).",
    metadata={"producer": "sparc.run.gwen_variable_selection:save_gwen_diagnostics"},
)

STAGE1_VARIABLE_IMPORTANCE = Dataset(
    stage="1",
    name="importance",
    kind="table",
    storage_format="parquet",
    data_schema=DatasetSchema(
        kind="table",
        columns=[
            ColumnSpec(name="feature", dtype="string"),
            ColumnSpec(name="selection_frequency", dtype="float64"),
            ColumnSpec(name="mean_local_coef", dtype="float64"),
            ColumnSpec(name="spatial_variability", dtype="float64"),
        ],
        primary_key=["feature"],
    ),
    description="Per-feature GWEN importance metrics.",
    metadata={"producer": "sparc.run.gwen_variable_selection:save_gwen_results"},
)

STAGE1_SELECTED_FEATURES = Dataset(
    stage="1",
    name="selected_features",
    kind="metadata",
    storage_format="json",
    data_schema=DatasetSchema(
        kind="metadata",
        description="JSON list of feature names approved by GWEN selection.",
    ),
    description=(
        "Canonical selected-features list. The legacy ``selected_features.txt`` "
        "companion is still written for backward compatibility (PR3 will "
        "convert consumers to read the JSON dataset instead)."
    ),
    metadata={"producer": "sparc.run.gwen_variable_selection:save_gwen_results"},
)


# ---------------------------------------------------------------------------
# Indexed lookup
# ---------------------------------------------------------------------------

STAGE0_DATASETS: tuple[Dataset, ...] = (
    STAGE0_DATASET_PROFILE,
    STAGE0_CORRELOGRAM_RESULTS,
    STAGE0_CORRELOGRAM_SUMMARY,
    STAGE0_SPATIAL_CV_CONFIG,
)

STAGE1_DATASETS: tuple[Dataset, ...] = (
    STAGE1_GWEN_RESULTS,
    STAGE1_GWEN_DIAGNOSTICS,
    STAGE1_VARIABLE_IMPORTANCE,
    STAGE1_SELECTED_FEATURES,
)

ALL_DATASETS: dict[str, Dataset] = {
    d.qualified_name: d
    for d in (*STAGE0_DATASETS, *STAGE1_DATASETS)
}


def datasets_for_stage(stage: str | int) -> tuple[Dataset, ...]:
    """Return the registered Dataset specs owned by *stage*."""
    s = str(stage)
    return tuple(d for d in ALL_DATASETS.values() if d.stage == s)


__all__ = [
    "STAGE0_DATASET_PROFILE",
    "STAGE0_CORRELOGRAM_RESULTS",
    "STAGE0_CORRELOGRAM_SUMMARY",
    "STAGE0_SPATIAL_CV_CONFIG",
    "STAGE1_GWEN_RESULTS",
    "STAGE1_GWEN_DIAGNOSTICS",
    "STAGE1_VARIABLE_IMPORTANCE",
    "STAGE1_SELECTED_FEATURES",
    "STAGE0_DATASETS",
    "STAGE1_DATASETS",
    "ALL_DATASETS",
    "datasets_for_stage",
]
