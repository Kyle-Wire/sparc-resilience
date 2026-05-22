"""
StageConfig — typed, validated gateway from ``project.yml`` into Stage 2.

All string-key lookups from ``self.base_config`` that were previously
scattered across 15+ methods in ``EnhancedSpatialCV`` are now performed
*once*, here, at construction time.  Downstream code reads typed fields;
the config dict is never accessed again outside this module.

Usage
-----
::

    from sparc.run.stage_config import StageConfig
    from sparc.config.config import load_config

    cfg = StageConfig.from_dict(load_config())

    # typed, IDE-navigable, fail-fast on bad project.yml:
    cfg.target          # str
    cfg.features        # list[str]
    cfg.gwr_bandwidth   # float | None
    cfg.block_size      # float | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StageConfig:
    """Typed representation of every project.yml key consumed by Stage 2.

    Prefer reading typed fields over accessing the raw dict.  See
    :meth:`from_dict` for how each field maps to ``project.yml`` paths.
    """

    # ── data paths ───────────────────────────────────────────────────────────
    raw_csv_path: str
    output_dir: str

    # ── variable identifiers ─────────────────────────────────────────────────
    target: str
    identifier: str
    coordinates: List[str]
    features: List[str]

    # ── CRS settings ─────────────────────────────────────────────────────────
    initial_crs: str
    target_crs: str

    # ── spatial CV settings ──────────────────────────────────────────────────
    block_size: Optional[float]
    buffer_size: float
    buffer_size_auto: bool
    n_splits: int
    self_supervised_scoring: bool

    # ── model hyperparameters ─────────────────────────────────────────────────
    gwr_params: Dict        # full gwr sub-dict from project.yml
    gwrf_params: Dict       # full gwrf sub-dict from project.yml
    ggpgam_params: Dict     # full ggpgam sub-dict from project.yml

    # ── pipeline flags ────────────────────────────────────────────────────────
    skip_stage_2_base_models: bool    skip_stage_2b_full_retrain: bool    use_dataset_profiler: bool
    benchmark_metrics_enabled: bool

    # ── output CRS for GeoPackage files ──────────────────────────────────────
    output_crs: str

    # ── continual learning ────────────────────────────────────────────────────
    continual_registry_path: Optional[str]
    continual_prior_city: Optional[str]

    # ── stage output dir name (for registry key) ─────────────────────────────
    stage2_dir_name: str

    # ── escape hatch for keys not yet promoted to typed fields ───────────────
    raw: Dict = field(default_factory=dict, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict) -> "StageConfig":
        """Parse and validate a loaded ``project.yml`` config dict.

        Fails fast at pipeline start-up rather than deep inside a fold when
        a key is absent or the wrong type.

        Parameters
        ----------
        raw : dict
            Config dict as returned by ``sparc.config.config.load_config()``.

        Returns
        -------
        StageConfig
        """
        # ── required keys ────────────────────────────────────────────────────
        try:
            raw_csv_path = raw["paths"]["raw_csv_path"]
        except KeyError as exc:
            raise ValueError(
                "project.yml is missing paths.raw_csv_path — "
                "Stage 2 cannot locate the input data."
            ) from exc

        try:
            target = raw["variables"]["target"]
        except KeyError as exc:
            raise ValueError(
                "project.yml is missing variables.target — "
                "Stage 2 cannot identify the response variable."
            ) from exc

        # ── optional keys with typed defaults ────────────────────────────────
        paths_section     = raw.get("paths", {})
        output_section    = raw.get("output", {})
        variables_section = raw.get("variables", {})
        predictors_section = raw.get("predictors", {})
        crs_section       = raw.get("crs", {})
        models_section    = raw.get("models", {})
        spatial_cv_sec    = models_section.get("spatial_cv", {})
        pipeline_section  = raw.get("pipeline", {})
        pipeline_exec_sec = raw.get("pipeline_execution", {})
        benchmark_sec     = raw.get("benchmark_metrics", {})
        continual_sec     = raw.get("continual", {})
        stage_dirs        = output_section.get("stage_dirs", {})

        # block_size: explicit float | None
        _bs = spatial_cv_sec.get("block_size", None)
        block_size = float(_bs) if _bs is not None else None

        # buffer_size: explicit float, default 0
        _buf = spatial_cv_sec.get("buffer_size", 0)
        buffer_size = float(_buf) if _buf is not None else 0.0

        return cls(
            raw_csv_path=raw_csv_path,
            output_dir=str(
                output_section.get(
                    "base_dir",
                    paths_section.get("output_dir", "output"),
                )
            ),
            target=target,
            identifier=variables_section.get("identifier", "id"),
            coordinates=list(variables_section.get("coordinates", ["lon", "lat"])),
            features=list(predictors_section.get("base_model", [])),
            initial_crs=str(crs_section.get("initial", "EPSG:4326")),
            target_crs=str(crs_section.get("target_projected", "EPSG:26919")),
            block_size=block_size,
            buffer_size=buffer_size,
            buffer_size_auto=bool(spatial_cv_sec.get("buffer_size_auto", False)),
            n_splits=int(spatial_cv_sec.get("n_splits", 5)),
            self_supervised_scoring=bool(
                spatial_cv_sec.get("self_supervised_hparam_scoring", False)
            ),
            gwr_params=dict(models_section.get("gwr", {})),
            gwrf_params=dict(models_section.get("gwrf", {})),
            ggpgam_params=dict(models_section.get("ggpgam", {})),
            skip_stage_2_base_models=bool(
                pipeline_exec_sec.get("skip_stage_2_base_models", False)
            ),
            skip_stage_2b_full_retrain=bool(
                pipeline_exec_sec.get("skip_stage_2b_full_retrain", False)
            ),
            use_dataset_profiler=bool(
                pipeline_section.get("use_dataset_profiler", False)
            ),
            benchmark_metrics_enabled=bool(benchmark_sec.get("enabled", False)),
            output_crs=str(crs_section.get("target_projected", "EPSG:26919")),
            continual_registry_path=continual_sec.get("registry_path"),
            continual_prior_city=continual_sec.get("prior_city"),
            stage2_dir_name=stage_dirs.get("stage_2", "Stage_2_Spatial_CV"),
            raw=raw,
        )

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def auto_buffer_from_block(self) -> float:
        """Compute auto buffer as 1/3 of ``block_size``, or 0 if not set."""
        if self.block_size is not None:
            return max(100.0, self.block_size / 3.0)
        return 0.0

    def effective_buffer_size(self) -> float:
        """Return the buffer to use in fold generation.

        Returns the auto-computed value when ``buffer_size_auto`` is *True*
        and ``block_size`` is set; otherwise returns the explicit
        ``buffer_size`` (default 0).
        """
        if self.buffer_size_auto and self.block_size is not None:
            return self.auto_buffer_from_block()
        return self.buffer_size
