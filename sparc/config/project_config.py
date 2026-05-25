"""
sparc/config/project_config.py — Single deep module for pipeline configuration.

All config loading, merging, hardware detection, and physics priors resolution
flow through one interface: ``ProjectConfig.load()``.

Before this module existed, callers assembled config from 7 separate patterns:
  - load_config()              (main YAML → dict)
  - merged_causal_block()      (causal defaults merge)
  - detect_profile()           (hardware detection)
  - load_physics_priors()      (priors.yml)
  - build_physics_priors_map() (merged priors map)
  - *_config.py files          (model-level configs)
  - InterventionConfig(...)    (intervention schema)

Now callers do one thing::

    cfg = ProjectConfig.load("path/to/project.yml")
    cfg.output_dir
    cfg.physics_priors_map
    cfg.model_spec("gwr")

The underlying ``dict`` is still accessible via ``cfg.raw`` for backward compat
with modules that index directly into the config dict.

Interface
---------
::

    ProjectConfig.load(path: str | None = None) → ProjectConfig
    cfg.raw            → dict          (full config dict, same shape as load_config())
    cfg.output_dir     → str           (single authoritative output directory)
    cfg.data_path      → str           (resolved input data path)
    cfg.causal         → dict          (merged causal block)
    cfg.physics        → dict          (physics section from project YAML)
    cfg.hardware       → HardwareProfile (detected hardware tier)
    cfg.physics_priors_raw  → dict     (raw priors.yml content)
    cfg.physics_priors_map  → dict     (merged per-variable coefficient map)
    cfg.model_spec(name)    → ModelSpec (in sparc.models.spec)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Typed accessor dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpatialCVConfig:
    """Typed view of the ``models.spatial_cv`` config block."""
    strategy: str = "spatial_block"
    n_folds: int = 5
    block_size: int = 1000


@dataclass(frozen=True)
class PhysicsConfig:
    """Typed view of the ``physics`` config block."""
    monotone_constraints: bool = False
    pde_weight: float = 0.1


@dataclass(frozen=True)
class StageConfig:
    """Typed view of a single pipeline stage config."""
    fast_mode: bool = False


class ProjectConfig:
    """Single source of truth for a pipeline run's configuration.

    Do not construct directly; use :meth:`load`.
    """

    def __init__(self, raw: dict, source_path: Optional[str] = None) -> None:
        self._raw = raw
        self._source_path = source_path

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ProjectConfig":
        """Load, validate, and fully resolve pipeline configuration.

        Parameters
        ----------
        path:
            Path to a ``project.yml``.  When *None*, the ``SPARC_PROJECT``
            env var is checked, then the legacy Brown-UHI defaults are
            returned with a deprecation warning.

        Returns
        -------
        ProjectConfig
            Fully resolved configuration.  ``hardware`` and
            ``physics_priors_map`` are loaded lazily on first access.
        """
        from .config import load_config
        raw = load_config(path)
        return cls(raw, source_path=path)

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    @property
    def raw(self) -> Dict[str, Any]:
        """Full config dict — same shape as the old ``load_config()`` return value."""
        return self._raw

    @property
    def source_path(self) -> Optional[str]:
        """Absolute path to the ``project.yml`` that was loaded, or ``None`` for defaults."""
        return self._source_path

    @property
    def output_dir(self) -> str:
        """Authoritative output directory — resolves the three-key ambiguity.

        Resolution order:
        1. ``config['output']['base_dir']``
        2. ``config['paths']['output_dir']``
        3. ``./output`` relative to cwd
        """
        out = self._raw.get("output", {}).get("base_dir")
        if out:
            return str(out)
        paths_out = self._raw.get("paths", {}).get("output_dir")
        if paths_out:
            return str(paths_out)
        return str(Path("output").resolve())

    @property
    def data_path(self) -> str:
        """Resolved path to the input data file."""
        return str(
            self._raw.get("data", {}).get("file_path")
            or self._raw.get("paths", {}).get("raw_csv_path")
            or ""
        )

    @property
    def target_column(self) -> str:
        return str(self._raw.get("data", {}).get("target_column", ""))

    @property
    def coord_columns(self) -> list:
        return list(self._raw.get("data", {}).get("coord_columns", []))

    @property
    def causal(self) -> Dict[str, Any]:
        """Fully merged causal block (user + Wager-2025 defaults)."""
        return dict(self._raw.get("causal", {}))

    @property
    def physics(self) -> Dict[str, Any]:
        """Physics section from project YAML."""
        return dict(self._raw.get("physics", {}))

    @property
    def models(self) -> Dict[str, Any]:
        """Models section from project YAML."""
        return dict(self._raw.get("models", {}))

    @property
    def pipeline(self) -> Dict[str, Any]:
        """Pipeline execution parameters."""
        return dict(self._raw.get("pipeline", {}))

    @property
    def disk_writes(self) -> bool:
        """Whether per-stage disk artefacts are written (vs artifact DB only)."""
        return bool(self._raw.get("output", {}).get("disk_writes", False))

    # ------------------------------------------------------------------
    # Hardware (lazy)
    # ------------------------------------------------------------------

    @cached_property
    def hardware(self) -> Any:
        """Detected :class:`~sparc.config.hardware_profile.HardwareProfile`.

        Resolution order: SPARC_HARDWARE_TIER env → project ``performance``
        block → user prefs → auto-detect.
        """
        from .hardware_profile import detect_profile, set_runtime_overrides
        set_runtime_overrides(self._raw.get("performance", {}))
        return detect_profile()

    # ------------------------------------------------------------------
    # Physics priors (lazy)
    # ------------------------------------------------------------------

    @cached_property
    def physics_priors_raw(self) -> Dict[str, Any]:
        """Raw ``priors.yml`` content as a dict; ``{}`` if no priors file configured."""
        from .config import load_physics_priors
        return load_physics_priors(self._raw)

    @cached_property
    def physics_priors_map(self) -> Dict[str, Dict[str, Any]]:
        """Merged per-variable coefficient map (literature × w + OLS × (1-w)).

        Returns ``{}`` if no priors file is configured.
        """
        try:
            from sparc.interventions.physics_priors_registry import build_physics_priors_map
            return build_physics_priors_map(self._raw)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Model spec accessor
    # ------------------------------------------------------------------

    def model_spec(self, name: str) -> Any:
        """Return the resolved :class:`~sparc.models.spec.ModelSpec` for *name*.

        Parameters
        ----------
        name:
            One of ``"gwr"``, ``"gwrf"``, ``"gwen"``, ``"ggpgam"``,
            ``"ols"``, ``"neural"``.
        """
        from sparc.models.spec import ModelSpec
        return ModelSpec.from_project(self._raw, name)

    # ------------------------------------------------------------------
    # Dict-like passthrough (backward compat)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._raw

    def __repr__(self) -> str:
        src = self._source_path or "defaults"
        return f"ProjectConfig(source={src!r}, output_dir={self.output_dir!r})"

    # ------------------------------------------------------------------
    # Typed sub-accessors (C3 — eliminate raw-dict KeyErrors)
    # ------------------------------------------------------------------

    @property
    def spatial_cv(self) -> SpatialCVConfig:
        """Typed view of the ``models.spatial_cv`` config block."""
        raw = self._raw.get("models", {}).get("spatial_cv", {})
        return SpatialCVConfig(
            strategy=str(raw.get("strategy", "spatial_block")),
            n_folds=int(raw.get("n_folds", 5)),
            block_size=int(raw.get("block_size", 1000)),
        )

    @property
    def physics_config(self) -> PhysicsConfig:
        """Typed view of the ``physics`` config block."""
        raw = self._raw.get("physics", {})
        return PhysicsConfig(
            monotone_constraints=bool(raw.get("monotone_constraints", False)),
            pde_weight=float(raw.get("pde_weight", 0.1)),
        )

    def stage(self, n: int) -> StageConfig:
        """Return typed config for pipeline stage *n* (0-based).

        Returns a default :class:`StageConfig` when the stage is not configured.
        """
        key = f"stage_{n}"
        raw = self._raw.get("pipeline", {}).get("stages", {}).get(key, {})
        return StageConfig(fast_mode=bool(raw.get("fast_mode", False)))

