"""SPARCConfig — unified typed facade over ``project.yml``.

Previously each pipeline stage read raw dict keys directly from the
``load_config()`` output, producing silent failures when keys were
misspelt or missing.  This module centralises all key access behind
typed, documented attributes.

Backward compatibility
----------------------
The existing :class:`~sparc.run.stage_config.StageConfig` continues to
work unchanged — it operates on the same underlying dict.

Usage
-----
::

    from sparc.config.sparc_config import SPARCConfig

    cfg = SPARCConfig.load("project.yml")

    # IDE-navigable, fail-fast on bad YAML:
    cfg.variables.target          # str
    cfg.variables.coordinates     # list[str]
    cfg.causal.dag_file           # str | None
    cfg.causal.estimator          # str

    # Escape hatch: raw dict still accessible
    cfg.raw["some_exotic_key"]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class VariablesConfig:
    """Variables block from ``project.yml``."""

    target: str
    identifier: str
    coordinates: List[str]
    predictors: List[str]
    location: Optional[str] = None

    @classmethod
    def _from_raw(cls, cfg: dict) -> "VariablesConfig":
        variables = cfg.get("variables", {}) or {}
        predictors_raw = cfg.get("predictors", {})
        if isinstance(predictors_raw, dict):
            predictors = predictors_raw.get("base_model", [])
        elif isinstance(predictors_raw, list):
            predictors = predictors_raw
        else:
            predictors = []
        return cls(
            target=variables.get("target", cfg.get("target", "")),
            identifier=variables.get("identifier", cfg.get("identifier", "")),
            coordinates=variables.get("coordinates", cfg.get("coordinates", [])),
            predictors=list(predictors),
            location=variables.get("location"),
        )


@dataclass
class CRSConfig:
    """CRS block from ``project.yml``."""

    initial: str
    target_projected: str

    @classmethod
    def _from_raw(cls, cfg: dict) -> "CRSConfig":
        crs = cfg.get("crs", {}) or {}
        return cls(
            initial=crs.get("initial", "EPSG:4326"),
            target_projected=crs.get("target_projected", "EPSG:32618"),
        )


@dataclass
class CausalConfig:
    """Causal block from ``project.yml``."""

    dag_file: Optional[str]
    estimator: str
    inference: str
    ipw_enabled: bool
    gps_enabled: bool
    matching_enabled: bool
    doubly_robust: bool
    estimate_cate: bool
    bootstrap_n: int
    use_stage2_outcome_nuisance: bool

    @classmethod
    def _from_raw(cls, cfg: dict) -> "CausalConfig":
        causal = cfg.get("causal", {}) or {}
        return cls(
            dag_file=causal.get("dag_file"),
            estimator=causal.get("estimator", "hgb"),
            inference=causal.get("inference", "bayesian"),
            ipw_enabled=bool(causal.get("ipw_enabled", True)),
            gps_enabled=bool(causal.get("gps_enabled", True)),
            matching_enabled=bool(causal.get("matching_enabled", True)),
            doubly_robust=bool(causal.get("doubly_robust", True)),
            estimate_cate=bool(causal.get("estimate_cate", True)),
            bootstrap_n=int(causal.get("bootstrap_n", 100)),
            use_stage2_outcome_nuisance=bool(
                causal.get("use_stage2_outcome_nuisance", True)
            ),
        )


@dataclass
class OutputConfig:
    """Paths block from ``project.yml``."""

    output_dir: str
    raw_csv_path: str

    @classmethod
    def _from_raw(cls, cfg: dict) -> "OutputConfig":
        paths = cfg.get("paths", {}) or {}
        return cls(
            output_dir=paths.get("output_dir", cfg.get("output_dir", "output")),
            raw_csv_path=paths.get(
                "raw_csv_path", cfg.get("raw_csv_path", "")
            ),
        )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class SPARCConfig:
    """Typed facade over the ``project.yml``-derived config dict.

    Attributes
    ----------
    variables : VariablesConfig
    crs : CRSConfig
    causal : CausalConfig
    output : OutputConfig
    raw : dict
        The full underlying dict (escape hatch for keys not yet
        promoted to typed fields).
    """

    variables: VariablesConfig
    crs: CRSConfig
    causal: CausalConfig
    output: OutputConfig
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "SPARCConfig":
        """Load a ``project.yml`` from *path* and return a :class:`SPARCConfig`.

        Parameters
        ----------
        path : str | Path
            Absolute or relative path to the project YAML file.

        Returns
        -------
        SPARCConfig

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        ValueError
            When the file is empty or unparseable.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"project.yml not found: {path}")
        from sparc.config.config import load_config
        raw = load_config(str(path))
        if not raw:
            raise ValueError(f"Empty or invalid project.yml: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "SPARCConfig":
        """Build a :class:`SPARCConfig` from a raw config dict.

        Parameters
        ----------
        d : dict
            As returned by :func:`sparc.config.config.load_config`.

        Returns
        -------
        SPARCConfig

        Raises
        ------
        TypeError
            When *d* is not a dict.
        """
        if not isinstance(d, dict):
            raise TypeError(
                f"SPARCConfig.from_dict() requires a dict, got {type(d).__name__}"
            )
        return cls(
            variables=VariablesConfig._from_raw(d),
            crs=CRSConfig._from_raw(d),
            causal=CausalConfig._from_raw(d),
            output=OutputConfig._from_raw(d),
            raw=d,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def dag_file(self) -> Optional[str]:
        """Shortcut to ``causal.dag_file``."""
        return self.causal.dag_file

    @property
    def target(self) -> str:
        """Shortcut to ``variables.target``."""
        return self.variables.target

    @property
    def predictors(self) -> List[str]:
        """Shortcut to ``variables.predictors``."""
        return self.variables.predictors
