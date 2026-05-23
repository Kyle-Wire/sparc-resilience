"""
sparc/models/spec.py — Unified ModelSpec interface for all spatial model configurations.

Before this module, callers imported one of 6 separate ``*_config.py`` files and
constructed a config class directly:

    from sparc.models.gwr_config import GWRConfig
    cfg = GWRConfig(bandwidth=100, kernel="bisquare")
    from sparc.models.gwrf_config import GWRFConfig
    cfg = GWRFConfig(n_estimators=200)
    ...

With ``ModelSpec``, a caller supplies the model name and the project config dict,
and receives the right spec back — without needing to know which class handles
which model:

    from sparc.models.spec import ModelSpec
    spec = ModelSpec.from_project(config, "gwr")

``ModelSpec`` is a discriminated union over the six concrete spec types.
Individual spec classes remain importable from their original modules for
backward compatibility.

Interface
---------
::

    ModelSpec.from_project(config: dict, name: str) → ModelSpec
        Construct the right spec for *name* from the project config dict.

    ModelSpec.from_kwargs(name: str, **kwargs) → ModelSpec
        Construct with explicit kwargs (for tests).

    spec.name → str          (canonical model name)
    spec.as_gwr_config()     (cast to GWRConfig — raises if wrong type)
    spec.as_gwrf_config()    (cast to GWRFConfig — raises if wrong type)
    ...
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Re-import all concrete spec classes (they stay in their own files)
# ---------------------------------------------------------------------------

from .gwr_config import GWRConfig
from .gwrf_config import GWRFConfig
from .gwen_config import GWENConfig
from .ggpgam_config import GGPGAMConfig

# Neural + process rate configs are optional (torch may not be installed)
try:
    from .neural_meta_config import NeuralMetaConfig
    _NEURAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    NeuralMetaConfig = None  # type: ignore[assignment,misc]
    _NEURAL_AVAILABLE = False

try:
    from .process_rate_config import ProcessRateConfig
    _PROCESS_RATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ProcessRateConfig = None  # type: ignore[assignment,misc]
    _PROCESS_RATE_AVAILABLE = False

# OLS has no dedicated config file (uses inline kwargs); represent as a stub
from dataclasses import dataclass


@dataclass
class OLSConfig:
    """Minimal config for OLSModel (no hyperparameters required)."""
    use_laplacian: bool = False
    n_eigenmaps: int = 150


# ---------------------------------------------------------------------------
# ModelSpec discriminated union
# ---------------------------------------------------------------------------

_ConcreteSpec = Union[GWRConfig, GWRFConfig, GWENConfig, GGPGAMConfig, OLSConfig]

#: All model names recognised by :meth:`ModelSpec.from_project`.
ModelName = Literal["gwr", "gwrf", "gwen", "ggpgam", "ols", "neural", "process_rate"]


class ModelSpec:
    """Factory and thin wrapper around a concrete model configuration.

    Usage::

        spec = ModelSpec.from_project(config, "gwr")
        spec = ModelSpec.from_kwargs("gwen", k_neighbors=100, cv_folds=5)
        gwr_cfg = spec.as_gwr_config()
    """

    def __init__(self, name: str, spec: Any) -> None:
        self._name = name
        self._spec = spec

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_project(cls, config: Dict[str, Any], name: str) -> "ModelSpec":
        """Build the right config for *name* from a project config dict.

        Parameters
        ----------
        config:
            Full pipeline config dict (from ``load_config()`` or
            ``ProjectConfig.load().raw``).
        name:
            One of ``"gwr"``, ``"gwrf"``, ``"gwen"``, ``"ggpgam"``,
            ``"ols"``, ``"neural"``, ``"process_rate"``.
        """
        name = name.lower().strip()
        models_cfg: Dict[str, Any] = config.get("models", {})
        cv_params: Dict[str, Any] = config.get("cv_params", {})
        gwen_params: Dict[str, Any] = config.get("gwen_params", {})
        lap_params: Dict[str, Any] = config.get("laplacian_params", {})

        if name == "gwr":
            gwr_raw = models_cfg.get("gwr", {})
            spec = GWRConfig(
                bandwidth=gwr_raw.get("bandwidth"),
                kernel=gwr_raw.get("kernel", "gaussian"),
                alpha=float(gwr_raw.get("alpha", 0.1)),
                min_points=int(gwr_raw.get("min_points", 50)),
                use_constrained_regression=bool(
                    gwr_raw.get("use_constrained_regression", True)
                ),
            )

        elif name == "gwrf":
            gwrf_raw = models_cfg.get("gwrf", {})
            spec = GWRFConfig(
                n_estimators=int(gwrf_raw.get("n_estimators", 100)),
                k_neighbors=int(gwrf_raw.get("k_neighbors", 100)),
                min_samples_leaf=int(gwrf_raw.get("min_samples_leaf", 5)),
                n_jobs=int(gwrf_raw.get("n_jobs", -1)),
            )

        elif name == "gwen":
            spec = GWENConfig(
                k_neighbors=int(gwen_params.get("k_neighbors", 500)),
                cv_folds=int(gwen_params.get("cv_folds", 5)),
                n_alphas=int(gwen_params.get("n_alphas", 100)),
                l1_ratios=list(gwen_params.get("l1_ratios", [0.1, 0.5, 0.9, 0.99])),
                sample_size=gwen_params.get("sample_size"),
                selection_threshold=float(
                    gwen_params.get("selection_threshold", 0.1)
                ),
            )

        elif name == "ggpgam":
            ggpgam_raw = models_cfg.get("ggpgam", {})
            spec = GGPGAMConfig(
                n_splines=int(ggpgam_raw.get("n_splines", 5)),
                n_spatial_bases=int(ggpgam_raw.get("n_spatial_bases", 10)),
                lam=float(ggpgam_raw.get("lam", 0.6)),
            )

        elif name == "ols":
            spec = OLSConfig(
                use_laplacian=bool(
                    config.get("flags", {}).get("use_laplacian_eigenmaps_in_ols", False)
                ),
                n_eigenmaps=int(lap_params.get("n_eigenmaps", 150)),
            )

        elif name == "neural":
            if not _NEURAL_AVAILABLE:
                raise ImportError(
                    "NeuralMetaConfig requires torch. Install with: pip install torch"
                )
            neural_raw = models_cfg.get("neural", {})
            spec = NeuralMetaConfig(**{
                k: neural_raw[k] for k in neural_raw
                if k in NeuralMetaConfig.__dataclass_fields__
            })

        elif name == "process_rate":
            if not _PROCESS_RATE_AVAILABLE:
                raise ImportError("ProcessRateConfig unavailable")
            pr_raw = config.get("process_rate", {})
            spec = ProcessRateConfig(**{
                k: pr_raw[k] for k in pr_raw
                if k in ProcessRateConfig.__dataclass_fields__
            })

        else:
            raise ValueError(
                f"Unknown model name {name!r}. "
                f"Choose from: gwr, gwrf, gwen, ggpgam, ols, neural, process_rate"
            )

        return cls(name, spec)

    @classmethod
    def from_kwargs(cls, name: str, **kwargs: Any) -> "ModelSpec":
        """Construct with explicit kwargs — useful in tests."""
        name = name.lower().strip()
        _cls_map = {
            "gwr": GWRConfig,
            "gwrf": GWRFConfig,
            "gwen": GWENConfig,
            "ggpgam": GGPGAMConfig,
            "ols": OLSConfig,
        }
        if name not in _cls_map:
            raise ValueError(f"Unknown model name for from_kwargs: {name!r}")
        return cls(name, _cls_map[name](**kwargs))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Canonical model name (e.g. ``"gwr"``)."""
        return self._name

    @property
    def spec(self) -> Any:
        """The underlying concrete config object."""
        return self._spec

    def as_gwr_config(self) -> GWRConfig:
        if not isinstance(self._spec, GWRConfig):
            raise TypeError(f"spec is {type(self._spec).__name__}, not GWRConfig")
        return self._spec

    def as_gwrf_config(self) -> GWRFConfig:
        if not isinstance(self._spec, GWRFConfig):
            raise TypeError(f"spec is {type(self._spec).__name__}, not GWRFConfig")
        return self._spec

    def as_gwen_config(self) -> GWENConfig:
        if not isinstance(self._spec, GWENConfig):
            raise TypeError(f"spec is {type(self._spec).__name__}, not GWENConfig")
        return self._spec

    def as_ggpgam_config(self) -> GGPGAMConfig:
        if not isinstance(self._spec, GGPGAMConfig):
            raise TypeError(f"spec is {type(self._spec).__name__}, not GGPGAMConfig")
        return self._spec

    def as_ols_config(self) -> OLSConfig:
        if not isinstance(self._spec, OLSConfig):
            raise TypeError(f"spec is {type(self._spec).__name__}, not OLSConfig")
        return self._spec

    def __repr__(self) -> str:
        return f"ModelSpec(name={self._name!r}, spec={self._spec!r})"
