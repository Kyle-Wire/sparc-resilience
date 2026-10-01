"""SPARC core pipeline — lean, tested implementation of the roadmap S0–S7.

Stages (see ``docs/roadmap/CORE_ROADMAP.md`` §2):

    S0 data + QA → S1 influence → S2 base models (+ physics) →
    S3 physics-informed stacker → S4 response/saturation → S5 scenarios →
    S6 causal validation → S7 budget optimisation

The package deliberately imports nothing from ``sparc.models`` (which pulls
in geopandas / pygam / statsmodels at import time).  Heavy dependencies are
limited to numpy, scipy, pandas, scikit-learn and torch.

``plan_stages`` and ``open_run`` are exported lazily (module ``__getattr__``):
importing ``sparc.core`` stays cheap, and ``open_run`` (``sparc.core.session``)
resolves only when that module is installed.
"""

__all__ = ["run_core", "load_core_config", "plan_stages", "open_run"]

_LAZY = {"plan_stages": "sparc.core.pipeline", "open_run": "sparc.core.session"}


def run_core(*args, **kwargs):
    from sparc.core.pipeline import run_core as _run

    return _run(*args, **kwargs)


def load_core_config(*args, **kwargs):
    from sparc.core.config import load_core_config as _load

    return _load(*args, **kwargs)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'sparc.core' has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise AttributeError(f"sparc.core.{name} is not available: {module} is not installed") from exc
    return getattr(mod, name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
