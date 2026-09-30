"""SPARC core pipeline — lean, tested implementation of the roadmap S0–S7.

Stages (see ``docs/roadmap/CORE_ROADMAP.md`` §2):

    S0 data + QA → S1 influence → S2 base models (+ physics) →
    S3 physics-informed stacker → S4 response/saturation → S5 scenarios →
    S6 causal validation → S7 budget optimisation

The package deliberately imports nothing from ``sparc.models`` (which pulls
in geopandas / pygam / statsmodels at import time).  Heavy dependencies are
limited to numpy, scipy, pandas, scikit-learn and torch.
"""

__all__ = ["run_core", "load_core_config"]


def run_core(*args, **kwargs):
    from sparc.core.pipeline import run_core as _run

    return _run(*args, **kwargs)


def load_core_config(*args, **kwargs):
    from sparc.core.config import load_core_config as _load

    return _load(*args, **kwargs)
