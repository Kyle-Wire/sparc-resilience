"""
Model implementations for SPARC.

This module includes:
- Base Models:
  - Geographically Weighted Elastic Net (GWEN)
  - Ordinary Least Squares (OLS)
  - Geographically Weighted Regression (GWR)
  - Geographically Weighted Random Forest (GWRF)
  - Geographically Weighted Generalized Additive Model (GGPGAM)
- Neural Meta-Learner:
  - SPARCMetaLearner (PyTorch, physics-informed, multi-stream)
- ModelSpec:
  - Unified discriminated-union config factory
"""

from .gwen import GWENModel
from .ols import OLSModel
from .gwr import GWRModel
from .gwrf import GWRFModel
from .ggpgam import GGPGAM_SVC
from .spec import ModelSpec

# V2 neural modules (lazy imports — require torch)
try:
    from .neural_meta import SPARCMetaLearner
    _V2_AVAILABLE = True
except ImportError:
    _V2_AVAILABLE = False

__all__ = [
    'GWENModel',
    'OLSModel',
    'GWRModel',
    'GWRFModel',
    'GGPGAM_SVC',
    'ModelSpec',
    # V2 Neural — single public entry point
    'SPARCMetaLearner',
]
