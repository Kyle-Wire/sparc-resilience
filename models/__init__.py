"""
Model implementations for SPARC.

This module includes:
- Base Models:
  - Geographically Weighted Elastic Net (GWEN)
  - Ordinary Least Squares (OLS)
  - Geographically Weighted Regression (GWR)
  - Geographically Weighted Random Forest (GWRF)
  - Geographically Weighted Generalized Additive Model (GGPGAM)
- Meta Models:
  - LightGBM Meta-Ensemble (with monotonic constraints)
  - Deep Kriging V2 Residual Correction (Wendland basis + spatial smoothness)
"""

from .gwen import GWENModel
from .ols import OLSModel
from .gwr import GWRModel
from .gwrf import GWRFModel
from .ggpgam import GGPGAM_SVC
from .meta_ensemble import MetaEnsemble
from .deep_kriging_v2 import DeepKrigingV2

__all__ = [
    'GWENModel',
    'OLSModel',
    'GWRModel',
    'GWRFModel',
    'GGPGAM_SVC',
    'MetaEnsemble',
    'DeepKrigingV2',
] 