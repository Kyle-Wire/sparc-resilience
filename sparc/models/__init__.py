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

# V2 neural modules (lazy imports — require torch)
try:
    from .neural_meta import SPARCMetaLearner
    from .surrogates import DifferentiableGWR, DifferentiableGWRF, DifferentiableGGPGAM
    from .process_rate_net import ProcessRateNet
    from .spatial_attention import SIRENLayer, SparseSpatialAttention
    _V2_AVAILABLE = True
except ImportError:
    _V2_AVAILABLE = False

__all__ = [
    'GWENModel',
    'OLSModel',
    'GWRModel',
    'GWRFModel',
    'GGPGAM_SVC',
    'MetaEnsemble',
    'DeepKrigingV2',
    # V2
    'SPARCMetaLearner',
    'DifferentiableGWR',
    'DifferentiableGWRF',
    'DifferentiableGGPGAM',
    'ProcessRateNet',
    'SIRENLayer',
    'SparseSpatialAttention',
] 