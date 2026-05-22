"""evaluation — Model evaluation utilities."""

from .evaluation import SpatialEvaluator

try:
    from .sensitivity import SensitivityAnalyzer
except ImportError:
    SensitivityAnalyzer = None

try:
    from .conformal import SpatialConformalPredictor, PredictionIntervals
except ImportError:
    SpatialConformalPredictor = None
    PredictionIntervals = None

__all__ = [
    'SpatialEvaluator',
    'SensitivityAnalyzer',
    'SpatialConformalPredictor',
    'PredictionIntervals',
]
