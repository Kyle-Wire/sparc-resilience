"""evaluation — Model evaluation utilities."""

from .evaluation import SpatialEvaluator

try:
    from .sensitivity import SensitivityAnalyzer
except ImportError:
    SensitivityAnalyzer = None

__all__ = [
    'SpatialEvaluator',
    'SensitivityAnalyzer',
]
