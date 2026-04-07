"""
Feature engineering module for SPARC.

This module implements:
- Laplacian Eigenmaps for spatial feature extraction
- Fold-Aware Laplacian Eigenmaps (train/test separated, no leakage)
- Graph-based feature generation
- Spatial relationship encoding
"""

from .laplacian import LaplacianEigenmap
from .fold_aware_laplacian import FoldAwareLaplacianEigenmap, generate_fold_aware_laplacian

# V2 sinusoidal encoding (lazy — requires torch)
try:
    from .sinusoidal_encoding import SinusoidalSpatialEncoding
    _V2_ENCODING_AVAILABLE = True
except ImportError:
    _V2_ENCODING_AVAILABLE = False

__all__ = [
    'LaplacianEigenmap',
    'FoldAwareLaplacianEigenmap',
    'generate_fold_aware_laplacian',
    'SinusoidalSpatialEncoding',
]
