"""
Feature engineering module for SPARC.

Active in V2:
  - SinusoidalSpatialEncoding  — replaces Laplacian eigenmaps for V2 training
  - LaplacianEigenmap          — V1 baseline; used by enhanced_spatial_cv
  - geometry                   — KNN / cardinal-neighbour utilities (shared across pipeline)

Inactive / V1-only:
  - FoldAwareLaplacianEigenmap / generate_fold_aware_laplacian — V1 fold-separated
    eigenmaps.  No callers in V2.  Available as ``sparc.features.fold_aware_laplacian``
    for direct import if needed; not re-exported here to keep the public interface clean.
"""

from .laplacian import LaplacianEigenmap

# V2 sinusoidal encoding (lazy — requires torch)
try:
    from .sinusoidal_encoding import SinusoidalSpatialEncoding
    _V2_ENCODING_AVAILABLE = True
except ImportError:
    _V2_ENCODING_AVAILABLE = False

__all__ = [
    'LaplacianEigenmap',
    'SinusoidalSpatialEncoding',
]
