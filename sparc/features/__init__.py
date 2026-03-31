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

__all__ = ['LaplacianEigenmap', 'FoldAwareLaplacianEigenmap', 'generate_fold_aware_laplacian']
