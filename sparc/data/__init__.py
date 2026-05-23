"""
Data handling module for SPARC.

This module provides utilities for:
- Data loading
- Cleaning
- Type conversion
- CRS projection
"""

def load_and_preprocess_data(*args, **kwargs):
    """Lazy wrapper — defers pandas import until first call."""
    from .data_utils import load_and_preprocess_data as _impl
    return _impl(*args, **kwargs)


__all__ = ['load_and_preprocess_data']
