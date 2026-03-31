"""
Data handling module for SPARC.

This module provides utilities for:
- Data loading
- Cleaning
- Type conversion
- CRS projection
"""

from .data_utils import load_and_preprocess_data

__all__ = ['load_and_preprocess_data']
