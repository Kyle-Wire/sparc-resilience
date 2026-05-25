"""
Configuration module for SPARC.

This module handles all configuration management, including:
- File paths
- Model parameters
- Constants
- Feature lists
"""

from .config import load_config
from .project_config import ProjectConfig

__all__ = ['load_config', 'ProjectConfig']
