"""ConfigGateway — unified facade for all SPARC configuration sources.

Collects load_config, HardwareProfile detection, causal defaults, and
SPARCConfig loading behind a single import point so callers don't need to know
which sub-module owns each concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from sparc.config.hardware_profile import HardwareProfile
    from sparc.config.sparc_config import SPARCConfig


class ConfigGateway:
    """Unified facade for SPARC configuration sources.

    All methods are static — no instance required.
    """

    @staticmethod
    def load_project(path: "str | Path") -> dict:
        """Load a project YAML file and return the raw config dict.

        Wraps :func:`sparc.config.load_config`.
        """
        from sparc.config.config import load_config  # private module
        return load_config(str(path))

    @staticmethod
    def get_hardware_profile() -> "HardwareProfile":
        """Detect and return the current :class:`~sparc.config.hardware_profile.HardwareProfile`.

        Wraps :func:`sparc.config.hardware_profile.detect_profile`.
        """
        from sparc.config.hardware_profile import detect_profile
        return detect_profile()

    @staticmethod
    def get_causal_defaults() -> dict:
        """Return the SPARC causal defaults dict.

        Wraps :data:`sparc.config.causal_defaults.CAUSAL_DEFAULTS`.
        """
        from sparc.config.causal_defaults import CAUSAL_DEFAULTS
        return CAUSAL_DEFAULTS

    @staticmethod
    def get_sparc_config(path: "str | Path") -> "SPARCConfig":
        """Load a project YAML file as a typed :class:`~sparc.config.sparc_config.SPARCConfig`.

        Wraps :meth:`sparc.config.sparc_config.SPARCConfig.load`.
        """
        from sparc.config.sparc_config import SPARCConfig
        return SPARCConfig.load(str(path))
