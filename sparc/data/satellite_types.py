"""
Satellite feature type definitions for SPARC V4 zero-shot inference.

Defines the schema for satellite-derived feature sets that will be used
in V4 to enable zero-shot prediction for unseen cities using only
remote sensing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SatelliteBand(Enum):
    """Common satellite band types."""
    NDVI = "ndvi"
    NDBI = "ndbi"
    NDWI = "ndwi"
    LST = "lst"
    ALBEDO = "albedo"
    EMISSIVITY = "emissivity"
    ISA = "impervious_surface_area"
    TREE_COVER = "tree_cover"
    BUILDING_HEIGHT = "building_height"
    SVF = "sky_view_factor"
    CUSTOM = "custom"


@dataclass
class SatelliteFeatureSet:
    """
    Container for satellite-derived feature arrays.

    This interface will be used in V4 to pass satellite observations
    into the zero-shot inference pipeline.

    Attributes
    ----------
    coords : (N, 2) longitude/latitude coordinates
    bands : dict mapping SatelliteBand → (N,) value arrays
    resolution_m : spatial resolution in metres
    acquisition_date : ISO date string (YYYY-MM-DD)
    crs : coordinate reference system (e.g. "EPSG:4326")
    city_name : optional city identifier
    climate_zone : optional Köppen climate zone string (e.g. "Cfa")
    """

    coords: np.ndarray  # (N, 2)
    bands: dict[SatelliteBand, np.ndarray] = field(default_factory=dict)
    resolution_m: float = 30.0
    acquisition_date: str | None = None
    crs: str = "EPSG:4326"
    city_name: str | None = None
    climate_zone: str | None = None

    @property
    def n_points(self) -> int:
        return self.coords.shape[0]

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    def to_feature_matrix(self) -> np.ndarray:
        """
        Stack all band arrays into an (N, B) feature matrix.

        Band order follows SatelliteBand enum order for reproducibility.
        """
        if not self.bands:
            return np.empty((self.n_points, 0))
        ordered = sorted(self.bands.items(), key=lambda kv: kv[0].value)
        return np.column_stack([arr for _, arr in ordered])

    def band_names(self) -> list[str]:
        """Return sorted list of band name strings."""
        return sorted(b.value for b in self.bands)
