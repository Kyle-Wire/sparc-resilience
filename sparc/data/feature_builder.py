"""SpatialFeatureBuilder — constructs SatelliteFeatureSet from real data sources.

Gives the SatelliteFeatureSet schema a real entry point so that tests,
the feature pipeline, and callers can build valid instances without
touching HTTP or the training loop.

Currently supported builders
-----------------------------
from_tabular(df, coords)
    Build from a pandas DataFrame where column names match recognised
    SatelliteBand names (case-insensitive).  Unrecognised columns are
    stored under SatelliteBand.CUSTOM (merged into a single array).

Planned (not yet implemented — will raise NotImplementedError)
--------------------------------------------------------------
from_raster(path, coords)
from_geojson(path)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparc.data.satellite_types import SatelliteBand, SatelliteFeatureSet

# Map lowercase column name → SatelliteBand enum member.
# Add entries here as new standard band names are agreed.
_COLUMN_TO_BAND: dict[str, SatelliteBand] = {
    "ndvi":            SatelliteBand.NDVI,
    "ndbi":            SatelliteBand.NDBI,
    "ndwi":            SatelliteBand.NDWI,
    "lst":             SatelliteBand.LST,
    "albedo":          SatelliteBand.ALBEDO,
    "emissivity":      SatelliteBand.EMISSIVITY,
    "isa":             SatelliteBand.ISA,
    "impervious_surface_area": SatelliteBand.ISA,
    "tree_cover":      SatelliteBand.TREE_COVER,
    "building_height": SatelliteBand.BUILDING_HEIGHT,
    "svf":             SatelliteBand.SVF,
    "sky_view_factor": SatelliteBand.SVF,
}


class SpatialFeatureBuilder:
    """Factory for SatelliteFeatureSet instances."""

    @staticmethod
    def from_tabular(
        df: pd.DataFrame,
        coords: np.ndarray,
        resolution_m: float = 30.0,
        crs: str = "EPSG:4326",
    ) -> SatelliteFeatureSet:
        """Build a SatelliteFeatureSet from a pandas DataFrame.

        Parameters
        ----------
        df         : DataFrame where each column is a feature band.
                     Column names are matched case-insensitively against
                     known SatelliteBand names.
        coords     : (N, 2) array of lon/lat coordinates.
        resolution_m : spatial resolution in metres (default 30).
        crs        : coordinate reference system string (default EPSG:4326).

        Returns
        -------
        SatelliteFeatureSet

        Raises
        ------
        ValueError
            If *df* has no columns or the number of rows in *df* does not
            match the number of rows in *coords*.
        """
        if df.empty or len(df.columns) == 0:
            raise ValueError(
                "SpatialFeatureBuilder.from_tabular: empty DataFrame or no columns."
            )

        coords_arr = np.asarray(coords)
        n_points = len(df)

        if coords_arr.shape[0] != n_points:
            raise ValueError(
                f"SpatialFeatureBuilder.from_tabular: coord rows ({coords_arr.shape[0]}) "
                f"do not match DataFrame rows ({n_points})."
            )

        bands: dict[SatelliteBand, np.ndarray] = {}
        custom_arrays: list[np.ndarray] = []

        for col in df.columns:
            band = _COLUMN_TO_BAND.get(col.lower())
            arr = df[col].to_numpy(dtype=np.float32)
            if band is not None:
                bands[band] = arr
            else:
                custom_arrays.append(arr)

        # Merge all unrecognised columns into a single CUSTOM array
        # (mean across custom columns keeps shape (N,))
        if custom_arrays:
            bands[SatelliteBand.CUSTOM] = np.column_stack(custom_arrays).mean(axis=1).astype(np.float32)

        return SatelliteFeatureSet(
            coords=coords_arr,
            bands=bands,
            resolution_m=resolution_m,
            crs=crs,
        )

    @staticmethod
    def from_raster(path: str, coords: np.ndarray) -> SatelliteFeatureSet:
        """Build from a GeoTIFF / raster file. Not yet implemented."""
        raise NotImplementedError(
            "SpatialFeatureBuilder.from_raster is not yet implemented."
        )

    @staticmethod
    def from_geojson(path: str) -> SatelliteFeatureSet:
        """Build from a GeoJSON file. Not yet implemented."""
        raise NotImplementedError(
            "SpatialFeatureBuilder.from_geojson is not yet implemented."
        )
