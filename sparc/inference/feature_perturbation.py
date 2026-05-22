"""JD-5: Physics-informed feature perturbation for re-encode latent rollout.

``PhysicsCascade`` encodes domain knowledge about how a primary treatment
variable propagates to correlated feature variables.  When used with
``multi_step_latent_rollout(mode='reencode')``, the physics features are
updated at each step before re-encoding through the JEPA trunk, so the
latent state reflects physically plausible downstream effects.

Example cascade config (UHI template)::

    treatment_cascades:
      pct_canopy:
        ndvi:    +0.80
        albedo:  -0.05
        et_flux: +0.30

Cascade factors are linear: ``feature += scale_factor * delta_x``.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


class PhysicsCascade:
    """
    Linear physics-cascade for treatment-induced feature co-perturbation.

    Parameters
    ----------
    cascade_config : dict
        Mapping of ``treatment_name → {feature_col: scale_factor}``.
        Treatment names and feature column names are matched
        case-insensitively.
    """

    def __init__(self, cascade_config: Dict[str, Dict[str, float]]) -> None:
        # Normalise keys to lower-case for case-insensitive matching
        self._cascades: Dict[str, Dict[str, float]] = {
            k.lower(): {fk.lower(): float(fv) for fk, fv in v.items()}
            for k, v in cascade_config.items()
        }

    @classmethod
    def from_caps(cls, caps_path: str) -> "PhysicsCascade":
        """Load cascade config from a ``caps.yml`` / ``project.yml`` file.

        Reads the ``treatment_cascades:`` top-level key.

        Parameters
        ----------
        caps_path : str
            Path to a YAML file containing a ``treatment_cascades:`` section.

        Returns
        -------
        PhysicsCascade
        """
        import yaml

        with open(caps_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        cascades = raw.get("treatment_cascades", {}) or {}
        return cls(cascades)

    def apply(
        self,
        df: pd.DataFrame,
        treatment: str,
        delta_x: float,
    ) -> pd.DataFrame:
        """
        Return a copy of *df* with the treatment and correlated features perturbed.

        The treatment column is incremented by ``delta_x``.  For each feature
        listed under ``treatment`` in the cascade config, the feature column is
        incremented by ``scale_factor * delta_x``.

        Columns not present in *df* are silently skipped.

        Parameters
        ----------
        df : pd.DataFrame
        treatment : str
            Name of the treatment column (must exist in *df*).
        delta_x : float
            Signed magnitude of the intervention.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with perturbed values.
        """
        df_out = df.copy()

        # Perturb the treatment column itself
        treatment_lower = treatment.lower()
        if treatment in df_out.columns:
            df_out[treatment] = df_out[treatment] + delta_x
        elif treatment_lower in df_out.columns:
            df_out[treatment_lower] = df_out[treatment_lower] + delta_x

        # Cascade to correlated features
        correlated = self._cascades.get(treatment_lower, {})
        for feat, scale in correlated.items():
            # Try exact case first, then lower-cased
            if feat in df_out.columns:
                df_out[feat] = df_out[feat] + scale * delta_x
            elif feat.lower() in df_out.columns:
                col = feat.lower()
                df_out[col] = df_out[col] + scale * delta_x
            # else: column absent in df — skip silently

        return df_out
