"""JEPATrunkAdapter — wraps a PI-JEPA trunk checkpoint for SpatialResidualizer.

Bridges the PI-JEPA trunk (trained in train_multicity_jepa.py) to the
``model.encode(physics_t, alpha_t) -> Tensor(N, hidden_dim)`` interface
expected by :class:`~sparc.causal.spatial_residualizer.SpatialResidualizer`.

Key responsibilities:
- Loads and reconstructs the trunk MLP from the checkpoint dict
- Applies per-feature normalization using the stored X_mean/X_std
- Aligns input columns by name to the trunk's training feature order
  (case-insensitive, with zero-fill for missing columns)
- Exposes ``feature_names`` so callers can query which columns to pass

Checkpoint format (produced by train_multicity_jepa.py)::

    {
        'trunk_state': OrderedDict,   # MLP state dict
        'in_dim':      int,           # number of input features (18)
        'hidden_dim':  int,           # trunk hidden dimension (256)
        'X_mean':      list[float],   # per-feature training mean
        'X_std':       list[float],   # per-feature training std  (+1e-6)
        'C_mean':      list[float],   # coordinate mean (unused here)
        'C_std':       list[float],   # coordinate std  (unused here)
        'feat_cols':   list[str],     # OPTIONAL: training feature names
    }

If ``feat_cols`` is absent from the checkpoint the adapter falls back to
the canonical 18-column order from ``configs/multicity_pilot.yml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Canonical feature order used during PI-JEPA pretraining.
# Matches configs/multicity_pilot.yml → multicity.feature_columns.land_surface
# plus era5_morning/midday/evening.  Keep in sync with that config.
_CANONICAL_FEAT_COLS: List[str] = [
    "pct_impervious", "pct_canopy", "land_cover", "ndvi", "ndbi",
    "lst", "ndvi_s2", "ndbi_s2", "mndwi_s2", "albedo",
    "bldg_height_mean", "bldg_coverage", "svf", "cdc_svi",
    "elevation_m", "slope_deg", "aspect_deg", "distance_water_m",
    "era5_morning_t2m", "era5_morning_windspeed",
    "era5_morning_winddir", "era5_morning_rh",
    "era5_midday_t2m", "era5_midday_windspeed",
    "era5_midday_winddir", "era5_midday_rh",
    "era5_evening_t2m", "era5_evening_windspeed",
    "era5_evening_winddir", "era5_evening_rh",
]

# Alternative spellings seen in legacy SPARC configs (title-case / mixed)
_ALIAS_MAP: dict[str, str] = {
    "pct_impervious": "pct_impervious",
    "pctimpervious": "pct_impervious",
    "impervious": "pct_impervious",
    "pct_canopy": "pct_canopy",
    "pctcanopy": "pct_canopy",
    "canopy": "pct_canopy",
    "albedo": "albedo",
    "ndvi": "ndvi",
    "ndbi": "ndbi",
    "lst": "lst",
    "svf": "svf",
    "cdc_svi": "cdc_svi",
    "elevation_m": "elevation_m",
    "elevation": "elevation_m",
    "slope_deg": "slope_deg",
    "slope": "slope_deg",
    "bldg_height_mean": "bldg_height_mean",
    "bldg_coverage": "bldg_coverage",
}


def _normalize_col(name: str) -> str:
    """Lowercase + strip alias resolution for column matching."""
    lc = name.lower().strip()
    return _ALIAS_MAP.get(lc, lc)


class JEPATrunkAdapter:
    """Wrap a PI-JEPA trunk checkpoint for use with SpatialResidualizer.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the ``.pt`` file produced by ``run_jepa_pretraining()``.
    device : str
        PyTorch device string (default ``"cpu"``).
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        import torch
        self._device = torch.device(device)
        ckpt = torch.load(str(checkpoint_path), map_location=self._device, weights_only=False)

        self._in_dim    = int(ckpt["in_dim"])
        self._hidden    = int(ckpt["hidden_dim"])
        self._X_mean    = np.array(ckpt["X_mean"], dtype=np.float32)
        self._X_std     = np.array(ckpt["X_std"],  dtype=np.float32)

        # Feature names — use checkpoint's stored list if present, else canonical
        self._feat_cols: List[str] = list(
            ckpt.get("feat_cols", _CANONICAL_FEAT_COLS[: self._in_dim])
        )
        # Lowercase lookup: canonical name → column index in trunk's training data
        self._col_to_idx: dict[str, int] = {
            _normalize_col(c): i for i, c in enumerate(self._feat_cols)
        }

        # Reconstruct trunk MLP (must match _build_trunk in train_multicity_jepa.py)
        # Architecture: Linear(in_dim, hidden) → GELU → Linear(hidden, hidden) → GELU → Linear(hidden, hidden)
        from torch import nn
        self._trunk = nn.Sequential(
            nn.Linear(self._in_dim, self._hidden),
            nn.GELU(),
            nn.Linear(self._hidden, self._hidden),
            nn.GELU(),
            nn.Linear(self._hidden, self._hidden),
        ).to(self._device)
        self._trunk.load_state_dict(ckpt["trunk_state"])
        self._trunk.eval()

        log.info(
            "JEPATrunkAdapter loaded: in_dim=%d  hidden=%d  feat_cols=%d  device=%s",
            self._in_dim, self._hidden, len(self._feat_cols), device,
        )

    # ------------------------------------------------------------------
    @property
    def feature_names(self) -> List[str]:
        """Canonical training feature names for this trunk (in training order)."""
        return list(self._feat_cols)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    # ------------------------------------------------------------------
    def to(self, device) -> "JEPATrunkAdapter":
        import torch
        self._device = torch.device(device)
        self._trunk  = self._trunk.to(self._device)
        return self

    def eval(self) -> "JEPATrunkAdapter":
        self._trunk.eval()
        return self

    # ------------------------------------------------------------------
    def encode(self, physics_t, alpha_t=None) -> "torch.Tensor":
        """Encode a raw feature tensor into trunk embeddings.

        Parameters
        ----------
        physics_t : Tensor of shape ``(N, P)``
            Raw (un-normalized) feature values.  Columns must correspond to
            the features in ``feature_names`` in the same order.  If P < in_dim,
            missing columns are zero-filled (mean-centered equivalent).
        alpha_t : Tensor or None
            Process-rate gating — ignored by the PI-JEPA trunk (accepted for
            interface compatibility with SPARCMetaLearner).

        Returns
        -------
        Tensor of shape ``(N, hidden_dim)``
        """
        import torch

        x_np = physics_t.detach().cpu().numpy().astype(np.float32)
        N, P = x_np.shape

        # Expand / align to in_dim columns if needed
        if P != self._in_dim:
            x_full = np.zeros((N, self._in_dim), dtype=np.float32)
            x_full[:, :P] = x_np
        else:
            x_full = x_np

        # Normalize using training statistics
        x_norm = (x_full - self._X_mean) / self._X_std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

        x_t = torch.tensor(x_norm, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            h = self._trunk(x_t)
        return h

    # ------------------------------------------------------------------
    def encode_by_names(
        self,
        X_raw: np.ndarray,
        col_names: Sequence[str],
    ) -> np.ndarray:
        """Encode a numpy array where columns are identified by name.

        Handles column-name alignment: incoming columns are matched by
        normalized name to the trunk's training feature order.  Missing
        columns are zero-filled (equivalent to mean-centered).

        Parameters
        ----------
        X_raw : (N, P) float array — raw (un-normalized) feature values
        col_names : sequence of P column names

        Returns
        -------
        h_np : (N, hidden_dim) float32 array
        """
        import torch

        N = X_raw.shape[0]
        x_full = np.zeros((N, self._in_dim), dtype=np.float32)

        for j, name in enumerate(col_names):
            key = _normalize_col(name)
            idx = self._col_to_idx.get(key)
            if idx is not None and j < X_raw.shape[1]:
                x_full[:, idx] = X_raw[:, j].astype(np.float32)
            # else: column not in trunk training set → stays zero (mean-centered)

        x_norm = (x_full - self._X_mean) / self._X_std
        x_norm = np.nan_to_num(x_norm, nan=0.0)

        x_t = torch.tensor(x_norm, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            h = self._trunk(x_t)
        return h.detach().cpu().numpy()


def load_jepa_trunk_adapter(
    output_root: str | Path,
    device: str = "cpu",
) -> "JEPATrunkAdapter | None":
    """Try to load a PI-JEPA trunk adapter from the standard output locations.

    Searches in order:
    1. ``output_root/jepa_pretrained_trunk.pt``          (latest trained trunk)
    2. ``output_root/cities/jepa_pretrained_trunk_t66_best.pt``  (NaN-fix baseline)

    Returns ``None`` if no checkpoint is found so callers can degrade gracefully.
    """
    root = Path(output_root)
    candidates = [
        root / "jepa_pretrained_trunk.pt",
        root / "cities" / "jepa_pretrained_trunk_t66_best.pt",
    ]
    for path in candidates:
        if path.exists():
            try:
                adapter = JEPATrunkAdapter(path, device=device)
                log.info("Loaded JEPATrunkAdapter from %s", path)
                return adapter
            except Exception as exc:
                log.warning("Failed to load JEPA trunk from %s: %s", path, exc)
    log.info("No PI-JEPA trunk checkpoint found in %s — SpatialResidualizer will use neural_meta.pt", root)
    return None
