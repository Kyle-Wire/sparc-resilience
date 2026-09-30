"""Feature assembly: encoded predictors + correlogram-scaled focal features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparc.core.base_models import FeatureContext
from sparc.core.config import CoreConfig
from sparc.core.data import CoreData, encode_features


def focal_columns(cfg: CoreConfig) -> list[str]:
    """Raw numeric predictors that get focal (neighbourhood) versions."""
    enc = cfg.raw.get("encodings") or {}
    skip = set(enc.get("categorical") or []) | set(enc.get("circular_degrees") or [])
    return [c for c in cfg.predictors if c not in skip]


def build_context(frame: pd.DataFrame, data: CoreData, ranges_m: dict[str, float], cfg: CoreConfig,
                  F_override: pd.DataFrame | None = None) -> FeatureContext:
    """Model inputs for a (possibly edited) predictor frame on ``data``'s grid.

    Focal features are recomputed from ``frame`` so edits propagate to
    neighbours (spillover); pass ``F_override`` to supply them directly
    (used for own-only perturbations).
    """
    X = encode_features(frame, cfg.raw.get("encodings"))
    if F_override is not None:
        F = F_override
    elif ranges_m:
        from sparc.core.influence import focal_features

        cols = [c for c in focal_columns(cfg) if c in ranges_m]
        F = focal_features(frame, data.grid, ranges_m, scales=tuple(cfg.raw["influence"].get("scales", (0.5, 1, 2))),
                           columns=cols)
        F = F.fillna(pd.Series({c: np.nanmean(F[c]) for c in F.columns}))
    else:
        F = pd.DataFrame(index=frame.index)
    return FeatureContext(frame=frame, X=X, F=F, coords=data.coords, grid=data.grid, meta={"y": data.y})
