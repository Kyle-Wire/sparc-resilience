"""Mediator chain for scenarios.

Some predictors are *consequences* of the actionable ones — NDVI rises when
canopy is added.  Holding them fixed in a scenario would understate (or
misattribute) the effect.  Each mediator M gets a monotone gradient-boosting
model m(parents, context), and scenarios update it by *abduction*:

    M' = M + m(parents', context) − m(parents, context)

so the observed M (including its idiosyncratic part) is kept and only the
change implied by the edited parents is added.  A zero edit therefore leaves
M exactly unchanged.  The models use predictors only — no target labels — so
fitting them on all rows does not leak.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class MediatorChain:
    def __init__(self, spec: dict | None, seed: int = 0):
        self.spec = dict(spec or {})
        self.seed = seed
        self.models: dict = {}

    def fit(self, frame: pd.DataFrame) -> "MediatorChain":
        from sklearn.ensemble import HistGradientBoostingRegressor

        for med, s in self.spec.items():
            parents = list(s.get("parents", []))
            context = [c for c in s.get("context", []) if c in frame.columns]
            cols = parents + context
            mono = s.get("monotone") or {}
            cst = [int(mono.get(c, 0)) for c in cols]
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, min_samples_leaf=30,
                                              monotonic_cst=cst, random_state=self.seed)
            m.fit(frame[cols].to_numpy(float), frame[med].to_numpy(float))
            r2 = float(m.score(frame[cols].to_numpy(float), frame[med].to_numpy(float)))
            self.models[med] = (m, cols, parents)
            log.info("mediator %s ← %s (in-sample R² %.2f)", med, parents, r2)
        return self

    def update(self, base: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
        """Propagate parent edits to mediators (abduction); returns a copy."""
        out = edited.copy()
        for med, (m, cols, parents) in self.models.items():
            changed = np.zeros(len(base), dtype=bool)
            for p in parents:
                changed |= ~np.isclose(base[p].to_numpy(float), edited[p].to_numpy(float))
            if not changed.any():
                continue
            b = m.predict(base.loc[changed, cols].to_numpy(float))
            e = m.predict(edited.loc[changed, cols].to_numpy(float))
            vals = out[med].to_numpy(float).copy()
            vals[changed] = base[med].to_numpy(float)[changed] + (e - b)
            out[med] = vals
        return out

    def summary(self) -> dict:
        return {med: {"parents": parents, "inputs": cols} for med, (_m, cols, parents) in self.models.items()}
