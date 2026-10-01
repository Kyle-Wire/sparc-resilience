"""S5 — scenario engine: the trained model *is* the scenario engine.

A scenario edits the predictor frame (add / set / scale, optionally only
where a mask is true), clips to physical bounds, applies optional coupling
constraints *relative to the baseline*, propagates edits to mediators
(abduction), recomputes the correlogram-scaled focal features (so an edit
spills over to neighbours within each variable's area of influence),
re-runs the physics solve and the stacker, and reports ΔT against the
baseline prediction.

Uncertainty of a scenario delta is the spread of Δ across the K fold stacks
(differences cancel most in-sample fitting bias).  Every edited point also
gets a Mahalanobis extrapolation score (> 1 ⇒ outside the 95th percentile of
the training predictor distribution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparc.core.config import CoreConfig
from sparc.core.data import CoreData
from sparc.core.ensemble import FittedEnsemble
from sparc.core.features import build_context
from sparc.core.mediators import MediatorChain

log = logging.getLogger(__name__)


@dataclass
class Intervention:
    variable: str
    mode: str = "add"             # add | set | scale
    amount: float = 0.0
    where: np.ndarray | None = None
    per_point: np.ndarray | None = None   # explicit per-point amounts (optimizer allocations)


@dataclass
class ScenarioSpec:
    name: str
    interventions: list[Intervention]
    dose: float | None = None      # nominal dose for single-variable sweeps
    variable: str | None = None


@dataclass
class ScenarioResult:
    name: str
    baseline: np.ndarray
    scenario: np.ndarray
    delta: np.ndarray
    delta_sd: np.ndarray
    extrapolation: np.ndarray
    realized: dict = field(default_factory=dict)   # variable → realized per-point change
    frame: pd.DataFrame | None = None
    delta_folds: np.ndarray | None = None          # (K, n) per-fold deltas

    def summary(self) -> dict:
        d = self.delta
        se = None
        if self.delta_folds is not None and len(self.delta_folds) > 1:
            m = np.asarray(self.delta_folds, float).mean(axis=1)
            se = float(m.std() * np.sqrt(len(m) - 1))      # jackknife SE of the city-wide mean
        return {
            "name": self.name,
            "mean_delta": float(np.mean(d)),
            "mean_delta_se": se,
            "p10_delta": float(np.percentile(d, 10)),
            "p90_delta": float(np.percentile(d, 90)),
            "mean_delta_sd": float(np.mean(self.delta_sd)),
            "frac_extrapolated": float(np.mean(self.extrapolation > 1.0)),
            "mean_realized": {k: float(np.mean(v)) for k, v in self.realized.items()},
        }


def specs_from_config(cfg: CoreConfig) -> list[ScenarioSpec]:
    """Translate the legacy ``scenarios`` / ``joint_scenarios`` format."""
    specs = []
    for s in cfg.raw.get("scenarios") or []:
        sign = -1.0 if str(s.get("direction", "increase")).lower() == "decrease" else 1.0
        for inc in s.get("increments", []):
            specs.append(ScenarioSpec(name=f"{s['name']} {'+' if sign > 0 else '−'}{inc:g}",
                                      interventions=[Intervention(s["variable"], "add", sign * float(inc))],
                                      dose=float(inc), variable=s["variable"]))
    for j in cfg.raw.get("joint_scenarios") or []:
        ivs = []
        for iv in j.get("interventions", []):
            sign = -1.0 if str(iv.get("direction", "increase")).lower() == "decrease" else 1.0
            ivs.append(Intervention(iv["variable"], "add", sign * float(iv.get("increment", 0.0))))
        specs.append(ScenarioSpec(name=j["name"], interventions=ivs))
    return specs


class ScenarioEngine:
    """Scenario Δ against the fold stacks' baseline predictions.

    ``base_fold`` (K, n) is the baseline pass ``ensemble.fold_predictions``
    of the unedited context; pass a saved copy to skip that pass (it is the
    same array for the same run, data and code), read it back from
    ``self.base_fold``.
    """

    def __init__(self, data: CoreData, cfg: CoreConfig, ensemble: FittedEnsemble, ranges_m: dict[str, float],
                 mediators: MediatorChain | None = None, base_fold: np.ndarray | None = None):
        self.data = data
        self.cfg = cfg
        self.ens = ensemble
        self.ranges_m = ranges_m
        self.mediators = mediators
        self.base_ctx = build_context(data.frame, data, ranges_m, cfg)
        # Support of every statistical-model input: scenario inputs are clipped
        # to it (only the physics term — a structural model — extrapolates).
        self.clip_support = bool((cfg.raw.get("response") or {}).get("clip_to_support", True))
        XF = self.base_ctx.XF
        self._lo = XF.quantile(0.005)
        self._hi = XF.quantile(0.995)
        self._sd = XF.std().replace(0.0, 1.0)
        if base_fold is None:
            base_fold = ensemble.fold_predictions(self.base_ctx)
        else:
            base_fold = np.asarray(base_fold, dtype=float)
            if base_fold.shape != (len(ensemble.stacks), data.n):
                raise ValueError(f"base_fold has shape {base_fold.shape}; expected {(len(ensemble.stacks), data.n)}")
        self._base_fold = base_fold
        self.baseline = ensemble.honest(self._base_fold)
        Z = data.frame[cfg.predictors].to_numpy(float)
        self._mu = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        self._icov = np.linalg.pinv(cov)
        d0 = self._mahal(Z)
        self._d95 = float(np.percentile(d0, 95)) or 1.0

    @property
    def base_fold(self) -> np.ndarray:
        """(K, n) baseline predictions of every fold stack (the engine's baseline pass)."""
        return self._base_fold

    # ------------------------------------------------------------ helpers
    def _mahal(self, Z: np.ndarray) -> np.ndarray:
        D = Z - self._mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self._icov, D))

    def _bounds(self, var: str) -> tuple[float, float]:
        a = self.cfg.actionable.get(var, {})
        clip = (self.cfg.raw["qa"].get("clip") or {}).get(var)
        lo = a.get("min", clip[0] if clip else -np.inf)
        hi = a.get("max", clip[1] if clip else np.inf)
        return float(lo), float(hi)

    def apply(self, spec: ScenarioSpec) -> tuple[pd.DataFrame, dict]:
        base = self.data.frame
        new = base.copy()
        for iv in spec.interventions:
            x = new[iv.variable].to_numpy(float).copy()
            where = np.ones(len(x), dtype=bool) if iv.where is None else np.asarray(iv.where, dtype=bool)
            if iv.per_point is not None:
                x[where] = x[where] + np.asarray(iv.per_point, dtype=float)[where]
            elif iv.mode == "add":
                x[where] = x[where] + iv.amount
            elif iv.mode == "set":
                x[where] = iv.amount
            elif iv.mode == "scale":
                x[where] = x[where] * iv.amount
            else:
                raise ValueError(f"unknown intervention mode {iv.mode!r}")
            lo, hi = self._bounds(iv.variable)
            # Never push a baseline value that already sits outside the
            # bounds further out, and never "correct" it either.
            b = base[iv.variable].to_numpy(float)
            x = np.where(x > hi, np.maximum(hi, np.minimum(x, b)), x)
            x = np.where(x < lo, np.minimum(lo, np.maximum(x, b)), x)
            new[iv.variable] = x
        for c in self.cfg.raw.get("coupling") or []:
            cols, cap = c["sum"], float(c.get("max", 100.0))
            tot_new = new[cols].sum(axis=1).to_numpy(float)
            tot_base = base[cols].sum(axis=1).to_numpy(float)
            allowed = np.maximum(cap, tot_base)          # relative to the baseline
            excess = np.maximum(tot_new - allowed, 0.0)
            if excess.any():
                # Take the excess from the non-edited column(s) first.
                edited = {iv.variable for iv in spec.interventions}
                for col in [c2 for c2 in cols if c2 not in edited] + [c2 for c2 in cols if c2 in edited]:
                    v = new[col].to_numpy(float)
                    take = np.minimum(excess, np.maximum(v, 0.0))
                    new[col] = v - take
                    excess = excess - take
        if self.mediators is not None:
            new = self.mediators.update(base, new)
            for med in self.mediators.models:
                lo, hi = self._bounds(med)
                new[med] = np.clip(new[med].to_numpy(float), lo, hi)
        realized = {iv.variable: new[iv.variable].to_numpy(float) - base[iv.variable].to_numpy(float)
                    for iv in spec.interventions}
        return new, realized

    def scenario_context(self, frame: pd.DataFrame):
        """Model inputs for an edited frame; returns (ctx, support_exceedance).

        ``support_exceedance`` is, per point, the largest distance (in training
        SDs) by which any encoded or focal input left its 0.5–99.5 % training
        range — 0 inside the support."""
        ctx = build_context(frame, self.data, self.ranges_m, self.cfg)
        XF = ctx.XF
        bXF = self.base_ctx.XF[XF.columns]
        hi = np.maximum(self._hi[XF.columns].to_numpy()[None, :], bXF.to_numpy())
        lo = np.minimum(self._lo[XF.columns].to_numpy()[None, :], bXF.to_numpy())
        v = XF.to_numpy(float)
        over = pd.DataFrame((np.clip(v - hi, 0, None) + np.clip(lo - v, 0, None)) / self._sd[XF.columns].to_numpy()[None, :],
                            index=XF.index, columns=XF.columns)
        exceed = over.max(axis=1).to_numpy(float)
        return self.clip_ctx(ctx), exceed

    def clip_ctx(self, ctx):
        """Clip inputs to the training support, but never beyond a cell's own
        baseline value (so unedited cells are untouched: a zero edit is an
        exact identity)."""
        if self.clip_support:
            for attr in ("X", "F"):
                cur = getattr(ctx, attr)
                if cur is None or cur.shape[1] == 0:
                    continue
                base = getattr(self.base_ctx, attr)[cur.columns]
                lo = np.minimum(self._lo[cur.columns].to_numpy()[None, :], base.to_numpy())
                hi = np.maximum(self._hi[cur.columns].to_numpy()[None, :], base.to_numpy())
                setattr(ctx, attr, pd.DataFrame(np.clip(cur.to_numpy(float), lo, hi), index=cur.index,
                                                columns=cur.columns))
        return ctx

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        ctx, _ = self.scenario_context(frame)
        # Physics sees the unclipped frame (ctx.frame is the edited frame).
        return self.ens.fold_predictions(ctx)

    def run(self, spec: ScenarioSpec, keep_frame: bool = False) -> ScenarioResult:
        new, realized = self.apply(spec)
        ctx, exceed = self.scenario_context(new)
        fold = self.ens.fold_predictions(ctx)
        d_fold = fold - self._base_fold
        delta = self.ens.decision(d_fold)                 # fold-averaged: no seams, no fold-specific gains
        scen = self.baseline + delta
        extr = self._mahal(new[self.cfg.predictors].to_numpy(float)) / self._d95
        changed = np.zeros(len(new), dtype=bool)
        for v in realized.values():
            changed |= np.abs(v) > 0
        extr = np.where(changed, np.maximum(extr, np.where(exceed > 0, 1.0 + exceed, 0.0)), 0.0)
        return ScenarioResult(name=spec.name, baseline=self.baseline, scenario=scen, delta=delta,
                              delta_sd=self.ens.jackknife_sd(d_fold), extrapolation=extr, realized=realized,
                              frame=new if keep_frame else None, delta_folds=d_fold)
