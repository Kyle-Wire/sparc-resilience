"""S4 — response surfaces: saturation, own-only and footprint effects.

For each actionable variable three complementary quantities are mapped:

1. **Neighbourhood-adoption curve** (saturation).  The whole study area
   adopts dose d (clipped to each cell's headroom); the model — with focal
   features recomputed and the physics re-solved — gives each cell's cooling
   B_i(d) = −ΔT_i(d).  Because caps make each cell's *own* realised dose
   differ, the curve is fitted against the realised *neighbourhood* dose
   D_i(d) = [K_r ∗ min(d, h)]_i (kernel of the variable's influence range):

       B_i(D) = A_i·(1 − exp(−D/d_s,i))

   by a vectorised grid search on d_s with closed-form A.  Reported: A (max
   cooling), d_s, d90 = d_s·ln 10 (only if within the tested doses, else the
   cell is *censored*: "no saturation within headroom"), the marginal benefit
   at today's level A/d_s, and headroom.  A linear fit is kept when the
   saturating form is not clearly better.  This answers "where does this
   intervention stop paying off if the neighbourhood adopts it".

2. **Own-only marginal** ∂ŷ_i/∂x_i — only cell i changes (its own column,
   its own contribution to its focal features, and the physics self-response
   a·G(0)·∂q_i/∂x_i).  This is the comparator for the doubly-robust causal
   dose-response (which assumes no interference).

3. **Footprint marginal** TE_i = Σ_j ∂ŷ_j/∂x_i — total cooling across the
   neighbourhood caused by changing cell i.  With focal features
   F_{s,j} = Σ_i K_s(j−i) m_i x_i / D_{s,j}:

       TE_i = s_own,i + Σ_s m_i·[K_s ∗ (g_s / D_s)]_i + a·(Gᵀ∗1_mask)_i·∂q_i/∂x_i

   where s_own and g_s are finite-difference derivatives of the (pointwise)
   stack w.r.t. the own column and each focal column.  This is "where to
   treat" — it feeds the budget optimiser.

Sweep curves summed over cells would double-count spillover; the optimiser
uses the footprint marginal instead.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparc.core import operators as ops
from sparc.core.features import build_context
from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_saturation(D: np.ndarray, B: np.ndarray, valid: np.ndarray, n_grid: int = 32, min_valid: int = 4,
                   improvement: float = 0.05, allow_sigmoid: bool = True) -> dict:
    """Per-column fit of the dose–benefit curve (arrays shaped doses × cells).

    Three shapes compete, each with closed-form amplitude A on a grid of its
    shape parameters:

    * linear      B = s·D
    * saturating  B = A·(1 − exp(−D/d_s))                     (diminishing returns)
    * sigmoid     B = A·[σ((D−D0)/w) − σ(−D0/w)] / [1 − σ(−D0/w)]
                  (accelerating, then saturating — e.g. canopy cooling that
                  rises sharply above ~40% cover, Ziter et al. 2019)

    A more complex shape is kept only if it lowers the SSE by ``improvement``
    relative to the simpler ones *and* has the lower AIC.  Returns arrays over
    cells: A, ds, inflection (D0), d90, r2, slope0 (marginal benefit at the
    current dose), model ("linear" | "saturating" | "sigmoid" | "insufficient"),
    censored.
    """
    D = np.where(valid, D, 0.0)
    B = np.where(valid, B, 0.0)
    n_valid = valid.sum(axis=0)
    dmax = np.where(valid, D, 0.0).max(axis=0)
    pos = dmax[dmax > 0]
    top = float(np.max(pos)) if pos.size else 1.0
    lo = max(top * 0.02, 1e-9)
    grid = np.geomspace(lo, 5.0 * top, n_grid)
    sst = np.sum(np.where(valid, (B - np.where(valid, B, 0).sum(0) / np.maximum(n_valid, 1)) ** 2, 0.0), axis=0)
    best_sse = np.full(D.shape[1], np.inf)
    best_A = np.zeros(D.shape[1])
    best_ds = np.full(D.shape[1], np.nan)
    for ds in grid:
        f = np.where(valid, 1.0 - np.exp(-D / ds), 0.0)
        ff = (f * f).sum(axis=0)
        A = np.where(ff > 0, (f * B).sum(axis=0) / np.maximum(ff, 1e-300), 0.0)
        sse = (np.where(valid, B - A * f, 0.0) ** 2).sum(axis=0)
        better = sse < best_sse
        best_sse = np.where(better, sse, best_sse)
        best_A = np.where(better, A, best_A)
        best_ds = np.where(better, ds, best_ds)
    dd = (D * D).sum(axis=0)
    lin_slope = np.where(dd > 0, (D * B).sum(axis=0) / np.maximum(dd, 1e-300), 0.0)
    lin_sse = (np.where(valid, B - lin_slope * D, 0.0) ** 2).sum(axis=0)

    enough = n_valid >= min_valid
    saturating = enough & (best_sse < (1.0 - improvement) * lin_sse)

    # sigmoid alternative (3 parameters)
    sg_sse = np.full(D.shape[1], np.inf)
    sg_A = np.zeros(D.shape[1])
    sg_D0 = np.full(D.shape[1], np.nan)
    sg_w = np.full(D.shape[1], np.nan)
    if allow_sigmoid:
        for D0 in np.linspace(0.15, 0.85, 8) * top:
            for w in np.array([0.04, 0.08, 0.15]) * top:
                s0 = _sig(-D0 / w)
                f = np.where(valid, (_sig((D - D0) / w) - s0) / (1.0 - s0), 0.0)
                ff = (f * f).sum(axis=0)
                A = np.where(ff > 0, (f * B).sum(axis=0) / np.maximum(ff, 1e-300), 0.0)
                sse = (np.where(valid, B - A * f, 0.0) ** 2).sum(axis=0)
                better = sse < sg_sse
                sg_sse, sg_A = np.where(better, sse, sg_sse), np.where(better, A, sg_A)
                sg_D0, sg_w = np.where(better, D0, sg_D0), np.where(better, w, sg_w)
    nv = np.maximum(n_valid, 1)

    def aic(sse, k):
        return nv * np.log(np.maximum(sse, 1e-300) / nv) + 2 * k

    simpler = np.where(saturating, best_sse, lin_sse)
    sigmoid = (enough & allow_sigmoid & (sg_sse < (1.0 - improvement) * simpler)
               & (aic(sg_sse, 3) < np.where(saturating, aic(best_sse, 2), aic(lin_sse, 1))))
    saturating = saturating & ~sigmoid

    censored = saturating & (best_ds >= grid[-2])                  # knee beyond the tested range
    d90 = best_ds * math.log(10.0)
    d90 = np.where(saturating & ~censored & (d90 <= dmax), d90, np.nan)
    # sigmoid: d90 solves f(d) = 0.9
    s0 = _sig(-sg_D0 / sg_w)
    p90 = 0.9 * (1.0 - s0) + s0
    with np.errstate(invalid="ignore", divide="ignore"):
        d90_sg = sg_D0 + sg_w * np.log(p90 / (1.0 - p90))
    sg_cens = sigmoid & ~(d90_sg <= dmax)
    censored = censored | sg_cens
    d90 = np.where(sigmoid & ~sg_cens, d90_sg, d90)
    slope_sg = sg_A * s0 * (1.0 - s0) / sg_w / (1.0 - s0)
    slope = np.where(sigmoid, slope_sg, np.where(saturating, best_A / best_ds, lin_slope))
    sse = np.where(sigmoid, sg_sse, np.where(saturating, best_sse, lin_sse))
    r2 = np.where(sst > 0, 1.0 - sse / np.maximum(sst, 1e-300), np.nan)
    model = np.where(~enough, "insufficient",
                     np.where(sigmoid, "sigmoid", np.where(saturating, "saturating", "linear")))
    return {
        "A": np.where(sigmoid, sg_A, np.where(saturating, best_A, np.nan)),
        "ds": np.where(saturating & ~censored, best_ds, np.nan),
        "inflection": np.where(sigmoid, sg_D0, np.nan),
        "d90": d90,
        "slope0": np.where(enough, slope, np.nan),
        "r2": np.where(enough, r2, np.nan),
        "model": model,
        "censored": censored,
        "n_valid": n_valid,
        "dmax": dmax,
    }


# ---------------------------------------------------------------------------
# Response engine
# ---------------------------------------------------------------------------


@dataclass
class VariableResponse:
    variable: str
    direction: str
    doses: list
    curve: pd.DataFrame                     # dose, mean benefit, fold sd, frac extrapolated
    maps: pd.DataFrame                      # per point
    summary: dict = field(default_factory=dict)
    own_pd: dict = field(default_factory=dict)


class ResponseEngine:
    def __init__(self, engine: ScenarioEngine, influence_scales=(0.5, 1.0, 2.0)):
        self.eng = engine
        self.data = engine.data
        self.cfg = engine.cfg
        self.ens = engine.ens
        self.scales = tuple(influence_scales)
        self.base_ctx = engine.base_ctx
        self._phys_base = self.ens.physics_predictions(self.base_ctx)

    # --------------------------------------------------------- utilities
    def _sigma_cells(self, var: str, scale: float = 1.0) -> float:
        r = self.eng.ranges_m.get(var)
        return 0.0 if not r else scale * r / 2.0 / self.data.grid.dx

    def _neigh_dose(self, own: np.ndarray, var: str) -> np.ndarray:
        g = self.data.grid
        sig = self._sigma_cells(var)
        if sig <= 0:
            return own
        r = ops.masked_gaussian(g.rasterize(own), g.mask, sig)
        return np.nan_to_num(g.sample(r))

    def _focal_cols(self, var: str) -> list[tuple[str, float]]:
        cols = []
        for s in self.scales:
            name = f"{var}__f{s:g}"
            if name in self.base_ctx.F.columns:
                cols.append((name, s))
        return cols

    def _edited(self, var: str, new_values: np.ndarray) -> pd.DataFrame:
        """Frame with ``var`` set to ``new_values`` and mediators updated by
        abduction (planting trees raises NDVI too)."""
        base = self.data.frame
        fr = base.copy()
        fr[var] = new_values
        if self.eng.mediators is not None:
            fr = self.eng.mediators.update(base, fr)
        return fr

    def _moved_columns(self, var: str, fr: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-point change of every raw column moved by an edit of ``var``
        (the variable itself plus its mediators)."""
        base = self.data.frame
        out = {}
        for c in fr.columns:
            d = fr[c].to_numpy(float) - base[c].to_numpy(float)
            if np.any(d != 0):
                out[c] = d
        out.setdefault(var, np.zeros(len(base)))
        return out

    def _physics_terms(self, fr: pd.DataFrame, delta: np.ndarray):
        """Per fold: a, G0, Gᵀ∗mask at points, and ∂q_i/∂x_i (incl. mediators)."""
        if not self.ens.has_physics:
            return None
        g = self.data.grid
        out = []
        for st in self.ens.stacks:
            pm = st.physics.model
            p = pm.params
            a = float(p["a"])
            L, v = float(p["L_m"]), (float(p.get("vx_m", 0.0)), float(p.get("vy_m", 0.0)))
            dq = (pm.source_points(fr) - pm.source_points(self.data.frame)) / delta
            g0 = ops.green_centre(L, v, g.dx, g.dy)
            gm = g.sample(ops.green_mass(g.mask.astype(float), L, v, g.dx, g.dy))
            out.append({"a": a, "g0": g0, "gmass": gm, "dq": dq})
        return out

    def _step(self, var: str, h: float) -> np.ndarray:
        lo, hi = self.eng._bounds(var)
        x = self.data.frame[var].to_numpy(float)
        return np.where(x + h <= hi, h, -h)

    # ------------------------------------------------------ marginals
    def marginals(self, var: str, h: float | None = None) -> dict:
        """Own-only and footprint marginal effects (per unit of ``var``),
        including the mediator pathway (e.g. canopy → NDVI → ΔT)."""
        data, g = self.data, self.data.grid
        x = data.frame[var].to_numpy(float)
        if h is None:
            h = 0.05 * (np.nanpercentile(x, 95) - np.nanpercentile(x, 5)) or 1e-3
        delta = self._step(var, h)
        base_fp = self.ens.fold_predictions(self.base_ctx, phys_override=self._phys_base)

        fr = self._edited(var, x + delta)
        moved = self._moved_columns(var, fr)          # {column: per-point change}
        # own columns only (focal + physics held fixed)
        ctx_own = build_context(fr, data, self.eng.ranges_m, self.cfg, F_override=self.base_ctx.F)
        s_own = (self.ens.fold_predictions(ctx_own, phys_override=self._phys_base) - base_fp) / delta

        own = s_own.copy()
        foot = s_own.copy()
        for colname, dcol in moved.items():
            ratio = dcol / delta                      # ∂col_i/∂var_i (1 for the variable itself)
            if not np.any(ratio):
                continue
            for col, s in self._focal_cols(colname):
                sig = self._sigma_cells(colname, s)
                Fv = self.base_ctx.F[col]
                hF = 0.05 * (np.nanpercentile(Fv, 95) - np.nanpercentile(Fv, 5)) or 1e-3
                F2 = self.base_ctx.F.copy()
                F2[col] = F2[col] + hF
                ctx_f = build_context(data.frame, data, self.eng.ranges_m, self.cfg, F_override=F2)
                gs = (self.ens.fold_predictions(ctx_f, phys_override=self._phys_base) - base_fp) / hF   # (K, n)
                Dn = g.sample(ops.gaussian_conv(g.mask.astype(float), sig))                         # normaliser
                w0 = ops.self_weight(sig)
                own += gs * (w0 / np.maximum(Dn, 1e-9)) * ratio
                for k in range(gs.shape[0]):
                    rr = g.rasterize(gs[k] / np.maximum(Dn, 1e-9), fill=0.0)
                    foot[k] += g.sample(ops.gaussian_conv(np.nan_to_num(rr), sig)) * ratio
        phys = self._physics_terms(fr, delta)
        if phys is not None:
            # Sensitivity of each fold stack to its physics input (1 in
            # backbone mode; learned in feature mode).
            hp = 0.05
            g_phys = (self.ens.fold_predictions(self.base_ctx, phys_override=self._phys_base + hp) - base_fp) / hp
            for k, pt in enumerate(phys):
                p = self.ens.stacks[k].physics.model.params
                L, v = float(p["L_m"]), (float(p.get("vx_m", 0.0)), float(p.get("vy_m", 0.0)))
                own[k] += g_phys[k] * pt["a"] * pt["g0"] * pt["dq"]
                # Σ_j g_j·G(j−i): adjoint solve of the sensitivity raster
                gr = g.rasterize(g_phys[k], fill=0.0)
                adj = g.sample(ops.solve(np.nan_to_num(gr), L, (-v[0], -v[1]), dx=g.dx, dy=g.dy))
                foot[k] += pt["a"] * adj * pt["dq"]
        return {
            "own": self.ens.decision(own), "own_sd": self.ens.jackknife_sd(own),
            "footprint": self.ens.decision(foot), "footprint_sd": self.ens.jackknife_sd(foot),
            "own_folds": own, "footprint_folds": foot,
            "mediators_moved": [c for c in moved if c != var],
        }

    def own_only_pd(self, var: str, t_grid: np.ndarray) -> dict:
        """E_i[ŷ_i(T_i = t)] with neighbours unchanged (own-only partial
        dependence; mediators follow the edited cell)."""
        data, g = self.data, self.data.grid
        phys = None
        if self.ens.has_physics:
            phys = [(st.physics.model, st.physics.model.params) for st in self.ens.stacks]
            q0 = [pm.source_points(data.frame) for pm, _ in phys]
        means, sds = [], []
        for t in t_grid:
            lo, hi = self.eng._bounds(var)
            fr = self._edited(var, np.full(data.n, float(np.clip(t, lo, hi))))
            moved = self._moved_columns(var, fr)
            F2 = self.base_ctx.F.copy()
            for colname, dcol in moved.items():
                for col, s in self._focal_cols(colname):
                    sig = self._sigma_cells(colname, s)
                    Dn = g.sample(ops.gaussian_conv(g.mask.astype(float), sig))
                    F2[col] = F2[col] + ops.self_weight(sig) / np.maximum(Dn, 1e-9) * dcol
            po = None
            if phys is not None:
                po = np.empty_like(self._phys_base)
                for k, (pm, p) in enumerate(phys):
                    g0 = ops.green_centre(float(p["L_m"]), (float(p.get("vx_m", 0)), float(p.get("vy_m", 0))), g.dx)
                    po[k] = self._phys_base[k] + float(p["a"]) * g0 * (pm.source_points(fr) - q0[k])
            ctx = self.eng.clip_ctx(build_context(fr, data, self.eng.ranges_m, self.cfg, F_override=F2))
            fp = self.ens.fold_predictions(ctx, phys_override=po)
            means.append(float(fp.mean()))
            sds.append(float(fp.mean(axis=1).std() * np.sqrt(max(fp.shape[0] - 1, 1))))
        return {"t": [float(v) for v in t_grid], "y": means, "se": sds}

    # ----------------------------------------------------------- sweeps
    def sweep(self, var: str) -> VariableResponse:
        spec = self.cfg.actionable[var]
        direction = str(spec.get("direction", "increase")).lower()
        sign = -1.0 if direction == "decrease" else 1.0
        doses = [float(d) for d in spec.get("doses", [0, 5, 10, 20, 30])]
        if doses[0] != 0.0:
            doses = [0.0] + doses
        n = self.data.n
        B = np.zeros((len(doses), n))
        Dn = np.zeros((len(doses), n))
        own = np.zeros((len(doses), n))
        rows = []
        for i, d in enumerate(doses):
            if d == 0.0:
                rows.append({"dose": 0.0, "mean_benefit": 0.0, "mean_se": 0.0, "frac_extrapolated": 0.0})
                continue
            res = self.eng.run(ScenarioSpec(name=f"{var} {d:g}", interventions=[Intervention(var, "add", sign * d)],
                                            dose=d, variable=var))
            realized = np.abs(res.realized[var])
            B[i] = -res.delta
            own[i] = realized
            Dn[i] = self._neigh_dose(realized, var)
            se = res.summary()["mean_delta_se"]
            rows.append({"dose": d, "mean_benefit": float(B[i].mean()), "mean_se": float(se if se is not None else 0.0),
                         "frac_extrapolated": float(np.mean(res.extrapolation > 1.0)),
                         "mean_realized_dose": float(realized.mean())})
        # A dose is valid for a cell while its own dose is not capped (headroom).
        valid = np.zeros_like(B, dtype=bool)
        valid[0] = True
        for i, d in enumerate(doses[1:], start=1):
            valid[i] = own[i] >= d - 1e-9
        fit = fit_saturation(Dn, B, valid, n_grid=int(self.cfg.raw["response"].get("ds_grid", 32)),
                             min_valid=int(self.cfg.raw["response"].get("min_valid_doses", 4)))
        lo, hi = self.eng._bounds(var)
        x = self.data.frame[var].to_numpy(float)
        headroom = (hi - x) if sign > 0 else (x - lo)
        m = self.marginals(var)
        maps = pd.DataFrame({
            "max_cooling_A": fit["A"],
            "saturation_scale_ds": fit["ds"],
            "inflection_dose": fit["inflection"],
            "d90": fit["d90"],
            "marginal_benefit_per_unit": fit["slope0"],
            "curve_model": fit["model"],
            "censored": fit["censored"],
            "fit_r2": fit["r2"],
            "headroom": headroom,
            # ∂ΔT/∂x per unit of the variable (sign as in the data; cooling < 0 for canopy)
            "own_effect_per_unit": m["own"],
            "own_effect_sd": m["own_sd"],
            "footprint_effect_per_unit": m["footprint"],
            "footprint_effect_sd": m["footprint_sd"],
        })
        sat = fit["model"] == "saturating"
        summary = {
            "variable": var,
            "direction": direction,
            "frac_saturating": float(np.mean(sat)),
            "frac_censored": float(np.mean(fit["censored"])),
            "frac_linear": float(np.mean(fit["model"] == "linear")),
            "frac_sigmoid": float(np.mean(fit["model"] == "sigmoid")),
            "median_d90": float(np.nanmedian(fit["d90"])) if np.isfinite(fit["d90"]).any() else None,
            "median_max_cooling": float(np.nanmedian(fit["A"])) if np.isfinite(fit["A"]).any() else None,
            "mean_own_effect": float(np.mean(m["own"])),
            "mean_footprint_effect": float(np.mean(m["footprint"])),
            "footprint_to_own_ratio": float(np.mean(m["footprint"]) / np.mean(m["own"])) if np.mean(m["own"]) else None,
            "own_fold_means": [float(v) for v in m["own_folds"].mean(axis=1)],
            "footprint_fold_means": [float(v) for v in m["footprint_folds"].mean(axis=1)],
        }
        return VariableResponse(variable=var, direction=direction, doses=doses, curve=pd.DataFrame(rows), maps=maps,
                                summary=summary, own_pd={})

    def model_effects(self, var: str, vr: VariableResponse, t_grid: np.ndarray | None = None) -> dict:
        """Model-implied effects in the form the causal audit expects
        (per unit of the variable, in the variable's natural sign).

        SEs are the spread of the population-mean slope across the K fold
        stacks — an epistemic proxy, not a sampling SE."""
        sign = -1.0 if vr.direction == "decrease" else 1.0
        d1 = next((d for d in vr.doses if d > 0), None)
        res = self.eng.run(ScenarioSpec(name=f"{var} slope", interventions=[Intervention(var, "add", sign * d1)]))
        realized = res.realized[var]
        ok = np.abs(realized) > 1e-9
        per_point_adopt = np.where(ok, res.delta / np.where(ok, realized, 1.0), np.nan)
        fold_slopes = [float(np.mean(res.delta_folds[k][ok] / realized[ok])) for k in range(res.delta_folds.shape[0])]
        own_fold_means = vr.summary.get("own_fold_means") or [float(np.mean(vr.maps["own_effect_per_unit"]))]
        out = {
            "adoption_slope": float(np.nanmean(per_point_adopt)),
            "adoption_se": float(np.std(fold_slopes)),
            "own_slope": float(np.mean(vr.maps["own_effect_per_unit"].to_numpy(float))),
            "own_se": float(np.std(own_fold_means)),
            "per_point_slope": per_point_adopt,
            "per_point_own_slope": vr.maps["own_effect_per_unit"].to_numpy(float),
        }
        if t_grid is not None and len(t_grid):
            out["own_pd_curve"] = self.own_only_pd(var, np.asarray(t_grid, dtype=float))
        return out
