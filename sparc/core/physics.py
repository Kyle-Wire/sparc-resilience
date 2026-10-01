"""Physics base model (roadmap §2.1): a linearised canopy-layer heat budget.

The anomaly obeys the steady advection–diffusion–relaxation equation of
:mod:`sparc.core.operators`,

    (1 − L²∇² + v·∇) φ = q,        ΔT_phys = a·φ + b + γ·elev_c + βx·x_c + βy·y_c

solved spectrally with the *same* discrete operator (5-point Laplacian,
central-difference advection, Ĝ = 1/(1 + L²λ + i·adv)), so the PDE residual
used by the stacker (:meth:`PhysicsModel.operator_residual_torch`) is exactly
zero on this model's own prediction.

Source (normalised sensible heat, per point, from the configured roles)
----------------------------------------------------------------------
Day window (SW = sw_down, LW = lw_net)::

    Q*   = SW·(1 − albedo)·(1 − s·h(canopy/100)) + LW
    h(c) = (1 − e^{−c/κ}) / (1 − e^{−1/κ})            (default: saturating; κ → ∞ is linear)
    h(c) = [σ((c−c0)/w) − σ(−c0/w)] / [σ((1−c0)/w) − σ(−c0/w)]   (shade_form: sigmoid —
           accelerating above c0, e.g. canopy that cools sharply above ~40% cover)
    EF   = sigmoid(e0 + e1⁺·ndvi_c + e2⁺·canopy_c/100 − e3⁺·imp_c/100)    (x⁺ = softplus)
    ΔQ_S = a1·(imp/100)·Q*
    Q_H  = (Q* − ΔQ_S)·(1 − EF) − w⁺·1[water_distance ≤ dx/2]
    q    = Q_H / SW

Night window: SW = 0, Q* = LW, ΔQ_S = −a1n·(imp/100)·|LW| (stored-heat
release) and q = Q_H/|LW|.  Any missing role drops its term (albedo falls
back to a constant 0.15).  ``_c`` means centred with constants fixed on the
fitting frame.

The point source q_raw is averaged into raster cells, centred by its mean
over valid cells *of the fitting frame* and set to 0 outside the mask
(Ĝ(0) = 1, so the constant is absorbed by b).  The centring constant is
frozen at fit time so an edited frame (scenario) with a uniform canopy
increase still changes q everywhere.

Identifiability and fitting
---------------------------
* s, a1, a1n are learnable but held near literature values (0.6, 0.3, 0.5)
  by Gaussian penalties; the shade-saturation scale κ ∈ [0.05, 20] has a weak
  log-normal prior (log κ ~ N(0, 3²); h_1 is mildly concave, large κ is linear); e1..e3 ≥ 0 via softplus; e0 starts at logit(0.35).
* a, b, γ, βx, βy are profiled out by variable projection: for the current
  nonlinear parameters the linear least-squares problem on the training
  points is solved inside the (differentiable) loss.
* log L ∈ [log dx, log min(L_max, extent/4)] through a sigmoid; v = v_max·u
  with penalty |u|² = (|v|/v_max)²; multi-start over v ∈ {0, (±v_max/2, 0),
  (0, ±v_max/2)} (plus wind·τ if configured) when advection is fit: each
  start is probed for ``start_iter`` iterations and the lowest-loss one is
  continued for ``max_iter``.
* torch L-BFGS (strong Wolfe), float64, fixed zero padding
  ``pad_cells(L_max, v_max, dx)`` for the whole fit; torch threads are
  capped at ``num_threads`` (default 1) during fit/predict.

Only training labels enter the loss; q uses every cell's *features*, which
is legitimate (features are not labels), so held-out predictions come from
a model fitted on the training fold alone.

Marginal effects: :meth:`PhysicsModel.source_points` returns the pointwise,
uncentred q_raw (finite-differenceable per point) and
:meth:`PhysicsModel.source_raster` its cell mean minus the fit-frame
valid-cell mean, so ∂ΔT/∂x_i = a·(G ⊛ ∂q/∂x)_i with G the fitted operator.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from sparc.core import operators as ops
from sparc.core.grid import Grid

log = logging.getLogger(__name__)

PHYSICS_DEFAULTS: dict[str, Any] = {
    "window": "day",
    "sw_down": 800.0,
    "lw_net": -100.0,
    "roles": {},
    "wind": None,
    # Optional linear rescale of a non-broadband albedo layer onto broadband
    # values, e.g. {from: auto, to: [0.08, 0.25]} (see S0 QA flags).
    "albedo_map": None,
    "tau_s": 1800.0,
    "L_max_m": 2000.0,
    "v_max_m": 1000.0,
    "fit_advection": True,
    "max_iter": 60,
    "start_iter": 0,             # iterations per multi-start probe (0 → max(8, max_iter // 4))
    # torch intra-op threads during fit/predict (capped, restored afterwards).  The
    # per-evaluation tensors are small: on a quiet 4-core host 2 threads are only
    # ~15 % faster than 1, but OpenMP threads collapse (3–15× slower) as soon as
    # the host is oversubscribed.  None → leave torch's setting unchanged.
    "num_threads": 1,
    # Penalty weights relative to the normalised data term MSE/var(y).
    "prior_weight": 1e-2,
    "v_penalty": 1e-2,
}

#: Literature priors (mean, sd) for the penalised source coefficients.
SOURCE_PRIORS = {"s": (0.6, 0.1), "a1": (0.3, 0.1), "a1n": (0.5, 0.15)}
KAPPA_BOUNDS = (0.05, 20.0)        # canopy-fraction scale of shade saturation
KAPPA_PRIOR = (0.0, 3.0)           # log κ ~ N(0, 3²): a weak regulariser — a tighter prior pinned κ near 1
                                   # on the synthetic city (planted 0.15) and halved the shade nonlinearity
KAPPA_INIT = 2.0
# Optional S-shaped shade (physics.shade_form: sigmoid): h(c) rises fastest near c0.
C0_BOUNDS = (0.05, 0.95)
CW_BOUNDS = (0.02, 0.5)
_EF0 = 0.35                      # initial evaporative fraction
_EF_INIT = {"e1": 2.0, "e2": 1.0, "e3": 1.0}
_EF_PRIOR_SD = 5.0               # weak: keeps flat directions from drifting
_W_INIT = 50.0                   # W m⁻² water-cell sink
_W_SCALE = 100.0
_DEFAULT_ALBEDO = 0.15
_ROLES = ("albedo", "canopy", "impervious", "ndvi", "elevation", "water_distance")


# ---------------------------------------------------------------------------
# Small numerics helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _torch_threads(n: int | None):
    """Temporarily cap torch's intra-op thread count at ``n``."""
    import torch

    prev = torch.get_num_threads()
    if n is None or int(n) <= 0 or int(n) >= prev:
        yield
        return
    torch.set_num_threads(int(n))
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


class _SpectralSolver:
    """:func:`operators.solve_torch` with the symbols cached for a fixed
    (shape, dx, pad): identical operator, no per-call symbol rebuild."""

    def __init__(self, ny: int, nx: int, dx: float, pad: int):
        import torch

        self.ny, self.nx, self.dx, self.pad = ny, nx, float(dx), int(pad)
        self.py, self.px = ops._padded_shape(ny, nx, self.pad)
        lam, sx, sy = ops.symbols(self.py, self.px, self.dx, self.dx)
        self.lam = torch.as_tensor(lam, dtype=torch.float64)
        self.sx = torch.as_tensor(sx, dtype=torch.float64)
        self.sy = torch.as_tensor(sy, dtype=torch.float64)

    def __call__(self, q, L, vx, vy):
        import torch

        p, ny, nx = self.pad, self.ny, self.nx
        buf = q.new_zeros((self.py, self.px))
        buf[p:p + ny, p:p + nx] = q
        denom = torch.complex(1.0 + (L**2) * self.lam, vx * self.sx + vy * self.sy)
        phi = torch.fft.irfft2(torch.fft.rfft2(buf) / denom, s=(self.py, self.px))
        return phi[p:p + ny, p:p + nx]


def _varpro(D, y, ridge: float = 1e-10):
    """Differentiable linear least squares min‖D·c − y‖² (column-normalised
    normal equations with a tiny ridge).  Returns (coef, fitted)."""
    import torch

    scale = D.detach().pow(2).mean(dim=0).sqrt().clamp_min(1e-300)
    Dn = D / scale
    n = D.shape[0]
    G = Dn.T @ Dn / n + ridge * torch.eye(D.shape[1], dtype=D.dtype)
    c = torch.linalg.solve(G, Dn.T @ y / n)
    return c / scale, Dn @ c


def _lbfgs(params: list, objective: Callable, max_iter: int) -> float:
    """Minimise ``objective()`` over ``params`` with strong-Wolfe L-BFGS."""
    import torch

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=int(max_iter), max_eval=int(1.25 * max_iter) + 1,
                            tolerance_grad=1e-10, tolerance_change=1e-13, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(objective())


def _multistart(starts: list, make_params: Callable, leaves: Callable, objective: Callable, max_iter: int,
                start_iter: int = 0) -> tuple[float, dict, list[float]]:
    """L-BFGS from every start for ``start_iter`` iterations (default
    max(8, max_iter // 4)), then continue the lowest-loss start for
    ``max_iter`` iterations.  With a single start this is one full run."""
    if len(starts) == 1:
        P = make_params(starts[0])
        loss = _lbfgs(leaves(P), lambda: objective(P), max_iter)
        return loss, P, [loss]
    start_iter = start_iter if start_iter > 0 else max(8, max_iter // 4)
    best, losses = None, []
    for v0 in starts:
        P = make_params(v0)
        loss = _lbfgs(leaves(P), lambda P=P: objective(P), start_iter)
        losses.append(loss)
        log.debug("multistart: v0=%s → loss %.6g", v0, loss)
        if best is None or loss < best[0]:
            best = (loss, P)
    P = best[1]
    loss = _lbfgs(leaves(P), lambda: objective(P), max_iter)
    return loss, P, losses


def _v_starts(v_max: float, v_init: tuple[float, float] | None, fit_advection: bool) -> list[tuple[float, float]]:
    if not fit_advection:
        return [(0.0, 0.0)]
    h = 0.5 * v_max
    starts = [(0.0, 0.0), (h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)]
    if v_init is not None and np.hypot(*v_init) > 0:
        starts.insert(0, (float(v_init[0]), float(v_init[1])))
    return starts


def _L_bounds(dx: float, L_max: float, extent: float) -> tuple[float, float]:
    lo = math.log(dx)
    hi = math.log(max(min(L_max, extent / 4.0), 1.5 * dx))
    return lo, hi


def _uL_from_L(L: float, lo: float, hi: float) -> float:
    f = (math.log(max(L, 1e-9)) - lo) / (hi - lo)
    f = min(max(f, 0.02), 0.98)
    return _logit(f)


# ---------------------------------------------------------------------------
# Known-source operator fit (tests / diagnostics)
# ---------------------------------------------------------------------------


def fit_operator(q: np.ndarray, y_raster: np.ndarray, mask: np.ndarray, dx: float, L_init: float, v_max: float,
                 fit_advection: bool = True, L_max: float = 2000.0, max_iter: int = 60,
                 v_penalty: float = 1e-2, num_threads: int | None = 1) -> dict:
    """Fit y ≈ a·solve(q; L, v) + b on ``mask`` cells for a *known* source.

    L and v are found by L-BFGS (multi-start over v, penalty
    ``v_penalty``·(|v|/v_max)²), a and b by variable projection; log L is
    bounded to [log dx, log min(L_max, extent/4)].  Returns L_m, vx_m, vy_m,
    v_norm_m, a, b, rmse, loss, pad_cells.
    """
    with _torch_threads(num_threads):
        return _fit_operator(q, y_raster, mask, dx, L_init, v_max, fit_advection, L_max, max_iter, v_penalty)


def _fit_operator(q, y_raster, mask, dx, L_init, v_max, fit_advection, L_max, max_iter, v_penalty) -> dict:
    import torch

    q = np.asarray(q, dtype=np.float64)
    yr = np.asarray(y_raster, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(yr)
    ny, nx = q.shape
    lo, hi = _L_bounds(dx, L_max, min(ny, nx) * dx)
    pad = ops.pad_cells(math.exp(hi), v_max if fit_advection else 0.0, dx)
    solver = _SpectralSolver(ny, nx, dx, pad)
    qt = torch.as_tensor(np.nan_to_num(q), dtype=torch.float64)
    idx = torch.as_tensor(np.flatnonzero(m.ravel()))
    yt = torch.as_tensor(yr.ravel()[m.ravel()], dtype=torch.float64)
    var_y = float(yt.var())
    ones = torch.ones_like(yt)

    def unpack(P):
        L = torch.exp(lo + (hi - lo) * torch.sigmoid(P["uL"]))
        vx = v_max * P["ux"] if fit_advection else torch.zeros((), dtype=torch.float64)
        vy = v_max * P["uy"] if fit_advection else torch.zeros((), dtype=torch.float64)
        return L, vx, vy

    def evaluate(P):
        L, vx, vy = unpack(P)
        phi = solver(qt, L, vx, vy).reshape(-1)[idx]
        coef, fit = _varpro(torch.stack([phi, ones], dim=1), yt)
        mse = torch.mean((fit - yt) ** 2)
        pen = v_penalty * (P["ux"] ** 2 + P["uy"] ** 2) if fit_advection else 0.0
        return mse / var_y + pen, coef, mse

    def make_params(v0):
        return {
            "uL": torch.tensor(_uL_from_L(L_init, lo, hi), dtype=torch.float64, requires_grad=True),
            "ux": torch.tensor(v0[0] / v_max, dtype=torch.float64, requires_grad=fit_advection),
            "uy": torch.tensor(v0[1] / v_max, dtype=torch.float64, requires_grad=fit_advection),
        }

    def leaves(P):
        return [P["uL"]] + ([P["ux"], P["uy"]] if fit_advection else [])

    loss, P, _ = _multistart(_v_starts(v_max, None, fit_advection), make_params, leaves,
                             lambda P: evaluate(P)[0], max_iter)
    with torch.no_grad():
        _, coef, mse = evaluate(P)
        L, vx, vy = unpack(P)
    return {
        "L_m": float(L), "vx_m": float(vx), "vy_m": float(vy), "v_norm_m": float(math.hypot(float(vx), float(vy))),
        "a": float(coef[0]), "b": float(coef[1]), "rmse": float(math.sqrt(float(mse))), "loss": float(loss),
        "pad_cells": int(pad),
    }


# ---------------------------------------------------------------------------
# Physics base model
# ---------------------------------------------------------------------------


class PhysicsModel:
    """Physics base model ΔT = a·G⊛q(features; θ) + b + γ·elev_c + trend.

    Parameters
    ----------
    grid : Grid
        Raster mapping of the study area; every frame passed to ``fit`` /
        ``predict`` must have one row per grid point, in grid order.
    cfg_physics : dict
        The config ``physics`` section (see :data:`PHYSICS_DEFAULTS`).
    L_init : float, optional
        Initial length scale (m) — the pipeline passes the S1 prior
        ``L_prior_m``; default 3·dx.
    """

    def __init__(self, grid: Grid, cfg_physics: dict | None, L_init: float | None = None, seed: int = 0):
        self.grid = grid
        self.cfg = {**PHYSICS_DEFAULTS, **(cfg_physics or {})}
        self.seed = int(seed)
        self.dx = float(grid.dx)
        self.window = str(self.cfg.get("window") or "day").lower()
        self.shade_form = str(self.cfg.get("shade_form") or "saturating").lower()
        if self.shade_form not in ("saturating", "sigmoid"):
            raise ValueError(f"physics.shade_form must be 'saturating' or 'sigmoid', got {self.shade_form!r}")
        if self.window not in ("day", "night"):
            raise ValueError(f"physics.window must be 'day' or 'night', got {self.window!r}")
        self.roles = {k: v for k, v in (self.cfg.get("roles") or {}).items() if v and k in _ROLES}
        self.v_max = float(self.cfg["v_max_m"])
        fa = self.cfg.get("fit_advection", "auto")
        # "auto": advection is only identifiable with a wind record — without
        # one, a free v drifts to its bound and absorbs regional gradients.
        self.fit_advection = (self.cfg.get("wind") is not None) if fa in ("auto", None) else bool(fa)
        self.L_lo, self.L_hi = _L_bounds(self.dx, float(self.cfg["L_max_m"]), min(grid.nx, grid.ny) * self.dx)
        self.L_init = float(L_init) if L_init else 3.0 * self.dx
        wind = self.cfg.get("wind")
        self.v_init = None
        if wind is not None:
            v = np.asarray(wind, dtype=float)[:2] * float(self.cfg["tau_s"])
            nv = float(np.hypot(*v))
            if nv > self.v_max:
                v = v * self.v_max / nv
            self.v_init = (float(v[0]), float(v[1]))
        self.pad = ops.pad_cells(math.exp(self.L_hi), self.v_max if self.fit_advection else 0.0, self.dx)
        self._solver: _SpectralSolver | None = None
        self._fitted = False
        self.fit_warning: str | None = None
        self.fit_info: dict = {}

    # ------------------------------------------------------------ features
    def _forcing(self) -> tuple[float, float, float]:
        sw = float(self.cfg["sw_down"]) if self.window == "day" else 0.0
        lw = float(self.cfg["lw_net"])
        norm = sw if sw > 0 else max(abs(lw), 1e-9)
        return sw, lw, norm

    def _check_frame(self, frame: pd.DataFrame) -> None:
        if len(frame) != self.grid.n_points:
            raise ValueError(f"frame has {len(frame)} rows but the grid maps {self.grid.n_points} points")

    def _raw_features(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Role columns present in ``frame`` (raw units), NaN-filled with the
        fit-time means."""
        out = {}
        for role, col in self.roles.items():
            if col not in frame.columns:
                continue
            v = frame[col].to_numpy(dtype=np.float64).copy()
            bad = ~np.isfinite(v)
            if bad.any():
                fill = self._fill.get(role) if self._fitted else None
                v[bad] = np.nanmean(v) if fill is None else fill
            if role == "albedo" and self.cfg.get("albedo_map"):
                v = self._map_albedo(v)
            out[role] = v
        return out

    def _map_albedo(self, v: np.ndarray) -> np.ndarray:
        """Linear rescale of a non-broadband albedo layer (``physics.albedo_map``:
        ``{from: [lo, hi] | auto, to: [lo, hi]}``; ``auto`` = the 2nd/98th
        percentiles at fit time).  Scenario doses stay in source units and
        are scaled by the same map."""
        am = self.cfg["albedo_map"]
        if not self._fitted or getattr(self, "_albedo_from", None) is None:
            src = am.get("from", "auto")
            self._albedo_from = (tuple(float(x) for x in np.nanpercentile(v, [2, 98]))
                                 if src in ("auto", None) else (float(src[0]), float(src[1])))
        lo, hi = self._albedo_from
        to_lo, to_hi = (float(x) for x in am.get("to", (0.08, 0.25)))
        if hi <= lo:
            raise ValueError(f"physics.albedo_map: empty source range {self._albedo_from}")
        return to_lo + (v - lo) * (to_hi - to_lo) / (hi - lo)

    def _set_feature_constants(self, feats: dict[str, np.ndarray]) -> None:
        self.roles_used = sorted(feats)
        self._fill = {k: float(np.mean(v)) for k, v in feats.items()}
        self._centre = {
            "ndvi": self._fill.get("ndvi", 0.0),
            "canopy": self._fill.get("canopy", 0.0) / 100.0,
            "impervious": self._fill.get("impervious", 0.0) / 100.0,
        }
        self._elev_mean = self._fill.get("elevation", 0.0)

    def _torch_features(self, feats: dict[str, np.ndarray]) -> dict:
        import torch

        t = {}
        if "albedo" in feats:
            t["albedo"] = torch.as_tensor(feats["albedo"], dtype=torch.float64)
        for role in ("canopy", "impervious"):
            if role in feats:
                t[role] = torch.as_tensor(feats[role] / 100.0, dtype=torch.float64)
        if "ndvi" in feats:
            t["ndvi"] = torch.as_tensor(feats["ndvi"], dtype=torch.float64)
        if "water_distance" in feats:
            t["water"] = torch.as_tensor((feats["water_distance"] <= self.dx / 2.0).astype(np.float64))
        return t

    # -------------------------------------------------------------- params
    def _init_params(self, v0: tuple[float, float]) -> dict:
        import torch

        def p(val, grad=True):
            return torch.tensor(float(val), dtype=torch.float64, requires_grad=grad)

        return {
            "uL": p(_uL_from_L(self.L_init, self.L_lo, self.L_hi)),
            "ux": p(v0[0] / self.v_max, self.fit_advection),
            "uy": p(v0[1] / self.v_max, self.fit_advection),
            "s": p(_logit(SOURCE_PRIORS["s"][0])),
            "ukc": p(_logit((math.log(KAPPA_INIT) - math.log(KAPPA_BOUNDS[0]))
                            / (math.log(KAPPA_BOUNDS[1]) - math.log(KAPPA_BOUNDS[0])))),
            "uc0": p(_logit((0.4 - C0_BOUNDS[0]) / (C0_BOUNDS[1] - C0_BOUNDS[0]))),
            "ucw": p(_logit((math.log(0.1) - math.log(CW_BOUNDS[0])) / (math.log(CW_BOUNDS[1]) - math.log(CW_BOUNDS[0])))),
            "a1": p(_logit(SOURCE_PRIORS["a1"][0])),
            "a1n": p(_logit(SOURCE_PRIORS["a1n"][0])),
            "e0": p(_logit(_EF0)),
            "e1": p(_inv_softplus(_EF_INIT["e1"])),
            "e2": p(_inv_softplus(_EF_INIT["e2"])),
            "e3": p(_inv_softplus(_EF_INIT["e3"])),
            "w": p(_inv_softplus(_W_INIT / _W_SCALE)),
        }

    def _active_leaves(self, P: dict, feats: dict) -> list:
        names = ["uL", "e0"]
        if self.fit_advection:
            names += ["ux", "uy"]
        if self.window == "day" and "canopy" in feats:
            names += ["s"] + (["uc0", "ucw"] if self.shade_form == "sigmoid" else ["ukc"])
        if "impervious" in feats:
            names.append("a1" if self.window == "day" else "a1n")
        names += [e for e, r in (("e1", "ndvi"), ("e2", "canopy"), ("e3", "impervious")) if r in feats]
        if "water" in feats:
            names.append("w")
        return [P[n] for n in names]

    def _transform(self, P: dict) -> dict:
        import torch
        import torch.nn.functional as F

        zero = torch.zeros((), dtype=torch.float64)
        return {
            "L": torch.exp(self.L_lo + (self.L_hi - self.L_lo) * torch.sigmoid(P["uL"])),
            "vx": self.v_max * P["ux"] if self.fit_advection else zero,
            "vy": self.v_max * P["uy"] if self.fit_advection else zero,
            "s": torch.sigmoid(P["s"]),
            "kc": torch.exp(math.log(KAPPA_BOUNDS[0]) + (math.log(KAPPA_BOUNDS[1]) - math.log(KAPPA_BOUNDS[0]))
                            * torch.sigmoid(P["ukc"])),
            "c0": C0_BOUNDS[0] + (C0_BOUNDS[1] - C0_BOUNDS[0]) * torch.sigmoid(P["uc0"]),
            "cw": torch.exp(math.log(CW_BOUNDS[0]) + (math.log(CW_BOUNDS[1]) - math.log(CW_BOUNDS[0]))
                            * torch.sigmoid(P["ucw"])),
            "a1": torch.sigmoid(P["a1"]),
            "a1n": torch.sigmoid(P["a1n"]),
            "e0": P["e0"],
            "e1": F.softplus(P["e1"]),
            "e2": F.softplus(P["e2"]),
            "e3": F.softplus(P["e3"]),
            "w": _W_SCALE * F.softplus(P["w"]),
        }

    def _penalty(self, P: dict, T: dict, feats: dict):
        import torch

        pen = 0.0
        if self.window == "day" and "canopy" in feats:
            pen = pen + ((T["s"] - SOURCE_PRIORS["s"][0]) / SOURCE_PRIORS["s"][1]) ** 2
            if self.shade_form == "sigmoid":
                pen = pen + ((T["c0"] - 0.4) / 0.3) ** 2 + ((torch.log(T["cw"]) - math.log(0.1)) / 1.5) ** 2
            else:
                pen = pen + ((torch.log(T["kc"]) - KAPPA_PRIOR[0]) / KAPPA_PRIOR[1]) ** 2
        if "impervious" in feats:
            k = "a1" if self.window == "day" else "a1n"
            pen = pen + ((T[k] - SOURCE_PRIORS[k][0]) / SOURCE_PRIORS[k][1]) ** 2
        pen = pen + ((T["e0"] - _logit(_EF0)) / _EF_PRIOR_SD) ** 2
        for e, r in (("e1", "ndvi"), ("e2", "canopy"), ("e3", "impervious")):
            if r in feats:
                pen = pen + ((T[e] - _EF_INIT[e]) / _EF_PRIOR_SD) ** 2
        out = float(self.cfg["prior_weight"]) * pen
        if self.fit_advection:
            out = out + float(self.cfg["v_penalty"]) * (P["ux"] ** 2 + P["uy"] ** 2)
        return out

    # -------------------------------------------------------------- source
    def _q_points_torch(self, tf: dict, T: dict):
        """Uncentred normalised source q_raw = Q_H / forcing at every point."""
        import torch

        sw, lw, norm = self._forcing()
        n = next(iter(tf.values())).shape[0] if tf else self.grid.n_points
        c = self._centre
        z = T["e0"] * torch.ones(n, dtype=torch.float64)
        if "ndvi" in tf:
            z = z + T["e1"] * (tf["ndvi"] - c["ndvi"])
        if "canopy" in tf:
            z = z + T["e2"] * (tf["canopy"] - c["canopy"])
        if "impervious" in tf:
            z = z - T["e3"] * (tf["impervious"] - c["impervious"])
        one_minus_ef = 1.0 - torch.sigmoid(z)
        if self.window == "day":
            alb = tf["albedo"] if "albedo" in tf else _DEFAULT_ALBEDO
            if "canopy" in tf and self.shade_form == "sigmoid":
                c0, cw = T["c0"], T["cw"]
                s0 = torch.sigmoid(-c0 / cw)
                h = (torch.sigmoid((tf["canopy"] - c0) / cw) - s0) / (torch.sigmoid((1.0 - c0) / cw) - s0)
                shade = 1.0 - T["s"] * h
            elif "canopy" in tf:
                kc = T["kc"]
                shade = 1.0 - T["s"] * (1.0 - torch.exp(-tf["canopy"] / kc)) / (1.0 - torch.exp(-1.0 / kc))
            else:
                shade = 1.0
            qstar = sw * (1.0 - alb) * shade + lw
            dqs = T["a1"] * tf["impervious"] * qstar if "impervious" in tf else 0.0
        else:
            qstar = lw * torch.ones(n, dtype=torch.float64)
            dqs = -T["a1n"] * tf["impervious"] * abs(lw) if "impervious" in tf else 0.0
        qh = (qstar - dqs) * one_minus_ef
        if "water" in tf:
            qh = qh - T["w"] * tf["water"]
        return qh / norm

    def _cell_mean(self, values):
        """Differentiable per-cell mean of a point tensor → flat (ny·nx,), 0 off-mask."""
        flat = values.new_zeros(self.grid.ny * self.grid.nx)
        flat = flat.index_add(0, self._cell_idx_t, values)
        return flat / self._counts_t

    def _q_raster_torch(self, tf: dict, T: dict, q_mean=None):
        """Centred source raster (ny, nx); returns (raster, mean used)."""
        q_cells = self._cell_mean(self._q_points_torch(tf, T))
        if q_mean is None:
            q_mean = q_cells[self._valid_flat_t].mean()
        q = (q_cells - q_mean) * self._valid_flat_f
        return q.view(self.grid.ny, self.grid.nx), q_mean

    # ----------------------------------------------------------------- fit
    def _prepare_grid_tensors(self) -> None:
        import torch

        g = self.grid
        self._cell_idx_t = torch.as_tensor(g.cell_index)
        counts = g.counts.ravel().astype(np.float64)
        self._valid_flat = counts > 0
        self._counts_t = torch.as_tensor(np.maximum(counts, 1.0))
        self._valid_flat_t = torch.as_tensor(self._valid_flat)
        self._valid_flat_f = torch.as_tensor(self._valid_flat.astype(np.float64))
        if self._solver is None:
            self._solver = _SpectralSolver(g.ny, g.nx, self.dx, self.pad)

    def _linear_columns(self, feats: dict[str, np.ndarray], coords: np.ndarray) -> np.ndarray:
        cols = []
        if "elevation" in feats:
            cols.append(feats["elevation"] - self._elev_mean)
        cols.append(coords[:, 0] - self._xy_mean[0])
        cols.append(coords[:, 1] - self._xy_mean[1])
        return np.column_stack(cols)

    def fit(self, frame: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, coords: np.ndarray) -> "PhysicsModel":
        """Fit on the labels at ``train_idx`` (int indices or boolean mask);
        the source uses every point's features."""
        with _torch_threads(self.cfg.get("num_threads")):
            return self._fit(frame, y, train_idx, coords)

    def _fit(self, frame: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, coords: np.ndarray) -> "PhysicsModel":
        import torch

        t0 = time.perf_counter()
        self._check_frame(frame)
        self._fitted = False
        coords = np.asarray(coords, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        tr = np.asarray(train_idx)
        tr = np.flatnonzero(tr) if tr.dtype == bool else tr.astype(np.int64)
        tr = tr[np.isfinite(y[tr])]
        if tr.size < 10:
            raise ValueError(f"PhysicsModel.fit: only {tr.size} finite training labels")

        feats = self._raw_features(frame)
        missing = sorted(set(self.roles) - set(feats))
        if missing:
            log.warning("physics: role columns missing from frame, terms dropped: %s", missing)
        self._set_feature_constants(feats)
        self._xy_mean = coords.mean(axis=0)
        self._prepare_grid_tensors()
        tf = self._torch_features(feats)

        lin = self._linear_columns(feats, coords)[tr]
        lin_t = torch.as_tensor(lin, dtype=torch.float64)
        ones = torch.ones(tr.size, dtype=torch.float64)
        yt = torch.as_tensor(y[tr], dtype=torch.float64)
        var_y = float(yt.var()) if tr.size > 1 else 1.0
        var_y = var_y if var_y > 0 else 1.0
        tr_cells = torch.as_tensor(self.grid.cell_index[tr])
        solver = self._solver

        def evaluate(P):
            T = self._transform(P)
            q, q_mean = self._q_raster_torch(tf, T)
            phi = solver(q, T["L"], T["vx"], T["vy"]).reshape(-1)[tr_cells]
            D = torch.cat([phi[:, None], ones[:, None], lin_t], dim=1)
            coef, fit = _varpro(D, yt)
            mse = torch.mean((fit - yt) ** 2)
            # Physical sign: more absorbed/sensible heat must warm the air
            # (a ≥ 0).  Push the search away from anti-physical optima.
            neg = torch.clamp(-coef[0] * phi.detach().std(), min=0.0) ** 2 / var_y
            return mse / var_y + self._penalty(P, T, tf) + 10.0 * neg, coef, mse, T, q_mean

        max_iter = int(self.cfg["max_iter"])
        starts = _v_starts(self.v_max, self.v_init, self.fit_advection)
        loss, P, losses = _multistart(
            starts, self._init_params, lambda P: self._active_leaves(P, tf), lambda P: evaluate(P)[0], max_iter,
            int(self.cfg.get("start_iter") or 0))
        with torch.no_grad():
            _, coef, mse, T, q_mean = evaluate(P)
        self._P = {k: v.detach().clone() for k, v in P.items()}
        self._T = {k: float(v.detach()) for k, v in T.items()}
        self._q_mean = float(q_mean)
        c = coef.numpy()
        self.a, self.b = float(c[0]), float(c[1])
        k = 2
        self.gamma = 0.0
        if "elevation" in feats:
            self.gamma = float(c[k])
            k += 1
        self.beta_x, self.beta_y = float(c[k]), float(c[k + 1])
        self.train_rmse = float(math.sqrt(float(mse)))
        self.train_r2 = float(1.0 - float(mse) / var_y)
        self.fit_warning = None
        if self.a <= 0:
            # Anti-physical optimum even under the sign penalty: switch the
            # non-local term off (a = 0) and refit the linear part only.
            self.fit_warning = (f"fitted source amplitude a = {self.a:.4g} ≤ 0 (source anti-correlated with ΔT); "
                                "physics term disabled (a = 0)")
            log.warning("physics: %s", self.fit_warning)
            D0 = torch.cat([ones[:, None], lin_t], dim=1)
            c0, fit0 = _varpro(D0, yt)
            c0 = c0.detach().numpy()
            self.a, self.b = 0.0, float(c0[0])
            k = 1
            if "elevation" in feats:
                self.gamma = float(c0[k])
                k += 1
            self.beta_x, self.beta_y = float(c0[k]), float(c0[k + 1])
            mse0 = float(torch.mean((fit0 - yt) ** 2))
            self.train_rmse = float(math.sqrt(mse0))
            self.train_r2 = float(1.0 - mse0 / var_y)
        self._fitted = True
        self._store_operator_rasters(frame)
        self.fit_info = {"loss": float(loss), "start_losses": losses, "starts": starts,
                         "n_train": int(tr.size), "seconds": time.perf_counter() - t0, "pad_cells": int(self.pad)}
        log.info("physics: L=%.0f m v=(%.0f, %.0f) m a=%.3g train RMSE %.3f (%.1f s)", self._T["L"],
                 self._T["vx"], self._T["vy"], self.a, self.train_rmse, self.fit_info["seconds"])
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("PhysicsModel is not fitted")

    def _store_operator_rasters(self, frame: pd.DataFrame) -> None:
        """Rasters needed by :meth:`operator_residual_torch` (fit frame)."""
        g = self.grid
        self._q_fit_raster = self.source_raster(frame)
        feats = self._raw_features(frame)
        if "elevation" in feats:
            e = g.rasterize(feats["elevation"]) - self._elev_mean
            self._elev_c_raster = np.nan_to_num(e)
        else:
            self._elev_c_raster = np.zeros(g.shape)
        X, Y = g.cell_centres()
        self._xc_raster = X - self._xy_mean[0]
        self._yc_raster = Y - self._xy_mean[1]

    # ------------------------------------------------------------ predict
    def _fitted_torch(self, frame: pd.DataFrame):
        import torch

        self._require_fit()
        self._check_frame(frame)
        feats = self._raw_features(frame)
        tf = self._torch_features(feats)
        T = {k: torch.tensor(v, dtype=torch.float64) for k, v in self._T.items()}
        return feats, tf, T

    def source_points(self, frame: pd.DataFrame) -> np.ndarray:
        """Uncentred normalised source q_raw = Q_H/forcing at every point of
        ``frame`` — a pointwise function of each row's own features."""
        import torch

        _, tf, T = self._fitted_torch(frame)
        with torch.no_grad():
            return self._q_points_torch(tf, T).numpy().copy()

    def source_raster(self, frame: pd.DataFrame) -> np.ndarray:
        """Centred source q (ny, nx): cell mean of :meth:`source_points` minus
        the fit-frame valid-cell mean; 0 outside the mask."""
        import torch

        _, tf, T = self._fitted_torch(frame)
        with torch.no_grad():
            q, _ = self._q_raster_torch(tf, T, q_mean=torch.tensor(self._q_mean, dtype=torch.float64))
        return q.numpy().copy()

    def phi_raster(self, frame: pd.DataFrame) -> np.ndarray:
        """φ = G⊛q on the raster (same padding as the fit)."""
        import torch

        q = torch.as_tensor(self.source_raster(frame))
        with torch.no_grad(), _torch_threads(self.cfg.get("num_threads")):
            phi = self._solver(q, torch.tensor(self._T["L"]), torch.tensor(self._T["vx"]), torch.tensor(self._T["vy"]))
        return phi.numpy().copy()

    def predict(self, frame: pd.DataFrame, coords: np.ndarray) -> np.ndarray:
        """ΔT_phys at every point of ``frame`` (same grid as the fit)."""
        self._require_fit()
        coords = np.asarray(coords, dtype=np.float64)
        phi = self.grid.sample(self.phi_raster(frame))
        feats = self._raw_features(frame)
        lin = self._linear_columns(feats, coords)
        coefs = ([self.gamma] if "elevation" in feats else []) + [self.beta_x, self.beta_y]
        return self.a * phi + self.b + lin @ np.asarray(coefs)

    # --------------------------------------------------------- PDE residual
    def operator_residual_torch(self, dT_raster, valid_mask, q_raster=None):
        """Residual of the fitted operator on a ΔT raster, in source units.

        R = [L_op(ΔT − b − γ·elev_c − βx·x_c − βy·y_c) − a·q] / a on the cells
        where :func:`operators.stencil_valid` (valid_mask) holds, as a 1-D
        tensor (differentiable in ``dT_raster``).  ``q_raster`` defaults to
        the fit frame's source; pass :meth:`source_raster` of an edited
        frame for scenarios.
        """
        import torch

        self._require_fit()
        dt = dT_raster.dtype
        dev = dT_raster.device
        vm = valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
        sv = ops.stencil_valid(vm.astype(bool))[1:-1, 1:-1]
        q = self._q_fit_raster if q_raster is None else np.asarray(q_raster, dtype=np.float64)
        base = (self.b + self.gamma * self._elev_c_raster + self.beta_x * self._xc_raster
                + self.beta_y * self._yc_raster)
        field = dT_raster - torch.as_tensor(base, dtype=dt, device=dev)
        Lop = ops.apply_operator_torch(field, self._T["L"], self._T["vx"], self._T["vy"], self.dx)
        R = Lop - self.a * torch.as_tensor(q[1:-1, 1:-1], dtype=dt, device=dev)
        a = self.a if abs(self.a) > 1e-12 else math.copysign(1e-12, self.a or 1.0)
        return R[torch.as_tensor(sv, device=dev)] / a

    # --------------------------------------------------------------- params
    @property
    def params(self) -> dict:
        self._require_fit()
        T = self._T
        return {
            "L_m": T["L"], "vx_m": T["vx"], "vy_m": T["vy"], "v_norm_m": float(math.hypot(T["vx"], T["vy"])),
            "a": self.a, "b": self.b, "gamma": self.gamma, "beta_x": self.beta_x, "beta_y": self.beta_y,
            "s": T["s"], "shade_form": self.shade_form, "kappa_canopy": T["kc"],
            "shade_c0": T["c0"], "shade_width": T["cw"], "a1": T["a1"], "a1n": T["a1n"], "e0": T["e0"], "e1": T["e1"], "e2": T["e2"],
            "e3": T["e3"], "w": T["w"], "q_mean": self._q_mean,
            "train_rmse": self.train_rmse, "train_r2": self.train_r2, "fit_warning": self.fit_warning,
            "window": self.window, "roles_used": list(self.roles_used), "pad_cells": int(self.pad),
            "albedo_map": ({"from": list(self._albedo_from), "to": list(self.cfg["albedo_map"].get("to", (0.08, 0.25)))}
                           if self.cfg.get("albedo_map") and getattr(self, "_albedo_from", None) else None),
            "L_bounds_m": [math.exp(self.L_lo), math.exp(self.L_hi)],
        }
