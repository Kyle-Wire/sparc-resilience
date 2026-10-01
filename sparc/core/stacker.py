"""S3 — physics-informed neural stacker.

The final prediction is a base prediction plus a learned residual:

    ΔT̂ = Ẑ·w + r_θ(z),         z = [Ẑ_other, Ẑ_phys, X, focal]      ("feature" mode, default)
    ΔT̂ = ΔT_phys + r_θ(z),     z = [Ẑ_other, X, focal]              ("backbone" mode)

where Ẑ are *out-of-fold* base-model predictions and w ≥ 0, Σw = 1 are
non-negative least-squares weights (a convex super-learner base) or, as a
candidate, equal weights — fitted weights can transfer poorly across a few
large held-out blocks (the forecast-combination puzzle).  The MLP
residual r_θ is early-stopped on an inner spatial-block split of the training
rows and **gated**: if it does not beat the convex base on those held-out
blocks it is switched off and the stacker is Ẑ·w.  In feature mode the
physics model is one of the base models the network digests — its weight in
the final prediction (and therefore in scenario magnitudes) is learned from
data, which matters on single-snapshot data where the physics gain is weakly
identified.  Backbone mode forces physics-driven scenario magnitudes.

The loss is

    Σ (y − ΔT̂)² / (n σ_y²)  +  λ · mean((𝓛 r)²) / mean((𝓛 e)²),   r = ΔT̂ − ΔT_phys

𝓛 = 1 − L²∇² + v·∇ is the fitted physics operator (same discrete stencils as
the physics solve), so 𝓛r/a is the extra heat source the learned residual
would imply.  It is normalised by the same quantity for the data's own misfit
e = y − ΔT_phys on training cells, so λ = 1 penalises a residual that is as
"unphysical" as the raw misfit, and λ = 0 recovers a plain residual MLP.  The
penalty is evaluated on the full raster (rows without labels are valid
collocation points — only features are used there).  Because 𝓛 contains the
identity it doubles as a Sobolev smoother on r.  Without a physics model the
stacker falls back to ΔT̂ = Ẑ·w + r_θ(z) with no PDE term.

Uncertainty: cross-conformal intervals — for fold k the interval half-width
is the (1−α) quantile of |OOF residuals| from the *other* folds, so reported
coverage is honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class StackerInputs:
    Z: np.ndarray             # (n, m) OOF (or scenario) base predictions, physics excluded
    phys: np.ndarray | None   # (n,) physics prediction or None
    feats: np.ndarray         # (n, f) raw + focal features


def _mlp(n_in: int, hidden: int, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.05),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, 1),
    )
    with torch.no_grad():  # start at "residual ≈ 0"
        net[-1].weight.mul_(0.01)
        net[-1].bias.zero_()
    return net.double()


class PhysicsInformedStacker:
    """One stacker (one fold).  ``physics`` is the fold's fitted
    :class:`sparc.core.physics.PhysicsModel` (or None)."""

    def __init__(self, cfg: dict, grid, physics=None, sigma_q: float | None = None, lambda_pde: float | None = None,
                 seed: int = 0, base_mode: str = "nnls"):
        self.cfg = cfg
        self.grid = grid
        self.physics = physics
        self.sigma_q = sigma_q
        self.lambda_pde = float(cfg.get("lambda_pde", 1.0) if lambda_pde is None else lambda_pde)
        self.seed = seed
        # "feature" (default): the physics prediction is one of the base-model
        # inputs the network digests, so the data decide how much of it to
        # use.  "backbone": ΔT̂ = ΔT_phys + r (physics passes through with
        # coefficient 1 — forces physics-driven scenario magnitudes).
        self.mode = str(cfg.get("physics_mode", "feature")).lower()
        self.use_features = bool(cfg.get("use_features", True))
        self.base_mode = base_mode               # "nnls" (convex blend) | "mean" (equal weights)
        self.weights: np.ndarray | None = None
        self.gated = False
        if self.mode not in ("feature", "backbone"):
            raise ValueError(f"stacker.physics_mode must be 'feature' or 'backbone', got {self.mode!r}")

    # ---------------------------------------------------------------- utils
    def _base_matrix(self, inp: StackerInputs) -> np.ndarray:
        if inp.phys is None or self.mode == "backbone":
            return inp.Z
        return np.column_stack([inp.Z, inp.phys])

    def _design(self, inp: StackerInputs) -> tuple[np.ndarray, np.ndarray]:
        B = self._base_matrix(inp)
        D = np.hstack([B, inp.feats]) if self.use_features else B
        if inp.phys is not None and self.mode == "backbone":
            return D, np.asarray(inp.phys, float)
        return D, B @ self.weights

    def _inner_split(self, train_idx: np.ndarray, groups: np.ndarray | None,
                     buffer_m: float = 0.0) -> tuple[np.ndarray, np.ndarray | None]:
        """Hold out ~25 % of the training *blocks* for early stopping and the
        residual gate, dropping inner-training rows within ``buffer_m`` of the
        held-out rows (random rows, or unbuffered blocks, leak through spatial
        autocorrelation and make the residual look better than it is)."""
        frac = float(self.cfg.get("val_fraction", 0.25))
        rng = np.random.default_rng(self.seed)
        if groups is not None:
            g = np.asarray(groups)[train_idx]
            ug = np.unique(g)
            if ug.size >= 4:
                vg = rng.choice(ug, max(1, int(round(frac * ug.size))), replace=False)
                is_val = np.isin(g, vg)
            else:
                is_val = rng.random(train_idx.size) < frac
        else:
            is_val = rng.random(train_idx.size) < frac
        fit_idx, val_idx = train_idx[~is_val], train_idx[is_val]
        if buffer_m > 0 and val_idx.size:
            from scipy.ndimage import distance_transform_edt

            g = self.grid
            vm = np.zeros(g.shape, dtype=bool)
            vm[g.iy[val_idx], g.ix[val_idx]] = True
            dist = distance_transform_edt(~vm, sampling=(g.dy, g.dx))
            fit_idx = fit_idx[dist[g.iy[fit_idx], g.ix[fit_idx]] > buffer_m]
        if val_idx.size < 20 or fit_idx.size < 20:
            return train_idx, None
        return fit_idx, val_idx

    def fit(self, inp: StackerInputs, y: np.ndarray, train_idx: np.ndarray,
            groups: np.ndarray | None = None, buffer_m: float = 0.0,
            residual: bool = True) -> "PhysicsInformedStacker":
        """``groups`` (spatial block ids for every row) and ``buffer_m`` set
        up the inner block split used for early stopping and the residual
        gate.  ``residual=False`` fits the convex base only."""
        import copy

        import torch
        from scipy.optimize import nnls

        fit_idx, val_idx = self._inner_split(np.asarray(train_idx), groups, buffer_m)
        if not residual:
            fit_idx, val_idx = np.asarray(train_idx), None
        # Convex base: non-negative least-squares weights of the OOF base
        # predictions, normalised to sum to one (super-learner style).
        B = self._base_matrix(inp)
        if inp.phys is not None and self.mode == "backbone":
            self.weights = None
        elif self.base_mode == "mean":
            self.weights = np.full(B.shape[1], 1.0 / B.shape[1])
        else:
            w, _ = nnls(B[fit_idx], np.asarray(y, float)[fit_idx])
            self.weights = w / w.sum() if w.sum() > 1e-12 else np.full(B.shape[1], 1.0 / B.shape[1])

        D, base = self._design(inp)
        if not residual:
            self.gated, self.val_mse_base, self.val_mse_best, self.best_epoch = True, None, None, 0
            self.history = []
            return self
        self.mu = D[fit_idx].mean(axis=0)
        sd = D[fit_idx].std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        Dt = torch.as_tensor((D - self.mu) / self.sd, dtype=torch.float64)
        base_t = torch.as_tensor(base, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        tr = torch.as_tensor(fit_idx, dtype=torch.long)
        va = torch.as_tensor(val_idx, dtype=torch.long) if val_idx is not None else None
        sigma_y2 = float(np.var(y[fit_idx])) + 1e-12

        use_pde = self.physics is not None and self.lambda_pde > 0
        if use_pde:
            from sparc.core.operators import apply_operator_torch, stencil_valid

            pp = self.physics.params
            L, vx, vy = float(pp["L_m"]), float(pp.get("vx_m", 0.0)), float(pp.get("vy_m", 0.0))
            interior = torch.as_tensor(stencil_valid(self.grid.mask)[1:-1, 1:-1])
            # Normalise by the operator roughness of the data's own misfit
            # (y − ΔT_phys) on training cells: λ = 1 then penalises a learned
            # residual that is as "unphysical" as the raw misfit itself.
            phys_np = np.array(inp.phys, dtype=float)
            phys_t = torch.as_tensor(phys_np, dtype=torch.float64)
            with torch.no_grad():
                resid_rast = self.grid.rasterize(np.asarray(y, float) - phys_np, subset=fit_idx)
                ok = np.isfinite(resid_rast)
                sv = stencil_valid(ok)[1:-1, 1:-1]
                if sv.sum() >= 50:
                    Rr = apply_operator_torch(torch.as_tensor(np.nan_to_num(resid_rast)), L, vx, vy,
                                              self.grid.dx, self.grid.dy)[torch.as_tensor(sv)]
                    self.pde_scale = float((Rr**2).mean()) or 1.0
                else:
                    self.pde_scale = float(np.var(np.asarray(y, float)[fit_idx])) or 1.0
        self.net = _mlp(D.shape[1], int(self.cfg.get("hidden", 64)), self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg.get("lr", 3e-3)),
                               weight_decay=float(self.cfg.get("weight_decay", 1e-4)))
        epochs = int(self.cfg.get("epochs", 400))
        every = int(self.cfg.get("eval_every", 10))
        patience = int(self.cfg.get("patience", 10))
        self.history = []
        best = (np.inf, None, 0)
        self.val_mse_base = float(((base_t[va] - y_t[va]) ** 2).mean()) if va is not None else None
        bad = 0
        for ep in range(epochs):
            self.net.train()
            opt.zero_grad()
            r = self.net(Dt).squeeze(-1)
            pred = base_t + r
            loss = ((pred[tr] - y_t[tr]) ** 2).mean() / sigma_y2
            if use_pde:
                # The PDE acts on the part of the prediction the physics does
                # not explain (ΔT̂ − ΔT_phys; = r in backbone mode).
                r_rast, _ = self.grid.rasterize_torch(pred - phys_t)
                R = apply_operator_torch(r_rast, L, vx, vy, self.grid.dx, self.grid.dy)[interior]
                pde = (R**2).mean() / self.pde_scale if R.numel() else r.new_zeros(())
                loss = loss + self.lambda_pde * pde
            loss.backward()
            opt.step()
            if ep % 100 == 0 or ep == epochs - 1:
                self.history.append(float(loss.detach()))
            if va is not None and (ep % every == every - 1 or ep == epochs - 1):
                self.net.eval()
                with torch.no_grad():
                    v = float(((base_t[va] + self.net(Dt[va]).squeeze(-1) - y_t[va]) ** 2).mean())
                if v < best[0]:
                    best, bad = (v, copy.deepcopy(self.net.state_dict()), ep + 1), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        self.best_epoch = best[2]
        self.val_mse_best = best[0] if np.isfinite(best[0]) else None
        # Gate: the learned residual must beat the convex base on held-out
        # blocks by ≥ min_gain (relative), otherwise the stacker is the base.
        min_gain = float(self.cfg.get("min_gain", 0.01))
        self.gated = bool(self.val_mse_base is not None and self.val_mse_best is not None
                          and self.val_mse_best > (1.0 - min_gain) * self.val_mse_base)
        self.net.eval()
        return self

    def predict(self, inp: StackerInputs) -> np.ndarray:
        import torch

        D, base = self._design(inp)
        if self.gated:
            return base
        with torch.no_grad():
            r = self.net(torch.as_tensor((D - self.mu) / self.sd, dtype=torch.float64)).squeeze(-1).numpy()
        return base + r

    def summary(self, names: list[str]) -> dict:
        return {"weights": (None if self.weights is None else {n: float(w) for n, w in zip(names, self.weights)}),
                "residual_gated_off": self.gated, "val_mse_base": self.val_mse_base,
                "val_mse_with_residual": self.val_mse_best, "best_epoch": self.best_epoch}


def cross_conformal_halfwidth(y: np.ndarray, oof: np.ndarray, fold_id: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    """Per-point interval half-width using residual scores from the other folds."""
    res = np.abs(y - oof)
    hw = np.empty_like(res)
    for k in np.unique(fold_id):
        other = res[fold_id != k]
        n = other.size
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        hw[fold_id == k] = np.quantile(other, level)
    return hw


def cross_conformal_adaptive(y: np.ndarray, oof: np.ndarray, fold_id: np.ndarray, difficulty: np.ndarray,
                             coverage: float = 0.9) -> np.ndarray:
    """Normalised (locally adaptive) cross-conformal half-widths.

    A scale σ̂(u) = max(a + b·u, floor) is fitted to |residual| against a
    difficulty score u (here log(1 + distance to training / cell)) on the
    *other* folds; the conformal quantile of |r|/σ̂ from those folds then
    scales σ̂ at each point of fold k.  Validity is the same as the global
    version; widths follow how far a point is from training data."""
    res = np.abs(np.asarray(y, float) - np.asarray(oof, float))
    u = np.asarray(difficulty, float)
    hw = np.empty_like(res)
    for k in np.unique(fold_id):
        o = fold_id != k
        A = np.column_stack([np.ones(o.sum()), u[o]])
        coef = np.linalg.lstsq(A, res[o], rcond=None)[0]
        floor = max(1e-6, 0.25 * float(np.median(res[o])))
        sig_o = np.maximum(A @ coef, floor)
        n = int(o.sum())
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        q = np.quantile(res[o] / sig_o, level)
        sig_k = np.maximum(coef[0] + coef[1] * u[fold_id == k], floor)
        hw[fold_id == k] = q * sig_k
    return hw


def global_conformal_halfwidth(y: np.ndarray, oof: np.ndarray, coverage: float = 0.9) -> float:
    res = np.abs(y - oof)
    n = res.size
    return float(np.quantile(res, min(1.0, np.ceil((n + 1) * coverage) / n)))
