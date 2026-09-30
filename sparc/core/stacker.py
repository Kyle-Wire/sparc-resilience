"""S3 — physics-informed neural stacker.

The final prediction is a base prediction plus a learned residual:

    ΔT̂ = mean(Ẑ) + r_θ(z),     z = [Ẑ_other, Ẑ_phys, X, focal]      ("feature" mode, default)
    ΔT̂ = ΔT_phys + r_θ(z),     z = [Ẑ_other, X, focal]              ("backbone" mode)

where Ẑ are *out-of-fold* base-model predictions.  In feature mode the
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
stacker falls back to ΔT̂ = mean(Ẑ) + r_θ(z) with no PDE term.

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
                 seed: int = 0):
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
        if self.mode not in ("feature", "backbone"):
            raise ValueError(f"stacker.physics_mode must be 'feature' or 'backbone', got {self.mode!r}")

    # ---------------------------------------------------------------- utils
    def _design(self, inp: StackerInputs) -> tuple[np.ndarray, np.ndarray]:
        if inp.phys is None:
            return np.hstack([inp.Z, inp.feats]), inp.Z.mean(axis=1)
        if self.mode == "backbone":
            return np.hstack([inp.Z, inp.feats]), inp.phys
        allb = np.column_stack([inp.Z, inp.phys])
        return np.hstack([allb, inp.feats]), allb.mean(axis=1)

    def fit(self, inp: StackerInputs, y: np.ndarray, train_idx: np.ndarray) -> "PhysicsInformedStacker":
        import torch

        D, base = self._design(inp)
        self.mu = D[train_idx].mean(axis=0)
        sd = D[train_idx].std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        Dt = torch.as_tensor((D - self.mu) / self.sd, dtype=torch.float64)
        base_t = torch.as_tensor(base, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        tr = torch.as_tensor(train_idx, dtype=torch.long)
        sigma_y2 = float(np.var(y[train_idx])) + 1e-12

        use_pde = self.physics is not None and self.lambda_pde > 0
        if use_pde:
            from sparc.core.operators import apply_operator_torch, stencil_valid

            pp = self.physics.params
            L, vx, vy = float(pp["L_m"]), float(pp.get("vx_m", 0.0)), float(pp.get("vy_m", 0.0))
            interior = torch.as_tensor(stencil_valid(self.grid.mask)[1:-1, 1:-1])
            # Normalise by the operator roughness of the data's own residual
            # (y − base) on training cells: λ = 1 then penalises a learned
            # residual that is as "unphysical" as the raw misfit itself.
            phys_np = np.array(inp.phys, dtype=float)
            phys_t = torch.as_tensor(phys_np, dtype=torch.float64)
            with torch.no_grad():
                resid_rast = self.grid.rasterize(np.asarray(y, float) - phys_np, subset=train_idx)
                ok = np.isfinite(resid_rast)
                sv = stencil_valid(ok)[1:-1, 1:-1]
                if sv.sum() >= 50:
                    Rr = apply_operator_torch(torch.as_tensor(np.nan_to_num(resid_rast)), L, vx, vy,
                                              self.grid.dx, self.grid.dy)[torch.as_tensor(sv)]
                    self.pde_scale = float((Rr**2).mean()) or 1.0
                else:
                    self.pde_scale = float(np.var(np.asarray(y, float)[train_idx])) or 1.0
        self.net = _mlp(D.shape[1], int(self.cfg.get("hidden", 64)), self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg.get("lr", 3e-3)),
                               weight_decay=float(self.cfg.get("weight_decay", 1e-4)))
        epochs = int(self.cfg.get("epochs", 400))
        self.history = []
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
        self.net.eval()
        return self

    def predict(self, inp: StackerInputs) -> np.ndarray:
        import torch

        D, base = self._design(inp)
        with torch.no_grad():
            r = self.net(torch.as_tensor((D - self.mu) / self.sd, dtype=torch.float64)).squeeze(-1).numpy()
        return base + r


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


def global_conformal_halfwidth(y: np.ndarray, oof: np.ndarray, coverage: float = 0.9) -> float:
    res = np.abs(y - oof)
    n = res.size
    return float(np.quantile(res, min(1.0, np.ceil((n + 1) * coverage) / n)))
