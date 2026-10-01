"""A fast linear emulator of the fitted model for interactive scenario design.

The exact scenario engine re-predicts every base model for an edit (seconds
to minutes).  For the results page's design tool, the fold-averaged model is
linearised around today's city, per lever ``x``:

    ΔT_j = own_j·Δx_j
           + Σ_channels coef_c,j · [K_σc ∗ (m·w_c·Δx)]_j          (neighbourhood features)
           + [P ∗ (dq·Δx)]_j                                     (physics source → temperature)

* ``own`` — sensitivity to the cell's own value (incl. the mediator, e.g.
  canopy → NDVI);
* one *channel* per neighbourhood feature that the edit moves (the lever and
  its mediators, at each focal scale): a Gaussian kernel K_σ over the mask
  ``m``, source weight ``w`` (∂column/∂lever, 1 for the lever itself) and
  receiving coefficient ``coef`` = (∂ŷ/∂F)/(K∗m) — exactly the terms of
  :meth:`ResponseEngine.marginals`;
* the physics term: per-cell source change per unit ``dq`` convolved with a
  fold-averaged kernel P = mean_k w_k·a_k·G_k (Green's function of each
  fold's fitted operator, truncated to ``kernel_cells``).

Everything is fold-averaged, matching the decision maps.  The emulator is
exact for small edits up to the model's curvature; :func:`validate`
compares it with the exact engine on random patch edits and a city-wide
package so the page can show its error.  Saturation (large doses on cells
near their limit) and clipping to the observed support are not emulated.
"""

from __future__ import annotations

import logging

import numpy as np

from sparc.core import operators as ops

log = logging.getLogger(__name__)


def build_emulator(resp, var: str, kernel_cells: int | None = None, max_kernel_cells: int = 120,
                   step: float | None = None) -> dict:
    """Emulator arrays for one lever from a :class:`ResponseEngine`.

    Sensitivities are secants over ``step`` (default: the median configured
    dose of the lever), so typical design edits — not infinitesimal ones —
    are reproduced, including the curvature the model has over that range."""
    from sparc.core.features import build_context

    data, g, ens = resp.data, resp.data.grid, resp.ens
    x = data.frame[var].to_numpy(float)
    h = float(step) if step else design_dose(resp.cfg, var, x)
    delta = resp._step(var, h)
    base_fp = ens.fold_predictions(resp.base_ctx, phys_override=resp._phys_base)
    fr = resp._edited(var, x + delta)
    moved = resp._moved_columns(var, fr)
    ctx_own = build_context(fr, data, resp.eng.ranges_m, resp.cfg, F_override=resp.base_ctx.F)
    own = ens.decision((ens.fold_predictions(ctx_own, phys_override=resp._phys_base) - base_fp) / delta)
    channels = []
    for colname, dcol in moved.items():
        ratio = dcol / delta
        if not np.any(ratio):
            continue
        for col, s in resp._focal_cols(colname):
            sig = resp._sigma_cells(colname, s)
            Fv = resp.base_ctx.F[col]
            hF = 0.05 * (np.nanpercentile(Fv, 95) - np.nanpercentile(Fv, 5)) or 1e-3
            F2 = resp.base_ctx.F.copy()
            F2[col] = F2[col] + hF
            ctx_f = build_context(data.frame, data, resp.eng.ranges_m, resp.cfg, F_override=F2)
            gs = ens.decision((ens.fold_predictions(ctx_f, phys_override=resp._phys_base) - base_fp) / hF)
            Dn = g.sample(ops.gaussian_conv(g.mask.astype(float), sig))
            channels.append({"column": colname, "feature": col, "sigma_cells": float(sig),
                             "weight": ratio.astype(float), "coef": gs / np.maximum(Dn, 1e-9)})
    phys = None
    terms = resp._physics_terms(fr, delta)
    if terms is not None:
        hp = 0.05
        g_phys = (ens.fold_predictions(resp.base_ctx, phys_override=resp._phys_base + hp) - base_fp) / hp
        if kernel_cells is None:                             # ≈ 4 L (+ drift) of the longest fold operator
            reach = max(4.0 * float(st.physics.model.params["L_m"])
                        + float(np.hypot(st.physics.model.params.get("vx_m", 0.0),
                                         st.physics.model.params.get("vy_m", 0.0))) for st in ens.stacks)
            kernel_cells = int(np.clip(np.ceil(reach / g.dx), 8, max_kernel_cells))
        k = int(kernel_cells)
        imp = np.zeros((2 * k + 1, 2 * k + 1))
        imp[k, k] = 1.0
        kern = np.zeros_like(imp)
        dq = np.zeros(data.n)
        for f, pt in enumerate(terms):
            p = ens.stacks[f].physics.model.params
            L, v = float(p["L_m"]), (float(p.get("vx_m", 0.0)), float(p.get("vy_m", 0.0)))
            G = ops.solve(imp, L, v, dx=g.dx, dy=g.dy, pad=k)
            w = float(np.median(g_phys[f]))                  # the stack's weight on its physics input
            kern += w * pt["a"] * G
            dq += pt["dq"]
        phys = {"kernel": kern / len(terms), "dq": dq / len(terms), "kernel_cells": k}
    return {"variable": var, "own": own, "channels": channels, "physics": phys,
            "n": int(data.n), "grid": {"nx": g.nx, "ny": g.ny}, "step": float(h)}


def design_dose(cfg, var: str, x: np.ndarray) -> float:
    """A typical design change of a lever: the lower-quartile configured dose,
    capped at one standard deviation of the layer (so it stays inside the
    observed range for most cells), to two significant figures."""
    doses = sorted(abs(float(d)) for d in (cfg.actionable.get(var) or {}).get("doses", []) if float(d) != 0)
    sd = float(np.nanstd(x))
    d = doses[len(doses) // 4] if doses else sd
    d = min(d, sd) if sd > 0 else d
    return float(f"{d:.2g}")


def emulate(em: dict, grid, dx: np.ndarray) -> np.ndarray:
    """ΔT at every point for a per-point change ``dx`` of the lever (the same
    algorithm the results page runs in the browser)."""
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve

    dx = np.asarray(dx, float)
    out = em["own"] * dx
    m = grid.mask.astype(float)
    for ch in em["channels"]:
        src = grid.rasterize(ch["weight"] * dx, fill=0.0) * m
        sm = gaussian_filter(np.nan_to_num(src), ch["sigma_cells"], mode="constant", truncate=4.0)
        out = out + ch["coef"] * grid.sample(sm)
    if em.get("physics"):
        ph = em["physics"]
        src = np.nan_to_num(grid.rasterize(ph["dq"] * dx, fill=0.0))
        out = out + grid.sample(fftconvolve(src, ph["kernel"], mode="same"))
    return out


def _patch(grid, centre: int, radius_cells: float, data) -> np.ndarray:
    d = np.hypot(grid.ix - grid.ix[centre], grid.iy - grid.iy[centre])
    return d <= radius_cells


def validate(engine, em: dict, var: str, dose: float, n_patches: int = 12, radii_m=(90.0, 250.0),
             seed: int = 0) -> dict:
    """Emulator vs exact engine: random circular patch edits and a uniform edit."""
    from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec  # noqa: F401

    data, g = engine.data, engine.data.grid
    rng = np.random.default_rng(seed)
    lo, hi = engine._bounds(var)
    x = data.frame[var].to_numpy(float)
    rows = []
    for r_m in radii_m:
        for c in rng.choice(data.n, n_patches, replace=False):
            sel = _patch(g, int(c), r_m / g.dx, data)
            dx = np.where(sel, np.clip(x + dose, lo, hi) - x, 0.0)
            exact = engine.run(ScenarioSpec(name="patch", interventions=[
                Intervention(var, "add", 0.0, per_point=dx)])).delta
            em_d = emulate(em, g, dx)
            near = _patch(g, int(c), (r_m + 500.0) / g.dx, data)
            rows.append({"radius_m": r_m, "cells": int(sel.sum()),
                         "exact_patch_mean": float(exact[sel].mean()), "emu_patch_mean": float(em_d[sel].mean()),
                         "exact_area_sum": float(exact[near].sum()), "emu_area_sum": float(em_d[near].sum()),
                         "p95_abs_err": float(np.percentile(np.abs(exact[near] - em_d[near]), 95))})
    dx = np.clip(x + dose, lo, hi) - x
    exact = engine.run(ScenarioSpec(name="uniform", interventions=[Intervention(var, "add", 0.0, per_point=dx)])).delta
    em_d = emulate(em, g, dx)
    pm_err = [abs(r["emu_patch_mean"] - r["exact_patch_mean"]) for r in rows]
    rel = [abs(r["emu_patch_mean"] - r["exact_patch_mean"]) / max(abs(r["exact_patch_mean"]), 1e-9) for r in rows]
    ok = [e <= 0.05 or q <= 0.10 for e, q in zip(pm_err, rel)]
    return {"variable": var, "dose": float(dose), "patches": rows,
            "patch_mean_abs_err_median": float(np.median(pm_err)), "patch_mean_rel_err_median": float(np.median(rel)),
            "patch_pass_rate": float(np.mean(ok)),
            "p95_cell_err_median": float(np.median([r["p95_abs_err"] for r in rows])),
            "uniform": {"exact_mean": float(exact.mean()), "emu_mean": float(em_d.mean()),
                        "rel_err": float(abs(em_d.mean() - exact.mean()) / max(abs(exact.mean()), 1e-9))}}


def emulator_for_run(run_dir, cfg, validate_doses: dict | None = None, n_patches: int = 8) -> dict:
    """Build and validate emulators for every actionable lever of a finished
    run (from its checkpoint); writes ``emulator.npz`` and ``emulator.json``."""
    import json
    import pickle
    from pathlib import Path

    from sparc.core.baselines import load_run
    from sparc.core.mediators import MediatorChain
    from sparc.core.response import ResponseEngine
    from sparc.core.scenarios import ScenarioEngine

    run_dir = Path(run_dir)
    data, folds, m, pred = load_run(run_dir, cfg)
    with open(run_dir / "checkpoint.pkl", "rb") as fh:
        st = pickle.load(fh)
    ens, inf = st["ensemble"], st["influence"]
    med = MediatorChain(cfg.mediators).fit(data.frame) if cfg.mediators else None
    eng = ScenarioEngine(data, cfg, ens, dict(inf.ranges_m), med)
    resp = ResponseEngine(eng, influence_scales=cfg.raw["influence"].get("scales", (0.5, 1.0, 2.0)))
    arrays, meta = {}, {"levers": {}, "kernel_cells": None}
    for var, spec in cfg.actionable.items():
        em = build_emulator(resp, var)
        dose = (validate_doses or {}).get(var) or design_dose(cfg, var, data.frame[var].to_numpy(float))
        dose = -abs(dose) if str(spec.get("direction", "increase")) == "decrease" else abs(dose)
        val = validate(eng, em, var, dose, n_patches=n_patches)
        arrays[f"{var}__own"] = em["own"]
        chans = []
        for c, ch in enumerate(em["channels"]):
            arrays[f"{var}__coef{c}"] = ch["coef"]
            wkey = f"{var}__w__{ch['column']}"
            if wkey not in arrays:
                arrays[wkey] = ch["weight"]
            chans.append({"column": ch["column"], "feature": ch["feature"], "sigma_cells": ch["sigma_cells"],
                          "coef": f"{var}__coef{c}", "weight": wkey,
                          "unit_weight": bool(np.allclose(ch["weight"], 1.0))})
        if em["physics"]:
            arrays[f"{var}__dq"] = em["physics"]["dq"]
            arrays["physics_kernel"] = em["physics"]["kernel"]
            meta["kernel_cells"] = em["physics"]["kernel_cells"]
        meta["levers"][var] = {"channels": chans, "physics": bool(em["physics"]), "validation": val, "design_dose": dose,
                               "bounds": list(eng._bounds(var)), "direction": spec.get("direction", "increase")}
        log.info("emulator %s: patch-mean error median %.3f (rel %.0f%%), p95 cell error %.3f, uniform rel %.0f%%",
                 var, val["patch_mean_abs_err_median"], 100 * val["patch_mean_rel_err_median"],
                 val["p95_cell_err_median"], 100 * val["uniform"]["rel_err"])
    np.savez_compressed(run_dir / "emulator.npz", ids=data.ids, **arrays)
    (run_dir / "emulator.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    return meta
