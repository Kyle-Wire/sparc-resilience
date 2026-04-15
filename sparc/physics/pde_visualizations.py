"""
PDE physics diagnostic visualizations for SPARC V3.

Generates matplotlib plots for:
  - Stage 2: alpha field, blend weights, PDE residuals, loss curves,
             gradient/curvature diagnostics, boundary overlay
  - Stage 4: tier comparison, alpha-modulated scenario maps, Gini dashboard

All PNGs saved to stage output directories are auto-discovered by
the report generator via ``_collect_plots()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Lazy matplotlib setup ────────────────────────────────────────────────
def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ── Shared helpers ───────────────────────────────────────────────────────

def _stats_text(values: np.ndarray, label: str = "") -> str:
    """One-line stats summary for annotation boxes."""
    return (
        f"{label}   N={len(values):,}\n"
        f"mean={np.nanmean(values):.4f}  "
        f"std={np.nanstd(values):.4f}\n"
        f"min={np.nanmin(values):.4f}  "
        f"max={np.nanmax(values):.4f}"
    )


def _save(fig, path: Path, dpi: int = 250) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt = _get_plt()
    plt.close(fig)
    logger.info("  Saved %s", path.name)


def _scatter_map(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap: str,
    label: str,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    s: float = 0.8,
) -> None:
    """Standard SPARC scatter-map on an axis."""
    sc = ax.scatter(
        x, y, c=values, cmap=cmap, s=s,
        vmin=vmin, vmax=vmax, edgecolors="none",
    )
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    stats = _stats_text(values, label)
    ax.text(
        0.02, 0.02, stats, transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )


# =====================================================================
#  Stage 2 — Training-time PDE diagnostics
# =====================================================================

def generate_stage2_pde_plots(
    *,
    coords: np.ndarray,
    alpha: np.ndarray,
    T_pred: np.ndarray,
    y_true: np.ndarray,
    neighbor_idx: np.ndarray,
    h: float | np.ndarray,
    w_source: np.ndarray | None = None,
    loss_history: list[dict[str, float]] | None = None,
    output_dir: Path,
) -> None:
    """
    Generate all Stage 2 PDE diagnostic plots.

    Parameters
    ----------
    coords : (N, 2) — lon/lat or projected x/y
    alpha : (N,) — learned process rate α(s)
    T_pred : (N,) — full-retrain predicted values (normalized)
    y_true : (N,) — target values (normalized)
    neighbor_idx : (N, 4) int64 — cardinal neighbors [N,S,E,W]
    h : scalar or (N,) — grid spacing
    w_source : (N,) or None — blend weight from PDEInformedPhysicsEncoder
    loss_history : list of per-epoch component dicts, or None
    output_dir : directory to save PNGs (typically stage2_dir/v2_neural)
    """
    import torch

    plt = _get_plt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = coords[:, 0], coords[:, 1]
    N = len(coords)

    # Convert to tensors for PDE operator calls
    T_t = torch.tensor(T_pred, dtype=torch.float32)
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    nb_t = torch.tensor(neighbor_idx, dtype=torch.long)
    h_val = h if isinstance(h, (int, float)) else torch.tensor(h, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 1. Alpha field map
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    _scatter_map(
        ax, x, y, alpha, cmap="viridis",
        label="α(s)", title="Learned Spatial Responsiveness α(s)",
    )
    _save(fig, output_dir / "pde_01_alpha_field.png")

    # ------------------------------------------------------------------
    # 2. Alpha histogram + Gini
    # ------------------------------------------------------------------
    from sparc.interventions.scenario_simulator import spatial_gini_coefficient
    gini = spatial_gini_coefficient(alpha)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(alpha, bins=50, color="#602468", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(alpha), color="#E74C3C", ls="--", lw=2, label=f"mean = {np.mean(alpha):.4f}")
    ax.set_xlabel("α(s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"α(s) Distribution  —  Gini = {gini:.3f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(
        0.98, 0.95,
        f"N = {N:,}\nstd = {np.std(alpha):.4f}\nGini = {gini:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    _save(fig, output_dir / "pde_02_alpha_histogram.png")

    # ------------------------------------------------------------------
    # 3. Blend weight w(s) map (if available)
    # ------------------------------------------------------------------
    if w_source is not None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 9))
        _scatter_map(
            ax, x, y, w_source, cmap="RdYlBu",
            label="w(s)", title="Source-Driven Blend Weight w(s)",
            vmin=0, vmax=1,
        )
        _save(fig, output_dir / "pde_03_blend_weight.png")

    # ------------------------------------------------------------------
    # 4. Heat diffusion residual map:  r = α∇²T - S
    # ------------------------------------------------------------------
    from sparc.physics.pde_operators import laplacian, gradient_magnitude, directional_curvatures
    lap_vals, valid_lap = laplacian(T_t, nb_t, h_val)
    lap_full = np.zeros(N)
    valid_np = valid_lap.numpy()
    lap_full[valid_np] = lap_vals.numpy()

    heat_residual = alpha * lap_full  # α∇²T (source term not available here, just show α∇²T)
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    vabs = np.percentile(np.abs(heat_residual[valid_np]), 95)
    _scatter_map(
        ax, x, y, heat_residual, cmap="RdBu_r",
        label="α∇²T", title="Heat Diffusion Term α∇²T",
        vmin=-vabs, vmax=vabs,
    )
    _save(fig, output_dir / "pde_04_heat_diffusion.png")

    # ------------------------------------------------------------------
    # 5. Laplacian map
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    vabs_lap = np.percentile(np.abs(lap_full[valid_np]), 95)
    _scatter_map(
        ax, x, y, lap_full, cmap="RdBu_r",
        label="∇²T", title="Laplacian ∇²T",
        vmin=-vabs_lap, vmax=vabs_lap,
    )
    _save(fig, output_dir / "pde_05_laplacian.png")

    # ------------------------------------------------------------------
    # 6. Gradient magnitude map
    # ------------------------------------------------------------------
    grad_mag, df_dx, df_dy, valid_g = gradient_magnitude(T_t, nb_t, h_val)
    grad_full = np.zeros(N)
    valid_g_np = valid_g.numpy()
    grad_full[valid_g_np] = grad_mag.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    _scatter_map(
        ax, x, y, grad_full, cmap="hot_r",
        label="|∇T|", title="Gradient Magnitude |∇T|",
        vmin=0,
    )
    _save(fig, output_dir / "pde_06_gradient_magnitude.png")

    # ------------------------------------------------------------------
    # 7. Anisotropy map: |∂²T/∂x² - ∂²T/∂y²|
    # ------------------------------------------------------------------
    d2_dx2, d2_dy2, valid_d = directional_curvatures(T_t, nb_t, h_val)
    aniso_full = np.zeros(N)
    valid_d_np = valid_d.numpy()
    aniso_full[valid_d_np] = np.abs(d2_dx2.numpy() - d2_dy2.numpy())

    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    _scatter_map(
        ax, x, y, aniso_full, cmap="YlOrRd",
        label="|∂²T/∂x² − ∂²T/∂y²|", title="Anisotropy (Directional Curvature Difference)",
        vmin=0,
    )
    _save(fig, output_dir / "pde_07_anisotropy.png")

    # ------------------------------------------------------------------
    # 8. Boundary point overlay
    # ------------------------------------------------------------------
    from sparc.physics.boundary_conditions import detect_boundary_points
    boundary, _edge_masks = detect_boundary_points(nb_t)
    boundary_np = boundary.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    ax.scatter(
        x[~boundary_np], y[~boundary_np],
        c="#CCCCCC", s=0.4, label="Interior", edgecolors="none",
    )
    ax.scatter(
        x[boundary_np], y[boundary_np],
        c="#E74C3C", s=1.5, label="Boundary", edgecolors="none",
    )
    ax.set_title("Boundary Point Detection", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    n_bnd = boundary_np.sum()
    ax.text(
        0.02, 0.02,
        f"Interior: {N - n_bnd:,}  |  Boundary: {n_bnd:,}  ({100*n_bnd/N:.1f}%)",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    _save(fig, output_dir / "pde_08_boundary_points.png")

    # ------------------------------------------------------------------
    # 9. 4-panel diagnostic dashboard
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    _scatter_map(axes[0, 0], x, y, alpha, "viridis", "α(s)", "α(s) Field")
    _scatter_map(axes[0, 1], x, y, lap_full, "RdBu_r", "∇²T", "Laplacian",
                 vmin=-vabs_lap, vmax=vabs_lap)
    _scatter_map(axes[1, 0], x, y, grad_full, "hot_r", "|∇T|", "Gradient Magnitude",
                 vmin=0)
    _scatter_map(axes[1, 1], x, y, aniso_full, "YlOrRd",
                 "|∂²T/∂x² − ∂²T/∂y²|", "Anisotropy", vmin=0)
    fig.suptitle("PDE Physics Diagnostics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "pde_09_diagnostics_dashboard.png")

    # ------------------------------------------------------------------
    # 10. Loss convergence curves (if loss_history provided)
    # ------------------------------------------------------------------
    if loss_history and len(loss_history) > 1:
        _plot_loss_curves(loss_history, output_dir)

    logger.info("Stage 2 PDE visualizations complete (%s)", output_dir)


def _plot_loss_curves(
    loss_history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot training loss curves with PDE sub-term breakdown."""
    plt = _get_plt()
    epochs = np.arange(1, len(loss_history) + 1)

    # Determine which keys exist
    all_keys = set()
    for d in loss_history:
        all_keys.update(d.keys())

    # ---- A: Total loss ----
    if "total" in all_keys:
        total = [d.get("total", 0) for d in loss_history]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(epochs, total, color="#602468", lw=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Total Loss", fontsize=12)
        ax.set_title("Training Loss", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir / "pde_10_loss_total.png", dpi=200)

    # ---- B: Core terms ----
    core_terms = {
        "mse": ("#3498DB", "MSE"),
        "physics": ("#E74C3C", "Physics ResidualOld"),
        "cross_entropy": ("#2ECC71", "Cross-Entropy"),
        "neighborhood": ("#F39C12", "Neighbor"),
        "alpha_prior": ("#9B59B6", "α Prior"),
    }
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for key, (color, label) in core_terms.items():
        if key in all_keys:
            vals = [d.get(key, 0) for d in loss_history]
            if max(vals) > 0:
                ax.plot(epochs, vals, color=color, lw=1.5, label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Core Loss Components", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "pde_11_loss_core.png", dpi=200)

    # ---- C: PDE sub-terms ----
    pde_terms = {
        "pde_heat_diffusion": ("#E74C3C", "Heat Diffusion"),
        "pde_energy_balance": ("#3498DB", "Energy Balance"),
        "pde_directional": ("#2ECC71", "Directional"),
        "pde_anisotropy": ("#F39C12", "Anisotropy"),
        "pde_gradient_flux": ("#9B59B6", "Gradient Flux"),
        "pde_gaussian_curv": ("#1ABC9C", "Gaussian Curv"),
        "pde_alpha_smooth": ("#E67E22", "α Smooth"),
        "pde_alpha_prior": ("#95A5A6", "α Prior (PDE)"),
    }
    has_pde = any(k in all_keys for k in pde_terms)
    if has_pde:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for key, (color, label) in pde_terms.items():
            if key in all_keys:
                vals = [d.get(key, 0) for d in loss_history]
                if max(vals) > 0:
                    ax.plot(epochs, vals, color=color, lw=1.5, label=label)
        # Mark staged activation epochs
        for act_epoch, act_label in [(1, "heat"), (10, "+energy"), (20, "+dir/aniso"), (30, "+all")]:
            if act_epoch <= len(epochs):
                ax.axvline(act_epoch, color="grey", ls=":", alpha=0.5, lw=0.8)
                ax.text(act_epoch + 0.5, ax.get_ylim()[1] * 0.9, act_label,
                        fontsize=7, color="grey", rotation=90, va="top")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("PDE Sub-Loss", fontsize=12)
        ax.set_title("PDE Loss — Staged Sub-Curriculum", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir / "pde_12_loss_pde_terms.png", dpi=200)

    # ---- D: BC + IC terms ----
    aux_terms = {
        "pde_total": ("#E74C3C", "PDE Total"),
        "bc_total": ("#3498DB", "BC Total"),
        "ic_total": ("#2ECC71", "IC Total"),
    }
    has_aux = any(k in all_keys and max(d.get(k, 0) for d in loss_history) > 0
                  for k in aux_terms)
    if has_aux:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for key, (color, label) in aux_terms.items():
            if key in all_keys:
                vals = [d.get(key, 0) for d in loss_history]
                if max(vals) > 0:
                    ax.plot(epochs, vals, color=color, lw=1.5, label=label)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("PDE / BC / IC Loss Terms", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir / "pde_13_loss_pde_bc_ic.png", dpi=200)


# =====================================================================
#  Stage 4 — Scenario-time PDE diagnostics
# =====================================================================

def generate_stage4_pde_plots(
    *,
    results_df,
    coords: np.ndarray,
    alpha_field: np.ndarray | None,
    alpha_coords: np.ndarray | None,
    tier_sources: dict[str, list[str]] | None = None,
    output_dir: Path,
) -> None:
    """
    Generate Stage 4 PDE-aware scenario diagnostic plots.

    Parameters
    ----------
    results_df : DataFrame with scenario results (must have lon/lat + delta cols)
    coords : (N, 2) — data point coordinates
    alpha_field : (M,) — alpha values (may differ from N if interpolated)
    alpha_coords : (M, 2) — coordinates for alpha field
    tier_sources : {variable: [source_per_point]} — tier attribution per variable
    output_dir : scenario maps directory
    """
    import pandas as pd

    plt = _get_plt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if alpha_field is None:
        logger.info("No alpha field available — skipping Stage 4 PDE plots")
        return

    x_a, y_a = alpha_coords[:, 0], alpha_coords[:, 1]

    # ------------------------------------------------------------------
    # 1. Alpha-normalized field map for scenario context
    # ------------------------------------------------------------------
    alpha_norm = alpha_field / (np.mean(alpha_field) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _scatter_map(
        axes[0], x_a, y_a, alpha_field, "viridis",
        "α(s)", "Raw α(s) Field",
    )
    _scatter_map(
        axes[1], x_a, y_a, alpha_norm, "RdYlBu_r",
        "α(s)/mean(α)", "Normalized α — Spatial Multiplier",
        vmin=0, vmax=2.0,
    )
    fig.suptitle("PDE α(s) Field for Scenario Modulation", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "pde_s4_01_alpha_context.png")

    # ------------------------------------------------------------------
    # 2. Tier attribution summary (if available)
    # ------------------------------------------------------------------
    if tier_sources:
        from sparc.interventions.scenario_simulator import spatial_gini_coefficient

        tier_labels = ["pde_alpha", "mgwr", "pdp", "literature", "other"]
        tier_colors = ["#602468", "#3498DB", "#2ECC71", "#F39C12", "#95A5A6"]

        n_vars = len(tier_sources)
        fig, axes = plt.subplots(1, max(n_vars, 1), figsize=(5 * max(n_vars, 1) + 1, 5))
        if n_vars == 1:
            axes = [axes]

        for ax, (var, sources) in zip(axes, tier_sources.items()):
            counts = {}
            for src in sources:
                bucket = "other"
                for tl in tier_labels[:-1]:
                    if tl in src.lower().replace(" ", "_"):
                        bucket = tl
                        break
                counts[bucket] = counts.get(bucket, 0) + 1
            labels_present = [l for l in tier_labels if counts.get(l, 0) > 0]
            sizes = [counts[l] for l in labels_present]
            colors = [tier_colors[tier_labels.index(l)] for l in labels_present]

            ax.pie(sizes, labels=labels_present, colors=colors, autopct="%1.0f%%",
                   textprops={"fontsize": 9})
            ax.set_title(var, fontsize=11, fontweight="bold")

        fig.suptitle("Tier Attribution per Variable", fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, output_dir / "pde_s4_02_tier_attribution.png")

    # ------------------------------------------------------------------
    # 3. Spatial Gini comparison across scenario variables
    # ------------------------------------------------------------------
    if "delta_total" in results_df.columns or any("delta" in c for c in results_df.columns):
        from sparc.interventions.scenario_simulator import spatial_gini_coefficient

        delta_cols = [c for c in results_df.columns if c.startswith("delta_")]
        if delta_cols:
            gini_data = {}
            for col in delta_cols:
                vals = results_df[col].values
                finite = vals[np.isfinite(vals)]
                if len(finite) > 0:
                    gini_data[col.replace("delta_", "")] = spatial_gini_coefficient(np.abs(finite))

            if gini_data:
                fig, ax = plt.subplots(1, 1, figsize=(max(8, len(gini_data) * 1.5), 5))
                names = list(gini_data.keys())
                ginis = [gini_data[n] for n in names]
                bars = ax.bar(names, ginis, color="#602468", alpha=0.85, edgecolor="white")
                ax.set_ylabel("Spatial Gini Coefficient", fontsize=12)
                ax.set_title("Spatial Inequality of Scenario Effects", fontsize=13, fontweight="bold")
                ax.set_ylim(0, min(1.0, max(ginis) * 1.3 + 0.05))
                for bar, g in zip(bars, ginis):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{g:.3f}", ha="center", fontsize=9)
                ax.tick_params(axis="x", rotation=30, labelsize=9)
                fig.tight_layout()
                _save(fig, output_dir / "pde_s4_03_gini_comparison.png")

    logger.info("Stage 4 PDE visualizations complete (%s)", output_dir)
