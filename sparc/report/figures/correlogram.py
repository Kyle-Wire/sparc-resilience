"""On-demand correlogram figure renderer.

Reads ``stage=0`` artifact ``correlogram_results`` (struct) from the
active ArtifactStore and rebuilds the per-variable enhanced correlogram
plot the pipeline used to write to disk.

The plot is identical in spirit to the legacy version produced by
``sparc.run.correlogram_analysis`` but is generated *only when asked*.
"""

from __future__ import annotations

import io
from typing import Any

from sparc.report.render import register_figure_renderer, RenderError


@register_figure_renderer("correlogram")
def render_correlogram(store, stage, artifact_id, *, variable: str | None = None, **_: Any) -> bytes:
    """Render the enhanced correlogram for one variable as PNG bytes.

    Parameters
    ----------
    store : ArtifactStore
    stage, artifact_id :
        Identifies the comprehensive correlogram struct
        (typically ``stage='0'``, ``artifact_id='correlogram_results'``).
    variable :
        Variable name to plot. If ``None``, the first variable in
        ``individual_results`` is used.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    payload = store.read_struct(stage, artifact_id)
    if not isinstance(payload, dict):
        raise RenderError("correlogram payload is not a dict")

    individual = payload.get("individual_results") or {}
    if not individual:
        raise RenderError("correlogram payload has no individual_results")

    var = variable or next(iter(individual))
    if var not in individual:
        raise RenderError(f"variable {var!r} not found in correlogram results")

    info = individual[var]
    inner = info.get("correlogram_results", {})
    cdata = inner.get("correlogram_results") if isinstance(inner, dict) else None
    if cdata is None:
        cdata = info.get("correlogram_results")

    if isinstance(cdata, dict):
        distances = np.asarray(cdata.get("lag_distances", []), dtype=float)
        morans_i = np.asarray(cdata.get("morans_i_values", []), dtype=float)
        p_values = np.asarray(cdata.get("p_values", []), dtype=float)
    elif isinstance(cdata, list):
        distances = np.asarray([r.get("lag_distance", 0.0) for r in cdata], dtype=float)
        morans_i = np.asarray([r.get("morans_i", 0.0) for r in cdata], dtype=float)
        p_values = np.asarray([r.get("p_value", 1.0) for r in cdata], dtype=float)
    else:
        raise RenderError("unexpected correlogram_results shape")

    if distances.size == 0:
        raise RenderError(f"no correlogram lags for variable {var!r}")

    optimal_bandwidth = float(info.get("optimal_bandwidth", 0.0))
    significant = p_values < 0.05
    moran_q = np.percentile(morans_i, [25, 50, 75])
    dist_q = np.percentile(distances, [25, 50, 75])
    max_idx = int(np.argmax(morans_i))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = ["#d62728" if s else "#1f77b4" for s in significant]
    sizes = [60 if s else 30 for s in significant]
    ax1.scatter(
        distances, morans_i, c=colors, s=sizes, alpha=0.7,
        edgecolors="black", linewidth=0.5,
    )
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5, label="No Autocorrelation")
    ax1.axhline(moran_q[1], color="orange", linestyle=":", alpha=0.7,
                label=f"Median Moran's I ({moran_q[1]:.3f})")
    ax1.axvline(dist_q[0], color="blue", linewidth=2, alpha=0.8, label=f"Q1 Block ({dist_q[0]:.0f}m)")
    ax1.axvline(dist_q[1], color="purple", linewidth=2, alpha=0.8, label=f"Median Block ({dist_q[1]:.0f}m)")
    ax1.axvline(dist_q[2], color="navy", linewidth=2, alpha=0.8, label=f"Q3 Block ({dist_q[2]:.0f}m)")
    if optimal_bandwidth > 0:
        ax1.axvline(optimal_bandwidth, color="red", alpha=0.7, label=f"GWR BW ({optimal_bandwidth:.0f}m)")
    ax1.scatter(distances[max_idx], morans_i[max_idx], s=150, c="yellow",
                marker="*", edgecolors="black", linewidth=2,
                label=f"Max AC ({morans_i[max_idx]:.3f} at {distances[max_idx]:.0f}m)")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Moran's I (Spatial Autocorrelation)")
    ax1.set_title(f"Spatial Correlogram - {var}", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    edges = [0.0, dist_q[0], dist_q[1], dist_q[2], float(distances.max())]
    labels = ["Small\n(0-Q1)", "Medium\n(Q1-Med)", "Large\n(Med-Q3)", "XLarge\n(Q3-Max)"]
    bar_colors = ["#e74c3c", "#f39c12", "#f1c40f", "#27ae60"]
    means = []
    for i in range(4):
        m = (distances >= edges[i]) & (distances < edges[i + 1])
        means.append(float(np.mean(morans_i[m])) if np.any(m) else 0.0)
    ax2.bar(labels, means, color=bar_colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Mean Moran's I")
    ax2.set_xlabel("CV Block Size Category")
    ax2.set_title("Spatial Autocorrelation by Block Size", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
