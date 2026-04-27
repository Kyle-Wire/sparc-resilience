"""
Server-side matplotlib figure rendering for SPARC artifacts.

The desktop renders all visualizations live (deck.gl, Plotly, custom SVG),
but the report generator and the ``Download as PNG`` button on every page
need server-rendered images. This module owns that responsibility.

Rules
-----
* Headless ``matplotlib.use("Agg")`` — no display required.
* Every renderer reads exclusively through :class:`sparc.registry.store.ArtifactStore`;
  no disk fallbacks, no ``.gpkg`` opens.
* Output is always PNG bytes at the requested ``dpi`` (default 144).
* Renderers MUST NOT mutate state (no figure caching here — caching lives
  one layer up via :func:`render_for_artifact`).
* Missing data / unsupported artifact -> :class:`FigureRenderError`.

Public API
----------
* :data:`FIGURES_REGISTRY` — ``{(stage, artifact_id): renderer}``
* :func:`render_for_artifact` — dispatch + cache helper used by
  ``GET /artifacts/{stage}/{artifact_id}.png`` and the report generator.
"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sparc.registry.run_registry import RunRegistry, get_active_registry  # noqa: E402
from sparc.registry.store import ArtifactStore  # noqa: E402
from sparc.scenario.bundle import ScenarioBundle  # noqa: E402
from sparc.scenario.contract import (  # noqa: E402
    SCENARIO_ATTRIBUTION,
    SCENARIO_LIBRARY,
    SCENARIO_MC_UNCERTAINTY,
    SCENARIO_TRAJECTORY,
)


__all__ = [
    "FigureRenderError",
    "FIGURES_REGISTRY",
    "render_for_artifact",
]


# ---------------------------------------------------------------------------
# Errors / helpers
# ---------------------------------------------------------------------------


class FigureRenderError(RuntimeError):
    """Raised when a PNG cannot be rendered for the given artifact."""


Renderer = Callable[[ArtifactStore, str, str, dict[str, Any]], bytes]


def _resolve_store(registry: Optional[RunRegistry] = None) -> ArtifactStore:
    reg = registry or get_active_registry()
    if reg is None:
        raise FigureRenderError("No active RunRegistry")
    return ArtifactStore(reg)


def _fig_to_png(fig: "plt.Figure", dpi: int = 144) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _require_table(store: ArtifactStore, stage: str, artifact_id: str):
    if not store.has(stage, artifact_id):
        raise FigureRenderError(f"Missing artifact {stage}/{artifact_id}")
    try:
        return store.read_table(stage, artifact_id)
    except Exception as exc:  # noqa: BLE001
        raise FigureRenderError(f"Cannot read table {stage}/{artifact_id}: {exc}") from exc


def _require_struct(store: ArtifactStore, stage: str, artifact_id: str):
    if not store.has(stage, artifact_id):
        raise FigureRenderError(f"Missing artifact {stage}/{artifact_id}")
    try:
        return store.read_struct(stage, artifact_id)
    except Exception as exc:  # noqa: BLE001
        raise FigureRenderError(f"Cannot read struct {stage}/{artifact_id}: {exc}") from exc


def _plot_geo(ax, gdf, column: str, *, cmap: str = "magma", legend: bool = True):
    """Plot a GeoDataFrame column with sensible defaults; falls back to centroids."""
    try:
        gdf.plot(ax=ax, column=column, cmap=cmap, legend=legend, markersize=8)
    except Exception:
        # Geometry-less or invalid CRS: fall back to scatter on representative_point.
        try:
            pts = gdf.geometry.representative_point()
            ax.scatter(pts.x, pts.y, c=gdf[column], cmap=cmap, s=8)
        except Exception as exc:  # noqa: BLE001
            raise FigureRenderError(f"cannot plot column {column!r}: {exc}") from exc
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Stage 0 — data overview
# ---------------------------------------------------------------------------


def render_data_histogram(store, stage, artifact_id, opts):
    """Histogram of the loaded dataset's target column (or first numeric col)."""
    df = _require_table(store, stage, artifact_id)
    col = opts.get("column")
    if col is None:
        numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not numeric:
            raise FigureRenderError("no numeric columns to histogram")
        col = numeric[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    series = df[col].dropna()
    ax.hist(series, bins=int(opts.get("bins", 30)), color="#7a3a3a", alpha=0.85, edgecolor="white")
    ax.set_title(f"{col} (n={len(series)})")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Stage 1 — causal discovery / DAG
# ---------------------------------------------------------------------------


def render_dag(store, stage, artifact_id, opts):
    """Render the inferred DAG (edge list -> spring layout)."""
    payload = _require_struct(store, stage, artifact_id)
    edges = payload.get("edges") or payload.get("edge_list") or []
    if not edges:
        raise FigureRenderError("DAG struct has no edges")
    try:
        import networkx as nx
    except ImportError as exc:
        raise FigureRenderError("networkx required for DAG rendering") from exc
    g = nx.DiGraph()
    for e in edges:
        if isinstance(e, dict):
            src, dst = e.get("source"), e.get("target")
            w = e.get("weight") or e.get("probability") or 1.0
        else:
            src, dst = e[0], e[1]
            w = e[2] if len(e) > 2 else 1.0
        if src is not None and dst is not None:
            g.add_edge(src, dst, weight=float(w))
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(g, seed=42)
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#fdfbf7", edgecolors="#7a3a3a", node_size=900)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        g, pos, ax=ax, width=[1 + 3 * w for w in weights], edge_color="#7a3a3a", arrows=True, arrowsize=12
    )
    ax.set_axis_off()
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Stage 2 — predictions / CATE
# ---------------------------------------------------------------------------


def render_predictions_map(store, stage, artifact_id, opts):
    gdf = _require_table(store, stage, artifact_id)
    col = opts.get("column", "prediction")
    if col not in gdf.columns:
        # Try common fallbacks.
        for candidate in ("pred", "y_hat", "prediction_mean", "mean"):
            if candidate in gdf.columns:
                col = candidate
                break
        else:
            raise FigureRenderError(f"no prediction column on {artifact_id}")
    fig, ax = plt.subplots(figsize=(7, 7))
    _plot_geo(ax, gdf, col, cmap=opts.get("cmap", "magma"))
    ax.set_title(opts.get("title", "Predictions"))
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_cate(store, stage, artifact_id, opts):
    df = _require_table(store, stage, artifact_id)
    col = opts.get("column", "cate")
    if col not in df.columns:
        for candidate in ("CATE", "tau", "treatment_effect"):
            if candidate in df.columns:
                col = candidate
                break
        else:
            raise FigureRenderError(f"no CATE column on {artifact_id}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[col].dropna(), bins=int(opts.get("bins", 40)), color="#7a3a3a", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#444", linestyle="--", linewidth=1)
    ax.set_title("Conditional average treatment effect")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Stage 3 — sensitivity / dose-response / posteriors
# ---------------------------------------------------------------------------


def render_sensitivity_tornado(store, stage, artifact_id, opts):
    payload = _require_struct(store, stage, artifact_id)
    items = payload.get("items") or payload.get("variables") or []
    if not items:
        raise FigureRenderError("sensitivity_tornado has no items")
    items = sorted(items, key=lambda r: abs(float(r.get("effect", 0.0))), reverse=True)
    names = [str(r.get("name", "?")) for r in items]
    effects = [float(r.get("effect", 0.0)) for r in items]
    lows = [float(r.get("low", e)) for r, e in zip(items, effects)]
    highs = [float(r.get("high", e)) for r, e in zip(items, effects)]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    for yi, lo, hi, eff in zip(y, lows, highs, effects):
        ax.plot([lo, hi], [yi, yi], color="#7a3a3a", linewidth=4, solid_capstyle="butt")
        ax.plot([eff], [yi], "o", color="#222")
    ax.axvline(0, color="#444", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("effect")
    ax.set_title("Sensitivity tornado")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_dose_response(store, stage, artifact_id, opts):
    df = _require_table(store, stage, artifact_id)
    x_col = opts.get("x") or ("dose" if "dose" in df.columns else df.columns[0])
    y_col = opts.get("y") or ("response" if "response" in df.columns else df.columns[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[x_col], df[y_col], color="#7a3a3a", linewidth=1.5)
    if {"low", "high"}.issubset(df.columns):
        ax.fill_between(df[x_col], df["low"], df["high"], color="#7a3a3a", alpha=0.18)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Dose–response")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_posteriors(store, stage, artifact_id, opts):
    payload = _require_struct(store, stage, artifact_id)
    series = payload.get("samples") or payload.get("posteriors") or {}
    if not series:
        raise FigureRenderError("no posterior samples")
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.6 * n + 0.5), squeeze=False)
    for ax, (name, samples) in zip(axes[:, 0], series.items()):
        arr = np.asarray(samples, dtype=float).ravel()
        ax.hist(arr, bins=40, color="#7a3a3a", alpha=0.85, edgecolor="white")
        ax.set_title(str(name), fontsize=10)
    fig.tight_layout()
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_posterior_trace(store, stage, artifact_id, opts):
    payload = _require_struct(store, stage, artifact_id)
    chains = payload.get("chains") or payload.get("traces") or {}
    if not chains:
        raise FigureRenderError("no trace data")
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, samples in chains.items():
        arr = np.asarray(samples, dtype=float).ravel()
        ax.plot(arr, linewidth=0.8, label=str(name))
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("iteration")
    ax.set_title("Posterior trace")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Stage 4 — scenarios
# ---------------------------------------------------------------------------


def _scenario_index_from_opts(bundle: ScenarioBundle, opts: dict[str, Any]) -> int:
    sid = opts.get("scenario_id") or opts.get("scenario_index")
    scenarios = bundle.list_scenarios()
    if not scenarios:
        raise FigureRenderError("no scenarios available")
    if sid is None:
        return scenarios[0]
    try:
        return int(sid)
    except (TypeError, ValueError):
        return scenarios[0]


def render_scenario_map(store, stage, artifact_id, opts):
    bundle = ScenarioBundle.from_store(store)
    if not bundle.list_scenarios() or bundle.results is None:
        raise FigureRenderError("ScenarioBundle has no scenarios")
    idx = _scenario_index_from_opts(bundle, opts)
    sc = bundle.get_scenario(idx)
    if sc is None:
        raise FigureRenderError(f"scenario {idx} unavailable")
    gdf = bundle.results
    column = opts.get("column") or sc.delta_column or sc.pred_column
    if column not in gdf.columns:
        column = sc.pred_column
    fig, ax = plt.subplots(figsize=(7, 7))
    _plot_geo(ax, gdf, column, cmap=opts.get("cmap", "RdBu_r"))
    ax.set_title(f"Scenario {idx} · {column}")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_uncertainty_band_map(store, stage, artifact_id, opts):
    bundle = ScenarioBundle.from_store(store)
    if not bundle.has_uncertainty() or bundle.results is None:
        raise FigureRenderError("no uncertainty bands available")
    idx = _scenario_index_from_opts(bundle, opts)
    sc = bundle.get_scenario(idx)
    if sc is None:
        raise FigureRenderError(f"scenario {idx} unavailable")
    gdf = bundle.results
    band_col = None
    for cand in ("pred_std", "std", "pred_p95"):
        if cand in gdf.columns:
            band_col = cand
            break
    if band_col is None:
        raise FigureRenderError("no uncertainty column")
    fig, ax = plt.subplots(figsize=(7, 7))
    _plot_geo(ax, gdf, band_col, cmap=opts.get("cmap", "viridis"))
    ax.set_title(f"Scenario {idx} · uncertainty ({band_col})")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_attribution_waterfall(store, stage, artifact_id, opts):
    df = _require_table(store, stage, artifact_id)
    name_col = next((c for c in ("driver", "factor", "name") if c in df.columns), df.columns[0])
    val_col = next(
        (c for c in ("contribution", "delta", "effect", "value") if c in df.columns),
        df.columns[1] if len(df.columns) > 1 else None,
    )
    if val_col is None:
        raise FigureRenderError("waterfall needs a value column")
    df = df.sort_values(val_col, ascending=False)
    names = df[name_col].astype(str).tolist()
    vals = df[val_col].astype(float).to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(vals)[:-1]])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names) + 1)))
    colors = ["#7a3a3a" if v >= 0 else "#3a5a7a" for v in vals]
    ax.barh(names, vals, left=cum, color=colors, edgecolor="white")
    ax.axvline(0, color="#444", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel(val_col)
    ax.set_title("Attribution waterfall")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_scenario_trajectory(store, stage, artifact_id, opts):
    df = _require_table(store, stage, artifact_id)
    t_col = next((c for c in ("t", "time", "year", "step") if c in df.columns), df.columns[0])
    val_cols = [c for c in df.columns if c.startswith("pred_") or c.startswith("delta_")]
    if not val_cols:
        val_cols = [c for c in df.columns if c != t_col][:6]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in val_cols:
        ax.plot(df[t_col], df[c], linewidth=1.4, label=c)
    ax.set_xlabel(t_col)
    ax.set_title("Scenario trajectory")
    ax.legend(fontsize=8, loc="best")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_scenario_library_timeline(store, stage, artifact_id, opts):
    payload = _require_struct(store, stage, artifact_id)
    events = payload.get("events") or payload.get("scenarios") or []
    if not events:
        raise FigureRenderError("scenario library is empty")
    rows = []
    for i, e in enumerate(events):
        rows.append((str(e.get("name") or f"scenario {i}"), float(e.get("t", i)), str(e.get("kind", ""))))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    times = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    y = np.arange(len(names))
    ax.scatter(times, y, color="#7a3a3a", s=60, edgecolor="white")
    for yi, t, n in zip(y, times, names):
        ax.text(t, yi, f"  {n}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("time")
    ax.set_title("Scenario library timeline")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Stage 5 — decision
# ---------------------------------------------------------------------------


def render_decision_allocation(store, stage, artifact_id, opts):
    payload = _require_struct(store, stage, artifact_id)
    items = payload.get("allocations") or payload.get("items") or []
    if not items:
        raise FigureRenderError("no allocations to plot")
    names = [str(r.get("name") or r.get("id") or i) for i, r in enumerate(items)]
    values = [float(r.get("value", r.get("amount", 0.0))) for r in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh(names, values, color="#7a3a3a", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("allocation")
    ax.set_title("Decision allocation")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Generic / PDP / correlogram
# ---------------------------------------------------------------------------


def render_pdp(store, stage, artifact_id, opts):
    df = _require_table(store, stage, artifact_id)
    cols = [c for c in df.columns if c not in ("x", "feature")]
    x_col = "x" if "x" in df.columns else df.columns[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in cols[:8]:
        ax.plot(df[x_col], df[c], linewidth=1.4, label=c)
    ax.set_xlabel(x_col)
    ax.set_title("Partial dependence")
    ax.legend(fontsize=8, loc="best")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


def render_correlogram(store, stage, artifact_id, opts):
    payload = _require_struct(store, stage, artifact_id)
    matrix = payload.get("matrix") or payload.get("correlation")
    labels = payload.get("labels") or payload.get("variables") or []
    if matrix is None:
        raise FigureRenderError("correlogram has no matrix")
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlogram")
    return _fig_to_png(fig, dpi=opts.get("dpi", 144))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# (stage, artifact_id) -> renderer.  Callers may register additional renderers
# by importing FIGURES_REGISTRY and assigning new keys.
FIGURES_REGISTRY: dict[tuple[str, str], Renderer] = {
    # Stage 0
    ("0", "data"): render_data_histogram,
    ("0", "data_summary"): render_data_histogram,
    ("0", "data_histogram_bins"): render_data_histogram,
    # Stage 1
    ("1", "dag"): render_dag,
    ("1", "edge_inclusion_probs"): render_dag,
    ("1", "correlogram"): render_correlogram,
    # Stage 2
    ("2", "predictions"): render_predictions_map,
    ("2", "cate"): render_cate,
    ("2", "spatial_cate"): render_predictions_map,
    ("2", "pdp"): render_pdp,
    # Stage 3
    ("3", "sensitivity_tornado"): render_sensitivity_tornado,
    ("3", "dose_response"): render_dose_response,
    ("3", "posteriors"): render_posteriors,
    ("3", "posterior_trace"): render_posterior_trace,
    # Stage 4 — scenario contract
    ("4", "scenario_results"): render_scenario_map,
    ("4", "scenario_results_dag"): render_scenario_map,
    ("4", "scenario_results_reprediction"): render_scenario_map,
    ("4", "scenario_results_hybrid"): render_scenario_map,
    ("4", SCENARIO_ATTRIBUTION): render_attribution_waterfall,
    ("4", SCENARIO_TRAJECTORY): render_scenario_trajectory,
    ("4", SCENARIO_MC_UNCERTAINTY): render_uncertainty_band_map,
    ("4", SCENARIO_LIBRARY): render_scenario_library_timeline,
    # Stage 5
    ("5", "decision_allocation"): render_decision_allocation,
}


_FIGURE_CACHE_STAGE = "report_figures"


def _cache_key(stage: str, artifact_id: str, opts: dict[str, Any]) -> str:
    parts = [stage, artifact_id]
    for k in sorted(opts):
        parts.append(f"{k}={opts[k]}")
    return "__".join(parts).replace("/", "_")


def render_for_artifact(
    stage: str | int,
    artifact_id: str,
    *,
    registry: Optional[RunRegistry] = None,
    use_cache: bool = True,
    **opts: Any,
) -> bytes:
    """Render the PNG for a given artifact, with content-hash caching.

    Looks up ``(stage, artifact_id)`` in :data:`FIGURES_REGISTRY`, calls the
    renderer with ``(store, stage_str, artifact_id, opts)``, and (optionally)
    caches the bytes back into the artifact store under the
    ``report_figures`` stage keyed by the source artifact's content hash.
    """
    stage_str = str(stage)
    key = (stage_str, artifact_id)
    renderer = FIGURES_REGISTRY.get(key)
    if renderer is None:
        raise FigureRenderError(f"No renderer registered for {stage_str}/{artifact_id}")

    store = _resolve_store(registry)
    src_entry = store.registry.lookup(stage_str, artifact_id)
    if src_entry is None:
        raise FigureRenderError(f"Unknown source artifact {stage_str}/{artifact_id}")

    cache_id = _cache_key(stage_str, artifact_id, opts)
    src_hash = getattr(src_entry, "sha256", None) or getattr(src_entry, "blob_sha256", None)

    if use_cache and store.has(_FIGURE_CACHE_STAGE, cache_id):
        cached_entry = store.registry.lookup(_FIGURE_CACHE_STAGE, cache_id)
        cached_meta = (cached_entry.metadata or {}) if cached_entry else {}
        if not src_hash or cached_meta.get("source_hash") == src_hash:
            try:
                return store.read_blob(_FIGURE_CACHE_STAGE, cache_id)
            except Exception:  # noqa: BLE001
                pass  # fall through to re-render

    png = renderer(store, stage_str, artifact_id, dict(opts))

    if use_cache:
        try:
            store.write_blob(
                _FIGURE_CACHE_STAGE,
                cache_id,
                png,
                metadata={
                    "source_stage": stage_str,
                    "source_artifact": artifact_id,
                    "source_hash": src_hash,
                    "opts": dict(opts),
                    "content_type": "image/png",
                },
            )
        except Exception:  # noqa: BLE001
            pass  # caching is best-effort

    return png
