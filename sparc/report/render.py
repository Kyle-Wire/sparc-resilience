"""
On-demand artifact rendering.

The SPARC pipeline writes artifacts into ``artifacts.db`` as tables /
structs / blobs (see :mod:`sparc.registry.store`).  User-facing files
(CSV, PNG, HTML, GeoJSON, ...) are NOT produced by the pipeline; they
are produced lazily here, when the desktop app, the report generator,
or a CLI export asks for them.

Two layers:

* **Format renderers** (`render_csv`, `render_png`, `render_geojson`,
  `render_json`) — pure functions ``(stage, artifact_id, **opts) -> bytes``.
  Each looks up the artifact via the active registry and serializes it
  to the requested format.

* **Figure renderers** — registered via :func:`register_figure_renderer`.
  Stage C migrations move existing matplotlib code from pipeline modules
  into renderer functions keyed by ``figure_kind`` (e.g. ``"dag_heatmap"``,
  ``"trace_plot"``).  Renderers receive the underlying tables/structs and
  return PNG bytes so the same pipeline run can yield arbitrarily many
  styled figures without re-executing analysis.

This module has no side effects on the registry — it is read-only.
"""

from __future__ import annotations

import io
import json
from typing import Any, Callable, Optional

from sparc.registry.run_registry import RunRegistry, get_active_registry
from sparc.registry.store import ArtifactStore, ArtifactStoreError


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RenderError(RuntimeError):
    """Raised when an artifact cannot be rendered to the requested format."""


# ---------------------------------------------------------------------------
# Registry of figure-kind renderers
# ---------------------------------------------------------------------------

# Signature: (store, stage, artifact_id, **opts) -> bytes (PNG)
FigureRenderer = Callable[..., bytes]
_FIGURE_RENDERERS: dict[str, FigureRenderer] = {}


def register_figure_renderer(figure_kind: str) -> Callable[[FigureRenderer], FigureRenderer]:
    """Decorator: register a figure renderer keyed by ``figure_kind``.

    Example::

        @register_figure_renderer("dag_heatmap")
        def _render_dag_heatmap(store, stage, artifact_id, **opts) -> bytes:
            df = store.read_table(stage, artifact_id)
            ...
            return png_bytes
    """

    def _decorate(fn: FigureRenderer) -> FigureRenderer:
        _FIGURE_RENDERERS[figure_kind] = fn
        return fn

    return _decorate


def list_figure_kinds() -> list[str]:
    try:
        import sparc.report.figures  # noqa: F401
    except Exception:
        pass
    return sorted(_FIGURE_RENDERERS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_store(registry: Optional[RunRegistry] = None) -> ArtifactStore:
    reg = registry or get_active_registry()
    if reg is None:
        raise RenderError("No active RunRegistry — cannot resolve artifact storage")
    return ArtifactStore(reg)


def _require_entry(store: ArtifactStore, stage: str | int, artifact_id: str):
    entry = store.registry.lookup(stage, artifact_id)
    if entry is None:
        raise RenderError(f"Unknown artifact: stage={stage} id={artifact_id}")
    return entry


# ---------------------------------------------------------------------------
# Format renderers
# ---------------------------------------------------------------------------


def render_csv(
    stage: str | int,
    artifact_id: str,
    *,
    registry: Optional[RunRegistry] = None,
    index: bool = False,
) -> bytes:
    """Serialize a registered table as CSV bytes."""
    store = _resolve_store(registry)
    entry = _require_entry(store, stage, artifact_id)

    if entry.storage_kind == "table":
        df = store.read_table(stage, artifact_id)
        # Drop any geometry columns — emit GeoJSON via render_geojson instead.
        try:
            import geopandas as gpd

            if isinstance(df, gpd.GeoDataFrame):
                df = df.drop(columns=[df.geometry.name])
        except ImportError:
            pass
        buf = io.StringIO()
        df.to_csv(buf, index=index)
        return buf.getvalue().encode("utf-8")

    if entry.storage_kind == "struct":
        # Best-effort: dict-of-lists -> tabular; otherwise key/value rows.
        import pandas as pd

        payload = store.read_struct(stage, artifact_id)
        try:
            df = pd.DataFrame(payload)
        except (ValueError, TypeError):
            df = pd.DataFrame(
                [(k, json.dumps(v, default=str)) for k, v in payload.items()],
                columns=["key", "value"],
            )
        buf = io.StringIO()
        df.to_csv(buf, index=index)
        return buf.getvalue().encode("utf-8")

    if entry.storage_kind == "legacy_path":
        abs_path = store.registry.resolve(entry)
        if entry.format == "csv" and abs_path.exists():
            return abs_path.read_bytes()
        raise RenderError(
            f"legacy_path artifact {artifact_id} (format={entry.format}) "
            "cannot be rendered as CSV"
        )

    raise RenderError(
        f"Cannot render {artifact_id} as CSV (storage_kind={entry.storage_kind})"
    )


def render_json(
    stage: str | int,
    artifact_id: str,
    *,
    registry: Optional[RunRegistry] = None,
    indent: int = 2,
) -> bytes:
    """Serialize a struct (or table) as JSON bytes."""
    store = _resolve_store(registry)
    entry = _require_entry(store, stage, artifact_id)

    if entry.storage_kind == "struct":
        payload = store.read_struct(stage, artifact_id)
        return json.dumps(payload, indent=indent, default=str).encode("utf-8")

    if entry.storage_kind == "table":
        df = store.read_table(stage, artifact_id)
        try:
            import geopandas as gpd

            if isinstance(df, gpd.GeoDataFrame):
                # Geometry-bearing tables export via render_geojson.
                return df.drop(columns=[df.geometry.name]).to_json(
                    orient="records"
                ).encode("utf-8")
        except ImportError:
            pass
        return df.to_json(orient="records", indent=indent).encode("utf-8")

    if entry.storage_kind == "legacy_path":
        abs_path = store.registry.resolve(entry)
        if abs_path.exists() and entry.format in ("json", "geojson"):
            return abs_path.read_bytes()

    raise RenderError(
        f"Cannot render {artifact_id} as JSON (storage_kind={entry.storage_kind})"
    )


def render_geojson(
    stage: str | int,
    artifact_id: str,
    *,
    registry: Optional[RunRegistry] = None,
) -> bytes:
    """Serialize a geometry-bearing table as GeoJSON bytes."""
    store = _resolve_store(registry)
    entry = _require_entry(store, stage, artifact_id)

    if entry.storage_kind != "table":
        raise RenderError(
            f"render_geojson requires a table; got {entry.storage_kind}"
        )

    df = store.read_table(stage, artifact_id)
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise RenderError("geopandas is required to render GeoJSON") from exc

    if not hasattr(df, "geometry"):
        raise RenderError(
            f"Artifact {artifact_id} has no geometry column; cannot render GeoJSON"
        )
    return df.to_json().encode("utf-8")


def render_png(
    figure_kind: str,
    stage: str | int,
    artifact_id: str,
    *,
    registry: Optional[RunRegistry] = None,
    **opts: Any,
) -> bytes:
    """Render a PNG via a registered figure renderer.

    ``figure_kind`` selects the renderer (e.g. ``"dag_heatmap"``,
    ``"trace_plot"``).  Renderers are registered in module-level scope by
    stage migration code via :func:`register_figure_renderer`.
    """
    # Lazy import so the built-in figure pack self-registers on first use
    # without creating an import cycle at module load time.
    try:
        import sparc.report.figures  # noqa: F401
    except Exception:
        pass

    fn = _FIGURE_RENDERERS.get(figure_kind)
    if fn is None:
        raise RenderError(
            f"No figure renderer for kind={figure_kind!r}. "
            f"Known kinds: {list_figure_kinds()}"
        )
    store = _resolve_store(registry)
    return fn(store, str(stage), artifact_id, **opts)


# ---------------------------------------------------------------------------
# Convenience: render a registered artifact in its native format.
# ---------------------------------------------------------------------------


def render_native(
    stage: str | int,
    artifact_id: str,
    *,
    registry: Optional[RunRegistry] = None,
) -> tuple[bytes, str]:
    """Return (bytes, suggested_extension) using the artifact's native format.

    Picks the most appropriate format renderer for db-resident artifacts:
    tables -> CSV (or GeoJSON if they have geometry); structs -> JSON;
    blobs -> raw bytes (caller decides how to interpret).
    """
    store = _resolve_store(registry)
    entry = _require_entry(store, stage, artifact_id)

    if entry.storage_kind == "struct":
        return render_json(stage, artifact_id, registry=registry), "json"

    if entry.storage_kind == "table":
        # Detect geometry — emit GeoJSON; otherwise CSV.
        meta = entry.metadata or {}
        if meta.get("geometry_col"):
            return render_geojson(stage, artifact_id, registry=registry), "geojson"
        return render_csv(stage, artifact_id, registry=registry), "csv"

    if entry.storage_kind in ("blob_inline", "blob_external"):
        # Re-serialize via the store; caller can decide what to do with it.
        from sparc.registry.store import ArtifactStore as _AS

        # We return the raw stored bytes (NOT the deserialized object).
        inline = entry.storage_kind == "blob_inline"
        if inline:
            with store.registry.sqlite_connection() as conn:
                row = conn.execute(
                    "SELECT bytes FROM internal_blobs WHERE stage=? AND artifact_id=?",
                    (str(stage), artifact_id),
                ).fetchone()
            if row is None:
                raise RenderError(f"Inline blob payload missing for {stage}/{artifact_id}")
            data = bytes(row[0])
        else:
            sha = entry.blob_sha256 or ""
            data = (store.blobs_dir / sha[:2] / sha).read_bytes()
        ext = (entry.metadata or {}).get("serializer", "bin")
        ext_map = {"pickle": "pkl", "joblib": "joblib", "torch": "pt",
                   "json": "json", "raw": "bin"}
        return data, ext_map.get(ext, "bin")

    if entry.storage_kind == "legacy_path":
        abs_path = store.registry.resolve(entry)
        if not abs_path.exists():
            raise RenderError(f"Legacy file missing: {abs_path}")
        return abs_path.read_bytes(), entry.format

    raise RenderError(f"Unsupported storage_kind: {entry.storage_kind}")


__all__ = [
    "RenderError",
    "register_figure_renderer",
    "list_figure_kinds",
    "render_csv",
    "render_json",
    "render_geojson",
    "render_png",
    "render_native",
]
