"""
dispatch.py — Domain-agnostic group fetch dispatcher.

``sync_group_fetch`` is the seam that replaces the ``_sync_group_fetch``
if/elif chain in ``sparc.server.app``.  It dispatches to the registered
:class:`~sparc.data.collect.sources.DataSource` adapter for the given group
name and uses :class:`~sparc.data.collect.sources.FetchResult` to update
the :class:`~sparc.data.collect.manifest.VariableManifest` in a uniform way.

This module contains no FastAPI or HTTP-layer code, making it testable
without a running server.
"""

from __future__ import annotations

from .sources import FetchResult, SourceRegistry, SOURCE_REGISTRY


def sync_group_fetch(
    group: str,
    fishnet: object,
    boundary: object,
    manifest: object,
    cfg: dict,
    *,
    registry: SourceRegistry | None = None,
) -> FetchResult:
    """Dispatch a synchronous group fetch through the :class:`SourceRegistry`.

    Parameters
    ----------
    group : str
        Source group identifier (e.g. ``"era5"``, ``"nlcd"``).
    fishnet : GeoDataFrame (typed as object)
        The 30 m analysis grid to enrich.  Passed directly to the adapter.
    boundary : BoundaryResult (typed as object)
        Resolved study boundary.  ``boundary.bbox`` is extracted and passed
        to the adapter.
    manifest : VariableManifest (typed as object)
        The active collection manifest to update.
    cfg : dict
        Fetch parameters forwarded to the adapter.
    registry : SourceRegistry | None
        Registry to use.  Defaults to the module-level
        :data:`~sparc.data.collect.sources.SOURCE_REGISTRY`.

    Returns
    -------
    FetchResult
        The result returned by the adapter (contains enriched gdf, column
        attribution, coverage percentages, and optional metadata).

    Raises
    ------
    ValueError
        If no adapter is registered for *group*.
    """
    if registry is None:
        registry = SOURCE_REGISTRY

    try:
        source = registry.get(group)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    bbox = boundary.bbox  # type: ignore[union-attr]
    result = source.fetch(fishnet, bbox, cfg)

    # Update manifest from what the adapter declared it produced
    for col, src_name in result.columns.items():
        _update_manifest(manifest, col, src_name, result.coverage.get(col, 0.0))

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _update_manifest(manifest: object, col: str, src_name: str, coverage: float) -> None:
    """Apply a coverage result to a manifest entry if the column is registered."""
    entries = getattr(manifest, "entries", {})
    if col not in entries:
        return
    entry = entries[col]
    if coverage == 0.0 and not entry.required:
        manifest.skip(col, f"No {src_name} coverage")  # type: ignore[union-attr]
    else:
        manifest.update(col, coverage_pct=coverage, source_name=src_name)  # type: ignore[union-attr]
