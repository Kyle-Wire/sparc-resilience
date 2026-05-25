"""figure_registry.py — Deep module for server-side figure renderer dispatch.

``FigureRendererRegistry`` replaces the bare ``FIGURES_REGISTRY`` dict in
``figures.py``.  Renderers register themselves via :meth:`register` or the
:meth:`renderer` decorator.  Tests inject mock renderers without matplotlib
or an :class:`~sparc.registry.store.ArtifactStore`.

Public interface
----------------
::

    reg = FigureRendererRegistry()

    @reg.renderer("0", "data")
    def render_histogram(store, stage, artifact_id, opts):
        ...
        return png_bytes

    png = reg.render("0", "data", store=my_store, opts={"dpi": 144})
    keys = reg.list_registered()       # [("0", "data"), ...]

    # Module-level singleton wired by figures.py:
    from sparc.report.figure_registry import FIGURE_REGISTRY
"""

from __future__ import annotations

from typing import Any, Callable


# Renderer callable type: (store, stage, artifact_id, opts) → bytes
Renderer = Callable[[Any, str, str, dict[str, Any]], bytes]


class FigureRendererRegistry:
    """Deep module — register renderers, dispatch by (stage, artifact_id).

    A small interface hides the dispatch table, the error type, and the
    calling convention.  Tests can create a fresh instance and inject any
    callable; production code uses the module-level :data:`FIGURE_REGISTRY`.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], Renderer] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, stage: str, artifact_id: str, fn: Renderer) -> None:
        """Register *fn* as the renderer for *(stage, artifact_id)*."""
        self._registry[(str(stage), artifact_id)] = fn

    def renderer(self, stage: str, artifact_id: str) -> Callable[[Renderer], Renderer]:
        """Decorator that registers the decorated function and returns it unchanged.

        Usage::

            @FIGURE_REGISTRY.renderer("0", "data")
            def render_data_histogram(store, stage, artifact_id, opts):
                ...
                return png_bytes
        """
        def _decorator(fn: Renderer) -> Renderer:
            self.register(stage, artifact_id, fn)
            return fn

        return _decorator

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def render(
        self,
        stage: str,
        artifact_id: str,
        store: Any,
        opts: dict[str, Any],
    ) -> bytes:
        """Call the renderer registered for *(stage, artifact_id)*.

        Parameters
        ----------
        stage:
            Stage key, e.g. ``"0"``, ``"3"``.
        artifact_id:
            Artifact identifier, e.g. ``"data"``, ``"predictions"``.
        store:
            An :class:`~sparc.registry.store.ArtifactStore` instance (or any
            compatible object) that renderers use to fetch artifact data.
        opts:
            Keyword options forwarded to the renderer, e.g. ``{"dpi": 144}``.

        Returns
        -------
        bytes
            PNG image bytes produced by the renderer.

        Raises
        ------
        FigureRenderError
            If no renderer is registered for *(stage, artifact_id)*.
        """
        key = (str(stage), artifact_id)
        fn = self._registry.get(key)
        if fn is None:
            # Import here to avoid circular imports — figures.py imports us.
            from sparc.report.figures import FigureRenderError

            raise FigureRenderError(
                f"no renderer registered for stage={stage!r}, "
                f"artifact_id={artifact_id!r}"
            )
        return fn(store, str(stage), artifact_id, opts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_registered(self) -> list[tuple[str, str]]:
        """Return all registered *(stage, artifact_id)* keys, in insertion order."""
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# Module-level singleton — wired by sparc/report/figures.py at import time
# ---------------------------------------------------------------------------

#: The shared global registry.  ``figures.py`` populates this at module load;
#: callers that want to add renderers at runtime call
#: ``FIGURE_REGISTRY.register(...)`` or use ``@FIGURE_REGISTRY.renderer(...)``.
FIGURE_REGISTRY: FigureRendererRegistry = FigureRendererRegistry()
