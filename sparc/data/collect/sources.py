"""
sources.py — DataSource protocol, FetchResult, and SourceRegistry.

Defines the seam through which all data collection source adapters communicate
with the assembler and the portal's fetch dispatcher.

No heavy dependencies at import time — geopandas, pandas, and network clients
are imported lazily inside adapter ``fetch()`` implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# FetchResult — uniform return type for every DataSource
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Uniform return envelope from any DataSource.fetch() call.

    Attributes
    ----------
    gdf : object
        Enriched GeoDataFrame (typed as object to avoid hard geopandas import
        at protocol level).  May be ``None`` if the source produced no geometry.
    columns : dict[str, str]
        Mapping of *column_name* → *source_attribution_string* for every column
        the adapter added to ``gdf``.  Used by the manifest updater.
    coverage : dict[str, float]
        Mapping of *column_name* → *fraction 0.0–1.0* of cells that have a
        non-null value.  Computed by the adapter (it knows its own data).
    metadata : dict[str, object]
        Optional pass-through data (e.g. ``{"anchor_dates": [...]}``) that
        downstream callers or the session store may need.
    """

    gdf: object
    columns: Dict[str, str] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataSource protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DataSource(Protocol):
    """Protocol every data-source adapter must satisfy.

    Implementations are thin wrappers around the existing ``fetch_*`` functions
    in ``sparc.data.collect``.  Adapters normalise diverse function signatures
    into the single ``fetch(fishnet, bbox, params) → FetchResult`` shape so
    that the assembler and portal dispatch are source-agnostic.
    """

    @property
    def group_name(self) -> str:
        """Short identifier used as the dispatch key (e.g. ``"era5"``)."""
        ...

    def fetch(
        self,
        fishnet: object,
        bbox: tuple,
        params: dict,
    ) -> FetchResult:
        """Execute the source fetch and return a :class:`FetchResult`.

        Parameters
        ----------
        fishnet : GeoDataFrame (typed as object)
            The 30 m analysis grid to enrich.
        bbox : tuple[float, float, float, float]
            Study-area bounding box ``(minx, miny, maxx, maxy)`` in EPSG:4326.
        params : dict
            Fetch parameters (dates, cloud cover, enabled indices, etc.).

        Returns
        -------
        FetchResult
            Enriched fishnet + column attribution + coverage + optional metadata.
        """
        ...


# ---------------------------------------------------------------------------
# SourceRegistry
# ---------------------------------------------------------------------------

class SourceRegistry:
    """Maps group names to :class:`DataSource` adapter instances.

    The assembler and portal dispatch loop iterate the registry rather than
    naming individual sources — adding a new domain source means registering
    a new adapter, not editing the assembler or server.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, DataSource] = {}

    def register(self, source: DataSource) -> None:
        """Register *source* under its ``group_name``.

        If a source with the same ``group_name`` is already registered it is
        replaced (last-write wins, useful for testing overrides).
        """
        self._sources[source.group_name] = source

    def get(self, group_name: str) -> DataSource:
        """Return the adapter registered under *group_name*.

        Raises
        ------
        KeyError
            If no adapter has been registered for *group_name*.
        """
        if group_name not in self._sources:
            raise KeyError(f"No adapter registered for source group '{group_name}'")
        return self._sources[group_name]

    def __iter__(self) -> Iterator[DataSource]:
        return iter(self._sources.values())

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, group_name: str) -> bool:
        return group_name in self._sources


# ---------------------------------------------------------------------------
# Module-level singleton used by the assembler and portal
# ---------------------------------------------------------------------------

#: The global registry populated by adapter modules at import time.
#: Tests can register mock adapters on a fresh :class:`SourceRegistry` instance
#: and pass it explicitly — no need to mutate this singleton.
SOURCE_REGISTRY: SourceRegistry = SourceRegistry()
