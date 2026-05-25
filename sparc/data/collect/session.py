"""
session.py — Typed session state for the data collection wizard.

Replaces the module-level ``_collect_session: dict`` in ``sparc.server.app``
with a dataclass that enforces a schema, makes the implicit ``scene_dates``
hand-off between service calls explicit, and persists a lightweight JSON
sidecar so the session survives server restarts.

Public interface
----------------
session = CollectSession()
session.set_boundary(boundary_result)          # initialises the fishnet
session.apply_fetch(fetch_result)              # merges new columns + anchors
session.manifest                               # VariableManifest
session.fishnet                                # GeoDataFrame | None
session.anchor_dates                           # list[date]
session.save(path)                             # write JSON sidecar
session = CollectSession.load(path)            # restore from sidecar
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from .boundary import BoundaryResult
from .manifest import VariableManifest
from .sources import FetchResult


# ---------------------------------------------------------------------------
# Fishnet builder (module-level so tests can patch it)
# ---------------------------------------------------------------------------

def _build_fishnet(br: BoundaryResult) -> object:
    """Construct the 30m analysis grid from a resolved boundary.

    Returns a GeoDataFrame in EPSG:3857 with ``cell_id``, ``cell_x``,
    ``cell_y``, and ``geometry`` columns.
    """
    from sparc.data.processing import create_fishnet, clip_to_boundary

    boundary_proj = br.gdf.to_crs("EPSG:3857")  # type: ignore[union-attr]
    bounds = tuple(boundary_proj.total_bounds)
    fishnet = create_fishnet(bounds, resolution=30.0, crs="EPSG:3857")
    fishnet = clip_to_boundary(fishnet, boundary_proj)
    fishnet["cell_id"] = range(len(fishnet))
    fishnet["cell_x"] = fishnet.geometry.centroid.x
    fishnet["cell_y"] = fishnet.geometry.centroid.y
    return fishnet


# ---------------------------------------------------------------------------
# CollectSession
# ---------------------------------------------------------------------------

@dataclass
class CollectSession:
    """Typed in-process state for a single data-collection wizard run.

    Attributes
    ----------
    boundary : BoundaryResult | None
        The resolved study-area boundary.  Set by :meth:`set_boundary`.
    boundary_source : str
        ``BoundaryResult.source`` string, persisted to the JSON sidecar.
    boundary_place_name : str | None
        Canonical place name, persisted to the JSON sidecar.
    fishnet : GeoDataFrame | None
        The 30m analysis grid.  Initialised by :meth:`set_boundary`;
        enriched in-place by :meth:`apply_fetch`.
    manifest : VariableManifest
        Tracks variable fetch status for every UHI variable.
    anchor_dates : list[date]
        CAPA measurement dates discovered during the CAPA fetch.
        Populated automatically when a :class:`FetchResult` with
        ``metadata["anchor_dates"]`` is passed to :meth:`apply_fetch`.
    """

    boundary: Optional[BoundaryResult] = field(default=None, repr=False)
    boundary_source: str = ""
    boundary_place_name: Optional[str] = None
    fishnet: Optional[object] = field(default=None, repr=False)
    manifest: VariableManifest = field(default_factory=VariableManifest.for_uhi)
    anchor_dates: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation interface
    # ------------------------------------------------------------------

    def set_boundary(self, br: BoundaryResult) -> None:
        """Store the boundary and (re-)build the 30m analysis fishnet.

        Calling this a second time replaces the existing fishnet — all
        previously fetched columns are discarded.
        """
        self.boundary = br
        self.boundary_source = br.source
        self.boundary_place_name = br.place_name
        self.fishnet = _build_fishnet(br)
        self.anchor_dates = []

    def apply_fetch(self, result: FetchResult) -> None:
        """Merge a completed :class:`FetchResult` into the session.

        If ``result.error`` is set the session is unchanged — the caller
        is responsible for surfacing the error to the user.

        Replaces ``self.fishnet`` with ``result.gdf`` (the enriched grid).
        Captures ``result.metadata["anchor_dates"]`` when present (CAPA
        enriches the session's temporal anchor list on first fetch).
        """
        if result.error:
            return
        if result.gdf is not None:
            self.fishnet = result.gdf
        if "anchor_dates" in result.metadata:
            self.anchor_dates = list(result.metadata["anchor_dates"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        """Write a lightweight JSON sidecar for restart survival.

        The fishnet is *not* persisted here — it is large and already stored
        in the live process.  The sidecar records boundary metadata and anchor
        dates so that on restart the caller knows which boundary was active and
        which CAPA dates were discovered.
        """
        data = {
            "boundary_source": self.boundary_source,
            "boundary": self.boundary_source,  # backward-compat alias
            "boundary_place_name": self.boundary_place_name,
            "anchor_dates": [d.isoformat() if isinstance(d, date) else str(d)
                             for d in self.anchor_dates],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: "Path | str") -> "CollectSession":
        """Restore a :class:`CollectSession` from a JSON sidecar.

        The fishnet is not restored (it would require re-fetching or
        loading the last-written GeoParquet); only boundary metadata and
        anchor dates are recovered.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        session = cls.__new__(cls)
        session.boundary = None
        session.boundary_source = data.get("boundary_source", data.get("boundary", ""))
        session.boundary_place_name = data.get("boundary_place_name")
        session.fishnet = None
        session.manifest = VariableManifest.for_uhi()
        session.anchor_dates = [
            date.fromisoformat(d) for d in data.get("anchor_dates", [])
        ]
        return session
