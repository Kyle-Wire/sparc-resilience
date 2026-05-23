"""ResidualContext — explicit carrier for JEPA spatial residualisation results.

Replaces the in-place ``self.graph = _new_graph`` mutation inside
``CausalValidator.fit()`` (lines 494–498).  Instead of modifying the graph
that was loaded from the DAG YAML (and thereby making ``self.graph`` implicitly
mean different things before and after the call), ``SpatialResidualizer`` or
``EdgeFitter`` returns a ``ResidualContext`` alongside the augmented DataFrame.
Callers that need remapped edges call ``ctx.remap_graph(graph)`` explicitly,
which returns a *new* graph and never mutates the original.

Design properties
-----------------
* **Immutable input** — ``remap_graph`` never modifies its argument.
* **Idempotent** — calling ``remap_graph`` on an already-remapped graph
  that contains ``{col}_resid`` nodes does not rename them again to
  ``{col}_resid_resid``.  Only nodes that appear as keys in ``resid_map``
  are remapped; if the key is absent in the graph the edge is left
  unchanged.
* **Transparent** — ``is_empty()`` lets callers skip the remap fast path
  when no residualisation was performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ResidualContext:
    """Maps original treatment column names to their residualised variants.

    Attributes
    ----------
    resid_map : dict[str, str]
        ``{original_col: residualised_col}``, e.g.
        ``{"Pct_Canopy": "Pct_Canopy_resid"}``.  Only includes columns
        that were successfully residualised — missing original columns mean
        residualisation was skipped for that treatment.
    """

    resid_map: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True when no residualisation was performed."""
        return len(self.resid_map) == 0

    # ------------------------------------------------------------------

    def remap_graph(self, graph) -> object:
        """Return a new DiGraph with residualised parent node names.

        Edges whose parent column appears in ``resid_map`` are rewritten to
        use the residualised column name.  All other edges are preserved
        unchanged.  The input *graph* is **never modified**.

        Parameters
        ----------
        graph : networkx.DiGraph
            The DAG loaded from YAML.  Not mutated.

        Returns
        -------
        networkx.DiGraph
            A new graph with remapped edges.  Node attributes from the
            original graph are preserved.
        """
        import networkx as nx

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(graph.nodes(data=True))

        for parent, child in graph.edges():
            new_parent = self.resid_map.get(parent, parent)
            new_graph.add_edge(new_parent, child)

        return new_graph
