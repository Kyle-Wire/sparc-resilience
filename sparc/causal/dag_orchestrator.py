"""DAG orchestrator — resolves and validates a causal DAG.

Hides the three DAG resolution paths that previously lived inside
``CausalValidator.load_dag()`` and ``run_causal_discovery()``:

1. **File path** — ``config['causal']['dag_file']`` exists → load + validate.
2. **Discovery** — no file, or file missing, and
   ``config['causal']['discovery']['enabled'] == True`` → run PC / LiNGAM / GES.
3. **Fallback** — discovery disabled or no data → raise a clear error.

Decoupling this from ``CausalValidator`` makes it testable in isolation and
lets other pipeline components (server, batch CLI) resolve a DAG without
loading the entire validator.

Usage
-----
::

    orch = DAGOrchestrator(config)
    result = orch.resolve(data=df)   # DAGResult namedtuple

    result.graph   # networkx.DiGraph
    result.roles   # {"treatments": [...], "mediators": [...], ...}
    result.source  # "file" | "discovery" | "fallback"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DAGResult:
    """Holds a resolved causal DAG and associated node-role metadata.

    Attributes
    ----------
    graph : networkx.DiGraph
        The resolved causal graph.
    dag_def : dict | None
        Raw dict loaded from the DAG YAML file (``None`` for
        discovery-derived graphs).
    roles : dict[str, list[str]]
        Node roles: ``treatments``, ``mediators``, ``confounders``,
        ``outcomes``.
    source : str
        One of ``"file"``, ``"discovery"``, or ``"fallback"``.
    discovery_report : dict | None
        Discovery run report when ``source == "discovery"``.
    """

    graph: Any  # networkx.DiGraph
    dag_def: Optional[dict]
    roles: Dict[str, List[str]]
    source: str
    discovery_report: Optional[dict] = field(default=None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DAGOrchestrator:
    """Resolves a causal DAG from config and optional data.

    Parameters
    ----------
    config : dict
        Project config as produced by ``sparc.config.config.load_config``.
    """

    def __init__(self, config: dict):
        self._config = config or {}
        self._causal_cfg = self._config.get("causal", {}) or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        data: Any = None,  # pd.DataFrame | None
    ) -> DAGResult:
        """Return a fully-resolved :class:`DAGResult`.

        Resolution order
        ----------------
        1. ``config['causal']['dag_file']`` exists → load from file.
        2. ``config['causal']['discovery']['enabled']`` is true and
           ``data`` is provided → run discovery.
        3. Raise ``ValueError`` with a clear message.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Training data — required for discovery-based resolution.

        Returns
        -------
        DAGResult

        Raises
        ------
        ValueError
            When no DAG source is available.
        FileNotFoundError
            When ``dag_file`` is specified but the file does not exist.
        """
        dag_file = self._causal_cfg.get("dag_file")

        if dag_file:
            return self._from_file(dag_file)

        disc_cfg = self._causal_cfg.get("discovery") or {}
        if disc_cfg.get("enabled", False) and data is not None:
            return self._from_discovery(data, disc_cfg)

        raise ValueError(
            "No causal DAG available.  Either specify "
            "'causal.dag_file' in project.yml, or enable discovery "
            "with 'causal.discovery.enabled: true' and provide data."
        )

    # ------------------------------------------------------------------
    # File-based resolution
    # ------------------------------------------------------------------

    def _from_file(self, dag_file: str) -> DAGResult:
        if not os.path.exists(dag_file):
            raise FileNotFoundError(
                f"DAG file not found: {dag_file}.  "
                "Check 'causal.dag_file' in project.yml."
            )
        from sparc.causal.dag_definition import (
            load_dag,
            dag_to_networkx,
            get_node_roles,
        )
        dag_def = load_dag(dag_file)
        graph = dag_to_networkx(dag_def)
        roles = get_node_roles(graph)
        log.info(
            "DAGOrchestrator: loaded from file %s (%d nodes, %d edges)",
            dag_file,
            len(graph.nodes),
            len(graph.edges),
        )
        return DAGResult(
            graph=graph,
            dag_def=dag_def,
            roles=roles,
            source="file",
        )

    # ------------------------------------------------------------------
    # Discovery-based resolution
    # ------------------------------------------------------------------

    def _from_discovery(self, data: Any, disc_cfg: dict) -> DAGResult:
        try:
            from sparc.causal.causal_discovery import CausalDiscoveryValidator
        except ImportError as exc:
            raise ValueError(
                "causal-learn is not installed; cannot run discovery. "
                "Install it with: pip install causal-learn"
            ) from exc

        methods = disc_cfg.get("methods", ["pc_stable"])
        alpha = disc_cfg.get("alpha", 0.05)
        consensus_threshold = disc_cfg.get("consensus_threshold", 1)

        # Determine which columns to use
        node_cfg = self._causal_cfg.get("nodes") or {}
        if isinstance(node_cfg, list):
            node_names = [n if isinstance(n, str) else n.get("name", "") for n in node_cfg]
        else:
            node_names = [c for c in data.columns if data[c].dtype.kind in "fiu"]

        cols = [c for c in node_names if c in data.columns]
        if len(cols) < 3:
            raise ValueError(
                f"Too few columns ({len(cols)}) for causal discovery.  "
                "Specify node names in 'causal.nodes' or use dag_file."
            )

        validator = CausalDiscoveryValidator(data[cols], node_names=cols)

        if "pc_stable" in methods:
            try:
                validator.run_pc_stable(alpha=alpha)
            except Exception as exc:
                log.warning("DAGOrchestrator: pc_stable failed: %s", exc)
        if "lingam" in methods:
            try:
                validator.run_lingam()
            except Exception as exc:
                log.warning("DAGOrchestrator: lingam failed: %s", exc)
        if "ges" in methods:
            try:
                validator.run_ges()
            except Exception as exc:
                log.warning("DAGOrchestrator: ges failed: %s", exc)

        if not validator.learned_graphs:
            raise ValueError(
                "All causal discovery methods failed.  "
                "Check that causal-learn is correctly installed and "
                "the input data has sufficient variation."
            )

        try:
            validator.consensus_dag(min_votes=consensus_threshold)
        except Exception as exc:
            log.warning("DAGOrchestrator: consensus DAG failed: %s", exc)

        import networkx as nx
        graph = nx.DiGraph()

        key = "consensus" if "consensus" in validator.learned_graphs else next(iter(validator.learned_graphs))
        adj = validator.learned_graphs[key]
        if hasattr(adj, "tolist"):
            n = len(cols)
            for i in range(n):
                for j in range(n):
                    if adj[i, j] != 0:
                        graph.add_edge(cols[i], cols[j])
        else:
            graph = adj  # already a DiGraph

        from sparc.causal.dag_definition import get_node_roles
        roles = get_node_roles(graph)
        report = validator.generate_report()

        log.info(
            "DAGOrchestrator: resolved via discovery (%d nodes, %d edges)",
            len(graph.nodes),
            len(graph.edges),
        )
        return DAGResult(
            graph=graph,
            dag_def=None,
            roles=roles,
            source="discovery",
            discovery_report=report,
        )
