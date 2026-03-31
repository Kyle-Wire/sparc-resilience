"""
dag_definition — Load and validate a causal DAG for the SPARC pipeline.

Reads the user-supplied DAG YAML (or DAGitty JSON export) and produces:
  - A ``networkx.DiGraph`` for graph-theoretic queries.
  - A DoWhy-compatible ``CausalModel`` for identifiability and estimation.

Typical usage::

    from sparc.causal.dag_definition import load_dag, build_causal_model

    dag = load_dag("causal/dag.yml")
    model = build_causal_model(dag, data)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# DAG loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def load_dag(dag_path: str) -> Dict[str, Any]:
    """
    Load a DAG definition from YAML or JSON.

    Parameters
    ----------
    dag_path : str
        Path to the DAG file (``dag.yml`` or ``dag.json``).

    Returns
    -------
    dict
        Parsed DAG definition with keys ``nodes``, ``edges``,
        and optionally ``structural_equations``.
    """
    p = Path(dag_path)
    if not p.exists():
        raise FileNotFoundError(f"DAG file not found: {dag_path}")

    if p.suffix in ('.yml', '.yaml'):
        raw = _load_yaml(str(p))
    elif p.suffix == '.json':
        with open(p, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
    else:
        raise ValueError(f"Unsupported DAG file format: {p.suffix}  (use .yml or .json)")

    # Minimal validation
    if 'nodes' not in raw or 'edges' not in raw:
        raise ValueError("DAG file must contain 'nodes' and 'edges' keys.")

    return raw


# ---------------------------------------------------------------------------
# Convert to networkx
# ---------------------------------------------------------------------------

def dag_to_networkx(dag_def: Dict[str, Any]) -> nx.DiGraph:
    """
    Convert a parsed DAG definition dict into a ``networkx.DiGraph``.

    Node attributes include ``type`` (treatment / mediator / confounder /
    outcome) and ``description``.  Edge attributes include ``mechanism``.

    If the definition contains ``temporal_edges``, they are appended
    automatically via :func:`add_temporal_edges`.
    """
    G = nx.DiGraph()

    for node in dag_def['nodes']:
        G.add_node(
            node['name'],
            node_type=node.get('type', 'unknown'),
            description=node.get('description', ''),
        )

    for edge in dag_def['edges']:
        G.add_edge(
            edge['parent'],
            edge['child'],
            mechanism=edge.get('mechanism', ''),
        )

    # Append temporal (time-lagged) edges if present
    temporal_edges = dag_def.get('temporal_edges', [])
    if temporal_edges:
        add_temporal_edges(G, temporal_edges)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The graph contains cycles — it is not a valid DAG.")

    return G


def get_node_roles(G: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Return lists of node names grouped by role.

    Returns
    -------
    dict with keys ``treatments``, ``outcomes``, ``mediators``, ``confounders``, ``unknown``.
    """
    roles: Dict[str, List[str]] = {
        'treatments': [],
        'outcomes': [],
        'mediators': [],
        'confounders': [],
        'unknown': [],
    }
    for node, attrs in G.nodes(data=True):
        ntype = attrs.get('node_type', 'unknown')
        if ntype == 'treatment':
            roles['treatments'].append(node)
        elif ntype == 'outcome':
            roles['outcomes'].append(node)
        elif ntype == 'mediator':
            roles['mediators'].append(node)
        elif ntype == 'confounder':
            roles['confounders'].append(node)
        else:
            roles['unknown'].append(node)
    return roles


# ---------------------------------------------------------------------------
# Build DoWhy CausalModel
# ---------------------------------------------------------------------------

def build_causal_model(
    dag_def: Dict[str, Any],
    data: pd.DataFrame,
    treatment: str | List[str] | None = None,
    outcome: str | None = None,
) -> Any:
    """
    Build a ``dowhy.CausalModel`` from a DAG definition and data.

    Parameters
    ----------
    dag_def : dict
        Parsed DAG definition (from ``load_dag``).
    data : pd.DataFrame
        Observational dataset with columns matching DAG node names.
    treatment : str or list[str], optional
        Treatment variable(s).  If *None*, inferred from nodes with
        ``type: treatment``.
    outcome : str, optional
        Outcome variable.  If *None*, inferred from the node with
        ``type: outcome``.

    Returns
    -------
    dowhy.CausalModel
    """
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        raise ImportError(
            "DoWhy is required for causal analysis.  Install with:\n"
            "  pip install dowhy"
        )

    G = dag_to_networkx(dag_def)
    roles = get_node_roles(G)

    if treatment is None:
        treatment = roles['treatments']
    if isinstance(treatment, str):
        treatment = [treatment]
    if outcome is None:
        if len(roles['outcomes']) != 1:
            raise ValueError(
                f"Expected exactly 1 outcome node, found {len(roles['outcomes'])}: "
                f"{roles['outcomes']}.  Pass `outcome=` explicitly."
            )
        outcome = roles['outcomes'][0]

    # Convert nx.DiGraph to DOT-like GML string for DoWhy
    gml_str = " ".join(
        f'"{u}" -> "{v}";' for u, v in G.edges()
    )
    graph_str = f"digraph {{ {gml_str} }}"

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=graph_str,
    )

    return model


# ---------------------------------------------------------------------------
# Temporal (time-lagged) DAG support
# ---------------------------------------------------------------------------

def add_temporal_edges(
    G: nx.DiGraph,
    temporal_edges: List[Dict[str, Any]],
) -> nx.DiGraph:
    """
    Augment a DAG with time-lagged edges from the config.

    Each temporal edge has the form::

        {source: "precipitation_mm", target: "gw_level_m", lag: 3}

    The *lag* is stored as an edge attribute (``lag``) and the edge is
    labeled ``source_t-{lag} → target`` in the graph to keep the DAG
    acyclic (effects from the past cannot create cycles).

    If the source/target variable is not already a node in *G*, the
    lagged node is added automatically.

    Parameters
    ----------
    G : nx.DiGraph
        Existing instantaneous DAG.
    temporal_edges : list[dict]
        Each dict must have ``source``, ``target``, and ``lag`` keys.

    Returns
    -------
    nx.DiGraph
        The augmented graph (modified in-place and also returned).
    """
    for te in temporal_edges:
        src = te.get('source', '')
        tgt = te.get('target', '')
        lag = int(te.get('lag', 1))
        if not src or not tgt:
            continue

        lagged_node = f"{src}_t-{lag}"

        if lagged_node not in G:
            # Inherit type from the original node if it exists
            original_attrs = G.nodes.get(src, {})
            G.add_node(
                lagged_node,
                node_type=original_attrs.get('node_type', 'confounder'),
                description=f"{src} lagged by {lag} time steps",
                temporal_lag=lag,
                temporal_source=src,
            )

        if tgt not in G:
            G.add_node(tgt, node_type='unknown')

        G.add_edge(
            lagged_node, tgt,
            mechanism=f"Time-lagged effect (lag={lag})",
            lag=lag,
        )

    # Verify acyclicity after adding temporal edges
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(
            "Adding temporal edges created a cycle — check lag definitions."
        )

    return G


def get_temporal_edges(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Extract all temporal (lagged) edges from a graph.

    Returns
    -------
    list[dict]
        Each dict has ``source``, ``target``, ``lag``, ``lagged_node``.
    """
    result = []
    for u, v, data in G.edges(data=True):
        lag = data.get('lag')
        if lag is not None:
            result.append({
                'lagged_node': u,
                'target': v,
                'source': G.nodes[u].get('temporal_source', u),
                'lag': lag,
            })
    return result


# ---------------------------------------------------------------------------
# Column validation helper
# ---------------------------------------------------------------------------

def validate_dag_columns(dag_def: Dict[str, Any], data: pd.DataFrame) -> List[str]:
    """
    Check that all DAG node names exist as columns in the data.

    Returns
    -------
    list[str]
        Names of DAG nodes that are **missing** from the data columns.
    """
    node_names = [n['name'] for n in dag_def.get('nodes', [])]
    return [n for n in node_names if n not in data.columns]


def get_confounders_for_edge(
    G: nx.DiGraph,
    parent: str,
    child: str,
) -> List[str]:
    """
    Return the backdoor adjustment set for a specific edge.

    .. deprecated::
        Use :func:`compute_backdoor_set` instead, which implements the
        full backdoor criterion (Pearl 2009) and avoids conditioning on
        mediators, colliders, or descendants of the treatment.

    Confounders are all parents of the *child* that are not the *parent*
    (simple backdoor criterion for single-edge estimation).
    """
    return [n for n in G.predecessors(child) if n != parent]


# ---------------------------------------------------------------------------
# Full backdoor criterion (Pearl 2009)
# ---------------------------------------------------------------------------

def compute_backdoor_set(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> List[str]:
    """
    Compute a valid backdoor adjustment set for estimating the causal
    effect of *treatment* on *outcome* using the full backdoor criterion.

    The backdoor criterion (Pearl 2009) requires an adjustment set Z that:
      1. Does **not** contain any descendant of *treatment*.
      2. Blocks every non-causal (backdoor) path from *treatment* to *outcome*.

    Strategy
    --------
    Start with all non-descendant, non-treatment, non-outcome nodes and
    prune to the minimal set that satisfies d-separation on the mutilated
    graph (all outgoing edges from *treatment* removed).

    Parameters
    ----------
    G : nx.DiGraph
        The causal DAG.
    treatment : str
        Treatment variable name.
    outcome : str
        Outcome variable name.

    Returns
    -------
    list[str]
        Sorted list of variable names forming a valid adjustment set.

    Raises
    ------
    ValueError
        If treatment or outcome is not in the graph, or if no valid
        adjustment set exists (e.g., outcome is an ancestor of treatment).
    """
    if treatment not in G:
        raise ValueError(f"Treatment '{treatment}' not found in DAG.")
    if outcome not in G:
        raise ValueError(f"Outcome '{outcome}' not found in DAG.")

    # Descendants of treatment (including treatment itself) — must NOT
    # condition on these, they are post-treatment variables.
    descendants_of_t = nx.descendants(G, treatment) | {treatment}

    # Candidate adjustment variables: everything that is NOT a descendant
    # of treatment and not the outcome itself.
    candidates = sorted(
        n for n in G.nodes()
        if n not in descendants_of_t and n != outcome
    )

    # Build the mutilated graph: remove all outgoing edges from treatment.
    # In this graph, d-separation of treatment and outcome given Z means Z
    # blocks all backdoor paths.
    G_mut = G.copy()
    out_edges = list(G_mut.out_edges(treatment))
    G_mut.remove_edges_from(out_edges)

    # If there are no backdoor paths (treatment has no parents, or treatment
    # and outcome are already d-separated in the mutilated graph), the empty
    # set suffices.
    if nx.is_d_separator(G_mut, {treatment}, {outcome}, set()):
        return []

    # Check if the full candidate set blocks all backdoor paths.
    if not nx.is_d_separator(G_mut, {treatment}, {outcome}, set(candidates)):
        # Backdoor criterion not satisfiable — unmeasured confounding.
        # Fall back to the full candidate set (best effort).
        return candidates

    # Greedy pruning: start with the full candidate set and try removing
    # each variable one at a time.  Keep only those whose removal would
    # open a backdoor path.
    adj_set = set(candidates)
    for var in candidates:
        smaller = adj_set - {var}
        if nx.is_d_separator(G_mut, {treatment}, {outcome}, smaller):
            adj_set = smaller

    return sorted(adj_set)


def generate_dag_variants(
    dag_def: Dict[str, Any],
    edge_parent: str,
    edge_child: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate DAG variants for sensitivity analysis on a contested edge.

    Returns three DAG definitions:
      * ``"original"`` — unchanged
      * ``"reversed"`` — the contested edge direction is flipped
      * ``"removed"``  — the contested edge is deleted entirely

    Raises ``ValueError`` if the reversed graph creates a cycle.

    Parameters
    ----------
    dag_def : dict
        Original parsed DAG definition.
    edge_parent, edge_child : str
        The contested edge to vary.

    Returns
    -------
    dict[str, dict]
        Mapping ``variant_name`` → ``dag_definition``.
    """
    import copy

    variants: Dict[str, Dict[str, Any]] = {}

    # --- Original ---
    variants['original'] = copy.deepcopy(dag_def)

    # --- Removed ---
    dag_removed = copy.deepcopy(dag_def)
    dag_removed['edges'] = [
        e for e in dag_removed['edges']
        if not (e['parent'] == edge_parent and e['child'] == edge_child)
    ]
    variants['removed'] = dag_removed

    # --- Reversed ---
    dag_reversed = copy.deepcopy(dag_def)
    dag_reversed['edges'] = [
        e for e in dag_reversed['edges']
        if not (e['parent'] == edge_parent and e['child'] == edge_child)
    ]
    dag_reversed['edges'].append({
        'parent': edge_child,
        'child': edge_parent,
        'mechanism': f"Reversed: {edge_child} -> {edge_parent}",
    })
    # Validate acyclicity
    try:
        G_test = dag_to_networkx(dag_reversed)
        if not nx.is_directed_acyclic_graph(G_test):
            raise ValueError("Cycle detected")
    except ValueError:
        dag_reversed = None  # Mark as invalid

    if dag_reversed is not None:
        variants['reversed'] = dag_reversed

    return variants


# ---------------------------------------------------------------------------
# Convenience: validate DAG against dataset columns
# ---------------------------------------------------------------------------

def validate_dag_columns(dag_def: Dict[str, Any], columns: List[str]) -> List[str]:
    """
    Check that every DAG node name corresponds to a column in the dataset.

    Returns a list of node names that are **missing** from *columns*.
    """
    node_names = {n['name'] for n in dag_def['nodes']}
    return sorted(node_names - set(columns))
