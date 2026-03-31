"""
causal_discovery — Data-driven DAG validation via structure-learning algorithms.

Runs multiple causal discovery methods (PC-Stable, DirectLiNGAM, GES/FGES)
on the observational data and compares the data-learned graph against the
expert-specified DAG.  This serves as **validation** — the expert DAG
remains primary, but discrepancies are flagged for review.

References
----------
- Liu & Niyogi (2020): Compared PC, PC-Stable, CPC, LiNGAM, FGES, IMaGES
  for UHI causal discovery.
- Assaf et al. (2023): Bayesian networks for UHI factor identification.

Typical usage::

    from sparc.causal.causal_discovery import CausalDiscoveryValidator

    cdv = CausalDiscoveryValidator(data, node_names)
    results = cdv.run_all(methods=['pc_stable', 'lingam', 'ges'])
    report = cdv.compare_with_expert(expert_graph)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx


class CausalDiscoveryValidator:
    """
    Run multiple causal discovery algorithms and compare with an expert DAG.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data with columns matching DAG node names.
    node_names : list[str]
        Column names to include in discovery (should match DAG nodes).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        node_names: List[str],
        random_state: int = 42,
    ):
        available = [c for c in node_names if c in data.columns]
        self.data = data[available].copy()
        self.node_names = available
        self.random_state = random_state

        # Results from each method: {method_name: nx.DiGraph}
        self.learned_graphs: Dict[str, nx.DiGraph] = {}
        self.comparison_report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # PC-Stable (constraint-based)
    # ------------------------------------------------------------------

    def run_pc_stable(self, alpha: float = 0.05) -> nx.DiGraph:
        """
        Run the PC-Stable algorithm (order-independent constraint-based).

        Parameters
        ----------
        alpha : float
            Significance level for conditional independence tests.

        Returns
        -------
        nx.DiGraph
            Learned (partially) directed graph.
        """
        from causallearn.search.ConstraintBased.PC import pc

        X = self.data.values
        cg = pc(
            X,
            alpha=alpha,
            stable=True,
            uc_rule=0,       # 0 = standard orientation
            uc_priority=-1,  # default priority
            verbose=False,
        )

        G = self._clearn_to_networkx(cg.G, self.node_names)
        self.learned_graphs['pc_stable'] = G
        return G

    # ------------------------------------------------------------------
    # DirectLiNGAM (functional causal model, exploits non-Gaussianity)
    # ------------------------------------------------------------------

    def run_lingam(self) -> nx.DiGraph:
        """
        Run DirectLiNGAM to learn a fully-oriented DAG.

        Exploits non-Gaussianity of error terms for full edge orientation.
        Works best when at least some variables have non-Gaussian distributions.

        Returns
        -------
        nx.DiGraph
            Fully oriented DAG with edge weights (causal coefficients).
        """
        from causallearn.search.FCMBased.lingam import DirectLiNGAM

        X = self.data.values
        model = DirectLiNGAM()
        model.fit(X)

        G = nx.DiGraph()
        for name in self.node_names:
            G.add_node(name)

        adjacency = getattr(model, 'adjacency_matrix_', None)
        if adjacency is not None:
            for i in range(len(self.node_names)):
                for j in range(len(self.node_names)):
                    if abs(adjacency[i, j]) > 1e-6:
                        # adjacency_matrix_[i, j] means j → i (effect on row i from col j)
                        G.add_edge(
                            self.node_names[j],
                            self.node_names[i],
                            weight=float(adjacency[i, j]),
                        )

        # Remove cycles if any (LiNGAM should produce DAG but guard against it)
        if not nx.is_directed_acyclic_graph(G):
            G = self._break_cycles(G)

        self.learned_graphs['lingam'] = G
        return G

    # ------------------------------------------------------------------
    # GES — Greedy Equivalence Search (score-based)
    # ------------------------------------------------------------------

    def run_ges(self, score_func: str = 'local_score_BIC') -> nx.DiGraph:
        """
        Run Greedy Equivalence Search (GES) with BIC scoring.

        Parameters
        ----------
        score_func : str
            Score function name for causal-learn GES.

        Returns
        -------
        nx.DiGraph
            Learned (partially) directed graph.
        """
        from causallearn.search.ScoreBased.GES import ges

        X = self.data.values
        record = ges(
            X,
            score_func=score_func,
            maxP=None,
        )

        G = self._clearn_to_networkx(record['G'], self.node_names)
        self.learned_graphs['ges'] = G
        return G

    # ------------------------------------------------------------------
    # PCMCI — time-lagged causal discovery (Runge et al. 2019)
    # ------------------------------------------------------------------

    def run_pcmci(
        self,
        alpha: float = 0.05,
        tau_max: int = 3,
        time_column: str | None = None,
        id_column: str | None = None,
    ) -> nx.DiGraph:
        """
        Run PCMCI for time-lagged causal discovery on panel / time-series data.

        Requires the ``tigramite`` package.  Data must be a panel
        (id × time) or a single time-series sorted chronologically.

        Parameters
        ----------
        alpha : float
            Significance level for conditional independence tests.
        tau_max : int
            Maximum time lag to test.
        time_column : str, optional
            Column with timestamps (will be dropped from node variables).
        id_column : str, optional
            Panel identifier column (will be dropped from node variables).

        Returns
        -------
        nx.DiGraph
            Learned time-lagged graph with ``lag`` edge attributes.
        """
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError:
            raise ImportError(
                "tigramite is required for PCMCI.  Install with:\n"
                "  pip install tigramite"
            )

        df = self.data.copy()

        # Drop non-variable columns
        drop_cols = []
        if time_column and time_column in df.columns:
            drop_cols.append(time_column)
        if id_column and id_column in df.columns:
            drop_cols.append(id_column)
        df = df.drop(columns=drop_cols, errors='ignore')

        var_names = list(df.columns)
        dataframe = pp.DataFrame(
            df.values,
            var_names=var_names,
        )

        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)

        # Build networkx graph from PCMCI results
        G = nx.DiGraph()
        for name in var_names:
            G.add_node(name)

        p_matrix = results['p_matrix']
        val_matrix = results['val_matrix']

        for j, target in enumerate(var_names):
            for i, source in enumerate(var_names):
                for tau in range(0, tau_max + 1):
                    if p_matrix[i, j, tau] < alpha:
                        if tau == 0 and i == j:
                            continue  # skip self-loops at lag 0
                        if tau == 0:
                            G.add_edge(
                                source, target,
                                weight=float(val_matrix[i, j, tau]),
                                lag=0,
                            )
                        else:
                            lagged_name = f"{source}_t-{tau}"
                            if lagged_name not in G:
                                G.add_node(
                                    lagged_name,
                                    temporal_lag=tau,
                                    temporal_source=source,
                                )
                            G.add_edge(
                                lagged_name, target,
                                weight=float(val_matrix[i, j, tau]),
                                lag=tau,
                            )

        # Break cycles if any
        if not nx.is_directed_acyclic_graph(G):
            G = self._break_cycles(G)

        self.learned_graphs['pcmci'] = G
        return G

    # ------------------------------------------------------------------
    # Run all methods
    # ------------------------------------------------------------------

    def run_all(
        self,
        methods: List[str] | None = None,
        alpha: float = 0.05,
    ) -> Dict[str, nx.DiGraph]:
        """
        Run all specified discovery methods.

        Parameters
        ----------
        methods : list[str], optional
            Subset of ``['pc_stable', 'lingam', 'ges']``.
            Defaults to all three.
        alpha : float
            Significance level for PC-Stable.

        Returns
        -------
        dict
            ``{method_name: nx.DiGraph}``
        """
        if methods is None:
            methods = ['pc_stable', 'lingam', 'ges']

        for method in methods:
            try:
                if method == 'pc_stable':
                    self.run_pc_stable(alpha=alpha)
                elif method == 'lingam':
                    self.run_lingam()
                elif method == 'ges':
                    self.run_ges()
                elif method == 'pcmci':
                    self.run_pcmci(alpha=alpha)
                else:
                    warnings.warn(f"Unknown discovery method: {method}")
            except Exception as e:
                warnings.warn(f"Discovery method '{method}' failed: {e}")

        return self.learned_graphs

    # ------------------------------------------------------------------
    # Consensus DAG (majority voting)
    # ------------------------------------------------------------------

    def consensus_dag(self, min_votes: int = 2) -> nx.DiGraph:
        """
        Build a consensus DAG: an edge is included only if at least
        *min_votes* methods agree on its existence.

        Undirected edges are counted as present regardless of direction.

        Parameters
        ----------
        min_votes : int
            Minimum number of methods that must agree on an edge.

        Returns
        -------
        nx.DiGraph
            Consensus graph.
        """
        if not self.learned_graphs:
            raise RuntimeError("No learned graphs available. Call run_all() first.")

        # Count how many methods include each undirected edge
        edge_votes: Dict[Tuple[str, str], int] = {}
        edge_direction_votes: Dict[Tuple[str, str], int] = {}

        for _method, G in self.learned_graphs.items():
            seen_undirected: Set[Tuple[str, str]] = set()
            for u, v in G.edges():
                key = (min(u, v), max(u, v))
                if key not in seen_undirected:
                    seen_undirected.add(key)
                    edge_votes[key] = edge_votes.get(key, 0) + 1
                # Track directed edge votes
                edge_direction_votes[(u, v)] = edge_direction_votes.get((u, v), 0) + 1

        # Build consensus graph
        consensus = nx.DiGraph()
        for name in self.node_names:
            consensus.add_node(name)

        for (u, v), votes in edge_votes.items():
            if votes >= min_votes:
                # Choose the majority direction
                fwd = edge_direction_votes.get((u, v), 0)
                bwd = edge_direction_votes.get((v, u), 0)
                if fwd >= bwd:
                    consensus.add_edge(u, v, votes=votes)
                else:
                    consensus.add_edge(v, u, votes=votes)

        # Break any resulting cycles
        if not nx.is_directed_acyclic_graph(consensus):
            consensus = self._break_cycles(consensus)

        self.learned_graphs['consensus'] = consensus
        return consensus

    # ------------------------------------------------------------------
    # Compare learned DAG with expert DAG
    # ------------------------------------------------------------------

    def compare_with_expert(
        self,
        expert_graph: nx.DiGraph,
        method: str | None = None,
    ) -> Dict[str, Any]:
        """
        Compare a learned graph (or all learned graphs) with the expert DAG.

        Computes:
        - Structural Hamming Distance (SHD)
        - Edge precision / recall / F1
        - Lists of confirmed, missing, and extra edges

        Parameters
        ----------
        expert_graph : nx.DiGraph
            The user-specified expert DAG.
        method : str, optional
            Compare only this method's graph. If None, compare all.

        Returns
        -------
        dict
            Comparison metrics per method.
        """
        graphs_to_compare = {}
        if method is not None:
            if method in self.learned_graphs:
                graphs_to_compare[method] = self.learned_graphs[method]
        else:
            graphs_to_compare = self.learned_graphs

        expert_edges = set(expert_graph.edges())
        expert_undirected = {(min(u, v), max(u, v)) for u, v in expert_edges}

        report = {}
        for name, learned in graphs_to_compare.items():
            learned_edges = set(learned.edges())
            learned_undirected = {(min(u, v), max(u, v)) for u, v in learned_edges}

            # SHD: count of edge additions + deletions + reversals
            # (using undirected skeleton for additions/deletions, directed for reversals)
            additions = learned_undirected - expert_undirected
            deletions = expert_undirected - learned_undirected
            shared = expert_undirected & learned_undirected

            reversals = 0
            for u, v in shared:
                expert_has_fwd = (u, v) in expert_edges
                expert_has_bwd = (v, u) in expert_edges
                learned_has_fwd = (u, v) in learned_edges
                learned_has_bwd = (v, u) in learned_edges
                if expert_has_fwd and not learned_has_fwd and learned_has_bwd:
                    reversals += 1
                elif expert_has_bwd and not learned_has_bwd and learned_has_fwd:
                    reversals += 1

            shd = len(additions) + len(deletions) + reversals

            # Precision/Recall on skeleton (undirected)
            tp = len(shared)
            fp = len(additions)
            fn = len(deletions)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)

            # Edge-level details
            confirmed_edges = [
                (u, v) for u, v in expert_edges
                if (min(u, v), max(u, v)) in learned_undirected
            ]
            missing_edges = [
                (u, v) for u, v in expert_edges
                if (min(u, v), max(u, v)) not in learned_undirected
            ]
            extra_edges = [
                (u, v) for u, v in learned_edges
                if (min(u, v), max(u, v)) not in expert_undirected
            ]

            report[name] = {
                'shd': shd,
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1': round(f1, 3),
                'n_expert_edges': len(expert_edges),
                'n_learned_edges': len(learned_edges),
                'confirmed_edges': confirmed_edges,
                'missing_edges': missing_edges,
                'extra_edges': extra_edges,
                'n_reversals': reversals,
            }

        self.comparison_report = report
        return report

    # ------------------------------------------------------------------
    # D-Separation validation
    # ------------------------------------------------------------------

    def validate_dseparation(
        self,
        expert_graph: nx.DiGraph,
        data: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Test conditional independence (d-separation) implications of the
        expert DAG against the observed data.

        For each pair of non-adjacent nodes (X, Y), finds the d-separating
        set Z implied by the DAG and tests X ⊥ Y | Z using partial
        correlation (Fisher's z-test).

        Parameters
        ----------
        expert_graph : nx.DiGraph
            The expert-specified DAG.
        data : pd.DataFrame, optional
            Data to test against. Uses ``self.data`` if None.
        alpha : float
            Significance level for independence tests.

        Returns
        -------
        dict
            ``{'total': int, 'passed': int, 'failed': int, 'pass_rate': float,
               'violations': list[dict]}``
        """

        if data is None:
            data = self.data

        nodes = [n for n in expert_graph.nodes() if n in data.columns]
        n_obs = len(data)
        violations = []
        total = 0
        passed = 0

        # For each pair of non-adjacent nodes, test d-separation
        for i, x in enumerate(nodes):
            for y in nodes[i + 1:]:
                if expert_graph.has_edge(x, y) or expert_graph.has_edge(y, x):
                    continue  # Adjacent — skip

                # Find a minimal d-separating set
                try:
                    undirected = expert_graph.to_undirected()
                    # Ancestors of X and Y form a candidate conditioning set
                    ancestors_x = nx.ancestors(expert_graph, x)
                    ancestors_y = nx.ancestors(expert_graph, y)
                    candidates = (ancestors_x | ancestors_y) - {x, y}
                    candidates = [c for c in candidates if c in data.columns]

                    # Check d-separation with the candidate set
                    is_dsep = nx.d_separated(
                        expert_graph, {x}, {y}, set(candidates)
                    )
                    if not is_dsep:
                        continue  # Not d-separated — can't test this pair

                except Exception:
                    continue

                # Partial correlation test (Fisher's z-test)
                total += 1
                try:
                    p_value = self._partial_correlation_test(
                        data, x, y, candidates, n_obs,
                    )
                    if p_value >= alpha:
                        passed += 1  # Independence holds
                    else:
                        violations.append({
                            'x': x, 'y': y,
                            'conditioning_set': candidates,
                            'p_value': round(p_value, 6),
                        })
                except Exception:
                    pass  # Skip on numerical issues

        pass_rate = passed / total if total > 0 else 1.0
        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': round(pass_rate, 3),
            'violations': violations,
        }

    @staticmethod
    def _partial_correlation_test(
        data: pd.DataFrame,
        x: str,
        y: str,
        z_vars: List[str],
        n: int,
    ) -> float:
        """
        Test X ⊥ Y | Z using partial correlation with Fisher's z-transform.

        Returns the p-value for the null hypothesis of zero partial correlation.
        """
        from scipy import stats

        x_vals = np.asarray(data[x].values, dtype=float)
        y_vals = np.asarray(data[y].values, dtype=float)

        if not z_vars:
            # Simple correlation
            result = stats.pearsonr(x_vals, y_vals)
            return float(result.pvalue if hasattr(result, 'pvalue') else result[1])

        # Partial correlation via regression residuals
        from sklearn.linear_model import LinearRegression

        Z = data[z_vars].values
        lr_x = LinearRegression().fit(Z, x_vals)
        lr_y = LinearRegression().fit(Z, y_vals)
        res_x = x_vals - lr_x.predict(Z)
        res_y = y_vals - lr_y.predict(Z)

        result = stats.pearsonr(res_x, res_y)
        r_val = float(result.statistic if hasattr(result, 'statistic') else result[0])

        # Fisher's z-transform for significance
        k = len(z_vars)
        if n - k - 3 <= 0:
            return 1.0  # Not enough degrees of freedom

        z_stat = 0.5 * np.log((1 + abs(r_val)) / max(1 - abs(r_val), 1e-10))
        se = 1.0 / np.sqrt(n - k - 3)
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat) / se))
        return float(p_value)

    # ------------------------------------------------------------------
    # Generate full report
    # ------------------------------------------------------------------

    def generate_report(self, expert_graph: nx.DiGraph | None = None) -> dict:
        """
        Produce a structured report of discovery results as a dict.

        If *expert_graph* is provided and comparisons have been run,
        includes the comparison metrics.
        """
        report: Dict[str, Any] = {
            'variables': self.node_names,
            'n_observations': len(self.data),
            'methods_run': list(self.learned_graphs.keys()),
            'learned_graphs': {},
        }

        for name, G in self.learned_graphs.items():
            report['learned_graphs'][name] = {
                'n_edges': G.number_of_edges(),
                'edges': [(u, v) for u, v in G.edges()],
            }

        if self.comparison_report:
            report['expert_comparison_by_method'] = self.comparison_report
            # Consensus: pick the method with the lowest SHD
            best = min(
                self.comparison_report.values(),
                key=lambda m: m.get('shd', float('inf')),
            )
            report['expert_comparison'] = {
                'shd': best.get('shd'),
                'precision': best.get('precision'),
                'recall': best.get('recall'),
                'f1': best.get('f1'),
                'n_reversals': best.get('n_reversals'),
            }

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clearn_to_networkx(
        clearn_graph: Any,
        node_names: List[str],
    ) -> nx.DiGraph:
        """
        Convert a causal-learn GeneralGraph to a networkx DiGraph.

        Handles directed (-->), undirected (---), and bidirected (<->)
        edge marks from causal-learn's internal representation.
        """
        G = nx.DiGraph()
        for name in node_names:
            G.add_node(name)

        graph_np = clearn_graph.graph  # numpy adjacency matrix
        n = len(node_names)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # causal-learn edge encoding:
                # graph[i,j] = -1, graph[j,i] = 1  →  i --> j
                # graph[i,j] = -1, graph[j,i] = -1 →  i --- j (undirected)
                # graph[i,j] = 1,  graph[j,i] = 1  →  i <-> j (bidirected)
                if graph_np[i, j] == -1 and graph_np[j, i] == 1:
                    G.add_edge(node_names[i], node_names[j])
                elif graph_np[i, j] == -1 and graph_np[j, i] == -1:
                    # Undirected: add both directions (will be handled in comparison)
                    if not G.has_edge(node_names[j], node_names[i]):
                        G.add_edge(node_names[i], node_names[j])

        return G

    @staticmethod
    def _break_cycles(G: nx.DiGraph) -> nx.DiGraph:
        """Remove edges involved in cycles (weakest-weight first)."""
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                # Remove edge with smallest absolute weight
                min_edge = min(
                    cycle,
                    key=lambda e: abs(G.edges[e[0], e[1]].get('weight', 1.0)),
                )
                G.remove_edge(min_edge[0], min_edge[1])
            except nx.NetworkXNoCycle:
                break
        return G
