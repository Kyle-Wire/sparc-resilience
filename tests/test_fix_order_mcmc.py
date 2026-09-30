"""Regression test for G7: ``order_mcmc._score_ordering`` called
``prior.log_prior(dag, dag.node_names)`` although
``PhysicsInformedGraphPrior.log_prior`` takes only the DAG → ``TypeError``
on first use of :func:`run_order_mcmc`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sparc.causal.mc3 import PhysicsInformedGraphPrior
from sparc.causal.order_mcmc import OrderMCMCResult, run_order_mcmc


def _chain_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 1.5 * a + 0.3 * rng.normal(size=n)
    c = -1.2 * b + 0.3 * rng.normal(size=n)
    d = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def test_run_order_mcmc_runs_on_tiny_problem():
    data = _chain_data()
    names = list(data.columns)
    p = len(names)
    probs = np.full((p, p), 0.2)
    np.fill_diagonal(probs, 0.0)
    prior = PhysicsInformedGraphPrior(edge_probs=probs, penalty_acyclic=0.5)

    res = run_order_mcmc(
        data, names, prior, n_iter=60, burnin_frac=0.25, k_max_parents=2,
        seed=1, log_every=20,
    )

    assert isinstance(res, OrderMCMCResult)
    assert np.isfinite(res.best_score)
    assert res.best_dag.node_names == names
    assert res.edge_inclusion_probs.shape == (p, p)
    assert np.all((res.edge_inclusion_probs >= 0) & (res.edge_inclusion_probs <= 1))
    assert res.n_total == 60
    assert len(res.trace) == 3
    # The strong a–b and b–c dependencies must be captured (in some direction).
    adj = res.best_dag.adj
    assert adj[0, 1] or adj[1, 0]
    assert adj[1, 2] or adj[2, 1]


def test_run_order_mcmc_three_nodes_from_config_prior():
    data = _chain_data(n=200, seed=3)[["a", "b", "c"]]
    names = ["a", "b", "c"]
    prior = PhysicsInformedGraphPrior.from_config(
        names, {"edges": [{"parent": "a", "child": "b"}]}, penalty=1.0,
    )
    res = run_order_mcmc(data, names, prior, n_iter=20, k_max_parents=1, seed=0, log_every=10)
    assert np.isfinite(res.best_score)
    assert res.best_dag.adj.shape == (3, 3)
