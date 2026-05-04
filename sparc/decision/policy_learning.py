"""Empirical-welfare maximization (Athey-Wager 2021).

Wager (2025) audit Gap 8. Given AIPW pseudo-outcomes Γ̂_i (per-unit
doubly-robust treatment-effect scores), learn a policy π: X → {0, 1}
maximising estimated welfare ``W(π) = E[π(X) Γ̂(X)]`` over a restricted
policy class:

- ``"linear"``     — score-based threshold rule on linear combinations
                     of features; sweep thresholds for budget feasibility.
- ``"tree"``       — depth-≤2 decision tree over features.
- ``"sparse"``     — k-sparse linear rule (top-k features by gain).

Subject to optional per-unit cost / budget constraints. AIPW scores can
be supplied directly or computed via :func:`sparc.causal.cate_validation.aipw_pseudo_outcomes`.

Cost-aware ranking reuses :func:`sparc.decision.optimizer.score_equity` and
:func:`sparc.decision.optimizer.rank_interventions` patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sparc.causal._audit import mark_addressed

GAP_8_IMPLEMENTED = True


@dataclass
class PolicyResult:
    treated: np.ndarray
    welfare: float
    policy_class: str
    threshold: Optional[float] = None
    feature_weights: Optional[np.ndarray] = None
    tree: Optional[object] = None


class EmpiricalWelfareMaximizer:
    """Restricted-class empirical-welfare maximisation."""

    def __init__(self,
                 policy_class: str = "linear",
                 budget: Optional[float] = None,
                 sparsity_k: int = 3,
                 tree_depth: int = 2):
        if policy_class not in ("linear", "tree", "sparse"):
            raise ValueError(f"unknown policy_class: {policy_class}")
        self.policy_class = policy_class
        self.budget = budget
        self.sparsity_k = sparsity_k
        self.tree_depth = tree_depth
        self._result: Optional[PolicyResult] = None

    # ------------------------------------------------------------------ fit

    def fit(self,
            X: np.ndarray,
            aipw_scores: np.ndarray,
            cost: Optional[np.ndarray] = None) -> "EmpiricalWelfareMaximizer":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 1 and X.shape[1] != len(aipw_scores):
            X = X.T
        gamma = np.asarray(aipw_scores, dtype=float).ravel()
        n, d = X.shape
        if cost is None:
            cost = np.ones(n)
        cost = np.asarray(cost, dtype=float).ravel()

        if self.policy_class == "linear":
            self._result = self._fit_linear(X, gamma, cost)
        elif self.policy_class == "sparse":
            self._result = self._fit_sparse(X, gamma, cost)
        else:
            self._result = self._fit_tree(X, gamma, cost)

        mark_addressed(8)
        return self

    # ------------------------------------------------------------------ api

    def recommend(self) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("fit() must be called first")
        return self._result.treated.astype(int)

    @property
    def welfare_(self) -> float:
        if self._result is None:
            raise RuntimeError("fit() must be called first")
        return self._result.welfare

    @property
    def policy_(self) -> PolicyResult:
        if self._result is None:
            raise RuntimeError("fit() must be called first")
        return self._result

    # --------------------------------------------------------------- linear

    def _fit_linear(self, X: np.ndarray, gamma: np.ndarray,
                    cost: np.ndarray) -> PolicyResult:
        # 1. Score units by an OLS regression of gamma on X (linear scoring).
        Xc = X - X.mean(axis=0)
        coefs = np.linalg.pinv(Xc.T @ Xc) @ Xc.T @ gamma
        scores = Xc @ coefs

        # 2. Sweep thresholds for budget-feasible welfare maximisation.
        order = np.argsort(-scores)
        cumcost = np.cumsum(cost[order])
        cumwel = np.cumsum(gamma[order])
        if self.budget is None:
            best_k = int(np.argmax(cumwel)) + 1
        else:
            feasible = np.where(cumcost <= self.budget)[0]
            best_k = (int(feasible[np.argmax(cumwel[feasible])]) + 1
                      if len(feasible) else 0)
        treated = np.zeros(len(gamma), dtype=bool)
        if best_k > 0:
            treated[order[:best_k]] = True
        return PolicyResult(
            treated=treated,
            welfare=float(gamma[treated].sum()),
            policy_class="linear",
            threshold=float(scores[order[best_k - 1]]) if best_k > 0 else None,
            feature_weights=coefs,
        )

    # --------------------------------------------------------------- sparse

    def _fit_sparse(self, X: np.ndarray, gamma: np.ndarray,
                    cost: np.ndarray) -> PolicyResult:
        # Pick top-k features by absolute correlation with gamma.
        d = X.shape[1]
        scores_per_feat = np.array([
            abs(np.corrcoef(X[:, j], gamma)[0, 1]) for j in range(d)
        ])
        scores_per_feat = np.nan_to_num(scores_per_feat)
        top = np.argsort(-scores_per_feat)[:self.sparsity_k]
        Xs = X[:, top]
        sub = EmpiricalWelfareMaximizer(policy_class="linear",
                                        budget=self.budget).fit(
            Xs, gamma, cost=cost
        )
        res = sub.policy_
        full_weights = np.zeros(d)
        if res.feature_weights is not None:
            full_weights[top] = res.feature_weights
        return PolicyResult(
            treated=res.treated,
            welfare=res.welfare,
            policy_class="sparse",
            threshold=res.threshold,
            feature_weights=full_weights,
        )

    # ----------------------------------------------------------------- tree

    def _fit_tree(self, X: np.ndarray, gamma: np.ndarray,
                  cost: np.ndarray) -> PolicyResult:
        tree = DecisionTreeRegressor(max_depth=self.tree_depth)
        tree.fit(X, gamma)
        leaf_pred = tree.predict(X)
        # Treat units whose leaf has positive predicted score.
        treated = leaf_pred > 0
        if self.budget is not None:
            # Cost-aware budget enforcement: rank positive-score units by
            # predicted gain (descending) and take the longest prefix
            # whose cumulative ``cost`` does not exceed ``budget``.  This
            # matches the contract documented in the module docstring and
            # in :meth:`_fit_linear`.
            order = np.argsort(-leaf_pred)
            cumcost = np.cumsum(cost[order])
            positive = leaf_pred[order] > 0
            feasible = (cumcost <= self.budget) & positive
            keep = order[feasible]
            treated = np.zeros_like(treated, dtype=bool)
            treated[keep] = True
        return PolicyResult(
            treated=treated,
            welfare=float(gamma[treated].sum()),
            policy_class="tree",
            tree=tree,
        )
