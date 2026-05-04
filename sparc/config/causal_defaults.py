"""Default Causal block for project.yml — Wager (2025) gold-standard.

Centralises the Bayesian + Wager-2025 defaults so that every template,
every project, and the in-code fallback share one source of truth.
"""

from __future__ import annotations

from typing import Any, Dict


CAUSAL_DEFAULTS: Dict[str, Any] = {
    "inference": "bayesian",     # MC³ + NUTS + Wager-2025 audit
    "estimator": "dml",          # cross-fit Debiased ML structural coefficients
    "estimate_cate": True,       # spatial CATE per treatment
    "cate_estimator": "bayesian",  # BayesianSpatialCATE (low-rank GP + NUTS)

    # Causal discovery — validates expert DAG against data.
    "discovery": {
        "enabled": True,
        "methods": ["pc_stable", "lingam", "ges"],
        "alpha": 0.05,
        "consensus_threshold": 2,
        "compare_expert": True,
    },

    # MC³ structure-learning (Bayesian DAG search).
    "mc3": {
        "n_iterations": 10000,
        "n_chains": 4,
        "burnin_fraction": 0.25,
        "edge_penalty": 1.0,
        "seed": 42,
    },

    # NUTS posterior sampling for global ATE.
    "nuts": {
        "n_samples": 2000,
        "n_warmup": 500,
        "n_chains": 2,
        "target_accept_rate": 0.85,
        "max_tree_depth": 8,
    },

    # Bayesian spatial CATE (random Fourier features + NUTS).
    "bayesian_cate": {
        "n_features": 32,
        "n_samples": 2000,
        "n_warmup": 500,
        "n_chains": 2,
        "target_accept_rate": 0.85,
        "max_tree_depth": 8,
        "kernel_lengthscale": 0.20,
    },

    # Wager (2025) audit gap toggles. Default: every gap on.
    "wager2025": {
        "overlap": True,         # Gap 1
        "rate_qini": True,       # Gap 2
        "spillover": True,       # Gap 3
        "iv": True,              # Gaps 4, 5
        "sequential_aipw": True, # Gap 6
        "panel": True,           # Gap 7
        "policy_learning": True, # Gap 8
        "cbps": True,            # Gap 9
    },
}


_REQUIRED_SUBKEYS = (
    "discovery", "mc3", "nuts", "bayesian_cate", "wager2025",
)


def merged_causal_block(user: Dict[str, Any]) -> Dict[str, Any]:
    """Return the user's causal block merged onto the defaults.

    Sub-block dicts are deep-merged one level (sub-block keys win when
    user provides them, missing keys fall back to defaults).
    """
    user = user or {}
    out: Dict[str, Any] = {}
    for k, v in CAUSAL_DEFAULTS.items():
        if isinstance(v, dict):
            out[k] = {**v, **(user.get(k) or {})}
        else:
            out[k] = user.get(k, v)
    # Carry over any user keys that the defaults don't know about
    # (e.g. ``actionable_variables``, ``fixed_variables``, ``dag_file``).
    for k, v in user.items():
        if k not in out:
            out[k] = v
    return out


def warn_missing_subblocks(user: Dict[str, Any]) -> list[str]:
    """Return human-readable warnings for missing sub-blocks."""
    user = user or {}
    msgs: list[str] = []
    if "causal" not in {"causal"} and not user:
        return msgs
    for sub in _REQUIRED_SUBKEYS:
        if sub not in user:
            msgs.append(
                f"causal.{sub} block missing — using Wager-2025 defaults "
                f"({list(CAUSAL_DEFAULTS[sub].keys())[:4]}...)."
            )
    return msgs
