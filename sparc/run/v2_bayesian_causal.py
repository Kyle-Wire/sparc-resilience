"""
V2 Bayesian causal inference orchestrator for SPARC pipeline.

Integrates MC³ structure learning and NUTS posterior sampling into
Stage 3 of the SPARC pipeline (causal validation).

Usage::

    from sparc.run.v2_bayesian_causal import run_bayesian_causal

    results = run_bayesian_causal(
        data=data_df,
        dag_def=dag_definition,
        neural_model=trained_model,
        config=cfg,
        output_dir=stage3_dir,
    )
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def run_bayesian_causal(
    data: pd.DataFrame,
    dag_def: dict[str, Any],
    neural_model: Optional[Any] = None,
    config: dict | None = None,
    output_dir: str | Path = ".",
) -> dict[str, Any]:
    """
    Run Bayesian causal analysis with MC³ + NUTS.

    Parameters
    ----------
    data : DataFrame — observational data
    dag_def : dict — parsed DAG definition (from load_dag)
    neural_model : optional SPARCMetaLearner for NUTS likelihood
    config : pipeline config dict
    output_dir : where to save artifacts

    Returns
    -------
    dict with ``mc3_results``, ``nuts_results`` (if neural_model given),
    ``edge_probs``, ``posterior_summaries``.
    """
    from sparc.causal.mc3 import (
        DAGStructure,
        MC3Results,
        PhysicsInformedGraphPrior,
        run_mc3,
    )

    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    causal_cfg = config.get("causal", {})
    mc3_cfg = causal_cfg.get("mc3", {})
    nuts_cfg = causal_cfg.get("nuts", {})

    # Node names from DAG definition
    node_names = [n["name"] for n in dag_def.get("nodes", [])]
    available_cols = [n for n in node_names if n in data.columns]

    if len(available_cols) < 2:
        logger.warning(
            "Only %d DAG nodes found in data columns — skipping MC³",
            len(available_cols),
        )
        return {"mc3_results": None, "nuts_results": None}

    # Build prior
    prior = PhysicsInformedGraphPrior.from_config(
        node_names=available_cols,
        dag_def=dag_def,
        penalty=mc3_cfg.get("edge_penalty", 1.0),
    )

    # Run MC³
    logger.info("Running MC³ structure learning over %d nodes...", len(available_cols))
    mc3_results = run_mc3(
        data=data[available_cols],
        node_names=available_cols,
        prior=prior,
        n_iter=mc3_cfg.get("n_iterations", 10_000),
        n_chains=mc3_cfg.get("n_chains", 4),
        temperatures=mc3_cfg.get("temperatures"),
        burnin_frac=mc3_cfg.get("burnin_fraction", 0.25),
        seed=mc3_cfg.get("seed", 42),
    )

    # Save MC³ results
    edge_probs = mc3_results.edge_inclusion_probs
    np.save(output_dir / "edge_inclusion_probs.npy", edge_probs)

    mc3_summary = {
        "n_accepted": mc3_results.n_accepted,
        "n_total": mc3_results.n_total,
        "acceptance_rate": mc3_results.n_accepted / max(mc3_results.n_total, 1),
        "best_score": mc3_results.best_score,
        "node_names": available_cols,
    }
    with open(output_dir / "mc3_summary.json", "w") as f:
        json.dump(mc3_summary, f, indent=2)

    # Save edge probability matrix as CSV for interpretability
    edge_df = pd.DataFrame(edge_probs, index=available_cols, columns=available_cols)
    edge_df.to_csv(output_dir / "edge_inclusion_probs.csv")

    logger.info(
        "MC³ done: %d accepted / %d total, best score=%.2f",
        mc3_results.n_accepted, mc3_results.n_total, mc3_results.best_score,
    )

    # ---- NUTS posterior sampling (if neural model provided) ----
    nuts_results = None

    if neural_model is not None and causal_cfg.get("inference") == "bayesian":
        logger.info("Running NUTS posterior sampling...")
        # Thread DAG definition into config for NUTS treatment identification
        nuts_config = dict(config)
        nuts_config.setdefault("causal", {})["dag"] = dag_def
        nuts_results = _run_nuts_sampling(
            data=data,
            node_names=available_cols,
            neural_model=neural_model,
            config=nuts_config,
            output_dir=output_dir,
        )

    return {
        "mc3_results": mc3_results,
        "nuts_results": nuts_results,
        "edge_probs": edge_probs,
        "mc3_summary": mc3_summary,
    }


def _run_nuts_sampling(
    data: pd.DataFrame,
    node_names: list[str],
    neural_model: Any,
    config: dict,
    output_dir: Path,
) -> dict[str, Any]:
    """Run NUTS posterior sampling using neural meta-learner likelihood.

    The neural model provides the baseline spatial prediction.  NUTS
    estimates treatment-effect coefficients (beta) on top of that
    baseline, giving a semi-parametric causal model:

        y = neural_baseline + X_treatment @ beta + ε

    where neural_baseline captures complex spatial patterns and
    confounding, while beta captures the causal treatment effects.
    """
    from sparc.causal.nuts import NUTSBlock, run_nuts

    nuts_cfg = config.get("causal", {}).get("nuts", {})
    dag_def = config.get("causal", {}).get("dag", {})
    target_col = config.get("variables", {}).get("target", node_names[-1])

    # Determine treatment variables from DAG definition
    treatments = [
        n["name"]
        for n in dag_def.get("nodes", [])
        if n.get("type") == "treatment"
    ]
    if not treatments:
        treatments = [n for n in node_names if n != target_col][:3]

    n_treatments = len(treatments)

    device = next(neural_model.parameters()).device
    dtype = torch.float64

    # --- Subsample for NUTS on large datasets ---
    max_nuts_rows = nuts_cfg.get("max_rows", 10000)
    if len(data) > max_nuts_rows:
        rng_sub = np.random.default_rng(nuts_cfg.get("seed", 42))
        idx = rng_sub.choice(len(data), size=max_nuts_rows, replace=False)
        idx.sort()
        data_nuts = data.iloc[idx].reset_index(drop=True)
        logger.info(
            "NUTS subsampled %d -> %d rows for tractable posterior",
            len(data), max_nuts_rows,
        )
    else:
        data_nuts = data

    y_obs = torch.tensor(data_nuts[target_col].values, dtype=dtype, device=device)

    # Standardize treatment features so all beta dimensions have
    # similar gradient magnitudes (prevents ill-conditioned Hessian
    # when features are on very different scales, e.g. 0-100 vs 0-1).
    X_raw = data_nuts[treatments].values.astype(np.float64)
    X_means = X_raw.mean(axis=0)
    X_stds = X_raw.std(axis=0)
    X_stds[X_stds < 1e-12] = 1.0  # avoid division by zero
    X_standardized = (X_raw - X_means) / X_stds
    X_treatment = torch.tensor(X_standardized, dtype=dtype, device=device)
    logger.info(
        "NUTS treatment standardization: means=%s  stds=%s",
        np.round(X_means, 4).tolist(), np.round(X_stds, 4).tolist(),
    )

    n_obs = len(data_nuts)

    # Get neural model baseline prediction (no gradient, fixed)
    neural_baseline = None
    try:
        # Build the input dict that predict_for_nuts expects
        neural_baseline_np = neural_model.predict_for_nuts(
            _build_nuts_input_dict(data_nuts, node_names, neural_model, config)
        )
        # Denormalise if the model was trained on z-scored targets
        _y_mean = config.get("training", {}).get("y_mean", 0.0)
        _y_std = config.get("training", {}).get("y_std", 1.0)
        neural_baseline_np = neural_baseline_np * _y_std + _y_mean

        neural_baseline = torch.tensor(
            neural_baseline_np, dtype=dtype, device=device,
        )
        logger.info(
            "Neural baseline for NUTS: mean=%.4f std=%.4f  (n=%d)",
            neural_baseline.mean().item(), neural_baseline.std().item(), n_obs,
        )
    except Exception as exc:
        logger.warning("Could not compute neural baseline for NUTS: %s", exc)

    # Initialize sigma2 at residual MLE so gradients are O(1) at start.
    # Without this, sigma2=1 while true MLE≈5+ causes gradient ~4500,
    # making leapfrog integration immediately unstable.
    residual_init = y_obs.clone()
    if neural_baseline is not None:
        residual_init = residual_init - neural_baseline
    sigma2_mle = float(torch.mean(residual_init ** 2).clamp(min=1e-6))
    log_sigma2_init = math.log(sigma2_mle)
    logger.info(
        "NUTS sigma2 init at residual MLE: %.4f (log=%.4f)",
        sigma2_mle, log_sigma2_init,
    )

    # Build sign-constraint map from DAG definition.
    # Physics: canopy cools (-1), impervious warms (+1), albedo cools (-1).
    sign_constraints = {}
    for node in dag_def.get("nodes", []):
        sc = node.get("sign_constraint")
        if sc is not None and node["name"] in treatments:
            sign_constraints[treatments.index(node["name"])] = sc
    # Defaults for well-known UHI treatments (if not in DAG)
    _default_signs = {"Pct_Canopy": -1, "Pct_Impervious": +1, "Albedo": -1}
    for i, tname in enumerate(treatments):
        if i not in sign_constraints and tname in _default_signs:
            sign_constraints[i] = _default_signs[tname]
    if sign_constraints:
        logger.info("NUTS physics sign constraints: %s",
                     {treatments[i]: s for i, s in sign_constraints.items()})

    # Define parameter blocks — only sample beta + sigma2.
    # (phi, alpha_latent, rho are structural placeholders saved
    #  as zero arrays for downstream compatibility.)
    # Initialize beta on the correct side of sign constraints.
    beta_init = np.zeros(n_treatments)
    for idx, required_sign in sign_constraints.items():
        beta_init[idx] = required_sign * 0.01  # small nudge in correct direction
    blocks = [
        NUTSBlock(name="beta", dim=n_treatments, init=beta_init),
        NUTSBlock(name="sigma2", dim=1, init=np.array([log_sigma2_init]),
                  transform="log"),
    ]

    def log_prob(params: dict[str, torch.Tensor]) -> torch.Tensor:
        beta = params["beta"]
        sigma2 = params["sigma2"]

        # Physics sign constraints via log-barrier (HMC-friendly).
        # Steep penalty that pushes the sampler away from sign violations
        # while keeping gradients finite for leapfrog integration.
        lp_sign = torch.tensor(0.0, dtype=dtype, device=device)
        for idx, required_sign in sign_constraints.items():
            b = beta[idx]
            if required_sign == -1:
                # beta must be ≤ 0 → log-barrier on (-b)
                lp_sign = lp_sign + torch.log(torch.clamp(-b, min=1e-12)) * 2.0
            elif required_sign == +1:
                # beta must be ≥ 0 → log-barrier on (b)
                lp_sign = lp_sign + torch.log(torch.clamp(b, min=1e-12)) * 2.0

        # Semi-parametric model: y = baseline + X @ beta + noise
        mu = X_treatment @ beta
        if neural_baseline is not None:
            mu = mu + neural_baseline

        # Normal log-likelihood
        ll = -0.5 * torch.sum(torch.log(sigma2) + (y_obs - mu) ** 2 / sigma2)

        # Priors
        lp_beta = -0.5 * torch.sum(beta ** 2 / 10.0)   # N(0, 10)
        lp_sigma = -2.0 * torch.log(sigma2)             # Inv-Gamma approx
        return ll + lp_beta + lp_sigma + lp_sign

    results = run_nuts(
        log_prob_fn=log_prob,
        blocks=blocks,
        n_samples=nuts_cfg.get("n_samples", 2000),
        n_warmup=nuts_cfg.get("n_warmup", 500),
        max_depth=nuts_cfg.get("max_tree_depth", 10),
        target_accept=nuts_cfg.get("target_accept_rate", 0.80),
        seed=nuts_cfg.get("seed", 42),
        device=str(device),
    )

    # Convert beta chains from standardized to original feature scale:
    # beta_original = beta_standardized / feature_std
    beta_chain_std = results.samples["beta"]  # (n_samples, n_treatments)
    beta_chain_orig = beta_chain_std / X_stds[np.newaxis, :]

    # Save chains in original scale
    np.save(output_dir / "nuts_beta.npy", beta_chain_orig)
    for name, chain in results.samples.items():
        if name != "beta":
            np.save(output_dir / f"nuts_{name}.npy", chain)

    # Save zero placeholders for structural blocks not sampled
    n_out = beta_chain_orig.shape[0]
    for placeholder, dim in [("phi", min(50, len(data))),
                             ("alpha_latent", 1), ("rho", 1)]:
        if placeholder not in results.samples:
            np.save(output_dir / f"nuts_{placeholder}.npy",
                    np.zeros((n_out, dim)))

    nuts_summary = {
        "acceptance_rate": results.acceptance_rate,
        "n_divergences": results.n_divergences,
        "treatments": treatments,
        "beta_mean": beta_chain_orig.mean(axis=0).tolist(),
        "beta_std": beta_chain_orig.std(axis=0).tolist(),
        "r_hat": {k: v.tolist() for k, v in results.r_hat.items()},
        "ess": {k: v.tolist() for k, v in results.ess.items()},
    }
    with open(output_dir / "nuts_summary.json", "w") as f:
        json.dump(nuts_summary, f, indent=2)

    logger.info(
        "NUTS done: %.1f%% accept, %d divergences",
        results.acceptance_rate * 100, results.n_divergences,
    )

    return nuts_summary


def _build_nuts_input_dict(
    data: pd.DataFrame,
    node_names: list[str],
    neural_model: Any,
    config: dict,
) -> dict[str, torch.Tensor]:
    """
    Build the X_dict for neural_model.predict_for_nuts().

    This requires matching the forward() signature of SPARCMetaLearner:
    base_preds, physics_feats, X_spatial, coords, knn_index, alpha.

    For NUTS baseline, we use dummy surrogate preds (zeros) and let
    the meta-learner rely on physics + spatial streams.
    """
    from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding
    from sparc.run.v2_neural_training import _build_knn_index

    device = next(neural_model.parameters()).device

    neural_cfg = config.get("models", {}).get("neural", {})
    n_freq = neural_cfg.get("sinusoidal_frequencies", 64)
    max_neighbors = neural_cfg.get("max_neighbors", 128)

    # Identify coordinate and feature columns
    feature_names = config.get("training", {}).get("feature_names", [])
    if not feature_names:
        target = config.get("variables", {}).get("target", node_names[-1])
        coord_cols = config.get("variables", {}).get("coords", ["POINT_X", "POINT_Y"])
        feature_names = [c for c in data.columns if c != target and c not in coord_cols]

    coord_cols = config.get("variables", {}).get("coords", ["POINT_X", "POINT_Y"])
    available_features = [f for f in feature_names if f in data.columns]
    available_coords = [c for c in coord_cols if c in data.columns]

    if len(available_coords) < 2 or len(available_features) < 1:
        raise ValueError("Cannot build NUTS input: insufficient columns in data")

    coords_np = data[available_coords].values.astype(np.float32)
    features_np = data[available_features].values.astype(np.float32)
    N = len(data)

    # Build spatial encoding
    encoder = SinusoidalSpatialEncoding(n_frequencies=n_freq)
    coords_t = torch.tensor(coords_np, dtype=torch.float32, device=device)
    X_spatial = encoder(coords_t)

    # KNN index
    knn_idx = torch.tensor(
        _build_knn_index(coords_np, max_neighbors),
        dtype=torch.long, device=device,
    )

    # Physics features
    physics_t = torch.tensor(features_np, dtype=torch.float32, device=device)

    # Dummy base predictions (surrogates not available in NUTS context)
    n_base = config.get("models", {}).get("neural", {}).get("n_base_models", 3)
    base_preds = torch.zeros(N, n_base, dtype=torch.float32, device=device)

    # Dummy alpha
    alpha = torch.full((N, 1), 1.0, dtype=torch.float32, device=device)

    return {
        "base_preds": base_preds,
        "physics_feats": physics_t,
        "X_spatial": X_spatial,
        "coords": coords_t,
        "knn_index": knn_idx,
        "alpha": alpha,
    }
