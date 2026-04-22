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
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import psutil
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware detection (matches enhanced_spatial_cv approach)
# Each sub-stage (MC³ → DML → NUTS) runs sequentially and gets
# exclusive access to all CPU cores and available RAM.
# ---------------------------------------------------------------------------
_TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
_N_CORES = os.cpu_count() or 1
_MEMORY_LIMIT_GB = min(_TOTAL_RAM_GB * 0.75, 48)
_HIGH_MEMORY = _TOTAL_RAM_GB >= 32


def run_bayesian_causal(
    data: pd.DataFrame,
    dag_def: dict[str, Any],
    neural_model: Optional[Any] = None,
    config: dict | None = None,
    output_dir: str | Path = ".",
    approval_gate: Optional[Callable[[dict], None]] = None,
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
    approval_gate : optional callback invoked after MC³ with the MC³
        result summary dict.  The callback should block until the user
        approves (or raise ``RuntimeError`` to abort).  When *None* the
        pipeline proceeds without pausing.

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

    # Hardware banner — each sub-stage runs sequentially with full resources
    print(f"Stage 3 hardware-optimized configuration:")
    print(f"  - CPU cores: {_N_CORES}")
    print(f"  - Available RAM: {_TOTAL_RAM_GB:.1f} GB")
    print(f"  - Memory limit: {_MEMORY_LIMIT_GB:.0f} GB (75% of total)")
    print(f"  - High-memory mode: {'Enabled' if _HIGH_MEMORY else 'Disabled'}")
    print(f"  - Strategy: each sub-stage (MC³ → DML → NUTS) gets all cores + RAM")

    # Let numpy/BLAS use all cores for MC³ & DML linear algebra
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_MAX_THREADS"):
        os.environ[env_var] = str(_N_CORES)

    # Let PyTorch use all cores for NUTS tensor operations
    torch.set_num_threads(_N_CORES)
    try:
        torch.set_num_interop_threads(min(_N_CORES, 4))
    except RuntimeError:
        pass  # can only call once per process

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

    # V2 dev doc output: posterior_edge_probs.csv (alias for edge_inclusion_probs)
    edge_df.to_csv(output_dir / "posterior_edge_probs.csv")

    # V2 dev doc output: edge_confidence.csv — classify edges by strength
    edge_rows = []
    for i, src in enumerate(available_cols):
        for j, tgt in enumerate(available_cols):
            if i == j:
                continue
            prob = float(edge_probs[i, j])
            if prob >= 0.90:
                confidence = "strong"
            elif prob >= 0.70:
                confidence = "moderate"
            elif prob >= 0.50:
                confidence = "weak"
            else:
                confidence = "absent"
            edge_rows.append({
                "source": src,
                "target": tgt,
                "inclusion_prob": prob,
                "confidence": confidence,
            })
    pd.DataFrame(edge_rows).to_csv(output_dir / "edge_confidence.csv", index=False)

    # V2 dev doc output: median_probability_dag.json — edges above 0.50 threshold
    median_dag = {"nodes": available_cols, "edges": []}
    for i, src in enumerate(available_cols):
        for j, tgt in enumerate(available_cols):
            if i != j and edge_probs[i, j] >= 0.50:
                median_dag["edges"].append({
                    "source": src,
                    "target": tgt,
                    "probability": float(edge_probs[i, j]),
                })
    with open(output_dir / "median_probability_dag.json", "w") as f:
        json.dump(median_dag, f, indent=2)

    logger.info(
        "MC³ done: %d accepted / %d total, best score=%.2f",
        mc3_results.n_accepted, mc3_results.n_total, mc3_results.best_score,
    )

    # ---- MC³ visualizations ----
    _plot_mc3_visuals(edge_probs, available_cols, output_dir)

    # ---- Approval gate: pause for user review of MC³ results ----
    if approval_gate is not None:
        gate_payload = {
            "node_names": available_cols,
            "edge_probs": edge_probs.tolist(),
            "mc3_summary": mc3_summary,
            "median_dag": median_dag,
        }
        logger.info("Awaiting DAG approval from user...")
        approval_gate(gate_payload)  # blocks until user approves or raises
        logger.info("DAG approved — continuing to NUTS.")

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
            mc3_results=mc3_results,
        )

    # ---- NUTS visualizations ----
    if nuts_results is not None:
        _plot_nuts_visuals(nuts_results, output_dir)

    # ---- Publication-quality DAG + mediation maps ----
    _plot_dag_publication(edge_probs, available_cols, dag_def, output_dir)
    if nuts_results is not None:
        _plot_mediation_map(data, dag_def, nuts_results, config, output_dir)

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
    mc3_results: Any = None,
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

    print(f"  NUTS: {torch.get_num_threads()} CPU threads, "
          f"{_TOTAL_RAM_GB:.0f} GB RAM available", flush=True)

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

    # FIX B2: filter treatments by MC3 edge inclusion probabilities.
    # Only keep treatments with edges supported by the data (p > threshold).
    mc3_edge_threshold = nuts_cfg.get("mc3_edge_threshold", 0.3)
    if mc3_results is not None and hasattr(mc3_results, "edge_inclusion_probs"):
        edge_probs = mc3_results.edge_inclusion_probs
        target_idx = (
            node_names.index(target_col) if target_col in node_names else None
        )
        if target_idx is not None:
            supported = []
            for tname in treatments:
                if tname in node_names:
                    tidx = node_names.index(tname)
                    prob = edge_probs[tidx, target_idx]
                    if prob >= mc3_edge_threshold:
                        supported.append(tname)
                        logger.info(
                            "MC3 supports edge %s → %s (p=%.3f >= %.2f)",
                            tname, target_col, prob, mc3_edge_threshold,
                        )
                    else:
                        logger.info(
                            "MC3 dropped edge %s → %s (p=%.3f < %.2f)",
                            tname, target_col, prob, mc3_edge_threshold,
                        )
                else:
                    supported.append(tname)  # keep if not in MC3 scope
            if supported:
                treatments = supported
            else:
                logger.warning(
                    "MC3 dropped all treatments — keeping original set"
                )

    n_treatments = len(treatments)

    device = next(neural_model.parameters()).device
    dtype = torch.float64

    # Use CUDA if available for faster log_prob evaluation
    if torch.cuda.is_available() and str(device) == "cpu":
        device = torch.device("cuda")
        logger.info("CUDA available \u2014 switching NUTS to GPU")
        print("  NUTS: CUDA GPU detected, using GPU acceleration", flush=True)

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
    # Diagnostic: verify per-treatment standardization (esp. Albedo)
    for i, tname in enumerate(treatments):
        logger.info(
            "  %s: X_std=%.6f  raw_range=[%.4f, %.4f]  std_range=[%.4f, %.4f]",
            tname, X_stds[i],
            float(X_raw[:, i].min()), float(X_raw[:, i].max()),
            float(X_standardized[:, i].min()), float(X_standardized[:, i].max()),
        )
        print(f"    {tname}: X_std={X_stds[i]:.6f}  "
              f"raw=[{X_raw[:, i].min():.4f}, {X_raw[:, i].max():.4f}]  "
              f"standardized=[{X_standardized[:, i].min():.4f}, {X_standardized[:, i].max():.4f}]",
              flush=True)

    n_obs = len(data_nuts)

    # Get neural model baseline prediction (no gradient, fixed)
    neural_baseline = None
    try:
        # FIX B1: exclude treatment variables from neural baseline
        neural_baseline_np = neural_model.predict_for_nuts(
            _build_nuts_input_dict(
                data_nuts, node_names, neural_model, config,
                exclude_treatments=treatments,
            )
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

    # ------------------------------------------------------------------
    # Load ProcessRateNet alpha field for spatially-varying weights
    # ------------------------------------------------------------------
    # High-responsiveness areas (large α) are in active thermal adjustment
    # zones — their observations are more informative for estimating β.
    # Implemented as per-point weights on the log-likelihood.
    alpha_weights = None
    model_dir = Path(config.get("paths", {}).get("model_dir", "."))
    alpha_path = model_dir / "v2_neural" / "alpha_field.npy"
    alpha_coords_path = model_dir / "v2_neural" / "alpha_field_coords.npy"
    if alpha_path.exists():
        try:
            alpha_raw = np.load(alpha_path)
            alpha_coords = np.load(alpha_coords_path) if alpha_coords_path.exists() else None

            # Match alpha to NUTS subsample via nearest-neighbor coords
            coord_cols = config.get("variables", {}).get("coordinates",
                         config.get("variables", {}).get("coords", ["POINT_X", "POINT_Y"]))
            available_coords = [c for c in coord_cols if c in data_nuts.columns]
            if alpha_coords is not None and len(available_coords) >= 2:
                nuts_coords = data_nuts[available_coords].values.astype(np.float64)
                # Nearest-neighbor matching
                from scipy.spatial import cKDTree
                tree = cKDTree(alpha_coords)
                _, nn_idx = tree.query(nuts_coords, k=1)
                alpha_matched = alpha_raw[nn_idx]
            elif len(alpha_raw) == len(data_nuts):
                alpha_matched = alpha_raw
            else:
                alpha_matched = None

            if alpha_matched is not None:
                alpha_mean = float(np.mean(alpha_matched))
                if alpha_mean > 1e-12:
                    alpha_norm_np = alpha_matched / alpha_mean
                    # Clamp to [0.5, 2.0] — prevents extreme weighting
                    alpha_norm_np = np.clip(alpha_norm_np, 0.5, 2.0)
                    alpha_weights = torch.tensor(
                        alpha_norm_np, dtype=dtype, device=device,
                    )
                    logger.info(
                        "NUTS alpha weights: mean=%.4f std=%.4f min=%.4f max=%.4f",
                        float(alpha_weights.mean()), float(alpha_weights.std()),
                        float(alpha_weights.min()), float(alpha_weights.max()),
                    )
        except Exception as exc:
            logger.warning("Could not load alpha field for NUTS: %s", exc)
            alpha_weights = None

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

    # ------------------------------------------------------------------
    # Parameter blocks & log-probability
    # ------------------------------------------------------------------
    beta_init = np.zeros(n_treatments)
    for _idx, required_sign in sign_constraints.items():
        beta_init[_idx] = required_sign * 0.01

    # Per-treatment priors in STANDARDIZED beta space.
    prior_means_std = torch.zeros(n_treatments, dtype=dtype, device=device)
    prior_vars_std = torch.full((n_treatments,), 10.0, dtype=dtype, device=device)
    _informative_priors: dict[str, tuple[float, float]] = {
        "Albedo": (-5.0, 2.0),   # (mean, std) in original coeff scale
    }
    for i, tname in enumerate(treatments):
        if tname in _informative_priors:
            orig_mean, orig_std = _informative_priors[tname]
            prior_means_std[i] = orig_mean * X_stds[i]
            prior_vars_std[i] = (orig_std * X_stds[i]) ** 2
            _prior_std = math.sqrt(float(prior_vars_std[i]))
            logger.info(
                "Informative prior for %s: N(%.2f, %.2f) original "
                "→ N(%.4f, %.4f) standardized  [X_std=%.6f]",
                tname, orig_mean, orig_std,
                float(prior_means_std[i]), _prior_std,
                X_stds[i],
            )
            print(f"    {tname} prior: N({orig_mean}, {orig_std}) orig "
                  f"→ N({float(prior_means_std[i]):.4f}, {_prior_std:.4f}) std  "
                  f"[X_std={X_stds[i]:.6f}]", flush=True)

    blocks = [
        NUTSBlock(name="beta", dim=n_treatments, init=beta_init),
        NUTSBlock(name="sigma2", dim=1,
                  init=np.array([log_sigma2_init]), transform="log"),
    ]

    def log_prob(params: dict[str, torch.Tensor]) -> torch.Tensor:
        beta = params["beta"]
        sigma2 = params["sigma2"]

        barrier_scale = 10.0
        lp_sign = torch.tensor(0.0, dtype=dtype, device=device)
        for sc_idx, required_sign in sign_constraints.items():
            b = beta[sc_idx]
            if required_sign == -1:
                lp_sign = lp_sign - torch.nn.functional.softplus(b * barrier_scale)
            elif required_sign == +1:
                lp_sign = lp_sign - torch.nn.functional.softplus(-b * barrier_scale)

        mu = X_treatment @ beta
        if neural_baseline is not None:
            mu = mu + neural_baseline

        # Per-point log-likelihood with optional α-derived weights.
        # Higher α → higher weight → that point constrains β more.
        per_point_ll = -0.5 * (torch.log(sigma2) + (y_obs - mu) ** 2 / sigma2)
        if alpha_weights is not None:
            ll = torch.sum(alpha_weights * per_point_ll)
        else:
            ll = torch.sum(per_point_ll)

        lp_beta = -0.5 * torch.sum(
            (beta - prior_means_std) ** 2 / prior_vars_std
        )
        lp_sigma = -2.0 * torch.log(sigma2)
        return ll + lp_beta + lp_sigma + lp_sign

    results = run_nuts(
        log_prob_fn=log_prob,
        blocks=blocks,
        n_samples=nuts_cfg.get("n_samples", 15000),
        n_warmup=nuts_cfg.get("n_warmup", 1000),
        max_depth=nuts_cfg.get("max_tree_depth", 10),
        target_accept=nuts_cfg.get("target_accept_rate", 0.65),
        seed=nuts_cfg.get("seed", 42),
        device=str(device),
    )

    # ------------------------------------------------------------------
    # Extract posterior chains and save results
    # ------------------------------------------------------------------
    beta_chain_std = results.samples["beta"]  # (n_samples, n_treatments)
    beta_chain_orig = beta_chain_std / X_stds[np.newaxis, :]

    np.save(output_dir / "nuts_beta.npy", beta_chain_orig)
    for name, chain in results.samples.items():
        if name != "beta":
            np.save(output_dir / f"nuts_{name}.npy", chain)

    nuts_summary = {
        "acceptance_rate": results.acceptance_rate,
        "n_divergences": results.n_divergences,
        "treatments": treatments,
        "beta_mean": beta_chain_orig.mean(axis=0).tolist(),
        "beta_std": beta_chain_orig.std(axis=0).tolist(),
        "r_hat": {k: v.tolist() for k, v in results.r_hat.items()},
        "ess": {k: v.tolist() for k, v in results.ess.items()},
        "alpha_weighted": alpha_weights is not None,
    }
    with open(output_dir / "nuts_summary.json", "w") as f:
        json.dump(nuts_summary, f, indent=2)

    # ------------------------------------------------------------------
    # V2 dev doc output files (Stage 3 registry)
    # ------------------------------------------------------------------

    # 1. parameter_posteriors.csv — per-treatment posterior summary
    #    "mean" columns are per-raw-unit; "mean_per_std" is per-σ of treatment
    beta_mean = beta_chain_orig.mean(axis=0)
    beta_std = beta_chain_orig.std(axis=0)
    beta_q05 = np.percentile(beta_chain_orig, 5, axis=0)
    beta_q25 = np.percentile(beta_chain_orig, 25, axis=0)
    beta_q50 = np.percentile(beta_chain_orig, 50, axis=0)
    beta_q75 = np.percentile(beta_chain_orig, 75, axis=0)
    beta_q95 = np.percentile(beta_chain_orig, 95, axis=0)

    # Standardized (per-σ) summaries for comparable effect sizes
    beta_mean_std = beta_chain_std.mean(axis=0)
    beta_std_std = beta_chain_std.std(axis=0)
    beta_q05_std = np.percentile(beta_chain_std, 5, axis=0)
    beta_q95_std = np.percentile(beta_chain_std, 95, axis=0)

    posteriors_df = pd.DataFrame({
        "treatment": treatments,
        "treatment_std": X_stds,
        "mean": beta_mean,
        "std": beta_std,
        "ci_5": beta_q05,
        "ci_25": beta_q25,
        "median": beta_q50,
        "ci_75": beta_q75,
        "ci_95": beta_q95,
        "mean_per_std": beta_mean_std,
        "std_per_std": beta_std_std,
        "ci_5_per_std": beta_q05_std,
        "ci_95_per_std": beta_q95_std,
    })
    posteriors_df.to_csv(output_dir / "parameter_posteriors.csv", index=False)

    # 2. convergence_diagnostics.csv — R-hat and ESS per parameter
    diag_rows = []
    for block_name in ["beta", "sigma2"]:
        rh = results.r_hat.get(block_name, np.array([]))
        es = results.ess.get(block_name, np.array([]))
        conv = results.converged.get(block_name, np.array([]))
        if block_name == "beta":
            for i, tname in enumerate(treatments):
                diag_rows.append({
                    "parameter": f"beta[{tname}]",
                    "r_hat": float(rh[i]) if i < len(rh) else np.nan,
                    "ess": float(es[i]) if i < len(es) else np.nan,
                    "converged": bool(conv[i]) if i < len(conv) else False,
                })
        else:
            for i in range(len(rh)):
                diag_rows.append({
                    "parameter": f"{block_name}[{i}]",
                    "r_hat": float(rh[i]),
                    "ess": float(es[i]) if i < len(es) else np.nan,
                    "converged": bool(conv[i]) if i < len(conv) else False,
                })
    pd.DataFrame(diag_rows).to_csv(
        output_dir / "convergence_diagnostics.csv", index=False
    )

    # 3. bma_coefficients.csv — Bayesian model-averaged treatment effects
    #    Weighted by MC3 edge inclusion probabilities
    if mc3_results is not None and hasattr(mc3_results, "edge_inclusion_probs"):
        ep = mc3_results.edge_inclusion_probs
        target_idx = (
            node_names.index(target_col) if target_col in node_names else None
        )
        bma_rows = []
        for i, tname in enumerate(treatments):
            if tname in node_names and target_idx is not None:
                tidx = node_names.index(tname)
                edge_prob = float(ep[tidx, target_idx])
            else:
                edge_prob = 1.0
            bma_rows.append({
                "treatment": tname,
                "treatment_std": float(X_stds[i]),
                "posterior_mean": float(beta_mean[i]),
                "posterior_std": float(beta_std[i]),
                "edge_inclusion_prob": edge_prob,
                "bma_effect": float(beta_mean[i] * edge_prob),
                "bma_ci_5": float(beta_q05[i] * edge_prob),
                "bma_ci_95": float(beta_q95[i] * edge_prob),
                "bma_effect_per_std": float(beta_mean_std[i] * edge_prob),
                "bma_ci_5_per_std": float(beta_q05_std[i] * edge_prob),
                "bma_ci_95_per_std": float(beta_q95_std[i] * edge_prob),
            })
        pd.DataFrame(bma_rows).to_csv(
            output_dir / "bma_coefficients.csv", index=False
        )

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
    exclude_treatments: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Build the X_dict for neural_model.predict_for_nuts().

    This requires matching the forward() signature of SPARCMetaLearner:
    base_preds, physics_feats, X_spatial, coords, knn_index, alpha.

    For NUTS baseline, we use dummy surrogate preds (zeros) and let
    the meta-learner rely on physics + spatial streams.

    FIX B1: If ``exclude_treatments`` is given, those variables are
    zeroed-out in physics_feats so the neural baseline captures only
    confounder + spatial effects, not treatment effects.
    """
    from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding
    from sparc.run.v2_neural_training import _build_knn_index

    device = next(neural_model.parameters()).device

    neural_cfg = config.get("models", {}).get("neural", {})
    n_freq = neural_cfg.get("sinusoidal_frequencies", 64)
    max_neighbors = neural_cfg.get("max_neighbors", 128)

    # Identify coordinate and feature columns.
    # Priority: 1) training.feature_names  2) predictors.base_model  3) fallback
    feature_names = config.get("training", {}).get("feature_names", [])
    if not feature_names:
        feature_names = config.get("predictors", {}).get("base_model", [])
    if not feature_names:
        target = config.get("variables", {}).get("target", node_names[-1])
        id_col = config.get("variables", {}).get("identifier", "")
        coord_cols = config.get("variables", {}).get("coordinates",
                     config.get("variables", {}).get("coords", ["POINT_X", "POINT_Y"]))
        exclude = {target, id_col, "projected_X", "projected_Y"} | set(coord_cols)
        feature_names = [
            c for c in data.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(data[c])
        ]

    coord_cols = config.get("variables", {}).get("coordinates",
                 config.get("variables", {}).get("coords", ["POINT_X", "POINT_Y"]))
    available_features = [f for f in feature_names if f in data.columns]
    available_coords = [c for c in coord_cols if c in data.columns]

    if len(available_coords) < 2 or len(available_features) < 1:
        raise ValueError("Cannot build NUTS input: insufficient columns in data")

    coords_np = data[available_coords].values.astype(np.float32)
    features_np = data[available_features].values.astype(np.float32)
    N = len(data)

    # Build spatial encoding — fit normalisation on these coords
    encoder = SinusoidalSpatialEncoding(n_frequencies=n_freq)
    coords_t = torch.tensor(coords_np, dtype=torch.float32, device=device)
    encoder.fit(coords_t)
    X_spatial = encoder(coords_t)

    # KNN index
    knn_idx = torch.tensor(
        _build_knn_index(coords_np, max_neighbors),
        dtype=torch.long, device=device,
    )

    # Physics features
    physics_t = torch.tensor(features_np, dtype=torch.float32, device=device)

    # Pad physics features to match the trained model's expected input dimension.
    # The neural model was trained with extended features (original + PDE derivatives),
    # so physics_t may need to be padded from 6 raw features to the full expected dim.
    try:
        expected_phys_dim = neural_model.physics_enc.source_enc.net[0].in_features
        if physics_t.shape[1] < expected_phys_dim:
            pad = torch.zeros(
                N, expected_phys_dim - physics_t.shape[1],
                dtype=torch.float32, device=device,
            )
            physics_t = torch.cat([physics_t, pad], dim=1)
    except Exception:
        pass  # fall back to original dim; model will surface the shape error naturally

    # FIX B1: zero-out treatment columns so neural baseline is confounder-only
    if exclude_treatments:
        for tname in exclude_treatments:
            if tname in available_features:
                tidx = available_features.index(tname)
                physics_t[:, tidx] = 0.0

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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_mc3_visuals(
    edge_probs: np.ndarray,
    node_names: list[str],
    output_dir: Path,
) -> None:
    """Generate MC³ edge-probability heatmap and median-probability DAG plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping MC³ plots")
        return

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    n = len(node_names)

    # 1. Edge Probability Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(edge_probs, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Inclusion Probability", fontsize=11)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(node_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(node_names, fontsize=9)
    ax.set_xlabel("Target", fontsize=11)
    ax.set_ylabel("Source", fontsize=11)
    ax.set_title("MC³ Edge Inclusion Probabilities", fontsize=13, fontweight="bold")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = edge_probs[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(figs_dir / "dag_posterior_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Median Probability DAG (network diagram for edges >= 0.30)
    fig, ax = plt.subplots(figsize=(10, 8))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 3.0
    pos_x = radius * np.cos(angles)
    pos_y = radius * np.sin(angles)

    for i, name in enumerate(node_names):
        ax.plot(pos_x[i], pos_y[i], "o", markersize=28, color="#4A90D9",
                markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.text(pos_x[i], pos_y[i], name, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            prob = edge_probs[i, j]
            if prob < 0.30:
                continue
            dx = pos_x[j] - pos_x[i]
            dy = pos_y[j] - pos_y[i]
            dist = np.sqrt(dx**2 + dy**2)
            shrink = 0.7 / dist if dist > 0 else 0
            sx = pos_x[i] + dx * shrink
            sy = pos_y[i] + dy * shrink
            ex = pos_x[j] - dx * shrink
            ey = pos_y[j] - dy * shrink

            alpha_val = max(0.2, min(1.0, prob))
            width = 1.0 + 3.0 * prob
            color = "#E74C3C" if prob >= 0.50 else "#F39C12"
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=width, alpha=alpha_val,
                                        connectionstyle="arc3,rad=0.1"))
            mx = (sx + ex) / 2 + 0.15
            my = (sy + ey) / 2 + 0.15
            ax.text(mx, my, f"{prob:.2f}", fontsize=7, color=color,
                    alpha=alpha_val, fontweight="bold")

    ax.set_xlim(-radius - 1.5, radius + 1.5)
    ax.set_ylim(-radius - 1.5, radius + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Median Probability DAG (edges \u2265 0.30)", fontsize=13,
                 fontweight="bold")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#E74C3C", lw=2, label="Strong (p \u2265 0.50)"),
        Line2D([0], [0], color="#F39C12", lw=2, linestyle="--",
               label="Weak (0.30 \u2264 p < 0.50)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs_dir / "median_probability_dag.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    logger.info("MC³ visualizations saved to %s", figs_dir)


def _plot_nuts_visuals(
    nuts_summary: dict,
    output_dir: Path,
) -> None:
    """Generate NUTS posterior density and convergence diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping NUTS plots")
        return

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    treatments = nuts_summary.get("treatments", [])
    r_hat = nuts_summary.get("r_hat", {})
    ess = nuts_summary.get("ess", {})

    beta_chain_path = output_dir / "nuts_beta.npy"
    if not beta_chain_path.exists() or not treatments:
        return
    beta_chain = np.load(beta_chain_path)
    n_samples, n_treat = beta_chain.shape

    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]

    # 1. Posterior Density Plot
    fig, axes = plt.subplots(1, n_treat, figsize=(4 * n_treat, 4), squeeze=False)
    for i, tname in enumerate(treatments):
        ax = axes[0, i]
        chain = beta_chain[:, i]
        ax.hist(chain, bins=50, density=True, alpha=0.7,
                color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(chain), color="black", lw=2,
                   label=f"mean={np.mean(chain):.4f}")
        ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.5)
        ci5, ci95 = np.percentile(chain, [5, 95])
        ax.axvspan(ci5, ci95, alpha=0.15, color=colors[i % len(colors)],
                   label=f"90% CI [{ci5:.4f}, {ci95:.4f}]")
        ax.set_title(tname, fontsize=12, fontweight="bold")
        ax.set_xlabel("\u03b2 coefficient", fontsize=10)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("NUTS Posterior Densities", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figs_dir / "posterior_densities.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Trace Plot
    fig, axes = plt.subplots(n_treat, 1, figsize=(12, 3 * n_treat), squeeze=False)
    for i, tname in enumerate(treatments):
        ax = axes[i, 0]
        chain = beta_chain[:, i]
        ax.plot(chain, lw=0.3, alpha=0.7, color=colors[i % len(colors)])
        ax.axhline(np.mean(chain), color="black", lw=1, ls="--", alpha=0.7)
        ax.set_ylabel(tname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Sample" if i == n_treat - 1 else "", fontsize=10)
        ax.tick_params(labelsize=8)
        rh = r_hat.get("beta", [])
        es = ess.get("beta", [])
        rh_val = rh[i] if i < len(rh) else float("nan")
        es_val = es[i] if i < len(es) else float("nan")
        ax.text(0.98, 0.95, f"R\u0302={rh_val:.3f}  ESS={es_val:.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.suptitle("NUTS Trace Plots", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs_dir / "trace_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. R-hat & ESS Summary
    beta_rhat = r_hat.get("beta", [])
    beta_ess = ess.get("beta", [])
    if beta_rhat and beta_ess:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        x_pos = range(len(treatments))

        rh_colors = ["#2ECC71" if r < 1.1 else "#E74C3C" for r in beta_rhat]
        ax1.bar(x_pos, beta_rhat, color=rh_colors, edgecolor="white")
        ax1.axhline(1.1, color="red", ls="--", lw=1, label="threshold (1.1)")
        ax1.axhline(1.0, color="green", ls=":", lw=1, alpha=0.5)
        ax1.set_xticks(list(x_pos))
        ax1.set_xticklabels(treatments, rotation=30, ha="right", fontsize=9)
        ax1.set_ylabel("R\u0302", fontsize=11)
        ax1.set_title("Convergence (R\u0302)", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8)

        ax2.bar(x_pos, beta_ess, color="#3498DB", edgecolor="white")
        ax2.axhline(400, color="orange", ls="--", lw=1,
                    label="min recommended (400)")
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(treatments, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("ESS", fontsize=11)
        ax2.set_title("Effective Sample Size", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)

        fig.suptitle("NUTS Convergence Diagnostics", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(figs_dir / "rhat_ess_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info("NUTS visualizations saved to %s", figs_dir)


def _plot_dag_publication(
    edge_probs: np.ndarray,
    node_names: list[str],
    dag_def: dict[str, Any],
    output_dir: Path,
) -> None:
    """Publication-quality DAG with hierarchical layout and edge widths ∝ inclusion probability."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Classify nodes by role from DAG definition
    roles: dict[str, str] = {}
    for node in dag_def.get("nodes", []):
        roles[node["name"]] = node.get("type", "other")

    # Assign hierarchical y positions by role
    role_y = {"confounder": 3.0, "treatment": 2.0, "mediator": 1.0, "outcome": 0.0}
    role_colors = {
        "confounder": "#8E44AD", "treatment": "#E74C3C",
        "mediator": "#F39C12", "outcome": "#2ECC71", "other": "#95A5A6",
    }

    # Group nodes by role for x-spacing
    grouped: dict[str, list[str]] = {}
    for name in node_names:
        r = roles.get(name, "other")
        grouped.setdefault(r, []).append(name)

    pos: dict[str, tuple[float, float]] = {}
    for role, names in grouped.items():
        y = role_y.get(role, 1.5)
        n = len(names)
        x_start = -(n - 1) / 2.0
        for i, name in enumerate(names):
            pos[name] = (x_start + i, y)

    n = len(node_names)
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw edges with width ∝ inclusion probability
    for i, src in enumerate(node_names):
        for j, tgt in enumerate(node_names):
            if i == j:
                continue
            prob = edge_probs[i, j]
            if prob < 0.30:
                continue
            if src not in pos or tgt not in pos:
                continue

            sx, sy = pos[src]
            tx, ty = pos[tgt]
            dx, dy = tx - sx, ty - sy
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.01:
                continue
            shrink = 0.35 / dist
            x0 = sx + dx * shrink
            y0 = sy + dy * shrink
            x1 = tx - dx * shrink
            y1 = ty - dy * shrink

            width = 1.0 + 4.0 * prob
            alpha = max(0.3, min(1.0, prob))
            color = "#2C3E50" if prob >= 0.90 else "#E74C3C" if prob >= 0.50 else "#BDC3C7"
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=width,
                    alpha=alpha, connectionstyle="arc3,rad=0.08",
                    mutation_scale=15,
                ),
            )
            mx = (x0 + x1) / 2 + 0.08
            my = (y0 + y1) / 2 + 0.08
            ax.text(mx, my, f"{prob:.2f}", fontsize=7, color=color,
                    alpha=alpha, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Draw nodes
    for name in node_names:
        if name not in pos:
            continue
        x, y = pos[name]
        role = roles.get(name, "other")
        color = role_colors.get(role, "#95A5A6")
        circle = plt.Circle((x, y), 0.30, facecolor=color, edgecolor="white",
                             linewidth=2.5, zorder=5, alpha=0.92)
        ax.add_patch(circle)
        ax.text(x, y, name.replace("_", "\n"), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=role_colors["confounder"], label="Confounder"),
        Patch(facecolor=role_colors["treatment"], label="Treatment"),
        Patch(facecolor=role_colors["mediator"], label="Mediator"),
        Patch(facecolor=role_colors["outcome"], label="Outcome"),
        Line2D([0], [0], color="#2C3E50", lw=3, label="Strong (p ≥ 0.90)"),
        Line2D([0], [0], color="#E74C3C", lw=2, label="Moderate (0.50–0.90)"),
        Line2D([0], [0], color="#BDC3C7", lw=1.5, label="Weak (0.30–0.50)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    pad = 0.8
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Bayesian Causal DAG — Edge Inclusion Probabilities (MC³)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(figs_dir / "dag_publication.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Publication DAG saved to %s", figs_dir / "dag_publication.png")


def _plot_mediation_map(
    data: pd.DataFrame,
    dag_def: dict[str, Any],
    nuts_results: dict,
    config: dict | None,
    output_dir: Path,
) -> None:
    """Spatial map of the indirect Canopy → NDVI → AAT_z mediation pathway magnitude."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    config = config or {}
    coord_cols = config.get("variables", {}).get("coordinates", [])
    if len(coord_cols) < 2:
        logger.warning("No coordinate columns — skipping mediation map")
        return
    if not all(c in data.columns for c in coord_cols):
        logger.warning("Coordinate columns not in data — skipping mediation map")
        return

    # Load structural coefficients for the mediation pathway
    coeff_path = output_dir.parent / "scenario_coefficients.json"
    if not coeff_path.exists():
        coeff_path = output_dir / ".." / "scenario_coefficients.json"
    if not coeff_path.exists():
        logger.warning("scenario_coefficients.json not found — skipping mediation map")
        return

    with open(coeff_path) as f:
        coeffs = json.load(f)

    # Get Canopy→NDVI and NDVI→AAT_z structural coefficients
    edges = coeffs.get("edges", {})
    canopy_ndvi = edges.get("Pct_Canopy->NDVI", {}).get("structural_coeff")
    ndvi_aat = coeffs.get("direct_effects", {}).get("NDVI", {}).get("structural_coeff")

    if canopy_ndvi is None or ndvi_aat is None:
        # Try to estimate from data correlation
        if "Pct_Canopy" in data.columns and "NDVI" in data.columns:
            from numpy.linalg import lstsq
            X_can = data["Pct_Canopy"].values.reshape(-1, 1)
            X_can = np.column_stack([X_can, np.ones(len(X_can))])
            canopy_ndvi = float(lstsq(X_can, data["NDVI"].values, rcond=None)[0][0])
        if ndvi_aat is None:
            ndvi_aat = -4.131  # fallback from structural coefficient
        if canopy_ndvi is None:
            logger.warning("Cannot compute mediation coefficients — skipping")
            return

    x = data[coord_cols[0]].values
    y = data[coord_cols[1]].values

    # Indirect effect at each point = canopy_value * β(canopy→NDVI) * β(NDVI→AAT_z)
    # This represents the local mediation magnitude
    canopy_vals = data["Pct_Canopy"].values if "Pct_Canopy" in data.columns else np.zeros(len(data))
    ndvi_vals = data["NDVI"].values if "NDVI" in data.columns else np.zeros(len(data))

    # Mediation strength: how much of the temperature at each point
    # is explained by the Canopy→NDVI→AAT_z pathway
    indirect_effect = ndvi_vals * ndvi_aat  # NDVI contribution to temperature

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: NDVI-mediated cooling contribution
    vmax = max(abs(np.percentile(indirect_effect, 2)),
               abs(np.percentile(indirect_effect, 98)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc1 = axes[0].scatter(x, y, c=indirect_effect, cmap="RdBu_r",
                          norm=norm, s=0.3, alpha=0.6, rasterized=True)
    plt.colorbar(sc1, ax=axes[0], shrink=0.8, label="NDVI→AAT_z effect (z-score)")
    axes[0].set_title("NDVI-Mediated Temperature Effect", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(coord_cols[0])
    axes[0].set_ylabel(coord_cols[1])
    axes[0].set_aspect("equal")

    # Panel 2: Canopy-driven NDVI (mediator source)
    sc2 = axes[1].scatter(x, y, c=canopy_vals, cmap="Greens",
                          s=0.3, alpha=0.6, rasterized=True)
    plt.colorbar(sc2, ax=axes[1], shrink=0.8, label="Pct_Canopy (%)")
    axes[1].set_title("Canopy Cover (Mediation Source)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(coord_cols[0])
    axes[1].set_ylabel(coord_cols[1])
    axes[1].set_aspect("equal")

    coeff_text = f"β(Canopy→NDVI) = {canopy_ndvi:.4f}\nβ(NDVI→AAT_z) = {ndvi_aat:.3f}"
    fig.text(0.5, 0.01, coeff_text, ha="center", fontsize=10, style="italic")
    fig.suptitle("NDVI Mediation Pathway: Pct_Canopy → NDVI → AAT_z",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figs_dir / "mediation_map_ndvi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Mediation map saved to %s", figs_dir / "mediation_map_ndvi.png")
