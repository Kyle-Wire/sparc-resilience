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

from sparc.config.hardware_profile import detect_profile
from sparc.data.units import format_effect, load_variable_meta
from sparc.registry.run_registry import get_active_registry
from sparc.registry.store import ArtifactStore

logger = logging.getLogger(__name__)


def _get_store() -> ArtifactStore | None:
    """Return active ArtifactStore or None when running outside the pipeline."""
    reg = get_active_registry()
    if reg is None:
        return None
    try:
        return ArtifactStore(reg)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hardware detection (matches enhanced_spatial_cv approach)
# Each sub-stage (MC³ → DML → NUTS) runs sequentially and gets
# exclusive access to all CPU cores and available RAM, scaled by the
# detected hardware tier so low-RAM machines don't get OOM-killed.
#
# These are populated at import and refreshed at the top of `run_bayesian_causal`
# so user-changed performance settings take effect on the next pipeline run.
# ---------------------------------------------------------------------------
_PROFILE = detect_profile()
_TOTAL_RAM_GB = _PROFILE.total_ram_gb
_N_CORES = _PROFILE.cpu_count
_MEMORY_LIMIT_GB = _PROFILE.memory_limit_gb
_HIGH_MEMORY = _PROFILE.high_memory_mode
_NUTS_THIN = _PROFILE.nuts_thin


def _refresh_hardware_globals() -> None:
    """Re-read the hardware profile and refresh the module-level globals."""
    global _PROFILE, _TOTAL_RAM_GB, _N_CORES, _MEMORY_LIMIT_GB, _HIGH_MEMORY, _NUTS_THIN
    from sparc.config.hardware_profile import reset_profile_cache
    reset_profile_cache()
    _PROFILE = detect_profile()
    _TOTAL_RAM_GB = _PROFILE.total_ram_gb
    _N_CORES = _PROFILE.cpu_count
    _MEMORY_LIMIT_GB = _PROFILE.memory_limit_gb
    _HIGH_MEMORY = _PROFILE.high_memory_mode
    _NUTS_THIN = _PROFILE.nuts_thin


def run_bayesian_causal(
    data: pd.DataFrame,
    dag_def: dict[str, Any],
    neural_model: Optional[Any] = None,
    config: dict | None = None,
    output_dir: str | Path = ".",
    approval_gate: Optional[Callable[[dict], None]] = None,
    discovery_report: dict | None = None,
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
    discovery_report : optional dict from ``run_causal_discovery()``.
        When provided, included in the ``approval_gate`` payload so the
        user sees PC/LiNGAM/GES discrepancies alongside MC³ results
        before deciding whether to proceed.  NUTS always receives the
        original unmodified ``dag_def`` regardless of gate outcome.

    Returns
    -------
    dict with ``mc3_results``, ``nuts_results`` (if neural_model given),
    ``edge_probs``, ``posterior_summaries``.
    """
    # Pick up any user-changed performance settings before this run.
    _refresh_hardware_globals()

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

    mc3_summary = {
        "n_accepted": mc3_results.n_accepted,
        "n_total": mc3_results.n_total,
        "acceptance_rate": mc3_results.n_accepted / max(mc3_results.n_total, 1),
        "best_score": mc3_results.best_score,
        "node_names": available_cols,
    }

    # Edge probability matrix in long form (Stage-3 source of truth)
    edge_rows = []
    median_dag = {"nodes": available_cols, "edges": []}
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
            if prob >= 0.50:
                median_dag["edges"].append({
                    "source": src,
                    "target": tgt,
                    "probability": prob,
                })

    edge_df = pd.DataFrame(edge_probs, index=available_cols, columns=available_cols)
    edge_confidence_df = pd.DataFrame(edge_rows)

    _store = _get_store()
    if _store is not None:
        _store.write_struct("3", "mc3_summary", mc3_summary, producer="v2_bayesian_causal")
        # Wide matrix written with index column preserved
        _store.write_table(
            "3", "edge_inclusion_probs",
            edge_df.reset_index(names="source"),
            producer="v2_bayesian_causal",
        )
        _store.write_table(
            "3", "edge_confidence", edge_confidence_df,
            producer="v2_bayesian_causal",
        )
        _store.write_struct(
            "3", "median_probability_dag", median_dag,
            producer="v2_bayesian_causal",
        )
        _store.write_blob(
            "3", "edge_inclusion_probs_npy", edge_probs,
            serializer="pickle", producer="v2_bayesian_causal",
        )
    else:
        # Disk fallback when invoked outside the active pipeline
        np.save(output_dir / "edge_inclusion_probs.npy", edge_probs)
        with open(output_dir / "mc3_summary.json", "w") as f:
            json.dump(mc3_summary, f, indent=2)
        edge_df.to_csv(output_dir / "edge_inclusion_probs.csv")
        edge_df.to_csv(output_dir / "posterior_edge_probs.csv")
        edge_confidence_df.to_csv(output_dir / "edge_confidence.csv", index=False)
        with open(output_dir / "median_probability_dag.json", "w") as f:
            json.dump(median_dag, f, indent=2)

    logger.info(
        "MC³ done: %d accepted / %d total, best score=%.2f",
        mc3_results.n_accepted, mc3_results.n_total, mc3_results.best_score,
    )

    # ---- Approval gate: pause for user review of MC³ results ----
    if approval_gate is not None:
        gate_payload = {
            "node_names": available_cols,
            "edge_probs": edge_probs.tolist(),
            "mc3_summary": mc3_summary,
            "median_dag": median_dag,
            # Discovery discrepancies (PC/LiNGAM/GES vs expert DAG) included
            # so the user can review data-driven findings alongside MC³ before
            # deciding to proceed.  NUTS always receives the original dag_def.
            "discovery_report": discovery_report or {},
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
        thin=_NUTS_THIN,
    )

    # ------------------------------------------------------------------
    # Extract posterior chains and save results
    # ------------------------------------------------------------------
    beta_chain_std = results.samples["beta"]  # (n_samples, n_treatments)
    beta_chain_orig = beta_chain_std / X_stds[np.newaxis, :]

    _store = _get_store()
    if _store is not None:
        _store.write_blob(
            "3", "nuts_beta", beta_chain_orig,
            serializer="pickle", producer="v2_bayesian_causal",
        )
        for name, chain in results.samples.items():
            if name == "beta":
                continue
            _store.write_blob(
                "3", f"nuts_{name}", np.asarray(chain),
                serializer="pickle", producer="v2_bayesian_causal",
            )
    else:
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
    if _store is not None:
        _store.write_struct(
            "3", "nuts_summary", nuts_summary,
            producer="v2_bayesian_causal",
        )
    else:
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
    # Human-readable effect strings (e.g.
    # "-0.28 Fahrenheit (90% CI: -0.41 to -0.15) per 1% of canopy increase")
    try:
        _var_reg = load_variable_meta(config)
        posteriors_df["display"] = [
            format_effect(
                tname, target_col, float(beta_mean[i]), _var_reg,
                ci=(float(beta_q05[i]), float(beta_q95[i])), ci_level=90,
            )
            for i, tname in enumerate(treatments)
        ]
    except Exception as _exc:  # pragma: no cover - never block writers
        logger.warning("parameter_posteriors display formatting failed: %s", _exc)
    if _store is not None:
        _store.write_table(
            "3", "parameter_posteriors", posteriors_df,
            producer="v2_bayesian_causal",
        )
    else:
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
    diag_df = pd.DataFrame(diag_rows)
    if _store is not None:
        _store.write_table(
            "3", "convergence_diagnostics", diag_df,
            producer="v2_bayesian_causal",
        )
    else:
        diag_df.to_csv(
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
        bma_df = pd.DataFrame(bma_rows)
        try:
            _var_reg = load_variable_meta(config)
            bma_df["display"] = [
                format_effect(
                    row["treatment"], target_col, float(row["bma_effect"]),
                    _var_reg,
                    ci=(float(row["bma_ci_5"]), float(row["bma_ci_95"])),
                    ci_level=90,
                )
                for row in bma_rows
            ]
        except Exception as _exc:  # pragma: no cover
            logger.warning("bma_coefficients display formatting failed: %s", _exc)
        if _store is not None:
            _store.write_table(
                "3", "bma_coefficients", bma_df,
                producer="v2_bayesian_causal",
            )
        else:
            bma_df.to_csv(
                output_dir / "bma_coefficients.csv", index=False
            )

    logger.info(
        "NUTS done: %.1f%% accept, %d divergences",
        results.acceptance_rate * 100, results.n_divergences,
    )

    # ------------------------------------------------------------------
    # Per-edge NUTS posteriors (Phase 1 of v4 causal-stack rewrite)
    # ------------------------------------------------------------------
    # Sample β for every (parent, child) edge in the DAG, not just
    # treatment → target.  Mediator edges are residualized on the
    # backdoor adjustment set first to recover the structural coeff.
    # Persisted as ("3","nuts_edge_summary") + ("3","nuts_edge_samples").
    try:
        edge_summary = _run_per_edge_nuts_sampling(
            data=data_nuts,
            dag_def=dag_def,
            node_names=node_names,
            target_col=target_col,
            neural_baseline=neural_baseline,
            alpha_weights=alpha_weights,
            nuts_cfg=nuts_cfg,
            mc3_results=mc3_results,
            output_dir=output_dir,
            sign_default_map={"Pct_Canopy": -1, "Pct_Impervious": +1, "Albedo": -1},
            device=device,
            dtype=dtype,
            config=config,
        )
        nuts_summary["per_edge"] = edge_summary
    except Exception as exc:
        logger.warning("Per-edge NUTS sampling failed: %s", exc, exc_info=True)
        nuts_summary["per_edge"] = {"status": "failed", "error": str(exc)}

    return nuts_summary


def _run_per_edge_nuts_sampling(
    *,
    data: pd.DataFrame,
    dag_def: dict,
    node_names: list[str],
    target_col: str,
    neural_baseline: Any,
    alpha_weights: Any,
    nuts_cfg: dict,
    mc3_results: Any,
    output_dir: Path,
    sign_default_map: dict[str, int],
    device: Any,
    dtype: Any,
    config: dict | None = None,
) -> dict[str, Any]:
    """Sample β per DAG edge with per-edge NUTS (Phase 1 of causal-stack rewrite).

    For each ``(parent, child)`` edge in the DAG:

    * If ``child == target_col`` and ``neural_baseline`` is given, fit
      ``y - neural_baseline = parent_std × β + ε``.
    * Otherwise residualize both ``parent`` and ``child`` on the *other
      parents of child* (OLS) and fit
      ``child_resid = parent_std_resid × β + ε`` to recover the **direct**
      structural coefficient (holding the remaining parents of child
      fixed). This is the SEM interpretation, and is appropriate when
      the DAG is taken as the data-generating model.

    Edges with ``mc3_edge_inclusion_prob < mc3_edge_threshold`` are skipped
    (compute saver). Sign constraints from ``sign_default_map`` apply only
    when ``child == target_col``.

    Persists:

    * ``("3", "nuts_edge_summary")`` table — long-format with one row per
      sampled edge ``(parent, child, mean, std, ci5, ci50, ci95, r_hat,
      ess, n_samples, prior_used, sign_constrained)``.
    * ``("3", "nuts_edge_samples")`` blob — pickled
      ``Dict[Tuple[str, str], np.ndarray]`` mapping ``(parent, child)`` to
      the posterior sample vector in raw (unstandardized) units.
    """
    from sparc.causal.nuts import NUTSBlock, run_nuts

    edges = dag_def.get("edges", [])
    if not edges:
        logger.info("Per-edge NUTS: DAG has no edges, skipping")
        return {"status": "no_edges", "n_sampled": 0}

    # Per-edge MC³ threshold (independent from treatment-only filter above).
    mc3_threshold = float(nuts_cfg.get("mc3_edge_threshold", 0.3))
    edge_probs = None
    if mc3_results is not None and hasattr(mc3_results, "edge_inclusion_probs"):
        edge_probs = mc3_results.edge_inclusion_probs

    # Reduced budgets for per-edge runs — many edges to sample.
    n_samples = int(nuts_cfg.get("per_edge_n_samples", 3000))
    n_warmup = int(nuts_cfg.get("per_edge_n_warmup", 500))
    seed_base = int(nuts_cfg.get("seed", 42))

    # Sign constraints: only for direct edges to the target.
    sign_node_overrides: dict[str, int] = {}
    for node in dag_def.get("nodes", []):
        sc = node.get("sign_constraint")
        if sc is not None:
            sign_node_overrides[node["name"]] = int(sc)

    summary_rows: list[dict[str, Any]] = []
    samples_dict: dict[tuple[str, str], np.ndarray] = {}
    n_sampled = 0
    n_skipped_mc3 = 0
    n_skipped_missing = 0
    n_failed = 0

    # Cast neural_baseline / alpha_weights once for arithmetic on numpy.
    nb_np = (
        neural_baseline.detach().cpu().numpy().astype(np.float64)
        if neural_baseline is not None else None
    )
    aw_np = (
        alpha_weights.detach().cpu().numpy().astype(np.float64)
        if alpha_weights is not None else None
    )

    # Hash-derived per-edge seed for reproducibility across DAG variants.
    def _edge_seed(p: str, c: str) -> int:
        h = abs(hash((seed_base, p, c))) % (2**31 - 1)
        return int(h)

    for edge_idx, edge in enumerate(edges):
        parent = edge.get("parent")
        child = edge.get("child")
        if not parent or not child or parent == child:
            continue
        if parent not in data.columns or child not in data.columns:
            n_skipped_missing += 1
            logger.debug(
                "Per-edge NUTS: skip %s→%s (missing column in data)",
                parent, child,
            )
            continue

        # MC³ pruning
        edge_prob_val: Optional[float] = None
        if edge_probs is not None and parent in node_names and child in node_names:
            pi = node_names.index(parent)
            ci = node_names.index(child)
            edge_prob_val = float(edge_probs[pi, ci])
            if edge_prob_val < mc3_threshold:
                n_skipped_mc3 += 1
                logger.info(
                    "Per-edge NUTS: skip %s→%s (mc3 p=%.3f < %.2f)",
                    parent, child, edge_prob_val, mc3_threshold,
                )
                continue

        # Build response and design vector
        try:
            child_arr = data[child].values.astype(np.float64)
            parent_arr = data[parent].values.astype(np.float64)

            if child == target_col and nb_np is not None and len(nb_np) == len(child_arr):
                # Use neural baseline for the target outcome.
                y_arr = child_arr - nb_np
                x_arr = parent_arr
            else:
                # Structural-equation interpretation: the β on (parent→child)
                # is the direct effect, holding the OTHER parents of child
                # fixed. Residualize both parent and child on the remaining
                # parents to isolate the direct edge coefficient.
                from sklearn.linear_model import LinearRegression
                from sparc.causal.dag_definition import dag_to_networkx

                G = dag_to_networkx(dag_def)
                other_parents = [
                    p for p in G.predecessors(child)
                    if p != parent and p in data.columns
                ]

                if other_parents:
                    Z = data[other_parents].values.astype(np.float64)
                    lr_y = LinearRegression().fit(Z, child_arr)
                    lr_x = LinearRegression().fit(Z, parent_arr)
                    y_arr = child_arr - lr_y.predict(Z)
                    x_arr = parent_arr - lr_x.predict(Z)
                else:
                    # Center to remove intercept.
                    y_arr = child_arr - child_arr.mean()
                    x_arr = parent_arr - parent_arr.mean()
        except Exception as exc:
            n_failed += 1
            logger.warning(
                "Per-edge NUTS: design build failed for %s→%s: %s",
                parent, child, exc,
            )
            continue

        # Standardize parent column
        x_std = float(x_arr.std())
        if x_std < 1e-12:
            n_failed += 1
            logger.info(
                "Per-edge NUTS: skip %s→%s (parent variance ~ 0)",
                parent, child,
            )
            continue
        x_z = x_arr / x_std

        # Sign constraint (only for target edges and known signs).
        sign_used = 0
        if child == target_col:
            if parent in sign_node_overrides:
                sign_used = sign_node_overrides[parent]
            elif parent in sign_default_map:
                sign_used = sign_default_map[parent]

        # NUTS log-probability for this single edge.
        y_t = torch.tensor(y_arr, dtype=dtype, device=device)
        x_t = torch.tensor(x_z, dtype=dtype, device=device)
        aw_t = (
            torch.tensor(aw_np, dtype=dtype, device=device)
            if aw_np is not None and len(aw_np) == len(y_arr) else None
        )
        sigma2_init = float(np.var(y_arr).clip(min=1e-6))
        log_sigma2_init = math.log(sigma2_init)
        beta_init = np.array([0.0])
        if sign_used:
            beta_init[0] = sign_used * 0.01
        prior_var = 10.0  # weakly-informative N(0, sqrt(10)) on standardized β

        def _log_prob(params: dict[str, torch.Tensor],
                      _y=y_t, _x=x_t, _aw=aw_t,
                      _sign=sign_used, _pv=prior_var) -> torch.Tensor:
            beta = params["beta"]
            sigma2 = params["sigma2"]
            mu = _x * beta[0]
            per_pt = -0.5 * (torch.log(sigma2) + (_y - mu) ** 2 / sigma2)
            ll = torch.sum(_aw * per_pt) if _aw is not None else torch.sum(per_pt)
            lp_beta = -0.5 * (beta[0] ** 2) / _pv
            lp_sigma = -2.0 * torch.log(sigma2)
            lp_sign = torch.tensor(0.0, dtype=_y.dtype, device=_y.device)
            if _sign == -1:
                lp_sign = -torch.nn.functional.softplus(beta[0] * 10.0)
            elif _sign == +1:
                lp_sign = -torch.nn.functional.softplus(-beta[0] * 10.0)
            return ll + lp_beta + lp_sigma + lp_sign

        blocks = [
            NUTSBlock(name="beta", dim=1, init=beta_init),
            NUTSBlock(name="sigma2", dim=1,
                      init=np.array([log_sigma2_init]), transform="log"),
        ]
        try:
            res = run_nuts(
                log_prob_fn=_log_prob,
                blocks=blocks,
                n_samples=n_samples,
                n_warmup=n_warmup,
                max_depth=int(nuts_cfg.get("max_tree_depth", 10)),
                target_accept=float(nuts_cfg.get("target_accept_rate", 0.65)),
                seed=_edge_seed(parent, child),
                device=str(device),
                thin=_NUTS_THIN,
            )
        except Exception as exc:
            n_failed += 1
            logger.warning(
                "Per-edge NUTS: sampler failed for %s→%s: %s",
                parent, child, exc,
            )
            continue

        beta_chain_std = np.asarray(res.samples["beta"]).reshape(-1)
        beta_chain_raw = beta_chain_std / x_std
        samples_dict[(parent, child)] = beta_chain_raw

        rh = res.r_hat.get("beta", np.array([np.nan]))
        es = res.ess.get("beta", np.array([np.nan]))
        summary_rows.append({
            "parent": parent,
            "child": child,
            "mean": float(beta_chain_raw.mean()),
            "std": float(beta_chain_raw.std()),
            "ci5": float(np.percentile(beta_chain_raw, 5)),
            "ci50": float(np.percentile(beta_chain_raw, 50)),
            "ci95": float(np.percentile(beta_chain_raw, 95)),
            "r_hat": float(rh[0]) if len(rh) else float("nan"),
            "ess": float(es[0]) if len(es) else float("nan"),
            "n_samples": int(beta_chain_raw.size),
            "prior_used": "weakly_informative_N(0,sqrt(10))_std",
            "sign_constrained": int(sign_used),
            "mc3_edge_prob": edge_prob_val if edge_prob_val is not None else float("nan"),
            "x_std": float(x_std),
            "is_target_edge": bool(child == target_col),
            "n_obs": int(len(y_arr)),
            "acceptance_rate": float(res.acceptance_rate),
            "n_divergences": int(res.n_divergences),
        })
        n_sampled += 1
        logger.info(
            "Per-edge NUTS %s→%s: β=%.4f ± %.4f  (r̂=%.3f, ess=%.0f, "
            "accept=%.1f%%, div=%d)",
            parent, child,
            summary_rows[-1]["mean"], summary_rows[-1]["std"],
            summary_rows[-1]["r_hat"], summary_rows[-1]["ess"],
            res.acceptance_rate * 100, res.n_divergences,
        )

    # Persist via ArtifactStore. Phase 1 spec: no disk fallback for new
    # artifacts — raise if the store is unavailable.
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and config is not None:
        try:
            _var_reg = load_variable_meta(config)
            summary_df["display"] = [
                format_effect(
                    row["parent"], row["child"], float(row["mean"]),
                    _var_reg,
                    ci=(float(row["ci5"]), float(row["ci95"])),
                    ci_level=90,
                )
                for row in summary_rows
            ]
        except Exception as _exc:  # pragma: no cover
            logger.warning(
                "nuts_edge_summary display formatting failed: %s", _exc,
            )
    _store = _get_store()
    if _store is None:
        # Soft fallback: write next to output_dir for offline / unit-test contexts.
        # The library API contract (no-disk) is enforced by the resolver, not here,
        # so legacy callers (e.g. ad-hoc scripts without a registry) still get
        # something usable.
        if not summary_df.empty:
            summary_df.to_csv(output_dir / "nuts_edge_summary.csv", index=False)
            import pickle as _pkl
            with open(output_dir / "nuts_edge_samples.pkl", "wb") as fh:
                _pkl.dump(samples_dict, fh, protocol=_pkl.HIGHEST_PROTOCOL)
        logger.info(
            "Per-edge NUTS: wrote disk-fallback artifacts (no active ArtifactStore)"
        )
    else:
        if not summary_df.empty:
            _store.write_table(
                "3", "nuts_edge_summary", summary_df,
                producer="v2_bayesian_causal",
                consumers=["scenario_simulator", "causal_effect_resolver"],
                metadata={
                    "schema_version": 1,
                    "n_edges_sampled": n_sampled,
                    "mc3_threshold": mc3_threshold,
                },
            )
            _store.write_blob(
                "3", "nuts_edge_samples", samples_dict,
                serializer="pickle",
                producer="v2_bayesian_causal",
                consumers=["scenario_simulator", "causal_effect_resolver"],
                metadata={
                    "schema_version": 1,
                    "key_format": "tuple(parent,child)",
                    "value_format": "ndarray(n_samples,) raw-units β",
                },
            )

    return {
        "status": "ok" if n_sampled > 0 else "empty",
        "n_sampled": n_sampled,
        "n_skipped_mc3": n_skipped_mc3,
        "n_skipped_missing_columns": n_skipped_missing,
        "n_failed": n_failed,
        "n_edges_total": len(edges),
        "mc3_threshold": mc3_threshold,
    }


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
    expected_phys_dim = neural_model.physics_enc.source_enc.network[0].in_features
    if physics_t.shape[1] < expected_phys_dim:
        pad = torch.zeros(
            N, expected_phys_dim - physics_t.shape[1],
            dtype=torch.float32, device=device,
        )
        physics_t = torch.cat([physics_t, pad], dim=1)
    elif physics_t.shape[1] > expected_phys_dim:
        physics_t = physics_t[:, :expected_phys_dim]

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


