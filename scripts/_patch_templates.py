"""
Patch all non-UHI domain templates with missing V2 sections.

Run from repo root:
    python scripts/_patch_templates.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Shared tail block appended to every non-UHI template after `caps:` section
# ---------------------------------------------------------------------------
TAIL_BLOCK = """
# ---------------------------------------------------------------------------
# Pipeline Globals (additional V2 keys)
# ---------------------------------------------------------------------------

flags:
  use_gwen_selection: true
  include_gwrf_in_ensemble: true
  use_laplacian_eigenmaps_in_ols: false

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
models:

  gwr:
    bandwidth: null          # null = auto-select via correlogram
    kernel: "gaussian"
    alpha: 0.1
    min_points: 50

  gwrf:
    n_estimators: 100
    k_neighbors: 100
    min_samples_leaf: 5
    n_jobs: 1
    subsample_fraction: null
    subsample_n: null

  ggpgam:
    n_splines: 5
    n_spatial_bases: 10
    lam: 0.6
    max_iter: 100

  meta_ensemble:
    algorithm: "neural"
    include_base_features: true
    include_laplacian_pca: true

  meta_learner: "neural"

  neural:
    hidden_dim: 256
    dropout: 0.1
    n_heads: 4
    max_neighbors: 128
    siren_omega: 30.0
    exceedance_thresholds: [0.25, 0.50, 0.75]
    geo_pe_dim: 32             # Geographic positional encoding dim (Space2Vec Fourier; 0 = disabled)

  spatial_cv:
    block_size: null           # null = auto from Stage 1 correlogram
    buffer_size: 300           # metres
    block_size_source: "correlogram"
    method: "block"
    stratify_y: true

# ---------------------------------------------------------------------------
# GWEN Variable Selection
# ---------------------------------------------------------------------------
gwen:
  sample_size: 5000
  k_neighbors: 500
  cv_folds: 5
  selection_threshold: 0.1
  l1_ratios: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
  n_alphas: 100
  spatial_cv: false
  local_cv: false
  stability_folds: 0

# ---------------------------------------------------------------------------
# Laplacian Eigenmaps
# ---------------------------------------------------------------------------
laplacian:
  n_eigenmaps: 150
  k_for_swm: 10

# ---------------------------------------------------------------------------
# V2 Training Configuration
# ---------------------------------------------------------------------------
training:
  n_epochs: 100
  batch_size: 512
  learning_rate: 0.001
  lambda_mse: 1.0
  lambda_exceedance: 0.1
  lambda_physics: 0.01
  lambda_smooth_pred: 0.01
  lambda_smooth_alpha: 0.001
  lambda_alpha_prior: 1.0
  lambda_surrogate: 0.1
  lambda_neighbor: 0.1
  lambda_pde: 0.05             # V3 multi-term PDE loss (0 = disabled)

# ---------------------------------------------------------------------------
# V2 Optimization
# ---------------------------------------------------------------------------
optimization:
  run_cma_es: false
  clip_norm: 1.0
  swa_epochs: 20
  capacity_sweep: false

# ---------------------------------------------------------------------------
# JEPA — Self-supervised auxiliary objective on the SharedTrunk
# ---------------------------------------------------------------------------
# V-JEPA 2-style: EMA target trunk + masked-channel context + latent
# predictor with VICReg anti-collapse.  Set enable: true once you have
# validated the supervised pipeline for this domain.
#
# Phase 2 weights (lambda_scenario, lambda_latent_pde) are 0 by default —
# enable Phase 1 first, then raise these once it trains stably.
# ---------------------------------------------------------------------------
jepa:
  enable: false                # Set true to activate self-supervised pretraining
  pretrain_epochs: 20          # Phase 1 EMA-trunk pretraining epochs (0 = skip)
  mask_ratio: 0.4
  lambda: 1.0

  ema_tau_start: 0.99
  ema_tau_end:   0.9999
  ema_warmup_steps: 1000

  curriculum_start: 5
  curriculum_end:   15

  lambda_align:      1.0
  lambda_variance:   1.0
  lambda_covariance: 0.04
  variance_gamma:    1.0

  action_dim:        64
  predictor_blocks:  2
  predictor_film:    false     # Enable after Phase 1 is stable
  lambda_scenario:   0.0       # Phase 2: action-conditioned distillation (0 = off)
  lambda_latent_pde: 0.0       # Phase 2: latent Laplacian smoothness (0 = off)
  scenario_perturb_std: 0.5

# ---------------------------------------------------------------------------
# V2 Process Rate Network  (domain-specific — uncomment and customise)
# ---------------------------------------------------------------------------
# process_rate:
#   enabled: false
#   name: "domain_process_rate"   # e.g. "fire_spread_rate", "runoff_coefficient"
#   units: "domain units"
#   bounds: [0.0, 1.0]
#   prior_mean: 0.5
#   inputs: []                    # predictor columns that drive the process rate

# ---------------------------------------------------------------------------
# Hardware Performance Override  (optional)
# ---------------------------------------------------------------------------
# performance:
#   hardware_tier_override: null  # null = auto-detect; "standard" | "high"
#   preset: null                  # null = auto; "balanced" | "max"
#   cuda_graphs: false
#   force_cpu: false

# ---------------------------------------------------------------------------
# V3 Transfer / Continual Learning Registry  (optional)
# ---------------------------------------------------------------------------
# registry:
#   path: "sparc_registry"
#   city_name: ""               # Unique identifier for this study area
#   use_global_trunk: false
#   ewc_lambda: 0.0
#   replay_lambda: 0.0
#   coreset_size: 400
"""

# ---------------------------------------------------------------------------
# Patches to existing sections (old text → new text)
# ---------------------------------------------------------------------------

PIPELINE_OLD = """pipeline:
  random_seed: 42
  n_spatial_folds: 5"""

PIPELINE_NEW = """pipeline:
  random_seed: 42
  n_spatial_folds: 5
  fast_mode: false
  overwrite_outputs: false
  run_mc_uncertainty: true     # Monte-Carlo uncertainty propagation in Stage 4
  n_mc_draws: 50               # Number of MC draws
  scenario_mode: "auto"        # auto | hybrid | dag_coefficient | model_reprediction"""

OUTPUT_OLD = """output:
  base_dir: "output\""""

OUTPUT_NEW = """output:
  base_dir: "output"
  stage_dirs:
    stage_0: "Stage_0_Correlogram"
    stage_1: "Stage_1_GWEN"
    stage_2: "Stage_2_Spatial_CV"
    stage_3: "Stage_3_Causal_Validation"
    stage_4: "Stage_4_Scenarios"
    final:   "Final_Interpretation_Results"
"""

# NOTE: output_old has a trailing quote included; the replacement adds stage_dirs
# but also drops the stray trailing quote from `"output"` to just `"output"`.

CAUSAL_MC3_OLD = """  mc3:
    n_iterations: 10000
    n_chains: 4
    burnin_fraction: 0.25
    edge_penalty: 1.0
    seed: 42"""

CAUSAL_MC3_NEW = """  inference_backend: "mc3"   # "mc3" | "dibs" | "order_mcmc"
  mc3:
    n_iterations: 50000        # max cap; converges early via moving-window check
    min_iterations: 10000      # floor before convergence check activates
    converge_tol: 0.005        # stop when max edge-prob change < 0.5% over window
    converge_window: 2000      # window size (post-burnin samples)
    n_chains: 4
    temperatures: [1.0, 0.75, 0.55, 0.40]
    burnin_fraction: 0.25
    edge_penalty: 1.0
    seed: 42
    warm_start:
      enabled: true
      n_particles: 4
      n_steps: 500
      lambda_h: 1.0"""

CAUSAL_NUTS_OLD = """  nuts:
    n_samples: 2000
    n_warmup: 500
    n_chains: 2
    target_accept_rate: 0.85
    max_tree_depth: 8"""

CAUSAL_NUTS_NEW = """  nuts:
    n_samples: 2000
    n_warmup: 1000             # ≥1000 for good mass-matrix adaptation
    per_edge_n_samples: 2000
    per_edge_n_warmup: 800
    n_chains: 2
    target_accept_rate: 0.90   # higher = smaller step size = better mixing
    max_tree_depth: 10"""

COLLECT_INSERT_AFTER_PROJECT = """
# ---------------------------------------------------------------------------
# Data Collection (sparc collect)
# ---------------------------------------------------------------------------
collect:
  grid_resolution_m: 30        # Analysis grid cell size in metres

"""

# Domains that already have a collect: block — skip the collect insert
DOMAINS_WITH_COLLECT = {"uhi"}


def patch_template(path: Path) -> bool:
    """Apply all patches to a single project.yml.  Returns True if changed."""
    original = path.read_text(encoding="utf-8")
    text = original

    # 1. Expand pipeline block
    if PIPELINE_OLD in text:
        text = text.replace(PIPELINE_OLD, PIPELINE_NEW, 1)

    # 2. Expand output block — normalise "output" string value then add stage_dirs
    if 'output:\n  base_dir: "output"' in text and "stage_dirs:" not in text:
        text = text.replace(
            'output:\n  base_dir: "output"',
            OUTPUT_NEW.rstrip(),
            1,
        )

    # 3. Add inference_backend + advanced mc3 params
    if CAUSAL_MC3_OLD in text:
        text = text.replace(CAUSAL_MC3_OLD, CAUSAL_MC3_NEW, 1)

    # 4. Improve nuts params
    if CAUSAL_NUTS_OLD in text:
        text = text.replace(CAUSAL_NUTS_OLD, CAUSAL_NUTS_NEW, 1)

    # 5. Insert collect block after project: block (before data:)
    domain = path.parent.name
    if domain not in DOMAINS_WITH_COLLECT and "collect:" not in text:
        text = re.sub(
            r'(^data:\n)',
            COLLECT_INSERT_AFTER_PROJECT + r'\1',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    # 6. Append tail block if not already present
    if "jepa:" not in text:
        text = text.rstrip() + "\n" + TAIL_BLOCK

    if text == original:
        print(f"  (unchanged) {path}")
        return False

    path.write_text(text, encoding="utf-8")
    print(f"  patched     {path}")
    return True


def main():
    domains = [
        "air_quality", "blank", "coastal", "drought", "geotechnical",
        "groundwater", "noise", "seismic", "stormwater", "water_quality", "wildfire",
    ]
    template_roots = [
        REPO_ROOT / "templates",
        REPO_ROOT / "sparc" / "templates",
    ]

    changed = 0
    for root in template_roots:
        for domain in domains:
            yml = root / domain / "project.yml"
            if not yml.exists():
                print(f"  SKIP (not found): {yml}")
                continue
            if patch_template(yml):
                changed += 1

    print(f"\nDone. {changed} files updated.")


if __name__ == "__main__":
    main()
