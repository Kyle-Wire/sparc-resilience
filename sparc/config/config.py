# sparc_core/config/config.py
"""
Central configuration for the SPARC pipeline.

Two loading modes:
  1. **Project YAML** (recommended for new projects):
       config = load_config("path/to/project.yml")
     Reads a validated YAML project file and translates it into the internal
     config dictionary consumed by every pipeline stage.

  2. **Legacy fallback** (no arguments):
       config = load_config()
     Returns the original hard-coded Brown-UHI defaults so that existing
     scripts continue to work without modification.

The returned dictionary has the same shape in both modes so downstream code
never needs to know which path was taken.
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Legacy constants (preserved for backward compatibility)
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
SPARC_PKG_DIR = os.path.dirname(CONFIG_DIR)             # sparc/ package root
PROJECT_ROOT = os.path.dirname(SPARC_PKG_DIR)            # repo root
SPARC_CORE_DIR = PROJECT_ROOT                             # alias for legacy code
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "brown4.csv")

TARGET_VARIABLE = 'AAT_z'
IDENTIFIER_COL = 'OBJECTID'
COORDS_COLS = ['POINT_X', 'POINT_Y']

INITIAL_CRS = "EPSG:3438"
TARGET_PROJECTED_CRS = "EPSG:26919"

BASE_MODEL_PREDICTORS = [
    'Distance_from_water_m',
    'Pct_Impervious',
    'Pct_Canopy',
    'NDVI',
    'Albedo',
    'Elevation_m'
]

GWEN_PREDICTORS = list(BASE_MODEL_PREDICTORS)

INCLUDE_GWRF_IN_ENSEMBLE = True
OVERWRITE_EXISTING_OUTPUTS = False
USE_LAPLACIAN_EIGENMAPS_IN_OLS = False
USE_GWEN_SELECTION = True
N_SPATIAL_FOLDS = 5
K_FOR_MORAN_SWM = 100
N_LAPLACIAN_EIGENMAPS = 150
K_FOR_LAPLACIAN_SWM = 10
GWEN_SAMPLE_SIZE = 5000
GWEN_K_NEIGHBORS = 500
GWEN_CV_FOLDS = 5
GWEN_SELECTION_THRESHOLD = 0.1
GWEN_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
GWEN_N_ALPHAS = 100
N_OPTUNA_TRIALS_LGBM = 25

# ---------------------------------------------------------------------------
# YAML project-file loader
# ---------------------------------------------------------------------------

def _resolve_path(base_dir: str, candidate: str) -> str:
    """Resolve *candidate* relative to *base_dir* unless it is already absolute."""
    p = Path(candidate)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return the parsed dict."""
    import yaml  # lazy import — only needed when a project.yml is used
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def _validate_project_yaml(raw: dict, yaml_path: str) -> None:
    """
    Validate *raw* against the JSON Schema shipped with the package.
    Raises ``jsonschema.ValidationError`` on invalid input.
    """
    try:
        import jsonschema
    except ImportError:
        # If jsonschema is not available, skip validation with a warning
        import warnings
        warnings.warn("jsonschema not installed — skipping project.yml validation", stacklevel=2)
        return

    schema_path = Path(__file__).parent / "project_schema.json"
    if not schema_path.exists():
        return  # schema not shipped — skip
    try:
        with open(schema_path, 'r', encoding='utf-8') as fh:
            schema = json.load(fh)
    except (PermissionError, OSError):
        # Inside a PyInstaller _MEI bundle the extracted file may not be
        # readable on some Windows configurations — skip validation.
        return
    jsonschema.validate(instance=raw, schema=schema)


def _yaml_to_config(raw: dict, yaml_path: str) -> dict:
    """
    Translate a validated project YAML dict into the internal config shape
    that every pipeline module already expects.
    """
    base_dir = str(Path(yaml_path).resolve().parent)

    # Resolve data file path
    data_file = _resolve_path(base_dir, raw['data']['file_path'])

    # Resolve output directory
    out_section = raw.get('output', {})
    output_base = _resolve_path(base_dir, out_section.get('base_dir', 'output'))

    predictors = list(raw['predictors'])
    pipeline = raw.get('pipeline', {})
    flags = raw.get('flags', {})
    gwen_cfg = raw.get('gwen', {})
    laplacian_cfg = raw.get('laplacian', {})
    models_cfg = raw.get('models', {})
    random_seed = pipeline.get('random_seed', 42)

    config = {
        # ---- paths (backward-compat keys) ----
        'paths': {
            'config_dir': CONFIG_DIR,
            'sparc_core_dir': SPARC_CORE_DIR,
            'project_root': base_dir,
            'output_dir': output_base,
            'raw_csv_path': data_file,
        },
        'data': {
            'file_path': data_file,
            'target_column': raw['data']['target_column'],
            'identifier_column': raw['data'].get('identifier_column', 'OBJECTID'),
            'coord_columns': list(raw['data']['coord_columns']),
        },
        'output': {
            'base_dir': output_base,
            'stage_dirs': out_section.get('stage_dirs', {
                'stage_0': 'Stage_0_Correlogram',
                'stage_1': 'Stage_1_GWEN',
                'stage_2': 'Stage_2_Spatial_CV',
                'stage_3': 'Stage_3_Causal_Validation',
                'stage_4': 'Stage_4_Scenarios',
                'final':   'Final_Interpretation_Results',
            }),
        },
        'variables': {
            'target': raw['data']['target_column'],
            'identifier': raw['data'].get('identifier_column', 'OBJECTID'),
            'coordinates': list(raw['data']['coord_columns']),
        },
        'crs': {
            'initial': raw['crs']['input'],
            'target_projected': raw['crs']['projected'],
        },
        'predictors': {
            'base_model': predictors,
            'gwen': predictors,  # GWEN uses the same predictor set
        },
        'flags': {
            'include_gwrf_in_ensemble': flags.get('include_gwrf_in_ensemble', True),
            'overwrite_existing_outputs': pipeline.get('overwrite_outputs', False),
            'use_laplacian_eigenmaps_in_ols': flags.get('use_laplacian_eigenmaps_in_ols', False),
            'use_gwen_selection': flags.get('use_gwen_selection', True),
        },
        'cv_params': {
            'n_spatial_folds': pipeline.get('n_spatial_folds', 5),
            'k_for_moran_swm': K_FOR_MORAN_SWM,
        },
        'laplacian_params': {
            'n_eigenmaps': laplacian_cfg.get('n_eigenmaps', 150),
            'k_for_swm': laplacian_cfg.get('k_for_swm', 10),
        },
        'tsne_params': {
            'perplexity_values': [30, 50, 100],
            'n_components_list': [2, 3],
            'n_iter': 1000,
            'learning_rate': 'auto',
            'init': 'pca',
            'random_state': random_seed,
        },
        'gwen_params': {
            'sample_size': gwen_cfg.get('sample_size', 5000),
            'k_neighbors': gwen_cfg.get('k_neighbors', 500),
            'cv_folds': gwen_cfg.get('cv_folds', 5),
            'selection_threshold': gwen_cfg.get('selection_threshold', 0.1),
            'l1_ratios': gwen_cfg.get('l1_ratios', [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]),
            'n_alphas': gwen_cfg.get('n_alphas', 100),
        },
        'lightgbm_params': {
            'n_optuna_trials': models_cfg.get('meta_ensemble', {}).get('n_optuna_trials', 25),
        },
        # ---- NEW: project-level metadata available to all stages ----
        'project': raw.get('project', {}),
        'physics': raw.get('physics', {}),
        'causal':  {'inference': 'bayesian', **raw.get('causal', {})},
        'models':  {**models_cfg,
            'meta_learner': models_cfg.get('meta_learner', 'neural'),
            'neural': models_cfg.get('neural', {
                'hidden_dim': 256,
                'dropout': 0.1,
                'n_heads': 4,
                'max_neighbors': 128,
                'siren_omega': 30.0,
                'sinusoidal_frequencies': 64,
                'exceedance_thresholds': [0.25, 0.50, 0.75],
                'mc_dropout_samples': 100,
            }),
        },
        'scenarios': raw.get('scenarios', []),
        'joint_scenarios': raw.get('joint_scenarios', []),
        'interaction_scenarios': raw.get('interaction_scenarios', []),
        'pipeline': pipeline,
        'temporal': raw.get('temporal', {'enabled': False}),
        'benchmark_metrics': raw.get('benchmark_metrics', {}),
        'fingerprint': raw.get('fingerprint', {}),
        # ---- V2: neural training, process rate, optimization ----
        'process_rate': raw.get('process_rate', {
            'enabled': True,
            'name': 'spatial_responsiveness',
            'units': 'dimensionless',
            'bounds': [1.0e-7, 9.0e-7],
            'prior_mean': 5.0e-7,
        }),
        'optimization': raw.get('optimization', {
            'run_cma_es': False,
            'cma_es_popsize': 20,
            'cma_es_maxiter': 50,
            'clip_norm': 1.0,
            'swa_epochs': 20,
            'capacity_sweep': True,
            'capacity_sweep_dims': [64, 128, 256, 512, 1024],
        }),
        'training': raw.get('training', {
            'n_epochs': 100,
            'batch_size': 2048,
            'warmup_epochs': 10,
            'ramp_epochs': 30,
            'lambda_physics': 0.1,
            'lambda_smooth': 0.01,
            'lambda_alpha_smooth': 0.01,
            'lambda_prior': 0.01,
            'lambda_base': 0.2,
            'lambda_neighbor': 0.05,
        }),
    }

    # Resolve physics file paths relative to the YAML location
    physics = config['physics']
    if physics.get('priors_file'):
        physics['priors_file'] = _resolve_path(base_dir, physics['priors_file'])
    if physics.get('caps_file'):
        physics['caps_file'] = _resolve_path(base_dir, physics['caps_file'])

    # Resolve causal DAG file path
    causal = config['causal']
    if causal.get('dag_file'):
        causal['dag_file'] = _resolve_path(base_dir, causal['dag_file'])

    # Resolve areas-of-interest file
    if raw['data'].get('areas_of_interest_file'):
        config['data']['areas_of_interest_file'] = _resolve_path(
            base_dir, raw['data']['areas_of_interest_file']
        )

    return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> dict:
    """
    Load pipeline configuration.

    Parameters
    ----------
    config_path : str or None
        • **Path to a ``project.yml``**: loads and validates the YAML, then
          translates it into the internal config dict.
        • **None** (default): returns the legacy hard-coded Brown-UHI
          configuration for backward compatibility.

    Returns
    -------
    dict
        Configuration dictionary consumed by every pipeline module.
    """
    # --- YAML project file mode ---
    # Check the SPARC_PROJECT env var if no explicit path was given.
    # The CLI sets this so that downstream load_config() calls (which pass
    # no arguments) automatically pick up the project.yml.
    if config_path is None:
        config_path = os.environ.get('SPARC_PROJECT')

    if config_path is not None:
        p = Path(config_path)
        if p.suffix in ('.yml', '.yaml') and p.exists():
            raw = _load_yaml(str(p))
            _validate_project_yaml(raw, str(p))
            return _yaml_to_config(raw, str(p))
        elif not p.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

    # --- Legacy fallback (no project.yml) ---
    import warnings
    warnings.warn(
        "No project.yml provided. Falling back to hard-coded Brown-UHI defaults. "
        "Create a project.yml with `sparc init` or pass --project <path>.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {
        'paths': {
            'config_dir': CONFIG_DIR,
            'sparc_core_dir': SPARC_CORE_DIR,
            'project_root': PROJECT_ROOT,
            'output_dir': OUTPUT_DIR,
            'raw_csv_path': RAW_CSV_PATH
        },
        'data': {
            'file_path': RAW_CSV_PATH,
            'target_column': TARGET_VARIABLE,
            'identifier_column': IDENTIFIER_COL,
            'coord_columns': COORDS_COLS
        },
        'output': {
            'base_dir': OUTPUT_DIR
        },
        'variables': {
            'target': TARGET_VARIABLE,
            'identifier': IDENTIFIER_COL,
            'coordinates': COORDS_COLS
        },
        'crs': {
            'initial': INITIAL_CRS,
            'target_projected': TARGET_PROJECTED_CRS
        },
        'predictors': {
            'base_model': BASE_MODEL_PREDICTORS,
            'gwen': GWEN_PREDICTORS
        },
        'flags': {
            'include_gwrf_in_ensemble': INCLUDE_GWRF_IN_ENSEMBLE,
            'overwrite_existing_outputs': OVERWRITE_EXISTING_OUTPUTS,
            'use_laplacian_eigenmaps_in_ols': USE_LAPLACIAN_EIGENMAPS_IN_OLS,
            'use_gwen_selection': USE_GWEN_SELECTION
        },
        'cv_params': {
            'n_spatial_folds': N_SPATIAL_FOLDS,
            'k_for_moran_swm': K_FOR_MORAN_SWM
        },
        'laplacian_params': {
            'n_eigenmaps': N_LAPLACIAN_EIGENMAPS,
            'k_for_swm': K_FOR_LAPLACIAN_SWM
        },
        'tsne_params': {
            'perplexity_values': [30, 50, 100],
            'n_components_list': [2, 3],
            'n_iter': 1000,
            'learning_rate': 'auto',
            'init': 'pca',
            'random_state': 42
        },
        'gwen_params': {
            'sample_size': GWEN_SAMPLE_SIZE,
            'k_neighbors': GWEN_K_NEIGHBORS,
            'cv_folds': GWEN_CV_FOLDS,
            'selection_threshold': GWEN_SELECTION_THRESHOLD,
            'l1_ratios': GWEN_L1_RATIOS,
            'n_alphas': GWEN_N_ALPHAS
        },
        'lightgbm_params': {
            'n_optuna_trials': N_OPTUNA_TRIALS_LGBM
        },
        # Empty new-style keys so downstream code can safely `.get()` them
        'project': {},
        'physics': {},
        'causal': {'inference': 'bayesian'},
        'models': {
            'meta_learner': 'neural',
            'neural': {
                'hidden_dim': 256,
                'dropout': 0.1,
                'n_heads': 4,
                'max_neighbors': 128,
                'siren_omega': 30.0,
                'sinusoidal_frequencies': 64,
                'exceedance_thresholds': [0.25, 0.50, 0.75],
                'mc_dropout_samples': 100,
            },
        },
        'scenarios': [],
        'pipeline': {'random_seed': 42},
        'temporal': {'enabled': False},
        # V2 keys with V1-compatible defaults (disabled)
        'process_rate': {'enabled': False},
        'optimization': {'run_cma_es': False, 'capacity_sweep': False},
        'training': {},
    }


# ---------------------------------------------------------------------------
# Physics / monotone-constraint helpers
# ---------------------------------------------------------------------------

def load_monotone_constraints(config: dict) -> dict:
    """
    Return the monotone-constraint dict from the config.

    Checks ``config['physics']['monotone_constraints']`` first (project.yml
    mode).  Falls back to the legacy hard-coded UHI constraints if absent.

    Returns
    -------
    dict
        Mapping of predictor name → ``{-1, 0, 1}``.
    """
    constraints = config.get('physics', {}).get('monotone_constraints')
    if constraints:
        return dict(constraints)

    # Legacy fallback
    return {
        'Pct_Canopy': -1,
        'Pct_Impervious': 1,
        'NDVI': -1,
        'Albedo': -1,
        'Elevation_m': 0,
        'Distance_from_water_m': 0,
    }


def load_physics_priors(config: dict) -> dict:
    """
    Load physics priors from the YAML file referenced by
    ``config['physics']['priors_file']``.

    Returns an empty dict if no priors file is configured.
    """
    priors_path = config.get('physics', {}).get('priors_file')
    if priors_path and os.path.exists(priors_path):
        return _load_yaml(priors_path)
    return {}


def load_caps(config: dict) -> dict:
    """
    Load domain caps/constraints from the YAML file referenced by
    ``config['physics']['caps_file']``.

    Returns an empty dict if no caps file is configured.
    """
    caps_path = config.get('physics', {}).get('caps_file')
    if caps_path and os.path.exists(caps_path):
        return _load_yaml(caps_path)
    return {}


def validate_block_size_for_models(candidate_block_sizes: list[float], model_bandwidths: dict[str, float | None]) -> dict:
    """
    Validate candidate block sizes against model bandwidth requirements
    
    Parameters:
    -----------
    candidate_block_sizes : list
        List of candidate block sizes from correlogram analysis
    model_bandwidths : dict
        Dictionary mapping model names to their bandwidth values
        
    Returns:
    --------
    dict : Validation results for each block size
    """
    validation_results = {}
    
    for block_size in candidate_block_sizes:
        validation_results[block_size] = {}
        
        for model, bandwidth in model_bandwidths.items():
            if bandwidth is None:  # Global models like OLS
                validation_results[block_size][model] = "OK (global model)"
            else:
                # Rule of thumb: block size should be at least 2-3 times the bandwidth
                # to ensure adequate spatial separation for cross-validation
                min_required = bandwidth * 2.5
                if block_size >= min_required:
                    validation_results[block_size][model] = "OK"
                else:
                    validation_results[block_size][model] = f"Too small (need >= {min_required:.0f}m)"
    
    return validation_results


def get_recommended_block_size(model_bandwidths: dict[str, float | None]) -> float:
    """
    Get recommended minimum block size based on model bandwidths
    
    Parameters:
    -----------
    model_bandwidths : dict
        Dictionary mapping model names to their bandwidth values
        
    Returns:
    --------
    float : Recommended minimum block size in meters
    """
    # Filter out global models (None bandwidths)
    spatial_bandwidths = [bw for bw in model_bandwidths.values() if bw is not None]
    
    if not spatial_bandwidths:
        # If no spatial models, use a conservative default
        return 5000.0  # 5km default
    
    # Use the maximum bandwidth * 2.5 as minimum safe block size
    max_bandwidth = max(spatial_bandwidths)
    recommended_size = max_bandwidth * 2.5
    
    # Ensure a reasonable minimum (at least 1km)
    return max(recommended_size, 1000.0)
