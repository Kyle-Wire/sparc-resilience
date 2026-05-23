#!/usr/bin/env python3
"""
Enhanced Spatial Cross-Validation with Variable-Specific Optimization
Stage 2 of SPARC Pipeline - Uses variogram-optimized parameters
"""

import os
import sys
import warnings

# Suppress sklearn parallel warnings BEFORE any sklearn imports
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning,ignore::RuntimeWarning'

import json
import math
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from sparc.config.hardware_profile import detect_profile
from sparc.run.pipeline_paths import get_paths
from sparc.run.spatial_autocorr_comprehensive import SpatialAutocorrelationAnalyzer
from sparc.run.memory_efficient_spatial_analysis import analyze_model_residuals_morans_i
from sparc.run.disk_policy import disk_writes_enabled
from sparc.run.artifact_io import (
    save_blob_path, load_blob_path, save_struct_path,
    save_table_path, save_geo_path, ensure_dir,
    load_table_path, exists_path,
)

# Hardware-tier driven configuration (low / standard / high RAM machines).
# On low-tier (e.g. 8GB MacBook Pro) parallel workers collapse to 1 and batch
# sizes shrink so the pipeline does not get OOM-killed by the OS.
#
# HARDWARE_CONFIG is rebuilt by `refresh_hardware_config()` whenever the user
# changes their performance settings. The pipeline `main()` calls it at the
# top of each run so per-project overrides take effect without restarting
# the server process.
HARDWARE_CONFIG: dict = {}
_PROFILE = None  # populated by refresh_hardware_config() below


def refresh_hardware_config(verbose: bool = True) -> dict:
    """Rebuild the module-level HARDWARE_CONFIG from the current hardware profile.

    Call from any entry point (e.g. pipeline ``main()``) after the user may
    have changed performance settings. Returns the updated config dict.
    """
    global _PROFILE, HARDWARE_CONFIG
    _PROFILE = detect_profile()
    low = _PROFILE.tier == "low"
    HARDWARE_CONFIG = {
        'cpu_cores': _PROFILE.cpu_count,
        'max_workers': _PROFILE.max_workers,
        'memory_limit_gb': _PROFILE.memory_limit_gb,
        'batch_size_large': _PROFILE.batch_size if low else 4096,
        'batch_size_medium': _PROFILE.batch_size if low else 2048,
        'batch_size_small': _PROFILE.batch_size if low else 1024,
        'parallel_cv': _PROFILE.max_workers > 1,
        'high_memory_mode': _PROFILE.high_memory_mode,
        'enable_aggressive_optimization': not low,
        'laplacian_batch_size': 500 if low else 2000,
        'max_eigen_iterations': 1000,
        'spatial_cluster_threshold': 1000,
        'per_model_parallel': _PROFILE.outer_jobs > 1,
        'outer_jobs': _PROFILE.outer_jobs,
        'inner_jobs': _PROFILE.inner_jobs,
        'gwrf_local_jobs': 1,
    }
    if verbose:
        print(f"Hardware-optimized configuration detected (CPU-only):")
        print(f"  - {_PROFILE.banner()}")
        print(f"  - CPU cores: {HARDWARE_CONFIG['cpu_cores']}")
        print(f"  - Available RAM: {_PROFILE.total_ram_gb:.1f} GB")
        print(f"  - Memory limit: {HARDWARE_CONFIG['memory_limit_gb']:.1f} GB")
        print(f"  - Parallel processing: {'Enabled' if HARDWARE_CONFIG['parallel_cv'] else 'Disabled (low-memory safety)'}")
        print(f"  - Per-model parallelism: {HARDWARE_CONFIG['outer_jobs']} folds x {HARDWARE_CONFIG['inner_jobs']} inner jobs")
        print(f"  - High-memory mode: {'Enabled' if HARDWARE_CONFIG['high_memory_mode'] else 'Disabled'}")
    return HARDWARE_CONFIG


# Initial population at import (uses any overrides already set on the resolver).
refresh_hardware_config(verbose=True)

# Spatial analysis imports
from libpysal.weights import DistanceBand
from esda.moran import Moran

# Centralized path management
from sparc.run.pipeline_paths import get_paths

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from sparc.config.config import load_config, load_monotone_constraints
from sparc.data.data_utils import load_and_preprocess_data
from sparc.models.ols import OLSModel
from sparc.models.gwr import GWRModel
from sparc.models.gwrf import GWRFModel
from sparc.models.ggpgam import GGPGAM_SVC
from sparc.models.gwr_config import GWRConfig
from sparc.models.gwrf_config import GWRFConfig
from sparc.models.ggpgam_config import GGPGAMConfig
from sparc.evaluation.evaluation import SpatialEvaluator
from copy import deepcopy

# Shared evaluator instance for benchmark metrics
_evaluator = SpatialEvaluator()

# Global flag for OOF extraction — enabled now that oof_extraction_hooks is implemented
EXTRACT_OOF_INTELLIGENCE = True

def train_single_model_fold_worker(args):
    """
    Worker for one fold of one model.
    Args tuple: (fold_idx, train_idx, test_idx, X, y, coords, model, model_name, feature_names)
    Returns: (fold_idx, test_idx, predictions, oof_intelligence) or (fold_idx, test_idx, predictions)
    """
    fold_idx, train_idx, test_idx, X, y, coords, model, model_name, feature_names = args
    # Copy model so processes don’t share state
    model_copy = deepcopy(model)
    # Slice out train/test
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y[train_idx]
    coords_tr, coords_te = coords[train_idx], coords[test_idx]
    
    # Runtime safety: delegate fold-size invariants to each model.
    # GWRModel.validate_for_fold clamps bandwidth/min_points;
    # GWRFModel.validate_for_fold clamps k_neighbors/subsample_n.
    try:
        n_train = len(train_idx)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        if hasattr(model_copy, 'validate_for_fold'):
            model_copy.validate_for_fold(n_train, n_features)
    except Exception as safety_e:
        print(f"WARNING: Safety check failed for {model_name} fold {fold_idx}: {safety_e}")
    
    # Fit
    try:
        if 'coords' in model_copy.fit.__code__.co_varnames:
            model_copy.fit(X_tr, y_tr, coords_tr)
        else:
            model_copy.fit(X_tr, y_tr)
        # Predict
        if 'coords' in model_copy.predict.__code__.co_varnames:
            preds = model_copy.predict(X_te, coords_te)
        else:
            preds = model_copy.predict(X_te)
        
        # Handle models that return (predictions, uncertainty) tuples (e.g., GWRF v2)
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # === OOF SPATIAL INTELLIGENCE EXTRACTION ===
        oof_intelligence = None
        if EXTRACT_OOF_INTELLIGENCE and model_name in ['gwr', 'gwrf', 'ggpgam']:
            try:
                from sparc.interventions.oof_extraction_hooks import extract_from_fitted_model
                
                oof_intelligence = extract_from_fitted_model(
                    model=model_copy,
                    model_name=model_name,
                    X_test=X_te,
                    coords_test=coords_te,
                    feature_names=feature_names,
                    test_idx=test_idx
                )
            except Exception as oof_e:
                print(f"WARNING: OOF extraction failed for {model_name} fold {fold_idx}: {oof_e}")
                oof_intelligence = None
        
        return fold_idx, test_idx, preds, oof_intelligence
    except Exception as e:
        print(f"ERROR: {model_name} Fold {fold_idx} failed: {e}")
        # Return NaN predictions as fallback
        return fold_idx, test_idx, np.full(len(test_idx), np.nan), None

# ---------------------------------------------------------------------------
# Spatial fold helpers — implementations live in spatial_fold_factory.py.
# Re-exported here for backward compatibility with existing importers.
# ---------------------------------------------------------------------------
from sparc.run.spatial_fold_factory import (  # noqa: E402
    calculate_fold_spatial_autocorr,
    _spatial_smooth,
    score_model_spatial_consistency,
    estimate_spatial_autocorrelation_range,
    spatial_kfold_enhanced,
    latent_guided_spatial_kfold,
    _vsba_fold_quality_score,
)


class EnhancedSpatialCV:
    """
    Enhanced Spatial Cross-Validation using variable-specific optimized parameters
    """
    
    def __init__(self, hw: dict | None = None):
        # Use centralized path management
        self.paths = get_paths()

        self.base_config = load_config()
        # Typed, validated config facade — fail fast on bad project.yml rather
        # than deep inside a fold.  All Stage 2 code should prefer ``self._cfg``
        # over ``self.base_config`` for config lookups.
        from sparc.run.stage_config import StageConfig
        self._cfg = StageConfig.from_dict(self.base_config)
        # Centralised physics constraints from project.yml (or legacy defaults)
        self._monotone_constraints = load_monotone_constraints(self.base_config)
        # Hardware config: accept an explicit dict (for tests / overrides), else
        # fall back to the module-level global populated at import time.
        self._hw = hw if hw is not None else HARDWARE_CONFIG

    def _load_prior_trunk(self):
        """
        Try to load the prior-city trunk checkpoint from CityRegistry for
        latent-space AoA scoring.  Returns None on single-city runs or
        when the registry / prior city are not configured.
        """
        registry_path = self._cfg.continual_registry_path
        prior_city = self._cfg.continual_prior_city
        if not registry_path or not prior_city:
            return None
        try:
            from sparc.registry.city_registry import CityRegistry
            from sparc.models.neural_meta import SPARCMetaLearner

            record = CityRegistry(registry_path).load_city(prior_city)
            if record.trunk_checkpoint is None:
                return None
            sd = record.trunk_checkpoint
            # Infer trunk architecture dims from state-dict weight shapes
            n_phys = sd["physics_enc.source_enc.net.0.weight"].shape[1]
            hidden_dim = sd["trunk_fusion.0.weight"].shape[0]
            trunk = SPARCMetaLearner(
                n_base_models=3,
                n_physics_features=n_phys,
                d_spatial=64,
                hidden_dim=hidden_dim,
            )
            trunk.load_state_dict(sd, strict=False)
            trunk.eval()
            for p in trunk.parameters():
                p.requires_grad_(False)
            print(f"Loaded prior-city trunk from '{prior_city}' for latent AoA scoring")
            return trunk
        except Exception as exc:
            print(f"Could not load prior trunk for latent AoA (non-fatal): {exc}")
            return None

    def _load_correlogram_results(self):
        """Load correlogram_analysis_results.json — store-first, disk fallback."""
        if hasattr(self, '_correlogram_cache'):
            return self._correlogram_cache
        corr_path = self.paths.correlogram_results
        try:
            from sparc.run.artifact_io import load_struct_path
            self._correlogram_cache = load_struct_path(
                corr_path, stage="0", artifact_id="correlogram_results",
            )
        except FileNotFoundError:
            self._correlogram_cache = None
        return self._correlogram_cache

    def get_block_size_from_config(self):
        """Get block_size (cached): user config > correlogram > ERM p10 > None."""
        if not hasattr(self, '_cv_block_size'):
            self._cv_block_size = self._compute_block_size()
        return self._cv_block_size

    def _compute_block_size(self):
        """
        Get the block size: user config > correlogram results > ERM self-supervised > None.
        """
        # 1. User-specified in models.spatial_cv.block_size (via UI or project.yml)
        block_size = self._cfg.block_size
        if block_size is not None:
            print(f"Block size from user config: {block_size}m")
            return float(block_size)

        # 2. Correlogram analysis results
        corr = self._load_correlogram_results()
        if corr:
            opt = corr.get('spatial_cv_configuration', {}).get('optimal_block_size')
            if opt is not None:
                print(f"Block size from correlogram analysis: {opt}m")
                return float(opt)

        # 3. Self-supervised: 10th-percentile of effective ranges from Stage 0 ERM
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None:
                erm = _store.read_struct("0", "effective_range_matrix")
                if erm is not None:
                    M = erm.get("range_matrix", [])
                    ranges = [v for row in M for v in row if v is not None and v > 0]
                    if ranges:
                        mer = float(np.percentile(ranges, 10))
                        mer = max(mer, 500.0)
                        print(f"Self-supervised block size from MER (p10 of effective ranges): {mer:.0f}m")
                        return mer
        except Exception:
            pass

        return None

    def get_buffer_size_from_config(self):
        """Get buffer_size (cached): user config > auto-computed from block_size."""
        if not hasattr(self, '_cv_buffer_size'):
            self._cv_buffer_size = self._compute_buffer_size()
        return self._cv_buffer_size

    def _compute_buffer_size(self):
        """
        Get the buffer size from project configuration.
        """
        # Auto-calculate buffer based on block size if enabled
        if self._cfg.buffer_size_auto:
            block_size = self.get_block_size_from_config()
            if block_size is not None:
                auto_buffer = max(100, int(block_size / 3))
                print(f"Auto-calculated buffer size: {auto_buffer}m (1/3 of block size {block_size}m)")
                return auto_buffer

        return self._cfg.buffer_size

    def get_variable_bandwidths(self):
        """Get per-variable bandwidths (cached): correlogram > manual_parameters > None."""
        if not hasattr(self, '_cv_variable_bandwidths'):
            self._cv_variable_bandwidths = self._compute_variable_bandwidths()
        return self._cv_variable_bandwidths

    def _compute_variable_bandwidths(self):
        """Delegate to the pure resolve_bandwidth() function."""
        from sparc.run.gwr_bandwidth import resolve_bandwidth

        result = resolve_bandwidth(
            config=self._cfg.raw,
            stage0_result=self._load_correlogram_results(),
            target_var=self._cfg.target,
        )
        if result:
            print(f"Loaded {len(result)} per-variable bandwidths")
        return result
    
    def get_kernel_field(self, predictor_names):
        """Phase 5a: assemble the matrix-aware ``KernelField`` for the
        target variable from Stage-0 artifacts (``correlogram_matern_fit``,
        ``correlogram_anisotropy``, ``effective_range_matrix``).

        Returns ``None`` when the artifacts are not available so legacy
        callers continue to work unchanged.
        """
        target_var = self._cfg.target
        if not target_var or not predictor_names:
            return None
        try:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is None:
                return None
        except Exception:
            return None

        def _safe_read(stage, aid):
            try:
                return store.read_struct(stage, aid)
            except Exception:
                return None

        matern = _safe_read("0", "correlogram_matern_fit")
        aniso = _safe_read("0", "correlogram_anisotropy")
        erm = _safe_read("0", "effective_range_matrix")
        if matern is None and aniso is None and erm is None:
            return None
        try:
            from sparc.models.kernel_field import KernelField
            kf = KernelField.from_artifacts(
                outcome_name=target_var,
                predictor_names=list(predictor_names),
                matern_artifact=matern,
                anisotropy_artifact=aniso,
                effective_range_matrix=erm,
            )
            print(f"KernelField assembled for outcome '{target_var}' "
                  f"with {len(kf.predictors)} predictors "
                  f"(matern={'yes' if matern else 'no'}, "
                  f"aniso={'yes' if aniso else 'no'}, "
                  f"erm={'yes' if erm else 'no'})")
            return kf
        except Exception as e:
            print(f"KernelField assembly failed (falling back to legacy): {e}")
            return None
    
    def apply_profiler_overrides(self, profiler_recommendations):
        """
        Overlay DatasetProfiler-recommended hyperparameters onto the
        base_config in-memory so that ``create_optimized_models()``
        picks them up automatically.

        Parameters
        ----------
        profiler_recommendations : dict
            Output of ``DatasetProfiler.recommend_parameters()``.  Keys are
            model names (``gwrf``, ``ggpgam``, etc.).
        """
        models_section = self.base_config.setdefault('models', {})
        for model_key, recs in profiler_recommendations.items():
            if model_key in ('correlogram', 'spatial_cv'):
                if model_key == 'spatial_cv' and 'spatial_cv' in models_section:
                    models_section['spatial_cv'].update(recs)
                continue
            if model_key in models_section:
                for k, v in recs.items():
                    if isinstance(v, dict) and isinstance(models_section[model_key].get(k), dict):
                        models_section[model_key][k].update(v)
                    else:
                        models_section[model_key][k] = v

        print("Applied DatasetProfiler adaptive overrides to project config.")

    def create_optimized_models(self, n_samples=None):
        """
        Create base models with global hyperparameters from pipeline configuration
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples in the dataset (for safety checks)
        """
        model_configs = {'gwr': self._cfg.gwr_params, 'gwrf': self._cfg.gwrf_params, 'ggpgam': self._cfg.ggpgam_params}

        models = []

        # Resolve shared per-model dependencies once, before any model branch.
        # Both GWR and GWRF need kernel_field; resolving here means the GWRF
        # branch can use `kernel_field` directly with no NameError risk.
        variable_bandwidths = self.get_variable_bandwidths()
        predictor_names_for_kf = list((variable_bandwidths or {}).keys())
        if not predictor_names_for_kf:
            predictor_names_for_kf = list(
                (self.base_config.get('manual_parameters', {}) or {}).get('bandwidths', {}).keys()
            )
        kernel_field = self.get_kernel_field(predictor_names_for_kf)

        # OLS (no configurable parameters - it's a baseline model)
        models.append(OLSModel())

        # GWR with variable-specific bandwidths
        if 'gwr' in model_configs:
            gwr_params = model_configs['gwr'].copy()

            if variable_bandwidths:
                print(f"GWR Variable-Specific Bandwidths:")
                for var, bw in variable_bandwidths.items():
                    print(f"  {var}: {bw}")
                
                # Use variable-specific bandwidths instead of global bandwidth
                gwr_params['variable_bandwidths'] = variable_bandwidths
                # Remove the single global bandwidth if it exists
                if 'bandwidth' in gwr_params:
                    del gwr_params['bandwidth']
            else:
                # Fallback to global bandwidth with safety checks
                if n_samples is not None:
                    min_fold_size = int(n_samples * 0.8 * 0.85)  # 80% training * 85% buffer
                    raw_bandwidth = gwr_params.get('bandwidth')
                    config_bandwidth = int(raw_bandwidth) if raw_bandwidth is not None else 500
                    adaptive_bandwidth = min(config_bandwidth, min_fold_size)
                    gwr_params['bandwidth'] = max(adaptive_bandwidth, 50)
                    
                    print(f"GWR Global Bandwidth (fallback):")
                    print(f"  Dataset size: {n_samples}")
                    print(f"  Config bandwidth: {config_bandwidth}")
                    print(f"  Final bandwidth: {gwr_params['bandwidth']}")
            
            # Build a validated GWRConfig — no hard-coded param lists needed.
            # Constrained regression is disabled during CV to avoid systematic
            # bias; sign constraints are applied post-hoc for interpretation only.
            _gwr_cfg_kwargs = {k: v for k, v in gwr_params.items()
                               if k in GWRConfig.__dataclass_fields__}
            _gwr_cfg_kwargs['use_constrained_regression'] = False
            _gwr_cfg_kwargs['sign_constraints'] = self._monotone_constraints
            if kernel_field is not None:
                _gwr_cfg_kwargs['kernel_field'] = kernel_field

            try:
                models.append(GWRModel.from_config(GWRConfig(**_gwr_cfg_kwargs)))
            except Exception as e:
                print(f"Warning: Failed to create GWR with configured params: {e}")
                models.append(GWRModel(bandwidth=100, kernel='gaussian'))
        
        # GWRF with global hyperparameters  
        if 'gwrf' in model_configs:
            gwrf_params = model_configs['gwrf'].copy()
            
            # Dataset size safety adjustments
            if n_samples is not None:
                # Account for CV fold size (typically ~80% of data for training)
                expected_train_size = int(n_samples * 0.8)
                min_fold_size = int(expected_train_size * 0.85)  # Additional safety margin
                config_k_neighbors = int(gwrf_params.get('k_neighbors', 100))
                adaptive_k_neighbors = min(config_k_neighbors, min_fold_size // 2)
                gwrf_params['k_neighbors'] = max(5, adaptive_k_neighbors)
                
                # Calculate optimal subsample size if using subsampling
                # IMPORTANT: Base calculation on expected training fold size, not full dataset
                if gwrf_params.get('subsample_n') is None and gwrf_params.get('subsample_fraction') is None:
                    if n_samples > 5000:
                        optimal_subsample = self._calculate_optimal_subsample_size(
                            expected_train_size, gwrf_params.get('k_neighbors', 100), gwrf_params
                        )
                        # Ensure subsample is significantly smaller than expected training size
                        safe_subsample = min(optimal_subsample, int(expected_train_size * 0.75))
                        gwrf_params['subsample_n'] = safe_subsample
                        print(f"GWRF: Using spatial subsampling with {safe_subsample} locations (based on expected train size {expected_train_size})")
                
                # Force single thread for local RFs to avoid nested parallelism warnings
                # Outer CV parallelization handles efficient resource utilization
                gwrf_params['n_jobs'] = self._hw.get('gwrf_local_jobs', 1)
                
                print(f"GWRF Global Parameters:")
                print(f"  Dataset size: {n_samples}")
                print(f"  Config k_neighbors: {config_k_neighbors}")
                print(f"  Final k_neighbors: {gwrf_params['k_neighbors']}")
            
            # Build a validated GWRFConfig — physics_signs flows in via config,
            # no post-instantiation attribute assignment needed.
            _gwrf_cfg_kwargs = {k: v for k, v in gwrf_params.items()
                                if k in GWRFConfig.__dataclass_fields__}
            if kernel_field is not None:
                _gwrf_cfg_kwargs['kernel_field'] = kernel_field

            try:
                gwrf_cfg = GWRFConfig(**_gwrf_cfg_kwargs)
                gwrf = GWRFModel.from_config(gwrf_cfg)
                # physics_signs is injected here; GWRF stores it for downstream
                # monotonicity diagnostics.
                gwrf.physics_signs = self._monotone_constraints
                models.append(gwrf)
            except Exception as e:
                print(f"Warning: Failed to create GWRF with configured params: {e}")
                models.append(GWRFModel(n_estimators=50, k_neighbors=100, min_samples_leaf=5, n_jobs=1))
                
        # GGPGAM with global hyperparameters (adaptive via DatasetProfiler)
        if 'ggpgam' in model_configs:
            ggpgam_params = model_configs['ggpgam'].copy()

            # Build a validated GGPGAMConfig — physics_signs is a constructor
            # argument now, eliminating the post-instantiation assignment.
            _ggpgam_cfg_kwargs = {k: v for k, v in ggpgam_params.items()
                                  if k in GGPGAMConfig.__dataclass_fields__}
            _ggpgam_cfg_kwargs['physics_signs'] = self._monotone_constraints

            print(f"GGPGAM Parameters (adaptive):")
            for k, v in _ggpgam_cfg_kwargs.items():
                if k != 'physics_signs':
                    print(f"  {k}: {v}")

            try:
                models.append(GGPGAM_SVC.from_config(GGPGAMConfig(**_ggpgam_cfg_kwargs)))
            except Exception as e:
                print(f"Warning: Failed to create GGPGAM with configured params: {e}")
                ggpgam = GGPGAM_SVC()
                ggpgam.physics_signs = self._monotone_constraints
                models.append(ggpgam)
        else:
            ggpgam = GGPGAM_SVC()
            ggpgam.physics_signs = self._monotone_constraints
            models.append(ggpgam)
        
        # Set output CRS on all models for GeoPackage export
        _out_crs = self._cfg.output_crs
        for m in models:
            m._output_crs = _out_crs
        
        return models
    
    def _average_model_params(self, model_params_dict):
        """Calculate average parameters across all variables for a model type"""
        if not model_params_dict:
            return {}
        
        # Collect all numeric parameters
        param_sums = {}
        param_counts = {}
        categorical_params = {}
        
        for variable, params in model_params_dict.items():
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    if param_name not in param_sums:
                        param_sums[param_name] = 0
                        param_counts[param_name] = 0
                    param_sums[param_name] += param_value
                    param_counts[param_name] += 1
                else:
                    # For categorical parameters, use mode (most common)
                    if param_name not in categorical_params:
                        categorical_params[param_name] = {}
                    if param_value not in categorical_params[param_name]:
                        categorical_params[param_name][param_value] = 0
                    categorical_params[param_name][param_value] += 1
        
        # Calculate averages and modes
        averaged_params = {}
        
        # Numeric parameters
        for param_name in param_sums:
            averaged_params[param_name] = param_sums[param_name] / param_counts[param_name]
        
        # Categorical parameters (use most frequent)
        for param_name, value_counts in categorical_params.items():
            most_common = max(value_counts.items(), key=lambda x: x[1])
            averaged_params[param_name] = most_common[0]
        
        return averaged_params
    
    def determine_optimal_block_size(self, X, y, coords, target_variable_name="temperature"):
        """
        Use correlogram analysis to determine optimal block size for spatial CV
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        coords : array-like
            Spatial coordinates
        target_variable_name : str
            Name of target variable for plotting
            
        Returns:
        --------
        float : Optimal block size based on first non-significant Moran's I
        dict : Detailed correlogram analysis results
        """
        print("Determining optimal block size using spatial correlogram analysis...")
        
        # Create spatial autocorrelation analyzer
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        _auto_max_dist = np.linalg.norm(bounds[1] - bounds[0]) * 0.3
        analyzer = SpatialAutocorrelationAnalyzer(coords, max_distance=_auto_max_dist)
        
        # Compute correlogram for target variable
        correlogram_results = analyzer.compute_correlogram(
            y, 
            plot=True, 
            title=f"Spatial Correlogram - {target_variable_name}"
        )
        
        optimal_block_size = correlogram_results['optimal_block_size']
        
        print(f"Correlogram Analysis Results:")
        print(f"  Optimal block size: {optimal_block_size:.1f}m")
        print(f"  Based on first non-significant Moran's I (|z| ≤ 1.96)")
        print(f"  Total distance lags analyzed: {len(correlogram_results['correlogram_results'])}")
        
        # Show summary of significant lags
        significant_lags = [r for r in correlogram_results['correlogram_results'] if r['significant']]
        print(f"  Significant spatial autocorrelation found at {len(significant_lags)} distance lags")
        
        if len(significant_lags) > 0:
            max_sig_distance = max([r['lag_distance'] for r in significant_lags])
            print(f"  Maximum significant distance: {max_sig_distance:.1f}m")
        
        return optimal_block_size, correlogram_results
    
    def analyze_residual_autocorrelation(self, oof_predictions, y, coords, model_names, output_dir):
        """
        Perform memory-efficient spatial autocorrelation analysis of model residuals
        
        Parameters:
        -----------
        oof_predictions : array-like
            Out-of-fold predictions from all models
        y : array-like
            True target values
        coords : array-like
            Spatial coordinates
        model_names : list
            List of model names
        output_dir : str
            Output directory for saving results
            
        Returns:
        --------
        pandas.DataFrame : Formal Moran's I results table
        """
        print("Performing memory-efficient spatial autocorrelation analysis...")
        
        # Use memory-efficient analysis
        results_df = analyze_model_residuals_morans_i(
            oof_predictions, y, coords, model_names, output_dir
        )
        
        print(f"Spatial autocorrelation analysis completed!")
        print(f"Results saved to: {output_dir}")
        
        return results_df
    
    def _create_autocorrelation_summary_report(self, results_df, output_dir):
        """Create a formatted summary report of spatial autocorrelation analysis"""
        report_path = os.path.join(output_dir, "spatial_autocorrelation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("SPATIAL AUTOCORRELATION ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("Global Moran's I Analysis - Model Residuals\n")
            f.write("-"*40 + "\n\n")
            
            for _, row in results_df.iterrows():
                f.write(f"Model: {row['Model']}\n")
                f.write(f"  Moran's I: {row['Morans_I']:.6f}\n")
                f.write(f"  Expected I: {row['Expected_I']:.6f}\n")
                f.write(f"  Z-Score: {row['Z_Score']:.3f}\n")
                f.write(f"  P-Value: {row['P_Value']:.6f}\n")
                f.write(f"  Significant: {'Yes' if row['Significant'] else 'No'}\n")
                f.write(f"  Interpretation: {row['Interpretation']}\n\n")
            
            # Summary statistics
            significant_models = results_df[results_df['Significant'] == True]
            f.write("SUMMARY\n")
            f.write("-"*20 + "\n")
            f.write(f"Total models analyzed: {len(results_df)}\n")
            f.write(f"Models with significant spatial autocorrelation: {len(significant_models)}\n")
            f.write(f"Proportion with significant autocorrelation: {len(significant_models)/len(results_df):.2%}\n\n")
            
            if len(significant_models) > 0:
                f.write("Models with significant residual autocorrelation:\n")
                for _, row in significant_models.iterrows():
                    f.write(f"  - {row['Model']}: Moran's I = {row['Morans_I']:.4f}, Z = {row['Z_Score']:.2f}\n")
            else:
                f.write("No models show significant residual spatial autocorrelation.\n")
                f.write("This indicates good spatial model performance.\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def generate_optimized_oof_predictions(self, X, y, coords, folds, output_dir, feature_names=None):
        """
        Generate out-of-fold predictions using optimized models with per-model parallelization
        Each model runs in parallel across all folds, then moves to the next model
        
        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or RawFeatureMatrix
            **Unscaled** feature matrix.  Pass a ``RawFeatureMatrix`` at the call
            site to make the scaling contract explicit.  Passing a
            ``ScaledFeatureMatrix`` raises ``TypeError`` immediately — this method
            applies its own canonical scaler internally; double-scaling silently
            corrupts every base-model fit.
        feature_names : list, optional
            List of feature names for OOF extraction. If None, will be extracted from X if it's a DataFrame.
        """
        # ── scaling-contract guard ────────────────────────────────────────────
        from sparc.run.feature_matrix import RawFeatureMatrix, ScaledFeatureMatrix
        if isinstance(X, ScaledFeatureMatrix):
            raise TypeError(
                "generate_optimized_oof_predictions received a ScaledFeatureMatrix. "
                "Pass unscaled features (RawFeatureMatrix or a plain ndarray/DataFrame). "
                "This method applies its own canonical StandardScaler internally; "
                "passing pre-scaled features would double-scale and corrupt all base models."
            )
        if isinstance(X, RawFeatureMatrix):
            X = X.data  # unwrap to ndarray; canonical scaler runs below

        # Extract feature names if not provided
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        models = self.create_optimized_models(n_samples=len(X))
        model_names = ['ols', 'gwr', 'gwrf', 'ggpgam']

        # ------------------------------------------------------------------
        # Centralized feature scaling: fit one canonical scaler on the full
        # feature matrix so every base model operates on the same normalized
        # representation.  Each model still calls its own fit_transform
        # internally, but since X is already ≈N(0,1), those internal scalers
        # become a near-identity pass-through.
        # This eliminates GWEN's subsample-induced scale drift and ensures
        # all surrogate predictions share a consistent feature basis for
        # meta-learner blending.
        # ------------------------------------------------------------------
        _X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        from sklearn.preprocessing import StandardScaler as _CanonicalScaler
        _canonical_scaler = _CanonicalScaler()
        X = _canonical_scaler.fit_transform(_X_arr)

        n_samples = len(y)
        n_models = len(models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        print(f"Generating OOF predictions with {n_models} optimized models...")
        print(f"Hardware acceleration: {self._hw['max_workers']} cores, {self._hw['memory_limit_gb']}GB RAM limit")
        print(f"Strategy: Per-model parallelization - each model runs all {len(folds)} folds in parallel")
        
        # Run each model in parallel across all folds
        if self._hw['parallel_cv'] and len(folds) >= 2:
            print(f"\n{'='*80}")
            print(f"PARALLEL MODE: Each fold will be processed in parallel (per model)")
            print(f"  • Models will be processed sequentially: {', '.join(model_names)}")
            print(f"  • Each model's {len(folds)} folds will run in parallel")
            print(f"  • Max parallel workers per model: {min(len(folds), self._hw['outer_jobs'])}")
            print(f"{'='*80}\n")
            try:
                oof_predictions = self._parallel_cv_training(X, y, coords, models, model_names, folds, feature_names)
                print("\n✓ Per-model parallel CV completed successfully!")
            except Exception as e:
                print(f"\n✗ Per-model parallel CV failed: {e}")
                print("Falling back to sequential processing...")
                oof_predictions = self._sequential_cv_training(X, y, coords, models, model_names, folds, feature_names)
        else:
            print(f"\n{'='*80}")
            print(f"SEQUENTIAL MODE: Processing folds one at a time")
            print(f"{'='*80}\n")
            oof_predictions = self._sequential_cv_training(X, y, coords, models, model_names, folds, feature_names)

        # --- Label-free spatial consistency scoring (gated by config flag) ---
        if self._cfg.self_supervised_scoring:
            _bw = float(self.get_block_size_from_config() or 500.0)
            print("\n=== Label-Free Spatial Consistency Scores (lower = better) ===")
            for _mi, _mn in enumerate(model_names):
                _sc = score_model_spatial_consistency(oof_predictions[:, _mi], coords, bandwidth=_bw)
                print(f"  {_mn}: spatial_consistency_score = {_sc:.4f}  (bandwidth={_bw:.0f}m)")

        # Save OOF predictions — route through artifact_io (store-or-disk seam).
        oof_df = pd.DataFrame(oof_predictions, columns=model_names)
        _oof_path = os.path.join(output_dir, 'optimized_oof_predictions.csv')
        save_table_path(
            oof_df, _oof_path,
            stage="2", artifact_id="oof_predictions",
            producer="enhanced_spatial_cv",
            consumers=["server:/results/spatial_cv/predictions"],
        )

        # Save spatial predictions as a geometry-bearing table — route through
        # artifact_io so store and disk are written consistently.
        try:
            import geopandas as gpd
            # Access the source GeoDataFrame from the parent pipeline
            src_data = getattr(self, '_source_geodataframe', None)
            if src_data is not None and hasattr(src_data, 'geometry') and len(src_data) == len(oof_predictions):
                target_col = self._cfg.target
                gpkg_gdf = gpd.GeoDataFrame(geometry=src_data.geometry.values, crs=src_data.crs)
                # Add observed target
                if target_col in src_data.columns:
                    gpkg_gdf['observed'] = src_data[target_col].values
                # Add identifier if available
                id_col = self._cfg.identifier
                if id_col and id_col in src_data.columns:
                    gpkg_gdf['identifier'] = src_data[id_col].values
                # Add model predictions and residuals
                for i, mname in enumerate(model_names):
                    gpkg_gdf[f'pred_{mname}'] = oof_predictions[:, i]
                    if target_col in src_data.columns:
                        gpkg_gdf[f'residual_{mname}'] = oof_predictions[:, i] - src_data[target_col].values
                # Annotate the GPKG with the target unit so every consumer
                # (QGIS, the API, the renderer) is self-describing without
                # needing to consult a separate artifact.
                try:
                    from sparc.data.units import load_variable_meta as _lvm
                    _var_reg = _lvm(self.base_config)
                    _unit_str = _var_reg.get(target_col).unit or ""
                except Exception:
                    _unit_str = ""
                if _unit_str:
                    gpkg_gdf['_unit'] = _unit_str
                    print(f"  Target unit annotated on GPKG: {_unit_str!r}")
                gpkg_path = os.path.join(output_dir, 'spatial_cv_predictions.gpkg')
                save_geo_path(
                    gpkg_gdf, gpkg_path,
                    stage="2", artifact_id="spatial_cv_predictions",
                    producer="enhanced_spatial_cv",
                    consumers=["server:/results/2/predictions",
                               "server:/results/spatial_cv/predictions"],
                )
                print(f"Spatial CV predictions saved (artifacts.db + {gpkg_path})")
            else:
                print("  [INFO] Source GeoDataFrame not available for gpkg export")
        except Exception as e:
            print(f"  [WARNING] Could not save spatial predictions gpkg: {e}")
        
        # Check for and handle NaN values in OOF predictions
        print("\n=== Checking OOF predictions for NaN values ===")
        nan_counts = np.isnan(oof_predictions).sum(axis=0)
        print(f"NaN counts per model: {dict(zip(model_names, nan_counts))}")
        
        # Handle NaN values by replacing with mean predictions
        for i, (model_name, nan_count) in enumerate(zip(model_names, nan_counts)):
            if nan_count > 0:
                print(f"WARNING: {model_name} has {nan_count} NaN predictions ({nan_count/len(oof_predictions)*100:.1f}%)")
                # Replace NaN values with the mean of non-NaN predictions for this model
                model_predictions = oof_predictions[:, i]
                valid_mask = ~np.isnan(model_predictions)
                if valid_mask.sum() > 0:
                    mean_pred = np.mean(model_predictions[valid_mask])
                    oof_predictions[~valid_mask, i] = mean_pred
                    print(f"  Replaced NaN values with mean prediction: {mean_pred:.4f}")
                else:
                    print(f"  ERROR: All predictions are NaN for {model_name}")
        
        # Calculate and display OOF performance
        print("\n=== Optimized OOF Performance ===")
        for model_idx, model_name in enumerate(model_names):
            try:
                # Double-check for remaining NaN values
                if np.isnan(oof_predictions[:, model_idx]).sum() > 0:
                    print(f"SKIP: {model_name.upper()} still contains NaN values")
                    continue
                    
                r2 = r2_score(y, oof_predictions[:, model_idx])
                rmse = np.sqrt(mean_squared_error(y, oof_predictions[:, model_idx]))
                msg = f"{model_name.upper()}: R² = {r2:.4f}, RMSE = {rmse:.4f}"
                # Append benchmark metrics when enabled
                bm_cfg = {'enabled': self._cfg.benchmark_metrics_enabled}
                if bm_cfg.get('enabled', False):
                    nrmse = _evaluator.calculate_nrmse(y, oof_predictions[:, model_idx])
                    pcorr = _evaluator.calculate_pattern_correlation(y, oof_predictions[:, model_idx])
                    aratio = _evaluator.calculate_amplitude_ratio(y, oof_predictions[:, model_idx])
                    msg += f", nRMSE = {nrmse:.4f}, PatCorr = {pcorr:.4f}, AmpRatio = {aratio:.4f}"
                    # Extreme-value / tail metrics
                    ext = _evaluator.calculate_extreme_metrics(y, oof_predictions[:, model_idx])
                    msg += (f"\n        Tail-lo RMSE={ext['tail_low_rmse']:.4f}  "
                            f"Tail-hi RMSE={ext['tail_high_rmse']:.4f}  "
                            f"Extreme ratio={ext['extreme_rmse_ratio']:.2f}")
                    # Baseline skill scores
                    coords_arr = None
                    if hasattr(self, 'coords') and self.coords is not None:
                        coords_arr = self.coords
                    bl = _evaluator.calculate_baseline_comparisons(
                        y, oof_predictions[:, model_idx], coords=coords_arr,
                    )
                    msg += (f"\n        Skill vs global-mean={bl['skill_vs_global_mean']:.4f}  "
                            f"vs spatial-KNN={bl['skill_vs_spatial_knn']:.4f}")
                print(msg)
            except Exception as e:
                print(f"ERROR calculating performance for {model_name}: {e}")
        
        # OOF intelligence aggregation is handled inside _parallel_cv_training/_sequential_cv_training
        
        return oof_predictions, model_names
    
    def _calculate_optimal_subsample_size(self, n_samples, k_neighbors, gwrf_params):
        """
        Calculate optimal subsample size for GWRF based on dataset size and hardware configuration
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples in the dataset
        k_neighbors : int
            Number of neighbors for GWRF
        gwrf_params : dict
            GWRF parameters dictionary
            
        Returns:
        --------
        int
            Optimal subsample size
        """
        # Base subsample size based on hardware configuration
        if self._hw['high_memory_mode']:
            base_subsample = min(n_samples, 3000)
        else:
            base_subsample = min(n_samples, 1500)
        
        # Adjust based on k_neighbors - need enough samples for meaningful spatial analysis
        min_subsample = max(k_neighbors * 5, 500)  # At least 10x k_neighbors
        
        # Adjust based on available memory
        memory_adjusted = min(base_subsample, self._hw['memory_limit_gb'] * 50)
        
        # Take the maximum of constraints to ensure meaningful spatial analysis
        optimal_subsample = max(min_subsample, memory_adjusted)
        
        # Ensure we don't exceed the total sample size
        optimal_subsample = min(optimal_subsample, n_samples)
        
        return int(optimal_subsample)
    
    def _parallel_cv_training(self, X, y, coords, models, model_names, folds, feature_names=None):
        """
        Enhanced parallel cross-validation training using per-model parallelization
        Each model runs across all folds in parallel for optimal resource utilization
        """
        n_samples = len(y)
        n_models = len(models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        # Storage for OOF intelligence if extraction is enabled
        oof_intelligence_storage = {model_name: [] for model_name in model_names} if EXTRACT_OOF_INTELLIGENCE else None
        
        # Dynamic worker allocation based on dataset size and available resources
        optimal_workers = min(
            len(folds),  # One worker per fold (now per model)
            self._hw['outer_jobs'],  # Controlled outer parallelization
            max(1, self._hw['memory_limit_gb'] // 4)  # More memory per worker for model-level parallelization
        )
        
        print(f"Per-model parallel CV configuration:")
        print(f"  - Workers per model: {optimal_workers}")
        print(f"  - Memory per worker: ~{self._hw['memory_limit_gb'] // optimal_workers}GB")
        print(f"  - Models to process: {n_models} ({', '.join(model_names)})")
        
        # Train each model across all folds in parallel
        for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
            print(f"\n=== Training {model_name} across all folds in parallel ===")
            print(f"[MODEL_START] {model_name} ({model_idx+1}/{n_models})")
            
            # Prepare fold data for this specific model
            fold_data_list = [
                (i, train_idx, test_idx, X, y, coords, model, model_name, feature_names) 
                for i, (train_idx, test_idx) in enumerate(folds)
            ]
            
            completed_folds = 0
            try:
                # Use ProcessPoolExecutor for CPU-intensive tasks with timeout
                with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                    with tqdm(total=len(folds), desc=f"{model_name} CV Folds", unit="fold") as pbar:
                        # Submit all fold tasks for this model
                        future_to_fold = {
                            executor.submit(train_single_model_fold_worker, fold_data): i 
                            for i, fold_data in enumerate(fold_data_list)
                        }
                        
                        for future in future_to_fold:
                            try:
                                # Get result with timeout (30 minutes per fold max)
                                result = future.result(timeout=1800)
                                if EXTRACT_OOF_INTELLIGENCE and len(result) == 4:
                                    fold_idx, test_idx, fold_predictions, oof_intel = result
                                    if oof_intel and model_name in oof_intelligence_storage:
                                        oof_intelligence_storage[model_name].append(oof_intel)
                                else:
                                    fold_idx, test_idx, fold_predictions = result[:3] if len(result) >= 3 else result
                                oof_predictions[test_idx, model_idx] = fold_predictions
                                completed_folds += 1
                                pbar.update(1)
                                pbar.set_postfix(
                                    completed=f"{completed_folds}/{len(folds)}",
                                    current_fold=f"{fold_idx+1}",
                                    model=model_name
                                )
                            except Exception as e:
                                fold_idx = future_to_fold[future]
                                print(f"{model_name} Fold {fold_idx+1} failed: {e}")
                                # Use sequential fallback for failed fold
                                train_idx, test_idx = folds[fold_idx]
                                X_train, X_test = X[train_idx], X[test_idx]
                                y_train = y[train_idx]
                                coords_train, coords_test = coords[train_idx], coords[test_idx]
                                
                                try:
                                    if 'coords' in model.fit.__code__.co_varnames:
                                        model.fit(X_train, y_train, coords_train)
                                    else:
                                        model.fit(X_train, y_train)
                                    
                                    if 'coords' in model.predict.__code__.co_varnames:
                                        predictions = model.predict(X_test, coords_test)
                                    else:
                                        predictions = model.predict(X_test)
                                    
                                    # Handle tuple returns (e.g., GWRF returns (preds, uncertainty))
                                    if isinstance(predictions, tuple):
                                        predictions = predictions[0]
                                    
                                    oof_predictions[test_idx, model_idx] = predictions
                                except:
                                    oof_predictions[test_idx, model_idx] = np.mean(y_train)
                                
                                pbar.update(1)
                                
            except Exception as e:
                print(f"Parallel processing failed for {model_name}: {e}")
                # Sequential fallback for entire model
                print(f"Using sequential fallback for {model_name}")
                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train = y[train_idx]
                    coords_train, coords_test = coords[train_idx], coords[test_idx]
                    
                    try:
                        if 'coords' in model.fit.__code__.co_varnames:
                            model.fit(X_train, y_train, coords_train)
                        else:
                            model.fit(X_train, y_train)
                        
                        if 'coords' in model.predict.__code__.co_varnames:
                            predictions = model.predict(X_test, coords_test)
                        else:
                            predictions = model.predict(X_test)
                        
                        # Handle tuple returns (e.g., GWRF returns (preds, uncertainty))
                        if isinstance(predictions, tuple):
                            predictions = predictions[0]
                        
                        oof_predictions[test_idx, model_idx] = predictions
                    except:
                        oof_predictions[test_idx, model_idx] = np.mean(y_train)
            
            print(f"{model_name} completed: {completed_folds}/{len(folds)} folds successful")
            print(f"[MODEL_DONE] {model_name} ({model_idx+1}/{n_models})")
        
        
        # Aggregate OOF intelligence if extraction was enabled
        if EXTRACT_OOF_INTELLIGENCE and oof_intelligence_storage:
            try:
                from sparc.interventions.oof_extraction_hooks import aggregate_oof_intelligence
                print(f"\n=== Aggregating OOF Intelligence ===")
                for model_name, fold_results in oof_intelligence_storage.items():
                    if fold_results:
                        print(f"Aggregating {len(fold_results)} folds for {model_name}")
                        oof_aggregated = aggregate_oof_intelligence(
                            fold_results=fold_results,
                            feature_names=feature_names,
                            n_samples=n_samples,
                            output_dir=self.paths.stage3_dir
                        )
                        print(f"  [OK] {model_name} OOF intelligence saved")
            except Exception as e:
                print(f"Warning: Could not aggregate OOF intelligence: {e}")
        
        print(f"\nAll models completed in per-model parallel CV")
        return oof_predictions

    def _sequential_cv_training(self, X, y, coords, models, model_names, folds, feature_names=None):
        """
        Sequential cross-validation training with enhanced model-specific optimizations
        """
        n_samples = len(y)
        n_models = len(models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        with tqdm(total=len(folds), desc="Sequential CV Folds", unit="fold") as pbar:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                fold_predictions = np.zeros((len(test_idx), n_models))
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]
                coords_train, coords_test = coords[train_idx], coords[test_idx]
                
                print(f"\nFold {fold_idx + 1}/{len(folds)}")
                print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
                
                for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
                    if fold_idx == 0:
                        print(f"[MODEL_START] {model_name} ({model_idx+1}/{n_models})")
                    try:
                        print(f"    Training {model_name} on {len(train_idx)} samples...")
                        start_time = time.time()
                        
                        # Model-specific training
                        if 'coords' in model.fit.__code__.co_varnames:
                            model.fit(X_train, y_train, coords_train)
                        else:
                            model.fit(X_train, y_train)
                        
                        train_time = time.time() - start_time
                        print(f"    {model_name} training completed in {train_time:.2f}s")
                        
                        # Predict
                        if 'coords' in model.predict.__code__.co_varnames:
                            predictions = model.predict(X_test, coords_test)
                        else:
                            predictions = model.predict(X_test)
                        
                        # Handle tuple returns (e.g., GWRF returns (preds, uncertainty))
                        if isinstance(predictions, tuple):
                            predictions = predictions[0]
                        
                        fold_predictions[:, model_idx] = predictions
                        print(f"  {model_name} completed successfully")
                        
                    except Exception as e:
                        print(f"Sequential training error - Fold {fold_idx+1}, {model_name}: {e}")
                        fold_predictions[:, model_idx] = np.mean(y_train)
                        print(f"  Using fallback prediction for {model_name}")
                
                oof_predictions[test_idx] = fold_predictions
                pbar.update(1)
                pbar.set_postfix(fold=f"{fold_idx+1}")
        
        return oof_predictions

    def run_enhanced_spatial_cv(self):
        """
        Run the complete enhanced spatial CV pipeline with stage checking
        """
        print("\n=== Enhanced Spatial CV with Optimized Parameters ===\n")
        
        # Use centralized paths instead of config paths
        stage2_dir = str(self.paths.stage2_dir)
        os.makedirs(stage2_dir, exist_ok=True)
        
        # Check if OOF predictions already exist (store-first, disk fallback)
        oof_predictions_path = str(self.paths.oof_predictions)
        folds_path = str(self.paths.folds_file)
        
        if exists_path(oof_predictions_path, stage="2", artifact_id="oof_predictions") and \
           exists_path(folds_path, stage="2", artifact_id="folds"):
            print("=== Loading existing OOF predictions ===")
            print(f"Found existing OOF predictions at: {self.paths.get_relative_path(oof_predictions_path)}")
            print(f"Found existing folds at: {self.paths.get_relative_path(folds_path)}")
            
            # Load existing results (store-first, CSV fallback)
            oof_df = load_table_path(oof_predictions_path, stage="2", artifact_id="oof_predictions")
            model_names = oof_df.columns.tolist()
            oof_predictions = oof_df.values
            
            # Load existing folds
            folds = load_blob_path(folds_path, stage="2", artifact_id="folds")
            if folds is None:
                folds = joblib.load(folds_path)
            
            # Load data for performance calculation
            print("=== Loading Data for Performance Calculation ===")
            selected_features = list(self._cfg.features)
            
            data = load_and_preprocess_data(
                raw_data_path=self._cfg.raw_csv_path,
                identifier_col=self._cfg.identifier,
                target_col=self._cfg.target,
                coords_cols=self._cfg.coordinates,
                predictor_cols=selected_features,
                initial_crs=self._cfg.initial_crs,
                target_crs=self._cfg.target_crs,
                output_dir=self._cfg.output_dir,
            )

            # Store reference for gpkg export
            self._source_geodataframe = data
            
            available_features = data.columns.tolist()
            selected_features = [f for f in selected_features if f in available_features]
            feature_names = selected_features
            
            y = data[self._cfg.target].values
            
            # Check for NaN values in OOF predictions and handle them
            print("\n=== Checking OOF predictions for NaN values ===")
            nan_counts = np.isnan(oof_predictions).sum(axis=0)
            print(f"NaN counts per model: {dict(zip(model_names, nan_counts))}")
            
            # Handle NaN values by replacing with mean predictions or skipping problematic models
            for i, (model_name, nan_count) in enumerate(zip(model_names, nan_counts)):
                if nan_count > 0:
                    print(f"WARNING: {model_name} has {nan_count} NaN predictions ({nan_count/len(oof_predictions)*100:.1f}%)")
                    # Replace NaN values with the mean of non-NaN predictions for this model
                    model_predictions = oof_predictions[:, i]
                    valid_mask = ~np.isnan(model_predictions)
                    if valid_mask.sum() > 0:
                        mean_pred = np.mean(model_predictions[valid_mask])
                        oof_predictions[valid_mask == False, i] = mean_pred
                        print(f"  Replaced NaN values with mean prediction: {mean_pred:.4f}")
                    else:
                        print(f"  ERROR: All predictions are NaN for {model_name}")
            
            # Display performance
            print("\n=== Existing OOF Performance ===")
            performance_summary = {'individual_models': {}}

            for i, model_name in enumerate(model_names):
                try:
                    # Double-check for remaining NaN values
                    if np.isnan(oof_predictions[:, i]).sum() > 0:
                        print(f"SKIP: {model_name.upper()} still contains NaN values")
                        performance_summary['individual_models'][model_name] = {
                            'r2': np.nan, 'rmse': np.nan, 'status': 'failed_nan'
                        }
                        continue
                        
                    model_r2 = r2_score(y, oof_predictions[:, i])
                    model_rmse = np.sqrt(mean_squared_error(y, oof_predictions[:, i]))
                    perf = {'r2': model_r2, 'rmse': model_rmse, 'status': 'success'}
                    msg = f"{model_name.upper()}: R² = {model_r2:.4f}, RMSE = {model_rmse:.4f}"
                    if self._cfg.benchmark_metrics_enabled:
                        perf['nrmse'] = _evaluator.calculate_nrmse(y, oof_predictions[:, i])
                        perf['pattern_correlation'] = _evaluator.calculate_pattern_correlation(y, oof_predictions[:, i])
                        perf['amplitude_ratio'] = _evaluator.calculate_amplitude_ratio(y, oof_predictions[:, i])
                        msg += f", nRMSE = {perf['nrmse']:.4f}, PatCorr = {perf['pattern_correlation']:.4f}, AmpRatio = {perf['amplitude_ratio']:.4f}"
                        ext = _evaluator.calculate_extreme_metrics(y, oof_predictions[:, i])
                        perf.update(ext)
                        msg += (f"\n        Tail-lo RMSE={ext['tail_low_rmse']:.4f}  "
                                f"Tail-hi RMSE={ext['tail_high_rmse']:.4f}  "
                                f"Extreme ratio={ext['extreme_rmse_ratio']:.2f}")
                        coords_arr = getattr(self, 'coords', None)
                        bl = _evaluator.calculate_baseline_comparisons(
                            y, oof_predictions[:, i], coords=coords_arr,
                        )
                        perf.update(bl)
                        msg += (f"\n        Skill vs global-mean={bl['skill_vs_global_mean']:.4f}  "
                                f"vs spatial-KNN={bl['skill_vs_spatial_knn']:.4f}")
                    performance_summary['individual_models'][model_name] = perf
                    print(msg)
                except Exception as e:
                    print(f"ERROR calculating performance for {model_name}: {e}")
                    performance_summary['individual_models'][model_name] = {
                        'r2': np.nan, 'rmse': np.nan, 'status': f'error_{type(e).__name__}'
                    }
            
            print(f"\n=== Spatial CV Complete (loaded existing results) ===")
            print(f"Results loaded from: {stage2_dir}/")
            
            return {
                'oof_predictions': oof_predictions,
                'feature_names': feature_names,
                'performance': performance_summary
            }
        
        # If results don't exist, run the full pipeline
        print("=== Running Spatial CV pipeline ===")
        
        # Load and preprocess data
        print("=== Loading and Preprocessing Data ===")
        selected_features = list(self._cfg.features)
        
        # GUARD: never allow the target variable into the feature matrix
        target_col = self._cfg.target
        if target_col in selected_features:
            print(f"[WARNING] Target variable '{target_col}' found in selected_features — removing to prevent data leakage!")
            selected_features = [f for f in selected_features if f != target_col]
        
        data = load_and_preprocess_data(
            raw_data_path=self._cfg.raw_csv_path,
            identifier_col=self._cfg.identifier,
            target_col=self._cfg.target,
            coords_cols=self._cfg.coordinates,
            predictor_cols=selected_features,
            initial_crs=self._cfg.initial_crs,
            target_crs=self._cfg.target_crs,
            output_dir=self._cfg.output_dir,
        )

        # Store reference to the source GeoDataFrame for gpkg export
        self._source_geodataframe = data
        
        # Filter features to only those available
        available_features = data.columns.tolist()
        selected_features = [f for f in selected_features if f in available_features]
        
        # ── DatasetProfiler: adapt model hyper-parameters to this dataset ──
        try:
            from sparc.run.dataset_profiler import DatasetProfiler
            profiler = DatasetProfiler(
                data,
                coord_cols=self._cfg.coordinates,
                feature_cols=selected_features,
            )
            print(profiler.summary())
            recs = profiler.recommend_parameters()
            # Only apply profiler overrides if explicitly enabled in config;
            # by default, honour the user-specified hyperparameters.
            if self._cfg.use_dataset_profiler:
                self.apply_profiler_overrides(recs)
                print("DatasetProfiler overrides applied.")
            else:
                print("DatasetProfiler: profiling only (overrides disabled). "
                      "Set pipeline.use_dataset_profiler: true to enable.")
            # Persist profile for downstream stages (scenarios, report)
            import json as _json
            profile_payload = profiler.profile()
            try:
                from sparc.registry.store import get_active_store
                _store = get_active_store()
                if _store is not None:
                    _store.write_struct("2", "dataset_profile", profile_payload,
                                        producer="enhanced_spatial_cv")
                else:
                    profile_path = os.path.join(stage2_dir, 'dataset_profile.json')
                    with open(profile_path, 'w') as _fp:
                        _json.dump(profile_payload, _fp, indent=2)
            except Exception:
                profile_path = os.path.join(stage2_dir, 'dataset_profile.json')
                with open(profile_path, 'w') as _fp:
                    _json.dump(profile_payload, _fp, indent=2)
        except Exception as _e:
            print(f"Warning: DatasetProfiler unavailable ({_e}). Using project config defaults.")
        
        X_gwen = data[selected_features].values
        y = data[self._cfg.target].values
        # Use projected (metric) coordinates so block_size/buffer_size are in metres.
        # load_and_preprocess_data always writes projected_X / projected_Y.
        if 'projected_X' in data.columns and 'projected_Y' in data.columns:
            coords = data[['projected_X', 'projected_Y']].values
        else:
            coords = data[self._cfg.coordinates].values

        # ── Optional SLX spatial-lag augmentation ──────────────────────────
        # Appends neighbor-averaged predictors WX to X_gwen so that base
        # models can distinguish own-location from spill-over covariates.
        # Activated by setting ``predictors.include_spatial_lag: true`` in
        # project.yml.  k controls queen contiguity (default 8 on a fishnet).
        _slx_cfg = self._cfg.raw.get('predictors', {}) or {}
        if _slx_cfg.get('include_spatial_lag', False):
            try:
                from sparc.features.spatial_lag import SpatialLagTransformer
                _slx_k = int(_slx_cfg.get('spatial_lag_k', 8))
                _slx = SpatialLagTransformer(k=_slx_k)
                X_gwen, selected_features = _slx.fit_transform(
                    coords, X_gwen, list(selected_features)
                )
                print(
                    f"SLX: augmented feature matrix → {X_gwen.shape[1]} columns "
                    f"({X_gwen.shape[1] // 2} predictors × 2, k={_slx_k} neighbors)"
                )
            except Exception as _slx_e:
                print(f"Warning: SLX augmentation failed ({_slx_e}). Using original features.")
        # ───────────────────────────────────────────────────────────────────

        print(f"Using {len(selected_features)} features: {selected_features}")
        
        # CRITICAL: Base models have INTERNAL scalers!
        # DO NOT pre-scale features - models handle scaling internally
        print("\n[WARNING]  IMPORTANT: Base models have internal StandardScalers")
        print("   Using UNSCALED features - models will scale internally")
        print(f"   Feature ranges:")
        for i, feat in enumerate(selected_features[:3]):  # Show first 3
            print(f"     {feat}: [{X_gwen[:, i].min():.1f}, {X_gwen[:, i].max():.1f}]")
        
        # NOTE: No feature scaling here! Base models have INTERNAL scalers.
        # The feature_scaler.pkl gets created in main() for meta-ensemble training only.
        # DO NOT create a scaler here - it would be fitted on wrong data or unused.
        
        # Save feature names for reference
        feature_info = {
            'feature_names': selected_features,
            'n_features': len(selected_features),
            'scaling_applied': False  # Base models scale internally!
        }
        feature_info_path = os.path.join(stage2_dir, 'feature_info.json')
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None:
                _store.write_struct("2", "feature_info", feature_info,
                                    producer="enhanced_spatial_cv")
                print("Feature info written to artifacts.db (struct 'feature_info')")
            else:
                with open(feature_info_path, 'w') as f:
                    json.dump(feature_info, f, indent=2)
                print(f"Feature info saved to: {feature_info_path}")
        except Exception:
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"Feature info saved to: {feature_info_path}")
        
        # Use UNSCALED features for modeling (models scale internally)
        X_augmented = X_gwen  # NOT X_gwen_scaled!
        feature_names = selected_features
        
        # Generate spatial folds
        print("\n=== Generating Spatial Folds ===")
        folds = latent_guided_spatial_kfold(
            X=X_augmented,
            y=y,
            coords=coords,
            n_splits=5,
            block_size=self.get_block_size_from_config(),
            buffer_size=self.get_buffer_size_from_config(),
            method="block",
            stratify_y=True,
            prior_trunk=self._load_prior_trunk(),
        )
        
        # Generate OOF predictions
        print("\n=== Base Models ===")
        oof_predictions, model_names = self.generate_optimized_oof_predictions(
            X_augmented, y, coords, folds, stage2_dir, feature_names
        )
        
        # Save folds for later use in Stage 3
        save_blob_path(folds, self.paths.folds_file, stage="2", artifact_id="folds",
                       producer="enhanced_spatial_cv",
                       consumers=["stage2:retrain", "stage3:causal"])
        print(f"Saved spatial folds to {self.paths.get_relative_path(self.paths.folds_file)}")
        
        # Simple performance summary
        performance_summary = {
            'individual_models': {}
        }
        
        for i, model_name in enumerate(model_names):
            model_r2 = r2_score(y, oof_predictions[:, i])
            model_rmse = np.sqrt(mean_squared_error(y, oof_predictions[:, i]))
            perf = {'r2': model_r2, 'rmse': model_rmse}
            msg = f"{model_name.upper()}: R² = {model_r2:.4f}, RMSE = {model_rmse:.4f}"
            if self._cfg.benchmark_metrics_enabled:
                perf['nrmse'] = _evaluator.calculate_nrmse(y, oof_predictions[:, i])
                perf['pattern_correlation'] = _evaluator.calculate_pattern_correlation(y, oof_predictions[:, i])
                perf['amplitude_ratio'] = _evaluator.calculate_amplitude_ratio(y, oof_predictions[:, i])
                msg += f", nRMSE = {perf['nrmse']:.4f}, PatCorr = {perf['pattern_correlation']:.4f}, AmpRatio = {perf['amplitude_ratio']:.4f}"
                ext = _evaluator.calculate_extreme_metrics(y, oof_predictions[:, i])
                perf.update(ext)
                msg += (f"\n        Tail-lo RMSE={ext['tail_low_rmse']:.4f}  "
                        f"Tail-hi RMSE={ext['tail_high_rmse']:.4f}  "
                        f"Extreme ratio={ext['extreme_rmse_ratio']:.2f}")
                coords_arr = getattr(self, 'coords', None)
                bl = _evaluator.calculate_baseline_comparisons(
                    y, oof_predictions[:, i], coords=coords_arr,
                )
                perf.update(bl)
                msg += (f"\n        Skill vs global-mean={bl['skill_vs_global_mean']:.4f}  "
                        f"vs spatial-KNN={bl['skill_vs_spatial_knn']:.4f}")
            performance_summary['individual_models'][model_name] = perf
            print(msg)
        
        print(f"\n=== Spatial CV Complete ===")
        print(f"Results saved to: {stage2_dir}/")
        
        return {
            'oof_predictions': oof_predictions,
            'feature_names': feature_names,
            'performance': performance_summary
        }

def main(fast_mode=False):
    """
    Main function to run the complete enhanced spatial cross-validation pipeline
    with integrated V2 Neural Meta-Learner
    """
    # Re-resolve hardware profile so that any performance settings the user
    # changed since this module was first imported take effect for this run.
    from sparc.config.hardware_profile import reset_profile_cache
    reset_profile_cache()
    refresh_hardware_config(verbose=True)

    # Ensure stdout can handle Unicode (Windows cp1252 cannot encode emoji)
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    try:
        # Initialize the enhanced spatial CV system
        cv_system = EnhancedSpatialCV()
        
        # Check if Stage 2 (base model OOF) should be skipped entirely
        skip_stage_2 = cv_system._cfg.skip_stage_2_base_models
        
        if skip_stage_2:
            # =================================================================
            # Stage 2: SKIPPED — load data & folds only, no base model training
            # =================================================================
            print("\n=== Base Models: SKIPPED (skip_stage_2_base_models=true) ===")
            print("   Surrogates will train against y directly (no V1 OOF pretraining)")

            base_fitted_values = {}
            print("WARNING: Stage 2 skipped — no base model fitted values available.")
            print("         Surrogates will train against raw y. This degrades surrogate quality.")
            print("         Set skip_stage_2_base_models=false to enable proper surrogate pretraining.")
            
            selected_features = list(cv_system._cfg.features)
            data = load_and_preprocess_data(
                raw_data_path=cv_system._cfg.raw_csv_path,
                identifier_col=cv_system._cfg.identifier,
                target_col=cv_system._cfg.target,
                coords_cols=cv_system._cfg.coordinates,
                predictor_cols=selected_features,
                initial_crs=cv_system._cfg.initial_crs,
                target_crs=cv_system._cfg.target_crs,
                output_dir=cv_system._cfg.output_dir,
            )
            
            paths = get_paths()
            available_features = data.columns.tolist()
            selected_features_filtered = [f for f in selected_features if f in available_features]
            data_unscaled = data.copy()
            X_original_features = data_unscaled[selected_features_filtered].values
            
            target_col = cv_system._cfg.target
            coords = data[['projected_X', 'projected_Y']].values
            y = data[target_col].values
            
            base_model_names = []
            base_predictions = {}
            
            # Save a feature scaler for downstream compatibility
            stage2_dir = str(cv_system.paths.stage2_dir)
            ensure_dir(stage2_dir)
            scaler_path = os.path.join(paths.stage2_dir, 'feature_scaler.pkl')
            from sklearn.preprocessing import StandardScaler
            feature_scaler = StandardScaler()
            feature_scaler.fit(data[selected_features_filtered].values)
            save_blob_path(feature_scaler, scaler_path, stage="2", artifact_id="feature_scaler",
                           producer="enhanced_spatial_cv",
                           consumers=["stage4:scenario"])

            # Generate or load spatial folds
            folds_path = str(cv_system.paths.folds_file)
            cached_folds = load_blob_path(folds_path, stage="2", artifact_id="folds")
            if cached_folds is not None:
                folds = cached_folds
                print(f"Loaded {len(folds)} spatial folds from cache")
            else:
                print("Generating spatial folds...")
                folds = spatial_kfold_enhanced(
                    X=X_original_features,
                    y=y,
                    coords=coords,
                    n_splits=5,
                    block_size=cv_system.get_block_size_from_config(),
                    buffer_size=cv_system.get_buffer_size_from_config(),
                    method='block',
                    stratify_y=True,
                )
                save_blob_path(folds, folds_path, stage="2", artifact_id="folds",
                               producer="enhanced_spatial_cv")
                print(f"Saved {len(folds)} spatial folds")

            spatial_intel_dir = os.path.join(paths.output_dir, "spatial_intelligence")
            ensure_dir(spatial_intel_dir)
            
            print(f"Data: {len(y)} samples, {len(selected_features_filtered)} features")
            print(f"Features: {selected_features_filtered}")
        else:
            # =================================================================
            # Stage 2: Run base models with OOF predictions (normal path)
            # =================================================================
            print("\n=== Base Models ===")
            stage2_results = cv_system.run_enhanced_spatial_cv()
            
            # Load and merge data by ID
            print("\n=== Loading and Merging Data by ID ===")
            
            # Load raw data and process it the same way as in Stage 2
            raw_data = pd.read_csv(cv_system._cfg.raw_csv_path)
            
            # Process the data the same way as Stage 2 to get correct coordinates
            selected_features = list(cv_system._cfg.features)
            
            data = load_and_preprocess_data(
                raw_data_path=cv_system._cfg.raw_csv_path,
                identifier_col=cv_system._cfg.identifier,
                target_col=cv_system._cfg.target,
                coords_cols=cv_system._cfg.coordinates,
                predictor_cols=selected_features,
                initial_crs=cv_system._cfg.initial_crs,
                target_crs=cv_system._cfg.target_crs,
                output_dir=cv_system._cfg.output_dir,
            )
            
            # CRITICAL: Save UNSCALED features NOW before meta-ensemble scaling modifies them!
            # Base models need the original unscaled data (they scale internally)
            paths = get_paths()
            available_features = data.columns.tolist()
            selected_features_filtered = [f for f in selected_features if f in available_features]
            
            # Save unscaled features for base model retraining (Stage 2b)
            data_unscaled = data.copy()  # Deep copy with original unscaled features
            print(f"Saved unscaled feature copy for base model retraining")
            
            # Extract original (unscaled) features for monotonic constraint enforcement
            # in the neural meta-learner — these are physical variables whose direction matters
            X_original_features = data_unscaled[selected_features_filtered].values
            print(f"Prepared original features for monotonic constraints: {selected_features_filtered}")
            
            # NOW create/apply scaling for meta-ensemble training (this modifies 'data' in-place)
            scaler_path = os.path.join(paths.stage2_dir, 'feature_scaler.pkl')
            
            # Create a NEW feature scaler fitted on these features
            from sklearn.preprocessing import StandardScaler
            feature_scaler = StandardScaler()
            X_features = data[selected_features_filtered].values
            X_features_scaled = feature_scaler.fit_transform(X_features)
            
            # Save the scaler for later use
            save_blob_path(feature_scaler, scaler_path, stage="2", artifact_id="feature_scaler",
                           producer="enhanced_spatial_cv",
                           consumers=["stage4:scenario"])
            print(f"Created and saved feature scaler to: {scaler_path}")
            print(f"  Scaler fitted on {len(selected_features_filtered)} features")
            
            # Replace features in 'data' with scaled ones (for meta-ensemble only!)
            for i, feature_name in enumerate(selected_features_filtered):
                data[feature_name] = X_features_scaled[:, i]
            
            print(f"Applied feature scaling to 'data' for meta-ensemble training")
            print(f"  Note: 'data_unscaled' still contains original values for base models")
            
            # Load OOF predictions (store-first, CSV fallback)
            oof_predictions = load_table_path(
                str(cv_system.paths.oof_predictions),
                stage="2", artifact_id="oof_predictions",
            )
            
            # Get the identifier column name from config
            id_col = cv_system._cfg.identifier
            
            # Add ID column to OOF predictions.
            # Row counts must match — both come from the same load_and_preprocess_data
            # call (deterministic CSV read), so ordering is consistent.
            if id_col not in oof_predictions.columns:
                # Check if ID is a column or the index
                if id_col in data.columns:
                    if len(data) != len(oof_predictions):
                        raise ValueError(
                            f"OOF/data row mismatch: data={len(data)}, "
                            f"oof_predictions={len(oof_predictions)}. "
                            "Re-run Stage 2 from scratch to regenerate OOF predictions."
                        )
                    oof_predictions[id_col] = data[id_col].values
                elif data.index.name == id_col:
                    oof_predictions[id_col] = data.index.values
                else:
                    # Fallback: use row indices
                    print(f"  Warning: {id_col} not found in data, using row indices")
                    oof_predictions[id_col] = np.arange(len(oof_predictions))
                    data[id_col] = np.arange(len(data))
            
            # Ensure data has ID as a column for join
            if id_col not in data.columns:
                if data.index.name == id_col:
                    data = data.reset_index()
                else:
                    data[id_col] = np.arange(len(data))
            
            # Set index and join
            data = data.set_index(id_col)
            oof_predictions = oof_predictions.set_index(id_col)
            
            # Join dataframes
            joined_data = data.join(oof_predictions, how='inner')
            
            print(f"Joined data shape: {joined_data.shape}")
            
            # Extract components
            target_col = cv_system._cfg.target
            coords = joined_data[['projected_X', 'projected_Y']].values
            y = joined_data[target_col].values
            base_model_names = ['ols', 'gwr', 'gwrf', 'ggpgam']
            
            # Extract base model predictions by name
            base_predictions = {}
            for model_name in base_model_names:
                base_predictions[model_name] = joined_data[model_name].values
            
            # Load saved folds
            print("\n=== Loading Spatial Folds ===")
            folds_path = str(cv_system.paths.folds_file)
            folds = load_blob_path(folds_path, stage="2", artifact_id="folds")
            if folds is None:
                print(f"{cv_system.paths.get_relative_path(folds_path)} not found, this should have been created in Stage 2")
                return None
            print(f"Loaded {len(folds)} spatial folds")

            # Define spatial_intel_dir early so it's available for all stages
            paths = get_paths()
            spatial_intel_dir = os.path.join(paths.output_dir, "spatial_intelligence")
            ensure_dir(spatial_intel_dir)
        
        # ============================================================================
        # Retrain Base Models on Full Dataset for Deployment
        # ============================================================================
        # Check if full retrain should be skipped
        # (force-skip when base CV was skipped — no base models to retrain)
        skip_stage_2b = skip_stage_2 or cv_system._cfg.skip_stage_2b_full_retrain
        
        if skip_stage_2b:
            print("\n=== Full Retrain: SKIPPED (skip_stage_2b_full_retrain=true) ===")
            print("   To enable full model retraining, set skip_stage_2b_full_retrain to false")
            full_models_dir = None  # Set to None to indicate skipped
            if not skip_stage_2:
                # Stage 2 ran but 2b was skipped — no fitted values available
                base_fitted_values = {}
                print("WARNING: Stage 2b skipped — no base model fitted values available.")
                print("         Surrogates will train against raw y. This degrades surrogate quality.")
        else:
            print("\n=== Retraining Base Models on Full Dataset ===")
            
            # Create output directory for full models
            paths = get_paths()
            full_models_dir = os.path.join(paths.stage2_dir, "base_models_full")
            ensure_dir(full_models_dir)
            print(f"Full models will be saved to: {full_models_dir}")
        
            # Prepare full dataset using same preprocessing as Stage 2
            print("Preparing full dataset with UNSCALED features for base model retraining...")
            
            # CRITICAL: Use data_unscaled (saved before meta-ensemble scaling!)
            # Get selected features
            selected_features_2b = list(cv_system._cfg.features)
            available_features_2b = data_unscaled.columns.tolist()
            selected_features_2b = [f for f in selected_features_2b if f in available_features_2b]
            
            # Extract UNSCALED features, target, and coordinates
            X_full_df = data_unscaled[selected_features_2b]          # DataFrame (keeps column names for GWR)
            X_full = X_full_df.values                                 # ndarray for other models
            y_full = data_unscaled[cv_system._cfg.target].values
            coords_full = data_unscaled[cv_system._cfg.coordinates].values
            
            print(f"Full dataset: {len(X_full)} samples, {len(selected_features_2b)} features")
            print(f"Features: {selected_features_2b}")
            
            # Verify features are truly unscaled
            print(f"[OK] Feature ranges (UNSCALED):")
            for i, fname in enumerate(selected_features_2b[:3]):  # Show first 3
                print(f"    {fname}: [{X_full[:, i].min():.2f}, {X_full[:, i].max():.2f}]")
            
            # CRITICAL: Base models (OLS, GWR, GWRF, GGPGAM) have INTERNAL scalers!
            # They expect UNSCALED features and handle scaling internally.
            # We're using data_unscaled which was saved before meta-ensemble scaling.
            print("\n[WARNING]  IMPORTANT: Base models have internal StandardScalers")
            print("   Passing UNSCALED features - models will scale internally")
            
            # Create base models with same hyperparameters as Stage 2
            print("\nCreating base models with optimized hyperparameters...")
            models = cv_system.create_optimized_models(n_samples=len(X_full))
            model_names_2b = ['ols', 'gwr', 'gwrf', 'ggpgam']
            
            print(f"Models to train: {model_names_2b}")
            
            # NOTE: GWR constrained regression (use_constrained_regression=True) is now
            # enabled by default in GWRModel.__init__ — physics sign constraints will be
            # automatically enforced during fitting.
            
            # Train and save each model (PASS UNSCALED FEATURES!)
            # Each model checks for an existing .pkl and skips if found (resume).

            ols_path = os.path.join(full_models_dir, "ols_model_full.pkl")
            cached = load_blob_path(ols_path, stage="2", artifact_id="ols_model_full")
            if cached is not None:
                print("\n--- OLS: already trained -- skipping ---")
                ols_model = cached
                models[0] = ols_model
            else:
                print("\n--- Training OLS on full dataset ---")
                ols_model = models[0]
                ols_model.fit(X_full, y_full)  # UNSCALED!
                save_blob_path(ols_model, ols_path, stage="2", artifact_id="ols_model_full",
                               producer="enhanced_spatial_cv")
                print(f"[OK] Saved ols_model_full.pkl")
            
            gwr_path = os.path.join(full_models_dir, "gwr_model_full.pkl")
            gwr_coef_output_path = os.path.join(full_models_dir, "mgwr_local_coefficients.csv")
            cached_gwr = load_blob_path(gwr_path, stage="2", artifact_id="gwr_model_full")
            if cached_gwr is not None and (not disk_writes_enabled() or os.path.exists(gwr_coef_output_path)):
                print("\n--- GWR: already trained -- skipping ---")
                gwr_model = cached_gwr
                models[1] = gwr_model
            else:
                print("\n--- Training GWR on full dataset ---")
                gwr_model = models[1]

                # Build DAG-informed physics priors for prior-centered ridge regression.
                try:
                    from sparc.config.config import load_physics_priors
                    _physics_raw = load_physics_priors(cv_system._cfg.raw)
                    _coefficients_section = _physics_raw.get('coefficients', {})
                    _prior_dict = {}
                    _temp_scaler = StandardScaler()
                    _temp_scaler.fit(X_full_df.values)
                    for fidx, fname in enumerate(selected_features_2b):
                        entry = _coefficients_section.get(fname, {})
                        if isinstance(entry, dict) and 'value' in entry:
                            lit_val = entry['value']
                            if 'Pct' in fname or 'pct' in fname:
                                unit_inc = 10.0
                            elif fname in ('NDVI', 'Albedo'):
                                unit_inc = 0.1
                            else:
                                unit_inc = _physics_raw.get('unit_increments', {}).get(fname, 1.0)
                            lit_per_raw = lit_val / unit_inc
                            _prior_dict[fname] = lit_per_raw * _temp_scaler.scale_[fidx]
                    if _prior_dict:
                        gwr_model.physics_priors = _prior_dict
                        print("   DAG-informed physics priors (scaled units):")
                        for k, v in _prior_dict.items():
                            print(f"      {k:30s}  prior={v:+.6f}")
                except Exception as e:
                    print(f"   Warning: Could not build physics priors: {e}")

                gwr_model.fit(X_full_df, y_full, coords_full,
                             extract_coefficients=True,
                             output_path=gwr_coef_output_path)
                save_blob_path(gwr_model, gwr_path, stage="2", artifact_id="gwr_model_full",
                               producer="enhanced_spatial_cv")
                print(f"[OK] Saved gwr_model_full.pkl")
                print(f"[OK] Saved MGWR local coefficients to {gwr_coef_output_path}")

            spatial_intel_dir = os.path.join(paths.output_dir, "spatial_intelligence")

            gwrf_path = os.path.join(full_models_dir, "gwrf_model_full.pkl")
            # Canonical output: Stage_2_Spatial_CV/gwrf_pdp/ (fallback to legacy spatial_intelligence/)
            gwrf_curves_dir = os.path.join(paths.stage2_dir, "gwrf_pdp")
            gwrf_curves_path = os.path.join(gwrf_curves_dir, "gwrf_condition_curves.json")
            # Also check legacy location
            legacy_curves_path = os.path.join(spatial_intel_dir, "gwrf_pdp", "gwrf_condition_curves.json")
            if exists_path(gwrf_path, stage="2", artifact_id="gwrf_model_full"):
                print("\n--- GWRF: already trained (found gwrf_model_full) -- skipping ---")
                gwrf_model = load_blob_path(gwrf_path, stage="2", artifact_id="gwrf_model_full")
                if gwrf_model is None:
                    gwrf_model = joblib.load(gwrf_path)
                models[2] = gwrf_model
            else:
                print("\n--- Training GWRF on full dataset ---")
                gwrf_model = models[2]
                gwrf_model.fit(X_full, y_full, coords_full,
                              extract_derivatives=False,
                              feature_names=selected_features_2b)  # UNSCALED
                save_blob_path(gwrf_model, gwrf_path, stage="2", artifact_id="gwrf_model_full",
                               producer="enhanced_spatial_cv")
                print(f"[OK] Saved gwrf_model_full.pkl")

            # Export GWRF condition curves (consumed by Stage 4) -- skip if already done
            skip_pdp = True  # GWRF PDP condition curves not consumed downstream
            if skip_pdp:
                print("\n--- GWRF condition curves: SKIPPED (skip_pdp=true) ---")
            elif os.path.exists(gwrf_curves_path) or os.path.exists(legacy_curves_path):
                print("\n--- GWRF condition curves: already exported -- skipping ---")
            else:
                print("\n--- Extracting GWRF condition curves (PDP + saturation) ---")
                ensure_dir(gwrf_curves_dir)
                try:
                    condition_curves = gwrf_model.export_condition_curves(
                        X=X_full,
                        output_dir=gwrf_curves_dir,
                        variable_names=selected_features_2b,
                        curve_type='logistic',
                        save_plots=True,
                    )
                    print(f"[OK] GWRF condition curves exported: {len(condition_curves)} variables")
                except Exception as e:
                    print(f"[WARN] GWRF condition curve extraction failed: {e}")
                    print("   Stage 4 will fall back to coefficient-based extrapolation.")

            ggpgam_path = os.path.join(full_models_dir, "ggpgam_model_full.pkl")
            cached_ggp = load_blob_path(ggpgam_path, stage="2", artifact_id="ggpgam_model_full")
            if cached_ggp is not None:
                print("\n--- GGPGAM: already trained -- skipping ---")
                ggpgam_model = cached_ggp
                models[3] = ggpgam_model
            else:
                print("\n--- Training GGPGAM on full dataset ---")
                ggpgam_model = models[3]
                ggpgam_model.fit(X_full, y_full, coords_full,
                                extract_derivatives=False,
                                output_dir=spatial_intel_dir)  # UNSCALED
                save_blob_path(ggpgam_model, ggpgam_path, stage="2", artifact_id="ggpgam_model_full",
                               producer="enhanced_spatial_cv")
                print(f"[OK] Saved ggpgam_model_full.pkl")
            
            # Verification: Quick R² check on training data
            print("\n--- Verification: Full Dataset R² Scores ---")
            for model, model_name in zip(models, model_names_2b):
                try:
                    # Generate predictions (pass UNSCALED features!)
                    if model_name in ['gwr', 'gwrf', 'ggpgam']:
                        preds = model.predict(X_full, coords_full)
                    else:
                        preds = model.predict(X_full)
                    
                    # Handle tuple returns (e.g., GWRF returns (preds, uncertainty))
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    
                    # Calculate R²
                    r2 = r2_score(y_full, preds)
                    rmse = np.sqrt(mean_squared_error(y_full, preds))
                    pred_range = f"[{preds.min():.1f}, {preds.max():.1f}]"
                    print(f"{model_name.upper():8s} - R²: {r2:.4f}, RMSE: {rmse:.4f}, Range: {pred_range}")
                except Exception as e:
                    print(f"{model_name.upper():8s} - Verification failed: {e}")
            
            print(f"\n✅ Stage 2b Complete: All base models retrained and saved to {full_models_dir}/")
            print(f"These models can now be used for scenario predictions and deployment.")

            # Collect full-model fitted values for surrogate pretraining
            # These are in-sample predictions (not OOF) — surrogates should
            # learn to reproduce the base model's full fitted surface.
            base_fitted_values = {}
            for model, model_name in zip(models, model_names_2b):
                try:
                    if model_name in ('gwr', 'gwrf', 'ggpgam'):
                        preds = model.predict(X_full, coords_full)
                    else:
                        preds = model.predict(X_full)
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    base_fitted_values[model_name] = preds
                    print(f"Collected fitted values for {model_name}: shape {preds.shape}")
                except Exception as e:
                    print(f"WARNING: Could not collect fitted values for {model_name}: {e}")
        
        # ============================================================================
        # End of full retrain
        # ============================================================================
        
        # Load hyperparameters from project config
        print("\n=== Loading Hyperparameters ===")
        cfg = cv_system.base_config  # keep raw dict for hyperparams not yet in StageConfig
        
        # Compute base OOF residuals (needed by both cached and fresh paths)
        base_oof_residuals = {}
        if base_predictions:
            print("Computing OOF residuals from base model predictions...")
            for model_name in base_model_names:
                if model_name in base_predictions:
                    residuals = y - base_predictions[model_name]
                    base_oof_residuals[model_name] = residuals
                    print(f"  {model_name}: residual mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}")
        else:
            print("No base model predictions available (Stage 2 skipped)")

        stage2_dir = str(cv_system.paths.stage2_dir)

        # ================================================================
        # Neural Meta-Learner
        # ================================================================
        v2_neural_result = None
        best_approach = "V2 Neural"
        print("\n=== Neural Meta-Learner ===")
        try:
            # Ensure the neural training logger is at INFO level so epoch
            # updates are emitted.  Do NOT call basicConfig here — it would
            # add a duplicate handler when running under the server's
            # _EventCapture (stream.py already bridges logging → capture).
            import logging as _logging
            _logging.getLogger("sparc.run.v2_neural_training").setLevel(_logging.INFO)
            from sparc.run.v2_neural_training import train_neural_meta, run_cma_es_search

            # Optional CMA-ES hyperparameter search
            if cfg.get("optimization", {}).get("run_cma_es", False):
                print("Running CMA-ES hyperparameter search (this may take a while)...")
                best_hparams = run_cma_es_search(
                    y=y,
                    coords=coords,
                    feature_matrix=X_original_features,
                    feature_names=selected_features_filtered,
                    folds=folds,
                    config=cfg,
                    output_dir=stage2_dir,
                )
                # Merge best hyperparams into config
                for k, v in best_hparams.items():
                    if k.startswith("lambda_"):
                        cfg.setdefault("training", {})[k] = v
                    elif k == "learning_rate":
                        cfg.setdefault("training", {})["learning_rate"] = v
                    elif k == "dropout":
                        cfg.setdefault("models", {}).setdefault("neural", {})["dropout"] = v
                print(f"CMA-ES best params applied: {best_hparams}")

            # Build base-model fitted values dict for surrogate pretraining
            # Use full-model fitted values (Stage 2b) instead of OOF predictions
            _base_oof = {}
            for _mn in ('gwr', 'gwrf', 'ggpgam'):
                if _mn in base_fitted_values:
                    fv = base_fitted_values[_mn]
                    if len(fv) == len(y):
                        _base_oof[_mn] = fv
                    else:
                        print(f"WARNING: base_fitted_values[{_mn}] length {len(fv)} "
                              f"!= y length {len(y)} — skipping (row mismatch)")
            v2_neural_result = train_neural_meta(
                y=y,
                coords=coords,
                feature_matrix=X_original_features,
                feature_names=selected_features_filtered,
                folds=folds,
                config=cfg,
                output_dir=stage2_dir,
                base_oof_predictions=_base_oof or None,
            )

            v2_r2 = v2_neural_result["metrics"]["r2"]
            v2_rmse = v2_neural_result["metrics"]["rmse"]
            print(f"\nV2 Neural Meta-Learner: R² = {v2_r2:.4f}, RMSE = {v2_rmse:.4f}")

            best_meta_predictions = v2_neural_result["oof_predictions"]
            best_meta_r2 = v2_r2
            best_meta_rmse = v2_rmse

            # --- Residual analysis + Moran's I ---
            v2_oof_preds = v2_neural_result["oof_predictions"]
            v2_residuals = y - v2_oof_preds
            print(f"\nV2 Neural Residuals - Mean: {np.mean(v2_residuals):.4f}, "
                  f"Std: {np.std(v2_residuals):.4f}, "
                  f"Range: [{np.min(v2_residuals):.4f}, {np.max(v2_residuals):.4f}]")

            # --- Uncertainty validation ---
            v2_oof_unc = v2_neural_result.get("oof_uncertainty")
            if v2_oof_unc is not None:
                v2_unc = np.asarray(v2_oof_unc)
                n_neg = int(np.sum(v2_unc < 0))
                n_nan = int(np.sum(np.isnan(v2_unc)))
                n_inf = int(np.sum(np.isinf(v2_unc)))
                print(f"V2 Uncertainty - Mean: {np.mean(v2_unc):.4f}, "
                      f"Std: {np.std(v2_unc):.4f}, "
                      f"Range: [{np.min(v2_unc):.4f}, {np.max(v2_unc):.4f}]")
                if n_neg or n_nan or n_inf:
                    print(f"[WARNING] V2 uncertainty has {n_neg} negative, "
                          f"{n_nan} NaN, {n_inf} Inf values")
                else:
                    print("[OK] V2 uncertainty values are all valid (non-negative, finite)")
            else:
                print("[WARNING] V2 Neural Meta-Learner did not return uncertainty estimates")

        except Exception as e:
            print(f"[WARNING] V2 Neural Meta-Learner failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use average of base model predictions
            if base_predictions:
                best_meta_predictions = np.mean(
                    [base_predictions[m] for m in base_model_names], axis=0
                )
            else:
                best_meta_predictions = np.full(len(y), np.mean(y))
            best_meta_r2 = r2_score(y, best_meta_predictions)
            best_meta_rmse = np.sqrt(mean_squared_error(y, best_meta_predictions))
            best_approach = "Base Average (fallback)"

        # Compute residuals for the neural meta-learner
        best_meta_residuals = y - best_meta_predictions
        print(f"Best Meta Residuals ({best_approach}) - Mean: {np.mean(best_meta_residuals):.4f}, Std: {np.std(best_meta_residuals):.4f}")
        
        # Generate Laplacian eigenmaps early to ensure they get saved
        print("\n=== Generating Enhanced Laplacian Eigenmaps (150 components) ===")
        
        def generate_enhanced_laplacian_features(coords, n_components=150):
            """Generate enhanced Laplacian eigenmap features with proper scaling"""
            try:
                from sparc.features.laplacian import LaplacianEigenmap
                from sklearn.preprocessing import StandardScaler, PolynomialFeatures
                from sklearn.neighbors import NearestNeighbors
                
                print(f"Computing enhanced Laplacian eigenmaps for {len(coords)} samples...")
                print(f"Target components: {n_components}")
                
                # Scale coordinates first
                coord_scaler = StandardScaler()
                coords_scaled = coord_scaler.fit_transform(coords)
                
                # Adaptive k_neighbors based on data size and target components
                k_neighbors = min(max(20, n_components // 5), len(coords) // 10)
                print(f"Using k_neighbors: {k_neighbors}")
                
                laplacian = LaplacianEigenmap(
                    n_components=n_components,
                    k_neighbors=k_neighbors,
                    target_crs=None
                )
                
                start_time = time.time()
                X_laplacian = laplacian.fit_transform(coords_scaled)  # Use scaled coordinates
                computation_time = time.time() - start_time
                
                print(f"Laplacian computation completed in {computation_time:.2f}s")
                print(f"Generated {X_laplacian.shape[1]} Laplacian eigenmap features")
                
                # Scale Laplacian features
                laplacian_scaler = StandardScaler()
                X_laplacian_scaled = laplacian_scaler.fit_transform(X_laplacian)
                
                # Add polynomial coordinate features for richer spatial representation
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                coords_poly = poly.fit_transform(coords_scaled)
                
                # Add local spatial context features
                nbrs = NearestNeighbors(n_neighbors=min(10, len(coords) // 20)).fit(coords_scaled)
                distances, indices = nbrs.kneighbors(coords_scaled)
                local_density = distances.mean(axis=1).reshape(-1, 1)
                
                # Combine all enhanced features
                enhanced_features = np.column_stack([
                    coords_scaled,           # Scaled coordinates (2 features)
                    coords_poly,            # Polynomial coordinates (more features)
                    X_laplacian_scaled,     # All 150 Laplacian features (no PCA reduction!)
                    local_density           # Local spatial density (1 feature)
                ])
                
                print(f"Enhanced feature matrix shape: {enhanced_features.shape}")
                print(f"Features breakdown:")
                print(f"  - Scaled coordinates: {coords_scaled.shape[1]}")
                print(f"  - Polynomial coordinates: {coords_poly.shape[1]}")
                print(f"  - Laplacian eigenmaps: {X_laplacian_scaled.shape[1]} (NO PCA reduction)")
                print(f"  - Local density: {local_density.shape[1]}")
                print(f"  - Total features: {enhanced_features.shape[1]}")
                
                return enhanced_features, {
                    'laplacian': laplacian,
                    'coord_scaler': coord_scaler,
                    'laplacian_scaler': laplacian_scaler,
                    'poly': poly,
                    'nbrs': nbrs
                }
                
            except Exception as e:
                print(f"Error generating enhanced Laplacian features: {e}")
                print("Using enhanced PCA fallback...")
                
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler, PolynomialFeatures
                
                # Scale coordinates
                coord_scaler = StandardScaler()
                coords_scaled = coord_scaler.fit_transform(coords)
                
                # Generate more PCA components as fallback
                max_components = min(n_components, coords.shape[0] - 1)
                pca_coord = PCA(n_components=max_components)
                X_pca = pca_coord.fit_transform(coords_scaled)
                
                # Add polynomial features
                poly = PolynomialFeatures(degree=2, include_bias=False)
                coords_poly = poly.fit_transform(coords_scaled)
                
                # Combine features
                enhanced_features = np.column_stack([coords_scaled, coords_poly, X_pca])
                
                print(f"Generated {enhanced_features.shape[1]} enhanced PCA features as fallback")
                
                return enhanced_features, {
                    'laplacian': pca_coord,
                    'coord_scaler': coord_scaler,
                    'laplacian_scaler': StandardScaler().fit(X_pca),
                    'poly': poly,
                    'nbrs': None
                }
        
        # Laplacian eigenmaps are superseded by sinusoidal spatial encoding in the
        # neural meta-learner and are not consumed by any downstream stage.
        # Skip by default; set compute_laplacian_features: true in config to re-enable.
        _compute_laplacian = cv_system._cfg.raw.get(
            'pipeline_execution', {}
        ).get('compute_laplacian_features', False)
        if _compute_laplacian:
            print("=== Generating Enhanced Laplacian Eigenmaps (150 components) ===")
            X_laplacian_full, feature_transformers = generate_enhanced_laplacian_features(coords, n_components=150)
            paths = get_paths()
            stage2_dir = paths.stage2_dir
            save_blob_path(feature_transformers,
                           os.path.join(stage2_dir, 'enhanced_spatial_features.pkl'),
                           stage="2", artifact_id="enhanced_spatial_features",
                           producer="enhanced_spatial_cv")
            print(f"Saved Laplacian artifacts to {stage2_dir}/")
        else:
            print("=== Skipping Laplacian Eigenmaps (superseded by sinusoidal encoding) ===")
        
        # Check spatial autocorrelation in meta-ensemble residuals
        print("\n=== Spatial Autocorrelation Analysis of Meta-Ensemble Residuals ===")
        
        # Estimate sensible distance for spatial weights
        autocorr_range = estimate_spatial_autocorrelation_range(coords, best_meta_residuals)
        print(f"Estimated autocorrelation range: {autocorr_range:.0f}m")
        
        # Create spatial weights matrix (this can be slow for large datasets)
        print(f"Creating spatial weights matrix for {len(coords)} points with {autocorr_range:.0f}m threshold...")
        print("This may take a moment for large datasets...")
        
        # Optimize for large datasets by using a more reasonable threshold
        max_reasonable_threshold = min(autocorr_range, 2000)  # Cap at 2km
        
        if max_reasonable_threshold < autocorr_range:
            print(f"Reducing threshold from {autocorr_range:.0f}m to {max_reasonable_threshold:.0f}m for computational efficiency")
        
        w = DistanceBand(coords, threshold=max_reasonable_threshold, binary=True, silence_warnings=True)
        print(f"Spatial weights matrix created successfully")
        
        # Compute Moran's I
        print("Computing Moran's I statistic...")
        moran = Moran(best_meta_residuals, w, two_tailed=False)
        print(f"Best meta ({best_approach}) residuals Moran I = {moran.I:.3f}, p-value = {moran.p_sim:.3f}")
        print(f"Spatial weight threshold used: {max_reasonable_threshold:.0f}m")
        
        # Deep Kriging has been retired — final predictions are from the neural meta-learner
        final_predictions = best_meta_predictions
        
        # Calculate final performance
        final_r2 = r2_score(y, final_predictions)
        final_rmse = np.sqrt(mean_squared_error(y, final_predictions))
        
        print(f"\n=== Final Results ===")
        print(f"Base Models Best (GWRF):       R² = {stage2_results['performance']['individual_models']['gwrf']['r2']:.4f}") # pyright: ignore[reportPossiblyUnboundVariable]
        if v2_neural_result is not None:
            print(f"V2 Neural Meta-Learner:        R² = {v2_neural_result['metrics']['r2']:.4f}, RMSE = {v2_neural_result['metrics']['rmse']:.4f}")
        else:
            print(f"V2 Neural Meta-Learner:        FAILED (using base average fallback)")
        print(f"Final Ensemble ({best_approach}): R² = {final_r2:.4f}, RMSE = {final_rmse:.4f}")
        
        # Detailed improvement analysis
        base_gwrf_r2 = stage2_results['performance']['individual_models']['gwrf']['r2'] # pyright: ignore[reportPossiblyUnboundVariable]
        best_meta_improvement = best_meta_r2 - base_gwrf_r2
        
        print(f"\nDetailed Performance Analysis:")
        print(f"Neural Meta vs Base GWRF:      {best_meta_improvement:+.4f} R² points")
        
        # Simple spatial performance summary
        print(f"\n=== Spatial Performance Summary ===")
        print(f"Neural Meta Residuals Moran's I: {moran.I:.3f} (p = {moran.p_sim:.3f})")
        if moran.I > 0.3:
            print("[WARNING] High spatial autocorrelation detected - consider spatial regularization")
        elif moran.I > 0.1:
            print("[WARNING] Moderate spatial autocorrelation - monitoring recommended") 
        else:
            print("[OK] Low spatial autocorrelation - good spatial performance")
        
        # Save final results
        final_results = {
            'base_models': stage2_results['performance']['individual_models'], # pyright: ignore[reportPossiblyUnboundVariable]
            'v2_neural_meta': {
                'r2': v2_neural_result['metrics']['r2'] if v2_neural_result else None,
                'rmse': v2_neural_result['metrics']['rmse'] if v2_neural_result else None,
                'executed': v2_neural_result is not None,
            },
            'meta_ensemble_best': {'r2': best_meta_r2, 'rmse': best_meta_rmse, 'approach': best_approach},
            'spatial_autocorrelation': {
                'moran_i': moran.I,
                'p_value': moran.p_sim,
                'autocorr_range': autocorr_range,
            },
            'final_ensemble': {'r2': final_r2, 'rmse': final_rmse},
            'improvements': {
                'best_meta_vs_base': best_meta_improvement,
            }
        }
        
        paths = get_paths()
        results_file = paths.stage2_dir / 'final_ensemble_results.json'
        predictions_file = paths.stage2_dir / 'final_ensemble_predictions.csv'

        # Save final predictions
        final_results_df = pd.DataFrame({
            'OBJECTID': joined_data.index, # pyright: ignore[reportPossiblyUnboundVariable]
            'actual': y,
            'best_meta_approach': best_approach,
            'final_ensemble': final_predictions,
        })

        # Add V2 Neural columns if it ran
        if v2_neural_result is not None:
            final_results_df['v2_neural_predictions'] = v2_neural_result['oof_predictions']
            final_results_df['v2_neural_residuals'] = y - v2_neural_result['oof_predictions']
            v2_unc_out = v2_neural_result.get('oof_uncertainty')
            if v2_unc_out is not None:
                final_results_df['v2_neural_uncertainty'] = np.asarray(v2_unc_out)

        # Persist to artifacts.db (single source of truth).
        wrote_results_to_store = False
        wrote_preds_to_store = False
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None:
                _store.write_struct("2", "ensemble_results", final_results,
                                    producer="enhanced_spatial_cv",
                                    consumers=["server:/results/model_performance"])
                wrote_results_to_store = True
                _store.write_table("2", "ensemble_predictions", final_results_df,
                                   producer="enhanced_spatial_cv")
                wrote_preds_to_store = True
        except Exception as _e:
            print(f"  [WARNING] Could not write ensemble outputs to store: {_e}")

        if not wrote_results_to_store:
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
        if not wrote_preds_to_store:
            final_results_df.to_csv(predictions_file, index=False)

        if not wrote_results_to_store or not wrote_preds_to_store:
            try:
                from sparc.registry.run_registry import register_path
                if not wrote_results_to_store:
                    register_path(results_file, stage="2", artifact_id="ensemble_results",
                                  format="json", producer="enhanced_spatial_cv",
                                  consumers=["server:/results/model_performance"])
                if not wrote_preds_to_store:
                    register_path(predictions_file, stage="2", artifact_id="ensemble_predictions",
                                  format="csv", producer="enhanced_spatial_cv")
            except Exception:
                pass

        print(f"\nResults saved to:")
        print(f"  - final_ensemble_results.json")
        print(f"  - final_ensemble_predictions.csv")
        
        # Save final composite scaler for the neural meta-learner
        print(f"\n=== Saving Final Model Artifacts ===")
        
        # Extract the scalers that were created during training
        final_scaler_components = {
            'approach': best_approach,
            'feature_names': selected_features,
            'coordinate_columns': cv_system._cfg.coordinates,
            'model_performance': {
                'r2': best_meta_r2,
                'rmse': best_meta_rmse,
                'approach': best_approach
            },
            'training_info': {
                'n_samples': len(y),
                'n_features': len(selected_features),
                'cv_folds': len(folds)
            }
        }
        
        # Save the composite scaler
        scaler_path = paths.stage2_dir / 'final_meta_ensemble_scaler.pkl'
        save_blob_path(final_scaler_components, scaler_path, stage="2",
                       artifact_id="final_meta_ensemble_scaler",
                       producer="enhanced_spatial_cv",
                       consumers=["stage4:scenario"])
        
        print(f"  Final scaler saved to: {scaler_path}")
        print(f"   - Model approach: {best_approach}")
        print(f"   - Performance: R² = {best_meta_r2:.4f}, RMSE = {best_meta_rmse:.4f}")
        
        # Save the V2 neural model checkpoint if available
        if v2_neural_result is not None:
            final_model_path = paths.stage2_dir / 'final_meta_ensemble.pkl'
            save_blob_path(v2_neural_result, final_model_path, stage="2",
                           artifact_id="final_meta_ensemble",
                           producer="enhanced_spatial_cv",
                           consumers=["stage4:scenario"])
            print(f"  V2 Neural model saved to: {final_model_path}")
        
        # Print comprehensive summary of all saved artifacts
        print("\n" + "="*80)
        print("📦 COMPLETE MODEL ARTIFACT SUMMARY")
        print("="*80)
        
        print("\n🔷 BASE MODELS (Stage 2b):")
        print(f"   └─ {os.path.join(full_models_dir, 'ols_model_full.pkl')}") # pyright: ignore[reportPossiblyUnboundVariable]
        print(f"   └─ {os.path.join(full_models_dir, 'gwr_model_full.pkl')}") # pyright: ignore[reportPossiblyUnboundVariable]
        print(f"   └─ {os.path.join(full_models_dir, 'gwrf_model_full.pkl')}") # pyright: ignore[reportPossiblyUnboundVariable]
        print(f"   └─ {os.path.join(full_models_dir, 'ggpgam_model_full.pkl')}") # pyright: ignore[reportPossiblyUnboundVariable]
        
        print("\n🔷 META-LEARNER (Stage 3):")
        if v2_neural_result is not None:
            print(f"   └─ V2 Neural: {final_model_path}")
        
        print("\n🔷 SCALERS & PREPROCESSING:")
        print(f"   ├─ Feature Scaler: {os.path.join(stage2_dir, 'feature_scaler.pkl')}")
        print(f"   └─ Meta Scaler: {scaler_path}")
        
        print("\n🔷 PREDICTIONS & RESULTS:")
        print(f"   ├─ OOF Predictions: {os.path.join(stage2_dir, 'optimized_oof_predictions.csv')}")
        print(f"   ├─ Final Predictions: {os.path.join(stage2_dir, 'final_ensemble_predictions.csv')}")
        print(f"   └─ Spatial Folds: {str(cv_system.paths.folds_file)}")
        
        print("\n🎯 FOR SCENARIO PREDICTION, YOU NEED:")
        print(f"   1. Base Models: {full_models_dir}/")
        print(f"   2. Feature Scaler: {os.path.join(stage2_dir, 'feature_scaler.pkl')}")
        print(f"   3. Final Meta-Ensemble: {final_model_path}")
        print(f"   4. Final Meta Scaler: {scaler_path}")
        
        print("\n" + "="*80)
        
        # Consolidate sensitivity package
        print("\n" + "="*80)
        print("CONSOLIDATING SPATIAL SENSITIVITY PACKAGE")
        print("="*80)
        
        def consolidate_sensitivity_package(spatial_intel_dir, selected_features, stage2_dir):
            """
            Merge all derivative surfaces and create sensitivity package JSON.
            """
            import glob
            
            sensitivity_package = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'variables': selected_features,
                'physics_signs': {
                    "Pct_Impervious": +1,
                    "Pct_Canopy": -1,
                    "NDVI": -1,
                    "Albedo": -1
                },
                'models': {},
                'files': {}
            }
            
            # Collect MGWR coefficients
            mgwr_raw = os.path.join(stage2_dir, 'base_models_full/mgwr_local_coefficients.csv')
            mgwr_corr = os.path.join(stage2_dir, 'base_models_full/mgwr_local_coefficients_corrected.csv')
            if os.path.exists(mgwr_raw):
                sensitivity_package['files']['mgwr_raw'] = mgwr_raw
                sensitivity_package['files']['mgwr_corrected'] = mgwr_corr
                sensitivity_package['models']['mgwr'] = {
                    'description': 'Local coefficient surfaces from MGWR',
                    'raw_file': mgwr_raw,
                    'corrected_file': mgwr_corr
                }
            
            # Collect GWRF derivatives
            gwrf_pdp_files = glob.glob(os.path.join(spatial_intel_dir, 'gwrf_pdp/*.csv'))
            gwrf_deriv_files = glob.glob(os.path.join(spatial_intel_dir, 'gwrf_derivatives/*.csv'))
            if gwrf_deriv_files:
                sensitivity_package['files']['gwrf_pdp'] = {
                    os.path.basename(f): f for f in gwrf_pdp_files
                }
                sensitivity_package['files']['gwrf_derivatives'] = {
                    os.path.basename(f): f for f in gwrf_deriv_files
                }
                sensitivity_package['models']['gwrf'] = {
                    'description': 'ICE curves and saturation derivatives from GWRF',
                    'n_pdp_curves': len(gwrf_pdp_files),
                    'n_derivatives': len(gwrf_deriv_files)
                }
            
            # Collect GWRF condition curves (aggregate PDP + saturation fits for Stage 4)
            condition_curves_path = os.path.join(spatial_intel_dir, 'gwrf_pdp', 'gwrf_condition_curves.json')
            if os.path.exists(condition_curves_path):
                sensitivity_package['files']['gwrf_condition_curves'] = condition_curves_path
                sensitivity_package['models'].setdefault('gwrf', {})
                sensitivity_package['models']['gwrf']['condition_curves'] = condition_curves_path
                sensitivity_package['models']['gwrf']['condition_curves_description'] = (
                    'Aggregate PDP with logistic saturation curve fits — used by Stage 4 '
                    'for saturation-aware scenario predictions'
                )
            
            # Collect GGPGAM derivatives
            ggpgam_files = glob.glob(os.path.join(spatial_intel_dir, 'ggpgam_derivatives/*.csv'))
            if ggpgam_files:
                sensitivity_package['files']['ggpgam_derivatives'] = {
                    os.path.basename(f): f for f in ggpgam_files
                }
                sensitivity_package['models']['ggpgam'] = {
                    'description': 'Partial derivatives from GAM splines',
                    'n_derivatives': len(ggpgam_files)
                }
            
            # Collect Meta-ensemble derivatives
            meta_pdp_files = glob.glob(os.path.join(spatial_intel_dir, 'meta_pdp/*.csv'))
            meta_shap_files = glob.glob(os.path.join(spatial_intel_dir, 'meta_shap_derivatives/*.csv'))
            if meta_pdp_files:
                sensitivity_package['files']['meta_pdp'] = {
                    os.path.basename(f): f for f in meta_pdp_files
                }
                sensitivity_package['models']['meta_ensemble'] = {
                    'description': 'PDP curves from neural meta-learner',
                    'n_pdp_curves': len(meta_pdp_files)
                }
            if meta_shap_files:
                sensitivity_package['files']['meta_shap'] = {
                    os.path.basename(f): f for f in meta_shap_files
                }
            
            # Collect Laplacian derivatives
            laplacian_files = glob.glob(os.path.join(spatial_intel_dir, 'laplacian_derivatives/*.csv'))
            if laplacian_files:
                # Separate eigen-dimension and variable-projected files
                eigen_files = [f for f in laplacian_files if '_z' in os.path.basename(f)]
                var_files = [f for f in laplacian_files if '_z' not in os.path.basename(f)]
                
                sensitivity_package['files']['laplacian_derivatives'] = {
                    os.path.basename(f): f for f in laplacian_files
                }
                sensitivity_package['models']['laplacian'] = {
                    'description': 'Manifold gradients on Laplacian eigenmap',
                    'n_eigen_derivatives': len(eigen_files),
                    'n_variable_derivatives': len(var_files)
                }
            
            # Save package manifest
            package_path = os.path.join(spatial_intel_dir, 'sensitivity_package.json')
            save_struct_path(sensitivity_package, package_path, stage="2",
                             artifact_id="sensitivity_package",
                             producer="enhanced_spatial_cv",
                             consumers=["stage4:scenario", "report"])
            
            # Print summary
            total_derivatives = len(gwrf_deriv_files) + len(ggpgam_files) + len(meta_pdp_files) + len(laplacian_files)
            print(f"\n✅ Sensitivity package consolidated: {package_path}")
            print(f"   📊 Total derivative surfaces: {total_derivatives}")
            print(f"   📂 Model contributions:")
            for model_name, model_info in sensitivity_package['models'].items():
                print(f"      • {model_name}: {model_info['description']}")
            
            return sensitivity_package
        
        try:
            sensitivity_package = consolidate_sensitivity_package(
                spatial_intel_dir=spatial_intel_dir,
                selected_features=selected_features,
                stage2_dir=stage2_dir
            )
        except Exception as e:
            print(f"Warning: Could not consolidate sensitivity package: {e}")
            sensitivity_package = None

        # ----------------------------------------------------------------
        # ANP uncertainty surface — few-shot inference using the OOF
        # predictions as calibration context.  Produces anp_uncertainty.json
        # with per-point mean and uncertainty estimates.
        # ----------------------------------------------------------------
        try:
            from sparc.inference.few_shot import few_shot_predict
            from sparc.data.satellite_types import SatelliteFeatureSet
            import numpy as _np_anp

            _coords_np = self._coords if hasattr(self, "_coords") else final_results.get("coords")
            _feat_np = self._feature_matrix if hasattr(self, "_feature_matrix") else None

            if _coords_np is not None and _feat_np is not None:
                _N = len(_coords_np)
                _feat_np = _np_anp.asarray(_feat_np, dtype=_np_anp.float32)
                _coords_np = _np_anp.asarray(_coords_np, dtype=_np_anp.float32)

                _feats_obj = SatelliteFeatureSet(coords=_coords_np)

                # Use OOF predictions (normalised back to raw space) as calibration y
                _oof_pred = final_results.get("oof_predictions")
                _y_mean = final_results.get("y_mean", 0.0)
                _y_std = final_results.get("y_std", 1.0)
                if _oof_pred is not None:
                    _calib_y = _np_anp.asarray(_oof_pred, dtype=_np_anp.float32) * _y_std + _y_mean
                    _anp_result = few_shot_predict(
                        features=_feats_obj,
                        calibration_X=_feat_np,
                        calibration_y=_calib_y,
                        calibration_coords=_coords_np,
                    )
                    import json as _json_anp
                    _anp_payload = {
                        "mean": _anp_result.y_pred.tolist(),
                        "uncertainty": _anp_result.uncertainty.tolist(),
                        "n_calibration": _anp_result.n_calibration,
                        "note": "SpatialANP uncertainty surface (untrained prior — improve by loading trunk checkpoint)",
                    }
                    _anp_path = os.path.join(stage2_dir, "anp_uncertainty.json")
                    with open(_anp_path, "w") as _fh:
                        _json_anp.dump(_anp_payload, _fh)
                    try:
                        save_struct_path(_anp_payload, _anp_path, stage="2",
                                         artifact_id="anp_uncertainty",
                                         producer="enhanced_spatial_cv",
                                         consumers=["stage4:scenario", "report"])
                    except Exception:
                        pass
                    print(f"✅ ANP uncertainty surface written: {_anp_path}")
        except Exception as _anp_exc:
            print(f"Warning: ANP uncertainty surface skipped: {_anp_exc}")

        print("\n" + "="*80)
        
        return final_results
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()