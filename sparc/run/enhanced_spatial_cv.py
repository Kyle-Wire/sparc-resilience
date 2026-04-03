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
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from sparc.run.pipeline_paths import get_paths
from sparc.run.spatial_autocorr_comprehensive import SpatialAutocorrelationAnalyzer
from sparc.run.memory_efficient_spatial_analysis import analyze_model_residuals_morans_i

# Enhanced hardware optimization settings for high-performance workstations (CPU-only)
HARDWARE_CONFIG = {
    'cpu_cores': mp.cpu_count(),  # Use all available cores
    'max_workers': min(mp.cpu_count(), 24),  # Allow more workers for I/O tasks
    'memory_limit_gb': min(psutil.virtual_memory().total // (1024**3) * 0.75, 48),  # Use 75% of available RAM, cap at 48GB
    'batch_size_large': 4096,  # Larger batch sizes for abundant RAM
    'batch_size_medium': 2048,
    'batch_size_small': 1024,
    'parallel_cv': True,  # Enable parallel cross-validation
    'high_memory_mode': psutil.virtual_memory().total >= 32 * (1024**3),  # Auto-detect high memory systems (32GB+)
    'enable_aggressive_optimization': True,  # Enable all speed optimizations
    'laplacian_batch_size': 2000,  # Batch size for Laplacian computations
    'max_eigen_iterations': 1000,  # Maximum iterations for eigendecomposition
    'spatial_cluster_threshold': 1000,  # Use spatial clustering for datasets larger than this
    # Per-model parallelism control (NEW APPROACH)
    'per_model_parallel': True,  # Enable per-model parallelization
    'outer_jobs': min(mp.cpu_count(), 6),  # Number of parallel folds per model
    'inner_jobs': max(1, mp.cpu_count() // 6),  # Jobs per individual model instance
    'gwrf_local_jobs': 1,  # Single thread per local RF to allow outer parallelization
}

print(f"Hardware-optimized configuration detected (CPU-only):")
print(f"  - CPU cores: {HARDWARE_CONFIG['cpu_cores']}")
print(f"  - Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"  - Memory limit: {HARDWARE_CONFIG['memory_limit_gb']} GB")
print(f"  - Parallel processing: Enabled (per-model parallelization)")
print(f"  - Per-model parallelism: {HARDWARE_CONFIG['outer_jobs']} folds x {HARDWARE_CONFIG['inner_jobs']} inner jobs")
print(f"  - High-memory mode: {'Enabled' if HARDWARE_CONFIG['high_memory_mode'] else 'Disabled'}")
print(f"  - Strategy: Each model runs in parallel across all folds for optimal resource utilization")

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
from sparc.models.meta_ensemble import MetaEnsemble
from sparc.models.deep_kriging_v2 import DeepKrigingV2
from sparc.evaluation.evaluation import SpatialEvaluator
from copy import deepcopy

# Shared evaluator instance for benchmark metrics
_evaluator = SpatialEvaluator()

# Global flag for OOF extraction (set to True to enable spatial intelligence extraction)
# Requires oof_extraction_hooks module — disable until implemented
EXTRACT_OOF_INTELLIGENCE = False

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
    
    # Runtime safety checks for spatial models
    try:
        n_train = len(train_idx)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        
        if model_name == 'gwr':
            # Check if using MGWR (variable_bandwidths) or single bandwidth
            if hasattr(model_copy, 'variable_bandwidths') and model_copy.variable_bandwidths:
                # MGWR with distance-based variable bandwidths (in meters)
                # These are spatial distances, not neighbor counts - no adjustment needed
                # But ensure min_points is reasonable for the fold size
                if hasattr(model_copy, 'min_points'):
                    max_min_points = max(n_features + 2, min(50, n_train // 10))
                    if model_copy.min_points > max_min_points:
                        old_min = model_copy.min_points
                        model_copy.min_points = max_min_points
                        print(f"INFO: MGWR fold {fold_idx}: Adjusted min_points {old_min} -> {model_copy.min_points}")
            elif hasattr(model_copy, 'bandwidth') and model_copy.bandwidth:
                # Single global bandwidth (neighbor count)
                min_safe_bandwidth = max(n_features + 5, 20)
                max_safe_bandwidth = int(n_train * 0.5)
                
                if model_copy.bandwidth >= n_train:
                    safe_bandwidth = max(min_safe_bandwidth, min(int(n_train * 0.3), 500))
                    print(f"WARNING: GWR bandwidth ({model_copy.bandwidth}) >= training size ({n_train}) for fold {fold_idx}")
                    print(f"         Adjusting to safe bandwidth: {safe_bandwidth}")
                    model_copy.bandwidth = safe_bandwidth
                elif model_copy.bandwidth > max_safe_bandwidth:
                    safe_bandwidth = max(min_safe_bandwidth, min(max_safe_bandwidth, model_copy.bandwidth))
                    print(f"INFO: GWR bandwidth ({model_copy.bandwidth}) > 50% of training ({max_safe_bandwidth}) for fold {fold_idx}")
                    print(f"      Capping bandwidth at: {safe_bandwidth}")
                    model_copy.bandwidth = safe_bandwidth
                
                # Also ensure min_points is reasonable
                if hasattr(model_copy, 'min_points'):
                    if model_copy.min_points > n_train // 2:
                        model_copy.min_points = max(n_features + 2, n_train // 4)
                        print(f"INFO: Adjusted GWR min_points to {model_copy.min_points}")
                
        elif model_name == 'gwrf' and hasattr(model_copy, 'k_neighbors'):
            if model_copy.k_neighbors >= n_train:
                print(f"WARNING: GWRF k_neighbors ({model_copy.k_neighbors}) >= training size ({n_train}) for fold {fold_idx}")
                # Adjust k_neighbors to safe value
                safe_k_neighbors = max(3, int(n_train * 0.8))
                print(f"         Adjusting to safe k_neighbors: {safe_k_neighbors}")
                model_copy.k_neighbors = safe_k_neighbors
                
            # Additional check for subsample_n parameter
            if hasattr(model_copy, 'subsample_n') and model_copy.subsample_n is not None:
                if model_copy.subsample_n >= n_train:
                    print(f"WARNING: GWRF subsample_n ({model_copy.subsample_n}) >= training size ({n_train}) for fold {fold_idx}")
                    safe_subsample = max(100, int(n_train * 0.65))  # Conservative subsample
                    print(f"         Adjusting to safe subsample_n: {safe_subsample}")
                    model_copy.subsample_n = safe_subsample
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
                # Import extraction hooks
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interventions'))
                from oof_extraction_hooks import extract_from_fitted_model
                
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

def calculate_fold_spatial_autocorr(residuals, coords, threshold=1000):
    """
    Calculate Moran's I for a single fold's residuals using K-NN weights.
    Returns a simple spatial autocorrelation metric.
    """
    try:
        if len(residuals) < 10:  # Skip tiny folds
            return 0.0
            
        from sklearn.neighbors import NearestNeighbors
        k_neighbors = min(8, len(residuals) - 1)
        
        if k_neighbors < 1:
            return 0.0
            
        # Use K-NN weights for consistency
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Simple Moran's I calculation
        n = len(residuals)
        y_mean = np.mean(residuals)
        y_centered = residuals - y_mean
        
        # Create weights matrix (row-standardized)
        total_similarity = 0.0
        total_weight = 0.0
        
        for i in range(n):
            for j in range(1, k_neighbors+1):  # Skip self
                if j < len(indices[i]):
                    neighbor_idx = indices[i, j]
                    weight = 1.0 / k_neighbors  # Equal weights for neighbors
                    total_similarity += weight * y_centered[i] * y_centered[neighbor_idx]
                    total_weight += weight
        
        if total_weight > 0 and np.sum(y_centered**2) > 0:
            morans_i = (n / total_weight) * (total_similarity / np.sum(y_centered**2))
            return morans_i
        else:
            return 0.0
            
    except Exception as e:
        # Silently return 0 on errors to avoid breaking the pipeline
        return 0.0

def estimate_spatial_autocorrelation_range(coords, y, max_distance=None, n_bins=20, sample_size=5000):
    """
    Estimate spatial autocorrelation range using Moran's I over increasing distance bands.
    Uses downsampled spatial weights for memory efficiency.
    """
    if max_distance is None:
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        max_distance = np.linalg.norm(bounds[1] - bounds[0]) * 0.3
    print("Estimating spatial autocorrelation range...")
    
    # Downsample to avoid memory overload (seeded for reproducibility)
    if len(coords) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(coords), sample_size, replace=False)
        coords = coords[idx]
        y = y[idx]

    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    moran_i_values = []

    for i in range(len(bins) - 1):
        try:
            # Create spatial weight matrix within band
            w = DistanceBand(coords, threshold=bins[i+1], binary=True, silence_warnings=True)
            if len(w.neighbors) < 10:
                moran_i_values.append(0)
                continue

            moran = Moran(y, w, two_tailed=False)
            moran_i_values.append(moran.I)
        except Exception:
            moran_i_values.append(0)
    
    # Find distance where autocorrelation drops below threshold
    threshold = 0.1
    for i, mi in enumerate(moran_i_values):
        if abs(mi) < threshold:
            print(f"Autocorrelation drops at ~{bin_centers[i]:.0f}m (Moran's I = {mi:.2f})")
            return bin_centers[i]
    
    print(f"No clear drop detected — defaulting to max: {max_distance}m")
    return max_distance

def spatial_kfold_enhanced(X, y, coords, n_splits=5, block_size=None, buffer_size=0, method='block', stratify_y=True):
    """
    Generate stratified spatial cross-validation folds with optional buffer.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target variable
    coords : np.ndarray, shape (n_samples, 2)
        Spatial coordinates
    n_splits : int
        Number of CV folds
    block_size : float, optional
        Block size in meters. If None, estimated from spatial autocorrelation
    buffer_size : float, optional
        Buffer distance in meters. Training samples within this distance of
        any test block are excluded from training. Default is 0 (no buffer).
    method : str
        'block' for spatial blocks, 'kmeans' for clustering
    stratify_y : bool
        Whether to stratify by y distribution
        
    Returns:
    --------
    list of tuples : (train_idx, test_idx) for each fold
    """
    n_samples = len(X)
    
    # Estimate block size if not provided
    if block_size is None:
        print("Determining optimal block size from target variable autocorrelation...")
        
        try:
            from sparc.run.spatial_autocorr_comprehensive import SpatialAutocorrelationAnalyzer
            # Use a reasonable max_distance proportional to the spatial extent
            bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
            spatial_extent = np.linalg.norm(bounds[1] - bounds[0])
            max_dist = spatial_extent * 0.4
            analyzer = SpatialAutocorrelationAnalyzer(coords, max_distance=max_dist)
            correlogram_results = analyzer.compute_correlogram(
                y, 
                plot=False,
                title="Block Size Selection Correlogram"
            )
            
            # Use first-zero-crossing of the target variable
            block_size = correlogram_results.get('first_zero_crossing',
                                                  correlogram_results['optimal_block_size'])
            print(f"Target variable zero-crossing: {block_size:.0f}m")
            
            # Cap at spatial_extent / (2 * n_splits) to keep folds viable
            max_viable = spatial_extent / (2.0 * n_splits)
            if block_size > max_viable:
                print(f"Capping block size: {block_size:.0f}m > extent/(2*{n_splits}) = {max_viable:.0f}m")
                block_size = max_viable
            
            # Floor
            block_size = max(block_size, 500)
                
        except Exception as e:
            print(f"Correlogram analysis failed: {e}")
            print("Falling back to traditional autocorrelation range estimation...")
            autocorr_range = estimate_spatial_autocorrelation_range(coords, y)
            block_size = max(autocorr_range * 2, 500)
        
        print(f"Final block size: {block_size:.0f}m")
    
    if method == 'block':
        # Create spatial blocks
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        n_blocks_x = int(np.ceil((bounds[1, 0] - bounds[0, 0]) / block_size))
        n_blocks_y = int(np.ceil((bounds[1, 1] - bounds[0, 1]) / block_size))
        
        # Assign points to blocks
        block_assignments = np.zeros(n_samples, dtype=int)
        for i, (x, y_coord) in enumerate(coords):
            block_x = int((x - bounds[0, 0]) / block_size)
            block_y = int((y_coord - bounds[0, 1]) / block_size)
            block_assignments[i] = block_x * n_blocks_y + block_y
        
        # Create folds from blocks (seeded for reproducibility)
        unique_blocks = np.unique(block_assignments)
        rng = np.random.RandomState(42)
        rng.shuffle(unique_blocks)
        
        if stratify_y:
            # Stratify blocks by y distribution
            block_means = [y[block_assignments == block].mean() for block in unique_blocks]
            sorted_blocks = [x for _, x in sorted(zip(block_means, unique_blocks))]
        else:
            sorted_blocks = unique_blocks
        
        # Distribute blocks across folds
        folds = []
        for fold_idx in range(n_splits):
            fold_blocks = sorted_blocks[fold_idx::n_splits]
            test_mask = np.isin(block_assignments, fold_blocks)
            test_idx = np.where(test_mask)[0]

            if buffer_size > 0:
                # Compute distance from every point to nearest test point
                from scipy.spatial import cKDTree
                print(f"Applying {buffer_size}m buffer to fold {fold_idx + 1}")
                tree = cKDTree(coords[test_idx])
                dist, _ = tree.query(coords, k=1)
                buffer_mask = dist <= buffer_size
                
                # Count excluded training points
                excluded_count = np.sum(buffer_mask & ~test_mask)
                print(f"  Excluded {excluded_count} training points within buffer zone")
            else:
                buffer_mask = np.zeros(len(coords), dtype=bool)

            train_idx = np.where(~(test_mask | buffer_mask))[0]
            folds.append((train_idx, test_idx))
    
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        
        # Cluster coordinates
        kmeans = KMeans(n_clusters=n_splits, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Create folds from clusters
        folds = []
        for fold_idx in range(n_splits):
            test_mask = cluster_labels == fold_idx
            test_idx = np.where(test_mask)[0]
            
            if buffer_size > 0:
                # Compute distance from every point to nearest test point
                from scipy.spatial import cKDTree
                print(f"Applying {buffer_size}m buffer to fold {fold_idx + 1}")
                tree = cKDTree(coords[test_idx])
                dist, _ = tree.query(coords, k=1)
                buffer_mask = dist <= buffer_size
                
                # Count excluded training points
                excluded_count = np.sum(buffer_mask & ~test_mask)
                print(f"  Excluded {excluded_count} training points within buffer zone")
            else:
                buffer_mask = np.zeros(len(coords), dtype=bool)
            
            train_idx = np.where(~(test_mask | buffer_mask))[0]
            folds.append((train_idx, test_idx))
    
    else:
        raise ValueError("Method must be 'block' or 'kmeans'")
    
    # Validate fold sizes
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        if len(test_idx) < 50:
            print(f"Warning: Fold {i+1} has very small test set ({len(test_idx)} samples)")
    
    return folds



class EnhancedSpatialCV:
    """
    Enhanced Spatial Cross-Validation using variable-specific optimized parameters
    """
    
    def __init__(self, pipeline_config_path=None):
        # Use centralized path management
        self.paths = get_paths()
        
        # Use provided path or default from paths utility
        if pipeline_config_path is None:
            self.pipeline_config_path = str(self.paths.pipeline_config)
        elif not os.path.isabs(pipeline_config_path):
            self.pipeline_config_path = str(self.paths.run_dir / pipeline_config_path)
        else:
            self.pipeline_config_path = pipeline_config_path
            
        self.pipeline_config = self.load_pipeline_config()
        self.base_config = load_config()
        # Centralised physics constraints from project.yml (or legacy defaults)
        self._monotone_constraints = load_monotone_constraints(self.base_config)
        
    def load_pipeline_config(self):
        """Load the optimized pipeline configuration"""
        if not os.path.exists(self.pipeline_config_path):
            raise FileNotFoundError(f"Pipeline config not found: {self.pipeline_config_path}")
        
        with open(self.pipeline_config_path, 'r') as f:
            return json.load(f)
    
    def get_block_size_from_config(self):
        """
        Get the block size from pipeline configuration
        """
        # Try to get from sparc.models.spatial_cv first
        block_size = self.pipeline_config.get('models', {}).get('spatial_cv', {}).get('block_size', None)
        if block_size is None:
            # Fallback to manual_parameters
            block_size = self.pipeline_config.get('manual_parameters', {}).get('block_size', None)
        
        return block_size
    
    def get_buffer_size_from_config(self):
        """
        Get the buffer size from pipeline configuration
        """
        buffer_size = self.pipeline_config.get('models', {}).get('spatial_cv', {}).get('buffer_size', 0)
        
        # Auto-calculate buffer based on block size if enabled
        if self.pipeline_config.get('models', {}).get('spatial_cv', {}).get('buffer_size_auto', False):
            block_size = self.get_block_size_from_config()
            if block_size is not None:
                # Use 1/3 of block size as buffer (common spatial CV rule of thumb)
                auto_buffer = max(100, int(block_size / 3))
                print(f"Auto-calculated buffer size: {auto_buffer}m (1/3 of block size {block_size}m)")
                return auto_buffer
        
        return buffer_size if buffer_size is not None else 0
    
    def get_variable_bandwidths(self):
        """
        Get variable-specific bandwidths from pipeline configuration
        
        Returns:
        --------
        dict or None: Dictionary mapping variable names to bandwidths, or None if not configured
        """
        variable_bandwidths = self.pipeline_config.get('manual_parameters', {}).get('bandwidths', None)
        
        if variable_bandwidths:
            # Ensure all values are numeric
            processed_bandwidths = {}
            for var, bandwidth in variable_bandwidths.items():
                try:
                    processed_bandwidths[var] = float(bandwidth)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid bandwidth value for {var}: {bandwidth}. Skipping.")
                    continue
            
            return processed_bandwidths if processed_bandwidths else None
        
        return None
    
    def apply_profiler_overrides(self, profiler_recommendations):
        """
        Overlay DatasetProfiler-recommended hyperparameters onto the
        pipeline_config in-memory so that ``create_optimized_models()``
        picks them up automatically.

        Parameters
        ----------
        profiler_recommendations : dict
            Output of ``DatasetProfiler.recommend_parameters()``.  Keys are
            model names (``gwrf``, ``ggpgam``, ``meta_ensemble``, etc.).
        """
        models_section = self.pipeline_config.setdefault('models', {})
        for model_key, recs in profiler_recommendations.items():
            if model_key in ('correlogram', 'spatial_cv'):
                # spatial_cv is a top-level key in models
                if model_key == 'spatial_cv' and 'spatial_cv' in models_section:
                    models_section['spatial_cv'].update(recs)
                continue
            if model_key in models_section:
                # deep-merge one level for dict values (e.g. meta_learner_params)
                for k, v in recs.items():
                    if isinstance(v, dict) and isinstance(models_section[model_key].get(k), dict):
                        models_section[model_key][k].update(v)
                    else:
                        models_section[model_key][k] = v
        # Deep kriging lives at its own top-level key
        if 'deep_kriging' in profiler_recommendations:
            dk_section = self.pipeline_config.setdefault('deep_kriging', {})
            dk_section.update(profiler_recommendations['deep_kriging'])

        print("Applied DatasetProfiler adaptive overrides to pipeline config.")

    def create_optimized_models(self, n_samples=None):
        """
        Create base models with global hyperparameters from pipeline configuration
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples in the dataset (for safety checks)
        """
        model_configs = self.pipeline_config['models']
        
        models = []
        
        # OLS (no configurable parameters - it's a baseline model)
        models.append(OLSModel())
        
        # GWR with variable-specific bandwidths
        if 'gwr' in model_configs:
            gwr_params = model_configs['gwr'].copy()
            
            # Get variable-specific bandwidths from manual_parameters
            variable_bandwidths = self.get_variable_bandwidths()
            
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
                    config_bandwidth = int(gwr_params.get('bandwidth', 500))
                    adaptive_bandwidth = min(config_bandwidth, min_fold_size)
                    gwr_params['bandwidth'] = max(adaptive_bandwidth, 50)
                    
                    print(f"GWR Global Bandwidth (fallback):")
                    print(f"  Dataset size: {n_samples}")
                    print(f"  Config bandwidth: {config_bandwidth}")
                    print(f"  Final bandwidth: {gwr_params['bandwidth']}")
            
            # Extract only parameters supported by GWRModel constructor
            gwr_constructor_params = {}
            valid_params = ['bandwidth', 'variable_bandwidths', 'kernel', 'coords_cols', 'alpha', 'min_points']
            for param in valid_params:
                if param in gwr_params:
                    gwr_constructor_params[param] = gwr_params[param]
            # Constrained regression is disabled during CV to avoid systematic
            # bias; sign constraints are applied post-hoc for interpretation only.
            gwr_constructor_params['use_constrained_regression'] = False
            gwr_constructor_params['sign_constraints'] = self._monotone_constraints
            
            try:
                models.append(GWRModel(**gwr_constructor_params))
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
                gwrf_params['n_jobs'] = HARDWARE_CONFIG.get('gwrf_local_jobs', 1)
                
                print(f"GWRF Global Parameters:")
                print(f"  Dataset size: {n_samples}")
                print(f"  Config k_neighbors: {config_k_neighbors}")
                print(f"  Final k_neighbors: {gwrf_params['k_neighbors']}")
            
            # Extract only parameters supported by GWRFModel constructor
            gwrf_constructor_params = {}
            valid_params = ['n_estimators', 'k_neighbors', 'min_samples_leaf', 'n_jobs', 
                          'subsample_fraction', 'subsample_n']
            for param in valid_params:
                if param in gwrf_params:
                    gwrf_constructor_params[param] = gwrf_params[param]
            
            try:
                gwrf = GWRFModel(**gwrf_constructor_params)
                # Inject physics sign constraints from centralised config
                gwrf.physics_signs = self._monotone_constraints
                models.append(gwrf)
            except Exception as e:
                print(f"Warning: Failed to create GWRF with configured params: {e}")
                models.append(GWRFModel(n_estimators=50, k_neighbors=100, min_samples_leaf=5, n_jobs=1))
                
        # GGPGAM with global hyperparameters (adaptive via DatasetProfiler)
        if 'ggpgam' in model_configs:
            ggpgam_params = model_configs['ggpgam'].copy()
            
            # Remove parameters that aren't part of GGPGAM_SVC constructor
            ggpgam_constructor_params = {}
            valid_params = ['n_splines', 'n_spatial_bases', 'lam']
            for param in valid_params:
                if param in ggpgam_params:
                    ggpgam_constructor_params[param] = ggpgam_params[param]

            print(f"GGPGAM Parameters (adaptive):")
            for k, v in ggpgam_constructor_params.items():
                print(f"  {k}: {v}")
            
            try:
                ggpgam = GGPGAM_SVC(**ggpgam_constructor_params)
                # Inject physics sign constraints from centralised config
                ggpgam.physics_signs = self._monotone_constraints
                models.append(ggpgam)
            except Exception as e:
                print(f"Warning: Failed to create GGPGAM with configured params: {e}")
                models.append(GGPGAM_SVC())
        else:
            ggpgam = GGPGAM_SVC()
            ggpgam.physics_signs = self._monotone_constraints
            models.append(ggpgam)
        
        # Set output CRS on all models for GeoPackage export
        _out_crs = self.base_config.get('crs', {}).get('target_projected', 'EPSG:26919')
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
        feature_names : list, optional
            List of feature names for OOF extraction. If None, will be extracted from X if it's a DataFrame.
        """
        # Extract feature names if not provided
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        models = self.create_optimized_models(n_samples=len(X))
        model_names = ['ols', 'gwr', 'gwrf', 'ggpgam']
        
        n_samples = len(y)
        n_models = len(models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        print(f"Generating OOF predictions with {n_models} optimized models...")
        print(f"Hardware acceleration: {HARDWARE_CONFIG['max_workers']} cores, {HARDWARE_CONFIG['memory_limit_gb']}GB RAM limit")
        print(f"Strategy: Per-model parallelization - each model runs all {len(folds)} folds in parallel")
        
        # Run each model in parallel across all folds
        if HARDWARE_CONFIG['parallel_cv'] and len(folds) >= 2:
            print(f"\n{'='*80}")
            print(f"PARALLEL MODE: Each fold will be processed in parallel (per model)")
            print(f"  • Models will be processed sequentially: {', '.join(model_names)}")
            print(f"  • Each model's {len(folds)} folds will run in parallel")
            print(f"  • Max parallel workers per model: {min(len(folds), HARDWARE_CONFIG['outer_jobs'])}")
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
        # Save OOF predictions
        oof_df = pd.DataFrame(oof_predictions, columns=model_names)
        oof_df.to_csv(os.path.join(output_dir, 'optimized_oof_predictions.csv'), index=False)
        
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
                    oof_predictions[valid_mask == False, i] = mean_pred
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
                bm_cfg = getattr(self, 'base_config', {}).get('benchmark_metrics', {}) if hasattr(self, 'base_config') else {}
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
        if HARDWARE_CONFIG['high_memory_mode']:
            base_subsample = min(n_samples, 3000)
        else:
            base_subsample = min(n_samples, 1500)
        
        # Adjust based on k_neighbors - need enough samples for meaningful spatial analysis
        min_subsample = max(k_neighbors * 5, 500)  # At least 10x k_neighbors
        
        # Adjust based on available memory
        memory_adjusted = min(base_subsample, HARDWARE_CONFIG['memory_limit_gb'] * 50)
        
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
            HARDWARE_CONFIG['outer_jobs'],  # Controlled outer parallelization
            max(1, HARDWARE_CONFIG['memory_limit_gb'] // 4)  # More memory per worker for model-level parallelization
        )
        
        print(f"Per-model parallel CV configuration:")
        print(f"  - Workers per model: {optimal_workers}")
        print(f"  - Memory per worker: ~{HARDWARE_CONFIG['memory_limit_gb'] // optimal_workers}GB")
        print(f"  - Models to process: {n_models} ({', '.join(model_names)})")
        
        # Train each model across all folds in parallel
        for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
            print(f"\n=== Training {model_name} across all folds in parallel ===")
            
            # Prepare fold data for this specific model
            fold_data_list = [
                (i, train_idx, test_idx, X, y, coords, model, model_name, feature_names) 
                for i, (train_idx, test_idx) in enumerate(folds)
            ]
            
            try:
                # Use ProcessPoolExecutor for CPU-intensive tasks with timeout
                with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                    with tqdm(total=len(folds), desc=f"{model_name} CV Folds", unit="fold") as pbar:
                        # Submit all fold tasks for this model
                        future_to_fold = {
                            executor.submit(train_single_model_fold_worker, fold_data): i 
                            for i, fold_data in enumerate(fold_data_list)
                        }
                        
                        completed_folds = 0
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
        
        # Check if OOF predictions already exist
        oof_predictions_path = str(self.paths.oof_predictions)
        folds_path = str(self.paths.folds_file)
        
        if os.path.exists(oof_predictions_path) and os.path.exists(folds_path):
            print("=== Stage 2: Loading existing OOF predictions ===")
            print(f"Found existing OOF predictions at: {self.paths.get_relative_path(oof_predictions_path)}")
            print(f"Found existing folds at: {self.paths.get_relative_path(folds_path)}")
            
            # Load existing results
            oof_df = pd.read_csv(oof_predictions_path)
            model_names = oof_df.columns.tolist()
            oof_predictions = oof_df.values
            
            # Load existing folds
            folds = joblib.load(folds_path)
            
            # Load data for performance calculation
            print("=== Loading Data for Performance Calculation ===")
            selected_features = self.pipeline_config['features']['selected_features']
            
            data = load_and_preprocess_data(
                raw_data_path=self.base_config["paths"]["raw_csv_path"],
                identifier_col=self.base_config['variables']['identifier'],
                target_col=self.base_config['variables']['target'],
                coords_cols=self.base_config['variables']['coordinates'],
                predictor_cols=selected_features,
                initial_crs=self.base_config['crs']['initial'],
                target_crs=self.base_config['crs']['target_projected'],
                output_dir=self.base_config.get('output', {}).get('base_dir'),
            )
            
            available_features = data.columns.tolist()
            selected_features = [f for f in selected_features if f in available_features]
            feature_names = selected_features
            
            y = data[self.base_config['variables']['target']].values
            
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
                    bm_cfg = self.base_config.get('benchmark_metrics', {})
                    if bm_cfg.get('enabled', False):
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
            
            print(f"\n=== Stage 2 Complete (loaded existing results) ===")
            print(f"Results loaded from: {stage2_dir}/")
            
            return {
                'oof_predictions': oof_predictions,
                'feature_names': feature_names,
                'performance': performance_summary
            }
        
        # If results don't exist, run the full pipeline
        print("=== Stage 2: Running full pipeline ===")
        
        # Load and preprocess data
        print("=== Loading and Preprocessing Data ===")
        selected_features = self.pipeline_config['features']['selected_features']
        
        # GUARD: never allow the target variable into the feature matrix
        target_col = self.base_config['variables']['target']
        if target_col in selected_features:
            print(f"[WARNING] Target variable '{target_col}' found in selected_features — removing to prevent data leakage!")
            selected_features = [f for f in selected_features if f != target_col]
        
        data = load_and_preprocess_data(
            raw_data_path=self.base_config["paths"]["raw_csv_path"],
            identifier_col=self.base_config['variables']['identifier'],
            target_col=self.base_config['variables']['target'],
            coords_cols=self.base_config['variables']['coordinates'],
            predictor_cols=selected_features,
            initial_crs=self.base_config['crs']['initial'],
            target_crs=self.base_config['crs']['target_projected'],
            output_dir=self.base_config.get('output', {}).get('base_dir'),
        )
        
        # Filter features to only those available
        available_features = data.columns.tolist()
        selected_features = [f for f in selected_features if f in available_features]
        
        # ── DatasetProfiler: adapt model hyper-parameters to this dataset ──
        try:
            from sparc.run.dataset_profiler import DatasetProfiler
            profiler = DatasetProfiler(
                data,
                coord_cols=self.base_config['variables']['coordinates'],
                feature_cols=selected_features,
            )
            print(profiler.summary())
            recs = profiler.recommend_parameters()
            # Only apply profiler overrides if explicitly enabled in config;
            # by default, honour the user-specified hyperparameters.
            if self.base_config.get('pipeline', {}).get('use_dataset_profiler', False):
                self.apply_profiler_overrides(recs)
                print("DatasetProfiler overrides applied.")
            else:
                print("DatasetProfiler: profiling only (overrides disabled). "
                      "Set pipeline.use_dataset_profiler: true to enable.")
            # Persist profile for downstream stages (scenarios, report)
            import json as _json
            profile_path = os.path.join(stage2_dir, 'dataset_profile.json')
            with open(profile_path, 'w') as _fp:
                _json.dump(profiler.profile(), _fp, indent=2)
        except Exception as _e:
            print(f"Warning: DatasetProfiler unavailable ({_e}). Using pipeline_config defaults.")
        
        X_gwen = data[selected_features].values
        y = data[self.base_config['variables']['target']].values
        coords = data[self.base_config['variables']['coordinates']].values
        
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
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"Feature info saved to: {feature_info_path}")
        
        # Use UNSCALED features for modeling (models scale internally)
        X_augmented = X_gwen  # NOT X_gwen_scaled!
        feature_names = selected_features
        
        # Generate spatial folds
        print("\n=== Generating Spatial Folds ===")
        folds = spatial_kfold_enhanced(
            X=X_augmented,
            y=y,
            coords=coords,
            n_splits=5,
            block_size=self.get_block_size_from_config(),
            buffer_size=self.get_buffer_size_from_config(),
            method='block',
            stratify_y=True
        )
        
        # Generate OOF predictions
        print("\n=== Base Models ===")
        oof_predictions, model_names = self.generate_optimized_oof_predictions(
            X_augmented, y, coords, folds, stage2_dir, feature_names
        )
        
        # Save folds for later use in Stage 3
        joblib.dump(folds, self.paths.folds_file)
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
            bm_cfg = self.base_config.get('benchmark_metrics', {})
            if bm_cfg.get('enabled', False):
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
        
        print(f"\n=== Stage 2 Complete ===")
        print(f"Results saved to: {stage2_dir}/")
        
        return {
            'oof_predictions': oof_predictions,
            'feature_names': feature_names,
            'performance': performance_summary
        }

def main(fast_mode=False):
    """
    Main function to run the complete enhanced spatial cross-validation pipeline
    with integrated Meta Ensemble and Deep Kriging
    """
    try:
        # Initialize the enhanced spatial CV system
        cv_system = EnhancedSpatialCV()
        
        # Run Stage 2: Base models with OOF predictions
        print("\n=== Stage 2: Base Models ===")
        stage2_results = cv_system.run_enhanced_spatial_cv()
        
        # Load and merge data by ID
        print("\n=== Loading and Merging Data by ID ===")
        
        # Load raw data and process it the same way as in Stage 2
        raw_data = pd.read_csv(cv_system.base_config["paths"]["raw_csv_path"])
        
        # Process the data the same way as Stage 2 to get correct coordinates
        selected_features = cv_system.pipeline_config['features']['selected_features']
        
        data = load_and_preprocess_data(
            raw_data_path=cv_system.base_config["paths"]["raw_csv_path"],
            identifier_col=cv_system.base_config['variables']['identifier'],
            target_col=cv_system.base_config['variables']['target'],
            coords_cols=cv_system.base_config['variables']['coordinates'],
            predictor_cols=selected_features,
            initial_crs=cv_system.base_config['crs']['initial'],
            target_crs=cv_system.base_config['crs']['target_projected'],
            output_dir=cv_system.base_config.get('output', {}).get('base_dir'),
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
        # in the MetaEnsemble — these are physical variables whose direction matters
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
        joblib.dump(feature_scaler, scaler_path)
        print(f"Created and saved feature scaler to: {scaler_path}")
        print(f"  Scaler fitted on {len(selected_features_filtered)} features")
        
        # Replace features in 'data' with scaled ones (for meta-ensemble only!)
        for i, feature_name in enumerate(selected_features_filtered):
            data[feature_name] = X_features_scaled[:, i]
        
        print(f"Applied feature scaling to 'data' for meta-ensemble training")
        print(f"  Note: 'data_unscaled' still contains original values for base models")
        
        # Load OOF predictions
        oof_predictions = pd.read_csv(str(cv_system.paths.oof_predictions))
        
        # Get the identifier column name from config
        id_col = cv_system.base_config['variables']['identifier']
        
        # Add ID column to OOF predictions (assuming they're in same order as processed data)
        if id_col not in oof_predictions.columns:
            # Check if ID is a column or the index
            if id_col in data.columns:
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
        target_col = cv_system.base_config['variables']['target']
        coords = joined_data[['projected_X', 'projected_Y']].values
        y = joined_data[target_col].values
        base_model_names = ['ols', 'gwr', 'gwrf', 'ggpgam']
        
        # Extract base model predictions by name
        base_predictions = {}
        for model_name in base_model_names:
            base_predictions[model_name] = joined_data[model_name].values
        
        # Load saved folds
        print("\n=== Loading Spatial Folds ===")
        try:
            folds_path = str(cv_system.paths.folds_file)
            folds = joblib.load(folds_path)
            print(f"Loaded {len(folds)} spatial folds from {cv_system.paths.get_relative_path(folds_path)}")
        except FileNotFoundError:
            print(f"{cv_system.paths.get_relative_path(folds_path)} not found, this should have been created in Stage 2")
            return None
        
        # Define spatial_intel_dir early so it's available for all stages
        paths = get_paths()
        spatial_intel_dir = os.path.join(paths.output_dir, "spatial_intelligence")
        os.makedirs(spatial_intel_dir, exist_ok=True)
        
        # ============================================================================
        # Stage 2b: Retrain Base Models on Full Dataset for Deployment
        # ============================================================================
        # Check if Stage 2b should be skipped based on pipeline config
        skip_stage_2b = cv_system.pipeline_config.get('pipeline_execution', {}).get('skip_stage_2b_full_retrain', False)
        
        if skip_stage_2b:
            print("\n=== Stage 2b: SKIPPED (skip_stage_2b_full_retrain=true in pipeline_config.json) ===")
            print("   To enable full model retraining, set skip_stage_2b_full_retrain to false")
            full_models_dir = None  # Set to None to indicate skipped
        else:
            print("\n=== Stage 2b: Retraining Base Models on Full Dataset for Deployment ===")
            
            # Create output directory for full models
            paths = get_paths()
            full_models_dir = os.path.join(paths.stage2_dir, "base_models_full")
            os.makedirs(full_models_dir, exist_ok=True)
            print(f"Full models will be saved to: {full_models_dir}")
        
            # Prepare full dataset using same preprocessing as Stage 2
            print("Preparing full dataset with UNSCALED features for base model retraining...")
            
            # CRITICAL: Use data_unscaled (saved before meta-ensemble scaling!)
            # Get selected features
            selected_features_2b = cv_system.pipeline_config['features']['selected_features']
            available_features_2b = data_unscaled.columns.tolist()
            selected_features_2b = [f for f in selected_features_2b if f in available_features_2b]
            
            # Extract UNSCALED features, target, and coordinates
            X_full_df = data_unscaled[selected_features_2b]          # DataFrame (keeps column names for GWR)
            X_full = X_full_df.values                                 # ndarray for other models
            y_full = data_unscaled[cv_system.base_config['variables']['target']].values
            coords_full = data_unscaled[cv_system.base_config['variables']['coordinates']].values
            
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
            if os.path.exists(ols_path):
                print("\n--- OLS: already trained (found ols_model_full.pkl) -- skipping ---")
                ols_model = joblib.load(ols_path)
                models[0] = ols_model
            else:
                print("\n--- Training OLS on full dataset ---")
                ols_model = models[0]
                ols_model.fit(X_full, y_full)  # UNSCALED!
                joblib.dump(ols_model, ols_path)
                print(f"[OK] Saved ols_model_full.pkl")
            
            gwr_path = os.path.join(full_models_dir, "gwr_model_full.pkl")
            gwr_coef_output_path = os.path.join(full_models_dir, "mgwr_local_coefficients.csv")
            if os.path.exists(gwr_path) and os.path.exists(gwr_coef_output_path):
                print("\n--- GWR: already trained (found gwr_model_full.pkl) -- skipping ---")
                gwr_model = joblib.load(gwr_path)
                models[1] = gwr_model
            else:
                print("\n--- Training GWR on full dataset ---")
                gwr_model = models[1]

                # Build DAG-informed physics priors for prior-centered ridge regression.
                try:
                    from sparc.config.config import load_physics_priors
                    _physics_raw = load_physics_priors(cv_system.base_config)
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
                joblib.dump(gwr_model, gwr_path)
                print(f"[OK] Saved gwr_model_full.pkl")
                print(f"[OK] Saved MGWR local coefficients to {gwr_coef_output_path}")

            spatial_intel_dir = os.path.join(paths.output_dir, "spatial_intelligence")

            gwrf_path = os.path.join(full_models_dir, "gwrf_model_full.pkl")
            gwrf_curves_path = os.path.join(spatial_intel_dir, "gwrf_pdp", "gwrf_condition_curves.json")
            if os.path.exists(gwrf_path):
                print("\n--- GWRF: already trained (found gwrf_model_full.pkl) -- skipping ---")
                gwrf_model = joblib.load(gwrf_path)
                models[2] = gwrf_model
            else:
                print("\n--- Training GWRF on full dataset ---")
                gwrf_model = models[2]
                gwrf_model.fit(X_full, y_full, coords_full,
                              extract_derivatives=False,
                              feature_names=selected_features_2b)  # UNSCALED
                joblib.dump(gwrf_model, gwrf_path)
                print(f"[OK] Saved gwrf_model_full.pkl")

            # Export GWRF condition curves (consumed by Stage 4) -- skip if already done
            skip_pdp = cv_system.base_config.get('pipeline', {}).get('skip_pdp', False)
            if skip_pdp:
                print("\n--- GWRF condition curves: SKIPPED (skip_pdp=true) ---")
            elif os.path.exists(gwrf_curves_path):
                print("\n--- GWRF condition curves: already exported -- skipping ---")
            else:
                print("\n--- Extracting GWRF condition curves (PDP + saturation) ---")
                gwrf_curves_dir = os.path.join(spatial_intel_dir, "gwrf_pdp")
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
            if os.path.exists(ggpgam_path):
                print("\n--- GGPGAM: already trained (found ggpgam_model_full.pkl) -- skipping ---")
                ggpgam_model = joblib.load(ggpgam_path)
                models[3] = ggpgam_model
            else:
                print("\n--- Training GGPGAM on full dataset ---")
                ggpgam_model = models[3]
                # Derivative extraction disabled (not consumed by any downstream stage)
                ggpgam_model.fit(X_full, y_full, coords_full,
                                extract_derivatives=False,
                                output_dir=spatial_intel_dir)  # UNSCALED
                joblib.dump(ggpgam_model, ggpgam_path)
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
        
        # ============================================================================
        # End of Stage 2b
        # ============================================================================
        
        # Load hyperparameters from pipeline config
        print("\n=== Loading Hyperparameters ===")
        config_path = str(cv_system.paths.pipeline_config)
        with open(config_path, "r") as f:
            cfg = json.load(f)
        
        # Extract and filter meta_ensemble parameters
        meta_config = cfg.get("models", {}).get("meta_ensemble", {})
        _default_trials = 5 if fast_mode else 25
        meta_params = {
            "n_trials": meta_config.get("n_optuna_trials", _default_trials) if not fast_mode else min(5, meta_config.get("n_optuna_trials", 5)),
            "val_size": meta_config.get("meta_learner_options", {}).get("validation_split", 0.2),
            "random_state": 42,
            "monotone_constraints": cv_system._monotone_constraints,
        }
        
        # Extract and filter deep_kriging parameters
        dk_config = cfg.get("deep_kriging", {})
        dk_enabled = dk_config.get("enabled", True)
        dk_version = dk_config.get("version", 2)
        
        # Enhanced Deep Kriging parameters for better residual modeling
        dk_params = {
            "hidden_layers": dk_config.get("hidden_layers", [128, 64, 32, 16]),  # Deeper network
            "dropout_rate": dk_config.get("dropout_rate", 0.3),  # Higher dropout for regularization
            "learning_rate": dk_config.get("learning_rate", 0.0005),  # Lower learning rate
            "val_size": dk_config.get("validation_split", 0.15),  # Smaller validation set
            "random_state": 42,
            "batch_size": dk_config.get("batch_size", min(512, len(coords) // 10)),  # Adaptive batch size
            "epochs": dk_config.get("epochs", 200),  # More epochs for better learning
        }
        
        # Handle nested early_stopping patience parameter
        if 'early_stopping' in dk_config and 'patience' in dk_config['early_stopping']:
            dk_params['patience'] = dk_config['early_stopping']['patience']
        else:
            dk_params['patience'] = 30  # More patience for convergence
        
        print(f"Enhanced Deep Kriging parameters: {dk_params}")
        
        # Stage 3: Meta Ensemble
        print("\n=== Stage 3: Meta Ensemble ===")
        meta_model = MetaEnsemble(**meta_params)
        
        # Compute OOF residuals from base model predictions
        print("Computing OOF residuals from base model predictions...")
        base_oof_residuals = {}
        for model_name in base_model_names:
            if model_name in base_predictions:
                residuals = y - base_predictions[model_name]
                base_oof_residuals[model_name] = residuals
                print(f"  {model_name}: residual mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}")
        
        # Generate Meta Ensemble OOF predictions and true OOF residuals
        meta_oof_predictions = np.zeros(len(y))
        meta_oof_residuals = np.zeros(len(y))  # True OOF residuals for Deep Kriging
        
        print(f"Running Meta Ensemble CV with {len(folds)} folds...")
        with tqdm(total=len(folds), desc="Meta Ensemble CV", unit="fold") as pbar:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                # Get training and test data
                base_train = {name: base_predictions[name][train_idx] for name in base_model_names}
                base_test = {name: base_predictions[name][test_idx] for name in base_model_names}
                
                # Get OOF residuals for training (only for the training indices)
                oof_residuals_train = {name: base_oof_residuals[name][train_idx] for name in base_model_names if name in base_oof_residuals}
                
                y_train = y[train_idx]
                y_test = y[test_idx]  # True test values
                coords_train = coords[train_idx]
                coords_test = coords[test_idx]
                
                # Train meta ensemble with OOF residuals + monotonic constraints
                meta_fold = MetaEnsemble(**meta_params)
                meta_fold.fit(base_train, y_train, coords_train, oof_residuals=oof_residuals_train,
                             original_X=X_original_features[train_idx],
                             original_feature_names=selected_features_filtered)
                
                # Predict on test set (no residuals available for prediction)
                fold_predictions = meta_fold.predict(base_test, coords_test,
                                                    original_X=X_original_features[test_idx])
                meta_oof_predictions[test_idx] = fold_predictions
                
                # Calculate true OOF residuals (actual - predicted for held-out data)
                meta_oof_residuals[test_idx] = y_test - fold_predictions
                
                # Calculate spatial autocorrelation for this fold
                fold_residuals = y_test - fold_predictions
                fold_spatial_autocorr = calculate_fold_spatial_autocorr(fold_residuals, coords_test)
                
                # Print spatial performance for monitoring
                if fold_spatial_autocorr > 0.3:  # High spatial autocorrelation
                    print(f"  Fold {fold_idx+1}: High spatial autocorr (Moran's I = {fold_spatial_autocorr:.3f})")
                elif fold_spatial_autocorr > 0.1:  # Moderate spatial autocorrelation  
                    print(f"  Fold {fold_idx+1}: Moderate spatial autocorr (Moran's I = {fold_spatial_autocorr:.3f})")
                else:
                    print(f"  Fold {fold_idx+1}: Low spatial autocorr (Moran's I = {fold_spatial_autocorr:.3f})")
                
                pbar.update(1)
        
        # Calculate Meta Ensemble performance
        meta_r2 = r2_score(y, meta_oof_predictions)
        meta_rmse = np.sqrt(mean_squared_error(y, meta_oof_predictions))
        
        print(f"\nMeta Ensemble Performance: R² = {meta_r2:.4f}, RMSE = {meta_rmse:.4f}")
        
        # Train final meta-model on full dataset for deployment
        print("Training final meta-ensemble model on full dataset with OOF residuals...")
        final_meta_model = MetaEnsemble(**meta_params)
        final_meta_model.fit(base_predictions, y, coords, oof_residuals=base_oof_residuals,
                            original_X=X_original_features,
                            original_feature_names=selected_features_filtered)
        
        # Save final meta-model
        paths = get_paths()
        stage2_dir = paths.stage2_dir
        os.makedirs(stage2_dir, exist_ok=True)
        
        # Save the LightGBM model (LightGBM native format)
        if hasattr(final_meta_model, 'model') and hasattr(final_meta_model.model, 'save_model'):
            model_path = os.path.join(stage2_dir, 'optimized_meta_model_run1.txt')
            final_meta_model.model.save_model(model_path)
            print(f"Saved final meta-model (LightGBM format) to {model_path}")
        else:
            print("Warning: Could not save meta-model - model not available or doesn't support saving")
        
        # ALSO save the complete MetaEnsemble object as .pkl for easy loading
        meta_ensemble_pkl_path = os.path.join(stage2_dir, 'standard_meta_ensemble.pkl')
        joblib.dump(final_meta_model, meta_ensemble_pkl_path)
        print(f"✅ Saved complete MetaEnsemble object to {meta_ensemble_pkl_path}")
        
        # Stage 3b: Second Meta Ensemble Run with Laplacian Eigenmaps
        print("\n=== Stage 3b: Meta Ensemble with Laplacian Eigenmaps ===")
        
        # Generate Laplacian eigenmaps for enhanced spatial features
        print("Generating Laplacian eigenmaps (150 components) for meta-learner...")
        
        def generate_laplacian_features(coords, n_components=150):
            """Generate Laplacian eigenmap features"""
            try:
                from sparc.features.laplacian import LaplacianEigenmap
                from sklearn.preprocessing import StandardScaler
                
                print(f"Computing Laplacian eigenmaps for {len(coords)} samples...")
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
                X_laplacian = laplacian.fit_transform(coords_scaled)
                computation_time = time.time() - start_time
                
                print(f"Laplacian computation completed in {computation_time:.2f}s")
                print(f"Generated {X_laplacian.shape[1]} Laplacian eigenmap features")
                
                # Scale Laplacian features
                laplacian_scaler = StandardScaler()
                X_laplacian_scaled = laplacian_scaler.fit_transform(X_laplacian)
                
                return X_laplacian_scaled, {'laplacian': laplacian, 'coord_scaler': coord_scaler, 'laplacian_scaler': laplacian_scaler}
                
            except Exception as e:
                print(f"Error generating Laplacian features: {e}")
                print("Using PCA fallback...")
                
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                coord_scaler = StandardScaler()
                coords_scaled = coord_scaler.fit_transform(coords)
                
                max_components = min(n_components, coords.shape[0] - 1)
                pca_coord = PCA(n_components=max_components)
                X_pca = pca_coord.fit_transform(coords_scaled)
                
                laplacian_scaler = StandardScaler()
                X_pca_scaled = laplacian_scaler.fit_transform(X_pca)
                
                print(f"Generated {X_pca_scaled.shape[1]} PCA features as fallback")
                
                return X_pca_scaled, {'laplacian': pca_coord, 'coord_scaler': coord_scaler, 'laplacian_scaler': laplacian_scaler}
        
        # Generate Laplacian features (global — used for final model training,
        # but per-fold fold-aware Laplacian is used during CV below)
        X_laplacian, laplacian_transformers = generate_laplacian_features(coords, n_components=150)
        
        # Skip Laplacian derivative extraction - not needed for predictions, only for interpretability
        print("\nSkipping Laplacian manifold derivative extraction (not needed for predictions)")
        
        # Keep base predictions as-is (don't modify them)
        print("Enhanced Meta Ensemble will use:")
        print(f"  - Base model predictions: {list(base_predictions.keys())}")
        print(f"  - Fold-aware Laplacian features: 150 components (recomputed per fold)")
        
        # Generate second Meta Ensemble OOF predictions
        meta_oof_predictions_v2 = np.zeros(len(y))
        meta_oof_residuals_v2 = np.zeros(len(y))
        
        print(f"Running Enhanced Meta Ensemble CV with {len(folds)} folds (fold-aware Laplacian)...")
        with tqdm(total=len(folds), desc="Enhanced Meta Ensemble CV", unit="fold") as pbar:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                # Get training and test data - keep base predictions as 1D arrays
                base_train = {name: base_predictions[name][train_idx] for name in base_model_names}
                base_test = {name: base_predictions[name][test_idx] for name in base_model_names}
                
                # === Fold-aware Laplacian Eigenmaps (prevents spatial leakage) ===
                # Recompute eigenmaps per fold: fit on train coords, Nyström-extend to test
                try:
                    from sparc.features.fold_aware_laplacian import FoldAwareLaplacianEigenmap
                    coords_train_fold = coords[train_idx]
                    coords_test_fold = coords[test_idx]
                    
                    fold_laplacian = FoldAwareLaplacianEigenmap(
                        n_components=150,
                        k_neighbors=min(max(20, 150 // 5), len(train_idx) // 10)
                    )
                    laplacian_train = fold_laplacian.fit_transform(coords_train_fold)
                    laplacian_test = fold_laplacian.transform(coords_test_fold)
                    
                    # Scale Laplacian features
                    from sklearn.preprocessing import StandardScaler as _SS
                    lap_scaler = _SS()
                    laplacian_train = lap_scaler.fit_transform(laplacian_train)
                    laplacian_test = lap_scaler.transform(laplacian_test)

                except Exception as lap_e:
                    print(f"  Fold {fold_idx+1}: Fold-aware Laplacian failed ({lap_e}), using global fallback")
                    laplacian_train = X_laplacian[train_idx]
                    laplacian_test = X_laplacian[test_idx]
                
                # Get OOF residuals for training (from first meta-learner)
                oof_residuals_train = {name: base_oof_residuals[name][train_idx] for name in base_model_names if name in base_oof_residuals}
                
                y_train = y[train_idx]
                y_test = y[test_idx]
                coords_train = coords[train_idx]
                coords_test = coords[test_idx]
                
                # Train enhanced meta ensemble with Laplacian features + monotonic constraints
                meta_fold_v2 = MetaEnsemble(**meta_params)
                meta_fold_v2.fit(base_train, y_train, coords_train, 
                               laplacian_features=laplacian_train, 
                               oof_residuals=oof_residuals_train,
                               original_X=X_original_features[train_idx],
                               original_feature_names=selected_features_filtered)
                
                # Predict on test set with Laplacian features
                fold_predictions_v2 = meta_fold_v2.predict(base_test, coords_test, 
                                                         laplacian_features=laplacian_test,
                                                         original_X=X_original_features[test_idx])
                meta_oof_predictions_v2[test_idx] = fold_predictions_v2
                
                # Calculate residuals
                meta_oof_residuals_v2[test_idx] = y_test - fold_predictions_v2
                
                pbar.update(1)
        
        # Calculate Enhanced Meta Ensemble performance
        meta_r2_v2 = r2_score(y, meta_oof_predictions_v2)
        meta_rmse_v2 = np.sqrt(mean_squared_error(y, meta_oof_predictions_v2))
        
        print(f"\nEnhanced Meta Ensemble Performance: R² = {meta_r2_v2:.4f}, RMSE = {meta_rmse_v2:.4f}")
        
        # Train final enhanced meta-model on full dataset
        print("Training final enhanced meta-ensemble model on full dataset...")

        final_meta_model_v2 = MetaEnsemble(**meta_params)
        final_meta_model_v2.fit(base_predictions, y, coords, 
                               laplacian_features=X_laplacian, 
                               oof_residuals=base_oof_residuals,
                               original_X=X_original_features,
                               original_feature_names=selected_features_filtered)
        
        # Extract meta-ensemble derivatives (PDP curves)
        print("\nExtracting meta-ensemble sensitivity surfaces...")
        final_meta_model_v2.export_meta_derivatives(
            X=data[selected_features].values,
            feature_names=selected_features,
            coords=coords,
            output_dir=spatial_intel_dir
        )
        
        # Save enhanced meta-model
        if hasattr(final_meta_model_v2, 'model') and hasattr(final_meta_model_v2.model, 'save_model'):
            model_path_v2 = os.path.join(stage2_dir, 'optimized_meta_model_run2_laplacian.txt')
            final_meta_model_v2.model.save_model(model_path_v2)
            print(f"Saved enhanced meta-model (LightGBM format) to {model_path_v2}")
        
        # ALSO save the complete enhanced MetaEnsemble object as .pkl for easy loading
        enhanced_meta_ensemble_pkl_path = os.path.join(stage2_dir, 'enhanced_meta_ensemble.pkl')
        joblib.dump(final_meta_model_v2, enhanced_meta_ensemble_pkl_path)
        print(f"✅ Saved complete enhanced MetaEnsemble object to {enhanced_meta_ensemble_pkl_path}")
        
        # Save Laplacian feature artifacts
        laplacian_path = os.path.join(stage2_dir, 'laplacian_features.pkl')
        joblib.dump(laplacian_transformers, laplacian_path)
        print(f"Saved Laplacian artifacts to {laplacian_path}")
        
        # =======================================================================
        # EXPORT DEPLOYMENT ARTIFACTS
        # =======================================================================
        print("\n=== Exporting Deployment Artifacts ===")
        try:
            from sparc.data.artifacts.export_hook import export_training_artifacts
            
            # Get the original data with OBJECTID (from joined_data which has OBJECTID as index)
            export_data = joined_data.reset_index()  # Restore OBJECTID as column
            
            # Check if we have GWR local coefficients from Stage 2b
            local_coefficients = None
            coefficient_feature_names = None
            if full_models_dir is not None:
                gwr_coef_path = os.path.join(full_models_dir, "mgwr_local_coefficients.csv")
                if os.path.exists(gwr_coef_path):
                    print(f"  Found GWR coefficients from Stage 2b: {gwr_coef_path}")
                    gwr_coef_df = pd.read_csv(gwr_coef_path)
                    # Extract coefficient columns (Coeff_* pattern)
                    coef_cols = [c for c in gwr_coef_df.columns if c.startswith('Coeff_') or c.startswith('coef_')]
                    if coef_cols:
                        local_coefficients = gwr_coef_df[coef_cols].values
                        coefficient_feature_names = [c.replace('Coeff_', '').replace('coef_', '') for c in coef_cols]
                        print(f"  Loaded {len(coef_cols)} coefficient columns: {coefficient_feature_names}")
            
            # Export all artifacts needed for deployment inference
            artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'artifacts')
            artifact_paths = export_training_artifacts(
                data=export_data,
                y=y,
                coords=coords,
                feature_names=selected_features,
                laplacian_features=X_laplacian,
                laplacian_transformers=laplacian_transformers,
                meta_ensemble=final_meta_model_v2,
                base_model_predictions=base_predictions,
                oof_predictions=meta_oof_predictions_v2,
                model_performance={'r2': meta_r2_v2, 'rmse': meta_rmse_v2},
                n_cv_folds=len(folds),
                feature_scaler=feature_scaler,
                artifacts_dir=artifacts_dir,
                base_models_dir=full_models_dir,  # Copy base models from Stage 2b
                areas_of_interest_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Areas_of_Interest.csv'),
                local_coefficients=local_coefficients,
                coefficient_feature_names=coefficient_feature_names,
                generate_spatial_intelligence=True
            )
            print(f"✅ Deployment artifacts exported to: {artifacts_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not export deployment artifacts: {e}")
            import traceback
            traceback.print_exc()
        
        # Compare the two meta-learner approaches
        print(f"\n=== Meta-Learner Comparison ===")
        print(f"Standard Meta-Learner:  R² = {meta_r2:.4f}, RMSE = {meta_rmse:.4f}")
        print(f"Enhanced Meta-Learner:  R² = {meta_r2_v2:.4f}, RMSE = {meta_rmse_v2:.4f}")
        
        r2_improvement = meta_r2_v2 - meta_r2
        rmse_improvement = meta_rmse - meta_rmse_v2  # Lower RMSE is better
        
        print(f"Improvement: R² = {r2_improvement:+.4f}, RMSE = {rmse_improvement:+.4f}")
        
        if r2_improvement > 0.001:
            print("[OK] Laplacian eigenmaps provide improvement!")
            best_meta_predictions = meta_oof_predictions_v2
            best_meta_r2 = meta_r2_v2
            best_meta_rmse = meta_rmse_v2
            best_approach = "Enhanced (with Laplacian)"
        else:
            print("~ Standard approach performs similarly or better")
            best_meta_predictions = meta_oof_predictions
            best_meta_r2 = meta_r2
            best_meta_rmse = meta_rmse
            best_approach = "Standard"
        
        print(f"Best approach: {best_approach}")
        
        # Use true OOF residuals for Deep Kriging (no leakage) - BUT WE'LL SKIP IT
        print(f"True OOF Residuals - Mean: {np.mean(meta_oof_residuals):.4f}, Std: {np.std(meta_oof_residuals):.4f}")
        print("Deep Kriging will be skipped per user request")
        
        # Generate Laplacian eigenmaps early to ensure they get saved
        print("\n=== Stage 4: Generating Enhanced Laplacian Eigenmaps (150 components) ===")
        
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
        
        X_laplacian_full, feature_transformers = generate_enhanced_laplacian_features(coords, n_components=150)
        
        # Save enhanced feature artifacts for final interpretation
        paths = get_paths()
        stage2_dir = paths.stage2_dir
        joblib.dump(feature_transformers, os.path.join(stage2_dir, 'enhanced_spatial_features.pkl'))
        print(f"Saved Laplacian artifacts to {stage2_dir}/")
        
        # Check spatial autocorrelation in meta-ensemble residuals
        print("\n=== Spatial Autocorrelation Analysis of Meta-Ensemble Residuals ===")
        
        # Estimate sensible distance for spatial weights
        autocorr_range = estimate_spatial_autocorrelation_range(coords, meta_oof_residuals)
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
        moran = Moran(meta_oof_residuals, w, two_tailed=False)
        print(f"Meta residuals Moran I = {moran.I:.3f}, p-value = {moran.p_sim:.3f}")
        print(f"Spatial weight threshold used: {max_reasonable_threshold:.0f}m")
        
        # Set threshold for running Deep Kriging
        moran_threshold = 0.1
        p_value_threshold = 0.05
        
        # Enable Deep Kriging based on config flag + spatial autocorrelation in residuals
        if not dk_enabled:
            run_deep_kriging = False
            print(f"Deep Kriging disabled in config (models.deep_kriging.enabled: false)")
        elif moran.I > moran_threshold and moran.p_sim < p_value_threshold:
            run_deep_kriging = True
            print(f"[OK] Significant spatial autocorrelation detected (I > {moran_threshold}, p < {p_value_threshold})")
            print(f"  -> Running Deep Kriging V{dk_version} to model spatial residuals")
        else:
            run_deep_kriging = False
            print(f"X Spatial signal too weak (I = {moran.I:.3f} <= {moran_threshold} or p = {moran.p_sim:.3f} >= {p_value_threshold})")
            print("  -> Skipping Deep Kriging, using Meta-Ensemble only")
        
        # Stage 5: Conditional Deep Kriging on meta-ensemble residuals
        if run_deep_kriging:
            print("\n=== Stage 5: Enhanced Deep Kriging on Meta-Ensemble OOF Residuals ===")
            
            # Debug analysis function
            def debug_deep_kriging_performance(meta_oof_residuals, coords, X_laplacian_full, folds):
                """Debug function to understand Deep Kriging inputs"""
                print("\n=== Deep Kriging Debug Analysis ===")
                
                # Analyze residual characteristics
                print(f"Residual statistics:")
                print(f"  Mean: {np.mean(meta_oof_residuals):.6f}")
                print(f"  Std: {np.std(meta_oof_residuals):.6f}")
                print(f"  Min: {np.min(meta_oof_residuals):.6f}")
                print(f"  Max: {np.max(meta_oof_residuals):.6f}")
                print(f"  Range: {np.max(meta_oof_residuals) - np.min(meta_oof_residuals):.6f}")
                
                residual_range = np.max(meta_oof_residuals) - np.min(meta_oof_residuals)
                if residual_range < 1.0:
                    print(f"  WARNING: Small residual range ({residual_range:.6f}) - may be challenging to learn")
                
                # Check enhanced features
                print(f"\nEnhanced spatial feature statistics:")
                print(f"  Shape: {X_laplacian_full.shape}")
                print(f"  Mean: {np.mean(X_laplacian_full):.6f}")
                print(f"  Std: {np.std(X_laplacian_full):.6f}")
                print(f"  Has NaN: {np.any(np.isnan(X_laplacian_full))}")
                print(f"  Has Inf: {np.any(np.isinf(X_laplacian_full))}")
                
                # Check if we have enough variation to learn from
                if np.std(meta_oof_residuals) < 0.1:
                    print("  WARNING: Very low variance in residuals - network might struggle to learn")
                elif np.std(meta_oof_residuals) > 1.0:
                    print("  INFO: Good variance in residuals - network should be able to learn patterns")
                
                return {
                    'residual_range': residual_range,
                    'residual_std': np.std(meta_oof_residuals),
                    'feature_shape': X_laplacian_full.shape,
                    'has_issues': np.any(np.isnan(X_laplacian_full)) or np.any(np.isinf(X_laplacian_full))
                }
            
            # Run debug analysis
            debug_info = debug_deep_kriging_performance(meta_oof_residuals, coords, X_laplacian_full, folds)
            
            print(f"Input data shapes:")
            print(f"  Coordinates: {coords.shape}")
            print(f"  Enhanced spatial features: {X_laplacian_full.shape}")
            print(f"  Meta-ensemble OOF residuals: {meta_oof_residuals.shape}")
            print(f"OOF Residuals stats - Mean: {np.mean(meta_oof_residuals):.4f}, Std: {np.std(meta_oof_residuals):.4f}")
            
            # Clean features if needed
            if debug_info['has_issues']:
                print("Cleaning features (removing NaN/Inf values)...")
                X_laplacian_full = np.nan_to_num(X_laplacian_full, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Generate Deep Kriging OOF predictions on meta-ensemble OOF residuals
            dk_oof_predictions = np.zeros(len(meta_oof_residuals))
            
            print(f"Running Enhanced Deep Kriging CV with {len(folds)} folds...")
            print(f"Enhanced DK parameters: {dk_params}")
            
            with tqdm(total=len(folds), desc="Enhanced Deep Kriging CV", unit="fold") as pbar:
                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    try:
                        # Get training and test data
                        coords_train = coords[train_idx]
                        coords_test = coords[test_idx]
                        features_train = X_laplacian_full[train_idx]
                        features_test = X_laplacian_full[test_idx]
                        residuals_train = meta_oof_residuals[train_idx]  # Use true OOF residuals
                        
                        # Train Deep Kriging on meta-ensemble OOF residuals
                        if dk_version == 2:
                            # DeepKrigingV2 uses multi-scale Wendland basis + spatial smoothness
                            dk_fold = DeepKrigingV2(
                                n_wendland_centers=min(200, len(train_idx) // 50),
                                hidden_dims=[128, 64, 32],
                                dropout_rate=dk_params.get('dropout_rate', 0.3),
                                learning_rate=dk_params.get('learning_rate', 0.0005),
                                epochs=dk_params.get('epochs', 200),
                                batch_size=dk_params.get('batch_size', 512),
                                patience=dk_params.get('patience', 20),
                                smoothness_weight=0.01
                            )
                        else:
                            dk_fold = DeepKrigingV2(**dk_params)
                        
                        print(f"  Fold {fold_idx+1}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples")
                        print(f"  Train residuals - Mean: {np.mean(residuals_train):.4f}, Std: {np.std(residuals_train):.4f}")
                        
                        # Fit with enhanced features
                        dk_fold.fit(coords_train, residuals_train, features_train, verbose=0)
                        
                        # Predict residuals on test set
                        fold_predictions = dk_fold.predict(coords_test, features_test)
                        dk_oof_predictions[test_idx] = fold_predictions
                        
                        # Calculate fold performance
                        fold_residuals_true = meta_oof_residuals[test_idx]
                        fold_r2 = r2_score(fold_residuals_true, fold_predictions)
                        fold_rmse = np.sqrt(mean_squared_error(fold_residuals_true, fold_predictions))
                        print(f"  Fold {fold_idx+1} performance: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")
                        
                    except Exception as e:
                        print(f"  Fold {fold_idx+1} failed: {e}")
                        print("  Using zero predictions for this fold")
                        dk_oof_predictions[test_idx] = 0.0
                    
                    pbar.update(1)
            
            # Calculate Enhanced Deep Kriging performance on OOF residuals
            dk_r2 = r2_score(meta_oof_residuals, dk_oof_predictions)
            dk_rmse = np.sqrt(mean_squared_error(meta_oof_residuals, dk_oof_predictions))
            
            print(f"\nEnhanced Deep Kriging Performance on OOF Residuals:")
            print(f"  R² = {dk_r2:.4f}, RMSE = {dk_rmse:.4f}")
            
            # Analyze residual reduction
            original_residual_std = np.std(meta_oof_residuals)
            remaining_residual_std = np.std(meta_oof_residuals - dk_oof_predictions)
            residual_reduction = (original_residual_std - remaining_residual_std) / original_residual_std * 100
            
            print(f"  Original residual std: {original_residual_std:.4f}")
            print(f"  Remaining residual std: {remaining_residual_std:.4f}")
            print(f"  Residual reduction: {residual_reduction:.1f}%")
            
            # Final ensemble predictions: meta-ensemble + deep kriging residual predictions
            final_predictions = meta_oof_predictions + dk_oof_predictions
            
            print(f"  Deep Kriging correction range: [{np.min(dk_oof_predictions):.4f}, {np.max(dk_oof_predictions):.4f}]")
            print(f"  Mean absolute correction: {np.mean(np.abs(dk_oof_predictions)):.4f}")
            
        else:
            print("\n=== Stage 5: Skipping Deep Kriging (disabled per user request) ===")
            # Use the best meta-learner predictions as final predictions
            final_predictions = best_meta_predictions
            dk_oof_predictions = np.zeros(len(meta_oof_residuals))  # Zero correction
            dk_r2 = 0.0  # No improvement from DK
            dk_rmse = 0.0  # Not applicable
        
        # Calculate final performance using best meta-learner
        final_r2 = r2_score(y, final_predictions)
        final_rmse = np.sqrt(mean_squared_error(y, final_predictions))
        
        print(f"\n=== Enhanced Final Results ===")
        print(f"Base Models Best (GWRF):       R² = {stage2_results['performance']['individual_models']['gwrf']['r2']:.4f}")
        print(f"Standard Meta Ensemble:        R² = {meta_r2:.4f}, RMSE = {meta_rmse:.4f}")
        print(f"Enhanced Meta Ensemble:        R² = {meta_r2_v2:.4f}, RMSE = {meta_rmse_v2:.4f}")
        print(f"Best Meta Ensemble ({best_approach}): R² = {best_meta_r2:.4f}, RMSE = {best_meta_rmse:.4f}")
        print(f"Deep Kriging:                  DISABLED")
        print(f"Final Ensemble (Best Meta):    R² = {final_r2:.4f}, RMSE = {final_rmse:.4f}")
        
        # Detailed improvement analysis
        base_gwrf_r2 = stage2_results['performance']['individual_models']['gwrf']['r2']
        standard_meta_improvement = meta_r2 - base_gwrf_r2
        enhanced_meta_improvement = meta_r2_v2 - base_gwrf_r2
        best_meta_improvement = best_meta_r2 - base_gwrf_r2
        laplacian_benefit = meta_r2_v2 - meta_r2
        
        print(f"\nDetailed Performance Analysis:")
        print(f"Standard Meta vs Base GWRF:    {standard_meta_improvement:+.4f} R² points")
        print(f"Enhanced Meta vs Base GWRF:    {enhanced_meta_improvement:+.4f} R² points")
        print(f"Best Meta vs Base GWRF:        {best_meta_improvement:+.4f} R² points")
        print(f"Laplacian Enhancement Benefit: {laplacian_benefit:+.4f} R² points")
        
        if laplacian_benefit > 0.001:
            print(f"[OK] Laplacian eigenmaps provide meaningful improvement!")
        elif laplacian_benefit > 0:
            print(f"~ Laplacian eigenmaps provide small improvement")
        else:
            print(f"✗ Laplacian eigenmaps did not improve performance")
        
        # Simple spatial performance summary
        print(f"\n=== Spatial Performance Summary ===")
        print(f"Meta-Ensemble Residuals Moran's I: {moran.I:.3f} (p = {moran.p_sim:.3f})")
        if moran.I > 0.3:
            print("[WARNING] High spatial autocorrelation detected - consider spatial regularization")
        elif moran.I > 0.1:
            print("[WARNING] Moderate spatial autocorrelation - monitoring recommended") 
        else:
            print("[OK] Low spatial autocorrelation - good spatial performance")
        
        print(f"Deep Kriging: DISABLED per user request")
        
        # Save final results
        final_results = {
            'base_models': stage2_results['performance']['individual_models'],
            'meta_ensemble_standard': {'r2': meta_r2, 'rmse': meta_rmse},
            'meta_ensemble_enhanced': {'r2': meta_r2_v2, 'rmse': meta_rmse_v2},
            'meta_ensemble_best': {'r2': best_meta_r2, 'rmse': best_meta_rmse, 'approach': best_approach},
            'spatial_autocorrelation': {
                'moran_i': moran.I,
                'p_value': moran.p_sim,
                'autocorr_range': autocorr_range,
                'threshold_met': False  # Deep Kriging disabled
            },
            'deep_kriging': {
                'r2': 0.0, 
                'rmse': 0.0,
                'executed': False  # Disabled per user request
            },
            'final_ensemble': {'r2': final_r2, 'rmse': final_rmse},
            'improvements': {
                'standard_meta_vs_base': standard_meta_improvement,
                'enhanced_meta_vs_base': enhanced_meta_improvement,
                'best_meta_vs_base': best_meta_improvement,
                'laplacian_benefit': laplacian_benefit
            }
        }
        
        paths = get_paths()
        results_file = paths.stage2_dir / 'final_ensemble_results.json'
        predictions_file = paths.stage2_dir / 'final_ensemble_predictions.csv'
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save final predictions
        final_results_df = pd.DataFrame({
            'OBJECTID': joined_data.index,
            'actual': y,
            'meta_ensemble_standard': meta_oof_predictions,
            'meta_ensemble_enhanced': meta_oof_predictions_v2,
            'meta_standard_residuals': meta_oof_residuals,
            'meta_enhanced_residuals': meta_oof_residuals_v2,
            'best_meta_approach': best_approach,
            'final_ensemble': final_predictions,
            'deep_kriging_executed': False  # Disabled
        })
        
        final_results_df.to_csv(predictions_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - final_ensemble_results.json")
        print(f"  - final_ensemble_predictions.csv")
        
        # Save final composite scaler for the winning meta-ensemble model
        print(f"\n=== Saving Final Composite Scaler ===")
        print("Creating scaler for the winning meta-ensemble with Laplacian eigenmaps...")
        
        # Extract the scalers that were created during training
        final_scaler_components = {
            'approach': best_approach,
            'feature_names': selected_features,
            'coordinate_columns': cv_system.base_config['variables']['coordinates'],
            'laplacian_transformers': laplacian_transformers,  # Contains coord_scaler, laplacian_scaler, etc.
            'best_meta_model': final_meta_model_v2 if best_approach == "Enhanced (with Laplacian)" else final_meta_model,
            'model_performance': {
                'r2': best_meta_r2,
                'rmse': best_meta_rmse,
                'approach': best_approach
            },
            'training_info': {
                'n_samples': len(y),
                'n_features': len(selected_features),
                'laplacian_components': 150,
                'cv_folds': len(folds)
            }
        }
        
        # Save the composite scaler
        scaler_path = paths.stage2_dir / 'final_meta_ensemble_scaler.pkl'
        joblib.dump(final_scaler_components, scaler_path)
        
        print(f"✅ Final composite scaler saved to: {scaler_path}")
        print(f"   - Model approach: {best_approach}")
        print(f"   - Performance: R² = {best_meta_r2:.4f}, RMSE = {best_meta_rmse:.4f}")
        print(f"   - Features: {len(selected_features)} base + 150 Laplacian components")
        print(f"   - Use this scaler for all future predictions!")
        
        # Save the winning meta-ensemble model as final_meta_ensemble.pkl
        best_model = final_meta_model_v2 if best_approach == "Enhanced (with Laplacian)" else final_meta_model
        final_model_path = paths.stage2_dir / 'final_meta_ensemble.pkl'
        joblib.dump(best_model, final_model_path)
        
        print(f"\n✅ FINAL META-ENSEMBLE MODEL saved to: {final_model_path}")
        print(f"   - This is the WINNING model ({best_approach})")
        print(f"   - R² = {best_meta_r2:.4f}, RMSE = {best_meta_rmse:.4f}")
        print(f"   - Use this model + final_meta_ensemble_scaler.pkl for all predictions!")
        
        # Print comprehensive summary of all saved artifacts
        print("\n" + "="*80)
        print("📦 COMPLETE MODEL ARTIFACT SUMMARY")
        print("="*80)
        
        print("\n🔷 BASE MODELS (Stage 2b):")
        print(f"   └─ {os.path.join(full_models_dir, 'ols_model_full.pkl')}")
        print(f"   └─ {os.path.join(full_models_dir, 'gwr_model_full.pkl')}")
        print(f"   └─ {os.path.join(full_models_dir, 'gwrf_model_full.pkl')}")
        print(f"   └─ {os.path.join(full_models_dir, 'ggpgam_model_full.pkl')}")
        
        print("\n🔷 META-ENSEMBLE MODELS (Stage 3):")
        print(f"   ├─ Standard: {os.path.join(stage2_dir, 'standard_meta_ensemble.pkl')}")
        print(f"   ├─ Enhanced: {os.path.join(stage2_dir, 'enhanced_meta_ensemble.pkl')}")
        print(f"   └─ FINAL (Winner): {final_model_path}")
        
        print("\n🔷 SCALERS & PREPROCESSING:")
        print(f"   ├─ Feature Scaler: {os.path.join(stage2_dir, 'feature_scaler.pkl')}")
        print(f"   ├─ Laplacian Features: {os.path.join(stage2_dir, 'laplacian_features.pkl')}")
        print(f"   └─ FINAL Meta Scaler: {scaler_path}")
        
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
                    'description': 'PDP curves from LightGBM meta-ensemble',
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
            with open(package_path, 'w') as f:
                json.dump(sensitivity_package, f, indent=2)
            
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
        
        print("\n" + "="*80)
        
        return final_results
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()