#!/usr/bin/env python3
"""
Correlogram-Based Spatial Analysis Module for SPARC Pipeline
Replaces variogram analysis with correlogram analysis to determine optimal model bandwidths and CV block sizes
Analyzes spatial autocorrelation for ALL variables (temperature + predictors) using Moran's I
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import warnings
import joblib
from pathlib import Path

# When installed via `pip install -e .`, the package root is already on sys.path.
# This fallback supports direct-script execution (python run/correlogram_analysis.py).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sparc.run.pipeline_paths import get_paths
from sparc.config.config import load_config
from sparc.data.data_utils import load_and_preprocess_data

warnings.filterwarnings('ignore')

# Import spatial autocorrelation components
from sparc.run.spatial_autocorr_comprehensive import SpatialAutocorrelationAnalyzer
from sparc.run.correlogram_matern_fit import fit_matern, MaternFitResult
from sparc.run.anisotropy import fit_anisotropy, AnisotropyResult
from sparc.run.cross_correlogram import (
    compute_and_summarise as compute_cross_summary,
    build_effective_range_matrix,
    aggregate_outcome_cross_ranges,
)

class CorrelogramSpatialAnalyzer:
    """
    Analyzes spatial correlograms for all variables to determine optimal model parameters and CV block sizes
    Optimized for large datasets with intelligent sampling
    """
    
    def __init__(self, max_distance=None, n_lags=15, max_sample_size=5000, cache_dir=None):
        """
        Initialize correlogram analyzer
        
        Args:
            max_distance: Maximum distance for spatial analysis (meters).
                          If None, auto-detected as 30% of data bounding-box diagonal.
            n_lags: Number of distance lags for correlogram
            max_sample_size: Maximum sample size for analysis (to speed up computation)
            cache_dir: Optional directory for joblib caching of expensive Moran's I computations
        """
        self.max_distance = max_distance
        self.n_lags = n_lags
        self.max_sample_size = max_sample_size
        self._memory = joblib.Memory(str(cache_dir), verbose=0) if cache_dir else None
        self.n_lags = n_lags
        self.max_sample_size = max_sample_size
        
    def analyze_variable_correlogram(self, coords, values, variable_name, output_dir):
        """
        Analyze spatial correlogram for a single variable to determine optimal bandwidth
        Optimized for large datasets with intelligent sampling
        
        Parameters:
        -----------
        coords : array-like
            Spatial coordinates
        values : array-like
            Variable values
        variable_name : str
            Name of the variable
        output_dir : str
            Output directory for plots and results
            
        Returns:
        --------
        dict : Analysis results including optimal bandwidth and spatial parameters
        """
        print(f"Computing spatial correlogram for {variable_name}...")
        
        # Auto-detect max_distance from data extent if not set
        if self.max_distance is None:
            bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
            self.max_distance = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.3)
            print(f"  Auto-detected max_distance: {self.max_distance:.0f}")
        
        # Apply intelligent sampling for large datasets
        n_samples = len(coords)
        if n_samples > self.max_sample_size:
            print(f"  Large dataset detected ({n_samples:,} points). Sampling {self.max_sample_size:,} points for efficiency...")
            
            # Use stratified sampling to maintain spatial distribution
            import numpy as np
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(n_samples, size=self.max_sample_size, replace=False)
            coords_sample = coords[sample_indices]
            values_sample = values[sample_indices]
            
            print(f"  Using {len(coords_sample):,} sampled points for correlogram analysis")
        else:
            coords_sample = coords
            values_sample = values
            print(f"  Using all {n_samples:,} points for correlogram analysis")
        
        # Create spatial autocorrelation analyzer
        analyzer = SpatialAutocorrelationAnalyzer(coords_sample, max_distance=self.max_distance)
        
        # Compute correlogram (possibly cached via joblib)
        if self._memory is not None:
            @self._memory.cache
            def _cached_correlogram(coords_key, values_key, max_dist, n_lags):
                """Pure-function wrapper so joblib can hash the inputs."""
                _analyzer = SpatialAutocorrelationAnalyzer(coords_key, max_distance=max_dist)
                return _analyzer.compute_correlogram(values_key, plot=False,
                                                     title=f"Spatial Correlogram")
            correlogram_results = _cached_correlogram(
                coords_sample, values_sample, self.max_distance, self.n_lags
            )
        else:
            correlogram_results = analyzer.compute_correlogram(
                values_sample,
                plot=False,
                title=f"Spatial Correlogram - {variable_name}"
            )
        
        # Extract key spatial parameters BEFORE plotting
        optimal_block_size = correlogram_results['optimal_block_size']
        # Bandwidth = first lag where Moran's I dips below 0 (zero-crossing)
        first_zero_crossing = correlogram_results.get('first_zero_crossing')
        if first_zero_crossing is not None and first_zero_crossing < self.max_distance:
            optimal_bandwidth = first_zero_crossing
        else:
            # Fallback: half the block size if autocorrelation never reaches 0
            optimal_bandwidth = optimal_block_size * 0.5
        effective_range = optimal_bandwidth
        
        # NOTE: Per-variable correlogram plots used to be saved to disk here
        # via plt.savefig(). The pipeline no longer renders PNGs at all:
        # the desktop app reads the structured ``correlogram_results``
        # struct from artifacts.db and renders the live visualization,
        # which the user can export directly from the UI.

        # Determine best kernel based on correlogram pattern
        correlogram_data = correlogram_results['correlogram_results']
        
        # Debug: Print the structure to understand what we're working with
        print(f"  Correlogram results keys: {correlogram_results.keys()}")
        if correlogram_data and len(correlogram_data) > 0:
            print(f"  First correlogram entry keys: {correlogram_data[0].keys()}")
        
        # Extract Moran's I values safely
        moran_values = []
        significant_lags = 0
        
        for r in correlogram_data:
            if 'morans_i' in r:  # Correct key name from debug output
                if r.get('significant', False):
                    moran_values.append(r['morans_i'])
                    significant_lags += 1
        
        # Choose kernel based on autocorrelation decay pattern (legacy
        # heuristic, retained as fallback when the Bayesian Matérn fit
        # below cannot run, e.g. < 4 lags).
        if len(moran_values) >= 3:
            # Check decay pattern
            decay_rate = (moran_values[0] - moran_values[-1]) / len(moran_values)
            if decay_rate > 0.1:  # Fast decay
                best_kernel = 'exponential'
            elif decay_rate > 0.05:  # Medium decay
                best_kernel = 'gaussian'
            else:  # Slow decay
                best_kernel = 'spherical'
        else:
            best_kernel = 'gaussian'  # Default

        # ─── Bayesian Matérn fit (Phase 1: kernel-field foundation) ───
        # Fits ρ(h) = σ²·Matérn(κ,ν) + τ²·𝟙[h=0] to the empirical Moran's
        # I curve via NUTS over (κ, σ², τ²) with ν chosen from the
        # closed-form half-integer grid {0.5, 1.5, 2.5}.  When the fit
        # succeeds we promote ``best_kernel`` to "matern" and attach the
        # full posterior summary; the legacy heuristic above is kept as
        # a guaranteed-non-empty fallback for back-compat.
        matern_fit_payload: dict | None = None
        try:
            lag_dist_arr = np.asarray(
                correlogram_results.get('lag_distances', []), dtype=np.float64
            )
            morans_arr = np.array(
                [r.get('morans_i', 0.0) for r in correlogram_data], dtype=np.float64
            )
            if len(lag_dist_arr) >= 4 and len(lag_dist_arr) == len(morans_arr):
                fit_res = fit_matern(
                    lag_dist_arr, morans_arr,
                    method="bayes", n_samples=400, n_warmup=300, n_chains=2, seed=42,
                )
                matern_fit_payload = fit_res.to_payload()
                matern_fit_payload['variable'] = variable_name
                if fit_res.converged:
                    best_kernel = 'matern'
                print(
                    f"  Matérn fit ({fit_res.method}): κ={fit_res.kappa_mean:.4g} "
                    f"[ν={fit_res.nu}], rmse={fit_res.fit_rmse:.4f}, "
                    f"converged={fit_res.converged}"
                )
        except Exception as _exc:  # noqa: BLE001
            print(f"  Matérn fit unavailable for {variable_name}: {_exc}")
            matern_fit_payload = None

        # ─── Phase 2: Anisotropic Matérn fit (directional correlogram) ───
        # Bins point pairs by both lag and angle (4 directors by default)
        # then fits ρ(h, φ) = σ²·Matérn(h, κ_eff(φ), ν) with
        # κ_eff(φ)² = (κ_x cos(φ-θ))² + (κ_y sin(φ-θ))² via NUTS over
        # (κ_x, κ_y, θ, σ², τ², obs_σ²).  Reuses the κ posterior mean
        # from Phase 1 as a heuristic init when available.
        anisotropy_payload: dict | None = None
        try:
            kappa_init_aniso = None
            if matern_fit_payload is not None:
                kappa_init_aniso = matern_fit_payload.get('kappa', {}).get('mean')
            dir_corr = analyzer.compute_directional_correlogram(
                values_sample, n_angle_bins=4,
            )
            aniso_res = fit_anisotropy(
                dir_corr['lag_distances'],
                dir_corr['angle_centers_rad'],
                dir_corr['morans_i'],
                n_pairs=dir_corr['n_pairs'],
                method="bayes",
                n_samples=300, n_warmup=250, n_chains=2, seed=42,
                kappa_init=kappa_init_aniso,
            )
            anisotropy_payload = aniso_res.to_payload()
            anisotropy_payload['variable'] = variable_name
            anisotropy_payload['angle_centers_deg'] = [
                float(x) for x in dir_corr['angle_centers_deg'].tolist()
            ]
            anisotropy_payload['lag_distances'] = [
                float(x) for x in dir_corr['lag_distances'].tolist()
            ]
            print(
                f"  Anisotropy ({aniso_res.method}): {aniso_res.dominant_direction_hint}; "
                f"converged={aniso_res.converged}, rmse={aniso_res.fit_rmse:.4f}"
            )
        except Exception as _exc:  # noqa: BLE001
            print(f"  Anisotropy fit unavailable for {variable_name}: {_exc}")
            anisotropy_payload = None
        
        # Calculate additional spatial statistics
        max_moran = max(moran_values) if moran_values else 0
        
        analysis_result = {
            'variable': variable_name,
            'optimal_bandwidth': optimal_bandwidth,
            'effective_range': effective_range,
            'optimal_block_size': optimal_block_size,
            'best_kernel': best_kernel,
            'max_moran_i': max_moran,
            'significant_lags': significant_lags,
            'correlogram_results': correlogram_results,
            'spatial_parameters': {
                'kernel': best_kernel,
                'bandwidth': optimal_bandwidth,
                'range': effective_range,
                'block_size': optimal_block_size
            },
            'matern_fit': matern_fit_payload,
            'anisotropy': anisotropy_payload,
        }
        
        return analysis_result
    
    def determine_model_bandwidths(self, all_results, target_variable=None):
        """
        Determine model-specific bandwidths from correlogram analysis results.
        Excludes the target variable -- bandwidths are for *predictors* only.
        """
        print("Determining model-specific bandwidths from correlogram analysis...")
        
        # Extract bandwidths from predictor variables only
        bandwidths = [
            result['optimal_bandwidth']
            for var, result in all_results.items()
            if var != target_variable
        ]
        
        if not bandwidths:
            print("Warning: No bandwidth values found")
            return {}
        
        # Calculate statistics
        mean_bandwidth = np.mean(bandwidths)
        median_bandwidth = np.median(bandwidths)
        max_bandwidth = np.max(bandwidths)
        min_bandwidth = np.min(bandwidths)
        
        print(f"Bandwidth statistics:")
        print(f"  Mean: {mean_bandwidth:.1f}m")
        print(f"  Median: {median_bandwidth:.1f}m") 
        print(f"  Range: {min_bandwidth:.1f}m - {max_bandwidth:.1f}m")
        
        # Assign model-specific bandwidths based on spatial characteristics
        model_bandwidths = {
            "OLS": None,  # Global model
            "GWR": median_bandwidth,  # Use median for robustness
            "GWRF": mean_bandwidth * 1.2,  # Slightly larger for RF ensemble effect
            "GGPGAM": max_bandwidth * 0.8  # Conservative for GP model
        }
        
        print(f"Model bandwidth assignments:")
        for model, bw in model_bandwidths.items():
            if bw is None:
                print(f"  {model}: Global model (no bandwidth)")
            else:
                print(f"  {model}: {bw:.1f}m")
        
        return model_bandwidths
    
    def determine_optimal_cv_block_size(self, all_results, model_bandwidths, target_variable=None, spatial_extent=None, n_folds=5):
        """
        Determine optimal CV block size from the TARGET variable's spatial
        autocorrelation range.  Block size is decoupled from model bandwidths.

        Logic
        -----
        1. Use the target variable's first-zero-crossing (the distance at which
           outcome autocorrelation decays to zero).  This is the scientifically
           correct basis because CV independence is about the *outcome*, not
           individual predictors.
        2. Cap at  spatial_extent / (2 * n_folds)  so we always have enough
           room for viable folds.
        3. The buffer zone (applied in enhanced_spatial_cv) handles any
           residual autocorrelation leakage.
        """
        print("\nDetermining optimal CV block size...")

        # --- Step 1: target variable's zero-crossing -----------------------
        target_zc = None
        if target_variable and target_variable in all_results:
            target_res = all_results[target_variable]
            target_zc = target_res.get('optimal_bandwidth')   # = first_zero_crossing
            target_block = target_res.get('optimal_block_size')  # first non-significant lag
            print(f"  Target variable ({target_variable}):")
            print(f"    First zero-crossing (Moran's I <= 0): {target_zc:.0f}m")
            print(f"    First non-significant lag:            {target_block:.0f}m")

        if target_zc is None or target_zc <= 0:
            # Fallback: median zero-crossing across all variables
            all_zc = [r.get('optimal_bandwidth', 0) for r in all_results.values() if r.get('optimal_bandwidth', 0) > 0]
            target_zc = float(np.median(all_zc)) if all_zc else 5000.0
            print(f"  No usable target zero-crossing; falling back to median across variables: {target_zc:.0f}m")

        proposed_block = target_zc
        print(f"  Proposed block size (target zero-crossing): {proposed_block:.0f}m")

        # --- Step 2: spatial-extent cap ------------------------------------
        if spatial_extent and spatial_extent > 0:
            max_viable = spatial_extent / (2.0 * n_folds)
            if proposed_block > max_viable:
                print(f"  Capping: {proposed_block:.0f}m > spatial_extent/(2*{n_folds}) = {max_viable:.0f}m")
                proposed_block = max_viable

        # --- Step 3: sensible floor ----------------------------------------
        floor = 500.0
        if proposed_block < floor:
            print(f"  Block size {proposed_block:.0f}m below floor; raising to {floor:.0f}m")
            proposed_block = floor

        optimal_block_size = proposed_block
        print(f"  Final CV block size: {optimal_block_size:.0f}m")

        # Build a lightweight validation summary (informational only)
        validation_results = {}
        return optimal_block_size, validation_results

def main(fast_mode=False):
    """
    Main function to run correlogram-based spatial analysis on all variables
    
    Parameters:
    -----------
    fast_mode : bool
        If True, use reduced sample sizes and fewer lags for faster analysis
    """
    print("\n=== SPARC Correlogram-Based Spatial Analysis ===\n")
    
    if fast_mode:
        print("[FAST] Running in FAST MODE - reduced precision but much faster")
    
    # Load configuration
    config = load_config()
    
    # Create Stage 0 output directory using centralized paths
    paths = get_paths()
    stage0_dir = paths.stage0_dir
    # Only mkdir when disk writes are enabled; otherwise the canonical
    # store is artifacts.db and Stage_0 directory must remain absent.
    from sparc.run.disk_policy import disk_writes_enabled
    if disk_writes_enabled():
        os.makedirs(stage0_dir, exist_ok=True)
    
    # Stage 0 runs before GWEN, so always use base model predictors
    selected_features = config['predictors']['base_model']
    
    print(f"Analyzing spatial correlograms for features: {selected_features}")
    
    # Load and preprocess data
    data = load_and_preprocess_data(
        raw_data_path=config['paths']['raw_csv_path'],
        identifier_col=config['variables']['identifier'],
        target_col=config['variables']['target'],
        coords_cols=config['variables']['coordinates'],
        predictor_cols=selected_features,
        initial_crs=config['crs']['initial'],
        target_crs=config['crs']['target_projected'],
        output_dir=config.get('output', {}).get('base_dir'),
    )
    
    # Filter features to only those available in data
    available_features = data.columns.tolist()
    selected_features = [f for f in selected_features if f in available_features]
    missing_features = [f for f in selected_features if f not in available_features]
    
    if missing_features:
        print(f"Warning: Missing features {missing_features} - skipping these")
    
    # Add target variable to analysis
    target_variable = config['variables']['target']
    if target_variable in data.columns:
        all_variables = [target_variable] + selected_features
    else:
        all_variables = selected_features
        print(f"Warning: Target variable {target_variable} not found in data")
    
    coords = data[config['variables']['coordinates']].values
    
    # â”€â”€ Use DatasetProfiler for data-driven correlogram parameters â”€â”€â”€
    from sparc.run.dataset_profiler import DatasetProfiler
    profiler = DatasetProfiler(
        data,
        coord_cols=config['variables']['coordinates'],
        feature_cols=selected_features,
    )
    profile = profiler.profile()
    corr_recs = profiler.recommend_parameters().get("correlogram", {})
    print(profiler.summary())
    
    # Save profile for downstream stages (db-resident).
    import json as _json  # noqa: F401  (kept for legacy callers below)
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:  # noqa: BLE001
        _store = None
    if _store is not None:
        _store.write_struct(
            stage="0",
            artifact_id="dataset_profile",
            payload=profile,
            producer="correlogram_analysis.main",
            consumers=["server:/results/correlogram", "pipeline:stage0b"],
        )
        print("Dataset profile written to artifacts.db (stage=0, id=dataset_profile)")
    elif disk_writes_enabled():
        # Back-compat: no active registry (e.g. ad-hoc invocation) AND
        # disk writes explicitly enabled.
        os.makedirs(stage0_dir, exist_ok=True)
        profile_path = os.path.join(stage0_dir, 'dataset_profile.json')
        with open(profile_path, 'w') as _fp:
            _json.dump(profile, _fp, indent=2)
        print(f"Dataset profile saved to: {profile_path}")
    
    # Initialize analyzer with data-driven parameters (no hardcoded 3000 m cap)
    if fast_mode:
        max_sample_size = min(1500, corr_recs.get("max_sample_size", 1500))
        max_distance = corr_recs.get("max_distance", profile["spatial_extent"] * 0.25)
        n_lags = 10
    else:
        max_sample_size = corr_recs.get("max_sample_size", 3000)
        max_distance = corr_recs.get("max_distance", profile["spatial_extent"] * 0.40)
        n_lags = corr_recs.get("n_lags", 15)
    
    cache_dir = Path(stage0_dir) / '.cache'
    analyzer = CorrelogramSpatialAnalyzer(
        max_distance=max_distance,
        n_lags=n_lags,
        max_sample_size=max_sample_size,
        cache_dir=cache_dir,
    )
    
    print(f"Dataset size: {len(data):,} points")
    print(f"Max correlogram distance: {max_distance:,.0f} m (data-driven)")
    print(f"Using sample size: {max_sample_size:,} points for correlogram analysis")
    estimated_time = len(all_variables) * (1 if fast_mode else 2)
    print(f"Estimated analysis time: ~{estimated_time:.0f} minutes")
    
    # Analyze each variable
    all_results = {}
    
    # Use tqdm for progress tracking
    feature_progress = tqdm(all_variables, desc="Correlogram Analysis", unit="variable")
    
    for variable in feature_progress:
        feature_progress.set_description(f"Analyzing {variable}")
        
        if variable in data.columns:
            values = data[variable].values
            result = analyzer.analyze_variable_correlogram(coords, values, variable, stage0_dir)
            all_results[variable] = result
            feature_progress.set_postfix(status="Complete")
        else:
            feature_progress.set_postfix(status="Skipped")
    
    feature_progress.close()
    
    # Determine model bandwidths from correlogram results (predictors only)
    model_bandwidths = analyzer.determine_model_bandwidths(all_results, target_variable=target_variable)
    
    # Determine CV block size — honour user override if present
    spatial_cv_cfg = config.get('optimization', {}).get('spatial_cv', {})
    if not spatial_cv_cfg:
        spatial_cv_cfg = config.get('models', {}).get('spatial_cv', {})
    block_size_source = spatial_cv_cfg.get('block_size_source', 'correlogram')
    user_block_size = spatial_cv_cfg.get('block_size')

    if block_size_source == 'user' and user_block_size is not None and user_block_size > 0:
        optimal_cv_block_size = float(user_block_size)
        cv_validation_results = {}
        print(f"\n  Block size source: USER override")
        print(f"  User-specified CV block size: {optimal_cv_block_size:.0f}m")
        # Phase 4: still report what the auto-uplift WOULD have suggested,
        # purely informational — the user's choice is not modified.
        cv_cross_range_uplift = None
        try:
            if 'effective_range_matrix_payload' in dir() or True:
                erm = locals().get('effective_range_matrix_payload')
                if erm is not None:
                    _outcome_cross = aggregate_outcome_cross_ranges(erm, target_variable)
                    _max_xr = _outcome_cross.get('max_cross_range_m')
                    if _max_xr is not None and _max_xr > optimal_cv_block_size:
                        print(
                            f"  Note: Phase 4 would have suggested ≥ {_max_xr:.0f}m "
                            f"(max target↔predictor cross-range) — keeping user value."
                        )
                        cv_cross_range_uplift = {
                            'user_block_size_m': float(optimal_cv_block_size),
                            'cross_ranges': _outcome_cross['cross_ranges'],
                            'max_cross_range_m': float(_max_xr),
                            'applied': False,
                            'reason': 'user_override',
                        }
        except Exception:  # noqa: BLE001
            pass
    else:
        # Auto-determine from TARGET variable + spatial extent cap
        spatial_extent = profile.get('spatial_extent', None)
        optimal_cv_block_size, cv_validation_results = analyzer.determine_optimal_cv_block_size(
            all_results, model_bandwidths,
            target_variable=target_variable,
            spatial_extent=spatial_extent,
            n_folds=5
        )
        # ─── Phase 4: extend block size by max (target ↔ predictor) cross-range ─
        # The naive block size is the target's *own* zero-crossing.  But if any
        # predictor has a significant cross-range with the target that exceeds
        # the target's marginal range, CV folds can leak through that channel.
        # Take the max of the two (still capped by spatial_extent / 2·n_folds).
        cv_cross_range_uplift = None
        if effective_range_matrix_payload is not None:
            try:
                _outcome_cross = aggregate_outcome_cross_ranges(
                    effective_range_matrix_payload, target_variable,
                )
                _max_xr = _outcome_cross.get('max_cross_range_m')
                if _max_xr is not None and _max_xr > optimal_cv_block_size:
                    _max_viable = (
                        spatial_extent / 10.0 if spatial_extent and spatial_extent > 0
                        else float('inf')
                    )
                    _new_block = min(_max_xr, _max_viable)
                    if _new_block > optimal_cv_block_size:
                        print(
                            f"  Phase 4 uplift: max (target↔predictor) cross-range "
                            f"{_max_xr:.0f}m exceeds target zero-crossing "
                            f"{optimal_cv_block_size:.0f}m → block size raised to "
                            f"{_new_block:.0f}m"
                        )
                        cv_cross_range_uplift = {
                            'previous_block_size_m': float(optimal_cv_block_size),
                            'cross_ranges': _outcome_cross['cross_ranges'],
                            'max_cross_range_m': float(_max_xr),
                            'new_block_size_m': float(_new_block),
                            'applied': True,
                            'reason': 'auto_target_zero_crossing_too_small',
                        }
                        optimal_cv_block_size = float(_new_block)
            except Exception as _exc:  # noqa: BLE001
                print(f"Phase 4 CV uplift skipped: {_exc}")
    
    # Create comprehensive results structure
    comprehensive_results = {
        'metadata': {
            'analysis_type': 'correlogram_based_spatial_analysis',
            'variables_analyzed': list(all_results.keys()),
            'total_variables': len(all_results),
            'includes_target': target_variable in all_results
        },
        'individual_results': all_results,
        'model_bandwidths': model_bandwidths,
        'spatial_cv_configuration': {
            'optimal_block_size': optimal_cv_block_size,
            'validation_results': cv_validation_results,
            'block_size_candidates': [result['optimal_block_size'] for result in all_results.values()],
            'cross_range_uplift': locals().get('cv_cross_range_uplift', None),
        },
        'summary_statistics': {
            'bandwidth_range': {
                'min': min([r['optimal_bandwidth'] for r in all_results.values()]),
                'max': max([r['optimal_bandwidth'] for r in all_results.values()]),
                'mean': np.mean([r['optimal_bandwidth'] for r in all_results.values()]),
                'median': np.median([r['optimal_bandwidth'] for r in all_results.values()])
            }
        }
    }

    # ─── Phase 1: κ_PDE divergence diagnostic ───────────────────────────
    # For each variable with a successful Bayesian Matérn fit, compare the
    # correlogram-derived κ posterior to the PDE-derived κ point estimate
    # (Whittle relation).  The summary feeds the desktop "stationarity
    # warning" badge and informs Phase 2's matrix-kernel design.
    try:
        from sparc.physics.kappa_estimator import (
            estimate_kappa_pde,
            kappa_ratio_summary,
        )
        kappa_pde_res = estimate_kappa_pde(config)
        per_variable_diagnostics: dict[str, dict] = {}
        for _var, _res in all_results.items():
            _mf = _res.get('matern_fit')
            if not _mf or _mf.get('method') != 'bayes':
                continue
            _samples = _mf.get('kappa', {}).get('samples', [])
            per_variable_diagnostics[_var] = kappa_ratio_summary(
                _samples, kappa_pde_res.kappa_pde,
            )
        matern_artifact_payload = {
            'kappa_pde': {
                'value': kappa_pde_res.kappa_pde,
                'regime': kappa_pde_res.regime,
                'source': kappa_pde_res.source,
                'inputs': kappa_pde_res.inputs,
            },
            'per_variable_fits': {
                _var: _res.get('matern_fit')
                for _var, _res in all_results.items()
                if _res.get('matern_fit') is not None
            },
            'kappa_ratio_diagnostics': per_variable_diagnostics,
            'stationarity_warnings': [
                _var for _var, _d in per_variable_diagnostics.items()
                if _d.get('stationarity_warning')
            ],
        }
    except Exception as _exc:  # noqa: BLE001
        print(f"Matérn artifact assembly failed: {_exc}")
        matern_artifact_payload = None

    # ─── Phase 2: Anisotropy artifact ────────────────────────────────────
    # Aggregates per-variable anisotropy posteriors (ellipse + dominant
    # direction hint) into a single struct for the report and desktop.
    try:
        per_variable_aniso = {
            _var: _res.get('anisotropy')
            for _var, _res in all_results.items()
            if _res.get('anisotropy') is not None
        }
        if per_variable_aniso:
            anisotropy_artifact_payload = {
                'per_variable_fits': per_variable_aniso,
                'strong_anisotropy_variables': [
                    _var for _var, _p in per_variable_aniso.items()
                    if (_p.get('ellipse', {})
                          .get('eccentricity', {})
                          .get('mean', 0.0)) >= 0.5
                ],
            }
        else:
            anisotropy_artifact_payload = None
    except Exception as _exc:  # noqa: BLE001
        print(f"Anisotropy artifact assembly failed: {_exc}")
        anisotropy_artifact_payload = None

    # ─── Phase 3: Cross-correlogram kernel field ──────────────────────────
    # V×V matrix-valued correlogram with sym/antisym decomposition.  The
    # antisymmetric block exposes directed (causal-direction) coupling
    # between variable pairs; the symmetric block carries co-variation.
    try:
        # Build (N, V) matrix in the same row order as `coords`; drop NaNs
        # internally inside compute_and_summarise().
        _values_matrix = data[all_variables].to_numpy(dtype=np.float64, copy=True)
        # Sub-sample to the same cap used for per-variable analysis to keep
        # the O(N²) tensor computation tractable.
        _N_full = _values_matrix.shape[0]
        if _N_full > max_sample_size:
            _rng = np.random.default_rng(42)
            _idx = _rng.choice(_N_full, size=max_sample_size, replace=False)
            _coords_cc = coords[_idx]
            _values_cc = _values_matrix[_idx]
        else:
            _coords_cc = coords
            _values_cc = _values_matrix
        _n_perm_cc = 0 if fast_mode else 50
        cross_correlogram_payload = compute_cross_summary(
            _coords_cc, _values_cc, all_variables,
            max_distance=float(max_distance),
            n_lags=min(int(n_lags), 12),
            n_angle_bins=4,
            n_perm=_n_perm_cc,
            seed=42,
        )
    except Exception as _exc:  # noqa: BLE001
        print(f"Cross-correlogram assembly failed: {_exc}")
        cross_correlogram_payload = None

    # ─── Phase 4: Per-pair effective-range matrix ────────────────────────
    # Build the V×V effective-range matrix from the cross-correlogram per-pair
    # summary, with the marginal effective ranges on the diagonal.  Flag any
    # pairs whose cross-range differs by >10× from either endpoint's marginal
    # range — the "wrong-lag false-zero" failure mode that motivates Phase 4.
    effective_range_matrix_payload = None
    if cross_correlogram_payload is not None:
        try:
            _auto_bw = {
                _var: _res.get('effective_range')
                for _var, _res in all_results.items()
                if _res.get('effective_range') is not None
            }
            effective_range_matrix_payload = build_effective_range_matrix(
                cross_correlogram_payload,
                auto_bandwidths=_auto_bw,
                mismatch_factor=10.0,
                require_significance=True,
            )
            print(
                "Effective-range matrix built "
                f"(V={len(effective_range_matrix_payload['variable_names'])}, "
                f"significant_pairs={effective_range_matrix_payload['significant_pair_count']}, "
                f"mismatch_warnings={len(effective_range_matrix_payload['mismatch_warnings'])})"
            )
        except Exception as _exc:  # noqa: BLE001
            print(f"Effective-range matrix assembly failed: {_exc}")
            effective_range_matrix_payload = None

    # ─── Phase 6: Scale-hierarchy / fractal-signature diagnostic ────────
    # Per-variable spectral exponent β, lacunarity, κ scale-drift, and
    # stationarity-class label. Persisted as `stage=0, scale_hierarchy`.
    # Uses the same sub-sampled (coords, values) matrix already prepared
    # for the cross-correlogram so timing stays bounded by Stage 0's
    # existing N≤max_sample_size budget.
    try:
        from sparc.run.scale_hierarchy import compute_scale_hierarchy
        if cross_correlogram_payload is not None:
            _sh_coords = _coords_cc
            _sh_values_matrix = _values_cc
            _sh_variables = list(all_variables)
        else:
            # Fallback path when cross-correlogram was skipped/failed
            _sh_values_matrix = data[all_variables].to_numpy(
                dtype=np.float64, copy=True)
            if _sh_values_matrix.shape[0] > max_sample_size:
                _rng_sh = np.random.default_rng(43)
                _idx_sh = _rng_sh.choice(
                    _sh_values_matrix.shape[0],
                    size=max_sample_size, replace=False,
                )
                _sh_coords = coords[_idx_sh]
                _sh_values_matrix = _sh_values_matrix[_idx_sh]
            else:
                _sh_coords = coords
            _sh_variables = list(all_variables)
        _sh_var_dict = {
            v: _sh_values_matrix[:, i]
            for i, v in enumerate(_sh_variables)
        }
        # Lighter NUTS budget when fast_mode is on
        _sh_warmup = 200 if fast_mode else 500
        _sh_samples = 400 if fast_mode else 1000
        scale_hierarchy_payload = compute_scale_hierarchy(
            _sh_coords, _sh_var_dict,
            n_grid=64, nuts_warmup=_sh_warmup, nuts_samples=_sh_samples,
            seed=42, persist=True,
        )
        _sc = scale_hierarchy_payload.get("summary", {}).get(
            "stationarity_counts", {})
        print(
            "Scale-hierarchy diagnostic complete "
            f"(stationarity_counts={_sc})"
        )
    except Exception as _exc:  # noqa: BLE001
        print(f"Scale-hierarchy diagnostic failed: {_exc}")
        scale_hierarchy_payload = None

    # Save comprehensive results (replaces variogram_analysis_results.json).
    # Now persisted via ArtifactStore below; legacy disk path retained only
    # as a fallback when no active registry is installed.

    def convert_numpy_types(obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert the entire results structure
    json_safe_results = convert_numpy_types(comprehensive_results)

    # Persist comprehensive results + summary table to artifacts.db.
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:  # noqa: BLE001
        _store = None

    summary_data = []
    for variable, result in all_results.items():
        summary_data.append({
            'Variable': variable,
            'Optimal_Bandwidth': result['optimal_bandwidth'],
            'Effective_Range': result['effective_range'],
            'Block_Size': result['optimal_block_size'],
            'Best_Kernel': result['best_kernel'],
            'Max_Moran_I': result['max_moran_i'],
            'Significant_Lags': result['significant_lags']
        })
    summary_df = pd.DataFrame(summary_data)

    if _store is not None:
        _store.write_struct(
            stage="0",
            artifact_id="correlogram_results",
            payload=json_safe_results,
            producer="correlogram_analysis.main",
            consumers=[
                "server:/results/correlogram",
                "report:correlogram",
                "desktop:correlogram_view",
            ],
        )
        _store.write_table(
            stage="0",
            artifact_id="correlogram_summary",
            df=summary_df,
            producer="correlogram_analysis.main",
            consumers=["server:/results/correlogram"],
        )
        if matern_artifact_payload is not None:
            _store.write_struct(
                stage="0",
                artifact_id="correlogram_matern_fit",
                payload=convert_numpy_types(matern_artifact_payload),
                producer="correlogram_analysis.main",
                consumers=[
                    "server:/results/correlogram",
                    "report:correlogram",
                    "desktop:correlogram_view",
                    "pipeline:stage2_mgwr",
                ],
            )
            print(
                "Matérn fit + κ_PDE diagnostics written to artifacts.db "
                f"(stage=0, id=correlogram_matern_fit, "
                f"warnings={len(matern_artifact_payload['stationarity_warnings'])})"
            )
        if anisotropy_artifact_payload is not None:
            _store.write_struct(
                stage="0",
                artifact_id="correlogram_anisotropy",
                payload=convert_numpy_types(anisotropy_artifact_payload),
                producer="correlogram_analysis.main",
                consumers=[
                    "server:/results/correlogram",
                    "report:correlogram",
                    "desktop:correlogram_view",
                    "pipeline:stage2_mgwr",
                ],
            )
            print(
                "Anisotropy ellipses written to artifacts.db "
                f"(stage=0, id=correlogram_anisotropy, "
                f"strong={len(anisotropy_artifact_payload['strong_anisotropy_variables'])})"
            )
        if cross_correlogram_payload is not None:
            _store.write_struct(
                stage="0",
                artifact_id="cross_correlogram_kernel_field",
                payload=convert_numpy_types(cross_correlogram_payload),
                producer="correlogram_analysis.main",
                consumers=[
                    "server:/results/correlogram",
                    "report:correlogram",
                    "desktop:correlogram_view",
                    "pipeline:stage2_mgwr",
                    "pipeline:stage3_causal",
                ],
            )
            print(
                "Cross-correlogram kernel field written to artifacts.db "
                f"(stage=0, id=cross_correlogram_kernel_field, "
                f"V={len(cross_correlogram_payload['variable_names'])}, "
                f"antisym_flags={len(cross_correlogram_payload['antisymmetric_flags'])})"
            )
        if effective_range_matrix_payload is not None:
            _store.write_struct(
                stage="0",
                artifact_id="effective_range_matrix",
                payload=convert_numpy_types(effective_range_matrix_payload),
                producer="correlogram_analysis.main",
                consumers=[
                    "server:/results/correlogram",
                    "report:correlogram",
                    "desktop:correlogram_view",
                    "pipeline:stage1_gwen",
                    "pipeline:stage2_mgwr",
                ],
            )
            print(
                "Effective-range matrix written to artifacts.db "
                f"(stage=0, id=effective_range_matrix, "
                f"warnings={len(effective_range_matrix_payload['mismatch_warnings'])})"
            )
        print("Correlogram results + summary written to artifacts.db (stage=0)")
    elif disk_writes_enabled():
        # Back-compat path for ad-hoc invocation without an active registry.
        os.makedirs(stage0_dir, exist_ok=True)
        results_path = os.path.join(stage0_dir, 'correlogram_analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        summary_df.to_csv(os.path.join(stage0_dir, 'correlogram_summary.csv'), index=False)

    # Persist CV-validation summary as a struct (rendered to text on demand
    # via the report layer). The legacy ``spatial_cv_configuration.txt`` is
    # no longer written from the pipeline.
    cv_config_payload = {
        "model_bandwidths": {
            k: (None if v is None else float(v)) for k, v in model_bandwidths.items()
        },
        "optimal_cv_block_size_m": float(optimal_cv_block_size),
        "block_size_validation": {
            str(bs): dict(model_statuses)
            for bs, model_statuses in cv_validation_results.items()
        },
    }
    if _store is not None:
        _store.write_struct(
            stage="0",
            artifact_id="spatial_cv_configuration",
            payload=cv_config_payload,
            producer="correlogram_analysis.main",
            consumers=["server:/results/correlogram", "report:correlogram"],
        )
    
    print(f"\n=== Correlogram-Based Spatial Analysis Complete ===")
    print(f"Results saved to: {stage0_dir}/")
    print(f"Analyzed {len(all_results)} variables")
    print(f"Optimal CV block size: {optimal_cv_block_size:.0f}m")
    print(f"\nSummary:")
    print(summary_df.to_string(index=False))
    
    # -- Auto-wire per-variable bandwidths into pipeline_config.json --
    _auto_wire_bandwidths(all_results, model_bandwidths, optimal_cv_block_size, paths, target_variable=target_variable)
    
    return comprehensive_results


def _auto_wire_bandwidths(all_results, model_bandwidths, optimal_cv_block_size, paths, target_variable=None):
    """
    Write per-variable bandwidths and block size from the correlogram into
    pipeline_config.json so that Stage 2 picks them up automatically.
    """
    # Build per-variable bandwidth map (exclude target -- it has no bandwidth role)
    per_variable_bandwidths = {}
    for var_name, result in all_results.items():
        if var_name == target_variable:
            continue  # target is not a predictor
        bw = result.get('optimal_bandwidth')
        if bw is not None and bw > 0:
            per_variable_bandwidths[var_name] = float(bw)

    print(f"\n>>> Auto-wiring correlogram bandwidths into pipeline_config.json")
    for var, bw in per_variable_bandwidths.items():
        print(f"    {var}: {bw:.0f} m")

    # Load existing pipeline_config.json (if it exists), patch in bandwidths.
    # Prefer the artifact store; fall back to disk for legacy runs.
    config_path = paths.pipeline_config
    pipeline_cfg = None
    try:
        from sparc.registry.store import get_active_store
        _existing_store = get_active_store()
    except Exception:  # noqa: BLE001
        _existing_store = None
    if _existing_store is not None and _existing_store.has("0", "pipeline_config"):
        try:
            pipeline_cfg = _existing_store.read_struct("0", "pipeline_config")
        except Exception:  # noqa: BLE001
            pipeline_cfg = None
    if pipeline_cfg is None:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                pipeline_cfg = json.load(f)
        else:
            pipeline_cfg = {}

    # Patch
    pipeline_cfg.setdefault('manual_parameters', {})
    pipeline_cfg['manual_parameters']['bandwidths'] = per_variable_bandwidths
    pipeline_cfg['manual_parameters']['block_size'] = float(optimal_cv_block_size)
    pipeline_cfg['manual_parameters']['source'] = 'correlogram_auto'

    # Preserve block_size_source so downstream stages know origin
    config = load_config()
    spatial_cv_cfg = config.get('optimization', {}).get('spatial_cv', {})
    if not spatial_cv_cfg:
        spatial_cv_cfg = config.get('models', {}).get('spatial_cv', {})
    pipeline_cfg['manual_parameters']['block_size_source'] = spatial_cv_cfg.get('block_size_source', 'correlogram')

    # Also store the model-level bandwidths the correlogram determined
    pipeline_cfg['correlogram_model_bandwidths'] = {
        k: float(v) if v is not None else None
        for k, v in model_bandwidths.items()
    }

    # Persist: artifact store is canonical; disk is opt-in.
    from sparc.run.disk_policy import disk_writes_enabled as _disk_on
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:  # noqa: BLE001
        _store = None
    if _store is not None:
        _store.write_struct(
            stage="0",
            artifact_id="pipeline_config",
            payload=pipeline_cfg,
            producer="correlogram_analysis._auto_wire_bandwidths",
            consumers=["pipeline:stage1+"],
        )

    if _disk_on():
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(pipeline_cfg, f, indent=2)
        print(f"    Saved to: {config_path}")
    else:
        print("    pipeline_config persisted to artifacts.db (stage=0, id=pipeline_config)")

if __name__ == "__main__":
    import sys
    
    # Check for fast mode flag
    fast_mode = "--fast" in sys.argv
    
    if fast_mode:
        print("[FAST] FAST MODE ENABLED: Using reduced sample size and fewer lags for quick analysis")
        print("Note: Results may be less precise but much faster\n")
    
    main(fast_mode=fast_mode)
