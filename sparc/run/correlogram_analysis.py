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
from sparc.run.correlogram_report import CorrelogramReport, Degradation

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


def _analyze_variable_worker(
    variable: str,
    coords: "np.ndarray",
    values: "np.ndarray",
    stage0_dir: str,
    max_distance: float,
    n_lags: int,
    max_sample_size: int,
    cache_dir_str: str,
    fast_mode: bool,
    n_threads: int,
) -> tuple:
    """Module-level worker so joblib/loky can pickle it.

    Constructs a fresh CorrelogramSpatialAnalyzer inside the spawned process
    to avoid serialising the joblib.Memory object across the process boundary.
    Sets torch thread count to prevent CPU oversubscription when multiple
    variable workers run in parallel.
    """
    try:
        import torch
        torch.set_num_threads(max(1, n_threads))
    except Exception:  # noqa: BLE001 — torch may not be imported yet
        pass
    _worker_analyzer = CorrelogramSpatialAnalyzer(
        max_distance=max_distance,
        n_lags=n_lags,
        max_sample_size=max_sample_size,
        cache_dir=Path(cache_dir_str) if cache_dir_str else None,
        fast_mode=fast_mode,
    )
    result = _worker_analyzer.analyze_variable_correlogram(
        coords, values, variable, stage0_dir
    )
    return variable, result

class CorrelogramSpatialAnalyzer:
    """
    Analyzes spatial correlograms for all variables to determine optimal model parameters and CV block sizes
    Optimized for large datasets with intelligent sampling
    """
    
    def __init__(self, max_distance=None, n_lags=15, max_sample_size=5000, cache_dir=None, fast_mode=False, unit_label="m"):
        """
        Initialize correlogram analyzer
        
        Args:
            max_distance: Maximum distance for spatial analysis (working CRS units).
                          If None, auto-detected as 30% of data bounding-box diagonal.
            n_lags: Number of distance lags for correlogram
            max_sample_size: Maximum sample size for analysis (to speed up computation)
            cache_dir: Optional directory for joblib caching of expensive Moran's I computations
            fast_mode: If True, use reduced NUTS sample counts for faster runs
            unit_label: Display unit string (e.g. "m", "ft") from the working CRS.
        """
        self.max_distance = max_distance
        self.n_lags = n_lags
        self.max_sample_size = max_sample_size
        self._memory = joblib.Memory(str(cache_dir), verbose=0) if cache_dir else None
        self.n_lags = n_lags
        self.max_sample_size = max_sample_size
        self.fast_mode = fast_mode
        self.unit_label = unit_label
        
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
            if len(coords) == 0:
                raise ValueError(
                    f"Cannot compute correlogram for '{variable_name}': "
                    "coordinate array is empty."
                )
            bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
            self.max_distance = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.3)
            print(f"  Auto-detected max_distance: {self.max_distance:.0f}")
        
        # Sample for the directional correlogram (anisotropy NUTS always needs
        # a fixed-size set to bound the O(n_sample²) distance-matrix construction).
        n_points = len(coords)
        if n_points > self.max_sample_size:
            _rng_s = np.random.default_rng(42)
            _idx = _rng_s.choice(n_points, size=self.max_sample_size, replace=False)
            coords_sample = coords[_idx]
            values_sample = values[_idx]
        else:
            coords_sample = coords
            values_sample = values

        # Create analyzer with sampled coords — only used for the directional
        # correlogram (compute_directional_correlogram needs the distance matrix).
        analyzer = SpatialAutocorrelationAnalyzer(coords_sample, max_distance=self.max_distance, unit_label=self.unit_label)

        # Main isotropic correlogram:
        #  - Large datasets  → FFT path on ALL points, O(n log n), no subsampling.
        #  - Small datasets  → classic Moran's I on all points.
        if n_points > self.max_sample_size:
            from sparc.run.spatial_autocorr_comprehensive import fft_correlogram as _fft_corr
            print(
                f"  FFT correlogram on all {n_points:,} points "
                f"(directional uses {self.max_sample_size:,} sampled)..."
            )
            if self._memory is not None:
                @self._memory.cache
                def _cached_fft(c, v, md, nl):
                    from sparc.run.spatial_autocorr_comprehensive import fft_correlogram as _f
                    return _f(c, v, md, nl)
                correlogram_results = _cached_fft(
                    coords, values, self.max_distance, self.n_lags
                )
            else:
                correlogram_results = _fft_corr(
                    coords, values, self.max_distance, self.n_lags
                )
        else:
            print(f"  Using all {n_points:,} points for correlogram analysis")
            if self._memory is not None:
                @self._memory.cache
                def _cached_correlogram(coords_key, values_key, max_dist, n_lags):
                    """Pure-function wrapper so joblib can hash the inputs."""
                    _analyzer = SpatialAutocorrelationAnalyzer(coords_key, max_distance=max_dist, unit_label=self.unit_label)
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
        _per_var_degradations: list[Degradation] = []
        matern_fit_payload: dict | None = None
        try:
            lag_dist_arr = np.asarray(
                correlogram_results.get('lag_distances', []), dtype=np.float64
            )
            morans_arr = np.array(
                [r.get('morans_i', 0.0) for r in correlogram_data], dtype=np.float64
            )
            if len(lag_dist_arr) >= 4 and len(lag_dist_arr) == len(morans_arr):
                _matern_samples = 400 if self.fast_mode else 800
                _matern_warmup = 200 if self.fast_mode else 500
                fit_res = fit_matern(
                    lag_dist_arr, morans_arr,
                    method="bayes", n_samples=_matern_samples, n_warmup=_matern_warmup, n_chains=2, seed=42,
                )
                matern_fit_payload = fit_res.to_payload()
                matern_fit_payload['variable'] = variable_name
                # Promote to 'matern' when the kernel-shape parameters converge
                # (kappa, sigma2), even when nuisance noise parameters (tau2,
                # obs_sigma2) do not.  tau2/obs_sigma2 are near-identifiable in
                # fast mode (both absorb non-spatial variance) and chronically
                # fail ESS, but they have no bearing on the kernel's spatial
                # shape — only kappa (range) and sigma2 (signal fraction) do.
                # Use payload r_hat values (Python floats via to_payload()) to
                # avoid a ValueError from numpy array truth-value comparison.
                _kappa_rhat = matern_fit_payload.get('r_hat', {}).get('kappa', 9.9)
                _sigma2_rhat = matern_fit_payload.get('r_hat', {}).get('sigma2', 9.9)
                if _kappa_rhat < 1.05 and _sigma2_rhat < 1.05:
                    best_kernel = 'matern'
                print(
                    f"  Matern fit ({fit_res.method}): kappa={fit_res.kappa_mean:.4g} "
                    f"[nu={fit_res.nu}], rmse={fit_res.fit_rmse:.4f}, "
                    f"converged={fit_res.converged}"
                )
        except Exception as _exc:  # noqa: BLE001
            print(f"  [WARNING] Matern fit unavailable for {variable_name}: {_exc}")
            matern_fit_payload = None
            _per_var_degradations.append(
                Degradation(phase="matern_fit", fallback_used="none", reason=str(_exc))
            )

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
            _aniso_samples = 400 if self.fast_mode else 800
            _aniso_warmup = 200 if self.fast_mode else 500
            aniso_res = fit_anisotropy(
                dir_corr['lag_distances'],
                dir_corr['angle_centers_rad'],
                dir_corr['morans_i'],
                n_pairs=dir_corr['n_pairs'],
                method="bayes",
                n_samples=_aniso_samples, n_warmup=_aniso_warmup, n_chains=2, seed=42,
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
            _per_var_degradations.append(
                Degradation(phase="anisotropy_fit", fallback_used="isotropic", reason=str(_exc))
            )

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
            'degradations': _per_var_degradations,
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


# ---------------------------------------------------------------------------
# Pure helper — assembles a CorrelogramReport from the payloads produced by
# main().  Extracted so it can be unit-tested without running the full pipeline.
# ---------------------------------------------------------------------------

def _build_correlogram_report(
    *,
    all_results: dict,
    model_bandwidths: dict,
    optimal_cv_block_size: float,
    block_size_source: str,
    target_variable: "str | None",
    matern_artifact_payload: "dict | None",
    anisotropy_artifact_payload: "dict | None",
    effective_range_matrix_payload: "dict | None",
    cross_range_uplift: "dict | None" = None,
    dataset_profile: "dict | None" = None,
    sampling_config: "dict | None" = None,
    degradations: "list[Degradation] | None" = None,
) -> CorrelogramReport:
    """Assemble a :class:`CorrelogramReport` from Stage 0 assembled payloads.

    Parameters
    ----------
    all_results:
        Per-variable analysis results dict (keyed by variable name).
    model_bandwidths:
        Model-level bandwidth mapping from :meth:`determine_model_bandwidths`.
    optimal_cv_block_size:
        Final spatial-CV block size in metres.
    block_size_source:
        How the block size was determined (``"user"`` / ``"correlogram"`` …).
    target_variable:
        The outcome variable name; excluded from ``bandwidths``.
    matern_artifact_payload, anisotropy_artifact_payload, effective_range_matrix_payload:
        Phase 1-4 payloads (``None`` when the phase degraded).
    cross_range_uplift:
        Phase 4 CV uplift event dict, or ``None``.
    dataset_profile:
        Dataset-level summary from :class:`~sparc.run.dataset_profiler.DatasetProfiler`.
    sampling_config:
        NUTS / permutation budgets used during Stage 0.
    degradations:
        :class:`Degradation` objects collected during ``main()``.
    """
    if degradations is None:
        degradations = []

    bandwidths: dict[str, float] = {
        var: float(res["optimal_bandwidth"])
        for var, res in all_results.items()
        if res.get("optimal_bandwidth") is not None
    }
    variable_effective_ranges: dict[str, float] = {
        var: float(res["effective_range"])
        for var, res in all_results.items()
        if res.get("effective_range") is not None
    }
    stationarity_warnings: list[str] = []
    if matern_artifact_payload is not None:
        stationarity_warnings = list(
            matern_artifact_payload.get("stationarity_warnings", [])
        )

    return CorrelogramReport(
        bandwidths=bandwidths,
        block_size=float(optimal_cv_block_size),
        block_size_source=block_size_source,
        stationarity_warnings=stationarity_warnings,
        variable_effective_ranges=variable_effective_ranges,
        cross_range_uplift=cross_range_uplift,
        dataset_profile=dataset_profile,
        degradations=list(degradations),
        sampling_config=sampling_config,
        matern_fits=matern_artifact_payload,
        anisotropy_fits=anisotropy_artifact_payload,
        effective_range_matrix=effective_range_matrix_payload,
    )


def main(ctx, *, fast_mode=False):
    """
    Main function to run correlogram-based spatial analysis on all variables
    
    Parameters:
    -----------
    ctx : RunContext
        Pipeline-wide dependencies (config, paths, registry).
    fast_mode : bool
        If True, use reduced sample sizes and fewer lags for faster analysis
    """
    print("\n=== SPARC Correlogram-Based Spatial Analysis ===\n")

    # Collect Degradation records from all fallback paths.
    # A Degradation is appended each time a non-critical phase is caught and
    # falls back to a default; the list is attached to the CorrelogramReport
    # at the end of main() so downstream consumers can see execution health
    # without grepping logs.
    _degradations: list[Degradation] = []

    if fast_mode:
        print("[FAST] Running in FAST MODE - reduced precision but much faster")
    
    # Load configuration
    config = ctx.config
    
    # Create Stage 0 output directory using centralized paths
    paths = ctx.paths
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

    if data is None or len(data) == 0:
        raise ValueError(
            "No data available after loading — the dataset is empty. "
            "Verify your CSV path and that 'crs.input' in project.yml matches "
            "the actual coordinate system of your source data."
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
    
    # Use working CRS coordinates (m, ft, etc.); load_and_preprocess_data
    # always writes projected_X / projected_Y in the working CRS units.
    if 'projected_X' in data.columns and 'projected_Y' in data.columns:
        coords = data[['projected_X', 'projected_Y']].values
        coord_cols_for_profiler = ['projected_X', 'projected_Y']
    else:
        coords = data[config['variables']['coordinates']].values
        coord_cols_for_profiler = config['variables']['coordinates']

    # â”€â”€ Use DatasetProfiler for data-driven correlogram parameters â”€â”€â”€
    # Resolve working-CRS unit for correct labels / floor values.
    from sparc.config.crs_resolver import crs_unit as _crs_unit, crs_unit_label as _crs_unit_label
    _working_crs = config['crs'].get('working') or config['crs'].get('target_projected', 'EPSG:4326')
    try:
        _unit = _crs_unit(_working_crs)
        _unit_label = _crs_unit_label(_working_crs)
    except Exception:
        _unit, _unit_label = 'm', 'meters'

    from sparc.run.dataset_profiler import DatasetProfiler
    profiler = DatasetProfiler(
        data,
        coord_cols=coord_cols_for_profiler,
        feature_cols=selected_features,
        crs_unit=_unit,
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
        fast_mode=fast_mode,
        unit_label=_unit,
    )
    
    print(f"Dataset size: {len(data):,} points")
    print(f"Max correlogram distance: {max_distance:,.0f} {_unit_label} (data-driven)")
    print(f"Using sample size: {max_sample_size:,} points for correlogram analysis")
    estimated_time = len(all_variables) * (1 if fast_mode else 2)
    print(f"Estimated analysis time: ~{estimated_time:.0f} minutes")

    # Analyze variables — parallel across all available variables.
    # Each call is independent (same coords, different values + NUTS chains).
    # n_jobs capped at the number of valid variables; torch threads per worker
    # are divided so cores are not oversubscribed across simultaneous workers.
    available_vars = [v for v in all_variables if v in data.columns]
    n_parallel = min(len(available_vars), int(os.cpu_count() or 4))
    # Leave at least 1 thread per worker for PyTorch CPU ops; divide evenly.
    n_torch_threads = max(1, (os.cpu_count() or 4) // max(n_parallel, 1))
    cache_dir_str = str(cache_dir) if cache_dir else ""

    print(
        f"Parallelizing {len(available_vars)} variables across "
        f"{n_parallel} workers ({n_torch_threads} torch thread(s) each)..."
    )

    worker_results = joblib.Parallel(n_jobs=n_parallel, backend="loky", verbose=2)(
        joblib.delayed(_analyze_variable_worker)(
            variable,
            coords,
            data[variable].values,
            stage0_dir,
            max_distance,
            n_lags,
            max_sample_size,
            cache_dir_str,
            fast_mode,
            n_torch_threads,
        )
        for variable in available_vars
    )

    all_results = {name: result for name, result in worker_results}

    # Harvest per-variable degradations from worker results into _degradations.
    # Each worker returns a 'degradations' key; aggregate here in the main process.
    for _var, _res in all_results.items():
        for _d in _res.get("degradations", []):
            _degradations.append(_d)

    # Report any variables that were skipped (not in data columns)
    for variable in all_variables:
        if variable not in data.columns:
            print(f"  Skipped {variable} (not in data)")
    
    # Determine model bandwidths from correlogram results (predictors only)
    model_bandwidths = analyzer.determine_model_bandwidths(all_results, target_variable=target_variable)
    
    # Determine CV block size — honour user override if present
    spatial_cv_cfg = config.get('optimization', {}).get('spatial_cv', {})
    if not spatial_cv_cfg:
        spatial_cv_cfg = config.get('models', {}).get('spatial_cv', {})
    block_size_source = spatial_cv_cfg.get('block_size_source', 'correlogram')
    user_block_size = spatial_cv_cfg.get('block_size')
    effective_range_matrix_payload = None  # initialised here; assigned in Phase 4 below

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
                _degradations.append(
                    Degradation(phase="phase4_uplift", fallback_used="none", reason=str(_exc))
                )

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
        _degradations.append(
            Degradation(phase="matern_fit", fallback_used="none", reason=str(_exc))
        )

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
        _degradations.append(
            Degradation(phase="anisotropy_fit", fallback_used="none", reason=str(_exc))
        )

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
        # 20 permutations gives p-value resolution of 1/21 ≈ 0.048,
        # which matches the downstream significance_alpha=0.05 threshold
        # while running ~2.5× faster than the previous default of 50.
        _n_perm_cc = 0 if fast_mode else 20
        # Use the host hardware profile's outer_jobs to parallelise the
        # per-pair permutation loop (was the silent ~50-minute bottleneck).
        try:
            from sparc.config.hardware_profile import detect_profile as _detect_hw
            _hw = _detect_hw()
            _n_jobs_cc = max(1, int(getattr(_hw, "outer_jobs", 1)))
        except Exception:
            _n_jobs_cc = 1
        cross_correlogram_payload = compute_cross_summary(
            _coords_cc, _values_cc, all_variables,
            max_distance=float(max_distance),
            n_lags=min(int(n_lags), 12),
            n_angle_bins=4,
            n_perm=_n_perm_cc,
            seed=42,
            n_jobs=_n_jobs_cc,
        )
    except Exception as _exc:  # noqa: BLE001
        print(f"Cross-correlogram assembly failed: {_exc}")
        cross_correlogram_payload = None
        _degradations.append(
            Degradation(phase="cross_correlogram", fallback_used="none", reason=str(_exc))
        )

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
            _degradations.append(
                Degradation(phase="cross_correlogram", fallback_used="none", reason=str(_exc))
            )

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
        _degradations.append(
            Degradation(phase="artifact_write", fallback_used="none", reason=str(_exc))
        )

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

        # ─── Phase C-1: Unified KernelField artifact ────────────────────
        # Assemble the matrix-aware ``KernelField`` from the per-variable
        # Matérn fits + anisotropy ellipses + cross-range matrix and
        # persist it under a single canonical id so Stage 2 (GWR), Stage 3
        # (CATE), the report, and the desktop can consume one struct.
        try:
            from sparc.models.kernel_field import KernelField
            _predictor_names = [
                v for v in all_results.keys() if v != target_variable
            ]
            if _predictor_names:
                _kf = KernelField.from_artifacts(
                    outcome_name=target_variable,
                    predictor_names=_predictor_names,
                    matern_artifact=matern_artifact_payload,
                    anisotropy_artifact=anisotropy_artifact_payload,
                    effective_range_matrix=effective_range_matrix_payload,
                )
                _store.write_struct(
                    stage="0",
                    artifact_id="kernel_field",
                    payload=convert_numpy_types(_kf.to_artifact_payload()),
                    producer="correlogram_analysis.main",
                    consumers=[
                        "server:/results/kernel_field",
                        "report:/stage0/kernel_field",
                        "desktop:kernel_field_view",
                        "pipeline:stage2_gwr",
                        "pipeline:stage3_cate",
                        "pipeline:stage4_scenarios",
                    ],
                )
                print(
                    "Unified KernelField written to artifacts.db "
                    f"(stage=0, id=kernel_field, "
                    f"outcome={target_variable}, "
                    f"predictors={len(_predictor_names)})"
                )
        except Exception as _exc:  # noqa: BLE001
            print(f"Unified KernelField persistence skipped: {_exc}")
            _degradations.append(
                Degradation(phase="artifact_write", fallback_used="none", reason=str(_exc))
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

    # ─── Build and persist the typed CorrelogramReport ───────────────────
    # This is the canonical typed seam between Stage 0 and all downstream
    # consumers.  It is built here — after all phases have run — so that the
    # degradations list is complete.
    _sampling_cfg = {
        "fast_mode": fast_mode,
        "n_samples": 400 if fast_mode else 800,
        "n_warmup": 200 if fast_mode else 500,
        "n_chains": 2,
        "n_perm": 0 if fast_mode else 20,
        "sample_cap": max_sample_size,
    }
    correlogram_report = _build_correlogram_report(
        all_results=all_results,
        model_bandwidths=model_bandwidths,
        optimal_cv_block_size=optimal_cv_block_size,
        block_size_source=block_size_source,
        target_variable=target_variable,
        matern_artifact_payload=matern_artifact_payload,
        anisotropy_artifact_payload=anisotropy_artifact_payload,
        effective_range_matrix_payload=effective_range_matrix_payload,
        cross_range_uplift=locals().get("cv_cross_range_uplift"),
        dataset_profile=profile,
        sampling_config=_sampling_cfg,
        degradations=_degradations,
    )
    if _store is not None:
        try:
            _store.write_struct(
                stage="0",
                artifact_id="correlogram_report",
                payload=convert_numpy_types(correlogram_report.to_artifact()),
                producer="correlogram_analysis.main",
                consumers=[
                    "pipeline:stage1_gwen",
                    "pipeline:stage2_mgwr",
                    "server:/results/correlogram",
                    "report:correlogram",
                ],
            )
            _status = "clean" if correlogram_report.is_clean else f"{len(correlogram_report.degradations)} degradation(s)"
            print(f"CorrelogramReport written to artifacts.db (stage=0, id=correlogram_report, status={_status})")
        except Exception as _exc:  # noqa: BLE001
            print(f"CorrelogramReport persistence skipped: {_exc}")

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

    fast_mode = "--fast" in sys.argv

    if fast_mode:
        print("[FAST] FAST MODE ENABLED: Using reduced sample size and fewer lags for quick analysis")
        print("Note: Results may be less precise but much faster\n")

    from sparc.run.orchestrator import RunContext
    from sparc.run.pipeline_paths import PipelinePaths
    from sparc.config.config import load_config as _load_config
    _cfg = _load_config()
    _paths = PipelinePaths.from_config(_cfg)
    try:
        from sparc.registry.run_registry import RunRegistry
        _registry = RunRegistry(_paths.base_dir)
    except Exception:
        _registry = None
    _ctx = RunContext(config=_cfg, paths=_paths, registry=_registry, project_path="")
    main(_ctx, fast_mode=fast_mode)
