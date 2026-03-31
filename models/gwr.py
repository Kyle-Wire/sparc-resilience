"""
Geographically Weighted Regression (GWR) model implementation.
This module provides a GWR model that captures spatial non-stationarity
in regression relationships.
"""
# Suppress warnings BEFORE any imports to prevent console flooding
import warnings
import os

# Suppress sklearn parallel warnings (very noisy in multiprocessing contexts)
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress numpy/scipy warnings from linear algebra operations
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*Ill-conditioned matrix.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set environment variable for subprocesses
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning,ignore::RuntimeWarning'
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import json
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize as scipy_minimize
from typing import Tuple, Optional
from sklearn.model_selection import cross_val_score

# Default physics sign constraints for local regression coefficients.
# -1 = must be negative, +1 = must be positive, 0 = unconstrained.
# These are ONLY used when no constraints are provided via project.yml.
# Prefer loading via config.config.load_monotone_constraints(config).
try:
    from config.config import load_monotone_constraints as _load_mc
    DEFAULT_PHYSICS_SIGN_CONSTRAINTS = _load_mc({})  # empty config → legacy fallback
except Exception:
    DEFAULT_PHYSICS_SIGN_CONSTRAINTS = {
        "Pct_Canopy": -1,
        "Pct_Impervious": +1,
        "NDVI": -1,
        "Albedo": -1,
        "Elevation_m": 0,
        "Distance_from_water_m": 0,
    }

# Backward-compatible alias
PHYSICS_SIGN_CONSTRAINTS = DEFAULT_PHYSICS_SIGN_CONSTRAINTS

class GWRModel(BaseEstimator, RegressorMixin):
    """
    Geographically Weighted Regression (GWR) model with adaptive bandwidth.
    
    Parameters:
    -----------
    bandwidth : int, optional
        Number of nearest neighbors to use for adaptive bandwidth (if variable_bandwidths not provided)
    variable_bandwidths : dict, optional
        Dictionary mapping feature names to their specific bandwidths
    kernel : str, default='gaussian'
        Kernel function to use. One of ['gaussian', 'exponential', 'bisquare']
    coords_cols : list
        Names of columns containing coordinates [x, y]
    alpha : float, default=0.1
        L2 regularization strength
    min_points : int, default=50
        Minimum number of points to use for local regression
    """
    
    def __init__(self, bandwidth: int = None, variable_bandwidths: dict = None, 
                 kernel: str = 'gaussian', coords_cols: list = None,
                 alpha: float = 0.1, min_points: int = 50,
                 use_constrained_regression: bool = True,
                 physics_priors: dict = None,
                 sign_constraints: dict = None):
        self.bandwidth = bandwidth
        self.variable_bandwidths = variable_bandwidths
        self.kernel = kernel
        self.coords_cols = coords_cols
        self.alpha = alpha
        self.min_points = min_points
        self.use_constrained_regression = use_constrained_regression
        self.physics_priors = physics_priors  # {var_name: prior_coeff_per_scaled_unit}
        # sign_constraints: {var_name: -1|0|1}  — overrides module-level defaults
        self._sign_constraints = sign_constraints if sign_constraints is not None else PHYSICS_SIGN_CONSTRAINTS
        self.coefficients_ = None
        self.coefficients_unconstrained_ = None  # Save raw for diagnostics
        self.sign_violations_ = None  # Flag where unconstrained wanted wrong sign
        self.intercepts_ = None
        self.feature_names_ = None
        self.coords_ = None
        self.nn_ = None
        self.scaler = StandardScaler()
        self.M_ik = None  # Spatial modifier matrix for interventions
        
    def _gaussian_kernel(self, dists: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute Gaussian kernel weights"""
        return np.exp(-0.5 * (dists / bandwidth) ** 2)
    
    def _exponential_kernel(self, dists: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute exponential kernel weights"""
        return np.exp(-dists / bandwidth)
    
    def _bisquare_kernel(self, dists: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute bisquare kernel weights"""
        return ((1 - (dists / bandwidth) ** 2) ** 2) * (dists <= bandwidth)
    
    def _get_kernel_function(self):
        """Get the appropriate kernel function"""
        kernel_functions = {
            'gaussian': self._gaussian_kernel,
            'exponential': self._exponential_kernel,
            'bisquare': self._bisquare_kernel
        }
        return kernel_functions.get(self.kernel.lower(), self._gaussian_kernel)
    
    def _local_regression(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray, 
                         point_idx: int) -> Tuple[np.ndarray, float]:
        """
        Perform local weighted regression at a single point
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        coords : array-like of shape (n_samples, 2)
            Coordinate pairs [x, y]
        point_idx : int
            Index of the point to fit local regression at
            
        Returns:
        --------
        coefficients : np.ndarray
            Local regression coefficients
        intercept : float
            Local regression intercept
        """
        n_features = X.shape[1]
        
        try:
            if self.variable_bandwidths:
                # Use variable-specific bandwidths (MGWR)
                # Use k-NN for fast neighbor lookup instead of full distance matrix
                if self.nn_ is not None:
                    neighbor_distances, indices = self.nn_.kneighbors(coords[point_idx:point_idx+1])
                    neighbor_distances = neighbor_distances[0]
                    indices = indices[0]
                else:
                    # Fallback to full distance computation
                    distances = cdist(coords[point_idx:point_idx+1], coords)[0]
                    indices = np.argsort(distances)[:min(500, len(distances))]
                    neighbor_distances = distances[indices]
                
                # Filter to points within max bandwidth
                max_bandwidth = max(self.variable_bandwidths.values())
                within_bandwidth = neighbor_distances <= max_bandwidth * 1.1
                
                if np.sum(within_bandwidth) >= self.min_points:
                    valid_indices = indices[within_bandwidth]
                    local_distances = neighbor_distances[within_bandwidth]
                else:
                    # Use min_points nearest neighbors
                    valid_indices = indices[:self.min_points]
                    local_distances = neighbor_distances[:self.min_points]
                
                X_local = X[valid_indices]
                y_local = y[valid_indices]
                
                # Use the mean bandwidth for kernel weighting
                bandwidth = np.mean(list(self.variable_bandwidths.values())) + 1e-10
                
            else:
                # Use single global bandwidth
                if self.nn_ is not None:
                    neighbor_distances, indices = self.nn_.kneighbors(coords[point_idx:point_idx+1])
                    neighbor_distances = neighbor_distances[0]
                    indices = indices[0]
                    
                    if len(indices) < self.min_points:
                        more_distances, more_indices = self.nn_.kneighbors(
                            coords[point_idx:point_idx+1],
                            n_neighbors=self.min_points
                        )
                        neighbor_distances = more_distances[0]
                        indices = more_indices[0]
                        
                    X_local = X[indices]
                    y_local = y[indices]
                    local_distances = neighbor_distances
                    bandwidth = neighbor_distances[-1] + 1e-10  # Prevent zero bandwidth
                else:
                    closest_indices = np.argsort(distances)[:self.min_points]
                    local_distances = distances[closest_indices]
                    X_local = X[closest_indices]
                    y_local = y[closest_indices]
                    bandwidth = local_distances[-1] + 1e-10  # Prevent zero bandwidth
            
            # Validate we have enough data for regression
            if len(X_local) < n_features + 1:
                return np.zeros(n_features), np.mean(y_local) if len(y_local) > 0 else 0.0
            
            # Calculate weights using kernel function
            kernel_func = self._get_kernel_function()
            weights = kernel_func(local_distances, bandwidth)
            
            # Ensure weights are valid (no zeros or NaNs)
            weights = np.clip(weights, 1e-10, None)
            if np.any(~np.isfinite(weights)):
                weights = np.ones(len(weights)) / len(weights)
            
            # Add regularization term
            n_local_features = X_local.shape[1]
            reg_matrix = self.alpha * np.eye(n_local_features + 1)  # +1 for intercept
            reg_matrix[0, 0] = 0  # Don't regularize intercept
            
            # Build prior-centered ridge target: β_prior (0 for intercept)
            beta_prior = np.zeros(n_local_features + 1)
            if self.physics_priors and self.feature_names_ is not None:
                for fidx in range(n_local_features):
                    fname = self.feature_names_[fidx] if fidx < len(self.feature_names_) else ''
                    if fname in self.physics_priors:
                        beta_prior[fidx + 1] = self.physics_priors[fname]  # +1 skips intercept
            
            # Weighted least squares with regularization
            W = np.diag(weights)
            X_aug = np.column_stack([np.ones(len(X_local)), X_local])
            X_w = np.sqrt(W) @ X_aug
            y_w = np.sqrt(W) @ y_local
            
            # Check for numerical issues before solving
            if np.any(~np.isfinite(X_w)) or np.any(~np.isfinite(y_w)):
                return np.zeros(n_features), np.mean(y_local)
            
            # Solve: (X'WX + αI)β = X'Wy + αβ_prior  (prior-centered ridge)
            XtX = X_w.T @ X_w + reg_matrix
            Xty = X_w.T @ y_w + reg_matrix @ beta_prior
            
            try:
                # Try lstsq first (most robust)
                beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=1e-10)
            except:
                try:
                    # Fallback to solve with more regularization
                    XtX += 0.1 * np.eye(n_local_features + 1)
                    beta = np.linalg.solve(XtX, Xty)
                except:
                    # Last resort: use pseudo-inverse
                    try:
                        beta = np.linalg.pinv(XtX) @ Xty
                    except:
                        # Ultimate fallback: return simple mean prediction
                        return np.zeros(n_features), np.mean(y_local)
            
            # Validate output
            if np.any(~np.isfinite(beta)):
                return np.zeros(n_features), np.mean(y_local)
            
            return beta[1:], beta[0]  # coefficients, intercept
            
        except Exception:
            # If anything fails, return safe defaults
            return np.zeros(n_features), np.mean(y) if len(y) > 0 else 0.0
    
    def _constrained_local_regression(self, X, y, coords, point_idx):
        """
        Physics-constrained local regression using bounded optimisation.
        
        Solves:  min  ||W^{1/2}(X*beta - y)||^2 + alpha*||beta||^2
                 s.t.  beta_canopy <= 0, beta_impervious >= 0, etc.
        
        The magnitude varies freely; only the sign is enforced.
        Returns both constrained and unconstrained coefficients for diagnostics.
        """
        # First get the unconstrained solution
        coefs_unc, intercept_unc = self._local_regression(X, y, coords, point_idx)
        
        # Build sign bounds from physics constraints
        feature_names = self.feature_names_ if self.feature_names_ is not None else []
        n_features = X.shape[1]
        
        # Determine bounds for each coefficient
        bounds = []
        has_constraint = False
        for fidx in range(n_features):
            fname = feature_names[fidx] if fidx < len(feature_names) else ''
            constraint_sign = self._sign_constraints.get(fname, 0)
            if constraint_sign == -1:
                bounds.append((-np.inf, 0.0))  # must be <= 0
                has_constraint = True
            elif constraint_sign == 1:
                bounds.append((0.0, np.inf))   # must be >= 0
                has_constraint = True
            else:
                bounds.append((-np.inf, np.inf))  # unconstrained
        
        # If no constraint is active, just return unconstrained
        if not has_constraint:
            return coefs_unc, intercept_unc, coefs_unc, np.zeros(n_features, dtype=int)
        
        # Check if unconstrained already satisfies constraints
        all_ok = True
        violations = np.zeros(n_features, dtype=int)
        for fidx in range(n_features):
            lo, hi = bounds[fidx]
            if coefs_unc[fidx] < lo or coefs_unc[fidx] > hi:
                all_ok = False
                violations[fidx] = 1
        
        if all_ok:
            return coefs_unc, intercept_unc, coefs_unc, violations
        
        # Need to solve the constrained QP
        try:
            # Get local data (replicate the kernel weighting from _local_regression)
            if self.nn_ is not None:
                neighbor_distances, indices = self.nn_.kneighbors(coords[point_idx:point_idx+1])
                neighbor_distances = neighbor_distances[0]
                indices = indices[0]
            else:
                distances = cdist(coords[point_idx:point_idx+1], coords)[0]
                indices = np.argsort(distances)[:self.min_points]
                neighbor_distances = distances[indices]
            
            X_local = X[indices]
            y_local = y[indices]
            
            if self.variable_bandwidths:
                bandwidth = np.mean(list(self.variable_bandwidths.values())) + 1e-10
            else:
                bandwidth = neighbor_distances[-1] + 1e-10
            
            kernel_func = self._get_kernel_function()
            weights = kernel_func(neighbor_distances, bandwidth)
            weights = np.clip(weights, 1e-10, None)
            
            W_sqrt = np.sqrt(weights)
            
            # Build prior vector for prior-centered regularisation
            beta_prior_vec = np.zeros(n_features)
            if self.physics_priors and self.feature_names_ is not None:
                for fidx in range(n_features):
                    fname = self.feature_names_[fidx] if fidx < len(self.feature_names_) else ''
                    if fname in self.physics_priors:
                        beta_prior_vec[fidx] = self.physics_priors[fname]
            
            # Objective: ||W^{1/2}(X*beta + intercept - y)||^2 + alpha*||beta - beta_prior||^2
            def objective(params):
                beta = params[1:]      # coefficients
                b0 = params[0]         # intercept
                pred = X_local @ beta + b0
                resid = W_sqrt * (pred - y_local)
                diff = beta - beta_prior_vec
                return 0.5 * np.dot(resid, resid) + 0.5 * self.alpha * np.dot(diff, diff)
            
            # Intercept is unconstrained; coefficients have physics bounds
            full_bounds = [(-np.inf, np.inf)] + bounds  # [intercept] + coefficients
            
            x0 = np.concatenate([[intercept_unc], coefs_unc])
            
            result = scipy_minimize(objective, x0, method='L-BFGS-B',
                                     bounds=full_bounds,
                                     options={'maxiter': 200, 'ftol': 1e-10})
            
            coefs_con = result.x[1:]
            intercept_con = result.x[0]
            
            return coefs_con, intercept_con, coefs_unc, violations
            
        except Exception:
            # Fall back to unconstrained with manual clipping
            coefs_con = coefs_unc.copy()
            for fidx in range(n_features):
                lo, hi = bounds[fidx]
                coefs_con[fidx] = np.clip(coefs_con[fidx], lo, hi)
            return coefs_con, intercept_unc, coefs_unc, violations
    
    def tune_bandwidth(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray, 
                      bandwidth_range: list = None, cv_folds: int = 5) -> int:
        """
        Tune bandwidth using cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        coords : np.ndarray
            Spatial coordinates
        bandwidth_range : list, optional
            List of bandwidth values to try. If None, will use default range.
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        int : Optimal bandwidth value
        """
        if bandwidth_range is None:
            # Default bandwidth range based on data size
            n_samples = len(X)
            min_bw = max(50, X.shape[1] + 2)  # At least number of features + 2
            max_bw = min(n_samples // 2, 1000)  # At most half the data or 1000
            bandwidth_range = list(range(min_bw, max_bw + 1, 50))
        
        print(f"Tuning bandwidth from range: {bandwidth_range}")
        
        best_score = -np.inf
        best_bandwidth = bandwidth_range[0]
        
        for bw in bandwidth_range:
            try:
                # Create temporary model with current bandwidth
                temp_model = GWRModel(bandwidth=bw, kernel=self.kernel, 
                                    coords_cols=self.coords_cols, 
                                    alpha=self.alpha, min_points=self.min_points)
                
                # Perform cross-validation
                scores = cross_val_score(temp_model, X, y, cv=cv_folds, 
                                       scoring='r2', n_jobs=1)
                mean_score = scores.mean()
                
                print(f"  Bandwidth {bw}: CV R² = {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_bandwidth = bw
                    
            except Exception as e:
                print(f"  Bandwidth {bw} failed: {str(e)}")
                continue
        
        print(f"Selected bandwidth: {best_bandwidth} (CV R² = {best_score:.4f})")
        return best_bandwidth
    
    def fit(self, X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None, 
            tune_bandwidth: bool = False, bandwidth_range: list = None,
            extract_coefficients: bool = False, output_path: str = None) -> 'GWRModel':
        """
        Fit GWR model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        coords : array-like of shape (n_samples, 2), optional
            Coordinate pairs [x, y]. Required if coords_cols not provided.
        tune_bandwidth : bool, default=False
            Whether to tune the bandwidth
        bandwidth_range : list, optional
            List of bandwidth values to try if tuning
        extract_coefficients : bool, default=False
            Whether to extract and save MGWR local coefficients (only for full model)
        output_path : str, optional
            Path to save coefficient CSV (default: 'mgwr_local_coefficients.csv')
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            
        X = np.asarray(X)
        y = np.asarray(y)
        
        if coords is None and self.coords_cols is None:
            raise ValueError("Either coords array or coords_cols must be provided")
            
        if coords is None:
            coords = X[:, [X.shape[1] - 2, X.shape[1] - 1]]  # Assume last two columns are coordinates
            X = X[:, :-2]  # Remove coordinate columns from features
        
        # Tune bandwidth if requested
        if tune_bandwidth:
            print("Tuning bandwidth...")
            self.bandwidth = self.tune_bandwidth(X, y, coords, bandwidth_range)
            
        self.coords_ = coords
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Initialize nearest neighbors based on bandwidth configuration
        if self.variable_bandwidths:
            print("Using MGWR with variable-specific bandwidths:")
            for var, bw in self.variable_bandwidths.items():
                if var in self.feature_names_:
                    print(f"  {var}: {bw}m")
            # For MGWR, precompute k-nearest neighbors for efficiency
            # Use max bandwidth to determine k (convert meters to approximate neighbor count)
            max_bw = max(self.variable_bandwidths.values())
            # Estimate k based on spatial density - use more neighbors for larger bandwidths
            estimated_k = min(500, max(self.min_points * 2, len(coords) // 10))
            print(f"  Using k-NN acceleration with k={estimated_k} for neighbor lookup")
            self.nn_ = NearestNeighbors(n_neighbors=estimated_k)
            self.nn_.fit(coords)
        elif self.bandwidth and self.bandwidth > 0:
            self.nn_ = NearestNeighbors(n_neighbors=self.bandwidth)
            self.nn_.fit(coords)
        else:
            # Default fallback
            self.bandwidth = min(100, len(X) // 2)
            self.nn_ = NearestNeighbors(n_neighbors=self.bandwidth)
            self.nn_.fit(coords)
        
        # Fit local models at each point - with parallel batch processing for speed
        n_samples, n_features = X.shape
        self.coefficients_ = np.zeros((n_samples, n_features))
        self.coefficients_unconstrained_ = np.zeros((n_samples, n_features))
        self.sign_violations_ = np.zeros((n_samples, n_features), dtype=int)
        self.intercepts_ = np.zeros(n_samples)
        
        # Use parallel processing for large datasets
        use_parallel = n_samples > 500
        use_constrained = self.use_constrained_regression
        
        if use_constrained:
            print(f"Using physics-constrained local regression (bounded QP)...")
            active_constraints = {k: v for k, v in self._sign_constraints.items() 
                                   if v != 0 and k in (self.feature_names_ or [])}
            if active_constraints:
                for k, v in active_constraints.items():
                    direction = '<= 0' if v == -1 else '>= 0'
                    print(f"  {k}: beta {direction}")
        
        if use_parallel:
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Use threads (not processes) to avoid pickle issues with model state
            n_workers = min(mp.cpu_count(), 8)  # Cap at 8 threads for GWR
            batch_size = max(100, n_samples // (n_workers * 4))  # Adaptive batch size
            
            print(f"Using parallel GWR fitting: {n_workers} workers, batch_size={batch_size}")
            
            def fit_batch(batch_indices):
                """Fit local regression for a batch of points"""
                results = []
                for i in batch_indices:
                    try:
                        if use_constrained:
                            coef, intercept, coef_unc, violations = self._constrained_local_regression(X, y, coords, i)
                            results.append((i, coef, intercept, coef_unc, violations))
                        else:
                            coef, intercept = self._local_regression(X, y, coords, i)
                            results.append((i, coef, intercept, coef, np.zeros(n_features, dtype=int)))
                    except Exception as e:
                        # Return zeros on failure
                        results.append((i, np.zeros(n_features), np.mean(y), np.zeros(n_features), np.zeros(n_features, dtype=int)))
                return results
            
            # Create batches
            batches = [list(range(i, min(i + batch_size, n_samples))) 
                      for i in range(0, n_samples, batch_size)]
            
            completed = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(fit_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    try:
                        results = future.result(timeout=300)  # 5 min timeout per batch
                        for i, coef, intercept, coef_unc, violations in results:
                            self.coefficients_[i] = coef
                            self.intercepts_[i] = intercept
                            self.coefficients_unconstrained_[i] = coef_unc
                            self.sign_violations_[i] = violations
                        completed += len(futures[future])
                        if completed % 1000 < batch_size:
                            print(f"Fitted {completed}/{n_samples} local models")
                    except Exception as e:
                        # Handle failed batch - fill with fallback values
                        batch = futures[future]
                        print(f"WARNING: Batch starting at {batch[0]} failed: {e}")
                        for i in batch:
                            self.coefficients_[i] = np.zeros(n_features)
                            self.intercepts_[i] = np.mean(y)
                            self.coefficients_unconstrained_[i] = np.zeros(n_features)
                            self.sign_violations_[i] = np.zeros(n_features, dtype=int)
                        completed += len(batch)
            
            print(f"Fitted {n_samples}/{n_samples} local models (parallel)")
        else:
            # Sequential fitting for small datasets
            for i in range(n_samples):
                if use_constrained:
                    coef, intercept, coef_unc, violations = self._constrained_local_regression(X, y, coords, i)
                    self.coefficients_[i] = coef
                    self.intercepts_[i] = intercept
                    self.coefficients_unconstrained_[i] = coef_unc
                    self.sign_violations_[i] = violations
                else:
                    coef, intercept = self._local_regression(X, y, coords, i)
                    self.coefficients_[i] = coef
                    self.intercepts_[i] = intercept
                    self.coefficients_unconstrained_[i] = coef
                
                if (i + 1) % 1000 == 0:
                    print(f"Fitted {i + 1}/{n_samples} local models")
        
        # Report sign violation statistics
        if use_constrained and self.feature_names_ is not None:
            total_violations = self.sign_violations_.sum(axis=0)
            for fidx, fname in enumerate(self.feature_names_):
                if fname in self._sign_constraints and self._sign_constraints[fname] != 0:
                    pct = total_violations[fidx] / n_samples * 100
                    print(f"  {fname}: {int(total_violations[fidx])} sign violations corrected ({pct:.1f}%)")
        
        # ===============================================================
        # === EXTRACT TRUE MGWR LOCAL COEFFICIENT SURFACES            ===
        # === (Only when extract_coefficients=True for full model)    ===
        # ===============================================================
        if extract_coefficients:
            import os
            
            # Fetch the raw MGWR local parameter matrix (skip intercept)
            local_betas = self.coefficients_    # shape: (n_cells, n_predictors) — already constrained
            local_betas_unc = self.coefficients_unconstrained_  # unconstrained for diagnostics

            # Retrieve predictor names from the same order X was built in.
            predictor_names = self.feature_names_  # must match X column order

            # ---------------------------------------------------------------
            # Build DataFrame of CONSTRAINED coefficients (primary output)
            # ---------------------------------------------------------------
            coef_df = pd.DataFrame(local_betas, columns=predictor_names)
            coef_df["POINT_X"] = coords[:, 0]
            coef_df["POINT_Y"] = coords[:, 1]

            if output_path is None:
                output_path = "mgwr_local_coefficients.csv"
            
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            # Save constrained coefficients
            constrained_path = os.path.join(output_dir, f"{base_name}.csv")
            coef_df.to_csv(constrained_path, index=False)
            print(f"✓ MGWR local coefficients (CONSTRAINED) saved to {constrained_path}")
            
            # ---------------------------------------------------------------
            # Save UNCONSTRAINED coefficients for scientific diagnostics
            # ---------------------------------------------------------------
            coef_df_unc = pd.DataFrame(local_betas_unc, columns=predictor_names)
            coef_df_unc["POINT_X"] = coords[:, 0]
            coef_df_unc["POINT_Y"] = coords[:, 1]
            
            # Add sign violation flags
            if self.sign_violations_ is not None:
                for fidx, fname in enumerate(predictor_names):
                    if fname in self._sign_constraints and self._sign_constraints[fname] != 0:
                        coef_df_unc[f'{fname}_sign_violation'] = self.sign_violations_[:, fidx]
            
            unconstrained_path = os.path.join(output_dir, f"{base_name}_unconstrained.csv")
            coef_df_unc.to_csv(unconstrained_path, index=False)
            print(f"✓ MGWR local coefficients (UNCONSTRAINED + diagnostics) saved to {unconstrained_path}")

            # ---------------------------------------------------------------
            # Store coefficients as M_ik attribute for spatial intelligence
            # Store the CONSTRAINED version for downstream use
            # ---------------------------------------------------------------
            self.M_ik = coef_df[predictor_names].values
            print(f"✓ M_ik coefficient matrix stored: shape {self.M_ik.shape}")
        
        return self
    
    def predict(self, X: np.ndarray, coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using fitted GWR model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        coords : array-like of shape (n_samples, 2), optional
            Coordinate pairs [x, y] for prediction points
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted values
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X)
        
        if coords is None and self.coords_cols is not None:
            coords = X[:, [X.shape[1] - 2, X.shape[1] - 1]]
            X = X[:, :-2]
            
        if coords is None:
            coords = self.coords_  # Use training coordinates
            
        # Scale features
        X = self.scaler.transform(X)
            
        # Batch nearest-neighbor lookup (avoids per-point loop overhead)
        if self.nn_ is not None:
            _, indices = self.nn_.kneighbors(coords)
            nearest_indices = indices[:, 0]
        else:
            from sklearn.neighbors import BallTree
            tree = BallTree(self.coords_)
            _, indices = tree.query(coords, k=1)
            nearest_indices = indices[:, 0]

        # Vectorised prediction: sum of element-wise coef*feature + intercept
        predictions = np.einsum('ij,ij->i',
                                self.coefficients_[nearest_indices], X) \
                      + self.intercepts_[nearest_indices]
            
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray, coords: Optional[np.ndarray] = None) -> float:
        """
        Return R² score
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values
        coords : array-like of shape (n_samples, 2), optional
            Coordinate pairs [x, y] for test points
            
        Returns:
        --------
        float
            R² score
        """
        y_pred = self.predict(X, coords)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def save_coefficients(self, output_path: str, feature_names: list = None) -> None:
        """
        Save GWR coefficients as a GeoPackage
        
        Parameters:
        -----------
        output_path : str
            Path to save coefficients GeoPackage
        feature_names : list, optional
            Names of features (for column names)
        """
        if feature_names is None:
            feature_names = self.feature_names_
            
        # Create coefficient DataFrame
        coef_df = pd.DataFrame(
            self.coefficients_,
            columns=[f'{name}_coef' for name in feature_names]
        )
        coef_df['intercept'] = self.intercepts_
        
        # Add coordinates
        coef_df['x'] = self.coords_[:, 0]
        coef_df['y'] = self.coords_[:, 1]
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            coef_df,
            geometry=gpd.points_from_xy(coef_df.x, coef_df.y),
            crs=getattr(self, '_output_crs', 'EPSG:26919')
        )
        
        # Save to GeoPackage
        gdf.to_file(output_path, driver='GPKG')

    def get_coefficient_summary(self):
        """
        Get summary statistics for local coefficients
        
        Returns:
        --------
        DataFrame
            DataFrame containing coefficient statistics
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before getting coefficient summary")
            
        summary = pd.DataFrame({
            'predictor': self.feature_names_,
            'mean': np.mean(self.coefficients_, axis=0),
            'std': np.std(self.coefficients_, axis=0),
            'min': np.min(self.coefficients_, axis=0),
            'max': np.max(self.coefficients_, axis=0)
        })
        
        return summary
    
    def save_bandwidth(self, output_path):
        """
        Save the selected bandwidth value to a JSON file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the bandwidth JSON
        """
        if self.bandwidth is None:
            raise ValueError("Bandwidth not selected. Call fit() first.")
        
        with open(output_path, 'w') as f:
            json.dump({'adaptive_k': self.bandwidth}, f)
        print(f"\nBandwidth value saved to: {output_path}")
    
    def load_bandwidth(self, input_path):
        """
        Load a previously saved bandwidth value.
        
        Parameters:
        -----------
        input_path : str
            Path to the bandwidth JSON file
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
            self.bandwidth = data['adaptive_k']
        print(f"Loaded bandwidth value: {self.bandwidth}")
    
    # === OOF SPATIAL INTELLIGENCE EXTRACTION (EPIC 3) ===
    
    def compute_global_coefficients(self) -> np.ndarray:
        """
        Compute global (non-spatial) baseline coefficients.
        
        This uses the median of all local coefficients as a robust
        estimate of the global effect.
        
        Returns:
        --------
        np.ndarray of shape (n_features,)
            Global coefficient values (one per feature)
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before computing global coefficients")
        
        # Use median for robustness to outliers
        global_coefs = np.median(self.coefficients_, axis=0)
        return global_coefs
    
    def compute_local_global_ratios(
        self, 
        min_ratio: float = 0.6,
        max_ratio: float = 1.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute M_ik = |β_local| / |β_global| spatial modifier ratios.
        
        This quantifies how much each location deviates from the global average.
        Ratios > 1.0 indicate stronger local effects; < 1.0 indicate weaker.
        
        Parameters:
        -----------
        min_ratio : float, default=0.6
            Minimum allowed ratio (clip lower values)
        max_ratio : float, default=1.6
            Maximum allowed ratio (clip upper values)
        
        Returns:
        --------
        modifier_ratios : np.ndarray of shape (n_samples, n_features)
            M_ik values per location and variable
        global_coefficients : np.ndarray of shape (n_features,)
            Global baseline coefficients
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before computing ratios")
        
        # Get global coefficients
        global_coefs = self.compute_global_coefficients()
        
        # Compute ratios: |β_local| / |β_global|
        modifier_ratios = np.abs(self.coefficients_) / (np.abs(global_coefs) + 1e-10)
        
        # Clip to reasonable bounds
        modifier_ratios = np.clip(modifier_ratios, min_ratio, max_ratio)
        
        return modifier_ratios, global_coefs
    
    def export_modifier_maps(
        self,
        output_dir: str,
        variable_names: Optional[list] = None,
        min_ratio: float = 0.6,
        max_ratio: float = 1.6,
        save_formats: list = ['gpkg', 'csv']
    ) -> dict:
        """
        Export spatial modifier maps (M_ik) to files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save modifier maps
        variable_names : list, optional
            Names of variables. If None, uses self.feature_names_
        min_ratio : float, default=0.6
            Minimum modifier ratio
        max_ratio : float, default=1.6
            Maximum modifier ratio
        save_formats : list, default=['gpkg', 'csv']
            Formats to save: 'gpkg' (GeoPackage), 'csv', 'npy' (NumPy)
        
        Returns:
        --------
        dict
            Paths to saved files
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if variable_names is None:
            variable_names = self.feature_names_
        
        # Compute modifier ratios
        modifier_ratios, global_coefs = self.compute_local_global_ratios(
            min_ratio, max_ratio
        )
        
        saved_files = {}
        
        # Create DataFrame with modifiers
        modifier_df = pd.DataFrame(
            modifier_ratios,
            columns=[f'{var}_modifier' for var in variable_names]
        )
        modifier_df['x'] = self.coords_[:, 0]
        modifier_df['y'] = self.coords_[:, 1]
        
        # Add global coefficients as metadata columns
        for i, var in enumerate(variable_names):
            modifier_df[f'{var}_global_coef'] = global_coefs[i]
            modifier_df[f'{var}_local_coef'] = self.coefficients_[:, i]
        
        # Save as CSV
        if 'csv' in save_formats:
            csv_path = output_dir / 'gwr_modifier_maps.csv'
            modifier_df.to_csv(csv_path, index=False)
            saved_files['csv'] = str(csv_path)
            print(f"Saved modifier CSV: {csv_path}")
        
        # Save as GeoPackage
        if 'gpkg' in save_formats:
            gdf = gpd.GeoDataFrame(
                modifier_df,
                geometry=gpd.points_from_xy(modifier_df.x, modifier_df.y),
                crs=getattr(self, '_output_crs', 'EPSG:26919')
            )
            gpkg_path = output_dir / 'gwr_modifier_maps.gpkg'
            gdf.to_file(gpkg_path, driver='GPKG')
            saved_files['gpkg'] = str(gpkg_path)
            print(f"Saved modifier GeoPackage: {gpkg_path}")
        
        # Save as NumPy arrays
        if 'npy' in save_formats:
            npy_path = output_dir / 'gwr_modifier_maps.npz'
            np.savez(
                npy_path,
                modifier_ratios=modifier_ratios,
                global_coefficients=global_coefs,
                local_coefficients=self.coefficients_,
                coords=self.coords_,
                variable_names=variable_names
            )
            saved_files['npy'] = str(npy_path)
            print(f"Saved modifier NumPy: {npy_path}")
        
        # Save summary statistics
        summary_path = output_dir / 'gwr_modifier_summary.json'
        summary = {
            'variable_names': variable_names,
            'global_coefficients': {
                var: float(global_coefs[i]) 
                for i, var in enumerate(variable_names)
            },
            'modifier_statistics': {}
        }
        
        for i, var in enumerate(variable_names):
            summary['modifier_statistics'][var] = {
                'mean': float(np.mean(modifier_ratios[:, i])),
                'std': float(np.std(modifier_ratios[:, i])),
                'min': float(np.min(modifier_ratios[:, i])),
                'max': float(np.max(modifier_ratios[:, i])),
                'median': float(np.median(modifier_ratios[:, i])),
                'pct_clipped_low': float(np.mean(modifier_ratios[:, i] == min_ratio) * 100),
                'pct_clipped_high': float(np.mean(modifier_ratios[:, i] == max_ratio) * 100)
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = str(summary_path)
        print(f"Saved modifier summary: {summary_path}")
        
        return saved_files
    
    def get_modifier_stability_metrics(
        self,
        other_fold_modifiers: list,
        variable_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Compute stability metrics across CV folds.
        
        This checks how consistent the modifier ratios are across different
        folds, flagging cells with high inter-fold variance as "unstable".
        
        Parameters:
        -----------
        other_fold_modifiers : list of np.ndarray
            Modifier ratios from other folds (each shape (n_samples, n_features))
        variable_names : list, optional
            Variable names
        
        Returns:
        --------
        pd.DataFrame
            Stability metrics per variable with columns:
            - variable, mean_interfold_std, pct_unstable, median_modifier
        """
        if variable_names is None:
            variable_names = self.feature_names_
        
        # Get this fold's modifiers
        this_fold_modifiers, _ = self.compute_local_global_ratios()
        
        # Stack all fold modifiers
        all_modifiers = [this_fold_modifiers] + other_fold_modifiers
        modifier_stack = np.stack(all_modifiers, axis=0)  # (n_folds, n_samples, n_features)
        
        # Compute inter-fold standard deviation per cell
        interfold_std = np.std(modifier_stack, axis=0)  # (n_samples, n_features)
        
        # Define "unstable" as SD > 0.15 (15% variation across folds)
        unstable_threshold = 0.15
        pct_unstable = np.mean(interfold_std > unstable_threshold, axis=0) * 100
        
        stability_metrics = []
        for i, var in enumerate(variable_names):
            stability_metrics.append({
                'variable': var,
                'mean_interfold_std': np.mean(interfold_std[:, i]),
                'pct_unstable': pct_unstable[i],
                'median_modifier': np.median(this_fold_modifiers[:, i])
            })
        
        return pd.DataFrame(stability_metrics) 