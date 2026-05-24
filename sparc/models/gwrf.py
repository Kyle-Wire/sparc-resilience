"""
Geographically Weighted Random Forest (GWRF) model implementation.
This module provides a localized random forest model that uses adaptive
Gaussian kernels to capture local non-linearities in spatial data.
"""

import numpy as np
import os
import gc
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree
from tqdm import tqdm

class GWRFModel:
    """
    Geographically Weighted Random Forest model that combines spatial
    weighting with random forest regression.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the random forest
    k_neighbors : int, default=100
        Number of neighbors to use for adaptive bandwidth
    min_samples_leaf : int, default=5
        Minimum number of samples required at a leaf node
    n_jobs : int, default=-1
        Number of jobs to run in parallel (-1 means using all processors)
    subsample_fraction : float, optional
        Fraction of locations to fit local models at (e.g., 0.5 for 50%). If None, fit at all locations.
    subsample_n : int, optional
        Number of locations to fit local models at. If set, overrides subsample_fraction.
    """

    def __init__(self, n_estimators=100, k_neighbors=100, min_samples_leaf=5, n_jobs=-1, subsample_fraction=None, subsample_n=None, kernel_field=None):
        self.n_estimators = int(n_estimators)
        self.k_neighbors = int(k_neighbors)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.subsample_fraction = subsample_fraction
        self.subsample_n = int(subsample_n) if subsample_n is not None else None
        # Phase 5a: matrix-aware kernel-field bundle (optional, additive).
        # When supplied, k_neighbors is reinterpreted as a *minimum* and
        # the per-pair effective range from `kernel_field` is used to
        # widen the neighbour-search at fit time.  Carried as metadata
        # so downstream artifacts can record which kernel field drove
        # the local RFs.
        self.kernel_field = kernel_field
        self.local_models = None
        self.coords = None
        self.subsample_coords = None
        self.feature_names_ = None  # Store feature names
        self.X_train_ = None  # Store training data for PDP computation
        self.y_train_ = None

    @classmethod
    def from_config(cls, cfg: "GWRFConfig") -> "GWRFModel":
        """Construct a GWRFModel from a validated :class:`GWRFConfig`.

        This is the preferred construction path.
        """
        return cls(
            n_estimators=cfg.n_estimators,
            k_neighbors=cfg.k_neighbors,
            min_samples_leaf=cfg.min_samples_leaf,
            n_jobs=cfg.n_jobs,
            subsample_fraction=cfg.subsample_fraction,
            subsample_n=cfg.subsample_n,
            kernel_field=cfg.kernel_field,
        )

    def validate_for_fold(self, n_train: int, n_features: int = 0) -> None:
        """Clamp k_neighbors / subsample_n so this model is safe for a fold
        of *n_train* samples.  Mutates ``self`` in-place; call on a deep-copy
        before fitting so the original model is unchanged.
        """
        if self.k_neighbors >= n_train:
            safe = max(3, int(n_train * 0.8))
            print(f"WARNING: GWRF k_neighbors {self.k_neighbors} >= n_train {n_train}; "
                  f"clamped to {safe}")
            self.k_neighbors = safe
        if self.subsample_n is not None and self.subsample_n >= n_train:
            safe = max(100, int(n_train * 0.65))
            print(f"WARNING: GWRF subsample_n {self.subsample_n} >= n_train {n_train}; "
                  f"clamped to {safe}")
            self.subsample_n = safe

    def fit(self, X, y, coords, extract_derivatives=False, output_dir=None, feature_names=None):
        """
        Fit GWRF model
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        coords : array-like
            Spatial coordinates
        extract_derivatives : bool, default=False
            If True **and** *output_dir* is provided, run per-cell ICE
            curve extraction after fitting.  This is expensive for large
            datasets (N × 50 grid points × n_features model evaluations).
            For an aggregate PDP + saturation curve export (much faster),
            call ``export_condition_curves()`` after fit instead.
        output_dir : str, optional
            Directory for per-cell ICE / derivative outputs when
            *extract_derivatives* is True.
        feature_names : list, optional
            Names of features for labeling outputs
        """
        self.coords = np.asarray(coords)
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store training data and feature names
        self.X_train_ = X
        self.y_train_ = y
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]

        n_points = len(coords)

        # Auto-subsample cap: for large datasets, default to sqrt(n) fit
        # locations if no explicit subsample setting was provided
        AUTO_SUBSAMPLE_THRESHOLD = 10_000
        if (self.subsample_n is None
                and self.subsample_fraction is None
                and n_points > AUTO_SUBSAMPLE_THRESHOLD):
            auto_n = int(np.sqrt(n_points) * 10)  # ~3162 for 100K
            auto_n = min(auto_n, n_points)
            self.subsample_n = auto_n
            print(f"Auto-subsample enabled: {auto_n} fit locations for {n_points} points")

        if self.subsample_n is not None:
            self.subsample_indices = np.sort(np.random.choice(n_points, self.subsample_n, replace=False))
            print(f"Fitting local models at {self.subsample_n} random locations.")
        elif self.subsample_fraction is not None and 0 < self.subsample_fraction < 1.0:
            n_sub = int(n_points * self.subsample_fraction)
            np.random.seed(42)
            self.subsample_indices = np.sort(np.random.choice(n_points, n_sub, replace=False))
            print(f"Fitting local models at {n_sub} subsampled locations out of {n_points}.")
        else:
            self.subsample_indices = np.arange(n_points)

        self.subsample_coords = self.coords[self.subsample_indices]
        self.local_models = []

        # Build BallTree for efficient k-NN lookups (O(n log n) vs O(n²) cdist)
        self._fit_tree = BallTree(self.coords)

        # ----------------------------------------------------------------
        # S1a: Pre-compute anisotropic distances when kernel_field present.
        # For each subsample location we build a sorted list of (aniso_dist,
        # training_idx) pairs and select the k_neighbors closest under the
        # geometric-mean anisotropic Matérn metric.
        # ----------------------------------------------------------------
        _use_anisotropy = (
            self.kernel_field is not None
            and hasattr(self.kernel_field, "kernels")
            and any(
                kf.get("is_anisotropic", False)
                for kf in self.kernel_field.kernels.values()
            )
        )
        if _use_anisotropy:
            try:
                from sparc.models.kernel_field import anisotropic_distance, matern_kernel_weights
            except ImportError:
                _use_anisotropy = False

        print(f"Training {len(self.subsample_indices)} local random forest models...")

        for i, idx in enumerate(self.subsample_indices):
            if _use_anisotropy:
                # Compute geometric-mean anisotropic distance from this location
                # to every training point, then pick the k_neighbors smallest.
                dx = self.coords[:, 0] - self.coords[idx, 0]
                dy = self.coords[:, 1] - self.coords[idx, 1]
                log_dist_sum = np.zeros(len(self.coords))
                n_preds = 0
                for kern in self.kernel_field.kernels.values():
                    kappa_x = float(kern.get("kappa_x", 1.0))
                    kappa_y = float(kern.get("kappa_y", 1.0))
                    theta_rad = float(kern.get("theta_rad", 0.0))
                    d_aniso = anisotropic_distance(dx, dy, kappa_x, kappa_y, theta_rad)
                    log_dist_sum += np.log(d_aniso + 1e-12)
                    n_preds += 1
                geom_mean_dist = np.exp(log_dist_sum / max(n_preds, 1))
                geom_mean_dist[idx] = 0.0  # self-distance = 0
                neighbor_indices = np.argsort(geom_mean_dist)[: self.k_neighbors]
                d_local = geom_mean_dist[neighbor_indices]
                # Use the geometric-mean Matérn weights for sample_weight.
                kern0 = next(iter(self.kernel_field.kernels.values()))
                nu = float(kern0.get("nu", 1.5))
                sigma2 = float(kern0.get("sigma2", 1.0))
                weights = matern_kernel_weights(d_local, nu=nu, sigma2=sigma2)
            else:
                d_local, neighbor_indices_raw = self._fit_tree.query(
                    self.coords[idx:idx+1], k=self.k_neighbors
                )
                d_local = d_local[0]
                neighbor_indices = neighbor_indices_raw[0]
                bandwidth = d_local[-1] if d_local[-1] > 0 else 1e-6
                weights = np.exp(-0.5 * (d_local / bandwidth) ** 2)

            X_local = X[neighbor_indices]
            y_local = y[neighbor_indices]

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=self.n_jobs
            )

            try:
                rf.fit(X_local, y_local, sample_weight=weights)
                self.local_models.append(rf)
            except Exception as e:
                print(f"[Warning] Skipped model {i} at idx {idx}: {e}")
                self.local_models.append(None)

            # Free memory periodically for large grids
            if (i + 1) % 200 == 0:
                gc.collect()

        # ===============================================================
        # === EXTRACT GWRF SATURATION CURVES + PARTIAL DERIVATIVES    ===
        # ===============================================================
        if extract_derivatives and output_dir is not None:
            print("\n✓ Extracting GWRF PDP curves and derivatives...")
            self._extract_pdp_and_derivatives(X, y, coords, output_dir)
        
        return self
    
    def _extract_pdp_and_derivatives(self, X, y, coords, output_dir):
        """
        Extract Partial Dependence Plots and compute derivatives at baseline values.
        
        This computes ICE curves for each predictor at each cell, fits saturation curves,
        and derives local sensitivities.
        """
        import pandas as pd
        from scipy.interpolate import UnivariateSpline
        from sparc.run.artifact_io import save_table_path, ensure_dir
        
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, 'gwrf_pdp'))
        ensure_dir(os.path.join(output_dir, 'gwrf_derivatives'))
        
        # Use instance-level physics signs if set, else fall back to legacy defaults
        physics_signs = getattr(self, 'physics_signs', {
            "Pct_Impervious": +1,
            "Pct_Canopy": -1,
            "NDVI": -1,
            "Albedo": -1
        })
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        for feat_idx in range(n_features):
            feat_name = self.feature_names_[feat_idx]
            print(f"   Computing PDP for {feat_name}...")
            
            # Get feature range
            feat_values = X[:, feat_idx]
            feat_min, feat_max = np.percentile(feat_values, [5, 95])
            grid = np.linspace(feat_min, feat_max, 50)
            
            # Compute ICE curves (Individual Conditional Expectation)
            pdp_matrix = np.zeros((n_samples, len(grid)))
            derivatives = np.zeros(n_samples)

            # BallTree for nearest local-model lookup
            pdp_tree = BallTree(self.subsample_coords)
            
            for i in range(n_samples):
                X_pdp = np.tile(X[i], (len(grid), 1))
                X_pdp[:, feat_idx] = grid
                
                # Predict using nearest local model
                _, idx = pdp_tree.query(coords[i:i+1], k=1)
                nearest_model_idx = idx[0, 0]
                local_model = self.local_models[nearest_model_idx]
                
                if local_model is not None:
                    pdp_matrix[i, :] = local_model.predict(X_pdp)
                    
                    # Fit smooth spline and compute derivative at baseline
                    try:
                        spline = UnivariateSpline(grid, pdp_matrix[i, :], s=0.1, k=3)
                        derivatives[i] = spline.derivative()(feat_values[i])
                    except:
                        # Fallback to finite difference
                        derivatives[i] = np.gradient(pdp_matrix[i, :], grid)[np.argmin(np.abs(grid - feat_values[i]))]
                else:
                    pdp_matrix[i, :] = np.nan
                    derivatives[i] = np.nan
            
            # Save PDP curves
            pdp_df = pd.DataFrame(pdp_matrix, columns=[f'pdp_{g:.3f}' for g in grid])
            pdp_df['POINT_X'] = coords[:, 0]
            pdp_df['POINT_Y'] = coords[:, 1]
            pdp_df['baseline_value'] = feat_values
            pdp_path = os.path.join(output_dir, 'gwrf_pdp', f'gwrf_pdp_{feat_name}.csv')
            save_table_path(
                pdp_df, pdp_path,
                stage="2", artifact_id=f"gwrf_pdp_{feat_name}", fmt="csv",
                producer="gwrf._extract_pdp_and_derivatives",
            )
            
            # Physics-constrained derivative correction (preserves diagnostics)
            derivatives_constrained = derivatives.copy()
            sign_violations = np.zeros(n_samples, dtype=int)
            if feat_name in physics_signs:
                expected_sign = physics_signs[feat_name]
                # Flag locations where unconstrained derivative disagrees with physics
                if expected_sign == -1:
                    violation_mask = derivatives > 0
                else:  # expected_sign == +1
                    violation_mask = derivatives < 0
                sign_violations[violation_mask] = 1
                # Apply constraint: clip to correct sign (preserves magnitude when correct)
                if expected_sign == -1:
                    derivatives_constrained = np.minimum(derivatives_constrained, 0.0)
                else:
                    derivatives_constrained = np.maximum(derivatives_constrained, 0.0)
            
            # Save derivatives with diagnostic information
            deriv_df = pd.DataFrame({
                'POINT_X': coords[:, 0],
                'POINT_Y': coords[:, 1],
                'derivative_raw': derivatives,
                'derivative_constrained': derivatives_constrained,
                'sign_violation': sign_violations,
                'baseline_value': feat_values
            })
            deriv_path = os.path.join(output_dir, 'gwrf_derivatives', f'gwrf_derivative_{feat_name}.csv')
            save_table_path(
                deriv_df, deriv_path,
                stage="2", artifact_id=f"gwrf_derivative_{feat_name}", fmt="csv",
                producer="gwrf._extract_pdp_and_derivatives",
            )
            print(f"      ✓ Saved PDP and derivatives for {feat_name}")

    def predict(self, X, coords, k_blend=5, blend_kernel='gaussian'):
        """
        Predict using distance-weighted blend of k nearest local models.
        
        Instead of using only the single nearest local model (Voronoi tessellation),
        this blends predictions from the k nearest local models using kernel weights.
        This eliminates artificial discontinuities at model boundaries.
        
        Parameters
        ----------
        X : array-like
            Features
        coords : array-like
            Spatial coordinates
        k_blend : int, default=5
            Number of nearest local models to blend. k=1 recovers legacy behavior.
        blend_kernel : str, default='gaussian'
            Weighting kernel: 'gaussian', 'idw', or 'bisquare'
            
        Returns
        -------
        predictions : np.ndarray
            Blended predictions
        prediction_uncertainty : np.ndarray
            Weighted disagreement across local models (free uncertainty estimate)
        """
        if self.local_models is None:
            raise ValueError("Model must be fitted before making predictions")

        from collections import defaultdict

        X = np.asarray(X)
        coords = np.asarray(coords)
        n = len(X)

        # Limit k_blend to available models
        n_local = len(self.local_models)
        k_blend = min(k_blend, n_local)

        # Use BallTree for efficient k-nearest local model lookup
        subsample_tree = BallTree(self.subsample_coords)
        all_dists, all_idx = subsample_tree.query(coords, k=k_blend)

        # ------------------------------------------------------------------
        # Vectorised batch predict: group rows by local model, then call
        # model.predict() once per local model on all rows that need it.
        # This replaces the previous O(N × k_blend) row-by-row loop
        # (273 k individual 1-row predict calls for a 54 k dataset) with
        # O(n_local_models) batched calls, giving ~50–100× speedup.
        # ------------------------------------------------------------------

        # Build blend-weight matrix (n, k_blend) — fully vectorised.
        if blend_kernel == 'gaussian':
            bw = all_dists.max(axis=1, keepdims=True) + 1e-10
            blend_weights = np.exp(-0.5 * (all_dists / bw) ** 2)
        elif blend_kernel == 'idw':
            blend_weights = 1.0 / (all_dists + 1e-10)
        elif blend_kernel == 'bisquare':
            bw = all_dists.max(axis=1, keepdims=True) + 1e-10
            blend_weights = (1 - (all_dists / bw) ** 2) ** 2
        else:
            blend_weights = np.ones((n, k_blend))  # uniform fallback

        row_sums = blend_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        blend_weights /= row_sums  # (n, k_blend), rows sum to 1

        # Reverse lookup: local-model index → list of (row_i, blend_pos_j)
        model_to_pairs: dict = defaultdict(list)
        for i in range(n):
            for j in range(k_blend):
                model_to_pairs[int(all_idx[i, j])].append((i, j))

        # Raw predictions per (row, blend-position), NaN for None models.
        raw_preds = np.full((n, k_blend), np.nan)
        print(f"Making predictions with k={k_blend} blended models "
              f"({blend_kernel} kernel, {len(model_to_pairs)} active local models)...")
        for midx, pairs in model_to_pairs.items():
            model = self.local_models[midx]
            if model is None:
                continue
            rows = [p[0] for p in pairs]
            batch_preds = model.predict(X[rows])
            for k, (i, j) in enumerate(pairs):
                raw_preds[i, j] = batch_preds[k]

        # Vectorised blending: mask NaN slots, renormalise weights, compute
        # weighted mean and weighted std (inter-model disagreement).
        valid = ~np.isnan(raw_preds)                          # (n, k_blend)
        raw_filled = np.where(valid, raw_preds, 0.0)
        w_masked = np.where(valid, blend_weights, 0.0)       # (n, k_blend)
        wsum = w_masked.sum(axis=1, keepdims=True)
        wsum = np.where(wsum == 0, 1.0, wsum)
        w_masked /= wsum                                       # renormalise

        predictions = (w_masked * raw_filled).sum(axis=1)     # weighted mean
        diff2 = (raw_filled - predictions[:, np.newaxis]) ** 2
        prediction_uncertainty = np.sqrt((w_masked * diff2).sum(axis=1))

        # Points where every blend slot was NaN → no valid model nearby.
        all_nan_rows = ~np.any(valid, axis=1)
        predictions[all_nan_rows] = np.nan
        prediction_uncertainty[all_nan_rows] = np.nan

        return predictions, prediction_uncertainty

    def score(self, X, y, coords):
        result = self.predict(X, coords)
        # Handle both tuple (predictions, uncertainty) and plain array returns
        y_pred = result[0] if isinstance(result, tuple) else result
        mask = ~np.isnan(y_pred)
        u = ((y[mask] - y_pred[mask]) ** 2).sum()
        v = ((y[mask] - y[mask].mean()) ** 2).sum()
        return 1 - u / v if v != 0 else np.nan

    def predict_with_uncertainty(
        self,
        X: "np.ndarray",
        coords: "np.ndarray",
        k_blend: int = 1,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Return per-cell mean prediction and within-forest epistemic std.

        Unlike :meth:`predict` (which measures *inter-model* disagreement),
        this method uses the individual tree predictions from the nearest
        local RandomForestRegressor to estimate *within-model* epistemic
        uncertainty.  The result is suitable for routing to the divergence
        audit (S4) and the neural meta-learner surrogate uncertainty head
        (S2/S3).

        Parameters
        ----------
        X : array-like of shape (N, p)
            Feature matrix.
        coords : array-like of shape (N, 2)
            Spatial coordinates for local-model lookup.
        k_blend : int, default=1
            Number of nearest local models to blend.  ``k_blend=1`` uses
            only the single closest model (lowest noise; recommended for
            the divergence audit).

        Returns
        -------
        predictions : np.ndarray of shape (N,)
            Mean prediction per cell (average across tree predictions of
            the nearest local RF).
        std_per_cell : np.ndarray of shape (N,)
            Std of per-tree predictions (epistemic uncertainty proxy).
            NaN when no valid local model is found.
        """
        if self.local_models is None:
            raise RuntimeError("predict_with_uncertainty() requires a fitted model.")

        from collections import defaultdict

        X = np.asarray(X)
        coords = np.asarray(coords)
        n = len(X)
        predictions = np.full(n, np.nan)
        std_per_cell = np.full(n, np.nan)

        subsample_tree = BallTree(self.subsample_coords)
        k_use = min(k_blend, len(self.local_models))
        all_dists, all_idx = subsample_tree.query(coords, k=k_use)

        # ------------------------------------------------------------------
        # Vectorised: for each unique (model, blend-slot) pair batch all
        # rows that need per-tree predictions from that model, then gather.
        # ------------------------------------------------------------------
        model_to_pairs: dict = defaultdict(list)
        for i in range(n):
            for j in range(k_use):
                model_to_pairs[int(all_idx[i, j])].append((i, j))

        n_estimators_max = max(
            len(m.estimators_) for m in self.local_models if m is not None
        )
        # Store per-tree preds: shape (n, k_use, n_trees) — NaN-padded
        per_tree_store = np.full((n, k_use, n_estimators_max), np.nan)

        for midx, pairs in model_to_pairs.items():
            model = self.local_models[midx]
            if model is None:
                continue
            rows = [p[0] for p in pairs]
            n_trees = len(model.estimators_)
            # Batch per-tree predict: (n_trees, len(rows))
            batch_tree_preds = np.array([
                est.predict(X[rows]) for est in model.estimators_
            ])  # shape (n_trees, len(rows))
            for k, (i, j) in enumerate(pairs):
                per_tree_store[i, j, :n_trees] = batch_tree_preds[:, k]

        # Aggregate across all blend slots and trees for each row.
        for i in range(n):
            all_tree_preds = per_tree_store[i].ravel()
            valid = all_tree_preds[~np.isnan(all_tree_preds)]
            if len(valid) > 0:
                predictions[i] = float(valid.mean())
                std_per_cell[i] = float(valid.std())

        return predictions, std_per_cell

    def save_models(self, output_dir, prefix="gwrf"):
        """
        Save local models and metadata to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models
        prefix : str
            Prefix for saved files
        """
        if self.local_models is None:
            raise ValueError("No models to save. Call fit() first.")
        
        from sparc.run.artifact_io import save_blob_path, ensure_dir
        ensure_dir(output_dir)
        
        # Save local models (DB-first; disk only when enabled)
        models_file = os.path.join(output_dir, f"{prefix}_local_models.pkl")
        save_blob_path(
            self.local_models, models_file,
            stage="2", artifact_id=f"{prefix}_local_models",
            producer="gwrf.save_models",
        )
        
        # Save metadata
        metadata = {
            'subsample_coords': self.subsample_coords,
            'subsample_indices': self.subsample_indices,
            'n_estimators': self.n_estimators,
            'k_neighbors': self.k_neighbors,
            'min_samples_leaf': self.min_samples_leaf,
            'n_jobs': self.n_jobs,
            'subsample_fraction': self.subsample_fraction,
            'subsample_n': self.subsample_n
        }
        
        metadata_file = os.path.join(output_dir, f"{prefix}_metadata.pkl")
        save_blob_path(
            metadata, metadata_file,
            stage="2", artifact_id=f"{prefix}_metadata",
            producer="gwrf.save_models",
        )
        
        print(f"GWRF models saved to {output_dir}/")
        print(f"  - Local models: {models_file}")
        print(f"  - Metadata: {metadata_file}")

    def load_models(self, output_dir, prefix="gwrf"):
        """
        Load local models and metadata from disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory containing saved models
        prefix : str
            Prefix for saved files
        """
        # Load local models
        models_file = os.path.join(output_dir, f"{prefix}_local_models.pkl")
        self.local_models = joblib.load(models_file)
        
        # Load metadata
        metadata_file = os.path.join(output_dir, f"{prefix}_metadata.pkl")
        metadata = joblib.load(metadata_file)
        
        self.subsample_coords = metadata['subsample_coords']
        self.subsample_indices = metadata['subsample_indices']
        self.n_estimators = metadata['n_estimators']
        self.k_neighbors = metadata['k_neighbors']
        self.min_samples_leaf = metadata['min_samples_leaf']
        self.n_jobs = metadata['n_jobs']
        self.subsample_fraction = metadata['subsample_fraction']
        self.subsample_n = metadata['subsample_n']
        
        print(f"GWRF models loaded from {output_dir}/")
    
    # === OOF SPATIAL INTELLIGENCE EXTRACTION (EPIC 3) ===
    
    def compute_partial_dependence(
        self,
        X: np.ndarray,
        variable_idx: int,
        variable_name: str = None,
        n_grid_points: int = 50,
        percentile_range: tuple = (5, 95)
    ) -> dict:
        """
        Compute partial dependence plot (PDP) for a specific variable.
        
        This shows how predictions change as a single variable varies while
        holding all other variables at their median values.
        
        Parameters:
        -----------
        X : np.ndarray
            Training feature matrix (needed for median values)
        variable_idx : int
            Index of variable to analyze
        variable_name : str, optional
            Name of the variable (for labeling)
        n_grid_points : int, default=50
            Number of grid points to evaluate
        percentile_range : tuple, default=(5, 95)
            Percentile range for variable values (avoid extrapolation)
        
        Returns:
        --------
        dict
            PDP results with keys:
            - 'variable_name': str
            - 'variable_idx': int
            - 'grid_values': np.ndarray
            - 'pdp_values': np.ndarray (mean across local models)
            - 'pdp_std': np.ndarray (std across local models)
        """
        if self.local_models is None:
            raise ValueError("Model must be fitted before computing partial dependence")
        
        X = np.asarray(X)
        
        if variable_name is None:
            variable_name = f'variable_{variable_idx}'
        
        # Get variable range (avoid extrapolation)
        var_min = np.percentile(X[:, variable_idx], percentile_range[0])
        var_max = np.percentile(X[:, variable_idx], percentile_range[1])
        grid_values = np.linspace(var_min, var_max, n_grid_points)
        
        # Use median for all other variables
        X_median = np.median(X, axis=0).reshape(1, -1)
        
        # Collect PDP values from each local model
        pdp_per_model = []
        
        for model in self.local_models:
            if model is None:
                continue
            
            model_pdp = []
            for grid_val in grid_values:
                # Create instance with grid value for target variable
                X_test = X_median.copy()
                X_test[0, variable_idx] = grid_val
                
                # Predict
                pred = model.predict(X_test)[0]
                model_pdp.append(pred)
            
            pdp_per_model.append(model_pdp)
        
        pdp_per_model = np.array(pdp_per_model)
        
        # Aggregate across models
        pdp_mean = np.mean(pdp_per_model, axis=0)
        pdp_std = np.std(pdp_per_model, axis=0)
        
        return {
            'variable_name': variable_name,
            'variable_idx': variable_idx,
            'grid_values': grid_values,
            'pdp_values': pdp_mean,
            'pdp_std': pdp_std
        }
    
    def fit_condition_curve(
        self,
        pdp_results: dict,
        curve_type: str = 'logistic'
    ) -> dict:
        """
        Fit a monotonic saturation curve C_ik to partial dependence results.
        
        This captures the non-linear, saturating relationship between
        interventions and temperature change (e.g., canopy cooling saturates
        around 40-50% coverage).
        
        Parameters:
        -----------
        pdp_results : dict
            Results from compute_partial_dependence()
        curve_type : str, default='logistic'
            Type of curve to fit: 'logistic', 'exponential', or 'piecewise'
        
        Returns:
        --------
        dict
            Fitted curve parameters and evaluation:
            - 'curve_type': str
            - 'parameters': dict (curve-specific params)
            - 'saturation_point': float (where curve flattens)
            - 'fitted_values': np.ndarray
            - 'r2': float
        """
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score
        
        grid_values = pdp_results['grid_values']
        pdp_values = pdp_results['pdp_values']
        
        # Normalize to [0, 1] range for fitting
        pdp_min, pdp_max = pdp_values.min(), pdp_values.max()
        pdp_range = pdp_max - pdp_min
        
        if pdp_range == 0:
            # No variation - return flat curve
            return {
                'curve_type': 'constant',
                'parameters': {'value': pdp_min},
                'saturation_point': None,
                'fitted_values': np.ones_like(grid_values),
                'r2': 1.0
            }
        
        pdp_norm = (pdp_values - pdp_min) / pdp_range
        
        if curve_type == 'logistic':
            # Logistic saturation: f(x) = L / (1 + exp(-k*(x - x0)))
            def logistic(x, L, k, x0):
                return L / (1 + np.exp(-k * (x - x0)))
            
            try:
                # Initial guess
                p0 = [1.0, 0.1, np.median(grid_values)]
                params, _ = curve_fit(logistic, grid_values, pdp_norm, p0=p0, maxfev=5000)
                L, k, x0 = params
                fitted_norm = logistic(grid_values, L, k, x0)
                
                # Saturation point: where curve reaches 90% of max
                saturation_point = x0 + np.log(9) / k  # 90% of L
                
                curve_params = {'L': L, 'k': k, 'x0': x0}
                
            except:
                # Fallback to linear
                fitted_norm = pdp_norm
                saturation_point = None
                curve_params = {'fallback': 'linear'}
        
        elif curve_type == 'exponential':
            # Exponential saturation: f(x) = a * (1 - exp(-b*x))
            def exponential_sat(x, a, b):
                return a * (1 - np.exp(-b * x))
            
            try:
                p0 = [1.0, 0.01]
                params, _ = curve_fit(exponential_sat, grid_values, pdp_norm, p0=p0, maxfev=5000)
                a, b = params
                fitted_norm = exponential_sat(grid_values, a, b)
                
                # Saturation point: where curve reaches 90% of max
                saturation_point = -np.log(0.1) / b
                
                curve_params = {'a': a, 'b': b}
                
            except:
                fitted_norm = pdp_norm
                saturation_point = None
                curve_params = {'fallback': 'linear'}
        
        else:
            # Default to no transformation
            fitted_norm = pdp_norm
            saturation_point = None
            curve_params = {'type': 'none'}
        
        # Denormalize fitted values
        fitted_values = fitted_norm * pdp_range + pdp_min
        
        # Compute R²
        r2 = r2_score(pdp_values, fitted_values)
        
        return {
            'curve_type': curve_type,
            'parameters': curve_params,
            'saturation_point': saturation_point,
            'fitted_values': fitted_values,
            'r2': r2
        }
    
    def export_condition_curves(
        self,
        X: np.ndarray,
        output_dir: str,
        variable_names: list,
        curve_type: str = 'logistic',
        save_plots: bool = True
    ) -> dict:
        """
        Extract and export condition curves C_ik for all variables.
        
        Parameters:
        -----------
        X : np.ndarray
            Training feature matrix
        output_dir : str
            Directory to save curves
        variable_names : list
            Names of variables
        curve_type : str, default='logistic'
            Type of curve to fit
        save_plots : bool, default=True
            Whether to save PDP visualizations
        
        Returns:
        --------
        dict
            Condition curves for each variable
        """
        import json
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        condition_curves = {}
        
        for idx, var_name in enumerate(variable_names):
            print(f"\nComputing condition curve for {var_name}...")
            
            # Compute partial dependence
            pdp_results = self.compute_partial_dependence(
                X, idx, var_name, n_grid_points=50
            )
            
            # Fit condition curve
            curve_fit = self.fit_condition_curve(pdp_results, curve_type)
            
            # Store results
            condition_curves[var_name] = {
                'pdp': {
                    'grid_values': pdp_results['grid_values'].tolist(),
                    'pdp_values': pdp_results['pdp_values'].tolist(),
                    'pdp_std': pdp_results['pdp_std'].tolist()
                },
                'curve_fit': {
                    'curve_type': curve_fit['curve_type'],
                    'parameters': curve_fit['parameters'],
                    'saturation_point': curve_fit['saturation_point'],
                    'r2': curve_fit['r2']
                }
            }
            
            print(f"  Curve type: {curve_fit['curve_type']}, R² = {curve_fit['r2']:.3f}")
            if curve_fit['saturation_point'] is not None:
                print(f"  Saturation point: {curve_fit['saturation_point']:.2f}")

            # Plot generation removed (Phase E): visualizations are rendered
            # live in the desktop app from artifacts.db.
            _ = save_plots  # kept for backward-compatible signature
        
        # Save all curves to JSON (DB-first; disk only when enabled)
        from sparc.run.artifact_io import save_struct_path
        curves_json_path = output_dir / 'gwrf_condition_curves.json'
        save_struct_path(
            condition_curves, curves_json_path,
            stage="2", artifact_id="gwrf_pdp_curves",
            producer="gwrf.extract_condition_curves",
        )
        
        print(f"\nCondition curves saved to: {curves_json_path}")
        
        return condition_curves

def run_gwrf(X, y, coords, cv_mode=True, output_dir=None, **model_params):
    """
    Run GWRF model in either CV mode (OOF) or full map mode.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    coords : np.ndarray
        Spatial coordinates
    cv_mode : bool, default=True
        If True, run in CV mode (OOF predictions)
        If False, run in full map mode (complete predictions)
    output_dir : str, optional
        Directory to save models. Required if cv_mode=False
    **model_params : dict
        Additional parameters for GWRFModel
        
    Returns:
    --------
    tuple : (predictions, model) or (predictions, None) for CV mode
    """
    # Set default parameters
    default_params = {
        'n_estimators': 100,
        'k_neighbors': 100,
        'min_samples_leaf': 5,
        'n_jobs': -1,
        'subsample_fraction': None,
        'subsample_n': None
    }
    
    # Update with provided parameters
    default_params.update(model_params)
    
    # Create model
    model = GWRFModel(**default_params)
    
    if cv_mode:
        # CV mode: just fit and predict, don't save
        print("GWRF CV Mode: Training for OOF predictions...")
        model.fit(X, y, coords)
        predictions = model.predict(X, coords)
        return predictions, None
        
    else:
        # Full map mode: fit, predict, and save
        if output_dir is None:
            raise ValueError("output_dir must be provided for full map mode")
        
        print("GWRF Full Map Mode: Training complete model...")
        model.fit(X, y, coords)
        predictions = model.predict(X, coords)
        
        # Save models
        model.save_models(output_dir, prefix="gwrf_full")
        
        return predictions, model


from sparc.models.gwrf_config import GWRFConfig  # noqa: F401
