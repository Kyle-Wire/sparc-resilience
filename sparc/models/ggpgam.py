"""
Geographically Guided Penalized Generalized Additive Model (GGP-GAM) implementation.
This module provides a Python implementation of the GGPGAM model, which combines
spatial and non-spatial components using penalized splines.
"""

import numpy as np
import pandas as pd
from pygam import LinearGAM, s, te
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

class GGPGAM_SVC:
    def __init__(self, n_splines=5, n_spatial_bases=10, lam=0.6, clamp_features=False):
        self.n_splines = n_splines
        self.n_spatial_bases = n_spatial_bases
        self.lam = lam
        self.clamp_features = clamp_features
        self.feature_scaler = StandardScaler()
        self.coords_scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self._feat_min_ = None
        self._feat_max_ = None
        self.X_train_ = None  # Store for derivative computation
        self.coords_train_ = None

    def fit(self, X, y, coords, extract_derivatives=False, output_dir=None,
            elevation=None, elevation_scale=None):
        """
        Fit GGPGAM model
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        coords : array-like
            Spatial coordinates (N, 2)
        extract_derivatives : bool, default=False
            Whether to extract partial derivative surfaces after fitting
        output_dir : str, optional
            Directory to save derivative outputs
        elevation : array-like, optional
            (N,) elevation values in metres.  When provided the spatial
            basis is extended to a 3D tensor-product spline over (x, y, z)
            so that terrain shape warps the smooth spatial surface.
        elevation_scale : float, optional
            Metres-to-projection-unit scale.  Auto-detected from data when
            None: ``max(x_range, y_range) / elevation_range``.
        """
        print("Fitting GGPGAM-SVC...")

        X = np.asarray(X)
        y = np.asarray(y).ravel()
        coords = np.asarray(coords)

        # C2: lift (x, y) to (x, y, z) when elevation is provided
        if elevation is not None:
            elevation = np.asarray(elevation, dtype=float).ravel()
            if elevation_scale is None:
                xy_range = max(
                    float(coords[:, 0].max() - coords[:, 0].min()),
                    float(coords[:, 1].max() - coords[:, 1].min()),
                )
                z_range = float(elevation.max() - elevation.min())
                elevation_scale = float(xy_range / z_range) if z_range > 1e-9 else 1.0
            self._elevation_scale_ = elevation_scale
            coords = np.column_stack([coords, elevation * elevation_scale])
        else:
            self._elevation_scale_ = None

        self._n_coord_dims_ = coords.shape[1]  # 2 or 3
        
        # Store training data for derivative computation
        self.X_train_ = X
        self.coords_train_ = coords

        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'X{i}' for i in range(X.shape[1])]

        # Scale features and coordinates separately
        X_scaled = self.feature_scaler.fit_transform(X)
        coords_scaled = self.coords_scaler.fit_transform(coords)

        n_coord_dims = self._n_coord_dims_
        if n_coord_dims == 3:
            # C2: 3D terrain-aware spatial basis
            terms = te(0, 1, 2, n_splines=[self.n_spatial_bases,
                                            self.n_spatial_bases,
                                            self.n_spatial_bases])
            for i in range(X.shape[1]):
                terms = terms + s(i + 3, n_splines=self.n_splines)
            for i in range(X.shape[1]):
                terms = terms + te(0, 1, i + 3,
                                   n_splines=[self.n_spatial_bases,
                                              self.n_spatial_bases,
                                              self.n_splines])
        else:
            # Original 2D spatial basis
            terms = s(0, n_splines=self.n_spatial_bases) + s(1, n_splines=self.n_spatial_bases)
            for i in range(X.shape[1]):
                terms = terms + s(i + 2, n_splines=self.n_splines)
            for i in range(X.shape[1]):
                terms = terms + te(0, i + 2, n_splines=[self.n_spatial_bases, self.n_splines])
                terms = terms + te(1, i + 2, n_splines=[self.n_spatial_bases, self.n_splines])

        # Initialize GAM with provided lambda
        self.model = LinearGAM(terms, lam=self.lam)

        # Combine scaled coords and scaled features
        X_combined = np.column_stack([coords_scaled, X_scaled])

        # Fit the model with progress bar
        with tqdm(total=1, desc="Training GGPGAM-SVC") as pbar:
            self.model.fit(X_combined, y)
            pbar.update(1)
        
        # ===============================================================
        # === EXTRACT GGPGAM PARTIAL DERIVATIVE SURFACES              ===
        # ===============================================================
        if extract_derivatives and output_dir is not None:
            print("\n✓ Extracting GGPGAM partial derivatives...")
            self._extract_partial_derivatives(X, coords, output_dir)
    
    def _extract_partial_derivatives(self, X, coords, output_dir):
        """
        Extract partial derivative surfaces from the fitted GAM model.
        
        Computes partial derivatives using finite differences on the fitted splines.
        """
        import pandas as pd
        import os
        from sparc.run.artifact_io import save_table_path, ensure_dir
        
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, 'ggpgam_derivatives'))
        
        # Use instance-level physics signs if set, else load from config
        if hasattr(self, 'physics_signs') and self.physics_signs:
            physics_signs = self.physics_signs
        else:
            try:
                from sparc.config.config import load_monotone_constraints
                physics_signs = load_monotone_constraints({})
            except Exception:
                physics_signs = {
                    "Pct_Impervious": +1,
                    "Pct_Canopy": -1,
                    "NDVI": -1,
                    "Albedo": -1
                }
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Scale data for model input
        X_scaled = self.feature_scaler.transform(X)
        coords_scaled = self.coords_scaler.transform(coords)
        X_combined = np.column_stack([coords_scaled, X_scaled])
        
        for feat_idx in range(n_features):
            feat_name = self.feature_names[feat_idx]
            print(f"   Computing partial derivatives for {feat_name}...")
            
            derivatives = np.zeros(n_samples)
            
            # Compute numerical partial derivative
            epsilon = 0.01  # Small perturbation in scaled space
            
            for i in range(n_samples):
                X_base = X_combined[i:i+1].copy()
                X_plus = X_base.copy()
                X_plus[0, feat_idx + 2] += epsilon  # +2 offset for coords
                
                pred_base = self.model.predict(X_base)[0]
                pred_plus = self.model.predict(X_plus)[0]
                
                # Derivative in original space (account for feature scaling)
                feature_std = self.feature_scaler.scale_[feat_idx]
                derivatives[i] = (pred_plus - pred_base) / (epsilon * feature_std)
            
            # Physics-constrained derivative correction (preserves diagnostics)
            derivatives_constrained = derivatives.copy()
            sign_violations = np.zeros(n_samples, dtype=int)
            if feat_name in physics_signs:
                expected_sign = physics_signs[feat_name]
                if expected_sign == -1:
                    violation_mask = derivatives > 0
                else:
                    violation_mask = derivatives < 0
                sign_violations[violation_mask] = 1
                # Clip to correct sign rather than abs() * sign
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
                'baseline_value': X[:, feat_idx]
            })
            deriv_path = os.path.join(output_dir, 'ggpgam_derivatives', f'ggpgam_derivative_{feat_name}.csv')
            save_table_path(
                deriv_df, deriv_path,
                stage="2", artifact_id=f"ggpgam_derivative_{feat_name}", fmt="csv",
                producer="ggpgam._extract_partial_derivatives",
            )
            print(f"      ✓ Saved derivatives for {feat_name}")

    def predict(self, X, coords, elevation=None):
        X = np.asarray(X)
        coords = np.asarray(coords)
        X_scaled = self.feature_scaler.transform(X)
        if (elevation is not None
                and getattr(self, '_n_coord_dims_', 2) == 3
                and self._elevation_scale_ is not None):
            elevation = np.asarray(elevation, dtype=float).ravel()
            coords = np.column_stack([coords, elevation * self._elevation_scale_])
        coords_scaled = self.coords_scaler.transform(coords)
        X_combined = np.column_stack([coords_scaled, X_scaled])
        return self.model.predict(X_combined)
    
    # === OOF SPATIAL INTELLIGENCE EXTRACTION (EPIC 3) ===
    
    def predict_with_uncertainty(self, X, coords, n_bootstrap: int = 100):
        """
        Predict with uncertainty estimates using bootstrap approximation.
        
        Since PyGAM doesn't provide built-in prediction intervals,
        we use the standard errors from the GAM prediction intervals
        as uncertainty estimates.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        coords : np.ndarray
            Coordinate matrix
        n_bootstrap : int, default=100
            Number of bootstrap samples for uncertainty estimation
            (currently uses GAM's built-in intervals instead)
        
        Returns:
        --------
        dict
            Predictions with uncertainty:
            - 'predictions': np.ndarray - Mean predictions
            - 'std_errors': np.ndarray - Standard errors
            - 'lower_95': np.ndarray - Lower 95% CI
            - 'upper_95': np.ndarray - Upper 95% CI
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.feature_scaler.transform(X)
        coords_scaled = self.coords_scaler.transform(coords)
        X_combined = np.column_stack([coords_scaled, X_scaled])
        
        # Get predictions with confidence intervals
        predictions = self.model.predict(X_combined)
        
        # Get prediction intervals (95% CI)
        try:
            intervals = self.model.prediction_intervals(X_combined, width=0.95)
            lower_95 = intervals[:, 0]
            upper_95 = intervals[:, 1]
            
            # Estimate standard error from CI width
            # 95% CI ≈ ±1.96 * SE, so SE ≈ (upper - lower) / (2 * 1.96)
            std_errors = (upper_95 - lower_95) / (2 * 1.96)
            
        except Exception as e:
            print(f"Warning: Could not compute prediction intervals: {e}")
            print("Using fallback uncertainty estimation...")
            
            # Fallback: use residual standard error
            # This is less accurate but better than nothing
            std_errors = np.ones(len(predictions)) * np.std(predictions) * 0.1
            lower_95 = predictions - 1.96 * std_errors
            upper_95 = predictions + 1.96 * std_errors
        
        return {
            'predictions': predictions,
            'std_errors': std_errors,
            'lower_95': lower_95,
            'upper_95': upper_95
        }
    
    def compute_uncertainty_weights(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        weight_range: tuple = (0.5, 1.0),
        uncertainty_threshold: float = None
    ) -> dict:
        """
        Convert predictive uncertainty to confidence weights W_unc.
        
        Lower uncertainty → higher weight (more confident)
        Higher uncertainty → lower weight (less confident)
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        coords : np.ndarray
            Coordinate matrix
        weight_range : tuple, default=(0.5, 1.0)
            Range for normalized weights [min_weight, max_weight]
        uncertainty_threshold : float, optional
            Uncertainty threshold for flagging "needs study" cells.
            If None, uses 75th percentile of uncertainties.
        
        Returns:
        --------
        dict
            Uncertainty weights and diagnostics:
            - 'weights': np.ndarray - W_unc values [0.5, 1.0]
            - 'std_errors': np.ndarray - Raw standard errors
            - 'low_confidence_mask': np.ndarray - Boolean mask for low confidence cells
            - 'pct_low_confidence': float - Percentage flagged as low confidence
        """
        # Get predictions with uncertainty
        results = self.predict_with_uncertainty(X, coords)
        std_errors = results['std_errors']
        
        # Inverse scaling: low SE → high weight
        # Normalize SE to [0, 1]
        se_min, se_max = np.percentile(std_errors, [5, 95])
        se_norm = (std_errors - se_min) / (se_max - se_min + 1e-10)
        se_norm = np.clip(se_norm, 0, 1)
        
        # Invert: high SE → low weight
        weight_norm = 1 - se_norm
        
        # Scale to desired range
        min_weight, max_weight = weight_range
        weights = min_weight + weight_norm * (max_weight - min_weight)
        
        # Flag low confidence cells
        if uncertainty_threshold is None:
            # Use 75th percentile of SE as threshold
            uncertainty_threshold = np.percentile(std_errors, 75)
        
        low_confidence_mask = std_errors > uncertainty_threshold
        pct_low_confidence = np.mean(low_confidence_mask) * 100
        
        return {
            'weights': weights,
            'std_errors': std_errors,
            'low_confidence_mask': low_confidence_mask,
            'pct_low_confidence': pct_low_confidence,
            'uncertainty_threshold': uncertainty_threshold
        }
    
    def export_uncertainty_maps(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        output_dir: str,
        weight_range: tuple = (0.5, 1.0),
        save_formats: list = ['csv', 'gpkg']
    ) -> dict:
        """
        Export uncertainty weight maps W_unc to files.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        coords : np.ndarray
            Coordinate matrix
        output_dir : str
            Directory to save maps
        weight_range : tuple, default=(0.5, 1.0)
            Weight range
        save_formats : list, default=['csv', 'gpkg']
            Formats to save
        
        Returns:
        --------
        dict
            Paths to saved files
        """
        import pandas as pd
        import geopandas as gpd
        import json
        from pathlib import Path
        
        output_dir = Path(output_dir)
        from sparc.run.artifact_io import (
            save_table_path, save_geo_path, save_struct_path, ensure_dir,
        )
        ensure_dir(output_dir)
        
        # Compute uncertainty weights
        unc_results = self.compute_uncertainty_weights(X, coords, weight_range)
        
        # Get predictions for context
        pred_results = self.predict_with_uncertainty(X, coords)
        
        # Create DataFrame
        unc_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'W_unc': unc_results['weights'],
            'std_error': unc_results['std_errors'],
            'prediction': pred_results['predictions'],
            'lower_95': pred_results['lower_95'],
            'upper_95': pred_results['upper_95'],
            'low_confidence': unc_results['low_confidence_mask'].astype(int)
        })
        
        saved_files = {}
        
        # Save as CSV
        if 'csv' in save_formats:
            csv_path = output_dir / 'ggpgam_uncertainty_maps.csv'
            save_table_path(
                unc_df, csv_path,
                stage="2", artifact_id="ggpgam_uncertainty_maps", fmt="csv",
                producer="ggpgam.export_uncertainty_maps",
            )
            saved_files['csv'] = str(csv_path)
            print(f"Saved uncertainty CSV: {csv_path}")
        
        # Save as GeoPackage
        if 'gpkg' in save_formats:
            gdf = gpd.GeoDataFrame(
                unc_df,
                geometry=gpd.points_from_xy(unc_df.x, unc_df.y),
                crs=getattr(self, '_output_crs', 'EPSG:26919')
            )
            gpkg_path = output_dir / 'ggpgam_uncertainty_maps.gpkg'
            save_geo_path(
                gdf, gpkg_path,
                stage="2", artifact_id="ggpgam_uncertainty_maps_gpkg",
                driver='GPKG', producer="ggpgam.export_uncertainty_maps",
            )
            saved_files['gpkg'] = str(gpkg_path)
            print(f"Saved uncertainty GeoPackage: {gpkg_path}")
        
        # Save summary statistics
        summary = {
            'weight_range': weight_range,
            'uncertainty_threshold': float(unc_results['uncertainty_threshold']),
            'pct_low_confidence': float(unc_results['pct_low_confidence']),
            'statistics': {
                'weights': {
                    'mean': float(np.mean(unc_results['weights'])),
                    'std': float(np.std(unc_results['weights'])),
                    'min': float(np.min(unc_results['weights'])),
                    'max': float(np.max(unc_results['weights']))
                },
                'std_errors': {
                    'mean': float(np.mean(unc_results['std_errors'])),
                    'std': float(np.std(unc_results['std_errors'])),
                    'min': float(np.min(unc_results['std_errors'])),
                    'max': float(np.max(unc_results['std_errors']))
                }
            }
        }
        
        summary_path = output_dir / 'ggpgam_uncertainty_summary.json'
        save_struct_path(
            summary, summary_path,
            stage="2", artifact_id="ggpgam_uncertainty_summary",
            producer="ggpgam.export_uncertainty_maps",
        )
        saved_files['summary'] = str(summary_path)
        print(f"Saved uncertainty summary: {summary_path}")
        
        print(f"\nUncertainty Weight Summary:")
        print(f"  Weight range: [{weight_range[0]:.2f}, {weight_range[1]:.2f}]")
        print(f"  Low confidence cells: {unc_results['pct_low_confidence']:.1f}%")
        print(f"  Mean weight: {np.mean(unc_results['weights']):.3f}")
        
        return saved_files

    def feature_importance(self):
        if self.model is None:
            raise ValueError("Model must be fitted first.")

        coef_indices = self.model.terms.get_coef_indices()
        coefs = self.model.coef_

        importance = []

        # Spatial effects (X and Y coordinates)
        for i, coord in enumerate(['X', 'Y']):
            spatial_coefs = coefs[coef_indices[i]]
            importance.append({
                'feature': f'spatial_{coord}',
                'mean_abs_coef': np.mean(np.abs(spatial_coefs)),
                'std_coef': np.std(spatial_coefs)
            })

        # Individual feature effects
        for i, feature in enumerate(self.feature_names):
            feat_coefs = coefs[coef_indices[i + 2]]
            importance.append({
                'feature': feature,
                'mean_abs_coef': np.mean(np.abs(feat_coefs)),
                'std_coef': np.std(feat_coefs)
            })

        # Spatial interactions
        offset = 2 + len(self.feature_names)
        for i, feature in enumerate(self.feature_names):
            for coord in ['X', 'Y']:
                svc_coefs = coefs[coef_indices[offset + 2*i + (0 if coord == 'X' else 1)]]
                importance.append({
                    'feature': f'spatial_{coord}_{feature}',
                    'mean_abs_coef': np.mean(np.abs(svc_coefs)),
                    'std_coef': np.std(svc_coefs)
                })

        return pd.DataFrame(importance)

    def plot_spatial_effect(self, grid_x, grid_y):
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        grid_points_scaled = self.coords_scaler.transform(grid_points)
        dummy_features = np.zeros((grid_points.shape[0], len(self.feature_names)))
        combined = np.column_stack([grid_points_scaled, dummy_features])

        preds = self.model.predict(combined).reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, preds, cmap='viridis')
        plt.colorbar(label='Spatial Effect')
        plt.title('Main Spatial Effect (GGPGAM-SVC)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    def score(self, X, y, coords):
        try:
            y_pred = self.predict(X, coords)
        except Exception as e:
            raise RuntimeError(f"GGPGAM_SVC prediction failed: {e}")
        y = np.asarray(y).ravel()
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v if v != 0 else 0.0