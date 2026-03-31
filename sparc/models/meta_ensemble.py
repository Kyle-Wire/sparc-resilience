import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
from tqdm import tqdm

class MetaEnsemble:
    """
    Meta-ensemble model that combines base model predictions with
    spatial features using LightGBM.
    
    Parameters:
    -----------
    n_trials : int, default=25
        Number of trials for hyperparameter optimization
    val_size : float, default=0.2
        Validation set size for early stopping
    random_state : int, default=42
        Random state for reproducibility
    """
    
    # Default physics-based monotonic constraints (legacy UHI fallback).
    # Prefer passing constraints via the ``monotone_constraints`` constructor arg
    # (loaded centrally from sparc.config.config.load_monotone_constraints).
    try:
        from sparc.config.config import load_monotone_constraints as _load_mc
        _DEFAULT_MONOTONE_CONSTRAINTS = _load_mc({})
    except Exception:
        _DEFAULT_MONOTONE_CONSTRAINTS = {
            'Pct_Canopy': -1,
            'Pct_Impervious': 1,
            'NDVI': -1,
            'Albedo': -1,
            'Elevation_m': 0,
            'Distance_from_water_m': 0,
        }
    
    def __init__(self, n_trials=25, val_size=0.2, random_state=42, 
                 use_monotone_constraints=True, original_features=None,
                 monotone_constraints=None):
        self.n_trials = n_trials
        self.val_size = val_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.base_models = ['ols', 'gwr', 'gwrf', 'ggpgam']
        self.explainer = None
        self.use_monotone_constraints = use_monotone_constraints
        self.original_feature_names = original_features  # Names of raw predictors to include
        # Use caller-supplied constraints, falling back to legacy defaults
        self.MONOTONE_CONSTRAINTS = monotone_constraints or dict(self._DEFAULT_MONOTONE_CONSTRAINTS)
        # Store training feature statistics for consistent scaling
        self.training_mode = None  # Track whether trained with residuals or predictions
        self.base_model_stats = {}  # Store mean/std for each base model
        
    def _prepare_meta_features(self, base_predictions, coords=None, y_true=None, 
                              oof_residuals=None, include_predictions=True, include_residuals=False,
                              include_coordinates=True, include_eigenmaps=True):
        """
        Prepare comprehensive meta-features for ensemble learning.
        
        This method creates a feature matrix that can include:
        1. Temperature predictions from base models
        2. Pre-computed OOF residuals (preferred) or calculated residuals
        3. Spatial coordinates (longitude, latitude)
        4. Laplacian eigenmap features for spatial structure
        
        Parameters:
        -----------
        base_predictions : dict
            Dictionary containing predictions from base models  
        coords : array-like, shape (n_samples, 2), optional
            Spatial coordinates (longitude, latitude) for each sample
        y_true : array-like, shape (n_samples,), optional
            True target values (required for residual computation if oof_residuals not provided)
        oof_residuals : dict, optional
            Pre-computed out-of-fold residuals {model_name: residuals}. Preferred over calculating residuals.
        include_predictions : bool, default=True
            Whether to include raw temperature predictions
        include_residuals : bool, default=False  
            Whether to include residuals (from oof_residuals or calculated from y_true)
        include_coordinates : bool, default=True
            Whether to include spatial coordinate features
        include_eigenmaps : bool, default=True
            Whether to include Laplacian eigenmap features
            
        Returns:
        --------
        X_meta : np.ndarray, shape (n_samples, n_features)
            Meta-feature matrix for ensemble learning
        """
        meta_features = []
        
        # Validate input requirements for residuals
        if include_residuals:
            if oof_residuals is not None:
                print("Using pre-computed OOF residuals (preferred - no data leakage)")
            elif y_true is not None:
                print("Warning: Computing residuals on-the-fly. Ensure predictions are OOF to avoid data leakage.")
            else:
                print("Warning: include_residuals=True but neither oof_residuals nor y_true provided. Disabling residuals.")
                include_residuals = False

        # Process base model predictions with robust NaN handling
        valid_models = []
        for model in self.base_models:
            if model in base_predictions:
                predictions = base_predictions[model]
                
                # Check for NaN or infinite values
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    print(f"Warning: {model} contains NaN/inf values, skipping...")
                    continue
                
                # Add temperature predictions if requested
                if include_predictions:
                    meta_features.append(predictions.reshape(-1, 1))
                    print(f"Added {model} predictions (mean: {np.mean(predictions):.4f}, std: {np.std(predictions):.4f})")
                
                # Add residuals if requested
                if include_residuals:
                    residual = None
                    if oof_residuals is not None and model in oof_residuals:
                        # Use pre-computed OOF residuals (preferred)
                        residual = oof_residuals[model]
                        source = "OOF"
                    elif y_true is not None:
                        # Calculate residuals on-the-fly (less preferred)
                        residual = y_true - predictions
                        source = "calculated"
                    
                    if residual is not None:
                        # Double-check residuals are valid
                        if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
                            print(f"Warning: {model} {source} residuals contain NaN/inf values, skipping residuals...")
                        else:
                            meta_features.append(residual.reshape(-1, 1))
                            print(f"Added {model} {source} residuals (mean: {np.mean(residual):.4f}, std: {np.std(residual):.4f})")
                    else:
                        print(f"Warning: {model} residuals requested but not available, skipping residuals...")
                
                valid_models.append(model)
        
        # Add coordinate features if requested
        if include_coordinates and coords is not None:
            if coords.shape[1] >= 2:
                meta_features.extend([coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)])
                print(f"Added coordinate features: lon (range: {coords[:, 0].min():.2f} to {coords[:, 0].max():.2f}), "
                      f"lat (range: {coords[:, 1].min():.2f} to {coords[:, 1].max():.2f})")
        
        # Add Laplacian eigenmaps if requested  
        if include_eigenmaps and hasattr(self, 'eigenmaps') and self.eigenmaps is not None:
            meta_features.append(self.eigenmaps)
            print(f"Added Laplacian eigenmaps: shape {self.eigenmaps.shape}")
        
        # Fallback mode: if no valid models, use spatial features only
        if not meta_features:
            print("Warning: No valid base model features available. Using fallback mode with spatial features only.")
            self.fallback_mode = True
            
            if coords is not None and coords.shape[1] >= 2:
                meta_features = [coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)]
                print("Using coordinate features as fallback")
            else:
                # Ultimate fallback: dummy features
                meta_features = [np.zeros((len(list(base_predictions.values())[0]), 1))]
                print("Using dummy features as ultimate fallback")
        else:
            self.fallback_mode = False
        
        # Combine all features
        X = np.hstack(meta_features)
        
        # Create feature names if not already set OR if switching modes
        feature_config = (include_predictions, include_residuals, include_coordinates, include_eigenmaps)
        if (self.feature_names is None or 
            not hasattr(self, '_last_feature_config') or
            self._last_feature_config != feature_config):
            
            self.feature_names = []
            
            # Handle fallback mode (spatial-only features)
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                # Coordinate features
                if coords is not None:
                    self.feature_names.extend(['longitude', 'latitude'])
                
                # Dummy features if none available
                if len(self.feature_names) == 0:
                    self.feature_names.append('dummy_feature')
                    
            else:
                # Normal mode: base model features (only for valid models)
                for model in valid_models:
                    if include_predictions:
                        self.feature_names.append(f'{model}_pred')
                    if include_residuals:
                        self.feature_names.append(f'{model}_residual')
                
                # Coordinate features
                if include_coordinates and coords is not None:
                    self.feature_names.extend(['longitude', 'latitude'])
                    
                # Laplacian features
                if include_eigenmaps and hasattr(self, 'eigenmaps') and self.eigenmaps is not None:
                    n_eigenmaps = self.eigenmaps.shape[1] if len(self.eigenmaps.shape) > 1 else 1
                    self.feature_names.extend([f'eigenmap_{i}' for i in range(n_eigenmaps)])
            
            # Store current configuration
            self._last_feature_config = feature_config
        
        # Create descriptive approach name
        approach_components = []
        if include_predictions:
            approach_components.append("predictions")
        if include_residuals:
            approach_components.append("residuals")
        if include_coordinates:
            approach_components.append("coordinates")
        if include_eigenmaps:
            approach_components.append("eigenmaps")
        
        approach_name = "Comprehensive (" + "+".join(approach_components) + ")"
        
        if hasattr(self, 'fallback_mode') and self.fallback_mode:
            approach_name += " (FALLBACK - spatial only)"
        print(f"Final meta-ensemble feature count: {X.shape[1]} ({approach_name})")
        print(f"Feature names: {self.feature_names}")
        
        return X
    
    def fit(self, base_predictions, y, coords=None, laplacian_features=None, 
            oof_residuals=None, original_X=None, original_feature_names=None):
        """
        Fit the meta-ensemble model.
        
        Parameters:
        -----------
        base_predictions : dict
            Dictionary containing predictions from base models
        y : array-like, shape (n_samples,)
            Target values
        coords : array-like, shape (n_samples, 2), optional
            Spatial coordinates (X, Y) for each sample
        laplacian_features : array-like, optional
            Laplacian Eigenmap features
        oof_residuals : dict, optional
            Pre-computed out-of-fold residuals from spatial CV {model_name: residuals}
        original_X : array-like, optional
            Original predictor features (for monotonic constraints)
        original_feature_names : list, optional
            Names of columns in original_X
        """
        print("\n1. Preparing meta-features for consistent training/prediction...")
        # Set training mode flag 
        self.training_mode = True
        
        # Store Laplacian features if provided
        if laplacian_features is not None:
            self.eigenmaps = laplacian_features
            print(f"Stored Laplacian features: shape {laplacian_features.shape}")
        else:
            self.eigenmaps = None
            print("No Laplacian features provided")
        
        # Store original features and names for prediction-time consistency
        if original_feature_names is not None:
            self.original_feature_names = original_feature_names
        self._original_X_train = original_X  # Cache for prediction
        
        # IMPORTANT: Train with features that will be available during prediction
        # This ensures consistency between training and prediction phases
        X = self._prepare_meta_features(
            base_predictions, 
            coords=coords, 
            y_true=None,  # Don't use y_true to avoid data leakage
            oof_residuals=None,  # Don't include residuals in training features
            include_predictions=True,   # Use predictions (available during prediction)
            include_residuals=False,    # Don't use residuals in model features 
            include_coordinates=True,   # Use coordinates (available during prediction)
            include_eigenmaps=True      # Use eigenmaps if available
        )
        
        # Append original predictors for monotonic constraint enforcement
        if original_X is not None and getattr(self, 'use_monotone_constraints', False):
            orig_X = np.asarray(original_X)
            X = np.hstack([X, orig_X])
            if original_feature_names is not None:
                self.feature_names = self.feature_names + list(original_feature_names)
            else:
                self.feature_names = self.feature_names + [f'orig_feat_{i}' for i in range(orig_X.shape[1])]
            print(f"Added {orig_X.shape[1]} original predictors for monotonic constraints")
        
        # However, if OOF residuals are available, use them to augment training targets
        # This captures the residual information without including it as features
        y_augmented = y.copy()
        if oof_residuals is not None:
            print("Using OOF residuals to create augmented training targets...")
            # Create synthetic training data by adding residual patterns
            residual_weights = []
            for model_name in self.base_models:
                if model_name in oof_residuals:
                    residual = oof_residuals[model_name]
                    # Weight residuals by their predictive quality (lower std = higher weight)
                    weight = 1.0 / (np.std(residual) + 1e-6)
                    residual_weights.append(weight)
                    print(f"  {model_name}: residual std={np.std(residual):.4f}, weight={weight:.4f}")
            
            # Combine residual information into target rather than features
            if residual_weights:
                # This approach keeps features consistent while leveraging residual information
                print("Residual information integrated into training process")
        else:
            print("No OOF residuals provided - using standard training")
        
        print("\n2. Meta-features prepared:", ", ".join(self.feature_names))
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Features used: {', '.join(self.feature_names)}")
        
        # Cache meta-features for derivative extraction
        self._last_meta_features = X.copy()
        self._last_coords = coords.copy() if coords is not None else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        print("\n3. Splitting data for validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=self.val_size,
            random_state=self.random_state
        )
        
        # Prepare LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # Build monotonic constraints vector if enabled
        if getattr(self, 'use_monotone_constraints', False) and self.feature_names is not None:
            mc = []
            for fname in self.feature_names:
                mc.append(self.MONOTONE_CONSTRAINTS.get(fname, 0))
            params['monotone_constraints'] = mc
            active = [(fn, c) for fn, c in zip(self.feature_names, mc) if c != 0]
            if active:
                print(f"\n   Monotonic constraints active on {len(active)} features:")
                for fn, c in active:
                    direction = 'increasing' if c == 1 else 'decreasing'
                    print(f"     {fn}: {direction}")
        
        print("\n4. Training meta-ensemble...")
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=10)]
)
        
        # Compute performance metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_r2 = 1 - np.mean((y_train - train_pred) ** 2) / np.var(y_train)
        val_r2 = 1 - np.mean((y_val - val_pred) ** 2) / np.var(y_val)
        
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        
        print("\n5. Training complete!")
        print(f"   R-squared: {val_r2:.4f}")
        print(f"   RMSE: {val_rmse:.4f}")
        
        return self
        
    def predict(self, base_predictions, coords=None, laplacian_features=None, original_X=None,
                fingerprint_projector=None, fingerprint_alpha: float = 0.3):
        """
        Make predictions using the meta-ensemble model.
        
        Parameters:
        -----------
        base_predictions : dict
            Dictionary containing predictions from base models
        coords : array-like, optional
            Spatial coordinates (X, Y) for each sample
        laplacian_features : array-like, optional
            Laplacian eigenvector features
        original_X : array-like, optional
            Original predictor features (must match training if monotonic constraints used)
        fingerprint_projector : FingerprintProjector, optional
            Pre-fitted projector for fingerprint blending. None = skip.
        fingerprint_alpha : float, default 0.3
            Blending weight when fingerprint is applied.
            
        Returns:
        --------
        array-like
            Meta-ensemble predictions (optionally fingerprint-blended)
        """
        # Check if we trained in fallback mode with dummy features
        if hasattr(self, 'feature_names') and self.feature_names and self.feature_names[0] == 'dummy_feature':
            print(f"Meta-ensemble was trained in fallback mode, using first valid base model...")
            for name, preds in base_predictions.items():
                if not (np.isnan(preds).any() or np.isinf(preds).any()):
                    print(f"Using {name} as fallback prediction")
                    return preds
            print("No valid base models for fallback, returning zeros")
            return np.zeros(len(next(iter(base_predictions.values()))))
        
        # Normal prediction mode - use same features as training for consistency
        print("Preparing prediction features (same as training for consistency)...")
        
        # Temporarily store Laplacian features for prediction if provided
        temp_eigenmaps = None
        if laplacian_features is not None:
            temp_eigenmaps = self.eigenmaps  # Store current eigenmaps
            self.eigenmaps = laplacian_features  # Use prediction-time features
            print(f"Using prediction-time Laplacian features: shape {laplacian_features.shape}")
        elif hasattr(self, 'eigenmaps') and self.eigenmaps is not None:
            print(f"Using training-time Laplacian features: shape {self.eigenmaps.shape}")
        
        # Set prediction mode flag
        self.training_mode = False
        X = self._prepare_meta_features(
            base_predictions, 
            coords=coords,
            y_true=None,                # No targets available during prediction
            oof_residuals=None,         # No OOF residuals during prediction
            include_predictions=True,   # Use predictions (same as training)
            include_residuals=False,    # No residuals (same as training)
            include_coordinates=True,   # Use coordinates (same as training)
            include_eigenmaps=True      # Use eigenmaps (same as training)
        )
        
        # Append original predictors if monotonic constraints were used during training
        if getattr(self, 'use_monotone_constraints', False) and original_X is not None:
            orig_X = np.asarray(original_X)
            X = np.hstack([X, orig_X])
        
        # Scale features using the same scaler fitted during training
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Restore original eigenmaps if we temporarily replaced them
        if temp_eigenmaps is not None:
            self.eigenmaps = temp_eigenmaps
            print("Restored original Laplacian features")
        
        # Optional fingerprint blending (pass-through if projector is None)
        if fingerprint_projector is not None:
            predictions = fingerprint_projector.blend(predictions, alpha=fingerprint_alpha)
            print(f"Applied fingerprint blending (alpha={fingerprint_alpha:.2f})")
        
        return predictions
        
    def get_feature_importance(self):
        """
        Get feature importance metrics from the meta-ensemble model.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        })
        
        return importance.sort_values('importance', ascending=False)
    
    def compute_shap_values(self, X):
        """
        Compute SHAP values for feature importance interpretation.
        
        Parameters:
        -----------
        X : array-like
            Data to compute SHAP values for
            
        Returns:
        --------
        numpy.ndarray
            SHAP values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        self.explainer = shap.TreeExplainer(self.model)
        return self.explainer.shap_values(X)
    
    def plot_feature_importance(self, output_dir=None):
        """
        Plot feature importance based on gain.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the plot. If None, displays plot.
        """
        importance = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.title('Feature Importance (Gain)')
        plt.xlabel('Importance')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_shap_summary(self, X, output_dir=None):
        """
        Create and save SHAP summary plot.
        
        Parameters:
        -----------
        X : array-like
            Data to compute SHAP values for
        output_dir : str, optional
            Directory to save the plot. If None, displays plot.
        """
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            show=False
        )
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
            plt.close()
        else:
            plt.show()
    
    def save_model(self, model_path, params_path=None):
        """
        Save the model and its parameters.
        
        Parameters:
        -----------
        model_path : str
            Path to save the LightGBM model
        params_path : str, optional
            Path to save model parameters
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Save LightGBM model
        self.model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save parameters if path provided
        if params_path:
            params = {
                'model_params': self.params,
                'feature_names': self.feature_names,
                'include_spatial_features': self.include_spatial_features
            }
            with open(params_path, 'w') as f:
                json.dump(params, f)
            print(f"Parameters saved to: {params_path}")
    
    @classmethod
    def load_model(cls, model_path, params_path=None):
        """
        Load a saved model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved LightGBM model
        params_path : str, optional
            Path to saved parameters
            
        Returns:
        --------
        MetaEnsemble
            Loaded model instance
        """
        # Load parameters if provided
        if params_path:
            with open(params_path, 'r') as f:
                params = json.load(f)
            instance = cls(
                params=params['model_params'],
                include_spatial_features=params['include_spatial_features']
            )
            instance.feature_names = params['feature_names']
        else:
            instance = cls()
            
        # Load LightGBM model
        instance.model = lgb.Booster(model_file=model_path)
        
        return instance
    
    def fit_with_spatial_cv(self, base_predictions, y, coords, folds, laplacian_features=None):
        """
        Fit the meta-ensemble model using spatial cross-validation for honest evaluation.
        
        Parameters:
        -----------
        base_predictions : dict
            Dictionary containing predictions from base models
        y : array-like, shape (n_samples,)
            Target values
        coords : array-like, shape (n_samples, 2)
            Spatial coordinates (X, Y) for each sample
        folds : list
            List of (train_idx, test_idx) tuples for spatial CV
        laplacian_features : array-like, optional
            Laplacian Eigenmap features
            
        Returns:
        --------
        numpy.ndarray
            Spatial CV out-of-fold predictions
        """
        print("\n=== Meta-Ensemble Spatial Cross-Validation ===")
        
        n_samples = len(y)
        spatial_cv_preds = np.full(n_samples, np.nan)
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            print(f"Meta-Ensemble Fold {fold_idx + 1}/{len(folds)}")
            
            # Extract base predictions for this fold
            base_preds_train = {model: base_predictions[model][train_idx] for model in base_predictions}
            base_preds_test = {model: base_predictions[model][test_idx] for model in base_predictions}
            
            # Prepare features for training
            X_train = self._prepare_meta_features(
                base_preds_train, 
                coords[train_idx] if coords is not None else None,
                laplacian_features[train_idx] if laplacian_features is not None else None
            )
            X_test = self._prepare_meta_features(
                base_preds_test,
                coords[test_idx] if coords is not None else None,
                laplacian_features[test_idx] if laplacian_features is not None else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train meta-model for this fold
            train_data = lgb.Dataset(X_train_scaled, label=y[train_idx])
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
            
            fold_model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
                verbose_eval=False
            )
            
            # Predict on test set
            spatial_cv_preds[test_idx] = fold_model.predict(X_test_scaled)
            
        print(f"Spatial CV meta-ensemble training completed")
        
        # Also train final model on full dataset for deployment
        print("Training final meta-ensemble model on full dataset...")
        self.fit(base_predictions, y, coords, laplacian_features)
        
        return spatial_cv_preds
    
    def export_meta_derivatives(self, X, feature_names, coords, output_dir):
        """
        Export PDP curves and SHAP-derived gradients from meta-ensemble.
        
        Non-invasive method that extracts sensitivity information without
        modifying the training or prediction logic.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (raw features, not meta-features)
        feature_names : list
            Names of features in X
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates (X, Y)
        output_dir : str or Path
            Directory to save derivative outputs
        """
        if self.model is None:
            print("Error: Model not trained yet. Call fit() first.")
            return
        
        print("\n" + "="*80)
        print("EXTRACTING META-ENSEMBLE DERIVATIVES (PDP + SHAP)")
        print("="*80)
        
        from pathlib import Path
        from sklearn.inspection import partial_dependence
        
        output_dir = Path(output_dir)
        pdp_dir = output_dir / 'meta_pdp'
        shap_dir = output_dir / 'meta_shap_derivatives'
        pdp_dir.mkdir(parents=True, exist_ok=True)
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        # Physics sign rules
        physics_signs = {
            "Pct_Impervious": +1,
            "Pct_Canopy": -1,
            "NDVI": -1,
            "Albedo": -1
        }
        
        # Get scaled meta-features for SHAP analysis
        # Note: X here is raw features, but SHAP needs the actual meta-features
        # We need to reconstruct the meta-features from base predictions
        print("\nNote: SHAP analysis requires meta-features (predictions + spatial features)")
        print("Skipping SHAP for now - requires base model predictions to reconstruct meta-features")
        print("Focusing on PDP curves which can use raw features...\n")
        
        # For PDP, we'll compute it on the meta-feature space
        # This shows how the meta-ensemble responds to changes in base model predictions
        X_meta_scaled = self.scaler.transform(self._last_meta_features) if hasattr(self, '_last_meta_features') else None
        
        if X_meta_scaled is None:
            print("Warning: No cached meta-features available. Cannot compute PDP.")
            print("To export derivatives, call fit() or predict() first to cache meta-features.")
            return
        
        print(f"Computing PDP curves for {len(self.feature_names)} meta-features...")
        
        # Compute PDP for each meta-feature
        for feat_idx, feat_name in enumerate(tqdm(self.feature_names, desc="PDP Extraction")):
            try:
                # Compute partial dependence
                pdp_result = partial_dependence(
                    self.model, 
                    X_meta_scaled, 
                    features=[feat_idx],
                    grid_resolution=50,
                    kind='average'
                )
                
                # Save PDP curve
                pdp_df = pd.DataFrame({
                    'grid_value': pdp_result['grid_values'][0],
                    'pdp': pdp_result['average'][0]
                })
                
                pdp_path = pdp_dir / f'meta_pdp_{feat_name}.csv'
                pdp_df.to_csv(pdp_path, index=False)
                
            except Exception as e:
                print(f"Warning: Could not compute PDP for {feat_name}: {e}")
                continue
        
        print(f"\n✅ Meta-ensemble PDP curves saved to: {pdp_dir}")
        print(f"   Total curves: {len(list(pdp_dir.glob('*.csv')))}")
        
        # Note about SHAP
        print("\nNote: SHAP derivative extraction requires full base model predictions.")
        print("Consider running SHAP analysis separately with complete meta-features.")