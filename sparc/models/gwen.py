"""
Geographically Weighted Elastic Net (GWEN) model implementation.
This module provides variable selection capabilities using spatially
weighted elastic net regression.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GWENModel:
    """
    Geographically Weighted Elastic Net model for spatial variable selection.
    
    Parameters:
    -----------
    k_neighbors : int, default=None
        Number of neighbors for adaptive bandwidth. If None, uses min(500, n_samples//2)
    cv_folds : int, default=5
        Number of cross-validation folds for ElasticNet
    n_alphas : int, default=100
        Number of alpha values to try
    l1_ratios : list, default=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        List of l1_ratio values to try
    sample_size : int, optional
        Number of samples to use for fitting. If None, uses all samples
    quick_mode : bool, default=False
        When True, use fixed ElasticNet (no CV) and reduced sample for speed
    n_jobs : int, default=1
        Number of parallel jobs for ElasticNetCV (-1 = all cores)
    """
    
    def __init__(self, k_neighbors=None, cv_folds=5, n_alphas=100,
                 l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], sample_size=None,
                 quick_mode=False, n_jobs=1, spatial_cv_folds=None):
        self.k_neighbors = k_neighbors
        self.cv_folds = cv_folds
        self.n_alphas = n_alphas
        self.l1_ratios = l1_ratios
        self.sample_size = sample_size
        self.quick_mode = quick_mode
        self.n_jobs = n_jobs
        self.spatial_cv_folds = spatial_cv_folds  # list of (train_idx, test_idx) tuples
        self.scaler = StandardScaler()
        self.global_model = None
        self.local_models = None
        self.feature_importance = None
        self.sample_indices = None
        self._nn_indices = None  # Cached k-NN indices
        
    def fit(self, X, y, coords):
        """
        Fit the GWEN model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        coords : array-like, shape (n_samples, 2)
            Spatial coordinates (X, Y) for each sample
        """
        print("1. Scaling features...")
        X = np.asarray(X)
        y = np.asarray(y)
        coords = np.asarray(coords)
        
        # Apply sampling if specified (or force smaller sample in quick_mode)
        effective_sample = self.sample_size
        if self.quick_mode and (effective_sample is None or effective_sample > 2000):
            effective_sample = min(2000, len(X))
            print(f"   Quick mode: capping fit locations to {effective_sample}")

        if effective_sample is not None and effective_sample < len(X):
            print(f"   Sampling {effective_sample} points from {len(X)} total points...")
            np.random.seed(42)
            self.sample_indices = np.random.choice(len(X), effective_sample, replace=False)
            X = X[self.sample_indices]
            y = y[self.sample_indices]
            coords = coords[self.sample_indices]
            print(f"   Sampled data shape: {X.shape}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of neighbors
        n_samples = len(X)
        if self.k_neighbors is None:
            self.k_neighbors = min(500, n_samples // 2)
        
        # Fit global model
        print("\n2. Tuning global ElasticNet hyperparameters...")

        # Build CV splitter: spatial folds (preferred) or standard k-fold
        if self.spatial_cv_folds is not None:
            cv_iter = self.spatial_cv_folds
            print(f"   Using {len(cv_iter)}-fold spatial block CV")
        else:
            cv_iter = self.cv_folds

        self.global_model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            n_alphas=self.n_alphas,
            cv=cv_iter,
            random_state=42,
            n_jobs=self.n_jobs,
        )
        self.global_model.fit(X_scaled, y)
        
        print(f"   Selected alpha={self.global_model.alpha_:.4f}, "
              f"l1_ratio={self.global_model.l1_ratio_:.2f}")
        
        # Pre-compute k-NN indices once (reusable across all local models)
        print(f"\n3. Computing adaptive bandwidths using {self.k_neighbors}-NN...")
        nn = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='ball_tree')
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        self._nn_indices = indices
        
        # Initialize local models
        self.local_models = []
        local_coefs = []
        
        # Train local models
        print("\n4. Training local models...")
        # In quick_mode, use fixed alpha/l1_ratio from global model (skip CV)
        use_global_alpha = self.quick_mode and self.global_model is not None
        if use_global_alpha:
            print("   Quick mode: using global alpha/l1_ratio (no per-location CV)")

        for i in tqdm(range(len(coords)), desc="Training"):
            local_indices = indices[i]
            X_local = X_scaled[local_indices]
            y_local = y[local_indices]
            
            if use_global_alpha:
                model = ElasticNet(
                    alpha=self.global_model.alpha_,
                    l1_ratio=self.global_model.l1_ratio_,
                    random_state=42,
                )
            else:
                model = ElasticNetCV(
                    l1_ratio=self.l1_ratios,
                    n_alphas=self.n_alphas,
                    cv=self.cv_folds,
                    random_state=42,
                    n_jobs=self.n_jobs,
                )
            model.fit(X_local, y_local)
            
            self.local_models.append(model)
            local_coefs.append(model.coef_)
        
        # Compute feature importance
        local_coefs = np.array(local_coefs)
        self.feature_importance = pd.DataFrame({
            'feature': range(X.shape[1]),
            'global_coef': self.global_model.coef_,
            'mean_local_coef': np.mean(np.abs(local_coefs), axis=0),
            'std_local_coef': np.std(np.abs(local_coefs), axis=0),
            'selection_frequency': np.mean(local_coefs != 0, axis=0)
        })
        
        return self
    
    def predict(self, X, coords):
        """
        Make predictions using the GWEN model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        coords : array-like, shape (n_samples, 2)
            Spatial coordinates (X, Y) for each sample
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted values
        """
        if self.local_models is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        coords = np.asarray(coords)
        predictions = np.zeros(len(X))
        
        # Use nearest local model for each prediction point
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        
        for i, idx in enumerate(indices.flatten()):
            predictions[i] = self.local_models[idx].predict(X_scaled[i:i+1])[0]
            
        return predictions
    
    def fit_on_subset(self, X, y, coords, subset_idx):
        """Fit a lightweight GWEN on a data subset and return local coefficients.

        Used by stability scoring to evaluate feature selection consistency across
        spatial CV folds.  Returns a 2-D array of shape (n_fit_locations, n_features)
        containing the local ElasticNet coefficients.
        """
        X_sub = np.asarray(X)[subset_idx]
        y_sub = np.asarray(y)[subset_idx]
        coords_sub = np.asarray(coords)[subset_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        k = min(self.k_neighbors or min(500, len(X_sub) // 2), len(X_sub) - 1)

        # Global alpha from the subset
        global_model = ElasticNetCV(
            l1_ratio=self.l1_ratios, n_alphas=self.n_alphas,
            cv=min(self.cv_folds, 3), random_state=42, n_jobs=self.n_jobs,
        )
        global_model.fit(X_scaled, y_sub)

        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        nn.fit(coords_sub)
        _, indices = nn.kneighbors(coords_sub)

        # Subsample fit locations for speed
        n_fit = min(500, len(X_sub))
        rng = np.random.RandomState(42)
        fit_locs = rng.choice(len(X_sub), n_fit, replace=False)

        local_coefs = []
        for loc in fit_locs:
            local_idx = indices[loc]
            model = ElasticNet(
                alpha=global_model.alpha_,
                l1_ratio=global_model.l1_ratio_,
                random_state=42,
            )
            model.fit(X_scaled[local_idx], y_sub[local_idx])
            local_coefs.append(model.coef_)

        return np.array(local_coefs)

    def get_variable_importance(self):
        """
        Get variable importance metrics.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing variable importance metrics
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before getting variable importance")
        return self.feature_importance
    
    def get_selected_predictors(self, feature_names=None, threshold=0.1):
        """
        Get selected predictors based on variable importance.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of the features. If None, will use X1, X2, etc.
        threshold : float, default=0.1
            Threshold for selection frequency (percentage of local models that selected the variable)
            
        Returns:
        --------
        list
            List of selected predictor names
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before getting selected predictors")
        
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(self.feature_importance.shape[0])]
        
        # Select predictors based on selection frequency
        selected_mask = self.feature_importance['selection_frequency'] >= threshold
        selected_indices = self.feature_importance[selected_mask]['feature'].values
        
        selected_predictors = [feature_names[int(idx)] for idx in selected_indices]
        
        print(f"Selected {len(selected_predictors)} predictors with selection frequency >= {threshold}")
        print(f"Selected predictors: {selected_predictors}")
        
        return selected_predictors
    
    def compute_variable_importance(self, feature_names=None):
        """
        Compute variable importance metrics based on local coefficients.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of the features. If None, will use X1, X2, etc.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing variable importance metrics
        """
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(self.local_coefs.shape[1]-1)]
            
        print("\n5. Computing variable importance metrics...")
        importance = []
        
        for j, var in enumerate(feature_names):
            coefs_j = self.local_coefs[:, j+1]
            nonzero_pct = np.mean(coefs_j != 0)
            mean_abs = np.mean(np.abs(coefs_j[coefs_j != 0])) if nonzero_pct > 0 else 0.0
            median_abs = np.median(np.abs(coefs_j[coefs_j != 0])) if nonzero_pct > 0 else 0.0
            score = nonzero_pct * mean_abs
            
            importance.append({
                'predictor': var,
                'nonzero_pct': nonzero_pct,
                'mean_abs_coef': mean_abs,
                'median_abs_coef': median_abs,
                'importance_score': score
            })
        
        return pd.DataFrame(importance).sort_values('importance_score', ascending=False)
    
    def save_variable_importance(self, output_path, feature_names=None):
        """
        Compute and save variable importance metrics to a CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the CSV file
        feature_names : list, optional
            Names of the features
            
        Returns:
        --------
        pandas.DataFrame
            The computed variable importance metrics
        """
        imp_df = self.compute_variable_importance(feature_names)
        imp_df.to_csv(output_path, index=False)
        print(f"\n✅ GWEN variable importance saved to: {output_path}")
        print("\nTop variables by importance score:")
        print(imp_df.head())
        return imp_df 