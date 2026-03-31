"""
Ordinary Least Squares (OLS) regression model implementation.
This module provides a baseline OLS model with optional Laplacian
Eigenmap features for capturing spatial structure.
"""

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm

class OLSModel:
    """
    Ordinary Least Squares (OLS) baseline model implementation.
    Uses both scikit-learn for predictions and statsmodels for detailed statistics.
    """
    
    def __init__(self):
        """Initialize OLS model instances for both sklearn and statsmodels"""
        self.sk_model = LinearRegression()
        self.model = None  # statsmodels model (for detailed statistics)
        self.scaler = StandardScaler()
        self.summary_stats = None
        self.feature_names = None
        self._fitted_on_scaled = False
        
    def fit(self, X, y):
        """
        Fit OLS model to training data (on *scaled* features for consistency)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (UNSCALED - will be scaled internally)
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'X{i}' for i in range(X.shape[1])]
        
        # --- Scale features and fit on scaled data (FIX for consistency)
        X_scaled = self.scaler.fit_transform(X)
        self.sk_model.fit(X_scaled, y)
        self._fitted_on_scaled = True
        
        # Fit statsmodels OLS for detailed statistics (also on scaled data)
        X_with_const = sm.add_constant(X_scaled)
        self.model = sm.OLS(y, X_with_const).fit()
        
        # Store feature names if available (fallback)
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'X{i}' for i in range(X.shape[1])]
        
        # Compute summary statistics
        y_pred = self.sk_model.predict(X_scaled)
        n_samples = len(y)
        n_features = X.shape[1]
        
        # R-squared and adjusted R-squared
        r2 = self.sk_model.score(X_scaled, y)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        
        # Standard errors and t-statistics
        mse = np.mean((y - y_pred) ** 2)
        X_with_intercept = np.column_stack([np.ones(n_samples), X_scaled])
        coef_with_intercept = np.concatenate([[self.sk_model.intercept_], self.sk_model.coef_])
        
        # Compute variance-covariance matrix
        try:
            var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(var_covar))
            t_stats = coef_with_intercept / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_samples-n_features-1))
        except np.linalg.LinAlgError:
            std_errors = np.full(n_features + 1, np.nan)
            t_stats = np.full(n_features + 1, np.nan)
            p_values = np.full(n_features + 1, np.nan)
        
        # Store summary statistics
        self.summary_stats = pd.DataFrame({
            'feature': ['intercept'] + self.feature_names,
            'coefficient': coef_with_intercept,
            'std_error': std_errors,
            't_statistic': t_stats,
            'p_value': p_values
        })
        
        self.summary_stats.loc['model_stats'] = [
            'model',
            r2,
            adj_r2,
            np.sqrt(mse),
            n_samples
        ]
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict (UNSCALED - will be scaled internally)
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted values
        """
        if self.sk_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # --- Scale features before prediction (FIX for consistency)
        # Backward compatibility: handle models saved before _fitted_on_scaled attribute
        if not hasattr(self, '_fitted_on_scaled'):
            # Old model - assume it was fitted on scaled data (default behavior)
            self._fitted_on_scaled = True
        
        if not self._fitted_on_scaled:
            raise RuntimeError("Model was not fitted on scaled data - internal error")
        
        X_scaled = self.scaler.transform(X)
        return self.sk_model.predict(X_scaled)
    
    def score(self, X, y):
        """
        Return R-squared score
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values
            
        Returns:
        --------
        float
            R-squared score
        """
        if self.sk_model is None:
            raise ValueError("Model must be fitted before scoring")
        return self.sk_model.score(X, y)
    
    def get_summary(self):
        """
        Get detailed model summary from statsmodels
        
        Returns:
        --------
        str
            Model summary string
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return self.model.summary().as_text()
    
    @property
    def coef_(self):
        """Get model coefficients excluding intercept"""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.model.params[1:]  # Exclude intercept
    
    @property
    def intercept_(self):
        """Get model intercept"""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing intercept")
        return self.model.params[0]
    
    @property
    def bse(self):
        """Get coefficient standard errors excluding intercept"""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing standard errors")
        return self.model.bse[1:]  # Exclude intercept
    
    @property
    def tvalues(self):
        """Get coefficient t-statistics excluding intercept"""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing t-values")
        return self.model.tvalues[1:]  # Exclude intercept
    
    @property
    def pvalues(self):
        """Get coefficient p-values excluding intercept"""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing p-values")
        return self.model.pvalues[1:]  # Exclude intercept 