import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score, mean_squared_error
from libpysal.weights import KNN as PysalKNN
from esda.moran import Moran
import matplotlib.pyplot as plt
from scipy import stats
from typing import Union, Dict, Optional, Tuple
import os

class SpatialEvaluator:
    """
    A comprehensive evaluation module for spatial models that calculates
    and visualizes various performance metrics including R², RMSE,
    and spatial autocorrelation measures.
    """
    
    def __init__(
        self,
        k_neighbors: int = 100,
        permutations: int = 999,
        random_state: int = 42
    ):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        k_neighbors : int, default=100
            Number of neighbors for spatial weights matrix
        permutations : int, default=999
            Number of permutations for Moran's I calculation
        random_state : int, default=42
            Random state for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.permutations = permutations
        self.random_state = random_state
        
    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        sample_weight : array-like, optional
            Sample weights
            
        Returns:
        --------
        dict
            Dictionary containing R² and RMSE values
        """
        # Ensure arrays are aligned and clean
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = np.asarray(y_true)[mask]
        y_pred_clean = np.asarray(y_pred)[mask]
        
        if len(y_true_clean) < 2:
            return {
                'r2_score': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_samples': len(y_true_clean)
            }
        
        # Calculate metrics
        r2 = r2_score(y_true_clean, y_pred_clean, sample_weight=sample_weight)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean, sample_weight=sample_weight))
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true_clean)
        }

    # ------------------------------------------------------------------
    # Benchmark metrics (nRMSE, pattern correlation, amplitude ratio)
    # ------------------------------------------------------------------

    def calculate_nrmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Normalized RMSE: RMSE / std(y_true).  Lower is better; 1.0 = no skill."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        if len(yt) < 2 or np.std(yt) == 0:
            return np.nan
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        return float(rmse / np.std(yt))

    def calculate_pattern_correlation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lat: Optional[np.ndarray] = None,
    ) -> float:
        """Pearson pattern correlation between two spatial fields.

        If *lat* (degrees) is provided, observations are weighted by
        cos(lat) to account for grid-cell area on a regular lat-lon grid.
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        if len(yt) < 3:
            return np.nan
        if lat is not None:
            w = np.cos(np.deg2rad(np.asarray(lat)[mask]))
            w = w / w.sum()
            yt_mean = np.average(yt, weights=w)
            yp_mean = np.average(yp, weights=w)
            cov = np.sum(w * (yt - yt_mean) * (yp - yp_mean))
            std_t = np.sqrt(np.sum(w * (yt - yt_mean) ** 2))
            std_p = np.sqrt(np.sum(w * (yp - yp_mean) ** 2))
            if std_t == 0 or std_p == 0:
                return np.nan
            return float(cov / (std_t * std_p))
        # Unweighted Pearson r
        return float(np.corrcoef(yt, yp)[0, 1])

    def calculate_amplitude_ratio(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Amplitude (variance) ratio: std(y_pred) / std(y_true).  1.0 = perfect."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        if len(yt) < 2 or np.std(yt) == 0:
            return np.nan
        return float(np.std(yp) / np.std(yt))

    def calculate_benchmark_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lat: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return all evaluation metrics in one dict (basic + benchmark).

        Parameters
        ----------
        y_true, y_pred : array-like
        lat : array-like, optional  — latitude in degrees for area-weighted pattern corr
        sample_weight : array-like, optional
        """
        basic = self.calculate_basic_metrics(y_true, y_pred, sample_weight)
        basic['nrmse'] = self.calculate_nrmse(y_true, y_pred)
        basic['pattern_correlation'] = self.calculate_pattern_correlation(y_true, y_pred, lat)
        basic['amplitude_ratio'] = self.calculate_amplitude_ratio(y_true, y_pred)
        return basic

    # ------------------------------------------------------------------
    # Extreme-value / tail metrics
    # ------------------------------------------------------------------

    def calculate_extreme_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentiles: Tuple[float, float] = (10, 90),
    ) -> Dict[str, float]:
        """Evaluate model skill in the tails of the distribution.

        Parameters
        ----------
        y_true, y_pred : array-like
        percentiles : tuple of (low, high) percentile cut-offs
            Default (10, 90) evaluates the bottom and top deciles.

        Returns
        -------
        dict with keys:
            tail_low_rmse, tail_low_r2, tail_high_rmse, tail_high_r2,
            tail_low_bias, tail_high_bias, extreme_rmse_ratio
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        if len(yt) < 10:
            return {k: np.nan for k in (
                'tail_low_rmse', 'tail_low_r2', 'tail_low_bias',
                'tail_high_rmse', 'tail_high_r2', 'tail_high_bias',
                'extreme_rmse_ratio',
            )}

        lo_thr = np.percentile(yt, percentiles[0])
        hi_thr = np.percentile(yt, percentiles[1])

        lo_mask = yt <= lo_thr
        hi_mask = yt >= hi_thr

        def _tail_stats(m):
            if m.sum() < 2:
                return np.nan, np.nan, np.nan
            rmse = np.sqrt(np.mean((yt[m] - yp[m]) ** 2))
            r2 = 1 - np.sum((yt[m] - yp[m]) ** 2) / max(np.sum((yt[m] - yt[m].mean()) ** 2), 1e-12)
            bias = float(np.mean(yp[m] - yt[m]))
            return rmse, r2, bias

        lo_rmse, lo_r2, lo_bias = _tail_stats(lo_mask)
        hi_rmse, hi_r2, hi_bias = _tail_stats(hi_mask)

        global_rmse = np.sqrt(np.mean((yt - yp) ** 2))
        tail_max = max(lo_rmse if not np.isnan(lo_rmse) else 0,
                       hi_rmse if not np.isnan(hi_rmse) else 0)
        ratio = tail_max / global_rmse if global_rmse > 0 else np.nan

        return {
            'tail_low_rmse': float(lo_rmse),
            'tail_low_r2': float(lo_r2),
            'tail_low_bias': float(lo_bias),
            'tail_high_rmse': float(hi_rmse),
            'tail_high_r2': float(hi_r2),
            'tail_high_bias': float(hi_bias),
            'extreme_rmse_ratio': float(ratio),
        }

    # ------------------------------------------------------------------
    # Baseline comparisons (skill scores)
    # ------------------------------------------------------------------

    def calculate_baseline_comparisons(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coords: Optional[np.ndarray] = None,
        k_baseline: int = 50,
    ) -> Dict[str, float]:
        """Compare model predictions against naive baselines.

        Baselines:
          1. **Global mean** — predicts mean(y_true) everywhere.
          2. **Spatial KNN mean** — predicts mean of k nearest neighbours
             in the training set.  Only computed when *coords* is provided.

        Returns skill scores: 1 - MSE_model / MSE_baseline.  Positive
        values indicate the model outperforms the baseline.
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        if len(yt) < 2:
            return {'skill_vs_global_mean': np.nan, 'skill_vs_spatial_knn': np.nan}

        mse_model = float(np.mean((yt - yp) ** 2))

        # 1. Global mean baseline
        mse_global = float(np.mean((yt - yt.mean()) ** 2))
        skill_global = 1.0 - mse_model / max(mse_global, 1e-12)

        # 2. Spatial KNN baseline
        skill_spatial = np.nan
        if coords is not None:
            try:
                from sklearn.neighbors import KNeighborsRegressor
                c = np.asarray(coords)[mask]
                k = min(k_baseline, len(c) - 1)
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                knn.fit(c, yt)
                yp_knn = knn.predict(c)
                mse_knn = float(np.mean((yt - yp_knn) ** 2))
                skill_spatial = 1.0 - mse_model / max(mse_knn, 1e-12)
            except Exception:
                pass

        return {
            'skill_vs_global_mean': float(skill_global),
            'skill_vs_spatial_knn': float(skill_spatial),
        }

    def calculate_spatial_autocorrelation(
        self,
        residuals: np.ndarray,
        coords: np.ndarray,
        return_weights: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], PysalKNN]]:
        """
        Calculate Moran's I spatial autocorrelation statistic.
        
        Parameters:
        -----------
        residuals : array-like
            Model residuals
        coords : array-like
            Spatial coordinates (n_samples, 2)
        return_weights : bool, default=False
            Whether to return the spatial weights matrix
            
        Returns:
        --------
        dict or tuple
            Dictionary containing Moran's I statistics and optionally
            the spatial weights matrix
        """
        # Clean data
        mask = ~np.isnan(residuals)
        residuals_clean = residuals[mask]
        coords_clean = coords[mask]
        
        if len(residuals_clean) < self.k_neighbors + 1:
            result = {
                'morans_i': np.nan,
                'p_value': np.nan,
                'z_score': np.nan,
                'n_samples': len(residuals_clean)
            }
            return (result, None) if return_weights else result
        
        # Calculate spatial weights
        try:
            w = PysalKNN.from_array(
                coords_clean,
                k=min(self.k_neighbors, len(coords_clean) - 1)
            )
            w.transform = 'R'  # Row-standardization
            
            # Calculate Moran's I
            moran = Moran(
                residuals_clean,
                w,
                permutations=self.permutations
            )
            
            result = {
                'morans_i': moran.I,
                'p_value': moran.p_sim,
                'z_score': moran.z_sim,
                'n_samples': len(residuals_clean)
            }
            
            return (result, w, moran) if return_weights else result
            
        except Exception as e:
            print(f"Error in Moran's I calculation: {e}")
            result = {
                'morans_i': np.nan,
                'p_value': np.nan,
                'z_score': np.nan,
                'n_samples': len(residuals_clean)
            }
            return (result, None, None) if return_weights else result
    
    def plot_morans_scatter(
        self,
        residuals: np.ndarray,
        coords: np.ndarray,
        output_dir: Optional[str] = None,
        title: str = "Moran's I Scatter Plot"
    ) -> None:
        """
        Create Moran's I scatter plot (residuals vs spatial lag).
        
        Parameters:
        -----------
        residuals : array-like
            Model residuals
        coords : array-like
            Spatial coordinates (n_samples, 2)
        output_dir : str, optional
            Directory to save the plot
        title : str, default="Moran's I Scatter Plot"
            Plot title
        """
        # Calculate spatial autocorrelation with weights
        spatial_result = self.calculate_spatial_autocorrelation(
            residuals, coords, return_weights=True
        )
        
        if len(spatial_result) == 3:
            metrics, w, moran = spatial_result
        else:
            metrics, w = spatial_result
            moran = None
            
        if w is None or moran is None:
            print("⚠ Could not calculate Moran's I - insufficient data or error")
            return
            
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get spatial lag
        spatial_lag = w.sparse.toarray().dot(residuals)
        
        # Create scatter plot
        ax.scatter(residuals, spatial_lag, alpha=0.6, s=20)
        
        # Add regression line
        z = np.polyfit(residuals, spatial_lag, 1)
        p = np.poly1d(z)
        ax.plot(residuals, p(residuals), "r--", alpha=0.8, linewidth=2)
        
        # Add zero lines
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add labels and title
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Spatial Lag of Residuals')
        ax.set_title(f"{title}\nMoran's I = {metrics['morans_i']:.4f} (p = {metrics['p_value']:.4f})")
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'morans_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_morans_permutation(
        self,
        residuals: np.ndarray,
        coords: np.ndarray,
        output_dir: Optional[str] = None,
        title: str = "Moran's I Permutation Distribution"
    ) -> None:
        """
        Create Moran's I permutation distribution plot with confidence intervals.
        
        Parameters:
        -----------
        residuals : array-like
            Model residuals
        coords : array-like
            Spatial coordinates (n_samples, 2)
        output_dir : str, optional
            Directory to save the plot
        title : str, default="Moran's I Permutation Distribution"
            Plot title
        """
        # Calculate spatial autocorrelation with weights
        spatial_result = self.calculate_spatial_autocorrelation(
            residuals, coords, return_weights=True
        )
        
        if len(spatial_result) == 3:
            metrics, w, moran = spatial_result
        else:
            metrics, w = spatial_result
            moran = None
            
        if w is None or moran is None:
            print("⚠ Could not calculate Moran's I - insufficient data or error")
            return
            
        # Create permutation distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of permutation values
        ax.hist(moran.sim, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add observed value
        ax.axvline(metrics['morans_i'], color='red', linestyle='--', linewidth=2, 
                  label=f"Observed: {metrics['morans_i']:.4f}")
        
        # Add confidence intervals
        alpha = 0.05
        lower_ci = np.percentile(moran.sim, (alpha/2) * 100)
        upper_ci = np.percentile(moran.sim, (1 - alpha/2) * 100)
        
        ax.axvline(lower_ci, color='orange', linestyle=':', linewidth=2,
                  label=f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
        ax.axvline(upper_ci, color='orange', linestyle=':', linewidth=2)
        
        # Add labels and title
        ax.set_xlabel("Moran's I")
        ax.set_ylabel('Frequency')
        ax.set_title(f"{title}\np-value = {metrics['p_value']:.4f}")
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'morans_permutation.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spatial_residuals_comparison(
        self,
        gdf: gpd.GeoDataFrame,
        residuals_before: np.ndarray,
        residuals_after: np.ndarray,
        output_dir: Optional[str] = None,
        title_prefix: str = "Spatial Residuals"
    ) -> None:
        """
        Create side-by-side comparison of spatial residuals before and after deep kriging.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            GeoDataFrame with geometry
        residuals_before : array-like
            Residuals before deep kriging (meta-model residuals)
        residuals_after : array-like
            Residuals after deep kriging (final residuals)
        output_dir : str, optional
            Directory to save the plot
        title_prefix : str, default="Spatial Residuals"
            Prefix for plot titles
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Create temporary GeoDataFrames
        gdf_before = gdf.copy()
        gdf_before['residuals'] = residuals_before
        
        gdf_after = gdf.copy()
        gdf_after['residuals'] = residuals_after
        
        # Plot before (meta-model residuals)
        gdf_before.plot(
            column='residuals',
            cmap='RdYlBu',
            legend=True,
            ax=ax1,
            legend_kwds={'shrink': 0.8}
        )
        ax1.set_title(f"{title_prefix} - Before Deep Kriging\n(Meta-Model Residuals)")
        ax1.set_aspect('equal')
        
        # Plot after (final residuals)
        gdf_after.plot(
            column='residuals',
            cmap='RdYlBu',
            legend=True,
            ax=ax2,
            legend_kwds={'shrink': 0.8}
        )
        ax2.set_title(f"{title_prefix} - After Deep Kriging\n(Final Residuals)")
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'spatial_residuals_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coords: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation including basic metrics,
        benchmark metrics (nRMSE, pattern correlation, amplitude ratio),
        and spatial autocorrelation of residuals.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        coords : array-like
            Spatial coordinates (n_samples, 2)
        sample_weight : array-like, optional
            Sample weights
        lat : array-like, optional
            Latitude in degrees for area-weighted pattern correlation
            
        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        # Calculate basic + benchmark metrics
        metrics = self.calculate_benchmark_metrics(y_true, y_pred, lat, sample_weight)
        
        # Calculate residuals and their spatial autocorrelation
        residuals = y_true - y_pred
        spatial_metrics = self.calculate_spatial_autocorrelation(residuals, coords)
        
        # Combine metrics
        return {**metrics, **spatial_metrics}
    
    def plot_residuals_map(
        self,
        gdf: gpd.GeoDataFrame,
        residuals: np.ndarray,
        output_dir: Optional[str] = None,
        title: str = "Spatial Distribution of Residuals"
    ) -> None:
        """
        Plot spatial distribution of residuals.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            GeoDataFrame with geometry
        residuals : array-like
            Model residuals
        output_dir : str, optional
            Directory to save the plot
        title : str, default="Spatial Distribution of Residuals"
            Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create temporary GeoDataFrame with residuals
        gdf_temp = gdf.copy()
        gdf_temp['residuals'] = residuals
        
        # Plot residuals
        gdf_temp.plot(
            column='residuals',
            cmap='RdYlBu',
            legend=True,
            ax=ax
        )
        
        plt.title(title)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'residuals_map.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_qq(
        self,
        residuals: np.ndarray,
        output_dir: Optional[str] = None,
        title: str = "Q-Q Plot of Residuals"
    ) -> None:
        """
        Create Q-Q plot of residuals.
        
        Parameters:
        -----------
        residuals : array-like
            Model residuals
        output_dir : str, optional
            Directory to save the plot
        title : str, default="Q-Q Plot of Residuals"
            Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Clean residuals
        residuals_clean = residuals[~np.isnan(residuals)]
        
        # Create Q-Q plot
        stats.probplot(residuals_clean, dist="norm", plot=ax)
        ax.set_title(title)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'qq_plot.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_residuals_vs_predicted(
        self,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        output_dir: Optional[str] = None,
        title: str = "Residuals vs Predicted Values"
    ) -> None:
        """
        Plot residuals against predicted values.
        
        Parameters:
        -----------
        y_pred : array-like
            Predicted values
        residuals : array-like
            Model residuals
        output_dir : str, optional
            Directory to save the plot
        title : str, default="Residuals vs Predicted Values"
            Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Clean data
        mask = ~(np.isnan(y_pred) | np.isnan(residuals))
        y_pred_clean = y_pred[mask]
        residuals_clean = residuals[mask]
        
        # Create scatter plot
        plt.scatter(y_pred_clean, residuals_clean, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'residuals_vs_predicted.png'))
            plt.close()
        else:
            plt.show()
    
    def create_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coords: np.ndarray,
        gdf: Optional[gpd.GeoDataFrame] = None,
        output_dir: Optional[str] = None,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Create a comprehensive evaluation report including metrics
        and diagnostic plots.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        coords : array-like
            Spatial coordinates (n_samples, 2)
        gdf : GeoDataFrame, optional
            GeoDataFrame for spatial plotting
        output_dir : str, optional
            Directory to save plots and results
        model_name : str, default="Model"
            Name of the model being evaluated
            
        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Calculate all metrics
        metrics = self.evaluate_model(y_true, y_pred, coords)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(
                os.path.join(output_dir, f'{model_name.lower()}_metrics.csv'),
                index=False
            )
            
            # Create diagnostic plots
            if gdf is not None:
                self.plot_residuals_map(
                    gdf, residuals,
                    output_dir=output_dir,
                    title=f"{model_name}: Spatial Distribution of Residuals"
                )
            
            self.plot_qq(
                residuals,
                output_dir=output_dir,
                title=f"{model_name}: Q-Q Plot of Residuals"
            )
            
            self.plot_residuals_vs_predicted(
                y_pred, residuals,
                output_dir=output_dir,
                title=f"{model_name}: Residuals vs Predicted Values"
            )
            
            # Create Moran's I plots
            self.plot_morans_scatter(
                residuals, coords,
                output_dir=output_dir,
                title=f"{model_name}: Moran's I Scatter Plot"
            )
            
            self.plot_morans_permutation(
                residuals, coords,
                output_dir=output_dir,
                title=f"{model_name}: Moran's I Permutation Distribution"
            )
        
        # Print summary
        print("\nEvaluation Results:")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        if 'nrmse' in metrics and not np.isnan(metrics.get('nrmse', np.nan)):
            print(f"nRMSE: {metrics['nrmse']:.4f}")
            print(f"Pattern Correlation: {metrics['pattern_correlation']:.4f}")
            print(f"Amplitude Ratio: {metrics['amplitude_ratio']:.4f}")
        print(f"Moran's I: {metrics['morans_i']:.4f} (p-value: {metrics['p_value']:.4f})")
        
        return metrics