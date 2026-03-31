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
    
    def export_predictions_gpkg(
        self,
        gdf: gpd.GeoDataFrame,
        y_true: np.ndarray,
        meta_predictions: np.ndarray,
        dk_corrections: np.ndarray,
        final_predictions: np.ndarray,
        output_path: str,
        additional_columns: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Export predictions and residuals to a GeoPackage file.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Base GeoDataFrame with geometry
        y_true : array-like
            Ground truth values
        meta_predictions : array-like
            Meta-model predictions
        dk_corrections : array-like
            Deep kriging corrections
        final_predictions : array-like
            Final predictions (meta + deep kriging)
        output_path : str
            Path to save the GeoPackage file
        additional_columns : dict, optional
            Additional columns to include in the export
        """
        # Create a copy of the GeoDataFrame
        export_gdf = gdf.copy()
        
        # Add prediction columns
        export_gdf['y_true'] = y_true
        export_gdf['meta_pred'] = meta_predictions
        export_gdf['dk_correction'] = dk_corrections
        export_gdf['final_pred'] = final_predictions
        
        # Add residual columns
        export_gdf['meta_residual'] = y_true - meta_predictions
        export_gdf['final_residual'] = y_true - final_predictions
        
        # Add improvement column
        export_gdf['improvement'] = np.abs(export_gdf['meta_residual']) - np.abs(export_gdf['final_residual'])
        
        # Add additional columns if provided
        if additional_columns:
            for col_name, col_data in additional_columns.items():
                export_gdf[col_name] = col_data
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export to GeoPackage
        export_gdf.to_file(output_path, driver='GPKG')
        print(f"✓ Predictions exported to: {output_path}")
        print(f"  Shape: {export_gdf.shape}")
        print(f"  Columns: {list(export_gdf.columns)}")
    
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

    def evaluate_deep_kriging_corrections(
        self,
        y_true: np.ndarray,
        meta_predictions: np.ndarray,
        dk_corrections: np.ndarray,
        coords: np.ndarray,
        gdf: Optional[gpd.GeoDataFrame] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the performance improvement from deep kriging corrections.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        meta_predictions : array-like
            Meta-model predictions
        dk_corrections : array-like
            Deep kriging corrections
        coords : array-like
            Spatial coordinates (n_samples, 2)
        gdf : GeoDataFrame, optional
            GeoDataFrame for spatial plotting
        output_dir : str, optional
            Directory to save plots and results
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics for both models
        """
        # Calculate final predictions
        final_predictions = meta_predictions + dk_corrections
        
        # Calculate residuals
        meta_residuals = y_true - meta_predictions
        final_residuals = y_true - final_predictions
        
        # Evaluate meta-model
        meta_metrics = self.evaluate_model(y_true, meta_predictions, coords)
        
        # Evaluate final model
        final_metrics = self.evaluate_model(y_true, final_predictions, coords)
        
        # Print comparison
        print("\n=== Deep Kriging Correction Performance ===")
        print(f"Meta-Model Only: R² = {meta_metrics['r2_score']:.4f}, RMSE = {meta_metrics['rmse']:.4f}")
        print(f"Meta + Deep Kriging: R² = {final_metrics['r2_score']:.4f}, RMSE = {final_metrics['rmse']:.4f}")
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save comparison metrics
            comparison_df = pd.DataFrame({
                'Model': ['Meta-Model', 'Meta + Deep Kriging'],
                'R²': [meta_metrics['r2_score'], final_metrics['r2_score']],
                'RMSE': [meta_metrics['rmse'], final_metrics['rmse']],
                'MAE': [meta_metrics['mae'], final_metrics['mae']],
                'Moran_I': [meta_metrics['morans_i'], final_metrics['morans_i']],
                'Moran_p_value': [meta_metrics['p_value'], final_metrics['p_value']]
            })
            comparison_df.to_csv(os.path.join(output_dir, 'deep_kriging_comparison.csv'), index=False)
            
            # Create spatial residuals comparison
            if gdf is not None:
                self.plot_spatial_residuals_comparison(
                    gdf, meta_residuals, final_residuals,
                    output_dir=output_dir,
                    title_prefix="Spatial Residuals"
                )
            
            # Create diagnostic plots for final model
            self.plot_qq(
                final_residuals,
                output_dir=output_dir,
                title="Meta + Deep Kriging: Q-Q Plot of Residuals"
            )
            
            self.plot_residuals_vs_predicted(
                final_predictions, final_residuals,
                output_dir=output_dir,
                title="Meta + Deep Kriging: Residuals vs Predicted Values"
            )
            
            # Create Moran's I plots for final model
            self.plot_morans_scatter(
                final_residuals, coords,
                output_dir=output_dir,
                title="Meta + Deep Kriging: Moran's I Scatter Plot"
            )
            
            self.plot_morans_permutation(
                final_residuals, coords,
                output_dir=output_dir,
                title="Meta + Deep Kriging: Moran's I Permutation Distribution"
            )
        
        return {
            'meta_metrics': meta_metrics,
            'final_metrics': final_metrics
        } 