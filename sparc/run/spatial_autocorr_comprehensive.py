"""
Comprehensive Spatial Autocorrelation Analysis Module

This module implements:
1. Correlogram-based block size selection for spatial CV
2. Formal Global Moran's I reporting with statistical significance
3. Climate autocorrelation scaling analysis across resolutions
4. Spatial residual mapping and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class SpatialAutocorrelationAnalyzer:
    """
    Comprehensive spatial autocorrelation analysis with formal statistical testing
    """
    
    def __init__(self, coords, max_distance=None, n_lags=20):
        """
        Initialize the spatial autocorrelation analyzer
        
        Parameters:
        -----------
        coords : array-like, shape (n_samples, 2)
            Spatial coordinates (x, y)
        max_distance : float, optional
            Maximum distance for correlogram analysis
        n_lags : int, default=20
            Number of distance lags for correlogram
        """
        self.coords = np.array(coords)
        
        # Remove any NaN coordinates
        valid_coords_mask = ~np.isnan(self.coords).any(axis=1)
        self.coords = self.coords[valid_coords_mask]
        
        self.n_samples = len(self.coords)
        
        if self.n_samples < 10:
            raise ValueError(f"Insufficient valid coordinates: {self.n_samples}. Need at least 10 points.")
        
        # Calculate distance matrix
        self.distances = squareform(pdist(self.coords))
        
        # Set maximum distance for analysis with reasonable default threshold
        if max_distance is None:
            # Use 1800m as default maximum distance to prevent excessive computation
            self.max_distance = min(1800.0, np.percentile(self.distances[self.distances > 0], 50))
        else:
            self.max_distance = max_distance
            
        self.n_lags = n_lags
        self.lag_distances = np.linspace(0, self.max_distance, n_lags + 1)
        
    def calculate_global_morans_i(self, values, weights_matrix=None, use_knn=True, k_neighbors=8):
        """
        Calculate Global Moran's I with formal statistical testing
        
        Parameters:
        -----------
        values : array-like
            Variable values for spatial autocorrelation analysis
        weights_matrix : array-like, optional
            Custom spatial weights matrix
        use_knn : bool, default=True
            Use k-nearest neighbors for weights if no custom matrix provided
        k_neighbors : int, default=8
            Number of neighbors for KNN weights
            
        Returns:
        --------
        dict : Dictionary containing:
            - morans_i : float, Moran's I statistic
            - expected_i : float, Expected Moran's I under null hypothesis
            - variance_i : float, Variance of Moran's I
            - z_score : float, Standardized z-score
            - p_value : float, Two-tailed p-value
            - interpretation : str, Statistical interpretation
        """
        values = np.array(values)
        
        # Remove NaN values and corresponding coordinates
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 10:
            return {
                'morans_i': 0.0,
                'expected_i': -1/9,
                'variance_i': 0.0,
                'z_score': 0.0,
                'p_value': 1.0,
                'interpretation': 'Insufficient valid data points'
            }
        
        values = values[valid_mask]
        valid_coords = self.coords[valid_mask] if hasattr(self, '_coord_mask') else self.coords[:len(values)][valid_mask]
        n = len(values)
        
        # Create weights matrix if not provided
        if weights_matrix is None:
            if use_knn:
                weights_matrix = self._create_knn_weights_for_subset(valid_coords, k_neighbors)
            else:
                weights_matrix = self._create_distance_weights_for_subset(valid_coords)
        else:
            # If custom weights provided, subset it to valid indices
            weights_matrix = weights_matrix[valid_mask][:, valid_mask]
        # Ensure weights matrix is proper
        weights_matrix = np.array(weights_matrix)
        np.fill_diagonal(weights_matrix, 0)  # No self-neighbors
        
        # Row-standardize weights
        row_sums = np.sum(weights_matrix, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights_matrix = weights_matrix / row_sums[:, np.newaxis]
        
        # Calculate Moran's I
        values_centered = values - np.mean(values)
        
        # Numerator: spatial covariance (vectorised — avoids O(n²) Python loop)
        numerator = float(values_centered @ weights_matrix @ values_centered)
        
        # Denominator: variance
        denominator = np.sum(values_centered**2)
        
        # Sum of weights
        W = np.sum(weights_matrix)
        
        if denominator == 0 or W == 0:
            return {
                'morans_i': 0.0,
                'expected_i': -1/(n-1),
                'variance_i': 0.0,
                'z_score': 0.0,
                'p_value': 1.0,
                'interpretation': 'No variation in data'
            }
        
        morans_i = (n / W) * (numerator / denominator)
        
        # Expected value under null hypothesis
        expected_i = -1 / (n - 1)
        
        # Calculate variance of Moran's I (simplified formula)
        # For large n, variance approximates to 1/(n-1)
        variance_i = 1 / (n - 1)
        
        # Z-score and p-value
        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        else:
            z_score = 0.0
            p_value = 1.0
        
        # Interpretation
        if abs(z_score) <= 1.96:
            interpretation = "No significant spatial autocorrelation (|z| ≤ 1.96)"
        elif z_score > 1.96:
            interpretation = f"Significant positive spatial autocorrelation (z = {z_score:.3f})"
        else:
            interpretation = f"Significant negative spatial autocorrelation (z = {z_score:.3f})"
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'variance_i': variance_i,
            'z_score': z_score,
            'p_value': p_value,
            'interpretation': interpretation
        }
    
    def compute_correlogram(self, values, plot=True, title="Spatial Correlogram"):
        """
        Compute spatial correlogram and identify optimal block size
        
        Parameters:
        -----------
        values : array-like
            Variable values for correlogram analysis
        plot : bool, default=True
            Whether to create correlogram plot
        title : str
            Title for the plot
            
        Returns:
        --------
        dict : Dictionary containing:
            - lag_distances : array, Distance lags
            - morans_i_values : array, Moran's I at each lag
            - z_scores : array, Z-scores at each lag
            - p_values : array, P-values at each lag
            - optimal_block_size : float, First non-significant distance
            - correlogram_results : list, Detailed results for each lag
        """
        values = np.array(values)
        correlogram_results = []
        morans_i_values = []
        z_scores = []
        p_values = []
        
        for i in range(self.n_lags):
            lag_min = self.lag_distances[i]
            lag_max = self.lag_distances[i + 1]
            
            # Create distance-based weights for this lag
            weights_matrix = np.zeros((self.n_samples, self.n_samples))
            
            # Find pairs within this distance range
            mask = (self.distances >= lag_min) & (self.distances < lag_max)
            weights_matrix[mask] = 1
            
            # Skip if no pairs in this lag
            if np.sum(weights_matrix) == 0:
                correlogram_results.append({
                    'lag_distance': (lag_min + lag_max) / 2,
                    'n_pairs': 0,
                    'morans_i': 0,
                    'z_score': 0,
                    'p_value': 1.0,
                    'significant': False
                })
                morans_i_values.append(0)
                z_scores.append(0)
                p_values.append(1.0)
                continue
            
            # Calculate Moran's I for this lag
            result = self.calculate_global_morans_i(values, weights_matrix)
            
            lag_result = {
                'lag_distance': (lag_min + lag_max) / 2,
                'n_pairs': int(np.sum(weights_matrix) / 2),  # Divide by 2 for symmetric matrix
                'morans_i': result['morans_i'],
                'z_score': result['z_score'],
                'p_value': result['p_value'],
                'significant': abs(result['z_score']) > 1.96
            }
            
            correlogram_results.append(lag_result)
            morans_i_values.append(result['morans_i'])
            z_scores.append(result['z_score'])
            p_values.append(result['p_value'])
        
        # Find optimal block size (first non-significant Moran's I)
        optimal_block_size = self.max_distance  # Default fallback
        for result in correlogram_results:
            if not result['significant'] and result['n_pairs'] > 0:
                optimal_block_size = result['lag_distance']
                break
        
        # Find first zero-crossing: lag distance where Moran's I first dips below 0
        # This is the effective range / bandwidth — beyond this distance,
        # spatial autocorrelation has decayed to nothing.
        first_zero_crossing = None
        for result in correlogram_results:
            if result['n_pairs'] > 0 and result['morans_i'] <= 0:
                first_zero_crossing = result['lag_distance']
                break
        # Fallback: if Moran's I never dips below 0, use the max distance
        if first_zero_crossing is None:
            first_zero_crossing = self.max_distance
        
        # Create plot if requested
        if plot:
            self._plot_correlogram(correlogram_results, title, optimal_block_size)
        
        return {
            'lag_distances': np.array([r['lag_distance'] for r in correlogram_results]),
            'morans_i_values': np.array(morans_i_values),
            'z_scores': np.array(z_scores),
            'p_values': np.array(p_values),
            'optimal_block_size': optimal_block_size,
            'first_zero_crossing': first_zero_crossing,
            'correlogram_results': correlogram_results
        }
    
    def _create_knn_weights(self, k_neighbors):
        """Create k-nearest neighbors weights matrix"""
        k_neighbors = min(k_neighbors, self.n_samples - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(self.coords)
        distances, indices = nbrs.kneighbors(self.coords)
        
        weights_matrix = np.zeros((self.n_samples, self.n_samples))
        
        for i in range(self.n_samples):
            # Skip the first neighbor (self)
            for j in range(1, k_neighbors + 1):
                if j < len(indices[i]):
                    neighbor_idx = indices[i][j]
                    weights_matrix[i][neighbor_idx] = 1
        
        return weights_matrix
    
    def _create_knn_weights_for_subset(self, coords, k_neighbors):
        """Create k-nearest neighbors weights matrix for subset of coordinates"""
        n_points = len(coords)
        k_neighbors = min(k_neighbors, n_points - 1)
        
        if k_neighbors < 1:
            return np.zeros((n_points, n_points))
        
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        weights_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            # Skip the first neighbor (self)
            for j in range(1, min(k_neighbors + 1, len(indices[i]))):
                neighbor_idx = indices[i][j]
                weights_matrix[i][neighbor_idx] = 1
        
        return weights_matrix
    
    def _create_distance_weights(self, threshold=None):
        """Create distance-based weights matrix"""
        if threshold is None:
            threshold = self.max_distance / 4
        
        weights_matrix = (self.distances <= threshold).astype(float)
        np.fill_diagonal(weights_matrix, 0)
        
        return weights_matrix
    
    def _create_distance_weights_for_subset(self, coords, threshold=None):
        """Create distance-based weights matrix for subset of coordinates"""
        if threshold is None:
            threshold = self.max_distance / 4
        
        distances = squareform(pdist(coords))
        weights_matrix = (distances <= threshold).astype(float)
        np.fill_diagonal(weights_matrix, 0)
        
        return weights_matrix
    
    def _plot_correlogram(self, correlogram_results, title, optimal_block_size):
        """Plot spatial correlogram"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        distances = [r['lag_distance'] for r in correlogram_results]
        morans_i = [r['morans_i'] for r in correlogram_results]
        z_scores = [r['z_score'] for r in correlogram_results]
        significant = [r['significant'] for r in correlogram_results]
        
        # Plot Moran's I values
        colors = ['red' if sig else 'blue' for sig in significant]
        ax1.scatter(distances, morans_i, c=colors, alpha=0.7)
        ax1.plot(distances, morans_i, 'k-', alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=optimal_block_size, color='green', linestyle='--', 
                   label=f'Optimal Block Size: {optimal_block_size:.1f}m')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel("Moran's I")
        ax1.set_title(f'{title} - Moran\'s I')
        ax1.set_ylim(-1, 1)  # Set y-axis range from -1 to +1
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot z-scores with significance threshold
        ax2.scatter(distances, z_scores, c=colors, alpha=0.7)
        ax2.plot(distances, z_scores, 'k-', alpha=0.3)
        ax2.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax2.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axvline(x=optimal_block_size, color='green', linestyle='--')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Z-score')
        ax2.set_title('Statistical Significance (Z-scores)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_model_residuals(self, residuals_dict, model_names):
        """
        Analyze spatial autocorrelation in model residuals
        
        Parameters:
        -----------
        residuals_dict : dict
            Dictionary with model names as keys and residual arrays as values
        model_names : list
            List of model names to analyze
            
        Returns:
        --------
        pandas.DataFrame : Results table with Moran's I statistics for each model
        """
        results = []
        
        for model_name in model_names:
            if model_name in residuals_dict:
                residuals = residuals_dict[model_name]
                
                # Skip if all residuals are NaN
                if np.all(np.isnan(residuals)):
                    result = {
                        'Model': model_name,
                        'Morans_I': np.nan,
                        'Expected_I': np.nan,
                        'Z_Score': np.nan,
                        'P_Value': np.nan,
                        'Significant': False,
                        'Interpretation': 'All residuals are NaN'
                    }
                else:
                    # Remove NaN values
                    valid_mask = ~np.isnan(residuals)
                    if np.sum(valid_mask) < 10:  # Need minimum samples
                        result = {
                            'Model': model_name,
                            'Morans_I': np.nan,
                            'Expected_I': np.nan,
                            'Z_Score': np.nan,
                            'P_Value': np.nan,
                            'Significant': False,
                            'Interpretation': 'Insufficient valid residuals'
                        }
                    else:
                        valid_residuals = residuals[valid_mask]
                        valid_coords = self.coords[valid_mask]
                        
                        # Create temporary analyzer for valid data
                        temp_analyzer = SpatialAutocorrelationAnalyzer(valid_coords)
                        moran_result = temp_analyzer.calculate_global_morans_i(valid_residuals)
                        
                        result = {
                            'Model': model_name,
                            'Morans_I': moran_result['morans_i'],
                            'Expected_I': moran_result['expected_i'],
                            'Z_Score': moran_result['z_score'],
                            'P_Value': moran_result['p_value'],
                            'Significant': abs(moran_result['z_score']) > 1.96,
                            'Interpretation': moran_result['interpretation']
                        }
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def climate_autocorrelation_scaling(self, data_dict, resolutions, variables):
        """
        Analyze climate autocorrelation scaling across resolutions
        
        Parameters:
        -----------
        data_dict : dict
            Nested dictionary: {resolution: {variable: values}}
        resolutions : list
            List of spatial resolutions to analyze
        variables : list
            List of variable names to analyze
            
        Returns:
        --------
        pandas.DataFrame : Scaling analysis results
        dict : Dictionary with detailed results for each resolution/variable
        """
        scaling_results = []
        detailed_results = {}
        
        for resolution in resolutions:
            if resolution not in data_dict:
                continue
                
            detailed_results[resolution] = {}
            
            for variable in variables:
                if variable not in data_dict[resolution]:
                    continue
                
                values = data_dict[resolution][variable]
                coords = data_dict[resolution]['coords']  # Assume coords are included
                
                # Create analyzer for this resolution
                analyzer = SpatialAutocorrelationAnalyzer(coords)
                moran_result = analyzer.calculate_global_morans_i(values)
                
                # Store detailed results
                detailed_results[resolution][variable] = moran_result
                
                # Add to scaling results
                scaling_results.append({
                    'Resolution': resolution,
                    'Variable': variable,
                    'Morans_I': moran_result['morans_i'],
                    'Z_Score': moran_result['z_score'],
                    'P_Value': moran_result['p_value'],
                    'Significant': abs(moran_result['z_score']) > 1.96
                })
        
        scaling_df = pd.DataFrame(scaling_results)
        
        # Create scaling plots
        self._plot_scaling_analysis(scaling_df, variables)
        
        return scaling_df, detailed_results
    
    def _plot_scaling_analysis(self, scaling_df, variables):
        """Plot resolution vs. Moran's I for climate autocorrelation scaling"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Moran's I vs Resolution
        for variable in variables:
            var_data = scaling_df[scaling_df['Variable'] == variable]
            if len(var_data) > 0:
                axes[0].plot(var_data['Resolution'], var_data['Morans_I'], 
                           'o-', label=variable, linewidth=2, markersize=6)
        
        axes[0].set_xlabel('Spatial Resolution (m)')
        axes[0].set_ylabel("Moran's I")
        axes[0].set_title('Climate Autocorrelation Scaling')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Statistical Significance
        for variable in variables:
            var_data = scaling_df[scaling_df['Variable'] == variable]
            if len(var_data) > 0:
                colors = ['red' if sig else 'blue' for sig in var_data['Significant']]
                axes[1].scatter(var_data['Resolution'], var_data['Z_Score'], 
                              c=colors, label=variable, s=60, alpha=0.7)
        
        axes[1].axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        axes[1].axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_xlabel('Spatial Resolution (m)')
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title('Statistical Significance Across Resolutions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_spatial_residual_maps(coords, residuals_dict, model_names, output_dir=None):
    """
    Create spatial residual maps for visualization
    
    Parameters:
    -----------
    coords : array-like
        Spatial coordinates
    residuals_dict : dict
        Dictionary with model residuals
    model_names : list
        List of model names
    output_dir : str, optional
        Directory to save plots
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        if i >= len(axes):
            break
            
        if model_name in residuals_dict:
            residuals = residuals_dict[model_name]
            
            # Remove NaN values for plotting
            valid_mask = ~np.isnan(residuals)
            if np.sum(valid_mask) > 0:
                valid_coords = coords[valid_mask]
                valid_residuals = residuals[valid_mask]
                
                # Create scatter plot of residuals
                scatter = axes[i].scatter(valid_coords[:, 0], valid_coords[:, 1], 
                                        c=valid_residuals, cmap='RdBu_r', 
                                        s=20, alpha=0.7)
                axes[i].set_title(f'{model_name} Residuals')
                axes[i].set_xlabel('X Coordinate')
                axes[i].set_ylabel('Y Coordinate')
                plt.colorbar(scatter, ax=axes[i])
            else:
                axes[i].text(0.5, 0.5, 'No Valid Residuals', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{model_name} Residuals')
        else:
            axes[i].text(0.5, 0.5, 'Model Not Found', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{model_name} Residuals')
    
    # Hide unused subplots
    for i in range(len(model_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/spatial_residual_maps.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/spatial_residual_maps.pdf", bbox_inches='tight')
    
    plt.show()