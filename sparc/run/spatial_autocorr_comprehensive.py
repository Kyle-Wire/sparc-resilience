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
        
        # Remove any non-finite coordinates (NaN or ±inf); pdist requires finite data
        valid_coords_mask = np.isfinite(self.coords).all(axis=1)
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

    def compute_directional_correlogram(
        self,
        values,
        n_angle_bins: int = 4,
    ):
        """Phase 2: Moran's I as a function of (lag, angle).

        Bins point pairs by both inter-point distance (existing ``n_lags``
        scheme) AND by the angle of their displacement vector mod π
        (directors are unsigned: pair (i,j) and (j,i) share an angle).

        Parameters
        ----------
        values : (n,) array — variable values.
        n_angle_bins : int, default=4
            Angular bins on [0, π).  ``n_angle_bins=4`` → centres
            22.5°, 67.5°, 112.5°, 157.5°.

        Returns
        -------
        dict with keys:

        - ``lag_distances`` : (L,) lag-bin centres (m)
        - ``angle_centers_rad`` : (K,) angle-bin centres (radians, [0, π))
        - ``angle_centers_deg`` : (K,) same in degrees
        - ``morans_i`` : (L, K) Moran's I per (lag, angle) bin
        - ``z_scores``  : (L, K)
        - ``p_values``  : (L, K)
        - ``n_pairs``   : (L, K) integer counts
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] != self.n_samples:
            raise ValueError(
                f"values length {values.shape[0]} != n_samples {self.n_samples}"
            )

        # Pre-compute pair angles mod π (directors).  Use upper-triangular
        # vectors to halve memory; we'll symmetrise into the weight matrix
        # per bin below.
        x = self.coords[:, 0]
        y = self.coords[:, 1]
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        angles = np.arctan2(dy, dx)               # in (-π, π]
        angles = np.mod(angles, np.pi)            # director: [0, π)

        # Angle bin edges
        angle_edges = np.linspace(0.0, np.pi, n_angle_bins + 1)
        angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

        L = self.n_lags
        K = n_angle_bins
        morans = np.zeros((L, K))
        zsc = np.zeros((L, K))
        pv = np.ones((L, K))
        npairs = np.zeros((L, K), dtype=np.int64)

        for li in range(L):
            lag_min = self.lag_distances[li]
            lag_max = self.lag_distances[li + 1]
            in_lag = (self.distances >= lag_min) & (self.distances < lag_max)
            for ki in range(K):
                a_min = angle_edges[ki]
                a_max = angle_edges[ki + 1]
                in_angle = (angles >= a_min) & (angles < a_max)
                W = (in_lag & in_angle).astype(np.float64)
                np.fill_diagonal(W, 0.0)
                # Symmetrise: directors are unsigned, but the upper-tri
                # mask above already gives both (i,j) and (j,i) the same
                # angle, so W is already symmetric.
                count = int(W.sum() / 2)
                npairs[li, ki] = count
                if count == 0:
                    continue
                res = self.calculate_global_morans_i(values, weights_matrix=W)
                morans[li, ki] = res["morans_i"]
                zsc[li, ki] = res["z_score"]
                pv[li, ki] = res["p_value"]

        return {
            "lag_distances": np.array(
                [0.5 * (self.lag_distances[i] + self.lag_distances[i + 1])
                 for i in range(L)]
            ),
            "angle_centers_rad": angle_centers,
            "angle_centers_deg": np.degrees(angle_centers),
            "morans_i": morans,
            "z_scores": zsc,
            "p_values": pv,
            "n_pairs": npairs,
        }


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


# ---------------------------------------------------------------------------
# FFT-based correlogram (module-level — no distance matrix required)
# ---------------------------------------------------------------------------

def fft_correlogram(
    coords: np.ndarray,
    values: np.ndarray,
    max_distance: float,
    n_lags: int,
) -> dict:
    """Compute spatial correlogram via 2-D FFT (Wiener-Khinchin theorem).

    O(n log n) alternative to the O(n²) distance-matrix approach.  Suitable
    for full datasets of 50 k+ points without subsampling.

    The point cloud is rasterized onto a regular grid at approximately
    ``max_distance / (2 * n_lags)`` resolution, the spatial autocorrelation
    field is computed via FFT cross-correlation of the mean-centred gridded
    field, and the result is radially averaged into ``n_lags`` distance bins.

    The zero-padding trick (2× in each dimension) converts the circular FFT
    convolution into a linear one, so wrap-around contamination is zero for
    lags ≤ ``max_distance``.

    Returns
    -------
    Same dict format as ``SpatialAutocorrelationAnalyzer.compute_correlogram``:
    ``lag_distances``, ``morans_i_values``, ``z_scores``, ``p_values``,
    ``optimal_block_size``, ``first_zero_crossing``, ``correlogram_results``.
    """
    from scipy import stats as _stats

    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    coords = coords[valid]
    values = values[valid]

    x, y = coords[:, 0], coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    # Grid resolution: at least 2 grid cells per lag band; never below 5 m.
    # Cap at 2048×2048 (~16 MB for float64) to bound memory.
    cell_size = max(max_distance / (n_lags * 2.0), 5.0)
    nx = min(int(np.ceil(x_span / cell_size)) + 2, 2048)
    ny = min(int(np.ceil(y_span / cell_size)) + 2, 2048)
    dx = x_span / (nx - 1) if nx > 1 else cell_size
    dy = y_span / (ny - 1) if ny > 1 else cell_size

    # --- Rasterize: assign each point to nearest grid cell; average values ---
    ix = np.clip(np.round((x - x_min) / dx).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.round((y - y_min) / dy).astype(np.int64), 0, ny - 1)

    grid_sum = np.zeros((ny, nx), dtype=np.float64)
    grid_cnt = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(grid_sum, (iy, ix), values)
    np.add.at(grid_cnt, (iy, ix), 1.0)

    occupied = grid_cnt > 0
    grid = np.where(occupied, grid_sum / np.where(occupied, grid_cnt, 1.0), np.nan)

    # Mean-centre the field; fill empty cells with 0 (neutral for cross-corr)
    global_mean = float(np.nanmean(grid))
    z = np.where(occupied, grid - global_mean, 0.0)

    # Variance of ALL original point values — used to normalise ACF → [−1, 1]
    variance = float(np.var(values))
    if variance < 1e-12:
        variance = 1e-12

    # --- FFT cross-correlation (Wiener-Khinchin) ---
    # 2× zero-padding in each dimension → linear (not circular) autocorrelation
    pad_y = 1 << int(np.ceil(np.log2(max(2 * ny, 4))))
    pad_x = 1 << int(np.ceil(np.log2(max(2 * nx, 4))))

    F_z = np.fft.rfft2(z, s=(pad_y, pad_x))
    acf_raw = np.fft.irfft2(F_z * np.conj(F_z), s=(pad_y, pad_x))[:ny, :nx]

    # Count valid cell pairs at each offset via FFT of the occupancy mask
    F_m = np.fft.rfft2(occupied.astype(np.float64), s=(pad_y, pad_x))
    pair_counts = np.fft.irfft2(F_m * np.conj(F_m), s=(pad_y, pad_x))[:ny, :nx]
    pair_counts = np.maximum(pair_counts, 1e-3)   # guard /0

    # Normalise: ACF(offset) = raw_xcorr(offset) / (n_pairs(offset) * variance)
    acf = acf_raw / (pair_counts * variance)

    # --- Physical distance of each grid offset (i_off, j_off) ---
    i_off = np.arange(ny, dtype=np.float64)
    j_off = np.arange(nx, dtype=np.float64)
    I_OFF, J_OFF = np.meshgrid(i_off, j_off, indexing='ij')   # (ny, nx)
    dist_grid = np.sqrt((I_OFF * dy) ** 2 + (J_OFF * dx) ** 2)

    dist_flat = dist_grid.ravel()
    acf_flat = acf.ravel()
    cnt_flat = pair_counts.ravel()

    # --- Radial averaging into lag bins ---
    lag_edges = np.linspace(0.0, max_distance, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    correlogram_results = []
    morans_i_values: list[float] = []
    z_scores_out: list[float] = []
    p_values_out: list[float] = []

    for i in range(n_lags):
        in_bin = (dist_flat >= lag_edges[i]) & (dist_flat < lag_edges[i + 1])
        in_bin &= cnt_flat > 0.5

        w = cnt_flat[in_bin]
        # Each offset (di, dj) with di>0 or dj>0 contributes 2 directed pairs
        # (i→j and j→i); offset (0,0) contributes n_occupied self-pairs.
        n_pairs = int(np.round(w.sum() / 2.0))

        if n_pairs < 1 or w.sum() < 1.0:
            acf_lag = 0.0
        else:
            acf_lag = float(np.average(acf_flat[in_bin], weights=w))

        # Significance: under spatial randomness E[ACF] ≈ 0 (h > 0),
        # SE ≈ 1 / sqrt(n_pairs).
        if n_pairs > 2:
            se = 1.0 / np.sqrt(n_pairs)
            z_score = float(acf_lag / se)
            p_value = float(2.0 * (1.0 - _stats.norm.cdf(abs(z_score))))
        else:
            z_score = 0.0
            p_value = 1.0

        significant = abs(z_score) > 1.96
        correlogram_results.append({
            'lag_distance': float(lag_centers[i]),
            'n_pairs': n_pairs,
            'morans_i': float(acf_lag),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': bool(significant),
        })
        morans_i_values.append(float(acf_lag))
        z_scores_out.append(float(z_score))
        p_values_out.append(float(p_value))

    # First non-significant lag → optimal block size
    optimal_block_size = max_distance
    for res in correlogram_results:
        if not res['significant'] and res['n_pairs'] > 0:
            optimal_block_size = res['lag_distance']
            break

    # First zero-crossing → effective spatial range / bandwidth
    first_zero_crossing: float | None = None
    for res in correlogram_results:
        if res['n_pairs'] > 0 and res['morans_i'] <= 0.0:
            first_zero_crossing = res['lag_distance']
            break
    if first_zero_crossing is None:
        first_zero_crossing = max_distance

    return {
        'lag_distances': np.array([r['lag_distance'] for r in correlogram_results]),
        'morans_i_values': np.array(morans_i_values),
        'z_scores': np.array(z_scores_out),
        'p_values': np.array(p_values_out),
        'optimal_block_size': float(optimal_block_size),
        'first_zero_crossing': float(first_zero_crossing),
        'correlogram_results': correlogram_results,
    }

