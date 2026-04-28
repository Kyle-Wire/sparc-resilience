#!/usr/bin/env python3
"""
Memory-Efficient Spatial Autocorrelation Analysis

This script provides memory-efficient Moran's I analysis focused on:
1. Model residuals from current pipeline run
2. Temperature autocorrelation from current dataset
3. Intelligent sampling for large datasets to prevent memory issues
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.neighbors import NearestNeighbors
from scipy import stats
warnings.filterwarnings('ignore')

class MemoryEfficientSpatialAnalyzer:
    """
    Memory-efficient spatial autocorrelation analyzer with intelligent sampling
    """
    
    def __init__(self, coords, max_sample_size=5000, max_distance=1800.0, k_neighbors=8):
        """
        Initialize with memory-efficient settings
        
        Parameters:
        -----------
        coords : array-like, shape (n_samples, 2)
            Spatial coordinates (x, y)
        max_sample_size : int, default=5000
            Maximum sample size for analysis to prevent memory issues
        max_distance : float, default=1800.0
            Maximum distance for spatial analysis
        k_neighbors : int, default=8
            Number of neighbors for KNN weights
        """
        self.coords = np.array(coords)
        self.original_n_samples = len(coords)
        self.max_distance = max_distance
        self.k_neighbors = k_neighbors
        
        # Remove NaN coordinates
        valid_mask = ~np.isnan(self.coords).any(axis=1)
        self.coords = self.coords[valid_mask]
        self.valid_indices = np.where(valid_mask)[0]
        
        print(f"Original dataset: {self.original_n_samples:,} points")
        print(f"Valid coordinates: {len(self.coords):,} points")
        
        # Intelligent sampling for large datasets
        if len(self.coords) > max_sample_size:
            self.use_sampling = True
            self.sample_indices = self._create_spatial_sample(max_sample_size)
            self.coords_sample = self.coords[self.sample_indices]
            print(f"Using spatial sampling: {len(self.coords_sample):,} points for analysis")
        else:
            self.use_sampling = False
            self.coords_sample = self.coords
            self.sample_indices = np.arange(len(self.coords))
            print(f"Using full dataset: {len(self.coords):,} points")
        
        self.n_sample = len(self.coords_sample)
    
    def _create_spatial_sample(self, target_size):
        """
        Create spatially representative sample using grid-based approach
        """
        print("Creating spatially representative sample...")
        
        # Create spatial grid
        n_cells_side = int(np.sqrt(target_size))
        
        # Get bounds
        x_min, y_min = self.coords.min(axis=0)
        x_max, y_max = self.coords.max(axis=0)
        
        # Create grid
        x_edges = np.linspace(x_min, x_max, n_cells_side + 1)
        y_edges = np.linspace(y_min, y_max, n_cells_side + 1)
        
        sample_indices = []
        
        for i in range(n_cells_side):
            for j in range(n_cells_side):
                # Find points in this grid cell
                x_mask = (self.coords[:, 0] >= x_edges[i]) & (self.coords[:, 0] < x_edges[i+1])
                y_mask = (self.coords[:, 1] >= y_edges[j]) & (self.coords[:, 1] < y_edges[j+1])
                cell_mask = x_mask & y_mask
                
                cell_indices = np.where(cell_mask)[0]
                
                if len(cell_indices) > 0:
                    # Sample from this cell (take center point or random sample)
                    if len(cell_indices) == 1:
                        sample_indices.append(cell_indices[0])
                    else:
                        # Take a few points from this cell
                        n_from_cell = min(3, len(cell_indices))
                        selected = np.random.choice(cell_indices, n_from_cell, replace=False)
                        sample_indices.extend(selected)
        
        sample_indices = np.array(sample_indices)
        
        # If we don't have enough, add random samples
        if len(sample_indices) < target_size and len(sample_indices) < len(self.coords):
            remaining_indices = np.setdiff1d(np.arange(len(self.coords)), sample_indices)
            n_additional = min(target_size - len(sample_indices), len(remaining_indices))
            additional = np.random.choice(remaining_indices, n_additional, replace=False)
            sample_indices = np.concatenate([sample_indices, additional])
        
        return sample_indices[:target_size]
    
    def calculate_morans_i_efficient(self, values, variable_name="variable"):
        """
        Calculate Moran's I efficiently with sampling if needed
        
        Parameters:
        -----------
        values : array-like
            Variable values for analysis
        variable_name : str
            Name of variable for reporting
            
        Returns:
        --------
        dict : Moran's I results
        """
        values = np.array(values)
        
        # Handle sampling
        if self.use_sampling:
            # Map values to sample
            if len(values) == self.original_n_samples:
                # Full dataset provided
                values_valid = values[self.valid_indices]
                values_sample = values_valid[self.sample_indices]
            elif len(values) == len(self.coords):
                # Already filtered for valid coordinates
                values_sample = values[self.sample_indices]
            else:
                raise ValueError(f"Values length {len(values)} doesn't match expected sizes")
        else:
            values_sample = values
        
        # Remove NaN values from sample
        valid_mask = ~np.isnan(values_sample)
        if np.sum(valid_mask) < 10:
            return self._create_nan_result(f"Insufficient valid data for {variable_name}")
        
        values_clean = values_sample[valid_mask]
        coords_clean = self.coords_sample[valid_mask]
        n = len(values_clean)
        
        print(f"Analyzing {variable_name}: {n:,} valid points")
        
        # Create KNN weights efficiently
        try:
            weights = self._create_efficient_weights(coords_clean)
        except Exception as e:
            print(f"Warning: Failed to create weights for {variable_name}: {e}")
            return self._create_nan_result(f"Weight creation failed for {variable_name}")
        
        # Calculate Moran's I
        try:
            return self._calculate_morans_i_core(values_clean, weights, variable_name)
        except Exception as e:
            print(f"Warning: Moran's I calculation failed for {variable_name}: {e}")
            return self._create_nan_result(f"Calculation failed for {variable_name}")
    
    def _create_efficient_weights(self, coords):
        """Create weights matrix efficiently"""
        n = len(coords)
        k = min(self.k_neighbors, n - 1)
        
        if k < 1:
            return np.zeros((n, n))
        
        # Use KNN with limited neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Create sparse-like weights
        weights = np.zeros((n, n))
        
        for i in range(n):
            # Skip self (first neighbor)
            for j in range(1, min(k + 1, len(indices[i]))):
                neighbor_idx = indices[i][j]
                if distances[i][j] <= self.max_distance:  # Distance threshold
                    weights[i][neighbor_idx] = 1
        
        return weights
    
    def _calculate_morans_i_core(self, values, weights, variable_name):
        """Core Moran's I calculation"""
        n = len(values)
        
        # Row-standardize weights
        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums[:, np.newaxis]
        
        # Center values
        values_centered = values - np.mean(values)
        
        # Calculate Moran's I
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * values_centered[i] * values_centered[j]
        
        denominator = np.sum(values_centered**2)
        W = np.sum(weights)
        
        if denominator == 0 or W == 0:
            return self._create_nan_result(f"No variation in {variable_name}")
        
        morans_i = (n / W) * (numerator / denominator)
        
        # Statistical testing
        expected_i = -1 / (n - 1)
        variance_i = 1 / (n - 1)  # Simplified variance
        
        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0.0
            p_value = 1.0
        
        # Interpretation
        if abs(z_score) <= 1.96:
            interpretation = "No significant spatial autocorrelation"
        elif z_score > 1.96:
            interpretation = f"Significant positive spatial autocorrelation"
        else:
            interpretation = f"Significant negative spatial autocorrelation"
        
        return {
            'variable': variable_name,
            'morans_i': morans_i,
            'expected_i': expected_i,
            'variance_i': variance_i,
            'z_score': z_score,
            'p_value': p_value,
            'significant': abs(z_score) > 1.96,
            'interpretation': interpretation,
            'sample_size': n,
            'sampling_used': self.use_sampling
        }
    
    def _create_nan_result(self, reason):
        """Create result dict for failed analysis"""
        return {
            'variable': 'unknown',
            'morans_i': np.nan,
            'expected_i': np.nan,
            'variance_i': np.nan,
            'z_score': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': reason,
            'sample_size': 0,
            'sampling_used': self.use_sampling
        }

def analyze_model_residuals_morans_i(oof_predictions, y_true, coords, model_names, output_dir):
    """
    Analyze Moran's I for model residuals efficiently
    
    Parameters:
    -----------
    oof_predictions : array-like, shape (n_samples, n_models)
        Out-of-fold predictions from models
    y_true : array-like
        True target values
    coords : array-like
        Spatial coordinates
    model_names : list
        Model names
    output_dir : str or Path
        Output directory for results
        
    Returns:
    --------
    pandas.DataFrame : Results table
    """
    print("Analyzing spatial autocorrelation in model residuals...")
    
    # Create analyzer
    analyzer = MemoryEfficientSpatialAnalyzer(coords)
    
    results = []
    
    # Analyze each model's residuals
    for i, model_name in enumerate(model_names):
        if i < oof_predictions.shape[1]:
            print(f"\nAnalyzing {model_name} residuals...")
            
            # Calculate residuals
            residuals = y_true - oof_predictions[:, i]
            
            # Analyze Moran's I
            result = analyzer.calculate_morans_i_efficient(residuals, f"{model_name}_residuals")
            result['model'] = model_name
            results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Persist: artifact store is canonical, disk is opt-in.
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
    except Exception:  # noqa: BLE001
        _store = None
    if _store is not None:
        _store.write_table(
            stage="spatial",
            artifact_id="model_residuals_morans_i",
            df=results_df,
            producer="memory_efficient_spatial_analysis.analyze_model_residuals_morans_i",
            consumers=["server:/results/spatial_autocorr", "report:spatial"],
        )

    from sparc.run.disk_policy import disk_writes_enabled
    if disk_writes_enabled():
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "model_residuals_morans_i.csv")
        results_df.to_csv(results_path, index=False)
        # Create summary report (disk-only convenience)
        create_residuals_summary_report(results_df, output_dir)
        print(f"\nModel residuals Moran's I analysis completed!")
        print(f"Results saved to: {results_path}")
    else:
        print("\nModel residuals Moran's I analysis completed (artifacts.db: spatial/model_residuals_morans_i)")

    return results_df

def analyze_temperature_morans_i(data_file, output_dir):
    """
    Analyze Moran's I for temperature from current dataset
    
    Parameters:
    -----------
    data_file : str or Path
        Path to current dataset file
    output_dir : str or Path
        Output directory for results
        
    Returns:
    --------
    dict : Temperature Moran's I results
    """
    print(f"Analyzing temperature spatial autocorrelation from: {data_file}")
    
    try:
        # Load data
        data = pd.read_csv(data_file)
        print(f"Loaded dataset with {len(data):,} points")
        
        # Extract coordinates and temperature
        coords = None
        temperature = None
        
        # Try different coordinate column names
        coord_cols = [('POINT_X', 'POINT_Y'), ('x', 'y'), ('X', 'Y')]
        for x_col, y_col in coord_cols:
            if x_col in data.columns and y_col in data.columns:
                coords = data[[x_col, y_col]].values
                print(f"Found coordinates: {x_col}, {y_col}")
                break
        
        # Try different temperature column names
        temp_cols = ['AAT_z', 'temperature', 'temp', 'Temperature']
        for temp_col in temp_cols:
            if temp_col in data.columns:
                temperature = data[temp_col].values
                print(f"Found temperature: {temp_col}")
                break
        
        if coords is None:
            raise ValueError("No coordinate columns found")
        if temperature is None:
            raise ValueError("No temperature column found")
        
        # Create analyzer
        analyzer = MemoryEfficientSpatialAnalyzer(coords)
        
        # Analyze temperature
        result = analyzer.calculate_morans_i_efficient(temperature, "temperature")

        # Persist: artifact store is canonical, disk is opt-in.
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
        except Exception:  # noqa: BLE001
            _store = None
        if _store is not None:
            _store.write_struct(
                stage="spatial",
                artifact_id="temperature_morans_i",
                payload=result if isinstance(result, dict) else {"value": result},
                producer="memory_efficient_spatial_analysis.analyze_temperature_morans_i",
                consumers=["server:/results/spatial_autocorr", "report:spatial"],
            )

        from sparc.run.disk_policy import disk_writes_enabled
        if disk_writes_enabled():
            os.makedirs(output_dir, exist_ok=True)
            temp_df = pd.DataFrame([result])
            temp_path = os.path.join(output_dir, "temperature_morans_i.csv")
            temp_df.to_csv(temp_path, index=False)
            # Create detailed report
            create_temperature_summary_report(result, output_dir)
            print(f"\nTemperature Moran's I analysis completed!")
            print(f"Results saved to: {temp_path}")
        else:
            print("\nTemperature Moran's I analysis completed (artifacts.db: spatial/temperature_morans_i)")

        return result
        
    except Exception as e:
        print(f"Error analyzing temperature: {e}")
        return None

def create_residuals_summary_report(results_df, output_dir):
    """Create summary report for model residuals"""
    
    report_path = os.path.join(output_dir, "model_residuals_spatial_autocorr_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("MODEL RESIDUALS SPATIAL AUTOCORRELATION ANALYSIS\n")
        f.write("="*55 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*20 + "\n")
        f.write(f"Total models analyzed: {len(results_df)}\n")
        f.write(f"Models with significant spatial autocorrelation: {results_df['significant'].sum()}\n")
        f.write(f"Proportion with significant autocorrelation: {results_df['significant'].mean():.2%}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-"*20 + "\n\n")
        
        # Sort by Moran's I value
        sorted_df = results_df.sort_values('morans_i', ascending=False)
        
        for _, row in sorted_df.iterrows():
            f.write(f"Model: {row['model']}\n")
            f.write(f"  Moran's I: {row['morans_i']:.6f}\n")
            f.write(f"  Z-Score: {row['z_score']:.3f}\n")
            f.write(f"  P-Value: {row['p_value']:.6f}\n")
            f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n")
            f.write(f"  Sample Size: {row['sample_size']:,}\n")
            f.write(f"  Sampling Used: {'Yes' if row['sampling_used'] else 'No'}\n")
            f.write(f"  Interpretation: {row['interpretation']}\n\n")
        
        # Best and worst models
        if len(results_df) > 0:
            best_model = results_df.loc[results_df['morans_i'].abs().idxmin()]
            worst_model = results_df.loc[results_df['morans_i'].abs().idxmax()]
            
            f.write("MODEL PERFORMANCE RANKING (by residual spatial autocorrelation)\n")
            f.write("-"*50 + "\n")
            f.write(f"Best (lowest residual autocorrelation): {best_model['model']} (Moran's I = {best_model['morans_i']:.4f})\n")
            f.write(f"Worst (highest residual autocorrelation): {worst_model['model']} (Moran's I = {worst_model['morans_i']:.4f})\n\n")
            
            f.write("INTERPRETATION\n")
            f.write("-"*15 + "\n")
            f.write("Lower absolute Moran's I values in residuals indicate better spatial model performance.\n")
            f.write("Significant positive autocorrelation suggests the model is missing spatial patterns.\n")
            f.write("Non-significant autocorrelation indicates good spatial model performance.\n")
    
    print(f"Model residuals summary report saved to: {report_path}")

def create_temperature_summary_report(result, output_dir):
    """Create summary report for temperature analysis"""
    
    report_path = os.path.join(output_dir, "temperature_spatial_autocorr_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("TEMPERATURE SPATIAL AUTOCORRELATION ANALYSIS\n")
        f.write("="*45 + "\n\n")
        
        f.write("MORAN'S I ANALYSIS RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"Variable: {result['variable']}\n")
        f.write(f"Moran's I: {result['morans_i']:.6f}\n")
        f.write(f"Expected I: {result['expected_i']:.6f}\n")
        f.write(f"Z-Score: {result['z_score']:.3f}\n")
        f.write(f"P-Value: {result['p_value']:.6f}\n")
        f.write(f"Significant: {'Yes' if result['significant'] else 'No'}\n")
        f.write(f"Sample Size: {result['sample_size']:,}\n")
        f.write(f"Sampling Used: {'Yes' if result['sampling_used'] else 'No'}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*15 + "\n")
        f.write(f"{result['interpretation']}\n\n")
        
        if result['significant']:
            if result['z_score'] > 0:
                f.write("The temperature data shows significant positive spatial autocorrelation,\n")
                f.write("meaning similar temperature values tend to be clustered together in space.\n")
                f.write("This is typical for environmental variables like temperature.\n")
            else:
                f.write("The temperature data shows significant negative spatial autocorrelation,\n")
                f.write("meaning dissimilar temperature values tend to be adjacent in space.\n")
                f.write("This is unusual and may indicate data quality issues.\n")
        else:
            f.write("The temperature data shows no significant spatial autocorrelation,\n")
            f.write("suggesting a random spatial distribution of temperature values.\n")
            f.write("This is unusual for environmental data and may indicate:\n")
            f.write("1. Very fine-scale temperature variation\n")
            f.write("2. Data quality issues\n") 
            f.write("3. Insufficient spatial resolution\n")
    
    print(f"Temperature summary report saved to: {report_path}")

if __name__ == "__main__":
    print("Memory-Efficient Spatial Autocorrelation Analysis")
    print("This module provides efficient Moran's I analysis for large datasets")
    print("Use the functions analyze_model_residuals_morans_i() and analyze_temperature_morans_i()")