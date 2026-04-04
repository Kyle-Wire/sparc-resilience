#!/usr/bin/env python3
"""
Pipeline Configuration Generator for SPARC
Reads Stage 1 variogram results and creates optimal configuration for Spatial CV
"""

import os
import sys
import json
import numpy as np
from sparc.run.pipeline_paths import get_paths

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from sparc.config.config import load_config

class PipelineConfigurator:
    """
    Generates pipeline configuration with manual parameter specification
    """
    
    def __init__(self, stage1_dir=None, correlogram_dir=None):
        if correlogram_dir is not None:
            self.stage1_dir = correlogram_dir
        elif stage1_dir is not None:
            self.stage1_dir = stage1_dir
        else:
            paths = get_paths()
            self.stage1_dir = paths.stage0_dir
        self.base_config = load_config()
        
        # Default parameters that can be manually overridden
        self.default_variables = ['Elevation_m', 'Distance_from_water_m', 'Pct_Canopy', 'Pct_Impervious', 'Pct_GreenSpace']
        self.default_bandwidths = {
            'Elevation_m': 2000.0,
            'Distance_from_water_m': 1500.0,
            'Pct_Canopy': 1800.0,
            'Pct_Impervious': 2200.0,
            'Pct_GreenSpace': 1600.0
        }

        # Block size: prefer user override from project.yml, else default
        spatial_cv_cfg = self.base_config.get('models', {}).get('spatial_cv', {})
        user_block = spatial_cv_cfg.get('block_size')
        block_src = spatial_cv_cfg.get('block_size_source', 'correlogram')
        if block_src == 'user' and user_block is not None and user_block > 0:
            self.default_block_size = float(user_block)
        else:
            self.default_block_size = 3000.0  # meters

        self.default_kernel = 'gaussian'
        
        # Comprehensive model hyperparameters
        self._initialize_model_defaults()
        # Comprehensive model hyperparameters
        self._initialize_model_defaults()
        
    def _initialize_model_defaults(self):
        """Initialize default hyperparameters for all models"""
        
        # OLS Model (Ordinary Least Squares)
        self.ols_defaults = {
            'fit_intercept': True,
            'normalize': False,  # deprecated in sklearn, but kept for compatibility
            'copy_X': True,
            'n_jobs': None,
            'positive': False  # Force coefficients to be positive
        }
        
        # GWR Model (Geographically Weighted Regression)
        self.gwr_defaults = {
            'bandwidth': None,  # Will be set from default_bandwidths
            'kernel': 'gaussian',  # 'gaussian', 'exponential', 'bisquare'
            'coords_cols': None,  # Coordinate column names
            'alpha': 0.1,  # L2 regularization strength
            'min_points': 50  # Minimum number of points for local regression
        }
        
        # GWRF Model (Geographically Weighted Random Forest)
        self.gwrf_defaults = {
            'n_estimators': 100,  # Number of trees in each local forest
            'k_neighbors': 100,  # Number of neighbors for spatial weighting
            'min_samples_leaf': 5,  # Minimum samples at leaf node
            'n_jobs': 1,  # Number of jobs (1 for compatibility with spatial CV)
            'subsample_fraction': None,  # Fraction of locations to fit models at
            'subsample_n': None  # Number of locations to fit models at
        }
        
        # GGPGAM Model (Geographically Guided Penalized GAM)
        self.ggpgam_defaults = {
            'n_splines': 5,  # Number of splines for non-spatial features
            'n_spatial_bases': 10,  # Number of basis functions for spatial effects
            'lam': 0.6,  # Smoothing parameter (regularization strength)
            'fit_intercept': True,  # Include intercept term
            'max_iter': 100,  # Maximum iterations for fitting
            'tol': 1e-4,  # Tolerance for convergence
            'verbose': False,  # Print fitting progress
            'callbacks': None,  # Callback functions during fitting
            'distribution': 'normal',  # Distribution family for GAM
            'link': 'identity',  # Link function
            'dtype': 'float64',  # Data type for computations
            'scale': None,  # Scale parameter (if None, estimated)
            'fit_linear': False,  # Include linear terms alongside splines
            'constraints': None,  # Constraints on coefficients
            'weights': None  # Sample weights
        }
        
        # Meta Ensemble hyperparameters
        self.meta_ensemble_defaults = {
            'algorithm': 'lightgbm',  # Primary algorithm
            'alternative_algorithms': ['xgboost', 'catboost', 'linear'],
            'n_optuna_trials': 100,  # Number of Optuna trials for hyperparameter optimization
            'include_base_features': True,  # Include original features
            'include_laplacian_pca': True,  # Include Laplacian PCA features
            'use_spatial_cv': True,  # Use spatial CV for meta-model training
            'cv_folds': 5,  # Number of CV folds for meta-model
            'early_stopping': True,  # Enable early stopping
            'patience': 20,  # Early stopping patience
            'validation_split': 0.2,  # Validation split for final training
            'regularization': 'l2',  # Regularization type for linear meta-learner
            'meta_learner_params': {
                'lightgbm': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                },
                'xgboost': {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'catboost': {
                    'objective': 'RMSE',
                    'learning_rate': 0.05,
                    'depth': 6,
                    'random_state': 42,
                    'verbose': False
                }
            }
        }
        
        # Deep Kriging hyperparameters  
        self.deep_kriging_defaults = {
            'hidden_layers': [64, 32, 16],  # Architecture of hidden layers
            'activation': 'relu',  # Activation function
            'dropout_rate': 0.2,  # Dropout rate
            'batch_size': 256,  # Batch size for training
            'epochs': 100,  # Number of training epochs
            'learning_rate': 0.001,  # Initial learning rate
            'optimizer': 'adam',  # Optimizer type
            'loss': 'mse',  # Loss function
            'metrics': ['mae'],  # Additional metrics to track
            'use_residuals': True,  # Use residuals from base models
            'use_laplacian_features': True,  # Include Laplacian eigenmap features
            'validation_split': 0.2,  # Validation split
            'shuffle': True,  # Shuffle training data
            'random_state': 42,  # Random state
            'early_stopping': {
                'enabled': True,
                'patience': 15,  # Stop if no improvement for N epochs
                'min_delta': 1e-4,  # Minimum change to qualify as improvement
                'restore_best_weights': True,
                'monitor': 'val_loss'  # Metric to monitor
            },
            'regularization': {
                'l1_factor': 0.0,  # L1 regularization
                'l2_factor': 1e-4,  # L2 regularization
                'dropout_schedule': 'constant'  # 'constant', 'decay', 'adaptive'
            },
            'adaptive_config': {
                'reduce_lr_on_plateau': True,
                'lr_reduction_factor': 0.5,
                'lr_patience': 10,
                'min_lr': 1e-7
            },
            'callbacks': {
                'tensorboard': False,  # Enable TensorBoard logging
                'model_checkpoint': True,  # Save best model
                'csv_logger': True  # Log metrics to CSV
            }
        }
        
        # Spatial CV hyperparameters
        self.spatial_cv_defaults = {
            'n_folds': 5,  # Number of spatial folds
            'method': 'spatial_blocking',  # CV method
            'block_size': None,  # Will be set from default_block_size
            'buffer_size': 300,  # Buffer distance in meters
            'stratify_y': True,  # Stratify by target distribution
            'buffer_size_auto': True,  # Auto-calculate buffer based on block size
            'min_test_size': 50,  # Minimum test set size per fold
            'spatial_autocorr_threshold': 0.1,  # Threshold for spatial independence
            'overlap_tolerance': 0.05,  # Allowable overlap between train/test
            'clustering_method': 'kmeans',  # Alternative spatial clustering method
            'random_state': 42  # Random state for reproducible folds
        }

        
    def set_manual_parameters(self, variables=None, bandwidths=None, block_size=None, kernel=None):
        """
        Set manual parameters for pipeline configuration
        
        Parameters:
        -----------
        variables : list, optional
            List of variable names to include in the configuration
        bandwidths : dict, optional
            Dictionary mapping variable names to bandwidth values
        block_size : float, optional
            Block size for spatial cross-validation
        kernel : str, optional
            Kernel type to use for spatial models
        """
        if variables is not None:
            self.default_variables = variables
        if bandwidths is not None:
            self.default_bandwidths.update(bandwidths)
        if block_size is not None:
            self.default_block_size = block_size
        if kernel is not None:
            self.default_kernel = kernel
            
        print(f"Updated configuration with:")
        print(f"  Variables: {self.default_variables}")
        print(f"  Bandwidths: {self.default_bandwidths}")
        print(f"  Block size: {self.default_block_size}m")
        print(f"  Kernel: {self.default_kernel}")
        
    def set_model_hyperparameters(self, model_name, **hyperparameters):
        """
        Set hyperparameters for specific models
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('ols', 'gwr', 'gwrf', 'ggpgam', 'meta_ensemble', 'deep_kriging', 'spatial_cv')
        **hyperparameters : dict
            Hyperparameters to update for the specified model
        """
        model_defaults_map = {
            'ols': 'ols_defaults',
            'gwr': 'gwr_defaults', 
            'gwrf': 'gwrf_defaults',
            'ggpgam': 'ggpgam_defaults',
            'meta_ensemble': 'meta_ensemble_defaults',
            'deep_kriging': 'deep_kriging_defaults',
            'spatial_cv': 'spatial_cv_defaults'
        }
        
        if model_name not in model_defaults_map:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_defaults_map.keys())}")
        
        defaults_attr = model_defaults_map[model_name]
        current_defaults = getattr(self, defaults_attr)
        
        # Update nested parameters if provided
        for key, value in hyperparameters.items():
            if isinstance(value, dict) and key in current_defaults and isinstance(current_defaults[key], dict):
                current_defaults[key].update(value)
            else:
                current_defaults[key] = value
        
        print(f"Updated {model_name} hyperparameters:")
        for key, value in hyperparameters.items():
            print(f"  {key}: {value}")
    
    def get_model_hyperparameters(self, model_name):
        """
        Get current hyperparameters for a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Current hyperparameters for the model
        """
        model_defaults_map = {
            'ols': self.ols_defaults,
            'gwr': self.gwr_defaults, 
            'gwrf': self.gwrf_defaults,
            'ggpgam': self.ggpgam_defaults,
            'meta_ensemble': self.meta_ensemble_defaults,
            'deep_kriging': self.deep_kriging_defaults,
            'spatial_cv': self.spatial_cv_defaults
        }
        
        if model_name not in model_defaults_map:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_defaults_map.keys())}")
        
        return model_defaults_map[model_name].copy()
    
    def load_correlogram_results(self):
        """
        Load real correlogram analysis results from Stage 1 output.

        Returns
        -------
        dict | None
            Parsed ``correlogram_analysis_results.json``, or *None* if the
            file does not exist yet (Stage 1 has not been run).
        """
        results_path = os.path.join(self.stage1_dir, 'correlogram_analysis_results.json')
        if not os.path.exists(results_path):
            return None
        with open(results_path, 'r') as f:
            return json.load(f)

    def _correlogram_to_bandwidths(self, corr_results):
        """
        Extract per-variable bandwidths from correlogram results.

        Returns a dict  ``{variable: bandwidth_m}``.
        The target variable is excluded — it is not a predictor.
        """
        individual = corr_results.get('individual_results', {})
        target_var = self.base_config.get('variables', {}).get('target', '')
        bandwidths = {}
        for var, res in individual.items():
            if var == target_var:
                continue  # target is not a predictor
            bw = res.get('optimal_bandwidth')
            if bw is not None and bw > 0:
                bandwidths[var] = float(bw)
        return bandwidths

    def create_mock_variogram_results(self):
        """
        Create mock variogram results using manual parameters.
        Used only when Stage 1 correlogram has not been run.
        """
        mock_results = {}
        
        for variable in self.default_variables:
            bandwidth = self.default_bandwidths.get(variable, 2000.0)
            mock_results[variable] = {
                'optimal_parameters': {
                    'optimal_bandwidth': bandwidth,
                    'kernel': self.default_kernel,
                    'range': bandwidth * 0.8,  # Typical range is slightly less than bandwidth
                    'sill': 1.0,  # Normalized variance
                    'nugget': 0.1  # Small nugget effect
                }
            }
        
        return mock_results
    
    def generate_model_configs(self, mock_results=None):
        """
        Generate global model configurations with comprehensive hyperparameters
        """        
        model_configs = {
            'gwr': dict(self.gwr_defaults),
            'gwrf': dict(self.gwrf_defaults),
            'ggpgam': dict(self.ggpgam_defaults),
            'meta_ensemble': dict(self.meta_ensemble_defaults),
            'spatial_cv': dict(self.spatial_cv_defaults)
        }
        
        # Set block size from defaults
        model_configs['spatial_cv']['block_size'] = self.default_block_size
        
        # Use average bandwidth for GWR
        avg_bandwidth = int(sum(self.default_bandwidths.values()) / len(self.default_bandwidths))
        model_configs['gwr']['bandwidth'] = avg_bandwidth
        
        return model_configs
    

    def generate_deep_kriging_config(self, mock_results=None):
        """
        Generate Deep Kriging configuration using defaults with adaptive adjustments
        """
        if mock_results is None:
            mock_results = self.create_mock_variogram_results()
            
        # Start with defaults
        deep_kriging_config = dict(self.deep_kriging_defaults)
        
        # Adaptive adjustments based on spatial complexity
        if mock_results:
            ranges = [result['optimal_parameters']['range'] 
                     for result in mock_results.values() 
                     if 'optimal_parameters' in result]
            
            n_kernels = len(set(result['optimal_parameters']['kernel'] 
                               for result in mock_results.values() 
                               if 'optimal_parameters' in result))
            
            avg_range = np.mean(ranges) if ranges else 1000.0
            
            # Adapt network complexity based on spatial complexity
            if n_kernels > 2:
                deep_kriging_config['hidden_layers'] = [128, 64, 32, 16]
                deep_kriging_config['epochs'] = 150
            elif avg_range > 2000:
                deep_kriging_config['hidden_layers'] = [128, 64, 32]
                deep_kriging_config['epochs'] = 120
            else:
                deep_kriging_config['hidden_layers'] = [64, 32]
                deep_kriging_config['epochs'] = 100
        
        return deep_kriging_config
    
    def apply_profiler_recommendations(self, recommendations):
        """
        Merge DatasetProfiler recommendations into model defaults.

        Parameters
        ----------
        recommendations : dict
            Output of ``DatasetProfiler.recommend_parameters()``.
        """
        for model_key, rec in recommendations.items():
            if model_key == 'correlogram':
                continue  # handled by correlogram_analysis itself
            try:
                self.set_model_hyperparameters(model_key, **rec)
            except ValueError:
                pass  # model not in defaults map (e.g. 'correlogram')
        print("Applied DatasetProfiler recommendations to model defaults.")

    def create_complete_config(self):
        """
        Create complete pipeline configuration with comprehensive hyperparameters.

        Prefers real correlogram results if available; falls back to mock data.
        """
        # Try loading real correlogram results from Stage 1
        corr_results = self.load_correlogram_results()
        if corr_results is not None:
            real_bw = self._correlogram_to_bandwidths(corr_results)
            if real_bw:
                self.default_bandwidths = real_bw
                self.default_variables = list(real_bw.keys())
                # Also update block size if correlogram determined one
                cv_cfg = corr_results.get('spatial_cv_configuration', {})
                opt_block = cv_cfg.get('optimal_block_size')
                if opt_block:
                    self.default_block_size = float(opt_block)
                print(f"Using REAL correlogram bandwidths for {len(real_bw)} variables.")

                # Respect user override for block size if specified in project.yml
                spatial_cv_cfg = self.base_config.get('models', {}).get('spatial_cv', {})
                block_size_source = spatial_cv_cfg.get('block_size_source', 'correlogram')
                user_block_size = spatial_cv_cfg.get('block_size')
                if block_size_source == 'user' and user_block_size is not None and user_block_size > 0:
                    self.default_block_size = float(user_block_size)
                    print(f"Block size overridden by user: {self.default_block_size:.0f}m")
            else:
                print("Correlogram results found but no per-variable bandwidths — using defaults.")
        else:
            print("No correlogram results found — using default/mock bandwidths.")

        print("Creating configuration with comprehensive hyperparameters...")
        mock_results = self.create_mock_variogram_results()
        
        print("Generating model configurations...")
        model_configs = self.generate_model_configs(mock_results)
        
        print("Generating Deep Kriging configuration...")
        deep_kriging_config = self.generate_deep_kriging_config(mock_results)
        
        # Combine all configurations
        complete_config = {
            'metadata': {
                'created_from': 'Global_Hyperparameter_Configuration',
                'base_config': 'config.py',
                'description': 'Pipeline configuration with global hyperparameters for all models (not per-variable)',
                'configurable_models': ['gwr', 'gwrf', 'ggpgam', 'meta_ensemble', 'deep_kriging', 'spatial_cv'],
                'approach': 'global_parameters'  # Single set of params per model type
            },
            'features': {
                'selected_features': self.default_variables,
                'manually_configured': True
            },
            'manual_parameters': {
                'bandwidths': self.default_bandwidths,
                'block_size': self.default_block_size,
                'kernel': self.default_kernel
            },
            'models': model_configs,
            'deep_kriging': deep_kriging_config,
            'stage_outputs': {
                'stage_1': 'Comprehensive_Hyperparameter_Configuration',
                'stage_2': 'Stage_2_Spatial_CV',
                'final_output': 'Final_Interpretation_Results'
            }
        }
        
        return complete_config
    
    def save_pipeline_config(self, output_path=None):
        """
        Save the complete pipeline configuration
        """
        if output_path is None:
            paths = get_paths()
            output_path = paths.pipeline_config
        
        config = self.create_complete_config()
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Pipeline configuration saved to: {output_path}")
        
        # Also save a human-readable summary
        summary_path = str(output_path).replace('.json', '_summary.txt')
        self.create_config_summary(config, summary_path)
        
        return config
    
    def create_config_summary(self, config, summary_path):
        """
        Create a human-readable summary of the configuration
        """
        with open(summary_path, 'w') as f:
            f.write("SPARC PIPELINE CONFIGURATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write("CONFIGURATION TYPE: Global Hyperparameter Configuration\n")
            f.write("APPROACH: Single set of parameters per model type (not per-variable)\n\n")
            
            f.write("SELECTED FEATURES:\n")
            for feature in config['features']['selected_features']:
                f.write(f"  - {feature}\n")
            f.write("\n")
            
            f.write("MANUAL PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Block Size: {config['manual_parameters']['block_size']}m\n")
            f.write(f"Kernel Type: {config['manual_parameters']['kernel']}\n")
            f.write("Bandwidths:\n")
            for var, bw in config['manual_parameters']['bandwidths'].items():
                f.write(f"  {var}: {bw}m\n")
            f.write("\n")
            
            f.write("MODEL-SPECIFIC CONFIGURATIONS:\n")
            f.write("-" * 30 + "\n")
            
            # Check if models section exists
            if config.get('models') is None:
                f.write("No model configurations found.\n")
                return
            
            models = config['models']
            
            # OLS - baseline model (no hyperparameters)
            f.write("OLS: Baseline model (no configurable hyperparameters)\n\n")
            
            # GWR configurations
            if 'gwr' in models:
                f.write("GWR Parameters:\n")
                gwr_params = config['models']['gwr']
                for key, value in gwr_params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # GWRF configurations  
            if 'gwrf' in config['models']:
                f.write("GWRF Parameters:\n")
                gwrf_params = config['models']['gwrf']
                for key, value in gwrf_params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
            # GGPGAM configurations  
            if 'ggpgam' in config['models']:
                f.write("GGPGAM Parameters:\n")
                ggpgam_params = config['models']['ggpgam']
                for key, value in ggpgam_params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("SPATIAL CROSS-VALIDATION:\n")
            cv = config['models']['spatial_cv']
            f.write(f"  Block Size: {cv['block_size']}m\n")
            f.write(f"  Number of Folds: {cv['n_folds']}\n")
            f.write(f"  Method: {cv['method']}\n")
            f.write("\n")
            
            f.write("DEEP KRIGING:\n")
            dk = config['deep_kriging']
            f.write(f"  Hidden Layers: {dk['hidden_layers']}\n")
            f.write(f"  Dropout Rate: {dk['dropout_rate']}\n")
            f.write(f"  Epochs: {dk['epochs']}\n")
            f.write(f"  Early Stopping: {dk['early_stopping']['enabled']} (patience: {dk['early_stopping']['patience']})\n")
            f.write(f"  L2 Regularization: {dk['regularization']['l2_factor']}\n")
            f.write("\n")
            
            f.write("META-ENSEMBLE OPTIONS:\n")
            meta = config['models']['meta_ensemble']
            f.write(f"  Primary Algorithm: {meta.get('algorithm', 'lightgbm')}\n")
            f.write(f"  Alternative Algorithms: {', '.join(meta.get('alternative_algorithms', ['xgboost', 'catboost']))}\n")
            f.write(f"  Include Base Features: {meta.get('include_base_features', True)}\n")
            f.write(f"  Include Laplacian PCA: {meta.get('include_laplacian_pca', True)}\n")
            f.write(f"  Use Spatial CV: {meta.get('use_spatial_cv', True)}\n")
            f.write(f"  CV Folds: {meta.get('cv_folds', 5)}\n")
            f.write(f"  Early Stopping: {meta.get('early_stopping', True)}\n")
            f.write(f"  Patience: {meta.get('patience', 20)}\n")
            f.write("\n")
            
            f.write("NOTES:\n")
            f.write("- This configuration uses global hyperparameters (not per-variable)\n")
            f.write("- Each model type has a single set of parameters applied globally\n")
            f.write("- Parameters can be edited directly in the JSON configuration file\n")
            f.write("- To modify parameters, edit the 'models' section for each model type\n")
            f.write("- Spatial models use average bandwidth across all variables\n")
            f.write("- Use configurator.set_model_hyperparameters() to update programmatically\n")

    def interactive_hyperparameter_setup(self):
        """
        Interactive method to set hyperparameters for each model
        """
        print("\n=== Interactive Hyperparameter Configuration ===")
        print("Configure hyperparameters for each model. Press Enter to keep current values.\n")
        
        models = ['gwr', 'gwrf', 'ggpgam', 'deep_kriging', 'spatial_cv']
        
        for model_name in models:
            print(f"\n--- {model_name.upper()} Configuration ---")
            current_params = self.get_model_hyperparameters(model_name)
            new_params = {}
            
            if model_name == 'gwr':
                # GWR specific parameters
                params_to_configure = [
                    ('kernel', 'str', 'Kernel type (gaussian, exponential, bisquare)'),
                    ('fixed', 'bool', 'Use fixed bandwidth (True/False)'),
                    ('alpha', 'float', 'L2 regularization strength'),
                    ('min_points', 'int', 'Minimum points for local regression'),
                    ('criterion', 'str', 'Bandwidth selection criterion (AIC, AICc, BIC, CV)'),
                    ('max_iter', 'int', 'Maximum iterations for bandwidth search')
                ]
                
            elif model_name == 'gwrf':
                # GWRF specific parameters
                params_to_configure = [
                    ('n_estimators', 'int', 'Number of trees in each local forest'),
                    ('k_neighbors', 'int', 'Number of neighbors for spatial weighting'),
                    ('min_samples_leaf', 'int', 'Minimum samples at leaf node'),
                    ('max_depth', 'int', 'Maximum depth of trees (or None)'),
                    ('max_features', 'str', 'Features per split (sqrt, log2, or number)'),
                    ('bootstrap', 'bool', 'Use bootstrap sampling'),
                    ('random_state', 'int', 'Random seed')
                ]
                
            elif model_name == 'ggpgam':
                # GGPGAM specific parameters  
                params_to_configure = [
                    ('n_splines', 'int', 'Number of splines for features'),
                    ('n_spatial_bases', 'int', 'Number of spatial basis functions'),
                    ('lam', 'float', 'Smoothing parameter (regularization)'),
                    ('max_iter', 'int', 'Maximum fitting iterations'),
                    ('tol', 'float', 'Convergence tolerance')
                ]
                
            elif model_name == 'deep_kriging':
                # Deep Kriging specific parameters
                params_to_configure = [
                    ('epochs', 'int', 'Number of training epochs'),
                    ('batch_size', 'int', 'Batch size for training'),
                    ('learning_rate', 'float', 'Initial learning rate'),
                    ('dropout_rate', 'float', 'Dropout rate'),
                    ('validation_split', 'float', 'Validation data fraction')
                ]
                
            elif model_name == 'spatial_cv':
                # Spatial CV parameters
                params_to_configure = [
                    ('n_folds', 'int', 'Number of spatial folds'),
                    ('method', 'str', 'CV method (spatial_blocking, kmeans)'),
                    ('stratify_y', 'bool', 'Stratify by target distribution'),
                    ('min_test_size', 'int', 'Minimum test set size per fold')
                ]
            
            for param_name, param_type, description in params_to_configure:
                current_value = current_params.get(param_name, 'Not set')
                prompt = f"  {param_name} ({description})\n    Current: {current_value}\n    New value: "
                
                try:
                    user_input = input(prompt).strip()
                    if user_input:
                        if param_type == 'int':
                            new_params[param_name] = int(user_input)
                        elif param_type == 'float':
                            new_params[param_name] = float(user_input)
                        elif param_type == 'bool':
                            new_params[param_name] = user_input.lower() in ['true', 't', 'yes', 'y', '1']
                        else:  # str
                            new_params[param_name] = user_input
                except ValueError as e:
                    print(f"    Invalid input: {e}. Keeping current value.")
            
            # Update parameters if any were changed
            if new_params:
                self.set_model_hyperparameters(model_name, **new_params)
                print(f"  Updated {len(new_params)} parameters for {model_name}")
        
        print("\n=== Interactive Configuration Complete ===")
        return True
    
    def save_hyperparameter_template(self, output_path=None):
        """
        Save a template configuration file with all available hyperparameters
        """
        if output_path is None:
            paths = get_paths()
            output_path = os.path.join(os.path.dirname(paths.pipeline_config), 'hyperparameter_template.json')
        
        template = {
            'description': 'Template showing all configurable hyperparameters',
            'instructions': 'Edit the "models" section to adjust hyperparameters for each model type',
            'models': {
                'gwr': dict(self.gwr_defaults),
                'gwrf': dict(self.gwrf_defaults),
                'ggpgam': dict(self.ggpgam_defaults),
                'meta_ensemble': dict(self.meta_ensemble_defaults),
                'spatial_cv': dict(self.spatial_cv_defaults)
            },
            'deep_kriging': dict(self.deep_kriging_defaults)
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Hyperparameter template saved to: {output_path}")
        return output_path

def main():
    """
    Main function to generate pipeline configuration with comprehensive hyperparameters
    """
    print("\n=== SPARC Comprehensive Pipeline Configuration Generator ===\n")
    
    configurator = PipelineConfigurator()
    
    # Display current default parameters
    print("Current default parameters:")
    print(f"  Variables: {configurator.default_variables}")
    print(f"  Bandwidths: {configurator.default_bandwidths}")
    print(f"  Block size: {configurator.default_block_size}m")
    print(f"  Kernel: {configurator.default_kernel}")
    print()
    
    # Show available options
    print("Configuration options:")
    print("1. Generate configuration with current defaults")
    print("2. Interactive hyperparameter setup")
    print("3. Save hyperparameter template")
    print("4. Example: Set specific hyperparameters programmatically")
    print()
    
    choice = input("Choose option (1-4) or press Enter for option 1: ").strip()
    
    if choice == '2':
        # Interactive setup
        configurator.interactive_hyperparameter_setup()
        
    elif choice == '3':
        # Save template
        template_path = configurator.save_hyperparameter_template()
        print(f"\nTemplate saved. You can edit this file and copy parameters to your configuration.")
        return template_path
        
    elif choice == '4':
        # Example programmatic setup
        print("\n--- Example: Programmatic Hyperparameter Setting ---")
        
        # Example: Configure GGPGAM for higher precision
        configurator.set_model_hyperparameters('ggpgam',
            n_splines=12,
            n_spatial_bases=15, 
            lam=0.4,
            max_iter=200
        )
        
        # Example: Configure GWRF for larger datasets
        configurator.set_model_hyperparameters('gwrf',
            n_estimators=200,
            k_neighbors=150,
            min_samples_leaf=10,
            max_depth=None,
            bootstrap=True
        )
        
        # Example: Configure Deep Kriging for more training
        configurator.set_model_hyperparameters('deep_kriging',
            epochs=200,
            batch_size=512,
            learning_rate=0.001,
            dropout_rate=0.3,
            **{'early_stopping': {'patience': 25, 'min_delta': 1e-5}}
        )
        
        print("Example hyperparameters set. You can use configurator.set_model_hyperparameters() in your code.")
        print()
    
    try:
        config = configurator.save_pipeline_config()
        print("\n=== Configuration Generation Complete ===")
        print(f"Generated configuration for {len(config['features']['selected_features'])} features")
        print(f"All model hyperparameters included: {list(config['models'].keys())}")
        print("\nTo customize hyperparameters:")
        print("1. Edit the JSON configuration file directly")
        print("2. Use configurator.set_model_hyperparameters() method")
        print("3. Run interactive setup with option 2")
        print("4. Generate template file to see all available options")
        
        return config
        
    except Exception as e:
        print(f"Error generating configuration: {e}")
        return None

if __name__ == "__main__":
    import numpy as np  # Import needed for calculations
    main()
