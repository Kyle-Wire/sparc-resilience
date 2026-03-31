#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from pathlib import Path
import warnings
from tqdm import tqdm

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import SPARC modules
from config.config import load_config
from data.data_utils import load_and_prepare_data
from features.laplacian import LaplacianEigenmap
from models.gwen import GWENModel
from models.ols import OLSModel
from models.gwr import GWRModel
from models.gwrf import GWRFModel
from models.ggpgam import GGPGAMModel
from models.meta_ensemble import MetaEnsemble
from models.deep_kriging import DeepKriging
from evaluation.evaluation import SpatialEvaluator

class SPARCPipeline:
    """
    Main pipeline runner for the SPARC project. Orchestrates the entire modeling process
    from data loading through final evaluation.
    """
    
    def __init__(self, config_path: str, output_dir: str = None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        output_dir : str, optional
            Base output directory. If None, uses timestamp-based directory.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"sparc_run_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.evaluator = SpatialEvaluator(
            k_neighbors=self.config.get('n_neighbors_moran', 100)
        )
        
        # Initialize model artifacts
        self.data = None
        self.laplacian_features = None
        self.models = {}
        self.predictions = {}
        
        logging.info(f"Pipeline initialized with config from: {config_path}")
        logging.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Configure logging with both file and console handlers"""
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        
        # Configure logging format
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _save_predictions(self, stage_name: str, predictions: np.ndarray):
        """Save predictions from a pipeline stage"""
        if self.data is None:
            logging.warning(f"Cannot save predictions for {stage_name}: No data loaded")
            return
            
        pred_df = pd.DataFrame({
            self.config['identifier_col']: self.data[self.config['identifier_col']],
            f'{stage_name}_pred': predictions
        })
        
        output_path = os.path.join(self.output_dir, f'{stage_name}_predictions.csv')
        pred_df.to_csv(output_path, index=False)
        logging.info(f"Saved {stage_name} predictions to: {output_path}")
        
        self.predictions[stage_name] = predictions
    
    def load_data(self):
        """Load and prepare input data"""
        logging.info("\n=== Stage 1: Data Loading and Preparation ===")
        
        try:
            self.data = load_and_prepare_data(
                self.config['input_data_path'],
                target_col=self.config['target_variable'],
                predictor_cols=self.config['predictor_variables'],
                coord_cols=self.config['coordinate_columns'],
                id_col=self.config['identifier_col']
            )
            
            logging.info(f"Loaded data shape: {self.data.shape}")
            
            # Save prepared data
            output_path = os.path.join(self.output_dir, 'prepared_data.gpkg')
            self.data.to_file(output_path, driver='GPKG')
            logging.info(f"Saved prepared data to: {output_path}")
            
        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise
    
    def generate_features(self):
        """Generate Laplacian eigenmaps"""
        logging.info("\n=== Stage 2: Feature Engineering ===")
        
        try:
            laplacian = LaplacianEigenmap(
                n_components=self.config.get('n_laplacian_components', 150),
                n_neighbors=self.config.get('n_neighbors_laplacian', 10)
            )
            
            coords = self.data[self.config['coordinate_columns']].values
            self.laplacian_features = laplacian.fit_transform(coords)
            
            # Add features to data
            for i in range(self.laplacian_features.shape[1]):
                self.data[f'LAPEIG_{i+1}'] = self.laplacian_features[:, i]
            
            # Save features
            output_path = os.path.join(self.output_dir, 'laplacian_features.csv')
            pd.DataFrame(
                self.laplacian_features,
                columns=[f'LAPEIG_{i+1}' for i in range(self.laplacian_features.shape[1])]
            ).to_csv(output_path, index=False)
            
            logging.info(f"Generated {self.laplacian_features.shape[1]} Laplacian features")
            
        except Exception as e:
            logging.error(f"Error in feature generation: {str(e)}")
            raise
    
    def train_base_models(self):
        """Train all base models"""
        logging.info("\n=== Stage 3: Base Model Training ===")
        
        # Prepare common data
        X = self.data[self.config['predictor_variables']]
        y = self.data[self.config['target_variable']]
        coords = self.data[self.config['coordinate_columns']].values
        
        # Train GWEN for variable selection
        try:
            logging.info("Training GWEN model...")
            gwen = GWENModel(random_state=self.config.get('random_state', 42))
            gwen.fit(X, y, coords)
            self.models['gwen'] = gwen
            
            # Save variable importance
            gwen.save_variable_importance(
                os.path.join(self.output_dir, 'gwen_variable_importance.csv')
            )
        except Exception as e:
            logging.error(f"Error in GWEN training: {str(e)}")
        
        # Train OLS baseline
        try:
            logging.info("Training OLS model...")
            ols = OLSModel(use_laplacian=self.config.get('use_laplacian_in_ols', True))
            ols.fit(X, y, laplacian_features=self.laplacian_features)
            self.models['ols'] = ols
            
            # Save predictions and summary
            ols_preds = ols.predict(X, self.laplacian_features)
            self._save_predictions('ols', ols_preds)
            ols.save_summary(os.path.join(self.output_dir, 'ols_summary.txt'))
        except Exception as e:
            logging.error(f"Error in OLS training: {str(e)}")
        
        # Train GWR
        try:
            logging.info("Training GWR model...")
            gwr = GWRModel(
                adaptive=self.config.get('gwr_adaptive', True),
                kernel=self.config.get('gwr_kernel', 'gaussian')
            )
            gwr.fit(X, y, coords)
            self.models['gwr'] = gwr
            
            # Save predictions and coefficients
            gwr_preds = gwr.predict(X, coords)
            self._save_predictions('gwr', gwr_preds)
            gwr.save_coefficients(
                self.data,
                os.path.join(self.output_dir, 'gwr_coefficients.gpkg')
            )
        except Exception as e:
            logging.error(f"Error in GWR training: {str(e)}")
        
        # Train GWRF
        if self.config.get('include_gwrf', True):
            try:
                logging.info("Training GWRF model...")
                gwrf = GWRFModel(
                    n_estimators=self.config.get('gwrf_n_estimators', 100),
                    min_samples_leaf=self.config.get('gwrf_min_samples_leaf', 10)
                )
                gwrf.fit(X, y, coords)
                self.models['gwrf'] = gwrf
                
                # Save predictions
                gwrf_preds = gwrf.predict(X, coords)
                self._save_predictions('gwrf', gwrf_preds)
            except Exception as e:
                logging.error(f"Error in GWRF training: {str(e)}")
        
        # Train GGPGAM
        try:
            logging.info("Training GGPGAM model...")
            ggpgam = GGPGAMModel()
            ggpgam.fit(X, y, coords)
            self.models['ggpgam'] = ggpgam
            
            # Save predictions and plots
            ggpgam_preds = ggpgam.predict(X, coords)
            self._save_predictions('ggpgam', ggpgam_preds)
            ggpgam.save_diagnostic_plots(os.path.join(self.output_dir, 'ggpgam_plots'))
        except Exception as e:
            logging.error(f"Error in GGPGAM training: {str(e)}")
    
    def train_deep_kriging(self):
        """Train deep kriging model for residual correction"""
        logging.info("\n=== Stage 4: Deep Kriging Residual Correction ===")
        
        try:
            # Load meta-model from spatial_cv_v2 output if available
            meta_model_path = os.path.join(self.output_dir, 'meta_model.txt')
            if os.path.exists(meta_model_path):
                import lightgbm as lgb
                meta_model = lgb.Booster(model_file=meta_model_path)
                logging.info("Loaded meta-model from spatial_cv_v2 output")
                
                # Load OOF predictions to reconstruct meta-features
                oof_path = os.path.join(self.output_dir, 'oof_predictions.csv')
                if os.path.exists(oof_path):
                    oof_preds = pd.read_csv(oof_path).values
                    X = self.data[self.config['predictor_variables']].values
                    X_meta = np.concatenate([oof_preds, X], axis=1)
                    
                    # Generate meta-model predictions
                    y_pred_meta = meta_model.predict(X_meta)
                    residuals = self.data[self.config['target_variable']] - y_pred_meta
                    
                    logging.info("Using meta-model residuals for deep kriging")
                else:
                    logging.warning("OOF predictions not found, using OLS residuals")
                    y_pred_meta = self.predictions.get('ols', np.zeros(len(self.data)))
                    residuals = self.data[self.config['target_variable']] - y_pred_meta
            else:
                logging.warning("Meta-model not found, using OLS residuals for deep kriging")
                y_pred_meta = self.predictions.get('ols', np.zeros(len(self.data)))
                residuals = self.data[self.config['target_variable']] - y_pred_meta
            
            # Initialize and train deep kriging
            dk = DeepKriging(
                hidden_layers=self.config.get('dk_hidden_layers', [128, 64, 32]),
                dropout_rate=self.config.get('dk_dropout_rate', 0.2)
            )
            
            dk.fit(
                self.data[self.config['coordinate_columns']].values,
                residuals,
                self.laplacian_features
            )
            
            self.models['deep_kriging'] = dk
            
            # Generate and save corrections
            dk_corrections = dk.predict(
                self.data[self.config['coordinate_columns']].values,
                self.laplacian_features
            )
            self._save_predictions('deep_kriging', dk_corrections)
            
            # Save final predictions (meta-model + corrections)
            final_predictions = y_pred_meta + dk_corrections
            self._save_predictions('final', final_predictions)
            
            # Save model
            dk.save_model(os.path.join(self.output_dir, 'deep_kriging'))
            
        except Exception as e:
            logging.error(f"Error in deep kriging training: {str(e)}")
            raise
    
    def evaluate_models(self):
        """Evaluate all models and generate reports"""
        logging.info("\n=== Stage 5: Model Evaluation ===")
        
        try:
            y_true = self.data[self.config['target_variable']]
            coords = self.data[self.config['coordinate_columns']].values
            
            # Create evaluation directory
            eval_dir = os.path.join(self.output_dir, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Evaluate each model
            for name, predictions in self.predictions.items():
                model_eval_dir = os.path.join(eval_dir, name)
                
                self.evaluator.create_evaluation_report(
                    y_true=y_true,
                    y_pred=predictions,
                    coords=coords,
                    gdf=self.data,
                    output_dir=model_eval_dir,
                    model_name=name
                )
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        try:
            # Stage 1: Load Data
            self.load_data()
            
            # Stage 2: Generate Features
            self.generate_features()
            
            # Stage 3: Train Base Models
            self.train_base_models()
            
            # Stage 4: Train Deep Kriging
            self.train_deep_kriging()
            
            # Stage 5: Evaluate Models
            self.evaluate_models()
            
            logging.info("\n=== Pipeline Execution Complete ===")
            logging.info(f"Results saved in: {self.output_dir}")
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise

def main():
    """Main entry point for the SPARC pipeline"""
    parser = argparse.ArgumentParser(description='SPARC Pipeline Runner')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Base output directory (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run pipeline
        pipeline = SPARCPipeline(args.config, args.output_dir)
        pipeline.run_pipeline()
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 