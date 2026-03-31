#!/usr/bin/env python3
"""
.. deprecated:: 2.1.0
   This module is **RETIRED**.  Ensemble logic has been consolidated into
   Stage 2 (``run.enhanced_spatial_cv``) and Stage 4
   (``interventions.scenario_simulator``).  This file is kept only for
   reference and will be removed in a future release.

Stage 3 Ensemble Process: Meta Ensemble + Deep Kriging on Residuals
Part of SPARC Pipeline - Uses Stage 2 OOF predictions as input
"""
import warnings as _w
_w.warn(
    "run.stage_3_ensemble is deprecated and no longer called by the "
    "pipeline. Use run.causal_validation (Stage 3) and "
    "interventions.scenario_simulator (Stage 4) instead.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import warnings
import psutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
warnings.filterwarnings('ignore')

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.config import load_config
from data.data_utils import load_and_preprocess_data
from features.laplacian import LaplacianEigenmap
from run.enhanced_spatial_cv import spatial_kfold_enhanced, HARDWARE_CONFIG
from models.meta_ensemble import MetaEnsemble
from models.deep_kriging import DeepKriging

class Stage3EnsembleProcessor:
    """
    Stage 3 Ensemble processor that runs Meta Ensemble and Deep Kriging
    """
    
    def __init__(self, stage2_dir="Stage_2_Spatial_CV"):
        self.stage2_dir = stage2_dir
        self.stage3_dir = "Stage_3_Ensemble"
        self.base_config = load_config()
        
        # Load pipeline config if it exists
        self.pipeline_config = self.load_pipeline_config()
        
        # Create Stage 3 output directory
        os.makedirs(self.stage3_dir, exist_ok=True)
        
        print(f"=== Stage 3 Ensemble Processor Initialized ===")
        print(f"Input directory: {self.stage2_dir}")
        print(f"Output directory: {self.stage3_dir}")
        
    def load_pipeline_config(self):
        """Load pipeline configuration if available"""
        try:
            with open("pipeline_config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("No pipeline_config.json found, using defaults")
            return {
                'features': {
                    'selected_features': ['Elevation_m', 'Distance_from_water_m', 'Pct_Canopy', 'Pct_Impervious', 'Pct_GreenSpace']
                },
                'stage_outputs': {
                    'stage_2': 'Stage_2_Spatial_CV',
                    'stage_3': 'Stage_3_Ensemble'
                }
            }
    
    def load_stage2_data(self):
        """Load Stage 2 OOF predictions and original data"""
        print("=== Loading Stage 2 Data ===")
        
        # Load OOF predictions
        oof_path = os.path.join(self.stage2_dir, "optimized_oof_predictions.csv")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Stage 2 OOF predictions not found at {oof_path}")
            
        oof_predictions = pd.read_csv(oof_path)
        print(f"Loaded OOF predictions: {oof_predictions.shape}")
        print(f"Models: {list(oof_predictions.columns)}")
        
        # Load original data to get target and coordinates
        selected_features = self.pipeline_config['features']['selected_features']
        
        data = load_and_preprocess_data(
            raw_data_path=self.base_config['paths']['raw_csv_path'],
            identifier_col=self.base_config['variables']['identifier'],
            target_col=self.base_config['variables']['target'],
            coords_cols=self.base_config['variables']['coordinates'],
            predictor_cols=selected_features,
            initial_crs=self.base_config['crs']['initial'],
            target_crs=self.base_config['crs']['target_projected']
        )
        
        # Extract target and coordinates
        y = data[self.base_config['variables']['target']].values
        coords = data[self.base_config['variables']['coordinates']].values
        
        # Also get the original features for Laplacian eigenmaps
        available_features = [f for f in selected_features if f in data.columns]
        X_original = data[available_features].values
        
        print(f"Target variable shape: {y.shape}")
        print(f"Coordinates shape: {coords.shape}")
        print(f"Original features shape: {X_original.shape}")
        
        return oof_predictions.values, y, coords, X_original, available_features
    
    def generate_laplacian_features(self, X_original, coords, n_components=150):
        """Generate Laplacian eigenmap features"""
        print(f"=== Generating Laplacian Eigenmap Features ===")
        
        try:
            # Use hardware-optimized batch size
            batch_size = HARDWARE_CONFIG.get('laplacian_batch_size', 2000)
            max_iterations = HARDWARE_CONFIG.get('max_eigen_iterations', 1000)
            
            laplacian = LaplacianEigenmap(
                n_components=n_components,
                k_neighbors=min(50, len(X_original) // 20),
                target_crs=None
            )
            
            # Combine original features with coordinates for Laplacian
            # But LaplacianEigenmap only expects coordinates
            
            print(f"Computing Laplacian eigenmaps for {len(coords)} samples...")
            start_time = time.time()
            
            X_laplacian = laplacian.fit_transform(coords)
            
            computation_time = time.time() - start_time
            print(f"Laplacian computation completed in {computation_time:.2f}s")
            print(f"Generated {X_laplacian.shape[1]} Laplacian features")
            
            # Now combine with original features
            X_combined = np.hstack([X_original, X_laplacian])
            
            return X_combined
            
        except Exception as e:
            print(f"Error generating Laplacian features: {e}")
            print("Using PCA as fallback...")
            
            # Fallback to PCA
            from sklearn.decomposition import PCA
            
            # Use minimum of requested components and available features
            max_components = min(n_components, X_original.shape[1] + coords.shape[1])
            
            if max_components > 0:
                pca = PCA(n_components=max_components)
                X_coords_pca = pca.fit_transform(coords)
                
                # Combine original features with PCA coordinates
                X_combined = np.hstack([X_original, X_coords_pca])
                
                print(f"Generated {X_coords_pca.shape[1]} PCA coordinate features combined with {X_original.shape[1]} original features")
                return X_combined
            else:
                print("No PCA components possible, using original features only")
                return X_original
    
    def run_meta_ensemble_cv(self, X_base_predictions, y, coords, folds):
        """Run Meta Ensemble through spatial cross-validation"""
        print("\n=== Meta Ensemble Spatial Cross-Validation ===")
        
        n_samples = len(y)
        meta_oof_predictions = np.zeros(n_samples)
        meta_model_performances = []
        
        print(f"Running Meta Ensemble on {n_samples} samples with {X_base_predictions.shape[1]} base model predictions")
        
        with tqdm(total=len(folds), desc="Meta Ensemble CV", unit="fold") as pbar:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    # Get fold data
                    X_train = X_base_predictions[train_idx]
                    X_test = X_base_predictions[test_idx]
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    coords_train = coords[train_idx]
                    coords_test = coords[test_idx]
                    
                    # Convert base predictions to dictionary format expected by MetaEnsemble
                    base_models = ['ols', 'gwr', 'gwrf', 'ggpgam']
                    base_predictions_train = {}
                    base_predictions_test = {}
                    
                    for i, model_name in enumerate(base_models):
                        base_predictions_train[model_name] = X_train[:, i]
                        base_predictions_test[model_name] = X_test[:, i]
                    
                    # Create and train meta ensemble
                    meta_model = MetaEnsemble(
                        n_trials=25,
                        val_size=0.2,
                        random_state=42
                    )
                    
                    print(f"  Fold {fold_idx + 1}: Training meta model on {len(train_idx)} samples...")
                    start_time = time.time()
                    
                    # Train meta ensemble
                    meta_model.fit(base_predictions_train, y_train, coords_train)
                    
                    # Predict on test set
                    fold_predictions = meta_model.predict(base_predictions_test, coords_test)
                    meta_oof_predictions[test_idx] = fold_predictions
                    
                    # Calculate fold performance
                    fold_r2 = r2_score(y_test, fold_predictions)
                    fold_rmse = np.sqrt(mean_squared_error(y_test, fold_predictions))
                    
                    meta_model_performances.append({
                        'fold': fold_idx + 1,
                        'r2': fold_r2,
                        'rmse': fold_rmse,
                        'train_size': len(train_idx),
                        'test_size': len(test_idx)
                    })
                    
                    train_time = time.time() - start_time
                    print(f"  Fold {fold_idx + 1} completed in {train_time:.2f}s: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")
                    
                except Exception as e:
                    print(f"  Fold {fold_idx + 1} failed: {e}")
                    # Use mean prediction as fallback
                    meta_oof_predictions[test_idx] = np.mean(y[train_idx])
                    meta_model_performances.append({
                        'fold': fold_idx + 1,
                        'r2': 0.0,
                        'rmse': np.std(y),
                        'train_size': len(train_idx),
                        'test_size': len(test_idx),
                        'error': str(e)
                    })
                
                pbar.update(1)
                pbar.set_postfix(fold=f"{fold_idx+1}/{len(folds)}")
        
        # Calculate overall Meta Ensemble performance
        meta_r2 = r2_score(y, meta_oof_predictions)
        meta_rmse = np.sqrt(mean_squared_error(y, meta_oof_predictions))
        
        print(f"\n=== Meta Ensemble Results ===")
        print(f"Overall Performance: R² = {meta_r2:.4f}, RMSE = {meta_rmse:.4f}")
        
        # Save Meta Ensemble results
        meta_results = {
            'overall_performance': {'r2': meta_r2, 'rmse': meta_rmse},
            'fold_performances': meta_model_performances
        }
        
        with open(os.path.join(self.stage3_dir, 'meta_ensemble_results.json'), 'w') as f:
            json.dump(meta_results, f, indent=2)
        
        # Save Meta Ensemble OOF predictions
        pd.DataFrame({
            'meta_ensemble_predictions': meta_oof_predictions
        }).to_csv(os.path.join(self.stage3_dir, 'meta_ensemble_oof_predictions.csv'), index=False)
        
        return meta_oof_predictions
    
    def run_deep_kriging_cv(self, X_laplacian, meta_residuals, coords, folds):
        """Run Deep Kriging on Meta Ensemble residuals through spatial cross-validation"""
        print("\n=== Deep Kriging on Meta Residuals ===")
        
        n_samples = len(meta_residuals)
        deep_kriging_predictions = np.zeros(n_samples)
        deep_kriging_performances = []
        
        print(f"Running Deep Kriging on {n_samples} residual samples with {X_laplacian.shape[1]} Laplacian features")
        
        with tqdm(total=len(folds), desc="Deep Kriging CV", unit="fold") as pbar:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    # Get fold data
                    X_train = X_laplacian[train_idx]
                    X_test = X_laplacian[test_idx]
                    y_train = meta_residuals[train_idx]
                    y_test = meta_residuals[test_idx]
                    coords_train = coords[train_idx]
                    coords_test = coords[test_idx]
                    
                    # Create and train Deep Kriging model
                    dk_model = DeepKriging(
                        hidden_layers=[64, 32],
                        dropout_rate=0.1,
                        learning_rate=0.001,
                        val_size=0.2,
                        random_state=42,
                        batch_size=min(256, len(train_idx) // 4),
                        epochs=50,
                        patience=5
                    )
                    
                    print(f"  Fold {fold_idx + 1}: Training Deep Kriging on {len(train_idx)} residual samples...")
                    start_time = time.time()
                    
                    # Train Deep Kriging
                    dk_model.fit(coords_train, y_train, X_train, verbose=0)
                    
                    # Predict on test set
                    fold_predictions = dk_model.predict(coords_test, X_test)
                    deep_kriging_predictions[test_idx] = fold_predictions
                    
                    # Calculate fold performance
                    fold_r2 = r2_score(y_test, fold_predictions)
                    fold_rmse = np.sqrt(mean_squared_error(y_test, fold_predictions))
                    
                    deep_kriging_performances.append({
                        'fold': fold_idx + 1,
                        'r2': fold_r2,
                        'rmse': fold_rmse,
                        'train_size': len(train_idx),
                        'test_size': len(test_idx)
                    })
                    
                    train_time = time.time() - start_time
                    print(f"  Fold {fold_idx + 1} completed in {train_time:.2f}s: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")
                    
                except Exception as e:
                    print(f"  Fold {fold_idx + 1} failed: {e}")
                    # Use zero prediction as fallback (no residual correction)
                    deep_kriging_predictions[test_idx] = 0.0
                    deep_kriging_performances.append({
                        'fold': fold_idx + 1,
                        'r2': 0.0,
                        'rmse': np.std(y_test) if len(y_test) > 0 else 0.0,
                        'train_size': len(train_idx),
                        'test_size': len(test_idx),
                        'error': str(e)
                    })
                
                pbar.update(1)
                pbar.set_postfix(fold=f"{fold_idx+1}/{len(folds)}")
        
        # Calculate overall Deep Kriging performance on residuals
        if np.std(meta_residuals) > 0:
            dk_r2 = r2_score(meta_residuals, deep_kriging_predictions)
            dk_rmse = np.sqrt(mean_squared_error(meta_residuals, deep_kriging_predictions))
        else:
            dk_r2 = 0.0
            dk_rmse = 0.0
        
        print(f"\n=== Deep Kriging Results (on residuals) ===")
        print(f"Residual Modeling Performance: R² = {dk_r2:.4f}, RMSE = {dk_rmse:.4f}")
        
        # Save Deep Kriging results
        dk_results = {
            'residual_performance': {'r2': dk_r2, 'rmse': dk_rmse},
            'fold_performances': deep_kriging_performances
        }
        
        with open(os.path.join(self.stage3_dir, 'deep_kriging_results.json'), 'w') as f:
            json.dump(dk_results, f, indent=2)
        
        # Save Deep Kriging predictions
        pd.DataFrame({
            'deep_kriging_residual_predictions': deep_kriging_predictions
        }).to_csv(os.path.join(self.stage3_dir, 'deep_kriging_residual_predictions.csv'), index=False)
        
        return deep_kriging_predictions
    
    def run_complete_ensemble_pipeline(self):
        """Run the complete Stage 3 ensemble pipeline"""
        print("\n=== Stage 3 Ensemble Pipeline ===")
        
        # Load Stage 2 data
        X_base_predictions, y, coords, X_original, feature_names = self.load_stage2_data()
        
        # Generate spatial folds (same as Stage 2 for consistency)
        print("\n=== Generating Spatial Folds ===")
        folds = spatial_kfold_enhanced(
            X=X_base_predictions,
            y=y,
            coords=coords,
            n_splits=5,
            block_size=None,
            method='block',
            stratify_y=True
        )
        
        # Step 1: Run Meta Ensemble on base model predictions
        meta_oof_predictions = self.run_meta_ensemble_cv(X_base_predictions, y, coords, folds)
        
        # Step 2: Calculate Meta Ensemble residuals
        meta_residuals = y - meta_oof_predictions
        print(f"\nMeta Ensemble residuals - Mean: {np.mean(meta_residuals):.4f}, Std: {np.std(meta_residuals):.4f}")
        
        # Step 3: Generate Laplacian eigenmap features
        X_laplacian = self.generate_laplacian_features(X_original, coords, n_components=150)
        
        # Step 4: Run Deep Kriging on Meta Ensemble residuals
        deep_kriging_residual_predictions = self.run_deep_kriging_cv(X_laplacian, meta_residuals, coords, folds)
        
        # Step 5: Combine Meta Ensemble + Deep Kriging predictions
        # Only add Deep Kriging if it actually improves performance
        final_predictions = meta_oof_predictions + deep_kriging_residual_predictions
        
        # Calculate final ensemble performance
        final_r2 = r2_score(y, final_predictions)
        final_rmse = np.sqrt(mean_squared_error(y, final_predictions))
        
        # Check if Deep Kriging actually helped
        meta_r2 = r2_score(y, meta_oof_predictions)
        meta_rmse = np.sqrt(mean_squared_error(y, meta_oof_predictions))
        
        # If Deep Kriging made things worse, use just the Meta Ensemble
        if final_r2 < meta_r2:
            print(f"\nWARNING: Deep Kriging degraded performance (R²: {meta_r2:.4f} → {final_r2:.4f})")
            print("Using Meta Ensemble only for final predictions")
            final_predictions = meta_oof_predictions
            final_r2 = meta_r2
            final_rmse = meta_rmse
            deep_kriging_residual_predictions = np.zeros_like(deep_kriging_residual_predictions)
        
        print(f"\n=== Final Ensemble Results ===")
        print(f"Meta Ensemble: R² = {meta_r2:.4f}, RMSE = {meta_rmse:.4f}")
        print(f"Final Ensemble (Meta + Deep Kriging): R² = {final_r2:.4f}, RMSE = {final_rmse:.4f}")
        
        # Calculate improvement
        improvement = final_r2 - meta_r2
        print(f"Deep Kriging improvement: {improvement:+.4f} R² points")
        
        # Save final results
        final_results = {
            'meta_ensemble_performance': {
                'r2': r2_score(y, meta_oof_predictions),
                'rmse': np.sqrt(mean_squared_error(y, meta_oof_predictions))
            },
            'final_ensemble_performance': {
                'r2': final_r2,
                'rmse': final_rmse
            },
            'feature_info': {
                'original_features': feature_names,
                'laplacian_components': X_laplacian.shape[1],
                'base_models': ['ols', 'gwr', 'gwrf', 'ggpgam']
            }
        }
        
        with open(os.path.join(self.stage3_dir, 'final_ensemble_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save final predictions
        pd.DataFrame({
            'meta_ensemble_predictions': meta_oof_predictions,
            'deep_kriging_residual_predictions': deep_kriging_residual_predictions,
            'final_ensemble_predictions': final_predictions,
            'actual_values': y
        }).to_csv(os.path.join(self.stage3_dir, 'final_ensemble_predictions.csv'), index=False)
        
        print(f"\n=== Stage 3 Complete ===")
        print(f"Results saved to: {self.stage3_dir}/")
        
        return {
            'meta_predictions': meta_oof_predictions,
            'deep_kriging_residuals': deep_kriging_residual_predictions,
            'final_predictions': final_predictions,
            'performance': final_results
        }

def main():
    """Main function to run Stage 3 ensemble pipeline"""
    try:
        # Initialize Stage 3 processor
        processor = Stage3EnsembleProcessor()
        
        # Run complete ensemble pipeline
        results = processor.run_complete_ensemble_pipeline()
        
        print("\n=== Stage 3 Pipeline Completed Successfully ===")
        print(f"Final ensemble performance: R² = {results['performance']['final_ensemble_performance']['r2']:.4f}")
        print(f"Final ensemble RMSE: {results['performance']['final_ensemble_performance']['rmse']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error in Stage 3 pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
