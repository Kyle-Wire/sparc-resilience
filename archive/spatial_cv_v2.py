#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import lightgbm as lgb
import joblib
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Spatial analysis imports
from libpysal.weights import DistanceBand
from esda.moran import Moran

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.config import load_config
from data.data_utils import load_and_preprocess_data
from features.laplacian import LaplacianEigenmap
from models.ols import OLSModel
from models.gwr import GWRModel
from models.deep_kriging import DeepKriging

from models.gwrf import GWRFModel
from models.ggpgam import GGPGAM_SVC

def check_files_exist(files, directory):
    """Check if all files in a list exist in a directory."""
    return all(os.path.exists(os.path.join(directory, f)) for f in files)

def estimate_spatial_autocorrelation_range(coords, y, max_distance=None, n_bins=20, sample_size=5000):
    """
    Estimate spatial autocorrelation range using Moran's I over increasing distance bands.
    Uses downsampled spatial weights for memory efficiency.
    """
    if max_distance is None:
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        max_distance = np.linalg.norm(bounds[1] - bounds[0]) * 0.3
    print("Estimating spatial autocorrelation range...")
    
    # Downsample to avoid memory overload
    if len(coords) > sample_size:
        idx = np.random.choice(len(coords), sample_size, replace=False)
        coords = coords[idx]
        y = y[idx]

    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    moran_i_values = []

    for i in range(len(bins) - 1):
        try:
            # Create spatial weight matrix within band
            w = DistanceBand(coords, threshold=bins[i+1], binary=True, silence_warnings=True)
            if len(w.neighbors) < 10:
                moran_i_values.append(0)
                continue

            moran = Moran(y, w, two_tailed=False)
            moran_i_values.append(moran.I)
        except Exception:
            moran_i_values.append(0)
    
    # Find distance where autocorrelation drops below threshold
    threshold = 0.1
    for i, mi in enumerate(moran_i_values):
        if abs(mi) < threshold:
            print(f"Autocorrelation drops at ≈ {bin_centers[i]:.0f}m (Moran's I = {mi:.2f})")
            return bin_centers[i]
    
    print(f"No clear drop detected — defaulting to max: {max_distance}m")
    return max_distance

def spatial_kfold(X, y, coords, n_splits=5, block_size=None, method='block', stratify_y=True):
    """
    Generate stratified spatial cross-validation folds.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target variable
    coords : np.ndarray, shape (n_samples, 2)
        Spatial coordinates
    n_splits : int
        Number of CV folds
    block_size : float, optional
        Block size in meters. If None, estimated from spatial autocorrelation
    method : str
        'block' for spatial blocks, 'kmeans' for clustering
    stratify_y : bool
        Whether to stratify by y distribution
        
    Returns:
    --------
    list of tuples : (train_idx, test_idx) for each fold
    """
    n_samples = len(X)
    
    # Estimate block size if not provided
    if block_size is None:
        autocorr_range = estimate_spatial_autocorrelation_range(coords, y)
        block_size = max(autocorr_range * 2, 500)  # At least 2x autocorr range
        print(f"Using block size: {block_size:.0f}m")
    
    if method == 'block':
        # Create spatial blocks
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        n_blocks_x = int(np.ceil((bounds[1, 0] - bounds[0, 0]) / block_size))
        n_blocks_y = int(np.ceil((bounds[1, 1] - bounds[0, 1]) / block_size))
        
        # Assign points to blocks
        block_assignments = np.zeros(n_samples, dtype=int)
        for i, (x, y_coord) in enumerate(coords):
            block_x = int((x - bounds[0, 0]) / block_size)
            block_y = int((y_coord - bounds[0, 1]) / block_size)
            block_assignments[i] = block_x * n_blocks_y + block_y
        
        # Create folds from blocks
        unique_blocks = np.unique(block_assignments)
        np.random.shuffle(unique_blocks)
        
        if stratify_y:
            # Stratify blocks by y distribution
            block_means = [y[block_assignments == block].mean() for block in unique_blocks]
            sorted_blocks = [x for _, x in sorted(zip(block_means, unique_blocks))]
        else:
            sorted_blocks = unique_blocks
        
        # Distribute blocks across folds
        folds = []
        for fold_idx in range(n_splits):
            fold_blocks = sorted_blocks[fold_idx::n_splits]
            test_mask = np.isin(block_assignments, fold_blocks)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            folds.append((train_idx, test_idx))
    
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        
        # Cluster coordinates
        kmeans = KMeans(n_clusters=n_splits, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Create folds from clusters
        folds = []
        for fold_idx in range(n_splits):
            test_mask = cluster_labels == fold_idx
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            folds.append((train_idx, test_idx))
    
    else:
        raise ValueError("Method must be 'block' or 'kmeans'")
    
    # Validate fold sizes
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        if len(test_idx) < 50:
            print(f"Warning: Fold {i+1} has very small test set ({len(test_idx)} samples)")
    
    return folds

def generate_oof_predictions(X, y, coords, models, folds, output_dir=None):
    """
    Generate out-of-fold predictions for all models across all folds.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    coords : np.ndarray
        Spatial coordinates
    models : list
        List of model instances
    folds : list
        List of (train_idx, test_idx) tuples
    output_dir : str, optional
        Directory to save model parameters
        
    Returns:
    --------
    tuple : (oof_predictions, model_parameters)
    """
    n_samples = len(X)
    n_models = len(models)
    oof_preds = np.zeros((n_samples, n_models))
    model_params_path = os.path.join(output_dir, 'model_params.pkl')
    if os.path.exists(model_params_path):
        import joblib
        model_params = joblib.load(model_params_path)
        print("Loaded saved model_params.")
    else:
        print("[Warning] model_params.pkl not found — retraining GWR will fail unless bandwidth is hardcoded.")
        model_params = {}
    
    print(f"\nGenerating OOF predictions for {n_models} models across {len(folds)} folds...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/{len(folds)}")
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        coords_train = coords[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        coords_test = coords[test_idx]
        
        for model_idx, model in enumerate(models):
            try:
                # Train model on training fold
                if hasattr(model, 'fit'):
                    if 'coords' in model.fit.__code__.co_varnames:
                        model.fit(X_train, y_train, coords_train)
                    else:
                        model.fit(X_train, y_train)
                
                # Capture model parameters (e.g., bandwidth for GWR)
                # Check for bandwidth attribute (both with and without underscore)
                bandwidth = None
                if hasattr(model, 'bandwidth_'):
                    bandwidth = model.bandwidth_
                elif hasattr(model, 'bandwidth'):
                    bandwidth = model.bandwidth
                
                if bandwidth is not None:
                    if model_idx not in model_params:
                        model_params[model_idx] = {}
                    model_params[model_idx]['bandwidth'] = bandwidth
                    print(f"  Model {model_idx + 1}: Captured bandwidth = {bandwidth}")
                
                # Predict on test fold
                if hasattr(model, 'predict'):
                    if 'coords' in model.predict.__code__.co_varnames:
                        preds = model.predict(X_test, coords_test)
                    else:
                        preds = model.predict(X_test)
                    
                    oof_preds[test_idx, model_idx] = preds
                    print(f"  Model {model_idx + 1}: R² = {r2_score(y_test, preds):.4f}")
                
            except Exception as e:
                print(f"  Model {model_idx + 1} failed: {str(e)}")
                oof_preds[test_idx, model_idx] = np.nan
    
    return oof_preds, model_params

def spatial_cv_meta_model(oof_preds, X, y, folds, model_type='lightgbm'):
    print("\n=== Performing Spatial CV on Meta-Model ===")
    n_samples = len(y)
    y_pred_meta_cv = np.full(n_samples, np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"Meta-Fold {fold_idx + 1}/{len(folds)}")
        X_train_meta = np.concatenate([oof_preds[train_idx], X[train_idx]], axis=1)
        X_test_meta = np.concatenate([oof_preds[test_idx], X[test_idx]], axis=1)
        y_train = y[train_idx]

        model = train_meta_model(oof_preds[train_idx], X[train_idx], y_train, model_type=model_type)
        y_pred_meta_cv[test_idx] = model.predict(X_test_meta)

    return y_pred_meta_cv

def spatial_cv_deep_kriging(meta_preds, y, coords, lap_eig_pca, folds):
    print("\n=== Performing Spatial CV on Deep Kriging ===")
    n_samples = len(y)
    final_preds = np.full(n_samples, np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"Deep Kriging Fold {fold_idx + 1}/{len(folds)}")
        residuals_train = y[train_idx] - meta_preds[train_idx]

        model = DeepKriging()
        model.fit(coords[train_idx], residuals_train, lap_eig_pca[train_idx], verbose=0)

        correction = model.predict(coords[test_idx], lap_eig_pca[test_idx])
        final_preds[test_idx] = meta_preds[test_idx] + correction

    return final_preds

def train_meta_model(oof_preds, X, y, model_type='lightgbm'):
    """
    Train meta-model on OOF predictions and original features.
    
    Parameters:
    -----------
    oof_preds : np.ndarray
        Out-of-fold predictions
    X : np.ndarray
        Original features
    y : np.ndarray
        Target variable
    model_type : str
        'lightgbm' or 'gradient_boosting'
        
    Returns:
    --------
    Trained meta-model
    """
    print(f"\nTraining {model_type} meta-model...")
    
    # Combine OOF predictions with original features
    X_meta = np.concatenate([oof_preds, X], axis=1)
    
    # Remove rows with NaN predictions
    valid_mask = ~np.isnan(oof_preds).any(axis=1)
    X_meta_valid = X_meta[valid_mask]
    y_valid = y[valid_mask]
    
    print(f"Using {valid_mask.sum()}/{len(y)} samples for meta-model training")
    
    if model_type == 'lightgbm':
        # LightGBM parameters
        params = {
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
        }
        
        # Create dataset
        train_data = lgb.Dataset(X_meta_valid, label=y_valid)
        
        # Train model
        meta_model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingRegressor
        meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        meta_model.fit(X_meta_valid, y_valid)
    
    else:
        raise ValueError("model_type must be 'lightgbm' or 'gradient_boosting'")
    
    # Evaluate meta-model
    if hasattr(meta_model, 'predict') and not isinstance(meta_model, lgb.Booster):
        y_pred_meta = meta_model.predict(X_meta_valid)
    else:
        # Handles lgb.Booster and other cases
        y_pred_meta = meta_model.predict(X_meta_valid)
    
    meta_r2 = r2_score(y_valid, y_pred_meta)
    meta_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_meta))
    
    print(f"Meta-model performance: R² = {meta_r2:.4f}, RMSE = {meta_rmse:.4f}")
    
    return meta_model

def retrain_base_models(X, y, coords, models, model_params=None, output_dir=None, retrain_gwrf=False):
    """
    Retrain base models on full dataset for deployment.
    
    Parameters:
    -----------
    X : np.ndarray
        Full feature matrix
    y : np.ndarray
        Full target variable
    coords : np.ndarray
        Full spatial coordinates
    models : list
        List of model instances
    model_params : dict, optional
        Dictionary of model parameters captured during OOF training
    output_dir : str, optional
        Directory to save bandwidth files
    retrain_gwrf : bool
        Whether to retrain GWRF on the full dataset
        
    Returns:
    --------
    list : Retrained models
    """
    print("\nRetraining base models on full dataset...")
    
    retrained_models = []
    for i, model in enumerate(models):
        try:
            # Conditionally skip GWRF retraining
            if isinstance(model, GWRFModel) and not retrain_gwrf:
                print(f"  Skipping retraining of Model {i + 1} (GWRF) as requested.")
                retrained_models.append(None)
                continue

            # Create a fresh instance with proper parameters
            model_class = type(model)
            
            # Special handling for GWRF to preserve its parameters
            if isinstance(model, GWRFModel):
                fresh_model = model_class(
                    n_estimators=model.n_estimators,
                    k_neighbors=model.k_neighbors,
                    min_samples_leaf=model.min_samples_leaf,
                    n_jobs=model.n_jobs,
                    subsample_fraction=model.subsample_fraction,
                    subsample_n=model.subsample_n
                )
            # Handling for GWR and other models
            elif model_params and i in model_params and 'bandwidth' in model_params[i]:
                bandwidth = model_params[i]['bandwidth']
                print(f"  Model {i + 1}: Using captured bandwidth = {bandwidth}")
                
                if model_class.__name__ == 'GWRModel':
                    fresh_model = model_class(bandwidth=bandwidth, kernel='gaussian')
                    if output_dir is not None:
                        bandwidth_file = os.path.join(output_dir, f'model_{i}_bandwidth.json')
                        fresh_model.save_bandwidth(bandwidth_file)
                        print(f"  Model {i + 1}: Bandwidth saved to {bandwidth_file}")
                else:
                    fresh_model = model_class(bandwidth=bandwidth)
            else:
                # Fallback to default initialization for other models
                fresh_model = model_class()
            
            # Train on full data
            if 'coords' in fresh_model.fit.__code__.co_varnames:
                fresh_model.fit(X, y, coords)
            else:
                fresh_model.fit(X, y)
            
            retrained_models.append(fresh_model)
            print(f"  Model {i + 1} retrained successfully")
            
        except Exception as e:
            print(f"  Model {i + 1} retraining failed: {str(e)}")
            retrained_models.append(None)
    
    return retrained_models

def load_gwen_results(output_dir="../output"):
    """
    Load GWEN variable selection results and validate approval.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing GWEN results
        
    Returns:
    --------
    list : Selected feature names from GWEN
    
    Raises:
    -------
    FileNotFoundError : If GWEN results not found or not approved
    """
    import json
    
    gwen_file = os.path.join(output_dir, 'gwen_results.json')
    approved_file = os.path.join(output_dir, 'gwen_approved.txt')
    
    if not os.path.exists(gwen_file):
        raise FileNotFoundError(
            f"❌ GWEN results not found: {gwen_file}\n"
            "   Please run gwen_variable_selection.py first to perform variable selection."
        )
    
    if not os.path.exists(approved_file):
        raise FileNotFoundError(
            f"❌ GWEN results not approved: {approved_file}\n"
            "   Please review and approve GWEN selection before proceeding to spatial CV.\n"
            f"   Create the file '{approved_file}' after reviewing results."
        )
    
    print("✅ GWEN results found and approved")
    
    with open(gwen_file, 'r') as f:
        gwen_results = json.load(f)
    
    selected_features = gwen_results['selected_features']
    total_features = gwen_results['total_features']
    threshold = gwen_results['selection_threshold']
    
    print(f"📊 GWEN Variable Selection Summary:")
    print(f"   • Total features evaluated: {total_features}")
    print(f"   • Selected features: {len(selected_features)}")
    print(f"   • Selection threshold: {threshold}")
    print(f"   • Features selected: {selected_features}")
    
    return selected_features

def main():
    """Run robust spatial cross-validation with GWEN-selected features (GW3C compliant)."""
    print("\n=== SPARC Robust Spatial Cross-Validation v2 (GW3C Compliant) ===\n")
    
    # --- Configuration ---
    # Set to False to skip retraining GWRF on the full dataset
    RETRAIN_GWRF_FULL = False
    
    # Load configuration
    config = load_config()
    print("Configuration loaded successfully")
    
    # NEW: Feature Selection (Configurable: GWEN or Manual)
    print("\n=== Step 0: Feature Selection Configuration ===")
    use_gwen = config['flags'].get('use_gwen_selection', True)  # Default to GWEN for compliance
    
    if use_gwen:
        print("🔍 Using GWEN Variable Selection (GW3C Compliant)")
        try:
            selected_features = load_gwen_results("../output")
        except FileNotFoundError as e:
            print(str(e))
            print("\n🛑 PIPELINE STOPPED: GWEN variable selection required")
            print("   Run gwen_variable_selection.py first, then approve results to proceed.")
            print("   Or set 'use_gwen_selection': false in config to use manual selection.")
            return
    else:
        print("👤 Using Manual Feature Selection")
        selected_features = config['predictors']['base_model']
        print(f"📋 Using {len(selected_features)} manually configured features:")
        print(f"   Features: {selected_features}")
        print("⚠️  Note: Manual selection bypasses GW3C compliance requirements")
    
    # Get paths and parameters
    raw_data_path = config['data']['file_path']
    identifier_col = config['variables']['identifier']
    target_col = config['variables']['target']
    coords_cols = config['variables']['coordinates']
    initial_crs = config['crs']['initial']
    target_crs = config['crs']['target_projected']
    
    print(f"\nParameters:")
    print(f"- Data file: {raw_data_path}")
    print(f"- Target: {target_col}")
    print(f"- Coordinates: {coords_cols}")
    
    # Create output directory
    output_dir = "spatial_cv_v2_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data (using selected features)
    print("\n=== Step 1: Loading and Preprocessing Data ===")
    selection_method = "GWEN-selected" if use_gwen else "manually configured"
    print(f"🔍 Using {selection_method} features: {selected_features}")
    
    # Validate that selected features exist in config
    original_features = config['predictors']['base_model']
    missing_features = [f for f in selected_features if f not in original_features]
    if missing_features:
        print(f"⚠️  Warning: GWEN selected features not in original config: {missing_features}")
        # Filter out missing features
        selected_features = [f for f in selected_features if f in original_features]
        print(f"   Using available features: {selected_features}")
    
    data = load_and_preprocess_data(
        raw_data_path=raw_data_path,
        identifier_col=identifier_col,
        target_col=target_col,
        coords_cols=coords_cols,
        predictor_cols=original_features,  # Load all, filter below
        initial_crs=initial_crs,
        target_crs=target_crs
    )
    
    # Prepare data using ONLY GWEN-selected features
    X = data[selected_features].values  # ⭐ KEY CHANGE: Only selected features
    y = data[target_col].values
    coords = data[coords_cols].values
    
    print(f"📊 Data Summary:")
    print(f"   • Original features available: {len(original_features)}")
    print(f"   • {selection_method} features used: {len(selected_features)}")
    print(f"   • Final data shape: {X.shape}")
    print(f"   • Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"   • Target mean: {y.mean():.3f}, std: {y.std():.3f}")
    print(f"   • Selected features: {selected_features}")
    
    # Step 2: Generate spatial folds
    print("\n=== Step 2: Generating Stratified Spatial Folds ===")
    fold_file = 'fold_assignments.csv'
    
    if os.path.exists(os.path.join(output_dir, fold_file)):
        print(f"Loading existing fold assignments from {fold_file}...")
        fold_assignments = pd.read_csv(os.path.join(output_dir, fold_file))['fold'].values
        
        # Reconstruct folds list
        n_splits = fold_assignments.max() + 1
        folds = []
        for i in range(int(n_splits)):
            test_idx = np.where(fold_assignments == i)[0]
            train_idx = np.where(fold_assignments != i)[0]
            folds.append((train_idx, test_idx))
            
        data['fold'] = fold_assignments
        
    else:
        print("Generating new spatial folds...")
        folds = spatial_kfold(
            X=X,
            y=y,
            coords=coords,
            n_splits=5,
            block_size=None,  # Will be estimated from spatial autocorrelation
            method='block',
            stratify_y=True
        )
        
        # Save fold assignments
        fold_assignments = np.zeros(len(data))
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            fold_assignments[test_idx] = fold_idx
        
        data['fold'] = fold_assignments
        data[['fold']].to_csv(os.path.join(output_dir, fold_file), index=False)
        print(f"Saved fold assignments to {fold_file}")
    
    # Step 3: Initialize base models (using selected features)
    print("\n=== Step 3: Initializing Base Models ===")
    base_models = [
        OLSModel(),
        GWRModel(bandwidth=500, kernel='gaussian'),
        GWRFModel(n_estimators=50, k_neighbors=100, min_samples_leaf=20, n_jobs=1, subsample_n=500),
        GGPGAM_SVC()
    ]
    
    model_names = ['OLS', 'GWR', 'GWRF', 'GGPGAM']
    print(f"📋 Base models: {model_names}")
    print(f"🔍 All models will use {len(selected_features)} {selection_method} features")
    print(f"   Features: {selected_features}")
    if use_gwen:
        print("   ✅ GW3C compliant feature selection enforced")
    else:
        print("   ⚠️  Manual selection - GW3C compliance bypassed")
    
    # Step 4: Generate OOF predictions
    print("\n=== Step 4: Generating Out-of-Fold Predictions ===")
    oof_file = 'oof_predictions.csv'
    
    if os.path.exists(os.path.join(output_dir, oof_file)):
        print(f"Loading existing OOF predictions from {oof_file}...")
        oof_preds = pd.read_csv(os.path.join(output_dir, oof_file)).values
        # Load saved model parameters
        import joblib
        model_params_path = os.path.join(output_dir, 'model_params.pkl')
        if os.path.exists(model_params_path):
            model_params = joblib.load(model_params_path)
            print("Loaded saved model parameters.")
        else:
            print("[Warning] model_params.pkl not found — retraining GWR will fail unless bandwidth is hardcoded.")
            model_params = {}
    else:
        print("Generating new OOF predictions...")
        oof_preds, model_params = generate_oof_predictions(X, y, coords, base_models, folds, output_dir)
        
        # Save OOF predictions
        oof_df = pd.DataFrame(oof_preds, columns=model_names)
        oof_df.to_csv(os.path.join(output_dir, oof_file), index=False)
        print(f"Saved OOF predictions to {oof_file}")
        
        # Save model parameters
        import joblib
        joblib.dump(model_params, os.path.join(output_dir, 'model_params.pkl'))
        print("Saved model parameters to model_params.pkl")
    


    # Step 5: Train meta-model
    print("\n=== Step 5: Training Meta-Model ===")
    meta_model_file_txt = 'meta_model.txt'
    meta_model_file_pkl = 'meta_model.pkl'

    if os.path.exists(os.path.join(output_dir, meta_model_file_txt)):
        print(f"Loading existing meta-model from {meta_model_file_txt}...")
        meta_model = lgb.Booster(model_file=os.path.join(output_dir, meta_model_file_txt))
    elif os.path.exists(os.path.join(output_dir, meta_model_file_pkl)):
        import joblib
        print(f"Loading existing meta-model from {meta_model_file_pkl}...")
        meta_model = joblib.load(os.path.join(output_dir, meta_model_file_pkl))
    else:
        print("Training new meta-model...")
        meta_model = train_meta_model(oof_preds, X, y, model_type='lightgbm')
        
        # Save meta-model
        if hasattr(meta_model, 'save_model'):
            meta_model.save_model(os.path.join(output_dir, meta_model_file_txt))
            print(f"Saved meta-model to {meta_model_file_txt}")
        else:
            import joblib
            joblib.dump(meta_model, os.path.join(output_dir, meta_model_file_pkl))
            print(f"Saved meta-model to {meta_model_file_pkl}")
    
    print("\n=== SPARC Spatial CV Pipeline with Meta + Deep Kriging ===\n")
    config = load_config()
    output_dir = "spatial_cv_v2_output"
    os.makedirs(output_dir, exist_ok=True)

    use_gwen = config['flags'].get('use_gwen_selection', True)
    selected_features = config['predictors']['base_model']  # default
    if use_gwen:
        from json import load
        with open("../output/gwen_results.json") as f:
            selected_features = load(f)['selected_features']

    data = load_and_preprocess_data(
        raw_data_path=config['data']['file_path'],
        identifier_col=config['variables']['identifier'],
        target_col=config['variables']['target'],
        coords_cols=config['variables']['coordinates'],
        predictor_cols=config['predictors']['base_model'],
        initial_crs=config['crs']['initial'],
        target_crs=config['crs']['target_projected']
    )
    
    # Filter selected features to only include those that exist in the data
    available_features = data.columns.tolist()
    original_selected = selected_features.copy() if isinstance(selected_features, list) else selected_features
    selected_features = [f for f in selected_features if f in available_features]
    
    # Report any missing features
    missing_features = [f for f in original_selected if f not in available_features]
    if missing_features:
        print(f"Warning: The following selected features are not available in the data and will be excluded: {missing_features}")
    print(f"Using GWEN-selected features: {selected_features}")

    # Extract base data
    X_gwen = data[selected_features].values
    y = data[config['variables']['target']].values
    coords = data[config['variables']['coordinates']].values

    print("\n=== Generating Laplacian Eigenmap Features ===")
    # 1. Generate Laplacian Eigenmap
    lap = LaplacianEigenmap(n_components=10, k_neighbors=10)
    lap_eig = lap.fit_transform(coords)
    print(f"Generated {lap_eig.shape[1]} Laplacian Eigenmap components")

    # 2. Apply PCA to Laplacian Eigenmap
    print("=== Applying PCA to Laplacian Features ===")
    pca = PCA(n_components=5)
    lap_eig_pca = pca.fit_transform(lap_eig)
    print(f"Generated {lap_eig_pca.shape[1]} Laplacian PCA components")
    
    # 3. Augment feature matrix
    X_augmented = np.concatenate([X_gwen, lap_eig_pca], axis=1)
    print(f"Augmented feature matrix shape: {X_augmented.shape}")
    
    # 4. Update feature names
    feature_names = list(selected_features) + [f"lap_pca_{i+1}" for i in range(lap_eig_pca.shape[1])]
    print(f"Final feature names: {feature_names}")
    
    # 5. Save Laplacian/PCA transformers for deployment
    transformer_dir = os.path.join(output_dir, "transformers")
    os.makedirs(transformer_dir, exist_ok=True)
    joblib.dump(lap, os.path.join(transformer_dir, 'laplacian_eigenmap.pkl'))
    joblib.dump(pca, os.path.join(transformer_dir, 'laplacian_pca.pkl'))
    print(f"Saved transformers to {transformer_dir}")
    
    # Use augmented matrix as X for all subsequent operations
    X = X_augmented

    print("\n=== Generating Spatial Folds ===")
 
    folds = spatial_kfold(X, y, coords, n_splits=5, block_size=None)

    print("\n=== Base Models (OLS, GWR, GWRF, GGPGAM) ===")
    base_models = [
        OLSModel(),
        GWRModel(bandwidth=500, kernel='gaussian'),
        GWRFModel(n_estimators=50, k_neighbors=100, min_samples_leaf=20, n_jobs=1, subsample_n=500),
        GGPGAM_SVC()
    ]
    
    oof_preds, model_params = generate_oof_predictions(X, y, coords, base_models, folds, output_dir)

    print("\n=== Meta-Model via Spatial CV ===")
    meta_preds_cv = spatial_cv_meta_model(oof_preds, X, y, folds)
    r2_cv_meta = r2_score(y, meta_preds_cv)
    rmse_cv_meta = np.sqrt(mean_squared_error(y, meta_preds_cv))
    print(f"Meta (SCV): R²={r2_cv_meta:.4f}, RMSE={rmse_cv_meta:.4f}")

    print("\n=== Deep Kriging Correction via SCV ===")
    final_corrected_preds = spatial_cv_deep_kriging(
        meta_preds=meta_preds_cv,
        y=y,
        coords=coords,
        lap_eig_pca=lap_eig_pca,
        folds=folds
    )
    r2_final = r2_score(y, final_corrected_preds)
    rmse_final = np.sqrt(mean_squared_error(y, final_corrected_preds))
    print(f"Deep Kriging (Final): R²={r2_final:.4f}, RMSE={rmse_final:.4f}")

    print("\nSaving Outputs...")
    pd.DataFrame({
        'coords_x': coords[:, 0],
        'coords_y': coords[:, 1],
        'truth': y,
        'meta_cv': meta_preds_cv,
        'final_corrected': final_corrected_preds
    }).to_csv(os.path.join(output_dir, 'predictions_summary.csv'), index=False)


    # Step 6: Retrain base models on full dataset
    print("\n=== Step 6: Retraining Base Models for Deployment ===")
    
    final_model_files = [f'final_{name.lower()}_model.pkl' for name in model_names]
    # GWRF is handled by the retrain_gwrf flag, so exclude it from the check
    if isinstance(base_models[2], GWRFModel) and not RETRAIN_GWRF_FULL:
        final_model_files.pop(2)
        
    if check_files_exist(final_model_files, output_dir):
        print("All final models already exist, skipping retraining...")
        final_models = []
        for i, name in enumerate(model_names):
            if isinstance(base_models[i], GWRFModel) and not RETRAIN_GWRF_FULL:
                final_models.append(None)
                continue
            
            try:
                import joblib
                model_path = os.path.join(output_dir, f'final_{name.lower()}_model.pkl')
                final_models.append(joblib.load(model_path))
                print(f"  Loaded final_{name.lower()}_model.pkl")
            except FileNotFoundError:
                print(f"  Could not find final_{name.lower()}_model.pkl, setting to None.")
                final_models.append(None)

    else:
        print("Retraining base models...")
        final_models = retrain_base_models(
            X, y, coords, base_models, model_params, output_dir,
            retrain_gwrf=RETRAIN_GWRF_FULL
        )
        
        # Save final models
        for i, (model, name) in enumerate(zip(final_models, model_names)):
            if model is not None:
                try:
                    import joblib
                    joblib.dump(model, os.path.join(output_dir, f'final_{name.lower()}_model.pkl'))
                    print(f"  Saved {name} model")
                except Exception as e:
                    print(f"  Failed to save {name} model: {str(e)}")

    # Step 7: Final evaluation
    print("\n=== Step 7: Final Evaluation ===")
    perf_summary_file = 'performance_summary.csv'

    if os.path.exists(os.path.join(output_dir, perf_summary_file)):
        print(f"Loading existing performance summary from {perf_summary_file}...")
        performance_df = pd.read_csv(os.path.join(output_dir, perf_summary_file))
        print("\nOut-of-Fold Performance:")
        print(performance_df.to_string(index=False))

    else:
        print("Calculating and saving performance summary...")
        # Calculate OOF performance for each model
        oof_performance = {}
        for i, name in enumerate(model_names):
            valid_mask = ~np.isnan(oof_preds[:, i])
            if valid_mask.sum() > 0:
                oof_r2 = r2_score(y[valid_mask], oof_preds[valid_mask, i])
                oof_rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_preds[valid_mask, i]))
                oof_performance[name] = {'R²': oof_r2, 'RMSE': oof_rmse}
        
        # Calculate meta-model performance
        valid_mask = ~np.isnan(oof_preds).any(axis=1)
        X_meta = np.concatenate([oof_preds, X], axis=1)
        X_meta_valid = X_meta[valid_mask]
        y_valid = y[valid_mask]
        
        if hasattr(meta_model, 'predict') and not isinstance(meta_model, lgb.Booster):
            y_pred_meta = meta_model.predict(X_meta_valid)
        else:
            # Handles lgb.Booster and other cases
            y_pred_meta = meta_model.predict(X_meta_valid)
        
        meta_performance = {
            'R²': r2_score(y_valid, y_pred_meta),
            'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_meta))
        }
        
        # Print results
        print(f"\n📊 Out-of-Fold Performance (using {len(selected_features)} {selection_method} features):")
        for name, perf in oof_performance.items():
            print(f"  {name}: R² = {perf['R²']:.4f}, RMSE = {perf['RMSE']:.4f}")
        
        print(f"\n🎯 Meta-Model Performance:")
        print(f"  LightGBM: R² = {meta_performance['R²']:.4f}, RMSE = {meta_performance['RMSE']:.4f}")
        
        if use_gwen:
            print(f"\n✅ GW3C Compliance: All models used only GWEN-selected features")
        else:
            print(f"\n⚠️  Manual Selection: GW3C compliance bypassed")
        print(f"   Selected features: {selected_features}")
        
        # Save performance summary
        performance_data = {
            'Model': list(oof_performance.keys()) + ['Meta-LightGBM'],
            'R²': [perf['R²'] for perf in oof_performance.values()] + [meta_performance['R²']],
            'RMSE': [perf['RMSE'] for perf in oof_performance.values()] + [meta_performance['RMSE']]
        }
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(os.path.join(output_dir, perf_summary_file), index=False)
        print(f"\nResults saved to {output_dir}/")

    compliance_status = "GW3C Compliant" if use_gwen else "Manual Selection"
    print(f"\n=== SPARC Robust Spatial Cross-Validation Complete ({compliance_status}) ===")
    print(f"✅ All models trained using {len(selected_features)} {selection_method} features")
    if use_gwen:
        print(f"✅ GW3C algorithm compliance enforced throughout pipeline")
    else:
        print(f"⚠️  Manual feature selection used - GW3C compliance bypassed")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main() 