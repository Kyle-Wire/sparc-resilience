"""
.. deprecated:: 2.1.0
   This module is **RETIRED**.  All scenario simulation logic has been
   consolidated into ``interventions.scenario_simulator.ScenarioSimulator``.
   This file is kept only for reference and will be removed in a future
   release.

PHYSICS-CONSTRAINED SCENARIO PREDICTOR v2
==========================================
With extended increments and SPATIAL HETEROGENEITY.

Spatial heterogeneity approach:
- Use GWR local coefficients to create a spatial multiplier
- Constrain multiplier to preserve physics sign
- Result: Physics-correct deltas that vary spatially

Author: SPARC Labs
Date: January 2026
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
from pathlib import Path
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Import extrapolation guard for scenario confidence assessment
try:
    from interventions.extrapolation_guard import (
        compute_extrapolation_score,
        classify_prediction_confidence
    )
    EXTRAPOLATION_GUARD_AVAILABLE = True
except ImportError:
    EXTRAPOLATION_GUARD_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(r'C:\Users\KWire\OneDrive - Kleinfelder\Documents\GitHub\SPARC_LABS_GW3C_v1\Final_Version_Research\Phase_1')
DATA_PATH = BASE_DIR / 'data' / 'brown4.csv'
MODEL_DIR = BASE_DIR / 'output' / 'Stage_2_Spatial_CV'
OUTPUT_DIR = BASE_DIR / 'output' / 'physics_constrained_scenarios'
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature order (from model scaler)
FEATURES = ['Elevation_m', 'Distance_from_water_m', 'Pct_Canopy', 'Pct_Impervious', 'Albedo', 'NDVI']
COORDS = ['POINT_X', 'POINT_Y']

# District mapping (0 = cells not in a specific district)
DISTRICT_NAMES = {
    0: 'Other',
    1: 'Athletics',
    2: 'Biochem', 
    3: 'Residential',
    4: 'Jewelry_District'
}

# =============================================================================
# HYBRID PHYSICS PRIORS - Extended increments
# =============================================================================
LITERATURE_WEIGHT = 0.5  # 50/50 blend

OLS_COEFFICIENTS = {
    'Pct_Canopy': -0.00005,      # °F per 1pp
    'Pct_Impervious': +0.0102,   # °F per 1pp
    'NDVI': -0.452,              # °F per 0.1 NDVI
    'Albedo': -0.050,            # °F per 0.1 albedo (in our scale)
}

LITERATURE_COEFFICIENTS = {
    'Pct_Canopy': -0.028,        # °F per 1pp
    'Pct_Impervious': +0.033,    # °F per 1pp
    'NDVI': -0.270,              # °F per 0.1
    'Albedo': -0.230,            # °F per 0.1
}

def blend_coefficient(var_name: str, weight: float = LITERATURE_WEIGHT) -> float:
    """Blend OLS and literature coefficients."""
    ols = OLS_COEFFICIENTS[var_name]
    lit = LITERATURE_COEFFICIENTS[var_name]
    return (1 - weight) * ols + weight * lit

PHYSICS_PRIORS = {
    'Pct_Canopy': {
        'coefficient': blend_coefficient('Pct_Canopy'),
        'unit_increment': 1,
        'direction': 'cooling',
        'ols_coef': OLS_COEFFICIENTS['Pct_Canopy'],
        'lit_coef': LITERATURE_COEFFICIENTS['Pct_Canopy'],
    },
    'Pct_Impervious': {
        'coefficient': blend_coefficient('Pct_Impervious'),
        'unit_increment': 1,
        'direction': 'warming',
        'ols_coef': OLS_COEFFICIENTS['Pct_Impervious'],
        'lit_coef': LITERATURE_COEFFICIENTS['Pct_Impervious'],
    },
    'NDVI': {
        'coefficient': blend_coefficient('NDVI'),
        'unit_increment': 0.1,
        'direction': 'cooling',
        'ols_coef': OLS_COEFFICIENTS['NDVI'],
        'lit_coef': LITERATURE_COEFFICIENTS['NDVI'],
    },
    'Albedo': {
        'coefficient': blend_coefficient('Albedo'),
        'unit_increment': 0.1,
        'direction': 'cooling',
        'ols_coef': OLS_COEFFICIENTS['Albedo'],
        'lit_coef': LITERATURE_COEFFICIENTS['Albedo'],
    }
}

# EXTENDED Scenario definitions - FULL RANGE TESTING
# Test each variable to its maximum theoretical change
SCENARIOS = {
    'Pct_Canopy': {
        'increments': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Full 0→100% range
        'direction': 'increase',
        'min_val': 0,
        'max_val': 100,
        'unit': 'pp'
    },
    'Pct_Impervious': {
        'increments': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Full 100→0% range
        'direction': 'decrease',
        'min_val': 0,
        'max_val': 100,
        'unit': 'pp'
    },
    'NDVI': {
        'increments': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],  # Full range to ~0.7
        'direction': 'increase',
        'min_val': -0.10,
        'max_val': 1.0,
        'unit': ''
    },
    'Albedo': {
        'increments': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],  # Full 0→1 range
        'direction': 'increase',
        'min_val': 0.0,
        'max_val': 1.0,
        'unit': ''
    }
}

# =============================================================================
# LOAD MODELS
# =============================================================================
print("="*70)
print("PHYSICS-CONSTRAINED SCENARIO PREDICTOR v2")
print("With Extended Increments + Spatial Heterogeneity")
print("="*70)

print("\n1. Loading models...")
meta_model = joblib.load(MODEL_DIR / 'standard_meta_ensemble.pkl')
ols_model = joblib.load(MODEL_DIR / 'base_models_full' / 'ols_model_full.pkl')
gwr_model = joblib.load(MODEL_DIR / 'base_models_full' / 'gwr_model_full.pkl')
gwrf_model = joblib.load(MODEL_DIR / 'base_models_full' / 'gwrf_model_full.pkl')
ggpgam_model = joblib.load(MODEL_DIR / 'base_models_full' / 'ggpgam_model_full.pkl')
print("   ✓ All models loaded")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n2. Loading baseline data...")
baseline_df = pd.read_csv(DATA_PATH)
print(f"   ✓ {len(baseline_df)} cells loaded")

target_mask = baseline_df['Location'].isin(DISTRICT_NAMES.keys())
print(f"   Target district cells: {target_mask.sum()}")

# =============================================================================
# SPATIAL HETEROGENEITY - Calculate local sensitivity multipliers
# =============================================================================
print("\n3. Computing spatial heterogeneity multipliers...")

# Approach: Use baseline temperature to modulate sensitivity
# Hotter cells may respond differently than cooler cells
# We'll create a multiplier based on z-score of baseline temp

# Get baseline predicted temps from the meta-ensemble (run once)
def predict_baseline_temperature(df, verbose=False):
    """Run full prediction pipeline."""
    X = df[FEATURES].values
    coords = df[COORDS].values
    
    if verbose:
        print("   Running OLS...")
    ols_pred = ols_model.predict(X)
    
    if verbose:
        print("   Running GWR...")
    gwr_pred = gwr_model.predict(X, coords)
    
    if verbose:
        print("   Running GWRF...")
    gwrf_pred = gwrf_model.predict(X, coords)
    # Handle GWRF v2 tuple return (predictions, uncertainty)
    if isinstance(gwrf_pred, tuple):
        gwrf_pred = gwrf_pred[0]
    
    if verbose:
        print("   Running GGPGAM...")
    ggpgam_pred = ggpgam_model.predict(X, coords)
    
    base_predictions = {
        'ols': ols_pred,
        'gwr': gwr_pred,
        'gwrf': gwrf_pred,
        'ggpgam': ggpgam_pred
    }
    
    if verbose:
        print("   Running meta-ensemble...")
    final_pred = meta_model.predict(base_predictions, coords=coords)
    
    return final_pred, ols_pred, gwr_pred, gwrf_pred, ggpgam_pred

print("   Computing baseline predictions...")
baseline_pred, ols_base, gwr_base, gwrf_base, ggpgam_base = predict_baseline_temperature(baseline_df, verbose=True)
print(f"\n   Baseline mean: {baseline_pred.mean():.2f}°F")

# Calculate spatial multiplier based on how each model differs from OLS
# This captures local sensitivity variation
# Multiplier = 1 + (local_model - global_model) / global_model * scaling_factor

# Use the range of predictions as a proxy for local sensitivity
pred_range = np.maximum.reduce([ols_base, gwr_base, gwrf_base, ggpgam_base]) - \
             np.minimum.reduce([ols_base, gwr_base, gwrf_base, ggpgam_base])

# Normalize to create a multiplier centered at 1.0
# Higher disagreement = higher uncertainty = larger potential response
pred_range_norm = pred_range / pred_range.mean()

# Also use baseline temperature deviation
temp_zscore = (baseline_pred - baseline_pred.mean()) / baseline_pred.std()

# Combine into spatial multiplier
# Hotter areas (positive z-score) might respond MORE to cooling interventions
# Model disagreement areas might have MORE uncertainty/variability
spatial_multiplier = 1.0 + 0.2 * temp_zscore + 0.1 * (pred_range_norm - 1.0)

# Clip to reasonable range (0.5 to 1.5 - no more than 50% variation)
spatial_multiplier = np.clip(spatial_multiplier, 0.5, 1.5)

print(f"\n   Spatial multiplier range: {spatial_multiplier.min():.2f} to {spatial_multiplier.max():.2f}")
print(f"   Mean multiplier: {spatial_multiplier.mean():.3f}")

# Store in results
results_df = baseline_df.copy()
results_df['pred_baseline'] = baseline_pred
results_df['spatial_multiplier'] = spatial_multiplier

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================
print("\n" + "="*70)
print("4. HYBRID PHYSICS PRIORS")
print("="*70)
print(f"\n   Blending: {(1-LITERATURE_WEIGHT)*100:.0f}% OLS + {LITERATURE_WEIGHT*100:.0f}% Literature")
print("\n   Variable         OLS Coef    Lit Coef    HYBRID      per unit")
print("   " + "-"*64)
for var, prior in PHYSICS_PRIORS.items():
    ols = prior['ols_coef']
    lit = prior['lit_coef']
    hybrid = prior['coefficient']
    unit = prior['unit_increment']
    unit_str = f"1pp" if 'Pct' in var else f"{unit}"
    print(f"   {var:15s} {ols:+.5f}   {lit:+.5f}   {hybrid:+.5f}   per {unit_str}")

print("\n   Spatial heterogeneity: ENABLED")
print("   - Multiplier based on baseline temp z-score + model disagreement")
print("   - Range: 0.5x to 1.5x of base coefficient")

# =============================================================================
# PHYSICS-BASED DELTA WITH SPATIAL HETEROGENEITY
# =============================================================================
def calculate_physics_delta_spatial(variable: str, actual_change: np.ndarray, 
                                    spatial_mult: np.ndarray) -> np.ndarray:
    """
    Calculate temperature change with spatial heterogeneity.
    
    Parameters:
    -----------
    variable : str
        Variable name
    actual_change : np.ndarray
        Actual change in variable per cell
    spatial_mult : np.ndarray
        Spatial multiplier per cell (from heterogeneity calculation)
    
    Returns:
    --------
    np.ndarray
        Temperature change per cell (°F)
    """
    prior = PHYSICS_PRIORS[variable]
    coef = prior['coefficient']
    unit = prior['unit_increment']
    
    # Base physics delta
    base_delta = coef * (actual_change / unit)
    
    # Apply spatial multiplier
    # For cooling (negative coef), higher multiplier = more cooling
    # For warming (positive coef), higher multiplier = more warming
    spatial_delta = base_delta * spatial_mult
    
    return spatial_delta

# =============================================================================
# RUN ALL SCENARIOS
# =============================================================================
print("\n" + "="*70)
print("5. RUNNING EXTENDED SCENARIOS (With Spatial Heterogeneity)")
print("="*70)

summary_rows = []

for var_name, config in SCENARIOS.items():
    print(f"\n--- Variable: {var_name} ---")
    prior = PHYSICS_PRIORS[var_name]
    
    for increment in config['increments']:
        if config['direction'] == 'increase':
            scenario_name = f"{var_name}_plus_{str(increment).replace('.', 'p')}"
            delta = increment
        else:
            scenario_name = f"{var_name}_minus_{str(increment).replace('.', 'p')}"
            delta = -increment
        
        print(f"\n   Scenario: {scenario_name}")
        
        # Calculate actual change per cell (accounting for bounds)
        original_values = baseline_df.loc[target_mask, var_name].values.copy()
        modified_values = original_values + delta
        modified_values = np.clip(modified_values, config['min_val'], config['max_val'])
        actual_change = modified_values - original_values
        
        # Get spatial multiplier for target cells
        target_spatial_mult = spatial_multiplier[target_mask]
        
        # Calculate physics-based delta WITH spatial heterogeneity
        temp_delta_target = calculate_physics_delta_spatial(var_name, actual_change, target_spatial_mult)
        
        # Create scenario predictions
        scenario_pred = baseline_pred.copy()
        scenario_pred[target_mask] = baseline_pred[target_mask] + temp_delta_target
        
        # Store
        results_df[f'pred_{scenario_name}'] = scenario_pred
        
        # === Extrapolation guard: flag cells pushed outside training domain ===
        if EXTRAPOLATION_GUARD_AVAILABLE:
            try:
                # Build training feature matrix from baseline
                X_train = baseline_df[FEATURES].values
                
                # Build scenario feature matrix (modified variable)
                X_scenario = baseline_df[FEATURES].values.copy()
                var_idx = FEATURES.index(var_name)
                X_scenario[target_mask, var_idx] = modified_values
                
                # Compute extrapolation scores for scenario cells
                extrap_scores = compute_extrapolation_score(X_train, X_scenario[target_mask])
                
                # Classify confidence
                confidence_labels = classify_prediction_confidence(
                    extrap_scores,
                    temp_delta_target,
                    np.std(baseline_pred[target_mask])
                )
                
                # Store confidence flags
                conf_col = f'confidence_{scenario_name}'
                results_df[conf_col] = 'N/A'
                results_df.loc[target_mask, conf_col] = confidence_labels
                
                # Report
                n_speculative = np.sum(confidence_labels == 'SPECULATIVE')
                n_low = np.sum(confidence_labels == 'LOW')
                if n_speculative > 0 or n_low > 0:
                    print(f"      [GUARD] {n_speculative} SPECULATIVE + {n_low} LOW confidence cells")
            except Exception as guard_e:
                print(f"      [GUARD] Extrapolation check failed: {guard_e}")
        
        # Per-district summaries
        for loc_id, loc_name in DISTRICT_NAMES.items():
            mask = baseline_df['Location'] == loc_id
            if mask.sum() > 0:
                district_indices = baseline_df.index[mask]
                target_indices = baseline_df.index[target_mask]
                district_in_target = np.isin(target_indices, district_indices)
                
                district_var_change = actual_change[district_in_target].mean()
                district_temp_delta = temp_delta_target[district_in_target].mean()
                district_temp_std = temp_delta_target[district_in_target].std()
                district_mult_mean = target_spatial_mult[district_in_target].mean()
                
                print(f"      {loc_name}: {district_temp_delta:+.4f}°F ± {district_temp_std:.4f} (var: {district_var_change:+.2f}{config['unit']}, mult: {district_mult_mean:.2f})")
                
                summary_rows.append({
                    'Variable': var_name,
                    'Scenario': scenario_name,
                    'Increment': increment,
                    'Direction': config['direction'],
                    'District': loc_name,
                    'N_Cells': mask.sum(),
                    'Avg_Var_Change': district_var_change,
                    'Avg_Temp_Delta_F': district_temp_delta,
                    'Temp_Delta_Std': district_temp_std,
                    'Avg_Spatial_Multiplier': district_mult_mean,
                    'Physics_Coef': prior['coefficient']
                })

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("6. SAVING RESULTS")
print("="*70)

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / 'physics_constrained_extended_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\n   ✓ Summary saved: {summary_path}")

# Pivot table
print("\n   Temperature Delta Summary (°F) - With Spatial Heterogeneity:")
pivot = summary_df.pivot_table(
    index=['Variable', 'Increment'],
    columns='District',
    values='Avg_Temp_Delta_F',
    aggfunc='mean'
)
print(pivot.round(4).to_string())

# =============================================================================
# SAVE GEOPACKAGE
# =============================================================================
print("\n\n7. Creating GeoPackage...")

geometry = [Point(xy) for xy in zip(baseline_df['POINT_X'], baseline_df['POINT_Y'])]
gdf = gpd.GeoDataFrame(results_df, geometry=geometry, crs="EPSG:3438")
gdf['District'] = gdf['Location'].map(DISTRICT_NAMES)

# Select columns
base_cols = ['geometry', 'District', 'Location', 'POINT_X', 'POINT_Y', 'AAT_z', 'spatial_multiplier']
feature_cols = FEATURES
pred_cols = [c for c in gdf.columns if c.startswith('pred_')]
final_cols = base_cols + feature_cols + pred_cols

gdf_export = gdf[final_cols]
gpkg_path = OUTPUT_DIR / 'physics_constrained_extended.gpkg'
gdf_export.to_file(gpkg_path, driver='GPKG')
print(f"   ✓ GeoPackage saved: {gpkg_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("EXTENDED SCENARIO RUN COMPLETE")
print("="*70)
print(f"\nFiles created:")
print(f"  1. {summary_path}")
print(f"  2. {gpkg_path}")
print(f"\nScenarios run: {len([c for c in results_df.columns if c.startswith('pred_')]) - 1}")
print(f"\nKey features:")
print(f"  - Extended increments (Canopy up to +80pp, Impervious down to -70pp)")
print(f"  - NDVI increments up to +0.40 (reaches ~0.6)")
print(f"  - Albedo increments up to +1.50 (reaches ~2.8 in our scale)")
print(f"  - Spatial heterogeneity via multiplier (0.5x to 1.5x)")
print(f"  - Hotter areas respond more strongly to interventions")
