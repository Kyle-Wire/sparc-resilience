"""
SENSITIVITY / SATURATION CURVE VISUALIZATIONS
==============================================
Visualize how each district responds to incremental changes in each variable.
Shows diminishing returns and saturation effects.

Author: SPARC Labs
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(r'C:\Users\KWire\OneDrive - Kleinfelder\Documents\GitHub\SPARC_LABS_GW3C_v1\Final_Version_Research\Phase_1')
OUTPUT_DIR = BASE_DIR / 'output' / 'physics_constrained_scenarios'

# Try to load extended summary first, fall back to regular
summary_path = OUTPUT_DIR / 'physics_constrained_extended_summary.csv'
if not summary_path.exists():
    summary_path = OUTPUT_DIR / 'physics_constrained_summary.csv'

print(f"Loading: {summary_path}")
df = pd.read_csv(summary_path)

# District colors (0=Other, 1=Athletics, 2=Biochem, 3=Residential, 4=Jewelry District)
DISTRICT_COLORS = {
    'Other': '#95a5a6',          # Gray
    'Athletics': '#377eb8',       # Blue
    'Biochem': '#e41a1c',         # Red
    'Residential': '#4daf4a',     # Green
    'Jewelry_District': '#ff7f00' # Orange
}

# Variable display names
VAR_DISPLAY = {
    'Pct_Canopy': 'Tree Canopy',
    'Pct_Impervious': 'Impervious Surface',
    'NDVI': 'NDVI',
    'Albedo': 'Albedo'
}

# Variable colors for per-district plots
VAR_COLORS = {
    'Pct_Canopy': '#27ae60',      # Green
    'Pct_Impervious': '#7f8c8d',  # Gray
    'NDVI': '#16a085',            # Teal
    'Albedo': '#e67e22'           # Orange
}

# Maximum increments for each variable (for normalization to 0-100% scale)
VAR_MAX_INCREMENT = {
    'Pct_Canopy': 100,      # 100pp max
    'Pct_Impervious': 100,  # 100pp max (decrease)
    'NDVI': 1.00,           # 0.70 max increase
    'Albedo': 1.00          # 1.0 max increase
}

# =============================================================================
# FIGURE 1: SENSITIVITY CURVES BY DISTRICT (NORMALIZED X-AXIS)
# =============================================================================
print("\nCreating sensitivity curves by district (normalized scale)...")

# Get unique districts (excluding 'Other' if desired)
districts = [d for d in DISTRICT_COLORS.keys() if d != 'Other']
variables = df['Variable'].unique()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, district in enumerate(districts):
    ax = axes[idx]
    district_data = df[df['District'] == district]
    
    for var in variables:
        var_data = district_data[district_data['Variable'] == var].sort_values('Increment')
        
        if len(var_data) > 0:
            # Normalize X to 0-100% of max possible intervention
            max_inc = VAR_MAX_INCREMENT.get(var, var_data['Increment'].max())
            x_normalized = (var_data['Increment'].values / max_inc) * 100
            y = var_data['Avg_Temp_Delta_F'].values
            
            # Add error bars if we have std
            if 'Temp_Delta_Std' in var_data.columns:
                yerr = var_data['Temp_Delta_Std'].values
                ax.errorbar(x_normalized, y, yerr=yerr, marker='o', markersize=8, 
                           label=VAR_DISPLAY.get(var, var), color=VAR_COLORS[var],
                           capsize=3, linewidth=2)
            else:
                ax.plot(x_normalized, y, marker='o', markersize=8, label=VAR_DISPLAY.get(var, var), 
                       color=VAR_COLORS[var], linewidth=2)
    
    # Formatting
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Intervention Intensity (% of Maximum)', fontsize=12)
    ax.set_ylabel('Temperature Change (°F)', fontsize=12)
    ax.set_title(f'{district} District - All Variables', fontsize=14, fontweight='bold',
                color=DISTRICT_COLORS[district])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)  # Extend slightly past 100 for visibility

plt.tight_layout()
fig_path = OUTPUT_DIR / 'sensitivity_curves_by_district.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
print(f"   Saved: {fig_path} (+ SVG)")
plt.close()

# =============================================================================
# FIGURE 2: CUMULATIVE COOLING POTENTIAL BY DISTRICT
# =============================================================================
print("\nCreating cumulative cooling potential chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, district in enumerate(districts):
    ax = axes[idx]
    district_data = df[df['District'] == district]
    
    cooling_data = []
    for var in variables:
        var_data = district_data[district_data['Variable'] == var]
        if len(var_data) > 0:
            # Get the maximum cooling (most negative) value
            max_effect = var_data['Avg_Temp_Delta_F'].min()
            max_cooling = -max_effect if max_effect < 0 else 0
            cooling_data.append({
                'Variable': VAR_DISPLAY.get(var, var),
                'Cooling': max_cooling,
                'Color': VAR_COLORS[var]
            })
    
    if cooling_data:
        cooling_df = pd.DataFrame(cooling_data)
        bars = ax.bar(cooling_df['Variable'], cooling_df['Cooling'], 
                     color=[d['Color'] for d in cooling_data], alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, cooling_df['Cooling']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}°F', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Maximum Cooling (°F)', fontsize=11)
    ax.set_title(f'{district} District', fontsize=13, fontweight='bold',
                color=DISTRICT_COLORS[district])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Maximum Cooling Potential by Intervention Type', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIR / 'max_cooling_potential_by_district.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
print(f"   Saved: {fig_path} (+ SVG)")
plt.close()

# =============================================================================
# FIGURE 3: SATURATION ANALYSIS BY DISTRICT
# =============================================================================
print("\nCreating saturation/diminishing returns analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, district in enumerate(districts):
    ax = axes[idx]
    district_data = df[df['District'] == district]
    
    for var in variables:
        var_data = district_data[district_data['Variable'] == var].sort_values('Increment')
        
        if len(var_data) > 1:
            x = var_data['Avg_Var_Change'].abs().values  # Absolute change
            y = var_data['Avg_Temp_Delta_F'].abs().values  # Absolute temp change
            
            # Calculate marginal efficiency (slope between points)
            marginal_eff = np.diff(y) / np.diff(x)
            marginal_x = (x[:-1] + x[1:]) / 2  # Midpoints
            
            ax.plot(marginal_x, marginal_eff, marker='s', markersize=6,
                   label=VAR_DISPLAY.get(var, var), color=VAR_COLORS[var], linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cumulative Variable Change', fontsize=11)
    ax.set_ylabel('Marginal °F per Unit Change', fontsize=11)
    ax.set_title(f'{district} - Marginal Efficiency\n(Diminishing Returns)', 
                fontsize=13, fontweight='bold', color=DISTRICT_COLORS[district])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / 'diminishing_returns_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
print(f"   Saved: {fig_path} (+ SVG)")
plt.close()

# =============================================================================
# FIGURE 4: HEATMAP - Districts x Variables
# =============================================================================
print("\nCreating sensitivity heatmap...")

# Get sensitivity at a standard increment for comparison
standard_sensitivities = []

for var in variables:
    var_data = df[df['Variable'] == var]
    
    # Find a "standard" increment
    if 'Pct' in var:
        target_inc = 10
    elif var == 'NDVI':
        target_inc = 0.1
    else:  # Albedo
        target_inc = 0.1
    
    # Get closest increment
    increments = var_data['Increment'].unique()
    closest_inc = increments[np.argmin(np.abs(increments - target_inc))]
    
    for district in districts:
        subset = var_data[(var_data['District'] == district) & 
                          (var_data['Increment'] == closest_inc)]
        if len(subset) > 0:
            temp_delta = subset['Avg_Temp_Delta_F'].values[0]
            var_change = subset['Avg_Var_Change'].values[0]
            
            standard_sensitivities.append({
                'Variable': VAR_DISPLAY.get(var, var),
                'District': district,
                'Temp_Delta': temp_delta,
                'Standard_Inc': closest_inc
            })

sens_df = pd.DataFrame(standard_sensitivities)
heatmap_data = sens_df.pivot(index='District', columns='Variable', values='Temp_Delta')

fig, ax = plt.subplots(figsize=(12, 6))

# Create heatmap
im = ax.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Temperature Change (°F)', fontsize=12)

# Add text annotations
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.iloc[i, j]
        text_color = 'white' if abs(val) > abs(heatmap_data.values).max() * 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=text_color, fontsize=12)

# Labels
ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns, fontsize=11)
ax.set_yticklabels(heatmap_data.index, fontsize=11)
ax.set_xlabel('Intervention Type', fontsize=12)
ax.set_ylabel('District', fontsize=12)
ax.set_title('Temperature Response Heatmap\n(at standard increment per variable)', 
            fontsize=14, fontweight='bold')

plt.tight_layout()
fig_path = OUTPUT_DIR / 'sensitivity_heatmap.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
print(f"   Saved: {fig_path} (+ SVG)")
plt.close()

# =============================================================================
# FIGURE 5: DOSE-RESPONSE CURVES BY DISTRICT (NORMALIZED)
# =============================================================================
print("\nCreating dose-response curves by district (normalized scale)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, district in enumerate(districts):
    ax = axes[idx]
    district_data = df[df['District'] == district]
    
    for var in variables:
        var_data = district_data[district_data['Variable'] == var].sort_values('Increment')
        
        if len(var_data) > 0:
            # Normalize X to 0-100% of max possible intervention
            max_inc = VAR_MAX_INCREMENT.get(var, var_data['Increment'].max())
            x_normalized = (var_data['Increment'].values / max_inc) * 100
            y = var_data['Avg_Temp_Delta_F'].values
            
            ax.plot(x_normalized, y, marker='o', markersize=6, label=VAR_DISPLAY.get(var, var), 
                   color=VAR_COLORS[var], linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Intervention Intensity (% of Maximum)', fontsize=11)
    ax.set_ylabel('Temp Change (°F)', fontsize=11)
    ax.set_title(f'{district} District', fontsize=13, fontweight='bold',
                color=DISTRICT_COLORS[district])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(0, 105)

plt.suptitle('Dose-Response Curves: Temperature Change vs Intervention Intensity', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIR / 'dose_response_curves_combined.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
print(f"   Saved: {fig_path} (+ SVG)")
plt.close()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("SENSITIVITY SUMMARY BY DISTRICT")
print("="*70)

# Print summary for each district
print("\nTemperature Change (°F) at Maximum Intervention:")
print("-"*70)

for district in districts:
    print(f"\n{district}:")
    district_data = df[df['District'] == district]
    
    for var in variables:
        var_data = district_data[district_data['Variable'] == var]
        if len(var_data) > 0:
            max_row = var_data.loc[var_data['Avg_Temp_Delta_F'].abs().idxmax()]
            print(f"  {VAR_DISPLAY.get(var, var):20s}: {max_row['Avg_Temp_Delta_F']:+.3f}°F "
                  f"(at {max_row['Avg_Var_Change']:+.1f} change)")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nFigures saved to: {OUTPUT_DIR}")
print("  1. sensitivity_curves_by_district.png - Main sensitivity curves")
print("  2. max_cooling_potential_by_district.png - Bar chart of max cooling")
print("  3. diminishing_returns_analysis.png - Marginal efficiency curves")
print("  4. sensitivity_heatmap.png - Heatmap comparison")
print("  5. dose_response_curves_combined.png - Combined dose-response")
