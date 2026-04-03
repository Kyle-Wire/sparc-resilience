#!/usr/bin/env python3
"""
GWEN Variable Selection Stage
Stage 1 of the GW3C Pipeline

This script performs geographically weighted variable selection using GWEN
and saves results for review before proceeding to spatial cross-validation.

Usage:
    python gwen_variable_selection.py [--config config_path]

Output Files:
    - gwen_results.json: Complete results for API access
    - gwen_variable_importance.csv: Detailed importance metrics
    - selected_features.txt: Simple list of selected features
    - pipeline_status.json: Stage completion status
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sparc.config.config import load_config
from sparc.data.data_utils import load_data, prepare_data
from sparc.models.gwen import GWENModel


def setup_output_directory(output_dir):
    """Create output directory and subdirectories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gwen'), exist_ok=True)
    return output_dir


def save_gwen_diagnostics(gwen_model, selected_features, feature_names, config, output_dir):
    """Save comprehensive GWEN diagnostics for reproducibility."""
    
    # Get local coefficients for stability analysis
    local_coefs = np.array([model.coef_ for model in gwen_model.local_models])
    
    # Compute additional diagnostics
    diagnostics = {
        'tuning_parameters': {
            'bandwidth_k': gwen_model.k_neighbors,
            'alpha_range': f"{gwen_model.global_model.alphas_.min():.4f} - {gwen_model.global_model.alphas_.max():.4f}",
            'selected_alpha': gwen_model.global_model.alpha_,
            'l1_ratio_range': config['gwen_params']['l1_ratios'],
            'selected_l1_ratio': gwen_model.global_model.l1_ratio_,
            'cv_folds': gwen_model.cv_folds,
            'sample_size': gwen_model.sample_size,
            'n_alphas': config['gwen_params']['n_alphas']
        },
        'stability_metrics': {},
        'selection_rule': {
            'threshold': config['gwen_params']['selection_threshold'],
            'criterion': 'selection_frequency >= threshold',
            'rationale': f'Variables selected in ≥{config["gwen_params"]["selection_threshold"]*100:.0f}% of local models considered spatially stable'
        }
    }
    
    # Compute coefficient sign stability for each feature
    for i, feature_name in enumerate(feature_names):
        feature_coefs = local_coefs[:, i]
        nonzero_coefs = feature_coefs[feature_coefs != 0]
        
        if len(nonzero_coefs) > 0:
            sign_consistency = np.mean(np.sign(nonzero_coefs) == np.sign(nonzero_coefs[0]))
            diagnostics['stability_metrics'][feature_name] = {
                'selection_frequency': np.mean(feature_coefs != 0),
                'mean_abs_coefficient': np.mean(np.abs(nonzero_coefs)),
                'coefficient_sign_stability': sign_consistency,
                'spatial_variability': np.std(np.abs(feature_coefs)),
                'coefficient_range': f"{nonzero_coefs.min():.4f} to {nonzero_coefs.max():.4f}"
            }
        else:
            diagnostics['stability_metrics'][feature_name] = {
                'selection_frequency': 0.0,
                'mean_abs_coefficient': 0.0,
                'coefficient_sign_stability': 0.0,
                'spatial_variability': 0.0,
                'coefficient_range': "Not selected"
            }
    
    # Save diagnostics
    diagnostics_file = os.path.join(output_dir, 'gwen_diagnostics.json')
    with open(diagnostics_file, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    return diagnostics


def create_gwen_diagnostic_plots(gwen_model, feature_names, output_dir):
    """Create diagnostic plots for GWEN variable selection."""
    
    # Get local coefficients
    local_coefs = np.array([model.coef_ for model in gwen_model.local_models])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GWEN Variable Selection Diagnostics', fontsize=16, fontweight='bold')
    
    # 1. Selection frequency by variable
    importance_df = gwen_model.get_variable_importance()
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(feature_names)), importance_df['selection_frequency'], 
                   alpha=0.7, color='steelblue')
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Selection threshold (10%)')
    ax1.set_xlabel('Variables', fontweight='bold')
    ax1.set_ylabel('Selection Frequency', fontweight='bold')
    ax1.set_title('Variable Selection Frequency Across Local Models', fontweight='bold')
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coefficient stability (box plot)
    ax2 = axes[0, 1]
    coef_data = []
    labels = []
    for i, name in enumerate(feature_names):
        nonzero_coefs = local_coefs[:, i][local_coefs[:, i] != 0]
        if len(nonzero_coefs) > 0:
            coef_data.append(nonzero_coefs)
            labels.append(name)
    
    if coef_data:
        bp = ax2.boxplot(coef_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
    ax2.set_xlabel('Variables', fontweight='bold')
    ax2.set_ylabel('Local Coefficient Values', fontweight='bold')
    ax2.set_title('Coefficient Distribution Across Space', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Alpha vs L1 ratio parameter space
    ax3 = axes[1, 0]
    alphas = [getattr(m, 'alpha_', m.alpha) for m in gwen_model.local_models]
    l1_ratios = [getattr(m, 'l1_ratio_', m.l1_ratio) for m in gwen_model.local_models]
    scatter = ax3.scatter(alphas, l1_ratios, alpha=0.6, c=range(len(alphas)), 
                         cmap='viridis', s=30)
    ax3.set_xlabel('Selected Alpha (log scale)', fontweight='bold')
    ax3.set_ylabel('Selected L1 Ratio', fontweight='bold')
    ax3.set_title('Parameter Selection Across Local Models', fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Model Index', fontweight='bold')
    
    # 4. Selection frequency vs coefficient magnitude
    ax4 = axes[1, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
    for i, name in enumerate(feature_names):
        sel_freq = importance_df.iloc[i]['selection_frequency']
        mean_coef = importance_df.iloc[i]['mean_local_coef']
        ax4.scatter(sel_freq, mean_coef, s=120, label=name, color=colors[i], 
                   alpha=0.8, edgecolors='black', linewidth=1)
    ax4.axvline(x=0.1, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='Selection threshold')
    ax4.set_xlabel('Selection Frequency', fontweight='bold')
    ax4.set_ylabel('Mean |Coefficient|', fontweight='bold')
    ax4.set_title('Selection Frequency vs Coefficient Magnitude', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'gwen_diagnostics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   [OK] Diagnostic plots saved: {plot_file}")
    return plot_file


def print_selection_rationale(diagnostics, selected_features, feature_names):
    """Print detailed selection rationale based on GWEN diagnostics."""
    
    print(f"\n🔍 GWEN SELECTION RATIONALE:")
    print(f"{'='*60}")
    
    # Tuning parameters
    tuning = diagnostics['tuning_parameters']
    print(f"\n📋 TUNING PARAMETERS:")
    print(f"   Bandwidth (k-NN): {tuning['bandwidth_k']} neighbors")
    print(f"   Alpha range: {tuning['alpha_range']}")
    print(f"   Selected α: {tuning['selected_alpha']:.4f}")
    print(f"   L1 ratio range: {tuning['l1_ratio_range']}")
    print(f"   Selected l1_ratio: {tuning['selected_l1_ratio']:.2f}")
    print(f"   Sample size: {tuning['sample_size'] or 'All samples'}")
    print(f"   CV folds: {tuning['cv_folds']}")
    
    # Selection rule
    rule = diagnostics['selection_rule']
    print(f"\n📏 SELECTION RULE:")
    print(f"   Criterion: {rule['criterion']}")
    print(f"   Threshold: {rule['threshold']}")
    print(f"   Rationale: {rule['rationale']}")
    
    # Stability analysis
    print(f"\n📊 VARIABLE STABILITY ANALYSIS:")
    print(f"   {'Variable':<15} {'Sel.Freq':<10} {'Sign.Stab':<10} {'Mean|Coef|':<12} {'Status':<10}")
    print(f"   {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    
    for feature_name in feature_names:
        if feature_name in diagnostics['stability_metrics']:
            metrics = diagnostics['stability_metrics'][feature_name]
            status = "SELECTED" if feature_name in selected_features else "DROPPED"
            print(f"   {feature_name:<15} "
                  f"{metrics['selection_frequency']:<10.3f} "
                  f"{metrics['coefficient_sign_stability']:<10.3f} "
                  f"{metrics['mean_abs_coefficient']:<12.4f} "
                  f"{status:<10}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Selection frequency: Fraction of local models selecting variable")
    print(f"   • Sign stability: Consistency of coefficient direction across space")
    print(f"   • Mean |Coef|: Average absolute coefficient magnitude when selected")
    
    # Summary of decision
    n_selected = len(selected_features)
    n_total = len(feature_names)
    print(f"\n🎯 SELECTION SUMMARY:")
    print(f"   • {n_selected}/{n_total} variables met spatial stability criteria")
    if n_selected == n_total:
        print(f"   • All physically-motivated predictors showed spatial stability")
        print(f"   • No evidence of problematic spatial instability")
        print(f"   • GWEN confirms retention of full predictor set")
    else:
        print(f"   • {n_total - n_selected} variables dropped due to spatial instability")
        print(f"   • GWEN identified spatially unstable predictors for exclusion")


def save_gwen_results(gwen_model, selected_features, feature_names, config, output_dir):
    """Save GWEN results in multiple formats for different use cases."""
    
    # Get variable importance
    importance_df = gwen_model.get_variable_importance()
    
    # Prepare comprehensive results
    gwen_results = {
        'timestamp': datetime.now().isoformat(),
        'total_features': len(feature_names),
        'selected_features': selected_features,
        'selection_threshold': config['gwen_params']['selection_threshold'],
        'model_params': {
            'k_neighbors': gwen_model.k_neighbors,
            'sample_size': gwen_model.sample_size,
            'cv_folds': gwen_model.cv_folds,
            'l1_ratios': config['gwen_params']['l1_ratios'],
            'n_alphas': config['gwen_params']['n_alphas']
        },
        'selection_summary': {
            'features_selected': len(selected_features),
            'features_total': len(feature_names),
            'selection_rate': len(selected_features) / len(feature_names)
        },
        'feature_importance': importance_df.to_dict('records')
    }
    
    # Save comprehensive JSON results
    results_file = os.path.join(output_dir, 'gwen_results.json')
    with open(results_file, 'w') as f:
        json.dump(gwen_results, f, indent=2)
    
    # Save importance table as CSV
    importance_file = os.path.join(output_dir, 'gwen_variable_importance.csv')
    importance_df.to_csv(importance_file, index=False)
    
    # Save selected features as simple text list
    features_file = os.path.join(output_dir, 'selected_features.txt')
    with open(features_file, 'w') as f:
        f.write('\n'.join(selected_features))
    
    # Save to gwen subdirectory for organization
    gwen_dir = os.path.join(output_dir, 'gwen')
    with open(os.path.join(gwen_dir, 'results.json'), 'w') as f:
        json.dump(gwen_results, f, indent=2)
    
    importance_df.to_csv(os.path.join(gwen_dir, 'importance.csv'), index=False)
    
    return results_file, importance_file, features_file


def update_pipeline_status(output_dir, status, details=None):
    """Update pipeline status tracking."""
    status_file = os.path.join(output_dir, 'pipeline_status.json')
    
    # Load existing status or create new
    pipeline_status = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                pipeline_status = json.load(f)
        except:
            pipeline_status = {}
    
    # Update GWEN stage status
    pipeline_status['gwen'] = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'needs_approval': status == 'completed_pending_review',
        'details': details or {}
    }
    
    # Save updated status
    with open(status_file, 'w') as f:
        json.dump(pipeline_status, f, indent=2)


def check_prerequisites(config):
    """Check if all prerequisites are met for GWEN stage."""
    data_file = config['data']['file_path']
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"[OK] Data file found: {data_file}")
    return True


def print_results_summary(selected_features, feature_names, importance_df, output_dir):
    """Print a comprehensive summary of GWEN results."""
    print(f"\n{'='*60}")
    print(f"🎯 GWEN VARIABLE SELECTION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\n📊 SELECTION SUMMARY:")
    print(f"   Total Features Available: {len(feature_names)}")
    print(f"   Features Selected: {len(selected_features)}")
    print(f"   Selection Rate: {len(selected_features)/len(feature_names):.1%}")
    
    print(f"\n🔝 TOP 10 SELECTED FEATURES:")
    selected_importance = importance_df[importance_df['feature'].isin(range(len(selected_features)))]
    for i, (_, row) in enumerate(selected_importance.head(10).iterrows()):
        feature_idx = int(row['feature'])
        if feature_idx < len(feature_names):
            feature_name = feature_names[feature_idx]
            print(f"   {i+1:2d}. {feature_name:<30} (freq: {row['selection_frequency']:.3f})")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"   📄 {os.path.join(output_dir, 'gwen_results.json')}")
    print(f"   📊 {os.path.join(output_dir, 'gwen_variable_importance.csv')}")
    print(f"   📝 {os.path.join(output_dir, 'selected_features.txt')}")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. 🔍 Review GWEN results in frontend: /pipeline/gwen")
    print(f"   2. [OK] Approve selection to proceed to spatial CV")
    print(f"   3. 🚀 Run spatial_cv_v2.py after approval")


# ── Phase 2 enhancements ─────────────────────────────────────────────

def _make_spatial_folds(coords, n_splits=5, block_size=None):
    """Create spatial block CV folds for GWEN."""
    from sparc.run.enhanced_spatial_cv import spatial_kfold_enhanced
    dummy_X = np.zeros((len(coords), 1))
    dummy_y = np.zeros(len(coords))
    return spatial_kfold_enhanced(
        dummy_X, dummy_y, np.asarray(coords),
        n_splits=n_splits, block_size=block_size, method='block',
    )


def compute_stability_scores(gwen_model, X, y, coords, feature_names, folds=None,
                              n_folds=5, block_size=None, threshold=0.1):
    """Run GWEN on each spatial CV fold and compute selection stability.

    Returns a DataFrame with per-feature stability metrics.
    """
    if folds is None:
        folds = _make_spatial_folds(coords, n_splits=n_folds, block_size=block_size)

    n_features = np.asarray(X).shape[1]
    fold_selected = np.zeros((len(folds), n_features))  # 1 if selected in fold
    fold_signs = np.full((len(folds), n_features), np.nan)
    fold_ranks = np.full((len(folds), n_features), np.nan)

    for fi, (train_idx, _) in enumerate(folds):
        coefs = gwen_model.fit_on_subset(X, y, coords, train_idx)
        sel_freq = np.mean(coefs != 0, axis=0)
        mean_coef = np.mean(coefs, axis=0)

        fold_selected[fi] = (sel_freq >= threshold).astype(float)
        fold_signs[fi] = np.sign(mean_coef)
        # Rank by absolute mean coefficient (1 = most important)
        abs_mean = np.mean(np.abs(coefs), axis=0)
        ranks = np.empty_like(abs_mean)
        order = np.argsort(-abs_mean)
        ranks[order] = np.arange(1, len(abs_mean) + 1)
        fold_ranks[fi] = ranks

    # Aggregate metrics
    selection_stability = np.mean(fold_selected, axis=0)
    sign_consistency = np.zeros(n_features)
    for j in range(n_features):
        signs_j = fold_signs[:, j]
        valid = ~np.isnan(signs_j)
        if valid.any():
            dominant = np.sign(np.nansum(signs_j))
            sign_consistency[j] = np.mean(signs_j[valid] == dominant) if dominant != 0 else 0.5

    # Kendall rank stability (average pairwise tau)
    rank_stability = np.zeros(n_features)
    from scipy.stats import kendalltau
    n_f = len(folds)
    for j in range(n_features):
        taus = []
        for a in range(n_f):
            for b in range(a + 1, n_f):
                tau, _ = kendalltau(fold_ranks[a], fold_ranks[b])
                taus.append(tau)
        rank_stability[j] = np.nanmean(taus) if taus else 0.0

    rows = []
    for j in range(n_features):
        rows.append({
            'feature': feature_names[j] if j < len(feature_names) else f'X{j}',
            'selection_stability': selection_stability[j],
            'sign_consistency': sign_consistency[j],
            'rank_stability': rank_stability[j],
        })

    return pd.DataFrame(rows)


def auto_suggest_gwen_params(config, correlogram_dir=None):
    """Suggest GWEN hyperparameters from correlogram results and data.

    Returns a dict that can be merged into ``gwen_params``.
    """
    suggestions = {}
    corr_path = None
    if correlogram_dir:
        corr_path = os.path.join(correlogram_dir, 'correlogram_analysis_results.json')
    if corr_path and os.path.exists(corr_path):
        with open(corr_path) as f:
            corr = json.load(f)
        # Extract median autocorrelation range across variables
        ranges = []
        for var_result in corr.get('variable_results', {}).values():
            r = var_result.get('effective_range') or var_result.get('optimal_bandwidth')
            if r:
                ranges.append(r)
        if ranges:
            median_range = float(np.median(ranges))
            # k_neighbors ~ number of points within autocorrelation range
            # (rough heuristic: 30% of dataset per range circle)
            suggestions['_autocorrelation_range_m'] = median_range

    # VIF-based l1_ratio suggestion
    try:
        data_path = config.get('data', {}).get('file_path')
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path, nrows=5000)
            predictors = config.get('predictors', {}).get('base_model', [])
            predictors = [p for p in predictors if p in df.columns]
            if len(predictors) >= 2:
                from numpy.linalg import cond
                X_check = df[predictors].dropna().values
                if len(X_check) > 50:
                    cn = cond(X_check)
                    if cn > 30:
                        suggestions['l1_ratios'] = [0.7, 0.8, 0.9, 0.95, 0.99]
                        suggestions['_condition_number'] = float(cn)
    except Exception:
        pass

    return suggestions


def generate_gwen_html_report(gwen_model, feature_names, selected_features,
                               stability_df, diagnostics, output_dir):
    """Generate a standalone HTML report summarising GWEN variable selection."""
    import html as _html

    local_coefs = np.array([m.coef_ for m in gwen_model.local_models])
    importance_df = gwen_model.get_variable_importance()

    # Build table rows
    table_rows = ""
    for j, fname in enumerate(feature_names):
        sel_freq = float(importance_df.iloc[j]['selection_frequency']) if j < len(importance_df) else 0
        mean_coef = float(importance_df.iloc[j]['mean_local_coef']) if j < len(importance_df) else 0
        stab_row = stability_df[stability_df['feature'] == fname]
        sel_stab = f"{float(stab_row['selection_stability'].iloc[0]):.2f}" if len(stab_row) else "—"
        sign_con = f"{float(stab_row['sign_consistency'].iloc[0]):.2f}" if len(stab_row) else "—"
        status = "SELECTED" if fname in selected_features else "DROPPED"
        bg = "#e8f5e9" if status == "SELECTED" else "#ffebee"
        table_rows += (
            f'<tr style="background:{bg}">'
            f'<td>{_html.escape(fname)}</td>'
            f'<td>{sel_freq:.3f}</td><td>{mean_coef:.4f}</td>'
            f'<td>{sel_stab}</td><td>{sign_con}</td>'
            f'<td><b>{status}</b></td></tr>\n'
        )

    # Correlation matrix of selected features (as HTML table)
    corr_html = ""
    if len(selected_features) >= 2 and local_coefs.shape[1] == len(feature_names):
        sel_idx = [feature_names.index(f) for f in selected_features if f in feature_names]
        coef_sub = local_coefs[:, sel_idx]
        corr = np.corrcoef(coef_sub.T)
        corr_html = '<h2>Coefficient Correlation Matrix (Selected Features)</h2><table border="1" cellpadding="4"><tr><th></th>'
        for f in selected_features:
            corr_html += f'<th>{_html.escape(f)}</th>'
        corr_html += '</tr>'
        for i, f in enumerate(selected_features):
            corr_html += f'<tr><th>{_html.escape(f)}</th>'
            for jj in range(len(selected_features)):
                v = corr[i, jj]
                color = "#c8e6c9" if abs(v) < 0.5 else "#ffcdd2"
                corr_html += f'<td style="background:{color}">{v:.2f}</td>'
            corr_html += '</tr>'
        corr_html += '</table>'

    report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GWEN Variable Selection Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background: #37474f; color: white; }}
h1 {{ color: #1565c0; }}
h2 {{ color: #37474f; margin-top: 30px; }}
.summary {{ background: #e3f2fd; padding: 16px; border-radius: 8px; margin: 20px 0; }}
</style></head><body>
<h1>GWEN Variable Selection Report</h1>
<div class="summary">
<b>Selected:</b> {len(selected_features)} / {len(feature_names)} features &nbsp;|&nbsp;
<b>Threshold:</b> {diagnostics.get('selection_rule', {}).get('threshold', '—')} &nbsp;|&nbsp;
<b>K-neighbors:</b> {gwen_model.k_neighbors} &nbsp;|&nbsp;
<b>Alpha:</b> {gwen_model.global_model.alpha_:.4f}
</div>

<h2>Feature Selection Summary</h2>
<table>
<tr><th>Feature</th><th>Selection Freq</th><th>Mean |Coef|</th>
<th>Stability</th><th>Sign Consistency</th><th>Status</th></tr>
{table_rows}
</table>

{corr_html}

<h2>Tuning Parameters</h2>
<pre>{json.dumps(diagnostics.get('tuning_parameters', {}), indent=2)}</pre>
</body></html>"""

    report_path = os.path.join(output_dir, 'gwen_selection_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   [OK] HTML report saved: {report_path}")
    return report_path


def main(config_path=None, fast_mode=False):
    """Main GWEN variable selection workflow."""
    print("🔍 GW3C Pipeline - Stage 1: GWEN Variable Selection")
    print("=" * 60)
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        config = load_config(config_path)
        print(f"   [OK] Configuration loaded")
        
        # Setup output directory
        output_dir = config['output']['base_dir']
        setup_output_directory(output_dir)
        print(f"   [OK] Output directory: {output_dir}")
        
        # Check prerequisites
        print("\n2. Checking prerequisites...")
        check_prerequisites(config)
        
        # Load and prepare data
        print("\n3. Loading and preparing data...")
        df = load_data(config['data']['file_path'])
        X, y, coords, feature_names = prepare_data(df, config)
        
        print(f"   Dataset shape: {X.shape}")
        print(f"   Target variable: {config['data']['target_column']}")
        print(f"   Coordinate columns: {config['data']['coord_columns']}")
        print(f"   Total features: {len(feature_names)}")
        
        # Initialize GWEN model
        print("\n4. Initializing GWEN model...")
        gwen_params = config['gwen_params']

        # Auto-suggest params from correlogram if available
        gwen_cfg = config.get('gwen', {})
        if gwen_cfg.get('auto_tune', True):
            from sparc.run.pipeline_paths import get_paths as _get_paths
            try:
                _paths = _get_paths()
                stage1_dir = str(_paths.stage1_dir)
            except Exception:
                stage1_dir = os.path.join(output_dir, 'Stage_1_Correlogram_Analysis')
            suggestions = auto_suggest_gwen_params(config, correlogram_dir=stage1_dir)
            if suggestions:
                # Merge suggestions (user config takes precedence)
                for k, v in suggestions.items():
                    if not k.startswith('_') and k not in gwen_params:
                        gwen_params[k] = v
                        print(f"   Auto-tuned {k} = {v}")
                if '_condition_number' in suggestions:
                    print(f"   High multicollinearity detected (cond={suggestions['_condition_number']:.0f}), using L1-heavy ratios")

        # Build spatial block folds if spatial_cv is enabled
        spatial_folds = None
        use_spatial = gwen_cfg.get('spatial_cv', False)  # default: False — standard k-fold is sufficient for screening
        if use_spatial and not fast_mode:
            try:
                print("   Building spatial block CV folds for GWEN...")
                # Inherit block_size from models.spatial_cv if not set in gwen config
                gwen_block = gwen_cfg.get('block_size')
                if gwen_block is None:
                    gwen_block = config.get('models', {}).get('spatial_cv', {}).get('block_size')
                    if gwen_block is not None:
                        print(f"   Using shared block size from spatial_cv config: {gwen_block}m")
                spatial_folds = _make_spatial_folds(
                    coords, n_splits=gwen_params.get('cv_folds', 5),
                    block_size=gwen_block,
                )
            except Exception as e:
                print(f"   Warning: Could not create spatial folds ({e}), falling back to standard CV")

        local_cv = gwen_cfg.get('local_cv', False)  # default: False — use global alpha for local models
        gwen = GWENModel(
            k_neighbors=gwen_params['k_neighbors'],
            sample_size=gwen_params.get('sample_size'),
            cv_folds=gwen_params['cv_folds'],
            l1_ratios=gwen_params['l1_ratios'],
            n_alphas=gwen_params['n_alphas'],
            quick_mode=fast_mode,
            n_jobs=-1 if fast_mode else 1,
            spatial_cv_folds=spatial_folds,
            local_cv=local_cv,
        )
        
        print(f"   K-neighbors: {gwen.k_neighbors}")
        print(f"   Sample size: {gwen.sample_size or 'All samples'}")
        print(f"   CV folds: {gwen.cv_folds}")
        
        # Fit GWEN model
        print("\n5. Fitting GWEN model...")
        print("   This may take several minutes depending on data size...")
        gwen.fit(X, y, coords)
        print("   [OK] GWEN model fitted successfully")
        
        # Get variable importance and selected features  
        print("\n6. Computing variable importance and selecting features...")
        importance_df = gwen.get_variable_importance()
        selected_features = gwen.get_selected_predictors(
            feature_names=feature_names,
            threshold=gwen_params['selection_threshold']
        )
        
        # Generate comprehensive diagnostics
        print("\n7. Generating comprehensive diagnostics...")
        diagnostics = save_gwen_diagnostics(gwen, selected_features, feature_names, config, output_dir)
        
        # Create diagnostic plots
        create_gwen_diagnostic_plots(gwen, feature_names, output_dir)
        
        print(f"   [OK] Variable importance computed")
        print(f"   [OK] Feature selection completed")
        print(f"   [OK] Diagnostics generated and saved")

        # Stability scoring (skip in fast mode; disabled by default)
        stability_df = pd.DataFrame()
        stability_folds = gwen_cfg.get('stability_folds', 0)  # default: 0 — disabled; overkill for screening
        if not fast_mode and stability_folds > 0:
            print("\n7b. Computing stability scores across spatial folds...")
            try:
                stability_df = compute_stability_scores(
                    gwen, X, y, coords, feature_names,
                    folds=spatial_folds,
                    n_folds=stability_folds,
                    threshold=gwen_params['selection_threshold'],
                )
                stab_path = os.path.join(output_dir, 'gwen_stability.csv')
                stability_df.to_csv(stab_path, index=False)
                print(f"   [OK] Stability scores saved: {stab_path}")
            except Exception as stab_e:
                print(f"   Warning: Stability scoring failed ({stab_e}), continuing without")

        # HTML report
        print("\n7c. Generating HTML selection report...")
        try:
            generate_gwen_html_report(
                gwen, feature_names, selected_features,
                stability_df, diagnostics, output_dir,
            )
        except Exception as rpt_e:
            print(f"   Warning: HTML report generation failed ({rpt_e})")

        # Save results with enhanced documentation
        print("\n8. Saving GWEN results...")
        results_file, importance_file, features_file = save_gwen_results(
            gwen, selected_features, feature_names, config, output_dir
        )
        
        print(f"   [OK] Results saved to multiple formats")
        
        # Print detailed selection rationale
        print_selection_rationale(diagnostics, selected_features, feature_names)
        
        # Update pipeline status
        update_pipeline_status(output_dir, 'completed_pending_review', {
            'features_selected': len(selected_features),
            'features_total': len(feature_names),
            'results_file': results_file
        })
        
        # Print comprehensive summary
        print_results_summary(selected_features, feature_names, importance_df, output_dir)
        
        # Check if approval already exists (for pipeline automation)
        approval_file = os.path.join(output_dir, 'gwen_approved.txt')
        if os.path.exists(approval_file):
            print(f"\n[OK] GWEN selection already approved")
            print(f"   Pipeline can proceed to next stage")
            update_pipeline_status(output_dir, 'approved')
            return True
        else:
            print(f"\n[PAUSE]  PIPELINE PAUSED FOR REVIEW")
            print(f"   Create 'gwen_approved.txt' in output directory to proceed")
            print(f"   Or approve via frontend interface at /pipeline/gwen")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR in GWEN stage: {str(e)}")
        # Create fallback output directory for error logging
        fallback_dir = './output'
        os.makedirs(fallback_dir, exist_ok=True)
        update_pipeline_status(output_dir if 'output_dir' in locals() else fallback_dir, 'failed', {
            'error': str(e)
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GWEN variable selection stage')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    success = main(args.config)
    sys.exit(0 if success else 1) 