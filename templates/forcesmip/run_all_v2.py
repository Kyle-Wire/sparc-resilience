"""
Run all ForceSMIP v2 pipelines (TAS, PR, PSL) + fingerprint post-processing.
Stages 2, 3, 4 sequentially for each variable.
"""
import subprocess, sys, os, time, json, glob
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from models.fingerprint import FingerprintProjector

VARS = ['tas', 'pr', 'psl']
BASE = '.'

def run_stage(var, stage):
    yml = f'pooled_{var}_v2_project.yml'
    cmd = [sys.executable, '-m', 'sparc', 'run', '--project', yml, '--stage', str(stage), '--skip-gwen']
    print(f"\n{'='*80}")
    print(f"  RUNNING: {var.upper()} Stage {stage}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"\n>>> {var.upper()} Stage {stage} finished in {elapsed/60:.1f} min (exit code {result.returncode})")
    return result.returncode

def apply_fingerprint(var):
    """Apply same-model + LOO MMM fingerprinting post-hoc."""
    print(f"\n{'='*80}")
    print(f"  FINGERPRINTING: {var.upper()}")
    print(f"{'='*80}\n")

    data_csv = f'data/forcesmip_{var}_pooled.csv'
    s2_dir = f'output/forcesmip_{var}_pooled_v2/Stage_2_Spatial_CV'
    oof_csv = os.path.join(s2_dir, 'optimized_oof_predictions.csv')
    ens_csv = os.path.join(s2_dir, 'final_ensemble_predictions.csv')

    if not os.path.exists(oof_csv):
        print(f"  SKIP — OOF predictions not found: {oof_csv}")
        return

    df = pd.read_csv(data_csv)
    oof = pd.read_csv(oof_csv)
    ens = pd.read_csv(ens_csv)

    target_col = f'{var}_trend'
    forced_col = f'{var}_trend_forced'

    # Align predictions
    df['pred_ols'] = oof['ols'].values
    df['pred_gwr'] = oof['gwr'].values
    df['pred_gwrf'] = oof['gwrf'].values
    df['pred_ggpgam'] = oof['ggpgam'].values
    df['pred_meta'] = ens['meta_ensemble_standard'].values

    pred_cols = ['pred_ols', 'pred_gwr', 'pred_gwrf', 'pred_ggpgam', 'pred_meta']

    # ── Same-model fingerprint ──────────────────
    valid = df[forced_col].notna()
    dv = df[valid].reset_index(drop=True)
    n_dropped = (~valid).sum()
    print(f"  Valid rows: {dv.shape[0]} (dropped {n_dropped} NaN forced)")

    truth_forced = dv[forced_col].values
    truth_total = dv[target_col].values
    lats = dv['lat'].values

    fp = FingerprintProjector()
    fp.from_ensemble_mean(truth_forced)

    def pat_corr(a, b, w=None):
        mask = ~(np.isnan(a)|np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 3: return np.nan
        if w is not None:
            ww = np.cos(np.radians(w[mask]))
            a, b = a*np.sqrt(ww), b*np.sqrt(ww)
        return float(np.corrcoef(a, b)[0,1])

    def amp_rat(a, b):
        mask = ~(np.isnan(a)|np.isnan(b))
        return float(np.std(a[mask])/np.std(b[mask])) if np.std(b[mask]) > 0 else np.nan

    def r2(p, t):
        mask = ~(np.isnan(p)|np.isnan(t))
        ss_r = np.sum((t[mask]-p[mask])**2)
        ss_t = np.sum((t[mask]-np.mean(t[mask]))**2)
        return float(1-ss_r/ss_t) if ss_t > 0 else np.nan

    def nrmse(p, t):
        mask = ~(np.isnan(p)|np.isnan(t))
        return float(np.sqrt(np.mean((p[mask]-t[mask])**2))/np.std(t[mask])) if np.std(t[mask]) > 0 else np.nan

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {}

    print(f"\n  {'Model':<10} {'alpha':>6} {'R2_forced':>10} {'R2_total':>10} {'PatCorr_f':>10} {'AmpRat_f':>9}")
    print(f"  {'-'*60}")

    for col in pred_cols:
        raw = dv[col].values
        name = col.replace('pred_', '')
        for alpha in alphas:
            bl = fp.blend(raw, alpha=alpha)
            r2f = r2(bl, truth_forced)
            r2t = r2(bl, truth_total)
            pcf = pat_corr(bl, truth_forced)
            arf = amp_rat(bl, truth_forced)

            label = name if alpha == 0.0 else f"  +fp({alpha})"
            print(f"  {label:<10} {alpha:>6.2f} {r2f:>10.4f} {r2t:>10.4f} {pcf:>10.4f} {arf:>9.4f}")

            results[(name, alpha)] = {
                'r2_forced': round(r2f, 4), 'r2_total': round(r2t, 4),
                'pattern_corr_forced': round(pcf, 4), 'amplitude_ratio_forced': round(arf, 4),
            }
        print()

    # Save
    out_path = os.path.join(s2_dir, 'benchmark_metrics_fingerprinted.json')
    best_per_model = {}
    for col in pred_cols:
        name = col.replace('pred_', '')
        best_a, best_s = 0.0, -999
        for alpha in alphas:
            r = results[(name, alpha)]
            score = 0.5*r['r2_total'] + 0.3*r['pattern_corr_forced'] + 0.2*(1-abs(1-r['amplitude_ratio_forced']))
            if score > best_s:
                best_s = score
                best_a = alpha
        best_per_model[name] = best_a

    out = {
        'variable': var,
        'fingerprint_source': f'ensemble_mean ({forced_col})',
        'optimal_alphas': best_per_model,
        'models': {},
    }
    for name in ['gwrf', 'meta']:
        a = best_per_model[name]
        r = results[(name, a)]
        out['models'][name] = {'optimal_alpha': a, **r}
        print(f"  BEST {name}: alpha={a}, R2f={r['r2_forced']}, PatCorr={r['pattern_corr_forced']}")

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")

    # ── LOO Multi-Model Mean ────────────────────
    ensmean_files = sorted(glob.glob(f'data/forcesmip_ensmean_{var}_*.csv'))
    if ensmean_files:
        model_forced = {}
        for ef in ensmean_files:
            tag = ef.split('_')[-1].replace('.csv', '')
            edf = pd.read_csv(ef).set_index('grid_cell_id')
            model_forced[tag] = edf[forced_col]

        members = sorted(df['member'].unique())
        models_with_ensmean = set(model_forced.keys())

        loo_results = []
        for member in members:
            mrows = df[df.member == member].copy()
            gids = mrows['grid_cell_id'].values
            others = [m for m in models_with_ensmean if m != member]
            if member not in models_with_ensmean:
                others = list(models_with_ensmean)

            loo_pattern = np.full(len(gids), np.nan)
            for i, gid in enumerate(gids):
                vals = [model_forced[om].get(gid, np.nan) for om in others]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    loo_pattern[i] = np.mean(vals)

            v2 = ~np.isnan(loo_pattern) & mrows[forced_col].notna().values
            if v2.sum() < 100:
                continue

            fp_loo = FingerprintProjector()
            fp_loo.from_ensemble_mean(loo_pattern[v2])

            tf = mrows.loc[v2, forced_col].values
            tt = mrows.loc[v2, target_col].values

            for col in pred_cols:
                raw = mrows.loc[v2, col].values
                name = col.replace('pred_', '')
                for alpha in [0.0, 0.3, 0.5]:
                    bl = fp_loo.blend(raw, alpha=alpha)
                    loo_results.append({
                        'member': member, 'model': name, 'alpha': alpha,
                        'r2_forced': r2(bl, tf), 'r2_total': r2(bl, tt),
                        'pat_corr_forced': pat_corr(bl, tf),
                        'amp_ratio_forced': amp_rat(bl, tf),
                    })

        if loo_results:
            ldf = pd.DataFrame(loo_results)
            print(f"\n  LOO MMM Fingerprint (averaged across members):")
            for name in ['gwrf', 'meta']:
                for alpha in [0.0, 0.3, 0.5]:
                    sub = ldf[(ldf.model == name) & (ldf.alpha == alpha)]
                    print(f"    {name} alpha={alpha}: R2f={sub.r2_forced.mean():.4f}, "
                          f"R2t={sub.r2_total.mean():.4f}, "
                          f"PatCorr={sub.pat_corr_forced.mean():.4f}")

            loo_path = os.path.join(s2_dir, 'benchmark_metrics_loo_mmm_fingerprint.json')
            loo_out = {'variable': var, 'method': 'leave_one_out_multi_model_mean', 'models': {}}
            for name in ['gwrf', 'meta']:
                loo_out['models'][name] = {}
                for alpha in [0.0, 0.3, 0.5]:
                    sub = ldf[(ldf.model == name) & (ldf.alpha == alpha)]
                    loo_out['models'][name][f'alpha_{alpha}'] = {
                        'r2_forced': round(sub.r2_forced.mean(), 4),
                        'r2_total': round(sub.r2_total.mean(), 4),
                        'pattern_corr_forced': round(sub.pat_corr_forced.mean(), 4),
                        'amplitude_ratio_forced': round(sub.amp_ratio_forced.mean(), 4),
                    }
            with open(loo_path, 'w') as f:
                json.dump(loo_out, f, indent=2)
            print(f"  Saved: {loo_path}")


if __name__ == '__main__':
    # Check which vars still need Stage 2
    for var in VARS:
        s2_dir = f'output/forcesmip_{var}_pooled_v2/Stage_2_Spatial_CV'
        oof_csv = os.path.join(s2_dir, 'optimized_oof_predictions.csv')
        if os.path.exists(oof_csv):
            print(f">>> {var.upper()}: Stage 2 already done, skipping.")
        else:
            rc = run_stage(var, 2)
            if rc != 0:
                print(f">>> {var.upper()} Stage 2 FAILED (exit {rc}), stopping.")
                sys.exit(1)

        # Stage 3
        s3_dir = f'output/forcesmip_{var}_pooled_v2/Stage_3_Causal_Validation'
        s3_done = os.path.join(s3_dir, 'scenario_coefficients.json')
        if os.path.exists(s3_done):
            print(f">>> {var.upper()}: Stage 3 already done, skipping.")
        else:
            rc = run_stage(var, 3)
            if rc != 0:
                print(f">>> {var.upper()} Stage 3 FAILED (exit {rc}), continuing.")

        # Stage 4
        s4_dir = f'output/forcesmip_{var}_pooled_v2/Stage_4_Scenarios'
        s4_done = os.path.join(s4_dir, 'scenario_summary_dag.csv')
        if os.path.exists(s4_done):
            print(f">>> {var.upper()}: Stage 4 already done, skipping.")
        else:
            rc = run_stage(var, 4)
            if rc != 0:
                print(f">>> {var.upper()} Stage 4 FAILED (exit {rc}), continuing.")

        # Fingerprinting
        apply_fingerprint(var)

    print("\n\n" + "=" * 80)
    print("  ALL VARIABLES COMPLETE")
    print("=" * 80)
