"""Quick fingerprint analysis for PSL (base models only since meta ensemble csv missing)."""
import pandas as pd, numpy as np, json, os, glob
import importlib.util

spec = importlib.util.spec_from_file_location('fingerprint', '../../models/fingerprint.py')
fp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp_mod)
FingerprintProjector = fp_mod.FingerprintProjector

var = 'psl'
forced_col = f'{var}_trend_forced'
target_col = f'{var}_trend'
s2_dir = f'output/forcesmip_{var}_pooled_v2/Stage_2_Spatial_CV'

df = pd.read_csv(f'data/forcesmip_{var}_pooled.csv')
oof = pd.read_csv(os.path.join(s2_dir, 'optimized_oof_predictions.csv'))

df['pred_ols'] = oof['ols'].values
df['pred_gwr'] = oof['gwr'].values
df['pred_gwrf'] = oof['gwrf'].values
df['pred_ggpgam'] = oof['ggpgam'].values

# Generate meta from LightGBM text model
import lightgbm as lgb
meta_txt = os.path.join(s2_dir, 'optimized_meta_model_run1.txt')
meta_feats = oof[['ols', 'gwr', 'gwrf', 'ggpgam']].values
try:
    booster = lgb.Booster(model_file=meta_txt)
    df['pred_meta'] = booster.predict(meta_feats)
    print(f'Meta predictions generated from LightGBM text model')
except Exception as e:
    df['pred_meta'] = np.nan
    print(f'Could not generate meta predictions: {e}')

pred_cols = ['pred_ols', 'pred_gwr', 'pred_gwrf', 'pred_ggpgam', 'pred_meta']
valid = df[forced_col].notna()
dv = df[valid].reset_index(drop=True)
print(f'Valid rows: {dv.shape[0]} (dropped {(~valid).sum()} NaN)')

truth_forced = dv[forced_col].values
truth_total = dv[target_col].values

fp = FingerprintProjector()
fp.from_ensemble_mean(truth_forced)

def r2(p, t):
    m = ~(np.isnan(p)|np.isnan(t))
    ss_r = np.sum((t[m]-p[m])**2)
    ss_t = np.sum((t[m]-np.mean(t[m]))**2)
    return float(1-ss_r/ss_t) if ss_t > 0 else np.nan

def pat_corr(a, b):
    m = ~(np.isnan(a)|np.isnan(b))
    a, b = a[m], b[m]
    return float(np.corrcoef(a,b)[0,1]) if len(a) > 2 else np.nan

def amp_rat(a, b):
    m = ~(np.isnan(a)|np.isnan(b))
    return float(np.std(a[m])/np.std(b[m])) if np.std(b[m]) > 0 else np.nan

alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
results = {}

for col in pred_cols:
    raw = dv[col].values
    name = col.replace('pred_', '')
    if np.all(np.isnan(raw)):
        print(f'{name}: all NaN, skip')
        continue
    for alpha in alphas:
        bl = fp.blend(raw, alpha=alpha)
        r2f = r2(bl, truth_forced)
        r2t = r2(bl, truth_total)
        pcf = pat_corr(bl, truth_forced)
        arf = amp_rat(bl, truth_forced)
        print(f'{name:>8} a={alpha:.1f}: R2f={r2f:.4f} R2t={r2t:.4f} PC={pcf:.4f} AR={arf:.4f}')
        results[(name, alpha)] = {
            'r2_forced': round(r2f, 4), 'r2_total': round(r2t, 4),
            'pattern_corr_forced': round(pcf, 4), 'amplitude_ratio_forced': round(arf, 4),
        }
    print()

# Save same-model fingerprint
best_per_model = {}
for col in pred_cols:
    name = col.replace('pred_', '')
    if np.all(np.isnan(dv[col].values)):
        continue
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
    if name not in best_per_model:
        continue
    a = best_per_model[name]
    r = results[(name, a)]
    out['models'][name] = {'optimal_alpha': a, **r}
    print(f'BEST {name}: alpha={a} R2f={r["r2_forced"]} PatCorr={r["pattern_corr_forced"]}')

with open(os.path.join(s2_dir, 'benchmark_metrics_fingerprinted.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('Saved fingerprinted metrics')

# LOO MMM
ensmean_files = sorted(glob.glob(f'data/forcesmip_ensmean_{var}_*.csv'))
print(f'\nFound {len(ensmean_files)} ensemble-mean files for LOO MMM')
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
            raw_m = mrows.loc[v2, col].values
            if np.all(np.isnan(raw_m)):
                continue
            name = col.replace('pred_', '')
            for alpha in [0.0, 0.3, 0.5]:
                bl = fp_loo.blend(raw_m, alpha=alpha)
                loo_results.append({
                    'member': member, 'model': name, 'alpha': alpha,
                    'r2_forced': r2(bl, tf), 'r2_total': r2(bl, tt),
                    'pat_corr_forced': pat_corr(bl, tf),
                    'amp_ratio_forced': amp_rat(bl, tf),
                })
    
    if loo_results:
        ldf = pd.DataFrame(loo_results)
        print('\nLOO MMM (averaged):')
        for name in ['gwrf', 'meta']:
            for alpha in [0.0, 0.3, 0.5]:
                sub = ldf[(ldf.model == name) & (ldf.alpha == alpha)]
                if len(sub) == 0:
                    continue
                print(f'  {name} a={alpha}: R2f={sub.r2_forced.mean():.4f} R2t={sub.r2_total.mean():.4f} PC={sub.pat_corr_forced.mean():.4f} AR={sub.amp_ratio_forced.mean():.4f}')
        
        loo_out = {'variable': var, 'method': 'leave_one_out_multi_model_mean', 'models': {}}
        for name in ['gwrf', 'meta']:
            loo_out['models'][name] = {}
            for alpha in [0.0, 0.3, 0.5]:
                sub = ldf[(ldf.model == name) & (ldf.alpha == alpha)]
                if len(sub) == 0:
                    continue
                loo_out['models'][name][f'alpha_{alpha}'] = {
                    'r2_forced': round(sub.r2_forced.mean(), 4),
                    'r2_total': round(sub.r2_total.mean(), 4),
                    'pattern_corr_forced': round(sub.pat_corr_forced.mean(), 4),
                    'amplitude_ratio_forced': round(sub.amp_ratio_forced.mean(), 4),
                }
        with open(os.path.join(s2_dir, 'benchmark_metrics_loo_mmm_fingerprint.json'), 'w') as f:
            json.dump(loo_out, f, indent=2)
        print('Saved LOO MMM fingerprint metrics')

print('\nDONE')
