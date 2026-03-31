import numpy as np, pandas as pd, geopandas as gpd
from sklearn.linear_model import LinearRegression

# 1) OLS coefficients
data = pd.read_csv('data/brown4.csv')
features = ['Distance_from_water_m','Pct_Impervious','Pct_Canopy','NDVI','Albedo','Elevation_m']
target = 'AAT_z'
lr = LinearRegression()
lr.fit(data[features], data[target])
print('=== OLS coefficients ===')
for f, c in zip(features, lr.coef_):
    print(f'  {f:25s}: {c:+.6f}')
print(f'  R2 = {lr.score(data[features], data[target]):.4f}')

# 2) Latitude from GPKG geometry
gdf = gpd.read_file('brown_uhi_run/output/Stage_4_Scenarios/scenario_results_reprediction.gpkg')
lat = gdf.geometry.y
vals = gdf['repred_delta_Pct_Canopy_plus_25'].values
lat_dec = pd.qcut(lat, 10, labels=False, duplicates='drop')
print('\n=== Canopy +25pp delta by latitude decile (S to N) ===')
for d in sorted(lat_dec.unique()):
    mask = (lat_dec == d).values
    if mask.sum() == 0:
        continue
    v = vals[mask]
    la = lat.values[mask]
    warm_pct = 100 * np.mean(v > 0)
    cool_pct = 100 * np.mean(v < 0)
    print(f'  Lat {d} ({la.min():.0f}-{la.max():.0f}): mean={v.mean():+.5f}  warm={warm_pct:.0f}%  cool={cool_pct:.0f}%')

# 3) Feature correlations
corr = data[features].corr()
print('\n=== Pct_Canopy correlations ===')
canopy_col = 'Pct_Canopy'
for f in features:
    print(f'  Canopy vs {f:25s}: {corr.loc[canopy_col, f]:+.3f}')
canopy_target_corr = data[[canopy_col, target]].corr().iloc[0, 1]
print(f'  Canopy vs {target:25s}: {canopy_target_corr:+.3f}')

# 4) VIF analysis
from numpy.linalg import inv
X = data[features].values
Xc = X - X.mean(axis=0)
corr_mat = np.corrcoef(Xc.T)
try:
    vif_diag = np.diag(inv(corr_mat))
    print('\n=== Variance Inflation Factors ===')
    for f, v in zip(features, vif_diag):
        print(f'  {f:25s}: {v:.2f}')
except:
    print('\n  VIF computation failed (singular matrix)')

# 5) What does univariate say about canopy?
uni_lr = LinearRegression()
uni_lr.fit(data[[canopy_col]], data[target])
print(f'\n=== Univariate OLS: Pct_Canopy -> {target} ===')
print(f'  coefficient: {uni_lr.coef_[0]:+.6f}')
print(f'  If +25pp:    {uni_lr.coef_[0] * 25:+.4f} F')
print(f'  R2:          {uni_lr.score(data[[canopy_col]], data[target]):.4f}')
