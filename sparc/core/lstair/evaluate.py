"""Three models of the traverse air-temperature anomaly, scored on cities they never saw.

Target: ``dTa`` (°C), each run's cell-mean traverse air temperature minus the run's city background (its median
over the driven cells; :mod:`~sparc.core.lstair.dataset`).  Every model is evaluated **leave-one-campaign-out**: fitted
on the other cities, scored on every driven cell of the held-out city, run window by run window (af first, then
am, pm and ni where enough cities have them).  The held-out unit is the **campaign group**
(:func:`campaign_groups`): catalog "cities" that are one campaign (same OSF node and date, e.g. Bronx and
Manhattan) or neighbours (grids within 10 km: New York and its boroughs, the two Kansas Citys) are held out
together, and the paired comparison treats each group, not each city, as one independent unit.

Models
------
* **A — land cover.**  Canopy, impervious, NDVI, albedo, elevation, log(1 + distance to water), and the Gaussian
  focal means of canopy, impervious, NDVI and albedo at 100, 300 and 1000 m, each as a within-run anomaly (minus
  the run's median over its cells: the target carries no city-level offset, so the predictors do not either), plus
  absolute canopy and impervious so saturating responses can be learned.  Two families: an additive spline ridge
  (cubic B-splines per predictor, a GAM-style smoother; Hastie & Tibshirani 1990) and histogram gradient boosting;
  the family and its regularisation are chosen by an **inner leave-city-out CV** on the training cities.  Cities
  weigh equally (sample weight ∝ 1/n per city, mean 1), and at most ``max_rows_per_city`` cells per city train a model.
* **B — A + LST**: the same, plus the LST anomaly of the cell and its 100 and 300 m focal means (minus the run's
  median LST).  A and B are compared on the same cells (cells with every A and B predictor).
* **C — physics with LST.**  :class:`sparc.core.physics.PhysicsModel` with the ``lst`` role: LST enters the heat
  source as surface–air coupling (q += c_lst⁺·(LST − T_bg)_c, spread by the same Green's function) next to the
  land-cover source, on the city's full raster (cells with features but no traverse feed the source, never the
  loss).  It is fitted **once per city and window** on that city's own cells of one run (the window's run with
  the most cells, with that run's ERA5 forcing and background; :func:`physics_run_id`; no advection: no wind
  record is assumed), and a held-out city's runs are each predicted with their own forcing.  **Transfer rule** to a held-out city: every nonlinear parameter (L, shading, storage, evaporative
  fraction, water sink) and the lapse coefficient γ is the **median over the training cities' fits**; the gain is
  transferred in physical units, a/forcing, and the LST term as the median over cities of the product
  a_phys·h_lst (the air's K-per-K response to LST, which the data determine; a and c_lst trade off within a fit),
  both converted back with the held-out city's own forcing (ERA5 SW↓ of the run, else a default per run); the
  intercept and the linear trend are city-specific and set to 0.  The regime is fixed per window (af and am day,
  pm and ni night: :data:`WINDOW_REGIME`), so every city of a window is fitted, and transferred, with one source
  formula; parameters the regime's source does not use are listed as unused, never transferred as information.  No label of the held-out city is used.
  ``C0`` is the same physics without LST (secondary).  C − C0 shows what LST adds *inside the physics*, not
  beyond land cover: the physics' land-cover source has a fixed form, and an LST that is only a function of land
  cover can still make up for what that form misses (the toy worlds of the tests show it).  The test of the
  claim is B − A, whose land-cover model is flexible and tuned.

Every model's predictions on the held-out city are centred to median zero over that run's scored cells (the
target's own definition; it uses positions, not labels), and the target is re-centred on the same cells
(:func:`recentre_target`): cells dropped for a missing predictor would otherwise leave the scored target off zero.

Scores and verdict
------------------
Per held-out city: RMSE (°C), R² (1 − SSE/SST about the city's mean anomaly) and the spatial Pearson r.  Paired
comparisons B − A and C − A: the mean over held-out cities of the difference in MSE, with a 95% t-interval
(n − 1 df) and a one-sided sign-flip test over cities (:func:`paired`; exact for up to 16 cities).  The verdict
"LST adds information beyond land cover" is given only when the sign-flip p ≤ 0.025, the t-interval of B − A lies
below 0 and the MSE falls by at least 5% (C − A likewise for the physics); otherwise "not shown" (and, when the
interval lies above 0, LST made predictions worse).  With fewer than :data:`MIN_VERDICT_CITIES` (8) held-out
cities the verdict is "not testable": with 5 the smallest sign-flip p is 1/32, and a percentile bootstrap over so
few cities claimed at about three times its nominal rate under a true null.  A secondary within-city block CV (1 km blocks) is reported and labelled
interpolation-prone: held-out blocks share the city's weather, streets and campaign with the training blocks, and
CAPA's own maps are built this way (Shandas et al. 2019).

References: Fisher, R.A. (1935), *The Design of Experiments*, Oliver & Boyd (the randomisation test); Good, P.
(2005), *Permutation, Parametric and Bootstrap Tests of Hypotheses*, 3rd ed., Springer; Hastie, T. &
Tibshirani, R. (1990), *Generalized Additive Models*, Chapman & Hall; Ho et al. (2014), *RSE* 154, 38–45,
doi:10.1016/j.rse.2014.08.012 (random forest for daytime maximum air temperature, RMSE 2.31 °C); Roberts et al.
(2017), *Ecography* 40(8), 913–929, doi:10.1111/ecog.02881 (block cross-validation for dependent data).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.core.lstair import RUN_ORDER, to_json
from sparc.core.lstair.dataset import FEATURES, FOCAL_FEATURES, FOCAL_RADII, LST_RADII, CityData, load_dataset

log = logging.getLogger(__name__)

MIN_CITIES = 3                 # leave-one-city-out needs at least two training cities
MIN_VERDICT_CITIES = 8         # fewer held-out cities: no verdict (with 5 the smallest sign-flip p is 1/32)
ALPHA = 0.025                  # one-sided level of the paired test (the claim is a one-sided improvement)
MIN_REL_GAIN = 0.05            # the claim also needs a relative MSE reduction of at least 5%
EXACT_SIGN_FLIP_MAX = 16       # up to this many cities every sign vector is enumerated (2^16)
GROUP_KM = 10.0                # cities whose grids come this close are one campaign group (held out together)
MIN_CELLS = 30                 # a run with fewer driven cells is not scored
DEFAULT_FORCING = {"af": (650.0, -90.0), "am": (150.0, -60.0), "pm": (40.0, -80.0), "ni": (0.0, -80.0)}

CENTRED = (["canopy", "impervious", "ndvi", "albedo", "elevation", "log_water"]
           + [f"{k}_f{r}" for k in FOCAL_FEATURES for r in FOCAL_RADII])
A_COLS = [f"{k}_c" for k in CENTRED] + ["canopy", "impervious"]
LST_COLS = ["lst_c"] + [f"lst_f{r}_c" for r in LST_RADII]
B_COLS = A_COLS + LST_COLS


@dataclass
class Settings:
    windows: tuple = RUN_ORDER
    boot: int = 4000
    seed: int = 0
    max_rows_per_city: int = 6000
    ridge_alphas: tuple = (0.3, 3.0, 30.0, 300.0)
    gbm: bool = True
    gbm_iter: int = 300
    inner_folds: int = 5
    physics: bool = True
    physics_cell_m: float = 60.0
    physics_iter: int = 60
    physics_L_max_m: float = 2000.0
    physics_variants: tuple = ("C", "C0")
    block_m: float = 1000.0
    within_city: bool = True


# --------------------------------------------------------------------------- #
# Design                                                                       #
# --------------------------------------------------------------------------- #
def design(table: pd.DataFrame) -> pd.DataFrame:
    """The table with the model columns: within-run anomalies ``<name>_c`` (minus the run's median over its
    cells) of the land-cover, focal and LST predictors; ``log_water`` = log(1 + distance to water)."""
    df = table.copy()
    df["log_water"] = np.log1p(np.clip(df["water_distance"].to_numpy(float), 0.0, None))
    g = df.groupby(["slug", "run_id"], sort=False)
    for k in CENTRED:
        df[f"{k}_c"] = df[k] - g[k].transform("median")
    lst_med = g["lst"].transform("median")
    df["lst_c"] = df["lst"] - lst_med
    for r in LST_RADII:
        df[f"lst_f{r}_c"] = df[f"lst_f{r}"] - lst_med
    return df


def _candidates(s: Settings) -> list[tuple[str, dict]]:
    out = [("ridge_spline", {"alpha": float(a)}) for a in s.ridge_alphas]
    if s.gbm:
        out.append(("gbm", {"max_leaf_nodes": 15, "learning_rate": 0.05, "max_iter": int(s.gbm_iter),
                            "min_samples_leaf": 40, "l2_regularization": 1.0}))
    return out


def _make(kind: str, params: dict, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    if kind == "ridge_spline":
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3, extrapolation="linear"),
                             Ridge(alpha=params["alpha"]))
    if kind == "gbm":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(kind)


def _fit_predict(kind: str, params: dict, Xtr, ytr, wtr, Xte, seed: int) -> np.ndarray:
    m = _make(kind, params, seed)
    if kind == "ridge_spline":
        m.fit(Xtr, ytr, ridge__sample_weight=wtr)
    else:
        m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xte)


def centre(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Predictions minus their median within each group (a run of a city): the target's own centring."""
    s = pd.Series(np.asarray(pred, float))
    return (s - s.groupby(np.asarray(groups)).transform("median")).to_numpy()


def _weights(slugs: np.ndarray) -> np.ndarray:
    """Equal total weight per city, scaled to mean 1 (so a ridge penalty keeps its meaning per sample)."""
    s = pd.Series(slugs)
    n_c = s.map(s.value_counts()).to_numpy(float)
    return (len(s) / s.nunique()) / n_c


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df
    parts = []
    for i, (_, g) in enumerate(df.groupby("slug", sort=True)):
        parts.append(g if len(g) <= n else g.sample(n, random_state=seed + i))
    return pd.concat(parts)


def city_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    mse = float(np.mean(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r = float(np.corrcoef(y, p)[0, 1]) if y.size > 2 and np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return {"n": int(y.size), "mse": mse, "rmse": math.sqrt(mse), "r2": 1.0 - float(np.sum(e ** 2)) / sst if sst > 0
            else float("nan"), "r": r}


def recentre_target(df: pd.DataFrame, n_all: pd.Series | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """``dTa`` re-centred on the cells actually scored: Ta minus its median over each run's scored cells.

    The dataset centres ``dTa`` on the median over all of a run's driven cells, but cells missing a predictor
    (an LST cloud gap over part of the city) are dropped before scoring and the predictions are centred on the
    cells that remain; without re-centring the target, a cloud over the cooler cells left the scored target with
    a non-zero median, an offset that inflated every model's MSE.  Returns the table and, per (city, run), the
    shift applied and the share of its cells dropped (``n_all``: the run's cells before the predictor filter)."""
    out = df.copy()
    g = out.groupby(["slug", "run_id"], sort=False)
    shift = g["Ta"].transform("median") - out["ta_background"]
    out["dTa"] = out["Ta"] - g["Ta"].transform("median")
    rows = []
    for (sl, rid), part in out.groupby(["slug", "run_id"], sort=True):
        n0 = int(n_all.get((sl, rid), len(part))) if n_all is not None else len(part)
        rows.append({"slug": sl, "run_id": rid, "shift_c": float(shift.loc[part.index].iloc[0]),
                     "n_scored": int(len(part)), "share_dropped": float(1.0 - len(part) / max(n0, 1))})
    return out, rows


def campaign_groups(cities: list[CityData], near_km: float = GROUP_KM) -> dict[str, str]:
    """The unit of independence: cities of one campaign or metro area form a group, held out together and resampled
    as one.  Two cities are linked when they share an OSF node and a campaign date (one campaign published as
    several catalog "cities", e.g. Bronx and Manhattan in j6eqr) or when their grids lie within ``near_km`` of each
    other (New York beside its boroughs, Kansas City KS beside MO): their held-out traverses share the day's
    weather, the vehicles and sensors, and neighbouring streets.  Returns slug → group id (the member slugs
    joined by ``+``; a city alone is its own group)."""
    info = []
    for c in cities:
        cat = c.meta.get("catalog") or {}
        try:
            w, so, e, n = c.grid.bounds_lonlat()
        except Exception:                             # noqa: BLE001 - no usable georeference: position unknown
            w = so = e = n = float("nan")
        info.append((c.slug, cat.get("node"), c.meta.get("campaign_date"), (w, so, e, n)))
    parent = {sl: sl for sl, *_ in info}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, (a, na, da, ba) in enumerate(info):
        for b, nb, db, bb in info[i + 1:]:
            same = na is not None and na == nb and da is not None and da == db
            lat = math.radians(0.5 * (ba[1] + bb[3])) if np.isfinite(ba[1]) and np.isfinite(bb[3]) else 0.0
            gx = max(0.0, max(ba[0], bb[0]) - min(ba[2], bb[2])) * 111.32 * math.cos(lat)
            gy = max(0.0, max(ba[1], bb[1]) - min(ba[3], bb[3])) * 110.57
            near = np.isfinite(gx) and np.isfinite(gy) and math.hypot(gx, gy) <= near_km
            if same or near:
                parent[find(a)] = find(b)
    members: dict[str, list[str]] = {}
    for sl, *_ in info:
        members.setdefault(find(sl), []).append(sl)
    return {sl: "+".join(sorted(m)) for m in members.values() for sl in m}


def inner_select(train: pd.DataFrame, cols: list[str], s: Settings) -> tuple[tuple[str, dict], list[dict]]:
    """The candidate with the lowest mean (over inner held-out cities) MSE in a leave-group-out CV on ``train``
    (campaign groups, :func:`campaign_groups`, when the table has a ``group`` column, else cities; grouped K-fold
    when there are more than ``inner_folds``)."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    groups = train["group" if "group" in train else "slug"].to_numpy()
    slugs = train["slug"].to_numpy()
    k = np.unique(groups).size
    split = LeaveOneGroupOut() if k <= s.inner_folds else GroupKFold(n_splits=s.inner_folds)
    X, y = train[cols].to_numpy(float), train["dTa"].to_numpy(float)
    w = _weights(slugs)
    rows = []
    folds = list(split.split(X, y, groups))
    for kind, params in _candidates(s):
        per_city = []
        for tr, te in folds:
            p = centre(_fit_predict(kind, params, X[tr], y[tr], w[tr], X[te], s.seed), train["run_id"].to_numpy()[te]
                       + "|" + slugs[te])
            for c in np.unique(slugs[te]):
                sel = slugs[te] == c
                per_city.append(float(np.mean((p[sel] - y[te][sel]) ** 2)))
        rows.append({"model": kind, "params": params, "inner_mse": float(np.mean(per_city))})
    best = min(rows, key=lambda r: r["inner_mse"])
    return (best["model"], best["params"]), rows


# --------------------------------------------------------------------------- #
# Physics (model C)                                                            #
# --------------------------------------------------------------------------- #
def _coarsen(r: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(r, float)
    ny, nx = r.shape
    py, px = -(-ny // k) * k, -(-nx // k) * k
    buf = np.full((py, px), np.nan)
    buf[:ny, :nx] = r
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(buf.reshape(py // k, k, px // k, k), axis=(1, 3))


def physics_run_id(city: CityData, run: str) -> str | None:
    """The run a city's physics fit for window ``run`` uses: the window's run with the most driven cells (ties:
    the earliest id).  One run per fit, so its forcing, background and target are of one day and hour."""
    t = city.table[city.table["run"] == run]
    if not len(t):
        return None
    n = t.groupby("run_id").size()
    return str(sorted(n.index, key=lambda r: (-int(n[r]), str(r)))[0])


def _forcing(city: CityData, run: str, run_id: str | None = None) -> tuple[float, float, str]:
    runs = [r for r in city.meta.get("runs", []) if "error" not in r]
    rec = next((r for r in runs if run_id is not None and r.get("run_id") == run_id), None)
    rec = rec or next((r for r in runs if r.get("run") == run), {})
    e = rec.get("era5") or {}
    if "sw_down" in e and "lw_net" in e and np.isfinite(e["sw_down"]) and np.isfinite(e["lw_net"]):
        return float(e["sw_down"]), float(e["lw_net"]), "ERA5"
    sw, lw = DEFAULT_FORCING.get(run, DEFAULT_FORCING["af"])
    return sw, lw, "default"


def physics_inputs(city: CityData, run: str, use_lst: bool, cell_m: float, run_id: str | None = None) -> dict:
    """Grid, frame (every raster cell), labels at the driven cells and the fine→coarse cell map for one run:
    ``run_id`` (default :func:`physics_run_id`), with that run's own forcing and background — never a blend of
    the window's runs."""
    from sparc.core.grid import Grid

    g = city.grid
    k = max(int(round(cell_m / g.dx)), 1)
    roles = [f for f in FEATURES if f in city.rasters] + (["lst"] if use_lst and "lst" in city.rasters else [])
    if use_lst and "lst" not in roles:
        raise ValueError(f"{city.slug}: no LST raster")
    R = {f: _coarsen(city.rasters[f], k) for f in roles}
    ny, nx = next(iter(R.values())).shape
    iy, ix = np.divmod(np.arange(nx * ny, dtype=np.int64), nx)
    ones = np.ones((ny, nx), dtype=np.int64)
    dx = g.dx * k
    x0, y0 = g.x0 + 0.5 * (k - 1) * g.dx, g.y0 + 0.5 * (k - 1) * g.dx
    grid = Grid(x0=x0, y0=y0, dx=dx, dy=dx, nx=nx, ny=ny, ix=ix, iy=iy, mask=ones > 0, counts=ones)
    frame = pd.DataFrame({f: R[f].ravel() for f in roles})
    if use_lst and frame["lst"].notna().mean() < 0.2:
        raise ValueError(f"{city.slug}: LST covers {frame['lst'].notna().mean():.0%} of the grid")
    run_id = run_id or physics_run_id(city, run)
    t = city.table[(city.table["run"] == run) & (city.table["run_id"] == run_id)]
    fy, fx = np.divmod(t["cell"].to_numpy(np.int64), g.nx)
    coarse = (fy // k) * nx + (fx // k)
    y = np.full(nx * ny, np.nan)
    s = pd.Series(t["dTa"].to_numpy(float)).groupby(coarse).mean()
    y[s.index.to_numpy()] = s.to_numpy()
    coords = np.column_stack([x0 + ix * dx, y0 + iy * dx]).astype(float)
    sw, lw, fsrc = _forcing(city, run, run_id)
    return {"grid": grid, "frame": frame, "y": y, "coords": coords, "coarse": coarse, "roles": roles, "run": run,
            "factor": k, "sw": sw, "lw": lw, "forcing_source": fsrc, "table": t, "run_id": run_id,
            "background": float(t["ta_background"].iloc[0]) if len(t) else 0.0}


#: The physics regime of each Heat Watch window, fixed per window (not per city's ERA5 SW: the 6–7 h and 19–20 h
#: runs straddle any SW threshold, and a transfer would then mix the day and night source formulas).
WINDOW_REGIME = {"af": "day", "am": "day", "pm": "night", "ni": "night"}
#: The parameters each regime's source uses (the others keep their starting values in a fit: never transferred).
REGIME_KEYS = {"day": ("L", "s", "kc", "a1", "e0", "e1", "e2", "e3", "w"),
               "night": ("L", "a1n", "e0", "e1", "e2", "e3", "w")}


def regime(run: str, sw: float) -> str:
    return WINDOW_REGIME.get(str(run), "day" if sw > 50.0 else "night")


def physics_cfg(inp: dict, s: Settings) -> dict:
    window = regime(inp.get("run", ""), inp["sw"])
    return {"roles": {r: r for r in inp["roles"]}, "window": window, "sw_down": inp["sw"],
            "lw_net": inp["lw"], "fit_advection": False, "max_iter": int(s.physics_iter),
            "L_max_m": float(s.physics_L_max_m), "num_threads": 1, "lst_background": inp["background"]}


def _norm(cfg: dict) -> float:
    sw = float(cfg["sw_down"]) if cfg["window"] == "day" else 0.0
    return sw if sw > 0 else max(abs(float(cfg["lw_net"])), 1e-9)


TRANSFER_KEYS = ("L", "s", "kc", "a1", "a1n", "e0", "e1", "e2", "e3", "w")


def fit_city_physics(city: CityData, run: str, use_lst: bool, s: Settings) -> dict:
    """Fit the physics on one city's run; the record the transfer rule needs (natural parameters, physical gain)."""
    from sparc.core.physics import PhysicsModel

    inp = physics_inputs(city, run, use_lst, s.physics_cell_m)
    tr = np.flatnonzero(np.isfinite(inp["y"]))
    if tr.size < MIN_CELLS:
        raise ValueError(f"{city.slug} {run}: {tr.size} labelled cells")
    cfg = physics_cfg(inp, s)
    t0 = time.perf_counter()
    m = PhysicsModel(inp["grid"], cfg, L_init=max(200.0, 3 * inp["grid"].dx)).fit(inp["frame"], inp["y"], tr,
                                                                                  inp["coords"])
    T = m._T
    norm = _norm(cfg)
    rec = {"slug": city.slug, "run": run, "run_id": inp["run_id"],
           "run_rule": "the window's run with the most driven cells", "lst": bool(use_lst),
           "natural": {k: float(T[k]) for k in TRANSFER_KEYS},
           "a": float(m.a), "gamma": float(m.gamma), "norm": norm, "a_phys": float(m.a) / norm,
           "train_r2": float(m.train_r2), "train_rmse": float(m.train_rmse), "fit_warning": m.fit_warning,
           "n_train": int(tr.size), "cell_m": float(inp["grid"].dx), "forcing": [inp["sw"], inp["lw"], inp["forcing_source"]],
           "window": cfg["window"], "seconds": round(time.perf_counter() - t0, 2), "roles": inp["roles"]}
    if use_lst:
        rec["c_lst"] = float(T["c_lst"])
        rec["h_lst"] = float(T["c_lst"]) * norm                      # the coupling in forcing units (W m⁻² K⁻¹)
        # what the data determine: the air's response per K of LST anomaly, a·c_lst = a_phys·h_lst (K per K);
        # a and c_lst trade off within a fit, so their separate values (and any r_a read from them) are not
        # usable as aerodynamic resistances
        rec["g_lst"] = rec["a_phys"] * rec["h_lst"]
    return rec


def transfer_params(fits: list[dict], use_lst: bool) -> dict:
    """The transfer rule: medians over the training cities' fits (gain and coupling in physical units).

    The LST term's effect on the air is a·c_lst·ΔLST = a_phys·h_lst·ΔLST, and within a fit a and c_lst trade off
    (anti-correlated across cities), so the separate medians of a_phys and h_lst misstate their product.  The
    transferred LST gain is the **median of the per-city products** g = a_phys·h_lst (K per K), and h_lst is set
    to g / median(a_phys), so the held-out city gets a·c_lst = g; median(a_phys) itself serves the land-cover
    source only.  With median(a_phys) ≤ 0 the LST term cannot be carried and h_lst is 0 (recorded)."""
    want = regime_of_fits(fits)
    if want is not None:
        fits = [f for f in fits if f.get("window", want) == want]          # one source formula, never a mix
    keys = REGIME_KEYS.get(want or "day", TRANSFER_KEYS)
    nat = {k: float(np.median([f["natural"][k] for f in fits])) for k in TRANSFER_KEYS}
    a_med = float(np.median([f["a_phys"] for f in fits]))
    out = {"natural": nat, "a_phys": a_med, "gamma": float(np.median([f["gamma"] for f in fits])),
           "n_cities": len(fits), "cities": [f["slug"] for f in fits], "regime": want,
           "fitted_keys": list(keys),
           "unused_keys": [k for k in TRANSFER_KEYS if k not in keys]}           # unused by the regime's source
    if use_lst:
        g = float(np.median([f.get("g_lst", f["a_phys"] * f["h_lst"]) for f in fits]))
        out["g_lst"] = g
        out["h_lst"] = g / a_med if a_med > 0 else 0.0
        if a_med <= 0:
            out["note"] = "median gain a_phys ≤ 0: the LST term cannot be carried (h_lst = 0)"
    return out


def regime_of_fits(fits: list[dict]) -> str | None:
    """The regime most of ``fits`` were fitted in (their ``window``), None when they do not say."""
    w = [f.get("window") for f in fits if f.get("window")]
    return max(set(w), key=w.count) if w else None


def predict_physics(city: CityData, run: str, use_lst: bool, tp: dict, s: Settings) -> pd.Series:
    """Model C on a held-out city's window from the transferred parameters: every run of the window predicted
    with **its own** forcing and background, centred to median zero per run; a Series indexed by (run_id, cell).
    Only the city's features and forcing are used."""
    from sparc.core.physics import PhysicsModel

    parts = []
    for rid in sorted(city.table.loc[city.table["run"] == run, "run_id"].unique()):
        inp = physics_inputs(city, run, use_lst, s.physics_cell_m, run_id=rid)
        cfg = physics_cfg(inp, s)
        norm = _norm(cfg)
        nat = dict(tp["natural"])
        if use_lst:
            nat["c_lst"] = tp["h_lst"] / norm
        m = PhysicsModel(inp["grid"], cfg, L_init=nat["L"]).assign(inp["frame"], inp["coords"], nat,
                                                                   a=tp["a_phys"] * norm, gamma=tp["gamma"])
        p = m.predict(inp["frame"], inp["coords"])[inp["coarse"]]
        t = inp["table"]
        parts.append(pd.Series(centre(p, np.zeros(len(p))),
                               index=pd.MultiIndex.from_arrays([t["run_id"].to_numpy(), t["cell"].to_numpy()])))
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def _fit_key(city: CityData, run: str, use_lst: bool, s: Settings) -> str:
    t = city.table[(city.table["run"] == run) & (city.table["run_id"] == physics_run_id(city, run))]
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(t[["cell", "dTa"]], index=False).to_numpy().tobytes())
    h.update(json.dumps([city.meta.get("built_utc"), use_lst, s.physics_cell_m, s.physics_iter, s.physics_L_max_m]).encode())
    return h.hexdigest()[:16]


def physics_fits(cities: list[CityData], run: str, use_lst: bool, s: Settings, cache: Path | None) -> dict:
    """Every city's physics fit for ``run`` (cached per city in ``cache``: reruns skip finished fits)."""
    out = {}
    for c in cities:
        key = _fit_key(c, run, use_lst, s)
        path = None if cache is None else cache / f"{c.slug}__{run}__{'lst' if use_lst else 'nolst'}.json"
        if path is not None and path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("key") == key:
                out[c.slug] = rec
                continue
        try:
            rec = fit_city_physics(c, run, use_lst, s)
        except Exception as exc:                      # noqa: BLE001 - one city's failed fit is recorded
            rec = {"slug": c.slug, "run": run, "error": str(exc).splitlines()[0][:200] if str(exc) else repr(exc)}
            log.warning("lstair physics: %s %s failed: %s", c.slug, run, rec["error"])
        rec["key"] = key
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_text(to_json(rec), encoding="utf-8")
            tmp.replace(path)
        out[c.slug] = rec
    return out


# --------------------------------------------------------------------------- #
# Paired comparison and verdict                                                #
# --------------------------------------------------------------------------- #
def sign_flip(d: np.ndarray, draws: int = 4000, seed: int = 0) -> dict:
    """One-sided sign-flip (randomisation) test of a mean paired difference over independent units.

    Under the null that each unit's difference is symmetric about 0, every assignment of signs is equally
    likely; the p-value of an improvement (mean < 0) is the share of the 2ⁿ sign vectors whose mean is at most
    the observed one (Fisher 1935; Good 2005).  Exact (all 2ⁿ vectors) up to :data:`EXACT_SIGN_FLIP_MAX` units,
    else ``draws`` random vectors plus the observed one.  With 5 units the smallest p is 1/32 > 0.025: no claim
    is possible, which is the honest answer."""
    d = np.asarray(d, float)
    n = d.size
    obs = float(d.mean())
    if n <= EXACT_SIGN_FLIP_MAX:
        signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)
        method = f"exact ({2 ** n} sign vectors)"
    else:
        rng = np.random.default_rng(seed)
        signs = np.vstack([np.ones(n), rng.choice([-1.0, 1.0], size=(int(draws), n))])
        method = f"Monte Carlo ({int(draws)} sign vectors + the observed)"
    means = signs @ d / n
    tol = 1e-12 * max(1.0, abs(obs))
    return {"p_less": float(np.mean(means <= obs + tol)), "p_greater": float(np.mean(means >= obs - tol)),
            "method": method}


def paired(mse_x: np.ndarray, mse_a: np.ndarray, boot: int = 4000, seed: int = 0, label: str = "B − A",
           claim: str = "LST adds information beyond land cover") -> dict:
    """Mean over held-out units (cities, or campaign groups) of MSE_x − MSE_A, its 95% t-interval (n − 1 df) and
    a one-sided sign-flip test, and the verdict.

    The claim is made only when all three hold: the sign-flip p of an improvement is at most :data:`ALPHA`
    (0.025), the t-interval lies below 0, and the relative MSE change is at most −:data:`MIN_REL_GAIN` (5%).
    With fewer than :data:`MIN_VERDICT_CITIES` units the verdict is "not testable".  (A percentile bootstrap over
    5–10 cities ignored the n − 1 degrees of freedom and claimed at about three times its stated rate under a
    true null.)  ``boot`` is the number of random sign vectors when the test cannot be enumerated."""
    from scipy import stats

    d = np.asarray(mse_x, float) - np.asarray(mse_a, float)
    n = d.size
    out = {"comparison": label, "n_cities": int(n), "claim": claim}
    if n == 0:
        out.update(verdict="not testable", reason="no held-out city")
        return out
    mean = float(d.mean())
    base = float(np.mean(mse_a))
    rel = mean / base if base > 0 else None
    out.update(mean_diff_mse=mean, rel_change=rel, n_better=int((d < 0).sum()), n_worse=int((d > 0).sum()),
               alpha_one_sided=ALPHA, min_rel_gain=MIN_REL_GAIN)
    if n >= 2:
        se = float(d.std(ddof=1)) / math.sqrt(n)
        t = float(stats.t.ppf(1.0 - ALPHA, n - 1))
        lo, hi = mean - t * se, mean + t * se
        out.update(ci95=[float(lo), float(hi)], ci_method=f"t-interval, {n - 1} df")
        sf = sign_flip(d, boot, seed)
        out.update(p_sign_flip=sf["p_less"], p_sign_flip_worse=sf["p_greater"], sign_flip=sf["method"])
    if n < MIN_VERDICT_CITIES:
        out.update(verdict="not testable", reason=f"{n} held-out units (< {MIN_VERDICT_CITIES}): too few for a test "
                                                  "over cities")
        return out
    better = out["p_sign_flip"] <= ALPHA and out["ci95"][1] < 0
    worse = out["p_sign_flip_worse"] <= ALPHA and out["ci95"][0] > 0
    if better and rel is not None and rel <= -MIN_REL_GAIN:
        out.update(verdict=claim, reason=f"sign-flip p = {out['p_sign_flip']:.3g} ≤ {ALPHA}, the 95% t-interval of "
                                         f"the paired MSE difference lies below 0 and MSE falls by {-100 * rel:.0f}%")
    elif better:
        out.update(verdict="not shown", reason=f"significant (sign-flip p = {out['p_sign_flip']:.3g}) but the MSE falls "
                                               f"by only {-100 * (rel or 0):.1f}% (< {100 * MIN_REL_GAIN:.0f}%)")
    elif worse:
        out.update(verdict="not shown", reason="the 95% t-interval lies above 0: adding it made predictions worse")
    else:
        out.update(verdict="not shown", reason=f"sign-flip p = {out['p_sign_flip']:.3g} (one-sided) and the 95% "
                                               f"t-interval {out['ci95'][0]:+.4f} to {out['ci95'][1]:+.4f}")
    return out


# --------------------------------------------------------------------------- #
# Within-city block CV (secondary)                                             #
# --------------------------------------------------------------------------- #
def within_city(df: pd.DataFrame, s: Settings) -> dict:
    """Spatial block CV inside each city (``block_m`` blocks, ≤ 5 folds), additive spline ridge (α = 3) on A and
    on B columns.  Interpolation-prone: neighbouring blocks share the city's streets, weather and campaign."""
    from sklearn.model_selection import GroupKFold

    rows = []
    for slug, g in df.groupby("slug", sort=True):
        blocks = (np.floor(g["x"] / s.block_m).astype(np.int64) * 100003 + np.floor(g["y"] / s.block_m).astype(np.int64)).to_numpy()
        nb = np.unique(blocks).size
        if nb < 3 or len(g) < MIN_CELLS:
            continue
        kf = GroupKFold(n_splits=min(5, nb))
        y = g["dTa"].to_numpy(float)
        rec = {"slug": slug, "n": int(len(g)), "n_blocks": int(nb)}
        for name, cols in (("A", A_COLS), ("B", B_COLS)):
            X = g[cols].to_numpy(float)
            if not np.isfinite(X).all():                  # B without LST (a dataset built with --no-lst)
                continue
            p = np.full(len(g), np.nan)
            for tr, te in kf.split(X, y, blocks):
                p[te] = _fit_predict("ridge_spline", {"alpha": 3.0}, X[tr], y[tr], np.ones(tr.size), X[te], s.seed)
            rec[name] = city_metrics(y, centre(p, g["run_id"].to_numpy()))
        rows.append(rec)
    if not rows:
        return {"label": "within-city block CV (interpolation-prone)", "cities": []}
    return {"label": "within-city block CV (interpolation-prone: held-out blocks share the city's streets, weather "
                     "and campaign with the training blocks)", "block_m": s.block_m, "cities": rows,
            "median_r2": {m: float(np.median([r[m]["r2"] for r in rows if m in r])) for m in ("A", "B")
                          if any(m in r for r in rows)},
            "mean_diff_mse_B_minus_A": (float(np.mean([r["B"]["mse"] - r["A"]["mse"] for r in rows if "B" in r]))
                                        if any("B" in r for r in rows) else None)}


# --------------------------------------------------------------------------- #
# The harness                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_window(cities: list[CityData], run: str, s: Settings, cache: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Leave-one-campaign-group-out A, B, C (and C0) on one run window; (results, per-cell predictions)."""
    t0 = time.perf_counter()
    tabs = [c.table[c.table["run"] == run] for c in cities]
    keep = [c for c, t in zip(cities, tabs) if len(t) >= MIN_CELLS]
    df = design(pd.concat([c.table[c.table["run"] == run] for c in keep], ignore_index=True)) if keep else pd.DataFrame()
    res: dict = {"run": run, "cities": [c.slug for c in keep]}
    if len(keep) < MIN_CITIES:
        res.update(status="not run", reason=f"{len(keep)} cities with ≥ {MIN_CELLS} driven cells in this run (need "
                                            f"{MIN_CITIES})")
        return res, pd.DataFrame()
    ok_a = np.isfinite(df[A_COLS + ["dTa"]].to_numpy(float)).all(axis=1)
    has_lst = np.isfinite(df[LST_COLS].to_numpy(float)).all(axis=1)
    no_lst = not (ok_a & has_lst).any()
    res["lst"] = ("none: the dataset has no LST in this window (built with --no-lst, or no city had a clear Landsat "
                  "scene); A and C0 only") if no_lst else "present"
    ok = ok_a if no_lst else ok_a & has_lst
    res["n_cells_total"] = int(len(df))
    res["n_cells_dropped_missing_predictor"] = int((~ok).sum())
    n_all = df.groupby(["slug", "run_id"]).size()
    excluded = []
    if not no_lst:
        for sl in sorted(df["slug"].unique()):
            sel = (df["slug"] == sl).to_numpy()
            if (ok_a & sel).sum() >= MIN_CELLS and (ok & sel).sum() < MIN_CELLS:
                excluded.append({"slug": sl, "reason": f"no LST over its driven cells ({int((has_lst & sel).sum())} of "
                                                       f"{int(sel.sum())}): no clear Landsat scene"})
    res["excluded"] = excluded
    df = df[ok].reset_index(drop=True)
    by_city = {c.slug: c for c in keep}
    slugs = [c for c in sorted(df["slug"].unique()) if (df["slug"] == c).sum() >= MIN_CELLS]
    if len(slugs) < MIN_CITIES:
        why = "every predictor" if no_lst else "every predictor (missing LST excluded " + \
            (f"{len(excluded)} of them)" if excluded else "none)")
        res.update(status="not run", reason=f"{len(slugs)} cities with ≥ {MIN_CELLS} cells with {why}")
        return res, pd.DataFrame()
    df = df[df["slug"].isin(slugs)].reset_index(drop=True)
    df, res["target_recentred"] = recentre_target(df, n_all)
    groups = campaign_groups([by_city[c] for c in slugs])
    df["group"] = df["slug"].map(groups)
    units = sorted(set(groups.values()))
    res["groups"] = {g: [c for c in slugs if groups[c] == g] for g in units}
    if len(units) < MIN_CITIES:
        res.update(status="not run", reason=f"{len(units)} independent campaign groups (need {MIN_CITIES}): "
                                            f"{', '.join(units)}")
        return res, pd.DataFrame()
    key = (df["slug"] + "|" + df["run_id"]).to_numpy()                 # a run of a city: the centring unit
    preds = {m: np.full(len(df), np.nan) for m in ("A", "B", "C", "C0")}
    chosen = {}
    fits = {}
    models = ("A",) if no_lst else ("A", "B")
    if s.physics:
        for v in s.physics_variants:
            if v == "C" and no_lst:
                continue
            fits[v] = physics_fits([by_city[c] for c in slugs], run, v == "C", s,
                                   None if cache is None else cache / "physics")
    per_city = []
    for g in units:                                                    # hold out a whole campaign group
        members = res["groups"][g]
        te_g = (df["group"] == g).to_numpy()
        train = _subsample(df[~te_g], s.max_rows_per_city, s.seed)
        recs = {h: {"slug": h, "city": str(df.loc[df["slug"] == h, "city"].iloc[0]),
                    "n": int((df["slug"] == h).sum()), "group": g} for h in members}
        for name, cols in (("A", A_COLS), ("B", B_COLS))[:len(models)]:
            (kind, params), table = inner_select(train, cols, s)
            p = _fit_predict(kind, params, train[cols].to_numpy(float), train["dTa"].to_numpy(float),
                             _weights(train["slug"].to_numpy()), df.loc[te_g, cols].to_numpy(float), s.seed)
            preds[name][te_g] = centre(p, key[te_g])
            chosen.setdefault(name, []).append({"held_out": g, "model": kind, "params": params, "inner": table})
            for h in members:
                te = (df["slug"] == h).to_numpy()
                recs[h][name] = city_metrics(df.loc[te, "dTa"], preds[name][te])
                recs[h][name]["model"] = kind
        for v, fv in fits.items():
            others = [f for k, f in fv.items() if k not in members and "error" not in f]
            for h in members:
                rec = recs[h]
                te = (df["slug"] == h).to_numpy()
                if len(others) < MIN_CITIES - 1:
                    rec[v] = {"error": f"{len(others)} training cities with a physics fit"}
                    continue
                tp = transfer_params(others, v == "C")
                try:
                    pc = predict_physics(by_city[h], run, v == "C", tp, s)
                except Exception as exc:              # noqa: BLE001
                    rec[v] = {"error": str(exc).splitlines()[0][:200] if str(exc) else repr(exc)}
                    continue
                # the physics predicts every driven cell of every run; score the cells A and B are scored on
                idx = pd.MultiIndex.from_arrays([df.loc[te, "run_id"], df.loc[te, "cell"]])
                preds[v][te] = centre(pc.reindex(idx).to_numpy(), key[te])
                rec[v] = city_metrics(df.loc[te, "dTa"], preds[v][te])
                rec[v]["transfer"] = {"a_phys": tp["a_phys"], "L_m": tp["natural"]["L"],
                                      **({"g_lst": tp["g_lst"], "h_lst": tp["h_lst"]} if v == "C" else {})}
        for h in members:
            if "lst_offset_days" in df:
                off = df.loc[df["slug"] == h, "lst_offset_days"]
                recs[h]["lst_offset_days"] = None if off.isna().all() else float(off.abs().median())
            per_city.append(recs[h])
    res["per_city"] = per_city
    res["chosen"] = {k: [{kk: vv for kk, vv in c.items() if kk != "inner"} for c in v] for k, v in chosen.items()}
    res["inner_cv"] = {k: [c["inner"] for c in v] for k, v in chosen.items()}

    def pair(x: str, base: str) -> tuple[np.ndarray, np.ndarray]:
        """Per campaign group (the independent unit): the mean over its cities of each model's MSE."""
        xs, bs = [], []
        for g in units:
            rows = [r for r in per_city if r["group"] == g]
            if rows and all("mse" in (r.get(x) or {}) and "mse" in (r.get(base) or {}) for r in rows):
                xs.append(float(np.mean([r[x]["mse"] for r in rows])))
                bs.append(float(np.mean([r[base]["mse"] for r in rows])))
        return np.array(xs), np.array(bs)

    res["summary"] = {}
    for m in ("A", "B", "C", "C0"):
        rows = [r[m] for r in per_city if "mse" in (r.get(m) or {})]
        if rows:
            res["summary"][m] = {"n_cities": len(rows), "mean_rmse": float(np.mean([x["rmse"] for x in rows])),
                                 "median_r2": float(np.median([x["r2"] for x in rows])),
                                 "median_r": float(np.nanmedian([x["r"] for x in rows]))}
    if no_lst:
        nt = {"verdict": "not testable", "n_cities": 0, "reason": "no LST: " + res["lst"].split(": ", 1)[1]}
        res["paired"] = {"B-A": {"comparison": "B − A", "claim": "LST adds information beyond land cover", **nt}}
    else:
        res["paired"] = {"B-A": paired(*pair("B", "A"), s.boot, s.seed)}
    if "C" in res["summary"]:
        res["paired"]["C-A"] = paired(*pair("C", "A"), s.boot, s.seed + 1, "C − A", "physics with LST beats land cover")
    if "C" in res["summary"] and "C0" in res["summary"]:
        res["paired"]["C-C0"] = paired(*pair("C", "C0"), s.boot, s.seed + 2, "C − C0",
                                       "LST adds information inside the physics")
    res["verdict"] = res["paired"]["B-A"]["verdict"]
    offs = [(r["lst_offset_days"], r["B"]["mse"] - r["A"]["mse"]) for r in per_city
            if r.get("lst_offset_days") is not None and "B" in r]
    if len(offs) >= 4 and len({o for o, _ in offs}) > 1:
        from scipy.stats import spearmanr

        rho = spearmanr([o for o, _ in offs], [d for _, d in offs]).correlation
        res["gain_vs_lst_offset"] = {"spearman": float(rho) if np.isfinite(rho) else None, "n": len(offs),
                                     "note": "B − A MSE difference against the scene's |offset| in days (descriptive)"}
    if fits:
        res["physics_fits"] = {v: list(f.values()) for v, f in fits.items()}
    if s.within_city:
        res["within_city"] = within_city(df, s)
    res["status"] = "done"
    res["seconds"] = round(time.perf_counter() - t0, 1)
    pred = df[["city", "slug", "run", "run_id", "cell", "x", "y", "dTa"]].copy()
    for m, v in preds.items():
        if np.isfinite(v).any():
            pred[f"pred_{m}"] = v
    return res, pred


def evaluate(cities: list[CityData], s: Settings | None = None, cache: Path | None = None) -> tuple[dict, pd.DataFrame]:
    s = s or Settings()
    out = {"schema": 1, "target": "dTa: traverse air temperature anomaly from the run's city median (°C)",
           "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(s).items()},
           "transfer_rule": TRANSFER_RULE, "n_cities": len(cities), "cities": [c.slug for c in cities], "windows": {}}
    preds = []
    present = {r for c in cities for r in c.table["run"].unique()}
    for run in [w for w in s.windows if w in present]:
        log.info("lstair evaluate: window %s", run)
        res, pred = evaluate_window(cities, run, s, cache)
        out["windows"][run] = res
        if len(pred):
            preds.append(pred)
    return out, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


TRANSFER_RULE = ("Model C is fitted once per city and window on that city's own driven cells of one run (the "
                 "window's run with the most cells, with that run's forcing and background).  For a held-out city "
                 "every nonlinear physics parameter and the lapse coefficient are the medians of the training "
                 "cities' fits; the gain is transferred in physical units (a/forcing) and the LST term as the "
                 "median of the per-city products a_phys·h_lst (K per K), converted with the held-out city's own "
                 "run forcing; the intercept and the "
                 "linear trend are 0; each of its runs is predicted with that run's own forcing and centred to "
                 "median zero over the run's cells.  No held-out "
                 "label is used.")


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #
def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def results_markdown(res: dict) -> str:
    L = ["# LST → air temperature: leave-one-city-out", "",
         f"Target: {res['target']}.  {res['n_cities']} cities.  Models: A land cover + focal features; B A + LST; "
         "C physics with LST in the heat source; C0 physics without LST (secondary).", "",
         f"Transfer rule (C): {res['transfer_rule']}", ""]
    for run, w in res["windows"].items():
        L += [f"## Run {run}", ""]
        if w.get("status") != "done":
            L += [f"Not run: {w.get('reason')}", ""]
            continue
        pb = w["paired"]["B-A"]
        L += [f"**Verdict: {pb['verdict']}.** {pb.get('reason', '')}.", ""]
        L += ["| comparison | mean ΔMSE (°C²) | 95% CI (t, n − 1 df) | sign-flip p (one-sided) | relative | "
              "units better / worse | verdict |",
              "|---|---|---|---|---|---|---|"]
        for k, p in w["paired"].items():
            if "ci95" not in p:
                L.append(f"| {p['comparison']} | — | — | — | — | — | {p['verdict']} |")
                continue
            rel = "—" if p.get("rel_change") is None else f"{100 * p['rel_change']:+.0f}%"
            L.append(f"| {p['comparison']} | {p['mean_diff_mse']:+.4f} | {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f} | "
                     f"{p['p_sign_flip']:.3g} | {rel} | {p['n_better']} / {p['n_worse']} | {p['verdict']} |")
        L += ["", "| held-out city | cells | A RMSE | B RMSE | C RMSE | C0 RMSE | A R² | B R² | C R² | A r | B r | C r |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in w["per_city"]:
            def g(m, k):
                return _f((r.get(m) or {}).get(k))
            L.append(f"| {r['city']} | {r['n']:,} | {g('A', 'rmse')} | {g('B', 'rmse')} | {g('C', 'rmse')} | "
                     f"{g('C0', 'rmse')} | {g('A', 'r2')} | {g('B', 'r2')} | {g('C', 'r2')} | {g('A', 'r')} | "
                     f"{g('B', 'r')} | {g('C', 'r')} |")
        wc = w.get("within_city") or {}
        if wc.get("cities"):
            dm = wc.get("mean_diff_mse_B_minus_A")
            L += ["", f"Secondary, {wc['label']}: median R² A {_f(wc['median_r2'].get('A'))}, B "
                      f"{_f(wc['median_r2'].get('B'))}; mean ΔMSE (B − A) {_f(dm, 4)} °C²."]
        if w.get("excluded"):
            L += ["", "Excluded: " + "; ".join(f"{e['slug']} ({e['reason']})" for e in w["excluded"]) + "."]
        if w.get("lst", "present") != "present":
            L += ["", f"LST: {w['lst']}."]
        if w.get("gain_vs_lst_offset"):
            g = w["gain_vs_lst_offset"]
            L += ["", f"B − A against the LST scene's offset from the campaign: Spearman {_f(g['spearman'])} over {g['n']} "
                      "cities (descriptive)."]
        L.append("")
    return "\n".join(L) + "\n"


def run_evaluation(data_dir: str | Path, out_dir: str | Path, s: Settings | None = None) -> dict:
    """``sparc core lstair evaluate``: load the dataset, evaluate, write results.json, results.md, predictions."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cities = load_dataset(data_dir)
    if len(cities) < MIN_CITIES:
        raise ValueError(f"{len(cities)} finished cities under {data_dir}: leave-one-city-out needs {MIN_CITIES}")
    res, pred = evaluate(cities, s, cache=out)
    res["data"] = str(data_dir)
    (out / "results.json").write_text(to_json(res), encoding="utf-8")
    (out / "results.md").write_text(results_markdown(res), encoding="utf-8")
    if len(pred):
        pred.to_parquet(out / "predictions.parquet", index=False)
    return res


__all__ = ["Settings", "design", "A_COLS", "B_COLS", "LST_COLS", "city_metrics", "paired", "sign_flip", "centre",
           "inner_select", "campaign_groups", "recentre_target", "WINDOW_REGIME", "regime",
           "fit_city_physics", "transfer_params", "predict_physics", "physics_fits", "evaluate_window", "evaluate",
           "results_markdown", "run_evaluation", "TRANSFER_RULE", "MIN_VERDICT_CITIES"]
