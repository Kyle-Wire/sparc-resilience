"""Three models of the traverse air-temperature anomaly, scored on cities they never saw.

Target: ``dTa`` (°C), each run's cell-mean traverse air temperature minus the run's city background (its median
over the driven cells; :mod:`~sparc.core.lstair.dataset`).  Every model is evaluated **leave-one-city-out**: fitted
on the other cities, scored on every driven cell of the held-out city, run window by run window (af first, then
am, pm and ni where enough cities have them).

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
  loss).  It is fitted **once per city and run** on that city's own cells (no advection: no wind record is
  assumed).  **Transfer rule** to a held-out city: every nonlinear parameter (L, shading, storage, evaporative
  fraction, water sink) and the lapse coefficient γ is the **median over the training cities' fits**; the gain and
  the coupling are transferred in physical units, a/forcing (= r_a/ρc_p) and c_lst·forcing (= ρc_p/r_a), and
  converted back with the held-out city's own forcing (ERA5 SW↓ of the run, else a default per run); the
  intercept and the linear trend are city-specific and set to 0.  No label of the held-out city is used.
  ``C0`` is the same physics without LST (secondary).  C − C0 shows what LST adds *inside the physics*, not
  beyond land cover: the physics' land-cover source has a fixed form, and an LST that is only a function of land
  cover can still make up for what that form misses (the toy worlds of the tests show it).  The test of the
  claim is B − A, whose land-cover model is flexible and tuned.

Every model's predictions on the held-out city are centred to median zero over that run's cells (the target's
own definition; it uses positions, not labels).

Scores and verdict
------------------
Per held-out city: RMSE (°C), R² (1 − SSE/SST about the city's mean anomaly) and the spatial Pearson r.  Paired
comparisons B − A and C − A: the mean over held-out cities of the difference in MSE, with a 95% percentile
bootstrap over cities (Efron & Tibshirani 1993).  The verdict "LST adds information beyond land cover" is given
only when the 95% interval of B − A lies below 0 (C − A likewise for the physics); otherwise "not shown" (and,
when the interval lies above 0, LST made predictions worse).  With fewer than :data:`MIN_VERDICT_CITIES` held-out
cities the verdict is "not testable".  A secondary within-city block CV (1 km blocks) is reported and labelled
interpolation-prone: held-out blocks share the city's weather, streets and campaign with the training blocks, and
CAPA's own maps are built this way (Shandas et al. 2019).

References: Efron, B. & Tibshirani, R.J. (1993), *An Introduction to the Bootstrap*, Chapman & Hall; Hastie, T. &
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

from sparc.core.lstair import RUN_ORDER
from sparc.core.lstair.dataset import FEATURES, FOCAL_FEATURES, FOCAL_RADII, LST_RADII, CityData, load_dataset

log = logging.getLogger(__name__)

MIN_CITIES = 3                 # leave-one-city-out needs at least two training cities
MIN_VERDICT_CITIES = 5         # fewer held-out cities: the bootstrap over cities is too coarse for a verdict
MIN_CELLS = 30                 # a run with fewer driven cells is not scored
DEFAULT_FORCING = {"af": (650.0, -90.0), "am": (150.0, -60.0), "pm": (40.0, -80.0), "ni": (0.0, -80.0)}
RHO_CP = 1170.0                # J m⁻³ K⁻¹ (air near 30 °C), for the implied r_a

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


def inner_select(train: pd.DataFrame, cols: list[str], s: Settings) -> tuple[tuple[str, dict], list[dict]]:
    """The candidate with the lowest mean (over inner held-out cities) MSE in a leave-city-out CV on ``train``
    (grouped K-fold over cities when there are more than ``inner_folds``)."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    groups = train["slug"].to_numpy()
    k = np.unique(groups).size
    split = LeaveOneGroupOut() if k <= s.inner_folds else GroupKFold(n_splits=s.inner_folds)
    X, y = train[cols].to_numpy(float), train["dTa"].to_numpy(float)
    w = _weights(groups)
    rows = []
    folds = list(split.split(X, y, groups))
    for kind, params in _candidates(s):
        per_city = []
        for tr, te in folds:
            p = centre(_fit_predict(kind, params, X[tr], y[tr], w[tr], X[te], s.seed), train["run_id"].to_numpy()[te]
                       + "|" + groups[te])
            for c in np.unique(groups[te]):
                sel = groups[te] == c
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


def _forcing(city: CityData, run: str) -> tuple[float, float, str]:
    rec = next((r for r in city.meta.get("runs", []) if r.get("run") == run and "error" not in r), {})
    e = rec.get("era5") or {}
    if "sw_down" in e and "lw_net" in e and np.isfinite(e["sw_down"]) and np.isfinite(e["lw_net"]):
        return float(e["sw_down"]), float(e["lw_net"]), "ERA5"
    sw, lw = DEFAULT_FORCING.get(run, DEFAULT_FORCING["af"])
    return sw, lw, "default"


def physics_inputs(city: CityData, run: str, use_lst: bool, cell_m: float) -> dict:
    """Grid, frame (every raster cell), labels at the driven cells and the fine→coarse cell map for one run."""
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
    t = city.table[city.table["run"] == run]
    fy, fx = np.divmod(t["cell"].to_numpy(np.int64), g.nx)
    coarse = (fy // k) * nx + (fx // k)
    y = np.full(nx * ny, np.nan)
    s = pd.Series(t["dTa"].to_numpy(float)).groupby(coarse).mean()
    y[s.index.to_numpy()] = s.to_numpy()
    coords = np.column_stack([x0 + ix * dx, y0 + iy * dx]).astype(float)
    sw, lw, fsrc = _forcing(city, run)
    return {"grid": grid, "frame": frame, "y": y, "coords": coords, "coarse": coarse, "roles": roles,
            "factor": k, "sw": sw, "lw": lw, "forcing_source": fsrc, "table": t,
            "background": float(t["ta_background"].iloc[0]) if len(t) else 0.0}


def physics_cfg(inp: dict, s: Settings) -> dict:
    day = inp["sw"] > 50.0
    return {"roles": {r: r for r in inp["roles"]}, "window": "day" if day else "night", "sw_down": inp["sw"],
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
    rec = {"slug": city.slug, "run": run, "lst": bool(use_lst), "natural": {k: float(T[k]) for k in TRANSFER_KEYS},
           "a": float(m.a), "gamma": float(m.gamma), "norm": norm, "a_phys": float(m.a) / norm,
           "train_r2": float(m.train_r2), "train_rmse": float(m.train_rmse), "fit_warning": m.fit_warning,
           "n_train": int(tr.size), "cell_m": float(inp["grid"].dx), "forcing": [inp["sw"], inp["lw"], inp["forcing_source"]],
           "window": cfg["window"], "seconds": round(time.perf_counter() - t0, 2), "roles": inp["roles"]}
    if use_lst:
        rec["c_lst"] = float(T["c_lst"])
        rec["h_lst"] = float(T["c_lst"]) * norm                      # W m⁻² K⁻¹ = ρc_p / r_a
        rec["r_a_lst_s_m"] = RHO_CP / rec["h_lst"] if rec["h_lst"] > 0 else None
    if m.a > 0:
        rec["r_a_s_m"] = float(m.a) / norm * RHO_CP                   # a = r_a·SW/ρc_p in the local limit
    return rec


def transfer_params(fits: list[dict], use_lst: bool) -> dict:
    """The transfer rule: medians over the training cities' fits (gain and coupling in physical units)."""
    nat = {k: float(np.median([f["natural"][k] for f in fits])) for k in TRANSFER_KEYS}
    out = {"natural": nat, "a_phys": float(np.median([f["a_phys"] for f in fits])),
           "gamma": float(np.median([f["gamma"] for f in fits])), "n_cities": len(fits),
           "cities": [f["slug"] for f in fits]}
    if use_lst:
        out["h_lst"] = float(np.median([f["h_lst"] for f in fits]))
    return out


def predict_physics(city: CityData, run: str, use_lst: bool, tp: dict, s: Settings) -> np.ndarray:
    """Model C on a held-out city's run from the transferred parameters: predictions at the run's cells
    (table order), centred to median zero.  Only the city's features and forcing are used."""
    from sparc.core.physics import PhysicsModel

    inp = physics_inputs(city, run, use_lst, s.physics_cell_m)
    cfg = physics_cfg(inp, s)
    norm = _norm(cfg)
    nat = dict(tp["natural"])
    if use_lst:
        nat["c_lst"] = tp["h_lst"] / norm
    m = PhysicsModel(inp["grid"], cfg, L_init=nat["L"]).assign(inp["frame"], inp["coords"], nat,
                                                               a=tp["a_phys"] * norm, gamma=tp["gamma"])
    p = m.predict(inp["frame"], inp["coords"])[inp["coarse"]]
    return centre(p, inp["table"]["run_id"].to_numpy())


def _fit_key(city: CityData, run: str, use_lst: bool, s: Settings) -> str:
    t = city.table[city.table["run"] == run]
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
            tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            tmp.replace(path)
        out[c.slug] = rec
    return out


# --------------------------------------------------------------------------- #
# Paired comparison and verdict                                                #
# --------------------------------------------------------------------------- #
def paired(mse_x: np.ndarray, mse_a: np.ndarray, boot: int = 4000, seed: int = 0, label: str = "B − A",
           claim: str = "LST adds information beyond land cover") -> dict:
    """Mean over cities of MSE_x − MSE_A with a 95% percentile bootstrap over cities, and the verdict."""
    d = np.asarray(mse_x, float) - np.asarray(mse_a, float)
    n = d.size
    out = {"comparison": label, "n_cities": int(n), "claim": claim}
    if n == 0:
        out.update(verdict="not testable", reason="no held-out city")
        return out
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, n, size=(int(boot), n))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(bm, [2.5, 97.5]))
    mean = float(d.mean())
    out.update(mean_diff_mse=mean, ci95=[lo, hi], rel_change=mean / float(np.mean(mse_a)) if np.mean(mse_a) > 0 else None,
               n_better=int((d < 0).sum()), n_worse=int((d > 0).sum()), boot=int(boot))
    if n < MIN_VERDICT_CITIES:
        out.update(verdict="not testable", reason=f"{n} held-out cities (< {MIN_VERDICT_CITIES}): too few for a "
                                                  "bootstrap over cities")
    elif hi < 0:
        out.update(verdict=claim, reason="the 95% interval of the paired MSE difference lies below 0")
    elif lo > 0:
        out.update(verdict="not shown", reason="the 95% interval lies above 0: adding it made predictions worse")
    else:
        out.update(verdict="not shown", reason="the 95% interval of the paired MSE difference includes 0")
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
            p = np.full(len(g), np.nan)
            for tr, te in kf.split(X, y, blocks):
                p[te] = _fit_predict("ridge_spline", {"alpha": 3.0}, X[tr], y[tr], np.ones(tr.size), X[te], s.seed)
            rec[name] = city_metrics(y, centre(p, g["run_id"].to_numpy()))
        rows.append(rec)
    if not rows:
        return {"label": "within-city block CV (interpolation-prone)", "cities": []}
    return {"label": "within-city block CV (interpolation-prone: held-out blocks share the city's streets, weather "
                     "and campaign with the training blocks)", "block_m": s.block_m, "cities": rows,
            "median_r2": {m: float(np.median([r[m]["r2"] for r in rows])) for m in ("A", "B")},
            "mean_diff_mse_B_minus_A": float(np.mean([r["B"]["mse"] - r["A"]["mse"] for r in rows]))}


# --------------------------------------------------------------------------- #
# The harness                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_window(cities: list[CityData], run: str, s: Settings, cache: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Leave-one-city-out A, B, C (and C0) on one run window; (results, per-cell predictions)."""
    t0 = time.perf_counter()
    tabs = [c.table[c.table["run"] == run] for c in cities]
    keep = [c for c, t in zip(cities, tabs) if len(t) >= MIN_CELLS]
    df = design(pd.concat([c.table[c.table["run"] == run] for c in keep], ignore_index=True)) if keep else pd.DataFrame()
    res: dict = {"run": run, "cities": [c.slug for c in keep]}
    if len(keep) < MIN_CITIES:
        res.update(status="not run", reason=f"{len(keep)} cities with ≥ {MIN_CELLS} driven cells in this run (need "
                                            f"{MIN_CITIES})")
        return res, pd.DataFrame()
    ok = np.isfinite(df[B_COLS + ["dTa"]].to_numpy(float)).all(axis=1)
    res["n_cells_total"] = int(len(df))
    res["n_cells_dropped_missing_predictor"] = int((~ok).sum())
    df = df[ok].reset_index(drop=True)
    by_city = {c.slug: c for c in keep}
    slugs = [c for c in sorted(df["slug"].unique()) if (df["slug"] == c).sum() >= MIN_CELLS]
    if len(slugs) < MIN_CITIES:
        res.update(status="not run", reason=f"{len(slugs)} cities with ≥ {MIN_CELLS} cells with every predictor")
        return res, pd.DataFrame()
    df = df[df["slug"].isin(slugs)].reset_index(drop=True)
    preds = {m: np.full(len(df), np.nan) for m in ("A", "B", "C", "C0")}
    chosen = {}
    fits = {}
    if s.physics:
        for v in s.physics_variants:
            fits[v] = physics_fits([by_city[c] for c in slugs], run, v == "C", s,
                                   None if cache is None else cache / "physics")
    per_city = []
    for h in slugs:
        te = (df["slug"] == h).to_numpy()
        train = _subsample(df[~te], s.max_rows_per_city, s.seed)
        rec = {"slug": h, "city": str(df.loc[te, "city"].iloc[0]), "n": int(te.sum())}
        for name, cols in (("A", A_COLS), ("B", B_COLS)):
            (kind, params), table = inner_select(train, cols, s)
            p = _fit_predict(kind, params, train[cols].to_numpy(float), train["dTa"].to_numpy(float),
                             _weights(train["slug"].to_numpy()), df.loc[te, cols].to_numpy(float), s.seed)
            preds[name][te] = centre(p, df.loc[te, "run_id"].to_numpy())
            chosen.setdefault(name, []).append({"held_out": h, "model": kind, "params": params, "inner": table})
            rec[name] = city_metrics(df.loc[te, "dTa"], preds[name][te])
            rec[name]["model"] = kind
        for v, fv in fits.items():
            others = [f for k, f in fv.items() if k != h and "error" not in f]
            if len(others) < MIN_CITIES - 1:
                rec[v] = {"error": f"{len(others)} training cities with a physics fit"}
                continue
            tp = transfer_params(others, v == "C")
            try:
                pc = predict_physics(by_city[h], run, v == "C", tp, s)
            except Exception as exc:                  # noqa: BLE001
                rec[v] = {"error": str(exc).splitlines()[0][:200] if str(exc) else repr(exc)}
                continue
            # the physics predicts every driven cell; score the cells A and B are scored on
            t_all = by_city[h].table[by_city[h].table["run"] == run]
            m = pd.Series(pc, index=pd.MultiIndex.from_arrays([t_all["run_id"], t_all["cell"]]))
            idx = pd.MultiIndex.from_arrays([df.loc[te, "run_id"], df.loc[te, "cell"]])
            preds[v][te] = centre(m.reindex(idx).to_numpy(), df.loc[te, "run_id"].to_numpy())
            rec[v] = city_metrics(df.loc[te, "dTa"], preds[v][te])
            rec[v]["transfer"] = {"a_phys": tp["a_phys"], "L_m": tp["natural"]["L"], **({"h_lst": tp["h_lst"]}
                                                                                       if v == "C" else {})}
        if "lst_offset_days" in df:
            off = df.loc[te, "lst_offset_days"]
            rec["lst_offset_days"] = None if off.isna().all() else float(off.abs().median())
        per_city.append(rec)
    res["per_city"] = per_city
    res["chosen"] = {k: [{kk: vv for kk, vv in c.items() if kk != "inner"} for c in v] for k, v in chosen.items()}
    res["inner_cv"] = {k: [c["inner"] for c in v] for k, v in chosen.items()}

    def pair(x: str, base: str) -> tuple[np.ndarray, np.ndarray]:
        rows = [r for r in per_city if "mse" in (r.get(x) or {}) and "mse" in (r.get(base) or {})]
        return np.array([r[x]["mse"] for r in rows]), np.array([r[base]["mse"] for r in rows])

    res["summary"] = {}
    for m in ("A", "B", "C", "C0"):
        rows = [r[m] for r in per_city if "mse" in (r.get(m) or {})]
        if rows:
            res["summary"][m] = {"n_cities": len(rows), "mean_rmse": float(np.mean([x["rmse"] for x in rows])),
                                 "median_r2": float(np.median([x["r2"] for x in rows])),
                                 "median_r": float(np.nanmedian([x["r"] for x in rows]))}
    res["paired"] = {"B-A": paired(*pair("B", "A"), s.boot, s.seed)}
    if "C" in res["summary"]:
        res["paired"]["C-A"] = paired(*pair("C", "A"), s.boot, s.seed + 1, "C − A", "physics with LST beats land cover")
    if "C" in res["summary"] and "C0" in res["summary"]:
        res["paired"]["C-C0"] = paired(*pair("C", "C0"), s.boot, s.seed + 2, "C − C0",
                                       "LST adds information inside the physics")
    res["verdict"] = res["paired"]["B-A"]["verdict"]
    offs = [(r["lst_offset_days"], r["B"]["mse"] - r["A"]["mse"]) for r in per_city if r.get("lst_offset_days") is not None]
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


TRANSFER_RULE = ("Model C is fitted once per city and run on that city's own driven cells.  For a held-out city "
                 "every nonlinear physics parameter and the lapse coefficient are the medians of the training "
                 "cities' fits; the gain and the LST coupling are transferred in physical units (a/forcing and "
                 "c_lst·forcing) and converted with the held-out city's own run forcing; the intercept and the "
                 "linear trend are 0; predictions are centred to median zero over the run's cells.  No held-out "
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
        L += ["| comparison | mean ΔMSE (°C²) | 95% CI (bootstrap over cities) | relative | cities better / worse | verdict |",
              "|---|---|---|---|---|---|"]
        for k, p in w["paired"].items():
            if "ci95" not in p:
                L.append(f"| {p['comparison']} | — | — | — | — | {p['verdict']} |")
                continue
            rel = "—" if p.get("rel_change") is None else f"{100 * p['rel_change']:+.0f}%"
            L.append(f"| {p['comparison']} | {p['mean_diff_mse']:+.4f} | {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f} | "
                     f"{rel} | {p['n_better']} / {p['n_worse']} | {p['verdict']} |")
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
            L += ["", f"Secondary, {wc['label']}: median R² A {_f(wc['median_r2']['A'])}, B {_f(wc['median_r2']['B'])}; "
                      f"mean ΔMSE (B − A) {wc['mean_diff_mse_B_minus_A']:+.4f} °C²."]
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
    (out / "results.json").write_text(json.dumps(res, indent=1, default=_json), encoding="utf-8")
    (out / "results.md").write_text(results_markdown(res), encoding="utf-8")
    if len(pred):
        pred.to_parquet(out / "predictions.parquet", index=False)
    return res


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


__all__ = ["Settings", "design", "A_COLS", "B_COLS", "LST_COLS", "city_metrics", "paired", "centre", "inner_select",
           "fit_city_physics", "transfer_params", "predict_physics", "physics_fits", "evaluate_window", "evaluate",
           "results_markdown", "run_evaluation", "TRANSFER_RULE", "MIN_VERDICT_CITIES"]
