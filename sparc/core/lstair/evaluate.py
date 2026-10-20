"""Models of the traverse air-temperature anomaly, scored on cities they never saw: the pre-registered analysis of
``docs/results/lst_air_pilot.md``.

Target: ``dTa`` (°C), each run's cell-mean traverse air temperature minus the run's city background (its median
over the driven cells; :mod:`~sparc.core.lstair.dataset`).  Every model is evaluated **leave-one-campaign-out**: fitted
on the other cities, scored on the scored cells of the held-out city, run window by run window (af first: the
primary window; then am, pm and ni where enough cities have them).  The held-out unit is the **campaign group**
(:func:`campaign_groups`): catalog "cities" that are one campaign (same OSF node and date, grids within 50 km, e.g.
Bronx and Manhattan) or neighbours (grids within 10 km: New York and its boroughs, the two Kansas Citys) are held
out together, and the paired comparison treats each group, not each city, as one independent unit.

Runs scored
-----------
Car-mounted runs with an identifiable drift, after the data checks: rows whose ``qa_exclude`` is not empty (the
reason their run is excluded: a single-vehicle run, a walking campaign, rain over most routes; written by the
dataset's QA step) are dropped before anything else and listed in the results; ``include_flagged`` keeps them
(a sensitivity).  A table without the column is all usable.  A run with fewer than :data:`MIN_CELLS` scored cells
is not scored.

Models
------
* **N — no anomaly.**  Predicts zero after centring: the floor any model must clear (its skill is 0).
* **A — land cover.**  Canopy, impervious, NDVI, albedo, elevation, log(1 + distance to water), and the Gaussian
  focal means of canopy, impervious, NDVI and albedo at 100, 300 and 1000 m, each as a within-run anomaly (minus
  the run's median over its scored cells: the target carries no city-level offset, so the predictors do not
  either), plus absolute canopy and impervious so saturating responses can be learned.  Two families: an additive
  spline ridge (cubic B-splines per predictor, a GAM-style smoother; Hastie & Tibshirani 1990), **clamped outside
  the training range** (``extrapolation="constant"``: a linear tail carried Los Angeles's elevation anomaly, SD
  140 m against 6–46 m elsewhere, far past every training city), and histogram gradient boosting; the family and
  its regularisation are chosen by an **inner leave-city-out CV** on the training cities, by mean skill (below)
  so that no single city decides the choice either.  **Training weights**: each training city weighs equally
  relative to its own variance, sample weight ∝ 1/(n_city · var_city(dTa)) from training labels only (mean 1;
  ``weights="n"``: ∝ 1/n, the sensitivity), and at most ``max_rows_per_city`` cells per city train a model.
* **B — A + LST**: the same, plus the LST anomaly of the cell and its 100 and 300 m focal means (minus the run's
  median LST).  A and B are scored on the same cells (cells with every A and B predictor).
* **C — physics with LST.**  :class:`sparc.core.physics.PhysicsModel` with the ``lst`` role: LST enters the heat
  source as surface–air coupling (q += c_lst⁺·(LST − T_bg)_c, spread by the same Green's function) next to the
  land-cover source, on the city's full raster (cells with features but no traverse feed the source, never the
  loss).  It is fitted **once per city and window** on that city's own cells of one run (the window's run with
  the most cells, with that run's ERA5 forcing and background; :func:`physics_run_id`; no advection: no wind
  record is assumed), and a held-out city's runs are each predicted with their own forcing.  **Transfer rule** to
  a held-out city: every nonlinear parameter (L, shading, storage, evaporative fraction, water sink) and the lapse
  coefficient γ is the **median over the training cities' fits**; the gain is transferred in physical units,
  a/forcing, and the LST term as the median over cities of the product a_phys·h_lst (the air's K-per-K response
  to LST, which the data determine; a and c_lst trade off within a fit), both converted back with the held-out
  city's own forcing (ERA5 SW↓ of the run, else a default per run); the intercept and the linear trend are
  city-specific and set to 0.  ``physics_transfer="medoid"`` (a sensitivity) takes instead the training city
  whose whole fitted parameter vector is most central (:func:`medoid_fit`).  The regime is fixed per window (af
  and am day, pm and ni night: :data:`WINDOW_REGIME`), so every city of a window is fitted, and transferred, with
  one source formula; parameters the regime's source does not use are listed as unused, never transferred as
  information.  No label of the held-out city is used.  Per window the results count the fits whose L sits
  within 5% of a bound (the cell size below, ``L_max`` above).
* **C0** is the same physics without LST.  C − C0 shows what LST adds *inside the physics*, not beyond land
  cover: the physics' land-cover source has a fixed form, and an LST that is only a function of land cover can
  still make up for what that form misses (the toy worlds of the tests show it).  The test of the claim is
  B − A, whose land-cover model is flexible and tuned.

**Centring.**  The predictor filter (cells with every A and B predictor) comes first; every predictor anomaly,
the LST anomaly and the target are then centred on the cells that are scored (:func:`window_table`,
:func:`recentre_target`), and every model's predictions are centred to median zero over each run's scored
cells.  Centring uses positions, never labels.  (Centred over all driven cells, Seattle's scored cells, where
LST exists, had a median impervious of 69% against the run's 53%: an offset in every anomaly.)

Scores and verdict
------------------
Per held-out unit and run, skill SS_M = 1 − MSE_M / MSE_N, with MSE_N the mean square of the centred target on
the scored cells (model N's MSE, so SS_N = 0; it is the target's variance up to the square of its mean, which
the median centring leaves near 0; Murphy 1988), averaged over the unit's runs of the window.  The comparisons
are ΔSS = SS_B − SS_A (the claim "LST adds information beyond land cover"), SS_C − SS_A (the physics claim) and
SS_C − SS_C0 (what LST adds inside the physics), per unit: a one-sided sign-flip test over units
(:func:`sign_flip`; exact up to 16 units) and a 95% t-interval (n − 1 df) of the mean (:func:`paired_skill`).
A claim (:func:`skill_verdict`, :func:`finalise`) needs all of: the sign-flip p ≤ 0.025 after **Holm's
correction** over the registered family, the three windows am, af, pm × the two claims B − A and C − A (a member
not run or not testable counts with p = 1), the t-interval above 0, mean ΔSS ≥ :data:`MIN_DSS` (0.05: 5% of
the target variance) and the model beating N on average (mean SS_B > 0, resp. SS_C > 0).  The **af** window is
primary; the others get Holm-adjusted p too and are secondary.  C − C0 is outside the family (unadjusted p).
With fewer than :data:`MIN_VERDICT_CITIES` (8) units the verdict is "not testable".  Each claim's
**leave-one-unit-out influence** (:func:`influence`) recomputes mean ΔSS, interval, p and verdict without each unit
and names any unit whose removal flips the verdict.  The absolute ΔMSE (°C², :func:`paired`) is kept as a
secondary row, never used for the claim: in the 7-city smoke run Los Angeles alone made −3.82 of its summed
−4.22 °C².  A secondary within-city block CV (1 km blocks) is reported and labelled interpolation-prone: held-out
blocks share the city's weather, streets and campaign with the training blocks, and CAPA's own maps are built
this way (Shandas et al. 2019).

Sensitivities (reported, never used for the claim; :data:`VARIANTS`, :func:`run_sensitivity`): weights ∝ 1/n;
the node + date link at any distance (the eight Virginia cities of 15 July 2021 as one group); flagged runs
included; B with the local LST residual only (LST minus its 3 km focal mean); A and B with regional controls (3 km
focal elevation and impervious); NLCD tree canopy and percent impervious in place of WorldCover; physics at 30 m
cells; the medoid physics transfer.

References: Fisher, R.A. (1935), *The Design of Experiments*, Oliver & Boyd (the randomisation test); Good, P.
(2005), *Permutation, Parametric and Bootstrap Tests of Hypotheses*, 3rd ed., Springer; Hastie, T. &
Tibshirani, R. (1990), *Generalized Additive Models*, Chapman & Hall; Ho et al. (2014), *RSE* 154, 38–45,
doi:10.1016/j.rse.2014.08.012 (random forest for daytime maximum air temperature, RMSE 2.31 °C); Holm, S. (1979),
*Scand. J. Stat.* 6, 65–70 (the step-down correction); Murphy, A.H. (1988), *Mon. Weather Rev.* 116, 2417–2424
(MSE skill scores); Roberts et al. (2017), *Ecography* 40(8), 913–929, doi:10.1111/ecog.02881 (block
cross-validation for dependent data).
"""

from __future__ import annotations

import dataclasses
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
from sparc.core.lstair.dataset import (
    FEATURES,
    FOCAL_FEATURES,
    FOCAL_RADII,
    LST_RADII,
    CityData,
    focal_mean,
    load_dataset,
)

log = logging.getLogger(__name__)

MIN_CITIES = 3                 # leave-one-city-out needs at least two training cities
MIN_VERDICT_CITIES = 8         # fewer held-out units: no verdict (with 5 the smallest sign-flip p is 1/32)
ALPHA = 0.025                  # one-sided level of the paired test (the claim is a one-sided improvement)
MIN_DSS = 0.05                 # the claim needs a mean skill gain of at least 0.05 (5% of the target variance)
MIN_REL_GAIN = 0.05            # secondary absolute ΔMSE: a relative MSE reduction of at least 5%
EXACT_SIGN_FLIP_MAX = 16       # up to this many units every sign vector is enumerated (2^16)
GROUP_KM = 10.0                # cities whose grids come this close are one campaign group (held out together)
SAME_NODE_KM = 50.0            # one OSF node and date links two catalog cities only when their grids come this close
MIN_CELLS = 30                 # a run with fewer scored cells is not scored
REGIONAL_M = 3000              # radius of the regional focal mean (σ = 1.5 km, as dataset.focal_mean): LST split, controls
L_BOUND_TOL = 0.05             # a physics fit within 5% of an L bound is counted as sitting on it
DEFAULT_FORCING = {"af": (650.0, -90.0), "am": (150.0, -60.0), "pm": (40.0, -80.0), "ni": (0.0, -80.0)}

PRIMARY_WINDOW = "af"                          # the registered primary window (15–16 h)
HOLM_WINDOWS = ("am", "af", "pm")              # Holm's family: these windows × HOLM_CLAIMS
HOLM_CLAIMS = ("B-A", "C-A")
CLAIMS = {"B-A": "LST adds information beyond land cover", "C-A": "physics with LST beats land cover",
          "C-C0": "LST adds information inside the physics"}
COMPARISONS = {"B-A": ("B", "A", "B − A"), "C-A": ("C", "A", "C − A"), "C-C0": ("C", "C0", "C − C0")}
MODELS = ("N", "A", "B", "C", "C0")

CENTRED = (["canopy", "impervious", "ndvi", "albedo", "elevation", "log_water"]
           + [f"{k}_f{r}" for k in FOCAL_FEATURES for r in FOCAL_RADII])
REGIONAL = [f"elevation_f{REGIONAL_M}", f"impervious_f{REGIONAL_M}"]      # the regional controls (``regional``)
A_COLS = [f"{k}_c" for k in CENTRED] + ["canopy", "impervious"]
LST_COLS = ["lst_c"] + [f"lst_f{r}_c" for r in LST_RADII]
B_COLS = A_COLS + LST_COLS
#: ``land_cover="nlcd"``: the dataset's continuous US columns (percent 0–100) that stand in for WorldCover's
NLCD_SWAP = {k: v for base, new in (("canopy", "tcc"), ("impervious", "nlcd_imp"))
             for k, v in [(base, new)] + [(f"{base}_f{r}", f"{new}_f{r}") for r in FOCAL_RADII]}

WEIGHTS = ("variance", "n")
LST_MODES = ("full", "local")
LAND_COVERS = ("worldcover", "nlcd")
TRANSFERS = ("median", "medoid")


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
    weights: str = "variance"                   # "variance": ∝ 1/(n·var(dTa)) per city (registered); "n": ∝ 1/n
    same_node_km: float | None = SAME_NODE_KM   # None: the node + date link at any distance (Virginia as one group)
    include_flagged: bool = False               # keep runs the data checks flagged (qa_exclude)
    lst_mode: str = "full"                      # "local": B's LST minus its 3 km focal mean
    regional: bool = False                      # A and B also get 3 km focal elevation and impervious anomalies
    land_cover: str = "worldcover"              # "nlcd": NLCD tree canopy and percent impervious instead
    physics: bool = True
    physics_cell_m: float = 60.0
    physics_iter: int = 60
    physics_L_max_m: float = 2000.0
    physics_variants: tuple = ("C", "C0")
    physics_transfer: str = "median"            # "medoid": the most central training city's parameters, whole
    block_m: float = 1000.0
    within_city: bool = True

    def __post_init__(self):
        for name, allowed in (("weights", WEIGHTS), ("lst_mode", LST_MODES), ("land_cover", LAND_COVERS),
                              ("physics_transfer", TRANSFERS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"Settings.{name} must be one of {allowed}, not {getattr(self, name)!r}")


# --------------------------------------------------------------------------- #
# Design                                                                       #
# --------------------------------------------------------------------------- #
def model_columns(s: Settings | None = None) -> tuple[list[str], list[str]]:
    """The A and B model columns under ``s`` (``regional`` adds the 3 km elevation and impervious anomalies)."""
    s = s or Settings()
    if not s.regional:
        return list(A_COLS), list(B_COLS)
    a = [f"{k}_c" for k in CENTRED + REGIONAL] + ["canopy", "impervious"]
    return a, a + LST_COLS


def raw_predictors(s: Settings | None = None) -> tuple[list[str], list[str]]:
    """The table columns a cell needs for A, and in addition for B, under ``s`` (the predictor filter)."""
    s = s or Settings()
    a = [k if k != "log_water" else "water_distance" for k in CENTRED] + (REGIONAL if s.regional else [])
    lst = ["lst"] + [f"lst_f{r}" for r in LST_RADII] + ([f"lst_f{REGIONAL_M}"] if s.lst_mode == "local" else [])
    return a, lst


def design(table: pd.DataFrame, s: Settings | None = None) -> pd.DataFrame:
    """The table with the model columns: within-run anomalies ``<name>_c`` (minus the run's median over the rows
    given: pass the scored cells, :func:`window_table`) of the land-cover, focal and LST predictors;
    ``log_water`` = log(1 + distance to water).  ``s.lst_mode="local"``: the LST columns are LST, and its 100 and
    300 m focal means, minus the 3 km focal mean (``lst_f3000``), centred on the run's median local residual."""
    s = s or Settings()
    df = table.copy()
    df["log_water"] = np.log1p(np.clip(df["water_distance"].to_numpy(float), 0.0, None))
    keys = [df["slug"], df["run_id"]]
    g = df.groupby(keys, sort=False)
    for k in CENTRED + (REGIONAL if s.regional else []):
        df[f"{k}_c"] = df[k] - g[k].transform("median")
    reg = df[f"lst_f{REGIONAL_M}"] if s.lst_mode == "local" else 0.0
    base = df["lst"] - reg
    med = base.groupby(keys, sort=False).transform("median")
    df["lst_c"] = base - med
    for r in LST_RADII:
        df[f"lst_f{r}_c"] = df[f"lst_f{r}"] - reg - med
    return df


def _swap_land_cover(c: CityData, t: pd.DataFrame, need_rasters: bool) -> tuple[pd.DataFrame, dict] | None:
    """``c``'s table and rasters with WorldCover's canopy and impervious (and focal means) replaced by the NLCD
    columns; None when the city lacks them."""
    if not set(NLCD_SWAP.values()) <= set(t.columns):
        return None
    if need_rasters and not {"tcc", "nlcd_imp"} <= set(c.rasters):
        return None
    t = t.copy()
    for base, new in NLCD_SWAP.items():
        t[base] = t[new]
    R = dict(c.rasters)
    for base, new in NLCD_SWAP.items():
        if new in c.rasters:
            R[base] = c.rasters[new]
    return t, R


def _regional_column(c: CityData, t: pd.DataFrame, src: str, R: dict) -> np.ndarray:
    """The 3 km focal mean (NaN-aware, :func:`~sparc.core.lstair.dataset.focal_mean`) of raster ``src`` at the
    table's cells; NaN for a city without LST (no ``lst`` raster and no LST in its table).  A missing raster is
    otherwise an error, not a silent drop of every cell (load the dataset with its rasters)."""
    cells = t["cell"].to_numpy(np.int64)
    if src not in R:
        if src == "lst" and not ("lst" in t and np.isfinite(t["lst"].to_numpy(float)).any()):
            return np.full(len(cells), np.nan)
        raise ValueError(f"{c.slug}: no {src} raster for its {REGIONAL_M} m focal mean (load the dataset with its "
                         "rasters)")
    return focal_mean(R[src], REGIONAL_M, c.grid.dx).ravel()[cells]


def prepare(cities: list[CityData], s: Settings) -> tuple[list[CityData], list[dict]]:
    """The cities as the evaluation sees them under ``s``: rows of runs the data checks flagged (``qa_exclude``
    not empty) dropped unless ``include_flagged``; ``land_cover="nlcd"`` swaps in the NLCD columns and rasters;
    the 3 km focal columns ``lst_mode="local"`` and ``regional`` need are computed from the rasters (no rebuild).
    Returns the cities and the flagged runs (slug, run, run_id, cells, reason, whether included).

    ``land_cover="nlcd"`` refuses (ValueError) when a city that would be scored lacks the NLCD columns: any city
    with at least :data:`MIN_CELLS` usable cells with LST (with every cell when no city has LST); a city that is
    not scored anyway (no LST over its cells) is left out as it would be."""
    flagged, usable = [], []
    for c in cities:
        t = c.table
        if "qa_exclude" in t:
            why = t["qa_exclude"].fillna("").astype(str).str.strip()
            bad = (why != "").to_numpy()
            if bad.any():
                for (run, rid), part in t[bad].groupby(["run", "run_id"], sort=True):
                    flagged.append({"slug": c.slug, "run": str(run), "run_id": str(rid), "n_cells": int(len(part)),
                                    "reason": str(why[part.index].iloc[0]), "included": bool(s.include_flagged)})
                if not s.include_flagged:
                    t = t[~bad].reset_index(drop=True)
        usable.append(t)
    rasters = [c.rasters for c in cities]
    if s.land_cover == "nlcd":
        any_lst = any("lst" in t and np.isfinite(t["lst"].to_numpy(float)).any() for t in usable)
        lacking = []
        for i, (c, t) in enumerate(zip(cities, usable)):
            sw = _swap_land_cover(c, t, s.physics)
            if sw is not None:
                usable[i], rasters[i] = sw
                continue
            if not any_lst:
                scored = len(t)
            else:
                scored = int(_finite(t, [k for k in raw_predictors(s)[1] if k != f"lst_f{REGIONAL_M}"]).sum())
            if scored >= MIN_CELLS:
                lacking.append(c.slug)
            else:                                    # not scored anyway: its land cover is unknown, not WorldCover's
                usable[i] = t.assign(**{k: np.nan for k in NLCD_SWAP})
        if lacking:
            raise ValueError(f"--land-cover nlcd: {', '.join(lacking)} lack the NLCD columns "
                             f"({', '.join(sorted(set(NLCD_SWAP.values())))}"
                             + (" and the tcc / nlcd_imp rasters" if s.physics else "") + "); every scored city "
                             "needs them (add them with the US land-cover step, or evaluate with --land-cover "
                             "worldcover)")
    extra = ({f"lst_f{REGIONAL_M}": "lst"} if s.lst_mode == "local" else {}) | (
        {f"elevation_f{REGIONAL_M}": "elevation", f"impervious_f{REGIONAL_M}": "impervious"} if s.regional else {})
    out = []
    for c, t, R in zip(cities, usable, rasters):
        add = {col: _regional_column(c, t, src, R) for col, src in extra.items() if col not in t}
        if add:
            t = t.assign(**add)
        out.append(c if (t is c.table and R is c.rasters) else
                   CityData(slug=c.slug, meta=c.meta, table=t, grid=c.grid, rasters=R))
    return out, flagged


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
        # clamped: outside the training range a predictor's spline holds its boundary value (a linear tail
        # extrapolated Los Angeles's elevation anomaly, SD 140 m against 6–46 m elsewhere)
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3, extrapolation="constant"),
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


def _weights(slugs: np.ndarray, y: np.ndarray | None = None, mode: str = "variance") -> np.ndarray:
    """Training sample weights, scaled to mean 1 (so a ridge penalty keeps its meaning per sample).

    ``"variance"`` (registered): ∝ 1/(n_city · var_city(y)), so each training city weighs equally *relative to
    its own variance*: with ∝ 1/n, Los Angeles's afternoon (dTa SD 3.5 °C against 0.3–1.1 °C elsewhere) held most
    of the squared error any fit minimises.  ``y`` are the training labels (never a held-out city's).  ``"n"``:
    ∝ 1/n, equal total weight per city (the sensitivity)."""
    s = pd.Series(np.asarray(slugs))
    n_c = s.map(s.value_counts()).to_numpy(float)
    w = 1.0 / n_c
    if mode == "variance":
        if y is None:
            raise ValueError("variance weights need the training labels")
        var = pd.Series(np.asarray(y, float)).groupby(s.to_numpy()).var(ddof=0)
        w = w / np.maximum(s.map(var).to_numpy(float), 1e-6)
    elif mode != "n":
        raise ValueError(f"weights must be one of {WEIGHTS}")
    return w * (len(w) / w.sum())


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df
    parts = []
    for i, (_, g) in enumerate(df.groupby("slug", sort=True)):
        parts.append(g if len(g) <= n else g.sample(n, random_state=seed + i))
    return pd.concat(parts)


def _r(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.corrcoef(y, p)[0, 1]) if y.size > 2 and np.std(p) > 0 and np.std(y) > 0 else float("nan")


def city_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    mse = float(np.mean(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(y.size), "mse": mse, "rmse": math.sqrt(mse), "r2": 1.0 - float(np.sum(e ** 2)) / sst if sst > 0
            else float("nan"), "r": _r(y, p)}


def skill(y: np.ndarray, p: np.ndarray) -> float:
    """SS = 1 − MSE/MSE_N on one run's scored cells; MSE_N = mean(y²) is model N's (zero prediction) MSE."""
    y = np.asarray(y, float)
    mse_n = float(np.mean(y ** 2))
    return 1.0 - float(np.mean((np.asarray(p, float) - y) ** 2)) / mse_n if mse_n > 0 else float("nan")


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


def campaign_groups(cities: list[CityData], near_km: float = GROUP_KM,
                    same_node_km: float | None = SAME_NODE_KM) -> dict[str, str]:
    """The unit of independence: cities of one campaign or metro area form a group, held out together and resampled
    as one.  Two cities are linked when they share an OSF node and a campaign date **and** their grids lie within
    ``same_node_km`` (50 km) of each other (one campaign published as several catalog "cities", e.g. Bronx and
    Manhattan in j6eqr), or when their grids lie within ``near_km`` of each other (New York beside its boroughs,
    Kansas City KS beside MO): their held-out traverses share the day's weather, the vehicles and sensors, and
    neighbouring streets.  Returns slug → group id (the member slugs joined by ``+``; a city alone is its own group).

    The node + date link once ignored distance.  VFIC's node 3xvmg holds eight separate city campaigns of 15 July
    2021 (Abingdon to Virginia Beach, grids 19–505 km apart, each with its own sensors) and NIHHIS's tdsy7 holds
    Boise and Sacramento (642 km apart) on one date, so 31 cities formed 23 units instead of 30 (review of
    2026-10-20).  That merge cost power and down-weighted those cities (eight cities averaged into one pair); it
    was conservative, not anti-conservative.  ``same_node_km=None`` restores it: the "Virginia as one group"
    sensitivity (same day, regional weather and, for three of them, one Landsat scene)."""
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
            lat = math.radians(0.5 * (ba[1] + bb[3])) if np.isfinite(ba[1]) and np.isfinite(bb[3]) else 0.0
            gx = max(0.0, max(ba[0], bb[0]) - min(ba[2], bb[2])) * 111.32 * math.cos(lat)
            gy = max(0.0, max(ba[1], bb[1]) - min(ba[3], bb[3])) * 110.57
            gap = math.hypot(gx, gy) if np.isfinite(gx) and np.isfinite(gy) else float("nan")
            same = na is not None and na == nb and da is not None and da == db and (
                same_node_km is None or (np.isfinite(gap) and gap <= same_node_km))
            near = np.isfinite(gap) and gap <= near_km
            if same or near:
                parent[find(a)] = find(b)
    members: dict[str, list[str]] = {}
    for sl, *_ in info:
        members.setdefault(find(sl), []).append(sl)
    return {sl: "+".join(sorted(m)) for m in members.values() for sl in m}


def inner_select(train: pd.DataFrame, cols: list[str], s: Settings) -> tuple[tuple[str, dict], list[dict]]:
    """The candidate with the highest mean skill (over inner held-out cities, each the mean over its runs of
    1 − MSE/MSE_N) in a leave-group-out CV on ``train`` (campaign groups, :func:`campaign_groups`, when the table
    has a ``group`` column, else cities; grouped K-fold when there are more than ``inner_folds``).  Skill, not
    MSE, so that the most variable city does not choose the model for all; the inner MSE is listed too.  Each
    fold's weights come from its own training rows (:func:`_weights`)."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    groups = train["group" if "group" in train else "slug"].to_numpy()
    slugs = train["slug"].to_numpy()
    runs = train["run_id"].astype(str).to_numpy()
    k = np.unique(groups).size
    split = LeaveOneGroupOut() if k <= s.inner_folds else GroupKFold(n_splits=s.inner_folds)
    X, y = train[cols].to_numpy(float), train["dTa"].to_numpy(float)
    rows = []
    folds = list(split.split(X, y, groups))
    for kind, params in _candidates(s):
        per_city, per_city_ss = [], []
        for tr, te in folds:
            w = _weights(slugs[tr], y[tr], s.weights)
            p = centre(_fit_predict(kind, params, X[tr], y[tr], w, X[te], s.seed), runs[te] + "|" + slugs[te])
            for c in np.unique(slugs[te]):
                sel = slugs[te] == c
                per_city.append(float(np.mean((p[sel] - y[te][sel]) ** 2)))
                ss = [skill(y[te][sel & (runs[te] == r)], p[sel & (runs[te] == r)]) for r in np.unique(runs[te][sel])]
                ss = [v for v in ss if np.isfinite(v)]
                if ss:
                    per_city_ss.append(float(np.mean(ss)))
        rows.append({"model": kind, "params": params, "inner_ss": float(np.mean(per_city_ss)) if per_city_ss
                     else float("nan"), "inner_mse": float(np.mean(per_city))})
    best = max(rows, key=lambda r: r["inner_ss"] if np.isfinite(r["inner_ss"]) else -np.inf)
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
           "L_bounds_m": [float(math.exp(m.L_lo)), float(math.exp(m.L_hi))],
           "window": cfg["window"], "seconds": round(time.perf_counter() - t0, 2), "roles": inp["roles"]}
    if use_lst:
        rec["c_lst"] = float(T["c_lst"])
        rec["h_lst"] = float(T["c_lst"]) * norm                      # the coupling in forcing units (W m⁻² K⁻¹)
        # what the data determine: the air's response per K of LST anomaly, a·c_lst = a_phys·h_lst (K per K);
        # a and c_lst trade off within a fit, so their separate values (and any r_a read from them) are not
        # usable as aerodynamic resistances
        rec["g_lst"] = rec["a_phys"] * rec["h_lst"]
    return rec


def _fit_value(f: dict, k: str) -> float:
    if k in TRANSFER_KEYS:
        return float(f["natural"][k])
    if k == "g_lst":
        return float(f.get("g_lst", f["a_phys"] * f.get("h_lst", 0.0)))
    return float(f[k])


def medoid_fit(fits: list[dict], keys, use_lst: bool) -> tuple[dict, dict]:
    """The training fit whose parameter vector is most central: the regime's fitted ``keys``, the physical gain
    a_phys, γ and (with LST) the LST response g = a_phys·h_lst; parameters positive in every fit as logarithms;
    each standardised over the fits (constant ones dropped); the fit with the least summed Euclidean distance to
    all the others (ties: the first slug).  Returns the fit and a record of the choice."""
    names = list(keys) + ["a_phys", "gamma"] + (["g_lst"] if use_lst else [])
    V = np.array([[_fit_value(f, k) for k in names] for f in fits], float)
    ok = np.isfinite(V).all(axis=1)
    if not ok.any():
        raise ValueError("no training fit with a finite parameter vector")
    cand = [f for f, o in zip(fits, ok) if o]
    V = V[ok]
    logs = [k for j, k in enumerate(names) if (V[:, j] > 0).all()]
    for j, k in enumerate(names):
        if k in logs:
            V[:, j] = np.log(V[:, j])
    sd = V.std(axis=0)
    use = sd > 1e-12 * np.maximum(np.abs(V).max(axis=0), 1.0)
    Z = (V[:, use] - V[:, use].mean(axis=0)) / sd[use] if use.any() else np.zeros((len(cand), 1))
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)).sum(axis=1)
    i = min(range(len(cand)), key=lambda j: (round(float(D[j]), 12), str(cand[j]["slug"])))
    return cand[i], {"medoid": cand[i]["slug"], "vector": [k for k, u in zip(names, use) if u], "log": logs,
                     "summed_distance": {str(f["slug"]): float(d) for f, d in zip(cand, D)}}


def transfer_params(fits: list[dict], use_lst: bool, rule: str = "median") -> dict:
    """The transfer rule: medians over the training cities' fits (gain and coupling in physical units).

    The LST term's effect on the air is a·c_lst·ΔLST = a_phys·h_lst·ΔLST, and within a fit a and c_lst trade off
    (anti-correlated across cities), so the separate medians of a_phys and h_lst misstate their product.  The
    transferred LST gain is the **median of the per-city products** g = a_phys·h_lst (K per K), and h_lst is set
    to g / median(a_phys), so the held-out city gets a·c_lst = g; median(a_phys) itself serves the land-cover
    source only.  With median(a_phys) ≤ 0 the LST term cannot be carried and h_lst is 0 (recorded).

    ``rule="medoid"`` (a sensitivity): per-parameter medians can combine values no city has together; the medoid
    fit (:func:`medoid_fit`) is a coherent vector and is used whole (its parameters, a_phys, γ and h_lst)."""
    want = regime_of_fits(fits)
    if want is not None:
        fits = [f for f in fits if f.get("window", want) == want]          # one source formula, never a mix
    keys = REGIME_KEYS.get(want or "day", TRANSFER_KEYS)
    common = {"n_cities": len(fits), "cities": [f["slug"] for f in fits], "regime": want, "rule": rule,
              "fitted_keys": list(keys), "unused_keys": [k for k in TRANSFER_KEYS if k not in keys]}
    if rule == "medoid":
        f, why = medoid_fit(fits, keys, use_lst)
        out = {"natural": {k: float(f["natural"][k]) for k in TRANSFER_KEYS}, "a_phys": float(f["a_phys"]),
               "gamma": float(f["gamma"]), **common, **why}
        if use_lst:
            out["h_lst"] = float(f["h_lst"])
            out["g_lst"] = out["a_phys"] * out["h_lst"]
        return out
    if rule != "median":
        raise ValueError(f"physics transfer must be one of {TRANSFERS}")
    nat = {k: float(np.median([f["natural"][k] for f in fits])) for k in TRANSFER_KEYS}
    a_med = float(np.median([f["a_phys"] for f in fits]))
    out = {"natural": nat, "a_phys": a_med, "gamma": float(np.median([f["gamma"] for f in fits])), **common}
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
    h.update(json.dumps([city.meta.get("built_utc"), use_lst, s.physics_cell_m, s.physics_iter, s.physics_L_max_m]
                        + ([s.land_cover] if s.land_cover != "worldcover" else [])).encode())
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


def l_bounds_report(fits: dict, s: Settings) -> dict:
    """Per physics variant: how many fits sit within :data:`L_BOUND_TOL` (5%) of an L bound, the lower bound being
    the physics cell size and the upper ``L_max`` (or a quarter of a small grid's extent: the fit's own bounds when
    it recorded them).  Fits on a bound transfer a length the data did not determine."""
    out = {}
    for v, fv in fits.items():
        ok = [f for f in fv.values() if "error" not in f and "natural" in f]
        low, high = [], []
        for f in ok:
            lo, hi = f.get("L_bounds_m") or [f.get("cell_m", s.physics_cell_m), s.physics_L_max_m]
            L = float(f["natural"]["L"])
            if L <= lo * (1.0 + L_BOUND_TOL):
                low.append(f["slug"])
            elif L >= hi * (1.0 - L_BOUND_TOL):
                high.append(f["slug"])
        out[v] = {"n_fits": len(ok), "n_lower": len(low), "n_upper": len(high), "lower": low, "upper": high,
                  "rule": f"L within {100 * L_BOUND_TOL:.0f}% of the cell size (lower) or of L_max (upper)"}
    return out


# --------------------------------------------------------------------------- #
# Paired comparison and verdict                                                #
# --------------------------------------------------------------------------- #
def sign_flip(d: np.ndarray, draws: int = 4000, seed: int = 0) -> dict:
    """One-sided sign-flip (randomisation) test of a mean paired difference over independent units.

    Under the null that each unit's difference is symmetric about 0, every assignment of signs is equally
    likely; the p-value of an improvement (mean < 0) is the share of the 2ⁿ sign vectors whose mean is at most
    the observed one (Fisher 1935; Good 2005); ``p_greater`` is the other side (a skill gain, mean > 0).  Exact
    (all 2ⁿ vectors) up to :data:`EXACT_SIGN_FLIP_MAX` units, else ``draws`` random vectors plus the observed
    one.  With 5 units the smallest p is 1/32 > 0.025: no claim is possible, which is the honest answer."""
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
    """**Secondary** (never used for the claim): mean over held-out units of MSE_x − MSE_A (°C²), its 95%
    t-interval (n − 1 df), a one-sided sign-flip test and the absolute-MSE verdict of the earlier design.

    That verdict needed the sign-flip p of an improvement ≤ :data:`ALPHA` (0.025, unadjusted), the t-interval
    below 0 and a relative MSE change ≤ −:data:`MIN_REL_GAIN` (5%); with fewer than :data:`MIN_VERDICT_CITIES`
    units "not testable".  (A percentile bootstrap over 5–10 cities ignored the n − 1 degrees of freedom and
    claimed at about three times its stated rate under a true null.)  Absolute MSE is dominated by the most
    variable city (Los Angeles: −3.82 of the smoke run's summed −4.22 °C²), so the registered statistic is the
    skill difference (:func:`paired_skill`).  ``boot`` is the number of random sign vectors when the test cannot
    be enumerated."""
    from scipy import stats

    d = np.asarray(mse_x, float) - np.asarray(mse_a, float)
    n = d.size
    out = {"comparison": label, "n_cities": int(n), "claim": claim, "role": "secondary (never used for the claim)"}
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


def paired_skill(ss_x: np.ndarray, ss_base: np.ndarray, units: list[str] | None = None, boot: int = 4000,
                 seed: int = 0, key: str = "B-A") -> dict:
    """The registered comparison ``key`` (B-A, C-A or C-C0): ΔSS = SS_x − SS_base per held-out unit, its mean,
    the 95% t-interval (n − 1 df) and the one-sided sign-flip p of a gain (``p_sign_flip``; ``p_sign_flip_worse``
    the other side), the mean skills, and the per-unit values (``units``, for :func:`influence`).  The verdict set
    here uses the unadjusted p; :func:`finalise` replaces it with the Holm-adjusted one."""
    from scipy import stats

    x, base, label = COMPARISONS[key]
    ss_x, ss_base = np.asarray(ss_x, float), np.asarray(ss_base, float)
    d = ss_x - ss_base
    n = d.size
    units = list(units) if units is not None else [str(i) for i in range(n)]
    out = {"comparison": label, "key": key, "claim": CLAIMS[key], "x": x, "base": base, "n_units": int(n),
           "statistic": f"ΔSS = SS_{x} − SS_{base} per held-out unit (SS = 1 − MSE/MSE_N, mean over the unit's runs)",
           "draws": int(boot), "seed": int(seed), "p_holm": None, "holm": "not applied"}
    if n == 0:
        out.update(verdict="not testable", reason="no held-out unit")
        return out
    out.update(mean_dss=float(d.mean()), mean_ss_x=float(ss_x.mean()), mean_ss_base=float(ss_base.mean()),
               n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()), alpha_one_sided=ALPHA, min_dss=MIN_DSS,
               units=[{"unit": u, "ss_x": float(a), "ss_base": float(b), "d": float(a - b)}
                      for u, a, b in zip(units, ss_x, ss_base)])
    if n >= 2:
        se = float(d.std(ddof=1)) / math.sqrt(n)
        t = float(stats.t.ppf(1.0 - ALPHA, n - 1))
        out.update(ci95=[float(d.mean() - t * se), float(d.mean() + t * se)], ci_method=f"t-interval, {n - 1} df")
        sf = sign_flip(d, boot, seed)
        out.update(p_sign_flip=sf["p_greater"], p_sign_flip_worse=sf["p_less"], sign_flip=sf["method"])
    out.update(skill_verdict(out, out.get("p_sign_flip")))
    return out


def _testable(rec: dict) -> bool:
    return rec.get("n_units", 0) >= MIN_VERDICT_CITIES and "p_sign_flip" in rec and "ci95" in rec


def skill_verdict(rec: dict, p: float | None, adjusted: bool = False) -> dict:
    """The claim rule on a :func:`paired_skill` record with p-value ``p`` (Holm-adjusted when ``adjusted``):
    p ≤ :data:`ALPHA`, the 95% t-interval above 0, mean ΔSS ≥ :data:`MIN_DSS` and the model beating N on average
    (mean SS_x > 0).  ``{"verdict", "reason"}``."""
    x, n = rec.get("x", "B"), rec.get("n_units", 0)
    if n < MIN_VERDICT_CITIES or p is None or "ci95" not in rec:
        return {"verdict": "not testable", "reason": f"{n} held-out units (< {MIN_VERDICT_CITIES}): too few for a "
                                                     "test over cities"}
    pl = f"{'Holm-adjusted ' if adjusted else ''}sign-flip p = {p:.3g}"
    lo, hi = rec["ci95"]
    better = p <= ALPHA and lo > 0
    if better and rec["mean_dss"] >= MIN_DSS and rec["mean_ss_x"] > 0:
        return {"verdict": rec["claim"], "reason": f"{pl} ≤ {ALPHA}, the 95% t-interval of ΔSS lies above 0 "
                                                   f"({lo:+.3f} to {hi:+.3f}), mean ΔSS {rec['mean_dss']:+.3f} ≥ "
                                                   f"{MIN_DSS} and {x} beats N (mean SS_{x} {rec['mean_ss_x']:+.3f})"}
    if better and rec["mean_dss"] < MIN_DSS:
        return {"verdict": "not shown", "reason": f"significant ({pl}) but mean ΔSS is only {rec['mean_dss']:+.3f} "
                                                  f"(< {MIN_DSS})"}
    if better:
        return {"verdict": "not shown", "reason": f"significant ({pl}) but {x} does not beat N on average (mean "
                                                  f"SS_{x} {rec['mean_ss_x']:+.3f} ≤ 0)"}
    if rec.get("p_sign_flip_worse", 1.0) <= ALPHA and hi < 0:
        return {"verdict": "not shown", "reason": "the 95% t-interval of ΔSS lies below 0: adding it made "
                                                  "predictions worse"}
    return {"verdict": "not shown", "reason": f"{pl} (one-sided) and the 95% t-interval {lo:+.3f} to {hi:+.3f}"}


def holm(p: dict) -> dict:
    """Holm's (1979) step-down adjusted p-values: sorted ascending, p̃_(i) = max_{j ≤ i} min(1, (m − j + 1)·p_(j)).
    A family-wise level α holds when every member with p̃ ≤ α is rejected."""
    order = sorted(p, key=lambda k: (p[k], str(k)))
    m, run, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * float(p[k])))
        out[k] = run
    return out


def _family_p(windows: dict) -> dict:
    """The registered Holm family, (window, claim) → unadjusted p; a member not run or not testable counts with
    p = 1 (it cannot reject, and it still takes its share of the level)."""
    fam = {}
    for w in HOLM_WINDOWS:
        for k in HOLM_CLAIMS:
            rec = ((windows.get(w) or {}).get("paired") or {}).get(k) or {}
            fam[(w, k)] = float(rec["p_sign_flip"]) if _testable(rec) else 1.0
    return fam


def _apply(rec: dict, w: str, adj: dict, fam: dict, untested: list[str]) -> None:
    if (w, rec["key"]) in adj:
        rec["p_holm"] = adj[(w, rec["key"])]
        rec["holm"] = {"m": len(fam), "family": [f"{a} {b}" for a, b in fam], "untested_as_p1": untested}
        rec.update(skill_verdict(rec, rec["p_holm"], adjusted=True))
    else:
        rec["p_holm"] = None
        rec["holm"] = "outside the registered Holm family (unadjusted p)"
        rec.update(skill_verdict(rec, rec.get("p_sign_flip")))


def _drop_unit(rec: dict, unit: str) -> dict:
    rows = [u for u in rec.get("units", []) if u["unit"] != unit]
    return paired_skill([u["ss_x"] for u in rows], [u["ss_base"] for u in rows], [u["unit"] for u in rows],
                        rec.get("draws", 4000), rec.get("seed", 0), rec["key"])


def influence(windows: dict, w: str) -> None:
    """Leave-one-unit-out influence on window ``w``'s comparisons: without each held-out unit (dropped from every
    comparison of the window), mean ΔSS, the interval, the Holm-adjusted p (the other windows' members as they
    are) and the verdict; per comparison the ranges and the units whose removal flips the verdict."""
    P = (windows[w].get("paired") or {})
    recs = {k: r for k, r in P.items() if r.get("units")}
    units = sorted({u["unit"] for r in recs.values() for u in r["units"]})
    rows = {k: [] for k in recs}
    base = _family_p(windows)
    for u in units:
        minus = {k: _drop_unit(r, u) for k, r in recs.items()}
        fam = dict(base)
        for k in HOLM_CLAIMS:
            if (w, k) in fam and k in minus:
                fam[(w, k)] = float(minus[k]["p_sign_flip"]) if _testable(minus[k]) else 1.0
        adj = holm(fam)
        for k, m in minus.items():
            p = adj.get((w, k), m.get("p_sign_flip"))
            v = skill_verdict(m, p, adjusted=(w, k) in adj)
            rows[k].append({"dropped": u, "n_units": m["n_units"], "mean_dss": m.get("mean_dss"), "ci95": m.get("ci95"),
                            "p_sign_flip": m.get("p_sign_flip"), "p": p, "verdict": v["verdict"]})
    for k, r in recs.items():
        rs = rows[k]

        def rng(field, idx=None):
            v = [(x[field] if idx is None else x[field][idx]) for x in rs if x.get(field) is not None]
            return [float(min(v)), float(max(v))] if v else None
        flips = [x["dropped"] for x in rs if x["verdict"] != r["verdict"]]
        r["influence"] = {"rule": "each held-out unit dropped in turn (Holm over the family as it then stands; a "
                                  "window left with fewer than 8 units is not testable)",
                          "n": len(rs), "mean_dss_range": rng("mean_dss"), "ci95_low_range": rng("ci95", 0),
                          "p_range": rng("p"), "p_sign_flip_range": rng("p_sign_flip"), "flips": flips,
                          "flipped_to": {x["dropped"]: x["verdict"] for x in rs if x["dropped"] in flips},
                          "any_flip": bool(flips),
                          "verdicts": {v: sum(x["verdict"] == v for x in rs) for v in dict.fromkeys(x["verdict"] for x in rs)},
                          "rows": rs}


def finalise(windows: dict) -> None:
    """Holm-adjust the registered family across ``windows`` (run → result), set every comparison's verdict by the
    claim rule (:func:`skill_verdict`), each window's headline verdict (B − A) and the leave-one-unit-out
    influence (:func:`influence`).  Comparisons outside the family (C − C0; the ni window) keep the unadjusted p."""
    fam = _family_p(windows)
    adj = holm(fam)
    untested = [f"{a} {b}" for a, b in fam if not _testable(((windows.get(a) or {}).get("paired") or {}).get(b) or {})]
    for w, res in windows.items():
        for rec in (res.get("paired") or {}).values():
            if rec.get("units"):                      # else no LST / no unit: its verdict stands ("not testable")
                _apply(rec, w, adj, fam, untested)
    for w, res in windows.items():
        if res.get("paired"):
            res["verdict"] = (res["paired"].get("B-A") or {}).get("verdict", "not testable")
            influence(windows, w)


# --------------------------------------------------------------------------- #
# Within-city block CV (secondary)                                             #
# --------------------------------------------------------------------------- #
def within_city(df: pd.DataFrame, s: Settings) -> dict:
    """Spatial block CV inside each city (``block_m`` blocks, ≤ 5 folds), additive spline ridge (α = 3) on A and
    on B columns.  Interpolation-prone: neighbouring blocks share the city's streets, weather and campaign."""
    from sklearn.model_selection import GroupKFold

    a_cols, b_cols = model_columns(s)
    rows = []
    for slug, g in df.groupby("slug", sort=True):
        blocks = (np.floor(g["x"] / s.block_m).astype(np.int64) * 100003 + np.floor(g["y"] / s.block_m).astype(np.int64)).to_numpy()
        nb = np.unique(blocks).size
        if nb < 3 or len(g) < MIN_CELLS:
            continue
        kf = GroupKFold(n_splits=min(5, nb))
        y = g["dTa"].to_numpy(float)
        rec = {"slug": slug, "n": int(len(g)), "n_blocks": int(nb)}
        for name, cols in (("A", a_cols), ("B", b_cols)):
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
def _finite(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df]
    if missing:
        return np.zeros(len(df), bool)
    return np.isfinite(df[cols].to_numpy(float)).all(axis=1)


def window_table(cities: list[CityData], run: str, s: Settings) -> tuple[pd.DataFrame, dict]:
    """The scored cells of window ``run`` (cities as :func:`prepare` returns them) and what was left out.

    Order matters: the predictor filter (cells with every A and B predictor; A only when no city has LST) comes
    first, then runs with fewer than :data:`MIN_CELLS` scored cells are dropped, and only then are the predictor
    anomalies (:func:`design`) and the target (:func:`recentre_target`) centred, so all of them are centred on the
    cells that are scored.  Returns the designed table (empty when too few cities remain) and a record: cities,
    LST status, cells dropped, cities without LST, runs dropped, the target shifts and, when not run, the reason."""
    info: dict = {}
    keep = [c for c in cities if (c.table["run"] == run).sum() >= MIN_CELLS]
    info["cities"] = [c.slug for c in keep]
    if len(keep) < MIN_CITIES:
        info["reason"] = f"{len(keep)} cities with ≥ {MIN_CELLS} driven cells in this run (need {MIN_CITIES})"
        return pd.DataFrame(), info
    raw = pd.concat([c.table[c.table["run"] == run] for c in keep], ignore_index=True)
    ra, rl = raw_predictors(s)
    ok_a = _finite(raw, ra + ["Ta", "dTa"])
    has_lst = _finite(raw, rl)
    no_lst = not (ok_a & has_lst).any()
    info["lst"] = ("none: the dataset has no LST in this window (built with --no-lst, or no city had a clear Landsat "
                   "scene); A and C0 only") if no_lst else "present"
    ok = ok_a if no_lst else ok_a & has_lst
    info["n_cells_total"] = int(len(raw))
    info["n_cells_dropped_missing_predictor"] = int((~ok).sum())
    n_all = raw.groupby(["slug", "run_id"]).size()
    excluded = []
    if not no_lst:
        for sl in sorted(raw["slug"].unique()):
            sel = (raw["slug"] == sl).to_numpy()
            if (ok_a & sel).sum() >= MIN_CELLS and (ok & sel).sum() < MIN_CELLS:
                excluded.append({"slug": sl, "reason": f"no LST over its driven cells ({int((has_lst & sel).sum())} of "
                                                       f"{int(sel.sum())}): no clear Landsat scene"})
    info["excluded"] = excluded
    df = raw[ok].reset_index(drop=True)
    n_run = df.groupby(["slug", "run_id"])["cell"].transform("size").to_numpy()
    small = df[n_run < MIN_CELLS].groupby(["slug", "run_id"]).size()
    info["runs_dropped"] = [{"slug": sl, "run_id": rid, "n_scored": int(n), "n_cells": int(n_all.get((sl, rid), n)),
                             "reason": f"{int(n)} cells with every predictor (< {MIN_CELLS})"}
                            for (sl, rid), n in small.items()]
    df = df[n_run >= MIN_CELLS].reset_index(drop=True)
    slugs = sorted(df["slug"].unique())
    if len(slugs) < MIN_CITIES:
        why = "every predictor" if no_lst else "every predictor (missing LST excluded " + \
            (f"{len(excluded)} of them)" if excluded else "none)")
        info["reason"] = f"{len(slugs)} cities with ≥ {MIN_CELLS} cells with {why}"
        return pd.DataFrame(), info
    df = design(df, s)                                 # anomalies centred on the scored cells
    df, info["target_recentred"] = recentre_target(df, n_all)
    info["no_lst"] = no_lst
    return df, info


def score_runs(df: pd.DataFrame, preds: dict) -> list[dict]:
    """Per (city, run) of the scored table: MSE_N, and per model with a prediction on every cell its MSE, skill
    SS = 1 − MSE/MSE_N and spatial Pearson r."""
    y_all = df["dTa"].to_numpy(float)
    out = []
    for (sl, rid), idx in sorted(df.groupby(["slug", "run_id"], sort=True).indices.items()):
        y = y_all[idx]
        mse_n = float(np.mean(y ** 2))
        rec = {"slug": sl, "run_id": rid, "group": str(df["group"].iat[idx[0]]), "n": int(idx.size), "mse_N": mse_n,
               "N": {"mse": mse_n, "ss": 0.0, "r": None}}
        if mse_n <= 0:
            rec["note"] = "constant target: no skill defined"
            out.append(rec)
            continue
        for m, p_all in preds.items():
            p = p_all[idx]
            if not np.isfinite(p).all():
                continue
            mse = float(np.mean((p - y) ** 2))
            rec[m] = {"mse": mse, "ss": 1.0 - mse / mse_n, "r": _r(y, p)}
        out.append(rec)
    return out


def _nanmean(v) -> float | None:
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def score_units(per_run: list[dict], groups: dict[str, list[str]]) -> list[dict]:
    """Per held-out unit: each model's skill, MSE and r averaged over the unit's runs (a model enters a unit only
    with a prediction on every run of it)."""
    out = []
    for g, members in groups.items():
        rr = [r for r in per_run if r["group"] == g and "note" not in r]
        rec = {"unit": g, "members": list(members), "n_runs": len(rr), "n": int(sum(r["n"] for r in rr))}
        for m in MODELS:
            if rr and all(m in r for r in rr):
                rec[m] = {"ss": float(np.mean([r[m]["ss"] for r in rr])), "mse": float(np.mean([r[m]["mse"] for r in rr])),
                          "r": _nanmean([r[m]["r"] for r in rr])}
        out.append(rec)
    return out


def _unit_arrays(per_unit: list[dict], x: str, base: str, field: str = "ss"):
    rows = [u for u in per_unit if x in u and base in u]
    return ([u["unit"] for u in rows], np.array([u[x][field] for u in rows], float),
            np.array([u[base][field] for u in rows], float))


def evaluate_window(cities: list[CityData], run: str, s: Settings, cache: Path | None = None,
                    flagged: list[dict] | None = None) -> tuple[dict, pd.DataFrame]:
    """Leave-one-campaign-group-out N, A, B, C (and C0) on one run window; (results, per-cell predictions).

    ``cities`` as loaded (:func:`prepare` runs here) or, with ``flagged`` given, as :func:`prepare` returned them.
    The verdicts are set with Holm's correction over this window alone (the other registered windows count as
    untested, p = 1); :func:`evaluate` re-applies it over every window."""
    t0 = time.perf_counter()
    if flagged is None:
        cities, flagged = prepare(cities, s)
    res: dict = {"run": run, "primary": run == PRIMARY_WINDOW,
                 "qa_flagged": [f for f in flagged if f["run"] == run]}
    df, info = window_table(cities, run, s)
    res.update({k: v for k, v in info.items() if k not in ("reason", "no_lst")})
    if df.empty:
        res.update(status="not run", reason=info["reason"])
        return res, pd.DataFrame()
    no_lst = info["no_lst"]
    by_city = {c.slug: c for c in cities}
    slugs = sorted(df["slug"].unique())
    groups = campaign_groups([by_city[c] for c in slugs], same_node_km=s.same_node_km)
    df["group"] = df["slug"].map(groups)
    units = sorted(set(groups.values()))
    res["groups"] = {g: [c for c in slugs if groups[c] == g] for g in units}
    if len(units) < MIN_CITIES:
        res.update(status="not run", reason=f"{len(units)} independent campaign groups (need {MIN_CITIES}): "
                                            f"{', '.join(units)}")
        return res, pd.DataFrame()
    a_cols, b_cols = model_columns(s)
    key = (df["slug"] + "|" + df["run_id"]).to_numpy()                 # a run of a city: the centring unit
    preds = {m: np.full(len(df), np.nan) for m in ("A", "B", "C", "C0")}
    chosen = {}
    fits = {}
    models = (("A", a_cols),) if no_lst else (("A", a_cols), ("B", b_cols))
    if s.physics:
        for v in s.physics_variants:
            if v == "C" and no_lst:
                continue
            fits[v] = physics_fits([by_city[c] for c in slugs], run, v == "C", s,
                                   None if cache is None else cache / "physics")
    transfers = {}
    for g in units:                                                    # hold out a whole campaign group
        members = res["groups"][g]
        te_g = (df["group"] == g).to_numpy()
        train = _subsample(df[~te_g], s.max_rows_per_city, s.seed)
        w_train = _weights(train["slug"].to_numpy(), train["dTa"].to_numpy(float), s.weights)
        for name, cols in models:
            (kind, params), table = inner_select(train, cols, s)
            p = _fit_predict(kind, params, train[cols].to_numpy(float), train["dTa"].to_numpy(float), w_train,
                             df.loc[te_g, cols].to_numpy(float), s.seed)
            preds[name][te_g] = centre(p, key[te_g])
            chosen.setdefault(name, []).append({"held_out": g, "model": kind, "params": params, "inner": table})
        for v, fv in fits.items():
            others = [f for k, f in fv.items() if k not in members and "error" not in f]
            tp, why = None, f"{len(others)} training cities with a physics fit"
            if len(others) >= MIN_CITIES - 1:
                try:
                    tp = transfer_params(others, v == "C", s.physics_transfer)
                except Exception as exc:              # noqa: BLE001 - e.g. no fit with a finite parameter vector
                    why = str(exc).splitlines()[0][:200] if str(exc) else repr(exc)
            for h in members:
                if tp is None:
                    transfers.setdefault(h, {})[v] = {"error": why}
                    continue
                te = (df["slug"] == h).to_numpy()
                try:
                    pc = predict_physics(by_city[h], run, v == "C", tp, s)
                except Exception as exc:              # noqa: BLE001
                    transfers.setdefault(h, {})[v] = {"error": str(exc).splitlines()[0][:200] if str(exc) else repr(exc)}
                    continue
                # the physics predicts every driven cell of every run; score the cells A and B are scored on
                idx = pd.MultiIndex.from_arrays([df.loc[te, "run_id"], df.loc[te, "cell"]])
                preds[v][te] = centre(pc.reindex(idx).to_numpy(), key[te])
                transfers.setdefault(h, {})[v] = {"rule": tp["rule"], "a_phys": tp["a_phys"], "L_m": tp["natural"]["L"],
                                                  **({"medoid": tp["medoid"]} if "medoid" in tp else {}),
                                                  **({"g_lst": tp["g_lst"], "h_lst": tp["h_lst"]} if v == "C" else {})}
    preds = {m: v for m, v in preds.items() if np.isfinite(v).any()}
    per_run = score_runs(df, {"N": np.zeros(len(df)), **preds})
    per_unit = score_units(per_run, res["groups"])
    res["per_run"] = per_run
    res["per_unit"] = per_unit
    per_city = []
    for h in slugs:
        te = (df["slug"] == h).to_numpy()
        rr = [r for r in per_run if r["slug"] == h]
        rec = {"slug": h, "city": str(df.loc[te, "city"].iloc[0]), "n": int(te.sum()), "group": groups[h],
               "n_runs": len(rr)}
        for m, p in preds.items():
            if np.isfinite(p[te]).all():
                rec[m] = city_metrics(df.loc[te, "dTa"], p[te])
                rec[m]["ss"] = _nanmean([r[m]["ss"] for r in rr if m in r])
                if m in ("A", "B"):
                    rec[m]["model"] = next(c["model"] for c in chosen[m] if c["held_out"] == groups[h])
        for v, tr in (transfers.get(h) or {}).items():
            if "error" in tr:
                rec[v] = {"error": tr["error"]}
            elif v in rec:
                rec[v]["transfer"] = tr
        if "lst_offset_days" in df:
            off = df.loc[te, "lst_offset_days"]
            rec["lst_offset_days"] = None if off.isna().all() else float(off.abs().median())
        per_city.append(rec)
    res["per_city"] = per_city
    res["chosen"] = {k: [{kk: vv for kk, vv in c.items() if kk != "inner"} for c in v] for k, v in chosen.items()}
    res["inner_cv"] = {k: [c["inner"] for c in v] for k, v in chosen.items()}
    res["summary"] = {}
    for m in MODELS:
        us = [u[m] for u in per_unit if m in u]
        cs = [r[m] for r in per_city if "mse" in (r.get(m) or {})]
        if not us:
            continue
        rec = {"n_units": len(us), "mean_ss": float(np.mean([u["ss"] for u in us])),
               "median_ss": float(np.median([u["ss"] for u in us]))}
        if cs:
            rs = np.array([x["r"] for x in cs], float)
            rec.update(n_cities=len(cs), mean_rmse=float(np.mean([x["rmse"] for x in cs])),
                       median_r2=float(np.median([x["r2"] for x in cs])),
                       median_r=float(np.nanmedian(rs)) if np.isfinite(rs).any() else None)
        res["summary"][m] = rec
    ra = [(u["unit"], u["A"]["r"]) for u in per_unit if "A" in u and u["A"]["r"] is not None]
    res["units_r_A_negative"] = {"n": sum(r < 0 for _, r in ra), "of": len(ra), "units": [g for g, r in ra if r < 0],
                                 "note": "held-out units whose mean over runs of r(A, dTa) is below 0"}
    res["paired"], res["absolute"] = {}, {}
    for k, (x, base, label) in COMPARISONS.items():
        if k == "B-A" and no_lst:
            res["paired"][k] = {"comparison": label, "key": k, "claim": CLAIMS[k], "x": x, "base": base,
                                "verdict": "not testable", "n_units": 0, "reason": "no LST: " + info["lst"].split(": ", 1)[1]}
            continue
        if x not in res["summary"] or base not in res["summary"]:
            continue
        seed = s.seed + list(COMPARISONS).index(k)
        names, sx, sb = _unit_arrays(per_unit, x, base)
        res["paired"][k] = paired_skill(sx, sb, names, s.boot, seed, k)
        _, mx, mb = _unit_arrays(per_unit, x, base, "mse")
        res["absolute"][k] = paired(mx, mb, s.boot, seed, label, CLAIMS[k])
    finalise({run: res})
    offs = [(r["lst_offset_days"], r["B"]["ss"] - r["A"]["ss"]) for r in per_city
            if r.get("lst_offset_days") is not None and "B" in r and "A" in r]
    if len(offs) >= 4 and len({o for o, _ in offs}) > 1:
        from scipy.stats import spearmanr

        rho = spearmanr([o for o, _ in offs], [d for _, d in offs]).correlation
        res["gain_vs_lst_offset"] = {"spearman": float(rho) if np.isfinite(rho) else None, "n": len(offs),
                                     "note": "B − A skill difference against the scene's |offset| in days (descriptive)"}
    if fits:
        res["physics_fits"] = {v: list(f.values()) for v, f in fits.items()}
        res["physics_L_bounds"] = l_bounds_report(fits, s)
    if s.within_city:
        res["within_city"] = within_city(df, s)
    res["status"] = "done"
    res["seconds"] = round(time.perf_counter() - t0, 1)
    pred = df[["city", "slug", "group", "run", "run_id", "cell", "x", "y", "dTa"]].copy()
    for m, v in preds.items():
        pred[f"pred_{m}"] = v
    return res, pred


def _settings_json(s: Settings) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(s).items()}


def evaluate(cities: list[CityData], s: Settings | None = None, cache: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Every window of ``s.windows`` the cities have, the registered Holm correction across them (:func:`finalise`)
    and the primary verdict (af)."""
    s = s or Settings()
    cities, flagged = prepare(cities, s)
    out = {"schema": 2, "target": "dTa: traverse air temperature anomaly from the run's city median (°C)",
           "settings": _settings_json(s), "claim_rule": CLAIM_RULE,
           "transfer_rule": TRANSFER_RULE if s.physics_transfer == "median" else TRANSFER_RULE + MEDOID_NOTE,
           "n_cities": len(cities), "cities": [c.slug for c in cities], "qa_flagged": flagged, "windows": {}}
    preds = []
    present = {r for c in cities for r in c.table["run"].unique()}
    for run in [w for w in s.windows if w in present]:
        log.info("lstair evaluate: window %s", run)
        res, pred = evaluate_window(cities, run, s, cache, flagged=flagged)
        out["windows"][run] = res
        if len(pred):
            preds.append(pred)
    finalise(out["windows"])
    out["primary"] = primary_summary(out)
    return out, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def primary_summary(res: dict) -> dict:
    """The registered verdicts: the af window's B − A and C − A."""
    w = res["windows"].get(PRIMARY_WINDOW) or {}
    out = {"window": PRIMARY_WINDOW, "status": w.get("status", "not run")}
    if w.get("status") != "done":
        out["reason"] = w.get("reason", f"no {PRIMARY_WINDOW} runs")
        return out
    for k in HOLM_CLAIMS:
        p = (w.get("paired") or {}).get(k)
        if p:
            out[k] = {q: p.get(q) for q in ("claim", "verdict", "reason", "n_units", "mean_dss", "ci95", "p_sign_flip",
                                            "p_holm", "mean_ss_x", "mean_ss_base")}
            out[k]["any_single_unit_flips"] = (p.get("influence") or {}).get("any_flip")
    return out


TRANSFER_RULE = ("Model C is fitted once per city and window on that city's own driven cells of one run (the "
                 "window's run with the most cells, with that run's forcing and background).  For a held-out city "
                 "every nonlinear physics parameter and the lapse coefficient are the medians of the training "
                 "cities' fits; the gain is transferred in physical units (a/forcing) and the LST term as the "
                 "median of the per-city products a_phys·h_lst (K per K), converted with the held-out city's own "
                 "run forcing; the intercept and the "
                 "linear trend are 0; each of its runs is predicted with that run's own forcing and centred to "
                 "median zero over the run's cells.  No held-out "
                 "label is used.")
MEDOID_NOTE = ("  Sensitivity in use (physics_transfer = medoid): instead of per-parameter medians, the training "
               "city whose whole fitted parameter vector (log of positive parameters, standardised) has the least "
               "summed distance to the others is used whole.")
CLAIM_RULE = ("Per held-out unit and run, skill SS = 1 − MSE/MSE_N (N: no anomaly), averaged over the unit's runs; "
              "a claim (af primary) needs the one-sided sign-flip p of mean ΔSS ≤ 0.025 after Holm's correction over "
              "am, af, pm × (B − A, C − A), the 95% t-interval of ΔSS above 0, mean ΔSS ≥ 0.05 and mean SS of B "
              "(resp. C) above 0; at least 8 units.")


# --------------------------------------------------------------------------- #
# Sensitivities                                                                #
# --------------------------------------------------------------------------- #
#: The registered sensitivities, each a full rerun of :func:`evaluate` (name → description, Settings overrides).
#: The absolute ΔMSE and the leave-one-unit-out influence come from the primary run itself (no rerun).
VARIANTS = {
    "weights_n": ("training weights ∝ 1/n (not 1/(n·var))", {"weights": "n"}),
    "virginia_one_group": ("one node and date linked at any distance (the eight Virginia cities of 15 July 2021 "
                           "as one group)", {"same_node_km": None}),
    "include_flagged": ("walking campaigns, single-car runs and the other flagged runs included",
                        {"include_flagged": True}),
    "lst_local": ("B with the local LST residual only (LST minus its 3 km focal mean)", {"lst_mode": "local"}),
    "regional": ("A and B with regional controls (3 km focal elevation and impervious)", {"regional": True}),
    "nlcd": ("NLCD tree canopy and percent impervious in place of WorldCover", {"land_cover": "nlcd"}),
    "physics_30m": ("physics at 30 m cells", {"physics_cell_m": 30.0}),
    "physics_medoid": ("physics transferred as the medoid city's whole parameter vector",
                       {"physics_transfer": "medoid"}),
}
#: Settings a physics fit (and its cache key) depends on: a variant that changes one gets its own fit cache.
PHYSICS_FIT_FIELDS = ("physics_cell_m", "physics_iter", "physics_L_max_m", "land_cover", "include_flagged")


def _variant_row(name: str, desc: str, res: dict, windows) -> dict:
    row = {"variant": name, "description": desc, "status": "done", "windows": {}}
    for w in windows:
        wr = res["windows"].get(w)
        if not wr:
            continue
        if wr.get("status") != "done":
            row["windows"][w] = {"status": "not run", "reason": wr.get("reason")}
            continue
        row["windows"][w] = {k: {q: (wr["paired"].get(k) or {}).get(q) for q in
                                 ("n_units", "mean_dss", "ci95", "p_sign_flip", "p_holm", "verdict")}
                             for k in HOLM_CLAIMS if k in wr.get("paired", {})}
    return row


def run_sensitivity(data_dir: str | Path, out_dir: str | Path, s: Settings | None = None,
                    variants: list[str] | None = None, cache: str | Path | None = None,
                    cities: list[CityData] | None = None) -> dict:
    """``sparc core lstair sensitivity``: the primary analysis and each registered sensitivity (:data:`VARIANTS`),
    one after another on the dataset loaded once; sensitivity.json and sensitivity.md (one row per variant:
    per window the mean ΔSS of B − A and C − A, the 95% interval, the Holm-adjusted p and the verdict), plus the
    primary run's absolute ΔMSE and leave-one-unit-out rows, and each run's results under OUT/<variant>/.

    Physics fits are cached in ``cache`` (default OUT/primary; pass a ``lstair evaluate --out`` folder to reuse
    its fits) and reused wherever their key allows; a variant that changes the fits (:data:`PHYSICS_FIT_FIELDS`)
    caches its own under OUT/<variant>.  A variant that cannot run (e.g. ``nlcd`` without the NLCD columns) is
    recorded with the reason."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = s or Settings()
    names = list(VARIANTS) if variants is None else list(variants)
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown sensitivity variant(s) {unknown}; known: {', '.join(VARIANTS)}")
    if cities is None:
        cities = load_dataset(data_dir)
    if len(cities) < MIN_CITIES:
        raise ValueError(f"{len(cities)} finished cities under {data_dir}: leave-one-city-out needs {MIN_CITIES}")
    shared = Path(cache) if cache is not None else out / "primary"
    t0 = time.perf_counter()

    def run_one(name: str, vs: Settings, where: Path) -> dict:
        res, _ = evaluate(cities, vs, cache=where)
        res["data"] = str(data_dir)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(to_json(res), encoding="utf-8")
        (d / "results.md").write_text(results_markdown(res), encoding="utf-8")
        return res

    log.info("lstair sensitivity: primary analysis")
    prim = run_one("primary", base, shared)
    windows = [w for w in base.windows if w in prim["windows"]]
    rows = [_variant_row("primary", "the registered analysis", prim, windows)]
    absolute = {"variant": "absolute_dmse", "description": "absolute ΔMSE (°C²), secondary statistic of the primary "
                "run (no rerun; unadjusted sign-flip p)", "status": "done", "windows": {}}
    loo = {"variant": "leave_one_unit_out", "description": "the primary run without each held-out unit in turn (no "
           "rerun)", "status": "done", "windows": {}}
    for w in windows:
        wr = prim["windows"][w]
        if wr.get("status") != "done":
            continue
        absolute["windows"][w] = {k: {q: (wr.get("absolute", {}).get(k) or {}).get(q) for q in
                                      ("n_cities", "mean_diff_mse", "ci95", "p_sign_flip", "verdict")}
                                  for k in HOLM_CLAIMS if k in wr.get("absolute", {})}
        loo["windows"][w] = {k: {q: (wr["paired"][k].get("influence") or {}).get(q) for q in
                                 ("mean_dss_range", "ci95_low_range", "p_range", "flips", "any_flip")}
                             for k in HOLM_CLAIMS if k in wr.get("paired", {})}
    rows += [absolute, loo]
    for name in names:
        desc, over = VARIANTS[name]
        vs = dataclasses.replace(base, **over)
        own = any(getattr(vs, f) != getattr(base, f) for f in PHYSICS_FIT_FIELDS)
        log.info("lstair sensitivity: %s", name)
        try:
            res = run_one(name, vs, out / name if own else shared)
        except ValueError as exc:
            rows.append({"variant": name, "description": desc, "status": "not run", "reason": str(exc)})
            continue
        rows.append(_variant_row(name, desc, res, windows))
        rows[-1]["settings"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in over.items()}
    sens = {"schema": 1, "data": str(data_dir), "settings": _settings_json(base), "windows": windows,
            "primary_window": PRIMARY_WINDOW, "claim_rule": CLAIM_RULE, "rows": rows,
            "note": "sensitivities are reported, never used for the claim (docs/results/lst_air_pilot.md)",
            "seconds": round(time.perf_counter() - t0, 1)}
    (out / "sensitivity.json").write_text(to_json(sens), encoding="utf-8")
    (out / "sensitivity.md").write_text(sensitivity_markdown(sens), encoding="utf-8")
    return sens


def sensitivity_markdown(sens: dict) -> str:
    windows = sens["windows"]
    L = ["# LST → air temperature: sensitivities", "",
         f"Each row is a full rerun of the evaluation with one change (absolute ΔMSE and leave-one-unit-out come from "
         f"the primary run).  Primary window: {sens['primary_window']}.  {sens['note'].capitalize()}.", "",
         f"Claim rule: {sens['claim_rule']}", ""]

    def cell(v: dict | None, absolute: bool = False) -> str:
        if not v:
            return "—"
        if absolute:
            ci = v.get("ci95")
            return (f"ΔMSE {_f(v.get('mean_diff_mse'), 3, True)} ({_ci(ci, 3)}), p {_f(v.get('p_sign_flip'), 3)}, "
                    f"{v.get('verdict')}")
        if "mean_dss_range" in v:
            return (f"ΔSS {_range(v.get('mean_dss_range'))}; "
                    + ("flips: " + ", ".join(v["flips"]) if v.get("flips") else "no single unit flips the verdict"))
        return (f"{_f(v.get('mean_dss'), 3, True)} ({_ci(v.get('ci95'), 3)}), Holm p {_f(v.get('p_holm'), 3)}, "
                f"{v.get('verdict')}")

    for k in HOLM_CLAIMS:
        x, base, label = COMPARISONS[k]
        L += [f"## {label}: mean ΔSS (95% CI), Holm-adjusted p, verdict", "",
              "| variant | " + " | ".join(windows) + " |", "|---|" + "---|" * len(windows)]
        for r in sens["rows"]:
            if r.get("status") != "done":
                L.append(f"| {r['variant']} ({r['description']}) | not run: {r.get('reason')} |"
                         + " |" * max(len(windows) - 1, 0))
                continue
            cells = []
            for w in windows:
                wv = (r.get("windows") or {}).get(w)
                if wv is not None and wv.get("status") == "not run":
                    cells.append(f"not run: {wv.get('reason')}")
                else:
                    cells.append(cell((wv or {}).get(k), absolute=r["variant"] == "absolute_dmse"))
            L.append(f"| {r['variant']} ({r['description']}) | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #
def _f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(ci, nd=3):
    return "—" if not ci else f"{ci[0]:+.{nd}f} to {ci[1]:+.{nd}f}"


def _p(x):
    return "—" if x is None else f"{x:.3g}"


def _range(r, nd=3, sign=True) -> str:
    r = r or [None, None]
    return f"{_f(r[0], nd, sign)} to {_f(r[1], nd, sign)}"


def _verdict_lines(w: dict) -> list[str]:
    L = []
    for k in HOLM_CLAIMS:
        p = (w.get("paired") or {}).get(k)
        if not p:
            continue
        line = f"* **{p['comparison']}: {p['verdict']}.** {p.get('reason', '')}."
        inf = p.get("influence")
        if inf:
            line += (" Leave-one-unit-out: " + ("dropping " + ", ".join(inf["flips"]) + " flips it" if inf["flips"]
                                               else "no single unit flips it")
                     + f" (mean ΔSS {_range(inf['mean_dss_range'])}).")
        L.append(line)
    return L


def results_markdown(res: dict) -> str:
    L = ["# LST → air temperature: leave-one-campaign-out", "",
         f"Target: {res['target']}.  {res['n_cities']} cities.  Models: N no anomaly (the floor); A land cover + focal "
         "features; B A + LST; C physics with LST in the heat source; C0 physics without LST.", "",
         f"Claim rule: {res.get('claim_rule', CLAIM_RULE)}", "",
         f"Transfer rule (C): {res['transfer_rule']}", ""]
    prim = res.get("primary") or {}
    L += [f"## Primary verdict ({prim.get('window', PRIMARY_WINDOW)}, as registered)", ""]
    pw = res["windows"].get(prim.get("window", PRIMARY_WINDOW)) or {}
    if pw.get("status") == "done":
        L += _verdict_lines(pw) + [""]
    else:
        L += [f"Not run: {prim.get('reason') or pw.get('reason')}", ""]
    order = sorted(res["windows"], key=lambda r: (r != PRIMARY_WINDOW, list(res["windows"]).index(r)))
    for run in order:
        w = res["windows"][run]
        L += [f"## Run {run}" + (" (primary)" if run == PRIMARY_WINDOW else " (secondary)"), ""]
        if w.get("status") != "done":
            L += [f"Not run: {w.get('reason')}", ""]
            continue
        if run != PRIMARY_WINDOW:
            L += _verdict_lines(w) + [""]
        multi = {g: m for g, m in (w.get("groups") or {}).items() if len(m) > 1}
        L += [f"Held-out units: {len(w.get('groups') or {})} campaign groups"
              + (" (held out together: " + "; ".join(", ".join(m) for m in multi.values()) + ")" if multi else "")
              + ".", ""]
        L += ["| comparison | mean ΔSS | 95% CI (t, n − 1 df) | sign-flip p | Holm p | mean SS (x / base) | "
              "units better / worse | verdict |", "|---|---|---|---|---|---|---|---|"]
        for k, p in w["paired"].items():
            if "ci95" not in p:
                L.append(f"| {p['comparison']} | — | — | — | — | — | — | {p['verdict']} |")
                continue
            L.append(f"| {p['comparison']} | {p['mean_dss']:+.3f} | {_ci(p['ci95'])} | {_p(p.get('p_sign_flip'))} | "
                     f"{_p(p.get('p_holm')) if p.get('p_holm') is not None else 'not in family'} | "
                     f"{p['mean_ss_x']:+.3f} / {p['mean_ss_base']:+.3f} | {p['n_better']} / {p['n_worse']} | "
                     f"{p['verdict']} |")
        rn = w.get("units_r_A_negative") or {}
        L += ["", f"Units with r(A, dTa) < 0: {rn.get('n', 0)} of {rn.get('of', 0)}"
              + (f" ({', '.join(rn['units'])})" if rn.get("units") else "") + ".", ""]
        L += ["| held-out unit | cities | runs | cells | SS_N | SS_A | SS_B | SS_C | SS_C0 | r_A | r_B | r_C | r_C0 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for u in w.get("per_unit", []):
            def g(m, k):
                return _f((u.get(m) or {}).get(k), 3, k == "ss")
            L.append(f"| {u['unit']} | {len(u['members'])} | {u['n_runs']} | {u['n']:,} | 0 | {g('A', 'ss')} | "
                     f"{g('B', 'ss')} | {g('C', 'ss')} | {g('C0', 'ss')} | {g('A', 'r')} | {g('B', 'r')} | "
                     f"{g('C', 'r')} | {g('C0', 'r')} |")
        L += ["", "Leave-one-unit-out influence:", ""]
        for k, p in w["paired"].items():
            inf = p.get("influence")
            if not inf:
                continue
            pr, ps = inf["p_range"] or [None, None], inf.get("p_sign_flip_range") or [None, None]
            flips = "; ".join(f"without {u}: {v}" for u, v in (inf.get("flipped_to") or {}).items())
            L.append(f"* {p['comparison']}: mean ΔSS {_range(inf['mean_dss_range'])}, CI lower bound "
                     f"{_range(inf['ci95_low_range'])}, sign-flip p {_p(ps[0])} to {_p(ps[1])} "
                     f"({'Holm-adjusted ' + _p(pr[0]) + ' to ' + _p(pr[1]) if p.get('p_holm') is not None else 'unadjusted'}); "
                     + (f"the verdict flips ({flips})" if inf["flips"] else "no single unit flips the verdict") + ".")
        lb = w.get("physics_L_bounds") or {}
        if lb:
            L += ["", "Physics fits on an L bound (within 5%; the upper bound is L_max, or a quarter of a small "
                      "grid's extent): " + "; ".join(
                          f"{v} {b['n_lower']} at the cell size, {b['n_upper']} at the upper bound of {b['n_fits']}"
                          for v, b in lb.items()) + "."]
        wc = w.get("within_city") or {}
        if wc.get("cities"):
            dm = wc.get("mean_diff_mse_B_minus_A")
            L += ["", f"Secondary, {wc['label']}: median R² A {_f(wc['median_r2'].get('A'))}, B "
                      f"{_f(wc['median_r2'].get('B'))}; mean ΔMSE (B − A) {_f(dm, 4)} °C²."]
        ex = []
        if w.get("qa_flagged"):
            ex.append("data checks: " + "; ".join(f"{e['slug']} {e['run_id']} ({e['reason']}"
                                                  + (", included" if e.get("included") else "") + ")"
                                                  for e in w["qa_flagged"]))
        if w.get("excluded"):
            ex.append("; ".join(f"{e['slug']} ({e['reason']})" for e in w["excluded"]))
        if w.get("runs_dropped"):
            ex.append("runs not scored: " + "; ".join(f"{e['slug']} {e['run_id']} ({e['reason']})"
                                                      for e in w["runs_dropped"]))
        if ex:
            L += ["", "Excluded: " + ". ".join(ex) + "."]
        if w.get("lst", "present") != "present":
            L += ["", f"LST: {w['lst']}."]
        if w.get("gain_vs_lst_offset"):
            g = w["gain_vs_lst_offset"]
            L += ["", f"B − A skill against the LST scene's offset from the campaign: Spearman {_f(g['spearman'])} over "
                      f"{g['n']} cities (descriptive)."]
        L.append("")
    L += ["## Secondary: absolute ΔMSE (°C², never used for the claim)", "",
          "| run | comparison | mean ΔMSE | 95% CI | sign-flip p (unadjusted) | relative | units better / worse | "
          "verdict of the earlier rule |", "|---|---|---|---|---|---|---|---|"]
    for run in order:
        for k, p in (res["windows"][run].get("absolute") or {}).items():
            if "ci95" not in p:
                continue
            rel = "—" if p.get("rel_change") is None else f"{100 * p['rel_change']:+.0f}%"
            L.append(f"| {run} | {p['comparison']} | {p['mean_diff_mse']:+.4f} | {_ci(p['ci95'], 4)} | "
                     f"{_p(p['p_sign_flip'])} | {rel} | {p['n_better']} / {p['n_worse']} | {p['verdict']} |")
    if res.get("qa_flagged"):
        inc = res["settings"].get("include_flagged")
        L += ["", "## Runs flagged by the data checks" + (" (included: sensitivity)" if inc else " (excluded)"), ""]
        L += [f"* {e['slug']} {e['run_id']} ({e['run']}, {e['n_cells']:,} cells): {e['reason']}" for e in res["qa_flagged"]]
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


__all__ = ["Settings", "design", "model_columns", "raw_predictors", "prepare", "window_table", "A_COLS", "B_COLS",
           "LST_COLS", "city_metrics", "skill", "paired", "paired_skill", "skill_verdict", "holm", "finalise",
           "influence", "sign_flip", "centre", "inner_select", "campaign_groups", "recentre_target", "WINDOW_REGIME",
           "regime", "fit_city_physics", "medoid_fit", "transfer_params", "predict_physics", "physics_fits",
           "l_bounds_report", "score_runs", "score_units", "evaluate_window", "evaluate", "primary_summary",
           "results_markdown", "run_evaluation", "run_sensitivity", "sensitivity_markdown", "VARIANTS",
           "TRANSFER_RULE", "CLAIM_RULE", "CLAIMS", "MIN_VERDICT_CITIES", "MIN_DSS", "SAME_NODE_KM", "HOLM_WINDOWS",
           "HOLM_CLAIMS", "PRIMARY_WINDOW"]
