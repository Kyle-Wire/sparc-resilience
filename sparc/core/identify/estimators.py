"""Canopy designs.  Every effect is the change in air temperature (°F; negative = cooler) at a point
when canopy rises by +10 pp, with a cluster-robust standard error over 1 km spatial blocks.  Designs
differ in **where** the canopy is raised (the estimand) and in **what variation** identifies it.

Estimands (``estimand``):

* ``within_100`` / ``within_300`` / ``within_1000``: canopy raised within 100 m / 300 m / 1 km of the
  point (a disk edit).  How far the cooling of a tree reaches decides how much of the city-wide
  effect lives inside each disk.
* ``total``: canopy raised everywhere (the city-wide uniform edit the scenarios report).

Measurement level (traverse points; ``df`` has ``cell``, ``t_s``, ``vehicle``, ``seg``, ``temp``):

* ``street_100``, ``street_300``, ``reach_1000`` - **spatial first differences along each pass**:
  samples of one pass up to 300 m apart, seconds to a minute apart.  The warming drift, the vehicle's
  sensor offset and every confounder that is smooth over 300 m cancel; what is left is how temperature
  changes along a street as the canopy around it changes.  Canopy enters through a flexible response
  in the cell and in rings out to 1 km (:mod:`~sparc.core.identify.layers`); impervious surface,
  albedo, elevation and distance to water are controls.  One fit, three readings: the effect of
  +10 pp within 100 m, 300 m and 1 km, at the street points the vehicles drove.  The 1 km reading
  reaches beyond what a 300 m difference holds fixed, so kilometre-scale confounders can leak into it.
* ``updown`` - **the wind signature**: the same differences with canopy in the 1 km sector upwind and
  the 1 km sector downwind.  Air carries the cooling of trees downwind, so a physical effect makes
  upwind canopy matter more; land-use confounding and an isotropic interpolator have no direction.
  Returns the contrast (upwind − downwind) for +10 pp.
* ``levels`` - the naive regression of temperature levels on canopy and impervious footprints with a
  smooth spatial basis, time and vehicle (the "regress the map" approach, on measurements).

Map level (one value per cell, the forest-made product):

* ``map_footprint`` - the levels regression on every cell of the map.
* ``map_street_300`` - the street design applied to the map (first differences between neighbouring
  cells): the same estimator as ``street_300``, so any gap between the two is the product's doing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sparc.core.identify import DOSE

Z95 = 1.959963984540054
LAGS = (1, 2, 3, 4, 5, 6, 8, 10)                     # sample lags along a pass (≈ 30–300 m)
SIGNATURE_LAGS = LAGS + (13, 16, 20)                  # ≈ 600 m: the 1 km sectors need longer contrasts
FOOTPRINTS_CANOPY = ("canopy_0", "canopy_100", "canopy_300", "canopy_1000")
FOOTPRINTS_IMP = ("impervious_0", "impervious_100", "impervious_300", "impervious_1000")

ESTIMATORS = ("street_100", "street_300", "reach_1000", "updown", "levels", "map_footprint", "map_street_300")
LABELS = {
    "street_100": "Street differences: canopy within 100 m",
    "street_300": "Street differences: canopy within 300 m",
    "reach_1000": "Street differences: canopy within 1 km",
    "updown": "Wind signature: upwind − downwind canopy (1 km)",
    "levels": "Traverse levels regression (footprints + smooth basis)",
    "map_footprint": "Map: levels regression (footprints + smooth basis)",
    "map_street_300": "Map: neighbour differences, canopy within 300 m",
    "map_street_100": "Map: neighbour differences, canopy within 100 m",
}
ESTIMAND = {"street_100": "within_100", "street_300": "within_300", "reach_1000": "within_1000",
            "updown": "contrast", "levels": "total", "map_footprint": "total", "map_street_300": "within_300",
            "map_street_100": "within_100"}
WHERE = {"street_100": "street", "street_300": "street", "reach_1000": "street", "updown": "street",
         "levels": "street", "map_footprint": "city", "map_street_300": "city", "map_street_100": "city"}
LEVEL = {k: ("map" if k.startswith("map_") else "traverse") for k in ESTIMAND}


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with a CR1 cluster-robust covariance (clusters = ``groups``)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X, groups = y[ok], X[ok], np.asarray(groups)[ok]
    scale = np.maximum(np.abs(X).max(axis=0), 1e-12)        # conditioning only; undone below
    Xs = X / scale
    xtx_inv = np.linalg.pinv(Xs.T @ Xs)
    b = xtx_inv @ (Xs.T @ y)
    u = y - Xs @ b
    _, inv = np.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    S = np.zeros((G, Xs.shape[1]))
    np.add.at(S, inv, Xs * u[:, None])
    n, k = Xs.shape
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = adj * xtx_inv @ (S.T @ S) @ xtx_inv
    return b / scale, V / np.outer(scale, scale)


def _result(name: str, b: np.ndarray, V: np.ndarray, c: np.ndarray, n: int, n_clusters: int, **extra) -> dict:
    est = float(c @ b)
    se = float(math.sqrt(max(float(c @ V @ c), 0.0)))
    return {"estimator": name, "label": LABELS.get(name, name), "estimand": ESTIMAND.get(name), "where": WHERE.get(name),
            "level": LEVEL.get(name), "kind": "contrast" if ESTIMAND.get(name) == "contrast" else "effect",
            "estimate": est, "se": se, "lo": est - Z95 * se, "hi": est + Z95 * se, "n": int(n),
            "n_clusters": int(n_clusters), **extra}


def _vec(cols: list[str], weights: dict[str, float]) -> np.ndarray:
    return np.array([weights.get(c, 0.0) for c in cols])


# --------------------------------------------------------------------------- #
# Pairs along a pass                                                           #
# --------------------------------------------------------------------------- #
def sfd_pairs(df: pd.DataFrame, lags=LAGS) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (i, j) of samples ``lag`` apart within one pass (same segment and vehicle, in time
    order)."""
    order = np.lexsort((df["t_s"].to_numpy(), df["vehicle"].to_numpy(), df["seg"].to_numpy()))
    seg = df["seg"].to_numpy()[order]
    veh = df["vehicle"].to_numpy()[order]
    ii, jj = [], []
    for lag in lags:
        if lag >= order.size:
            continue
        same = (seg[lag:] == seg[:-lag]) & (veh[lag:] == veh[:-lag])
        a = np.flatnonzero(same)
        ii.append(order[a])
        jj.append(order[a + lag])
    if not ii:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def _differences(df: pd.DataFrame, L, feats: list[str], lags) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Δtemperature, [Δfeatures, Δtime], cluster of the pair)."""
    i, j = sfd_pairs(df, lags)
    cells = df["cell"].to_numpy()
    F = L.mat(feats)
    t = df["t_s"].to_numpy(float) / 3600.0
    y = df["temp"].to_numpy(float)
    dX = np.column_stack([F[cells[j]] - F[cells[i]], t[j] - t[i]])
    return y[j] - y[i], dX, L.blocks[cells[i]]


def street_features(L, reach: float | None = None) -> list[str]:
    imp = L.has(["impervious_0"] + [f"imp_r{int(r)}" for r in L.meta["rings"]])
    return L.canopy_terms(reach) + imp + L.geography()


def edit_weights(L, cells: np.ndarray, reach: float) -> dict[str, float]:
    """Contrast weights: the mean change of every canopy term within ``reach`` under a +10 pp edit, over
    ``cells``; dotted with the coefficients this is the mean effect of that disk edit at those cells."""
    return {t: float(np.mean(L.values[f"edit_{t}"][cells])) for t in L.canopy_terms(reach)}


def street_fit(df: pd.DataFrame, L, lags=LAGS) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    feats = street_features(L)
    dy, dX, cl = _differences(df, L, feats, lags)
    b, V = cluster_ols(dy, dX, cl)
    return b, V, feats + ["dt"], int(dy.size), int(np.unique(cl).size)


def street_effects(df: pd.DataFrame, L, lags=LAGS, reaches=(100, 300, 1000), city: bool = True) -> list[dict]:
    """``street_100``, ``street_300``, ``reach_1000`` from one fit, at the street points driven (and, as
    ``city_estimate``, the same model applied to every cell of the city: an extrapolation from streets)."""
    b, V, cols, n, G = street_fit(df, L, lags)
    pts = np.unique(df["cell"].to_numpy())
    allc = np.arange(L.blocks.size)
    out = []
    for name, R in zip(("street_100", "street_300", "reach_1000"), reaches):
        c = _vec(cols, edit_weights(L, pts, R))
        extra = {"lags": list(lags), "n_points": int(pts.size)}
        if city:
            cc = _vec(cols, edit_weights(L, allc, R))
            extra.update(city_estimate=float(cc @ b), city_se=float(math.sqrt(max(float(cc @ V @ cc), 0.0))))
        out.append(_result(name, b, V, c, n, G, **extra))
    return out


def updown(df: pd.DataFrame, L, lags=SIGNATURE_LAGS, reach: float = 1000.0) -> dict | None:
    up, dn = f"canopy_up{int(reach)}", f"canopy_down{int(reach)}"
    if up not in L.values:
        return None
    feats = L.canopy_terms(300) + [up, dn] + street_features(L)[len(L.canopy_terms()):]
    dy, dX, cl = _differences(df, L, feats, lags)
    b, V = cluster_ols(dy, dX, cl)
    return _result("updown", b, V, _vec(feats + ["dt"], {up: DOSE, dn: -DOSE}), dy.size, np.unique(cl).size,
                   reach_m=reach, wind=list(L.wind) if L.wind else None)


def levels(df: pd.DataFrame, L) -> dict:
    cells = df["cell"].to_numpy()
    t = df["t_s"].to_numpy(float) / 3600.0
    veh = df["vehicle"].to_numpy()
    ctrl = [np.ones(len(df)), t - t.mean(), (t - t.mean()) ** 2] + [(veh == v).astype(float) for v in np.unique(veh)[1:]]
    feats = L.has(FOOTPRINTS_CANOPY) + L.has(FOOTPRINTS_IMP) + L.geography()
    X = np.column_stack(ctrl + [L.mat(feats)[cells], L.basis[cells]])
    cols = ["c"] * len(ctrl) + feats + ["basis"] * L.basis.shape[1]
    b, V = cluster_ols(df["temp"].to_numpy(float), X, L.blocks[cells])
    return _result("levels", b, V, _vec(cols, {k: DOSE for k in L.has(FOOTPRINTS_CANOPY)}), len(df),
                   np.unique(L.blocks[cells]).size)


# --------------------------------------------------------------------------- #
# Map level                                                                    #
# --------------------------------------------------------------------------- #
def map_footprint(y: np.ndarray, L) -> dict:
    feats = L.has(FOOTPRINTS_CANOPY) + L.has(FOOTPRINTS_IMP) + L.geography()
    X = np.column_stack([np.ones(y.size), L.mat(feats), L.basis])
    cols = ["const"] + feats + ["basis"] * L.basis.shape[1]
    b, V = cluster_ols(y, X, L.blocks)
    return _result("map_footprint", b, V, _vec(cols, {k: DOSE for k in L.has(FOOTPRINTS_CANOPY)}), y.size,
                   np.unique(L.blocks).size)


def neighbour_pairs(grid, lags=LAGS, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Point indices of cells ``lag`` apart along rows and columns (both valid)."""
    idx = np.full(grid.shape, -1, dtype=np.int64)
    idx[grid.iy, grid.ix] = np.arange(grid.iy.size)
    ii, jj = [], []
    for lag in lags:
        for a, b in ((idx[::stride, :-lag], idx[::stride, lag:]), (idx[:-lag, ::stride], idx[lag:, ::stride])):
            ok = (a >= 0) & (b >= 0)
            ii.append(a[ok])
            jj.append(b[ok])
    return np.concatenate(ii), np.concatenate(jj)


def map_street(y: np.ndarray, L, grid, lags=LAGS, reach: float = 300.0) -> dict:
    """The street design on the map: differences between cells along grid lines (every 4th line, so the
    pairs are about as many as a campaign's), effect within ``reach`` over every cell."""
    i, j = neighbour_pairs(grid, lags, stride=4)
    feats = street_features(L)
    F = L.mat(feats)
    y = np.asarray(y, float)
    b, V = cluster_ols(y[j] - y[i], F[j] - F[i], L.blocks[i])
    c = _vec(feats, edit_weights(L, np.arange(y.size), reach))
    return _result(f"map_street_{int(reach)}", b, V, c, i.size, np.unique(L.blocks[i]).size, lags=list(lags))


def run_all(df: pd.DataFrame | None, product: np.ndarray | None, L, grid, which=ESTIMATORS) -> list[dict]:
    """Every design in ``which`` that has its data (traverse designs need ``df``; map designs ``product``)."""
    which = set(which)
    out: list[dict] = []
    have_df = df is not None and len(df) > 0
    have_map = product is not None and np.isfinite(product).any()

    def guard(fn, *a):
        try:
            r = fn(*a)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return r

    if have_df and which & {"street_100", "street_300", "reach_1000"}:
        out += [r for r in (guard(street_effects, df, L) or []) if r["estimator"] in which]
    if have_df and "updown" in which:
        out.append(guard(updown, df, L))
    if have_df and "levels" in which:
        out.append(guard(levels, df, L))
    if have_map and "map_footprint" in which:
        out.append(guard(map_footprint, product, L))
    if have_map and "map_street_300" in which:
        out.append(guard(map_street, product, L, grid))
    return [r for r in out if r is not None]
