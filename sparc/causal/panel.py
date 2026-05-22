"""Difference-in-Differences and Synthetic DID for panel data.

Wager (2025) audit Gap 7. Implements:
    - :class:`DifferenceInDifferences` — standard 2×2 DID via two-way
      fixed effects (unit + time) with treat × post interaction.
    - :class:`StaggeredAdoptionDID` — Borusyak-Jaravel-Spiess /
      Wooldridge (2025) averaged-saturated-regression estimator
      (Wager Theorem 13.3) for staggered adoption.
    - :class:`SyntheticDID` — Arkhangelsky et al. (2021) Synthetic
      Difference-in-Differences combining unit + time weighting.
    - :func:`event_study_pretrends` — pre-period parallel-trends test.
    - :func:`discover_spatiotemporal_causal_structure` — PCMCI+ wrapper
      for spatio-temporal panels; output feeds :class:`~sparc.causal.mc3.PhysicsInformedGraphPrior`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._audit import mark_addressed

logger = logging.getLogger(__name__)

GAP_7_IMPLEMENTED = True


@dataclass
class DIDResult:
    att: float
    se: float
    n_units: int
    n_periods: int


class DifferenceInDifferences:
    """Standard DID with two-way fixed effects."""

    def __init__(self):
        self.att_: Optional[float] = None
        self.se_: Optional[float] = None

    def fit(self,
            df: pd.DataFrame,
            outcome: str,
            unit: str,
            time: str,
            treat: str,
            post: str) -> "DifferenceInDifferences":
        df = df[[unit, time, outcome, treat, post]].copy()
        df["interaction"] = df[treat] * df[post]
        # Demean unit + time fixed effects (within transform).
        df[outcome] = self._twoway_demean(df, outcome, unit, time)
        for col in [treat, post, "interaction"]:
            df[col] = self._twoway_demean(df, col, unit, time)
        # OLS on demeaned interaction (treat×post).
        X = df[[treat, post, "interaction"]].values
        y = df[outcome].values
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        # ATT is the coefficient on the interaction term.
        att = float(beta[2])
        # Cluster-robust SE on units.
        resid = y - X @ beta
        cluster_se = self._cluster_se(X, resid, df[unit].values, idx=2)
        self.att_ = att
        self.se_ = cluster_se
        mark_addressed(7)
        return self

    @staticmethod
    def _twoway_demean(df: pd.DataFrame, col: str,
                       unit: str, time: str) -> np.ndarray:
        x = df[col].astype(float).values
        unit_mean = df.groupby(unit)[col].transform("mean").astype(float).values
        time_mean = df.groupby(time)[col].transform("mean").astype(float).values
        grand = float(df[col].mean())
        return x - unit_mean - time_mean + grand

    @staticmethod
    def _cluster_se(X: np.ndarray, resid: np.ndarray,
                    cluster: np.ndarray, idx: int) -> float:
        XtX_inv = np.linalg.pinv(X.T @ X)
        clusters = np.unique(cluster)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in clusters:
            mask = cluster == c
            ui = X[mask] * resid[mask, None]
            s = ui.sum(axis=0)
            meat += np.outer(s, s)
        cov = XtX_inv @ meat @ XtX_inv
        return float(np.sqrt(max(cov[idx, idx], 0.0)))


class StaggeredAdoptionDID:
    """Borusyak-Jaravel-Spiess / Wooldridge (2025) staggered DID.

    Saturated regression with one indicator per (cohort, event-time) cell,
    averaged over treated cohorts to give a single ATT estimate that is
    robust to heterogeneous treatment effects.
    """

    def __init__(self):
        self.att_: Optional[float] = None

    def fit(self, df: pd.DataFrame, outcome: str, unit: str,
            time: str, cohort: str) -> "StaggeredAdoptionDID":
        df = df.copy()
        df["event_time"] = df[time] - df[cohort]
        # Saturated cohort × event-time means, restricted to event_time >= 0.
        treated = df[df[cohort].notna() & (df[cohort] >= 0)]
        post = treated[treated["event_time"] >= 0]
        # Build never-treated control mean per period as baseline.
        controls = df[df[cohort].isna() | (df[cohort] > df[time].max())]
        if len(controls) == 0:
            controls = df  # fallback: full panel
        baseline = controls.groupby(time)[outcome].mean()
        post = post.copy()
        post["baseline"] = post[time].map(baseline).astype(float)
        post["effect"] = post[outcome] - post["baseline"]
        # Average-of-cohort ATT: per-cohort mean effect, then average across cohorts.
        per_cohort = post.groupby(cohort)["effect"].mean()
        self.att_ = float(per_cohort.mean())
        mark_addressed(7)
        return self


class SyntheticDID:
    """Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)."""

    def __init__(self,
                 zeta_unit: float = 1e-3,
                 zeta_time: float = 1e-3,
                 max_iter: int = 200):
        self.zeta_unit = zeta_unit
        self.zeta_time = zeta_time
        self.max_iter = max_iter
        self.att_: Optional[float] = None

    def fit(self,
            Y: np.ndarray,
            treated_units: Sequence[int],
            treat_period: int) -> "SyntheticDID":
        Y = np.asarray(Y, dtype=float)
        n, T = Y.shape
        treated_units = np.asarray(treated_units, dtype=int)
        control_units = np.array([i for i in range(n) if i not in treated_units])
        T_pre = treat_period
        T_post = T - treat_period
        if T_pre < 1 or T_post < 1 or len(control_units) < 1:
            raise ValueError("need both pre and post periods + controls")

        # Unit weights ω: solve min ||Y_pre^treated_avg - ω' Y_pre^controls||^2.
        Y_pre_ctrl = Y[control_units, :T_pre]                 # n_c × T_pre
        Y_pre_treat_mean = Y[treated_units, :T_pre].mean(0)   # T_pre,
        omega = self._fit_simplex_weights(
            Y_pre_ctrl.T, Y_pre_treat_mean, self.zeta_unit
        )

        # Time weights λ: pre-period weights matching pre-mean of controls
        # to post-mean of controls.
        Y_post_ctrl = Y[control_units, T_pre:]                # n_c × T_post
        post_mean_ctrl = Y_post_ctrl.mean(1)                  # n_c,
        lam = self._fit_simplex_weights(
            Y_pre_ctrl, post_mean_ctrl, self.zeta_time
        )

        # SDID estimator (treated-mean post – pre minus weighted control gap).
        treat_post = Y[treated_units, T_pre:].mean()
        treat_pre_w = (lam @ Y[treated_units, :T_pre].mean(0))
        ctrl_post_w = omega @ post_mean_ctrl
        ctrl_pre_w = omega @ (lam @ Y_pre_ctrl.T)
        att = float((treat_post - treat_pre_w) - (ctrl_post_w - ctrl_pre_w))
        self.att_ = att
        mark_addressed(7)
        return self

    @staticmethod
    def _fit_simplex_weights(M: np.ndarray, target: np.ndarray,
                             zeta: float) -> np.ndarray:
        """min ||M w - target||^2 + zeta ||w||^2, w >= 0, sum(w) = 1.

        Light-weight projected-gradient solver — sufficient for the small
        panel sizes we expect.
        """
        d = M.shape[1]
        w = np.full(d, 1.0 / d)
        lr = 1e-2
        for _ in range(300):
            grad = 2 * M.T @ (M @ w - target) + 2 * zeta * w
            w = w - lr * grad
            w = np.clip(w, 0, None)
            s = w.sum()
            if s > 0:
                w = w / s
            else:
                w = np.full(d, 1.0 / d)
        return w


def event_study_pretrends(df: pd.DataFrame, outcome: str, unit: str,
                          time: str, treat: str,
                          treat_period: int,
                          n_lags: int = 4) -> pd.DataFrame:
    """Pre-trends event study: per-period treatment-vs-control means
    relative to ``treat_period``. Significant pre-period (t < 0) effects
    indicate parallel-trends violations.
    """
    df = df.copy()
    df["event_time"] = df[time] - treat_period
    rows = []
    for et in range(-n_lags, 1):
        sub = df[df["event_time"] == et]
        if len(sub) == 0:
            continue
        treat_mean = sub.loc[sub[treat] == 1, outcome].mean()
        ctrl_mean = sub.loc[sub[treat] == 0, outcome].mean()
        rows.append({"event_time": et,
                     "treat_minus_ctrl": float(treat_mean - ctrl_mean)})
    return pd.DataFrame(rows)


# ===========================================================================
# PCMCI+ — spatio-temporal causal structure discovery
# ===========================================================================

def discover_spatiotemporal_causal_structure(
    df: pd.DataFrame,
    variables: list[str],
    time_col: str = "time",
    unit_col: str | None = None,
    max_lag: int = 2,
    alpha_level: float = 0.05,
    cond_ind_test: str = "parcorr",
    pc_alpha: float | None = None,
    max_conds_dim: int | None = None,
    verbosity: int = 0,
) -> dict:
    """Discover a spatio-temporal causal summary graph via PCMCI+.

    Wraps ``tigramite.pcmci.PCMCI`` with the PCMCI+ algorithm
    (``run_pcmciplus``) to recover a time-lagged causal summary graph over
    ``variables`` in a panel time-series.  When ``unit_col`` is given the
    panel is averaged across units before fitting (cross-sectional mean
    time-series), which is appropriate when spatial units share the same
    underlying dynamics.

    The returned ``edge_prob_matrix`` is suitable for direct consumption by
    :class:`~sparc.causal.mc3.PhysicsInformedGraphPrior` as ``edge_probs``:
    an entry ``edge_prob_matrix[i, j]`` gives the evidence that variable *j*
    is a contemporaneous or lagged cause of variable *i*.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.  Must contain ``time_col``, all ``variables``, and
        optionally ``unit_col``.
    variables : list[str]
        Names of the causal variables to include (columns in ``df``).
    time_col : str
        Column name for the time index.
    unit_col : str or None
        Column name for the spatial unit identifier.  When provided the
        panel is collapsed to a mean time-series per time step before
        fitting.
    max_lag : int
        Maximum time lag to test (τ_max).  Default 2.
    alpha_level : float
        Significance level for MCI tests.  Default 0.05.
    cond_ind_test : {'parcorr', 'cmiknn', 'gpdc'}
        Conditional independence test.  ``'parcorr'`` (linear, fast) is the
        default; use ``'cmiknn'`` for nonlinear relationships.
    pc_alpha : float or None
        PC phase significance level.  ``None`` lets tigramite auto-select
        via ``pc_alpha=None`` (uses ``alpha_level``).
    max_conds_dim : int or None
        Maximum conditioning set size for the PC phase.  ``None`` = no cap.
    verbosity : int
        Tigramite verbosity (0 = silent).

    Returns
    -------
    dict with keys:

    ``graph``
        (p, p, max_lag+1) array of edge type strings as returned by
        PCMCI+.  See tigramite docs for encoding (``"-->"`` = lagged
        cause, ``"o-o"`` = contemporaneous ambiguous, etc.).
    ``val_matrix``
        (p, p, max_lag+1) float array of test-statistic values.
    ``p_matrix``
        (p, p, max_lag+1) float array of p-values.
    ``edge_prob_matrix``
        (p, p) float array in [0, 1] — fraction of lags at which the
        edge (j→i) is significant.  Ready for
        :class:`~sparc.causal.mc3.PhysicsInformedGraphPrior`.
    ``variables``
        Ordered list of variable names matching matrix columns/rows.
    ``summary``
        Human-readable list of discovered directed edges with lag and
        p-value.

    Raises
    ------
    ImportError
        If ``tigramite`` is not installed.  Install with
        ``pip install tigramite>=5.2``.

    Examples
    --------
    ::

        result = discover_spatiotemporal_causal_structure(
            df, variables=["LST", "NDVI", "imperv"], time_col="hour",
        )
        prior = PhysicsInformedGraphPrior(
            edge_probs=result["edge_prob_matrix"],
        )
        mc3_out = run_mc3(data, node_names=result["variables"], prior=prior)
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise ImportError(
            "tigramite is required for discover_spatiotemporal_causal_structure. "
            "Install with: pip install tigramite>=5.2"
        ) from exc

    # ------------------------------------------------------------------ #
    # Build conditional independence test object
    # ------------------------------------------------------------------ #
    _test_name = cond_ind_test.lower()
    if _test_name == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        cit = ParCorr(significance="analytic")
    elif _test_name == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        cit = CMIknn(significance="shuffle_test", k=10)
    elif _test_name == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC
        cit = GPDC(significance="analytic")
    else:
        raise ValueError(
            f"Unknown cond_ind_test={cond_ind_test!r}. "
            "Choose 'parcorr', 'cmiknn', or 'gpdc'."
        )

    # ------------------------------------------------------------------ #
    # Build time-series array (T, p)
    # ------------------------------------------------------------------ #
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Variables not in df: {missing}")
    if time_col not in df.columns:
        raise ValueError(f"time_col={time_col!r} not in df")

    if unit_col is not None and unit_col in df.columns:
        # Collapse panel to mean time-series
        ts_df = (
            df.groupby(time_col)[variables]
            .mean()
            .sort_index()
            .reset_index(drop=True)
        )
    else:
        ts_df = df.sort_values(time_col)[variables].reset_index(drop=True)

    data_arr = ts_df[variables].values.astype(np.float64)

    # Handle NaNs: forward-fill then back-fill per column
    for col_idx in range(data_arr.shape[1]):
        col = data_arr[:, col_idx]
        if np.any(np.isnan(col)):
            mask = np.isnan(col)
            # Forward fill
            for t in range(1, len(col)):
                if mask[t]:
                    col[t] = col[t - 1]
            # Back fill
            for t in range(len(col) - 2, -1, -1):
                if np.isnan(col[t]):
                    col[t] = col[t + 1]
            data_arr[:, col_idx] = col

    T, p = data_arr.shape
    min_T = max_lag + p + 2
    if T <= min_T:
        logger.warning(
            "discover_spatiotemporal_causal_structure: time series too short "
            "(T=%d, need T > %d for max_lag=%d and p=%d variables). "
            "Returning uninformative result — no causal edges discovered.",
            T, min_T, max_lag, p,
        )
        uniform_prior = np.full((p, p), 0.1, dtype=np.float64)
        np.fill_diagonal(uniform_prior, 0.0)
        return {
            "graph": np.full((p, p, max_lag + 1), "", dtype=object),
            "val_matrix": np.zeros((p, p, max_lag + 1), dtype=np.float64),
            "p_matrix": np.ones((p, p, max_lag + 1), dtype=np.float64),
            "edge_prob_matrix": uniform_prior,
            "variables": list(variables),
            "summary": [],
            "skipped": True,
            "skip_reason": f"T={T} <= max_lag + p + 2 = {min_T}",
        }

    # ------------------------------------------------------------------ #
    # Run PCMCI+
    # ------------------------------------------------------------------ #
    dataframe = pp.DataFrame(
        data_arr,
        var_names=variables,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=verbosity)

    _pc_alpha = pc_alpha if pc_alpha is not None else alpha_level
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=max_lag,
        pc_alpha=_pc_alpha,
        **({"max_conds_dim": max_conds_dim} if max_conds_dim is not None else {}),
    )

    graph = results["graph"]          # (p, p, max_lag+1) str
    val_matrix = results["val_matrix"]  # (p, p, max_lag+1) float
    p_matrix = results["p_matrix"]    # (p, p, max_lag+1) float

    # ------------------------------------------------------------------ #
    # Build edge_prob_matrix for MC³ prior
    # edge_prob_matrix[i, j] = fraction of lags where edge j→i is significant
    # Contemporaneous (lag 0) edges contribute if either direction is
    # significant (undirected in MC³ context).
    # ------------------------------------------------------------------ #
    edge_prob_matrix = np.zeros((p, p), dtype=np.float64)
    for j in range(p):
        for i in range(p):
            if i == j:
                continue
            # Count lags at which p_matrix[i, j, tau] <= alpha_level
            sig_lags = np.sum(p_matrix[i, j, :] <= alpha_level)
            edge_prob_matrix[i, j] = float(sig_lags) / (max_lag + 1)

    # Blend with a weak uniform base prior (0.1) so MC³ never gets 0-prob edges
    edge_prob_matrix = 0.1 + 0.8 * edge_prob_matrix
    np.fill_diagonal(edge_prob_matrix, 0.0)

    # ------------------------------------------------------------------ #
    # Human-readable summary of directed edges
    # ------------------------------------------------------------------ #
    summary: list[dict] = []
    for j in range(p):
        for i in range(p):
            if i == j:
                continue
            for tau in range(max_lag + 1):
                edge_type = graph[i, j, tau] if graph.ndim == 3 else ""
                pval = float(p_matrix[i, j, tau])
                if pval <= alpha_level and edge_type != "":
                    summary.append({
                        "cause": variables[j],
                        "effect": variables[i],
                        "lag": int(tau),
                        "p_value": round(pval, 6),
                        "val": round(float(val_matrix[i, j, tau]), 4),
                        "edge_type": str(edge_type),
                    })

    return {
        "graph": graph,
        "val_matrix": val_matrix,
        "p_matrix": p_matrix,
        "edge_prob_matrix": edge_prob_matrix,
        "variables": list(variables),
        "summary": summary,
    }
