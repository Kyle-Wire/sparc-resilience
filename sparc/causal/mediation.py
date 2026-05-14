"""
mediation — Natural Direct and Indirect Effect decomposition.

Estimates NDE (Natural Direct Effect), NIE (Natural Indirect Effect), and
CTE (Controlled Total Effect = NDE + NIE) for each treatment → mediator →
outcome path identified in the causal DAG.

Two estimation paths:
  - **Linear** (product-of-coefficients): Fast, interpretable. Assumes
    linear structural equations. Bootstrap CIs (1000 draws).
  - **Nonlinear** (g-computation via Monte Carlo): Consistent under
    nonlinear models. Uses cross-fit HGB nuisance models. 500 MC draws.

References
----------
- Pearl (2001): Direct and indirect effects.
- VanderWeele (2015): Explanation in Causal Inference.
- Imai, Keele & Tingley (2010): A general approach to causal mediation analysis.

Typical usage::

    from sparc.causal.mediation import MediationDecomposer

    decomposer = MediationDecomposer()
    results = decomposer.decompose_all(data, dag_def, roles)
    # results: list[MediationResult]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MediationResult:
    """Per (treatment, mediator, outcome) decomposition."""
    treatment: str
    mediator: str
    outcome: str

    # Linear path (product-of-coefficients + bootstrap CIs)
    NDE_linear: float = float("nan")
    NIE_linear: float = float("nan")
    CTE_linear: float = float("nan")
    NDE_linear_ci: Tuple[float, float] = (float("nan"), float("nan"))
    NIE_linear_ci: Tuple[float, float] = (float("nan"), float("nan"))
    CTE_linear_ci: Tuple[float, float] = (float("nan"), float("nan"))

    # Nonlinear path (g-computation + bootstrap CIs)
    NDE_nonlinear: float = float("nan")
    NIE_nonlinear: float = float("nan")
    CTE_nonlinear: float = float("nan")
    NDE_nonlinear_ci: Tuple[float, float] = (float("nan"), float("nan"))
    NIE_nonlinear_ci: Tuple[float, float] = (float("nan"), float("nan"))
    CTE_nonlinear_ci: Tuple[float, float] = (float("nan"), float("nan"))

    n_obs: int = 0
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "treatment": self.treatment,
            "mediator": self.mediator,
            "outcome": self.outcome,
            "NDE_linear": _f(self.NDE_linear),
            "NIE_linear": _f(self.NIE_linear),
            "CTE_linear": _f(self.CTE_linear),
            "NDE_linear_ci": [_f(self.NDE_linear_ci[0]), _f(self.NDE_linear_ci[1])],
            "NIE_linear_ci": [_f(self.NIE_linear_ci[0]), _f(self.NIE_linear_ci[1])],
            "CTE_linear_ci": [_f(self.CTE_linear_ci[0]), _f(self.CTE_linear_ci[1])],
            "NDE_nonlinear": _f(self.NDE_nonlinear),
            "NIE_nonlinear": _f(self.NIE_nonlinear),
            "CTE_nonlinear": _f(self.CTE_nonlinear),
            "NDE_nonlinear_ci": [_f(self.NDE_nonlinear_ci[0]), _f(self.NDE_nonlinear_ci[1])],
            "NIE_nonlinear_ci": [_f(self.NIE_nonlinear_ci[0]), _f(self.NIE_nonlinear_ci[1])],
            "CTE_nonlinear_ci": [_f(self.CTE_nonlinear_ci[0]), _f(self.CTE_nonlinear_ci[1])],
            "n_obs": self.n_obs,
            "warnings": self.warnings,
        }


def _f(v: float) -> float:
    """Return None-safe float (NaN → None for JSON serialization)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return float(v)


def _ols_coeff(X: np.ndarray, y: np.ndarray, col_idx: int) -> float:
    """OLS coefficient for column col_idx in X predicting y."""
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(coeffs[col_idx])
    except Exception:
        return float("nan")


def _fit_hgb(X: np.ndarray, y: np.ndarray):
    """Fit a HistGradientBoostingRegressor; return fitted model."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )
    m.fit(X, y)
    return m


class MediationDecomposer:
    """
    Decomposes total treatment effects into NDE and NIE per mediator path.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap draws for linear-path CIs.
    n_mc : int
        Number of Monte Carlo draws for nonlinear g-computation CIs.
    treatment_delta : float
        Magnitude of the treatment change used to define effects (1.0 = 1 SD
        or 1 raw unit, depending on whether data is standardized).
    random_state : int
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        n_mc: int = 500,
        treatment_delta: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.n_mc = n_mc
        self.treatment_delta = treatment_delta
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def decompose_all(
        self,
        data: pd.DataFrame,
        dag_def: dict,
        roles: Dict[str, List[str]],
    ) -> List[MediationResult]:
        """
        Decompose effects for all (treatment, mediator, outcome) triples
        found in the DAG.

        Parameters
        ----------
        data : observational dataset
        dag_def : parsed DAG definition dict (from ``load_dag()``)
        roles : node-role mapping from ``get_node_roles(dag_def)``

        Returns
        -------
        list of MediationResult — one per identified mediator path
        """
        treatments = roles.get("treatments", [])
        mediators = roles.get("mediators", [])
        outcomes = roles.get("outcomes", [])

        if not treatments or not mediators or not outcomes:
            return []

        # Build adjacency from DAG edges
        edges = {(e["source"], e["target"]) for e in dag_def.get("edges", [])}

        results = []
        for treatment in treatments:
            for mediator in mediators:
                for outcome in outcomes:
                    # Only decompose when T→M and M→Y edges exist in expert DAG
                    if (treatment, mediator) not in edges:
                        continue
                    if (mediator, outcome) not in edges:
                        continue
                    if treatment not in data.columns:
                        continue
                    if mediator not in data.columns:
                        continue
                    if outcome not in data.columns:
                        continue

                    # Confounders: all measured confounders in data columns
                    confounder_nodes = [
                        n["name"] for n in dag_def.get("nodes", [])
                        if n.get("type") == "confounder"
                        and n["name"] in data.columns
                    ]

                    result = self._decompose_path(
                        data=data,
                        treatment=treatment,
                        mediator=mediator,
                        outcome=outcome,
                        confounders=confounder_nodes,
                    )
                    results.append(result)

        return results

    # ------------------------------------------------------------------
    # Single-path decomposition
    # ------------------------------------------------------------------

    def _decompose_path(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        confounders: List[str],
    ) -> MediationResult:
        result = MediationResult(
            treatment=treatment,
            mediator=mediator,
            outcome=outcome,
            n_obs=len(data),
        )

        df = data[[treatment, mediator, outcome] + confounders].dropna()
        if len(df) < 30:
            result.warnings.append(
                f"Only {len(df)} complete cases — mediation estimates unreliable."
            )

        # ---- Linear path ----
        try:
            nde_l, nie_l, cte_l, nde_ci, nie_ci, cte_ci = self._linear_decompose(
                df, treatment, mediator, outcome, confounders
            )
            result.NDE_linear = nde_l
            result.NIE_linear = nie_l
            result.CTE_linear = cte_l
            result.NDE_linear_ci = nde_ci
            result.NIE_linear_ci = nie_ci
            result.CTE_linear_ci = cte_ci
        except Exception as exc:
            result.warnings.append(f"Linear mediation failed: {exc}")

        # ---- Nonlinear path ----
        try:
            nde_nl, nie_nl, cte_nl, nde_ci_nl, nie_ci_nl, cte_ci_nl = (
                self._gcomputation_decompose(
                    df, treatment, mediator, outcome, confounders
                )
            )
            result.NDE_nonlinear = nde_nl
            result.NIE_nonlinear = nie_nl
            result.CTE_nonlinear = cte_nl
            result.NDE_nonlinear_ci = nde_ci_nl
            result.NIE_nonlinear_ci = nie_ci_nl
            result.CTE_nonlinear_ci = cte_ci_nl
        except Exception as exc:
            result.warnings.append(f"Nonlinear (g-computation) mediation failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Linear path — product-of-coefficients + bootstrap
    # ------------------------------------------------------------------

    def _linear_decompose(
        self,
        df: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        confounders: List[str],
    ) -> Tuple:
        """
        Product-of-coefficients (Baron-Kenny / Judd-Kenny) approach.

        alpha  = T → M coefficient (T equation, controlling for confounders)
        gamma  = M → Y coefficient (Y equation, controlling for T + confounders)
        direct = T → Y coefficient (Y equation, controlling for M + confounders)

        NIE = alpha * gamma  (indirect via M)
        NDE = direct         (direct, M held at natural value)
        CTE = NDE + NIE
        """
        T = df[treatment].values
        M = df[mediator].values
        Y = df[outcome].values
        W = df[confounders].values if confounders else None

        nde, nie = self._linear_point(T, M, Y, W)
        cte = nde + nie

        # Bootstrap CIs
        n = len(T)
        boot_nde = np.zeros(self.n_bootstrap)
        boot_nie = np.zeros(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, size=n)
            T_b = T[idx]; M_b = M[idx]; Y_b = Y[idx]
            W_b = W[idx] if W is not None else None
            try:
                boot_nde[b], boot_nie[b] = self._linear_point(T_b, M_b, Y_b, W_b)
            except Exception:
                boot_nde[b] = nde
                boot_nie[b] = nie

        boot_cte = boot_nde + boot_nie
        alpha = 0.025  # 95% CI
        nde_ci = (float(np.quantile(boot_nde, alpha)), float(np.quantile(boot_nde, 1 - alpha)))
        nie_ci = (float(np.quantile(boot_nie, alpha)), float(np.quantile(boot_nie, 1 - alpha)))
        cte_ci = (float(np.quantile(boot_cte, alpha)), float(np.quantile(boot_cte, 1 - alpha)))

        return nde, nie, cte, nde_ci, nie_ci, cte_ci

    def _linear_point(
        self,
        T: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray | None,
    ) -> Tuple[float, float]:
        """Return (NDE, NIE) for one sample."""
        n = len(T)

        # M equation: M ~ T + W  →  alpha = coeff of T
        if W is not None and W.shape[1] > 0:
            X_m = np.column_stack([np.ones(n), T, W])
        else:
            X_m = np.column_stack([np.ones(n), T])
        alpha = _ols_coeff(X_m, M, col_idx=1)

        # Y equation: Y ~ T + M + W  →  direct (NDE) and gamma (M coeff)
        if W is not None and W.shape[1] > 0:
            X_y = np.column_stack([np.ones(n), T, M, W])
        else:
            X_y = np.column_stack([np.ones(n), T, M])
        coeffs = np.linalg.lstsq(X_y, Y, rcond=None)[0]
        direct = float(coeffs[1]) * self.treatment_delta   # NDE
        gamma = float(coeffs[2])

        nie = alpha * gamma * self.treatment_delta          # NIE
        return direct, nie

    # ------------------------------------------------------------------
    # Nonlinear path — g-computation (Monte Carlo integration)
    # ------------------------------------------------------------------

    def _gcomputation_decompose(
        self,
        df: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        confounders: List[str],
    ) -> Tuple:
        """
        G-computation via Monte Carlo integration.

        Fits two nonlinear models:
          f_M(T, W) → predicted mediator
          f_Y(T, M, W) → predicted outcome

        NDE = E[f_Y(T+δ, f_M(T, W), W)] - E[f_Y(T, f_M(T, W), W)]
              (treatment changes, mediator held at its natural value under T)
        NIE = E[f_Y(T, f_M(T+δ, W), W)] - E[f_Y(T, f_M(T, W), W)]
              (treatment unchanged, but mediator shifts as it would under T+δ)
        CTE = NDE + NIE
        """
        T = df[treatment].values
        M = df[mediator].values
        Y = df[outcome].values
        W = df[confounders].values if confounders else None
        delta = self.treatment_delta

        nde, nie = self._gcomp_point(T, M, Y, W, delta)
        cte = nde + nie

        # Bootstrap CIs
        n = len(T)
        boot_nde = np.zeros(self.n_mc)
        boot_nie = np.zeros(self.n_mc)
        for b in range(self.n_mc):
            idx = self.rng.integers(0, n, size=n)
            T_b = T[idx]; M_b = M[idx]; Y_b = Y[idx]
            W_b = W[idx] if W is not None else None
            try:
                boot_nde[b], boot_nie[b] = self._gcomp_point(T_b, M_b, Y_b, W_b, delta)
            except Exception:
                boot_nde[b] = nde
                boot_nie[b] = nie

        boot_cte = boot_nde + boot_nie
        alpha = 0.025
        nde_ci = (float(np.quantile(boot_nde, alpha)), float(np.quantile(boot_nde, 1 - alpha)))
        nie_ci = (float(np.quantile(boot_nie, alpha)), float(np.quantile(boot_nie, 1 - alpha)))
        cte_ci = (float(np.quantile(boot_cte, alpha)), float(np.quantile(boot_cte, 1 - alpha)))

        return nde, nie, cte, nde_ci, nie_ci, cte_ci

    def _gcomp_point(
        self,
        T: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray | None,
        delta: float,
    ) -> Tuple[float, float]:
        """Return (NDE, NIE) via g-computation for one sample."""
        n = len(T)

        # Build feature matrices
        if W is not None and W.shape[1] > 0:
            X_m_base = np.column_stack([T, W])
            X_m_shift = np.column_stack([T + delta, W])
        else:
            X_m_base = T.reshape(-1, 1)
            X_m_shift = (T + delta).reshape(-1, 1)

        # f_M: mediator model
        f_m = _fit_hgb(X_m_base, M)
        M_natural = f_m.predict(X_m_base)        # M under observed T
        M_shifted = f_m.predict(X_m_shift)       # M under T + delta

        # f_Y: outcome model
        if W is not None and W.shape[1] > 0:
            X_y_base   = np.column_stack([T,         M_natural, W])
            X_y_direct = np.column_stack([T + delta, M_natural, W])
            X_y_nie    = np.column_stack([T,         M_shifted, W])
        else:
            X_y_base   = np.column_stack([T,         M_natural])
            X_y_direct = np.column_stack([T + delta, M_natural])
            X_y_nie    = np.column_stack([T,         M_shifted])

        f_y = _fit_hgb(
            np.column_stack([T, M_natural, W]) if W is not None and W.shape[1] > 0
            else np.column_stack([T, M_natural]),
            Y,
        )

        Y_base   = f_y.predict(X_y_base)
        Y_direct = f_y.predict(X_y_direct)
        Y_nie    = f_y.predict(X_y_nie)

        nde = float(np.mean(Y_direct - Y_base))   # T shifts, M stays natural
        nie = float(np.mean(Y_nie   - Y_base))    # T stays, M shifts as under T+δ
        return nde, nie

    # ------------------------------------------------------------------
    # Fairness audit — stratified NDE/NIE by protected attribute
    # ------------------------------------------------------------------

    def fairness_audit(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        confounders: List[str],
        protected_attr: str,
        n_strata: int = 10,
    ) -> "FairnessAuditResult":
        """Stratify NDE / NIE by a protected attribute and report disparities.

        For each stratum of ``protected_attr`` (quantile-bins for continuous
        variables, unique levels for categorical) the full linear + nonlinear
        mediation decomposition is re-run.  A disparity summary reports the
        max–min range across strata for each effect estimate.

        Parameters
        ----------
        data : pd.DataFrame
        treatment, mediator, outcome, confounders : str / list[str]
        protected_attr : str
            Column to stratify by.  Continuous columns are binned into
            ``n_strata`` equal-frequency quantile groups.  Categorical (object /
            bool) columns are used as-is.
        n_strata : int
            Number of quantile bins when ``protected_attr`` is numeric.

        Returns
        -------
        FairnessAuditResult
        """
        if protected_attr not in data.columns:
            raise ValueError(
                f"protected_attr '{protected_attr}' not found in data columns."
            )

        col_dtype = data[protected_attr].dtype
        if col_dtype.kind in ("O", "b", "U"):  # categorical / boolean / string
            strata_labels = data[protected_attr].unique().tolist()
            strata_masks = {
                str(lbl): data[protected_attr] == lbl
                for lbl in strata_labels
            }
        else:
            # Numeric: bin into n_strata equal-frequency quantiles
            try:
                bins = pd.qcut(
                    data[protected_attr], q=n_strata, duplicates="drop"
                )
            except ValueError:
                bins = pd.cut(data[protected_attr], bins=n_strata)
            strata_masks = {
                str(interval): bins == interval
                for interval in bins.cat.categories
            }

        stratum_results: Dict[str, MediationResult] = {}
        for label, mask in strata_masks.items():
            stratum_df = data[mask].reset_index(drop=True)
            if len(stratum_df) < 30:
                logger.warning(
                    "Fairness audit: stratum '%s' has only %d rows — skipping.",
                    label, len(stratum_df),
                )
                continue
            result = self._decompose_path(
                data=stratum_df,
                treatment=treatment,
                mediator=mediator,
                outcome=outcome,
                confounders=confounders,
            )
            stratum_results[label] = result

        return FairnessAuditResult(
            treatment=treatment,
            mediator=mediator,
            outcome=outcome,
            protected_attr=protected_attr,
            stratum_results=stratum_results,
        )


@dataclass
class FairnessAuditResult:
    """Per-stratum mediation decomposition and disparity summary."""

    treatment: str
    mediator: str
    outcome: str
    protected_attr: str
    stratum_results: Dict[str, "MediationResult"] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Disparity metrics
    # ------------------------------------------------------------------

    def _collect(self, attr: str) -> List[float]:
        return [
            getattr(r, attr)
            for r in self.stratum_results.values()
            if np.isfinite(getattr(r, attr, float("nan")))
        ]

    @property
    def nde_disparity(self) -> float:
        """Max − min NDE (linear) across strata."""
        vals = self._collect("NDE_linear")
        return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")

    @property
    def nie_disparity(self) -> float:
        """Max − min NIE (linear) across strata."""
        vals = self._collect("NIE_linear")
        return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")

    @property
    def cte_disparity(self) -> float:
        """Max − min CTE (linear) across strata."""
        vals = self._collect("CTE_linear")
        return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")

    @property
    def disparity_ratio(self) -> float:
        """max|NDE| / min|NDE| — relative size of the most- vs least-affected stratum.

        Returns ``nan`` when any NDE is ~zero (avoids division by zero).
        """
        vals = [abs(v) for v in self._collect("NDE_linear")]
        if len(vals) < 2 or min(vals) < 1e-9:
            return float("nan")
        return float(max(vals) / min(vals))

    def summary_table(self) -> pd.DataFrame:
        """DataFrame with one row per stratum and key effect estimates."""
        rows = []
        for label, r in self.stratum_results.items():
            rows.append({
                "stratum": label,
                "n_obs": r.n_obs,
                "NDE_linear": _f(r.NDE_linear),
                "NIE_linear": _f(r.NIE_linear),
                "CTE_linear": _f(r.CTE_linear),
                "NDE_nonlinear": _f(r.NDE_nonlinear),
                "NIE_nonlinear": _f(r.NIE_nonlinear),
                "CTE_nonlinear": _f(r.CTE_nonlinear),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            # Append disparity row
            disp_row = {
                "stratum": "_disparity_",
                "n_obs": None,
                "NDE_linear": _f(self.nde_disparity),
                "NIE_linear": _f(self.nie_disparity),
                "CTE_linear": _f(self.cte_disparity),
                "NDE_nonlinear": None,
                "NIE_nonlinear": None,
                "CTE_nonlinear": None,
            }
            df = pd.concat([df, pd.DataFrame([disp_row])], ignore_index=True)
        return df

    def as_dict(self) -> dict:
        return {
            "treatment": self.treatment,
            "mediator": self.mediator,
            "outcome": self.outcome,
            "protected_attr": self.protected_attr,
            "n_strata": len(self.stratum_results),
            "nde_disparity": _f(self.nde_disparity),
            "nie_disparity": _f(self.nie_disparity),
            "cte_disparity": _f(self.cte_disparity),
            "disparity_ratio": _f(self.disparity_ratio),
            "strata": {k: v.as_dict() for k, v in self.stratum_results.items()},
        }
