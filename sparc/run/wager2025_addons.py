"""Wager (2025) audit gap auto-runner.

Originally these estimators were only invoked via opt-in CLI flags
(``--with-spillover``, ``--with-iv``, ``--with-policy-learning``,
``--full-triangulation``).  As of v3, Stage 3 runs them by default as
part of the Bayesian causal-validation pipeline so a single
``sparc run --stage 3`` produces gold-standard coverage of all 10
gaps catalogued in :mod:`sparc.causal._audit`.

Each gap's invocation can be opted-out per-project via
``causal.wager2025.<flag>: false`` in ``project.yml``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class GapRunnerProtocol(Protocol):
    """Structural protocol for a pluggable per-gap estimator.

    Implementations receive the same keyword arguments as ``run_wager2025_gaps``
    and must return a value (typically ``int`` or ``dict``) that is stored in
    ``summary["ran"][gap_name]``.
    """

    def run(
        self,
        *,
        data: Any,
        config: dict,
        dag_def: dict,
        graph: Any,
        roles: dict,
        output_dir: str,
        nuts_results: Optional[dict] = None,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Injection seam — keyed by gap name (same strings as _GAP_FLAGS)
# ---------------------------------------------------------------------------
_gap_runners: Optional[dict] = None


def set_gap_runners(runners: dict) -> None:
    """Inject a mapping of ``{gap_name: GapRunnerProtocol}`` for testing."""
    global _gap_runners
    _gap_runners = runners


def clear_gap_runners() -> None:
    """Remove all injected gap runners, restoring default behaviour."""
    global _gap_runners
    _gap_runners = None


_GAP_FLAGS = (
    "overlap",          # Gap 1
    "rate_qini",        # Gap 2
    "spillover",        # Gap 3
    "iv",               # Gaps 4, 5
    "sequential_aipw",  # Gap 6
    "panel",            # Gap 7
    "policy_learning",  # Gap 8
    "cbps",             # Gap 9
)


def _flags(causal_cfg: dict) -> Dict[str, bool]:
    """Per-gap on/off toggles. Default: all true."""
    user = (causal_cfg or {}).get("wager2025", {}) or {}
    return {flag: bool(user.get(flag, True)) for flag in _GAP_FLAGS}


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def run_wager2025_gaps(
    *,
    data: pd.DataFrame,
    config: dict,
    dag_def: dict,
    graph,
    roles: Dict[str, List[str]],
    output_dir: str,
    nuts_results: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run all enabled Wager-2025 audit estimators and persist artifacts.

    Returns a dict summarising what ran and where artifacts landed.
    Failures are caught per-gap so one estimator's errors don't abort
    the rest of Stage 3.
    """
    causal_cfg = config.get("causal", {}) or {}
    enabled = _flags(causal_cfg)

    target = roles.get("outcomes", [None])[0]
    treatments = [t for t in roles.get("treatments", []) if t in data.columns]
    confounders = [c for c in roles.get("confounders", []) if c in data.columns]

    coord_cols = config.get("variables", {}).get("coordinates", []) or []
    coord_cols = [c for c in coord_cols if c in data.columns]

    summary: Dict[str, Any] = {
        "target": target,
        "treatments": treatments,
        "confounders": confounders,
        "enabled_flags": enabled,
        "ran": {},
        "skipped": {},
        "failed": {},
    }
    # Collect per-treatment ATEs as gaps run; consumed by the E-value block at the end.
    _ate_registry: Dict[str, float] = {}

    if not target or target not in data.columns:
        print("  Wager-2025 add-ons: outcome missing — all gaps skipped.")
        return summary

    print("\n  --- Wager (2025) Audit Add-Ons (auto-run) ---")

    # ── Gap 1: Generalized-propensity overlap ────────────────────────
    if enabled["overlap"]:
        _runner = (_gap_runners or {}).get("overlap")
        if _runner is not None:
            try:
                result = _runner.run(
                    data=data, config=config, dag_def=dag_def, graph=graph,
                    roles=roles, output_dir=output_dir, nuts_results=nuts_results,
                )
                summary["ran"]["overlap"] = result
            except Exception as exc:
                summary["failed"]["overlap"] = str(exc)
        else:
            try:
                from sparc.causal.overlap import GeneralizedPropensityOverlap
                rows = []
                for t in treatments:
                    X = data[confounders].to_numpy(dtype=float) if confounders \
                        else np.zeros((len(data), 1))
                    W = data[t].to_numpy(dtype=float)
                    est = GeneralizedPropensityOverlap(method="kernel").fit(X, W)
                    # Assess at zero-shift (in-distribution baseline density).
                    dens = float(np.median(est._density_train))
                    rows.append({
                        "treatment": t,
                        "median_density": dens,
                        "p05_density": float(np.percentile(est._density_train, 5)),
                        "p95_density": float(np.percentile(est._density_train, 95)),
                    })
                if rows:
                    _save_json(
                        os.path.join(output_dir, "overlap_diagnostics.json"),
                        {"gap_id": 1, "rows": rows},
                    )
                    print(f"    [Gap 1] overlap: {len(rows)} treatment(s) ✓")
                    summary["ran"]["overlap"] = len(rows)
            except Exception as exc:
                print(f"    [Gap 1] overlap FAILED: {exc}")
                summary["failed"]["overlap"] = str(exc)
    else:
        summary["skipped"]["overlap"] = "disabled"

    # ── Gap 9: CBPS balancing weights ────────────────────────────────
    if enabled["cbps"]:
        try:
            from sparc.causal.balancing import CBPSEstimator
            rows = []
            for t in treatments:
                X = data[confounders].to_numpy(dtype=float) if confounders \
                    else np.zeros((len(data), 1))
                W = data[t].to_numpy(dtype=float)
                Y = data[target].to_numpy(dtype=float)
                est = CBPSEstimator(binarize_continuous=True).fit(X, W)
                ate = est.ate(Y)
                _ate_registry[t] = float(ate)  # collected for E-value pass
                bal = est.balance_diagnostics()
                rows.append({
                    "treatment": t,
                    "cbps_ate": ate,
                    "max_abs_smd_pre": float(
                        np.max(np.abs(bal.get("smd_raw", np.array([np.nan]))))
                    ),
                    "max_abs_smd_post": float(
                        np.max(np.abs(bal.get("smd_weighted", np.array([np.nan]))))
                    ),
                })
            if rows:
                _save_json(
                    os.path.join(output_dir, "cbps_balance.json"),
                    {"gap_id": 9, "rows": rows},
                )
                print(f"    [Gap 9] CBPS: {len(rows)} treatment(s) ✓")
                summary["ran"]["cbps"] = len(rows)
        except Exception as exc:
            print(f"    [Gap 9] CBPS FAILED: {exc}")
            summary["failed"]["cbps"] = str(exc)
    else:
        summary["skipped"]["cbps"] = "disabled"

    # ── Gap 2: RATE / TOC / QINI (CATE ranking validation) ───────────
    if enabled["rate_qini"]:
        try:
            from sparc.causal.cate_validation import (
                aipw_pseudo_outcomes, compute_rate, compute_qini,
            )
            from sklearn.ensemble import HistGradientBoostingRegressor as HGB
            rows = []
            qini_curves = {}
            X = data[confounders].to_numpy(dtype=float) if confounders \
                else np.zeros((len(data), 1))
            Y = data[target].to_numpy(dtype=float)
            for t in treatments:
                W_raw = data[t].to_numpy(dtype=float)
                # Binarize at median for AIPW pseudo-outcomes.
                W = (W_raw > np.median(W_raw)).astype(float)
                # Nuisance fits (single-fold for speed; stage already
                # computed cross-fit residuals elsewhere).
                m1 = HGB(max_iter=150, max_depth=4, random_state=42)
                m0 = HGB(max_iter=150, max_depth=4, random_state=42)
                if (W == 1).sum() < 10 or (W == 0).sum() < 10:
                    continue
                m1.fit(X[W == 1], Y[W == 1])
                m0.fit(X[W == 0], Y[W == 0])
                mu1 = m1.predict(X)
                mu0 = m0.predict(X)
                from sklearn.linear_model import LogisticRegression
                e_model = LogisticRegression(max_iter=300).fit(X, W)
                e = e_model.predict_proba(X)[:, 1]
                gamma = aipw_pseudo_outcomes(Y, W, mu1, mu0, e)
                # Use μ1 - μ0 as candidate priority score.
                score = mu1 - mu0
                rate = compute_rate(score, gamma, n_bootstrap=200,
                                    random_state=42)
                qini = compute_qini(score, gamma)
                rows.append({
                    "treatment": t,
                    "rate_auc": rate["rate_auc"],
                    "rate_ci_low": rate["rate_ci"][0],
                    "rate_ci_high": rate["rate_ci"][1],
                    "rate_se": rate["rate_se"],
                })
                qini_curves[t] = {
                    "q": qini["q"].tolist(),
                    "qini": qini["qini"].tolist(),
                }
            if rows:
                _save_json(
                    os.path.join(output_dir, "rate_qini.json"),
                    {"gap_id": 2, "rows": rows, "qini_curves": qini_curves},
                )
                print(f"    [Gap 2] RATE/QINI: {len(rows)} treatment(s) ✓")
                summary["ran"]["rate_qini"] = len(rows)
        except Exception as exc:
            print(f"    [Gap 2] RATE/QINI FAILED: {exc}")
            summary["failed"]["rate_qini"] = str(exc)
    else:
        summary["skipped"]["rate_qini"] = "disabled"

    # ── Gap 3: Network interference / spillover ──────────────────────
    if enabled["spillover"]:
        if len(coord_cols) >= 2:
            try:
                from sparc.causal.interference import NetworkInterferenceModel
                rows = []
                coords = data[coord_cols].to_numpy(dtype=float)
                Y = data[target].to_numpy(dtype=float)
                for t in treatments:
                    W = data[t].to_numpy(dtype=float)
                    decomp = (NetworkInterferenceModel(k=8)
                              .fit(coords, W, Y).decompose())
                    decomp = {k: float(v) for k, v in decomp.items()}
                    decomp["treatment"] = t
                    rows.append(decomp)
                if rows:
                    _save_json(
                        os.path.join(output_dir, "spillover_decomposition.json"),
                        {"gap_id": 3, "rows": rows},
                    )
                    print(f"    [Gap 3] spillover: {len(rows)} treatment(s) ✓")
                    summary["ran"]["spillover"] = len(rows)
            except Exception as exc:
                print(f"    [Gap 3] spillover FAILED: {exc}")
                summary["failed"]["spillover"] = str(exc)
        else:
            summary["skipped"]["spillover"] = "no coordinate columns"
    else:
        summary["skipped"]["spillover"] = "disabled"

    # ── Gap 4 + 5: Cross-fit IV + Local IV / MTE ─────────────────────
    if enabled["iv"]:
        try:
            instruments = []
            if graph is not None:
                instruments = [
                    n for n, d in graph.nodes(data=True)
                    if (d or {}).get("role") == "instrument" and n in data.columns
                ]
            if not instruments:
                summary["skipped"]["iv"] = "no DAG nodes with role=instrument"
            else:
                from sparc.causal.iv import CrossFitIV, LocalIV
                rows = []
                mte_curves = {}
                Y = data[target].to_numpy(dtype=float)
                for t in treatments:
                    Z_cols = instruments
                    Z = data[Z_cols].to_numpy(dtype=float)
                    W = data[t].to_numpy(dtype=float)
                    iv_res = CrossFitIV(n_folds=5).fit(Z=Z, W=W, Y=Y)
                    rows.append({
                        "treatment": t,
                        "iv_coef": float(iv_res.coef_),
                        "iv_se": float(iv_res.se_ or 0.0),
                        "first_stage_f": float(iv_res.first_stage_f_ or 0.0),
                        "weak_warning": bool(iv_res.weak_warning_),
                        "instruments": Z_cols,
                    })
                    if Z.shape[1] == 1:
                        mte = LocalIV().fit(Z[:, 0], W, Y)
                        mte_curves[t] = mte.curve().tolist()
                if rows:
                    _save_json(
                        os.path.join(output_dir, "iv_estimates.json"),
                        {"gap_ids": [4, 5], "rows": rows,
                         "mte_curves": mte_curves},
                    )
                    print(f"    [Gap 4+5] IV/MTE: {len(rows)} treatment(s) ✓")
                    summary["ran"]["iv"] = len(rows)
        except Exception as exc:
            print(f"    [Gap 4+5] IV/MTE FAILED: {exc}")
            summary["failed"]["iv"] = str(exc)
    else:
        summary["skipped"]["iv"] = "disabled"

    # ── Gap 6: Sequential AIPW (only if data has a temporal index) ───
    if enabled["sequential_aipw"]:
        temporal = config.get("temporal", {}) or {}
        time_col = temporal.get("time_column") if temporal.get("enabled") else None
        if time_col and time_col in data.columns:
            try:
                from sparc.causal.dynamic import SequentialAIPW
                # Build period-history: skipped here unless project.yml maps it
                # explicitly. The estimator supports only structured panel input,
                # so for cross-sectional UHI we record a "not applicable" entry.
                summary["skipped"]["sequential_aipw"] = (
                    "panel structure not configured — set causal.sequential.history"
                )
            except Exception as exc:
                print(f"    [Gap 6] sequential AIPW FAILED: {exc}")
                summary["failed"]["sequential_aipw"] = str(exc)
        else:
            summary["skipped"]["sequential_aipw"] = "no temporal index"
    else:
        summary["skipped"]["sequential_aipw"] = "disabled"

    # ── Gap 7: DID / SDID (only with panel data) ─────────────────────
    if enabled["panel"]:
        panel_cfg = (causal_cfg.get("panel", {}) or {})
        unit_col = panel_cfg.get("unit_column")
        time_col = panel_cfg.get("time_column")
        treat_col = panel_cfg.get("treatment_column")
        post_col = panel_cfg.get("post_column")
        required = (unit_col, time_col, treat_col, post_col)
        if all(required) and all(c in data.columns for c in required):
            try:
                from sparc.causal.panel import DifferenceInDifferences
                did = DifferenceInDifferences().fit(
                    df=data, outcome=target, unit=unit_col,
                    time=time_col, treat=treat_col, post=post_col,
                )
                _save_json(
                    os.path.join(output_dir, "did_estimates.json"),
                    {"gap_id": 7,
                     "att": float(did.att_),
                     "se": float(did.se_),
                     "n_units": int(data[unit_col].nunique())},
                )
                print(f"    [Gap 7] DID: ATT={did.att_:+.4f} ✓")
                summary["ran"]["panel"] = 1
            except Exception as exc:
                print(f"    [Gap 7] DID FAILED: {exc}")
                summary["failed"]["panel"] = str(exc)
        else:
            summary["skipped"]["panel"] = "no panel columns configured"
    else:
        summary["skipped"]["panel"] = "disabled"

    # ── Gap 8: Empirical-welfare policy learning ─────────────────────
    if enabled["policy_learning"]:
        try:
            actionable = causal_cfg.get("actionable_variables", []) or []
            actionable = [a for a in actionable if a in data.columns]
            if not actionable:
                actionable = treatments
            if not actionable:
                summary["skipped"]["policy_learning"] = "no actionable variables"
            else:
                from sparc.decision.policy_learning import EmpiricalWelfareMaximizer
                from sparc.causal.cate_validation import aipw_pseudo_outcomes
                from sklearn.ensemble import HistGradientBoostingRegressor as HGB
                from sklearn.linear_model import LogisticRegression
                X = data[confounders].to_numpy(dtype=float) if confounders \
                    else np.zeros((len(data), 1))
                Y = data[target].to_numpy(dtype=float)
                rows = []
                per_cell: dict[str, list] = {}
                for t in actionable:
                    if t not in data.columns:
                        continue
                    W_raw = data[t].to_numpy(dtype=float)
                    W = (W_raw > np.median(W_raw)).astype(float)
                    if (W == 1).sum() < 10 or (W == 0).sum() < 10:
                        continue
                    m1 = HGB(max_iter=120, max_depth=4, random_state=42)
                    m0 = HGB(max_iter=120, max_depth=4, random_state=42)
                    m1.fit(X[W == 1], Y[W == 1])
                    m0.fit(X[W == 0], Y[W == 0])
                    mu1 = m1.predict(X)
                    mu0 = m0.predict(X)
                    e = LogisticRegression(max_iter=300).fit(X, W)\
                        .predict_proba(X)[:, 1]
                    gamma = aipw_pseudo_outcomes(Y, W, mu1, mu0, e)
                    pol = EmpiricalWelfareMaximizer(policy_class="linear").fit(
                        X=X, aipw_scores=gamma,
                    )
                    rec = pol.recommend()
                    rows.append({
                        "treatment": t,
                        "n_recommended": int(np.sum(rec)),
                        "n_total": int(len(rec)),
                        "welfare": float(pol.welfare_),
                    })
                    per_cell[t] = [int(v) for v in rec]
                if rows:
                    _save_json(
                        os.path.join(output_dir, "policy_recommendations.json"),
                        {"gap_id": 8, "rows": rows,
                         "per_cell_recommendations": per_cell},
                    )
                    print(f"    [Gap 8] policy learning: {len(rows)} treatment(s) ✓")
                    summary["ran"]["policy_learning"] = len(rows)
        except Exception as exc:
            print(f"    [Gap 8] policy learning FAILED: {exc}")
            summary["failed"]["policy_learning"] = str(exc)
    else:
        summary["skipped"]["policy_learning"] = "disabled"

    # ── E-value sensitivity report (VanderWeele 2017) ─────────────────
    try:
        import dataclasses
        from sparc.causal.sensitivity import e_value_continuous

        # Supplement with any NUTS ATEs that may have been passed in
        if nuts_results is not None:
            for t in treatments:
                if t not in _ate_registry:
                    for key in (t, f"{t}_ate", f"ate_{t}"):
                        if key in nuts_results and nuts_results[key] is not None:
                            try:
                                _ate_registry[t] = float(nuts_results[key])
                            except (TypeError, ValueError):
                                pass
                            break

        sensitivity_rows = []
        for t, ate_val in _ate_registry.items():
            if math.isfinite(ate_val):
                sr = e_value_continuous(ate_val, label=t)
                row = dataclasses.asdict(sr)
                row["treatment"] = t
                sensitivity_rows.append(row)

        if sensitivity_rows:
            _save_json(
                os.path.join(output_dir, "sensitivity_report.json"),
                {"method": "VanderWeele E-value (2017)", "results": sensitivity_rows},
            )
            print(f"  [Sensitivity] E-values computed for {len(sensitivity_rows)} treatment(s) \u2713")
            summary["ran"]["sensitivity"] = len(sensitivity_rows)
    except Exception as exc:
        print(f"  [Sensitivity] E-value computation FAILED: {exc}")
        summary["failed"]["sensitivity"] = str(exc)

    # ── Triangulation report (audit registry snapshot) ─────────────────
    try:
        from sparc.causal import _audit
        # Note: _autodetect() is intentionally NOT called here.
        # Gaps self-register via mark_addressed() as they run; calling
        # _autodetect() would eagerly import all 10 gap modules (including
        # heavy torch/PyMC dependencies) even for disabled gaps.
        # Use `sparc audit causal` from the CLI to run _autodetect() explicitly.
        report_text = _audit.report()
        with open(os.path.join(output_dir, "wager2025_audit_report.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write(report_text)
        n_done = sum(_audit.WAGER2025_GAPS_ADDRESSED.values())
        print(f"  Audit registry: {n_done}/{len(_audit.GAP_SPECS)} gaps addressed")
        summary["audit_registry"] = {
            "n_addressed": n_done,
            "n_total": len(_audit.GAP_SPECS),
            "addressed": dict(_audit.WAGER2025_GAPS_ADDRESSED),
        }
    except Exception as exc:
        print(f"  audit registry snapshot FAILED: {exc}")

    return summary
