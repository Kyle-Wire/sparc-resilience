"""Pairwise diff utilities for SPARC pipeline runs.

Operates on already-written artifacts on disk so users can compare
*any* two output directories without re-running anything. Produces
structured JSON suitable for side-by-side rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists to dotted-path keys."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        # Only dive into lists of primitives or dicts; encode index in the path.
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


# ----------------------------------------------------------------------
# Specific differs
# ----------------------------------------------------------------------

def diff_configs(cfg_a: Optional[dict], cfg_b: Optional[dict]) -> list[dict]:
    """Return a list of changes between two config dicts."""
    if not cfg_a and not cfg_b:
        return []
    flat_a = _flatten(cfg_a or {})
    flat_b = _flatten(cfg_b or {})
    keys = sorted(set(flat_a) | set(flat_b))
    rows: list[dict] = []
    for k in keys:
        a = flat_a.get(k)
        b = flat_b.get(k)
        if a == b:
            continue
        rows.append({
            "path": k,
            "a": a,
            "b": b,
            "kind": "added" if a is None else ("removed" if b is None else "changed"),
        })
    return rows


def diff_models(perf_a: Optional[dict], perf_b: Optional[dict]) -> list[dict]:
    """Diff per-model performance metrics (R², RMSE, MAE).

    Accepts either ``{model_name: {r2, rmse, mae}}`` or
    ``{"models": [{"name", "r2", ...}, ...]}``.
    """
    def _norm(p: Optional[dict]) -> dict[str, dict]:
        if not p:
            return {}
        if isinstance(p.get("models"), list):
            return {str(m.get("name", "?")): m for m in p["models"] if isinstance(m, dict)}
        return {k: v for k, v in p.items() if isinstance(v, dict)}

    a = _norm(perf_a)
    b = _norm(perf_b)
    rows: list[dict] = []
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name) or {}, b.get(name) or {}
        rows.append({
            "model": name,
            "r2_a": ra.get("r2") or ra.get("R2"),
            "r2_b": rb.get("r2") or rb.get("R2"),
            "rmse_a": ra.get("rmse") or ra.get("RMSE"),
            "rmse_b": rb.get("rmse") or rb.get("RMSE"),
            "mae_a": ra.get("mae") or ra.get("MAE"),
            "mae_b": rb.get("mae") or rb.get("MAE"),
        })
    # Compute deltas where both are numeric.
    for r in rows:
        for metric in ("r2", "rmse", "mae"):
            a_v, b_v = r[f"{metric}_a"], r[f"{metric}_b"]
            try:
                r[f"{metric}_delta"] = float(b_v) - float(a_v) if (a_v is not None and b_v is not None) else None
            except (TypeError, ValueError):
                r[f"{metric}_delta"] = None
    return rows


def diff_causal(payload_a: Optional[dict], payload_b: Optional[dict]) -> list[dict]:
    """Diff causal effect estimates (direct/indirect/total or list[effects])."""
    def _extract(p: Optional[dict]) -> dict[str, dict]:
        if not p:
            return {}
        out: dict[str, dict] = {}
        # Shape A: list of effects.
        if isinstance(p.get("effects"), list):
            for e in p["effects"]:
                if not isinstance(e, dict):
                    continue
                name = str(e.get("name") or e.get("label") or "effect")
                out[name] = {
                    "estimate": e.get("estimate") or e.get("mean") or e.get("effect"),
                    "ci_low": e.get("ci_low") or e.get("ci_lower"),
                    "ci_high": e.get("ci_high") or e.get("ci_upper"),
                }
        # Shape B: scalar or dict for direct/indirect/total.
        for key in ("direct_effect", "indirect_effect", "total_effect"):
            v = p.get(key)
            if isinstance(v, dict):
                out[key] = {
                    "estimate": v.get("mean") or v.get("estimate"),
                    "ci_low": v.get("ci_low") or v.get("ci_lower"),
                    "ci_high": v.get("ci_high") or v.get("ci_upper"),
                }
            elif isinstance(v, (int, float)):
                out[key] = {"estimate": float(v), "ci_low": None, "ci_high": None}
        return out

    a = _extract(payload_a)
    b = _extract(payload_b)
    rows: list[dict] = []
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name) or {}, b.get(name) or {}
        try:
            delta = float(rb.get("estimate")) - float(ra.get("estimate")) \
                if ra.get("estimate") is not None and rb.get("estimate") is not None else None
        except (TypeError, ValueError):
            delta = None
        rows.append({
            "effect": name,
            "estimate_a": ra.get("estimate"),
            "estimate_b": rb.get("estimate"),
            "ci_a": [ra.get("ci_low"), ra.get("ci_high")],
            "ci_b": [rb.get("ci_low"), rb.get("ci_high")],
            "delta": delta,
        })
    return rows


def diff_dag(dag_a: Optional[dict], dag_b: Optional[dict]) -> dict:
    """Edge-level diff of two DAG payloads. Accepts ``{edges: [{from, to, ...}]}``."""
    def _edges(d: Optional[dict]) -> dict[tuple[str, str], dict]:
        if not d or not isinstance(d.get("edges"), list):
            return {}
        out: dict[tuple[str, str], dict] = {}
        for e in d["edges"]:
            if not isinstance(e, dict):
                continue
            src = str(e.get("from") or e.get("source") or "")
            dst = str(e.get("to") or e.get("target") or "")
            if not src or not dst:
                continue
            out[(src, dst)] = e
        return out

    a = _edges(dag_a)
    b = _edges(dag_b)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    common = sorted(set(a) & set(b))
    changed: list[dict] = []
    for k in common:
        ea, eb = a[k], b[k]
        prob_a = ea.get("probability") or ea.get("prob")
        prob_b = eb.get("probability") or eb.get("prob")
        try:
            if prob_a is not None and prob_b is not None and abs(float(prob_a) - float(prob_b)) > 0.05:
                changed.append({"from": k[0], "to": k[1], "prob_a": prob_a, "prob_b": prob_b,
                                "delta": float(prob_b) - float(prob_a)})
        except (TypeError, ValueError):
            pass

    return {
        "added":   [{"from": s, "to": d, **(b[(s, d)])} for s, d in added],
        "removed": [{"from": s, "to": d, **(a[(s, d)])} for s, d in removed],
        "changed": changed,
        "n_a": len(a),
        "n_b": len(b),
    }


# ----------------------------------------------------------------------
# Top-level: diff two run directories
# ----------------------------------------------------------------------

def _resolve_artifact(run_dir: Path, *candidates: str) -> Optional[Path]:
    for c in candidates:
        p = run_dir / c
        if p.exists():
            return p
    return None


def diff_runs(run_a: Path, run_b: Path) -> dict:
    """Top-level entry point — load known artifacts and diff them."""
    run_a = Path(run_a)
    run_b = Path(run_b)
    if not run_a.exists() or not run_b.exists():
        raise FileNotFoundError(f"One or both run directories do not exist: {run_a} / {run_b}")

    # Config: try project_config.json or config.yaml location.
    cfg_a_path = _resolve_artifact(run_a, "project_config.json", "config.json")
    cfg_b_path = _resolve_artifact(run_b, "project_config.json", "config.json")

    # Causal payload: scenario_coefficients.json or causal_diagnostics.json from Stage_3.
    causal_a = _resolve_artifact(
        run_a / "Stage_3_Causal_Validation",
        "scenario_coefficients.json", "causal_diagnostics.json", "causal_results.json",
    )
    causal_b = _resolve_artifact(
        run_b / "Stage_3_Causal_Validation",
        "scenario_coefficients.json", "causal_diagnostics.json", "causal_results.json",
    )

    # Model performance: try a few common names.
    perf_a = _resolve_artifact(
        run_a / "Stage_2_Spatial_CV",
        "model_performance.json", "performance.json",
    ) or _resolve_artifact(run_a, "model_performance.json")
    perf_b = _resolve_artifact(
        run_b / "Stage_2_Spatial_CV",
        "model_performance.json", "performance.json",
    ) or _resolve_artifact(run_b, "model_performance.json")

    # DAG: dag.json/yaml or mc3_results.json.
    dag_a = _resolve_artifact(run_a / "Stage_3_Causal_Validation", "mc3_results.json") \
            or _resolve_artifact(run_a, "dag.json")
    dag_b = _resolve_artifact(run_b / "Stage_3_Causal_Validation", "mc3_results.json") \
            or _resolve_artifact(run_b, "dag.json")

    return {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "config":   {
            "found_a": bool(cfg_a_path),
            "found_b": bool(cfg_b_path),
            "changes": diff_configs(_load_json(cfg_a_path) if cfg_a_path else None,
                                    _load_json(cfg_b_path) if cfg_b_path else None),
        },
        "causal": {
            "found_a": bool(causal_a),
            "found_b": bool(causal_b),
            "effects": diff_causal(_load_json(causal_a) if causal_a else None,
                                   _load_json(causal_b) if causal_b else None),
        },
        "models": {
            "found_a": bool(perf_a),
            "found_b": bool(perf_b),
            "rows": diff_models(_load_json(perf_a) if perf_a else None,
                                _load_json(perf_b) if perf_b else None),
        },
        "dag": {
            "found_a": bool(dag_a),
            "found_b": bool(dag_b),
            **diff_dag(_load_json(dag_a) if dag_a else None,
                       _load_json(dag_b) if dag_b else None),
        },
    }


def discover_runs(search_root: Path, *, max_depth: int = 4) -> list[dict]:
    """Find candidate SPARC output directories under *search_root*.

    A directory is considered a "run" if it contains at least one of the
    canonical stage subdirectories. Returns each match with mtime so the
    UI can sort by recency.
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return []

    markers = (
        "Stage_3_Causal_Validation",
        "Stage_2_Spatial_CV",
        "Stage_4_Scenarios",
        "Final_Interpretation_Results",
    )
    found: list[dict] = []
    seen: set[Path] = set()

    def _scan(d: Path, depth: int) -> None:
        if depth > max_depth or d in seen:
            return
        seen.add(d)
        try:
            children = list(d.iterdir())
        except (OSError, PermissionError):
            return
        if any((d / m).exists() for m in markers):
            try:
                mtime = d.stat().st_mtime
            except OSError:
                mtime = 0.0
            found.append({"path": str(d), "name": d.name, "mtime": mtime})
            return  # don't recurse below a matched run
        for ch in children:
            if ch.is_dir() and not ch.name.startswith((".", "__")):
                _scan(ch, depth + 1)

    _scan(search_root, 0)
    found.sort(key=lambda r: r["mtime"], reverse=True)
    return found
