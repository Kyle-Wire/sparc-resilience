"""DAG management routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from sparc.server import deps

router = APIRouter(tags=["dag"])


@router.get("/dag")
async def get_dag():
    """Return the project's DAG definition."""
    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    dag_file = state.project_config.get("causal", {}).get("dag_file")
    if not dag_file or not Path(dag_file).exists():
        return {"nodes": [], "edges": []}

    from sparc.causal.dag_definition import load_dag

    return load_dag(dag_file)


@router.put("/dag")
async def save_dag(dag: dict):
    """Validate and persist an edited DAG definition back to the project dag_file."""
    import networkx as nx
    import yaml as _yaml

    from sparc.causal.dag_definition import dag_to_networkx

    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    dag_file = state.project_config.get("causal", {}).get("dag_file")
    if not dag_file:
        raise HTTPException(400, "No dag_file configured in project")

    try:
        G = dag_to_networkx(dag)
    except Exception as exc:
        raise HTTPException(422, f"Invalid DAG: {exc}")

    if not nx.is_directed_acyclic_graph(G):
        raise HTTPException(422, "DAG contains cycles")

    p = Path(dag_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        _yaml.safe_dump(dag, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return {
        "status": "saved",
        "n_nodes": len(dag.get("nodes", [])),
        "n_edges": len(dag.get("edges", [])),
    }


@router.post("/dag/validate")
async def validate_dag(dag: dict):
    """Validate a DAG definition without saving it."""
    import networkx as nx

    from sparc.causal.dag_definition import dag_to_networkx

    try:
        G = dag_to_networkx(dag)
        is_dag = nx.is_directed_acyclic_graph(G)
        return {
            "valid": is_dag,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "error": None if is_dag else "Graph contains cycles",
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@router.get("/dag/mc3_result")
async def get_mc3_result():
    """Return pending MC³ edge-inclusion probabilities for DAG approval."""
    state = deps.state
    if state.pending_mc3 is None:
        raise HTTPException(404, "No MC³ result pending")
    return state.pending_mc3


@router.post("/dag/approve")
async def approve_dag():
    """Approve the discovered DAG and unblock the pipeline."""
    state = deps.state
    if state.pending_mc3 is None:
        raise HTTPException(400, "No MC³ result pending approval")
    state.dag_approved.set()
    return {"status": "approved"}


@router.post("/dag/reject")
async def reject_dag():
    """Reject the discovered DAG and cancel the pipeline."""
    state = deps.state
    if state.pending_mc3 is None:
        raise HTTPException(400, "No MC³ result pending approval")
    state.pending_mc3 = None
    state.dag_approved.set()
    return {"status": "rejected"}


@router.post("/dag/suggest-edges")
async def post_dag_suggest_edges(payload: dict = Body(default={})):
    """Suggest plausible causal edges based on partial-correlation analysis.

    Uses the project data to rank variable pairs by partial correlation
    (|r| > threshold), filtered by known domain priors from physics.yml.
    Returns ordered list of (parent, child, score, reason) suggestions.

    Body (all optional)::

        {
          "threshold": 0.15,      // minimum |partial_r| to include
          "max_suggestions": 10   // cap on output list length
        }
    """
    state = deps.state
    if state.project_config is None:
        raise HTTPException(400, "No project loaded")

    threshold = float(payload.get("threshold", 0.15))
    max_suggestions = int(payload.get("max_suggestions", 10))

    try:
        import numpy as _np
        import pandas as _pd

        from sparc.run.pipeline_paths import PipelinePaths

        _paths = PipelinePaths.from_config(state.project_config)

        _data_path = _paths.output_dir / "merged_data.parquet"
        if not _data_path.exists():
            _data_path = _paths.output_dir / "merged_data.csv"
        if not _data_path.exists():
            _data_path = _paths.stage2_dir / "merged_data.parquet"

        if not _data_path.exists():
            raise HTTPException(424, "Merged data file not found — run Stage 0 first")

        _df = (
            _pd.read_parquet(str(_data_path))
            if str(_data_path).endswith(".parquet")
            else _pd.read_csv(str(_data_path))
        )

        _predictors = state.project_config.get("predictors", [])
        _outcome = (state.project_config.get("outcomes") or ["target"])[0]
        _cols = [c for c in _predictors + [_outcome] if c in _df.columns]
        _df_num = _df[_cols].select_dtypes(include=["number"]).dropna()

        if _df_num.shape[1] < 2:
            return {"suggestions": [], "message": "Insufficient numeric columns for correlation analysis"}

        _corr = _df_num.corr().values
        _col_names = list(_df_num.columns)
        _p = len(_col_names)

        try:
            _precision = _np.linalg.inv(_corr)
        except _np.linalg.LinAlgError:
            _precision = _np.linalg.pinv(_corr)

        _diag = _np.diag(_precision)
        _partial_r = _np.zeros((_p, _p))
        for _i in range(_p):
            for _j in range(_p):
                if _i == _j:
                    continue
                _denom = _np.sqrt(abs(_diag[_i]) * abs(_diag[_j]))
                _partial_r[_i, _j] = -_precision[_i, _j] / _denom if _denom > 0 else 0.0

        _existing_edges: set = set()
        _dag_path = _paths.output_dir / "dag.yml"
        try:
            import yaml as _yaml

            if _dag_path.exists():
                with open(_dag_path) as _df2:
                    _dag_data = _yaml.safe_load(_df2) or {}
                for _e in _dag_data.get("edges", []):
                    _existing_edges.add((_e.get("parent"), _e.get("child")))
        except Exception:
            pass

        _physics_priors: dict = {}
        try:
            from sparc.config.config import load_physics_priors

            _pp = load_physics_priors(state.project_config)
            _physics_priors = _pp.get("coefficients", {})
        except Exception:
            pass

        _candidates = []
        _outcome_idx = _col_names.index(_outcome) if _outcome in _col_names else -1
        for _i in range(_p):
            for _j in range(_p):
                if _i == _j:
                    continue
                _r = float(_partial_r[_i, _j])
                if abs(_r) < threshold:
                    continue
                _parent = _col_names[_i]
                _child = _col_names[_j]
                if (_parent, _child) in _existing_edges:
                    continue
                _toward_outcome = _j == _outcome_idx
                _prior_known = _parent in _physics_priors and _child in _physics_priors
                _reason = (
                    "strong partial correlation toward outcome"
                    if _toward_outcome
                    else "both variables have physics priors"
                    if _prior_known
                    else "partial correlation above threshold"
                )
                _candidates.append({
                    "parent": _parent,
                    "child": _child,
                    "score": round(abs(_r), 4),
                    "partial_r": round(_r, 4),
                    "toward_outcome": _toward_outcome,
                    "reason": _reason,
                })

        _candidates.sort(key=lambda x: (-int(x["toward_outcome"]), -x["score"]))
        _candidates = _candidates[:max_suggestions]

        return {
            "suggestions": _candidates,
            "n_columns_analysed": _p,
            "threshold": threshold,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Edge suggestion failed: {exc}") from exc
