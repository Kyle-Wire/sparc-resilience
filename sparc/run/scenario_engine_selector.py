"""ScenarioEngineSelector — Stage-4 mode resolution and engine dispatch.

Extracts the mode-resolution, alias-translation, and engine-dispatch logic
from ``sparc.__main__._run_scenarios`` into a single, testable seam.

Usage::

    selector = ScenarioEngineSelector(config, sim, data)
    mode = selector.resolve_mode(
        requested=config["pipeline"].get("scenario_mode", "auto"),
        has_dag=has_dag,
        force_full_audit=getattr(args, "full_audit", False),
    )
    summary_df, results_gdf = selector.run(
        mode, has_dag=has_dag, force_legacy=force_legacy
    )

Downgrade policy
----------------
Only ``MissingArtifactsError`` from ``ScenarioEngineV4`` triggers a silent
downgrade to the legacy path.  All other exceptions propagate so that real
v4 bugs are not hidden.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

__all__ = ["ScenarioEngineSelector"]

# Map legacy mode strings → canonical v4 keys.
_LEGACY_ALIASES: dict[str, str] = {
    "physics":            "mode_1_physics",
    "dag_coefficient":    "mode_2_dag_local",
    "model_reprediction": "mode_3_full_ensemble",
    "hybrid":             "mode_4_hybrid",
}

# Modes that require v4-only artifacts; not dispatchable by legacy engine.
_V4_ONLY_MODES = frozenset({"mode_5_full_audit"})

_VALID_MODES = frozenset({
    "auto",
    "mode_1_physics",
    "mode_2_dag_local",
    "mode_3_full_ensemble",
    "mode_4_hybrid",
    "mode_5_full_audit",
})


class ScenarioEngineSelector:
    """Stateful helper that drives Stage-4 mode resolution and dispatch.

    Parameters
    ----------
    config:
        Full pipeline config from ``load_config()``.
    sim:
        A loaded ``ScenarioSimulator`` instance (base models already loaded).
    data:
        Baseline DataFrame for the current project.
    """

    def __init__(
        self,
        config: dict,
        sim,  # ScenarioSimulator — avoid circular import
        data: pd.DataFrame,
    ) -> None:
        self._config = config
        self._sim = sim
        self._data = data

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def resolve_mode(
        self,
        requested: str,
        has_dag: bool,
        *,
        force_full_audit: bool = False,
    ) -> str:
        """Return the concrete scenario mode to use.

        Handles 'auto' introspection, legacy alias translation, and the
        ``--full-audit`` override.

        Raises
        ------
        ValueError
            For unknown mode strings.
        RuntimeError
            For the removed ``'bayesian'`` mode.
        """
        if requested == "bayesian":
            raise RuntimeError(
                "scenario_mode='bayesian' was removed in SPARC v4. "
                "Use 'mode_3_full_ensemble' or 'auto' — credible intervals are "
                "now native to all modes via the per-edge NUTS posterior + "
                "Bayesian Spatial CATE."
            )

        mode = _LEGACY_ALIASES.get(requested, requested)
        if mode != requested:
            warnings.warn(
                f"scenario_mode='{requested}' is deprecated; "
                f"use '{mode}' instead. (Auto-translated for this run.)",
                DeprecationWarning,
                stacklevel=2,
            )

        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown scenario_mode '{mode}'. "
                f"Valid: {', '.join(sorted(_VALID_MODES))}."
            )

        if mode == "auto":
            from sparc.__main__ import _resolve_auto_scenario_mode
            mode = _resolve_auto_scenario_mode(has_dag=has_dag)
            print(f"  [auto] scenario_mode resolved to '{mode}'")

        if force_full_audit and mode != "mode_5_full_audit":
            print(f"  [--full-audit] overriding '{mode}' → 'mode_5_full_audit'")
            mode = "mode_5_full_audit"

        return mode

    # ------------------------------------------------------------------
    # Engine dispatch
    # ------------------------------------------------------------------

    def run(
        self,
        mode: str,
        *,
        has_dag: bool,
        force_legacy: bool = False,
    ) -> Tuple[pd.DataFrame, object]:
        """Run the scenario and return ``(summary_df, results_gdf)``.

        Tries the v4 unified engine first unless *force_legacy* is True
        or the mode is legacy-only.  Falls back to legacy **only** when
        the v4 engine raises ``MissingArtifactsError``; all other
        exceptions propagate.
        """
        if not force_legacy:
            result = self._try_v4(mode, has_dag=has_dag)
            if result is not None:
                summary_df, results_gdf = result
                print(f"  [v4 engine] mode={mode}: {len(summary_df)} summary rows")
                return summary_df, results_gdf

        if force_legacy:
            print("  [--legacy] forcing legacy scenario path")

        return self._run_legacy(mode, has_dag=has_dag)

    # ------------------------------------------------------------------
    # V4 engine attempt
    # ------------------------------------------------------------------

    def _try_v4(
        self,
        mode: str,
        *,
        has_dag: bool,
    ) -> Optional[Tuple[pd.DataFrame, object]]:
        """Try the v4 engine; return ``None`` only on ``MissingArtifactsError``.

        Unlike the previous implementation, exceptions other than
        ``MissingArtifactsError`` are not swallowed — they propagate so
        that real v4 bugs surface immediately.
        """
        try:
            from sparc.interventions.scenario_engine_v4 import (
                ScenarioEngineV4,
                MissingArtifactsError,
            )
        except ImportError:
            return None

        dag = self._load_dag_for_mode(mode, has_dag=has_dag)
        if dag is False:  # DAG required but unavailable/failed
            return None

        ensemble_pred: Optional[object] = None
        if mode in ("mode_3_full_ensemble", "mode_4_hybrid"):
            ensemble_pred = _build_ensemble_predictor(self._sim)
            if ensemble_pred is None:
                return None  # no usable base ensemble → skip v4

        try:
            engine = ScenarioEngineV4(
                self._config,
                mode=mode,
                dag=dag,
                ensemble_predictor=ensemble_pred,
            )
        except MissingArtifactsError as exc:
            print(f"  [v4 engine] {exc} — downgrading to legacy path")
            return None
        # Other exceptions propagate intentionally.

        return engine.run(self._data, verbose=True)

    def _load_dag_for_mode(
        self,
        mode: str,
        *,
        has_dag: bool,
    ):
        """Load the DAG for modes that require it.

        Returns the DAG graph, ``None`` (not required), or ``False``
        (required but could not be loaded → skip v4).
        """
        if mode not in ("mode_2_dag_local", "mode_4_hybrid", "mode_5_full_audit"):
            return None  # not required

        if not has_dag:
            return False  # required but absent

        try:
            from sparc.causal.dag_definition import load_dag, dag_to_networkx
            dag_file = self._config["causal"]["dag_file"]
            return dag_to_networkx(load_dag(dag_file))
        except Exception as exc:
            print(f"  [v4 engine] DAG load failed ({exc}); using legacy path")
            return False

    # ------------------------------------------------------------------
    # Legacy engine dispatch
    # ------------------------------------------------------------------

    def _run_legacy(
        self,
        mode: str,
        *,
        has_dag: bool,
    ) -> Tuple[pd.DataFrame, object]:
        """Dispatch to the appropriate legacy ``ScenarioSimulator`` method."""
        sim = self._sim
        data = self._data

        if mode == "mode_4_hybrid":
            print("  [legacy] mode_4_hybrid: ensemble direct + DAG/NUTS indirect")
            return sim.run_with_hybrid_reprediction(data, verbose=True)

        if mode == "mode_3_full_ensemble":
            print("  [legacy] mode_3_full_ensemble: resolver + base-model reprediction blend")
            return sim.run_with_model_reprediction(data, verbose=True)

        if mode == "mode_2_dag_local":
            if has_dag:
                print("  [legacy] mode_2_dag_local: resolver + DAG + per-edge NUTS")
                return sim.run_with_causal_dag(data, verbose=True)
            print("  [legacy] mode_2_dag_local requested but no DAG — falling back to mode_1_physics")
            return sim.run(verbose=True)

        if mode == "mode_1_physics":
            print("  [legacy] mode_1_physics: resolver with literature priors")
            return sim.run(verbose=True)

        if mode in _V4_ONLY_MODES:
            raise RuntimeError(
                f"scenario_mode='{mode}' requires the v4 engine which "
                "could not be initialised.  Ensure Stage 2 / Stage 3 "
                "artifacts are present."
            )

        raise ValueError(
            f"No legacy handler for scenario_mode='{mode}'."
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _build_ensemble_predictor(sim) -> Optional[object]:
    """Wrap loaded base models into a ``df → ndarray`` callable.

    Returns ``None`` when *sim* has no usable base ensemble.
    """
    if not getattr(sim, "_models", None) or not getattr(sim, "_meta_model", None):
        return None
    try:
        import numpy as _np

        def _predict(df):
            base_pred, *_ = sim._predict_baseline(df, verbose=False)
            return _np.asarray(base_pred, dtype=_np.float64).reshape(-1)

        return _predict
    except Exception:
        return None
