"""Regression tests for the Stage-4 dispatch crash chain (A29 / A30).

* A29 — ``sparc.__main__._run_scenarios`` referenced an out-of-scope
  ``args`` → ``NameError`` whenever ``sparc run`` reached Stage 4.
* ``cmd_scenario --scenario NAME`` filtered the scenario list but never
  handed it to the engine.
* A30 — ``ScenarioEngineSelector._try_v4`` only built the ensemble predictor
  for mode_3/mode_4, while the v4 engine requires it for mode_5 as well; and
  ``_build_ensemble_predictor`` refused to build without a meta model.
"""
from __future__ import annotations

import inspect
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sparc.__main__ as cli
import sparc.run.stage4_runner as stage4_runner
import sparc.run.scenario_engine_selector as selector_mod


# ---------------------------------------------------------------------------
# A29 — _run_scenarios delegates to stage4_runner, no ``args`` reference
# ---------------------------------------------------------------------------

def test_run_scenarios_has_no_free_args_reference():
    code = cli._run_scenarios.__code__
    # ``args`` as a global/free name is exactly what raised NameError.
    assert "args" not in code.co_names
    assert "args" not in code.co_freevars
    assert "_run_scenario_engine" in inspect.getsource(cli._run_scenarios)


def test_dead_v4_helpers_removed_but_auto_resolver_kept():
    assert not hasattr(cli, "_try_run_with_v4_engine")
    assert not hasattr(cli, "_build_v4_ensemble_predictor")
    assert callable(cli._resolve_auto_scenario_mode)


def _record_engine(monkeypatch):
    calls = []

    def _fake_engine(config, paths, **kwargs):
        calls.append({
            "config": config,
            "paths": paths,
            "kwargs": kwargs,
            "force_full_audit": config.get("_force_full_audit"),
            "scenarios": list(config.get("scenarios", [])),
        })
        return pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]})

    monkeypatch.setattr(stage4_runner, "_run_scenario_engine", _fake_engine)
    return calls


@pytest.mark.parametrize("flag", [True, False])
def test_run_scenarios_propagates_force_full_audit(monkeypatch, flag):
    calls = _record_engine(monkeypatch)
    config = {"scenarios": [{"name": "a"}], "_force_full_audit": flag}
    paths = object()

    summary, results = cli._run_scenarios(config, paths, "project.yml")

    assert len(calls) == 1
    assert calls[0]["paths"] is paths
    assert calls[0]["force_full_audit"] is flag
    assert list(summary.columns) == ["x"]
    assert list(results.columns) == ["y"]
    # Post-processing (conservation / MC / SA / Wager add-ons) must stay on.
    assert not calls[0]["kwargs"].get("fast_mode", False)


def _patch_cmd_scenario_env(monkeypatch, tmp_path, config):
    import sparc.config.config as config_mod
    import sparc.run.pipeline_paths as pp

    monkeypatch.setattr(config_mod, "load_config", lambda _p: config)
    monkeypatch.setattr(pp, "set_paths_from_config", lambda _c: SimpleNamespace(output_dir=tmp_path))
    monkeypatch.setattr(cli, "_resolve_project_path", lambda _a: str(tmp_path / "project.yml"))
    monkeypatch.delenv("SPARC_PROJECT", raising=False)


def test_cmd_scenario_applies_name_filter_and_full_audit(monkeypatch, tmp_path):
    config = {
        "scenarios": [
            {"name": "trees", "variable": "canopy", "increments": [0.1]},
            {"name": "roofs", "variable": "albedo", "increments": [0.2]},
        ],
    }
    _patch_cmd_scenario_env(monkeypatch, tmp_path, config)
    calls = _record_engine(monkeypatch)

    cli.cmd_scenario(Namespace(scenario="roofs", legacy=False, full_audit=True))

    assert len(calls) == 1
    assert [s["name"] for s in calls[0]["scenarios"]] == ["roofs"]
    assert calls[0]["force_full_audit"] is True


def test_cmd_scenario_without_filter_runs_all(monkeypatch, tmp_path):
    config = {
        "scenarios": [
            {"name": "trees", "variable": "canopy", "increments": [0.1]},
            {"name": "roofs", "variable": "albedo", "increments": [0.2]},
        ],
    }
    _patch_cmd_scenario_env(monkeypatch, tmp_path, config)
    calls = _record_engine(monkeypatch)

    # The ``scenario`` sub-parser has no --full-audit flag → defaults False.
    cli.cmd_scenario(Namespace(scenario=None, legacy=True))

    assert [s["name"] for s in calls[0]["scenarios"]] == ["trees", "roofs"]
    assert calls[0]["force_full_audit"] is False
    assert calls[0]["config"]["_force_legacy_scenarios"] is True


# ---------------------------------------------------------------------------
# A30 — ensemble predictor for mode_5 and without a meta model
# ---------------------------------------------------------------------------

class _StubSim:
    """Minimal stand-in exposing what ``_build_ensemble_predictor`` needs."""

    def __init__(self, meta_model=None):
        self._models = {"ols": object(), "gwr": object(), "gwrf": object(), "ggpgam": object()}
        self._meta_model = meta_model
        self.baseline_calls = 0

    def _predict_baseline(self, df, verbose=False):
        self.baseline_calls += 1
        base = np.arange(len(df), dtype=float)
        return base, base, base, base, base


def test_build_ensemble_predictor_without_meta_model():
    sim = _StubSim(meta_model=None)
    pred = selector_mod._build_ensemble_predictor(sim)
    assert callable(pred)
    out = pred(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0])
    assert sim.baseline_calls == 1


def test_build_ensemble_predictor_requires_base_models():
    sim = _StubSim()
    sim._models = {}
    assert selector_mod._build_ensemble_predictor(sim) is None


def test_ensemble_modes_include_mode5():
    from sparc.interventions.scenario_engine_v4 import _HYBRID_LIKE_MODES

    for m in ("mode_3_full_ensemble",) + tuple(_HYBRID_LIKE_MODES):
        assert m in selector_mod._ENSEMBLE_MODES
    assert "mode_5_full_audit" in selector_mod._ENSEMBLE_MODES


@pytest.mark.parametrize("mode", ["mode_3_full_ensemble", "mode_4_hybrid", "mode_5_full_audit"])
def test_try_v4_passes_ensemble_predictor(monkeypatch, mode):
    import sparc.interventions.scenario_engine_v4 as v4

    captured = {}

    class _FakeEngine:
        def __init__(self, config, *, mode, dag=None, ensemble_predictor=None, **kw):
            captured["mode"] = mode
            captured["ensemble_predictor"] = ensemble_predictor
            captured["dag"] = dag

        def run(self, data, verbose=True):
            return pd.DataFrame({"s": [1]}), pd.DataFrame({"r": [1]})

    monkeypatch.setattr(v4, "ScenarioEngineV4", _FakeEngine)

    data = pd.DataFrame({"a": [1.0, 2.0]})
    sel = selector_mod.ScenarioEngineSelector({"scenarios": []}, _StubSim(), data)
    # Bypass DAG loading: pretend the DAG is loaded for modes that need it.
    monkeypatch.setattr(sel, "_load_dag_for_mode", lambda m, has_dag: "DAG" if m != "mode_3_full_ensemble" else None)

    result = sel._try_v4(mode, has_dag=True)

    assert result is not None
    assert captured["mode"] == mode
    assert callable(captured["ensemble_predictor"])
    np.testing.assert_allclose(captured["ensemble_predictor"](data), [0.0, 1.0])
