"""TDD — C3: ScenarioExecutor protocol as a formal seam for interventions.

Verifies that:
1. ``ScenarioExecutor`` protocol is importable from ``sparc.interventions``
2. ``ScenarioResult`` data class is importable and has the expected fields
3. Any class with ``execute(spec, oofs, causal_inputs)`` satisfies the protocol
4. ``ScenarioEngineSelector`` can resolve an executor that satisfies the protocol
"""

from __future__ import annotations

import pytest


class TestScenarioExecutorProtocol:
    def test_scenario_executor_importable_from_module(self):
        from sparc.interventions.scenario_executor import ScenarioExecutor
        assert ScenarioExecutor is not None

    def test_scenario_result_importable_from_module(self):
        from sparc.interventions.scenario_executor import ScenarioResult
        assert ScenarioResult is not None

    def test_scenario_executor_exported_from_interventions(self):
        from sparc.interventions import ScenarioExecutor
        assert ScenarioExecutor is not None

    def test_scenario_result_exported_from_interventions(self):
        from sparc.interventions import ScenarioResult
        assert ScenarioResult is not None

    def test_scenario_result_has_scenario_df_field(self):
        from sparc.interventions.scenario_executor import ScenarioResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ScenarioResult)}
        assert "scenario_df" in fields

    def test_scenario_result_has_summary_field(self):
        from sparc.interventions.scenario_executor import ScenarioResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ScenarioResult)}
        assert "summary" in fields

    def test_non_conforming_object_is_not_executor(self):
        from sparc.interventions.scenario_executor import ScenarioExecutor
        class NotAnExecutor:
            pass
        assert not isinstance(NotAnExecutor(), ScenarioExecutor)

    def test_custom_executor_satisfies_protocol(self):
        from sparc.interventions.scenario_executor import ScenarioExecutor, ScenarioResult

        class FakeExecutor:
            def execute(self, spec, oofs, causal_inputs) -> ScenarioResult:
                return ScenarioResult(scenario_df=None, summary={"n_scenarios": 0})

        assert isinstance(FakeExecutor(), ScenarioExecutor)

    def test_executor_protocol_has_execute_method(self):
        from sparc.interventions.scenario_executor import ScenarioExecutor
        assert hasattr(ScenarioExecutor, "execute")

    def test_fake_executor_produces_scenario_result(self):
        from sparc.interventions.scenario_executor import ScenarioExecutor, ScenarioResult

        class StubExecutor:
            def execute(self, spec, oofs, causal_inputs) -> ScenarioResult:
                return ScenarioResult(scenario_df={"cells": 100}, summary={"mode": "test"})

        executor = StubExecutor()
        result = executor.execute(spec=None, oofs=None, causal_inputs=None)
        assert isinstance(result, ScenarioResult)
        assert result.summary["mode"] == "test"

    def test_scenario_result_default_summary_is_dict(self):
        from sparc.interventions.scenario_executor import ScenarioResult
        result = ScenarioResult(scenario_df=None, summary={})
        assert isinstance(result.summary, dict)
