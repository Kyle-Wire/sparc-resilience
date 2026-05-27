"""ScenarioExecutor protocol — the formal seam for intervention scenario execution.

Defines a structural protocol that both :class:`ScenarioEngineV4` and
:class:`ScenarioSimulator` can satisfy, enabling callers to depend on the
abstract ``ScenarioExecutor`` interface instead of a concrete engine class.

Usage
-----
::

    from sparc.interventions.scenario_executor import ScenarioExecutor, ScenarioResult

    def run_scenarios(executor: ScenarioExecutor, spec, oofs, causal) -> ScenarioResult:
        return executor.execute(spec, oofs, causal)

    # In tests, pass a FakeExecutor; in production, pass a ScenarioEngineV4 adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ScenarioResult:
    """Output of a single scenario execution pass.

    Attributes
    ----------
    scenario_df:
        Per-cell intervention effect estimates (typically a ``pd.DataFrame``
        or ``gpd.GeoDataFrame``). May be ``None`` for dry-run or stub executors.
    summary:
        Scalar / aggregate statistics for the run — number of scenarios,
        effect ranges, budget consumed, etc.
    """

    scenario_df: Any
    summary: dict = field(default_factory=dict)


@runtime_checkable
class ScenarioExecutor(Protocol):
    """Structural protocol for scenario execution engines.

    Any object with an ``execute`` method that accepts ``(spec, oofs, causal_inputs)``
    and returns a :class:`ScenarioResult` satisfies this protocol.  Both
    ``ScenarioEngineV4`` and the legacy ``ScenarioSimulator`` can be wrapped to
    satisfy it.

    Parameters to ``execute``
    -------------------------
    spec:
        Scenario specification (dict, ``ScenarioSpec``, or equivalent).
    oofs:
        Out-of-fold spatial intelligence predictions.
    causal_inputs:
        :class:`~sparc.interventions.causal_inputs.CausalInputs` instance
        providing calibrated physics priors.
    """

    def execute(self, spec: Any, oofs: Any, causal_inputs: Any) -> ScenarioResult: ...
