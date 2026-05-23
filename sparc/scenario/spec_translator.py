"""ScenarioSpecTranslator — converts InterventionSpec objects to SimulatorConfig.

Centralises the two private functions that app.py previously inlined:
  - _runtime_spec_to_config_scenarios
  - _runtime_specs_to_interaction_scenarios

Both callers (run_scenarios and optimize_scenario in app.py) can now delegate
here instead of duplicating the sign-convention logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterventionSpec:
    """A named scenario with a mapping of variable → signed magnitude."""

    name: str
    interventions: dict[str, float]


@dataclass
class SimulatorConfig:
    """Structured output for ScenarioSimulator configuration."""

    scenarios: list[dict] = field(default_factory=list)
    joint_scenarios: list[dict] = field(default_factory=list)


class ScenarioSpecTranslator:
    """Translate InterventionSpec objects into ScenarioSimulator config dicts.

    Usage::

        config = ScenarioSpecTranslator.to_simulator_config([
            InterventionSpec(name="Green Infra", interventions={
                "Pct_Canopy": 20.0,
                "Pct_Impervious": -15.0,
            })
        ])
        sim.configure(config.scenarios, config.joint_scenarios)
    """

    @staticmethod
    def to_simulator_config(specs: list[InterventionSpec]) -> SimulatorConfig:
        """Convert a list of InterventionSpec objects to a SimulatorConfig.

        For each spec:
        - Emits one *scenarios* entry per variable (direction derived from sign).
        - Emits one *joint_scenarios* entry if the spec has >1 variable.

        The sign convention: positive magnitude → direction="increase";
        negative magnitude → direction="decrease". The ``increments`` list and
        ``increment`` field always contain the absolute value.

        Args:
            specs: List of InterventionSpec objects to translate.

        Returns:
            SimulatorConfig with ``scenarios`` and ``joint_scenarios`` lists.
        """
        scenarios: list[dict] = []
        joint_scenarios: list[dict] = []

        for spec in specs:
            for var, magnitude in spec.interventions.items():
                direction = "decrease" if magnitude < 0 else "increase"
                increment = abs(magnitude)
                scenarios.append(
                    {
                        "name": spec.name,
                        "variable": var,
                        "direction": direction,
                        "increments": [increment],
                        "min_val": None,
                        "max_val": None,
                        "unit": "",
                    }
                )

            if len(spec.interventions) > 1:
                joint_scenarios.append(
                    {
                        "name": spec.name,
                        "interventions": [
                            {
                                "variable": var,
                                "increment": abs(mag),
                                "direction": "decrease" if mag < 0 else "increase",
                            }
                            for var, mag in spec.interventions.items()
                        ],
                    }
                )

        return SimulatorConfig(scenarios=scenarios, joint_scenarios=joint_scenarios)
