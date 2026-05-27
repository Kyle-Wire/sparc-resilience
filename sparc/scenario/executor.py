"""ScenarioExecutor: pure-Python chain-rollout logic, no FastAPI dependency."""

from __future__ import annotations

import numpy as np


class ScenarioExecutor:
    """Encapsulates the multi-step latent rollout result-shaping logic."""

    @staticmethod
    def run(
        *,
        actions: list,
        rollout_fn,
        rollout_kwargs: dict,
        mode: str = "latent",
    ) -> dict:
        """Run a latent-space chain rollout and return a shaped response dict.

        Parameters
        ----------
        actions:
            List of ``(treatment, delta_x, delta_t)`` tuples (at least one).
        rollout_fn:
            Callable that performs the actual rollout.  Must accept all keys
            in ``rollout_kwargs`` plus ``actions``, ``mode``, and
            ``max_steps`` as keyword arguments and return an object with
            ``.steps``, ``.final``, and ``.n_steps`` attributes.
        rollout_kwargs:
            Extra keyword arguments forwarded to ``rollout_fn``.
        mode:
            Rollout mode string passed through to ``rollout_fn``.
        """
        if not actions:
            raise ValueError("'actions' must contain at least one item")

        result = rollout_fn(
            **rollout_kwargs,
            actions=actions,
            mode=mode,
            max_steps=10,
        )

        steps_out = []
        for i, step in enumerate(result.steps):
            steps_out.append(
                {
                    "step": i + 1,
                    "treatment": step.treatment,
                    "delta_x": step.delta_x,
                    "mean_delta": float(np.mean(step.delta)),
                    "p5_delta": float(np.percentile(step.delta, 5)),
                    "p95_delta": float(np.percentile(step.delta, 95)),
                }
            )

        cumulative = float(np.mean(result.final.delta)) if result.n_steps > 0 else 0.0
        return {
            "n_steps": result.n_steps,
            "steps": steps_out,
            "cumulative_mean_delta": cumulative,
        }
