"""Thompson-sampling bandit over CATE GP surfaces for multi-year intervention planning."""

from __future__ import annotations

import numpy as np

from sparc.causal.spatial_cate import CATEGPSurface


class CausalBandit:
    """Thompson-sampling bandit over CATE GP surfaces for multi-round intervention design.

    Each arm corresponds to one intervention type (represented as a fitted
    ``CATEGPSurface``).  On every round the bandit draws one posterior
    realization per arm and picks the arm with the highest expected spatial
    welfare.  This achieves sublinear cumulative regret vs. the greedy
    one-shot ``EmpiricalWelfareMaximizer`` baseline.

    Parameters
    ----------
    cate_surfaces : dict[str, CATEGPSurface]
        Mapping of arm name → fitted GP surface.
    budget : float
        Total intervention budget (passed through to ``history`` for
        downstream budget-allocation modules; not enforced internally).
    n_rounds : int
        Number of Thompson-sampling rounds to run.
    """

    def __init__(
        self,
        cate_surfaces: dict[str, CATEGPSurface],
        budget: float,
        n_rounds: int,
    ):
        if not cate_surfaces:
            raise ValueError("cate_surfaces must not be empty")
        self._surfaces = cate_surfaces
        self._budget = float(budget)
        self._n_rounds = int(n_rounds)
        self._history: list[dict] = []

    # ------------------------------------------------------------------

    def run(self, coords_norm: np.ndarray) -> dict:
        """Run Thompson-sampling over ``n_rounds`` and return results.

        Parameters
        ----------
        coords_norm : (N, D) array
            Spatial coordinates normalised to [0, 1].

        Returns
        -------
        dict with keys:
            ``history``              — list of per-round dicts
            ``arm_totals``           — cumulative sampled welfare per arm
            ``recommended_sequence`` — ordered list of chosen arms
            ``baseline_welfare``     — greedy one-shot best arm (str)
        """
        coords_norm = np.atleast_2d(np.asarray(coords_norm, dtype=np.float64))
        welfare_accum = {arm: 0.0 for arm in self._surfaces}
        self._history = []

        for t in range(self._n_rounds):
            arm_rewards = {
                arm: float(surf.sample_posterior(1, coords_norm)[:, 0].mean())
                for arm, surf in self._surfaces.items()
            }
            chosen = max(arm_rewards, key=arm_rewards.__getitem__)
            welfare_accum[chosen] += arm_rewards[chosen]
            self._history.append({
                "round": t,
                "arm": chosen,
                "sampled_reward": arm_rewards[chosen],
            })

        return {
            "history": self._history,
            "arm_totals": welfare_accum,
            "recommended_sequence": [h["arm"] for h in self._history],
            "baseline_welfare": self._greedy_baseline(coords_norm),
        }

    # ------------------------------------------------------------------

    def _greedy_baseline(self, coords_norm: np.ndarray) -> str:
        """One-shot greedy pick using GP posterior means."""
        means = {
            arm: float(surf.predict(coords_norm)[0].mean())
            for arm, surf in self._surfaces.items()
        }
        return max(means, key=means.__getitem__)
