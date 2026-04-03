"""Global Sensitivity Analysis for SPARC scenario simulations.

Provides Morris (screening) and Sobol (variance-based) methods to
quantify how scenario-input uncertainty propagates to prediction
uncertainty.  Uses SALib when available, with a pure-NumPy Morris
fallback.

Usage::

    from sparc.evaluation.sensitivity import SensitivityAnalyzer

    sa = SensitivityAnalyzer(config, simulator)
    results = sa.run(data, method='morris')  # or 'sobol'
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from SALib.sample import morris as morris_sample, saltelli as saltelli_sample  # type: ignore[import-untyped]
    from SALib.analyze import morris as morris_analyze, sobol as sobol_analyze  # type: ignore[import-untyped]
    SALIB_AVAILABLE = True
except ImportError:
    morris_sample = None  # type: ignore[assignment]
    saltelli_sample = None  # type: ignore[assignment]
    morris_analyze = None  # type: ignore[assignment]
    sobol_analyze = None  # type: ignore[assignment]
    SALIB_AVAILABLE = False


class SensitivityAnalyzer:
    """Global sensitivity analysis for scenario-simulation inputs.

    Parameters
    ----------
    config : dict
        SPARC project config (must include ``scenarios`` key).
    simulator : ScenarioSimulator
        An initialised and model-loaded ``ScenarioSimulator`` instance.
    """

    def __init__(self, config: dict, simulator: Any):
        self.config = config
        self.sim = simulator
        self.scenarios = config.get('scenarios', [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        method: str = 'morris',
        n_trajectories: int = 10,
        n_sobol: int = 512,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute sensitivity analysis.

        Parameters
        ----------
        data : DataFrame
            Baseline spatial data.
        method : ``'morris'`` | ``'sobol'``
        n_trajectories : int
            Number of Morris trajectories (only for ``morris``).
        n_sobol : int
            Number of Saltelli samples (only for ``sobol``; total model
            evals = ``n_sobol * (2*k + 2)``).
        verbose : bool

        Returns
        -------
        dict with keys ``method``, ``problem``, ``results``, ``summary_df``.
        """
        problem = self._build_problem()
        if verbose:
            print(f"\n  [SA] Running {method.upper()} sensitivity analysis")
            print(f"       Variables: {problem['names']}")

        if method == 'morris':
            return self._run_morris(data, problem, n_trajectories, verbose)
        elif method == 'sobol':
            return self._run_sobol(data, problem, n_sobol, verbose)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'morris' or 'sobol'.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_problem(self) -> dict:
        """Build a SALib-compatible problem dict from scenario config."""
        names, bounds = [], []
        for sc in self.scenarios:
            var = sc['variable']
            # Bounds: ±max(increments) around mean
            incs = sc.get('increments', [1.0])
            max_inc = max(incs)
            names.append(var)
            bounds.append([-max_inc, max_inc])
        return {
            'num_vars': len(names),
            'names': names,
            'bounds': bounds,
        }

    def _evaluate_model(
        self,
        data: pd.DataFrame,
        delta_matrix: np.ndarray,
        var_names: List[str],
    ) -> np.ndarray:
        """Apply each row of *delta_matrix* as simultaneous variable
        perturbations and return the mean prediction delta for each sample.
        """
        n_samples = delta_matrix.shape[0]
        outputs = np.zeros(n_samples)

        for i in range(n_samples):
            modified = data.copy()
            for j, var in enumerate(var_names):
                if var in modified.columns:
                    modified[var] = modified[var].values + delta_matrix[i, j]
                    # Respect physical bounds from scenario config
                    sc = next((s for s in self.scenarios if s['variable'] == var), {})
                    bmin = sc.get('min_val')
                    bmax = sc.get('max_val')
                    if bmin is not None:
                        modified[var] = modified[var].clip(lower=bmin)
                    if bmax is not None:
                        modified[var] = modified[var].clip(upper=bmax)

            # Cheap path: use physics-only run to get mean ΔT
            try:
                summary, _ = self.sim.run(modified, verbose=False)
                outputs[i] = summary['Total_Delta_Mean'].mean()
            except Exception:
                outputs[i] = 0.0

        return outputs

    # --- Morris --------------------------------------------------------

    def _run_morris(
        self, data, problem, n_trajectories, verbose,
    ) -> Dict[str, Any]:
        if SALIB_AVAILABLE:
            X = morris_sample.sample(problem, N=n_trajectories)  # type: ignore[union-attr]
        else:
            X = self._morris_sample_fallback(problem, n_trajectories)

        if verbose:
            print(f"       {X.shape[0]} model evaluations")

        Y = self._evaluate_model(data, X, problem['names'])

        if SALIB_AVAILABLE:
            Si = morris_analyze.analyze(problem, X, Y)  # type: ignore[union-attr]
            mu_star = Si['mu_star']
            sigma = Si['sigma']
        else:
            mu_star, sigma = self._morris_analyze_fallback(
                problem, X, Y, n_trajectories,
            )

        summary = pd.DataFrame({
            'Variable': problem['names'],
            'mu_star': mu_star,
            'sigma': sigma,
            'mu_star_rank': pd.Series(mu_star).rank(ascending=False).astype(int),
        }).sort_values('mu_star', ascending=False)

        if verbose:
            print("\n  [SA] Morris Elementary Effects:")
            for _, row in summary.iterrows():
                print(f"       {row['Variable']:25s}  μ*={row['mu_star']:.4f}  σ={row['sigma']:.4f}")

        return {
            'method': 'morris',
            'problem': problem,
            'results': {'mu_star': mu_star, 'sigma': sigma},
            'summary_df': summary,
        }

    # --- Sobol ---------------------------------------------------------

    def _run_sobol(
        self, data, problem, n_sobol, verbose,
    ) -> Dict[str, Any]:
        if not SALIB_AVAILABLE:
            warnings.warn("SALib not installed; falling back to Morris.")
            return self._run_morris(data, problem, 10, verbose)

        X = saltelli_sample.sample(problem, N=n_sobol)  # type: ignore[union-attr]
        if verbose:
            print(f"       {X.shape[0]} model evaluations")

        Y = self._evaluate_model(data, X, problem['names'])
        Si = sobol_analyze.analyze(problem, Y)  # type: ignore[union-attr]

        summary = pd.DataFrame({
            'Variable': problem['names'],
            'S1': Si['S1'],
            'ST': Si['ST'],
        }).sort_values('ST', ascending=False)

        if verbose:
            print("\n  [SA] Sobol Indices:")
            for _, row in summary.iterrows():
                print(f"       {row['Variable']:25s}  S1={row['S1']:.4f}  ST={row['ST']:.4f}")

        return {
            'method': 'sobol',
            'problem': problem,
            'results': Si,
            'summary_df': summary,
        }

    # --- Pure-NumPy Morris fallback ------------------------------------

    @staticmethod
    def _morris_sample_fallback(problem: dict, n_traj: int) -> np.ndarray:
        """Generate a simple OAT Morris sample without SALib."""
        k = problem['num_vars']
        bounds = np.array(problem['bounds'])
        rng = np.random.default_rng(42)
        # Each trajectory: k+1 points (base + one-at-a-time perturbations)
        samples = []
        for _ in range(n_traj):
            base = rng.uniform(bounds[:, 0], bounds[:, 1])
            samples.append(base.copy())
            order = rng.permutation(k)
            for j in order:
                delta = (bounds[j, 1] - bounds[j, 0]) * 0.5
                direction = rng.choice([-1, 1])
                new_point = samples[-1].copy()
                new_point[j] = np.clip(
                    new_point[j] + direction * delta, bounds[j, 0], bounds[j, 1],
                )
                samples.append(new_point)
        return np.array(samples)

    @staticmethod
    def _morris_analyze_fallback(
        problem: dict, X: np.ndarray, Y: np.ndarray, n_traj: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute elementary effects from OAT sample."""
        k = problem['num_vars']
        step = k + 1
        ee: Dict[int, list] = {j: [] for j in range(k)}

        for t in range(n_traj):
            start = t * step
            for i in range(k):
                dx = X[start + i + 1] - X[start + i]
                changed = np.where(np.abs(dx) > 1e-12)[0]
                if len(changed) == 1:
                    j = changed[0]
                    dy = Y[start + i + 1] - Y[start + i]
                    ee[j].append(dy / dx[j])

        mu_star = np.array([
            np.mean(np.abs(ee[j])) if ee[j] else 0.0 for j in range(k)
        ])
        sigma = np.array([
            np.std(ee[j]) if len(ee[j]) > 1 else 0.0 for j in range(k)
        ])
        return mu_star, sigma
