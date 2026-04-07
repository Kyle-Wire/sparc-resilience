"""
CMA-ES hyperparameter search for SPARC V2.

Optimises 12-dimensional log-space vector via Covariance Matrix Adaptation
Evolution Strategy.  The population is evaluated in parallel (1 GPU job per
member) and the objective is the cross-validated SPARC joint loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---- Parameter space definition -------------------------------------------

@dataclass
class HyperparamSpec:
    """One axis of the 12-dim search space — log₁₀ scale."""
    name: str
    log_lo: float
    log_hi: float
    default_log: float


DEFAULT_SPACE: list[HyperparamSpec] = [
    HyperparamSpec("learning_rate",       -5.0, -2.0, -3.0),
    HyperparamSpec("lambda_mse",          -2.0,  1.0,  0.0),
    HyperparamSpec("lambda_exceedance",   -2.0,  1.0, -1.0),
    HyperparamSpec("lambda_physics",      -4.0,  0.0, -2.0),
    HyperparamSpec("lambda_smooth_pred",  -4.0,  0.0, -2.0),
    HyperparamSpec("lambda_smooth_alpha", -4.0,  0.0, -3.0),
    HyperparamSpec("lambda_alpha_prior",  -3.0,  1.0,  0.0),
    HyperparamSpec("lambda_surrogate",    -3.0,  0.0, -1.0),
    HyperparamSpec("lambda_neighbor",     -3.0,  0.0, -1.0),
    HyperparamSpec("clip_norm",           -1.0,  2.0,  0.0),
    HyperparamSpec("dropout",             -2.0, -0.3, -1.0),
    HyperparamSpec("siren_omega",          0.5,  2.5,  1.5),
]


def decode_vector(x: np.ndarray, space: list[HyperparamSpec] | None = None) -> dict[str, float]:
    """Convert a 12-dim log₁₀ vector to a named hyperparameter dict."""
    space = space or DEFAULT_SPACE
    return {s.name: float(10.0 ** x[i]) for i, s in enumerate(space)}


def encode_defaults(space: list[HyperparamSpec] | None = None) -> np.ndarray:
    """Return the default starting point in log space."""
    space = space or DEFAULT_SPACE
    return np.array([s.default_log for s in space])


# ---- Search state --------------------------------------------------------

@dataclass
class CMAESResult:
    best_params: dict[str, float]
    best_loss: float
    n_generations: int
    history: list[dict[str, Any]] = field(default_factory=list)


# ---- Main search entry point --------------------------------------------

def run_cma_es(
    objective_fn: Callable[[dict[str, float]], float],
    *,
    space: list[HyperparamSpec] | None = None,
    max_generations: int = 50,
    population_size: int | None = None,
    sigma0: float = 0.5,
    seed: int = 42,
    tol_fun: float = 1e-6,
) -> CMAESResult:
    """
    Run CMA-ES hyperparameter search.

    Parameters
    ----------
    objective_fn : Callable
        Takes ``dict[str, float]`` of hyperparams → returns scalar loss.
        Lower is better.
    space : optional list of HyperparamSpec
    max_generations : maximum CMA-ES generations
    population_size : override CMA's default lambda
    sigma0 : initial step-size in log space
    seed : random seed
    tol_fun : convergence tolerance on function value

    Returns
    -------
    CMAESResult with best params, loss, and history.
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "CMA-ES requires the 'cma' package.  Install with:  pip install cma>=3.0"
        )

    space = space or DEFAULT_SPACE
    x0 = encode_defaults(space)
    bounds_lo = [s.log_lo for s in space]
    bounds_hi = [s.log_hi for s in space]

    opts: dict[str, Any] = {
        "seed": seed,
        "maxiter": max_generations,
        "tolfun": tol_fun,
        "bounds": [bounds_lo, bounds_hi],
        "verbose": -1,
    }
    if population_size is not None:
        opts["popsize"] = population_size

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    history: list[dict[str, Any]] = []
    gen = 0

    while not es.stop():
        solutions = es.ask()

        fitnesses = []
        for sol in solutions:
            params = decode_vector(np.asarray(sol), space)
            try:
                loss = objective_fn(params)
            except Exception:
                logger.warning("Objective function failed for %s; assigning inf", params)
                loss = float("inf")
            fitnesses.append(loss)

        es.tell(solutions, fitnesses)
        gen += 1

        best_idx = int(np.argmin(fitnesses))
        history.append({
            "generation": gen,
            "best_loss": fitnesses[best_idx],
            "best_params": decode_vector(np.asarray(solutions[best_idx]), space),
            "mean_loss": float(np.mean([f for f in fitnesses if np.isfinite(f)])),
        })
        logger.info(
            "CMA-ES gen %d  best=%.6f  mean=%.6f",
            gen, fitnesses[best_idx],
            history[-1]["mean_loss"],
        )

    best_x = es.result.xbest
    best_params = decode_vector(np.asarray(best_x), space)
    best_loss = float(es.result.fbest)

    logger.info("CMA-ES finished after %d generations.  Best loss=%.6f", gen, best_loss)
    return CMAESResult(
        best_params=best_params,
        best_loss=best_loss,
        n_generations=gen,
        history=history,
    )
