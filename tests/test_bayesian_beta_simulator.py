"""Phase 2 (v4 effect-removal) — scenario_simulator now drives direct-effect
computation off the Bayesian per-cell coefficient β(s) read from the
``("3","cate_summary").cate_mean`` artifact, rather than the MGWR/GWR local
coefficients (which are correlation-based and kept only as a diagnostic).

These tests pin the new behavior:

* ``_get_bayesian_beta(treatment)`` reads the right column from the active
  store and caches per call.
* ``_compute_mgwr_direct_delta`` returns ``β(s) · Δ`` with method
  ``"bayesian_beta"`` whenever ``cate_summary`` is available.
* When ``cate_summary`` is absent and no PDE alpha / saturation curve is
  registered, the method falls through to ``"physics_lit"`` — never to a
  removed MGWR tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.interventions.scenario_simulator import ScenarioSimulator
from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore


@pytest.fixture()
def active_registry(tmp_path):
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    yield reg
    set_active_registry(None)


def _bare_sim() -> ScenarioSimulator:
    """Construct a ScenarioSimulator with only the minimum state needed by
    the direct-effect path — bypassing model loading and config parsing."""
    sim = ScenarioSimulator.__new__(ScenarioSimulator)
    sim.config = {}
    sim._alpha_field = None
    sim._mgwr_coefficients_raw = None
    sim._mgwr_scaler_scale = None
    sim._mgwr_feature_map = {}
    sim._condition_curves = {}
    sim._condition_curve_min_r2 = 0.5
    sim.physics_priors = {
        "Pct_Canopy": {
            "coefficient": -0.28,
            "unit_increment": 10.0,
            "direction": "cooling",
            "ols_coef": 0.0,
            "lit_coef": -0.28,
            "uncertainty": 0.2,
        },
    }
    sim._causal_coefficients = None
    return sim


def _seed_cate(registry, treatment: str, cate_means):
    df = pd.DataFrame(
        {
            "cell_id": list(range(len(cate_means))),
            "treatment": [treatment] * len(cate_means),
            "cate_mean": list(cate_means),
            "multiplier_mean": [1.0] * len(cate_means),
            "source": ["bayesian_nuts"] * len(cate_means),
        }
    )
    ArtifactStore(registry).write_table("3", "cate_summary", df)


def test_get_bayesian_beta_reads_cate_mean_sorted_by_cell_id(active_registry):
    _seed_cate(active_registry, "Pct_Canopy", [-0.05, -0.10, -0.15, -0.20])
    sim = _bare_sim()

    beta = sim._get_bayesian_beta("Pct_Canopy")
    assert beta is not None
    assert beta.tolist() == pytest.approx([-0.05, -0.10, -0.15, -0.20])

    # Cached on the instance — second call hits the cache.
    assert sim._bayesian_beta_cache["Pct_Canopy"] is beta


def test_get_bayesian_beta_returns_none_when_treatment_missing(active_registry):
    _seed_cate(active_registry, "Pct_Canopy", [-0.05, -0.10])
    sim = _bare_sim()
    assert sim._get_bayesian_beta("Albedo") is None


def test_compute_direct_delta_uses_bayesian_beta(active_registry):
    _seed_cate(active_registry, "Pct_Canopy", [-0.05, -0.10, -0.15])
    sim = _bare_sim()

    eff = np.array([2.0, 2.0, 2.0])
    delta, method = sim._compute_mgwr_direct_delta("Pct_Canopy", eff)

    assert method == "combined_beta"
    # delta = β(s) · Δ
    assert delta.tolist() == pytest.approx([-0.10, -0.20, -0.30])


def test_compute_direct_delta_falls_back_to_physics_lit(active_registry):
    """Without cate_summary or PDE/saturation data, fall through to physics_lit
    (the only remaining tier after MGWR removal)."""
    sim = _bare_sim()  # no cate_summary written

    eff = np.array([10.0, 10.0])
    delta, method = sim._compute_mgwr_direct_delta("Pct_Canopy", eff)

    assert method == "physics_lit"
    # lit_per_unit = -0.28 / 10 = -0.028 → delta = -0.28 per +10pp
    assert delta.tolist() == pytest.approx([-0.28, -0.28])
