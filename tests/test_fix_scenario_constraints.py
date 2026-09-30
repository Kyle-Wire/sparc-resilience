"""Regression tests for the G6 scenario-constraint fixes.

A35 — Canopy + Impervious ≤ 100 honours caps.yml ``enforcement``
      ("warning" never modifies inputs; "enforce" caps only the increment,
      relative to the baseline).
A28 — v4 saturation clipping compares the absolute knee with each cell's
      dose path [t0, t0 + increment] (needs ``treatment_baseline``).
A31 — ScenarioSimulator.load_models tolerates a missing meta-ensemble.
A9  — ``project_coords`` + the simulator query models in projected metres.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from sparc.data.data_utils import project_coords
from sparc.interventions.scenario_engine_v4 import ScenarioEngineV4
from sparc.interventions.scenario_simulator import (
    ScenarioSimulator,
    apply_canopy_impervious_constraint,
)


# ---------------------------------------------------------------------------
# A35 — combined cover constraint
# ---------------------------------------------------------------------------


def _cover_df():
    # Cells 0–1 already exceed 100 % at baseline (canopy overhangs pavement).
    return pd.DataFrame({
        "Pct_Canopy":     [60.0, 70.0, 30.0, 10.0, 95.0],
        "Pct_Impervious": [50.0, 45.0, 50.0, 20.0,  0.0],
    })


class TestCanopyImperviousConstraint:
    @pytest.mark.parametrize("mode", ["warning", "enforce"])
    def test_zero_increment_leaves_inputs_unchanged(self, mode):
        base = _cover_df()
        mod = base.copy()
        stats = apply_canopy_impervious_constraint(base, mod, enforcement=mode)
        pd.testing.assert_frame_equal(mod, base)
        assert stats["n_clipped"] == 0
        assert stats["n_baseline_above_cap"] == 2
        assert stats["n_new_above_cap"] == 0

    def test_warning_mode_never_modifies(self):
        base = _cover_df()
        mod = base.copy()
        mod["Pct_Canopy"] += 20.0
        expected = mod.copy()
        stats = apply_canopy_impervious_constraint(base, mod, enforcement="warning")
        pd.testing.assert_frame_equal(mod, expected)
        assert stats["mode"] == "warning"
        assert stats["n_above_cap"] == 3          # totals 130, 135, 100, 50, 115
        assert stats["n_new_above_cap"] == 1      # only cell 4 is new
        assert stats["n_clipped"] == 0

    def test_enforce_caps_only_the_increment(self):
        base = _cover_df()
        mod = base.copy()
        mod["Pct_Canopy"] += 20.0
        stats = apply_canopy_impervious_constraint(base, mod, enforcement="enforce")
        total_base = (base.Pct_Canopy + base.Pct_Impervious).to_numpy()
        total_mod = (mod.Pct_Canopy + mod.Pct_Impervious).to_numpy()
        # allowed = max(100, baseline total): baseline cells are never pushed below
        # their own baseline total, only the increment is constrained
        assert np.all(total_mod <= np.maximum(100.0, total_base) + 1e-9)
        np.testing.assert_allclose(total_mod, [110.0, 115.0, 100.0, 50.0, 100.0])
        # excess removed from impervious first, then canopy (cell 4 has none)
        np.testing.assert_allclose(mod.Pct_Impervious, [30.0, 25.0, 50.0, 20.0, 0.0])
        np.testing.assert_allclose(mod.Pct_Canopy, [80.0, 90.0, 50.0, 30.0, 100.0])
        assert stats["n_clipped"] == 3

    def test_legacy_hard_cap_would_have_altered_baseline(self):
        """Documents A35: the old absolute cap (≤100) changed cells whose
        baseline already exceeded 100 even for a zero-increment scenario."""
        base = _cover_df()
        total = (base.Pct_Canopy + base.Pct_Impervious).to_numpy()
        assert (total > 100).sum() == 2

    def test_simulator_helper_reads_caps_enforcement(self):
        sim = ScenarioSimulator.__new__(ScenarioSimulator)
        base = _cover_df()

        sim.config = {"caps": {}}   # default → warning
        mod = base.copy()
        mod["Pct_Canopy"] += 20.0
        changes = {"Pct_Canopy": mod.Pct_Canopy.to_numpy() - base.Pct_Canopy.to_numpy()}
        stats = sim._apply_combined_cover_constraint(base, mod, changes)
        assert stats["mode"] == "warning"
        np.testing.assert_allclose(mod.Pct_Impervious, base.Pct_Impervious)

        sim.config = {"caps": {"combined_constraints": {
            "canopy_impervious_sum": {"enforcement": "enforce"}}}}
        mod = base.copy()
        mod["Pct_Impervious"] += 10.0
        changes = {"Pct_Impervious": np.full(len(base), 10.0)}
        stats = sim._apply_combined_cover_constraint(base, mod, changes)
        assert stats["mode"] == "enforce"
        # impervious increment clipped where it would exceed the allowed total
        np.testing.assert_allclose(
            changes["Pct_Impervious"],
            mod.Pct_Impervious.to_numpy() - base.Pct_Impervious.to_numpy(),
        )
        assert changes["Pct_Impervious"][0] == pytest.approx(0.0)   # 110 baseline
        assert changes["Pct_Impervious"][3] == pytest.approx(10.0)  # 30 baseline


# ---------------------------------------------------------------------------
# A28 — v4 saturation clipping with treatment baseline
# ---------------------------------------------------------------------------


def _bare_engine(curves):
    eng = ScenarioEngineV4.__new__(ScenarioEngineV4)
    eng.saturation_marginal_floor = 0.5
    eng._load_dose_response = lambda: curves
    return eng


def _concave_curve(knee_at=50.0):
    doses = np.linspace(0.0, 100.0, 21)
    slopes = np.where(doses < knee_at, 1.0, 0.1)
    return {"dose_levels": doses.tolist(), "marginal_effects": slopes.tolist()}


class TestSaturationClipping:
    def test_knee_is_absolute_and_ignores_initial_flat_region(self):
        eng = _bare_engine({})
        doses = np.linspace(0, 100, 11)
        # flat start, steep middle, flat top: the knee is the upper one
        slopes = np.array([0.05, 0.1, 0.5, 1.0, 1.0, 1.0, 0.9, 0.3, 0.1, 0.05, 0.02])
        knee = eng._saturation_knee({"dose_levels": doses.tolist(),
                                     "marginal_effects": slopes.tolist()})
        assert knee == pytest.approx(70.0)

    def test_scale_uses_per_cell_baseline(self):
        eng = _bare_engine({})
        t0 = np.array([20.0, 45.0, 60.0, 20.0, 55.0, 80.0])
        inc = np.array([10.0, 10.0, 10.0, -10.0, -10.0, -10.0])
        s = eng._saturation_scale(50.0, t0, inc)
        # increases: below knee → 1; straddling (45→55) → 0.5; past knee → 0
        # decreases (mirror): 20→10 → 1; 55→45 → 0.5; 80→70 (all saturated) → 0
        np.testing.assert_allclose(s, [1.0, 0.5, 0.0, 1.0, 0.5, 0.0])
        assert eng._saturation_scale(50.0, np.array([10.0]), np.array([0.0]))[0] == 1.0

    def test_apply_clipping_on_synthetic_long_df(self):
        eng = _bare_engine({"Pct_Canopy": _concave_curve(50.0)})
        long_df = pd.DataFrame({
            "variable": ["Pct_Canopy"] * 3 + ["Albedo"],
            "increment": [10.0, 10.0, 10.0, 0.1],
            "treatment_baseline": [20.0, 45.0, 70.0, 0.2],
            "delta_mean": [-1.0, -1.0, -1.0, -0.5],
            "delta_ci5": [-2.0, -2.0, -2.0, -1.0],
            "delta_ci50": [-1.0, -1.0, -1.0, -0.5],
            "delta_ci95": [0.0, 0.0, 0.0, 0.0],
        })
        out = eng._apply_saturation_clipping(long_df)
        knee = eng._saturation_knee(_concave_curve(50.0))
        expected = np.clip((knee - np.array([20.0, 45.0, 70.0])) / 10.0, 0, 1)
        np.testing.assert_allclose(out["delta_mean"].to_numpy()[:3], -expected)
        assert out["delta_mean"].iloc[3] == -0.5              # other variable untouched
        assert list(out["delta_saturation_clipped"]) == [False, True, True, False]
        np.testing.assert_allclose(out["delta_pre_saturation"], long_df["delta_mean"])

    def test_old_absolute_vs_increment_bug_is_gone(self):
        """Knee 50 vs increment 10: the old rule min(1, knee/inc) never clipped a
        cell already past the knee; now it is fully saturated."""
        eng = _bare_engine({"Pct_Canopy": _concave_curve(50.0)})
        long_df = pd.DataFrame({"variable": ["Pct_Canopy"], "increment": [10.0],
                                "treatment_baseline": [90.0], "delta_mean": [-1.0]})
        out = eng._apply_saturation_clipping(long_df)
        assert out["delta_mean"].iloc[0] == pytest.approx(0.0)

    def test_missing_treatment_baseline_skips_clipping(self):
        eng = _bare_engine({"Pct_Canopy": _concave_curve(50.0)})
        long_df = pd.DataFrame({"variable": ["Pct_Canopy"], "increment": [10.0],
                                "delta_mean": [-1.0]})
        with pytest.warns(RuntimeWarning, match="treatment_baseline"):
            out = eng._apply_saturation_clipping(long_df)
        assert out["delta_mean"].iloc[0] == -1.0

    def test_run_rows_carry_treatment_baseline(self, tmp_path):
        from sparc.registry.run_registry import RunRegistry, set_active_registry
        from sparc.registry.store import ArtifactStore
        from sparc.registry.run_registry import get_active_registry
        from tests.test_scenario_engine_v4 import _baseline_df, _config, _populate_store

        prev = get_active_registry()
        reg = RunRegistry(tmp_path / "registry")
        set_active_registry(reg)
        try:
            store = ArtifactStore(reg)
            _populate_store(store)
            base = _baseline_df()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                eng = ScenarioEngineV4(_config(tmp_path), mode="mode_1_physics")
            eng.run(base, n_draws=20, verbose=False)
            df = store.read_table("4", "scenario_results")
            assert "treatment_baseline" in df.columns
            first = df[df["increment"] == 10.0].sort_values("cell_id")
            np.testing.assert_allclose(first["treatment_baseline"].to_numpy(),
                                       base["Pct_Canopy"].to_numpy())
        finally:
            # ``set_active_registry(None)`` is a no-op unless forced; restore
            # whatever was active before so no registry leaks into later tests.
            set_active_registry(prev, force=True)


# ---------------------------------------------------------------------------
# A9 — projected coordinates
# ---------------------------------------------------------------------------


_RI_CFG = {
    "variables": {"coordinates": ["POINT_X", "POINT_Y"]},
    "crs": {"input": "EPSG:3438", "working": "EPSG:26919"},
}


class TestProjectCoords:
    def test_state_plane_feet_to_utm_metres(self):
        # Two points 1000 US-ft apart (E–W) and one 2000 ft N, near Providence
        df = pd.DataFrame({"POINT_X": [350000.0, 351000.0, 350000.0],
                           "POINT_Y": [270000.0, 270000.0, 272000.0]})
        xy = project_coords(df, _RI_CFG)
        assert xy.shape == (3, 2)
        assert 250_000 < xy[0, 0] < 350_000 and 4.5e6 < xy[0, 1] < 4.7e6   # UTM 19N
        d01 = np.linalg.norm(xy[1] - xy[0])
        d02 = np.linalg.norm(xy[2] - xy[0])
        assert d01 == pytest.approx(1000 * 0.3048, rel=2e-3)
        assert d02 == pytest.approx(2000 * 0.3048, rel=2e-3)

    def test_prefers_projected_columns(self):
        df = pd.DataFrame({"POINT_X": [1.0], "POINT_Y": [2.0],
                           "projected_X": [10.0], "projected_Y": [20.0]})
        np.testing.assert_array_equal(project_coords(df, _RI_CFG), [[10.0, 20.0]])

    def test_same_crs_or_legacy_keys(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        cfg = {"data": {"coord_columns": ["x", "y"]},
               "crs": {"initial": "EPSG:26919", "target_projected": "EPSG:26919"}}
        np.testing.assert_array_equal(project_coords(df, cfg), [[1.0, 3.0], [2.0, 4.0]])

    def test_no_crs_warns_and_returns_raw(self):
        df = pd.DataFrame({"x": [1.0], "y": [3.0]})
        with pytest.warns(RuntimeWarning, match="raw coordinates"):
            out = project_coords(df, {"variables": {"coordinates": ["x", "y"]}})
        np.testing.assert_array_equal(out, [[1.0, 3.0]])


class _RecordingModel:
    """Stand-in base model that records the coordinates it is queried with."""

    def __init__(self, offset):
        self.offset = offset
        self.seen_coords = None

    def predict(self, X, coords=None):
        if coords is not None:
            self.seen_coords = np.asarray(coords)
        return np.full(len(X), self.offset, dtype=float)


def _bare_sim(tmp_path):
    sim = ScenarioSimulator.__new__(ScenarioSimulator)
    sim.config = {"output": {"base_dir": str(tmp_path), "stage_dirs": {}},
                  "caps": {}, **_RI_CFG}
    sim._store = None
    sim.model_dir = tmp_path / "Stage_2_Spatial_CV"
    sim.features = ["a", "b"]
    sim.coord_cols = ["POINT_X", "POINT_Y"]
    sim._models = {}
    sim._meta_model = "sentinel"
    sim._feature_scaler = None
    sim._mgwr_coefficients_raw = None
    sim._mgwr_scaler_scale = None
    sim._mgwr_feature_map = {}
    sim._causal_coefficients = None
    sim._condition_curves = {}
    sim._condition_curve_min_r2 = 0.5
    sim._base_model_weights = {}
    return sim


@pytest.fixture()
def no_active_registry():
    """Isolate from registries leaked by other tests (restored afterwards)."""
    from sparc.registry.run_registry import get_active_registry, set_active_registry
    prev = get_active_registry()
    set_active_registry(None, force=True)
    yield
    set_active_registry(prev, force=True)


@pytest.mark.usefixtures("no_active_registry")
class TestSimulatorModelsAndCoords:
    def test_load_models_without_meta_ensemble(self, tmp_path):
        from sklearn.linear_model import LinearRegression
        sim = _bare_sim(tmp_path)
        full = sim.model_dir / "base_models_full"
        full.mkdir(parents=True)
        for n in ("ols", "gwr", "gwrf", "ggpgam"):
            joblib.dump(LinearRegression().fit(np.eye(2), [0.0, 1.0]), full / f"{n}_model_full.pkl")
        assert not (sim.model_dir / "standard_meta_ensemble.pkl").exists()
        sim.load_models()                     # used to raise FileNotFoundError
        assert sim._meta_model is None
        assert sorted(sim._models) == ["ggpgam", "gwr", "gwrf", "ols"]

    def test_predict_baseline_falls_back_and_uses_projected_coords(self, tmp_path):
        sim = _bare_sim(tmp_path)
        sim._meta_model = None
        sim._models = {n: _RecordingModel(i + 1.0)
                       for i, n in enumerate(("ols", "gwr", "gwrf", "ggpgam"))}
        sim._base_model_weights = {"ols": 0.25, "gwr": 0.25, "gwrf": 0.25, "ggpgam": 0.25}
        df = pd.DataFrame({"a": [0.1, 0.2], "b": [1.0, 2.0],
                           "POINT_X": [350000.0, 351000.0], "POINT_Y": [270000.0, 270000.0]})
        final, *_ = sim._predict_baseline(df)
        np.testing.assert_allclose(final, 2.5)            # mean of 1, 2, 3, 4
        seen = sim._models["gwr"].seen_coords
        np.testing.assert_allclose(seen, project_coords(df, sim.config))
        assert np.linalg.norm(seen[1] - seen[0]) == pytest.approx(304.8, rel=2e-3)


class _FittedModel(_RecordingModel):
    def __init__(self, offset, coords_):
        super().__init__(offset)
        self.coords_ = coords_


@pytest.mark.usefixtures("no_active_registry")
class TestFittedCoordSpaceDetection:
    DF = pd.DataFrame({"a": [0.1, 0.2], "b": [1.0, 2.0],
                       "POINT_X": [350000.0, 351000.0], "POINT_Y": [270000.0, 270000.0]})

    def _sim(self, tmp_path, fitted_coords):
        sim = _bare_sim(tmp_path)
        sim._meta_model = None
        sim._models = {n: _RecordingModel(1.0) for n in ("ols", "gwrf", "ggpgam")}
        sim._models["gwr"] = _FittedModel(1.0, fitted_coords)
        return sim

    def test_projected_fit_uses_projected_coords(self, tmp_path):
        proj = project_coords(self.DF, _RI_CFG)
        sim = self._sim(tmp_path, proj + 5.0)
        np.testing.assert_allclose(sim._model_coords(self.DF), proj)
        assert sim._coord_space == "projected"

    def test_legacy_raw_fit_keeps_raw_coords(self, tmp_path):
        raw = self.DF[["POINT_X", "POINT_Y"]].to_numpy()
        sim = self._sim(tmp_path, raw + 5.0)
        with pytest.warns(RuntimeWarning, match="raw input-CRS"):
            got = sim._model_coords(self.DF)
        np.testing.assert_allclose(got, raw)
        assert sim._coord_space == "raw"
