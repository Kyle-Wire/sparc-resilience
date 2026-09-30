"""Regression tests for the GW-model defects A3 / A4 / A5.

* A3 — ``GWRModel._apply_kernel_field`` only *filled gaps* (``setdefault``),
  so the target↔predictor cross-range never replaced a predictor's own
  (auto, Stage-0) range.  It must now override auto bandwidths while leaving
  explicit user bandwidths (``manual_parameters.bandwidths``) alone.
* A4 — numpy inputs got ``feature_i`` names, so every KernelField lookup
  failed and anisotropic weights collapsed to 1 in CV.  ``fit`` now takes
  ``feature_names`` and the CV fold helper forwards them.
* A5 — the GWRF anisotropy branch used a nonexistent ``kernel_field.kernels``
  and module-level ``anisotropic_distance``; it is rewritten on the real API.
* B2 gate — a predictor only counts as anisotropic when b/a ≤ 0.87.
"""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from sparc.models.gwr import GWRModel, is_effectively_anisotropic
from sparc.models.gwrf import GWRFModel
from sparc.models.kernel_field import KernelField, PredictorKernel
from sparc.models.spec import ModelSpec
from sparc.run.gwr_bandwidth import (
    ResolvedBandwidths,
    bandwidths_are_user_specified,
    resolve_bandwidth,
)


def _grid(n: int = 12, spacing: float = 10.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    return np.column_stack([xx.ravel() * spacing, yy.ravel() * spacing]).astype(float)


def _aniso_kernel(name="a", kx=1 / 200.0, ky=1 / 10.0, theta=0.0, **kw) -> PredictorKernel:
    return PredictorKernel(name=name, kappa_x=kx, kappa_y=ky, theta_rad=theta, **kw)


# ---------------------------------------------------------------------------
# B2 anisotropy gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ratio, expected",
    [(0.5, True), (0.87, True), (0.90, False), (0.95, False), (1.0, False)],
)
def test_anisotropy_gate_axis_ratio(ratio, expected):
    p = _aniso_kernel(kx=ratio * 0.1, ky=0.1)
    assert is_effectively_anisotropic(p) is expected
    # Orientation of the ratio must not matter.
    q = _aniso_kernel(kx=0.1, ky=ratio * 0.1)
    assert is_effectively_anisotropic(q) is expected


def test_weak_anisotropy_is_isotropic_in_gwr_and_gwrf():
    weak = KernelField("t", [_aniso_kernel("a", kx=0.095, ky=0.1, bandwidth_to_outcome=50.0)])
    strong = KernelField("t", [_aniso_kernel("a", kx=0.02, ky=0.1, bandwidth_to_outcome=50.0)])

    for kf, expected in ((weak, False), (strong, True)):
        gwr = GWRModel(kernel_field=kf)
        gwr.feature_names_ = ["a"]
        assert gwr._has_anisotropic_field() is expected

        gwrf = GWRFModel(kernel_field=kf)
        gwrf.feature_names_ = ["a"]
        assert gwrf._has_anisotropic_field() is expected

    # Weakly anisotropic predictor → isotropic Matérn on its cross-range.
    gwr = GWRModel(kernel_field=weak)
    gwr.feature_names_ = ["a"]
    pts = np.array([[10.0, 0.0], [0.0, 10.0]])
    w = gwr._per_predictor_anisotropic_weights(np.zeros(2), pts)
    assert w[0] == pytest.approx(w[1])


# ---------------------------------------------------------------------------
# A4 — feature names for numpy input
# ---------------------------------------------------------------------------

def _gwr_data(n=10, seed=0):
    rng = np.random.default_rng(seed)
    coords = _grid(n)
    X = rng.normal(size=(len(coords), 2))
    y = X @ np.array([1.0, -0.5]) + 0.01 * coords[:, 0] + 0.1 * rng.normal(size=len(coords))
    return X, y, coords


def test_gwr_numpy_fit_uses_feature_names_for_anisotropic_weights():
    X, y, coords = _gwr_data()
    kf = KernelField("t", [_aniso_kernel("a"), PredictorKernel(name="b", bandwidth_to_outcome=60.0)])

    model = GWRModel(variable_bandwidths={"a": 60.0, "b": 60.0}, kernel_field=kf,
                     min_points=10, use_constrained_regression=False)
    model.fit(X, y, coords, feature_names=["a", "b"])
    assert model.feature_names_ == ["a", "b"]
    assert model._has_anisotropic_field()

    w = model._per_predictor_anisotropic_weights(coords[0], coords[:40])
    assert np.ptp(w) > 0.1                          # not uniform
    # Elongated along x (κx ≪ κy): a point 50 m along x outweighs 50 m along y.
    w_xy = model._per_predictor_anisotropic_weights(np.zeros(2), np.array([[50.0, 0.0], [0.0, 50.0]]))
    assert w_xy[0] > w_xy[1]

    # Without names (the pre-fix CV situation) the field cannot be used.
    legacy = GWRModel(variable_bandwidths={"a": 60.0, "b": 60.0}, kernel_field=kf,
                      min_points=10, use_constrained_regression=False)
    legacy.fit(X, y, coords)
    assert legacy.feature_names_ == ["feature_0", "feature_1"]
    assert not legacy._has_anisotropic_field()


def test_gwr_dataframe_columns_take_precedence_over_feature_names():
    import pandas as pd

    X, y, coords = _gwr_data(n=8)
    df = pd.DataFrame(X, columns=["a", "b"])
    model = GWRModel(bandwidth=20, min_points=10, use_constrained_regression=False)
    model.fit(df, y, coords, feature_names=["x", "y"])
    assert model.feature_names_ == ["a", "b"]


def test_cv_fold_helper_forwards_feature_names():
    from sparc.run.enhanced_spatial_cv import _fit_predict_fold

    class _NamedModel:
        def fit(self, X, y, coords, feature_names=None):
            self.seen = (coords is not None, feature_names)
            return self

        def predict(self, X, coords):
            return np.zeros(len(X)), np.ones(len(X))  # (mean, uncertainty)

    class _LocalNameModel:
        def fit(self, X, y):
            feature_names = ["local only"]  # a local, not a parameter
            self.names = feature_names
            return self

        def predict(self, X):
            return np.ones(len(X))

    X = np.zeros((6, 2))
    m = _NamedModel()
    out = _fit_predict_fold(m, X, np.zeros(6), X, X[:2], X[:2], feature_names=["a", "b"])
    assert m.seen == (True, ["a", "b"])
    np.testing.assert_array_equal(out, [0.0, 0.0])

    m2 = _LocalNameModel()
    out2 = _fit_predict_fold(m2, X, np.zeros(6), X, X[:3], X[:3], feature_names=["a", "b"])
    np.testing.assert_array_equal(out2, [1.0, 1.0, 1.0])


def test_cv_worker_passes_fold_feature_names_to_gwr():
    import sparc.run.enhanced_spatial_cv as escv

    X, y, coords = _gwr_data(n=8)
    kf = KernelField("t", [_aniso_kernel("a"), _aniso_kernel("b", kx=0.1, ky=0.02)])
    model = GWRModel(variable_bandwidths={"a": 60.0, "b": 60.0}, kernel_field=kf,
                     min_points=10, use_constrained_regression=False)
    idx = np.arange(len(y))
    train_idx, test_idx = idx[idx % 4 != 0], idx[idx % 4 == 0]

    import functools

    seen = {}
    orig_fit = GWRModel.fit

    @functools.wraps(orig_fit)
    def _spy_fit(self, *args, **kwargs):
        res = orig_fit(self, *args, **kwargs)
        seen["names"] = list(self.feature_names_)
        seen["aniso"] = self._has_anisotropic_field()
        return res

    GWRModel.fit = _spy_fit
    old_flag = escv.EXTRACT_OOF_INTELLIGENCE
    escv.EXTRACT_OOF_INTELLIGENCE = False
    try:
        _, _, preds, _ = escv.train_single_model_fold_worker(
            (0, train_idx, test_idx, X, y, coords, model, "gwr", ["a", "b"])
        )
    finally:
        GWRModel.fit = orig_fit
        escv.EXTRACT_OOF_INTELLIGENCE = old_flag
    assert seen == {"names": ["a", "b"], "aniso": True}
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# A3 — cross-range vs auto / manual bandwidths
# ---------------------------------------------------------------------------

def _kf_with_cross_range():
    return KernelField("t", [
        PredictorKernel(name="a", bandwidth_to_outcome=800.0),
        PredictorKernel(name="b", bandwidth_to_outcome=900.0),
    ])


def test_resolve_bandwidth_tags_source_and_stays_dict_compatible():
    stage0 = {"individual_results": {"a": {"optimal_bandwidth": 300}, "t": {"optimal_bandwidth": 5}}}
    auto = resolve_bandwidth({}, stage0, target_var="t")
    assert isinstance(auto, dict) and isinstance(auto, ResolvedBandwidths)
    assert auto == {"a": 300.0}
    assert auto.source == "stage0" and not bandwidths_are_user_specified(auto)

    manual = resolve_bandwidth({"manual_parameters": {"bandwidths": {"a": "250"}}})
    assert manual == {"a": 250.0}
    assert manual.source == "manual" and bandwidths_are_user_specified(manual)

    wired = resolve_bandwidth({"manual_parameters": {"bandwidths": {"a": 250},
                                                     "source": "correlogram_auto"}})
    assert wired.source == "auto_wired" and not bandwidths_are_user_specified(wired)

    assert resolve_bandwidth({}) is None
    # Tag survives copies and pickling (models are deep-copied per CV fold).
    for clone in (copy.deepcopy(auto), pickle.loads(pickle.dumps(auto))):
        assert clone == auto and clone.source == "stage0"


def test_cross_range_overrides_auto_bandwidths():
    auto = resolve_bandwidth({}, {"individual_results": {"a": {"optimal_bandwidth": 300},
                                                         "c": {"optimal_bandwidth": 400}}})
    m = GWRModel(variable_bandwidths=auto, kernel_field=_kf_with_cross_range())
    assert m.variable_bandwidths == {"a": 800.0, "b": 900.0, "c": 400.0}


def test_cross_range_does_not_override_manual_bandwidths():
    manual = resolve_bandwidth({"manual_parameters": {"bandwidths": {"a": 300}}})
    m = GWRModel(variable_bandwidths=manual, kernel_field=_kf_with_cross_range())
    # Manual value kept; missing predictor filled from the cross-range.
    assert m.variable_bandwidths == {"a": 300.0, "b": 900.0}

    # Untagged plain dicts (direct construction) keep the historical behaviour.
    m2 = GWRModel(variable_bandwidths={"a": 300.0}, kernel_field=_kf_with_cross_range())
    assert m2.variable_bandwidths == {"a": 300.0, "b": 900.0}


def test_auto_source_survives_model_spec_construction_path():
    auto = resolve_bandwidth({}, {"individual_results": {"a": {"optimal_bandwidth": 300}}})
    cfg = ModelSpec.from_kwargs("gwr", variable_bandwidths=auto,
                                kernel_field=_kf_with_cross_range()).as_gwr_config()
    m = GWRModel.from_config(cfg)
    assert m.variable_bandwidths["a"] == 800.0
    # Idempotent when re-constructed from the merged bandwidths.
    m_again = GWRModel(variable_bandwidths=m.variable_bandwidths,
                       kernel_field=_kf_with_cross_range())
    assert m_again.variable_bandwidths == m.variable_bandwidths


# ---------------------------------------------------------------------------
# A5 — GWRF anisotropic neighbourhoods
# ---------------------------------------------------------------------------

class _SpyRF:
    """Stand-in for RandomForestRegressor that records each local fit."""

    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def fit(self, X, y, sample_weight=None):
        _SpyRF.calls.append((np.asarray(X).copy(), np.asarray(sample_weight).copy()))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _fit_gwrf_with_spy(monkeypatch, kernel_field):
    import sparc.models.gwrf as gwrf_mod

    _SpyRF.calls = []
    monkeypatch.setattr(gwrf_mod, "RandomForestRegressor", _SpyRF)
    coords = _grid(21, 10.0)
    # Features ARE the coordinates, so each local X records its neighbourhood.
    X = coords.copy()
    y = np.zeros(len(coords))
    model = GWRFModel(n_estimators=2, k_neighbors=40, n_jobs=1, kernel_field=kernel_field)
    model.fit(X, y, coords, feature_names=["a", "b"])
    centre = int(np.argmin(np.hypot(coords[:, 0] - 100.0, coords[:, 1] - 100.0)))
    X_local, w = _SpyRF.calls[centre]
    return X_local, w


def test_gwrf_anisotropic_neighbourhood_elongated_along_x(monkeypatch):
    # κx ≪ κy, θ = 0 → long range along x, short along y.
    kf = KernelField("t", [_aniso_kernel("a", kx=1 / 200.0, ky=1 / 10.0, theta=0.0)])
    X_local, w = _fit_gwrf_with_spy(monkeypatch, kf)
    x_extent = np.ptp(X_local[:, 0])
    y_extent = np.ptp(X_local[:, 1])
    assert x_extent >= 4 * y_extent
    assert len(X_local) == 40
    assert np.ptp(w) > 0  # Matérn sample weights, not uniform


def test_gwrf_rotated_anisotropy_follows_theta(monkeypatch):
    kf = KernelField("t", [_aniso_kernel("a", kx=1 / 200.0, ky=1 / 10.0, theta=np.pi / 2)])
    X_local, _ = _fit_gwrf_with_spy(monkeypatch, kf)
    assert np.ptp(X_local[:, 1]) >= 4 * np.ptp(X_local[:, 0])


def test_gwrf_weak_anisotropy_uses_isotropic_knn(monkeypatch):
    kf = KernelField("t", [_aniso_kernel("a", kx=0.095, ky=0.1, theta=0.0)])
    X_local, w = _fit_gwrf_with_spy(monkeypatch, kf)
    ratio = np.ptp(X_local[:, 0]) / max(np.ptp(X_local[:, 1]), 1e-9)
    assert 0.6 <= ratio <= 1.6
