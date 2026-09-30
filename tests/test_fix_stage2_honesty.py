"""Regression tests for Stage-2 honesty defects (A8–A11).

* A8  — the neural meta-learner's per-fold surrogate targets were in-sample
        full-data fits (named ``base_oof_predictions``); they must be the TRUE
        Stage-2a out-of-fold predictions, with the full-data fits reserved for
        the final retrain (``base_full_fitted``).
* A9  — Stage 2b refit used raw config coordinates (e.g. US feet) while CV
        used projected metres.
* A10 — failed folds were silently filled with ``mean(y)``.
* A11 — ``n_splits=5`` hard-coded; 0.8 train-fraction heuristics.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sparc.run.enhanced_spatial_cv as escv
import sparc.run.v2_neural_training as v2


# ---------------------------------------------------------------------------
# A9 — coordinate helper
# ---------------------------------------------------------------------------

def test_resolve_model_coords_prefers_projected_columns():
    df = pd.DataFrame({
        "POINT_X": [1000.0, 2000.0], "POINT_Y": [3000.0, 4000.0],   # US feet
        "projected_X": [304.8, 609.6], "projected_Y": [914.4, 1219.2],  # metres
    })
    coords, cols = escv._resolve_model_coords(df, ["POINT_X", "POINT_Y"])
    assert cols == ["projected_X", "projected_Y"]
    np.testing.assert_allclose(coords, df[["projected_X", "projected_Y"]].values)

    raw_only = df[["POINT_X", "POINT_Y"]]
    coords, cols = escv._resolve_model_coords(raw_only, ["POINT_X", "POINT_Y"])
    assert cols == ["POINT_X", "POINT_Y"]
    np.testing.assert_allclose(coords, raw_only.values)


# ---------------------------------------------------------------------------
# A8 — surrogate target selection
# ---------------------------------------------------------------------------

def test_build_neural_base_targets_uses_oof_for_folds_and_full_for_final():
    n = 6
    oof = {m: np.arange(n, dtype=float) + k for k, m in enumerate(("ols", "gwr", "gwrf", "ggpgam"))}
    full = {m: np.full(n, 100.0 + k) for k, m in enumerate(("ols", "gwr", "gwrf", "ggpgam"))}

    oof_t, full_t = escv._build_neural_base_targets(oof, full, n, usable_models=["ols", "gwr", "gwrf", "ggpgam"])
    assert set(oof_t) == {"gwr", "gwrf", "ggpgam"}
    for m in oof_t:
        np.testing.assert_array_equal(oof_t[m], oof[m])
        np.testing.assert_array_equal(full_t[m], full[m])


def test_build_neural_base_targets_excludes_failed_nan_and_mismatched():
    n = 5
    oof = {"gwr": np.ones(n), "gwrf": np.array([1, 2, np.nan, 4, 5.0]), "ggpgam": np.ones(n + 1)}
    full = {"gwr": np.zeros(n), "gwrf": np.zeros(n), "ggpgam": np.zeros(n)}
    oof_t, full_t = escv._build_neural_base_targets(oof, full, n, usable_models=["gwr", "gwrf", "ggpgam"])
    assert set(oof_t) == {"gwr"}
    # Final retrain anchors exactly the models the CV folds anchored.
    assert set(full_t) == {"gwr"}

    oof_t, full_t = escv._build_neural_base_targets({"gwr": np.ones(n)}, full, n, usable_models=[])
    assert oof_t is None and full_t is None

    # No Stage-2b fits → the final retrain falls back to OOF inside v2.
    oof_t, full_t = escv._build_neural_base_targets({"gwr": np.ones(n)}, {}, n, usable_models=["gwr"])
    assert set(oof_t) == {"gwr"} and full_t is None


def test_train_neural_meta_final_retrain_target_source():
    oof, full = {"gwr": np.ones(3)}, {"gwr": np.zeros(3)}
    assert v2._final_retrain_base_source(oof, full) is full
    assert v2._final_retrain_base_source(oof, None) is oof
    assert v2._final_retrain_base_source(None, None) is None

    sig = inspect.signature(v2.train_neural_meta)
    assert sig.parameters["base_full_fitted"].default is None
    # Added after the existing parameters: positional callers are unaffected.
    names = list(sig.parameters)
    assert names.index("base_full_fitted") > names.index("quick_eval")


# ---------------------------------------------------------------------------
# A10 — failed folds are NaN + recorded, never mean-filled
# ---------------------------------------------------------------------------

class _FailOnFoldModel:
    """Linear stub that raises when its training set contains a sentinel row."""

    def __init__(self, fail_if_train_contains=None):
        self.fail_if_train_contains = fail_if_train_contains

    def fit(self, X, y, coords):
        if self.fail_if_train_contains is not None and np.any(np.isclose(X[:, 0], self.fail_if_train_contains)):
            raise RuntimeError("boom")
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X, coords):
        return np.full(len(X), self.mu_ + 0.5)


def _toy_cv(n=12, k=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n) + 10.0
    coords = rng.uniform(0, 100, size=(n, 2))
    idx = np.arange(n)
    folds = [(idx[idx % k != f], idx[idx % k == f]) for f in range(k)]
    return X, y, coords, folds


def _bare_cv():
    obj = object.__new__(escv.EnhancedSpatialCV)
    obj._hw = {"outer_jobs": 2, "memory_limit_gb": 8}
    obj.paths = SimpleNamespace(stage3_dir="unused")
    return obj


class _FailWithoutRows(_FailOnFoldModel):
    """Fails whenever its training targets lack all of *required_y*."""

    def __init__(self, required_y):
        super().__init__()
        self.required_y = np.asarray(required_y)

    def fit(self, X, y, coords):
        if not np.any(np.isin(np.round(y, 12), np.round(self.required_y, 12))):
            raise RuntimeError("training split lacks the required rows")
        return super().fit(X, y, coords)


def test_sequential_cv_failed_fold_is_nan_and_recorded():
    X, y, coords, folds = _toy_cv()
    cv = _bare_cv()
    # Only fold 2's training split lacks fold 2's rows → only fold 2 fails.
    flaky = _FailWithoutRows(required_y=y[folds[2][1]])

    oof = cv._sequential_cv_training(X, y, coords, [_FailOnFoldModel(), flaky], ["ok", "flaky"], folds, ["a", "b"])

    assert cv.failed_folds_ == {"flaky": [2]}
    assert np.all(np.isnan(oof[folds[2][1], 1]))
    assert np.all(np.isfinite(oof[folds[0][1], 1]))
    assert np.all(np.isfinite(oof[:, 0]))
    # Never the mean of y_train (the old silent fill).
    for tr, te in folds:
        assert not np.any(np.isclose(oof[te, 1], np.mean(y[tr])))


def test_fold_worker_failure_returns_nan():
    X, y, coords, folds = _toy_cv()
    tr, te = folds[0]
    model = _FailOnFoldModel()
    model.fit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fit fails"))  # noqa: E731
    fold_idx, test_idx, preds, intel = escv.train_single_model_fold_worker(
        (0, tr, te, X, y, coords, model, "ols", ["a", "b"])
    )
    assert np.all(np.isnan(preds)) and intel is None


class _AlwaysFails:
    def fit(self, X, y, coords):
        raise RuntimeError("always")

    def predict(self, X, coords):  # pragma: no cover
        return np.zeros(len(X))


def test_parallel_cv_records_worker_failures(monkeypatch):
    X, y, coords, folds = _toy_cv()
    monkeypatch.setattr(escv, "EXTRACT_OOF_INTELLIGENCE", False)
    cv = _bare_cv()
    cv.failed_folds_ = {}
    oof = cv._parallel_cv_training(
        X, y, coords, [_FailOnFoldModel(), _AlwaysFails()], ["ok", "bad"], folds, ["a", "b"],
    )
    assert cv.failed_folds_ == {"bad": [0, 1, 2]}
    assert np.all(np.isnan(oof[:, 1]))
    assert np.all(np.isfinite(oof[:, 0]))


def test_failed_fold_detection_and_usable_models():
    X, y, coords, folds = _toy_cv()
    oof = np.column_stack([y + 1.0, y + 2.0])
    oof[folds[1][1], 1] = np.nan
    assert escv._failed_folds_from_predictions(oof, folds, ["a", "b"]) == {"b": [1]}

    preds = {"a": oof[:, 0], "b": oof[:, 1], "c": y}
    assert escv._usable_base_models(preds, {"c": [0]}) == ["a"]


def test_finite_scores_never_feed_nan_to_metrics(monkeypatch):
    seen = []
    real_r2 = escv.r2_score

    def _guarded_r2(a, b):
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        seen.append(len(a))
        return real_r2(a, b)

    monkeypatch.setattr(escv, "r2_score", _guarded_r2)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, np.nan, 3.0, 5.0])
    r2, rmse, n = escv._finite_scores(y, pred)
    assert n == 3 and seen == [3]
    assert rmse == pytest.approx(np.sqrt(1.0 / 3.0))
    assert np.isnan(escv._finite_scores(y, np.full(4, np.nan))[0])


# ---------------------------------------------------------------------------
# A11 — configured n_splits / train fraction
# ---------------------------------------------------------------------------

def test_create_optimized_models_uses_configured_train_fraction(monkeypatch):
    cv = object.__new__(escv.EnhancedSpatialCV)
    cv._cfg = SimpleNamespace(
        gwr_params={"bandwidth": 10_000}, gwrf_params={"k_neighbors": 10_000},
        ggpgam_params={}, n_splits=4, output_crs=None,
    )
    cv.base_config = {}
    cv._monotone_constraints = {}
    cv._hw = {"gwrf_local_jobs": 1}
    monkeypatch.setattr(cv, "get_variable_bandwidths", lambda: None, raising=False)
    monkeypatch.setattr(cv, "get_kernel_field", lambda names: None, raising=False)

    models = cv.create_optimized_models(n_samples=1000)
    # (k−1)/k = 0.75 → train 750; GWR global bw cap = int(1000·0.75·0.85) = 637.
    assert models[1].bandwidth == 637
    # GWRF k cap = int(750·0.85) // 2 = 318.
    assert models[2].k_neighbors == 318


# ---------------------------------------------------------------------------
# End-to-end wiring of Stage 2a → 2b → 2c in main() (heavy parts mocked)
# ---------------------------------------------------------------------------

class _Abort(BaseException):
    """Stops main() right after train_neural_meta's arguments are captured."""


class _StubBase:
    offset = 0.0

    def fit(self, X, y, coords=None, **kw):
        self.fit_coords = None if coords is None else np.asarray(coords).copy()
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, X, coords=None):
        return self._y + self.offset


class _StubOLS(_StubBase):
    offset = 20.0

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return self._y + self.offset


class _StubGWR(_StubBase):
    offset = 21.0

    def fit(self, X, y, coords, extract_coefficients=False, output_path=None):
        return super().fit(X, y, coords)


class _StubGWRF(_StubBase):
    offset = 22.0

    def fit(self, X, y, coords, extract_derivatives=False, output_dir=None, feature_names=None):
        return super().fit(X, y, coords)


class _StubGGPGAM(_StubBase):
    offset = 23.0

    def fit(self, X, y, coords, extract_derivatives=False, output_dir=None):
        return super().fit(X, y, coords)


def test_main_passes_true_oof_to_folds_and_full_fits_to_final(tmp_path, monkeypatch):
    n = 12
    rng = np.random.default_rng(3)
    proj = rng.uniform(0, 500, size=(n, 2))
    data = pd.DataFrame({
        "OBJECTID": np.arange(100, 100 + n),
        "T": rng.normal(size=n) + 30.0,
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "POINT_X": proj[:, 0] / 0.3048, "POINT_Y": proj[:, 1] / 0.3048,  # feet
        "projected_X": proj[:, 0], "projected_Y": proj[:, 1],            # metres
    })
    csv = tmp_path / "raw.csv"
    data.to_csv(csv, index=False)
    idx = np.arange(n)
    folds = [(idx[idx % 2 == 1], idx[idx % 2 == 0]), (idx[idx % 2 == 0], idx[idx % 2 == 1])]
    y = data["T"].values
    oof_table = pd.DataFrame({
        "ols": y + 1.0, "gwr": y + 2.0, "gwrf": y + 3.0, "ggpgam": y + 4.0,
    })
    oof_table.loc[folds[1][1], "ggpgam"] = np.nan  # ggpgam fold 1 failed

    stubs = [_StubOLS(), _StubGWR(), _StubGWRF(), _StubGGPGAM()]
    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir()

    class _FakeCV:
        def __init__(self):
            self._cfg = SimpleNamespace(
                skip_stage_2_base_models=False, skip_stage_2b_full_retrain=False,
                features=["f1", "f2"], identifier="OBJECTID", target="T",
                coordinates=["POINT_X", "POINT_Y"], initial_crs=None, target_crs=None,
                output_dir=str(tmp_path), raw_csv_path=str(csv), n_splits=2, raw={},
            )
            self.paths = SimpleNamespace(
                oof_predictions=stage2_dir / "oof.csv", folds_file=stage2_dir / "folds.pkl",
                stage2_dir=stage2_dir, get_relative_path=lambda p: str(p),
            )
            self.base_config = {}

        def run_enhanced_spatial_cv(self):
            return {"failed_folds": {"ggpgam": [1]},
                    "performance": {"individual_models": {"gwrf": {"r2": 0.0}}}}

        def create_optimized_models(self, n_samples=None):
            return stubs

    def _load_blob(path, stage=None, artifact_id=None):
        return list(folds) if artifact_id == "folds" else None

    monkeypatch.setattr(escv, "EnhancedSpatialCV", _FakeCV)
    monkeypatch.setattr(escv, "load_and_preprocess_data", lambda **kw: data.copy())
    monkeypatch.setattr(escv, "load_table_path", lambda *a, **k: oof_table.copy())
    monkeypatch.setattr(escv, "load_blob_path", _load_blob)
    monkeypatch.setattr(escv, "save_blob_path", lambda *a, **k: None)
    monkeypatch.setattr(escv, "exists_path", lambda *a, **k: False)

    captured = {}

    def _spy_train_neural_meta(**kwargs):
        captured.update(kwargs)
        raise _Abort()

    monkeypatch.setattr(v2, "train_neural_meta", _spy_train_neural_meta)

    ctx = SimpleNamespace(paths=SimpleNamespace(stage2_dir=stage2_dir, output_dir=tmp_path))
    with pytest.raises(_Abort):
        escv.main(ctx)

    # A8: folds see the Stage-2a OOF arrays; ggpgam (failed fold) excluded.
    oof_t = captured["base_oof_predictions"]
    assert set(oof_t) == {"gwr", "gwrf"}
    np.testing.assert_allclose(oof_t["gwr"], y + 2.0)
    np.testing.assert_allclose(oof_t["gwrf"], y + 3.0)
    # …and the final retrain sees the Stage-2b in-sample fits.
    full_t = captured["base_full_fitted"]
    assert set(full_t) == {"gwr", "gwrf"}
    np.testing.assert_allclose(full_t["gwr"], y + _StubGWR.offset)
    np.testing.assert_allclose(full_t["gwrf"], y + _StubGWRF.offset)

    # A9: Stage-2b refits use the projected (metre) coordinates.
    for stub in stubs[1:]:
        np.testing.assert_allclose(stub.fit_coords, proj)
    np.testing.assert_allclose(captured["coords"], proj)
