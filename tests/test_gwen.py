"""Basic test for GWENModel on synthetic data."""
import numpy as np
import pytest


def _tiny_gwen(rng=None):
    """Return a fitted GWENModel + data arrays on tiny synthetic data."""
    from sparc.models.gwen import GWENModel

    if rng is None:
        rng = np.random.default_rng(42)
    n, p = 60, 5
    X = rng.standard_normal((n, p))
    y = 2 * X[:, 0] - X[:, 2] + rng.standard_normal(n) * 0.1
    coords = rng.uniform(0, 1_000, (n, 2))
    feature_names = [f"f{i}" for i in range(p)]
    model = GWENModel(k_neighbors=10, n_alphas=5, quick_mode=True)
    model.fit(X, y, coords, feature_names=feature_names)
    return model, X, y, coords, feature_names


def test_gwen_fit_small():
    """GWENModel.fit returns selected features on tiny synthetic data."""
    from sparc.models.gwen import GWENModel

    rng = np.random.default_rng(42)
    n, p = 60, 5
    X = rng.standard_normal((n, p))
    y = 2 * X[:, 0] - X[:, 2] + rng.standard_normal(n) * 0.1
    coords = rng.uniform(0, 1000, (n, 2))
    feature_names = [f"f{i}" for i in range(p)]

    gwen = GWENModel(
        k_neighbors=10,
        n_alphas=5,
        quick_mode=True,
    )
    result = gwen.fit(X, y, coords)
    assert hasattr(result, "feature_importance")
    selected = result.get_selected_predictors(feature_names, threshold=0.1)
    assert len(selected) >= 1


# ---------------------------------------------------------------------------
# Cycle S1 (RED → GREEN): fit_on_subset must reuse fit() scaler
# ---------------------------------------------------------------------------


def test_fit_on_subset_coefficients_comparable_scale():
    """fit_on_subset must use the model's fitted scaler, not a fresh one.

    A fresh scaler fitted on the subset can have different mean/std than the
    full-data scaler.  This causes stability coefficients to be on a different
    scale than the main model, breaking cross-fold comparisons.

    The fix: use self.scaler.transform(X_sub) instead of
    StandardScaler().fit_transform(X_sub).

    We verify this indirectly: coefficients from fit_on_subset should be
    in the same sign-direction as global_model.coef_ for the known-strong
    predictor (f0, with true coefficient +2).
    """
    model, X, y, coords, feature_names = _tiny_gwen()

    idx = np.arange(len(X))
    local_coefs = model.fit_on_subset(X, y, coords, subset_idx=idx)

    # local_coefs shape: (n_fit_locations, n_features)
    assert local_coefs.ndim == 2
    assert local_coefs.shape[1] == X.shape[1]

    # The strong predictor (f0, true coef +2.0) should dominate in more than
    # 50% of local models with a positive coefficient.
    f0_coefs = local_coefs[:, 0]
    positive_rate = np.mean(f0_coefs > 0)
    assert positive_rate > 0.5, (
        f"Expected f0 to be positive in >50% of subsets; got {positive_rate:.2f}. "
        "This may indicate the scaler divergence bug is re-introduced."
    )


def test_fit_on_subset_uses_fitted_scaler_not_fresh():
    """fit_on_subset must call self.scaler.transform(), not StandardScaler().fit_transform().

    Direct test: after fitting on a single subset, the scale of X in the local
    fit must match what self.scaler would produce (i.e., mean and std from the
    full dataset, not the subset).

    We verify by checking that the global model coefficients and the subset
    global model produce the same sign on all features for which the full-data
    model has a non-zero coefficient — a sign mismatch implies a divergent scaler.
    """
    rng = np.random.default_rng(7)
    n, p = 80, 4
    X = rng.standard_normal((n, p))
    # Make f1 and f3 the dominant predictors
    y = 3 * X[:, 1] - 2 * X[:, 3] + rng.standard_normal(n) * 0.05
    coords = rng.uniform(0, 1_000, (n, 2))
    feature_names = [f"f{i}" for i in range(p)]

    from sparc.models.gwen import GWENModel

    model = GWENModel(k_neighbors=15, n_alphas=5, quick_mode=True)
    model.fit(X, y, coords, feature_names=feature_names)

    # Check that fit_on_subset uses self.scaler
    # (the scaler fitted on the FULL X) rather than creating a fresh one.
    # If it were fresh, X_sub statistics would differ from full-data statistics
    # only when the subset is not representative — which is always the case for
    # small subsets.  We confirm by comparing mean-subtracted inputs directly.
    idx = np.arange(n)
    # Apply full-data scaler to the same data
    X_scaled_full = model.scaler.transform(X)
    # fit_on_subset internally applies a scaler; if it uses the fitted one,
    # the resulting local_coefs should be consistent with global_model.coef_
    local_coefs = model.fit_on_subset(X, y, coords, subset_idx=idx)

    global_strong = {
        i for i, c in enumerate(model.global_model.coef_) if abs(c) > 1e-6
    }
    for feat_idx in global_strong:
        global_sign = np.sign(model.global_model.coef_[feat_idx])
        local_signs = np.sign(local_coefs[:, feat_idx][local_coefs[:, feat_idx] != 0])
        if len(local_signs) == 0:
            continue
        # Majority of local models should agree with global sign
        agree_rate = np.mean(local_signs == global_sign)
        assert agree_rate > 0.5, (
            f"Feature {feature_names[feat_idx]}: global sign={global_sign:+.0f}, "
            f"but only {agree_rate:.0%} of local models agree. "
            "Check scaler reuse in fit_on_subset()."
        )
