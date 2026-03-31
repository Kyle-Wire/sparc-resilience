"""Basic test for GWENModel on synthetic data."""
import numpy as np
import pytest


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
