"""
Extrapolation Guard — Applicability Domain + Confidence Flags
=============================================================
Provides multi-layered protection against unreliable scenario predictions:

  Layer 0 — Generalized-propensity overlap (Wager 2025 Gap 1)
  Layer 1 — Mahalanobis-based extrapolation scoring
  Layer 2 — Saturation-aware response curves (uses GWRF condition curves)
  Layer 3 — Per-cell confidence classification (HIGH / MODERATE / LOW / SPECULATIVE)

Author: SPARC Labs
Date: February 2026
"""

import numpy as np
from scipy.spatial.distance import mahalanobis


# -------------------------------------------------------------------------
# Layer 0: Generalized-propensity overlap (continuous-treatment positivity)
# -------------------------------------------------------------------------

def compute_overlap_flags(X_train, W_train, X_modified, delta_w):
    """Per-cell overlap flag using :class:`GeneralizedPropensityOverlap`.

    Returns
    -------
    dict with arrays ``percentile`` (0-100) and ``flag``
    (HIGH/LOW/EXTRAPOLATED) shape (n_cells,).
    """
    from sparc.causal.overlap import GeneralizedPropensityOverlap

    ov = GeneralizedPropensityOverlap(method="kernel").fit(X_train, W_train)
    X_modified = np.atleast_2d(np.asarray(X_modified, dtype=float))
    delta_w = np.atleast_1d(np.asarray(delta_w, dtype=float))
    if len(delta_w) == 1 and len(X_modified) > 1:
        delta_w = np.repeat(delta_w, len(X_modified))
    pcts = np.empty(len(X_modified))
    flags: list[str] = []
    for i in range(len(X_modified)):
        a = ov.assess_scenario(X_modified[i], float(delta_w[i]))
        pcts[i] = a["percentile"]
        flags.append(a["flag"])
    return {"percentile": pcts, "flag": np.array(flags, dtype=object)}


# -------------------------------------------------------------------------
# Layer 1: Applicability Domain Detection
# -------------------------------------------------------------------------

def compute_extrapolation_score(X_modified, X_train):
    """
    Per-cell extrapolation score via Mahalanobis distance.

    A score of 0 means the modified point sits at the training-data mean.
    A score > 1 means it lies outside the 95th percentile of training
    Mahalanobis distances — i.e. genuinely out-of-distribution.

    Parameters
    ----------
    X_modified : ndarray, shape (n_cells, n_features)
        Feature vectors *after* the scenario perturbation.
    X_train : ndarray, shape (n_train, n_features)
        Training feature matrix (original, un-perturbed).

    Returns
    -------
    scores : ndarray, shape (n_cells,)
        Normalised Mahalanobis distance (0 = at mean, >1 = extrapolation).
    """
    X_modified = np.asarray(X_modified, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)

    mean_train = np.mean(X_train, axis=0)
    cov_train = np.cov(X_train.T)
    # Regularise for invertibility
    cov_train += 1e-6 * np.eye(cov_train.shape[0])
    cov_inv = np.linalg.pinv(cov_train)

    # Reference threshold: 95th percentile of training Mahalanobis distances
    train_distances = np.array([
        mahalanobis(x, mean_train, cov_inv) for x in X_train
    ])
    threshold = np.percentile(train_distances, 95)
    if threshold < 1e-10:
        threshold = 1.0  # Safety fallback

    scores = np.array([
        mahalanobis(x, mean_train, cov_inv) for x in X_modified
    ]) / threshold

    return scores


# -------------------------------------------------------------------------
# Layer 2: Saturation-Aware Response Curves
# -------------------------------------------------------------------------

def predict_scenario_with_saturation(variable_name, baseline_values,
                                      modified_values, condition_curve,
                                      spatial_multiplier=None):
    """
    Replace linear coefficient extrapolation with the fitted PDP
    saturation curve from the GWRF condition-curve analysis.

    Parameters
    ----------
    variable_name : str
        Name of the perturbed variable (for logging).
    baseline_values : ndarray
        Current (un-perturbed) values for each cell.
    modified_values : ndarray
        Post-perturbation values for each cell.
    condition_curve : dict
        Output of ``GWRFModel.fit_condition_curve()`` — must contain
        ``'fitted_values'`` and the grid used.
    spatial_multiplier : ndarray or None
        Optional per-cell scaling (e.g. from GWR M_ik ratios).

    Returns
    -------
    delta_temp : ndarray
        Temperature change per cell (°F), accounting for saturation.
    """
    from scipy.interpolate import interp1d

    # Build interpolator from the PDP condition curve grid
    grid = np.asarray(condition_curve['grid_values'])
    pdp = np.asarray(condition_curve['pdp_values'])

    # Extrapolate with flat ends (nearest) to avoid wild extrapolation
    curve_func = interp1d(grid, pdp, kind='linear',
                          bounds_error=False, fill_value=(pdp[0], pdp[-1]))

    temp_at_baseline = curve_func(baseline_values)
    temp_at_modified = curve_func(modified_values)

    delta_temp = temp_at_modified - temp_at_baseline

    if spatial_multiplier is not None:
        delta_temp = delta_temp * spatial_multiplier

    return delta_temp


# -------------------------------------------------------------------------
# Layer 3: Confidence Classification
# -------------------------------------------------------------------------

def classify_prediction_confidence(extrapolation_score, actual_change,
                                    max_observed_change):
    """
    Per-cell confidence tier for scenario predictions.

    Parameters
    ----------
    extrapolation_score : float or ndarray
        Output of ``compute_extrapolation_score`` (0 = at mean, >1 = OOD).
    actual_change : float or ndarray
        Absolute magnitude of the variable perturbation.
    max_observed_change : float
        Maximum change observed in the training data for that variable.

    Returns
    -------
    labels : ndarray of str
        One of 'HIGH', 'MODERATE', 'LOW', 'SPECULATIVE' per cell.
    """
    extrapolation_score = np.asarray(extrapolation_score, dtype=float)
    actual_change = np.asarray(actual_change, dtype=float)

    ratio = np.abs(actual_change) / (max_observed_change + 1e-10)

    labels = np.full(extrapolation_score.shape, 'SPECULATIVE', dtype='U12')
    # Apply from most permissive to most restrictive so tighter rules win
    labels[extrapolation_score < 1.5] = 'LOW'
    labels[(extrapolation_score < 1.0) & (ratio < 1.0)] = 'MODERATE'
    labels[(extrapolation_score < 0.5) & (ratio < 0.5)] = 'HIGH'

    return labels


# -------------------------------------------------------------------------
# Convenience wrapper — full guard
# -------------------------------------------------------------------------

def apply_extrapolation_guard(variable_name, baseline_values, modified_values,
                               X_train, condition_curve=None,
                               spatial_multiplier=None,
                               physics_coefficient=None, unit_increment=1.0,
                               X_modified_full=None):
    """
    Run all three guard layers and return scenario deltas + diagnostics.

    If a ``condition_curve`` is supplied, the saturation-aware path is used;
    otherwise falls back to linear coefficient extrapolation.

    Parameters
    ----------
    variable_name : str
    baseline_values : ndarray
    modified_values : ndarray
    X_train : ndarray
        Full training feature matrix (all predictors).
    condition_curve : dict or None
        GWRF condition curve for this variable.
    spatial_multiplier : ndarray or None
    physics_coefficient : float or None
        Fallback linear coefficient (°F per unit_increment).
    unit_increment : float
    X_modified_full : ndarray or None
        Full modified feature matrix (all predictors) with the perturbed
        column already substituted.  When provided, the full multivariate
        Mahalanobis distance is used for extrapolation scoring.  When
        *None*, falls back to single-column approximation (deprecated).

    Returns
    -------
    dict with keys:
        'delta_temp'          — ndarray of temperature changes
        'extrapolation_score' — ndarray (0 = safe, >1 = OOD)
        'confidence'          — ndarray of str labels
        'method'              — 'saturation_curve' or 'linear'
    """
    actual_change = modified_values - baseline_values

    # -- Layer 1: Extrapolation score -----------------------------------------
    if X_modified_full is not None and X_train.ndim > 1:
        # Full multivariate Mahalanobis—proper applicability domain
        extrap_scores = compute_extrapolation_score(
            X_modified_full, X_train
        )
    else:
        # Legacy single-column approximation (fallback)
        extrap_scores = compute_extrapolation_score(
            modified_values.reshape(-1, 1),
            X_train[:, :1] if X_train.ndim > 1 else X_train.reshape(-1, 1)
        )

    # -- Layer 2: Delta temperature -------------------------------------------
    if condition_curve is not None:
        delta_temp = predict_scenario_with_saturation(
            variable_name, baseline_values, modified_values,
            condition_curve, spatial_multiplier
        )
        method = 'saturation_curve'
    else:
        # Fallback to linear coefficient
        if physics_coefficient is None:
            raise ValueError(
                f"No condition_curve or physics_coefficient for {variable_name}")
        base_delta = physics_coefficient * (actual_change / unit_increment)
        delta_temp = base_delta * (spatial_multiplier if spatial_multiplier is not None else 1.0)
        method = 'linear'

    # -- Layer 3: Confidence labels -------------------------------------------
    max_obs_change = np.ptp(X_train[:, 0]) if X_train.ndim > 1 else np.ptp(X_train)
    confidence = classify_prediction_confidence(
        extrap_scores, actual_change, max_obs_change
    )

    return {
        'delta_temp': delta_temp,
        'extrapolation_score': extrap_scores,
        'confidence': confidence,
        'method': method,
    }
