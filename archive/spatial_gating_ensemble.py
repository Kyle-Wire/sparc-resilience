"""
Spatially-Varying Model Selection Layer — Gating Ensemble
=========================================================
Instead of a single meta-learner that always blends all four base models
equally, this uses a **mixture-of-experts** architecture where a gating
network assigns location-specific weights to each base model.

The gate is trained to predict which base model has the lowest absolute
error at each location, using only spatial features (coordinates,
Laplacian eigenvectors, local statistics — NOT base model predictions,
which would create a circular dependency).

Stage A — Oracle model selection → train gate classifier
Stage B — Residual correction on gated predictions

When to use:
  If pairwise correlation of OOF predictions across base models is < 0.85
  for at least some pairs, the gating network can capture spatial regime
  differences.  If all correlations > 0.95, skip and use standard stacking.

Author: SPARC Labs
Date: February 2026
"""

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


class SpatialGatingEnsemble:
    """
    Spatially-varying model selection layer.

    At each location a gating network assigns softmax weights to base
    models based on spatial features.  The final prediction is the
    weighted sum of base-model predictions.

    Parameters
    ----------
    model_names : list of str
        Ordered names of base models (must match keys in prediction dicts).
    gate_n_estimators : int
        LightGBM trees for the gate classifier.
    gate_num_leaves : int
        Tree complexity for the gate.
    correction_n_estimators : int
        Trees for the optional residual correction model.
    correction_num_leaves : int
        Tree complexity for correction.
    """

    def __init__(self, model_names=None, gate_n_estimators=100,
                 gate_num_leaves=31, correction_n_estimators=50,
                 correction_num_leaves=15):
        self.model_names = model_names or ['ols', 'gwr', 'gwrf', 'ggpgam']
        self.n_models = len(self.model_names)
        self.gate_n_estimators = gate_n_estimators
        self.gate_num_leaves = gate_num_leaves
        self.correction_n_estimators = correction_n_estimators
        self.correction_num_leaves = correction_num_leaves

        self.gate_model = None
        self.correction_model = None
        self.scaler = StandardScaler()
        self._base_model_correlations = None

    # -----------------------------------------------------------------
    # Diagnostic: should you use the gating network at all?
    # -----------------------------------------------------------------
    @staticmethod
    def diagnose_model_diversity(base_predictions_dict):
        """
        Compute pairwise Pearson correlations of base-model OOF predictions.

        If all correlations > 0.95, the gating network adds complexity for
        no gain and you should skip it.

        Returns
        -------
        corr_matrix : ndarray, shape (n_models, n_models)
        recommendation : str
            'USE_GATING' or 'SKIP_GATING'
        """
        names = sorted(base_predictions_dict.keys())
        preds = np.column_stack([base_predictions_dict[n] for n in names])
        corr = np.corrcoef(preds.T)

        # Check if any off-diagonal pair is < 0.85
        n = len(names)
        min_corr = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                min_corr = min(min_corr, corr[i, j])

        recommendation = 'USE_GATING' if min_corr < 0.85 else 'SKIP_GATING'

        print(f"\nBase-model prediction correlations:")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j > i:
                    print(f"  {ni} vs {nj}: {corr[i, j]:.4f}")
        print(f"Minimum pairwise correlation: {min_corr:.4f}")
        print(f"Recommendation: {recommendation}\n")

        return corr, recommendation

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------
    def fit(self, base_predictions_dict, y, coords, spatial_features):
        """
        Two-stage training.

        Stage A — Oracle selection: for each training cell, determine which
        base model has the lowest absolute error.  Train a LightGBM
        multiclass classifier to predict the best model from spatial
        features only.

        Stage B — Residual correction: train a light regressor on what
        remains after gated blending.

        Parameters
        ----------
        base_predictions_dict : dict {model_name: ndarray}
            OOF predictions from each base model.
        y : ndarray, shape (n,)
            Ground truth.
        coords : ndarray, shape (n, 2)
            Spatial coordinates.
        spatial_features : ndarray, shape (n, d)
            Laplacian eigenvectors, local statistics, etc.
            Must NOT include base-model predictions.
        """
        base_preds = np.column_stack([
            base_predictions_dict[m] for m in self.model_names
        ])

        # -- Stage A: Oracle model selection --------------------------------
        errors = np.abs(base_preds - y[:, np.newaxis])
        best_model_per_cell = np.argmin(errors, axis=1)

        gate_X = np.column_stack([coords, spatial_features])
        gate_X_scaled = self.scaler.fit_transform(gate_X)

        self.gate_model = lgb.LGBMClassifier(
            n_estimators=self.gate_n_estimators,
            num_leaves=self.gate_num_leaves,
            objective='multiclass',
            num_class=self.n_models,
            verbose=-1,
        )
        self.gate_model.fit(gate_X_scaled, best_model_per_cell)

        # Soft gate weights (predicted probabilities)
        gate_probs = self.gate_model.predict_proba(gate_X_scaled)
        gated_preds = np.sum(base_preds * gate_probs, axis=1)

        # -- Stage B: Residual correction -----------------------------------
        residuals = y - gated_preds
        if np.std(residuals) > 0.01:
            self.correction_model = lgb.LGBMRegressor(
                n_estimators=self.correction_n_estimators,
                num_leaves=self.correction_num_leaves,
                verbose=-1,
            )
            correction_X = np.column_stack([
                gated_preds.reshape(-1, 1), gate_X_scaled
            ])
            self.correction_model.fit(correction_X, residuals)
            corrected = gated_preds + self.correction_model.predict(correction_X)
            print(f"Gated ensemble — correction residual std: "
                  f"{np.std(y - corrected):.4f}")
        else:
            print("Gated ensemble — residuals already small, skipping correction.")

        # Diagnostics
        from sklearn.metrics import r2_score as _r2
        train_r2 = _r2(y, gated_preds)
        print(f"Gated ensemble train R²: {train_r2:.4f}")

        # Store oracle distribution for interpretability
        counts = np.bincount(best_model_per_cell, minlength=self.n_models)
        for name, cnt in zip(self.model_names, counts):
            print(f"  Oracle selects {name}: {cnt} cells "
                  f"({cnt / len(y) * 100:.1f}%)")

        return self

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------
    def predict(self, base_predictions_dict, coords, spatial_features):
        """
        Predict using the gated ensemble.

        Returns
        -------
        predictions : ndarray
        gate_probs : ndarray, shape (n, n_models)
            Per-cell model weights — useful for interpretability maps.
        """
        base_preds = np.column_stack([
            base_predictions_dict[m] for m in self.model_names
        ])

        gate_X = np.column_stack([coords, spatial_features])
        gate_X_scaled = self.scaler.transform(gate_X)
        gate_probs = self.gate_model.predict_proba(gate_X_scaled)

        gated_preds = np.sum(base_preds * gate_probs, axis=1)

        if self.correction_model is not None:
            correction_X = np.column_stack([
                gated_preds.reshape(-1, 1), gate_X_scaled
            ])
            correction = self.correction_model.predict(correction_X)
            gated_preds = gated_preds + correction

        return gated_preds, gate_probs

    # -----------------------------------------------------------------
    # Interpretability helpers
    # -----------------------------------------------------------------
    def get_dominant_model_map(self, gate_probs):
        """
        Convert gate probabilities to a map of the 'winning' model.

        Returns
        -------
        dominant_model_idx : ndarray of int
        dominant_model_names : list of str  (per cell)
        """
        dominant_idx = np.argmax(gate_probs, axis=1)
        dominant_names = [self.model_names[i] for i in dominant_idx]
        return dominant_idx, dominant_names

    def get_gate_feature_importance(self):
        """Return feature importances from the gate classifier."""
        if self.gate_model is None:
            raise ValueError("Gate not trained.")
        return self.gate_model.feature_importances_
