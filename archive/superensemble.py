"""
SuperEnsemble — second-level stacker over per-member SPARC predictions.

Train 10 independent SPARC pipelines (one per ensemble member), then
blend their predictions with a LightGBM super-learner.  Optionally
includes spatial coordinates and fingerprint projection scores as
additional features.

Usage
-----
    from models.superensemble import SuperEnsemble

    se = SuperEnsemble()
    se.fit(member_preds, y_truth, coords=coords)
    blended = se.predict(member_preds, coords=coords)
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


class SuperEnsemble:
    """Second-level LightGBM stacker over per-member SPARC predictions.

    Parameters
    ----------
    val_size : float
        Fraction of data for early-stopping validation.
    random_state : int
    num_boost_round : int
    early_stopping_rounds : int
    """

    def __init__(
        self,
        val_size: float = 0.2,
        random_state: int = 42,
        num_boost_round: int = 300,
        early_stopping_rounds: int = 25,
    ) -> None:
        self.val_size = val_size
        self.random_state = random_state
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.Booster] = None
        self.scaler = StandardScaler()
        self.member_ids: Optional[list] = None
        self.feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _build_features(
        self,
        member_preds: Dict[str, np.ndarray],
        coords: Optional[np.ndarray] = None,
        extra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stack member predictions (+ optional coords/extra) into a feature matrix."""
        if self.member_ids is None:
            self.member_ids = sorted(member_preds.keys())
        parts = [member_preds[m].reshape(-1, 1) for m in self.member_ids]
        names = [f"pred_{m}" for m in self.member_ids]

        if coords is not None:
            parts.append(np.asarray(coords))
            names += ["coord_x", "coord_y"]

        if extra is not None:
            ex = np.asarray(extra)
            if ex.ndim == 1:
                ex = ex.reshape(-1, 1)
            parts.append(ex)
            names += [f"extra_{i}" for i in range(ex.shape[1])]

        self.feature_names = names
        return np.hstack(parts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        member_preds: Dict[str, np.ndarray],
        y_true: np.ndarray,
        coords: Optional[np.ndarray] = None,
        extra: Optional[np.ndarray] = None,
    ) -> "SuperEnsemble":
        """Train the super-learner.

        Parameters
        ----------
        member_preds : dict
            ``{member_id: predictions}`` from 10 independent SPARC runs.
        y_true : array (n_samples,)
        coords : array (n_samples, 2), optional
        extra : array, optional
            Additional features (e.g. fingerprint projection scores).
        """
        X = self._build_features(member_preds, coords, extra)
        X_scaled = self.scaler.fit_transform(X)
        y = np.asarray(y_true).ravel()

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.val_size, random_state=self.random_state,
        )

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }

        self.model = lgb.train(
            params,
            train_ds,
            valid_sets=[val_ds],
            num_boost_round=self.num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=25),
            ],
        )

        val_pred = self.model.predict(X_val)
        val_r2 = 1 - np.mean((y_val - val_pred) ** 2) / np.var(y_val)
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        print(f"SuperEnsemble — val R² = {val_r2:.4f}, RMSE = {val_rmse:.4f}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        member_preds: Dict[str, np.ndarray],
        coords: Optional[np.ndarray] = None,
        extra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SuperEnsemble not fitted — call fit() first.")
        X = self._build_features(member_preds, coords, extra)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict:
        if self.model is None:
            raise RuntimeError("Not fitted")
        imp = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, imp))
