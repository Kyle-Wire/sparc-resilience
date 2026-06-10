"""JD-2: JEPA-trunk spatial residualizer for causal deconfounding.

Encodes physics features via the JEPA trunk, compresses to a low-dimensional
PCA representation, and then regresses each treatment column on that latent
basis to produce residualised treatment columns ``{col}_resid``.

The residualised columns represent treatment variation that is orthogonal to
the spatially-structured latent factors captured by the trunk — making the
subsequent DML coefficient estimates less biased by unobserved spatial
confounders (e.g. topology, climate zones, land-cover gradients).

Usage::

    from sparc.causal.spatial_residualizer import SpatialResidualizer

    resid = SpatialResidualizer(pca_dims=16)
    df_aug = resid.fit_transform(
        df,
        model=sparc_meta_learner,
        physics_feat_cols=["Pct_Canopy", "Albedo", ...],
        treatment_cols=["Pct_Canopy", "Pct_Impervious"],
        alpha=alpha_array,
        device="cpu",
    )
    # df_aug gains columns: Pct_Canopy_resid, Pct_Impervious_resid
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpatialResidualizer:
    """
    Residualise treatment columns on JEPA trunk embeddings.

    Parameters
    ----------
    pca_dims : int
        Number of PCA components to keep from the ``(N, hidden_dim)``
        trunk embedding before passing to the Ridge regression.  Keeping a
        compact representation avoids over-fitting when N is small.
    ridge_alpha : float
        Regularisation strength for the Ridge regression (default 1.0).
    """

    def __init__(self, pca_dims: int = 16, ridge_alpha: float = 1.0) -> None:
        self.pca_dims = pca_dims
        self.ridge_alpha = ridge_alpha

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        df: pd.DataFrame,
        model,
        physics_feat_cols: List[str],
        treatment_cols: List[str],
        alpha: np.ndarray,
        device: str = "cpu",
    ) -> pd.DataFrame:
        """
        Augment *df* with residualised treatment columns.

        When *model* is ``None`` the original DataFrame is returned unchanged
        and a log message is emitted so operators know the path was skipped.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with ``physics_feat_cols`` and ``treatment_cols`` present.
        model : SPARCMetaLearner, JEPATrunkAdapter, or None
            Trunk encoder.  Must expose either:
            - ``encode(physics_t, alpha_t) -> Tensor(N, hidden_dim)`` [SPARCMetaLearner]
            - ``encode_by_names(X_raw, col_names) -> ndarray(N, hidden_dim)`` [JEPATrunkAdapter]
            The JEPATrunkAdapter path is used automatically when the model
            exposes ``encode_by_names``.  Pass ``None`` to skip.
        physics_feat_cols : list[str]
            Columns in *df* used as physics inputs to the trunk.
        treatment_cols : list[str]
            Columns to residualise.  Adds ``{col}_resid`` to the output.
        alpha : np.ndarray of shape ``(N,)``
            Process-rate values (passed to SPARCMetaLearner; ignored by
            JEPATrunkAdapter which is physics-only).
        device : str
            PyTorch device string (e.g. ``"cpu"`` or ``"cuda"``).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with additional ``{col}_resid`` columns.
        """
        if model is None:
            logger.info(
                "SpatialResidualizer: no JEPA trunk found — "
                "skipping spatial residualisation."
            )
            return df

        try:
            import torch
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge
        except ImportError as exc:  # pragma: no cover
            logger.warning(
                "SpatialResidualizer: missing dependency (%s); "
                "returning df unchanged.", exc,
            )
            return df

        n = len(df)

        # 1. Encode physics features via trunk.
        # JEPATrunkAdapter exposes encode_by_names() for column-aligned encoding;
        # SPARCMetaLearner uses the legacy encode(tensor, alpha) path.
        if hasattr(model, "encode_by_names"):
            physics_np = df[physics_feat_cols].values.astype(np.float32)
            h_np: np.ndarray = model.encode_by_names(physics_np, physics_feat_cols)
        else:
            physics_np = df[physics_feat_cols].values.astype(np.float32)
            alpha_np = np.asarray(alpha, dtype=np.float32).reshape(-1, 1)

            physics_t = torch.tensor(physics_np, dtype=torch.float32).to(device)
            alpha_t = torch.tensor(alpha_np, dtype=torch.float32).to(device)

            model = model.to(device)
            model.eval()

            with torch.no_grad():
                h = model.encode(physics_t, alpha_t)

            h_np: np.ndarray = h.detach().cpu().numpy()  # (N, hidden_dim)

        # 2. PCA-reduce trunk embedding
        n_components = min(self.pca_dims, h_np.shape[1], n - 1)
        if n_components < 1:
            n_components = 1

        pca = PCA(n_components=n_components, random_state=42)
        h_pca: np.ndarray = pca.fit_transform(h_np)  # (N, pca_dims)

        # 3. Ridge-regress each treatment column on h_pca → residuals
        df_out = df.copy()
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)

        for col in treatment_cols:
            if col not in df.columns:
                logger.warning(
                    "SpatialResidualizer: treatment column '%s' not found "
                    "in df — skipping.", col,
                )
                continue
            y = df[col].values
            ridge.fit(h_pca, y)
            resid = y - ridge.predict(h_pca)
            df_out[f"{col}_resid"] = resid

        logger.info(
            "SpatialResidualizer: residualised %d treatment columns "
            "(pca_dims=%d, ridge_alpha=%.3g).",
            len(treatment_cols), n_components, self.ridge_alpha,
        )
        return df_out
