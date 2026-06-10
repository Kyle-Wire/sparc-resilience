"""
Few-shot prediction stub for SPARC V4.

V4 will support few-shot calibration: given a small number of ground-truth
observations (e.g. 10–50 points), rapidly fine-tune the city head while
keeping the shared trunk frozen.  This module defines the interface;
implementation is deferred to V4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sparc.data.satellite_types import SatelliteFeatureSet


@dataclass
class FewShotPrediction:
    """
    Container for few-shot inference results.

    Attributes
    ----------
    y_pred : (N,) predicted target values
    uncertainty : (N,) prediction uncertainty
    coords : (N, 2) coordinates
    n_calibration : number of ground-truth points used for calibration
    calibration_r2 : R² on calibration points (LOO-CV)
    metadata : additional info
    """
    y_pred: np.ndarray
    uncertainty: np.ndarray
    coords: np.ndarray
    n_calibration: int = 0
    calibration_r2: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def few_shot_predict(
    features: SatelliteFeatureSet,
    calibration_X: np.ndarray,
    calibration_y: np.ndarray,
    calibration_coords: np.ndarray,
    trunk_path: str | None = None,
    registry_path: str | None = None,
    n_finetune_epochs: int = 50,
) -> FewShotPrediction:
    """
    Few-shot inference with rapid calibration.

    Parameters
    ----------
    features : SatelliteFeatureSet for the target city
    calibration_X : (K, D) feature matrix for calibration points
    calibration_y : (K,) target values for calibration points
    calibration_coords : (K, 2) coordinates for calibration points
    trunk_path : path to a trained shared trunk checkpoint
    registry_path : path to city registry.  When supplied, the registry is
        queried by ``features.climate_zone`` first; if no city matches the
        zone, the global trunk is used; if no global trunk exists either, a
        fresh model is initialised.  ``trunk_path`` takes precedence when
        both are supplied.
    n_finetune_epochs : epochs for city head fine-tuning

    Returns
    -------
    FewShotPrediction
    """
    if registry_path is not None and trunk_path is None:
        # Resolve trunk from registry in-memory via TrunkLoader (no tempfile needed).
        # x_dim is not known yet at this point — TrunkLoader.from_registry is called
        # later inside the main flow once feat_np.shape[1] is determined.
        _use_registry = True
    else:
        _use_registry = False

    import torch
    import torch.nn.functional as F
    import numpy as np
    from sparc.inference.anp import SpatialANP

    # ------------------------------------------------------------------ #
    # Step 1 — Build raw feature matrices                                  #
    # ------------------------------------------------------------------ #
    coords = features.coords                  # (N, 2)
    band_mat = features.to_feature_matrix()   # (N, B)
    if band_mat.shape[1] > 0:
        feat_np = np.concatenate([coords, band_mat], axis=1).astype("float32")
    else:
        feat_np = coords.astype("float32")

    calib_X_np = calibration_X.astype("float32")       # (K, D)
    calib_c_np = calibration_coords.astype("float32")  # (K, 2)
    ctx_feat = np.concatenate([calib_c_np, calib_X_np], axis=1)  # (K, 2+D)

    # ------------------------------------------------------------------ #
    # Step 2 — Optional climate zone conditioning                          #
    #          Append a learned 8-dim embedding to every feature vector.   #
    #          When trunk_path is a bundle, use its saved encoder weights   #
    #          instead of random initialisation.                            #
    # ------------------------------------------------------------------ #
    if features.climate_zone is not None:
        from sparc.models.climate_encoder import ClimateZoneEncoder
        # Prefer saved encoder weights from the checkpoint bundle; fall back
        # to random init when no encoder is present (plain checkpoint or none).
        if trunk_path is not None:
            _cze = ClimateZoneEncoder.from_checkpoint(trunk_path) or ClimateZoneEncoder(embed_dim=8)
        else:
            _cze = ClimateZoneEncoder(embed_dim=8)
        _idx = torch.tensor([ClimateZoneEncoder.zone_to_index(features.climate_zone)])
        with torch.no_grad():
            climate_emb = _cze(_idx).numpy()  # (1, 8)
        feat_np = np.concatenate(
            [feat_np, np.tile(climate_emb, (feat_np.shape[0], 1))], axis=1
        )  # (N, x_dim + 8)
        ctx_feat = np.concatenate(
            [ctx_feat, np.tile(climate_emb, (ctx_feat.shape[0], 1))], axis=1
        )  # (K, ctx_dim + 8)

    K = ctx_feat.shape[0]

    # ------------------------------------------------------------------ #
    # Step 3 — Load or create the trunk                                    #
    # ------------------------------------------------------------------ #
    def _align_cols(mat: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate a 2-D array to *target_dim* columns."""
        d = mat.shape[1]
        if d < target_dim:
            return np.pad(mat, ((0, 0), (0, target_dim - d)))
        return mat[:, :target_dim]

    if trunk_path is not None:
        anp = SpatialANP.from_checkpoint(trunk_path)
        # Align feature matrices to the x_dim embedded in the checkpoint.
        ckpt_x_dim = anp.x_dim
        feat_np = _align_cols(feat_np, ckpt_x_dim)
        ctx_feat = _align_cols(ctx_feat, ckpt_x_dim)
    elif _use_registry:
        x_dim = feat_np.shape[1]
        ctx_feat = _align_cols(ctx_feat, x_dim)
        from sparc.inference.trunk import TrunkLoader
        anp = TrunkLoader.from_registry(registry_path, features.climate_zone, x_dim=x_dim)
        # Align feature matrices to the x_dim embedded in the checkpoint
        # (same as the trunk_path branch above).
        ckpt_x_dim = anp.x_dim
        feat_np   = _align_cols(feat_np,   ckpt_x_dim)
        ctx_feat  = _align_cols(ctx_feat,  ckpt_x_dim)
    else:
        x_dim = feat_np.shape[1]
        # Align context features to the target x_dim.
        ctx_feat = _align_cols(ctx_feat, x_dim)
        anp = SpatialANP(x_dim=x_dim)

    target_t = torch.from_numpy(feat_np)
    ctx_x_t = torch.from_numpy(ctx_feat)
    ctx_y_t = torch.from_numpy(calibration_y.astype("float32")).unsqueeze(1)  # (K, 1)

    # ------------------------------------------------------------------ #
    # Step 4 — City-head fine-tuning on calibration data                   #
    #          Trunk (encoder + attention) stays frozen;                    #
    #          only the decoder (city head) is updated.                     #
    # ------------------------------------------------------------------ #
    calibration_r2: float | None = None

    if n_finetune_epochs > 0:
        # Freeze trunk; enable gradients only for the decoder.
        for name, param in anp.named_parameters():
            param.requires_grad = name.startswith("decoder")

        optimizer = torch.optim.Adam(
            [p for p in anp.decoder.parameters()], lr=1e-3
        )
        anp.train()

        for _epoch in range(n_finetune_epochs):
            if K < 2:
                # Single calibration point: use it as both context and target.
                optimizer.zero_grad()
                mean_cal, _ = anp(ctx_x_t, ctx_y_t, ctx_x_t)
                loss = F.mse_loss(mean_cal, ctx_y_t)
            else:
                # Random 80/20 context/target split within calibration.
                perm = torch.randperm(K)
                n_ctx = max(1, K * 4 // 5)
                ctx_idx = perm[:n_ctx]
                # Ensure at least one target point even when K is tiny.
                tgt_idx = perm[n_ctx:] if n_ctx < K else perm[-1:]
                optimizer.zero_grad()
                mean_cal, _ = anp(
                    ctx_x_t[ctx_idx], ctx_y_t[ctx_idx], ctx_x_t[tgt_idx]
                )
                loss = F.mse_loss(mean_cal, ctx_y_t[tgt_idx])
            loss.backward()
            optimizer.step()

        anp.eval()
        # Freeze all params now that fine-tuning is done.
        for param in anp.parameters():
            param.requires_grad = False

        # Compute LOO calibration R² to characterise fit quality.
        if K >= 2:
            with torch.no_grad():
                loo_preds: list[float] = []
                for i in range(K):
                    mask = torch.ones(K, dtype=torch.bool)
                    mask[i] = False
                    m, _ = anp(ctx_x_t[mask], ctx_y_t[mask], ctx_x_t[[i]])
                    loo_preds.append(float(m.squeeze()))
            loo_arr = np.array(loo_preds, dtype="float32")
            y_true = calibration_y.astype("float32")
            ss_res = float(np.sum((y_true - loo_arr) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            calibration_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    # ------------------------------------------------------------------ #
    # Step 5 — Full-context prediction on the target city                  #
    # ------------------------------------------------------------------ #
    anp.eval()
    with torch.no_grad():
        mean, std = anp(ctx_x_t, ctx_y_t, target_t)

    return FewShotPrediction(
        y_pred=mean.squeeze(1).numpy(),
        uncertainty=std.squeeze(1).numpy(),
        coords=coords,
        n_calibration=K,
        calibration_r2=calibration_r2,
    )
