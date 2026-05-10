# PRD: Spatial Adaptive Learner Upgrades

**SPARC Labs LLC | May 2026**
**Status: Ready for Implementation**

---

## Problem Statement

SPARC's meta-learner fuses MGWR, GGPGAM, and GWRF surrogate predictions through a globally uniform MLP (`base_enc: Linear(3 → H)`). This projection applies identically to every cell, learning "on average MGWR contributes X" — but MGWR's interpretable linear structure is trustworthy in smooth suburban zones while GWRF's nonlinear local RFs are more reliable in complex urban cores. The current architecture cannot represent this: the blend is fixed regardless of location.

Two additional gaps: GWRF fits local RFs using isotropic Euclidean neighbor selection despite `kernel_field` being available with full Matérn anisotropy; and GWRF produces no per-cell uncertainty, leaving the `divergence_audit` unable to use its posterior-aware flagging path (the `gwr_std` parameter is always `None`).

---

## Goals

1. Replace the uniform base model blend with a spatially-adaptive `SpatialGatingHead` that produces per-cell softmax weights over the three surrogates, conditioned on local spatial context.
2. Wire GWRF anisotropy — same Matérn elliptical distance already implemented in GWR Phase 7.
3. Produce GWRF per-cell prediction uncertainty (tree disagreement std) and route it into the divergence audit and the gating head.

## Non-Goals

- Per-cell uncertainty for MGWR (WLS coefficient SEs are not prediction uncertainty; cost/benefit unfavorable).
- Per-cell uncertainty for GGPGAM (PyGAM intervals unreliable for regression).
- GWEN — excluded from scope entirely.
- Multi-model ensemble layer on top of the meta-learner (the meta-learner is already the ensemble).
- Spatio-temporal kernels or lagged spatial effects (separate future item).

---

## Design Decisions

### 1. SpatialGatingHead — Spatially-Adaptive Base Model Blending

**File:** `sparc/models/neural_meta.py`

**New submodule:** `SpatialGatingHead` (inner class or standalone, used only by `SPARCMetaLearner`)

**Inputs:**
- `h_spatial` — `(N, H)` from `SparseSpatialAttention` (already computed in `forward()`)
- `h_trunk` — `(N, H)` from `trunk_fusion` (already computed in `forward()`)
- `surrogate_std` — `(N, 1)` std of the 3 raw surrogate predictions per cell; communicates cross-model disagreement magnitude

**Architecture:**
```
cat([h_spatial, h_trunk, surrogate_std])  →  (N, 2H+1)
  → Linear(2H+1, H//2)
  → GELU
  → LayerNorm(H//2)
  → Linear(H//2, 3)
  → Softmax(dim=-1)                        →  gate_weights (N, 3)
```

**Weighted blend:**
```
blend = (gate_weights * base_preds).sum(dim=-1, keepdim=True)   (N, 1)
h_base = blend_proj(blend)                                        (N, H)
```
where `blend_proj = Linear(1, H)`.

**Residual / backward compatibility:**
- `SpatialGatingHead` linear layers are Xavier-initialized with gain=0.01 → gates start near-uniform (softmax ≈ [1/3, 1/3, 1/3]).
- `blend_proj` is zero-initialized → `h_base` starts near zero, so existing `base_enc` residual carries the signal at t=0.
- `base_enc` is retained as a residual stream: `h_base_final = h_base_gated + 0 * base_enc(base_preds)` with a learnable scalar `gate_residual_weight` initialized to 0, allowing gradients to gradually engage the gated path.
- Result: existing checkpoints load without error; the gated path activates progressively during fine-tuning.

**Interpretability output:** `gate_weights (N, 3)` is returned from `forward()` alongside existing `attn_weights`. Maps to `["mgwr", "ggpgam", "gwrf"]` in surrogate order. Saved as `stage2/spatial_gate_weights.npz` for visualization.

**Fusion layer unchanged:** `fusion` still receives `cat([h_trunk, h_base_final, h_spatial], dim=-1)` → `(N, 3H)` — shape preserved.

### 2. GWRF Anisotropy

**File:** `sparc/models/gwrf.py`

**Current state:** `GWRFModel` stores `self.kernel_field` but `fit()` builds a `BallTree(coords)` for isotropic Euclidean neighbor selection and fits local RFs with uniform sample weights.

**Change:** When `self.kernel_field is not None` and at least one predictor `is_anisotropic`:
1. **Neighbor selection:** Replace `BallTree` Euclidean k-NN with anisotropic effective-distance ranking. For each fit point, compute `anisotropic_distance(dx, dy, κ_x, κ_y, θ)` (geometric mean across anisotropic predictors, same as GWR `_per_predictor_anisotropic_weights()`). Select the `k_neighbors` nearest in this metric.
2. **Sample weights:** Pass Matérn weights `matern_kernel_weights(d_aniso, kappa=1.0, nu=nu)` as `sample_weight` to `RandomForestRegressor.fit()`. RFs already support `sample_weight` — no architecture change.

**Fallback:** When `kernel_field is None` or no predictor is anisotropic, behavior is byte-identical to current.

**Performance note:** Anisotropic distance computation is `O(N · k · P)` where P = n_predictors. For typical SPARC datasets (N ≤ 10k, k = 100, P ≤ 10) this is trivially fast compared to RF fitting.

### 3. GWRF Per-Cell Prediction Uncertainty

**File:** `sparc/models/gwrf.py`

**Method:** After fitting each local RF at a subsample location, collect per-tree predictions at that point. `std(tree_predictions)` is the epistemic uncertainty for that cell. For non-subsample locations (interpolated via nearest-submodel assignment), propagate the nearest submodel's uncertainty as a lower bound.

**New method:** `predict_with_uncertainty(X, coords) → (predictions, std_per_cell)`

**Routing:**
1. **Divergence audit:** `causal_validation.py` calls `gwrf_model.predict_with_uncertainty()` and passes `gwr_std=std_per_cell` to `divergence_audit_for_all()`. This activates the posterior-aware flagging path: `|β − τ| > 2·√(σ_GWRF² + σ_CATE²)` instead of the current global-std fallback.
2. **SpatialGatingHead:** `surrogate_std` is computed as `std(base_preds, dim=-1, keepdim=True)` in `_forward_surrogates()` — this is cross-model disagreement (different signal from per-model uncertainty), and is the primary gating signal. GWRF std is not directly fed to the gate; it influences the gate indirectly by widening GWRF's prediction variance, which increases `surrogate_std` when GWRF disagrees with the others.

---

## Technical Approach

### Plumbing: `surrogate_std` through the training loop

**`_forward_surrogates()` in `v2_neural_training.py`:**
```python
# Before: returns base_preds (N, 3)
# After: returns (base_preds, surrogate_std)
gwr_pred, _ = surrogates["gwr"](physics_feats, spatial_feats)
gwrf_pred   = surrogates["gwrf"](...)
ggpgam_pred = surrogates["ggpgam"](physics_feats, spatial_feats)
base_preds = torch.stack([gwr_pred, gwrf_pred, ggpgam_pred], dim=-1)  # (N, 3)
surrogate_std = base_preds.std(dim=-1, keepdim=True)                  # (N, 1)
return base_preds, surrogate_std
```

**`SPARCMetaLearner.forward()` signature change:**
```python
def forward(
    self,
    base_preds: torch.Tensor,
    physics_feats: torch.Tensor,
    X_spatial: torch.Tensor,
    coords: torch.Tensor,
    knn_index: torch.Tensor,
    alpha: torch.Tensor,
    time_idx: torch.Tensor | None = None,
    surrogate_std: torch.Tensor | None = None,   # NEW — optional
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    # Returns: T_pred, exceedance, attn_weights, gate_weights (NEW)
```

`surrogate_std=None` defaults to `torch.zeros(N, 1, device=base_preds.device)` — neutral gating, backward-compatible with any caller that omits it.

### File Changes Summary

| File | Change Type | Summary |
|------|-------------|---------|
| `sparc/models/neural_meta.py` | Edit | Add `SpatialGatingHead`, `blend_proj`, `gate_residual_weight`; update `forward()` signature and return |
| `sparc/models/gwrf.py` | Edit | Anisotropic neighbor selection + weighting; add `predict_with_uncertainty()` |
| `sparc/run/v2_neural_training.py` | Edit | `_forward_surrogates()` returns `(base_preds, surrogate_std)`; pass `surrogate_std` to `model.forward()` |
| `sparc/run/causal_validation.py` | Edit | Call `gwrf_model.predict_with_uncertainty()`; pass `gwr_std` to `divergence_audit_for_all()` |

---

## Acceptance Criteria

1. **SpatialGatingHead:** `gate_weights.sum(dim=-1)` ≈ 1.0 for all N cells (softmax constraint). At init, all weights ≈ 0.333. After training on a heterogeneous dataset, gate weights show spatial variation (std > 0.05 across cells for at least one model).
2. **Backward compatibility:** Loading an existing `neural_meta.pt` checkpoint and running inference produces numerically identical output (gate starts neutral, residual weight = 0).
3. **GWRF anisotropy:** With a synthetic dataset where the true range is 2× longer in the x-direction, GWRF with anisotropy assigns higher weights to x-direction neighbors and achieves lower OOF RMSE than isotropic GWRF.
4. **GWRF uncertainty:** `predict_with_uncertainty()` returns `std_per_cell` of shape `(N,)` with all values ≥ 0. Values are higher in regions where the local RF trees disagree (high spatial heterogeneity), lower in smooth regions.
5. **Divergence audit:** With GWRF `std_per_cell` passed as `gwr_std`, `DivergenceReport.flagged_fraction` differs from the current global-std baseline (confirms the posterior-aware path is active).
6. **Gate weights saved:** `stage2/spatial_gate_weights.npz` written with arrays `mgwr`, `ggpgam`, `gwrf` each of shape `(N,)`.

## Open Questions

- None — all major decisions resolved.
