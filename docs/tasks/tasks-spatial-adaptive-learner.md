# Tasks: Spatial Adaptive Learner Upgrades

Related PRD: docs/prd/prd-spatial-adaptive-learner.md

---

## Tasks

### S1 — GWRF Anisotropy (no dependencies, ship first)

- [ ] **S1a** — Add anisotropic neighbor selection to `GWRFModel.fit()`
  - When `self.kernel_field is not None` and any predictor `is_anisotropic`, compute geometric-mean Matérn anisotropic distances for each fit point (mirrors `GWRModel._per_predictor_anisotropic_weights()`)
  - Replace `BallTree` k-NN with anisotropic-distance ranked selection
  - Pass resulting Matérn weights as `sample_weight` to `RandomForestRegressor.fit()`
  - Fallback to current Euclidean behavior when `kernel_field is None`

- [ ] **S1b** — Add `predict_with_uncertainty()` to `GWRFModel`
  - Collect per-tree predictions at each cell: `np.array([t.predict(X_local) for t in rf.estimators_])`
  - Return `(mean_predictions, std_per_cell)` — shape `(N,)` each
  - For interpolated (non-subsample) cells, propagate nearest submodel's std

### S2 — Surrogate std plumbing (prerequisite for S3)

- [ ] **S2a** — Update `_forward_surrogates()` in `sparc/run/v2_neural_training.py`
  - Return `(base_preds, surrogate_std)` instead of just `base_preds`
  - `surrogate_std = base_preds.std(dim=-1, keepdim=True)`  — `(N, 1)`
  - Update all call sites in the training loop to unpack the tuple

- [ ] **S2b** — Update `SPARCMetaLearner.forward()` signature
  - Add `surrogate_std: torch.Tensor | None = None` as last optional param
  - Default: `torch.zeros(N, 1, device=base_preds.device)`
  - Update `encode()` and `decode()` accordingly
  - Add `gate_weights` to return tuple: `(T_pred, exceedance, attn_weights, gate_weights)`

### S3 — SpatialGatingHead (depends on S2)

- [ ] **S3a** — Implement `SpatialGatingHead` in `sparc/models/neural_meta.py`
  - Input: `cat([h_spatial, h_trunk, surrogate_std])` → `(N, 2H+1)`
  - Architecture: `Linear → GELU → LayerNorm → Linear → Softmax(dim=-1)` → `(N, 3)`
  - Xavier init with gain=0.01 → near-uniform gates at t=0

- [ ] **S3b** — Wire gating into `SPARCMetaLearner`
  - Add `self.spatial_gate = SpatialGatingHead(hidden_dim, n_base_models=3)`
  - Add `self.blend_proj = Linear(1, hidden_dim)` — zero-initialized
  - Add `self.gate_residual_weight = nn.Parameter(torch.zeros(1))`
  - In `forward()`: compute `gate_weights`, `blend`, `h_base_gated`; combine with residual `base_enc`
  - Final: `h_base_final = h_base_gated + gate_residual_weight * base_enc(base_preds)`
  - Confirm `fusion` input shape `(N, 3H)` unchanged

- [ ] **S3c** — Save gate weights as artifact
  - In `v2_neural_training.py` post-training inference pass, collect `gate_weights`
  - Save as `stage2/spatial_gate_weights.npz` with keys `mgwr`, `ggpgam`, `gwrf`
  - Register in artifact store

### S4 — Route GWRF uncertainty to divergence audit (depends on S1b)

- [ ] **S4a** — Call `predict_with_uncertainty()` in `causal_validation.py`
  - After Stage 2 GWR/GWRF predictions are loaded for the divergence audit
  - Pass `gwr_std=gwrf_std_per_cell` to `divergence_audit_for_all()`

### Tests

- [ ] **S5** — Unit tests
  - `test_gwrf_anisotropy.py`: synthetic dataset with 2× x-direction range; assert anisotropic GWRF OOF RMSE < isotropic
  - `test_gwrf_uncertainty.py`: assert `predict_with_uncertainty()` returns shape `(N,)`, all ≥ 0, higher std in heterogeneous region
  - `test_spatial_gating.py`: assert `gate_weights.sum(dim=-1) ≈ 1`; assert at init weights ≈ 0.333; assert existing checkpoint loads without error
  - `test_divergence_audit_posterior.py`: assert `flagged_fraction` differs when `gwr_std` is provided vs None
