# SPARC — Research Backlog

**Maintained by:** synthesis agent  
**Last updated:** 2026-06-02 (fresh start — train22 complete, train23 queued; head covariate shift + seed reproducibility as primary active hypotheses; previous backlog archived to `docs/research/journal/decomissioned_backlogs/decomissioned_backlog_07.02.26.md`)

Items are ranked by impact/effort. Improvement agent picks the top `[ ]` item with complexity **low** or **medium**.

---

## Legend
- `[ ]` — todo
- `[~]` — in progress
- `[x]` — done
- `[!]` — blocked (reason noted inline)

Complexity: **low** = < 1 hour of focused edits | **medium** = half-day | **high** = multi-day / architectural

---

## ACTIVE QUEUE — train23 Phase (2026-06-02)

Context: Train22 complete. MAD outlier detection working. All-windows-positive correlation elusive.
Best ever: t19 morning +0.304, t16 midday +0.262 — never achieved simultaneously.
Root causes: (1) Head covariate shift — Raleigh CityHead predicts 95°F on Philly (trained at 84°F).
(2) Stochastic convergence — no seed set.

### T-1 Head-specific weight decay to reduce CityHead scale bias

- [x] **T-1 Add `head_weight_decay` param to `build_optimizer()` for city-specific head regularization** — complexity: **low** (~15 lines, 1 file)
  - ✅ Implemented: `head_weight_decay` kwarg added to `build_optimizer()`; applied to `base_enc`, `fusion`, `regression_head`, `exceedance_heads` param groups via per-group `weight_decay` override. Threaded through `_TrainingConfig`, all 4 `build_optimizer()` call sites, and `project.yml` (`head_weight_decay: 0.1`). Tests: 62 passed.
  - **Gap:** `build_optimizer()` in `sparc/training/optimizer.py` applies a single `weight_decay` uniformly across all param groups (trunk + city head). CityHead (`regression_head`, `fusion`, `base_enc`) memorizes the source city's temperature scale. Strong L2 on the head alone prevents scale bias without regularizing the trunk.
  - **Hypothesis for train23:** `weight_decay=0.1` on CityHead components shrinks the bias term in `regression_head` → head predicts closer to the trunk embedding's coordinate system rather than the Raleigh temperature scale.
  - **File:** `sparc/training/optimizer.py` — add `head_weight_decay: float = 1e-4` kwarg; apply to `base_enc`, `fusion`, `regression_head`, `exceedance_heads` param groups specifically.
  - **Config:** add `training.head_weight_decay: 0.1` to `project.yml` for train23.
  - **Risk:** Low — trunk weight decay unchanged (stays at default 1e-4). Head groups already separate in optimizer param list.
  - **Success criterion:** Train log shows `head param groups: weight_decay=0.1`; train23 OOF R² ≥ train22 best (0.304 morning); no NaN losses.

### T-2 Deterministic seed before CityHead initialization

- [x] **T-2 Set `torch.manual_seed(seed)` before model construction in `_exec_cv_fold`** — complexity: **low** (~5 lines, 1 file)
  - ✅ Implemented: `seed: int | None = None` added to `_TrainingConfig`; `training.seed` read from config; `torch.manual_seed(seed + fold_idx)` inserted before model instantiation in `_exec_cv_fold`, `FoldTrainer.build_models()`, and final retrain path. `project.yml` sets `seed: 42`. Tests: 55 passed.
  - **Gap:** No seed set before `SPARCMetaLearner` construction inside `_exec_cv_fold`. CityHead weight init is stochastic → different runs produce different starting biases → irreproducible convergence even at same hyperparameters.
  - **File:** `sparc/run/v2_neural_training.py` — in `_exec_cv_fold()`, before `model = SPARCMetaLearner(...)` (~line 680 area), add `torch.manual_seed(seed + fold_idx)`.
  - **Config:** Add `training.seed: 42` to `project.yml`. Read via `config.get("training", {}).get("seed", None)`.
  - **Risk:** Zero functional change when seed is None (existing behavior preserved).
  - **Success criterion:** Two identical config runs produce identical OOF predictions; train23 reproducible from train24 onward.

### T-3 Window-correlation consistency investigation

- [x] **T-3 Audit per-window head output distributions for covariate shift evidence** — complexity: **low** (diagnostic)
  - ✅ Implemented: `_rh_bias_before` captured before training loop; drift logged after training as `[T-3 head_bias_drift] fold=N before=X after=Y delta=Z`; per-window prediction/target mean gap logged at OOF time as `[T-3 window_scale] fold=N window=W pred_mean=X tgt_mean=Y gap=Z`. Guards for `time_idx is None` (temporal snapshots not configured).
  - **Gap:** t19 morning (+0.304) and t16 midday (+0.262) never achieved simultaneously. One window improving when another degrades suggests the CityHead is at a saddle.
  - **Investigation plan:**
    1. After train23, dump `model.regression_head.bias.data` per fold before/after head fine-tuning.
    2. Check `(predictions_mean_morning - predictions_mean_midday)` vs. `(target_mean_morning - target_mean_midday)`.
    3. If scale-gap varies by window → confirms covariate shift in temporal dimension.
  - **Hypothesis:** Philly morning temps smaller variance than Raleigh → head bias shrinks toward morning but overshoots midday.
  - **Success criterion:** Log shows `head_bias drift = {before: X, after: Y}` per fold per window; confirms or rejects hypothesis.

### T-4 Commit current stable work

- [ ] **T-4 Commit everything since 62b2f02 (train13)** — complexity: **low** (git operations)
  - **Gap:** Nothing committed since train13. All architecture work (FoldTrainer, preprocessing pipeline, ScenarioExecutor, data collection refactor, surrogate gate feedback, CityReplayState, LambdaOptimizer, route modules) is uncommitted.
  - **Action:** Once train23 shows stable improvement, commit. Suggested: `feat: train16-22 architecture — head gate, replay, meta-lambda, route extraction`
  - **Caution:** Verify all tests pass first (247/248 last known state, 1 intentional skip).

---

## Research Queue — JEPA / Spatial-SSL (2026-06-02)

- [ ] **R-1 V-JEPA 2 review + frozen encoder probing best practices** — complexity: **low** (research only)
  - **Questions:**
    1. V-JEPA 2 (Meta, 2025): what changed from V-JEPA 1? Does the online/target encoder update schedule apply to SPARC's `EMATrunk` pattern?
    2. Frozen encoder probing: linear probe vs. head-only fine-tuning for cross-city geographic regression?
    3. Few-shot geo transfer: minimum target-city samples to fine-tune CityHead given well-trained trunk?
  - **SPARC connection:** Informs EMATrunk momentum setting in `sparc/training/jepa_loss.py`; informs optimal head fine-tuning strategy relevant to T-1.
  - **Output:** Summary added to `docs/research/journal/` next session.

---

## Blue-Sky (2026-06-02)

### BS-A Add `coral_window_alignment()` to ewc.py for CityHead temporal covariate shift

- [ ] **BS-A Add `coral_window_alignment()` to `sparc/training/ewc.py`** — complexity: **medium** (~30 lines, 2 files)
  - **Root cause targeted:** CityHead regression_head bias cannot simultaneously compensate morning vs. midday covariate shift — reaches a saddle. Single scalar bias ≠ two distinct window distributions.
  - **Formula:** `L_CORAL = ‖Cov(src_window_k) - Cov(tgt_window_k)‖²_F / (4d²)` summed over windows k (Sun & Saenko 2016). No kernel bandwidth to tune. `O(d²)` per window.
  - **Infrastructure seam:** `wasserstein_trunk_alignment()` in `sparc/training/ewc.py` (line 180) — CORAL adds a parallel function in the same file.
  - **Implementation sketch:**
    ```python
    def coral_window_alignment(
        src_activations: torch.Tensor,   # (B, D) — source city window
        tgt_activations: torch.Tensor,   # (B, D) — target city window
    ) -> torch.Tensor:
        d = src_activations.shape[1]
        src_c = _centered_cov(src_activations)   # (D, D)
        tgt_c = _centered_cov(tgt_activations)   # (D, D)
        diff = src_c - tgt_c
        return (diff * diff).sum() / (4.0 * d * d)

    def _centered_cov(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return x.T @ x / (x.shape[0] - 1).clamp_min(1)
    ```
  - **Wiring:** `sparc/run/v2_neural_training.py` head fine-tune phase — read `config["continual"].get("coral_lambda", 0.0)`, accumulate over time_idx windows, add to head loss.
  - **Config:** `continual.coral_lambda: 0.01` in `project.yml`.
  - **Risk:** Medium — requires window-tagged activations during head fine-tune; need to confirm `time_idx` is accessible in that phase.
  - **Success criterion:** Morning + midday OOF R² both positive in train24; `coral_loss` logged per epoch.
  - **Depends on:** T-3 (audit confirms covariate shift hypothesis).

### BS-B Replace linear EMA schedule with cosine in `EMATrunk.current_tau()`

- [x] **BS-B Replace linear τ interpolation with cosine schedule in `sparc/models/ema_trunk.py`** — complexity: **low** (~3 lines, 1 file)
  - ✅ Implemented: `import math` added; line 137 replaced with cosine formula `frac = (1.0 - math.cos(math.pi * step / float(self.warmup_steps))) / 2.0`. Boundary values unchanged (step=0 → tau_start, step=warmup_steps → tau_end). Midpoint now biased toward tau_start (slow ramp up), graceful late-phase freeze.
  - **Root cause targeted:** `EMATrunk.current_tau()` (line 129) uses linear interpolation `tau_start + (tau_end - tau_start) * frac`. V-JEPA 2 (Bardes et al., arXiv:2506.09985) uses cosine schedule τ: 0.996→1.0 — smooth S-curve slows early thrashing, graceful late-phase freeze.
  - **Formula:** `τ_t = τ_start + (τ_end − τ_start) * (1 − cos(π * step / warmup_steps)) / 2`
  - **Current code (line 136–137):**
    ```python
    frac = step / float(self.warmup_steps)
    return self.tau_start + (self.tau_end - self.tau_start) * frac
    ```
  - **Proposed replacement:**
    ```python
    import math
    frac = (1.0 - math.cos(math.pi * step / float(self.warmup_steps))) / 2.0
    return self.tau_start + (self.tau_end - self.tau_start) * frac
    ```
  - **Files:** `sparc/models/ema_trunk.py` only. No config changes needed (existing `tau_start`/`tau_end`/`warmup_steps` still apply).
  - **Risk:** Minimal — same boundary values at step=0 and step=warmup_steps. Midpoint behavior only changes. Fully backward compatible.
  - **Success criterion:** `current_tau()` test confirms cosine shape (tau at step=warmup_steps//2 closer to tau_start than tau_end); JEPA pretrain loss curve smoother in train24 logs.
  - **Note:** derivatives.md incorrectly listed this under `jepa_loss.py` — actual location is `sparc/models/ema_trunk.py`.