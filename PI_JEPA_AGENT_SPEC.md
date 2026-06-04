# PI-JEPA Implementation Spec — Agent Instructions

**Target file:** `scripts/train_multicity_jepa.py` (the version that produced morning 0.502 / midday 0.457 / evening 0.318, sum 1.277)
**Goal:** Turn the isotropic, climate-blind JEPA pretext into a **Physics-Informed JEPA (PI-JEPA)** by injecting the per-city, per-variable Matérn anisotropy (κ_x, κ_y, θ, eccentricity) that Stage 0 *already computes and stores* but that Phase 1 currently ignores.

---

## 0. Operating rules (read first, do not skip)

These rules exist because this project has a documented history of run-to-run variance masking which change helped (train50→57 correlation collapse 0.36→0.01; train59 cosine-schedule regression). Violating them produces results that cannot be trusted.

1. **One change per run.** Never stack I1+I2+I3 or any experimental component in a single run. Each gated component is ablated alone against the locked baseline.
2. **Every new behavior is behind a config flag that defaults to current behavior.** A fresh checkout with no config edits must reproduce 1.277. If a flag defaults to "on," you have changed the baseline and broken reproducibility.
3. **A change is kept only if it beats baseline by ≥ 2× the measured run-to-run σ** (σ established in B1). Otherwise revert. Do not tune a losing change in place hoping it turns around.
4. **Degrade gracefully, non-fatally.** If a Stage 0 artifact is missing or malformed, log a single `[WARNING]` line and fall back to current isotropic behavior. Never crash the run. (Mirror the VSBA precedent — see §1.)
5. **Append one line to `output/pi_jepa_runlog.md` per run:** date, flag delta, per-window corr, sum, baseline-σ, decision (KEEP/REVERT), one-sentence why.
6. **ASCII-only in any terminal-facing print statement.** Stage 0 had repeated Windows CP437/CP1252 crashes from Unicode math symbols (κ, ν, θ, °, ±). Write `kappa`, `nu`, `theta`, `deg`, `+/-`. Use Unicode only in comments and the runlog markdown.
7. **Do not touch Phase 2 head training or the LOO ensemble logic in this work package.** PI-JEPA is a Phase-1-only intervention. Keeping the rest fixed is what lets you attribute any delta to the trunk.

---

## 1. Reference implementation to study before writing anything

The codebase already contains the exact pattern PI-JEPA needs. **Read these first:**

- `sparc/models/vsba.py` — `VariationalSpatialBlockAutoencoder`. Its `from_correlogram(...)` classmethod reads the Stage 0 Matérn payload and seeds `rho = 1/kappa_mean`, `nu` from the payload. Its `_matern_kij` evaluates Matérn(ν, ρ) for ν ∈ {0.5, 1.5, 2.5, ∞}. **This is the template** for how PI-JEPA reads anisotropy and how it degrades when the payload is absent.
- `sparc/run/v2_neural_training.py` — the VSBA pretext block (gated by `config["models"]["spatial_cv"]["vsba_fold_scoring"]`) shows the established convention: load Stage 0 payload from the artifact store, pass it to the new component, wrap in try/except as non-fatal.
- `sparc/training/jepa_loss.py` — `spatial_patch_mask(coords, mask_ratio, n_patches, ...)`. This is the function PI-JEPA's I1 generalizes. Read its exact signature and return type (boolean mask over batch rows) before writing `anisotropic_patch_mask`.
- `sparc/run/correlogram_analysis.py` and `sparc/run/anisotropy.py` — where `κ_x`, `κ_y`, `θ`, eccentricity, `r_hat`, ESS are produced and how `to_payload()` converts raw numpy `r_hat` arrays to floats. **Always read scalars from the payload dict, never from the raw result object** (documented bug: `MaternFitResult.r_hat` stores numpy arrays, `array < 1.05` raises "truth value ambiguous").

**Artifact keys (confirmed):**
- `("0", "correlogram_matern_fit")` — per-variable Matérn payload incl. r_hat, kappa posterior.
- Anisotropy payload contains `kappa_x`, `kappa_y`, `theta_unit`, eccentricity. **Known reliability issue:** `kappa_x`/`kappa_y`/`theta_unit` are "chronically ESS-limited with 400 fast-mode draws... `converged=False` is normal in fast mode." This is why B2 (the audit) gates everything.

Read access uses the artifact store read pattern already used by VSBA (`_safe_read_store("0", "...")` style — confirm the exact helper name in `v2_neural_training.py`).

---

## 2. BASELINE PHASE — mandatory, no model changes (B1–B4)

Do all four. Do not proceed to §3 until B2's verdict is recorded.

### B1 — Lock the baseline and measure noise
- Run the **unmodified** uploaded script 3×, seeds 42, 43, 44, identical config.
- Record per-window corr and sum for each. Compute mean and σ per window and for the sum.
- Write `output/diagnostics/baseline.json`: `{per_window_mean, per_window_std, sum_mean, sum_std, seeds}`.
- **Gate:** mean sum must land at 1.277 ± its own σ. If it does not, the harness is not reproducing the promising run — STOP and report before doing anything else.
- This σ is the bar every later change must clear by 2×.

### B2 — Anisotropy audit (decides whether full PI-JEPA is justified)
This is the most important baseline task. It answers: *is the directional structure you want to build on actually real and reliable in your cities?*

- Write `scripts/audit_anisotropy.py` (standalone, read-only).
- For each training city, run/locate its Stage 0 output and load `("0","correlogram_matern_fit")` + anisotropy payload.
- Emit a table (CSV + console) with columns: `city, variable, eccentricity, theta_deg, kappa_mean, r_hat_theta, ess_theta, converged`.
- Focus variables: `lst`, `pct_impervious`, `pct_canopy`, `svf`, `ndvi`, `albedo`.
- Compute three summary verdicts and print them explicitly:
  - **V-ISO:** fraction of (city,variable) pairs with eccentricity < 1.15 (effectively isotropic).
  - **V-DIR-UNRELIABLE:** fraction with eccentricity ≥ 1.15 but ess_theta below the Stage 0 ESS floor (theta direction not trustworthy).
  - **V-DIR-GOOD:** fraction with eccentricity ≥ 1.15 AND converged theta.
- **Branch the whole plan on this:**
  - If V-ISO dominates → directionality is low-value. Implement **I2 only** (range-scaling), skip I1's angle component, skip I3. The eccentricity≈1 case means I1 degenerates to the isotropic disk anyway.
  - If V-DIR-UNRELIABLE dominates → real anisotropy, untrustworthy angle. Implement I1 using **eccentricity magnitude only** (axis ratio), with θ set to 0 or to a low-confidence weak prior. Skip the θ-dependent rotation, or apply it with ESS-weighting. Skip I3 or run it with near-zero weight.
  - If V-DIR-GOOD covers the key thermal variables (lst, impervious, canopy) → full PI-JEPA (I1 with rotation + I3) is justified.
- Write `output/diagnostics/anisotropy_audit.csv` and a one-paragraph verdict to the runlog.

### B3 — Trunk embedding diagnostic
- After a baseline Phase 1, compute the frozen-trunk embedding for ~2000 stratified points per city; PCA to 2D; scatter colored by city; also write the per-city centroid pairwise distance matrix.
- Output: `output/diagnostics/trunk_pca.png` + `trunk_centroids.json`.
- **Interpretation:** if climatically distant cities (albuquerque_nm BSk vs boston_ma Dfb) interleave, the global-coordinate normalization in `run_jepa_pretraining` is mixing cities — which is exactly what I1's per-city masking fixes. If clusters are already clean and well-separated, the trunk is not the bottleneck and PI-JEPA gains will be modest; say so in the runlog.

### B4 — Per-window residual decomposition on the holdout
- For the baseline LOO predictions on the holdout city, compute spatial correlation between each window's residual and each land feature, especially `svf`, `bldg_height_mean`, `pct_impervious`.
- Output: `output/diagnostics/residual_feature_corr.csv`.
- **Interpretation:** if evening residuals correlate strongly with SVF/building height, evening's weakness is canyon-geometry representation — which anisotropic masking (I1) plausibly helps, and which also flags a possible SVF data-quality issue. If evening residuals don't track any feature, PI-JEPA alone won't rescue evening; note this.

**Deliverable of the baseline phase:** `output/diagnostics/` populated, runlog seeded with baseline σ and the B2 verdict. The B2 verdict selects which of I1/I2/I3 you build.



---

## 3. IMPLEMENT (100%) — gated, one at a time, in this order

Order matters: I2 establishes the per-city masking refactor that I1 builds on.

### I2 — Correlation-length-scaled patch radius, per city
**Why first / why safe:** uses the *best-converged* Stage 0 parameter (range from κ), not the worst (θ). Makes the masked fraction physically equivalent across cities of different extents (the Albuquerque-50km² vs Chicago-450km² problem). Pure refactor of mask geometry; no new loss.

**What to do:**
- In `run_jepa_pretraining`, the current global normalization concatenates all cities then normalizes coords once. Refactor so each city's coords and patch radius are handled **in that city's own physical units** before any cross-city batching. Practically: carry a per-row `city_id` alongside `X_t`/`C_t`, and compute a per-city patch radius (meters) from that city's Stage 0 effective range, converting to the masking routine's expected units per city.
- Config: `jepa.range_scaled_radius: false` (default off). When off, behavior is identical to today.
- Source the per-city range from Stage 0 (`first_zero_crossing` / effective range from the Matérn payload). If absent for a city, fall back to the current geometric default for that city only and log it.

**Best practice / research grounding:** block/patch size keyed to the spatial autocorrelation range is the standard decorrelation principle from Valavi et al. (2019, *blockCV*) and Roberts et al. (2017, *Ecography* cross-validation for structured data). The same Stage-0-range→block-size wiring already governs your Stage 2 spatial CV; I2 simply extends that logic to the self-supervised pretext.

**Ablation:** baseline vs `range_scaled_radius: true`, 3 seeds. KEEP iff sum gain ≥ 2σ.

### I1 — Anisotropic spatial patch mask (the headline component)
**Why:** the core of PI-JEPA. The pretext target stops being an isotropic disk and becomes an ellipse shaped like the city's actual thermal correlation footprint. The trunk is forced to learn directional spatial structure. Also closes the global-normalization bug (per-city masking means distant cities can't be normalized neighbors).

**What to do:**
- New function `anisotropic_patch_mask(coords, mask_ratio, n_patches, eccentricity, theta, min_patch_radius=None)` in `scripts/train_multicity_jepa.py` (or alongside `spatial_patch_mask` in `sparc/training/jepa_loss.py` if you prefer it library-side — match the existing test conventions there).
  - Generalize the disk to an ellipse: a point is "inside" patch center `c` if the **rotated, scaled** distance is ≤ 1:
    - rotate the displacement `(dx, dy)` by `-theta`, scale x-axis by `1/a` and y-axis by `1/b` where `a/b = eccentricity`, keep area comparable to the isotropic disk of the same `mask_ratio` (normalize `a·b` to the original `r²`).
  - When `eccentricity == 1.0`, the ellipse must reduce **exactly** to the current disk — verify this in a unit test (it guarantees the flag-off path is unchanged).
- Per the B2 verdict: pass real `theta` only in the V-DIR-GOOD case; pass `theta=0` (axis-aligned ellipse, eccentricity-only) in the V-DIR-UNRELIABLE case.
- Config: `jepa.anisotropic_mask: false` (default off). Pull per-city `eccentricity`/`theta` from the Stage 0 payload via the VSBA-style reader. Missing payload → fall back to isotropic `spatial_patch_mask` for that city, log once.

**Best practice / research grounding:**
- Anisotropic spatial covariance: Paciorek & Schervish (2006, *Environmetrics*) — nonstationary anisotropic covariance; this is the same κ_x/κ_y/θ ellipse your Stage 0 already fits and is cited in your own spatial-statistics resource.
- Masked self-supervised pretext: Assran et al. (2023, I-JEPA) and Bardes et al. (2024, V-JEPA) — your `jepa_loss` is built on these. PI-JEPA's novelty is making the **mask geometry physics-derived** rather than random/isotropic; that is not in the I-JEPA/V-JEPA literature and is the publishable contribution.
- Decorrelation rationale identical to I2 (Valavi 2019).

**Ablation:** baseline vs `anisotropic_mask: true` (with I2 on, since I1 depends on the per-city refactor), 3 seeds. KEEP iff ≥ 2σ. Also re-run B3 trunk PCA — the city clusters should be cleaner if the normalization bug was a factor.

### I3 — Anisotropy-alignment loss (soft, ESS-weighted) — only if B2 = V-DIR-GOOD
**Why:** your own architecture spec already lists "penalize isotropic fields where Stage 0 shows directional structure" as an intended Stage 2 constraint. I3 brings that to Phase 1: encourage the trunk's local latent gradient to align with the Stage 0 anisotropy direction, weighted by eccentricity **and** θ-confidence so it self-disables when θ is noisy.

**What to do:**
- Add an auxiliary term in the Phase 1 loop (near where `jepa_loss(...)` is called). On a KNN subsample of the batch (reuse the existing `_knn_subsample` pattern), compute the dominant direction of change in `h_pred` across neighbors and penalize misalignment with the city's `theta`.
- Weight `w_align = aniso_align_weight * clip(eccentricity - 1, 0, ·) * confidence(theta)` where `confidence` is a monotone function of ESS/r_hat from the payload (e.g. `min(1, ess_theta / ess_floor)`). When θ is unreliable, `confidence → 0` and the term vanishes harmlessly.
- Config: `jepa.aniso_align_weight: 0.0` (default off / zero).

**Best practice / research grounding:** physics-informed soft constraints in the loss (Raissi et al. 2019, PINNs) and curriculum staging of physics terms (Bengio et al. 2009; mirrored by your existing 10-term PDE curriculum). Keep I3 *soft* — the documented θ unreliability means a hard constraint would inject noise as physics.

**Ablation:** baseline (or best-kept I1/I2 stack) vs adding `aniso_align_weight` at a small value (start 0.01–0.05). 3 seeds. KEEP iff ≥ 2σ.

---

## 4. EXPLORE — only after the I-series is validated and attributed

Each is a separate gated experiment, ranked by expected value. Do not start these until I1/I2 (and I3 if applicable) are decided and logged.

- **E1 — Matérn-kernel-weighted JEPA target.** Replace binary masked/visible with a loss weighted by the Matérn covariance between masked and context points (soft kriging in latent space). Grounding: Matérn (1960); Gneiting et al. (2010). Cost: pairwise kernel evals on subsamples. Only pursue if I1 lands and you want a more principled successor.
- **E2 — θ as explicit trunk input.** Append `(eccentricity, sin θ, cos θ)` per cell to trunk input. Simpler than I1/I3 but bakes noisy θ into representations rather than into the task — run one ablation against I1, prefer I1 if comparable.
- **E3 — Energy-balance pretext head.** Auxiliary head predicting `(1 - albedo)` shortwave-absorption proxy from the trunk embedding (the disabled `energy_balance_weight` path in the other script version). Orthogonal physical prior; complements PI-JEPA's spatial prior. Grounding: Oke (1982) surface energy balance. Verify the albedo column index before wiring (the legacy code assumed column 0 = albedo — confirm against the actual feature order or it trains on garbage).
- **E4 — Per-variable anisotropy masking.** Use the V×V effective-range matrix to mask each feature channel with its own ellipse. High complexity, likely overkill at the current city count. Park unless E-series shows directionality is a large lever.

### E5 — EXTEND/FRONTIER — Intrinsic spatial frame: predict eigenmap structure instead of coordinates

**One-line thesis:** Stop treating the spatial graph's Laplacian eigenmaps as *input* side-information and
start treating a small set of low-frequency eigenmap coordinates as the JEPA *prediction target* — so the
trunk learns each location's intrinsic structural role in the connectivity graph (core / boundary / corridor)
rather than its arbitrary extrinsic lat/lon. This is a representation-frame change, not a feature addition.

**Tier note:** rated EXTEND for the within-city version (machinery exists, the target is already computed and
saved) but FRONTIER for the cross-city version, because making eigenvectors comparable across cities is an
unsolved-in-this-codebase problem. Implement the EXTEND half first; only attempt the FRONTIER half if it pays.

---

#### What exists today (and why this is a flip, not a build)

The pipeline already computes Laplacian eigenmaps from a KNN spatial-weights matrix and **saves them**:
- `laplacian_features.pkl` — the eigenvectors (the eigenmap coordinates per location)
- `laplacian_eigenvalues.pkl` — the eigenvalue spectrum (frequency ordering)
- `spatial_weights.pkl` — the spatial weights matrix the Laplacian was built from
- config: `laplacian.n_eigenmaps: 150`, `laplacian.k_for_swm`

Today these 150 eigenvectors are fed *in* as features (to OLS, and as PCA components to the meta-ensemble via
`meta_ensemble.include_laplacian_pca: true`). In that role they are a smarter coordinate system the model reads
from — but they are still on the input side, playing the same part as raw x/y. The +4.2 pp R² uplift in the
Providence study shows they carry real signal as input.

B4 moves them to the **output side** of the JEPA pretext. The target artifact already exists; the work is a new
prediction head and an auxiliary loss term, not new feature engineering.

---

#### The idea, precisely

In the current JEPA Phase 1 pretext, the trunk sees masked-context features and predicts the masked region's
*latent embedding* (aligned to the EMA target). B4 adds (or substitutes) a second prediction target: the masked
region's **eigenmap coordinates** — its position in the spectral embedding of the spatial graph.

Mechanically, add an auxiliary head `eigen_head: trunk_embedding -> R^m` that predicts the first `m`
low-frequency eigenmap coordinates of each masked cell, with an MSE (or cosine, see sign caveat) auxiliary loss
term gated behind `jepa.eigen_target_weight: 0.0`.

Why this is the right shape for the trunk to learn:
- Lat/lon is **extrinsic and arbitrary** — Philadelphia's coordinates carry no meaning for Chicago's model.
- Eigenmap position is **intrinsic** — it describes a location by its connectivity role in the spatial
  structure, which is a transferable concept. A trunk that predicts "where does this cell sit in the
  connectivity spectrum" is reasoning about structural role, not absolute place. For a multi-city model with a
  held-out city, that is exactly the inductive bias you want.

---

#### Research grounding

- Belkin & Niyogi (2003, *Neural Computation*) — Laplacian eigenmaps as the intrinsic low-dimensional geometry
  of a graph; the low eigenvectors are the smoothest functions over the domain.
- Griffith (2003) — Moran-eigenvector spatial filtering; eigenvectors of the spatial weights matrix as a
  meaningful spatial basis (the same basis already used as input here).
- Belkin & Niyogi spectral embedding + the diffusion-maps view (Coifman & Lafon 2006) — locations close in
  eigenmap space are close in diffusion/connectivity distance, not Euclidean distance.
- For the cross-city alignment problem: functional maps / spectral correspondence (Ovsjanikov et al. 2012) and
  the broader literature on aligning Laplacian eigenbases across different graphs.

---

#### The three hard problems the agent MUST confront (do not skip)

These are the difference between a real result and a self-deception.

1. **Eigenvectors are not comparable across cities out of the box.** Each city has its own graph, its own
   Laplacian, its own eigenvectors. Eigenvectors are defined only up to sign, and for repeated/near-degenerate
   eigenvalues up to rotation within the eigenspace. The k-th eigenvector of Philadelphia is NOT "the same axis"
   as the k-th of Chicago. So a naive "predict eigenmap coordinate k" target is per-city-arbitrary and will NOT
   transfer — which defeats the entire purpose. Two consequences:
   - **Sign/rotation invariance in the loss.** Use a loss that is invariant to eigenvector sign (and ideally to
     within-eigenspace rotation), e.g. predict the Gram matrix / pairwise eigen-distances between masked points
     rather than the raw signed coordinates, or align signs to a fixed reference per city before computing loss.
   - **Cross-city alignment (the FRONTIER half).** To make the *target* transferable, eigenbases must be aligned
     across cities (functional maps / Procrustes on low-frequency subspaces). This is the genuinely hard,
     possibly-unnecessary part — gate it behind its own flag and only attempt it if the within-city version
     earns its place first.

2. **High-frequency eigenvectors are unstable; only the low ones transfer.** Adding/removing a few nodes can
   reshuffle high-index eigenvectors substantially. The stable, semantically-meaningful "core vs. boundary vs.
   corridor" structure lives in the first ~10–20 eigenvectors. **Predict a small `m` (start m≈10–15), never the
   full 150.** The 150 are fine as input side-information but far too many — and far too noisy — as a stable
   prediction target. This is non-negotiable.

3. **It may be redundant with PI-JEPA.** PI-JEPA's anisotropic mask (I1) already teaches the trunk directional
   spatial structure. Predicting eigenmap position teaches connectivity structure. These overlap. The required
   ablation is not "B4 vs. coordinate baseline" — it is "B4 *on top of* the kept PI-JEPA result." The hypothesis
   is that B4 adds the **cross-city-transferable** piece that per-city PI-JEPA does not. If B4 doesn't beat the
   PI-JEPA-only result by ≥ 2σ, it is capturing the same structure through a different door and should be cut.

---

#### Staged implementation plan (build the cheap, decisive test first)

**Stage 1 — within-city sanity test (EXTEND, days).** Add `eigen_head` + auxiliary loss, gated by
`jepa.eigen_target_weight`. Use sign-invariant loss on `m≈10` low-frequency eigenmaps. Train and evaluate on a
SINGLE city (no cross-city transfer yet). The only question: does predicting intrinsic eigen-position as an
auxiliary pretext improve that city's held-out window correlations at all, vs. the same setup without the
eigen target? If it does not help even within one city, STOP — the expensive cross-city alignment is not worth
building. This is the decisive, cheap gate.

**Stage 2 — multi-city, per-city targets (EXTEND, ~1 week).** Run across cities with each city predicting its
*own* eigen-coordinates (still no cross-city alignment). Tests whether the auxiliary objective helps the shared
trunk's general spatial reasoning even when the target is per-city-defined. Ablate against PI-JEPA-only.

**Stage 3 — cross-city eigenbasis alignment (FRONTIER, weeks, only if Stages 1–2 paid).** Implement
functional-map / Procrustes alignment of the low-frequency eigen-subspaces across cities so the target becomes
transferable, then test whether held-out-city zero-shot correlation improves over PI-JEPA-only. This is the part
that could be genuinely novel and publishable — "intrinsic-spatial-frame self-supervised pretraining with
cross-city spectral alignment" is not, to my knowledge, in the spatial ML literature — but it is also the part
most likely to fail or to need real research iteration. Do not start it before the gates above pass.

---

#### Config and artifacts

- New config: `jepa.eigen_target_weight: 0.0` (default off), `jepa.eigen_target_m: 12` (low-freq count),
  `jepa.eigen_target_align: false` (Stage 3 cross-city alignment, default off).
- Reuse existing artifacts: read eigenmaps from `laplacian_features.pkl` and eigenvalues from
  `laplacian_eigenvalues.pkl` (ordering = frequency). Confirm these are produced per-city in the multi-city
  run before relying on them; if a city lacks them, degrade to no-op for that city and log once (same graceful
  contract as everything else).
- Keep distinct from C1 (TDA). C1 is the topology of the *thermal field* (sublevel sets of predicted
  temperature). B4 is the geometry of the *spatial graph* (eigenstructure of the connectivity Laplacian).
  Different object, different machinery — do not merge them in code or in the runlog.

---

#### Why this is worth doing despite the hard problems

If it works, it changes what the trunk fundamentally represents: from "a function of where you are" to "a
function of your structural role in the spatial system." That is the most direct realization of the long-stated
goal of moving SPARC's spatial representation from coordinate-based to structure-based. It is also the single
idea in this document most likely to improve **zero-shot transfer to held-out cities specifically**, because
intrinsic structural role is portable in a way absolute coordinates never are — and held-out-city transfer is
the metric the multi-city LOO setup actually measures. The within-city Stage-1 test costs only days and
decisively tells you whether the premise holds before any expensive alignment work. That asymmetry — cheap
decisive test, large potential upside — is exactly what makes it worth queuing.

**Scheduling note (per the operator's intent):** queue this as the first exploration item in the *next* agent
spec, AFTER the core PI-JEPA package (B1–B4 baseline, I1–I3) and the Category A plug-ins are validated. It needs
a stable, well-understood PI-JEPA baseline to ablate against — running it before PI-JEPA is settled would make
its lift impossible to attribute.
---

## 5. DO NOT (explicit guardrails)

- **Do not hard-constrain the trunk to Stage 0 θ.** θ is your least-converged parameter. Every directional component must degrade gracefully via ESS-weighting or eccentricity-only fallback.
- **Do not re-enable the cosine LR schedule** to "help" PI-JEPA. The train59 regression is real and unexplained; reintroducing it confounds the ablation.
- **Do not add Spatial MAE alongside JEPA** in this work package. Two competing self-supervised objectives with no attribution. Revisit only if B3 shows a badly-structured trunk and the I-series underdelivers.
- **Do not change batch_size / n_epochs** as part of a PI-JEPA ablation. The existing batch_size=4096 is documented as t52-proven for VICReg variance/covariance health; changing it confounds the geometry change you're testing.
- **Do not modify Phase 2 heads or LOO weighting here.** Out of scope; keep them fixed to isolate the trunk effect.

---

## 6. Decision flow (summary)

```
B1 lock baseline + sigma ─┐
B2 anisotropy audit ──────┤
B3 trunk PCA ─────────────┤──► verdict selects components
B4 residual decomposition ┘
                              │
                V-ISO ────────► I2 only
                V-DIR-UNRELIABLE ► I2 + I1(ecc-only)
                V-DIR-GOOD ─────► I2 + I1(rotated) + I3
                              │
        each component: gated flag, default off, 3-seed ablation,
        KEEP iff sum gain >= 2*sigma, runlog entry, else REVERT
                              │
        re-run B3/B4 after each KEEP to confirm trunk + evening improved
                              │
        only then ► E1..E4 (separate gated experiments)
```

## 7. Acceptance criteria for the whole package
- `output/diagnostics/` contains baseline.json, anisotropy_audit.csv, trunk_pca.png, trunk_centroids.json, residual_feature_corr.csv.
- `output/pi_jepa_runlog.md` has one honest line per run with KEEP/REVERT decisions.
- A fresh checkout with no config changes reproduces 1.277 ± σ (all PI-JEPA flags default off).
- Every kept change beat baseline by ≥ 2σ and has a unit test proving the flag-off path is byte-identical to current behavior (esp. the `eccentricity==1.0` reduces-to-disk test for I1).

## NOTE — Frozen vs. unfrozen trunk (decision: FROZEN for this work package)

**Use a frozen trunk during Phase 2.** Do not unfreeze as part of PI-JEPA.

**What the terms mean:** The trunk maps land features -> spatial representation; the
CityHeads decode it (+ ERA5) -> UHI anomaly. Frozen = trunk weights locked during
Phase 2, only heads train. Unfrozen = CAPA temperature gradients flow back into the
trunk, adapting the representation itself.

**Why frozen, grounded in our own runs:**
- Full unfreezing was tested and was catastrophic: train11 = -38 deg F bias,
  train12 = -261 deg F bias (code comment, lines ~576-580). Mechanism is catastrophic
  forgetting (McCloskey & Cohen 1989; French 1999).
- Our Phase 2 loop updates ONE shared trunk SEQUENTIALLY in-place across cities. An
  unfrozen trunk drifts toward whichever cities are processed last and forgets earlier
  ones. Because all LOO heads share that one final trunk state, the drift becomes a
  systematic bias rather than averaging out.
- Several Phase-1 cities (providence_ri, burlington_vt, wilmington_de) have NO CAPA
  labels and contribute to the trunk only via self-supervision. Unfreezing would bias
  the shared representation toward the labeled cities and silently drop the structure
  the label-free cities provided. Freezing preserves the all-cities representation.

**The cost of freezing (accepted):** The frozen trunk learned morphology spatially but
not thermally. The heads must bridge that gap. This is fine for morning (0.502) but
likely part of why evening lags (0.318). PI-JEPA's job is to make the *frozen* trunk's
representation richer (directionally physics-aware) so the heads have better material.
Frozen trunk and PI-JEPA are complementary, not in tension.

**Confound warning:** Unfreezing during a PI-JEPA ablation would make it impossible to
attribute a delta to the anisotropic mask vs. trunk drift. Keep frozen so the ablation
is clean.

**Middle path for LATER (not this package):** A per-city affine adapter (learned gamma
scale + beta shift, dim = hidden_dim, identity-init gamma=1/beta=0) inserted between the
frozen trunk and the heads. Trunk weights stay frozen -> forgetting is structurally
impossible -> but each city can thermally specialize the embedding. This is the
disciplined version of unfreezing and mirrors the existing few_shot.py "freeze trunk,
tune adapter" design (MAML inner loop, Finn et al. 2017).

**If true unfreezing is ever revisited:** only with EWC (Kirkpatrick et al. 2017) +
coreset replay (Lopez-Paz & Ranzato 2017), both of which exist in the repo
(sparc/training/ewc.py, replay.py) but are dormant when frozen. NOTE: replay is
currently BLOCKED by an interface mismatch (backlog 1.1-b) — not a quick toggle.