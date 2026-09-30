# ADR-0002: Limit ADR-0001's scope, and rebuild PI-JEPA as a gated spatial encoder

**Status:** Proposed
**Date:** 2026-09-30
**Amends:** ADR-0001 (anisotropic spatial patch masking abandoned)

## Context

ADR-0001 abandoned I1 (anisotropic patch masks) and I2 (range-scaled patch radius) after four ablations degraded `sum_corr`. It also instructed future reviews not to re-suggest them.

A review of the architecture those ablations ran on shows that mask geometry could not have had a spatial effect.

**The trunk sees one pixel at a time.**
- In `scripts/train_multicity_jepa.py:162-176` the trunk is a per-pixel MLP (`Linear → GELU → Linear → GELU → Linear`).
- It never sees neighbouring pixels.

**The training step has no context → target prediction** (`scripts/train_multicity_jepa.py:805-814`):
- Masked rows are set to zero.
- The online trunk encodes each row independently.
- The predictor maps a pixel's online embedding to the same pixel's EMA-target embedding, and the loss covers all rows.
- So a masked pixel only ever sees a constant zero vector, and an unmasked pixel is plain self-distillation.

**Batches are not spatial neighbourhoods.** They are 4,096 random pixels drawn from about 15M pixels across 22 cities, with globally normalized coordinates. `configs/multicity_pilot.yml:426` itself notes that the default radius is "effectively random".

**Mask shape can only change which rows get zeroed.** The observed differences therefore reflect how many rows were zeroed and seed variance, not spatial learning. The noise level makes this worse:
- The keep-threshold σ = 0.024 came from `--skip-pretrain` runs, which vary only the head initialization.
- Trunk-retrain variance observed in B6 was σ ≈ 0.34.

**The "physics-informed" part is not physics.** The A2 head predicts `1 − albedo` (`:819-828`), which is an input feature. That is feature reconstruction, not a physical constraint.

## Decision

1. **ADR-0001's conclusion is limited** to the per-pixel architecture it was tested on. It does not bind a spatial encoder.
2. **The current JEPA code is frozen as reference.** No further tuning of the per-pixel trunk, its masks, or its auxiliary heads.
3. **PI-JEPA is rebuilt in roadmap phase P7** (`docs/roadmap/CORE_ROADMAP.md`) as a real spatial JEPA, and only after the single-city core (P0–P6) works:
   - **Encoder:** a ViT/CNN over raster tiles, with tokens that are pixel patches. Each batch holds tiles from a single city.
   - **Objective:** I-JEPA style. The predictor receives context-token embeddings plus positional mask tokens, predicts target-block embeddings, and the loss applies only to masked targets.
   - **Pretraining data:** unlabeled rasters from many cities. The inputs are nationwide, so labels are not the bottleneck.
   - **Physics-informed targets and masks:**
     - Target-block size is scaled to the S1 influence range and elongated along the ERA5 wind direction. This is where I1/I2 become testable.
     - Auxiliary heads predict **non-input** physical fields: the physics-model ΔT (label-free) and held-out Landsat LST.
   - **Integration:** the encoder output fills the stacker's embedding slot (S3). It is not a parallel pipeline.
4. **Gate.** The encoder is kept only if it beats the correlogram-scaled focal-feature baseline in rotating leave-one-city-out evaluation (at least 5 holdouts). The margin must exceed σ measured with **full retrain** over at least 3 seeds.

## Consequences

- The I1/I2/B6/E5 runlog results stay as a historical record, labelled *not diagnostic of spatial masking*.
- `anisotropic_patch_mask` and `spatial_patch_mask` in `sparc/training/jepa_loss.py` may be reused for token-level masking in the rebuilt encoder.
- Until P7, the headline multi-city numbers (`sum_corr` 1.533 / 1.544) are not treated as the baseline to beat. They were tuned against the same holdout city. A new locked test set and protocol are defined in roadmap P0.
