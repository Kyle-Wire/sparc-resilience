# ADR-0001: Anisotropic Spatial Patch Masking (PI-JEPA I1/I2) Abandoned

**Status:** Accepted  
**Date:** 2026-06-09

## Context

The PI-JEPA spec (PI_JEPA_AGENT_SPEC.md) proposed two masking interventions:

- **I2** — correlation-length-scaled patch radius (`range_scaled_radius: true`)
- **I1** — anisotropic elliptical patch mask shaped by Stage 0 Matérn eccentricity (`anisotropic_mask: true`)

The Stage 0 anisotropy audit (B2) returned **V-DIR-UNRELIABLE-DOMINANT** (60.4% of city×variable pairs have unreliable θ direction; only 12.5% are V-DIR-GOOD). This already signalled that full I1 with rotation was unjustified.

Three implementation variants were ablated against the locked baseline (mean sum_corr = 1.397, σ = 0.024, threshold = 1.445):

| run | variant | sum_corr | decision |
|---|---|---|---|
| I2-degen | range_scaled_radius=true, global coords | 1.383 | REVERT |
| I2v2 | range_scaled_radius=true, city-local coords | 0.648 (-3.1σ) | REVERT |
| I2v3 | range_scaled_radius=true + per_city_batching | 0.967 (-21σ) | REVERT |
| I1 | anisotropic_mask=true (ecc=1.358) | 0.966 (-21σ) | REVERT |

All four variants degraded performance, with I1 and I2v3 catastrophically so (-21σ).

## Decision

**Do not wire `anisotropic_patch_mask` into the training pipeline.** The function remains in `sparc/training/jepa_loss.py` as a library function (it passes its own unit tests) but:

1. `_JEPAConfig` will **not** gain `anisotropic_mask` or `range_scaled_radius` fields.
2. `FoldState` will **not** carry a `city_aniso_payload` field for the masking path.
3. Future architecture reviews should **not** re-suggest wiring I1/I2 masking as a training improvement — the ablation evidence is conclusive at this city count.

## Consequences

- `anisotropic_patch_mask` in `jepa_loss.py` is retained as a tested utility (may be useful for E4 per-variable masking in a future higher-data regime).
- C2 from the June 2026 architecture review is permanently closed.
- The 1.533 hybrid zero-shot+few-shot result (no anisotropic masking) remains the performance ceiling to beat.
- The E3 energy-balance pretext head (A2 in the runlog) is the next candidate to ablate.
