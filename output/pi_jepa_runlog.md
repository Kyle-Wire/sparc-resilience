# PI-JEPA Run Log

One line per run. Format: `date | flags | morning | midday | evening | sum | vs-baseline | decision | note`

## Baseline (σ not yet established — Stage 0 artifacts needed)

| date | flags | morning | midday | evening | sum | vs-σ | decision | note |
|------|-------|---------|--------|---------|-----|------|----------|------|
| 2026-06-04 | default (seed=42, --skip-stages) | -0.064 | +0.086 | -0.115 | -0.093 | N/A | BASELINE-FAIL | High variance, thread non-determinism, sum far below 1.277 target |
| 2026-06-04 | seed=42 CUDA GPU --skip-stages | TBD | TBD | TBD | TBD | N/A | PENDING | pi-jepa-dev branch, seeding fixes applied |

## B1 — Lock baseline (3-seed runs required)

> Status: NOT YET RUN. Requires seeding fixes (done on pi-jepa-dev) + consistent run conditions.
> Run: `python scripts/train_multicity_jepa.py --skip-collect --skip-stages` x3 (seeds 42, 43, 44 via jepa.seed in config)

## B2 — Anisotropy audit

> Status: DATA MISSING. Stage 0 Matérn + anisotropy artifacts not generated for any training city.
> Run: `python scripts/train_multicity_jepa.py --skip-collect --stages 0` to generate Stage 0 artifacts.
> Then: `python scripts/audit_anisotropy.py`

## B3 — Trunk embedding PCA

> Status: NOT YET RUN. Run after B1 baseline is locked.

## B4 — Per-window residual decomposition

> Status: NOT YET RUN. Run after B1 baseline is locked.

---

*Format for future entries:*
```
| YYYY-MM-DD | flag_delta | morning_corr | midday_corr | evening_corr | sum_corr | (sum-baseline)/sigma | KEEP/REVERT | one-sentence why |
```
