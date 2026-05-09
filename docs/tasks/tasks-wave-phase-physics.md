# Tasks: Wave-Phase Physics

Related PRD: docs/prd/prd-wave-phase-physics.md

---

## Phase A — Gradient Direction Regulariser

### Backend

- [ ] A-1  Create `sparc/physics/regime_classifier.py` — `classify_wave_regime(domain, predictors) → WaveRegime`
- [ ] A-2  Add `gradient_direction(field, neighbor_idx, h)` operator to `sparc/physics/pde_operators.py`
- [ ] A-3  Add `PDELossWeights.phase_alignment` field and `_phase_alignment_term()` to `sparc/physics/pde_loss.py`; slot into `_ACTIVATION_SCHEDULE` at offset 20
- [ ] A-4  Add `solar_metadata` block to `project.yml` schema; wire `lambda_phase_alignment = 0.0` when absent
- [ ] A-5  Add `source_coherence_score()` to `sparc/physics/pde_operators.py` (or new `sparc/physics/source_coherence.py`) and write `source_coherence` artifact in Stage 0/2 analysis
- [ ] A-6  Add `albedo_weighted_neighbor_index` computed feature to UHI Stage 0 preprocessing
- [ ] A-7  Update `solar_metadata` parsing in `sparc/config/` to compute solar azimuth from lat/lon/datetime using stdlib `datetime` + simple solar geometry (no external API)
- [ ] A-8  Update `kappa_ratio_summary` / `divergence_audit` artifact to include `wave_regime` and `phase_alignment_residual` fields
- [ ] A-9  Update all domain `project.yml` templates to include commented-out `solar_metadata` block

### Tests (must pass before Phase B begins)

- [ ] A-T1  `tests/test_regime_classifier.py` — all 11 templates classified correctly; custom domain with `Albedo` predictor → mixed
- [ ] A-T2  `tests/test_phase_alignment_loss.py` — loss is zero when metadata absent; non-zero and decreasing for synthetic wave field with known source
- [ ] A-T3  `tests/test_source_coherence.py` — `rho_src` near 1.0 for radially symmetric field; near 0 for random gradient field
- [ ] A-T4  `tests/test_albedo_neighbor_index.py` — AWNI values decrease monotonically with distance from high-albedo point
- [ ] A-T5  All existing tests pass unchanged

### Desktop UI

- [ ] A-U1  Add "Source & Solar Metadata" conditional section to `sparc-desktop/src/components/project/ProjectMetadataForm.tsx`; visible for `uhi, noise, seismic, wildfire, air_quality, coastal, drought, forcesmip`
- [ ] A-U2  Add info pill (not warning) when solar fields are empty
- [ ] A-U3  Add `source_coherence` display to `KernelFieldPanel.tsx` — circular variance score + arrow indicating mean gradient direction
- [ ] A-U4  Update `ProjectConfig` type in `sparc-desktop/src/lib/types.ts` to include `solar_metadata`

---

## Phase B — Complex Phase Field

*Begin only after all Phase A tests pass and `phase_alignment_residual > 0.05` confirmed in at least one smoke test domain.*

### Backend

- [ ] B-1  Add `phase_head` to `ProcessRateNet` in `sparc/models/process_rate_net.py`; return `ProcessRateOutput(rate, phase)` named tuple; backward-compat shim for `.squeeze(-1)` callers
- [ ] B-2  Add `complex_gradient()` and `probability_current()` to `sparc/physics/pde_operators.py`
- [ ] B-3  Add `PDELossWeights.schrodinger_residual` and `_schrodinger_residual_term()` to `sparc/physics/pde_loss.py`; slot at offset 30
- [ ] B-4  Write `phase_field` artifact (phi_rad, J_x, J_y, J_magnitude, source_backprojection) in Stage 2 analysis when `training.phase_field: true`
- [ ] B-5  Implement source backprojection (gradient descent on $-\mathbf{J}$ from random seeds) in `sparc/physics/source_coherence.py`
- [ ] B-6  Add `training.phase_field: false` flag to all domain `project.yml` templates (commented note: "enable after Phase A signal confirmed")

### Tests

- [ ] B-T1  `tests/test_phase_field_net.py` — `ProcessRateNet` with phase head; legacy callers (`.squeeze(-1)`) unaffected; phase output bounded $(-\pi, \pi]$
- [ ] B-T2  `tests/test_schrodinger_loss.py` — residual is zero for analytic complex Matérn field; non-zero for random field
- [ ] B-T3  `tests/test_source_backprojection.py` — backprojection recovers known point source location within 2× grid spacing for synthetic radial field
- [ ] B-T4  All Phase A tests still pass

### Desktop UI

- [ ] B-U1  Create `sparc-desktop/src/components/insights/panels/PhaseFluxPanel.tsx` — arrow glyph overlay on spatial map; source candidate rings; graceful empty state when artifact absent
- [ ] B-U2  Register `PhaseFluxPanel` in insights panel registry
- [ ] B-U3  Add `phase_field` artifact type to `sparc-desktop/src/lib/types.ts`

---

## Status

All tasks: `[ ]` not started
