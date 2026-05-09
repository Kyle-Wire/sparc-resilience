# PRD: Wave-Phase Physics for SPARC Spatial Regression

## Problem Statement

SPARC's spatial regression pipeline uses a Matérn covariance kernel (via the screened-Poisson
SPDE) to characterise spatial correlation. The Matérn captures the *magnitude* envelope
$|\psi(\mathbf{x})|$ of the underlying field — equivalent to the ground-state amplitude of a
wave function after imaginary-time relaxation (Wick rotation: substitute $t \to -it$ in the heat
equation and recover the Schrödinger equation). What SPARC currently discards is the **phase**
$\Phi(\mathbf{x})$: the directional memory of how the field propagated to its observed
configuration.

This has concrete consequences:

- In **wave-propagation domains** (noise, seismic, wildfire, air quality) the underlying physics
  is $1/r$ amplitude decay from a point source, not screened-Poisson diffusion. The model fits
  the right coefficients but with the wrong PDE — extrapolation is unreliable and coefficient
  magnitudes are physically uninterpretable.
- In **radiative-diffusive mixed domains** (UHI, coastal, drought) key predictors — albedo,
  wave height, solar radiation — carry $1/r$ propagation physics that the purely diffusive loss
  does not encode.
- Across **all domains**, the gradient direction field of the predicted response $\hat{y}$ carries
  information about the spatial source structure that no current SPARC output exposes.

The Wick rotation bridge motivates a unified fix: treat the spatial field as a collapsed wave
function, enforce phase consistency as a loss term, and ultimately learn the full complex field
$\psi = T \cdot e^{i\Phi}$ to recover the probability current (spatial influence flux).

---

## Goals

1. Add a **gradient direction regulariser** (Phase A) that penalises spatial gradient fields
   inconsistent with the domain's expected propagation direction — implemented as a 9th PDE loss
   term, curriculum-staged, non-destructive to the existing pipeline.
2. Add a **complex phase field** (Phase B) via a new `ProcessRateNet` head, enabling full
   Schrödinger residual loss and a probability current output $\mathbf{J} = |\psi|^2 \nabla\Phi$
   — a spatial influence flux map.
3. Extend both phases across **all applicable domains** via an automatic regime classification
   (wave / mixed / diffusive) derived from domain name and declared predictors.
4. Add a **source coherence analysis** on the predicted target field — domain-agnostic, reports
   how coherently $\hat{y}$'s gradient points toward candidate source locations.
5. Update the **desktop UI** to collect solar/source metadata where required, with graceful
   info-level degradation when metadata is absent.

---

## Non-Goals

- Replacing the Matérn kernel in `KernelField` or `GWRModel` (a possible Phase C, not in scope).
- Modifying diffusive domains: `groundwater`, `stormwater`, `geotechnical` — screened-Poisson is
  correct physics for these.
- Complex-valued GWR bandwidth fitting or correlogram analysis.
- Real-time solar ephemeris API calls — static metadata fields only.

---

## Domain Regime Classification

Auto-detected at run time from `project.yml` `domain` + `predictors`. Stored in the
`divergence_audit` artifact as `wave_regime`.

### Regime 1 — Wave Propagation
Target variable IS the wave amplitude. $1/r$ attenuation is the primary spatial decay mechanism.

| Domain | Target | Phase term $\theta_{domain}$ | Auto-detectable source |
|---|---|---|---|
| `noise` | `Leq_dBA` | Source bearing from road geometry | Road coordinate in data |
| `seismic` | `PGA_g` | Epicenter direction | `epicenter_lat/lon` in `solar_metadata` |
| `wildfire` | `dNBR` | Ignition point + wind bearing | `wind_direction_deg` in metadata |
| `air_quality` | `PM25_ugm3` | Upwind bearing | `wind_direction_deg` in metadata |

### Regime 2 — Radiative-Diffusive Mixed
Target is diffusive but one or more predictors carry $1/r$ propagation physics.

| Domain | Key radiative predictor | Phase term $\theta_{domain}$ |
|---|---|---|
| `uhi` | `Albedo`, `NDVI` | Solar azimuth from lat/lon + datetime |
| `coastal` | `sig_wave_height_m`, `mean_wave_period_s` | Wave approach angle |
| `drought` | `solar_radiation_Wm2` | Solar azimuth |
| `forcesmip` | `solar_forcing_index`, `GHG_forcing_index` | Polar gradient direction |

Also activates for any `custom` domain that declares a predictor matching the regex
`albedo|reflectance|radiation|wave_height|solar`.

### Regime 3 — Diffusive (no wave physics)
`groundwater`, `stormwater`, `geotechnical`. All new loss terms remain at `lambda = 0.0`.

---

## Design Decisions

### D1 — Additive loss, not kernel replacement
Phase A and B are additive PDE loss terms slotted into the existing `_ACTIVATION_SCHEDULE` in
`pde_loss.py`. Legacy call sites are unchanged. Blast radius is confined to `pde_loss.py`,
`pde_operators.py`, `process_rate_net.py`, `project.yml` schema, and the UI metadata form.

### D2 — Domain-gated UI fields
`ProjectMetadataForm` renders a "Source & Solar Metadata" section only for wave/mixed domains.
Fields: `lat`, `lon`, `collection_datetime` (ISO-8601), `wind_direction_deg`, `source_bearing_deg`
(override for seismic/noise/wildfire). Absent metadata → `lambda_phase_alignment: 0.0` +
`wave_regime: "phase_disabled — metadata absent"` in audit artifact. Info pill in UI, not warning.

### D3 — UHI albedo neighbor effect as engineered feature
Rather than a separate loss term, a computed predictor `albedo_weighted_neighbor_index` is added
to UHI template predictors. Captures the $1/r$-weighted albedo influence from spatial neighbors.
Computed in Stage 0 preprocessing. Zero new loss infrastructure.

### D4 — Source coherence as a domain-agnostic diagnostic
Regardless of regime, after Stage 2 fitting, compute the gradient direction field of $\hat{y}$
and report a **source coherence score** $\rho_{src} \in [0, 1]$ measuring how consistently
gradients point toward a common source. When source location is known, also report angular
residual. Stored in a new `source_coherence` artifact. Displayed in the Kernel Field panel.

### D5 — Phase B gated on Phase A signal
Phase B (complex field) is implemented in the same PR sequence but its activation in production
templates is gated behind a `phase_field: enabled` flag in `project.yml`, defaulting `false`.
If Phase A's `phase_alignment_residual` in the audit artifact is $> 0.05$ (meaningful signal),
the user is prompted to enable Phase B.

### D6 — Sequential PRs with test suite between phases
Phase A ships and passes all tests. Phase B follows. No parallel development.

---

## Technical Approach

### Phase A — Gradient Direction Regulariser

**New `pde_operators.py` function:**
```python
def gradient_direction(field, neighbor_idx, h) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (phi_rad, grad_mag, valid) where phi_rad = atan2(df/dy, df/dx)."""
```

**New `pde_loss.py` term (`phase_alignment`):**
```python
L_phase = mean( circular_distance(phi_predicted, theta_domain) ** 2 * grad_mag_weight )
```
- `theta_domain`: scalar or per-point tensor, derived from solar azimuth or source bearing
- `grad_mag_weight`: gates the loss near saddle points where $|\nabla T| \approx 0$
  (threshold: `grad_mag < 0.01 * grad_mag.max()` → weight = 0)
- Activation offset: epoch 20 (after all existing 8 terms)
- Ramp: 5 epochs (matching `_RAMP_EPOCHS`)
- Default weight in `PDELossWeights`: `phase_alignment: 0.15`
- Zeroed automatically when `lambda_phase_alignment = 0.0` (metadata absent or diffusive regime)

**`project.yml` schema addition:**
```yaml
solar_metadata:                    # Optional. Enables phase_alignment loss term.
  lat: 41.827                      # Decimal degrees N
  lon: -71.400                     # Decimal degrees W (negative = west)
  collection_datetime: "2023-08-15T14:00:00"  # ISO-8601, local solar time
  wind_direction_deg: 225          # Meteorological convention (direction FROM, degrees)
  source_bearing_deg: null         # Override: explicit source bearing (noise/seismic/wildfire)
```

**`kappa_estimator.py` / new `regime_classifier.py`:**
```python
def classify_wave_regime(domain: str, predictors: list[str]) -> WaveRegime:
    """Returns WaveRegime(regime='wave'|'mixed'|'diffusive', phase_term_source=...)"""
```

**`albedo_weighted_neighbor_index` (UHI only):**
Added to Stage 0 preprocessing. For each point $j$:
$$\text{AWNI}_j = \sum_{i \neq j} \frac{\alpha_i}{r_{ij}}$$
where the sum is over the $k$-nearest spatial neighbors (configurable, default $k=8$).

**Source coherence diagnostic:**
```python
def source_coherence_score(df_dx, df_dy, valid, source_xy=None) -> dict:
    """
    Computes gradient direction field, circular variance (rho_src),
    and if source_xy given, mean angular residual from source direction.
    """
```
Stored as `source_coherence` artifact. Displayed in Kernel Field panel alongside anisotropy.

**Desktop UI (`ProjectMetadataForm.tsx`):**
- New conditional section "Source & Solar Metadata" rendered when
  `['uhi','noise','seismic','wildfire','air_quality','coastal','drought','forcesmip']
  .includes(domain)`
- Fields: Latitude, Longitude, Collection datetime (date+time picker), Wind direction (°),
  Source bearing override (°, optional)
- Info pill when fields empty: *"Phase alignment term inactive — add location data to enable"*
- Tooltip links to docs section on wave-phase physics

---

### Phase B — Complex Phase Field

**`ProcessRateNet` addition:**
New `phase_head: nn.Linear(hidden, 1)` outputs raw logit → `torch.tanh(x) * π` → $\Phi \in (-\pi, \pi]$.
Output shape: `(N, 1)` — returned alongside existing rate output as a named tuple
`ProcessRateOutput(rate, phase)`.

Backward-compatible: callers that only use `.rate` are unaffected.

**`pde_operators.py` additions:**
```python
def complex_gradient(T, Phi, neighbor_idx, h):
    """Gradient of psi = T * exp(i*Phi) in real+imaginary channels."""

def probability_current(T, Phi, neighbor_idx, h):
    """J = |psi|^2 * grad(Phi) — spatial influence flux vector field."""
```

**`pde_loss.py` new term (`schrodinger_residual`):**
```python
# Time-independent Schrödinger in imaginary time = steady-state heat eq. in complex plane
L_sch = || alpha * complex_laplacian(psi) - kappa^2 * psi ||^2_C
```
Activation offset: epoch 30. Weight: `schrodinger_residual: 0.10`.

**New artifact: `phase_field`**
```json
{
  "phi_rad": [...],          // per-point phase values
  "J_x": [...],              // probability current x-component
  "J_y": [...],              // probability current y-component
  "J_magnitude": [...],      // |J| at each point
  "source_backprojection": { // inferred source candidate(s)
    "method": "gradient_descent_on_neg_J",
    "candidates": [{"x": ..., "y": ..., "weight": ...}]
  }
}
```

**New UI panel: `PhaseFluxPanel`**
- Renders $\mathbf{J}$ as arrow glyphs overlaid on the spatial prediction map
- Arrow length ∝ $|\mathbf{J}|$, direction = $\arg(\mathbf{J})$
- Source candidate locations shown as pulsing rings
- Only visible when `phase_field` artifact present in run output

**`project.yml` opt-in flag:**
```yaml
training:
  phase_field: false   # Set true after Phase A confirms signal (phase_alignment_residual > 0.05)
```

---

## Acceptance Criteria

### Phase A
- [ ] `regime_classifier.py` correctly classifies all 11 templates (unit tested)
- [ ] `phase_alignment` loss term activates at epoch 20 for wave/mixed domains, stays zero for diffusive
- [ ] When `solar_metadata` absent: `lambda_phase_alignment = 0.0` in audit artifact, info pill visible in UI
- [ ] When `solar_metadata` present: `phase_alignment_residual` written to `divergence_audit` artifact
- [ ] `source_coherence` artifact produced for all domains regardless of regime
- [ ] `albedo_weighted_neighbor_index` computed and available as predictor in UHI template
- [ ] `ProjectMetadataForm` shows/hides solar section based on domain
- [ ] All existing tests pass unchanged
- [ ] New tests: `test_phase_alignment_loss.py`, `test_regime_classifier.py`, `test_source_coherence.py`

### Phase B (gated on Phase A passing)
- [ ] `ProcessRateNet` `forward()` returns `ProcessRateOutput(rate, phase)` — legacy `.squeeze(-1)` callers unaffected
- [ ] `phase_field` artifact written when `training.phase_field: true`
- [ ] `PhaseFluxPanel` renders without crashing when artifact absent (graceful empty state)
- [ ] `schrodinger_residual` loss term stays zero when `phase_field: false`
- [ ] Source backprojection candidates visually co-locate with known sources in smoke tests (noise: road; seismic: epicenter)
- [ ] New tests: `test_phase_field_net.py`, `test_schrodinger_loss.py`, `test_phase_flux_panel.tsx`

---

## Open Questions

1. **Circular distance metric** — use `1 - cos(Δφ)` (smooth, zero at 0) or `|Δφ|` (linear, kink at π)?
   Recommend `1 - cos(Δφ)` to avoid gradient discontinuity at ±π.
2. **AWNI radius** — fixed $k=8$ neighbors or bandwidth-adaptive from Stage 0 correlogram range?
   Recommend adaptive: use `bandwidth_to_outcome` from `KernelField` if available, else $k=8$.
3. **`forcesmip` phase term** — polar gradient direction is ambiguous (poleward vs equatorward warming).
   May need to leave `forcesmip` in "mixed" but with `phase_term_source: null` until further research.
4. **Phase B phase reference pinning** — pin $\Phi = 0$ at domain centroid or at the inferred source
   candidate? Centroid is simpler; source pinning is more physically interpretable but requires Phase A
   source coherence to be reliable first.
