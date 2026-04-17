# SPARC V2 — Development Roadmap
**Spatial Analysis and Research Core · SPARC Labs · Kyle Wire · 2026**

---

## Temporal Integration

### The Short Answer

A time-of-day variable alone is insufficient and physically wrong. Here's why, and what to do instead.

### What Three Temperature Snapshots Actually Give You

Morning, midday, and nighttime observations are not just "more data" — they carry fundamentally different physical signals:

| Snapshot | Physical Regime | What It Reveals |
|----------|----------------|-----------------|
| **Morning (06:00–08:00)** | Post-nocturnal cooling, minimal solar input | Thermal inertia — surfaces that stayed warm overnight are high-mass impervious materials |
| **Midday (12:00–14:00)** | Peak solar forcing, maximum source term S | Direct radiative response — which surfaces absorb and re-emit most strongly |
| **Nighttime (22:00–00:00)** | No solar input, longwave emission dominant | Longwave re-radiation — urban canyon geometry, sky view factor effects |

The diurnal temperature range `ΔT = T_midday − T_morning` is a physically meaningful quantity: high ΔT = low thermal inertia (bare soil, sparse vegetation), low ΔT = high thermal inertia (water, dense urban). This is a direct empirical proxy for `α` (thermal diffusivity) in the PDE — meaning **temporal observations directly constrain your ProcessRateNet** rather than it having to infer `α` from land cover alone.

### How This Integrates Into the PDE System

The heat equation in steady state is `∇²T = S/α`. With temporal observations you recover the transient term:

```
∂T/∂t = α · ∇²T + S/ρc
```

where `∂T/∂t` is approximated from your three snapshots:

```python
dT_dt_morning_to_midday = (T_midday - T_morning) / dt_1   # heating rate
dT_dt_midday_to_night   = (T_night  - T_midday)  / dt_2   # cooling rate
dT_dt_night_to_morning  = (T_morning - T_night)  / dt_3   # nocturnal decay
```

These temporal derivatives become additional supervision signals for the PDE loss:

```python
# Existing steady-state PDE loss (Step 6)
pde_loss = heat_diffusion_loss(T_midday, alpha, source_term)

# NEW: transient consistency loss
transient_loss = MSE(alpha * laplacian(T_midday) + source/rho_c, dT_dt_morning_to_midday)
nocturnal_loss = MSE(alpha * laplacian(T_night),                 dT_dt_midday_to_night)
# nocturnal: S ≈ 0 at night, so ∂T/∂t ≈ α·∇²T (pure diffusion)
```

The nocturnal snapshot is especially valuable: with no solar forcing, the heat equation simplifies to pure diffusion, giving a clean constraint on `α` that isn't confounded by the source term.

### What Changes in the Pipeline

**Data side — three changes:**

1. Add three temperature columns to your input CSV: `AAT_morning`, `AAT_midday`, `AAT_night` (or z-scored equivalents). These are the same variable measured at different times, not different variables.

2. Compute derived temporal features once before the training loop:

```python
dT_dt_heat    = (T_midday - T_morning) / 6.0   # ~6 hours morning→midday
dT_dt_cool    = (T_night  - T_midday)  / 8.0   # ~8 hours midday→night
diurnal_range = T_midday - T_morning            # thermal inertia proxy → ProcessRateNet input
```

3. Add `diurnal_range` as an input to `ProcessRateNet` alongside the land cover variables. This is the most direct empirical constraint on `α` you can give it.

**Model side — two changes:**

1. Use a learned time embedding, not a raw scalar:

```python
# Time embedding (learned, not a raw scalar)
self.time_embed = nn.Embedding(3, time_dim)   # 3 snapshots → learned vector
# morning=0, midday=1, night=2

# In forward():
t_vec      = self.time_embed(time_idx)        # (N, time_dim)
trunk_out  = self.fusion(spatial, physics, t_vec)
```

This is architecturally cleaner than a raw time-of-day float because it lets the model learn fundamentally different representations for each regime rather than interpolating along a scalar axis.

2. Add the transient PDE loss terms to `pde_loss.py` (Step 6). Weight the nocturnal term higher initially because it provides cleaner `α` supervision with no source term confounding.

**`project.yml` additions:**

```yaml
temporal:
  snapshots: [morning, midday, night]
  dt_morning_to_midday_hours: 6.0
  dt_midday_to_night_hours:   8.0
  dt_night_to_morning_hours:  10.0
  use_transient_pde_loss:     true
  lambda_transient:           0.05
  lambda_nocturnal:           0.08    # higher weight — cleaner physics signal
```

### What NOT to Do

- **Do not** add a raw `time_of_day` scalar (0, 6, 12, 18, 22) as a feature alongside spatial predictors. This treats time as a continuous spatial feature, which is physically meaningless — 6am and 18pm are not "closer" than 6am and 12pm in any physically meaningful sense.
- **Do not** concatenate all three temperature snapshots as separate input features without temporal structure. This discards the ordering information (morning < midday in solar forcing) that is physically meaningful.

### Expected Benefit

Beyond R² improvement, temporal observations enable:

- **Direct α calibration** from nocturnal cooling curves — removes the need to infer diffusivity purely from land cover
- **Diurnal UHI intensity** as an output — not just mean temperature but how much hotter cities get during peak solar forcing vs. overnight
- **Thermal inertia mapping** — `diurnal_range` as a spatial product identifies high-mass urban materials

---

## Track A — PDE Physics Integration

> **Starting point:** Phase 0 complete. Surrogates fixed, validation gate in place (R² > 0.95 required before joint training).

---

### Step 1 — Foundation ✅ Complete

**Files:** `surrogates.py`, `enhanced_spatial_cv.py`, `v2_neural_training.py`

What was fixed:
- `inv_bw` output suppression removed from `DifferentiableGWR`
- Per-predictor bandwidth modulation corrected (no more `bw_signal` averaging)
- Fitted values (not OOF predictions) passed as surrogate target
- `validate_surrogates()` gate blocks joint training on R² < 0.95

---

### Step 2 — PDE Operators

**File:** `sparc/physics/pde_operators.py`
**R² lift:** Infrastructure — no direct lift; enables everything downstream

**Why second:** All physics losses, boundary conditions, and input derivatives depend on these operators. Build once, test in isolation, reuse everywhere. A bug here propagates to all downstream steps.

```python
laplacian(T, neighbor_idx, resolution)
# Test: f(x,y) = x² + y²  →  ∇²f = 4 everywhere (constant)

directional_curvatures(T, neighbor_idx, resolution)
# Test: f = x²  →  d2_dx2 = 2, d2_dy2 = 0

gradient_magnitude(T, neighbor_idx, resolution)
# Test: f = x  →  dT_dx = 1, dT_dy = 0, |∇f| = 1

hessian_invariants(T, neighbor_idx, resolution)
# Test: f = x² + y²  →  det(H) = 4   (elliptic point)
# Test: f = x² - y²  →  det(H) = -4  (saddle point)
```

**Deliverable:** All four operators passing unit tests. No downstream step begins until confirmed.

---

### Step 3 — Input Field Derivatives as Features

**File:** `sparc/physics/input_derivatives.py`
**R² lift:** 0.75 → 0.80+

**Why third:** Pure preprocessing — no architectural changes. Adds derived features before they enter the surrogates and meta-learner. Immediate signal enrichment with zero risk to existing components.

**Per-predictor outputs** (Canopy, Impervious, NDVI, Albedo, Elevation, Dist_water):
- `∇²predictor` — edge detection (where land cover transitions sharply)
- `|∇predictor|` — transition intensity
- `∂²/∂x²`, `∂²/∂y²` — directional curvature
- `det(H)`, anisotropy — shape classification

**Composite features:**
- `thermal_stress_laplacian = ∇²(Impervious × (1 − Canopy))`
- `cooling_potential_laplacian = ∇²(Canopy × NDVI)`
- `albedo_canopy_gradient = |∇(Albedo × (1 − Canopy))|`

**If using temporal data, also compute:**
- `∇²T_morning`, `∇²T_midday`, `∇²T_night`
- `|∇(diurnal_range)|` — where thermal inertia transitions spatially

**Wire into training loop:**

```python
# Compute once per fold, before training
derived_features, derived_names = compute_predictor_derivatives(
    X, neighbor_idx, resolution, var_names
)
derived_norm, feat_means, feat_stds = normalize_derivatives(derived_features)
physics_feats_extended = torch.cat([physics_feats_raw, derived_norm], dim=1)
# Update n_physics_features in meta-learner
```

**Deliverable:** Normalized derivative features flowing into training loop. R² baseline re-established at 0.80+.

---

### Step 4 — PDE-Informed Physics Encoder

**File:** `sparc/models/pde_encoder.py`
**R² lift:** 0.80 → 0.83+

**Why fourth:** Most architecturally significant change. SIREN activations naturally represent solutions to Laplace's equation (`∇²T_h = 0`), making the architecture structurally biased toward physically valid solutions rather than merely penalized toward them.

```python
class SIRENLayer(nn.Module):
    # Sinusoidal activation with LayerNorm pre-activation
    # SIREN-specific initialization (ω₀ = 30)

class SourceDrivenEncoder(nn.Module):
    # GELU MLP — particular solution T_p
    # Inputs: solar forcing, albedo, impervious, anthropogenic heat
    # If temporal: modulated by time_embed per snapshot

class HarmonicEncoder(nn.Module):
    # SIREN network — homogeneous solution T_h
    # Inputs: spatial coordinates, morphology features
    # SIREN naturally represents ∇²T_h = 0 solutions

class PDEInformedPhysicsEncoder(nn.Module):
    # Combines both with learnable blend weight w (sigmoid-bounded scalar)
    # w → 1: source-dominated domain (dense urban core)
    # w → 0: boundary-dominated domain (suburban fringe, coastal)
    # Scientific output: w_source logged per epoch
```

**Replace in `neural_meta.py`:**

```python
# Remove:
self.physics_enc = SIRENLayer(n_physics → 32)

# Add:
self.physics_enc = PDEInformedPhysicsEncoder(
    n_physics_features=n_physics_extended,
    out_dim=32,
    omega=30.0
)

# In forward():
p, w_source, w_harmonic = self.physics_enc(physics_feats_extended)
# Store w_source for diagnostic logging
```

**Deliverable:** `PDEInformedPhysicsEncoder` in place. Blend weight logged per epoch. R² at 0.83+.

---

### Step 5 — Spatially-Varying Process Rate Network

**File:** `sparc/models/process_rate_net.py`
**R² lift:** Indirect — improves physics loss quality for Steps 6–8

**Why fifth:** A scalar thermal diffusivity `α` forces every point to satisfy the same diffusion equation regardless of surface type. Asphalt, tree canopy, and water have fundamentally different `α` values. With temporal data, nocturnal cooling curves provide direct empirical calibration.

```python
class ProcessRateNet(nn.Module):
    # Sigmoid output hard-constrains α to physical bounds
    # (5×10⁻⁷ to 1×10⁻⁶ m²/s for UHI)

    # Inputs: Pct_Impervious, Pct_Canopy, Pct_Water, NDVI
    # + diurnal_range if temporal data available (direct α constraint)

    def compute_mixture_prior(self, land_cover, material_table):
        # α_prior = Σ f_class · α_class
        # impervious: 7.5×10⁻⁷
        # canopy:     3.0×10⁻⁷
        # water:      1.4×10⁻⁷
        # soil:       3.5×10⁻⁷
```

**Pre-train 30 epochs toward mixture prior before joint training:**

```python
for epoch in range(30):
    alpha_pred  = process_rate_net(land_cover)
    alpha_prior = process_rate_net.compute_mixture_prior(land_cover, material_table)
    loss = F.mse_loss(alpha_pred, alpha_prior)
    loss.backward()
    opt.step()
```

**If temporal data available, add nocturnal calibration loss:**

```python
# Nocturnal: S ≈ 0, so ∂T/∂t ≈ α·∇²T
# Direct α supervision from observed cooling rate
nocturnal_alpha_loss = MSE(
    alpha * laplacian(T_night, neighbor_idx, resolution),
    dT_dt_midday_to_night
)
```

**Deliverable:** `ProcessRateNet` pretrained and producing spatially-varying `α` field. Validation report printed. If temporal: nocturnal calibration loss active.

---

### Step 6 — Full Multi-Term PDE Loss

**Files:** `sparc/physics/energy_balance.py`, `sparc/physics/pde_loss.py`
**R² lift:** 0.83 → 0.86+

**Why sixth:** With surrogates, derivatives, and ProcessRateNet working, the PDE loss now has high-quality inputs. All residuals normalized by running std before weighting — prevents any single term from dominating due to scale mismatch.

**Energy balance terms:**

```python
net_radiation()      # Q* = SW·(1−α) + LW_down − LW_up
sensible_heat_flux() # QH = −k·∇²T·depth
latent_heat_flux()   # QE = Priestley-Taylor via canopy + NDVI
storage_flux()       # QS = ρ·c·d·T
advection_flux()     # QA = −ρ·cp·(u·∂T/∂x + v·∂T/∂y)  [needs wind data]
```

**Loss weights:**

```python
@dataclass
class PDELossWeights:
    heat_diffusion: float = 0.10   # primary PDE term
    directional:    float = 0.05   # per-axis consistency
    energy_balance: float = 0.05   # energy conservation
    anisotropy:     float = 0.02   # structure constraint
    gradient_flux:  float = 0.02   # ∇T ~ QH coupling
    gaussian_curv:  float = 0.01   # elliptic preference
    alpha_smooth:   float = 0.01   # process rate smoothness
    alpha_prior:    float = 0.01   # mixture prior regularization
    # Temporal terms (if snapshots available):
    transient:      float = 0.05   # ∂T/∂t = α·∇²T + S/ρc
    nocturnal:      float = 0.08   # nighttime pure diffusion (S≈0)
```

**`project.yml` — required scalars (no new data collection):**

```yaml
physics:
  solar_forcing_Wm2:     600.0   # W/m², mean summer insolation Providence RI
  anthropogenic_Wm2:      20.0   # W/m², urban background heat
  T_sky_K:               280.0   # K, effective sky temperature
  T_water_K:             285.0   # K, water body Dirichlet BC
  k_boundary_convective:  10.0   # W/(m²·K), surface-atmosphere exchange
  surface_layer_depth_m:   0.3   # m, thermal storage depth
```

**Deliverable:** `compute_pde_loss()` replacing single physics loss line. R² at 0.86+.

---

### Step 7 — Boundary Conditions

**File:** `sparc/physics/boundary_conditions.py`
**R² lift:** 0.86 → 0.88+

**Why seventh:** Interior PDE is well-specified after Step 6. Without BCs the solution is underdetermined at domain edges. Neumann zero-flux is the physically correct default for UHI — domain edges are observation boundaries, not heat sinks.

```python
detect_boundary_points()  # from neighbor_idx, identify N/S/E/W edges
neumann_loss()            # ∂T/∂n = 0 — default UHI (insulating edges)
dirichlet_loss()          # T = T_boundary at water bodies
robin_loss()              # ∂T/∂n + β·T = γ for convective boundaries
periodic_loss()           # T(east) = T(west) for ForceSMIP global grid
compute_bc_loss()         # dispatches from domain config
```

**UHI default:**

```python
bc_specs = [
    BoundarySpec(BoundaryType.NEUMANN,   value=0.0,     direction='all'),
    BoundarySpec(BoundaryType.DIRICHLET, value=T_water, direction='water_mask'),
]
```

**Deliverable:** BC loss integrated into joint loss. Edge artifacts reduced. R² at 0.88+.

---

### Step 8 — Initial Conditions

**File:** `sparc/physics/initial_conditions.py`
**R² lift:** Small direct; large indirect (better-conditioned training)

**Why eighth:** For steady-state UHI, ICs act as a spatial prior regularizing predictions near the OLS baseline. Warmup schedule prevents IC loss from dominating early training.

With temporal data, the morning snapshot is a natural initial condition — no approximation needed:

```python
# Without temporal data:
T0_ols   = X @ ols_weights
T0_equil = physics_equilibrium_ic(S, alpha, k=10)   # S/(α·k)
T0       = 0.5 * T0_ols + 0.5 * T0_equil

# With temporal data (preferred):
T0 = T_morning   # actual observed morning temperature is the IC
ic_loss = ic_consistency_loss(
    T_midday_pred,
    T_morning + alpha * laplacian(T_morning) * dt
)
```

**Warmup schedule — ramps over first 20 epochs:**

```python
ic_loss_weighted = warmup_ic_schedule(epoch, ic_raw, warmup_epochs=20, max_weight=0.05)
```

**Deliverable:** IC module with warmup integrated. Training loss curves smoother. No R² regression.

---


### Step 9 — Diagnostic Maps and Scientific Outputs

**Files:** Output additions to existing pipeline

```python
# GeoTIFF exports:
# heat_diffusion_residual.tif
# laplacian.tif
# gaussian_curvature.tif
# anisotropy.tif
# gradient_magnitude.tif
# diurnal_range.tif              (if temporal)
# alpha_field.tif                (learned thermal diffusivity)

# Per-epoch logging:
w_source = meta_learner.physics_enc.blend_weight
print(f"Epoch {epoch}: {w_source:.3f} source / {1-w_source:.3f} harmonic")
# w_source → 1: source-dominated (strong local forcing)
# w_source → 0: morphology-dominated (boundary effects)
```

**Deliverable:** All GeoTIFFs exported. Blend weight convergence plot. `sensitivity_package.json` extended with PDE diagnostics.

---

## Track B — Transfer Learning

### B1 — Architecture Separation

**Depends on:** Track A Steps 1–4 complete (PDE encoder in place)

Separate the shared trunk (universal physics) from the city adapter (local idiosyncrasy):

```python
class SPARCTransferableModel(nn.Module):
    def __init__(self):
        # Shared trunk — accumulates cross-city physics
        self.pde_encoder      = PDEInformedPhysicsEncoder(...)
        self.spatial_encoder  = SpatialEncoder(...)
        self.process_rate_net = ProcessRateNet(...)

        # City adapter — small, retrained per city
        self.city_adapter = nn.Sequential(
            nn.Linear(trunk_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
```

Training rule: new city → freeze trunk weights, train adapter only. Adapter converges with far less data because the trunk already understands urban heat physics.

**Deliverable:** Providence trains as City 1. Trunk checkpoint saved. Cold vs. warm-start R² gap measured on held-out spatial block.

---

### B2 — Warm-Start Validation

Train Boston twice: cold start (random init) and warm start (Providence trunk).

**Key metrics:**
- Cold-start R² vs. warm-start R² on Boston test set
- Epochs to convergence (warm-start should be 3–5× faster)
- Data efficiency: warm-start R² at N=500 samples vs. cold-start at N=500
- Physics convergence: does `w_source` converge faster on Boston? (it should)

**Deliverable:** Boston trained both ways. R² gap and convergence curves documented. **This result alone is publishable toward JAMES as a transfer learning contribution.**

---

## Track C — Continual Spatial Learning

### C1 — Elastic Weight Consolidation (EWC)

After training on each city, compute and store the Fisher information matrix for the shared trunk's parameters. Protects important weights from being overwritten by the next city.

**Joint loss with EWC:**

```python
total_loss = (
    task_loss
  + λ_physics * pde_loss
  + λ_bc      * bc_loss
  + λ_ic      * ic_loss
  + λ_ewc     * Σ_i F_i * (θ_i − θ*_i)²   # EWC penalty
  + λ_replay  * replay_loss                 # covered in C2
)
```

**Fisher computation (once per city, after training):**

```python
fisher = {}
for name, param in trunk.named_parameters():
    fisher[name] = grad(log_likelihood, param) ** 2   # squared gradients
```

**Deliverable:** After Boston training, Providence R² degrades by less than 0.5pp. Fisher matrix saved to registry.

---

### C2 — Experience Replay Coreset

After each city, select 300–500 representative points via K-medoids on the joint (features, spatial coords) space. Stored in the registry and replayed during all future city training.

```python
replay_loss = sum(
    MSE(model(coreset.X, coreset.spatial, coreset.physics), coreset.y)
    for coreset in registry.load_all_coresets()
) / len(registry)
```

**Deliverable:** After City 3, Providence performance within 1pp of Providence-only model.

---

### C3 — City Registry

```
sparc_registry/
    providence/
        trunk_checkpoint.pt       <- shared encoder weights after City 1
        fisher_matrix.pt          <- EWC Fisher matrix for City 1
        coreset.npz               <- 400 representative points
        welford_state.pkl         <- running mean/variance
        metrics.json              <- R², Moran's I, pattern correlation
    boston/  city_3/  ...         (same structure per city)
    global_trunk.pt               <- best shared trunk across all cities
```

**Deliverable:** Registry operational. Three cities trained sequentially. `global_trunk.pt` updated after each city.

---

### C4 — Welford Online Scalers

Replace per-city `StandardScaler` with an online Welford scaler that updates running mean and variance incrementally across cities — no distribution shift between cities in the registry.

**Deliverable:** Normalisation consistent across all cities. Scaler state saved to registry.

---

## Full Sequencing

```
Phase 0    Surrogates fixed ✅

Phase A1   Steps 2–3: PDE operators + input derivatives
           (pure preprocessing, lowest risk, do first)

Phase A2   Steps 4–5: PDE encoder + process rate net
           (architecture change)

Phase A3   Steps 6–7: Full PDE loss + BCs
           ──────────────────────── Phase B1 begins in parallel here

Phase A4   Step 8: Initial conditions (stability)

Phase A5   Steps 9–10: NUTS + MC³ fixes
           (fully parallel with A2–A4 — operates on existing OOF predictions)

Phase B1   Architecture separation: shared trunk + city adapter

Phase B2   Warm-start validation: Providence → Boston cold vs. warm

Phase C1   EWC: Fisher matrix + forgetting penalty

Phase C2   Replay + Registry: coreset selection, city registry operational

Phase C3   Welford scalers + 3-city end-to-end validation

Phase A6   Step 11: Diagnostic maps (last, after everything works)
```

---

## R² Progression

| Step | Component | Target R² |
|------|-----------|-----------|
| 0 | Surrogate fix baseline | 0.75+ |
| 3 | Input derivatives | 0.80+ |
| 4 | PDE encoder | 0.83+ |
| 6 | Full PDE loss | 0.86+ |
| 7 | Boundary conditions | 0.88+ |
| Temporal | Transient PDE + nocturnal calibration | 0.90+ |
| V1 baseline | Enhanced spatial CV ensemble | 0.915 |