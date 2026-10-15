# SPARC Physics: Plan

**Status:** in progress: W0, W1 and the W3 parameter panel started 2026-10-15 · **Date:** 2026-10-15 · **Scope:** the physics base model in `sparc/core/` (`physics.py`, `operators.py`) and the checks that decide how far to trust it · **Builds on:** [`CORE_ROADMAP.md`](CORE_ROADMAP.md) §2.1 and Appendices C–G

> Supersedes [`prd-wave-phase-physics.md`](../prd/prd-wave-phase-physics.md) and its task list, retired 2026-10-15 before any of its tasks were started; see §2.3.

---

## TL;DR

- **Does "everything is a wavelength" make sense for SPARC?** Yes, in three specific senses. All three are standard physics or statistics, and SPARC already uses the first without saying so.
  1. **Spatial wavelengths.** A linear PDE with constant coefficients acts on each spatial wavelength separately. It passes or damps each one by a fixed gain, and shifts it by a fixed phase. SPARC's heat solver already works this way: `operators.py` solves Ĝ(k) = 1/(1 + L²|k|² + i·v·k).
     - Heating patterns much larger than 2πL (about 1.6 km at Providence's L = 256 m) reach the air almost whole.
     - Patterns much smaller than that are smeared out.
  2. **Wavelengths of light.** The energy budget splits by electromagnetic wavelength:
     - sunlight (about 0.3–3 µm), governed by albedo;
     - heat radiation (about 3–100 µm), governed by emissivity and how much sky a street sees.

     SPARC's albedo layer is not broadband, and longwave trapping is missing.
  3. **Wavelengths in time.** The daily cycle is a wave. Built surfaces store heat and give it back hours later, which shows up as a phase lag.
- **Not in the literal, quantum sense.** Street-level air temperature is a classical, diffusive quantity. The legacy wave-phase PRD (a Schrödinger "phase field" and a "probability current" for temperature) has nothing measurable behind it and should be retired.
  - Its one sound intuition was that wind should leave a directional signature. That intuition is kept here, as a quantity you can measure: the phase of the source-to-temperature transfer function.
- **The causal "if–then" approach fits well.** A steady-state PDE is a structural causal model: it says what the temperature would be *if* the surface changed. The plan writes every physics mechanism as an if–then claim with a test, and the test decides whether the mechanism is kept, flagged or dropped (§4).
- **Wavelength and causality meet in one place: confounding depends on scale.** Suppose unmeasured confounders vary only over long wavelengths. Then an effect can be identified from the treatment's short-wavelength variation.
  - Appendix G of the core roadmap found exactly this on Providence: canopy within 100 m is identified, but canopy within 1 km is not.
  - The physics model predicts how an effect should change from one wavelength band to the next, so the physics and the causal estimates can check each other (§5, W5).
- **This review found four problems** (§3):
  1. The physics is fitted to CAPA's interpolated map, not to measurements, so L and the canopy shading may partly describe the interpolation rather than the air.
  2. The heat source is set to "average" everywhere outside the labelled cells, including the rivers on the city's east and south edges. Meanwhile, an indicative regression gives an elevation effect 2.7× the dry-adiabatic lapse rate.
  3. The central-difference wind term produces negative Green's-function lobes, down to −81% of the peak, at short L and strong wind. A standard scheme (exponential fitting) removes them.
  4. Fitted parameters are never converted into physical quantities. Once converted, they can be checked against the literature. The fitted gain implies an aerodynamic resistance of about 46 s/m, inside the roughly 20–100 s/m reported for urban surfaces. That is first evidence the term behaves physically, but it was fitted to the product.
- **Sequence:**
  1. numerical verification and the mechanism register (cheap, and they change what can be claimed today);
  2. parameter audit, source completeness and radiation;
  3. spectral and causal tests;
  4. multi-window time physics on the raw traverses.

---

## 1. The physics model today

**The equation.** `PhysicsModel` (`sparc/core/physics.py`) solves the steady advection–diffusion–relaxation equation of `operators.py` for the afternoon air-temperature anomaly:

```
(1 − L²∇² + v·∇) φ = q(features; θ)          ΔT_phys = a·φ + b + γ·elev + βx·x + βy·y
```

- **q** is normalised sensible heat, Q_H / SW↓: absorbed sunlight with canopy shading, impervious storage, a sigmoid evaporative fraction and a water sink.
- **L = √(Kτ)** is the mixing length; **v = u·τ** is the advection offset.
- **The solve** is spectral: Ĝ = 1/(1 + L²λ(k) + i·adv(k)) on a zero-padded grid. The PDE residual is therefore exactly zero on the model's own prediction.
- **The fit** profiles a, b, γ and β out by variable projection; L, v and the source parameters are fitted by L-BFGS under weak priors.

**What has been checked.**
- It recovers planted L, v and a on a known source, and the planted canopy shape on the synthetic city (`tests/core/test_physics.py`).
- On Providence it scores R² 0.49–0.51 alone on unseen 2 km blocks (L = 256 ± 20 m, gain a = 45 ± 5 °F), and takes 0.29–0.35 of the stack's weight.
- Placebo layers exposed a prior-driven canopy effect, and the priors were weakened in response (Appendix D).
- Advection was tested out of fold and dropped (ΔRMSE +0.003 ± 0.017 °F).

**What it does not have:**
- a statement of what each term predicts *beyond* fitting one afternoon map;
- physical units for its fitted parameters;
- any test against raw measurements.

---

## 2. Does the wavelength idea make sense? The honest answer

### 2.1 Where it is standard science

| Sense | What it means for SPARC | Status today | Basis |
|---|---|---|---|
| **Spatial wavelength** (Fourier modes of a field) | Each wavelength of the heating pattern reaches the air with gain a/(1 + L²k²). Wind adds a phase shift arctan(v·k / (1 + L²k²)): heat appears downwind of its source. For planners, a green patch much smaller than L cools mostly its neighbours, not itself. This agrees with field data: cooling distance grows with green-space size, and spaces under 0.5 ha showed no measurable effect (Monteiro et al. 2016, London, night). | Used inside the solver, never tested against data | Fourier diagonalisation of constant-coefficient operators (Trefethen 2000); transfer-function estimation (Bendat & Piersol 2010) |
| **Matérn spectrum** | With white-noise forcing, the same operator produces a Matérn ν = 1 field with spectrum ∝ (1/L² + k²)⁻². This is why the S1 correlogram range gives the physics prior L = R/√8. | Used for the L prior | Whittle (1954); Lindgren, Rue & Lindström (2011) |
| **Scale-dependent confounding** | A confounder that varies only over long wavelengths biases only the long-wavelength part of an effect estimate. | Used implicitly: the wind-shift test's 3 km basis, Appendix G's 100 m street differences | Paciorek (2010); Keller & Szpiro (2020); Guan et al. (2023) |
| **Electromagnetic wavelength** | Shortwave (albedo, shade) and longwave (emissivity, sky view factor) are separate terms of the energy balance. | Shortwave only, and from a non-broadband albedo layer | Liang (2001); Bonafoni & Sekertekin (2020); Oke (1981); Oke et al. (2017) |
| **Temporal wavelength** (the diurnal cycle) | Periodic heating, with storage that lags it: the hysteresis term a2·dQ*/dt. | Absent: one window, steady state | Camuffo & Bernardi (1982); Grimmond, Cleugh & Oke (1991); Carslaw & Jaeger (1959) |

### 2.2 Where it is not

- **Diffusion has no wave speed.** The heat equation is parabolic. A disturbance spreads and decays; it does not travel, reflect or interfere the way a wave does. Air temperature at street level is a classical continuum quantity, and quantum wave–particle duality plays no role at these scales.
- **Wave physics is real in other SPARC domains.**
  - Sound is an acoustic wave governed by the Helmholtz equation. Engineering practice computes its attenuation terms (geometric divergence, atmospheric absorption, ground, barriers) with ISO 9613-2.
  - Ground motion follows the elastic wave equation.

  The website's domain page already says so. Neither domain is in the core, and neither is this plan's subject.
- **Air quality and wildfire are not wave domains.**
  - Pollutants are carried by advection–diffusion; see [`AIR_QUALITY_PLAN.md`](AIR_QUALITY_PLAN.md).
  - Fire spread is modelled with semi-empirical spread equations (Rothermel 1972).

  The legacy PRD's "1/r wave regime" for both is not supported.

### 2.3 The legacy wave-phase PRD

| PRD idea | Problem | What replaces it |
|---|---|---|
| Treat temperature as a wave function ψ = T·e^{iΦ}; learn the phase Φ with a "Schrödinger residual" | Only T is observed. Φ is then constrained by nothing except the loss that invents it, so it cannot be identified. The Wick-rotation link holds for the *time-dependent* heat equation, but SPARC's model is steady. Its operator (1 − L²∇²) has decaying solutions, not oscillating ones. | Nothing: drop it |
| "Probability current" J = \|ψ\|²∇Φ as an influence-flux map | Temperature already has a physical flux, −K∇T + uT. "Where heat comes from" is the Green's-function footprint, which S4 already maps. | Footprint maps (S4) plus the advection phase test (W4) |
| Align ∇T with the solar azimuth (Phase A) | The sun's azimuth sets shadow direction at the scale of buildings, a few metres. It does not set the direction of 30 m air-temperature gradients, which follow the layout of sources and the wind. I found no published support for this alignment. | Shading in the source term (exists) and advection tests (W4) |
| A 1/r regime for air quality and wildfire | Wrong mechanism (§2.2) | AQ plan; out of scope here |
| A source-coherence diagnostic | The sound part: does the field point back at its sources? | The cross-correlogram peak offset and the cross-spectral phase (W4) |

**Decision (2026-10-15):** the PRD and its task list are retired and marked *superseded by this plan*.

---

## 3. Review findings

| # | Finding | Evidence | Why it matters | Fix |
|---|---|---|---|---|
| **F1** | **The physics may be fitted to the product, not the air** | `AAT_z` is CAPA's area-wide map. Traverse readings are extended to every cell by a machine-learning model on satellite predictors averaged over buffers of 50 m and up (Shandas et al. 2019). SPARC's emulation of it, a random forest, writes two thirds of the canopy artefact in the null simulation (Appendix D). | L, the shading strength s and the shade scale κ may describe the product's buffer radii. Every physics test run on the map partly tests CAPA's model. | W7 |
| **F2** | **The heat source stops at the labelled cells** | The grid is 331 × 285 and 58% filled. Of the empty cells, 96% lie *outside* the study outline; holes enclosed by data in all four directions are 1.7% of the grid. The Seekonk River (east) and the Providence River / upper bay (south) sit at that edge. Outside the labelled cells the centred q is 0, i.e. average heating. The campaign wind was from the south. | The bay's cooling, carried inland by a south wind, can only be absorbed by the elevation, water-distance and linear-trend terms. Indicative check: an OLS of temperature on the six predictors plus x and y gives an elevation slope of **−26 K/km**. The physical range is −6.5 (standard atmosphere) to −9.8 K/km (dry adiabatic). | W2 |
| **F3** | **The wind term can make heating cool its neighbours** | Central differences for v·∇ give negative Green's-function lobes once the cell Péclet number \|v\|·dx/(2L²) exceeds about 1 (table below). This is the textbook failure of central differencing for convection–diffusion (Patankar 1980). Exponential fitting (Il'in 1969) removes the lobes and keeps the downwind shift. | No effect on Providence today (L = 256 m, advection dropped, Pe ≤ 0.23). But the fit box allows L down to 30 m with \|v\| up to 1 km, and the AQ plan needs sharp sources with wind. | W1 |
| **F4** | **Parameters are not in physical units** | In the local limit ΔT = a·Q_H/SW↓, while bulk transfer gives ΔT = r_a·Q_H/(ρc_p). So **r_a = a·ρc_p/SW↓**. With a = 45 ± 5 °F, ρc_p ≈ 1170 J m⁻³ K⁻¹ and SW↓ = 636 W m⁻², r_a ≈ 41–51 s/m. Published values are about 20–100 s/m: Masson's (2000) canyon formula gives ≈ 100 s/m in calm air and ≈ 36 s/m at 5 m/s, and Crawford et al. (2018) measured ≈ 40 s/m in central London. Similarly L² = Kτ and r_a = τ/h. | The one number checkable today is in range. But a single afternoon map pins only two combinations (r_a and L), not τ, h and K separately; advection (v = u_c·τ) or a time lag (ωτ) would pin τ. The non-broadband albedo (F5) also biases r_a. Without units, nobody can tell whether a fit is physics or a smoother. | W3 |
| **F5** | **The radiation term is approximate** | The albedo layer averages 0.40, against 0.155 for Sentinel-2 broadband. `albedo_map` rescales it linearly to an assumed range. There is no longwave trapping (sky view factor) and no anthropogenic heat. | Absorbed sunlight sets the scale of q, and with it a and r_a. Night and canyon effects are missing. | W8 |
| **F6** | **Identifiability is not reported per parameter** | Uncertainty is fold spread only. The canopy shading term once followed its prior (Appendix D). | A parameter the data cannot pin down should not carry a mechanism claim. | W3 |
| **F7** | **One steady window** | The night branch (a1n) is untested on data, and OHM hysteresis (a2) is absent. Heat Watch drove the same routes at 6–7, 15–16 and 19–20 EDT, plus 0–1 the next night. | Storage, shading and the lapse rate all change sign or size through the day. That change is the strongest test of each mechanism available. | W6 |
| **F8** | **Advection is judged by RMSE alone** | ΔRMSE +0.003 ± 0.017 °F out of fold | An RMSE test cannot tell "no advection" from "advection the product cannot show". The product's predictors are isotropic buffers, so the map is wind-blind by construction. | W4, W7 |
| **F9** | **Mechanism claims are not written down** | — | Nothing says what each term predicts beyond fitting the afternoon map, so nothing can contradict it. | W0 |

**F3 in numbers.** Normalised minimum of the Green's function for a unit source (dx = 30 m), from `operators.solve` and a one-line exponentially fitted variant:

| L (m) | \|v\| (m) | Cell Péclet | min G / max G, central (today) | min G / max G, exponentially fitted |
|---|---|---|---|---|
| 30 | 100 | 1.7 | −0.14 | 0.00 |
| 30 | 1000 | 16.7 | −0.81 | 0.00 |
| 60 | 300 | 1.25 | −0.07 | 0.00 |
| 120 | 1000 | 1.04 | −0.01 | 0.00 |
| 256 | 1000 | 0.23 | 0.00 | 0.00 |

In both schemes the downwind centroid of G stays at about \|v\|/dx cells.

**What the model already does well** (keep all of it):
- the sign constraint on a, and the honest fallback when the fit is anti-physical;
- priors weakened after placebo tests;
- advection gated by out-of-fold evidence;
- an exact residual identity, the synthetic recovery tests, and the `simcheck` physics generator.

---

## 4. The organising idea: mechanisms as if–then claims

**The causal reading of the physics.** The model is a two-step structural causal model:

```
surface features X ──(local energy balance)──▶ source q(X) ──(transport G)──▶ ΔT = a·G ⊛ q + …
```

- An intervention do(canopy := c′) changes q only in the edited cells; transport carries the change to neighbours.
- This is what S5 scenarios already compute. The equilibrium of a dynamical system defines a structural causal model under stated conditions (Mooij, Janzing & Schölkopf 2013). Feedbacks, such as the temperature changing the flux, are allowed in cyclic SCMs (Bongers et al. 2021); here they are linearised away.

**The rule.** Each mechanism is written as *IF the mechanism holds, THEN a signature must appear that the fit itself did not force*. The test of that signature gives a verdict:

- **Supported** — the signature appears;
- **Not testable with this data** — the signature cannot be measured from what is available;
- **Contradicted** — the signature is absent, or reversed.

A Contradicted mechanism is switched off, or reported as a statistical term rather than physics. This is the same logic as the effect verdicts (Robust / Direction only / Not established), applied to mechanisms. It follows the standard view that a model of an open natural system can be confirmed or refuted by tests, not verified (Oreskes, Shrader-Frechette & Belitz 1994).

### Mechanism register

| ID | Mechanism | IF it holds, THEN… | Test | Today |
|---|---|---|---|---|
| **M1** | **Absorbed sunlight** (albedo, canopy shade) | (a) Its effect per unit albedo or canopy scales with SW↓ across runs, and the shading term vanishes at night. (b) Its spatial footprint follows G. | (a) multi-window fit (W6); (b) band-transfer test (W4) | Not testable (one window) |
| **M2** | **Evaporative cooling** (EF from NDVI, canopy, impervious) | Cooling scales with available energy Q* − ΔQ_S, so it is smaller in the morning and evening runs. NDVI is a mediator, so scenarios route through it. | Multi-window fit; the mediator handling already exists | Not testable |
| **M3** | **Heat storage** (impervious, OHM) | Impervious cover warms the 15–16 run less than its absorbed energy implies, and warms the 19–20 and 0–1 runs more. Fitted a1 and a2 fall inside the published OHM ranges (Grimmond & Oke 1999). | Multi-window fit; parameter panel (W3) | Not testable |
| **M4** | **Horizontal mixing** (L) | (a) Band gains of ΔT on q fall as 1/(1 + L²k²). (b) The residual spectrum matches the implied Matérn ν = 1. (c) L fitted on the raw traverses ≈ L fitted on the map. | W4 (a, b); W7 (c) | Not tested |
| **M5** | **Advection** (v) | (a) The q→ΔT cross-correlogram peaks *upwind* of the receptor, and the transfer phase is non-zero along the wind. (b) A street's cooling follows the canopy upwind of it when the wind changes between runs. | (a) W4; (b) `identify/windshift.py`, extended to the physics | (b) built; 8% power on one day |
| **M6** | **Vertical exchange** (r_a = τ/h) | (a) The implied r_a lies in the published range for urban surfaces. (b) Across runs and campaigns, the gain a falls as wind speed rises; heat-island intensity weakens with wind (Oke 1973). | (a) W3; (b) multi-window and multi-campaign | (a) ≈ 46 s/m, inside ~20–100 s/m: **Supported, indicative** (fitted to the product) |
| **M7** | **Elevation** (lapse rate) | γ lies between about −9.8 and −6.5 K/km in the well-mixed afternoon. At night it can be near zero or positive (inversions, cold-air pooling). | Parameter panel (W3); multi-window | **Indicative OLS −26 K/km: Contradicted as a pure lapse rate.** Elevation is standing in for something else, possibly the bay's cooling (F2). |
| **M8** | **Water bodies** | Cooling near water is stronger *downwind* of it under the campaign wind, and water just outside the study area counts too. | Directional water-distance response after source completion (W2) | Not tested |
| **M9** | **Longwave trapping** (sky view factor; missing) | After land cover is accounted for, the residual ΔT tracks sky view factor, mostly at night (Oke 1981). | Residual vs SVF by run (W8) | Not modelled |
| **M10** | **Anthropogenic heat** (missing) | After land cover, the residual tracks a Q_F map at magnitudes inside published ranges (Sailor 2011). | Residual vs AH4GUC Q_F (Varquez et al. 2021) (W8) | Not modelled |

**Deliverable.**
- A new module `sparc/core/mechanisms.py` writes `mechanisms.json`: one row per mechanism with its claim, test, statistic, CI and verdict.
- The results page and Studio get a "Physics: what holds" table.
- A term whose verdict is Contradicted cannot be used to explain a scenario. The scenario text then says the effect is statistical.

---

## 5. Workstreams

Effort key from the core roadmap: S ≈ days, M ≈ 1–2 weeks, L ≈ 3+ weeks. Every workstream ships with tests, and keeps the golden synthetic numbers unless it is meant to change them.

### W0 Mechanism register (S): do this first
- Write §4's table into `mechanisms.py`, with every test initially "Not testable".
- The other workstreams fill in the tests.
- **Exit:** `mechanisms.json` is in every run, and the results page shows it.

### W1 Numerical verification (S)
1. **Exponential fitting for the wind term.**
   - In `operators.symbols`, `apply_operator` and `apply_operator_torch`, replace L² on each axis by L²·Pe·coth(Pe), with Pe = v_axis·dx/(2L²). This is the exponentially fitted scheme (Il'in 1969; Patankar 1980's exponential scheme).
   - It stays an exact discrete inverse, so the stacker's residual identity is kept.
   - It reduces to the current central scheme as Pe → 0, so Providence's numbers should not move.
2. **Positivity test:** G ≥ −10⁻¹² for a unit source over a grid of (L, v) covering the whole fit box.
3. **Analytic oracle.**
   - The continuous Green's function of (1 − L²∇² + v·∇) in 2D is (1/2πL²)·exp(v·r/2L²)·K₀(r·√(1/L² + |v|²/4L⁴)). This is the steady state of the classical continuous point-source solution in uniform flow with first-order decay (Wilson & Miller 1978).
   - Test: the discrete G converges to it away from the source as dx shrinks.
   - Checked during this review: at L = 256 m and v = 300 m, the median error on the 1.5L–2L ring is 0.2% at dx = 30 m and 0.05% at dx = 15 m. That is second-order convergence.
4. **Method of manufactured solutions** (Roache 2002): pick a smooth φ, apply the continuous operator analytically to get q, and check the solve converges at second order.
5. **Truncation test:** responses inside the domain change by < 1% when the padding is doubled.
- **Exit:** all tests in CI. Providence physics R² and S5 scenarios change by < 0.01 °F.

### W2 Source completeness (M)
- **Build q beyond the labelled cells.**
  - Cover the whole raster plus a margin of at least 3·(L + |v|), using open layers. `features_open.py` already builds WorldCover land and water, Sentinel-2 and the Copernicus DEM.
  - Cells with features but no labels feed the source, never the loss. This is the rule `physics.py` already states.
- **Model water as a source.** Replace the binary "within dx/2 of water" sink with a water-surface term from WorldCover's water class (a fitted negative sensible-heat source), so the rivers and bay enter through G and the wind.
- **If–then test (M7, M8).** IF the bay's cooling is what elevation is absorbing, THEN completing the source (with advection allowed) should:
  - move γ toward the lapse-rate range;
  - shrink the linear trend;
  - make cooling downwind of water exceed cooling upwind of it.

  If γ does not move, elevation stands for something else, and γ is reported as a statistical term.
- **Exit:**
  - held-out physics R² is no worse;
  - the γ, trend and water-asymmetry results are written to `mechanisms.json`;
  - a lapse-rate prior on γ is added as a multiverse variant, not as the default.

### W3 Physical units, identifiability and priors (S–M)
- **Parameter panel**, in the style of `literature.py` but for physical parameters:
  - r_a = a·ρc_p/SW↓;
  - γ in K/km;
  - L;
  - the OHM and EF coefficients.

  Each is shown against a published range with its source, and marked inside, outside or not comparable. The sources are:
  - r_a: Masson (2000) and Crawford et al. (2018);
  - γ: the dry-adiabatic and standard-atmosphere lapse rates;
  - OHM: Grimmond & Oke (1999).

  Where two quantities are only identified as a product (τ and h, K and τ), the panel says so instead of inventing a split.
- **Profile likelihood per parameter** (Raue et al. 2009), for L, s, κ, a1 and the EF slopes, refitting the rest each time. A flat profile means "not identified from this data". The mechanism then cannot be Supported, whatever its point estimate.
- **Prior sensitivity:**
  - power-scale each prior up and down and report how much each estimate moves (Kallioinen et al. 2024);
  - add a prior–data conflict check (Evans & Moshonov 2006).

  The placebo finding in Appendix D (s held by its prior) is the case this automates.
- **Exit:** the panel and profiles appear in the run. Canopy shading's identifiability on Providence is stated as a number, not as prose.

### W4 Spatial wavelengths: skill by scale and the transfer test (M)
- **Band-pass decomposition** of ΔT, the prediction and q on the masked raster:
  - differences of mask-normalised Gaussians, which form a Laplacian pyramid (Burt & Adelson 1983), in bands from about 60 m to 4 km;
  - this reuses `operators.masked_gaussian`, so no new FFT code is needed.
- **Skill by wavelength**, out of fold, per band: observed, predicted and residual variance, and their correlation. Forecast verification decomposes skill this way (Briggs & Levine 1997; Casati et al. 2004).
  - The result is the shortest wavelength at which the map has skill: a plain rule for planners ("don't read this map below X m").
  - It complements the skill-vs-distance table.
- **Transfer test (M4a).** Per band, regress held-out ΔT_band on q_band to get an empirical gain with a block-bootstrap CI. Compare it with the *same* regression run on the physics model's own prediction (a model-equivalent comparison, so the band shapes cancel).
  - IF transport is as modelled, THEN the two gain curves agree in every band.
  - A mismatch at short bands means the mixing length is wrong. A mismatch at long bands points to a missing source (F2) or to confounding (W5).
- **Residual spectrum (M4b).** Compare the residual's band variances with the Matérn ν = 1 shape implied by the fitted L (Whittle 1954), which closes the loop with the S1 prior.
- **Advection phase (M5a).**
  - Use the full-plane q→ΔT cross-correlogram that `influence.py` already computes. Under advection its peak sits upwind by about v; by the shift theorem, that is the same information as the cross-spectral phase.
  - Inference rotates the wind direction, as `windshift.py` does: the null is the peak offset under rotated winds.
  - This is the measurable version of the PRD's "phase".
- **Exit:**
  - skill-by-wavelength and gain curves are on the results page;
  - M4 and M5 verdicts are on the map;
  - every test passes on the synthetic city, where the truth is known, before it is run on Providence.

### W5 Causal identification by wavelength (M)
- **Band-wise effects.**
  - Decompose each treatment (canopy, impervious, albedo) into the W4 bands, and estimate each band's effect with the existing spatial DML, adjusting for longer-wavelength structure. This follows the spectral-adjustment approach (Guan et al. 2023) and scale-selection (Keller & Szpiro 2020).
  - Under the stated assumption that confounding lives at long wavelengths, the short-band estimates are unbiased (Paciorek 2010).
  - This formalises what Appendix G found by design: 100 m identified, 1 km not.
- **Physics–causal consistency.** The physics predicts how the effect should change across bands (∝ the band gain from W4).
  - IF both the physics and the confounding assumption hold, THEN the band-wise causal estimates follow the physics curve.
  - Disagreement concentrated in the long bands is the signature of confounding, so the long-band and city-wide figures are flagged.
  - Disagreement in the short bands points at the physics.
- **Physics-defined spillover.**
  - Today's exposure mapping uses a Gaussian neighbourhood (σ = radius/2). Add the fitted Green's function as the exposure kernel (Aronow & Samii 2017), as physical-process interference models do (Wikle & Zigler 2024).
  - Nested test: IF G is the right interference structure, THEN a Gaussian exposure added on top explains nothing more.
- **Caveat:** each piece is published, but this combination is not. It must pass `simcheck` (all six generators, including `confounded` and `null`) and the identification lab before any real result is reported.
- **Exit:**
  - band-wise effects with CIs for all three levers;
  - a consistency verdict per lever;
  - false-positive rate ≤ 10% on the `null` generator.

### W6 Time as a wavelength: multi-window physics (L; after W7's data)
- **The quasi-steady argument, as an if–then.**
  - The daily harmonic has ω = 2π/86,400 s ≈ 7.3 × 10⁻⁵ s⁻¹.
  - IF the effective mixing depth h is canopy-layer scale (tens of metres), THEN τ = r_a·h ≈ 15–40 min, and ωτ ≈ 0.07–0.2. The air then follows its source within about 10° of phase, so each run can keep the steady operator, and all time physics goes into the source.
  - IF instead h were the whole boundary layer (around 1.5 km), τ would be hours and the steady model would fail. The joint fit decides between the two by fitting ωτ in Ĝ(k, ω) = 1/(1 + iωτ + L²k² + i·v·k).
- **Time-varying source.**
  - Q*(t) comes from hourly ERA5 for each run.
  - Storage follows OHM, ΔQ_S = a1·Q* + a2·dQ*/dt + a3, with coefficients per surface type and priors from published tables (Grimmond, Cleugh & Oke 1991; Grimmond & Oke 1999).
  - Shading is proportional to SW↓, so it switches off by itself.
- **Joint fit** across the Heat Watch runs (6–7, 15–16 and 19–20 local time is the CAPA protocol; Providence also drove 0–1, per `configs/core_providence.yml`):
  - shared surface parameters, L and r_a;
  - run-specific forcing and wind.
- **Why this matters most for trust.** Shading is the only term that switches off with the sun. Storage is the only one that reverses after sunset. The lapse rate changes sign under a night inversion. So the multi-window fit is the strongest available test of M1, M3, M6 and M7, and the most promising physical route to the canopy question.
- **Exit:**
  - multi-window held-out skill is no worse than per-window fits;
  - M1–M3 and M7 verdicts are filled;
  - τ is identified, or explicitly not identified.

### W7 Validate on raw observations (M; needs the OSF `wu9v7` traverses)
- **Observation operator.** Fit the physics directly to the traverse points:
  - sample the model raster at each reading's location;
  - add the vehicle offsets and warming drift that `identify/campaign.py` already models.
- **Product-vs-atmosphere test (M4c).**
  - IF L, s and r_a describe the atmosphere, THEN fits on the traverses match fits on the map within their CIs.
  - Parameters that differ are reported as *properties of the product*.
- **Along-track spectra.**
  - Compute temperature spectra along each pass for the observations and for the model sampled at the same points, with the same estimator, in the style of aircraft-transect spectra (Nastrom & Gage 1985).
  - Comparing like with like keeps sampling geometry out of the result.
- **Exit:** traverse-fitted parameters, the product-vs-atmosphere table, and an M4c verdict. Until this has run, the parameter panel labels every physics number "fitted to the CAPA map".

### W8 Radiation by wavelength (M)
- **Broadband albedo.** Compute it from Sentinel-2 bands with published narrow-to-broadband coefficients (Bonafoni & Sekertekin 2020; Liang 2001). This replaces the linear `albedo_map` rescale, and the scene date goes into provenance.
- **Longwave and sky view factor.**
  - Derive SVF from open building footprints with heights, and add a longwave-trapping term to Q*.
  - It matters mostly at night (Oke 1981), so it is gated on W6.
- **Anthropogenic heat (optional).** Add a Q_F source from AH4GUC (Varquez et al. 2021), bounded by the ranges in Sailor (2011).
- **Gate:** a new term stays only if it improves held-out skill and its mechanism test (M9, M10) is Supported.

---

## 6. Order of work

| Step | Workstreams | Why this order |
|---|---|---|
| 1 | W0, W1 | Cheap. They fix a latent numerical defect and set up the verdicts everything else reports into. |
| 2 | W3 | Changes what can be claimed *today* with no new data: units, identifiability, priors. |
| 3 | W2, W8 (albedo part) | The two input-side corrections. Both are likely to move γ, a and r_a. |
| 4 | W4, then W5 | The spectral tests, then the causal tests that depend on W4's bands and gains. |
| 5 | W7, then W6, W8 (longwave) | Need the raw traverses. W6 needs W7's observation operator. |

---

## 7. What this plan does not do

- **No Schrödinger, complex-phase or "probability current" fields** (§2.3).
- **No PDE-penalised neural residual.**
  - Under honest CV it never beat the blend (Appendix C).
  - PDE-penalised networks also have known training failure modes (Krishnapriyan et al. 2021).
  - The physics gets better by improving the physics model itself, not the penalty.
- **No new domains,** and no changes to S1–S7 beyond what is listed: W5 extends `causal.py`, and W4 reuses `influence.py`.

---

## 8. Open decisions

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Wind-term scheme | Exponential fitting in the operator, with the analytic K₀ Green's function as a test oracle only. This keeps the residual identity and the spectral speed. |
| D2 | Lapse-rate prior on γ | Report γ against the lapse rate first. Add the prior only as a multiverse variant. |
| D3 | Retire the wave-phase PRD | **Decided 2026-10-15:** retired, superseded by this plan |
| D4 | Require W7 (raw traverses) before any physics parameter is called "physical" | Yes. Until then the panel says "fitted to the product". |

---

## References

Every reference was checked against a publisher, DOI or index record for this plan; notes mark the limits of that check. In-repo results (Appendices C–G, `providence_core.md`, `canopy_identification.md`) are cited by section.

**Physics, numerics and spectra**
- Bendat, J.S. & Piersol, A.G. (2010). *Random Data: Analysis and Measurement Procedures*, 4th ed. Wiley. doi:10.1002/9781118032428
- Burt, P.J. & Adelson, E.H. (1983). The Laplacian pyramid as a compact image code. *IEEE Trans. Commun.* 31(4), 532–540.
- Carslaw, H.S. & Jaeger, J.C. (1959). *Conduction of Heat in Solids*, 2nd ed. Clarendon Press. (Periodic surface heating: a damped thermal wave with a phase lag.)
- Il'in, A.M. (1969). Differencing scheme for a differential equation with a small parameter affecting the highest derivative. *Math. Notes* 6(2), 596–602.
- Lindgren, F., Rue, H. & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: the SPDE approach. *J. R. Stat. Soc. B* 73(4), 423–498. doi:10.1111/j.1467-9868.2011.00777.x
- Nastrom, G.D. & Gage, K.S. (1985). A climatology of atmospheric wavenumber spectra of wind and temperature observed by commercial aircraft. *J. Atmos. Sci.* 42(9), 950–960. doi:10.1175/1520-0469(1985)042<0950:ACOAWS>2.0.CO;2
- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere. (Central differencing fails for convection–diffusion at large cell Péclet numbers; upwind, exponential and power-law schemes.)
- Roache, P.J. (2002). Code verification by the method of manufactured solutions. *J. Fluids Eng.* 124(1), 4–10. doi:10.1115/1.1436090
- Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM. doi:10.1137/1.9780898719598
- Whittle, P. (1954). On stationary processes in the plane. *Biometrika* 41(3–4), 434–449. doi:10.1093/biomet/41.3-4.434
- Wilson, J.L. & Miller, P.J. (1978). Two-dimensional plume in uniform ground-water flow. *J. Hydraul. Div. ASCE* 104(HY4), 503–514. doi:10.1061/JYCEAJ.0004975

**Urban climate and radiation**
- Bonafoni, S. & Sekertekin, A. (2020). Albedo retrieval from Sentinel-2 by new narrow-to-broadband conversion coefficients. *IEEE Geosci. Remote Sens. Lett.* 17(9), 1618–1622. doi:10.1109/LGRS.2020.2967085
- Camuffo, D. & Bernardi, A. (1982). An observational study of heat fluxes and their relationships with net radiation. *Boundary-Layer Meteorol.* 23(3), 359–368. doi:10.1007/BF00121121
- Crawford, B., Grimmond, C.S.B., Gabey, A., Marconcini, M., Ward, H.C. & Kent, C.W. (2018). Variability of urban surface temperatures and implications for aerodynamic energy exchange in unstable conditions. *Q. J. R. Meteorol. Soc.* 144(715), 1719–1741. doi:10.1002/qj.3325 (The ≈ 40 s/m central-London value was taken from an indexed summary.)
- Grimmond, C.S.B., Cleugh, H.A. & Oke, T.R. (1991). An objective urban heat storage model and its comparison with other schemes. *Atmos. Environ. B* 25(3), 311–326. doi:10.1016/0957-1272(91)90003-W
- Grimmond, C.S.B. & Oke, T.R. (1999). Heat storage in urban areas: local-scale observations and evaluation of a simple model. *J. Appl. Meteorol.* 38(7), 922–940.
- Liang, S. (2001). Narrowband to broadband conversions of land surface albedo I: Algorithms. *Remote Sens. Environ.* 76(2), 213–238. doi:10.1016/S0034-4257(00)00205-4
- Masson, V. (2000). A physically-based scheme for the urban energy budget in atmospheric models. *Boundary-Layer Meteorol.* 94, 357–397. doi:10.1023/A:1002463829265 (Canyon exchange formula checked through secondary copies only.)
- Monteiro, M.V., Doick, K.J., Handley, P. & Peace, A. (2016). The impact of greenspace size on the extent of local nocturnal air temperature cooling in London. *Urban For. Urban Green.* 16, 160–169. doi:10.1016/j.ufug.2016.02.008
- Oke, T.R. (1973). City size and the urban heat island. *Atmos. Environ.* 7(8), 769–779. doi:10.1016/0004-6981(73)90140-6
- Oke, T.R. (1981). Canyon geometry and the nocturnal urban heat island. *J. Climatol.* 1(3), 237–254. doi:10.1002/joc.3370010304
- Oke, T.R., Mills, G., Christen, A. & Voogt, J.A. (2017). *Urban Climates*. Cambridge University Press. doi:10.1017/9781139016476
- Sailor, D.J. (2011). A review of methods for estimating anthropogenic heat and moisture emissions in the urban environment. *Int. J. Climatol.* 31(2), 189–199. doi:10.1002/joc.2106
- Shandas, V., Voelkel, J., Williams, J. & Hoffman, J. (2019). Integrating satellite and ground measurements for predicting locations of extreme urban heat. *Climate* 7(1), 5. doi:10.3390/cli7010005
- Varquez, A.C.G., Kiyomoto, S., Khanh, D.N. & Kanda, M. (2021). Global 1-km present and future hourly anthropogenic heat flux. *Sci. Data* 8, 64. doi:10.1038/s41597-021-00850-w (Typical hourly values per month, not a time series.)

**Causal inference and spatial confounding**
- Aronow, P.M. & Samii, C. (2017). Estimating average causal effects under general interference. *Ann. Appl. Stat.* 11(4), 1912–1947. doi:10.1214/16-AOAS1005
- Bongers, S., Forré, P., Peters, J. & Mooij, J.M. (2021). Foundations of structural causal models with cycles and latent variables. *Ann. Stat.* 49(5), 2885–2915. doi:10.1214/21-AOS2064
- Guan, Y., Page, G.L., Reich, B.J., Ventrucci, M. & Yang, S. (2023). Spectral adjustment for spatial confounding. *Biometrika* 110(3), 699–719. doi:10.1093/biomet/asac069
- Keller, J.P. & Szpiro, A.A. (2020). Selecting a scale for spatial confounding adjustment. *J. R. Stat. Soc. A* 183(3), 1121–1143. doi:10.1111/rssa.12556
- Mooij, J.M., Janzing, D. & Schölkopf, B. (2013). From ordinary differential equations to structural causal models: the deterministic case. *Proc. UAI 29*, 440–448. arXiv:1304.7920
- Paciorek, C.J. (2010). The importance of scale for spatial-confounding bias and precision of spatial regression estimators. *Stat. Sci.* 25(1), 107–125. doi:10.1214/10-STS326
- Wikle, N.B. & Zigler, C.M. (2024). Causal health impacts of power plant emission controls under modeled and uncertain physical process interference. *Ann. Appl. Stat.* 18(4), 2753–2774. doi:10.1214/24-AOAS1904

**Model checking and trust**
- Evans, M. & Moshonov, H. (2006). Checking for prior-data conflict. *Bayesian Anal.* 1(4), 893–914. doi:10.1214/06-BA129
- Kallioinen, N., Paananen, T., Bürkner, P.-C. & Vehtari, A. (2024). Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Stat. Comput.* 34, 57. doi:10.1007/s11222-023-10366-5
- Krishnapriyan, A.S., Gholami, A., Zhe, S., Kirby, R.M. & Mahoney, M.W. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS 34*. arXiv:2109.01050
- Oreskes, N., Shrader-Frechette, K. & Belitz, K. (1994). Verification, validation, and confirmation of numerical models in the Earth sciences. *Science* 263(5147), 641–646. doi:10.1126/science.263.5147.641
- Raue, A. et al. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25(15), 1923–1929. doi:10.1093/bioinformatics/btp358

**Forecast verification by scale**
- Briggs, W.M. & Levine, R.A. (1997). Wavelets and field forecast verification. *Mon. Weather Rev.* 125(6), 1329–1341. doi:10.1175/1520-0493(1997)125<1329:WAFFV>2.0.CO;2
- Casati, B., Ross, G. & Stephenson, D.B. (2004). A new intensity-scale approach for the verification of spatial precipitation forecasts. *Meteorol. Appl.* 11(2), 141–154. doi:10.1017/S1350482704001239

**Other domains (§2.2)**
- ISO 9613-2:1996, *Acoustics — Attenuation of sound during propagation outdoors — Part 2: General method of calculation* (revised 2024).
- Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in wildland fuels*. USDA Forest Service Research Paper INT-115.
