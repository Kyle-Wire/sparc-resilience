# SPARC Physics-Informed Loss Functions

Reference for the simulated-user skill. Each of the 11 PDE loss terms is explained at three levels so any persona can engage authentically.

---

## What Is a Physics-Informed Loss?

SPARC's neural network doesn't just minimize prediction error — it is also penalized for violating physical laws. The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \sum_{k=1}^{11} w_k \cdot \mathcal{L}_k$$

$\lambda_{\text{PDE}}$ ramps from 0 to 1 over the first 30 training epochs (outer curriculum). Each term $\mathcal{L}_k$ then activates on its own inner schedule (sub-curriculum), ramping over 5 epochs after its designated start.

**Why staged?** Activating all physics constraints at epoch 1 destabilizes a partially-converged network. The staged approach mirrors the way a human student learns — basic conservation first, higher-order corrections later.

---

## Staged Activation Schedule

| Epoch offset from PDE start | Terms active |
|-----------------------------|-------------|
| 0–5 | heat_diffusion only |
| 5–10 | + energy_balance |
| 10–15 | + directional + anisotropy |
| 15+ | + gradient_flux, gaussian_curv, alpha_smooth, alpha_prior |
| When temporal data present | + transient, nocturnal |
| When sheaf_delta provided | + sheaf |

---

## Term 1 — Heat Diffusion (`heat_diffusion`, weight 1.0)

**The equation:**  $\alpha(s)\nabla^2 T - S(s) \approx 0$

| Level | Explanation |
|-------|-------------|
| **Plain** | Heat spreads from hot places to cooler ones. This term penalizes the model if it predicts temperatures that couldn't physically diffuse the way heat would in the real world. |
| **Applied** | The Laplacian $\nabla^2 T$ measures how much a location's temperature differs from its neighbors. Multiplied by the local process rate $\alpha(s)$, this should equal the local source term $S(s)$ (impervious cover, waste heat, etc.). If it doesn't, the model is learning spatial patterns the physics can't support. |
| **Technical** | Discrete 4-neighbor finite-difference Laplacian on the irregular spatial graph. $\alpha(s)$ is a learned per-point diffusivity field (initialized from the Bayesian Matérn correlogram length scale). $S(s)$ is a learned source term with physical priors. Residual is normalized by running std to prevent scale dominance. This term carries the highest weight (1.0) and is active throughout training. |

**Persona hooks:**
- *Remote sensing scientist*: "How is $\alpha(s)$ initialized? Is it the thermal diffusivity of the surface material?"
- *Hydrologist*: "Does this correctly handle latent heat suppression over open water?"
- *Journalist*: "Basically it checks that heat flows downhill — and if the model pretends it doesn't, it gets penalized?"

---

## Term 2 — Energy Balance (`energy_balance`, weight 0.50)

**The equation:**  $Q^* - Q_H - Q_E \approx 0$

Where:
- $Q^*$ = net all-wave radiation (shortwave absorbed + longwave exchange)
- $Q_H$ = sensible heat flux (Fourier's law through surface layer)
- $Q_E$ = latent heat flux (Priestley-Taylor evapotranspiration)

| Level | Explanation |
|-------|-------------|
| **Plain** | Energy in must equal energy out. This term checks that the model's temperatures are consistent with how much solar radiation is absorbed vs. how much heat is released as evaporation and surface warming. |
| **Applied** | Uses the surface energy balance $Q^* = Q_H + Q_E + Q_S$. Albedo and vegetation fraction control how much radiation is absorbed vs. evaporated away. The model is penalized if the implied fluxes don't close the budget. Key parameter: `Priestley-Taylor α` (0.26 = dry, 1.26 = wet) — worth checking in `project.yml`. |
| **Technical** | $Q^*$ computed from Stefan-Boltzmann longwave (emissivity 0.95) + shortwave net. $Q_H$ from $-k\nabla^2 T \cdot d$ (depth 0.5 m, $k_{\text{soil}}$ = 1.5 W/m/K). $Q_E$ Priestley-Taylor with spatially varying $\alpha_{PT}$ from land cover. Residual normalized per-point. Activates at epoch offset +5. |

**Persona hooks:**
- *Soil scientist*: "The $k_{\text{soil}}$ default of 1.5 W/m/K — is that right for agricultural soils here? Irrigated fields could be quite different."
- *Coastal engineer*: "Is the latent heat model appropriate over estuarine surfaces?"
- *Climate risk analyst*: "If this term's residual is large in certain zones, does that flag unreliable CATE estimates there?"

---

## Term 3 — Directional Curvature (`directional`, weight 0.20)

**The equation:**  consistency of $\partial^2 T/\partial x^2$ and $\partial^2 T/\partial y^2$

| Level | Explanation |
|-------|-------------|
| **Plain** | Temperature gradients should be consistent in all directions unless there's a physical reason (like a wind corridor) for them not to be. |
| **Applied** | Checks that second-order spatial derivatives are physically consistent with the data's directional patterns. Prevents the model from "learning" curvature artifacts from sampling geometry. |
| **Technical** | Finite-difference second derivatives in x and y using the 4-neighbor stencil. Penalizes inconsistency between the two principal curvature axes relative to anisotropy observed in the data. Activates at epoch offset +10 alongside anisotropy. |

---

## Term 4 — Anisotropy (`anisotropy`, weight 0.10)

| Level | Explanation |
|-------|-------------|
| **Plain** | If the temperature data has a directional pattern (e.g., a wind-aligned heat corridor), the model should reflect that, not wash it out into a blob. |
| **Applied** | Where the Matérn correlogram reveals anisotropic spatial structure (different range parameters in different directions), spurious isotropy in the predicted field is penalized. |
| **Technical** | Penalizes cases where $\partial^2 T/\partial x^2 \approx \partial^2 T/\partial y^2$ when the Matérn fit indicates anisotropy ratio $> 1.5$. Co-activates with directional at epoch offset +10. |

---

## Term 5 — Gradient Flux (`gradient_flux`, weight 0.10)

**The equation:**  Fourier's law consistency $\mathbf{q} = -\alpha \nabla T$

| Level | Explanation |
|-------|-------------|
| **Plain** | Heat flow direction should be consistent with the temperature gradient — heat flows from hot to cold, and the amount depends on how steep the gradient is. |
| **Applied** | Verifies that the implied heat flux vector field is consistent with the learned $\alpha(s)$ and the spatial gradient of $T$. Catches cases where the model learns an $\alpha$ that implies impossible heat flow directions. |
| **Technical** | Computes $\|\nabla T\|^2$ via the 4-neighbor gradient magnitude operator, then checks consistency with the divergence of $\alpha\nabla T$. Residual normalized by local gradient magnitude. Activates at epoch offset +15. |

---

## Term 6 — Gaussian Curvature Regularizer (`gaussian_curv`, weight 0.05)

| Level | Explanation |
|-------|-------------|
| **Plain** | Prevents the model from producing extreme "spikes" or "pits" in the temperature field that aren't physically plausible. |
| **Applied** | Penalizes large values of $\det(H(T))$ — the Hessian determinant — which correspond to saddle points or extreme local curvature. Acts as a smoothness prior on the second-order structure. |
| **Technical** | Computes Hessian invariants $(H_{xx}H_{yy} - H_{xy}^2)$ from finite differences. Penalizes large absolute values, with a soft threshold to allow genuine physical features (e.g., hotspot peaks). Activates at epoch offset +15. |

---

## Term 7 — Alpha Smoothness (`alpha_smooth`, weight 0.10)

| Level | Explanation |
|-------|-------------|
| **Plain** | The model learns a different "heat responsiveness" value for each location. This term ensures those values change gradually across the landscape, not erratically from pixel to pixel. |
| **Applied** | The learned process rate $\alpha(s)$ should vary smoothly unless land cover or material properties change abruptly. Prevents overfitting in the $\alpha$ field. |
| **Technical** | Penalizes $\|\nabla \alpha\|^2$ using the same 4-neighbor gradient operator. Spatially varying regularization: higher penalty in homogeneous land cover zones, relaxed at known boundary conditions (major land cover transitions). Activates at epoch offset +15. |

**Persona hooks:**
- *Urban ecologist*: "Does this smooth across habitat patch boundaries? I don't want park edge effects washed out."
- *Geotechnical engineer*: "The $\alpha$ field should be discontinuous at material boundaries — does this term handle that?"

---

## Term 8 — Alpha Prior (`alpha_prior`, weight 0.10)

| Level | Explanation |
|-------|-------------|
| **Plain** | We have a prior belief about what "heat responsiveness" should be in each land cover type. This term penalizes the model for straying too far from those expectations without evidence. |
| **Applied** | The prior $\alpha_0(s)$ is a spatial mixture of land-cover-specific diffusivity values (e.g., impervious ~ 0.8, vegetation ~ 0.3, water ~ 0.6). The penalty is $\|\alpha(s) - \alpha_0(s)\|^2$, weighted by prior confidence. |
| **Technical** | $\alpha_0$ is constructed as a mixture prior: each land cover class contributes a Gaussian component; the mixture weight is the fractional coverage at each point. The penalty is the KL-weighted squared deviation from this mixture mean. Activates at epoch offset +15. |

---

## Term 9 — Transient / Temporal (`transient`, weight 0.05)

*Activates only when multi-snapshot data is provided (`T_prev`, `dt_hours` in project.yml).*

**The equation:**  $\partial T / \partial t \approx \alpha \nabla^2 T - S$

| Level | Explanation |
|-------|-------------|
| **Plain** | If you have temperature data from multiple time points, this term checks that the model's predicted change over time is physically consistent with how heat diffuses. |
| **Applied** | Enforces the transient heat equation across snapshots. The observed $\Delta T / \Delta t$ between two timestamps should match the spatial diffusion predicted by the steady-state model. |
| **Technical** | Forward finite-difference temporal derivative $\approx (T_t - T_{t-1}) / \Delta t$. Compared against $\alpha\nabla^2 T - S$ evaluated at time $t$. Normalized residual. Only activates when `T_prev` and `dt_hours` are non-null. |

---

## Term 10 — Nocturnal Cooling (`nocturnal`, weight 0.08)

*Activates only when nighttime temperature data is available.*

| Level | Explanation |
|-------|-------------|
| **Plain** | At night, solar heating stops. The model should predict nighttime cooling patterns that are consistent with how quickly heat radiates away from different surfaces. |
| **Applied** | Nighttime acts as a natural experiment — source term $S \approx 0$, so any remaining temperature gradient is purely due to stored heat and radiative cooling. Penalizes models that can't reproduce this regime. |
| **Technical** | With $S \approx 0$ at night, enforces $\partial T_{\text{night}} / \partial t \approx \alpha \nabla^2 T_{\text{night}}$. Strong constraint on $\alpha(s)$ because the source confound is removed. Particularly valuable for separating thermal mass effects from ongoing anthropogenic heat sources. |

**Persona hooks:**
- *Urban heat researcher*: "This is essentially using nighttime as a thermal tracer — very clean. What's the minimum number of nighttime snapshots needed?"
- *Remote sensing scientist*: "What's the LST product used for $T_{\text{night}}$? MODIS Terra or Aqua overpass timing matters here."

---

## Term 11 — Sheaf Laplacian / MAUP Resistance (`sheaf`, weight 0.03)

*Activates only when `sheaf_delta` (coboundary operator $\delta^0$) is provided.*

| Level | Explanation |
|-------|-------------|
| **Plain** | Different data layers (census tracts, weather stations, satellite pixels) don't line up perfectly. This term ensures the model's predictions are consistent across those different spatial scales, so the answer doesn't change just because you zoomed in or out. |
| **Applied** | The Modifiable Areal Unit Problem (MAUP): aggregating spatial data to different units changes apparent patterns. The Sheaf Laplacian penalizes predictions that are inconsistent when observed at multiple spatial resolutions simultaneously. Critical for health outcomes mapped at census tract vs. point-level temperature predictors. |
| **Technical** | Given a cellular sheaf $\mathcal{F}$ over the spatial graph with restriction maps from fine to coarse resolutions, the sheaf Laplacian coboundary $\delta^0$ measures inter-scale consistency. $\mathcal{L}_{\text{sheaf}} = \|\delta^0 \mathbf{T}\|^2$ penalizes predictions that are self-inconsistent across scales. Lowest weight (0.03) — a regularizer, not a primary constraint. Requires explicit graph construction by the user; see `project.yml` `sheaf_resolution_levels`. |

**Persona hooks:**
- *Epidemiologist*: "This is exactly the MAUP problem I run into constantly. Does this actually work in practice or is it mostly theoretical?"
- *Federal reviewer*: "If this term is disabled, does that invalidate the multi-scale claims in the methods section?"
- *Graduate researcher*: "Is the sheaf here the constant sheaf, or do the restriction maps carry meaningful geometric information?"

---

## Reading the Loss History

The file `output/Stage_2_Spatial_CV/v2_neural/loss_history.npz` contains per-epoch, per-term loss values. When browsing as a simulated user:

1. Check that `heat_diffusion` loss decreases monotonically after activation
2. Flag if `energy_balance` residual plateaus high — suggests misconfigured land cover or albedo
3. Note the epoch when each term activates — compare against the schedule above
4. A **rising** `alpha_prior` loss with a **falling** `alpha_smooth` loss is a signal of overfitting in the $\alpha$ field
5. If `sheaf` is missing from the history, the project hasn't provided multi-resolution data

---

## Domain Equivalents

For non-thermal domains, the physics terms map analogously:

| SPARC (thermal) | Hydrology | Air Quality | Wildfire | Noise |
|-----------------|-----------|-------------|---------|-------|
| heat_diffusion | Darcy's law $K\nabla^2 h$ | Fickian dispersion | Fire spread rate | Wave propagation |
| energy_balance | Water balance $P - ET - Q$ | Source-receptor balance | Fire-weather energy | Sound power balance |
| alpha(s) | Hydraulic conductivity $K(s)$ | Eddy diffusivity $K_z(s)$ | Spread rate coefficient | Propagation loss |
| Source term $S(s)$ | Recharge / pumping | Emission flux | Spotting ignition | Point source power |
| Transient | Storage $S_s \partial h/\partial t$ | Unsteady advection | Fire progression | Time-varying source |

When a non-thermal domain is detected from `project.yml`, substitute these equivalents in the persona's questions.
