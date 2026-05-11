# SPARC Domain Atlas

**SPARC Labs LLC | May 2026**
**Every Field Where Spatial Causal Inference Changes the Answer**

---

## How to Read This Atlas

Each domain section contains:

- **What SPARC solves** — the core spatial causal question
- **Spatial outcome variable** — what SPARC predicts
- **Example use cases** — concrete applications
- **Governing physics** — the PDE that constrains the spatial process
- **Process-rate analog** — the domain's equivalent of thermal diffusivity $\alpha(\mathbf{x})$, the spatially-varying learned field
- **Causal structure** — what the treatment, mediators, and outcomes typically look like
- **Physics module status** — whether existing SPARC physics can be reused or a new PDE loss module is required
- **New module spec** (if required) — conceptual description of what would need to be implemented
- **Existing template** — which SPARC template maps to this domain, if any

---

## Domain Clusters

1. [Health & Epidemiology](#1-health--epidemiology)
2. [Public Safety & Crime](#2-public-safety--crime)
3. [Neuroscience & Neural Mapping](#3-neuroscience--neural-mapping)
4. [Climate & Ecology](#4-climate--ecology)
5. [Infrastructure & Engineering](#5-infrastructure--engineering)
6. [Agriculture & Food Systems](#6-agriculture--food-systems)
7. [Economics & Social Science](#7-economics--social-science)
8. [Defense & Intelligence](#8-defense--intelligence)
9. [Energy Systems](#9-energy-systems)
10. [Oceans & Marine](#10-oceans--marine)

---

## 1. Health & Epidemiology

### What SPARC Solves
Disease burden is not randomly distributed across space. Environmental exposures — air pollution, heat, flood proximity, green space access, food environment — causally shape health outcomes in ways that vary by neighborhood. SPARC identifies where disease burden concentrates, what environmental factors drive it, and what interventions would reduce it.

### Spatial Outcome Variable
- Incidence rate (disease cases per population per unit area per time)
- Hospitalization rate, mortality rate
- Exposure burden (PM2.5 concentration, heat index, noise level)
- Social determinant index (food access score, walkability, healthcare proximity)

### Example Use Cases
- Identify census tracts with anomalously high asthma rates and causally attribute them to traffic-related air pollution after controlling for poverty
- Estimate the causal effect of green space access on cardiovascular mortality across a metro area
- Map spatial heterogeneity in COVID-19 excess mortality and identify structural mediators
- Predict malaria transmission risk from satellite-derived vegetation and water body data in low-resource settings
- Design optimal health center placement to minimize travel burden for highest-risk populations

### Governing Physics

The **FKPP reaction-diffusion equation** governs the spread of infectious disease through a population distributed across space:

$$\frac{\partial I(\mathbf{x}, t)}{\partial t} = D(\mathbf{x})\nabla^2 I - \nabla \cdot \left[\mathbf{v}(\mathbf{x}) I\right] + \beta(\mathbf{x}) S(\mathbf{x}) I - \gamma(\mathbf{x}) I$$

Where:
- $I(\mathbf{x}, t)$ — infectious population density at location $\mathbf{x}$ and time $t$
- $D(\mathbf{x})$ — spatial diffusion coefficient (human mobility)
- $\mathbf{v}(\mathbf{x})$ — directional mobility flow (commuting patterns, transit corridors)
- $\beta(\mathbf{x})$ — spatially-varying transmission rate
- $S(\mathbf{x})$ — susceptible population density
- $\gamma(\mathbf{x})$ — recovery rate

For non-infectious disease burden and chronic exposure, a **steady-state diffusion-reaction** model applies:

$$-\nabla \cdot \left[D(\mathbf{x})\nabla u\right] + r(\mathbf{x}) u = f(\mathbf{x})$$

Where $u$ is the burden field, $D(\mathbf{x})$ is the spatial smoothing (environmental exposure diffuses across neighborhoods), $r(\mathbf{x})$ is a decay term (access to care reduces burden), and $f(\mathbf{x})$ is the source term (pollution sources, food deserts, proximity to toxics).

### Process-Rate Analog
$D(\mathbf{x})$ — **spatial mobility/diffusion coefficient** — the rate at which disease risk or environmental exposure spreads from high-burden to lower-burden areas. High $D$ in transit corridors; low $D$ in isolated communities. Learned spatially from mobility data, transit access, and barrier features.

### Causal Structure
- **Treatment:** Environmental exposure (PM2.5, heat, flood risk, green space)
- **Mediators:** Healthcare access, income, age distribution, pre-existing conditions
- **Outcome:** Disease incidence, hospitalization, mortality
- **Confounders:** Socioeconomic status, historical redlining, built environment

### Physics Module Status
⚠️ **New module required.** The heat diffusion PDE in `sparc/physics/pde_loss.py` is structurally similar but has domain-specific source and reaction terms that differ from thermal processes.

### New Module Spec
`sparc/physics/epidemiology_pde.py` — Implements:
- Steady-state diffusion-reaction loss: $\mathcal{L}_{diff} = \|\nabla \cdot [D\nabla u] - r u - f\|^2$
- Spatial smoothness regularizer on $D(\mathbf{x})$ (mobility doesn't change discontinuously)
- Non-negativity constraint on burden field $u \geq 0$
- Population-weighted boundary conditions (burden at domain boundary matches regional rates)
- Optional: SEIR compartmental structure for infectious disease mode

### Existing Template
`templates/air_quality/` (partial reuse for exposure mapping); no dedicated epidemiology template yet.

---

## 2. Public Safety & Crime

### What SPARC Solves
Crime is spatially clustered — not randomly distributed — and environmental criminology has a rich theoretical basis explaining why. SPARC goes beyond hotspot mapping to answer: what environmental conditions *cause* crime concentration, what interventions (lighting, green space, broken windows remediation, police presence) causally reduce it, and where those interventions have the largest effect.

### Spatial Outcome Variable
- Crime event density (incidents per unit area per time period)
- Crime type-specific rates (violent, property, drug-related)
- Fear of crime / perceived safety indices
- Police response time distribution

### Example Use Cases
- Identify which environmental features causally drive robbery hotspots after controlling for poverty and demographics
- Estimate the causal effect of street lighting improvement on nighttime crime
- Map spatial heterogeneity in the effect of vacant lot remediation on violent crime
- Evaluate a police resource reallocation policy for unintended spillover effects in neighboring areas
- Model the diffusion of crime displacement after a targeted intervention

### Governing Physics

The **Short et al. crime hotspot model** (2008) describes the spatiotemporal dynamics of crime attractiveness and offender density:

$$\frac{\partial A(\mathbf{x}, t)}{\partial t} = \eta \nabla^2 A - \omega A + \theta \rho(\mathbf{x}, t) + A_0(\mathbf{x})$$

$$\frac{\partial \rho(\mathbf{x}, t)}{\partial t} = \nabla \cdot \left[\rho \nabla \ln A\right] - \rho A + \gamma$$

Where:
- $A(\mathbf{x}, t)$ — crime attractiveness field (risk surface)
- $\rho(\mathbf{x}, t)$ — criminal/offender density
- $\eta$ — attractiveness diffusion coefficient (how crime risk spreads spatially)
- $\omega$ — attractiveness decay rate (how quickly risk decays without reinforcement)
- $\theta$ — reinforcement rate (successful crime increases local attractiveness)
- $A_0(\mathbf{x})$ — baseline attractiveness from static environment features
- The second equation: criminals perform a biased random walk toward high-attractiveness areas

For static or slow-moving spatial analysis, the steady-state attractiveness field simplifies to:

$$-\eta \nabla^2 A + \omega A = \theta \rho + A_0(\mathbf{x})$$

### Process-Rate Analog
$\eta(\mathbf{x})$ — **attractiveness diffusion rate** — how quickly crime risk spreads spatially from a hotspot. High $\eta$ in areas with high pedestrian connectivity; low $\eta$ near physical barriers (highways, rivers, parks). Learned spatially from street network connectivity, land use, and barrier features.

### Causal Structure
- **Treatment:** Environmental intervention (lighting, greening, infrastructure repair, policing strategy)
- **Mediators:** Routine activity patterns, guardianship, visibility
- **Outcome:** Crime event density by type
- **Confounders:** Poverty, demographics, housing density, historical patterns

### Physics Module Status
🔲 **New module required.** The crime attractiveness PDE is structurally distinct from the heat equation — it has a bilinear coupling between $A$ and $\rho$ that has no analog in thermal physics.

### New Module Spec
`sparc/physics/crime_pde.py` — Implements:
- Static attractiveness loss: $\mathcal{L}_{attr} = \|-\eta\nabla^2 A + \omega A - \theta\rho - A_0\|^2$
- Non-negativity constraint on $A$ and $\rho$
- Spatial smoothness on $\eta(\mathbf{x})$ (diffusion doesn't change abruptly at most locations)
- Boundary condition: crime attractiveness at domain edge matches county-level background rate
- Intervention mode: zero-out $A_0$ in treated areas and simulate new steady-state

### Existing Template
None. Crime is a new domain for SPARC.

---

## 3. Neuroscience & Neural Mapping

### What SPARC Solves
Brain imaging data is inherently spatial — neural activity, connectivity, lesion effects, and disorder-related changes are all distributed across a 3D spatial manifold (the cortical surface or volumetric brain space). SPARC's spatial causal inference applies directly: identify where a stimulus, lesion, or condition *causes* changes in neural activity, quantify spatial heterogeneity in neural responses, and map structure-to-function relationships with uncertainty.

### Spatial Outcome Variable
- BOLD signal magnitude (fMRI)
- Neural firing rate / local field potential (electrophysiology)
- Cortical thickness (structural MRI)
- White matter tract integrity (diffusion tensor imaging)
- Connectivity strength between regions (functional connectivity)

### Example Use Cases
- Map the causal effect of a transcranial magnetic stimulation (TMS) pulse on spatially distributed BOLD activity
- Identify which white matter tracts causally mediate the relationship between a brain lesion and a cognitive deficit
- Estimate spatially-varying treatment response to a neural intervention across patient cohorts
- Characterize the spatial diffusion of neural excitation from a seizure focus
- Map population-level spatial heterogeneity in cortical thickness and its causal relationship to age, disease status, and cognitive performance

### Governing Physics

The **Wilson-Cowan neural field equation** describes the spatiotemporal dynamics of neural excitation across a continuous cortical surface:

$$\tau \frac{\partial u(\mathbf{x}, t)}{\partial t} = -u(\mathbf{x}, t) + \mathcal{S}\left(\int w(\mathbf{x}, \mathbf{x}') u(\mathbf{x}', t) \, d\mathbf{x}' + I(\mathbf{x}, t)\right)$$

Where:
- $u(\mathbf{x}, t)$ — mean neural activity at cortical location $\mathbf{x}$ and time $t$
- $\tau$ — time constant of neural response
- $w(\mathbf{x}, \mathbf{x}')$ — synaptic connectivity kernel (how strongly location $\mathbf{x}'$ drives $\mathbf{x}$)
- $\mathcal{S}(\cdot)$ — sigmoid activation function
- $I(\mathbf{x}, t)$ — external input (stimulus, TMS pulse)

For spatial smoothing of static structural imaging (cortical thickness, connectivity maps), the diffusion equation with source terms applies:

$$-D(\mathbf{x})\nabla^2 u + \lambda u = f(\mathbf{x})$$

Where $D(\mathbf{x})$ encodes local cortical geometry (curvature, sulcal depth) and $f(\mathbf{x})$ is the measured signal.

### Process-Rate Analog
$w(\mathbf{x}, \mathbf{x}')$ — **synaptic connectivity kernel** — the spatial weight function governing how strongly nearby cortical regions influence each other. This is the neural analog of thermal diffusivity: how readily "activation" spreads across the cortical surface. Learned from functional connectivity data; constrained by white matter tractography.

### Causal Structure
- **Treatment:** Neural stimulation (TMS, tDCS, DBS), drug, lesion, stimulus
- **Mediators:** Connectivity pathways, cortical excitability
- **Outcome:** BOLD activity, behavioral performance, symptom severity
- **Confounders:** Age, baseline connectivity, individual anatomy

### Physics Module Status
🔲 **New module required.** The neural field equation's convolution integral and sigmoid nonlinearity are structurally distinct from diffusion PDEs.

### New Module Spec
`sparc/physics/neural_field_pde.py` — Implements:
- Steady-state neural field loss on a cortical surface mesh: $\mathcal{L}_{nf} = \|u - \mathcal{S}(w \ast u + I)\|^2$
- Matérn approximation for the connectivity kernel $w(\mathbf{x}, \mathbf{x}')$ (smoothly decaying with cortical distance)
- Excitation-inhibition balance constraint: integral of $w$ over the cortical surface must be bounded
- Anatomical boundary conditions: no signal crosses sulcal boundaries without white matter support
- Note: brain coordinates require registration to a common atlas (MNI152 or fsaverage) before SPARC processing

### Existing Template
None. Neuroscience is a new domain. Note: SPARC's 2D spatial model can be applied to unfolded cortical surface representations; full 3D volumetric analysis requires a surface-to-volume projection step outside SPARC.

---

## 4. Climate & Ecology

### What SPARC Solves
This is SPARC's founding domain. The extensions beyond Urban Heat Island into broader climate and ecology are natural: any process governed by energy balance, mass transport, or reaction-diffusion on a spatial surface maps directly onto SPARC's existing physics infrastructure.

### Spatial Outcome Variables
- Air / surface temperature (UHI, heat island analysis)
- Precipitation / runoff / soil moisture
- Species occurrence probability / habitat suitability
- Carbon stock / flux
- Vegetation phenology (green-up dates, growing season length)
- Sea level, coastal inundation probability

### Example Use Cases
- UHI causal attribution (existing, fully supported)
- ForceSMIP climate forcing attribution (existing, fully supported)
- Biodiversity hotspot identification with habitat degradation causal analysis
- Carbon stock mapping from satellite spectral indices with land use change causal attribution
- Wildfire risk spatial prediction with causal attribution to fuels, drought, ignition sources
- Drought prediction and groundwater depletion causal analysis

### Governing Physics

**Heat equation (UHI)** — existing, fully wired:
$$\frac{\partial T}{\partial t} = \alpha(\mathbf{x})\nabla^2 T + S(\mathbf{x}) - R(\mathbf{x}) T$$

**Richards equation (soil moisture / groundwater)**:
$$\frac{\partial \theta}{\partial t} = \nabla \cdot \left[K(\theta, \mathbf{x})\nabla h\right] + q(\mathbf{x})$$

Where $\theta$ is volumetric water content, $K(\theta, \mathbf{x})$ is hydraulic conductivity, $h$ is hydraulic head, $q$ is source/sink (recharge, extraction). Supported by `templates/groundwater/`.

**Advection-diffusion (precipitation / pollutant transport)**:
$$\frac{\partial C}{\partial t} + \mathbf{v} \cdot \nabla C = D(\mathbf{x})\nabla^2 C + S(\mathbf{x})$$

**Lotka-Volterra reaction-diffusion (species spread)**:
$$\frac{\partial u_i}{\partial t} = D_i(\mathbf{x})\nabla^2 u_i + r_i(\mathbf{x}) u_i \left(1 - \frac{\sum_j \alpha_{ij} u_j}{K_i(\mathbf{x})}\right)$$

### Process-Rate Analog
$\alpha(\mathbf{x})$ (thermal diffusivity), $K(\theta, \mathbf{x})$ (hydraulic conductivity), $D_i(\mathbf{x})$ (species dispersal rate) — all spatially-varying learned fields, directly analogous to the existing ProcessRateNet.

### Causal Structure
- **Treatment:** Land use change, infrastructure, emissions policy, conservation intervention
- **Outcome:** Temperature, moisture, species occurrence, carbon flux
- **Confounders:** Elevation, soil type, proximity to water, historical land cover

### Physics Module Status
✅ **Largely reusable.** Heat equation, diffusion-advection, and Richards equation variants are either already in `sparc/physics/pde_loss.py` or in existing templates. Lotka-Volterra requires a new reaction term.

### New Module Spec (partial)
`sparc/physics/ecology_pde.py` — Adds:
- Logistic growth reaction term: $r(\mathbf{x})u(1 - u/K(\mathbf{x}))$
- Species competition coupling: $-\alpha_{ij} u_i u_j$
- Habitat suitability as a spatially-varying carrying capacity $K(\mathbf{x})$

### Existing Templates
`templates/uhi/`, `templates/air_quality/`, `templates/drought/`, `templates/groundwater/`, `templates/wildfire/`, `templates/coastal/`, `templates/water_quality/`, `templates/stormwater/`, `templates/forcesmip/`

---

## 5. Infrastructure & Engineering

### What SPARC Solves
Infrastructure systems — roads, bridges, utilities, buildings — degrade spatially according to physical laws (fatigue, corrosion, thermal cycling). Their failure risk is spatially autocorrelated (aging infrastructure clusters geographically), and interventions (maintenance, retrofit, replacement) have spatially-varying cost-effectiveness. SPARC estimates where failure risk is highest, what causes it, and which interventions are most cost-effective per unit of risk reduction.

### Spatial Outcome Variables
- Bridge / pavement condition score
- Pipeline failure probability
- Building damage index (seismic, wind, flood)
- Traffic density / congestion index
- Utility outage probability

### Example Use Cases
- Map bridge failure risk across a state road network, causally attributed to traffic load, age, and freeze-thaw cycles
- Estimate the causal effect of preventive maintenance schedules on pavement condition
- Identify which neighborhoods are most vulnerable to combined sewer overflow during heavy rain events
- Predict seismic damage distribution for a scenario earthquake
- Optimize utility infrastructure investment to maximize resilience per dollar

### Governing Physics

**Linear elasticity / fatigue (structural integrity)**:
$$\nabla \cdot \boldsymbol{\sigma}(\mathbf{x}) + \mathbf{b}(\mathbf{x}) = \mathbf{0}$$
$$\boldsymbol{\sigma} = \mathbf{C}(\mathbf{x}) : \boldsymbol{\varepsilon}$$

Where $\boldsymbol{\sigma}$ is the stress tensor, $\mathbf{C}(\mathbf{x})$ is the spatially-varying stiffness tensor, $\boldsymbol{\varepsilon}$ is the strain tensor, and $\mathbf{b}$ is the body force. In practice for spatial prediction, this reduces to a scalar surrogate for structural condition.

**LWR kinematic wave equation (traffic)**:
$$\frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x}\left[\rho v(\rho)\right] = q(\mathbf{x}, t)$$

Where $\rho$ is vehicle density, $v(\rho)$ is the fundamental diagram (speed as a function of density), and $q$ is source/sink (on-ramps, off-ramps).

**Seismic attenuation (ground motion)**:
$$\text{PGA}(\mathbf{x}) = A \cdot \exp\left(-b_1 R(\mathbf{x})\right) \cdot R(\mathbf{x})^{-b_2} \cdot f(\mathbf{x}_{\text{soil}})$$

Where PGA is peak ground acceleration, $R(\mathbf{x})$ is distance from source, $f(\mathbf{x}_{\text{soil}})$ is a site amplification factor from soil type. Supported by `templates/seismic/`.

### Process-Rate Analog
$\mathbf{C}(\mathbf{x})$ — **local material stiffness** — spatially-varying structural capacity. For traffic: $v(\rho, \mathbf{x})$ — location-specific speed-density relationship. Learned from material properties, age, maintenance history.

### Causal Structure
- **Treatment:** Maintenance intervention, load restriction, retrofit, policy change
- **Outcome:** Condition score, failure probability, congestion level
- **Confounders:** Age, traffic volume, climate exposure, soil conditions

### Physics Module Status
⚠️ **Partially reusable.** Seismic attenuation is already in `templates/seismic/`. Linear elasticity surrogate and LWR traffic model require new PDE loss terms.

### New Module Spec
`sparc/physics/structural_pde.py` — Implements:
- Scalar structural condition decay loss: $\mathcal{L}_{str} = \|{-D\nabla^2 c + \lambda c - f_{load}}\|^2$ where $c$ is condition score
- Non-negativity constraint on condition field
- Maintenance intervention as a source term: $f_{maint}(\mathbf{x}) > 0$ in treated areas

`sparc/physics/traffic_pde.py` — Implements:
- Conservation of vehicles: $\mathcal{L}_{LWR} = \|\partial_t\rho + \partial_x(\rho v(\rho))\|^2$
- Fundamental diagram constraint: $v(\rho)$ is monotone decreasing in $\rho$

### Existing Templates
`templates/seismic/`, `templates/geotechnical/`, `templates/stormwater/`

---

## 6. Agriculture & Food Systems

### What SPARC Solves
Crop yield, soil health, and water availability are spatially heterogeneous processes governed by well-understood physics (water transport, nutrient diffusion, photosynthesis). SPARC quantifies where yield is below potential, causally attributes the gap to soil, water, input, or management factors, and optimizes intervention placement — the precision agriculture problem at scale.

### Spatial Outcome Variables
- Crop yield (tons per hectare)
- Soil organic carbon (SOC) stock
- Volumetric soil water content
- Plant nitrogen / phosphorus status
- Pest or pathogen infestation probability

### Example Use Cases
- Map yield gaps across a watershed and causally attribute them to soil organic carbon deficits vs. water stress vs. input application rates
- Estimate the causal effect of cover cropping on soil carbon accumulation across heterogeneous soil types
- Predict irrigation demand spatial distribution under climate scenarios
- Identify which field zones would respond most to targeted fertilizer application (spatial CATE for precision nutrient management)
- Forecast pest spread risk from a known infestation origin point

### Governing Physics

**Richards equation (soil water transport)** — same as groundwater but at field scale:
$$C(\psi, \mathbf{x})\frac{\partial \psi}{\partial t} = \nabla \cdot \left[K(\psi, \mathbf{x})(\nabla \psi + \mathbf{e}_z)\right] - S_r(\mathbf{x})$$

Where $\psi$ is matric potential, $C = \partial\theta/\partial\psi$ is the specific moisture capacity, $K(\psi, \mathbf{x})$ is unsaturated hydraulic conductivity, $\mathbf{e}_z$ is the unit vector in the gravitational direction, and $S_r$ is root water uptake.

**Nutrient transport (advection-diffusion-reaction)**:
$$\frac{\partial C_N}{\partial t} = \nabla \cdot \left[D_N(\mathbf{x})\nabla C_N\right] - \nabla \cdot \left[\mathbf{q} C_N\right] + r_N(\mathbf{x}) + f_{app}(\mathbf{x})$$

Where $C_N$ is nutrient concentration, $D_N(\mathbf{x})$ is diffusion-dispersion coefficient, $\mathbf{q}$ is Darcy flux (water carrying nutrients), $r_N$ is mineralization/immobilization rate, and $f_{app}$ is fertilizer application.

**Photosynthesis-respiration (Farquhar model spatial surrogate)**:
$$\text{GPP}(\mathbf{x}) = \min\left(A_c(\mathbf{x}), A_j(\mathbf{x})\right) - R_d(\mathbf{x})$$

Where GPP is gross primary productivity, $A_c$ is Rubisco-limited assimilation, $A_j$ is RuBP-limited assimilation (light reaction), and $R_d$ is dark respiration. SPARC uses a simplified scalar surrogate driven by NDVI, soil moisture, and temperature.

### Process-Rate Analog
$K(\psi, \mathbf{x})$ — **unsaturated hydraulic conductivity** — the soil's spatially-varying water transmission capacity. This is the agricultural analog of thermal diffusivity: how readily moisture moves through the soil at each location. Learned from soil texture, organic matter, and bulk density data.

### Causal Structure
- **Treatment:** Irrigation, fertilizer application, tillage practice, crop variety
- **Mediators:** Soil water availability, nutrient status, plant stress
- **Outcome:** Yield, SOC, water use efficiency
- **Confounders:** Soil type, drainage class, slope, historical land use

### Physics Module Status
⚠️ **Partially reusable.** Richards equation is already supported through `templates/groundwater/` and `templates/drought/`. Nutrient transport and photosynthesis surrogate require extensions.

### New Module Spec
`sparc/physics/agriculture_pde.py` — Implements:
- Nutrient transport loss: $\mathcal{L}_{nut} = \|\partial_t C_N - \nabla\cdot(D_N\nabla C_N) + \nabla\cdot(\mathbf{q}C_N) - r_N\|^2$
- Yield-soil-water coupling: $\text{Yield}(\mathbf{x}) \sim f(\theta(\mathbf{x}), C_N(\mathbf{x}), \text{GPP}(\mathbf{x}))$ as a learned surrogate
- Non-negativity on all concentration fields

### Existing Templates
`templates/groundwater/`, `templates/drought/`, `templates/water_quality/`

---

## 7. Economics & Social Science

### What SPARC Solves
Economic and social phenomena are profoundly spatial — property values, income, employment, poverty, and social mobility all exhibit strong spatial autocorrelation. SPARC brings causal rigor to questions that are usually answered with correlational spatial regressions: does opening a transit station *cause* property value increases, or do they simply co-locate? Does neighborhood greenery *cause* health improvements, or do healthier neighborhoods attract more investment?

### Spatial Outcome Variables
- Property value / rental price per unit area
- Income / wages at census block level
- Poverty rate, Gini coefficient, social mobility index
- Employment rate / labor market participation
- Educational attainment / school performance index

### Example Use Cases
- Estimate the causal effect of a new transit station on surrounding property values, controlling for pre-existing trends (DiD with spatial spillovers)
- Identify neighborhoods where targeted investment would have the highest causal impact on social mobility
- Map spatial heterogeneity in the effect of minimum wage increases on local employment
- Estimate whether gentrification displacement is a causal outcome of urban green space investment
- Predict spatial distribution of poverty under different housing policy scenarios

### Governing Physics

Economic spatial processes are governed not by physical PDEs but by **economic equilibrium and spatial interaction models**. SPARC treats these as "soft physics" — constraints that encode economic behavior rather than physical law.

**Spatial hedonic equilibrium (Rosen-Roback)**:
$$p(\mathbf{x}) = \mathbf{X}(\mathbf{x})^T \boldsymbol{\beta} + f(\mathbf{a}(\mathbf{x})) + \epsilon(\mathbf{x})$$

With spatial autocorrelation structure:
$$\epsilon(\mathbf{x}) = \rho W \epsilon + \nu(\mathbf{x})$$

Where $p(\mathbf{x})$ is log property value, $\mathbf{X}(\mathbf{x})$ are structural and neighborhood attributes, $\mathbf{a}(\mathbf{x})$ is the amenity vector, $W$ is a spatial weights matrix, $\rho$ is the spatial autocorrelation parameter.

**Spatial diffusion of economic shocks (gravity model)**:
$$\frac{\partial y(\mathbf{x}, t)}{\partial t} = D(\mathbf{x})\nabla^2 y - \lambda(\mathbf{x}) y + S(\mathbf{x}, t)$$

Where $y$ is income growth, $D(\mathbf{x})$ encodes how quickly economic gains spread spatially (agglomeration gradient), $\lambda$ is a decay term, and $S$ is an external shock (plant opening, transit investment, policy change).

### Process-Rate Analog
$D(\mathbf{x})$ — **economic agglomeration diffusivity** — how rapidly economic benefits spread from their origin location. High in connected, high-density cores; low in isolated suburban or rural areas. Learned from commuting flows, transit access, and economic connectivity data.

### Causal Structure
- **Treatment:** Policy (transit investment, zoning change, subsidy), external shock (plant opening/closing, natural disaster)
- **Mediators:** Amenity access, labor market access, social network density
- **Outcome:** Property values, income, employment, mobility
- **Confounders:** Pre-existing trends, initial income level, demographics, historical investment patterns

### Physics Module Status
⚠️ **Soft physics — new module required.** Economic processes don't have PDEs in the physical sense, but spatial diffusion and equilibrium constraints can be encoded as loss terms.

### New Module Spec
`sparc/physics/economic_spatial.py` — Implements:
- Spatial autocorrelation residual loss: $\mathcal{L}_{SAR} = \|\epsilon - \rho W \epsilon\|^2$ (penalizes unexplained spatial clustering)
- Hedonic gradient consistency: property value gradient should align with amenity gradient
- Economic diffusion regularizer: income growth should not spread faster than labor market access allows
- Monocentricity soft constraint: distance-decay in value from economic centers is a prior, not a hard constraint

### Existing Template
None dedicated; `templates/uhi/` spatial structure is reusable.

---

## 8. Defense & Intelligence

### What SPARC Solves
Defense and intelligence applications require spatial reasoning about terrain, sensor coverage, pattern-of-life signals, and logistical network optimization. SPARC's physics-informed spatial prediction and causal inference apply to non-classified terrain analysis, sensor placement optimization, and environmental factor analysis for mission planning. **This domain is limited to unclassified, legally authorized applications.**

### Spatial Outcome Variables
- Line-of-sight / terrain visibility index
- Sensor coverage probability map
- Mobility corridor quality index
- Environmental exposure index (heat, noise, air quality for force protection)
- Logistical network efficiency score

### Example Use Cases
- Compute optimal sensor placement locations to maximize coverage probability across a terrain (unclassified)
- Map terrain trafficability for vehicles under different seasonal conditions
- Estimate the causal effect of terrain features (vegetation density, slope, soil type) on mobility index
- Predict heat exposure risk for personnel operating in specific terrain and climate conditions
- Model communication signal attenuation across heterogeneous terrain for network planning

### Governing Physics

**Ray tracing / visibility (geometric optics)**:
$$\text{LOS}(\mathbf{x}, \mathbf{x}_s) = \mathbf{1}\left[\text{elevation}(t) < z_{\text{ray}}(t) \;\forall\; t \in [0,1]\right]$$

Where line-of-sight from source $\mathbf{x}_s$ to point $\mathbf{x}$ exists if the terrain elevation never exceeds the ray height.

**Signal attenuation (Friis / ITU-R propagation)**:
$$P_r(\mathbf{x}) = P_t G_t G_r \left(\frac{\lambda}{4\pi d(\mathbf{x})}\right)^2 \cdot L_{veg}(\mathbf{x}) \cdot L_{terrain}(\mathbf{x})$$

Where $L_{veg}(\mathbf{x})$ is vegetation attenuation and $L_{terrain}(\mathbf{x})$ is terrain diffraction loss — both spatially varying.

**Terrain trafficability (diffusion model)**:
$$v_{max}(\mathbf{x}) = f\left(\text{slope}(\mathbf{x}), \text{soil}(\mathbf{x}), \text{vegetation}(\mathbf{x}), \text{moisture}(\mathbf{x})\right)$$

Trafficability is a learned spatial field constrained by physical limits (slope > 45° is impassable; saturated clay has near-zero trafficability).

### Process-Rate Analog
$L_{terrain}(\mathbf{x})$ — **local terrain impedance** — how much the physical environment attenuates movement, signal, or visibility at each location. The defense analog of thermal diffusivity. Learned from elevation, slope, vegetation density, and soil type.

### Causal Structure
- **Treatment:** Infrastructure change (road construction, clearing), environmental condition change (season, weather), sensor placement
- **Outcome:** Coverage probability, trafficability, exposure index
- **Confounders:** Baseline terrain, vegetation phenology, seasonal conditions

### Physics Module Status
🔲 **New module required.** Visibility and signal propagation are geometric/electromagnetic rather than thermal processes.

### New Module Spec
`sparc/physics/terrain_physics.py` — Implements:
- Terrain smoothness regularizer: slope field is the gradient of the elevation DEM; predictions must be consistent with the measured DEM
- Physical trafficability bounds: slope > 45° → $v_{max} = 0$; soil saturation → $v_{max}$ capped
- Signal attenuation loss: predicted signal strength must be monotone decreasing with distance (modulo terrain effects)
- Note: Full LOS ray-tracing is a geometric computation, not a PDE — SPARC uses it as a feature input, not a loss term

### Existing Template
`templates/seismic/` (terrain physics reuse); `templates/noise/` (signal attenuation partial reuse).

---

## 9. Energy Systems

### What SPARC Solves
Energy demand, renewable resource availability, and grid load are spatially distributed, physics-governed, and causally linked to land use, climate, and socioeconomic factors. SPARC predicts where energy demand will be highest, causally attributes demand patterns to their drivers, and simulates how interventions (efficiency programs, distributed generation, grid investment) would change the spatial distribution of demand and generation.

### Spatial Outcome Variables
- Building energy demand intensity (kWh/m²/year)
- Distributed generation potential (solar irradiance, wind speed)
- Grid load per distribution circuit
- Energy poverty index (energy cost burden by household)
- Carbon intensity of local energy supply

### Example Use Cases
- Map building energy demand across a city and causally attribute spatial variation to building age, land use, occupancy, and climate exposure
- Estimate the causal effect of a weatherization assistance program on energy consumption for low-income households
- Identify optimal rooftop solar placement to maximize generation while minimizing grid impact
- Predict grid load under extreme heat scenarios (connects to UHI domain)
- Map energy poverty hotspots and identify which housing stock interventions have the highest cost-effectiveness

### Governing Physics

**Building energy balance (EnergyPlus surrogate)**:
$$C_{th}(\mathbf{x})\frac{dT_{in}}{dt} = Q_{solar}(\mathbf{x}) + Q_{internal}(\mathbf{x}) - UA(\mathbf{x})(T_{in} - T_{out}) + Q_{HVAC}(\mathbf{x})$$

Where $C_{th}$ is building thermal mass, $UA(\mathbf{x})$ is spatially-varying thermal conductance (envelope quality), $Q_{solar}$ is solar gain, $Q_{internal}$ is internal gain, $Q_{HVAC}$ is HVAC output.

This is structurally identical to the **heat equation** — SPARC's existing thermal PDE applies directly, with $UA(\mathbf{x})$ as the process-rate field instead of $\alpha(\mathbf{x})$.

**Wind resource (Navier-Stokes surrogate)**:
$$\mathbf{v}(\mathbf{x}) = \mathbf{v}_{\infty} \cdot g(\text{elevation}(\mathbf{x}), \text{roughness}(\mathbf{x}), \text{obstacles}(\mathbf{x}))$$

Where $g$ is a terrain-following correction factor learned from reanalysis data and lidar.

**Grid power flow (DC approximation)**:
$$P_i = \sum_j B_{ij}(\theta_i - \theta_j)$$

Where $P_i$ is net power injection at bus $i$, $B_{ij}$ is line susceptance, and $\theta$ is voltage angle. This is a spatial Laplacian system — the electrical network analog of heat diffusion.

### Process-Rate Analog
$UA(\mathbf{x})$ — **building envelope thermal conductance** — how rapidly buildings lose heat to the outdoor environment. High in poorly insulated older stock; low in well-insulated new construction. The energy analog of thermal diffusivity. Learned from building age, type, and vintage data.

### Causal Structure
- **Treatment:** Efficiency program, distributed generation, tariff structure, building code change
- **Mediators:** Building thermal performance, occupant behavior, grid infrastructure
- **Outcome:** Energy consumption, demand peak, carbon emissions, energy cost burden
- **Confounders:** Building age, household income, climate exposure, baseline infrastructure

### Physics Module Status
✅ **Largely reusable.** Building energy balance maps directly onto the heat equation already in SPARC. Grid power flow requires a new module.

### New Module Spec
`sparc/physics/grid_pde.py` — Implements:
- DC power flow Laplacian loss: $\mathcal{L}_{grid} = \|BP\boldsymbol{\theta} - \mathbf{P}_{net}\|^2$ where $B$ is the bus admittance matrix
- Power balance constraint: sum of net injections = 0 (Kirchhoff's current law)
- Line capacity soft constraint: $|P_{ij}| \leq P_{ij}^{max}$

### Existing Templates
`templates/uhi/` (building energy balance reuse), `templates/air_quality/` (pollution dispatch overlap)

---

## 10. Oceans & Marine

### What SPARC Solves
Ocean and coastal phenomena — sea surface temperature, harmful algal blooms, coral bleaching, wave energy, fish habitat — are governed by fluid dynamics and thermodynamics applied to a spatial domain with complex boundaries. SPARC's physics-informed spatial prediction is well-suited for marine spatial planning, coastal resilience assessment, and marine ecosystem analysis.

### Spatial Outcome Variables
- Sea surface temperature (SST)
- Chlorophyll-a concentration (proxy for algal bloom / productivity)
- Dissolved oxygen / hypoxia index
- Significant wave height / coastal flood inundation
- Coral bleaching probability
- Fish species occurrence probability

### Example Use Cases
- Map harmful algal bloom (HAB) probability in a coastal estuary and causally attribute it to nutrient loading from upstream agriculture
- Estimate the causal effect of coral reef degradation on fish biomass in adjacent fishery zones
- Predict coastal inundation extent under storm surge scenarios
- Identify optimal marine protected area (MPA) placement to maximize biodiversity benefit
- Map hypoxia zones in the Gulf of Mexico and causally attribute them to Mississippi River nitrogen flux

### Governing Physics

**Ocean heat transport (advection-diffusion)**:
$$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \kappa(\mathbf{x})\nabla^2 T + Q(\mathbf{x}) / (\rho c_p h)$$

Where $T$ is sea surface temperature, $\mathbf{u}$ is the surface current velocity field, $\kappa(\mathbf{x})$ is turbulent thermal diffusivity, $Q(\mathbf{x})$ is net surface heat flux, $\rho c_p h$ is the mixed layer heat capacity. Structurally identical to the UHI heat equation with an added advection term.

**Nutrient-phytoplankton dynamics (NPZD reaction-diffusion)**:
$$\frac{\partial N}{\partial t} = D_N\nabla^2 N - \mu(\mathbf{x})P + r Z + \epsilon D_Z + \text{sources}$$
$$\frac{\partial P}{\partial t} = D_P\nabla^2 P + \mu(\mathbf{x})P - g Z - m_P P$$

Where $N$ is nutrient concentration, $P$ is phytoplankton (algal) biomass, $Z$ is zooplankton, $\mu(\mathbf{x})$ is spatially-varying phytoplankton growth rate (dependent on light, temperature, nutrients).

**Shallow water wave equation (coastal flooding)**:
$$\frac{\partial \eta}{\partial t} + \nabla \cdot \left[(h + \eta)\mathbf{u}\right] = 0$$
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -g\nabla\eta - \frac{c_f}{h+\eta}|\mathbf{u}|\mathbf{u}$$

Where $\eta$ is sea surface elevation, $h$ is bathymetric depth, $c_f$ is bottom friction. Partially covered by `templates/coastal/`.

### Process-Rate Analog
$\kappa(\mathbf{x})$ — **ocean turbulent diffusivity** — how rapidly thermal energy mixes laterally in the surface ocean. High in energetic frontal zones; low in stratified, calm regions. Learned from current variability, mixed layer depth, and wind stress data.

### Causal Structure
- **Treatment:** Nutrient loading (agricultural runoff, wastewater discharge), thermal discharge, fishing pressure, MPA establishment
- **Mediators:** Water temperature, nutrient availability, light penetration, current patterns
- **Outcome:** SST, chlorophyll-a, fish biomass, bleaching probability, hypoxia extent
- **Confounders:** Background climate (ENSO phase), bathymetry, baseline nutrient state

### Physics Module Status
⚠️ **Partially reusable.** Ocean heat transport maps directly onto the advection-diffusion variant of SPARC's heat equation (adds advection term $\mathbf{u} \cdot \nabla T$). NPZD reaction-diffusion and shallow water waves require extensions.

### New Module Spec
`sparc/physics/ocean_pde.py` — Implements:
- Advection-diffusion SST loss: $\mathcal{L}_{SST} = \|\partial_t T + \mathbf{u}\cdot\nabla T - \kappa\nabla^2 T - Q/(\rho c_p h)\|^2$
- NPZD coupling loss: phytoplankton growth balanced by nutrient drawdown and grazing
- Non-negativity on all biological concentrations
- Current consistency: the advection velocity $\mathbf{u}$ must be divergence-free in the interior (mass conservation)

### Existing Templates
`templates/coastal/`, `templates/water_quality/`, `templates/stormwater/`

---

## Summary Table

| Domain | Governing Physics | Process-Rate Analog | Physics Module Status | Existing Template |
|---|---|---|---|---|
| Health & Epidemiology | FKPP reaction-diffusion | Mobility diffusion coefficient $D(\mathbf{x})$ | 🔲 New module | Partial (air_quality) |
| Crime | Short et al. attractiveness-offender system | Attractiveness diffusion $\eta(\mathbf{x})$ | 🔲 New module | None |
| Neuroscience | Wilson-Cowan neural field | Synaptic connectivity kernel $w(\mathbf{x},\mathbf{x}')$ | 🔲 New module | None |
| Climate & Ecology | Heat eq., Richards eq., Lotka-Volterra | Thermal diffusivity / hydraulic conductivity | ✅ Largely reusable | Full suite |
| Infrastructure | Linear elasticity, LWR traffic, seismic attenuation | Material stiffness / impedance $\mathbf{C}(\mathbf{x})$ | ⚠️ Partial (seismic) | seismic, geotechnical |
| Agriculture | Richards eq., nutrient ADR, Farquhar surrogate | Hydraulic conductivity $K(\psi,\mathbf{x})$ | ⚠️ Partial (groundwater) | groundwater, drought |
| Economics | SAR / spatial diffusion of shocks | Agglomeration diffusivity $D(\mathbf{x})$ | ⚠️ Soft physics — new | None |
| Defense | LOS ray tracing, signal attenuation, trafficability | Terrain impedance $L_{terrain}(\mathbf{x})$ | 🔲 New module | seismic, noise (partial) |
| Energy | Building energy balance, DC power flow | Envelope conductance $UA(\mathbf{x})$ | ✅ Heat eq. reusable | uhi (partial) |
| Oceans & Marine | Ocean advection-diffusion, NPZD, shallow water | Turbulent diffusivity $\kappa(\mathbf{x})$ | ⚠️ Partial (coastal) | coastal, water_quality |

---

## Module Development Priority

For teams expanding SPARC into new domains, the recommended development sequence based on physics reuse and impact:

**Tier 1 — High impact, near-reuse (minimal new code):**
- Energy Systems (building energy balance ≡ heat equation)
- Agriculture (Richards equation already in groundwater template)

**Tier 2 — High impact, moderate new code:**
- Health & Epidemiology (steady-state diffusion-reaction close to existing PDE structure)
- Oceans & Marine (adds advection term to heat equation)

**Tier 3 — High impact, significant new code:**
- Crime (new coupled PDE system)
- Economics (soft physics constraints, not PDEs)
- Infrastructure (structural + traffic PDEs)

**Tier 4 — Specialized, new code + domain expertise required:**
- Neuroscience (cortical surface geometry, neural field convolution integral)
- Defense (geometric ray tracing, signal propagation, classified data constraints)
