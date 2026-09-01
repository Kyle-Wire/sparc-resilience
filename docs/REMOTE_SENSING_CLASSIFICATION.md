# Classification as Inverse Physics

**A remote sensing exploration of the SPARC `--stage all` pipeline**

> The run-all-stages pipeline already contains a forward model that maps fractional
> land cover to observable physics. Image classification is that model, run backwards
> — which is why it fits, and why the physics constraints really can act as a denoiser.

**Scope of review:** the `--stage all` execution path only — `sparc/run/orchestrator.py`,
the six stage runners, and the models, physics and evaluation modules they call.
Side experiments, the desktop server and the multi-city training scripts were read
for context but are not part of the recommendations.

---

## 1. The argument in three claims

### Claim 1 — Shape

Do not bolt a softmax head onto Stage 2. Change the *algebra of the target*: from ℝ to
the simplex Δ^K. Predict a fractional-cover vector per location, constrained to be
non-negative and sum to one, and take the class label as its argmax. Every stage —
correlogram, GWEN, spatial CV, causal validation, scenarios, reporting — survives that
change essentially intact, because a simplex-valued field is still a field.

### Claim 2 — Precedent

`ProcessRateNet` already accepts `land_cover: (N, n_classes)` **fractional coverage** and
does linear mixing in physical units to produce a per-cell process rate α. It even ships
`material_validation_report()`, which checks learned per-class α against literature values.

That is a spectral-unmixing operator wearing a thermal costume. SPARC runs it forward
today: cover → physics. Classification is the inverse: observation → cover. You do not
need a new architecture; you need to close a loop that is already half-built.

### Claim 3 — Noise

The physics intuition is right, and it is sharper than "constraints regularise." Physical
laws define a low-dimensional admissible set. Sensor noise, atmospheric path radiance,
speckle and mis-registration are generically *off* that set. Projecting onto it removes
the orthogonal component — that is variance reduction, purchased with bias if the physics
is wrong.

And Stage 0 *already estimates the noise floor*: the Matérn fit samples a full posterior
over (κ, ν, σ², τ²), where τ² is the nugget. That estimate is then thrown away —
`PredictorKernel` carries κ, ν, σ² and the anisotropy ellipse downstream, and no τ².
There is a free, posterior-quantified, per-variable noise model sitting unread in the
artifact store.

---

## 2. What the run-all spine actually is

`PipelineOrchestrator.run("all")` walks `_STAGE_ORDER = ["0","1","2","3","4","5"]` and
dispatches to six real runners.

| Stage | Runner | What it produces | Shape of its output |
|---|---|---|---|
| 0 | `correlogram_analysis:main` | Moran's I curves, Bayesian Matérn(κ,ν,σ²,τ²) per variable, anisotropy ellipses, cross-correlogram range matrix → `KernelField` | Continuous scalar fields |
| 1 | `gwen_variable_selection:main` | Geographically weighted elastic net ranking, per-cell importance maps, human approval gate | Gaussian-loss linear model |
| 2 | `enhanced_spatial_cv:main` | OLS / GWR / GWRF / GGPGAM on buffered spatial folds, differentiable surrogates, `SPARCMetaLearner`, 12-term PDE curriculum, MC-Dropout UQ | One continuous output + *k* sigmoid exceedance heads |
| 3 | `causal_validation:main` | MC³ DAG search, DML/CATE, NUTS edge posteriors, DoWhy refutations, E-values | Continuous treatment → continuous outcome |
| 4 | `stage4_runner:main` | Four-tier counterfactual simulation, physics guardrails, budget-constrained allocation with Gini equity | Δ on a continuous field |
| 5 | `stage5_runner:main` | Audience reports + decision ranking | Ranked interventions |

### Three properties that decide the port

**It is per-location tabular, not convolutional.** Rasters enter through zonal statistics
onto a fishnet (`run_zonal_stats` in the server data routes) and become rows with
coordinates. There is no receptive field, no learned texture filter. The spatial operator
is `SparseSpatialAttention` — KNN attention over point coordinates, O(N · max_neighbors),
with a bandwidth seeded from Stage 0.

**It is cross-sectional.** A single time slice, with optional multi-snapshot diurnal
terms. Time series classification is not a small edit.

**It is regression the whole way down.** `GWRModel` is a `RegressorMixin`; `gwrf.py`
imports `RandomForestRegressor`; `ggpgam.py` builds a `LinearGAM`;
`SPARCMetaLearner.regression_head` emits one scalar. The only categorical machinery in the
whole package is the set of sigmoid exceedance heads — which are, notably, already binary
classifiers wired through the full PDE-trained fusion stack.

> **The useful consequence.** Because everything is per-location and continuous, a
> simplex-valued target costs you a link function and a loss, not a rewrite. A hard-label
> target, by contrast, breaks Stage 0 (no variogram on a nominal variable), breaks Stage 3
> (no dose-response on a category), and makes Stage 4's whole vocabulary of "Δ of +5
> percentage points" meaningless. That asymmetry is the entire design argument.

---

## 3. Which "image classification" you should mean

| Task | Example | Fit | Verdict |
|---|---|---|---|
| **Scene / chip classification** (one label per 64×64 tile) | EuroSAT, BigEarthNet, RESISC45 | Poor. Discriminative signal is texture, shape and context; SPARC has no receptive field and no shift-equivariant feature extractor. | Don't lead here. Reachable later via attention-pooled readout over a tile's KNN subgraph. |
| **Per-pixel semantic labelling** (land cover mapping) | Dynamic World, NLCD, ESA WorldCover | Good. Per-location prediction with spatial structure is exactly SPARC's geometry. | The obvious target — but take it as a by-product, not the objective. |
| **Fractional / sub-pixel cover** (an abundance vector per location) | Impervious surface fraction, tree canopy cover, spectral unmixing | Excellent, and already half-implemented. The Brown project's own predictors — `Pct_Canopy`, `Pct_Impervious` — *are* a fractional cover vector, already carrying a `Canopy + Impervious ≤ 100%` combined constraint in Stage 4's guardrails. | **Make this the target.** Hard labels fall out as argmax. |

### Why the simplex is the right object

A hard label is a lossy projection of a mixture. Every 10 m Sentinel-2 pixel over a city
is a mixture; every 30 m Landsat pixel emphatically is. Committing to "this pixel is
*built*" discards the information that it is 60% roof, 25% asphalt and 15% street tree —
and that discarded information is precisely what Stages 2 through 4 consume. A fractional
target:

- **Keeps Stage 0 well-posed.** Each class fraction is a continuous field with a genuine
  variogram, a real range, a real nugget, and a real anisotropy ellipse. Nominal labels
  admit only join-count statistics, which do not feed a Matérn fit.
- **Keeps Stage 3 meaningful.** "A one-percentage-point increase in impervious fraction"
  is a coherent treatment with a dose-response curve. "Changing class 4 to class 7" is not.
- **Keeps Stage 4 intact.** The existing physics guardrails — bounds, delta caps,
  diminishing-return tapers, Mahalanobis extrapolation guards, combined constraints —
  already operate on exactly this kind of quantity.
- **Is honest about mixed pixels,** which is the dominant error source in every
  operational land cover product and the thing that separates a research map from a
  usable one.

The MAUP-resistance term already in the PDE loss (the sheaf-Laplacian, term 11) exists
because the team already worried about aggregation sensitivity. Fractional cover is the
representation that worry points toward.

---

## 4. The unit problem: pixels are the wrong row

The Brown run was 54,701 points; ForceSMIP was 71,498 grid cells. One Sentinel-2 tile is
roughly 120 million pixels. Fitting four geographically weighted base models per location
at that scale is not a tuning problem, it is a category error — GWR alone solves a
weighted least squares system *per location*.

Remote sensing solved this decades ago and the answer happens to be exactly what SPARC
wants: **object-based image analysis**. Segment the imagery (SLIC, SNIC, or a watershed on
the gradient), and treat each segment as one row.

- 120M pixels → roughly 10⁵–10⁶ segments, landing back in the scale SPARC demonstrably
  runs at.
- Segment centroids become `coord_columns`; per-segment band statistics become predictors.
  The ingestion path already exists — `run_zonal_stats` over an arbitrary polygon layer
  instead of a fishnet.
- Segments respect real boundaries (field edges, roof lines, shorelines) rather than an
  arbitrary grid, which softens the MAUP problem the sheaf term is fighting.
- **Segmentation is itself a denoiser.** Averaging *n* pixels within a homogeneous segment
  divides the white-noise variance by *n* while leaving the spatially correlated signal
  intact. This is the nugget/sill decomposition applied as a preprocessing step — and
  Stage 0 can measure exactly how much it bought you, before and after.

> `sparc/data/satellite_types.py` already defines a `SatelliteBand` enum (NDVI, NDBI,
> NDWI, LST, albedo, emissivity, impervious surface area, tree cover, building height, sky
> view factor) and a `SatelliteFeatureSet` dataclass with coords, resolution, acquisition
> date, CRS and Köppen zone — written for a "V4 zero-shot" path and currently unwired.
> That is the ingestion schema for this work, already designed. The segment table is what
> should fill it.

---

## 5. What each stage becomes

| | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|---|---|---|---|---|---|---|
| **Today** | Matérn per variable: κ, ν, σ², τ² + ellipse → KernelField | Elastic net rank: which predictors matter, where | 4 GW regressors → surrogates → meta-learner, PDE curriculum | DAG + DML: edge posteriors, refutations, E-values | Counterfactual ΔT: 4 tiers + budget allocation | Audience report, ranked interventions |
| **Classification** | Per-class variograms; τ² becomes the label-noise estimate; ellipse becomes the smoothing metric | Band × class × place: which band separates which class, where | 4 GW classifiers, simplex head, 3 new physics terms | Deconfound optics: illumination, aerosol, phenology as confounders | Two modes: cover transitions & label-budget allocation | Area with CIs; conformal class sets, not bare accuracy |

### Stage 0 — the noise floor becomes a first-class output

Fit the Matérn per *class fraction* rather than per predictor. Three things fall out, and
the third is the one that matters.

The **range** (1/κ) sets the spatial CV block size, which quietly fixes a real problem in
the remote sensing literature: reported accuracies are routinely inflated because training
and test pixels come from the same field, the same rooftop, the same forest stand. SPARC's
Stage 0 → Stage 2 block wiring makes leak-free validation the default rather than a virtue
you have to remember. That is a publishable claim on its own.

The **anisotropy ellipse** (κ_x, κ_y, θ) becomes the metric for spatial smoothing.
Everyone smooths land cover maps; almost everyone does it isotropically, with a 3×3
majority filter or an isotropic CRF. Agricultural rows, street grids, drainage networks
and ridge lines are not isotropic. Smoothing along a data-estimated ellipse instead of a
circle is a small change with a visible effect on exactly the linear features that
isotropic filters destroy.

The **nugget** τ² is the finding. It is fit with a full posterior and an 89% HDI in
`correlogram_matern_fit.py`, fit again in the anisotropic model in `anisotropy.py`, and
then dropped: `PredictorKernel` carries `kappa`, `nu`, `sigma2`, `kappa_x`, `kappa_y`,
`theta_rad`, `eccentricity`, `bandwidth_to_outcome` and `kappa_posterior_samples` — and no
nugget. Grep confirms `tau2` appears nowhere downstream of the two fitters. Meanwhile
`pipeline_configurator.py` hard-codes `'nugget': 0.1`. You are estimating the thing
carefully and then substituting a constant.

> **Carry τ² through, and four things become derivable rather than tuned.**
> 1. The label-smoothing coefficient, from τ²/(σ²+τ²) — the noise-to-total ratio of the
>    label field.
> 2. Per-band SNR weighting into GWEN and into the `SpatialGatingHead`.
> 3. The irreducible-error floor to report against, so you stop chasing accuracy past the
>    noise ceiling.
> 4. A robust-loss temperature (symmetric or generalised cross-entropy) set from data
>    instead of by hand.
>
> This is the single highest-value, lowest-risk change in the document.

### Stage 1 — which band matters where

Swap the Gaussian-loss elastic net for a geographically weighted multinomial elastic net.
The output is a band × class × location importance surface: SWIR1 separates bare soil from
built in an arid basin but barely helps in a humid temperate city; red-edge carries crop
discrimination in one region and nothing in another.

Practitioners know this and handle it by stacking every band and index they have and
letting a gradient booster sort it out. A per-region, per-class importance map that says
*why* is a genuinely differentiated output, and the existing human approval gate
(`gwen_approved.txt`) is the right place to review it. Keep that gate — it is more valuable
here than in regression, because band selection is where domain knowledge actually lives.

### Stage 2 — four classifiers and a simplex head

The base model swaps are close to mechanical:

- **OLS** → multinomial logistic regression. Same design matrix, softmax link.
- **GWR** → geographically weighted logistic / multinomial regression. This is standard in
  the literature (generalised GWR with a logit link); the change is IRLS inside
  `_local_regression` instead of weighted least squares. The kernel, bandwidth and
  anisotropy machinery is untouched.
- **GWRF** → `RandomForestClassifier`. A one-line import swap plus `predict_proba`. The
  existing per-location epistemic-uncertainty estimator carries over directly to class
  probabilities.
- **GGPGAM** → pygam's `LogisticGAM` for binary, one-vs-rest for multi-class. Same
  tensor-product spatial terms.

For the meta-learner, the smallest correct change is to replace `regression_head`'s single
output with *K* outputs and a softmax — but the more interesting change is to keep it as a
Dirichlet head, emitting concentration parameters rather than a point on the simplex. That
gives you aleatoric and epistemic uncertainty separately, for free, and composes naturally
with the MC-Dropout passes already running at inference. Mixed pixels then read as high
Dirichlet concentration on a genuinely mixed mean; unfamiliar terrain reads as low total
concentration. Those are different failures and they should not look the same on a map.

Two pieces need no change at all. The exceedance heads are already sigmoid classifiers on
the fused representation. And `SpatialGatingHead` already computes a softmax over
surrogates using spatial context and cross-surrogate disagreement — the exact structure you
want for per-location model selection among four classifiers with different regional
competence.

### Stage 3 — the part nobody else does

This is where a physics-and-causality pipeline earns its keep, because remote sensing
classification has a confounding problem it mostly ignores.

A shadowed conifer stand and a still water body are both dark in the visible. A classifier
trained on a scene where shadows are common learns "dark → water," and that association is
*confounded by illumination geometry*, which is fully computable from a DEM and the solar
position at acquisition. Put illumination in the DAG as a confounder of both observed
reflectance and assigned label, and the question becomes: what would this surface look like
under neutral illumination? That is a backdoor adjustment, and SPARC has the machinery for
it.

The same structure handles the standard failure list. Aerosol optical depth confounds haze
with bright bare soil. Phenological stage confounds senescent grass with bare ground.
Sensor view angle confounds BRDF hot-spot brightening with cover density. Soil moisture
confounds dark soil with organic content.

Two concrete outputs:

- **Confusion as a causal object.** Instead of a confusion matrix that tells you *that*
  water and shadow are confused, a per-cell CATE that tells you *how much* of that
  confusion is attributable to illumination — and therefore how much a topographic
  correction would recover. The existing CATE-vs-GWR divergence audit is the template for
  reporting it.
- **Refutation applied to maps.** Placebo treatment, random common cause, subset analysis
  and E-values, run on a land cover product. An E-value on "this class boundary is real,
  not an illumination artifact" is a diagnostic no operational land cover product currently
  ships.

> **Scope this hard.** MC³ over a DAG with twenty class-indicator nodes will not finish,
> and the posterior would be uninterpretable if it did. Keep the DAG in *process* variables
> — illumination, aerosol, moisture, phenology, view angle, and four or five continuous
> cover fractions — not in class dummies. Ten to fifteen nodes, expert-specified, is the
> operating range Stage 3 was built for.

### Stage 4 — two counterfactual modes, one of them unexpected

**Mode A, cover transitions.** The natural reading. Convert a parcel from impervious to
canopy and propagate: what does the sensor see, what does the model classify, and — through
the existing forward physics — what happens to surface temperature? This closes the loop
with the UHI work rather than sitting beside it. The four tiers survive: Tier 0 uses the
learned α field, Tier 1 samples NUTS edge posteriors, Tier 2 solves the PDE forward under
new forcing, Tier 3 blends.

**Mode B, the interesting one: Stage 4 as an active-learning planner.** Stage 4 already
solves a budget-constrained allocation problem — per-cell predicted benefit against a
per-cell cost surface, via greedy, greedy-2opt or MILP, swept into a Pareto frontier, and
scored for equity with a Gini coefficient. Re-read those three inputs:

- **Benefit** → expected reduction in classification uncertainty from labelling this
  location (entropy, or Dirichlet concentration, or conformal set size).
- **Cost** → what it actually costs to get that label: field visit, analyst time,
  commercial high-resolution tasking.
- **Equity** → the constraint that keeps labelling effort from concentrating in the places
  that are already well-mapped.

That is a cost-aware, equity-constrained active learning system, and it is one adapter away
from code that already exists. It is worth saying plainly why it matters: land cover
reference data is systematically denser in wealthy, accessible, temperate, well-surveyed
places, and every map trained on it inherits that gradient. A labelling planner with a Gini
constraint is a defensible answer to that, and no classification toolchain ships one. If
you want a single differentiating feature for a remote sensing product, this is it.

A third mode is worth prototyping cheaply: **acquisition planning.** The same optimiser
over sensing configurations rather than locations — what does one more SWIR band, or a
second acquisition at a different sun angle, or a June revisit, buy in expected accuracy
per unit cost?

### Stage 5 — report the map the way a statistician would

**Area estimation with confidence intervals.** Pixel counting is a biased area estimator,
and the bias is a function of the confusion matrix. The standard correction (Olofsson and
colleagues' good-practice estimators) produces an unbiased area estimate with a proper
standard error. Stage 5's report generator is the right home for it, and "urban area grew
by 4.2 km² ± 0.9" is what belongs in a policy document. A bare overall-accuracy number is
not.

**Conformal class sets.** `SpatialConformalPredictor` already implements weighted
split-conformal with a kernel bandwidth derived from Stage 0's κ. It currently produces
regression intervals. Swapping the nonconformity score to an adaptive-prediction-set score
gives you class *sets* with a coverage guarantee — and the spatial weighting is retained, so
calibration is local. "This pixel is {crop, grass} at 95% coverage" is a far more useful
output than a confident wrong argmax, and spatially-weighted conformal classification for
land cover is, as far as I can tell, not something that currently exists in the wild.

---

## 6. Why the physics constraints really do denoise

A physical law is a constraint that carves out a low-dimensional admissible set inside the
space of possible observations. Energy must balance. Reflectance must be a non-negative
mixture of endmember spectra. Diffusion must satisfy its PDE. The set of physically
realisable scenes is a thin manifold in the ambient measurement space.

Corruption is generically off that manifold. Shot noise does not conserve energy. Path
radiance does not obey the mixing model. Speckle does not satisfy a diffusion equation. So
an estimator constrained to the manifold cannot represent the orthogonal component of the
corruption — it is projected out, and that projection is variance reduction. This is the
same mechanism as denoising by projection onto a learned subspace, except the subspace comes
from physics rather than from data, which means it does not need labels and does not overfit.

The projection removes whatever component of the corruption is orthogonal to the physics.
It cannot touch the component that lies *along* the manifold, and it introduces bias in
exact proportion to how wrong the manifold is.

### What this buys, corruption by corruption

| Corruption | Signature | Constraint that removes it | Status in SPARC |
|---|---|---|---|
| **Sensor / shot noise, striping** | Spatially white; zero correlation length | It *is* the Matérn nugget τ². Any spatially smooth prior removes it. | **Measured, discarded.** Carry τ² into `PredictorKernel`. |
| **Atmospheric path radiance** | Spatially smooth, low frequency; spectrally structured (Rayleigh ∝ λ⁻⁴) | Model as an additive nuisance field with a spectral-shape prior, not as class evidence — structurally identical to the learned α field. | New. Reuses the α-field pattern exactly. |
| **Topographic illumination** | Deterministic given DEM and solar geometry (cos *i*) | Not noise at all — a computable nuisance. C-correction or Minnaert as a hard term, plus illumination as a Stage 3 confounder. | Elevation already ingested. Needs solar geometry per acquisition. |
| **BRDF / view angle** | Smooth function of view and sun angles | Reflectance constrained to a linear combination of isotropic, volumetric and geometric kernels. | New. A three-parameter linear constraint; cheap. |
| **SAR speckle** | Multiplicative, Gamma-distributed | Work in log space — multiplicative becomes additive, and the Matérn nugget fit becomes valid again. | Preprocessing only. Free. |
| **Mixed pixels** | Not noise; a representation error | Simplex constraint plus a linear spectral mixing residual. | The core proposal. Sheaf term already fights the aggregation half. |
| **Label noise** | *Spatially clustered* — a mislabelled polygon is a whole blob, not a nugget | Physical implausibility. A class mixture implies albedo, emissivity, thermal inertia and a Bowen ratio; those must reproduce the observed LST through the energy balance. | New, and the best idea here. |
| **Co-registration error** | A small spatial translation | Shift-equivariance of the representation. | `ActionEmbedding` already encodes (Δx magnitude, sign, Δt). The JEPA pretext can be made to enforce it. |

Note that the two most damaging (mixed pixels, label noise) are not white noise and are
untouched by conventional smoothing.

### Three PDE terms to add

The loss already runs to twelve terms with a staged curriculum, per-residual normalisation,
and linear five-epoch ramps. Three more slot in without disturbing that structure.

**T13 — spectral mixing residual.** ‖ρ_obs − Σ_k f_k E_k‖², where f lies on the simplex and
E_k are endmember spectra, either taken from a library or learned with a non-negativity
constraint. This is the direct analogue of `ProcessRateNet`'s existing
weighted-sum-of-material-values operator, applied to reflectance rather than process rate.
Activate it early — it is the term that does the most work.

**T14 — energy-balance-consistent labels.** A cover mixture implies material properties.
Those properties, pushed through `energy_balance_residual(T, albedo, solar, T_sky)`, must
reproduce the observed land surface temperature. A label that cannot balance its energy
budget is probably wrong. Activate this late, once the mixing term has stabilised.

**T15 — anisotropic simplex total variation.** Penalise the gradient of f under the Stage 0
ellipse metric rather than the Euclidean one: smooth along θ, free to jump across it. This
replaces the isotropic CRF smoothing step that most land cover pipelines bolt on at the end,
and does it with a data-estimated orientation instead of a hand-tuned one.

> **T14 doubles as a label auditor.** Run it in diagnostic mode and it becomes a
> physics-based label quality tool. For every reference polygon, compute the energy-balance
> residual its assigned class implies against the observed thermal band. Rank by residual.
> The top of that list is your mislabelled training data — found without a single additional
> human annotation, and found by a mechanism completely independent of the spectral features
> the classifier uses, which is exactly what makes it credible. Given that label noise is the
> dominant error source in operational land cover mapping and that it is spatially clustered
> rather than white, this may be worth more than the classifier itself.

> **The honest cost.** Projection reduces variance and adds bias, and the bias is
> proportional to how wrong the constraint is. A misspecified endmember library injects error
> precisely where it is reducing variance, which makes it hard to detect — the map looks
> *cleaner* and is *more wrong*. Two guardrails, both consistent with the operating rules
> already written into the PI-JEPA spec: (1) every physics term stays behind a config flag
> defaulting to off, and every run reports the physics-free ablation alongside; (2) hold out a
> labelled region and report calibration, not just accuracy — over-constrained models are
> confidently wrong, and calibration curves catch that where accuracy does not.

---

## 7. Where the physics will not save you

**Geometric classes are invisible to radiometry.** A parking lot and a wide road are the
same asphalt. A warehouse roof and a big-box retail roof are the same membrane.
Distinguishing them is a matter of shape, adjacency and context — and no energy balance or
mixing model has an opinion about shape. This is precisely where a convolutional or
transformer receptive field wins and SPARC's KNN attention over segment centroids does not.
If the class ontology is functional rather than material, physics constraints contribute
nothing and the pipeline will underperform a plain CNN.

**Some classes are physically identical by construction.** A lawn and a golf course fairway
are the same grass with the same albedo, the same evapotranspiration and the same thermal
inertia. Physics correctly says they are the same thing, because they are. That is a
mismatch between the label ontology and the measurement, not a noise problem, and no
constraint will resolve it. Worse, adding physics will actively hurt: it will push the two
toward each other. Screen the class scheme for this before committing.

**Computational limits are real.** NUTS in Stage 0 over twenty class fractions plus the
anisotropy fit is a large multiple of the current cost, and the PI-JEPA spec already
documents that θ and the anisotropic κ parameters are chronically ESS-limited at 400
fast-mode draws. Fit the Matérn on a handful of derived continuous indices (NDVI, NDBI,
NDWI, LST, albedo) rather than on every class indicator, and reuse those kernels across
classes. That keeps Stage 0 at roughly today's cost and sidesteps a convergence problem you
already know you have.

---

## 8. Don't compete with the foundation models — sit on top of them

On raw accuracy over benchmark chips, a pretrained geospatial encoder will beat this
pipeline. That is not a close call, and it is not the fight to pick. The honest competitive
read:

- **Leak-free accuracy.** A large share of published remote sensing accuracies are inflated
  by spatial autocorrelation between train and test splits. Stage 0 → Stage 2 block wiring
  makes correct spatial CV structural rather than optional. Numbers from this pipeline will
  look *worse* and be *right*, and that is a defensible position if you say so loudly.
- **Calibrated set-valued output** with local coverage guarantees, via the spatially-weighted
  conformal predictor that already exists.
- **Counterfactuals.** Nobody else can answer "what would this map look like if we converted
  that parcel, and what happens to surface temperature."
- **Label economics.** The equity-constrained labelling planner from §5.
- **Auditability.** Per-region band importance, causal deconfounding of illumination,
  refutation tests, E-values. This is what gets a map through a procurement review.

And the strongest single move available: **use a pretrained encoder as a feature source, not
a competitor.** Take per-segment embeddings from any geospatial foundation model, hand them
to Stage 1 as predictors alongside the physical bands, and let the rest of the pipeline do
what only it does. SPARC becomes the calibrated, causal, physics-constrained head on top of
somebody else's representation learning — which is a much better position than trying to
out-pretrain them, and it costs one adapter.

---

## 9. Where the code actually changes

| File | Change | Size |
|---|---|---|
| `sparc/config/project_schema.json` | New `task:` block — type, classes, representation, label-noise policy | S |
| `sparc/models/kernel_field.py` | Add `tau2` (and its HDI) to `PredictorKernel`; plumb from both fitters | S |
| `sparc/run/pipeline_configurator.py` | Replace hard-coded `'nugget': 0.1` with the estimated posterior mean | S |
| `sparc/models/gwrf.py` | `RandomForestClassifier` branch on task type; `predict_proba` | S |
| `sparc/models/ggpgam.py` | `LogisticGAM` branch, one-vs-rest for K > 2 | S |
| `sparc/evaluation/conformal.py` | Adaptive-prediction-set nonconformity score; keep the κ-weighted calibration | S/M |
| `sparc/models/gwr.py` | IRLS with a logit/softmax link inside `_local_regression`; kernel code unchanged | M |
| `sparc/models/neural_meta.py` | `regression_head` → Dirichlet/simplex head; exceedance heads unchanged | M |
| `sparc/models/surrogates.py` | Softmax outputs on the three differentiable surrogates | M |
| `sparc/run/correlogram_analysis.py` | Per-class-fraction variograms; emit τ² as a first-class artifact | M |
| `sparc/run/gwen_variable_selection.py` | Multinomial elastic net; band × class × place importance surface | M |
| `sparc/physics/pde_loss.py` | Terms 13–15 (mixing, energy-balance labels, anisotropic simplex TV) with curriculum offsets | M/L |
| `sparc/data/` + `run_zonal_stats` | OBIA segmentation → segment table; wire the existing `SatelliteFeatureSet` schema | L |
| `sparc/run/stage5_runner.py` | Olofsson-style area estimation with standard errors; conformal set reporting | M |
| `sparc/decision/optimizer.py` | Adapter: benefit = expected uncertainty reduction, cost = labelling cost | M |

Nine of fourteen are small or medium; the genuinely large item is data ingestion, not
modelling.

### Config sketch

```yaml
task:
  type: fractional          # regression | fractional | hard
  classes: [water, tree, grass, crop, built, bare]
  representation: simplex   # simplex | dirichlet | hard
  argmax_label: true        # emit a hard map as a by-product

  label_noise:
    estimate_from: stage0_nugget   # derive smoothing from tau2/(sigma2+tau2)
    robust_loss: symmetric_ce
    audit: energy_balance          # rank reference polygons by physics residual

physics:
  spectral_mixing:
    enabled: false          # flag-gated, defaults to current behaviour
    endmembers: physics/endmembers.csv
  illumination_correction: c_correction
  brdf_kernels: ross_li
```

Every new behaviour behind a flag defaulting to current behaviour — the operating rule
already written into the PI-JEPA spec.

---

## 10. Four phases, each with something that can fail

Ordered so the cheapest falsifiable test comes first and no phase depends on new data until
phase C.

### Phase A — Prove the shape on data you already have

Treat `Pct_Canopy`, `Pct_Impervious` and a residual "other" as a three-class simplex target
on the Brown dataset. Run `--stage all`. No new data, no new physics, no segmentation — only
the target's algebra changes. This is a load-bearing test of whether Stages 0, 3 and 4
tolerate a simplex target at all.

*Gate:* all six stages complete; Stage 0 reports τ² per class fraction; Stage 4 scenarios
remain interpretable under the sum-to-one constraint.

### Phase B — Carry the nugget through and use it

Plumb τ² into `PredictorKernel`, replace the hard-coded 0.1, and derive label smoothing and
per-band SNR weights from it. Test by injecting synthetic label corruption at known rates
into the phase-A target and measuring degradation with and without the noise-aware loss.

*Gate:* beats the flag-off baseline by at least 2× the measured run-to-run σ at a 10%
corruption rate — the same bar the PI-JEPA spec sets. Otherwise revert rather than tune.

### Phase C — Real imagery, real physics

OBIA ingestion of one Sentinel-2 tile over a city you already have a Stage 0 fit for. Add
PDE term 13 (spectral mixing) alone, flag-gated. Then term 14 (energy balance) alone. One
change per run, ablated against the locked baseline.

*Gate:* the pipeline runs at segment scale within the existing memory watchdog budget; each
physics term is kept only on its own evidence.

### Phase D — The outputs nobody else ships

Conformal class sets in Stage 5. Olofsson area estimation with standard errors. The Stage 4
labelling-budget adapter with the Gini equity constraint. Term 14 in diagnostic mode as a
label auditor over a public reference dataset.

*Gate:* empirical coverage of the conformal sets lands within tolerance of nominal on a
held-out region; the label auditor's top-ranked polygons are confirmed mislabelled at a rate
well above base.

---

## 11. The one-sentence version

SPARC should not learn to classify images; it should learn to *invert* the forward physical
model it already runs — recovering the fractional cover vector that best explains the
observed radiance, temperature and spatial structure, with the class label as a by-product
of that inversion.

That framing keeps the pipeline whole, which was the constraint. It makes the physics
load-bearing rather than decorative, because inversion problems are exactly where
constraints buy you well-posedness. It explains the noise intuition rigorously: the
constraint set is a manifold, corruption is off-manifold, projection removes it, and the
price is bias when the manifold is wrong. And it points at the one change that costs almost
nothing and should happen regardless of whether any of the rest gets built — Stage 0 already
knows how noisy your data is, in full posterior, per variable, and nothing downstream is
listening.

---

**Verified in code:** the τ² plumbing gap, `ProcessRateNet`'s fractional-cover input and
linear mixing, the regression-only base models, the existing sigmoid exceedance heads, the
softmax `SpatialGatingHead`, and the unwired `SatelliteFeatureSet` schema. Everything else
is proposal.
