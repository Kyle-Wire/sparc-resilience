# PI-JEPA — Blue-Sky Exploration Roadmap

**Scope:** What to explore *after* the core PI-JEPA work package (B1–B4 baseline, I1–I3 anisotropic
masking) is validated and the evening-correlation gap is addressed. This is the "push the frontier"
document — directions that advance JEPA into geospatial territory no existing spatial ML tool occupies.

**Reading guide — three honesty tiers per item:**
- **PLUG-IN** — the machinery already exists in the codebase; this is wiring, days-to-weeks, low risk.
- **EXTEND** — partial machinery exists; this is real new code on a proven base, weeks, medium risk.
- **FRONTIER** — genuinely new research, months, uncertain payoff, publishable if it lands.

The single most important rule carries over from the core spec: **one change per run, gated behind a
default-off flag, kept only if it beats the locked baseline by ≥ 2× the measured run-to-run σ.** Blue-sky
does not mean abandoning the discipline that made the baseline trustworthy. It means the *ideas* are
ambitious, not that the *validation* is loose.

---

## Prerequisite gate

Do not start anything in this document until ALL of the following hold:
1. PI-JEPA core (I1/I2, and I3 if the B2 audit justified it) is validated and logged.
2. The frozen-trunk decision still holds (see the frozen-vs-unfrozen note in the core spec). Most
   blue-sky items below assume a frozen trunk; the few that need adaptation say so explicitly.
3. A fresh checkout with all flags off still reproduces the promising baseline.
4. You have rotating-holdout validation working (≥ 2–3 holdout cities, mean ± std per window). Several
   items below can overfit to a single holdout city; you cannot trust their gains on one city alone.

---

## Category A — Items that PLUG IN to existing JEPA machinery (do these first)

These exploit infrastructure the codebase already has. They are the highest expected-value blue-sky
moves because the hard part is built.

### A1 — PLUG-IN — Matérn-kernel-weighted JEPA target loss
**What exists:** `spatial_patch_mask`, the JEPA loss, and the Stage 0 Matérn payload (κ, ν per variable).
The VSBA module already evaluates Matérn(ν, ρ) kernels (`_matern_kij`) — reuse that exact function.
**Idea:** Today the JEPA target is binary (masked vs. visible). Replace it with a *soft* target where
the prediction loss on each masked point is weighted by the Matérn covariance between that point and its
context. Strongly-correlated-with-context points get higher weight. The pretext becomes a soft kriging
problem in latent space — the trunk learns to predict what the covariance structure says should be
predictable, and is not penalized for failing to predict what the covariance says is independent.
**Why it advances things:** This is the natural successor to I1. I1 makes the *mask geometry* physics-aware;
A1 makes the *loss weighting* physics-aware. Together they make both the question and the grading of the
self-supervised task derive from the estimated spatial process.
**Research grounding:** Matérn (1960); Gneiting et al. (2010, JASA, Matérn cross-covariance). Kriging as
the BLUP under a Gaussian process (Stein 1999, *Interpolation of Spatial Data*).
**Cost / risk:** Medium. Pairwise kernel evals are O(k) on KNN subsamples (reuse `_knn_subsample`), so
tractable. Risk: the benefit over I1 alone is unproven — run it as a clean ablation against the I1 result,
not against the original baseline.

### A2 — PLUG-IN — Energy-balance / shortwave-absorption pretext head
**What exists:** the disabled `energy_balance_weight` path and `(1 - albedo)` proxy in the sibling script
version; the auxiliary-head pattern is trivial.
**Idea:** Add an auxiliary pretext head during Phase 1 that predicts a shortwave-absorption proxy
`(1 - albedo)` from the trunk embedding. This gives the trunk a *thermodynamic* prior orthogonal to PI-JEPA's
*spatial* prior — the embedding is nudged toward representing the dominant UHI energy driver.
**Why it advances things:** PI-JEPA teaches the trunk *where* structure is; A2 teaches it *what physically
drives heat*. The two priors are complementary, and because A2 is supervised by a feature the trunk already
sees, it cannot leak labels.
**Research grounding:** Oke (1982, *QJRMS*, energetic basis of the UHI) — net shortwave ∝ SW_in·(1 − albedo)
is the first term of the surface energy balance that the whole pipeline rests on.
**Cost / risk:** Low. **Critical pre-check:** the legacy code assumed feature column 0 = albedo. Confirm the
actual column index against the live feature order or the head trains on the wrong channel. Gate behind
`jepa.energy_balance_weight: 0.0`. Validate only after PI-JEPA so gains are attributable.

### A3 — PLUG-IN — Trunk → causal Stage 3 spatial deconfounding (SpatialResidualizer)
**What exists:** the entire design (`SpatialResidualizer`, JD-2 in the JEPA-deep-integration PRD) — trunk
embedding → PCA-16 → per-treatment Ridge probe → residualized treatment features, with graceful no-op when
no trunk is present. JD-1 (Stage 2 OOF as DML outcome nuisance) is the zero-risk precursor.
**Idea:** A richer PI-JEPA trunk is a *better spatial deconfounder*. The whole point of the residualizer is to
strip spatially-structured confounding from treatment features before DML. If PI-JEPA produces a trunk that
captures directional spatial structure the isotropic trunk missed, the residualizer removes more confounding,
and your causal ATE credible intervals tighten.
**Why it advances things:** This is the bridge from "PI-JEPA improves prediction" to "PI-JEPA improves *causal
inference*" — the headline goal of the whole SPARC framework (associational → causal). It is the most
strategically important plug-in because it connects the JEPA work to the framework's reason for existing.
**Research grounding:** Frisch-Waugh-Lovell (for JD-1); Schölkopf et al. (2021, causal representation learning)
and Chernozhukov et al. (2018, Double ML) for JD-2.
**Cost / risk:** Low-medium (the design is fully specified). Risk: measure the deconfounding lift (Moran's I of
residualized features before/after) directly; don't assume a better trunk automatically helps Stage 3 — prove it.

---

## Category B — Items that EXTEND existing machinery into new geospatial capability

Partial infrastructure exists; these are real new code on a proven base.

### B1 — EXTEND — Multi-step latent rollout as a spatial world model
**What exists:** single-step `latent_rollout.py`, action embedding, the Phase 2 action-conditioned FiLM
predictor, and a fully-specified design (JD-4/JD-5) for chaining + a physics cascade table.
**Idea:** Chain the predictor: `h_{t+1} = predictor(h_t, action_embed(action_t))` to simulate compound,
multi-year interventions ("canopy year 1 → cool roofs year 2 → permeable pavement year 3") as a trajectory.
Two modes: pure latent chaining (fast, capped at ~5 steps to bound drift) and re-encode through the trunk
using a physics cascade table (`tree_planting → {ndvi:+0.8, albedo:−0.05, et:+0.3}`).
**Why it advances things:** Turns JEPA from a representation learner into a *world model* — the thing that
makes JEPA architectures interesting in the broader ML literature (V-JEPA-2-AC). For SPARC specifically it
unlocks sequential intervention planning, which no urban-climate tool offers.
**Research grounding:** Assran et al. (2025, V-JEPA 2, arXiv:2506.09985) action-conditioned rollout; Camacho &
Bordons (2007) model predictive control as the eventual planning layer.
**Cost / risk:** Medium. Risk: latent drift over steps — the 5-step cap and the re-encode mode both exist
specifically to manage this. Validate that a 1-step rollout exactly matches the existing single-step output
(regression guard) before trusting multi-step.

### B2 — EXTEND — Anisotropic / directional patch masking per *variable* (not per city)
**What exists:** I1 (per-city anisotropic mask) and the Stage 0 V×V effective-range matrix (per-variable-pair
spatial ranges and directions).
**Idea:** I1 uses one anisotropy ellipse per city. B2 uses the per-variable directional structure: each feature
channel is masked with its *own* ellipse derived from that variable's Stage 0 anisotropy. LST may be anisotropic
along a coastline while impervious is isotropic — mask each accordingly.
**Why it advances things:** Pushes the "physics-derived mask geometry" idea to its logical limit and exploits the
full richness of the Stage 0 output that even I1 leaves on the table.
**Research grounding:** Paciorek & Schervish (2006) nonstationary anisotropic covariance; Gneiting et al. (2010)
multivariate Matérn.
**Cost / risk:** Medium-high, and **gated on the same B2 anisotropy audit from the core spec** — if per-variable
θ is as ESS-limited as the aggregate θ, this amplifies noise per channel. Likely overkill at the current city
count; park unless I1 shows directionality is a large lever.

### B3 — EXTEND — Wasserstein trunk-geometry transfer for cross-climate cities
**What exists:** `wasserstein_trunk_alignment()` (OT penalty, already added to `ewc.py`), K-medoids coresets per
city in the registry (natural empirical distributions for W₂).
**Idea:** When transferring to a climatically distant holdout (e.g. an arid city when training was mostly humid),
align the *distribution* of trunk activations rather than just penalizing parameter changes. Preserves latent
geometry across climate zones, improving zero-shot calibration on out-of-distribution cities.
**Why it advances things:** Directly attacks your weakest generalization case — the BSk/arid outlier among
temperate cities — with a principled geometry-preservation guarantee rather than a heuristic.
**Research grounding:** Villani (2008); Peyré & Cuturi (2019, computational OT); the OT-vs-EWC framing in your
own derivatives file.
**Cost / risk:** Medium. Risk: only matters if you have genuinely OOD holdout cities; on an all-Cfa holdout it
won't show. Pair with the rotating-holdout requirement.

---

## Category C — FRONTIER: genuinely new research that would define the field

These are months of work with uncertain payoff. Each is publishable if it lands. Do not start more than one,
and only after Categories A and B have delivered measurable wins — you want a strong, well-understood base
before adding this much novelty.

### C1 — FRONTIER — Topological Data Analysis features for thermal-field structure
**What exists:** Laplacian eigenmaps (Stage 2), a sheaf Laplacian (PDE Term 11), and a documented intent to use
persistent homology for "heat island topology" (themes.md).
**Idea:** Compute persistent homology of the thermal surface (sublevel-set filtration on predicted temperature)
to produce features that describe the *topology* of heat — how many distinct hot regions, how they merge as you
sweep a temperature threshold, the persistence of cool corridors. Feed persistence-diagram summaries (or a
persistent-Laplacian spectrum, extending the existing eigenmap approach) as side-information to the trunk or heads.
**Why it advances things:** Moves SPARC's spatial representation from *coordinate-based* to *structure-based* —
exactly the stated long-term goal of using topology to represent overall data structure. UHI mitigation is
fundamentally about connectivity (cool corridors, heat-island merging) which coordinate features capture poorly
and topology captures natively.
**Research grounding:** Edelsbrunner & Harer (2010, *Computational Topology*); Carlsson (2009, topology and data);
persistent Laplacians (Wang, Nguyen & Wei 2020). For environmental monitoring applications, the TDA-for-spatial
literature your project already references.
**Cost / risk:** High. Persistent homology is expensive at 50k+ points (use sublevel filtrations on the fishnet
graph, not Vietoris-Rips on raw points, or it won't scale). Risk: TDA features are notoriously hard to make
*predictive* rather than merely *descriptive* — set a hard go/no-go: do persistence features beat the Laplacian
eigenmaps they'd sit beside, on the locked baseline, by ≥ 2σ? If not, it's a beautiful dead end.

### C2 — FRONTIER — Road-network graph JEPA for directional heat transport
**What exists:** sheaf Laplacian (generalizes to directed graphs), spatial attention over arbitrary graphs, and
a full design (BS-4) using OSMnx for road extraction.
**Idea:** The 30m Cartesian fishnet is isotropic — all neighbors equidistant and cardinal. Heat advects along
roads (asymmetric, canyon-channeled). Build a *second* JEPA pretext on the road-network graph where edges encode
orientation, canyon aspect ratio, and width; mask graph neighborhoods; let the trunk learn directional transport
the symmetric grid Laplacian misses. The sheaf Laplacian's asymmetric edge stalks naturally encode wind direction
× road orientation.
**Why it advances things:** This is the deepest version of "physics-informed" — the *graph topology itself* becomes
the inductive bias, not just the mask shape on a grid. It targets exactly the high-aspect-ratio canyon zones where
the current grid model underestimates extremes, which plausibly connects to the evening-correlation gap (canyon
longwave trapping).
**Research grounding:** Kipf & Welling (2017, GCN); Veličković et al. (2018, GAT); Hansen & Ghrist (2020) and
Bodnar et al. (2022) for sheaf neural networks; Oke et al. (2017) urban canyon energetics.
**Cost / risk:** High. OSMnx extraction + fishnet-to-road spatial join + a graph-attention pretext is a lot of new
surface. Risk: dual-representation training (grid JEPA + road-graph JEPA) is hard to balance and attribute. Start
with road features as *covariates* in the existing grid model (cheap) before building a full second pretext —
if canyon covariates don't move evening as covariates, the graph pretext probably won't either.

### C3 — FRONTIER — Latent diffusion for full posterior over intervention scenarios
**What exists:** `LatentScenarioDiffuser` (a DDPM latent diffuser, already built), and the observation that the
JEPA VICReg variance/covariance terms regularize the latent to be near-Gaussian / non-collapsed — exactly the
condition a latent-diffusion first stage needs.
**Idea:** Instead of a point estimate per intervention scenario, train a conditional diffusion model in the trunk's
latent space to produce a *full posterior* — sample many plausible spatial temperature fields under "increase
canopy by 20%," yielding per-pixel credible intervals on the *outcome of an action*, not just on a prediction.
**Why it advances things:** Uncertainty-on-interventions is the single most valuable thing for the planning/insurance
use case the framework targets — "tree planting cools this block by −1.2°C, 90% CI [−2.1, −0.4]" with a full spatial
distribution. No spatial UHI tool offers posterior scenario fields.
**Research grounding:** Ho et al. (2020, DDPM); Rombach et al. (2022, latent diffusion); the VICReg-as-latent-
regularizer connection (Bardes et al. 2022) noted in your own derivatives.
**Cost / risk:** High. A 256-dim trunk latent is large for a diffusion UNet — compress to a ~32-dim bottleneck
first. Risk: diffusion training is finicky and the scenario engine already produces uncertainty via MC-Dropout +
NUTS; you must show the diffusion posterior is *better-calibrated* (coverage test against held-out reality), not
just fancier. This is the highest-ceiling, highest-variance item — do it last, if at all.

---

## Category D — Cross-cutting capability multipliers (do alongside, not instead)

Not JEPA changes per se, but they multiply the value of everything above.

- **Rotating-holdout validation harness** — already a prerequisite, listed here because it is itself a deliverable.
  Without it, every Category B/C gain is an unverified single-city anecdote.
- **Trunk-embedding capability-vector registry** (`TrunkLoader` extension, already in derivatives) — retrieve the
  K nearest trunks for a new city by embedding similarity rather than exact climate-zone match. Makes zero-shot
  trunk initialization smarter and feeds the Wasserstein transfer (B3).
- **Fold-level reproducibility (`FoldState.to_json`)** — serialize the exact fold recipe so any blue-sky run is
  reproducible and resumable. The variance discipline depends on being able to re-run an exact config; this makes
  that cheap.

---

## Recommended exploration sequence

```
Core PI-JEPA validated  ──►  A2 energy-balance pretext (cheap, complementary prior)
                        ──►  A1 Matérn-weighted target (natural I1 successor)
                        ──►  A3 trunk → causal deconfounding (connects JEPA to the framework's purpose)
                                   │
                        ──►  B1 multi-step world model (unlocks planning)
                        ──►  B3 Wasserstein transfer (fixes OOD-city generalization)
                                   │
                        ──►  pick ONE frontier: C1 TDA  OR  C2 road-graph  OR  C3 diffusion
                                   (C2 if evening/canyon is still the gap; C3 if planning/uncertainty is
                                    the priority; C1 if connectivity/corridor structure is the research story)
```

## Honest closing assessment

The plug-ins in Category A are where the real near-term value is, and A3 specifically is the one that matters
most strategically — it is the bridge from "we improved a prediction metric" to "we improved causal inference,"
which is the entire reason this framework exists. The frontier items in Category C are genuinely exciting and
genuinely risky; each one is a paper if it works and a month lost if it doesn't. The right discipline is to bank
the certain wins (A) and the strong extensions (B) first, so that when you spend a month on a frontier bet you do
it from a position of a strong, well-understood, reproducible baseline — and you can tell whether the exotic method
actually beat the simpler thing it replaced. The history in this project's own journals (correlation collapses,
unexplained schedule regressions) is a standing reminder that novelty without rigorous attribution produces motion,
not progress.
