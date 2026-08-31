# What SPARC Is Actually Built For

**Method:** structural read of the codebase only — what the pipeline terminates in,
what its outputs are shaped like, and what has no infrastructure at all. Conclusions
derived from components, then checked against the current regulatory landscape.

---

## 1. The answer

**SPARC is a regulatory-grade spatial decision-support system for public-sector
environmental capital allocation.**

More precisely: it takes a spatial environmental problem, establishes *causal* effect
estimates with disclosed uncertainty, and produces a **ranked, costed,
equity-weighted intervention plan** with an audit trail, packaged for named
institutional audiences.

That is not a remote sensing product, not an early-warning system, and not a
simulator. Those are all things it *touches*. It is a **decision instrument**.

---

## 2. How the code says so

Six structural facts, none of which depend on interpretation.

### 2.1 The objective function is capital allocation

`sparc/decision/optimizer.py`, `rank_interventions`:

```python
score = c.equity_weight * (sign * risk_adj) / (c.cost + _EPS)
```

Equity-weighted, risk-adjusted benefit per dollar. That is not a prediction, a
detection, or a simulation — it is a **budget-allocation objective**. It is the last
thing the pipeline computes, and everything upstream feeds it.

### 2.2 The audiences are an institutional stakeholder map

`sparc/report/audience.py` ships seven templates:

```
technical · planner · public · council · scientist · equity · auditor
```

rendered as *Technical Report, Planner Brief, Public Summary, **Council Memo**,
Scientist Report, **Equity Brief**, **Audit Dossier***.

Nobody builds an Audit Dossier for a remote sensing pipeline. That list is the exact
stakeholder set of a municipal or agency capital program going through public review.
The Council template renders a cost table with `$/unit` and a budget-allocation
section.

### 2.3 Stage 3 only makes sense if you intend to intervene

The causal stack — expert DAG, backdoor adjustment, DML, per-edge NUTS posteriors,
DoWhy refutations, E-values — is **dead weight for prediction**. You do not need
identification to forecast, monitor, or detect. You need it to answer *"if we do X,
what happens?"* and to defend that answer.

Roughly 8,000 lines of the codebase exist solely to support intervention claims. That
is the single strongest signal of purpose in the repository.

### 2.4 The physics constrains plans, not dynamics

`caps.yml` files carry variable bounds, sign enforcement, delta caps, and combined
constraints (canopy + impervious ≤ 100%). The Mahalanobis extrapolation guard flags
scenarios pushed outside the training envelope.

These answer *"is this plan physically buildable and are we extrapolating?"* — not
*"is this simulation numerically correct."* The forward solver is 126 lines out of
~107,000. Physics here is a **plausibility constraint on proposals**.

### 2.5 There is no operational infrastructure at all

Searched across `sparc/decision/` and `sparc/report/`: **zero** matches for alerting,
notification, lead time, or forecast. No streaming ingest. No live data path. The
time embedding is `nn.Embedding(3, ...)` — three slots.

An early-warning system without a time axis, a lead time, or an alert is not an
early-warning system. This absence is not an oversight to be filled; it is
information about what was built.

### 2.6 The UI is a deliberation workbench

`sparc-desktop` pages: Project, Data, Variables, DAG, Physics, Models, Scenarios, Run,
Insights, Compare, **DecisionSupport**, **Report**.

There is no Alerts page, no Monitor page, no live console. There *is* a visual causal
DAG editor, a physics-constraint editor, a scenario builder, a run-comparison view,
and an audience-targeted report exporter. That is an analyst preparing a defensible
recommendation, not an operator watching a feed.

Supporting detail: the scenario library is an **append-only versioned log** — a
decision record, not a cache.

---

## 3. Why not the alternatives

| Candidate | Verdict | Reason from the code |
|---|---|---|
| **Remote sensing** | input adapter, not the product | `sparc/data/collect/` is 12 modules of ~4k lines feeding a 107k-line pipeline. Landsat, Sentinel-2, ERA5, DEM, NLCD are *sources*. Nothing downstream produces an imagery product, a classification, or a derived-index dataset. |
| **Early-warning system** | structurally absent | No lead time, forecast, alerting, or streaming anywhere. Three-slot time embedding. Exceedance heads exist but are pointed at z-score thresholds and consumed by the report layer, not an alerting layer. |
| **Simulation** | one small component | `pde_solver.py` is 126 lines, steady-state only, no time-stepping. Tier 2 of four scenario tiers. Real simulators are the whole product; here it is an option. |
| **Research / academic tool** | closest runner-up | Genuinely defensible — the Bayesian machinery and 13 domain templates support it. But research tools do not ship Council Memos, budget optimisers, equity weights, or Audit Dossiers. Those exist to survive a public process. |

The runner-up matters. SPARC *is* a credible research platform, and that is a real
fallback. But the decision layer is the part that has no academic reason to exist, and
it is the terminal stage.

---

## 4. Why this sector, now

Two live, dated, mandatory drivers — and one dead one worth knowing about.

### 4.1 FEMA benefit-cost analysis (mandatory, live)

FEMA hazard mitigation funding — BRIC, HMGP, FMA — requires a benefit-cost ratio of
**1.0 or greater**, and *"in no case will FEMA award a hazard mitigation project that
is not cost-effective."* Applicants must use FEMA-approved methodologies such as the
BCA Toolkit.

This is a legally required, spatially explicit, uncertainty-relevant computation of
exactly the shape `rank_interventions` already produces.

**Critical constraint, and it is the same lesson as the solver question:** FEMA
prescribes the methodology. **Produce BCA inputs, not a replacement BCA.** The
defensible product is per-cell avoided-damage estimates with disclosed uncertainty
that feed the approved toolkit — not a competing benefit-cost engine. FEMA has
published an alternative cost-effectiveness methodology path for BRIC/FMA, which is
worth reading closely, but the default posture is *feed the standard, don't replace
it*.

### 4.2 State cumulative-impact requirements (live as of this summer)

The federal picture receded and the states took over. As of 2026:

| State | Requirement | Status |
|---|---|---|
| New Jersey | EJ permitting law; cumulative impact assessment in overburdened communities; permit denial authority | appellate panel upheld state authority, January 2026 |
| **New York** | SEQRA amendments requiring EJ consideration **and cumulative impact assessment** for permit approvals | **effective 12 June 2026** |
| **Massachusetts** | EFSB cumulative impact analysis + site suitability for energy infrastructure | **applications filed on/after 1 July 2026** |
| CA, CO, MD, PA | active EJ programs, state-specific | ongoing |

"Cumulative impact assessment" is a spatial, multi-stressor, exposure-weighted
analysis over a defined community. That is a direct description of what this pipeline
computes — and `sparc/data/census_equity.py` (720 lines of tract-level ACS
vulnerability) plus `decision/equity.py` (`disparity_index`, layer combination) were
built for it.

Two of these obligations took effect in the last three months. That is unusually
good timing, and it is not something you can manufacture.

### 4.3 What is dead: Justice40

Justice40 (EO 14008, 40% of federal investment benefits to disadvantaged communities)
was **rescinded by EO 14148**. Do not build positioning, marketing, or a funding
thesis on it. The equity capability is not dead — its driver moved from federal
policy to **state statute**, which is more durable but geographically fragmented.
Position accordingly: the addressable market for the equity module is
state-by-state, not national.

---

## 5. What this implies for how you build

### 5.1 Defensibility is the product, not accuracy

This is the single largest reframe.

In a research setting, R² is the currency and a methods flaw is a reviewer comment.
In a permitting or grant setting, the output enters a public process where it will be
**challenged by people paid to discredit it** — opposing counsel, a competing
applicant, an intervenor group, a state auditor.

That inverts the quality bar. What matters:

- **Provenance.** Every number traceable to an input and a method version.
- **Reproducibility.** Same inputs, same run, same answer, years later.
- **Disclosed uncertainty.** Intervals that mean what they claim.
- **Method acceptance.** Established, citable methods beat novel ones.

It also promotes the review findings from housekeeping to **existential**. A
leaking cross-validation, an interval that excludes its own point estimate, or an
E-value computed from an unadjusted regression is not a nitpick when the methods
section is discovery material. Fix those before any regulatory-facing pilot. See
[`pipeline-remediation.md`](pipeline-remediation.md).

### 5.2 The simple models may be the product

Counterintuitive but important, and it cuts against the instinct to lead with the
newest thing.

In a hearing, **GWR and kriging are defensible** — decades of literature, standard
practice, an expert witness can explain them to a judge. A SIREN trunk with sparse
spatial attention, a 12-term PDE curriculum, and JEPA pretraining is, in that room, a
liability: unexplainable, uncitable, and impossible to cross-examine favourably.

This does not mean delete the neural stack. It means **the regulatory product line and
the research product line are different configurations of the same pipeline**, and
the regulatory one should default to the interpretable path with the neural stack as
an optional comparison. Ship a "defensible mode" that pins the estimator to methods
with a citation trail.

### 5.3 Offline-first is a procurement feature

Tauri desktop, no cloud, no API key required for the core pipeline, data never
leaves the machine. In consumer software that reads as a limitation. In agency
procurement it clears a security review that would otherwise take nine months.
Lead with it.

### 5.4 The report is the deliverable

Seven audience templates is not over-engineering for this sector — it is the product.
Invest there:

- **Regulatory submission packets.** Not "a report" — a NJ EJ impact statement
  section, a NY SEQRA cumulative-impact appendix, a FEMA BCA input workbook. Each
  named regulation has a required structure. Match it.
- **The Audit Dossier is the moat.** Provenance, method versions, refutation results,
  uncertainty disclosure, and the append-only scenario library, in one signed
  artifact. Nobody else ships this.

### 5.5 Sequence

1. **Remediation first.** Non-negotiable for a regulatory-facing product.
2. **Pin one regulation and build to its literal output format.** NY SEQRA
   cumulative-impact (live June 2026) or a FEMA BCA input workbook. One, fully, not
   both partially.
3. **Defensible mode** — interpretable estimator default, full citation trail.
4. **One reference engagement**, ideally with a state agency or a mid-size consultant,
   where the output actually goes into a filing. That reference is worth more than
   any benchmark.
5. **Then** widen domains. The 13 templates are latent inventory; they become
   valuable only after one is proven in a real filing.

---

## 6. Who buys, and the honest risk

**Buyers:** state environmental agencies (NJ DEP, NY DEC, MassDEP first — they own the
new mandates), city sustainability and resilience offices, regional planning
organisations, water utilities, and the engineering consultancies that serve them
(AECOM, Jacobs, Tetra Tech, Arup).

**The consultants are both channel and competitor.** They already do this work with
GIS analysts and spreadsheets, they own the client relationships, and they bill hours
that your tool would compress. Selling *to* them as a capability that wins bids is a
far better motion than selling *against* them.

**Risks worth naming:**

| Risk | Note |
|---|---|
| Regulatory-grade means liability | An output that enters a permit decision and is wrong has consequences beyond a bad review. Insurance, disclaimers, and a human-in-the-loop posture are not optional. |
| Long procurement | Agency cycles are 6–18 months. Runway must match. |
| Method acceptance risk | See 5.2. Novelty is a cost here, not a feature. |
| State fragmentation | Each state's rule differs. Serving NJ + NY + MA is three products' worth of output formatting. |
| Federal reversal | The feds receded; they can return. Do not over-index on the current federal posture in either direction. |
| Single-analyst tool | Public processes are collaborative. Desktop-only with no shared state may not survive contact with a real review team. |

---

## 7. One-line summary

Built for: **turning spatial environmental data into a defensible, costed,
equity-weighted intervention plan that survives public challenge.** Remote sensing is
how it gets data in; simulation is one of four ways it answers a scenario; early
warning is a different product it does not currently have the temporal architecture
for. The decision layer is the point.

---

*Regulatory facts current as of August 2026 and should be re-verified before any
commitment — this landscape moved twice in eighteen months. Nothing here is legal
advice.*
