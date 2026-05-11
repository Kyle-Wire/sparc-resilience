# PRD: SPARC Labs Roadmap Documentation Suite

**SPARC Labs LLC | May 2026**
**Status: Active**

---

## Problem Statement

SPARC is a deeply sophisticated spatial analysis engine, but its strategic vision, integration gaps, future trajectory, and domain applicability exist only in scattered files (`SPARC_V4_Roadmap_Integrated.md`, README, various PRDs). There is no unified, audience-appropriate document set that can serve three distinct needs:

1. A **stakeholder/partner audience** that needs to understand what makes SPARC state-of-the-art and where it applies.
2. An **engineering audience** that needs to know exactly what is wired vs. stubbed, and what the technical path forward is.
3. A **domain expansion audience** (researchers, enterprise clients, academics) that needs to understand how SPARC maps to their specific field.

---

## Goals

1. Produce a SPARC SOTA Report that authoritatively documents every differentiating technical capability.
2. Produce an Integration Status report that is honest and specific about what is built vs. not wired vs. stubbed.
3. Produce a Future Roadmap that covers zero-shot, causal inference leadership, and emerging field integration with clear phases.
4. Produce a Domain Atlas that maps SPARC to 10 domain clusters, each with governing physics, process-rate analog, and module reuse assessment.

---

## Non-Goals

- This is documentation only — no code changes.
- Does not replace or duplicate `SPARC_V4_Roadmap_Integrated.md` (engineering scratchpad); these are polished strategic documents.
- Does not commit to delivery timelines.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| File location | `docs/roadmap/` | Keeps roadmap docs separate from user manuals and task tracking |
| SOTA Report tone | Stakeholder-readable, no code snippets | Suitable for partners, investors, academic collaborators |
| Domain Atlas tone | Stakeholder-readable with equations | Technical enough to be credible, accessible enough for domain experts |
| Integration Status | Engineering-facing, file-level specifics | Engineers need to know exactly what to wire and where |
| Future Roadmap | Engineering-facing with tiered milestones | Actionable phases, not vague aspirations |
| Domain count | 10 clusters | Full coverage without redundancy |
| Physics treatment | Governing PDE per domain + reuse flag | Makes Atlas a genuine technical reference |
| Causal pillars | 4 (spatial heterogeneity, physical identifiability, longitudinal, automated discovery) | Defines SPARC's differentiated causal inference thesis |
| Emerging fields | 8 tiered (near/medium/long-horizon) | Grounded ambition, not speculative vaporware |

---

## Technical Approach

Four Markdown files under `docs/roadmap/`:

### `SPARC_SOTA_Report.md`
- SPARC positioning statement
- Pipeline overview (5 stages, what makes each unique)
- Key differentiators vs. competing tools (DoWhy, EconML, GeoDa, etc.)
- Architecture highlights: SharedTrunk/CityHead, SIREN + PDE loss, MC³ + NUTS, Bayesian MGWR
- MAUP-aware inference as a unique spatial differentiator
- Transfer/continual learning foundation

### `SPARC_Integration_Status.md`
- Status table: built + fully wired / built + partially wired / stub only
- Detailed gap analysis: EWC, replay, temporal features, JEPA, zero-shot, few-shot
- What the `_continual` config key needs in the training loop
- LLM-DAG construction current state
- PCMCI+ / DYNOTEARS gap

### `SPARC_Future_Roadmap.md`
- 4 causal inference pillars (spatial heterogeneity, physical identifiability, longitudinal, automated discovery)
- 8 emerging field integrations with tier labels
- Zero-shot prediction phases (satellite ingestion, climate zone encoder, global registry)
- Self-sufficient spatial reasoning engine end state
- Phase dependency graph

### `SPARC_Domain_Atlas.md`
- 10 domain clusters
- Each: description, spatial outcome variable, example use case, governing PDE, process-rate analog, reuse vs. new-module flag, conceptual spec for new modules, SPARC template mapping

---

## Acceptance Criteria

- All four files exist under `docs/roadmap/`
- SOTA Report covers all 5 pipeline stages and ≥ 6 unique differentiators
- Integration Status has a status table for every V3 module with file references
- Future Roadmap covers all 4 causal pillars and all 8 emerging fields
- Domain Atlas covers all 10 clusters with PDE, process-rate analog, and reuse flag
- All equations rendered as LaTeX/KaTeX

---

## Open Questions

- None — all major decisions resolved in grill-me session.
