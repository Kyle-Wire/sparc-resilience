---
name: sparc-roadmap
description: Deep-dive into SPARC's goals and architecture, then produce a prioritized roadmap of future improvements scored by end-user ROI and spatial world-model breakthrough potential. Use when the user wants to understand the program's direction, identify high-impact next steps, plan a research sprint, evaluate what to build next, or asks "what should SPARC do next", "where should we focus", "what has the most ROI", "spatial world model opportunities", or invokes /sparc-roadmap.
argument-hint: 'Optional focus area (e.g., "zero-shot", "end-user UX", "causal inference", "spatial world model") or leave blank for full roadmap.'
---

# SPARC Roadmap Skill

## Purpose

Produce a living, prioritized roadmap for SPARC that:
1. Is grounded in the program's actual architecture and current state
2. Scores every candidate improvement by **end-user ROI** (practitioner time saved, decision quality, new capabilities unlocked)
3. Highlights **spatial world model breakthroughs** that could leap SPARC ahead of the SOTA

---

## Phase 1 — Load Program Context

Read these files in order before forming any opinions. Do not skip any.

### Required reads

| File | Purpose |
|------|---------|
| `README.md` | Mission, pipeline overview, supported domains, current results |
| `docs/roadmap/SPARC_Future_Roadmap.md` | Official phased roadmap (Phases 1–6) |
| `docs/roadmap/SPARC_SOTA_Report.md` | State-of-the-art positioning and technical differentiators |
| `docs/research/themes.md` | Core intellectual DNA — what SPARC *is* |
| `docs/research/backlog.md` | Ranked implementation backlog with complexity estimates |
| `docs/research/derivatives.md` | Novel research ideas (new cross-pollination, not yet built) |
| `docs/roadmap/SPARC_Integration_Status.md` | What is built vs. wired vs. missing |

Use semantic_search or grep_search to fill any gaps not covered by the above.

### Optional deep-dives (load only if relevant to the focus area)

| File | Purpose |
|------|---------|
| `sparc/models/neural_meta.py` | SPARCMetaLearner architecture |
| `sparc/training/ewc.py`, `replay.py` | Continual learning modules |
| `sparc/causal/` | Full causal inference stack |
| `sparc/physics/pde_loss.py` | 10-term PDE curriculum |
| `sparc/inference/` | Zero-shot and few-shot inference modules |

---

## Phase 2 — Understand the End-User

Before scoring anything, establish **who benefits and how**. SPARC's primary end-users:

| Persona | What they need | ROI metric |
|---------|---------------|------------|
| **Urban planner / climate analyst** | Fast, defensible scenario answers; budget allocation maps | Hours saved per planning cycle; quality of intervention decisions |
| **Academic researcher** | Reproducible causal inference; publishable uncertainty estimates | Paper credibility; novel methodology claims |
| **City government / resilience officer** | Equity-audited outcomes; explainable recommendations | Political defensibility; grant eligibility |
| **Consultant / domain specialist** | Multi-domain portability; self-configuring pipeline | Billable hours saved; client confidence |
| **Zero-data city** (future) | Predictions without ground sensors | Market expansion; access equity |

For each roadmap item, ask: **which personas benefit most, and by how much?**

---

## Phase 3 — Evaluate Each Candidate

Score every candidate improvement (from backlog, derivatives, roadmap phases, or new ideas) on four axes. Use the ROI scoring guide in [REFERENCE.md](./REFERENCE.md).

### Scoring axes

| Axis | Scale | Description |
|------|-------|-------------|
| **User ROI** | 1–5 | Direct practitioner value: speed, decision quality, new capability |
| **Spatial World Model Lift** | 1–5 | Does this advance SPARC's ability to reason about spatial fields as a generalist model? |
| **Implementation Effort** | S/M/L/XL | Low = hours; Medium = days; Large = weeks; XL = months |
| **Dependency / Unlock** | Boolean | Does this unblock higher-value work? |

### Priority formula

```
Priority = (User ROI × 2 + Spatial World Model Lift) × Dependency Multiplier
           ─────────────────────────────────────────────────────────────────
                              Effort Weight (S=1, M=2, L=4, XL=8)
```

`Dependency Multiplier`: 1.5 if the item unblocks ≥2 other high-value items, 1.0 otherwise.

---

## Phase 4 — Spatial World Model Breakthrough Scan

The **North Star** from `SPARC_Future_Roadmap.md`:
> *A practitioner types a city name, draws a boundary, or provides a single weather station reading. SPARC produces a probabilistic spatial field, a causal attribution map, and a counterfactual intervention analysis — with no local training data.*

This is SPARC as a **spatial world model** — analogous to a foundation model for language, but over geographic space with physics constraints. Evaluate every candidate through this lens.

### Breakthrough categories to scan

1. **Zero-shot spatial generalization** — Can new city predictions be made without any local training data? (Phases 3–4 in roadmap)
2. **Physics-grounded latent space** — Does the trunk's latent geometry encode physically meaningful spatial structure? (JEPA, SIREN, PDE loss curriculum)
3. **Cross-city transfer and continual learning** — Does performance improve with more cities, without forgetting? (EWC, replay, Wasserstein alignment)
4. **Causal world model** — Does the model learn *why* spatial patterns exist, not just *what* they look like? (MC³ + NUTS + DML + DoWhy)
5. **Formal spatial consistency guarantees** — Is the model's spatial reasoning scale-invariant and MAUP-resistant? (Sheaf Laplacian, multi-scale consistency)
6. **Action-conditioned prediction** — Can the trunk predict spatial consequences of interventions without re-training? (JEPA ActionEmbedding, α-field)

For each breakthrough category, identify: what is built, what is wired, what is missing, and what is the highest-leverage next step.

---

## Phase 5 — Output: Structured Roadmap

Produce a roadmap in this format:

### Roadmap Output Format

```markdown
## SPARC Roadmap — [date]

### Program Mission Synthesis
[2–3 sentence summary of what SPARC is and its North Star]

### End-User ROI Summary
[Top 3 ROI opportunities across all personas]

### Spatial World Model Status
[Current capabilities, gaps, and breakthrough candidates ranked by spatial WM lift score]

---

### Tier 1 — Quick Wins (Effort: S/M, Priority ≥ 8)
For each item:
- **Title** | ROI: X/5 | SWM Lift: X/5 | Effort: S/M | Priority: XX
- What it does for users
- What file(s) to touch
- Success criterion

### Tier 2 — Platform Capabilities (Effort: M/L, Priority 5–8)
[Same format]

### Tier 3 — Breakthrough Investments (Effort: L/XL, Priority < 5 but SWM Lift ≥ 4)
[Same format]

### Blocked / Deferred
[Items with hard dependencies or low ROI at current stage]

---

### Recommended Next 3 Actions
1. [Specific actionable item — link to file or backlog entry]
2. ...
3. ...
```

---

## Phase 6 — Iterate and Refine

After presenting the roadmap:

1. **Ask the user** which tier or item they want to drill into
2. Use [REFERENCE.md](./REFERENCE.md) for deeper ROI analysis frameworks
3. Offer to handoff to `/research-improve` (for deep research), `/tdd` (to implement), or `/to-issues` (to create tickets)

---

## Notes

- Always cite the source file for any claim about current implementation status
- Never invent capabilities — verify against `SPARC_Integration_Status.md` and the relevant source files
- Distinguish between "built and wired", "built but not wired", and "not yet built"
- The backlog complexity ratings are ground-truth; do not override them without verifying the code
