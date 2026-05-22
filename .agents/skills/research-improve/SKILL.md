---
name: research-improve
description: >
  Research-backed accuracy uplift for the SPARC pipeline. Explores the codebase
  in compressed mode, surfaces recent and under-explored theories that apply to
  the current implementation, explains expected accuracy/confidence gains at a
  high level, then hands off to /improve-codebase-architecture for architectural
  reasoning. Use when user wants to improve pipeline accuracy, boost model
  confidence, apply new research ideas, find forgotten theories, or says
  "research improvements", "accuracy uplift", "what research applies here",
  "find better theories", or invokes /research-improve.
argument-hint: 'Optional: target a specific stage or module (e.g., "causal module", "PDE loss", "Stage 2")'
---

# Research-Improve

Surface research-backed improvements to the SPARC pipeline. Mission: raise accuracy and confidence by connecting the current implementation to the best available theory — including recent work and ideas the field has drifted away from.

## Workflow

### Phase 1 — Compress

Activate `/caveman` immediately. Every message in this skill runs in caveman mode to minimise context window consumption during the deep exploration phases. Caveman mode stays active until Phase 4 summary is delivered.

> "Caveman mode ON. Exploring codebase."

---

### Phase 2 — Understand the Codebase

Invoke the `Explore` subagent with thoroughness = **thorough** to map the pipeline. Focus on:

- Pipeline stages and their sequence (Stage 0 → Stage N)
- Loss functions, objective terms, and physics constraints in use
- Causal modules: DAG structure, estimators, ATE/CATE methods
- Model architectures: what families, what inductive biases
- Evaluation metrics and confidence outputs
- Known gaps: TODOs, `# TODO`, `raise NotImplementedError`, or thin test coverage

Capture the findings as a compact internal map (bullet list, no prose). If the user supplied an argument, scope exploration to that module or stage.

**Deliverable**: Concise map — stages, models, losses, causal methods, gaps. Caveman format.

---

### Phase 3 — Research

With the codebase map in hand, reason across **three research lenses**. For each lens, surface 2–4 specific ideas with citations or theory names. Prioritise:

1. **Recent work** (post-2022): papers or methods the codebase hasn't incorporated yet
2. **Under-explored territory**: adjacent techniques that are well-proven but rarely applied in this domain
3. **Forgotten or rediscovered theory**: classical results (statistical, physical, causal) that modern ML pipelines often bypass but that directly address the identified gaps

**Lens A — Statistical / Causal**
Focus on: identification, variance reduction, confounding control, uncertainty quantification, calibration.

Examples of things to consider:
- Doubly-robust / cross-fit estimators (DML, AIPW) vs. what's currently used
- Nonparametric sensitivity analysis (Rosenbaum bounds, Cinelli-Hazlett)
- Conformal prediction for coverage guarantees on spatial outputs
- Higher-order interactions in heterogeneous treatment effects (CATE forests, R-learner variants)

**Lens B — Physics / PDE**
Focus on: constraint satisfaction, conservation laws, operator learning, physics-informed residuals.

Examples:
- PINN residual weighting strategies (adaptive loss balancing, curriculum weighting)
- Neural operators (FNO, DeepONet) for PDE surrogates vs. finite-difference residuals
- Symplectic or structure-preserving integrators for time-series physics
- Thermodynamic consistency constraints as soft priors

**Lens C — Representation / Geometry**
Focus on: spatial inductive biases, graph structure, invariances, domain adaptation.

Examples:
- Gaussian process priors with Matérn kernels vs. learned spatial embeddings
- Equivariant networks for rotation/translation invariance in geospatial features
- Optimal transport for distribution shift between training and deployment regions
- Graph neural networks over the causal DAG structure

**Deliverable**: Ranked shortlist (up to 8 ideas total across lenses). For each:

```
[IDEA] <name>
[LENS] A/B/C
[GAP IT CLOSES] <one line>
[EXPECTED GAIN] <metric or property that improves>
[EFFORT] Low / Medium / High
[REF] <paper title, year, or known theorem name>
```

Caveman format for surrounding prose, but idea cards formatted clearly.

---

### Phase 4 — Explain

Exit caveman mode temporarily for this phase.

Write a **plain-English briefing** (3–6 paragraphs) covering:

1. What the codebase is currently doing well and where it is leaving accuracy/confidence on the table
2. The top 2–3 research ideas from Phase 3, and *why* they apply here specifically (not generically)
3. The expected change in pipeline outputs: which metrics improve, how confidence intervals narrow, what failure modes close
4. Any risk or prerequisite for each idea (data requirements, compute cost, theoretical assumptions)

This section must be human-readable. No jargon shortcuts. The goal is for a domain expert who hasn't seen the code to understand what would change and why it matters.

**Resume caveman after this section.**

---

### Phase 5 — Hand Off to Architecture

Invoke `/improve-codebase-architecture` with the research shortlist as context.

Prompt the architecture skill with:

> "Use the following research shortlist as candidate improvements. For each item rated effort=Low or effort=Medium, identify the deepest seam in the current codebase where the change would live, apply the deletion test, and produce a before/after card."

Pass the full Phase 3 shortlist inline so the architecture skill has the research grounding.

The architecture skill will produce its HTML report as normal — candidates ranked by recommendation strength, before/after diagrams, locality/leverage framing.

---

## Completion Check

The skill is done when:
- [ ] Codebase map produced (Phase 2)
- [ ] Research shortlist delivered (Phase 3) — at least 4 ideas across at least 2 lenses
- [ ] Plain-English briefing written (Phase 4)
- [ ] `/improve-codebase-architecture` invoked with shortlist as context (Phase 5)
- [ ] Architecture HTML report opened for user

If the user supplied a scoped argument (e.g., "causal module"), all phases are scoped to that area. Exploration does not need to cover the full pipeline.

---

## Notes for the Agent

- Do not propose changes in Phases 2–3. Those phases are read-only and research-only.
- Do not implement anything directly. This skill ends at the architecture-reasoning hand-off.
- If a research idea requires a literature search beyond training knowledge, use `fetch_webpage` to retrieve an arXiv abstract or a known reference URL.
- The Phase 4 briefing is the contract between research and engineering. It should be specific enough that the architecture skill can act on it without re-reading the literature.
