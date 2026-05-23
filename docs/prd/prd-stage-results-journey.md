# PRD: Stage Results Journey — Per-Stage Pipeline Results Experience

## Problem Statement

The SPARC Desktop "Insights" page is a monolithic, audience-gated panel wall that collapses the entire pipeline into a single scrollable list. Users must self-classify as practitioner, researcher, or public before seeing any results; this gates science behind labels and prevents any user from naturally discovering the depth of what SPARC computes.

The deeper problem is philosophical: each SPARC pipeline stage is a distinct scientific act — measuring geometry, selecting variables, predicting outcomes, validating causation, and recommending interventions. These acts are both mathematically rigorous and visually beautiful. The current layout gives them no room to breathe. A Correlogram buried under "Causal diagnostics → researcher only" receives the same visual weight as a loading spinner. The science deserves better framing.

Users — researchers, planners, students — leave the pipeline run without fully understanding what was computed or why it matters. The learning opportunity that is latent in every pipeline run is lost.

## Solution

Replace the `InsightsPage` / `InsightsShell` experience with a dedicated per-stage **Results Journey**. Each of the five pipeline stages (0 = Geometry, 1 = Selection, 2 = Prediction, 3 = Causation, 4 = Decision) gets its own full-page view organized around a **wonder → structure → proof** narrative arc.

A persistent **Pipeline Navigator** strip spans all stage pages, giving users spatial orientation within the pipeline and one-click travel between stages. The audience toggle is retired entirely. Technical depth is always present and always accessible via **progressive disclosure** (`+ What this means technically` chevron sections) — never gated by self-categorization.

The experience is designed to be genuinely inspiring: large hero visualizations, narrative callouts that explain *why* the math matters, and a sense that each stage is a chapter in an argument the data is making about the world.

---

## User Stories

### Navigation & Discovery

1. As a pipeline user, I want a persistent visual timeline showing all five pipeline stages so that I always know where I am in the scientific journey and can jump to any stage.
2. As a pipeline user, I want each stage to have a distinct visual identity (color, name, icon) so that I can develop an intuitive map of the pipeline over time.
3. As a pipeline user, I want the active stage to be clearly highlighted in the navigator so that I can orient myself instantly after switching pages.
4. As a pipeline user, I want locked/incomplete stages to appear visually distinct from completed stages so that I understand what results are available.
5. As a pipeline user, I want to navigate to any completed stage directly from the sidebar under "Results" so that I can revisit any part of the journey.
6. As a pipeline user, I want a stage-name sticky strip to appear as I scroll down within a stage page so that I never lose context about which stage I'm reading.

### Hero & Wonder

7. As a pipeline user, I want each stage page to open with a large, animated hero visualization so that the first impression is awe rather than a table of numbers.
8. As a pipeline user, I want the Stage 0 hero to show an animated spatial scatter of my dataset points — revealing outward from the centroid — so that I feel the geographic scope of what was analyzed.
9. As a pipeline user, I want the Stage 1 hero to show a grid of variable-importance choropleths so that I can see at a glance which spatial patterns drove variable selection.
10. As a pipeline user, I want the Stage 2 hero to show the full-extent predictions map with a pulsing uncertainty overlay so that the model's confidence is immediately visible.
11. As a pipeline user, I want the Stage 3 hero to show the CATE spatial heterogeneity map so that the geographic distribution of causal effects is the first thing I see.
12. As a pipeline user, I want the Stage 4 hero to show two or three intervention scenarios side-by-side so that the decision comparison is immediately actionable.
13. As a pipeline user, I want each hero section to display 3–4 key statistic chips (e.g., n points, effective range, R², ATE) so that headline numbers are visible without scrolling.

### Structure & Narrative

14. As a pipeline user, I want each visualization figure to be accompanied by a sticky narrative callout on the left that explains what the chart is showing so that I understand the finding without external documentation.
15. As a pipeline user, I want narrative callouts to highlight the single most important insight extracted from that figure (e.g., "Effective range: 4.2 km — spatial autocorrelation matters at city-block scale") so that I leave each figure with a concrete takeaway.
16. As a pipeline user, I want the layout to use a left-narrative / right-chart split so that text and visualization are always in context with each other rather than separated.
17. As a pipeline user, I want static hero copy on each stage page that contextualizes the stage's scientific purpose so that even a first-time user understands why this stage exists.

### Stage 0 — Geometry

18. As a pipeline user, I want a Correlogram figure showing spatial autocorrelation decay with a threshold line at the 0.1 bandwidth cutoff so that I can see exactly where autocorrelation becomes negligible.
19. As a pipeline user, I want an Effective Ranges figure showing the bandwidth for each variable as a bar chart so that variable-level spatial structure is comparable at a glance.
20. As a pipeline user, I want an Anisotropy figure showing directional autocorrelation as a polar/radar diagram so that I can see whether spatial patterns differ by compass direction.
21. As a pipeline user, I want to understand the spatial structure of my data before seeing any predictions so that I trust the model's choices about kernel bandwidth and neighborhood size.

### Stage 1 — Selection

22. As a pipeline user, I want a variable importance ranking figure so that I can see which predictors GWEN retained and why.
23. As a pipeline user, I want to see which variables were dropped and the reason (e.g., multicollinearity, low spatial signal) so that I understand what was excluded from the model.
24. As a pipeline user, I want a spatial importance map for the top 3–5 retained variables so that I can see where each predictor's influence is strongest geographically.
25. As a pipeline user, I want to see a stability diagnostic for the selection process (e.g., bootstrap agreement across folds) so that I can assess how robust the variable list is.

### Stage 2 — Prediction

26. As a pipeline user, I want a spatially continuous predictions map as the primary figure so that model output is immediately visible as a geographic surface.
27. As a pipeline user, I want out-of-fold (OOF) performance metrics per cross-validation fold so that I can see whether prediction quality is consistent across the study area.
28. As a pipeline user, I want an R² and RMSE summary with spatial breakdown so that I can compare model performance to baselines.
29. As a pipeline user, I want a residual map overlaid on the predictions so that I can identify spatial regions where the model underperforms.
30. As a pipeline user, I want optional model comparison cards (if multiple model types were run) in the technical disclosure section so that I can evaluate model choice without it cluttering the primary view.

### Stage 3 — Causation

31. As a pipeline user, I want a dose-response curve as the primary figure so that the causal relationship between treatment and outcome is immediately visible.
32. As a pipeline user, I want the Average Treatment Effect (ATE) displayed with its Bayesian credible interval so that I understand both the magnitude and uncertainty of the causal estimate.
33. As a pipeline user, I want a CATE spatial map showing heterogeneous treatment effects so that I can see where interventions will be most and least effective.
34. As a pipeline user, I want negative-control test results visible so that I can verify the causal model is not confounded by spurious geography.
35. As a pipeline user, I want sensitivity analysis results in the technical disclosure section so that I can assess how the ATE changes under different confounding assumptions.
36. As a pipeline user, I want a KL-divergence / ELBO diagnostic in the technical disclosure section so that I can verify the Bayesian model converged appropriately.

### Stage 4 — Decision

37. As a pipeline user, I want the headline best-intervention recommendation as the primary statement so that the most actionable output is the first thing visible.
38. As a pipeline user, I want scenario outcome cards (2–3 interventions) showing predicted impact, cost, and equity score so that scenarios are directly comparable.
39. As a pipeline user, I want a budget-frontier chart showing the cost/benefit tradeoff across scenarios so that decision-makers can see the efficient frontier.
40. As a pipeline user, I want an equity map showing the geographic distribution of intervention benefits so that equity considerations are visually prominent.
41. As a pipeline user, I want uncertainty bands on all scenario projections in the technical disclosure section so that I understand the confidence range of each recommendation.

### Progressive Disclosure

42. As a pipeline user, I want each stage page to have a collapsible "What this means technically" section at the bottom so that deep technical content is always accessible but never mandatory.
43. As a pipeline user, I want the technical disclosure section to open with a smooth expand animation so that the interaction feels deliberate and rewarding.
44. As a pipeline user, I want the technical disclosure for Stage 0 to include a cross-correlogram heatmap and Matérn fit chart so that spatial statisticians can verify the kernel fitting process.
45. As a pipeline user, I want technical sections to remember their open/closed state within a session so that I don't have to re-expand them after navigation.

### Narrative & Learning

46. As a pipeline user, I want each stage to open with one to two sentences of "passion copy" that explains why this stage matters in the context of understanding the world so that the experience feels like learning, not just reporting.
47. As a pipeline user, I want narrative callout text to be generated by the Claude integration when an API key is present so that insights are tailored to my specific dataset and results.
48. As a pipeline user, I want a graceful templated fallback for narrative text when no API key is configured so that the experience is coherent even without AI.
49. As a pipeline user, I want consistent terminology across the stage pages (correlogram, effective range, CATE, ATE, dose-response) that matches the SPARC glossary so that the app teaches me the correct vocabulary as I use it.
50. As a pipeline user, I want to feel, by the time I reach Stage 4, that I've followed a complete scientific argument — from spatial structure through evidence to causal claim to decision — so that I trust the recommendation.

---

## Implementation Decisions

### Module Map

**New components (deep modules, each independently testable):**

- **`PipelineNavigator`** — Stateless presentational component. Receives `stages[]` with `{ id, label, color, status: 'active' | 'complete' | 'locked' }`. Renders the sticky 2-line strip: 2px ramp gradient bar on top, then a row of 5 stage nodes connected by a gradient line. Active node pulses; locked nodes are dashed and muted. Emits `onStageClick(id)`.

- **`StageHero`** — Receives `stageColor`, `stageLabel`, `stageNumber`, `stats: StatChip[]`, and a `heroContent` render slot (canvas animation, map, chart). Renders the full-viewport hero panel with the left accent bar, sticky metadata row, and scroll cue. Collapses gracefully when the user scrolls past it.

- **`StageStrip`** — Secondary sticky bar that appears as the hero scrolls out of view. Shows `Stage N · <Name>` with stage color accent. Sits at `top: 60px` beneath the app topbar. Implemented with an IntersectionObserver watching the hero's bottom edge.

- **`NarrativeCallout`** — Receives `headline`, `body`, `insight` (highlighted stat/quote from the chart), and an optional `loading` flag. Sticky within its figure section. Renders as a card with a left colored border. Falls back to templated content when no LLM response is available.

- **`TechnicalDisclosure`** — Receives a `label` string and `children` render slot. Renders a chevron toggle button. Expands with CSS max-height transition. Persists open/closed state in component-local state (session lifetime).

- **`Stage0Page`**, **`Stage1Page`**, **`Stage2Page`**, **`Stage3Page`**, **`Stage4Page`** — One per stage. Each composes `StageHero`, `NarrativeCallout`, the relevant existing panel components (`CorrelogramPanel`, `KernelFieldPanel`, `CatePanel`, etc.), and `TechnicalDisclosure`. These are the primary orchestration modules for each stage's content.

- **`StageResultsPage`** — Container routed from the Sidebar under the "Results" nav item. Holds `PipelineNavigator` and renders the active stage page based on internal state. Manages stage-switching and scroll position reset on stage change.

**Modified components:**

- **`Sidebar`** — `PageName` union type extended with `"Results"`. `SECTIONS` updated: "Pipeline" section gains `"Results"` (replaces or augments `"Insights"`). The existing `"Insights"` entry should remain temporarily and be marked deprecated during the transition.

- **`App.tsx`** — Import `StageResultsPage`, add it to the page switch render. Wire auto-navigation: after a successful pipeline run completes, navigate to `"Results"` defaulting to the highest completed stage.

- **`InsightsPage`** / **`InsightsShell`** — Left unchanged during implementation. `StageResultsPage` is additive. Once `StageResultsPage` is proven, a follow-up task removes `InsightsPage`.

### Navigation Architecture

A single "Results" sidebar entry navigates to `StageResultsPage`. Internal stage switching is managed by component state (no URL hash required for the desktop Tauri context). The `PipelineNavigator` is the primary stage-switching affordance; the sidebar entry always opens to the last-viewed or highest-complete stage.

### Stage Color Identity

Stage identity comes from the existing SPARC brand ramp tokens (already in `index.css`):

```
Stage 0 — Geometry:   --s0: #602468  (deep purple)
Stage 1 — Selection:  --s1: #e94d9b  (magenta)
Stage 2 — Prediction: --s2: #e73c25  (crimson)
Stage 3 — Causation:  --s3: #e79024  (amber)
Stage 4 — Decision:   --s4: #f0b632  (gold)
```

The ramp strip in `PipelineNavigator` uses `--ramp` (the full linear-gradient already defined). This is not new color work — it is surfacing tokens that already exist.

### Hero Animation Strategy

Each stage hero contains a canvas-based or SVG-based animation specific to that stage's data type. Animations must complete within ~1.5 seconds and must not block rendering of the rest of the page. Animations use `requestAnimationFrame` and are cancelled on component unmount. All canvas animation logic lives in `_useCanvas.ts` (already exists in the panels directory) or a new `useHeroAnimation.ts` hook.

### Narrative / LLM Integration

Narrative callout text follows a two-path strategy:
1. **AI path**: When a Claude API key is present (checked via `getConfig()`), callout text is requested from the sidecar using a stage-specific prompt template with the run's artifact data injected.
2. **Fallback path**: Templated strings that reference specific data values (e.g., `"Effective range: ${range} km — spatial autocorrelation is meaningful at city-block scale"`) are used when no key is configured or the AI request fails.

`NarrativeCallout` renders a skeleton loader while the AI path is in-flight.

### Progressive Disclosure Pattern

The `TechnicalDisclosure` component is always rendered (never conditionally mounted), only toggled with CSS. This ensures charts inside it initialize their Canvas/WebGL contexts on mount, preventing a blank state on first open.

### Audience Toggle Retirement

The `audience` prop on `InsightsPanelDescriptor` and the `useAudience()` hook in `InsightsProvider` are not deleted in this PRD's scope — they remain for the legacy `InsightsPage`. The new stage pages do not use audience gating at all. All content is always present; depth is controlled exclusively by `TechnicalDisclosure` placement.

### Prototype Reference

A high-fidelity HTML prototype implementing Stage 0's full design is available at `output/stage0_prototype.html`. Key patterns from the prototype that encode implementation decisions:

```
// Hero layout pattern
<div class="stage-hero"> 
  <aside class="hero-meta"> <!-- left: accent bar, stage label, stat chips -->
  <canvas id="hero-canvas"> <!-- right: animated scatter/visualization -->
</div>

// Sticky strip (IntersectionObserver target is hero bottom edge)
<div class="stage-strip sticky"> Stage 0 · Geometry </div>

// Figure layout
<div class="figure-grid"> <!-- CSS grid: 38% left / 62% right -->
  <aside class="narrative sticky-narrative"> <!-- NarrativeCallout -->
  <div class="chart-card"> <!-- visualization -->
</div>

// Progressive disclosure
<section class="tech-zone">
  <button class="tech-toggle">+ What this means technically</button>
  <div class="tech-body"> <!-- expands via max-height transition -->
</div>
```

---

## Testing Decisions

### What makes a good test here

Tests should verify **observable behavior from the outside** — what a user or parent component can see — not internal implementation details. For example: does `PipelineNavigator` render a node as "active" when the active stage prop changes? Does `TechnicalDisclosure` toggle its `aria-expanded` attribute? Does `NarrativeCallout` render skeleton while `loading` is true and callout text after?

Do not test: which CSS class names are applied internally, animation frame counts, or the exact pixel dimensions of rendered elements.

### Modules to test

| Module | Test type | Notes |
|---|---|---|
| `PipelineNavigator` | Unit / component | Verify node count, active state, locked state, click emission |
| `StageStrip` | Unit / component | Verify it's hidden initially and shown after hero scrolls out (mock IntersectionObserver) |
| `NarrativeCallout` | Unit / component | Verify skeleton state, AI text rendering, fallback text rendering |
| `TechnicalDisclosure` | Unit / component | Verify toggle open/close, aria-expanded, children rendered when open |
| `StageResultsPage` | Integration | Verify correct stage page mounts on `PipelineNavigator` click, scroll resets |
| `Stage0Page`–`Stage4Page` | Integration | Verify each stage page renders without crashing with mock artifact data |

### Prior art in the codebase

Tests in `tests/test_artifact_store.py` and `tests/test_collect_manifest.py` demonstrate testing patterns for async data-fetching modules in the Python backend. For the React components, existing panel tests (if any exist) or Vitest component tests should be used as the baseline pattern. The `_useCanvas.ts` hook should be tested via its consuming components, not in isolation.

---

## Out of Scope

- **Removing `InsightsPage`**: `InsightsPage` and `InsightsShell` are not deleted in this PRD. They remain accessible under the legacy "Insights" sidebar item during the transition and are removed in a follow-up cleanup task.
- **Mobile / responsive layout**: SPARC Desktop runs in a Tauri warm window. Responsive breakpoints below 1024px are not required.
- **Live-updating results**: Stage pages display results from the completed run artifact store. Live streaming updates during a pipeline run are handled by `RunPage` and are out of scope here.
- **Stage-level re-run controls**: Triggering partial pipeline re-runs from a stage page is not part of this PRD.
- **Export / sharing per stage**: Per-stage PDF or image export is a future enhancement.
- **Replacing the audience toggle in `InsightsPage`**: The legacy page is not modified.
- **AI narrative for all five stages simultaneously**: The initial implementation may ship AI-generated narrative for Stages 0 and 3 (where the outputs are most well-defined) and fallback text for the rest.
- **Mobile or web version of this experience**: Desktop-only.

---

## Further Notes

### Design Philosophy

The Stage Results Journey is not a dashboard redesign. It is a pedagogical redesign. The intent is that a user who runs a pipeline emerges from the Results pages with a genuine understanding of what SPARC computed, why each step was necessary, and why the final recommendation can be trusted. The math is the story; the visualizations are the illustrations.

The guiding arc for every stage page is **wonder → structure → proof**:
- **Wonder**: Hero visualization, passion copy, key statistics — the result is beautiful and surprising
- **Structure**: Narrative callouts + primary figures — here is what the math found and why it matters
- **Proof**: Technical disclosure — here is the evidence that the computation was done correctly

### Stage Arc Summary

| Stage | Hero | Primary Figures | Technical Disclosure |
|---|---|---|---|
| 0 — Geometry | Animated spatial scatter | Correlogram, Effective Ranges, Anisotropy | Cross-correlogram heatmap, Matérn fit |
| 1 — Selection | Variable-importance choropleth grid | Importance ranking, Dropped variables, Stability | Bootstrap agreement, VIF table |
| 2 — Prediction | Predictions map + uncertainty | OOF performance per fold, R²/RMSE, Residual map | Model comparison (if applicable) |
| 3 — Causation | CATE heterogeneity map | Dose-response curve, ATE + credible interval, CATE map | Neg-control results, sensitivity, ELBO/KL divergence |
| 4 — Decision | Scenario comparison (side-by-side) | Headline recommendation, Scenario cards, Budget frontier, Equity map | Uncertainty bands on projections |

### Prototype Availability

A complete working prototype for Stage 0 is committed at `output/stage0_prototype.html`. It demonstrates: the Pipeline Navigator strip, hero layout with canvas scatter animation, sticky NarrativeCallout, figure grid layout, and TechnicalDisclosure expand/collapse. This prototype is self-contained HTML/CSS/JS and can be opened directly in a browser for design review. It should be used as the visual specification for Stage 0 implementation and as the structural template for Stages 1–4.
