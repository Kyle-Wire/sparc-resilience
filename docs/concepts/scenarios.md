# What-If Scenarios

This is the whole point.

Everything before this — the spatial analysis, the physics constraints, the causal validation, the uncertainty quantification — is in service of one thing: being able to answer "what would happen if we did X?" in a way that's honest, physically grounded, and causally defensible.

But "what-if" is harder than it sounds. Let's talk about why.

---

## The Extrapolation Problem

The most naive approach to a scenario is also the most common: fit a model on your data, then change an input and see what the model predicts.

This works if the scenario is close to the data you trained on. It fails badly if it isn't.

Imagine you've trained on a city where tree canopy ranges from 5% to 35% across neighborhoods. Now you ask: "What if we raised canopy to 60% everywhere?" Your model will give you an answer. It has no choice — you're feeding it numbers, and it produces numbers. But those numbers are **extrapolations into territory the model has never seen**. The model learned the relationship between canopy and temperature in the 5%–35% range. Whether that relationship holds at 60% is unknown. And without a physical constraint, the model will happily extrapolate in whatever direction its learned function happens to point — which may bear no resemblance to physical reality.

SPARC addresses this in two ways.

First, it checks whether your proposed scenario is within the physical support of the training data (using a statistical measure called the Mahalanobis distance). If your scenario is far from any observed conditions, SPARC flags this explicitly. The prediction still comes out — but with a warning that you're in extrapolation territory.

Second, physics constraints provide a safety net. Even in extrapolation, a model that is constrained to satisfy thermodynamic equations can't predict that trees cause warming. The physics provide a floor of physical plausibility that purely statistical models don't have.

---

## The Causal Twist

Here's the deeper problem with naive scenario simulation: even if you stay within the training distribution, the prediction might be wrong for causal reasons.

Consider a neighborhood where lots of trees and low temperatures tend to co-occur. A statistical model will learn this pattern. Now ask "what happens if we add trees?" The model says "temperature goes down." So far so good.

But what if trees and low temperatures co-occur because wealthy neighborhoods have both — trees from deliberate planting, and lower temperatures because of lower-density housing and more green space? In that case, the model has learned a confounded relationship. If you add trees to a dense, poor neighborhood without changing the housing density, you might not see the same cooling effect the model predicts — because the cooling in the training data wasn't entirely caused by trees.

This is why SPARC runs the causal validation stage before scenarios. The scenario predictions use causal effect estimates — the portion of the temperature-canopy relationship that's attributable to canopy itself, after removing the influence of shared causes — not raw correlations. The difference between these two quantities can be enormous.

---

## Four Ways to Compute a Scenario

SPARC chooses among four computational approaches for each scenario, automatically selecting the most informative one given what the previous pipeline stages produced.

**The fast, physics-anchored approach** uses the spatial process-rate field — the learned map of how strongly physics operates at each location — to translate a proposed intervention into a predicted effect. This is fastest and is grounded directly in the physics the model learned during training.

**The Bayesian approach** draws from the full posterior over causal edge strengths — the distribution of plausible effect sizes for each causal relationship. Instead of using a single effect estimate, it samples many times from this posterior and computes the distribution of predicted outcomes. The result is a prediction that propagates causal uncertainty all the way to the final number.

**The physics solve approach** solves the governing physical equations directly under the new forcing conditions. This captures something the other approaches miss: spatial spillovers. If you plant trees in one location, some of the cooling diffuses to neighboring locations through the physics of heat transfer. Modeling this requires actually running the physical equations, not just applying a local coefficient.

**The hybrid approach** combines all of the above: jointly simulating multiple treatments, accounting for their interactions, and composing the physical and causal uncertainties. This is the most comprehensive — and the most computationally intensive — but produces the richest picture of what an intervention would actually do.

---

## Diminishing Returns

One of the most important (and often ignored) features of real-world interventions is that they don't scale linearly. Doubling the trees doesn't double the cooling. Tripling the reflective pavement doesn't triple the albedo effect.

This is a fundamental feature of physical systems — they saturate. Trees start competing for water. Cool roofs cover an increasingly large fraction of total surface area. The marginal benefit of the next intervention decreases as you've already captured the low-hanging fruit.

SPARC builds diminishing returns into scenario predictions explicitly. The dose-response curves from the causal analysis — which show how the outcome changes as you dial up a treatment across its full range — capture where saturation begins. And the scenario simulator applies these curves, not a linear extrapolation, when computing predicted effects.

This matters enormously for budget allocation. If tree planting saturates above 40% canopy, then concentrating all your trees in one already-green neighborhood is wasteful — even if the model predicts high per-unit benefit there. The optimal strategy spreads investment across the city in a way that accounts for diminishing returns.

---

## Budget and Equity

Most policy decisions are budget-constrained. You have $5M to spend. Where does it go to do the most good?

SPARC takes the per-location predicted benefit from the scenario simulation and combines it with user-supplied cost estimates to solve an allocation problem: given this budget, which locations should receive interventions to maximize total predicted benefit?

But "maximize total benefit" can hide a lot. An allocation that cools the whole city by 1°F on average might achieve that entirely by heavily investing in already-cool neighborhoods, leaving the hottest (and often most vulnerable) areas unchanged.

SPARC reports spatial equity metrics alongside benefit estimates. The Gini coefficient of the benefit distribution tells you how concentrated or distributed the predicted benefits are. This lets decision-makers explicitly trade off total efficiency against distributional fairness — a tradeoff that most optimization tools don't even acknowledge.

---

## What a Scenario Result Actually Is

When SPARC presents a scenario result, it's showing you:

- The **predicted change** in the outcome variable at each location, given the proposed intervention
- The **uncertainty band** around that prediction, reflecting both causal uncertainty and model uncertainty
- Which **computational approach** was used and why
- Whether any locations are in **extrapolation territory**
- The **equity distribution** of predicted benefits
- Where the intervention hits **diminishing returns**

This is not a simple answer. But simple answers to complex policy questions are usually wrong answers dressed up to look confident. SPARC aims to be genuinely useful — which means being honest about what it knows, what it doesn't, and how certain you should be.

---

*That's the full conceptual arc. If you want to go deeper on the technical side, the [Pipeline Guide](../PIPELINE_GUIDE.md) walks through each stage in detail. If you want to understand how to read SPARC's outputs, the [Interpretation Guide](../INTERPRETATION_GUIDE.md) explains every number.*
