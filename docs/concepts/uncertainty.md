# Embracing Uncertainty

Every number in SPARC's output comes with a range. Not a single predicted temperature, but a range of plausible values. Not a single causal effect estimate, but a distribution over possible effect sizes. Not one scenario prediction, but a cloud of possibilities with quantified probability.

If you're used to tools that give you clean single-point answers, this might feel like hedging. It isn't.

Single-point predictions without uncertainty are overconfident. They present a specific number — 87.3°F — without acknowledging that this number is the output of an imperfect model trained on incomplete data with unverified causal assumptions. The certainty is fake. The precision is theater.

Uncertainty quantification isn't an admission of weakness. It's an act of honesty. And once you understand what the uncertainty actually represents, it becomes one of the most useful things SPARC produces.

---

## Two Kinds of Uncertainty

There's a crucial distinction that's worth making explicit.

**Aleatoric uncertainty** is irreducible randomness in the world. Even if you had a perfect model and infinite data, you couldn't predict exactly where the next lightning strike will land. Some things are just variable.

**Epistemic uncertainty** is uncertainty from *not knowing enough*. If you had more data or a better model, this kind of uncertainty would shrink. It's the uncertainty that comes from the limits of your knowledge, not the limits of the universe.

SPARC's uncertainty bands are primarily epistemic. They tell you: given the data you have, the model architecture, and the causal assumptions, this is the range of predictions that are consistent with the evidence. Collect more data, and the band narrows. Improve the model, and the band narrows. The band is an honest report of what the evidence actually supports.

---

## How SPARC Generates Uncertainty Estimates

SPARC generates uncertainty from multiple sources simultaneously, which it then combines into the bands you see in outputs.

**Model ensemble spread.** SPARC trains several fundamentally different model families — linear, non-linear, semi-parametric — and combines them into a meta-learner. The spread of predictions across these models captures one source of epistemic uncertainty: disagreement about the right functional form.

**Monte Carlo sampling.** At inference time, SPARC runs its neural meta-learner hundreds of times with different random dropout configurations. Each run produces a slightly different prediction. The distribution across these runs approximates a Bayesian posterior over the model's predictions — a statistically principled way to turn a deterministic model into an uncertainty-aware one.

**Causal posterior sampling.** For scenario predictions, SPARC doesn't use a single causal effect estimate. It draws samples from the full posterior distribution over causal edge strengths — the result of a Bayesian inference procedure that propagates the uncertainty in your causal model all the way through to the final prediction.

The result is an 89% credible interval: a range of values such that, given the evidence and assumptions, there's an 89% probability the true value lies within it. (SPARC uses 89% rather than 95% as a matter of convention in Bayesian practice — it's a less "magic-number-feeling" threshold that doesn't encourage false precision.)

---

## Reading the Bands

When you see a prediction with a wide band, that's information. It's telling you something specific:

- **Wide band on a model prediction:** This location is difficult to predict from the available features. Perhaps it's an unusual combination of conditions. Perhaps the data is sparse nearby. Trust this prediction less; consider collecting more data here.

- **Wide band on a causal effect:** The evidence for this causal relationship is thin, or there are many plausible causal stories consistent with the data. You may need more data, or a stronger prior from domain knowledge, or a different experimental design.

- **Wide band on a scenario prediction:** The intervention effect is uncertain. This could mean the causal pathway is unclear, the intervention is far from the training distribution, or the physical dynamics are complex. Widen your planning margin accordingly.

A narrow band means the model is confident. This is good news when the model has seen many similar situations. But narrowness alone doesn't guarantee correctness — a model can be confidently wrong if the training data doesn't represent the prediction context.

---

## Uncertainty and Decision-Making

Here's what uncertainty bands actually enable: **honest risk assessment**.

If you're a city planner deciding whether to invest $2M in a cool-roof program, you don't want a point estimate saying "this will cool the district by 1.8°F." You want to know: what's the plausible range? What's the worst plausible outcome? What's the probability that the effect is larger than 1°F — the threshold that would justify the investment?

These are the questions that uncertainty estimates let you answer. And they're the questions that actually matter for decisions made under real resource constraints.

SPARC's scenario outputs include the full distribution of predicted effects, not just the mean. This lets you plan for downside risk, not just expected value. It lets you prioritize interventions that have both high expected benefit *and* low uncertainty — which often aren't the same interventions.

---

## Confounding Uncertainty and Honesty

One more thing that SPARC reports: **sensitivity to unmeasured confounders** (the E-value, described in the causation guide). This is a third kind of uncertainty — not about model fit or causal posterior, but about whether the causal assumptions themselves hold.

Even if SPARC is certain about the causal effect given the measured variables, that certainty evaporates if an important confounder was left out of the analysis. The E-value tells you how much it would take to flip the conclusion — and lets decision-makers assess whether unmeasured confounders of that magnitude are plausible in their domain.

Together, these uncertainty reports form a complete picture: prediction uncertainty, causal uncertainty, and assumption uncertainty. Real analysis acknowledges all three.

---

*Next: [What-If Scenarios](scenarios.md) — what it actually means to simulate an intervention, and why it's more subtle than you'd think.*
