# Learning Physics

There's a long tradition in machine learning of treating models as black boxes. You feed in data, you get out predictions. Whether the model is physically plausible — whether it respects the laws of thermodynamics, conservation of energy, or basic fluid dynamics — has historically been someone else's problem.

This tradition has produced genuinely impressive results in domains where "physically plausible" doesn't matter: recommending movies, recognizing faces, translating text. But in the physical world — in the city, in the watershed, in the atmosphere — physically implausible predictions aren't just inaccurate. They're *wrong in a specific, dangerous way*.

---

## The Problem With Physically Ignorant Models

Imagine you've trained a model to predict urban temperatures, and it learns that in your particular dataset, areas with high canopy cover tend to have higher temperatures. This might be true in the data — maybe wealthy tree-lined neighborhoods also have more heat-generating buildings, or your dataset has some sampling quirk. The model doesn't know. It just sees the correlation and uses it.

Now you use that model to simulate a "plant more trees" scenario. The model, having learned that trees correlate with heat, predicts that more trees will make the city warmer.

This is physically absurd. Trees cool through evapotranspiration, shading, and albedo effects. But a data-only model has no mechanism to know that. It learned a pattern; it applied the pattern.

This failure mode isn't rare. It's the default behavior of any model that doesn't have physics built in.

---

## What It Means to Build in Physics

SPARC's approach is to include physical equations directly in the training objective — to make the model *accountable* to physics, not just to data.

The intuition is this: when you train a model, you have a loss function — a number that measures how wrong the model's predictions are. You update the model's parameters to make that number smaller. Standard training only includes one component: the gap between predictions and observed data.

SPARC's loss function includes additional components that measure physical violations. How much does the predicted temperature field violate the heat diffusion equation? How badly does the surface energy balance fail to close? How inconsistently do the spatial gradients of temperature align with known forcing directions?

Every time the model makes a prediction that contradicts the physics, it gets penalized — even if that prediction happened to match the data. The result is a model that has learned to be physically coherent, not just statistically accurate.

---

## Why This Makes Models Better (Not Just More Constrained)

Here's the part that surprises people: physics constraints typically *improve* predictive accuracy, not just physical plausibility.

Why? Because real physical relationships carry enormous information. If you know that heat diffuses according to a specific equation, that's equivalent to having thousands of additional training examples — all of them saying "a field that diffuses like this is more plausible than one that doesn't."

Physics is, in a sense, structured knowledge about what patterns are *possible* in the world. When your model has this knowledge, it has a much smaller space of possibilities to search through. It converges to better solutions faster, with less data, and generalizes more reliably to situations it hasn't seen.

This is especially important in environmental modeling, where data is often sparse, expensive to collect, and spatially uneven. A model that can borrow strength from physics is more robust than one that only has data to work with.

---

## A Learnable Physics Parameter

One of the more interesting pieces of SPARC's architecture is that it doesn't assume a single, fixed physical process rate across the entire study area.

Consider thermal diffusivity — how quickly heat spreads from one location to another. In a patch of asphalt, heat diffuses slowly through a dense medium. In a pond, it moves differently. In a forest canopy, differently still.

SPARC learns a spatial map of the process rate — a function of location, not a single number. This is the α(x) field: a surface that describes how strongly the physics operate at each point in space. Areas with high α diffuse heat quickly; areas with low α retain it. The model learns this from data, constrained to be physically plausible.

This learned process-rate field becomes one of the most useful outputs of a SPARC run. It tells you, in a physically meaningful way, which parts of your city have fast thermal dynamics and which have slow ones — information that directly shapes which interventions will be most effective where.

---

## Physics Guardrails in Scenario Simulation

The same physics that constrain training also protect scenario simulation.

When you ask SPARC "what happens if I paint all the roofs white?", it doesn't just extrapolate the statistical model. It applies physical bounds: albedo can't exceed 1. The energy balance must close. Temperature changes can't violate thermodynamic limits. If your proposed intervention would push variables outside the range supported by physics — not just outside the training data, but outside what's physically possible — SPARC flags this before presenting predictions.

This is the difference between a tool that tells you "your scenario predicts −15°F of cooling" (physically impossible) and one that says "your scenario approaches physical limits; here is the plausible range."

---

## What You Don't Need to Know

You don't need to understand the specific physics equations SPARC uses. You don't need to know what "heat diffusion" looks like mathematically, or what "surface energy balance" means in quantitative terms.

What matters is the *principle*: SPARC's predictions are constrained to be consistent with how heat, water, and air actually behave. When you see a scenario prediction, it's not just a number a model produced — it's a number that has been filtered through physical reality.

That's a meaningful guarantee.

---

*Next: [Embracing Uncertainty](uncertainty.md) — why error bars aren't a sign of weakness, and why you should be suspicious of any model that doesn't have them.*
