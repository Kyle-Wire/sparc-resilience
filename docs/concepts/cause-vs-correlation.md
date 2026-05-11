# Cause vs. Correlation

Let's talk about ice cream.

Studies have found that ice cream sales are strongly correlated with drowning deaths. In months when more ice cream is sold, more people drown. The correlation is robust — it replicates year after year across different cities.

Should we ban ice cream?

Of course not. Both ice cream sales and drowning deaths go up in summer. Hot weather causes both. Remove summer, and the correlation vanishes. Ice cream doesn't cause drowning. They're just both caused by the same third thing.

This is the **confounding** problem, and it is absolutely everywhere in urban data — far worse than the ice cream example, because urban systems are genuinely tangled. Heat, poverty, impervious surfaces, lack of trees, pollution, flooding — all of these correlate with each other in complex ways, because cities are built by humans who made decisions that affected everything simultaneously.

If you try to do policy from correlation alone, you will get burned. Badly.

---

## What Causation Actually Means

The philosopher Judea Pearl spent decades formalizing what "causation" actually means, and the framework he developed is the one SPARC uses.

The key idea is the **intervention**. Correlation asks: "In the data, when X is high, is Y also high?" Causation asks a different question: "If I *set* X to a particular value — regardless of what caused X to be that value — what happens to Y?"

Pearl writes this as *do(X = x)* — the "do" operator. It's a small notation with an enormous conceptual weight. It's the difference between observing that smokers have higher cancer rates (correlation) and actually forcing someone to smoke (intervention). The former can be achieved by confounders; the latter cannot.

For urban planning, this distinction is everything. You're not trying to understand the correlation between trees and temperature in your existing city. You're asking: **if I plant trees here, what will happen?** That's a do-question. And answering do-questions from observational data requires causal inference, not correlation.

---

## The Causal Graph

The tool SPARC uses to represent causal structure is called a **directed acyclic graph**, or DAG. It sounds technical, but you've been drawing DAGs on whiteboards your whole life — they're the diagrams where arrows show what causes what.

Here's a simple one for urban heat:

```
Canopy ──────────────────────────────→ Temperature
   │                                         ↑
   └──→ NDVI (vegetation index) ────────────→│
                                             │
Impervious surface ──────────────────────────→│
                                             │
Albedo (reflectance) ──────────────────────→│
                                             │
Elevation ────────────────────────────────→  │  (confounder)
Distance from water ──────────────────────→  │  (confounder)
```

Each arrow says: "this causes that." No arrow says: "these are merely correlated."

The DAG does something powerful: it makes your assumptions explicit. When you draw an arrow from Canopy to Temperature, you're committing to the claim that canopy has a direct effect on temperature. When you draw Elevation as a confounder (affecting both other variables and the outcome), you're saying it's a shared cause that needs to be accounted for.

SPARC asks you to define this DAG before running the causal stage. Not because it already knows the answer — but because making the causal assumptions explicit is half the work, and it forces a kind of rigor that most analyses skip.

---

## How SPARC Tests Causal Claims

Once you've specified a DAG, SPARC doesn't just accept it. It tests it.

**Structure learning.** Multiple algorithmic approaches search the data for DAG structures that are consistent with the statistical patterns. If the data-learned structure is wildly different from your expert DAG, that's a signal worth investigating. (These algorithms aren't perfect — they can't distinguish between some types of causal structures — but agreement between your DAG and data-learned structures is reassuring.)

**Double machine learning.** For each causal edge you've specified, SPARC estimates the effect size using a technique that's doubly robust — it controls for confounders using machine learning, then residualizes both the treatment and outcome before estimating the causal coefficient. The math is elegant: it removes the influence of everything *except* the causal relationship you're trying to measure.

**Refutation tests.** This is the part that separates rigorous causal analysis from cargo-cult statistics. SPARC runs a battery of adversarial tests: What if we replace the real treatment with a random placebo? Does the effect vanish? (It should.) What if we add a random noise variable as a common cause? Does the estimate hold? What if we analyze only a random subset of the data? These tests probe whether the finding is real or an artifact.

**Spatial heterogeneity.** The average causal effect across your city tells you something, but it hides enormous variation. In some neighborhoods, planting trees has a strong cooling effect; in others, the same intervention might be half as effective. SPARC estimates spatially-varying treatment effects — giving you a map of *where* each intervention works best, not just the city-wide average.

---

## The E-value: A Measure of Honesty

One of SPARC's outputs is something called an **E-value** for each causal edge. This is a number that answers a specific question: how strong would an unmeasured confounder need to be to completely explain away the effect we've found?

If the E-value is 1.0, then even a tiny unmeasured confounder could nullify the finding — be skeptical.

If the E-value is 4.0, then an unmeasured confounder would need to be *four times* as strongly associated with both the treatment and the outcome as any of the measured confounders — a much harder alternative to believe.

This is how you report causal findings honestly. Not "we found X causes Y" but "we found evidence consistent with X causing Y, and here is exactly how wrong we'd have to be about unmeasured confounders for that to be false."

---

## What SPARC Is Not Claiming

SPARC does not claim to run randomized controlled trials from observational data. That's not possible.

What it claims is more modest and more useful: given your data, your DAG, and your assumptions about what was measured and what wasn't, here are the causal estimates that are most consistent with the evidence — along with explicit bounds on how sensitive those estimates are to the things you didn't measure.

This is exactly what good epidemiology, good econometrics, and good environmental science do. SPARC automates it at spatial scale.

---

*Next: [Learning Physics](learning-physics.md) — why the laws of thermodynamics make your model smarter, not slower.*
