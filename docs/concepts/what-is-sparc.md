# What Is SPARC?

Let's start with a problem that's real and urgent.

---

## The Problem With Maps

Imagine you're a city planner in Providence, Rhode Island. You have a heat wave coming. You know some neighborhoods will hit 95°F while others stay at 85°F. You have a budget to plant trees, install cool roofs, and resurface roads with reflective pavement — and you have to decide where to spend it.

You pull up a temperature map. It's a beautiful gradient — reds where it's hot, blues where it's cool. And then... what? 

The map tells you *what* is happening. It says almost nothing about *why*. And it says absolutely nothing about what would happen if you planted 500 trees in the East Side vs. 500 trees in Olneyville.

This is the gap SPARC is designed to close.

---

## The Difference Between Describing and Understanding

Most spatial data tools are extraordinarily good at describing the world. They can show you that impervious surfaces correlate with heat. They can cluster census tracts by demographic and environmental variables. They can draw beautiful interpolation surfaces.

But correlation isn't causation — and description isn't understanding.

Here's the thing that makes urban environments genuinely hard: *everything is correlated with everything*. Dense development correlates with heat. Dense development also correlates with fewer trees. Fewer trees correlate with heat. Poverty correlates with impervious surfaces. Impervious surfaces correlate with flooding. Flooding correlates with disease.

When you stare at a correlation matrix of an urban dataset, it looks like spaghetti. And from spaghetti, you can't make policy.

SPARC's job is to untangle that spaghetti — to take the messy web of correlations and recover the underlying causal structure: *what actually drives what*, and by how much.

---

## What SPARC Actually Does

At the highest level, SPARC runs five things in sequence:

**1. It listens to the geometry.** Before fitting any model, SPARC asks: how does this variable behave in space? Does it cluster tightly in small patches, or does it vary smoothly over kilometers? This isn't a preprocessing step — it's information that reshapes everything downstream. The scale at which a variable varies tells you which models are appropriate, how to split your data, and which relationships are plausible.

**2. It learns local rules.** The relationship between tree canopy and temperature isn't the same in a dense urban core as it is in a leafy suburb. SPARC fits models that allow the rules to *change across space* — because the physics of heat, water, and air don't ignore geography.

**3. It builds in physics.** This is the part that makes SPARC different from a very sophisticated regression. The model isn't just trying to fit the data — it's being asked to fit the data *in ways that are physically plausible*. Heat must diffuse. Energy must balance. Trees must cool (not warm). These constraints are woven into training, not bolted on afterward.

**4. It validates causation.** SPARC uses the same tools that pharmaceutical trials and economic policy analysis rely on to distinguish correlation from causation. It tests whether the causal story you've proposed is consistent with the data, and it quantifies how certain each causal claim actually is.

**5. It simulates interventions.** Once the causal structure is validated, SPARC can answer "what-if" questions with real uncertainty bounds. Not just "impervious surfaces correlate with heat" — but "if we convert this specific area from asphalt to permeable pavement, here is the predicted temperature effect, here is the 89% credible interval, and here is how that interacts with the existing tree canopy."

---

## Why This Matters

The gap between "we have a lot of data about our city" and "we know what to do" is enormous, and it's not a data problem. It's a *reasoning* problem.

SPARC is a tool for reasoning carefully — bringing together spatial statistics, causal inference, machine learning, and physics into a single coherent workflow. It won't make policy decisions for you. But it will give you something that most planning processes don't have: a quantified, uncertainty-aware, causally grounded picture of what your interventions will actually do.

That's not a small thing.

---

*Next: [How Space Thinks](how-space-thinks.md) — why your location on a map is one of the most powerful pieces of information you can have.*
