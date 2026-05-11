# How Space Thinks

Here's a question: if I tell you that a particular block in a city is 87°F, what's your best guess for the temperature of the block next to it?

If you said "probably also close to 87°F" — you just understood the foundational insight of spatial statistics.

Things that are close together tend to be similar. Statisticians call this **spatial autocorrelation**, and while the name sounds intimidating, the intuition is something you've known your whole life. Hot neighborhoods cluster with hot neighborhoods. Wealthy areas cluster with wealthy areas. Disease outbreaks cluster with disease outbreaks.

This is so obvious it barely seems worth saying. But it turns out that almost all of classical statistics *ignores* it — and that turns out to be a massive problem.

---

## Why Standard Statistics Breaks in Space

Standard statistics — the kind you learned in school, the kind that runs most scientific analyses — assumes your data points are **independent**. One measurement doesn't tell you anything about the next.

That's fine for a lot of things. Coin flips are independent. People randomly sampled from a population are (roughly) independent.

But spatial data isn't independent. At all.

When your 50,000 temperature readings are all taken across a single city at 30-meter resolution, the reading at point 1,247 is *extremely* informative about the reading at point 1,248 — because they're 30 meters apart. Pretending otherwise doesn't make the correlation go away. It just makes your statistics wrong.

Specifically: standard cross-validation splits your data into training and test sets randomly. But if your test points are 30 meters from your training points, and both are measuring the same smooth temperature field, your model looks far more accurate than it actually is. You've tested whether it can memorize its immediate neighborhood, not whether it can actually predict.

SPARC catches this by doing something that sounds simple but requires care: it keeps training and test sets **spatially separated**. The "block size" for those folds is set to the natural correlation range of the outcome variable — which SPARC measures first, in Stage 0, before fitting anything.

---

## Measuring How Space Clusters

The most famous tool for measuring spatial autocorrelation is called **Moran's I**. The name honors a statistician from the 1950s, but the idea is beautifully simple.

Imagine computing the average temperature in your city. Then, for every location, measure how far that location's temperature deviates from the city average. Now ask: at locations where it's above average, are their *neighbors* also above average? Or is the hot-and-cold pattern essentially random?

Moran's I gives you a number between −1 and +1. 

- **Close to +1** means strong positive clustering — hot blocks near hot blocks, like most urban temperatures.
- **Close to 0** means the pattern is spatially random — no relationship between a location and its neighbors.
- **Close to −1** means a checkerboard pattern — every hot block surrounded by cold ones, which almost never happens in nature.

SPARC computes Moran's I at many different distance scales for every variable in your dataset. That answers a crucial question: *at what range does this variable cluster?* Is the pattern in tree canopy organized at the block level, the neighborhood level, or the district level? Different variables cluster at different scales, and knowing this changes everything about how you model them.

---

## Models That Learn Local Rules

Once you understand that space matters, you face a choice: build one model for the whole city, or let the model adapt to local conditions?

A single global model might find that trees cool things down by, say, 0.02°F per percentage point of canopy. But in a dense urban canyon, where trees also trap heat at night, the effect might be different than in an open residential neighborhood. The global model averages over all of this.

SPARC uses a class of models called **geographically weighted regression** — a family of methods that fits local relationships by weighting each data point according to how close it is to the location being predicted. The result is a map of *coefficients*, not a single coefficient — showing you where the relationship is strong, where it's weak, and where it might even change sign.

This isn't just statistically useful. It's how you produce outputs that a city planner can actually act on. "Tree canopy cools by 0.03°F per point in the East Side and 0.01°F per point downtown" is far more actionable than "tree canopy cools by 0.02°F per point on average."

---

## What SPARC Does With All Of This

Stage 0 of SPARC — before any model is trained — is entirely dedicated to measuring the spatial structure of your data. It produces a rich picture: the correlation range of every variable, the direction of any spatial anisotropy (does heat spread more strongly east-west than north-south, perhaps following a coastline?), and the cross-correlations between predictors and the outcome at different spatial scales.

All of this feeds forward automatically. The correlation range sets the cross-validation block size. The directional anisotropy shapes the kernel geometry of every downstream model. The cross-correlation structure informs which predictors are operating at what spatial scale.

It's not preprocessing. It's the foundation.

---

*Next: [Cause vs. Correlation](cause-vs-correlation.md) — why knowing that two things cluster together still doesn't tell you what to do about it.*
