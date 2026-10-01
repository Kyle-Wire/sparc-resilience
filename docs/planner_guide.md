# SPARC for planners: reading the heat model

This guide is for staff who use SPARC's results page and report to decide where cooling investments go. It assumes no statistics. The [model card](results/) and `methods.md` of each run hold the technical detail.

## What SPARC answers

| Your question | Where to look | What it means |
|---|---|---|
| Where is it hottest on a hot afternoon? | **Temperature** map | Measured (campaign) and modelled afternoon air temperature for every cell. |
| What happens if we plant trees here, or across the city? | **Scenarios** and **Cooling effects** | The predicted change in air temperature for a defined change in canopy, paving or surface reflectance, with a ± range. |
| Where does each extra tree do the most good? | **Saturation** and **Cooling effects** (footprint) | Some areas still gain a lot from more canopy. Others are already near the point where more trees add little. |
| What does a budget buy? | **Budget plan** | The cells that give the most cooling per unit of effort, and how much cooling the plan delivers once all changes act together. |
| How hot will it get, and how much can we offset? | **Climate futures** | Summer afternoon temperatures under four emissions pathways and three periods, with and without the adaptation package. |
| Can we trust it? | **CV design**, the model card and the checks below | How well the model predicts neighbourhoods it never saw, and whether its effects pass independent checks. |

## Five rules for reading the numbers

1. **Read neighbourhoods, not single cells.**
   - The model learns from how temperature changes across blocks and neighbourhoods. Cooling from trees mostly comes from the surrounding area, not the cell itself.
   - Use the hex or district summaries, or look at clusters of cells.
2. **Look at the ± range.**
   - Every scenario has a standard error from re-fitting the model on different parts of the city.
   - If two options differ by less than about twice that error, treat them as equal.
   - The model card lists the smallest difference worth acting on (the "noise floor").
3. **Prefer changes inside the observed range.**
   - Each scenario reports the share of cells pushed outside the combinations of canopy, paving and reflectance that exist in the city today.
   - A high share (over about 20%) means the result leans on extrapolation.
4. **Check the causal band.**
   - Each scenario is also estimated by an independent causal method that does not use the predictive models.
   - "Within the causal band" means the two approaches agree. If they disagree, the page says so; treat that result as uncertain.
5. **Compare like with like.**
   - A +10 percentage-point canopy change and a +0.1 albedo change are different efforts.
   - Use the budget plan, or the cost settings in the configuration, to compare levers per unit of cost.

## What makes these results trustworthy

- **Honest validation.**
  - Accuracy is measured by hiding whole 2 km neighbourhoods during training and predicting them.
  - Scoring random cells would inflate the score, so the page shows both, along with the curve in between.
- **Beats the standard tools.** On the same hidden neighbourhoods the model is compared with kriging, machine-learning and interpolation baselines. The report states plainly whether it wins.
- **Fake inputs give no effect.** In the placebo tests the pipeline is re-run with tree and pavement layers moved to the wrong place, and with a made-up random layer added. They should show (almost) no effect.
- **Independent causal check.** Each effect is re-estimated by double machine learning with spatial cross-fitting and neighbourhood spillovers, then compared with the model.
- **Literature check.** City-wide effects are set against published cooling rates per +0.10 canopy or albedo.
- **Reproducible.** Every run records the exact data, code and package versions. `sparc core reproduce` re-runs it and confirms the numbers.

## Known limits (read before deciding)

- **One hot afternoon.** The model describes conditions like the campaign day (for Providence, the 2020-07-29 afternoon traverse). It says nothing about night-time heat.
- **The temperature map is itself a model product.** Heat Watch maps extend driven measurements to every cell using land-cover data.
  - Effects partly reflect that mapping model.
  - Re-fitting on the raw traverse points, when available, removes this.
- **Albedo is relative.** In the Providence data the "albedo" layer is not true broadband reflectance. Albedo effects describe a change per unit of that layer, so convert carefully before costing cool roofs or pavements.
- **No plantable-space limit yet.** Until a plantable-space layer is added, the budget plan may place trees where they cannot physically go (rooftops, roads). Treat it as a ranking of where to look, not a site plan.
- **Statistical, not engineering, effects.** Building shade, irrigation and traffic heat are not measured and can shift local effects.

## A typical workflow

1. Open the **Temperature** map and the **Climate futures** section to agree the size of the problem: today's hot spots and how many cells cross 90 °F in each future.
2. Open **Scenarios**:
   - pick the package that matches realistic ambitions;
   - note the city-wide cooling, its ± range, and whether it lies within the causal band.
3. Open **Saturation** to find where more canopy still pays off, and where it does not.
4. Open the **Budget plan**:
   - set the budget;
   - export the selected cells;
   - overlay them with plantable space, ownership and equity layers outside SPARC.
5. Before presenting, read the model card's limitations, and quote the ± ranges alongside every number.

## Glossary

| Term | Meaning |
|---|---|
| ΔT | Change in afternoon air temperature caused by a scenario (negative = cooler). |
| Footprint | Total cooling, summed over the neighbourhood, from a change in one cell. |
| Saturation (d90) | The amount of extra canopy (for example) that delivers 90% of the achievable cooling in an area. |
| Jackknife SE | The spread of a result when the model is re-fitted with parts of the city left out. |
| Held-out R² | Share of temperature variation the model predicts in neighbourhoods it was not trained on (1 = perfect). |
| Causal band | The 95% range of the independent causal estimate for the same change. |
