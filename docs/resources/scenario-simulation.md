# Scenario Simulation & Decision Research

Sources that shaped **Stage 4** — the scenario simulator, budget-constrained allocation, Pareto frontier, equity analysis, physics guardrails, Mahalanobis extrapolation detection, and the four-tier computational framework.

---

## Potential Outcomes Framework

> **Rubin, D.B. (1974).** "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies." *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350

> **Imbens, G.W., & Rubin, D.B. (2015).** *Causal Inference for Statistics, Social, and Biomedical Sciences.* Cambridge University Press. ISBN: 978-0521885881.

The counterfactual framing of Stage 4 — "what would temperature be if we planted trees here?" — is grounded in the Rubin potential outcomes framework: Y(t) denotes the outcome under treatment t, and the intervention effect is Y(1) − Y(0). The scenario simulator estimates E[Y(t)] by sampling from the causal posterior (NUTS draws) and applying the learned α(x) process-rate field, rather than naively extrapolating the trained model.

---

## Causal Counterfactual Simulation

> **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press. ISBN: 978-0521895606.

> **Bareinboim, E., & Pearl, J. (2016).** "Causal inference and the data-fusion problem." *Proceedings of the National Academy of Sciences*, 113(27), 7345–7352. https://doi.org/10.1073/pnas.1510507113

The `do`-operator semantics used in Stage 4 (intervening on a variable by setting it to a fixed value rather than conditioning on it) follows Pearl's do-calculus. The `ScenarioSimulator` explicitly separates "what happens if we force canopy to 40%?" (do-intervention) from "what happens in places that happen to have 40% canopy?" (conditioning) — a distinction that matters for non-randomized observational data.

---

## Budget-Constrained Spatial Allocation

> **Nemhauser, G.L., Wolsey, L.A., & Fisher, M.L. (1978).** "An analysis of approximations for maximizing submodular set functions — I." *Mathematical Programming*, 14, 265–294. https://doi.org/10.1007/BF01588971

> **Cornuejols, G., Fisher, M.L., & Nemhauser, G.L. (1977).** "Exceptional paper — Location of bank accounts to optimize float: An analytic study of exact and approximate algorithms." *Management Science*, 23(8), 789–810. https://doi.org/10.1287/mnsc.23.8.789

The greedy allocation optimizer in Stage 4 is an instance of the budgeted maximum coverage problem: maximize the total predicted cooling benefit subject to a cost budget. The greedy algorithm (add the highest benefit-to-cost ratio cell at each step) provides an (1 − 1/e) ≈ 63% approximation of the optimal submodular maximizer. This bound holds when the benefit function is submodular — a reasonable assumption for diminishing-returns cooling effects.

---

## Mixed-Integer Linear Programming (MILP)

> **Wolsey, L.A. (1998).** *Integer Programming.* Wiley-Interscience. ISBN: 978-0471283669.

> **Bixby, R.E. (2012).** "A Brief History of Linear and Mixed-Integer Programming Computation." *Documenta Mathematica*, Extra Volume ISMP, 107–121.

When the MILP optimizer is selected in Stage 4, SPARC uses PuLP to formulate the budget allocation as a binary program: binary decision variables per cell, a budget constraint, and a linearized benefit objective. The MILP finds the exact optimal allocation for small to medium instances, useful for formal planning documents where the greedy solution's suboptimality would be unacceptable.

---

## Pareto Frontier

> **Coello Coello, C.A., Lamont, G.B., & Van Veldhuizen, D.A. (2007).** *Evolutionary Algorithms for Solving Multi-Objective Problems*, 2nd ed. Springer. ISBN: 978-0387332543.

> **Marler, R.T., & Arora, J.S. (2004).** "Survey of multi-objective optimization methods for engineering." *Structural and Multidisciplinary Optimization*, 26, 369–395. https://doi.org/10.1007/s00158-003-0368-6

Stage 4 sweeps over budget multipliers (0.5×, 0.75×, 1.0×, 1.25×, 1.5×) and plots spend vs. total benefit — a Pareto frontier in budget-benefit space. Decision-makers use this to identify the knee point (where marginal benefit per dollar decreases sharply) rather than committing to a single budget.

---

## Equity Analysis — Gini Coefficient

> **Gini, C. (1912).** "Variabilità e Mutabilità." *Reprinted in Pizetti, E., & Salvemini, T. (Eds.), (1955). Memorie di Metodologica Statistica.* Libreria Eredi Virgilio Veschi.

> **Theil, H. (1967).** *Economics and Information Theory.* North-Holland.

Each Stage 4 allocation solution is scored for spatial equity using the Gini coefficient over the distribution of intervention benefits across census tracts or grid cells. A Gini of 0 means perfectly equal benefit distribution; a Gini of 1 means all benefit flows to one cell. The explicit equity reporting follows the environmental justice mandate that cooling interventions should not simply benefit already-advantaged neighborhoods.

---

## Mahalanobis Distance — Extrapolation Detection

> **Mahalanobis, P.C. (1936).** "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 2(1), 49–55.

> **King, G., & Zeng, L. (2006).** "The Dangers of Extreme Counterfactuals." *Political Analysis*, 14(2), 131–159. https://doi.org/10.1093/pan/mpj004

Physics guardrails in Stage 4 include Mahalanobis extrapolation guards: if a proposed intervention scenario pushes feature values beyond the training data covariance envelope, those cells are flagged as extrapolation risk before predictions are reported. King & Zeng (2006) articulates why extreme counterfactuals — scenarios far from the support of the training data — are particularly dangerous for causal estimates.

---

## Diminishing Returns in Environmental Interventions

> **Bowler, D.E., Buyung-Ali, L., Knight, T.M., & Pullin, A.S. (2010).** "Urban greening to cool towns and cities: A systematic review of the empirical evidence." *Landscape and Urban Planning*, 97(3), 147–155. https://doi.org/10.1016/j.landurbplan.2010.05.006

> **Ziter, C.D., Pedersen, E.J., Kucharik, C.J., & Turner, M.G. (2019).** "Scale-dependent interactions between tree canopy cover and impervious surfaces reduce daytime urban heat during summer." *Proceedings of the National Academy of Sciences*, 116(15), 7575–7580. https://doi.org/10.1073/pnas.1817561116

The √-taper diminishing-returns function applied to predicted cooling benefits in Stage 4 is grounded in empirical findings: Ziter et al. (2019) demonstrate that tree canopy cooling effect saturates above ~40% cover, and Bowler et al. (2010) find consistent diminishing marginal returns across greening interventions. SPARC formalizes this as a parameterized taper, with saturation points determined by Stage 3's causal PDP curves.

---

## Joint Scenario Simulation

> **Tian, J., & Pearl, J. (2002).** "A general identification condition for causal effects." *Proceedings of the Eighteenth National Conference on Artificial Intelligence (AAAI)*, 567–573.

> **VanderWeele, T.J., & Vansteelandt, S. (2009).** "Conceptual issues concerning mediation, interventions and composition." *Statistics and Its Interface*, 2(4), 457–468. https://doi.org/10.4310/SII.2009.v2.n4.a7

`JointScenarioBundle` supports three multi-treatment composition modes: sequential (apply treatments in order, each to the previous output), independent superposition (sum individual effects), and PDE-mediated interaction (solve the PDE jointly). The interaction mode follows Pearl's joint intervention formula: P(Y | do(X=x, Z=z)) ≠ P(Y | do(X=x)) + P(Y | do(Z=z)) when treatments have interaction effects, which is common in urban environments (canopy + albedo synergies).

---

## Scenario Library — Versioning and Reproducibility

> **Bollen, K.A. (1989).** *Structural Equations with Latent Variables.* Wiley. ISBN: 978-0471011712. *(Methodological foundation for structural causal models underlying scenario versioning.)*

> **Provost, F., & Fawcett, T. (2013).** *Data Science for Business.* O'Reilly Media. ISBN: 978-1449361327.

Every scenario run is appended to a versioned scenario library (`scenarios/library.jsonl`) — a Git-like, append-only JSONL log. Storing the full intervention configuration (treatment variables, target cells, budget, optimizer, tier) alongside results ensures every plan is reproducible: re-running the same configuration produces identical outputs given the same trained model state, enabling proper comparison between planning alternatives.
