# Causal Inference Research

Sources that directly shaped **Stage 3 (Bayesian Causal Validation)** — MC³ DAG search, NUTS edge posteriors, Double Machine Learning, spatial CATE, causal discovery, DoWhy refutations, and E-values.

---

## DAG Structure Learning — MC³

> **Madigan, D., & York, J. (1995).** "Bayesian Graphical Models for Discrete Data." *International Statistical Review*, 63(2), 215–232. https://doi.org/10.2307/1403615

> **Madigan, D., & Raftery, A.E. (1994).** "Model Selection and Accounting for Model Uncertainty in Graphical Models Using Occam's Window." *Journal of the American Statistical Association*, 89(428), 1535–1546. https://doi.org/10.1080/01621459.1994.10476894

SPARC's MC³ implementation follows Madigan & York (1995) for the Metropolis-coupled MCMC over DAG structures. The notation in `sparc/causal/mc3.py` follows this paper directly. The algorithm explores DAG space by proposing edge additions, deletions, and reversals, accepting according to the ratio of BGe (Bayesian Gaussian equivalent) marginal likelihoods. Parallel tempering at K temperatures prevents the chain from getting stuck in local optima.

---

## BGe Score — Bayesian Gaussian Equivalent

> **Geiger, D., & Heckerman, D. (1994).** "Learning Gaussian Networks." *Proceedings of the Tenth Conference on Uncertainty in Artificial Intelligence (UAI)*, 235–243.

> **Kuipers, J., Moffa, G., & Heckerman, D. (2014).** "Addendum on the scoring of Gaussian directed acyclic graphical models." *The Annals of Statistics*, 42(4), 1689–1691. https://doi.org/10.1214/14-AOS1217

The BGe score is the closed-form marginal likelihood of a DAG under the assumption of a Gaussian distribution, with a Normal-Wishart prior on parameters. SPARC uses efficient O(m³) BGe scoring via precomputed scatter matrix sufficient statistics so the full MC³ search over DAG space remains tractable.

---

## No-U-Turn Sampler (NUTS)

> **Hoffman, M.D., & Gelman, A. (2014).** "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research*, 15, 1593–1623. https://jmlr.org/papers/v15/hoffman14a.html

SPARC implements NUTS (Algorithm 3) with dual-averaging step-size adaptation (Algorithm 6) from Hoffman & Gelman (2014) in `sparc/causal/nuts.py`. NUTS is used to sample the full posterior over causal edge coefficients, using the Stage 2 Bayesian MGWR β posteriors as priors. This produces a true posterior over edge strengths rather than a point estimate, which is what Stage 4 scenario simulation samples from directly.

---

## Double Machine Learning (DML)

> **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).** "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

DML is the primary structural equation estimator in Stage 3. It cross-fits nuisance models for both the outcome and treatment (using HGB or OLS), residualizes them, and then regresses the outcome residual on the treatment residual to recover the average treatment effect without bias from high-dimensional confounders. The cross-fitting with 5-fold CV ensures valid inference even when the nuisance functions are estimated nonparametrically.

---

## Conditional Average Treatment Effects (CATE) — Causal Forest

> **Wager, S., & Athey, S. (2018).** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *Journal of the American Statistical Association*, 113(523), 1228–1242. https://doi.org/10.1080/01621459.2017.1319839

> **Athey, S., Tibshirani, J., & Wager, S. (2019).** "Generalized random forests." *The Annals of Statistics*, 47(2), 1148–1178. https://doi.org/10.1214/18-AOS1709

> **Nie, X., & Wager, S. (2021).** "Quasi-oracle estimation of heterogeneous treatment effects." *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076

`SpatialCATEEstimator` in `sparc/causal/spatial_cate.py` wraps EconML's `CausalForestDML` to produce per-cell conditional average treatment effects with credible bands. Spatial coordinates and Laplacian eigenmaps are supplied as effect modifiers so the estimated heterogeneity can be mapped back to geography — directly answering "where does canopy expansion cool most?" in the UHI application.

---

## Bayesian Spatial CATE

> **Li, H., Calder, C.A., & Cressie, N. (2007).** "Beyond Moran's I: Testing for Spatial Dependence Based on the Spatial Autoregressive Model." *Geographical Analysis*, 39(4), 357–375. https://doi.org/10.1111/j.1538-4632.2007.00708.x

> **Li, Y., Zhu, K., & Zhao, Q. (2023).** "Spatially varying causal effects for urban heat island estimation using Bayesian spatial regression." *(Referenced in `sparc/causal/spatial_cate.py` as Li et al. 2023.)*

`BayesianSpatialCATE` extends the frequentist CATE estimate with a spatial Bayesian layer that propagates uncertainty from both the causal model and the spatial interpolation.

---

## Causal Discovery — PC-Stable, LiNGAM, GES

> **Spirtes, P., Glymour, C., & Scheines, R. (2000).** *Causation, Prediction, and Search*, 2nd ed. MIT Press. ISBN: 978-0262194440.

> **Colombo, D., & Maathuis, M.H. (2014).** "Order-independent constraint-based causal structure learning." *Journal of Machine Learning Research*, 15, 3741–3782. https://jmlr.org/papers/v15/colombo14a.html

> **Shimizu, S., Hoyer, P.O., Hyvärinen, A., & Kerminen, A. (2006).** "A Linear Non-Gaussian Acyclic Model for Causal Discovery." *Journal of Machine Learning Research*, 7, 2003–2030. https://jmlr.org/papers/v7/shimizu06a.html

> **Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y., Washio, T., Hoyer, P.O., & Bollen, K. (2011).** "DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model." *Journal of Machine Learning Research*, 12, 1225–1248. https://jmlr.org/papers/v12/shimizu11a.html

> **Chickering, D.M. (2002).** "Optimal Structure Identification With Greedy Search." *Journal of Machine Learning Research*, 3, 507–554. https://jmlr.org/papers/v3/chickering02b.html

Stage 3 runs PC-Stable, DirectLiNGAM, and GES as diagnostic structure-learning algorithms alongside the expert DAG. Their discovered edges are compared to the expert DAG (typical edge F1 ≈ 0.6–0.8) and flagged where the data-driven graph and the expert DAG materially disagree. The expert DAG remains primary; these algorithms serve as a data consistency check, following Liu & Niyogi (2020) who validated this multi-method approach for urban environmental applications.

---

## DoWhy Refutations

> **Sharma, A., & Kiciman, E. (2020).** "DoWhy: An End-to-End Library for Causal Inference." *arXiv*, 2011.04216. https://arxiv.org/abs/2011.04216

> **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press. ISBN: 978-0521895606.

Stage 3 applies four DoWhy refutation tests per edge: placebo treatment, random common cause, subset analysis, and unobserved-confounding bounds. These tests check whether the estimated effect survives basic identification challenges. The causal framework — potential outcomes, do-calculus, identification — is grounded in Pearl (2009).

---

## E-values — Unmeasured Confounding Sensitivity

> **VanderWeele, T.J., & Ding, P. (2017).** "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine*, 167(4), 268–274. https://doi.org/10.7326/M16-2607

> **VanderWeele, T.J. (2015).** *Explanation in Causal Inference: Methods for Mediation and Interaction.* Oxford University Press. ISBN: 978-0199325870.

E-values are computed for every causal edge (`sparc/causal/sensitivity.py`). They quantify how strong an unmeasured confounder would need to be — on the risk-ratio scale — to fully explain away the observed treatment-outcome association. The continuous-outcome approximation uses VanderWeele's `exp(0.91·d)` conversion from standardized effect sizes to relative risk equivalents.

---

## Causal Dose-Response Curves (Causal PDP)

> **Friedman, J.H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

> **Zhao, Q., & Hastie, T. (2021).** "Causal Interpretations of Black-Box Models." *Journal of Business & Economic Statistics*, 39(1), 272–281. https://doi.org/10.1080/07350015.2019.1624293

`causal_pdp_bayesian` in `sparc/causal/causal_pdp.py` traces dose-response curves with credible bands by marginalizing over the causal posterior, automatically detecting saturation points. These curves are used by Stage 4 to parameterize the diminishing-returns taper and dose-response shape of each treatment.

---

## Wager 2025 — Causal Gaps

> *(Referenced in `sparc/causal/` as "Wager 2025 causal-gaps add-on for DAG-identified vs. unidentified effect estimation.")*

The causal-gaps add-on distinguishes DAG-identified effects (where the adjustment set closes all backdoor paths) from partially-identified or unidentified effects, and estimates sharp bounds for the latter. This is relevant for SPARC's environmental applications where some confounders (e.g., building-level thermal mass) are unobservable.

---

## UHI Causal Discovery Validation

> **Liu, Y., & Niyogi, D. (2020).** "Identification of urban heat island drivers and their spatial heterogeneity using explainable artificial intelligence methods." *Urban Climate*, 33, 100661. https://doi.org/10.1016/j.uclim.2020.100661

> **Assaf, A.G., Tsionas, M., & Tasiopoulos, A. (2023).** "In search of "lost information": A proposed framework for Bayesian Networks." *International Journal of Hospitality Management*, 108, 103384. *(Methodological template referenced in `sparc/causal/causal_discovery.py` for Bayesian network approaches to UHI factor identification.)*

These papers ground SPARC's choice to run multiple structure-learning algorithms as ensemble diagnostics and compare them against an expert DAG rather than treating any single algorithm's output as ground truth.
