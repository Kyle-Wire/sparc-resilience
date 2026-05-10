# Spatial Statistics Research

Sources that directly shaped **Stage 0 (Correlogram Analysis)**, **Stage 1 (GWEN)**, and the geographically-weighted base models in **Stage 2**.

---

## Moran's I — Spatial Autocorrelation

> **Moran, P.A.P. (1950).** "Notes on Continuous Stochastic Phenomena." *Biometrika*, 37(1/2), 17–23. https://doi.org/10.2307/2332142

Stage 0 computes Moran's I at multiple distance lags as the first diagnostic of spatial structure. The statistic answers "does this variable cluster in space, and at what scale?" which directly drives the cross-correlogram and the effective-range matrix.

---

## Matérn Covariance — Bayesian Correlogram Fitting

> **Matérn, B. (1960).** *Spatial Variation.* Meddelanden från Statens Skogsforskningsinstitut, 49(5). (Reprinted as Springer Lecture Notes in Statistics, 1986.)

> **Gneiting, T., Kleiber, W., & Schlather, M. (2010).** "Matérn Cross-Covariance Functions for Multivariate Random Fields." *Journal of the American Statistical Association*, 105(491), 1167–1177. https://doi.org/10.1198/jasa.2010.tm09420

SPARC fits a Bayesian Matérn(κ, ν) model per variable using NUTS, retaining the full posterior over the range parameter κ. This posterior is used downstream as:
- A prior for NUTS edge sampling in Stage 3
- The basis for bandwidth selection in GWR and GWRF
- The block size for spatial cross-validation

The Matérn family was chosen because its smoothness parameter ν gives an explicit differentiability guarantee that matches domain assumptions (e.g., temperature fields are at least once-differentiable). The half-integer cases ν = 1/2, 3/2, 5/2 are computationally tractable.

---

## Anisotropic Spatial Covariance

> **Paciorek, C.J., & Schervish, M.J. (2006).** "Spatial modelling using a new class of nonstationary covariance functions." *Environmetrics*, 17(5), 483–506. https://doi.org/10.1002/env.785

Stage 0 fits separate correlograms along cardinal and 45° diagonal axes, yielding per-predictor ellipse parameters (κ_x, κ_y, θ). These are wired into every downstream kernel-dependent model so anisotropic spatial structure (e.g., along prevailing wind or coastal gradients) is captured automatically rather than assumed isotropic.

---

## Geographically Weighted Regression (GWR)

> **Fotheringham, A.S., Brunsdon, C., & Charlton, M. (2002).** *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships.* Wiley. ISBN: 978-0471496168.

> **Brunsdon, C., Fotheringham, A.S., & Charlton, M.E. (1996).** "Geographically Weighted Regression: A Method for Exploring Spatial Nonstationarity." *Geographical Analysis*, 28(4), 281–298. https://doi.org/10.1111/j.1538-4632.1996.tb00936.x

GWR is the primary local linear base model in Stage 2. Its spatially-varying coefficients serve two purposes: (1) a strong predictor on their own (R² ≈ 0.83 in the Providence UHI study) and (2) the foundation for the Bayesian MGWR ensemble whose per-cell β posteriors become priors for Stage 3 NUTS.

---

## Multiscale Geographically Weighted Regression (MGWR)

> **Fotheringham, A.S., Yang, W., & Kang, W. (2017).** "Multiscale Geographically Weighted Regression (MGWR)." *Annals of the American Association of Geographers*, 107(6), 1247–1265. https://doi.org/10.1080/24694452.2017.1352480

When the Bayesian MGWR ensemble is enabled, SPARC draws bandwidth values from the Stage 0 Matérn posterior to stack many GWR fits into a per-cell β posterior (mean, std, 89% HDI). This follows the MGWR insight that different predictors operate at different spatial scales and that uncertainty in the bandwidth should propagate into the coefficients.

---

## Geographically Weighted Random Forest (GWRF)

> **Georganos, S., Grippa, T., Gadiaga, A.N., Linard, C., Lennert, M., Vanhuysse, S., Mboga, N., Wolff, E., & Kalogirou, S. (2019).** "Geographical random forests: a spatial extension of the random forest algorithm to address spatial heterogeneity in remote sensing and population modelling." *Geocarto International*, 36(2), 121–136. https://doi.org/10.1080/10106049.2019.1595177

GWRF extends random forests to incorporate local spatial context via a distance-decay kernel for subsetting training observations per prediction location. In Stage 2, GWRF is the strongest non-linear classical base model (R² ≈ 0.90 in Providence UHI) and its local feature importance maps complement GWR coefficient maps.

---

## Geographically-Weighted Elastic Net (GWEN / Stage 1)

> **Tibshirani, R. (1996).** "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

> **Zou, H., & Hastie, T. (2005).** "Regularization and Variable Selection via the Elastic Net." *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

GWEN (Stage 1) adapts elastic net regularization to a local geographic window (bandwidth from Stage 0) to rank predictor importance across space. The lasso and elastic-net literature provides the shrinkage theoretical basis; the geographically-weighted wrapping follows the same framework as GWR.

---

## Laplacian Eigenmaps — Spatial Side Information

> **Belkin, M., & Niyogi, P. (2003).** "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation." *Neural Computation*, 15(6), 1373–1396. https://doi.org/10.1162/089976603321780317

> **Griffith, D.A. (2003).** *Spatial Autocorrelation and Spatial Filtering: Gaining Understanding Through Theory and Scientific Visualization.* Springer. https://doi.org/10.1007/978-3-540-24806-4

The Laplacian eigenmaps of the spatial weight matrix are injected as side-information features into both the meta-learner and the OLS baseline. In the Providence UHI study, 150 Laplacian eigenmaps provided a +4.2 pp R² uplift over the meta-ensemble without them. The eigenvectors encode global and local spatial patterns in the graph structure of the observation network.

---

## Spatial Cross-Validation

> **Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guélat, G., Kery, M., Lahoz-Monfort, J.J., Schofield, M.R., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F., & Dormann, C.F. (2017).** "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

> **Valavi, R., Elith, J., Lahoz-Monfort, J.J., & Guillera-Arroita, G. (2019).** "blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models." *PLOS ONE*, 14(7), e0225111. https://doi.org/10.1371/journal.pone.0225111

Stage 2 uses buffered spatial block cross-validation where the block size is set to the outcome variable's effective correlation range from Stage 0. The buffer prevents spatial leakage — the phenomenon where standard k-fold CV gives over-optimistic estimates because train and test points are close enough to be autocorrelated.
