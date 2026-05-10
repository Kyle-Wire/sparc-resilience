# SPARC Research Resources

Collected bibliography organized by the pipeline stage each source directly inspired. Every entry here corresponds to a specific design decision, algorithm, or constraint in the codebase — not general background reading.

---

## How to Use This Folder

Each file covers one domain. Within each file, entries are ordered by pipeline relevance and include:
- The full citation
- Which SPARC module or design decision it influenced
- A brief note on *what specifically* was taken from the paper

---

## Files

| File | Pipeline Stage / Component |
|------|---------------------------|
| [spatial-statistics.md](spatial-statistics.md) | Stage 0 (Correlogram), Stage 1 (GWEN), Stage 2 base models |
| [causal-inference.md](causal-inference.md) | Stage 3 (Bayesian Causal Validation) |
| [physics-ml.md](physics-ml.md) | Stage 2 neural meta-learner, PDE loss, SIREN, JEPA |
| [continual-learning.md](continual-learning.md) | V3 EWC, replay, transfer, City Registry |
| [urban-heat-island.md](urban-heat-island.md) | UHI domain template, ForceSMIP, all 13 domain templates |
| [scenario-simulation.md](scenario-simulation.md) | Stage 4 (Scenario Simulation), budget optimizer, equity |

---

## Quick Reference — Key Papers per Pipeline Piece

### Stage 0 — Correlogram Analysis
- **Moran's I**: Moran (1950), *Biometrika*
- **Matérn covariance / NUTS fitting**: Matérn (1960); Gneiting et al. (2010), *JASA*
- **Anisotropy**: Paciorek & Schervish (2006), *Environmetrics*
- **Spatial CV block size**: Roberts et al. (2017), *Ecography*

### Stage 1 — GWEN Variable Selection
- **Elastic Net**: Tibshirani (1996), *JRSS-B*; Zou & Hastie (2005), *JRSS-B*

### Stage 2 — Spatial CV & Neural Meta-Learner
- **GWR**: Fotheringham, Brunsdon & Charlton (2002), Wiley
- **MGWR ensemble**: Fotheringham, Yang & Kang (2017), *APCG*
- **GWRF**: Georganos et al. (2019), *Geocarto International*
- **Laplacian eigenmaps**: Belkin & Niyogi (2003), *Neural Computation*
- **SIREN layers**: Sitzmann et al. (2020), *NeurIPS*
- **PINNs / PDE loss**: Raissi et al. (2019), *JCP*
- **Surface energy balance**: Oke (1988), *Progress in Physical Geography*
- **MC-Dropout uncertainty**: Gal & Ghahramani (2016), *ICML*
- **Sparse spatial attention**: Vaswani et al. (2017), *NeurIPS*
- **Curriculum learning**: Bengio et al. (2009), *ICML*
- **JEPA pretraining**: LeCun (2022); Assran et al. (2023), *CVPR*; Bardes et al. (2024), *ICLR*
- **VICReg**: Bardes et al. (2022), *ICLR*

### Stage 3 — Bayesian Causal Validation
- **MC³ DAG search**: Madigan & York (1995), *International Statistical Review*
- **BGe score**: Geiger & Heckerman (1994), *UAI*
- **NUTS sampler**: Hoffman & Gelman (2014), *JMLR*
- **Double Machine Learning**: Chernozhukov et al. (2018), *Econometrics Journal*
- **CausalForest / GRF**: Wager & Athey (2018), *JASA*; Athey, Tibshirani & Wager (2019), *AoS*
- **DoWhy refutations**: Sharma & Kiciman (2020), arXiv; Pearl (2009), Cambridge
- **E-values**: VanderWeele & Ding (2017), *Annals of Internal Medicine*
- **PC-Stable**: Colombo & Maathuis (2014), *JMLR*
- **LiNGAM / DirectLiNGAM**: Shimizu et al. (2006); Shimizu et al. (2011), *JMLR*
- **GES**: Chickering (2002), *JMLR*
- **UHI causal discovery validation**: Liu & Niyogi (2020), *Urban Climate*

### Stage 4 — Scenario Simulation
- **Potential outcomes / counterfactuals**: Rubin (1974); Imbens & Rubin (2015), Cambridge
- **do-calculus**: Pearl (2009), Cambridge
- **Budget allocation (greedy)**: Nemhauser et al. (1978), *Mathematical Programming*
- **MILP**: Wolsey (1998), Wiley
- **Pareto frontier**: Marler & Arora (2004), *Structural and Multidisciplinary Optimization*
- **Equity (Gini)**: Gini (1912)
- **Mahalanobis extrapolation guard**: Mahalanobis (1936); King & Zeng (2006), *Political Analysis*
- **Diminishing returns**: Ziter et al. (2019), *PNAS*; Bowler et al. (2010), *LUPJ*

### V3 Continual Learning
- **EWC**: Kirkpatrick et al. (2017), *PNAS*
- **Experience replay (iCaRL)**: Rebuffi et al. (2017), *CVPR*
- **K-medoids coreset**: Kaufman & Rousseeuw (1990), Wiley
- **Welford online statistics**: Welford (1962), *Technometrics*
- **Federated learning**: McMahan et al. (2017), *AISTATS*

### Domain Templates
- **UHI (published)**: Wire (2025), *Urban Climate* — DOI: 10.1016/j.uclim.2025.102671
- **UHI foundational physics**: Oke (1982), *QJRMS*; Oke et al. (2017), Cambridge
- **Groundwater**: Freeze & Cherry (1979), Prentice Hall
- **Air quality**: Seinfeld & Pandis (2016), Wiley
- **Wildfire**: Rothermel (1972), USDA
- **Seismic**: Kramer (1996), Prentice Hall

---

## Citation Formats

All entries use APA 7th edition format. DOIs and arXiv links are included where available. For papers without public DOIs (e.g., Matérn 1960, Gini 1912), historical publication details are noted.

---

*Last updated: May 2026*
