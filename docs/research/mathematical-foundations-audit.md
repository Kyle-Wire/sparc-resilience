# SPARC — Mathematical Foundations Audit

**Scope:** every core estimator in the pipeline — Stage 0 correlogram/Matérn/anisotropy, the
geographically-weighted stack (GWR/"MGWR", GWRF, GGPGAM, Bayesian MGWR ensemble), the NUTS
engine, the neural meta-learner, the PDE/physics loss, PI-JEPA, and the Bayesian causal stack
(DAG, backdoor, MC³/BGe, DML/CATE, refutations, sensitivity, interference), plus the decision
and UQ layers.

**Method:** each section states the governing theory and equations, then what the code actually
computes (with `file:line`), then the gap and the concrete fix. Findings are graded:

| Grade | Meaning |
|---|---|
| **A** | Implementation matches the theory. |
| **B** | Defensible approximation, but the deviation should be documented or tightened. |
| **C** | Materially wrong or inert — changes the numbers the pipeline reports. |
| **D** | Claimed in the README/docstrings but not actually implemented, or dead code. |

---

## 0. Executive summary

The architecture is sound and the *choice* of methods is genuinely strong — Metropolis-coupled
structure MCMC, Neyman-orthogonal DML, VICReg-regularised JEPA, split conformal, Cinelli–Hazlett
sensitivity, anonymous-exposure interference. Most of what follows is not "wrong idea", it is
"right idea, wrong constant / wrong estimator / not wired up."

Seven findings change reported numbers materially and should be treated as the next sprint:

| # | Finding | Grade | Blast radius |
|---|---|---|---|
| 1 | Two incompatible Matérn parameterisations between the fitter and the kernel consumer; `bandwidth = 1/κ` is 3–5.9× short of the practical range | **C** | every GWR/GWRF/CATE bandwidth, CV block size, kernel geometry |
| 2 | PDE residuals are divided by their own batch std with gradients flowing through the divisor → the physics loss is exactly scale-invariant | **C** | the entire "10-term PDE curriculum" |
| 3 | DML cross-fitting uses random `KFold(shuffle=True)` on spatially autocorrelated data | **C** | every CATE, every credible band, Stage 4 Tier 1/3 |
| 4 | `variable_bandwidths` are collapsed to an arithmetic mean → "MGWR" is single-bandwidth GWR | **C** | all local β, β posterior, divergence audit |
| 5 | Moran's I variance hardcoded to `1/(n−1)`; FFT path uses `1/√n_pairs` — opposite biases in the two code paths | **C** | auto-tuned block size and bandwidth; path-dependent results |
| 6 | BGe local score omits the multivariate-gamma sum and the `−m+1` determinant shift | **C** | MC³ edge-inclusion posteriors |
| 7 | Energy-balance term, GWRF anisotropy path, and action-conditioning are dead code / disabled | **D** | README claims vs. reality |

A recurring meta-pattern is worth naming on its own: **silent numerical fallbacks**. In at least
eight places the code catches an exception and substitutes a fabricated value — `β = 0` on a failed
local regression (`gwr.py:452`), `σ = 0.1·std(ŷ)` on a failed GAM interval (`ggpgam.py:306`),
`CI = ĉ ± 1.96·std(ĉ)` on failed causal-forest inference (`spatial_cate.py:237`), the full candidate
set when the backdoor criterion is unsatisfiable (`dag_definition.py:427`). Each produces a number
that is indistinguishable, downstream, from a real estimate. For a pipeline whose selling point is
calibrated uncertainty, these should fail loudly or propagate `NaN` + a mask.

---

## 1. Spatial autocorrelation — Moran's I and the correlogram

### 1.1 Theory

Global Moran's I for values $z_i = y_i - \bar y$ under spatial weights $w_{ij}$:

$$I = \frac{n}{S_0}\cdot\frac{\sum_i\sum_j w_{ij} z_i z_j}{\sum_i z_i^2},\qquad S_0=\sum_i\sum_j w_{ij}$$

Under the normality null, $E[I] = -1/(n-1)$ and

$$\mathrm{Var}_N(I)=\frac{n^2S_1-nS_2+3S_0^2}{S_0^2(n^2-1)}-E[I]^2$$

with $S_1=\tfrac12\sum_i\sum_j(w_{ij}+w_{ji})^2$ and $S_2=\sum_i(w_{i\cdot}+w_{\cdot i})^2$
(Cliff & Ord 1981). The variance depends on the *weights topology*, not just on $n$.

### 1.2 What SPARC does

`sparc/run/spatial_autocorr_comprehensive.py:150` computes $I$ correctly (vectorised, row-standardised).
Then `:157`:

```python
# Calculate variance of Moran's I (simplified formula)
# For large n, variance approximates to 1/(n-1)
variance_i = 1 / (n - 1)
```

### 1.3 Gap — **Grade C**

`1/(n−1)` is not an approximation to $\mathrm{Var}_N(I)$; it drops $S_1$ and $S_2$ entirely. For
row-standardised kNN weights, $S_0=n$, $S_1\approx n/k$, $S_2\approx 4n$, giving
$\mathrm{Var}_N(I)\approx 1/(nk)$ — smaller than $1/(n-1)$ by roughly a factor of $k$. Numerically,
for $n=2000$, $k=8$:

```
correct Var_N(I) ≈ 6.20e-05  → sd 0.0079
code    Var(I)   = 5.00e-04  → sd 0.0224
z-scores understated by 2.84×
observed I = 0.06:  correct z = 7.68   code z = 2.70
```

This is *conservative*, which sounds safe but is not, because `optimal_block_size` is defined as
**the first non-significant lag** (`:257`). Understated z-scores declare non-significance too early,
so the auto-tuned spatial-CV block size is systematically **too small** — which is exactly the
condition that lets spatial leakage back into cross-validation. The bias runs in the direction that
inflates reported CV skill.

Two more issues in the same file:

- **`:107`** — `valid_coords = self.coords[valid_mask] if hasattr(self, '_coord_mask') else self.coords[:len(values)][valid_mask]`.
  `_coord_mask` is never set anywhere in the repo, so the `else` branch always runs; but `values`
  was already subset on the line above, so `self.coords[:len(values)]` has length $m$ while
  `valid_mask` has length $n$. With any NaN present this raises `IndexError`. It only works when
  the mask is all-True.
- **`:98`** — the insufficient-data stub returns `expected_i = -1/9` instead of `-1/(n-1)`.

### 1.4 The FFT path disagrees with the exact path — **Grade C**

`fft_correlogram` (`:647`) is used whenever $n >$ `max_sample_size`, and it is a *different estimator
with a different null*:

- It computes a normalised autocovariance, not Moran's I, but stores it under the key `'morans_i'`.
- Significance uses `se = 1/√n_pairs` (`:769`), treating every gridded cell pair as independent.
  With millions of pairs this makes essentially every lag significant — the **opposite** bias to the
  exact path. So `optimal_block_size` jumps to `max_distance` on the FFT path and collapses toward
  the first lag on the exact path. **The same dataset, subsampled differently, produces materially
  different CV block sizes.**
- Normalisation divides by `np.var(values)` — the variance of the *raw points* — while the numerator
  uses the *grid-cell-averaged* field. Cell averaging removes nugget variance, so $\hat\rho(0)<1$
  systematically.
- The autocorrelation is sliced as `[:ny, :nx]` — the $(+,+)$ offset quadrant only. Since
  $\rho(-\mathbf{h})=\rho(\mathbf{h})$, that quadrant duplicates $(-,-)$ and **omits $(+,-)$ and
  $(-,+)$ entirely**. Radial averaging therefore samples only half the direction space — a real bias
  for exactly the anisotropic fields Stage 0 is meant to detect.

### 1.5 Fix

1. Implement $\mathrm{Var}_N(I)$ (and optionally $\mathrm{Var}_R(I)$ under randomisation, which adds
   the kurtosis term $b_2$) from $S_0,S_1,S_2$. Cross-check against `esda.Moran` on a fixture.
   For the correlogram specifically, prefer a **permutation null** (999 conditional permutations,
   `esda.Moran(..., permutations=999)`) — it is robust to the non-normality of most land-surface
   variables and costs little at the lag-band level.
2. Make the FFT path report the same statistic under the same null: convert the ACF to Moran's I
   with the $n/S_0$ scaling, and estimate the null by permuting the gridded field rather than
   assuming independent pairs. Add a regression test asserting the two paths agree to within
   Monte-Carlo error on a synthetic Matérn field.
3. Use the full $(\pm,\pm)$ offset plane before radial averaging.
4. Stop defining bandwidth by the raw zero-crossing (`first_zero_crossing`, `:263`) — that is a
   noise-driven order statistic. Take the **practical range from the fitted Matérn** (§2), which is
   what the parametric fit exists for.

---

## 2. Bayesian Matérn correlogram fit

### 2.1 Theory

The Matérn correlation, in the Handcock–Wallis / Stein parameterisation:

$$\rho(h)=\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\,h}{\ell}\right)^{\nu}K_\nu\!\left(\frac{\sqrt{2\nu}\,h}{\ell}\right)$$

The $\sqrt{2\nu}$ factor is what makes $\ell$ comparable across $\nu$. Closed forms:

$$\nu=\tfrac12:\ e^{-\sqrt{1}z},\qquad \nu=\tfrac32:\ (1+\sqrt3 z)e^{-\sqrt3 z},\qquad \nu=\tfrac52:\ \left(1+\sqrt5 z+\tfrac53 z^2\right)e^{-\sqrt5 z},\quad z=h/\ell$$

The alternative convention drops $\sqrt{2\nu}$ and writes $\rho(h)=\frac{2^{1-\nu}}{\Gamma(\nu)}(\kappa h)^\nu K_\nu(\kappa h)$,
giving $(1+z)e^{-z}$ and $(1+z+z^2/3)e^{-z}$ with $z=\kappa h$. **Both are valid; mixing them is not.**

The *practical range* — the distance at which $\rho=0.05$ — is the comparable, interpretable quantity:
$\approx 3/\kappa$, $4.75/\kappa$, $5.92/\kappa$ for $\nu = 0.5, 1.5, 2.5$ in the no-$\sqrt{2\nu}$ convention.

### 2.2 The parameterisation is mixed — **Grade C, highest blast radius**

- `correlogram_matern_fit.py:60-66` **fits** in the no-$\sqrt{2\nu}$ convention: `(1+z)*exp(-z)`, `(1+z+z*z/3)*exp(-z)`.
- `kernel_field.py:396-399` **consumes** in the $\sqrt{2\nu}$ convention: `(1+√3·h)*exp(-√3·h)`, `(1+√5·h+(5/3)h²)*exp(-√5·h)`.

The κ estimated in Stage 0 is fed directly into the Stage 2 kernel. Verified numerically at $\kappa=1/500\,\mathrm{m}^{-1}$:

| ν | fitter practical range | GWR-kernel practical range | ratio | fitter range ÷ (1/κ) |
|---|---|---|---|---|
| 0.5 | 1498 m | 1498 m | 1.00 | 3.00 |
| 1.5 | 2372 m | 1369 m | **1.73** | 4.74 |
| 2.5 | 2959 m | 1323 m | **2.24** | 5.92 |

So for $\nu=2.5$ the GWR kernel decays **2.24× faster** than the spatial structure Stage 0 measured.

Compounding this, `bayesian_mgwr_ensemble.py:121` sets `bandwidth_to_outcome = 1.0 / k`, and
`kernel_field.py:83` treats `bandwidth_to_outcome` as $1/\kappa$ in reverse. But $1/\kappa$ is the
correlation *length scale*, not the range — the last column above shows the true practical range is
**3× to 5.9× larger**. Net effect: every downstream bandwidth is several times too narrow, local
regressions are fit on far too few neighbours, and every model in the stack is over-localised.

**Fix:** pick one convention, define it once in `kernel_field.py`, and have `matern_correlation_np`,
`matern_correlation_torch`, and `matern_kernel_weights` all import from it. Then stop passing κ
across module boundaries at all — pass the **practical range** $h_{0.05}(\kappa,\nu)$, which is
convention-independent and is the quantity a bandwidth actually wants. Add a unit test asserting
`matern_correlation_np(h, κ, ν) == matern_kernel_weights(h, κ, ν)` for all three ν.

### 2.3 The nugget τ² is unidentified — **Grade C**

`correlogram_matern_fit.py:270`:

```python
mu = sigma2 * rho + tau2 * (h_t == 0).to(h_t.dtype)
```

`lags` are **bin centres** — `compute_correlogram` returns `(lag_min+lag_max)/2`, so `lags[0] > 0`
always. `(h_t == 0)` is therefore never true, `tau2` never enters the likelihood, and the reported
`tau2.mean` / `tau2.hdi89` are **draws from the prior**. Same defect at `anisotropy.py:315`.

**Fix:** either drop τ² from the model, or estimate the nugget properly by fitting to the empirical
*variogram* with the intercept as $h\to0^+$ (Cressie 1993 §2.4), or include an $h=0$ bin. Do not
report a prior as a posterior.

### 2.4 ν is selected by the harmonic-mean estimator, applied to the wrong quantity — **Grade C**

`correlogram_matern_fit.py:330`:

```python
log_marg = lp_max - math.log(np.mean(np.exp(lp_max - log_probs)))
```

Two problems stacked:

1. This is the **harmonic-mean estimator** of the marginal likelihood (Newton & Raftery 1994) — it has
   infinite variance and is famously unusable; Neal's assessment ("the worst Monte Carlo method ever")
   is the standard citation.
2. `log_probs` are the values returned by `flat_log_prob`, which is the **log posterior plus the
   change-of-variables Jacobian** (`nuts.py:614`), not the log likelihood. The harmonic-mean identity
   $p(y)^{-1}=E_{\theta|y}[p(y|\theta)^{-1}]$ requires the likelihood. So the quantity being maximised
   over ν is not an estimator of anything.

Effectively ν is chosen close to arbitrarily — and ν determines both the kernel shape and (via §2.2)
the κ scale.

**Fix:** return pointwise log-likelihoods from the sampler and select ν by **PSIS-LOO or WAIC**
(`arviz.compare`) — cheap, robust, and appropriate for a 3-way comparison. If a true marginal
likelihood is wanted, use bridge sampling (Meng & Wong 1996) or stepping-stone.

### 2.5 Heteroscedastic weighting is ad hoc — **Grade B**

`correlogram_matern_fit.py:257`: `weight = 1.0 + lags/lags.max()`. The comment concedes this is a
stand-in for pair counts. But $\mathrm{Var}(\hat I(h)) \propto 1/N(h)$ and `n_pairs` is already
computed and returned per lag. The anisotropy fitter *does* use `wmax/n_pairs` (`anisotropy.py:279`),
so the two fitters weight inconsistently.

**Fix:** weight by $1/N(h)$ in both, or adopt Cressie's WLS weights $N(h)/\gamma(h)^2$ if you move
to a variogram target.

### 2.6 Fitting a correlation model to Moran's I — **Grade B**

Fitting $\rho(h)$ to $\hat I(h)$ treats Moran's I as an estimator of the correlation function. It is
not — it carries the $n/S_0$ scaling and the $-1/(n-1)$ centring, and its lag-band construction is not
the same as a covariogram bin. The docstring is honest about "up to a multiplicative sill", and the
σ² parameter absorbs the scale, so this is defensible. But the cleaner target is the **empirical
covariogram/variogram** (unbiased for $C(h)$), or — best — direct maximum likelihood on the data via
a **Vecchia approximation** (Katzfuss & Guinness 2021), which is $O(n)$ and gives the joint posterior
over $(\ell,\nu,\sigma^2,\tau^2)$ without the two-stage binning loss.

---

## 3. Anisotropy

### 3.1 Theory

Geometric anisotropy: the correlation depends on displacement through an elliptical metric. With
principal axes at angle θ and inverse-ranges $\kappa_x,\kappa_y$,

$$\kappa_{\text{eff}}(\phi)=\sqrt{\kappa_x^2\cos^2(\phi-\theta)+\kappa_y^2\sin^2(\phi-\theta)}$$

so the level set $\{h:h\,\kappa_{\text{eff}}(\phi)=1\}$ is the ellipse with semi-axes $1/\kappa_x,1/\kappa_y$.

### 3.2 What SPARC does — **Grade A on the geometry**

`anisotropy.py:296-300` implements exactly this. `kernel_field.py:190-197` rotates and scales
correctly. The θ prior is flat on $(0,\pi)$ via a logit transform relying on the Jacobian — correct.
Circular statistics mod π for the θ posterior (`:152`, `:166`) — correct.

### 3.3 Label switching makes the κ posteriors meaningless — **Grade C**

$(\kappa_x,\kappa_y,\theta)$ and $(\kappa_y,\kappa_x,\theta+\pi/2)$ give an **identical likelihood**.
Nothing in the model breaks the symmetry: `kappa_x` and `kappa_y` get identical `LogNormal(log κ₀, 1)`
priors (`:317-318`) and identical inits (`:328-330`). The posterior is therefore bimodal, the chains
can land in either mode, and the pooled posterior means for $\kappa_x$, $\kappa_y$ are averages
*across modes* — i.e. both converge toward the same value, systematically **understating the
anisotropy ratio**. This will also inflate R̂ on those blocks, which is worth checking in the run logs
as a confirmation.

**Fix:** reparameterise to a symmetry-free basis:

$$\kappa_{\text{geo}}=\sqrt{\kappa_x\kappa_y}\ \ (\text{log-normal prior}),\qquad r=\kappa_x/\kappa_y\ge1\ \ (\text{log-normal, constrained}),\qquad \theta\in(0,\pi)$$

This is the standard fix and also gives directly interpretable outputs (a geometric mean range and
an anisotropy ratio) for the report.

### 3.4 Angular resolution — **Grade B**

`compute_directional_correlogram(..., n_angle_bins=4)` gives 45° bins. Fitting a 3-parameter ellipse
to 4 directions is identified but fragile. Use ≥8 bins with an explicit angular tolerance, and
require a minimum pair count per (lag, angle) cell before the bin enters the likelihood.

---

## 4. GWR and "MGWR"

### 4.1 Theory

GWR (Brunsdon, Fotheringham & Charlton 1996) solves a WLS problem per location $i$:

$$\hat{\boldsymbol\beta}(u_i,v_i)=\left(\mathbf{X}^\top \mathbf{W}(i)\mathbf{X}\right)^{-1}\mathbf{X}^\top \mathbf{W}(i)\mathbf{y}$$

with $\mathbf{W}(i)=\mathrm{diag}(w_{i1},\dots,w_{in})$ from a distance kernel. The fitted values are
$\hat{\mathbf y}=\mathbf{S}\mathbf y$ with $\mathbf{s}_i^\top=\mathbf{x}_i^\top(\mathbf{X}^\top\mathbf{W}(i)\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}(i)$.
This hat matrix is the basis for everything inferential:

- effective number of parameters $\mathrm{ENP}=\mathrm{tr}(\mathbf S)$,
- $\mathrm{AIC}_c = 2n\ln\hat\sigma + n\ln 2\pi + n\frac{n+\mathrm{tr}(\mathbf S)}{n-2-\mathrm{tr}(\mathbf S)}$ — the standard bandwidth criterion,
- local standard errors $\mathrm{Var}(\hat\beta_k(i))=\hat\sigma^2[\mathbf{C}(i)\mathbf{C}(i)^\top]_{kk}$,
- the Da Silva & Fotheringham (2016) multiple-testing correction for local t-tests.

**MGWR** (Fotheringham, Yang & Kang 2017) gives each coefficient surface its *own* bandwidth and fits
by **back-fitting**: iterate $k=1..p$, form the partial residual $\boldsymbol\varepsilon_{(k)}=\mathbf y-\sum_{l\ne k}\mathbf X_l\circ\hat{\boldsymbol\beta}_l$,
smooth it against $\mathbf X_k$ with $\mathbf W(b_k)$, re-select $b_k$, repeat to SOC-RSS convergence.

### 4.2 "MGWR" is single-bandwidth GWR — **Grade C**

`gwr.py:351` (and `:519`), inside the branch explicitly commented `# Use variable-specific bandwidths (MGWR)`:

```python
# Use the mean bandwidth for kernel weighting
bandwidth = np.mean(list(self.variable_bandwidths.values())) + 1e-10
```

One scalar bandwidth, one weight vector, one WLS solve. There is no back-fitting loop anywhere in the
repo. The per-predictor bandwidths that Stage 0 works hard to estimate are **averaged away**.

The anisotropic path (`:242`) has the same structure problem from the other direction: it computes
per-predictor Matérn weights and then collapses them with a **geometric mean** into a single scalar
weight per row (`:290`). That preserves the WLS structure, as the docstring says — which is precisely
why it cannot be MGWR. Multi-scale weighting is not expressible as a single $\mathbf W(i)$.

And `_matern_kernel` (`:186`) ignores its own `bandwidth` argument whenever a kernel field is attached,
substituting `np.mean([p.kappa for p in predictors])` — averaging κ across predictors a second time.

**Fix (this is the big one for Stage 2):** implement the Fotheringham–Yang–Kang back-fitting loop.
It is ~60 lines around the existing `_local_regression`:

```
initialise β̂ from single-bandwidth GWR
repeat until SOC-RSS < tol:
    for k in 1..p:
        ε_(k) = y − Σ_{l≠k} X_l ∘ β̂_l
        b_k   = argmin AICc( smooth(ε_(k) ~ X_k, W(b)) )     # golden-section
        β̂_k   = smooth(ε_(k) ~ X_k, W(b_k))
```

Track ENP per surface so the Stage-3 divergence audit compares like with like. If back-fitting is too
costly at production N, at minimum stop averaging the bandwidths — run separate single-bandwidth GWRs
per predictor group and report them as such, rather than labelling the mean-bandwidth fit "MGWR".

### 4.3 No hat matrix, no AICc, no local standard errors — **Grade C/D**

Grep for `AICc`, `trace`, `hat_matrix`, `enp` across `gwr.py`, `gwr_bandwidth.py`, `bandwidth_advisor.py`:
zero hits. Consequences:

- `tune_bandwidth` (`:568`) uses `sklearn.cross_val_score(..., cv=5)` — **random k-fold on spatially
  autocorrelated data**. This is the same leakage the pipeline elsewhere prevents with block CV, and it
  biases bandwidth selection *downward* (over-localised fits look better under leaky CV).
- The grid is `range(min_bw, max_bw, 50)` — a fixed step-50 sweep instead of the standard
  golden-section search on AICc.
- No local β standard errors means no local t-tests, no significance maps, and no honest input to the
  CATE-vs-GWR divergence audit, which currently compares a *point* GWR β against a *posterior* CATE.

**Fix:** accumulate $\mathrm{tr}(\mathbf S)$ and $\mathrm{tr}(\mathbf S^\top\mathbf S)$ during the fit
loop (each $\mathbf s_i$ is already available at the point of solve — it costs one extra dot product per
location), switch bandwidth selection to golden-section on AICc, and emit local SEs with the
Da Silva–Fotheringham corrected critical value.

### 4.4 Adaptive Gaussian kernel has a weight cliff — **Grade B**

`:372`: `bandwidth = neighbor_distances[-1]`, then Gaussian $\exp(-\tfrac12(d/b)^2)$. The k-th neighbour
therefore carries weight $e^{-0.5}\approx0.607$ and the (k+1)-th carries zero. That is a 0.607 → 0
discontinuity in the weight function, which propagates into a discontinuous β surface. This is why
adaptive-bandwidth GWR conventionally uses **bisquare** — it reaches exactly zero at the k-th neighbour.

**Fix:** use bisquare for adaptive bandwidths, Gaussian only for fixed bandwidths; or widen the
neighbour set to ~3b so the Gaussian tail is genuinely negligible at truncation.

### 4.5 Ridge and silent β = 0 — **Grade C**

`:404`: the ridge penalty `alpha * I` is added to $\mathbf X^\top\mathbf W\mathbf X$ where the weights
are **unnormalised** kernel values. The effective shrinkage therefore scales with $\sum_j w_{ij}$, which
varies by location and by bandwidth — so `alpha` means something different at every point, and changes
meaning when the bandwidth changes. Row-standardise the weights (so $\sum_j w_{ij}=n_{\text{local}}$)
before adding the penalty.

More seriously, `:452`:

```python
except Exception:
    # If anything fails, return safe defaults
    return np.zeros(n_features), np.mean(y) if len(y) > 0 else 0.0
```

A failed solve returns **β = 0 with no marker**. Downstream, "the local effect is zero" and "the solve
failed" are the same number. These β feed the Bayesian MGWR ensemble, the NUTS priors, and the
divergence audit. Return `NaN` plus a validity mask, and count failures in the fit diagnostics.

---

## 5. GWRF (geographically weighted random forest)

### 5.1 Theory

Georganos et al. (2019, 2021): fit a local RF at each location $i$ on the $k$ nearest observations,
weighting samples by a spatial kernel; predict at $i$ from the local model. Bandwidth (here $k$) is
selected by minimising out-of-bag or CV error.

### 5.2 The entire kernel-field path is dead code — **Grade D**

`gwrf.py:169`:

```python
from sparc.models.kernel_field import anisotropic_distance, matern_kernel_weights
except ImportError:
    _use_anisotropy = False
```

`anisotropic_distance` does **not exist at module level** — it is a `@staticmethod` on `KernelField`
(`kernel_field.py:179`). The import raises `ImportError` and the flag is silently cleared. Even if it
resolved, the guard above it tests `hasattr(self.kernel_field, "kernels")` — `KernelField` exposes
`.predictors`, not `.kernels` — so the branch is unreachable twice over. And the call inside it,
`matern_kernel_weights(d_local, nu=nu, sigma2=sigma2)`, would `TypeError`: the signature is
`(distances, kappa, nu=1.5)`; there is no `sigma2` parameter and `kappa` is never passed.

Net: **GWRF never consumes the KernelField.** `k_neighbors` is whatever the config says (default 100),
never derived from Stage 0. The `__init__` docstring's claim that "the per-pair effective range from
`kernel_field` is used to widen the neighbour-search at fit time" is not implemented, and the README's
"All four consume the `KernelField`" is inaccurate for GWRF.

### 5.3 Reproducibility — **Grade C**

`:141`: `np.random.choice(n_points, self.subsample_n, replace=False)` with **no seed**. `np.random.seed(42)`
appears only in the `subsample_fraction` branch (`:145`), but the auto-subsample path (`:131`, triggered
at n > 10 000 — i.e. the production path) uses the unseeded branch. `RandomForestRegressor(...)` at `:215`
has no `random_state`. Two independent runs on identical inputs give different local models, different
β surfaces, and different Stage 4 scenarios.

### 5.4 Uncertainty is model disagreement only — **Grade B**

`predict` returns the weighted std of the $k$ blended *local model means*. That is between-model
disagreement; it omits within-forest variance entirely and has no calibration guarantee. Use the
infinitesimal-jackknife variance (Wager, Hastie & Efron 2014) available from sklearn RF ensembles, or
route the output through the existing `evaluation/conformal.py` and report the conformalised interval
as the headline number.

### 5.5 Fix

1. Fix the import (`KernelField.anisotropic_distance`) and the `matern_kernel_weights` call signature,
   then add a test that asserts `_use_anisotropy is True` when an anisotropic field is attached — a
   dead branch that fails silently is worse than no branch.
2. Derive $k$ from the Stage-0 practical range: $k \approx \pi h_{0.05}^2 \cdot \hat\rho_{\text{points}}$.
3. Seed both the subsample RNG and every `RandomForestRegressor`.
4. Replace the adaptive Gaussian at `:210` with bisquare (same cliff issue as §4.4).

---

## 6. GGPGAM

### 6.1 Theory

The GGP-GAM of Comber, Harris & Brunsdon (2023, 2024) specifies spatially varying coefficients as
Gaussian-process splines over location:

$$y_i=\beta_0(u_i,v_i)+\sum_k \beta_k(u_i,v_i)\,x_{ik}+\varepsilon_i$$

fitted in `mgcv` as `y ~ s(u,v,bs="gp") + s(u,v,bs="gp",by=x_1) + ... + s(u,v,bs="gp",by=x_p)`.
Two things are load-bearing: the smooth is **bivariate over (u,v)**, and the `by=` construction makes
each term **linear in $x_k$** so that $\beta_k(u,v)$ is a readable coefficient surface. Smoothing
parameters are selected by **REML or GCV** per term (Wood 2011), and basis dimension is checked with
`k.check`.

### 6.2 The 2-D spatial basis is additive, not bivariate — **Grade C**

`ggpgam.py:123`:

```python
terms = s(0, n_splines=self.n_spatial_bases) + s(1, n_splines=self.n_spatial_bases)
```

That is $f(u)+g(v)$ — a ridge surface. It cannot represent *any* spatial interaction: no local maximum,
no hotspot, no diagonal gradient except as a sum of separable ridges. The 3-D elevation branch two lines
above **does** use `te(0,1,2)`, so the default path is structurally weaker than the optional one.

The covariate terms have the same problem: `te(0, i+2) + te(1, i+2)` (`:128-129`) is
$f(u,x_k)+g(v,x_k)$, again additive in the spatial dimensions.

### 6.3 The class is named `_SVC` but does not produce SVCs — **Grade C**

`te(0, i+2)` is a full tensor-product interaction between a coordinate and a covariate — it fits an
arbitrary nonlinear surface $f(u, x_k)$, not $\beta_k(u)\cdot x_k$. There is no way to extract a
coefficient surface from it, so the GGPGAM's "coefficients" are not comparable to GWR's β, and the
Stage-3 divergence audit and Stage-4 Tier-1 β sampling cannot use them on equal footing.

**Fix:** use pygam's `by`-style construction, or move this model to `mgcv` via `rpy2` where
`s(u,v,bs="gp",by=x_k)` is native. The minimal in-place fix is:

```python
terms = te(0, 1, n_splines=[self.n_spatial_bases]*2)          # bivariate spatial intercept
for i in range(X.shape[1]):
    terms += te(0, 1, i+2, n_splines=[nb, nb, 2])              # near-linear in x_k → SVC-like
```

with the covariate marginal held at low basis dimension so the term stays approximately linear in $x_k$.

### 6.4 No smoothing-parameter selection, and there is no GP basis — **Grade C / D**

- `LinearGAM(terms, lam=self.lam)` with a **single hand-set `lam=0.6`** across $2+3p$ terms. With
  $p=15$ that is 47 smooths sharing one penalty. pygam ships `gridsearch()`; use it, or better, select
  per-term λ by GCV/REML.
- pygam's `s()` is a **P-spline**, not a GP basis. The "Geographically Gaussian Process" name does not
  correspond to anything in the implementation, and `GGPGAMConfig` has **no `kernel_field` field at all**
  — so unlike GWR and (nominally) GWRF, GGPGAM has no path to consume the Stage-0 Matérn structure.
- No basis-dimension adequacy check (`k.check` equivalent: test residual correlation with the fitted
  smooth's neighbours).

### 6.5 Fabricated uncertainty on the fallback path — **Grade C**

`ggpgam.py:306`:

```python
std_errors = np.ones(len(predictions)) * np.std(predictions) * 0.1
```

When `prediction_intervals` throws, the model emits a **constant, invented** standard error equal to 10%
of the prediction spread, and it flows into `compute_uncertainty_weights` and the uncertainty maps with
no marker. Raise instead.

Also `predict` (`:238`): when the model was fitted with elevation (`_n_coord_dims_ == 3`) but `elevation`
is not passed at predict time, `coords_scaler.transform` receives 2 columns against a 3-column fit and
raises — worth an explicit guard with a clear message.

---

## 7. Bayesian MGWR ensemble

### 7.1 The reported β "posterior" omits the dominant variance component — **Grade C**

`bayesian_mgwr_ensemble.py` refits GWR $M$ times with κ resampled from the Stage-0 posterior and reports
the spread of $\hat\beta$ across refits as a posterior with an 89% HDI. By the law of total variance,

$$\mathrm{Var}(\beta\mid \text{data})=\underbrace{E_\kappa\!\left[\mathrm{Var}(\beta\mid\kappa,\text{data})\right]}_{\text{sampling variance of the WLS fit}}+\underbrace{\mathrm{Var}_\kappa\!\left[E(\beta\mid\kappa,\text{data})\right]}_{\text{what the ensemble computes}}$$

Only the second term is computed. The first — the ordinary WLS sampling variance
$\hat\sigma^2(\mathbf X^\top\mathbf W\mathbf X)^{-1}\mathbf X^\top\mathbf W^2\mathbf X(\mathbf X^\top\mathbf W\mathbf X)^{-1}$
— is typically the **larger** of the two and is dropped entirely. The result is a badly under-dispersed
"posterior".

This matters more than usual because that under-dispersed posterior is then used as the **prior for the
Stage-3 NUTS edge model**. An artificially tight prior does not just mislead Stage 2; it dominates the
Stage-3 likelihood and makes the edge posteriors — the pipeline's headline causal output — overconfident.
The under-coverage compounds across stages.

**Fix:** once §4.3 gives you the hat matrix, add the within-κ sampling variance:
draw $\beta^{(m)}\sim N(\hat\beta(\kappa_m),\widehat{\mathrm{Var}}(\hat\beta\mid\kappa_m))$ rather than
taking $\hat\beta(\kappa_m)$ itself. This is a ~5-line change and it is the difference between a
sensitivity band and a posterior.

### 7.2 Secondary

- κ draws are independent per predictor (`:87`), consistent with the marginal Stage-0 fits, but the
  resulting "joint" ignores cross-variable posterior dependence. Worth a note in the artifact.
- `bandwidth_to_outcome = 1/κ` — see §2.2; should be the practical range.
- κ is fitted to each variable's **own** autocorrelation, but `bandwidth_to_outcome` is semantically the
  **cross**-correlogram range to the outcome. Two different quantities are being conflated at this
  boundary; `cross_correlogram.py` computes the right one, so wire that through instead.

---

## 8. The NUTS engine

### 8.1 Theory

NUTS (Hoffman & Gelman 2014): sample $r\sim N(0,\mathbf M)$, simulate Hamiltonian dynamics with
$H=-\log p(\theta)+\tfrac12 r^\top\mathbf M^{-1}r$, double the trajectory until the no-U-turn condition
fires. Betancourt (2017) is the reference for the mass-matrix and diagnostic conventions.

Two conventions are load-bearing:

- **U-turn on velocity, not momentum.** The criterion is
  $(\theta^+-\theta^-)\cdot\mathbf M^{-1}r^\pm\ge0$. With a non-identity mass matrix, $r$ and
  $\mathbf M^{-1}r$ point in different directions.
- **$\mathbf M^{-1}\leftarrow\hat{\boldsymbol\Sigma}$.** Stan sets the *inverse* mass matrix to the
  estimated posterior covariance, so that $\theta$-steps $\varepsilon\mathbf M^{-1}r$ scale *with* the
  posterior sd in each direction.

### 8.2 The mass matrix is inverted — **Grade C**

`nuts.py:682`:

```python
var_est = np.var(warmup_arr, axis=0)
var_est = np.clip(var_est, 0.01, 1e6)
inv_mass_diag = torch.tensor(1.0 / var_est, ...)
```

This sets $\mathbf M^{-1}=\hat{\boldsymbol\Sigma}^{-1}$, i.e. $\mathbf M=\hat{\boldsymbol\Sigma}$ — the
reciprocal of the correct choice. Trace the consequence: `mass_diag = 1/inv_mass_diag = var`, so
$r\sim N(0,\hat\sigma^2)$, and the position update is $\theta \mathrel{+}= \varepsilon\,\hat\sigma^{-2}r$.
A direction with **large** posterior variance therefore receives a **small** position step. The
preconditioner is exactly backwards: it makes the sampler worse than identity mass precisely in the
anisotropic geometries adaptation exists to fix.

The variance floor of `0.01` is also arbitrary in a rescaled space and will clamp genuinely tight
directions.

**Fix:** `inv_mass_diag = torch.tensor(var_est)`. Validate against a known-covariance multivariate
normal: sample a $N(0,\mathrm{diag}(1,100))$ target and confirm ESS/gradient improves rather than
degrades relative to identity mass.

### 8.3 The U-turn criterion uses momentum, not velocity — **Grade C**

`nuts.py:315` and `:421`:

```python
delta = theta_plus - theta_minus
s = s_prime and (float(delta @ r_minus) >= 0) and (float(delta @ r_plus) >= 0)
```

Correct once the mass matrix is identity; wrong the moment adaptation kicks in. Trajectories terminate
at the wrong points, biasing the sampler.

**Fix:** `delta @ (inv_mass_diag * r_minus)` and likewise for `r_plus`.

### 8.4 Step-size adaptation is decoupled from the actual transition — **Grade C**

`_dual_average_step_size` (`:432`) runs its **own single-leapfrog-step Metropolis chain** and tunes ε to
the acceptance rate of *one* leapfrog step. Hoffman & Gelman Algorithm 6 adapts ε to the NUTS
trajectory's mean acceptance statistic $\bar\alpha$, averaged over all $2^J$ leaves. Energy error
accumulates across a trajectory, so a single-step acceptance of 0.8 corresponds to a much lower
depth-8 acceptance. The tuned ε is therefore systematically **too large** for the transitions it will
actually be used for → more divergences, biased posteriors, especially in the funnel-like geometries
these hierarchical models produce.

`_nuts_step` already returns `mean_accept` computed correctly (`:424`). Move the dual-averaging recursion
inside the warmup transition loop and feed it that value.

### 8.5 Divergences are under-reported — **Grade C**

`:416`:

```python
if not s_prime and n_alpha_prime > 0 and (alpha_prime / n_alpha_prime) < 1e-10:
    divergent = True
```

`s_prime` goes False for *either* a U-turn or an energy blow-up, and the extra `< 1e-10` condition
suppresses most genuine divergences. Divergence count is the single most important HMC diagnostic —
a run with silent divergences looks converged and is not. Flag divergence exactly where the base case
detects it: `H' < log_u − Δmax`.

### 8.6 Convergence diagnostics are pre-2021 — **Grade B**

- `_split_r_hat` (`:488`) is applied to the **pooled** chain in `correlogram_matern_fit.py:327`
  (`np.concatenate` across chains), so for 2 chains it accidentally compares chain 1 vs chain 2 and for
  >2 chains it is meaningless. Proper split-R̂ splits *each* chain, giving $2m$ halves.
- No rank-normalisation, no folded-R̂, no tail-ESS. Vehtari et al. (2021) is the current standard and
  recommends a threshold of **1.01**, not the 1.05 used at `correlogram_matern_fit.py:365`.
- `_ess_bulk` (`:503`) uses a single-chain Geyer truncation without the between-chain variance term.

**Fix:** these are all `arviz.rhat(..., method="rank")` / `arviz.ess(..., method="tail")` one-liners once
draws are shaped `(chain, draw)`. Export the trace as an `InferenceData` and get the whole diagnostic
suite for free.

### 8.7 What is right — **Grade A**

FindReasonableEpsilon (`:126`) matches Algorithm 4. The tree recursion, slice criterion, and
biased-progressive acceptance $\min(1,n'/n)$ match Algorithm 3. $\Delta_{\max}=1000$ is the published
value. Jacobians for `log` and `logit` transforms are handled correctly at `:614`.

---

## 9. The PDE / physics loss

### 9.1 Theory

A PINN residual loss (Raissi, Perdikaris & Karniadakis 2019) penalises
$\mathcal{L}_{\text{phys}}=\frac1N\sum_i\|\mathcal{N}[\hat u](x_i)\|^2$. The literature is unambiguous
that **term weighting is the hard part**: Wang, Teng & Perdikaris (2021) show gradient pathologies from
badly scaled terms; Wang, Sankaran & Perdikaris (2022) give the NTK-based weighting; McClenny &
Braga-Neto (2023) give self-adaptive weights. The standard prescription is to **non-dimensionalise the
PDE** so the residual is $O(1)$ by construction, then apply either fixed or gradient-norm-balanced
weights computed from **detached** statistics.

### 9.2 The residual normalisation makes the physics loss scale-invariant — **Grade C, highest impact**

`pde_operators.py:519`:

```python
scale = r.detach().std() if detach_scale else r.std()
return residual / scale.clamp(min=1e-6)
```

`pde_loss.py:105` calls this with **`detach_scale=False`** for every V3 term. So each term is

$$\mathcal L=\frac{1}{N}\sum_i\left(\frac{r_i}{\mathrm{sd}(r)}\right)^2=\frac{\overline{r^2}}{\mathrm{sd}(r)^2}=1+\frac{\bar r^{\,2}}{\mathrm{sd}(r)^2}$$

which is **exactly invariant** under $r\mapsto cr$ for any $c>0$. The network can make the physics
residual arbitrarily large at no cost, provided its shape is unchanged. What the loss actually penalises
is the ratio of the residual's mean to its spread — it pushes the residual toward *zero-mean*, not
toward *zero*.

Consequences beyond the obvious:

- Every logged per-term loss sits at ≈1 regardless of how well physics is satisfied, so the training
  logs cannot distinguish a physically consistent model from an inconsistent one.
- The relative weights (`heat_diffusion: 1.0` … `gaussian_curv: 0.05`) are being applied to quantities
  all pinned at ≈1, so they no longer express relative importance of *physics*, only of *shape*.
- The `detach_scale=True` (V2) path is less broken — gradients do push $r\to0$ within a step — but since
  σ is recomputed each batch, the effective gradient magnitude is held constant as $r$ shrinks, which is
  an implicit adaptive-scale scheme nobody chose.

**Fix, in order:**

1. **Non-dimensionalise.** Pick characteristic scales $T_c$ (outcome sd), $L_c$ (Stage-0 practical range),
   $\alpha_c$ (prior mean α). Write the residual in $\tilde T=T/T_c$, $\tilde x=x/L_c$. Then
   $\tilde r=\tilde\alpha\tilde\nabla^2\tilde T-\tilde S$ is $O(1)$ by construction, with no normalisation.
2. Replace `_normalize_residual` with a **fixed** scale computed once from the training set, or an EMA
   updated with `torch.no_grad()`. Never divide by the current batch's own std, and never let gradient
   flow through the divisor.
3. If adaptive weighting is wanted, use the gradient-norm rule
   $\lambda_i \leftarrow (1-\eta)\lambda_i+\eta\,\|\nabla_\theta\mathcal L_{\text{data}}\|/\|\nabla_\theta\mathcal L_i\|$,
   all statistics detached.
4. Log the **raw, unnormalised** residual RMS alongside the weighted loss so the physics is auditable.

### 9.3 Term-by-term audit

| # | Term | Code | Verdict |
|---|---|---|---|
| 1 | heat diffusion $\alpha\nabla^2T-S$ | `pde_loss.py:212` | **B** — the only genuine PDE residual, but neutered by §9.2 and dimensionally mixed (α normalised to O(1), $\nabla^2T$ in K/m², $S$ in unknown units) |
| 2 | energy balance $Q^*-Q_H-Q_E$ | — | **D — does not exist** |
| 3 | directional | `:233` | **C — identically zero** |
| 4 | anisotropy | `:254` | **C — inverted vs. its docstring; never touches Stage 0** |
| 5 | gradient flux | `:273` | **C — mislabelled; it is a flatness prior** |
| 6 | Gaussian curvature | `:288` | **B — an honest smoothness regulariser, described as physics** |
| 7 | α smoothness | ✓ | **A** |
| 8 | α prior | ✓ | **A** |

**Term 2 is absent.** `energy_balance` appears in the module docstring (`pde_loss.py:6`, `:17`) and in
the README, but it has no entry in `PDELossWeights`, no entry in `_ACTIVATION_SCHEDULE`, and no
computation block. `sparc/physics/energy_balance.py` defines `energy_balance_residual` — and grep
confirms **nothing in the repository imports or calls it.** The headline surface-energy-balance
constraint does not participate in training.

**Term 3 is algebraically zero.** The 5-point Laplacian is
$\nabla^2f=(f_E+f_W-2f)/h^2+(f_N+f_S-2f)/h^2$ (`pde_operators.py:132`), and `directional_curvatures`
(`:141`) returns exactly those two summands. So

```python
dir_residual = d2_dx2_full + d2_dy2_full - lap_T_full   # ≡ 0
```

up to float error, over the identical validity mask. `_normalize_residual` then divides ~0 by
`std(~0).clamp(min=1e-6)`, **amplifying pure floating-point noise to O(1)** and adding it to the loss.
Term 3 contributes noise, not physics.

**Term 4 contradicts its own docstring.** The docstring says "penalize spurious *isotropy* where data is
anisotropic". The code (`:254`) is `|∂²T/∂x² − ∂²T/∂y²|²` — which penalises *anisotropy*, forcing the
curvature to be isotropic. It is the opposite of the stated intent, and it never reads
$(\kappa_x,\kappa_y,\theta)$ from Stage 0. The README's "anisotropy alignment with Stage 0" is not
implemented. The correct term is a residual against the measured ellipse, e.g.
$\left\|\nabla^2_{\text{aniso}}T\right\|^2$ with
$\nabla^2_{\text{aniso}}=\kappa_x^2\partial^2_{u}+\kappa_y^2\partial^2_{v}$ in the rotated frame.

**Term 5 is not Fourier's law.** Fourier's law $\mathbf q=-k\nabla T$ is a *definition*; there is no
residual to penalise. `flux = alpha_norm * grad_mag_full` followed by `mean(flux²)` penalises the
**magnitude of the heat flux itself**, driving $|\nabla T|\to0$ — i.e. pushing the temperature field
toward spatially constant, in direct opposition to the data-fit term. A genuine flux constraint would be
$\nabla\cdot\mathbf q - S = 0$ (which is term 1) or a boundary flux condition.

### 9.4 The energy-balance module's physics — **Grade C** (relevant when you wire it up)

- `sensible_heat_flux = -k·∇²T·d` (`energy_balance.py:88`). Dimensionally this is W/m², but physically it
  is the *convergence of lateral conduction* — negligible in soil and air — not the turbulent sensible
  heat flux $Q_H=\rho c_p C_H U(T_s-T_a)$. Off by orders of magnitude and by mechanism.
- `latent_heat_flux` claims Priestley–Taylor but returns $\alpha_{PT}/\alpha_{PT,\text{wet}}\in[0,1]$ with
  **no $\Delta/(\Delta+\gamma)$ term at all** — the psychrometric slope and constant, which are the whole
  content of PT, are absent. It is a linear canopy-fraction rescaling under a borrowed name.
- `storage_flux = ρ·c·d·T` has units J/m² — an energy per area, not a flux. The correct form is
  $\Delta Q_S=\rho c\, d\,\partial T/\partial t$.
- `solar_Wm2=800`, `T_sky_K=260` are global constants — no sky-view factor, no shading, no diurnal or
  seasonal variation. For UHI work, SVF and shading are first-order controls on $Q^*$.
- The $\varepsilon\sigma T^4$ terms require **Kelvin**; nothing validates the unit. Passing °C is off by
  ~$10^{10}$ and would fail silently.

**Fix:** either implement the standard urban surface energy balance properly (SVF-weighted $Q^*$,
aerodynamic-resistance $Q_H$, full PT or Penman–Monteith $Q_E$, OHM storage — Grimmond & Oke 1999,
2002), with a unit-checked interface; or delete the module and remove the claim. The current state —
a dimensionally inconsistent module that is never called but is advertised — is the worst of both.

---

## 10. PI-JEPA

### 10.1 Theory

I-JEPA / V-JEPA (Assran et al. 2023; Bardes et al. 2024): an online encoder maps a *context* view to
$h_c$; a predictor maps $h_c$ (optionally conditioned on an action or position) to $\hat h_t$; an
**EMA copy with stop-gradient** maps the full view to $h_t$; the loss is $\|\hat h_t-h_t\|$.
VICReg (Bardes, Ponce & LeCun 2022) prevents collapse via
$\mathcal L=\lambda s(Z,Z')+\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')]$ with $\lambda=\mu=25,\nu=1$.
V-JEPA-2-AC adds **action conditioning**, which is the point of the "AC".

### 10.2 What is right — **Grade A**

`ema_trunk.py:141,173` — `@torch.no_grad()` on both `update` and `encode_target`; in-place
$p_t\leftarrow\tau p_t+(1-\tau)p_o$ with cosine τ warmup. Correct.
`jepa_loss.py:75` — off-diagonal covariance Frobenius scaled by $1/D$: matches VICReg.
`variance_loss` hinge $\mathrm{ReLU}(\gamma-\mathrm{sd})$: matches.

### 10.3 The action conditioning is inert by default — **Grade D**

The pretraining loop at `v2_neural_training.py:2869` calls `_pt_latent_predictor(h_context)` — **no
action argument**. `LatentPredictor.forward(context, action=None)` then skips the FiLM modulation
entirely (`latent_predictor.py:83`). `ActionEmbedding` is trained only when `lambda_jepa_scenario > 0`,
whose default is `0.0` (`joint_loss.py:105`).

But `inference/latent_rollout.py:105` *does* call `predictor(h_state, action_embed(...))` — applying
FiLM $\big((1+\gamma)u+\beta\big)$ with $\gamma,\beta$ from an `ActionEmbedding` that, on the default
path, was never trained. Under the default configuration the "action-conditioned" rollout applies a
near-random affine modulation to the latent.

**Fix:** either pass a real action during pretraining (the treatment delta is available), or gate
`latent_rollout` on a trained-action-embedding flag and fail loudly otherwise. Do not silently apply an
untrained FiLM layer.

### 10.4 A randomly-initialised projection is injected into the target — **Grade C**

`v2_neural_training.py:2878-2879`, inside `with torch.no_grad()`:

```python
if _pt_beta_proj is not None and _pt_beta_t is not None:
    h_target = h_target + _pt_beta_proj(_pt_beta_t[_pt_b_idx])
```

Because it is inside `no_grad`, `_pt_beta_proj` **never receives a gradient** — it stays at its random
initialisation for the whole run. So the JEPA target is the EMA embedding plus a *fixed random linear
projection of the β map*: structured noise added to the learning signal, with no mechanism to become
useful. Either train it in a branch that gets gradients (as an auxiliary head, not on the target), or
use a fixed principled encoder, or remove it.

### 10.5 Symmetric VICReg on an asymmetric objective — **Grade B**

`jepa_loss.py:138-146` computes `var_t` and `cov_t` on `h_target`. Since the target is detached, these
are gradient-free constants — wasted compute and, more importantly, **misleading logs**: the reported
`jepa_variance` is the average of a term that is being optimised and one that is not. VICReg is
symmetric because both its branches are trainable; JEPA is not. Regularise the online branch only, and
log `var_t`/`cov_t` separately as *diagnostics of target collapse* (which is genuinely useful).

Also: `alignment_loss` uses cosine distance, whereas I-JEPA/V-JEPA use L1/L2 in latent space. Cosine
discards magnitude, which the variance hinge then has to police separately. Defensible, but it is a
deviation from the cited references, and the weight ratios (`alignment` vs `variance=1.0` vs
`covariance=0.04`) bear no relation to VICReg's published 25:25:1.

---

## 11. The neural meta-learner

### 11.1 What is right — **Grade A**

SIREN initialisation (`pde_encoder.py:46-49`) matches Sitzmann et al. 2020 exactly: first layer
$U(-1/n,1/n)$, hidden $U(-\sqrt{6/n}/\omega_0,\sqrt{6/n}/\omega_0)$, activation $\sin(\omega_0 Wx+b)$.
The zero-initialised `blend_proj` and `gate_residual_weight` (`neural_meta.py:189-194`) are a clean
residual-gating pattern — the model starts as the backward-compatible baseline and learns its way out.
Sparse KNN attention at $O(N\cdot k)$ is the right structure for this problem.

One note: $\omega_0=30$ is calibrated for inputs on $[-1,1]$. If `physics_feats` are standardised to
mean 0 / sd 1, roughly 32% of inputs fall outside $[-1,1]$ and the effective frequency content is
higher than intended. Min-max the physics features, or tune $\omega_0$.

### 11.2 Exceedance heads are incoherent with the regression head — **Grade C**

`neural_meta.py:228` builds one independent sigmoid classifier per threshold, trained with
`F.binary_cross_entropy` against `(y_true > thresh)` (`loss.py:305`). Nothing ties them to each other or
to `T_pred`. Two failures follow directly:

- **Non-monotone in τ.** Nothing enforces $P(T>\tau_1)\ge P(T>\tau_2)$ for $\tau_1<\tau_2$. The heads can
  and will cross — the quantile-crossing problem, familiar from quantile regression.
- **Incoherent with the point prediction.** The model can output $\hat T=32^\circ$ and $P(T>30)=0.2$.

**Fix:** derive exceedance from a single predictive distribution. Either (a) add a heteroscedastic head
$(\mu(x),\log\sigma^2(x))$ trained by Gaussian NLL and compute $P(T>\tau)=1-\Phi((\tau-\mu)/\sigma)$ —
monotone and coherent by construction; or (b) keep the classifiers but parameterise them cumulatively,
$P(T>\tau_j)=\prod_{l\le j}\sigma(z_l)$, which enforces monotonicity structurally. Option (a) also gives
you the aleatoric variance you need in §11.3.

Minor: `Sigmoid` inside the module plus `binary_cross_entropy` is less stable than logits plus
`binary_cross_entropy_with_logits`.

### 11.3 MC-Dropout gives epistemic variance only — **Grade B**

`predict_with_uncertainty` (`:435`) returns `predictions.std(0)`. Gal & Ghahramani (2016) show MC dropout
approximates a deep-GP posterior only under a specific relation between dropout rate $p$, weight decay
$\lambda$, and observation noise, $\tau=\frac{p\ell^2}{2N\lambda}$, and the **predictive** variance is

$$\mathrm{Var}[y^*]\approx\tau^{-1}+\frac1S\sum_s\hat y_s^{*2}-\left(\frac1S\sum_s\hat y_s^*\right)^2$$

The $\tau^{-1}$ aleatoric term is omitted. Separately, dropout appears only in `base_enc`, `fusion`,
`regression_head`, and the exceedance heads — **not** in `physics_enc`, `alpha_emb`, or `trunk_fusion`.
So the MC variance measures the head's epistemic uncertainty and nothing about the trunk's.

The docstring correctly says "epistemic", which is honest — but any downstream consumer treating this as
a predictive interval will be under-covered on both counts.

**Fix:** add the aleatoric head from §11.2 and report $\sqrt{\sigma^2_{\text{alea}}+\sigma^2_{\text{MC}}}$;
then run the result through `evaluation/conformal.py` and publish the **conformalised** interval as the
headline. Report empirical coverage on a spatially held-out fold — that number is the one that matters.

---

## 12. Causal inference — identification

### 12.1 Theory

Pearl's back-door criterion: $Z$ is admissible for $P(y\mid do(x))$ if (i) no node in $Z$ is a descendant
of $X$, and (ii) $Z$ d-separates $X$ from $Y$ in $G_{\underline X}$ (the graph with $X$'s outgoing edges
removed). Then $P(y\mid do(x))=\sum_z P(y\mid x,z)P(z)$.

### 12.2 What is right — **Grade A**

`dag_definition.py:401-421` builds the mutilated graph, restricts candidates to non-descendants of the
treatment, and tests d-separation with `nx.is_d_separator`. That is a faithful implementation of the
criterion — better than the pattern (parents-of-treatment) that most pipelines ship.

### 12.3 Non-identification is handled by two stacked silent fallbacks — **Grade C**

`dag_definition.py:425-428`:

```python
if not nx.is_d_separator(G_mut, {treatment}, {outcome}, set(candidates)):
    # Backdoor criterion not satisfiable — unmeasured confounding.
    # Fall back to the full candidate set (best effort).
    return candidates
```

and `causal_validation.py:760`, `:1893`:

```python
identified = model.identify_effect(proceed_when_unidentifiable=True)
```

When the effect is **not identified from the DAG**, the pipeline returns an adjustment set anyway,
tells DoWhy to proceed anyway, and emits an "ATE" with refutation results and an E-value attached — with
no flag distinguishing it from an identified effect. Identifiability is the one thing a causal DAG is
*for*; disabling the check globally removes the pipeline's central guarantee.

A related leak: if the DAG declares latent/unobserved nodes, they enter `candidates` and can make the
d-separation test pass, but `causal_validation.py:1968-1969` then filters
`if n in data_sub.columns` — silently dropping exactly the variables that made the effect identified.

**Fix:**
1. Return `None` (not "best effort") when the criterion is unsatisfiable, and propagate an
   `identified: false` flag into every artifact and report row for that edge.
2. Set `proceed_when_unidentifiable=False`; catch the exception and record it as a first-class result
   ("this edge is not identified under your DAG"), which is a genuinely useful finding for the user.
3. Restrict `candidates` to observed variables *before* the d-separation test, so the test answers the
   question that matters: is this effect identified **from the data you have**.
4. Consider the **optimal** adjustment set (Henckel, Perković & Maathuis 2022) rather than an arbitrary
   greedy-minimal one — it is asymptotically the lowest-variance valid set.

---

## 13. MC³ DAG search and the BGe score

### 13.1 Theory

BGe (Geiger & Heckerman 2002, corrected by Kuipers, Moffa & Heckerman 2014). For a set of $m$ columns
with scatter $S_N$, prior precision $T_0=\frac{\alpha_\mu(\alpha_w-m-1)}{\alpha_\mu+1}I$, and
$T_N=T_0+S_N+\frac{\alpha_\mu n}{\alpha_\mu+n}(\bar x-\nu)(\bar x-\nu)^\top$:

$$\log p(D)= -\frac{nm}{2}\log\pi+\frac m2\log\frac{\alpha_\mu}{\alpha_\mu+n}+\sum_{i=1}^{m}\left[\log\Gamma\!\left(\tfrac{\alpha_w+n-m+i}{2}\right)-\log\Gamma\!\left(\tfrac{\alpha_w-m+i}{2}\right)\right]+\frac{\alpha_w-m+1}{2}\log|T_0|-\frac{\alpha_w+n-m+1}{2}\log|T_N|$$

MC³ (Madigan & York 1995) runs Metropolis over graph space; Metropolis-coupling adds $K$ chains at
inverse temperatures $\beta_k$ with swap acceptance $\min\{1,\exp[(\beta_i-\beta_j)(S_j-S_i)]\}$.

### 13.2 The MC³ machinery is right — **Grade A**

`mc3.py:771` — $\log\alpha=\beta_k(S'-S)+\log H$ with a correct Hastings term for the asymmetric
add/remove pools (`:600-613`, citing Green 1995). `:785` — the swap acceptance matches the formula above.
`:844` — edge probabilities accumulate from the **cold chain only**, which is correct. Swap-rate
diagnostics with a warning below 0.10 (`:891`). This is a properly built sampler.

### 13.3 The BGe score has two errors, both in the complexity penalty — **Grade C**

`mc3.py:137` (fast path, the one that runs) and `:290` (slow path):

```python
0.5 * m * (math.lgamma((alpha_n - m + 1.0) / 2.0) - math.lgamma((alpha_w - m + 1.0) / 2.0))
...
+ 0.5 * self.alpha_w * logdet_0 - 0.5 * alpha_n * logdet_n
```

versus the formula above:

1. The gamma term should be a **sum over $i=1..m$** — a multivariate gamma ratio. The code takes $m$
   copies of the $i=1$ term only. The two agree at $m=1$ and diverge as $m$ grows.
2. The determinant exponents should be $\frac{\alpha_w-m+1}{2}$ and $\frac{\alpha_w+n-m+1}{2}$. The code
   uses $\frac{\alpha_w}{2}$ and $\frac{\alpha_w+n}{2}$ — dropping the $-m+1$ shift in both.

Both errors are functions of $m$, which is exactly the parent-set size. So the error is concentrated in
the **complexity penalty** — the part of the score that decides whether an edge is worth adding. Edge
inclusion probabilities, the median-probability DAG, and the Bayes factors are all affected, and the
direction of the bias varies with $m$ and $n$.

A third, subtler issue: $T_N$ uses $\bar x\bar x^\top$, implying prior mean $\nu=0$, but the data are not
centred (`local_score` slices the raw `_col_means_t`). BGe should be invariant to location shifts given
the prior mean; here a variable measured in Kelvin vs °C gets a different score. **Standardise the data
before scoring.**

**Fix:** port the corrected Kuipers–Moffa–Heckerman formula (their published R/Python reference, or
`bnlearn`'s `bge` score) and add a unit test comparing against a known-good implementation on a small
fixture. This is a self-contained ~20-line change with a clean oracle.

### 13.4 Prior, convergence, and Markov equivalence — **Grade B**

- **No structural prior.** A uniform prior over DAGs is *not* uniform over edges — it concentrates on
  intermediate-density graphs. Add an edge-count prior $p(G)\propto\rho^{|E|}$, and — more valuable here —
  an **informative prior around the expert DAG**, $p(G)\propto\exp(-\lambda\,d(G,G_{\text{expert}}))$.
  The pipeline already asks the user for a DAG; using it only as a comparison target rather than as a
  prior leaves information on the table.
- **The convergence test is not a convergence test.** `:817` stops when the max change in edge-inclusion
  probability across successive windows drops below tolerance. A chain stuck in a metastable mode
  satisfies that criterion perfectly. Run ≥4 independent MC³ chains from dispersed inits and compute R̂
  on the edge-inclusion indicators.
- **Markov equivalence.** BGe is score-equivalent: all DAGs in an equivalence class score identically.
  Reporting *directed* edge-inclusion probabilities from a DAG-space sampler therefore conflates "the
  data support this adjacency" with "the sampler happened to orient it this way." Report **CPDAG/essential
  graph** features, or switch to order-MCMC / partition-MCMC (`causal/order_mcmc.py` already exists —
  wire it in). At minimum, report adjacency probability and orientation probability as separate columns.
- Edge-reversal moves mix poorly in structure MCMC; the Grzegorczyk & Husmeier (2008) new-edge-reversal
  move is the standard remedy.

---

## 14. DML and spatial CATE

### 14.1 Theory

DML (Chernozhukov et al. 2018): in the partially linear model $Y=\theta_0 T+g_0(X)+U$,
$T=m_0(X)+V$, the Robinson-transformed score
$\psi=(Y-\hat g(X))-\theta(T-\hat m(X))$ is Neyman-orthogonal, so first-stage errors enter at second
order. **The guarantee requires cross-fitting on folds that are independent of the evaluation points.**

### 14.2 Cross-fitting uses random folds on spatially dependent data — **Grade C, highest causal impact**

`spatial_cate.py:742`:

```python
kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
```

and `:224`: `CausalForestDML(..., cv=3)` — econml's default splitter, also random.

With spatial autocorrelation, a random fold's training points are the *immediate neighbours* of its test
points. The nuisance models $\hat g$ and $\hat m$ then interpolate rather than predict, over-fitting the
local surface. Both residuals are over-shrunk, $\hat V=T-\hat m(X)$ loses genuine treatment variation,
and $\hat\theta$ is attenuated while its standard error collapses. Independence of the nuisance and
evaluation samples is precisely the condition orthogonality is traded against — break it and the
asymptotics do not hold.

This is a **direct internal contradiction**: Stage 2 goes to real trouble to build spatially blocked,
buffered folds (`spatial_fold_factory.py:184`) specifically to prevent this, and Stage 3 then discards
them.

**Fix:** pass the Stage-0/Stage-2 spatial block assignment as groups and use `GroupKFold` (or a custom
buffered splitter) for both the manual cross-fitting loop and `CausalForestDML(cv=...)`. econml accepts
any splitter object, so this is a one-parameter change once the block labels are threaded through.
Roberts et al. (2017, *Ecography*) is the canonical reference for why; for DML under dependence, the
clustered-cross-fitting variant in the Chernozhukov et al. line is the right framing.

### 14.3 Other CATE issues

- **`model.effect(X, T0=0, T1=1)`** (`:228`) hardcodes a 0→1 contrast. For continuous treatments (canopy
  fraction, impervious %) this is a full-range extrapolation, often well outside the observed support.
  The contrast should come from config, and should be checked against the treatment's empirical range.
  — **Grade C**
- **Fabricated CIs** (`:237`, `:291`): when `effect_inference` throws, the code substitutes
  `cate ± 1.96*np.std(cate)` — the *cross-sectional spread of point estimates* used as a standard error,
  identical at every location. Raise instead. — **Grade C**
- **Bayesian CATE treats estimated residuals as data.** `BayesianSpatialCATE` plugs cross-fitted
  residuals into a Bayesian likelihood (`:787`) with no allowance for nuisance-estimation error. DML's
  orthogonality delivers valid *frequentist* CIs; it does not make the plug-in Bayesian posterior valid.
  The credible bands will be too narrow. Either propagate nuisance uncertainty (sample nuisance fits) or
  label these as approximate. — **Grade B**
- **RFF construction is correct** (`:641-644`): $\Phi=\sqrt{2/K}\cos(\omega^\top x+b)$, $\omega\sim N(0,\ell^{-2})$,
  $w\sim N(0,1)$ — a unit-variance RBF GP prior independent of $K$. Rahimi & Recht 2007, done right. — **Grade A**
  But the kernel is **RBF (ν→∞)** while everything else in the pipeline is Matérn with $\nu\le2.5$. The
  Stage-0 ν is discarded here. RFF for Matérn is a one-line change: draw $\omega$ from a multivariate
  Student-$t_{2\nu}$ instead of a Gaussian.
- **`_normalize_coords`** (`:647`) min-max scales x and y **independently**, distorting the aspect ratio,
  so the isotropic RBF becomes anisotropic in real space by an arbitrary factor. Scale both axes by a
  single factor. The `clip(0.02, 1.0)` on `eff_lengthscale` (`:727`) also silently overrides genuinely
  short correlation ranges. — **Grade B**
- **No positivity/overlap gate.** `causal/overlap.py` implements a proper generalised-propensity overlap
  diagnostic for continuous treatments — but it is a diagnostic, not a gate. CATE estimation should
  refuse (or flag) cells below an overlap threshold. — **Grade B**

---

## 15. Refutations and sensitivity analysis

### 15.1 Refutations test a model nobody uses — **Grade C**

`causal_validation.py:1894` runs all four DoWhy refuters against
`method_name="backdoor.linear_regression"`. But the numbers that reach the report and Stage 4 come from
`CausalForestDML` and the NUTS edge posteriors. The robustness certificate therefore applies to a
different estimator than the estimate it is attached to.

### 15.2 Pass/fail thresholds are arbitrary, and the null distribution is discarded — **Grade C**

```python
placebo_pass = abs(placebo.new_effect) < abs(ate_val) * 0.5      # :1909
rcc_pass     = abs(rcc.new_effect - ate_val) < abs(ate_val) * 0.2 # :1920
subset_pass  = abs(subset_ref.new_effect - ate_val) < abs(ate_val) * 0.3
ucc_pass     = abs(ucc.new_effect - ate_val) < abs(ate_val) * 0.25
```

DoWhy's `RefutationResult` carries a **p-value** computed from the simulated null; the code uses only
`new_effect` (the mean) and compares it against hand-picked percentage bands. The placebo test in
particular has a natural statistical form — "is the placebo effect distinguishable from zero, given the
simulation distribution" — which is discarded here.

Also: `data_subset_refuter(num_simulations=5)` — five draws cannot characterise stability.

And `add_unobserved_common_cause` is run at a **single, negligible** strength
(`effect_strength_on_treatment=0.01`, `on_outcome=0.02`). "PASS" therefore means only "a nearly-zero
confounder does not change the answer", which is close to vacuous. The informative version sweeps a grid
and reports the contour at which the effect changes sign — DoWhy supports array-valued effect strengths
for exactly this.

### 15.3 The Cinelli–Hazlett robustness value is the wrong formula — **Grade C**

`sensitivity.py:405`:

```python
return (f2_t - f2_c) / (f2_t - f2_c + df)
```

That expression is the **partial $R^2$ of the treatment**, $R^2_{Y\sim D\mid X}=t^2/(t^2+\mathrm{df})$.
The robustness value (Cinelli & Hazlett 2020, JRSS-B, Prop. 1) is

$$RV_q=\frac12\left(\sqrt{f_q^4+4f_q^2}-f_q^2\right),\qquad f_q=\frac{q\,|t|}{\sqrt{\mathrm{df}}}$$

Worked example, $t=4$, $\mathrm{df}=100$: the code returns $16/116=0.138$; the correct $RV_1$ is
$f_1=0.4$, $f_1^2=0.16$, $RV=\tfrac12(\sqrt{0.0256+0.64}-0.16)=0.328$. The code **understates robustness
by ~2.4×**, so genuinely robust findings are reported as fragile.

### 15.4 E-values are computed on unstandardised effects — **Grade C**

`sensitivity.py:50` uses VanderWeele & Ding's $RR\approx\exp(0.91\,d)$ — where $d$ is **Cohen's $d$**, a
standardised mean difference. `annotate_causal_payload` (`:159`) walks the payload and calls
`e_value_continuous(float(v))` on raw effect values with no standardisation step. An effect of
3.5 °C per unit canopy fraction becomes $\exp(0.91\times3.5)=24.3$ and an E-value of ~48 — a number with
no meaning.

**Fix:** require an explicit outcome SD at the call site and divide by it; refuse to compute when it is
unavailable. Also note in the report that the $d\to RR$ bridge is an approximation validated for binary
outcomes and should be read as an order-of-magnitude sensitivity statement.

### 15.5 What is right — **Grade A**

`_e_value_from_rr` (`:54`) — $E=RR+\sqrt{RR(RR-1)}$ with the $RR<1$ inversion: correct.
`causal/interference.py` — anonymous-exposure spillover with HAC variance, following the
Hu–Li–Wager treatment: a genuinely well-founded module.
`causal/overlap.py` — generalised-propensity overlap for continuous treatments: the right generalisation
of positivity.

---

## 16. Decision layer, UQ, and cross-cutting concerns

### 16.1 Gini of the allocation is the wrong equity metric — **Grade C (conceptual)**

`scenario/budget.py:100` computes the Gini coefficient correctly —
$G=\frac{n+1-2\sum_i \mathrm{cum}_i/\sum x}{n}$ is algebraically the standard formula. **Grade A on the arithmetic.**

The problem is what it is applied to. `_gini(allocation)` measures inequality of the **spend**. But an
equitable heat-mitigation plan concentrates spend on the most burdened neighbourhoods — which produces a
*highly unequal* allocation and therefore a *high* Gini. Reporting allocation-Gini as the fairness score
means the metric penalises exactly the targeting that equity requires.

**Fix — use a measure that references who benefits:**
- **Concentration index** (Kakwani / Wagstaff — the standard in health equity):
  $CI=\frac{2}{n\bar b}\sum_i b_i R_i - 1$, where $b_i$ is benefit and $R_i$ is the cell's fractional rank
  on a disadvantage index. Negative $CI$ = pro-poor. This is the metric that says what you want it to say.
- **Atkinson index** with inequality aversion $\varepsilon$ on the *post-intervention outcome*
  distribution, which lets the user dial in how much they weight the worst-off.
- Report allocation-Gini too if useful, but label it "spend concentration", not equity.

Minor: `if np.any(x < 0): x = x - x.min()` (`:92`) — Gini is undefined for negatives; shifting changes the
value arbitrarily. Raise, or use a generalised index.

### 16.2 Conformal prediction — **Grade B**

`evaluation/conformal.py` implements weighted split conformal with the correct finite-sample quantile
level $\lceil(1-\alpha)(n+1)\rceil/n$ (`:182`) and spatial proximity weights. Good.

Two theoretical points to make explicit:
- The weights are a **spatial-proximity kernel**, not likelihood ratios $d\tilde P/dP$. That makes this
  *localized* conformal prediction (Guan 2023), not Tibshirani et al. (2019) weighted conformal — the
  guarantee is approximate/local, not the exact marginal one.
- Barber et al. (2023) — cited in the module docstring — give coverage under non-exchangeability as
  $1-\alpha-\Delta$ with an explicit gap term
  $\Delta=\frac{2}{\sum w}\sum_i w_i\, d_{TV}(Z,Z^i)$. The implementation uses the weighted quantile but
  never reports or bounds $\Delta$, so the interval is presented at $1-\alpha$ when the theory does not
  support that.

**Fix:** report empirical coverage on a **spatially held-out** block as the primary evidence (this is the
number that actually settles it), and state the localized-CP framing in the docs rather than the exact
weighted-conformal one.

### 16.3 Extrapolation guard — **Grade B**

`interventions/extrapolation_guard.py:84` thresholds Mahalanobis distance at the **95th percentile of the
training distances**. That is a relative threshold, so by construction 5% of the training data is always
flagged as extrapolation. Under multivariate normality $D^2\sim\chi^2_p$, so $\chi^2_{p,0.95}$ is the
principled absolute threshold. Also: the covariance should be robust (MCD) — a single outlier inflates
$\hat\Sigma$ and makes everything look in-distribution.

### 16.4 Hard-coded metre constants break CRS-agnosticism — **Grade C**

The pipeline carries a `unit_label` from the working CRS and advertises support for global (ForceSMIP)
work, but these are hardcoded:

- `spatial_fold_factory.py:225` — `block_size = max(block_size, 500)`
- `:69` — `bandwidth=500`, `:30` — `threshold=1000`
- `spatial_autocorr_comprehensive.py:60` — `min(1800.0, ...)`
- `neural_meta.py:113` — `init_bandwidth: float = 1000.0`

In a CRS with US-survey-feet units these are 3.3× off; on a degree grid they are meaningless (500° is
larger than the planet). Derive every length constant from the Stage-0 practical range or the data extent,
and assert unit consistency at the CRS boundary.

### 16.5 Reproducibility — **Grade C**

Beyond §5.3: `GWRFModel.fit` unseeded subsample + unseeded RF; `np.random.seed(42)` used as a global
side-effect rather than a `default_rng` instance. Adopt one convention (`np.random.default_rng(seed)`
threaded through), seed torch and sklearn explicitly, and add a test asserting that two runs of Stage 2
on a fixture produce bit-identical β surfaces.

---

## 17. Prioritised roadmap

Ordered by (impact on reported numbers) ÷ (implementation cost).

### Sprint 1 — correctness fixes that change published results

| # | Change | Files | Effort |
|---|---|---|---|
| 1 | Unify the Matérn parameterisation; pass **practical range**, not κ, across module boundaries | `correlogram_matern_fit.py`, `kernel_field.py`, `bayesian_mgwr_ensemble.py` | S |
| 2 | Non-dimensionalise the PDE residuals; remove `detach_scale=False`; fixed/EMA scales | `pde_operators.py`, `pde_loss.py` | M |
| 3 | Delete term 3 (identically zero); fix term 4's sign and wire it to the Stage-0 ellipse; rename terms 5–6 as regularisers | `pde_loss.py` | S |
| 4 | Spatial `GroupKFold` for all DML cross-fitting | `spatial_cate.py`, `ate_estimator_stack.py` | S |
| 5 | Correct `Var_N(I)` from $S_0,S_1,S_2$; reconcile the FFT and exact paths under one null | `spatial_autocorr_comprehensive.py` | M |
| 6 | Correct the BGe score (multivariate gamma sum + $-m+1$ exponents); standardise data | `mc3.py` | S |
| 7 | Fix the NUTS mass matrix (drop the reciprocal) and the U-turn velocity term | `nuts.py` | S |
| 8 | Correct the Cinelli–Hazlett RV formula; standardise E-value inputs | `sensitivity.py` | S |
| 9 | Replace every fabricated-uncertainty fallback with a raise or `NaN` + mask | `gwr.py`, `ggpgam.py`, `spatial_cate.py` | S |
| 10 | `proceed_when_unidentifiable=False`; propagate an `identified` flag | `causal_validation.py`, `dag_definition.py` | S |

### Sprint 2 — implement what is claimed

| # | Change | Files | Effort |
|---|---|---|---|
| 11 | MGWR back-fitting (Fotheringham–Yang–Kang) with per-surface bandwidths | `gwr.py` | L |
| 12 | GWR hat matrix → ENP, AICc bandwidth selection (golden-section), local SEs | `gwr.py`, `gwr_bandwidth.py` | M |
| 13 | Bayesian MGWR: add the within-κ sampling variance (law of total variance) | `bayesian_mgwr_ensemble.py` | S |
| 14 | GGPGAM: bivariate `te(u,v)` spatial basis; `by=`-style SVC terms; per-term λ by GCV/REML | `ggpgam.py` | M |
| 15 | Fix GWRF's kernel-field import + call signature; derive $k$ from Stage 0; seed everything | `gwrf.py` | S |
| 16 | Either implement the urban surface energy balance properly and wire it in, or remove the claim | `energy_balance.py`, `pde_loss.py`, README | M |
| 17 | Coherent exceedance: heteroscedastic $(\mu,\sigma)$ head, or cumulative-product classifiers | `neural_meta.py`, `loss.py` | M |
| 18 | Action-condition the JEPA predictor during pretraining; remove or train `_pt_beta_proj` | `v2_neural_training.py` | M |
| 19 | ν selection by PSIS-LOO/WAIC instead of the harmonic-mean estimator | `correlogram_matern_fit.py` | S |
| 20 | Anisotropy reparameterisation $(\kappa_{\text{geo}}, r\ge1, \theta)$ to kill label switching | `anisotropy.py` | S |

### Sprint 3 — statistical maturity

| # | Change | Effort |
|---|---|---|
| 21 | Rank-normalised split-R̂ and tail-ESS via `arviz`; divergence flagging at the base case; threshold 1.01 | S |
| 22 | Informative structural prior around the expert DAG; report CPDAG features and adjacency-vs-orientation separately; multi-chain R̂ on edge probabilities | M |
| 23 | Refute the estimator actually used; use DoWhy p-values; sweep the unobserved-confounder grid to the sign-flip contour | M |
| 24 | Replace allocation-Gini with a benefit concentration index and/or Atkinson on post-intervention outcomes | S |
| 25 | Aleatoric + epistemic variance, conformalised, with spatially-held-out coverage as the reported number | M |
| 26 | Derive every length constant from the Stage-0 range; unit assertions at the CRS boundary | S |
| 27 | Fix the nugget: drop τ² or identify it from a true $h\to0$ bin | S |
| 28 | Vecchia-approximated direct ML for the Matérn, replacing the two-stage binned fit | L |

### Suggested validation harness

Several of these fixes have clean oracles. Worth building alongside:

- **Synthetic Matérn fields** with known $(\ell,\nu,\kappa_x,\kappa_y,\theta)$ → assert Stage 0 recovers
  them within posterior HDI. Catches §2.2, §2.3, §3.3 at once.
- **`esda.Moran` cross-check** on a fixture → catches §1.3 and §1.4.
- **Known-covariance Gaussian target** for NUTS → catches §8.2, §8.3, §8.4.
- **`bnlearn`/reference BGe** on a small fixture → catches §13.3.
- **Simulated DGP with known ATE and known spatial confounding** → the single best test of §14.2; the
  attenuation from random-fold DML will show up immediately as bias against the truth.
- **Manufactured PDE solution** (method of manufactured solutions): pick $T(x,y)$ analytically, derive
  $S=\alpha\nabla^2T$, and assert the physics loss actually decreases when the network approaches it.
  This is the test that would have caught §9.2 on day one.

---

## References

**Spatial statistics**
Cliff & Ord (1981) *Spatial Processes*. ·
Cressie (1993) *Statistics for Spatial Data*. ·
Stein (1999) *Interpolation of Spatial Data*. ·
Handcock & Wallis (1994) JASA 89(426). ·
Katzfuss & Guinness (2021) *Statist. Sci.* 36(1) — Vecchia. ·
Roberts et al. (2017) *Ecography* 40 — blocked CV.

**Geographically weighted models**
Brunsdon, Fotheringham & Charlton (1996) *Geogr. Anal.* 28(4). ·
Fotheringham, Brunsdon & Charlton (2002) *Geographically Weighted Regression*. ·
Fotheringham, Yang & Kang (2017) *Ann. AAG* 107(6) — MGWR. ·
Da Silva & Fotheringham (2016) *Ann. AAG* 106(5) — multiple testing. ·
Georganos et al. (2021) *Geocarto Int.* 36(2) — GWRF. ·
Comber, Harris & Brunsdon (2023/2024) — GGP-GAM. ·
Wood (2011) *JRSS-B* 73(1) — REML for GAMs.

**Bayesian computation**
Hoffman & Gelman (2014) *JMLR* 15 — NUTS. ·
Betancourt (2017) arXiv:1701.02434. ·
Vehtari et al. (2021) *Bayesian Anal.* 16(2) — rank-normalised R̂, ESS. ·
Vehtari, Gelman & Gabry (2017) *Stat. Comput.* 27 — PSIS-LOO. ·
Neal (2008) on the harmonic-mean estimator. ·
Meng & Wong (1996) *Statist. Sinica* 6 — bridge sampling.

**Physics-informed ML**
Raissi, Perdikaris & Karniadakis (2019) *JCP* 378. ·
Wang, Teng & Perdikaris (2021) *SIAM J. Sci. Comput.* 43(5). ·
Wang, Sankaran & Perdikaris (2022) arXiv:2203.07404 — NTK weighting. ·
Sitzmann et al. (2020) NeurIPS — SIREN. ·
Grimmond & Oke (1999, 2002) *J. Appl. Meteorol.* — urban energy balance, OHM.

**Self-supervised learning**
Assran et al. (2023) CVPR — I-JEPA. ·
Bardes et al. (2024) — V-JEPA. ·
Bardes, Ponce & LeCun (2022) ICLR — VICReg. ·
Gal & Ghahramani (2016) ICML — MC dropout. ·
Rahimi & Recht (2007) NeurIPS — random Fourier features.

**Causal inference**
Pearl (2009) *Causality*, 2nd ed. ·
Chernozhukov et al. (2018) *Econom. J.* 21(1) — DML. ·
Wager & Athey (2018) *JASA* 113(523) — causal forests. ·
Geiger & Heckerman (2002) *Ann. Statist.* 30(5) — BGe. ·
Kuipers, Moffa & Heckerman (2014) *Ann. Statist.* 42(4) — BGe correction. ·
Madigan & York (1995) *Int. Statist. Rev.* 63(2) — MC³. ·
Grzegorczyk & Husmeier (2008) *Mach. Learn.* 71 — edge reversal. ·
Henckel, Perković & Maathuis (2022) *JRSS-B* 84(2) — optimal adjustment. ·
VanderWeele & Ding (2017) *Ann. Intern. Med.* 167(4) — E-values. ·
Cinelli & Hazlett (2020) *JRSS-B* 82(1) — omitted-variable bias.

**Conformal & equity**
Tibshirani et al. (2019) NeurIPS — weighted conformal. ·
Barber et al. (2023) *Ann. Statist.* 51(2) — beyond exchangeability. ·
Guan (2023) *Biometrika* 110(1) — localized conformal. ·
Wagstaff, Paci & van Doorslaer (1991) *Soc. Sci. Med.* 33 — concentration index. ·
Atkinson (1970) *J. Econ. Theory* 2(3).
