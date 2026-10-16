"""Identifiability, prior sensitivity and prior–data conflict of the physics model (PHYSICS_PLAN W3).

A post-run study of a finished run (``sparc core identifiability -r RUN_DIR``; Studio job ``post.identifiability``):
it refits the physics model of :mod:`sparc.core.physics` on *all* labelled cells and asks, per fitted parameter,
how much the data alone say about it, how much the run's priors decide it, and whether the two disagree.  The
run's mechanism register is refreshed with it (:mod:`sparc.core.mechanisms`: the parameter panel's identifiability
columns, the rule that a mechanism whose key parameter the data cannot pin down cannot be Supported, and the lever
statements), and the results page, ``methods.md`` and ``model_card.md`` report it.

Why the likelihood must be adjusted
-----------------------------------
The physics fit minimises a Gaussian independence objective (MSE over ~55,000 Providence cells).  Its residuals
are correlated over kilometres (the S1 target-residual range reaches its 2 km cap; CV uses 2 km blocks), so the
independence log-likelihood overstates the information by a large factor, and a naive profile likelihood calls
almost every parameter "identified".  The curvature is corrected with the sandwich of Chandler & Bate (2007),
with the run's spatial CV blocks (:attr:`sparc.core.cv.SpatialFolds.block_id`) as the independent clusters.

Method
------
1. **Reference fits** on every labelled cell with the run's config, its advection selection (``fit_advection``
   off when the out-of-fold test dropped it) and its advection scheme: ``penalized`` (the run's objective) and
   ``data-only`` (every prior term off, ``prior_weight`` 0 including the EF and κ terms; the a ≥ 0 sign push
   and the drift penalty stay).  Both are multi-started (the default start, the other fit's optimum, and
   perturbed starts) and polished with a longer L-BFGS run.
2. **Independence log-likelihood** l_I(θ) = −(n/2)·log(RSS(θ)/n), θ = every free parameter: the active
   nonlinear parameters (each in its walk coordinate: log L, log κ, log width, the natural value otherwise; see
   below) and the linear coefficients (a, b, γ, βx, βy) as explicit parameters (× their column RMS, for
   conditioning), not profiled by variable projection, so the derivatives are exact
   (:class:`sparc.core.physics.FitTerms`, torch autograd, float64).  H_I is the Hessian
   of l_I at the data-only maximiser θ̂.  Cluster scores U_b are the gradients of block b's part of l_I with
   σ² held at its MLE; V is their covariance, Σ_rob = H_I⁻¹VH_I⁻¹, Σ_naive = −H_I⁻¹ (Chandler & Bate 2007).
   Two score covariances are computed, and every interval is reported with both (below): ``blocks``,
   Chandler & Bate's V = Σ_b (U_b − Ū)(U_b − Ū)ᵀ (``ci95``, ``se_robust``, ``adjust_factor``), and
   ``neighbours``, a sensitivity that adds adjacent blocks' cross-products (``ci95_neighbours``, …).  A
   warning is raised when there are fewer than two clusters per parameter.
3. **Adjusted profile** per parameter ψ in natural units (L in m, s, κ or c₀ and width, a₁/a₁ₙ, e₀–e₃, w, |v|).
   The vertical adjustment (Chandler & Bate 2007, as documented for the chandwich package)

       l_A(θ) = l_I(θ̂) + [δᵀH_Aδ / δᵀH_Iδ]·[l_I(θ) − l_I(θ̂)],   δ = θ − θ̂,   H_A⁻¹ = −H_I⁻¹VH_I⁻¹,

   gives Λ_A(ψ) = 2[l_A(θ̂) − l_A(θ_ψ)].  At each ψ, l_I is maximised over every other parameter (L-BFGS
   walking outward from θ̂ in both directions with adaptive steps; a side stops once every Λ_A and the naive
   Λ_I exceed χ²₁(0.95) = 3.841 by ``margin``, at the fit bound, or after ``max_points``), and from that point
   l_A is maximised over the other parameters too (below).  No maximisation on a profile is single-start
   (below: *Profile maximisation*).  The 95% pointwise interval is
   {ψ : Λ_A ≤ 3.841}, its ends interpolated in √Λ; the naive interval uses Λ_I.  Following Raue et al. (2009)
   a side of the profile that never exceeds the threshold is *open* (practically non-identifiable on that
   side), wherever it stops: at a physical limit of the parameter (a fraction's 0 or 1, the 0 of a
   non-negative slope or sink: :data:`PHYSICAL_LIMITS`; ``sides_at_limit``: the data do not exclude the
   limit), at a fit bound or search range (L's, κ's; the bound an estimate sits on), or within the explored
   range.  Classes (``identifiability``): *identified* when both sides cross; *one_sided* when one side
   crosses and the other is open (``open_sides``, ``open_reasons``); *practically non-identifiable* when
   neither crosses; *structurally non-identifiable* when Λ_A < 0.1 everywhere explored or the parameter does
   not enter the model (a missing role); *not assessable* when its adjustment is not computable (a parameter on
   its fit bound without a well-conditioned curvature, below).  The class rests on the wider of the two
   adjusted intervals per side (``ci95_conservative``).  The interval keeps its informative part: ``ci95``
   ends at a physical limit or the bound an estimate sits on (finite, with that side reported open), and
   ``summary`` says it in words, e.g. "s ≤ 0.49; lower side open at the physical limit 0 (the estimate is on
   it); one-sided 95% bound at Λ = 2.706 (Self & Liang 1987)".  The scalar
   c_j = [Σ_naive]_jj / [Σ_rob]_jj (natural units, delta method) summarises how much the residual dependence
   shrinks the information on θ_j.
4. **Linear coefficients and derived quantities**: naive and sandwich SEs of a, γ, b, βx, βy at both optima,
   and r_a = a_K·ρc_p/norm and γ in K/km with the helpers of :mod:`sparc.core.mechanisms`, so the units are
   the parameter panel's.
5. **Prior influence on the run's estimate**: pull_j = (θ_pen − θ_data)/SE_rob (natural units) and
   ``prior_share_run`` = the share of the run objective's profile curvature in θ_j (Schur complement, other
   parameters re-optimised) that the penalty contributes, 1 − S_data/S_total, at the penalized optimum.
6. **Calibrated posterior** (Ribatet, Cooley & Davison 2012): log p(θ) = l_A(θ) + Σ_j log N(θ_j; μ_j, sd_j²)
   over the parameters with a *nominal* prior — the literature priors as configured (``physics.priors`` /
   ``SOURCE_PRIORS`` for s, a₁, a₁ₙ), κ's log-normal, the EF and shade-sigmoid priors as coded
   (:meth:`PhysicsModel.prior_terms`) — not the ``prior_weight``-scaled penalty the run minimises.  L, w, the
   drift and the linear coefficients have flat priors.  Laplace approximation at the posterior mode (L-BFGS
   from θ̂), covariance −(H_A + prior Hessian)⁻¹.  **This is a Laplace approximation, not MCMC**, and it is
   conditional on the parameters it holds fixed (those on a fit bound, and flat ones without a nominal prior:
   ``posterior.conditional_on``, repeated in each row's ``description``).  Its adequacy is checked (the
   importance-sampling ESS of its draws for exp(l_A + log prior), below); when the check fails, every Laplace
   row's ``posterior`` and ``psens`` carry ``reliable: false`` with the reason, the markdown and the register do
   not show them, and they are not findings.  A parameter on its bound with a nominal prior has a 1-D *profile*
   posterior instead (``method: profile``, its own ``description``; the Laplace check does not apply to it).
7. **Power-scaling sensitivity** (Kallioinen et al. 2024), computed as ArviZ 0.20.0's ``psens`` does:
   ``draws`` draws from the Laplace posterior, importance log-weights (α − 1)·(component log value) for
   α = 1/1.01 and 1.01, Pareto-smoothed as ArviZ's ``_powerscale_lw`` does (:func:`psislw`, a port of ArviZ's
   ``psislw``/``_psislw``/``_gpdfit``/``_gpinv`` for a 1-D array; checked against ArviZ 0.20.0's own functions,
   identical to the last bit on 200 random weight vectors and 50 sensitivities; the tests pin reference values
   computed with ArviZ 0.20.0's ``_cjs_dist``, ``_powerscale_lw``, ``_powerscale_sens`` and ``psens`` on an
   InferenceData), the cumulative Jensen–Shannon
   distance (:func:`cjs_distance`, ArviZ's ``_cjs_dist``) on the draws and their negation (the larger),
   sensitivity = (CJS_lower + CJS_upper)/(2·log₂ 1.01).  Reported with each component: the smoothed weights'
   effective sample size (the smaller of the two α) and the larger Pareto k̂ of the two tail fits (ArviZ's
   ``psens`` computes k̂ but does not report it; a large k̂ means the importance weights, and so the
   sensitivity, are unreliable).
   Prior sensitivity ≥ 0.05 means an informative prior, likelihood sensitivity < 0.05 a weak likelihood; both
   sensitive is Kallioinen et al.'s potential prior–data conflict, prior only a strong prior with a weak
   likelihood.
8. **Prior–data conflict** (Evans & Moshonov 2006), Gaussian approximation of the prior-predictive tail:
   z = (θ̂_data − μ)/√(sd² + SE_rob²), p = 2(1 − Φ(|z|)), conflict when p < 0.05 (κ on the log scale).  Not
   testable when the data-only estimate sits on a fit bound or the parameter is not identified.

Choices beyond the plan (each argued; the simulation numbers are from the albedo world of
``tests/core/test_identifiability.py``: planted L = 90 m, 36 blocks of 300 m, 60 replicates per setting)
-----------------------------------------------------------------------------------------------------------
* **Score covariance: Chandler & Bate's blocks, and adjacent blocks as a sensitivity.**  Chandler & Bate's
  V = Σ_b (U_b − Ū)(U_b − Ū)ᵀ (``blocks``, the default and the verified method; invariant to the parameter
  coordinates) assumes independent clusters.  Spatial blocks that touch are not when the residual range is
  comparable to the block size, which is Providence's case (practical range ≥ 2 km, blocks of 2 km): the scores
  of smooth directions (the gain a, the trends) covary across block edges, and L̂ inherits it through its
  coupling to a.  So every interval is also reported with ``neighbours``: Var(Σ_b U_b) = Σ_b Σ_b′ Cov(U_b,
  U_b′), and V adds the cross-products Σ_{b~b′} (U_b − Ū)(U_b′ − Ū)ᵀ of edge- and corner-adjacent blocks, a
  block-level spatial HAC in the manner of Conley (1999) (a uniform kernel over adjacent blocks).  As Conley
  notes, such a sum need not be positive semi-definite; this module's own choice is to add only its positive
  part *relative to the block V* (the generalized eigenvalues of the cross-product matrix with respect to V,
  clipped at 0: :func:`_pos_part_rel`), which is invariant to the parameter coordinates (a test rescales them)
  and never puts V below Chandler & Bate's.  It is *not* Chandler & Bate's V under independence: there the
  sampled cross-products are noise and their positive part is kept, so it is upward-biased (on the albedo world
  with independent noise its variance for L̂ is a median 14% above the block V's, 10–90% range 0–54%).  The
  identifiability class rests on the wider of the two intervals (per side), so it is conservative too.  The
  linearisation holds on the albedo world with noise of practical range = the block size: the ideal sandwich
  (V from 300 noise draws) gives sd(L̂) = 11.4 m against the 12.1 m spread of the estimates, while the block V
  gives 8.1 m (rms over replicates) and the neighbours V 9.8 m.
  Validation (``tests/core/test_identifiability.py``, slow; coverage of the planted value by the 95% intervals):
  on the albedo world (L only; 60 replicates, 36 blocks of 300 m) with that correlated noise, blocks 80%,
  neighbours 88%, naive 20% (median c 0.048 and 0.035); with independent noise blocks 92%, neighbours 93%, naive
  93% (median c 1.15 and 0.96).  On the multi-parameter shade world (closer to Providence: L, canopy shading
  s saturating with κ, the EF intercept and canopy slope, 9 free parameters, 36 blocks of 180 m, noise of
  practical range = the block size; the data are the model's own forward map, 60 replicates): L blocks 80%,
  neighbours 97%, naive 38%; s blocks 72%, neighbours 80%, naive 25% (median c 0.11 and 0.12 with the block V).
  s under-covers because its estimator is biased upward (mean 0.47 for a planted 0.40, sd 0.13; the EF canopy
  slope, planted 1.0, landed on its bound 0 in 14 replicates), which no variance adjustment can fix; the
  adjusted intervals are still far closer to nominal than the naive ones.
  V is centred (U_b − Ū): identical at an interior maximum (Ū = ∇l_I/G = 0), and the score
  variance where the mean score is not zero (the penalized optimum, a bound).
* **Coordinates of H_I, V and δ: walk coordinates, not the fit's unconstrained leaves.**  The vertical
  adjustment's ratio δᵀH_Aδ/δᵀH_Iδ depends on the coordinates away from θ̂.  In the leaves, a parameter (or a
  nuisance parameter along a profile) that reaches a softplus floor or a sigmoid end has a leaf running to −∞
  while l_I goes flat, so δ is dominated by that one coordinate: on Providence the L profile jumped from
  Λ_A = 2.5 to 26 between 414 and 438 m (R 0.010 → 0.077) and e₂'s from 0.3 to 5.5 at its floor.  The walk
  coordinates are finite on every bound, and the same as the profile's own steps.  The optimisers still run in
  the leaves (which keep the bounds).
* **The adjusted profile maximises l_A, not only l_I.**  Evaluating l_A at the l_I-profile point (the
  design's θ_ψ) overstates Λ_A: in the quadratic approximation, along the l_I path δ ∝ Σ_naive·e_j, so
  Λ_A = (ψ − ψ̂)²·aᵀΣ_rob⁻¹a/a_j² with a = Σ_naive·e_j, which by Cauchy–Schwarz is ≥ (ψ − ψ̂)²/[Σ_rob]_jj,
  the adjusted profile's value, with equality only when the naive and robust covariances share the profile
  direction.  So l_A is re-maximised over the other parameters (as chandwich's ``conf_intervals`` profile the
  adjusted log-likelihood); ``lambda_adj_ipath`` keeps the design's value.
* **Profile maximisation.**  Each profile point is maximised from several starts and the best optimum is
  kept: for l_I, the previous point's optimum, the reference optimum with ψ substituted, and both with the
  parameters held on a bound released inside it; for l_A, the l_I-profile point, the previous point's l_A
  optimum, the quadratic adjusted path θ̂ + Δψ·Σ_rob e_ψ/[Σ_rob]_ψψ and the reference optimum with ψ
  substituted.  Every distinct start is run for an eighth of the iterations (:data:`SCREEN_FRAC`) and the best
  continued for the full count (the physics fit's own multistart scheme, with a shorter screen), and the best
  value seen is kept, so Λ_A never exceeds
  its value at the l_I-profile point without taking a minimum over different paths.  A profile (a maximum
  over the other parameters) cannot rise inward: a point whose Λ (naive or either adjusted one) exceeds the
  next point outward's by more than :data:`MONO_TOL` is re-maximised from that point's optimum and the better
  optima kept (two passes); what remains is recorded (``profile.monotone``, ``non_monotone``) and warned
  about.  On the 13-parameter city world of the tests the single-start version gave L's adjusted profile 3.94
  at 92.0 m and 3.48 at 92.2 m, and an upper 95% end of 91.9 m instead of 92.8 m.
* **Profile range.**  A side also stops at a physical limit (:data:`PHYSICAL_LIMITS`: a fraction's 0 or 1,
  the 0 of a non-negative slope, sink or drift), where that side is open but the interval ends, and an EF
  parameter stops where its own term reaches logit ±30 (:func:`_ef_walk_limit`): σ rounds to 1 in float64
  past ≈ 37, where the model's source vanishes numerically (in the albedo world e₀'s exactly flat profile
  jumped to Λ ≈ 5000 there).  A held parameter with no local curvature whose profile is not flat (a saturated
  EF) is never called identified.
* **Restarts.**  When a profile point beats the data-only optimum by more than Δl = 0.05 (0.1 in naive Λ; l_D,
  below, with advection fitted),
  the reference is refitted from that point and the curvature and profiles are recomputed (at most twice,
  then a warning): a profile is only meaningful relative to the maximum.
* **Held parameters.**  A parameter on a fit bound at the data-only optimum (a softplus floor or a sigmoid
  end; on Providence's data-only fit s, a₁, w and κ) has a vanishing derivative in the unconstrained leaf, so H_I would be singular:
  it is held at the bound for H, V and the Laplace step, and gets a one-sided profile.  So is a parameter that
  carries no information (a zero diagonal of H_I, or a direction of H_I with no curvature: e.g. the EF
  intercept when the source has no other term, which the gain a absorbs exactly); its profile is flat, and it
  enters the posterior only through its prior.  "Held" applies to the curvature only: **a profile maximises
  over every other parameter, held ones included** (within their bounds).  Their leaves cannot move from the
  bound (the softplus or sigmoid slope there is ~10⁻¹¹), so each profile point is also started with them
  released inside their bounds (:data:`RELEASE_LEAF`) and the better optimum is kept.  They enter l_I there,
  not δ: the adjustment is that of the profile of l_I over the bound-held ones, whose curvature and scores at θ̂
  are those in the main coordinates (the bound stays active near θ̂), and the l_A maximisation moves them too.
  A flat one is maximised in l_I but keeps the l_I-profile point's value in the l_A maximisation: it lies in a
  direction of H_I without curvature (e₀ confounded with a), so moving it would move δ along that direction at
  no cost in l_I and defeat the adjustment (on a test world it narrowed L's adjusted interval by a third).
  Holding them fixed made the profiles too steep: on Providence (60 m cells) L = 460 m had Λ_I = 409 with s,
  a₁, w and κ held and 126 with them free; on the sink world of the tests, 6.18 against 5.78.
* **A parameter on its fit bound (Self & Liang 1987).**  Its data-only estimate is a boundary maximum: the
  likelihood-ratio statistic for the bound value is then asymptotically a 50:50 mixture of a point mass at 0
  and χ²₁, so its profile is one-sided (walked from the bound inward) and its one-sided 95% bound is where Λ
  reaches χ²₁(0.90) = 2.706 (``threshold``; :data:`CHI2_90`), for the adjusted and the naive profile alike.
  The vertical adjustment is derived at an interior maximum and its ratio δᵀH_Aδ/δᵀH_Iδ depends on the
  nuisance displacement, so it is not used here; this module's choice is the scalar adjustment
  Λ_A(ψ) = c·Λ_I(ψ), c = [Σ_naive]_ψψ/[Σ_rob]_ψψ (per score covariance), with Σ from H_I and the cluster scores
  at the boundary point in the main coordinates extended by ψ's walk coordinate (one-sided derivatives: the
  model is analytic in ψ's natural value).  For a scalar ψ this is the adjustment's value in the quadratic
  approximation, Λ_I ≈ Δψ²/[Σ_naive]_ψψ and Λ_A ≈ Δψ²/[Σ_rob]_ψψ, while Λ_I itself is the true profile.  c is
  only used when that curvature is well-conditioned: l_I concave along ψ, the unit-diagonal −H_I extended by ψ
  positive definite with smallest eigenvalue ≥ :data:`BOUND_MIN_EIG`, and ψ's variance inflation ≤
  :data:`BOUND_MAX_VIF` (``boundary_adjustment``: c, the eigenvalue, the inflation, the reason).  Otherwise
  (an indefinite or near-singular extended H_I, e.g. a₁ nearly confounded with e₃ at a₁ = 0, which earlier
  drove a₁'s vertically adjusted profile to ~0) there is no adjusted interval: ``ci95`` is None, the naive
  one-sided bound is reported as ``ci95_naive`` with ``ci95_naive_label`` "unadjusted (too narrow under
  spatial dependence)", and the parameter is *not assessable* rather than identified; it gets no profile
  posterior either.  On Providence at 120 m cells: s (inflation 56, c = 0.023) and w (5.5, 0.053) are
  adjusted, a₁ (inflation 846) is not assessable, κ carries no information at its bound.
* **SEs and curvature at the penalized optimum** are those of the M-estimator solving the run's estimating
  equation ∇J = 0 (J = MSE/var(y) + penalty): Σ = A⁻¹VA⁻¹ with A = k·(G_D + ∇²penalty), k = n·var(y)/(2·MSE),
  naive A⁻¹(JᵀJ/σ²)A⁻¹; G_D = 2JᵀJ/(n·var y) is the data term's expected (Gauss–Newton) curvature, J = ∂r/∂θ.
  Away from the data optimum the exact Hessian of the MSE can be indefinite (on the synthetic city its smallest
  eigenvalue was −3·10⁻⁵ there), and a negative profile curvature has no "share".
* **Power-scaling components.**  On the scale of the block-adjusted uncertainty l_I is far from quadratic (on
  the synthetic city, one robust sd from θ̂, l_I fell ~9× faster than its quadratic form), so the Laplace
  draws do not follow exp(l_A + log prior), and importance weights built from l_A at the draws measure that
  mismatch rather than the prior–likelihood balance (likelihood sensitivities of 2–6 on the city, against ≈ 0.1
  for an exactly Gaussian posterior).  The likelihood component is therefore the Laplace posterior's own
  likelihood term ½δᵀH_Aδ, the one its covariance was built from; l_A is still evaluated at every draw (one
  model evaluation each): it gives ``likelihood_lA`` (the design's variant, for comparison) and the
  importance-sampling ESS of the Laplace draws for exp(l_A + log prior), reported as the approximation's
  adequacy (a warning below 10%).
* **Laplace coordinates.**  The Laplace step works in the coordinates of H_A (the walk coordinates), in which
  every nominal prior is Gaussian exactly as stated (natural values; κ and the shade width on the log scale), so
  no Jacobian enters and the prior component of power-scaling is the prior as specified.  In the fit's
  unconstrained leaves a softplus or sigmoid saturates: on Providence the mode search ran a leaf to −196 and
  could not return (its gradient there is ~10⁻⁸⁵).  Draws outside the fit bounds are discarded (the posterior
  is restricted to the parameter space: ``draws`` in-bounds draws, ``posterior.in_bounds_frac``).  A parameter
  held on its fit bound is not in the Laplace step; when it has a nominal prior its posterior is exp(−Λ_A/2)·prior along its adjusted profile
  (1-D, sampled by inverse CDF), with power-scaling on those draws (``method: profile``).
* With advection fitted, the drift penalty v_penalty·|u|² stays in the data-only objective, as the design
  asks: it keeps |v| off its bound when the wind is weakly identified.  The study then works with the objective
  the data-only fits maximise, in log-likelihood units: l_D = l_I − k̂·v_penalty·|u|², k̂ = n·var(y)/(2·MSE(θ̂))
  (−k̂·ΔJ = Δl_I to first order, J the fitted objective MSE/var(y) + penalty), in place of l_I everywhere:
  profiles, the restart test, H (its drift block gains the penalty's curvature) and the posterior; the cluster
  scores stay those of l_I (the penalty is not data), and V is centred.  Comparing l_I alone made a profile
  point "beat" the reference only by trading penalty (on a 48 × 48 drift world |v| = 122 m beat 119 m by
  Δl_I = 0.2 while losing 0.7 in penalty), which triggered two futile reference refits (twice the runtime) and
  a "not a maximum" warning.  Without advection l_D = l_I.

References (verified list of PHYSICS_PLAN.md and the W3 task)
------------------------------------------------------------
See :data:`REFERENCES`.  Thresholds: χ²₁(0.95) = 3.841 and "Λ_A < 0.1 everywhere" from Raue et al. (2009);
0.05 for power-scaling from Kallioinen et al. (2024) / ArviZ; p < 0.05 for the conflict tail probability.
``AT_BOUND_FRAC`` and ``FLOOR_TOL`` are the parameter panel's (:mod:`sparc.core.mechanisms`).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from sparc.core import progress

log = logging.getLogger(__name__)

__all__ = ["SCHEMA", "CHI2_95", "REFERENCES", "PARAMS", "cjs_distance", "power_scaling_sensitivity",
           "psens_diagnosis", "prior_data_conflict", "analyse", "identifiability_report",
           "identifiability_markdown"]

SCHEMA = 1
#: χ²₁(0.95): the 95% pointwise profile-likelihood threshold (Raue et al. 2009)
CHI2_95 = 3.841458820694124
#: χ²₁(0.90): the one-sided 95% bound of a parameter whose estimate is on the boundary of its parameter space,
#: where the likelihood-ratio statistic is a 50:50 mixture of a point mass at 0 and χ²₁ (Self & Liang 1987)
CHI2_90 = 2.705543454095404
#: A parameter on its fit bound gets its block adjustment (the scalar c = Σ_naive,ψψ/Σ_rob,ψψ at the bound) only
#: when the curvature it comes from is well-conditioned: the unit-diagonal −H_I extended by ψ positive definite
#: with smallest eigenvalue ≥ BOUND_MIN_EIG, and ψ's variance inflation [(−H)⁻¹]_ψψ·(−H)_ψψ ≤ BOUND_MAX_VIF
#: (ψ not nearly confounded with the other parameters at the bound).  Operational tolerances of this module.
BOUND_MIN_EIG = 1e-4
BOUND_MAX_VIF = 100.0
#: Λ_A below this everywhere explored: a flat profile (Raue et al. 2009's structural non-identifiability)
FLAT_LAMBDA = 0.1
PSENS_DELTA = 0.01                 # power-scaling perturbation (ArviZ psens default)
PSENS_THRESHOLD = 0.05             # Kallioinen et al. (2024) / ArviZ
CONFLICT_ALPHA = 0.05
#: The Laplace approximation is called adequate when the importance-sampling ESS of its draws for
#: exp(l_A + log prior) is at least this share of the draws (an operational threshold of this module); below it
#: the Laplace posterior summaries and power-scaling are flagged ``reliable: false``.
LAPLACE_MIN_ESS_FRAC = 0.1
#: Diagonal of −H_I below this fraction of the intercept's information n/σ²: the coordinate carries no
#: information (an operational tolerance of this module, not a published threshold).
FLAT_INFO = 1e-10
#: Eigenvalue of the unit-diagonal −H_I below this: a parameter combination with no curvature (operational).
NULL_EIG = 1e-9
#: A profile point better than the data-only optimum by more than this (l_I units; 0.1 in naive Λ) triggers a
#: restart of the reference from that point (operational).
RESTART_DL = 0.05
#: Fit bounds that are physical limits of a parameter (a fraction in [0, 1]; a non-negative slope, sink or
#: drift): a profile that reaches one without crossing the threshold is open there (practically
#: non-identifiable on that side, Raue et al. 2009), its interval ending at the limit.  The other bounds (L, κ,
#: the shade sigmoid's c₀ and width, an EF parameter's walk limit) are search ranges: reaching one leaves that
#: side open too, with no finite end.
PHYSICAL_LIMITS = {"s": ("lower", "upper"), "a1": ("lower", "upper"), "a1n": ("lower", "upper"),
                   "e1": ("lower",), "e2": ("lower",), "e3": ("lower",), "w": ("lower",), "v": ("lower",)}
#: The profile of an EF parameter stops where its term alone reaches this logit (:func:`_ef_walk_limit`).
EF_WALK_LOGIT = 30.0

DEFAULTS = {
    "max_points": 12,        # profile points per side
    "margin": 1.0,           # a side stops once Λ_A ≥ 3.841 + margin
    "target_step": 1.0,      # aimed-at increase of Λ_A per profile step
    "refine": 2,             # bisections of the crossing bracket
    "profile_iter": 40,      # L-BFGS iterations per profile point (best start; each start gets an eighth)
    "ref_starts": 4,         # starts per reference fit (default, the other fit's optimum, 2 perturbed)
    "polish_iter": 200,      # L-BFGS iterations polishing the best start
    "mode_iter": 100,        # L-BFGS iterations for the posterior mode
    "draws": 2000,
    "seed": 0,
    "delta": PSENS_DELTA,
    "score_covariance": "blocks",   # the V of the posterior and the run's SEs: "blocks" (Chandler & Bate) or
                                    # "neighbours" (+ adjacent blocks); both intervals are always reported
    "max_restarts": 2,       # data-only reference refits from a better profile point
    "adj_iter": 60,          # L-BFGS iterations maximising l_A per profile point
}
#: The score covariances (module docstring): Chandler & Bate's block V, and this module's adjacent-block variant
SCORE_COVARIANCES = ("blocks", "neighbours")
#: report-key suffix of each score covariance's interval, SE, c and profile
_SUFFIX = {"blocks": "", "neighbours": "_neighbours"}

REFERENCES = {
    "raue2009": "Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U. & Timmer, J. "
                "(2009). Structural and practical identifiability analysis of partially observed dynamical models "
                "by exploiting the profile likelihood. Bioinformatics 25(15), 1923–1929. "
                "doi:10.1093/bioinformatics/btp358",
    "chandler2007": "Chandler, R.E. & Bate, S. (2007). Inference for clustered data using the independence "
                    "loglikelihood. Biometrika 94(1), 167–183. doi:10.1093/biomet/asm015",
    "ribatet2012": "Ribatet, M., Cooley, D. & Davison, A.C. (2012). Bayesian inference from composite likelihoods, "
                   "with an application to spatial extremes. Statistica Sinica 22, 813–845. "
                   "doi:10.5705/ss.2009.248",
    "kallioinen2024": "Kallioinen, N., Paananen, T., Bürkner, P.-C. & Vehtari, A. (2024). Detecting and diagnosing "
                      "prior and likelihood sensitivity with power-scaling. Statistics and Computing 34, 57. "
                      "doi:10.1007/s11222-023-10366-5",
    "evans2006": "Evans, M. & Moshonov, H. (2006). Checking for prior-data conflict. Bayesian Analysis 1(4), "
                 "893–914. doi:10.1214/06-BA129",
    "selfliang1987": "Self, S.G. & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and "
                     "likelihood ratio tests under nonstandard conditions. Journal of the American Statistical "
                     "Association 82(398), 605–610",
    "conley1999": "Conley, T.G. (1999). GMM estimation with cross sectional dependence. Journal of Econometrics "
                  "92(1), 1–45. doi:10.1016/S0304-4076(98)00084-0",
}

#: Profiled parameters: report key → (natural key of :data:`physics.NATURAL_LEAF`, label, unit, walk scale).
#: The walk scale is the coordinate the profile steps in (log for the log-mapped scales).
PARAMS = {
    "L": ("L", "Mixing length L", "m", "log"),
    "v": (None, "Advection drift |v|", "m", "lin"),
    "s": ("s", "Canopy shading strength s", "fraction of absorbed sunlight", "lin"),
    "kappa_canopy": ("kc", "Shade saturation scale κ", "canopy fraction", "log"),
    "shade_c0": ("c0", "Shade inflection c₀", "canopy fraction", "lin"),
    "shade_width": ("cw", "Shade transition width", "canopy fraction", "log"),
    "a1": ("a1", "Impervious heat storage a₁", "fraction of Q*", "lin"),
    "a1n": ("a1n", "Impervious heat release a₁ₙ", "fraction of |LW_net|", "lin"),
    "e0": ("e0", "Evaporative-fraction intercept e₀", "logit", "lin"),
    "e1": ("e1", "Evaporative-fraction slope on NDVI e₁", "logit per NDVI unit", "lin"),
    "e2": ("e2", "Evaporative-fraction slope on canopy e₂", "logit per canopy fraction", "lin"),
    "e3": ("e3", "Evaporative-fraction slope on impervious e₃", "logit per impervious fraction", "lin"),
    "w": ("w", "Water-cell sink w", "W/m²", "lin"),
}
ALIASES = {"kappa": "kappa_canopy", "kc": "kappa_canopy", "c0": "shade_c0", "cw": "shade_width",
           "width": "shade_width", "|v|": "v", "v_norm": "v", "v_norm_m": "v", "L_m": "L"}
_NAT_KEY = {spec[0]: k for k, spec in PARAMS.items() if spec[0]}
#: why a parameter is not in this run's model, by the role it needs
_NEEDS = {"s": "canopy", "kappa_canopy": "canopy", "shade_c0": "canopy", "shade_width": "canopy",
          "a1": "impervious", "a1n": "impervious", "e1": "ndvi", "e2": "canopy", "e3": "impervious",
          "w": "water_distance"}
_LINEAR_LABELS = {"a": ("Source gain a", "target units per unit q"), "b": ("Offset b", "target units"),
                  "gamma": ("Elevation coefficient γ", "target units per m"),
                  "beta_x": ("Trend βx", "target units per m"), "beta_y": ("Trend βy", "target units per m")}
#: Kallioinen et al. (2024)'s reading: prior ≥ 0.05 = informative prior, likelihood < 0.05 = weak likelihood, both
#: ≥ 0.05 = potential prior–data conflict.  ``weak_likelihood`` (priorsense's "strong prior / weak likelihood") is an
#: informative prior with a weak likelihood; a likelihood below 0.05 with a prior below it is still a weak likelihood
#: (the markdown says so; the diagnosis stays "none", as in priorsense).
DIAGNOSIS_TEXT = {"prior_data_conflict": "potential prior–data conflict (prior and likelihood ≥ 0.05)",
                  "weak_likelihood": "informative prior, weak likelihood (prior ≥ 0.05, likelihood < 0.05)",
                  "none": "—"}


# --------------------------------------------------------------------------- small helpers
def _t(x):
    import torch

    return torch.as_tensor(x, dtype=torch.float64)


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _neg_def_inverse(H: np.ndarray, floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray, float]:
    """(−H⁻¹, the clipped H, the smallest eigenvalue of the unit-diagonal −H).  Eigenvalues of the unit-diagonal
    −H below ``floor`` are raised to it (a non-negative-definite H at a non-converged or flat point)."""
    A = _sym(-np.asarray(H, float))
    d = np.sqrt(np.maximum(np.abs(np.diag(A)), 1e-300))
    C = A / np.outer(d, d)
    w, Q = np.linalg.eigh(C)
    wmin = float(w.min()) if w.size else 1.0
    w = np.maximum(w, floor)
    Cinv = (Q / w) @ Q.T
    Cc = (Q * w) @ Q.T
    return Cinv / np.outer(d, d), -(Cc * np.outer(d, d)), wmin


def _pos_inverse(V: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """V⁻¹ of a positive semi-definite matrix, eigenvalues of its unit-diagonal form floored at ``floor``."""
    V = _sym(np.asarray(V, float))
    d = np.sqrt(np.maximum(np.diag(V), 1e-300))
    C = V / np.outer(d, d)
    w, Q = np.linalg.eigh(C)
    w = np.maximum(w, floor * max(float(w.max()), 1e-300))
    return ((Q / w) @ Q.T) / np.outer(d, d)


def _pos_part_rel(C: np.ndarray, M: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Positive part of the symmetric C relative to the positive semi-definite metric M: B·max(K, 0)·Bᵀ with
    M = BBᵀ (B spans M's range) and C = BKBᵀ, i.e. C's eigenvalues relative to M (the generalized eigenproblem
    Cv = λMv) clipped at 0.  For C whose range lies in M's (cross-products of the same centred cluster scores),
    this is equivariant under any invertible reparametrisation θ → Aθ: scores U → UA⁻¹, C → A⁻ᵀCA⁻¹,
    M → A⁻ᵀMA⁻¹, and the result → A⁻ᵀ·result·A⁻¹.  The plain eigenvalue clip of C is not (it depends on the
    parameter coordinates).  Computed in M's unit-diagonal scaling (so the rank cut ``rtol`` is scale-free)."""
    M, C = _sym(np.asarray(M, float)), _sym(np.asarray(C, float))
    d = np.sqrt(np.maximum(np.diag(M), 1e-300))
    Ms, Cs = M / np.outer(d, d), C / np.outer(d, d)
    w, Q = np.linalg.eigh(Ms)
    keep = w > rtol * max(float(w.max()), 1e-300)
    if not keep.any():
        return np.zeros_like(C)
    B = Q[:, keep] * np.sqrt(w[keep])
    Bp = (Q[:, keep] / np.sqrt(w[keep])).T
    k, R = np.linalg.eigh(_sym(Bp @ Cs @ Bp.T))
    return _sym(B @ ((R * np.maximum(k, 0.0)) @ R.T) @ B.T) * np.outer(d, d)


def _schur(H: np.ndarray, j: int) -> float:
    """Profile curvature of coordinate j: H_jj − H_j,−j · pinv(H_−j,−j) · H_−j,j."""
    H = _sym(np.asarray(H, float))
    o = [i for i in range(H.shape[0]) if i != j]
    if not o:
        return float(H[j, j])
    Ho = H[np.ix_(o, o)]
    d = np.sqrt(np.maximum(np.abs(np.diag(Ho)), 1e-300))
    Hs = Ho / np.outer(d, d)
    h = H[o, j] / d
    return float(H[j, j] - h @ np.linalg.pinv(Hs, rcond=1e-10, hermitian=True) @ h)


def _norm_sf2(z: float) -> float:
    """Two-sided normal tail 2(1 − Φ(|z|))."""
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


# --------------------------------------------------------------------------- power-scaling (ArviZ 0.20.0 psens)
def cjs_distance(draws, weights) -> float:
    """Cumulative Jensen–Shannon distance between the empirical distribution of ``draws`` and its
    ``weights``-reweighted version: a port of ArviZ 0.20.0 ``arviz.stats._cjs_dist`` (Kallioinen et al. 2024),
    the same arithmetic, except that draws with no spread (both ECDF integrals 0) give 0 where ArviZ gives
    NaN (0/0)."""
    draws = np.asarray(draws, float).ravel()
    weights = np.asarray(weights, float).ravel()
    order = np.argsort(draws)
    draws, weights = draws[order], weights[order]
    binwidth = np.diff(draws)
    n = draws.size
    cdf_p = np.linspace(1.0 / n, 1.0 - 1.0 / n, n - 1)
    cdf_q = np.cumsum(weights / np.sum(weights))[:-1]
    cdf_p_int = float(np.dot(cdf_p, binwidth))
    cdf_q_int = float(np.dot(cdf_q, binwidth))
    pq_numer = np.log2(cdf_p, out=np.zeros_like(cdf_p), where=cdf_p != 0)
    qp_numer = np.log2(cdf_q, out=np.zeros_like(cdf_q), where=cdf_q != 0)
    denom = 0.5 * (cdf_p + cdf_q)
    denom = np.log2(denom, out=np.zeros_like(denom), where=denom != 0)
    cjs_pq = float(np.sum(binwidth * (cdf_p * (pq_numer - denom)))) + 0.5 / math.log(2.0) * (cdf_q_int - cdf_p_int)
    cjs_qp = float(np.sum(binwidth * (cdf_q * (qp_numer - denom)))) + 0.5 / math.log(2.0) * (cdf_p_int - cdf_q_int)
    cjs_pq, cjs_qp = max(cjs_pq, 0.0), max(cjs_qp, 0.0)
    bound = cdf_p_int + cdf_q_int
    if not bound > 0:
        return 0.0
    return float(math.sqrt((cjs_pq + cjs_qp) / bound))


def _logsumexp(x: np.ndarray) -> float:
    """log Σ exp(x), stable (ArviZ 0.20.0 ``stats_utils.logsumexp`` for a 1-D array, b = None)."""
    m = float(np.max(x))
    return float(np.log(np.sum(np.exp(x - m))) + m)


def _gpdfit(ary: np.ndarray) -> tuple[float, float]:
    """Empirical-Bayes estimate (k, σ) of the generalized Pareto distribution of the sorted 1-D ``ary``: a port of
    ArviZ 0.20.0 ``arviz.stats.stats._gpdfit`` (same constants: prior_bs 3, prior_k 10, m_est = 30 + ⌊√n⌋)."""
    prior_bs, prior_k = 3, 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]
    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    weights /= weights.sum()
    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * ary).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return float(k_post), float(sigma)


def _gpinv(probs: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    """Inverse generalized Pareto distribution function (ArviZ 0.20.0 ``_gpinv``)."""
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        x[probs == 1] = np.inf if kappa >= 0 else -sigma / kappa
    return x


def psislw(log_weights, reff: float = 1.0) -> tuple[np.ndarray, float]:
    """Pareto-smoothed importance sampling of one vector of log-weights: (smoothed, truncated and normalised
    log-weights, Pareto shape estimate k̂).  A port of ArviZ 0.20.0 ``arviz.stats.psislw`` / ``_psislw`` for a
    1-D numpy array (the xarray wrapping dropped; the tail length, the cutoff, the generalized Pareto fit and
    the truncation at the largest raw weight are ArviZ's).  k̂ is inf when the tail has ≤ 4 draws."""
    x = np.array(log_weights, dtype=float, copy=True).ravel()
    n_samples = x.size
    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)
    x -= np.max(x)
    x_sort_ind = np.argsort(x)
    xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)
    expxcutoff = np.exp(xcutoff)
    (tailinds,) = np.where(x > xcutoff)
    x_tail = x[tailinds]
    tail_len = len(x_tail)
    if tail_len <= 4:
        k = np.inf
    else:
        x_tail_si = np.argsort(x_tail)
        x_tail = np.exp(x_tail) - expxcutoff
        k, sigma = _gpdfit(x_tail[x_tail_si])
        if np.isfinite(k):
            sti = np.arange(0.5, tail_len) / tail_len
            smoothed_tail = np.log(_gpinv(sti, k, sigma) + expxcutoff)
            x[tailinds[x_tail_si]] = smoothed_tail
            x[x > 0] = 0
    x -= _logsumexp(x)
    return x, float(k)


def _psens_weights(component, delta: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalised importance weights for power-scaling a component by α = 1/(1+δ) and 1+δ, with the Pareto k̂ of
    each: log-weights (α − 1)·(component's log value), Pareto-smoothed (:func:`psislw`), exponentiated and
    normalised, as ArviZ 0.20.0 ``psens`` / ``_powerscale_lw`` do."""
    c = np.asarray(component, float)
    out, ks = [], []
    for alpha in (1.0 / (1.0 + delta), 1.0 + delta):
        lw, k = psislw((alpha - 1.0) * c)
        w = np.exp(lw)
        out.append(w / np.sum(w))
        ks.append(k)
    return out[0], out[1], ks[0], ks[1]


def _ess(w: np.ndarray) -> float:
    return float(1.0 / np.sum(w * w))


def power_scaling_sensitivity(draws, log_prior=None, log_lik=None, delta: float = PSENS_DELTA) -> dict:
    """Prior and likelihood power-scaling sensitivity of one parameter's ``draws`` (ArviZ 0.20 ``psens``;
    Kallioinen et al. 2024): sensitivity = (CJS_lower + CJS_upper)/(2·log₂(1+δ)), each CJS the larger of the
    draws' and their negation's, the importance weights Pareto-smoothed (:func:`psislw`) as ArviZ does.
    ``log_prior`` / ``log_lik``: the component's log value at each draw.  Returns ``{prior, likelihood,
    ess_prior, ess_likelihood, khat_prior, khat_likelihood, diagnosis}`` (None for a component not given; k̂ is
    the larger of the two power-scalings')."""
    x = np.asarray(draws, float)
    denom = 2.0 * math.log2(1.0 + delta)
    out: dict = {}
    for name, comp in (("prior", log_prior), ("likelihood", log_lik)):
        if comp is None:
            out[name], out[f"ess_{name}"], out[f"khat_{name}"] = None, None, None
            continue
        lo, hi, k_lo, k_hi = _psens_weights(comp, delta)
        cjs_lo = max(cjs_distance(x, lo), cjs_distance(-x, lo))
        cjs_hi = max(cjs_distance(x, hi), cjs_distance(-x, hi))
        out[name] = float((cjs_lo + cjs_hi) / denom)
        out[f"ess_{name}"] = float(min(_ess(lo), _ess(hi)))
        out[f"khat_{name}"] = _f(max(k_lo, k_hi)) if math.isfinite(max(k_lo, k_hi)) else None
    out["diagnosis"] = psens_diagnosis(out.get("prior"), out.get("likelihood"))
    return out


def psens_diagnosis(prior: float | None, likelihood: float | None, threshold: float = PSENS_THRESHOLD) -> str | None:
    """Kallioinen et al. (2024): prior AND likelihood sensitive → prior–data conflict; prior sensitive, likelihood
    not → a strong prior with a weak likelihood (the prior dominates); otherwise none."""
    if prior is None or likelihood is None:
        return None
    if prior >= threshold and likelihood >= threshold:
        return "prior_data_conflict"
    if prior >= threshold:
        return "weak_likelihood"
    return "none"


def prior_data_conflict(estimate: float | None, se: float | None, mean: float, sd: float, *,
                        at_bound: bool = False, identified: bool = True, reason: str | None = None) -> dict:
    """Evans & Moshonov (2006) prior-predictive tail probability in its Gaussian approximation:
    z = (θ̂ − μ)/√(sd² + SE²), p = 2(1 − Φ(|z|)), conflict when p < 0.05.  All on the prior's scale.  Not
    testable (``reason``) when the estimate sits on a fit bound, the parameter is not identified, or there is
    no SE."""
    why = reason
    if why is None and at_bound:
        why = "the data-only estimate sits on a fit bound: the data do not give an interior estimate to compare"
    if why is None and not identified:
        why = "the parameter is not identified by the data (profile likelihood), so there is no data estimate"
    if why is None and (_f(estimate) is None or _f(se) is None):
        why = "no data-only estimate with a standard error"
    if why is not None:
        return {"z": None, "p": None, "flag": None, "reason": why}
    z = (float(estimate) - float(mean)) / math.sqrt(float(sd) ** 2 + float(se) ** 2)
    p = _norm_sf2(z)
    return {"z": float(z), "p": float(p), "flag": bool(p < CONFLICT_ALPHA), "reason": None}


# --------------------------------------------------------------------------- coordinate systems
#: scale parameters: their coordinate (for H, V, δ and the profile walk) is the logarithm
_LOG_KEYS = ("L", "kc", "cw")


class _Space:
    """Coordinates θ for H_I, V, H_A and δ = θ − θ̂: each nonlinear parameter in its *walk* coordinate (the log
    of a scale parameter L, κ, width; the natural value ÷ a unit scale otherwise; the drift's leaves), then the
    linear coefficients × their column RMS.  Unlike the fit's unconstrained leaves these stay finite on every
    fit bound, so δ does not run off to infinity when a parameter reaches a softplus floor or a sigmoid end.
    ``base`` holds the leaves of everything not in ``keys`` (held parameters)."""

    def __init__(self, st: "_Study", base: dict, keys):
        self.st = st
        self.base = dict(base)
        self.keys = list(keys)
        self.names = self.keys + list(st.terms.linear_names)
        self.p = len(self.names)
        self.nl = len(self.keys)

    def scale(self, k: str) -> float:
        return 1.0 if k in ("vx", "vy") else self.st.model._leaf_scale(k)

    def coord(self, k: str, nat: float) -> float:
        if k in ("vx", "vy"):
            return float(nat) / self.st.model.v_max
        return math.log(max(float(nat), 1e-300)) if k in _LOG_KEYS else float(nat) / self.scale(k)

    def natural(self, k: str, x: float) -> float:
        if k in ("vx", "vy"):
            return float(x) * self.st.model.v_max
        return math.exp(x) if k in _LOG_KEYS else float(x) * self.scale(k)

    def coord_t(self, k: str, nat_t):
        import torch

        if k in ("vx", "vy"):
            return nat_t / self.st.model.v_max
        return torch.log(nat_t) if k in _LOG_KEYS else nat_t / self.scale(k)

    def unpack(self, theta):
        """(leaves P, natural overrides, linear coefficients) of θ (a tensor)."""
        from sparc.core.physics import NATURAL_LEAF

        P = {k: _t(v) for k, v in self.base.items()}
        nat = {}
        for i, k in enumerate(self.keys):
            if k in ("vx", "vy"):
                P[NATURAL_LEAF[k]] = theta[i]            # the drift's coordinate is its leaf (v = v_max·u)
            elif k in _LOG_KEYS:
                nat[k] = theta[i].exp()
            else:
                nat[k] = theta[i] * self.scale(k)
        return P, nat, theta[self.nl:] / self.st.col_t

    def pack(self, natural: dict, coef) -> np.ndarray:
        return np.asarray([self.coord(k, natural[k]) for k in self.keys]
                          + list(np.asarray(coef, float) * self.st.col), float)

    def theta_t(self, T: dict, coef_scaled):
        """θ (tensor, differentiable) from transformed parameters T and the scaled linear coefficients."""
        import torch

        if not self.keys:
            return coef_scaled
        return torch.cat([torch.stack([self.coord_t(k, T[k]) for k in self.keys]), coef_scaled])

    def index(self, name: str) -> int | None:
        return self.names.index(name) if name in self.names else None

    def dcoord_dleaf(self, k: str, u: float) -> float:
        """d coordinate / d leaf of parameter ``k`` at leaf value ``u`` (1 for the drift)."""
        if k in ("vx", "vy"):
            return 1.0
        m = self.st.model
        sl = m.natural_slope(k, u)
        return sl / max(m.to_natural(k, u), 1e-300) if k in _LOG_KEYS else sl / self.scale(k)


# --------------------------------------------------------------------------- the study
class _Study:
    def __init__(self, grid, frame, y, coords, clusters, cfg_physics: dict, L_init, settings: dict, seed: int = 0):
        from sparc.core.mechanisms import AT_BOUND_FRAC, FLOOR_TOL

        self.grid, self.frame, self.coords = grid, frame, np.asarray(coords, float)
        self.y = np.asarray(y, float)
        self.idx = np.flatnonzero(np.isfinite(self.y))
        self.clusters_all = np.asarray(clusters)
        self.cfg = dict(cfg_physics)
        self.L_init = L_init
        self.S = {**DEFAULTS, **{k: v for k, v in settings.items() if v is not None}}
        self.seed = int(seed)
        self.timings: dict[str, float] = {}
        self.warnings: list[str] = []
        self.at_bound_frac, self.floor_tol = AT_BOUND_FRAC, FLOOR_TOL
        self.k_pen, self.v_pen = 0.0, 0.0

    def warn(self, msg: str) -> None:
        if msg in self.warnings:
            return
        log.warning("identifiability: %s", msg)
        self.warnings.append(msg)

    # ------------------------------------------------------------ reference fits
    def _fit(self, prior_scale, init=None, max_iter=None):
        from sparc.core.physics import PhysicsModel

        cfg = dict(self.cfg)
        if max_iter:
            cfg["max_iter"] = int(max_iter)
        m = PhysicsModel(self.grid, cfg, L_init=self.L_init, seed=self.seed)
        return m.fit(self.frame, self.y, self.idx, self.coords, prior_scale=prior_scale, init=init)

    def _perturbed(self, m, k: int) -> dict:
        """A perturbed start (leaf values) around the default one: k even short and weak, k odd long and strong."""
        from sparc.core.physics import NATURAL_LEAF

        P = {n: float(v.detach()) for n, v in m._init_params((0.0, 0.0)).items()}
        lo, hi = math.exp(m.L_lo), math.exp(m.L_hi)
        L0 = float(self.L_init or 3.0 * m.dx)
        nat = ({"L": L0 / 2.5, "s": 0.25, "kc": 0.5, "c0": 0.3, "cw": 0.06, "a1": 0.15, "a1n": 0.25,
                "e1": 0.5, "e2": 0.5, "e3": 0.5, "w": 20.0} if k % 2 == 0 else
               {"L": L0 * 2.5, "s": 0.85, "kc": 8.0, "c0": 0.6, "cw": 0.2, "a1": 0.6, "a1n": 0.75,
                "e1": 4.0, "e2": 3.0, "e3": 3.0, "w": 150.0})
        nat["L"] = min(max(nat["L"], lo * 1.05), hi / 1.05)
        for key, val in nat.items():
            P[NATURAL_LEAF[key]] = m.to_leaf(key, val)
        if m.v_init is not None:
            P["ux"], P["uy"] = m.v_init[0] / m.v_max, m.v_init[1] / m.v_max
        return P

    def reference(self, which: str, warm: dict | None = None):
        """Multi-started, polished fit: ``which`` = "penalized" (the run's objective) or "data_only"."""
        prior_scale = None if which == "penalized" else 0.0
        t0 = time.perf_counter()
        n_starts = max(1, int(self.S["ref_starts"]))
        with progress.task("reference_fit", key=which, unit="physics_start") as sp:
            first = self._fit(prior_scale)
            starts: list = [warm] if warm else []
            starts += [self._perturbed(first, k) for k in range(max(0, n_starts - 1 - len(starts)))]
            fits = [first]
            progress.tick(1, 1 + len(starts), unit="physics_start", loss=first.fit_info["loss"])
            for k, init in enumerate(starts):
                progress.check_cancel()
                fits.append(self._fit(prior_scale, init=init))
                progress.tick(k + 2, 1 + len(starts), unit="physics_start", loss=fits[-1].fit_info["loss"])
            ok = [m for m in fits if m.a > 0] or fits
            best = min(ok, key=lambda m: m.fit_info["loss"])
            losses = [float(m.fit_info["loss"]) for m in fits]
            progress.check_cancel()
            pol = self._fit(prior_scale, init=best.leaves, max_iter=int(self.S["polish_iter"]))
            if pol.a > 0 and pol.fit_info["loss"] <= best.fit_info["loss"]:
                best = pol
            sp.metrics.update(loss=float(best.fit_info["loss"]), n_starts=len(fits))
        self.timings[f"reference_{which}_s"] = round(self.timings.get(f"reference_{which}_s", 0.0)
                                                     + time.perf_counter() - t0, 2)
        return best, losses + [float(pol.fit_info["loss"])]

    # ------------------------------------------------------------ likelihood pieces
    def drift_l(self, P: dict):
        """The drift penalty v_penalty·|u|² of the data-only objective in log-likelihood units, k̂·v_penalty·|u|²
        with k̂ = n·var(y)/(2·MSE(θ̂)) (so that −k̂·ΔJ ≈ Δl_I to first order); 0 without advection.  The study's
        log-likelihood l_D = l_I − this is the objective the data-only fits maximise (module docstring)."""
        if not self.k_pen:
            return 0.0
        return self.k_pen * self.v_pen * (P["ux"] ** 2 + P["uy"] ** 2)

    def loglik(self, space: _Space, theta):
        import torch

        P, nat, coef = space.unpack(theta)
        r = self.terms.residuals(P, coef, nat or None)
        return -0.5 * self.n * torch.log((r * r).sum() / self.n) - self.drift_l(P)

    def hessian(self, space: _Space, theta0: np.ndarray) -> np.ndarray:
        import torch

        H = torch.autograd.functional.hessian(lambda th: self.loglik(space, th), _t(theta0))
        return _sym(H.detach().numpy())

    def jacobian(self, space: _Space, theta0: np.ndarray) -> np.ndarray:
        """J = ∂r/∂θ (n × p), one Jacobian-vector product per coordinate."""
        import torch

        th = _t(theta0)

        def res(t):
            P, nat, coef = space.unpack(t)
            return self.terms.residuals(P, coef, nat or None)

        cols = []
        for j in range(space.p):
            e = torch.zeros(space.p, dtype=torch.float64)
            e[j] = 1.0
            cols.append(torch.autograd.functional.jvp(res, th, e)[1].detach().numpy())
        return np.stack(cols, axis=1)

    def scores(self, space: _Space, theta0: np.ndarray) -> np.ndarray:
        """Cluster scores U_b = ∂/∂θ of block b's part of l_I, σ² held at its MLE (rows: clusters)."""
        import torch

        th = _t(theta0).clone().requires_grad_(True)
        P, nat, coef = space.unpack(th)
        r = self.terms.residuals(P, coef, nat or None)
        s2 = float((r.detach() ** 2).mean())
        U = np.empty((len(self.cl_idx), space.p))
        for g, ix in enumerate(self.cl_idx):
            gb, = torch.autograd.grad((r[ix] ** 2).sum(), th, retain_graph=g < len(self.cl_idx) - 1)
            U[g] = -gb.detach().numpy() / (2.0 * s2)
        return U

    def score_cov(self, U: np.ndarray, mode: str = "blocks") -> np.ndarray:
        """V from the cluster scores.  "blocks": Σ_b (U_b − Ū)(U_b − Ū)ᵀ (Chandler & Bate 2007).  "neighbours":
        that plus the positive part, relative to it (:func:`_pos_part_rel`, so invariant to the parameter
        coordinates), of the cross-products of adjacent blocks Σ_{b~b′} (U_b − Ū)(U_b′ − Ū)ᵀ
        (:meth:`_adjacency`): a block-level spatial HAC in the manner of Conley (1999) (module docstring)."""
        Ub = U - U.mean(axis=0)
        V = Ub.T @ Ub
        if mode == "neighbours":
            if getattr(self, "adj", None) is None:
                raise ValueError("score_covariance 'neighbours' needs the block size (block_m)")
            V = V + _pos_part_rel(Ub.T @ self.adj @ Ub, V)
        elif mode != "blocks":
            raise ValueError(f"unknown score_covariance {mode!r}; use one of {SCORE_COVARIANCES}")
        return _sym(V)

    def curvatures(self, H: np.ndarray, U: np.ndarray) -> dict:
        """{mode: Σ_naive, Σ_rob, H_A (and the clipped H_I)} from H_I and the cluster scores, for every score
        covariance this study computes (:attr:`modes`)."""
        Sn, Hc, wmin = _neg_def_inverse(H)
        out = {}
        for mode in self.modes:
            V = self.score_cov(U, mode)
            out[mode] = {"H": Hc, "Sn": _sym(Sn), "Sr": _sym(Sn @ V @ Sn), "HA": _sym(-Hc @ _pos_inverse(V) @ Hc),
                         "V": V, "min_eig": wmin, "neg_def": bool(wmin > 1e-10)}
        return out

    def ratio(self, C: dict, delta: np.ndarray) -> float | None:
        """δᵀH_Aδ / δᵀH_Iδ (None when δ carries no curvature)."""
        den = float(delta @ C["H"] @ delta)
        if not den < -1e-300:
            return None
        return float(delta @ C["HA"] @ delta) / den

    def l_adj_t(self, lI, theta, HA, HI, th_hat):
        """l_A = l_I(θ̂) + [δᵀH_Aδ/δᵀH_Iδ]·[l_I(θ) − l_I(θ̂)] (torch; δ = θ − θ̂)."""
        import torch

        d = theta - th_hat
        den = d @ HI @ d
        den = torch.where(den < -1e-300, den, torch.full_like(den, -1e-300))
        return self.l_hat + (d @ HA @ d) / den * (lI - self.l_hat)

    def max_adj(self, space: _Space, C: dict, th_hat: np.ndarray, psi: str, value: float, starts: list,
                vspec: tuple | None = None) -> tuple[float, dict, np.ndarray]:
        """max of l_A over every parameter but ψ (= ``value``; |v| fixed through its angle when ``vspec`` =
        (|v|, angle0)) by L-BFGS over the leaves and scaled linear coefficients, from the best of ``starts``
        (leaf dicts with a "_coef" entry): the adjusted profile.  Returns (l_A, leaves, θ)."""
        import torch

        from sparc.core.physics import NATURAL_LEAF, _lbfgs

        m = self.model
        drift = vspec is not None
        free = [NATURAL_LEAF[k] for k in space.keys if k != psi and not (drift and k in ("vx", "vy"))]
        # parameters held on a fit bound at θ̂ are maximised too: they enter l_I (a true profile maximises over every
        # other parameter), not δ; the bound is active near θ̂, so the curvature in the main coordinates is that of
        # the profile over them.  Flat ones are not: they lie in a direction of H_I without curvature (e.g. e₀
        # confounded with the gain a), so moving them would move δ along it at no cost in l_I and defeat the
        # adjustment; they keep the start's value (an l_I-profile point's own maximiser)
        skip = {"ux", "uy"} if drift else {NATURAL_LEAF[psi]}
        free += [NATURAL_LEAF[k] for k in (getattr(self, "held_bound", None) or {})
                 if NATURAL_LEAF[k] in self.active and NATURAL_LEAF[k] not in free and NATURAL_LEAF[k] not in skip]
        const = [n for n in self.active if n not in free and n not in skip]       # flat: the start's values
        HA, HI, thh = _t(C["HA"]), _t(C["H"]), _t(th_hat)
        natural = None if drift else {psi: float(value)}

        def build(leaves: dict, coef_s, alpha=None, fixed=None):
            P = {k: _t(v) for k, v in space.base.items()}
            P.update(fixed or {})
            for n in free:
                P[n] = leaves[n]
            if drift:
                P["ux"] = float(vspec[0]) * torch.cos(alpha) / m.v_max
                P["uy"] = float(vspec[0]) * torch.sin(alpha) / m.v_max
            T = self.terms.transform(P, natural)
            th = space.theta_t(T, coef_s)
            r = self.terms.residuals(P, coef_s / self.col_t, natural)
            lI = -0.5 * self.n * torch.log((r * r).sum() / self.n) - self.drift_l(P)
            return self.l_adj_t(lI, th, HA, HI, thh), th

        def fixed_of(st_):
            return {n: _t(st_.get(n, space.base.get(n))) for n in const if n in st_ or n in space.base}

        def run(st_: dict, iters: int):
            """L-BFGS on −l_A from ``st_`` for ``iters`` iterations: (l_A, state, θ) at the end (and at the
            start: the better of the two, so a failed line search never loses a start's value)."""
            fx = fixed_of(st_)
            x = {n: _t(st_[n]).clone().requires_grad_(True) for n in free}
            c = _t(np.asarray(st_["_coef"], float) * self.col).clone().requires_grad_(True)
            params = list(x.values()) + [c]
            alpha = None
            if drift:
                alpha = _t(st_.get("_alpha", vspec[1])).clone().requires_grad_(True)
                params.append(alpha)
            with torch.no_grad():
                v0, th0 = build(x, c, alpha, fx)
            if not math.isfinite(float(v0)):
                return None
            snap0 = (float(v0), _state(x, c, alpha, fx), th0.numpy().copy())
            if iters > 0:
                _lbfgs(params, lambda: -build(x, c, alpha, fx)[0], int(iters))
            with torch.no_grad():
                v1, th1 = build(x, c, alpha, fx)
            if not (math.isfinite(float(v1)) and float(v1) >= snap0[0]):
                return snap0
            return float(v1), _state(x, c, alpha, fx), th1.numpy().copy()

        def _state(x, c, alpha, fx):
            out = {n: float(v.detach()) for n, v in x.items()}
            out.update({n: float(v) for n, v in fx.items()})
            out["_coef"] = (c.detach().numpy() / self.col).tolist()
            if drift:
                out["_alpha"] = float(alpha.detach())
            return out

        # every distinct start is run for a few iterations (_screen_iter), the best is continued for the full count
        # (the physics fit's own multistart scheme), and the best value seen is kept
        uniq = _distinct([st_ for st_ in starts if st_ is not None], free + (["_alpha"] if drift else []) + ["_coef"])
        iters = int(self.S["adj_iter"])
        if len(uniq) == 1:
            res = [run(uniq[0], iters)]
        else:
            res = [run(st_, _screen_iter(iters)) for st_ in uniq]
            ok = [q for q in res if q is not None]
            if ok:
                top = max(ok, key=lambda q: q[0])
                res.append(run(top[1], iters))
        ok = [q for q in res if q is not None]
        if not ok:
            raise FloatingPointError("no finite start for the adjusted profile")
        val, out, th = max(ok, key=lambda q: q[0])
        return float(val), out, th

    # ------------------------------------------------------------ setup at the data-only optimum
    def setup(self, model) -> None:
        import torch

        self.model = model
        self.terms = model.fit_terms(self.frame, self.y, self.idx, self.coords)
        self.n = self.terms.n
        cl = self.clusters_all[self.terms.idx]
        labels, inv = np.unique(cl, return_inverse=True)
        self.cl_idx = [torch.as_tensor(np.flatnonzero(inv == g)) for g in range(labels.size)]
        self.cl_sizes = [int(len(ix)) for ix in self.cl_idx]
        self.adj = self._adjacency(inv, labels.size)
        self.modes = ["blocks"] + (["neighbours"] if self.adj is not None else [])
        want = self.S.get("score_covariance") or "blocks"
        if want not in SCORE_COVARIANCES:
            raise ValueError(f"unknown score_covariance {want!r}; use one of {SCORE_COVARIANCES}")
        if want not in self.modes:
            self.warn("score_covariance 'neighbours' needs the block size: the block covariance is used")
        self.primary = want if want in self.modes else "blocks"
        P = {k: _t(v) for k, v in model.leaves.items()}
        with torch.no_grad():
            phi, _ = self.terms.phi(self.terms.transform(P))
            D = self.terms.design(phi)
        self.col = np.sqrt(np.mean(D.numpy() ** 2, axis=0))
        self.col[self.col <= 0] = 1.0
        self.col_t = _t(self.col)
        self.active = list(self.terms.leaf_names)
        # with advection fitted the data-only objective keeps the drift penalty: the study compares and profiles
        # that objective, in log-likelihood units (drift_l), so a point is never "better" only by trading penalty
        self.v_pen = float(model.cfg.get("v_penalty", 0.0)) if model.fit_advection else 0.0
        if self.v_pen > 0:
            with torch.no_grad():
                mse0 = float(self.terms.evaluate(P, prior_scale=0.0)[2])
            self.k_pen = self.n * self.terms.var_y / (2.0 * max(mse0, 1e-300))
        else:
            self.k_pen = 0.0

    def _adjacency(self, inv: np.ndarray, G: int) -> np.ndarray | None:
        """Off-diagonal adjacency (0/1) of the clusters as square blocks of side ``block_m`` (edge or corner
        neighbours), from the block each cluster's points fall in; None without a block size."""
        B = self.S.get("block_m")
        if not B:
            return None
        c = self.coords[self.terms.idx]
        x0, y0 = self.S.get("block_origin") or (float(self.coords[:, 0].min()), float(self.coords[:, 1].min()))
        bx = np.floor((c[:, 0] - x0) / B).astype(np.int64)
        by = np.floor((c[:, 1] - y0) / B).astype(np.int64)
        cen = np.array([[np.median(bx[inv == g]), np.median(by[inv == g])] for g in range(G)])
        A = (np.abs(cen[:, None, 0] - cen[None, :, 0]) <= 1) & (np.abs(cen[:, None, 1] - cen[None, :, 1]) <= 1)
        np.fill_diagonal(A, False)
        return A.astype(float)

    def bound_side(self, key: str, value: float) -> str | None:
        """"lower"/"upper" when a natural value sits on its fit bound (the panel's tolerances), else None."""
        lo, hi, kind = self.model.natural_bounds(key)
        if kind == "free":
            return None
        if kind == "floor":
            return "lower" if value <= lo + self.floor_tol.get(key, 1e-3) else None
        a, b, x = (lo, hi, value) if kind == "lin" else (math.log(lo), math.log(hi), math.log(max(value, 1e-300)))
        tol = self.at_bound_frac * (b - a)
        if x <= a + tol:
            return "lower"
        if x >= b - tol:
            return "upper"
        return None

    def optimum(self, model, prior_scale) -> dict:
        """Leaves, natural values, linear coefficients, mse and l_I of a fitted reference model."""
        import torch

        from sparc.core.physics import LEAF_NATURAL

        P = {k: _t(v) for k, v in model.leaves.items()}
        with torch.no_grad():
            loss, coef, mse, T, _ = self.terms.evaluate(P, prior_scale=prior_scale)
        nat = {k: float(v) for k, v in T.items()}
        at = {}
        for n in self.active:
            k = LEAF_NATURAL[n]
            if k in ("vx", "vy"):
                continue
            side = self.bound_side(k, nat[k])
            if side:
                at[k] = side
        mse = float(mse)
        l_i = -0.5 * self.n * math.log(mse)
        return {"leaves": dict(model.leaves), "natural": nat, "coef": coef.numpy().astype(float), "mse": mse,
                "loss": float(loss), "l": l_i - float(self.drift_l(P)), "l_I": l_i, "at_bound": at,
                "ef_saturated_frac": self.ef_saturation(T)}

    def ef_saturation(self, T: dict) -> float:
        """Share of points whose evaporative-fraction logit z (``PhysicsModel._q_points_torch``) has |z| > 20,
        where σ′(z) < 2·10⁻⁹: there the EF is a step function of the covariates and its slopes carry no local
        information (an operational diagnostic)."""
        import torch

        tf, c = self.terms.tf, self.model._centre
        n = int(next(iter(tf.values())).shape[0]) if tf else int(self.grid.n_points)
        with torch.no_grad():
            z = T["e0"] * torch.ones(n, dtype=torch.float64)
            if "ndvi" in tf:
                z = z + T["e1"] * (tf["ndvi"] - c["ndvi"])
            if "canopy" in tf:
                z = z + T["e2"] * (tf["canopy"] - c["canopy"])
            if "impervious" in tf:
                z = z - T["e3"] * (tf["impervious"] - c["impervious"])
        return float((z.abs() > 20.0).double().mean())


# --------------------------------------------------------------------------- analysis
#: Smallest accepted ``max_points`` (profile points per side): one point cannot bracket a crossing, so a side
#: would be called open by construction.
MIN_MAX_POINTS = 2
#: Smallest accepted ``draws``: power-scaling's CJS distance and the importance ESS need a sample (operational).
MIN_DRAWS = 100


def normalise_params(params) -> list[str] | None:
    """Report keys of :data:`PARAMS` for the requested names (aliases resolved, duplicates dropped), or None for
    "every active parameter".  Raises ValueError naming the unknown ones."""
    if params is None:
        return None
    if isinstance(params, str):
        params = [p for p in params.split(",")]
    names = [str(p).strip() for p in params if str(p).strip()]
    if not names:
        return None
    unknown = [p for p in names if ALIASES.get(p, p) not in PARAMS]
    if unknown:
        raise ValueError(f"unknown physics parameter(s) {', '.join(repr(p) for p in unknown)}; known: "
                         f"{', '.join(PARAMS)} (aliases: {', '.join(sorted(ALIASES))})")
    out: list[str] = []
    for p in names:
        k = ALIASES.get(p, p)
        if k not in out:
            out.append(k)
    return out


def check_arguments(params=None, max_points=None, draws=None, posterior: bool = True, coarse=None) -> list[str] | None:
    """Validate the study's arguments before any data are loaded or fitted (fail fast): known parameter names,
    ``max_points`` ≥ :data:`MIN_MAX_POINTS`, ``draws`` ≥ :data:`MIN_DRAWS` (when the posterior is computed), a
    positive ``coarse``.  Returns the normalised parameter keys (:func:`normalise_params`)."""
    keys = normalise_params(params)
    if max_points is not None:
        if isinstance(max_points, bool) or int(max_points) != max_points or int(max_points) < MIN_MAX_POINTS:
            raise ValueError(f"max_points must be an integer ≥ {MIN_MAX_POINTS} (profile points per side; one point "
                             f"cannot bracket the 95% threshold), got {max_points!r}")
    if draws is not None and posterior:
        if isinstance(draws, bool) or int(draws) != draws or int(draws) < MIN_DRAWS:
            raise ValueError(f"draws must be an integer ≥ {MIN_DRAWS} (Laplace-posterior draws for power-scaling), "
                             f"got {draws!r}")
    if coarse is not None and not (float(coarse) > 0):
        raise ValueError(f"coarse must be a positive cell size in metres, got {coarse!r}")
    return keys


def _requested(params, active_keys: list[str]) -> list[str]:
    keys = normalise_params(params)
    return list(active_keys) if keys is None else keys


def analyse(grid, frame, y, coords, clusters, cfg_physics: dict, *, L_init: float | None = None,
            params=None, max_points: int | None = None, draws: int | None = None, seed: int = 0,
            units: str | None = None, settings: dict | None = None, posterior: bool = True,
            fold_params: list | None = None, context: dict | None = None) -> dict:
    """The identifiability report of a physics model on (grid, frame, y, coords) with ``clusters`` the
    independent spatial blocks of each point (the run's ``folds.block_id``).  ``cfg_physics`` is the run's
    physics section as the run fitted it (advection selection and scheme applied).  ``params``: report keys of
    :data:`PARAMS` (default every active nonlinear parameter).  ``posterior=False`` skips steps 6–8 (the
    Laplace posterior, power-scaling and conflict).  Returns the JSON-ready report (module docstring)."""
    from sparc.core.physics import _torch_threads

    S = {**(settings or {})}
    params = check_arguments(params, max_points if max_points is not None else S.get("max_points"),
                             draws if draws is not None else S.get("draws"), posterior)
    if max_points is not None:
        S["max_points"] = int(max_points)
    if draws is not None:
        S["draws"] = int(draws)
    S["seed"] = int(seed)
    ctx = context or {}
    if ctx.get("block_m") and not S.get("block_m"):
        S["block_m"] = float(ctx["block_m"])
    if ctx.get("block_origin") and not S.get("block_origin"):
        S["block_origin"] = [float(v) for v in ctx["block_origin"]]
    cfg_physics = dict(cfg_physics)
    if S.get("num_threads"):
        cfg_physics["num_threads"] = int(S.pop("num_threads"))
    st = _Study(grid, frame, y, coords, clusters, cfg_physics, L_init, S, seed=seed)
    with _torch_threads(cfg_physics.get("num_threads", 1)):
        return _analyse(st, params, units, posterior, fold_params, context or {})


def _analyse(st: _Study, params, units, posterior: bool, fold_params, context: dict) -> dict:
    from sparc.core.physics import LEAF_NATURAL

    t_all = time.perf_counter()
    pen_model, pen_losses = st.reference("penalized")
    dat_model, dat_losses = st.reference("data_only", warm=pen_model.leaves)
    st.setup(dat_model)
    dat = st.optimum(dat_model, 0.0)
    pen = st.optimum(pen_model, None)
    st.l_hat = dat["l"]
    m = dat_model
    active_keys: list[str] = []
    for n in st.active:
        k = LEAF_NATURAL[n]
        k = "v" if k in ("vx", "vy") else _NAT_KEY[k]
        if k not in active_keys:
            active_keys.append(k)
    active_keys.sort(key=list(PARAMS).index)
    keys = _requested(params, active_keys)
    G = len(st.cl_idx)
    lin_names = list(st.terms.linear_names)
    out = {"schema": SCHEMA, "status": "ok", "n": st.n, "n_clusters": G,
           "cluster": {"kind": context.get("cluster_kind", "cv_block"), "block_m": context.get("block_m"),
                       "sizes_min": int(min(st.cl_sizes)), "sizes_median": float(np.median(st.cl_sizes)),
                       "sizes_max": int(max(st.cl_sizes))},
           "coarse_m": context.get("coarse_m"), "units": units, "window": m.window,
           "advection": {"fit": bool(m.fit_advection), "scheme": m._scheme(),
                         "selection": context.get("advection_selection")},
           "reference": {"penalized": _ref_summary(pen, pen_losses, pen_model, active_keys, lin_names),
                         "data_only": _ref_summary(dat, dat_losses, dat_model, active_keys, lin_names)},
           "settings": {**st.S, "chi2_95": CHI2_95, "flat_lambda": FLAT_LAMBDA, "psens_threshold": PSENS_THRESHOLD,
                        "conflict_alpha": CONFLICT_ALPHA, "flat_info": FLAT_INFO, "null_eig": NULL_EIG,
                        "posterior": bool(posterior)},
           "covariance": {"computed": list(st.modes), "used": st.primary},
           "references": dict(REFERENCES), "warnings": st.warnings, "timings": st.timings}
    if dat_model.a <= 0 or dat["coef"][0] <= 0:
        st.warn("the data-only fit is anti-physical (a ≤ 0): the physics term is disabled, nothing to profile")
        out.update(status="disabled", parameters=[], linear=[], derived={})
        return out

    # ---- curvature and profiles at the data-only optimum; restart from a better profile point if one is found
    restarts = 0
    while True:
        dat, rows, held_bound, held_flat, partners, prior_spec, best = _at_optimum(st, dat_model, pen, keys,
                                                                                fold_params)
        improve = best["l"] - st.l_hat
        if improve <= RESTART_DL or restarts >= int(st.S["max_restarts"]):
            break
        restarts += 1
        st.warn(f"a profile point of {best['key']} improved on the data-only optimum by Δl = {improve:.3g}: "
                f"refitting the data-only reference from it (restart {restarts})")
        with progress.task("reference_fit", key=f"data_only_restart{restarts}"):
            cand = st._fit(0.0, init=best["leaves"], max_iter=int(st.S["polish_iter"]))
        if cand.a > 0:
            dat_model = cand
            dat_losses = dat_losses + [float(cand.fit_info["loss"])]
        st.setup(dat_model)
        dat = st.optimum(dat_model, 0.0)
        st.l_hat = dat["l"]
    if improve > RESTART_DL:
        st.warn(f"after {restarts} restart(s) a profile point still improves on the data-only optimum by "
                f"Δl = {improve:.3g} ({best['key']} = {best['value']:.4g}): the data-only likelihood keeps "
                "increasing along that direction, so the reference is not a maximum and the profiles are "
                "approximate; the parameters it moves are not identified by the data")
    out["reference"]["data_only"] = _ref_summary(dat, dat_losses, dat_model, active_keys, lin_names)
    out["reference"]["data_only"]["restarts"] = restarts
    out["reference"]["data_only"]["max_profile_improvement_l"] = float(max(improve, 0.0))
    if dat.get("ef_saturated_frac", 0.0) > 0.5:
        st.warn(f"the data-only evaporative fraction is saturated (|logit| > 20) at {dat['ef_saturated_frac']:.0%} of "
                "the points: a step function of the covariates, whose slopes run off toward infinity; the EF "
                "parameters are not identified by the data")
    p_main = st.main.p

    # ---- the run's optimum: linear and derived SEs, prior share, pull
    t0 = time.perf_counter()
    with progress.task("curvature", key="penalized"):
        pen_info = _penalized(st, pen, dat, held_flat)
    st.timings["penalized_s"] = round(time.perf_counter() - t0, 2)
    linear = _linear_rows(st, dat, pen, pen_info)
    for r in rows:
        if not r["_in_model"]:
            continue
        if r["_nat"]:
            r["prior_share_run"] = pen_info["share"].get(r["_nat"])
            if r["_nat"] in pen["at_bound"]:
                r["notes"].append("on its fit bound at the run's (penalized) optimum: no prior share there")
        if r.get("se_robust") and r["estimate_run"] is not None and r["estimate_data"] is not None:
            r["pull"] = (r["estimate_run"] - r["estimate_data"]) / r["se_robust"]

    # ---- calibrated posterior, power-scaling, prior–data conflict
    if posterior:
        t0 = time.perf_counter()
        _posterior(st, rows, dat, prior_spec, held_bound, held_flat)
        st.timings["posterior_s"] = round(time.perf_counter() - t0, 2)
        out["posterior"] = getattr(st, "posterior_info", None)
    for r in rows:
        _conflict(r, prior_spec)

    out.update(hessian={"parameters": st.main.names, "held_at_bound": sorted(_NAT_KEY.get(k, k) for k in held_bound),
                        "held_flat": sorted(_NAT_KEY.get(k, k) for k in held_flat),
                        "flat_partners": {_NAT_KEY.get(k, k): v for k, v in partners.items()},
                        "negative_definite": st.C["neg_def"], "min_eig_unit": st.C["min_eig"],
                        "n_parameters": p_main, "clusters_per_parameter": G / max(p_main, 1)},
               parameters=[_public(r) for r in rows], linear=linear, derived=_derived(linear, units, context))
    st.timings["total_s"] = round(time.perf_counter() - t_all, 2)
    return out


def _at_optimum(st: _Study, dat_model, pen: dict, keys: list[str], fold_params):
    """Curvature (H_I, V, Σ_naive, Σ_rob, H_A), natural-unit SEs and profiles at the data-only optimum of
    ``dat_model``.  Returns (dat, rows, held_bound, held_flat, partners, prior_spec, best profile point)."""
    from sparc.core.physics import LEAF_NATURAL

    m = dat_model
    dat = st.optimum(dat_model, 0.0)
    st.l_hat = dat["l"]
    G = len(st.cl_idx)
    t0 = time.perf_counter()
    with progress.task("curvature", key="data_only"):
        held_bound = set(dat["at_bound"])
        st.held_bound = dict(dat["at_bound"])
        keys0 = [LEAF_NATURAL[n] for n in st.active if LEAF_NATURAL[n] not in held_bound]
        sp0 = _Space(st, dat["leaves"], keys0)
        th0 = sp0.pack(dat["natural"], dat["coef"])
        H0 = st.hessian(sp0, th0)
        U0 = st.scores(sp0, th0)
        flat, partners = _flat_coordinates(st, sp0, H0, dat["mse"])
        keep = [i for i, nm in enumerate(sp0.names) if nm not in flat]
        st.main = _Space(st, dat["leaves"], [k for k in keys0 if k not in flat])
        st.th_main = st.main.pack(dat["natural"], dat["coef"])
        st.Cs = st.curvatures(H0[np.ix_(keep, keep)], U0[:, keep])
        st.C = st.Cs[st.primary]
    st.timings["curvature_s"] = round(st.timings.get("curvature_s", 0.0) + time.perf_counter() - t0, 2)
    p_main = st.main.p
    if G < 2 * p_main:
        st.warn(f"{G} clusters for {p_main} parameters (fewer than two per parameter): the cluster-score "
                "covariance V is poorly estimated, so the sandwich SEs and adjusted profiles are approximate")
    if not st.C["neg_def"]:
        st.warn("H_I is not negative definite at the data-only optimum (smallest eigenvalue of the unit-diagonal "
                f"−H_I {st.C['min_eig']:.3g}): a flat or non-converged direction; its eigenvalues were floored")
    held_flat = set(flat)
    prior_spec = {k: (sc, mu, sd) for k, sc, mu, sd in m.prior_terms(st.terms.tf)}
    cache: dict = {}
    rows = [_row_base(st, key, dat, pen, prior_spec, fold_params, held_bound, held_flat, partners) for key in keys]
    for r in rows:
        if r["_in_model"]:
            _row_se(st, r, dat, cache)
    t0 = time.perf_counter()
    prof = [r for r in rows if r["_in_model"]]
    best = {"l": st.l_hat, "key": None, "value": None, "leaves": None}
    with progress.task("profiles", unit="parameter") as sp:
        for i, r in enumerate(prof):
            progress.check_cancel()
            with progress.task("profile", key=r["key"], unit="profile_point"):
                _profile(st, r, dat, cache)
            progress.tick(i + 1, len(prof), unit="parameter", key=r["key"])
            for q in r["_points"]:
                if q["l"] > best["l"]:
                    best = {"l": q["l"], "key": r["key"], "value": q["value"], "leaves": q["leaves"]}
        for r in prof:
            _finish_profile(st, r, max(best["l"], st.l_hat))
        sp.metrics["n_points"] = sum(len(r["_points"]) for r in prof)
    st.timings["profiles_s"] = round(st.timings.get("profiles_s", 0.0) + time.perf_counter() - t0, 2)
    st.cache = cache
    return dat, rows, held_bound, held_flat, partners, prior_spec, best


def _ref_summary(opt: dict, losses, model, active_keys: list[str], linear_names: list[str]) -> dict:
    """A reference fit for the report: objective, l_I, natural values of the active parameters, linear part."""
    nat = {}
    for k in active_keys:
        if k == "v":
            nat["v"] = float(math.hypot(opt["natural"]["vx"], opt["natural"]["vy"]))
        else:
            nat[k] = float(opt["natural"][PARAMS[k][0]])
    return {"loss": opt["loss"], "mse": opt["mse"], "loglik_independence": opt.get("l_I", opt["l"]),
            "loglik_study": opt["l"],
            "start_losses": [float(x) for x in losses], "natural": nat,
            "linear": {n: float(c) for n, c in zip(linear_names, opt["coef"])},
            "at_bound": sorted(_NAT_KEY.get(k, k) for k in opt["at_bound"]), "a_model": float(model.a),
            "ef_saturated_frac": opt.get("ef_saturated_frac"),
            "train_rmse": float(model.train_rmse), "train_r2": float(model.train_r2)}


def _flat_coordinates(st: _Study, space: _Space, H: np.ndarray, mse: float) -> tuple[list[str], dict]:
    """Nonlinear parameters that carry no information at θ̂: a diagonal of −H_I below :data:`FLAT_INFO` of the
    intercept's information n/σ², then, one at a time, the parameter with the largest weight in a direction of
    the unit-diagonal −H_I with no curvature (|eigenvalue| < :data:`NULL_EIG`).  Returns (natural keys,
    {key: the report keys of that direction})."""
    info_b = st.n / max(mse, 1e-300)
    diag = -np.diag(H)
    flat: list[str] = [nm for i, nm in enumerate(space.names[:space.nl]) if diag[i] <= FLAT_INFO * info_b]
    partners: dict = {}
    for _ in range(space.nl):
        keep = [i for i, nm in enumerate(space.names) if nm not in flat]
        A = -H[np.ix_(keep, keep)]
        d = np.sqrt(np.maximum(np.abs(np.diag(A)), 1e-300))
        w, Q = np.linalg.eigh(_sym(A / np.outer(d, d)))
        j = int(np.argmin(w))
        if not (-1e-6 < w[j] < NULL_EIG):
            break
        v = np.abs(Q[:, j])
        cand = [(v[a], space.names[keep[a]]) for a in range(len(keep)) if keep[a] < space.nl]
        if not cand:
            break
        _, pick = max(cand)
        flat.append(pick)
        partners[pick] = [_NAT_KEY.get(space.names[keep[a]], space.names[keep[a]])
                          for a in range(len(keep)) if v[a] > 0.1 * v.max() and space.names[keep[a]] != pick]
    return flat, partners


def _folds_mean(fold_params, key: str) -> float | None:
    """Mean of a parameter over the folds with the physics term on (``physics.json``)."""
    pk = {"L": "L_m", "v": "v_norm_m"}.get(key, key)
    vals = [_f(p.get(pk)) for p in (fold_params or []) if isinstance(p, dict) and (_f(p.get("a")) or 0) > 0
            and not p.get("fit_warning")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


_PRIOR_SOURCE = {"s": "physics.priors / SOURCE_PRIORS (literature mean, weakly informative)",
                 "a1": "physics.priors / SOURCE_PRIORS (literature mean, weakly informative)",
                 "a1n": "physics.priors / SOURCE_PRIORS (literature mean, weakly informative)",
                 "kc": "KAPPA_PRIOR: log κ ~ N(0, 3²), a weak regulariser",
                 "c0": "shade-sigmoid prior as coded", "cw": "shade-sigmoid prior as coded (on log width)",
                 "e0": "EF prior as coded (sd 5, weak)", "e1": "EF prior as coded (sd 5, weak)",
                 "e2": "EF prior as coded (sd 5, weak)", "e3": "EF prior as coded (sd 5, weak)"}


def _row_base(st: _Study, key: str, dat: dict, pen: dict, prior_spec: dict, fold_params, held_bound: set,
              held_flat: set, partners: dict) -> dict:
    from sparc.core.physics import NATURAL_LEAF

    m = st.model
    nat, label, unit, _ = PARAMS[key]
    row = {"key": key, "label": label, "unit": unit, "_nat": nat, "_in_model": True, "estimate_run": None,
           "estimate_data": None, "estimate_run_folds": _folds_mean(fold_params, key), "at_bound": None,
           "held": None, "se_naive": None, "se_robust": None, "se_robust_neighbours": None, "adjust_factor": None,
           "adjust_factor_neighbours": None, "profile": None, "ci95": None, "ci95_neighbours": None,
           "ci95_naive": None, "ci95_naive_label": None, "threshold": None, "boundary_adjustment": None,
           "ci95_conservative": None, "identifiability": None, "class_by_covariance": None, "summary": None,
           "open_sides": [], "open_reasons": {}, "sides_at_limit": [],
           "pull": None,
           "prior": None, "prior_share_run": None, "posterior": None, "psens": None, "conflict": None,
           "notes": []}
    if key == "v":
        if not m.fit_advection:
            row.update(_in_model=False, identifiability="not_fitted")
            row["notes"].append("advection is not fitted in this run (no wind record, or dropped by the out-of-fold "
                                "test), so v = 0 is fixed")
            return row
        row["estimate_data"] = float(math.hypot(dat["natural"]["vx"], dat["natural"]["vy"]))
        row["estimate_run"] = float(math.hypot(pen["natural"]["vx"], pen["natural"]["vy"]))
        row["notes"].append("no nominal prior: the drift penalty v_penalty·|u|² is a regulariser, kept in both fits")
        return row
    if NATURAL_LEAF[nat] not in st.active:
        role = _NEEDS.get(key)
        why = (f"no {role} role in this run's physics" if role and role not in (m.roles or {}) else
               "not a parameter of this window/shade form" if role else "not a parameter of this model")
        if key in ("s", "kappa_canopy", "shade_c0", "shade_width") and m.window != "day":
            why = "the night source has no shade term"
        row.update(_in_model=False, identifiability="structurally_non_identifiable")
        row["notes"].append(f"does not enter the model ({why})")
        return row
    row["estimate_data"] = float(dat["natural"][nat])
    row["estimate_run"] = float(pen["natural"][nat])
    if nat in held_bound:
        side = dat["at_bound"][nat]
        lo, hi, _ = m.natural_bounds(nat)
        row["at_bound"] = {"side": side, "bound": lo if side == "lower" else hi}
        row["held"] = "bound"
        row["notes"].append(f"on its {side} fit bound at the data-only optimum: held there for H, V and the "
                            "Laplace posterior; profiled from the bound inward only")
    elif nat in held_flat:
        row["held"] = "flat"
        pt = partners.get(nat)
        row["notes"].append("carries no information at the data-only optimum"
                            + (f" (a direction of H_I without curvature, with {', '.join(pt)})" if pt else "")
                            + ": held for H and V")
    spec = prior_spec.get(nat)
    if spec:
        sc, mu, sd = spec
        row["prior"] = {"mean": float(mu), "sd": float(sd), "scale": sc, "source": _PRIOR_SOURCE.get(nat),
                        "nominal": True}
    return row


def _ext(st: _Study, nat: str, dat: dict, cache: dict) -> dict:
    """The main coordinates plus a parameter ``nat`` held on its fit bound (its walk coordinate, finite there): H_I
    and the cluster scores at the boundary point (one-sided derivatives: the model is analytic in the natural
    value, so autograd's derivative at the bound is the one-sided one), whether that curvature is well-conditioned
    (:data:`BOUND_MIN_EIG`, :data:`BOUND_MAX_VIF`), and then ψ's SEs and its scalar adjustment
    c = Σ_naive,ψψ/Σ_rob,ψψ per score covariance (``c``; None when not computable)."""
    if nat in cache:
        return cache[nat]
    sp = _Space(st, dat["leaves"], st.main.keys + [nat])
    th = sp.pack(dat["natural"], dat["coef"])
    H = st.hessian(sp, th)
    j = sp.index(nat)
    flat = bool(abs(H[j, j]) <= FLAT_INFO * st.n / max(dat["mse"], 1e-300))
    convex = bool(not flat and H[j, j] > 0)          # l_I not concave along ψ at the bound
    A = _sym(-H)
    d = np.sqrt(np.maximum(np.abs(np.diag(A)), 1e-300))
    Cu = A / np.outer(d, d)
    w = np.linalg.eigvalsh(Cu)
    min_eig = float(w.min())
    vif = None
    if min_eig > 0:
        vif = float(np.linalg.inv(Cu)[j, j])
    well = bool(not flat and not convex and min_eig >= BOUND_MIN_EIG and vif is not None and vif <= BOUND_MAX_VIF)
    Cs, c = None, None
    if not flat:
        U = st.scores(sp, th)
        Cs = st.curvatures(H, U)
        if well:                       # positive definite: the exact inverse, no eigenvalue floor
            Sn = _sym(np.linalg.inv(A))
            c = {}
            for mode, C in Cs.items():
                Sr = _sym(Sn @ C["V"] @ Sn)
                C.update(Sn=Sn, Sr=Sr)
                c[mode] = float(Sn[j, j] / Sr[j, j]) if Sr[j, j] > 0 else None
            if any(v is None or not math.isfinite(v) for v in c.values()):
                well, c = False, None
    why = None
    if not well:
        why = ("the data carry no information on it at the bound" if flat else
               "l_I is not concave along it at the bound (a positive second derivative)" if convex else
               f"the curvature at the bound is not positive definite (smallest eigenvalue of the unit-diagonal −H_I "
               f"{min_eig:.3g})" if min_eig <= 0 else
               f"the curvature at the bound is ill-conditioned (smallest eigenvalue {min_eig:.3g} < {BOUND_MIN_EIG:g}"
               f"{'' if vif is None else f', variance inflation of ψ {vif:.3g}'})" if min_eig < BOUND_MIN_EIG else
               f"at the bound it is nearly confounded with other parameters (variance inflation {vif:.3g} > "
               f"{BOUND_MAX_VIF:g})")
    out = {"space": sp, "theta": th, "Cs": Cs, "j": j, "flat": flat, "well": well, "c": c, "min_eig": min_eig,
           "vif": vif, "why": why}
    cache[nat] = out
    return out


def _dnat(space: _Space, k: str, nat_value: float) -> float:
    """d natural / d coordinate."""
    return float(nat_value) if k in _LOG_KEYS else space.scale(k)


def _row_se(st: _Study, r: dict, dat: dict, cache: dict) -> None:
    """Naive and sandwich SEs in natural units (delta method) and c_j = Σ_naive,jj / Σ_rob,jj, for each score
    covariance (``se_robust``/``adjust_factor``: blocks; ``…_neighbours``: adjacent blocks included)."""
    m, main, Cs = st.model, st.main, st.Cs
    if r["key"] == "v":
        iu, iv = main.index("vx"), main.index("vy")
        vx, vy = dat["natural"]["vx"], dat["natural"]["vy"]
        nv = math.hypot(vx, vy)
        if iu is None or iv is None or nv < 1e-6:
            r["notes"].append("|v| = 0: no delta-method SE for |v| at zero drift")
            return
        g = np.zeros(main.p)
        g[iu], g[iv] = m.v_max * vx / nv, m.v_max * vy / nv
        var = {mode: (float(g @ C["Sn"] @ g), float(g @ C["Sr"] @ g)) for mode, C in Cs.items()}
    elif r["held"]:
        e = _ext(st, r["_nat"], dat, cache)
        if r["held"] == "bound":
            r["boundary_adjustment"] = {"computable": e["well"], "c": e["c"], "min_eig_unit": e["min_eig"],
                                        "vif": e["vif"], "reason": e["why"]}
        if not e["well"]:
            r["notes"].append(f"no SE: {e['why'] or 'no well-conditioned curvature'}")
            return
        j, d = e["j"], _dnat(e["space"], r["_nat"], dat["natural"][r["_nat"]])
        var = {mode: (float(C["Sn"][j, j]) * d ** 2, float(C["Sr"][j, j]) * d ** 2) for mode, C in e["Cs"].items()}
    else:
        j, d = main.index(r["_nat"]), _dnat(main, r["_nat"], dat["natural"][r["_nat"]])
        var = {mode: (float(C["Sn"][j, j]) * d ** 2, float(C["Sr"][j, j]) * d ** 2) for mode, C in Cs.items()}
    vn = var["blocks"][0]
    r["se_naive"] = math.sqrt(max(vn, 0.0))
    for mode, (_, vr) in var.items():
        r["se_robust" + _SUFFIX[mode]] = math.sqrt(max(vr, 0.0))
        r["adjust_factor" + _SUFFIX[mode]] = vn / vr if vr > 0 else None


# --------------------------------------------------------------------------- profiles
def _walk_bounds(st: _Study, r: dict) -> tuple[float | None, float | None]:
    """Edges of the walk coordinate: a sigmoid-mapped parameter stops where the panel calls it "at the bound"
    (AT_BOUND_FRAC of the range on the transform's scale), a floored one at the floor, |v| at 0."""
    if r["key"] == "v":
        return 0.0, None
    lo, hi, kind = st.model.natural_bounds(r["_nat"])
    if kind in ("lin", "log"):
        a, b = (lo, hi) if kind == "lin" else (math.log(lo), math.log(hi))
        tol = st.at_bound_frac * (b - a)
        return a + tol, b - tol
    ef = _ef_walk_limit(st, r["_nat"])
    if kind == "floor":
        return float(lo), ef
    if ef is not None:
        return -ef, ef
    return None, None


def _ef_walk_limit(st: _Study, nat: str) -> float | None:
    """How far an EF parameter is walked: until its own term alone reaches |logit| = :data:`EF_WALK_LOGIT`
    (e₀ itself; a slope times the largest centred covariate).  Beyond it the evaporative fraction is 0 or 1 to
    within 10⁻¹³ and, past |logit| ≈ 37, σ rounds to 1 in float64 so the source vanishes numerically: the model
    is not defined there, so a profile cannot test it."""
    if nat == "e0":
        return EF_WALK_LOGIT
    role = {"e1": "ndvi", "e2": "canopy", "e3": "impervious"}.get(nat)
    if role is None or role not in st.terms.tf:
        return None
    dev = float((st.terms.tf[role] - st.model._centre[role]).abs().max())
    return EF_WALK_LOGIT / dev if dev > 0 else None


def _to_walk(r: dict, value: float) -> float:
    return math.log(max(value, 1e-300)) if PARAMS[r["key"]][3] == "log" else float(value)


def _from_walk(r: dict, x: float) -> float:
    return math.exp(x) if PARAMS[r["key"]][3] == "log" else float(x)


def _quadratic_start(st: _Study, space: _Space, C: dict, th_hat: np.ndarray, psi: str, value: float,
                     base: dict) -> dict | None:
    """Leaves on the adjusted profile path of the quadratic approximation, θ̂ + Δψ·Σ_rob e_ψ / [Σ_rob]_ψψ: a
    start for the l_A maximisation."""
    from sparc.core.physics import NATURAL_LEAF

    j = space.index(psi)
    if j is None or not (math.isfinite(C["Sr"][j, j]) and C["Sr"][j, j] > 0):
        return None
    m = st.model
    th = th_hat + (space.coord(psi, value) - th_hat[j]) * C["Sr"][:, j] / C["Sr"][j, j]
    if not np.all(np.isfinite(th)):
        return None
    out = dict(base)
    clipped = False
    for i, k in enumerate(space.keys):
        if k == psi:
            continue
        x = float(th[i])
        lo_x, hi_x = _coord_limits(st, space, k)
        if (lo_x is not None and x < lo_x) or (hi_x is not None and x > hi_x):
            # an indefinite or singular H_I (eigenvalues floored) sends the quadratic path far outside the fit
            # bounds: clip it there instead of overflowing (the start is only a candidate; the best one is kept)
            x = min(max(x, lo_x if lo_x is not None else x), hi_x if hi_x is not None else x)
            clipped = True
        if k in ("vx", "vy"):
            out[NATURAL_LEAF[k]] = x
            continue
        nat = space.natural(k, x)
        lo, hi, kind = m.natural_bounds(k)
        if kind == "floor":
            nat = max(nat, 1e-9 * m._leaf_scale(k))
        elif kind in ("lin", "log"):
            nat = min(max(nat, lo + 1e-9 * (hi - lo)), hi - 1e-9 * (hi - lo))
        out[NATURAL_LEAF[k]] = m.to_leaf(k, nat)
    if clipped and not C.get("neg_def", True):
        st.warn("H_I is not negative definite, so the quadratic profile path left the fit bounds; those profile "
                "starts were clipped to the bounds (the best start of each profile point is kept)")
    out["_coef"] = (th[space.nl:] / st.col).tolist()
    return out


def _coord_limits(st: _Study, space: _Space, k: str) -> tuple[float | None, float | None]:
    """Finite limits of parameter ``k``'s coordinate in ``space``: its fit bounds (on the log scale for L, κ and
    the width), the evaporative-fraction walk limit for e₀ and the slopes (:func:`_ef_walk_limit`), |u| ≤ 1 for the
    drift's leaves (|v| ≤ v_max)."""
    if k in ("vx", "vy"):
        return -1.0, 1.0
    lo, hi, kind = st.model.natural_bounds(k)
    if kind == "log":
        return math.log(lo), math.log(hi)
    if kind == "lin":
        return space.coord(k, lo), space.coord(k, hi)
    ef = _ef_walk_limit(st, k) if k in ("e0", "e1", "e2", "e3") else None
    if kind == "floor":
        return space.coord(k, lo), (space.coord(k, ef) if ef is not None else None)
    return (-ef, ef) if ef is not None else (None, None)


def _point(st: _Study, r: dict, value: float, warms, dat: dict, cache: dict) -> dict:
    """One profile point: ψ fixed at ``value``, every other parameter refitted (data-only objective, linear
    coefficients by variable projection); then the adjusted profile value, l_A maximised over the other
    parameters (:meth:`_Study.max_adj`).  Both maximisations start from several candidates and keep the best:
    for l_I, ``warms`` (the previous point's optimum; a neighbour's when repairing), the reference optimum with ψ
    substituted, and the first with the parameters held on a bound released; for l_A, the l_I-profile point,
    each warm's l_A optimum, the quadratic adjusted path and the reference optimum with ψ substituted."""
    import torch

    from sparc.core.physics import NATURAL_LEAF, _lbfgs

    m = st.model
    nat = r["_nat"]
    drift = r["key"] == "v"
    psi_leaves = {"ux", "uy"} if drift else {NATURAL_LEAF[nat]}
    # a profile maximises over every other parameter: those held on a fit bound or flat at θ̂ included (within
    # their bounds: the leaves map onto them)
    free = [n for n in st.active if n not in psi_leaves]
    natural = None if drift else {nat: float(value)}

    def fit_from(start: dict, iters: int):
        P = {k: _t(v) for k, v in dat["leaves"].items()}
        for n in free:
            P[n] = _t(start[n]).clone().requires_grad_(True)
        params = [P[n] for n in free]
        alpha = None
        if drift:
            a0 = start.get("_alpha")
            if a0 is None:
                a0 = math.atan2(start["uy"], start["ux"]) if math.hypot(start["ux"], start["uy"]) > 0 else 0.0
            alpha = _t(a0).clone().requires_grad_(True)
            params.append(alpha)

        def set_v():
            if drift:
                P["ux"] = value * torch.cos(alpha) / m.v_max
                P["uy"] = value * torch.sin(alpha) / m.v_max

        def objective():
            set_v()
            return st.terms.evaluate(P, prior_scale=0.0, natural=natural)[0]

        if params and iters > 0:
            _lbfgs(params, objective, int(iters))
        with torch.no_grad():
            set_v()
            res = st.terms.evaluate(P, prior_scale=0.0, natural=natural)
        return float(res[0]), P, alpha, res

    def as_start(P, alpha):
        out = {n: float(P[n].detach()) for n in free}
        if drift:
            out["_alpha"] = float(alpha.detach())
        return out

    warms = [warms] if isinstance(warms, dict) else [w for w in warms if w is not None]
    ref = {**dat["leaves"], "_coef": list(dat["coef"])}
    if drift:
        ref["_alpha"] = math.atan2(dat["leaves"]["uy"], dat["leaves"]["ux"])
    # the parameters held on a bound sit where their leaf has a vanishing gradient, so a start cannot move them:
    # also start with them released inside their bounds (from the warm start and from the reference)
    starts = warms + [ref] + [x for x in (_released(st, warms[0], psi_leaves) if warms else None,
                                          _released(st, ref, psi_leaves)) if x is not None]
    starts = _distinct(starts, free + (["_alpha"] if drift else []))
    iters = int(st.S["profile_iter"])
    if len(starts) == 1:
        cands = [fit_from(starts[0], iters)]
    else:                                   # each start for a few iterations, the best continued
        cands = [fit_from(x, _screen_iter(iters)) for x in starts]
        top = min(cands, key=lambda c: c[0])
        cands.append(fit_from(as_start(top[1], top[2]), iters))
    _, P, alpha, (loss, coef, mse, T, _) = min(cands, key=lambda c: c[0])
    leaves = {k: float(v.detach()) for k, v in P.items()}
    if drift:
        leaves["_alpha"] = float(alpha.detach())
    else:
        leaves[NATURAL_LEAF[nat]] = m.to_leaf(nat, float(value))     # ψ itself (a restart starts here)
    leaves["_coef"] = coef.numpy().tolist()
    with torch.no_grad():
        lval = -0.5 * st.n * math.log(float(mse)) - float(st.drift_l(P))
    lam_i = max(2.0 * (st.l_hat - lval), 0.0)
    if r["held"] == "bound":
        # a boundary maximum (Self & Liang 1987): the scalar adjustment Λ_A = c·Λ_I with c = Σ_naive,ψψ/Σ_rob,ψψ
        # from the well-conditioned curvature at the bound, or none (Λ_A = Λ_I, the row is not assessable)
        e = _ext(st, nat, dat, cache)
        out = {"value": float(value), "l": lval, "loss": float(loss), "leaves": leaves, "lam_i": lam_i,
               "R": {}, "R_ipath": {}, "lam_a": {}, "lam_a_ipath": {}}
        for mode in st.modes:
            cm = e["c"][mode] if e["c"] else None
            out["R"][mode] = out["R_ipath"][mode] = cm
            out["lam_a"][mode] = out["lam_a_ipath"][mode] = lam_i * cm if cm is not None else lam_i
        return out
    if r["held"]:
        e = _ext(st, nat, dat, cache)
        sp, th_hat, Cs, flat = e["space"], e["theta"], e["Cs"], e["flat"]
    else:
        sp, th_hat, Cs, flat = st.main, st.th_main, st.Cs, False
    th = sp.pack({k: float(T[k].detach()) for k in sp.keys}, coef.numpy())
    out = {"value": float(value), "l": lval, "loss": float(loss), "leaves": leaves, "lam_i": lam_i,
           "R": {}, "R_ipath": {}, "lam_a": {}, "lam_a_ipath": {}}
    leaves["_adj"] = {}
    for mode in st.modes:                              # one profile of l_I, one adjustment per score covariance
        C = None if flat else Cs[mode]
        R = None if flat else st.ratio(C, th - th_hat)
        lam_ipath = lam_i * R if R is not None else None
        if R is None:
            lam_a, R_eff = lam_i, None                 # no curvature to adjust with: Λ_A = Λ_I (noted)
        else:
            # the l_I-profile point is one of the starts and the best value seen is kept, so Λ_A ≤ Λ_A at the
            # l_I-profile point (the design's value) without taking a minimum over different paths
            starts = [leaves] + [(w.get("_adj") or {}).get(mode) for w in warms] + [ref]
            if not drift:
                starts.append(_quadratic_start(st, sp, C, th_hat, nat, value, dat["leaves"]))
            lA, adj, _ = st.max_adj(sp, C, th_hat, "v" if drift else nat, value, starts,
                                    vspec=(float(value), float(alpha.detach())) if drift else None)
            lam_a = max(2.0 * (st.l_hat - lA), 0.0)
            R_eff = lam_a / lam_i if lam_i > 0 else R
            leaves["_adj"][mode] = adj
        out["R"][mode], out["R_ipath"][mode] = R_eff, R
        out["lam_a"][mode], out["lam_a_ipath"][mode] = lam_a, lam_ipath
    return out


#: L-BFGS iterations each start of a profile maximisation gets before the best is continued (an eighth of the
#: full count, at least 4: operational)
SCREEN_FRAC = 8


def _screen_iter(iters: int) -> int:
    return max(4, int(iters) // SCREEN_FRAC)


def _distinct(starts: list[dict], keys: list[str]) -> list[dict]:
    """``starts`` without repeats (the same values of ``keys`` to 12 significant digits), in order."""
    seen, out = set(), []
    for st_ in starts:
        sig = []
        for k in keys:
            v = st_.get(k)
            if isinstance(v, (list, tuple, np.ndarray)):
                sig += [float(f"{float(u):.12g}") for u in v]
            elif v is not None:
                sig.append(float(f"{float(v):.12g}"))
        key = tuple(sig)
        if key not in seen:
            seen.add(key)
            out.append(st_)
    return out


#: Λ (naive or adjusted) of a profile point above the next point outward by more than this: the profile is not
#: monotone, so a maximisation failed (operational tolerance; 0.1 = the restart threshold in Λ)
MONO_TOL = 0.1


#: Leaf value a parameter held on a fit bound is released to (inside the bound: σ(±3) = 0.047 of a sigmoid's range
#: in, softplus(−3) = 0.049 × scale above a floor), where its leaf has a usable gradient (operational).
RELEASE_LEAF = 3.0


def _released(st: _Study, start: dict, skip: set) -> dict | None:
    """``start`` with the leaves of the parameters held on a fit bound at θ̂ (other than ``skip``) moved inside
    their bounds (:data:`RELEASE_LEAF`); None when there are none."""
    from sparc.core.physics import NATURAL_LEAF

    out, moved = dict(start), False
    for k, side in (getattr(st, "held_bound", None) or {}).items():
        leaf = NATURAL_LEAF[k]
        if leaf in skip or leaf not in st.active:
            continue
        out[leaf] = -RELEASE_LEAF if side == "lower" else RELEASE_LEAF
        moved = True
    return out if moved else None


def _crossing(seq: list[dict], x0: float, lam0: float, key: str, thr: float = CHI2_95):
    """First outward crossing of ``thr`` along ``seq`` (points ordered outward from x0), interpolated linearly
    in √Λ (exact for a quadratic profile).  None when the side does not cross."""
    xp, lp = x0, max(lam0, 0.0)
    if lp >= thr:
        return x0
    for p in seq:
        lam = max(p[key], 0.0)
        if lam >= thr:
            a, b = math.sqrt(lp), math.sqrt(lam)
            f = (math.sqrt(thr) - a) / (b - a) if b > a else 1.0
            return xp + f * (p["x"] - xp)
        xp, lp = p["x"], lam
    return None


def _walk(st: _Study, r: dict, side: int, x_hat: float, h0: float, lo_w, hi_w, dat: dict, cache: dict) -> dict:
    S = st.S
    thr, stop = r["_thr"], CHI2_95 + float(S["margin"])
    pts: list[dict] = []
    warm0 = {**dat["leaves"], "_coef": list(dat["coef"])}
    if (side > 0 and hi_w is not None and x_hat >= hi_w) or (side < 0 and lo_w is not None and x_hat <= lo_w):
        # the estimate already lies at or beyond the walk limit on this side (e.g. a saturated evaporative
        # fraction past its EF walk limit): there is nothing to walk on that side, and clamping to the limit
        # would step *backwards*, onto the other side of the estimate.  The side stays open.
        return {"points": [], "edge": True, "beyond_limit": True}
    x_prev, lam_prev, warm, h, edge = x_hat, 0.0, warm0, h0, False
    for _ in range(int(S["max_points"])):
        progress.check_cancel()
        x = x_prev + side * h
        if side > 0 and hi_w is not None and x >= hi_w:
            x, edge = hi_w, True
        if side < 0 and lo_w is not None and x <= lo_w:
            x, edge = lo_w, True
        if abs(x - x_prev) <= 1e-12 * max(1.0, abs(x)):
            break
        p = _point(st, r, _from_walk(r, x), [warm], dat, cache)
        p["x"], p["side"] = x, side
        pts.append(p)
        lam_min = min(p["lam_a"].values())                  # the smallest adjusted Λ (the widest interval)
        progress.tick(len(pts), int(S["max_points"]), unit="profile_point", lam_adj=lam_min, side=side)
        if min(lam_min, p["lam_i"]) >= stop or edge:          # every adjusted and the naive profile crossed
            break
        dl = lam_min - lam_prev
        f = math.sqrt(float(S["target_step"]) / dl) if dl > 1e-9 else 3.0
        h *= min(max(f, 0.5), 3.0)
        x_prev, lam_prev, warm = x, lam_min, p["leaves"]
    # bisect a wide crossing bracket (of each adjusted profile) so the interval end is interpolated over a short step
    for mode in st.modes:
        for _ in range(int(S["refine"])):
            seq = sorted(pts, key=lambda q: abs(q["x"] - x_hat))
            prev = {"x": x_hat, "lam_a": {mode: 0.0}, "leaves": warm0}
            br = None
            for q in seq:
                if q["lam_a"][mode] >= thr:
                    br = (prev, q)
                    break
                prev = q
            if br is None:
                break
            a, b = br
            if b["lam_a"][mode] - thr <= float(S["margin"]) and thr - a["lam_a"][mode] <= float(S["margin"]):
                break
            xm = 0.5 * (a["x"] + b["x"])
            p = _point(st, r, _from_walk(r, xm), [a["leaves"], b["leaves"]], dat, cache)
            p["x"], p["side"] = xm, side
            pts.append(p)
    pts.sort(key=lambda q: abs(q["x"] - x_hat))
    bad = _repair_monotone(st, r, pts, dat, cache)
    at_edge = bool(pts) and ((lo_w is not None and side < 0 and pts[-1]["x"] <= lo_w)
                             or (hi_w is not None and side > 0 and pts[-1]["x"] >= hi_w))
    return {"points": pts, "edge": bool(edge or at_edge), "non_monotone": bad}


def _nonmonotone(st: _Study, seq: list[dict]) -> list[tuple[int, str]]:
    """(i, which) where point i's Λ (naive "naive", or an adjusted one by its score covariance) exceeds that of
    the next point outward, i + 1, by more than :data:`MONO_TOL` (``seq`` ordered outward)."""
    bad = []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a["lam_i"] > b["lam_i"] + MONO_TOL:
            bad.append((i, "naive"))
        bad += [(i, mode) for mode in st.modes if a["lam_a"][mode] > b["lam_a"][mode] + MONO_TOL]
    return bad


def _merge_point(st: _Study, p: dict, q: dict) -> None:
    """Keep, in ``p``, the better l_I optimum and the better l_A optimum per score covariance of ``p`` and of
    ``q`` (two maximisations at the same ψ)."""
    if q["l"] > p["l"]:
        adj = p["leaves"].get("_adj") or {}
        p.update(l=q["l"], lam_i=q["lam_i"], loss=q["loss"])
        p["leaves"] = {**q["leaves"], "_adj": dict(adj)}
    for mode in st.modes:
        if q["lam_a"][mode] < p["lam_a"][mode]:
            p["lam_a"][mode] = q["lam_a"][mode]
            p["lam_a_ipath"][mode] = q["lam_a_ipath"][mode]
            if (q["leaves"].get("_adj") or {}).get(mode) is not None:
                p["leaves"].setdefault("_adj", {})[mode] = q["leaves"]["_adj"][mode]
        if p["R"][mode] is not None or q["R"][mode] is not None:
            p["R"][mode] = p["lam_a"][mode] / p["lam_i"] if p["lam_i"] > 0 else (p["R"][mode] or q["R"][mode])


def _repair_monotone(st: _Study, r: dict, seq: list[dict], dat: dict, cache: dict) -> list[dict]:
    """A profile (max over the other parameters) cannot rise inward: where a point's Λ exceeds the next point
    outward's by more than :data:`MONO_TOL`, its maximisation failed, so it is redone with the outer point's
    optimum as an extra start and the better optima kept (:func:`_merge_point`; two passes).  Returns the
    violations left, [{inner, outer, profile}] (ψ values), each one also warned about."""
    for _ in range(2):
        bad = _nonmonotone(st, seq)
        if not bad:
            break
        for i in sorted({i for i, _ in bad}, reverse=True):
            inner, outer = seq[i], seq[i + 1]
            q = _point(st, r, inner["value"], [outer["leaves"], inner["leaves"]], dat, cache)
            _merge_point(st, inner, q)
    left = [{"inner": seq[i]["value"], "outer": seq[i + 1]["value"], "profile": which}
            for i, which in _nonmonotone(st, seq)]
    for v in left:
        st.warn(f"the {v['profile']} profile of {r['key']} is not monotone beyond {MONO_TOL} in Λ between "
                f"{v['inner']:.4g} and {v['outer']:.4g} after re-maximising from the outer point: a local optimum or a "
                "rough likelihood there; its interval end on that side is uncertain")
    return left


def _profile(st: _Study, r: dict, dat: dict, cache: dict) -> None:
    lo_w, hi_w = _walk_bounds(st, r)                     # on the walk scale (logs for L, κ, width)
    est = r["estimate_data"]
    sides = (1, -1)
    if r["at_bound"]:
        sides = (1,) if r["at_bound"]["side"] == "lower" else (-1,)
        est = float(r["at_bound"]["bound"])
    # an estimate on its fit bound: the profile starts at the bound itself, where Λ = 0 (not at the walk limit
    # AT_BOUND_FRAC inside it, which would put the first crossing too far in when the profile is steep)
    x_hat = _to_walk(r, est)
    se_w = None
    if r.get("se_robust"):
        se_w = r["se_robust"] / max(abs(r["estimate_data"]), 1e-300) if PARAMS[r["key"]][3] == "log" \
            else r["se_robust"]
    rng = (hi_w - lo_w) if lo_w is not None and hi_w is not None else None
    if se_w and se_w > 0:
        h0 = 0.8 * se_w
    else:
        h0 = 0.05 * rng if rng else 0.1 * max(abs(x_hat), 1.0)
    if rng:
        h0 = min(h0, 0.25 * rng)
    h0 = max(h0, 1e-6 * max(1.0, abs(x_hat)))
    r["_x_hat"], r["_sides"] = x_hat, {}
    # an estimate on the boundary of the parameter space: one-sided 95% bound at χ²₁(0.90) (Self & Liang 1987)
    r["_thr"] = CHI2_90 if r["at_bound"] else CHI2_95
    r["threshold"] = r["_thr"]
    pts = []
    for side in sides:
        res = _walk(st, r, side, x_hat, h0, lo_w, hi_w, dat, cache)
        r["_sides"]["upper" if side > 0 else "lower"] = res
        pts += res["points"]
    r["_points"] = pts


def _crossing_mode(seq: list[dict], x0: float, mode: str, thr: float = CHI2_95):
    """:func:`_crossing` of one adjusted profile (``lam_a[mode]``)."""
    return _crossing([{"x": q["x"], "lam": q["lam_a"][mode]} for q in seq], x0, 0.0, "lam", thr)


def _side_status(r: dict, name: str, res: dict, x_end) -> tuple[str, float | None, str | None]:
    """(status, end, why open) of one explored side of one profile: ("cross", end, None) when Λ crossed the
    threshold at ``x_end`` (walk scale); ("limit", the limit, why) when it reached a physical limit
    (:data:`PHYSICAL_LIMITS`) without crossing; ("open", None, why) otherwise (a search range or a walk limit, or
    the explored range).  A side that does not cross is open (Raue et al. 2009), wherever it stops."""
    seq = res["points"]
    if x_end is not None:
        return "cross", _from_walk(r, x_end), None
    far = _from_walk(r, max(seq, key=lambda q: abs(q["x"] - r["_x_hat"]))["x"]) if seq else None
    if res.get("beyond_limit"):
        return "open", None, "the estimate lies beyond the walk limit (a numerically saturated model)"
    if seq and res["edge"] and name in PHYSICAL_LIMITS.get(r["key"], ()):
        return "limit", far, f"at the physical limit {_g(far)}"
    if seq and res["edge"]:
        return "open", None, f"at the end of the search range ({_g(far)})"
    return "open", None, ("not crossed within the explored range" + (f" (to {_g(far)})" if far is not None else ""))


def _union_status(statuses: list[tuple], name: str) -> tuple[str, float | None, str | None]:
    """The wider of several sides' (status, end, why): open beats a limit, a limit beats a crossing, and of two
    crossings the one further out (the conservative interval)."""
    for st_ in statuses:
        if st_[0] == "open":
            return st_
    lim = [st_ for st_ in statuses if st_[0] == "limit"]
    if lim:
        return (min if name == "lower" else max)(lim, key=lambda t: t[1])
    xs = [x for _, x, _ in statuses]
    return "cross", (min(xs) if name == "lower" else max(xs)), None


def _all_sides(r: dict, status: dict) -> dict:
    """The explored sides' statuses plus, for an estimate on its fit bound, the side the profile cannot walk: open
    at that bound (``bound``)."""
    sides = dict(status)
    if r["at_bound"]:
        n, b = r["at_bound"]["side"], r["at_bound"]["bound"]
        kind = "the physical limit" if n in PHYSICAL_LIMITS.get(r["key"], ()) else "the fit bound"
        sides[n] = ("bound", b, f"at {kind} {_g(b)} (the estimate is on it)")
    return sides


def _classify(r: dict, status: dict, flat: bool) -> tuple[str, list[str], list[str]]:
    """(class, open sides, sides open at a physical limit) from the sides' statuses (Raue et al. 2009): identified
    when both sides cross the threshold; ``one_sided`` when one does and the other stays open (at a physical
    limit, a fit bound or the explored range: practically non-identifiable on that side); practically
    non-identifiable when neither does; structurally non-identifiable when the profile is flat."""
    sides = _all_sides(r, status)
    crossed = [n for n in ("lower", "upper") if n in sides and sides[n][0] == "cross"]
    open_sides = [n for n in ("lower", "upper") if n in sides and sides[n][0] != "cross"]
    at_limit = [n for n in open_sides if sides[n][0] == "limit"
                or (sides[n][0] == "bound" and n in PHYSICAL_LIMITS.get(r["key"], ()))]
    if flat:
        return "structurally_non_identifiable", open_sides, at_limit
    if r["held"] == "flat":
        return "practically_non_identifiable", open_sides, at_limit
    if len(crossed) == 2:
        return "identified", open_sides, at_limit
    if len(crossed) == 1:
        return "one_sided", open_sides, at_limit
    return "practically_non_identifiable", open_sides, at_limit


#: short symbols of the profiled parameters (for the interval summaries)
_SYMBOL = {"L": "L", "v": "|v|", "s": "s", "kappa_canopy": "κ", "shade_c0": "c₀", "shade_width": "width",
           "a1": "a₁", "a1n": "a₁ₙ", "e0": "e₀", "e1": "e₁", "e2": "e₂", "e3": "e₃", "w": "w"}


def _summary(r: dict, cls: str, sides: dict) -> str:
    """The interval in words, keeping its informative part (e.g. "s ≤ 0.49; lower side open at the physical limit
    0 (the estimate is on it); one-sided 95% bound (Self & Liang 1987)")."""
    sym = _SYMBOL.get(r["key"], r["key"])
    u = f" {r['unit']}" if r.get("unit") in ("m", "W/m²") else ""
    lo, hi = sides.get("lower"), sides.get("upper")
    tail = "; one-sided 95% bound at Λ = 2.706 (Self & Liang 1987)" if r["at_bound"] else ""
    if cls == "structurally_non_identifiable":
        return f"{sym}: flat profile (no information)"
    if cls == "not_assessable":
        nv = r.get("ci95_naive") or [None, None]
        b = (f"{sym} ≤ {_g(nv[1])}{u}" if r["at_bound"]["side"] == "lower" else f"{sym} ≥ {_g(nv[0])}{u}")
        return f"unadjusted {b} (too narrow under spatial dependence){tail}"
    if cls == "identified":
        return f"{_g(lo[1])} ≤ {sym} ≤ {_g(hi[1])}{u}"
    parts = []
    if cls == "one_sided":
        parts.append(f"{sym} ≤ {_g(hi[1])}{u}" if hi[0] == "cross" else f"{sym} ≥ {_g(lo[1])}{u}")
    for n, sd in (("lower", lo), ("upper", hi)):
        if sd is not None and sd[0] != "cross":
            parts.append(f"{n} side open {sd[2]}")
    return "; ".join(parts) + tail


def _finish_profile(st: _Study, r: dict, best_l: float) -> None:
    pts = r.get("_points") or []
    for p in pts:
        p["lam_i"] = max(2.0 * (best_l - p["l"]), 0.0)
        p["lam_a"] = {mode: p["lam_i"] * (R if R is not None else 1.0) for mode, R in p["R"].items()}
    lam0 = max(2.0 * (best_l - st.l_hat), 0.0)          # Λ_I at θ̂ (0 unless a profile point did better)
    x_hat = r["_x_hat"]
    explored = list(r["_sides"])
    status = {mode: {} for mode in st.modes}
    status_naive = {}
    for name, res in r["_sides"].items():
        seq = sorted(res["points"], key=lambda q: abs(q["x"] - x_hat))
        res["points"] = seq
        if res.get("beyond_limit"):
            r["notes"].append(f"the data-only estimate lies beyond the profile's walk limit on the {name} side (the "
                              "model is numerically saturated there): that side is not explored and stays open")
        for mode in st.modes:
            status[mode][name] = _side_status(r, name, res, _crossing_mode(seq, x_hat, mode, r["_thr"]))
        status_naive[name] = _side_status(r, name, res, _crossing(seq, x_hat, lam0, "lam_i", r["_thr"]))
    bound = r["at_bound"]["bound"] if r["at_bound"] else None

    def ci(stat):
        lo = stat["lower"][1] if "lower" in stat else bound
        hi = stat["upper"][1] if "upper" in stat else bound
        return [lo, hi]

    for mode in st.modes:
        r["ci95" + _SUFFIX[mode]] = ci(status[mode])
    r["ci95_naive"] = ci(status_naive)
    ba = r.get("boundary_adjustment") or {}
    not_assessable = bool(r["at_bound"] and not ba.get("computable"))
    if r["at_bound"]:
        r["notes"].append(f"its data-only estimate is on its {r['at_bound']['side']} fit bound: a boundary maximum, so "
                          "its profile is one-sided and its 95% bound is where Λ reaches χ²₁(0.90) = 2.706 (Self & "
                          "Liang 1987: the likelihood-ratio statistic is a 50:50 mixture of 0 and χ²₁ there)")
        if ba.get("computable"):
            r["notes"].append("adjusted by the scalar c = Σ_naive/Σ_rob of the well-conditioned curvature at the bound "
                              f"(c = {_g(ba['c'].get('blocks'), 3)} with the block V)")
    # the class rests on the wider of the adjusted intervals (conservative): per side, the union of the sides
    union = {name: _union_status([status[mode][name] for mode in st.modes], name) for name in explored}
    max_lam = max([min(p["lam_a"].values()) for p in pts], default=0.0)
    flat = max_lam < FLAT_LAMBDA
    by_mode = {mode: _classify(r, status[mode], flat)[0] for mode in st.modes}
    cls, open_sides, at_limit = _classify(r, union, flat)
    r["ci95_conservative"] = ci(union)                  # per side the wider of the adjusted intervals
    if not_assessable and not flat:
        # no adjustment from a well-conditioned curvature: no adjusted interval; the naive one-sided bound is
        # reported, labelled, and the parameter is not assessable (rather than identified)
        cls = "not_assessable"
        by_mode = {mode: "not_assessable" for mode in st.modes}
        for mode in st.modes:
            r["ci95" + _SUFFIX[mode]] = None
        r["ci95_conservative"] = None
        r["ci95_naive_label"] = "unadjusted (too narrow under spatial dependence)"
        r["notes"].append(f"not assessable: its block adjustment is not computable ({ba.get('reason')}); the naive "
                          "one-sided bound is unadjusted (too narrow under spatial dependence)")
    r["identifiability"] = cls
    r["class_by_covariance"] = by_mode
    r["open_sides"], r["sides_at_limit"] = open_sides, at_limit
    sides = _all_sides(r, union)
    r["open_reasons"] = {n: sides[n][2] for n in open_sides}
    r["summary"] = _summary(r, cls, sides)
    if len(st.modes) > 1 and len(set(by_mode.values())) > 1:
        r["notes"].append("the class rests on the wider of the two adjusted intervals ("
                          + ", ".join(f"{m}: {c}" for m, c in by_mode.items()) + ")")
    if at_limit:
        r["notes"].append(f"open on the {' and '.join(at_limit)} side at its physical limit: the data do not "
                          "exclude it")
    unadj = [p for p in pts if any(R is None for R in p["R"].values()) and p["lam_i"] > 0]
    if unadj:
        r["notes"].append(f"{len(unadj)} profile point(s) without curvature in the adjustment coordinates: Λ_A = Λ_I "
                          "there (unadjusted)")
    if flat:
        r["notes"].append(f"flat profile: Λ_A < {FLAT_LAMBDA} everywhere explored")
    elif r["held"] == "flat":
        # no local curvature at the optimum, yet the profile is not flat: the likelihood is not smooth there
        # (e.g. a saturated evaporative fraction, a step function of the covariates), so no interval is trusted
        r["notes"].append("no local curvature at the optimum but a profile that is not flat: the likelihood is not "
                          "smooth there (e.g. a saturated evaporative fraction), so its interval is not reliable")
    x0 = r["estimate_data"] if not r["at_bound"] else float(bound)
    grid = sorted([(x0, {m: 0.0 for m in st.modes}, lam0, {m: None for m in st.modes}, {m: 0.0 for m in st.modes})]
                  + [(p["value"], p["lam_a"], p["lam_i"], p["R"], p.get("lam_a_ipath") or {}) for p in pts],
                  key=lambda t: t[0])
    nonmono = [v for res in r["_sides"].values() for v in (res.get("non_monotone") or [])]
    prof = {"grid": [g[0] for g in grid], "lambda_naive": [g[2] for g in grid],
            "sides_explored": explored, "stopped_at_bound": [n for n, res in r["_sides"].items() if res["edge"]],
            "monotone": not nonmono, "non_monotone": nonmono}
    if nonmono:
        r["notes"].append("the profile is not monotone after re-maximisation ("
                          + "; ".join(f"{v['profile']}: {_g(v['inner'])} above {_g(v['outer'])}" for v in nonmono)
                          + "): its interval end on that side is uncertain")
    for mode in st.modes:
        sfx = _SUFFIX[mode]
        prof["lambda_adj" + sfx] = [g[1][mode] for g in grid]
        prof["ratio" + sfx] = [g[3].get(mode) for g in grid]
        prof["lambda_adj_ipath" + sfx] = [g[4].get(mode) for g in grid]
    r["profile"] = prof


# --------------------------------------------------------------------------- the run's (penalized) optimum
def _penalized(st: _Study, pen: dict, dat: dict, held_flat: set) -> dict:
    """Sandwich SEs of the run's estimator (an M-estimator of ∇J = 0) and each parameter's prior share of the
    run objective's profile curvature, at the penalized optimum.  The data term's curvature is its expected
    (Gauss–Newton, Fisher) form 2JᵀJ/(n·var y), J = ∂r/∂θ: away from the data optimum the exact Hessian of the
    MSE can be indefinite (a profile curvature below zero has no "share")."""
    import torch

    from sparc.core.physics import LEAF_NATURAL

    keys = [LEAF_NATURAL[n] for n in st.active if LEAF_NATURAL[n] not in pen["at_bound"]]
    sp = _Space(st, pen["leaves"], keys)
    th = sp.pack(pen["natural"], pen["coef"])

    def pen_fn(t):
        P, nat, coef = sp.unpack(t)
        out = st.terms.penalty(P, prior_scale=None, natural=nat or None)
        return out if isinstance(out, torch.Tensor) else (t * 0.0).sum() + float(out)

    J = st.jacobian(sp, th)
    s2 = pen["mse"]
    Hd = 2.0 * (J.T @ J) / (st.n * st.terms.var_y)
    Hp = _sym(torch.autograd.functional.hessian(pen_fn, _t(th)).numpy())
    V = st.score_cov(st.scores(sp, th), st.primary)
    k = st.n * st.terms.var_y / (2.0 * s2)
    Ainv = _pos_inverse(k * (Hd + Hp))
    Sr = _sym(Ainv @ V @ Ainv)
    Sn = _sym(Ainv @ ((J.T @ J) / s2) @ Ainv)
    share = {}
    for i, key in enumerate(sp.keys):
        if key in ("vx", "vy"):
            continue
        s_tot = _schur(Hd + Hp, i)
        s_dat = _schur(Hd, i)
        share[key] = float(min(max(1.0 - s_dat / s_tot, 0.0), 1.0)) if s_tot > 0 else None
    return {"space": sp, "Sr": Sr, "Sn": Sn, "share": share}


def _linear_rows(st: _Study, dat: dict, pen: dict, pen_info: dict) -> list[dict]:
    rows = []
    names = list(st.terms.linear_names)
    for j, nm in enumerate(names):
        label, unit = _LINEAR_LABELS[nm]
        i_d = st.main.index(nm)
        i_p = pen_info["space"].index(nm)
        c = st.col[j]
        sn_d = math.sqrt(max(st.Cs["blocks"]["Sn"][i_d, i_d], 0.0)) / c
        sr_d = math.sqrt(max(st.Cs["blocks"]["Sr"][i_d, i_d], 0.0)) / c
        sr_nb = (math.sqrt(max(st.Cs["neighbours"]["Sr"][i_d, i_d], 0.0)) / c if "neighbours" in st.Cs else None)
        sn_p = math.sqrt(max(pen_info["Sn"][i_p, i_p], 0.0)) / c
        sr_p = math.sqrt(max(pen_info["Sr"][i_p, i_p], 0.0)) / c
        rows.append({"key": nm, "label": label, "unit": unit,
                     "estimate_data": float(dat["coef"][j]), "se_naive_data": float(sn_d), "se_robust_data": float(sr_d),
                     "se_robust_data_neighbours": sr_nb,
                     "estimate_run": float(pen["coef"][j]), "se_naive_run": float(sn_p), "se_robust_run": float(sr_p),
                     "adjust_factor": float((sn_d / sr_d) ** 2) if sr_d > 0 else None})
    return rows


def _derived(linear: list[dict], units, context: dict) -> dict:
    """r_a = a·(K per target unit)·ρc_p/norm and γ in K/km, with the parameter panel's helpers and constants."""
    from sparc.core import mechanisms as M

    by = {r["key"]: r for r in linear}
    fK = M.to_kelvin_factor(units)
    if fK is None:
        return {"note": f"target units {units!r} are not a temperature scale: no physical units"}
    ph = context.get("physics") or {}
    window = context.get("window") or str(ph.get("window") or "day")
    norm, norm_label = M.forcing_norm(ph, window)
    t = M.forcing_t2m_K(ph, context.get("config_dir"))
    t_note = f"forcing 2 m temperature {t:.1f} K" if t is not None else f"assumed {M.T_FALLBACK_K:.2f} K"
    rcp = M.rho_cp(t if t is not None else M.T_FALLBACK_K)
    out = {"to_kelvin": fK, "rho_cp": rcp, "t2m_note": t_note, "norm": norm, "norm_label": norm_label,
           "window": window}

    def conv(r, f):
        return {"data_only": {"value": r["estimate_data"] * f, "se_naive": r["se_naive_data"] * abs(f),
                              "se_robust": r["se_robust_data"] * abs(f)},
                "run": {"value": r["estimate_run"] * f, "se_naive": r["se_naive_run"] * abs(f),
                        "se_robust": r["se_robust_run"] * abs(f)}}

    if "a" in by:
        out["r_a"] = {"unit": "s/m", **conv(by["a"], fK * rcp / norm),
                      "derivation": f"r_a = a·{'(5/9)·' if fK != 1.0 else ''}ρc_p/{norm_label}, ρc_p = {rcp:.0f} "
                                    f"J m⁻³ K⁻¹ ({t_note}), {norm_label} = {norm:.1f} W/m²"}
    if "gamma" in by:
        out["gamma_K_per_km"] = {"unit": "K/km", **conv(by["gamma"], fK * 1000.0)}
    panel = context.get("panel_rows") or {}
    for key, name in (("r_a", "r_a"), ("gamma", "gamma_K_per_km")):
        pr = panel.get(key)
        if name in out and pr and pr.get("value") is not None:
            out[name]["fold_jackknife"] = {"value": pr["value"], "se": pr.get("se"), "n": pr.get("n")}
    return out


# --------------------------------------------------------------------------- calibrated posterior
def _log_prior_nat(x: np.ndarray, spec: tuple) -> np.ndarray:
    """The nominal log prior at natural values ``x`` on its stated scale (no Jacobian)."""
    sc, mu, sd = spec
    v = np.log(np.maximum(x, 1e-300)) if sc == "log" else x
    return -0.5 * ((v - mu) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def _in_bounds(st: _Study, space: _Space, TH: np.ndarray) -> np.ndarray:
    """Rows of ``TH`` (space coordinates) whose nonlinear parameters lie inside their fit bounds."""
    ok = np.ones(TH.shape[0], dtype=bool)
    for i, k in enumerate(space.keys):
        if k in ("vx", "vy"):
            continue
        lo, hi, kind = st.model.natural_bounds(k)
        x = np.exp(np.minimum(TH[:, i], 700.0)) if k in _LOG_KEYS else TH[:, i] * space.scale(k)
        if lo is not None and kind != "free":
            ok &= x >= lo
        if hi is not None and kind in ("lin", "log"):
            ok &= x <= hi
    return ok


def _posterior(st: _Study, rows: list[dict], dat: dict, prior_spec: dict, held_bound: set,
               held_flat: set) -> None:
    """Laplace approximation of the calibrated posterior l_A + log nominal prior in the coordinates of H_A, its
    draws and power-scaling; a held parameter on its bound with a nominal prior gets the 1-D posterior of its
    adjusted profile (:func:`_profile_posterior`)."""
    import torch

    from sparc.core.physics import _lbfgs

    m, main, C = st.model, st.main, st.C
    flat_prior = [k for k in sorted(held_flat) if k in prior_spec]
    post = _Space(st, dat["leaves"], main.keys + flat_prior)
    th_hat = post.pack(dat["natural"], dat["coef"])
    p = post.p
    pos = [post.index(nm) for nm in main.names]
    HA = np.zeros((p, p))
    HA[np.ix_(pos, pos)] = C["HA"]
    HA_t, HI_t, th_hat_t = _t(C["HA"]), _t(C["H"]), _t(th_hat)
    pos_t = torch.as_tensor(pos, dtype=torch.long)
    pri = [(i, k, *prior_spec[k]) for i, k in enumerate(post.keys) if k in prior_spec]

    def log_prior(theta):
        """Σ log N(x; μ, sd²), x the natural value or its log as the prior is stated (no Jacobian: the
        coordinates are those scales up to a constant factor)."""
        tot = torch.zeros((), dtype=torch.float64)
        for i, k, sc, mu, sd in pri:
            if sc == "log":
                x = theta[i] if k in _LOG_KEYS else torch.log(theta[i] * post.scale(k))
            else:
                x = theta[i].exp() if k in _LOG_KEYS else theta[i] * post.scale(k)
            tot = tot - 0.5 * ((x - mu) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2.0 * math.pi)
        return tot

    def l_adj(theta):
        return st.l_adj_t(st.loglik(post, theta), theta[pos_t], HA_t, HI_t, th_hat_t[pos_t])

    with progress.task("posterior_mode"):
        th = _t(th_hat).clone().requires_grad_(True)
        _lbfgs([th], lambda: -(l_adj(th) + log_prior(th)), int(st.S["mode_iter"]))
        mode = th.detach().numpy().copy()
        if not (np.all(np.isfinite(mode)) and _in_bounds(st, post, mode[None, :])[0]):
            st.warn("the posterior mode search left the parameter space; the Laplace step uses θ̂ instead")
            mode = th_hat.copy()
        Hpi = _sym(torch.autograd.functional.hessian(lambda t: log_prior(t), _t(mode)).numpy())
        Sigma, _, wmin = _neg_def_inverse(HA + Hpi)
    if wmin <= 1e-10:
        st.warn("the Laplace posterior's precision −(H_A + prior Hessian) is not positive definite; floored")
    Sigma = _sym(Sigma)
    try:
        Lc = np.linalg.cholesky(Sigma + 1e-14 * np.diag(np.diag(Sigma)))
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(Sigma)
        Lc = Q * np.sqrt(np.maximum(w, 0.0))
    D = int(st.S["draws"])
    rng = np.random.default_rng(int(st.S["seed"]))
    # ``draws`` draws inside the fit bounds (rejection from the Gaussian; at most 50·draws proposals)
    kept, proposed = [], 0
    while sum(len(b) for b in kept) < D and proposed < 50 * D:
        B = mode + rng.standard_normal((D, p)) @ Lc.T
        proposed += D
        kept.append(B[_in_bounds(st, post, B)])
    TH = np.concatenate(kept)[:D]
    Dk = int(TH.shape[0])
    lA = np.empty(Dk)           # the calibrated log-likelihood l_A at each draw (one model evaluation each)
    lp = np.empty(Dk)           # the nominal log prior, on its stated scale (power-scaled as specified)
    dTH = (TH - th_hat)[:, pos]
    lq = 0.5 * np.einsum("ij,jk,ik->i", dTH, C["HA"], dTH)      # the Laplace approximation's likelihood term
    t0 = time.perf_counter()
    with progress.task("power_scaling", unit="draw"), torch.no_grad():
        for d in range(Dk):
            if d % 50 == 0:
                progress.check_cancel()
                progress.tick(d + 1, Dk, unit="draw")
            th_d = _t(TH[d])
            lA[d] = float(l_adj(th_d))
            lp[d] = float(log_prior(th_d))
        progress.tick(Dk, Dk, unit="draw")
    st.timings["draws_s"] = round(time.perf_counter() - t0, 2)
    st.timings["draw_eval_ms"] = round(1000.0 * (time.perf_counter() - t0) / max(Dk, 1), 2)
    if Dk < 10:
        st.warn(f"only {Dk} of {proposed} Laplace draws inside the fit bounds: no posterior summaries")
        st.posterior_info = {"method": "laplace", "n_draws": Dk, "n_proposed": proposed}
        return
    # adequacy of the Laplace approximation: importance weights of its draws for exp(l_A + log π) itself
    Z = TH - mode
    logq = -0.5 * np.einsum("ij,jk,ik->i", Z, _pos_inverse(Sigma), Z)
    lw = lA + lp - logq
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    is_ess = _ess(w)
    gap = (lA - st.l_hat) - lq
    adequate = bool(is_ess / Dk >= LAPLACE_MIN_ESS_FRAC)
    # the Laplace step holds the parameters on a fit bound at their bound, and the flat ones without a nominal
    # prior at their data-only values: its posterior is conditional on them
    cond = sorted([_NAT_KEY.get(k, k) for k in held_bound] + [_NAT_KEY.get(k, k) for k in held_flat
                                                               if k not in prior_spec])
    why_poor = None if adequate else (
        f"the Laplace approximation of the calibrated posterior is poor (importance-sampling ESS {is_ess:.0f} of {Dk} "
        f"draws for exp(l_A + log prior), below {LAPLACE_MIN_ESS_FRAC:.0%}): its mean, sd and power-scaling describe "
        "the Gaussian approximation, not the posterior")
    laplace_desc = ("Laplace (Gaussian) approximation of l_A + log nominal prior at its mode"
                    + (f", conditional on {', '.join(cond)} held fixed" if cond else ""))
    laplace_keys = []
    for r in rows:
        if not r["_in_model"]:
            continue
        if r["key"] == "v":
            if "vx" not in post.keys:
                continue
            iu, iv = post.index("vx"), post.index("vy")
            x = m.v_max * np.hypot(TH[:, iu], TH[:, iv])
            xm = m.v_max * math.hypot(mode[iu], mode[iv])
        elif r["_nat"] in post.keys:
            i = post.index(r["_nat"])
            x = np.array([post.natural(r["_nat"], v) for v in TH[:, i]])
            xm = post.natural(r["_nat"], mode[i])
        else:
            computable = (r.get("boundary_adjustment") or {}).get("computable")
            if r["at_bound"] and r["prior"] and r.get("_points") and computable:
                _profile_posterior(st, r, prior_spec[r["_nat"]], rng)
            else:
                r["posterior"] = {"mean": None, "sd": None, "method": None,
                                  "note": ("held at its fit bound and not assessable: no adjusted profile to build a "
                                           "posterior from" if r["at_bound"] and r["prior"] and not computable else
                                           "held at its fit bound, no nominal prior" if r["at_bound"] else
                                           "held (no information and no nominal prior)")}
            continue
        laplace_keys.append(r["key"])
        r["posterior"] = {"method": "laplace", "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
                          "mode": float(xm), "q025": float(np.quantile(x, 0.025)),
                          "q975": float(np.quantile(x, 0.975)), "mean_is": float(np.sum(w * x)),
                          "sd_is": float(math.sqrt(max(np.sum(w * (x - np.sum(w * x)) ** 2), 0.0))),
                          "reliable": adequate, "reason": why_poor, "description": laplace_desc,
                          "conditional_on": cond}
        ps = power_scaling_sensitivity(x, log_prior=lp, log_lik=lq, delta=float(st.S["delta"]))
        ps_a = power_scaling_sensitivity(x, log_lik=lA, delta=float(st.S["delta"]))
        r["psens"] = {"prior": ps["prior"], "likelihood": ps["likelihood"], "diagnosis": ps["diagnosis"],
                      "ess": {"prior": ps["ess_prior"], "likelihood": ps["ess_likelihood"]},
                      "khat": {"prior": ps["khat_prior"], "likelihood": ps["khat_likelihood"]},
                      "likelihood_lA": ps_a["likelihood"], "ess_likelihood_lA": ps_a["ess_likelihood"],
                      "method": "laplace", "reliable": adequate, "reason": why_poor,
                      "description": "power-scaling of the draws of the " + laplace_desc}
    st.posterior_info = {
        "method": "Laplace approximation at the mode of l_A + log nominal prior in the coordinates of H_A (not "
                  "MCMC); draws outside the fit bounds discarded",
        "mode_l_adj": float(l_adj(_t(mode)).detach()), "n_draws": Dk, "n_proposed": proposed,
        "in_bounds_frac": Dk / max(proposed, 1),
        "parameters": [_NAT_KEY.get(k, k) for k in post.keys] + list(st.terms.linear_names),
        "flat_prior_parameters": [_NAT_KEY.get(k, k) for k in flat_prior],
        "is_ess": is_ess, "is_ess_frac": is_ess / Dk,
        "lA_minus_quadratic": {"q10": float(np.quantile(gap, 0.1)), "median": float(np.median(gap)),
                               "q90": float(np.quantile(gap, 0.9))},
        "laplace_adequate": adequate, "reliable": adequate, "reason": why_poor, "conditional_on": cond,
        "laplace_parameters": laplace_keys,
        "profile_parameters": [r["key"] for r in rows if (r.get("posterior") or {}).get("method") == "profile"],
        "psens_components": "prior: the nominal log prior as stated; likelihood: the Laplace posterior's own "
                            "likelihood term ½δᵀH_Aδ (likelihood_lA: l_A at each draw, as a comparison)"}
    if not adequate:
        prof = st.posterior_info["profile_parameters"]
        st.warn(f"the Laplace approximation of the calibrated posterior is poor (importance-sampling ESS {is_ess:.0f} "
                f"of {Dk} draws; median l_A − quadratic {np.median(gap):.3g}): l_I is far from quadratic on the "
                "scale of the block-adjusted uncertainty, so the posterior mean/sd and power-scaling of "
                f"{', '.join(laplace_keys) or 'no parameter'} (the Laplace step) are flagged unreliable "
                "(reliable: false) and are not findings"
                + (f"; the 1-D profile posteriors of {', '.join(prof)} do not use it" if prof else "")
                + "; the profile intervals do not rely on it")


def _profile_posterior(st: _Study, r: dict, spec: tuple, rng) -> None:
    """1-D posterior of a parameter held on its fit bound: exp(−Λ_A(ψ)/2)·π(ψ) on its adjusted profile (√Λ_A
    interpolated linearly between profile points and extrapolated linearly to Λ = 36 past the last), sampled by
    inverse CDF; power-scaling with log π and −Λ_A/2 as the components.  The profile replaces integration
    over the other parameters (no Laplace volume term)."""
    pts = sorted([(float(r["at_bound"]["bound"]), 0.0)] + [(p["value"], p["lam_a"][st.primary])
                                                            for p in r["_points"]])
    xs = np.array([q[0] for q in pts])
    rs = np.sqrt(np.maximum(np.array([q[1] for q in pts]), 0.0))
    lower = r["at_bound"]["side"] == "lower"
    if xs.size >= 2:
        # extend the open end linearly in √Λ until √Λ = 6
        i0, i1 = (-2, -1) if lower else (1, 0)
        slope = (rs[i1] - rs[i0]) / (xs[i1] - xs[i0]) if xs[i1] != xs[i0] else 0.0
        if lower and slope > 0 and rs[-1] < 6.0:
            xs, rs = np.append(xs, xs[-1] + (6.0 - rs[-1]) / slope), np.append(rs, 6.0)
        elif not lower and slope < 0 and rs[0] < 6.0:
            xs, rs = np.insert(xs, 0, xs[0] + (6.0 - rs[0]) / slope), np.insert(rs, 0, 6.0)
    lo_b, hi_b, _ = st.model.natural_bounds(r["_nat"])
    if lo_b is not None:
        xs = np.maximum(xs, lo_b)
    if hi_b is not None:
        xs = np.minimum(xs, hi_b)
    grid = np.linspace(xs.min(), xs.max(), 2001)
    lam = np.interp(grid, xs, rs) ** 2
    if spec[0] == "log":
        dens_prior = _log_prior_nat(grid, spec) - np.log(np.maximum(grid, 1e-300))   # density in ψ
    else:
        dens_prior = _log_prior_nat(grid, spec)
    logp = -0.5 * lam + dens_prior
    pdf = np.exp(logp - logp.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    if not cdf[-1] > 0:
        r["posterior"] = {"mean": None, "sd": None, "method": "profile", "note": "degenerate profile posterior"}
        return
    cdf /= cdf[-1]
    D = int(st.S["draws"])
    x = np.interp(rng.random(D), cdf, grid)
    lam_x = np.interp(x, xs, rs) ** 2
    lp = _log_prior_nat(x, spec)
    desc = ("1-D profile posterior exp(−Λ_A/2) × nominal prior along this parameter's adjusted profile (the other "
            "parameters maximised, not integrated; no Laplace step, so the Laplace adequacy check does not apply)")
    r["posterior"] = {"method": "profile", "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
                      "mode": float(grid[int(np.argmax(pdf))]), "q025": float(np.quantile(x, 0.025)),
                      "q975": float(np.quantile(x, 0.975)), "reliable": True, "reason": None, "description": desc,
                      "note": "on its fit bound: the posterior is exp(−Λ_A/2) × nominal prior along its adjusted "
                              "profile (1-D, no Laplace step)"}
    ps = power_scaling_sensitivity(x, log_prior=lp, log_lik=-0.5 * lam_x, delta=float(st.S["delta"]))
    r["psens"] = {"prior": ps["prior"], "likelihood": ps["likelihood"], "diagnosis": ps["diagnosis"],
                  "ess": {"prior": ps["ess_prior"], "likelihood": ps["ess_likelihood"]},
                  "khat": {"prior": ps["khat_prior"], "likelihood": ps["khat_likelihood"]}, "method": "profile",
                  "reliable": True, "reason": None, "description": "power-scaling of the draws of the " + desc}


def _conflict(r: dict, prior_spec: dict) -> None:
    pr = r.get("prior")
    if not pr or not r["_in_model"]:
        r["conflict"] = None
        return
    est, se = r["estimate_data"], r["se_robust"]
    if pr["scale"] == "log" and est is not None and est > 0:
        se = se / est if se is not None else None
        est = math.log(est)
    r["conflict"] = prior_data_conflict(est, se, pr["mean"], pr["sd"], at_bound=bool(r["at_bound"]),
                                        identified=r["identifiability"] == "identified")
    r["conflict"]["scale"] = pr["scale"]


def _public(r: dict) -> dict:
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    out["note"] = "; ".join(out.pop("notes")) or None
    return out


# --------------------------------------------------------------------------- a finished run
def _config_from_manifest(m: dict, run_dir: Path):
    """The run's config rebuilt from its manifest (paths relative to the recorded config directory), with the
    advection scheme the run was fitted with."""
    from sparc.core.config import core_config_from_dict, recorded_advection_scheme, with_advection_scheme

    base = (m.get("provenance") or {}).get("config_dir")
    if base is None and not (m.get("provenance") or {}).get("input_frame") \
            and not (run_dir / "input_frame.parquet").exists():
        raise ValueError(f"{run_dir}: the manifest records no config_dir (older run); pass the config (-p)")
    raw = with_advection_scheme(m["config"], recorded_advection_scheme(m["config"]))
    return core_config_from_dict(raw, base_dir=base)


def _coarse_data(run_dir: Path, cfg, m: dict, coarse_m: float):
    """The run's data averaged onto ``coarse_m`` cells (their CV blocks are assigned by :func:`_blocks_at`)."""
    import pandas as pd

    from sparc.core.data import prepare_frame, read_input

    c2 = copy.deepcopy(cfg)
    c2.raw["data"]["coarse_m"] = float(coarse_m)
    frame_file = (m.get("provenance") or {}).get("input_frame") or "input_frame.parquet"
    df = pd.read_parquet(run_dir / frame_file) if (run_dir / frame_file).exists() else read_input(c2)
    return prepare_frame(df, c2)


def _blocks_at(coords: np.ndarray, origin: tuple[float, float], block_m: float) -> np.ndarray:
    """Block index (as :func:`sparc.core.cv.block_ids`, with a given origin) of each point."""
    c = np.asarray(coords, float)
    bx = np.floor((c[:, 0] - origin[0]) / block_m).astype(np.int64)
    by = np.floor((c[:, 1] - origin[1]) / block_m).astype(np.int64)
    bx -= bx.min()
    by -= by.min()
    _, inv = np.unique(by * (bx.max() + 1) + bx, return_inverse=True)
    return inv.astype(np.int64)


def identifiability_report(run_dir, cfg=None, *, params=None, coarse: float | None = None,
                           max_points: int | None = None, draws: int | None = None, seed: int = 0,
                           settings: dict | None = None, posterior: bool = True, write: bool = True,
                           refresh_register: bool = True) -> dict:
    """The identifiability study of a finished run (module docstring): refits on the run's data (at
    ``coarse`` m cells when given), clusters = the run's CV blocks.  ``cfg``: the run's CoreConfig (default:
    rebuilt from the manifest).  With ``write``, stores ``identifiability.json`` and the manifest section
    ``identifiability`` (``runio.update_manifest``); with ``refresh_register`` too, the mechanism register is
    recomputed with the study (:func:`sparc.core.mechanisms.mechanisms_for_run`; with the given ``cfg``'s raw
    config, else the manifest's) and ``mechanisms.json`` and the manifest's ``mechanisms`` section are written in
    the same manifest update.  ``rep["register"]`` records whether the register used the study."""
    from sparc.core import mechanisms as M
    from sparc.core import runio
    from sparc.core.baselines import load_run
    from sparc.core.config import recorded_advection_scheme

    params = check_arguments(params, max_points, draws, posterior, coarse)     # before loading or fitting anything
    run_dir = Path(run_dir)
    m0 = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    register_cfg = copy.deepcopy(cfg.raw) if cfg is not None else None
    cfg = _config_from_manifest(m0, run_dir) if cfg is None else copy.deepcopy(cfg)
    t0 = time.perf_counter()
    with progress.run_dir_scope(run_dir):
        with progress.task("load_run"):
            data, folds, m, _ = load_run(run_dir, cfg)
        off = M._physics_off(cfg.raw)
        if off:
            raise ValueError(f"{run_dir}: no physics model in this run ({off})")
        fold_params, params_note = M._run_physics_params(run_dir, m)
        ph = copy.deepcopy(cfg.raw.get("physics") or {})
        ph["advection_scheme"] = (ph.get("advection_scheme") or recorded_advection_scheme(m.get("config"), fold_params)
                                  or "hybrid")
        adv = m.get("physics_advection")
        if isinstance(adv, dict) and adv.get("kept") is False:
            ph["fit_advection"] = False            # the run's delivered models are its v = 0 refits
        L_init = (m.get("influence") or {}).get("L_prior_m")
        block_m = float(folds.block_m)
        if coarse:
            cd = _coarse_data(run_dir, cfg, m, float(coarse))
            origin = (float(np.min(data.x)), float(np.min(data.y_coord)))
            clusters = _blocks_at(cd.coords, origin, block_m)
            grid, frame, y, coords = cd.grid, cd.frame, cd.y, cd.coords
        else:
            clusters = folds.block_id
            grid, frame, y, coords = data.grid, data.frame, data.y, data.coords
        panel = M.physical_parameters(fold_params, cfg.raw, manifest=m) if fold_params else None
        cdir = (m.get("provenance") or {}).get("config_dir")
        context = {"block_m": block_m, "block_origin": [float(np.min(data.x)), float(np.min(data.y_coord))],
                   "coarse_m": float(coarse) if coarse else ((m.get("qa") or {}).get("coarse") or {}).get("cell_m"),
                   "advection_selection": adv if isinstance(adv, dict) else None, "physics": ph,
                   "config_dir": cdir, "window": str(ph.get("window") or "day"),
                   "panel_rows": {r["key"]: r for r in (panel or {}).get("rows", [])}}
        rep = analyse(grid, frame, y, coords, clusters, ph, L_init=L_init, params=params, max_points=max_points,
                      draws=draws, seed=seed, units=cfg.data.get("target_units"), settings=settings,
                      posterior=posterior, fold_params=fold_params, context=context)
        rep["timings"]["wall_s"] = round(time.perf_counter() - t0, 2)
        rep["run"] = {"run_dir": str(run_dir), "name": m.get("name"), "n_points_run": int(data.n),
                      "cv": m.get("cv"), "L_prior_m": L_init, "physics_json": params_note or "used",
                      "fold_means": {r["key"]: r.get("estimate_run_folds") for r in rep.get("parameters", [])},
                      "input_sha256": (m.get("provenance") or {}).get("input_sha256"),
                      "manifest_created_utc": m.get("created_utc")}
        if write:
            sections = {"identifiability": rep}
            reg = _register_with(run_dir, register_cfg, rep) if refresh_register else None
            if reg is not None:
                ib = reg.get("identifiability") or {}
                rep["register"] = {"refreshed": True, "identifiability": ib.get("status"), "note": ib.get("note")}
            runio.write_json_atomic(run_dir / "identifiability.json", rep, indent=1)
            progress.artifact(run_dir / "identifiability.json", role="identifiability")
            if reg is not None:
                runio.write_json_atomic(run_dir / "mechanisms.json", reg, indent=1)
                progress.artifact(run_dir / "mechanisms.json", role="mechanisms")
                sections["mechanisms"] = reg
            runio.update_manifest(run_dir, sections, source="identifiability")
            progress.artifact(run_dir / "manifest.json", role="manifest")
    return rep


def _register_with(run_dir: Path, config_raw: dict | None, rep: dict) -> dict | None:
    """The mechanism register of the run recomputed with this study (not written); None, with a
    ``physics.mechanisms_failed`` warning, when it cannot be computed (the study's own results still are)."""
    from sparc.core import mechanisms as M

    try:
        return M.mechanisms_for_run(run_dir, config_raw, write=False, identifiability=rep,
                                    identifiability_source="manifest")
    except Exception as exc:                         # noqa: BLE001 - the register must never fail the study
        msg = f"{type(exc).__name__}: {exc}"
        log.warning("mechanism register not refreshed: %s", msg)
        progress.warn("physics.mechanisms_failed", f"physics mechanism register not refreshed: {msg}",
                      error=msg[:300])
        return None


# --------------------------------------------------------------------------- markdown
_CLASS = {"identified": "identified", "one_sided": "one-sided (bounded on one side only)", "practically_non_identifiable":
          "practically non-identifiable", "structurally_non_identifiable": "structurally non-identifiable",
          "not_fitted": "not fitted", "not_assessable": "not assessable (no adjustment computable at its bound)",
          None: "—"}


def _g(v, nd=3) -> str:
    """``nd`` decimals for ordinary magnitudes, ``nd`` significant digits for very small or large ones."""
    v = _f(v)
    if v is None:
        return "—"
    a = abs(v)
    if a == 0:
        return "0"
    s = f"{v:.{nd}g}" if (a >= 1e4 or a < 0.1) else f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


def _ci(ci) -> str:
    if not ci:
        return "—"
    lo, hi = ci
    return f"[{_g(lo) if lo is not None else 'open'}, {_g(hi) if hi is not None else 'open'}]"


def identifiability_markdown(rep: dict | None) -> str:
    """The report as markdown (the CLI prints it)."""
    if not rep:
        return "_No identifiability study for this run._"
    if rep.get("status") == "disabled":
        return "_Identifiability: the data-only physics fit is anti-physical (a ≤ 0); nothing to profile._"
    L = [f"Profile likelihood per parameter (Raue et al. 2009) with the independence log-likelihood adjusted for "
         f"spatial dependence (Chandler & Bate 2007; clusters = the run's {rep['n_clusters']} CV blocks of "
         f"{_g(rep['cluster'].get('block_m'))} m; n = {rep['n']:,}"
         + (f", {_g(rep['coarse_m'])} m cells" if rep.get("coarse_m") else "") + "). 95% intervals: Λ_A ≤ 3.841, "
         "adjusted with Chandler & Bate's block score covariance and, as a sensitivity, with adjacent blocks' score "
         "cross-products added (a block-level Conley 1999 spatial HAC); the class rests on the wider of the two. "
         "c = naive / block-adjusted variance. Run (fold mean): the run's own value, the mean over its folds "
         "(physics.json). Full-data refit: this study's refit of the run's objective (penalty on) on every labelled "
         "cell, the value the prior share and the pull describe. Prior share: the penalty's share of the run "
         "objective's profile curvature. Posterior: Laplace approximation of l_A + nominal priors (Ribatet et al. 2012), not MCMC. "
         "Power-scaling (Kallioinen et al. 2024): a prior sensitivity ≥ 0.05 marks an informative prior, a likelihood "
         "sensitivity < 0.05 a weak likelihood, both ≥ 0.05 a potential prior–data conflict; 'not reliable' where "
         "it rests on a poor Laplace approximation. Conflict: Evans & Moshonov (2006) tail probability.", "",
         "| parameter | run (fold mean) | full-data refit | data-only | 95% CI (blocks) | 95% CI (neighbours) | "
         "95% CI (naive) | c | class | prior share | pull | posterior mean ± sd | psens prior / lik | conflict p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep.get("parameters") or []:
        cls = _CLASS.get(r.get("identifiability"), r.get("identifiability"))
        if r.get("summary") and r.get("identifiability") not in ("identified", "structurally_non_identifiable"):
            cls += f": {r['summary']}"
        if r.get("at_bound"):
            cls += f"; at {r['at_bound']['side']} bound"
        post = r.get("posterior") or {}
        tag = " (1-D profile)" if post.get("method") == "profile" else ""
        if post.get("mean") is None:
            pm = "—"
        elif post.get("reliable") is False:
            pm = "not reliable (Laplace poor)"        # flagged in the JSON, not a finding
        else:
            pm = f"{_g(post.get('mean'))} ± {_g(post.get('sd'))}{tag}"
        ps = r.get("psens") or {}
        if not ps:
            pss = "—"
        elif ps.get("reliable") is False:
            pss = "not reliable (Laplace poor)"
        else:
            lw = ps.get("diagnosis") in (None, "none") and _f(ps.get("likelihood")) is not None \
                and float(ps["likelihood"]) < PSENS_THRESHOLD
            pss = (f"{_g(ps.get('prior'), 2)} / {_g(ps.get('likelihood'), 2)}"
                   + (f" ({DIAGNOSIS_TEXT.get(ps.get('diagnosis'), ps.get('diagnosis'))})"
                      if ps.get("diagnosis") not in (None, "none") else
                      " (weak likelihood: < 0.05)" if lw else "") + tag)
        cf = r.get("conflict") or {}
        cfs = ("—" if not cf else f"{_g(cf['p'], 2)}" + (" **conflict**" if cf.get("flag") else "")
               if cf.get("p") is not None else "not testable")
        naive = _ci(r.get("ci95_naive")) + (f" {r['ci95_naive_label']}" if r.get("ci95_naive_label") else "")
        L.append(f"| {r['label']} ({r['unit']}) | {_g(r.get('estimate_run_folds'))} | {_g(r.get('estimate_run'))} | "
                 f"{_g(r.get('estimate_data'))} | "
                 f"{_ci(r.get('ci95'))} | {_ci(r.get('ci95_neighbours'))} | {naive} | "
                 f"{_g(r.get('adjust_factor'), 2)} | {cls} | "
                 f"{_g(r.get('prior_share_run'), 2)} | {_g(r.get('pull'), 2)} | {pm} | {pss} | {cfs} |")
    lin = rep.get("linear") or []
    if lin:
        L += ["", "| linear coefficient | data-only ± SE naive / block | full-data refit ± SE naive / "
              + ("adjacent-block" if (rep.get("covariance") or {}).get("used") == "neighbours" else "block")
              + " | c |",
              "|---|---|---|---|"]
        for r in lin:
            L.append(f"| {r['label']} | {_g(r['estimate_data'], 4)} ± {_g(r['se_naive_data'], 3)} / "
                     f"{_g(r['se_robust_data'], 3)} | {_g(r['estimate_run'], 4)} ± {_g(r['se_naive_run'], 3)} / "
                     f"{_g(r['se_robust_run'], 3)} | {_g(r.get('adjust_factor'), 2)} |")
    d = rep.get("derived") or {}
    used = (rep.get("covariance") or {}).get("used") or "blocks"
    for key, name in (("r_a", "r_a"), ("gamma_K_per_km", "γ")):
        x = d.get(key)
        if not x:
            continue
        fj = x.get("fold_jackknife")
        L += ["", f"*{name} ({x['unit']}):* full-data refit {_g(x['run']['value'])} ± {_g(x['run']['se_robust'])} "
                  f"({'adjacent-block' if used == 'neighbours' else 'block'} sandwich; naive ± "
                  f"{_g(x['run']['se_naive'])}), data-only {_g(x['data_only']['value'])} ± "
                  f"{_g(x['data_only']['se_robust'])} (block sandwich)"
              + (f"; fold jackknife {_g(fj['value'])} ± {_g(fj['se'])}" if fj else "") + "."]
    post = rep.get("posterior") or {}
    if post:
        lk, pk = post.get("laplace_parameters"), post.get("profile_parameters")
        L += ["", f"*Laplace adequacy:* importance-sampling ESS {_g(post.get('is_ess'))} of {post.get('n_draws')} "
                  f"draws for exp(l_A + log prior)" + ("" if post.get("laplace_adequate") else
                                                       " — poor: the Laplace posterior summaries and power-scaling"
                                                       + (f" ({', '.join(lk)})" if lk else "")
                                                       + " are not reliable (flagged) and not reported; the profile "
                                                         "intervals do not rely on it")
              + (f". The Laplace step is conditional on {', '.join(post['conditional_on'])} held fixed"
                 if post.get("conditional_on") else "")
              + (f". The posteriors of {', '.join(pk)} are 1-D profile posteriors (exp(−Λ_A/2) × prior), not the "
                 "Laplace step" if pk else "") + "."]
    for w in rep.get("warnings") or []:
        L += ["", f"*Warning:* {w}"]
    return "\n".join(L)
