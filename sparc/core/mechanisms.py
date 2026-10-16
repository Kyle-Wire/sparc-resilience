"""Physics mechanism register and physical-units parameter panel (PHYSICS_PLAN §4, W0 and W3).

Each mechanism of the physics model is written as an if–then claim: IF the mechanism holds, THEN a
signature must appear that fitting one afternoon map did not force.  The test of that signature gives a
verdict (:data:`VERDICTS`):

* ``supported`` — the signature appears;
* ``contradicted`` — it is absent or reversed: the term is then reported as a statistical term, not as
  physics, and cannot be used to explain a scenario;
* ``not_testable`` — this data cannot support the test (one time window, a missing layer, the physics
  term disabled …); the reason is given;
* ``not_yet_tested`` — the test belongs to a workstream that is not built yet; it is named.

A model of an open natural system can be confirmed or refuted by such tests, not verified (Oreskes,
Shrader-Frechette & Belitz 1994); the verdicts mirror the effect verdicts (Robust / Direction only / Not
established), applied to mechanisms.

Physical units (finding F4)
---------------------------
In the local limit the physics model gives ΔT = a·Q_H/norm (norm = SW↓ by day, |LW_net| at night; see
``PhysicsModel._forcing``), while bulk transfer gives ΔT = r_a·Q_H/(ρc_p).  Hence

    r_a = a_K·ρc_p / norm,        ρ = p/(R_d·T),

with a_K the gain in kelvin per unit q (× 5/9 for a °F target: a temperature *difference*, no offset).
The elevation coefficient γ (target units per metre) becomes K/km.  One afternoon map identifies only the
combinations r_a = τ/h and L² = Kτ, not τ, h and K separately; the time scales derived from an *assumed*
mixing depth h are labelled as such.

Uncertainty over the K cross-validation folds is the delete-a-group jackknife already used for every
decision quantity (:meth:`sparc.core.ensemble.FittedEnsemble.jackknife_sd`): SE = sd·√(K−1) = s·(K−1)/√K,
with s the sample sd of the K fold values.  The inflation depends on K, the number of groups (the fold models
share K−2 groups), so when the physics term is disabled in some folds the SE of the remaining m values is
s_m·(K−1)/√K, not the m-replicate formula s_m·(m−1)/√m (which would understate it ≈ 1.5× for m = 3 of K = 5
and ≈ 2.5× for m = 2).

Range tests (M6a, M7a)
----------------------
A fitted value is compared with a published range through its mean ± 2 SE interval:

* Not testable when the interval is not a valid jackknife interval: fewer than three usable folds (with two, the
  interval has one degree of freedom and ±2 SE is far too narrow), or fewer than K−1 of the K folds with the
  physics term on (a subset chosen by the fit itself);
* Contradicted when the interval lies entirely outside the range: the data exclude it, however wide;
* Supported only when the point estimate lies inside the range AND the interval is informative: no wider than
  the range itself (a wider interval cannot lie inside the range, so it cannot tell inside from outside) and,
  for r_a, above zero.  A parameter the data cannot pin down cannot be Supported, whatever its point estimate
  (plan W3), and an interval that only reaches the range from an estimate outside it is not evidence that the
  signature appears;
* Not testable otherwise, with the reason.

Fit bounds (plan F6)
--------------------
A fold value stopped by a fit bound (L at its bounds, s, a₁, κ … at the ends of their sigmoids, the EF slopes and
the water sink at their softplus floor) is not an estimate: the fit wanted to go further.  Such rows are flagged
"at the fit bound in k of m folds: not identified by this run", and when every fold sits there no SE is given
(the folds agree only because the bound stops them).

The ranges are stated for one regime: the lapse-rate band "in the well-mixed afternoon", and r_a's ≈ 20–100
s/m as the plan applies it to the afternoon map (Crawford et al. 2018 is for unstable conditions; the verified
list gives no night-time range).  So both tests need a day window inside the afternoon (:data:`AFTERNOON_LOCAL`,
from the forcing's local hours), and r_a needs the campaign's SW↓: with the generic default forcing,
r_a ∝ 1/SW↓_assumed says nothing about the air.

Identifiability from the data alone (W3)
----------------------------------------
When the run has an identifiability study (:mod:`sparc.core.identifiability`: ``identifiability.json`` / the
manifest's ``identifiability`` section; ``sparc core identifiability -r RUN_DIR``), the register uses it:

* **It must belong to the run** (:func:`_ident_mismatch`): the same number of cells, CV design, input data (when
  both record its SHA-256), target units and window, and the same physics fit — the fold means of the profiled
  parameters the study recorded equal this run's.  Otherwise it is ignored, with the reason in
  ``reg["identifiability"]["note"]`` (a re-run that refits the physics leaves an earlier study behind).
* **Panel rows** of profiled parameters gain the data-only 95% profile intervals (Raue et al. 2009) adjusted for
  spatial dependence two ways: Chandler & Bate (2007)'s block sandwich with the CV blocks as independent clusters
  (the default, ``ci95``), and, as a sensitivity, the same with the score cross-products of adjacent blocks
  added (a block-level spatial HAC in the manner of Conley 1999, ``ci95_neighbours``).  The identifiability
  class rests on the wider of the two per side.  They also gain the prior's share of the run objective's curvature,
  the pull, the calibrated posterior (Ribatet, Cooley & Davison 2012), power-scaling sensitivity with its
  diagnosis (Kallioinen et al. 2024) and the prior–data conflict p (Evans & Moshonov 2006).
* **Verdict rule** (plan W3): a signature cannot be Supported when a key parameter of its mechanism
  (:data:`KEY_PARAMS`) is practically or structurally non-identifiable from the data alone, whatever its point
  estimate; it becomes Not testable, with the reason.  The rule never makes a signature Contradicted.
* **r_a and γ** show the sandwich SE of the full-data reference fit (the block sandwich unless the study was run
  with ``score_covariance: neighbours``) beside the fold jackknife.  The verdict
  stays on the jackknife; a note is added when the two SEs differ by more than :data:`SANDWICH_FACTOR` (an
  operational threshold of the W3 task, not a published one).
* **Lever statements** cite, in plain words, the identifiability of the parameters their role acts through
  (:data:`ROLE_PARAMS`): whether the data alone pin the parameter down or do not exclude zero, and whether the
  study's full-data refit of the run's objective is set mostly by the penalty (prior share above one half: the
  penalty supplies most of the run objective's curvature in it).  The run's own value is its fold mean
  (``run_value``, as the panel shows); the refit (``estimate_run``) is always labelled "full-data refit".

Everything here is a pure function of plain dicts — the manifest, the per-fold physics parameters
(``physics.json``), the config and the identifiability report — so old run directories can be re-evaluated
(``sparc core mechanisms -r RUN_DIR``; :func:`mechanisms_for_run` is the only function that touches files).
Ranges and citations come from the verified reference list of ``docs/roadmap/PHYSICS_PLAN.md``; never add a
range without one.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

__all__ = ["VERDICTS", "VERDICT_LABELS", "TARGET_KINDS", "REGISTER", "REFERENCES", "RA_RANGE", "LAPSE_BAND",
           "AFTERNOON_LOCAL", "KEY_PARAMS", "ROLE_PARAMS", "NOT_IDENTIFIED", "IDENT_LABELS", "SANDWICH_FACTOR",
           "to_kelvin_factor", "rho_cp", "forcing_norm", "physical_parameters", "evaluate", "mechanisms_markdown",
           "identifiability_markdown_table", "forcing_t2m_K", "mechanisms_for_run", "run_identifiability"]

SCHEMA = 1

VERDICTS = ("supported", "contradicted", "not_testable", "not_yet_tested")
VERDICT_LABELS = {"supported": "Supported", "contradicted": "Contradicted",
                  "not_testable": "Not testable with this data", "not_yet_tested": "Not yet tested"}

#: ``data.target_kind``: what the target is.  "product" = an interpolated area-wide product (e.g. CAPA's
#: Heat Watch map), "measurements" = the readings themselves, "simulated" = a synthetic city.
TARGET_KINDS = ("product", "measurements", "simulated")
BASIS = {
    "product": "indicative: fitted to an interpolated product, not measurements (W7)",
    "measurements": "fitted to measurements",
    "simulated": "fitted to simulated data (a synthetic city with planted truths)",
    None: "basis unspecified: data.target_kind is not set, so it is not recorded whether the target is "
          "measurements or an interpolated product",
}

# --------------------------------------------------------------------------- physical constants
R_D = 287.05            # J kg⁻¹ K⁻¹, gas constant of dry air
C_P = 1005.0            # J kg⁻¹ K⁻¹, specific heat of air at constant pressure
P_STD = 101325.0        # Pa: the forcing file carries no surface pressure
T_FALLBACK_K = 293.15   # used (and said) when the forcing gives no 2 m temperature
OMEGA_DAY = 2.0 * math.pi / 86400.0       # s⁻¹, the daily harmonic (W6's quasi-steady argument)
H_ASSUMED_M = (20.0, 50.0)                # canopy-layer mixing depths assumed for the time scales (W6)
#: same defaults as ``physics.PHYSICS_DEFAULTS`` / ``config.DEFAULTS`` (not imported: keeps this module light)
_SW_DEFAULT, _LW_DEFAULT = 800.0, -100.0
#: Fit bounds of the source parameters, from ``physics.PhysicsModel._transform`` (not imported: keeps this module
#: light; a test checks them against physics.py): (low, high, scale).  "lin": a sigmoid onto [low, high];
#: "log": a sigmoid onto [log low, log high]; "floor": a softplus, ≥ 0 with no upper bound.  L's bounds are
#: per run (``L_bounds_m``, log scale).
FIT_BOUNDS = {
    "s": (0.0, 1.0, "lin"), "a1": (0.0, 1.0, "lin"), "a1n": (0.0, 1.0, "lin"),
    "kappa_canopy": (0.05, 20.0, "log"), "shade_c0": (0.05, 0.95, "lin"), "shade_width": (0.02, 0.5, "lin"),
    "e1": (0.0, None, "floor"), "e2": (0.0, None, "floor"), "e3": (0.0, None, "floor"), "w": (0.0, None, "floor"),
}
#: A fold value within this fraction of the bounded range (on the transform's own scale) is at the bound: the
#: sigmoid's slope there is below ≈ 0.5%, so the optimiser has stopped moving it.
AT_BOUND_FRAC = 0.005
#: A softplus-floored value below 10⁻³ of its unit is at the floor (the softplus slope is then < 10⁻³); the water
#: sink is w = 100·softplus(·) W/m² (``physics._W_SCALE``), so its floor tolerance is 0.1 W/m².
FLOOR_TOL = {"e1": 1e-3, "e2": 1e-3, "e3": 1e-3, "w": 0.1}
#: decimals shown per panel row (default 3)
_ND = {"L": 0, "v": 0, "w": 0, "r_a": 1, "gamma": 1}

# --------------------------------------------------------------------------- published ranges (verified list)
RA_RANGE = (20.0, 100.0)                  # s m⁻¹
RA_TEXT = "≈ 20–100 s/m"
RA_SOURCE = ("Masson (2000) canyon exchange formula: ≈ 100 s/m in calm air, ≈ 36 s/m at 5 m/s; "
             "Crawford et al. (2018): ≈ 40 s/m measured in central London")
LAPSE_BAND = (-9.8, -6.5)                 # K km⁻¹
LAPSE_TEXT = "−9.8 to −6.5 K/km"
LAPSE_SOURCE = "dry-adiabatic (−9.8 K/km) and standard-atmosphere (−6.5 K/km) lapse rates"
NIGHT_GAMMA = "no single physical range at night (inversions, cold-air pooling)"
NIGHT_RA = ("the verified reference list gives no night-time (stable) range for r_a: the ≈ 20–100 s/m range is "
            "applied to the afternoon, and Crawford et al. (2018) is for unstable conditions")
#: The window treated as the well-mixed afternoon: a forcing window inside these local hours.  An operational
#: choice of this module, not a published threshold: it admits the Heat Watch protocol's 15–16 run and excludes
#: its 6–7 and 19–20 runs, when the lapse rate and the exchange regime differ (plan F7).
AFTERNOON_LOCAL = (12, 18)

#: Reference keys used by the register and the panel (copied from PHYSICS_PLAN.md's verified list).
REFERENCES = {
    "masson2000": "Masson, V. (2000). A physically-based scheme for the urban energy budget in atmospheric models. "
                  "Boundary-Layer Meteorol. 94, 357–397. doi:10.1023/A:1002463829265",
    "crawford2018": "Crawford, B., Grimmond, C.S.B., Gabey, A., Marconcini, M., Ward, H.C. & Kent, C.W. (2018). "
                    "Variability of urban surface temperatures and implications for aerodynamic energy exchange in "
                    "unstable conditions. Q. J. R. Meteorol. Soc. 144(715), 1719–1741. doi:10.1002/qj.3325",
    "grimmond1991": "Grimmond, C.S.B., Cleugh, H.A. & Oke, T.R. (1991). An objective urban heat storage model and its "
                    "comparison with other schemes. Atmos. Environ. B 25(3), 311–326. "
                    "doi:10.1016/0957-1272(91)90003-W",
    "grimmond1999": "Grimmond, C.S.B. & Oke, T.R. (1999). Heat storage in urban areas: local-scale observations and "
                    "evaluation of a simple model. J. Appl. Meteorol. 38(7), 922–940.",
    "oke1973": "Oke, T.R. (1973). City size and the urban heat island. Atmos. Environ. 7(8), 769–779. "
               "doi:10.1016/0004-6981(73)90140-6",
    "oke1981": "Oke, T.R. (1981). Canyon geometry and the nocturnal urban heat island. J. Climatol. 1(3), 237–254. "
               "doi:10.1002/joc.3370010304",
    "sailor2011": "Sailor, D.J. (2011). A review of methods for estimating anthropogenic heat and moisture emissions "
                  "in the urban environment. Int. J. Climatol. 31(2), 189–199. doi:10.1002/joc.2106",
    "varquez2021": "Varquez, A.C.G., Kiyomoto, S., Khanh, D.N. & Kanda, M. (2021). Global 1-km present and future "
                   "hourly anthropogenic heat flux. Sci. Data 8, 64. doi:10.1038/s41597-021-00850-w",
    "whittle1954": "Whittle, P. (1954). On stationary processes in the plane. Biometrika 41(3–4), 434–449. "
                   "doi:10.1093/biomet/41.3-4.434",
    "bendat2010": "Bendat, J.S. & Piersol, A.G. (2010). Random Data: Analysis and Measurement Procedures, 4th ed. "
                  "Wiley. doi:10.1002/9781118032428",
    "shandas2019": "Shandas, V., Voelkel, J., Williams, J. & Hoffman, J. (2019). Integrating satellite and ground "
                   "measurements for predicting locations of extreme urban heat. Climate 7(1), 5. "
                   "doi:10.3390/cli7010005",
    "oreskes1994": "Oreskes, N., Shrader-Frechette, K. & Belitz, K. (1994). Verification, validation, and "
                   "confirmation of numerical models in the Earth sciences. Science 263(5147), 641–646. "
                   "doi:10.1126/science.263.5147.641",
    "mooij2013": "Mooij, J.M., Janzing, D. & Schölkopf, B. (2013). From ordinary differential equations to "
                 "structural causal models: the deterministic case. Proc. UAI 29, 440–448. arXiv:1304.7920",
    # W3 identifiability study (sparc.core.identifiability.REFERENCES; a test checks they agree)
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
    "conley1999": "Conley, T.G. (1999). GMM estimation with cross sectional dependence. Journal of Econometrics "
                  "92(1), 1–45. doi:10.1016/S0304-4076(98)00084-0",
    "selfliang1987": "Self, S.G. & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and "
                     "likelihood ratio tests under nonstandard conditions. Journal of the American Statistical "
                     "Association 82(398), 605–610",
}
_IDENT_REFS = ("raue2009", "chandler2007", "ribatet2012", "kallioinen2024", "evans2006", "conley1999",
               "selfliang1987")

# --------------------------------------------------------------------------- W3 identifiability (constants)
#: The key parameters of each mechanism, as keys of the identifiability report.  A signature cannot be Supported
#: while one of its mechanism's nonlinear key parameters is not identified by the data alone (plan W3).  The linear
#: coefficients ``a`` (M6, r_a) and ``gamma`` (M7) have no profile class: the study gives their block-sandwich SE,
#: shown beside the fold jackknife, which keeps the verdict.
KEY_PARAMS = {"M1": ("s", "kappa_canopy", "shade_c0", "shade_width"), "M2": ("e0", "e1", "e2", "e3"),
              "M3": ("a1", "a1n"), "M4": ("L",), "M5": ("v",), "M6": ("a",), "M7": ("gamma",), "M8": ("w",)}
#: The source parameters each lever's physics role acts through; its lever statement cites their identifiability.
#: Albedo has none of its own: it enters through absorbed sunlight, scaled by the gain a and spread by L.
ROLE_PARAMS = {"canopy": ("s", "kappa_canopy", "shade_c0", "shade_width", "e2"), "impervious": ("a1", "a1n", "e3"),
               "ndvi": ("e1",), "water_distance": ("w",)}
#: Raue et al. (2009)'s classes that block Supported: a parameter bounded on one side only is practically
#: non-identifiable on the other (``one_sided``), and one whose adjustment is not computable is not assessable
NOT_IDENTIFIED = ("practically_non_identifiable", "structurally_non_identifiable", "not_assessable", "one_sided")
IDENT_LABELS = {"identified": "identified", "practically_non_identifiable": "practically non-identifiable",
                "one_sided": "one-sided: bounded by the data on one side only",
                "not_assessable": "not assessable: on its fit bound, where no adjustment for spatial dependence is "
                                  "computable",
                "structurally_non_identifiable": "structurally non-identifiable", "not_fitted": "not fitted"}
#: Short plain-language labels (the results card's pills)
IDENT_SHORT = {"identified": "identified", "one_sided": "one side only", "practically_non_identifiable": "not identified",
               "structurally_non_identifiable": "not identified (flat)", "not_assessable": "not assessable",
               "not_fitted": "not fitted"}
#: Kallioinen et al. (2024)'s reading of power-scaling sensitivity (threshold 0.05, as ArviZ / priorsense): a prior
#: sensitivity ≥ 0.05 marks an informative prior, a likelihood sensitivity < 0.05 a weak (non-informative)
#: likelihood, and both ≥ 0.05 a potential prior–data conflict.
PSENS_THRESHOLD = 0.05
PSENS_READING = ("a prior sensitivity of 0.05 or more marks an informative prior; a likelihood sensitivity below "
                 "0.05 marks a weak (non-informative) likelihood; both at 0.05 or more is a potential prior–data "
                 "conflict (Kallioinen et al. 2024)")
#: Parameters whose physical lower limit is 0 (a fraction, a non-negative slope, sink or drift;
#: ``identifiability.PHYSICAL_LIMITS``): an interval that reaches 0 means the data do not exclude "no effect".
_ZERO_LIMIT = ("s", "a1", "a1n", "e1", "e2", "e3", "w", "v")
_SYM = {"L": "L", "v": "|v|", "s": "s", "kappa_canopy": "κ", "shade_c0": "c₀", "shade_width": "the shade width",
        "a1": "a₁", "a1n": "a₁ₙ", "e0": "e₀", "e1": "e₁", "e2": "e₂", "e3": "e₃", "w": "w"}
#: report key → parameter-panel row key, where they differ (the panel shows σ(e₀), the report e₀ itself)
_PANEL_KEY = {"e0": "ef0"}
_PSENS_TEXT = {"prior_data_conflict": "power-scaling flags a potential prior–data conflict (prior and likelihood "
                                      "sensitivity both 0.05 or more)",
               "weak_likelihood": "power-scaling flags an informative prior with a weak likelihood (prior sensitivity "
                                  "0.05 or more, likelihood sensitivity below 0.05)",
               "likelihood_only_weak": "power-scaling flags a weak likelihood (likelihood sensitivity below 0.05)"}
_PSENS_SHORT = {"prior_data_conflict": "potential prior–data conflict",
                "weak_likelihood": "informative prior, weak likelihood",
                "likelihood_only_weak": "weak likelihood"}
#: The register's note on a study that profiled nothing (``status: "disabled"``)
DISABLED_NOTE = ("the identifiability study profiled nothing: its data-only physics fit (every prior off) is "
                 "anti-physical (gain a ≤ 0), so no parameter's identifiability, prior share or prior sensitivity was "
                 "assessed (re-run `sparc core identifiability` after the physics fit changes)")
#: r_a and γ: the block-sandwich SE of the full-data reference and the fold jackknife SE are noted when they differ
#: by more than this factor (an operational threshold of the W3 task, not a published one).
SANDWICH_FACTOR = 2.0

# --------------------------------------------------------------------------- the register (plan §4)
_ONE_WINDOW = "one time window; needs the multi-window fit (W6)"
_ALL = "*"          # a transport mechanism: it shapes the effect of every lever that enters the source
_NORM_TEXT = "norm (norm = SW↓ by day, |LW_net| at night)"      # replaced by the run's normaliser in evaluate()

#: Ten mechanisms of plan §4.  ``kind`` of a signature: ``evaluate`` (tested here), ``multi_window``
#: (not testable from one window), ``workstream`` (the test is not built yet) or ``missing_term``
#: (the term is not in the model).  ``roles``: the physics roles whose effect the mechanism explains.
#: ``shortwave``: the pathway exists only with sunlight — the night source (Q* = LW_net) has no albedo or shade
#: term (``PhysicsModel._q_points_torch``), so M1 does not act in a night window.
REGISTER = (
    {"id": "M1", "name": "Absorbed sunlight", "terms": "albedo, canopy shade", "in_model": True, "shortwave": True,
     "chain": "albedo, canopy → absorbed shortwave SW↓·(1 − albedo)·(1 − s·h(canopy)) → Q* → Q_H → q → "
              "transport G → ΔT",
     "roles": ("albedo", "canopy"), "refs": (),
     "signatures": (
         {"sig": "a", "then": "Its effect per unit albedo or canopy scales with SW↓ across runs, and the shading "
                              "term vanishes at night.",
          "test": "Multi-window fit", "workstream": "W6", "kind": "multi_window"},
         {"sig": "b", "then": "Its spatial footprint follows G.",
          "test": "Band-transfer test", "workstream": "W4", "kind": "workstream"})},
    {"id": "M2", "name": "Evaporative cooling", "terms": "evaporative fraction from NDVI, canopy, impervious",
     "in_model": True,
     "chain": "NDVI, canopy, impervious → EF = σ(e₀ + e₁·ndvi + e₂·canopy − e₃·impervious) → "
              "Q_E = EF·(Q* − ΔQ_S) → Q_H → q → G → ΔT",
     "roles": ("ndvi", "canopy", "impervious"), "refs": (),
     "signatures": (
         {"sig": "a", "then": "Cooling scales with available energy Q* − ΔQ_S, so it is smaller in the morning and "
                              "evening runs.",
          "test": "Multi-window fit", "workstream": "W6", "kind": "multi_window",
          "note": "NDVI is a mediator, so scenarios route through it (the mediator handling already exists)."},)},
    {"id": "M3", "name": "Heat storage", "terms": "impervious, OHM", "in_model": True,
     "chain": "impervious → ΔQ_S = a₁·imp·Q* by day (stored), −a₁ₙ·imp·|LW_net| at night (released) → Q_H → q → "
              "G → ΔT",
     "roles": ("impervious",), "refs": ("grimmond1991", "grimmond1999"),
     "signatures": (
         {"sig": "a", "then": "Impervious cover warms the 15–16 run less than its absorbed energy implies, and "
                              "warms the 19–20 and 0–1 runs more.",
          "test": "Multi-window fit", "workstream": "W6", "kind": "multi_window"},
         {"sig": "b", "then": "Fitted a1 and a2 fall inside the published OHM ranges (Grimmond & Oke 1999).",
          "test": "Parameter panel with the OHM source of the multi-window fit", "workstream": "W3, W6",
          "kind": "multi_window",
          "reason": "one time window: the hysteresis coefficient a2 needs runs across the day, and this model's "
                    "a1 multiplies impervious fraction × Q*, not the per-surface OHM form (W6 adds it)"})},
    {"id": "M4", "name": "Horizontal mixing", "terms": "mixing length L (L² = Kτ)", "in_model": True,
     "chain": "q → transport G with mixing length L: heating patterns much larger than 2πL reach the air almost "
              "whole, smaller ones are smeared out → ΔT",
     "roles": (_ALL,), "refs": ("bendat2010", "whittle1954", "shandas2019"),
     "signatures": (
         {"sig": "a", "then": "Band gains of ΔT on q fall as 1/(1 + L²k²).",
          "test": "Band-transfer test", "workstream": "W4", "kind": "workstream"},
         {"sig": "b", "then": "The residual spectrum matches the implied Matérn ν = 1.",
          "test": "Residual band variances against the Matérn ν = 1 shape of the fitted L", "workstream": "W4",
          "kind": "workstream"},
         {"sig": "c", "then": "L fitted on the raw traverses ≈ L fitted on the map.",
          "test": "Product-vs-atmosphere fit on the raw traverses", "workstream": "W7", "kind": "workstream",
          "reason": "needs the raw traverse readings (W7)"})},
    {"id": "M5", "name": "Advection", "terms": "advection offset v = u·τ", "in_model": True,
     "chain": "q → G shifted downwind by v → heat appears downwind of its source",
     "roles": (_ALL,), "refs": (),
     "signatures": (
         {"sig": "a", "then": "The q→ΔT cross-correlogram peaks upwind of the receptor, and the transfer phase is "
                              "non-zero along the wind.",
          "test": "Cross-correlogram peak offset under rotated winds", "workstream": "W4", "kind": "workstream"},
         {"sig": "b", "then": "A street's cooling follows the canopy upwind of it when the wind changes between "
                              "runs.",
          "test": "identify/windshift.py, extended to the physics", "workstream": "W7", "kind": "workstream",
          "reason": "needs the raw traverses (W7); the wind-shift design is built but had 8% power on one day"})},
    {"id": "M6", "name": "Vertical exchange", "terms": "aerodynamic resistance r_a = τ/h", "in_model": True,
     "chain": "Q_H → ΔT = r_a·Q_H/(ρc_p): the gain a sets r_a = a·ρc_p/" + _NORM_TEXT,
     "roles": (_ALL,), "refs": ("masson2000", "crawford2018", "oke1973"),
     "signatures": (
         {"sig": "a", "then": "The implied r_a lies in the published range for urban surfaces.",
          "test": "Parameter panel: r_a = a·ρc_p/norm, mean ± 2 SE against " + RA_TEXT + " (afternoon window)",
          "workstream": "W3", "kind": "evaluate"},
         {"sig": "b", "then": "Across runs and campaigns, the gain a falls as wind speed rises; heat-island "
                              "intensity weakens with wind (Oke 1973).",
          "test": "Multi-window and multi-campaign fits", "workstream": "W6", "kind": "multi_window"})},
    {"id": "M7", "name": "Elevation (lapse rate)", "terms": "γ·elevation", "in_model": True,
     "chain": "elevation → γ·elev (an additive term beside the transported source) → ΔT",
     "roles": ("elevation",), "refs": (),
     "signatures": (
         {"sig": "a", "then": "γ lies between about −9.8 and −6.5 K/km in the well-mixed afternoon.",
          "test": "Parameter panel: γ in K/km, mean ± 2 SE against the lapse-rate band (afternoon window)",
          "workstream": "W3", "kind": "evaluate"},
         {"sig": "b", "then": "At night it can be near zero or positive (inversions, cold-air pooling).",
          "test": "Multi-window fit (day–night contrast)", "workstream": "W6", "kind": "multi_window"})},
    {"id": "M8", "name": "Water bodies", "terms": "water-cell sink w, distance to water", "in_model": True,
     "chain": "water → sink w·1[water cell] in Q_H (and the water-distance predictor) → q → G → ΔT",
     "roles": ("water_distance",), "refs": (),
     "signatures": (
         {"sig": "a", "then": "Cooling near water is stronger downwind of it under the campaign wind, and water "
                              "just outside the study area counts too.",
          "test": "Directional water-distance response after source completion", "workstream": "W2",
          "kind": "workstream"},)},
    {"id": "M9", "name": "Longwave trapping", "terms": "sky view factor (not in the model)", "in_model": False,
     "chain": "sky view factor → trapped longwave → Q* → Q_H → ΔT (not modelled)",
     "roles": (), "refs": ("oke1981",),
     "signatures": (
         {"sig": "a", "then": "After land cover is accounted for, the residual ΔT tracks sky view factor, mostly "
                              "at night (Oke 1981).",
          "test": "Residual vs sky view factor by run", "workstream": "W8", "kind": "missing_term"},)},
    {"id": "M10", "name": "Anthropogenic heat", "terms": "Q_F (not in the model)", "in_model": False,
     "chain": "anthropogenic heat Q_F → Q_H → ΔT (not modelled)",
     "roles": (), "refs": ("sailor2011", "varquez2021"),
     "signatures": (
         {"sig": "a", "then": "After land cover, the residual tracks a Q_F map at magnitudes inside published "
                              "ranges (Sailor 2011).",
          "test": "Residual vs AH4GUC Q_F (Varquez et al. 2021)", "workstream": "W8", "kind": "missing_term"},)},
)

#: plain statements written when a mechanism is Contradicted (scenario text cites them)
GATE_TEXT = {
    "M6": "The physics gain does not behave as vertical exchange (implied r_a outside the published range): its "
          "size is a fitted scale, not physics.",
    "M7": "Elevation enters as a statistical term, not as a lapse rate.",
}
ROLE_LABELS = {"albedo": "albedo", "canopy": "canopy", "impervious": "impervious cover", "ndvi": "NDVI",
               "elevation": "elevation", "water_distance": "distance to water"}
_PRECEDENCE = ("contradicted", "supported", "not_yet_tested", "not_testable")


# --------------------------------------------------------------------------- small helpers
def _num(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _jackknife_se(values, n_groups: int | None = None) -> float | None:
    """Delete-a-group jackknife SE over fold models (the repo's decision-quantity helper).

    With all K fold values it is ``FittedEnsemble.jackknife_sd``: sd·√(K−1) = s·(K−1)/√K.  With only m < K of them
    (``n_groups`` = K; the physics term disabled in the others) the inflation stays the K-group one,
    s_m·(K−1)/√K with s_m the sample sd of the m values: the replicates still share K−2 groups each."""
    import numpy as np

    from sparc.core.ensemble import FittedEnsemble

    v = np.asarray(values, float)
    if v.size < 2:
        return None
    K = int(n_groups or v.size)
    if K <= v.size:
        return float(FittedEnsemble.jackknife_sd(v))
    return float(v.std(ddof=1) * (K - 1) / math.sqrt(K))


def to_kelvin_factor(units) -> float | None:
    """Factor converting a temperature *difference* in ``units`` to kelvin (None: not a temperature scale)."""
    u = str(units or "").strip().lower()
    if u in ("degf", "f", "°f", "fahrenheit"):
        return 5.0 / 9.0
    if u in ("degc", "c", "°c", "celsius", "k", "kelvin"):
        return 1.0
    return None


def rho_cp(t_K: float, p_Pa: float = P_STD) -> float:
    """Volumetric heat capacity of air ρc_p (J m⁻³ K⁻¹), ideal gas ρ = p/(R_d·T)."""
    return p_Pa / (R_D * float(t_K)) * C_P


def forcing_norm(physics: dict | None, window: str = "day") -> tuple[float, str]:
    """The normaliser of q (``PhysicsModel._forcing``): SW↓ by day, |LW_net| at night or without sunlight."""
    ph = physics or {}
    sw = float(ph.get("sw_down", _SW_DEFAULT)) if window == "day" else 0.0
    lw = float(ph.get("lw_net", _LW_DEFAULT))
    if sw > 0:
        return sw, "SW↓"
    return max(abs(lw), 1e-9), "|LW_net|"


def _forcing_known(physics: dict | None, norm_label: str) -> tuple[bool, str]:
    """Whether q's normaliser is the campaign's (a forcing file, or a value set in the config) or the generic
    default.  r_a ∝ 1/norm and the fitted gain does not rescale to compensate (q = Q_H/SW↓ hardly depends on
    SW↓), so an assumed normaliser makes r_a an assumption too."""
    ph = physics or {}
    fi = ph.get("forcing_info")
    if isinstance(fi, dict) or ph.get("forcing"):
        return True, f"campaign forcing ({(fi or {}).get('file') or ph.get('forcing')})"
    key, default = ("sw_down", _SW_DEFAULT) if norm_label == "SW↓" else ("lw_net", _LW_DEFAULT)
    val = ph.get(key, default)
    if not _num(val) or float(val) == default:
        return False, "the generic default, assumed: no forcing file"
    return True, "set in the config (no forcing file)"


def _afternoon(physics: dict | None, window: str) -> tuple[bool, str | None]:
    """Whether a day window is the well-mixed afternoon (:data:`AFTERNOON_LOCAL`) the daytime ranges hold in,
    from the forcing's local hours; the reason when it is not or cannot be told."""
    if window != "day":
        return False, None                       # night: each test gives its own reason
    fi = (physics or {}).get("forcing_info")
    hrs = fi.get("hours_local") if isinstance(fi, dict) else None
    if not (isinstance(hrs, (list, tuple)) and len(hrs) == 2 and all(_num(h) for h in hrs)):
        return False, ("time of day unknown: the run records no forcing window (physics.forcing_info.hours_local), "
                       "and the published range holds for the well-mixed afternoon")
    h0, h1 = float(hrs[0]), float(hrs[1])
    if AFTERNOON_LOCAL[0] <= h0 and h1 <= AFTERNOON_LOCAL[1]:
        return True, None
    return False, (f"the {h0:g}–{h1:g} h local window is not the well-mixed afternoon (taken here as a window inside "
                   f"{AFTERNOON_LOCAL[0]}–{AFTERNOON_LOCAL[1]} h): the lapse rate and the exchange regime change "
                   "through the day (F7)")


def _active(p: dict) -> bool:
    """A fold whose physics term is on: positive gain and no anti-physical fallback."""
    return _num(p.get("a")) and float(p["a"]) > 0 and not p.get("fit_warning")


def _physics_off(cfg: dict | None) -> str | None:
    """Why the config fits no physics model (the pipeline's test, ``pipeline._base_model_names``); None when it
    does or the config does not say."""
    c = cfg or {}
    models, ph = c.get("models") or {}, c.get("physics") or {}
    if not models.get("physics", True):
        return "models.physics is off"
    if not ph.get("enabled", True):
        return "physics.enabled is false"
    if "roles" in ph and not (ph.get("roles") or {}):
        return "physics.roles is empty"
    return None


def _basis(target_kind) -> tuple[str | None, str]:
    kind = target_kind if target_kind in TARGET_KINDS else (None if target_kind in (None, "") else target_kind)
    if kind in BASIS:
        return kind, BASIS[kind]
    return kind, f"basis: data.target_kind = {kind!r} is not one of {', '.join(TARGET_KINDS)}"


class _Folds:
    """Per-fold physics parameters (``physics.json``) or, for runs without it, the manifest's mean/sd summary."""

    def __init__(self, params, summary, n_folds):
        self.params = [p for p in (params or []) if isinstance(p, dict)]
        self.summary = summary if isinstance(summary, dict) and "a" in summary else None
        self.source = "physics.json" if self.params else ("manifest summary" if self.summary else None)
        if self.params:
            self.n = len(self.params)
            self.active = [p for p in self.params if _active(p)]
            self.n_active = len(self.active)
            self.warnings = sorted({str(p["fit_warning"]) for p in self.params if p.get("fit_warning")})
        else:
            self.n = int(n_folds or 0) if self.summary else 0
            on = bool(self.summary) and _num(self.summary["a"].get("mean")) and float(self.summary["a"]["mean"]) > 0
            self.n_active = self.n if on else 0
            self.active = []
            self.warnings = []

    @property
    def present(self) -> bool:
        return self.source is not None

    def first(self, key, default=None):
        return self.params[0].get(key, default) if self.params else default

    def stat(self, key: str, scale: float = 1.0, transform=None) -> dict | None:
        """Mean, jackknife SE and per-fold values of ``key`` × ``scale`` over the folds with physics on; ``n`` the
        folds used, ``n_groups`` the K folds of the run (the jackknife's groups)."""
        if self.params:
            vals = [float(p[key]) for p in self.active if _num(p.get(key))]
            if not vals:
                return None
            f = transform or (lambda x: x * scale)
            vals = [float(f(v)) for v in vals]
            return {"value": sum(vals) / len(vals), "se": _jackknife_se(vals, self.n), "per_fold": vals,
                    "n": len(vals), "n_groups": self.n}
        if self.summary and self.n_active and isinstance(self.summary.get(key), dict):
            s = self.summary[key]
            if not _num(s.get("mean")):
                return None
            if transform is not None:          # nonlinear: no SE from a mean/sd summary
                return {"value": float(transform(float(s["mean"]))), "se": None, "per_fold": None, "n": self.n,
                        "n_groups": self.n}
            se = float(s.get("sd") or 0.0) * math.sqrt(self.n - 1) * abs(scale) if self.n > 1 else None
            return {"value": float(s["mean"]) * scale, "se": se, "per_fold": None, "n": self.n, "n_groups": self.n}
        return None


def _interval(st: dict | None) -> list[float] | None:
    """Mean ± 2 SE; None without a value or without a jackknife SE (fewer than two usable folds)."""
    if not st or st.get("value") is None or st.get("se") is None:
        return None
    return [st["value"] - 2.0 * st["se"], st["value"] + 2.0 * st["se"]]


def _fold_precision(st: dict | None) -> str | None:
    """Why the ±2 SE interval of ``st`` is not a valid jackknife interval (None when it is): no SE (one usable
    fold), fewer than K−1 of the K folds (a subset of the delete-a-group replicates chosen by the fit itself), or
    fewer than three fold values (one degree of freedom: ±2 SE is far too narrow)."""
    if _interval(st) is None:
        return "only one fold with the physics term on: no jackknife interval"
    n, K = st.get("n"), st.get("n_groups")
    if n is None:
        return None
    if K and n < K - 1:
        return (f"the physics term is on in only {n} of {K} folds: an interval from a subset of the delete-a-group "
                "replicates, chosen by the fit itself (the others were anti-physical), is not a valid jackknife "
                f"interval (a test needs at least {K - 1})")
    if n < 3:
        return (f"only {n} folds with the physics term on: their interval has one degree of freedom, so ±2 SE is far "
                "too narrow (a test needs at least three)")
    return None


def _fmt(v, nd=1) -> str:
    if not _num(v):
        return "—"
    s = f"{float(v):.{nd}f}"
    return s.replace("-", "−")


def _range_check(st: dict, band: tuple[float, float], sym: str, *, positive: bool = False) -> dict:
    """A fitted value against a published range (module docstring, "Range tests").

    ``status`` (the panel) is where the point estimate lies, inside / outside.  ``verdict`` (the register) comes
    from the mean ± 2 SE interval: not testable unless it is a valid jackknife interval (:func:`_fold_precision`);
    contradicted if it lies entirely outside the range; supported only if the point estimate lies inside the range
    and the interval is informative — no wider than the range (a wider interval cannot lie inside it, so it cannot
    tell inside from outside) and, for a ``positive`` quantity, above zero; otherwise not testable, with ``why``.
    ``informative``: the interval is precise enough to test the range (whatever the verdict)."""
    v = st["value"]
    inside = band[0] <= v <= band[1]
    out = {"status": "inside" if inside else "outside", "overlaps": None, "informative": False,
           "verdict": "not_testable", "why": None}
    iv = _interval(st)
    if iv is not None:
        out["overlaps"] = bool(iv[1] >= band[0] and iv[0] <= band[1])
    bad = _fold_precision(st)
    if bad:
        out["why"] = f"{bad}, so the range cannot be tested"
        return out
    span = f"±2 SE: {_fmt(iv[0])} to {_fmt(iv[1])}"
    if not out["overlaps"]:
        out.update(informative=True, verdict="contradicted")
    elif iv[1] - iv[0] > band[1] - band[0]:
        out["why"] = (f"interval too wide to test the range ({span}, wider than the range itself): {sym} is not "
                      "pinned down by this run")
    elif positive and iv[0] <= 0:
        out["why"] = f"the interval reaches {sym} ≤ 0, which is not physical ({span}): {sym} is not pinned down"
    elif not inside:
        out.update(informative=True,
                   why=(f"inconclusive: the point estimate {sym} = {_fmt(v)} lies outside the range and the ±2 SE "
                        f"interval only reaches it ({span}), so the data neither place {sym} inside the range nor "
                        "exclude it"))
    else:
        out.update(informative=True, verdict="supported")
    return out


def _stat_text(sym: str, st: dict | None, unit: str, nd: int = 1) -> str | None:
    if not st:
        return None
    se = f" ± {_fmt(st['se'], nd)}" if st.get("se") is not None else ""
    return f"{sym} = {_fmt(st['value'], nd)}{se} {unit}".rstrip()


# --------------------------------------------------------------------------- W3 parameter panel
def _context(manifest: dict | None, fold_params, config: dict | None, units=None) -> dict:
    m = manifest or {}
    cfg = config if config is not None else (m.get("config") or {})
    ph = cfg.get("physics") or {}
    data = cfg.get("data") or {}
    units = units if units is not None else data.get("target_units")
    off = _physics_off(cfg)
    # a config without a physics model has no physics parameters, whatever files lie in the run directory
    folds = _Folds(None, None, 0) if off else _Folds(fold_params, m.get("physics"), (m.get("cv") or {}).get("n_folds"))
    window = str(folds.first("window") or ph.get("window") or "day").lower()
    roles_used = folds.first("roles_used")
    if roles_used is None:
        roles_used = [r for r, col in (ph.get("roles") or {}).items() if col]
    shade_form = str(folds.first("shade_form") or ph.get("shade_form") or "saturating").lower()
    return {"cfg": cfg, "physics": ph, "units": units, "folds": folds, "window": window,
            "roles_used": sorted(roles_used), "shade_form": shade_form, "physics_off": off,
            # sunlight reaches the source: a day window with SW↓ > 0 (else Q* = LW_net, no albedo or shade term)
            "shortwave": window == "day" and forcing_norm(ph, window)[1] == "SW↓",
            "target_kind": data.get("target_kind"), "manifest": m}


def _bound_of(key: str, fold: dict | None) -> tuple[float, float | None, str] | None:
    """(low, high, scale) of a fitted parameter's fit bound (:data:`FIT_BOUNDS`; L from the fold's
    ``L_bounds_m``), None for an unbounded one."""
    if key == "L_m":
        lb = (fold or {}).get("L_bounds_m")
        if isinstance(lb, (list, tuple)) and len(lb) == 2 and all(_num(x) and float(x) > 0 for x in lb):
            return float(lb[0]), float(lb[1]), "log"
        return None
    return FIT_BOUNDS.get(key)


def _which_bound(v: float, bnd: tuple[float, float | None, str], key: str) -> tuple[str, float] | None:
    """("lower" | "upper", bound) when ``v`` sits at a fit bound (:data:`AT_BOUND_FRAC`, :data:`FLOOR_TOL`)."""
    lo, hi, scale = bnd
    if scale == "floor":
        return ("lower", lo) if v <= lo + FLOOR_TOL.get(key, 1e-3) else None
    if scale == "log":
        if v <= 0:
            return "lower", lo
        x, a, b = math.log(v), math.log(lo), math.log(hi)
    else:
        x, a, b = v, lo, hi
    tol = AT_BOUND_FRAC * (b - a)
    if x <= a + tol:
        return "lower", lo
    if x >= b - tol:
        return "upper", hi
    return None


def _flag_bound(row: dict, key: str, F: "_Folds") -> dict:
    """Flag a panel row whose fold values sit at a fit bound (plan F6): such a value is where the bound stopped
    the fit, not an estimate.  Sets ``at_bound`` and the note; with every fold there, drops the SE and interval
    (the folds agree only because the bound stops them).  Rows without per-fold values (manifest summary) are
    checked through their mean (at a bound only if every fold is)."""
    if row.get("value") is None:
        return row
    if F.params:
        folds = [p for p in F.active if _num(p.get(key))]
        pairs = [(float(p[key]), _bound_of(key, p)) for p in folds]
    else:
        pairs = [(float(row["value"]), _bound_of(key, None))] if _bound_of(key, None) else []
    hits = [h for v, b in pairs if b is not None for h in [_which_bound(v, b, key)] if h is not None]
    if not hits:
        return row
    n = len(pairs) if F.params else int(row.get("n") or 1)
    k = len(hits) if F.params else n
    sides = sorted({s for s, _ in hits})
    side = sides[0] if len(sides) == 1 else "lower and upper"
    nd = _ND.get(row["key"], 3)
    bound = " and ".join(f"{_fmt(b, nd)}" for b in sorted({b for _, b in hits}))
    unit = f" {row['unit']}" if row["key"] in ("L", "w") else ""
    every = k == n
    text = (f"at the {side} fit bound ({bound}{unit}) in {k} of {n} folds: not identified by this run"
            + (" (no SE: the folds agree only because the bound stops them)" if every else
               " (the SE treats the bound value as an estimate, so it understates the uncertainty)"))
    row["at_bound"] = {"side": side, "bound": [b for _, b in hits][0] if len(sides) == 1 else None, "k": k, "n": n,
                       "every_fold": every, "text": text}
    if every:
        row.update(se=None, interval=None)
    row["status_note"] = "; ".join(x for x in (text, row.get("status_note")) if x)
    return row


def _flag_grid(row: dict, F: "_Folds") -> dict:
    """Flag the L row when the grid, not L, sets the diffusion (W1, :func:`operators.grid_resolution`): a fold
    whose fit has a cell Péclet number above 1 reports ``L_resolved`` False.  With every active fold's diffusion
    independent of L on both axes (elasticity d ln D/d ln L below 0.1, e.g. the hybrid scheme past Pe = 1 on both
    axes) L is not identified, and the SE and interval are dropped as for a fit bound.  Old runs carry no
    resolution keys and are left as they are."""
    if row.get("value") is None or not F.params:
        return row
    folds = [p for p in F.active if p.get("L_resolved") is not None]
    hits = [p for p in folds if p.get("L_resolved") is False]
    if not hits:
        return row
    k, n = len(hits), len(folds)
    lmin = [float(p["L_min_resolved_m"]) for p in hits if _num(p.get("L_min_resolved_m"))]
    flat = [p for p in hits
            if all(_num(p.get(e)) and float(p[e]) < 0.1 for e in ("L_elasticity_x", "L_elasticity_y"))]
    every = len(flat) == n
    text = (f"grid-limited in {k} of {n} folds: a cell Péclet number above 1"
            + (f" (L below about {max(lmin):.0f} m is not resolved on these cells at the fitted drift)" if lmin else "")
            + ("; the diffusion no longer depends on L on either axis, so L is not identified (no SE)" if every else
               "; along that axis the grid, not L, sets the diffusion, so L is weakly identified"))
    row["grid_limited"] = {"k": k, "n": n, "L_min_resolved_m": max(lmin) if lmin else None, "every_fold": every,
                           "text": text}
    if every:
        row.update(se=None, interval=None)
    row["status_note"] = "; ".join(x for x in (text, row.get("status_note")) if x)
    return row


def _value_text(row: dict, bound: bool = True) -> str:
    """'value ± SE unit', with the fit-bound flag (``bound``) when the fold values sit at a bound."""
    if row.get("value") is None:
        return "—"
    nd = _ND.get(row["key"], 3)
    s = _fmt(row["value"], nd) + (f" ± {_fmt(row['se'], nd)}" if row.get("se") is not None else "") + f" {row['unit']}"
    ab = row.get("at_bound")
    if ab and bound:
        s += f" (at the {ab['side']} fit bound in {ab['k']} of {ab['n']} folds: not identified)"
    gl = row.get("grid_limited")
    if gl and bound:
        s += f" (grid-limited in {gl['k']} of {gl['n']} folds: " + ("not identified)" if gl["every_fold"] else
                                                                    "weakly identified)")
    return s


def _source_rows(ctx: dict) -> list[dict]:
    """The fitted source parameters (s, κ, a₁/a₁ₙ, EF slopes, water sink) with their priors as information."""
    F, roles, day = ctx["folds"], set(ctx["roles_used"]), ctx["window"] == "day"
    sun = ctx["shortwave"]                        # the shade terms multiply SW↓: none without sunlight
    priors = F.first("priors") or ctx["physics"].get("priors") or {}
    sigmoid = ctx["shade_form"] == "sigmoid"
    no_range = "no published range for this coefficient in the plan's verified reference list"
    specs = [
        ("s", "Canopy shading strength s", "fraction of absorbed sunlight", sun and "canopy" in roles, "s",
         "Appendix D: with a tight prior (sd 0.1) s stayed near its prior even for a displaced canopy layer; "
         "whether this map identifies s is the profile-likelihood test (W3)", ()),
        ("kappa_canopy", "Shade saturation scale κ", "canopy fraction", sun and "canopy" in roles and not sigmoid,
         None, "weak log-normal regulariser log κ ~ N(0, 3²); large κ is a linear shade response", ()),
        ("shade_c0", "Shade inflection c₀", "canopy fraction", sun and "canopy" in roles and sigmoid, None,
         no_range, ()),
        ("shade_width", "Shade transition width", "canopy fraction", sun and "canopy" in roles and sigmoid, None,
         no_range, ()),
        ("a1", "Impervious heat storage a₁", "fraction of Q*", day and "impervious" in roles, "a1",
         "Grimmond & Oke (1999) give OHM coefficients per surface type; this a₁ multiplies impervious fraction × "
         "Q* and the model has no a₂ (hysteresis) term, so the two are not directly comparable (W6 adds the "
         "OHM source)", ("grimmond1999",)),
        ("a1n", "Impervious heat release a₁ₙ", "fraction of |LW_net|", (not day) and "impervious" in roles, "a1n",
         "night-time stored-heat release; not the OHM form (no published range transcribed)", ("grimmond1999",)),
        ("e1", "Evaporative-fraction slope on NDVI e₁", "logit per NDVI unit", "ndvi" in roles, None, no_range, ()),
        ("e2", "Evaporative-fraction slope on canopy e₂", "logit per canopy fraction", "canopy" in roles, None,
         no_range, ()),
        ("e3", "Evaporative-fraction slope on impervious e₃ (lowers EF)", "logit per impervious fraction",
         "impervious" in roles, None, no_range, ()),
        ("w", "Water-cell sink w", "W/m²", "water_distance" in roles, None, no_range, ()),
    ]
    rows = []
    ef0 = F.stat("e0", transform=lambda z: 1.0 / (1.0 + math.exp(-z)))
    if ef0:
        rows.append({"key": "ef0", "label": "Evaporative fraction at mean cover σ(e₀)", "unit": "fraction", **ef0,
                     "interval": _interval(ef0), "published": None, "status": "not_comparable", "status_note": no_range,
                     "prior": None, "refs": [], "group": "source"})
    for key, label, unit, on, prior_key, note, refs in specs:
        if not on:
            continue
        st = F.stat(key)
        if st is None:
            continue
        pr = priors.get(prior_key) if prior_key else None
        rows.append(_flag_bound(
            {"key": key, "label": label, "unit": unit, **st, "interval": _interval(st), "published": None,
             "status": "not_comparable", "status_note": note, "refs": list(refs), "group": "source",
             "prior": ({"mean": float(pr[0]), "sd": float(pr[1]),
                        "note": "information only; prior sensitivity is not yet tested (W3 power-scaling)"}
                       if pr and len(pr) == 2 else None)}, key, F))
    return rows


def _t2m(ph: dict) -> tuple[float, str, bool]:
    t = (ph.get("forcing_info") or {}).get("t2m_K")
    if _num(t) and float(t) > 150.0:
        return float(t), f"ERA5 2 m temperature of the forcing window ({float(t):.1f} K)", True
    return T_FALLBACK_K, f"assumed {T_FALLBACK_K:.2f} K: the forcing gives no 2 m temperature", False


def _range_row(key, label, unit, st, band, text, source, sym, reasons, note, *, refs=(), positive=False,
               derivation=None) -> dict:
    """A panel row compared with a published range.  ``reasons`` say why the range does not apply to this run
    (night, not the afternoon, an assumed forcing …): the row is then not comparable and keeps its value."""
    row = {"key": key, "label": label, "unit": unit, **(st or {"value": None, "se": None, "per_fold": None, "n": 0}),
           "interval": _interval(st), "published": None, "status": "not_comparable", "comparable": False,
           "not_comparable_reason": "; ".join(reasons) or None, "interval_overlaps": None,
           "interval_informative": None, "refs": list(refs), "group": "transport",
           "status_note": "; ".join(x for x in (note, *reasons) if x) or None}
    if derivation:
        row["derivation"] = derivation
    if reasons:
        return row
    row.update(published={"low": band[0], "high": band[1], "text": text, "source": source}, comparable=True)
    if st is None or st.get("value") is None:
        return row
    chk = _range_check(st, band, sym, positive=positive)
    row.update(status=chk["status"], interval_overlaps=chk["overlaps"], interval_informative=chk["informative"],
               check_note=chk["why"])
    if chk["why"]:
        row["status_note"] = "; ".join(x for x in (row["status_note"], chk["why"]) if x)
    return row


def physical_parameters(fold_params: list[dict] | None, config: dict | None, *, manifest: dict | None = None,
                        units: str | None = None) -> dict:
    """The W3 parameter panel: fitted physics in physical units, against published ranges where the plan
    gives one.  Rows: r_a (s/m), γ (K/km), L (m), the advection drift when kept, then the source parameters.
    Values are fold means over the folds with the physics term on, SE the delete-a-group jackknife."""
    ctx = _context(manifest, fold_params, config, units)
    return _panel(ctx)


def _panel(ctx: dict) -> dict:
    F, ph, window = ctx["folds"], ctx["physics"], ctx["window"]
    fK = to_kelvin_factor(ctx["units"])
    t_K, t_note, t_known = _t2m(ph)
    rcp = rho_cp(t_K)
    norm, norm_label = forcing_norm(ph, window)
    kind, basis = _basis(ctx["target_kind"])
    notes = []
    if not F.present:
        status = "absent"
    elif F.n_active == 0:
        status = "disabled"
        notes.append("Physics term disabled (fitted gain a ≤ 0, anti-physical) in every fold: no physical "
                     "parameter is reported.")
    else:
        status = "active"
        if F.n_active < F.n:
            notes.append(f"Physics term disabled (anti-physical gain) in {F.n - F.n_active} of {F.n} folds; "
                         "statistics use the remaining folds.")
    if fK is None and F.present:
        notes.append(f"Target units {ctx['units']!r} are not a temperature scale (°F, °C or K): r_a and γ cannot be "
                     "put in physical units.")
    forcing_ok, forcing_note = _forcing_known(ph, norm_label)
    afternoon, tod_reason = _afternoon(ph, window)
    rows: list[dict] = []
    ra = None
    if fK is not None:
        scale = fK * rcp / norm
        ra = F.stat("a", scale)
        # the published range applies only to the afternoon window and the campaign's own forcing
        ra_reasons = ([f"night window: {NIGHT_RA}"] if window != "day" else []) \
            + ([] if forcing_ok else [f"r_a needs the campaign's {norm_label}: this run has no forcing file and "
                                      f"assumes the generic default {norm_label} = {norm:g} W/m², and r_a scales as "
                                      f"1/{norm_label}"]) \
            + ([tod_reason] if window == "day" and tod_reason else [])
        rows.append(_range_row(
            "r_a", "Aerodynamic resistance r_a", "s/m", ra, RA_RANGE, RA_TEXT, RA_SOURCE, "r_a", ra_reasons,
            "q's scale depends on the albedo layer, so a biased albedo biases a and r_a (F5)",
            refs=["masson2000", "crawford2018"], positive=True,
            derivation=(f"r_a = a·{'(5/9)·' if fK != 1.0 else ''}ρc_p/{norm_label} with ρc_p = {rcp:.0f} J m⁻³ K⁻¹ "
                        f"(p = {P_STD:.0f} Pa, T: {t_note}) and {norm_label} = {norm:.1f} W/m² "
                        f"({forcing_note})")))
        has_elev = "elevation" in ctx["roles_used"]
        g = F.stat("gamma", fK * 1000.0) if has_elev else None
        g_reasons = (["no elevation role: γ is not fitted"] if not has_elev else
                     [f"night window: {NIGHT_GAMMA}"] if window != "day" else [tod_reason] if tod_reason else [])
        rows.append(_range_row(
            "gamma", "Elevation coefficient γ", "K/km", g, LAPSE_BAND, LAPSE_TEXT, LAPSE_SOURCE, "γ", g_reasons,
            "assumes the elevation layer is in metres; an indicative γ far outside the band means elevation is "
            "standing in for something else (F2)" if has_elev else None))
    L = F.stat("L_m")
    lb = F.first("L_bounds_m")
    L_prior = ((ctx["manifest"].get("influence") or {}).get("L_prior_m"))
    rows.append(_flag_grid(_flag_bound(
        {"key": "L", "label": "Mixing length L", "unit": "m",
         **(L or {"value": None, "se": None, "per_fold": None, "n": 0}), "interval": _interval(L),
         "published": None, "status": "not_comparable", "refs": [], "group": "transport",
         "status_note": ("L² = Kτ: one map identifies the product, not K and τ separately (F4); no published "
                         "range in the verified list"
                         + (f"; S1 correlogram prior L₀ = {L_prior:.0f} m" if _num(L_prior) else "")
                         + (f"; fit bounds {lb[0]:.0f}–{lb[1]:.0f} m" if lb and len(lb) == 2 else ""))},
        "L_m", F), F))
    v = F.stat("v_norm_m")
    if v and v["value"] > 1.0:
        adv = ctx["manifest"].get("physics_advection")
        if isinstance(adv, dict) and "kept" in adv:
            sel = (f"{'kept' if adv['kept'] else 'dropped'} by the out-of-fold test (held-out RMSE with − without = "
                   f"{_fmt(adv.get('delta_rmse_mean'), 3)} ± {_fmt(adv.get('delta_rmse_se'), 3)} {ctx['units'] or ''})"
                   .replace(" )", ")"))
        else:
            sel = ("kept without the out-of-fold test (physics.select_advection off, or no selection recorded): an "
                   "untested drift")
        rows.append({"key": "v", "label": "Advection drift |v| = |u|·τ", "unit": "m", **v, "interval": _interval(v),
                     "published": None, "status": "not_comparable", "refs": [], "group": "transport",
                     "status_note": f"v = u_c·τ: the drift times a canopy-layer wind; {sel}"})
    rows += _source_rows(ctx)
    for r in rows:
        if r["value"] is None and r["status"] != "not_comparable":
            r["status"] = "not_comparable"
    for r in rows:
        r["basis"] = basis if r.get("value") is not None else None
        r["value_text"] = _value_text(r)
    pinned = [r["label"] for r in rows if r.get("at_bound")]
    if pinned:
        notes.append(f"At a fit bound in at least one fold: {', '.join(pinned)}. A value stopped by a fit bound is "
                     "not an estimate, so these parameters are not identified by this run whatever their SE (plan F6); "
                     + ("the W3 profile likelihood (identifiability columns) tests what the data alone pin down."
                        if ctx.get("ident") else "the W3 profile likelihood will test identifiability properly."))
    ra_row = next((r for r in rows if r["key"] == "r_a"), None)
    P = {"rows": rows, "physics_status": status, "n_folds": F.n, "n_active": F.n_active, "source": F.source,
         "window": window, "units": ctx["units"], "to_kelvin": fK, "t2m_K": t_K, "t2m_note": t_note,
         "t2m_known": t_known, "rho_cp": rcp, "pressure_Pa": P_STD, "norm": norm, "norm_label": norm_label,
         "basis": basis, "target_kind": kind, "forcing_known": forcing_ok, "forcing_note": forcing_note,
         "afternoon": afternoon,
         "timescales": _timescales(ra_row, v, ph, None if forcing_ok else f"the generic default {norm_label}"),
         "identified_products": "One afternoon map identifies r_a = τ/h and L² = Kτ, not τ, h and K separately "
                                "(F4); the panel does not split them.",
         "identifiability": "not yet tested: W3 profile likelihood",
         "prior_sensitivity": "not yet tested: W3 power-scaling",
         "identifiability_study": False,
         "uncertainty": "mean ± delete-a-group jackknife SE over the fold models (SE = sd·√(K−1) = s·(K−1)/√K; "
                        "with folds disabled, s is the sample sd of the remaining folds and K stays the run's "
                        "number of folds)",
         "at_bound": [r["key"] for r in rows if r.get("at_bound")],
         "notes": notes}
    if ctx.get("ident"):
        _ident_panel(P, ctx["ident"])
    return P


def _ra_pinned(ra: dict) -> str | None:
    """Why r_a is not pinned down by this run (None when it is): no valid jackknife interval
    (:func:`_fold_precision`), an interval reaching r_a ≤ 0, or — where the published range applies — one wider
    than that range (the range test's own criterion)."""
    bad = _fold_precision(ra)
    if bad:
        return bad
    iv = _interval(ra)
    span = f"{_fmt(iv[0])} to {_fmt(iv[1])} s/m"
    if iv[0] <= 0:
        return f"its ±2 SE interval, {span}, reaches r_a ≤ 0"
    pub = ra.get("published")
    if pub and iv[1] - iv[0] > pub["high"] - pub["low"]:
        return f"its ±2 SE interval, {span}, is wider than the published range"
    return None


def _timescales(ra: dict | None, v: dict | None, ph: dict, forcing_assumed: str | None = None) -> dict | None:
    """Informational time scales from an *assumed* mixing depth h: τ = r_a·h and ωτ (W6's quasi-steady
    argument), with ranges from r_a's ±2 SE interval (clipped at 0); a kept advection drift bounds τ from below,
    since the canopy-layer wind is at most the 10 m one.  When r_a is not pinned down (:func:`_ra_pinned`) the
    block says the time scales are not informative."""
    if not ra or not _num(ra.get("value")) or ra["value"] <= 0:
        return None
    iv = _interval(ra)
    rows = []
    for h in H_ASSUMED_M:
        tau = ra["value"] * h
        rng = [max(iv[0], 0.0) * h, iv[1] * h] if iv else None
        rows.append({"h_m": h, "tau_s": tau, "tau_min": tau / 60.0, "omega_tau": OMEGA_DAY * tau,
                     "phase_lag_deg": math.degrees(math.atan(OMEGA_DAY * tau)),
                     "tau_min_range": [x / 60.0 for x in rng] if rng else None,
                     "omega_tau_range": [OMEGA_DAY * x for x in rng] if rng else None})
    why = _ra_pinned(ra)
    out = {"assumption": f"h is assumed, not fitted: canopy-layer depths of {H_ASSUMED_M[0]:g} and "
                         f"{H_ASSUMED_M[1]:g} m (W6). One map identifies r_a = τ/h only."
                         + (f" r_a itself rests on {forcing_assumed}, assumed (no forcing file)."
                            if forcing_assumed else ""),
           "informative": why is None,
           "caveat": (f"r_a is not pinned down by this run ({why}), so these time scales are not informative and "
                      "say nothing about whether the steady operator holds" if why else None),
           "omega_per_s": OMEGA_DAY, "rows": rows, "advection_bound": None, "advection_note": None}
    wind = ph.get("wind")
    speed = math.hypot(float(wind[0]), float(wind[1])) if isinstance(wind, (list, tuple)) and len(wind) >= 2 \
        and _num(wind[0]) and _num(wind[1]) else None
    if v and v["value"] > 1.0 and speed and speed > 0:
        tmin = v["value"] / speed
        out["advection_bound"] = {"tau_min_s": tmin, "v_m": v["value"], "wind_m_s": speed,
                                  "text": f"τ ≥ |v|/|u| = {v['value']:.0f} m / {speed:.1f} m/s = {tmin / 60:.0f} min "
                                          "(lower bound: the canopy-layer wind is at most the 10 m wind)"}
    elif v is not None and v["value"] <= 1.0:
        out["advection_note"] = "Advection off or dropped (v = 0): no bound on τ from the drift"
    return out


# --------------------------------------------------------------------------- W3 identifiability (the study)
def _sig(v, unit: str = "") -> str:
    """Three significant digits (none after the point from 100 up), 0 for |v| < 5·10⁻⁴ (a value on a softplus
    floor), a true minus sign."""
    if not _num(v):
        return "—"
    x = float(v)
    s = "0" if abs(x) < 5e-4 else (f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}")
    return s.replace("-", "−") + (f" {unit}" if unit else "")


def _lc(label: str) -> str:
    return label[:1].lower() + label[1:] if label else label


def _fold_mean(F: "_Folds", key: str) -> float | None:
    """This run's fold mean of a report parameter over the folds with the physics term on (as the study records
    it, ``identifiability._folds_mean``); from the manifest summary only when every fold is on."""
    pk = {"L": "L_m", "v": "v_norm_m"}.get(key, key)
    if F.params:
        vals = [float(p[pk]) for p in F.active if _num(p.get(pk))]
        return sum(vals) / len(vals) if vals else None
    s = (F.summary or {}).get(pk)
    if F.summary and F.n_active == F.n and isinstance(s, dict) and _num(s.get("mean")):
        return float(s["mean"])
    return None


#: The sandwich behind the r_a / γ SEs of the full-data fit, by the study's ``score_covariance``
_COV_SE = {"blocks": ("block sandwich", "block-sandwich", "clusters = the CV blocks; Chandler & Bate 2007"),
           "neighbours": ("adjacent-block sandwich", "adjacent-block sandwich",
                          "the CV blocks' scores with adjacent blocks' cross-products added; Chandler & Bate 2007, "
                          "Conley 1999")}


def covariance_text(block: dict, short: bool = False) -> str:
    """How the identifiability study adjusts the likelihood for spatial dependence, in words (the register, the
    writeups and the results page say the same): Chandler & Bate (2007)'s block sandwich with the run's CV blocks as
    independent clusters by default, the Conley (1999)-type adjacent-block variant as a sensitivity, and the class
    from the wider of the two intervals.  ``block``: the register's ``identifiability`` block (``n_clusters``,
    ``block_m``)."""
    bm = f" of {_sig(block['block_m'])} m" if _num(block.get("block_m")) else ""
    n = block.get("n_clusters")
    if short:
        return (f"adjusted for spatial dependence with the run's {n} CV blocks{bm} as independent clusters (Chandler & "
                "Bate 2007's block sandwich) and, as a sensitivity, with adjacent blocks' score cross-products added "
                "(Conley 1999); the class rests on the wider of the two intervals")
    return (f"its curvature adjusted for spatial dependence with the run's {n} cross-validation blocks{bm} as "
            "independent clusters (the block sandwich of Chandler & Bate 2007, the default) and, as a sensitivity, "
            "with the score cross-products of adjacent blocks added (a block-level spatial HAC in the manner of "
            "Conley 1999, which allows for dependence across block edges); the identifiability class rests on the "
            "wider of the two 95% intervals on each side")


def _ident_mismatch(ident: dict, ctx: dict) -> str | None:
    """Why an identifiability report does not belong to this run (None when it does): not a report, no active
    physics term here, or a different number of cells, CV design, input data, target units, window or physics fit
    (the fold means of the profiled parameters it recorded against this run's)."""
    if not isinstance(ident.get("parameters"), list) or ident.get("schema") != 1:
        return "not an identifiability report of a known schema"
    if ctx["physics_off"]:
        return f"this run fits no physics model ({ctx['physics_off']})"
    F = ctx["folds"]
    if not F.present or F.n_active == 0:
        return "this run has no active physics term"
    m = ctx["manifest"] or {}
    run = ident.get("run") or {}
    n_i = run.get("n_points_run")
    if n_i is None and not ident.get("coarse_m"):
        n_i = ident.get("n")
    n_m = m.get("n_points")
    if _num(n_i) and _num(n_m) and int(n_i) != int(n_m):
        return f"it was made for a run of {int(n_i):,} cells, and this run has {int(n_m):,}"
    cv_i, cv_m = run.get("cv") or {}, m.get("cv") or {}
    for k in ("n_folds", "n_blocks", "block_m"):
        a, b = cv_i.get(k), cv_m.get(k)
        if _num(a) and _num(b) and not math.isclose(float(a), float(b), rel_tol=1e-9):
            return f"its cross-validation design differs ({k} {a:g} against this run's {b:g})"
    sha_i, sha_m = run.get("input_sha256"), (m.get("provenance") or {}).get("input_sha256")
    if sha_i and sha_m and sha_i != sha_m:
        return "it was made from different input data (the input SHA-256 differs)"
    for what, a, b in (("target units", ident.get("units"), ctx["units"]), ("window", ident.get("window"),
                                                                          ctx["window"])):
        if a and b and str(a).lower() != str(b).lower():
            return f"its {what} ({a}) differ from this run's ({b})"
    for key, v in (run.get("fold_means") or {}).items():
        cur = _fold_mean(F, key)
        if _num(v) and cur is not None and not math.isclose(float(v), cur, rel_tol=1e-6, abs_tol=1e-9):
            return (f"it was made for a different physics fit (fold mean of {key} {_sig(v)} in the study, {_sig(cur)} "
                    "in this run)")
    return None


def _ident_param(p: dict, laplace_ok, run_value: float | None = None) -> dict:
    """A report parameter row in the register's compact form, with its plain-words sentence.  ``run_value``: the
    run's own value of the parameter, the mean over its folds with the physics term on (``physics.json``, as the
    parameter panel shows it; default the fold mean the study recorded).  ``estimate_run`` is the study's full-data
    refit with the run's objective (penalty on), which is what the prior share and the pull describe: it is labelled
    "full-data refit" wherever it is shown, never "run value"."""
    cls = p.get("identifiability")
    ci = p.get("ci95") if isinstance(p.get("ci95"), (list, tuple)) and len(p["ci95"]) == 2 else None
    ab = p.get("at_bound") if isinstance(p.get("at_bound"), dict) else None
    zero = p.get("key") in _ZERO_LIMIT and cls in ("identified", "one_sided") and (
        (ci is not None and _num(ci[0]) and float(ci[0]) <= 0.0) or "lower" in (p.get("sides_at_limit") or [])
        or (ab is not None and ab.get("side") == "lower" and _num(ab.get("bound")) and float(ab["bound"]) <= 0.0))
    post, ps, cf = p.get("posterior") or {}, p.get("psens") or {}, p.get("conflict") or {}

    def unreliable(d: dict):
        # the study flags a Laplace posterior that its own adequacy check calls poor (``reliable: false``); a report
        # without the flag: a Laplace-method summary of a study whose Laplace approximation is poor
        if d.get("reliable") is False:
            return d.get("reason") or "the Laplace approximation it rests on is poor for this run"
        if "reliable" not in d and d.get("method") == "laplace" and laplace_ok is False:
            return "the Laplace approximation it rests on is poor for this run"
        return None

    post_bad, ps_bad = (unreliable(post) if post else None), (unreliable(ps) if ps else None)
    out = {"key": p.get("key"), "label": p.get("label") or p.get("key"), "unit": p.get("unit") or "",
           "class": cls, "class_label": IDENT_LABELS.get(cls, cls), "class_short": IDENT_SHORT.get(cls, cls),
           "identified": cls == "identified",
           "in_model": p.get("estimate_data") is not None,
           "run_value": run_value if _num(run_value) else (p.get("estimate_run_folds")
                                                           if _num(p.get("estimate_run_folds")) else None),
           "estimate_run": p.get("estimate_run"), "estimate_data": p.get("estimate_data"),
           "ci95": list(ci) if ci else None, "ci95_neighbours": p.get("ci95_neighbours"),
           "ci95_conservative": p.get("ci95_conservative"), "class_by_covariance": p.get("class_by_covariance"),
           "threshold": p.get("threshold"),
           "ci95_naive": p.get("ci95_naive"), "ci95_naive_label": p.get("ci95_naive_label"),
           "open_sides": list(p.get("open_sides") or []), "sides_at_limit": list(p.get("sides_at_limit") or []),
           "summary": p.get("summary"),
           "at_bound": ab, "held": p.get("held"), "reaches_zero": bool(zero),
           "se_robust": p.get("se_robust"), "adjust_factor": p.get("adjust_factor"), "prior": p.get("prior"),
           "prior_share_run": p.get("prior_share_run"), "pull": p.get("pull"),
           # unreliable posterior summaries and power-scaling are not presented (the study's JSON keeps them, flagged)
           "posterior": ({k: post.get(k) for k in ("mean", "sd", "q025", "q975", "method")}
                         if post.get("mean") is not None and not post_bad else None),
           "psens": ({k: ps.get(k) for k in ("prior", "likelihood", "diagnosis", "method")} if ps and not ps_bad
                     else None),
           "posterior_unreliable": post_bad or ps_bad,
           "conflict": ({k: cf.get(k) for k in ("p", "z", "flag", "reason")} if cf else None)}
    if out["psens"] is not None:
        out["psens"]["laplace_poor"] = False
        out["psens"].update(_psens_reading(out["psens"]))
    out["text"] = _ident_sentence(out)
    return out


def _psens_reading(ps: dict) -> dict:
    """Kallioinen et al. (2024)'s reading of one parameter's power-scaling sensitivities (:data:`PSENS_READING`):
    ``prior_informative`` (prior ≥ 0.05), ``likelihood_weak`` (likelihood < 0.05) and ``reading``, the flag to
    report: the study's diagnosis (``prior_data_conflict``, ``weak_likelihood`` = informative prior with a weak
    likelihood), else ``likelihood_only_weak`` when the likelihood alone is below 0.05, else None."""
    pr, lk = ps.get("prior"), ps.get("likelihood")
    pi = bool(_num(pr) and float(pr) >= PSENS_THRESHOLD)
    lw = bool(_num(lk) and float(lk) < PSENS_THRESHOLD)
    reading = ps.get("diagnosis") if ps.get("diagnosis") in ("prior_data_conflict", "weak_likelihood") else (
        "likelihood_only_weak" if lw else None)
    return {"prior_informative": pi if _num(pr) else None, "likelihood_weak": lw if _num(lk) else None,
            "reading": reading}


def _ident_sentence(p: dict) -> str:
    """One parameter's identifiability in plain words: what the data alone say, what the penalty sets, and any
    sensitivity or conflict flag."""
    sym = _SYM.get(p["key"], p["key"])
    u = f" {p['unit']}" if p["unit"] in ("m", "W/m²") else ""
    ci = p.get("ci95") or [None, None]
    span = f"{_sig(ci[0]) if ci[0] is not None else 'open'} to {_sig(ci[1]) if ci[1] is not None else 'open'}{u}"
    nb = p.get("ci95_neighbours")
    if isinstance(nb, (list, tuple)) and len(nb) == 2 and list(nb) != list(ci):
        # the class rests on the wider of the two adjusted intervals: quote both (block sandwich first)
        span += (f" with the block sandwich, {_sig(nb[0]) if nb[0] is not None else 'open'} to "
                 f"{_sig(nb[1]) if nb[1] is not None else 'open'}{u} with adjacent blocks")
    cls = p["class"]
    if cls == "not_fitted":
        return f"{p['label']}: not fitted in this run."
    if cls == "structurally_non_identifiable":
        core = "not identified by the data alone (flat profile likelihood)"
    elif cls == "one_sided":
        what = p.get("summary") or f"95% profile interval {span}"
        core = (f"the data alone do not exclude {sym} = 0 ({what})" if p["reaches_zero"] else
                f"bounded by the data alone on one side only ({what})")
    elif cls == "not_assessable":
        nv = p.get("ci95_naive") or [None, None]
        core = ("not assessable from the data alone (its data-only estimate sits on a fit bound, where no adjustment "
                "for spatial dependence is computable; the unadjusted one-sided bound, "
                f"{_sig(nv[0]) if nv[0] is not None else 'open'} to {_sig(nv[1]) if nv[1] is not None else 'open'}{u}, "
                "is too narrow)")
    elif cls == "practically_non_identifiable":
        if p["open_sides"]:
            core = (f"not identified by the data alone (its 95% profile interval, {span}, stays open on the "
                    f"{' and '.join(p['open_sides'])} side)")
        elif p.get("held") == "flat":
            core = "not identified by the data alone (no local curvature, and a profile that is not smooth)"
        else:
            core = f"not identified by the data alone (its 95% profile interval, {span}, spans its whole physical range)"
    elif p["reaches_zero"]:
        at0 = (p.get("at_bound") or {}).get("side") == "lower"
        core = (f"the data alone do not exclude {sym} = 0 ("
                + ("the data-only fit sits at 0" if at0 else f"data-only estimate {_sig(p['estimate_data'])}{u}")
                + f"; 95% profile interval {span})")
    elif cls == "identified":
        core = f"identified by the data alone (data-only estimate {_sig(p['estimate_data'])}{u}; 95% profile interval {span})"
    else:
        core = f"identifiability {cls}"
    parts = [core]
    sh = p.get("prior_share_run")
    if _num(sh) and p.get("estimate_run") is not None:
        # the prior share is the penalty's share of the run objective's curvature at the full-data refit
        rv = p.get("run_value")
        run = f"; the run's own value, its fold mean, is {_sig(rv)}{u}" if _num(rv) else ""
        parts.append(f"its full-data refit value {_sig(p['estimate_run'])}{u} is set mostly by the penalty (prior "
                     f"share {float(sh):.2f}{run})" if float(sh) > 0.5 else f"prior share {float(sh):.2f}")
    ps = p.get("psens") or {}
    if ps.get("reading") in _PSENS_TEXT:
        parts.append(_PSENS_TEXT[ps["reading"]])
    elif p.get("posterior_unreliable"):
        parts.append("its posterior summaries and power-scaling are not reliable, so they are not reported (they rest "
                     "on a Laplace approximation that is poor for this run)")
    cf = p.get("conflict") or {}
    if cf.get("flag") and _num(cf.get("p")):
        parts.append(f"the prior–data conflict check finds a conflict (p = {float(cf['p']):.2g})")
    return f"{p['label']}: " + "; ".join(parts) + "."


def _ident_info(ident, ctx: dict, source: str | None) -> tuple[dict | None, dict]:
    """(the study in the register's form or None, the register's ``identifiability`` block)."""
    block = {"status": "absent", "source": None, "note": None}
    if not isinstance(ident, dict) or not ident:
        return None, block
    block["source"] = source
    why = _ident_mismatch(ident, ctx)
    if why:
        block.update(status="ignored", note=f"identifiability study ignored: {why}; re-run `sparc core identifiability` "
                                            "for this run")
        return None, block
    cl = ident.get("cluster") or {}
    if ident.get("status") == "disabled":
        # the study stopped before any profile (its data-only fit has a ≤ 0): nothing it could say about any
        # parameter, so nothing is reported as profiled, identified or gated (PHYSICS_PLAN W3 review X16)
        block.update(status="disabled", study_status="disabled", n=ident.get("n"),
                     n_clusters=ident.get("n_clusters"), block_m=cl.get("block_m"), coarse_m=ident.get("coarse_m"),
                     warnings=[str(w) for w in ident.get("warnings") or []], parameters={}, linear={},
                     not_identified=[], refs=[],
                     note=DISABLED_NOTE)
        return None, block
    post = ident.get("posterior") or {}
    laplace_ok = post.get("laplace_adequate") if post else None
    params = {}
    for p in ident.get("parameters") or []:
        if isinstance(p, dict) and p.get("key"):
            params[p["key"]] = _ident_param(p, laplace_ok, run_value=_fold_mean(ctx["folds"], p["key"]))
    linear = {}
    for r in ident.get("linear") or []:
        if isinstance(r, dict) and r.get("key"):
            linear[r["key"]] = {k: r.get(k) for k in ("label", "unit", "estimate_run", "se_robust_run", "se_naive_run",
                                                       "estimate_data", "se_robust_data", "se_naive_data",
                                                       "adjust_factor")}
    info = {"params": params, "linear": linear, "n": ident.get("n"), "n_clusters": ident.get("n_clusters"),
            "block_m": cl.get("block_m"), "coarse_m": ident.get("coarse_m"),
            "score_covariance": ((ident.get("covariance") or {}).get("used")
                                 or (ident.get("settings") or {}).get("score_covariance") or "blocks"),
            "laplace_adequate": laplace_ok, "study_status": ident.get("status"),
            "warnings": [str(w) for w in ident.get("warnings") or []]}
    block.update(status="used", study_status=ident.get("status"), n=info["n"], n_clusters=info["n_clusters"],
                 block_m=info["block_m"], coarse_m=info["coarse_m"], score_covariance=info["score_covariance"],
                 laplace_adequate=laplace_ok, warnings=info["warnings"], parameters=params, linear=linear,
                 not_identified=[k for k, p in params.items() if p["in_model"] and p["class"] in NOT_IDENTIFIED],
                 refs=list(_IDENT_REFS))
    return info, block


def _sandwich(row: dict, lin: dict | None, scale: float, sym: str, unit: str, nd: int = 1,
              cov: str | None = "blocks") -> None:
    """r_a / γ: the sandwich SE of the full-data reference fit (the run's objective; the block sandwich unless the
    study used ``score_covariance: neighbours``, ``cov``) beside the fold jackknife; a note when the two SEs differ
    by more than :data:`SANDWICH_FACTOR`.  The verdict is not changed."""
    name, adj, how = _COV_SE.get(cov or "blocks", _COV_SE["blocks"])
    if not lin or row.get("value") is None or not (_num(lin.get("estimate_run")) and _num(lin.get("se_robust_run"))):
        return
    s = abs(scale)
    sw = {"value": float(lin["estimate_run"]) * scale, "se": float(lin["se_robust_run"]) * s,
          "se_naive": float(lin["se_naive_run"]) * s if _num(lin.get("se_naive_run")) else None,
          "data_only": ({"value": float(lin["estimate_data"]) * scale, "se": float(lin["se_robust_data"]) * s}
                        if _num(lin.get("estimate_data")) and _num(lin.get("se_robust_data")) else None),
          "jackknife_se": row.get("se"), "ratio": None, "note": None, "covariance": cov or "blocks", "kind": name}
    sw["text"] = (f"full-data reference fit: {sym} = {_fmt(sw['value'], nd)} ± {_fmt(sw['se'], nd)} {unit} ({name}, "
                  f"{how}"
                  + (f"; naive ± {_fmt(sw['se_naive'], nd + 1)}" if sw["se_naive"] is not None else "") + ")")
    jk = row.get("se")
    if _num(jk) and float(jk) > 0 and sw["se"] > 0:
        sw["ratio"] = sw["se"] / float(jk)
        if not (1.0 / SANDWICH_FACTOR <= sw["ratio"] <= SANDWICH_FACTOR):
            sw["note"] = (f"the {adj} SE of the full-data fit ({_fmt(sw['se'], nd)} {unit}) and the fold "
                          f"jackknife SE ({_fmt(jk, nd)} {unit}) differ by more than a factor of {SANDWICH_FACTOR:g}: "
                          "the verdict uses the jackknife, whose ±2 SE interval is then itself uncertain")
    row["sandwich"] = sw
    if sw["note"]:
        row["status_note"] = "; ".join(x for x in (row.get("status_note"), sw["note"]) if x)


def _ident_panel(P: dict, info: dict) -> None:
    """The study's columns on the parameter panel (module docstring)."""
    P["identifiability_study"] = True
    P["identifiability"] = ("W3 profile likelihood on all labelled cells (Raue et al. 2009), "
                            + covariance_text(info, short=True))
    P["prior_sensitivity"] = ("W3: the prior's share of the run objective's curvature and its pull; power-scaling of "
                              "the calibrated posterior (Kallioinen et al. 2024; Ribatet, Cooley & Davison 2012); "
                              "prior–data conflict (Evans & Moshonov 2006)")
    back = {v: k for k, v in _PANEL_KEY.items()}
    for r in P["rows"]:
        p = info["params"].get(back.get(r["key"], r["key"]))
        if not p or not p["in_model"]:
            continue
        r["identifiability"] = p
        pr = r.get("prior")
        if pr and "not yet tested" in str(pr.get("note") or ""):
            r["prior"] = {**pr, "note": "information; its influence is the W3 prior share and pull (at the study's "
                                        "full-data refit of the run's objective) and power-scaling"}
        if r.get("status_note"):
            r["status_note"] = r["status_note"].replace(
                "whether this map identifies s is the profile-likelihood test (W3)",
                "whether the data alone identify s is the W3 profile likelihood (identifiability column)")
    fK = P.get("to_kelvin")
    if fK is None:
        return
    for r in P["rows"]:
        if r["key"] == "r_a":
            _sandwich(r, info["linear"].get("a"), fK * P["rho_cp"] / P["norm"], "r_a", "s/m",
                      cov=info.get("score_covariance"))
        elif r["key"] == "gamma":
            _sandwich(r, info["linear"].get("gamma"), fK * 1000.0, "γ", "K/km", cov=info.get("score_covariance"))


def _ident_rows(rows: list[dict], info: dict, by_key: dict) -> None:
    """The verdict rule (a key parameter not identified by the data alone blocks Supported) and the identifiability
    context of each mechanism's first signature row."""
    seen = set()
    for r in rows:
        keys = KEY_PARAMS.get(r["mechanism"], ())
        params = [info["params"][k] for k in keys if k in info["params"] and info["params"][k]["in_model"]]
        if params:
            r["identifiability"] = [{k: p.get(k) for k in ("key", "class", "ci95", "prior_share_run", "reaches_zero")}
                                    for p in params]
        bad = [p for p in params if p["class"] in NOT_IDENTIFIED]
        if r["status"] == "supported" and bad:
            names = ", ".join(_lc(p["label"]) for p in bad)
            r.update(status="not_testable", label=VERDICT_LABELS["not_testable"], basis=None, identifiability_gate=True,
                     reason=(f"{names} {'is' if len(bad) == 1 else 'are'} not identified by the data alone (W3 "
                             "profile likelihood adjusted for spatial dependence; the class rests on the wider of the "
                             "block and adjacent-block intervals): a mechanism whose key parameter the data cannot pin "
                             "down cannot be Supported, whatever its point estimate (its own test alone would give "
                             "Supported)"))
        if r["mechanism"] in seen:
            continue
        seen.add(r["mechanism"])
        extra = []
        if params:
            extra.append("Identifiability from the data alone (W3): " + " ".join(p["text"] for p in params))
        prow = by_key.get({"M6": "r_a", "M7": "gamma"}.get(r["mechanism"], ""))
        if prow and prow.get("sandwich"):
            sw = prow["sandwich"]
            extra.append(f"Uncertainty check (W3): {sw['text']}, beside the fold jackknife ± "
                         f"{_fmt(prow.get('se'), 1)} {prow['unit']}; the verdict uses the jackknife.")
            if sw.get("note"):
                r["note"] = "; ".join(x for x in (r.get("note"), sw["note"]) if x)
        if extra:
            r["context"] = " ".join(x for x in (r.get("context"), *extra) if x)


def run_identifiability(run_dir, manifest: dict | None = None) -> tuple[dict | None, str | None]:
    """A run's identifiability study and where it came from: the manifest's ``identifiability`` section, else
    ``identifiability.json`` (None, None when it has neither).  Whether it belongs to the run is checked by
    :func:`evaluate`."""
    m = manifest
    if m is None:
        try:
            m = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = {}
    sec = (m or {}).get("identifiability")
    if isinstance(sec, dict) and sec:
        return sec, "manifest"
    try:
        blob = json.loads((Path(run_dir) / "identifiability.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    return (blob, "identifiability.json") if isinstance(blob, dict) and blob else (None, None)


# --------------------------------------------------------------------------- the register for a run
def _row(mech: dict, sig: dict, status: str, **kw) -> dict:
    return {"id": f"{mech['id']}{sig['sig']}", "mechanism": mech["id"], "sig": sig["sig"], "name": mech["name"],
            "then": sig["then"], "test": sig["test"], "workstream": sig["workstream"], "status": status,
            "label": VERDICT_LABELS[status], "statistic": None, "statistic_text": None, "interval": None,
            "reference": None, "basis": None, "reason": None, "note": sig.get("note"), "context": None,
            "refs": list(mech["refs"]), **kw}


def _default_row(mech: dict, sig: dict) -> dict:
    kind = sig["kind"]
    if kind == "multi_window":
        return _row(mech, sig, "not_testable", reason=sig.get("reason") or _ONE_WINDOW)
    if kind == "missing_term":
        return _row(mech, sig, "not_yet_tested",
                    reason=f"the term is not in the model; its test belongs to {sig['workstream']}")
    return _row(mech, sig, "not_yet_tested", reason=sig.get("reason") or f"the test belongs to {sig['workstream']}, "
                                                                          "which is not built yet")


def _physics_block(ctx: dict) -> str | None:
    """Why no physics test can run (None when the physics term is on in at least one fold)."""
    F = ctx["folds"]
    if ctx["physics_off"]:
        return f"no physics model in this run ({ctx['physics_off']} in the config)"
    if not F.present:
        return "no physics model in this run (no physics.json and no physics summary)"
    if F.n_active == 0:
        return ("physics term disabled: the fitted gain a ≤ 0 (anti-physical) in every fold, so the model "
                "fell back to its linear terms")
    if to_kelvin_factor(ctx["units"]) is None:
        return f"target units {ctx['units']!r} are not a temperature scale (°F, °C or K)"
    return None


def evaluate(manifest: dict | None, fold_params: list[dict] | None = None, config: dict | None = None, *,
             units: str | None = None, identifiability=None, identifiability_source: str | None = None) -> dict:
    """The mechanism register of a run: one row per mechanism signature with status, statistic, interval,
    basis and note; the W3 parameter panel; gates for scenario text.

    ``manifest``: the run's manifest (``physics_advection``, ``influence``, ``cv``, ``identifiability`` and,
    without ``fold_params``, the ``physics`` mean/sd summary are read from it); ``fold_params``: the per-fold
    ``PhysicsModel.params`` (``physics.json``); ``config``: the raw config dict (default: the manifest's);
    ``units``: the target units (default: ``data.target_units``); ``identifiability``: the run's identifiability
    report (default: the manifest's section; ``False``: none), used only when it belongs to this run (module
    docstring, "Identifiability from the data alone")."""
    ctx = _context(manifest, fold_params, config, units)
    if identifiability is None:
        identifiability = (ctx["manifest"] or {}).get("identifiability")
        identifiability_source = identifiability_source or "manifest"
    elif identifiability is False:
        identifiability = None
    ident, ident_block = _ident_info(identifiability, ctx, identifiability_source or "argument")
    ctx["ident"] = ident
    panel = _panel(ctx)
    if ident_block.get("status") == "disabled":
        panel["identifiability"] = f"W3 profile likelihood not done; {ident_block['note']}"
        panel["prior_sensitivity"] = "W3 prior sensitivity not done: the identifiability study profiled nothing"
    kind, basis = _basis(ctx["target_kind"])
    by_key = {r["key"]: r for r in panel["rows"]}
    block = _physics_block(ctx)
    F = ctx["folds"]
    partial = (f"physics term disabled in {F.n - F.n_active} of {F.n} folds; the statistic uses the other "
               f"{F.n_active}") if F.present and 0 < F.n_active < F.n else None
    adv = (ctx["manifest"] or {}).get("physics_advection")
    u = ctx["units"] or ""
    rows: list[dict] = []
    for mech in REGISTER:
        for sig in mech["signatures"]:
            r = _default_row(mech, sig)
            key = r["id"]
            if key == "M6a":
                r = _eval_ra(mech, sig, by_key.get("r_a"), block, panel)
                r["test"] = f"Parameter panel: r_a = a·ρc_p/{panel['norm_label']}, mean ± 2 SE against {RA_TEXT} " \
                            "(afternoon window)"
            elif key == "M7a":
                r = _eval_gamma(mech, sig, by_key.get("gamma"), block, ctx)
            if r["status"] in ("supported", "contradicted"):
                r["basis"] = basis
                if partial:
                    r["note"] = (r["note"] + "; " if r["note"] else "") + partial
            first = sig is mech["signatures"][0]          # context once per mechanism
            if mech["id"] == "M5" and adv and first:
                r["context"] = (f"Out-of-fold selection: advection {'kept' if adv.get('kept') else 'dropped'} "
                                f"(held-out RMSE with − without = {_fmt(adv.get('delta_rmse_mean'), 3)} ± "
                                f"{_fmt(adv.get('delta_rmse_se'), 3)} {u}). An RMSE test is not the signature test: "
                                "it cannot tell 'no advection' from 'advection the target cannot show' (F8).")
            if mech["id"] == "M4" and first and by_key.get("L") and by_key["L"].get("value") is not None:
                Lr = by_key["L"]
                r["context"] = (f"Fitted L = {_fmt(Lr['value'], 0)}"
                                + (f" ± {_fmt(Lr['se'], 0)}" if Lr.get("se") is not None else "") + " m"
                                + (f", {Lr['at_bound']['text']}." if Lr.get("at_bound") else "."))
            if mech.get("shortwave") and not ctx["shortwave"]:
                r["note"] = (r["note"] + " " if r["note"] else "") + \
                    "This run's source has no sunlight term (Q* = LW_net), so M1 does not act in its physics."
            missing = [x for x in mech["roles"] if x != _ALL and x not in ctx["roles_used"]]
            if sig["kind"] != "evaluate" and mech["roles"] and _ALL not in mech["roles"] \
                    and len(missing) == len(mech["roles"]):
                r["note"] = (r["note"] + " " if r["note"] else "") + \
                    f"No {', '.join(ROLE_LABELS.get(x, x) for x in missing)} layer in this run's physics."
            rows.append(r)
    if ident:
        _ident_rows(rows, ident, by_key)
    mechs = []
    for mech in REGISTER:
        mr = [r for r in rows if r["mechanism"] == mech["id"]]
        st = next(s for s in _PRECEDENCE if any(r["status"] == s for r in mr))
        chain = mech["chain"].replace(_NORM_TEXT, panel["norm_label"])      # the run's normaliser of q
        entry = {"id": mech["id"], "name": mech["name"], "terms": mech["terms"], "chain": chain,
                 "in_model": mech["in_model"], "roles": list(mech["roles"]), "status": st,
                 "shortwave": bool(mech.get("shortwave")),
                 "acts_in_run": bool(mech["in_model"] and (ctx["shortwave"] or not mech.get("shortwave"))),
                 "label": VERDICT_LABELS[st], "signatures": [r["id"] for r in mr], "refs": list(mech["refs"])}
        if ident:
            keys = [k for k in KEY_PARAMS.get(mech["id"], ()) if k in ident["params"] and ident["params"][k]["in_model"]]
            entry["identifiability"] = {"keys": keys, "not_identified": [k for k in keys if ident["params"][k]["class"]
                                                                         in NOT_IDENTIFIED]}
        mechs.append(entry)
    used = {k for m in REGISTER for k in m["refs"]} | {"oreskes1994", "mooij2013", "shandas2019"}
    if ident:
        used |= set(_IDENT_REFS)
    summary = {s: [r["id"] for r in rows if r["status"] == s] for s in VERDICTS}
    return {
        "schema": SCHEMA, "verdicts": dict(VERDICT_LABELS), "window": ctx["window"], "units": ctx["units"],
        "target_kind": kind, "basis": basis, "physics_status": panel["physics_status"],
        "rows": rows, "mechanisms": mechs, "summary": summary, "panel": panel,
        "gates": _gates(rows, mechs, ctx, panel["physics_status"]),
        "identifiability": ident_block,
        "references": {k: REFERENCES[k] for k in sorted(used)},
        "inputs": {"source": F.source, "n_folds": F.n, "n_active": F.n_active, "fit_warnings": F.warnings,
                   "advection": adv},
    }


def _eval_range(mech, sig, row, *, sym, name, unit, band, text, source, positive=False, notes=()) -> dict:
    """M6a / M7a: a panel row against its published range.  Not testable when the range does not apply to this
    run (the row's ``not_comparable_reason``) or the interval cannot test it (:func:`_range_check`)."""
    st = {k: row.get(k) for k in ("value", "se", "n", "n_groups")}
    common = dict(statistic={"name": name, "value": row["value"], "se": row["se"], "unit": unit, "n": row.get("n"),
                             "n_folds": row.get("n_groups")},
                  statistic_text=_stat_text(sym, st, unit), interval=_interval(st))
    if not row.get("comparable"):
        return _row(mech, sig, "not_testable", reason=row.get("not_comparable_reason"), **common)
    chk = _range_check(st, band, sym, positive=positive)
    return _row(mech, sig, chk["verdict"], reason=chk["why"], note="; ".join(notes) or None,
                reference={"low": band[0], "high": band[1], "text": text, "source": source}, **common)


def _eval_ra(mech, sig, row, block, panel) -> dict:
    if block:
        return _row(mech, sig, "not_testable", reason=block)
    if not row or row.get("value") is None:
        return _row(mech, sig, "not_testable", reason="no fitted gain a to convert")
    return _eval_range(mech, sig, row, sym="r_a", name="r_a", unit="s/m", band=RA_RANGE, text=RA_TEXT,
                       source=RA_SOURCE, positive=True,
                       notes=() if panel["t2m_known"] else (f"ρc_p uses T = {panel['t2m_note']}",))


def _eval_gamma(mech, sig, row, block, ctx) -> dict:
    if block:
        return _row(mech, sig, "not_testable", reason=block)
    if "elevation" not in ctx["roles_used"]:
        return _row(mech, sig, "not_testable", reason="no elevation layer in the physics roles: γ is not fitted")
    if not row or row.get("value") is None:
        return _row(mech, sig, "not_testable", reason="no fitted elevation coefficient")
    r = _eval_range(mech, sig, row, sym="γ", name="gamma", unit="K/km", band=LAPSE_BAND, text=LAPSE_TEXT,
                    source=LAPSE_SOURCE, notes=("assumes the elevation layer is in metres",))
    if r["status"] == "contradicted":
        r["note"] += ("; elevation is standing in for something else, possibly a missing source such as nearby "
                      "water carried by the wind (F2, tested by W2)")
    return r


def _gates(rows: list[dict], mechs: list[dict], ctx: dict, physics_status: str) -> dict:
    """Plain statements from contradicted rows, and per actionable lever which mechanisms explain it.

    A lever's own mechanisms are those naming its role (M1–M3, M8); the transport mechanisms (M4–M6) shape every
    lever's effect but test the overall gain and footprint, not the lever's pathway, so they never make a lever's
    explanation "supported" — they can only withdraw it, when contradicted.  Without an active physics term
    (absent, or disabled as anti-physical in every fold) no lever acts through the physics.  Only mechanisms that
    act in this run's source count: without sunlight (a night window) M1 is not in the source, so canopy acts
    through evaporation only and a lever whose only pathway is absorbed sunlight (albedo) does not enter it."""
    statements, plain = [], []
    contradicted = [m for m in mechs if m["status"] == "contradicted"]
    for m in contradicted:
        st = next((r["statistic_text"] for r in rows if r["mechanism"] == m["id"] and r["status"] == "contradicted"),
                  None)
        txt = GATE_TEXT.get(m["id"], f"{m['name']} is contradicted: its term is reported as statistical, not physics.")
        plain.append(txt)
        statements.append(txt[:-1] + f" ({st})." if st and txt.endswith(".") else txt)
    roles = (ctx["physics"].get("roles") or {})
    col_role = {col: role for role, col in roles.items() if col}
    status = physics_status
    active = status == "active"

    def brief(ms):
        return [{"id": m["id"], "name": m["name"], "status": m["status"], "label": m["label"]} for m in ms]

    def listing(ms):
        return ", ".join(f"{m['id']} {m['name']} ({m['label'].lower()})" for m in ms)

    levers = {}
    for col in (ctx["cfg"].get("actionable") or {}):
        role = col_role.get(col)
        named = [m for m in mechs if m["in_model"] and role is not None and role in m["roles"]]
        # without sunlight a role whose every pathway is absorbed sunlight (albedo) has no term in the source
        no_sun = bool(named) and not any(m["acts_in_run"] for m in named)
        in_source = role is not None and role in ctx["roles_used"] and role not in ("elevation",) and not no_sun
        on = in_source and active
        own = [m for m in named if on and m["acts_in_run"]]
        shared = [m for m in mechs if on and m["acts_in_run"] and _ALL in m["roles"]]
        entry = {"role": role, "in_physics": on, "physics_status": status, "mechanisms": brief(own),
                 "transport": brief(shared),
                 "supported": [m["id"] for m in own if m["status"] == "supported"],
                 "contradicted": [m["id"] for m in own + shared if m["status"] == "contradicted"],
                 "untested": [m["id"] for m in own if m["status"] in ("not_testable", "not_yet_tested")]}
        lab = ROLE_LABELS.get(role, role)
        name = f"{col}{f' ({lab})' if role and str(lab).lower() != col.lower() else ''}"
        if no_sun and role in ctx["roles_used"]:
            entry["statement"] = (f"{col} does not enter the physics source in this {ctx['window']} window (no "
                                  "sunlight: its only pathway, M1 absorbed sunlight, acts by day): its modelled effect "
                                  "is statistical only.")
        elif not in_source:
            entry["statement"] = f"{col} does not enter the physics source: its modelled effect is statistical only."
        elif not active:
            why = ("absent from this run" if status == "absent" else
                   "disabled in this run (anti-physical gain in every fold)")
            entry["statement"] = f"The physics model is {why}: {col}'s modelled effect is statistical only."
        else:
            s = (f"{name} acts in the physics model through {listing(own)}." if own else
                 f"{name} enters the physics source, but no mechanism of the register names it.")
            if shared:
                s += f" Transport and scale checks shared by every lever: {listing(shared)}."
            if entry["contradicted"]:
                s += (f" {', '.join(entry['contradicted'])} contradicted: do not cite "
                      f"{'it' if len(entry['contradicted']) == 1 else 'them'}; that part of the effect is statistical.")
            elif not entry["supported"]:
                s += (" None of its own mechanisms is supported yet, so the explanation is a hypothesis, not a tested "
                      "mechanism" + (" (the shared checks test the overall gain, not this lever's pathway)."
                                     if any(m["status"] == "supported" for m in shared) else "."))
            ident = ctx.get("ident")
            if ident:
                params = [ident["params"][k] for k in ROLE_PARAMS.get(role, ())
                          if k in ident["params"] and ident["params"][k]["in_model"]]
                if params:
                    entry["identifiability"] = {p["key"]: {k: p.get(k) for k in ("class", "ci95", "prior_share_run",
                                                                                   "reaches_zero", "run_value",
                                                                                   "estimate_run")}
                                                for p in params}
                    entry["not_identified"] = [p["key"] for p in params if p["class"] in NOT_IDENTIFIED]
                    s += " Identifiability from the data alone (W3): " + " ".join(p["text"] for p in params)
            entry["statement"] = s
        levers[col] = entry
    return {"statements": statements, "plain": plain, "contradicted": [m["id"] for m in contradicted],
            "levers": levers}


# --------------------------------------------------------------------------- markdown
def _md(s) -> str:
    return str(s if s is not None else "—").replace("|", "\\|")


def _tau_text(t: dict) -> str:
    """One assumed depth's time scale, with the range from r_a's ±2 SE interval when there is one."""
    s = f"h = {t['h_m']:g} m → τ = r_a·h ≈ {t['tau_min']:.0f} min, ωτ ≈ {t['omega_tau']:.2f}"
    tr, wr = t.get("tau_min_range"), t.get("omega_tau_range")
    if tr and wr:
        s += f" (±2 SE: τ {tr[0]:.0f}–{tr[1]:.0f} min, ωτ {wr[0]:.2f}–{wr[1]:.2f})"
    return s


def _ci_text(ci, unit: str = "") -> str:
    if not ci:
        return "—"
    lo, hi = ci
    return f"{_sig(lo) if lo is not None else 'open'} to {_sig(hi) if hi is not None else 'open'}" + (
        f" {unit}" if unit else "")


def identifiability_markdown_table(block: dict, reg: dict | None = None) -> str:
    """The register's identifiability block (``reg["identifiability"]``, status ``used``) as a markdown table, with
    the r_a / γ uncertainty check from the panel of ``reg``."""
    L = [f"**Identifiability from the data alone (W3).** Profile likelihood of each physics parameter refitted on all "
         f"labelled cells with every prior off (Raue et al. 2009; n = {int(block['n']) if _num(block.get('n')) else '—'}"
         + (f", {_sig(block['coarse_m'])} m cells" if _num(block.get("coarse_m")) else "") + "), "
         + covariance_text(block) + ". 95% interval: Λ_A ≤ 3.84 (2.71 one-sided for an estimate on its fit bound, "
         "Self & Liang 1987). Run: the run's own value (mean over its folds, as in the parameter panel). Full-data "
         "refit: the study's refit of the run's objective (penalty on) on all labelled cells, the value the prior "
         "share and the pull describe. Prior share: the penalty's share of the run objective's curvature in the "
         "parameter. Pull: (full-data refit − data-only estimate) / block-sandwich SE. Posterior: "
         "a Laplace approximation of the block-adjusted likelihood with the nominal priors (Ribatet, Cooley & "
         "Davison 2012)" + ("" if block.get("laplace_adequate") is not False else
                            ", poor for this run (the profile intervals do not rely on it)")
         + ". Power-scaling: " + PSENS_READING + "; \"not reliable\" where it rests on a poor Laplace "
         "approximation. Conflict: Evans & Moshonov (2006) tail probability.", "",
         "| parameter | run (fold mean) | full-data refit | data-only 95% CI (block sandwich) | with adjacent blocks | "
         "class | prior share | pull | posterior mean ± sd | power-scaling prior / likelihood | conflict p |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in (block.get("parameters") or {}).values():
        if not p.get("in_model"):
            continue
        u = p["unit"] if p["unit"] in ("m", "W/m²") else ""
        cls = p.get("class_label") or "—"
        if p.get("class") == "not_assessable":
            if p.get("ci95_naive"):
                cls += (f" (unadjusted one-sided bound {_ci_text(p['ci95_naive'], u)}, too narrow under spatial "
                        "dependence)")
        elif p.get("class") == "one_sided" and p.get("summary"):
            cls += f" ({p['summary']})"
        elif p.get("open_sides") and p.get("class") != "structurally_non_identifiable":
            cls += f" (open: {', '.join(p['open_sides'])})"
        if p.get("reaches_zero"):
            cls += "; does not exclude 0"
        post = p.get("posterior") or {}
        bad = "not reliable (Laplace poor)" if p.get("posterior_unreliable") else "—"
        pm = f"{_sig(post.get('mean'))} ± {_sig(post.get('sd'))}" if post.get("mean") is not None else bad
        ps = p.get("psens") or {}
        pss = (f"{_fmt(ps.get('prior'), 3)} / {_fmt(ps.get('likelihood'), 3)}"
               + (f": {_PSENS_SHORT[ps['reading']]}" if ps.get("reading") in _PSENS_SHORT else "")) if ps else bad
        cf = p.get("conflict") or {}
        cfs = (f"{float(cf['p']):.2g}" + (" (conflict)" if cf.get("flag") else "") if _num(cf.get("p")) else
               "not testable" if cf else "—")
        L.append(f"| {_md(p['label'])} | {_sig(p.get('run_value'), u)} | {_sig(p.get('estimate_run'), u)} | "
                 f"{_ci_text(p.get('ci95'), u)} | "
                 f"{_ci_text(p.get('ci95_neighbours'), u)} | {cls} | "
                 f"{_fmt(p.get('prior_share_run'), 2)} | {_fmt(p.get('pull'), 2)} | {pm} | {_md(pss)} | {cfs} |")
    for r in ((reg or {}).get("panel") or {}).get("rows") or []:
        sw = r.get("sandwich")
        if sw:
            L += ["", f"*{r['label']}:* {sw['text']}; fold jackknife {_fmt(r.get('value'), 1)} ± "
                      f"{_fmt(r.get('se'), 1)} {r['unit']} (the verdict uses the jackknife)"
                  + (f". Note: {sw['note']}" if sw.get("note") else "") + "."]
    for w in block.get("warnings") or []:
        L += ["", f"*Identifiability warning:* {w}"]
    return "\n".join(L)


def mechanisms_markdown(reg: dict | None) -> str:
    """The register, the parameter panel and the gates as markdown (report.md, the CLI)."""
    if not reg:
        return "_No mechanism register for this run._"
    if reg.get("error"):
        return f"_Mechanism register unavailable: {reg['error']}_"
    L = [f"Verdicts: **Supported** (the signature appears), **Contradicted** (absent or reversed: the term is "
         "reported as statistical, not physics), **Not testable with this data** (this run cannot measure the "
         "signature, or the estimate is too uncertain to test it; reason given), **Not yet tested** (the test belongs "
         "to a later workstream). A fitted value is Supported against a published range only when its point estimate "
         "lies inside the range and its ±2 SE interval (from at least three folds) is no wider than the range; it is "
         f"Contradicted when that interval lies entirely outside. Basis: {reg.get('basis')}.", "",
         "| mechanism | IF it holds, THEN … | test | verdict | statistic |", "|---|---|---|---|---|"]
    for r in reg.get("rows") or []:
        why = "; ".join(x for x in (r.get("reason") or r.get("basis"), r.get("note")) if x)
        verdict = f"**{r['label']}**" + (f" — {why}" if why else "")
        stat = r.get("statistic_text") or "—"
        if r.get("interval"):
            stat += (f" (±2 SE: {_fmt(r['interval'][0])} to {_fmt(r['interval'][1])}"
                     + (f"; range {r['reference']['text']}" if r.get("reference") else "") + ")")
        L.append(f"| {r['mechanism']}({r['sig']}) {_md(r['name'])} | {_md(r['then'])} | {_md(r['test'])} "
                 f"({r['workstream']}) | {_md(verdict)} | {_md(stat)} |")
    ctxs = [r for r in reg.get("rows") or [] if r.get("context")]
    seen = set()
    for r in ctxs:
        if r["mechanism"] not in seen:
            L += ["", f"*{r['mechanism']} context:* {r['context']}"]
            seen.add(r["mechanism"])
    P = reg.get("panel") or {}
    ib_status = (reg.get("identifiability") or {}).get("status")
    if P.get("rows"):
        later = ("Identifiability from the data alone and the priors' influence: the table after this one."
                 if P.get("identifiability_study") else
                 f"Identifiability: {P.get('identifiability')}." if ib_status == "disabled" else
                 f"Identifiability ({P.get('identifiability')}) and prior sensitivity ({P.get('prior_sensitivity')}) "
                 "come later.")
        L += ["", f"**Fitted physics in physical units** ({P.get('uncertainty')}; {P.get('n_active')} of "
                  f"{P.get('n_folds')} folds with the physics term on; {P.get('window')} window). "
                  f"{P.get('identified_products')} {later}", "",
              "| quantity | value ± SE | published range | source | status | note |", "|---|---|---|---|---|---|"]
        for r in P["rows"]:
            val = _value_text(r, bound=False)            # the status and note columns carry the fit bound
            pub = r["published"]["text"] if r.get("published") else "—"
            src = r["published"]["source"] if r.get("published") else "—"
            note = r.get("status_note") or ""
            if r.get("prior"):
                note += f" Prior N({r['prior']['mean']:g}, {r['prior']['sd']:g}²), information only."
            status = ("at fit bound" if r.get("at_bound") else r["status"].replace("_", " ")) + (
                " (imprecise: see note)" if r.get("interval_informative") is False else "")
            L.append(f"| {_md(r['label'])} | {_md(val)} | {_md(pub)} | {_md(src)} | {status} | {_md(note.strip())} |")
        if P.get("rows") and P["rows"][0].get("derivation"):
            L += ["", f"*Derivation:* {P['rows'][0]['derivation']}."]
        ts = P.get("timescales")
        if ts:
            L += ["", "*Implied time scales (assumptions, informational):* " + ts["assumption"] + " "
                  + (ts["caveat"] + ". " if ts.get("caveat") else "") + "; ".join(_tau_text(t) for t in ts["rows"]) + "."
                  + (f" {ts['advection_bound']['text']}." if ts.get("advection_bound") else "")
                  + (f" {ts['advection_note']}." if ts.get("advection_note") else "")]
        for n in P.get("notes") or []:
            L += ["", f"*Note:* {n}"]
    ib = reg.get("identifiability") or {}
    if ib.get("status") == "used":
        L += ["", identifiability_markdown_table(ib, reg)]
    elif ib.get("status") in ("ignored", "disabled"):
        L += ["", f"*Identifiability (W3):* {ib.get('note')}."]
        if ib.get("status") == "disabled":
            for w in ib.get("warnings") or []:
                L += ["", f"*Identifiability warning:* {w}"]
    G = reg.get("gates") or {}
    if G.get("statements"):
        L += ["", "**Gates:** " + " ".join(G["statements"])]
    if G.get("levers"):
        L += ["", "**Levers:**"] + [f"- {v['statement']}" for v in G["levers"].values()]
    return "\n".join(L)


# --------------------------------------------------------------------------- files (old run directories)
def _forcing_blob(physics: dict | None, config_dir=None) -> dict | None:
    """The forcing JSON named by ``physics.forcing`` (relative to ``config_dir``), or None."""
    rel = (physics or {}).get("forcing")
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute() and config_dir:
        p = Path(config_dir) / p
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def forcing_t2m_K(physics: dict | None, config_dir=None) -> float | None:
    """The ERA5 2 m temperature (K) of a run's forcing: ``forcing_info.t2m_K`` (runs made after it was
    recorded), else read from the forcing JSON (``physics.forcing``, relative to ``config_dir``)."""
    ph = physics or {}
    t = (ph.get("forcing_info") or {}).get("t2m_K")
    if _num(t):
        return float(t)
    era = (_forcing_blob(ph, config_dir) or {}).get("era5") or {}
    t = (era.get("values") or {}).get("t2m")
    if not _num(t) and _num(era.get("t2m_C")):
        t = float(era["t2m_C"]) + 273.15
    return float(t) if _num(t) else None


def _run_physics_params(run_dir: Path, m: dict) -> tuple[list | None, str | None]:
    """``physics.json`` of a run, only when it belongs to this run: the pipeline never deletes old outputs, so a
    re-run without physics (``models.physics: false``, or stopping before S2_S3) into the same directory leaves an
    earlier run's file behind.  It is used only when the manifest records a physics fit (``manifest["physics"]``,
    or for manifests without that key, a config that fits physics) and its fold means match the manifest's own
    summary.  Returns (params or None, a note on why the file was not used)."""
    p = run_dir / "physics.json"
    if not p.is_file():
        return None, None
    if "physics" in m:
        fitted = isinstance(m["physics"], dict) and bool(m["physics"])
    else:
        fitted = _physics_off(m.get("config") or {}) is None
    if not fitted:
        return None, "physics.json ignored: this run's manifest records no physics fit (a file left by an earlier run)"
    try:
        params = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "physics.json unreadable"
    if not isinstance(params, list) or not all(isinstance(x, dict) for x in params) or not params:
        return None, "physics.json is not a list of per-fold parameters"
    summary = m.get("physics") if isinstance(m.get("physics"), dict) else {}
    for key in ("a", "L_m", "gamma"):
        s = summary.get(key)
        if not (isinstance(s, dict) and _num(s.get("mean"))):
            continue
        vals = [x.get(key) for x in params]
        if not all(_num(v) for v in vals) or not math.isclose(
                sum(float(v) for v in vals) / len(vals), float(s["mean"]), rel_tol=1e-6, abs_tol=1e-9):
            return None, (f"physics.json ignored: its fold mean of {key} does not match this run's manifest (a file "
                          "left by an earlier run); the manifest's mean/sd summary is used")
    return params, None


def mechanisms_for_run(run_dir, config: dict | None = None, *, write: bool = True, identifiability=None,
                       identifiability_source: str = "argument", source: str = "mechanisms") -> dict:
    """Recompute the register of a finished run from ``manifest.json`` + ``physics.json`` and its identifiability
    study (:func:`run_identifiability`, or ``identifiability`` when given; used only when it belongs to the run);
    with ``write``, store ``mechanisms.json`` and the manifest's ``mechanisms`` section (``runio.update_manifest``,
    recorded with ``source``).  ``config``: a raw config dict to use instead of the manifest's (e.g. one that sets
    ``data.target_kind``).  ``physics.json`` is used only when it belongs to this run (:func:`_run_physics_params`)."""
    from sparc.core import progress, runio

    run_dir = Path(run_dir)
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if identifiability is None:
        identifiability, id_source = run_identifiability(run_dir, m)
        identifiability = identifiability or False
    else:
        id_source = identifiability_source
    params, params_note = _run_physics_params(run_dir, m)
    cfg = copy.deepcopy(config if config is not None else (m.get("config") or {}))
    ph = cfg.setdefault("physics", {})
    fi = ph.get("forcing_info")
    if ph.get("forcing") and not (isinstance(fi, dict) and _num(fi.get("t2m_K")) and fi.get("hours_local")):
        # runs made before forcing_info carried them: read the 2 m temperature and local hours from the file
        cdir = (m.get("provenance") or {}).get("config_dir")
        fi = dict(fi) if isinstance(fi, dict) else {"file": ph["forcing"]}
        t = forcing_t2m_K(ph, cdir)
        if t is not None and not _num(fi.get("t2m_K")):
            fi["t2m_K"] = t
        if not fi.get("hours_local"):
            fi["hours_local"] = (_forcing_blob(ph, cdir) or {}).get("hours_local")
        ph["forcing_info"] = fi
    reg = evaluate(m, params, cfg, identifiability=identifiability, identifiability_source=id_source)
    if params_note:
        reg["inputs"]["note"] = params_note
    if write:
        with progress.run_dir_scope(run_dir):
            runio.write_json_atomic(run_dir / "mechanisms.json", reg, indent=1)
            progress.artifact(run_dir / "mechanisms.json", role="mechanisms")
            runio.update_manifest(run_dir, {"mechanisms": reg}, source=source)
            progress.artifact(run_dir / "manifest.json", role="manifest")
    return reg
