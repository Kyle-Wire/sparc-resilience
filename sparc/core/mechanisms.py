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
decision quantity (:meth:`sparc.core.ensemble.FittedEnsemble.jackknife_sd`): SE = sd·√(K−1).

Everything here is a pure function of plain dicts — the manifest, the per-fold physics parameters
(``physics.json``) and the config — so old run directories can be re-evaluated (``sparc core mechanisms
-r RUN_DIR``; :func:`mechanisms_for_run` is the only function that touches files).  Ranges and citations
come from the verified reference list of ``docs/roadmap/PHYSICS_PLAN.md``; never add a range without one.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

__all__ = ["VERDICTS", "VERDICT_LABELS", "TARGET_KINDS", "REGISTER", "REFERENCES", "RA_RANGE", "LAPSE_BAND",
           "to_kelvin_factor", "rho_cp", "forcing_norm", "physical_parameters", "evaluate", "mechanisms_markdown",
           "forcing_t2m_K", "mechanisms_for_run"]

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

# --------------------------------------------------------------------------- published ranges (verified list)
RA_RANGE = (20.0, 100.0)                  # s m⁻¹
RA_TEXT = "≈ 20–100 s/m"
RA_SOURCE = ("Masson (2000) canyon exchange formula: ≈ 100 s/m in calm air, ≈ 36 s/m at 5 m/s; "
             "Crawford et al. (2018): ≈ 40 s/m measured in central London")
LAPSE_BAND = (-9.8, -6.5)                 # K km⁻¹
LAPSE_TEXT = "−9.8 to −6.5 K/km"
LAPSE_SOURCE = "dry-adiabatic (−9.8 K/km) and standard-atmosphere (−6.5 K/km) lapse rates"
NIGHT_GAMMA = "no single physical range at night (inversions, cold-air pooling)"

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
}

# --------------------------------------------------------------------------- the register (plan §4)
_ONE_WINDOW = "one time window; needs the multi-window fit (W6)"
_ALL = "*"          # a transport mechanism: it shapes the effect of every lever that enters the source

#: Ten mechanisms of plan §4.  ``kind`` of a signature: ``evaluate`` (tested here), ``multi_window``
#: (not testable from one window), ``workstream`` (the test is not built yet) or ``missing_term``
#: (the term is not in the model).  ``roles``: the physics roles whose effect the mechanism explains.
REGISTER = (
    {"id": "M1", "name": "Absorbed sunlight", "terms": "albedo, canopy shade", "in_model": True,
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
     "chain": "Q_H → ΔT = r_a·Q_H/(ρc_p): the gain a sets r_a = a·ρc_p/SW↓",
     "roles": (_ALL,), "refs": ("masson2000", "crawford2018", "oke1973"),
     "signatures": (
         {"sig": "a", "then": "The implied r_a lies in the published range for urban surfaces.",
          "test": "Parameter panel: r_a = a·ρc_p/SW↓, mean ± 2 SE against " + RA_TEXT, "workstream": "W3",
          "kind": "evaluate"},
         {"sig": "b", "then": "Across runs and campaigns, the gain a falls as wind speed rises; heat-island "
                              "intensity weakens with wind (Oke 1973).",
          "test": "Multi-window and multi-campaign fits", "workstream": "W6", "kind": "multi_window"})},
    {"id": "M7", "name": "Elevation (lapse rate)", "terms": "γ·elevation", "in_model": True,
     "chain": "elevation → γ·elev (an additive term beside the transported source) → ΔT",
     "roles": ("elevation",), "refs": (),
     "signatures": (
         {"sig": "a", "then": "γ lies between about −9.8 and −6.5 K/km in the well-mixed afternoon.",
          "test": "Parameter panel: γ in K/km, mean ± 2 SE against the lapse-rate band", "workstream": "W3",
          "kind": "evaluate"},
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


def _jackknife_se(values) -> float | None:
    """Delete-a-group jackknife SE over K fold models (the repo's decision-quantity helper)."""
    import numpy as np

    from sparc.core.ensemble import FittedEnsemble

    v = np.asarray(values, float)
    return float(FittedEnsemble.jackknife_sd(v)) if v.size > 1 else None


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


def _active(p: dict) -> bool:
    """A fold whose physics term is on: positive gain and no anti-physical fallback."""
    return _num(p.get("a")) and float(p["a"]) > 0 and not p.get("fit_warning")


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
        """Mean, jackknife SE and per-fold values of ``key`` × ``scale`` over the folds with physics on."""
        if self.params:
            vals = [float(p[key]) for p in self.active if _num(p.get(key))]
            if not vals:
                return None
            f = transform or (lambda x: x * scale)
            vals = [float(f(v)) for v in vals]
            return {"value": sum(vals) / len(vals), "se": _jackknife_se(vals), "per_fold": vals, "n": len(vals)}
        if self.summary and self.n_active and isinstance(self.summary.get(key), dict):
            s = self.summary[key]
            if not _num(s.get("mean")):
                return None
            if transform is not None:          # nonlinear: no SE from a mean/sd summary
                return {"value": float(transform(float(s["mean"]))), "se": None, "per_fold": None, "n": self.n}
            se = float(s.get("sd") or 0.0) * math.sqrt(self.n - 1) * abs(scale) if self.n > 1 else None
            return {"value": float(s["mean"]) * scale, "se": se, "per_fold": None, "n": self.n}
        return None


def _interval(st: dict | None) -> list[float] | None:
    if not st or st.get("value") is None:
        return None
    se = st.get("se") or 0.0
    return [st["value"] - 2.0 * se, st["value"] + 2.0 * se]


def _band_status(st: dict | None, band: tuple[float, float]) -> tuple[str, bool | None]:
    """Panel status of a point estimate and whether its mean ± 2 SE interval overlaps ``band``."""
    iv = _interval(st)
    if iv is None:
        return "not_comparable", None
    v = st["value"]
    inside = band[0] <= v <= band[1]
    return ("inside" if inside else "outside"), bool(iv[1] >= band[0] and iv[0] <= band[1])


def _fmt(v, nd=1) -> str:
    if not _num(v):
        return "—"
    s = f"{float(v):.{nd}f}"
    return s.replace("-", "−")


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
    folds = _Folds(fold_params, m.get("physics"), (m.get("cv") or {}).get("n_folds"))
    window = str(folds.first("window") or ph.get("window") or "day").lower()
    roles_used = folds.first("roles_used")
    if roles_used is None:
        roles_used = [r for r, col in (ph.get("roles") or {}).items() if col]
    shade_form = str(folds.first("shade_form") or ph.get("shade_form") or "saturating").lower()
    return {"cfg": cfg, "physics": ph, "units": units, "folds": folds, "window": window,
            "roles_used": sorted(roles_used), "shade_form": shade_form, "enabled": ph.get("enabled", True),
            "target_kind": data.get("target_kind"), "manifest": m}


def _source_rows(ctx: dict) -> list[dict]:
    """The fitted source parameters (s, κ, a₁/a₁ₙ, EF slopes, water sink) with their priors as information."""
    F, roles, day = ctx["folds"], set(ctx["roles_used"]), ctx["window"] == "day"
    priors = F.first("priors") or ctx["physics"].get("priors") or {}
    sigmoid = ctx["shade_form"] == "sigmoid"
    no_range = "no published range for this coefficient in the plan's verified reference list"
    specs = [
        ("s", "Canopy shading strength s", "fraction of absorbed sunlight", day and "canopy" in roles, "s",
         "Appendix D: with a tight prior (sd 0.1) s stayed near its prior even for a displaced canopy layer; "
         "whether this map identifies s is the profile-likelihood test (W3)", ()),
        ("kappa_canopy", "Shade saturation scale κ", "canopy fraction", day and "canopy" in roles and not sigmoid,
         None, "weak log-normal regulariser log κ ~ N(0, 3²); large κ is a linear shade response", ()),
        ("shade_c0", "Shade inflection c₀", "canopy fraction", day and "canopy" in roles and sigmoid, None,
         no_range, ()),
        ("shade_width", "Shade transition width", "canopy fraction", day and "canopy" in roles and sigmoid, None,
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
        rows.append({"key": key, "label": label, "unit": unit, **st, "interval": _interval(st), "published": None,
                     "status": "not_comparable", "status_note": note, "refs": list(refs), "group": "source",
                     "prior": ({"mean": float(pr[0]), "sd": float(pr[1]),
                                "note": "information only; prior sensitivity is not yet tested (W3 power-scaling)"}
                               if pr and len(pr) == 2 else None)})
    return rows


def _t2m(ph: dict) -> tuple[float, str, bool]:
    t = (ph.get("forcing_info") or {}).get("t2m_K")
    if _num(t) and float(t) > 150.0:
        return float(t), f"ERA5 2 m temperature of the forcing window ({float(t):.1f} K)", True
    return T_FALLBACK_K, f"assumed {T_FALLBACK_K:.2f} K: the forcing gives no 2 m temperature", False


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
    rows: list[dict] = []
    ra = None
    if fK is not None:
        scale = fK * rcp / norm
        ra = F.stat("a", scale)
        st_ra, overlaps = _band_status(ra, RA_RANGE)
        rows.append({"key": "r_a", "label": "Aerodynamic resistance r_a", "unit": "s/m",
                     **(ra or {"value": None, "se": None, "per_fold": None, "n": 0}), "interval": _interval(ra),
                     "published": {"low": RA_RANGE[0], "high": RA_RANGE[1], "text": RA_TEXT, "source": RA_SOURCE},
                     "status": st_ra, "interval_overlaps": overlaps, "refs": ["masson2000", "crawford2018"],
                     "derivation": (f"r_a = a·{'(5/9)·' if fK != 1.0 else ''}ρc_p/{norm_label} with ρc_p = {rcp:.0f} "
                                    f"J m⁻³ K⁻¹ (p = {P_STD:.0f} Pa, T: {t_note}) and {norm_label} = {norm:.1f} W/m²"),
                     "status_note": ("q's scale depends on the albedo layer, so a biased albedo biases a and r_a (F5)"
                                     + ("; Crawford et al. (2018) is for unstable (daytime) conditions"
                                        if window != "day" else "")),
                     "group": "transport"})
        has_elev = "elevation" in ctx["roles_used"]
        g = F.stat("gamma", fK * 1000.0) if has_elev else None
        if window == "day":
            st_g, ov_g = _band_status(g, LAPSE_BAND)
            pub = {"low": LAPSE_BAND[0], "high": LAPSE_BAND[1], "text": LAPSE_TEXT, "source": LAPSE_SOURCE}
            note = ("assumes the elevation layer is in metres; an indicative γ far outside the band means elevation "
                    "is standing in for something else (F2)")
        else:
            st_g, ov_g, pub = ("not_comparable", None, None)
            note = NIGHT_GAMMA
        if not has_elev:
            st_g, ov_g, note = "not_comparable", None, "no elevation role: γ is not fitted"
        rows.append({"key": "gamma", "label": "Elevation coefficient γ", "unit": "K/km",
                     **(g or {"value": None, "se": None, "per_fold": None, "n": 0}), "interval": _interval(g),
                     "published": pub, "status": st_g, "interval_overlaps": ov_g, "refs": [], "status_note": note,
                     "group": "transport"})
    L = F.stat("L_m")
    lb = F.first("L_bounds_m")
    L_prior = ((ctx["manifest"].get("influence") or {}).get("L_prior_m"))
    rows.append({"key": "L", "label": "Mixing length L", "unit": "m",
                 **(L or {"value": None, "se": None, "per_fold": None, "n": 0}), "interval": _interval(L),
                 "published": None, "status": "not_comparable", "refs": [], "group": "transport",
                 "status_note": ("L² = Kτ: one map identifies the product, not K and τ separately (F4); no published "
                                 "range in the verified list"
                                 + (f"; S1 correlogram prior L₀ = {L_prior:.0f} m" if _num(L_prior) else "")
                                 + (f"; fit bounds {lb[0]:.0f}–{lb[1]:.0f} m" if lb and len(lb) == 2 else ""))})
    v = F.stat("v_norm_m")
    if v and v["value"] > 1.0:
        rows.append({"key": "v", "label": "Advection drift |v| = |u|·τ", "unit": "m", **v, "interval": _interval(v),
                     "published": None, "status": "not_comparable", "refs": [], "group": "transport",
                     "status_note": "v = u_c·τ: the drift times a canopy-layer wind; kept after the out-of-fold test"})
    rows += _source_rows(ctx)
    for r in rows:
        if r["value"] is None and r["status"] != "not_comparable":
            r["status"] = "not_comparable"
    for r in rows:
        r["basis"] = basis if r.get("value") is not None else None
    return {"rows": rows, "physics_status": status, "n_folds": F.n, "n_active": F.n_active, "source": F.source,
            "window": window, "units": ctx["units"], "to_kelvin": fK, "t2m_K": t_K, "t2m_note": t_note,
            "t2m_known": t_known, "rho_cp": rcp, "pressure_Pa": P_STD, "norm": norm, "norm_label": norm_label,
            "basis": basis, "target_kind": kind, "timescales": _timescales(ra, v, ph),
            "identified_products": "One afternoon map identifies r_a = τ/h and L² = Kτ, not τ, h and K separately "
                                   "(F4); the panel does not split them.",
            "identifiability": "not yet tested: W3 profile likelihood",
            "prior_sensitivity": "not yet tested: W3 power-scaling",
            "uncertainty": "mean ± delete-a-group jackknife SE over the fold models (SE = sd·√(K−1))",
            "notes": notes}


def _timescales(ra: dict | None, v: dict | None, ph: dict) -> dict | None:
    """Informational time scales from an *assumed* mixing depth h: τ = r_a·h and ωτ (W6's quasi-steady
    argument); a kept advection drift bounds τ from below, since the canopy-layer wind is at most the 10 m one."""
    if not ra or not _num(ra.get("value")) or ra["value"] <= 0:
        return None
    rows = []
    for h in H_ASSUMED_M:
        tau = ra["value"] * h
        rows.append({"h_m": h, "tau_s": tau, "tau_min": tau / 60.0, "omega_tau": OMEGA_DAY * tau,
                     "phase_lag_deg": math.degrees(math.atan(OMEGA_DAY * tau))})
    out = {"assumption": f"h is assumed, not fitted: canopy-layer depths of {H_ASSUMED_M[0]:g} and "
                         f"{H_ASSUMED_M[1]:g} m (W6). One map identifies r_a = τ/h only.",
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
    if not ctx["enabled"]:
        return "the physics model is disabled in this run's config"
    if not F.present:
        return "no physics model in this run (no physics.json and no physics summary)"
    if F.n_active == 0:
        return ("physics term disabled: the fitted gain a ≤ 0 (anti-physical) in every fold, so the model "
                "fell back to its linear terms")
    if to_kelvin_factor(ctx["units"]) is None:
        return f"target units {ctx['units']!r} are not a temperature scale (°F, °C or K)"
    return None


def _band_verdict(st: dict, band: tuple[float, float]) -> str:
    iv = _interval(st)
    return "supported" if (iv[1] >= band[0] and iv[0] <= band[1]) else "contradicted"


def evaluate(manifest: dict | None, fold_params: list[dict] | None = None, config: dict | None = None, *,
             units: str | None = None) -> dict:
    """The mechanism register of a run: one row per mechanism signature with status, statistic, interval,
    basis and note; the W3 parameter panel; gates for scenario text.

    ``manifest``: the run's manifest (``physics_advection``, ``influence``, ``cv`` and, without
    ``fold_params``, the ``physics`` mean/sd summary are read from it); ``fold_params``: the per-fold
    ``PhysicsModel.params`` (``physics.json``); ``config``: the raw config dict (default: the manifest's);
    ``units``: the target units (default: ``data.target_units``)."""
    ctx = _context(manifest, fold_params, config, units)
    panel = _panel(ctx)
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
                r = _eval_ra(mech, sig, by_key.get("r_a"), block, ctx, panel)
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
                r["context"] = f"Fitted L = {_fmt(by_key['L']['value'], 0)} ± {_fmt(by_key['L']['se'], 0)} m."
            missing = [x for x in mech["roles"] if x != _ALL and x not in ctx["roles_used"]]
            if sig["kind"] != "evaluate" and mech["roles"] and _ALL not in mech["roles"] \
                    and len(missing) == len(mech["roles"]):
                r["note"] = (r["note"] + " " if r["note"] else "") + \
                    f"No {', '.join(ROLE_LABELS.get(x, x) for x in missing)} layer in this run's physics."
            rows.append(r)
    mechs = []
    for mech in REGISTER:
        mr = [r for r in rows if r["mechanism"] == mech["id"]]
        st = next(s for s in _PRECEDENCE if any(r["status"] == s for r in mr))
        mechs.append({"id": mech["id"], "name": mech["name"], "terms": mech["terms"], "chain": mech["chain"],
                      "in_model": mech["in_model"], "roles": list(mech["roles"]), "status": st,
                      "label": VERDICT_LABELS[st], "signatures": [r["id"] for r in mr], "refs": list(mech["refs"])})
    used = sorted({k for m in REGISTER for k in m["refs"]} | {"oreskes1994", "mooij2013", "shandas2019"})
    summary = {s: [r["id"] for r in rows if r["status"] == s] for s in VERDICTS}
    return {
        "schema": SCHEMA, "verdicts": dict(VERDICT_LABELS), "window": ctx["window"], "units": ctx["units"],
        "target_kind": kind, "basis": basis, "physics_status": panel["physics_status"],
        "rows": rows, "mechanisms": mechs, "summary": summary, "panel": panel,
        "gates": _gates(rows, mechs, ctx, by_key),
        "references": {k: REFERENCES[k] for k in used},
        "inputs": {"source": F.source, "n_folds": F.n, "n_active": F.n_active, "fit_warnings": F.warnings,
                   "advection": adv},
    }


def _eval_ra(mech, sig, row, block, ctx, panel) -> dict:
    if block:
        return _row(mech, sig, "not_testable", reason=block)
    if not row or row.get("value") is None:
        return _row(mech, sig, "not_testable", reason="no fitted gain a to convert")
    st = {"value": row["value"], "se": row["se"]}
    status = _band_verdict(st, RA_RANGE)
    note = None if panel["t2m_known"] else f"ρc_p uses T = {panel['t2m_note']}"
    if ctx["window"] != "day":
        note = (note + "; " if note else "") + "night window: Crawford et al. (2018) is for unstable (daytime) conditions"
    if status == "supported" and not (RA_RANGE[0] <= row["value"] <= RA_RANGE[1]):
        note = (note + "; " if note else "") + "the point estimate lies outside the range, the ±2 SE interval reaches it"
    return _row(mech, sig, status, statistic={"name": "r_a", "value": row["value"], "se": row["se"], "unit": "s/m"},
                statistic_text=_stat_text("r_a", st, "s/m"), interval=_interval(st),
                reference={"low": RA_RANGE[0], "high": RA_RANGE[1], "text": RA_TEXT, "source": RA_SOURCE}, note=note)


def _eval_gamma(mech, sig, row, block, ctx) -> dict:
    if block:
        return _row(mech, sig, "not_testable", reason=block)
    if "elevation" not in ctx["roles_used"]:
        return _row(mech, sig, "not_testable", reason="no elevation layer in the physics roles: γ is not fitted")
    if not row or row.get("value") is None:
        return _row(mech, sig, "not_testable", reason="no fitted elevation coefficient")
    st = {"value": row["value"], "se": row["se"]}
    common = dict(statistic={"name": "gamma", "value": row["value"], "se": row["se"], "unit": "K/km"},
                  statistic_text=_stat_text("γ", st, "K/km"), interval=_interval(st))
    if ctx["window"] != "day":
        return _row(mech, sig, "not_testable", reason=f"night window: {NIGHT_GAMMA}", **common)
    status = _band_verdict(st, LAPSE_BAND)
    note = "assumes the elevation layer is in metres"
    if status == "contradicted":
        note += ("; elevation is standing in for something else, possibly a missing source such as nearby water "
                 "carried by the wind (F2, tested by W2)")
    return _row(mech, sig, status, reference={"low": LAPSE_BAND[0], "high": LAPSE_BAND[1], "text": LAPSE_TEXT,
                                              "source": LAPSE_SOURCE}, note=note, **common)


def _gates(rows: list[dict], mechs: list[dict], ctx: dict, by_key: dict) -> dict:
    """Plain statements from contradicted rows, and per actionable lever which mechanisms explain it."""
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
    levers = {}
    for col in (ctx["cfg"].get("actionable") or {}):
        role = col_role.get(col)
        in_source = role is not None and role in ctx["roles_used"] and role not in ("elevation",)
        rel = [m for m in mechs if in_source and m["in_model"] and (role in m["roles"] or _ALL in m["roles"])]
        entry = {"role": role, "in_physics": in_source,
                 "mechanisms": [{"id": m["id"], "name": m["name"], "status": m["status"], "label": m["label"]}
                                for m in rel],
                 "supported": [m["id"] for m in rel if m["status"] == "supported"],
                 "contradicted": [m["id"] for m in rel if m["status"] == "contradicted"],
                 "untested": [m["id"] for m in rel if m["status"] in ("not_testable", "not_yet_tested")]}
        if not in_source:
            entry["statement"] = (f"{col} does not enter the physics source: its modelled effect is statistical "
                                  "only.")
        else:
            parts = ", ".join(f"{m['id']} {m['name']} ({m['label'].lower()})" for m in rel)
            lab = ROLE_LABELS.get(role, role)
            s = f"{col}{f' ({lab})' if lab != col else ''} acts in the physics model through {parts}."
            if entry["contradicted"]:
                s += (f" {', '.join(entry['contradicted'])} contradicted: do not cite "
                      f"{'it' if len(entry['contradicted']) == 1 else 'them'}; that part of the effect is statistical.")
            elif not entry["supported"]:
                s += " None is supported yet, so the explanation is a hypothesis, not a tested mechanism."
            entry["statement"] = s
        levers[col] = entry
    return {"statements": statements, "plain": plain, "contradicted": [m["id"] for m in contradicted],
            "levers": levers}


# --------------------------------------------------------------------------- markdown
def _md(s) -> str:
    return str(s if s is not None else "—").replace("|", "\\|")


def mechanisms_markdown(reg: dict | None) -> str:
    """The register, the parameter panel and the gates as markdown (report.md, the CLI)."""
    if not reg:
        return "_No mechanism register for this run._"
    if reg.get("error"):
        return f"_Mechanism register unavailable: {reg['error']}_"
    L = [f"Verdicts: **Supported** (the signature appears), **Contradicted** (absent or reversed: the term is "
         "reported as statistical, not physics), **Not testable with this data** (reason given), **Not yet tested** "
         f"(the test belongs to a later workstream). Basis: {reg.get('basis')}.", "",
         "| mechanism | IF it holds, THEN … | test | verdict | statistic |", "|---|---|---|---|---|"]
    for r in reg.get("rows") or []:
        why = "; ".join(x for x in (r.get("reason") or r.get("basis"), r.get("note")) if x)
        verdict = f"**{r['label']}**" + (f" — {why}" if why else "")
        stat = r.get("statistic_text") or "—"
        if r.get("interval"):
            stat += f" (±2 SE: {_fmt(r['interval'][0])} to {_fmt(r['interval'][1])}; range {r['reference']['text']})" \
                if r.get("reference") else ""
        L.append(f"| {r['mechanism']}({r['sig']}) {_md(r['name'])} | {_md(r['then'])} | {_md(r['test'])} "
                 f"({r['workstream']}) | {_md(verdict)} | {_md(stat)} |")
    ctxs = [r for r in reg.get("rows") or [] if r.get("context")]
    seen = set()
    for r in ctxs:
        if r["mechanism"] not in seen:
            L += ["", f"*{r['mechanism']} context:* {r['context']}"]
            seen.add(r["mechanism"])
    P = reg.get("panel") or {}
    if P.get("rows"):
        L += ["", f"**Fitted physics in physical units** ({P.get('uncertainty')}; {P.get('n_active')} of "
                  f"{P.get('n_folds')} folds with the physics term on; {P.get('window')} window). "
                  f"{P.get('identified_products')} Identifiability ({P.get('identifiability')}) and prior "
                  f"sensitivity ({P.get('prior_sensitivity')}) come later.", "",
              "| quantity | value ± SE | published range | source | status | note |", "|---|---|---|---|---|---|"]
        for r in P["rows"]:
            nd = 0 if r["key"] in ("L", "v", "w") else (1 if r["key"] in ("r_a", "gamma") else 3)
            val = "—" if r.get("value") is None else (
                f"{_fmt(r['value'], nd)}" + (f" ± {_fmt(r['se'], nd)}" if r.get("se") is not None else "")
                + f" {r['unit']}")
            pub = r["published"]["text"] if r.get("published") else "—"
            src = r["published"]["source"] if r.get("published") else "—"
            note = r.get("status_note") or ""
            if r.get("prior"):
                note += f" Prior N({r['prior']['mean']:g}, {r['prior']['sd']:g}²), information only."
            L.append(f"| {_md(r['label'])} | {_md(val)} | {_md(pub)} | {_md(src)} | {r['status'].replace('_', ' ')} "
                     f"| {_md(note.strip())} |")
        if P.get("rows") and P["rows"][0].get("derivation"):
            L += ["", f"*Derivation:* {P['rows'][0]['derivation']}."]
        ts = P.get("timescales")
        if ts:
            L += ["", "*Implied time scales (assumptions, informational):* " + ts["assumption"] + " " + "; ".join(
                f"h = {t['h_m']:g} m → τ = r_a·h ≈ {t['tau_min']:.0f} min, ωτ ≈ {t['omega_tau']:.2f}"
                for t in ts["rows"]) + "."
                + (f" {ts['advection_bound']['text']}." if ts.get("advection_bound") else "")
                + (f" {ts['advection_note']}." if ts.get("advection_note") else "")]
        for n in P.get("notes") or []:
            L += ["", f"*Note:* {n}"]
    G = reg.get("gates") or {}
    if G.get("statements"):
        L += ["", "**Gates:** " + " ".join(G["statements"])]
    if G.get("levers"):
        L += ["", "**Levers:**"] + [f"- {v['statement']}" for v in G["levers"].values()]
    return "\n".join(L)


# --------------------------------------------------------------------------- files (old run directories)
def forcing_t2m_K(physics: dict | None, config_dir=None) -> float | None:
    """The ERA5 2 m temperature (K) of a run's forcing: ``forcing_info.t2m_K`` (runs made after it was
    recorded), else read from the forcing JSON (``physics.forcing``, relative to ``config_dir``)."""
    ph = physics or {}
    t = (ph.get("forcing_info") or {}).get("t2m_K")
    if _num(t):
        return float(t)
    rel = ph.get("forcing")
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute() and config_dir:
        p = Path(config_dir) / p
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    era = blob.get("era5") or {}
    t = (era.get("values") or {}).get("t2m")
    if not _num(t) and _num(era.get("t2m_C")):
        t = float(era["t2m_C"]) + 273.15
    return float(t) if _num(t) else None


def mechanisms_for_run(run_dir, config: dict | None = None, *, write: bool = True) -> dict:
    """Recompute the register of a finished run from ``manifest.json`` + ``physics.json``; with ``write``,
    store ``mechanisms.json`` and the manifest's ``mechanisms`` section (``runio.update_manifest``).
    ``config``: a raw config dict to use instead of the manifest's (e.g. one that sets ``data.target_kind``)."""
    from sparc.core import progress, runio

    run_dir = Path(run_dir)
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        params = json.loads((run_dir / "physics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        params = None
    cfg = copy.deepcopy(config if config is not None else (m.get("config") or {}))
    ph = cfg.setdefault("physics", {})
    if not _num((ph.get("forcing_info") or {}).get("t2m_K")):
        t = forcing_t2m_K(ph, (m.get("provenance") or {}).get("config_dir"))
        if t is not None and ph.get("forcing_info") is not None:
            ph["forcing_info"] = {**ph["forcing_info"], "t2m_K": t}
    reg = evaluate(m, params if isinstance(params, list) else None, cfg)
    if write:
        with progress.run_dir_scope(run_dir):
            runio.write_json_atomic(run_dir / "mechanisms.json", reg, indent=1)
            progress.artifact(run_dir / "mechanisms.json", role="mechanisms")
            runio.update_manifest(run_dir, {"mechanisms": reg}, source="mechanisms")
            progress.artifact(run_dir / "manifest.json", role="manifest")
    return reg
