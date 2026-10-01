"""Published effect sizes beside SPARC's estimates (a consistency panel).

Each entry quotes a published quantity with its conditions and the
reference key in ``docs/references.bib``.  ``status`` says how the value was
checked: ``"abstract"`` = taken from the paper's abstract / an indexed
summary of it (the full text was not consulted), ``"verify"`` = from memory,
to be checked before publication.  Never edit a value without its source.

SPARC's comparable number is the city-wide mean change of a *uniform*
scenario (fold-averaged), converted to °C per +0.10 cover fraction or albedo
— the "cooling effectiveness" definition of Krayenhoff et al. (2021).
Literature values are from other cities, methods and scales; agreement
within a factor of ~2 is what "consistent" can mean here.
"""

from __future__ import annotations

LITERATURE = [
    {"key": "krayenhoff2021", "quantity": "canopy", "per": "+0.10 canopy cover",
     "low_C": 0.3, "high_C": 0.3,
     "value": "≈0.3 °C cooling per +0.10 tree canopy cover",
     "conditions": "systematic review of numerical models (47 higher-quality studies); afternoon, clear sky, summer; "
                   "near-surface air",
     "citation": "Krayenhoff et al. 2021, Environ. Res. Lett. 16, 053007", "status": "abstract"},
    {"key": "ziter2019", "quantity": "canopy", "per": "+0.10 canopy cover",
     "low_C": 0.07, "high_C": 0.15,
     "value": "0 → 100% canopy: −0.7 °C (10 m radius), −1.3 °C (30 m), > −1.5 °C (60–90 m); nonlinear, cooling "
              "accelerates above ~40% canopy at block scale",
     "conditions": "bicycle transects, Madison WI, summer daytime; per-0.10 range is the linear average of the "
                   "10 m and 60–90 m totals (the real curve is steeper above 40%)",
     "citation": "Ziter et al. 2019, PNAS 116, 7575–7580", "status": "abstract"},
    {"key": "krayenhoff2021", "quantity": "albedo", "per": "+0.10 neighbourhood albedo",
     "low_C": 0.2, "high_C": 0.6,
     "value": "≈0.2–0.6 °C cooling per +0.10 neighbourhood albedo",
     "conditions": "numerical models; afternoon, clear sky, summer", "citation": "Krayenhoff et al. 2021",
     "status": "abstract"},
    {"key": "santamouris2014", "quantity": "albedo", "per": "+0.10 city albedo",
     "low_C": 0.3, "high_C": 0.9,
     "value": "city-wide +0.1 albedo: ≈0.3 K mean and ≈0.9 K peak ambient temperature decrease",
     "conditions": "review of simulation studies; peak (≈ afternoon) is the comparable figure",
     "citation": "Santamouris 2014, Solar Energy 103, 682–703", "status": "abstract"},
]


def sparc_effects(manifest: dict, canopy: str | None, albedo: str | None, units: str = "degF") -> dict:
    """SPARC's uniform-scenario effect per +0.10 (°C, positive = cooling) with its SE."""
    to_c = 5.0 / 9.0 if units.lower() in ("degf", "f", "°f", "fahrenheit") else 1.0
    out = {}
    for q, var, step in (("canopy", canopy, 10.0), ("albedo", albedo, 0.10)):
        if not var:
            continue
        best = None
        for s in manifest.get("scenarios") or []:
            mr = (s.get("mean_realized") or {})
            if set(mr) == {var} and mr[var] > 0:
                if best is None or abs(mr[var] - step) < abs(best["mean_realized"][var] - step):
                    best = s
        if best is None:
            continue
        dose = float(best["mean_realized"][var])
        scale = step / dose
        se = best.get("mean_delta_se")
        out[q] = {"scenario": best["name"], "realized_dose": dose,
                  "cooling_C": -float(best["mean_delta"]) * scale * to_c,
                  "se_C": float(se) * abs(scale) * to_c if se is not None else None,
                  "frac_extrapolated": best.get("frac_extrapolated"),
                  "causal_C": (-float(best["causal_linear"]["delta"]) * scale * to_c
                               if best.get("causal_linear") else None)}
    return out


def literature_panel(manifest: dict, canopy: str | None, albedo: str | None, units: str = "degF",
                     notes: dict | None = None) -> dict:
    est = sparc_effects(manifest, canopy, albedo, units)
    rows = []
    for e in LITERATURE:
        s = est.get(e["quantity"])
        row = {**e, "sparc_C": s["cooling_C"] if s else None, "sparc_se_C": s["se_C"] if s else None,
               "sparc_scenario": s["scenario"] if s else None}
        if s:
            mid = 0.5 * (e["low_C"] + e["high_C"])
            row["ratio_to_mid"] = s["cooling_C"] / mid if mid else None
            row["within_factor_2"] = bool(e["low_C"] / 2.0 <= s["cooling_C"] <= e["high_C"] * 2.0)
        rows.append(row)
    return {"rows": rows, "sparc": est, "notes": notes or {}}


def literature_markdown(panel: dict) -> str:
    L = ["| quantity | published | conditions | source (status) | SPARC (°C per +0.10) | ratio | within ×2 |",
         "|---|---|---|---|---|---|---|"]
    for r in panel["rows"]:
        sp = "—" if r["sparc_C"] is None else (f"{r['sparc_C']:.2f}" + (f" ± {r['sparc_se_C']:.2f}"
                                                                         if r.get("sparc_se_C") else ""))
        ratio = "—" if r.get("ratio_to_mid") is None else f"{r['ratio_to_mid']:.2f}"
        w = "—" if "within_factor_2" not in r else ("yes" if r["within_factor_2"] else "no")
        L.append(f"| {r['quantity']} | {r['value']} | {r['conditions']} | {r['citation']} ({r['status']}) | {sp} | "
                 f"{ratio} | {w} |")
    return "\n".join(L)
