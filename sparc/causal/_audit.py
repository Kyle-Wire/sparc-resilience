"""Wager (2025) causal-inference audit registry.

Tracks which of the 10 audit gaps have landed. Two complementary
mechanisms populate :data:`WAGER2025_GAPS_ADDRESSED`:

* :func:`_autodetect` — invoked by ``sparc audit causal`` and the
  Stage-3 add-on runner; imports each gap module and (for Gap 10) checks
  a module-level capability flag. Used to surface implementation status
  even when the gap code path has not actually executed in this run.
* :func:`mark_addressed` — called at runtime from inside individual gap
  implementations (e.g. ``EmpiricalWelfareMaximizer.fit`` calls
  ``mark_addressed(8)``) to record that the gap actually ran during the
  current process.

``report()`` formats the resulting flags into a checklist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class GapSpec:
    gap_id: int
    name: str
    module: str
    description: str


GAP_SPECS: Dict[int, GapSpec] = {
    1: GapSpec(1, "Generalized-propensity overlap",
               "sparc.causal.overlap",
               "Continuous-treatment overlap diagnostics + EXTRAPOLATED flags."),
    2: GapSpec(2, "RATE / TOC / QINI",
               "sparc.causal.cate_validation",
               "CATE ranking validation per Yadlowsky et al. (2025)."),
    3: GapSpec(3, "Network interference / spillover",
               "sparc.causal.interference",
               "Anonymous-exposure spillover estimator (Wager Def. 11.1)."),
    4: GapSpec(4, "Instrumental variables",
               "sparc.causal.iv",
               "Cross-fit IV estimation (Theorem 9.4)."),
    5: GapSpec(5, "Marginal Treatment Effects",
               "sparc.causal.iv",
               "Local IV / MTE per Heckman-Vytlacil Theorem 10.7."),
    6: GapSpec(6, "Sequential unconfoundedness",
               "sparc.causal.dynamic",
               "Sequential AIPW g-formula (Theorem 14.4)."),
    7: GapSpec(7, "DID / Synthetic DID",
               "sparc.causal.panel",
               "Panel methods: DID, staggered, SDID."),
    8: GapSpec(8, "Empirical-welfare policy learning",
               "sparc.decision.policy_learning",
               "Athey-Wager (2021) policy learning."),
    9: GapSpec(9, "CBPS balancing weights",
               "sparc.causal.balancing",
               "Covariate balancing propensity score diagnostic."),
    10: GapSpec(10, "Per-edge cross-fitting fix",
                "sparc.causal.counterfactual_engine",
                "EdgeNuisanceCache prevents nuisance reuse across outcomes."),
}


WAGER2025_GAPS_ADDRESSED: Dict[int, bool] = {gid: False for gid in GAP_SPECS}


def mark_addressed(gap_id: int) -> None:
    if gap_id not in GAP_SPECS:
        raise KeyError(f"Unknown gap id: {gap_id}")
    WAGER2025_GAPS_ADDRESSED[gap_id] = True


def report() -> str:
    lines = ["Wager (2025) causal-inference audit:"]
    n_done = sum(WAGER2025_GAPS_ADDRESSED.values())
    lines.append(f"  Status: {n_done}/{len(GAP_SPECS)} gaps addressed")
    lines.append("")
    for gid, spec in GAP_SPECS.items():
        flag = "[x]" if WAGER2025_GAPS_ADDRESSED[gid] else "[ ]"
        lines.append(f"  {flag} Gap {gid:2d}: {spec.name}")
        lines.append(f"          {spec.description}")
        lines.append(f"          module: {spec.module}")
    return "\n".join(lines)


def _autodetect() -> None:
    """Attempt to import each gap module; flip flag on success."""
    import importlib
    for gid, spec in GAP_SPECS.items():
        try:
            importlib.import_module(spec.module)
        except Exception:
            continue
        if gid == 10:
            try:
                mod = importlib.import_module(spec.module)
                if getattr(mod, "EDGE_NUISANCE_CACHE_ENABLED", False):
                    mark_addressed(gid)
            except Exception:
                pass
        else:
            mod = importlib.import_module(spec.module)
            if getattr(mod, f"GAP_{gid}_IMPLEMENTED", False):
                mark_addressed(gid)
