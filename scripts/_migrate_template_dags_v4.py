"""
One-shot migration: append SPARC v4 `assumptions:` blocks to every
template DAG.  Idempotent — skips any file that already declares an
`assumptions:` key at the top level.

Usage:  python scripts/_migrate_template_dags_v4.py
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"

# Per-domain identification assumptions.  Spatial domains generally
# violate `no_interference` (spillover); we mark that explicitly so
# downstream reports aren't silently optimistic.
DOMAIN_ASSUMPTIONS: dict[str, dict[str, object]] = {
    "air_quality": {
        "no_interference": False,
        "notes": (
            "Air-quality measurements are spatially auto-correlated "
            "(plume transport, regional inversion).  Treat near-source "
            "spillover with monitor-network–level adjustments."
        ),
    },
    "coastal": {
        "no_interference": False,
        "notes": (
            "Longshore sediment transport links adjacent transects — "
            "SUTVA holds only at the sub-littoral-cell scale."
        ),
    },
    "drought": {
        "no_interference": False,
        "notes": (
            "Drought indices (SPI/SPEI) are smoothed across catchments. "
            "Watershed-level interventions interfere via shared "
            "groundwater + atmosphere."
        ),
    },
    "forcesmip": {
        "no_interference": True,
        "notes": (
            "Counterfactual forcing scenarios are independent by "
            "construction — each ensemble member is a self-contained "
            "Earth-system trajectory."
        ),
    },
    "geotechnical": {
        "no_interference": False,
        "notes": (
            "Subsurface stress fields couple adjacent boreholes; "
            "treat SUTVA as approximate at the borehole scale."
        ),
    },
    "groundwater": {
        "no_interference": False,
        "notes": (
            "Hydraulic head propagates through aquifers — pumping at "
            "one well affects neighbouring wells (Theis 1935)."
        ),
    },
    "noise": {
        "no_interference": False,
        "notes": (
            "Noise propagates omnidirectionally; barriers and source "
            "reductions affect a wide footprint, violating SUTVA at "
            "fine scales."
        ),
    },
    "seismic": {
        "no_interference": True,
        "notes": (
            "Site amplification is treated as a local property; "
            "regional source effects are absorbed by ground-motion "
            "prediction equations."
        ),
    },
    "stormwater": {
        "no_interference": False,
        "notes": (
            "Sub-catchments cascade — upstream LID/green infrastructure "
            "alters downstream loading.  Model with routing-aware "
            "spatial CV."
        ),
    },
    "uhi": {
        "no_interference": False,
        "notes": (
            "Heat-island advection couples adjacent blocks. "
            "Greening/cool-roof interventions diffuse 50–200 m."
        ),
    },
    "water_quality": {
        "no_interference": False,
        "notes": (
            "Pollutant transport along stream networks violates SUTVA. "
            "Use upstream/downstream graph adjacency in CV."
        ),
    },
    "wildfire": {
        "no_interference": False,
        "notes": (
            "Fire spreads — fuel-treatment effectiveness depends on "
            "neighbouring patches.  SUTVA holds only at landscape scale."
        ),
    },
    "blank": {
        "no_interference": True,
        "notes": "Edit these flags to reflect your domain's identification claims.",
    },
}

DEFAULT = {
    "conditional_exchangeability": True,
    "positivity": True,
    "consistency": True,
}


def render_block(domain: str) -> str:
    spec = {**DEFAULT, **DOMAIN_ASSUMPTIONS.get(domain, {})}
    notes = spec.pop("notes", "")
    lines = [
        "",
        "# -----------------------------------------------------------------------------",
        "# SPARC v4 — User-asserted identification assumptions.",
        "# Surfaced in the validation report alongside empirical checks",
        "# (positivity, SUTVA / Moran's I, d-separation).",
        "# Per-edge `sign_prior` / `confidence` / `kind` / `lag` may also be added.",
        "# -----------------------------------------------------------------------------",
        "assumptions:",
        f"  conditional_exchangeability: {str(spec['conditional_exchangeability']).lower()}",
        f"  positivity: {str(spec['positivity']).lower()}",
        f"  consistency: {str(spec['consistency']).lower()}",
        f"  no_interference: {str(spec['no_interference']).lower()}",
    ]
    if notes:
        lines.append("  notes: |")
        for ln in notes.split(". "):
            ln = ln.strip()
            if not ln:
                continue
            if not ln.endswith("."):
                ln += "."
            lines.append(f"    {ln}")
    return "\n".join(lines) + "\n"


def main() -> None:
    files = sorted(TEMPLATES.glob("*/causal/dag.yml"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        if re.search(r"^assumptions:", text, re.MULTILINE):
            print(f"  skip   {path.relative_to(ROOT)}  (already migrated)")
            continue
        domain = path.parent.parent.name
        block = render_block(domain)
        # Ensure a trailing newline before appending
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text + block, encoding="utf-8")
        print(f"  +block {path.relative_to(ROOT)}  ({domain})")


if __name__ == "__main__":
    main()
