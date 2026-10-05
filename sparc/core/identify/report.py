"""Plain-language reports: the identification lab (which designs can be trusted, on this city's layout) and
an estimate from a real campaign (what the trusted designs say, and how far to trust it)."""

from __future__ import annotations

ESTIMAND_TEXT = {
    "within_100": "+10 pp canopy within 100 m of a street point",
    "within_300": "+10 pp canopy within 300 m of a street point",
    "within_1000": "+10 pp canopy within 1 km of a street point",
    "total": "+10 pp canopy everywhere (the city-wide scenario)",
    "contrast": "upwind − downwind canopy, +10 pp",
}
STATUS_TEXT = {"trustworthy": "Trustworthy", "conservative": "Conservative (a lower bound)",
               "partial": "Partly trustworthy", "direction only": "Direction only", "underpowered": "Clean but underpowered",
               "not trustworthy": "Not trustworthy"}
WORLD_TEXT = {
    "null": "canopy has no effect",
    "additive": "cooling within ~150 m of the trees",
    "own_only": "cooling only in the tree's own cell",
    "physics": "cooling carried downwind over ~1 km (advection–diffusion)",
    "coarse_scale": "cooling spread over ~1 km",
    "confounded": "local cooling plus a hidden 2 km factor that tracks canopy",
}


def _f(v, fmt="{:+.3f}"):
    return "—" if v is None else fmt.format(v)


def overview_rows(summ: dict) -> list[dict]:
    """One row per design: its verdict and, per world, the share of its estimand it recovers."""
    rows = []
    for name, d in summ["designs"].items():
        cells = {}
        for k, v in d["generators"].items():
            if d["kind"] == "contrast":
                cells[k] = {"flag": v["excludes_zero"], "mean": v["mean"]}
            else:
                cells[k] = {"mean": v["mean"], "truth": v["truth"], "bias": v["bias"], "coverage": v["coverage"],
                            "excludes_zero": v["excludes_zero"], "share": v.get("share")}
        rows.append({"design": name, "label": d["label"], "estimand": d["estimand"], "level": d["level"],
                     "status": d["verdict"]["status"], "trustworthy": d["verdict"]["trustworthy"],
                     "reasons": d["verdict"]["reasons"], "worlds": cells})
    return rows


def lab_markdown(summ: dict) -> str:
    reps = summ.get("n_reps", {})
    lines = ["# Canopy identification lab", "",
             "Each design is run on simulated campaigns over the real layout, in worlds where the true canopy "
             "effect is known, and scored against **its own estimand**.",
             f"Replicates per world: {', '.join(f'{k} {n}' for k, n in reps.items())}.", "",
             "## Verdicts", "", "| design | estimand | verdict | why |", "|---|---|---|---|"]
    for name, d in summ["designs"].items():
        v = d["verdict"]
        lines.append(f"| {d['label']} | {ESTIMAND_TEXT.get(d['estimand'], d['estimand'])} | "
                     f"**{STATUS_TEXT.get(v['status'], v['status'])}** | {'; '.join(v['reasons'])} |")
    lines += ["", "## Worlds", ""] + [f"* `{k}`: {t}" for k, t in WORLD_TEXT.items() if k in reps] + [""]
    for name, d in summ["designs"].items():
        lines += [f"## {d['label']}", "", f"Estimand: {ESTIMAND_TEXT.get(d['estimand'], d['estimand'])}.", ""]
        if d["kind"] == "contrast":
            lines += ["| world | mean contrast (°F) | advects | flags advection |", "|---|---|---|---|"]
            for k, v in d["generators"].items():
                lines.append(f"| {k} | {v['mean']:+.3f} | {'yes' if v['advects'] else 'no'} | {v['excludes_zero']:.0%} |")
        else:
            city = any("city_mean" in v for v in d["generators"].values())
            lines += ["| world | estimate (°F) | truth | bias | 95% coverage | excludes 0 |"
                      + (" city extrapolation / truth |" if city else ""),
                      "|---|---|---|---|---|---|" + ("---|" if city else "")]
            for k, v in d["generators"].items():
                lines.append(f"| {k} | {v['mean']:+.3f} ± {v['sd'] or 0:.3f} | {v['truth']:+.3f} | {v['bias']:+.3f} | "
                             f"{v['coverage']:.0%} | {v['excludes_zero']:.0%} |"
                             + (f" {_f(v.get('city_mean'))} / {_f(v.get('city_truth'))} |" if city else ""))
        lines += [""]
    return "\n".join(lines)
