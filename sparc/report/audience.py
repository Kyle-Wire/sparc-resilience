"""
Audience-specific report generator (Phase 19)
=============================================

Produces three flavors of the same SPARC run:

* ``technical`` — full statistics, diagnostics, robustness checks,
  intended for an analyst or peer reviewer.
* ``planner``  — maps + recommendations + cost-effectiveness +
  equity summary, intended for a city planner or program manager.
* ``public``   — a one-page plain-language summary with the hero
  finding and a couple of headline numbers, intended for the public
  or elected officials.

Each report is rendered from a Jinja2 markdown template; the output
can be returned as Markdown text, HTML wrapped in the warm-paper
palette, or a PDF via WeasyPrint when available.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sparc.report.standalone_html import _collect_artifacts, _safe_load_json
from sparc.registry.provenance import load_provenance

AUDIENCES = ("technical", "planner", "public")
FORMATS = ("md", "html", "pdf")


# ------------------------------------------------------------------
# Template strings (Jinja2 markdown)
# ------------------------------------------------------------------

_TECHNICAL_TPL = """# {{ project_name }} — Technical Report
*Audience: analyst · Generated {{ generated_utc }} UTC*

**Domain:** {{ domain or "—" }}
{% if description %}> {{ description }}{% endif %}

---

## 1. Run provenance
{% if provenance %}
- Config hash: `{{ provenance.config_hash[:16] }}…`
- Data sha256: `{{ (provenance.data.sha256 or "")[:16] }}…` ({{ provenance.data.size_bytes }} bytes)
- Git commit: `{{ provenance.git.commit[:8] }}` ({{ "dirty" if provenance.git.dirty else "clean" }})
- Timestamp: {{ provenance.timestamp_utc }}
{% else %}
*No provenance sidecar present (run was produced before Phase 17 freeze).*
{% endif %}

## 2. Predictive performance
{% if model_performance %}
| Model | RMSE | MAE | R² | Spatial R² |
|-------|------|-----|----|------------|
{% for m in model_performance %}| {{ m.model or m.name }} | {{ "%.4f"|format(m.rmse or 0) }} | {{ "%.4f"|format(m.mae or 0) }} | {{ "%.4f"|format(m.r2 or 0) }} | {{ "%.4f"|format(m.spatial_r2 or m.cv_r2 or 0) }} |
{% endfor %}
{% else %}
*No `model_performance.json` artifact found.*
{% endif %}

## 3. Causal estimates
**Estimator:** {{ causal_cfg.estimator or "—" }} · DAG blend weight: {{ causal_cfg.dag_blend_weight or "—" }}
**Actionable variables:** {{ (causal_cfg.actionable_variables or [])|join(", ") or "—" }}

{% if treatments %}
| Variable | Coefficient | 95% CI | Elasticity | E-value | Placebo | RCC | Subset |
|----------|-------------|--------|------------|---------|---------|-----|--------|
{% for t in treatments %}| `{{ t.name }}` | {{ "%.4f"|format(t.coeff or 0) }} | [{{ "%.3f"|format(t.ci_lower or 0) }}, {{ "%.3f"|format(t.ci_upper or 0) }}] | {{ "%.3f"|format(t.elasticity or 0) }} | {{ "%.2f"|format(t.e_value or 0) }} | {{ "✓" if t.placebo else "✗" }} | {{ "✓" if t.rcc else "✗" }} | {{ "✓" if t.subset else "✗" }} |
{% endfor %}
{% endif %}

## 4. Physics constraints
{% if constraints %}{% for c in constraints %}- `{{ c.variable }}` → {{ c.direction }}
{% endfor %}{% else %}*No monotone constraints declared.*{% endif %}

## 5. MC³ structural-equation diagnostics
{% if mc3 %}
- Chains: {{ mc3.n_chains or "?" }} · Draws: {{ mc3.n_draws or "?" }}
- Mean R-hat: {{ "%.3f"|format(mc3.r_hat_mean or 0) }} · Max R-hat: {{ "%.3f"|format(mc3.r_hat_max or 0) }}
- Mean ESS: {{ mc3.ess_mean or "?" }} · Min ESS: {{ mc3.ess_min or "?" }}
{% if mc3.divergent_transitions %}- ⚠ Divergent transitions: {{ mc3.divergent_transitions }}{% endif %}
{% else %}
*No MC³ trace available.*
{% endif %}

## 6. Dose-response
{% if dose_response %}{{ dose_response.keys()|list|length }} curves estimated.
{% else %}*No dose-response curves.*{% endif %}

## 7. Scenarios evaluated
{% if scenarios %}
| Scenario | Δ outcome | Affected pop | Cost ($M) | $/unit |
|----------|-----------|--------------|-----------|--------|
{% for s in (scenarios.summary or scenarios) %}| {{ s.name or s.scenario_id or "—" }} | {{ "%.3f"|format(s.delta_outcome_mean or 0) }} | {{ s.affected_population or "—" }} | {{ "%.2f"|format(s.cost_million or 0) }} | {{ "%.2f"|format(s.cost_per_unit or 0) }} |
{% endfor %}
{% else %}
*No scenario summary present.*
{% endif %}

## 8. DAG discovery summary
{% if dag %}
- Nodes: {{ (dag.nodes or [])|length }} · Edges (validated): {{ (dag.edges_kept or dag.edges or [])|length }}
- PC algorithm orientation rule passes: {{ dag.pc_passes or "—" }}
{% else %}
*No DAG discovery artifact.*{% endif %}
"""

_PLANNER_TPL = """# {{ project_name }} — Planner Briefing
*Audience: city planner · {{ generated_utc }} UTC*

**Domain:** {{ domain or "—" }}
{% if description %}> {{ description }}{% endif %}

## What we found
{% if hero_finding %}**{{ hero_finding }}**{% else %}*Hero finding will appear once causal analysis completes.*{% endif %}

## Recommended interventions (ranked by cost-effectiveness)
{% if scenario_rows %}
| Rank | Intervention | Effect | Affected | Est. cost | $ per unit benefit |
|------|--------------|--------|----------|-----------|--------------------|
{% for s in scenario_rows %}| {{ loop.index }} | {{ s.name }} | {{ "%.2f"|format(s.delta) }} | {{ s.affected }} | ${{ "%.1f"|format(s.cost) }}M | ${{ "%.0f"|format(s.cost_per_unit) }} |
{% endfor %}
{% else %}*Run the scenarios stage to populate this table.*{% endif %}

## Equity considerations
{% if equity %}
- Most-affected demographic: {{ equity.most_affected or "—" }}
- Treatment-effect heterogeneity: {{ equity.heterogeneity or "—" }}
{% else %}
- The pipeline did not surface a CATE / equity breakdown for this run.
- Re-run with `causal.estimate_cate: true` to receive a per-tract effect estimate.
{% endif %}

## Confidence level
{% if confidence_blurb %}{{ confidence_blurb }}
{% else %}Causal estimates passed {{ passed_robustness or 0 }} of {{ total_robustness or 3 }} robustness checks (placebo, refute-with-confounder, subset stability).{% endif %}

## How to use this report
1. Use the ranked interventions above as a shortlist for council discussion.
2. The "$ per unit benefit" column is a *first-order* estimate; pair with a local cost study before budgeting.
3. The hero finding is conditional on the causal DAG approved during the pipeline run — see the technical report for the full structure.

## Provenance
{% if provenance %}Run frozen {{ provenance.timestamp_utc }} · git `{{ provenance.git.commit[:8] }}` · config hash `{{ provenance.config_hash[:8] }}…`{% else %}*Provenance sidecar not present.*{% endif %}
"""

_PUBLIC_TPL = """# {{ project_name }} — At a Glance
*{{ generated_utc[:10] }}*

{% if hero_finding %}## {{ hero_finding }}{% else %}## Analysis in progress{% endif %}

**What this means for you**
{{ public_blurb }}

### The numbers
{% for kv in public_numbers %}- **{{ kv.label }}:** {{ kv.value }}
{% endfor %}

### What we did
We combined satellite, demographic and infrastructure data with a
machine-learning model that respects the laws of physics. We then ran
"what-if" scenarios — for example, *what would happen if we planted
more trees on this block?* — to estimate which actions would actually
move the needle.

### Want more detail?
Ask a city analyst for the full **technical report**, which lists
every assumption, every confidence interval, and every robustness
check we ran.
"""

TEMPLATES = {
    "technical": _TECHNICAL_TPL,
    "planner": _PLANNER_TPL,
    "public": _PUBLIC_TPL,
}


# ------------------------------------------------------------------
# Context builders
# ------------------------------------------------------------------

def _scenario_rows(scenarios: Any) -> list[dict]:
    """Normalize scenario artifact into a ranked list of dicts."""
    rows = []
    src = []
    if isinstance(scenarios, dict):
        src = scenarios.get("summary") or scenarios.get("rows") or []
    elif isinstance(scenarios, list):
        src = scenarios
    for s in src or []:
        if not isinstance(s, dict):
            continue
        delta = s.get("delta_outcome_mean") or s.get("delta") or 0
        cost = s.get("cost_million") or s.get("cost") or 0
        cpu = s.get("cost_per_unit") or (cost * 1e6 / abs(delta) if delta else 0)
        rows.append({
            "name": s.get("name") or s.get("scenario_id") or "scenario",
            "delta": float(delta or 0),
            "affected": s.get("affected_population") or "—",
            "cost": float(cost or 0),
            "cost_per_unit": float(cpu or 0),
        })
    rows.sort(key=lambda r: r["cost_per_unit"] if r["cost_per_unit"] > 0 else 1e18)
    return rows


def _hero_finding(treatments: list[dict], scenarios_rows: list[dict]) -> str | None:
    """Pick a single sentence describing the most actionable result."""
    sig = [t for t in treatments if t.get("ci_lower") and t.get("ci_upper")
           and (t["ci_lower"] * t["ci_upper"] > 0)]  # CI excludes zero
    if scenarios_rows:
        s = scenarios_rows[0]
        sign = "reduce" if s["delta"] < 0 else "increase"
        return f"{s['name']} is the most cost-effective intervention — projected to {sign} the outcome by {abs(s['delta']):.2f} units at roughly ${s['cost_per_unit']:.0f} per unit of benefit."
    if sig:
        t = max(sig, key=lambda x: abs(x.get("elasticity") or 0))
        sign = "raises" if (t.get("coeff") or 0) > 0 else "lowers"
        return f"`{t['name']}` {sign} the outcome with high statistical confidence (elasticity {t.get('elasticity', 0):.2f})."
    return None


def _treatments_from_causal(causal_results: Any) -> list[dict]:
    out = []
    if not isinstance(causal_results, dict):
        return out
    direct = causal_results.get("direct_effects") or {}
    if isinstance(direct, dict):
        for var, eff in direct.items():
            if not isinstance(eff, dict):
                continue
            out.append({
                "name": var,
                "coeff": eff.get("structural_coeff"),
                "ci_lower": eff.get("bootstrap_ci_lower"),
                "ci_upper": eff.get("bootstrap_ci_upper"),
                "elasticity": eff.get("elasticity"),
                "e_value": eff.get("e_value"),
                "placebo": eff.get("placebo_pass"),
                "rcc": eff.get("rcc_pass"),
                "subset": eff.get("data_subset_pass"),
            })
    return out


def _model_perf_rows(perf: Any) -> list[dict]:
    if not perf:
        return []
    if isinstance(perf, dict):
        if "models" in perf and isinstance(perf["models"], list):
            return perf["models"]
        return [{"model": k, **(v if isinstance(v, dict) else {"value": v})}
                for k, v in perf.items()]
    if isinstance(perf, list):
        return perf
    return []


def build_context(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Compose the Jinja2 context shared by all three audience templates."""
    artifacts = _collect_artifacts(run_dir)
    proj = config.get("project", {}) if config else {}
    causal_cfg = config.get("causal", {}) if config else {}
    physics = config.get("physics", {}) if config else {}

    treatments = _treatments_from_causal(artifacts.get("scenario_coefficients"))
    scenario_rows = _scenario_rows(artifacts.get("scenarios"))
    model_performance = _model_perf_rows(artifacts.get("model_performance"))
    constraints = [
        {"variable": k, "direction": "increasing" if v > 0 else "decreasing" if v < 0 else "none"}
        for k, v in (physics.get("monotone_constraints") or {}).items()
    ]
    hero = _hero_finding(treatments, scenario_rows)

    # Robustness tally
    passed = 0
    total = 0
    for t in treatments:
        for k in ("placebo", "rcc", "subset"):
            if t.get(k) is not None:
                total += 1
                if t[k]:
                    passed += 1

    public_numbers = []
    if scenario_rows:
        s = scenario_rows[0]
        public_numbers.append({"label": "Top intervention", "value": s["name"]})
        public_numbers.append({"label": "Projected change", "value": f"{s['delta']:+.2f}"})
        public_numbers.append({"label": "Estimated cost", "value": f"${s['cost']:.1f} M"})
    if model_performance:
        first = model_performance[0]
        r2 = first.get("spatial_r2") or first.get("r2") or first.get("cv_r2")
        if r2:
            public_numbers.append({"label": "Model accuracy (R²)", "value": f"{r2:.2f}"})

    public_blurb = (
        "Our analysis identified the actions most likely to improve outcomes "
        "for your community, ranked from cheapest to most expensive per unit "
        "of benefit. The top recommendation appears above."
    )
    if not hero:
        public_blurb = (
            "We are still validating the causal estimates for this run — once "
            "the analyst signs off on the diagnostics, a plain-language "
            "summary of the recommended actions will appear here."
        )

    confidence_blurb = None
    if total:
        confidence_blurb = (
            f"Causal estimates passed **{passed} of {total}** robustness "
            f"checks across all treatments (placebo, refute-with-confounder, "
            f"subset stability)."
        )

    return {
        "project_name": proj.get("name") or run_dir.name,
        "domain": proj.get("domain"),
        "description": proj.get("description"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": load_provenance(run_dir),
        "model_performance": model_performance,
        "causal_cfg": causal_cfg,
        "treatments": treatments,
        "constraints": constraints,
        "mc3": artifacts.get("mc3") or {},
        "dose_response": artifacts.get("dose_response") or {},
        "scenarios": artifacts.get("scenarios"),
        "scenario_rows": scenario_rows,
        "dag": artifacts.get("dag"),
        "hero_finding": hero,
        "passed_robustness": passed,
        "total_robustness": total,
        "equity": (artifacts.get("scenario_coefficients") or {}).get("equity") if isinstance(artifacts.get("scenario_coefficients"), dict) else None,
        "public_numbers": public_numbers,
        "public_blurb": public_blurb,
        "confidence_blurb": confidence_blurb,
    }


# ------------------------------------------------------------------
# Renderers
# ------------------------------------------------------------------

def render_markdown(audience: str, context: dict[str, Any]) -> str:
    if audience not in TEMPLATES:
        raise ValueError(f"Unknown audience '{audience}' (expected one of {AUDIENCES})")
    from jinja2 import Environment

    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    return env.from_string(TEMPLATES[audience]).render(**context)


_HTML_WRAPPER = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
:root {{ --paper:#FAF7F0; --ink:#2A1F1A; --crimson:#C0392B; --purple:#5B3A8C;
        --amber:#D4A017; --muted:#8B7B6E; --line:#E5DCD0; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
     font:15px/1.55 'Iowan Old Style',Georgia,serif;padding:32px;max-width:880px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 6px;color:var(--ink)}}
h2{{font-size:18px;margin-top:28px;color:var(--purple);border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:14px;margin-top:18px;color:var(--ink);text-transform:uppercase;letter-spacing:.05em}}
em{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 18px}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line)}}
th{{background:#fff;color:var(--muted);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.05em}}
code{{background:rgba(91,58,140,.08);padding:1px 6px;border-radius:3px;font-size:12.5px;color:var(--purple)}}
blockquote{{border-left:3px solid var(--crimson);padding:4px 12px;margin:0 0 16px;color:var(--ink);background:#fff}}
.public h2{{font-size:22px;color:var(--crimson);border:0;padding:0}}
</style></head>
<body class="{audience_class}">
{body}
</body></html>"""


def render_html(audience: str, context: dict[str, Any]) -> str:
    md = render_markdown(audience, context)
    try:
        import markdown as md_lib
        body = md_lib.markdown(md, extensions=["tables"])
    except ImportError:
        # Minimal fallback: escape + wrap in <pre>
        body = f"<pre>{_html.escape(md)}</pre>"
    title = f"{context.get('project_name', 'SPARC')} — {audience.title()} Report"
    return _HTML_WRAPPER.format(title=_html.escape(title), audience_class=audience, body=body)


def render_pdf(audience: str, context: dict[str, Any]) -> bytes:
    html_str = render_html(audience, context)
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise RuntimeError("weasyprint is not installed; cannot render PDF") from e
    return HTML(string=html_str).write_pdf()


def generate_audience_report(
    run_dir: str | Path,
    config: dict[str, Any] | None,
    audience: str,
    fmt: str = "md",
) -> str | bytes:
    """One-stop generator. Returns text for md/html, bytes for pdf."""
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}")
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    ctx = build_context(Path(run_dir), config or {})
    if fmt == "md":
        return render_markdown(audience, ctx)
    if fmt == "html":
        return render_html(audience, ctx)
    return render_pdf(audience, ctx)
