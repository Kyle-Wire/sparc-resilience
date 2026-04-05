"""
SPARC Pipeline PDF Report Generator.

Renders a complete analysis report as PDF using Jinja2 templates + WeasyPrint.
Falls back to HTML-only if weasyprint is not installed.
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _encode_image(path: str | Path) -> str:
    """Return a base64 data-URI for an image file."""
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml"}.get(suffix, "image/png")
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _collect_plots(output_dir: Path) -> dict[str, list[dict]]:
    """Walk stage directories and collect plot images."""
    plots: dict[str, list[dict]] = {}
    for stage_dir in sorted(output_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        stage_plots = []
        for img in sorted(stage_dir.rglob("*.png")):
            stage_plots.append({
                "name": img.stem.replace("_", " ").title(),
                "data_uri": _encode_image(img),
                "path": str(img),
            })
        if stage_plots:
            plots[stage_dir.name] = stage_plots
    return plots


def generate_report_html(
    config: dict[str, Any],
    data_summary: dict[str, Any] | None = None,
    causal_results: dict[str, Any] | None = None,
    scenario_summary: list[dict] | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Generate the report as an HTML string.

    Parameters
    ----------
    config : dict
        The project YAML config (parsed).
    data_summary : dict, optional
        Data summary from state.data_summary.
    causal_results : dict, optional
        Causal results JSON (from causal_validation_summary or scenario_coefficients).
    scenario_summary : list[dict], optional
        Rows from the scenario summary CSV.
    output_dir : Path, optional
        Pipeline output directory to scan for plots.

    Returns
    -------
    str
        Complete HTML document.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    # Collect plots
    plots: dict[str, list[dict]] = {}
    if output_dir:
        output_path = Path(output_dir)
        if output_path.exists():
            plots = _collect_plots(output_path)

    # Build context
    project = config.get("project", {})
    causal_cfg = config.get("causal", {})
    physics = config.get("physics", {})
    pipeline = config.get("pipeline", {})
    predictors = config.get("predictors", {})
    if isinstance(predictors, dict):
        predictor_list = predictors.get("base_model", [])
    elif isinstance(predictors, list):
        predictor_list = predictors
    else:
        predictor_list = []

    # Treatments and effects
    treatments = []
    if causal_results and "direct_effects" in causal_results:
        for var, effect in causal_results["direct_effects"].items():
            treatments.append({
                "name": var,
                "coeff": effect.get("structural_coeff"),
                "ci_lower": effect.get("bootstrap_ci_lower"),
                "ci_upper": effect.get("bootstrap_ci_upper"),
                "elasticity": effect.get("elasticity"),
                "importance": effect.get("relative_importance"),
                "e_value": effect.get("e_value"),
                "placebo": effect.get("placebo_pass"),
                "rcc": effect.get("rcc_pass"),
                "subset": effect.get("data_subset_pass"),
            })

    # Monotone constraints
    constraints = []
    mc = physics.get("monotone_constraints", {})
    for k, v in mc.items():
        constraints.append({
            "variable": k,
            "direction": "increasing" if v > 0 else "decreasing" if v < 0 else "none",
        })

    context = {
        "project_name": project.get("name", "Untitled"),
        "domain": project.get("domain", ""),
        "description": project.get("description", ""),
        "data_summary": data_summary,
        "predictors": predictor_list,
        "causal_estimator": causal_cfg.get("estimator", "N/A"),
        "actionable_variables": causal_cfg.get("actionable_variables", []),
        "treatments": treatments,
        "constraints": constraints,
        "scenario_summary": scenario_summary or [],
        "pipeline": pipeline,
        "plots": plots,
    }

    return template.render(**context)


def generate_report_pdf(
    config: dict[str, Any],
    data_summary: dict[str, Any] | None = None,
    causal_results: dict[str, Any] | None = None,
    scenario_summary: list[dict] | None = None,
    output_dir: str | Path | None = None,
    dest_path: str | Path | None = None,
) -> bytes | Path:
    """Generate the report as PDF.

    If *dest_path* is given, writes to that file and returns the Path.
    Otherwise returns PDF bytes.
    """
    html_str = generate_report_html(
        config=config,
        data_summary=data_summary,
        causal_results=causal_results,
        scenario_summary=scenario_summary,
        output_dir=output_dir,
    )

    try:
        from weasyprint import HTML
    except ImportError:
        raise RuntimeError(
            "weasyprint is not installed. Install with: pip install weasyprint"
        )

    pdf_bytes = HTML(string=html_str).write_pdf()

    if dest_path:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pdf_bytes)
        return dest

    return pdf_bytes
