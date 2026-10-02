"""Scenario templates (SPEC §7.4, api.md §7.4): parameterised forms that emit ``ScenarioDoc`` s.

=========================  ===================================================================================
Template                   Document
=========================  ===================================================================================
``cool_roofs``             albedo ``set 0.35`` where ``layer:lc_built ≥ 0.5``
``shade_hottest``          canopy ``add 20`` within the top X% of observed temperature
``fill_plantable``         canopy ``fill_headroom`` (X% of plantable space)
``depave``                 impervious ``ceiling 60`` where impervious ≥ 80
``footprint_priority``     canopy ``add 10`` on the top-k cells by ``response:<canopy>:footprint_effect_per_unit``
                           (most cooling per unit first)
``around_sites``           a lever ``add`` within ``radius_m`` of uploaded points (lon/lat)
``configured_package``     a ``joint_scenarios`` package of the config as city-wide ``add`` edits
=========================  ===================================================================================

Each template lists what it ``requires`` (``layers``, ``canopy_role``, ``impervious_role``, ``albedo_role``,
``crs``); a run that lacks something gets ``422 template_unavailable`` with ``detail.missing``.
"""

from __future__ import annotations

from typing import Any, Callable

from sparc.studio.errors import ApiError

__all__ = ["TEMPLATES", "listing", "build", "requirements_missing"]


def _num(name: str, default: float, lo: float | None = None, hi: float | None = None, desc: str = "") -> dict:
    s: dict[str, Any] = {"type": "number", "default": default, "title": name}
    if lo is not None:
        s["minimum"] = lo
    if hi is not None:
        s["maximum"] = hi
    if desc:
        s["description"] = desc
    return s


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


def _role(ctx, role: str) -> str | None:
    return ((ctx.cfg_raw.get("physics") or {}).get("roles") or {}).get(role)


def _has_layers(ctx) -> bool:
    from sparc.studio.runs import layers as L

    return ctx.data is not None and L.people_layers(ctx) is not None


def _p(params: dict, key: str, default):
    v = params.get(key)
    return default if v is None else v


def _cool_roofs(ctx, p: dict) -> dict:
    alb = _role(ctx, "albedo")
    value, built = float(_p(p, "albedo", 0.35)), float(_p(p, "built_min", 0.5))
    return {"name": f"Cool roofs on built-up cells (albedo {value:g})", "tags": ["template:cool_roofs"],
            "edits": [{"lever": alb, "mode": "set", "amount": value, "label": "cool roofs",
                       "where": {"kind": "filter", "column": "layer:lc_built", "op": ">=", "value": built}}]}


def _shade_hottest(ctx, p: dict) -> dict:
    can = _role(ctx, "canopy")
    frac, add = float(_p(p, "frac", 0.1)), float(_p(p, "add", 20.0))
    return {"name": f"Shade the hottest {frac:.0%} (+{add:g} canopy)", "tags": ["template:shade_hottest"],
            "edits": [{"lever": can, "mode": "add", "amount": add, "label": "shade the hottest cells",
                       "where": {"kind": "top", "column": "pred:target", "frac": frac, "direction": "highest"}}]}


def _fill_plantable(ctx, p: dict) -> dict:
    can = _role(ctx, "canopy")
    share = float(_p(p, "share", 0.5))
    edit: dict[str, Any] = {"lever": can, "mode": "fill_headroom", "amount": share, "label": "fill plantable space",
                            "where": {"kind": "all"}}
    if p.get("paved_share") is not None:
        edit["paved_share"] = float(p["paved_share"])
    return {"name": f"Fill {share:.0%} of plantable space", "tags": ["template:fill_plantable"], "edits": [edit]}


def _depave(ctx, p: dict) -> dict:
    imp = _role(ctx, "impervious")
    cap, above = float(_p(p, "cap", 60.0)), float(_p(p, "min_impervious", 80.0))
    return {"name": f"Depave the most paved blocks (cap {cap:g} where ≥ {above:g})", "tags": ["template:depave"],
            "edits": [{"lever": imp, "mode": "ceiling", "amount": cap, "label": "depave",
                       "where": {"kind": "filter", "column": f"predictor:{imp}", "op": ">=", "value": above}}]}


def _footprint(ctx, p: dict) -> dict:
    can = _role(ctx, "canopy")
    add = float(_p(p, "add", 10.0))
    top: dict[str, Any] = {"kind": "top", "column": f"response:{can}:footprint_effect_per_unit",
                           "direction": "lowest"}
    if p.get("k") is not None:
        top["k"] = int(p["k"])
    else:
        top["frac"] = float(_p(p, "frac", 0.1))
    return {"name": f"Prioritise by footprint (+{add:g} canopy)", "tags": ["template:footprint_priority"],
            "edits": [{"lever": can, "mode": "add", "amount": add, "label": "highest cooling footprint",
                       "where": top}]}


def _around_sites(ctx, p: dict) -> dict:
    sites = p.get("sites") or []
    if not sites:
        raise ApiError("validation", "around_sites needs sites: [[lon, lat], …]",
                       detail={"errors": [{"path": "params.sites", "message": "required", "code": "missing"}]})
    lever = str(p.get("lever") or _role(ctx, "canopy"))
    radius, add = float(_p(p, "radius_m", 200.0)), float(_p(p, "amount", 10.0))
    circles = [{"kind": "circle", "crs": "EPSG:4326", "center": [float(s[0]), float(s[1])], "radius_m": radius}
               for s in sites]
    where = circles[0] if len(circles) == 1 else {"op": "or", "args": circles}
    return {"name": f"Around {len(sites)} site(s) ({lever} {add:+g} within {radius:g} m)".replace("+-", "−"),
            "tags": ["template:around_sites"],
            "edits": [{"lever": lever, "mode": "add", "amount": add, "label": "around sites", "where": where}]}


def _configured_package(ctx, p: dict) -> dict:
    joint = (ctx.cfg_raw.get("joint_scenarios") or [])
    if not joint:
        raise ApiError("template_unavailable", "this run's config has no joint_scenarios package",
                       detail={"missing": ["joint_scenarios"]})
    name = p.get("name") or joint[0].get("name")
    pkg = next((j for j in joint if j.get("name") == name), None)
    if pkg is None:
        raise ApiError("validation", f"no configured package {name!r}",
                       detail={"errors": [{"path": "params.name", "message": "unknown package", "code": "unknown"}]})
    edits = []
    for iv in pkg.get("interventions") or []:
        sign = -1.0 if str(iv.get("direction", "increase")).lower() == "decrease" else 1.0
        edits.append({"lever": iv["variable"], "mode": "add", "amount": sign * float(iv.get("increment", 0.0)),
                      "where": {"kind": "all"}})
    return {"name": str(name), "tags": ["template:configured_package", "configured"], "edits": edits}


TEMPLATES: dict[str, dict] = {
    "cool_roofs": {"label": "Cool roofs on built-up cells", "desc": "Set albedo to 0.35 where built-up land covers "
                   "at least half the cell.", "requires": ["layers", "albedo_role"], "build": _cool_roofs,
                   "params_schema": _schema({"albedo": _num("albedo", 0.35, 0, 1), "built_min": _num(
                       "built-up share", 0.5, 0, 1)})},
    "shade_hottest": {"label": "Shade the hottest X%", "desc": "Add 20 points of canopy within the hottest share of "
                      "cells (observed temperature).", "requires": ["canopy_role"], "build": _shade_hottest,
                      "params_schema": _schema({"frac": _num("share of cells", 0.1, 0, 1),
                                                "add": _num("canopy added (pp)", 20.0, 0, 100)})},
    "fill_plantable": {"label": "Fill X% of plantable space", "desc": "Plant a share of each cell's plantable "
                       "headroom (open land plus a share of paving).", "requires": ["canopy_role", "layers"],
                       "build": _fill_plantable,
                       "params_schema": _schema({"share": _num("share of headroom", 0.5, 0, 1),
                                                 "paved_share": _num("plantable share of paving", 0.2, 0, 1)})},
    "depave": {"label": "Depave the most paved blocks", "desc": "Cap impervious cover at 60% where it is at least "
               "80%.", "requires": ["impervious_role"], "build": _depave,
               "params_schema": _schema({"cap": _num("cap (pp)", 60.0, 0, 100),
                                         "min_impervious": _num("applies where impervious ≥", 80.0, 0, 100)})},
    "footprint_priority": {"label": "Prioritise by footprint", "desc": "Add canopy where each point cools the "
                           "neighbourhood most (S4 footprint effect).", "requires": ["canopy_role"],
                           "build": _footprint,
                           "params_schema": _schema({"k": {"type": "integer", "minimum": 1, "title": "cells"},
                                                     "frac": _num("share of cells", 0.1, 0, 1),
                                                     "add": _num("canopy added (pp)", 10.0, 0, 100)})},
    "around_sites": {"label": "Around sites", "desc": "Edit a lever within a radius of uploaded points.",
                     "requires": ["crs"], "build": _around_sites,
                     "params_schema": _schema({"sites": {"type": "array", "items": {"type": "array", "items": {
                         "type": "number"}, "minItems": 2, "maxItems": 2}, "title": "sites [lon, lat]"},
                         "lever": {"type": "string", "title": "lever"}, "radius_m": _num("radius (m)", 200.0, 0),
                         "amount": _num("amount", 10.0)}, ["sites"])},
    "configured_package": {"label": "Configured package", "desc": "A joint scenario package from the run's "
                           "config, as editable city-wide edits.", "requires": [], "build": _configured_package,
                           "params_schema": _schema({"name": {"type": "string", "title": "package"}})},
}


def listing() -> list[dict]:
    return [{"id": k, "label": t["label"], "desc": t["desc"], "params_schema": t["params_schema"],
             "requires": list(t["requires"])} for k, t in TEMPLATES.items()]


def requirements_missing(ctx, requires: list[str]) -> list[str]:
    missing = []
    for r in requires:
        if r == "layers" and not _has_layers(ctx):
            missing.append("layers")
        elif r.endswith("_role") and not _role(ctx, r[:-5]):
            missing.append(r)
        elif r == "crs" and not (ctx.grid is not None and ctx.grid.crs):
            missing.append("crs")
    return missing


def build(ctx, template: str, params: dict) -> dict:
    """The ScenarioDoc of ``template`` on the run of ``ctx`` (anchored to it)."""
    t = TEMPLATES.get(template)
    if t is None:
        raise ApiError("validation", f"unknown template {template!r}",
                       detail={"errors": [{"path": "template", "message": f"one of {', '.join(TEMPLATES)}",
                                           "code": "unknown_template"}]})
    missing = requirements_missing(ctx, t["requires"])
    if missing:
        raise ApiError("template_unavailable", f"{t['label']} needs {', '.join(missing)} on this run",
                       detail={"missing": missing})
    fn: Callable = t["build"]
    doc = fn(ctx, dict(params or {}))
    doc["anchor_run_id"] = ctx.run_id
    return doc
