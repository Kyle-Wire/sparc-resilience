"""``CoreConfigModel``: the pydantic model of a core config's ``core:`` block (SPEC §9.4, api.md §5.3).

It covers :data:`sparc.core.config.DEFAULTS` plus the keys core reads that
DEFAULTS leaves out: ``planner.*``, ``report.*``, ``response.clip_to_support``,
the physics extras (``albedo_map``, ``shade_form``, ``priors``,
``num_threads`` …), the extended influence keys, the ``dag_audit`` dict form
and ``output.scenario_detail``.  Every section uses ``extra="allow"``: unknown
keys pass through (``validate_deep`` reports them as info).  Defaults equal
DEFAULTS (a test diffs the two).

Each field carries an ``x-ui`` hint in its JSON schema - ``{group, advanced,
unit, help, enum_labels}`` - which the setup wizard and the config form use;
``group`` names the setup step (``data``, ``levers``, ``physics``,
``scenarios``, ``analysis``, ``about``) or ``run`` for run-level settings.
``GET /api/config/schema`` serves :func:`config_json_schema`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CoreConfigModel", "config_json_schema", "defaults_diff", "model_paths", "ROLE_NAMES", "SECTIONS", "x_ui"]

#: physics roles (``physics.roles``), in the order of the setup step
ROLE_NAMES = ("albedo", "canopy", "impervious", "ndvi", "elevation", "water_distance")

#: top-level keys ``PATCH /config/sections/{section}`` accepts (api.md §5.3)
SECTIONS = ("name", "data", "predictors", "encodings", "qa", "actionable", "coupling", "mediators", "physics",
            "influence", "cv", "models", "stacker", "response", "scenarios", "joint_scenarios", "causal", "climate",
            "optimize", "planner", "report", "output")

COORD_UNITS = ("m", "metre", "meter", "us_survey_foot", "us_ft", "ftus", "ft", "foot")


def x_ui(group: str, help: str = "", *, advanced: bool = False, unit: str | None = None,
         enum_labels: dict[str, str] | None = None) -> dict:
    """The ``json_schema_extra`` of a field: ``{"x-ui": {group, advanced, unit, help, enum_labels}}``."""
    return {"x-ui": {"group": group, "advanced": bool(advanced), "unit": unit, "help": help,
                     "enum_labels": enum_labels}}


def F(default: Any = None, group: str = "data", help: str = "", *, advanced: bool = False, unit: str | None = None,
      enum_labels: dict[str, str] | None = None, factory=None) -> Any:
    """A ``Field`` with a description and its ``x-ui`` hint."""
    extra = x_ui(group, help, advanced=advanced, unit=unit, enum_labels=enum_labels)
    if factory is not None:
        return Field(default_factory=factory, description=help or None, json_schema_extra=extra)
    return Field(default, description=help or None, json_schema_extra=extra)


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class DataJoin(_Section):
    path: str = F(None, "data", "Table merged by id (CSV or parquet; relative to the project folder)")
    key: str | None = F(None, "data", "Id column of the main table (default: data.id)")
    right_key: str | None = F(None, "data", "Id column of the joined table (default: id)")


class DataSection(_Section):
    path: str | None = F(None, "data", "Point table (CSV) relative to the project folder")
    target: str | None = F(None, "data", "Temperature column")
    target_units: str = F("degF", "data", "Units of the target", enum_labels={"degF": "°F", "degC": "°C", "K": "K"})
    id: str | None = F(None, "data", "Unique id column (rows are numbered when unset)")
    x: str = F("x", "data", "X coordinate column")
    y: str = F("y", "data", "Y coordinate column")
    coord_unit: Literal["m", "metre", "meter", "us_survey_foot", "us_ft", "ftus", "ft", "foot"] = F(
        "m", "data", "Units of x/y", enum_labels={"m": "metres", "us_survey_foot": "US survey feet",
                                                   "ft": "international feet"})
    crs: str | None = F(None, "data", "EPSG code of x/y (needed for open data, CMIP6 site, GeoTIFF, people objective)")
    reproject_to: str | None = F(None, "data", "Reproject coordinates to this CRS (metres)", advanced=True)
    background: float | str = F("median", "data", "ΔT reference: 'median', a number, or a column name")
    subsample: int | None = F(None, "data", "Keep only this many points around the centre (smoke tests)",
                              advanced=True)
    cell_m: float | None = F(None, "data", "Grid cell size (default: inferred from the lattice)", advanced=True,
                             unit="m")
    coarse_m: float | None = F(None, "data", "Aggregate onto cells of this size over the full extent",
                               advanced=True, unit="m")
    join: list[DataJoin] = F(group="data", help="Extra tables merged by id", factory=list)
    zone: str | None = F(None, "data", "Neighbourhood / zone code column (reporting only)")


class EncodingsSection(_Section):
    categorical: list[str] = F(group="levers", help="Predictors one-hot encoded", factory=list)
    circular_degrees: list[str] = F(group="levers", help="Predictors in degrees (sin/cos encoded)", factory=list)


class QaSection(_Section):
    clip: dict[str, list[float]] = F(group="levers", help="Clip predictor values to [lo, hi]", factory=dict)


# ---------------------------------------------------------------------------
# levers
# ---------------------------------------------------------------------------

class ActionableSpec(_Section):
    min: float | None = F(None, "levers", "Lowest value an edit may reach")
    max: float | None = F(None, "levers", "Highest value an edit may reach")
    unit: str | None = F(None, "levers", "Unit of the lever (e.g. pp, reflectance)")
    doses: list[float] = F(group="levers", help="Dose ladder of the S4 sweep (magnitudes, including 0)",
                           factory=lambda: [0.0, 5.0, 10.0, 20.0, 30.0])
    cost_per_unit: float = F(1.0, "levers", "Cost of one unit of dose on one cell")
    direction: Literal["increase", "decrease"] = F("increase", "levers", "Which way the lever moves",
                                                   enum_labels={"increase": "increase", "decrease": "decrease"})


class CouplingSpec(_Section):
    sum: list[str] = F(group="levers", help="Columns whose sum is capped", factory=list)
    max: float = F(100.0, "levers", "Cap on the sum")


class MediatorSpec(_Section):
    parents: list[str] = F(group="levers", help="Levers that move this predictor", factory=list)
    context: list[str] = F(group="levers", help="Other inputs of the mediator model", factory=list)
    monotone: dict[str, int] = F(group="levers", help="Sign constraints per parent (+1 / −1)", factory=dict)


# ---------------------------------------------------------------------------
# physics, influence, cv, models, stacker, response
# ---------------------------------------------------------------------------

class RolesSection(_Section):
    albedo: str | None = F(None, "physics", "Broadband albedo predictor")
    canopy: str | None = F(None, "physics", "Tree canopy (%) predictor")
    impervious: str | None = F(None, "physics", "Impervious surface (%) predictor")
    ndvi: str | None = F(None, "physics", "NDVI predictor")
    elevation: str | None = F(None, "physics", "Elevation predictor", unit="m")
    water_distance: str | None = F(None, "physics", "Distance-to-water predictor", unit="m")


class PhysicsSection(_Section):
    enabled: bool = F(True, "physics", "Fit the physics base model")
    window: Literal["day", "night"] = F("day", "physics", "Measurement window")
    sw_down: float = F(800.0, "physics", "Downward shortwave radiation", unit="W/m²")
    lw_net: float = F(-100.0, "physics", "Net longwave radiation", unit="W/m²")
    forcing: str | None = F(None, "physics", "Campaign forcing JSON (sets window, sw_down, lw_net, wind)")
    roles: RolesSection = F(group="physics", help="Which predictor plays each physical role", factory=RolesSection)
    wind: list[float] | None = F(None, "physics", "Wind vector [u, v] (m/s, towards east/north)", unit="m/s")
    tau_s: float = F(1800.0, "physics", "Advection time scale", advanced=True, unit="s")
    L_max_m: float = F(2000.0, "physics", "Largest diffusion length", advanced=True, unit="m")
    v_max_m: float = F(1000.0, "physics", "Largest advection displacement", advanced=True, unit="m")
    fit_advection: bool | Literal["auto"] = F("auto", "physics", "Fit advection (auto: when a wind is set)",
                                              advanced=True)
    select_advection: bool = F(True, "physics", "Keep advection only if it beats v = 0 out of fold", advanced=True)
    max_iter: int = F(60, "physics", "L-BFGS iterations", advanced=True)
    # extras core reads but DEFAULTS leaves out (sparc.core.physics.PHYSICS_DEFAULTS)
    albedo_map: dict[str, Any] | None = F(None, "physics", "Linear rescale of a non-broadband albedo layer, "
                                          "e.g. {from: auto, to: [0.08, 0.25]}", advanced=True)
    shade_form: Literal["saturating", "sigmoid"] = F("saturating", "physics", "Shape of canopy shading",
                                                     advanced=True, enum_labels={"saturating": "saturating",
                                                                                 "sigmoid": "S-shaped"})
    priors: dict[str, list[float]] | None = F(None, "physics", "Source-coefficient priors {name: [mean, sd]}",
                                              advanced=True)
    start_iter: int = F(0, "physics", "Iterations per multi-start probe (0: automatic)", advanced=True)
    num_threads: int | None = F(1, "physics", "Torch threads during the physics fit", advanced=True)
    prior_weight: float = F(1e-2, "physics", "Prior penalty weight", advanced=True)
    v_penalty: float = F(1e-2, "physics", "Advection penalty weight", advanced=True)


class InfluenceSection(_Section):
    max_lag_m: float = F(2000.0, "analysis", "Longest correlogram lag", advanced=True, unit="m")
    n_rings: int = F(10, "analysis", "Ring-profile rings", advanced=True)
    n_perm: int = F(19, "analysis", "Permutations of the significance test", advanced=True)
    scales: list[float] = F(group="analysis", help="Focal-feature scales (× influence range)", advanced=True,
                            factory=lambda: [0.5, 1.0, 2.0])
    mass: float = F(0.9, "analysis", "Kernel mass that defines the range", advanced=True)
    # extended keys (sparc.core.influence.DEFAULTS)
    n_bins: int = F(20, "analysis", "Radial bins", advanced=True)
    n_dirs: int = F(4, "analysis", "Directional sectors", advanced=True)
    n_boot: int = F(20, "analysis", "Bootstrap replicates of the ranges", advanced=True)
    ridge: float = F(1e-3, "analysis", "Ridge of the ring-profile regression", advanced=True)
    min_range_cells: float = F(2.0, "analysis", "Smallest range, in grid cells", advanced=True)
    families: list[str] = F(group="analysis", help="Kernel families tried", advanced=True,
                            factory=lambda: ["green", "gauss"])
    transform: str = F("nscore", "analysis", "Target transform of the ring profiles", advanced=True)
    n_scales: int = F(25, "analysis", "Candidate scales", advanced=True)
    nuisance_scale_m: float | None = F(None, "analysis", "Nuisance smooth scale", advanced=True, unit="m")
    z_thresh: float = F(2.0, "analysis", "Significance threshold (z)", advanced=True)
    jack_side: int = F(5, "analysis", "Jackknife blocks per side", advanced=True)


class DistanceCurve(_Section):
    enabled: bool = F(False, "analysis", "Skill-vs-distance CV curve (reporting only)", advanced=True)
    block_m: list[float] = F(group="analysis", help="Block sizes of the curve (0 = random points)", advanced=True,
                             factory=lambda: [0.0, 500.0, 1000.0], unit="m")


class CvSection(_Section):
    n_folds: int = F(5, "analysis", "Spatial-block CV folds", advanced=True)
    block_m: float | Literal["auto"] = F("auto", "analysis", "Block size (auto: the S1 range)", advanced=True,
                                         unit="m")
    buffer_m: float | Literal["auto"] = F("auto", "analysis", "Buffer between train and test", advanced=True,
                                          unit="m")
    seed: int = F(42, "analysis", "Fold seed", advanced=True)
    distance_curve: DistanceCurve = F(group="analysis", help="Skill-vs-distance curve", advanced=True,
                                      factory=DistanceCurve)
    baselines: bool | list[str] = F(True, "analysis", "Reference baselines on the same folds (true = all)",
                                    advanced=True)


class ModelsSection(_Section):
    ols: bool = F(True, "analysis", "Ordinary least squares", advanced=True)
    mgwr: bool = F(True, "analysis", "Multiscale GWR", advanced=True)
    gwrf: bool = F(True, "analysis", "Geographically weighted random forest", advanced=True)
    gam: bool = F(True, "analysis", "Generalised additive model", advanced=True)
    physics: bool = F(True, "analysis", "Physics model", advanced=True)
    spatial_plus: bool | list[str] = F(group="analysis", help="Spatial+ for these models (mgwr, gam)",
                                       advanced=True, factory=list)


class StackerSection(_Section):
    hidden: int = F(64, "analysis", "Hidden units", advanced=True)
    physics_mode: Literal["feature", "backbone"] = F("feature", "analysis", "How physics enters the stacker",
                                                     advanced=True)
    epochs: int = F(400, "analysis", "Training epochs", advanced=True)
    lr: float = F(3e-3, "analysis", "Learning rate", advanced=True)
    weight_decay: float = F(1e-4, "analysis", "Weight decay", advanced=True)
    lambda_pde: float = F(1.0, "analysis", "PDE penalty weight", advanced=True)
    tune_lambda: list[float] = F(group="analysis", help="PDE weights tried", advanced=True,
                                 factory=lambda: [0.0, 0.01, 0.1, 1.0])
    use_features: bool = F(True, "analysis", "Raw and focal features as MLP inputs", advanced=True)
    val_fraction: float = F(0.25, "analysis", "Inner held-out share", advanced=True)
    eval_every: int = F(10, "analysis", "Epochs between evaluations", advanced=True)
    patience: int = F(10, "analysis", "Evaluations without improvement before stopping", advanced=True)
    min_gain: float = F(0.01, "analysis", "Minimum relative gain to keep the residual", advanced=True)
    allow_residual_off: bool = F(True, "analysis", "Also score the convex base alone", advanced=True)
    coverage: float = F(0.9, "analysis", "Target coverage of the conformal interval", advanced=True)
    seed: int = F(0, "analysis", "Stacker seed", advanced=True)


class ResponseSection(_Section):
    min_valid_doses: int = F(4, "analysis", "Doses a cell needs for its response curve", advanced=True)
    ds_grid: int = F(32, "analysis", "Grid of the saturation fit", advanced=True)
    clip_to_support: bool = F(True, "analysis", "Clamp edits to the observed range of the lever", advanced=True)


# ---------------------------------------------------------------------------
# scenarios, causal, climate, optimize, planner, report, output
# ---------------------------------------------------------------------------

class ScenarioLadder(_Section):
    name: str = F("", "scenarios", "Ladder name (doses are appended, e.g. 'Canopy Increase +10')")
    variable: str = F("", "scenarios", "Actionable lever")
    direction: Literal["increase", "decrease"] = F("increase", "scenarios", "Direction of the edits")
    increments: list[float] = F(group="scenarios", help="Dose of each scenario", factory=list)


class JointIntervention(_Section):
    variable: str = F("", "scenarios", "Actionable lever")
    direction: Literal["increase", "decrease"] = F("increase", "scenarios", "Direction of the edit")
    increment: float = F(0.0, "scenarios", "Dose")


class JointScenario(_Section):
    name: str = F("", "scenarios", "Package name")
    interventions: list[JointIntervention] = F(group="scenarios", help="Edits applied together", factory=list)


class DagAudit(_Section):
    edges: list[Any] | None = F(None, "analysis", "Expert edges [{parent, child}] (default: from the config)",
                                advanced=True)
    n_boot: int = F(5, "analysis", "Bootstrap resamples", advanced=True)
    n_iter: int = F(2000, "analysis", "MC³ iterations", advanced=True)


class CausalSection(_Section):
    enabled: bool = F(True, "analysis", "Run the causal validation (S6)")
    treatments: list[str] = F(group="analysis", help="Levers validated causally", factory=list)
    confounders: dict[str, list[str]] = F(group="analysis", help="Adjustment set per treatment", factory=dict)
    exclude_controls: dict[str, list[str]] = F(group="analysis", help="Descendants never adjusted for",
                                               factory=dict)
    contrast: dict[str, float] = F(group="analysis", help="Contrast per treatment", factory=dict)
    spatial_basis_scale_m: float | Literal["auto"] = F("auto", "analysis", "Spatial confounding basis scale",
                                                       advanced=True, unit="m")
    n_boot: int = F(200, "analysis", "Bootstrap replicates", advanced=True)
    dag_audit: bool | DagAudit = F(False, "analysis", "Data-driven DAG audit (true or {edges, n_boot, n_iter})",
                                   advanced=True)


class ClimateSection(_Section):
    enabled: bool = F(False, "analysis", "CMIP6 projections × adaptation")
    source: Literal["table", "cmip6"] = F("table", "analysis", "Change factors from a table or fetched at run time",
                                          enum_labels={"table": "table (CSV)", "cmip6": "fetch CMIP6 at run time"})
    table: str | None = F(None, "analysis", "Change-factor CSV (from the CMIP6 input job)")
    site: list[float] | None = F(None, "analysis", "[lat, lon] (default: the data centroid)")
    experiments: list[str] = F(group="analysis", help="SSPs", factory=lambda: ["ssp126", "ssp245", "ssp370", "ssp585"],
                               enum_labels={"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0",
                                            "ssp585": "SSP5-8.5"})
    periods: dict[str, list[int]] = F(group="analysis", help="Future periods {name: [first, last]}",
                                      factory=lambda: {"2021-2040": [2021, 2040], "2041-2060": [2041, 2060],
                                                       "2081-2100": [2081, 2100]})
    months: list[int] = F(group="analysis", help="Months of the season", factory=lambda: [6, 7, 8])
    variable: Literal["tasmax", "tas"] = F("tasmax", "analysis", "CMIP6 variable")
    thresholds: list[float] | None = F(None, "analysis", "Exposure thresholds (target units)")
    adaptation: list[str] | None = F(None, "analysis", "Scenario names paired with warming")
    cache: str = F("cache", "analysis", "CMIP6 cache (relative to output.dir)", advanced=True)


class OptimizeSection(_Section):
    enabled: bool = F(True, "analysis", "Budget optimisation (S7)")
    variable: str | None = F(None, "analysis", "Lever to allocate")
    budget: float | None = F(None, "analysis", "Budget in dose units (dose × cells)")
    cost_per_unit: float = F(1.0, "analysis", "Cost of one dose unit")
    plantable: bool = F(True, "analysis", "Cap canopy doses by plantable space (needs planner.layers)")
    objective: Literal["cooling", "people"] = F("cooling", "analysis", "What the allocation maximises",
                                                enum_labels={"cooling": "cooling", "people": "cooling × residents"})
    equity_column: str | None = F(None, "analysis", "Column of equity scores")
    equity_focus: float = F(0.0, "analysis", "Weight of the equity scores (0–1)")


class PlannerSection(_Section):
    layers: str | None = F(None, "analysis", "People and land-cover table (from the layers input job)")
    paved_plantable_share: float = F(0.2, "analysis", "Share of paved area counted as plantable")
    ghcn_station: str | None = F(None, "analysis", "GHCN-Daily station for hot-day counts")


class ReportSection(_Section):
    title: str | None = F(None, "about", "Report title")
    place: str | None = F(None, "about", "Place name")
    area: str | None = F(None, "about", "Study area description")
    limitations: list[str] = F(group="about", help="Limitations shown in the model card", factory=list)
    caveats: list[str] = F(group="about", help="Caveats shown on the results page", factory=list)


class OutputSection(_Section):
    dir: str = F("output/core", "run", "Output folder (Studio sets it per run)", advanced=True)
    scenario_detail: bool = F(True, "run", "Write scenario_detail.npz (per-fold Δ for paired comparisons)",
                              advanced=True)


class CoreConfigModel(_Section):
    """The ``core:`` block of a SPARC core config (unknown keys pass through)."""

    name: str = F("core_run", "about", "Run name")
    data: DataSection = F(group="data", help="Input table and coordinates", factory=DataSection)
    predictors: list[str] = F(group="levers", help="Predictor columns", factory=list)
    encodings: EncodingsSection = F(group="levers", help="Categorical and circular predictors",
                                    factory=EncodingsSection)
    qa: QaSection = F(group="levers", help="Value clipping", factory=QaSection)
    actionable: dict[str, ActionableSpec] = F(group="levers", help="Levers (actionable predictors)", factory=dict)
    coupling: list[CouplingSpec] = F(group="levers", help="Sum caps between levers", factory=list)
    mediators: dict[str, MediatorSpec] = F(group="levers", help="Predictors moved by levers", factory=dict)
    physics: PhysicsSection = F(group="physics", help="Physics model and forcing", factory=PhysicsSection)
    influence: InfluenceSection = F(group="analysis", help="Area of influence (S1)", advanced=True,
                                    factory=InfluenceSection)
    cv: CvSection = F(group="analysis", help="Cross-validation", advanced=True, factory=CvSection)
    models: ModelsSection = F(group="analysis", help="Base models", advanced=True, factory=ModelsSection)
    stacker: StackerSection = F(group="analysis", help="Stacker", advanced=True, factory=StackerSection)
    response: ResponseSection = F(group="analysis", help="Response surfaces (S4)", advanced=True,
                                  factory=ResponseSection)
    scenarios: list[ScenarioLadder] = F(group="scenarios", help="Configured scenario ladders", factory=list)
    joint_scenarios: list[JointScenario] = F(group="scenarios", help="Configured packages", factory=list)
    causal: CausalSection = F(group="analysis", help="Causal validation (S6)", factory=CausalSection)
    climate: ClimateSection = F(group="analysis", help="Climate projections", factory=ClimateSection)
    optimize: OptimizeSection = F(group="analysis", help="Budget optimisation (S7)", factory=OptimizeSection)
    planner: PlannerSection | None = F(group="analysis", help="Planner pack inputs", factory=PlannerSection)
    report: ReportSection | None = F(group="about", help="Report text", factory=ReportSection)
    output: OutputSection = F(group="run", help="Run output", advanced=True, factory=OutputSection)


def config_json_schema() -> dict:
    """JSON schema of :class:`CoreConfigModel` with ``x-ui`` hints (``GET /api/config/schema``)."""
    schema = CoreConfigModel.model_json_schema()
    schema["title"] = "CoreConfig"
    schema["description"] = ("The core: block of a SPARC core config (configs/core_providence.yml). Unknown keys "
                             "are kept; x-ui gives {group, advanced, unit, help, enum_labels} per property.")
    return schema


def defaults_diff(defaults: dict | None = None) -> dict:
    """How :class:`CoreConfigModel` differs from core's DEFAULTS: ``{missing: [path], different: [{path, default,
    model}]}``.  A DEFAULTS mapping is walked into only where the model has a sub-model there (``physics``),
    not where the model has a free map (``climate.periods``, ``actionable``)."""
    if defaults is None:
        from sparc.core.config import DEFAULTS as defaults

    paths = model_paths()
    dumped = CoreConfigModel().model_dump(mode="json")
    missing: list[str] = []
    different: list[dict] = []

    def norm(v):
        if isinstance(v, tuple):
            return [norm(x) for x in v]
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            d = {str(k): norm(x) for k, x in v.items() if x is not None}
            return d
        if isinstance(v, int) and not isinstance(v, bool):
            return float(v)
        return v

    def walk(d: dict, md: Any, prefix: str) -> None:
        for k, v in d.items():
            p = f"{prefix}{k}"
            if p not in paths:
                missing.append(p)
                continue
            mv = md.get(k) if isinstance(md, dict) else None
            if isinstance(v, dict) and any(q.startswith(p + ".") for q in paths):
                walk(v, mv, p + ".")
            elif norm(v) != norm(mv):
                different.append({"path": p, "default": v, "model": mv})

    walk(defaults, dumped, "")
    return {"missing": missing, "different": different}


def _unwrap(annotation) -> list[type]:
    """The BaseModel classes inside an annotation (``X | None``, ``dict[str, X]``, ``list[X]``)."""
    import typing

    out = []
    origin = typing.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out.append(annotation)
        return out
    for arg in typing.get_args(annotation):
        out.extend(_unwrap(arg))
    return out


def model_paths(model: type[BaseModel] = CoreConfigModel, prefix: str = "") -> set[str]:
    """Dotted paths of every field of ``model``; a field typed as a sub-model (directly or optional) adds its
    own fields (``physics.roles.canopy``).  Maps and lists of models (``actionable``) stop at the field."""
    import typing

    out: set[str] = set()
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        out.add(path)
        ann = info.annotation
        origin = typing.get_origin(ann)
        direct = isinstance(ann, type) and issubclass(ann, BaseModel)
        optional = origin in (typing.Union, getattr(__import__("types"), "UnionType", None)) and \
            any(isinstance(a, type) and issubclass(a, BaseModel) for a in typing.get_args(ann)) and \
            not any(typing.get_origin(a) in (list, dict) for a in typing.get_args(ann))
        if direct or optional:
            for sub in _unwrap(ann):
                out |= model_paths(sub, path + ".")
    return out
