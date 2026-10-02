"""Output catalog of a core run (SPEC §6.1): every file the pipeline, the post-run actions, the studies and
Studio write, with the stage that produces it, the run-hub tab that shows it and its column dictionary.

One table serves Studio (output states, tab availability, file badges, ``/api/meta``), the results-page
builder and the tests (every file of a run must match an entry or :data:`IGNORED`).  ``files`` are globs
relative to the entry's ``root``:

* ``"run"``: the run directory (``run_core(run_dir=…)``);
* ``"study"``: a study directory (``studies/<sid>/``);
* ``"studio"``: Studio's side folder of a run (``<run_dir>/studio/`` or ``<ws>/imports/<run_id>/studio/``).

A ``{var}`` placeholder stands for an actionable lever (``response_{var}.parquet``); :func:`expand_outputs`
turns such an entry into one entry per lever of a config, and :meth:`OutputSpec.matches` treats it as ``*``.

Conventions surfaced by :data:`DATA_DICTIONARY` (and every legend, tooltip and export README):

* ``delta`` / ΔT: target units, **negative = cooler**; ``cooling`` / ``benefit``: target units, **positive =
  cooler**; totals are target·cells (°F·cells).
* ``own_effect_per_unit``: ∂ΔT per +1 lever unit, sign as in the data; ``footprint_effect_per_unit``:
  target·cells per lever unit.
* ``logger_sites.cell`` and ``before_after_pairs.treated/control`` are **row indices** of the run.

Stdlib only (no numpy, no torch): Studio's server imports it at start-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

__all__ = ["OutputSpec", "OUTPUTS", "IGNORED", "DATA_DICTIONARY", "GROUPS", "VIEWS", "units_for", "unit_label",
           "scenario_slug", "expand_outputs", "match_output", "is_ignored", "output_by_id", "dictionary_rows"]

#: output groups (the outputs grid and the files tree group by these)
GROUPS = ("model", "effects", "decisions", "trust", "docs", "state", "planner", "studio")
#: viewer kinds an output may name instead of a run tab
VIEWS = ("table", "markdown", "json", "map", "download")


@dataclass(frozen=True)
class OutputSpec:
    """One catalogued output (SPEC §6.1)."""

    id: str                     # stable id; "{var}" in templated ids ("response_maps:{var}")
    files: tuple[str, ...]      # globs relative to ``root``
    label: str
    group: str                  # model | effects | decisions | trust | docs | state | planner | studio
    produced_by: str            # "stage:S2_S3" | "post:planner" | "study:placebo" | "studio:scenario"
    view: str                   # a run tab id (SPEC §3.2) or a viewer kind: table|markdown|json|map|download
    formats: tuple[str, ...]    # native + conversions: csv|geojson|geotiff|json|html|md|zip|parquet|npz|...
    manifest_key: str | None = None   # manifest section mirroring it, if any
    dictionary: str | None = None     # DATA_DICTIONARY key
    root: str = "run"                 # run | study | studio
    optional: bool = False            # written only in some runs (frame input, newer code, enabled options)

    @property
    def templated(self) -> bool:
        return "{" in self.id or any("{" in f for f in self.files)

    def patterns(self) -> tuple[str, ...]:
        """The file globs with every ``{placeholder}`` widened to ``*``."""
        return tuple(re.sub(r"\{[^}]+\}", "*", f) for f in self.files)

    def matches(self, relpath: str) -> bool:
        rel = relpath.replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        return any(_glob(rel, p) for p in self.patterns())

    def for_var(self, var: str) -> "OutputSpec":
        """This (templated) entry for one lever."""
        return replace(self, id=self.id.replace("{var}", var), files=tuple(f.replace("{var}", var) for f in self.files),
                       label=self.label.replace("{var}", var))

    def var_of(self, relpath: str) -> str | None:
        """The ``{var}`` a templated file name binds (``response_canopy.parquet`` → ``canopy``)."""
        for f in self.files:
            if "{var}" not in f:
                continue
            rx = "^" + re.escape(f).replace(re.escape("{var}"), "(.+)") + "$"
            m = re.match(rx, relpath)
            if m:
                return m.group(1)
        return None


def _glob(rel: str, pattern: str) -> bool:
    """``fnmatch`` where ``*`` stays inside one path component and ``**`` crosses them."""
    rx = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            rx += ".*"
            i += 2
            if i < len(pattern) and pattern[i] == "/":
                i += 1
            continue
        if c == "*":
            rx += "[^/]*"
        elif c == "?":
            rx += "[^/]"
        else:
            rx += re.escape(c)
        i += 1
    return re.fullmatch(rx, rel) is not None


_S = OutputSpec
OUTPUTS: list[OutputSpec] = [
    # ---------------------------------------------------------------- run core
    _S("manifest", ("manifest.json",), "Run manifest", "state", "stage:finish", "overview", ("json",), None,
       "manifest"),
    _S("influence", ("influence.json",), "Areas of influence", "model", "stage:S1", "influence", ("json", "csv"),
       "influence", "influence.json"),
    _S("predictions", ("predictions.parquet",), "Held-out predictions", "model", "stage:S2_S3", "accuracy",
       ("parquet", "csv", "geojson"), None, "predictions.parquet"),
    _S("physics", ("physics.json",), "Physics parameters per fold", "model", "stage:S2_S3", "accuracy",
       ("json", "csv"), "physics", "physics.json"),
    _S("baselines", ("baselines.json",), "Reference baselines", "trust", "stage:baselines", "distance",
       ("json", "csv"), "baselines", "baselines.json"),
    _S("cv_distance", ("cv_distance.json",), "Skill vs distance", "trust", "stage:cv_curve", "distance",
       ("json", "csv"), "cv_distance", "cv_distance.json", optional=True),
    _S("response_maps:{var}", ("response_{var}.parquet",), "Response maps: {var}", "effects", "stage:S4",
       "response", ("parquet", "csv", "geojson"), "response", "response_{var}.parquet"),
    _S("response_curves", ("response_curves.json",), "Dose-response curves", "effects", "stage:S4", "response",
       ("json",), "response", "response_curves.json"),
    _S("scenarios", ("scenarios.json",), "Configured scenarios", "decisions", "stage:S5", "scenarios",
       ("json", "csv"), "scenarios", "scenarios.json"),
    _S("scenario_deltas", ("scenario_deltas.parquet",), "Scenario ΔT per cell", "decisions", "stage:S5",
       "scenarios", ("parquet", "csv", "geojson"), "scenarios", "scenario_deltas.parquet"),
    _S("scenario_detail", ("scenario_detail.npz",), "Scenario folds, SD and extrapolation", "decisions",
       "stage:S5", "scenarios", ("npz",), "scenarios", "scenario_detail.npz", optional=True),
    _S("climate", ("climate.json",), "Climate projections × adaptation", "decisions", "stage:climate", "climate",
       ("json", "csv"), "climate", "climate.json"),
    _S("causal", ("causal.json",), "Causal validation", "trust", "stage:S6", "causal", ("json",), "causal",
       "causal.json"),
    _S("causal_cells", ("causal_cells.parquet",), "Causal per-cell estimates", "trust", "stage:S6", "causal",
       ("parquet", "csv", "geojson"), "causal", "causal_cells.parquet", optional=True),
    _S("optimize", ("optimize.json",), "Budget allocation summary", "decisions", "stage:S7", "budget",
       ("json", "csv"), "optimize", "optimize.json"),
    _S("allocation", ("allocation.parquet",), "Budget allocation per cell", "decisions", "stage:S7", "budget",
       ("parquet", "csv", "geojson"), "optimize", "allocation.parquet"),
    _S("report", ("report.md",), "Run report (as of run end)", "docs", "stage:finish", "docs", ("md", "html")),
    _S("methods", ("methods.md",), "Methods", "docs", "stage:finish", "docs", ("md", "html")),
    _S("model_card", ("model_card.md",), "Model card", "docs", "stage:finish", "docs", ("md", "html")),
    _S("environment", ("environment.txt",), "Environment lock", "state", "stage:finish", "download", ("txt",)),
    _S("input_frame", ("input_frame.parquet",), "Input table (frame-input runs)", "state", "stage:S0", "data",
       ("parquet", "csv"), None, None, optional=True),
    _S("events", ("events.jsonl",), "Progress events (CLI --progress)", "state", "progress:sink", "download",
       ("jsonl",), optional=True),
    # ---------------------------------------------------------------- run state
    _S("checkpoint", ("checkpoint.pkl",), "Checkpoint", "state", "stage:S2_S3", "download", ("pkl",),
       optional=True),
    _S("checkpoint_json", ("checkpoint.json",), "Checkpoint sidecar", "state", "stage:S2_S3", "json", ("json",),
       optional=True),
    _S("run_state", ("run_state.json",), "Run state", "state", "stage:S0", "json", ("json",)),
    # ---------------------------------------------------------------- post-run
    _S("emulator", ("emulator.npz", "emulator.json"), "Scenario emulator", "decisions", "post:emulator", "json",
       ("npz", "json"), "emulator", optional=True),
    _S("uncertainty", ("uncertainty.json",), "Uncertainty envelopes", "trust", "post:uncertainty", "uncertainty",
       ("json", "csv"), "uncertainty", optional=True),
    _S("uncertainty_md", ("uncertainty.md",), "Uncertainty report", "docs", "post:uncertainty", "docs",
       ("md", "html"), "uncertainty", optional=True),
    _S("placebo_copy", ("placebo.json",), "Placebo results (attached)", "trust", "post:uncertainty", "json",
       ("json",), "placebo", optional=True),
    _S("planner", ("planner/planner.json",), "Planner pack summary", "planner", "post:planner", "planner",
       ("json",), "planner", optional=True),
    _S("planner_cells", ("planner/planner_cells.parquet",), "Planner per-cell layers", "planner", "post:planner",
       "planner", ("parquet", "csv", "geojson"), "planner", "planner/planner_cells.parquet", optional=True),
    _S("planner_hex:250", ("planner/hex_250m.csv",), "Hexagons 250 m", "planner", "post:planner", "planner",
       ("csv",), "planner", "planner/hex_{size}m.csv", optional=True),
    _S("planner_hex:500", ("planner/hex_500m.csv",), "Hexagons 500 m", "planner", "post:planner", "planner",
       ("csv",), "planner", "planner/hex_{size}m.csv", optional=True),
    _S("planner_hex_other", ("planner/hex_*m.csv",), "Hexagons (other sizes)", "planner", "post:planner",
       "planner", ("csv",), "planner", "planner/hex_{size}m.csv", optional=True),
    _S("planner_gpkg", ("planner/hexagons.gpkg",), "Hexagons GeoPackage", "planner", "post:planner", "planner",
       ("gpkg",), "planner", optional=True),
    _S("planner_geotiff", ("planner/geotiff/*.tif",), "Planner GeoTIFFs", "planner", "post:planner", "planner",
       ("geotiff", "zip"), "planner", optional=True),
    _S("logger_sites", ("planner/logger_sites.csv",), "Logger sites", "planner", "post:planner", "planner",
       ("csv", "geojson"), "planner", "planner/logger_sites.csv", optional=True),
    _S("before_after_pairs", ("planner/before_after_pairs.csv",), "Before/after pairs", "planner",
       "post:planner", "planner", ("csv", "geojson"), "planner", "planner/before_after_pairs.csv", optional=True),
    _S("results_page", ("results.html",), "Standalone results page (legacy)", "docs", "studio:export", "download",
       ("html",), optional=True),
    _S("reproduce", ("reproduce.json",), "Reproduction checklist", "trust", "study:reproduce", "json", ("json",),
       optional=True),
    # ---------------------------------------------------------------- studies (study dir)
    _S("placebo_study", ("placebo.json",), "Placebo suite", "trust", "study:placebo", "json", ("json",),
       root="study"),
    _S("placebo_md", ("placebo.md",), "Placebo report", "docs", "study:placebo", "docs", ("md", "html"),
       root="study"),
    _S("simcheck_rows", ("simcheck.jsonl",), "Simulation check rows", "trust", "study:simcheck", "table",
       ("jsonl", "csv"), root="study"),
    _S("simcheck_summary", ("simcheck_summary.json",), "Simulation check summary", "trust", "study:simcheck",
       "json", ("json",), root="study"),
    _S("simcheck_md", ("simcheck_summary.md",), "Simulation check report", "docs", "study:simcheck", "docs",
       ("md", "html"), root="study"),
    _S("multiverse_summary", ("multiverse_summary.json",), "Multiverse summary", "trust", "study:multiverse",
       "json", ("json",), root="study"),
    _S("multiverse_md", ("multiverse_summary.md",), "Multiverse report", "docs", "study:multiverse", "docs",
       ("md", "html"), root="study"),
    _S("benchmark", ("benchmark.json",), "Effect benchmark", "trust", "study:benchmark", "json", ("json",),
       root="study"),
    _S("benchmark_md", ("benchmark.md",), "Effect benchmark report", "docs", "study:benchmark", "docs",
       ("md", "html"), root="study"),
    _S("multiverse_maps:{variant}", ("{variant}_maps.npz",), "Multiverse variant maps: {variant}", "trust",
       "study:multiverse", "map", ("npz",), root="study"),
    _S("multiverse_variant:{variant}", ("{variant}.json",), "Multiverse variant: {variant}", "trust",
       "study:multiverse", "json", ("json",), root="study"),
    # ---------------------------------------------------------------- Studio's side folder (api.md §12.2)
    _S("launch", ("launch.json",), "Launch snapshot", "studio", "studio:launch", "json", ("json",), root="studio"),
    _S("launch_history", ("launch.*.json",), "Earlier launch snapshots", "studio", "studio:launch", "json",
       ("json",), root="studio"),
    _S("import_record", ("import.json",), "Import record", "studio", "studio:import", "json", ("json",),
       root="studio"),
    _S("grid_cache", ("cache/**",), "Grid and layer cache", "studio", "studio:reader", "download", ("npz",),
       root="studio"),
    _S("engine_cache", ("engine/**",), "Engine baseline cache", "studio", "studio:engine", "download", ("npy",),
       root="studio"),
    _S("studio_results", ("results/**",), "Exact scenario results", "studio", "studio:scenario", "json",
       ("json", "parquet", "npy"), root="studio"),
    _S("studio_plans", ("plans/**",), "Budget plans", "studio", "studio:plan", "json", ("json", "npy"),
       root="studio"),
    _S("studio_sweeps", ("sweeps/**",), "Dose sweeps", "studio", "studio:sweep", "json", ("json",),
       root="studio"),
    _S("studio_comparisons", ("comparisons/**",), "Comparisons", "studio", "studio:compare", "json",
       ("json", "npy"), root="studio"),
    _S("studio_blobs", ("blobs/**",), "Selection and brush blobs", "studio", "studio:selection", "download",
       ("bin",), root="studio"),
]
del _S

#: never catalogued or flagged (globs on the run-relative path; a bare name also matches in any folder)
IGNORED: tuple[str, ...] = (".sparc.lock", "*.tmp", "**/*.tmp", "studio/**", "__pycache__/**", "**/__pycache__/**",
                            "children/**", "FIXTURE.json")


def is_ignored(relpath: str) -> bool:
    rel = relpath.replace("\\", "/")
    name = rel.rsplit("/", 1)[-1]
    return any(_glob(rel, p) or ("/" not in p and _glob(name, p)) for p in IGNORED)


def match_output(relpath: str, root: str = "run", specs: Iterable[OutputSpec] | None = None) -> OutputSpec | None:
    """The first catalog entry of ``root`` whose globs match ``relpath`` (None when none does)."""
    rel = relpath.replace("\\", "/")
    for spec in (OUTPUTS if specs is None else specs):
        if spec.root == root and spec.matches(rel):
            return spec
    return None


def output_by_id(oid: str, specs: Iterable[OutputSpec] | None = None) -> OutputSpec | None:
    for spec in (OUTPUTS if specs is None else specs):
        if spec.id == oid:
            return spec
    return None


def expand_outputs(cfg=None, *, levers: Iterable[str] | None = None, root: str | None = "run") -> list[OutputSpec]:
    """Catalog entries with ``{var}`` expanded to the config's actionable levers (``levers`` overrides).

    Templated entries whose placeholder is not ``{var}`` (study variants) are kept as they are.
    """
    if levers is None:
        levers = list((_raw(cfg).get("actionable") or {}) if cfg is not None else ())
    levers = list(levers)
    out = []
    for spec in OUTPUTS:
        if root is not None and spec.root != root:
            continue
        if "{var}" in spec.id:
            out.extend(spec.for_var(v) for v in levers)
        else:
            out.append(spec)
    return out


# ---------------------------------------------------------------------------
# scenario ids
# ---------------------------------------------------------------------------

def scenario_slug(name: str, taken: Iterable[str] = frozenset()) -> str:
    """Configured-scenario id (SPEC §6.1): lower case; "−"/"-" before a number → "minus-", "+" → "plus-";
    every other run of non-[a-z0-9] → "-"; "-" stripped; "-2", "-3"… appended on collision with ``taken``.

    >>> scenario_slug("Impervious Decrease −10")
    'impervious-decrease-minus-10'
    >>> scenario_slug("Albedo Increase +0.1")
    'albedo-increase-plus-0-1'
    """
    s = str(name).lower()
    s = re.sub(r"[−\-](?=\s*\d)", " minus-", s)
    s = re.sub(r"\+(?=\s*\d)", " plus-", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "scenario"
    taken = set(taken)
    if s not in taken:
        return s
    i = 2
    while f"{s}-{i}" in taken:
        i += 1
    return f"{s}-{i}"


# ---------------------------------------------------------------------------
# units and the data dictionary
# ---------------------------------------------------------------------------

_UNIT_LABELS = {"degf": "°F", "degc": "°C", "f": "°F", "c": "°C", "k": "K", "kelvin": "K", "pp": "pp",
                "percent": "%", "pct": "%", "m": "m"}


def unit_label(unit: str | None) -> str:
    """Display form of a unit string (``degF`` → ``°F``)."""
    if not unit:
        return ""
    return _UNIT_LABELS.get(str(unit).strip().lower(), str(unit))


def _raw(cfg) -> dict:
    if cfg is None:
        return {}
    return cfg.raw if hasattr(cfg, "raw") else dict(cfg)


def units_for(cfg) -> dict:
    """Template values for a config: ``target``, ``lever:<var>`` (the lever's unit) and ``lever`` defaults.

    ``{"target": "°F", "lever:canopy": "pp", "levers": {"canopy": "pp", …}}``; a lever without a unit gets
    "units".
    """
    raw = _raw(cfg)
    target = unit_label((raw.get("data") or {}).get("target_units") or "degF")
    levers = {}
    for var, spec in (raw.get("actionable") or {}).items():
        levers[var] = unit_label((spec or {}).get("unit")) or "units"
    out: dict = {"target": target, "levers": levers}
    for var, u in levers.items():
        out[f"lever:{var}"] = u
    return out


def _fill(template: str, units: dict, var: str | None = None) -> str:
    out = template.replace("{target}", units.get("target", "°F"))
    if var is not None:
        out = out.replace("{lever}", units.get(f"lever:{var}", "units"))

    def sub(m):
        return units.get(f"lever:{m.group(1)}", "units")

    return re.sub(r"\{lever:([^}]+)\}", sub, out)


_COOLER = "negative = cooler"
_BENEFIT = "positive = cooler"
DATA_DICTIONARY: dict[str, dict[str, tuple[str, str, str]]] = {
    "predictions.parquet": {
        "id": ("", "", "Point id (the data.id column)"),
        "x_m": ("m", "", "Easting of the cell centre in the run frame"),
        "y_m": ("m", "", "Northing of the cell centre in the run frame"),
        "fold": ("", "", "Spatial CV fold that held the point out (0-based)"),
        "zone": ("", "", "Zone / neighbourhood code (reporting only)"),
        "target": ("{target}", "", "Observed temperature"),
        "dT": ("{target}", "", "Observed temperature minus the background"),
        "dT_pred": ("{target}", "", "Held-out (out-of-fold) stacked prediction of dT"),
        "pred": ("{target}", "", "Held-out stacked prediction of the temperature"),
        "pi_lo": ("{target}", "", "Lower end of the conformal prediction interval (global width)"),
        "pi_hi": ("{target}", "", "Upper end of the conformal prediction interval (global width)"),
        "pi_lo_adaptive": ("{target}", "", "Lower end of the distance-adaptive prediction interval"),
        "pi_hi_adaptive": ("{target}", "", "Upper end of the distance-adaptive prediction interval"),
        "dist_train_m": ("m", "", "Distance to the nearest training point of the point's fold"),
        "oof_*": ("{target}", "", "Held-out prediction of one base model"),
    },
    "response_{var}.parquet": {
        "id": ("", "", "Point id"),
        "x_m": ("m", "", "Easting of the cell centre"),
        "y_m": ("m", "", "Northing of the cell centre"),
        "max_cooling_A": ("{target}", _BENEFIT, "Maximum cooling of the fitted neighbourhood-adoption curve"),
        "saturation_scale_ds": ("{lever}", "", "Saturation scale d_s of the curve (neighbourhood dose)"),
        "inflection_dose": ("{lever}", "", "Inflection dose of an S-shaped curve"),
        "d90": ("{lever}", "", "Neighbourhood dose reaching 90% of the maximum cooling (within the tested doses)"),
        "marginal_benefit_per_unit": ("{target}/{lever}", _BENEFIT, "Cooling per extra lever unit at today's level"),
        "curve_model": ("", "", "Fitted curve shape: saturating, linear, sigmoid or insufficient"),
        "censored": ("", "", "True when the curve does not saturate within the tested doses"),
        "fit_r2": ("1", "", "R² of the fitted curve"),
        "headroom": ("{lever}", "", "Room to the lever's bound in its direction"),
        "own_effect_per_unit": ("{target}/{lever}", "sign as in the data", "∂ΔT of the cell per +1 unit of its own lever"),
        "own_effect_sd": ("{target}/{lever}", "", "Fold-to-fold SD of the own effect"),
        "footprint_effect_per_unit": ("{target}·cells/{lever}", "sign as in the data",
                                      "Total ΔT across the neighbourhood per +1 unit in this cell"),
        "footprint_effect_sd": ("{target}·cells/{lever}", "", "Fold-to-fold SD of the footprint effect"),
    },
    "scenario_deltas.parquet": {
        "id": ("", "", "Point id"),
        "*": ("{target}", _COOLER, "ΔT of the configured scenario (column name = scenario name)"),
    },
    "allocation.parquet": {
        "id": ("", "", "Point id"),
        "x_m": ("m", "", "Easting"),
        "y_m": ("m", "", "Northing"),
        "dose": ("{lever}", "", "Planned dose of the optimised lever"),
        "closed_loop_delta": ("{target}", _COOLER, "ΔT of the whole allocation run through the model"),
    },
    "causal_cells.parquet": {
        "id": ("", "", "Point id"),
        "cate:*": ("{target}/{lever}", "sign as in the data", "R-learner conditional effect per unit of the treatment"),
        "mslope:*": ("{target}/{lever}", "sign as in the data", "Model adoption slope per cell"),
        "mslope_own:*": ("{target}/{lever}", "sign as in the data", "Model own-cell slope per cell"),
    },
    "planner/planner_cells.parquet": {
        "id": ("", "", "Point id"),
        "people": ("people", "", "Residents of the cell (HRSL)"),
        "plantable_canopy_pp": ("pp", "", "Plantable canopy headroom"),
        "hot_days_ge_*": ("days/yr", "", "Days per year at or above the threshold (campaign-like offsets)"),
    },
    "planner/hex_{size}m.csv": {
        "hex": ("", "", "Hexagon key ((q + 100000)·1e6 + (r + 100000))"),
        "cx": ("m", "", "Hexagon centre easting"),
        "cy": ("m", "", "Hexagon centre northing"),
        "temperature": ("{target}", "", "Mean observed temperature"),
        "people": ("people", "", "Residents (sum)"),
        "plantable_pp": ("pp", "", "Mean plantable headroom"),
        "package_cooling": ("{target}", _BENEFIT, "Mean cooling of the adaptation package"),
        "n_cells": ("cells", "", "Cells in the hexagon"),
    },
    "planner/logger_sites.csv": {
        "cell": ("row", "", "Row index of the run (Studio adds id and lon/lat on export)"),
        "x_m": ("m", "", "Easting"),
        "y_m": ("m", "", "Northing"),
        "canopy": ("pp", "", "Canopy cover"),
        "impervious": ("pp", "", "Impervious cover"),
        "effect_sd": ("{target}·cells/pp", "", "Fold spread of the canopy footprint effect"),
        "role": ("", "", "low canopy / high canopy"),
    },
    "planner/before_after_pairs.csv": {
        "treated": ("row", "", "Row index of the treated cell"),
        "control": ("row", "", "Row index of the matched control cell"),
        "covariate_distance": ("1", "", "Standardised covariate distance of the pair"),
    },
    "influence.json": {
        "ranges_m": ("m", "", "Influence range per predictor"),
        "block_size_m": ("m", "", "CV block size suggested by the residual range"),
        "L_prior_m": ("m", "", "Prior relaxation length for the physics model"),
    },
    "physics.json": {
        "L_m": ("m", "", "Relaxation length"),
        "a": ("{target}", "", "Physics amplitude"),
        "s": ("1", "", "Shade efficiency"),
    },
    "baselines.json": {
        "delta_mse": ("{target}²", "positive = stack better", "MSE(baseline) − MSE(stack)"),
        "delta_mse_se": ("{target}²", "", "Block-clustered SE of ΔMSE"),
    },
    "cv_distance.json": {
        "block_m": ("m", "", "CV block size of the partition (≈ dx: random points)"),
        "r2": ("1", "", "Held-out R² of the stack"),
    },
    "scenarios.json": {
        "mean_delta": ("{target}", _COOLER, "City-mean ΔT"),
        "mean_delta_se": ("{target}", "", "Jackknife SE over folds"),
        "frac_extrapolated": ("1", "", "Share of cells outside observed conditions"),
    },
    "scenario_detail.npz": {
        "f{i}": ("{target}", _COOLER, "Per-fold ΔT of scenario i (K × n)"),
        "sd{i}": ("{target}", "", "Fold SD of ΔT of scenario i"),
        "ex{i}": ("1", "", "Extrapolation score of scenario i (> 1 = outside support)"),
    },
    "climate.json": {
        "warming": ("{target}", "positive = warmer", "Change in summer daily highs"),
        "offset_share_of_median_warming": ("1", "", "Share of the median warming an adaptation cancels"),
    },
    "causal.json": {
        "theta_own": ("{target}/{lever}", "sign as in the data", "Own-cell causal effect per unit"),
        "theta_sum": ("{target}/{lever}", "sign as in the data", "Own + neighbourhood causal effect per unit"),
    },
    "optimize.json": {
        "planned_total_cooling": ("{target}·cells", _BENEFIT, "Planned total cooling"),
        "realized_total_cooling": ("{target}·cells", _BENEFIT, "Closed-loop total cooling"),
    },
    "manifest": {
        "timings_s": ("s", "", "Seconds per stage"),
    },
}


def dictionary_rows(cfg=None) -> list[dict]:
    """Every dictionary entry with units resolved for ``cfg``: ``{output, column, unit, sign, description}``.

    Per-lever files (``response_{var}.parquet``) are listed once per actionable lever.
    """
    units = units_for(cfg)
    levers = list(units.get("levers") or {})
    opt_var = (_raw(cfg).get("optimize") or {}).get("variable")
    rows = []
    for output, cols in DATA_DICTIONARY.items():
        variants: list[tuple[str, str | None]]
        if "{var}" in output:
            variants = [(output.replace("{var}", v), v) for v in levers] or [(output, None)]
        else:
            var = opt_var if output in ("allocation.parquet", "optimize.json") else None
            variants = [(output, var)]
        for name, var in variants:
            for col, (unit, sign, desc) in cols.items():
                rows.append({"output": name, "column": col, "unit": _fill(unit, units, var),
                             "sign": sign or None, "description": desc})
    return rows
