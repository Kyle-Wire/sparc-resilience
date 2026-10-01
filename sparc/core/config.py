"""Configuration for the core pipeline.

A core config is a YAML file with a top-level ``core:`` block (or the block's
contents at top level).  Every section has defaults, so a minimal config only
needs ``data`` and ``predictors``.  See ``configs/core_providence.yml`` for a
fully commented example.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# US survey foot is exact: 1200/3937 m.  International foot: 0.3048 m.
UNIT_TO_METRES = {
    "m": 1.0,
    "metre": 1.0,
    "meter": 1.0,
    "us_survey_foot": 1200.0 / 3937.0,
    "us_ft": 1200.0 / 3937.0,
    "ftus": 1200.0 / 3937.0,
    "ft": 0.3048,
    "foot": 0.3048,
}

DEFAULTS: dict[str, Any] = {
    "name": "core_run",
    "data": {
        "path": None,
        "target": None,
        "target_units": "degF",
        "id": None,
        "x": "x",
        "y": "y",
        "coord_unit": "m",
        "crs": None,
        "reproject_to": None,
        "background": "median",
        "subsample": None,
        "cell_m": None,
        "coarse_m": None,          # aggregate to this cell size over the full extent (validation studies)
        "zone": None,
    },
    "predictors": [],
    "encodings": {"categorical": [], "circular_degrees": []},
    "qa": {"clip": {}},
    "actionable": {},
    "coupling": [],
    "mediators": {},
    "physics": {
        "enabled": True,
        "window": "day",
        "sw_down": 800.0,
        "lw_net": -100.0,
        "forcing": None,            # JSON from `sparc core forcing`: sets window, sw_down, lw_net, wind
        "roles": {},
        "wind": None,
        "tau_s": 1800.0,
        "L_max_m": 2000.0,
        "v_max_m": 1000.0,
        "fit_advection": "auto",    # True only when a wind record is configured
        "select_advection": True,   # keep advection only if it beats v = 0 out of fold
        "max_iter": 60,
    },
    "influence": {"max_lag_m": 2000.0, "n_rings": 10, "n_perm": 19, "scales": [0.5, 1.0, 2.0], "mass": 0.9},
    "cv": {
        "n_folds": 5, "block_m": "auto", "buffer_m": "auto", "seed": 42,
        # Reporting-only skill-vs-distance curve (0 = random points, a leaky reference).
        "distance_curve": {"enabled": False, "block_m": [0, 500, 1000]},
        # Reference baselines on the same folds (true = all; or a list of
        # regression_kriging, hgb_xy, hgb, idw, hgb_focal; false = off).
        "baselines": True,
    },
    "models": {"ols": True, "mgwr": True, "gwrf": True, "gam": True, "physics": True,
               # Spatial+: covariate terms see only what the model's spatial term cannot
               # represent, so spatial smooths stop absorbing (attenuating) covariate
               # effects.  List of models (mgwr, gam), true = both, [] = none.  Off by
               # default: it de-attenuates MGWR on the synthetic city but worsened
               # held-out accuracy on Providence (see CORE_ROADMAP Appendix C).
               "spatial_plus": []},
    "stacker": {
        "hidden": 64,
        "physics_mode": "feature",   # "feature" (physics is a base-model input) | "backbone"
        "epochs": 400,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "lambda_pde": 1.0,
        "tune_lambda": [0.0, 0.01, 0.1, 1.0],
        "use_features": True,        # raw + focal features as MLP inputs (besides base predictions)
        "val_fraction": 0.25,        # inner held-out share of training blocks (early stopping + gate)
        "eval_every": 10,
        "patience": 10,              # evaluations without improvement before stopping
        "min_gain": 0.01,            # residual kept only if it lowers held-out MSE by ≥ 1 %
        "allow_residual_off": True,  # also score the convex base alone; pick by outer out-of-fold RMSE
        "coverage": 0.9,
        "seed": 0,
    },
    "response": {"min_valid_doses": 4, "ds_grid": 32},
    "scenarios": [],
    "joint_scenarios": [],
    "causal": {
        "enabled": True,
        "treatments": [],
        "confounders": {},
        "exclude_controls": {},
        "contrast": {},
        "spatial_basis_scale_m": "auto",
        "n_boot": 200,
        "dag_audit": False,
    },
    "climate": {
        "enabled": False,
        # "table": per-model change factors from a CSV (``sparc core climate`` writes one);
        # "cmip6": fetch from the AWS Pangeo CMIP6 archive at run time (cached).
        "source": "table",
        "table": None,
        "site": None,                    # [lat, lon]; default: centroid of the data
        "experiments": ["ssp126", "ssp245", "ssp370", "ssp585"],
        "periods": {"2021-2040": [2021, 2040], "2041-2060": [2041, 2060], "2081-2100": [2081, 2100]},
        "months": [6, 7, 8],
        "variable": "tasmax",
        "thresholds": None,              # target units; default 90/95 °F or 32/35 °C
        "adaptation": None,              # scenario names; default: packages + largest in-support dose per variable
        "cache": "cache",                # relative to output.dir
    },
    "optimize": {
        "enabled": True,
        "variable": None,
        "budget": None,
        "cost_per_unit": 1.0,
        "plantable": True,          # cap canopy doses by plantable space when planner.layers is set
        "objective": "cooling",     # or "people": weight cooling by residents nearby (planner.layers)
        "equity_column": None,
        "equity_focus": 0.0,
    },
    "output": {"dir": "output/core"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class CoreConfig:
    raw: dict
    base_dir: Path = field(default_factory=Path.cwd)

    # ---------------------------------------------------------------- access
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return str(self.raw.get("name", "core_run"))

    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def predictors(self) -> list[str]:
        return list(self.raw["predictors"])

    @property
    def actionable(self) -> dict:
        return self.raw.get("actionable") or {}

    @property
    def mediators(self) -> dict:
        return self.raw.get("mediators") or {}

    def resolve_path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        pp = Path(p)
        return pp if pp.is_absolute() else (self.base_dir / pp).resolve()

    @property
    def data_path(self) -> Path:
        path = self.resolve_path(self.data.get("path"))
        if path is None:
            raise ValueError("core config: data.path is required")
        return path

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.raw["output"]["dir"])  # type: ignore[return-value]

    @property
    def coord_scale(self) -> float:
        unit = str(self.data.get("coord_unit", "m")).lower()
        if unit not in UNIT_TO_METRES:
            raise ValueError(f"Unknown coord_unit {unit!r}; use one of {sorted(UNIT_TO_METRES)}")
        return UNIT_TO_METRES[unit]

    def physics_role(self, role: str) -> str | None:
        return (self.raw["physics"].get("roles") or {}).get(role)

    def validate(self) -> None:
        d = self.data
        if not d.get("target"):
            raise ValueError("core config: data.target is required")
        if not self.predictors:
            raise ValueError("core config: predictors must list at least one column")
        for name in self.actionable:
            if name not in self.predictors:
                raise ValueError(f"actionable variable {name!r} is not a predictor")
        for med, spec in self.mediators.items():
            if med not in self.predictors:
                raise ValueError(f"mediator {med!r} is not a predictor")
            for p in spec.get("parents", []):
                if p not in self.predictors:
                    raise ValueError(f"mediator parent {p!r} is not a predictor")


def core_config_from_dict(raw: dict, base_dir: str | Path | None = None) -> CoreConfig:
    block = raw.get("core", raw)
    cfg = CoreConfig(raw=_deep_merge(DEFAULTS, block), base_dir=Path(base_dir or Path.cwd()))
    forcing = cfg.raw["physics"].get("forcing")
    if forcing:                       # campaign-day radiation / wind from `sparc core forcing`
        from sparc.core.forcing import apply_forcing_file

        cfg.raw["physics"] = apply_forcing_file(cfg.raw["physics"], cfg.resolve_path(forcing), label=str(forcing))
    cfg.validate()
    return cfg


def load_core_config(path: str | Path) -> CoreConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return core_config_from_dict(raw, base_dir=path.resolve().parent)
