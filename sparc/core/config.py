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
    "cv": {"n_folds": 5, "block_m": "auto", "buffer_m": "auto", "seed": 42},
    "models": {"ols": True, "mgwr": True, "gwrf": True, "gam": True, "physics": True},
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
    "optimize": {
        "enabled": True,
        "variable": None,
        "budget": None,
        "cost_per_unit": 1.0,
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
    cfg.validate()
    return cfg


def load_core_config(path: str | Path) -> CoreConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return core_config_from_dict(raw, base_dir=path.resolve().parent)
