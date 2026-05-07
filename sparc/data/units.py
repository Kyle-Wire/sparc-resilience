"""Consistent unit conversion + human-readable formatting for the SPARC pipeline.

All internal pipeline computations operate on z-scored (standardized)
values.  This module provides:

1. Conversion helpers between z-scores and original data units
   (``z_to_original``, ``original_to_z``, ``delta_z_to_original``).
2. A small variable-metadata layer (``VarMeta``, ``load_variable_meta``)
   that pulls units / display names from a SPARC ``project.yml`` config
   with sensible fallbacks for legacy configs.
3. A unified human-readable formatter (``format_value``,
   ``format_effect``, ``format_delta_summary``) that renders effect-size
   strings such as ``"-0.28 Fahrenheit per 1% of canopy increase"`` for
   reports, dashboards, and downstream surfaces.

Rule: No z-score should ever reach the UI layer.  All API responses
must call ``z_to_original()`` before returning values to the frontend.
The formatter helpers always operate in *original* units — callers
must back-transform first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


def z_to_original(
    z_values: np.ndarray | float,
    mean: float,
    std: float,
) -> np.ndarray | float:
    """Convert standardised z-scores back to original data units.

    Parameters
    ----------
    z_values : array or scalar
        Values in z-score space.
    mean : float
        Mean of the original variable (before standardisation).
    std : float
        Standard deviation of the original variable.

    Returns
    -------
    Values in the original data scale.
    """
    if std == 0:
        return np.full_like(z_values, mean) if isinstance(z_values, np.ndarray) else mean
    return z_values * std + mean


def original_to_z(
    values: np.ndarray | float,
    mean: float,
    std: float,
) -> np.ndarray | float:
    """Convert original-scale values to z-scores.

    Parameters
    ----------
    values : array or scalar
        Values in original units.
    mean : float
        Mean of the original variable.
    std : float
        Standard deviation of the original variable.

    Returns
    -------
    Values in z-score space.
    """
    if std == 0:
        return np.zeros_like(values) if isinstance(values, np.ndarray) else 0.0
    return (values - mean) / std


def delta_z_to_original(
    delta_z: np.ndarray | float,
    std: float,
) -> np.ndarray | float:
    """Convert a z-score *difference* to original-scale units.

    For deltas (e.g. scenario cooling), the mean cancels out:
        Δ_original = Δ_z × std

    Parameters
    ----------
    delta_z : array or scalar
        Change expressed in z-score units.
    std : float
        Standard deviation of the target variable.

    Returns
    -------
    Change in original data units (e.g. °F).
    """
    if std == 0:
        return np.zeros_like(delta_z) if isinstance(delta_z, np.ndarray) else 0.0
    return delta_z * std


# ---------------------------------------------------------------------------
# Variable metadata + human-readable formatter
# ---------------------------------------------------------------------------

# Canonical map of compact unit symbols → spelled-out names used in
# user-facing strings.  Anything not in this map is passed through
# verbatim, which lets templates use any custom label.
_UNIT_FULL_NAMES: dict[str, str] = {
    "°F": "Fahrenheit",
    "F": "Fahrenheit",
    "°C": "Celsius",
    "C": "Celsius",
    "K": "Kelvin",
    "%": "%",
    "percentage points": "percentage points",
    "pp": "percentage points",
    "m": "meters",
    "m/yr": "meters per year",
    "mm": "millimeters",
    "mm/yr": "millimeters per year",
    "m³/day": "cubic meters per day",
    "m3/day": "cubic meters per day",
    "kPa": "kilopascals",
    "dBA": "A-weighted decibels",
    "events/yr": "events per year",
    "km/h": "kilometers per hour",
    "mg/L": "milligrams per liter",
    "W·m⁻²·K⁻¹": "watts per square meter per kelvin",
    # Index-typed variables — keep the user's wording.
    "index": "index",
    "reflectance": "reflectance",
    "ratio": "ratio",
    "FS": "factor of safety",
    "SPEI": "SPEI index",
}

# Default per-unit display increments for "per 1 X" phrasing.
# 0.1 is friendlier than 1.0 for normalized indices; 1 is friendlier
# than 0.05 for percentage-typed treatments.
_DEFAULT_INCREMENT_BY_UNIT: dict[str, float] = {
    "index": 0.1,
    "reflectance": 0.1,
    "ratio": 0.1,
    "SPEI": 0.5,
}


@dataclass
class VarMeta:
    """Display metadata for a single variable column.

    ``unit`` is the raw symbol from the project config (e.g. ``"°F"``);
    ``unit_full`` is the spelled-out form used in human-readable strings
    (e.g. ``"Fahrenheit"``).  ``display_name`` defaults to the column
    name lowercased with underscores → spaces.
    """

    name: str
    unit: Optional[str] = None
    unit_full: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    actionable: bool = False
    direction: Optional[str] = None  # "increase" | "decrease" — from scenarios

    def __post_init__(self) -> None:
        if self.unit and not self.unit_full:
            self.unit_full = _UNIT_FULL_NAMES.get(self.unit, self.unit)
        if not self.display_name:
            self.display_name = self.name.replace("_", " ")


@dataclass
class VariableRegistry:
    """Lookup table of ``VarMeta`` keyed by column name.

    Returned by :func:`load_variable_meta`.  Use :meth:`get` to fetch a
    meta record for any column; missing columns yield a stub
    ``VarMeta(name=col)`` with no unit so formatting still produces a
    sensible (if generic) string.
    """

    metas: dict[str, VarMeta] = field(default_factory=dict)
    target: Optional[str] = None

    def get(self, name: str) -> VarMeta:
        if name in self.metas:
            return self.metas[name]
        return VarMeta(name=name)

    def __contains__(self, name: str) -> bool:
        return name in self.metas


def load_variable_meta(config: Mapping[str, Any]) -> VariableRegistry:
    """Build a :class:`VariableRegistry` from a SPARC project config.

    Resolution order (later wins):

    1. ``output.response_units`` → unit for the target column.
    2. ``scenarios[].variable`` + ``scenarios[].unit`` and
       ``scenarios[].direction`` → unit + direction for treatment columns.
    3. ``causal.actionable_variables`` → ``actionable=True``.
    4. ``variables_meta:`` top-level block — explicit per-column override
       with ``unit``, ``unit_full``, ``display_name``, ``description``.

    All sources are optional and missing keys fall through silently so
    legacy configs continue to work unchanged.
    """
    metas: dict[str, VarMeta] = {}

    # 1. Target unit.
    target_col: Optional[str] = None
    data_cfg = config.get("data") or {}
    if isinstance(data_cfg, Mapping):
        target_col = data_cfg.get("target_column")  # type: ignore[assignment]
    output_cfg = config.get("output") or {}
    target_unit: Optional[str] = None
    if isinstance(output_cfg, Mapping):
        target_unit = output_cfg.get("response_units")  # type: ignore[assignment]
    project_cfg = config.get("project") or {}
    if not target_unit and isinstance(project_cfg, Mapping):
        target_unit = project_cfg.get("response_units")  # type: ignore[assignment]
    if target_col:
        metas[target_col] = VarMeta(name=target_col, unit=target_unit)

    # 2. Scenario-derived treatment metadata.
    scenarios = config.get("scenarios") or []
    if isinstance(scenarios, list):
        for spec in scenarios:
            if not isinstance(spec, Mapping):
                continue
            var = spec.get("variable")
            if not var:
                continue
            unit = spec.get("unit")
            direction = spec.get("direction")
            existing = metas.get(var)
            if existing is None:
                metas[var] = VarMeta(
                    name=var, unit=unit, direction=direction,
                )
            else:
                if unit and not existing.unit:
                    existing.unit = unit
                    existing.unit_full = _UNIT_FULL_NAMES.get(unit, unit)
                if direction and not existing.direction:
                    existing.direction = direction

    # 3. Actionable flag.
    causal_cfg = config.get("causal") or {}
    actionable = []
    if isinstance(causal_cfg, Mapping):
        actionable = causal_cfg.get("actionable_variables") or []
    if isinstance(actionable, list):
        for var in actionable:
            if not isinstance(var, str):
                continue
            existing = metas.get(var)
            if existing is None:
                metas[var] = VarMeta(name=var, actionable=True)
            else:
                existing.actionable = True

    # 4. Explicit ``variables_meta:`` overrides (highest precedence).
    explicit = config.get("variables_meta") or {}
    if isinstance(explicit, Mapping):
        for var, payload in explicit.items():
            if not isinstance(var, str) or not isinstance(payload, Mapping):
                continue
            existing = metas.get(var) or VarMeta(name=var)
            unit = payload.get("unit")
            if unit:
                existing.unit = unit
                existing.unit_full = (
                    payload.get("unit_full")
                    or _UNIT_FULL_NAMES.get(unit, unit)
                )
            if payload.get("display_name"):
                existing.display_name = payload["display_name"]
            if payload.get("description"):
                existing.description = payload["description"]
            if "actionable" in payload:
                existing.actionable = bool(payload["actionable"])
            if payload.get("direction"):
                existing.direction = payload["direction"]
            metas[var] = existing

    return VariableRegistry(metas=metas, target=target_col)


def _round_to_sig(value: float, sig: int = 3) -> float:
    """Round ``value`` to ``sig`` significant figures (NaN-safe)."""
    if not np.isfinite(value) or value == 0.0:
        return float(value)
    digits = sig - int(np.floor(np.log10(abs(value)))) - 1
    return float(np.round(value, digits))


def _fmt_signed(value: float, sig: int = 3) -> str:
    """Format a numeric value with explicit sign and fixed sig figs.

    Uses ASCII ``-`` (never the Unicode minus) so the result is safe
    for CSVs, JSON, and Excel ingest.
    """
    if not np.isfinite(value):
        return "n/a"
    rounded = _round_to_sig(value, sig=sig)
    # Strip trailing .0 for whole numbers; otherwise keep up to ``sig`` digits.
    if rounded == int(rounded):
        return f"{int(rounded):+d}"
    # Decide decimal places from magnitude.
    mag = abs(rounded)
    if mag >= 1.0:
        places = max(0, sig - int(np.floor(np.log10(mag))) - 1)
    else:
        places = max(0, sig - int(np.floor(np.log10(mag))) - 1)
    text = f"{rounded:+.{places}f}"
    # Replace any unicode minus with ASCII (np / locale safety).
    text = text.replace("\u2212", "-")
    # Strip trailing zeros after the decimal point (e.g. "+0.100" → "+0.1")
    # but never remove the decimal entirely.
    if "." in text:
        text = text.rstrip("0").rstrip(".")
        # If stripping consumed everything past the sign, fall back to "+0"
        if text in ("+", "-"):
            text = f"{text}0"
    return text


def _fmt_unsigned(value: float, sig: int = 3) -> str:
    """Format a value without a leading ``+`` (used for increments)."""
    s = _fmt_signed(value, sig=sig)
    return s.lstrip("+")


def _default_increment_for(meta: VarMeta) -> float:
    """Pick a friendly default ``per N unit`` increment for a treatment."""
    if meta.unit and meta.unit in _DEFAULT_INCREMENT_BY_UNIT:
        return _DEFAULT_INCREMENT_BY_UNIT[meta.unit]
    return 1.0


def format_value(
    value: float,
    meta: Optional[VarMeta] = None,
    *,
    sig: int = 3,
    signed: bool = True,
) -> str:
    """Render a single numeric value with its unit.

    Examples
    --------
    >>> format_value(-0.284, VarMeta(name="AAT_z", unit="°F"))
    '-0.284 Fahrenheit'
    >>> format_value(7, VarMeta(name="Pct_Canopy", unit="%"), signed=False)
    '7%'
    """
    fmt = _fmt_signed if signed else _fmt_unsigned
    num = fmt(value, sig=sig)
    if meta is None or not meta.unit_full:
        return num
    if meta.unit == "%" and not signed:
        # Drop the space so "7%" reads naturally, but keep "+0.05 reflectance".
        return f"{num}%"
    return f"{num} {meta.unit_full}"


def format_effect(
    treatment: str,
    target: str,
    value: float,
    registry: Optional[VariableRegistry] = None,
    *,
    increment: Optional[float] = None,
    ci: Optional[tuple[float, float]] = None,
    ci_level: Optional[int] = None,
    sig: int = 3,
) -> str:
    """Render a causal effect string.

    Parameters
    ----------
    treatment, target :
        Column names.  Looked up in ``registry`` when provided.
    value :
        Effect size in *original* units of ``target`` per *one base unit*
        of ``treatment``.  Will be scaled by ``increment`` when set.
    registry :
        Optional :class:`VariableRegistry`.  When ``None`` a stub registry
        is used and the output falls back to bare numbers.
    increment :
        How many ``treatment`` units the effect string should describe.
        Defaults to a sensible per-unit value derived from the treatment's
        unit (1.0 for percentages, 0.1 for normalized indices, 1.0 otherwise).
    ci :
        Optional ``(low, high)`` tuple in original units of ``target``
        (per *one* treatment unit, like ``value``).
    ci_level :
        Confidence level percent for the CI label (e.g. ``90``).  Omitted
        from the rendered string when ``None``.

    Returns
    -------
    A string such as
    ``"-0.28 Fahrenheit per 1% of canopy increase"`` or
    ``"-0.28 Fahrenheit (90% CI: -0.41 to -0.15) per 1% of canopy increase"``.
    """
    reg = registry or VariableRegistry()
    t_meta = reg.get(treatment)
    y_meta = reg.get(target)
    inc = float(increment) if increment is not None else _default_increment_for(t_meta)
    scaled = float(value) * inc
    body = format_value(scaled, y_meta, sig=sig, signed=True)

    if ci is not None and all(np.isfinite(c) for c in ci):
        lo = float(ci[0]) * inc
        hi = float(ci[1]) * inc
        lo_s = _fmt_signed(lo, sig=sig)
        hi_s = _fmt_signed(hi, sig=sig)
        ci_label = f"{ci_level}% CI" if ci_level else "CI"
        body = f"{body} ({ci_label}: {lo_s} to {hi_s})"

    # "per N unit_full of <treatment> <direction-word>"
    inc_text = _fmt_unsigned(inc, sig=sig)
    if t_meta.unit == "%":
        per = f"{inc_text}%"
    elif t_meta.unit_full:
        per = f"{inc_text} {t_meta.unit_full}"
    else:
        per = f"{inc_text} unit"
    name = t_meta.display_name or treatment
    if t_meta.actionable:
        # Use scenario direction if known, else the natural "increase".
        dir_word = t_meta.direction or "increase"
    else:
        dir_word = "observed"
    return f"{body} per {per} of {name} {dir_word}"


def format_delta_summary(
    target: str,
    mean: float,
    ci_lo: Optional[float] = None,
    ci_hi: Optional[float] = None,
    registry: Optional[VariableRegistry] = None,
    *,
    ci_level: Optional[int] = None,
    sig: int = 3,
) -> str:
    """Render a scenario Δ summary in target units.

    Examples
    --------
    >>> reg = VariableRegistry(metas={"AAT_z": VarMeta("AAT_z", unit="°F")})
    >>> format_delta_summary("AAT_z", -1.42, -1.81, -0.93, reg, ci_level=95)
    '-1.42 Fahrenheit (95% CI: -1.81 to -0.93)'
    """
    reg = registry or VariableRegistry()
    y_meta = reg.get(target)
    body = format_value(mean, y_meta, sig=sig, signed=True)
    if (
        ci_lo is not None
        and ci_hi is not None
        and np.isfinite(ci_lo)
        and np.isfinite(ci_hi)
    ):
        lo_s = _fmt_signed(float(ci_lo), sig=sig)
        hi_s = _fmt_signed(float(ci_hi), sig=sig)
        ci_label = f"{ci_level}% CI" if ci_level else "CI"
        body = f"{body} ({ci_label}: {lo_s} to {hi_s})"
    return body


__all__ = [
    "z_to_original",
    "original_to_z",
    "delta_z_to_original",
    "VarMeta",
    "VariableRegistry",
    "load_variable_meta",
    "format_value",
    "format_effect",
    "format_delta_summary",
]
