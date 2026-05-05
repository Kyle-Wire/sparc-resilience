"""
PDE-derived κ estimator for SPARC's Stage 0 correlogram coupling.

The Whittle relation links the screened-Poisson SPDE

    (κ² − Δ) u(x) = W(x)        (1)

to the Matérn covariance with range parameter ``κ`` and smoothness ``ν``:
solutions of (1) have a Matérn(κ, ν=1) marginal correlation function
``ρ(h) = (κh) K_1(κh)``.  More generally for the fractional SPDE
``(κ² − Δ)^(α/2) u = W`` one obtains Matérn(κ, ν = α − d/2) on ℝ^d.

In SPARC the Track A "process rate" represents a domain-specific physical
rate parameter — for UHI it is **thermal admittance** (W·m⁻²·K⁻¹), for
coastal it is hydraulic diffusivity (m²/s), and so on.  Two regimes:

* **Diffusive regime** — when the process rate has units of diffusivity
  ``D`` (m²/s) and there is a relaxation time ``T`` (s), the stationary
  screened-Poisson balance ``D Δu − u/T = forcing`` gives
  ``κ_PDE² = 1 / (D · T)``.  We extract ``D`` from ``process_rate``
  and use a per-domain default ``T`` (configurable via
  ``physics.relaxation_time_s``).

* **Admittance / surface regime** — when the process rate is a surface
  admittance with no native length-scale (e.g. UHI thermal admittance in
  W·m⁻²·K⁻¹), we fall back to the **domain length-scale** prior:
  ``κ_PDE = 1 / L_char`` where ``L_char`` is configured via
  ``physics.characteristic_length_m`` (default = 30 m for UHI tile-scale
  processes).

When configuration is missing or ambiguous we return ``None`` so the
caller can record "κ_PDE unavailable" rather than fabricate a value.

This module is intentionally lightweight: it does NOT load the
``ProcessRateNet`` checkpoint.  The per-cell PDE-derived κ surface (for
spatially varying κ_PDE) belongs to a later phase; here we only need the
**domain-mean κ_PDE** to compute the ``kappa_ratio`` divergence
diagnostic in Stage 0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class KappaPDEResult:
    """Domain-mean κ_PDE plus provenance.

    ``kappa_pde`` may be ``None`` when the configuration does not allow a
    principled estimate; downstream consumers must handle that case.
    """
    kappa_pde: Optional[float]      # m⁻¹ (so 1/κ has units of metres)
    regime: str                     # "diffusive" | "admittance" | "unavailable"
    source: str                     # human-readable provenance string
    inputs: dict[str, Any]          # dict of resolved numeric inputs


def _is_diffusive_units(units: str) -> bool:
    if not units:
        return False
    u = units.lower().replace(" ", "")
    return ("m²/s" in u) or ("m^2/s" in u) or ("m2/s" in u) or ("m**2/s" in u)


def estimate_kappa_pde(config: dict) -> KappaPDEResult:
    """Estimate the domain-mean κ_PDE from project configuration.

    Parameters
    ----------
    config : dict
        Loaded project.yml (the same object passed to other Stage 0 helpers).
        We look for:

        - ``process_rate.prior_mean`` and ``process_rate.units`` to detect
          the regime and pull a numeric value.
        - ``physics.relaxation_time_s`` (optional, diffusive regime).
        - ``physics.characteristic_length_m`` (optional, admittance
          regime).  Default 30 m.

    Returns
    -------
    KappaPDEResult
    """
    pr_cfg = (config.get("process_rate") or {}) if isinstance(config, dict) else {}
    phys_cfg = (config.get("physics") or {}) if isinstance(config, dict) else {}

    prior_mean = pr_cfg.get("prior_mean")
    units = str(pr_cfg.get("units") or "")

    # ------------------------------------------------------------------
    # Diffusive regime: κ² = 1 / (D · T)
    # ------------------------------------------------------------------
    if prior_mean is not None and _is_diffusive_units(units):
        try:
            D = float(prior_mean)
        except (TypeError, ValueError):
            D = 0.0
        T = phys_cfg.get("relaxation_time_s", 3600.0)
        try:
            T_val = float(T)
        except (TypeError, ValueError):
            T_val = 3600.0
        if D > 0 and T_val > 0:
            kappa = 1.0 / math.sqrt(D * T_val)
            return KappaPDEResult(
                kappa_pde=float(kappa),
                regime="diffusive",
                source=f"Whittle κ² = 1/(D·T) with D={D:g} {units}, T={T_val:g} s",
                inputs={"D": D, "T_s": T_val, "units": units},
            )

    # ------------------------------------------------------------------
    # Admittance / surface regime: κ = 1 / L_char
    # ------------------------------------------------------------------
    L_char = phys_cfg.get("characteristic_length_m")
    if L_char is None:
        # UHI default: 30 m tile scale.  Other domains should override.
        L_char = 30.0
    try:
        L_val = float(L_char)
    except (TypeError, ValueError):
        L_val = 30.0
    if L_val > 0:
        kappa = 1.0 / L_val
        return KappaPDEResult(
            kappa_pde=float(kappa),
            regime="admittance",
            source=f"length-scale prior κ = 1/L_char with L_char={L_val:g} m",
            inputs={"L_char_m": L_val, "process_rate_units": units},
        )

    return KappaPDEResult(
        kappa_pde=None,
        regime="unavailable",
        source="no usable physics inputs in project.yml",
        inputs={},
    )


# ---------------------------------------------------------------------------
# Divergence diagnostic
# ---------------------------------------------------------------------------


def kappa_ratio_summary(
    kappa_correlogram_samples,
    kappa_pde: Optional[float],
    *,
    valid_lo: float = 0.5,
    valid_hi: float = 2.0,
) -> dict:
    """Compare κ_correlogram posterior to κ_PDE point estimate.

    Returns a JSON-friendly summary suitable for the Stage 0 artifact:

    - ``ratio_samples`` (list[float]) — posterior κ_corr / κ_PDE
    - ``ratio_mean`` / ``ratio_hdi89`` — posterior summaries
    - ``p_in_valid_range`` — posterior mass on ``[valid_lo, valid_hi]``
    - ``stationarity_warning`` — True when ``p_in_valid_range < 0.5``
    - ``kappa_pde`` — the value used (None when unavailable)

    When ``kappa_pde`` is None the diagnostic is skipped and a stub
    summary is returned with a clear ``regime`` flag.
    """
    import numpy as np

    samples = np.asarray(kappa_correlogram_samples, dtype=np.float64).ravel()
    if kappa_pde is None or kappa_pde <= 0:
        return {
            "kappa_pde": None,
            "ratio_samples": [],
            "ratio_mean": None,
            "ratio_hdi89": [None, None],
            "p_in_valid_range": None,
            "stationarity_warning": False,
            "note": "κ_PDE unavailable — divergence diagnostic skipped",
        }
    ratio = samples / float(kappa_pde)
    # 89% HDI
    s = np.sort(ratio)
    n = len(s)
    if n >= 2:
        k = max(1, int(math.floor(0.89 * n)))
        widths = s[k:] - s[: n - k]
        i = int(np.argmin(widths))
        hdi = (float(s[i]), float(s[i + k]))
    else:
        v = float(s[0]) if n == 1 else float("nan")
        hdi = (v, v)
    p_in = float(((ratio >= valid_lo) & (ratio <= valid_hi)).mean()) if n > 0 else 0.0
    return {
        "kappa_pde": float(kappa_pde),
        "ratio_samples": [float(x) for x in ratio.tolist()],
        "ratio_mean": float(ratio.mean()) if n > 0 else None,
        "ratio_hdi89": [hdi[0], hdi[1]],
        "p_in_valid_range": p_in,
        "stationarity_warning": bool(p_in < 0.5),
        "valid_range": [float(valid_lo), float(valid_hi)],
    }
