"""
Surface Energy Balance (SEB) helper components for SPARC V3.

Reference balance:
  Q* = QH + QE + QS   (+ ΔQA, anthropogenic, neglected here)

Where:
  Q*  — net all-wave radiation                       (:func:`net_radiation`)
  QH  — turbulent sensible heat flux to the air
  QE  — latent heat flux (evapotranspiration)        (:func:`latent_heat_flux`)
  QS  — ground/subsurface heat storage               (:func:`storage_flux`)

IMPORTANT: this module is NOT wired into the PDE loss
(:mod:`sparc.physics.pde_loss` has no energy-balance term) or anywhere else
in training.  It provides stand-alone helpers only.

In particular there is no sensible-heat parameterisation here: turbulent QH
requires a bulk-transfer form ρ·c_p·(T_s − T_a)/r_a, i.e. air temperature and
an aerodynamic resistance.  The former ``sensible_heat_flux`` computed
−k·∇²T·d, which is a (lateral) ground-conduction proxy; it is now
:func:`ground_conduction_proxy`, with ``sensible_heat_flux`` kept as a
deprecated alias.  Units: W/m².
"""

from __future__ import annotations

import warnings

import torch

# Stefan–Boltzmann constant (W m⁻² K⁻⁴)
SIGMA = 5.670374419e-8

# Thermal conductivity of dry soil/urban substrate (W m⁻¹ K⁻¹)
K_SOIL_DEFAULT = 1.5

# Volumetric heat capacity of urban substrate (J m⁻³ K⁻¹)
RHO_C_DEFAULT = 2.0e6

# Latent heat of vaporization (J/kg)
L_V = 2.45e6

# Priestley–Taylor coefficient (dry ≈ 0.26; wet ≈ 1.26)
ALPHA_PT_DRY = 0.26
ALPHA_PT_WET = 1.26


def net_radiation(
    T: torch.Tensor,
    albedo: torch.Tensor,
    solar_Wm2: float = 800.0,
    T_sky_K: float = 260.0,
    emissivity: float = 0.95,
) -> torch.Tensor:
    """
    Net all-wave radiation Q*.

    Q* = SW_in(1 - albedo) + ε·σ·T_sky⁴ - ε·σ·T⁴

    Parameters
    ----------
    T : (N,) surface temperature in Kelvin
    albedo : (N,) broadband albedo [0, 1]
    solar_Wm2 : incoming shortwave irradiance
    T_sky_K : effective sky temperature for longwave
    emissivity : surface longwave emissivity

    Returns
    -------
    Q_star : (N,) net radiation in W/m²
    """
    sw_net = solar_Wm2 * (1.0 - albedo)
    lw_down = emissivity * SIGMA * T_sky_K ** 4
    lw_up = emissivity * SIGMA * T ** 4
    return sw_net + lw_down - lw_up


def ground_conduction_proxy(
    laplacian_T: torch.Tensor,
    depth: float = 0.5,
    k_thermal: float = K_SOIL_DEFAULT,
) -> torch.Tensor:
    """
    Conductive-flux proxy from the lateral temperature curvature.

    G ≈ -k · ∇²T · d

    This is a (Fourier's-law) ground/substrate conduction proxy integrated
    over a layer of depth ``d``.  It is NOT the turbulent sensible heat flux
    QH to the atmosphere (which needs air temperature and an aerodynamic
    resistance).

    Parameters
    ----------
    laplacian_T : (N,) Laplacian of temperature field
    depth : surface layer depth (m)
    k_thermal : thermal conductivity (W m⁻¹ K⁻¹)

    Returns
    -------
    G : (N,) conduction proxy (W/m²)
    """
    return -k_thermal * laplacian_T * depth


def sensible_heat_flux(
    laplacian_T: torch.Tensor,
    depth: float = 0.5,
    k_thermal: float = K_SOIL_DEFAULT,
) -> torch.Tensor:
    """Deprecated alias of :func:`ground_conduction_proxy`.

    The formula −k·∇²T·d is ground conduction, not sensible heat.
    """
    warnings.warn(
        "sensible_heat_flux is deprecated and misnamed (it computes a ground "
        "conduction proxy -k*lap(T)*d, not sensible heat); use "
        "ground_conduction_proxy instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ground_conduction_proxy(laplacian_T, depth=depth, k_thermal=k_thermal)


def latent_heat_flux(
    canopy_fraction: torch.Tensor,
    ndvi: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Latent heat flux via Priestley–Taylor approximation.

    QE = α_PT · (Δ / (Δ + γ)) · Q*_approx

    Simplified: linearly interpolates between dry (impervious) and
    wet (vegetated) Priestley–Taylor coefficient based on canopy fraction.

    Parameters
    ----------
    canopy_fraction : (N,) fractional canopy cover [0, 1]
    ndvi : (N,) optional — if provided, modulates evaporative fraction

    Returns
    -------
    QE_scale : (N,) dimensionless evaporative scaling factor [0, 1]
        Multiply by net radiation to get QE in W/m².
    """
    alpha_pt = ALPHA_PT_DRY + (ALPHA_PT_WET - ALPHA_PT_DRY) * canopy_fraction.clamp(0, 1)
    if ndvi is not None:
        # NDVI modulation: scale by vegetation vigor
        ndvi_factor = ndvi.clamp(0, 1)
        alpha_pt = alpha_pt * (0.3 + 0.7 * ndvi_factor)
    # Normalize to [0, 1] scale factor
    return (alpha_pt / ALPHA_PT_WET).clamp(0, 1)


def storage_flux(
    T: torch.Tensor,
    rho_c: float = RHO_C_DEFAULT,
    depth: float = 0.5,
) -> torch.Tensor:
    """
    Subsurface heat storage flux.

    QS = ρ·c·d·T  (proportional to temperature for steady-state)

    For steady-state analysis (no ∂T/∂t), storage is proportional
    to T anomaly from reference.  Note that ρ·c·d·T is a heat *content*
    (J m⁻²), not a flux: the physical storage flux is ρ·c·d·∂T/∂t, which
    needs at least two snapshots.  Not used by the training loss.

    Parameters
    ----------
    T : (N,) temperature field
    rho_c : volumetric heat capacity (J m⁻³ K⁻¹)
    depth : storage layer depth (m)

    Returns
    -------
    QS : (N,) storage flux scaling
    """
    return rho_c * depth * T
