"""
Surface Energy Balance (SEB) components for SPARC V3.

Implements the four-component energy balance:
  Q* = QH + QE + QS

Where:
  Q*  — net all-wave radiation
  QH  — sensible heat flux (conduction through surface layer)
  QE  — latent heat flux (evapotranspiration)
  QS  — ground/subsurface heat storage

All fluxes are computed per spatial point from local surface properties
and the predicted temperature field.  Units: W/m².
"""

from __future__ import annotations

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


def sensible_heat_flux(
    laplacian_T: torch.Tensor,
    depth: float = 0.5,
    k_thermal: float = K_SOIL_DEFAULT,
) -> torch.Tensor:
    """
    Sensible heat flux through the surface layer via Fourier's law.

    QH = -k · ∇²T · d

    Parameters
    ----------
    laplacian_T : (N,) Laplacian of temperature field
    depth : surface layer depth (m)
    k_thermal : thermal conductivity (W m⁻¹ K⁻¹)

    Returns
    -------
    QH : (N,) sensible heat flux (W/m²)
    """
    return -k_thermal * laplacian_T * depth


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
    to T anomaly from reference.

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


def energy_balance_residual(
    T: torch.Tensor,
    albedo: torch.Tensor,
    laplacian_T: torch.Tensor,
    canopy_fraction: torch.Tensor,
    solar_Wm2: float = 800.0,
    T_sky_K: float = 260.0,
    k_thermal: float = K_SOIL_DEFAULT,
    depth: float = 0.5,
    ndvi: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the energy balance residual: Q* - QH - QE·Q* - QS.

    A physical temperature field should have residual ≈ 0.

    Parameters
    ----------
    T : (N,) temperature in Kelvin
    albedo : (N,) surface albedo
    laplacian_T : (N,) ∇²T from PDE operators
    canopy_fraction : (N,) vegetation cover fraction
    solar_Wm2 : incoming shortwave
    T_sky_K : sky temperature
    k_thermal : thermal conductivity
    depth : surface layer depth
    ndvi : optional NDVI for latent heat modulation

    Returns
    -------
    residual : (N,) energy balance residual (W/m²)
    """
    Q_star = net_radiation(T, albedo, solar_Wm2, T_sky_K)
    QH = sensible_heat_flux(laplacian_T, depth, k_thermal)
    QE_scale = latent_heat_flux(canopy_fraction, ndvi)
    QE = QE_scale * Q_star

    # Residual: Q* - QH - QE ≈ 0 for steady-state (ignoring storage)
    return Q_star - QH - QE
