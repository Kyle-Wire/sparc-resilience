"""
SPARC V3 physics module.

PDE-informed components for physics-based regularization, boundary
conditions, energy balance, and spatially-varying process rates.

Submodules
----------
pde_operators
    Finite-difference Laplacian, gradient, curvature, and Hessian operators.
input_derivatives
    Per-predictor spatial derivatives for physics feature enrichment.
energy_balance
    Urban energy balance terms (radiation, sensible, latent, storage).
pde_loss
    Multi-term PDE loss with staged curriculum activation.
boundary_conditions
    Neumann, Dirichlet, and Robin boundary condition losses.
initial_conditions
    Initial condition generation and warmup-scheduled IC loss.
"""
