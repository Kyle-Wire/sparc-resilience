"""
SPARC V3 physics module.

PDE-informed components for physics-based regularization, boundary
conditions, and spatially-varying process rates.  (The surface energy
balance helpers in ``energy_balance`` are stand-alone and are NOT used by
the PDE loss.)

Submodules
----------
pde_operators
    Finite-difference Laplacian, gradient, curvature, and Hessian operators.
input_derivatives
    Per-predictor spatial derivatives for physics feature enrichment.
energy_balance
    Stand-alone surface energy balance helpers (net radiation, latent-heat
    scaling, storage, ground-conduction proxy); not wired into training.
pde_loss
    Multi-term PDE loss with staged curriculum activation.
boundary_conditions
    Neumann, Dirichlet, and Robin boundary condition losses.
initial_conditions
    Initial condition generation and warmup-scheduled IC loss.
constraint
    PhysicsConstraint protocol — seam between training-time PDE loss
    (PDEAdapter) and inference-time prior shrinkage (PriorsAdapter).
"""

from .constraint import (
    PhysicsConstraint,
    PDEAdapter,
    PriorsAdapter,
    NullPhysicsConstraint,
    make_physics_constraint,
)

__all__ = [
    "PhysicsConstraint",
    "PDEAdapter",
    "PriorsAdapter",
    "NullPhysicsConstraint",
    "make_physics_constraint",
]
