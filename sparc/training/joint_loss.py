"""
JointLoss — thin, stateful wrapper around ``sparc_joint_loss``.

Reduces the 60+ parameter call-site to a compact ``forward()`` with
per-batch tensors, while the static lambda weights, thresholds, and
PDE configuration live in a single ``JointLoss`` instance.  Curriculum
weight decay for alpha prior is applied automatically from ``epoch``.

Usage::

    from sparc.training.joint_loss import JointLoss

    joint_loss = JointLoss.from_target_lambdas(
        target_lambdas=target_lambdas,  # dict from train_neural_meta
        thresholds=thresholds,
        resolution=resolution,
    )

    total, components = joint_loss(
        T_pred=T_pred,
        exceedance_preds=exceedance_preds,
        y_true=y_true,
        alpha=alpha,
        alpha_prior=alpha_prior,
        neighbor_idx=fold_cardinal,
        source_term=source_term,
        surrogate_preds=surrogate_preds,
        surrogate_targets=surrogate_targets,
        epoch=epoch,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sparc.training.loss import (
    get_prior_weight,
    sparc_joint_loss,
    PhysicsContext,
    CurriculumState,
    AuxTargets,
)


@dataclass
class JointLoss:
    """Stateful wrapper that stores lambda weights and calls ``sparc_joint_loss``.

    All per-epoch curriculum logic (alpha-prior decay) is handled
    internally; callers only pass the training epoch index.

    Parameters
    ----------
    thresholds : list[float]
        Exceedance probability thresholds (e.g. [0.25, 0.50, 0.75]).
    resolution : float
        Grid spacing in metres for the PDE finite-difference Laplacian.
    lambda_physics : float
        Weight for the physics residual (PDE satisfaction) term.
    lambda_smooth : float
        Weight for the prediction smoothness term.
    lambda_alpha_smooth : float
        Weight for the alpha field spatial-smoothness term.
    lambda_prior : float
        *Base* weight for the alpha prior regularisation — decays with
        epoch via :func:`get_prior_weight` automatically.
    lambda_base : float
        Weight for the surrogate alignment term.
    lambda_neighbor : float
        Weight for the spatial neighbourhood consistency term.
    lambda_pde : float
        Weight for the V3 full PDE loss (0 = V2 physics residual only).
    lambda_bc : float
        Weight for the boundary condition loss (V3).
    lambda_ic : float
        Weight for the initial condition loss (V3).
    lambda_jepa : float
        Weight for the JEPA self-supervised term.
    lambda_jepa_scenario : float
        Weight for the JEPA scenario-distillation pair term.
    lambda_jepa_latent_pde : float
        Weight for the JEPA latent-space PDE consistency term.
    lambda_alpha_class : float
        Weight for the alpha classification scaffold term.
    pde_loss_weights : Any
        V3 PDE loss weight object (passed through to ``sparc_joint_loss``).
    """

    thresholds: list
    resolution: float
    lambda_physics: float = 0.01
    lambda_smooth: float = 0.01
    lambda_alpha_smooth: float = 0.001
    lambda_prior: float = 0.01
    lambda_base: float = 0.2
    lambda_neighbor: float = 0.1
    lambda_pde: float = 0.0
    lambda_bc: float = 0.0
    lambda_ic: float = 0.0
    lambda_jepa: float = 0.0
    lambda_jepa_scenario: float = 0.0
    lambda_jepa_latent_pde: float = 0.0
    lambda_alpha_class: float = 0.0
    pde_loss_weights: Any = None

    # ------------------------------------------------------------------
    @classmethod
    def from_target_lambdas(
        cls,
        target_lambdas: dict,
        thresholds: list,
        resolution: float,
        *,
        pde_loss_weights: Any = None,
        lambda_ic: float = 0.0,
        lambda_jepa: float = 0.0,
        lambda_jepa_scenario: float = 0.0,
        lambda_jepa_latent_pde: float = 0.0,
    ) -> "JointLoss":
        """Build from the ``target_lambdas`` dict produced by ``train_neural_meta``.

        Parameters
        ----------
        target_lambdas : dict
            Keys: ``physics``, ``smooth``, ``alpha_smooth``, ``neighbor``,
            ``prior``, ``base``, ``pde``, ``bc``, ``alpha_class``.
        thresholds : list[float]
        resolution : float
        pde_loss_weights : optional V3 PDE weight object
        lambda_ic, lambda_jepa, lambda_jepa_scenario, lambda_jepa_latent_pde :
            Optional losses not represented in ``target_lambdas``.
        """
        return cls(
            thresholds=thresholds,
            resolution=resolution,
            lambda_physics=target_lambdas.get("physics", 0.01),
            lambda_smooth=target_lambdas.get("smooth", 0.01),
            lambda_alpha_smooth=target_lambdas.get("alpha_smooth", 0.001),
            lambda_prior=target_lambdas.get("prior", 0.01),
            lambda_base=target_lambdas.get("base", 0.2),
            lambda_neighbor=target_lambdas.get("neighbor", 0.1),
            lambda_pde=target_lambdas.get("pde", 0.0),
            lambda_bc=target_lambdas.get("bc", 0.0),
            lambda_ic=lambda_ic,
            lambda_jepa=lambda_jepa,
            lambda_jepa_scenario=lambda_jepa_scenario,
            lambda_jepa_latent_pde=lambda_jepa_latent_pde,
            lambda_alpha_class=target_lambdas.get("alpha_class", 0.0),
            pde_loss_weights=pde_loss_weights,
        )

    # ------------------------------------------------------------------
    def __call__(
        self,
        # Per-batch tensors
        T_pred: torch.Tensor,
        exceedance_preds: list,
        y_true: torch.Tensor,
        alpha: torch.Tensor,
        alpha_prior: torch.Tensor,
        neighbor_idx: torch.Tensor,
        source_term: torch.Tensor,
        surrogate_preds: list,
        surrogate_targets: list,
        epoch: int,
        # Optional per-batch tensors (V3 PDE / JEPA / etc.)
        h_field: "torch.Tensor | float | None" = None,
        alpha_prior_field: "torch.Tensor | None" = None,
        bc_specs: "dict | None" = None,
        water_mask: "torch.Tensor | None" = None,
        T_water: "float | None" = None,
        T0: "torch.Tensor | None" = None,
        T_prev: "torch.Tensor | None" = None,
        dt_hours: "float | None" = None,
        dT_dt_observed: "torch.Tensor | None" = None,
        nocturnal_dT_dt: "torch.Tensor | None" = None,
        T_night: "torch.Tensor | None" = None,
        jepa_h_pred: "torch.Tensor | None" = None,
        jepa_h_target: "torch.Tensor | None" = None,
        jepa_weights: Any = None,
        jepa_scenario_h_pred: "torch.Tensor | None" = None,
        jepa_scenario_h_target: "torch.Tensor | None" = None,
        jepa_latent_h: "torch.Tensor | None" = None,
        jepa_latent_neighbor_idx: "torch.Tensor | None" = None,
        jepa_latent_alpha: "torch.Tensor | None" = None,
        sparse_laplacian: "torch.Tensor | None" = None,
        valid_laplacian_mask: "torch.Tensor | None" = None,
        valid_mask: "torch.Tensor | None" = None,
        sheaf_delta: "torch.Tensor | None" = None,
        surr_fidelity_weights: "list[float] | None" = None,
        alpha_class_targets: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, dict[str, float]]":
        """Compute the joint loss for one training step.

        The alpha prior weight is automatically decayed from ``self.lambda_prior``
        using the curriculum schedule in :func:`get_prior_weight`.

        Returns
        -------
        total : scalar Tensor
        loss_components : dict[str, float]
        """
        lambda_prior_cur = get_prior_weight(epoch, self.lambda_prior)

        return sparc_joint_loss(
            T_pred=T_pred,
            exceedance_preds=exceedance_preds,
            y_true=y_true,
            thresholds=self.thresholds,
            alpha=alpha,
            alpha_prior=alpha_prior,
            neighbor_idx=neighbor_idx,
            source_term=source_term,
            resolution=self.resolution,
            surrogate_preds=surrogate_preds,
            surrogate_targets=surrogate_targets,
            lambda_physics=self.lambda_physics,
            lambda_smooth=self.lambda_smooth,
            lambda_alpha_smooth=self.lambda_alpha_smooth,
            lambda_prior=lambda_prior_cur,
            lambda_base=self.lambda_base,
            lambda_neighbor=self.lambda_neighbor,
            epoch=epoch,
            # V3 optional
            h_field=h_field,
            pde_loss_weights=self.pde_loss_weights,
            alpha_prior_field=alpha_prior_field,
            bc_specs=bc_specs,
            water_mask=water_mask,
            T_water=T_water,
            lambda_pde=self.lambda_pde,
            lambda_bc=self.lambda_bc,
            T0=T0,
            lambda_ic=self.lambda_ic,
            T_prev=T_prev,
            dt_hours=dt_hours,
            dT_dt_observed=dT_dt_observed,
            nocturnal_dT_dt=nocturnal_dT_dt,
            T_night=T_night,
            # JEPA optional
            jepa_h_pred=jepa_h_pred,
            jepa_h_target=jepa_h_target,
            jepa_weights=jepa_weights,
            lambda_jepa=self.lambda_jepa,
            jepa_scenario_h_pred=jepa_scenario_h_pred,
            jepa_scenario_h_target=jepa_scenario_h_target,
            lambda_jepa_scenario=self.lambda_jepa_scenario,
            jepa_latent_h=jepa_latent_h,
            jepa_latent_neighbor_idx=jepa_latent_neighbor_idx,
            jepa_latent_alpha=jepa_latent_alpha,
            lambda_jepa_latent_pde=self.lambda_jepa_latent_pde,
            # Misc
            sparse_laplacian=sparse_laplacian,
            valid_laplacian_mask=valid_laplacian_mask,
            valid_mask=valid_mask,
            sheaf_delta=sheaf_delta,
            surr_fidelity_weights=surr_fidelity_weights,
            alpha_class_targets=alpha_class_targets,
            lambda_alpha_class=self.lambda_alpha_class,
        )

    # ------------------------------------------------------------------
    def with_lambdas(self, **overrides: float) -> "JointLoss":
        """Return a copy with specific lambda weights overridden.

        Useful for CMA-ES runs that tune lambdas per fold::

            tuned = joint_loss.with_lambdas(lambda_physics=0.05)
        """
        import dataclasses
        return dataclasses.replace(self, **overrides)

    # ------------------------------------------------------------------
    def forward_bundled(
        self,
        T_pred: torch.Tensor,
        exceedance_preds: list,
        y_true: torch.Tensor,
        alpha: torch.Tensor,
        alpha_prior: torch.Tensor,
        physics: PhysicsContext,
        curriculum: CurriculumState,
        aux: AuxTargets,
    ) -> "tuple[torch.Tensor, dict[str, float]]":
        """Bundle-based entry point — structured alternative to ``__call__``.

        Accepts the three parameter bundles instead of 60+ flat kwargs.
        The ``JointLoss`` instance's stored lambda weights are used as
        *defaults* but are overridden by the values in ``curriculum``,
        so curriculum sweeps (e.g. CMA-ES) don't require rebuilding the
        ``JointLoss`` object.

        Parameters
        ----------
        T_pred : (N,) predicted outcome field
        exceedance_preds : list of (N,) per-threshold exceedance tensors
        y_true : (N,) target values
        alpha : (N,) or (N, 1) process rate field
        alpha_prior : (N,) or (N, 1) prior for alpha regularisation
        physics : :class:`PhysicsContext` — all spatial/PDE tensors
        curriculum : :class:`CurriculumState` — lambda weights + epoch
        aux : :class:`AuxTargets` — surrogate + JEPA targets

        Returns
        -------
        total : scalar Tensor — total loss for ``backward()``
        loss_components : dict[str, float] — per-term values for logging
        """
        lambda_prior_cur = get_prior_weight(curriculum.epoch, curriculum.lambda_prior)

        return sparc_joint_loss(
            T_pred=T_pred,
            exceedance_preds=exceedance_preds,
            y_true=y_true,
            thresholds=self.thresholds,
            alpha=alpha,
            alpha_prior=alpha_prior,
            neighbor_idx=physics.neighbor_idx,
            source_term=physics.source_term,
            resolution=physics.resolution,
            surrogate_preds=aux.surrogate_preds,
            surrogate_targets=aux.surrogate_targets,
            lambda_physics=curriculum.lambda_physics,
            lambda_smooth=curriculum.lambda_smooth,
            lambda_alpha_smooth=curriculum.lambda_alpha_smooth,
            lambda_prior=lambda_prior_cur,
            lambda_base=curriculum.lambda_base,
            lambda_neighbor=curriculum.lambda_neighbor,
            epoch=curriculum.epoch,
            # Physics context
            h_field=physics.h_field,
            pde_loss_weights=curriculum.pde_loss_weights,
            alpha_prior_field=physics.alpha_prior_field,
            bc_specs=physics.bc_specs,
            water_mask=physics.water_mask,
            T_water=physics.T_water,
            lambda_pde=curriculum.lambda_pde,
            lambda_bc=curriculum.lambda_bc,
            T0=physics.T0,
            lambda_ic=curriculum.lambda_ic,
            T_prev=physics.T_prev,
            dt_hours=physics.dt_hours,
            dT_dt_observed=physics.dT_dt_observed,
            nocturnal_dT_dt=physics.nocturnal_dT_dt,
            T_night=physics.T_night,
            sparse_laplacian=physics.sparse_laplacian,
            valid_laplacian_mask=physics.valid_laplacian_mask,
            valid_mask=physics.valid_mask,
            sheaf_delta=physics.sheaf_delta,
            # JEPA targets
            jepa_h_pred=aux.jepa_h_pred,
            jepa_h_target=aux.jepa_h_target,
            jepa_weights=aux.jepa_weights,
            lambda_jepa=aux.lambda_jepa,
            jepa_scenario_h_pred=aux.jepa_scenario_h_pred,
            jepa_scenario_h_target=aux.jepa_scenario_h_target,
            lambda_jepa_scenario=aux.lambda_jepa_scenario,
            jepa_latent_h=aux.jepa_latent_h,
            jepa_latent_neighbor_idx=aux.jepa_latent_neighbor_idx,
            jepa_latent_alpha=aux.jepa_latent_alpha,
            lambda_jepa_latent_pde=aux.lambda_jepa_latent_pde,
            # Misc
            surr_fidelity_weights=aux.surr_fidelity_weights,
            alpha_class_targets=aux.alpha_class_targets,
            lambda_alpha_class=curriculum.lambda_alpha_class,
        )
