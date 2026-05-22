"""
ModelBundle — factory for the full set of V2 neural models.

Groups the five objects that every CV fold instantiates from the same
:class:`_ArchConfig` into a single typed container, and exposes a
:meth:`ModelBundle.create` classmethod that mirrors the construction
logic previously scattered across ``_exec_cv_fold``.

Usage::

    from sparc.models.bundle import ModelBundle

    bundle = ModelBundle.create(ss.arch, device)
    model        = bundle.model
    process_net  = bundle.process_net
    source_net   = bundle.source_net
    surrogates   = bundle.surrogates   # {"gwr": ..., "gwrf": ..., "ggpgam": ...}

The caller remains responsible for:
  - Compiling models via ``_maybe_compile`` (torch.compile, optional)
  - Loading / transferring pretrained trunk weights
  - Moving models to a different device after the fact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    # Avoid circular imports at runtime; the real imports happen inside create().
    from sparc.models.neural_meta import SPARCMetaLearner
    from sparc.models.process_rate_net import ProcessRateNet, SourceTermNet
    from sparc.models.surrogates import (
        DifferentiableGGPGAM,
        DifferentiableGWR,
        DifferentiableGWRF,
    )


@dataclass
class ModelBundle:
    """Container for the five neural models that constitute one SPARC V2 fold.

    Attributes
    ----------
    model : SPARCMetaLearner
        The SharedTrunk + CityHead meta-learner.
    process_net : ProcessRateNet
        Auxiliary network that predicts the spatial process rate α.
    source_net : SourceTermNet
        Auxiliary network for the PDE source term.
    surrogates : dict[str, nn.Module]
        Keyed by ``"gwr"``, ``"gwrf"``, ``"ggpgam"``.
    """

    model: "SPARCMetaLearner"
    process_net: "ProcessRateNet"
    source_net: "SourceTermNet"
    surrogates: dict[str, nn.Module] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        arch: Any,  # sparc.run.v2_neural_training._ArchConfig
        device: torch.device,
    ) -> "ModelBundle":
        """Instantiate and move all models to *device*.

        Parameters
        ----------
        arch : _ArchConfig
            Architecture configuration produced by ``train_neural_meta``.
        device : torch.device
            Target device (CPU or CUDA).

        Returns
        -------
        ModelBundle
            All five models are on *device* and ready for training.
        """
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.models.process_rate_net import ProcessRateNet, SourceTermNet
        from sparc.models.surrogates import (
            DifferentiableGGPGAM,
            DifferentiableGWR,
            DifferentiableGWRF,
        )

        process_net = ProcessRateNet(
            n_inputs=arch.pr_input_dim,
            domain_config={
                "name": arch.pr_cfg.get("name", "rate"),
                "units": arch.pr_cfg.get("units", ""),
                "bounds": arch.pr_cfg.get("bounds", [0.0, 1.0]),
                "prior_mean": arch.pr_cfg.get("prior_mean", 0.5),
            },
            n_treatments=arch.n_treatments,
        ).to(device)

        source_net = SourceTermNet(n_inputs=arch.n_physics).to(device)

        init_bw = (
            float(arch.predictor_bandwidths.mean())
            if arch.predictor_bandwidths is not None
            else 1000.0
        )
        model = SPARCMetaLearner(
            n_base_models=arch.n_base,
            n_physics_features=arch.n_physics_extended,
            d_spatial=arch.d_spatial,
            hidden_dim=arch.hidden_dim,
            dropout=arch.dropout,
            thresholds=arch.thresholds,
            n_heads=arch.n_heads,
            max_neighbors=arch.max_neighbors,
            siren_omega=arch.siren_omega,
            init_bandwidth=init_bw,
            time_embed_dim=arch.time_embed_dim,
        ).to(device)

        surrogates: dict[str, nn.Module] = {
            "gwr": DifferentiableGWR(
                n_vars=arch.n_physics,
                n_spatial_features=arch.d_spatial,
                hidden_dim=arch.hidden_dim,
                bandwidths=arch.predictor_bandwidths,
                kernel_field=arch.predictor_kernel_field,
                feature_names=arch.feature_names,
            ).to(device),
            "gwrf": DifferentiableGWRF(
                n_vars=arch.n_physics,
                n_spatial_features=arch.d_spatial,
                hidden_dim=arch.hidden_dim,
                kernel_field=arch.predictor_kernel_field,
                feature_names=arch.feature_names,
            ).to(device),
            "ggpgam": DifferentiableGGPGAM(
                n_vars=arch.n_physics,
                n_spatial_features=arch.d_spatial,
                hidden_dim=min(arch.hidden_dim, 64),
            ).to(device),
        }

        return cls(
            model=model,
            process_net=process_net,
            source_net=source_net,
            surrogates=surrogates,
        )

    # ------------------------------------------------------------------
    def all_modules(self) -> list[nn.Module]:
        """Return all modules in a stable order (model, process_net, source_net, surrogates)."""
        return [
            self.model,
            self.process_net,
            self.source_net,
            *self.surrogates.values(),
        ]

    def train(self) -> "ModelBundle":
        """Set all modules to training mode."""
        for m in self.all_modules():
            m.train()
        return self

    def eval(self) -> "ModelBundle":
        """Set all modules to evaluation mode."""
        for m in self.all_modules():
            m.eval()
        return self

    def parameters(self):
        """Iterate over all parameters across all modules."""
        for m in self.all_modules():
            yield from m.parameters()
