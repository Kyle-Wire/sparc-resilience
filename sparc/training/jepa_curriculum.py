"""JEPACurriculum — JEPA Phase-1 warm-up as a standalone testable module.

Concentrates all Phase-1 JEPA logic that was previously inline inside the
v2_neural_training fold loop:

  - EMA target trunk initialisation
  - Per-step online encode → target encode → VICReg alignment loss
  - EMA momentum update
  - Step counter
  - Checkpoint serialisation / deserialisation

The training file becomes a thin orchestrator that calls
``curriculum.warm_up(loader, epochs)`` or ``curriculum.step(batch)``.

Usage
-----
>>> curriculum = JEPACurriculum.from_model(online_model)
>>> for batch in loader:
...     loss = curriculum.step(batch)
>>> ckpt = curriculum.checkpoint()
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class JEPACurriculum:
    """Owns Phase-1 JEPA warm-up: one step at a time, checkpointable.

    Parameters
    ----------
    online    : nn.Module with an ``encode(physics_feats, alpha)`` method
    ema       : EMATrunk wrapping *online*
    optimizer : torch Optimizer configured over *online* parameters
    """

    def __init__(
        self,
        online: nn.Module,
        ema: Any,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.online = online
        self.ema = ema
        self.optimizer = optimizer
        self.step_count: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_model(cls, online: nn.Module) -> "JEPACurriculum":
        """Convenience factory: wraps *online* in an EMATrunk and Adam."""
        from sparc.models.ema_trunk import EMATrunk

        ema = EMATrunk(online, warmup_steps=1_000)
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        return cls(online, ema, optimizer)

    # ------------------------------------------------------------------
    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """Run one JEPA training step.

        Expected batch keys
        -------------------
        physics_feats : (B, n_physics)
        alpha         : (B, *) — process rate; first column used if multi-dim

        Returns
        -------
        loss : float — scalar JEPA loss for this step
        """
        physics_feats: torch.Tensor = batch["physics_feats"]
        alpha_raw: torch.Tensor = batch["alpha"]

        # alpha must be (B, 1) for SPARCMetaLearner.encode
        if alpha_raw.dim() == 1:
            alpha = alpha_raw.unsqueeze(-1)
        elif alpha_raw.shape[-1] > 1:
            alpha = alpha_raw[:, :1]
        else:
            alpha = alpha_raw

        self.online.train()

        # Online embedding (gradients flow)
        h_online = self.online.encode(physics_feats, alpha)   # (B, H)

        # Target embedding (stop-grad)
        h_target = self.ema.encode_target(physics_feats, alpha)  # (B, H)

        # VICReg-style JEPA loss
        from sparc.training.jepa_loss import jepa_loss

        loss_tensor, _ = jepa_loss(h_online, h_target.detach())

        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()

        # EMA update
        self.ema.update(self.online)

        self.step_count += 1
        return float(loss_tensor.item())

    # ------------------------------------------------------------------
    def checkpoint(self) -> dict[str, Any]:
        """Return a serialisable checkpoint dict.

        Keys
        ----
        step      : int — number of completed steps
        online    : state_dict of the online model
        predictor : state_dict of the EMA target model
        """
        return {
            "step": self.step_count,
            "online": self.online.state_dict(),
            "predictor": self.ema.target_model.state_dict(),
        }
