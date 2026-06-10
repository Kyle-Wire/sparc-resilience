"""JEPACurriculum — JEPA Phase-1 warm-up as a standalone testable module.

Concentrates all Phase-1 JEPA logic that was previously inline inside the
v2_neural_training fold loop:

  - EMA target trunk initialisation
  - Per-step online encode → target encode → VICReg alignment loss
  - EMA momentum update
  - Step counter
  - Checkpoint serialisation / deserialisation

Phase 2 extensions (C1 deepening):

  - ``step_scenario``     — action-conditioned scenario distillation
  - ``step_latent_pde``   — latent Laplacian consistency loss
  - ``warm_up``           — multi-epoch convenience loop

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
    """Owns Phase-1 and Phase-2 JEPA warm-up: one step at a time, checkpointable.

    Parameters
    ----------
    online    : nn.Module with an ``encode(physics_feats, alpha)`` method
    ema       : EMATrunk wrapping *online*
    optimizer : torch Optimizer configured over *online* parameters
    predictor : LatentPredictor (optional) — required for step_scenario
    """

    def __init__(
        self,
        online: nn.Module,
        ema: Any,
        optimizer: torch.optim.Optimizer,
        predictor: Any = None,
    ) -> None:
        self.online = online
        self.ema = ema
        self.optimizer = optimizer
        self.predictor = predictor
        self.step_count: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_model(cls, online: nn.Module) -> "JEPACurriculum":
        """Convenience factory: wraps *online* in an EMATrunk, LatentPredictor, and Adam."""
        from sparc.models.ema_trunk import EMATrunk
        from sparc.models.latent_predictor import LatentPredictor

        ema = EMATrunk(online, warmup_steps=1_000)
        hidden_dim = _infer_hidden_dim(online)
        predictor = LatentPredictor(hidden_dim=hidden_dim)
        params = list(online.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return cls(online, ema, optimizer, predictor=predictor)

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
    def step_scenario(
        self,
        batch: dict[str, torch.Tensor],
        action_embed: Any,
        perturb_std: float = 0.1,
    ) -> float:
        """Phase 2: scenario-distillation step.

        Perturbs physics features with Gaussian noise, encodes both the
        original (via EMA stop-grad) and perturbed (via online encoder),
        then passes the perturbed encoding through the latent predictor
        conditioned on an action embedding and asks it to predict the
        EMA encoding of the original view.

        Parameters
        ----------
        batch       : dict with ``physics_feats`` and ``alpha`` keys
        action_embed: ActionEmbedding instance
        perturb_std : std of Gaussian physics perturbation

        Returns
        -------
        loss : float
        """
        if self.predictor is None:
            raise RuntimeError(
                "step_scenario requires a predictor; use JEPACurriculum.from_model()."
            )
        physics_feats: torch.Tensor = batch["physics_feats"]
        alpha_raw: torch.Tensor = batch["alpha"]
        if alpha_raw.dim() == 1:
            alpha = alpha_raw.unsqueeze(-1)
        elif alpha_raw.shape[-1] > 1:
            alpha = alpha_raw[:, :1]
        else:
            alpha = alpha_raw

        self.online.train()

        # Perturb physics context
        perturbed = physics_feats + perturb_std * torch.randn_like(physics_feats)

        # Online → encode perturbed context
        h_context = self.online.encode(perturbed, alpha)  # (B, H)

        # EMA → encode original view as target (stop-grad)
        h_target = self.ema.encode_target(physics_feats, alpha)  # (B, H)

        # Build a no-op action signal (none treatment, zero magnitudes)
        B = physics_feats.shape[0]
        device = physics_feats.device
        none_idx = action_embed.none_idx
        treatment_idx = torch.full((B,), none_idx, dtype=torch.long, device=device)
        delta_x = torch.zeros(B, device=device)
        delta_t = torch.zeros(B, device=device)
        a_vec = action_embed(treatment_idx, delta_x, delta_t)  # (B, action_dim)

        # Latent prediction — only pass action if the predictor expects it
        a_input = a_vec if (self.predictor.action_dim > 0) else None
        h_pred = self.predictor(h_context, a_input)  # (B, H)

        from sparc.training.jepa_loss import jepa_loss
        loss_tensor, _ = jepa_loss(h_pred, h_target.detach())

        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()
        self.ema.update(self.online)

        self.step_count += 1
        return float(loss_tensor.item())

    # ------------------------------------------------------------------
    def step_latent_pde(
        self,
        batch: dict[str, torch.Tensor],
        neighbor_idx: torch.Tensor,
    ) -> float:
        """Phase 2: latent Laplacian consistency step.

        Encourages spatially-neighbouring points to have similar trunk
        embeddings — a soft PDE prior in latent space.

        Parameters
        ----------
        batch        : dict with ``physics_feats`` and ``alpha`` keys
        neighbor_idx : (N, k) long tensor of KNN indices

        Returns
        -------
        loss : float (Laplacian smoothness, >= 0)
        """
        physics_feats: torch.Tensor = batch["physics_feats"]
        alpha_raw: torch.Tensor = batch["alpha"]
        if alpha_raw.dim() == 1:
            alpha = alpha_raw.unsqueeze(-1)
        elif alpha_raw.shape[-1] > 1:
            alpha = alpha_raw[:, :1]
        else:
            alpha = alpha_raw

        self.online.train()
        h = self.online.encode(physics_feats, alpha)  # (N, H)

        # Laplacian smoothness: mean squared difference to each neighbour
        h_neighbors = h[neighbor_idx]  # (N, k, H)
        diff = h.unsqueeze(1) - h_neighbors  # (N, k, H)
        loss_tensor = (diff ** 2).mean()

        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()
        self.ema.update(self.online)

        self.step_count += 1
        return float(loss_tensor.item())

    # ------------------------------------------------------------------
    def warm_up(
        self,
        batches: "list[dict[str, torch.Tensor]]",
        epochs: int = 1,
    ) -> dict[str, float]:
        """Multi-epoch warm-up loop over *batches*.

        Parameters
        ----------
        batches : list of batch dicts (each with ``physics_feats`` + ``alpha``)
        epochs  : number of passes over *batches* (default 1)

        Returns
        -------
        metrics : dict with ``"avg_loss"`` key
        """
        if epochs <= 0 or len(batches) == 0:
            return {"avg_loss": 0.0}

        total, count = 0.0, 0
        for _ in range(epochs):
            for batch in batches:
                total += self.step(batch)
                count += 1
        return {"avg_loss": total / count}

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_hidden_dim(model: nn.Module) -> int:
    """Infer hidden_dim from the first parameter of *model* that encodes it."""
    # SPARCMetaLearner exposes trunk_fusion as a Sequential whose first Linear
    # output is hidden_dim.  Fallback to 64 when the attribute is missing.
    fusion = getattr(model, "trunk_fusion", None)
    if fusion is not None and hasattr(fusion, "__iter__"):
        for layer in fusion:
            if hasattr(layer, "out_features"):
                return layer.out_features
    # Generic fallback: use the output size of the last Linear layer found
    for module in reversed(list(model.modules())):
        if hasattr(module, "out_features"):
            return module.out_features
    return 64
