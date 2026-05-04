"""
Latent predictor for JEPA-style training.

Maps a *context* embedding to a *target* embedding in the same latent
space.  In Phase 1 this is a small MLP used purely for the masked-input
self-supervised objective:

    h_context = trunk(physics_feats with masked channels)
    h_target  = ema_trunk(physics_feats unmasked)        [stop-grad]
    h_pred    = LatentPredictor(h_context)
    loss      = jepa_loss(h_pred, h_target)

In Phase 2 the same module is extended with action conditioning
(intervention vector + Δt) via FiLM (feature-wise linear modulation)
and an optional deeper residual stack.  Backward-compatible defaults
preserve Phase 1 behavior:

    LatentPredictor(hidden_dim, action_dim=0)            # Phase 1 MLP
    LatentPredictor(hidden_dim, action_dim=64, n_blocks=4, film=True)  # Phase 2

The "transformer over the point set" called out in the plan is
unnecessary in SPARC's flat per-point scheme — spatial mixing already
lives in ``CityHead.spatial_enc``.  A FiLM-conditioned residual block
stack delivers the action conditioning the predictor actually needs
without redundant attention layers.

All modules are dimension-preserving:  R^(N×H) → R^(N×H).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _FiLMBlock(nn.Module):
    """Pre-LN residual MLP block with optional FiLM action conditioning.

    Each block applies:
        u = LayerNorm(x)
        if film: u = (1 + γ(a)) ⊙ u + β(a)
        x = x + Linear(GELU(Linear(u)))

    γ, β are produced by a per-block Linear(action_dim → 2H) so each
    block can adapt the predictor's response to the current action.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        expansion: int,
        dropout: float,
        film: bool,
    ) -> None:
        super().__init__()
        inner = expansion * hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = film and action_dim > 0
        if self.film:
            self.film_proj = nn.Linear(action_dim, 2 * hidden_dim)
            # Zero-init γ and β so the block starts as a no-op
            # modulation — the predictor begins at near-identity.
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )
        # Zero-init the residual branch's final projection so the entire
        # block starts as identity (matches Phase 1 init discipline).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None,
    ) -> torch.Tensor:
        u = self.norm(x)
        if self.film and action is not None:
            gamma_beta = self.film_proj(action)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            u = (1.0 + gamma) * u + beta
        return x + self.mlp(u)


class LatentPredictor(nn.Module):
    """
    Per-point latent predictor in the SharedTrunk latent space.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the trunk embedding (matches ``SPARCMetaLearner.hidden_dim``).
    action_dim : int
        Dimension of the optional action embedding.  Set to 0 (Phase 1
        default) to disable action conditioning.
    expansion : int
        Inner MLP width = ``expansion * hidden_dim``.
    dropout : float
        Dropout probability applied between MLP layers.
    n_blocks : int
        Number of residual blocks.  ``1`` reproduces the Phase 1 single
        MLP block.  ``≥2`` stacks FiLM-conditioned blocks for the
        action-conditioned predictor.
    film : bool
        When True and ``action_dim > 0``, condition each block via FiLM.
        When False, action is concatenated to the input of the first
        block (Phase 1 behavior).

    Notes
    -----
    Pre-LayerNorm + residual is used to match the trunk's normalization
    discipline.  Final-projection / FiLM matrices are zero-initialized
    so the predictor starts at the identity (h_pred == h_context),
    which empirically reduces collapse risk during the JEPA warmup
    epochs.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 0,
        expansion: int = 2,
        dropout: float = 0.1,
        n_blocks: int = 1,
        film: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if action_dim < 0:
            raise ValueError("action_dim must be non-negative.")
        if expansion < 1:
            raise ValueError("expansion must be >= 1.")
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1.")

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.n_blocks = n_blocks
        self.use_film = bool(film and action_dim > 0)

        self.action_norm = nn.LayerNorm(action_dim) if action_dim > 0 else None

        if self.use_film:
            # All action conditioning happens inside FiLM — input is just
            # the context embedding.
            self.input_proj: nn.Module = nn.Identity()
            block_action_dim = action_dim
        else:
            # Phase 1 backward-compat path: concat action to input then
            # project back to hidden_dim before the residual blocks.
            in_dim = hidden_dim + action_dim
            if action_dim > 0:
                self.input_proj = nn.Linear(in_dim, hidden_dim)
                # Zero-init the action half of the projection so the
                # predictor starts at the identity even with concat.
                with torch.no_grad():
                    self.input_proj.weight[:, hidden_dim:].zero_()
                    self.input_proj.bias.zero_()
            else:
                self.input_proj = nn.Identity()
            block_action_dim = 0

        self.blocks = nn.ModuleList(
            [
                _FiLMBlock(
                    hidden_dim=hidden_dim,
                    action_dim=block_action_dim,
                    expansion=expansion,
                    dropout=dropout,
                    film=self.use_film,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context : (N, hidden_dim)
            The online trunk embedding for the context (masked) inputs.
        action : (N, action_dim) or None
            Optional per-point action embedding.

        Returns
        -------
        h_pred : (N, hidden_dim)
            Predicted target embedding.
        """
        if context.dim() != 2 or context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"context must be (N, {self.hidden_dim}); got {tuple(context.shape)}",
            )

        if self.action_dim > 0:
            if action is None:
                action = torch.zeros(
                    context.shape[0], self.action_dim,
                    device=context.device, dtype=context.dtype,
                )
            elif action.shape != (context.shape[0], self.action_dim):
                raise ValueError(
                    f"action must be (N={context.shape[0]}, {self.action_dim}); "
                    f"got {tuple(action.shape)}",
                )
            action = self.action_norm(action)
        elif action is not None:
            raise ValueError(
                "action_dim is 0 but action tensor was provided; "
                "construct LatentPredictor(action_dim=...) to enable conditioning.",
            )

        if self.use_film:
            x = self.input_proj(context)
            block_action = action
        else:
            if self.action_dim > 0:
                x = self.input_proj(torch.cat([context, action], dim=-1))
            else:
                x = self.input_proj(context)
            block_action = None

        residual_in = context  # final residual is added at the end
        h = x
        for blk in self.blocks:
            h = blk(h, block_action)

        # Final residual: predictor learns the *update* to context.
        # When all blocks zero-init, h == x ≈ context (identity start).
        if self.use_film:
            return residual_in + (h - x)
        return residual_in + (h - x) if self.action_dim > 0 else h
