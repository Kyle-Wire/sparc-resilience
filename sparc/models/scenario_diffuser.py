"""DDPM-based latent scenario diffuser for SPARCMetaLearner trunk posterior sampling.

Learns the distribution of trunk embeddings produced by the SharedTrunk
(h_trunk = trunk_fusion(physics_enc + alpha_emb)), then enables posterior
sampling of alternative trunk realizations given optional physics conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentScenarioDiffuser(nn.Module):
    """DDPM prior over the SPARCMetaLearner trunk latent space.

    Parameters
    ----------
    trunk_dim : int
        Dimension of the trunk embedding (must match SPARCMetaLearner.hidden_dim).
        Default 256.
    bottleneck : int
        Latent bottleneck dimension for the diffusion process. Default 32.
    cond_dim : int
        Dimension of conditioning features (e.g. n_physics_original).
        Set to 0 for unconditional mode. Default 0.
    T : int
        Number of DDPM diffusion timesteps. Default 200.
    """

    def __init__(
        self,
        trunk_dim: int = 256,
        bottleneck: int = 32,
        cond_dim: int = 0,
        T: int = 200,
    ) -> None:
        super().__init__()
        self.trunk_dim = trunk_dim
        self.bottleneck = bottleneck
        self.cond_dim = cond_dim
        self.T = T

        # Linear noise schedule — registered as buffers (serialized with state_dict)
        betas = torch.linspace(1e-4, 0.02, T)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

        # Projection layers between trunk space and bottleneck
        self.proj_down = nn.Linear(trunk_dim, bottleneck)
        self.proj_up = nn.Linear(bottleneck, trunk_dim)

        # Optional conditioning projection
        self.cond_proj: nn.Linear | None = None
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, bottleneck)

        # 3-layer SiLU noise predictor
        # Input: z_t (bottleneck) || cond_embed (bottleneck) || t_norm (1)
        net_in = bottleneck * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(net_in, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, bottleneck),
        )

    # ------------------------------------------------------------------

    def _cond_embed(
        self,
        cond: torch.Tensor | None,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return (N, bottleneck) conditioning vector, zeros when unconditional."""
        if self.cond_proj is not None and cond is not None:
            c = self.cond_proj(cond)           # (N, bottleneck) or (1, bottleneck)
            return c.expand(N, -1)
        return torch.zeros(N, self.bottleneck, device=device)

    # ------------------------------------------------------------------

    def loss(
        self,
        trunk_feats: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDPM denoising loss on trunk embeddings.

        Parameters
        ----------
        trunk_feats : (N, trunk_dim)
            Trunk embeddings from model.encode().
        cond : (N, cond_dim) or (1, cond_dim) or None
            Physics conditioning features. None → unconditional.

        Returns
        -------
        loss : scalar tensor
        """
        z0 = self.proj_down(trunk_feats)          # (N, bottleneck)
        N = z0.size(0)
        device = z0.device

        t = torch.randint(0, self.T, (N,), device=device)   # (N,)
        ab = self.alpha_bars[t].unsqueeze(1)                 # (N, 1)
        eps = torch.randn_like(z0)
        z_t = ab.sqrt() * z0 + (1.0 - ab).sqrt() * eps      # (N, bottleneck)

        c = self._cond_embed(cond, N, device)                # (N, bottleneck)
        t_n = t.float().unsqueeze(1) / self.T                # (N, 1)
        eps_pred = self.net(torch.cat([z_t, c, t_n], dim=1))

        return F.mse_loss(eps_pred, eps)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor | None,
        n_samples: int = 50,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw n_samples from the DDPM posterior via reverse diffusion.

        Parameters
        ----------
        cond : (M, cond_dim) or None
            Conditioning features. Spatial mean is taken → (1, cond_dim),
            then projected to (n_samples, bottleneck) and broadcast.
        n_samples : int
            Number of trunk samples to return.
        device : torch.device or None
            Target device. Defaults to the module's first parameter device.

        Returns
        -------
        samples : (n_samples, trunk_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n_samples, self.bottleneck, device=device)

        if self.cond_proj is not None and cond is not None:
            # Summarise spatial conditioning to a global mean
            cond_dev = cond.to(device)
            cond_mean = cond_dev.mean(0, keepdim=True)              # (1, cond_dim)
            c = self.cond_proj(cond_mean).expand(n_samples, -1)    # (n_samples, bottleneck)
        else:
            c = torch.zeros(n_samples, self.bottleneck, device=device)

        for t_idx in reversed(range(self.T)):
            t_n = torch.full((n_samples, 1), t_idx / self.T, device=device)
            eps = self.net(torch.cat([z, c, t_n], dim=1))
            ab = self.alpha_bars[t_idx]
            z = (z - (1.0 - ab).sqrt() * eps) / ab.sqrt()
            if t_idx > 0:
                z = z + self.betas[t_idx].sqrt() * torch.randn_like(z)

        return self.proj_up(z)    # (n_samples, trunk_dim)
