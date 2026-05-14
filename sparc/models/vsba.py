"""
Variational Spatial Block Autoencoder (VSBA) for label-free CV fold quality.

Architecture:
    Encoder:  X (N, F) → μ (N, D), log σ² (N, D)
    Decoder:  z (N, D) → X̂ (N, F)
    Prior:    z ~ N(0, K_Matérn(coords, ν, ρ))

The KL term uses the diagonal Matérn approximation:
    KL = 0.5 * Σ_i [ μ_i² / k_ii + σ²_i / k_ii - log(σ²_i / k_ii) - 1 ]

where k_ii = K_Matérn(coords_i, coords_i) = σ² (nugget-free = 1 after
normalisation). In practice we evaluate the Matérn correlation only on a
sparse nearest-neighbour graph (default K=8) to keep the computation O(N·K)
instead of O(N²).

Usage (standalone):
    vsba = VariationalSpatialBlockAutoencoder.from_correlogram(corr_payload)
    vsba.fit(X_train, coords_train, n_epochs=20)
    score = vsba.elbo_score(X_test, coords_test)   # higher = more in-dist

Usage (integrated, via _vsba_fold_quality_score in enhanced_spatial_cv.py):
    Enabled by config key ``models.spatial_cv.vsba_fold_scoring: true``.
    Trained on the union of all training folds; scored per test fold.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Matérn kernel helpers (O(N·K), no dense N×N matrix)
# ---------------------------------------------------------------------------

def _matern_kii(nu: float) -> float:
    """K_Matérn(x, x) = 1 (unit variance, normalised). Returns 1.0 always."""
    return 1.0


def _matern_kij(
    dists: torch.Tensor,
    nu: float,
    rho: float,
) -> torch.Tensor:
    """Evaluate Matérn(ν, ρ) correlation for a tensor of distances.

    ν ∈ {0.5, 1.5, 2.5, ∞} supported. Falls back to ν=1.5 for other values.
    rho : length-scale in the same units as dists.
    """
    d = dists / max(rho, 1e-6)

    if abs(nu - 0.5) < 0.1:
        # Ornstein-Uhlenbeck: exp(-d)
        return torch.exp(-d)
    elif abs(nu - 1.5) < 0.1:
        # Matérn 3/2: (1 + √3 d) exp(-√3 d)
        s3d = math.sqrt(3.0) * d
        return (1.0 + s3d) * torch.exp(-s3d)
    elif abs(nu - 2.5) < 0.1:
        # Matérn 5/2: (1 + √5 d + 5d²/3) exp(-√5 d)
        s5d = math.sqrt(5.0) * d
        return (1.0 + s5d + s5d ** 2 / 3.0) * torch.exp(-s5d)
    else:
        # Squared-exponential (ν → ∞): exp(-d²/2)
        return torch.exp(-0.5 * d ** 2)


def _build_sparse_prior_precision(
    coords: torch.Tensor,
    nu: float,
    rho: float,
    k_neighbors: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (nn_idx, nn_corr) for a K-NN sparse Matérn prior.

    nn_idx  : (N, K) long  — indices of K nearest neighbours (excluding self)
    nn_corr : (N, K) float — Matérn correlation to each neighbour

    Used to compute the approximate KL via the local neighbourhood
    structure rather than a full N×N inversion.
    """
    from sklearn.neighbors import NearestNeighbors

    coords_np = coords.cpu().numpy()
    k_actual = min(k_neighbors, len(coords_np) - 1)
    if k_actual < 1:
        N = len(coords_np)
        idx = torch.zeros(N, 1, dtype=torch.long, device=coords.device)
        corr = torch.zeros(N, 1, dtype=torch.float32, device=coords.device)
        return idx, corr

    nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="ball_tree").fit(coords_np)
    dists_np, idx_np = nn.kneighbors(coords_np)
    # Drop self (distance = 0 is always index 0 when coords are unique)
    dists_np = dists_np[:, 1:]
    idx_np = idx_np[:, 1:]

    dists_t = torch.tensor(dists_np, dtype=torch.float32, device=coords.device)
    idx_t = torch.tensor(idx_np, dtype=torch.long, device=coords.device)
    corr_t = _matern_kij(dists_t, nu, rho)

    return idx_t, corr_t


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, n_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# VSBA
# ---------------------------------------------------------------------------

class VariationalSpatialBlockAutoencoder(nn.Module):
    """VAE with a sparse Matérn GP prior on the latent space.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature space.
    latent_dim : int
        Latent space dimensionality. Default 16.
    hidden_dim : int
        Hidden layer width in encoder/decoder. Default 64.
    nu : float
        Matérn smoothness parameter (0.5, 1.5, 2.5, or inf). Default 1.5.
    rho : float
        Matérn length-scale in world coordinate units (metres). Default 500.
    k_neighbors : int
        Neighbours for sparse prior approximation. Default 8.
    beta : float
        KL weight (β-VAE). Default 1.0.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        nu: float = 1.5,
        rho: float = 500.0,
        k_neighbors: int = 8,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.nu = float(nu)
        self.rho = float(rho)
        self.k_neighbors = int(k_neighbors)
        self.beta = float(beta)

        self.encoder = _Encoder(n_features, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, n_features, hidden_dim)

    # ------------------------------------------------------------------
    # Factory from Stage 0 correlogram payload
    # ------------------------------------------------------------------

    @classmethod
    def from_correlogram(
        cls,
        corr_payload: dict | None,
        n_features: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        k_neighbors: int = 8,
        beta: float = 1.0,
    ) -> "VariationalSpatialBlockAutoencoder":
        """Build VSBA with Matérn parameters seeded from a Stage 0 payload.

        corr_payload format (from MaternCorrelogramResult.to_payload()):
            {"nu": float, "kappa": {"mean": float}, ...}
        kappa is the inverse length-scale (1/ρ) in the Matérn parameterisation.
        """
        nu = 1.5
        rho = 500.0
        if corr_payload:
            nu = float(corr_payload.get("nu", 1.5))
            kappa_mean = (
                (corr_payload.get("kappa") or {}).get("mean")
                or corr_payload.get("kappa")
            )
            if kappa_mean is not None:
                try:
                    kappa_float = float(kappa_mean)
                    if kappa_float > 0:
                        rho = 1.0 / kappa_float  # kappa = 1/rho
                except (TypeError, ValueError):
                    pass
        return cls(
            n_features=n_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nu=nu,
            rho=rho,
            k_neighbors=k_neighbors,
            beta=beta,
        )

    # ------------------------------------------------------------------
    # Forward / ELBO
    # ------------------------------------------------------------------

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        nn_idx: torch.Tensor | None = None,
        nn_corr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (recon_x, mu, logvar)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def elbo_loss(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        nn_idx: torch.Tensor | None = None,
        nn_corr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute −ELBO / N (scalar, minimise this during training).

        KL term uses the diagonal Matérn GP approximation:
            KL_i = 0.5 * [μ_i²/k_ii + σ²_i/k_ii − log(σ²_i/k_ii) − 1]
        With unit variance GP prior (k_ii = 1):
            KL_i = 0.5 * [μ_i² + σ²_i − log σ²_i − 1]
        plus a spatial smoothness term:
            KL_spatial_i = 0.5 * Σ_j corr_ij * μ_i · μ_j    (Gaussian MRF)
        """
        recon, mu, logvar = self.forward(x, coords, nn_idx, nn_corr)
        var = logvar.exp()

        # Reconstruction: mean squared error (Gaussian decoder)
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.shape[0]

        # Standard diagonal KL: N(μ, σ²) || N(0, I)
        kl_diag = 0.5 * (mu.pow(2) + var - logvar - 1.0)  # (N, D)
        kl = kl_diag.sum(dim=-1).mean()  # scalar

        # Spatial smoothness: penalise μ disagreeing with neighbours
        # ∑_j corr_ij (μ_i - μ_j)² / 2  (Matérn MRF approximation)
        if nn_idx is not None and nn_corr is not None and nn_idx.shape[1] > 0:
            mu_nb = mu[nn_idx]  # (N, K, D)
            diff_sq = (mu.unsqueeze(1) - mu_nb).pow(2).sum(dim=-1)  # (N, K)
            spatial_kl = (nn_corr * diff_sq).sum(dim=-1).mean() * 0.5
        else:
            spatial_kl = torch.tensor(0.0, device=x.device)

        return recon_loss + self.beta * (kl + spatial_kl)

    # ------------------------------------------------------------------
    # Fit / Score API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str | torch.device | None = None,
    ) -> "VariationalSpatialBlockAutoencoder":
        """Train on feature matrix X (N, F) with coordinates coords (N, 2)."""
        if device is None:
            device = torch.device("cpu")
        self.to(device)
        self.train()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

        # Precompute sparse prior structure once
        nn_idx, nn_corr = _build_sparse_prior_precision(
            coords_t, self.nu, self.rho, self.k_neighbors
        )

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        N = len(X_t)
        rng = np.random.default_rng(42)

        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                b_idx = perm[start : start + batch_size]
                b_idx_t = torch.tensor(b_idx, dtype=torch.long, device=device)
                bx = X_t[b_idx_t]
                bc = coords_t[b_idx_t]
                b_nn_idx = nn_idx[b_idx_t]  # (B, K) — neighbour global idx
                # Remap neighbour indices to batch-local space; fall back to
                # -1 (masked out) for neighbours not in the current batch.
                b_nn_corr = nn_corr[b_idx_t]  # (B, K)

                # Build index map: global → batch position
                idx_map = torch.full(
                    (N,), -1, dtype=torch.long, device=device
                )
                idx_map[b_idx_t] = torch.arange(len(b_idx), dtype=torch.long, device=device)
                b_nn_local = idx_map[b_nn_idx.clamp(0, N - 1)]  # (B, K)
                # Mask edges to out-of-batch neighbours
                valid_nb = b_nn_local >= 0
                b_nn_local_safe = b_nn_local.clamp(0, len(b_idx) - 1)
                b_nn_corr_safe = b_nn_corr * valid_nb.float()

                optimizer.zero_grad()
                loss = self.elbo_loss(bx, bc, b_nn_local_safe, b_nn_corr_safe)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        self.eval()
        # Store coords metadata for normalisation reference
        self._fit_coords_min = coords_t.min(dim=0).values
        self._fit_coords_max = coords_t.max(dim=0).values
        return self

    def elbo_score(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        device: str | torch.device | None = None,
    ) -> float:
        """Return mean ELBO per point (higher = more in-distribution).

        Uses the *negative* of the training loss, so larger values mean
        the test block is better explained by the learned spatial field model.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        with torch.inference_mode():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
            nn_idx, nn_corr = _build_sparse_prior_precision(
                coords_t, self.nu, self.rho, self.k_neighbors
            )
            loss = self.elbo_loss(X_t, coords_t, nn_idx, nn_corr)

        return float(-loss.item())  # higher = more in-distribution
