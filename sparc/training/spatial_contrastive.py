"""
Spatial Contrastive Pretext Learning for cross-city block encoder alignment.

2.SS-5 item: trains the SPARC SharedTrunk encoder via InfoNCE so that
coreset blocks from cities with similar spatial process bandwidths
("positive pairs") are close in embedding space, while blocks from
different bandwidth clusters are pushed apart.

Bandwidth proxy: ``log_bandwidth = log(rho)`` where ``rho`` is the
Matérn length-scale (``rho = 1 / kappa_mean``) stored either in
``CityRegistry`` city metrics or computed from the Stage 0 correlogram
``corr_payload``.

Usage (from v2_neural_training.py)::

    from sparc.training.spatial_contrastive import spatial_contrastive_pretext

    if config.get("continual", {}).get("contrastive_pretext", False):
        spatial_contrastive_pretext(
            model=model,
            coresets_with_bw=coresets,   # list of {"X": ndarray, "log_bw": float, ...}
            n_epochs=config.get("continual", {}).get("contrastive_epochs", 10),
            device=device,
        )

Architecture note: the contrastive loss runs over ``SPARCMetaLearner.encode``
which takes (physics_feats, alpha). For coreset data we proxy alpha as a
constant 1.0 column, matching the ProcessRateNet default prior.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

class SpatialContrastiveLoss(nn.Module):
    """
    InfoNCE-based spatial contrastive loss.

    Given a batch of embeddings and positive pair assignments, computes::

        L_NCE = -log[ exp(sim(a, p) / τ) / Σ_j exp(sim(a, n_j) / τ) ]

    where the denominator sums over all non-anchor entries in the row
    (in-batch negatives), consistent with SimCLR / NT-Xent style.

    Parameters
    ----------
    temperature : float
        Softmax temperature τ. Default 0.07 (standard for spatial tasks).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Parameters
        ----------
        embeddings : (B, D) — L2-normalised trunk embeddings
        anchor_idx : (P,) long — indices of anchor samples
        positive_idx : (P,) long — indices of corresponding positives

        Returns
        -------
        loss : scalar tensor
        """
        if len(anchor_idx) == 0:
            return embeddings.sum() * 0.0  # differentiable zero

        z = F.normalize(embeddings, dim=-1)  # (B, D)
        B = z.shape[0]

        z_a = z[anchor_idx]       # (P, D)
        z_p = z[positive_idx]     # (P, D)

        # Compute similarity of each anchor against ALL embeddings: (P, B)
        logits = torch.mm(z_a, z.T) / self.temperature   # (P, B)

        # Positive logit: sim(a, p) — (P,)
        pos_logit = (z_a * z_p).sum(dim=-1, keepdim=True) / self.temperature  # (P, 1)

        # Log-sum-exp over all B entries (in-batch negatives including self)
        log_denom = torch.logsumexp(logits, dim=-1)       # (P,)
        log_pos = pos_logit.squeeze(-1)                   # (P,)

        loss = -(log_pos - log_denom).mean()
        return loss


# ---------------------------------------------------------------------------
# Positive pair mining
# ---------------------------------------------------------------------------

def mine_positive_pairs(
    log_bw_labels: Sequence[float],
    threshold: float = 0.3,
) -> tuple[list[int], list[int]]:
    """
    Find positive pairs among coreset blocks based on bandwidth proximity.

    Two blocks i, j are a positive pair when::

        |log_bw_i − log_bw_j| < threshold

    Returns anchor and positive index lists (each pair appears once).

    Parameters
    ----------
    log_bw_labels : sequence of float
        ``log(rho)`` for each sample in the batch.
    threshold : float
        Log-ratio threshold. 0.3 ≈ bandwidths within a factor of e^0.3 ≈ 1.35.

    Returns
    -------
    anchor_idx, positive_idx : lists of int
    """
    labels = np.asarray(log_bw_labels, dtype=np.float32)
    N = len(labels)
    anchors, positives = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(float(labels[i]) - float(labels[j])) < threshold:
                anchors.append(i)
                positives.append(j)
    return anchors, positives


def query_coresets_by_bandwidth_cluster(
    registry,
    threshold: float = 0.3,
) -> list[dict]:
    """
    Load all city coresets from a ``CityRegistry`` and annotate each with
    its ``log_bw`` label, grouping cities into bandwidth clusters.

    Bandwidth is read from city metrics key ``"median_log_bandwidth"``.
    Cities without this key are assigned ``log_bw = 0.0`` (neutral).

    Parameters
    ----------
    registry : CityRegistry
        Loaded city registry instance.
    threshold : float
        Same threshold used in :func:`mine_positive_pairs`.

    Returns
    -------
    List of dicts: ``{"X": ndarray, "coords": ndarray, "log_bw": float, "city": str}``
    """
    raw = registry.load_all_coresets()
    result = []
    for entry in raw:
        city_name = entry.get("city", "unknown")
        log_bw = 0.0
        try:
            city_record = registry.load_city(city_name)
            log_bw = float(city_record.metrics.get("median_log_bandwidth", 0.0))
        except Exception:
            pass
        result.append({
            "X": entry["X"],
            "coords": entry.get("coords", np.zeros((len(entry["X"]), 2), dtype=np.float32)),
            "log_bw": log_bw,
            "city": city_name,
        })
    return result


def log_bandwidth_from_payload(corr_payload: dict | None) -> float:
    """
    Extract ``log(rho)`` from a Stage 0 correlogram payload.

    ``rho = 1 / kappa_mean``.  Returns 0.0 if payload is absent or malformed.
    """
    if corr_payload is None:
        return 0.0
    try:
        kappa_mean = float(corr_payload["kappa"]["mean"])
        if kappa_mean > 0:
            rho = 1.0 / kappa_mean
            return math.log(rho)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Pretext training loop
# ---------------------------------------------------------------------------

def _coreset_to_tensors(
    coreset: dict,
    n_physics: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a coreset dict to (physics_feats, alpha) tensors.

    We clip/pad to ``n_physics`` features (coreset may have different
    number of features than the current city if coresets come from a
    differently-sized dataset).
    """
    X = np.asarray(coreset["X"], dtype=np.float32)
    n, f = X.shape
    if f < n_physics:
        X = np.pad(X, ((0, 0), (0, n_physics - f)))
    elif f > n_physics:
        X = X[:, :n_physics]
    phys = torch.tensor(X, dtype=torch.float32, device=device)
    alpha = torch.ones((n, 1), dtype=torch.float32, device=device)
    return phys, alpha


def spatial_contrastive_pretext(
    model: "SPARCMetaLearner",
    coresets_with_bw: list[dict],
    n_epochs: int = 10,
    lr: float = 1e-4,
    temperature: float = 0.07,
    bw_threshold: float = 0.3,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> None:
    """
    Run spatial contrastive pretext training on the SPARC trunk encoder.

    Samples batches from all provided coresets, mines within-batch
    positive pairs by bandwidth proximity, and minimises InfoNCE loss
    over the trunk embeddings.

    This is a *pretext* phase — it modifies ``model.trunk_fusion``,
    ``model.physics_enc``, and ``model.alpha_emb`` in-place.  The city-
    specific heads are not touched.

    Parameters
    ----------
    model : SPARCMetaLearner
        The meta-learner whose trunk will be updated.
    coresets_with_bw : list of dicts
        Each dict must have keys ``"X"`` (ndarray, shape (N, F)) and
        ``"log_bw"`` (float).  Optionally ``"coords"``.
    n_epochs : int
        Number of pretext epochs. Default 10.
    lr : float
        Learning rate for trunk parameters. Default 1e-4.
    temperature : float
        InfoNCE temperature.
    bw_threshold : float
        Log-bandwidth proximity threshold for positive pair mining.
    batch_size : int
        Number of coreset samples per gradient step.
    device : torch.device or str or None
        Training device. Defaults to ``model`` parameters' device.
    """
    if not coresets_with_bw:
        logger.info("spatial_contrastive_pretext: no coresets available, skipping")
        return

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)

    n_physics = model.physics_enc.source_enc.network[0].in_features  # type: ignore[attr-defined]

    # Build combined coreset pool
    all_X: list[np.ndarray] = []
    all_log_bw: list[float] = []
    for entry in coresets_with_bw:
        X = np.asarray(entry["X"], dtype=np.float32)
        n, f = X.shape
        if f < n_physics:
            X = np.pad(X, ((0, 0), (0, n_physics - f)))
        elif f > n_physics:
            X = X[:, :n_physics]
        log_bw = float(entry.get("log_bw", 0.0))
        all_X.append(X)
        all_log_bw.extend([log_bw] * n)

    pool_X = np.concatenate(all_X, axis=0)  # (N_total, F)
    pool_log_bw = np.array(all_log_bw, dtype=np.float32)
    N_total = len(pool_X)

    if N_total < 2:
        logger.info("spatial_contrastive_pretext: too few coreset samples (%d), skipping", N_total)
        return

    # Optimise only trunk parameters
    trunk_params = (
        list(model.physics_enc.parameters())
        + list(model.alpha_emb.parameters())
        + list(model.trunk_fusion.parameters())
    )
    optimizer = torch.optim.Adam(trunk_params, lr=lr)
    criterion = SpatialContrastiveLoss(temperature=temperature)

    model.train()
    rng = np.random.default_rng(42)
    total_loss = 0.0
    n_steps = 0

    for epoch in range(n_epochs):
        perm = rng.permutation(N_total)
        epoch_loss = 0.0
        epoch_steps = 0
        for start in range(0, N_total, batch_size):
            batch_idx = perm[start : start + batch_size]
            if len(batch_idx) < 2:
                continue

            X_b = torch.tensor(pool_X[batch_idx], dtype=torch.float32, device=device)
            alpha_b = torch.ones((len(batch_idx), 1), dtype=torch.float32, device=device)
            log_bw_b = pool_log_bw[batch_idx]

            # Positive pair mining (in-batch)
            anch, pos = mine_positive_pairs(log_bw_b, threshold=bw_threshold)
            if not anch:
                continue  # no positive pairs in this batch — skip gradient step

            anch_t = torch.tensor(anch, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)

            # Forward: trunk embedding
            h = model.encode(X_b, alpha_b)   # (B, hidden_dim)

            loss = criterion(h, anch_t, pos_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trunk_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        if epoch_steps > 0:
            avg = epoch_loss / epoch_steps
            total_loss += avg
            n_steps += 1
            logger.debug(
                "Contrastive pretext epoch %d/%d — loss=%.4f", epoch + 1, n_epochs, avg
            )

    if n_steps > 0:
        logger.info(
            "Contrastive pretext complete: %d epochs, mean loss=%.4f, "
            "%d coresets, %d samples, bw_threshold=%.2f",
            n_epochs, total_loss / n_steps, len(coresets_with_bw), N_total, bw_threshold,
        )
    else:
        logger.info(
            "Contrastive pretext: no gradient steps taken (no positive pairs found in any batch)"
        )
