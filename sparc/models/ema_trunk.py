"""
EMA (Exponential Moving Average) target trunk for JEPA-style training.

The JEPA paradigm requires a *target encoder* whose parameters lag the
online encoder via an EMA update.  The target encoder produces the
embeddings that the online encoder is trained to predict (in the
masked / context-vs-target setup), with stop-gradient on the target.

We mirror the SharedTrunk subset of ``SPARCMetaLearner`` (physics_enc,
alpha_emb, trunk_fusion, optionally time_embed) into a frozen-parameter
copy and update it via:

    θ_target ← τ · θ_target + (1 − τ) · θ_online

The momentum coefficient τ follows a cosine warmup schedule (BS-B):
starts at ``tau_start`` (e.g. 0.99) and ramps toward ``tau_end``
(e.g. 0.9999) over the course of training.

Phase 1 scope: provides the encode-only interface used to compute the
JEPA target embedding ``h_trunk_target``.  Downstream supervised heads
(regression, exceedance) are *not* mirrored — they are trained
exclusively from the online model.
"""

from __future__ import annotations

import copy
import math
from typing import Iterable

import torch
import torch.nn as nn

from sparc.models.neural_meta import SPARCMetaLearner

# Canonical trunk module names — single source of truth lives on
# SPARCMetaLearner._TRUNK_KEYS (neural_meta.py).  Imported here so
# _iter_trunk_param_pairs can reference it without a hard-coded local copy.
_TRUNK_MODULE_NAMES = SPARCMetaLearner._TRUNK_KEYS


def _iter_trunk_param_pairs(
    online: SPARCMetaLearner, target: SPARCMetaLearner,
) -> Iterable[tuple[nn.Parameter, nn.Parameter]]:
    """Yield (online_param, target_param) pairs for trunk-only modules."""
    for name in _TRUNK_MODULE_NAMES:
        online_mod = getattr(online, name, None)
        target_mod = getattr(target, name, None)
        if online_mod is None or target_mod is None:
            continue
        for p_o, p_t in zip(online_mod.parameters(), target_mod.parameters()):
            yield p_o, p_t


def _iter_trunk_buffer_pairs(
    online: SPARCMetaLearner, target: SPARCMetaLearner,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (online_buf, target_buf) pairs for trunk-only modules."""
    for name in _TRUNK_MODULE_NAMES:
        online_mod = getattr(online, name, None)
        target_mod = getattr(target, name, None)
        if online_mod is None or target_mod is None:
            continue
        for b_o, b_t in zip(online_mod.buffers(), target_mod.buffers()):
            yield b_o, b_t


class EMATrunk(nn.Module):
    """
    Exponential-moving-average copy of an ``SPARCMetaLearner`` 's SharedTrunk.

    Parameters
    ----------
    online_model : SPARCMetaLearner
        The online model whose trunk will be tracked.  A *deep copy* of
        the trunk-only modules is taken; the online model is otherwise
        untouched.
    tau_start : float
        Initial EMA momentum (closer to 1.0 = slower target updates).
    tau_end : float
        Final EMA momentum after warmup.  Set equal to ``tau_start`` to
        disable scheduling.
    warmup_steps : int
        Number of EMA updates over which τ ramps from ``tau_start`` to
        ``tau_end``.

    Notes
    -----
    The target trunk's parameters always have ``requires_grad=False``.
    Calls to ``encode_target`` run in inference mode (no autograd).
    """

    def __init__(
        self,
        online_model: SPARCMetaLearner,
        tau_start: float = 0.99,
        tau_end: float = 0.9999,
        warmup_steps: int = 10_000,
    ) -> None:
        super().__init__()

        if not (0.0 < tau_start <= 1.0) or not (0.0 < tau_end <= 1.0):
            raise ValueError("tau_start and tau_end must be in (0, 1].")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")

        # Hold a deep-copied trunk-bearing model.  Storing a full
        # SPARCMetaLearner copy is the cleanest way to reuse its
        # ``encode`` method without forking encoder logic.
        self._target_model = copy.deepcopy(online_model)
        for p in self._target_model.parameters():
            p.requires_grad_(False)

        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.warmup_steps = int(warmup_steps)

        # Step counter as a buffer so it survives state_dict round-trips.
        self.register_buffer(
            "num_updates", torch.zeros(1, dtype=torch.long), persistent=True,
        )

    # ------------------------------------------------------------------
    @property
    def target_model(self) -> SPARCMetaLearner:
        """The frozen, EMA-tracked target SPARCMetaLearner."""
        return self._target_model

    # ------------------------------------------------------------------
    def current_tau(self) -> float:
        """Return the EMA momentum used for the *next* update."""
        if self.warmup_steps == 0:
            return self.tau_end
        step = int(self.num_updates.item())
        if step >= self.warmup_steps:
            return self.tau_end
        frac = (1.0 - math.cos(math.pi * step / float(self.warmup_steps))) / 2.0
        return self.tau_start + (self.tau_end - self.tau_start) * frac

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, online_model: SPARCMetaLearner) -> float:
        """
        Apply one EMA step:  θ_target ← τ · θ_target + (1 − τ) · θ_online.

        Buffers (e.g. LayerNorm running stats — none currently in trunk,
        but defensive) are *copied* from online to target rather than
        averaged, matching standard JEPA / BYOL practice.

        Returns
        -------
        tau : float
            The momentum coefficient that was used for this step.
        """
        tau = self.current_tau()
        one_minus_tau = 1.0 - tau

        for p_online, p_target in _iter_trunk_param_pairs(
            online_model, self._target_model,
        ):
            # In-place: p_t.mul_(τ).add_(p_o.detach(), alpha=1-τ)
            p_target.data.mul_(tau).add_(p_online.detach().data, alpha=one_minus_tau)

        for b_online, b_target in _iter_trunk_buffer_pairs(
            online_model, self._target_model,
        ):
            b_target.data.copy_(b_online.data)

        self.num_updates += 1
        return tau

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_target(
        self,
        physics_feats: torch.Tensor,
        alpha: torch.Tensor,
        time_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the target trunk embedding ``h_trunk`` (with stop-grad).

        Parameters mirror ``SPARCMetaLearner.encode``.
        """
        self._target_model.eval()
        return self._target_model.encode(
            physics_feats=physics_feats, alpha=alpha, time_idx=time_idx,
        )
