"""
Bilevel meta-gradient lambda optimizer for SPARC joint loss weights.

Implements a first-order DARTS approximation for online lambda adaptation.
After the curriculum ramp (Stage C), the optimizer refines each lambda using
gradient information from a held-out validation mini-batch, rather than
relying solely on the frozen CMA-ES solution.

Usage
-----
    opt = LambdaOptimizer(
        names=["physics", "smooth", "base", "neighbor"],
        init_vals={"physics": 0.1, "smooth": 0.05, "base": 0.3, "neighbor": 0.01},
        lr=1e-3,
    )

    # Every K epochs on a val batch:
    val_loss = compute_loss_with_lambdas(opt.values(), ...)
    opt.step(val_loss)

    # Use updated lambdas in the next training batch:
    lambdas = opt.values()  # dict[str, float]

Design notes
------------
- Lambdas are parameterised in softplus-space: λ = softplus(λ_raw), so they
  are always positive and gradient descent cannot collapse them to 0.
- Raw parameters are clamped to [-10, 5] to prevent numerical overflow.
- Gradient clipping (norm=1.0) stabilises updates when val_loss is spiky.
- The outer optimizer (Adam) uses a low default lr (1e-3) to ensure lambda
  drift is slow relative to the model weight updates.
- val_loss MUST be computed with all lambda parameters contributing to the
  computation graph (i.e. pass opt.tensors() instead of opt.values() when
  building the loss, then call opt.step(val_loss) directly).
"""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CLAMP_LO = -10.0
_CLAMP_HI = 5.0


class LambdaOptimizer:
    """
    Differentiable lambda optimizer using a first-order bilevel approximation.

    Parameters
    ----------
    names : ordered list of lambda term names
    init_vals : mapping from name → initial value (in physical units, not log)
    lr : outer Adam learning rate (default: 1e-3)
    """

    def __init__(
        self,
        names: list[str],
        init_vals: dict[str, float],
        lr: float = 1e-3,
    ) -> None:
        self._names = list(names)
        # Initialise raw params from physical values via inverse-softplus
        # softplus(x) ≈ x for x >> 0; use log(exp(v) - 1) as pseudo-inverse.
        self._raw: dict[str, torch.nn.Parameter] = {}
        for k in self._names:
            v = float(init_vals.get(k, 0.01))
            v = max(v, 1e-6)
            # inverse softplus approximation
            raw_init = math.log(math.expm1(v)) if v > 1e-3 else math.log(v)
            raw_init = float(max(min(raw_init, _CLAMP_HI), _CLAMP_LO))
            self._raw[k] = torch.nn.Parameter(torch.tensor(raw_init))
        self._opt = torch.optim.Adam(list(self._raw.values()), lr=lr)
        logger.debug("LambdaOptimizer initialised: %s", list(self._names))

    # ------------------------------------------------------------------
    def values(self) -> dict[str, float]:
        """Return current lambda values as plain floats (for loss kwargs)."""
        return {k: float(F.softplus(v).item()) for k, v in self._raw.items()}

    def tensors(self) -> dict[str, torch.Tensor]:
        """Return current lambda values as scalar Tensors (maintains grad)."""
        return {k: F.softplus(v) for k, v in self._raw.items()}

    # ------------------------------------------------------------------
    def step(self, val_loss: torch.Tensor) -> None:
        """
        Perform one outer-loop optimisation step on lambda parameters.

        val_loss must have been computed using tensors from ``self.tensors()``
        so that d(val_loss)/d(λ_raw) is non-zero.  Uses first-order
        approximation: we do NOT retain the inner-loop computation graph
        (no retain_graph=True) to avoid O(N²) memory overhead.

        Parameters
        ----------
        val_loss : scalar Tensor with grad_fn linking back through lambdas
        """
        self._opt.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self._raw.values()), max_norm=1.0)
        self._opt.step()
        # Clamp raw values to prevent softplus from overflowing
        with torch.no_grad():
            for v in self._raw.values():
                v.clamp_(_CLAMP_LO, _CLAMP_HI)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialise for checkpoint persistence."""
        return {
            "names": self._names,
            "raw": {k: v.data.item() for k, v in self._raw.items()},
            "opt": self._opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        for k, val in state.get("raw", {}).items():
            if k in self._raw:
                with torch.no_grad():
                    self._raw[k].fill_(val)
        try:
            self._opt.load_state_dict(state["opt"])
        except Exception:
            pass  # optimizer state mismatch is non-fatal
