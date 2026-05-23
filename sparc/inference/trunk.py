"""SpatialTrunk — structural Protocol for SPARC inference modules.

Any module that acts as a trained spatial representation trunk must satisfy
this Protocol so that training-produced checkpoints and inference consumers
share a single seam.

Two adapters currently exist:
  - SpatialANP (sparc.inference.anp) — Attentive Neural Process
  - SPARCMetaLearner (sparc.models.neural_meta) — trunk via save_trunk/load_trunk

Adding a new trunk variant: implement save_checkpoint and load_checkpoint and
the module automatically satisfies this Protocol at runtime (no inheritance
needed).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpatialTrunk(Protocol):
    """Protocol satisfied by any module with checkpoint persistence."""

    def save_checkpoint(self, path: str) -> None:
        """Persist the module's state to *path*."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Restore the module's state from *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist on disk.
        """
        ...
