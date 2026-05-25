"""SpatialTrunk protocol and TrunkLoader factory.

Any module that acts as a trained spatial representation trunk must satisfy
the SpatialTrunk Protocol so that training-produced checkpoints and inference
consumers share a single seam.

Two adapters currently exist:
  - SpatialANP (sparc.inference.anp) — Attentive Neural Process
  - SPARCMetaLearner (sparc.models.neural_meta) — trunk via save_trunk/load_trunk

Adding a new trunk variant: implement save_checkpoint and load_checkpoint and
the module automatically satisfies this Protocol at runtime (no inheritance
needed).

TrunkLoader
-----------
A factory class that replaces the tempfile anti-pattern in few_shot_predict
and zero_shot_predict.  Instead of writing a state dict to a NamedTemporaryFile
and immediately reading it back, callers use::

    model = TrunkLoader.from_registry(registry_path, climate_zone, x_dim)

or::

    model = TrunkLoader.from_state_dict(state_dict, x_dim)

No disk I/O is required in either case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch.nn as nn


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


class TrunkLoader:
    """Factory for SPARC trunk (SpatialANP) models.

    Replaces the ``tempfile.NamedTemporaryFile + torch.save`` pattern that was
    used to pass in-memory state dicts through the ``trunk_path`` file-path
    interface.  All methods are static; no instance state is needed.
    """

    @staticmethod
    def fresh(x_dim: int, hidden_dim: int = 64, n_heads: int = 4) -> "nn.Module":
        """Return a randomly-initialised :class:`~sparc.inference.anp.SpatialANP`.

        Parameters
        ----------
        x_dim : int
            Input feature dimensionality.
        hidden_dim : int
            Width of all hidden layers.
        n_heads : int
            Number of attention heads in cross-attention.
        """
        from sparc.inference.anp import SpatialANP
        return SpatialANP(x_dim=x_dim, hidden_dim=hidden_dim, n_heads=n_heads)

    @staticmethod
    def from_path(trunk_path: str, x_dim: int) -> "nn.Module":
        """Load a :class:`~sparc.inference.anp.SpatialANP` from a checkpoint file.

        Parameters
        ----------
        trunk_path : str
            Path to a ``.pt`` checkpoint produced by
            :meth:`~sparc.inference.anp.SpatialANP.save_checkpoint`.
        x_dim : int
            Input feature dimensionality (must match the saved checkpoint).
        """
        from sparc.inference.anp import SpatialANP
        model = SpatialANP(x_dim=x_dim)
        model.load_checkpoint(trunk_path)
        return model

    @staticmethod
    def from_state_dict(state_dict: dict, x_dim: int) -> "nn.Module":
        """Load a trunk from an in-memory state dict — no disk I/O.

        Parameters
        ----------
        state_dict : dict
            PyTorch state dict (e.g. from ``CityRegistry.load_by_climate``).
        x_dim : int
            Input feature dimensionality.
        """
        import torch
        from sparc.inference.anp import SpatialANP
        model = SpatialANP(x_dim=x_dim)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def from_registry(
        registry_path: str,
        climate_zone: "str | None",
        x_dim: int,
    ) -> "nn.Module":
        """Load a trunk from a :class:`~sparc.registry.city_registry.CityRegistry`.

        Queries the registry for a city matching *climate_zone*, falls back to
        the global trunk, and finally returns a fresh model when neither is
        present.  In all three cases the trunk is loaded directly from the
        state dict in memory — no temporary files are written.

        Parameters
        ----------
        registry_path : str
            Path to the city registry file.
        climate_zone : str or None
            Köppen climate classification string, e.g. ``"Cfa"``.  Pass
            ``None`` to skip the climate-zone lookup.
        x_dim : int
            Input feature dimensionality.
        """
        from sparc.registry.city_registry import CityRegistry

        reg = CityRegistry(registry_path)

        # 1. Try climate-zone specific record
        if climate_zone is not None:
            record = reg.load_by_climate(climate_zone)
            if record is not None and record.trunk_checkpoint is not None:
                return TrunkLoader.from_state_dict(record.trunk_checkpoint, x_dim=x_dim)

        # 2. Fall back to global trunk
        global_state = reg.load_global_trunk()
        if global_state is not None:
            return TrunkLoader.from_state_dict(global_state, x_dim=x_dim)

        # 3. Fresh initialisation
        return TrunkLoader.fresh(x_dim=x_dim)

