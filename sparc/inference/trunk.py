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
    def from_state_dict(
        state_dict: dict,
        x_dim: int = 0,
        model_type: str = "anp",
        hidden_dim: int = 64,
        n_heads: int = 4,
        **kwargs,
    ) -> "nn.Module":
        """Load a trunk from an in-memory state dict — no disk I/O.

        Parameters
        ----------
        state_dict : dict
            PyTorch state dict.
        x_dim : int
            Input feature dimensionality (required for ``model_type='anp'``).
        model_type : str
            ``"anp"`` (default) — returns a :class:`~sparc.inference.anp.SpatialANP`.
            ``"sparc_meta"`` — returns a :class:`~sparc.models.neural_meta.SPARCMetaLearner`
            constructed from *kwargs* (``n_base_models``, ``n_physics_features``,
            ``d_spatial``, ``hidden_dim``).
        hidden_dim : int
            Hidden dimension for ANP construction (default 64).  Unused when
            ``model_type='sparc_meta'``.
        n_heads : int
            Number of attention heads for ANP construction (default 4).  Unused
            when ``model_type='sparc_meta'``.
        **kwargs
            Forwarded to :class:`~sparc.models.neural_meta.SPARCMetaLearner` when
            ``model_type='sparc_meta'``.
        """
        import torch
        if model_type == "sparc_meta":
            from sparc.models.neural_meta_config import NeuralMetaConfig
            from sparc.models.neural_meta import SPARCMetaLearner
            # hidden_dim and n_heads are captured as named params above (for ANP);
            # merge them into kwargs for NeuralMetaConfig so the caller can pass
            # either via positional keyword args or via **kwargs.
            meta_kwargs = {"hidden_dim": hidden_dim, "n_heads": n_heads}
            meta_kwargs.update(kwargs)
            cfg = NeuralMetaConfig(**meta_kwargs)
            model = SPARCMetaLearner.from_config(cfg)
            model.load_state_dict(state_dict)
            return model
        # Default: SpatialANP
        from sparc.inference.anp import SpatialANP
        model = SpatialANP(x_dim=x_dim, hidden_dim=hidden_dim, n_heads=n_heads)
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

    @staticmethod
    def load(
        path: str,
        kind: str,
        *,
        n_physics: int | None = None,
        hidden_dim: int = 256,
        device: str = "cpu",
    ) -> "object":
        """Load any SPARC trunk checkpoint and return an encode-able object.

        Unified factory covering all three trunk shapes:

        - ``kind="jepa"``  — PI-JEPA trunk checkpoint (produced by
          ``train_multicity_jepa.py``).  Returns a
          :class:`~sparc.causal.jepa_trunk_adapter.JEPATrunkAdapter`.
        - ``kind="meta"``  — SPARCMetaLearner trunk checkpoint (produced
          by :meth:`~sparc.models.neural_meta.SPARCMetaLearner.save_trunk`).
          Returns a :class:`MetaLearnerTrunkAdapter`.
        - ``kind="anp"``   — SpatialANP checkpoint.  Returns a
          :class:`~sparc.inference.anp.SpatialANP` loaded via
          :meth:`from_path`.  Requires ``n_physics`` (= ``x_dim``).

        All returned objects expose::

            encode(physics_t: Tensor, alpha_t: Tensor | None) -> Tensor(N, H)

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        kind : str
            One of ``"jepa"``, ``"meta"``, ``"anp"``.
        n_physics : int or None
            Input feature dimension.  Required for ``kind="anp"``; used as
            a hint for ``kind="meta"`` (inferred from checkpoint if omitted).
        hidden_dim : int
            Hidden dimension for ``kind="meta"`` reconstruction (default 256).
        device : str
            PyTorch device string (default ``"cpu"``).

        Raises
        ------
        ValueError
            If ``kind`` is not one of the recognised values.
        """
        kind = kind.lower().strip()

        if kind == "jepa":
            from sparc.causal.jepa_trunk_adapter import JEPATrunkAdapter
            return JEPATrunkAdapter(path, device=device)

        if kind == "meta":
            return MetaLearnerTrunkAdapter.from_trunk_path(
                path,
                n_physics=n_physics,
                hidden_dim=hidden_dim,
                device=device,
            )

        if kind == "anp":
            if n_physics is None:
                raise ValueError("n_physics is required when kind='anp'")
            return TrunkLoader.from_path(path, x_dim=n_physics)

        raise ValueError(
            f"TrunkLoader.load: unknown kind={kind!r}. "
            f"Expected one of 'jepa', 'meta', 'anp'."
        )


class MetaLearnerTrunkAdapter:
    """Thin adapter that exposes SPARCMetaLearner.encode() from a trunk checkpoint.

    Reconstructs just the SharedTrunk layers from a ``shared_trunk.pt``
    checkpoint and wraps them behind the same ``encode(physics_t, alpha_t)``
    interface used by :class:`~sparc.causal.jepa_trunk_adapter.JEPATrunkAdapter`.

    Parameters
    ----------
    path : str
        Path to a ``shared_trunk.pt`` produced by
        :meth:`~sparc.models.neural_meta.SPARCMetaLearner.save_trunk`.
    n_physics : int
        Number of physics input features.
    hidden_dim : int
        Hidden dimension (must match the saved trunk).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        model: "nn.Module",
        device: str = "cpu",
    ) -> None:
        import torch
        self._model = model
        self._device = torch.device(device)
        self._model.eval()

    @classmethod
    def from_trunk_path(
        cls,
        path: str,
        n_physics: int | None = None,
        hidden_dim: int = 256,
        device: str = "cpu",
    ) -> "MetaLearnerTrunkAdapter":
        """Load from a ``shared_trunk.pt`` file.

        If ``n_physics`` is not provided, infers it from the first
        ``physics_enc`` weight tensor in the checkpoint.
        """
        import torch
        from sparc.models.neural_meta import SPARCMetaLearner

        trunk_state: dict = torch.load(path, map_location="cpu", weights_only=True)

        # Infer n_physics from checkpoint if not supplied
        if n_physics is None:
            # physics_enc is a PDEInformedPhysicsEncoder; its source_enc.network.0.weight
            # has shape (hidden, n_physics)
            for k, v in trunk_state.items():
                if "source_enc.network.0.weight" in k or "harmonic_enc.network.0.linear.weight" in k:
                    n_physics = v.shape[1]
                    break
            if n_physics is None:
                raise ValueError(
                    "Cannot infer n_physics from trunk checkpoint; supply it explicitly."
                )

        # Infer hidden_dim from checkpoint
        for k, v in trunk_state.items():
            if "trunk_fusion.0.weight" in k:
                hidden_dim = v.shape[0]
                break

        model = SPARCMetaLearner(
            n_base_models=1,
            n_physics_features=n_physics,
            d_spatial=4,
            hidden_dim=hidden_dim,
            dropout=0.0,
            thresholds=[0.5],
            n_heads=1,
            max_neighbors=4,
        )
        # load_trunk uses strict=False so only trunk keys are filled
        model.load_trunk(path)
        return cls(model, device=device)

    def encode(self, physics_t: "torch.Tensor", alpha_t: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Return the SharedTrunk embedding.

        Parameters
        ----------
        physics_t : Tensor(N, n_physics)
        alpha_t   : Tensor(N, 1) or None (defaults to ones)

        Returns
        -------
        Tensor(N, hidden_dim)
        """
        import torch
        dev = self._device
        X = physics_t.to(dev)
        if alpha_t is None:
            alpha = torch.ones(X.shape[0], 1, device=dev, dtype=X.dtype)
        else:
            alpha = alpha_t.to(dev)
        with torch.no_grad():
            return self._model.encode(X, alpha)

