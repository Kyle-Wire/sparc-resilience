"""
domain.py — DomainSpec: declarative domain collection requirements.

A DomainSpec replaces the assembler's hard-wired UHI-specific orchestration
with a data structure that any domain can provide.  The assembler iterates
``covariate_sources`` rather than naming individual fetch functions, making
it domain-agnostic.

Usage
-----
spec = DomainSpec.for_uhi(label_source=CAPAAdapter(), covariate_sources=[...])
manifest = spec.variable_manifest()
assembler.run_with_spec(config, spec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .manifest import VariableManifest
from .sources import DataSource


@dataclass
class DomainSpec:
    """Declarative description of everything a domain needs for data collection.

    Attributes
    ----------
    domain : str
        Domain identifier matching a key in
        :data:`sparc.data.collect.manifest._DOMAIN_VARIABLE_REGISTRY`
        (e.g. ``"uhi"``, ``"wildfire"``, ``"coastal"``).
    label_source : DataSource
        The primary label source whose fetch result determines anchor dates
        (e.g. ``CAPAAdapter`` for UHI, ``VIIRSAdapter`` for wildfire).
    covariate_sources : list[DataSource]
        Ordered list of covariate source adapters to run after the label source.
    target_column : str
        The target column name to write to ``project.yml``
        (e.g. ``"aat_residual"`` for UHI, ``"dnbr"`` for wildfire).
    derive_targets : callable | None
        Optional function ``(fishnet: GeoDataFrame) → GeoDataFrame`` that
        computes derived columns after all sources have run
        (e.g. ``aat_residual = aat_midday − era5_t2m`` for UHI).
    """

    domain: str
    label_source: DataSource
    covariate_sources: List[DataSource]
    target_column: str
    derive_targets: Optional[Callable] = field(default=None, repr=False)

    def variable_manifest(self) -> VariableManifest:
        """Return a fresh :class:`~sparc.data.collect.manifest.VariableManifest`
        for this domain.

        Delegates to :meth:`VariableManifest.for_domain` so that the variable
        set lives in one place (the manifest registry) and is always consistent
        between the assembler and the portal.
        """
        return VariableManifest.for_domain(self.domain)

    def all_sources(self) -> List[DataSource]:
        """Return ``[label_source] + covariate_sources`` in execution order."""
        return [self.label_source, *self.covariate_sources]

    # ------------------------------------------------------------------
    # Domain factories
    # ------------------------------------------------------------------

    @classmethod
    def for_uhi(
        cls,
        *,
        label_source: DataSource,
        covariate_sources: List[DataSource],
        derive_targets: Optional[Callable] = None,
    ) -> "DomainSpec":
        """Return the canonical UHI domain spec.

        Parameters
        ----------
        label_source : DataSource
            Primary label source (typically ``CAPAAdapter``).  Required —
            pass ``CAPAAdapter()`` or a mock for testing.
        covariate_sources : list[DataSource]
            Covariate adapters to run after the label source.
        derive_targets : callable | None
            Post-fetch column derivation (computes ``aat_residual`` by default
            when ``None``).
        """
        if derive_targets is None:
            def derive_targets(fishnet):  # type: ignore[misc]
                if "aat_midday" in fishnet.columns and "era5_t2m" in fishnet.columns:
                    fishnet["aat_residual"] = fishnet["aat_midday"] - fishnet["era5_t2m"]
                return fishnet

        return cls(
            domain="uhi",
            label_source=label_source,
            covariate_sources=list(covariate_sources),
            target_column="aat_residual",
            derive_targets=derive_targets,
        )
