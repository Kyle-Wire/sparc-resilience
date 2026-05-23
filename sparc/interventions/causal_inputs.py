"""
sparc/interventions/causal_inputs.py — CausalInputs seam + CausalInputLoader.

This module splits the I/O responsibility out of ``CausalEffectResolver``.

``CausalInputs``
    Pure data struct.  The seam between artifact loading and effect composition.
    Construction requires no I/O — tests inject it directly.

``CausalInputLoader``
    Reads from ``ArtifactStore``, ``priors_map``, and ``networkx.DiGraph``.
    Returns a ``CausalInputs`` instance.  All I/O is here; none is in the
    composer.

Before this split, ``CausalEffectResolver.__init__`` loaded artifacts *and*
composed effects — testing effect composition required a live registry.
Now:

    inputs = CausalInputLoader(store, dag, physics_priors).load()
    # --- or in tests ---
    inputs = CausalInputs(alpha_full=..., edge_samples={...}, ...)

Interface
---------
::

    loader = CausalInputLoader(store, dag=dag, physics_priors=priors_map)
    inputs = loader.load()

    # Then hand off to CausalEffectResolver (which now accepts CausalInputs):
    resolver = CausalEffectResolver.from_inputs(inputs)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CausalInputs — the seam
# ---------------------------------------------------------------------------


@dataclass
class CausalInputs:
    """Pure data struct carrying all artifacts needed for effect composition.

    No I/O in this class.  Construct from :class:`CausalInputLoader` in
    production, or manually in tests.

    Attributes
    ----------
    alpha_full : np.ndarray or None
        Neural alpha field, shape ``(N, n_treatments)``.
    alpha_coords : np.ndarray or None
        Coordinates for alpha field, shape ``(N, 2)``.
    alpha_treatments : list[str]
        Treatment labels corresponding to alpha columns.
    alpha_schema_version : int
        Schema version of the alpha artifact.
    edge_samples : dict[tuple[str,str], np.ndarray]
        Per-edge NUTS posterior samples.
    edge_summary : pd.DataFrame or None
        Summary statistics over NUTS samples.
    mc3_probs : dict[tuple[str,str], float]
        MC³ edge inclusion probabilities.
    cate_samples : dict[str, np.ndarray]
        Per-treatment CATE posterior samples, shape ``(D, N)``.
    cate_summary : pd.DataFrame or None
        Summary statistics over CATE samples.
    pdp_curves : dict[str, pd.DataFrame]
        Partial-dependence-like curves per treatment.
    missing : list[str]
        Artifact labels that could not be loaded (deduplicated).
    """

    alpha_full: Optional[np.ndarray] = None
    alpha_coords: Optional[np.ndarray] = None
    alpha_treatments: List[str] = field(default_factory=list)
    alpha_schema_version: int = 1

    edge_samples: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    edge_summary: Any = None  # pd.DataFrame | None

    mc3_probs: Dict[Tuple[str, str], float] = field(default_factory=dict)

    cate_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    cate_summary: Any = None  # pd.DataFrame | None

    pdp_curves: Dict[str, Any] = field(default_factory=dict)  # str → pd.DataFrame

    missing: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_alpha(self) -> bool:
        return self.alpha_full is not None

    @property
    def n_locations(self) -> int:
        if self.alpha_full is not None:
            return int(self.alpha_full.shape[0])
        if self.cate_samples:
            arr = next(iter(self.cate_samples.values()))
            return int(np.asarray(arr).shape[-1])
        return 0

    def summary(self) -> str:
        n_alpha = 0 if self.alpha_full is None else self.alpha_full.shape[0]
        n_alpha_t = 0 if self.alpha_full is None else self.alpha_full.shape[1]
        return (
            f"CausalInputs(α=({n_alpha},{n_alpha_t}) "
            f"edges={len(self.edge_samples)} "
            f"CATE={len(self.cate_samples)} "
            f"MC³={len(self.mc3_probs)} "
            f"missing={len(self.missing)})"
        )


# ---------------------------------------------------------------------------
# CausalInputLoader — all I/O here
# ---------------------------------------------------------------------------


class CausalInputLoader:
    """Loads causal artifacts from an ArtifactStore into a CausalInputs struct.

    Parameters
    ----------
    store : Any
        Object implementing the ArtifactStore interface
        (``has``, ``read_any``, ``read_table``).
    dag : networkx.DiGraph, optional
        DAG used to enumerate edges.  Can be ``None`` when no DAG is available.
    physics_priors : dict, optional
        Raw ``priors.yml`` content.  Not used during loading; passed through
        to ``CausalInputs`` metadata if needed by callers.
    """

    def __init__(
        self,
        store: Any,
        *,
        dag: Any = None,
        physics_priors: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._store = store
        self._dag = dag
        self._physics_priors = physics_priors or {}

    def load(self) -> CausalInputs:
        """Run all artifact loads and return a :class:`CausalInputs`.

        Missing artifacts emit ``RuntimeWarning`` and are recorded in
        ``CausalInputs.missing`` — loading never raises.
        """
        inputs = CausalInputs()

        if self._store is None:
            inputs.missing.append("active_store")
            warnings.warn(
                "CausalInputLoader: no ArtifactStore bound",
                RuntimeWarning,
                stacklevel=2,
            )
            return inputs

        self._try(inputs, "v2_alpha_field", self._load_alpha, inputs)
        self._try(inputs, "nuts_edge_samples", self._load_edge_samples, inputs)
        self._try(inputs, "nuts_edge_summary", self._load_edge_summary, inputs)
        self._try(inputs, "edge_inclusion_probs", self._load_mc3, inputs)
        self._try(inputs, "cate_samples", self._load_cate_samples, inputs)
        self._try(inputs, "cate_summary", self._load_cate_summary, inputs)

        print(f"   {inputs.summary()}")
        return inputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try(
        self,
        inputs: CausalInputs,
        label: str,
        fn,
        *args: Any,
    ) -> None:
        try:
            fn(*args)
        except Exception as exc:
            if label not in inputs.missing:
                inputs.missing.append(label)
            msg = f"CausalInputLoader: missing artifact '{label}' ({exc})"
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _load_alpha(self, inputs: CausalInputs) -> None:
        if not self._store.has("2", "v2_alpha_field"):
            raise FileNotFoundError("v2_alpha_field not in store")
        raw = np.asarray(self._store.read_any("2", "v2_alpha_field"))
        inputs.alpha_full = raw.reshape(-1, 1) if raw.ndim == 1 else raw

        if self._store.has("2", "v2_alpha_field_coords"):
            inputs.alpha_coords = np.asarray(
                self._store.read_any("2", "v2_alpha_field_coords")
            )

        meta: Dict[str, Any] = {}
        try:
            entry = self._store.registry.lookup("2", "v2_alpha_field")
            if entry is not None and getattr(entry, "metadata", None):
                meta = dict(entry.metadata)
        except Exception:
            pass
        inputs.alpha_treatments = list(meta.get("treatments") or [])
        inputs.alpha_schema_version = int(meta.get("schema_version", 1))

    def _load_edge_samples(self, inputs: CausalInputs) -> None:
        if not self._store.has("3", "nuts_edge_samples"):
            raise FileNotFoundError("nuts_edge_samples not in store")
        raw = self._store.read_any("3", "nuts_edge_samples")
        if not isinstance(raw, dict):
            raise ValueError(f"nuts_edge_samples is not a dict (got {type(raw)})")
        for k, v in raw.items():
            if isinstance(k, tuple) and len(k) == 2:
                inputs.edge_samples[(str(k[0]), str(k[1]))] = np.asarray(v)
            elif isinstance(k, str) and "->" in k:
                p, c = (s.strip() for s in k.split("->", 1))
                inputs.edge_samples[(p, c)] = np.asarray(v)

    def _load_edge_summary(self, inputs: CausalInputs) -> None:
        if not self._store.has("3", "nuts_edge_summary"):
            raise FileNotFoundError("nuts_edge_summary not in store")
        inputs.edge_summary = self._store.read_table("3", "nuts_edge_summary")

    def _load_mc3(self, inputs: CausalInputs) -> None:
        long_df = None
        if self._store.has("3", "edge_confidence"):
            try:
                long_df = self._store.read_table("3", "edge_confidence")
            except Exception:
                long_df = None
        if long_df is None:
            if not self._store.has("3", "edge_inclusion_probs"):
                raise FileNotFoundError("edge_inclusion_probs not in store")
            wide = self._store.read_table("3", "edge_inclusion_probs")
            src_col = "source" if "source" in wide.columns else wide.columns[0]
            value_cols = [c for c in wide.columns if c != src_col]
            long_df = wide.melt(
                id_vars=[src_col], value_vars=value_cols,
                var_name="target", value_name="inclusion_prob",
            ).rename(columns={src_col: "source"})

        for row in long_df.to_dict("records"):
            p = str(row.get("source") or row.get("parent") or row.get("from"))
            c = str(row.get("target") or row.get("child") or row.get("to"))
            prob = row.get("inclusion_prob") or row.get("probability") or 0.0
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if p and c and p != c:
                inputs.mc3_probs[(p, c)] = prob

    def _load_cate_samples(self, inputs: CausalInputs) -> None:
        if not self._store.has("3", "cate_samples"):
            raise FileNotFoundError("cate_samples not in store")
        raw = self._store.read_any("3", "cate_samples")
        if not isinstance(raw, dict):
            raise ValueError(f"cate_samples is not a dict (got {type(raw)})")
        for k, v in raw.items():
            inputs.cate_samples[str(k)] = np.asarray(v)

    def _load_cate_summary(self, inputs: CausalInputs) -> None:
        if not self._store.has("3", "cate_summary"):
            raise FileNotFoundError("cate_summary not in store")
        inputs.cate_summary = self._store.read_table("3", "cate_summary")
