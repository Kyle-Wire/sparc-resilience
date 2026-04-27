"""Unified posterior-aware causal-effect resolver (Phase 4 of v4 rewrite).

The :class:`CausalEffectResolver` consolidates the artifacts produced by
Stage 2 (neural alpha field + PDP curves) and Stage 3 (per-edge NUTS
posteriors, MC³ inclusion probabilities, Bayesian CATE posteriors) into
a single API used by the scenario simulator and the decision optimizer.

Composition formula for a *direct* treatment → outcome edge:

    Δ[d, i] = β_post[d] · α[i, t] · (τ_post[d, i] / mean_i τ_post[d, i])
              · pdp_local_slope[i] · Δx[i]

with literature shrinkage applied per
``priors.yml::calibration.shrinkage_weight``:

    β_eff = (1 - w) · β_post + w · β_lit

The resolver loads everything from the active ``ArtifactStore`` and
emits one ``WARNING`` per missing artifact (deduplicated, with an
end-of-run aggregated tally available via :meth:`missing_artifacts`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------


@dataclass
class ResolverConfig:
    n_draws_cap: int = 1000
    shrinkage_weight: float = 0.3
    multiplier_clip: Tuple[float, float] = (0.5, 1.5)
    # Threshold below which an MC³ edge is treated as "absent" by callers
    # querying :meth:`mc3_edge_prob`.  Resolver itself never prunes edges.
    mc3_threshold: float = 0.30


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@dataclass
class _LoadReport:
    loaded: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


class CausalEffectResolver:
    """Single composition surface for posterior-aware treatment effects.

    Parameters
    ----------
    store : Any
        Object implementing the ArtifactStore interface (``has``,
        ``read_any``, ``read_table``).  Use
        :func:`sparc.registry.store.get_active_store` to grab the
        currently bound store.
    dag : networkx.DiGraph, optional
        DAG used to enumerate edges and walk paths for indirect effects.
    physics_priors : dict, optional
        Loaded ``priors.yml`` content; consumed lazily for literature β
        and shrinkage weight.
    config : ResolverConfig, optional
        Numerical knobs (draw cap, shrinkage, multiplier clip).
    """

    def __init__(
        self,
        store: Any,
        *,
        dag: Any = None,
        physics_priors: Optional[Dict[str, Any]] = None,
        config: Optional[ResolverConfig] = None,
    ) -> None:
        self.store = store
        self.dag = dag
        self.physics_priors = physics_priors or {}
        self.config = config or ResolverConfig()

        # Apply shrinkage_weight override from priors.yml when present.
        cal = (self.physics_priors.get("calibration") or {}) if self.physics_priors else {}
        if "shrinkage_weight" in cal:
            try:
                self.config.shrinkage_weight = float(cal["shrinkage_weight"])
            except (TypeError, ValueError):
                pass

        self._warned: set[str] = set()
        self._report = _LoadReport()

        # Cached artifacts -------------------------------------------------
        self._alpha_full: Optional[np.ndarray] = None       # (N, n_treatments)
        self._alpha_coords: Optional[np.ndarray] = None     # (N, 2)
        self._alpha_treatments: List[str] = []
        self._alpha_schema_version: int = 1

        self._edge_samples: Dict[Tuple[str, str], np.ndarray] = {}
        self._edge_summary: Optional["pd.DataFrame"] = None  # type: ignore[name-defined]

        self._mc3_probs: Dict[Tuple[str, str], float] = {}

        self._cate_samples: Dict[str, np.ndarray] = {}      # treatment → (D, N)
        self._cate_summary: Optional["pd.DataFrame"] = None  # type: ignore[name-defined]

        self._pdp_curves: Dict[str, "pd.DataFrame"] = {}    # type: ignore[name-defined]

        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self, label: str, fn) -> bool:
        try:
            fn()
            self._report.loaded.append(label)
            return True
        except Exception as exc:
            self._missing(label, str(exc))
            return False

    def _missing(self, label: str, detail: str = "") -> None:
        if label in self._warned:
            return
        self._warned.add(label)
        self._report.missing.append(label)
        msg = f"CausalEffectResolver: missing artifact '{label}'"
        if detail:
            msg += f" ({detail})"
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _load_all(self) -> None:
        if self.store is None:
            self._missing("active_store", "no ArtifactStore bound")
            return

        self._try_load("v2_alpha_field", self._load_alpha)
        self._try_load("nuts_edge_samples", self._load_edge_samples)
        self._try_load("nuts_edge_summary", self._load_edge_summary)
        self._try_load("edge_inclusion_probs", self._load_mc3)
        self._try_load("cate_samples", self._load_cate_samples)
        self._try_load("cate_summary", self._load_cate_summary)

        # Construction summary (single line).
        n_edges = len(self._edge_samples)
        n_cate = len(self._cate_samples)
        n_alpha = 0 if self._alpha_full is None else self._alpha_full.shape[0]
        print(
            f"   CausalEffectResolver: α=({n_alpha},{self._alpha_full.shape[1] if self._alpha_full is not None else 0}) "
            f"edges={n_edges}  CATE={n_cate}  MC³={len(self._mc3_probs)}  "
            f"missing={len(self._report.missing)}"
        )

    # -- α field --------------------------------------------------------
    def _load_alpha(self) -> None:
        if not self.store.has("2", "v2_alpha_field"):
            raise FileNotFoundError("v2_alpha_field not in store")
        raw = np.asarray(self.store.read_any("2", "v2_alpha_field"))
        if raw.ndim == 1:
            self._alpha_full = raw.reshape(-1, 1)
        else:
            self._alpha_full = raw
        if self.store.has("2", "v2_alpha_field_coords"):
            self._alpha_coords = np.asarray(
                self.store.read_any("2", "v2_alpha_field_coords")
            )
        # Pull treatments + schema_version from sidecar metadata if present.
        meta: Dict[str, Any] = {}
        try:
            entry = self.store.registry.lookup("2", "v2_alpha_field")
            if entry is not None and getattr(entry, "metadata", None):
                meta = dict(entry.metadata)
        except Exception:
            meta = {}
        self._alpha_treatments = list(meta.get("treatments") or [])
        self._alpha_schema_version = int(meta.get("schema_version", 1))

    # -- per-edge NUTS samples -----------------------------------------
    def _load_edge_samples(self) -> None:
        if not self.store.has("3", "nuts_edge_samples"):
            raise FileNotFoundError("nuts_edge_samples not in store")
        raw = self.store.read_any("3", "nuts_edge_samples")
        if not isinstance(raw, dict):
            raise ValueError(f"nuts_edge_samples is not a dict (got {type(raw)})")
        for k, v in raw.items():
            if isinstance(k, tuple) and len(k) == 2:
                self._edge_samples[(str(k[0]), str(k[1]))] = np.asarray(v)
            elif isinstance(k, str) and "->" in k:
                p, c = (s.strip() for s in k.split("->", 1))
                self._edge_samples[(p, c)] = np.asarray(v)

    def _load_edge_summary(self) -> None:
        if not self.store.has("3", "nuts_edge_summary"):
            raise FileNotFoundError("nuts_edge_summary not in store")
        self._edge_summary = self.store.read_table("3", "nuts_edge_summary")

    # -- MC³ ------------------------------------------------------------
    def _load_mc3(self) -> None:
        # Prefer the long-format ``edge_confidence`` table if present;
        # otherwise melt the wide ``edge_inclusion_probs`` matrix.
        long_df = None
        if self.store.has("3", "edge_confidence"):
            try:
                long_df = self.store.read_table("3", "edge_confidence")
            except Exception:
                long_df = None
        if long_df is None:
            if not self.store.has("3", "edge_inclusion_probs"):
                raise FileNotFoundError("edge_inclusion_probs not in store")
            wide = self.store.read_table("3", "edge_inclusion_probs")
            src_col = "source" if "source" in wide.columns else wide.columns[0]
            value_cols = [c for c in wide.columns if c != src_col]
            long_df = wide.melt(
                id_vars=[src_col], value_vars=value_cols,
                var_name="target", value_name="inclusion_prob",
            ).rename(columns={src_col: "source"})
        for row in long_df.to_dict("records"):
            p = str(row.get("source") or row.get("parent") or row.get("from"))
            c = str(row.get("target") or row.get("child") or row.get("to"))
            prob = row.get("inclusion_prob")
            if prob is None:
                prob = row.get("probability") or row.get("prob") or 0.0
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if p and c and p != c:
                self._mc3_probs[(p, c)] = prob

    # -- CATE -----------------------------------------------------------
    def _load_cate_samples(self) -> None:
        if not self.store.has("3", "cate_samples"):
            raise FileNotFoundError("cate_samples not in store")
        raw = self.store.read_any("3", "cate_samples")
        if not isinstance(raw, dict):
            raise ValueError(f"cate_samples is not a dict (got {type(raw)})")
        for k, v in raw.items():
            self._cate_samples[str(k)] = np.asarray(v)

    def _load_cate_summary(self) -> None:
        if not self.store.has("3", "cate_summary"):
            raise FileNotFoundError("cate_summary not in store")
        self._cate_summary = self.store.read_table("3", "cate_summary")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def has_alpha(self) -> bool:
        return self._alpha_full is not None

    @property
    def available_treatments(self) -> List[str]:
        treats = set(self._cate_samples.keys())
        if self._alpha_treatments:
            treats.update(self._alpha_treatments)
        if self.dag is not None:
            for n, attrs in self.dag.nodes(data=True):
                if attrs.get("node_type") == "treatment":
                    treats.add(str(n))
        return sorted(treats)

    @property
    def dag_edges(self) -> List[Tuple[str, str]]:
        if self.dag is None:
            return []
        return [(str(u), str(v)) for u, v in self.dag.edges()]

    def has_posterior(self, edge: Tuple[str, str]) -> bool:
        return tuple(edge) in self._edge_samples

    def mc3_edge_prob(self, parent: str, child: str) -> float:
        return float(self._mc3_probs.get((str(parent), str(child)), 1.0))

    def missing_artifacts(self) -> List[str]:
        return list(self._report.missing)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alpha_for(self, treatment: str, n_cells: int) -> np.ndarray:
        """Return α slice of shape ``(n_cells,)`` for ``treatment``."""
        if self._alpha_full is None:
            self._missing("alpha_for_resolve", "alpha not loaded")
            return np.ones(n_cells, dtype=np.float64)
        full = self._alpha_full
        # Identify the column for this treatment.
        col = 0
        if self._alpha_treatments and treatment in self._alpha_treatments:
            col = self._alpha_treatments.index(treatment)
        elif full.shape[1] == 1:
            col = 0
        else:
            # Unknown treatment in a multi-head field: average across heads.
            col = None
        if col is None:
            alpha = full.mean(axis=-1)
        else:
            alpha = full[:, col]
        # Resize to n_cells if mismatch (KD-tree alignment is the
        # caller's responsibility; we just clip / repeat as a guard).
        if alpha.shape[0] != n_cells:
            if alpha.shape[0] > n_cells:
                alpha = alpha[:n_cells]
            else:
                alpha = np.concatenate([
                    alpha,
                    np.full(n_cells - alpha.shape[0], alpha.mean()),
                ])
        return alpha.astype(np.float64)

    def align_alpha_to_rows(self, target_coords: np.ndarray) -> np.ndarray:
        """KD-tree align α from its native coords to ``target_coords``.

        Returns a (n_target, n_treatments) ndarray.  Falls back to the
        first ``n_target`` rows when α coords are unavailable.
        """
        if self._alpha_full is None:
            n = target_coords.shape[0]
            return np.ones((n, 1), dtype=np.float64)
        if (
            self._alpha_coords is None
            or self._alpha_coords.shape[0] != self._alpha_full.shape[0]
        ):
            n = target_coords.shape[0]
            base = self._alpha_full
            if base.shape[0] >= n:
                return base[:n]
            pad = np.tile(base.mean(axis=0, keepdims=True), (n - base.shape[0], 1))
            return np.vstack([base, pad])
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(self._alpha_coords)
            _, idx = tree.query(target_coords, k=1)
            return self._alpha_full[idx]
        except Exception as exc:
            self._missing("alpha_kdtree", str(exc))
            n = target_coords.shape[0]
            return self._alpha_full[:n] if self._alpha_full.shape[0] >= n else self._alpha_full

    @staticmethod
    def summarize(samples_2d: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-cell mean / 5-50-95 percentile summary along axis 0."""
        arr = np.asarray(samples_2d)
        return {
            "mean": arr.mean(axis=0),
            "ci5": np.percentile(arr, 5, axis=0),
            "ci50": np.percentile(arr, 50, axis=0),
            "ci95": np.percentile(arr, 95, axis=0),
        }

    # ------------------------------------------------------------------
    # Edge posteriors
    # ------------------------------------------------------------------

    def resolve_structural_edge(
        self, parent: str, child: str, n_draws: int = 1000,
    ) -> np.ndarray:
        """Posterior samples for ``parent → child``.

        Returns an array of shape ``(n_draws,)``.  Falls back to a
        constant OLS-broadcast vector (with a one-shot warning) when no
        NUTS posterior exists.  Uses the edge-summary table for the
        fallback mean when available; otherwise emits 0.
        """
        key = (str(parent), str(child))
        n_draws = max(1, min(int(n_draws), int(self.config.n_draws_cap)))
        if key in self._edge_samples:
            samples = self._edge_samples[key]
            if samples.size == 0:
                self._missing(f"edge::{key}", "empty sample array")
                return self._edge_fallback(parent, child, n_draws)
            if samples.shape[0] >= n_draws:
                idx = np.linspace(0, samples.shape[0] - 1, n_draws).astype(int)
                return samples[idx].astype(np.float64)
            # Sample with replacement to reach n_draws.
            rng = np.random.default_rng(abs(hash(key)) % (2**32))
            return rng.choice(samples, size=n_draws, replace=True).astype(np.float64)
        # Fallback path
        return self._edge_fallback(parent, child, n_draws)

    def _edge_fallback(self, parent: str, child: str, n_draws: int) -> np.ndarray:
        self._missing(
            f"edge_posterior::{parent}->{child}",
            "no NUTS posterior; broadcasting OLS / literature mean",
        )
        mean = 0.0
        if self._edge_summary is not None:
            df = self._edge_summary
            try:
                row = df[
                    (df["parent"].astype(str) == str(parent))
                    & (df["child"].astype(str) == str(child))
                ]
                if len(row) > 0:
                    for col in ("mean", "beta_mean", "ols_beta", "estimate"):
                        if col in row.columns:
                            mean = float(row.iloc[0][col])
                            break
            except Exception:
                pass
        if mean == 0.0:
            # Try literature priors as last resort.
            mean = self._literature_beta(parent, child)
        return np.full(n_draws, float(mean), dtype=np.float64)

    def _literature_beta(self, parent: str, child: str) -> float:
        priors = self.physics_priors or {}
        coeffs = priors.get("coefficients") or {}
        for key in (f"{parent}->{child}", parent, child):
            entry = coeffs.get(key)
            if isinstance(entry, dict) and "value" in entry:
                try:
                    return float(entry["value"])
                except (TypeError, ValueError):
                    continue
        return 0.0

    # ------------------------------------------------------------------
    # CATE posterior
    # ------------------------------------------------------------------

    def _cate_for(self, treatment: str, n_draws: int, n_cells: int) -> np.ndarray:
        """Posterior CATE samples ``(n_draws, n_cells)`` for ``treatment``."""
        tau = self._cate_samples.get(str(treatment))
        if tau is None:
            self._missing(
                f"cate::{treatment}",
                "no posterior; broadcasting unit multiplier",
            )
            return np.ones((n_draws, n_cells), dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        # Trim or expand cells.
        if tau.shape[1] != n_cells:
            if tau.shape[1] > n_cells:
                tau = tau[:, :n_cells]
            else:
                pad_n = n_cells - tau.shape[1]
                pad = np.tile(tau.mean(axis=1, keepdims=True), (1, pad_n))
                tau = np.concatenate([tau, pad], axis=1)
        # Subsample draws.
        if tau.shape[0] >= n_draws:
            idx = np.linspace(0, tau.shape[0] - 1, n_draws).astype(int)
            return tau[idx]
        # Tile up if too few draws (degenerate frequentist case).
        rep = int(np.ceil(n_draws / max(tau.shape[0], 1)))
        return np.tile(tau, (rep, 1))[:n_draws]

    # ------------------------------------------------------------------
    # Direct effect
    # ------------------------------------------------------------------

    def resolve_direct_effect(
        self,
        treatment: str,
        target: str,
        baseline_x: np.ndarray,
        modified_x: np.ndarray,
        *,
        pdp_local_slope: Optional[np.ndarray] = None,
        n_draws: int = 1000,
    ) -> np.ndarray:
        """Sample posterior of the direct treatment → target effect per cell.

        Composition::

            Δ[d, i] = β_eff[d] · α[i, t] · (τ[d, i] / mean_i τ[d, i])
                      · pdp_slope[i] · Δx[i]

        Returns ``(n_draws_eff, n_cells)`` where ``n_draws_eff`` is the
        harmonized draw count across NUTS edge samples and CATE samples
        (capped by :attr:`ResolverConfig.n_draws_cap`).
        """
        baseline_x = np.asarray(baseline_x, dtype=np.float64).reshape(-1)
        modified_x = np.asarray(modified_x, dtype=np.float64).reshape(-1)
        n_cells = max(baseline_x.shape[0], modified_x.shape[0])
        if baseline_x.shape[0] != n_cells:
            baseline_x = np.broadcast_to(baseline_x, (n_cells,)).copy()
        if modified_x.shape[0] != n_cells:
            modified_x = np.broadcast_to(modified_x, (n_cells,)).copy()
        delta_x = modified_x - baseline_x

        # Harmonize draw counts.
        n_nuts = self._edge_samples.get((treatment, target), np.empty(0)).shape[0] or self.config.n_draws_cap
        n_cate = self._cate_samples.get(treatment, np.empty((0, 0))).shape[0] or self.config.n_draws_cap
        n_eff = max(1, min(n_draws, n_nuts, n_cate, self.config.n_draws_cap))

        # β posterior + literature shrinkage.
        beta_post = self.resolve_structural_edge(treatment, target, n_draws=n_eff)
        beta_lit = self._literature_beta(treatment, target)
        w = float(self.config.shrinkage_weight)
        beta_eff = (1.0 - w) * beta_post + w * beta_lit  # (n_eff,)

        # α slice for this treatment, broadcast over draws.
        alpha = self._alpha_for(treatment, n_cells)  # (n_cells,)
        alpha_norm_factor = alpha / (np.abs(alpha).mean() + 1e-12)

        # CATE multiplier τ / mean(τ), clipped.
        tau = self._cate_for(treatment, n_eff, n_cells)  # (n_eff, n_cells)
        tau_mean = tau.mean(axis=1, keepdims=True)
        tau_mean = np.where(np.abs(tau_mean) < 1e-10, 1.0, tau_mean)
        clip_lo, clip_hi = self.config.multiplier_clip
        multiplier = np.clip(tau / tau_mean, clip_lo, clip_hi)  # (n_eff, n_cells)

        # PDP local slope (optional).
        if pdp_local_slope is None:
            slope = np.ones(n_cells, dtype=np.float64)
        else:
            slope = np.asarray(pdp_local_slope, dtype=np.float64).reshape(-1)
            if slope.shape[0] != n_cells:
                slope = np.broadcast_to(slope, (n_cells,)).copy()

        # Compose: (n_eff, 1) · (1, n_cells) · (n_eff, n_cells) · (n_cells,) · (n_cells,)
        per_cell_factor = alpha_norm_factor * slope * delta_x  # (n_cells,)
        delta = beta_eff[:, None] * multiplier * per_cell_factor[None, :]
        return delta


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = ["CausalEffectResolver", "ResolverConfig"]
