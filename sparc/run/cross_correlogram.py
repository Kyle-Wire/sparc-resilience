"""
Phase 3: matrix-valued cross-correlograms with symmetric / antisymmetric
decomposition.

For ``V`` variables ``z₁ … z_V`` evaluated at ``N`` points, this module
computes the **V×V cross-correlogram tensor** at each lag bin (and,
optionally, each signed-angle bin):

    C[L, K, i, j] = (1 / |bin(L,K)|) · Σ_(k,l)∈bin(L,K)  ẑᵢ(k) · ẑⱼ(l)

with ẑ standardised to mean 0, variance 1.  Diagonal blocks ``C[·,·,i,i]``
are the standard cross-Moran of variable ``i`` with itself; off-diagonal
blocks measure spatial cross-coupling.

Two decompositions are exposed:

* **Isotropic** (``n_angle_bins=1``): the bin mask is naturally symmetric
  in ``(k,l)`` so ``C[L, 0, i, j] = C[L, 0, j, i]`` and the antisymmetric
  block is identically zero.  We still report ``C_sym = C`` for downstream
  consumers.

* **Directional** (``n_angle_bins>1``, signed angles on ``[0, 2π)``):
  the pair ``(k,l)`` with displacement angle ``φ`` enters bin ``K``;
  the reverse pair ``(l,k)`` with angle ``φ+π`` enters the opposite bin
  ``K′ = K + n_angle_bins/2  (mod n_angle_bins)``.  We define

      C_sym(L, K)[i,j]  = ½ ( C[L,K,i,j] + C[L,K′,j,i] )       (co-variation)
      C_anti(L, K)[i,j] = ½ ( C[L,K,i,j] − C[L,K′,j,i] )       (causal-direction)

  An advective coupling ``Y(x) = X(x − v·δt) + ε`` with velocity ``v`` in
  direction ``φ₀`` produces an antisymmetric peak at lag ``|v·δt|`` in
  direction ``φ₀``.

A permutation-null helper provides empirical p-values for the peak
magnitudes.  Per-pair summaries (peak magnitude, peak lag/angle, effective
range, permutation p-value) are assembled by :func:`per_pair_summary`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor computation
# ---------------------------------------------------------------------------


def _bin_indices(coords: np.ndarray,
                 max_distance: float,
                 n_lags: int,
                 n_angle_bins: int):
    """Return per-pair (L_idx, K_idx) integer matrices and the bin centres.

    For pairs with distance ≥ ``max_distance`` or i==j, ``L_idx`` is set to
    ``n_lags`` (a sentinel that we filter out below).
    """
    n = coords.shape[0]
    dx = coords[None, :, 0] - coords[:, None, 0]
    dy = coords[None, :, 1] - coords[:, None, 1]
    d = np.sqrt(dx * dx + dy * dy)
    lag_edges = np.linspace(0.0, max_distance, n_lags + 1)
    L_idx = np.digitize(d, lag_edges) - 1                  # 0 … n_lags-1, -1 for d<0 (none)
    L_idx[(d >= max_distance) | (d <= 0.0)] = n_lags       # sentinel for "skip"
    if n_angle_bins == 1:
        K_idx = np.zeros_like(L_idx, dtype=np.int64)
        angle_centers = np.array([0.0])
    else:
        # Signed angles in [0, 2π)
        ang = np.mod(np.arctan2(dy, dx), 2.0 * math.pi)
        edges = np.linspace(0.0, 2.0 * math.pi, n_angle_bins + 1)
        K_idx = np.digitize(ang, edges) - 1
        K_idx = np.clip(K_idx, 0, n_angle_bins - 1)
        angle_centers = 0.5 * (edges[:-1] + edges[1:])
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    return L_idx, K_idx, lag_centers, angle_centers, n


@dataclass
class CrossCorrelogramTensor:
    """Computed cross-correlogram tensor + bin metadata."""

    C: np.ndarray                   # shape (L, K, V, V)
    lag_distances: np.ndarray       # (L,)
    angle_centers_rad: np.ndarray   # (K,)
    n_pairs: np.ndarray             # (L, K) integer pair counts
    variable_names: list[str]
    n_angle_bins: int

    @property
    def is_directional(self) -> bool:
        return self.n_angle_bins > 1


def compute_cross_correlogram(
    coords: np.ndarray,
    values_matrix: np.ndarray,
    *,
    max_distance: float,
    n_lags: int = 12,
    n_angle_bins: int = 1,
    variable_names: Optional[list[str]] = None,
) -> CrossCorrelogramTensor:
    """Compute the V×V cross-correlogram tensor over (lag × angle) bins.

    Parameters
    ----------
    coords : (N, 2) point coordinates.
    values_matrix : (N, V) variable values; columns are standardised
        in-place to (mean 0, variance 1).  NaNs in any variable cause
        the corresponding row to be dropped.
    max_distance : float, max lag distance.
    n_lags : int, number of lag bins.
    n_angle_bins : int, ``1`` (isotropic) or even integer ≥ 2 for the
        directional decomposition.  When > 1, ``n_angle_bins`` MUST be
        even so ``K → K + n_angle_bins/2 (mod n_angle_bins)`` lands on
        an integer bin.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values_matrix = np.asarray(values_matrix, dtype=np.float64)
    if coords.shape[0] != values_matrix.shape[0]:
        raise ValueError("coords and values_matrix must share the first dim")
    if n_angle_bins != 1 and (n_angle_bins < 2 or n_angle_bins % 2 != 0):
        raise ValueError("n_angle_bins must be 1 or an even integer ≥ 2")

    # Drop rows with any NaN
    valid = ~np.isnan(values_matrix).any(axis=1)
    coords = coords[valid]
    values_matrix = values_matrix[valid]
    N, V = values_matrix.shape
    if N < 20:
        raise ValueError(f"Cross-correlogram needs ≥ 20 valid rows; got {N}")
    if variable_names is None:
        variable_names = [f"v{i}" for i in range(V)]

    # Standardise per variable
    mu = values_matrix.mean(axis=0, keepdims=True)
    sd = values_matrix.std(axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (values_matrix - mu) / sd                         # (N, V)

    L_idx, K_idx, lag_centers, angle_centers, _ = _bin_indices(
        coords, max_distance, n_lags, n_angle_bins,
    )

    L = n_lags
    K = n_angle_bins
    C = np.zeros((L, K, V, V), dtype=np.float64)
    n_pairs = np.zeros((L, K), dtype=np.int64)

    # Compute per-bin sums via boolean masks.  N=O(10³) keeps masks at
    # O(N²) ≈ 10⁶ booleans — well within memory.
    for li in range(L):
        for ki in range(K):
            mask = (L_idx == li) & (K_idx == ki)
            count = int(mask.sum())
            n_pairs[li, ki] = count
            if count == 0:
                continue
            # C[li, ki, i, j] = Σ_(k,l in bin) z_i(k) z_j(l) / count
            #                = (Z.T @ mask @ Z)[i,j] / count
            C[li, ki] = (Z.T @ mask.astype(np.float64) @ Z) / count

    return CrossCorrelogramTensor(
        C=C,
        lag_distances=lag_centers,
        angle_centers_rad=angle_centers,
        n_pairs=n_pairs,
        variable_names=list(variable_names),
        n_angle_bins=n_angle_bins,
    )


# ---------------------------------------------------------------------------
# Sym / antisym decomposition
# ---------------------------------------------------------------------------


def decompose_sym_antisym(
    tensor: CrossCorrelogramTensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(C_sym, C_anti)``, both of shape ``(L, K, V, V)``.

    For ``n_angle_bins == 1``: ``C_sym = C``, ``C_anti = 0`` (no opposite
    bin available).

    For directional tensors, with bin ``K`` and its opposite
    ``K′ = (K + n_angle_bins/2) mod n_angle_bins``:

        C_sym(L, K)[i,j]  = ½ ( C[L,K,i,j] + C[L,K′,i,j] )
        C_anti(L, K)[i,j] = ½ ( C[L,K,i,j] − C[L,K′,i,j] )

    These correspond to the symmetric / antisymmetric parts of the
    cross-covariance ``C_{ij}(+r)`` vs ``C_{ij}(−r)``.  An advective
    coupling ``Y(x) = X(x − v)`` produces a positive antisymmetric peak
    in the bin aligned with ``+v``.

    Note: keeping ``(i,j)`` fixed (rather than transposing) is essential.
    Because the signed-angle binning sends pair ``(k,l)`` (in bin ``K``)
    and pair ``(l,k)`` (in bin ``K′``) to opposite bins, we already have
    ``C[L,K,i,j] = C[L,K′,j,i]`` exactly; transposing ``(i,j)`` would
    therefore yield an identically-zero antisymmetric block.
    """
    C = tensor.C
    if tensor.n_angle_bins == 1:
        return C.copy(), np.zeros_like(C)

    K = tensor.n_angle_bins
    half = K // 2
    opp = (np.arange(K) + half) % K
    # C_opp[L, K, i, j] = C[L, K′, i, j]   (bin swap; i,j unchanged)
    C_opp = C[:, opp, :, :]
    C_sym = 0.5 * (C + C_opp)
    C_anti = 0.5 * (C - C_opp)
    return C_sym, C_anti


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------


def _peak_stat(C_block: np.ndarray) -> float:
    """Max |value| over all bins of a (L, K) block."""
    return float(np.max(np.abs(C_block))) if C_block.size else 0.0


def _cross_C_from_indices(
    Z2: np.ndarray,
    L_idx: np.ndarray,
    K_idx: np.ndarray,
    n_lags: int,
    n_angle_bins: int,
) -> np.ndarray:
    """Fast 2-variable cross-correlogram tensor given precomputed bin indices.

    Computes the (L, K, 2, 2) tensor without ever rebuilding the O(N²)
    distance matrix or the per-(L,K) boolean masks at full resolution.

    Parameters
    ----------
    Z2 : (N, 2) standardised values for variables (i, j).
    L_idx, K_idx : (N, N) integer index matrices from :func:`_bin_indices`.
        ``L_idx == n_lags`` flags pairs to skip.
    """
    L = n_lags
    K = n_angle_bins
    C = np.zeros((L, K, 2, 2), dtype=np.float64)
    n_pairs = np.zeros((L, K), dtype=np.int64)
    # Build a single (L, K)-aware composite index, masking out skipped pairs.
    for li in range(L):
        # Pre-filter rows that contribute to this lag bin to shrink the mask.
        lag_mask = (L_idx == li)
        if not lag_mask.any():
            continue
        for ki in range(K):
            mask = lag_mask & (K_idx == ki)
            count = int(mask.sum())
            n_pairs[li, ki] = count
            if count == 0:
                continue
            C[li, ki] = (Z2.T @ mask.astype(np.float64) @ Z2) / count
    return C


def _decompose_sym_antisym_array(C: np.ndarray, n_angle_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Sym/antisym split for a raw (L, K, V, V) array (no dataclass wrapper)."""
    if n_angle_bins == 1:
        return C.copy(), np.zeros_like(C)
    half = n_angle_bins // 2
    opp = (np.arange(n_angle_bins) + half) % n_angle_bins
    C_opp = C[:, opp, :, :]
    return 0.5 * (C + C_opp), 0.5 * (C - C_opp)


def _permute_pair(
    i: int,
    j: int,
    coords: np.ndarray,
    values_matrix: np.ndarray,
    *,
    max_distance: float,
    n_lags: int,
    n_angle_bins: int,
    n_perm: int,
    seed: int,
    peak_sym_obs: float,
    peak_anti_obs: float,
    L_idx: np.ndarray | None = None,
    K_idx: np.ndarray | None = None,
) -> tuple[tuple[int, int], dict]:
    """Worker for :func:`permutation_pvalues`: run ``n_perm`` shuffles of
    variable ``j`` against variable ``i`` and count peaks ≥ observed.

    When ``L_idx`` / ``K_idx`` are supplied (preferred path), the worker
    reuses precomputed spatial bin assignments and avoids the O(N²)
    distance / mask reconstruction that otherwise dominates memory.
    Pulled out to a top-level function so it pickles cleanly under
    ``joblib`` / ``multiprocessing`` backends.
    """
    rng = np.random.default_rng(seed)
    vij = values_matrix[:, [i, j]].astype(np.float64, copy=True)
    # Drop NaN rows once (must keep coords / indices aligned).
    valid = ~np.isnan(vij).any(axis=1)
    if L_idx is not None and K_idx is not None and not valid.all():
        vij = vij[valid]
        L_idx = L_idx[np.ix_(valid, valid)]
        K_idx = K_idx[np.ix_(valid, valid)]
    n_ge_sym = 0
    n_ge_anti = 0
    use_fast = L_idx is not None and K_idx is not None
    for _ in range(n_perm):
        vij_perm = vij.copy()
        rng.shuffle(vij_perm[:, 1])
        if use_fast:
            mu = vij_perm.mean(axis=0, keepdims=True)
            sd = vij_perm.std(axis=0, keepdims=True)
            sd = np.where(sd > 0, sd, 1.0)
            Z2 = (vij_perm - mu) / sd
            C = _cross_C_from_indices(Z2, L_idx, K_idx, n_lags, n_angle_bins)
            csp, cap = _decompose_sym_antisym_array(C, n_angle_bins)
        else:
            tperm = compute_cross_correlogram(
                coords, vij_perm,
                max_distance=max_distance, n_lags=n_lags,
                n_angle_bins=n_angle_bins,
            )
            csp, cap = decompose_sym_antisym(tperm)
        # In the 2-var subset the (i, j) slice is at index (0, 1)
        if _peak_stat(csp[:, :, 0, 1]) >= peak_sym_obs:
            n_ge_sym += 1
        if _peak_stat(cap[:, :, 0, 1]) >= peak_anti_obs:
            n_ge_anti += 1
    return (i, j), {
        "peak_sym_obs": peak_sym_obs,
        "peak_anti_obs": peak_anti_obs,
        "p_sym": (n_ge_sym + 1) / (n_perm + 1),
        "p_anti": (n_ge_anti + 1) / (n_perm + 1),
        "n_perm": int(n_perm),
    }


def _ram_capped_n_jobs(n_jobs: int, n_points: int, n_lags: int, n_angle_bins: int) -> int:
    """Cap ``n_jobs`` so peak per-worker memory fits in available RAM.

    Per-worker peak is dominated by the (N, N) bool / float bin masks
    constructed inside :func:`_cross_C_from_indices`. Empirically each
    worker holds ~3 such matrices live (L_idx + K_idx + transient mask).
    """
    if n_jobs <= 1:
        return max(1, n_jobs)
    try:
        import psutil  # type: ignore
        avail_bytes = int(psutil.virtual_memory().available)
    except Exception:
        return n_jobs
    # 8 bytes/int64 mask + small overhead; use 4× safety factor.
    per_worker_bytes = 4 * 8 * n_points * n_points
    if per_worker_bytes <= 0:
        return n_jobs
    cap = max(1, avail_bytes // per_worker_bytes)
    return int(min(n_jobs, cap))


def permutation_pvalues(
    coords: np.ndarray,
    values_matrix: np.ndarray,
    pair_indices: list[tuple[int, int]],
    *,
    max_distance: float,
    n_lags: int,
    n_angle_bins: int = 1,
    n_perm: int = 100,
    seed: int = 0,
    observed_sym: Optional[np.ndarray] = None,
    observed_anti: Optional[np.ndarray] = None,
    n_jobs: int = 1,
) -> dict[tuple[int, int], dict]:
    """Empirical permutation p-values for per-pair peak magnitudes.

    For each ``(i, j)`` in ``pair_indices`` (i ≠ j), permute the spatial
    labels of variable ``j`` (i.e. shuffle its values across points),
    recompute the cross-correlogram tensor restricted to that pair, and
    record the peak magnitudes of the resulting ``C_sym[…,i,j]`` and
    ``C_anti[…,i,j]`` slices.  The p-value is the fraction of permuted
    peaks ≥ the observed peak.

    When ``n_jobs > 1``, pairs are processed in parallel via joblib's
    loky backend (falls back to a serial loop if joblib is unavailable
    or only one pair is supplied).  Each worker gets a deterministic
    sub-seed derived from ``seed`` so results stay reproducible.

    Returns ``{(i, j): {"p_sym": …, "p_anti": …, "peak_sym_obs": …,
    "peak_anti_obs": …}}``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values_matrix = np.asarray(values_matrix, dtype=np.float64)

    # Compute observed if not supplied
    if observed_sym is None or observed_anti is None:
        obs_tensor = compute_cross_correlogram(
            coords, values_matrix,
            max_distance=max_distance, n_lags=n_lags,
            n_angle_bins=n_angle_bins,
        )
        observed_sym, observed_anti = decompose_sym_antisym(obs_tensor)

    # Pre-compute observed peaks per pair so workers don't see the
    # (potentially large) full tensor.
    pair_jobs = []
    for (i, j) in pair_indices:
        peak_sym_obs = _peak_stat(observed_sym[:, :, i, j])
        peak_anti_obs = _peak_stat(observed_anti[:, :, i, j])
        pair_jobs.append((i, j, peak_sym_obs, peak_anti_obs))

    # Precompute spatial bin indices once -- they depend only on coords
    # and the lag/angle binning, not on the (shuffled) values. This is the
    # single biggest memory + speed win versus rebuilding them inside every
    # worker permutation (which previously triggered OOM-kills under loky).
    try:
        L_idx_full, K_idx_full, _, _, _ = _bin_indices(
            coords, max_distance, n_lags, n_angle_bins,
        )
    except Exception as exc:
        logger.warning("Failed to precompute bin indices (%s); falling back to per-call computation.", exc)
        L_idx_full = None
        K_idx_full = None

    n_pairs = len(pair_jobs)
    requested_n_jobs = int(n_jobs) if n_jobs else 1
    capped_n_jobs = _ram_capped_n_jobs(
        requested_n_jobs, coords.shape[0], n_lags, n_angle_bins,
    )
    if capped_n_jobs < requested_n_jobs:
        print(
            f"  Cross-correlogram permutation: capping n_jobs {requested_n_jobs} -> {capped_n_jobs} "
            f"based on available RAM",
            flush=True,
        )
    use_parallel = capped_n_jobs > 1 and n_pairs > 1
    parallel = None
    if use_parallel:
        try:
            from joblib import Parallel, delayed  # type: ignore
            parallel = (Parallel, delayed)
        except Exception:
            parallel = None

    def _run_serial() -> dict[tuple[int, int], dict]:
        progress_every = 1 if n_pairs <= 10 else 5
        out_serial: dict[tuple[int, int], dict] = {}
        for k, (i, j, ps, pa) in enumerate(pair_jobs):
            key, payload = _permute_pair(
                i, j, coords, values_matrix,
                max_distance=max_distance, n_lags=n_lags,
                n_angle_bins=n_angle_bins, n_perm=n_perm,
                seed=int(seed) + 1000 * (k + 1),
                peak_sym_obs=ps, peak_anti_obs=pa,
                L_idx=L_idx_full, K_idx=K_idx_full,
            )
            out_serial[key] = payload
            if (k + 1) % progress_every == 0 or (k + 1) == n_pairs:
                print(
                    f"  Cross-correlogram permutation: {k + 1}/{n_pairs} pairs complete",
                    flush=True,
                )
        return out_serial

    if parallel is not None:
        Parallel, delayed = parallel
        print(
            f"  Cross-correlogram permutation: {n_pairs} pairs × {n_perm} "
            f"shuffles each, n_jobs={capped_n_jobs}",
            flush=True,
        )
        try:
            results = Parallel(
                n_jobs=capped_n_jobs,
                backend="loky",
                max_nbytes="50M",
                mmap_mode="r",
            )(
                delayed(_permute_pair)(
                    i, j, coords, values_matrix,
                    max_distance=max_distance, n_lags=n_lags,
                    n_angle_bins=n_angle_bins, n_perm=n_perm,
                    seed=int(seed) + 1000 * (k + 1),
                    peak_sym_obs=ps, peak_anti_obs=pa,
                    L_idx=L_idx_full, K_idx=K_idx_full,
                )
                for k, (i, j, ps, pa) in enumerate(pair_jobs)
            )
        except Exception as exc:  # noqa: BLE001 -- includes TerminatedWorkerError, BrokenProcessPool
            logger.warning(
                "Cross-correlogram parallel pool failed (%s); retrying serially.",
                exc,
            )
            print(
                f"  [WARN] Cross-correlogram parallel pool crashed ({type(exc).__name__}); "
                f"falling back to serial.",
                flush=True,
            )
            return _run_serial()
        out: dict[tuple[int, int], dict] = {key: val for key, val in results}
        print(
            f"  Cross-correlogram permutation: {n_pairs}/{n_pairs} pairs complete",
            flush=True,
        )
        return out

    # Serial fallback with progress logging every 5 pairs (or every pair
    # for small batches) so the user sees movement during the long
    # single-threaded grind.
    print(
        f"  Cross-correlogram permutation: {n_pairs} pairs × {n_perm} "
        f"shuffles each (serial)",
        flush=True,
    )
    return _run_serial()


# ---------------------------------------------------------------------------
# Per-pair summary
# ---------------------------------------------------------------------------


def per_pair_summary(
    tensor: CrossCorrelogramTensor,
    *,
    noise_floor: float = 0.05,
    significance_alpha: float = 0.05,
    perm_results: Optional[dict[tuple[int, int], dict]] = None,
) -> dict:
    """Assemble per-pair summary statistics from the cross-tensor.

    Returns
    -------
    dict
        ``{
            "variable_names": [...],
            "n_angle_bins": int,
            "lag_distances": [...],
            "angle_centers_deg": [...],
            "pairs": {
                "i_name|j_name": {
                    "i": int, "j": int,
                    "peak_sym": float, "peak_lag_sym_m": float,
                    "peak_anti": float, "peak_lag_anti_m": float,
                    "peak_angle_anti_deg": float | None,
                    "effective_range_sym_m": float | None,
                    "p_sym": float | None, "p_anti": float | None,
                    "significant_sym": bool, "significant_anti": bool,
                },
                ...
            },
            "antisymmetric_flags": [...],   # pair keys with significant antisymmetry
        }``
    """
    C_sym, C_anti = decompose_sym_antisym(tensor)
    L, K, V, _ = tensor.C.shape
    pairs: dict[str, dict] = {}
    flags_anti: list[str] = []

    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            blk_sym = C_sym[:, :, i, j]            # (L, K)
            blk_anti = C_anti[:, :, i, j]
            # Pool symmetric block over angle to get an isotropic curve
            sym_iso = blk_sym.mean(axis=1)         # (L,)
            peak_sym = float(np.max(np.abs(sym_iso))) if sym_iso.size else 0.0
            peak_lag_sym_idx = int(np.argmax(np.abs(sym_iso))) if sym_iso.size else 0
            peak_lag_sym = float(tensor.lag_distances[peak_lag_sym_idx]) if sym_iso.size else 0.0

            # Effective range: first lag where |sym_iso(h)| crosses below
            # noise_floor AFTER the peak.
            eff_range = None
            if sym_iso.size:
                for li in range(peak_lag_sym_idx, L):
                    if abs(sym_iso[li]) < noise_floor:
                        eff_range = float(tensor.lag_distances[li])
                        break

            # Antisymmetric peak (over both lag and angle when directional)
            if blk_anti.size:
                flat_idx = int(np.argmax(np.abs(blk_anti)))
                li_a, ki_a = np.unravel_index(flat_idx, blk_anti.shape)
                peak_anti = float(np.abs(blk_anti[li_a, ki_a]))
                peak_lag_anti = float(tensor.lag_distances[li_a])
                peak_angle_anti = (
                    float(math.degrees(tensor.angle_centers_rad[ki_a]))
                    if tensor.is_directional else None
                )
            else:
                peak_anti = 0.0
                peak_lag_anti = 0.0
                peak_angle_anti = None

            p_sym = None
            p_anti = None
            if perm_results is not None and (i, j) in perm_results:
                p_sym = float(perm_results[(i, j)]["p_sym"])
                p_anti = float(perm_results[(i, j)]["p_anti"])

            sig_sym = bool(p_sym is not None and p_sym < significance_alpha)
            sig_anti = bool(p_anti is not None and p_anti < significance_alpha)

            key = f"{tensor.variable_names[i]}|{tensor.variable_names[j]}"
            pairs[key] = {
                "i": int(i), "j": int(j),
                "i_name": tensor.variable_names[i],
                "j_name": tensor.variable_names[j],
                "peak_sym": peak_sym,
                "peak_lag_sym_m": peak_lag_sym,
                "peak_anti": peak_anti,
                "peak_lag_anti_m": peak_lag_anti,
                "peak_angle_anti_deg": peak_angle_anti,
                "effective_range_sym_m": eff_range,
                "p_sym": p_sym,
                "p_anti": p_anti,
                "significant_sym": sig_sym,
                "significant_anti": sig_anti,
            }
            if sig_anti:
                flags_anti.append(key)

    return {
        "variable_names": list(tensor.variable_names),
        "n_angle_bins": int(tensor.n_angle_bins),
        "lag_distances": [float(x) for x in tensor.lag_distances.tolist()],
        "angle_centers_deg": [
            float(math.degrees(x)) for x in tensor.angle_centers_rad.tolist()
        ],
        "pairs": pairs,
        "antisymmetric_flags": flags_anti,
        "n_pairs_per_bin": tensor.n_pairs.tolist(),
    }


# ---------------------------------------------------------------------------
# Phase 4: per-pair effective-range matrix + bandwidth-mismatch diagnostic
# ---------------------------------------------------------------------------


def build_effective_range_matrix(
    summary: dict,
    *,
    auto_bandwidths: Optional[dict[str, float]] = None,
    mismatch_factor: float = 10.0,
    require_significance: bool = True,
) -> dict:
    """Build the V×V effective-range matrix and bandwidth-mismatch warnings.

    Parameters
    ----------
    summary : dict
        Output of :func:`per_pair_summary` / :func:`compute_and_summarise`.
    auto_bandwidths : dict[var_name → float], optional
        Per-variable marginal effective range (e.g. the legacy
        ``effective_range`` produced by ``CorrelogramSpatialAnalyzer``).
        When supplied, mismatches between the per-pair range and either
        endpoint's marginal range that exceed ``mismatch_factor`` raise a
        warning entry.
    mismatch_factor : float, default 10.0
        A pair is flagged when ``range_pair / range_marginal`` is outside
        ``[1/mismatch_factor, mismatch_factor]`` for either endpoint
        variable.
    require_significance : bool, default True
        When True, only pairs flagged as ``significant_sym`` (Phase 3
        permutation null) contribute non-NaN entries to the matrix.

    Returns
    -------
    dict with keys:
        ``variable_names`` : list[str]
        ``range_matrix`` : (V, V) list of floats / None — symmetric in (i,j);
            diagonal is filled from ``auto_bandwidths`` when supplied.
        ``mismatch_factor`` : float
        ``mismatch_warnings`` : list of dicts ``{pair, range_pair_m,
            marginal_i_m, marginal_j_m, ratio_max}``.
        ``significant_pair_count`` : int
    """
    var_names = list(summary["variable_names"])
    V = len(var_names)
    pairs = summary.get("pairs", {})
    M: list[list[Optional[float]]] = [[None] * V for _ in range(V)]

    # Diagonal = marginal effective range
    if auto_bandwidths:
        for i, name in enumerate(var_names):
            r = auto_bandwidths.get(name)
            if r is not None and r > 0:
                M[i][i] = float(r)

    sig_count = 0
    warnings: list[dict] = []

    for key, p in pairs.items():
        i = int(p["i"])
        j = int(p["j"])
        rng = p.get("effective_range_sym_m")
        sig = bool(p.get("significant_sym", False))
        if require_significance and not sig:
            continue
        if rng is None:
            continue
        sig_count += 1
        # Symmetrise: keep the smaller (more conservative) range when both
        # (i,j) and (j,i) report a value.
        existing = M[i][j]
        if existing is None or rng < existing:
            M[i][j] = float(rng)
            M[j][i] = float(rng)

        if auto_bandwidths:
            r_i = auto_bandwidths.get(var_names[i])
            r_j = auto_bandwidths.get(var_names[j])
            ratios = []
            for r_marg in (r_i, r_j):
                if r_marg and r_marg > 0:
                    ratios.append(max(rng / r_marg, r_marg / rng))
            if ratios and max(ratios) > mismatch_factor:
                warnings.append({
                    "pair": f"{var_names[i]}|{var_names[j]}",
                    "range_pair_m": float(rng),
                    "marginal_i_m": float(r_i) if r_i else None,
                    "marginal_j_m": float(r_j) if r_j else None,
                    "ratio_max": float(max(ratios)),
                })

    return {
        "variable_names": var_names,
        "range_matrix": M,
        "mismatch_factor": float(mismatch_factor),
        "mismatch_warnings": warnings,
        "significant_pair_count": int(sig_count),
        "diagonal_source": "auto_bandwidths" if auto_bandwidths else None,
    }


def aggregate_outcome_cross_ranges(
    matrix_payload: dict,
    target_variable: str,
) -> dict:
    """Return the cross-ranges between ``target_variable`` and all others.

    Useful for spatial-CV block-size selection: the CV block must be as
    large as the maximum (target ↔ predictor) effective range so that
    cross-variable autocorrelation does not leak across folds.
    """
    var_names = list(matrix_payload["variable_names"])
    M = matrix_payload["range_matrix"]
    if target_variable not in var_names:
        return {"target_variable": target_variable, "cross_ranges": {},
                "max_cross_range_m": None}
    t = var_names.index(target_variable)
    out: dict[str, float] = {}
    for j, name in enumerate(var_names):
        if j == t:
            continue
        v = M[t][j]
        if v is not None:
            out[name] = float(v)
    max_r = max(out.values()) if out else None
    return {
        "target_variable": target_variable,
        "cross_ranges": out,
        "max_cross_range_m": max_r,
    }


# ---------------------------------------------------------------------------
# Convenience: end-to-end
# ---------------------------------------------------------------------------
def compute_and_summarise(
    coords: np.ndarray,
    values_matrix: np.ndarray,
    variable_names: list[str],
    *,
    max_distance: float,
    n_lags: int = 12,
    n_angle_bins: int = 1,
    n_perm: int = 0,
    significance_alpha: float = 0.05,
    noise_floor: float = 0.05,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    """Compute tensor → decompose → (optional permutation null) → summary.

    When ``n_perm == 0`` the permutation step is skipped and per-pair
    p-values are reported as ``None``.  ``n_jobs`` is forwarded to the
    permutation helper for parallel per-pair workers.
    """
    tensor = compute_cross_correlogram(
        coords, values_matrix,
        max_distance=max_distance, n_lags=n_lags,
        n_angle_bins=n_angle_bins,
        variable_names=variable_names,
    )
    perm_results = None
    if n_perm > 0:
        V = values_matrix.shape[1]
        # Off-diagonal pairs only
        pair_idx = [(i, j) for i in range(V) for j in range(V) if i != j]
        C_sym_obs, C_anti_obs = decompose_sym_antisym(tensor)
        perm_results = permutation_pvalues(
            coords, values_matrix, pair_idx,
            max_distance=max_distance, n_lags=n_lags,
            n_angle_bins=n_angle_bins,
            n_perm=n_perm, seed=seed,
            observed_sym=C_sym_obs, observed_anti=C_anti_obs,
            n_jobs=n_jobs,
        )
    return per_pair_summary(
        tensor,
        noise_floor=noise_floor,
        significance_alpha=significance_alpha,
        perm_results=perm_results,
    )
