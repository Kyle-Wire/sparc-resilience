"""
E5 Eigenmap Prep — Laplacian Eigenmaps + Fuzzy C-Means per training city.

Computes LAPEIG_1 and LAPEIG_2 (Fiedler vector + secondary) for each city,
sign-aligns them so LAPEIG_1 is positive at the urban core, runs Fuzzy C-Means
with k=4 clusters, and saves a PNG for visual inspection.

Usage:
    python scripts/e5_eigenmap_prep.py

Outputs:
    output/diagnostics/e5_<slug>_eigenmaps.png   -- spatial map of LAPEIG_1/2 + cluster memberships
    output/diagnostics/e5_summary.csv            -- per-city eigenvalue variance explained

The FCM membership tensors are not cached here -- they're recomputed at
pretraining time via compute_city_fcm_memberships() in train_multicity_jepa.py.
This script is for inspection only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "multicity_pilot.yml"
OUTPUT_DIR  = ROOT / "output"
DIAG_DIR    = OUTPUT_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

K_CLUSTERS  = 4      # Fuzzy C-Means clusters (core / transition / suburban / rural)
N_EIGEN     = 2      # LAPEIG_1 and LAPEIG_2 (Fiedler + secondary)
FCM_M       = 2.0    # fuzziness exponent (standard)
FCM_ITERS   = 100
FCM_EPS     = 1e-6
MAX_N_EIGEN = 15_000 # subsample threshold for eigsh — O(N^2) KNN is too slow above this


# ---------------------------------------------------------------------------
# Eigenmap helpers
# ---------------------------------------------------------------------------

def build_laplacian_eigenvectors(coords: np.ndarray, n: int = 2, k_nn: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvectors, eigenvalues) for the first n non-trivial modes.

    Uses normalised graph Laplacian on a KNN spatial weight matrix.
    Skips the trivial constant eigenvector (eigenvalue ≈ 0).

    Returns
    -------
    vecs : (N, n) float64 — eigenvectors, columns ordered by eigenvalue (ascending)
    vals : (n,) float64  — corresponding eigenvalues
    """
    from libpysal.weights import KNN as PysalKNN
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    N = coords.shape[0]
    k_eff = min(k_nn, N - 1)

    w = PysalKNN.from_array(coords, k=k_eff)
    L = laplacian(w.sparse, normed=True)

    # Request n+1 smallest eigenvalues so we can skip the trivial one
    k_req = min(n + 1, L.shape[0] - 2)
    vals_all, vecs_all = eigsh(L, k=k_req, which="SM")

    # Sort ascending (eigsh may not return sorted)
    order = np.argsort(vals_all)
    vals_all = vals_all[order]
    vecs_all = vecs_all[:, order]

    # Skip first trivial eigenvector (constant, eigenvalue ≈ 0)
    vecs = vecs_all[:, 1: n + 1]
    vals = vals_all[1: n + 1]

    return vecs, vals


def sign_align_fiedler(vecs: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Flip each eigenvector sign so it correlates positively with urban density proxy.

    Proxy: distance from city centroid (negated, so center=high value).
    Convention: LAPEIG_i should be HIGH at urban core (short distance from centroid).
    If the eigenvector correlates POSITIVELY with distance (high=periphery), flip it.
    """
    centroid = coords.mean(axis=0)
    dist = np.linalg.norm(coords - centroid, axis=1)  # (N,) — large = periphery

    aligned = vecs.copy()
    for i in range(vecs.shape[1]):
        # Positive correlation with dist means eigenvector is high at periphery — flip
        r = float(np.corrcoef(vecs[:, i], dist)[0, 1])
        if r > 0:
            aligned[:, i] = -vecs[:, i]
    return aligned


# ---------------------------------------------------------------------------
# Fuzzy C-Means
# ---------------------------------------------------------------------------

def fuzzy_cmeans(Z: np.ndarray, k: int, m: float = 2.0, max_iter: int = 100, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy C-Means clustering.

    Parameters
    ----------
    Z       : (N, d) feature matrix (eigenvectors, already sign-aligned)
    k       : number of clusters
    m       : fuzziness exponent (m=2 standard)
    max_iter: maximum iterations
    eps     : convergence threshold (max change in membership matrix)

    Returns
    -------
    U        : (N, k) membership matrix (rows sum to 1)
    centers  : (k, d) cluster centroids
    """
    rng = np.random.default_rng(42)
    N, d = Z.shape

    # Initialize membership matrix randomly (rows sum to 1)
    U = rng.dirichlet(np.ones(k), size=N).astype(np.float64)

    centers = np.zeros((k, d))
    for _ in range(max_iter):
        # Update centers
        Um = U ** m                          # (N, k) — raised to fuzziness power
        denom = Um.sum(axis=0)               # (k,)
        centers = (Um.T @ Z) / denom[:, None]   # (k, d)

        # Update memberships
        U_new = np.zeros_like(U)
        for i in range(N):
            dists = np.linalg.norm(Z[i] - centers, axis=1) ** 2   # (k,)
            dists = np.maximum(dists, 1e-10)                        # avoid /0
            for j in range(k):
                U_new[i, j] = 1.0 / np.sum((dists[j] / dists) ** (1.0 / (m - 1)))

        if np.max(np.abs(U_new - U)) < eps:
            U = U_new
            break
        U = U_new

    return U, centers


def sort_clusters_by_fiedler(U: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-order clusters so cluster-0 has the highest LAPEIG_1 centroid (urban core).

    This makes cluster ordering consistent across cities:
      Cluster 0 = densest urban core (highest LAPEIG_1)
      Cluster 3 = rural/suburban periphery (lowest LAPEIG_1)
    """
    order = np.argsort(centers[:, 0])[::-1]   # descending by LAPEIG_1 centroid
    return U[:, order], centers[order]


def _fcm_membership_from_centers(Z: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Recompute FCM memberships for any set of points given pre-fitted centers.

    Used to interpolate memberships from a subsample to the full city dataset.
    Identical to the FCM update formula — no iteration needed given fixed centers.

    Parameters
    ----------
    Z       : (N, d) points in eigenspace
    centers : (k, d) cluster centroids
    m       : fuzziness exponent

    Returns
    -------
    U : (N, k) float64 membership matrix (rows sum to 1)
    """
    N = Z.shape[0]
    k = centers.shape[0]
    # Squared distances from each point to each center: (N, k)
    dists = np.stack(
        [np.sum((Z - centers[j]) ** 2, axis=1) for j in range(k)],
        axis=1,
    )
    dists = np.maximum(dists, 1e-10)
    U = np.zeros((N, k), dtype=np.float64)
    exp = 1.0 / (m - 1)
    for j in range(k):
        U[:, j] = 1.0 / np.sum((dists[:, j: j + 1] / dists) ** exp, axis=1)
    return U


# ---------------------------------------------------------------------------
# Per-city pipeline
# ---------------------------------------------------------------------------

def compute_city_fcm_memberships(
    coords: np.ndarray,
    n_eigen: int = N_EIGEN,
    k_clusters: int = K_CLUSTERS,
    k_nn: int = 8,
    max_n: int = MAX_N_EIGEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline: coords → eigenmaps → FCM memberships.

    For cities with N > max_n pixels, spatially subsamples to max_n pixels
    for the expensive eigsh + KNN computation, then interpolates memberships
    back to all N pixels via nearest-neighbour assignment in coordinate space
    followed by FCM distance recomputation from fixed centers.

    Parameters
    ----------
    coords    : (N, 2) UTM coordinates (float64 recommended)
    n_eigen   : number of non-trivial eigenvectors to use (default 2)
    k_clusters: number of fuzzy clusters (default 4)
    k_nn      : KNN neighbours for spatial weight matrix
    max_n     : subsample threshold; above this eigsh is too slow

    Returns
    -------
    U        : (N, k_clusters) float32 membership matrix
    vecs     : (N, n_eigen) sign-aligned eigenvectors at all N points
    centers  : (k_clusters, n_eigen) cluster centroids in eigenspace
    """
    from sklearn.neighbors import NearestNeighbors

    N = coords.shape[0]
    rng = np.random.default_rng(42)

    if N > max_n:
        sub_idx = rng.choice(N, size=max_n, replace=False)
        coords_sub = coords[sub_idx]
        print(f"  subsampled {N:,} -> {max_n:,} for eigsh")
    else:
        sub_idx = np.arange(N)
        coords_sub = coords

    # Eigenvectors on subsample
    vecs_sub, _vals = build_laplacian_eigenvectors(coords_sub, n=n_eigen, k_nn=k_nn)
    vecs_sub = sign_align_fiedler(vecs_sub, coords_sub)

    # FCM on subsample — derives the cluster centers
    U_sub, centers = fuzzy_cmeans(vecs_sub, k=k_clusters, m=FCM_M, max_iter=FCM_ITERS, eps=FCM_EPS)
    U_sub, centers = sort_clusters_by_fiedler(U_sub, centers)

    if N > max_n:
        # Interpolate eigenvectors to full N via nearest-subsample-point assignment.
        # KD-tree lookup: O(N log max_n), fast even for 1.6M pixels.
        nn_model = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords_sub)
        _, nn_idx = nn_model.kneighbors(coords)
        nn_idx = nn_idx.ravel()           # (N,) — index into subsample

        vecs_full = vecs_sub[nn_idx]      # (N, n_eigen) — nearest-neighbor propagation

        # Recompute memberships for all N from the fixed centers
        U_full = _fcm_membership_from_centers(vecs_full, centers, m=FCM_M)
    else:
        vecs_full = vecs_sub
        U_full    = U_sub

    return U_full.astype(np.float32), vecs_full.astype(np.float32), centers.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation (for inspection only)
# ---------------------------------------------------------------------------

def _plot_city(slug: str, coords: np.ndarray, vecs: np.ndarray, U: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        K = U.shape[1]
        fig = plt.figure(figsize=(4 * (3 + K), 4), constrained_layout=True)
        gs  = GridSpec(1, 3 + K, figure=fig)

        def scatter(ax, vals, title, cmap="RdYlBu_r"):
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, s=3, rasterized=True)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)

        scatter(plt.subplot(gs[0]),          vecs[:, 0],  "LAPEIG_1 (Fiedler)")
        if vecs.shape[1] > 1:
            scatter(plt.subplot(gs[1]),      vecs[:, 1],  "LAPEIG_2")
        cluster_labels = U.argmax(axis=1)
        scatter(plt.subplot(gs[2]), cluster_labels, f"Hard label (k={K})", cmap="tab10")
        for ci in range(K):
            scatter(plt.subplot(gs[3 + ci]), U[:, ci], f"U_cluster{ci} (core->rural)")

        fig.suptitle(f"E5 Eigenmaps + FCM — {slug}", fontsize=11, fontweight="bold")
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"  Saved -> {out_path.name}")
    except Exception as exc:
        print(f"  [warn] plot failed for {slug}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mc  = cfg["multicity"]
    coord_cols = mc.get("pipeline", {}).get("coordinate_columns", ["centroid_x", "centroid_y"])
    all_cities  = mc["pilot_cities"]
    train_cfgs  = [c for c in all_cities if not c.get("holdout", False)]

    rows = []
    for city_cfg in train_cfgs:
        slug = city_cfg["city_slug"]
        pq   = OUTPUT_DIR / "cities" / slug / "data.geoparquet"
        if not pq.exists():
            print(f"  [skip] {slug} — no GeoParquet")
            continue

        print(f"\n{slug} ...")
        gdf = gpd.read_parquet(str(pq))
        coords = gdf[coord_cols].values.astype(np.float64)
        N = len(coords)

        U, vecs, centers = compute_city_fcm_memberships(coords, n_eigen=N_EIGEN, k_clusters=K_CLUSTERS)

        # Variance "explained" by each eigenvector (as fraction of total spatial variance)
        total_var = coords.var(axis=0).sum()
        lapeig_vars = [(np.cov(vecs[:, i], coords[:, 0])[0, 1]**2 + np.cov(vecs[:, i], coords[:, 1])[0, 1]**2) for i in range(N_EIGEN)]

        # Cluster size distribution (hard assignments)
        hard = U.argmax(axis=1)
        cluster_fracs = [float((hard == ci).mean()) for ci in range(K_CLUSTERS)]

        rows.append({
            "slug": slug,
            "N": N,
            "fiedler_mean": float(vecs[:, 0].mean()),
            "fiedler_std": float(vecs[:, 0].std()),
            **{f"cluster_{ci}_frac": cluster_fracs[ci] for ci in range(K_CLUSTERS)},
        })

        print(f"  N={N}  LAPEIG_1 std={vecs[:,0].std():.4f}  clusters={[f'{f:.0%}' for f in cluster_fracs]}")

        # Visual check
        out_png = DIAG_DIR / f"e5_{slug}_eigenmaps.png"
        _plot_city(slug, coords, vecs, U, out_png)

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = DIAG_DIR / "e5_summary.csv"
        df.to_csv(str(csv_path), index=False)
        print(f"\nSummary -> {csv_path}")
        print(df[["slug", "N", "fiedler_std", "cluster_0_frac", "cluster_3_frac"]].to_string(index=False))
    else:
        print("No cities processed.")


if __name__ == "__main__":
    main()
