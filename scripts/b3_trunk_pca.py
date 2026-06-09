"""B3 -- Trunk PCA Diagnostic

Loads the JEPA trunk (frozen encoder), runs forward pass on ~2000 stratified
pixels per training city, projects to 2D via PCA, and emits:
  - output/diagnostics/trunk_pca.png     scatter coloured by city
  - output/diagnostics/trunk_centroids.json  pairwise Euclidean distance matrix

Goal: check if trunk embeddings are well-separated by city (good for
city-specific heads) and roughly clustered by climate zone (good for
Koppen-weighted LOO ensemble).

Usage:
    python scripts/b3_trunk_pca.py [--config CONFIG] [--n-per-city N]
"""
from __future__ import annotations
import argparse, json, sys, logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("b3_trunk_pca")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/multicity_pilot.yml")
    p.add_argument("--trunk", default="output/cities/jepa_pretrained_trunk.pt",
                   help="Path to trunk checkpoint (.pt)")
    p.add_argument("--n-per-city", type=int, default=2000,
                   help="Max pixels to sample per city (default: 2000)")
    p.add_argument("--output-dir", default="output/diagnostics")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_trunk(trunk_path: Path):
    """Load trunk encoder from checkpoint.

    The checkpoint format is:
      {'trunk_state': OrderedDict, 'in_dim': int, 'hidden_dim': int,
       'X_mean': list, 'X_std': list, 'C_mean': list, 'C_std': list}

    Returns (model, hidden_dim, device, X_mean_np, X_std_np).
    """
    import torch
    import torch.nn as nn
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(trunk_path), map_location=device)

    state     = ckpt["trunk_state"]
    input_dim = int(ckpt["in_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    X_mean = np.array(ckpt.get("X_mean", [0.0] * input_dim), dtype=np.float32)
    X_std  = np.array(ckpt.get("X_std",  [1.0] * input_dim), dtype=np.float32)
    X_std  = np.where(X_std < 1e-8, 1.0, X_std)

    class Trunk(nn.Module):
        def __init__(self, in_dim, hid_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid_dim), nn.GELU(),
                nn.LayerNorm(hid_dim),
                nn.Linear(hid_dim, hid_dim), nn.GELU(),
                nn.LayerNorm(hid_dim),
                nn.Linear(hid_dim, hid_dim),
            )
        def forward(self, x):
            return self.net(x)

    trunk = Trunk(input_dim, hidden_dim).to(device)
    trunk.load_state_dict(state, strict=False)
    trunk.eval()
    log.info("Trunk loaded: input_dim=%d hidden_dim=%d device=%s", input_dim, hidden_dim, device)
    return trunk, hidden_dim, device, X_mean, X_std


def _embed_city(city_cfg: dict, output_root: Path, trunk, device, n_per_city: int,
                rng: np.random.Generator, feature_cols: list[str],
                X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray | None:
    """Load city data.geoparquet, sample n_per_city rows, embed with trunk.

    Returns (N, hidden_dim) numpy array or None if data not available.
    """
    import torch
    slug = city_cfg["city_slug"]
    data_path = output_root / slug / "data.geoparquet"
    if not data_path.exists():
        log.warning("[%s] data.geoparquet not found -- skipped", slug)
        return None

    try:
        import geopandas as gpd
        gdf = gpd.read_parquet(str(data_path))
    except Exception as exc:
        log.warning("[%s] failed to read data: %s", slug, exc)
        return None

    if not feature_cols:
        # Use all numeric non-geometry columns
        feature_cols_use = [c for c in gdf.columns
                            if gdf[c].dtype.kind in ("f", "i")
                            and c not in ("geometry",)]
    else:
        feature_cols_use = [c for c in feature_cols if c in gdf.columns]

    if not feature_cols_use:
        log.warning("[%s] no feature columns found -- skipped", slug)
        return None

    n_trunk_feats = len(X_mean)
    X_raw = gdf[feature_cols_use].values.astype(np.float32)

    # Pad/truncate X to match trunk's in_dim using saved normalization stats
    n_cols = X_raw.shape[1]
    if n_cols >= n_trunk_feats:
        X_raw = X_raw[:, :n_trunk_feats]
        mean_use = X_mean[:n_trunk_feats]
        std_use  = X_std[:n_trunk_feats]
    else:
        # Pad with zeros (normalized space) for missing columns
        X_pad = np.zeros((len(X_raw), n_trunk_feats), dtype=np.float32)
        X_pad[:, :n_cols] = X_raw
        X_raw = X_pad
        mean_use = X_mean
        std_use  = X_std

    # Normalize with saved trunk stats then nan→0
    X = np.nan_to_num((X_raw - mean_use) / std_use, nan=0.0)

    # Stratified sample (simple random without replacement)
    n = min(n_per_city, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X = X[idx]

    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        emb = trunk(Xt).cpu().numpy()
    log.info("[%s] embedded %d pixels -> (%d, %d)", slug, n, emb.shape[0], emb.shape[1])
    return emb


def main():
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        pilot_cfg = yaml.safe_load(fh)

    mc = pilot_cfg["multicity"]
    output_root = Path(mc["output_dir"])
    diag_dir = Path(args.output_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    train_cfgs = [c for c in mc["pilot_cities"] if not c.get("holdout", False)]
    feature_cols = pilot_cfg.get("feature_columns", [])
    if not feature_cols:
        # Fallback: read from first city's data
        log.warning("feature_columns not found in config; will use any shared columns")
        feature_cols = []

    trunk_path = Path(args.trunk)
    if not trunk_path.exists():
        log.error("Trunk not found: %s", trunk_path)
        sys.exit(1)

    trunk, hidden_dim, device, X_mean, X_std = _load_trunk(trunk_path)

    # Embed each city
    city_embeddings: dict[str, np.ndarray] = {}
    for city_cfg in train_cfgs:
        emb = _embed_city(city_cfg, output_root, trunk, device, args.n_per_city, rng,
                          feature_cols, X_mean, X_std)
        if emb is not None:
            city_embeddings[city_cfg["city_slug"]] = emb

    if not city_embeddings:
        log.error("No embeddings generated.")
        sys.exit(1)

    log.info("Collected embeddings for %d cities", len(city_embeddings))

    # Stack all embeddings for PCA
    all_emb = np.vstack(list(city_embeddings.values()))
    city_labels = []
    for slug, emb in city_embeddings.items():
        city_labels.extend([slug] * len(emb))
    city_labels = np.array(city_labels)

    # PCA to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=args.seed)
    coords_2d = pca.fit_transform(all_emb)
    log.info("PCA explained variance: PC1=%.1f%% PC2=%.1f%%",
             pca.explained_variance_ratio_[0] * 100,
             pca.explained_variance_ratio_[1] * 100)

    # Per-city centroids
    centroids: dict[str, list[float]] = {}
    for slug in city_embeddings:
        mask = city_labels == slug
        centroids[slug] = coords_2d[mask].mean(axis=0).tolist()

    # Pairwise centroid distances
    slugs = list(centroids)
    n_c = len(slugs)
    dist_matrix = {}
    for i, si in enumerate(slugs):
        ci = np.array(centroids[si])
        dist_matrix[si] = {}
        for j, sj in enumerate(slugs):
            cj = np.array(centroids[sj])
            dist_matrix[si][sj] = round(float(np.linalg.norm(ci - cj)), 4)

    # Save centroids JSON
    summary = {
        "pca_explained_variance_pc1": round(float(pca.explained_variance_ratio_[0]), 4),
        "pca_explained_variance_pc2": round(float(pca.explained_variance_ratio_[1]), 4),
        "centroids_2d": centroids,
        "pairwise_centroid_distances": dist_matrix,
        "n_cities": n_c,
        "n_total_pixels": len(all_emb),
    }
    json_path = diag_dir / "trunk_centroids.json"
    json_path.write_text(json.dumps(summary, indent=2))
    log.info("Centroids written --> %s", json_path)

    # Print distance matrix
    print()
    print("=" * 60)
    print("  B3 Trunk PCA Summary")
    print("=" * 60)
    print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  Cities embedded: {n_c}")
    print(f"  Total pixels: {len(all_emb)}")
    print()
    print("  Centroid pairwise distances (2D PCA space):")
    header = f"  {'':20s}" + "".join(f"{s[:10]:>11s}" for s in slugs)
    print(header)
    for si in slugs:
        row = f"  {si[:20]:<20s}" + "".join(f"{dist_matrix[si][sj]:>11.3f}" for sj in slugs)
        print(row)
    print("=" * 60)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colors = cm.tab20(np.linspace(0, 1, n_c))
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, slug in enumerate(city_embeddings):
            mask = city_labels == slug
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[colors[i]], label=slug, alpha=0.3, s=4)
            cx, cy = centroids[slug]
            ax.scatter(cx, cy, c=[colors[i]], s=120, marker="*", edgecolors="black", linewidth=0.5)
            ax.annotate(slug, (cx, cy), fontsize=7, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("JEPA Trunk Embeddings -- PCA 2D (training cities)")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        plt.tight_layout()

        png_path = diag_dir / "trunk_pca.png"
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)
        log.info("PCA plot saved --> %s", png_path)
        print(f"\n  Plot: {png_path}")
    except Exception as exc:
        log.warning("Matplotlib plot failed: %s", exc)

    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
