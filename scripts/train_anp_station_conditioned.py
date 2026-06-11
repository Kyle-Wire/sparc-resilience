"""
train_anp_station_conditioned.py

Episodic meta-training for the station-conditioned SpatialANP in JEPA trunk
embedding space.

Training protocol:
  For each epoch:
    For each training city:
      For each episode:
        1. Sample K ~ Poisson(lambda=3, min=1) pixel indices as "virtual stations"
        2. context_x  = trunk_embeddings[station_idx]  (K, 256)
        3. context_y  = uhi_anomaly[station_idx]       (K, 1)
        4. Held-out target pixels (all except stations, or random subset)
        5. ANP forward → mean, std
        6. MSE + NLL loss on held-out pixels
  Save ANP checkpoint: output/anp_station_conditioned.pt

UHI anomaly definition (per window):
    uhi_anomaly = aat_{window} - (era5_{window}_t2m * 9/5 + 32)
    [aat is in °F; ERA5 t2m is in °C → convert to °F for subtraction]

This trains the ANP to: given K observed UHI anomalies at embedding positions,
predict UHI anomalies at all other embedding positions. At inference, the context
observations are real ASOS weather station UHI anomalies.

Usage:
    python scripts/train_anp_station_conditioned.py \\
        --trunk-path output/cities/jepa_pretrained_trunk.pt \\
        --cities raleigh_nc chicago_il atlanta_ga seattle_wa albuquerque_nm \\
                 burlington_vt charlotte_nc new_orleans_la \\
        --output output/anp_station_conditioned.pt \\
        --n-epochs 60 --episodes-per-city 80 --pixels-per-city 4096 \\
        --poisson-lambda 3 --lr 1e-4 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
from typing import Optional

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_LABEL_COLS = {
    "morning": "aat_morning",
    "midday": "aat_midday",
    "evening": "aat_night",
}
_ERA5_COLS = {
    "morning": "era5_morning_t2m",
    "midday": "era5_midday_t2m",
    "evening": "era5_evening_t2m",
}
_FEAT_COLS = [
    "pct_impervious", "pct_canopy", "land_cover", "ndvi", "ndbi", "lst",
    "ndvi_s2", "ndbi_s2", "mndwi_s2", "albedo",
    "bldg_height_mean", "bldg_coverage", "svf", "cdc_svi",
    "elevation_m", "slope_deg", "aspect_deg", "distance_water_m",
]


def _poisson_k(lam: float, min_k: int = 1, max_k: int = 20, rng: Optional[np.random.Generator] = None) -> int:
    """Sample K ~ Poisson(lam) clamped to [min_k, max_k]."""
    if rng is None:
        rng = np.random.default_rng()
    k = rng.poisson(lam)
    return int(np.clip(k, min_k, max_k))


class TrunkEncoder(nn.Module):
    """Wraps a loaded trunk state_dict as a frozen inference module."""

    def __init__(self, trunk_state: dict, x_mean: torch.Tensor, x_std: torch.Tensor) -> None:
        super().__init__()
        # 3-layer MLP: Linear(18→256) → GELU → Linear(256→256) → GELU → Linear(256→256)
        self.net = nn.Sequential(
            nn.Linear(18, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.net.load_state_dict(trunk_state)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 18) → (N, 256) normalized trunk embeddings.

        NaN handling matches the main pipeline:
            x_norm = (x - x_mean) / x_std, then NaN → 0 in normalized space.
        This maps any missing feature to 0 (= "population mean" value).
        """
        x = x.to(self.x_mean.device)
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        x_norm = torch.nan_to_num(x_norm, nan=0.0)
        return self.net(x_norm)


def load_trunk(trunk_path: str) -> TrunkEncoder:
    ckpt = torch.load(trunk_path, map_location="cpu", weights_only=True)
    trunk_state = ckpt["trunk_state"]
    x_mean = torch.tensor(ckpt["X_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["X_std"], dtype=torch.float32)
    log.info("Loaded trunk from %s  (feat_cols: %s…)", trunk_path,
             ckpt.get("feat_cols", _FEAT_COLS)[:3])
    return TrunkEncoder(trunk_state, x_mean, x_std)


def load_city_tensors(
    city_slug: str,
    output_dir: str,
    trunk: TrunkEncoder,
    windows: list[str],
    pixels_per_city: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load and precompute tensors for one training city.

    Returns:
        {window: {"embeddings": (N, 256), "uhi_anomaly": (N, 1)}}
    """
    data_path = os.path.join(output_dir, "cities", city_slug, "data.geoparquet")
    if not os.path.exists(data_path):
        log.warning("City data not found: %s — skipping", data_path)
        return {}

    log.info("Loading city %s …", city_slug)
    gdf = gpd.read_parquet(data_path)

    # Filter to pixels with valid labels — per window independently
    # Build a per-window mask; pixels only need to be valid for each individual window
    per_window_masks: dict[str, np.ndarray] = {}
    any_valid = np.zeros(len(gdf), dtype=bool)
    for w in windows:
        label_col = _LABEL_COLS[w]
        if label_col not in gdf.columns:
            log.warning("  Missing label column %s — skipping window %s", label_col, w)
            continue
        m = gdf[label_col].notna().values
        per_window_masks[w] = m
        any_valid |= m

    if not per_window_masks:
        log.warning("  No valid windows — skipping city")
        return {}

    # Use the intersection of all available window masks for a single shared subsample
    # (so trunk embeddings are computed once)
    shared_mask = any_valid
    for m in per_window_masks.values():
        shared_mask = shared_mask & m
    if shared_mask.sum() < 100:
        # Fall back to pixels valid for at least one window
        shared_mask = any_valid

    gdf_valid = gdf[shared_mask].reset_index(drop=True)
    n_valid = int(shared_mask.sum())
    log.info("  %d / %d valid pixels", n_valid, len(gdf))

    if n_valid < 100:
        log.warning("  Too few valid pixels — skipping city")
        return {}

    # Subsample for memory efficiency
    rng = np.random.default_rng(seed)
    n_sample = min(pixels_per_city, n_valid)
    idx = rng.choice(n_valid, size=n_sample, replace=False)
    gdf_sample = gdf_valid.iloc[idx].reset_index(drop=True)

    # Build feature matrix — NaN values are handled via nan_to_num in trunk.encode()
    feat_cols = load_city_tensors._feat_cols if hasattr(load_city_tensors, "_feat_cols") else _FEAT_COLS
    for c in feat_cols:
        if c not in gdf_sample.columns:
            log.warning("  Missing feature column %s — skipping city", c)
            return {}

    X = torch.tensor(gdf_sample[feat_cols].values.astype(np.float32))  # (N, 18)

    # Compute trunk embeddings
    embeddings = trunk.encode(X)  # (N, 256)

    # Compute per-window UHI anomaly
    city_data: dict[str, dict[str, torch.Tensor]] = {}
    for w in windows:
        label_col = _LABEL_COLS[w]
        era5_col = _ERA5_COLS[w]
        if label_col not in gdf_sample.columns or era5_col not in gdf_sample.columns:
            log.warning("  Skipping window %s: missing column(s)", w)
            continue
        aat = gdf_sample[label_col].values.astype(np.float32)  # °F
        era5_c = gdf_sample[era5_col].values.astype(np.float32)  # °C
        era5_f = era5_c * 9.0 / 5.0 + 32.0  # → °F

        uhi_raw = aat - era5_f  # (N,)

        # Check validity mask for this window
        valid_w = np.isfinite(uhi_raw)
        if valid_w.sum() < 10:
            log.warning("  Skipping window %s: too few valid UHI values (%d)", w, valid_w.sum())
            continue

        # Sanity-clip: UHI anomaly outside [-25, +30]°F is physically implausible
        # (Burlington Nov ERA5 mismatch produces 44-58°F — hard-clip these)
        clip_lo, clip_hi = -25.0, 30.0
        n_clipped = int(((uhi_raw < clip_lo) | (uhi_raw > clip_hi)).sum())
        if n_clipped > 0:
            log.warning("  %s: clipping %d / %d UHI outliers to [%.0f, %.0f]°F",
                        w, n_clipped, len(uhi_raw), clip_lo, clip_hi)
        uhi_raw = np.clip(uhi_raw, clip_lo, clip_hi)

        # Normalize per city per window: mean=0, std=1
        # Store stats so inference can denormalize
        uhi_mean = float(np.mean(uhi_raw))
        uhi_std = float(np.std(uhi_raw)) + 1e-6
        uhi_norm = (uhi_raw - uhi_mean) / uhi_std

        uhi = torch.tensor(uhi_norm).unsqueeze(-1)  # (N, 1)

        city_data[w] = {
            "embeddings": embeddings,   # (N, 256) — same across windows
            "uhi_anomaly": uhi,         # (N, 1)
        }
        mean_uhi = float(uhi.mean())
        std_uhi = float(uhi.std())
        log.info("  %s: n=%d | UHI mean=%.2f°F std=%.2f°F",
                 w, n_sample, mean_uhi, std_uhi)

    return city_data


def anp_episode_loss(
    anp: nn.Module,
    embeddings: torch.Tensor,  # (N, 256)
    uhi_anomaly: torch.Tensor,  # (N, 1)
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Single-episode training loss for the ANP.

    Randomly sample K pixels as "stations" (context), held-out = remainder.
    Loss = MSE + 0.1 * NLL on held-out pixels.
    """
    N = embeddings.size(0)
    emb = embeddings.to(device)
    uhi = uhi_anomaly.to(device)

    # Sample K context indices; remainder = target
    perm = torch.randperm(N, device=device)
    ctx_idx = perm[:K]
    tgt_idx = perm[K:]

    if len(tgt_idx) < 1:
        tgt_idx = perm  # edge case: K >= N

    context_x = emb[ctx_idx]    # (K, 256)
    context_y = uhi[ctx_idx]    # (K, 1)
    target_x = emb[tgt_idx]     # (M, 256)
    target_y = uhi[tgt_idx]     # (M, 1)

    mean, std = anp(context_x, context_y, target_x)  # (M, 1), (M, 1)

    # MSE loss
    mse = nn.functional.mse_loss(mean, target_y)

    # Check for NaN before NLL
    if not torch.isfinite(mse):
        return mse  # NaN propagated from forward pass

    # NLL loss (Gaussian): - log p(y | mean, std)
    std_safe = std.clamp(min=1e-3)  # prevent division by near-zero std
    nll = (0.5 * ((target_y - mean) ** 2 / (std_safe ** 2)
                  + torch.log(std_safe ** 2)
                  + math.log(2 * math.pi))).mean()

    loss = mse + 0.01 * nll
    if not torch.isfinite(loss):
        return mse  # fall back to MSE if NLL explodes
    return loss


def train(
    trunk_path: str,
    city_slugs: list[str],
    output_path: str,
    output_dir: str,
    n_epochs: int,
    episodes_per_city: int,
    pixels_per_city: int,
    poisson_lambda: float,
    lr: float,
    seed: int,
    windows: list[str],
    hidden_dim: int,
    encoder_dim: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load trunk ────────────────────────────────────────────────────────────
    trunk = load_trunk(trunk_path)
    trunk.to(device)

    # Patch feat_cols onto load_city_tensors for column alignment
    ckpt = torch.load(trunk_path, map_location="cpu", weights_only=True)
    feat_cols = ckpt.get("feat_cols", _FEAT_COLS)
    # Monkey-patch so load_city_tensors can access it
    load_city_tensors._feat_cols = feat_cols  # type: ignore[attr-defined]

    # ── Pre-load city data ────────────────────────────────────────────────────
    log.info("Pre-loading city data …")
    all_city_data: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for slug in city_slugs:
        data = load_city_tensors(slug, output_dir, trunk, windows, pixels_per_city, seed)
        if data:
            all_city_data[slug] = data

    if not all_city_data:
        log.error("No city data loaded — aborting")
        return

    loaded_cities = list(all_city_data.keys())
    log.info("Loaded %d cities: %s", len(loaded_cities), loaded_cities)

    # ── Build ANP ─────────────────────────────────────────────────────────────
    from sparc.inference.anp import SpatialANP
    anp = SpatialANP.from_trunk_config(
        trunk_path,
        hidden_dim=hidden_dim,
        encoder_dim=encoder_dim,
    )
    anp.to(device)
    anp.train()

    optimizer = optim.Adam(anp.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)

    rng = np.random.default_rng(seed)
    best_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        anp.train()
        epoch_losses: list[float] = []

        for slug in loaded_cities:
            city_data = all_city_data[slug]

            # Get available windows for this city
            available_windows = list(city_data.keys())
            if not available_windows:
                continue

            for _ in range(episodes_per_city):
                # Randomly select a window for this episode from available ones
                w = str(rng.choice(available_windows))
                emb = city_data[w]["embeddings"]
                uhi = city_data[w]["uhi_anomaly"]

                K = _poisson_k(poisson_lambda, min_k=1, max_k=min(20, len(emb) - 1), rng=rng)

                optimizer.zero_grad()
                loss = anp_episode_loss(anp, emb, uhi, K, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(anp.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        lr_cur = float(scheduler.get_last_lr()[0])

        log.info("Epoch %3d/%d | loss=%.4f | lr=%.2e | episodes=%d",
                 epoch, n_epochs, mean_loss, lr_cur, len(epoch_losses))

        # Save best checkpoint
        if mean_loss < best_loss:
            best_loss = mean_loss
            anp.save_checkpoint(output_path + ".best.pt")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            anp.save_checkpoint(output_path + f".epoch{epoch:03d}.pt")

    # ── Save final checkpoint ─────────────────────────────────────────────────
    anp.save_checkpoint(output_path)
    log.info("Saved final ANP → %s  (best loss=%.4f)", output_path, best_loss)

    # ── Quick eval: zero-shot vs few-shot on last city ────────────────────────
    anp.eval()
    with torch.no_grad():
        last_slug = loaded_cities[-1]
        for w in windows[:1]:  # just morning for quick check
            emb = all_city_data[last_slug][w]["embeddings"].to(device)
            uhi = all_city_data[last_slug][w]["uhi_anomaly"].to(device)
            N = emb.size(0)

            # Zero-shot
            cx_empty = torch.zeros(0, 256, device=device)
            cy_empty = torch.zeros(0, 1, device=device)
            mean_zs, _ = anp(cx_empty, cy_empty, emb)
            rmse_zs = float(((mean_zs - uhi) ** 2).mean().sqrt())

            # Few-shot K=3
            K = 3
            idx3 = torch.randperm(N, device=device)[:K]
            mean_fs, _ = anp(emb[idx3], uhi[idx3], emb)
            rmse_fs = float(((mean_fs - uhi) ** 2).mean().sqrt())

            log.info("Final eval %s %s: RMSE zero-shot=%.3f°F  few-shot(K=3)=%.3f°F",
                     last_slug, w, rmse_zs, rmse_fs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train station-conditioned ANP on JEPA trunk embeddings"
    )
    parser.add_argument(
        "--trunk-path", default="output/cities/jepa_pretrained_trunk.pt",
        help="Path to A2-locked JEPA trunk checkpoint"
    )
    parser.add_argument(
        "--cities", nargs="+",
        default=["raleigh_nc", "chicago_il", "atlanta_ga", "seattle_wa",
                 "albuquerque_nm", "burlington_vt", "charlotte_nc", "new_orleans_la"],
        help="Training city slugs (leave out the LOO city)"
    )
    parser.add_argument(
        "--output", default="output/anp_station_conditioned.pt",
        help="Output checkpoint path"
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--episodes-per-city", type=int, default=80,
                        help="Number of episodes per city per epoch")
    parser.add_argument("--pixels-per-city", type=int, default=4096,
                        help="Max pixels to load per city (for memory efficiency)")
    parser.add_argument("--poisson-lambda", type=float, default=3.0,
                        help="Poisson lambda for station count K")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--windows", nargs="+", default=["morning", "midday", "evening"],
        help="Which temporal windows to train on"
    )
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="ANP hidden dimension (default: 128)")
    parser.add_argument("--encoder-dim", type=int, default=128,
                        help="ANP encoder dimension (default: 128)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    train(
        trunk_path=args.trunk_path,
        city_slugs=args.cities,
        output_path=args.output,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        episodes_per_city=args.episodes_per_city,
        pixels_per_city=args.pixels_per_city,
        poisson_lambda=args.poisson_lambda,
        lr=args.lr,
        seed=args.seed,
        windows=args.windows,
        hidden_dim=args.hidden_dim,
        encoder_dim=args.encoder_dim,
    )


if __name__ == "__main__":
    main()
