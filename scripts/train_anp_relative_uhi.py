"""
train_anp_relative_uhi.py

Episodic meta-training for the station-conditioned SpatialANP using
**relative UHI** formulation.

Relative UHI definition:
    UHI_relative(pixel) = UHI_absolute(pixel) - UHI_airport_background(city, window)

where:
    UHI_absolute = AAT_window - ERA5_window_degF
    UHI_airport_background = mean UHI of nearest ASOS airport station(s) for
                              that city on its campaign date

This formulation makes ASOS airport stations valid context points at inference:
    station_relative_UHI ≈ 0  (airports define the background)

At inference the ANP predicts: "how much warmer than the airport is each pixel?"
Final prediction = ERA5 + airport_bg_UHI + relative_UHI_prediction

Usage:
    python scripts/train_anp_relative_uhi.py \\
        --trunk-path output/cities/jepa_pretrained_trunk.pt \\
        --cities raleigh_nc chicago_il atlanta_ga seattle_wa albuquerque_nm \\
                 charlotte_nc milwaukee_wi \\
        --output output/anp_relative_uhi.pt \\
        --n-epochs 80 --episodes-per-city 80 --pixels-per-city 4096 \\
        --poisson-lambda 3 --lr 1e-4 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from typing import Optional

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

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

# Local UTC offsets for campaign dates (all summer so DST applies)
_CITY_UTC_OFFSET = {
    "raleigh_nc":     -4,  # EDT
    "chicago_il":     -5,  # CDT
    "atlanta_ga":     -4,  # EDT
    "seattle_wa":     -7,  # PDT
    "albuquerque_nm": -6,  # MDT
    "charlotte_nc":   -4,  # EDT
    "milwaukee_wi":   -5,  # CDT
    "philadelphia_pa": -4, # EDT
}

# CAPA window approximate local times (hours)
_WINDOW_LOCAL_HOURS = {
    "morning": [7, 8, 9],
    "midday":  [11, 12, 13],
    "evening": [19, 20, 21],
}

# IEM ASOS network by state
_IEM_NETWORK = {
    "raleigh_nc":     "NC_ASOS",
    "chicago_il":     "IL_ASOS",
    "atlanta_ga":     "GA_ASOS",
    "seattle_wa":     "WA_ASOS",
    "albuquerque_nm": "NM_ASOS",
    "charlotte_nc":   "NC_ASOS",
    "milwaukee_wi":   "WI_ASOS",
    "philadelphia_pa": "PA_ASOS",
}

# ── Airport background fetching ────────────────────────────────────────────────

def _utc_hours_for_window(city_slug: str, window: str) -> list[int]:
    """Convert local window hours to UTC hours (mod 24)."""
    offset = _CITY_UTC_OFFSET.get(city_slug, -5)
    return [((h - offset) % 24) for h in _WINDOW_LOCAL_HOURS[window]]


def _fetch_asos_hourly(sid: str, date: str, max_retries: int = 3) -> dict[int, float]:
    """Fetch hourly tmpf for a station+date from IEM. Returns {utc_hour: degF}."""
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={sid}&data=tmpf"
        f"&year1={date[:4]}&month1={date[5:7]}&day1={date[8:10]}"
        f"&year2={date[:4]}&month2={date[5:7]}&day2={date[8:10]}"
        f"&tz=UTC&format=onlycomma&latlon=no&missing=null&trace=null&direct=no&report_type=3"
    )
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                text = r.read().decode()
            lines = [l for l in text.strip().split("\n") if l and not l.startswith("#")]
            temps: dict[int, float] = {}
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) < 3 or parts[2] in ("null", "M", ""):
                    continue
                try:
                    hour = int(parts[1][11:13])
                    temps[hour] = float(parts[2])
                except Exception:
                    pass
            return temps
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + 1
                log.debug("  ASOS %s attempt %d failed (%s) — retry in %ds", sid, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                log.warning("  ASOS %s: all retries failed (%s)", sid, e)
    return {}


def _get_nearest_asos(network: str, lat: float, lon: float, n: int = 3) -> list[tuple]:
    """Return [(dist, sid, name, slat, slon), ...] for n nearest stations."""
    url = f"https://mesonet.agron.iastate.edu/geojson/network.php?network={network}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        dists = []
        for f in data.get("features", []):
            slat = f["geometry"]["coordinates"][1]
            slon = f["geometry"]["coordinates"][0]
            d = math.sqrt((slat - lat) ** 2 + (slon - lon) ** 2)
            sid = f["properties"].get("sid", "")
            name = f["properties"].get("sname", "")
            if sid:
                dists.append((d, sid, name, slat, slon))
        dists.sort()
        return dists[:n]
    except Exception as e:
        log.warning("  Network %s fetch failed: %s", network, e)
        return []


def fetch_airport_backgrounds(
    city_slug: str,
    lat: float,
    lon: float,
    campaign_date: str,
    windows: list[str],
    n_stations: int = 3,
) -> dict[str, float]:
    """Fetch airport background UHI per window for a training city.

    Returns {window: airport_bg_uhi_f} where airport_bg_uhi_f is the mean
    UHI anomaly (ASOS observed - ERA5) for the nearest n_stations airports.

    Falls back to 0.0 if no data can be fetched.
    """
    network = _IEM_NETWORK.get(city_slug, "")
    if not network:
        log.warning("  No IEM network for %s — using bg=0", city_slug)
        return {w: 0.0 for w in windows}

    # Fetch ERA5 background at city centre
    era5_bg: dict[str, float] = {}
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={campaign_date}&end_date={campaign_date}"
            f"&hourly=temperature_2m&temperature_unit=fahrenheit&models=era5"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
        hourly = d["hourly"]["temperature_2m"]
        for w in windows:
            hrs = _utc_hours_for_window(city_slug, w)
            vals = [hourly[h] for h in hrs if 0 <= h < len(hourly)]
            era5_bg[w] = sum(vals) / len(vals) if vals else 0.0
    except Exception as e:
        log.warning("  ERA5 fetch failed for %s: %s — using 0 bg", city_slug, e)
        return {w: 0.0 for w in windows}

    # Fetch nearest ASOS stations
    stations = _get_nearest_asos(network, lat, lon, n=n_stations)
    if not stations:
        log.warning("  No ASOS stations found for %s — using bg=0", city_slug)
        return {w: 0.0 for w in windows}

    # Accumulate UHI per window
    window_uhis: dict[str, list[float]] = {w: [] for w in windows}
    for i, (_, sid, name, _, _) in enumerate(stations):
        time.sleep(0.5)  # rate limiting
        temps = _fetch_asos_hourly(sid, campaign_date)
        if not temps:
            continue
        for w in windows:
            hrs = _utc_hours_for_window(city_slug, w)
            vals = [temps[h] for h in hrs if h in temps]
            if vals:
                obs_f = sum(vals) / len(vals)
                uhi = obs_f - era5_bg[w]
                window_uhis[w].append(uhi)
                log.info(
                    "  [%s] %s  %s obs=%.1fF  era5=%.1fF  UHI=%+.1fF",
                    sid, name[:30], w, obs_f, era5_bg[w], uhi,
                )

    result: dict[str, float] = {}
    for w in windows:
        if window_uhis[w]:
            result[w] = float(np.mean(window_uhis[w]))
            log.info("  %s  airport_bg[%s] = %+.2f°F  (n=%d)", city_slug, w, result[w], len(window_uhis[w]))
        else:
            result[w] = 0.0
            log.warning("  %s  no airport data for %s — bg=0", city_slug, w)

    return result


# ── Trunk ──────────────────────────────────────────────────────────────────────

def _poisson_k(lam: float, min_k: int = 1, max_k: int = 20, rng: Optional[np.random.Generator] = None) -> int:
    if rng is None:
        rng = np.random.default_rng()
    return int(np.clip(rng.poisson(lam), min_k, max_k))


class TrunkEncoder(nn.Module):
    def __init__(self, trunk_state: dict, x_mean: torch.Tensor, x_std: torch.Tensor) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
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
        x = x.to(self.x_mean.device)
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        return self.net(torch.nan_to_num(x_norm, nan=0.0))


def load_trunk(trunk_path: str) -> TrunkEncoder:
    ckpt = torch.load(trunk_path, map_location="cpu", weights_only=True)
    return TrunkEncoder(
        ckpt["trunk_state"],
        torch.tensor(ckpt["X_mean"], dtype=torch.float32),
        torch.tensor(ckpt["X_std"],  dtype=torch.float32),
    )


# ── City data loader ───────────────────────────────────────────────────────────

def load_city_tensors(
    city_slug: str,
    output_dir: str,
    trunk: TrunkEncoder,
    windows: list[str],
    pixels_per_city: int,
    seed: int,
    airport_bg: dict[str, float],   # {window: bg_uhi_f}
    feat_cols: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Load city data and compute relative UHI = UHI_pixel - airport_background."""
    data_path = os.path.join(output_dir, "cities", city_slug, "data.geoparquet")
    if not os.path.exists(data_path):
        log.warning("City data not found: %s — skipping", data_path)
        return {}

    log.info("Loading city %s …", city_slug)
    gdf = gpd.read_parquet(data_path)

    # Build shared valid mask (pixels with all window labels)
    per_window_masks: dict[str, np.ndarray] = {}
    any_valid = np.zeros(len(gdf), dtype=bool)
    for w in windows:
        col = _LABEL_COLS[w]
        if col not in gdf.columns:
            continue
        m = gdf[col].notna().values
        per_window_masks[w] = m
        any_valid |= m

    if not per_window_masks:
        log.warning("  No valid windows — skipping")
        return {}

    shared_mask = any_valid.copy()
    for m in per_window_masks.values():
        shared_mask &= m
    if shared_mask.sum() < 100:
        shared_mask = any_valid

    gdf_valid = gdf[shared_mask].reset_index(drop=True)
    n_valid = int(shared_mask.sum())
    if n_valid < 100:
        log.warning("  Too few valid pixels (%d) — skipping", n_valid)
        return {}

    # Subsample
    rng = np.random.default_rng(seed)
    n_sample = min(pixels_per_city, n_valid)
    gdf_s = gdf_valid.iloc[rng.choice(n_valid, size=n_sample, replace=False)].reset_index(drop=True)

    # Trunk embeddings
    for c in feat_cols:
        if c not in gdf_s.columns:
            log.warning("  Missing feature column %s — skipping city", c)
            return {}
    X = torch.tensor(gdf_s[feat_cols].values.astype(np.float32))
    embeddings = trunk.encode(X)

    city_data: dict[str, dict[str, torch.Tensor]] = {}
    for w in windows:
        label_col = _LABEL_COLS[w]
        era5_col  = _ERA5_COLS[w]
        if label_col not in gdf_s.columns or era5_col not in gdf_s.columns:
            continue
        aat    = gdf_s[label_col].values.astype(np.float32)
        era5_f = gdf_s[era5_col].values.astype(np.float32) * 9.0 / 5.0 + 32.0
        uhi_abs = aat - era5_f

        # Subtract airport background → relative UHI
        bg = airport_bg.get(w, 0.0)
        uhi_rel = uhi_abs - bg

        valid = np.isfinite(uhi_rel)
        if valid.sum() < 10:
            continue
        uhi_rel = np.clip(uhi_rel, -25.0, 30.0)

        uhi_t = torch.tensor(uhi_rel).unsqueeze(-1)  # (N, 1)
        city_data[w] = {"embeddings": embeddings, "uhi_relative": uhi_t}
        log.info(
            "  %s %s: n=%d | UHI_rel mean=%+.2f°F std=%.2f°F  (bg=%+.2f°F)",
            city_slug, w, len(uhi_rel),
            float(uhi_t.mean()), float(uhi_t.std()), bg,
        )

    return city_data


# ── Episode loss ───────────────────────────────────────────────────────────────

def anp_episode_loss(
    anp: nn.Module,
    embeddings: torch.Tensor,
    uhi_relative: torch.Tensor,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    N = embeddings.size(0)
    emb = embeddings.to(device)
    uhi = uhi_relative.to(device)

    perm = torch.randperm(N, device=device)
    ctx_idx = perm[:K]
    tgt_idx = perm[K:] if len(perm[K:]) > 0 else perm

    ctx_x = emb[ctx_idx]
    ctx_y = uhi[ctx_idx]
    tgt_x = emb[tgt_idx]
    tgt_y = uhi[tgt_idx]

    mean, std = anp(ctx_x, ctx_y, tgt_x)

    mse = nn.functional.mse_loss(mean, tgt_y)
    if not torch.isfinite(mse):
        return mse

    std_safe = std.clamp(min=1e-3)
    nll = (0.5 * ((tgt_y - mean) ** 2 / std_safe ** 2 + torch.log(std_safe ** 2) + math.log(2 * math.pi))).mean()

    loss = mse + 0.01 * nll
    return loss if torch.isfinite(loss) else mse


# ── Train ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    trunk = load_trunk(args.trunk_path)
    trunk.to(device)

    ckpt = torch.load(args.trunk_path, map_location="cpu", weights_only=True)
    feat_cols = ckpt.get("feat_cols", _FEAT_COLS)

    windows = ["morning", "midday", "evening"]

    # ── Fetch airport backgrounds for each city ────────────────────────────────
    log.info("Fetching airport backgrounds for %d cities …", len(args.cities))
    city_airport_bg: dict[str, dict[str, float]] = {}
    city_centers = _get_city_centers(args.cities, args.output_dir)

    for slug in args.cities:
        lat, lon, date = city_centers.get(slug, (0.0, 0.0, ""))
        if not date:
            log.warning("  Could not get center/date for %s", slug)
            city_airport_bg[slug] = {w: 0.0 for w in windows}
            continue
        log.info("  Fetching airport bg for %s (date=%s lat=%.3f lon=%.3f)", slug, date, lat, lon)
        bg = fetch_airport_backgrounds(slug, lat, lon, date, windows, n_stations=args.n_stations)
        city_airport_bg[slug] = bg
        time.sleep(1.0)  # rate limit between cities

    # ── Load city tensors ──────────────────────────────────────────────────────
    log.info("Pre-loading city data …")
    all_city_data: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for slug in args.cities:
        bg = city_airport_bg.get(slug, {w: 0.0 for w in windows})
        data = load_city_tensors(
            slug, args.output_dir, trunk, windows,
            args.pixels_per_city, args.seed, bg, feat_cols,
        )
        if data:
            all_city_data[slug] = data

    if not all_city_data:
        log.error("No city data loaded — aborting")
        return

    loaded_cities = list(all_city_data.keys())
    log.info("Loaded %d cities: %s", len(loaded_cities), loaded_cities)

    # Save airport backgrounds to output JSON for reference at inference time
    bg_path = args.output.replace(".pt", "_airport_bg.json")
    import json as _json
    with open(bg_path, "w") as f:
        _json.dump(city_airport_bg, f, indent=2)
    log.info("Airport backgrounds saved → %s", bg_path)

    # ── Build ANP ──────────────────────────────────────────────────────────────
    from sparc.inference.anp import SpatialANP
    anp = SpatialANP.from_trunk_config(
        args.trunk_path,
        hidden_dim=args.hidden_dim,
        encoder_dim=args.encoder_dim,
    )
    anp.to(device)
    anp.train()

    optimizer = optim.Adam(anp.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.05)

    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        anp.train()
        epoch_losses: list[float] = []

        for slug in loaded_cities:
            city_data = all_city_data[slug]
            available_windows = list(city_data.keys())
            if not available_windows:
                continue
            for _ in range(args.episodes_per_city):
                w = str(rng.choice(available_windows))
                emb = city_data[w]["embeddings"]
                uhi = city_data[w]["uhi_relative"]
                K = _poisson_k(args.poisson_lambda, min_k=1, max_k=min(20, len(emb) - 1), rng=rng)
                optimizer.zero_grad()
                loss = anp_episode_loss(anp, emb, uhi, K, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(anp.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        log.info("Epoch %3d/%d | loss=%.4f | lr=%.2e", epoch, args.n_epochs, mean_loss, scheduler.get_last_lr()[0])

        if mean_loss < best_loss:
            best_loss = mean_loss
            anp.save_checkpoint(args.output.replace(".pt", "_best.pt"))

    anp.save_checkpoint(args.output)
    log.info("Saved ANP → %s  (best_loss=%.4f)", args.output, best_loss)

    # Quick eval: few-shot on last city
    anp.eval()
    with torch.no_grad():
        slug = loaded_cities[-1]
        for w in ["midday"]:
            if w not in all_city_data[slug]:
                continue
            emb = all_city_data[slug][w]["embeddings"].to(device)
            uhi = all_city_data[slug][w]["uhi_relative"].to(device)
            N = emb.size(0)
            cx_e = torch.zeros(0, 256, device=device)
            cy_e = torch.zeros(0, 1, device=device)
            m_zs, _ = anp(cx_e, cy_e, emb)
            rmse_zs = float(((m_zs - uhi) ** 2).mean().sqrt())
            K = 3
            idx3 = torch.randperm(N, device=device)[:K]
            m_fs, _ = anp(emb[idx3], uhi[idx3], emb)
            rmse_fs = float(((m_fs - uhi) ** 2).mean().sqrt())
            log.info("Final eval %s %s: ZS=%.3f°F  FS(K=3)=%.3f°F", slug, w, rmse_zs, rmse_fs)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_city_centers(slugs: list[str], output_dir: str) -> dict[str, tuple[float, float, str]]:
    """Returns {slug: (lat, lon, campaign_date)} from geoparquet."""
    result = {}
    for slug in slugs:
        path = os.path.join(output_dir, "cities", slug, "data.geoparquet")
        try:
            gdf = gpd.read_parquet(path)
            centroids_wgs84 = gdf.geometry.centroid.to_crs("EPSG:4326")
            lat = float(centroids_wgs84.y.mean())
            lon = float(centroids_wgs84.x.mean())
            date = str(gdf["campaign_date"].iloc[0])[:10]
            result[slug] = (lat, lon, date)
        except Exception as e:
            log.warning("Could not read center for %s: %s", slug, e)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train ANP with relative UHI (airport background subtraction)")
    p.add_argument("--trunk-path",         default="output/cities/jepa_pretrained_trunk.pt")
    p.add_argument("--cities",             nargs="+",
                   default=["raleigh_nc", "chicago_il", "atlanta_ga", "seattle_wa",
                             "albuquerque_nm", "charlotte_nc", "milwaukee_wi"])
    p.add_argument("--output",             default="output/anp_relative_uhi.pt")
    p.add_argument("--output-dir",         default="output")
    p.add_argument("--n-epochs",           type=int, default=80)
    p.add_argument("--episodes-per-city",  type=int, default=80)
    p.add_argument("--pixels-per-city",    type=int, default=4096)
    p.add_argument("--poisson-lambda",     type=float, default=3.0)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--hidden-dim",         type=int, default=128)
    p.add_argument("--encoder-dim",        type=int, default=128)
    p.add_argument("--n-stations",         type=int, default=3,
                   help="Max nearest ASOS stations to fetch per city")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
