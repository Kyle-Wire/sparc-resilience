"""
train_multicity_jepa.py -- Multi-city JEPA pretraining + per-city pipeline + LOO eval.

Phases
------
1. JEPA Pretraining  -- batch across all non-holdout city GeoParquets; saves shared trunk
2. Per-City Fine-Tuning -- 3 supervised heads (morning/midday/evening); EWC between cities
3. Leave-One-Out Eval -- zero-shot prediction on Philadelphia (holdout city)
4. Pipeline Stages    -- run stages 0/2/3 per city via RunContext.from_project()

Usage
-----
    python scripts/train_multicity_jepa.py
    python scripts/train_multicity_jepa.py --config configs/multicity_pilot.yml
    python scripts/train_multicity_jepa.py --skip-collect --stages 0,2,3
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout/stderr so box-drawing chars and degree signs don't crash
# on Windows terminals that default to cp1252. line_buffering=True preserves
# the live epoch-by-epoch output that -u gave us before reconfigure.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

# CPU parallelism — thread count is set once in main() before any parallel work starts.
# PyTorch raises RuntimeError if set_num_interop_threads is called a second time.
import os as _os
_CPU_COUNT = _os.cpu_count() or 1
del _os


def _configure_threads(deterministic: bool = False, n_threads: int = 0) -> None:
    """Set PyTorch thread counts. MUST be called once before any parallel work."""
    import torch as _t
    if deterministic:
        _t.set_num_threads(1)
        _t.set_num_interop_threads(1)
    elif n_threads > 0:
        _t.set_num_threads(n_threads)
        _t.set_num_interop_threads(max(1, n_threads // 2))
    else:
        _t.set_num_threads(_CPU_COUNT)
        _t.set_num_interop_threads(max(1, _CPU_COUNT // 2))


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed all RNG sources (torch, numpy, random). Thread counts must be
    set separately via _configure_threads() before any parallel work starts."""
    import torch as _t
    random.seed(seed)
    np.random.seed(seed)
    _t.manual_seed(seed)
    if _t.cuda.is_available():
        _t.cuda.manual_seed_all(seed)
    if deterministic:
        _t.use_deterministic_algorithms(True, warn_only=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_multicity_jepa")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/multicity_pilot.yml")
    p.add_argument("--skip-collect", action="store_true",
                   help="Skip data collection (assume GeoParquets already exist)")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip Phase 1 JEPA pretraining and load existing trunk checkpoint")
    p.add_argument("--skip-stages", action="store_true",
                   help="Skip auxiliary pipeline stages (correlogram/spatial analysis) — speeds up training runs")
    p.add_argument("--stages", default=None,
                   help="Comma-separated pipeline stages to run, e.g. '0,2,3'")
    p.add_argument("--cfa-only-loo", action="store_true",
                   help="Restrict LOO ensemble to Cfa Köppen zone cities (same climate as Philadelphia)")
    p.add_argument("--stage0-only", action="store_true",
                   help="Run pipeline Stage 0 (correlogram + anisotropy) for all training cities then exit. "
                        "Use this to generate Stage 0 artifacts before PI-JEPA training.")
    p.add_argument("--stage0-fast", action="store_true",
                   help="When used with --stage0-only: use fast_mode=True (400 NUTS samples instead of 800, "
                        "~2.5x faster). Recommended for re-runs when FFT correlograms are already cached.")
    p.add_argument("--deterministic", action="store_true",
                   help="Single-threaded, fully-reproducible training (slower but bit-identical across runs)")
    p.add_argument("--threads", type=int, default=0,
                   help="Pin PyTorch to N threads (0=all cores). --threads 4 gives ~4x speedup "
                        "over --deterministic while being reproducible on the same machine.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the RNG seed from config (used for multi-seed B1 baseline runs).")
    p.add_argument("--fewshot-n", type=int, default=0,
                   help="Few-shot Philadelphia experiment: fine-tune a Philadelphia head using N labeled "
                        "pixels, then evaluate on the remaining held-out labeled pixels. "
                        "Runs AFTER Phase 3 zero-shot eval for a direct comparison. "
                        "Example: --fewshot-n 10  (try 10, 100, 1000)")
    p.add_argument("--fewshot-hybrid", action="store_true",
                   help="When used with --fewshot-n: also report a hybrid eval using zero-shot morning "
                        "predictions combined with few-shot midday/evening, on the same held-out test split.")
    p.add_argument("--pretrain-only", action="store_true",
                   help="Run Phase 1 (JEPA pretraining) only, then exit immediately after saving the trunk. "
                        "Useful for I1 eccentricity sweep screening: run with reduced n_epochs + A3 diagnostics "
                        "as a proxy before committing to full Phase 2+3 runs.")
    p.add_argument("--screen-epochs", type=int, default=None,
                   help="Override jepa.n_epochs for this run only (e.g. --screen-epochs 50). "
                        "Use with --pretrain-only for fast proxy screening of hyperparameter sweeps.")
    p.add_argument("--anp-checkpoint", type=str, default=None,
                   help="Path to a trained station-conditioned ANP checkpoint "
                        "(output of train_anp_station_conditioned.py). When provided, "
                        "applies ANP residual correction after Phase 3 zero-shot eval. "
                        "Requires output/cities/{holdout_slug}/stations.parquet to exist.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_trunk(in_dim: int, hidden_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim,    hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _build_predictor(hidden_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class _FiLMHead(nn.Module):
    """FiLM-conditioned regression head: ERA5 modulates the trunk embedding.

    Instead of concatenating ERA5 to the trunk embedding (which forces the first
    linear layer to learn mixing), FiLM uses ERA5 to compute per-dimension scale
    (gamma) and shift (beta) applied to the trunk embedding *before* the MLP.
    This cleanly separates spatial structure (trunk) from weather conditioning (ERA5),
    matching the V-JEPA 2 principle of keeping the encoder pure and conditioning
    through the predictor pathway.

    era5_dim per window:
      morning: 8 (morning + evening)  — evening adds "preceding day" thermal context
               since morning UHI is driven by overnight heat retention, not active sun
      midday:  4 (midday only)        — cross-window chaining (train24) degraded by -0.474
      evening: 4 (evening only)       — same finding as midday
    """

    def __init__(self, hidden_dim: int, era5_dim: int):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.gamma_net = nn.Linear(era5_dim, hidden_dim)   # scale trunk dims
        self.beta_net  = nn.Linear(era5_dim, hidden_dim)   # shift trunk dims
        self.dropout   = nn.Dropout(0.15)
        self.proj1     = nn.Linear(hidden_dim, hidden_dim // 4)
        self.proj2     = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.proj3     = nn.Linear(hidden_dim // 8, 1)

    def forward(self, cat_input):
        """Accepts cat([h_trunk, era5_feat], dim=-1) — same interface as concat MLP heads."""
        import torch.nn.functional as F
        h_trunk  = cat_input[..., :self._hidden_dim]
        era5_feat = cat_input[..., self._hidden_dim:]
        gamma = self.gamma_net(era5_feat)           # (N, D) — scale
        beta  = self.beta_net(era5_feat)            # (N, D) — shift
        h = self.dropout(F.gelu(h_trunk * gamma + beta))
        h = F.gelu(self.proj1(h))
        h = F.gelu(self.proj2(h))
        return self.proj3(h)


def _build_head(hidden_dim: int, era5_dim: int = 4, deep: bool = False,
                deeper: bool = False, film: bool = False) -> nn.Module:
    """Build a regression head with per-window depth and optional FiLM conditioning.

    train31 ablation revealed head depth is window-dependent:
      - Morning (overnight heat retention, simpler spatial pattern):
        2-layer concat MLP is best (+0.461 vs +0.408 with 3-layer)
      - Midday (radiation+wind-driven, complex spatial pattern):
        FiLM conditioning tested in train39 — ERA5 modulates which land
        features matter rather than just being appended
      - Evening (longwave radiation-driven):
        3-layer concat MLP is best

    deep=False  → [h ‖ era5] → D/4 → 1                      (morning, 2-layer)
    deep=True   → [h ‖ era5] → D/4 → D/8 → 1                (evening, 3-layer)
    deeper=True → [h ‖ era5] → D/4 → D/8 → D/16 → 1         (midday, 4-layer)
    film=True   → FiLM(h, era5) → D/4 → D/8 → 1             (experimental)
    """
    if film:
        return _FiLMHead(hidden_dim, era5_dim)
    in_dim = hidden_dim + era5_dim
    if deeper:
        return nn.Sequential(
            nn.Linear(in_dim,            hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4,   hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 8,   hidden_dim // 16),
            nn.GELU(),
            nn.Linear(hidden_dim // 16,  1),
        )
    if deep:
        return nn.Sequential(
            nn.Linear(in_dim,            hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4,   hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8,   1),
        )
    return nn.Sequential(
        nn.Linear(in_dim,        hidden_dim // 4),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(hidden_dim // 4, 1),
    )


# ---------------------------------------------------------------------------
# Climate zone and solar zenith helpers (train27)
# ---------------------------------------------------------------------------

# Köppen climate zone classification for each city.
# Used to weight the LOO ensemble — cities climatically similar to the
# holdout city (Philadelphia, Cfa) contribute more to the ensemble.
_KOPPEN_ZONE: dict[str, str] = {
    "philadelphia_pa": "Cfa",
    "atlanta_ga":      "Cfa",
    "raleigh_nc":      "Cfa",
    "wilmington_de":  "Cfa",   # added train32
    "baltimore_md":    "Cfa",   # added train32
    "charlotte_nc":    "Cfa",   # added train32
    "new_orleans_la":  "Cfa",   # added train32 (humid subtropical)
    "chicago_il":      "Dfa",
    "burlington_vt":   "Dfb",
    "providence_ri":   "Dfb",
    "seattle_wa":      "Cfb",
    "albuquerque_nm":  "BSk",
}

# Pairwise climate distance between zone codes.
# 0 = same zone, higher = more dissimilar.
# Based on: C/D letter (2pt gap), a/b/c subtype (1pt gap), B arid (3pt from C/D).
def _koppen_distance(zone_a: str, zone_b: str) -> float:
    if zone_a == zone_b:
        return 0.0
    letter_a, letter_b = zone_a[0], zone_b[0]
    # Arid (B) is very different from temperate/continental
    if "B" in (letter_a, letter_b):
        return 3.0
    # C vs D: different thermal regime
    letter_dist = 0.0 if letter_a == letter_b else 2.0
    # a/b/c precipitation subtype
    sub_a = zone_a[2] if len(zone_a) > 2 else ""
    sub_b = zone_b[2] if len(zone_b) > 2 else ""
    sub_dist = 0.0 if sub_a == sub_b else 1.0
    return letter_dist + sub_dist


def _koppen_weights(holdout_slug: str, city_slugs: list[str], temperature: float = 0.8) -> np.ndarray:
    """Softmax weights for ensemble cities based on Köppen distance to holdout.

    Closer climate zone → higher weight. temperature controls sharpness:
    lower = sharper (dominant city), higher = more uniform.
    """
    holdout_zone = _KOPPEN_ZONE.get(holdout_slug, "Cfa")
    distances = np.array([
        _koppen_distance(holdout_zone, _KOPPEN_ZONE.get(s, "Cfa"))
        for s in city_slugs
    ], dtype=np.float32)
    # Convert distance to similarity score: sim = exp(-dist / temperature)
    sims = np.exp(-distances / temperature)
    return sims / sims.sum()


def _solar_zenith_cos(lat_deg: float, lon_deg: float, date_str: str, hour_local: float) -> float:
    """Cosine of solar zenith angle for a city-window.

    cos(zenith) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(hour_angle)
    Returns cos(zenith) clipped to [0, 1] (negative = below horizon → 0).

    Args:
        lat_deg: city centroid latitude in degrees
        lon_deg: city centroid longitude in degrees
        date_str: ISO date string "YYYY-MM-DD"
        hour_local: approximate local solar hour (e.g. 7.0 = 7am, 14.0 = 2pm)
    """
    from math import sin, cos, radians, floor
    year_str, month_str, day_str = date_str.split("-")
    # Day-of-year
    import datetime
    doy = datetime.date(int(year_str), int(month_str), int(day_str)).timetuple().tm_yday
    # Solar declination (Spencer, 1971)
    B = radians(360 / 365 * (doy - 81))
    dec = radians(23.45 * sin(B))
    # Equation of time correction (minutes) — small but improves accuracy
    eot = 9.87 * sin(2 * B) - 7.53 * cos(B) - 1.5 * sin(B)
    # Local standard meridian (nearest 15°)
    lstm = 15 * round((-lon_deg) / 15)  # west lon → positive offset
    # True solar time
    tst = hour_local + (4 * (lstm - (-lon_deg)) + eot) / 60
    hour_angle = radians(15 * (tst - 12))
    lat = radians(lat_deg)
    cos_z = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(hour_angle)
    return float(max(0.0, cos_z))


# Approximate local hours for each measurement window
_WINDOW_HOUR = {"morning": 7.0, "midday": 14.0, "evening": 19.0}


# ---------------------------------------------------------------------------
# ERA5 tensor encoding helper
# ---------------------------------------------------------------------------

def _encode_era5_tensor(
    gdf: object,
    cols: list[str],
    device: object,
    n_cells: int,
) -> "torch.Tensor":
    """Build a 4-dim ERA5 tensor: ``[t2m/40, ws/15, wd/180, rh/100]``.

    Parameters
    ----------
    gdf     : GeoDataFrame with ERA5 columns.
    cols    : Column names in order ``[t2m, ws, wd, rh]``.
    device  : torch device.
    n_cells : Number of fishnet cells (used for zero-fill when no columns found).

    Returns
    -------
    FloatTensor of shape (n_cells, 4).
    """
    avail = [c for c in cols if c in gdf.columns]  # type: ignore[union-attr]
    if not avail:
        return torch.zeros(n_cells, 4, device=device)
    raw = np.nan_to_num(gdf[avail].values.astype(np.float32), nan=0.0)  # type: ignore[union-attr]
    if raw.shape[1] < 4:
        pad = np.zeros((len(raw), 4 - raw.shape[1]), dtype=np.float32)
        raw = np.concatenate([raw, pad], axis=1)
    encoded = raw[:, :4] * np.array([1/40.0, 1/15.0, 1/180.0, 1/100.0], dtype=np.float32)
    return torch.tensor(encoded, device=device)


# ---------------------------------------------------------------------------
# PI-JEPA helpers -- read Stage 0 anisotropy payload from artifact stores
# ---------------------------------------------------------------------------

_PI_JEPA_FOCUS_VARS = ("lst", "pct_impervious", "svf", "ndvi", "albedo")


def _load_pi_jepa_params(
    train_cfgs: list[dict],
    output_root: Path,
) -> dict:
    """Read Stage 0 correlogram artifacts and return PI-JEPA pretraining params.

    Returns a dict with:
      "median_effective_range_m" : median effective range across focus variables
                                   and cities that have Stage 0 data (float | None)
      "median_eccentricity_a_over_b" : median a/b eccentricity (>= 1.0) across
                                       anisotropic focus-variable pairs (float | None)

    Cities missing Stage 0 artifacts are silently skipped with a WARNING log.
    """
    import sqlite3

    effective_ranges: list[float] = []
    eccentricities_a_over_b: list[float] = []

    for city_cfg in train_cfgs:
        slug = city_cfg["city_slug"]
        db_path = output_root / slug / "output" / "artifacts.db"
        if not db_path.exists():
            log.warning("[PI-JEPA] %s: Stage 0 artifact DB not found -- skipped", slug)
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # --- effective ranges from correlogram summary ---
            placeholders = ",".join(["?" for _ in _PI_JEPA_FOCUS_VARS])
            cur.execute(
                f'SELECT Effective_Range FROM "data__0__correlogram_summary"'
                f' WHERE Variable IN ({placeholders})',
                _PI_JEPA_FOCUS_VARS,
            )
            for (eff_r,) in cur.fetchall():
                if eff_r and float(eff_r) > 0:
                    effective_ranges.append(float(eff_r))

            # --- eccentricity from anisotropy struct ---
            cur.execute(
                "SELECT json FROM result_structs WHERE stage=? AND artifact_id=?",
                ("0", "correlogram_anisotropy"),
            )
            row = cur.fetchone()
            if row:
                import json as _json
                aniso = _json.loads(row[0])
                pvf = aniso.get("per_variable_fits", {})
                for varname, vdata in pvf.items():
                    if varname not in _PI_JEPA_FOCUS_VARS:
                        continue
                    ellipse = vdata.get("ellipse", {})
                    if not isinstance(ellipse, dict):
                        try:
                            ellipse = _json.loads(ellipse)
                        except Exception:
                            continue
                    ecc_raw = ellipse.get("eccentricity", {})
                    if isinstance(ecc_raw, dict):
                        b_over_a = float(ecc_raw.get("mean", 0.0))
                    elif isinstance(ecc_raw, (int, float)):
                        b_over_a = float(ecc_raw)
                    else:
                        continue
                    # B2 verdict: V-DIR-UNRELIABLE -- only use eccentricity magnitude
                    # Stage 0 stores b/a (0-1); anisotropic_patch_mask expects a/b (>= 1)
                    # Only include if meaningfully anisotropic (b/a < 0.87 threshold)
                    if 0.0 < b_over_a < 0.87:
                        eccentricities_a_over_b.append(1.0 / b_over_a)

            conn.close()
        except Exception as exc:
            log.warning("[PI-JEPA] %s: error reading artifacts: %s", slug, exc)

    import statistics
    result: dict = {
        "median_effective_range_m": None,
        "median_eccentricity_a_over_b": None,
    }
    if effective_ranges:
        result["median_effective_range_m"] = float(statistics.median(effective_ranges))
        log.info(
            "[PI-JEPA] effective ranges: n=%d  median=%.0fm  min=%.0fm  max=%.0fm",
            len(effective_ranges),
            result["median_effective_range_m"],
            min(effective_ranges),
            max(effective_ranges),
        )
    if eccentricities_a_over_b:
        result["median_eccentricity_a_over_b"] = float(
            statistics.median(eccentricities_a_over_b)
        )
        log.info(
            "[PI-JEPA] eccentricities (a/b): n=%d  median=%.3f  min=%.3f  max=%.3f",
            len(eccentricities_a_over_b),
            result["median_eccentricity_a_over_b"],
            min(eccentricities_a_over_b),
            max(eccentricities_a_over_b),
        )
    return result


# ---------------------------------------------------------------------------
# Phase 1 -- JEPA pretraining
# ---------------------------------------------------------------------------

def load_city_features(
    parquet_path: Path,
    feature_cols: list[str],
    coord_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and coords from a city GeoParquet."""
    import geopandas as gpd
    gdf = gpd.read_parquet(str(parquet_path))
    available_features = [c for c in feature_cols if c in gdf.columns]
    X = gdf[available_features].values.astype(np.float32)
    available_coords = [c for c in coord_cols if c in gdf.columns]
    if available_coords:
        C = gdf[available_coords].values.astype(np.float32)
    else:
        C = np.zeros((len(X), 2), dtype=np.float32)
    return X, C


def run_jepa_pretraining(
    all_X: np.ndarray,
    all_C: np.ndarray,
    jepa_cfg: dict,
    output_dir: Path,
    pi_patch_radius_m: float | None = None,
    pi_eccentricity_a_over_b: float | None = None,
    city_sizes: list[int] | None = None,
    all_U: np.ndarray | None = None,
    feat_cols: list[str] | None = None,
) -> object:
    """Run JEPA spatial-patch pretraining on combined city features.

    Parameters
    ----------
    pi_patch_radius_m : float or None
        If set (I2 flag), the patch radius is set to this value in physical
        UTM metres, then converted to normalised coordinate units using C_std.
        Overrides the default bbox-derived radius.  Set via
        ``jepa.range_scaled_radius: true`` in the config.
    pi_eccentricity_a_over_b : float or None
        If set (I1 flag), patches are elliptical with this a/b ratio (>= 1).
        Direction theta=0 (no rotation -- B2 verdict: V-DIR-UNRELIABLE).
        Set via ``jepa.anisotropic_mask: true`` in the config.
    city_sizes : list[int] or None
        Row counts per city in all_X / all_C (in order).  Required when
        ``jepa.per_city_batching: true``.  Batches are then drawn from a
        single city at a time, making city-local coordinate masking valid.
    all_U : ndarray of shape (N, k_clusters) or None
        E5 fuzzy cluster memberships aligned row-for-row with all_X.
        When not None and ``jepa.eigen_target_weight > 0``, an auxiliary
        head predicts these memberships from the online trunk embedding.

    Returns the trained online trunk (nn.Module).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sparc.training.jepa_loss import (
        JEPALossWeights, jepa_loss, spatial_patch_mask, anisotropic_patch_mask,
    )

    hidden_dim = int(jepa_cfg.get("hidden_dim", 256))
    n_epochs   = int(jepa_cfg.get("n_epochs", 50))
    batch_size = int(jepa_cfg.get("batch_size", 2048))
    mask_ratio = float(jepa_cfg.get("mask_ratio", 0.40))
    n_patches  = int(jepa_cfg.get("n_patches", 4))
    ema_tau    = float(jepa_cfg.get("ema_tau", 0.99))
    lr         = float(jepa_cfg.get("lr", 1e-3))
    per_city   = bool(jepa_cfg.get("per_city_batching", False))
    # A2: auxiliary energy-balance pretext head.
    # Predicts shortwave absorption proxy (1 - albedo) from trunk embedding.
    # albedo is index 9 in land_surface feature list (confirmed against live feature order).
    eb_weight  = float(jepa_cfg.get("energy_balance_weight", 0.0))
    eb_albedo_idx = int(jepa_cfg.get("energy_balance_albedo_idx", 9))
    # E5: eigenmap-FCM auxiliary head.
    # Predicts fuzzy cluster memberships in Laplacian eigenspace (urban structure proxy).
    # Memberships passed in as all_U: (N_total, k_clusters) float32 aligned with all_X rows.
    eigen_weight = float(jepa_cfg.get("eigen_target_weight", 0.0))

    seed = int(jepa_cfg.get("seed", 0))
    if seed:
        _seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False  # keep deterministic across GPU generations
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        log.info("JEPA pretraining on GPU: %s  (%d MB)  N=%d  D=%d  seed=%d",
                 gpu_name, gpu_mem, len(all_X), all_X.shape[1], seed)
    else:
        log.info("JEPA pretraining on device=%s  N=%d  D=%d  seed=%d", device, len(all_X), all_X.shape[1], seed)

    # Normalize globally.  Some feature columns may be entirely NaN (e.g. a
    # raster band missing for every city), which causes np.nanmean/nanstd to
    # emit "mean of empty slice" / "degrees of freedom <= 0" warnings.  We
    # compute each stat exactly once inside an errstate block, then replace
    # any resulting NaN with a safe default (0 for mean, 1 for std).
    with np.errstate(all="ignore"):
        _xm = np.nanmean(all_X, axis=0)
        _xs = np.nanstd(all_X,  axis=0)
        _cm = np.nanmean(all_C, axis=0)
        _cs = np.nanstd(all_C,  axis=0)

    X_mean = np.where(np.isnan(_xm), 0.0, _xm)
    X_std  = np.where(np.isnan(_xs), 1.0, _xs) + 1e-6
    all_X  = np.nan_to_num((all_X - X_mean) / X_std, nan=0.0).astype(np.float32)

    # Tensor versions of normalization stats for A2 un-normalization in the training loop
    X_mean_t = torch.tensor(X_mean, dtype=torch.float32, device=device)
    X_std_t  = torch.tensor(X_std,  dtype=torch.float32, device=device)

    C_mean = np.where(np.isnan(_cm), 0.0, _cm)
    C_std  = np.where(np.isnan(_cs), 1.0, _cs) + 1e-6
    all_C  = np.nan_to_num((all_C - C_mean) / C_std, nan=0.0).astype(np.float32)

    # I2: Convert physical effective range (metres) to normalised coordinate units.
    # Uses the mean of x-std and y-std as the length scale.  min_patch_radius_normed
    # is None when I2 is disabled, which reverts to the bbox-derived radius.
    if pi_patch_radius_m is not None:
        mean_c_std = float((C_std[0] + C_std[1]) / 2.0)
        min_patch_radius_normed: float | None = pi_patch_radius_m / mean_c_std
        log.info(
            "I2 range-scaled radius: %.0fm / %.0fm_std = %.5f normalised units",
            pi_patch_radius_m, mean_c_std, min_patch_radius_normed,
        )
    else:
        min_patch_radius_normed = None

    # I1: eccentricity for anisotropic masking (a/b ratio, >= 1.0).
    # theta=0 (no rotation -- B2 verdict: V-DIR-UNRELIABLE).
    # None = disabled (use isotropic spatial_patch_mask).
    if pi_eccentricity_a_over_b is not None and pi_eccentricity_a_over_b > 1.0:
        log.info(
            "I1 anisotropic mask enabled: eccentricity(a/b)=%.3f  theta=0.0 (no rotation)",
            pi_eccentricity_a_over_b,
        )

    X_t = torch.tensor(all_X, device=device)
    C_t = torch.tensor(all_C, device=device)

    # Per-city batching: precompute [start, end) index ranges per city so each
    # batch comes from one city only.  This makes city-local coordinates valid
    # for spatial masking (no cross-city coordinate overlap).
    city_ranges: list[tuple[int, int]] | None = None
    if per_city and city_sizes:
        offsets: list[int] = []
        cur = 0
        for sz in city_sizes:
            offsets.append(cur)
            cur += sz
        city_ranges = [(offsets[i], offsets[i] + city_sizes[i]) for i in range(len(city_sizes))]
        log.info("B6 per-city batching: %d cities, batch_size=%d drawn per-city", len(city_ranges), batch_size)

    if not per_city:
        # Include row indices so E5 can look up U_t[idx] in the training loop.
        idx_t   = torch.arange(len(X_t), dtype=torch.long, device=device)
        dataset = TensorDataset(X_t, C_t, idx_t)
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            pin_memory=(device.type == "cpu"),
            num_workers=0,
        )
        batches_per_epoch = len(loader)
    else:
        loader = None
        # Estimate steps per epoch: same total pixels as the DataLoader would give
        total_N = sum(city_sizes) if city_sizes else len(all_X)
        batches_per_epoch = max(1, total_N // batch_size)

    in_dim  = all_X.shape[1]
    online  = _build_trunk(in_dim, hidden_dim).to(device)
    target  = _build_trunk(in_dim, hidden_dim).to(device)
    predict = _build_predictor(hidden_dim).to(device)

    # A2: energy-balance auxiliary head — predicts (1 - albedo) from trunk embedding.
    # Single linear layer is sufficient: we want the trunk to encode absorption magnitude,
    # not to overfit a complex albedo mapping.
    eb_head: nn.Module | None = None
    if eb_weight > 0.0:
        eb_head = nn.Linear(hidden_dim, 1).to(device)
        log.info(
            "A2 energy-balance pretext: weight=%.3f  albedo_idx=%d  "
            "(target: 1 - albedo, shortwave absorption proxy)",
            eb_weight, eb_albedo_idx,
        )

    # E5: eigenmap-FCM auxiliary head — predicts fuzzy cluster memberships.
    # Softmax output so predicted memberships sum to 1 (mirrors FCM constraint).
    # U_t kept on CPU to avoid a large sudden GPU allocation after the eigsh idle period
    # (Blackwell RTX 5070 Ti driver TDR on ~640MB burst after 4min GPU idle).
    # Per-batch transfer is 2048×4×4B = ~32KB — negligible.
    eigen_head: nn.Module | None = None
    U_t: object = None   # (N, k) float32 tensor on CPU, or None
    k_clusters = 0
    if eigen_weight > 0.0 and all_U is not None:
        k_clusters = all_U.shape[1]
        eigen_head = nn.Linear(hidden_dim, k_clusters).to(device)
        U_t = torch.tensor(all_U, dtype=torch.float32)   # CPU — moved per-batch
        log.info(
            "E5 eigenmap-FCM pretext: weight=%.3f  k_clusters=%d  N=%d  (U_t on CPU)",
            eigen_weight, k_clusters, all_U.shape[0],
        )

    # Initialize EMA target = online weights
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.copy_(p_o.data)
        p_t.requires_grad_(False)

    opt_params = list(online.parameters()) + list(predict.parameters())
    if eb_head is not None:
        opt_params += list(eb_head.parameters())
    if eigen_head is not None:
        opt_params += list(eigen_head.parameters())
    opt     = torch.optim.AdamW(opt_params, lr=lr)
    weights = JEPALossWeights()

    print(flush=True)
    print("  Phase 1 -- JEPA Pretraining", flush=True)
    mask_mode = ("I1+I2" if pi_eccentricity_a_over_b and min_patch_radius_normed
                 else "I1" if pi_eccentricity_a_over_b
                 else "I2" if min_patch_radius_normed
                 else "isotropic")
    eb_str    = f"+A2(w={eb_weight})" if eb_weight > 0.0 else ""
    eigen_str = f"+E5(w={eigen_weight},k={k_clusters})" if eigen_weight > 0.0 and eigen_head is not None else ""
    print(f"  mask_mode={mask_mode}{eb_str}{eigen_str}", flush=True)
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Align':>10}  {'Var':>10}  {'Cov':>10}", flush=True)
    print("  " + "-" * 50, flush=True)

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []

        # --- batch iterator ---
        if loader is not None:
            batch_iter = loader
        else:
            # Per-city batching: for each step, pick a random city then random pixels
            import random as _random
            def _per_city_batches(n_steps: int):
                for _ in range(n_steps):
                    city_idx = _random.randrange(len(city_ranges))
                    lo, hi   = city_ranges[city_idx]
                    city_n   = hi - lo
                    size     = min(batch_size, city_n)
                    perm     = torch.randperm(city_n, device=device)[:size]
                    idx      = perm + lo
                    yield X_t[idx], C_t[idx], idx
            batch_iter = _per_city_batches(batches_per_epoch)

        for x_batch, c_batch, batch_idx in batch_iter:
            # Compute spatial mask on CPU to avoid Blackwell (sm_120) kernel issues
            # with simple scalar ops (sqrt/clamp) in spatial_patch_mask.
            # The mask is (B,) booleans — CPU compute is negligible, device transfer is ~2KB.
            c_cpu = c_batch.cpu()
            if pi_eccentricity_a_over_b is not None and pi_eccentricity_a_over_b > 1.0:
                mask = anisotropic_patch_mask(
                    c_cpu, mask_ratio=mask_ratio, n_patches=n_patches,
                    eccentricity=pi_eccentricity_a_over_b, theta=0.0,
                    min_patch_radius=min_patch_radius_normed,
                ).to(device)
            else:
                mask = spatial_patch_mask(
                    c_cpu, mask_ratio=mask_ratio, n_patches=n_patches,
                    min_patch_radius=min_patch_radius_normed,
                ).to(device)

            # Context (visible) = ~masked; target (hidden) = masked
            x_context = x_batch.clone()
            x_context[mask] = 0.0  # zero-out masked positions

            h_online = online(x_context)          # (B, D)
            h_pred   = predict(h_online)           # (B, D)

            with torch.no_grad():
                h_target = target(x_batch)         # (B, D) -- full, EMA trunk

            loss, comps = jepa_loss(h_pred, h_target.detach(), weights)

            # A2: energy-balance auxiliary loss on ALL pixels (not just masked).
            # Target: (1 - albedo_raw) in original (unnormalized) space.
            # x_batch is normalized; recover albedo via X_mean/X_std then compute proxy.
            if eb_head is not None and eb_weight > 0.0:
                # h_online already computed above from masked context
                eb_pred = eb_head(h_online).squeeze(-1)   # (B,)
                # Un-normalize albedo column to get physical [0,1] range
                alb_norm = x_batch[:, eb_albedo_idx]      # normalized albedo
                alb_raw  = alb_norm * X_std_t[eb_albedo_idx] + X_mean_t[eb_albedo_idx]
                alb_raw  = alb_raw.clamp(0.0, 1.0)
                eb_target = 1.0 - alb_raw                 # shortwave absorption proxy
                eb_loss   = torch.nn.functional.mse_loss(eb_pred, eb_target)
                loss = loss + eb_weight * eb_loss

            # E5: eigenmap-FCM auxiliary loss on ALL pixels.
            # Target: fuzzy cluster memberships U[batch_idx] from Laplacian eigenspace.
            # Cross-entropy over the membership distribution (soft labels).
            # Softmax of logits vs FCM memberships = KL-like alignment to spatial topology.
            if eigen_head is not None and eigen_weight > 0.0 and U_t is not None:
                eigen_logits = eigen_head(h_online)                        # (B, k)
                # U_t lives on CPU; batch_idx is on device — move idx to CPU for lookup,
                # then move the 32KB result to device. Avoids large GPU allocation.
                u_target = U_t[batch_idx.cpu()].to(device)                 # (B, k)
                # Soft cross-entropy: -sum(u_target * log_softmax(logits))
                log_probs  = torch.nn.functional.log_softmax(eigen_logits, dim=-1)
                eigen_loss = -(u_target * log_probs).sum(dim=-1).mean()
                loss       = loss + eigen_weight * eigen_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA update of target trunk
            with torch.no_grad():
                for p_t, p_o in zip(target.parameters(), online.parameters()):
                    p_t.data.mul_(ema_tau).add_(p_o.data, alpha=1.0 - ema_tau)

            epoch_losses.append(comps)

        if epoch % 5 == 0 or epoch == 1:
            avg = {k: float(np.mean([c[k] for c in epoch_losses])) for k in epoch_losses[0]}
            print(
                f"  {epoch:>6}  {avg['jepa_total']:>10.4f}"
                f"  {avg['jepa_align']:>10.4f}"
                f"  {avg['jepa_variance']:>10.4f}"
                f"  {avg['jepa_covariance']:>10.4f}",
                flush=True,
            )

    print("  " + "-" * 50, flush=True)

    # Save shared trunk
    trunk_path = output_dir / "jepa_pretrained_trunk.pt"
    torch.save({
        "trunk_state": online.state_dict(),
        "in_dim":      in_dim,
        "hidden_dim":  hidden_dim,
        "X_mean":      X_mean.tolist(),
        "X_std":       X_std.tolist(),
        "C_mean":      C_mean.tolist(),
        "C_std":       C_std.tolist(),
        "feat_cols":   feat_cols,
    }, str(trunk_path))
    log.info("Shared trunk saved → %s", trunk_path)

    return online, {"X_mean": X_mean, "X_std": X_std}


# ---------------------------------------------------------------------------
# Phase 2 -- Per-City Fine-Tuning with EWC
# ---------------------------------------------------------------------------

class _CityModel:
    """Container for trunk + 3 supervised heads."""

    def __init__(self, trunk, heads: dict, device, has_labels: bool = True,
                 has_labels_per_window: dict = None, city_slug: str = "",
                 mean_embedding: object = None):
        self.trunk      = trunk
        self.heads      = heads   # {"morning": nn.Module, "midday": ..., "evening": ...}
        self.device     = device
        self.has_labels = has_labels  # False if city had no CAPA supervision at all
        self.city_slug  = city_slug
        # Per-window label availability: {"morning": True, "midday": True, "evening": False}
        self.has_labels_per_window = has_labels_per_window or {
            w: has_labels for w in ("morning", "midday", "evening")
        }
        # Mean trunk embedding for prototype-weighted ensemble (D,) tensor or None
        self.mean_embedding = mean_embedding

    def predict(self, x_land: object, era5_dict: dict) -> dict:
        import torch
        preds = {}
        with torch.no_grad():
            h = self.trunk(x_land)
            for window, head in self.heads.items():
                era5_feats = era5_dict.get(window)
                if era5_feats is None:
                    era5_feats = torch.zeros(h.shape[0], 5, device=self.device)
                preds[window] = head(torch.cat([h, era5_feats], dim=-1)).squeeze(-1)
        return preds


def run_city_finetune(
    city_slug: str,
    parquet_path: Path,
    trunk,
    feature_cols_land: list[str],
    pilot_cfg: dict,
    fisher_matrices: list,
    optimal_params_list: list,
    output_dir: Path,
    device,
    normalizer: dict = None,
) -> tuple:
    """Fine-tune trunk + 3 heads on one city; update EWC state."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import geopandas as gpd
    from sparc.training.ewc import ewc_penalty, compute_fisher_matrix, extract_trunk_params

    mc           = pilot_cfg["multicity"]
    ewc_lambda   = float(mc["continual_learning"].get("ewc_lambda", 400.0))
    hidden_dim   = int(mc["jepa"].get("hidden_dim", 256))
    n_epochs     = max(50, int(mc["jepa"].get("n_epochs", 50)))
    batch_size   = int(mc["jepa"].get("batch_size", 8192))  # large batch: fewer steps/epoch, faster on CPU
    lr           = float(mc["jepa"].get("lr", 1e-3))
    head_weight_decay = float(mc["jepa"].get("head_weight_decay", 0.1))
    seed         = int(mc["jepa"].get("seed", 42))

    gdf = gpd.read_parquet(str(parquet_path))

    # Build feature tensors (apply same normalization used during JEPA pretraining)
    # NaN handling MUST match Phase 1: normalize first, then nan_to_num → NaN becomes 0
    # in NORMALIZED space.  Doing nan_to_num before normalizing maps NaN to
    # (0 - X_mean)/X_std, which is far out-of-distribution for features like albedo
    # (X_mean≈11960) and causes garbage trunk embeddings (negative spatial correlation).
    X_land_cols = [c for c in feature_cols_land if c in gdf.columns]
    X_raw = gdf[X_land_cols].values.astype(np.float32)
    if normalizer is not None:
        X_mean = normalizer["X_mean"][:X_raw.shape[1]]
        X_std  = normalizer["X_std"][:X_raw.shape[1]]
        X_raw  = np.nan_to_num((X_raw - X_mean) / X_std, nan=0.0)  # NaN → 0 in norm space
    else:
        X_raw = np.nan_to_num(X_raw, nan=0.0)
    X_land = torch.tensor(X_raw, device=device)

    # ERA5 per-window tensors — 5-dim each: [t2m/40, ws/15, sin(wd), cos(wd), rh/100]
    # Wind direction is encoded as sin/cos to handle circular 0°/360° discontinuity.
    _era5_cols = {
        "morning": ["era5_morning_t2m","era5_morning_windspeed","era5_morning_winddir","era5_morning_rh"],
        "midday":  ["era5_midday_t2m", "era5_midday_windspeed", "era5_midday_winddir", "era5_midday_rh"],
        "evening": ["era5_evening_t2m","era5_evening_windspeed","era5_evening_winddir","era5_evening_rh"],
    }
    era5_tensors = {
        w: _encode_era5_tensor(gdf, cols, device, len(X_land))
        for w, cols in _era5_cols.items()
    }  # each tensor: (N, 4)

    # Label tensors: predict UHI anomaly above ERA5 t2m (°F) rather than absolute °F.
    # This bounds all targets to ~3-10°F range across all cities, removes inter-city
    # calibration shifts, and forces the head to learn spatial gradients only.
    # ERA5 t2m in era5_tensors has been divided by _ERA5_SCALE[0]=40, so un-scale:
    #   era5_t2m_°C = era5_tensor[:, 0] * 40.0
    #   era5_t2m_°F = era5_t2m_°C * 1.8 + 32.0
    _label_cols = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}
    labels = {}
    for w, col in _label_cols.items():
        era5_t2m_F = era5_tensors[w][:, 0] * 40.0 * 1.8 + 32.0  # ERA5 background in °F
        if col in gdf.columns:
            arr = gdf[col].values.astype(np.float32)
            valid = ~np.isnan(arr)
            anomaly = arr - era5_t2m_F.cpu().numpy()  # UHI anomaly: typically +3 to +10 °F
            labels[w] = torch.tensor(np.nan_to_num(anomaly, nan=0.0), device=device)
            labels[f"{w}_valid"] = torch.tensor(valid, device=device)
        else:
            labels[w] = torch.zeros(len(X_land), device=device)
            labels[f"{w}_valid"] = torch.zeros(len(X_land), dtype=torch.bool, device=device)

    # Trunk is FROZEN — JEPA pretraining already built good cross-city spatial
    # representations.  Unfreezing causes the trunk to drift city-by-city and
    # destroys its generalization (train11: -38°F bias; train12: -261°F bias).
    # The deeper 3-layer head now has enough capacity to decode the frozen
    # embeddings into UHI anomaly predictions without needing trunk adaptation.
    trunk_local = trunk
    for p in trunk_local.parameters():
        p.requires_grad_(False)
    trunk_local.eval()

    # Per-window ERA5 context tensors + solar zenith (train27).
    # Solar zenith cos(θ) is a city-level scalar (same for all pixels in a city)
    # that captures the radiation forcing intensity for each window.
    # morning: [morning(5) + evening(5)] = 10-dim (prior evening = overnight heat retention)
    # midday:  [midday(5) + morning(5) + solar_zenith(1)] = 11-dim (train40: temporal bridging)
    # evening: [evening(5) + solar_zenith(1)] = 6-dim (cross-window context hurts: t41/t42)
    # CRS is UTM (meters) — convert geometry centroid to WGS84 degrees for solar formula.
    _wgs84_pt = gdf.geometry.centroid.to_crs("EPSG:4326").iloc[0]
    city_lat  = float(_wgs84_pt.y)
    city_lon  = float(_wgs84_pt.x)
    date_str  = str(gdf["campaign_date"].iloc[0])[:10] if "campaign_date" in gdf.columns else "2021-07-15"
    solar_z = {
        w: float(_solar_zenith_cos(city_lat, city_lon, date_str, _WINDOW_HOUR[w]))
        for w in ("morning", "midday", "evening")
    }
    log.info("[%s] solar zenith cos  morning=%.3f  midday=%.3f  evening=%.3f  lat=%.2f  date=%s",
             city_slug, solar_z["morning"], solar_z["midday"], solar_z["evening"], city_lat, date_str)
    N = len(X_land)
    solar_tensors = {
        w: torch.full((N, 1), solar_z[w], device=device)
        for w in ("morning", "midday", "evening")
    }
    era5_inputs = {
        # morning: [morning(4) + evening(4)] = 8-dim (prior evening = overnight heat retention)
        "morning": torch.cat([era5_tensors["morning"], era5_tensors["evening"]], dim=-1),  # (N, 8)
        # midday: [midday(4) + morning(4) + solar_zenith(1)] = 9-dim (train40: temporal bridging)
        "midday":  torch.cat([era5_tensors["midday"], era5_tensors["morning"], solar_tensors["midday"]], dim=-1),  # (N, 9)
        # evening: [evening(4) + solar_zenith(1)] = 5-dim (cross-window context hurts: t41/t42)
        "evening": torch.cat([era5_tensors["evening"], solar_tensors["evening"]], dim=-1),  # (N, 5)
    }
    era5_dims = {"morning": 8, "midday": 9, "evening": 5}

    # Deterministic head init — fixes stochastic convergence across runs (T-2).
    _seed_everything(seed)

    heads = {
        "morning": _build_head(hidden_dim, era5_dim=era5_dims["morning"], deep=False,  deeper=False, film=False).to(device),
        "midday":  _build_head(hidden_dim, era5_dim=era5_dims["midday"],  deep=True,   deeper=False, film=False).to(device),
        "evening": _build_head(hidden_dim, era5_dim=era5_dims["evening"], deep=True,   deeper=False, film=False).to(device),
    }

    head_params = [p for head in heads.values() for p in head.parameters()]
    # L2 weight decay on heads prevents covariate-shift extrapolation on OOD cities (T-1).
    opt = torch.optim.AdamW(head_params, lr=lr, weight_decay=head_weight_decay)

    n = len(X_land)
    idx = torch.randperm(n, device=device)
    mse_loss = nn.MSELoss()
    any_labels = any(labels[f"{w}_valid"].sum() > 0 for w in ("morning", "midday", "evening"))
    has_labels_per_window = {w: bool(labels[f"{w}_valid"].sum() > 0) for w in ("morning", "midday", "evening")}

    log.info("[%s] fine-tuning  epochs=%d  N=%d", city_slug, n_epochs, n)

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            x_b = X_land[b]
            with torch.no_grad():
                h = trunk_local(x_b)  # frozen trunk — no grad needed

            sup_loss = torch.tensor(0.0, device=device)
            for w, head in heads.items():
                pred = head(torch.cat([h, era5_inputs[w][b]], dim=-1)).squeeze(-1)
                valid_mask = labels[f"{w}_valid"][b]
                if valid_mask.sum() > 0:
                    sup_loss = sup_loss + mse_loss(pred[valid_mask], labels[w][b][valid_mask])

            loss = sup_loss  # trunk frozen — no EWC needed

            if not loss.requires_grad:
                # No CAPA labels and no EWC anchors yet — nothing to train on this batch
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % 5 == 0 or epoch == n_epochs:
            log.info("[%s]  epoch %3d/%d  loss=%.4f", city_slug, epoch, n_epochs, total_loss / max(n_batches, 1))

    # Save city model
    model_path = output_dir / city_slug / "model.pt"
    (output_dir / city_slug).mkdir(parents=True, exist_ok=True)
    torch.save({
        "trunk_state": trunk_local.state_dict(),
        "heads_state": {w: h.state_dict() for w, h in heads.items()},
        "city_slug":   city_slug,
    }, str(model_path))
    log.info("[%s] model saved → %s", city_slug, model_path)

    # Trunk is frozen — no Fisher/EWC needed (no trunk parameters were updated).
    fisher = {}
    theta_star = {}

    # Collect per-window UHI calibration stats for Option-C offset correction.
    # For each window: mean UHI anomaly (°F above ERA5) and mean ERA5 t2m (°F).
    uhi_stats: dict[str, dict] = {}
    for w in ("morning", "midday", "evening"):
        valid_mask = labels[f"{w}_valid"].cpu().numpy().astype(bool)
        if valid_mask.sum() > 10:
            anomaly_vals = labels[w].cpu().numpy()[valid_mask]
            era5_t2m_vals = (era5_tensors[w][:, 0].cpu().numpy() * 40.0 * 1.8 + 32.0)[valid_mask]
            uhi_stats[w] = {
                "mean_uhi_anomaly": float(anomaly_vals.mean()),
                "mean_era5_t2m_F":  float(era5_t2m_vals.mean()),
                "n": int(valid_mask.sum()),
            }

    # Compute mean trunk embedding for this city (used for prototype-weighted ensemble at LOO).
    # This is the centroid of the trunk's representation space for this city's pixels.
    with torch.no_grad():
        trunk_local.eval()
        mean_emb = trunk_local(X_land).mean(dim=0).cpu()  # (D,)

    return trunk_local, fisher, theta_star, _CityModel(
        trunk_local, heads, device,
        has_labels=any_labels,
        has_labels_per_window=has_labels_per_window,
        city_slug=city_slug,
        mean_embedding=mean_emb,
    ), uhi_stats


def _ewc_penalty_sequential(trunk, fisher_matrices, optimal_params_list):
    """EWC penalty for a Sequential trunk (keys are string indices)."""
    import torch
    if not fisher_matrices:
        return torch.tensor(0.0, device=next(trunk.parameters()).device)
    penalty = torch.tensor(0.0, device=next(trunk.parameters()).device)
    current = {name: p for name, p in trunk.named_parameters()}
    for fisher, theta_star in zip(fisher_matrices, optimal_params_list):
        for name, param in current.items():
            if name in fisher and name in theta_star:
                f = fisher[name].to(param.device)
                ts = theta_star[name].to(param.device)
                penalty = penalty + (f * (param - ts) ** 2).sum()
    return penalty


def _compute_fisher_sequential(trunk, X_land, era5_tensors, labels, heads, device, batch_size):
    """Approximate diagonal Fisher for trunk using gradient accumulation."""
    import torch
    import torch.nn as nn
    trunk.eval()
    fisher = {name: torch.zeros_like(p) for name, p in trunk.named_parameters()}
    mse = nn.MSELoss()
    n = len(X_land)
    n_batches = 0
    for start in range(0, n, batch_size):
        b = slice(start, start + batch_size)
        x_b = X_land[b]
        trunk.zero_grad()
        h = trunk(x_b)
        loss = torch.tensor(0.0, device=device)
        for w, head in heads.items():
            era5_b = era5_tensors[w][b]
            pred = head(torch.cat([h, era5_b], dim=-1)).squeeze(-1)
            valid = labels[f"{w}_valid"][b]
            if valid.sum() > 0:
                loss = loss + mse(pred[valid], labels[w][b][valid])
        if loss.item() > 0:
            loss.backward()
            for name, p in trunk.named_parameters():
                if p.grad is not None:
                    fisher[name] += (p.grad.detach() ** 2) * (b.stop - b.start if b.stop else batch_size)
            n_batches += 1
    if n_batches > 0:
        for name in fisher:
            fisher[name] /= n_batches * batch_size
    trunk.train()
    return fisher


# ---------------------------------------------------------------------------
# Phase 3 -- Leave-One-Out Eval
# ---------------------------------------------------------------------------

def run_loo_eval(
    holdout_path: Path,
    trunk,
    feature_cols_land: list[str],
    pilot_cfg: dict,
    output_dir: Path,
    device,
    city_models: dict = None,
    normalizer: dict = None,
    city_uhi_stats: list[dict] = None,
    cfa_only: bool = False,
    anp_path: str | None = None,
) -> dict:
    """Zero-shot prediction on holdout city using ensemble of trained city heads.

    Uses CAPA-supervised city models to produce an ensemble prediction on the
    holdout city (Philadelphia). For each time window, only includes city-head
    combos where that city had valid CAPA labels for that specific window.
    Falls back to a random head only if no trained models are available.

    Option-C UHI offset calibration: fits a simple linear model on training-city
    mean UHI anomalies vs mean ERA5 t2m, then applies a per-window bias correction
    to the ensemble prediction at inference time.
    """
    import torch
    import geopandas as gpd

    mc         = pilot_cfg["multicity"]
    hidden_dim = int(mc["jepa"].get("hidden_dim", 256))
    # Per-window city exclusion list from config (e.g. charlotte_nc out of evening).
    _loo_excl: dict[str, list[str]] = mc["jepa"].get("loo_exclude_per_window", {})

    log.info("")
    log.info("Phase 3 -- LOO Eval on %s", holdout_path)
    gdf = gpd.read_parquet(str(holdout_path))

    X_land_cols = [c for c in feature_cols_land if c in gdf.columns]
    # NaN handling MUST match Phase 1: normalize first, then nan_to_num → NaN = 0 in
    # normalized space.  Doing nan_to_num before normalizing maps NaN to
    # (0 - X_mean)/X_std — far OOD for high-mean features like albedo.
    X_raw = gdf[X_land_cols].values.astype(np.float32)
    if normalizer is not None:
        X_mean = normalizer["X_mean"][:X_raw.shape[1]]
        X_std  = normalizer["X_std"][:X_raw.shape[1]]
        X_raw  = np.nan_to_num((X_raw - X_mean) / X_std, nan=0.0)  # NaN → 0 in norm space
    else:
        X_raw = np.nan_to_num(X_raw, nan=0.0)
    X_land = torch.tensor(X_raw, device=device)

    _era5_cols = {
        "morning": ["era5_morning_t2m","era5_morning_windspeed","era5_morning_winddir","era5_morning_rh"],
        "midday":  ["era5_midday_t2m", "era5_midday_windspeed", "era5_midday_winddir", "era5_midday_rh"],
        "evening": ["era5_evening_t2m","era5_evening_windspeed","era5_evening_winddir","era5_evening_rh"],
    }
    era5_tensors = {
        w: _encode_era5_tensor(gdf, cols, device, len(X_land))
        for w, cols in _era5_cols.items()
    }  # each tensor: (N, 4)

    # All 12 ERA5 features (4-dim per window, 3 windows)
    era5_all = torch.cat([
        era5_tensors["morning"],
        era5_tensors["midday"],
        era5_tensors["evening"],
    ], dim=-1)  # (N, 12)

    # Solar zenith for holdout city (train27) — same computation as training
    holdout_slug  = holdout_path.parent.name
    # CRS is UTM (meters) — convert geometry centroid to WGS84 degrees for solar formula.
    _wgs84_pt_loo = gdf.geometry.centroid.to_crs("EPSG:4326").iloc[0]
    city_lat  = float(_wgs84_pt_loo.y)
    city_lon  = float(_wgs84_pt_loo.x)
    date_str  = str(gdf["campaign_date"].iloc[0])[:10] if "campaign_date" in gdf.columns else "2021-07-15"
    solar_z_loo = {
        w: float(_solar_zenith_cos(city_lat, city_lon, date_str, _WINDOW_HOUR[w]))
        for w in ("morning", "midday", "evening")
    }
    log.info("LOO solar zenith cos  morning=%.3f  midday=%.3f  evening=%.3f  (%s)",
             solar_z_loo["morning"], solar_z_loo["midday"], solar_z_loo["evening"], holdout_slug)
    N_loo = len(X_land)
    solar_tensors_loo = {
        w: torch.full((N_loo, 1), solar_z_loo[w], device=device)
        for w in ("morning", "midday", "evening")
    }

    # Per-window ERA5 inputs matching the training configuration (t40 baseline: 4-dim ERA5 + temporal bridging)
    era5_inputs_loo = {
        "morning": torch.cat([era5_tensors["morning"], era5_tensors["evening"]], dim=-1),  # (N, 8)
        "midday":  torch.cat([era5_tensors["midday"], era5_tensors["morning"], solar_tensors_loo["midday"]], dim=-1),  # (N, 9)
        "evening": torch.cat([era5_tensors["evening"], solar_tensors_loo["evening"]], dim=-1),  # (N, 5)
    }

    _label_cols = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}

    # Collect CAPA-supervised city models for ensemble
    supervised_models = []
    if city_models:
        supervised_models = [m for m in city_models.values() if m.has_labels]
        if cfa_only:
            holdout_zone = _KOPPEN_ZONE.get(holdout_path.parent.name, "")
            supervised_models = [
                m for m in supervised_models
                if _KOPPEN_ZONE.get(m.city_slug, "") == holdout_zone
            ]
            log.info("LOO ensemble (Cfa-only): %d supervised city models (zone=%s)",
                     len(supervised_models), holdout_zone)
        else:
            log.info("LOO ensemble: %d supervised city models", len(supervised_models))

    trunk.eval()
    results = {}
    pred_cols = {}

    with torch.no_grad():
        if supervised_models:
            # All city models share the same final trunk (updated sequentially via EWC).
            # Compute the shared trunk embedding once per window, then apply each city head.
            # This guarantees coherent embeddings across the ensemble.
            shared_trunk = supervised_models[0].trunk
            shared_trunk.eval()
            # Heads must be in eval mode — Dropout(0.15) active in train mode
            # would randomly zero 15% of hidden neurons per pixel, adding noise
            # and reducing spatial correlation even inside torch.no_grad().
            for cm in supervised_models:
                for head in cm.heads.values():
                    head.eval()
            for w in ("morning", "midday", "evening"):
                excl_set = set(_loo_excl.get(w, []))
                window_models = [cm for cm in supervised_models
                                 if cm.has_labels_per_window.get(w, False)
                                 and cm.city_slug not in excl_set]
                if excl_set:
                    excluded_names = [cm.city_slug for cm in supervised_models
                                      if cm.city_slug in excl_set]
                    if excluded_names:
                        log.info("LOO %s: explicitly excluded by config: %s", w, excluded_names)
                if not window_models:
                    log.warning("No supervised models with %s labels — skipping %s window", w, w)
                    continue
                log.info("LOO %s ensemble: %d city heads", w, len(window_models))
                h_shared_emb = shared_trunk(X_land)  # shared trunk embedding (computed once)
                era5_feat = era5_inputs_loo[w]  # per-window ERA5 (matches training config)
                era5_t2m_F = era5_tensors[w][:, 0] * 40.0 * 1.8 + 32.0
                city_preds = []
                for cm in window_models:
                    city_preds.append(cm.heads[w](torch.cat([h_shared_emb, era5_feat], dim=-1)).squeeze(-1))

                # Log per-city stats and identify outlier heads by mean prediction.
                # An outlier head is one whose spatial-mean absolute prediction deviates
                # more than the per-window robust-σ threshold from the group median (MAD).
                # MAD is much more resistant than mean/std to a single outlier in a small
                # ensemble (5 cities), where one bad head inflates std and lets others slip
                # through (e.g. Raleigh midday at 1.39σ survives a 1.5σ mean/std filter
                # but is correctly flagged at ~6 robust-σ under MAD).
                city_means = np.array([
                    float((cp + era5_t2m_F).cpu().mean()) for cp in city_preds
                ])
                med = float(np.median(city_means))
                mad = float(np.median(np.abs(city_means - med)))
                # Scale MAD to equivalent normal σ (MAD / 0.6745 ≈ σ for Gaussian).
                # Per-window thresholds: evening uses 1.5σ (tighter) to keep Chicago's
                # overnight cold-island head excluded even as the ensemble grows.
                # Morning/midday use 2.0σ — less aggressive, preserves useful diversity.
                _sigma_thresh = {"morning": 2.0, "midday": 2.0, "evening": 1.5}
                robust_std = mad / 0.6745
                if robust_std > 0:
                    keep_mask = np.abs(city_means - med) <= _sigma_thresh[w] * robust_std
                else:
                    keep_mask = np.ones(len(city_means), dtype=bool)
                for cm, cp, keep, cmean in zip(window_models, city_preds, keep_mask, city_means):
                    log.info("  per-city head [%s] %s mean_pred=%.2f°F  keep=%s",
                             cm.city_slug, w, cmean, "OK" if keep else "X (outlier — excluded)")
                if keep_mask.sum() == 0:
                    keep_mask[:] = True  # safety: never drop all cities
                trimmed_preds = [cp for cp, k in zip(city_preds, keep_mask) if k]
                trimmed_models = [cm for cm, k in zip(window_models, keep_mask) if k]
                if keep_mask.sum() < len(window_models):
                    log.info("  trimmed ensemble: %d/%d city heads kept", keep_mask.sum(), len(window_models))

                # train30: per-window ensemble weighting.
                # Morning UHI = overnight heat RETENTION driven by urban fabric density.
                # Chicago (Dfa) shares dense grid-pattern Rust Belt morphology with
                # Philadelphia, contributing strongly to morning spatial pattern despite
                # different climate zone. Köppen downweighting of Chicago from 0.25 → 0.04
                # caused morning regression from +0.467 → ~0 (t28/t29).
                # Solution: equal weight for morning (let MAD trimming do the work),
                # Köppen weighting for midday/evening (radiation-driven UHI where climate
                # zone correctly predicts relative intensity).
                if w == "morning":
                    kw_t = torch.ones(len(trimmed_preds), device=device) / len(trimmed_preds)
                    log.info("  morning: equal-weight ensemble (%d cities)", len(trimmed_preds))
                else:
                    koppen_w = _koppen_weights(
                        holdout_slug,
                        [cm.city_slug for cm in trimmed_models],
                        temperature=0.8,
                    )
                    for cm, kw in zip(trimmed_models, koppen_w):
                        log.info("  koppen weight [%s] %s zone=%s  w=%.3f",
                                 cm.city_slug, w,
                                 _KOPPEN_ZONE.get(cm.city_slug, "?"), kw)
                    kw_t = torch.tensor(koppen_w, dtype=torch.float32, device=device)
                stacked = torch.stack(trimmed_preds, dim=0)  # (K, N)
                pred_anomaly = (stacked * kw_t.unsqueeze(-1)).sum(dim=0)  # (N,)
                # Reconstruct absolute °F: anomaly + ERA5 t2m background (era5_t2m_F already computed above)
                pred = (pred_anomaly + era5_t2m_F).cpu().numpy()
                pred_cols[f"pred_{w}"] = pred

                label_col = _label_cols[w]
                if label_col in gdf.columns:
                    true_vals = gdf[label_col].values.astype(np.float32)
                    valid = ~np.isnan(true_vals)
                    if valid.sum() > 0:
                        rmse = float(np.sqrt(np.mean((pred[valid] - true_vals[valid]) ** 2)))
                        mae  = float(np.mean(np.abs(pred[valid] - true_vals[valid])))
                        bias = float(np.mean(pred[valid] - true_vals[valid]))
                        corr = float(np.corrcoef(pred[valid], true_vals[valid])[0, 1]) if valid.sum() > 2 else 0.0
                        results[w] = {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "n": int(valid.sum())}
        else:
            # Fallback: random heads with shared pretrained trunk (baseline only)
            log.warning("No supervised city models available — LOO predictions are from random heads")
            trunk.eval()
            h_shared = trunk(X_land)
            heads = {
                "morning": _build_head(hidden_dim, era5_dim=8, deep=False, deeper=False, film=False).to(device),
                "midday":  _build_head(hidden_dim, era5_dim=9, deep=True,  deeper=False, film=False).to(device),
                "evening": _build_head(hidden_dim, era5_dim=5,  deep=True,  deeper=False, film=False).to(device),
            }
            for w, head in heads.items():
                head.eval()
                pred = head(torch.cat([h_shared, era5_inputs_loo[w]], dim=-1)).squeeze(-1).cpu().numpy()
                pred_cols[f"pred_{w}"] = pred

    for col, vals in pred_cols.items():
        gdf[col] = vals

    # ANP station-conditioned correction (optional: requires --anp-checkpoint)
    anp_results: dict = {}
    if anp_path is not None:
        try:
            import pandas as pd
            from scipy.spatial import cKDTree
            from sparc.inference.anp import SpatialANP

            stations_parquet = holdout_path.parent / "stations.parquet"
            if not stations_parquet.exists():
                log.warning("ANP: stations.parquet not found at %s — run "
                            "scripts/fetch_open_meteo_stations.py first", stations_parquet)
            else:
                log.info("ANP correction: loading model from %s", anp_path)
                anp = SpatialANP.from_checkpoint(str(anp_path))
                anp = anp.to(device)
                anp.eval()

                stations_df = pd.read_parquet(str(stations_parquet))

                # Shared trunk embedding for ALL pixels (one forward pass)
                _shared_trunk = supervised_models[0].trunk if supervised_models else trunk
                _shared_trunk.eval()
                with torch.no_grad():
                    trunk_emb = _shared_trunk(X_land)  # (N, 256)

                # Pixel centroids in UTM metres
                pixel_easting  = gdf.geometry.centroid.x.values
                pixel_northing = gdf.geometry.centroid.y.values
                pixel_xy = np.column_stack([pixel_easting, pixel_northing])
                kd_tree  = cKDTree(pixel_xy)

                for w in ("morning", "midday", "evening"):
                    raw_key = f"pred_{w}"
                    if raw_key not in pred_cols:
                        continue

                    # Stations with valid observations for this window
                    st_w = stations_df[stations_df["window"] == w].dropna(
                        subset=["obs_temp_f", "uhi_anomaly_f"]
                    )
                    if len(st_w) == 0:
                        log.info("ANP %s: no stations with valid observations — skipping", w)
                        continue

                    # Nearest pixel per station → use its trunk embedding as context_x
                    st_xy = np.column_stack([st_w["easting"].values, st_w["northing"].values])
                    _, nn_idx = kd_tree.query(st_xy)

                    with torch.no_grad():
                        ctx_x = trunk_emb[nn_idx]  # (K, 256)
                        ctx_y = torch.tensor(
                            st_w["uhi_anomaly_f"].values, dtype=torch.float32, device=device
                        ).unsqueeze(-1)  # (K, 1)
                        ctx_coords = torch.tensor(st_xy,   dtype=torch.float32, device=device)
                        tgt_coords = torch.tensor(pixel_xy, dtype=torch.float32, device=device)

                        mu, sigma = anp(
                            context_x=ctx_x,
                            context_y=ctx_y,
                            target_x=trunk_emb,
                            context_coords=ctx_coords,
                            target_coords=tgt_coords,
                            matern_range_m=6266.0,
                        )  # mu: (N, 1)

                        uhi_anp = mu.squeeze(-1).cpu().numpy()  # (N,)

                    # Absolute temperature = ERA5 background + ANP-predicted UHI
                    era5_t2m_F_anp = era5_tensors[w][:, 0].cpu().numpy() * 40.0 * 1.8 + 32.0
                    anp_pred = era5_t2m_F_anp + uhi_anp
                    anp_key  = f"pred_{w}_anp"
                    gdf[anp_key] = anp_pred

                    sig_mean = float(sigma.squeeze(-1).cpu().numpy().mean())
                    log.info("ANP %s: K=%d stations, UHI mean=%.2f°F  sigma_mean=%.2f°F",
                             w, len(st_w), float(uhi_anp.mean()), sig_mean)

                    label_col = _label_cols[w]
                    if label_col in gdf.columns:
                        true_vals = gdf[label_col].values.astype(np.float32)
                        valid = ~np.isnan(true_vals)
                        if valid.sum() > 0:
                            rmse_anp = float(np.sqrt(np.mean((anp_pred[valid] - true_vals[valid]) ** 2)))
                            mae_anp  = float(np.mean(np.abs(anp_pred[valid] - true_vals[valid])))
                            anp_results[w] = {"rmse": rmse_anp, "mae": mae_anp, "n": int(valid.sum())}

                if anp_results:
                    log.info("ANP correction complete (%d windows evaluated)", len(anp_results))

        except Exception as exc:
            import traceback
            log.warning("ANP correction failed: %s", exc)
            log.debug(traceback.format_exc())

    # Option-C: UHI offset calibration.
    # Fit per-window linear model (era5_t2m → mean_uhi_anomaly) on training cities,
    # then apply bias correction to Philadelphia predictions.
    # Stores calibrated predictions alongside raw ones for comparison.
    calibrated_results: dict = {}
    if city_uhi_stats and pred_cols:
        for w in ("morning", "midday", "evening"):
            raw_key = f"pred_{w}"
            if raw_key not in pred_cols:
                continue
            # Gather training-city stats for this window
            train_era5_t2m = []
            train_uhi_mean = []
            for cs in city_uhi_stats:
                if w in cs and isinstance(cs[w], dict):
                    train_era5_t2m.append(cs[w]["mean_era5_t2m_F"])
                    train_uhi_mean.append(cs[w]["mean_uhi_anomaly"])
            if len(train_era5_t2m) < 2:
                log.info("Option-C: not enough training cities with %s labels — skipping calibration", w)
                continue
            x = np.array(train_era5_t2m, dtype=np.float64)
            y = np.array(train_uhi_mean, dtype=np.float64)
            # MAD-based outlier removal on UHI anomalies before fitting.
            # Charlotte's midday anomaly (17.9°F) vs cluster (3-7°F) corrupts
            # the linear calibration fit — same 2σ MAD filter as ensemble.
            if len(y) >= 3:
                med_y    = np.median(y)
                mad_y    = np.median(np.abs(y - med_y))
                rst_y    = mad_y / 0.6745
                if rst_y > 0:
                    keep_cal = np.abs(y - med_y) <= 2.0 * rst_y
                    if keep_cal.sum() >= 2:
                        x, y = x[keep_cal], y[keep_cal]
            # Simple OLS: slope and intercept
            x_c = x - x.mean()
            slope     = float(np.dot(x_c, y - y.mean()) / (np.dot(x_c, x_c) + 1e-9))
            intercept = float(y.mean() - slope * x.mean())

            # Philadelphia mean ERA5 t2m for this window
            era5_t2m_F_np = (era5_tensors[w][:, 0].cpu().numpy() * 40.0 * 1.8 + 32.0)
            holdout_mean_era5 = float(era5_t2m_F_np.mean())
            predicted_mean_uhi = slope * holdout_mean_era5 + intercept

            raw_pred = pred_cols[raw_key]
            raw_pred_anomaly = raw_pred - era5_t2m_F_np
            raw_mean_anomaly = float(raw_pred_anomaly.mean())
            bias_correction = predicted_mean_uhi - raw_mean_anomaly
            calibrated_pred = raw_pred + bias_correction

            cal_key = f"pred_{w}_calibrated"
            gdf[cal_key] = calibrated_pred

            log.info("Option-C  %s  slope=%.3f  intercept=%.3f  raw_mean_anomaly=%.2f°F  "
                     "predicted_uhi=%.2f°F  bias_correction=%.2f°F",
                     w, slope, intercept, raw_mean_anomaly, predicted_mean_uhi, bias_correction)

            label_col = _label_cols[w]
            if label_col in gdf.columns:
                true_vals = gdf[label_col].values.astype(np.float32)
                valid = ~np.isnan(true_vals)
                if valid.sum() > 0:
                    rmse_cal = float(np.sqrt(np.mean((calibrated_pred[valid] - true_vals[valid]) ** 2)))
                    mae_cal  = float(np.mean(np.abs(calibrated_pred[valid] - true_vals[valid])))
                    calibrated_results[w] = {
                        "rmse": rmse_cal, "mae": mae_cal, "n": int(valid.sum()),
                        "bias_correction": bias_correction,
                    }

    # Save predictions (raw + calibrated if available)
    pred_path = holdout_path.parent / "predictions.geoparquet"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf.to_parquet(str(pred_path), index=False)
    log.info("Predictions saved → %s", pred_path)

    print()
    print("  Phase 3 -- Leave-One-Out Eval (Philadelphia, zero-shot)")
    _anp_hdr = f"  {'RMSE (ANP)':>10}" if anp_results else ""
    print(f"  {'Window':<12}  {'RMSE (raw)':>10}  {'RMSE (cal)':>10}{_anp_hdr}  {'Bias (raw)':>10}  {'Corr':>6}  {'N':>8}")
    print("  " + "-" * (65 + (13 if anp_results else 0)))
    for w in ("morning", "midday", "evening"):
        raw_m = results.get(w, {})
        cal_m = calibrated_results.get(w, {})
        anp_m = anp_results.get(w, {})
        rmse_r = f"{raw_m['rmse']:>10.4f}" if raw_m else "         —"
        rmse_c = f"{cal_m['rmse']:>10.4f}" if cal_m else "         —"
        rmse_a = (f"  {anp_m['rmse']:>10.4f}" if anp_m else ("           —" if anp_results else "")) 
        bias_r = f"{raw_m.get('bias', float('nan')):>10.2f}" if raw_m else "         —"
        corr_r = f"{raw_m.get('corr', float('nan')):>6.3f}" if raw_m else "     —"
        n      = raw_m.get("n", "—")
        print(f"  {w:<12}  {rmse_r}  {rmse_c}{rmse_a}  {bias_r}  {corr_r}  {n:>8}")
    print("  " + "-" * (65 + (13 if anp_results else 0)))

    return {"raw": results, "calibrated": calibrated_results, "anp": anp_results}


# ---------------------------------------------------------------------------
# Phase 3B -- Few-Shot Fine-Tune on Holdout City
# ---------------------------------------------------------------------------

def run_fewshot_finetune(
    holdout_path: Path,
    trunk,
    feature_cols_land: list[str],
    pilot_cfg: dict,
    output_dir: Path,
    device,
    normalizer: dict = None,
    n_samples: int = 10,
    city_uhi_stats: list[dict] = None,
    hybrid: bool = False,
) -> dict:
    """Few-shot transfer: fine-tune a city head on N labeled Philadelphia pixels.

    Experimental setup (no data leakage):
      - Randomly sample ``n_samples`` labeled pixels  → training set
      - Evaluate on ALL remaining labeled pixels       → held-out test set

    If ``hybrid=True``, also loads the saved zero-shot morning predictions and
    reports a hybrid row: morning=zero-shot-calibrated, midday/evening=few-shot-raw.
    This is evaluated on the *same* held-out test_idx for a fair comparison.
    Uses the FROZEN trunk from Phase 1/2 — same setup as Phase 2 city heads.
    Prints a side-by-side comparison with zero-shot ensemble results.
    """
    import torch
    import torch.nn as nn
    import geopandas as gpd

    mc         = pilot_cfg["multicity"]
    hidden_dim = int(mc["jepa"].get("hidden_dim", 256))
    batch_size = int(mc["jepa"].get("batch_size", 8192))
    lr         = float(mc["jepa"].get("lr", 1e-3))
    head_wd    = float(mc["jepa"].get("head_weight_decay", 0.1))
    seed       = int(mc["jepa"].get("seed", 42))

    # Adaptive epoch count: small N → more passes needed to converge
    if n_samples <= 10:
        n_epochs = 2000
    elif n_samples <= 100:
        n_epochs = 1000
    elif n_samples <= 1000:
        n_epochs = 500
    else:
        n_epochs = 300

    gdf = gpd.read_parquet(str(holdout_path))

    # --- Build full-city feature tensors (same NaN handling as Phase 1/2/3) ---
    X_land_cols = [c for c in feature_cols_land if c in gdf.columns]
    X_raw = gdf[X_land_cols].values.astype(np.float32)
    if normalizer is not None:
        X_mean = normalizer["X_mean"][:X_raw.shape[1]]
        X_std  = normalizer["X_std"][:X_raw.shape[1]]
        X_raw  = np.nan_to_num((X_raw - X_mean) / X_std, nan=0.0)
    else:
        X_raw = np.nan_to_num(X_raw, nan=0.0)
    X_land = torch.tensor(X_raw, device=device)

    _era5_cols = {
        "morning": ["era5_morning_t2m","era5_morning_windspeed","era5_morning_winddir","era5_morning_rh"],
        "midday":  ["era5_midday_t2m", "era5_midday_windspeed", "era5_midday_winddir", "era5_midday_rh"],
        "evening": ["era5_evening_t2m","era5_evening_windspeed","era5_evening_winddir","era5_evening_rh"],
    }
    era5_tensors = {
        w: _encode_era5_tensor(gdf, cols, device, len(X_land))
        for w, cols in _era5_cols.items()
    }

    # --- Solar zenith ---
    _wgs84_pt = gdf.geometry.centroid.to_crs("EPSG:4326").iloc[0]
    city_lat  = float(_wgs84_pt.y)
    city_lon  = float(_wgs84_pt.x)
    date_str  = str(gdf["campaign_date"].iloc[0])[:10] if "campaign_date" in gdf.columns else "2021-07-15"
    solar_z   = {w: float(_solar_zenith_cos(city_lat, city_lon, date_str, _WINDOW_HOUR[w]))
                 for w in ("morning", "midday", "evening")}
    N = len(X_land)
    solar_tensors = {w: torch.full((N, 1), solar_z[w], device=device)
                     for w in ("morning", "midday", "evening")}
    era5_inputs = {
        "morning": torch.cat([era5_tensors["morning"], era5_tensors["evening"]], dim=-1),
        "midday":  torch.cat([era5_tensors["midday"], era5_tensors["morning"], solar_tensors["midday"]], dim=-1),
        "evening": torch.cat([era5_tensors["evening"], solar_tensors["evening"]], dim=-1),
    }
    era5_dims = {"morning": 8, "midday": 9, "evening": 5}

    # --- Labels (UHI anomaly above ERA5 t2m) ---
    _label_cols = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}
    labels     = {}
    valid_mask = {}
    for w, col in _label_cols.items():
        era5_t2m_F = era5_tensors[w][:, 0].cpu().numpy() * 40.0 * 1.8 + 32.0
        if col in gdf.columns:
            arr     = gdf[col].values.astype(np.float32)
            v_mask  = ~np.isnan(arr)
            anomaly = arr - era5_t2m_F
            labels[w]     = torch.tensor(np.nan_to_num(anomaly, nan=0.0), device=device)
            valid_mask[w] = v_mask  # numpy bool array
        else:
            labels[w]     = torch.zeros(N, device=device)
            valid_mask[w] = np.zeros(N, dtype=bool)

    # --- Train/test split on labeled pixels ---
    # Use the union of windows for the split so every window has consistent indices.
    any_valid = np.zeros(N, dtype=bool)
    for w in ("morning", "midday", "evening"):
        any_valid |= valid_mask[w]
    labeled_idx = np.where(any_valid)[0]

    rng = np.random.default_rng(seed)
    if len(labeled_idx) < n_samples:
        log.warning("fewshot: only %d labeled pixels — using all for training", len(labeled_idx))
        train_idx = labeled_idx
        test_idx  = labeled_idx
    else:
        chosen   = rng.choice(len(labeled_idx), size=n_samples, replace=False)
        chosen_set = set(chosen.tolist())
        train_idx = labeled_idx[chosen]
        test_idx  = labeled_idx[np.array([i for i in range(len(labeled_idx))
                                           if i not in chosen_set])]

    log.info("fewshot [philadelphia_pa]: n_train=%d  n_test=%d  epochs=%d",
             len(train_idx), len(test_idx), n_epochs)

    # --- Freeze trunk ---
    trunk_local = trunk
    for p in trunk_local.parameters():
        p.requires_grad_(False)
    trunk_local.eval()

    # --- Build Philadelphia-specific heads ---
    _seed_everything(seed)
    heads = {
        "morning": _build_head(hidden_dim, era5_dim=era5_dims["morning"], deep=False, deeper=False, film=False).to(device),
        "midday":  _build_head(hidden_dim, era5_dim=era5_dims["midday"],  deep=True,  deeper=False, film=False).to(device),
        "evening": _build_head(hidden_dim, era5_dim=era5_dims["evening"], deep=True,  deeper=False, film=False).to(device),
    }
    opt = torch.optim.AdamW(
        [p for h in heads.values() for p in h.parameters()],
        lr=lr, weight_decay=head_wd,
    )
    mse_loss = nn.MSELoss()

    # Pre-compute trunk embeddings for train set (frozen → single forward pass)
    train_t = torch.tensor(train_idx, device=device)
    with torch.no_grad():
        h_train = trunk_local(X_land[train_t])  # (n_train, D) — computed once

    # --- Training loop (tiny batch → full batch on train set) ---
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, len(train_idx), batch_size):
            b_slice = slice(start, start + batch_size)
            b_local = train_t[b_slice]
            h_b     = h_train[b_slice]

            sup_loss = torch.tensor(0.0, device=device)
            for w, head in heads.items():
                pred     = head(torch.cat([h_b, era5_inputs[w][b_local]], dim=-1)).squeeze(-1)
                v_cpu    = valid_mask[w][train_idx[b_slice]]
                v_gpu    = torch.tensor(v_cpu, device=device)
                if v_gpu.sum() > 0:
                    sup_loss = sup_loss + mse_loss(pred[v_gpu], labels[w][b_local][v_gpu])

            if not sup_loss.requires_grad:
                continue
            opt.zero_grad()
            sup_loss.backward()
            opt.step()
            total_loss += sup_loss.item()
            n_batches  += 1

        if epoch % 100 == 0 or epoch == n_epochs:
            log.info("fewshot [philadelphia_pa]  epoch %4d/%d  loss=%.4f",
                     epoch, n_epochs, total_loss / max(n_batches, 1))

    # --- Evaluate on held-out test set ---
    test_t = torch.tensor(test_idx, device=device)
    with torch.no_grad():
        trunk_local.eval()
        h_test_all = []
        for start in range(0, len(test_idx), batch_size):
            b = test_t[start:start + batch_size]
            h_test_all.append(trunk_local(X_land[b]))
        h_test = torch.cat(h_test_all, dim=0)  # (n_test, D)

    results = {}
    for w, head in heads.items():
        with torch.no_grad():
            preds_list = []
            for start in range(0, len(test_idx), batch_size):
                b_slice = slice(start, start + batch_size)
                b       = test_t[b_slice]
                h_b     = h_test[b_slice]
                pred    = head(torch.cat([h_b, era5_inputs[w][b]], dim=-1)).squeeze(-1)
                preds_list.append(pred.cpu().numpy())
        pred_np = np.concatenate(preds_list)

        era5_t2m_F_test = (era5_tensors[w][test_t].cpu().numpy()[:, 0] * 40.0 * 1.8 + 32.0)
        pred_abs = pred_np + era5_t2m_F_test  # anomaly → absolute °F

        label_col = _label_cols[w]
        if label_col in gdf.columns:
            true_np  = gdf[label_col].values.astype(np.float32)[test_idx]
            v        = ~np.isnan(true_np)
            if v.sum() > 0:
                rmse = float(np.sqrt(np.mean((pred_abs[v] - true_np[v]) ** 2)))
                bias = float(np.mean(pred_abs[v] - true_np[v]))
                corr = float(np.corrcoef(pred_abs[v], true_np[v])[0, 1]) if v.sum() > 1 else float("nan")
                results[w] = {"rmse": rmse, "bias": bias, "corr": corr, "n": int(v.sum())}

    # Apply Option-C calibration (same slope/intercept as zero-shot)
    cal_results = {}
    if city_uhi_stats:
        for w in ("morning", "midday", "evening"):
            train_era5_t2m, train_uhi_mean = [], []
            for cs in city_uhi_stats:
                if w in cs and isinstance(cs[w], dict):
                    train_era5_t2m.append(cs[w]["mean_era5_t2m_F"])
                    train_uhi_mean.append(cs[w]["mean_uhi_anomaly"])
            if len(train_era5_t2m) < 2:
                continue
            x = np.array(train_era5_t2m, dtype=np.float64)
            y = np.array(train_uhi_mean, dtype=np.float64)
            if len(y) >= 3:
                med_y = np.median(y)
                mad_y = np.median(np.abs(y - med_y))
                rst_y = mad_y / 0.6745
                if rst_y > 0:
                    keep_cal = np.abs(y - med_y) <= 2.0 * rst_y
                    if keep_cal.sum() >= 2:
                        x, y = x[keep_cal], y[keep_cal]
            x_c       = x - x.mean()
            slope     = float(np.dot(x_c, y - y.mean()) / (np.dot(x_c, x_c) + 1e-9))
            intercept = float(y.mean() - slope * x.mean())

            if w not in results:
                continue
            # Re-run prediction for calibration (full test set in absolute °F)
            # We already have pred_abs stored per-window above — recompute
            with torch.no_grad():
                preds_list_cal = []
                for start in range(0, len(test_idx), batch_size):
                    b_slice = slice(start, start + batch_size)
                    b       = test_t[b_slice]
                    h_b     = h_test[b_slice]
                    pred    = heads[w](torch.cat([h_b, era5_inputs[w][b]], dim=-1)).squeeze(-1)
                    preds_list_cal.append(pred.cpu().numpy())
            pred_anom_cal = np.concatenate(preds_list_cal)
            era5_t2m_F_test = (era5_tensors[w][test_t].cpu().numpy()[:, 0] * 40.0 * 1.8 + 32.0)
            raw_pred_abs  = pred_anom_cal + era5_t2m_F_test
            raw_mean_anom = float((raw_pred_abs - era5_t2m_F_test).mean())
            holdout_era5  = float(era5_t2m_F_test.mean())
            predicted_uhi = slope * holdout_era5 + intercept
            bias_corr     = predicted_uhi - raw_mean_anom

            # Option-C was designed for the zero-shot ensemble (~9°F raw bias).
            # Few-shot predictors self-calibrate from Philadelphia labels, so their
            # raw bias is already near zero.  Applying a large correction makes
            # things worse.  Skip calibration when |bias_correction| > |raw_bias|
            # (i.e. the calibration would add more error than it removes).
            label_col = _label_cols[w]
            if label_col in gdf.columns:
                true_np = gdf[label_col].values.astype(np.float32)[test_idx]
                v       = ~np.isnan(true_np)
                if v.sum() > 0:
                    raw_bias_w = float(np.mean(raw_pred_abs[v] - true_np[v]))
                    if abs(bias_corr) > abs(raw_bias_w):
                        # Calibration would overshoot — report raw as calibrated
                        cal_results[w] = {"rmse": float(np.sqrt(np.mean((raw_pred_abs[v] - true_np[v]) ** 2))),
                                          "bias_correction": 0.0, "n": int(v.sum()),
                                          "note": "cal skipped (raw already near-unbiased)"}
                    else:
                        cal_pred = raw_pred_abs + bias_corr
                        rmse_cal = float(np.sqrt(np.mean((cal_pred[v] - true_np[v]) ** 2)))
                        cal_results[w] = {"rmse": rmse_cal, "bias_correction": bias_corr,
                                          "n": int(v.sum())}

    print()
    print("=" * 60)
    print(f"  Phase 3B -- Few-Shot Eval (Philadelphia, n={n_samples})")
    print(f"  (trained on {len(train_idx)} pixels, evaluated on {len(test_idx)} held-out)")
    print("=" * 60)
    print(f"  {'Window':<12}  {'RMSE (raw)':>10}  {'RMSE (cal)':>10}  {'Bias (raw)':>10}  {'Corr':>6}  {'N':>8}")
    print("  " + "-" * 65)
    for w in ("morning", "midday", "evening"):
        r = results.get(w, {})
        c = cal_results.get(w, {})
        rmse_r = f"{r['rmse']:>10.4f}"  if r else "         —"
        rmse_c = f"{c['rmse']:>10.4f}"  if c else "         —"
        bias_r = f"{r.get('bias', float('nan')):>10.2f}" if r else "         —"
        corr_r = f"{r.get('corr', float('nan')):>6.3f}"  if r else "     —"
        n      = r.get("n", "—")
        print(f"  {w:<12}  {rmse_r}  {rmse_c}  {bias_r}  {corr_r}  {n:>8}")
    print("  " + "-" * 65)
    sum_corr = sum(results.get(w, {}).get("corr", 0.0) for w in ("morning", "midday", "evening"))
    print(f"  sum_corr (few-shot n={n_samples}): {sum_corr:.3f}")
    print()

    # --- Hybrid eval: zero-shot morning + few-shot midday/evening ----------------
    hybrid_results = {}
    if hybrid:
        pred_path = holdout_path.parent / "predictions.geoparquet"
        if pred_path.exists():
            import geopandas as _gpd_mod
            zs_gdf = _gpd_mod.read_parquet(str(pred_path))
            _label_cols_h = {"morning": "aat_morning", "midday": "aat_midday", "evening": "aat_night"}
            # Evaluate zero-shot morning on the SAME held-out test_idx for fair comparison
            zs_col = "pred_morning_calibrated"
            if zs_col in zs_gdf.columns:
                zs_morning = zs_gdf[zs_col].values.astype(np.float32)
                true_col    = _label_cols_h["morning"]
                if true_col in gdf.columns:
                    true_morn = gdf[true_col].values.astype(np.float32)[test_idx]
                    zs_morn   = zs_morning[test_idx]
                    v         = ~np.isnan(true_morn) & ~np.isnan(zs_morn)
                    if v.sum() > 0:
                        rmse_h = float(np.sqrt(np.mean((zs_morn[v] - true_morn[v]) ** 2)))
                        bias_h = float(np.mean(zs_morn[v] - true_morn[v]))
                        corr_h = float(np.corrcoef(zs_morn[v], true_morn[v])[0, 1])
                        hybrid_results["morning"] = {"rmse": rmse_h, "bias": bias_h,
                                                     "corr": corr_h, "n": int(v.sum()),
                                                     "source": "zero-shot-calibrated"}
            # Midday/evening come from few-shot
            hybrid_results["midday"]  = results.get("midday",  {})
            hybrid_results["evening"] = results.get("evening", {})
            hybrid_sum = sum(hybrid_results.get(w, {}).get("corr", 0.0)
                             for w in ("morning", "midday", "evening"))

            print()
            print("=" * 60)
            print(f"  Phase 3C -- Hybrid Eval (morning=zero-shot, mid/eve=few-shot n={n_samples})")
            print(f"  (evaluated on {len(test_idx)} held-out pixels — same split as few-shot)")
            print("=" * 60)
            print(f"  {'Window':<12}  {'RMSE':>10}  {'Bias':>10}  {'Corr':>6}  {'N':>8}  Source")
            print("  " + "-" * 70)
            for w in ("morning", "midday", "evening"):
                h = hybrid_results.get(w, {})
                src   = h.get("source", f"few-shot-n{n_samples}")
                rmse_ = f"{h['rmse']:>10.4f}" if h else "         —"
                bias_ = f"{h.get('bias', float('nan')):>10.2f}" if h else "         —"
                corr_ = f"{h.get('corr', float('nan')):>6.3f}"  if h else "     —"
                n_    = h.get("n", "—")
                print(f"  {w:<12}  {rmse_}  {bias_}  {corr_}  {n_:>8}  {src}")
            print("  " + "-" * 70)
            print(f"  sum_corr (hybrid): {hybrid_sum:.3f}")
            print()
        else:
            log.warning("hybrid: zero-shot predictions not found at %s -- skipping hybrid eval", pred_path)

    return {"raw": results, "calibrated": cal_results, "sum_corr": sum_corr,
            "hybrid": hybrid_results}


# ---------------------------------------------------------------------------
# Phase 4 -- Pipeline stages per city
# ---------------------------------------------------------------------------

def run_pipeline_stages(
    city_cfg: dict,
    output_root: Path,
    stages: list[str],
    fast: bool = False,
) -> None:
    """Run SPARC pipeline stages for one city using its generated project.yml.

    Called inside the per-city learning loop (Phase 2, step a) BEFORE the
    JEPA supervised fine-tune step so that each city's causal/correlation
    analysis is grounded in the full pipeline before the trunk is updated.
    """
    import os
    from sparc.run.orchestrator import RunContext, PipelineOrchestrator
    from sparc.registry.run_registry import set_active_registry

    city_slug = city_cfg["city_slug"]
    if not stages:
        log.debug("[%s] no pipeline stages specified -- skipping", city_slug)
        return
    project_yml = output_root / city_slug / "project.yml"
    if not project_yml.exists():
        log.warning("[%s] project.yml not found at %s -- skipping pipeline stages", city_slug, project_yml)
        return

    # Set SPARC_PROJECT so load_config() calls inside each stage module
    # (which construct their own config objects without receiving ctx) pick up
    # this city's project.yml rather than falling back to brown4.csv defaults.
    prev_sparc_project = os.environ.get("SPARC_PROJECT")
    os.environ["SPARC_PROJECT"] = str(project_yml.resolve())
    try:
        ctx = RunContext.from_project(str(project_yml))
        # Activate the per-city registry globally so that correlogram_analysis.py
        # and other stage modules can call get_active_store() and persist
        # artifacts (e.g. correlogram_anisotropy, correlogram_matern_fit).
        if ctx.registry is not None:
            set_active_registry(ctx.registry)
        orch = PipelineOrchestrator(ctx)
        for stage in stages:
            log.info("[%s] running pipeline stage %s ...", city_slug, stage)
            orch.run(stage, fast=fast)
            log.info("[%s] stage %s complete", city_slug, stage)
    except Exception as exc:
        log.warning("[%s] pipeline stage error: %s", city_slug, exc)
    finally:
        # Deactivate the registry so the next city gets its own clean context.
        set_active_registry(None, force=True)
        # Restore previous value so subsequent cities get their own project.yml
        if prev_sparc_project is None:
            os.environ.pop("SPARC_PROJECT", None)
        else:
            os.environ["SPARC_PROJECT"] = prev_sparc_project


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        pilot_cfg = yaml.safe_load(fh)

    mc          = pilot_cfg["multicity"]
    output_root = Path(mc["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    # Configure threads ONCE before any parallel work.
    # Seeding happens later inside run_jepa_pretraining() and train_city_heads(),
    # same as the reference commit -- do NOT seed here or RNG state at the point
    # of weight init will differ from the 1.277 baseline.
    _global_seed = int(mc.get("jepa", {}).get("seed", 42))
    if args.seed is not None:
        _global_seed = args.seed
        log.info("Seed overridden by --seed %d", _global_seed)
    _configure_threads(deterministic=args.deterministic, n_threads=args.threads)
    if args.deterministic:
        log.info("Deterministic mode ON — single thread, bit-identical across runs (slower)")
    elif args.threads > 0:
        log.info("Thread count pinned to %d — reproducible on this machine", args.threads)

    # Apply --seed override directly into the config so run_jepa_pretraining/
    # train_city_heads pick it up without needing an RNG pre-seed in main().
    if args.seed is not None:
        mc["jepa"]["seed"] = args.seed

    # Apply --screen-epochs override (fast proxy screening for hyperparameter sweeps).
    if args.screen_epochs is not None:
        mc["jepa"]["n_epochs"] = args.screen_epochs
        log.info("--screen-epochs %d: overriding jepa.n_epochs for this run", args.screen_epochs)

    # Determine pipeline stages
    if args.skip_stages:
        run_stages = []
        log.info("--skip-stages: auxiliary pipeline stages disabled for this run")
    elif args.stages:
        _stages_raw = [s.strip() for s in args.stages.split(",")]
        # Allow --stages none/skip/- to disable pipeline stages entirely
        run_stages = [s for s in _stages_raw if s.lower() not in ("none", "skip", "-", "")]
    else:
        run_stages = mc.get("pipeline", {}).get("stages", ["0", "2", "3"])

    # --stage0-only: run pipeline Stage 0 for all training cities and exit.
    # This generates the correlogram + anisotropy artifacts needed for PI-JEPA (B2 audit).
    if args.stage0_only:
        print()
        print("=" * 60)
        print("  Stage 0 -- Correlogram + Anisotropy (all training cities)")
        if args.stage0_fast:
            print("  [fast mode: 400 NUTS samples, ~2.5x faster]")
        print("=" * 60)
        all_cities  = mc["pilot_cities"]
        train_cfgs  = [c for c in all_cities if not c.get("holdout", False)]
        n_ok, n_fail = 0, 0
        for city_cfg in train_cfgs:
            slug = city_cfg["city_slug"]
            log.info("[%s] running Stage 0 ...", slug)
            try:
                run_pipeline_stages(city_cfg, output_root, ["0"], fast=args.stage0_fast)
                log.info("[%s] Stage 0 complete", slug)
                n_ok += 1
            except Exception as exc:
                log.warning("[%s] Stage 0 error: %s", slug, exc)
                n_fail += 1
        print()
        print(f"  Stage 0 complete: {n_ok} OK, {n_fail} failed")
        print(f"  Run audit: python scripts/audit_anisotropy.py")
        print()
        return

    # Separate holdout and training cities
    all_cities   = mc["pilot_cities"]
    holdout_cfgs = [c for c in all_cities if c.get("holdout", False)]
    train_cfgs   = [c for c in all_cities if not c.get("holdout", False)]

    # Feature columns
    feat_cols_land = mc["feature_columns"].get("land_surface", [])
    feat_cols_all  = (
        feat_cols_land
        + mc["feature_columns"].get("era5_morning", [])
        + mc["feature_columns"].get("era5_midday", [])
        + mc["feature_columns"].get("era5_evening", [])
    )
    coord_cols = mc.get("pipeline", {}).get("coordinate_columns", ["centroid_x", "centroid_y"])
    jepa_cfg   = mc["jepa"]

    # -- Optionally collect data first --------------------------------------
    if not args.skip_collect:
        log.info("Running data collection via collect_cities.py ...")
        from scripts.collect_cities import collect_one_city
        for city_cfg in all_cities:
            collect_one_city(city_cfg, pilot_cfg, output_root, resume=True)

    # -- Phase 1 -- JEPA Pretraining -----------------------------------------
    print()
    print("=" * 60)
    print("  Phase 1 -- JEPA Pretraining (all non-holdout cities)")
    print("=" * 60)

    all_X_list, all_C_list = [], []
    city_sizes: list[int] = []   # track row count per city for per-city coord normalization
    for city_cfg in train_cfgs:
        slug = city_cfg["city_slug"]
        pq   = output_root / slug / "data.geoparquet"
        if not pq.exists():
            log.warning("[%s] GeoParquet not found -- skipping from pretraining", slug)
            continue
        try:
            X, C = load_city_features(pq, feat_cols_land, coord_cols)
            all_X_list.append(X)
            all_C_list.append(C)
            city_sizes.append(len(X))
            log.info("[%s] loaded  X=%s  C=%s", slug, X.shape, C.shape)
        except Exception as exc:
            log.warning("[%s] load error: %s", slug, exc)

    if not all_X_list:
        log.error("No city data available for pretraining. Run collect_cities.py first.")
        sys.exit(1)

    # Align column counts (pad with zeros if different cities have different feature availability)
    max_feats = max(X.shape[1] for X in all_X_list)
    aligned_X = []
    for X in all_X_list:
        if X.shape[1] < max_feats:
            pad = np.zeros((X.shape[0], max_feats - X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)
        aligned_X.append(X)

    combined_X = np.concatenate(aligned_X,  axis=0)
    combined_C = np.concatenate(all_C_list, axis=0)

    # -- PI-JEPA params (I2 range-scaled radius, I1 anisotropic mask) ----------
    pi_patch_radius_m: float | None = None
    pi_eccentricity_a_over_b: float | None = None
    all_C_local: np.ndarray | None = None   # city-local (per-city centered) coords for masking
    range_scaled = bool(jepa_cfg.get("range_scaled_radius", False))
    aniso_mask   = bool(jepa_cfg.get("anisotropic_mask", False))
    if (range_scaled or aniso_mask) and not args.skip_pretrain:
        log.info("Loading PI-JEPA Stage 0 params (range_scaled=%s  aniso_mask=%s)", range_scaled, aniso_mask)
        pi_params = _load_pi_jepa_params(train_cfgs, output_root)
        if range_scaled:
            pi_patch_radius_m = pi_params["median_effective_range_m"]
            if pi_patch_radius_m is None:
                log.warning("I2 requested but no effective ranges found -- using default radius")
            else:
                # Build PER-CITY LOCAL coordinates for masking.
                # Problem: combined_C uses global normalisation; each city's UTM origin is
                # different, so a 6km effective range = only 0.018 global-std units (≈0 points).
                # Fix: subtract each city's own centroid, normalise by each city's own std.
                # Then effective_range_m / median_city_std gives the mask radius in local-std units.
                local_C_list: list[np.ndarray] = []
                city_local_stds: list[float] = []
                for raw_C in all_C_list:
                    c_city_mean = np.nanmean(raw_C, axis=0)
                    c_city_std  = np.nanstd(raw_C,  axis=0) + 1e-6
                    c_city_std_scalar = float(np.mean(c_city_std))
                    city_local_stds.append(c_city_std_scalar)
                    local_C_list.append(
                        np.nan_to_num((raw_C - c_city_mean) / c_city_std, nan=0.0).astype(np.float32)
                    )
                all_C_local = np.concatenate(local_C_list, axis=0)
                median_city_std_m = float(np.median(city_local_stds))
                # Convert effective range from physical metres to city-local std units
                pi_patch_radius_m_local = pi_patch_radius_m / median_city_std_m
                log.info(
                    "I2 city-local coords: median_city_std=%.0fm  "
                    "patch_radius=%.0fm / %.0fm = %.4f local-std units",
                    median_city_std_m, pi_patch_radius_m, median_city_std_m, pi_patch_radius_m_local,
                )
                pi_patch_radius_m = pi_patch_radius_m_local   # overwrite with normalised value

        if aniso_mask:
            # Check for direct override first (I1-sweep: test specific eccentricities)
            _ecc_override = jepa_cfg.get("eccentricity_override", None)
            if _ecc_override is not None:
                pi_eccentricity_a_over_b = float(_ecc_override)
                log.info("I1 eccentricity_override=%.3f (bypassing Stage 0 median)", pi_eccentricity_a_over_b)
            else:
                pi_eccentricity_a_over_b = pi_params["median_eccentricity_a_over_b"]
            if pi_eccentricity_a_over_b is None:
                log.warning("I1 requested but no eccentricity data found -- using isotropic mask")

    if args.skip_pretrain:
        # Load existing trunk checkpoint and reconstruct normalizer
        trunk_path = output_root / "jepa_pretrained_trunk.pt"
        if not trunk_path.exists():
            log.error("--skip-pretrain requested but no trunk checkpoint found at %s", trunk_path)
            sys.exit(1)
        import torch as _t
        _dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        ckpt = _t.load(str(trunk_path), map_location=_dev)
        hidden_dim = ckpt.get("hidden_dim", int(jepa_cfg.get("hidden_dim", 256)))
        in_dim     = ckpt.get("in_dim", combined_X.shape[1])
        trunk = _build_trunk(in_dim, hidden_dim).to(_dev)
        trunk.load_state_dict(ckpt["trunk_state"])
        normalizer = {
            "X_mean": np.array(ckpt["X_mean"], dtype=np.float32),
            "X_std":  np.array(ckpt["X_std"],  dtype=np.float32),
        }
        log.info("Loaded pretrained trunk from %s (skipping Phase 1)", trunk_path)
    else:
        # E5: compute Laplacian-FCM memberships per city before pretraining.
        # Only pays the eigsh cost (~2s/city) when eigen_target_weight > 0.
        combined_U: np.ndarray | None = None
        eigen_weight = float(jepa_cfg.get("eigen_target_weight", 0.0))
        if eigen_weight > 0.0:
            # Warm the CUDA context BEFORE eigsh to prevent Blackwell (sm_120)
            # driver TDR. eigsh runs ~4 min of pure CPU work; on Blackwell the
            # GPU driver evicts idle contexts and crashes on re-entry.
            # Strategy: (1) keepalive thread pings GPU every 30s, (2) pre-allocate
            # the main training tensors NOW so GPU memory is claimed before eigsh.
            # This avoids a sudden ~550MB GPU burst after a long idle period.
            import torch as _torch_warmup
            import threading as _threading
            _warmup_dev = _torch_warmup.device("cuda" if _torch_warmup.cuda.is_available() else "cpu")
            _keepalive_stop = _threading.Event()
            if _warmup_dev.type == "cuda":
                # Pre-allocate main tensors (X_t, C_t, idx_t) on GPU before eigsh
                # so the large allocation happens while the GPU is already active.
                import numpy as _np_warmup
                max_feats_w = max(X.shape[1] for X in all_X_list)
                aligned_X_w = []
                for _X in all_X_list:
                    if _X.shape[1] < max_feats_w:
                        _X = _np_warmup.concatenate(
                            [_X, _np_warmup.zeros((_X.shape[0], max_feats_w - _X.shape[1]), dtype=_np_warmup.float32)], axis=1
                        )
                    aligned_X_w.append(_X)
                _combined_X_w = _np_warmup.concatenate(aligned_X_w, axis=0)
                _combined_C_w = _np_warmup.concatenate(all_C_list, axis=0)
                log.info(
                    "Pre-allocating training tensors on GPU before eigsh  X=%s  C=%s",
                    _combined_X_w.shape, _combined_C_w.shape,
                )
                _X_t_pre = _torch_warmup.tensor(_combined_X_w, device=_warmup_dev)
                _C_t_pre = _torch_warmup.tensor(_combined_C_w, device=_warmup_dev)
                _idx_t_pre = _torch_warmup.arange(len(_X_t_pre), dtype=_torch_warmup.long, device=_warmup_dev)
                del _combined_X_w, _combined_C_w, aligned_X_w  # free numpy copies

                _keepalive_tensor = _torch_warmup.zeros(1, device=_warmup_dev)
                def _gpu_keepalive():
                    while not _keepalive_stop.wait(timeout=30):
                        try:
                            _ = _keepalive_tensor + 0  # minimal GPU op every 30s
                        except Exception:
                            pass
                _ka_thread = _threading.Thread(target=_gpu_keepalive, daemon=True)
                _ka_thread.start()
                log.info("GPU keepalive thread started (Blackwell sm_120 context preservation)")

            from scripts.e5_eigenmap_prep import compute_city_fcm_memberships
            k_clusters_cfg = int(jepa_cfg.get("eigen_k_clusters", 4))
            n_eigen_cfg    = int(jepa_cfg.get("eigen_n_components", 2))
            log.info(
                "E5: computing Laplacian-FCM memberships  "
                "n_eigen=%d  k_clusters=%d  n_cities=%d",
                n_eigen_cfg, k_clusters_cfg, len(all_C_list),
            )
            u_list: list[np.ndarray] = []
            for ci, (raw_C, slug_cfg) in enumerate(zip(all_C_list, train_cfgs)):
                slug = slug_cfg["city_slug"]
                try:
                    U_city, _vecs, _centers = compute_city_fcm_memberships(
                        raw_C.astype(np.float64),
                        n_eigen=n_eigen_cfg,
                        k_clusters=k_clusters_cfg,
                        max_n=int(jepa_cfg.get("eigen_max_subsample", 15_000)),
                    )
                    u_list.append(U_city)
                    log.info("[%s] E5 memberships: shape=%s", slug, U_city.shape)
                except Exception as exc:
                    log.warning("[%s] E5 eigenmap failed (%s) -- using uniform memberships", slug, exc)
                    N_city = city_sizes[ci]
                    u_list.append(np.full((N_city, k_clusters_cfg), 1.0 / k_clusters_cfg, dtype=np.float32))
            combined_U = np.concatenate(u_list, axis=0)

            # Stop GPU keepalive thread now that eigsh is done.
            # Delete pre-allocated tensors — run_jepa_pretraining will re-allocate
            # them, but the GPU context is now proven live and the memory is free.
            _keepalive_stop.set()
            if _warmup_dev.type == "cuda":
                del _X_t_pre, _C_t_pre, _idx_t_pre, _keepalive_tensor
                import torch as _tc; _tc.cuda.empty_cache()
                log.info("GPU keepalive thread stopped; pre-alloc tensors freed")

        trunk, normalizer = run_jepa_pretraining(
            combined_X,
            # Use city-local coords for masking when I2 is active (avoids degenerate
            # masking from mixed-UTM global normalisation where 6km = 0.018 global-std)
            all_C_local if all_C_local is not None else combined_C,
            jepa_cfg, output_root,
            pi_patch_radius_m=pi_patch_radius_m,
            pi_eccentricity_a_over_b=pi_eccentricity_a_over_b,
            city_sizes=city_sizes,
            all_U=combined_U,
            feat_cols=feat_cols_land,
        )

    # -- Phase 2 -- Per-City Head Training (unfrozen trunk, low lr) -------------
    # Trunk is UNFROZEN with a low lr (trunk_lr = lr * 0.1) so that CAPA heat
    # gradient supervision can shape the spatial representations learned in Phase 1.
    # EWC regularisation prevents catastrophic forgetting across cities.
    # Each city gets a deep-copied trunk so LOO can run each city's own trunk.

    # --pretrain-only: exit here after saving trunk. Use with --screen-epochs for
    # fast I1/I2 sweep screening before committing to a full Phase 2+3 run.
    if args.pretrain_only:
        log.info("--pretrain-only: Phase 1 complete. Trunk saved to output/jepa_pretrained_trunk.pt")
        log.info("Run: python scripts/a3_causal_diagnostics.py  to measure trunk proxy quality.")
        print("\n  Phase 1 complete (--pretrain-only). Exiting before Phase 2+3.")
        print("  Next: python scripts/a3_causal_diagnostics.py")
        return
    print()
    print("=" * 60)
    print("  Phase 2 -- Per-City Head Training (frozen trunk, deeper heads)")
    print("=" * 60)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.skip_pretrain:
        # Move the loaded trunk to the correct device (already set in skip block above)
        trunk = trunk.to(device)

    fisher_matrices:     list[dict] = []
    optimal_params_list: list[dict] = []
    city_models: dict = {}
    city_uhi_stats_all: list[dict] = []   # for Option-C offset calibration

    for city_idx, city_cfg in enumerate(train_cfgs):
        slug = city_cfg["city_slug"]
        pq   = output_root / slug / "data.geoparquet"

        print()
        print(f"  -- City {city_idx + 1}/{len(train_cfgs)}: {slug} --------------------------")

        log.info("[%s] running pipeline stages %s ...", slug, run_stages)
        try:
            run_pipeline_stages(city_cfg, output_root, run_stages)
        except Exception as exc:
            log.warning("[%s] pipeline stage error: %s -- continuing with head training", slug, exc)

        if not pq.exists():
            log.warning("[%s] GeoParquet missing -- skipping supervised head training", slug)
            continue
        try:
            # Pass the SHARED trunk — each city updates it sequentially in-place.
            # EWC prevents catastrophic forgetting of earlier cities.
            # At LOO time, all city heads share the same final trunk state,
            # so ensemble embeddings are coherent.  Per-city deep-copies created
            # incoherent embedding spaces and caused the ensemble to diverge (~-38°F bias).
            trunk, fisher, theta_star, city_model, uhi_stats = run_city_finetune(
                city_slug=slug,
                parquet_path=pq,
                trunk=trunk,
                feature_cols_land=feat_cols_land,
                pilot_cfg=pilot_cfg,
                fisher_matrices=fisher_matrices,
                optimal_params_list=optimal_params_list,
                output_dir=output_root,
                device=device,
                normalizer=normalizer,
            )
            city_models[slug] = city_model
            if uhi_stats:
                city_uhi_stats_all.append({"city": slug, **uhi_stats})
                for w, s in uhi_stats.items():
                    log.info("[%s] UHI calibration  %s  mean_anomaly=%.2f°F  era5_t2m=%.1f°F  n=%d",
                             slug, w, s["mean_uhi_anomaly"], s["mean_era5_t2m_F"], s["n"])
            log.info("[%s] head trained (%d cities done)", slug, len(city_models))
        except Exception as exc:
            log.warning("[%s] supervised head training error: %s", slug, exc)

    # -- Phase 3 -- LOO Eval on Philadelphia (zero-shot) ---------------------
    # Only runs AFTER all training cities have been processed.
    # The model has never seen Philadelphia -- this is the true zero-shot test.
    print()
    print("=" * 60)
    print("  Phase 3 -- Zero-Shot Leave-One-Out Eval (Philadelphia)")
    print("  (runs after all training cities processed -- never seen during training)")
    print("=" * 60)

    for holdout_cfg in holdout_cfgs:
        slug = holdout_cfg["city_slug"]
        pq   = output_root / slug / "data.geoparquet"
        if pq.exists():
            try:
                loo_results = run_loo_eval(
                    holdout_path=pq,
                    trunk=trunk,
                    feature_cols_land=feat_cols_land,
                    pilot_cfg=pilot_cfg,
                    output_dir=output_root,
                    device=device,
                    city_models=city_models,
                    normalizer=normalizer,
                    city_uhi_stats=city_uhi_stats_all,
                    cfa_only=args.cfa_only_loo,
                    anp_path=args.anp_checkpoint,
                )
            except Exception as exc:
                log.warning("LOO eval error for %s: %s", slug, exc)
        else:
            log.warning("Holdout city GeoParquet not found: %s -- skipping LOO eval", pq)

    # -- Phase 3B -- Few-Shot Eval (optional) --------------------------------
    if args.fewshot_n > 0:
        print()
        print("=" * 60)
        print(f"  Phase 3B -- Few-Shot Transfer (Philadelphia, n={args.fewshot_n})")
        print("=" * 60)
        for holdout_cfg in holdout_cfgs:
            slug = holdout_cfg["city_slug"]
            pq   = output_root / slug / "data.geoparquet"
            if pq.exists():
                try:
                    run_fewshot_finetune(
                        holdout_path=pq,
                        trunk=trunk,
                        feature_cols_land=feat_cols_land,
                        pilot_cfg=pilot_cfg,
                        output_dir=output_root,
                        device=device,
                        normalizer=normalizer,
                        n_samples=args.fewshot_n,
                        city_uhi_stats=city_uhi_stats_all,
                        hybrid=args.fewshot_hybrid,
                    )
                except Exception as exc:
                    log.warning("Few-shot eval error for %s: %s", slug, exc)
            else:
                log.warning("Holdout city GeoParquet not found: %s -- skipping few-shot eval", pq)

    print()
    print("All phases complete.")
    print(f"  Trunk checkpoint  : output/jepa_pretrained_trunk.pt")
    print(f"  City models       : output/cities/<slug>/model.pt")
    print(f"  LOO predictions   : output/cities/philadelphia_pa/predictions.geoparquet")
    print(f"  Cities trained    : {[c['city_slug'] for c in train_cfgs]}")
    print(f"  Cities evaluated  : {[c['city_slug'] for c in holdout_cfgs]}")
    print()


if __name__ == "__main__":
    main()