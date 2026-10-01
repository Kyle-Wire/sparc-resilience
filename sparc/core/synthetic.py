"""Synthetic fixtures with planted, analytically known truths.

These drive the core unit tests: every stage must recover what was planted.

* :func:`make_operator_fixture` — a known source ``q`` pushed through the
  advection–diffusion–relaxation operator with known (L, v).
* :func:`make_synthetic_city` — a small city with canopy / impervious /
  albedo / NDVI (mediator) / elevation / water, a smooth confounder, a
  physics-shaped target and an exactly saturating canopy response.
* :func:`make_interference_fixture` and :func:`make_dose_response_fixture`
  — small causal fixtures with known own/neighbour effects and a known
  dose-response curve.
* :func:`write_demo_project` — the synthetic city as a complete project
  (CSV with a CRS, truth, people/land-cover layers, a CMIP6-style change
  table and a config): the one generator behind the Studio demo template,
  the committed Studio fixture and the event-invariant tests.

Target units are "°F-like" (arbitrary but realistic magnitudes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from sparc.core import operators as ops


def gaussian_random_field(shape: tuple[int, int], range_cells: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance smooth field (Gaussian-filtered white noise)."""
    z = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=max(range_cells / 2.0, 0.5), mode="wrap")
    return (z - z.mean()) / (z.std() + 1e-12)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def kernel_mass_radius(L: float, v: tuple[float, float], dx: float, mass: float = 0.9, n: int = 257) -> float:
    """Radius (m) around the source containing ``mass`` of |G| for the
    discrete operator Green's function — the true 'area of influence'."""
    q = np.zeros((n, n))
    c = n // 2
    q[c, c] = 1.0
    g = np.abs(ops.solve(q, L, v, dx=dx))
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c).ravel() * dx
    w = g.ravel()
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / w.sum()
    return float(r[order][np.searchsorted(cum, mass)])


# ---------------------------------------------------------------------------
# Operator fixture
# ---------------------------------------------------------------------------


def make_operator_fixture(
    n: int = 128,
    dx: float = 30.0,
    L: float = 150.0,
    v: tuple[float, float] = (90.0, -45.0),
    a: float = 2.0,
    b: float = 0.5,
    noise: float = 0.05,
    seed: int = 0,
) -> dict:
    """y = a·solve(q; L, v) + b + ε on a full n×n grid with a short-range source."""
    rng = np.random.default_rng(seed)
    q = gaussian_random_field((n, n), range_cells=2.0, rng=rng)
    phi = ops.solve(q, L, v, dx=dx)
    y = a * phi + b
    y = y + noise * y.std() * rng.standard_normal(y.shape)
    yy, xx = np.mgrid[0:n, 0:n]
    return {
        "q": q,
        "phi": phi,
        "y": y,
        "x": (xx * dx).ravel().astype(float),
        "y_coord": (yy * dx).ravel().astype(float),
        "values": y.ravel(),
        "truth": {"L": L, "v": v, "a": a, "b": b, "dx": dx},
    }


# ---------------------------------------------------------------------------
# Synthetic city
# ---------------------------------------------------------------------------


@dataclass
class SyntheticCity:
    frame: pd.DataFrame                      # x, y (m), T, predictors, U (hidden confounder)
    truth: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)  # full rasters (ny, nx), NaN outside mask
    mask: np.ndarray | None = None

    def true_footprint(self) -> np.ndarray:
        """Exact footprint per +1 pp canopy at every point: Σ_j ∂ΔT_j/∂c_i =
        a·(Gᵀ∗mask)_i·∂q_i/∂c_i, including the NDVI mediator path."""
        t = self.truth
        c = self.frame["canopy"].to_numpy(float)
        dq = -t["A_c"] * canopy_shape_derivative(c, t) - t["w_ndvi"] * t["k_ndvi"]
        gm = ops.green_mass(self.mask.astype(float), t["L"], t["v"], t["dx"])
        return t["a"] * gm[self.fields["_iy"], self.fields["_ix"]] * dq

    def true_response(self, canopy_increment: float) -> np.ndarray:
        """Exact ΔT change at every point for a uniform canopy increment
        (NDVI updated through its known mediator law)."""
        t = self.truth
        c = np.nan_to_num(self.fields["canopy"])
        dq = -t["A_c"] * (canopy_shape(c + canopy_increment, t) - canopy_shape(c, t))
        dq = np.where(self.mask, dq, 0.0)
        # NDVI mediator: ndvi += k_ndvi·Δc, entering q with weight -w_ndvi.
        dq = dq - t["w_ndvi"] * t["k_ndvi"] * canopy_increment * self.mask
        dphi = ops.solve(dq, t["L"], t["v"], dx=t["dx"])
        rows, cols = self.fields["_iy"], self.fields["_ix"]
        return t["a"] * dphi[rows, cols]


def canopy_shape(c: np.ndarray, t: dict) -> np.ndarray:
    """Planted canopy cooling shape h(c) ∈ [0, 1) (times A_c in the source)."""
    if t.get("canopy_form", "saturating") == "sigmoid":
        c0, w = t["c0"], t["cw"]
        s0 = _sigmoid(-c0 / w)
        return (_sigmoid((c - c0) / w) - s0) / (1.0 - s0)
    return 1.0 - np.exp(-c / t["d_c"])


def canopy_shape_derivative(c: np.ndarray, t: dict) -> np.ndarray:
    if t.get("canopy_form", "saturating") == "sigmoid":
        c0, w = t["c0"], t["cw"]
        s0, s = _sigmoid(-c0 / w), _sigmoid((c - c0) / w)
        return s * (1.0 - s) / w / (1.0 - s0)
    return np.exp(-c / t["d_c"]) / t["d_c"]


def make_synthetic_city(
    n: int = 96,
    dx: float = 30.0,
    mask_fraction: float = 0.7,
    L: float = 150.0,
    v: tuple[float, float] = (60.0, 0.0),
    seed: int = 0,
    noise_frac: float = 0.05,
    canopy_form: str = "saturating",
) -> SyntheticCity:
    """Synthetic city on an n×n lattice (dx metres) with a ragged footprint.

    Planted truths (returned in ``truth``):
      * physics: ΔT_phys = a·solve(q; L, v) with source q built from
        impervious, albedo, NDVI and a saturating canopy term
        −A_c·(1 − exp(−canopy/d_c)); water cells are a heat sink.
      * saturation: a uniform canopy increment d changes every cell by
        (…)·(1 − exp(−d/d_c)) exactly, so the per-cell saturation scale is d_c.
      * a local threshold effect of albedo (> 0.3 cools by 0.6) that only
        tree models capture;
      * an elevation lapse term and a smooth hidden confounder U that
        drives both canopy and temperature.
    """
    rng = np.random.default_rng(seed)
    shape = (n, n)

    # Ragged footprint: threshold a smooth field, keep the largest blob.
    foot = gaussian_random_field(shape, 12.0, rng)
    thr = np.quantile(foot, 1.0 - mask_fraction)
    mask = foot > thr
    lab, nlab = ndimage.label(mask)
    if nlab > 1:
        sizes = ndimage.sum(mask, lab, range(1, nlab + 1))
        mask = lab == (1 + int(np.argmax(sizes)))

    U = gaussian_random_field(shape, 50.0, rng)                       # ~1.5 km confounder
    z_c = gaussian_random_field(shape, 6.0, rng)
    z_1 = gaussian_random_field(shape, 8.0, rng)
    z_2 = gaussian_random_field(shape, 4.0, rng)
    elev = 20.0 + 10.0 * gaussian_random_field(shape, 30.0, rng)

    canopy = 60.0 * _sigmoid(1.2 * z_c + 0.8 * U)
    canopy[rng.random(shape) < 0.2] = 0.0                              # 20% zeros (hurdle)
    impervious = 100.0 * _sigmoid(-0.7 * z_1 + 0.7 * z_2 - 0.5 * U)
    albedo = np.clip(0.18 + 0.06 * gaussian_random_field(shape, 5.0, rng) - 0.03 * (impervious / 100.0 - 0.5), 0.05, 0.6)
    k_ndvi = 0.004
    ndvi = 0.1 + k_ndvi * canopy - 0.001 * impervious + 0.02 * rng.standard_normal(shape)

    # Water: a meandering river (cells within ~1.5 cells of a sine line).
    yy, xx = np.mgrid[0:n, 0:n]
    river_y = n * 0.35 + 6.0 * np.sin(xx / 9.0)
    water = np.abs(yy - river_y) < 1.5
    water_dist = ndimage.distance_transform_edt(~water) * dx

    A_c, d_c = 1.6, 15.0
    shape_t = {"canopy_form": canopy_form, "d_c": d_c, "c0": 40.0, "cw": 6.0}
    w_ndvi = 1.0
    w_water = 1.2
    q = (
        (1.0 - albedo) * (0.4 + 0.6 * impervious / 100.0)
        - A_c * canopy_shape(canopy, shape_t)
        - w_ndvi * ndvi
        - w_water * water
    )
    q = np.where(mask, q, 0.0)
    q = np.where(mask, q - q[mask].mean(), 0.0)
    a = 6.0
    phi = ops.solve(q, L, v, dx=dx)
    gamma = -0.02
    thresh = -0.6 * (albedo > 0.3)
    T_clean = a * phi + gamma * (elev - elev[mask].mean()) + thresh + 0.4 * U
    sd = T_clean[mask].std()
    T = T_clean + noise_frac * sd * rng.standard_normal(shape) + 88.0

    iy, ix = np.nonzero(mask)
    frame = pd.DataFrame(
        {
            "x": ix * dx + 1000.0,
            "y": iy * dx + 2000.0,
            "T": T[iy, ix],
            "canopy": canopy[iy, ix],
            "impervious": impervious[iy, ix],
            "albedo": albedo[iy, ix],
            "ndvi": ndvi[iy, ix],
            "elevation": elev[iy, ix],
            "water_dist": water_dist[iy, ix],
            "U": U[iy, ix],
        }
    )
    fields = {
        "canopy": np.where(mask, canopy, np.nan),
        "impervious": np.where(mask, impervious, np.nan),
        "albedo": np.where(mask, albedo, np.nan),
        "ndvi": np.where(mask, ndvi, np.nan),
        "q": q,
        "phi": phi,
        "_iy": iy,
        "_ix": ix,
    }
    truth = {
        "L": L, "v": v, "a": a, "dx": dx, "A_c": A_c, "d_c": d_c, "gamma": gamma,
        "canopy_form": canopy_form, "c0": shape_t["c0"], "cw": shape_t["cw"],
        "k_ndvi": k_ndvi, "w_ndvi": w_ndvi, "w_water": w_water,
        "influence_radius_90": kernel_mass_radius(L, v, dx, 0.9),
        "noise_sd": noise_frac * sd,
    }
    return SyntheticCity(frame=frame, truth=truth, fields=fields, mask=mask)


def synthetic_city_config(out_dir: str = "output/core/synthetic") -> dict:
    """A core config dict matching :func:`make_synthetic_city`'s columns."""
    return {
        "name": "synthetic_city",
        "data": {"path": None, "target": "T", "x": "x", "y": "y", "coord_unit": "m",
                 "target_units": "degF", "background": "median"},
        "predictors": ["canopy", "impervious", "albedo", "ndvi", "elevation", "water_dist"],
        "qa": {"clip": {"canopy": [0, 100], "impervious": [0, 100], "albedo": [0.02, 0.9]}},
        "actionable": {
            "canopy": {"min": 0, "max": 100, "doses": [0, 5, 10, 15, 20, 30, 40], "cost_per_unit": 1.0},
            "albedo": {"min": 0.02, "max": 0.9, "doses": [0, 0.05, 0.1, 0.15, 0.2]},
        },
        "mediators": {"ndvi": {"parents": ["canopy", "impervious"], "context": ["elevation"],
                               "monotone": {"canopy": 1, "impervious": -1}}},
        # The fixture's advection v = (60, 0) m corresponds to a light breeze
        # u = v/τ with τ = 1800 s — supplied as the "wind record" so advection
        # is fitted (as it would be for a real city with ERA5 wind).
        "physics": {"roles": {"albedo": "albedo", "canopy": "canopy", "impervious": "impervious",
                              "ndvi": "ndvi", "elevation": "elevation", "water_distance": "water_dist"},
                    "wind": [60.0 / 1800.0, 0.0], "tau_s": 1800.0},
        "cv": {"n_folds": 3},
        "stacker": {"epochs": 200, "tune_lambda": [0.0, 1.0]},
        "causal": {"treatments": ["canopy"], "confounders": {"canopy": ["impervious", "albedo", "elevation", "water_dist"]},
                   "contrast": {"canopy": 10}},
        "optimize": {"variable": "canopy", "budget": 2000.0},
        "output": {"dir": out_dir},
    }


# ---------------------------------------------------------------------------
# Causal fixtures
# ---------------------------------------------------------------------------


def make_interference_fixture(
    n: int = 64,
    dx: float = 30.0,
    theta_own: float = -0.05,
    theta_nbr: float = -0.03,
    radius_m: float = 150.0,
    seed: int = 0,
) -> dict:
    """Y = θ_o·T + θ_n·T̄ + g(X, s) + ε with confounding through X and space.

    T̄ is the self-excluded Gaussian neighbourhood mean of T with
    σ = radius/2 (the same exposure mapping the core uses).
    """
    rng = np.random.default_rng(seed)
    shape = (n, n)
    x1 = gaussian_random_field(shape, 8.0, rng)
    x2 = gaussian_random_field(shape, 3.0, rng)
    s = gaussian_random_field(shape, 30.0, rng)                      # spatial confounder
    T = 40.0 + 12.0 * x1 + 6.0 * s + 8.0 * gaussian_random_field(shape, 4.0, rng)
    mask = np.ones(shape, dtype=bool)
    sigma_cells = radius_m / dx / 2.0
    Tbar = ops.masked_gaussian(T, mask, sigma_cells, exclude_self=True)
    g = 0.8 * np.sin(x1) + 0.5 * x2**2 + 0.6 * s
    Y = theta_own * T + theta_nbr * Tbar + g + 0.15 * rng.standard_normal(shape)
    yy, xx = np.mgrid[0:n, 0:n]
    frame = pd.DataFrame({"x": (xx * dx).ravel(), "y": (yy * dx).ravel(), "Y": Y.ravel(), "T": T.ravel(),
                          "x1": x1.ravel(), "x2": x2.ravel()})
    return {"frame": frame, "truth": {"theta_own": theta_own, "theta_nbr": theta_nbr, "radius_m": radius_m, "dx": dx}}


def make_dose_response_fixture(n: int = 64, dx: float = 30.0, A: float = 2.0, d: float = 15.0, seed: int = 0) -> dict:
    """No interference: Y = −A·(1 − exp(−T/d)) + g(X) + ε, T confounded by X,
    with a 20% point mass at T = 0."""
    rng = np.random.default_rng(seed)
    shape = (n, n)
    x1 = gaussian_random_field(shape, 6.0, rng)
    x2 = gaussian_random_field(shape, 3.0, rng)
    T = 60.0 * _sigmoid(1.0 * x1 + 0.5 * rng.standard_normal(shape))
    T[rng.random(shape) < 0.2] = 0.0
    f = -A * (1.0 - np.exp(-T / d))
    Y = f + 0.7 * x1 + 0.3 * x2**2 + 0.1 * rng.standard_normal(shape)
    yy, xx = np.mgrid[0:n, 0:n]
    frame = pd.DataFrame({"x": (xx * dx).ravel(), "y": (yy * dx).ravel(), "Y": Y.ravel(), "T": T.ravel(),
                          "x1": x1.ravel(), "x2": x2.ravel()})

    def true_curve(t):
        return -A * (1.0 - np.exp(-np.asarray(t, dtype=float) / d))

    return {"frame": frame, "truth": {"A": A, "d": d, "curve": true_curve}}


# ---------------------------------------------------------------------------
# Demo project (SPARC Studio's synthetic city)
# ---------------------------------------------------------------------------

DEMO_CRS = "EPSG:32619"                       # UTM 19N; the offset below puts the city at a fictional site
DEMO_OFFSET_M = (300_000.0, 4_630_000.0)
DEMO_MODELS = {"DEMO-A": 0.75, "DEMO-B": 0.9, "DEMO-C": 1.0, "DEMO-D": 1.1, "DEMO-E": 1.25, "DEMO-F": 1.45}
DEMO_WARMING_K = {                            # multi-model median summer warming per SSP and period (K)
    "ssp126": {"2021-2040": 0.9, "2041-2060": 1.3, "2081-2100": 1.5},
    "ssp245": {"2021-2040": 1.0, "2041-2060": 1.7, "2081-2100": 2.6},
    "ssp370": {"2021-2040": 1.0, "2041-2060": 2.0, "2081-2100": 3.7},
    "ssp585": {"2021-2040": 1.1, "2041-2060": 2.4, "2081-2100": 4.6},
}
_DEMO_DECIMALS = {"T": 3, "canopy": 3, "impervious": 3, "albedo": 4, "ndvi": 4, "elevation": 2, "water_dist": 1}


def demo_config(name: str = "synthetic_demo", out_dir: str = "runs") -> dict:
    """The demo project's core config (paths relative to the project directory)."""
    raw = synthetic_city_config(out_dir=out_dir)
    raw["name"] = name
    raw["data"].update(path="data/city.csv", crs=DEMO_CRS, id="id")
    raw["actionable"] = {
        "canopy": {"min": 0, "max": 100, "doses": [0, 5, 10, 15, 20, 30, 40], "cost_per_unit": 1.0,
                   "direction": "increase", "unit": "pp"},
        "impervious": {"min": 0, "max": 100, "doses": [0, 5, 10, 20], "direction": "decrease", "unit": "pp"},
        "albedo": {"min": 0.02, "max": 0.9, "doses": [0, 0.05, 0.1, 0.15, 0.2], "direction": "increase",
                   "unit": "albedo"},
    }
    raw["scenarios"] = [
        {"name": "Canopy Increase", "variable": "canopy", "direction": "increase", "increments": [5, 10, 20]},
        {"name": "Impervious Decrease", "variable": "impervious", "direction": "decrease", "increments": [10, 20]},
        {"name": "Albedo Increase", "variable": "albedo", "direction": "increase", "increments": [0.05, 0.1]},
    ]
    raw["joint_scenarios"] = [{"name": "Cooling package", "interventions": [
        {"variable": "canopy", "direction": "increase", "increment": 10},
        {"variable": "impervious", "direction": "decrease", "increment": 10},
        {"variable": "albedo", "direction": "increase", "increment": 0.05}]}]
    raw["causal"]["treatments"] = ["canopy"]
    raw["climate"] = {"enabled": True, "source": "table", "table": "inputs/climate/demo_cmip6.csv"}
    raw["optimize"] = {"variable": "canopy", "budget": 2000.0}
    raw["planner"] = {"layers": "inputs/layers/demo_layers.parquet"}
    raw["report"] = {"title": "Synthetic city (DEMO)", "place": "a fictional city (synthetic data)"}
    return raw


def demo_layers(city: SyntheticCity, ids: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> pd.DataFrame:
    """People and WorldCover-style land-cover fractions derived deterministically from canopy/impervious."""
    f = city.frame
    built = np.clip(f["impervious"].to_numpy(float) / 100.0 * 0.9, 0.0, 1.0)
    water = (f["water_dist"].to_numpy(float) < 0.5 * float(city.truth["dx"])).astype(float)
    tree = np.minimum(np.clip(f["canopy"].to_numpy(float) / 100.0, 0.0, 1.0), 1.0 - built) * (1.0 - water)
    built = built * (1.0 - water)
    rest = np.clip(1.0 - built - tree - water, 0.0, 1.0)
    elev = f["elevation"].to_numpy(float)
    old = 0.12 + 0.10 * _sigmoid((elev - np.median(elev)) / 5.0)        # older residents on higher ground
    people = 22.0 * built + 2.0 * rest * (1.0 - water)
    out = {"id": ids, "x_m": x_m, "y_m": y_m,
           "people": people, "people_60_plus": people * old, "people_under_5": people * 0.06,
           "lc_tree": tree, "lc_shrub": 0.15 * rest, "lc_grass": 0.7 * rest, "lc_crop": 0.0 * rest,
           "lc_built": built, "lc_bare": 0.15 * rest, "lc_water": water, "lc_wetland": 0.0 * water}
    df = pd.DataFrame(out)
    for c in df.columns:
        if c not in ("id", "x_m", "y_m"):
            df[c] = df[c].round(4)
    return df


def demo_cmip6(site_lat: float, site_lon: float) -> pd.DataFrame:
    """Six pseudo-models × four SSPs × three periods in the ``cmip6_change_factors`` schema."""
    rows = []
    for i, (model, sens) in enumerate(DEMO_MODELS.items()):
        base = 300.0 + 0.4 * (i - 2.5)
        for exp, periods in DEMO_WARMING_K.items():
            for period, w in periods.items():
                d = round(w * sens, 4)
                rows.append({"site_lat": round(site_lat, 4), "site_lon": round(site_lon, 4), "model": model,
                             "member": "r1i1p1f1", "experiment": exp, "period": period,
                             "baseline_K": round(base, 4), "future_K": round(base + d, 4), "delta_K": d,
                             "land_weighted": True, "months": "6-7-8", "variable": "tasmax",
                             "baseline": "1995-2014"})
    return pd.DataFrame(rows)


def write_demo_project(out_dir, n: int = 96, seed: int = 0) -> dict:
    """Write the synthetic demo project into ``out_dir`` (SPEC §9.2).

    Files: ``data/city.csv`` (EPSG:32619 coordinates offset to a fictional
    site), ``data/city.truth.json`` (the generator's truths plus
    ``derived.true_scenarios`` from :meth:`SyntheticCity.true_response` for
    every configured canopy scenario), ``inputs/layers/demo_layers.parquet``,
    ``inputs/climate/demo_cmip6.csv`` and ``config.yml`` (paths relative to
    ``out_dir``).  Deterministic in ``(n, seed)``: the same bytes for the
    CSV, parquet and YAML files.  Returns ``{config_path, files, truth}``.
    """
    import yaml

    from sparc.core import runio
    from sparc.core.config import core_config_from_dict
    from sparc.core.scenarios import specs_from_config

    out = Path(out_dir)
    city = make_synthetic_city(n=n, seed=seed)
    f = city.frame
    ids = np.arange(len(f), dtype=np.int64)
    x = f["x"].to_numpy(float) + DEMO_OFFSET_M[0]
    y = f["y"].to_numpy(float) + DEMO_OFFSET_M[1]
    table = pd.DataFrame({"id": ids, "x": x, "y": y})
    for c, nd in _DEMO_DECIMALS.items():
        table[c] = f[c].to_numpy(float).round(nd)
    files = {"data": out / "data" / "city.csv", "truth": out / "data" / "city.truth.json",
             "layers": out / "inputs" / "layers" / "demo_layers.parquet",
             "climate": out / "inputs" / "climate" / "demo_cmip6.csv", "config": out / "config.yml"}
    runio.write_text_atomic(files["data"], table.to_csv(index=False, lineterminator="\n"))
    runio.write_parquet_atomic(demo_layers(city, ids, x, y), files["layers"])

    from pyproj import Transformer

    lon, lat = Transformer.from_crs(DEMO_CRS, "EPSG:4326", always_xy=True).transform(float(np.mean(x)),
                                                                                      float(np.mean(y)))
    runio.write_text_atomic(files["climate"], demo_cmip6(float(lat), float(lon)).to_csv(index=False,
                                                                                         lineterminator="\n"))
    raw = demo_config()
    runio.write_text_atomic(files["config"], yaml.safe_dump({"core": raw}, sort_keys=False, allow_unicode=True))

    cfg = core_config_from_dict(raw, base_dir=out)
    true_scenarios = {}
    for spec in specs_from_config(cfg):
        if len(spec.interventions) == 1 and spec.interventions[0].variable == "canopy":
            true_scenarios[spec.name] = float(np.mean(city.true_response(float(spec.interventions[0].amount))))
    derived = {"true_scenarios": true_scenarios, "true_footprint_mean": float(np.mean(city.true_footprint())),
               "n": int(n), "seed": int(seed)}
    truth = {**city.truth, "v": [float(c) for c in city.truth["v"]], "derived": derived}
    runio.write_json_atomic(files["truth"], truth)
    return {"config_path": files["config"], "files": files, "truth": truth}
