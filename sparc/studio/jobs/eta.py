"""Cost model, ETA and calibration (SPEC §5.4).

::

    seconds(unit) = rate(host, unit) × (n_cells / 54_701)^α(unit) × (4 / threads)^0.7
    α = 1.3 for base_fit:mgwr, 1.15 for base_fit:gwrf, 1.0 otherwise;
    network units (climate_model, remote_object) and study units (replicate:*, variant:*) are not scaled.

``rate(host, unit)`` is the median of the last 20 observations of the unit
on this host (``unit_timings``, normalised back to the reference 54,701
cells and 4 threads), else the **seed rate** measured on full Providence
with 4 cores (:data:`SEED_RATES`).  Inside a job, the in-job mean of a unit
replaces the prior after its first completion, each cv_curve partition is
estimated at 0.95 × the observed S2_S3 seconds, and the first engine pass
sets the rate for the remaining passes.  Ranges are p25–p75 of the rate
history propagated as a sum, or ±25% with fewer than three observations.

**Progress weights** are the seed rates alone (:func:`unit_weight`; no host
calibration, no n-scaling), so the Python projection and the web client's
reducer compute identical progress; ``/api/meta.unit_costs`` serves them.
"""

from __future__ import annotations

import hashlib
import os
import platform
import statistics
import time
from typing import Any, Iterable, Mapping

__all__ = [
    "REF_CELLS", "REF_THREADS", "SEED_RATES", "unit_costs", "unit_weight", "seed_rate", "alpha", "scale",
    "host_id", "CostModel", "estimate_units", "estimate_nodes", "projection_eta", "peak_ram_gb",
    "checkpoint_bytes", "RANGE_FRAC",
]

REF_CELLS = 54_701
REF_THREADS = 4
RANGE_FRAC = 0.25
HISTORY_N = 20

#: seed rates in seconds per unit, full Providence on 4 cores (SPEC §5.4).  ``*`` entries match a prefix.
SEED_RATES: dict[str, float] = {
    "base_fit:mgwr": 170.0,
    "base_fit:gwrf": 23.0,
    "base_fit:gam": 6.0,
    "base_fit:physics": 4.0,
    "base_fit:ols": 0.1,
    "adv_refit": 4.5,
    "stacker_fit:mean": 0.2,
    "stacker_fit:nnls": 0.2,
    "stacker_fit:residual": 14.0,
    "baseline_fit:*": 12.0,
    "engine_pass": 13.0,
    "causal_step:dml": 10.0,
    "causal_step:spillover": 15.0,
    "causal_step:cate": 1.0,
    "causal_step:dr": 22.0,
    "causal_step:sens": 0.5,
    "causal_step:audit": 0.5,
    "s0_load": 0.3,
    "s1_influence": 3.3,
    # 2 s per 500 MB of checkpoint; a full-Providence checkpoint is 9.6 kB × 54,701 cells ≈ 525 MB
    "checkpoint_save": 2.1,
    "pareto": 0.5,
    "climate_model": 20.0,
    "remote_object": 5.0,
    "replicate:*": 254.0,
    "variant:*": 1200.0,
    "unpickle": 26.25,          # 1 s per 20 MB of a ≈525 MB checkpoint
    "mediator_fit": 1.0,
}
DEFAULT_RATE = 1.0
_ALPHA = {"base_fit:mgwr": 1.3, "base_fit:gwrf": 1.15}
_UNSCALED_PREFIXES = ("climate_model", "remote_object", "replicate:", "variant:")


def unit_costs() -> dict[str, float]:
    """The seed table as served by ``/api/meta.unit_costs`` (progress weights and ETA priors)."""
    return dict(SEED_RATES)


def seed_rate(unit: str) -> float:
    """Seed seconds of ``unit``: an exact entry, else the ``<prefix>:*`` entry, else 1 s."""
    if unit in SEED_RATES:
        return SEED_RATES[unit]
    if ":" in unit:
        wild = unit.split(":", 1)[0] + ":*"
        if wild in SEED_RATES:
            return SEED_RATES[wild]
    return DEFAULT_RATE


def unit_weight(unit: str) -> float:
    """Progress weight of one ``unit`` = its seed rate (identical in the Python and TS reducers)."""
    return seed_rate(unit)


def alpha(unit: str) -> float:
    return _ALPHA.get(unit, 1.0)


def _scaled(unit: str) -> bool:
    return not unit.startswith(_UNSCALED_PREFIXES)


def scale(unit: str, n_cells: float | None, threads: float | None) -> float:
    """``(n_cells / 54,701)^α × (4 / threads)^0.7`` (1 for network and study units)."""
    if not _scaled(unit):
        return 1.0
    n = float(n_cells) if n_cells else REF_CELLS
    t = float(threads) if threads else REF_THREADS
    return (n / REF_CELLS) ** alpha(unit) * (REF_THREADS / max(t, 1.0)) ** 0.7


_HOST_ID: str | None = None


def host_id() -> str:
    """``sha1(cpu model + cpu count + total RAM)[:12]`` - stable per machine."""
    global _HOST_ID
    if _HOST_ID is None:
        model = _cpu_model()
        cpu = os.cpu_count() or 1
        try:
            import psutil

            mem = psutil.virtual_memory().total
        except Exception:  # pragma: no cover
            mem = 0
        _HOST_ID = hashlib.sha1(f"{model}|{cpu}|{mem}".encode()).hexdigest()[:12]
    return _HOST_ID


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class CostModel:
    """Per-host unit rates: medians of normalised history, seed rates otherwise.

    ``history`` maps a unit to its recent normalised rates (seconds at
    54,701 cells and 4 threads), newest last; :meth:`from_db` loads the last
    20 per unit for this host from ``unit_timings``.
    """

    def __init__(self, history: Mapping[str, Iterable[float]] | None = None):
        self.history = {u: [float(x) for x in list(v)[-HISTORY_N:] if x is not None and x > 0]
                        for u, v in (history or {}).items()}

    @classmethod
    def from_db(cls, db, host: str | None = None) -> "CostModel":
        host = host or host_id()
        rows = db.fetchall("SELECT unit, seconds, n_cells, threads FROM unit_timings WHERE host_id = ? "
                           "ORDER BY ts", (host,))
        hist: dict[str, list[float]] = {}
        for r in rows:
            if not r["seconds"] or r["seconds"] <= 0:
                continue
            s = scale(r["unit"], r["n_cells"], r["threads"])
            hist.setdefault(r["unit"], []).append(r["seconds"] / s if s > 0 else r["seconds"])
        return cls(hist)

    def rate(self, unit: str) -> float:
        h = self.history.get(unit)
        return statistics.median(h) if h else seed_rate(unit)

    def rate_range(self, unit: str) -> tuple[float, float]:
        h = self.history.get(unit)
        if h and len(h) >= 3:
            return _quantile(h, 0.25), _quantile(h, 0.75)
        r = self.rate(unit)
        return r * (1 - RANGE_FRAC), r * (1 + RANGE_FRAC)

    def seconds(self, unit: str, n_cells: float | None = None, threads: float | None = None) -> float:
        return self.rate(unit) * scale(unit, n_cells, threads)

    def seconds_range(self, unit: str, n_cells: float | None = None,
                      threads: float | None = None) -> tuple[float, float]:
        lo, hi = self.rate_range(unit)
        s = scale(unit, n_cells, threads)
        return lo * s, hi * s


def estimate_units(units: Mapping[str, float], n_cells: float | None = None, threads: float | None = None,
                   model: CostModel | None = None) -> tuple[float, float, float]:
    """``(est_s, lo, hi)`` for a ``{unit: count}`` table."""
    model = model or CostModel()
    est = lo = hi = 0.0
    for u, n in units.items():
        if not n:
            continue
        est += n * model.seconds(u, n_cells, threads)
        a, b = model.seconds_range(u, n_cells, threads)
        lo += n * a
        hi += n * b
    return est, lo, hi


def estimate_nodes(nodes: list[dict], n_cells: float | None = None, threads: float | None = None,
                   model: CostModel | None = None) -> list[dict]:
    """Plan nodes with ``est_s``/``est_lo``/``est_hi`` filled (0 for skipped and cached nodes)."""
    model = model or CostModel()
    out = []
    for node in nodes:
        node = dict(node)
        if node.get("state", "will_run") != "will_run":
            node["est_s"] = node["est_lo"] = node["est_hi"] = 0.0
        else:
            est, lo, hi = estimate_units(node.get("units") or {}, n_cells, threads, model)
            node["est_s"], node["est_lo"], node["est_hi"] = round(est, 3), round(lo, 3), round(hi, 3)
        out.append(node)
    return out


def peak_ram_gb(n_cells: float, k_folds: int = 5) -> float:
    """Launch estimate of peak RAM: 28 kB × n_cells × K^0.5 + 0.75 GB.

    Calibrated on measured peaks (process tree RSS) of Providence core runs: the fast run (8,037 cells,
    3 folds) peaked at 1.09 GB and the 60 m run (13,945 cells, 5 folds) at 1.56 GB; this gives 1.14 and
    1.62 GB, and about 4.2 GB for the full 30 m run (54,701 cells, 5 folds).  Core runs use threads, not
    processes, so the thread count barely moves it.  The preflight adds a 1 GB margin."""
    return 28e3 * float(n_cells) * max(int(k_folds), 1) ** 0.5 / 1e9 + 0.75


def checkpoint_bytes(n_cells: float) -> int:
    """Launch estimate of the checkpoint size: 9.6 kB × n_cells."""
    return int(9.6e3 * float(n_cells))


# ---------------------------------------------------------------------------
# live ETA of a projection
# ---------------------------------------------------------------------------

def _in_job(state: dict, unit: str) -> list[float]:
    return [x for x in (state.get("unit_obs") or {}).get(unit, []) if x and x > 0]


def _unit_seconds(state: dict, unit: str, model: CostModel, n_cells, threads) -> tuple[float, float, float]:
    obs = _in_job(state, unit)
    if obs:
        mean = sum(obs) / len(obs)
        if len(obs) >= 3:
            return mean, _quantile(obs, 0.25), _quantile(obs, 0.75)
        lo, hi = model.seconds_range(unit, n_cells, threads)
        base = model.seconds(unit, n_cells, threads)
        # keep the prior's relative spread around the in-job mean
        return mean, mean * (lo / base if base else 1 - RANGE_FRAC), mean * (hi / base if base else 1 + RANGE_FRAC)
    est = model.seconds(unit, n_cells, threads)
    lo, hi = model.seconds_range(unit, n_cells, threads)
    return est, lo, hi


def _remaining(node_units: Mapping[str, float], done: Mapping[str, float], partial: Mapping[str, float]) -> dict:
    return {u: max(0.0, float(n) - float(done.get(u, 0)) - float(partial.get(u, 0)))
            for u, n in node_units.items()}


def projection_eta(state: dict, model: CostModel | None = None, *, n_cells: float | None = None,
                   threads: float | None = None, now: float | None = None) -> dict:
    """Remaining seconds of a tracker projection: ``{eta_s, eta_lo, eta_hi, stages: {id: est_s}}``.

    Uses the plan's will-run nodes minus the completed and fractional units
    of each stage (SPEC §5.4 unit accounting).  Without a plan it
    extrapolates elapsed time from progress (``elapsed × (1 − p) / p``,
    ±25%) once progress passes 2%.  Finished projections have ``eta_s = 0``.
    """
    model = model or CostModel()
    n_cells = n_cells or state.get("n_points")
    out: dict[str, Any] = {"eta_s": None, "eta_lo": None, "eta_hi": None, "stages": {}}
    if state.get("finished"):
        out.update(eta_s=0.0, eta_lo=0.0, eta_hi=0.0)
        return out
    plan = state.get("plan")
    stage_done = state.get("stage_done") or {}
    stage_partial = state.get("stage_partial") or {}
    if plan:
        total = lo_t = hi_t = 0.0
        s23 = next((n for n in plan if n.get("id") == "S2_S3"), None)
        s23_elapsed = (state.get("stage_elapsed") or {}).get("S2_S3")
        for node in plan:
            if node.get("state", "will_run") != "will_run":
                continue
            sid = node.get("id")
            st = (state.get("stages") or {}).get(sid) or {}
            if st.get("state") in ("done", "failed", "cancelled", "skipped", "cached", "disabled", "not_requested"):
                out["stages"][sid] = 0.0
                continue
            units = node.get("units") or {}
            rem = _remaining(units, stage_done.get(sid, {}), stage_partial.get(sid, {}))
            if sid == "cv_curve" and s23 is not None and s23_elapsed:
                # each partition ≈ 0.95 × the observed S2_S3 seconds
                ref = {u: n for u, n in (s23.get("units") or {}).items() if u != "checkpoint_save"}
                common = [u for u in ref if u in units and ref[u]]
                parts = (units[common[0]] / ref[common[0]]) if common else 1.0
                w_tot = sum(unit_weight(u) * n for u, n in units.items())
                w_rem = sum(unit_weight(u) * n for u, n in rem.items())
                frac = (w_rem / w_tot) if w_tot else 0.0
                est = 0.95 * s23_elapsed * parts * frac
                total += est
                lo_t += est * (1 - RANGE_FRAC)
                hi_t += est * (1 + RANGE_FRAC)
                out["stages"][sid] = round(est, 3)
                continue
            est = lo = hi = 0.0
            for u, n in rem.items():
                if n <= 0:
                    continue
                s, a, b = _unit_seconds(state, u, model, n_cells, threads)
                est += n * s
                lo += n * a
                hi += n * b
            total += est
            lo_t += lo
            hi_t += hi
            out["stages"][sid] = round(est, 3)
        out.update(eta_s=round(total, 3), eta_lo=round(lo_t, 3), eta_hi=round(hi_t, 3))
        return out
    p = state.get("progress")
    t0 = state.get("first_ts")
    if p is not None and t0 is not None and p >= 0.02:
        elapsed = max(0.0, (now if now is not None else time.time()) - t0)
        if p >= 1.0:
            out.update(eta_s=0.0, eta_lo=0.0, eta_hi=0.0)
        else:
            est = elapsed * (1 - p) / p
            out.update(eta_s=round(est, 3), eta_lo=round(est * (1 - RANGE_FRAC), 3),
                       eta_hi=round(est * (1 + RANGE_FRAC), 3))
    return out
