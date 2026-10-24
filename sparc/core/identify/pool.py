"""Pooling cities: a random-effects model of canopy's block-scale effect across campaigns (Phase 3).

Every city run whose block-scale design passed its own real-route lab (the pool-entry rule) contributes one
estimate ``y_i`` with standard error ``s_i`` (+10 pp canopy within 100 m, °F).  The model is

    y_i = mu + u_i + e_i,    u_i ~ N(0, tau²),   e_i ~ N(0, s_i²)

* ``mu``: the pooled effect; ``tau²``: how much the true effect differs between cities (REML, with the
  DerSimonian–Laird moment estimate as the start and the fallback);
* the **prediction interval** for a new city, ``mu ± t_{k-2} sqrt(tau² + se(mu)²)`` (Higgins, Thompson &
  Spiegelhalter 2009), is what a planner in a city without traverses would use;
* **leave-one-city-out**: each city is predicted from the others and scored on whether its own estimate
  falls inside the prediction interval widened by its own sampling error (the Phase 3 gate);
* **Stouffer's** combination of the cities' one-sided z values for the kilometre (wind-shift) test.

Pooling is per time window (morning, midday, evening, night): canopy's effect differs through the day.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def dersimonian_laird(y, se) -> float:
    y, w = np.asarray(y, float), 1.0 / np.asarray(se, float) ** 2
    mu = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return max(0.0, (q - (y.size - 1)) / c) if c > 0 else 0.0


def reml_tau2(y, se, start: float | None = None, tol: float = 1e-10, max_iter: int = 200) -> float | None:
    """REML estimate of tau² by Fisher scoring; ``None`` when it does not converge."""
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    t2 = dersimonian_laird(y, np.sqrt(v)) if start is None else start
    for _ in range(max_iter):
        w = 1.0 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        # REML score and information for tau² (e.g. Viechtbauer 2005)
        p_trace = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        score = 0.5 * (np.sum(w ** 2 * (y - mu) ** 2) - p_trace)
        info = 0.5 * (np.sum(w ** 2) - 2 * np.sum(w ** 3) / np.sum(w) + (np.sum(w ** 2) / np.sum(w)) ** 2)
        if not np.isfinite(score) or info <= 0:
            return None
        new = max(0.0, t2 + score / info)
        if abs(new - t2) < tol * max(1.0, t2):
            return new
        t2 = new
    return None


def random_effects(y, se, method: str = "reml") -> dict:
    """Pooled effect, tau², I², and the 95% confidence and prediction intervals."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    k = int(y.size)
    if k == 0:
        raise ValueError("nothing to pool")
    used = "dl"
    t2 = dersimonian_laird(y, se) if k > 1 else 0.0
    if method == "reml" and k > 2:
        r = reml_tau2(y, se, start=t2)
        if r is not None:
            t2, used = r, "reml"
    w = 1.0 / (se ** 2 + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    se_mu = float(math.sqrt(1.0 / np.sum(w)))
    w0 = 1.0 / se ** 2
    q = float(np.sum(w0 * (y - np.sum(w0 * y) / np.sum(w0)) ** 2))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    out = {"k": k, "mu": mu, "se": se_mu, "lo": mu - 1.959964 * se_mu, "hi": mu + 1.959964 * se_mu,
           "tau2": float(t2), "tau": float(math.sqrt(t2)), "i2": float(i2), "q": q, "method": used}
    if k >= 3:
        half = stats.t.ppf(0.975, k - 2) * math.sqrt(t2 + se_mu ** 2)
        out.update(pi_lo=mu - half, pi_hi=mu + half)
    return out


def loco(y, se, labels=None, method: str = "reml") -> dict:
    """Leave-one-city-out: predict each city from the others; does its own estimate fall inside the
    prediction interval widened by its own sampling error?  Needs at least 4 cities (3 to fit)."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    labels = list(labels) if labels is not None else [str(i) for i in range(y.size)]
    rows = []
    for i in range(y.size):
        keep = np.arange(y.size) != i
        if keep.sum() < 3:
            continue
        fit = random_effects(y[keep], se[keep], method)
        half = stats.t.ppf(0.975, fit["k"] - 2) * math.sqrt(fit["tau2"] + fit["se"] ** 2 + se[i] ** 2)
        rows.append({"city": labels[i], "estimate": float(y[i]), "se": float(se[i]), "predicted": fit["mu"],
                     "lo": fit["mu"] - half, "hi": fit["mu"] + half, "covered": bool(abs(y[i] - fit["mu"]) <= half),
                     "abs_error": float(abs(y[i] - fit["mu"]))})
    cov = float(np.mean([r["covered"] for r in rows])) if rows else None
    return {"rows": rows, "coverage": cov, "mae": float(np.mean([r["abs_error"] for r in rows])) if rows else None}


def leave_one_out(y, se, labels, method: str = "reml") -> list[dict]:
    """The pooled estimate refitted with each city left out in turn (owner decision 12): does the pooled
    result rest on one city?"""
    y, se = np.asarray(y, float), np.asarray(se, float)
    out = []
    for i in range(y.size):
        keep = np.arange(y.size) != i
        if not keep.any():
            continue
        f = random_effects(y[keep], se[keep], method)
        out.append({"left_out": labels[i], "k": f["k"], "mu": f["mu"], "lo": f["lo"], "hi": f["hi"], "tau": f["tau"]})
    return out


def stouffer(z) -> dict:
    """Combined evidence from the cities' z values (unweighted Stouffer); negative z is cooling, so
    ``p_cooling`` is the lower-tail p-value."""
    z = np.asarray(z, float)
    zc = float(np.sum(z) / math.sqrt(z.size))
    return {"k": int(z.size), "z": zc, "p_cooling": float(stats.norm.cdf(zc))}


def pool_runs(rows: list[dict], estimate: str = "street_100", exclude=()) -> dict:
    """Per time window: random-effects pooling and leave-one-city-out over the runs that entered the pool
    (``rows`` as ``scripts/run_canopy_cities.py`` writes them to ``run_status.csv``).  ``street_100`` uses the
    ``pooled`` flag; ``street_300`` its own ``pooled_300`` flag (agent decision A4)."""
    flag = "pooled" if estimate == "street_100" else f"pooled_{estimate.rsplit('_', 1)[-1]}"
    out: dict = {"estimate": estimate, "excluded": list(exclude), "windows": {}}
    for win in sorted({r["window"] for r in rows}):
        rs = [r for r in rows if r["window"] == win and r.get(flag) in (True, "True", 1) and r["city"] not in exclude
              and r.get(estimate) is not None and r.get(f"{estimate}_se")]
        entry: dict = {"cities": [r["city"] for r in rs], "k": len(rs)}
        if rs:
            y = [float(r[estimate]) for r in rs]
            se = [float(r[f"{estimate}_se"]) for r in rs]
            entry["fit"] = random_effects(y, se)
            entry["loco"] = loco(y, se, [r["city"] for r in rs]) if len(rs) >= 4 else None
            entry["leave_one_out"] = leave_one_out(y, se, [r["city"] for r in rs]) if len(rs) >= 2 else []
        out["windows"][win] = entry
    return out


def pool_markdown(res: dict) -> str:
    reach = res.get("estimate", "street_100").rsplit("_", 1)[-1]
    lines = [f"# Pooled canopy effect (+10 pp within {reach} m, °F)", "",
             "| window | cities | pooled effect (95% CI) | tau | prediction interval, new city | LOCO coverage |",
             "|---|---|---|---|---|---|"]
    for win, e in res["windows"].items():
        f = e.get("fit")
        if not f:
            lines.append(f"| {win} | 0 | — | — | — | — |")
            continue
        pi = f"{f['pi_lo']:+.3f} to {f['pi_hi']:+.3f}" if "pi_lo" in f else "needs 3 cities"
        lc = e.get("loco")
        cov = f"{lc['coverage']:.0%} of {len(lc['rows'])}" if lc and lc["coverage"] is not None else "needs 4 cities"
        lines.append(f"| {win} | {f['k']} ({', '.join(e['cities'])}) | {f['mu']:+.3f} ({f['lo']:+.3f} to {f['hi']:+.3f}) | "
                     f"{f['tau']:.3f} | {pi} | {cov} |")
    sens = [(w, e) for w, e in res["windows"].items() if e.get("leave_one_out")]
    if sens:
        lines += ["", "## Each city left out in turn", "", "| window | left out | cities | pooled effect (95% CI) |",
                  "|---|---|---|---|"]
        for win, e in sens:
            f = e["fit"]
            lines.append(f"| {win} | none | {f['k']} | {f['mu']:+.3f} ({f['lo']:+.3f} to {f['hi']:+.3f}) |")
            for r in e["leave_one_out"]:
                lines.append(f"| {win} | {r['left_out']} | {r['k']} | {r['mu']:+.3f} ({r['lo']:+.3f} to {r['hi']:+.3f}) |")
    if res.get("excluded"):
        lines += ["", f"Excluded from every window: {', '.join(res['excluded'])}."]
    return "\n".join(lines) + "\n"
