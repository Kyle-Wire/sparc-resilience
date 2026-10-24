"""Phase 3 batch driver (docs/agent/ROADMAP.md): every city from setup to estimate, one status row each.

    python scripts/run_canopy_cities.py burlington_vt providence_ri albuquerque_nm
    python scripts/run_canopy_cities.py --all            # every pilot city with a catalog entry

Per city, each step a subprocess (a failure is logged in the city's row; the batch never aborts):

1. ``scripts/setup_core_city.py <slug>`` (skipped when ``configs/core_<slug>.yml`` exists, or the city has a
   hand-made config in ``--config``);
2. ``identify lab`` (idealised campaigns, 12 replicates);
3. ``identify estimate --window all``: per-run block-scale readings and each run's wind;
4. ``identify lab --routes --window <run> --reps 36`` for every run: the real-route verdict that decides pool
   entry (owner decisions 2 and 3, 2026-10-24: a run enters the pool only if street_100 is Trustworthy on
   its own routes; always 36 replicates);
5. ``identify kmlab`` under the runs' winds, then the final ``estimate``.

Steps whose outputs exist are skipped unless ``--force``.  The table goes to
``output/core/pooled/city_status.csv`` and ``city_status.md``; commit a summary to
``docs/results/canopy_pooled.md`` (``output/`` does not persist).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd

ROUTE_REPS = 36          # owner decision 3: fixed in advance for every city and run
HAND_CONFIGS = {"providence_ri": ("configs/core_providence.yml", "wu9v7")}   # slug → (hand-made config, traverse node)


# --------------------------------------------------------------------------- #
# Status rows (pure; tested in tests/core/test_run_canopy_cities.py)          #
# --------------------------------------------------------------------------- #
def route_tag(run: str, window: str, windows: list[str]) -> str:
    """The ``lab --window`` argument for a run: its window when that names one run, else the run itself."""
    return window if windows.count(window) == 1 else run


def run_rows(slug: str, estimate: dict, route_labs: dict[str, dict]) -> list[dict]:
    """One row per run: the block-scale reading, its real-route verdict and whether it enters the pool."""
    runs = [r for r in estimate.get("runs", []) if r.get("street")]
    windows = [r["window"] for r in runs]
    out = []
    for r in runs:
        tag = route_tag(r["run"], r["window"], windows)
        lab = route_labs.get(tag) or {}
        d = (lab.get("designs") or {}).get("street_100") or {}
        verdict = (d.get("verdict") or {}).get("status")
        reps = min((lab.get("n_reps") or {}).values(), default=0)
        s100 = r["street"].get("street_100") or {}
        s300 = r["street"].get("street_300") or {}
        out.append({"city": slug, "run": r["run"], "window": r["window"], "visits": r.get("qa", {}).get("n_samples"),
                    "street_100": s100.get("estimate"), "street_100_se": s100.get("se"),
                    "street_100_lo": s100.get("lo"), "street_100_hi": s100.get("hi"),
                    "street_300": s300.get("estimate"), "street_300_se": s300.get("se"),
                    "route_verdict": verdict or "not run", "route_reps": reps,
                    "floor_valid": bool((d.get("floor") or {}).get("valid", False)),
                    "pooled": bool(verdict == "trustworthy" and reps >= ROUTE_REPS)})
    return out


def city_row(slug: str, steps: dict, lab: dict | None, estimate: dict | None, rows: list[dict]) -> dict:
    """The city's summary: where it stopped, the idealised lab's verdict, the pooled estimate, pooled runs."""
    failed = [k for k, v in steps.items() if v.get("status") == "failed"]
    d = ((lab or {}).get("designs") or {}).get("street_100") or {}
    h = ((estimate or {}).get("headline") or {}).get("street_100") or {}
    km = (estimate or {}).get("kilometre") or {}
    return {"city": slug, "status": f"failed at {failed[0]}" if failed else "ok",
            "error": steps[failed[0]].get("error", "") if failed else "",
            "lab_street_100": (d.get("verdict") or {}).get("status"),
            "all_runs_street_100": h.get("estimate"), "all_runs_lo": h.get("lo"), "all_runs_hi": h.get("hi"),
            "runs": len(rows), "runs_pooled": sum(r["pooled"] for r in rows),
            "wind_shift": (km.get("result") or {}).get("estimate"), "wind_shift_lab": km.get("lab_status"),
            "minutes": round(sum(v.get("seconds", 0.0) for v in steps.values()) / 60.0, 1)}


def status_markdown(cities: list[dict], runs: list[dict]) -> str:
    def f(v, nd=3):
        return "—" if v is None or (isinstance(v, float) and v != v) else (f"{v:+.{nd}f}" if isinstance(v, float) else str(v))

    lines = ["# Canopy cities: status", "", "| city | status | idealised lab (street_100) | all runs, within 100 m (95% CI) "
             "| runs pooled | minutes |", "|---|---|---|---|---|---|"]
    for c in cities:
        ci = f"{f(c['all_runs_street_100'])} ({f(c['all_runs_lo'])} to {f(c['all_runs_hi'])})" \
            if c["all_runs_street_100"] is not None else "—"
        lines.append(f"| {c['city']} | {c['status']} | {c['lab_street_100'] or '—'} | {ci} | "
                     f"{c['runs_pooled']}/{c['runs']} | {c['minutes']} |")
    lines += ["", "| city | run | visits | within 100 m (SE) | real-route verdict (reps) | pooled |", "|---|---|---|---|---|---|"]
    for r in runs:
        se = "—" if r["street_100_se"] is None or r["street_100_se"] != r["street_100_se"] else f"{r['street_100_se']:.3f}"
        lines.append(f"| {r['city']} | {r['run']} | {r['visits'] or '—'} | {f(r['street_100'])} ({se}) | "
                     f"{r['route_verdict']} ({r['route_reps']}) | {'yes' if r['pooled'] else 'no'} |")
    return "\n".join(lines) + "\n"


def previous_status(pooled: Path, replacing: list[str]) -> tuple[list[dict], list[dict]]:
    """Rows already in ``city_status.csv`` / ``run_status.csv`` for cities not in this batch."""
    out = []
    for name in ("city_status.csv", "run_status.csv"):
        p = pooled / name
        rows = []
        if p.is_file() and p.stat().st_size:
            df = pd.read_csv(p)
            rows = df[~df["city"].isin(replacing)].astype(object).where(df.notna(), None).to_dict("records")
        out.append(rows)
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Running                                                                      #
# --------------------------------------------------------------------------- #
def _sh(cmd: list[str], log: Path, timeout_h: float) -> dict:
    t = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            p = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_h * 3600)
            ok = p.returncode == 0
            err = "" if ok else f"exit {p.returncode} (see {log.name})"
        except subprocess.TimeoutExpired:
            ok, err = False, f"timed out after {timeout_h} h"
    return {"status": "ok" if ok else "failed", "seconds": round(time.time() - t, 1), **({"error": err} if err else {})}


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def run_city(slug: str, args, py=sys.executable) -> tuple[dict, list[dict]]:
    from sparc.core.config import load_core_config

    hand = HAND_CONFIGS.get(slug)
    cfg_path = REPO / (hand[0] if hand else f"configs/core_{slug}.yml")
    steps: dict = {}
    pooled = REPO / "output" / "core" / "pooled"
    log = pooled / f"{slug}.log"
    ident = [py, "-m", "sparc.core.identify"]
    if not hand and (not cfg_path.is_file() or args.force):
        steps["setup"] = _sh([py, "scripts/setup_core_city.py", slug, "--skip-layers"]
                             + (["--force"] if args.force else []), log, args.timeout_h)
        if steps["setup"]["status"] != "ok" or not cfg_path.is_file():
            steps["setup"]["status"] = "failed"
            return city_row(slug, steps, None, None, []), []
    cfg = load_core_config(cfg_path)
    out = cfg.output_dir / "identify"
    raw = cfg.output_dir / "identify" / "raw"
    if hand and not raw.is_dir():
        from sparc.data.collect.capa import fetch_capa_raw

        try:
            fetch_capa_raw(hand[1], raw)
        except Exception as exc:  # noqa: BLE001 - recorded in the row
            steps["fetch"] = {"status": "failed", "seconds": 0.0, "error": str(exc)[:300]}
            return city_row(slug, steps, None, None, []), []
    p = ["-p", str(cfg_path)]
    if args.force or not (out / "identify_lab.json").is_file():
        steps["lab"] = _sh(ident + ["lab", *p, "--workers", str(args.workers)], log, args.timeout_h)
    steps["estimate"] = _sh(ident + ["estimate", *p, "--traverses", str(raw), "--window", "all"], log, args.timeout_h)
    est = _load(out / "identify_estimate.json")
    if est is None:
        return city_row(slug, steps, _load(out / "identify_lab.json"), None, []), []
    runs = [r for r in est.get("runs", []) if r.get("street")]
    windows = [r["window"] for r in runs]
    labs = {}
    for r in runs:
        tag = route_tag(r["run"], r["window"], windows)
        f = out / f"identify_lab_routes_{tag.replace(' ', '_')}.json"
        have = _load(f)
        if args.force or not have or min((have.get("n_reps") or {}).values(), default=0) < ROUTE_REPS:
            steps[f"routes {tag}"] = _sh(ident + ["lab", *p, "--workers", str(args.workers), "--reps", str(ROUTE_REPS),
                                                  "--routes", str(raw), "--window", tag], log, args.timeout_h)
        labs[tag] = _load(f) or {}
    winds = {r: w for r, w in ((est.get("kilometre") or {}).get("winds") or {}).items() if "from_deg" in w}
    if len(winds) >= 2:
        if args.force or not (out / "identify_kmlab.json").is_file():
            steps["kmlab"] = _sh(ident + ["kmlab", *p, "--workers", str(args.workers), "--winds",
                                          *[f"{w['from_deg']:.0f}:{w['speed']:.1f}" for w in winds.values()]],
                                 log, args.timeout_h)
        steps["estimate (final)"] = _sh(ident + ["estimate", *p, "--traverses", str(raw), "--window", "all"], log,
                                        args.timeout_h)
        est = _load(out / "identify_estimate.json") or est
    rows = run_rows(slug, est, labs)
    return city_row(slug, steps, _load(out / "identify_lab.json"), est, rows), rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cities", nargs="*", help="city slugs (configs/multicity_pilot.yml)")
    ap.add_argument("--all", action="store_true", help="every pilot city with a CAPA catalog entry")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    ap.add_argument("--timeout-h", type=float, default=6.0, help="per step (default 6 h)")
    args = ap.parse_args(argv)
    slugs = list(args.cities)
    if args.all:
        import yaml

        from sparc.data.collect.capa_catalog import CAPA_CATALOG

        def walk(o):
            if isinstance(o, dict):
                if "city_slug" in o:
                    yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)

        pilot = yaml.safe_load((REPO / "configs" / "multicity_pilot.yml").read_text(encoding="utf-8"))
        slugs += [c["city_slug"] for c in walk(pilot) if c.get("city") in CAPA_CATALOG and c["city_slug"] not in slugs]
    if not slugs:
        ap.error("name cities or pass --all")
    pooled = REPO / "output" / "core" / "pooled"
    pooled.mkdir(parents=True, exist_ok=True)
    cities, runs = previous_status(pooled, slugs)          # the table is cumulative: other cities' rows are kept
    for slug in slugs:
        try:
            c, rr = run_city(slug, args)
        except Exception as exc:  # noqa: BLE001 - one city never stops the batch
            c, rr = city_row(slug, {"driver": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}},
                             None, None, []), []
        cities.append(c)
        runs.extend(rr)
        pd.DataFrame(cities).to_csv(pooled / "city_status.csv", index=False)
        pd.DataFrame(runs).to_csv(pooled / "run_status.csv", index=False)
        (pooled / "city_status.md").write_text(status_markdown(cities, runs), encoding="utf-8")
        print(f"{slug}: {c['status']} ({c['runs_pooled']}/{c['runs']} runs pooled)", flush=True)
    print(status_markdown(cities, runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
