"""``python -m sparc.core.identify`` — isolating the causal effect of canopy.

  inspect   what a downloaded campaign folder holds: files read or skipped, runs, vehicles, grid overlap
  lab       which designs recover a planted canopy effect on this city's layout (simulated campaigns)
  kmlab     the kilometre design (same streets, different winds): false alarms and power
  estimate  run the designs on a real campaign's traverse points (CSV / GeoJSON / shapefile / zip / folder)
  map       what the map itself says under each design (for comparison; the lab shows why not to trust it)
  simulate  write a simulated campaign as a CAPA-style CSV (to try ``estimate`` end to end)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _out(args, cfg) -> Path:
    if args.out:
        return Path(args.out)
    city = str(cfg.name).split("_")[0] or "project"
    return Path("output") / "core" / city / "identify"


def _load_lab(path: str | None, out: Path, name: str = "identify_lab.json") -> dict | None:
    p = Path(path) if path else out / name
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _local_tz(args) -> str:
    """The project's local time zone (its forcing file), else --timezone, else America/New_York."""
    if getattr(args, "timezone", None):
        return args.timezone
    if getattr(args, "project", None):
        from sparc.core.config import load_core_config

        cfg = load_core_config(args.project)
        f = (cfg.raw.get("physics") or {}).get("forcing")
        if f:
            p = Path(f) if Path(f).is_absolute() else Path(cfg.base_dir) / f
            if p.is_file():
                tz = json.loads(p.read_text(encoding="utf-8")).get("tz")
                if tz:
                    return tz
    return "America/New_York"


def cmd_inspect(args) -> int:
    from sparc.core.identify.traverses import inspect_markdown, scan_traverses

    pts, report = scan_traverses(args.traverses, timezone=args.timezone, utc_to=_local_tz(args) if args.utc else None)
    cfg = layout = None
    if args.project:
        from sparc.core.config import load_core_config
        from sparc.core.simcheck import load_layout

        cfg = load_core_config(args.project)
        layout = load_layout(cfg) if len(pts) else None
    print(inspect_markdown(pts, report, cfg, layout, show_all=args.all))
    return 0 if len(pts) else 1


def cmd_lab(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.identify.validate import GENERATORS, lab_markdown, run_lab, summarize

    cfg = load_core_config(args.project)
    out = _out(args, cfg)
    out.mkdir(parents=True, exist_ok=True)
    gens = tuple(args.generators) if args.generators else GENERATORS
    if not args.append:
        (out / "identify_lab.jsonl").unlink(missing_ok=True)
    run_lab(cfg, generators=gens, reps=args.reps, seed=args.seed, out_dir=out, workers=args.workers)
    rows = [json.loads(line) for line in (out / "identify_lab.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summ = summarize(rows)
    summ["config"] = str(args.project)
    (out / "identify_lab.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    md = lab_markdown(summ)
    (out / "identify_lab.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {out / 'identify_lab.json'} and identify_lab.md")
    return 0


def _parse_winds(items) -> list[tuple[float, float]]:
    out = []
    for w in items:
        d, spd = (str(w).split(":") + ["0"])[:2]
        out.append((float(d), float(spd)))
    return out


def cmd_kmlab(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.identify.kmlab import KM_WORLDS, SEA_BREEZE_DAY, km_markdown, run_km_lab, summarize_km

    cfg = load_core_config(args.project)
    out = _out(args, cfg)
    out.mkdir(parents=True, exist_ok=True)
    if not args.append:
        (out / "identify_kmlab.jsonl").unlink(missing_ok=True)
    winds = _parse_winds(args.winds) if args.winds else list(SEA_BREEZE_DAY)
    run_km_lab(cfg, worlds=tuple(args.worlds) if args.worlds else KM_WORLDS, reps=args.reps, winds=winds,
               seed=args.seed, out_dir=out, workers=args.workers)
    rows = [json.loads(x) for x in (out / "identify_kmlab.jsonl").read_text(encoding="utf-8").splitlines() if x]
    summ = summarize_km(rows)
    (out / "identify_kmlab.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    md = km_markdown(summ)
    (out / "identify_kmlab.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def cmd_estimate(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.identify.traverses import estimate, estimate_markdown

    cfg = load_core_config(args.project)
    out = _out(args, cfg)
    out.mkdir(parents=True, exist_ok=True)
    winds = {}
    for w in args.wind or []:
        try:
            key, val = w.split("=", 1)
            d, spd = (val.split(":") + ["0"])[:2]
            winds[key.strip()] = (float(d), float(spd))
        except ValueError:
            raise SystemExit(f"--wind {w!r}: use run=FROM_DEG:SPEED, e.g. midday=170:7.7") from None
    lab = _load_lab(args.lab, out) or {}
    km = _load_lab(None, out, "identify_kmlab.json") or {}
    lab = {"designs": {**lab.get("designs", {}), **km.get("designs", {})}}
    res = estimate(cfg, args.traverses, window=args.window, timezone=args.timezone, lab=lab, winds=winds,
                   fetch_winds=not args.no_fetch_winds, cache_dir=out, utc_to=_local_tz(args) if args.utc else None)
    stem = args.name or "identify_estimate"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    md = estimate_markdown(res)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out / (stem + '.json')} and {stem}.md")
    return 0


def cmd_map(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.identify.estimators import map_footprint, map_street
    from sparc.core.identify.layers import build_layers
    from sparc.core.simcheck import load_layout

    cfg = load_core_config(args.project)
    layout = load_layout(cfg)
    L = build_layers(layout)
    y = layout.data.target_raw
    rows = [map_footprint(y, L), map_street(y, L, layout.grid, reach=100.0), map_street(y, L, layout.grid, reach=300.0)]
    lab = _load_lab(args.lab, _out(args, cfg))
    print(f"# What the map says ({cfg.data['target']}, {y.size:,} cells)\n")
    for r in rows:
        v = ((lab or {}).get("designs", {}).get(r["estimator"]) or {}).get("verdict", {})
        print(f"- {r['label']}: {r['estimate']:+.3f} °F (95% CI {r['lo']:+.3f} to {r['hi']:+.3f})"
              + (f"  [lab: {v['status']}]" if v else ""))
    out = _out(args, cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identify_map.json").write_text(json.dumps({"target": cfg.data["target"], "designs": rows}, indent=2),
                                           encoding="utf-8")
    return 0


def cmd_simulate(args) -> int:
    import numpy as np

    from sparc.core.config import load_core_config
    from sparc.core.identify.campaign import simulate
    from sparc.core.identify.traverses import campaign_csv
    from sparc.core.identify.validate import plant, prepare, truth_for

    cfg = load_core_config(args.project)
    layout, L, features, real = prepare(cfg)
    rng = np.random.default_rng(args.seed)
    T, gen = plant(args.world, layout, real, rng)
    camp = simulate(layout, T, rng, features, product=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(campaign_csv(camp.samples, cfg, layout))
    truth = truth_for(args.world, gen, layout)
    print(f"wrote {args.out}: {len(camp.samples):,} samples, world '{args.world}'")
    print("true effect of +10 pp canopy at street points: "
          + ", ".join(f"{k.replace('_', ' ')} {v:+.3f} °F" for k, v in truth["street"].items()))
    return 0


def main(argv=None) -> int:
    from sparc.core.identify.validate import GENERATORS

    ap = argparse.ArgumentParser(prog="python -m sparc.core.identify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--project", "-p", required=True, help="core config (e.g. configs/core_providence.yml)")
        p.add_argument("--out", default=None, help="output folder (default output/core/<city>/identify)")

    p = sub.add_parser("inspect", help="list what a downloaded campaign folder holds before estimating")
    p.add_argument("--traverses", required=True, help="traverse file, archive or folder (e.g. the OSF download)")
    p.add_argument("--project", "-p", default=None, help="core config, to check the readings land on its grid")
    p.add_argument("--timezone", default=None, help="local time zone of tz-aware timestamps (e.g. America/New_York)")
    p.add_argument("--all", action="store_true", help="list every skipped file, not only the first 25")
    p.add_argument("--utc", action="store_true", help="timestamps are UTC without saying so: convert to local time")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("lab", help="validate every design on simulated campaigns over the real layout")
    common(p)
    p.add_argument("--reps", type=int, default=12, help="replicates per world (default 12)")
    p.add_argument("--workers", type=int, default=4, help="parallel processes (default 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--generators", nargs="*", choices=GENERATORS, help="worlds to simulate (default all)")
    p.add_argument("--append", action="store_true", help="add to an existing identify_lab.jsonl")
    p.set_defaults(func=cmd_lab)

    p = sub.add_parser("kmlab", help="false alarms and power of the kilometre (wind-shift) design")
    common(p)
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--winds", nargs="*", metavar="FROM:SPEED",
                   help="the day's runs, e.g. 290:2.0 170:7.7 210:4.0 (default: a sea-breeze day); use the winds "
                        "estimate reports for your campaign")
    p.add_argument("--worlds", nargs="*", help="worlds to simulate (default all)")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_kmlab)

    p = sub.add_parser("estimate", help="run the designs on a real campaign's traverse points")
    common(p)
    p.add_argument("--traverses", required=True, help="traverse file, archive or folder")
    p.add_argument("--window", default="midday",
                   help="run to headline: morning, midday (afternoon), evening, night, a run name, or all (default midday)")
    p.add_argument("--wind", nargs="*", metavar="RUN=FROM:SPEED",
                   help="wind of a run (degrees it blows from, m/s), e.g. morning=290:2.0 midday=170:7.7; "
                        "default: the forcing file's station for each run's hours")
    p.add_argument("--no-fetch-winds", action="store_true", help="do not download station winds")
    p.add_argument("--utc", action="store_true", help="timestamps are UTC without saying so: convert to local time")
    p.add_argument("--timezone", default=None, help="local time zone of tz-aware timestamps (e.g. America/New_York)")
    p.add_argument("--lab", default=None, help="identify_lab.json to judge the designs by (default: in --out)")
    p.add_argument("--name", default=None, help="output file stem (default identify_estimate)")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("map", help="the map designs on the project's own target")
    common(p)
    p.add_argument("--lab", default=None)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("simulate", help="write a simulated campaign as a CAPA-style CSV")
    common(p)
    p.add_argument("--world", default="additive", choices=GENERATORS)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    args = ap.parse_args(argv)
    if args.cmd == "simulate" and not args.out:
        ap.error("simulate needs --out <file.csv>")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
