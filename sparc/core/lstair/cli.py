"""``sparc core lstair …``: the LST → air temperature pilot.

    sparc core lstair dataset  --cities "Providence, RI" "Boston, MA" --out DIR [--traverses-dir DIR]
                               [--no-lst] [--no-era5] [--utc] [--force]
    sparc core lstair dataset  --cities all --out DIR            (every CAPA catalog city)
    sparc core lstair evaluate --data DIR --out DIR [--windows af,am,pm] [--no-physics] [--boot 4000]
    sparc core lstair validate --out DIR [--cities 8] [--seeds 0,1,2] [--fast]   (synthetic worlds, known answers)
    sparc core lstair hosts [--probe]                            (the hosts a real run needs)

Exit codes (dataset): 0 done (some cities may have failed: see DIR/dataset.json), 1 nothing built, 2 a host the
build needs is blocked (named on stderr).  validate: 0 when every world gives the answer it should, else 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def add_lstair_parser(subs) -> None:
    p = subs.add_parser("lstair", help="LST → air temperature pilot (multi-city traverses, three-model harness)")
    s = p.add_subparsers(dest="lstair_kind", required=True)

    d = s.add_parser("dataset", help="traverse cells of every city with land-cover, focal and Landsat LST predictors")
    d.add_argument("--cities", nargs="+", required=True,
                   help='city names as in the CAPA catalog, one per argument (e.g. "Providence, RI"), or "all"')
    d.add_argument("--out", required=True, help="output folder (one cache per city under OUT/cities)")
    d.add_argument("--traverses-dir", default=None,
                   help="local campaign folders, one per city (named by slug, e.g. providence_ri, or by city name); "
                        "with one city, the folder itself.  Otherwise the traverses are downloaded from OSF.")
    d.add_argument("--no-lst", action="store_true", help="skip Landsat (LST columns NaN)")
    d.add_argument("--no-era5", action="store_true", help="skip ERA5 (ta_era5 NaN; physics uses default forcing)")
    d.add_argument("--utc", action="store_true", help="the traverse clocks are UTC without saying so: convert to local")
    d.add_argument("--force", action="store_true", help="rebuild cities that are already cached")
    d.add_argument("--campaign-date", action="append", default=[], metavar="CITY=YYYY-MM-DD",
                   help="local campaign date of a city whose traverse clock has no date (2019 NIHHIS time_n tables, "
                        "e.g. \"Boise, ID=2019-07-..\" from the city's report); repeatable")
    d.add_argument("--keep-appendix", action="store_true",
                   help="also read files whose name contains 'appendix' (CAPA's appendix archives repeat the main "
                        "traverses; repeated readings are read once either way)")
    d.add_argument("--cell", type=float, default=30.0, help="cell size in metres (default 30)")
    d.add_argument("--margin", type=float, default=2000.0, help="grid margin around the traverses, m (default 2000)")
    d.add_argument("--drift", choices=("fe", "protocol", "none"), default="fe",
                   help="drift/vehicle adjustment: cell fixed effects where repeat visits ≥ 15 min apart identify "
                        "them (SE ≤ 0.3 °C/h; else CAPA's line through the hour), protocol (always the line), none")
    d.add_argument("--lst-window-days", type=int, default=16, help="Landsat search window ± days (default 16)")
    d.add_argument("--lst-min-clear", type=float, default=0.5, help="minimum clear fraction over the study area")
    d.add_argument("--cache", default=None, help="Landsat COG window cache (default OUT/cache/landsat)")
    d.set_defaults(func=_tracked(cmd_dataset))

    e = s.add_parser("evaluate", help="leave-one-city-out: A land cover, B + LST, C physics with LST → results")
    e.add_argument("--data", required=True, help="the dataset folder (the --out of lstair dataset)")
    e.add_argument("--out", required=True, help="results folder (results.json, results.md, predictions.parquet)")
    e.add_argument("--windows", default="af,am,pm,ni", help="run windows in order (default af,am,pm,ni)")
    e.add_argument("--boot", type=int, default=4000, help="bootstrap resamples over cities (default 4000)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--max-rows", type=int, default=6000, help="training cells per city (default 6000; 0 = all)")
    e.add_argument("--no-gbm", action="store_true", help="additive spline ridge only (no gradient boosting)")
    e.add_argument("--no-physics", action="store_true", help="skip model C")
    e.add_argument("--physics-cell", type=float, default=60.0, help="physics cell size, m (default 60)")
    e.add_argument("--physics-iter", type=int, default=60, help="physics L-BFGS iterations (default 60)")
    e.add_argument("--no-within-city", action="store_true", help="skip the secondary within-city block CV")
    e.set_defaults(func=_tracked(cmd_evaluate))

    v = s.add_parser("validate", help="synthetic worlds with known answers: does the harness say what it should?")
    v.add_argument("--out", required=True, help="folder for validation.json / validation.md")
    v.add_argument("--cities", type=int, default=8, help="cities per world (default 8)")
    v.add_argument("--seeds", default="0", help="comma list of world seeds (default 0)")
    v.add_argument("--size", type=int, default=64, help="grid cells per side of each city (default 64)")
    v.add_argument("--fast", action="store_true", help="spline ridge only, short physics fits, 1000 bootstraps")
    v.set_defaults(func=_tracked(cmd_validate))

    h = s.add_parser("hosts", help="list the hosts a real run needs and probe them")
    h.add_argument("--probe", action="store_true", help="probe each host (short timeout) instead of only listing")
    h.set_defaults(func=_tracked(cmd_hosts))
    # progress flags on the group (``lstair --progress P dataset``) and on each command (``lstair dataset
    # --progress P``); a command's flags default to "not given" so they never overwrite the group's
    _progress_args(p, argparse.SUPPRESS)
    p.set_defaults(func=_tracked(lambda args: p.print_help() or 1))
    for q in s.choices.values():
        _progress_args(q, argparse.SUPPRESS)


def _tracked(fn):
    """``fn`` with SPARC's progress sink and cancellation (``sparc.core.cli.tracked_command``)."""
    from sparc.core.cli import tracked_command

    return tracked_command(fn)


def _progress_args(p: argparse.ArgumentParser, default=None) -> None:
    g = p.add_argument_group("progress (SPARC Studio)")
    g.add_argument("--progress", metavar="PATH", default=default,
                   help="append structured progress events (JSON lines) to PATH; same as SPARC_PROGRESS=PATH")
    g.add_argument("--job-id", default=default, help="job id stamped on every progress event (SPARC_JOB_ID)")


def _cities(names: list[str]) -> list[str]:
    from sparc.core.lstair.dataset import all_catalog_cities

    if len(names) == 1 and names[0].strip().lower() == "all":
        return all_catalog_cities()
    return [n.strip() for n in names if n.strip()]


def _dates(items: list[str]) -> dict[str, str]:
    import datetime as dt

    out = {}
    for it in items or []:
        name, sep, day = str(it).rpartition("=")
        if not sep or not name.strip():
            raise SystemExit(f"--campaign-date {it!r}: expected CITY=YYYY-MM-DD")
        out[name.strip()] = dt.date.fromisoformat(day.strip()).isoformat()
    return out


def cmd_dataset(args) -> int:
    from sparc.core.lstair import dataset as D

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    cities = _cities(args.cities)
    utc_to = None
    kw = dict(cell_m=args.cell, margin_m=args.margin, drift=args.drift, lst_window_days=args.lst_window_days,
              lst_min_clear=args.lst_min_clear, cache_dir=Path(args.cache) if args.cache else out / "cache" / "landsat",
              skip_files=() if args.keep_appendix else D.SKIP_FILES, campaign_dates=_dates(args.campaign_date))
    try:
        if args.utc:
            summary = {"cities": {}}
            for c in cities:                      # each city converts with its own time zone
                spec = D.city_spec(c)
                if not spec.tz:
                    print(f"{c}: no time zone known for --utc", file=sys.stderr)
                    continue
                s = D.build_dataset([c], out, traverses_dir=args.traverses_dir, lst=False if args.no_lst else None,
                                    era5=False if args.no_era5 else None, force=args.force, utc_to=spec.tz, **kw)
                summary["cities"].update(s["cities"])
        else:
            summary = D.build_dataset(cities, out, traverses_dir=args.traverses_dir, lst=False if args.no_lst else None,
                                      era5=False if args.no_era5 else None, force=args.force, utc_to=utc_to, **kw)
    except D.NetworkBlocked as exc:
        print(f"lstair dataset: {exc}", file=sys.stderr)
        return 2
    ok = [c for c, v in summary["cities"].items() if v["status"] == "ok"]
    for c, v in summary["cities"].items():
        line = f"{c}: {v['status']}"
        if v["status"] == "ok":
            line += f" ({v.get('rows', 'cached')} rows{', ' + ', '.join(v['runs']) if v.get('runs') else ''})"
        else:
            line += f" — {v.get('error')}"
        print(line)
    print(f"{len(ok)} of {len(summary['cities'])} cities ready in {out / 'cities'}")
    return 0 if ok else 1


def cmd_evaluate(args) -> int:
    from sparc.core.lstair.evaluate import Settings, run_evaluation

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    s = Settings(windows=tuple(w.strip() for w in args.windows.split(",") if w.strip()), boot=args.boot,
                 seed=args.seed, max_rows_per_city=args.max_rows, gbm=not args.no_gbm, physics=not args.no_physics,
                 physics_cell_m=args.physics_cell, physics_iter=args.physics_iter, within_city=not args.no_within_city)
    try:
        res = run_evaluation(args.data, args.out, s)
    except ValueError as exc:
        print(f"lstair evaluate: {exc}", file=sys.stderr)
        return 1
    for run, w in res["windows"].items():
        if w.get("status") != "done":
            print(f"{run}: not run ({w.get('reason')})")
            continue
        p = w["paired"]["B-A"]
        ci = p.get("ci95")
        print(f"{run}: {len(w['per_city'])} held-out cities; B − A mean ΔMSE "
              + (f"{p['mean_diff_mse']:+.4f} (95% CI {ci[0]:+.4f} to {ci[1]:+.4f})" if ci else "—")
              + f" → {p['verdict']}")
    print(f"results: {Path(args.out) / 'results.md'}")
    return 0


def cmd_validate(args) -> int:
    from sparc.core.lstair.evaluate import Settings
    from sparc.core.lstair.synthetic import validate, validation_markdown

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    s = (Settings(physics_cell_m=30.0, physics_iter=30, within_city=False, gbm=False, boot=1000) if args.fast
         else Settings(physics_cell_m=30.0, within_city=False))
    v = validate(n_cities=args.cities, seeds=tuple(int(x) for x in args.seeds.split(",") if x.strip()), settings=s,
                 n=args.size, out=args.out)
    print(validation_markdown(v))
    return 0 if v["all_as_expected"] else 1


def cmd_hosts(args) -> int:
    from sparc.core.lstair import dataset as D

    rows = []
    for name, (probes, hosts, purpose, remedy) in D.HOSTS.items():
        status = None
        if args.probe:
            status = "ok"
            for pr in probes:
                host, why = D.run_probe(pr)
                if why:
                    status = f"{host}: {why}"
                    break
        rows.append({"source": name, "hosts": list(hosts), "serves": purpose,
                     "probes": [D.probe_label(pr) for pr in probes], "status": status})
    print(json.dumps(rows, indent=1))
    return 0


__all__ = ["add_lstair_parser"]
