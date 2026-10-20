"""``sparc core lstair …``: the LST → air temperature pilot.

    sparc core lstair dataset  --cities "Providence, RI" "Boston, MA" --out DIR [--traverses-dir DIR]
                               [--no-lst] [--no-era5] [--utc] [--force]
    sparc core lstair dataset  --cities all --out DIR            (every CAPA catalog city)
    sparc core lstair evaluate --data DIR --out DIR [--windows af,am,pm] [--no-physics] [--boot 4000]
                               [--weights n] [--node-link-any-distance] [--include-flagged] [--lst-mode local]
                               [--regional] [--land-cover nlcd] [--physics-transfer medoid] [--physics-cell 30]
                               [--physics-L-max 2000]                (each non-default flag: a registered sensitivity)
    sparc core lstair sensitivity --data DIR --out DIR [--variants weights_n,lst_local,...] [--cache DIR]
                               (the primary analysis and every registered sensitivity, one after another)
    sparc core lstair validate --out DIR [--cities 8] [--seeds 0-49] [--fast]   (synthetic worlds, known answers)
    sparc core lstair hosts [--probe]                            (the hosts a real run needs)

Exit codes (dataset): 0 done (some cities may have failed: see DIR/dataset.json), 1 nothing built, 2 a host the
build needs is blocked (named on stderr).  evaluate, sensitivity: 0 done, 1 refused (reason on stderr).  validate: 0
when every world gives the answer it should, else 1.
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
    d.add_argument("--lst-prefer-clear", type=float, default=0.8,
                   help="the nearest scene at least this clear is taken; else the clearest above --lst-min-clear")
    d.add_argument("--cache", default=None, help="Landsat COG window cache (default OUT/cache/landsat)")
    d.set_defaults(func=_tracked(cmd_dataset))

    e = s.add_parser("evaluate", help="leave-one-campaign-out: N, A land cover, B + LST, C physics with LST → "
                                      "skill, the registered verdict")
    e.add_argument("--data", required=True, help="the dataset folder (the --out of lstair dataset)")
    e.add_argument("--out", required=True, help="results folder (results.json, results.md, predictions.parquet)")
    _evaluate_args(e)
    e.add_argument("--weights", choices=("variance", "n"), default="variance",
                   help="training weights: ∝ 1/(n·var(dTa)) per city (variance, registered) or ∝ 1/n (sensitivity)")
    e.add_argument("--node-link-any-distance", action="store_true",
                   help="link one OSF node and date at any distance (sensitivity: the eight Virginia cities of 15 July "
                        "2021 as one group); default only within 50 km")
    e.add_argument("--include-flagged", action="store_true",
                   help="keep runs the data checks flagged (qa_exclude: walking campaigns, single-car runs, ...)")
    e.add_argument("--lst-mode", choices=("full", "local"), default="full",
                   help="B's LST: full (registered) or local (minus its 3 km focal mean; sensitivity)")
    e.add_argument("--regional", action="store_true",
                   help="A and B also get 3 km focal elevation and impervious anomalies (sensitivity)")
    e.add_argument("--land-cover", choices=("worldcover", "nlcd"), default="worldcover",
                   help="canopy and impervious from WorldCover (registered) or NLCD tree canopy and percent impervious "
                        "(sensitivity; every scored city needs the NLCD columns)")
    e.add_argument("--physics-transfer", choices=("median", "medoid"), default="median",
                   help="physics transfer: per-parameter medians (registered) or the medoid city's whole parameter "
                        "vector (sensitivity)")
    e.add_argument("--no-within-city", action="store_true", help="skip the secondary within-city block CV")
    e.set_defaults(func=_tracked(cmd_evaluate))

    se = s.add_parser("sensitivity", help="the primary analysis and each registered sensitivity, one after another "
                                          "→ sensitivity.json / sensitivity.md")
    se.add_argument("--data", required=True, help="the dataset folder (the --out of lstair dataset)")
    se.add_argument("--out", required=True, help="folder for sensitivity.json / .md and each variant's results")
    se.add_argument("--variants", default="all",
                    help="comma list of variants, or all (default): weights_n, virginia_one_group, include_flagged, "
                         "lst_local, regional, nlcd, physics_30m, physics_medoid")
    se.add_argument("--cache", default=None,
                    help="physics fit cache root to reuse, e.g. the --out of an earlier lstair evaluate "
                         "(default OUT/primary)")
    _evaluate_args(se)
    se.set_defaults(func=_tracked(cmd_sensitivity))

    v = s.add_parser("validate", help="synthetic worlds with known answers: does the harness say what it should?")
    v.add_argument("--out", required=True, help="folder for validation.json / validation.md")
    v.add_argument("--cities", type=int, default=8, help="cities per world (default 8)")
    v.add_argument("--seeds", default="0",
                   help="world seeds: a comma list and/or ranges, e.g. 0-49 (default 0); with 10 or more the verdict "
                        "rests on the claim rates over seeds")
    v.add_argument("--size", type=int, default=64, help="grid cells per side of each city (default 64)")
    v.add_argument("--fast", action="store_true", help="spline ridge only, short physics fits")
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


def _evaluate_args(p: argparse.ArgumentParser) -> None:
    """The options ``evaluate`` and ``sensitivity`` share."""
    p.add_argument("--windows", default="af,am,pm,ni", help="run windows in order (default af,am,pm,ni)")
    p.add_argument("--boot", type=int, default=4000,
                   help="random sign vectors of the sign-flip test over more than 16 units (default 4000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=6000, help="training cells per city (default 6000; 0 = all)")
    p.add_argument("--no-gbm", action="store_true", help="additive spline ridge only (no gradient boosting)")
    p.add_argument("--no-physics", action="store_true", help="skip model C")
    p.add_argument("--physics-cell", type=float, default=60.0, help="physics cell size, m (default 60)")
    p.add_argument("--physics-iter", type=int, default=60, help="physics L-BFGS iterations (default 60)")
    p.add_argument("--physics-L-max", dest="physics_L_max", type=float, default=2000.0,
                   help="upper bound of the physics length scale L, m (default 2000)")


def _settings(args, **kw):
    from sparc.core.lstair.evaluate import Settings

    return Settings(windows=tuple(w.strip() for w in args.windows.split(",") if w.strip()), boot=args.boot,
                    seed=args.seed, max_rows_per_city=args.max_rows, gbm=not args.no_gbm, physics=not args.no_physics,
                    physics_cell_m=args.physics_cell, physics_iter=args.physics_iter,
                    physics_L_max_m=args.physics_L_max, **kw)


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
    kw = dict(cell_m=args.cell, margin_m=args.margin, drift=args.drift, lst_window_days=args.lst_window_days,
              lst_min_clear=args.lst_min_clear, lst_prefer_clear=args.lst_prefer_clear, cache_dir=Path(args.cache) if args.cache else out / "cache" / "landsat",
              skip_files=tuple(q for q in D.SKIP_FILES if q != "appendix") if args.keep_appendix else D.SKIP_FILES, campaign_dates=_dates(args.campaign_date))
    try:
        # one call for every city (each converts a --utc clock with its own time zone), so a city is never built
        # from the whole --traverses-dir and dataset.json lists every city
        summary = D.build_dataset(cities, out, traverses_dir=args.traverses_dir, lst=False if args.no_lst else None,
                                  era5=False if args.no_era5 else None, force=args.force, utc=bool(args.utc), **kw)
    except D.NetworkBlocked as exc:
        print(f"lstair dataset: {exc}", file=sys.stderr)
        return 2
    ok = [c for c, v in summary["cities"].items() if v["status"] == "ok"]
    na = [c for c, v in summary["cities"].items() if v["status"] == "unavailable"]
    for c, v in summary["cities"].items():
        line = f"{c}: {v['status']}"
        if v["status"] == "ok":
            line += f" ({v.get('rows', 'cached')} rows{', ' + ', '.join(v['runs']) if v.get('runs') else ''})"
        else:
            line += f" — {v.get('error')}"
        print(line)
    print(f"{len(ok)} of {len(summary['cities'])} cities ready in {out / 'cities'}"
          + (f" ({len(na)} have no traverse points in the catalog)" if na else ""))
    return 0 if ok else 1


def _line(name: str, p: dict) -> str:
    ci = p.get("ci95")
    holm = p.get("p_holm")
    return (f"{name} mean ΔSS " + (f"{p['mean_dss']:+.3f} (95% CI {ci[0]:+.3f} to {ci[1]:+.3f}"
                                   + (f", Holm p {holm:.3g}" if holm is not None else "") + ")" if ci else "—")
            + f" → {p['verdict']}")


def cmd_evaluate(args) -> int:
    from sparc.core.lstair.evaluate import PRIMARY_WINDOW, SAME_NODE_KM, run_evaluation

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        s = _settings(args, weights=args.weights, same_node_km=None if args.node_link_any_distance else SAME_NODE_KM,
                      include_flagged=args.include_flagged, lst_mode=args.lst_mode, regional=args.regional,
                      land_cover=args.land_cover, physics_transfer=args.physics_transfer,
                      within_city=not args.no_within_city)
        res = run_evaluation(args.data, args.out, s)
    except ValueError as exc:
        print(f"lstair evaluate: {exc}", file=sys.stderr)
        return 1
    for run, w in res["windows"].items():
        if w.get("status") != "done":
            print(f"{run}: not run ({w.get('reason')})")
            continue
        tag = " (primary)" if run == PRIMARY_WINDOW else ""
        print(f"{run}{tag}: {len(w['per_unit'])} held-out units; "
              + "; ".join(_line(p["comparison"], p) for k, p in w["paired"].items() if k in ("B-A", "C-A")))
    print(f"results: {Path(args.out) / 'results.md'}")
    return 0


def cmd_sensitivity(args) -> int:
    from sparc.core.lstair.evaluate import VARIANTS, run_sensitivity

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    names = list(VARIANTS) if args.variants.strip().lower() == "all" else \
        [v.strip() for v in args.variants.split(",") if v.strip()]
    try:
        sens = run_sensitivity(args.data, args.out, _settings(args, within_city=False), names, cache=args.cache)
    except ValueError as exc:
        print(f"lstair sensitivity: {exc}", file=sys.stderr)
        return 1
    for r in sens["rows"]:
        if r.get("status") != "done":
            print(f"{r['variant']}: not run ({r.get('reason')})")
            continue
        w = (r.get("windows") or {}).get(sens["primary_window"]) or {}
        b = w.get("B-A") or {}
        print(f"{r['variant']}: {sens['primary_window']} B − A " + (
            f"ΔSS {b['mean_dss']:+.3f}, Holm p {b['p_holm']:.3g} → {b['verdict']}" if b.get("mean_dss") is not None
            and b.get("p_holm") is not None else (b.get("verdict") or w.get("reason") or "see sensitivity.md")))
    print(f"sensitivities: {Path(args.out) / 'sensitivity.md'}")
    return 0


def _seeds(text: str) -> tuple[int, ...]:
    """``"0,1,2"`` or ranges ``"0-49"`` (inclusive)."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


def cmd_validate(args) -> int:
    from sparc.core.lstair.evaluate import Settings
    from sparc.core.lstair.synthetic import validate, validation_markdown

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    s = (Settings(physics_cell_m=30.0, physics_iter=30, within_city=False, gbm=False, boot=1000) if args.fast
         else Settings(physics_cell_m=30.0, within_city=False))
    v = validate(n_cities=args.cities, seeds=_seeds(args.seeds), settings=s, n=args.size, out=args.out)
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
