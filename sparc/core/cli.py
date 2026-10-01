"""Command-line entry for the core pipeline.

    sparc core run   --project configs/core_providence.yml [--stages S0,S1,S2,S3] [--fast | --coarse 60] [--resume] [--cv-curve]
    sparc core forcing --project configs/core_providence.yml --date 2020-07-18 --hours 15-16 \
                       --station 72507014765 --out configs/forcing/providence_2020-07-18.json
    sparc core synth --out output/core/synthetic_city.csv [--seed 0]
    python -m sparc.core run --project ...          (same, without the legacy CLI)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def add_core_subparsers(core_parser: argparse.ArgumentParser) -> None:
    subs = core_parser.add_subparsers(dest="core_kind", required=True)
    p_run = subs.add_parser("run", help="run the core pipeline (S0–S7) on a core config")
    p_run.add_argument("--project", "-p", required=True, help="core YAML config (see configs/core_providence.yml)")
    p_run.add_argument("--stages", default="all", help="comma list of S0..S7, or 'all'")
    p_run.add_argument("--fast", action="store_true", help="8k-point window, 3 folds, fewer epochs (smoke run)")
    p_run.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    p_run.add_argument("--resume", action="store_true",
                       help="reuse fitted stages from <run dir>/checkpoint.pkl when config, data and code match")
    p_run.add_argument("--cv-curve", action="store_true",
                       help="also report skill vs CV block size (random points, 500 m, 1000 m; reporting only)")
    p_run.add_argument("--coarse", type=float, default=None, metavar="M",
                       help="average the full extent onto M-metre cells (e.g. 60; validation-study resolution)")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_core_run)

    p_bm = subs.add_parser("benchmark", help="effect-recovery benchmark on the synthetic city (Spatial+ A/B)")
    p_bm.add_argument("--out", default="output/core/benchmark")
    p_bm.add_argument("--seed", type=int, default=0)
    p_bm.add_argument("--no-ab", action="store_true", help="only the default (Spatial+) setting")
    p_bm.add_argument("--threads", type=int, default=0)
    p_bm.set_defaults(func=cmd_core_benchmark)

    p_cl = subs.add_parser("climate", help="CMIP6 change factors at a site (AWS Pangeo archive) → CSV")
    p_cl.add_argument("--lat", type=float, required=True)
    p_cl.add_argument("--lon", type=float, required=True)
    p_cl.add_argument("--out", required=True, help="CSV of per-model change factors")
    p_cl.add_argument("--cache", default="output/core/cache", help="catalogue cache directory")
    p_cl.add_argument("--experiments", default="ssp126,ssp245,ssp370,ssp585")
    p_cl.add_argument("--months", default="6,7,8", help="season (default June–August)")
    p_cl.add_argument("--variable", default="tasmax", choices=["tasmax", "tas"])
    p_cl.add_argument("--models", default="", help="comma list (default: every model with all experiments)")
    p_cl.add_argument("--workers", type=int, default=4)
    p_cl.set_defaults(func=cmd_core_climate)

    p_fo = subs.add_parser("forcing", help="campaign-day radiation and wind (ERA5 + weather station) → JSON")
    p_fo.add_argument("--project", "-p", default=None, help="core config (site from climate.site)")
    p_fo.add_argument("--lat", type=float, default=None)
    p_fo.add_argument("--lon", type=float, default=None)
    p_fo.add_argument("--date", required=True, help="campaign date, YYYY-MM-DD")
    p_fo.add_argument("--hours", default="15-16", help="local hour range of the traverse (default 15-16)")
    p_fo.add_argument("--tz", default="America/New_York", help="IANA time zone of --hours")
    p_fo.add_argument("--station", default=None, help="NOAA Global Hourly station id (USAF+WBAN, e.g. 72507014765)")
    p_fo.add_argument("--wind-source", default="auto", choices=["auto", "station", "era5"])
    p_fo.add_argument("--cache", default="output/core/cache")
    p_fo.add_argument("--out", required=True, help="forcing JSON (reference it as physics.forcing)")
    p_fo.set_defaults(func=cmd_core_forcing)

    p_bl = subs.add_parser("baselines", help="score standard baselines on a finished run's folds (paired by block)")
    p_bl.add_argument("run_dir")
    p_bl.add_argument("--project", "-p", required=True, help="the run's core config")
    p_bl.set_defaults(func=cmd_core_baselines)

    p_pl = subs.add_parser("placebo", help="negative controls: re-fit with shifted/rotated/random layers (coarse)")
    p_pl.add_argument("--project", "-p", required=True)
    p_pl.add_argument("--kinds", default="grf,shift,rotate")
    p_pl.add_argument("--coarse", type=float, default=60.0, help="cell size in m (0 = native resolution)")
    p_pl.add_argument("--seed", type=int, default=0)
    p_pl.add_argument("--grf-range", type=float, default=600.0, help="random-field correlation range (m)")
    p_pl.add_argument("--threads", type=int, default=0)
    p_pl.set_defaults(func=cmd_core_placebo)

    p_rp = subs.add_parser("reproduce", help="re-run a finished run from its manifest and compare the numbers")
    p_rp.add_argument("run_dir")
    p_rp.add_argument("--stages", default="S0,S1,S2,S3", help="stages to re-run (default S0–S3; 'all' for everything)")
    p_rp.add_argument("--tol-r2", type=float, default=0.01)
    p_rp.add_argument("--tol-effect", type=float, default=0.05)
    p_rp.add_argument("--config-dir", default=None, help="directory the run's config paths are relative to "
                                                          "(only for runs made before provenance was recorded)")
    p_rp.add_argument("--threads", type=int, default=0)
    p_rp.set_defaults(func=cmd_core_reproduce)

    p_sc = subs.add_parser("simcheck", help="recover planted effects on the city's real layout (coarse, many runs)")
    p_sc.add_argument("--project", "-p", required=True)
    p_sc.add_argument("--design", default="physics=20,additive=20,null=20,own_only=8,coarse_scale=8,confounded=8",
                      help="generator=replicates, comma separated")
    p_sc.add_argument("--coarse", type=float, default=90.0)
    p_sc.add_argument("--epochs", type=int, default=200)
    p_sc.add_argument("--workers", type=int, default=1)
    p_sc.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p_sc.add_argument("--out", required=True, help="results directory (resumable)")
    p_sc.set_defaults(func=cmd_core_simcheck)

    p_ss = subs.add_parser("simcheck-summary", help="merge simcheck result directories and summarise")
    p_ss.add_argument("dirs", nargs="+")
    p_ss.add_argument("--out", default=None, help="write simcheck_summary.json/.md here")
    p_ss.set_defaults(func=cmd_core_simcheck_summary)

    p_mv = subs.add_parser("multiverse", help="re-run under alternative analysis choices; effect + priority stability")
    p_mv.add_argument("--project", "-p", required=True)
    p_mv.add_argument("--variants", default="", help="comma list (default: all; see sparc.core.multiverse.VARIANTS)")
    p_mv.add_argument("--coarse", type=float, default=60.0)
    p_mv.add_argument("--workers", type=int, default=1)
    p_mv.add_argument("--threads", type=int, default=1)
    p_mv.add_argument("--out", required=True)
    p_mv.set_defaults(func=cmd_core_multiverse)

    p_un = subs.add_parser("uncertainty", help="estimation / specification / attribution components for a run")
    p_un.add_argument("run_dir")
    p_un.add_argument("--multiverse", default=None, help="multiverse output directory")
    p_un.add_argument("--simcheck", nargs="*", default=[], help="simcheck result directories")
    p_un.set_defaults(func=cmd_core_uncertainty)

    p_ly = subs.add_parser("layers", help="open-data layers on the study grid (HRSL people, WorldCover) → parquet")
    p_ly.add_argument("--project", "-p", required=True)
    p_ly.add_argument("--out", required=True, help="parquet; reference it as planner.layers")
    p_ly.set_defaults(func=cmd_core_layers)

    p_pp = subs.add_parser("planner", help="planner pack for a finished run: exposure, equity, hot days, hexes, exports")
    p_pp.add_argument("run_dir")
    p_pp.add_argument("--project", "-p", required=True)
    p_pp.add_argument("--package", default=None, help="adaptation scenario (default: the climate stage's first)")
    p_pp.add_argument("--no-export", action="store_true")
    p_pp.set_defaults(func=cmd_core_planner)

    p_syn = subs.add_parser("synth", help="write the synthetic test city (with planted truths) to CSV")
    p_syn.add_argument("--out", required=True)
    p_syn.add_argument("--seed", type=int, default=0)
    p_syn.set_defaults(func=cmd_core_synth)


def cmd_core_run(args) -> int:
    from sparc.core.pipeline import ALL_STAGES, run_core

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.threads:
        import torch

        torch.set_num_threads(int(args.threads))
    stages = ALL_STAGES if args.stages in ("all", "", None) else tuple(s.strip().upper() for s in args.stages.split(","))
    res = run_core(Path(args.project), stages=stages, fast=bool(args.fast), resume=bool(args.resume),
                   cv_curve=True if args.cv_curve else None, coarse=args.coarse)
    m = res.manifest.get("metrics", {})
    if m:
        s = m.get("stacker", {})
        print(f"stacker OOF RMSE {s.get('rmse'):.3f} R² {s.get('r2'):.3f}  "
              f"coverage {s.get('interval_coverage', float('nan')):.3f}")
    if res.run_dir:
        print(f"outputs → {res.run_dir}  (see report.md)")
    return 0


def cmd_core_baselines(args) -> int:
    from sparc.core.baselines import baselines_for_run
    from sparc.core.config import load_core_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    res = baselines_for_run(args.run_dir, load_core_config(args.project))
    s = res["rows"][res["best_baseline"]]["stack_rmse"]
    print(f"stack RMSE {s:.3f}")
    for k, r in res["rows"].items():
        print(f"  {k:20s} RMSE {r['rmse']:.3f}  R² {r['r2']:.3f}  ΔMSE {r['delta_mse']:+.3f} ± {r['delta_mse_se']:.3f}")
    print("verdict:", res["verdict"])
    return 0


def cmd_core_placebo(args) -> int:
    import json

    from sparc.core.config import load_core_config
    from sparc.core.placebo import placebo_markdown, run_placebo_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.threads:
        import torch

        torch.set_num_threads(int(args.threads))
    cfg = load_core_config(args.project)
    res = run_placebo_suite(cfg, kinds=tuple(k.strip() for k in args.kinds.split(",") if k.strip()),
                            coarse=args.coarse or None, seed=args.seed, grf_range_m=args.grf_range)
    out = cfg.output_dir / f"{cfg.name}_placebo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "placebo.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = placebo_markdown(res, cfg.data.get("target_units", ""))
    (out / "placebo.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"model passes {res['n_pass_model']}/{res['n_placebos']}, causal passes "
          f"{res['n_pass_causal']}/{res['n_placebos']}  →  {out}")
    return 0


def cmd_core_reproduce(args) -> int:
    from sparc.core.pipeline import ALL_STAGES
    from sparc.core.reproduce import reproduce

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.threads:
        import torch

        torch.set_num_threads(int(args.threads))
    stages = ALL_STAGES if args.stages in ("all", "") else tuple(s.strip().upper() for s in args.stages.split(","))
    out = reproduce(args.run_dir, stages=stages, tol_r2=args.tol_r2, tol_effect=args.tol_effect,
                    config_dir=args.config_dir)
    for c in out["checks"]:
        print(f"  {'✓' if c['ok'] else ('✗' if c['hard'] else '·')} {c['check']}: {c['detail']}")
    print("REPRODUCED" if out["pass"] else "NOT REPRODUCED", "→", out["reproduction"])
    return 0 if out["pass"] else 1


def cmd_core_simcheck(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.simcheck import run_simcheck, simcheck_markdown

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    design = {k.strip(): int(v) for k, v in (kv.split("=") for kv in args.design.split(",") if kv.strip())}
    summ = run_simcheck(load_core_config(args.project), design, args.out, coarse=args.coarse or None,
                        epochs=args.epochs, workers=args.workers, threads=args.threads)
    print(simcheck_markdown(summ))
    return 0


def cmd_core_simcheck_summary(args) -> int:
    import json

    from sparc.core.simcheck import merge_results, simcheck_markdown, summarize

    summ = summarize(merge_results(args.dirs))
    md = simcheck_markdown(summ)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "simcheck_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
        (out / "simcheck_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


def cmd_core_multiverse(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.multiverse import multiverse_markdown, run_multiverse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_core_config(args.project)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] or None
    summ = run_multiverse(cfg, args.out, variants=variants, coarse=args.coarse or None, workers=args.workers,
                          threads=args.threads)
    md = multiverse_markdown(summ, cfg.data.get("target_units", ""))
    (Path(args.out) / "multiverse_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


def cmd_core_uncertainty(args) -> int:
    from sparc.core.uncertainty import uncertainty_markdown, uncertainty_report

    out = uncertainty_report(args.run_dir, args.multiverse, args.simcheck)
    print(uncertainty_markdown(out))
    return 0


def cmd_core_layers(args) -> int:
    from sparc.core.config import load_core_config
    from sparc.core.data import load_core_data
    from sparc.core.opendata import fetch_layers

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_core_config(args.project)
    cfg.raw["data"]["coarse_m"] = None
    cfg.raw["data"]["subsample"] = None
    lay = fetch_layers(load_core_data(cfg), cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lay.to_parquet(out, index=False)
    print(f"wrote {out}: {len(lay):,} cells, {lay['people'].sum():,.0f} residents (HRSL)")
    return 0


def cmd_core_planner(args) -> int:
    import json

    from sparc.core.config import load_core_config
    from sparc.core.planner import planner_pack

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    out = planner_pack(args.run_dir, load_core_config(args.project), package=args.package, export=not args.no_export)
    print(json.dumps({k: out[k] for k in ("people_total", "package", "plantable") if k in out}, indent=1))
    for r in out["exposure"]:
        print(r)
    return 0


def cmd_core_forcing(args) -> int:
    import json

    from sparc.core.forcing import campaign_forcing

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    lat, lon = args.lat, args.lon
    if (lat is None or lon is None) and args.project:
        from sparc.core.config import load_core_config

        site = (load_core_config(args.project).raw.get("climate") or {}).get("site")
        if site:
            lat, lon = float(site[0]), float(site[1])
    if lat is None or lon is None:
        raise SystemExit("give --lat/--lon or a --project with climate.site")
    h0, h1 = (int(x) for x in args.hours.split("-"))
    res = campaign_forcing(lat, lon, args.date, (h0, h1), args.tz, station=args.station,
                           wind_source=args.wind_source, cache_dir=args.cache)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    ph = res["physics"]
    print(f"sw_down {ph['sw_down']} W/m², lw_net {ph['lw_net']} W/m², wind {ph['wind']} m/s ({res['wind_source']})")
    for c in res["checks"]:
        print("  ·", c)
    print(f"wrote {out}  →  reference it as physics.forcing (path relative to the config file)")
    return 0


def cmd_core_benchmark(args) -> int:
    import json

    from sparc.core.diagnostics import benchmark_markdown, run_benchmark

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.threads:
        import torch

        torch.set_num_threads(int(args.threads))
    bench = run_benchmark(seed=int(args.seed), spatial_plus_ab=not args.no_ab)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    md = benchmark_markdown(bench)
    (out / "benchmark.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


def cmd_core_climate(args) -> int:
    from sparc.core.climate import cmip6_change_factors

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    df = cmip6_change_factors(args.lat, args.lon, args.cache,
                              experiments=tuple(e.strip() for e in args.experiments.split(",") if e.strip()),
                              months=tuple(int(m) for m in args.months.split(",")), variable=args.variable,
                              models=[m.strip() for m in args.models.split(",") if m.strip()] or None,
                              max_workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{df.model.nunique() if len(df) else 0} models → {out}")
    if len(df):
        print(df.groupby(["experiment", "period"]).delta_K.describe()[["count", "50%", "min", "max"]].round(2))
    return 0


def cmd_core_synth(args) -> int:
    import json

    from sparc.core.synthetic import make_synthetic_city

    city = make_synthetic_city(seed=int(args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    city.frame.to_csv(out, index=False)
    truth = {k: (list(v) if isinstance(v, tuple) else v) for k, v in city.truth.items()}
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(city.frame)} points) and {out.with_suffix('.truth.json')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sparc.core", description="SPARC core pipeline")
    add_core_subparsers(parser)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
