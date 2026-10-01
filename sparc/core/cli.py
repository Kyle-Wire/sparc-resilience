"""Command-line entry for the core pipeline.

    sparc core run   --project configs/core_providence.yml [--stages S0,S1,S2,S3] [--fast] [--resume] [--cv-curve]
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
                   cv_curve=True if args.cv_curve else None)
    m = res.manifest.get("metrics", {})
    if m:
        s = m.get("stacker", {})
        print(f"stacker OOF RMSE {s.get('rmse'):.3f} R² {s.get('r2'):.3f}  "
              f"coverage {s.get('interval_coverage', float('nan')):.3f}")
    if res.run_dir:
        print(f"outputs → {res.run_dir}  (see report.md)")
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
