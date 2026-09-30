"""Command-line entry for the core pipeline.

    sparc core run   --project configs/core_providence.yml [--stages S0,S1,S2,S3] [--fast]
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
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_core_run)

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
    res = run_core(Path(args.project), stages=stages, fast=bool(args.fast))
    m = res.manifest.get("metrics", {})
    if m:
        s = m.get("stacker", {})
        print(f"stacker OOF RMSE {s.get('rmse'):.3f} R² {s.get('r2'):.3f}  "
              f"coverage {s.get('interval_coverage', float('nan')):.3f}")
    if res.run_dir:
        print(f"outputs → {res.run_dir}  (see report.md)")
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
