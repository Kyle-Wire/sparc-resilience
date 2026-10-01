#!/usr/bin/env python
"""Regenerate the committed SPARC Studio run fixture (``tests/studio/fixtures/synth_run/``).

It writes the synthetic demo project (``sparc.core.synthetic.write_demo_project``,
n = 40, seed = 0), runs it in fast mode with every stage (causal validation
on canopy, a budget for S7, the table climate and the demo layers) with
``SPARC_PROGRESS`` set, and saves

* the run directory without ``checkpoint.pkl`` (the run's files at the top
  level, as the replay runner copies them by their ``artifact`` paths),
* ``events.jsonl`` — the progress events of the run,
* ``FIXTURE.json`` — ``{n, seed, mode, generator}`` (the replay e2e creates
  its demo project with the same ``n`` and ``seed``).

The run uses one thread, so re-running gives numerically identical arrays.
It works in a fixed directory (``--work``, default ``<tmp>/sparc-fixture``,
emptied first) and gives the data file a fixed modification time, so the
paths and the resume fingerprint recorded in the fixture do not change from
one regeneration to the next either.

    python scripts/make_studio_fixtures.py [--out tests/studio/fixtures/synth_run] [--work DIR] [--n 40] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "tests" / "studio" / "fixtures" / "synth_run"
DEFAULT_WORK = Path(tempfile.gettempdir()) / "sparc-fixture"
MARKER = ".sparc-fixture-work"
DATA_MTIME = 1_790_000_000                  # fixed data-file mtime: part of the resume fingerprint
SKIP = {"checkpoint.pkl", ".sparc.lock"}
MAX_BYTES = 3 * 1024 * 1024


def _fresh(work: Path) -> None:
    """Empty ``work`` — only a directory this script made (it holds the marker file) or an empty one."""
    if work.exists():
        if any(work.iterdir()) and not (work / MARKER).exists():
            raise SystemExit(f"{work} is not empty and was not made by this script; pass another --work")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / MARKER).write_text("scratch directory of scripts/make_studio_fixtures.py\n", encoding="utf-8")


def generate(out: Path, n: int = 40, seed: int = 0, work: Path = DEFAULT_WORK) -> dict:
    """Run the demo in ``work`` and write the fixture into ``out`` (both replaced); returns ``FIXTURE.json``."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from sparc.core import progress
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    work = Path(work).absolute()
    _fresh(work)
    try:
        demo = write_demo_project(work / "project", n=n, seed=seed)
        os.utime(demo["files"]["data"], (DATA_MTIME, DATA_MTIME))
        events = work / "events.jsonl"
        os.environ[progress.ENV_SINK] = str(events)
        os.environ.pop(progress.ENV_CANCEL, None)
        os.environ[progress.ENV_LEVEL] = "info"
        os.environ[progress.ENV_JOB] = "j_fixture"
        progress.configure_from_env()
        progress.set_threads(1)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        try:
            run_core(load_core_config(demo["config_path"]), fast=True, run_dir=work / "run",
                     run_meta={"origin": "fixture", "generator": "write_demo_project"})
        finally:
            progress.reset()
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        run = work / "run"
        for src in sorted(run.rglob("*")):
            rel = src.relative_to(run)
            if not src.is_file() or rel.name in SKIP or rel.name.endswith(".tmp"):
                continue
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out / rel)
        shutil.copyfile(events, out / "events.jsonl")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    fixture = {"n": int(n), "seed": int(seed), "mode": "fast", "generator": "write_demo_project"}
    (out / "FIXTURE.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > MAX_BYTES:
        raise SystemExit(f"fixture is {size / 2**20:.2f} MB, above the 3 MB budget")
    return fixture


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="fixture directory (replaced)")
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="scratch directory for the run (emptied, removed)")
    ap.add_argument("--n", type=int, default=40, help="synthetic city side (cells)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    generate(out, n=args.n, seed=args.seed, work=Path(args.work))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture → {out} ({size / 2**20:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
