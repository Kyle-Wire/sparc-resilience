"""Fixtures of the studies and exports tests (``tests/studio/studies``).

* ``demo``, ``place_run``, ``fixture_run``, ``synth`` - the runs tests' demo project (``n=40, seed=0``) and a
  copy of the committed synthetic fixture run in its ``runs/`` folder with a Studio ``launch.json``;
* ``fake_studies`` - every process job runs through ``fake_worker.py`` (the real worker with the core study
  functions faked, :mod:`tests.studio.studies.fakes`);
* ``calls(job)`` - the fake calls a job made (``<job dir>/fake_calls.jsonl``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.studio.runs.conftest import RUN_ID, _single_thread, demo, fixture_run, place_run, synth  # noqa: F401

FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"


@pytest.fixture
def fake_studies(ctx):
    """Run every process job through ``fake_worker.py``."""
    ex = ctx.jobs.executors["process"]
    ex.command = lambda job: [sys.executable, str(FAKE_WORKER), str(Path(job["job_dir"]))]
    return ex


@pytest.fixture
def calls(ctx):
    """``calls(job)`` → the fake library calls of a job (dicts with ``fn`` and the arguments)."""
    def read(job: dict) -> list[dict]:
        p = ctx.workspace.job_dir(job["id"]) / "fake_calls.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    return read


def snapshot_title(run_dir: Path, title: str = "SNAPSHOT CONFIG") -> str:
    """Mark the run's launch snapshot (``launch.json`` ``config_raw.report.title``) so a test can tell it from the
    project's current config."""
    p = Path(run_dir) / "studio" / "launch.json"
    snap = json.loads(p.read_text())
    snap["config_raw"].setdefault("report", {})["title"] = title
    p.write_text(json.dumps(snap))
    return title
