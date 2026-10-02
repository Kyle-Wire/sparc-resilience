"""Fixtures of the runs tests (``tests/studio/runs``).

* ``demo`` - the synthetic demo project with ``n=40, seed=0`` (the data the committed ``synth_run`` fixture
  was recorded on), created through ``POST /api/projects``;
* ``place_run(project, run_id, drop=..., edit_config=..., edit=...)`` - a copy of
  ``tests/studio/fixtures/synth_run`` in the project's ``runs/`` folder with a Studio ``launch.json`` (as if
  Studio had launched it), indexed by the registry; returns the run directory;
* ``fixture_run`` - ``(run_id, run_dir)`` of the plain copy;
* ``replay_runner`` - every ``run.core`` job replays the fixture (``SPARC_STUDIO_RUNNER=replay:…``);
* ``providence_runs`` - the directory of the recorded Providence runs, or skip.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from tests.studio.conftest import FIXTURES

SYNTH = FIXTURES / "synth_run"
RUN_ID = "20261001-212118-fast-a1b2"
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(var, "1")


@pytest.fixture
def synth() -> Path:
    if not (SYNTH / "manifest.json").is_file():
        pytest.skip("tests/studio/fixtures/synth_run is not present")
    return SYNTH


def create_demo(client, name: str = "Demo city") -> dict:
    r = client.post("/api/projects", json={"name": name, "template": "synthetic_demo",
                                           "options": {"n": 40, "seed": 0}})
    assert r.status_code == 201, r.text
    return r.json()["project"]


@pytest.fixture
def demo(client) -> dict:
    return create_demo(client)


def launch_snapshot(ctx, project: dict, run_id: str, args: dict, edit_config: Callable | None = None) -> dict:
    from sparc.studio.runs.launch import absolutise
    from sparc.studio.runs.reader import load_config_raw

    pdir = Path(project["dir"])
    raw = absolutise(load_config_raw(project["config_path"]), pdir, cache_dir=ctx.workspace.root / "cache",
                     runs_dir=pdir / "runs")
    if edit_config is not None:
        edit_config(raw)
    return {"studio_version": "1.0.0", "run_id": run_id, "project_id": project["id"], "config_raw": raw,
            "config_dir": str(pdir), "args": args, "created_utc": "2026-10-01T21:21:18Z", "job_id": None}


@pytest.fixture
def place_run(ctx, synth) -> Callable[..., Path]:
    """``place_run(project, run_id=RUN_ID, drop=(), edit_config=None, edit=None)`` → the indexed run folder."""
    from sparc.studio.runs.launch import mode_args

    def place(project: dict, run_id: str = RUN_ID, *, drop=(), edit_config=None, edit=None) -> Path:
        rd = Path(project["dir"]) / "runs" / run_id
        shutil.copytree(synth, rd, ignore=shutil.ignore_patterns("events.jsonl", "FIXTURE.json", *drop))
        if edit is not None:
            edit(rd)
        (rd / "studio").mkdir(exist_ok=True)
        snap = launch_snapshot(ctx, project, run_id, mode_args("fast"), edit_config)
        (rd / "studio" / "launch.json").write_text(json.dumps(snap))
        ctx.services["registry"].index_run_dir(rd, origin="studio", studio_dir=rd / "studio")
        return rd

    return place


@pytest.fixture
def fixture_run(demo, place_run) -> tuple[str, Path]:
    return RUN_ID, place_run(demo)


@pytest.fixture
def replay_runner(monkeypatch, synth):
    monkeypatch.setenv("SPARC_STUDIO_RUNNER", f"replay:{synth}")
    monkeypatch.setenv("SPARC_STUDIO_REPLAY_SPEED", "400")
    return synth


@pytest.fixture
def providence_runs() -> Path:
    d = Path(os.environ.get("SPARC_PROVIDENCE_RUNS") or REPO / "output" / "core" / "providence")
    if not (d / "providence_uhi_fast" / "manifest.json").is_file():
        pytest.skip(f"no recorded Providence runs at {d} (set SPARC_PROVIDENCE_RUNS)")
    return d


def f32(content: bytes) -> np.ndarray:
    return np.frombuffer(content, dtype="<f4")
