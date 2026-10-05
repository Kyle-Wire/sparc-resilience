"""Fixtures of the Scenario Lab tests (``tests/studio/engine``).

* ``demo`` - the synthetic demo project (``n=40, seed=0``: the data the committed ``synth_run`` fixture and
  the engine runs below are made from), created through ``POST /api/projects``;
* ``place_run(project, src, run_id)`` - a copy of a run folder in the project's ``runs/`` with a Studio
  ``launch.json`` (as if Studio had launched it), indexed by the registry;
* ``synth_run`` - ``(run_id, run_dir)`` of the committed fixture run (no checkpoint): compile, preview,
  plans, statistics and the library work on it;
* ``engine_run_dir`` (session) - a fresh fast run S0–S5 of the same demo city **with** its checkpoint
  (≈25 s, built once per session); ``engine_run`` places a copy in the test's project (``slow`` tests);
* ``fake_emulator(run_dir, ctx)`` - a small but complete ``emulator.npz`` / ``.json`` for a run's grid;
* ``wait_job`` comes from ``tests/studio/conftest.py``; every engine host a test started is killed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.studio.conftest import FIXTURES

SYNTH = FIXTURES / "synth_run"
RUN_ID = "20261001-212118-fast-a1b2"
ENGINE_RUN_ID = "20261002-050000-fast-e0e0"


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(var, "1")


@pytest.fixture(autouse=True)
def _reset_flights():
    from sparc.studio.engine import preview

    preview.FLIGHTS = preview.PreviewFlights()
    yield


def kill_host(workspace) -> None:
    """Kill the engine host of a workspace (tests never leave one behind)."""
    info_path = Path(workspace.engine_dir) / "host.json"
    try:
        info = json.loads(info_path.read_text())
        from tests.studio.conftest import kill_tree

        kill_tree(int(info["pid"]))
    except (OSError, ValueError, KeyError, ProcessLookupError, PermissionError):
        pass
    except Exception:                          # psutil.NoSuchProcess on Windows: already gone
        pass


@pytest.fixture
def ctx(client):
    sctx = client.app.state.studio
    yield sctx
    kill_host(sctx.workspace)


def create_demo(client, name: str = "Demo city") -> dict:
    r = client.post("/api/projects", json={"name": name, "template": "synthetic_demo",
                                           "options": {"n": 40, "seed": 0}})
    assert r.status_code == 201, r.text
    return r.json()["project"]


@pytest.fixture
def demo(client) -> dict:
    return create_demo(client)


def launch_snapshot(sctx, project: dict, run_id: str) -> dict:
    from sparc.studio.runs.launch import absolutise, mode_args
    from sparc.studio.runs.reader import load_config_raw

    pdir = Path(project["dir"])
    raw = absolutise(load_config_raw(project["config_path"]), pdir, cache_dir=sctx.workspace.root / "cache",
                     runs_dir=pdir / "runs")
    return {"studio_version": "1.0.0", "run_id": run_id, "project_id": project["id"], "config_raw": raw,
            "config_dir": str(pdir), "args": mode_args("fast"), "created_utc": "2026-10-02T05:00:00Z", "job_id": None}


@pytest.fixture
def place_run(ctx):
    def place(project: dict, src: Path, run_id: str) -> Path:
        rd = Path(project["dir"]) / "runs" / run_id
        shutil.copytree(src, rd, ignore=shutil.ignore_patterns("events.jsonl", "FIXTURE.json", "studio"))
        (rd / "studio").mkdir(exist_ok=True)
        (rd / "studio" / "launch.json").write_text(json.dumps(launch_snapshot(ctx, project, run_id)))
        ctx.services["registry"].index_run_dir(rd, origin="studio", studio_dir=rd / "studio")
        return rd

    return place


@pytest.fixture
def synth_run(demo, place_run) -> tuple[str, Path]:
    if not (SYNTH / "manifest.json").is_file():
        pytest.skip("tests/studio/fixtures/synth_run is not present")
    return RUN_ID, place_run(demo, SYNTH, RUN_ID)


@pytest.fixture
def run_ctx(ctx, synth_run):
    rid, _ = synth_run
    return ctx.services["reader"].get(rid)


def write_fake_emulator(run_dir: Path, rctx, levers=("canopy", "impervious", "albedo"), seed: int = 0) -> dict:
    """An ``emulator.npz`` / ``.json`` in core's format with random but realistic arrays."""
    from sparc.core import runio

    rng = np.random.default_rng(seed)
    n = rctx.grid.n
    arrays = {"ids": np.asarray(rctx.grid.ids)}
    meta = {"levers": {}, "kernel_cells": 3}
    k = 3
    kern = rng.normal(0, 0.01, (2 * k + 1, 2 * k + 1))
    arrays["physics_kernel"] = kern
    for j, var in enumerate(levers):
        arrays[f"{var}__own"] = rng.normal(-0.05, 0.01, n)
        arrays[f"{var}__coef0"] = rng.normal(-0.02, 0.005, n)
        arrays[f"{var}__w__{var}"] = np.ones(n)
        arrays[f"{var}__dq"] = rng.normal(0.1, 0.02, n)
        good = j == 0
        meta["levers"][var] = {
            "channels": [{"column": var, "feature": f"{var}__f1", "sigma_cells": 1.5, "coef": f"{var}__coef0",
                          "weight": f"{var}__w__{var}", "unit_weight": True}],
            "physics": True, "design_dose": 5.0, "bounds": [0.0, 100.0], "direction": "increase",
            "validation": {"patch_pass_rate": 0.95 if good else 0.6, "uniform": {"rel_err": 0.2 if good else 2.46}}}
    runio.write_npz_atomic(Path(run_dir) / "emulator.npz", compressed=True, **arrays)
    runio.write_json_atomic(Path(run_dir) / "emulator.json", meta)
    return meta


@pytest.fixture
def fake_emulator():
    return write_fake_emulator


# ---------------------------------------------------------------------------
# a real run with a checkpoint (slow tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_run_dir(tmp_path_factory) -> Path:
    """A fast S0–S5 run of the n=40 demo city with ``checkpoint.pkl`` (built once per session)."""
    from sparc.core import progress
    from sparc.core.config import load_core_config
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import write_demo_project

    root = tmp_path_factory.mktemp("engine_run")
    demo = write_demo_project(root / "proj", n=40, seed=0)
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in saved:
        os.environ[k] = "1"
    try:
        progress.set_threads(1)
        run_core(load_core_config(demo["config_path"]), stages=("S0", "S1", "S2", "S3", "S4", "S5"), fast=True,
                 run_dir=root / "run")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert (root / "run" / "checkpoint.pkl").is_file()
    return root / "run"


@pytest.fixture
def engine_run(demo, place_run, engine_run_dir) -> tuple[str, Path]:
    return ENGINE_RUN_ID, place_run(demo, engine_run_dir, ENGINE_RUN_ID)


# ---------------------------------------------------------------------------
# a stored exact result without the engine (from the fixture's configured scenario folds)
# ---------------------------------------------------------------------------

def make_result(sctx, run_ctx, name: str = "Canopy Increase +10", *, scenario: dict | None = None,
                kind: str = "exact", lever: str = "canopy", amount: float = 10.0) -> dict:
    """Write ``results/<id>/`` for a configured scenario's stored folds (as the engine host would) and index it."""
    from sparc.studio.engine import stats as S
    from sparc.studio.engine import store
    from sparc.studio.engine.ops import env_for
    from sparc.studio.workspace import new_id, utc_now

    det = run_ctx.scenario_detail()
    folds = np.asarray(det["folds"][name], dtype=np.float64)
    delta = folds.mean(axis=0)
    n = delta.size
    realized = {lever: np.full(n, amount)}
    rid = new_id("result")
    rdir = store.result_dir(run_ctx.studio_dir, rid)
    env = env_for(run_ctx, None)
    spec = {"schema": 1, "id": rid, "kind": kind, "name": name, "scenario_id": (scenario or {}).get("id"),
            "revision": (scenario or {}).get("revision"), "content_hash": (scenario or {}).get("content_hash", "h"),
            "run_id": run_ctx.run_id, "ckpt_key": store.checkpoint_key(run_ctx.run_dir), "code_sha": store.code_sha(),
            "created_utc": utc_now(), "job_id": None}
    summary = S.build_result(result_id=rid, kind=kind, run_id=run_ctx.run_id, created_utc=spec["created_utc"],
                             job_id=None, delta=delta, delta_sd=np.asarray(det["sd"][name], dtype=np.float64),
                             extrapolation=np.asarray(det["ex"][name], dtype=np.float64), folds=folds,
                             realized=realized, grid=run_ctx.grid, unit="°F", name=name,
                             scenario={"id": scenario["id"], "revision": 1, "name": name} if scenario else None,
                             people=env["people"], zones=env["zones"], lever_ranges=env["ranges"],
                             per_unit={lever: 1.0}, causal=run_ctx.json("causal.json"), cfg_raw=run_ctx.cfg_raw,
                             manifest=run_ctx.manifest, spec=spec, obs=env["obs"], threshold=env["threshold"],
                             warming_mid=env["warming_mid"], warming_label=env["warming_label"])
    store.write_result(rdir, ids=run_ctx.grid.ids, delta=delta, delta_sd=det["sd"][name],
                       extrapolation=det["ex"][name], folds=folds, realized=realized, spec=spec, summary=summary)
    return store.insert_result_row(sctx.db, rdir)


def write_incompatible_checkpoint(run_dir: Path) -> None:
    """A ``checkpoint.pkl`` whose ensemble class no longer exists in core (pickle drift)."""
    import pickle

    import sparc.core.ensemble as ens_mod

    class GhostEnsemble:
        pass

    GhostEnsemble.__module__ = "sparc.core.ensemble"
    GhostEnsemble.__qualname__ = "GhostEnsemble"
    ens_mod.GhostEnsemble = GhostEnsemble
    try:
        blob = pickle.dumps({"ensemble": GhostEnsemble(), "influence": None, "done": {"S3"}})
    finally:
        del ens_mod.GhostEnsemble
    (Path(run_dir) / "checkpoint.pkl").write_bytes(blob)
