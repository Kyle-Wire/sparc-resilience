"""An existing run's rebuilt config keeps the advection scheme it was fitted with.

Manifests store the config after merging ``DEFAULTS``, so a manifest config
without ``physics.advection_scheme`` is a run made before the key existed,
fitted with central differences.  Rebuilding its config (``sparc core
reproduce``, ``session.config_for_run``, Studio's ``RunContext``) merges
today's defaults, and must not turn it into the current default scheme.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sparc.core import operators as ops
from sparc.core.config import core_config_from_dict, recorded_advection_scheme, with_advection_scheme

FIXTURE_MANIFEST = Path(__file__).resolve().parents[1] / "studio" / "fixtures" / "synth_run" / "manifest.json"

# ``physics`` of the committed Studio fixture's manifest config as recorded before the scheme existed (203890d).
PRE_SCHEME_PHYSICS = {
    "enabled": True, "window": "day", "sw_down": 800.0, "lw_net": -100.0, "forcing": None,
    "roles": {"albedo": "albedo", "canopy": "canopy", "impervious": "impervious", "ndvi": "ndvi",
              "elevation": "elevation", "water_distance": "water_dist"},
    "wind": [0.03333333333333333, 0.0], "tau_s": 1800.0, "L_max_m": 2000.0, "v_max_m": 1000.0,
    "fit_advection": "auto", "select_advection": True, "max_iter": 60,
}


def test_recorded_scheme_comes_from_the_runs_records_not_the_defaults():
    old = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))["config"]
    old["physics"] = copy.deepcopy(PRE_SCHEME_PHYSICS)
    # the defect: merging today's DEFAULTS makes a pre-scheme config report the current default
    assert core_config_from_dict(old).raw["physics"]["advection_scheme"] == ops.DEFAULT_SCHEME != "central"
    assert recorded_advection_scheme(old) == ops.LEGACY_SCHEME == "central"
    assert recorded_advection_scheme({"core": old}) == "central"
    pinned = with_advection_scheme(old, recorded_advection_scheme(old))
    assert core_config_from_dict(pinned).raw["physics"]["advection_scheme"] == "central"
    assert "advection_scheme" not in old["physics"]                       # the input is not modified
    for s in ops.SCHEMES:                                                  # a recorded scheme is kept as is
        new = {**old, "physics": {**PRE_SCHEME_PHYSICS, "advection_scheme": s}}
        assert recorded_advection_scheme(new) == s
        assert with_advection_scheme(new, "central")["physics"]["advection_scheme"] == s
    # without a manifest config: the per-fold params in physics.json decide; with neither, nothing is known
    assert recorded_advection_scheme(None, [{"L_m": 200.0}, {"L_m": 210.0}]) == "central"
    assert recorded_advection_scheme(None, [{"advection_scheme": "hybrid"}]) == "hybrid"
    assert recorded_advection_scheme(None, None) is None and recorded_advection_scheme(None, []) is None
    assert with_advection_scheme({"cv": {}}, None) == {"cv": {}}
    pinned_core = with_advection_scheme({"core": {"physics": None}}, "central")
    assert pinned_core == {"core": {"physics": {"advection_scheme": "central"}}}


@pytest.fixture(scope="module")
def demo_project(tmp_path_factory):
    from sparc.core.synthetic import write_demo_project

    root = tmp_path_factory.mktemp("scheme_run")
    write_demo_project(root / "project", n=40, seed=0)             # the fixture's demo (data/city.csv)
    return root / "project"


def _run_dir(tmp_path: Path, project: Path, scheme: str | None, launch: bool = False) -> Path:
    """A finished run's records: the Studio fixture's manifest config (relative paths) with
    ``physics.advection_scheme`` removed (a pre-scheme run) or set, against ``project``."""
    m = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    cfg = m["config"]
    cfg["physics"].pop("advection_scheme", None)
    if scheme is not None:
        cfg["physics"]["advection_scheme"] = scheme
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"config": cfg, "provenance": {"config_dir": str(project)}}),
                                       encoding="utf-8")
    if launch:                       # a Studio launch snapshot: the user's raw config, which has no scheme key
        (run / "studio").mkdir()
        raw = {k: copy.deepcopy(cfg[k]) for k in ("data", "predictors", "physics")}
        raw["physics"].pop("advection_scheme", None)
        (run / "studio" / "launch.json").write_text(json.dumps({"config_raw": raw, "config_dir": str(project)}),
                                                    encoding="utf-8")
    return run


@pytest.mark.parametrize("launch", [False, True], ids=["manifest", "launch_snapshot"])
@pytest.mark.parametrize("recorded", [None, "hybrid", "exponential", "central"], ids=lambda s: f"recorded_{s}")
def test_config_for_run_and_studio_reader_keep_the_recorded_scheme(tmp_path, demo_project, recorded, launch):
    from sparc.core.session import config_for_run
    from sparc.studio.runs.reader import RunContext

    run = _run_dir(tmp_path, demo_project, recorded, launch=launch)
    want = recorded or "central"
    assert config_for_run(run).raw["physics"]["advection_scheme"] == want
    ctx = RunContext({"id": "r1", "run_dir": str(run)}, ("k",))
    assert ctx.cfg_source == ("launch" if launch else "manifest")
    assert ctx.cfg_raw["physics"]["advection_scheme"] == want


def test_reproduce_refits_a_pre_scheme_run_with_the_central_scheme(tmp_path, demo_project, monkeypatch):
    from sparc.core import pipeline
    from sparc.core.reproduce import reproduce

    seen = {}

    class Stop(Exception):
        pass

    def fake_run_core(cfg, **kw):
        seen["scheme"] = cfg.raw["physics"]["advection_scheme"]
        raise Stop

    monkeypatch.setattr(pipeline, "run_core", fake_run_core)
    for recorded, want in ((None, "central"), ("hybrid", "hybrid")):
        run = _run_dir(tmp_path / str(recorded), demo_project, recorded)
        with pytest.raises(Stop):
            reproduce(run)
        assert seen.pop("scheme") == want
