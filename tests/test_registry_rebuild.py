"""Tests for the database rebuild / migrate_from_disk functionality.

These tests verify that:
1. After pipeline artifacts are written to disk, migrate_from_disk picks them up.
2. The RunRegistry correctly persists and reloads from the JSON manifest.
3. The registry_rebuild helper in stream.py correctly updates state.registry.
"""
import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_paths(tmp_path: Path):
    """Build a PipelinePaths instance pointing at tmp_path."""
    from sparc.run.pipeline_paths import PipelinePaths

    cfg = {
        "paths": {"project_root": str(tmp_path)},
        "output": {
            "base_dir": str(tmp_path / "output"),
        },
    }
    return PipelinePaths.from_config(cfg)


def _get_run_registry():
    """Import RunRegistry directly (avoiding heavy CityRegistry transitive deps)."""
    from sparc.registry.run_registry import RunRegistry
    return RunRegistry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_migrate_from_disk_picks_up_gwen(tmp_path):
    """migrate_from_disk should register gwen_variable_importance.csv once
    it is written to Stage_1_GWEN."""
    RunRegistry = _get_run_registry()

    paths = _make_pipeline_paths(tmp_path)

    # Write a fake GWEN CSV to the expected location.
    gwen_csv = paths.stage1_dir / "gwen_variable_importance.csv"
    gwen_csv.write_text("variable,importance\nNDVI,0.42\nCanopy,0.31\n")

    reg = RunRegistry(paths.output_dir, autoload=False)
    # Initially empty.
    assert reg.lookup("1", "gwen_variable_importance") is None

    n = reg.migrate_from_disk(paths)

    entry = reg.lookup("1", "gwen_variable_importance")
    assert entry is not None, "GWEN artifact should be registered after migrate_from_disk"
    assert entry.partial is False
    assert n >= 1


def test_migrate_from_disk_does_not_duplicate(tmp_path):
    """Calling migrate_from_disk twice should not double-register artifacts."""
    RunRegistry = _get_run_registry()

    paths = _make_pipeline_paths(tmp_path)
    (paths.stage1_dir / "gwen_variable_importance.csv").write_text("variable,importance\nNDVI,0.1\n")

    reg = RunRegistry(paths.output_dir, autoload=False)
    n1 = reg.migrate_from_disk(paths)
    n2 = reg.migrate_from_disk(paths)

    assert n1 >= 1, "First scan should register at least one artifact"
    assert n2 == 0, "Second scan without force should register nothing new"


def test_manifest_persists_to_json(tmp_path):
    """Artifacts registered by migrate_from_disk should be saved to
    artifacts_manifest.json on disk so they survive a registry reload."""
    RunRegistry = _get_run_registry()

    paths = _make_pipeline_paths(tmp_path)
    corr_json = paths.stage0_dir / "correlogram_analysis_results.json"
    corr_json.write_text(json.dumps({"individual_results": {}}))

    reg = RunRegistry(paths.output_dir, autoload=False)
    reg.migrate_from_disk(paths)

    manifest_file = paths.output_dir / "artifacts_manifest.json"
    assert manifest_file.exists(), "artifacts_manifest.json must be written after migration"

    # Reload fresh registry and verify artifact survives.
    reg2 = RunRegistry(paths.output_dir, autoload=True)
    entry = reg2.lookup("0", "correlogram_results")
    assert entry is not None, "Artifact should survive a registry reload from disk"


def test_rebuild_registry_stream_helper(tmp_path):
    """The _rebuild_registry function from stream.py should call
    migrate_from_disk on state.registry and return a non-negative count."""
    RunRegistry = _get_run_registry()
    from sparc.server.stream import _rebuild_registry
    from sparc.server.state import ServerState

    paths = _make_pipeline_paths(tmp_path)
    (paths.stage1_dir / "gwen_variable_importance.csv").write_text("variable,importance\nNDVI,0.1\n")

    reg = RunRegistry(paths.output_dir, autoload=False)

    state = ServerState()
    state.project_config = {
        "paths": {"project_root": str(tmp_path)},
        "output": {"base_dir": str(tmp_path / "output")},
    }
    state.registry = reg

    count = _rebuild_registry(state)
    assert isinstance(count, int)
    assert count >= 1, "Should have registered at least the GWEN artifact"


def test_rebuild_registry_no_registry(tmp_path):
    """_rebuild_registry must not raise when state.registry is None."""
    from sparc.server.stream import _rebuild_registry
    from sparc.server.state import ServerState

    state = ServerState()
    state.project_config = {
        "paths": {"project_root": str(tmp_path)},
        "output": {"base_dir": str(tmp_path / "output")},
    }
    state.registry = None

    result = _rebuild_registry(state)
    assert result == 0
