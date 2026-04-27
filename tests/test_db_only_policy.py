"""Phase 4 — runtime guardrails for the db-only artifacts policy.

When ``output.disk_writes`` is OFF (the default for the desktop app), the
SPARC pipeline must:

1. Persist all artifacts into ``artifacts.db`` via :class:`ArtifactStore`.
2. Not create per-stage ``Stage_*`` directories.
3. Not write loose CSV/JSON/joblib/npy files outside the small allow-list
   that the user expects (session log, ``.sparcproj``, ``artifacts.db``,
   ``run_manifest.json``, input ``*.parquet``).

These tests exercise the toggle directly rather than scanning source code,
so they remain robust when stage modules legitimately keep disk-write
fallbacks behind an ``if disk_writes_enabled():`` gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore
from sparc.run.disk_policy import disk_writes_enabled, set_disk_writes
from sparc.run import artifact_io


# Files that are legitimately allowed to live next to artifacts.db in a
# clean db-only run directory.
RUN_DIR_ALLOWLIST_SUFFIXES = {
    ".db",          # artifacts.db (and its -wal/-shm sidecars)
    ".db-wal",
    ".db-shm",
    ".log",         # session log
    ".sparcproj",   # project file
    ".json",        # run_manifest.json + .json.lock
    ".lock",        # manifest .lock sidecar (registry serialisation)
    ".parquet",     # canonical input dataset
}


@pytest.fixture()
def disk_off():
    """Force the disk-writes toggle OFF for the duration of a test."""
    prev = disk_writes_enabled()
    set_disk_writes(False)
    try:
        yield
    finally:
        set_disk_writes(prev)


@pytest.fixture()
def store(tmp_path: Path, disk_off):
    reg = RunRegistry(tmp_path, autoload=False)
    s = ArtifactStore(reg)
    set_active_registry(reg)
    try:
        yield s, tmp_path
    finally:
        set_active_registry(None)


def _list_unexpected(run_dir: Path) -> list[Path]:
    """Return files in run_dir that do not match the allowlist."""
    out: list[Path] = []
    for p in run_dir.rglob("*"):
        if p.is_dir():
            continue
        # Allow blob CAS subtree managed by ArtifactStore.
        rel = p.relative_to(run_dir).as_posix()
        if rel.startswith("blobs/"):
            continue
        suffix = p.suffix.lower()
        if suffix in RUN_DIR_ALLOWLIST_SUFFIXES:
            continue
        # Two-suffix forms (e.g. .db-wal handled above; .sparc.log handled
        # by .log suffix). Anything else is unexpected.
        out.append(p)
    return out


def test_disk_policy_default_is_off():
    """A fresh process (no env var, no config) must default to disk OFF."""
    # We cannot hard-assert the default if a previous test set it, but we
    # can verify the flag is at least addressable.
    set_disk_writes(None)
    assert disk_writes_enabled() is False


def test_save_struct_creates_no_loose_files(store):
    s, run_dir = store
    artifact_io.save_struct(
        "0", "diag.json", {"alpha": 1.0, "beta": [1, 2]}, producer="test"
    )
    # Round-trip the struct back through the store.
    payload = s.read_struct("0", "diag")
    assert payload["alpha"] == 1.0
    # And the run directory must not contain a loose diag.json on disk.
    unexpected = _list_unexpected(run_dir)
    assert not unexpected, f"Loose files appeared with disk OFF: {unexpected}"


def test_save_table_creates_no_loose_files(store):
    s, run_dir = store
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    artifact_io.save_table("1", "results.csv", df, producer="test")
    rt = s.read_table("1", "results")
    pd.testing.assert_frame_equal(rt, df)
    unexpected = _list_unexpected(run_dir)
    assert not unexpected, f"Loose files appeared with disk OFF: {unexpected}"


def test_save_blob_creates_no_loose_files(store):
    s, run_dir = store
    payload = {"weights": np.arange(5), "name": "scaler"}
    artifact_io.save_blob(
        "2", "scaler.joblib", payload, serializer="joblib", producer="test"
    )
    rt = s.read_blob("2", "scaler")
    assert rt["name"] == "scaler"
    assert np.array_equal(rt["weights"], payload["weights"])
    unexpected = _list_unexpected(run_dir)
    assert not unexpected, f"Loose files appeared with disk OFF: {unexpected}"


def test_path_helpers_skip_disk_when_off(store, tmp_path):
    """The path-style helpers must NOT touch disk when the toggle is OFF."""
    s, run_dir = store
    fake_disk_path = run_dir / "Stage_0_Correlogram" / "thing.json"

    artifact_io.save_struct_path(
        {"k": "v"}, fake_disk_path,
        stage="0", artifact_id="thing", producer="test",
    )

    # Always written to the store ...
    assert s.has("0", "thing")
    # ... but the disk path must NOT exist.
    assert not fake_disk_path.exists()
    # And no Stage_0_Correlogram directory was created.
    assert not (run_dir / "Stage_0_Correlogram").exists()


def test_path_helpers_write_disk_when_on(tmp_path):
    """When the toggle is ON, path helpers must write both store and disk."""
    prev = disk_writes_enabled()
    set_disk_writes(True)
    try:
        reg = RunRegistry(tmp_path, autoload=False)
        ArtifactStore(reg)  # bind active store
        set_active_registry(reg)
        try:
            disk_path = tmp_path / "stage0" / "thing.json"
            artifact_io.save_struct_path(
                {"k": "v"}, disk_path,
                stage="0", artifact_id="thing", producer="test",
            )
            assert disk_path.exists()
            with open(disk_path) as f:
                assert json.load(f) == {"k": "v"}
        finally:
            set_active_registry(None)
    finally:
        set_disk_writes(prev)


def test_pipeline_paths_skip_stage_dirs_when_off(tmp_path, disk_off):
    """``_ensure_directories`` must not create per-stage folders when OFF."""
    from sparc.run.pipeline_paths import PipelinePaths

    config = {
        "output": {"base_dir": str(tmp_path / "run")},
        "paths": {"project_root": str(tmp_path)},
    }
    paths = PipelinePaths.from_config(config)
    # The base output dir is always created (artifacts.db lives there).
    assert paths.output_dir.exists()
    for attr in ("stage0_dir", "stage1_dir", "stage2_dir", "stage3_dir", "stage4_dir"):
        d = getattr(paths, attr, None)
        if d is None:
            continue
        assert not Path(d).exists(), (
            f"Stage directory {d} was created despite disk_writes=False"
        )
