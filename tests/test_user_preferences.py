"""Tests for the global user preferences store."""

from __future__ import annotations

import json

import pytest

from sparc.config import user_preferences as up
from sparc.config import hardware_profile as hp


@pytest.fixture(autouse=True)
def _isolated_prefs_dir(tmp_path, monkeypatch):
    """Point SPARC_PREFS_DIR at a temp directory so tests cannot pollute the user's home."""
    monkeypatch.setenv("SPARC_PREFS_DIR", str(tmp_path))
    hp.set_runtime_overrides(None)
    hp.reset_profile_cache()
    yield
    hp.set_runtime_overrides(None)
    hp.reset_profile_cache()


def test_get_prefs_dir_honors_env_var(tmp_path):
    assert up.get_prefs_dir() == tmp_path
    assert up.get_prefs_path() == tmp_path / "preferences.json"


def test_load_returns_empty_when_missing():
    assert up.load_preferences() == {}


def test_save_then_load_roundtrip():
    payload = {"performance": {"preset": "performance", "max_workers": 4}}
    path = up.save_preferences(payload)
    assert path.exists()
    assert up.load_preferences() == payload


def test_load_returns_empty_on_invalid_json(tmp_path):
    (tmp_path / "preferences.json").write_text("{ not json", encoding="utf-8")
    assert up.load_preferences() == {}


def test_update_performance_merges_and_strips_none():
    up.save_preferences({"performance": {"preset": "balanced", "max_workers": 8}})
    merged = up.update_performance_preferences({"preset": "max", "max_workers": None, "batch_size": 4096})
    perf = merged["performance"]
    assert perf["preset"] == "max"
    assert perf["batch_size"] == 4096
    # max_workers was explicitly cleared via None and should be removed.
    assert "max_workers" not in perf


def test_update_performance_invalidates_hardware_cache(monkeypatch):
    # Patch RAM detection so the resolver works in CI.
    from collections import namedtuple
    VMem = namedtuple("VMem", ["total", "available", "percent"])
    vm = VMem(total=int(16 * (1024**3)), available=int(8 * (1024**3)), percent=50.0)
    monkeypatch.setattr(hp.psutil, "virtual_memory", lambda: vm)

    up.update_performance_preferences({"preset": "max"})
    profile = hp.detect_profile()
    assert profile.preset == "max"


def test_save_atomic_replaces_existing_file():
    up.save_preferences({"performance": {"preset": "eco"}})
    up.save_preferences({"performance": {"preset": "balanced"}})
    data = json.loads(up.get_prefs_path().read_text(encoding="utf-8"))
    assert data["performance"]["preset"] == "balanced"


def test_save_rejects_non_dict():
    with pytest.raises(TypeError):
        up.save_preferences("not a dict")  # type: ignore[arg-type]
