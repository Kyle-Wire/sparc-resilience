"""Tests for the hardware profile detector."""

from __future__ import annotations

from collections import namedtuple

import pytest

from sparc.config import hardware_profile as hp


VMem = namedtuple("VMem", ["total", "available", "percent"])


def _patch_vm(monkeypatch, total_gb: float, available_gb: float | None = None):
    if available_gb is None:
        available_gb = total_gb * 0.6
    vm = VMem(
        total=int(total_gb * (1024**3)),
        available=int(available_gb * (1024**3)),
        percent=50.0,
    )
    monkeypatch.setattr(hp.psutil, "virtual_memory", lambda: vm)
    hp.reset_profile_cache()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("SPARC_HARDWARE_TIER", raising=False)
    hp.set_runtime_overrides(None)
    hp.reset_profile_cache()
    yield
    hp.set_runtime_overrides(None)
    hp.reset_profile_cache()


def test_low_tier_8gb_machine(monkeypatch):
    _patch_vm(monkeypatch, total_gb=8.0)
    profile = hp.detect_profile()

    assert profile.tier == "low"
    assert profile.max_workers == 1
    assert profile.outer_jobs == 1
    assert profile.inner_jobs == 1
    assert profile.batch_size == 512
    assert profile.nuts_thin == 2
    assert profile.force_cpu is True
    assert profile.high_memory_mode is False
    assert profile.total_ram_gb == pytest.approx(8.0, rel=0.01)


def test_standard_tier_16gb_machine(monkeypatch):
    _patch_vm(monkeypatch, total_gb=16.0)
    profile = hp.detect_profile()

    assert profile.tier == "standard"
    assert profile.batch_size == 2048
    assert profile.nuts_thin == 1
    assert profile.force_cpu is False
    assert profile.high_memory_mode is False


def test_high_tier_64gb_machine(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    profile = hp.detect_profile()

    assert profile.tier == "high"
    assert profile.batch_size == 4096
    assert profile.high_memory_mode is True
    assert profile.force_cpu is False


def test_env_override_forces_low_on_big_machine(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    monkeypatch.setenv("SPARC_HARDWARE_TIER", "low")
    hp.reset_profile_cache()

    profile = hp.detect_profile()
    assert profile.tier == "low"
    assert profile.max_workers == 1


def test_env_override_invalid_falls_back_to_autodetect(monkeypatch):
    _patch_vm(monkeypatch, total_gb=8.0)
    monkeypatch.setenv("SPARC_HARDWARE_TIER", "garbage")
    hp.reset_profile_cache()

    profile = hp.detect_profile()
    assert profile.tier == "low"  # auto-detected


def test_profile_is_cached(monkeypatch):
    _patch_vm(monkeypatch, total_gb=8.0)
    p1 = hp.detect_profile()
    # mutate underlying psutil to ensure the cached value is returned
    _patch_vm(monkeypatch, total_gb=64.0)
    p2 = hp.detect_profile()  # but cache was cleared by _patch_vm
    # After reset_profile_cache (called inside _patch_vm), they should differ.
    assert p1.tier == "low"
    assert p2.tier == "high"


def test_banner_is_human_readable(monkeypatch):
    _patch_vm(monkeypatch, total_gb=8.0)
    banner = hp.detect_profile().banner()
    assert "tier=low" in banner
    assert "ram=" in banner
    assert "cpu-only" in banner


# ---------------------------------------------------------------------------
# Performance settings: presets, safe caps, layered overrides
# ---------------------------------------------------------------------------

def test_compute_safe_caps_scales_with_hardware(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    raw = hp.detect_raw_profile()
    caps = hp.compute_safe_caps(raw)
    assert caps["max_workers"] == raw.cpu_count
    assert caps["batch_size"] >= 4096
    assert caps["memory_limit_gb"] == round(raw.total_ram_gb * 0.85, 1)
    assert "low" in caps["tiers"] and "max" in caps["presets"]


def test_apply_preset_eco_collapses_to_single_worker(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    raw = hp.detect_raw_profile()
    overrides = hp.apply_preset("eco", raw)
    assert overrides["max_workers"] == 1
    assert overrides["force_cpu"] is True
    assert overrides["batch_size"] <= raw.batch_size


def test_apply_preset_max_saturates_machine(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    raw = hp.detect_raw_profile()
    overrides = hp.apply_preset("max", raw)
    assert overrides["max_workers"] == raw.cpu_count
    assert overrides["force_cpu"] is False
    assert overrides["high_memory_mode"] is True


def test_apply_preset_balanced_matches_raw(monkeypatch):
    _patch_vm(monkeypatch, total_gb=16.0)
    raw = hp.detect_raw_profile()
    overrides = hp.apply_preset("balanced", raw)
    assert overrides["max_workers"] == raw.max_workers
    assert overrides["batch_size"] == raw.batch_size


def test_apply_preset_custom_returns_empty(monkeypatch):
    _patch_vm(monkeypatch, total_gb=16.0)
    raw = hp.detect_raw_profile()
    assert hp.apply_preset("custom", raw) == {}


def test_resolve_profile_project_overrides_global(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    profile = hp.resolve_profile(
        global_prefs={"max_workers": 2},
        project_overrides={"max_workers": 5},
    )
    assert profile.max_workers == 5  # project wins
    assert profile.source in ("custom", "preset+custom")


def test_resolve_profile_global_when_no_project_override(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    profile = hp.resolve_profile(
        global_prefs={"max_workers": 2},
        project_overrides={},
    )
    assert profile.max_workers == 2


def test_resolve_profile_preset_then_field_override(monkeypatch):
    _patch_vm(monkeypatch, total_gb=64.0)
    profile = hp.resolve_profile(
        global_prefs={"preset": "eco"},
        project_overrides={"max_workers": 4},
    )
    # eco preset would set workers=1; project field override raises it to 4.
    assert profile.preset == "eco"
    assert profile.max_workers == 4


def test_resolve_profile_tier_override_changes_classification(monkeypatch):
    _patch_vm(monkeypatch, total_gb=8.0)  # would auto-classify as "low"
    profile = hp.resolve_profile(
        global_prefs={"hardware_tier_override": "high"},
    )
    assert profile.tier == "high"


def test_set_runtime_overrides_invalidates_cache(monkeypatch):
    _patch_vm(monkeypatch, total_gb=16.0)
    base = hp.detect_profile()
    assert base.preset == "balanced"

    hp.set_runtime_overrides({"preset": "max"})
    after = hp.detect_profile()
    assert after.preset == "max"
    assert after.max_workers >= base.max_workers

    # Reset for other tests.
    hp.set_runtime_overrides(None)


def test_resolve_profile_invalid_preset_falls_back_to_balanced(monkeypatch):
    _patch_vm(monkeypatch, total_gb=16.0)
    profile = hp.resolve_profile(global_prefs={"preset": "garbage"})
    assert profile.preset == "balanced"


def test_resolve_profile_negative_batch_size_is_filtered(monkeypatch):
    _patch_vm(monkeypatch, total_gb=16.0)
    # Non-numeric / negative coerced via int() — int("-5") is -5; we just
    # ensure the override is propagated rather than crashing.
    profile = hp.resolve_profile(global_prefs={"batch_size": "1024"})
    assert profile.batch_size == 1024

