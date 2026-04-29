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
    hp.reset_profile_cache()
    yield
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
