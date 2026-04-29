"""
Hardware profile detection for the SPARC pipeline.

Auto-detects available system RAM at startup and classifies the machine into a
performance tier: ``low`` (<12 GB), ``standard`` (12-32 GB), or ``high`` (>=32 GB).
The detected profile drives parallelism, batch sizes, MCMC trace thinning, and
device placement across the pipeline so that low-memory machines (e.g. an 8 GB
MacBook Pro) can complete a run without being killed by the OS.

The profile is a process-wide singleton; call :func:`detect_profile` from any
stage to get the same answer. The environment variable ``SPARC_HARDWARE_TIER``
may be set to ``low``, ``standard``, or ``high`` to override auto-detection
(intended for testing and CI, not user-facing).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

import psutil

Tier = Literal["low", "standard", "high"]

# Tier thresholds, in GiB of total system RAM.
_LOW_MAX_GB = 12.0
_HIGH_MIN_GB = 32.0


@dataclass(frozen=True)
class HardwareProfile:
    """Resolved hardware settings for the current process."""

    tier: Tier
    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    max_workers: int
    outer_jobs: int
    inner_jobs: int
    batch_size: int
    nuts_thin: int
    force_cpu: bool
    high_memory_mode: bool
    memory_limit_gb: float

    def as_dict(self) -> dict:
        return asdict(self)

    def banner(self) -> str:
        device = "cpu-only" if self.force_cpu else "gpu-ok"
        return (
            f"[hardware] tier={self.tier} ram={self.total_ram_gb:.1f}GB "
            f"workers={self.max_workers} batch={self.batch_size} {device}"
        )


def _classify(total_ram_gb: float) -> Tier:
    if total_ram_gb < _LOW_MAX_GB:
        return "low"
    if total_ram_gb >= _HIGH_MIN_GB:
        return "high"
    return "standard"


def _build_profile(tier: Tier, total_ram_gb: float, available_ram_gb: float) -> HardwareProfile:
    cpu_count = mp.cpu_count()

    if tier == "low":
        # Safety profile: drop parallelism to a single worker, shrink batch
        # sizes, force CPU, and thin MCMC traces. No quality knobs touched.
        return HardwareProfile(
            tier="low",
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_count=cpu_count,
            max_workers=1,
            outer_jobs=1,
            inner_jobs=1,
            batch_size=512,
            nuts_thin=2,
            force_cpu=True,
            high_memory_mode=False,
            memory_limit_gb=max(2.0, available_ram_gb * 0.6),
        )

    if tier == "high":
        return HardwareProfile(
            tier="high",
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_count=cpu_count,
            max_workers=min(cpu_count, 24),
            outer_jobs=min(cpu_count, 6),
            inner_jobs=max(1, cpu_count // 6),
            batch_size=4096,
            nuts_thin=1,
            force_cpu=False,
            high_memory_mode=True,
            memory_limit_gb=min(total_ram_gb * 0.75, 48.0),
        )

    # standard tier
    return HardwareProfile(
        tier="standard",
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_count=cpu_count,
        max_workers=min(cpu_count, 8),
        outer_jobs=min(cpu_count, 4),
        inner_jobs=max(1, cpu_count // 4),
        batch_size=2048,
        nuts_thin=1,
        force_cpu=False,
        high_memory_mode=False,
        memory_limit_gb=min(total_ram_gb * 0.7, 24.0),
    )


@lru_cache(maxsize=1)
def detect_profile() -> HardwareProfile:
    """Detect and cache the hardware profile for this process."""

    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    available_ram_gb = vm.available / (1024**3)

    override = os.environ.get("SPARC_HARDWARE_TIER", "").strip().lower()
    if override in ("low", "standard", "high"):
        tier: Tier = override  # type: ignore[assignment]
    else:
        tier = _classify(total_ram_gb)

    return _build_profile(tier, total_ram_gb, available_ram_gb)


def reset_profile_cache() -> None:
    """Clear the cached profile. Intended for tests only."""

    detect_profile.cache_clear()
