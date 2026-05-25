"""
Hardware profile detection and resolution for the SPARC pipeline.

Auto-detects available system RAM at startup and classifies the machine into a
performance tier: ``low`` (<12 GB), ``standard`` (12-32 GB), or ``high`` (>=32 GB).
The detected profile drives parallelism, batch sizes, MCMC trace thinning, and
device placement across the pipeline so that low-memory machines (e.g. an 8 GB
MacBook Pro) can complete a run without being killed by the OS.

Resolution order (highest priority first):

1. ``SPARC_HARDWARE_TIER`` env var (low / standard / high).
2. Project YAML ``performance`` block (per-project override, travels with project).
3. Global user preferences from ``user_preferences`` (machine-wide default).
4. Auto-detected hardware tier (fallback).

The profile is a process-wide singleton; call :func:`detect_profile` from any
stage to get the same answer. Call :func:`reset_profile_cache` after mutating
preferences/overrides so the next read picks up the new values.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Optional

import psutil

Tier = Literal["low", "standard", "high"]
Preset = Literal["eco", "balanced", "performance", "max", "custom"]

VALID_TIERS: tuple[Tier, ...] = ("low", "standard", "high")
VALID_PRESETS: tuple[Preset, ...] = ("eco", "balanced", "performance", "max", "custom")

# Tier thresholds, in GiB of total system RAM.
_LOW_MAX_GB = 12.0
_HIGH_MIN_GB = 32.0


def _detect_gpu() -> tuple[bool, int, float, str]:
    """Return (gpu_available, gpu_count, gpu_vram_gb, gpu_name).

    Uses torch.cuda — returns all-zeros/False on CPU-only machines or when
    torch is unavailable.  Wrapped in a broad try/except so hardware quirks
    never crash the profiler.
    """
    try:
        import torch  # lazy import — torch may not be installed yet
        if not torch.cuda.is_available():
            return False, 0, 0.0, ""
        count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
        return True, count, vram_gb, props.name
    except Exception:  # pragma: no cover
        return False, 0, 0.0, ""


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
    preset: Preset = "balanced"
    source: str = "auto"  # "auto", "preset", "custom", "env"
    # GPU fields (CU-1)
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_vram_gb: float = 0.0
    gpu_name: str = ""
    # GPU-tuned batch size (CU-3); 0 when no GPU detected
    gpu_batch_size: int = 0
    # CU-9b: CUDA Graph capture for static-shape batch loops (default off)
    cuda_graphs: bool = False
    # CU-10a: Multi-GPU fold-parallel training via DDP (default off)
    ddp_enabled: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def banner(self) -> str:
        if self.force_cpu:
            device_tag = "cpu-only"
        elif self.gpu_available:
            device_tag = f"gpu={self.gpu_name}({self.gpu_vram_gb:.1f}GB)"
        else:
            device_tag = "gpu-ok"
        return (
            f"[hardware] tier={self.tier} ram={self.total_ram_gb:.1f}GB "
            f"workers={self.max_workers} batch={self.batch_size} {device_tag} "
            f"preset={self.preset}"
        )

    @classmethod
    def stub(cls) -> "HardwareProfile":
        """Return a deterministic minimal profile for tests — never calls psutil."""
        return cls(
            tier="low",
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            cpu_count=2,
            max_workers=1,
            outer_jobs=1,
            inner_jobs=1,
            batch_size=512,
            nuts_thin=2,
            force_cpu=True,
            high_memory_mode=False,
            memory_limit_gb=6.8,
            preset="eco",
            source="stub",
        )


# ---------------------------------------------------------------------------
# Tier classification + base profile builder (raw detection)
# ---------------------------------------------------------------------------

def _classify(total_ram_gb: float) -> Tier:
    if total_ram_gb < _LOW_MAX_GB:
        return "low"
    if total_ram_gb >= _HIGH_MIN_GB:
        return "high"
    return "standard"


def _build_profile(tier: Tier, total_ram_gb: float, available_ram_gb: float) -> HardwareProfile:
    """Construct the auto-detected HardwareProfile for *tier*."""
    cpu_count = mp.cpu_count()
    gpu_avail, gpu_count, gpu_vram_gb, gpu_name = _detect_gpu()

    # GPU-aware batch size: ~256 rows per GB VRAM (12 KB/row at hidden_dim=256).
    # Clamped to [512, 32768].
    gpu_bs = int(min(max(gpu_vram_gb * 256, 512), 32768)) if gpu_avail else 0

    if tier == "low":
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
            force_cpu=not gpu_avail,  # CU-1: decouple from RAM tier
            high_memory_mode=False,
            memory_limit_gb=max(2.0, available_ram_gb * 0.6),
            preset="balanced",
            source="auto",
            gpu_available=gpu_avail,
            gpu_count=gpu_count,
            gpu_vram_gb=gpu_vram_gb,
            gpu_name=gpu_name,
            gpu_batch_size=gpu_bs,
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
            preset="balanced",
            source="auto",
            gpu_available=gpu_avail,
            gpu_count=gpu_count,
            gpu_vram_gb=gpu_vram_gb,
            gpu_name=gpu_name,
            gpu_batch_size=gpu_bs,
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
        preset="balanced",
        source="auto",
        gpu_available=gpu_avail,
        gpu_count=gpu_count,
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=gpu_name,
        gpu_batch_size=gpu_bs,
    )


def detect_raw_profile() -> HardwareProfile:
    """Return the unfiltered hardware-derived profile (no overrides applied).

    The ``SPARC_HARDWARE_TIER`` env var still selects the tier here so that
    test harnesses and CI can exercise specific tiers without touching real RAM.
    """
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    available_ram_gb = vm.available / (1024**3)

    override = os.environ.get("SPARC_HARDWARE_TIER", "").strip().lower()
    if override in VALID_TIERS:
        tier: Tier = override  # type: ignore[assignment]
    else:
        tier = _classify(total_ram_gb)

    return _build_profile(tier, total_ram_gb, available_ram_gb)


# ---------------------------------------------------------------------------
# Safe caps — advisory upper bounds used by the warning system
# ---------------------------------------------------------------------------

def compute_safe_caps(raw: HardwareProfile) -> dict[str, Any]:
    """Return advisory upper bounds for each tunable based on the detected hardware.

    Values above these caps will trigger an inline warning + confirmation modal
    in the desktop UI but are NOT hard-clamped — the user retains override.
    """
    cpu = max(1, raw.cpu_count)
    ram = max(1.0, raw.total_ram_gb)
    return {
        "max_workers": cpu,
        "outer_jobs": cpu,
        "inner_jobs": cpu,
        "batch_size": 16384,
        "nuts_thin": 8,
        "memory_limit_gb": round(ram * 0.85, 1),
        "tiers": list(VALID_TIERS),
        "presets": list(VALID_PRESETS),
    }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def apply_preset(preset: Preset, raw: HardwareProfile) -> dict[str, Any]:
    """Map a preset name onto a set of overrides relative to *raw*.

    Returns a dict of field overrides (subset of HardwareProfile fields).
    "custom" returns an empty dict — the caller is expected to supply
    field-level overrides directly.
    """
    if preset == "custom":
        return {}

    cpu = max(1, raw.cpu_count)
    ram = max(1.0, raw.total_ram_gb)

    if preset == "eco":
        # Use a small slice of resources; safe to run alongside other apps.
        return {
            "max_workers": 1,
            "outer_jobs": 1,
            "inner_jobs": 1,
            "batch_size": 512,
            "nuts_thin": 2,
            "force_cpu": True,
            "high_memory_mode": False,
            "memory_limit_gb": round(min(ram * 0.35, 4.0), 1),
        }

    if preset == "balanced":
        # Mirror the auto-detected defaults for the raw profile.
        return {
            "max_workers": raw.max_workers,
            "outer_jobs": raw.outer_jobs,
            "inner_jobs": raw.inner_jobs,
            "batch_size": raw.batch_size,
            "nuts_thin": raw.nuts_thin,
            "force_cpu": raw.force_cpu,
            "high_memory_mode": raw.high_memory_mode,
            "memory_limit_gb": raw.memory_limit_gb,
        }

    if preset == "performance":
        # Push closer to the safe caps but leave headroom for the OS.
        return {
            "max_workers": min(cpu, max(raw.max_workers + 2, int(cpu * 0.75))),
            "outer_jobs": min(cpu, max(raw.outer_jobs + 1, int(cpu * 0.5))),
            "inner_jobs": max(1, cpu // 4),
            "batch_size": max(raw.batch_size, 2048) * 2,
            "nuts_thin": 1,
            "force_cpu": False,
            "high_memory_mode": True,
            "memory_limit_gb": round(min(ram * 0.8, 64.0), 1),
        }

    # "max" — saturate the machine; assume the user has nothing else running.
    return {
        "max_workers": cpu,
        "outer_jobs": cpu,
        "inner_jobs": max(1, cpu // 2),
        "batch_size": 8192,
        "nuts_thin": 1,
        "force_cpu": False,
        "high_memory_mode": True,
        "memory_limit_gb": round(ram * 0.85, 1),
    }


# ---------------------------------------------------------------------------
# Override merging + resolution
# ---------------------------------------------------------------------------

_OVERRIDABLE_FIELDS = (
    "max_workers",
    "outer_jobs",
    "inner_jobs",
    "batch_size",
    "nuts_thin",
    "force_cpu",
    "high_memory_mode",
    "memory_limit_gb",
    "cuda_graphs",
    "ddp_enabled",
)


def _coerce_overrides(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Filter *raw* to known overridable fields with light type coercion."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key in _OVERRIDABLE_FIELDS:
        if key not in raw or raw[key] is None:
            continue
        val = raw[key]
        if key in ("force_cpu", "high_memory_mode", "cuda_graphs", "ddp_enabled"):
            out[key] = bool(val)
        elif key == "memory_limit_gb":
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                continue
        else:
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                continue
    return out


def _apply_overrides(profile: HardwareProfile, overrides: dict[str, Any]) -> HardwareProfile:
    if not overrides:
        return profile
    return replace(profile, **{k: v for k, v in overrides.items() if k in _OVERRIDABLE_FIELDS})


def resolve_profile(
    *,
    global_prefs: Optional[dict[str, Any]] = None,
    project_overrides: Optional[dict[str, Any]] = None,
) -> HardwareProfile:
    """Merge raw detection with optional global + per-project overrides.

    Resolution order (later wins):
        raw → global_prefs → project_overrides → SPARC_HARDWARE_TIER env

    Each layer may include:
        - ``hardware_tier_override``: ``low`` | ``standard`` | ``high``
        - ``preset``: ``eco`` | ``balanced`` | ``performance`` | ``max`` | ``custom``
        - field overrides (max_workers, batch_size, etc.)
    """
    global_prefs = global_prefs or {}
    project_overrides = project_overrides or {}

    # Step 1: pick the tier. Project > global > env > auto-detected.
    tier_override = (
        project_overrides.get("hardware_tier_override")
        or global_prefs.get("hardware_tier_override")
    )
    raw = detect_raw_profile()
    if isinstance(tier_override, str) and tier_override.lower() in VALID_TIERS:
        raw = _build_profile(
            tier_override.lower(),  # type: ignore[arg-type]
            raw.total_ram_gb,
            raw.available_ram_gb,
        )

    # Step 2: apply preset (project preset wins over global).
    preset = (
        project_overrides.get("preset")
        or global_prefs.get("preset")
        or "balanced"
    )
    if preset not in VALID_PRESETS:
        preset = "balanced"

    preset_overrides = apply_preset(preset, raw)  # type: ignore[arg-type]
    profile = _apply_overrides(raw, preset_overrides)

    # Step 3: layer global prefs field overrides, then project overrides.
    profile = _apply_overrides(profile, _coerce_overrides(global_prefs))
    profile = _apply_overrides(profile, _coerce_overrides(project_overrides))

    # Step 4: stamp metadata.
    has_field_overrides = bool(
        _coerce_overrides(global_prefs) or _coerce_overrides(project_overrides)
    )
    if has_field_overrides and preset != "custom":
        source = "preset+custom"
    elif preset == "custom" or has_field_overrides:
        source = "custom"
    elif preset != "balanced" or tier_override:
        source = "preset"
    else:
        source = "auto"

    return replace(profile, preset=preset, source=source)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cached singleton accessor (public API)
# ---------------------------------------------------------------------------

_CACHED_PROFILE: Optional[HardwareProfile] = None
_RUNTIME_OVERRIDES: dict[str, Any] = {}


def set_runtime_overrides(overrides: Optional[dict[str, Any]]) -> None:
    """Set per-process runtime overrides (e.g. from a project just loaded by the server).

    Clears the cached profile so the next :func:`detect_profile` call re-resolves.
    """
    global _RUNTIME_OVERRIDES, _CACHED_PROFILE
    _RUNTIME_OVERRIDES = dict(overrides or {})
    _CACHED_PROFILE = None


def detect_profile() -> HardwareProfile:
    """Detect and cache the hardware profile for this process.

    Pulls global preferences from :mod:`sparc.config.user_preferences` and
    project overrides from any runtime overrides set via
    :func:`set_runtime_overrides`. Cached until :func:`reset_profile_cache`
    is called.
    """
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None:
        return _CACHED_PROFILE

    # Lazy import to avoid a circular dep with the config package.
    try:
        from sparc.config.user_preferences import load_preferences
        prefs = load_preferences()
        # Allow either a flat overrides dict or a {"performance": {...}} envelope.
        if isinstance(prefs.get("performance"), dict):
            global_prefs = prefs["performance"]
        else:
            global_prefs = prefs
    except Exception:  # noqa: BLE001 - defensive: never fail boot on a bad prefs file
        global_prefs = {}

    _CACHED_PROFILE = resolve_profile(
        global_prefs=global_prefs,
        project_overrides=_RUNTIME_OVERRIDES,
    )
    return _CACHED_PROFILE


def reset_profile_cache() -> None:
    """Clear the cached profile. Call after mutating prefs or overrides."""
    global _CACHED_PROFILE
    _CACHED_PROFILE = None
