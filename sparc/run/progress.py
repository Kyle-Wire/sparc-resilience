"""Pipeline-level stage progress tracker for the SPARC CLI."""
from __future__ import annotations

import sys
import time
from typing import Callable

from tqdm import tqdm

_STAGE_LABELS: dict[str, str] = {
    "0":  "Correlogram Analysis",
    "0b": "Pipeline Configuration",
    "1":  "GWEN Variable Selection",
    "2":  "Spatial CV + Neural Training",
    "3":  "Causal Validation",
    "4":  "Scenario Simulation",
    "5":  "Report Generation",
}

# Rough Stage 2 ETA table: (size_tier, has_gpu) → seconds
_ETA_TABLE: dict[tuple[str, bool], int] = {
    ("small",  False): 90,
    ("small",  True):  30,
    ("medium", False): 480,
    ("medium", True):  120,
    ("large",  False): 1800,
    ("large",  True):  360,
    ("xlarge", False): 5400,
    ("xlarge", True):  900,
}


class StageProgress:
    """Pipeline-level stage tracker.

    In a TTY, a tqdm bar shows [n/total stages] with elapsed / remaining.
    In a pipe (CI, log redirect), plain timestamped print() lines are used.
    """

    def __init__(
        self,
        total_stages: int,
        *,
        run_t0: float | None = None,
        timing_sink: Callable[[str, float], None] | None = None,
    ) -> None:
        self._total = total_stages
        self._done = 0
        self._run_t0 = run_t0 if run_t0 is not None else time.perf_counter()
        self._stage_t0: float | None = None
        self._current_key: str | None = None
        self._stage_times: dict[str, float] = {}
        self._timing_sink = timing_sink
        self._is_tty = sys.stdout.isatty()
        # No persistent tqdm bar: a live bar at position=0 interferes with
        # print() calls inside sub-stages (e.g. correlogram_analysis), causing
        # all progress output to be overwritten and appear as blank lines.
        # tqdm.write() is used throughout so output is compatible with any
        # per-variable tqdm bars that sub-stages open themselves.
        self._bar = None

    def stage_start(self, key: str, label: str | None = None) -> None:
        lbl = label or _STAGE_LABELS.get(key, key)
        self._current_key = key
        self._stage_t0 = time.perf_counter()
        elapsed = time.perf_counter() - self._run_t0
        tqdm.write(f"\n[{elapsed:6.1f}s] >>> Stage {key}: {lbl}")

    def stage_done(self, key: str) -> None:
        if self._stage_t0 is None:
            return
        elapsed = time.perf_counter() - self._stage_t0
        self._stage_times[key] = elapsed
        self._done += 1
        run_elapsed = time.perf_counter() - self._run_t0
        # "Stage N: complete in Xs" matches the _ALWAYS_PATTERNS regex
        # r"\bStage\s+\d+\s*:" so this line is never filtered out.
        tqdm.write(
            f"[{run_elapsed:6.1f}s]    Stage {key}: complete in {elapsed:.0f}s"
            f"  [{self._done}/{self._total} stages]"
        )
        self._current_key = None
        self._stage_t0 = None
        if self._timing_sink is not None:
            try:
                self._timing_sink(key, elapsed)
            except Exception:
                pass

    def print_eta_hint(self, size_tier: str, has_gpu: bool) -> None:
        """Print a rough Stage 2 ETA hint from the lookup table."""
        est_s = _ETA_TABLE.get((size_tier.lower(), has_gpu))
        if est_s is None:
            return
        units = f"{est_s // 60} min" if est_s >= 60 else f"{est_s}s"
        device_label = "GPU" if has_gpu else "CPU"
        tqdm.write(
            f"  [ETA hint] Stage 2 estimated ~{units}"
            f" ({size_tier} dataset, {device_label})"
            f" — rough estimate, refines after first run"
        )

    def finish(self) -> None:
        total_s = time.perf_counter() - self._run_t0
        timing = "  ".join(
            f"Stage {k}: {t:.0f}s"
            for k, t in sorted(self._stage_times.items())
        )
        tqdm.write(
            f"\nPipeline complete — {total_s:.1f}s ({total_s / 60:.1f} min)"
        )
        if timing:
            tqdm.write(f"  {timing}")
