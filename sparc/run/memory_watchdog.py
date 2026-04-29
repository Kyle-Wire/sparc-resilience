"""
Background memory watchdog for long-running pipeline stages.

The watchdog polls system memory pressure on a daemon thread.  When the
system crosses a *soft* threshold (default 85% used) it emits a warning
and triggers ``gc.collect()`` plus PyTorch cache release.  When pressure
crosses a *hard* threshold (default 93%) it sets an abort event and the
monitored code path raises :class:`MemoryPressureAbort` so the stage can
exit cleanly before the OS issues an OOM kill.

Usage::

    from sparc.run.memory_watchdog import MemoryWatchdog

    with MemoryWatchdog(stage_name="stage2") as wd:
        for fold in folds:
            wd.checkpoint()  # raises MemoryPressureAbort if hard threshold tripped
            run_fold(fold)
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Optional

import psutil

logger = logging.getLogger(__name__)


class MemoryPressureAbort(RuntimeError):
    """Raised when the watchdog requests a clean stage abort."""


@dataclass
class WatchdogEvent:
    kind: str  # "warn" | "abort"
    percent: float
    message: str


class MemoryWatchdog(AbstractContextManager):
    """Daemon-thread monitor for system memory pressure."""

    def __init__(
        self,
        stage_name: str = "pipeline",
        *,
        soft_percent: float = 85.0,
        hard_percent: float = 93.0,
        poll_interval: float = 2.0,
        on_event: Optional[Callable[[WatchdogEvent], None]] = None,
    ) -> None:
        self.stage_name = stage_name
        self.soft_percent = soft_percent
        self.hard_percent = hard_percent
        self.poll_interval = poll_interval
        self.on_event = on_event

        self._abort_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._abort_message = ""
        self._last_peak = 0.0
        self._warned = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "MemoryWatchdog":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._abort_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"sparc-memwatch-{self.stage_name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval * 2)
            self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def aborted(self) -> bool:
        return self._abort_event.is_set()

    @property
    def peak_percent(self) -> float:
        return self._last_peak

    def checkpoint(self) -> None:
        """Raise :class:`MemoryPressureAbort` if a hard-threshold abort was requested."""
        if self._abort_event.is_set():
            raise MemoryPressureAbort(self._abort_message)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _emit(self, event: WatchdogEvent) -> None:
        if event.kind == "abort":
            logger.error(event.message)
        else:
            logger.warning(event.message)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # pragma: no cover - callback errors must not kill watchdog
                logger.exception("MemoryWatchdog on_event callback raised")

    def _release_caches(self) -> None:
        gc.collect()
        # Only touch torch if it's already imported — importing it here would
        # block the watchdog thread for seconds on first use and defeat the
        # purpose of fast cache release.
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best effort
                pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pct = float(psutil.virtual_memory().percent)
            except Exception:  # pragma: no cover - psutil failure
                if self._stop_event.wait(self.poll_interval):
                    break
                continue

            if pct > self._last_peak:
                self._last_peak = pct

            if pct >= self.hard_percent and not self._abort_event.is_set():
                self._abort_message = (
                    f"[memory] critical pressure ({pct:.1f}% used) during "
                    f"stage '{self.stage_name}' — aborting cleanly to avoid OS kill"
                )
                self._abort_event.set()
                self._emit(WatchdogEvent("abort", pct, self._abort_message))
                self._release_caches()
            elif pct >= self.soft_percent and not self._warned:
                self._warned = True
                msg = (
                    f"[memory] high pressure ({pct:.1f}% used) during "
                    f"stage '{self.stage_name}' — releasing caches"
                )
                self._emit(WatchdogEvent("warn", pct, msg))
                self._release_caches()
            elif pct < self.soft_percent - 5.0:
                # hysteresis: re-arm soft warning once memory eases
                self._warned = False

            if self._stop_event.wait(self.poll_interval):
                break
