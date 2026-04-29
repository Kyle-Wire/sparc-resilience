"""Tests for the runtime memory watchdog."""

from __future__ import annotations

import time
from collections import namedtuple
from itertools import chain, repeat

import pytest

from sparc.run import memory_watchdog as mw


VMem = namedtuple("VMem", ["total", "available", "percent"])


def _vmem_sequence(monkeypatch, percents):
    """Patch psutil.virtual_memory to return a sequence of percent values, holding the last."""
    seq = chain(percents, repeat(percents[-1]))

    def _vm():
        pct = next(seq)
        return VMem(total=8 * 1024**3, available=int(8 * 1024**3 * (1 - pct / 100)), percent=pct)

    monkeypatch.setattr(mw.psutil, "virtual_memory", _vm)


def test_warn_then_abort(monkeypatch):
    events: list[mw.WatchdogEvent] = []
    _vmem_sequence(monkeypatch, [50.0, 87.0, 95.0])

    wd = mw.MemoryWatchdog(
        stage_name="test",
        soft_percent=85.0,
        hard_percent=93.0,
        poll_interval=0.05,
        on_event=events.append,
    )
    with wd:
        # Wait long enough for watchdog to observe all three samples
        deadline = time.time() + 2.0
        while time.time() < deadline and not wd.aborted:
            time.sleep(0.05)
        assert wd.aborted, "watchdog should have aborted on 95% sample"
        with pytest.raises(mw.MemoryPressureAbort):
            wd.checkpoint()

    kinds = [e.kind for e in events]
    assert "warn" in kinds
    assert "abort" in kinds


def test_no_abort_under_thresholds(monkeypatch):
    _vmem_sequence(monkeypatch, [40.0, 50.0, 60.0])
    wd = mw.MemoryWatchdog(
        stage_name="ok",
        soft_percent=85.0,
        hard_percent=93.0,
        poll_interval=0.05,
    )
    with wd:
        time.sleep(0.3)
    assert not wd.aborted
    wd.checkpoint()  # should not raise


def test_thread_shuts_down_cleanly(monkeypatch):
    _vmem_sequence(monkeypatch, [50.0])
    wd = mw.MemoryWatchdog(stage_name="t", poll_interval=0.05)
    wd.start()
    assert wd._thread is not None and wd._thread.is_alive()
    wd.stop()
    assert wd._thread is None


def test_checkpoint_no_op_when_healthy(monkeypatch):
    _vmem_sequence(monkeypatch, [10.0])
    wd = mw.MemoryWatchdog(stage_name="t", poll_interval=0.05)
    with wd:
        wd.checkpoint()
        wd.checkpoint()
