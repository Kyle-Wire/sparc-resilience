"""sparc/run/stage_result.py — Typed result object for stage runners.

All stage runner ``main()`` functions return a :class:`StageResult` so the
pipeline orchestrator can distinguish success from silent failure without
parsing stdout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageResult:
    """Outcome of a single stage runner execution.

    Parameters
    ----------
    ok:
        ``True`` when the stage completed successfully, ``False`` if any
        unrecoverable error occurred.
    stage:
        Stage number (0–5).
    error:
        Human-readable error description when ``ok is False``, else ``None``.
    """

    ok: bool
    stage: int
    error: Optional[str] = None
