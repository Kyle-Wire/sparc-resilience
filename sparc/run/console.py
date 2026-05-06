"""Central terminal verbosity gating for the SPARC CLI.

Three modes:

- ``summary``: stage banners, stage completion lines, errors, warnings,
  hardware banner, and per-stage summary blocks. Quiet otherwise.
- ``normal``  (default): everything in ``summary`` plus checkpoints
  (info / success / warn) and key metric lines.
- ``debug``:   full firehose -- the previous (pre-filter) behavior.

The filter is implemented as a thin ``io.TextIOBase`` wrapper installed on
``sys.stdout`` / ``sys.stderr`` at CLI startup. Each emitted line is
classified by regex (mirroring patterns already used in
``sparc.server.stream._EventCapture``) and dropped if its category falls
below the configured verbosity threshold.

The classifier never blocks the FastAPI WebSocket / desktop event stream:
that path uses :class:`sparc.server.stream._EventCapture` directly and
writes a complete event record to ``session_log.jsonl`` regardless of
the terminal verbosity. Verbosity affects ONLY what is printed to the
terminal.
"""

from __future__ import annotations

import io
import os
import re
import sys
from enum import IntEnum
from typing import TextIO


# ---------------------------------------------------------------------------
# Verbosity enum
# ---------------------------------------------------------------------------


class Verbosity(IntEnum):
    """Terminal verbosity levels (higher = more output)."""

    SUMMARY = 0
    NORMAL = 1
    DEBUG = 2

    @classmethod
    def parse(cls, value: str | int | "Verbosity" | None) -> "Verbosity":
        if value is None:
            return cls.NORMAL
        if isinstance(value, Verbosity):
            return value
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return cls.NORMAL
        norm = str(value).strip().lower()
        if norm in {"summary", "s", "quiet", "0"}:
            return cls.SUMMARY
        if norm in {"normal", "n", "info", "1"}:
            return cls.NORMAL
        if norm in {"debug", "d", "verbose", "v", "2", "3"}:
            return cls.DEBUG
        return cls.NORMAL


def get_verbosity() -> Verbosity:
    """Return the verbosity level currently configured via env var."""
    return Verbosity.parse(os.environ.get("SPARC_VERBOSITY"))


def set_verbosity(level: Verbosity | str | int | None) -> Verbosity:
    """Set verbosity via the ``SPARC_VERBOSITY`` env var so subprocesses inherit it."""
    v = Verbosity.parse(level)
    os.environ["SPARC_VERBOSITY"] = v.name.lower()
    return v


# ---------------------------------------------------------------------------
# Line classifier
# ---------------------------------------------------------------------------


class LineCategory(IntEnum):
    """Minimum verbosity required to print a given line."""

    ALWAYS = 0       # stage banners, errors, warnings, summary blocks
    CHECKPOINT = 1   # checkpoints, key metrics -- shown at NORMAL+
    DEBUG = 2        # raw logs, NUTS progress, warmup ticks -- DEBUG only


# Patterns that should always reach the terminal regardless of verbosity.
_ALWAYS_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*>>>\s*Stage\s+\d", re.IGNORECASE),
    re.compile(r"^\s*>>>\s*"),                        # any ">>> Phase" banner
    re.compile(r"^\s*===.*===\s*$"),                  # banner separators
    re.compile(r"\bStage\s+\d+\s*:", re.IGNORECASE),  # "Stage 3: Causal Validation"
    re.compile(r"\b(error|failed|traceback|exception)\b", re.IGNORECASE),
    re.compile(r"\[(WARN|WARNING|ERROR)\]"),
    re.compile(r"^\[hardware\]"),
    re.compile(r"^\s*\[SUMMARY\]"),                   # explicit summary marker
    re.compile(r"complete in\s+[\d.]+\s*[smh]"),      # stage timing summary
    re.compile(r"\bSPARC\b.*\bv?\d+\.\d", re.IGNORECASE),  # version banner
)

# Checkpoint-level patterns (shown at NORMAL and above).
_CHECKPOINT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*\[OK\]"),
    re.compile(r"\b(R[\u00b22])\s*=\s*[\d.]+", re.IGNORECASE),
    re.compile(r"\b(RMSE|MAE|MAPE|F1|SHD|precision|recall)\s*=\s*[\d.]+", re.IGNORECASE),
    re.compile(r"^\s*\[CURRICULUM\]"),
    re.compile(r"^\s*\[CONVERGENCE\]"),
    re.compile(r"^\s*\[MODEL_(START|DONE)\]"),
    re.compile(r"\bComputing correlogram for\b"),
    re.compile(r"\bSelected\s+\d+\s+predictors\b"),
    re.compile(r"\bGWEN\b.*\b(fitted|complete)\b", re.IGNORECASE),
    re.compile(r"\bSpatial CV Complete\b", re.IGNORECASE),
    re.compile(r"\bCorrelogram analysis complete\b", re.IGNORECASE),
    re.compile(r"\bDiscovery report saved\b", re.IGNORECASE),
    re.compile(r"\bd-separation:\s*\d", re.IGNORECASE),
    re.compile(r"\bFold\s+\d+\s*/\s*\d+\b", re.IGNORECASE),
    re.compile(r"^\s*Pipeline configuration saved"),
    re.compile(r"^\s*\[(checkpoint|metric)\]", re.IGNORECASE),
)

# Patterns that explicitly mark the noisiest debug-only lines. Listed
# first so we can short-circuit before falling through to the heuristic
# default (which would otherwise tag plain logs as DEBUG anyway).
_DEBUG_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*warmup\s+\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*NUTS sample\s+\d+\s*/\s*\d+"),
    re.compile(r"^\s*NUTS:\s*finding initial step size"),
    re.compile(r"^\s*\d+\s*%\|"),                    # tqdm bars
)


def classify_line(line: str) -> LineCategory:
    """Decide the minimum verbosity required to print *line*."""
    if not line or not line.strip():
        return LineCategory.DEBUG
    for pat in _ALWAYS_PATTERNS:
        if pat.search(line):
            return LineCategory.ALWAYS
    for pat in _DEBUG_PATTERNS:
        if pat.search(line):
            return LineCategory.DEBUG
    for pat in _CHECKPOINT_PATTERNS:
        if pat.search(line):
            return LineCategory.CHECKPOINT
    # Default: treat as raw debug log.
    return LineCategory.DEBUG


# ---------------------------------------------------------------------------
# Stdout filter
# ---------------------------------------------------------------------------


class _VerbosityFilter(io.TextIOBase):
    """``sys.stdout`` wrapper that drops lines below the verbosity threshold.

    Buffers partial lines so multi-call ``print(..., end="")`` and tqdm-style
    carriage-return updates still work.
    """

    def __init__(self, downstream: TextIO):
        self._downstream = downstream
        self._buffer = ""

    # ------------------------------------------------------------------
    # io.TextIOBase interface
    # ------------------------------------------------------------------

    def writable(self) -> bool:  # pragma: no cover - trivial
        return True

    def isatty(self) -> bool:
        try:
            return self._downstream.isatty()
        except Exception:
            return False

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._downstream, "encoding", "utf-8")

    def fileno(self) -> int:  # type: ignore[override]
        return self._downstream.fileno()

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        # Treat both \n and \r as line terminators (tqdm uses \r).
        while True:
            nl = self._buffer.find("\n")
            cr = self._buffer.find("\r")
            cuts = [c for c in (nl, cr) if c >= 0]
            if not cuts:
                break
            cut = min(cuts)
            line = self._buffer[:cut]
            self._buffer = self._buffer[cut + 1:]
            self._emit(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        try:
            self._downstream.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _emit(self, line: str) -> None:
        category = classify_line(line.rstrip("\r\n"))
        if int(get_verbosity()) >= int(category):
            try:
                self._downstream.write(line)
                self._downstream.flush()
            except Exception:
                pass


# Track installation so install() is idempotent and uninstall() is safe.
_INSTALLED: dict[str, TextIO] = {}


def install(level: Verbosity | str | int | None = None) -> Verbosity:
    """Install the verbosity filter on ``sys.stdout`` / ``sys.stderr``.

    Idempotent: calling twice updates the level but does not stack filters.
    Returns the resolved ``Verbosity`` level.
    """
    v = set_verbosity(level)
    if v == Verbosity.DEBUG:
        # Debug mode keeps the original streams -- no filtering at all.
        uninstall()
        return v
    if "stdout" not in _INSTALLED:
        _INSTALLED["stdout"] = sys.stdout
        sys.stdout = _VerbosityFilter(sys.stdout)  # type: ignore[assignment]
    if "stderr" not in _INSTALLED:
        _INSTALLED["stderr"] = sys.stderr
        sys.stderr = _VerbosityFilter(sys.stderr)  # type: ignore[assignment]
    return v


def uninstall() -> None:
    """Restore original streams (used by tests; rarely needed in production)."""
    if "stdout" in _INSTALLED:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = _INSTALLED.pop("stdout")
    if "stderr" in _INSTALLED:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stderr = _INSTALLED.pop("stderr")


# ---------------------------------------------------------------------------
# Helpers for explicit ALWAYS-level messages
# ---------------------------------------------------------------------------


def summary(message: str) -> None:
    """Emit a stage / pipeline summary line that survives every verbosity."""
    line = f"[SUMMARY] {message}"
    # When a filter is installed, the [SUMMARY] tag matches _ALWAYS_PATTERNS
    # so the line passes through. When not installed, behaves as plain print.
    print(line, flush=True)


def warn(message: str) -> None:
    """Emit a warning that survives every verbosity."""
    print(f"[WARN] {message}", flush=True)
