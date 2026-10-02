"""Small helpers shared by the runs modules: cached JSON reads, finite numbers, plain-language estimates."""

from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["read_json_cached", "file_stat", "finite", "fnum", "clean", "likely", "unit_label", "fmt_temp",
           "jackknife_se", "mtime_iso", "LRU"]

_JSON_LOCK = threading.Lock()
_JSON: "OrderedDict[tuple, Any]" = OrderedDict()
_JSON_MAX = 512


def file_stat(path: str | os.PathLike) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` of a file, or None when it is missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def read_json_cached(path: str | os.PathLike, default: Any = None) -> Any:
    """Parsed JSON of ``path``, cached by ``(path, mtime_ns, size)`` (SPEC §6.2); ``default`` when unreadable.

    The cached object is shared: callers must not mutate it.
    """
    p = os.fspath(path)
    st = file_stat(p)
    if st is None:
        return default
    key = (p, *st)
    with _JSON_LOCK:
        hit = _JSON.get(key)
        if hit is not None:
            _JSON.move_to_end(key)
            return hit
    try:
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return default
    with _JSON_LOCK:
        _JSON[key] = obj
        while len(_JSON) > _JSON_MAX:
            _JSON.popitem(last=False)
    return obj


def mtime_iso(path: str | os.PathLike) -> str | None:
    from sparc.studio.workspace import utc_iso

    try:
        return utc_iso(os.stat(path).st_mtime)
    except OSError:
        return None


def finite(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def fnum(v, default=None):
    """``float(v)`` when finite, else ``default``."""
    return float(v) if finite(v) else default


def clean(obj):
    """JSON-safe copy: numpy scalars/arrays → Python, NaN/Inf → None, tuples → lists, keys → str."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def unit_label(unit: str | None) -> str:
    from sparc.core.catalog import unit_label as _ul

    return _ul(unit)


def fmt_temp(v: float, unit: str, decimals: int = 2) -> str:
    return f"{abs(v):.{decimals}f} {unit}".strip()


def likely(estimate, se, unit: str = "°F", *, what: str = "", decimals: int = 2) -> dict | None:
    """A plain-language ΔT estimate (api.md ``Likely``): ``lo``/``hi`` = estimate ± 1.96·se.

    Negative = cooler.  ``confidence`` is ``confident_cools`` when the 95% range excludes 0 on the cool
    side, ``confident_warms`` on the warm side, ``could_be_zero`` otherwise and ``unknown`` without an SE.
    """
    if not finite(estimate):
        return None
    est = float(estimate)
    s = float(se) if finite(se) and float(se) >= 0 else None
    lo = hi = None
    if s is not None:
        lo, hi = est - 1.96 * s, est + 1.96 * s
        conf = "confident_cools" if hi < 0 else "confident_warms" if lo > 0 else "could_be_zero"
    else:
        conf = "unknown"
    direction = "Cools" if est < 0 else "Warms" if est > 0 else "Changes"
    head = f"{direction} {what + ' ' if what else ''}by {fmt_temp(est, unit, decimals)}"
    if lo is not None:
        a, b = sorted((abs(lo), abs(hi))) if lo * hi > 0 else (abs(lo), abs(hi))
        if lo * hi > 0:
            head += f" (likely range {a:.{decimals}f}–{b:.{decimals}f} {unit})"
        else:
            head += f" (likely range {lo:+.{decimals}f} to {hi:+.{decimals}f} {unit})".replace("-", "−")
    tail = {"confident_cools": "Confident it cools.", "confident_warms": "Confident it warms.",
            "could_be_zero": "Could be zero.", "unknown": "No uncertainty estimate."}[conf]
    return {"estimate": est, "se": s, "lo": lo, "hi": hi, "confidence": conf, "phrase": f"{head}. {tail}"}


def jackknife_se(fold_means) -> float | None:
    """``std_k(m_k)·√(K−1)`` over per-fold means (core ``ScenarioResult.summary``); None for K < 2."""
    m = np.asarray(fold_means, float)
    m = m[np.isfinite(m)]
    if m.size < 2:
        return None
    return float(m.std() * np.sqrt(m.size - 1))


class LRU:
    """A thread-safe LRU of values with a byte budget (``size(value)`` → bytes)."""

    def __init__(self, max_bytes: int, size=None):
        self.max_bytes = int(max_bytes)
        self.size = size or (lambda v: int(getattr(v, "nbytes", 0) or 0))
        self._d: "OrderedDict[Any, tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                return None
            self._d.move_to_end(key)
            return hit[0]

    def put(self, key, value) -> None:
        n = int(self.size(value))
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._d[key] = (value, n)
            self._bytes += n
            while self._bytes > self.max_bytes and len(self._d) > 1:
                _, (_, b) = self._d.popitem(last=False)
                self._bytes -= b

    def drop(self, pred) -> None:
        with self._lock:
            for k in [k for k in self._d if pred(k)]:
                _, b = self._d.pop(k)
                self._bytes -= b

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._d)
