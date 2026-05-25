"""EPSG code validation for CRS-related endpoints.

EPSG codes are integers in the range 1–999999 (per the EPSG registry).
This module provides a single public function that normalises and validates
a user-supplied EPSG string before it is passed to pyproj, preventing
injection attacks and unexpected crashes.
"""
from __future__ import annotations

import re

_EPSG_RE = re.compile(r"^epsg:(\d+)$", re.IGNORECASE)
_EPSG_MIN = 1
_EPSG_MAX = 999_999


def validate_epsg(raw: str) -> int:
    """Validate and normalise a user-supplied EPSG code string.

    Accepts:
      - Plain integer strings: "4326"
      - "EPSG:" or "epsg:" prefixed strings: "EPSG:4326"

    Returns the integer EPSG code.

    Raises:
      ValueError: if the string is not a valid numeric EPSG code in range
                  1–999999.
    """
    stripped = raw.strip()

    # Accept "EPSG:NNNN" or "epsg:NNNN"
    m = _EPSG_RE.match(stripped)
    if m:
        stripped = m.group(1)

    try:
        code = int(stripped)
    except ValueError:
        raise ValueError(
            f"EPSG code must be a numeric string, got: {raw!r}"
        )

    if not (_EPSG_MIN <= code <= _EPSG_MAX):
        raise ValueError(
            f"EPSG code {code} is out of range ({_EPSG_MIN}–{_EPSG_MAX})"
        )

    return code
