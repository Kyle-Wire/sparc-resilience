"""
Global user preferences for SPARC.

Persisted to a JSON file outside any project directory so the same machine-wide
defaults apply across all projects. Per-project overrides (in ``project.yml``)
take precedence over these values.

Location:
    - ``$SPARC_PREFS_DIR/preferences.json`` if the env var is set
    - else ``~/.sparc/preferences.json``

The file is created lazily on first write. Reads return ``{}`` when the file
does not exist or is unreadable — pipeline boot must never fail because of
preferences.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def get_prefs_dir() -> Path:
    """Resolve the directory holding the global preferences file."""
    env = os.environ.get("SPARC_PREFS_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".sparc"


def get_prefs_path() -> Path:
    """Full path to the preferences JSON file."""
    return get_prefs_dir() / "preferences.json"


def load_preferences() -> dict[str, Any]:
    """Load the preferences JSON. Returns ``{}`` when missing or invalid."""
    path = get_prefs_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_preferences(prefs: dict[str, Any]) -> Path:
    """Atomically write *prefs* to the preferences file. Returns the path."""
    if not isinstance(prefs, dict):
        raise TypeError("preferences must be a dict")

    path = get_prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory then atomic-rename so a
    # crashed write never leaves a partially written prefs file.
    fd, tmp = tempfile.mkstemp(prefix=".prefs-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def update_performance_preferences(performance: dict[str, Any]) -> dict[str, Any]:
    """Merge *performance* into the existing prefs under the ``performance`` key.

    Returns the updated full preferences dict. Resets the hardware profile
    cache so the new values take effect on the next :func:`detect_profile`.
    """
    prefs = load_preferences()
    existing = prefs.get("performance") if isinstance(prefs.get("performance"), dict) else {}
    merged = {**existing, **performance}
    # Drop keys explicitly set to None so callers can clear an override.
    merged = {k: v for k, v in merged.items() if v is not None}
    prefs["performance"] = merged
    save_preferences(prefs)

    # Invalidate the cached hardware profile so subsequent reads see the change.
    try:
        from sparc.config.hardware_profile import reset_profile_cache
        reset_profile_cache()
    except Exception:  # noqa: BLE001
        pass
    return prefs
