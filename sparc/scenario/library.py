"""
Append-only scenario library (Phase 17)
========================================

Git-like, append-only log of scenario configurations stored at
``<output_dir>/scenarios/library.jsonl``.

Each line is one entry:
    {
        "id": "<sha256[:16]>",
        "parent_id": "<sha256[:16] | null>",
        "author": "kw",
        "comment": "Tighten canopy band to 0.4-0.6",
        "created_utc": "2026-04-22T16:30:00+00:00",
        "scenario": { ... },     # full scenario dict (predictors, ranges, etc.)
        "config_hash": "abc..."  # so the entry can be tied to a project config
    }

Append-only: we never mutate prior lines. Branches are encoded by the
``parent_id`` field (multiple children may point at the same parent).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIBRARY_NAME = "library.jsonl"


def _library_path(output_dir: str | Path) -> Path:
    p = Path(output_dir) / "scenarios" / LIBRARY_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _entry_id(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def append_entry(
    output_dir: str | Path,
    scenario: dict,
    *,
    author: str = "anonymous",
    comment: str = "",
    parent_id: str | None = None,
    config_hash: str | None = None,
) -> dict:
    """Append a new scenario entry. Returns the persisted entry."""
    payload = {
        "scenario": scenario,
        "parent_id": parent_id,
        "author": author,
        "config_hash": config_hash,
    }
    entry = {
        "id": _entry_id(payload),
        "parent_id": parent_id,
        "author": author,
        "comment": comment,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "scenario": scenario,
    }
    path = _library_path(output_dir)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry


def list_entries(output_dir: str | Path) -> list[dict]:
    """Read all entries (oldest first)."""
    path = _library_path(output_dir)
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def get_entry(output_dir: str | Path, entry_id: str) -> dict | None:
    for e in list_entries(output_dir):
        if e.get("id") == entry_id:
            return e
    return None


def build_timeline(output_dir: str | Path) -> dict:
    """Return entries grouped into a parent→children adjacency map.

    Useful for the version timeline UI.
    """
    entries = list_entries(output_dir)
    children: dict[str | None, list[str]] = {}
    for e in entries:
        children.setdefault(e.get("parent_id"), []).append(e["id"])
    return {
        "entries": entries,
        "children": children,
        "roots": children.get(None, []),
        "count": len(entries),
    }


__all__ = [
    "append_entry",
    "list_entries",
    "get_entry",
    "build_timeline",
]
