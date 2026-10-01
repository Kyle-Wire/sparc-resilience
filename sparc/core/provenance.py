"""Provenance of a run: what data, configuration, code and environment made it.

Recorded in ``manifest.json`` (``provenance``) and, for the environment, as
``environment.txt`` (a ``pip freeze``-style lock) in the run directory, so a
reader can tell whether two runs are comparable and ``sparc core reproduce``
can check that it is re-running the same thing.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd

CORE_DIR = Path(__file__).resolve().parent


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def config_hash(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()


def code_fingerprint() -> str:
    """SHA-256 over the core package sources (what a resume checkpoint is keyed on)."""
    h = hashlib.sha256()
    for src in sorted(CORE_DIR.glob("*.py")):
        h.update(src.name.encode())
        h.update(src.read_bytes())
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=CORE_DIR, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--", str(CORE_DIR))
    return {"commit": commit, "core_dirty": bool(status) if status is not None else None}


def environment_lock() -> str:
    """Installed distributions as ``name==version`` lines (sorted)."""
    from importlib import metadata

    seen = {}
    for d in metadata.distributions():
        name = (d.metadata.get("Name") or "").strip()
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={d.version}"
    return "\n".join(seen[k] for k in sorted(seen)) + "\n"


def provenance(cfg, input_sha256: str | None, input_kind: str) -> dict:
    return {
        "input_sha256": input_sha256, "input_kind": input_kind,
        "data_path": str(cfg.data.get("path")) if cfg.data.get("path") else None,
        "config_dir": str(cfg.base_dir), "config_sha256": config_hash(cfg.raw),
        "code_sha256": code_fingerprint(), "git": git_state(),
        "platform": {"python": platform.python_version(), "system": platform.system(),
                     "machine": platform.machine()},
        "lock_file": "environment.txt",
    }
