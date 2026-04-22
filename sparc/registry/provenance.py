"""
Provenance & reproducibility helpers (Phase 17)
================================================

Stamps every artifact with:
    - config hash      (sha256 of canonical JSON of project.yml)
    - code commit      (git rev-parse HEAD if available)
    - data hash        (sha256 of input CSV/Parquet bytes, streamed)
    - env lock excerpt (versions of pinned scientific packages)
    - timestamp (UTC)

Sidecar `.provenance.json` files live next to each tracked artifact.
A single `run_provenance.json` is also written to the run's output_dir
summarising the freeze for the whole run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Packages whose versions we record (best-effort; missing ones are skipped)
LOCKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "lightgbm",
    "xgboost",
    "econml",
    "dowhy",
    "geopandas",
    "shapely",
    "pyproj",
    "fastapi",
    "uvicorn",
    "sparc",
)


# ------------------------------------------------------------------
# Hashing
# ------------------------------------------------------------------

def _canonical_json(obj: Any) -> bytes:
    """Stable JSON encoding for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def compute_config_hash(config: dict) -> str:
    """Hash a config dict deterministically."""
    return hash_obj(config)


def compute_file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    """Stream-hash a file. Returns None if missing."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def compute_data_hash(config: dict) -> dict[str, str | None]:
    """Hash the input data file referenced by a config."""
    data_cfg = config.get("data", {}) or {}
    candidates = [
        data_cfg.get("path"),
        data_cfg.get("file"),
        data_cfg.get("csv"),
        config.get("paths", {}).get("data_path"),
    ]
    info: dict[str, Any] = {"path": None, "sha256": None, "size_bytes": None}
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            info["path"] = str(p.resolve())
            info["sha256"] = compute_file_hash(p)
            info["size_bytes"] = p.stat().st_size
            break
    return info


# ------------------------------------------------------------------
# Environment / git
# ------------------------------------------------------------------

def git_commit(repo_root: str | Path | None = None) -> str | None:
    """Return current HEAD commit (None if git unavailable / not a repo)."""
    try:
        cwd = str(repo_root) if repo_root else None
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def git_dirty(repo_root: str | Path | None = None) -> bool | None:
    try:
        cwd = str(repo_root) if repo_root else None
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def env_lock_excerpt(packages: Iterable[str] = LOCKED_PACKAGES) -> dict[str, str]:
    """Return {package: version} for installed packages of interest."""
    lock: dict[str, str] = {}
    for pkg in packages:
        try:
            lock[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return lock


# ------------------------------------------------------------------
# Provenance objects
# ------------------------------------------------------------------

def make_provenance(
    config: dict,
    data_info: dict | None = None,
    repo_root: str | Path | None = None,
    extras: dict | None = None,
) -> dict:
    """Build a provenance dict for an artifact or whole run."""
    if data_info is None:
        data_info = compute_data_hash(config)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": compute_config_hash(config),
        "data": data_info,
        "git": {
            "commit": git_commit(repo_root),
            "dirty": git_dirty(repo_root),
        },
        "env": env_lock_excerpt(),
        "extras": extras or {},
    }


def write_sidecar(
    artifact_path: str | Path,
    config: dict,
    *,
    data_info: dict | None = None,
    extras: dict | None = None,
) -> Path:
    """Write `<artifact>.provenance.json` next to *artifact_path*.

    The sidecar also records the artifact's own sha256 so downstream
    verification can detect tampering.
    """
    artifact_path = Path(artifact_path)
    prov = make_provenance(config, data_info=data_info, extras=extras)
    prov["artifact"] = {
        "path": str(artifact_path.resolve()),
        "sha256": compute_file_hash(artifact_path),
        "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else None,
    }
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return sidecar


def freeze_run(
    output_dir: str | Path,
    config: dict,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Write a single ``run_provenance.json`` snapshot for a whole run.

    Also persists a frozen copy of the config as ``frozen_config.json``
    so a reproduce request can re-load the exact same settings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prov = make_provenance(config, repo_root=repo_root)
    prov_path = output_dir / "run_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    frozen = output_dir / "frozen_config.json"
    frozen.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    return prov_path


def load_provenance(run_dir: str | Path) -> dict | None:
    """Load ``run_provenance.json`` from a run directory if present."""
    p = Path(run_dir) / "run_provenance.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_frozen_config(run_dir: str | Path) -> dict | None:
    p = Path(run_dir) / "frozen_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------

def verify_sidecars(run_dir: str | Path) -> dict:
    """Re-hash every artifact that has a `.provenance.json` sidecar.

    Returns
    -------
    {
        "checked": [{path, expected, actual, match}],
        "missing": [paths whose sidecar refers to a nonexistent file],
        "n_match": int,
        "n_mismatch": int,
    }
    """
    run_dir = Path(run_dir)
    checked: list[dict] = []
    missing: list[str] = []
    for sidecar in run_dir.rglob("*.provenance.json"):
        if sidecar.name == "run_provenance.json":
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        art = meta.get("artifact") or {}
        art_path = Path(art.get("path") or "")
        expected = art.get("sha256")
        if not art_path.exists():
            missing.append(str(art_path))
            continue
        actual = compute_file_hash(art_path)
        checked.append({
            "path": str(art_path),
            "expected": expected,
            "actual": actual,
            "match": expected is not None and expected == actual,
        })
    return {
        "checked": checked,
        "missing": missing,
        "n_match": sum(1 for c in checked if c["match"]),
        "n_mismatch": sum(1 for c in checked if not c["match"]),
    }


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict:
    """Compare provenance of two runs (config/data/env/git).

    Useful after `POST /reproduce` to confirm the re-run produced the
    same inputs.
    """
    a = load_provenance(run_a) or {}
    b = load_provenance(run_b) or {}

    def _eq(field: str) -> bool:
        return a.get(field) == b.get(field)

    env_a = a.get("env", {}) or {}
    env_b = b.get("env", {}) or {}
    env_diffs = {
        k: {"a": env_a.get(k), "b": env_b.get(k)}
        for k in set(env_a) | set(env_b)
        if env_a.get(k) != env_b.get(k)
    }
    return {
        "config_hash_match": a.get("config_hash") == b.get("config_hash"),
        "data_hash_match": (a.get("data") or {}).get("sha256") == (b.get("data") or {}).get("sha256"),
        "git_commit_match": (a.get("git") or {}).get("commit") == (b.get("git") or {}).get("commit"),
        "env_diffs": env_diffs,
        "a": a,
        "b": b,
    }


__all__ = [
    "LOCKED_PACKAGES",
    "compute_config_hash",
    "compute_data_hash",
    "compute_file_hash",
    "git_commit",
    "git_dirty",
    "env_lock_excerpt",
    "make_provenance",
    "write_sidecar",
    "freeze_run",
    "load_provenance",
    "load_frozen_config",
    "verify_sidecars",
    "compare_runs",
]
