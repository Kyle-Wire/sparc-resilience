"""Workspace layout, ids, slugs, timestamps and atomic file helpers (SPEC §4.2, §10.6).

The workspace defaults to ``~/sparc-studio`` and is overridden by
``$SPARC_STUDIO_HOME`` or ``--workspace DIR``::

    studio.sqlite  studio.lock.json  token  cache/  engine/  jobs/<jid>/
    projects/<slug>/  imports/<run_id>/studio/

Every JSON file Studio owns is written atomically (temporary sibling +
``os.replace``) through :mod:`sparc.core.runio`; secrets (token, lock) are
created ``0600``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sparc.core import runio

__all__ = [
    "ENV_HOME", "DEFAULT_HOME", "ID_PREFIXES", "Workspace", "resolve_workspace",
    "new_id", "new_run_id", "slugify", "utc_now", "utc_iso", "parse_utc",
    "write_json_atomic", "read_json", "write_private", "dir_size",
]

ENV_HOME = "SPARC_STUDIO_HOME"
DEFAULT_HOME = "~/sparc-studio"

#: id prefix per object kind (api.md §0.1); run ids follow SPEC §4.3 instead.
ID_PREFIXES = {
    "project": "p", "job": "j", "study": "st", "scenario": "sc", "result": "res", "plan": "pl",
    "sweep": "sw", "comparison": "cmp", "region": "rg", "blob": "bl", "export": "ex", "finding": "fd",
}
_RUN_MODES = ("fast", "full", "custom")


class Workspace:
    """Paths of one Studio workspace.  Creating the object touches nothing; :meth:`ensure` creates the dirs."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    @property
    def db_path(self) -> Path:
        return self.root / "studio.sqlite"

    @property
    def lock_path(self) -> Path:
        return self.root / "studio.lock.json"

    @property
    def token_path(self) -> Path:
        return self.root / "token"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def engine_dir(self) -> Path:
        return self.root / "engine"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    @property
    def imports_dir(self) -> Path:
        return self.root / "imports"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def project_dir(self, slug: str) -> Path:
        return self.projects_dir / slug

    def ensure(self) -> "Workspace":
        for d in (self.root, self.cache_dir, self.engine_dir, self.jobs_dir, self.projects_dir, self.imports_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def resolve_workspace(path: str | os.PathLike | None = None) -> Workspace:
    """``--workspace DIR`` wins, then ``$SPARC_STUDIO_HOME``, then ``~/sparc-studio``."""
    raw = path or os.environ.get(ENV_HOME) or DEFAULT_HOME
    return Workspace(raw)


# ---------------------------------------------------------------------------
# ids and slugs
# ---------------------------------------------------------------------------

def new_id(prefix: str) -> str:
    """``<prefix>_`` plus 8 lowercase base32 characters (40 random bits), e.g. ``j_k3x9q2ab``.

    ``prefix`` is the short prefix (``"j"``) or an object kind from :data:`ID_PREFIXES` (``"job"``).
    """
    prefix = ID_PREFIXES.get(prefix, prefix).rstrip("_")
    return f"{prefix}_{base64.b32encode(secrets.token_bytes(5)).decode('ascii').lower()}"


def new_run_id(mode: str, now: float | None = None) -> str:
    """``YYYYMMDD-HHMMSS-<fast|coarse60|full|custom>-<4 hex>`` (SPEC §4.3), in UTC."""
    mode = str(mode).lower()
    if not (mode in _RUN_MODES or re.fullmatch(r"coarse\d+", mode)):
        raise ValueError(f"run mode must be fast, full, custom or coarse<M>, not {mode!r}")
    stamp = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{mode}-{secrets.token_hex(2)}"


def slugify(name: str, taken: Iterable[str] = (), *, max_len: int = 48) -> str:
    """Filesystem-safe lower-case slug of ``name``; ``-2``, ``-3`` … are appended on collision with ``taken``."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:max_len].strip("-") or "project"
    taken = set(taken)
    if slug not in taken:
        return slug
    i = 2
    while f"{slug}-{i}" in taken:
        i += 1
    return f"{slug}-{i}"


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

def utc_iso(ts: float | None) -> str | None:
    """ISO-8601 UTC string with a ``Z`` suffix (seconds resolution), or None."""
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> str:
    return utc_iso(time.time())


def parse_utc(text: str | None) -> float | None:
    """Unix seconds of an ISO-8601 string (``Z`` or an offset; naive strings are taken as UTC), or None."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def write_json_atomic(path: str | os.PathLike, obj: Any, *, private: bool = False, indent: int | None = 2) -> Path:
    """Atomic JSON write (NaN/Inf → null); ``private=True`` makes the file ``0600``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not private:
        return runio.write_json_atomic(path, obj, indent=indent)
    text = json.dumps(runio._jsonable(obj), indent=indent, allow_nan=False, default=str)
    write_private(path, text + "\n")
    return path


def write_private(path: str | os.PathLike, text: str) -> Path:
    """Atomically write ``text`` to a ``0600`` file (the temporary file is ``0600`` from creation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    return path


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    """Parsed JSON of ``path``, or ``default`` when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dir_size(path: str | os.PathLike, *, exclude: Iterable[str] = ()) -> int:
    """Total bytes of the files under ``path`` (a file's own size), skipping names in ``exclude``; 0 if missing."""
    p = Path(path)
    skip = set(exclude)
    try:
        if p.is_file():
            return p.stat().st_size
    except OSError:
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            if name in skip:
                continue
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total
