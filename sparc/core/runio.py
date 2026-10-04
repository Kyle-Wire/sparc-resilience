"""Atomic run-directory writes and the per-run manifest lock.

Every file a run directory gains is written to a hidden temporary sibling
(``.<name>.<pid>.<thread>.tmp``), flushed to disk and moved into place with
``os.replace``.  A reader, or a process killed mid-write, therefore sees
either the old file or the new one, never half of one.  A writer killed
outright (SIGKILL, the OOM killer) leaves its temporary behind; the next
atomic write of the same file removes the temporaries of writers that no
longer exist, and :func:`remove_stale_tmp` sweeps a whole folder.

``update_manifest`` is the one way post-run actions (baselines, planner,
uncertainty, emulator, studies) edit ``manifest.json``: under
:func:`run_lock` it re-reads the manifest, replaces the given top-level
sections, records them in ``post_run[]`` and writes it back atomically, so
concurrent actions on one run never lose each other's sections.

JSON is written with ``allow_nan=False`` after mapping NaN/Inf to ``null``.
Like ``progress.py`` this module is excluded from the resume fingerprint.
"""

from __future__ import annotations

import datetime as _dt
import glob as _glob
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt
except ImportError:
    msvcrt = None

__all__ = ["LOCK_NAME", "MANIFEST_NAME", "atomic_open", "write_json_atomic", "write_text_atomic",
           "write_bytes_atomic", "write_parquet_atomic", "write_npz_atomic", "remove_stale_tmp", "run_lock",
           "update_manifest"]

LOCK_NAME = ".sparc.lock"
MANIFEST_NAME = "manifest.json"


def _jsonable(obj):
    """JSON-safe copy: NaN/Inf → None, numpy/pandas values → Python, paths → str, sets → sorted lists."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_jsonable(v) for v in obj), key=str)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if hasattr(obj, "tolist") and type(obj).__module__.split(".")[0] in ("numpy", "pandas"):
        return _jsonable(obj.tolist())             # arrays, scalars, Series: their NaN become null too
    return obj                                     # json's default=str handles the rest


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


_TMP_RE = re.compile(r"^\.(?P<name>.+)\.(?P<pid>\d+)\.(?P<tid>\d+)\.tmp$")


def _pid_gone(pid: int) -> bool:
    """Whether no process ``pid`` runs on this host (a zombie counts as gone).  Unknown → ``False``."""
    if os.name != "posix" or pid <= 0:
        return False                               # os.kill(pid, 0) is not a probe on Windows
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:                                # EPERM: alive, another user's
        return False
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()[0] == b"Z"
    except (OSError, IndexError):
        return False


def _stale_tmps(directory: Path, name: str | None = None):
    """``(path, bytes)`` of the atomic-write temporaries in ``directory`` (of file ``name`` only, when given)
    whose writer process no longer exists."""
    pattern = f".{_glob.escape(name)}.*.tmp" if name is not None else ".*.tmp"
    try:
        entries = list(Path(directory).glob(pattern))
    except OSError:
        return
    for p in entries:
        m = _TMP_RE.match(p.name)
        if m is None or (name is not None and m.group("name") != name) or not _pid_gone(int(m.group("pid"))):
            continue
        try:
            yield p, p.stat().st_size
        except OSError:
            continue


def _unlink_all(found) -> int:
    freed = 0
    for p, size in found:
        try:
            p.unlink()
            freed += size
        except OSError:
            pass
    return freed


def remove_stale_tmp(directory) -> int:
    """Remove the temporaries that writers killed mid-write left in ``directory`` (not its sub-folders);
    returns the bytes freed.  Temporaries of live processes are kept."""
    return _unlink_all(_stale_tmps(Path(directory)))


@contextmanager
def atomic_open(path, mode: str = "wb", encoding: str | None = None) -> Iterator[IO]:
    """Open a temporary sibling of ``path`` for writing; on success it is fsynced and moved onto ``path``.

    On any exception (including ``Cancelled``) the temporary file is removed
    and ``path`` is left untouched.
    """
    if not mode.startswith("w"):
        raise ValueError("atomic_open writes whole files: use a 'w' mode")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _unlink_all(_stale_tmps(path.parent, path.name))   # left by a writer of this file killed mid-write
    tmp = _tmp_path(path)
    try:
        with open(tmp, mode, encoding=encoding) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(path, obj, *, indent: int | None = 2, sort_keys: bool = False) -> Path:
    """Write ``obj`` as JSON (NaN/Inf → null, ``allow_nan=False``) via tmp + ``os.replace``."""
    text = json.dumps(_jsonable(obj), indent=indent, sort_keys=sort_keys, allow_nan=False, default=str)
    return write_text_atomic(path, text)


def write_text_atomic(path, text: str, *, encoding: str = "utf-8") -> Path:
    with atomic_open(path, "w", encoding=encoding) as f:
        f.write(text)
    return Path(path)


def write_bytes_atomic(path, data: bytes) -> Path:
    with atomic_open(path, "wb") as f:
        f.write(data)
    return Path(path)


def write_parquet_atomic(df, path, **kwargs) -> Path:
    """``df.to_parquet`` via tmp + ``os.replace`` (``index=False`` unless given)."""
    kwargs.setdefault("index", False)
    with atomic_open(path, "wb") as f:
        df.to_parquet(f, **kwargs)
    return Path(path)


def write_npz_atomic(path, *, compressed: bool = False, **arrays) -> Path:
    """``np.savez`` (or ``savez_compressed``) via tmp + ``os.replace``; ``path`` is used as given (no suffix added)."""
    import numpy as np

    with atomic_open(path, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
    return Path(path)


# ---------------------------------------------------------------------------
# per-run lock
# ---------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_HELD = threading.local()


def _after_fork() -> None:
    global _LOCKS_GUARD, _HELD
    _LOCKS.clear()
    _LOCKS_GUARD = threading.Lock()
    _HELD = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def run_lock(run_dir, timeout: float = 600) -> Iterator[None]:
    """Exclusive lock on a run directory, across threads and processes.

    Holds a process-local lock for the directory plus ``fcntl.flock`` on a
    fresh descriptor of ``<run_dir>/.sparc.lock`` (``msvcrt.locking``,
    best-effort, on Windows).  Re-entrant within one thread.  Raises
    ``TimeoutError`` after ``timeout`` seconds.
    """
    key = os.path.realpath(os.fspath(run_dir))
    depth = _HELD.__dict__.setdefault("depth", {})
    if depth.get(key):
        depth[key] += 1
        try:
            yield
        finally:
            depth[key] -= 1
        return
    with _LOCKS_GUARD:
        tlock = _LOCKS.setdefault(key, threading.Lock())
    deadline = time.monotonic() + timeout
    if not tlock.acquire(timeout=max(0.0, timeout)):
        raise TimeoutError(f"run lock on {key} not acquired within {timeout:g} s")
    try:
        os.makedirs(key, exist_ok=True)
        fd = os.open(os.path.join(key, LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock_fd(fd, deadline, key, timeout)
            depth[key] = 1
            try:
                yield
            finally:
                depth.pop(key, None)
                _unlock_fd(fd)
        finally:
            os.close(fd)
    finally:
        tlock.release()


def _lock_fd(fd: int, deadline: float, key: str, timeout: float) -> None:
    wait = 0.005
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run lock on {key} held by another process for more than {timeout:g} s") from None
            time.sleep(wait)
            wait = min(wait * 2, 0.1)


def _unlock_fd(fd: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def update_manifest(run_dir, sections: dict, *, source: str) -> dict:
    """Replace top-level ``sections`` of ``<run_dir>/manifest.json`` under :func:`run_lock`.

    Each section is recorded in ``post_run[]`` as ``{section, at_utc, source}``.
    A missing manifest starts empty.  Returns the manifest as written.
    """
    if "post_run" in sections:
        raise ValueError("post_run is maintained by update_manifest itself")
    run_dir = Path(run_dir)
    path = run_dir / MANIFEST_NAME
    with run_lock(run_dir):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            manifest = {}
        at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        history = manifest.get("post_run")
        history = list(history) if isinstance(history, list) else []
        for name, value in sections.items():
            manifest[str(name)] = value
            history.append({"section": str(name), "at_utc": at, "source": source})
        manifest["post_run"] = history
        manifest = _jsonable(manifest)
        write_json_atomic(path, manifest)
    return manifest
