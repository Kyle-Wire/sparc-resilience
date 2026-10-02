#!/usr/bin/env python
"""Check that the committed SPA build matches its sources (SPEC §13.2).

``sparc/studio/static/`` is committed so pip users need no Node. This script fails when that
build is out of date or incomplete:

* ``BUILD_INFO.json`` exists, has ``src_sha256``, ``vite`` and ``react`` and no timestamp;
* ``src_sha256`` equals the hash of the current sources, computed exactly as ``sourceHash()`` in
  ``studio-web/vite.config.ts``: every file under ``studio-web/src`` (skipping any file or folder
  whose name starts with ``.``), plus ``package-lock.json``, ``vite.config.ts`` and
  ``index.html``; POSIX paths relative to ``studio-web/``, sorted by code point; SHA-256 over
  ``relpath + "\\0" + bytes + "\\0"`` per file;
* ``vite`` and ``react`` are the versions ``package-lock.json`` pins;
* ``index.html`` references hashed assets that all exist under ``static/assets/``.

Usage: ``python scripts/check_studio_assets.py [--web studio-web] [--static sparc/studio/static]``.
Exit status 0 when the build is current, 1 otherwise (each problem is printed). Rebuild with
``npm --prefix studio-web ci && npm --prefix studio-web run build``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA_FILES = ("package-lock.json", "vite.config.ts", "index.html")
HASHED = re.compile(r"^assets/[^/]+-[A-Za-z0-9_-]{6,}\.(?:js|css|woff2?|ttf|svg|png)$")
TIME_KEYS = ("built_utc", "built_at", "timestamp", "time", "date")


def _walk(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            yield from _walk(p)
        else:
            yield p


def source_hash(web: Path) -> str:
    """``sourceHash()`` of ``studio-web/vite.config.ts``, reimplemented."""
    files = list(_walk(web / "src")) + [web / name for name in EXTRA_FILES]
    rel = sorted((f.relative_to(web).as_posix() for f in files), key=lambda s: [ord(c) for c in s])
    h = hashlib.sha256()
    for r in rel:
        h.update(r.encode("utf-8"))
        h.update(b"\0")
        h.update((web / r).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def locked_version(web: Path, package: str) -> str | None:
    try:
        lock = json.loads((web / "package-lock.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return (lock.get("packages", {}).get(f"node_modules/{package}") or {}).get("version")


def asset_refs(html: str) -> list[str]:
    """Same-origin asset paths that index.html loads (scripts, preloads, stylesheets, icons)."""
    refs = re.findall(r'(?:src|href)="/([^"#?]+)"', html)
    return [r for r in refs if not r.startswith(("api/", "auth"))]


def check(web: Path, static: Path) -> list[str]:
    problems: list[str] = []
    info_path = static / "BUILD_INFO.json"
    if not info_path.is_file():
        return [f"{info_path} is missing: build the SPA (npm --prefix studio-web run build)"]
    try:
        info = json.loads(info_path.read_text("utf-8"))
    except ValueError as exc:
        return [f"{info_path} is not JSON: {exc}"]
    for key in ("src_sha256", "vite", "react"):
        if not info.get(key):
            problems.append(f"BUILD_INFO.json has no {key}")
    stamped = [k for k in info if k in TIME_KEYS]
    if stamped:
        problems.append(f"BUILD_INFO.json carries a timestamp ({', '.join(stamped)}): rebuilds would never match")
    current = source_hash(web)
    if info.get("src_sha256") and info["src_sha256"] != current:
        problems.append(f"the build is stale: BUILD_INFO.src_sha256 {info['src_sha256'][:12]}… but the sources hash to "
                        f"{current[:12]}…; rebuild with npm --prefix studio-web run build and commit sparc/studio/static")
    for package in ("vite", "react"):
        want = locked_version(web, package)
        if want and info.get(package) and info[package] != want:
            problems.append(f"BUILD_INFO.{package} is {info[package]} but package-lock.json pins {want}")
    index = static / "index.html"
    if not index.is_file():
        problems.append(f"{index} is missing")
        return problems
    refs = asset_refs(index.read_text("utf-8"))
    scripts = [r for r in refs if r.endswith(".js")]
    if not scripts:
        problems.append("index.html loads no script")
    for r in refs:
        if r.startswith("assets/") and not HASHED.match(r):
            problems.append(f"index.html references an unhashed asset: /{r}")
        if not (static / r).is_file():
            problems.append(f"index.html references /{r}, which is not in {static}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--web", type=Path, default=ROOT / "studio-web", help="the SPA sources (default studio-web)")
    ap.add_argument("--static", type=Path, default=ROOT / "sparc" / "studio" / "static",
                    help="the committed build (default sparc/studio/static)")
    args = ap.parse_args(argv)
    problems = check(args.web.resolve(), args.static.resolve())
    if problems:
        for p in problems:
            print(f"check_studio_assets: {p}", file=sys.stderr)
        return 1
    info = json.loads((args.static / "BUILD_INFO.json").read_text("utf-8"))
    print(f"check_studio_assets: build current (src_sha256 {info['src_sha256'][:12]}…, vite {info['vite']}, "
          f"react {info['react']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
