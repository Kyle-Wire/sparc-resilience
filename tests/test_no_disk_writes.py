"""Phase E — CI guardrails for the artifacts.db single-source policy.

Static checks that prevent regression of two design rules:

1. The pipeline never writes PNG files. Visualisations are rendered
   live in the desktop app from artifacts.db. Stage code therefore
   must not call ``plt.savefig`` / ``Figure.savefig`` / ``fig.savefig``
   nor open files with a ``.png`` suffix in write mode.

2. The server never renders or returns PNG bytes. It only serves
   structured data (CSV / JSON / GeoJSON / native blobs) sourced
   from artifacts.db.

These tests scan source files (not byte-compiled modules) and use a
small allowlist for legacy code paths that have been intentionally
preserved with a clear opt-out marker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPARC = REPO / "sparc"

# Files / globs that the policy intentionally ignores.
# - tests themselves (they may legitimately reference .png to assert
#   that we no longer write one),
# - the storage layer's _write/_read which handles legacy .png paths
#   by raising a TypeError and never actually writing,
# - the `report` package which still produces user-export bundles
#   when explicitly invoked by the desktop app.
PNG_ALLOWLIST = {
    "sparc/run/result_store.py",       # raises TypeError on png
    "sparc/registry/store.py",          # internal only
    "sparc/report",                     # explicit export bundles
}

SAVEFIG_PATTERN = re.compile(r"\.savefig\s*\(")
PNG_OPEN_PATTERN = re.compile(r"""['"][^'"]*\.png['"]""")


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        if any(rel == a or rel.startswith(a + "/") for a in PNG_ALLOWLIST):
            continue
        yield p, rel


def test_no_savefig_in_pipeline():
    """Pipeline modules must not call ``.savefig(...)`` anywhere."""
    offenders: list[str] = []
    for path, rel in _iter_py_files(SPARC):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if SAVEFIG_PATTERN.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "PNG savefig calls detected in pipeline (visualisations must be "
        "rendered live in the desktop app from artifacts.db):\n"
        + "\n".join(offenders)
    )


def test_no_png_string_writes_in_pipeline():
    """No source file may contain a ``.png`` filename literal in write contexts."""
    offenders: list[str] = []
    for path, rel in _iter_py_files(SPARC):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not PNG_OPEN_PATTERN.search(line):
                continue
            # Allow .png literals only in clearly read-only / docstring / log
            # contexts. We treat occurrences inside open(..., "w"), savefig,
            # save_figure, or to_png as offenders.
            lowered = line.lower()
            if any(tok in lowered for tok in (
                "savefig", "save_figure", "to_png", '"w"', "'w'", '"wb"', "'wb'",
            )):
                offenders.append(f"{rel}:{i}: {stripped}")
    assert not offenders, (
        "PNG write call sites detected in pipeline:\n" + "\n".join(offenders)
    )


def test_server_has_no_png_endpoints():
    """The server must not actively render or return PNG bytes.

    The ``/results/export`` endpoint is allowed: it receives a
    client-generated PNG (a download of a live visualisation) and
    stores it under ``output/exports/`` for the user. We therefore
    only forbid endpoints that *return* image/png as a media_type.
    """
    server_app = SPARC / "server" / "app.py"
    if not server_app.exists():
        pytest.skip("server/app.py not present")
    text = server_app.read_text(encoding="utf-8")
    forbidden = (
        'media_type="image/png"',
        "media_type='image/png'",
        'media_type="image/svg+xml"',
        "media_type='image/svg+xml'",
    )
    hits = [tok for tok in forbidden if tok in text]
    assert not hits, (
        "Server still returns image media types: " + ", ".join(hits)
    )


def test_result_store_save_figure_removed():
    """ResultStore.save_figure must no longer exist."""
    from sparc.run.result_store import ResultStore
    assert not hasattr(ResultStore, "save_figure"), (
        "ResultStore.save_figure was removed in Phase C7 — pipeline must not "
        "write PNGs. Re-export through the live desktop renderer instead."
    )
