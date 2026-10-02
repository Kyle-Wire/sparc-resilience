#!/usr/bin/env python
"""Check that ``docs/studio/api.md`` documents every route of the server, and nothing else.

The server's routes come from the generated OpenAPI document (``python -m sparc.studio
--dump-openapi PATH``, the same document ``GET /openapi.json`` serves and
``tests/studio/openapi.snapshot.json`` freezes). The documented routes are the endpoint
*definitions* in api.md: a ``METHOD /path`` in backticks on

* a heading (``### `GET /api/jobs` ``, also ``· ``-joined pairs on one heading),
* a line that starts with a bold endpoint (``**`POST /api/exports`**``), or
* a table row whose first cell is an endpoint (``| `GET /api/findings` | … |``).

Mentions in running prose, and everything in the changelog sections ("Review changes",
"As-built changes"), are not definitions. Paths compare with their parameter names blanked
(``/api/runs/{rid}`` equals ``/api/runs/{run_id}``), and query strings are ignored. ``HEAD`` and
``OPTIONS`` are not API operations.

The check fails (exit 1) when an OpenAPI operation has no definition ("missing") or a definition
names an operation the server does not have ("extra"). Parameter names that differ between the
two are listed as notes and do not fail the check.

Usage::

    python docs/studio/check_api_doc.py [--openapi PATH] [--api-md PATH] [--json]

Without ``--openapi`` the document is generated with this checkout's ``sparc.studio`` (no server
is started).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_MD = ROOT / "docs" / "studio" / "api.md"
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
ENDPOINT = re.compile(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s]*)`")
CHANGELOG = re.compile(r"^#{1,6}\s+\d+\.\s+(Review changes|As-built changes)\b", re.I)
PARAM = re.compile(r"\{[^}]*\}")


def normalise(path: str) -> str:
    """``/api/runs/{rid}/layers/{key}.bin?x=1`` → ``/api/runs/{}/layers/{}.bin``."""
    return PARAM.sub("{}", path.split("?", 1)[0].rstrip("/") or "/")


def generate_openapi() -> dict:
    """The OpenAPI document of this checkout (``python -m sparc.studio --dump-openapi``)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in [env.get("PYTHONPATH")] if p])
    with tempfile.TemporaryDirectory(prefix="sparc-api-doc-") as tmp:
        out = Path(tmp) / "openapi.json"
        subprocess.run([sys.executable, "-m", "sparc.studio", "--dump-openapi", str(out)], cwd=ROOT, env=env,
                       check=True, stdout=subprocess.DEVNULL)
        return json.loads(out.read_text("utf-8"))


def openapi_routes(doc: dict) -> dict[tuple[str, str], str]:
    """``{(METHOD, normalised path): path as the server spells it}`` for every operation."""
    out = {}
    for path, item in (doc.get("paths") or {}).items():
        for method in item:
            if method.upper() in METHODS:
                out[(method.upper(), normalise(path))] = path
    return out


def documented_routes(text: str) -> dict[tuple[str, str], tuple[str, int]]:
    """``{(METHOD, normalised path): (path as written, line number)}`` for every definition in api.md."""
    out: dict[tuple[str, str], tuple[str, int]] = {}
    in_changelog = False
    in_code = False
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if s.startswith("#"):
            in_changelog = bool(CHANGELOG.match(s)) or (in_changelog and not re.match(r"^##\s", s))
        if in_changelog:
            continue
        if s.startswith("#") or s.startswith("**`"):
            spans = [s]
        elif s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            spans = cells[:1]
        else:
            continue
        for span in spans:
            for m in ENDPOINT.finditer(span):
                method, path = m.group(1), m.group(2)
                if not (path.startswith("/api/") or path == "/auth" or path.startswith("/auth?")):
                    continue
                out.setdefault((method, normalise(path)), (path.split("?", 1)[0], no))
    return out


def param_names(path: str) -> list[str]:
    return PARAM.findall(path)


def compare(doc: dict, text: str) -> dict:
    served = openapi_routes(doc)
    documented = documented_routes(text)
    missing = sorted(set(served) - set(documented), key=lambda k: (k[1], k[0]))
    extra = sorted(set(documented) - set(served), key=lambda k: (k[1], k[0]))
    renamed = []
    for key in sorted(set(served) & set(documented), key=lambda k: (k[1], k[0])):
        spelled, no = documented[key]
        if param_names(spelled) != param_names(served[key]):
            renamed.append({"method": key[0], "openapi": served[key], "api_md": spelled, "line": no})
    return {
        "n_openapi": len(served),
        "n_documented": len(documented),
        "missing": [{"method": m, "path": served[(m, p)]} for m, p in missing],
        "extra": [{"method": m, "path": documented[(m, p)][0], "line": documented[(m, p)][1]} for m, p in extra],
        "param_names_differ": renamed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--openapi", type=Path, help="an OpenAPI JSON file (default: generate one from this checkout)")
    ap.add_argument("--api-md", type=Path, default=API_MD, help=f"the contract to check (default {API_MD})")
    ap.add_argument("--json", action="store_true", help="print the comparison as JSON")
    args = ap.parse_args(argv)

    doc = json.loads(args.openapi.read_text("utf-8")) if args.openapi else generate_openapi()
    report = compare(doc, args.api_md.read_text("utf-8"))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"OpenAPI operations: {report['n_openapi']}; documented in {args.api_md.name}: {report['n_documented']}")
        for row in report["missing"]:
            print(f"missing from api.md: {row['method']} {row['path']}")
        for row in report["extra"]:
            print(f"not served (api.md line {row['line']}): {row['method']} {row['path']}")
        for row in report["param_names_differ"]:
            print(f"note: api.md line {row['line']} writes {row['method']} {row['api_md']}; "
                  f"the server names it {row['openapi']}")
        if not report["missing"] and not report["extra"]:
            print("api.md lists every route: 0 missing, 0 extra")
    return 1 if report["missing"] or report["extra"] else 0


if __name__ == "__main__":
    sys.exit(main())
