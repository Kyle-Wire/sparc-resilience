"""The OpenAPI document against its committed snapshot (SPEC §14.4, api.md: "CI snapshots it").

* ``openapi.snapshot.json`` is the document ``sparc studio --dump-openapi`` writes. Any change
  to the wire contract (a path, a method, a schema field) fails here until the snapshot is
  updated on purpose::

      python tests/studio/test_openapi_snapshot.py --update
      npm --prefix studio-web run gen:api          # regenerate src/api/schema.gen.ts
      npm --prefix studio-web run typecheck        # contract.check.ts against the new types

* Every ``/api`` route of the app is in the document (only the SPA fallbacks are hidden).
* ``studio-web/src/api/schema.gen.ts`` was generated from this document: same paths, methods
  and schema names (so a forgotten ``gen:api`` fails here, not only in the typecheck).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = HERE / "openapi.snapshot.json"
SCHEMA_TS = ROOT / "studio-web" / "src" / "api" / "schema.gen.ts"
METHODS = ("get", "put", "post", "delete", "patch", "options", "head", "trace")
SPA_ROUTES = {"/assets/{path:path}", "/{path:path}"}


def dump_openapi(tmp_dir: Path) -> dict:
    """The document exactly as ``sparc studio --dump-openapi`` (and so ``gen:api``) writes it."""
    from sparc.studio.cli import main

    out = Path(tmp_dir) / "openapi.json"
    assert main(["--dump-openapi", str(out)]) == 0
    return json.loads(out.read_text("utf-8"))


def _operations(doc: dict) -> set[str]:
    return {f"{m.upper()} {p}" for p, ops in doc.get("paths", {}).items() for m in ops if m in METHODS}


def _describe_diff(old: dict, new: dict) -> str:
    """A short human summary of what changed between two documents."""
    lines = []
    o_ops, n_ops = _operations(old), _operations(new)
    lines += [f"  + {op}" for op in sorted(n_ops - o_ops)]
    lines += [f"  - {op}" for op in sorted(o_ops - n_ops)]
    for op in sorted(o_ops & n_ops):
        m, p = op.split(" ", 1)
        if old["paths"][p][m.lower()] != new["paths"][p][m.lower()]:
            lines.append(f"  ~ {op}")
    o_s = old.get("components", {}).get("schemas", {})
    n_s = new.get("components", {}).get("schemas", {})
    lines += [f"  + schema {k}" for k in sorted(set(n_s) - set(o_s))]
    lines += [f"  - schema {k}" for k in sorted(set(o_s) - set(n_s))]
    for k in sorted(set(o_s) & set(n_s)):
        if o_s[k] != n_s[k]:
            a, b = o_s[k].get("properties", {}), n_s[k].get("properties", {})
            detail = [f"+{f}" for f in sorted(set(b) - set(a))] + [f"-{f}" for f in sorted(set(a) - set(b))]
            detail += [f"~{f}" for f in sorted(set(a) & set(b)) if a[f] != b[f]]
            if o_s[k].get("required") != n_s[k].get("required"):
                detail.append("required")
            lines.append(f"  ~ schema {k} ({', '.join(detail) or 'other keys'})")
    for key in ("info", "openapi"):
        if old.get(key) != new.get(key):
            lines.append(f"  ~ {key}")
    return "\n".join(lines[:80]) + ("\n  …" if len(lines) > 80 else "")


def _ts_operations(text: str) -> set[str]:
    """``METHOD path`` pairs of the generated ``paths`` interface (methods that are not ``never``)."""
    block = text.split("export interface paths {", 1)[1].split("\nexport ", 1)[0]
    ops = set()
    for m in re.finditer(r'^    "(/[^"]*)": \{\n(.*?)^    \};', block, re.M | re.S):
        for mm in re.finditer(r"^        (\w+)\??: (.+);$", m.group(2), re.M):
            if mm.group(1) in METHODS and mm.group(2).strip() != "never":
                ops.add(f"{mm.group(1).upper()} {m.group(1)}")
    return ops


def _ts_schemas(text: str) -> set[str]:
    block = text.split("export interface components {", 1)[1].split("\n    responses:", 1)[0]
    return {m.group(1) or m.group(2) for m in re.finditer(r'^        (?:"([^"]+)"|(\w+)): ', block, re.M)}


def test_openapi_matches_snapshot(tmp_path):
    doc = dump_openapi(tmp_path)
    assert SNAPSHOT.exists(), f"{SNAPSHOT.name} is missing: run `python {Path(__file__).relative_to(ROOT)} --update`"
    snap = json.loads(SNAPSHOT.read_text("utf-8"))
    if doc != snap:
        raise AssertionError(
            "The OpenAPI document differs from the committed snapshot:\n" + _describe_diff(snap, doc)
            + f"\nIf the change is intended, run `python {Path(__file__).relative_to(ROOT)} --update`,"
            " then `npm --prefix studio-web run gen:api` and `npm --prefix studio-web run typecheck`.")


def test_dump_is_deterministic(tmp_path):
    assert dump_openapi(tmp_path / "a") == dump_openapi(tmp_path / "b")


def test_every_api_route_is_documented(tmp_path):
    from fastapi.routing import APIRoute

    from sparc.studio.app import create_app
    from sparc.studio.settings import StudioSettings

    app = create_app(StudioSettings(workspace=tmp_path / "ws", token="openapi", sampler=False))
    doc = app.openapi()
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    hidden = {r.path for r in routes if not r.include_in_schema}
    assert hidden <= SPA_ROUTES, f"routes left out of the OpenAPI document: {sorted(hidden - SPA_ROUTES)}"
    undocumented = sorted(f"{m} {r.path}" for r in routes if r.include_in_schema for m in r.methods - {"HEAD"}
                          if m.lower() not in doc["paths"].get(r.path, {}))
    assert not undocumented, f"routes missing from the document: {undocumented}"
    assert any(p.startswith("/api/") for p in doc["paths"])
    # FastAPI's interactive docs stay off (api.md §0.1)
    assert doc["info"]["title"] == "SPARC Studio"
    assert not any(p in doc["paths"] for p in ("/docs", "/redoc"))


def test_generated_types_follow_the_snapshot():
    assert SCHEMA_TS.exists(), "studio-web/src/api/schema.gen.ts is missing: run `npm --prefix studio-web run gen:api`"
    text = SCHEMA_TS.read_text("utf-8")
    snap = json.loads(SNAPSHOT.read_text("utf-8"))
    assert text.startswith("// Generated by `npm run gen:api`")
    assert f"// API: {snap['info']['title']} {snap['info']['version']}" in text
    stale = "studio-web/src/api/schema.gen.ts is stale: run `npm --prefix studio-web run gen:api`"
    ts_ops, doc_ops = _ts_operations(text), _operations(snap)
    assert ts_ops == doc_ops, f"{stale} (operations +{sorted(doc_ops - ts_ops)} -{sorted(ts_ops - doc_ops)})"
    ts_schemas, doc_schemas = _ts_schemas(text), set(snap["components"]["schemas"])
    assert ts_schemas == doc_schemas, (f"{stale} (schemas +{sorted(doc_schemas - ts_schemas)} "
                                       f"-{sorted(ts_schemas - doc_schemas)})")


def _update() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sparc-openapi-") as tmp:
        doc = dump_openapi(Path(tmp))
    SNAPSHOT.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT.relative_to(ROOT)} ({len(_operations(doc))} operations, "
          f"{len(doc['components']['schemas'])} schemas)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--update"]:
        sys.path.insert(0, str(ROOT))
        sys.exit(_update())
    print(f"usage: python {Path(__file__).relative_to(ROOT)} --update", file=sys.stderr)
    sys.exit(2)
