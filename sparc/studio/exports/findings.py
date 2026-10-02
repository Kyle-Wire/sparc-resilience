"""The findings notebook (SPEC §6.10, api.md §11): rows, their mirror files, images and the export.

A finding pins what was on screen: ``view`` (route id) and ``url_state`` (the query to reopen it), a
``title``, a Markdown ``note_md``, the ``snapshot`` of the numbers shown, and an optional image (chart SVG or
map PNG, uploaded raw, ≤ 10 MB).  Every change rewrites the mirror ``projects/<slug>/findings/<fid>.json``
(the Finding minus ``image_url``, plus ``image``: the file name or null) next to the image
``<fid>.png|svg``, so ``sparc studio --reindex`` rebuilds the table (:func:`_reindex`).

The export (``export.findings``) writes the findings as Markdown (a ZIP with ``findings.md`` and the images)
or one self-contained HTML file (images as data URIs in ``<img>``, never inlined SVG markup).  Both carry a
provenance block per run (run, commit, hashes) and the run's generated caveats; the numbers are the
snapshotted ones, so the export reproduces what was on screen.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from pathlib import Path

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError, validation_error
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.exports")

__all__ = ["IMAGE_TYPES", "IMAGE_MAX_BYTES", "PNG_SIGNATURE", "image_ok", "finding_out", "get_row", "get_row_or_none",
           "list_findings", "create_finding", "patch_finding", "delete_finding", "image_path", "set_image",
           "finding_blocks", "export_findings", "findings_dir"]

IMAGE_TYPES = {"image/png": ".png", "image/svg+xml": ".svg"}
IMAGE_MAX_BYTES = 10 * 2 ** 20
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: what may precede an SVG's root element: an XML declaration, comments, a doctype, whitespace
_SVG_PROLOG = re.compile(r"^(?:<\?xml[^>]*\?>|<!--.*?-->|<!DOCTYPE[^>]*>|\s+)*", re.S | re.I)


def image_ok(path: Path, suffix: str) -> bool:
    """Whether the uploaded file holds what its type says: the PNG signature, or an SVG document (an ``<svg``
    root element after the prolog)."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if suffix == ".png":
        return head.startswith(PNG_SIGNATURE)
    text = head.decode("utf-8", errors="replace").lstrip("\ufeff")
    return text[_SVG_PROLOG.match(text).end():].startswith("<svg")


def findings_dir(db, project_id: str) -> Path:
    p = db.fetchone("SELECT dir FROM projects WHERE id = ?", (project_id,))
    if p is None:
        raise ApiError("not_found", f"no project {project_id!r}")
    return Path(p["dir"]) / "findings"


def finding_out(row: dict) -> dict:
    """A ``findings`` row as the api.md ``Finding``."""
    return {"id": row["id"], "project_id": row.get("project_id") or "", "run_id": row.get("run_id"),
            "view": row.get("view") or "", "url_state": row.get("url_state") or "", "title": row.get("title") or "",
            "note_md": row.get("note_md") or "", "snapshot": dbmod.loads(row.get("snapshot_json"), {}) or {},
            "image_url": f"/api/findings/{row['id']}/image" if row.get("image_path") else None,
            "position": row.get("position") if row.get("position") is not None else 0,
            "created_utc": row.get("created_utc") or "", "updated_utc": row.get("updated_utc") or ""}


def get_row_or_none(db, fid: str) -> dict | None:
    return db.fetchone("SELECT * FROM findings WHERE id = ?", (fid,))


def get_row(db, fid: str) -> dict:
    row = get_row_or_none(db, fid)
    if row is None:
        raise ApiError("not_found", f"no finding {fid!r}")
    return row


def list_findings(db, project_id: str | None = None, run_id: str | None = None) -> list[dict]:
    where, args = [], []
    if project_id:
        where.append("project_id = ?")
        args.append(project_id)
    if run_id:
        where.append("run_id = ?")
        args.append(run_id)
    sql = "SELECT * FROM findings" + (" WHERE " + " AND ".join(where) if where else "") + \
        " ORDER BY position, created_utc, id"
    return [finding_out(r) for r in db.fetchall(sql, args)]


def _mirror(db, fid: str) -> None:
    row = get_row_or_none(db, fid)
    if row is None:
        return
    out = finding_out(row)
    out.pop("image_url")
    out["image"] = Path(row["image_path"]).name if row.get("image_path") else None
    out["schema"] = 1
    try:
        write_json_atomic(findings_dir(db, row["project_id"]) / f"{fid}.json", out)
    except (OSError, ApiError):
        log.exception("could not write the mirror of finding %s", fid)


def create_finding(db, body: dict) -> dict:
    pid = body["project_id"]
    findings_dir(db, pid)                                  # 404 for an unknown project
    if body.get("run_id"):
        r = db.fetchone("SELECT project_id FROM runs WHERE id = ?", (body["run_id"],))
        if r is None:
            raise ApiError("not_found", f"no run {body['run_id']!r}")
        if r.get("project_id") and r["project_id"] != pid:
            raise validation_error([{"path": "run_id", "code": "mismatch",
                                     "message": f"run {body['run_id']} belongs to another project"}],
                                   "the run belongs to another project")
    pos = db.fetchval("SELECT MAX(position) FROM findings WHERE project_id = ?", (pid,))
    now = utc_now()
    row = {"id": new_id("fd"), "project_id": pid, "run_id": body.get("run_id"), "view": body.get("view") or "",
           "url_state": body.get("url_state") or "", "title": body.get("title") or "", "note_md": body.get("note_md")
           or "", "snapshot_json": dbmod.dumps(body.get("snapshot") or {}), "image_path": None,
           "position": float(pos) + 1.0 if pos is not None else 0.0, "created_utc": now, "updated_utc": now}
    db.insert("findings", row)
    _mirror(db, row["id"])
    return finding_out(get_row(db, row["id"]))


def patch_finding(db, fid: str, patch: dict) -> dict:
    get_row(db, fid)
    vals = {k: patch[k] for k in ("title", "note_md", "position") if patch.get(k) is not None}
    if not vals:
        return finding_out(get_row(db, fid))
    vals["updated_utc"] = utc_now()
    db.update("findings", {"id": fid}, vals)
    _mirror(db, fid)
    return finding_out(get_row(db, fid))


def image_path(db, fid: str) -> Path | None:
    row = get_row(db, fid)
    if not row.get("image_path"):
        return None
    p = Path(row["image_path"])
    return p if p.is_file() else None


def set_image(db, fid: str, path: Path) -> dict:
    """Record ``path`` (already written) as the finding's image, removing an older one of the other type."""
    row = get_row(db, fid)
    old = row.get("image_path")
    if old and Path(old) != path:
        try:
            Path(old).unlink()
        except OSError:
            pass
    db.update("findings", {"id": fid}, {"image_path": str(path), "updated_utc": utc_now()})
    _mirror(db, fid)
    return finding_out(get_row(db, fid))


def delete_finding(db, fid: str) -> None:
    row = get_row(db, fid)
    db.execute("DELETE FROM findings WHERE id = ?", (fid,))
    d = findings_dir(db, row["project_id"])
    for p in [d / f"{fid}.json", *(d / f"{fid}{s}" for s in IMAGE_TYPES.values())]:
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# rendering (report section and export)
# ---------------------------------------------------------------------------

def _flatten(obj, prefix: str = "", out: list | None = None, depth: int = 0) -> list[tuple[str, str]]:
    out = [] if out is None else out
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list) and depth < 3 and all(not isinstance(x, (dict, list)) for x in obj):
        out.append((prefix, ", ".join(_scalar(x) for x in obj)))
    elif isinstance(obj, (dict, list)):
        out.append((prefix, json.dumps(obj, default=str)[:400]))
    else:
        out.append((prefix, _scalar(obj)))
    return out


def _scalar(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return "—" if v is None else str(v)


def finding_blocks(db, row: dict, *, images: str = "inline", image_name: str | None = None) -> list[dict]:
    """Report blocks of one finding: title, note, the snapshotted numbers, the image and where it came from.

    ``images``: ``inline`` (data URI) or ``file`` (Markdown ZIP: ``images/<image_name>``)."""
    f = finding_out(row)
    blocks: list[dict] = [{"t": "h", "level": 3, "text": f["title"] or "(untitled finding)"}]
    if f["note_md"]:
        blocks.append({"t": "md", "text": f["note_md"]})
    snap = _flatten(f["snapshot"]) if f["snapshot"] else []
    if snap:
        blocks.append({"t": "table", "head": ["Value on screen", ""], "rows": [[k, v] for k, v in snap],
                       "caption": "Snapshotted numbers (as shown when the finding was pinned)."})
    p = Path(row["image_path"]) if row.get("image_path") else None
    if p is not None and p.is_file():
        mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
        if images == "file":
            blocks.append({"t": "md", "text": f"![{f['title']}](images/{image_name or p.name})"})
        else:
            blocks.append({"t": "img", "mime": mime, "data": p.read_bytes(), "alt": f["title"], "caption": ""})
    where = f"View {f['view'] or '—'}" + (f" · run {f['run_id']}" if f["run_id"] else "") + \
        f" · pinned {f['created_utc']}" + (f" · state {f['url_state']}" if f["url_state"] else "")
    blocks.append({"t": "p", "text": where})
    return [{"t": "div", "cls": "finding", "blocks": blocks}]


def _run_ctx(db, run_id: str):
    from sparc.studio.runs.reader import RunContext

    row = db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        return None
    import time

    return RunContext(row, ("findings-export", time.time()))


def _run_blocks(db, run_ids: list[str]) -> list[dict]:
    """Per run: the provenance block and the generated caveats."""
    from sparc.studio.exports.report import provenance_rows
    from sparc.studio.runs.caveats import caveats_for

    out: list[dict] = []
    for rid in run_ids:
        ctx = _run_ctx(db, rid)
        if ctx is None:
            continue
        out.append({"t": "h", "level": 2, "text": f"Provenance: run {rid}"})
        out.append({"t": "kv", "rows": provenance_rows(ctx)})
        items = caveats_for(ctx)
        if items:
            out.append({"t": "h", "level": 3, "text": "Caveats generated from this run's numbers"})
            out.append({"t": "list", "items": items})
    return out


def export_findings(db, project_id: str, out_dir: Path, *, run_id: str | None = None, ids: list[str] | None = None,
                    fmt: str = "html") -> dict:
    """Write the findings export to ``out_dir``: ``findings.html`` or ``findings_md.zip``; ``{path, bytes}``."""
    from sparc.studio.exports.report import render_html, render_markdown

    if ids:
        rows = [get_row(db, fid) for fid in ids]
        bad = [r["id"] for r in rows if r["project_id"] != project_id]
        if bad:
            raise ValueError(f"findings {bad} belong to another project")
    else:
        sql = "SELECT * FROM findings WHERE project_id = ?" + (" AND run_id = ?" if run_id else "") + \
            " ORDER BY position, created_utc, id"
        rows = db.fetchall(sql, (project_id, run_id) if run_id else (project_id,))
    p = db.fetchone("SELECT name FROM projects WHERE id = ?", (project_id,)) or {}
    title = f"Findings · {p.get('name') or project_id}"
    sub = f"{len(rows)} finding{'s' if len(rows) != 1 else ''}" + (f" · run {run_id}" if run_id else "") + \
        f" · exported {utc_now()}"
    runs = list(dict.fromkeys(r["run_id"] for r in rows if r.get("run_id")))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        blocks = [b for r in rows for b in finding_blocks(db, r)] + _run_blocks(db, runs)
        path = out_dir / "findings.html"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(render_html(title, sub, blocks), encoding="utf-8")
        os.replace(tmp, path)
    else:
        blocks = []
        images = []
        for r in rows:
            ip = Path(r["image_path"]) if r.get("image_path") else None
            name = None
            if ip is not None and ip.is_file():
                name = f"{r['id']}{ip.suffix}"
                images.append((ip, name))
            blocks += finding_blocks(db, r, images="file", image_name=name)
        blocks += _run_blocks(db, runs)
        path = out_dir / "findings_md.zip"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("findings/findings.md", render_markdown(title, sub, blocks))
            for ip, name in images:
                zf.write(ip, f"findings/images/{name}")
        os.replace(tmp, path)
    return {"path": str(path), "bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------

def _reindex(db, workspace) -> dict:
    """Rebuild ``findings`` from ``projects/*/findings/<fid>.json`` (and the image next to each)."""
    n = 0
    root = Path(workspace.projects_dir)
    for mirror in sorted(root.glob("*/findings/fd_*.json")) if root.is_dir() else []:
        doc = read_json(mirror)
        if not isinstance(doc, dict) or not doc.get("id"):
            continue
        name = Path(str(doc.get("image") or "")).name      # a bare file name next to the mirror, nothing else
        img = mirror.parent / name if name and Path(name).suffix in IMAGE_TYPES.values() else None
        db.insert("findings", {
            "id": doc["id"], "project_id": doc.get("project_id"), "run_id": doc.get("run_id"),
            "view": doc.get("view"), "url_state": doc.get("url_state"), "title": doc.get("title"),
            "note_md": doc.get("note_md"), "snapshot_json": dbmod.dumps(doc.get("snapshot") or {}),
            "image_path": str(img) if img is not None and img.is_file() else None,
            "position": doc.get("position"), "created_utc": doc.get("created_utc"),
            "updated_utc": doc.get("updated_utc")}, replace=True)
        n += 1
    return {"findings": n}


try:
    dbmod.reindex_hook("findings", order=30)(_reindex)
except Exception:  # pragma: no cover
    log.exception("cannot register the findings reindex hook")
