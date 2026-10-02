"""The whole-run bundle (``export.bundle``, SPEC §6.7): a ZIP of the selected catalogued outputs, a contents
manifest and a README with the units and sign of every column.

Outputs are chosen by catalog id (:data:`sparc.core.catalog.OUTPUTS`, ``{var}`` entries expanded to the run's
levers); none means every output the run has.  ``checkpoint.pkl`` is left out unless it is asked for
(``include_checkpoint``): it is large and executes code when loaded.  Files are streamed into the archive
one by one (already-compressed formats are stored), so a bundle never sits in memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

__all__ = ["select_files", "build_bundle", "readme_text", "check_outputs", "STORED_SUFFIXES"]

#: formats that are already compressed (stored, not deflated)
STORED_SUFFIXES = (".parquet", ".npz", ".pkl", ".tif", ".tiff", ".gpkg", ".zip", ".png")
_SKIP_TOP = ("studio", "children", "__pycache__")


def _specs(cfg_raw: dict | None):
    from sparc.core.catalog import expand_outputs

    return expand_outputs(cfg_raw or {}, root="run")


def check_outputs(cfg_raw: dict | None, outputs: list[str] | None) -> list[str]:
    """The ids of ``outputs`` that are not catalogued run outputs."""
    known = {s.id for s in _specs(cfg_raw)}
    return [o for o in outputs or [] if o not in known]


def select_files(run_dir: Path, cfg_raw: dict | None, outputs: list[str] | None = None,
                 include_checkpoint: bool = False) -> list[tuple[str, str, Path]]:
    """``[(output id, relative path, path)]`` of the run's files that the bundle takes."""
    from sparc.core.catalog import is_ignored, match_output

    run_dir = Path(run_dir)
    specs = _specs(cfg_raw)
    want = set(outputs) if outputs else None
    out = []
    for dirpath, dirnames, filenames in os.walk(run_dir):
        rel_dir = Path(dirpath).relative_to(run_dir)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_TOP and not d.startswith(".")]
        dirnames.sort()
        for name in sorted(filenames):
            rel = (rel_dir / name).as_posix()
            if rel.startswith("./"):
                rel = rel[2:]
            if is_ignored(rel) or name.startswith("."):
                continue
            spec = match_output(rel, "run", specs)
            if spec is None:
                continue
            if spec.id == "checkpoint" and not include_checkpoint:
                continue
            if want is not None and spec.id not in want and not (spec.id == "checkpoint" and include_checkpoint):
                continue
            out.append((spec.id, rel, Path(dirpath) / name))
    return out


def readme_text(run: dict, cfg_raw: dict | None, files: list[tuple[str, str, Path]]) -> str:
    """README.txt: what the bundle is, the sign conventions, and the columns (with units) of each file."""
    from sparc.core.catalog import dictionary_rows, output_by_id, units_for

    units = units_for(cfg_raw or {})
    target = units.get("target") or ""
    rels = {rel for _, rel, _ in files}
    lines = [f"SPARC run bundle: {run.get('label') or run.get('id')}",
             "=" * 60, "",
             f"Run id       {run.get('id')}",
             f"Created      {run.get('created_utc') or '—'}",
             f"Code commit  {run.get('git_commit') or '—'}",
             f"Target unit  {target}", "",
             "Sign conventions",
             "----------------",
             f"* ΔT / delta / *_delta columns are changes in air temperature in {target}: negative = cooler.",
             f"* cooling / benefit columns are in {target} with positive = cooler.",
             f"* totals (planned_total_cooling, footprints) are {target}·cells: summed over the cells affected.",
             "* lever doses are in the lever's own unit (" + ", ".join(f"{v}: {u}" for v, u in
                                                               (units.get("levers") or {}).items()) + ").",
             "* logger_sites.cell and before_after_pairs.treated/control are row indices of the run "
             "(the order of predictions.parquet).", "",
             "Files", "-----"]
    for oid, rel, path in files:
        spec = output_by_id(oid, _specs(cfg_raw))
        lines.append(f"{rel:44s} {spec.label if spec else ''} ({path.stat().st_size:,} bytes)")
    rows = [r for r in dictionary_rows(cfg_raw or {}) if r["output"] in rels]
    if rows:
        lines += ["", "Columns", "-------"]
        cur = None
        for r in rows:
            if r["output"] != cur:
                cur = r["output"]
                lines += ["", cur]
            unit = f" [{r['unit']}]" if r["unit"] else ""
            sign = f" ({r['sign']})" if r["sign"] else ""
            lines.append(f"  {r['column']:30s}{unit}{sign} {r['description']}")
    if "checkpoint.pkl" in rels:
        lines += ["", "checkpoint.pkl executes code when it is loaded: open it only if you trust where it came from."]
    lines += ["", "contents.json lists every file with its size and SHA-256.", ""]
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_bundle(run: dict, run_dir: Path, cfg_raw: dict | None, out_zip: Path, outputs: list[str] | None = None,
                 include_checkpoint: bool = False, progress_cb=None) -> dict:
    """Write the bundle to ``out_zip`` (atomically); ``{path, bytes, files}``."""
    files = select_files(run_dir, cfg_raw, outputs, include_checkpoint)
    if not files:
        raise ValueError("none of the requested outputs exist in this run")
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    top = str(run.get("id") or Path(run_dir).name)
    tmp = out_zip.with_name(f".{out_zip.name}.{os.getpid()}.tmp")
    contents = []
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for i, (oid, rel, path) in enumerate(files):
                comp = zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                zf.write(path, f"{top}/{rel}", compress_type=comp)
                contents.append({"path": rel, "output": oid, "bytes": path.stat().st_size, "sha256": _sha256(path)})
                if progress_cb is not None:
                    progress_cb(i + 1, len(files), rel)
            zf.writestr(f"{top}/README.txt", readme_text(run, cfg_raw, files))
            zf.writestr(f"{top}/contents.json", json.dumps({"run_id": run.get("id"), "files": contents}, indent=1))
        os.replace(tmp, out_zip)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"path": str(out_zip), "bytes": out_zip.stat().st_size,
            "files": [f"{top}/{c['path']}" for c in contents] + [f"{top}/README.txt", f"{top}/contents.json"]}
