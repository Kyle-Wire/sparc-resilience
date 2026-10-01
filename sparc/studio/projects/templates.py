"""Project templates (SPEC §9.1–9.2): blank, synthetic demo, Providence example, and config import.

* ``blank`` - a ``core:`` block with the data keys to fill in.
* ``synthetic_demo`` - :func:`sparc.core.synthetic.write_demo_project`
  writes everything (data, truth, DEMO layers and climate table, config);
  Studio adds only ``project.json``, so the files are byte-identical to the
  generator's for the same ``n`` and ``seed`` (the replay e2e relies on it).
* ``providence_example`` - the Brown University / Providence inputs, from a
  repo checkout when present, else the packaged copy in
  ``sparc/studio/examples/providence/`` (``brown4.csv.gz`` is decompressed).
  The config's paths are rewritten to the project layout in place, so its
  comments survive.  From a checkout, ``import_existing_runs`` registers the
  runs and studies under ``output/core/providence/`` in place through the
  runs and studies items' lazy contracts (SPEC §10.2).
* ``POST /api/projects/import`` - an existing config with its paths made
  absolute (or its data copied in), plus run and study folders.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib
import inspect
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from sparc.studio.errors import ApiError
from sparc.studio.projects import service
from sparc.studio.projects.config_service import (
    CONFIG_NAME,
    absolutize_paths,
    core_block,
    dump_config,
    get_dotted,
    parse_yaml,
    path_keys,
    set_dotted,
)
from sparc.studio.workspace import slugify

log = logging.getLogger("sparc.studio.projects")

__all__ = ["TEMPLATES", "EXAMPLE_DIR", "repo_root", "providence_sources", "write_blank", "write_providence",
           "create_from_template", "import_config", "call_contract", "MISSING", "PROVIDENCE_RUNS"]

TEMPLATES = ("blank", "synthetic_demo", "providence_example")
EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "providence"
PROVIDENCE_RUNS = ("providence_uhi", "providence_uhi_fast")
PROVIDENCE_STUDIES = (("placebo", "placebo"), ("simcheck*", "simcheck"), ("multiverse", "multiverse"))
#: example inputs: (source path under the configs folder or the packaged example, project path)
_PROV_FILES = {
    "physics.forcing": ("forcing/providence_2020-07-29.json", "inputs/forcing/providence_2020-07-29.json"),
    "climate.table": ("climate/providence_cmip6_tasmax_jja.csv", "inputs/climate/providence_cmip6_tasmax_jja.csv"),
    "planner.layers": ("layers/providence_layers.parquet", "inputs/layers/providence_layers.parquet"),
}
_PROV_DATA = "data/brown4.csv"
UNAVAILABLE = "run import unavailable in this build"


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def repo_root() -> Path | None:
    """The repository checkout this ``sparc`` package was imported from, when it has the Providence inputs."""
    import sparc

    root = Path(sparc.__file__).resolve().parent.parent
    need = [root / "configs" / "core_providence.yml", root / "brown4.csv"] + \
        [root / "configs" / src for src, _dst in _PROV_FILES.values()]
    return root if all(p.is_file() for p in need) else None


def providence_sources() -> dict | None:
    """Where the example comes from: ``{kind: repo|packaged, root, config, data, files: {key: path}}``."""
    root = repo_root()
    if root is not None:
        return {"kind": "repo", "root": root, "config": root / "configs" / "core_providence.yml",
                "data": root / "brown4.csv",
                "files": {k: root / "configs" / src for k, (src, _dst) in _PROV_FILES.items()}}
    ex = EXAMPLE_DIR
    need = [ex / "core_providence.yml", ex / "brown4.csv.gz"] + [ex / src for src, _dst in _PROV_FILES.values()]
    if all(p.is_file() for p in need):
        return {"kind": "packaged", "root": None, "config": ex / "core_providence.yml", "data": ex / "brown4.csv.gz",
                "files": {k: ex / src for k, (src, _dst) in _PROV_FILES.items()}}
    return None


# ---------------------------------------------------------------------------
# writers (files only)
# ---------------------------------------------------------------------------

def write_blank(project_dir: Path, name: str) -> None:
    raw = {"name": slugify(name).replace("-", "_") or "core_run",
           "data": {"path": None, "target": None, "id": None, "x": "x", "y": "y", "coord_unit": "m", "crs": None},
           "predictors": []}
    for sub in ("data", "inputs/forcing", "inputs/climate", "inputs/layers", "inputs/features"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)
    from sparc.core import runio

    runio.write_text_atomic(project_dir / CONFIG_NAME, dump_config(raw))


def _rewrite_paths(text: str, targets: dict[str, str]) -> str:
    """Set ``targets`` (dotted key → value) in YAML ``text``, keeping comments when a line edit can.

    Each key's ``leaf: <old value>`` line is rewritten in place; the result is
    parsed back and must equal the original with exactly those keys changed,
    otherwise the config is re-dumped (comments lost).
    """
    doc = parse_yaml(text)
    raw = core_block(doc)
    expected = core_block(parse_yaml(text))
    for key, value in targets.items():
        set_dotted(expected, key, value)
    out = text
    for key, value in targets.items():
        old = get_dotted(raw, key)
        leaf = key.split(".")[-1]
        if old is None:
            continue
        pat = re.compile(rf"^(\s*{re.escape(leaf)}:\s*)(['\"]?){re.escape(str(old))}\2(\s*(#.*)?)$", re.M)
        out, n = pat.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", out, count=1)
        if n != 1:
            break
    try:
        if core_block(parse_yaml(out)) == expected:
            return out
    except ApiError:
        pass
    return dump_config(expected, {k: v for k, v in doc.items() if k != "core"} if "core" in doc else None)


def write_providence(project_dir: Path, src: dict) -> None:
    """Copy the example inputs into the project and write its config with project-relative paths."""
    from sparc.core import runio

    (project_dir / "data").mkdir(parents=True, exist_ok=True)
    data_dst = project_dir / _PROV_DATA
    if str(src["data"]).endswith(".gz"):
        with gzip.open(src["data"], "rb") as fin, runio.atomic_open(data_dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
    else:
        with open(src["data"], "rb") as fin, runio.atomic_open(data_dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
    targets = {"data.path": _PROV_DATA, "output.dir": "runs"}
    for key, path in src["files"].items():
        dst = project_dir / _PROV_FILES[key][1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        targets[key] = _PROV_FILES[key][1]
    (project_dir / "inputs" / "features").mkdir(parents=True, exist_ok=True)
    text = Path(src["config"]).read_text(encoding="utf-8")
    runio.write_text_atomic(project_dir / CONFIG_NAME, _rewrite_paths(text, targets))


# ---------------------------------------------------------------------------
# cross-item contracts (SPEC §10.2)
# ---------------------------------------------------------------------------

async def call_contract(sctx, module: str, func: str, *args, **kwargs) -> Any:
    """Call ``module.func`` (imported lazily); :data:`MISSING` when the module or function is not installed.

    A ``sctx``/``ctx`` parameter, when the callee declares one, receives the
    server context.  Sync callees run in a worker thread; a coroutine result
    is awaited.  Pydantic results are dumped to dicts.
    """
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == module or module.startswith(exc.name + ".")):
            log.warning("%s is not installed; %s skipped", module, func)
            return MISSING
        raise
    fn = getattr(mod, func, None)
    if fn is None:
        log.warning("%s has no %s; skipped", module, func)
        return MISSING
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    for name in ("sctx", "ctx"):
        if name in params and name not in kwargs:
            kwargs[name] = sctx
            break
    if inspect.iscoroutinefunction(fn):
        res = await fn(*args, **kwargs)
    else:
        res = await asyncio.to_thread(fn, *args, **kwargs)
        if inspect.isawaitable(res):
            res = await res
    if hasattr(res, "model_dump"):
        res = res.model_dump(mode="json")
    return res


async def _import_runs(sctx, project_id: str, run_dirs: list[Path], study_dirs: list[tuple[Path, str | None]], *,
                       config_path: Path | None, trust_pickles: bool, target_run: str | None = None) \
        -> tuple[list[dict], list[dict], list[str]]:
    """Import run and study folders in place; ``(runs, studies, warnings)``."""
    runs, studies, warnings = [], [], []
    for d in run_dirs:
        sctx.paths.allow(d)
        res = await call_contract(sctx, "sparc.studio.runs.registry", "import_run", d, project_id,
                                  config_path=config_path, trust_pickles=trust_pickles)
        if res is MISSING:
            warnings.append(f"{UNAVAILABLE}: {d.name} was not imported")
            continue
        runs.append(res if isinstance(res, dict) else {"id": str(res)})
    first = target_run or (runs[0].get("id") if runs else None)
    for d, kind in study_dirs:
        sctx.paths.allow(d)
        res = await call_contract(sctx, "sparc.studio.studies.service", "import_study_dir", d, project_id,
                                  kind=kind, target_run_id=first)
        if res is MISSING:
            warnings.append(f"{UNAVAILABLE}: study folder {d.name} was not imported")
            continue
        studies.append(res if isinstance(res, dict) else {"id": str(res)})
    return runs, studies, warnings


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def _claim(sctx, name: str) -> tuple[str, Path]:
    """Claim the folder of a new project named ``name``; ``409 conflict`` when its slug exists."""
    db = sctx.db
    slug = slugify(name)
    taken = {r["slug"] for r in db.fetchall("SELECT slug FROM projects")}
    d = sctx.workspace.project_dir(slug)
    if slug in taken or d.exists():
        raise ApiError("conflict", f"a project named {slug!r} already exists; choose another name",
                       detail={"slug": slug})
    sctx.workspace.projects_dir.mkdir(parents=True, exist_ok=True)
    try:
        d.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise ApiError("conflict", f"a project named {slug!r} already exists", detail={"slug": slug})
    return slug, d


async def create_from_template(sctx, name: str, template: str, options: dict | None = None) -> dict:
    """``POST /api/projects``: ``{project_row, imported_runs, warnings}``."""
    options = dict(options or {})
    if template not in TEMPLATES:
        raise ApiError("validation", f"unknown template {template!r}",
                       detail={"errors": [{"path": "template", "message": f"one of {', '.join(TEMPLATES)}",
                                           "code": "unknown_template"}]})
    src = None
    if template == "providence_example":
        src = providence_sources()
        if src is None:
            raise ApiError("example_unavailable", "the Providence example is not available: no repo checkout and "
                           "no packaged copy in this install")
    slug, pdir = _claim(sctx, name)
    warnings: list[str] = []
    demo = False
    meta: dict = {"source": {"template": template}}
    try:
        if template == "blank":
            await asyncio.to_thread(write_blank, pdir, name)
        elif template == "synthetic_demo":
            from sparc.core.synthetic import write_demo_project

            n, seed = int(options.get("n") or 96), int(options.get("seed") or 0)
            await asyncio.to_thread(write_demo_project, pdir, n=n, seed=seed)
            for sub in ("inputs/forcing", "inputs/features"):
                (pdir / sub).mkdir(parents=True, exist_ok=True)
            demo = True
            meta["source"].update(n=n, seed=seed)
        else:
            await asyncio.to_thread(write_providence, pdir, src)
            meta["source"]["from"] = src["kind"]
        row = await asyncio.to_thread(service.create_project, sctx.db, sctx.workspace, name=name, template=template,
                                      demo=demo, slug=slug, project_dir=pdir, meta=meta, note=f"template {template}")
    except BaseException:
        shutil.rmtree(pdir, ignore_errors=True)
        raise
    imported: list[str] = []
    if template == "providence_example":
        out_dir = (src["root"] / "output" / "core" / "providence") if src.get("root") else None
        want = options.get("import_existing_runs")
        if want is None:
            want = bool(out_dir and out_dir.is_dir())
        if want and not (out_dir and out_dir.is_dir()):
            warnings.append("no output/core/providence folder next to this install: nothing to import")
        elif want:
            run_dirs = [out_dir / r for r in PROVIDENCE_RUNS if (out_dir / r).is_dir()]
            study_dirs = []
            for pattern, kind in PROVIDENCE_STUDIES:
                study_dirs.extend((p, kind) for p in sorted(out_dir.glob(pattern)) if p.is_dir())
            runs, _studies, w = await _import_runs(sctx, row["id"], run_dirs, study_dirs,
                                                   config_path=Path(row["config_path"]),
                                                   trust_pickles=bool(options.get("trust_pickles", False)))
            imported = [r.get("id") for r in runs if r.get("id")]
            warnings.extend(w)
            if imported:
                warnings.append("The full run has no provenance config; Studio uses the example config for it.")
    return {"row": row, "imported_runs": imported, "warnings": warnings}


def _copy_into(pdir: Path, src: Path, sub: str) -> str:
    dst_dir = pdir / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    return f"{sub}/{src.name}"


async def import_config(sctx, *, config_path: str, name: str | None = None, copy_data: bool = False,
                        run_dirs: list[str] | None = None, study_dirs: list[str] | None = None,
                        trust_pickles: bool = False) -> dict:
    """``POST /api/projects/import``: ``{row, runs, studies, warnings}``."""
    from sparc.core.config import core_config_from_dict
    from sparc.studio.projects.validate import validate_deep

    cpath = Path(config_path).expanduser()
    if not cpath.is_absolute() or not cpath.is_file():
        raise ApiError("not_found", f"config not found: {config_path}", detail={"path": str(config_path)})
    cpath = cpath.resolve()
    text = cpath.read_text(encoding="utf-8")
    doc = parse_yaml(text)
    raw = core_block(doc)
    base = cpath.parent
    abs_raw = absolutize_paths(raw, base)
    if isinstance(abs_raw.get("output"), dict) and abs_raw["output"].get("dir"):
        od = Path(str(abs_raw["output"]["dir"]))
        abs_raw["output"]["dir"] = str(od if od.is_absolute() else (base / od).resolve())
    try:
        core_config_from_dict(abs_raw, base_dir=base)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errs = [{"path": i["path"], "message": i["message"], "code": i["code"]}
                for i in validate_deep(abs_raw, base) if i["level"] == "error"]
        if not errs:
            errs = [{"path": "", "message": str(exc), "code": "invalid"}]
        raise ApiError("validation", f"the config is not valid: {exc}", detail={"errors": errs})
    for d in [Path(p).expanduser() for p in (run_dirs or []) + (study_dirs or [])]:
        if not d.is_dir():
            raise ApiError("not_found", f"folder not found: {d}", detail={"path": str(d)})
    name = (name or str(raw.get("name") or cpath.stem)).strip()
    slug, pdir = _claim(sctx, name)
    warnings = ["Comments of the imported YAML are not kept in the project's config."]
    try:
        final = abs_raw
        if copy_data:
            final = absolutize_paths(abs_raw, base)          # a deep copy: the copies' paths replace these
            for key in path_keys(final):
                v = get_dotted(final, key)
                if not isinstance(v, str) or not v:
                    continue
                p = Path(v)
                if not p.is_file():
                    warnings.append(f"{key}: {v} not found; not copied")
                    continue
                sub = {"physics.forcing": "inputs/forcing", "climate.table": "inputs/climate",
                       "planner.layers": "inputs/layers"}.get(key, "data")
                set_dotted(final, key, await asyncio.to_thread(_copy_into, pdir, p, sub))
        else:
            sctx.paths.allow(base)
            for key in path_keys(final):
                v = get_dotted(final, key)
                if isinstance(v, str) and v and Path(v).parent.is_dir():
                    sctx.paths.allow(Path(v).parent)
        from sparc.core import runio

        others = {k: v for k, v in doc.items() if k != "core"} if "core" in doc else None
        runio.write_text_atomic(pdir / CONFIG_NAME, dump_config(final, others))
        for sub in ("data", "inputs/forcing", "inputs/climate", "inputs/layers", "inputs/features"):
            (pdir / sub).mkdir(parents=True, exist_ok=True)
        meta = {"source": {"template": "existing_config", "config_path": str(cpath), "copy_data": bool(copy_data)}}
        row = await asyncio.to_thread(service.create_project, sctx.db, sctx.workspace, name=name,
                                      template="existing_config", slug=slug, project_dir=pdir, meta=meta,
                                      note=f"imported from {cpath}")
    except BaseException:
        shutil.rmtree(pdir, ignore_errors=True)
        raise
    try:
        runs, studies, w = await _import_runs(
            sctx, row["id"], [Path(p).expanduser().resolve() for p in run_dirs or []],
            [(Path(p).expanduser().resolve(), None) for p in study_dirs or []],
            config_path=cpath, trust_pickles=trust_pickles)
    except ApiError:
        await asyncio.to_thread(service.delete_project, sctx.db, sctx.workspace, row["id"], files=True)
        raise
    return {"row": row, "runs": runs, "studies": studies, "warnings": warnings + w}
