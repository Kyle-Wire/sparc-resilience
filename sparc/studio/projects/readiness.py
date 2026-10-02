"""The readiness spine (SPEC §9.3): one row per setup concern, with a state and a one-click action.

Rows (api.md ``ReadinessRow``): data file, columns mapped, levers (n),
physics roles (k/6), campaign forcing, climate table, people layers, config
valid, runs (fast / coarse / full), emulator on the active run, studies.
``readiness_score`` counts the ``ok`` rows among those that apply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sparc.studio.projects.config_schema import ROLE_NAMES
from sparc.studio.projects.config_service import effective_config
from sparc.studio.projects.files import is_csv_name
from sparc.studio.projects.validate import validate_deep

__all__ = ["readiness", "readiness_score", "LABELS"]

LABELS = {"data": "Data file", "columns": "Columns mapped", "levers": "Levers", "roles": "Physics roles",
          "forcing": "Campaign forcing", "climate_table": "Climate table", "people_layers": "People & land cover",
          "config_valid": "Config valid", "runs": "Runs", "emulator": "Emulator", "studies": "Studies"}
STUDY_KINDS = ("baselines", "placebo", "multiverse", "simcheck", "uncertainty")
#: post-run outputs a run folder holds once that action ran (baselines and uncertainty are not study folders)
RUN_FILE_STUDIES = {"baselines": "baselines.json", "uncertainty": "uncertainty.json", "placebo": "placebo.json"}
#: study statuses that do not count as done
_NOT_DONE = ("queued", "blocked", "starting", "running", "cancelling", "failed", "cancelled", "interrupted")


def _row(key: str, state: str, detail: str, action: dict | None = None) -> dict:
    return {"key": key, "label": LABELS[key], "state": state, "detail": detail, "action": action}


def _open(label: str, path: str) -> dict:
    return {"kind": "open", "label": label, "method": "GET", "path": path}


def _fetch(pid: str, kind: str, label: str, body: Any = None) -> dict:
    return {"kind": "fetch_input", "label": label, "method": "POST", "path": f"/api/projects/{pid}/inputs/{kind}",
            "body": body if body is not None else {}}


def _resolve(pdir: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else pdir / q


def _size(p: Path) -> str:
    n = p.stat().st_size
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _sec(v: Any) -> dict:
    """A config section as a mapping: a saved config may hold any type there (``climate: on``, ``data: 5``), which
    validation reports as an issue; the spine must still build."""
    return v if isinstance(v, dict) else {}


def readiness(db, project: dict, raw: dict, *, issues: list[dict] | None = None) -> list[dict]:
    """The spine of ``project`` (a ``projects`` row) with raw config ``raw``."""
    pid = project["id"]
    pdir = Path(project["dir"])
    eff = effective_config(raw, pdir)
    d = _sec(eff.get("data"))
    issues = validate_deep(raw, pdir) if issues is None else issues
    errors = [i for i in issues if i["level"] == "error"]
    setup = f"/p/{pid}/setup"
    out = []

    # data
    path = d.get("path")
    data_ok = False
    if not path:
        out.append(_row("data", "missing", "no data file", _open("Add data", f"{setup}/data")))
    elif not _resolve(pdir, str(path)).is_file():
        out.append(_row("data", "missing", f"{path} not found", _open("Fix data file", f"{setup}/data")))
    elif not is_csv_name(Path(str(path)).name):
        out.append(_row("data", "missing", f"{Path(str(path)).name} is not a CSV (core reads the point table as CSV)",
                        _open("Fix data file", f"{setup}/data")))
    else:
        data_ok = True
        out.append(_row("data", "ok", f"{Path(str(path)).name} · {_size(_resolve(pdir, str(path)))}"))

    # columns
    col_errors = [i for i in errors if i["path"] in ("data.target", "data.x", "data.y", "data.id")
                  or (i["path"].startswith("predictors") and i["code"] in ("missing_column", "not_numeric"))]
    preds = list(eff.get("predictors") or []) if isinstance(eff.get("predictors"), list) else []
    if not data_ok:
        out.append(_row("columns", "missing", "add a data file first", _open("Map columns", f"{setup}/data")))
    elif col_errors or not d.get("target"):
        what = col_errors[0]["message"] if col_errors else "target not set"
        out.append(_row("columns", "missing", what, _open("Map columns", f"{setup}/data")))
    else:
        out.append(_row("columns", "ok" if d.get("id") else "warn",
                        f"target {d.get('target')}, x/y {d.get('x')}/{d.get('y')}, {len(preds)} predictors"
                        + ("" if d.get("id") else ", no id column"),
                        None if d.get("id") else _open("Set the id column", f"{setup}/data")))

    # levers
    act = _sec(eff.get("actionable"))
    lever_errors = [i for i in errors if i["path"].startswith("actionable")]
    if not preds:
        out.append(_row("levers", "missing", "no predictors yet", _open("Choose predictors", f"{setup}/levers")))
    elif not act:
        out.append(_row("levers", "warn", "no levers: scenarios, responses and the Lab need one",
                        _open("Add levers", f"{setup}/levers")))
    elif lever_errors:
        out.append(_row("levers", "missing", lever_errors[0]["message"], _open("Fix levers", f"{setup}/levers")))
    else:
        out.append(_row("levers", "ok", f"{len(act)} lever{'s' if len(act) != 1 else ''}: {', '.join(map(str, act))}"))

    # roles
    phys = _sec(eff.get("physics"))
    roles = _sec(phys.get("roles"))
    mapped = [r for r in ROLE_NAMES if roles.get(r) and roles.get(r) in preds]
    role_errors = [i for i in errors if i["path"].startswith("physics.roles")]
    k = len(mapped)
    if role_errors:
        out.append(_row("roles", "missing", role_errors[0]["message"], _open("Fix roles", f"{setup}/physics")))
    elif k == len(ROLE_NAMES):
        out.append(_row("roles", "ok", f"{k}/{len(ROLE_NAMES)} roles mapped"))
    elif k == 0:
        out.append(_row("roles", "missing", f"0/{len(ROLE_NAMES)} roles: the physics model is skipped",
                        _open("Map roles", f"{setup}/physics")))
    else:
        miss = [r for r in ROLE_NAMES if r not in mapped]
        warn = "; placebo, simcheck, planner and emulator need canopy and impervious" \
            if {"canopy", "impervious"} & set(miss) else ""
        out.append(_row("roles", "warn", f"{k}/{len(ROLE_NAMES)} roles mapped (missing {', '.join(miss)}){warn}",
                        _open("Map roles", f"{setup}/physics")))

    crs = d.get("crs") or d.get("reproject_to")
    # forcing
    forcing = phys.get("forcing")
    if not phys.get("enabled", True):
        out.append(_row("forcing", "n/a", "physics disabled"))
    elif forcing:
        fp = _resolve(pdir, str(forcing))
        f_err = [i for i in errors if i["path"] == "physics.forcing"]
        if f_err:
            out.append(_row("forcing", "missing", f_err[0]["message"], _open("Fix forcing", f"{setup}/inputs")))
        else:
            info = _sec(phys.get("forcing_info"))
            when = f" ({info['date']})" if info.get("date") else ""
            out.append(_row("forcing", "ok", f"{fp.name}{when}: SW↓ {phys.get('sw_down')} W/m², "
                                             f"LW net {phys.get('lw_net')} W/m²"))
    else:
        out.append(_row("forcing", "warn", "generic radiation (800 W/m²) and no wind: fetch the campaign day",
                        _open("Fetch campaign forcing", f"{setup}/inputs")))

    # climate table
    clim = _sec(eff.get("climate"))
    if not clim.get("enabled"):
        out.append(_row("climate_table", "n/a", "climate projections are off",
                        _open("Set up climate", f"{setup}/analysis")))
    elif str(clim.get("source", "table")) == "cmip6":
        out.append(_row("climate_table", "ok", "fetched from CMIP6 at run time"))
    elif clim.get("table") and _resolve(pdir, str(clim["table"])).is_file():
        out.append(_row("climate_table", "ok", Path(str(clim["table"])).name))
    else:
        action = _fetch(pid, "cmip6", "Fetch CMIP6 change factors") if (crs or clim.get("site")) else \
            _open("Set up climate", f"{setup}/inputs")
        out.append(_row("climate_table", "missing",
                        f"{clim.get('table')} not found" if clim.get("table") else "no change-factor table", action))

    # people layers
    planner = _sec(eff.get("planner"))
    layers = planner.get("layers")
    people_obj = str(_sec(eff.get("optimize")).get("objective", "cooling")) == "people"
    if layers and _resolve(pdir, str(layers)).is_file():
        out.append(_row("people_layers", "ok", Path(str(layers)).name))
    else:
        action = _fetch(pid, "layers", "Fetch people & land cover", {"link": True}) if crs else \
            _open("Set the CRS", f"{setup}/data")
        state = "missing" if (people_obj or layers) else "warn"
        detail = f"{layers} not found" if layers else "not fetched: the planner pack and people weighting need them"
        out.append(_row("people_layers", state, detail, action))

    # config valid
    if errors:
        out.append(_row("config_valid", "missing", f"{len(errors)} error{'s' if len(errors) != 1 else ''}: "
                        f"{errors[0]['message']}", _open("Open config", f"/p/{pid}/config")))
    else:
        warns = sum(1 for i in issues if i["level"] == "warn")
        out.append(_row("config_valid", "ok", "no errors" + (f", {warns} warning{'s' if warns != 1 else ''}"
                                                             if warns else "")))

    # runs
    runs = db.fetchall("SELECT id, mode, status, has_emulator, created_utc FROM runs WHERE project_id = ? "
                       "ORDER BY created_utc DESC", (pid,))
    good = [r for r in runs if r["status"] in ("complete", "imported", "partial")]
    if not runs:
        out.append(_row("runs", "missing", "no runs yet", _open("Launch first run", f"/p/{pid}/launch")))
    else:
        modes = sorted({("coarse" if str(r.get("mode") or "").startswith("coarse") else str(r.get("mode") or "custom"))
                        for r in good})
        detail = f"{len(runs)} run{'s' if len(runs) != 1 else ''}" + (f": {', '.join(modes)}" if modes else "")
        if not good:
            out.append(_row("runs", "warn", detail + " (none finished)", _open("Launch", f"/p/{pid}/launch")))
        elif modes == ["fast"]:
            out.append(_row("runs", "warn", detail + " (fast only)", _open("Launch a full run", f"/p/{pid}/launch")))
        else:
            out.append(_row("runs", "ok", detail))

    # emulator on the active run
    active = project.get("active_run_id") or (good[0]["id"] if good else None)
    arow = next((r for r in runs if r["id"] == active), None)
    if arow is None:
        out.append(_row("emulator", "n/a", "no finished run"))
    elif arow.get("has_emulator"):
        out.append(_row("emulator", "ok", f"built for {arow['id']}"))
    else:
        out.append(_row("emulator", "warn", f"none for {arow['id']}: Lab previews need it",
                        {"kind": "build_emulator", "label": "Build emulator", "method": "POST",
                         "path": f"/api/runs/{arow['id']}/actions/emulator", "body": {}}))

    # studies: study folders of the project (placebo, multiverse, simcheck) plus the post-run outputs in the
    # active run's folder (baselines.json, uncertainty.json; a CLI placebo leaves placebo.json there too)
    if not good:
        out.append(_row("studies", "n/a", "no finished run"))
    else:
        rid = active or good[0]["id"]
        marks = ",".join("?" for _ in _NOT_DONE)
        kinds = {r["kind"] for r in db.fetchall(f"SELECT DISTINCT kind FROM studies WHERE project_id = ? AND "
                                                f"(status IS NULL OR status NOT IN ({marks}))", (pid, *_NOT_DONE))}
        run_dir = db.fetchval("SELECT run_dir FROM runs WHERE id = ?", (rid,))
        if run_dir:
            kinds |= {k for k, f in RUN_FILE_STUDIES.items() if (Path(run_dir) / f).is_file()}
        done = [k for k in STUDY_KINDS if k in kinds]
        if len(done) == len(STUDY_KINDS):
            out.append(_row("studies", "ok", ", ".join(done)))
        else:
            missing = [k for k in STUDY_KINDS if k not in kinds]
            out.append(_row("studies", "warn", (f"done: {', '.join(done)}; " if done else "") +
                            f"not run: {', '.join(missing)}", _open("Open validation", f"/r/{rid}/validation")))
    return out


def readiness_score(rows: list[dict]) -> dict:
    applicable = [r for r in rows if r["state"] != "n/a"]
    return {"done": sum(1 for r in applicable if r["state"] == "ok"), "total": len(applicable)}
