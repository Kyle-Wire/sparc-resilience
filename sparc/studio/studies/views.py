"""What the Validation tab, the studies hub and Mission Control's child matrix read (SPEC §5.11, §8, api.md §9):

* :func:`status_rows` - one ``StudyStatusRow`` per kind for a run (state, headline, cost estimate, attached,
  the launch action and the requirement checks: roles, layers, checkpoint);
* :func:`requirements` - what a kind needs from the run (``422 requirements`` lists ``missing``);
* :func:`study_view` - the per-kind view of a study, read live from its folder while it runs (the placebo
  matrix, the simcheck generator × seed grid from ``simcheck.jsonl``, the multiverse board, the reproduce
  checklist, the benchmark);
* :func:`truth_rows` - the synthetic demo's "Truth vs recovered" card (``GET /runs/{rid}/truth``, SPEC §9.2).
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.studies import service
from sparc.studio.workspace import parse_utc, read_json, utc_iso

log = logging.getLogger("sparc.studio.studies")

__all__ = ["status_rows", "requirements", "study_view", "truth_rows", "placebo_view", "simcheck_view",
           "multiverse_view", "reproduce_view", "benchmark_view", "default_params"]

_JOB_STATE = {"queued": "queued", "blocked": "queued", "starting": "running", "running": "running",
              "cancelling": "running", "succeeded": "done", "failed": "failed", "interrupted": "failed",
              "cancelled": "failed", "done": "done"}
_FILES = {"baselines": ("baselines.json", "baselines"), "planner": ("planner/planner.json", "planner"),
          "emulator": ("emulator.json", "emulator"), "uncertainty": ("uncertainty.json", "uncertainty"),
          "writeup": ("methods.md", None)}


def _f(v, d: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.{d}f}" if math.isfinite(x) else "—"


def default_params(kind: str) -> dict:
    """The params a kind runs with when launched without any (for the status rows' estimates)."""
    from sparc.studio.studies.kinds import PARAMS

    if kind == "simcheck":
        return {"design": {"physics": 4, "additive": 4, "null": 4}}
    model = PARAMS.get(kind)
    return model().model_dump() if model is not None and kind != "simcheck" else {}


# ---------------------------------------------------------------------------
# requirements
# ---------------------------------------------------------------------------

def _roles_missing(ctx, need_all: bool) -> list[str]:
    raw = ctx.cfg_raw or {}
    roles = ((raw.get("physics") or {}).get("roles")) or {}
    preds = set(raw.get("predictors") or [])
    have = {r: bool(roles.get(r)) and roles.get(r) in preds for r in ("canopy", "impervious")}
    missing = [f"physics.roles.{r}" for r, ok in have.items() if not ok]
    if not need_all and any(have.values()):
        return []
    return missing


def _data_missing(ctx) -> list[str]:
    cfg = ctx.cfg
    if ctx.exists("input_frame.parquet"):
        return []
    try:
        return [] if cfg is not None and Path(cfg.data_path).is_file() else ["data.path"]
    except Exception:
        return ["data.path"]


def requirements(ctx, kind: str) -> dict:
    """``{ok, missing}``: what ``kind`` needs from the run behind ``ctx`` (a ``RunContext``).

    ``missing`` names config keys (``planner.layers``, ``physics.roles.canopy``, ``data.path``) or run files
    (``manifest``, ``predictions``, ``checkpoint``)."""
    missing: list[str] = []
    has_manifest = ctx.manifest_raw is not None
    if kind in ("baselines", "planner", "emulator", "uncertainty", "writeup", "reproduce") and not has_manifest:
        missing.append("manifest")
    if kind in ("baselines", "planner", "emulator") and not ctx.exists("predictions.parquet"):
        missing.append("predictions")
    if kind == "planner":
        layers = ((ctx.cfg_raw or {}).get("planner") or {}).get("layers")
        if not layers:
            missing.append("planner.layers")
        else:
            p = Path(str(layers))
            if not p.is_absolute() and ctx.config_dir:
                p = Path(ctx.config_dir) / p
            if not p.exists():
                missing.append("planner.layers")
    if kind == "emulator" and not ctx.exists("checkpoint.pkl"):
        missing.append("checkpoint")
    if kind == "placebo":
        missing += _roles_missing(ctx, need_all=False) + _data_missing(ctx)
    if kind == "simcheck":
        missing += _roles_missing(ctx, need_all=True) + _data_missing(ctx)
    if kind in ("multiverse", "baselines", "planner"):
        missing += [m for m in _data_missing(ctx) if m not in missing]
    if kind == "reproduce" and has_manifest:
        m = ctx.manifest_raw or {}
        prov = m.get("provenance") or {}
        if prov.get("input_kind") == "frame":
            missing.append("data.path")          # made from an in-memory table: nothing to re-read
        if not isinstance(m.get("config"), dict):
            missing.append("manifest.config")
        launch = ctx.launch or {}
        if not (launch.get("config_dir") or (ctx.import_rec or {}).get("config_path") or prov.get("config_dir")):
            missing.append("config_dir")
    return {"ok": not missing, "missing": list(dict.fromkeys(missing))}


# ---------------------------------------------------------------------------
# status rows
# ---------------------------------------------------------------------------

def _latest_job(db, run_id: str, kind: str) -> dict | None:
    return db.fetchone("SELECT id, status, finished_utc, created_utc, result_json FROM jobs WHERE run_id = ? "
                       "AND kind = ? ORDER BY created_utc DESC, rowid DESC", (run_id, kind))


def _mtime_iso(p: Path) -> str | None:
    try:
        return utc_iso(p.stat().st_mtime)
    except OSError:
        return None


def _post_headline(kind: str, ctx) -> str | None:
    m = ctx.manifest or {}
    if kind == "baselines":
        b = m.get("baselines") or ctx.json("baselines.json") or {}
        return str(b["verdict"]) if b.get("verdict") else None
    if kind == "planner":
        p = m.get("planner") or ctx.json("planner/planner.json") or {}
        if p.get("people_total") is not None:
            return f"{p['people_total']:,.0f} residents" + (f"; package {p['package']}" if p.get("package") else "")
        return None
    if kind == "emulator":
        e = ctx.json("emulator.json") or {}
        rates = [((d or {}).get("validation") or {}).get("patch_pass_rate") for d in (e.get("levers") or {}).values()]
        rates = [r for r in rates if isinstance(r, (int, float))]
        return f"patch pass rate {min(rates):.0%}–{max(rates):.0%}" if rates else None
    if kind == "uncertainty":
        u = m.get("uncertainty") or ctx.json("uncertainty.json") or {}
        rows = u.get("scenarios") or []
        if not rows:
            return None
        excl = sum(1 for r in rows if r.get("envelope_excludes_zero"))
        return f"{excl} of {len(rows)} scenario envelopes exclude zero"
    if kind == "writeup":
        return "methods.md and model_card.md" if ctx.exists("methods.md") else None
    return None


def _uncertainty_stale(db, ctx) -> bool:
    """An attached study finished after ``uncertainty.json`` was written."""
    p = ctx.path("uncertainty.json")
    try:
        t = p.stat().st_mtime
    except OSError:
        return False
    for s in service.attached_studies_of(db, ctx.run_id):
        if s.get("status") in ("succeeded", "done") and (parse_utc(s.get("updated_utc")) or 0) > t + 1:
            return True
    return False


def _study_for(db, rid: str, kind: str) -> dict | None:
    """The run's study of ``kind``: attached first, then the newest one targeting the run."""
    rows = db.fetchall("SELECT s.*, l.attached AS attached FROM studies s LEFT JOIN study_links l "
                       "ON l.study_id = s.id AND l.run_id = ? WHERE s.kind = ? AND (s.target_run_id = ? OR "
                       "l.attached = 1) ORDER BY COALESCE(l.attached, 0) DESC, s.created_utc DESC", (rid, kind, rid))
    return rows[0] if rows else None


def status_rows(sctx, ctx) -> list[dict]:
    """``GET /runs/{rid}/studies``: one row per kind of :data:`service.ROW_KINDS`."""
    from sparc.studio.studies.kinds import estimate_kind

    db = sctx.db
    rid = ctx.run_id
    run_row = ctx.row
    out = []
    for kind in service.ROW_KINDS:
        row = {"kind": kind, "state": "not_run", "study_id": None, "job_id": None, "updated_utc": None,
               "headline": None, "estimate": None, "attached": None, "action": None,
               "requirements": {"ok": True, "missing": []}}
        if kind in service.POST_KINDS:
            req = requirements(ctx, kind)
            j = _latest_job(db, rid, f"post.{kind}")
            rel, key = _FILES[kind]
            have = ctx.exists(rel) or (kind == "baselines" and bool((ctx.manifest_raw or {}).get("baselines")))
            if j is not None and j["status"] in service.ACTIVE_JOB:
                row["state"] = _JOB_STATE[j["status"]]
            elif have:
                row["state"] = "done"
                if (key and ctx.stale_file(rel, key)) or (kind == "uncertainty" and _uncertainty_stale(db, ctx)):
                    row["state"] = "stale"
            elif j is not None:
                row["state"] = _JOB_STATE.get(j["status"], "failed")
            if j is not None:
                row["job_id"] = j["id"]
                row["updated_utc"] = j.get("finished_utc") or j.get("created_utc")
            elif have:
                row["updated_utc"] = _mtime_iso(ctx.path(rel))
            row["headline"] = _post_headline(kind, ctx) if have else None
            row["requirements"] = req
            row["action"] = {"kind": "build_emulator" if kind == "emulator" else "run_job",
                             "label": {"baselines": "Score the baselines", "planner": "Run the planner pack",
                                       "emulator": "Build the emulator", "uncertainty": "Compute uncertainty",
                                       "writeup": "Regenerate methods & model card"}[kind],
                             "method": "POST", "path": f"/api/runs/{rid}/actions/{kind}", "body": {}}
            est_kind = kind
        elif kind in service.RUN_STUDY_KINDS:
            req = requirements(ctx, kind)
            s = _study_for(db, rid, kind)
            if s is not None:
                st = service.effective_status(db, s)
                row["state"] = _JOB_STATE.get(st, "done")
                if row["state"] == "done" and rid in service.stale_vs(db, s):
                    row["state"] = "stale"
                summ = dbmod.loads(s.get("summary_json")) or {}
                row.update(study_id=s["id"], job_id=s.get("job_id"), updated_utc=s.get("updated_utc"),
                           headline=summ.get("headline"),
                           attached=bool(s.get("attached")) if kind in service.ATTACHABLE else None)
            elif kind in service.ATTACHABLE:
                row["attached"] = False
            row["requirements"] = req
            row["action"] = {"kind": "run_job", "label": f"Run the {kind} study", "method": "POST",
                             "path": f"/api/runs/{rid}/studies/{kind}", "body": default_params(kind)}
            est_kind = kind
        elif kind == "literature":
            lit = (ctx.manifest or {}).get("literature") or {}
            rows = [r for r in lit.get("rows") or [] if r.get("sparc_C") is not None]
            if rows:
                within = sum(1 for r in rows if r.get("within_factor_2"))
                row.update(state="done", headline=f"{within} of {len(rows)} published rates within ×2")
            est_kind = None
        else:                                   # benchmark: the project's newest
            s = db.fetchone("SELECT * FROM studies WHERE project_id = ? AND kind = 'benchmark' "
                            "ORDER BY created_utc DESC", (ctx.project_id,)) if ctx.project_id else None
            if s is not None:
                summ = dbmod.loads(s.get("summary_json")) or {}
                row.update(state=_JOB_STATE.get(service.effective_status(db, s), "done"), study_id=s["id"],
                           job_id=s.get("job_id"), updated_utc=s.get("updated_utc"), headline=summ.get("headline"))
            if ctx.project_id:
                row["action"] = {"kind": "run_job", "label": "Run the effect benchmark", "method": "POST",
                                 "path": f"/api/projects/{ctx.project_id}/studies/benchmark", "body": {}}
            est_kind = "benchmark"
        if est_kind:
            try:
                e = estimate_kind(db, est_kind, run_row, default_params(est_kind))
                row["estimate"] = {"est_s": e["est_s"], "est_lo": e["est_lo"], "est_hi": e["est_hi"]}
            except Exception:
                log.exception("estimate of %s failed", est_kind)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# study views
# ---------------------------------------------------------------------------

def _job_active(db, row: dict) -> bool:
    if not row.get("job_id"):
        return False
    j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
    return j is not None and j["status"] in service.ACTIVE_JOB


def _children_rows(db, sid: str) -> list[dict]:
    return db.fetchall("SELECT id, run_dir, status, stages_json FROM runs WHERE study_id = ? "
                       "ORDER BY created_utc, id", (sid,))


def _role(r: dict) -> str | None:
    info = dbmod.loads(r.get("stages_json"), {}) or {}
    role = info.get("role") if isinstance(info, dict) else None
    if role:
        return str(role)
    name = Path(r["run_dir"]).name
    m = re.search(r"_placebo_(grf|shift|rotate)", name)
    if m:
        return f"placebo:{m.group(1)}"
    m = re.search(r"_mv_([A-Za-z0-9_]+?)(_coarse[0-9.]+)?$", name)
    if m:
        return f"variant:{m.group(1)}"
    return "reproduction" if name.endswith("_reproduce") else None


def _run_by_dir(db, run_dir) -> str | None:
    if not run_dir:
        return None
    try:
        key = str(Path(str(run_dir)).resolve())
    except OSError:
        key = str(run_dir)
    r = db.fetchone("SELECT id FROM runs WHERE run_dir = ?", (key,))
    return r["id"] if r else None


def placebo_view(db, row: dict) -> dict:
    out_dir = Path(row["out_dir"])
    data = read_json(out_dir / "placebo.json") or {}
    params = dbmod.loads(row.get("params_json"), {}) or {}
    verdicts: dict[str, str] = {}
    for r in data.get("rows") or []:
        if not r.get("placebo") or not isinstance(r.get("verdict"), dict):
            continue
        v = r["verdict"]
        ok_m, ok_c = bool(v.get("model_pass")), bool(v.get("causal_ci_covers_zero"))
        word = "passes" if ok_m and ok_c else "model fails" if ok_c else "causal fails" if ok_m else "fails"
        prev = verdicts.get(r["kind"])
        verdicts[r["kind"]] = word if prev in (None, "passes") else prev
    children, seen = [], set()
    for r in _children_rows(db, row["id"]):
        role = _role(r) or ""
        kind = role.split(":", 1)[1] if role.startswith("placebo:") else role or "child"
        seen.add(kind)
        children.append({"kind": kind, "run_id": r["id"], "status": r["status"], "verdict": verdicts.get(kind)})
    for kind, info in (data.get("runs") or {}).items():          # an imported suite's re-fits
        if kind not in seen:
            seen.add(kind)
            children.append({"kind": kind, "run_id": _run_by_dir(db, (info or {}).get("run_dir")),
                             "status": "complete", "verdict": verdicts.get(kind)})
    active = _job_active(db, row)
    for kind in params.get("kinds") or []:
        if kind not in seen:
            children.append({"kind": kind, "run_id": None, "status": "pending" if active else
                             ("missing" if row.get("status") in ("failed", "cancelled", "interrupted") else "pending"),
                             "verdict": None})
    return {"rows": data.get("rows") or [], "layer_correlation": data.get("layer_correlation_with_original") or {},
            "n_pass_model": data.get("n_pass_model"), "n_pass_causal": data.get("n_pass_causal"),
            "n_placebos": data.get("n_placebos"), "children": children}


def _jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        text = path.read_text("utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:                  # a line being appended right now
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def simcheck_view(db, row: dict) -> dict:
    """The generator × seed grid, live from ``simcheck.jsonl`` (pending cells from the study's design)."""
    from sparc.core.simcheck import summarize

    out_dir = Path(row["out_dir"])
    params = dbmod.loads(row.get("params_json"), {}) or {}
    rows = _jsonl(out_dir / "simcheck.jsonl")
    latest: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("generator"), r.get("product", "rf"), r.get("seed"))
        if "error" not in r or key not in latest:
            latest[key] = r
    design = {k: int(v) for k, v in (params.get("design") or {}).items() if v}
    if not design:                          # an imported folder: the design is what ran
        for (g, prod, seed) in latest:
            label = g if prod == "rf" else f"{g}/{prod}"
            design[label] = max(design.get(label, 0), int(seed or 0) + 1)
    active = _job_active(db, row)
    workers = max(1, int(params.get("workers") or 1))
    grid, pending = [], 0
    for label, n in design.items():
        gen, _, prod = label.partition("/")
        prod = prod or "rf"
        for seed in range(n):
            r = latest.get((gen, prod, seed))
            if r is None:
                status = "running" if active and pending < workers else "pending"
                pending += 1
                grid.append({"generator": label, "seed": seed, "status": status, "share": None, "ci_covers": None,
                             "causal_covers": None, "seconds": None, "gate_attempt": None})
                continue
            gate = r.get("gate") or {}
            status = "error" if "error" in r else ("gate_fail" if gate and not gate.get("pass") else "done")
            grid.append({"generator": label, "seed": seed, "status": status, "share": r.get("share"),
                         "ci_covers": r.get("ci_covers_truth"), "causal_covers": r.get("causal_covers_truth"),
                         "seconds": r.get("seconds"), "gate_attempt": gate.get("attempt")})
    secs = [float(r["seconds"]) for r in latest.values() if isinstance(r.get("seconds"), (int, float))]
    remaining = sum(1 for g in grid if g["status"] in ("pending", "running"))
    eta = (sum(secs) / len(secs)) * remaining / workers if secs and remaining and active else (0.0 if not remaining
                                                                                                else None)
    summ = summarize(list(latest.values())) if latest else {}
    return {"design": design, "grid": grid, "generators": summ.get("generators") or {},
            "bias_correction": summ.get("bias_correction") or None, "eta_s": eta}


def multiverse_view(db, row: dict) -> dict:
    """The variant board (done variants from ``<name>.json``), effects and priority agreement so far."""
    from sparc.core.multiverse import LABELS, VARIANTS, summarize

    out_dir = Path(row["out_dir"])
    params = dbmod.loads(row.get("params_json"), {}) or {}
    custom = params.get("custom_variants") or {}
    if params.get("variants") or custom or row.get("origin") == "studio":
        names = list(params.get("variants") or VARIANTS) + [n for n in custom if n not in (params.get("variants")
                                                                                            or ())]
    else:
        names = sorted((p.stem for p in out_dir.glob("*.json") if not p.stem.startswith("multiverse")
                        and p.name != service.MIRROR), key=lambda n: (n != "baseline", n))
    by_role = {(_role(r) or ""): r for r in _children_rows(db, row["id"])}
    active = _job_active(db, row)
    workers = max(1, int(params.get("workers") or 1))
    variants, pending = [], 0
    for n in names:
        doc = read_json(out_dir / f"{n}.json")
        child = by_role.get(f"variant:{n}")
        if isinstance(doc, dict):
            variants.append({"name": n, "label": doc.get("label") or LABELS.get(n, n), "status": "done",
                             "r2": doc.get("r2"), "rmse": doc.get("rmse"), "seconds": doc.get("seconds"),
                             "run_id": _run_by_dir(db, doc.get("run_dir")) or (child or {}).get("id")})
            continue
        if active:
            status = "running" if (child is not None or pending < workers) else "pending"
            pending += 1
        else:
            status = "failed" if row.get("status") in ("failed", "cancelled", "interrupted", "succeeded") else "pending"
        variants.append({"name": n, "label": LABELS.get(n, n), "status": status, "r2": None, "rmse": None,
                         "seconds": None, "run_id": (child or {}).get("id")})
    summ = {}
    if (out_dir / "multiverse_summary.json").is_file() and not active:
        summ = read_json(out_dir / "multiverse_summary.json") or {}
    elif (out_dir / "baseline.json").is_file() and (out_dir / "baseline_maps.npz").is_file():
        try:
            summ = summarize(out_dir, [n for n in names if (out_dir / f"{n}.json").is_file()])
        except Exception:
            log.exception("multiverse summary of %s failed", out_dir)
            summ = {}
    return {"variants": variants, "effects": summ.get("effects") or {}, "priority": summ.get("priority") or {},
            "stability": {k: summ.get(k) for k in ("sign_stability_min", "median_kendall_tau",
                                                   "median_top_decile_jaccard")}}


def reproduce_view(db, row: dict) -> dict:
    summ = dbmod.loads(row.get("summary_json")) or {}
    cand = [Path(r["run_dir"]) for r in _children_rows(db, row["id"])]
    if summ.get("reproduction"):
        cand.append(Path(str(summ["reproduction"])))
    cand.append(Path(row["out_dir"]))
    data = next((read_json(p / "reproduce.json") for p in cand if (p / "reproduce.json").is_file()), None) or {}
    return {"pass": data.get("pass"), "checks": data.get("checks") or [], "original": data.get("original"),
            "reproduction": data.get("reproduction")}


def benchmark_view(db, row: dict) -> dict:
    data = read_json(Path(row["out_dir"]) / "benchmark.json") or {}
    return {"runs": data.get("runs") or {}}


_VIEWS = {"placebo": placebo_view, "simcheck": simcheck_view, "multiverse": multiverse_view,
          "reproduce": reproduce_view, "benchmark": benchmark_view}


def study_view(db, row: dict) -> dict:
    fn = _VIEWS.get(row.get("kind") or "")
    if fn is None:
        raise ApiError("unknown_view", f"no view for a {row.get('kind')!r} study", detail={"kinds": list(_VIEWS)})
    return fn(db, row)


# ---------------------------------------------------------------------------
# truth vs recovered (synthetic demo)
# ---------------------------------------------------------------------------

def _truth_path(ctx, project_dir: Path | None) -> Path | None:
    cands = []
    cfg = ctx.cfg
    if cfg is not None:
        try:
            dp = Path(cfg.data_path)
            cands.append(dp.with_suffix(".truth.json"))
        except Exception:
            pass
    if project_dir is not None:
        cands += sorted((project_dir / "data").glob("*.truth.json")) + [project_dir / "truth.json"]
    return next((p for p in cands if p.is_file()), None)


def truth_rows(sctx, ctx) -> dict:
    """Truth (``truth.json``, SPEC §9.2) against what the run recovered.  ``404 not_found`` unless the run
    belongs to a demo project whose truth file exists."""
    import numpy as np

    from sparc.core.catalog import unit_label

    db = sctx.db
    p = db.fetchone("SELECT demo, dir FROM projects WHERE id = ?", (ctx.project_id,)) if ctx.project_id else None
    if p is None or not p.get("demo"):
        raise ApiError("not_found", "truth vs recovered is available for synthetic demo projects only")
    path = _truth_path(ctx, Path(p["dir"]) if p.get("dir") else None)
    truth = read_json(path) if path is not None else None
    if not isinstance(truth, dict):
        raise ApiError("not_found", "the demo's truth.json is missing")
    m = ctx.manifest or {}
    unit = unit_label(ctx.target_units)
    derived = truth.get("derived") or {}
    rows: list[dict] = []
    scen = {s.get("name"): s for s in m.get("scenarios") or [] if isinstance(s, dict)}
    for name, t in (derived.get("true_scenarios") or {}).items():
        s = scen.get(name) or {}
        rec, se = s.get("mean_delta"), s.get("mean_delta_se")
        rows.append({"quantity": "canopy_scenario", "label": f"{name}: city-mean ΔT", "truth": float(t),
                     "recovered": _num(rec), "se": _num(se), "share": _ratio(rec, t), "unit": unit, "scenario": name})
    roles = ((ctx.cfg_raw.get("physics") or {}).get("roles")) or {}
    can = roles.get("canopy") or "canopy"
    if derived.get("true_footprint_mean") is not None:
        rec = se = None
        df = ctx.parquet(f"response_{can}.parquet", ["footprint_effect_per_unit"])
        if df is not None and len(df):
            v = df["footprint_effect_per_unit"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                rec = float(v.mean())
                se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else None
        t = float(derived["true_footprint_mean"])
        rows.append({"quantity": "footprint_mean", "label": "Mean canopy footprint per +1 pp", "truth": t,
                     "recovered": rec, "se": se, "share": _ratio(rec, t), "unit": unit, "scenario": None})
    if truth.get("L") is not None:
        lm = (m.get("physics") or {}).get("L_m") or {}
        rec = lm.get("mean") if isinstance(lm, dict) else lm
        rows.append({"quantity": "L_m", "label": "Relaxation length L", "truth": float(truth["L"]),
                     "recovered": _num(rec), "se": _num(lm.get("sd")) if isinstance(lm, dict) else None,
                     "share": _ratio(rec, truth["L"]), "unit": "m", "scenario": None})
    if truth.get("influence_radius_90") is not None:
        rec = ((m.get("influence") or {}).get("ranges_m") or {}).get(can)
        rows.append({"quantity": "influence_radius_m", "label": "Canopy influence radius",
                     "truth": float(truth["influence_radius_90"]), "recovered": _num(rec), "se": None,
                     "share": _ratio(rec, truth["influence_radius_90"]), "unit": "m", "scenario": None})
    if truth.get("noise_sd") is not None:
        rec = ((m.get("metrics") or {}).get("stacker") or {}).get("rmse")
        rows.append({"quantity": "noise_sd", "label": "Noise floor vs held-out RMSE", "truth": float(truth["noise_sd"]),
                     "recovered": _num(rec), "se": None, "share": _ratio(rec, truth["noise_sd"]), "unit": unit,
                     "scenario": None})
    return {"rows": rows}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ratio(a, b):
    a, b = _num(a), _num(b)
    if a is None or b is None or b == 0:
        return None
    return a / b
