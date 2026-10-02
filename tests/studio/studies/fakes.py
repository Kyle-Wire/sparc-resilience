"""Stand-ins for the core study and post-run functions (fast, deterministic, no model fits).

:func:`install` patches the library functions in place; ``fake_worker.py`` calls it before the real job
worker, so ``post.*`` and ``study.*`` jobs run in real worker processes (events, ``result.json``, hooks)
without fitting anything.  Every fake has the real function's signature, writes the files the real one
writes (in the real shapes) and appends its arguments to ``<job dir>/fake_calls.jsonl``, which the
dispatch tests read.  Child runs are copies of the committed synthetic fixture run whose manifest carries
the ``run_meta`` the fake received, announced with a ``run.dir`` event as ``run_core`` would.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

SYNTH = Path(__file__).resolve().parents[1] / "fixtures" / "synth_run"
JOB_DIR: Path | None = None


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _cfg(cfg) -> dict:
    return {"name": cfg.name, "title": (cfg.raw.get("report") or {}).get("title"), "data_path": str(cfg.data_path),
            "output_dir": (cfg.raw.get("output") or {}).get("dir"), "base_dir": str(cfg.base_dir),
            "planner_layers": (cfg.raw.get("planner") or {}).get("layers")}


def record(fn: str, **kw) -> None:
    if JOB_DIR is None:
        return
    with open(JOB_DIR / "fake_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"fn": fn, **_jsonable(kw)}) + "\n")


def _child(run_dir: Path, run_meta: dict | None, role: str) -> Path:
    """A copy of the fixture run at ``run_dir`` (manifest ``run_meta`` + role), announced like ``run_core``."""
    from sparc.core import progress, runio

    run_dir = Path(run_dir)
    if not run_dir.exists():
        shutil.copytree(SYNTH, run_dir, ignore=shutil.ignore_patterns("events.jsonl", "FIXTURE.json"))
    m = json.loads((run_dir / "manifest.json").read_text())
    m["run_meta"] = {**(run_meta or {}), "role": role}
    m["name"] = run_dir.name
    runio.write_json_atomic(run_dir / "manifest.json", m, indent=1)
    progress.emit("run.dir", run_dir=str(run_dir.resolve()), fingerprint="c" * 16)
    return run_dir


# ---------------------------------------------------------------------------
# post-run actions
# ---------------------------------------------------------------------------

def fake_baselines_for_run(run_dir, cfg, models=("regression_kriging", "hgb_xy", "hgb", "idw", "hgb_focal")) -> dict:
    from sparc.core import runio

    record("baselines_for_run", run_dir=run_dir, cfg=_cfg(cfg), models=list(models))
    rows = {m: {"label": m, "rmse": 0.5 + 0.01 * i, "r2": 0.6 - 0.01 * i, "stack_rmse": 0.34, "delta_rmse": 0.16,
                "delta_mse": 0.13, "delta_mse_se": 0.02, "frac_blocks_stack_better": 0.8} for i, m in enumerate(models)}
    res = {"verdict": "the stack beats every baseline", "best_baseline": list(models)[0], "rows": rows}
    runio.write_json_atomic(Path(run_dir) / "baselines.json", res, indent=1)
    runio.update_manifest(run_dir, {"baselines": res}, source="baselines")
    return res


def fake_planner_pack(run_dir, cfg, out_dir=None, thresholds=None, package=None, hex_sizes=(250.0, 500.0),
                      export=True, cache_dir="output/core/cache") -> dict:
    from sparc.core import runio

    record("planner_pack", run_dir=run_dir, cfg=_cfg(cfg), thresholds=thresholds, package=package,
           hex_sizes=list(hex_sizes), export=export, cache_dir=cache_dir)
    out = Path(out_dir or Path(run_dir) / "planner")
    out.mkdir(parents=True, exist_ok=True)
    res = {"units": "degF", "thresholds": list(thresholds or [90, 95]), "people_total": 1234.0,
           "package": package or "Cooling package", "exposure": [], "hot_days": {"station": "USW00014765"}}
    for s in hex_sizes:
        runio.write_text_atomic(out / f"hex_{int(s)}m.csv", "hex,people\n1,10\n")
    runio.write_json_atomic(out / "planner.json", res, indent=1)
    runio.update_manifest(run_dir, {"planner": res}, source="planner")
    return res


def fake_emulator_for_run(run_dir, cfg, validate_doses=None, n_patches=8) -> dict:
    from sparc.core import runio

    record("emulator_for_run", run_dir=run_dir, cfg=_cfg(cfg), n_patches=n_patches)
    meta = {"levers": {v: {"validation": {"patch_pass_rate": 0.9, "uniform": {"rel_err": 0.05}}}
                       for v in (cfg.actionable or {})}}
    runio.write_json_atomic(Path(run_dir) / "emulator.json", meta, indent=1)
    return meta


def fake_uncertainty_report(run_dir, multiverse_dir=None, simcheck_dirs=(), placebo_path=None,
                            real_r2_gate=False) -> dict:
    from sparc.core import runio

    record("uncertainty_report", run_dir=run_dir, multiverse_dir=multiverse_dir, simcheck_dirs=list(simcheck_dirs),
           placebo_path=placebo_path, real_r2_gate=real_r2_gate)
    out = {"scenarios": [{"scenario": "Canopy Increase +10", "estimate": -1.03, "envelope": [-1.6, -0.4],
                          "envelope_excludes_zero": True}],
           "sources": {"run": str(run_dir), "multiverse": str(multiverse_dir) if multiverse_dir else None,
                       "simcheck": [str(d) for d in simcheck_dirs]}}
    runio.write_json_atomic(Path(run_dir) / "uncertainty.json", out, indent=1)
    runio.update_manifest(run_dir, {"uncertainty": out}, source="uncertainty")
    return out


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------

def _placebo_row(kind: str, passes: bool) -> dict:
    return {"kind": kind, "variable": "canopy", "placebo": True, "sd": 10.0,
            "model": [{"dose_sd": 1.0, "mean_delta": -0.02, "se": 0.05}], "causal_theta_sum_per_sd": 0.01,
            "causal_se_per_sd": 0.02, "footprint_per_sd": -0.01,
            "verdict": {"delta_1sd": -0.02, "se_1sd": 0.05, "ratio_to_real": 0.04, "model_within_2se": passes,
                        "model_below_10pct_of_real": passes, "model_pass": passes, "causal_ci_covers_zero": True},
            "reference": "canopy"}


def fake_run_placebo_suite(cfg, kinds=("grf", "shift", "rotate"), coarse=60.0, seed=0, grf_range_m=600.0, write=True,
                           frame=None, children_dir=None, resume=False, run_meta=None) -> dict:
    record("run_placebo_suite", cfg=_cfg(cfg), kinds=list(kinds), coarse=coarse, seed=seed, grf_range_m=grf_range_m,
           children_dir=children_dir, resume=resume, run_meta=run_meta)
    runs = {}
    for kind in kinds:
        d = Path(children_dir) / (f"{cfg.name}_placebo_{kind}" + (f"_coarse{float(coarse):g}" if coarse else ""))
        _child(d, run_meta, f"placebo:{kind}")
        runs[kind] = {"metrics": {"r2": 0.6}, "run_dir": str(d)}
    rows = [{"kind": "grf", "variable": "canopy", "placebo": False, "sd": 10.0,
             "model": [{"dose_sd": 1.0, "mean_delta": -0.5, "se": 0.1}], "causal_theta_sum_per_sd": -0.4,
             "causal_se_per_sd": 0.1, "footprint_per_sd": -0.3}] if "grf" in kinds else []
    rows += [_placebo_row(k, k != "rotate") for k in kinds if k != "grf"]
    placebos = [r for r in rows if r["placebo"]]
    return {"kinds": list(kinds), "coarse_m": coarse, "seed": seed, "layers": ["canopy", "impervious"], "rows": rows,
            "layer_correlation_with_original": {f"{k}:canopy": 0.01 for k in kinds if k != "grf"},
            "n_pass_model": sum(r["verdict"]["model_pass"] for r in placebos),
            "n_pass_causal": sum(r["verdict"]["causal_ci_covers_zero"] for r in placebos),
            "n_placebos": len(placebos), "runs": runs}


def simcheck_row(gen: str, seed: int, share: float | None = 0.9, product: str = "rf") -> dict:
    return {"generator": gen, "product": product, "seed": seed,
            "gate": {"pass": True, "attempt": 0, "sd_ratio": 1.0}, "seconds": 12.5,
            "true_mean_delta": 0.0 if gen == "null" else -0.25, "model_mean_delta": -0.22 if gen != "null" else -0.05,
            "model_se": 0.05, "share": None if gen == "null" else share, "ci_covers_truth": True,
            "significant": gen != "null", "causal_theta_pp": -0.02, "causal_se_pp": 0.01, "true_pp": -0.025,
            "causal_covers_truth": True, "causal_significant": gen != "null", "interval_coverage": 0.9,
            "oof_r2": 0.6, "rank_corr": 0.7}


def fake_run_simcheck(cfg, design, out_dir, coarse=90.0, epochs=200, workers=1, threads=1, product="rf") -> dict:
    from sparc.core import runio
    from sparc.core.simcheck import summarize

    record("run_simcheck", cfg=_cfg(cfg), design=dict(design), out_dir=out_dir, coarse=coarse, epochs=epochs,
           workers=workers, threads=threads)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "simcheck.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["generator"], r["seed"]))
    with open(path, "a", encoding="utf-8") as f:
        for gen, n in design.items():
            for s in range(int(n)):
                if (gen, s) not in done:
                    f.write(json.dumps(simcheck_row(gen, s, 0.8 + 0.05 * s)) + "\n")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summ = summarize(rows)
    runio.write_json_atomic(out_dir / "simcheck_summary.json", summ, indent=1)
    return summ


def write_variant(out_dir: Path, name: str, run_dir: Path, scale: float = 1.0) -> None:
    from sparc.core import runio

    ids = np.arange(50)
    runio.write_npz_atomic(out_dir / f"{name}_maps.npz", compressed=True, ids=ids,
                           canopy=np.linspace(0.1, 1.0, 50) * scale)
    doc = {"variant": name, "label": name.replace("_", " "), "seconds": 3.0, "r2": 0.6, "rmse": 0.4, "block_m": 390.0,
           "stacker_choice": "nnls", "scenarios": {"Canopy Increase +10": {"mean_delta": -1.0 * scale, "se": 0.2}},
           "run_dir": str(run_dir)}
    runio.write_json_atomic(out_dir / f"{name}.json", doc, indent=1)


def fake_run_multiverse(cfg, out_dir, variants=None, coarse=60.0, workers=1, threads=1, extra_variants=None,
                        run_meta=None) -> dict:
    from sparc.core import runio
    from sparc.core.multiverse import summarize

    record("run_multiverse", cfg=_cfg(cfg), out_dir=out_dir, variants=variants, coarse=coarse, workers=workers,
           threads=threads, extra_variants=extra_variants, run_meta=run_meta)
    out_dir = Path(out_dir)
    names = list(variants or ["baseline", "blocks_1km"]) + [n for n in (extra_variants or {}) if n not in (variants or ())]
    for i, n in enumerate(names):
        if (out_dir / f"{n}.json").exists():
            continue
        child = Path(cfg.raw["output"]["dir"]) / (f"{cfg.name}_mv_{n}" + (f"_coarse{float(coarse):g}" if coarse else ""))
        _child(child, run_meta, f"variant:{n}")
        write_variant(out_dir, n, child, 1.0 + 0.1 * i)
    summ = summarize(out_dir, names)
    runio.write_json_atomic(out_dir / "multiverse_summary.json", summ, indent=1)
    return summ


def fake_reproduce(run_dir, stages=("S0", "S1", "S2", "S3"), tol_r2=0.01, tol_effect=0.05, config_dir=None,
                   out_dir=None, run_meta=None) -> dict:
    from sparc.core import runio

    record("reproduce", run_dir=run_dir, stages=list(stages), tol_r2=tol_r2, tol_effect=tol_effect,
           config_dir=config_dir, out_dir=out_dir, run_meta=run_meta)
    child = _child(Path(out_dir), run_meta, "reproduction")
    out = {"pass": True, "checks": [{"check": "cv design", "ok": True, "detail": "same", "hard": True},
                                    {"check": "R² stacker", "ok": True, "detail": "0.61 → 0.61", "hard": True},
                                    {"check": "core code", "ok": False, "detail": "code changed", "hard": False}],
           "original": str(run_dir), "reproduction": str(child)}
    runio.write_json_atomic(child / "reproduce.json", out, indent=1)
    return out


def fake_run_benchmark(seed=0, spatial_plus_ab=True, epochs=150, n=96) -> dict:
    record("run_benchmark", seed=seed, spatial_plus_ab=spatial_plus_ab, epochs=epochs, n=n)

    def run(share):
        return {"models": {"mgwr": {"share": share - 0.1, "corr": 0.7}}, "stack": {"share": share, "corr": 0.8},
                "oof": {"stacker": {"rmse": 0.3, "r2": 0.7}, "mgwr": {"rmse": 0.35, "r2": 0.65}},
                "footprint": {"share": share + 0.05, "corr": 0.75}}

    runs = {"spatial_plus": run(0.92)}
    if spatial_plus_ab:
        runs["standard"] = run(0.71)
    return {"seed": seed, "n": n, "runs": runs}


def install() -> None:
    import sparc.core.baselines as baselines
    import sparc.core.diagnostics as diagnostics
    import sparc.core.emulator as emulator
    import sparc.core.multiverse as multiverse
    import sparc.core.placebo as placebo
    import sparc.core.planner as planner
    import sparc.core.reproduce as reproduce
    import sparc.core.simcheck as simcheck
    import sparc.core.uncertainty as uncertainty

    baselines.baselines_for_run = fake_baselines_for_run
    planner.planner_pack = fake_planner_pack
    emulator.emulator_for_run = fake_emulator_for_run
    uncertainty.uncertainty_report = fake_uncertainty_report
    placebo.run_placebo_suite = fake_run_placebo_suite
    simcheck.run_simcheck = fake_run_simcheck
    multiverse.run_multiverse = fake_run_multiverse
    reproduce.reproduce = fake_reproduce
    diagnostics.run_benchmark = fake_run_benchmark
