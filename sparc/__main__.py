#!/usr/bin/env python3
"""
SPARC CLI — Spatial Analysis and Research Core
===============================================

Usage
-----
    sparc init     --template uhi --output ./my_project
    sparc validate --project project.yml
    sparc run      --project project.yml [--stage 1|2|3|all] [--fast]
    sparc scenario --project project.yml [--scenario canopy_increase]
    sparc report   --project project.yml
    sparc server   [--port 8008] [--dev]
    sparc desktop  [--port 8008]

If the package is **not** pip-installed you can also run::

    python -m sparc <command> ...

    python -m sparc run -p project.yml -s 0

from the repository root (the directory containing ``pyproject.toml``).
"""

import argparse
import os
import shutil
import sys
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _resolve_project_path(args) -> str:
    """Return the absolute path to the project YAML file."""
    p = Path(args.project).resolve()
    if not p.exists():
        print(f"ERROR: Project file not found: {p}", file=sys.stderr)
        sys.exit(1)
    return str(p)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Scaffold a new project from a template."""
    template_name = args.template
    output_dir = Path(args.output).resolve()
    source_dir = TEMPLATES_DIR / template_name

    if not source_dir.exists():
        available = [d.name for d in TEMPLATES_DIR.iterdir() if d.is_dir()]
        print(f"ERROR: Template '{template_name}' not found.", file=sys.stderr)
        print(f"Available templates: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"WARNING: Output directory '{output_dir}' is not empty.")
        resp = input("Overwrite? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            sys.exit(0)

    print(f"Creating project from template '{template_name}' -> {output_dir}")
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
    print(f"Done. Edit {output_dir / 'project.yml'} and then run:")
    print(f"  sparc validate --project {output_dir / 'project.yml'}")


def cmd_validate(args):
    """Validate a project.yml and its referenced files."""
    project_path = _resolve_project_path(args)

    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sparc.config.config import load_config

    try:
        config = load_config(project_path)
    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # Check data file exists
    data_path = config['data']['file_path']
    if not os.path.exists(data_path):
        print(f"WARNING: Data file not found: {data_path}")
    else:
        import pandas as pd
        df = pd.read_csv(data_path, nrows=5)
        expected_cols = (
            [config['variables']['target']]
            + list(config['variables']['coordinates'])
            + list(config['predictors']['base_model'])
        )
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"WARNING: Missing columns in data: {missing}")
            print(f"  Available: {list(df.columns)}")
        else:
            print(f"OK: Data file loaded ({data_path})")

    # Check physics files
    for key in ('priors_file', 'caps_file'):
        fpath = config.get('physics', {}).get(key)
        if fpath:
            if os.path.exists(fpath):
                print(f"OK: {key} -> {fpath}")
            else:
                print(f"WARNING: {key} not found: {fpath}")

    # Check causal DAG file
    dag_file = config.get('causal', {}).get('dag_file')
    if dag_file:
        if os.path.exists(dag_file):
            print(f"OK: dag_file -> {dag_file}")
        else:
            print(f"WARNING: dag_file not found: {dag_file}")

    # Check CRS codes
    try:
        from pyproj import CRS
        CRS.from_user_input(config['crs']['initial'])
        CRS.from_user_input(config['crs']['target_projected'])
        print(f"OK: CRS codes valid ({config['crs']['initial']}, {config['crs']['target_projected']})")
    except Exception as e:
        print(f"WARNING: CRS validation failed: {e}")

    print("\nValidation complete.")


def cmd_run(args):
    """Run the SPARC pipeline (or a specific stage).

    Stage flow (when ``--stage all``):
      0  GWEN variable selection  (skippable with ``--skip-gwen``)
      1  Correlogram analysis     (auto-wires bandwidths into project config)
      1b Pipeline config generation
      2  Enhanced Spatial CV       (base models + neural meta-learner)
      3  Causal Validation         (DAG + DoWhy + structural coefficients)
      4  Scenario simulation       (DAG + physics blending)
    """
    project_path = _resolve_project_path(args)

    # Fix GDAL/fiona HOME on Windows before any GeoPackage I/O
    if sys.platform == 'win32':
        import tempfile
        for var in ('CPL_TMPDIR', 'GDAL_PAM_PROXY_DIR'):
            if var not in os.environ:
                os.environ[var] = tempfile.gettempdir()
        home = os.environ.get('HOME', '')
        if not home or 'systemprofile' in home.lower() or not os.path.isdir(home):
            os.environ['HOME'] = os.path.expanduser('~')

    # Add parent to path for imports
    repo_root = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, str(Path(repo_root) / 'run'))

    from sparc.config.config import load_config
    from sparc.run.pipeline_paths import set_paths_from_config

    config = load_config(project_path)
    paths = set_paths_from_config(config)

    # Propagate the project file so that downstream load_config() calls
    # (which pass no arguments) automatically use the same project.yml.
    os.environ['SPARC_PROJECT'] = project_path

    # Stash --legacy flag on config for _run_scenarios dispatch.
    if getattr(args, 'legacy', False):
        config['_force_legacy_scenarios'] = True

    stage = args.stage or 'all'
    fast = args.fast or config.get('pipeline', {}).get('fast_mode', False)
    skip_gwen = getattr(args, 'skip_gwen', False)
    resume = getattr(args, 'resume', False)

    print(f"\n{'='*60}")
    print(f"  SPARC Pipeline — {config.get('project', {}).get('name', 'Unnamed Project')}")
    print(f"  Stage(s): {stage}  |  Fast mode: {fast}")
    if skip_gwen:
        print(f"  GWEN variable selection: SKIPPED")
    print(f"{'='*60}\n")

    # ── Hardware profile + memory watchdog ──────────────────────
    try:
        from sparc.config.hardware_profile import detect_profile
        from sparc.run.memory_watchdog import MemoryWatchdog, MemoryPressureAbort
    except Exception:
        detect_profile = None  # type: ignore[assignment]
        MemoryWatchdog = None  # type: ignore[assignment]
        MemoryPressureAbort = RuntimeError  # type: ignore[assignment,misc]
    else:
        try:
            print(f"  {detect_profile().banner()}")
        except Exception:
            pass

    # ── Run-wide artifact registry ──────────────────────────────
    # A single registry instance is carried through the run; after every
    # stage we walk the output directories and register anything new
    # that legacy writers dropped straight to disk (the registry's
    # built-in bookkeeping only covers paths written via ResultStore).
    try:
        from sparc.registry import RunRegistry
        from sparc.registry.run_registry import set_active_registry
        _registry: "RunRegistry | None" = RunRegistry(paths.output_dir, autoload=True)
        _registry.manifest.project_name = config.get("project", {}).get("name")
        # Make this run's registry visible to writers across the codebase
        # so they can call ``register_path(...)`` without plumbing a param.
        set_active_registry(_registry)
    except Exception as _reg_err:
        print(f"  (registry unavailable: {_reg_err})")
        _registry = None

    def _rescan_registry(stage_label: str) -> None:
        if _registry is None:
            return
        try:
            _registry.start_stage(stage_label)
            n = _registry.migrate_from_disk(paths)
            _registry.complete_stage(stage_label, status="complete")
            if n:
                print(f"  [registry] stage {stage_label}: +{n} artifact(s)")
        except Exception as exc:
            import traceback
            print(f"  [registry] rescan failed for stage {stage_label}: {exc}")
            traceback.print_exc()

    # Memory watchdog: emits warnings on rising RAM, releases caches, and
    # signals a clean abort before the OS OOM-kills the process.
    _watchdog = None
    if MemoryWatchdog is not None:
        _watchdog = MemoryWatchdog(stage_name=f"run/{stage}")
        _watchdog.start()

    def _memory_checkpoint() -> None:
        if _watchdog is not None:
            _watchdog.checkpoint()


    # ── Helper: stage-complete checks for --resume ───────────────
    def _stage_done(stage_key: str) -> bool:
        """Check whether a stage is marked complete.

        Prefers an ArtifactStore status struct (db-only runs); falls
        back to legacy on-disk sentinel files for older runs.
        """
        if not resume:
            return False
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None:
                status = None
                try:
                    status = _store.read_struct(stage_key, "status")
                except Exception:
                    status = None
                if isinstance(status, dict) and status.get("complete"):
                    return True
        except Exception:
            pass
        # Legacy fallback: file sentinel.
        legacy_marker = {
            "0": paths.stage1_dir / ".correlogram_complete",
        }.get(stage_key)
        return bool(legacy_marker and legacy_marker.exists())

    def _mark_stage_done(stage_key: str) -> None:
        """Persist stage-complete status to ArtifactStore (and disk if on)."""
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None:
                _store.write_struct(
                    stage_key, "status",
                    {"complete": True},
                    producer="sparc.__main__",
                )
        except Exception:
            pass
        try:
            from sparc.run.disk_policy import disk_writes_enabled
            if disk_writes_enabled() and stage_key == "0":
                paths.stage1_dir.mkdir(parents=True, exist_ok=True)
                (paths.stage1_dir / ".correlogram_complete").write_text("done")
        except Exception:
            pass

    # ── Pipeline progress tracker ──────────────────────────────────
    from sparc.run.progress import StageProgress as _StageProgress
    _sp_stages: list[str] = []
    if stage in ('0', 'all'):
        _sp_stages += ['0', '0b']
    if stage in ('1', 'all') and not skip_gwen:
        _sp_stages.append('1')
    if stage in ('2', 'all'):
        _sp_stages.append('2')
    if stage in ('3', 'all'):
        _sp_stages.append('3')
    if (stage in ('4', 'all') and config.get('scenarios', [])
            and config.get('auto_run_scenarios_at_stage_4', True)):
        _sp_stages.append('4')
    if stage in ('5', 'all'):
        _sp_stages.append('5')

    def _stage_timing_sink(key: str, elapsed_s: float) -> None:
        try:
            from sparc.registry.store import get_active_store
            _timing_store = get_active_store()
            if _timing_store is not None:
                import time as _t
                _timing_store.write_struct(
                    key, "stage_timing",
                    {"stage": key, "elapsed_s": elapsed_s, "timestamp": _t.time()},
                    producer="sparc.__main__",
                )
        except Exception:
            pass

    _sp = _StageProgress(len(_sp_stages), timing_sink=_stage_timing_sink)

    # ────────────────────────────────────────────────────────────────
    # Stage 0: Correlogram Analysis  (runs first so GWEN can auto-tune)
    # ────────────────────────────────────────────────────────────────
    _memory_checkpoint()
    if stage in ('0', 'all'):
        if not _stage_done("0"):
            _sp.stage_start("0")
            from sparc.run.correlogram_analysis import main as run_correlogram
            run_correlogram(fast_mode=fast)
            _mark_stage_done("0")
            _sp.stage_done("0")
        else:
            print(">>> Stage 0: Correlogram — skipped (already complete)")
        _rescan_registry("0")

    # ────────────────────────────────────────────────────────────────
    # Stage 0b: Pipeline Configuration (auto-wire correlogram → config)
    # ────────────────────────────────────────────────────────────────
    if stage in ('0', 'all'):
        _sp.stage_start("0b")
        from sparc.run.pipeline_configurator import PipelineConfigurator
        configurator = PipelineConfigurator(stage1_dir=str(paths.stage0_dir))

        # Prefer artifacts.db; fall back to legacy on-disk JSON.
        _profile = None
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None and _store.has("0", "dataset_profile"):
                _profile = _store.read_any("0", "dataset_profile")
        except Exception as _exc:  # noqa: BLE001
            print(f"  Warning: could not read dataset_profile from artifacts.db: {_exc}")

        if _profile is None:
            profile_path = paths.stage0_dir / 'dataset_profile.json'
            if profile_path.exists():
                with open(profile_path) as _f:
                    _profile = json.load(_f)

        if _profile is not None:
            print(f"  Dataset tier: {_profile.get('size_tier', 'unknown').upper()}")
            try:
                import torch as _torch_eta
                _has_gpu = _torch_eta.cuda.is_available() or _torch_eta.backends.mps.is_available()
            except Exception:
                _has_gpu = False
            _sp.print_eta_hint(_profile.get('size_tier', 'unknown'), _has_gpu)

            try:
                from sparc.run.dataset_profiler import DatasetProfiler
                import pandas as pd
                _dummy = pd.DataFrame({'projected_X': [0], 'projected_Y': [0]})
                _profiler = DatasetProfiler(
                    _dummy,
                    coord_cols=config['variables']['coordinates'],
                    feature_cols=config.get('predictors', {}).get('base_model', []),
                )
                _profiler._profile = _profile
                configurator.apply_profiler_recommendations(_profiler.recommend_parameters())
            except Exception as e:
                print(f"  Warning: could not apply profiler recommendations: {e}")

        configurator.save_pipeline_config()
        _sp.stage_done("0b")

    # ────────────────────────────────────────────────────────────────
    # Stage 1: GWEN variable selection (optional, now uses correlogram)
    # ────────────────────────────────────────────────────────────────
    if stage in ('1', 'all') and not skip_gwen:
        _memory_checkpoint()
        use_gwen = config.get('flags', {}).get('use_gwen_selection', True)
        approval_file = Path(config.get('output', {}).get('base_dir', 'output')) / 'gwen_approved.txt'

        if use_gwen and not _stage_done('.gwen_complete'):
            _sp.stage_start("1")
            from sparc.run.gwen_variable_selection import main as run_gwen
            approved = run_gwen(config_path=project_path, fast_mode=fast)

            if not approved:
                _sp.finish()
                print(f"\n{'='*60}")
                print("  PIPELINE PAUSED — Review GWEN feature selection")
                print(f"  Approve by creating: {approval_file}")
                print("  Then re-run with:  sparc run --resume ...")
                print(f"{'='*60}")
                return

            (paths.stage1_dir).mkdir(parents=True, exist_ok=True)
            (paths.stage1_dir / '.gwen_complete').write_text('done')
            _sp.stage_done("1")
        else:
            print("\n>>> Stage 1: GWEN — skipped (already complete or disabled)")
        _rescan_registry("1")

    # ────────────────────────────────────────────────────────────────
    # Stage 2: Enhanced Spatial CV
    # ────────────────────────────────────────────────────────────────
    if stage in ('2', 'all'):
        _memory_checkpoint()
        _sp.stage_start("2")
        from sparc.run.enhanced_spatial_cv import main as run_spatial_cv
        run_spatial_cv(fast_mode=fast)
        _sp.stage_done("2")
        _rescan_registry("2")

    # ────────────────────────────────────────────────────────────────
    # Stage 3: Causal Validation
    # ────────────────────────────────────────────────────────────────
    if stage in ('3', 'all'):
        _memory_checkpoint()
        _sp.stage_start("3")
        try:
            from sparc.run.causal_validation import main as run_causal_validation
            run_causal_validation()
        except ImportError as e:
            print(f"  Stage 3 module not available ({e}) — skipping.")
        except Exception as e:
            print(f"  Stage 3 warning: {e}")
            print("  Continuing — Stage 4 will use physics priors only.")
        _sp.stage_done("3")
        _rescan_registry("3")

        # Wager (2025) audit add-ons.
        # As of v3 these run automatically inside Stage 3
        # (sparc.run.wager2025_addons). The legacy CLI flags are kept as
        # overrides — they re-run individual gaps if a user wants the
        # opt-in path or a specific subset persisted to the store.
        if any(getattr(args, attr, False) for attr in
               ("with_spillover", "with_iv", "with_policy_learning",
                "full_triangulation", "full_audit")):
            print("  (legacy CLI add-on flags — already auto-run in Stage 3)")
            _run_wager2025_audit_addons(args, config)
            _rescan_registry("3")

    # ────────────────────────────────────────────────────────────────
    # Stage 4: Scenario Simulation (DAG + Physics)
    # ────────────────────────────────────────────────────────────────
    if stage in ('4', 'all'):
        _memory_checkpoint()
        scenarios = config.get('scenarios', [])
        auto_run = config.get('auto_run_scenarios_at_stage_4', True)
        if scenarios and auto_run:
            _sp.stage_start("4")
            _run_scenarios(config, paths, project_path)
            _sp.stage_done("4")
        elif scenarios and not auto_run:
            print("\n>>> Stage 4: Scenarios defined but `auto_run_scenarios_at_stage_4` is false — "
                  "use the Scenario Runner page to launch.")
        else:
            print("\n>>> Stage 4: No scenarios defined in project.yml — skipping.")
        _rescan_registry("4")

    # Final bookkeeping: build master GPKG merging every spatial output.
    try:
        from sparc.run.master_gpkg import build_master_gpkg
        gpkg_path = build_master_gpkg(paths, registry=_registry, config=config)
        if gpkg_path:
            print(f"  [master gpkg] {gpkg_path}")
    except Exception as exc:
        import traceback
        print(f"  [master gpkg] skipped ({exc})")
        traceback.print_exc()

    # ────────────────────────────────────────────────────────────────
    # Stage 5: Final interpretation report (auto-invoked at end of run)
    # ────────────────────────────────────────────────────────────────
    if stage in ('5', 'all'):
        _sp.stage_start("5")
        try:
            from argparse import Namespace
            cmd_report(Namespace(
                project=project_path,
                audience='all',
                format='html',
                section=None,
            ))
            _sp.stage_done("5")
        except Exception as exc:
            print(f"\n>>> Stage 5 report generation failed ({exc})")

    _sp.finish()
    print(f"  Results in: {paths.output_dir}")


def _resolve_auto_scenario_mode(*, has_dag: bool) -> str:
    """Pick a concrete ``scenario_mode`` for ``auto`` based on artifacts.

    Policy (Phase 5f of the v4 rewrite + Wager 2025 audit):

    1. Per-edge NUTS + α + base ensemble + DAG → ``mode_5_full_audit``
       (DEFAULT — strict superset of mode_4_hybrid; same composition plus
       overlap, spillover, and triangulation diagnostic columns).
    2. Per-edge NUTS + α + DAG                 → ``mode_2_dag_local``.
    3. Base ensemble + α                       → ``mode_3_full_ensemble``.
    4. Otherwise                               → ``mode_1_physics``.
    """
    try:
        from sparc.registry.store import get_active_store
        store = get_active_store()
    except Exception:
        store = None

    has_alpha = bool(store and store.has("2", "v2_alpha_field"))
    has_nuts = bool(store and store.has("3", "nuts_edge_samples"))
    has_ensemble = bool(
        store
        and (
            store.has("2", "v2_neural_ensemble")
            or store.has("2", "ensemble_predictions")
            or store.has("2", "v2_ensemble_predictions")
        )
    )

    if has_dag and has_nuts and has_alpha and has_ensemble:
        return "mode_5_full_audit"
    if has_dag and has_nuts and has_alpha:
        return "mode_2_dag_local"
    if has_alpha and has_ensemble:
        return "mode_3_full_ensemble"
    return "mode_1_physics"


def _try_run_with_v4_engine(config, sim, data, scenario_mode, has_dag,
                             *, from_auto_fallback: bool = False):
    """Attempt to execute ``scenario_mode`` via the unified v4 engine.

    Returns ``(summary_df, results_gdf)`` on success, or ``None`` when
    the engine cannot be constructed (missing artifacts, missing DAG,
    or no ensemble predictor) — the caller then falls back to the
    legacy ``ScenarioSimulator`` path.
    """
    try:
        from sparc.interventions.scenario_engine_v4 import (
            ScenarioEngineV4, MissingArtifactsError,
        )
    except Exception:
        return None

    # Optional DAG load.
    dag = None
    if scenario_mode in ("mode_2_dag_local", "mode_4_hybrid") and has_dag:
        try:
            from sparc.causal.dag_definition import load_dag, dag_to_networkx
            dag = dag_to_networkx(load_dag(config["causal"]["dag_file"]))
        except Exception as exc:
            print(f"  [v4 engine] DAG load failed ({exc}); using legacy path")
            return None

    # Optional ensemble predictor (for mode_3 / mode_4) — adapter that
    # delegates to ``ScenarioSimulator._predict_consensus_delta``-style
    # base-model averaging.  When unavailable, the v4 engine is skipped.
    ensemble_pred = None
    if scenario_mode in ("mode_3_full_ensemble", "mode_4_hybrid"):
        ensemble_pred = _build_v4_ensemble_predictor(sim)
        if ensemble_pred is None:
            return None

    try:
        engine = ScenarioEngineV4(
            config,
            mode=scenario_mode,
            dag=dag,
            ensemble_predictor=ensemble_pred,
            _from_auto_fallback=from_auto_fallback,
        )
    except MissingArtifactsError as exc:
        print(f"  [v4 engine] {exc} — using legacy path")
        return None
    except Exception as exc:
        print(f"  [v4 engine] init failed ({exc}); using legacy path")
        return None

    return engine.run(data, verbose=True)


def _build_v4_ensemble_predictor(sim):
    """Wrap loaded base models into a ``df → ndarray`` callable.

    Returns ``None`` when the simulator has no usable base ensemble.
    """
    if not getattr(sim, "_models", None) or not getattr(sim, "_meta_model", None):
        return None
    try:
        # Re-use the existing baseline predictor to keep behaviour consistent.
        def _predict(df):
            base_pred, *_ = sim._predict_baseline(df, verbose=False)
            import numpy as _np
            return _np.asarray(base_pred, dtype=_np.float64).reshape(-1)
        return _predict
    except Exception:
        return None


def _run_scenarios(config, paths, project_path):
    """Execute scenario simulation.

    Modes:
      1. **DAG + MGWR coefficient blend** (primary) — causally-identified
         coefficients with per-point spatial heterogeneity and mediated
         indirect effects via the DAG.  Falls back to physics-only when
         no DAG is available.
      2. **Monte-Carlo uncertainty propagation** (optional) — toggled via
         ``run_mc_uncertainty: true`` in the pipeline section of project.yml.
         Number of draws set by ``n_mc_draws`` (default 50).

    The DAG-based result is saved as the primary output.
    """
    from sparc.interventions.scenario_simulator import ScenarioSimulator
    import pandas as pd

    sim = ScenarioSimulator(config)
    sim.load_models()

    # Load baseline data
    csv_path = config['paths']['raw_csv_path']
    data = pd.read_csv(csv_path)

    dag_file = config.get('causal', {}).get('dag_file')
    has_dag = dag_file and Path(dag_file).exists()
    requested_mode = config.get('pipeline', {}).get('scenario_mode', 'auto')

    # --- Translate legacy mode aliases (one-shot deprecation warning) ----
    _LEGACY_MODE_ALIASES = {
        'physics':            'mode_1_physics',
        'dag_coefficient':    'mode_2_dag_local',
        'model_reprediction': 'mode_3_full_ensemble',
        'hybrid':             'mode_4_hybrid',
    }
    scenario_mode = requested_mode
    if requested_mode == 'bayesian':
        raise RuntimeError(
            "scenario_mode='bayesian' was removed in SPARC v4. "
            "Use 'mode_3_full_ensemble' or 'auto' — credible intervals are "
            "now native to all modes via the per-edge NUTS posterior + "
            "Bayesian Spatial CATE."
        )
    if requested_mode in _LEGACY_MODE_ALIASES:
        new_key = _LEGACY_MODE_ALIASES[requested_mode]
        import warnings as _warnings
        _warnings.warn(
            f"scenario_mode='{requested_mode}' is deprecated; "
            f"use '{new_key}' instead. (Auto-translated for this run.)",
            DeprecationWarning, stacklevel=2,
        )
        scenario_mode = new_key

    # --- Resolve 'auto' to a concrete mode using artifact introspection --
    auto_resolved = False
    if scenario_mode == 'auto':
        scenario_mode = _resolve_auto_scenario_mode(has_dag=has_dag)
        auto_resolved = True
        print(f"  [auto] scenario_mode resolved to '{scenario_mode}'")

    # --full-audit forces mode_5 even when 'auto' would have picked something
    # weaker (e.g., when ensemble artifacts are missing).
    if getattr(args, 'full_audit', False) and scenario_mode != 'mode_5_full_audit':
        print(f"  [--full-audit] overriding '{scenario_mode}' → 'mode_5_full_audit'")
        scenario_mode = 'mode_5_full_audit'

    # --- Dispatch by scenario_mode ---------------------------------------
    # Prefer the v4 unified engine when its required artifacts are present;
    # fall back to the legacy ScenarioSimulator paths when they are not.
    # The top-level ``--legacy`` flag (stashed on config by argparse) forces
    # the legacy path even when v4 artifacts are available.
    force_legacy = bool(config.get('_force_legacy_scenarios'))
    if force_legacy:
        print("  [--legacy] forcing legacy V1/MGWR scenario path")
        used_v4 = None
    else:
        used_v4 = _try_run_with_v4_engine(
            config, sim, data, scenario_mode, has_dag,
            from_auto_fallback=auto_resolved,
        )
    if used_v4 is not None:
        summary_df, results_gdf = used_v4
        print(f"  [v4 engine] mode={scenario_mode}: {len(summary_df)} summary rows")
    elif scenario_mode == 'mode_4_hybrid':
        print("  [legacy] mode_4_hybrid: ensemble direct + DAG/NUTS indirect")
        summary_df, results_gdf = sim.run_with_hybrid_reprediction(data, verbose=True)
    elif scenario_mode == 'mode_3_full_ensemble':
        print("  [legacy] mode_3_full_ensemble: resolver + base-model reprediction blend")
        summary_df, results_gdf = sim.run_with_model_reprediction(data, verbose=True)
    elif scenario_mode == 'mode_2_dag_local':
        if has_dag:
            print("  [legacy] mode_2_dag_local: resolver + DAG + per-edge NUTS")
            summary_df, results_gdf = sim.run_with_causal_dag(data, verbose=True)
        else:
            print("  [legacy] mode_2_dag_local requested but no DAG — falling back to mode_1_physics")
            scenario_mode = 'mode_1_physics'
            summary_df, results_gdf = sim.run(verbose=True)
    elif scenario_mode == 'mode_1_physics':
        print("  [legacy] mode_1_physics: resolver with literature priors")
        summary_df, results_gdf = sim.run(verbose=True)
    else:
        raise ValueError(
            f"Unknown scenario_mode '{scenario_mode}'. "
            f"Valid: auto, mode_1_physics, mode_2_dag_local, "
            f"mode_3_full_ensemble, mode_4_hybrid."
        )

    print(f"  Scenario summary: {len(summary_df)} rows  (mode={scenario_mode})")

    # --- Conservation checks on scenario results ---------------------
    try:
        from sparc.interventions.physics_priors import ConservationChecker
        import numpy as np
        checker = ConservationChecker()
        for scenario in config.get('scenarios', []):
            var = scenario['variable']
            for inc in scenario.get('increments', []):
                direction = scenario.get('direction', 'increase')
                delta_signed = -inc if direction == 'decrease' else inc
                col_label = f"total_{var}_{'minus' if delta_signed < 0 else 'plus'}_{str(inc).replace('.', 'p')}"
                if hasattr(results_gdf, 'columns') and col_label in results_gdf.columns:
                    deltas = {var: np.full(len(data), delta_signed)}
                    target_deltas = results_gdf[col_label].values
                    checker.check(data, deltas, target_deltas=target_deltas, verbose=True)
    except Exception as e:
        print(f"  [CONSERVATION] Check skipped ({e})")

    # --- Mode 2: Monte-Carlo uncertainty propagation (optional) ------
    run_mc = config.get('pipeline', {}).get('run_mc_uncertainty', False)
    n_mc = config.get('pipeline', {}).get('n_mc_draws', 50)
    if run_mc:
        print(f"\n  [Mode 2] MC uncertainty — Base-Model Consensus (n={n_mc})")
        try:
            mc_summary, mc_meta = sim.run_with_consensus_uncertainty(
                data, n_mc=n_mc, verbose=True,
            )
            print(f"  MC meta: {mc_meta}")
        except Exception as e:
            print(f"  MC uncertainty propagation failed ({e})")
    else:
        print(f"\n  [Mode 2] MC uncertainty skipped (set run_mc_uncertainty: true, n_mc_draws: {n_mc} to enable)")

    # --- Global Sensitivity Analysis (optional) ----------------------
    run_sa = config.get('pipeline', {}).get('run_sensitivity_analysis', False)
    sa_method = config.get('pipeline', {}).get('sensitivity_method', 'morris')
    if run_sa:
        try:
            from sparc.evaluation.sensitivity import SensitivityAnalyzer
            sa = SensitivityAnalyzer(config, sim)
            sa_result = sa.run(data, method=sa_method, verbose=True)
            sa_out = paths.output_dir / f"sensitivity_{sa_method}.csv"
            sa_result['summary_df'].to_csv(sa_out, index=False)
            print(f"  Sensitivity analysis saved: {sa_out}")
        except Exception as e:
            print(f"  Sensitivity analysis failed ({e})")
    else:
        print(f"\n  [SA] Sensitivity analysis skipped (set run_sensitivity_analysis: true to enable)")

    # ------------------------------------------------------------------
    # Stage-4 Wager-2025 add-ons:
    #   * mirror Stage-3 audit artifacts under Stage 4 for Stage-5 report
    #   * Optimal-Targeted (Wager-EWM) policy replay
    #   * Budget Optimizer (allocation + Pareto frontier)
    # All best-effort; any single failure is logged and skipped.
    # ------------------------------------------------------------------
    try:
        from sparc.run.scenarios import (
            mirror_audit_artifacts_to_stage4,
            run_policy_replay,
            run_budget_optimization,
        )
        stage3_dir = str(paths.stage_dir(3)) if hasattr(paths, "stage_dir") else None
        if stage3_dir is None:
            stage_dirs = (config.get("output") or {}).get("stage_dirs") or {}
            base = (config.get("output") or {}).get("base_dir", "output")
            s3 = stage_dirs.get("stage_3", "Stage_3_Causal_Validation")
            stage3_dir = os.path.join(str(paths.output_dir), s3)
        mirror_audit_artifacts_to_stage4(stage3_dir)
        run_policy_replay(
            config=config, data=data,
            summary_df=summary_df,
            results_long_df=results_gdf,
        )
        run_budget_optimization(
            config=config, results_long_df=results_gdf,
        )
    except Exception as e:
        print(f"  [stage4 add-ons] failed ({e}); continuing")

    return summary_df, results_gdf


def cmd_scenario(args):
    """Run counterfactual scenario simulation."""
    project_path = _resolve_project_path(args)

    repo_root = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, str(Path(repo_root) / 'run'))

    from sparc.config.config import load_config
    from sparc.run.pipeline_paths import set_paths_from_config

    config = load_config(project_path)
    paths = set_paths_from_config(config)
    os.environ['SPARC_PROJECT'] = project_path

    # Stash --legacy flag on config for _run_scenarios dispatch.
    if getattr(args, 'legacy', False):
        config['_force_legacy_scenarios'] = True

    scenarios = config.get('scenarios', [])

    if not scenarios:
        print("No scenarios defined in project.yml. Add a 'scenarios' section.")
        sys.exit(1)

    if args.scenario:
        scenarios = [s for s in scenarios if s['name'] == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found. Available:")
            for s in config.get('scenarios', []):
                print(f"  - {s['name']}")
            sys.exit(1)

    print(f"Running {len(scenarios)} scenario(s)...")
    for s in scenarios:
        print(f"  - {s['name']}: {s['variable']} {s.get('direction', '')} {s.get('increments', [])}")

    # Run the actual scenario engine
    try:
        _run_scenarios(config, paths, project_path)
    except Exception as e:
        print(f"\nScenario simulation failed: {e}", file=sys.stderr)
        print("Ensure Stage 2 has completed and models are saved.")
        sys.exit(1)


def cmd_server(args):
    """Start the SPARC FastAPI server (used by the desktop app)."""
    # Fix GDAL/fiona HOME on Windows before any GeoPackage I/O
    if sys.platform == 'win32':
        import tempfile
        for var in ('CPL_TMPDIR', 'GDAL_PAM_PROXY_DIR'):
            if var not in os.environ:
                os.environ[var] = tempfile.gettempdir()
        home = os.environ.get('HOME', '')
        if not home or 'systemprofile' in home.lower() or not os.path.isdir(home):
            os.environ['HOME'] = os.path.expanduser('~')

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install 'sparc[server]'", file=sys.stderr)
        sys.exit(1)

    port = args.port
    dev = args.dev

    # If a project was specified, set it in the environment so the server
    # can auto-load it on startup.
    project_path = getattr(args, 'project', None)
    if project_path:
        resolved = str(Path(project_path).resolve())
        if not Path(resolved).exists():
            print(f"WARNING: --project file not found: {resolved}", file=sys.stderr)
        else:
            os.environ['SPARC_SERVER_PROJECT'] = resolved
            print(f"Auto-loading project: {resolved}")

    print(f"Starting SPARC server on http://127.0.0.1:{port}")
    if dev:
        print("  Dev mode: auto-reload enabled")
    uvicorn.run(
        "sparc.server.app:app",
        host="127.0.0.1",
        port=port,
        reload=dev,
    )


def cmd_report(args):
    """Generate final interpretation report.

    Renders one report per audience requested via ``--audience``
    (``all`` = every registered audience).  The default format is
    ``html``; pass ``--format pdf`` to also write a WeasyPrint PDF.
    Reports land under ``<output_dir>/Stage_5_Reports/``.
    """
    project_path = _resolve_project_path(args)

    repo_root = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo_root)

    from sparc.config.config import load_config
    from sparc.run.pipeline_paths import set_paths_from_config
    from sparc.report.audience import (
        AUDIENCES, FORMATS, generate_audience_report,
    )
    from sparc.report.stakeholder_index import write_stakeholder_index

    config = load_config(project_path)
    paths = set_paths_from_config(config)

    audience = getattr(args, "audience", "all")
    fmt      = getattr(args, "format",   "html")
    section  = getattr(args, "section",  None)
    if fmt not in FORMATS:
        print(f"ERROR: --format must be one of {FORMATS}", file=sys.stderr)
        sys.exit(2)

    if audience == "all":
        targets = list(AUDIENCES)
    else:
        if audience not in AUDIENCES:
            print(f"ERROR: --audience must be one of {AUDIENCES + ('all',)}",
                  file=sys.stderr)
            sys.exit(2)
        targets = [audience]

    out_dir = Path(paths.output_dir) / "Stage_5_Reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating report(s) for: {config.get('project', {}).get('name', 'Unnamed')}")
    print(f"  Output: {out_dir}")
    print(f"  Audiences: {', '.join(targets)}  ·  Format: {fmt}")

    written: list[Path] = []
    for aud in targets:
        try:
            payload = generate_audience_report(
                run_dir=Path(paths.output_dir),
                config=config,
                audience=aud,
                fmt=fmt,
            )
        except Exception as e:
            print(f"  [{aud}] FAILED: {e}", file=sys.stderr)
            continue
        suffix = {"md": "md", "html": "html", "pdf": "pdf"}[fmt]
        dest = out_dir / f"report_{aud}.{suffix}"
        if isinstance(payload, bytes):
            dest.write_bytes(payload)
        else:
            if section:
                # Crude markdown-section slice: keep the named ## heading
                # and everything until the next ## (or end of doc).
                import re as _re
                pat = _re.compile(rf"(^##\s+{_re.escape(section)}.*?)(?=^##\s+|\Z)",
                                  _re.M | _re.S)
                m = pat.search(payload)
                if m:
                    payload = m.group(1)
                else:
                    print(f"  [{aud}] section '{section}' not found; writing full report")
            dest.write_text(payload, encoding="utf-8")
        written.append(dest)
        print(f"  [{aud}] → {dest.name}")

    # Stakeholder question→section index (JSON)
    try:
        idx = write_stakeholder_index(out_dir / "stakeholder_index.json")
        print(f"  [index] → {idx.name}")
    except Exception as e:
        print(f"  [index] failed ({e})")

    print(f"Done. {len(written)} report(s) written.")


def cmd_desktop(args):
    """Launch the SPARC Desktop App (Tauri + FastAPI)."""
    import subprocess
    import threading

    # Start the FastAPI server in background
    port = args.port
    print(f"Starting SPARC server on http://127.0.0.1:{port}")

    def run_server():
        try:
            import uvicorn
            uvicorn.run("sparc.server.app:app", host="127.0.0.1", port=port, log_level="warning")
        except ImportError:
            print("ERROR: uvicorn not installed. Run: pip install 'sparc[server]'", file=sys.stderr)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Look for the Tauri binary
    desktop_dir = Path(__file__).resolve().parent.parent / "sparc-desktop"
    candidates = [
        desktop_dir / "src-tauri" / "target" / "release" / "SPARC Desktop",
        desktop_dir / "src-tauri" / "target" / "release" / "sparc-desktop",
        desktop_dir / "src-tauri" / "target" / "release" / "sparc-desktop.exe",
        desktop_dir / "src-tauri" / "target" / "debug" / "SPARC Desktop",
        desktop_dir / "src-tauri" / "target" / "debug" / "sparc-desktop",
    ]

    binary = None
    for c in candidates:
        if c.exists():
            binary = c
            break

    if binary:
        print(f"Launching desktop app: {binary}")
        subprocess.run([str(binary)], check=False)
    else:
        # Fallback: try pnpm tauri dev
        print("No compiled binary found. Attempting `pnpm tauri dev`...")
        try:
            subprocess.run(["pnpm", "tauri", "dev"], cwd=str(desktop_dir), check=True)
        except FileNotFoundError:
            print("ERROR: Neither a compiled SPARC Desktop binary nor pnpm was found.", file=sys.stderr)
            print("Build the desktop app first: cd sparc-desktop && pnpm tauri build", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# V3: Transfer learning
# ---------------------------------------------------------------------------

def cmd_transfer(args):
    """Run transfer learning validation: source city → target city."""
    from sparc.run.transfer_validation import run_transfer_validation

    source = Path(args.source_project).resolve()
    target = Path(args.target_project).resolve()
    if not source.exists():
        print(f"ERROR: Source project file not found: {source}", file=sys.stderr)
        sys.exit(1)
    if not target.exists():
        print(f"ERROR: Target project file not found: {target}", file=sys.stderr)
        sys.exit(1)
    output = Path(args.output).resolve()

    print(f"Transfer learning validation")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Output: {output}")
    print()

    comparison = run_transfer_validation(
        source_project=source,
        target_project=target,
        output_dir=output,
        run_finetune=not args.no_finetune,
        unfreeze_after=args.unfreeze_after,
    )

    print("\n=== Transfer Learning Results ===")
    for mode, data in comparison.items():
        if isinstance(data, dict) and "r2" in data:
            print(f"  {mode}: R²={data['r2']:.4f}" if data['r2'] else f"  {mode}: R²=N/A")


# ---------------------------------------------------------------------------
# V3: Continual learning
# ---------------------------------------------------------------------------

def cmd_continual(args):
    """Run continual learning across multiple cities."""
    from sparc.run.continual_training import train_continual

    city_paths = []
    for p in args.cities.split(","):
        cp = Path(p.strip()).resolve()
        if not cp.exists():
            print(f"ERROR: City project file not found: {cp}", file=sys.stderr)
            sys.exit(1)
        city_paths.append(str(cp))
    registry_path = Path(args.registry).resolve()
    output = Path(args.output).resolve()

    print(f"Continual learning across {len(city_paths)} cities")
    print(f"  Registry: {registry_path}")
    print(f"  Output:   {output}")
    for i, p in enumerate(city_paths, 1):
        print(f"  City {i}: {p}")
    print()

    result = train_continual(
        city_configs=city_paths,
        registry_path=registry_path,
        output_dir=output,
    )

    print("\n=== Continual Learning Results ===")
    for city_name, metrics in result.get("per_city_metrics", {}).items():
        print(f"  {city_name}: R²={metrics.get('r2', 'N/A')}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _run_wager2025_audit_addons(args, config: dict) -> None:
    """Run opt-in Wager (2025) audit estimators after Stage 3.

    Reads training data + DAG, fits selected estimators, and persists
    results to the active store at stage ``"3"`` so the resolver,
    scenario engine, and report all see them.
    """
    print("\n>>> Stage 3.5: Wager (2025) Audit Add-Ons")
    try:
        from sparc.registry.store import get_active_store
        from sparc.causal.dag_definition import load_dag, dag_to_networkx
        from sparc.data.data_utils import load_data
    except Exception as exc:
        print(f"  [skip] addon prerequisites unavailable: {exc}")
        return

    store = get_active_store()
    if store is None:
        print("  [skip] no active store; addons require Stage-3 store.")
        return

    data_path = (config.get('data') or {}).get('input_file') \
                or (config.get('data') or {}).get('path')
    if not data_path:
        print("  [skip] config has no data.input_file path; addons need raw data.")
        return
    try:
        data = load_data(data_path)
    except Exception as exc:
        print(f"  [skip] unable to load data: {exc}")
        return

    dag_path = (config.get('causal') or {}).get('dag_file')
    dag_def = load_dag(dag_path) if dag_path else None
    G = dag_to_networkx(dag_def) if dag_def else None

    target = config.get('variables', {}).get('target')
    treatments = list((config.get('predictors') or {}).get('treatment', []) or [])
    if not treatments and G is not None:
        treatments = [n for n, d in G.nodes(data=True)
                      if (d or {}).get('type') == 'treatment']

    # ---- Spillover ------------------------------------------------------
    if getattr(args, 'with_spillover', False):
        try:
            from sparc.causal.interference import NetworkInterferenceModel
            coord_cols = config.get('variables', {}).get('coordinates', ['x', 'y'])
            coords = data[coord_cols].to_numpy(dtype=float)
            rows = []
            for t in treatments:
                if t not in data.columns or target not in data.columns:
                    continue
                W = data[t].to_numpy(dtype=float)
                Y = data[target].to_numpy(dtype=float)
                d = NetworkInterferenceModel(k=8).fit(coords, W, Y).decompose()
                d.update({'treatment': t})
                rows.append(d)
            if rows:
                import pandas as pd
                store.write_table("3", "spillover_decomposition",
                                  pd.DataFrame(rows),
                                  producer="wager2025_audit",
                                  metadata={"treatments": treatments})
                print(f"  spillover: {len(rows)} treatment(s) decomposed.")
        except Exception as exc:
            print(f"  spillover failed: {exc}")

    # ---- IV -------------------------------------------------------------
    if getattr(args, 'with_iv', False):
        try:
            from sparc.causal.iv import CrossFitIV
            instruments = []
            if G is not None:
                instruments = [n for n, d in G.nodes(data=True)
                               if (d or {}).get('role') == 'instrument']
            if not instruments:
                print("  iv: no DAG nodes marked role:instrument; skipping.")
            else:
                rows = []
                for t in treatments:
                    if t not in data.columns or target not in data.columns:
                        continue
                    Z_cols = [z for z in instruments if z in data.columns]
                    if not Z_cols:
                        continue
                    Z = data[Z_cols].to_numpy(dtype=float)
                    iv = CrossFitIV(n_folds=5).fit(
                        Z=Z,
                        W=data[t].to_numpy(dtype=float),
                        Y=data[target].to_numpy(dtype=float),
                    )
                    rows.append({"treatment": t,
                                 "iv_coef": float(iv.coef_),
                                 "iv_se": float(iv.se_ or 0.0),
                                 "first_stage_f": float(iv.first_stage_f_ or 0.0),
                                 "weak_warning": iv.weak_warning_})
                if rows:
                    import pandas as pd
                    store.write_table("3", "iv_estimates", pd.DataFrame(rows),
                                      producer="wager2025_audit")
                    print(f"  iv: {len(rows)} treatment(s) estimated.")
        except Exception as exc:
            print(f"  iv failed: {exc}")

    # ---- Policy learning -----------------------------------------------
    if getattr(args, 'with_policy_learning', False):
        try:
            from sparc.decision.policy_learning import EmpiricalWelfareMaximizer
            from sparc.causal.cate_validation import aipw_pseudo_outcomes
            # Compute AIPW scores per treatment. Requires CATE/PDP context;
            # we use a simple residualisation as a proxy when DML nuisances
            # aren't already cached.
            import numpy as np
            import pandas as pd

            confounders = list((config.get('predictors') or {})
                               .get('confounders', []) or [])
            X_full = (data[confounders].to_numpy(dtype=float)
                      if confounders else
                      np.zeros((len(data), 1)))
            policies = []
            for t in treatments:
                if t not in data.columns or target not in data.columns:
                    continue
                W = (data[t].to_numpy(dtype=float)
                     > data[t].median()).astype(float)
                Y = data[target].to_numpy(dtype=float)
                # Naive nuisance proxies — refine when full DML cache exists.
                mu1 = np.full_like(Y, Y[W == 1].mean() if (W == 1).any() else 0.0)
                mu0 = np.full_like(Y, Y[W == 0].mean() if (W == 0).any() else 0.0)
                e = np.full_like(Y, max(W.mean(), 0.05))
                gamma = aipw_pseudo_outcomes(Y, W, mu1, mu0, e)
                pol = EmpiricalWelfareMaximizer(policy_class="linear").fit(
                    X=X_full, aipw_scores=gamma,
                )
                policies.append({"treatment": t,
                                 "n_recommended": int(pol.recommend().sum()),
                                 "welfare": float(pol.welfare_)})
            if policies:
                store.write_table("3", "policy_recommendations",
                                  pd.DataFrame(policies),
                                  producer="wager2025_audit")
                print(f"  policy_learning: {len(policies)} policies fit.")
        except Exception as exc:
            print(f"  policy_learning failed: {exc}")

    # ---- Triangulation summary -----------------------------------------
    if getattr(args, 'full_triangulation', False):
        try:
            from sparc.causal._audit import report
            store.write_blob("3", "wager2025_audit_report",
                             report(),
                             serializer="text",
                             producer="wager2025_audit")
            print("  triangulation: audit report persisted.")
        except Exception as exc:
            print(f"  triangulation failed: {exc}")


def cmd_audit_causal(args):
    """Print the Wager (2025) causal-inference audit gap registry."""
    from sparc.causal import _audit
    _audit._autodetect()
    print(_audit.report())
    n_done = sum(_audit.WAGER2025_GAPS_ADDRESSED.values())
    if n_done < len(_audit.GAP_SPECS):
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sparc',
        description='SPARC — Spatial Analysis and Research Core pipeline CLI',
    )
    parser.add_argument(
        '--verbosity', '-V',
        choices=['summary', 'normal', 'debug'],
        default=None,
        help='Terminal verbosity (default: normal). '
             'summary=stage banners + summaries + warnings/errors only; '
             'normal=adds checkpoints and key metrics; '
             'debug=full session log echoed to terminal. '
             'Can also be set via SPARC_VERBOSITY env var. '
             'session_log.jsonl always captures the full event stream.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- init ---
    p_init = subparsers.add_parser('init', help='Scaffold a new project from a template')
    p_init.add_argument('--template', '-t', default='blank',
                        help='Template name (default: blank). See templates/ directory.')
    p_init.add_argument('--output', '-o', required=True,
                        help='Output directory for the new project')
    p_init.set_defaults(func=cmd_init)

    # --- validate ---
    p_val = subparsers.add_parser('validate', help='Validate a project.yml configuration')
    p_val.add_argument('--project', '-p', required=True, help='Path to project.yml')
    p_val.set_defaults(func=cmd_validate)

    # --- run ---
    p_run = subparsers.add_parser('run', help='Run the SPARC pipeline')
    p_run.add_argument('--project', '-p', required=True, help='Path to project.yml')
    p_run.add_argument('--stage', '-s', choices=['0', '1', '2', '3', '4', '5', 'all'], default='all',
                       help='Which pipeline stage to run (0=Correlogram, 1=GWEN, 2=SpatialCV, 3=Causal, 4=Scenarios, all)')
    p_run.add_argument('--fast', action='store_true', help='Use fast mode (reduced precision)')
    p_run.add_argument('--skip-gwen', action='store_true', dest='skip_gwen',
                       help='Skip GWEN variable selection (Stage 1)')
    p_run.add_argument('--resume', action='store_true',
                       help='Resume from last completed stage (skip stages with markers)')
    p_run.add_argument('--legacy', action='store_true',
                       help='Force legacy V1/MGWR scenario paths (default: V4 engine when V2 neural artifacts present in DB)')
    # Wager (2025) audit opt-in flags. Default-off to preserve current
    # Stage-4 output; enabling these activates additive triangulation
    # columns (spillover, IV) and the policy-learning recommendation map.
    p_run.add_argument('--with-spillover', action='store_true',
                       dest='with_spillover',
                       help='Run network-interference (spillover) estimator alongside DML.')
    p_run.add_argument('--with-iv', action='store_true', dest='with_iv',
                       help='Run cross-fit IV estimation when DAG marks role:instrument nodes.')
    p_run.add_argument('--with-policy-learning', action='store_true',
                       dest='with_policy_learning',
                       help='Run empirical-welfare policy learning on AIPW scores.')
    p_run.add_argument('--full-triangulation', action='store_true',
                       dest='full_triangulation',
                       help='Run DML + CBPS + IV + DID across every edge (slow).')
    p_run.add_argument('--full-audit', action='store_true', dest='full_audit',
                       help='Run scenario engine in mode_5_full_audit.')
    p_run.set_defaults(func=cmd_run)

    # --- audit ---
    p_audit = subparsers.add_parser(
        'audit', help='Causal-inference audit utilities (Wager 2025).',
    )
    audit_subs = p_audit.add_subparsers(dest='audit_kind', required=True)
    p_audit_causal = audit_subs.add_parser(
        'causal', help='Show Wager (2025) audit gap-completion registry.',
    )
    p_audit_causal.set_defaults(func=cmd_audit_causal)

    # --- scenario ---
    p_scen = subparsers.add_parser('scenario', help='Run counterfactual scenario simulation')
    p_scen.add_argument('--project', '-p', required=True, help='Path to project.yml')
    p_scen.add_argument('--scenario', '-n', default=None,
                        help='Name of a specific scenario to run (default: all)')
    p_scen.add_argument('--legacy', action='store_true',
                        help='Force legacy V1/MGWR scenario paths (default: V4 engine when V2 neural artifacts present in DB)')
    p_scen.set_defaults(func=cmd_scenario)

    # --- report ---
    p_rep = subparsers.add_parser('report', help='Generate final interpretation report')
    p_rep.add_argument('--project', '-p', required=True, help='Path to project.yml')
    p_rep.add_argument('--audience', '-a', default='all',
                       help="Audience: technical | planner | public | council | "
                            "scientist | equity | auditor | all  (default: all)")
    p_rep.add_argument('--format', '-f', default='html',
                       choices=['md', 'html', 'pdf'],
                       help='Output format (default: html)')
    p_rep.add_argument('--section', default=None,
                       help='Render only the markdown section with this title '
                            '(applies only to md/html outputs)')
    p_rep.set_defaults(func=cmd_report)

    # --- server ---
    p_srv = subparsers.add_parser('server', help='Start the SPARC FastAPI server')
    p_srv.add_argument('--port', type=int, default=8008,
                       help='Port to bind (default: 8008)')
    p_srv.add_argument('--dev', action='store_true',
                       help='Enable auto-reload for development')
    p_srv.add_argument('--project', '-p', default=None,
                       help='Path to project.yml to auto-load on startup')
    p_srv.set_defaults(func=cmd_server)

    # --- desktop ---
    p_desk = subparsers.add_parser('desktop', help='Launch the SPARC Desktop App')
    p_desk.add_argument('--port', type=int, default=8008,
                        help='FastAPI server port (default: 8008)')
    p_desk.set_defaults(func=cmd_desktop)

    # --- transfer ---
    p_xfer = subparsers.add_parser('transfer', help='V3: Transfer learning validation (source → target city)')
    p_xfer.add_argument('--source-project', required=True,
                        help='Path to source city project.yml (trunk donor)')
    p_xfer.add_argument('--target-project', required=True,
                        help='Path to target city project.yml (trunk recipient)')
    p_xfer.add_argument('--output', '-o', default='transfer_results',
                        help='Output directory for transfer validation artifacts')
    p_xfer.add_argument('--no-finetune', action='store_true',
                        help='Skip warm-start-with-finetune mode')
    p_xfer.add_argument('--unfreeze-after', type=int, default=20,
                        help='Epoch to unfreeze trunk in finetune mode (default: 20)')
    p_xfer.set_defaults(func=cmd_transfer)

    # --- continual ---
    p_cont = subparsers.add_parser('continual', help='V3: Continual learning across multiple cities')
    p_cont.add_argument('--cities', required=True,
                        help='Comma-separated paths to city project.yml files')
    p_cont.add_argument('--registry', default='sparc_registry',
                        help='Path to city registry directory (default: sparc_registry)')
    p_cont.add_argument('--output', '-o', default='continual_results',
                        help='Output directory for continual training artifacts')
    p_cont.set_defaults(func=cmd_continual)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Install verbosity filter before any subcommand prints. The CLI flag
    # wins over the env var; if neither is set, default is "normal".
    try:
        from sparc.run.console import install as _install_verbosity
        _install_verbosity(getattr(args, 'verbosity', None) or os.environ.get('SPARC_VERBOSITY'))
    except Exception:
        pass  # Verbosity is a UX nicety -- never block the pipeline on it.
    try:
        args.func(args)
    except Exception as exc:
        # Surface low-memory aborts cleanly so the desktop log shows a
        # friendly message instead of a stack trace.
        try:
            from sparc.run.memory_watchdog import MemoryPressureAbort
        except Exception:
            raise
        if isinstance(exc, MemoryPressureAbort):
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"  PIPELINE ABORTED — {exc}", file=sys.stderr)
            print("  Free up memory and re-run with `sparc run --resume ...`", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == '__main__':
    main()
