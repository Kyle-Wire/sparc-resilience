#!/usr/bin/env python3
"""
SPARC CLI — Spatial Research
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

    # ────────────────────────────────────────────────────────────────
    # Stage 0: Correlogram Analysis  (runs first so GWEN can auto-tune)
    # ────────────────────────────────────────────────────────────────
    if stage in ('0', 'all'):
        if not _stage_done("0"):
            print(">>> Stage 0: Correlogram Analysis")
            from sparc.run.correlogram_analysis import main as run_correlogram
            run_correlogram(fast_mode=fast)
            _mark_stage_done("0")
        else:
            print(">>> Stage 0: Correlogram — skipped (already complete)")
        _rescan_registry("0")

    # ────────────────────────────────────────────────────────────────
    # Stage 0b: Pipeline Configuration (auto-wire correlogram → config)
    # ────────────────────────────────────────────────────────────────
    if stage in ('0', 'all'):
        print("\n>>> Stage 0b: Pipeline Configuration")
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

    # ────────────────────────────────────────────────────────────────
    # Stage 1: GWEN variable selection (optional, now uses correlogram)
    # ────────────────────────────────────────────────────────────────
    if stage in ('1', 'all') and not skip_gwen:
        use_gwen = config.get('flags', {}).get('use_gwen_selection', True)
        approval_file = Path(config.get('output', {}).get('base_dir', 'output')) / 'gwen_approved.txt'

        if use_gwen and not _stage_done('.gwen_complete'):
            print("\n>>> Stage 1: GWEN Variable Selection")
            from sparc.run.gwen_variable_selection import main as run_gwen
            approved = run_gwen(config_path=project_path, fast_mode=fast)

            if not approved:
                print(f"\n{'='*60}")
                print("  PIPELINE PAUSED — Review GWEN feature selection")
                print(f"  Approve by creating: {approval_file}")
                print("  Then re-run with:  sparc run --resume ...")
                print(f"{'='*60}")
                return

            (paths.stage1_dir).mkdir(parents=True, exist_ok=True)
            (paths.stage1_dir / '.gwen_complete').write_text('done')
        else:
            print("\n>>> Stage 1: GWEN — skipped (already complete or disabled)")
        _rescan_registry("1")

    # ────────────────────────────────────────────────────────────────
    # Stage 2: Enhanced Spatial CV
    # ────────────────────────────────────────────────────────────────
    if stage in ('2', 'all'):
        print("\n>>> Stage 2: Enhanced Spatial CV")
        from sparc.run.enhanced_spatial_cv import main as run_spatial_cv
        run_spatial_cv(fast_mode=fast)
        _rescan_registry("2")

    # ────────────────────────────────────────────────────────────────
    # Stage 3: Causal Validation
    # ────────────────────────────────────────────────────────────────
    if stage in ('3', 'all'):
        print("\n>>> Stage 3: Causal Validation")
        try:
            from sparc.run.causal_validation import main as run_causal_validation
            run_causal_validation()
        except ImportError as e:
            print(f"  Stage 3 module not available ({e}) — skipping.")
        except Exception as e:
            print(f"  Stage 3 warning: {e}")
            print("  Continuing — Stage 4 will use physics priors only.")
        _rescan_registry("3")

    # ────────────────────────────────────────────────────────────────
    # Stage 4: Scenario Simulation (DAG + Physics)
    # ────────────────────────────────────────────────────────────────
    if stage in ('4', 'all'):
        scenarios = config.get('scenarios', [])
        auto_run = config.get('auto_run_scenarios_at_stage_4', True)
        if scenarios and auto_run:
            print("\n>>> Stage 4: Scenario Simulation")
            _run_scenarios(config, paths, project_path)
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

    print(f"\nPipeline complete. Results in: {paths.output_dir}")


def _resolve_auto_scenario_mode(*, has_dag: bool) -> str:
    """Pick a concrete ``scenario_mode`` for ``auto`` based on artifacts.

    Policy (Phase 5f of the v4 rewrite):

    1. Per-edge NUTS + α + base ensemble + DAG → ``mode_4_hybrid``.
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
        return "mode_4_hybrid"
    if has_dag and has_nuts and has_alpha:
        return "mode_2_dag_local"
    if has_alpha and has_ensemble:
        return "mode_3_full_ensemble"
    return "mode_1_physics"


def _try_run_with_v4_engine(config, sim, data, scenario_mode, has_dag):
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
    if scenario_mode == 'auto':
        scenario_mode = _resolve_auto_scenario_mode(has_dag=has_dag)
        print(f"  [auto] scenario_mode resolved to '{scenario_mode}'")

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
    """Generate final interpretation report."""
    project_path = _resolve_project_path(args)

    repo_root = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo_root)

    from sparc.config.config import load_config
    from sparc.run.pipeline_paths import set_paths_from_config

    config = load_config(project_path)
    paths = set_paths_from_config(config)

    print(f"Generating report for: {config.get('project', {}).get('name', 'Unnamed')}")
    print(f"  Output dir: {paths.output_dir}")
    print(f"  (Report generation integration point — connect to final_interpretation.py)")


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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sparc',
        description='SPARC — Spatial Research pipeline CLI',
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
    p_run.add_argument('--stage', '-s', choices=['0', '1', '2', '3', '4', 'all'], default='all',
                       help='Which pipeline stage to run (0=Correlogram, 1=GWEN, 2=SpatialCV, 3=Causal, 4=Scenarios, all)')
    p_run.add_argument('--fast', action='store_true', help='Use fast mode (reduced precision)')
    p_run.add_argument('--skip-gwen', action='store_true', dest='skip_gwen',
                       help='Skip GWEN variable selection (Stage 1)')
    p_run.add_argument('--resume', action='store_true',
                       help='Resume from last completed stage (skip stages with markers)')
    p_run.add_argument('--legacy', action='store_true',
                       help='Force legacy V1/MGWR scenario paths (default: V4 engine when V2 neural artifacts present in DB)')
    p_run.set_defaults(func=cmd_run)

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
    args.func(args)


if __name__ == '__main__':
    main()
