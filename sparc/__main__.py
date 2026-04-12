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
      1  Correlogram analysis     (auto-wires bandwidths into pipeline_config)
      1b Pipeline config generation
      2  Enhanced Spatial CV       (base models + meta-ensemble + deep kriging)
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

    # ── Helper: stage-complete checks for --resume ───────────────
    def _stage_done(marker_name):
        return resume and (paths.stage1_dir / marker_name).exists()

    # ────────────────────────────────────────────────────────────────
    # Stage 0: Correlogram Analysis  (runs first so GWEN can auto-tune)
    # ────────────────────────────────────────────────────────────────
    if stage in ('0', 'all'):
        if not _stage_done('.correlogram_complete'):
            print(">>> Stage 0: Correlogram Analysis")
            from sparc.run.correlogram_analysis import main as run_correlogram
            run_correlogram(fast_mode=fast)
            (paths.stage1_dir).mkdir(parents=True, exist_ok=True)
            (paths.stage1_dir / '.correlogram_complete').write_text('done')
        else:
            print(">>> Stage 0: Correlogram — skipped (already complete)")

    # ────────────────────────────────────────────────────────────────
    # Stage 0b: Pipeline Configuration (auto-wire correlogram → config)
    # ────────────────────────────────────────────────────────────────
    if stage in ('0', 'all'):
        print("\n>>> Stage 0b: Pipeline Configuration")
        from sparc.run.pipeline_configurator import PipelineConfigurator
        configurator = PipelineConfigurator(stage1_dir=str(paths.stage0_dir))

        # If dataset_profile.json exists, apply profiler recommendations
        profile_path = paths.stage0_dir / 'dataset_profile.json'
        if profile_path.exists():
            with open(profile_path) as _f:
                _profile = json.load(_f)
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

    # ────────────────────────────────────────────────────────────────
    # Stage 2: Enhanced Spatial CV
    # ────────────────────────────────────────────────────────────────
    if stage in ('2', 'all'):
        print("\n>>> Stage 2: Enhanced Spatial CV")
        from sparc.run.enhanced_spatial_cv import main as run_spatial_cv
        run_spatial_cv(fast_mode=fast)

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

    # ────────────────────────────────────────────────────────────────
    # Stage 4: Scenario Simulation (DAG + Physics)
    # ────────────────────────────────────────────────────────────────
    if stage in ('4', 'all'):
        scenarios = config.get('scenarios', [])
        if scenarios:
            print("\n>>> Stage 4: Scenario Simulation")
            _run_scenarios(config, paths, project_path)
        else:
            print("\n>>> Stage 4: No scenarios defined in project.yml — skipping.")

    print(f"\nPipeline complete. Results in: {paths.output_dir}")


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
    scenario_mode = config.get('pipeline', {}).get('scenario_mode', 'auto')

    # --- Resolve 'auto' to a concrete mode ----------------------------
    if scenario_mode == 'auto':
        scenario_mode = 'hybrid' if has_dag else 'physics'

    # --- Dispatch by scenario_mode ------------------------------------
    if scenario_mode == 'hybrid':
        print("  [PRIMARY] Hybrid: Model-consensus direct + DAG indirect")
        summary_df, results_gdf = sim.run_with_hybrid_reprediction(data, verbose=True)
    elif scenario_mode == 'model_reprediction':
        print("  [PRIMARY] Model re-prediction (full consensus delta)")
        summary_df, results_gdf = sim.run_with_model_reprediction(data, verbose=True)
    elif scenario_mode == 'dag_coefficient':
        if has_dag:
            print("  [PRIMARY] DAG + MGWR coefficient-based scenario simulation")
            summary_df, results_gdf = sim.run_with_causal_dag(data, verbose=True)
        else:
            print("  [PRIMARY] dag_coefficient requested but no DAG — falling back to physics")
            summary_df, results_gdf = sim.run(verbose=True)
    elif scenario_mode == 'bayesian':
        print("  [PRIMARY] Bayesian posterior scenario simulation (MC³ + NUTS)")
        summary_df, results_gdf = sim.run_bayesian_scenarios(data, verbose=True)
    else:
        print(f"  [PRIMARY] Physics-prior blending (mode={scenario_mode})")
        summary_df, results_gdf = sim.run(verbose=True)

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


def cmd_ui(args):
    """Launch the deprecated Streamlit UI."""
    import warnings
    warnings.warn(
        "The Streamlit UI is deprecated. Use 'sparc desktop' or 'sparc server' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print("WARNING: The Streamlit UI is deprecated. Use 'sparc desktop' instead.")
    print("Starting Streamlit UI anyway...")
    import subprocess
    ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=False)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sparc',
        description='SPARC — Spatial Analysis and Research Core pipeline CLI',
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
    p_run.set_defaults(func=cmd_run)

    # --- scenario ---
    p_scen = subparsers.add_parser('scenario', help='Run counterfactual scenario simulation')
    p_scen.add_argument('--project', '-p', required=True, help='Path to project.yml')
    p_scen.add_argument('--scenario', '-n', default=None,
                        help='Name of a specific scenario to run (default: all)')
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

    # --- ui (deprecated) ---
    p_ui = subparsers.add_parser('ui', help='[DEPRECATED] Launch Streamlit UI')
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
