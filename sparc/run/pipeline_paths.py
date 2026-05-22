#!/usr/bin/env python3
"""
Centralized Path Management for SPARC Pipeline
Ensures consistent file paths regardless of which script is run or from where.

Two construction modes:
  1. ``PipelinePaths.from_config(config_dict)``  — preferred; all directory
     names come from ``project.yml`` via the config dict.
  2. ``PipelinePaths()``  — legacy auto-discovery for backward compatibility.
"""

import os
import warnings
from pathlib import Path

class PipelinePaths:
    """
    Centralized path management for the SPARC pipeline.
    All paths are absolute and consistent regardless of execution context.
    """
    
    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> 'PipelinePaths':
        """
        Build ``PipelinePaths`` from a loaded config dict (returned by
        ``load_config()``).  Stage directory names are read from
        ``config['output']['stage_dirs']`` when present.

        Parameters
        ----------
        config : dict
            Config dict as returned by ``config.config.load_config()``.

        Returns
        -------
        PipelinePaths
        """
        obj = cls.__new__(cls)

        # --- root dirs ---
        obj.run_dir = obj._find_run_directory()
        obj.phase1_dir = obj.run_dir.parent
        obj.project_root = Path(config.get('paths', {}).get('project_root', obj.phase1_dir.parent))

        # --- output root ---
        output_base = config.get('output', {}).get('base_dir',
                      config.get('paths', {}).get('output_dir', str(obj.phase1_dir / 'output')))
        obj.output_dir = Path(output_base)

        # --- stage directories (configurable names) ---
        stage_dirs = config.get('output', {}).get('stage_dirs', {})
        obj.stage0_dir = obj.output_dir / stage_dirs.get('stage_0', 'Stage_0_Correlogram')
        obj.stage1_dir = obj.output_dir / stage_dirs.get('stage_1', 'Stage_1_GWEN')
        obj.stage2_dir = obj.output_dir / stage_dirs.get('stage_2', 'Stage_2_Spatial_CV')
        obj.stage3_dir = obj.output_dir / stage_dirs.get('stage_3', 'Stage_3_Causal_Validation')
        obj.stage4_dir = obj.output_dir / stage_dirs.get('stage_4', 'Stage_4_Scenarios')
        obj.final_dir  = obj.output_dir / stage_dirs.get('final',   'Final_Interpretation_Results')
        obj.spatial_analysis_dir = obj.output_dir / 'Spatial_Analysis_Results'

        obj._ensure_directories()
        return obj

    def __init__(self):
        """Legacy auto-discovery constructor (backward-compatible).

        .. deprecated::
            Use ``PipelinePaths.from_config(config)`` instead.  The auto-
            discovery path heuristic can silently resolve to the wrong
            directory when the project tree differs from the expected layout.
        """
        warnings.warn(
            "PipelinePaths() auto-discovery is deprecated. "
            "Use PipelinePaths.from_config(config) or set the SPARC_PROJECT "
            "env var to your project.yml path.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Find the run directory (where scripts are located)
        self.run_dir = self._find_run_directory()
        
        # Define all standard directories relative to run directory
        self.phase1_dir = self.run_dir.parent
        self.project_root = self.phase1_dir.parent
        
        # Load output directory from project.yml via load_config()
        import sys
        
        config_loaded = False
        if str(self.phase1_dir) not in sys.path:
            sys.path.insert(0, str(self.phase1_dir))
        try:
            from sparc.config.config import load_config
            config = load_config()
            # Output directories - Use config-defined output directory
            self.output_dir = Path(config['output']['base_dir'])
            config_loaded = True
        except (ImportError, ModuleNotFoundError, KeyError) as e:
            # Final fallback to default output directory
            print(f"Warning: Could not load config ({e}), using default output directory")
            self.output_dir = self.phase1_dir / "Final_Results"
        
        self.stage0_dir = self.output_dir / "Stage_0_Correlogram"
        self.stage1_dir = self.output_dir / "Stage_1_GWEN"
        self.stage2_dir = self.output_dir / "Stage_2_Spatial_CV"
        self.stage3_dir = self.output_dir / "Stage_3_Causal_Validation"
        self.stage4_dir = self.output_dir / "Stage_4_Scenarios"
        self.final_dir = self.output_dir / "Final_Interpretation_Results"
        self.spatial_analysis_dir = self.output_dir / "Spatial_Analysis_Results"
        
        # Ensure all directories exist
        self._ensure_directories()
        
    def _find_run_directory(self):
        """
        Find the run directory by looking for characteristic files
        Works from any execution context
        """
        # Start from current script location
        current_file = Path(__file__).resolve()
        search_dir = current_file.parent
        
        # Look for the run directory by finding characteristic files
        for _ in range(5):  # Search up to 5 levels up
            if (search_dir / "enhanced_spatial_cv.py").exists() and \
               (search_dir / "pipeline_paths.py").exists():
                return search_dir
            search_dir = search_dir.parent
            
        # Fallback: assume we're in the run directory
        if current_file.name in ["enhanced_spatial_cv.py", "final_interpretation.py", "run_enhanced_pipeline.py"]:
            return current_file.parent
            
        # Final fallback: construct expected path
        # This handles cases where script is run from different locations
        expected_run_dir = Path.cwd() / "Phase_1" / "run"
        if expected_run_dir.exists():
            return expected_run_dir.resolve()
            
        # If all else fails, use current directory but warn loudly.
        fallback = Path.cwd()
        warnings.warn(
            f"PipelinePaths could not locate the sparc/run/ directory via "
            f"auto-discovery; falling back to cwd '{fallback}'. "
            "Artifact paths may be wrong. Use PipelinePaths.from_config(config) "
            "to specify paths explicitly.",
            RuntimeWarning,
            stacklevel=3,
        )
        return fallback
    
    def _ensure_directories(self):
        """Ensure necessary directories exist.

        When disk writes are disabled (the default), only the run-wide
        ``output_dir`` is created — that is where ``artifacts.db`` and the
        manifest sidecar live.  Per-stage subdirectories are skipped so the
        run directory stays clean (no empty ``Stage_*`` folders).  Path
        attributes are still populated so downstream code can compute
        ``stage2_dir / 'foo.csv'``-style identifiers without changes.
        """
        try:
            from sparc.run.disk_policy import disk_writes_enabled
            disk_on = disk_writes_enabled()
        except Exception:  # noqa: BLE001 — never fail path init
            disk_on = True

        # Always ensure the output root so RunRegistry can place artifacts.db.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not disk_on:
            return

        for directory in (
            self.stage0_dir,
            self.stage1_dir,
            self.stage2_dir,
            self.stage3_dir,
            self.stage4_dir,
            self.final_dir,
            self.spatial_analysis_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
    
    # Configuration files
    @property
    def pipeline_config(self):
        return self.stage2_dir / "pipeline_config.json"
    
    @property
    def folds_file(self):
        return self.stage2_dir / "folds.pkl"
    
    @property
    def pipeline_report_file(self):
        return self.project_root / "pipeline_execution_report.txt"
    
    # GWEN files (Stage 1)
    @property
    def gwen_results(self):
        return self.stage1_dir / "gwen_results.json"
    
    @property
    def gwen_approved(self):
        return self.stage1_dir / "gwen_approved.txt"
    
    @property
    def selected_features(self):
        return self.stage1_dir / "selected_features.txt"
    
    # Stage 0 files (Correlogram)
    @property
    def variogram_results(self):
        return self.stage0_dir / "variogram_analysis_results.json"
    
    @property
    def variogram_summary(self):
        return self.stage0_dir / "variogram_summary.csv"
    
    @property
    def correlogram_results(self):
        return self.stage0_dir / "correlogram_analysis_results.json"

    @property
    def correlogram_summary(self):
        # Stage 0 writes this alongside ``correlogram_analysis_results.json``;
        # the legacy filename is kept for backward compatibility, the
        # ``correlogram_*`` aliases keep artifact-id / filename naming
        # consistent with the registry catalogue.
        return self.stage0_dir / "variogram_summary.csv"
    
    # Stage 2 files
    @property
    def oof_predictions(self):
        return self.stage2_dir / "optimized_oof_predictions.csv"
    
    @property
    def meta_model(self):
        return self.stage2_dir / "optimized_meta_model.txt"
    
    @property
    def laplacian_features(self):
        return self.stage2_dir / "laplacian_features.pkl"
    
    @property
    def laplacian_eigenvalues(self):
        return self.stage2_dir / "laplacian_eigenvalues.pkl"
    
    @property
    def spatial_weights(self):
        return self.stage2_dir / "spatial_weights.pkl"
    
    # Stage 3 files - Causal Validation
    @property
    def spatial_intelligence_dir(self):
        """Directory for spatial intelligence outputs (legacy alias)"""
        return self.stage3_dir

    @property
    def scenario_coefficients(self):
        """DAG-adjusted scenario coefficients from causal validation"""
        return self.stage3_dir / "scenario_coefficients.json"

    @property
    def causal_validation_summary(self):
        """Human-readable causal validation summary"""
        return self.stage3_dir / "causal_validation_summary.txt"
    
    @property
    def gwr_modifier_maps(self):
        """GWR M_ik modifier maps (legacy)"""
        return self.stage3_dir / "gwr_modifier_maps.csv"
    
    @property
    def gwrf_condition_curves(self):
        """GWRF aggregate PDP condition curves (generated by Stage 2b).
        
        Used by Stage 4 ScenarioSimulator for saturation-aware scenario
        predictions via ``predict_scenario_with_saturation()``.

        Canonical location is ``Stage_2_Spatial_CV/gwrf_pdp/``.  Falls back
        to the legacy ``spatial_intelligence/gwrf_pdp/`` path for existing
        project outputs.
        """
        canonical = self.stage2_dir / "gwrf_pdp" / "gwrf_condition_curves.json"
        if canonical.exists():
            return canonical
        legacy = Path(self.output_dir) / "spatial_intelligence" / "gwrf_pdp" / "gwrf_condition_curves.json"
        if legacy.exists():
            return legacy
        return canonical  # default to canonical for new writes
    
    @property
    def ggpgam_uncertainty_weights(self):
        """GGPGAM W_unc uncertainty weights (legacy)"""
        return self.stage3_dir / "ggpgam_uncertainty_weights.csv"
    
    @property
    def oof_diagnostics(self):
        """OOF extraction diagnostics (legacy)"""
        return self.stage3_dir / "oof_diagnostics.json"
    
    # Stage 4 files - Scenario Simulation
    @property
    def scenario_summary(self):
        """Scenario simulation summary CSV"""
        return self.stage4_dir / "scenario_summary.csv"

    @property
    def scenario_results_gpkg(self):
        """Scenario results GeoPackage"""
        return self.stage4_dir / "scenario_results.gpkg"
    
    # Final interpretation files
    @property
    def final_results(self):
        return self.final_dir / "final_ensemble_results.json"
    
    @property
    def final_predictions(self):
        return self.final_dir / "final_ensemble_predictions.csv"
    
    @property
    def performance_metrics(self):
        return self.final_dir / "Performance_Analysis" / "model_performance_metrics.csv"
    
    # Spatial Analysis files
    @property
    def spatial_autocorr_fold_results(self):
        return self.spatial_analysis_dir / "fold_wise_spatial_autocorrelation.csv"
    
    @property
    def spatial_autocorr_summary(self):
        return self.spatial_analysis_dir / "spatial_autocorr_summary.csv"
    
    @property
    def spatial_autocorr_heatmap(self):
        return self.spatial_analysis_dir / "spatial_autocorr_heatmap.png"
    
    @property
    def laplacian_tsne_plot(self):
        return self.spatial_analysis_dir / "laplacian_tsne_visualization.png"
    
    @property
    def laplacian_tsne_coords(self):
        return self.spatial_analysis_dir / "laplacian_tsne_coordinates.csv"

    # Stage 1 additional files
    @property
    def correlogram_csv(self):
        return self.stage1_dir / "correlogram_analysis_results.csv"

    @property
    def correlogram_json(self):
        return self.stage1_dir / "correlogram_analysis_results.json"

    # Stage 2 additional files
    @property
    def ensemble_predictions_gpkg(self):
        return self.stage2_dir / "final_ensemble_predictions.gpkg"

    @property
    def model_metrics_csv(self):
        return self.stage2_dir / "model_performance_metrics.csv"

    @property
    def feature_scaler(self):
        return self.stage2_dir / "feature_scaler.pkl"

    # Stage 3 additional files
    @property
    def causal_diagnostics(self):
        return self.stage3_dir / "causal_diagnostics.json"

    # Stage 4 additional files
    @property
    def scenario_report(self):
        return self.stage4_dir / "scenario_report.txt"

    # Fishnet / data processing
    @property
    def fishnet_gpkg(self):
        return self.output_dir / "fishnet.gpkg"

    @property
    def fishnet_with_stats(self):
        return self.output_dir / "fishnet_with_stats.gpkg"

    def ensure_all_stage_dirs(self):
        """Ensure every stage directory (and parents) exists."""
        self._ensure_directories()
    
    # Data files (with fallback logic)
    def get_data_file(self):
        """
        Find the data file with fallback logic
        Looks in multiple possible locations
        """
        from sparc.config.config import RAW_CSV_PATH
        
        # Try the configured path first
        if os.path.exists(RAW_CSV_PATH):
            return RAW_CSV_PATH
            
        # Try relative to project root
        possible_paths = [
            self.project_root / "with_two_form_indices_2x_systematic.csv",
            self.project_root / "Phase_1" / "with_two_form_indices_2x_systematic.csv",
            self.phase1_dir / "with_two_form_indices_2x_systematic.csv"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Return configured path even if it doesn't exist (for error reporting)
        return RAW_CSV_PATH
    
    def get_relative_path(self, full_path):
        """Get a path relative to the run directory for display purposes"""
        try:
            return Path(full_path).relative_to(self.run_dir)
        except ValueError:
            return Path(full_path)
    
    def __str__(self):
        """String representation showing key paths"""
        return f"""PipelinePaths:
  Run Directory: {self.run_dir}
  Phase1 Directory: {self.phase1_dir}
  Project Root: {self.project_root}
  
  Output Directories:
    Stage 1 (Correlogram): {self.stage1_dir}
    Stage 2 (Spatial CV):  {self.stage2_dir}
    Stage 3 (Causal):      {self.stage3_dir}
    Stage 4 (Scenarios):   {self.stage4_dir}
    Final: {self.final_dir}
  
  Key Files:
    Pipeline Config: {self.pipeline_config}
    GWEN Results: {self.gwen_results}
    OOF Predictions: {self.oof_predictions}
    Scenario Coefficients: {self.scenario_coefficients}
"""

# ---------------------------------------------------------------------------
# Module-level singleton (legacy compatibility)
# ---------------------------------------------------------------------------
# When imported directly (``from pipeline_paths import get_paths``), a default
# auto-discovery instance is created.  When using project.yml, call
# ``set_paths_from_config(config)`` early in start-up instead.

_global_paths: PipelinePaths | None = None


def get_paths() -> PipelinePaths:
    """Return the global ``PipelinePaths`` instance (creates via auto-discovery on first call).

    If the ``SPARC_PROJECT`` environment variable is set and the singleton
    hasn't been explicitly initialised yet, builds paths from the
    referenced project.yml so that all stages share the same layout.
    """
    global _global_paths
    if _global_paths is None:
        project_yml = os.environ.get('SPARC_PROJECT')
        if project_yml:
            from sparc.config.config import load_config
            config = load_config(project_yml)
            _global_paths = PipelinePaths.from_config(config)
        else:
            _global_paths = PipelinePaths()
    return _global_paths


def set_paths_from_config(config: dict) -> PipelinePaths:
    """
    Initialise the global ``PipelinePaths`` from a config dict.
    Call this once at pipeline start-up when using ``project.yml``.
    """
    global _global_paths
    _global_paths = PipelinePaths.from_config(config)
    return _global_paths


# ---------------------------------------------------------------------------
# ResultStore singleton
# ---------------------------------------------------------------------------

_global_store = None


def get_result_store():
    """Return the global ``ResultStore`` instance, creating it on first call.

    Phase D: when an active registry is installed (see
    :func:`sparc.registry.run_registry.set_active_registry`), the
    ResultStore is created bound to it so every save_* call auto-registers.
    Subsequent calls re-use the same instance; if the registry was attached
    later, we attach it on the fly.
    """
    global _global_store
    from sparc.run.result_store import ResultStore
    try:
        from sparc.registry.run_registry import get_active_registry
        active_reg = get_active_registry()
    except Exception:
        active_reg = None
    if _global_store is None:
        _global_store = ResultStore(get_paths(), registry=active_reg)
    elif active_reg is not None and getattr(_global_store, "registry", None) is None:
        # Late-bound: registry came online after first use.
        _global_store.registry = active_reg
    return _global_store


def set_result_store(store):
    """Override the global ``ResultStore`` (e.g. in tests)."""
    global _global_store
    _global_store = store


# Backwards-compatible shorthand: code that does ``from pipeline_paths import paths``
# will get a lazy-initialised instance.  Prefer ``get_paths()`` in new code.
class _LazyPaths:
    """Descriptor that creates the PipelinePaths singleton on first access."""
    _instance = None
    def __getattr__(self, name):
        return getattr(get_paths(), name)

paths = _LazyPaths()


if __name__ == "__main__":
    # Test the path resolution
    p = get_paths()
    print(p)
    print(f"\nData file: {p.get_data_file()}")
    print(f"All directories created successfully: {all(d.exists() for d in [p.stage1_dir, p.stage2_dir, p.final_dir])}")