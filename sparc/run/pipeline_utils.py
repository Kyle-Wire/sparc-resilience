#!/usr/bin/env python3
"""
Pipeline Utilities for GW3C Pipeline Orchestrator
Provides helper functions for pipeline management, validation, and status tracking.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


def validate_pipeline_prerequisites() -> Tuple[bool, List[str]]:
    """
    Validate that all prerequisites for pipeline execution are met.
    
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if config can be loaded
    try:
        import sys
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from sparc.config.config import load_config
        config = load_config()
    except Exception as e:
        issues.append(f"Configuration loading failed: {str(e)}")
        return False, issues
    
    # Check if data file exists
    data_path = config['data']['file_path']
    if not os.path.exists(data_path):
        issues.append(f"Data file not found: {data_path}")
    
    # Check if required modules are available
    required_modules = [
        'models.ols', 'models.gwr', 'models.gwrf', 'models.ggpgam',
        'features.laplacian', 'data.data_utils'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            issues.append(f"Required module missing: {module} ({str(e)})")
    
    # Check output directory permissions
    output_dir = config['output']['base_dir']
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, 'test_permissions.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        issues.append(f"Output directory not writable: {output_dir} ({str(e)})")
    
    return len(issues) == 0, issues


def load_pipeline_status(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load pipeline status from file.
    
    Parameters:
    -----------
    output_dir : str
        Output directory containing pipeline status
        
    Returns:
    --------
    Dict[str, Any] or None
        Pipeline status dictionary or None if not found
    """
    status_file = os.path.join(output_dir, 'pipeline_status.json')
    
    if not os.path.exists(status_file):
        return None
    
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load pipeline status: {str(e)}")
        return None


def check_stage_completion(output_dir: str, stage: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a specific pipeline stage has been completed.
    
    Parameters:
    -----------
    output_dir : str
        Output directory to check
    stage : str
        Stage name ('gwen' or 'spatial_cv')
        
    Returns:
    --------
    Tuple[bool, Optional[str]]
        (is_completed, timestamp_or_none)
    """
    if stage == 'gwen':
        # Prefer artifacts.db; fall back to legacy on-disk JSON.
        approved_file = os.path.join(output_dir, 'gwen_approved.txt')
        gwen_data = None
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None and _store.has("1", "gwen_results"):
                gwen_data = _store.read_any("1", "gwen_results")
        except Exception:  # noqa: BLE001
            gwen_data = None
        if gwen_data is None:
            gwen_file = os.path.join(output_dir, 'gwen_results.json')
            if os.path.exists(gwen_file):
                try:
                    with open(gwen_file, 'r') as f:
                        gwen_data = json.load(f)
                except Exception:  # noqa: BLE001
                    gwen_data = None
        if gwen_data is not None and os.path.exists(approved_file):
            return True, gwen_data.get('timestamp')
        return False, None
    
    elif stage == 'spatial_cv':
        # Check for spatial CV output files
        required_files = [
            'oof_predictions.csv',
            'performance_summary.csv',
            'fold_assignments.csv'
        ]
        
        all_exist = all(os.path.exists(os.path.join(output_dir, f)) 
                       for f in required_files)
        
        if all_exist:
            # Get timestamp from most recent file
            timestamps = []
            for f in required_files:
                try:
                    stat = os.stat(os.path.join(output_dir, f))
                    timestamps.append(datetime.fromtimestamp(stat.st_mtime))
                except:
                    continue
            
            if timestamps:
                return True, max(timestamps).isoformat()
        
        return False, None
    
    return False, None


def get_gwen_summary(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get GWEN variable selection summary.
    
    Parameters:
    -----------
    output_dir : str
        Output directory containing GWEN results
        
    Returns:
    --------
    Dict[str, Any] or None
        GWEN summary or None if not available
    """
    gwen_data = None
    try:
        from sparc.registry.store import get_active_store
        _store = get_active_store()
        if _store is not None and _store.has("1", "gwen_results"):
            gwen_data = _store.read_any("1", "gwen_results")
    except Exception:  # noqa: BLE001
        gwen_data = None
    if gwen_data is None:
        gwen_file = os.path.join(output_dir, 'gwen_results.json')
        if not os.path.exists(gwen_file):
            return None
        try:
            with open(gwen_file, 'r') as f:
                gwen_data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not read GWEN results: {str(e)}")
            return None

    return {
        'total_features': gwen_data.get('total_features', 0),
        'selected_features': gwen_data.get('selected_features', []),
        'selection_threshold': gwen_data.get('selection_threshold', 0),
        'timestamp': gwen_data.get('timestamp'),
        'num_selected': len(gwen_data.get('selected_features', []))
    }


def get_spatial_cv_summary(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get spatial cross-validation summary.
    
    Parameters:
    -----------
    output_dir : str
        Output directory containing spatial CV results
        
    Returns:
    --------
    Dict[str, Any] or None
        Spatial CV summary or None if not available
    """
    perf_file = os.path.join(output_dir, 'performance_summary.csv')
    
    if not os.path.exists(perf_file):
        return None
    
    try:
        perf_df = pd.read_csv(perf_file)
        
        # Get best performing model
        if 'R²' in perf_df.columns:
            best_idx = perf_df['R²'].idxmax()
            best_model = perf_df.loc[best_idx, 'Model']
            best_r2 = perf_df.loc[best_idx, 'R²']
            best_rmse = perf_df.loc[best_idx, 'RMSE']
        else:
            best_model = 'Unknown'
            best_r2 = 0
            best_rmse = 0
        
        return {
            'num_models': len(perf_df),
            'best_model': best_model,
            'best_r2': best_r2,
            'best_rmse': best_rmse,
            'all_results': perf_df.to_dict('records')
        }
    except Exception as e:
        print(f"Warning: Could not read spatial CV results: {str(e)}")
        return None


def create_pipeline_report(output_dir: str) -> str:
    """
    Create a comprehensive pipeline execution report.
    
    Parameters:
    -----------
    output_dir : str
        Output directory containing pipeline results
        
    Returns:
    --------
    str
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("GW3C PIPELINE EXECUTION REPORT")
    report.append("=" * 80)
    
    # Load pipeline status
    status = load_pipeline_status(output_dir)
    if status:
        pipeline_info = status.get('pipeline_info', {})
        report.append(f"\nPipeline Information:")
        report.append(f"  Start Time: {pipeline_info.get('start_time', 'Unknown')}")
        report.append(f"  End Time: {pipeline_info.get('end_time', 'Unknown')}")
        report.append(f"  Method: {pipeline_info.get('feature_selection_method', 'Unknown')}")
        report.append(f"  Output: {pipeline_info.get('output_directory', output_dir)}")
    
    # Check stage completion
    gwen_completed, gwen_time = check_stage_completion(output_dir, 'gwen')
    cv_completed, cv_time = check_stage_completion(output_dir, 'spatial_cv')
    
    report.append(f"\nStage Completion:")
    report.append(f"  GWEN Variable Selection: {'✅ Complete' if gwen_completed else '❌ Incomplete'}")
    if gwen_time:
        report.append(f"    Completed: {gwen_time}")
    
    report.append(f"  Spatial Cross-Validation: {'✅ Complete' if cv_completed else '❌ Incomplete'}")
    if cv_time:
        report.append(f"    Completed: {cv_time}")
    
    # GWEN Summary
    gwen_summary = get_gwen_summary(output_dir)
    if gwen_summary:
        report.append(f"\nGWEN Variable Selection Summary:")
        report.append(f"  Total Features Evaluated: {gwen_summary['total_features']}")
        report.append(f"  Features Selected: {gwen_summary['num_selected']}")
        report.append(f"  Selection Threshold: {gwen_summary['selection_threshold']}")
        report.append(f"  Selected Features: {', '.join(gwen_summary['selected_features'])}")
    
    # Spatial CV Summary
    cv_summary = get_spatial_cv_summary(output_dir)
    if cv_summary:
        report.append(f"\nSpatial Cross-Validation Summary:")
        report.append(f"  Models Evaluated: {cv_summary['num_models']}")
        report.append(f"  Best Performing Model: {cv_summary['best_model']}")
        report.append(f"  Best R²: {cv_summary['best_r2']:.4f}")
        report.append(f"  Best RMSE: {cv_summary['best_rmse']:.4f}")
    
    # Output Files
    report.append(f"\nOutput Files in {output_dir}:")
    if os.path.exists(output_dir):
        for file in sorted(os.listdir(output_dir)):
            if os.path.isfile(os.path.join(output_dir, file)):
                report.append(f"  📄 {file}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def cleanup_pipeline_outputs(output_dir: str, keep_final_results: bool = True) -> List[str]:
    """
    Clean up intermediate pipeline outputs.
    
    Parameters:
    -----------
    output_dir : str
        Output directory to clean
    keep_final_results : bool
        Whether to keep final results files
        
    Returns:
    --------
    List[str]
        List of files that were removed
    """
    if not os.path.exists(output_dir):
        return []
    
    # Files to potentially remove
    intermediate_files = [
        'fold_assignments.csv',
        'model_params.pkl',
        'meta_model.txt',
        'meta_model.pkl',
        'pipeline_status.json'
    ]
    
    # Model files
    model_patterns = [
        'final_ols_model.pkl',
        'final_gwr_model.pkl',
        'final_gwrf_model.pkl',
        'final_ggpgam_model.pkl'
    ]
    
    files_to_remove = intermediate_files
    if not keep_final_results:
        files_to_remove.extend(model_patterns)
        files_to_remove.extend([
            'oof_predictions.csv',
            'performance_summary.csv',
            'gwen_results.json',
            'gwen_variable_importance.csv',
            'selected_features.txt'
        ])
    
    removed_files = []
    for file in files_to_remove:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(file)
            except Exception as e:
                print(f"Warning: Could not remove {file}: {str(e)}")
    
    return removed_files


def export_pipeline_config(output_dir: str, config: Dict[str, Any]) -> str:
    """
    Export pipeline configuration to a file.
    
    Parameters:
    -----------
    output_dir : str
        Output directory
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns:
    --------
    str
        Path to exported config file
    """
    config_file = os.path.join(output_dir, 'pipeline_config.json')
    
    # Add export metadata
    export_config = {
        'export_info': {
            'exported_at': datetime.now().isoformat(),
            'gw3c_pipeline_version': '1.0.0'
        },
        'configuration': config
    }
    
    with open(config_file, 'w') as f:
        json.dump(export_config, f, indent=2)
    
    return config_file 