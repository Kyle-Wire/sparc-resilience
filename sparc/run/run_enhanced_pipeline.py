#!/usr/bin/env python3
"""
Master Pipeline Runner for Enhanced SPARC Pipeline
Orchestrates the complete workflow from GWEN to Final Interpretation
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from tqdm import tqdm

# Centralized path management
from sparc.run.pipeline_paths import get_paths

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def check_stage_completion(stage_name):
    """
    Check if a pipeline stage has already been completed.

    Prefers the ``ResultStore`` manifest when available (which tracks
    partial vs complete artifacts), falling back to raw file-existence
    checks for backward compatibility.
    """
    paths = get_paths()

    # Try manifest-based check first
    try:
        from sparc.run.pipeline_paths import get_result_store
        store = get_result_store()
        stage_map = {
            "GWEN Auto-Approval": (1, ["gwen_approved.txt"]),
            "Variogram Analysis": (0, ["variogram_analysis_results.json", "variogram_summary.csv"]),
            "Pipeline Configuration": (None, None),  # not stage-managed
            "Enhanced Spatial CV": (2, ["optimized_oof_predictions.csv"]),
            "Causal Validation": (3, ["scenario_coefficients.json"]),
            "Final Interpretation": ("final", None),
        }
        if stage_name in stage_map:
            stage_key, required = stage_map[stage_name]
            if stage_key is not None:
                return store.is_stage_complete(stage_key, required)
    except Exception:
        pass

    # Fallback: file-existence checks
    if stage_name == "GWEN Auto-Approval":
        return os.path.exists(paths.gwen_approved)
    elif stage_name == "Variogram Analysis":
        return os.path.exists(paths.variogram_results) and os.path.exists(paths.variogram_summary)
    elif stage_name == "Pipeline Configuration":
        return os.path.exists(paths.pipeline_config)
    elif stage_name == "Enhanced Spatial CV":
        return os.path.exists(paths.oof_predictions) and os.path.exists(paths.folds_file)
    elif stage_name == "Causal Validation":
        return os.path.exists(os.path.join(str(paths.stage3_dir), 'scenario_coefficients.json'))
    elif stage_name == "Final Interpretation":
        return os.path.exists(paths.performance_metrics)
    else:
        return False

def run_stage(stage_name, script_path, description):
    """
    Run a pipeline stage and handle errors
    """
    print(f"\n{'='*60}")
    print(f"STAGE: {stage_name}")
    print(f"DESCRIPTION: {description}")
    print(f"SCRIPT: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if os.path.exists(script_path):
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, check=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"[SUCCESS] {stage_name} completed successfully in {duration:.2f} seconds")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            
            return True
            
        else:
            print(f"[FAILED] Script not found: {script_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[FAILED] {stage_name} failed after {duration:.2f} seconds")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        
        return False
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[FAILED] {stage_name} failed with exception after {duration:.2f} seconds")
        print(f"Exception: {e}")
        
        return False

def check_prerequisites():
    """
    Check if necessary files and directories exist
    """
    print("Checking prerequisites...")
    
    # Import config to get the current data file path
    from sparc.config.config import load_config
    try:
        _cfg = load_config()
        _data_file = _cfg.get('data', {}).get('file_path', '')
    except Exception:
        from sparc.config.config import RAW_CSV_PATH
        _data_file = RAW_CSV_PATH
    
    required_files = [
        _data_file
    ]
    
    optional_files = [
        "../output/stage1_gwen/results.json",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("[FAILED] Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure:")
        print("1. Data file '2x_systematic' exists in downsampled_data directory")
        return False
    
    # Check for GWEN results, run GWEN if missing
    paths = get_paths()
    gwen_results_file = paths.gwen_results
    
    print(f"[DEBUG] Looking for GWEN results at: {gwen_results_file}")
    print(f"[DEBUG] File exists: {os.path.exists(gwen_results_file)}")
    
    if not os.path.exists(gwen_results_file):
        print("[INFO] GWEN results not found. Will run GWEN variable selection first...")
        
        # Ensure output directory exists
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run GWEN variable selection
        print("[INFO] Running GWEN variable selection...")
        try:
            # Get the correct script path and working directory
            script_dir = os.path.dirname(__file__)
            gwen_script = os.path.join(script_dir, "gwen_variable_selection.py")
            
            result = subprocess.run([sys.executable, gwen_script], 
                                  capture_output=True, text=True, check=True, 
                                  cwd=script_dir)
            print("[SUCCESS] GWEN variable selection completed")
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] GWEN variable selection failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            print("Please run GWEN manually first")
            return False
    else:
        print("[SUCCESS] GWEN results found!")
    
    # Check if GWEN approval file exists, create it if missing
    gwen_approval_file = paths.gwen_approved
    if not os.path.exists(gwen_approval_file):
        print("[SUCCESS] Creating GWEN approval file for automated pipeline...")
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        with open(gwen_approval_file, 'w') as f:
            f.write(f"GWEN results automatically approved for enhanced pipeline\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Enhanced pipeline with variable-specific optimization\n")
    
    print("[SUCCESS] All prerequisites satisfied")
    return True

def create_pipeline_report(stage_results, total_time):
    """
    Create a comprehensive pipeline execution report
    """
    paths = get_paths()
    report_file = paths.pipeline_report_file
    
    with open(report_file, 'w') as f:
        f.write("SPARC ENHANCED PIPELINE EXECUTION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
        
        f.write("STAGE RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for stage_name, success in stage_results.items():
            status = "[SUCCESS]" if success else "[FAILED]"
            f.write(f"{stage_name}: {status}\n")
        
        f.write("\nOUTPUT DIRECTORIES:\n")
        f.write("-" * 30 + "\n")
        
        output_dirs = [
            ("GWEN Output", paths.output_dir),
            ("Stage 1 Variogram", paths.stage1_dir),
            ("Stage 2 Spatial CV", paths.stage2_dir), 
            ("Final Interpretation", paths.final_dir),
            ("Spatial Analysis", paths.spatial_analysis_dir)
        ]
        
        for dir_name, dir_path in output_dirs:
            exists = "[EXISTS]" if os.path.exists(dir_path) else "[MISSING]"
            f.write(f"{dir_name}: {exists}\n")
        
        f.write("\nFILES GENERATED:\n")
        f.write("-" * 30 + "\n")
        
        key_files = [
            ("GWEN Approval", paths.gwen_approved),
            ("Pipeline Config", paths.pipeline_config),
            ("Variogram Summary", paths.variogram_summary),
            ("OOF Predictions", paths.oof_predictions),
            ("Performance Metrics", paths.performance_metrics)
        ]
        
        for file_name, file_path in key_files:
            exists = "[EXISTS]" if os.path.exists(file_path) else "[MISSING]"
            f.write(f"{file_name}: {exists}\n")
    
    print(f"[SUCCESS] Pipeline report saved: {report_file}")

def main():
    """
    Main pipeline orchestrator
    """
    print("\n" + "="*80)
    print("SPARC ENHANCED PIPELINE WITH VARIABLE-SPECIFIC OPTIMIZATION")
    print("="*80)
    print("Pipeline Flow:")
    print("0. GWEN Variable Selection (if needed) → Feature Selection")
    print("1. GWEN Auto-Approval → Pipeline Automation")
    print("2. Variogram Analysis → Stage 1")
    print("3. Pipeline Configuration → Optimized Parameters") 
    print("4. Enhanced Spatial CV -> Stage 2 (+ V2 Neural Meta-Learner)")
    print("5. Causal Validation -> Stage 3 (+ V2 Bayesian MC3/NUTS)")
    print("6. Final Interpretation -> Local Coefficient Maps")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n[FAILED] Prerequisites not met. Exiting.")
        return False
    
    # Define pipeline stages with correct paths
    run_dir = os.path.join('Phase_1', 'run')
    stages = [
        {
            'name': 'GWEN Auto-Approval',
            'script': os.path.join(run_dir, 'gwen_auto_approve.py'),
            'description': 'Automatically approve GWEN results for pipeline automation'
        },
        {
            'name': 'Variogram Analysis',
            'script': os.path.join(run_dir, 'variogram_analysis.py'),
            'description': 'Analyze spatial autocorrelation for optimal bandwidth/kernel selection'
        },
        {
            'name': 'Pipeline Configuration',
            'script': os.path.join(run_dir, 'pipeline_configurator.py'),
            'description': 'Generate optimized pipeline configuration from variogram results'
        },
        {
            'name': 'Enhanced Spatial CV',
            'script': os.path.join(run_dir, 'enhanced_spatial_cv.py'),
            'description': 'Run spatial cross-validation with V2 neural meta-learner'
        },
        {
            'name': 'Causal Validation',
            'script': os.path.join(run_dir, 'causal_validation.py'),
            'description': 'Causal DAG validation + V2 Bayesian MC³ / NUTS (if enabled)'
        },
        {
            'name': 'Final Interpretation',
            'script': os.path.join(run_dir, 'final_interpretation.py'),
            'description': 'Generate local coefficient maps and comprehensive analysis'
        }
    ]
    
    # Track results and timing
    stage_results = {}
    pipeline_start_time = time.time()
    
    # Execute each stage with progress bar
    stage_progress = tqdm(stages, desc="Pipeline Stages", unit="stage")
    
    for stage in stage_progress:
        stage_progress.set_description(f"Running {stage['name']}")
        
        # Check if stage is already completed
        if check_stage_completion(stage['name']):
            print(f"\n[SKIP] {stage['name']} already completed - outputs exist")
            print(f"DESCRIPTION: {stage['description']}")
            stage_results[stage['name']] = True
            stage_progress.set_postfix(status="SKIPPED")
            continue
        
        success = run_stage(
            stage['name'],
            stage['script'],
            stage['description']
        )
        
        stage_results[stage['name']] = success
        
        if not success:
            stage_progress.close()
            print(f"\n[FAILED] Pipeline failed at stage: {stage['name']}")
            print("You can:")
            print("1. Fix the issue and re-run the entire pipeline")
            print("2. Run individual stages manually")
            print("3. Check the error messages above for debugging")
            break
        else:
            stage_progress.set_postfix(status="SUCCESS")
    
    if success:  # Only close if we completed normally
        stage_progress.close()
    
    # Calculate total time
    total_time = time.time() - pipeline_start_time
    
    # Generate report
    create_pipeline_report(stage_results, total_time)
    
    # Final summary
    successful_stages = sum(stage_results.values())
    total_stages = len(stage_results)
    
    print(f"\n{'='*80}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Stages Completed: {successful_stages}/{total_stages}")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if successful_stages == total_stages:
        print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated Outputs:")
        print("- stage0_correlogram/: Correlogram plots and optimal parameters")
        print("- stage1_gwen/: GWEN variable selection results")
        print("- stage2_spatial_cv/: Enhanced spatial CV results")
        print("- final/: Local coefficient maps and analysis")
    else:
        print("[FAILED] PIPELINE COMPLETED WITH ERRORS")
        print("Check individual stage outputs for details")
    
    print(f"{'='*80}")
    
    return successful_stages == total_stages

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
