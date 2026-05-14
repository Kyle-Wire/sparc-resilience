# PRD: Stage-by-Stage Pipeline Testing

## Problem Statement

The SPARC desktop application is in a stable state. Before declaring the pipeline ready for real analysis, we need to verify that each of the five pipeline stages runs to completion, produces the correct output artifacts, and passes them correctly to the next stage — using the real Brown University UHI dataset (`brown4.csv`).

## Goals

- Confirm each stage runs without errors under `--fast` mode
- Confirm each stage's expected output artifacts are produced and registered in `artifacts_manifest.json`
- Confirm the artifact hand-off between stages works (Stage N+1 can find and consume Stage N's outputs)
- Record the Stage 0 correlogram's recommended block size and compare it to the current user-specified 300m setting
- Establish a clean, reproducible baseline before any full-precision run

## Non-Goals

- Publication-quality model outputs (this is a plumbing test, not a science run)
- Desktop UI verification (second pass, after CLI baseline is clean)
- Hyperparameter tuning or model comparison
- Changing any config settings beyond what's needed to fix blockers

## Design Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Dataset | `my_project/brown4.csv` (54,701 pts) | Real data, UHI domain, known CRS |
| Config | `my_project/project.yml` | UHI template, already configured |
| Starting state | Wipe `my_project/output/` | Clean slate, no stale artifact conflicts |
| Run surface | CLI only | Raw stdout/stderr, isolates pipeline from desktop plumbing |
| Speed | `--fast` flag | Validate plumbing, not final results |
| Invocation | Stage-by-stage (`-s 0` through `-s 4`) | Manual inspection between stages |
| Branch | `testing` | Isolate any fixes from `main` |

## Technical Approach

Run stages sequentially from the `sparc-resilience` root:

```bash
# 0. Wipe output
rm -rf my_project/output/*

# 1. Stage by stage
sparc run -p my_project/project.yml -s 0 --fast
sparc run -p my_project/project.yml -s 1 --fast
sparc run -p my_project/project.yml -s 2 --fast
sparc run -p my_project/project.yml -s 3 --fast
sparc run -p my_project/project.yml -s 4 --fast
```

After each stage, inspect:
1. No Python exceptions / non-zero exit code
2. Expected artifact files exist on disk
3. `artifacts_manifest.json` has entries for the stage's artifacts

## Acceptance Criteria

### Stage 0 — Correlogram
- [ ] Exit code 0
- [ ] `variogram_analysis_results.json` written
- [ ] `pipeline_config.json` written
- [ ] `dataset_profile.json` written
- [ ] At least one `*_correlogram.png` written
- [ ] Note correlogram's recommended block size vs. user-set 300m

### Stage 1 — GWEN
- [ ] Exit code 0
- [ ] `gwen_results.json` written
- [ ] `selected_features.txt` written
- [ ] `gwen_variable_importance.csv` written
- [ ] All 6 predictors assessed; review which are selected

### Stage 2 — Spatial CV & Model Training
- [ ] Exit code 0
- [ ] `optimized_oof_predictions.csv` written
- [ ] `optimized_meta_model.txt` written
- [ ] `feature_scaler.pkl` written
- [ ] At least one model `.pkl` written
- [ ] 5 folds complete for each model (check fold counts in stdout)

### Stage 3 — Causal Validation
- [ ] Exit code 0
- [ ] `scenario_coefficients.json` written
- [ ] `causal_validation_summary.txt` written

### Stage 4 — Scenario Simulation
- [ ] Exit code 0
- [ ] `scenario_summary.csv` written
- [ ] `scenario_results.gpkg` written (or equivalent spatial output)

## Open Questions

- Will Stage 0's correlogram recommend a block size significantly different from 300m? If so, should we update `block_size_source: "correlogram"` in project.yml for the real run?
- Are there any stages that require additional project.yml config (e.g. scenario definitions for Stage 4) that haven't been verified yet?
