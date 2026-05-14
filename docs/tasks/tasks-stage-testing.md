# Tasks: Stage-by-Stage Pipeline Testing

Related PRD: docs/prd/prd-stage-testing.md

## Tasks

- [ ] Wipe `my_project/output/` for a clean slate
- [ ] Run Stage 0: `sparc run -p my_project/project.yml -s 0 --fast`
  - Verify: exit code 0
  - Verify: `Stage_0_Correlogram/variogram_analysis_results.json` exists
  - Verify: `Stage_0_Correlogram/pipeline_config.json` exists
  - Verify: `Stage_0_Correlogram/dataset_profile.json` exists
  - Verify: at least one `*_correlogram.png` exists
  - Record: correlogram-recommended block size vs. user-set 300m
- [ ] Run Stage 1: `sparc run -p my_project/project.yml -s 1 --fast`
  - Verify: exit code 0
  - Verify: `gwen_results.json` exists
  - Verify: `selected_features.txt` exists
  - Verify: `gwen_variable_importance.csv` exists
  - Review: which predictors were selected/dropped
- [ ] Run Stage 2: `sparc run -p my_project/project.yml -s 2 --fast`
  - Verify: exit code 0
  - Verify: `Stage_2_Spatial_CV/optimized_oof_predictions.csv` exists
  - Verify: `Stage_2_Spatial_CV/optimized_meta_model.txt` exists
  - Verify: `Stage_2_Spatial_CV/feature_scaler.pkl` exists
  - Verify: at least one model `.pkl` exists
  - Review: stdout fold counts (5 folds × 4 models)
- [ ] Run Stage 3: `sparc run -p my_project/project.yml -s 3 --fast`
  - Verify: exit code 0
  - Verify: `Stage_3_Causal_Validation/scenario_coefficients.json` exists
  - Verify: `Stage_3_Causal_Validation/causal_validation_summary.txt` exists
- [ ] Run Stage 4: `sparc run -p my_project/project.yml -s 4 --fast`
  - Verify: exit code 0
  - Verify: `Stage_4_Scenarios/scenario_summary.csv` exists
  - Verify: spatial output file exists (`scenario_results.gpkg` or similar)
- [ ] Review `artifacts_manifest.json` — confirm all stage artifacts are registered
- [ ] Decide: update `block_size_source: "correlogram"` for real run based on Stage 0 results

## Notes

- Branch: `testing`
- Run from repo root: `/Users/kylewire/Desktop/sparc-resilience`
- Full command prefix: `sparc run -p my_project/project.yml -s <N> --fast`
- Fix any blockers on the `testing` branch; merge to `main` only after all stages green
