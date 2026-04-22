{"timestamp": "2026-04-21T13:13:19.698853+00:00", "type": "log", "message": ">>> Correlogram Analysis", "phase": "Correlogram analysis"}
{"timestamp": "2026-04-21T13:13:48.103126+00:00", "type": "log", "message": "--- Starting Data Loading and Preprocessing ---", "phase": "Loading data"}
{"timestamp": "2026-04-21T13:13:48.916160+00:00", "type": "log", "message": "--- Finished Data Loading (from cache) ---"}
{"timestamp": "2026-04-21T13:13:48.928252+00:00", "type": "checkpoint", "message": "Computing correlogram for AAT_z", "progress_pct": 0, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:13:58.254709+00:00", "type": "checkpoint", "message": "Computing correlogram for Distance_from_water_m", "progress_pct": 14, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:14:07.243455+00:00", "type": "checkpoint", "message": "Computing correlogram for Pct_Impervious", "progress_pct": 29, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:14:16.160202+00:00", "type": "checkpoint", "message": "Computing correlogram for Pct_Canopy", "progress_pct": 43, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:14:25.486818+00:00", "type": "checkpoint", "message": "Computing correlogram for NDVI", "progress_pct": 57, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:14:34.546583+00:00", "type": "checkpoint", "message": "Computing correlogram for Albedo", "progress_pct": 71, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:14:44.002511+00:00", "type": "checkpoint", "message": "Computing correlogram for Elevation_m", "progress_pct": 86, "phase": "Analyzing variable", "checkpoint_id": "correlogram_var", "level": "info"}
{"timestamp": "2026-04-21T13:14:52.690525+00:00", "type": "log", "message": "Generating model configurations..."}
{"timestamp": "2026-04-21T13:14:52.694746+00:00", "type": "checkpoint", "message": "Pipeline configuration saved", "phase": "Pipeline configuration", "checkpoint_id": "config_saved", "level": "info"}
{"timestamp": "2026-04-21T13:14:52.907556+00:00", "type": "log", "message": ">>> GWEN Variable Selection", "phase": "GWEN variable selection"}
{"timestamp": "2026-04-21T13:15:00.125627+00:00", "type": "log", "message": "1. Loading configuration..."}
{"timestamp": "2026-04-21T13:15:00.128718+00:00", "type": "log", "message": "2. Checking prerequisites..."}
{"timestamp": "2026-04-21T13:15:00.129732+00:00", "type": "log", "message": "3. Loading and preparing data..."}
{"timestamp": "2026-04-21T13:15:00.220614+00:00", "type": "checkpoint", "message": "Fitting GWEN model \u2014 evaluating variable stability", "checkpoint_id": "gwen_fitting", "level": "info"}
{"timestamp": "2026-04-21T13:15:02.859939+00:00", "type": "checkpoint", "message": "GWEN model fitted successfully", "checkpoint_id": "gwen_fitted", "level": "success"}
{"timestamp": "2026-04-21T13:15:02.863337+00:00", "type": "log", "message": "7. Generating comprehensive diagnostics..."}
{"timestamp": "2026-04-21T13:15:04.002522+00:00", "type": "log", "message": "8. Saving GWEN results..."}
{"timestamp": "2026-04-21T13:15:04.016390+00:00", "type": "checkpoint", "message": "GWEN variable selection complete", "phase": "GWEN variable selection", "checkpoint_id": "gwen_done", "level": "success"}
{"timestamp": "2026-04-21T13:15:04.231109+00:00", "type": "log", "message": ">>> Enhanced Spatial CV"}
{"timestamp": "2026-04-21T13:15:06.361516+00:00", "type": "log", "message": "=== Running Spatial CV pipeline ==="}
{"timestamp": "2026-04-21T13:15:06.361516+00:00", "type": "log", "message": "=== Loading and Preprocessing Data ===", "phase": "Loading data"}
{"timestamp": "2026-04-21T13:15:06.520679+00:00", "type": "checkpoint", "message": "Generating spatial validation folds", "phase": "Generating spatial folds", "checkpoint_id": "folds_gen", "level": "info"}
{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 1 prepared \u2014 32914 train / 11011 test samples", "fold": 1, "checkpoint_id": "fold_size", "level": "info"}
{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 2 prepared \u2014 32965 train / 10942 test samples", "fold": 2, "checkpoint_id": "fold_size", "level": "info"}
{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 3 prepared \u2014 32957 train / 10926 test samples", "fold": 3, "checkpoint_id": "fold_size", "level": "info"}
{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 4 prepared \u2014 32984 train / 10847 test samples", "fold": 4, "checkpoint_id": "fold_size", "level": "info"}
{"timestamp": "2026-04-21T13:15:09.975100+00:00", "type": "log", "message": "Fold 1/5", "fold": 1}
{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 5 prepared \u2014 32980 train / 10975 test samples", "fold": 5, "checkpoint_id": "fold_size", "level": "info"}{"timestamp": "2026-04-21T13:15:06.786929+00:00", "type": "checkpoint", "message": "Training OLS (1/4)", "phase": "Training OLS", "model": "ols", "model_index": 1, "model_total": 4, "progress_pct": 5, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:09.995536+00:00", "type": "checkpoint", "message": "Training GWR (2/4)", "phase": "Training GWR", "model": "gwr", "model_index": 2, "model_total": 4, "progress_pct": 15, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:52.496194+00:00", "type": "log", "message": "gwr training completed in 42.50s"}
{"timestamp": "2026-04-21T13:15:53.053785+00:00", "type": "checkpoint", "message": "Training GWRF (3/4)", "phase": "Training GWRF", "model": "gwrf", "model_index": 3, "model_total": 4, "progress_pct": 35, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:17:49.616673+00:00", "type": "log", "message": "gwrf training completed in 116.58s"}

{"timestamp": "2026-04-21T13:20:45.656953+00:00", "type": "log", "message": "gwrf completed successfully"}
{"timestamp": "2026-04-21T13:20:45.657957+00:00", "type": "checkpoint", "message": "Training GGPGAM (4/4)", "phase": "Training GGPGAM", "model": "ggpgam", "model_index": 4, "model_total": 4, "progress_pct": 55, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:20:52.633165+00:00", "type": "log", "message": "Fold 2/5", "fold": 2}
{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 5 prepared \u2014 32980 train / 10975 test samples", "fold": 5, "checkpoint_id": "fold_size", "level": "info"}{"timestamp": "2026-04-21T13:15:06.786929+00:00", "type": "checkpoint", "message": "Training OLS (1/4)", "phase": "Training OLS", "model": "ols", "model_index": 1, "model_total": 4, "progress_pct": 5, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:09.995536+00:00", "type": "checkpoint", "message": "Training GWR (2/4)", "phase": "Training GWR", "model": "gwr", "model_index": 2, "model_total": 4, "progress_pct": 15, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:52.496194+00:00", "type": "log", "message": "gwr training completed in 42.50s"}
{"timestamp": "2026-04-21T13:15:53.053785+00:00", "type": "checkpoint", "message": "Training GWRF (3/4)", "phase": "Training GWRF", "model": "gwrf", "model_index": 3, "model_total": 4, "progress_pct": 35, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:17:49.616673+00:00", "type": "log", "message": "gwrf training completed in 116.58s"}

{"timestamp": "2026-04-21T13:20:45.656953+00:00", "type": "log", "message": "gwrf completed successfully"}
{"timestamp": "2026-04-21T13:20:45.657957+00:00", "type": "checkpoint", "message": "Training GGPGAM (4/4)", "phase": "Training GGPGAM", "model": "ggpgam", "model_index": 4, "model_total": 4, "progress_pct": 55, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:20:52.633165+00:00", "type": "log", "message": "Fold 3/5", "fold": 3}{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 5 prepared \u2014 32980 train / 10975 test samples", "fold": 5, "checkpoint_id": "fold_size", "level": "info"}{"timestamp": "2026-04-21T13:15:06.786929+00:00", "type": "checkpoint", "message": "Training OLS (1/4)", "phase": "Training OLS", "model": "ols", "model_index": 1, "model_total": 4, "progress_pct": 5, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:09.995536+00:00", "type": "checkpoint", "message": "Training GWR (2/4)", "phase": "Training GWR", "model": "gwr", "model_index": 2, "model_total": 4, "progress_pct": 15, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:52.496194+00:00", "type": "log", "message": "gwr training completed in 42.50s"}
{"timestamp": "2026-04-21T13:15:53.053785+00:00", "type": "checkpoint", "message": "Training GWRF (3/4)", "phase": "Training GWRF", "model": "gwrf", "model_index": 3, "model_total": 4, "progress_pct": 35, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:17:49.616673+00:00", "type": "log", "message": "gwrf training completed in 116.58s"}

{"timestamp": "2026-04-21T13:20:45.656953+00:00", "type": "log", "message": "gwrf completed successfully"}
{"timestamp": "2026-04-21T13:20:45.657957+00:00", "type": "checkpoint", "message": "Training GGPGAM (4/4)", "phase": "Training GGPGAM", "model": "ggpgam", "model_index": 4, "model_total": 4, "progress_pct": 55, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:20:52.633165+00:00", "type": "log", "message": "Fold 4/5", "fold": 4}{"timestamp": "2026-04-21T13:15:06.781635+00:00", "type": "checkpoint", "message": "Fold 5 prepared \u2014 32980 train / 10975 test samples", "fold": 5, "checkpoint_id": "fold_size", "level": "info"}{"timestamp": "2026-04-21T13:15:06.786929+00:00", "type": "checkpoint", "message": "Training OLS (1/4)", "phase": "Training OLS", "model": "ols", "model_index": 1, "model_total": 4, "progress_pct": 5, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:09.995536+00:00", "type": "checkpoint", "message": "Training GWR (2/4)", "phase": "Training GWR", "model": "gwr", "model_index": 2, "model_total": 4, "progress_pct": 15, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:15:52.496194+00:00", "type": "log", "message": "gwr training completed in 42.50s"}
{"timestamp": "2026-04-21T13:15:53.053785+00:00", "type": "checkpoint", "message": "Training GWRF (3/4)", "phase": "Training GWRF", "model": "gwrf", "model_index": 3, "model_total": 4, "progress_pct": 35, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:17:49.616673+00:00", "type": "log", "message": "gwrf training completed in 116.58s"}

{"timestamp": "2026-04-21T13:20:45.656953+00:00", "type": "log", "message": "gwrf completed successfully"}
{"timestamp": "2026-04-21T13:20:45.657957+00:00", "type": "checkpoint", "message": "Training GGPGAM (4/4)", "phase": "Training GGPGAM", "model": "ggpgam", "model_index": 4, "model_total": 4, "progress_pct": 55, "checkpoint_id": "model_start", "level": "info"}
{"timestamp": "2026-04-21T13:20:52.633165+00:00", "type": "log", "message": "Fold 5/5", "fold": 5}
{"timestamp": "2026-04-21T13:47:56.358232+00:00", "type": "checkpoint", "message": "Computing out-of-fold performance metrics", "checkpoint_id": "oof_perf", "level": "info"}
{"timestamp": "2026-04-21T13:47:56.360248+00:00", "type": "metric", "message": "OLS: R\u00b2 = 0.2942, RMSE = 1.4375", "metric": "rmse", "value": 1.4375}
{"timestamp": "2026-04-21T13:47:56.361527+00:00", "type": "metric", "message": "GWR: R\u00b2 = 0.8282, RMSE = 0.7092", "metric": "rmse", "value": 0.7092}
{"timestamp": "2026-04-21T13:47:56.363537+00:00", "type": "metric", "message": "GWRF: R\u00b2 = 0.9141, RMSE = 0.5014", "metric": "rmse", "value": 0.5014}
{"timestamp": "2026-04-21T13:47:56.521710+00:00", "type": "metric", "message": "GGPGAM: R\u00b2 = 0.7459, RMSE = 0.8626", "metric": "rmse", "value": 0.8626}
{"timestamp": "2026-04-21T13:47:56.700969+00:00", "type": "checkpoint", "message": "Retraining models on full dataset", "phase": "Retraining base models", "checkpoint_id": "retrain_start", "level": "info"}
{"timestamp": "2026-04-21T13:47:56.752596+00:00", "type": "log", "message": "--- Training GWR on full dataset ---"}
{"timestamp": "2026-04-21T13:49:19.106913+00:00", "type": "checkpoint", "message": "Model saved: gwr_model_full.pkl", "checkpoint_id": "model_saved", "level": "success"}
{"timestamp": "2026-04-21T13:49:19.122203+00:00", "type": "log", "message": "--- Training GWRF on full dataset ---"}
{"timestamp": "2026-04-21T13:52:34.765240+00:00", "type": "checkpoint", "message": "Model saved: gwrf_model_full.pkl", "checkpoint_id": "model_saved", "level": "success"}
{"timestamp": "2026-04-21T13:52:34.765240+00:00", "type": "log", "message": "--- Extracting GWRF condition curves (PDP + saturation) ---"}{"timestamp": "2026-04-21T14:31:21.032615+00:00", "type": "log", "message": "--- Training GGPGAM on full dataset ---"}
{"timestamp": "2026-04-21T14:31:29.692805+00:00", "type": "checkpoint", "message": "Model saved: ggpgam_model_full.pkl", "checkpoint_id": "model_saved", "level": "success"}
{"timestamp": "2026-04-21T15:01:06.348009+00:00", "type": "log", "message": "=== Neural Meta-Learner ===", "phase": "Neural meta-learner"}
{"timestamp": "2026-04-21T15:01:07.143694+00:00", "type": "log", "message": "Fold 1 / 5  (32914 train, 11011 test)", "fold": 1}
{"timestamp": "2026-04-21T15:02:37.804125+00:00", "type": "log", "message": "All surrogates passed validation (R\u00b2 >= 0.85)"}
{"timestamp": "2026-04-21T15:02:37.899199+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage A: Representation Warmup", "curriculum": "Stage A", "label": "Representation Warmup"}
{"timestamp": "2026-04-21T15:04:02.332154+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage B: Physics Activation", "curriculum": "Stage B", "label": "Physics Activation"}
{"timestamp": "2026-04-21T15:06:54.365988+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage C: Joint Optimization", "curriculum": "Stage C", "label": "Joint Optimization"}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "epoch_update", "message": "Epoch 100/100  loss=0.1403  [mse=0.013 phys=0.010 nbr=0.002 ce=0.000 pde=0.078 bc=0.001 prior=0.000 base=0.039]  (964.0s)", "epoch": 100, "n_epochs": 100, "total_loss": 0.1403, "train_phase": "cv", "components": {"mse": 0.013, "phys": 0.01, "nbr": 0.002, "ce": 0.0, "pde": 0.078, "bc": 0.001, "prior": 0.0, "base": 0.039}, "eta_seconds": 0.0, "elapsed_seconds": 864.6}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "convergence", "message": "[CONVERGENCE] converged", "status": "converged"}
{"timestamp": "2026-04-21T15:17:11.173193+00:00", "type": "log", "message": "Fold 1 training done in 964.0s", "fold": 1}
{"timestamp": "2026-04-21T15:19:34.808016+00:00", "type": "log", "message": "Fold 2 / 5  (32965 train, 10942 test)", "fold": 2}
{"timestamp": "2026-04-21T15:02:37.804125+00:00", "type": "log", "message": "All surrogates passed validation (R\u00b2 >= 0.85)"}
{"timestamp": "2026-04-21T15:02:37.899199+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage A: Representation Warmup", "curriculum": "Stage A", "label": "Representation Warmup"}
{"timestamp": "2026-04-21T15:04:02.332154+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage B: Physics Activation", "curriculum": "Stage B", "label": "Physics Activation"}
{"timestamp": "2026-04-21T15:06:54.365988+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage C: Joint Optimization", "curriculum": "Stage C", "label": "Joint Optimization"}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "epoch_update", "message": "Epoch 100/100  loss=0.1403  [mse=0.013 phys=0.010 nbr=0.002 ce=0.000 pde=0.078 bc=0.001 prior=0.000 base=0.039]  (964.0s)", "epoch": 100, "n_epochs": 100, "total_loss": 0.1403, "train_phase": "cv", "components": {"mse": 0.013, "phys": 0.01, "nbr": 0.002, "ce": 0.0, "pde": 0.078, "bc": 0.001, "prior": 0.0, "base": 0.039}, "eta_seconds": 0.0, "elapsed_seconds": 864.6}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "convergence", "message": "[CONVERGENCE] converged", "status": "converged"}
{"timestamp": "2026-04-21T15:17:11.173193+00:00", "type": "log", "message": "Fold 2 training done in 964.0s", "fold": 1}
{"timestamp": "2026-04-21T15:19:34.808016+00:00", "type": "log", "message": "Fold 3 / 5  (32965 train, 10942 test)", "fold": 3}
{"timestamp": "2026-04-21T15:02:37.804125+00:00", "type": "log", "message": "All surrogates passed validation (R\u00b2 >= 0.85)"}
{"timestamp": "2026-04-21T15:02:37.899199+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage A: Representation Warmup", "curriculum": "Stage A", "label": "Representation Warmup"}
{"timestamp": "2026-04-21T15:04:02.332154+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage B: Physics Activation", "curriculum": "Stage B", "label": "Physics Activation"}
{"timestamp": "2026-04-21T15:06:54.365988+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage C: Joint Optimization", "curriculum": "Stage C", "label": "Joint Optimization"}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "epoch_update", "message": "Epoch 100/100  loss=0.1403  [mse=0.013 phys=0.010 nbr=0.002 ce=0.000 pde=0.078 bc=0.001 prior=0.000 base=0.039]  (964.0s)", "epoch": 100, "n_epochs": 100, "total_loss": 0.1403, "train_phase": "cv", "components": {"mse": 0.013, "phys": 0.01, "nbr": 0.002, "ce": 0.0, "pde": 0.078, "bc": 0.001, "prior": 0.0, "base": 0.039}, "eta_seconds": 0.0, "elapsed_seconds": 864.6}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "convergence", "message": "[CONVERGENCE] converged", "status": "converged"}
{"timestamp": "2026-04-21T15:17:11.173193+00:00", "type": "log", "message": "Fold 3 training done in 964.0s", "fold": 1}
{"timestamp": "2026-04-21T15:19:34.808016+00:00", "type": "log", "message": "Fold 4 / 5  (32965 train, 10942 test)", "fold": 4}
{"timestamp": "2026-04-21T15:02:37.804125+00:00", "type": "log", "message": "All surrogates passed validation (R\u00b2 >= 0.85)"}
{"timestamp": "2026-04-21T15:02:37.899199+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage A: Representation Warmup", "curriculum": "Stage A", "label": "Representation Warmup"}
{"timestamp": "2026-04-21T15:04:02.332154+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage B: Physics Activation", "curriculum": "Stage B", "label": "Physics Activation"}
{"timestamp": "2026-04-21T15:06:54.365988+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage C: Joint Optimization", "curriculum": "Stage C", "label": "Joint Optimization"}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "epoch_update", "message": "Epoch 100/100  loss=0.1403  [mse=0.013 phys=0.010 nbr=0.002 ce=0.000 pde=0.078 bc=0.001 prior=0.000 base=0.039]  (964.0s)", "epoch": 100, "n_epochs": 100, "total_loss": 0.1403, "train_phase": "cv", "components": {"mse": 0.013, "phys": 0.01, "nbr": 0.002, "ce": 0.0, "pde": 0.078, "bc": 0.001, "prior": 0.0, "base": 0.039}, "eta_seconds": 0.0, "elapsed_seconds": 864.6}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "convergence", "message": "[CONVERGENCE] converged", "status": "converged"}
{"timestamp": "2026-04-21T15:17:11.173193+00:00", "type": "log", "message": "Fold 4 training done in 964.0s", "fold": 1}
{"timestamp": "2026-04-21T15:19:34.808016+00:00", "type": "log", "message": "Fold 5 / 5  (32965 train, 10942 test)", "fold": 5}
{"timestamp": "2026-04-21T15:02:37.804125+00:00", "type": "log", "message": "All surrogates passed validation (R\u00b2 >= 0.85)"}
{"timestamp": "2026-04-21T15:02:37.899199+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage A: Representation Warmup", "curriculum": "Stage A", "label": "Representation Warmup"}
{"timestamp": "2026-04-21T15:04:02.332154+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage B: Physics Activation", "curriculum": "Stage B", "label": "Physics Activation"}
{"timestamp": "2026-04-21T15:06:54.365988+00:00", "type": "curriculum_stage", "message": "[CURRICULUM] Stage C: Joint Optimization", "curriculum": "Stage C", "label": "Joint Optimization"}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "epoch_update", "message": "Epoch 100/100  loss=0.1403  [mse=0.013 phys=0.010 nbr=0.002 ce=0.000 pde=0.078 bc=0.001 prior=0.000 base=0.039]  (964.0s)", "epoch": 100, "n_epochs": 100, "total_loss": 0.1403, "train_phase": "cv", "components": {"mse": 0.013, "phys": 0.01, "nbr": 0.002, "ce": 0.0, "pde": 0.078, "bc": 0.001, "prior": 0.0, "base": 0.039}, "eta_seconds": 0.0, "elapsed_seconds": 864.6}
{"timestamp": "2026-04-21T15:17:11.172197+00:00", "type": "convergence", "message": "[CONVERGENCE] converged", "status": "converged"}
{"timestamp": "2026-04-21T15:17:11.173193+00:00", "type": "log", "message": "Fold 5 training done in 964.0s", "fold": 1}
{"timestamp": "2026-04-21T16:35:29.304108+00:00", "type": "metric", "message": "V2 Neural Meta OOF  R\u00b2=0.9447  RMSE=0.4023", "metric": "rmse", "value": 0.4023}

{"timestamp": "2026-04-21T16:38:34.863518+00:00", "type": "log", "message": "All full-retrain surrogates passed validation (R\u00b2 >= 0.85)"}
{"timestamp": "2026-04-21T17:05:25.239020+00:00", "type": "log", "message": "Starting SWA phase (20 epochs)..."}
{"timestamp": "2026-04-21T17:10:37.579814+00:00", "type": "log", "message": "SWA complete \u2014 averaged weights extracted."}
{"timestamp": "2026-04-21T17:18:07.085207+00:00", "type": "log", "message": "Stage 3: Causal Validation", "stage": 3, "phase": "Causal validation"}
{"timestamp": "2026-04-21T17:18:08.180871+00:00", "type": "log", "message": "--- V2 Bayesian Causal Analysis (MC\u00b3 + NUTS) ---"}
{"timestamp": "2026-04-21T17:18:14.134720+00:00", "type": "log", "message": "MC\u00b3 iter 1/10000  score=-1998272.28  edges=1  accepted=1"}
{"timestamp": "2026-04-21T17:18:17.712261+00:00", "type": "log", "message": "MC\u00b3 iter 5000/10000  score=-1936781.48  edges=20  accepted=153"}
{"timestamp": "2026-04-21T17:18:21.785415+00:00", "type": "log", "message": "[DAG_GATE] MC\u00b3 complete \u2014 20 edges above 0.50.  Awaiting user approval..."}
{"timestamp": "2026-04-21T17:18:21.785415+00:00", "type": "dag_approval_requested", "message": "[DAG_APPROVAL_REQUESTED] {\"n_edges\": 20, \"n_nodes\": 7}", "n_edges": 20, "n_nodes": 7}
{"timestamp": "2026-04-21T17:20:36.032802+00:00", "type": "log", "message": "NUTS: finding initial step size (200 adaptation steps, \u03b5\u2080=0.0156)..."}
{"timestamp": "2026-04-21T17:20:36.033303+00:00", "type": "checkpoint", "message": "NUTS warmup \u2014 800 transitions", "checkpoint_id": "nuts_warmup", "level": "info"}{"timestamp": "2026-04-21T17:20:59.533217+00:00", "type": "log", "message": "warmup 500 / 800"}
{"timestamp": "2026-04-21T17:21:15.702063+00:00", "type": "log", "message": "warmup 800 / 800"}
{"timestamp": "2026-04-21T17:21:16.252253+00:00", "type": "checkpoint", "message": "NUTS sampling \u2014 drawing 15000 posterior samples", "checkpoint_id": "nuts_sampling", "level": "info"}
{"timestamp": "2026-04-21T17:32:04.187145+00:00", "type": "log", "message": "NUTS sample 2500 / 15000  (divergences: 0, accept: 88.4%)", "progress_pct": 4}
{"timestamp": "2026-04-21T17:43:49.534466+00:00", "type": "log", "message": "NUTS sample 5000 / 15000  (divergences: 0, accept: 88.9%)", "progress_pct": 9}
{"timestamp": "2026-04-21T17:55:43.018362+00:00", "type": "log", "message": "NUTS sample 7500 / 15000  (divergences: 0, accept: 88.9%)", "progress_pct": 9}
{"timestamp": "2026-04-21T18:07:17.536668+00:00", "type": "log", "message": "NUTS sample 10000 / 15000  (divergences: 0, accept: 89.0%)", "progress_pct": 0}
