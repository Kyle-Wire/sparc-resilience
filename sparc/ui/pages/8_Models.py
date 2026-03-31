"""Page 8 — Model Hyperparameters & Pipeline Settings (Advanced)."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Models & Pipeline Settings")
st.markdown("Fine-tune model hyperparameters, GWEN variable selection, and pipeline flags.")

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Globals
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Pipeline Settings")
c1, c2, c3 = st.columns(3)
st.session_state["random_seed"] = c1.number_input(
    "Random seed", value=int(st.session_state["random_seed"]), min_value=0,
)
st.session_state["n_spatial_folds"] = c2.number_input(
    "Spatial folds", value=int(st.session_state["n_spatial_folds"]), min_value=2, max_value=20,
)
st.session_state["fast_mode"] = c3.checkbox("Fast mode", value=st.session_state["fast_mode"])

c4, c5, c6 = st.columns(3)
st.session_state["run_mc_uncertainty"] = c4.checkbox(
    "Monte-Carlo uncertainty", value=st.session_state["run_mc_uncertainty"],
)
st.session_state["n_mc_draws"] = c5.number_input(
    "MC draws", value=int(st.session_state["n_mc_draws"]), min_value=10, max_value=500,
)
st.session_state["overwrite_outputs"] = c6.checkbox(
    "Overwrite outputs", value=st.session_state["overwrite_outputs"],
)

# ── Flags ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## Feature Flags")
c1, c2, c3 = st.columns(3)
st.session_state["use_gwen_selection"] = c1.checkbox(
    "Use GWEN variable selection", value=st.session_state["use_gwen_selection"],
)
st.session_state["include_gwrf_in_ensemble"] = c2.checkbox(
    "Include GWRF in ensemble", value=st.session_state["include_gwrf_in_ensemble"],
)
st.session_state["use_laplacian_eigenmaps_in_ols"] = c3.checkbox(
    "Laplacian eigenmaps in OLS", value=st.session_state["use_laplacian_eigenmaps_in_ols"],
)

# ══════════════════════════════════════════════════════════════════════════════
# Model Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## Model Hyperparameters")

# ── GWR ───────────────────────────────────────────────────────────────────────
with st.expander("GWR — Geographically Weighted Regression"):
    c1, c2 = st.columns(2)
    bw = c1.text_input(
        "Bandwidth (blank = auto)",
        value="" if st.session_state["gwr_bandwidth"] is None else str(st.session_state["gwr_bandwidth"]),
    )
    st.session_state["gwr_bandwidth"] = float(bw) if bw.strip() else None
    st.session_state["gwr_kernel"] = c2.selectbox(
        "Kernel", ["gaussian", "bisquare", "exponential"],
        index=["gaussian", "bisquare", "exponential"].index(st.session_state["gwr_kernel"]),
    )
    c3, c4 = st.columns(2)
    st.session_state["gwr_alpha"] = c3.number_input(
        "Alpha (regularization)", value=float(st.session_state["gwr_alpha"]),
        min_value=0.001, max_value=1.0, step=0.01,
    )
    st.session_state["gwr_min_points"] = c4.number_input(
        "Min points", value=int(st.session_state["gwr_min_points"]), min_value=10,
    )

# ── GWRF ──────────────────────────────────────────────────────────────────────
with st.expander("GWRF — Geographically Weighted Random Forest"):
    c1, c2 = st.columns(2)
    st.session_state["gwrf_n_estimators"] = c1.number_input(
        "Trees", value=int(st.session_state["gwrf_n_estimators"]), min_value=10, step=10,
    )
    st.session_state["gwrf_k_neighbors"] = c2.number_input(
        "K neighbors", value=int(st.session_state["gwrf_k_neighbors"]), min_value=10, step=10,
    )
    c3, c4 = st.columns(2)
    st.session_state["gwrf_min_samples_leaf"] = c3.number_input(
        "Min samples leaf", value=int(st.session_state["gwrf_min_samples_leaf"]), min_value=1,
    )
    st.session_state["gwrf_n_jobs"] = c4.number_input(
        "N jobs", value=int(st.session_state["gwrf_n_jobs"]), min_value=1, max_value=32,
    )

# ── GGPGAM ────────────────────────────────────────────────────────────────────
with st.expander("GGPGAM — Gaussian Process GAM"):
    c1, c2 = st.columns(2)
    st.session_state["ggpgam_n_splines"] = c1.number_input(
        "N splines", value=int(st.session_state["ggpgam_n_splines"]), min_value=2,
    )
    st.session_state["ggpgam_n_spatial_bases"] = c2.number_input(
        "N spatial bases", value=int(st.session_state["ggpgam_n_spatial_bases"]), min_value=2,
    )
    c3, c4 = st.columns(2)
    st.session_state["ggpgam_lam"] = c3.number_input(
        "Lambda", value=float(st.session_state["ggpgam_lam"]),
        min_value=0.01, max_value=10.0, step=0.1,
    )
    st.session_state["ggpgam_max_iter"] = c4.number_input(
        "Max iterations", value=int(st.session_state["ggpgam_max_iter"]), min_value=10,
    )

# ── Meta-Ensemble ─────────────────────────────────────────────────────────────
with st.expander("Meta-Ensemble — LightGBM Stacking"):
    c1, c2 = st.columns(2)
    st.session_state["meta_algorithm"] = c1.selectbox(
        "Algorithm", ["lightgbm", "xgboost", "linear"],
        index=["lightgbm", "xgboost", "linear"].index(st.session_state["meta_algorithm"]),
    )
    st.session_state["meta_n_optuna_trials"] = c2.number_input(
        "Optuna trials", value=int(st.session_state["meta_n_optuna_trials"]),
        min_value=5, max_value=200,
    )
    c3, c4 = st.columns(2)
    st.session_state["meta_include_base_features"] = c3.checkbox(
        "Include base features", value=st.session_state["meta_include_base_features"],
    )
    st.session_state["meta_include_laplacian_pca"] = c4.checkbox(
        "Include Laplacian PCA", value=st.session_state["meta_include_laplacian_pca"],
    )

# ── Deep Kriging ──────────────────────────────────────────────────────────────
with st.expander("Deep Kriging — Neural Residuals"):
    layers_str = st.text_input(
        "Hidden layers (comma-separated)",
        value=", ".join(str(x) for x in st.session_state["dk_hidden_layers"]),
    )
    st.session_state["dk_hidden_layers"] = [
        int(x.strip()) for x in layers_str.split(",") if x.strip().isdigit()
    ] or [64, 32, 16]

    c1, c2 = st.columns(2)
    st.session_state["dk_dropout_rate"] = c1.number_input(
        "Dropout rate", value=float(st.session_state["dk_dropout_rate"]),
        min_value=0.0, max_value=0.9, step=0.05,
    )
    st.session_state["dk_epochs"] = c2.number_input(
        "Epochs", value=int(st.session_state["dk_epochs"]), min_value=10,
    )
    c3, c4 = st.columns(2)
    st.session_state["dk_learning_rate"] = c3.number_input(
        "Learning rate", value=float(st.session_state["dk_learning_rate"]),
        min_value=0.0001, max_value=0.1, step=0.0001, format="%.4f",
    )
    st.session_state["dk_batch_size"] = c4.number_input(
        "Batch size", value=int(st.session_state["dk_batch_size"]), min_value=16, step=16,
    )

# ── Spatial CV ────────────────────────────────────────────────────────────────
with st.expander("Spatial CV — Cross-Validation Strategy"):
    c1, c2 = st.columns(2)
    st.session_state["cv_block_size"] = c1.number_input(
        "Block size (m)", value=int(st.session_state["cv_block_size"]), min_value=50,
    )
    st.session_state["cv_buffer_size"] = c2.number_input(
        "Buffer size (m)", value=int(st.session_state["cv_buffer_size"]), min_value=0,
    )
    c3, c4 = st.columns(2)
    st.session_state["cv_block_size_source"] = c3.selectbox(
        "Block size source", ["user", "correlogram"],
        index=["user", "correlogram"].index(st.session_state["cv_block_size_source"]),
    )
    st.session_state["cv_method"] = c4.selectbox(
        "Method", ["block", "buffer"],
        index=["block", "buffer"].index(st.session_state["cv_method"]),
    )
    st.session_state["cv_stratify_y"] = st.checkbox(
        "Stratify by response", value=st.session_state["cv_stratify_y"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# GWEN & Laplacian
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## GWEN Variable Selection")
c1, c2, c3 = st.columns(3)
st.session_state["gwen_sample_size"] = c1.number_input(
    "Sample size", value=int(st.session_state["gwen_sample_size"]), min_value=100,
)
st.session_state["gwen_k_neighbors"] = c2.number_input(
    "K neighbors", value=int(st.session_state["gwen_k_neighbors"]), min_value=10,
)
st.session_state["gwen_cv_folds"] = c3.number_input(
    "CV folds", value=int(st.session_state["gwen_cv_folds"]), min_value=2, max_value=20,
)

st.markdown("## Laplacian Eigenmaps")
c1, c2 = st.columns(2)
st.session_state["laplacian_n_eigenmaps"] = c1.number_input(
    "N eigenmaps", value=int(st.session_state["laplacian_n_eigenmaps"]), min_value=10,
)
st.session_state["laplacian_k_for_swm"] = c2.number_input(
    "K for SWM", value=int(st.session_state["laplacian_k_for_swm"]), min_value=2,
)
