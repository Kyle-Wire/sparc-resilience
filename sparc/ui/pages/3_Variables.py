"""Page 3 — Variable Selection & Constraints."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Variables")
st.markdown("Select predictor variables, assign monotone constraints, and classify variables as actionable or fixed.")

# ── Available columns ─────────────────────────────────────────────────────────
columns = st.session_state.get("_csv_columns", [])
target   = st.session_state.get("target_column", "")
id_col   = st.session_state.get("identifier_column", "")
cx       = st.session_state.get("coord_x", "")
cy       = st.session_state.get("coord_y", "")

reserved = {target, id_col, cx, cy} - {""}
available = [c for c in columns if c not in reserved]

if not available:
    st.info("Load a data file on the **Data** page first, or type predictor names manually below.")

# ── Predictor selection ───────────────────────────────────────────────────────
st.markdown("## Predictor Selection")

if available:
    current = [p for p in st.session_state["predictors"] if p in available]
    selected = st.multiselect(
        "Select predictors from your data columns",
        options=available,
        default=current,
    )
    st.session_state["predictors"] = selected
else:
    manual = st.text_area(
        "Enter predictor names (one per line)",
        value="\n".join(st.session_state["predictors"]),
        height=150,
    )
    st.session_state["predictors"] = [v.strip() for v in manual.split("\n") if v.strip()]

predictors = st.session_state["predictors"]

if not predictors:
    st.stop()

# ── Monotone constraints ─────────────────────────────────────────────────────
st.divider()
st.markdown("## Monotone Constraints")
st.markdown(
    "Specify the expected direction of each variable's relationship with the target. "
    "**-1** = more → lower response (e.g. cooling), "
    "**+1** = more → higher response (e.g. warming), "
    "**0** = unconstrained."
)

constraints = dict(st.session_state.get("monotone_constraints", {}))

cols_per_row = 3
rows = [predictors[i:i + cols_per_row] for i in range(0, len(predictors), cols_per_row)]

for row_vars in rows:
    cols = st.columns(len(row_vars))
    for col, var in zip(cols, row_vars):
        with col:
            current_val = constraints.get(var, 0)
            constraints[var] = st.selectbox(
                var,
                options=[-1, 0, 1],
                index=[-1, 0, 1].index(current_val),
                key=f"mono_{var}",
            )

st.session_state["monotone_constraints"] = constraints

# ── Actionable vs Fixed ──────────────────────────────────────────────────────
st.divider()
st.markdown("## Actionable vs Fixed Variables")
st.markdown(
    "**Actionable** variables can be changed in scenario simulations (e.g. tree canopy %). "
    "**Fixed** variables cannot be modified (e.g. elevation)."
)

current_act = [v for v in st.session_state["actionable_variables"] if v in predictors]
current_fix = [v for v in st.session_state["fixed_variables"] if v in predictors]

col_a, col_b = st.columns(2)
with col_a:
    st.session_state["actionable_variables"] = st.multiselect(
        "Actionable variables",
        options=predictors,
        default=current_act,
    )
with col_b:
    remaining = [v for v in predictors if v not in st.session_state["actionable_variables"]]
    st.session_state["fixed_variables"] = st.multiselect(
        "Fixed variables",
        options=remaining,
        default=[v for v in current_fix if v in remaining],
    )

# ── Correlation heatmap ──────────────────────────────────────────────────────
st.divider()
st.markdown("## Correlation Preview")

csv_path = st.session_state.get("data_file_path", "")
if csv_path and predictors:
    p = Path(csv_path)
    if not p.is_absolute():
        wd = st.session_state.get("working_dir", "")
        if wd:
            p = Path(wd) / p

    if p.exists():
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        from sparc.ui.theme import get_matplotlib_style, SPECTRAL

        matplotlib.rcParams.update(get_matplotlib_style())

        try:
            df = pd.read_csv(str(p), usecols=[c for c in predictors + [target] if c])
            corr = df.corr(numeric_only=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("sparc", SPECTRAL)
            im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(corr.columns, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Predictor Correlation Matrix")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")
    else:
        st.info("Data file not found — heatmap will be available once data is loaded.")
else:
    st.info("Load data and select predictors to see the correlation heatmap.")
