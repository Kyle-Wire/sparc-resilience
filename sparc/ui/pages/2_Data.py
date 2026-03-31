"""Page 2 — Data Selection & Preview."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Data")
st.markdown("Point to your CSV dataset and map the key columns.")

# ── File path input ───────────────────────────────────────────────────────────
st.markdown("## Data Source")

tab_path, tab_upload = st.tabs(["File Path (local)", "Upload CSV"])

with tab_path:
    _raw_csv = st.text_input(
        "CSV File Path",
        value=st.session_state["data_file_path"],
        help="Absolute or relative path to your input CSV.",
    )
    st.session_state["data_file_path"] = _raw_csv.strip().strip('"').strip("'") if _raw_csv else ""

with tab_upload:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        wd = st.session_state.get("working_dir", "").strip().strip('"').strip("'")
        if not wd:
            st.warning("Set a **Working Directory** on the Project Setup page first.")
        else:
            dest = Path(wd) / uploaded.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getvalue())
            st.session_state["data_file_path"] = str(dest)
            st.success(f"Saved to `{dest}`")

# ── Load & preview ────────────────────────────────────────────────────────────
csv_path = st.session_state["data_file_path"]
columns: list[str] = []

if csv_path:
    # Resolve relative paths against working_dir
    p = Path(csv_path)
    if not p.is_absolute():
        wd = st.session_state.get("working_dir", "")
        if wd:
            p = Path(wd) / p

    if p.exists():
        import pandas as pd

        try:
            df_preview = pd.read_csv(str(p), nrows=200)
            columns = list(df_preview.columns)
            st.session_state["_csv_columns"] = columns

            st.markdown("## Data Preview")
            st.dataframe(df_preview.head(10), use_container_width=True, height=300)

            # Quick stats
            st.markdown("## Column Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows (preview)", len(df_preview))
            col2.metric("Columns", len(columns))
            col3.metric("Numeric Columns", df_preview.select_dtypes("number").shape[1])

            with st.expander("Column details"):
                stats = df_preview.describe(include="all").T
                st.dataframe(stats, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.warning(f"File not found: `{p}`")
else:
    columns = st.session_state.get("_csv_columns", [])

# ── Column mapping ────────────────────────────────────────────────────────────
if columns:
    st.divider()
    st.markdown("## Column Mapping")

    def _idx(val, options):
        try:
            return options.index(val)
        except ValueError:
            return 0

    opts = [""] + columns

    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state["target_column"] = st.selectbox(
            "Target Variable *",
            opts,
            index=_idx(st.session_state["target_column"], opts),
        )
        st.session_state["coord_x"] = st.selectbox(
            "X / Easting Column *",
            opts,
            index=_idx(st.session_state["coord_x"], opts),
        )
    with col_b:
        st.session_state["identifier_column"] = st.selectbox(
            "Identifier Column",
            opts,
            index=_idx(st.session_state["identifier_column"], opts),
        )
        st.session_state["coord_y"] = st.selectbox(
            "Y / Northing Column *",
            opts,
            index=_idx(st.session_state["coord_y"], opts),
        )

    st.session_state["areas_of_interest_file"] = st.text_input(
        "Areas of Interest File (optional)",
        value=st.session_state["areas_of_interest_file"],
        help="CSV mapping locations to named zones/districts.",
    )
