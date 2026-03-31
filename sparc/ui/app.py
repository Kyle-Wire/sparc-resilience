"""
SPARC — Streamlit Frontend
===========================
Multi-page application entry point.
Launch with:  streamlit run sparc/ui/app.py
"""
import streamlit as st
from pathlib import Path
import sys

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sparc.ui.state import init_state
from sparc.ui.theme import inject_css, render_sidebar_logo, LOGO_PATH

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SPARC — Spatial Analysis & Research Core",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise shared state & brand styling ───────────────────────────────────
init_state()
inject_css()
render_sidebar_logo()

# ── All pages ─────────────────────────────────────────────────────────────────
PAGES = {
    "1 — Project Setup":  "1_Project_Setup.py",
    "2 — Data":           "2_Data.py",
    "3 — Variables":      "3_Variables.py",
    "4 — CRS":            "4_CRS.py",
    "5 — DAG Builder":    "5_DAG_Builder.py",
    "6 — Physics":        "6_Physics.py",
    "7 — Scenarios":      "7_Scenarios.py",
    "8 — Models":         "8_Models.py",
    "9 — Run Pipeline":   "9_Run_Pipeline.py",
    "10 — Results":       "10_Results.py",
}

_LABELS = list(PAGES.keys())

# ── Navigation ────────────────────────────────────────────────────────────────
# Keep track of selected index so every page is always reachable.
if "_nav_index" not in st.session_state:
    st.session_state["_nav_index"] = 0

with st.sidebar:
    # Section header: Configure
    st.markdown(
        "<p class='nav-section'>CONFIGURE</p>",
        unsafe_allow_html=True,
    )
    for i, label in enumerate(_LABELS[:5]):
        if st.button(label, key=f"nav_{i}", use_container_width=True):
            st.session_state["_nav_index"] = i
            st.rerun()

    # Section header: Analyze & Run
    st.markdown(
        "<p class='nav-section' style='margin-top:1rem;'>ANALYZE &amp; RUN</p>",
        unsafe_allow_html=True,
    )
    for i, label in enumerate(_LABELS[5:], start=5):
        if st.button(label, key=f"nav_{i}", use_container_width=True):
            st.session_state["_nav_index"] = i
            st.rerun()

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p class='nav-section'>STATUS</p>",
        unsafe_allow_html=True,
    )

    def _check(key, label):
        val = st.session_state.get(key, "")
        ok = bool(val) if not isinstance(val, list) else len(val) > 0
        icon = "✓" if ok else "○"
        color = "#16a34a" if ok else "#94a3b8"
        st.markdown(
            f"<span style='font-family:monospace; font-size:0.8rem; color:{color};'>"
            f"{icon}</span>&ensp;<span style='font-size:0.82rem; color:#334155;'>{label}</span>",
            unsafe_allow_html=True,
        )

    _check("project_name",   "Project name")
    _check("data_file_path", "Data file")
    _check("target_column",  "Target variable")
    _check("predictors",     "Predictors")
    _check("crs_input",      "CRS defined")
    _check("dag_nodes",      "DAG configured")
    _check("scenarios",      "Scenarios defined")

# ── Route to selected page ────────────────────────────────────────────────────
selected_label = _LABELS[st.session_state["_nav_index"]]
page_file = Path(__file__).parent / "pages" / PAGES[selected_label]

if page_file.exists():
    with open(page_file, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, str(page_file), "exec"), {"__name__": "__page__", "__file__": str(page_file)})
else:
    st.info(f"Page **{selected_label}** is not yet implemented.")
