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
from sparc.ui.theme import inject_css, render_sidebar_logo, LOGO_PATH, PRIMARY, GRAY_MID

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

# ── All pages (label, filename) ───────────────────────────────────────────────
PAGES = [
    ("Project Setup",  "1_Project_Setup.py"),
    ("Data",           "2_Data.py"),
    ("Variables",      "3_Variables.py"),
    ("Coordinates",    "4_CRS.py"),
    ("DAG Builder",    "5_DAG_Builder.py"),
    ("Physics",        "6_Physics.py"),
    ("Scenarios",      "7_Scenarios.py"),
    ("Models",         "8_Models.py"),
    ("Run Pipeline",   "9_Run_Pipeline.py"),
    ("Results",        "10_Results.py"),
]

_LABELS = [lbl for lbl, _ in PAGES]
_FILES  = [fn  for _, fn  in PAGES]

# ── Navigation state ──────────────────────────────────────────────────────────
if "_nav_index" not in st.session_state:
    st.session_state["_nav_index"] = 0
if "_max_visited" not in st.session_state:
    st.session_state["_max_visited"] = 0

# Update high-water mark
st.session_state["_max_visited"] = max(
    st.session_state["_max_visited"],
    st.session_state["_nav_index"],
)

current = st.session_state["_nav_index"]
visited = st.session_state["_max_visited"]

# ── Build dynamic CSS for per-item dot / connector / label styling ────────────
_nav_rules = []
_sel = 'section[data-testid="stSidebar"] div[role="radiogroup"] > label'

for idx in range(len(_LABELS)):
    n = idx + 1  # CSS nth-child is 1-indexed

    # Dot colour
    if idx <= visited:
        dot_style = f"background: {PRIMARY}; border: none;"
    else:
        dot_style = "background: transparent; border: 2px solid #cbd5e1;"

    # Label text
    if idx == current:
        txt_style = f"font-weight: 600 !important; color: {PRIMARY} !important;"
    elif idx <= visited:
        txt_style = "font-weight: 500 !important; color: #334155 !important;"
    else:
        txt_style = f"font-weight: 400 !important; color: {GRAY_MID} !important;"

    _nav_rules.append(f"{_sel}:nth-child({n}) {{ {txt_style} }}")
    _nav_rules.append(f"{_sel}:nth-child({n})::before {{ {dot_style} }}")

    # Connector line (after every item except the last)
    if idx < len(_LABELS) - 1:
        line_bg = PRIMARY if idx < visited else "#e2e8f0"
        _nav_rules.append(f"{_sel}:nth-child({n})::after {{ background: {line_bg}; }}")

st.markdown(f"<style>{''.join(_nav_rules)}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p class='nav-section'>CONFIGURE</p>", unsafe_allow_html=True)

    choice = st.radio(
        "Navigation",
        _LABELS,
        index=current,
        key="_sidebar_nav",
        label_visibility="collapsed",
    )
    new_idx = _LABELS.index(choice)
    if new_idx != current:
        st.session_state["_nav_index"] = new_idx
        st.rerun()

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p class='nav-section'>STATUS</p>", unsafe_allow_html=True)

    def _check(key: str, label: str):
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
page_file = Path(__file__).parent / "pages" / _FILES[current]

if page_file.exists():
    with open(page_file, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, str(page_file), "exec"), {"__name__": "__page__", "__file__": str(page_file)})
else:
    st.info(f"Page **{_LABELS[current]}** is not yet implemented.")
