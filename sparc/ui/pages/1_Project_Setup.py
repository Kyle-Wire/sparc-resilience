"""Page 1 — Project Setup."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sparc.ui.state import load_template, load_from_yaml, available_templates

st.title("Project Setup")
st.markdown("Configure the basic metadata and working directory for your SPARC project.")

# ── Quick-start template loader ───────────────────────────────────────────────
st.markdown("## Quick Start")

templates = available_templates()
col1, col2 = st.columns([2, 1])
with col1:
    selected_template = st.selectbox(
        "Domain template",
        options=templates,
        index=templates.index("blank") if "blank" in templates else 0,
        help="Choose a domain template to pre-fill settings, or start blank.",
    )
with col2:
    st.markdown("<div style='height:1.65rem;'></div>", unsafe_allow_html=True)
    if st.button("Load Template", use_container_width=True):
        load_template(selected_template)
        st.rerun()

with st.expander("Or import an existing project.yml"):
    uploaded = st.file_uploader("Upload project.yml", type=["yml", "yaml"])
    if uploaded is not None:
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "project.yml"
        tmp.write_bytes(uploaded.getvalue())
        load_from_yaml(str(tmp))
        st.rerun()

st.divider()

# ── Project metadata ──────────────────────────────────────────────────────────
st.markdown("## Project Metadata")

col_a, col_b = st.columns(2)
with col_a:
    st.session_state["project_name"] = st.text_input(
        "Project Name *", value=st.session_state["project_name"]
    )
    st.session_state["project_author"] = st.text_input(
        "Author", value=st.session_state["project_author"]
    )
    st.session_state["project_domain"] = st.selectbox(
        "Domain",
        ["uhi", "groundwater", "air_quality", "ecology", "custom"],
        index=["uhi", "groundwater", "air_quality", "ecology", "custom"].index(
            st.session_state.get("project_domain", "custom")
        ),
    )
with col_b:
    st.session_state["project_version"] = st.text_input(
        "Version", value=st.session_state["project_version"]
    )
    st.session_state["response_units"] = st.text_input(
        "Response Units (e.g. °F, mg/L)", value=st.session_state["response_units"]
    )

st.session_state["project_description"] = st.text_area(
    "Description", value=st.session_state["project_description"], height=100
)

st.divider()

# ── Working directory ─────────────────────────────────────────────────────────
st.markdown("## Working Directory")
st.markdown(
    "All configuration files (`project.yml`, `dag.yml`, physics files) will be "
    "written here. Pipeline outputs go into a sub-folder."
)
_raw_wd = st.text_input(
    "Project Folder Path",
    value=st.session_state["working_dir"],
    help="Absolute path to the folder where config files will be generated.",
)
st.session_state["working_dir"] = _raw_wd.strip().strip('"').strip("'") if _raw_wd else ""

wd = st.session_state["working_dir"]
if wd:
    p = Path(wd)
    if p.exists():
        st.success(f"Folder exists: `{p}`")
    else:
        st.warning("Folder does not exist yet — it will be created when you generate config files.")
