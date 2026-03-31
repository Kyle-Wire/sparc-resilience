"""Page 9 -- Generate Config, Validate & Run Pipeline."""
import re
import streamlit as st
from pathlib import Path
import subprocess
import sys
import os
import io
import zipfile

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sparc.ui.yaml_generator import generate_all
from sparc.ui.state import is_cloud, get_cloud_working_dir

st.title("Run Pipeline")
st.markdown("Generate configuration files, validate, and execute the SPARC pipeline.")

# -- Pre-flight checks ---------------------------------------------------------
_CLOUD = is_cloud()

wd = st.session_state.get("working_dir", "").strip().strip('"').strip("'")
if not wd:
    if _CLOUD:
        wd = get_cloud_working_dir()
        st.session_state["working_dir"] = wd
    else:
        st.warning("Set a **Working Directory** on the Project Setup page before running.")
        st.stop()

project_yml = Path(wd) / "project.yml"

# ==============================================================================
# CLOUD MODE — generate & download config files
# ==============================================================================
if _CLOUD:
    st.markdown("## Generate & Download Configuration")
    st.info(
        "☁️ **Cloud mode** — the full pipeline requires local compute. "
        "Use this page to generate your configuration files, then download "
        "and run on your own machine."
    )

    st.markdown("### How to Run Locally")
    st.markdown(
        "**1. Clone the repo** (one-time setup)\n"
        "```bash\n"
        "git clone https://github.com/SPARC-Labs-LLC/SPARC_Labs_GW3C.git\n"
        "cd SPARC_Labs_GW3C\n"
        "```\n\n"
        "**2. Install** (one-time setup)\n"
        "```bash\n"
        "# Windows\n"
        "scripts\\Install_SPARC.bat\n\n"
        "# Mac/Linux\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -e \".[ui]\"\n"
        "```\n\n"
        "**3. Download your config ZIP** below and extract into a project folder\n\n"
        "**4. Run the pipeline**\n"
        "```bash\n"
        "sparc run --project path/to/project.yml --stage all\n"
        "```\n\n"
        "Or launch the full UI locally: `scripts\\Start_SPARC.bat` (Windows) "
        "or `streamlit run run_ui.py` (Mac/Linux)"
    )

    if st.button("Generate Config Files", use_container_width=True):
        try:
            generate_all(wd)
            st.success("Configuration files generated!")
            st.session_state["_cloud_config_generated"] = True
        except Exception as e:
            st.error(f"Error generating config: {e}")

    if st.session_state.get("_cloud_config_generated"):
        # Collect all generated files into a ZIP
        config_files = []
        for pattern in ["*.yml", "causal/*.yml", "physics/*.yml"]:
            config_files.extend(Path(wd).glob(pattern))

        if config_files:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in config_files:
                    zf.write(fp, fp.relative_to(wd))
            buf.seek(0)

            st.download_button(
                "📥 Download Config (ZIP)",
                data=buf,
                file_name=f"{st.session_state.get('project_name', 'sparc')}_config.zip",
                mime="application/zip",
                use_container_width=True,
            )

            # Also show individual file previews
            for fp in config_files:
                with st.expander(str(fp.relative_to(wd))):
                    st.code(fp.read_text(encoding="utf-8"), language="yaml")

    st.stop()  # Skip the local-only sections below

# -- Stage-specific progress milestones ----------------------------------------
# Each entry is (regex_pattern, progress_fraction, display_label)
STAGE_MILESTONES = {
    "0": [
        (r"Stage 0.*Correlogram|Correlogram.*Spatial Analysis", 0.05, "Starting correlogram analysis..."),
        (r"Computing spatial correlogram", 0.20, "Computing spatial correlogram..."),
        (r"Bandwidth statistics", 0.50, "Analyzing bandwidth statistics..."),
        (r"Optimal CV block size", 0.80, "Determined optimal block size"),
        (r"Correlogram.*Complete|Stage 0.*complete", 0.95, "Correlogram analysis complete"),
    ],
    "1": [
        (r"Stage 1.*GWEN|GWEN Variable Selection", 0.05, "Starting GWEN variable selection..."),
        (r"GWEN.*selection|Fitting GWEN", 0.30, "Running GWEN selection..."),
        (r"gwen.*complete|Stage 1.*complete", 0.95, "GWEN selection complete"),
    ],
    "2": [
        (r"Stage 2.*Enhanced Spatial CV|Hardware-optimized", 0.02, "Initializing Spatial CV..."),
        (r"PARALLEL MODE|Models will be processed", 0.05, "Configuring parallel execution..."),
        (r"OLS.*R.*=|Running OLS|Fitting OLS", 0.15, "Running OLS..."),
        (r"GWR.*R.*=|Running GWR|Fitting GWR|Tuning bandwidth", 0.30, "Running GWR (bandwidth tuning)..."),
        (r"GWRF.*R.*=|Running GWRF|Fitting GWRF", 0.45, "Running GWRF..."),
        (r"GGPGAM.*R.*=|Fitting GGPGAM|Running GGPGAM", 0.60, "Running GGPGAM..."),
        (r"Deep Kriging|Training Deep Kriging|deep.kriging", 0.72, "Training Deep Kriging..."),
        (r"meta.ensemble|Meta.Ensemble|stacking|LightGBM", 0.82, "Building meta-ensemble..."),
        (r"Stage 2 Complete|Results saved", 0.95, "Spatial CV complete"),
    ],
    "3": [
        (r"Stage 3.*Causal|Causal Validation", 0.05, "Starting causal validation..."),
        (r"DAG loaded", 0.10, "DAG loaded"),
        (r"structural coefficients", 0.25, "Estimating structural coefficients..."),
        (r"ATEs via DoWhy", 0.35, "Estimating ATEs (DoWhy)..."),
        (r"ATEs via IPW", 0.45, "Estimating ATEs (IPW)..."),
        (r"ATEs via GPS", 0.55, "Estimating ATEs (GPS)..."),
        (r"ATEs via Matching", 0.65, "Estimating ATEs (Matching)..."),
        (r"spatial CATE|CausalForestDML", 0.75, "Estimating spatial CATE..."),
        (r"dose.response", 0.85, "Computing dose-response curves..."),
        (r"Stage 3 complete", 0.95, "Causal validation complete"),
    ],
    "4": [
        (r"Stage 4.*Scenario|PHYSICS.CONSTRAINED", 0.05, "Starting scenario simulation..."),
        (r"Loading models", 0.10, "Loading models..."),
        (r"Loading baseline", 0.20, "Loading baseline data..."),
        (r"spatial heterogeneity", 0.30, "Computing spatial heterogeneity..."),
        (r"PHYSICS PRIORS|Literature", 0.40, "Applying physics priors..."),
        (r"RUNNING SCENARIOS", 0.50, "Running scenarios..."),
        (r"MONTE.CARLO|MC draw", 0.70, "Monte-Carlo uncertainty propagation..."),
        (r"MC complete|MC quantiles", 0.85, "Monte-Carlo complete"),
        (r"Stage 4.*complete|Scenario.*complete", 0.95, "Scenario simulation complete"),
    ],
    "all": [],  # dynamically built from stages 0-4 below
}

# For "all" mode, build a combined milestone list that spans the full 0-100% range
_all_stages = ["0", "1", "2", "3", "4"]
_per_stage_weight = {
    "0": (0.00, 0.12),   # 0-12%  (Correlogram)
    "1": (0.12, 0.20),   # 12-20% (GWEN)
    "2": (0.20, 0.60),   # 20-60% (heaviest stage)
    "3": (0.60, 0.82),   # 60-82%
    "4": (0.82, 1.00),   # 82-100%
}
for _stg, (_lo, _hi) in _per_stage_weight.items():
    for pattern, frac, label in STAGE_MILESTONES[_stg]:
        global_frac = _lo + frac * (_hi - _lo)
        STAGE_MILESTONES["all"].append((pattern, global_frac, label))


def _match_progress(line: str, milestones: list) -> tuple[float, str] | None:
    """Check a log line against milestones, return (fraction, label) or None."""
    for pattern, frac, label in milestones:
        if re.search(pattern, line, re.IGNORECASE):
            return frac, label
    return None


# ==============================================================================
# Step 1: Generate Config
# ==============================================================================
st.markdown("## 1. Generate Configuration Files")
st.markdown(f"Target directory: `{wd}`")

if st.button("Generate Config Files", use_container_width=True):
    try:
        out_path = generate_all(wd)
        st.success(f"Generated configuration files in `{wd}`")

        written = []
        for f in ["project.yml", "causal/dag.yml", "physics/priors.yml", "physics/caps.yml"]:
            fp = Path(wd) / f
            if fp.exists():
                written.append(f)
        st.markdown("**Files written:** " + ", ".join(f"`{f}`" for f in written))
    except Exception as e:
        st.error(f"Error generating config: {e}")

# ==============================================================================
# Step 2: Validate
# ==============================================================================
st.divider()
st.markdown("## 2. Validate Configuration")

if st.button("Validate", use_container_width=True):
    if not project_yml.exists():
        st.error("Generate config files first.")
    else:
        with st.spinner("Validating..."):
            result = subprocess.run(
                [sys.executable, "-m", "sparc", "validate", "--project", str(project_yml)],
                cwd=str(_REPO),
                capture_output=True, text=True, timeout=60,
            )
        if result.returncode == 0:
            st.success("Validation passed!")
        else:
            st.error("Validation failed.")
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")

# ==============================================================================
# Step 3: Run Pipeline
# ==============================================================================
st.divider()
st.markdown("## 3. Run Pipeline")

col1, col2 = st.columns(2)
stage = col1.selectbox(
    "Stage to run",
    ["all", "0", "1", "2", "3", "4"],
    help="0=Correlogram, 1=GWEN, 2=SpatialCV, 3=Causal, 4=Scenarios",
)
fast = col2.checkbox("Fast mode", value=st.session_state.get("fast_mode", False))

# Stage progress indicators
st.markdown("### Stage Progress")
output_dir = Path(wd) / st.session_state.get("output_base_dir", "output")
stage1_dir = output_dir / st.session_state.get("stage_dir_1", "Stage_1_Correlogram_Analysis")
stage2_dir = output_dir / st.session_state.get("stage_dir_2", "Stage_2_Spatial_CV")
stage3_dir = output_dir / st.session_state.get("stage_dir_3", "Stage_3_Causal_Validation")
stage4_dir = output_dir / st.session_state.get("stage_dir_4", "Stage_4_Scenarios")

progress_cols = st.columns(5)
checks = [
    (stage1_dir / ".correlogram_complete").exists(),
    (stage1_dir / ".gwen_complete").exists(),
    stage2_dir.exists() and any(stage2_dir.iterdir()) if stage2_dir.exists() else False,
    stage3_dir.exists() and any(stage3_dir.iterdir()) if stage3_dir.exists() else False,
    stage4_dir.exists() and any(stage4_dir.iterdir()) if stage4_dir.exists() else False,
]
labels = ["Correlogram", "GWEN", "Spatial CV", "Causal", "Scenarios"]
for col, label, done in zip(progress_cols, labels, checks):
    col.markdown(f"{'[done]' if done else '[    ]'} **{label}**")

# Run
if st.button("Run Pipeline", type="primary", use_container_width=True):
    if not project_yml.exists():
        st.error("Generate and validate config files first.")
    else:
        cmd = [
            sys.executable, "-m", "sparc", "run",
            "--project", str(project_yml),
            "--stage", stage,
        ]
        if fast:
            cmd.append("--fast")

        st.markdown(f"**Command:** `{' '.join(cmd)}`")

        # -- Progress UI elements --
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_expander = st.expander("Show detailed log", expanded=False)
        log_area = log_expander.empty()

        milestones = STAGE_MILESTONES.get(stage, [])
        current_progress = 0.0
        log_lines: list[str] = []

        # Build env with SPARC_PROJECT and PYTHONPATH so pipeline finds both.
        # PYTHONIOENCODING=utf-8 prevents UnicodeEncodeError when the
        # pipeline prints emoji/Unicode to a piped stdout on Windows.
        # PYTHONUNBUFFERED=1 forces real-time flushing so the log updates
        # continuously instead of freezing during long model training steps.
        env = os.environ.copy()
        env["SPARC_PROJECT"] = str(project_yml)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        if str(_REPO) not in env.get("PYTHONPATH", ""):
            env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [cmd[0], "-u"] + cmd[1:],   # -u = unbuffered stdout/stderr
            cwd=str(wd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        status_text.markdown("**Starting pipeline...**")

        for line in proc.stdout:
            log_lines.append(line)
            # Update log (last 150 lines)
            log_area.code("".join(log_lines[-150:]), language="text")

            # Check for progress milestone
            match = _match_progress(line, milestones)
            if match:
                frac, label = match
                if frac > current_progress:
                    current_progress = frac
                    progress_bar.progress(min(current_progress, 1.0))
                    status_text.markdown(f"**{label}**")

        proc.wait()

        # -- Persist the log so it survives page refreshes ----------------
        full_log = "".join(log_lines)
        log_path = Path(wd) / f"stage{stage}_log.txt"
        try:
            log_path.write_text(full_log, encoding="utf-8")
        except Exception:
            pass  # non-critical

        st.session_state["last_run_log"] = full_log
        st.session_state["last_run_rc"] = proc.returncode
        st.session_state["last_run_stage"] = stage

        if proc.returncode == 0:
            progress_bar.progress(1.0)
            status_text.markdown("**Pipeline completed successfully!**")
            st.success("Pipeline completed successfully!")
            st.caption(f"Full log saved to `{log_path}`")
        else:
            status_text.markdown(f"**Pipeline exited with code {proc.returncode}**")
            st.error(f"Pipeline exited with code {proc.returncode}")
            st.caption(f"Full log saved to `{log_path}`")
            # Show last 80 lines on failure so user can see the error
            st.code("".join(log_lines[-80:]), language="text")

# -- Show log from previous run if it exists --------------------------------
elif st.session_state.get("last_run_log"):
    rc = st.session_state.get("last_run_rc", -1)
    stg = st.session_state.get("last_run_stage", "?")
    if rc == 0:
        st.success(f"Previous run (stage {stg}) completed successfully.")
    else:
        st.error(f"Previous run (stage {stg}) exited with code {rc}.")
    with st.expander("Show log from previous run", expanded=(rc != 0)):
        st.code(st.session_state["last_run_log"][-8000:], language="text")
