"""Page 10 — Results Viewer."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Results")
st.markdown("Browse pipeline outputs — plots, tables, maps, and downloadable files.")

# ── Locate output directory ───────────────────────────────────────────────────
wd = st.session_state.get("working_dir", "")
base = st.session_state.get("output_base_dir", "output")

output_dir = Path(wd) / base if wd else None

if not output_dir or not output_dir.exists():
    st.info("No output directory found. Run the pipeline first, or set the working directory.")
    # Allow manual override
    manual = st.text_input("Or enter output directory path manually:")
    if manual:
        output_dir = Path(manual)
    else:
        st.stop()

if not output_dir.exists():
    st.warning(f"Directory not found: `{output_dir}`")
    st.stop()

# ── Discover stages ───────────────────────────────────────────────────────────
stage_dirs = {
    "Stage 1 — Correlogram":  st.session_state.get("stage_dir_1", "Stage_1_Correlogram_Analysis"),
    "Stage 2 — Spatial CV":   st.session_state.get("stage_dir_2", "Stage_2_Spatial_CV"),
    "Stage 3 — Causal":       st.session_state.get("stage_dir_3", "Stage_3_Causal_Validation"),
    "Stage 4 — Scenarios":    st.session_state.get("stage_dir_4", "Stage_4_Scenarios"),
    "Final Results":          st.session_state.get("stage_dir_final", "Final_Interpretation_Results"),
}

available_stages = {}
for label, dirname in stage_dirs.items():
    d = output_dir / dirname
    if d.exists() and any(d.rglob("*")):
        available_stages[label] = d

if not available_stages:
    st.info("No completed stages found in the output directory.")
    st.stop()

# ── Stage selector ────────────────────────────────────────────────────────────
selected_stage = st.selectbox("Select stage to view", list(available_stages.keys()))
stage_path = available_stages[selected_stage]

# ── Display results ───────────────────────────────────────────────────────────

# Find all files
all_files = sorted(stage_path.rglob("*"))
images = [f for f in all_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}]
csvs   = [f for f in all_files if f.suffix.lower() == ".csv"]
jsons  = [f for f in all_files if f.suffix.lower() == ".json"]
gpkgs  = [f for f in all_files if f.suffix.lower() == ".gpkg"]
others = [f for f in all_files if f.is_file() and f not in images + csvs + jsons + gpkgs]

# ── Images / plots ────────────────────────────────────────────────────────────
if images:
    st.markdown("### Plots & Maps")
    # Show in a grid
    cols_per_row = 2
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(images):
                with col:
                    st.image(str(images[idx]), caption=images[idx].name, use_container_width=True)

# ── CSV tables ────────────────────────────────────────────────────────────────
if csvs:
    st.markdown("### Data Tables")
    for csv_file in csvs:
        with st.expander(f"[table] {csv_file.name}"):
            import pandas as pd
            try:
                df = pd.read_csv(str(csv_file))
                st.dataframe(df, use_container_width=True, height=300)
                st.download_button(
                    f"Download {csv_file.name}",
                    data=csv_file.read_bytes(),
                    file_name=csv_file.name,
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(f"Could not read: {e}")

# ── JSON files ────────────────────────────────────────────────────────────────
if jsons:
    st.markdown("### JSON Outputs")
    for json_file in jsons:
        with st.expander(f"[json] {json_file.name}"):
            import json
            try:
                data = json.loads(json_file.read_text())
                st.json(data)
                st.download_button(
                    f"Download {json_file.name}",
                    data=json_file.read_bytes(),
                    file_name=json_file.name,
                    mime="application/json",
                )
            except Exception as e:
                st.warning(f"Could not read: {e}")

# ── GeoPackages ───────────────────────────────────────────────────────────────
if gpkgs:
    st.markdown("### GeoPackage Files")
    for gpkg in gpkgs:
        with st.expander(f"[geo] {gpkg.name}"):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(str(gpkg))
                st.markdown(f"**{len(gdf)} features**, CRS: `{gdf.crs}`")
                st.dataframe(gdf.drop(columns="geometry", errors="ignore").head(20),
                             use_container_width=True, height=300)
            except Exception as e:
                st.info(f"GeoPackage preview not available: {e}")
            st.download_button(
                f"Download {gpkg.name}",
                data=gpkg.read_bytes(),
                file_name=gpkg.name,
                mime="application/geopackage+sqlite3",
            )

# ── Other files ───────────────────────────────────────────────────────────────
if others:
    st.markdown("### Other Files")
    for f in others:
        st.download_button(
            f"Download {f.name}",
            data=f.read_bytes(),
            file_name=f.name,
            key=f"dl_{f.name}_{hash(str(f))}",
        )

# ── Summary ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Output Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Plots", len(images))
col2.metric("Tables", len(csvs))
col3.metric("JSON", len(jsons))
col4.metric("GeoPackages", len(gpkgs))
