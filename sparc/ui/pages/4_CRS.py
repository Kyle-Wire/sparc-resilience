"""Page 4 — Coordinate Reference System."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Coordinate Reference System")
st.markdown(
    "Specify the CRS of your input data and the projected CRS for spatial analysis. "
    "All spatial operations (distances, block sizes, etc.) use the projected CRS."
)

# ── Common presets ────────────────────────────────────────────────────────────
PRESETS = {
    "(select a preset)": ("", ""),
    "WGS 84 → UTM 17N (Southeast US)": ("EPSG:4326", "EPSG:32617"),
    "WGS 84 → UTM 18N (Mid-Atlantic)": ("EPSG:4326", "EPSG:32618"),
    "WGS 84 → UTM 19N (New England)": ("EPSG:4326", "EPSG:32619"),
    "RI State Plane → UTM 19N (Providence)": ("EPSG:3438", "EPSG:26919"),
    "NAD83 → UTM 10N (Pacific NW)": ("EPSG:4269", "EPSG:32610"),
    "WGS 84 → Web Mercator": ("EPSG:4326", "EPSG:3857"),
}

preset = st.selectbox("Common CRS Presets", list(PRESETS.keys()))
if preset != "(select a preset)":
    inp, proj = PRESETS[preset]
    st.session_state["crs_input"] = inp
    st.session_state["crs_projected"] = proj

st.divider()

# ── Manual entry ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.session_state["crs_input"] = st.text_input(
        "Input CRS (EPSG code) *",
        value=st.session_state["crs_input"],
        help="The CRS of your CSV coordinates, e.g. EPSG:4326",
    )

with col2:
    st.session_state["crs_projected"] = st.text_input(
        "Projected CRS (EPSG code) *",
        value=st.session_state["crs_projected"],
        help="Projected CRS for spatial analysis (metres), e.g. EPSG:26919",
    )

# ── Validation ────────────────────────────────────────────────────────────────
st.divider()

def _validate_crs(code: str) -> tuple[bool, str]:
    if not code:
        return False, "Not set"
    try:
        from pyproj import CRS
        c = CRS.from_user_input(code)
        return True, f"{c.name}  ({c.to_authority()})"
    except Exception as e:
        return False, str(e)

col_v1, col_v2 = st.columns(2)

with col_v1:
    ok1, msg1 = _validate_crs(st.session_state["crs_input"])
    if ok1:
        st.success(f"**Input CRS** — {msg1}")
    elif st.session_state["crs_input"]:
        st.error(f"**Input CRS** — {msg1}")
    else:
        st.info("Enter an input CRS code.")

with col_v2:
    ok2, msg2 = _validate_crs(st.session_state["crs_projected"])
    if ok2:
        st.success(f"**Projected CRS** — {msg2}")
    elif st.session_state["crs_projected"]:
        st.error(f"**Projected CRS** — {msg2}")
    else:
        st.info("Enter a projected CRS code.")
