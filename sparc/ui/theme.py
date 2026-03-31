"""
SPARC Design System — SPARC Labs LLC brand theme for Streamlit.

Color palette, CSS injection, chart theming, and brand constants.
"""
import streamlit as st
from pathlib import Path

# ── Primary palette — clean, scientific, sharp ───────────────────────────────
PRIMARY     = "#1e3a5f"   # deep navy
PRIMARY_LT  = "#2d5986"   # lighter navy for hover
ACCENT      = "#0ea5e9"   # sky-blue accent
ACCENT_DARK = "#0284c7"   # pressed accent

# Spectral ramp for charts (purple → magenta → red → orange → yellow)
PURPLE   = "#602468"
MAGENTA  = "#9B2D7B"
PINK     = "#C7385F"
RED      = "#E04B3D"
ORANGE   = "#F0882D"
YELLOW   = "#FBDD46"
SPECTRAL = [PRIMARY, ACCENT, PURPLE, MAGENTA, PINK, RED, ORANGE, YELLOW]

# UI chrome
WHITE       = "#FFFFFF"
BLACK       = "#0f172a"     # slate-900
GRAY_LIGHT  = "#e2e8f0"    # slate-200
GRAY_MID    = "#94a3b8"    # slate-400
GRAY_BG     = "#f8fafc"    # slate-50
SIDEBAR_BG  = "#f1f5f9"    # slate-100

# ── Font stack ──
FONT_STACK = (
    "'Inter', 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
)

# ── Paths ──
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH  = ASSETS_DIR / "logo.png"


def inject_css():
    """Inject the SPARC brand CSS into the current Streamlit page."""
    css = f"""
    <style>
    /* ── Import Inter from Google Fonts ────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Clean background (subtle dot grid) ────────────────────── */
    .stMain {{
        background-color: {GRAY_BG};
        background-image: radial-gradient(circle, {GRAY_LIGHT}66 1px, transparent 1px);
        background-size: 24px 24px;
    }}

    /* ── Typography ────────────────────────────────────────────── */
    html, body, .stApp {{
        font-family: {FONT_STACK};
        color: {BLACK};
    }}

    .stMarkdown h1 {{
        font-family: {FONT_STACK};
        font-weight: 700;
        letter-spacing: -0.025em;
        color: {PRIMARY};
        font-size: 1.75rem;
    }}

    .stMarkdown h2 {{
        font-family: {FONT_STACK};
        font-weight: 600;
        color: {PRIMARY};
        font-size: 1.25rem;
        letter-spacing: -0.01em;
    }}

    .stMarkdown h3 {{
        font-family: {FONT_STACK};
        font-weight: 600;
        color: {BLACK};
    }}

    /* ── Sidebar ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG};
        border-right: 1px solid {GRAY_LIGHT};
        padding-top: 0;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {{
        color: {BLACK} !important;
    }}

    /* ── Nav section headers ──────────────────────────────────── */
    .nav-section {{
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {GRAY_MID} !important;
        font-weight: 600;
        margin-bottom: 0.25rem;
        padding-left: 0.2rem;
    }}

    /* ── Nav buttons (inside sidebar) ─────────────────────────── */
    section[data-testid="stSidebar"] .stButton > button {{
        background-color: transparent;
        color: #334155;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.45rem 0.7rem;
        text-align: left;
        justify-content: flex-start;
        transition: background-color 0.15s ease, color 0.15s ease;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {PRIMARY}12;
        color: {PRIMARY};
    }}
    section[data-testid="stSidebar"] .stButton > button:focus {{
        background-color: {PRIMARY};
        color: {WHITE};
        box-shadow: none;
    }}

    /* Hide Streamlit default top nav in sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        display: none;
    }}

    /* ── Primary action buttons (main area) ─────────────────── */
    .stMainBlockContainer .stButton > button {{
        background-color: {PRIMARY};
        color: {WHITE};
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.4rem;
        letter-spacing: 0.01em;
        transition: background-color 0.15s ease;
    }}
    .stMainBlockContainer .stButton > button:hover {{
        background-color: {PRIMARY_LT};
        color: {WHITE};
    }}

    /* ── Metrics / KPI cards ──────────────────────────────────── */
    [data-testid="stMetric"] {{
        background-color: {WHITE};
        border: 1px solid {GRAY_LIGHT};
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.05);
    }}

    /* ── Inputs — sharper look ────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div {{
        border-radius: 6px !important;
        border-color: {GRAY_LIGHT} !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 2px {ACCENT}33 !important;
    }}

    /* ── Expander headers ─────────────────────────────────────── */
    details summary {{
        font-weight: 600 !important;
    }}

    /* ── Divider — more subtle ────────────────────────────────── */
    hr {{
        border-color: {GRAY_LIGHT} !important;
        opacity: 0.5;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar_logo():
    """Display the SPARC logo + wordmark at the top of the sidebar."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding:0.8rem 0 0.2rem;'>",
            unsafe_allow_html=True,
        )
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=130)
        st.markdown(
            f"<p style='text-align:center; font-size:0.65rem; "
            f"letter-spacing:0.06em; color:{GRAY_MID}; "
            f"margin-top:-4px; margin-bottom:0;'>"
            f"SPATIAL ANALYSIS &amp; RESEARCH CORE</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")


def get_matplotlib_style() -> dict:
    """Return a matplotlib rcParams dict using the SPARC spectral palette."""
    return {
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": GRAY_LIGHT,
        "axes.labelcolor": BLACK,
        "axes.prop_cycle": f"cycler('color', {SPECTRAL})",
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "grid.color": GRAY_LIGHT,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "medium",
        "figure.titlesize": 16,
    }
