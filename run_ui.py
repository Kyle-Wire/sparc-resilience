"""Launch the SPARC UI.  Usage:  streamlit run run_ui.py"""
import importlib, runpy, sys
from pathlib import Path

# Ensure repo root is on sys.path so sparc.* imports work
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Delegate to the real app module
from sparc.ui import app  # noqa: F401 — Streamlit executes on import
