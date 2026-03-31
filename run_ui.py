"""Launch the SPARC UI.  Usage:  streamlit run run_ui.py"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so sparc.* imports work
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Exec (not import) app.py so it re-runs on every Streamlit interaction.
# A plain import only executes once due to Python's module cache, causing
# blank pages after the first user click.
_app_path = str(_REPO_ROOT / "sparc" / "ui" / "app.py")
with open(_app_path, encoding="utf-8") as _f:
    exec(
        compile(_f.read(), _app_path, "exec"),
        {"__name__": "__main__", "__file__": _app_path},
    )
