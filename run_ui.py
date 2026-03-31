"""Launch the SPARC UI.  Usage:  python run_ui.py"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(HERE / "sparc" / "ui" / "app.py")],
        cwd=str(HERE),
    )
