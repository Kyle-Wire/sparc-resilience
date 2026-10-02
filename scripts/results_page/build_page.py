"""Build the interactive results page of a finished core run (shim for :mod:`sparc.core.results_page`).

    python scripts/results_page/build_page.py <run dir> <core config> [--out results.html] [--placebo placebo.json]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparc.core.results_page import build_results_page, collect, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
