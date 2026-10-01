"""``python -m sparc.studio`` - same as ``sparc studio``."""

import sys

from sparc.studio.cli import main

if __name__ == "__main__":
    sys.exit(main())
