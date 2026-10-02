"""Job worker with the core study functions faked: ``python fake_worker.py <job_dir>``.

The studies tests point the process executor at this script, so ``post.*`` and ``study.*`` jobs run in a
real worker process (events, hooks, ``result.json``) with :mod:`tests.studio.studies.fakes` in place of the
model fits.  The fakes record their arguments in ``<job_dir>/fake_calls.jsonl``.
"""

import sys
from pathlib import Path

from tests.studio.studies import fakes

fakes.JOB_DIR = Path(sys.argv[1]).resolve()
fakes.install()

from sparc.studio.jobs.worker import main  # noqa: E402 - after the fakes are installed

sys.exit(main(sys.argv[1:]))
