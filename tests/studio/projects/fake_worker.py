"""Job worker with the network fetchers faked: ``python fake_worker.py <job_dir>``.

``test_inputs.py`` points the process executor at this script, so input jobs
run in a real worker process (events, result.json, exit codes) but never
touch the network.
"""

import sys

from tests.studio.projects.fakes import install

install()

from sparc.studio.jobs.worker import main  # noqa: E402 - after the fakes are installed

sys.exit(main(sys.argv[1:]))
