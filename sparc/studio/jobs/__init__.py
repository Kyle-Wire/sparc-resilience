"""Tracked jobs (SPEC §5.6–5.9, §10.2–10.5): kind registry, scheduler, executors, worker, tailer, projection.

* :mod:`.kinds` - ``@job_kind`` registry, ``KIND_MODULES``, ``JobContext``.
* :mod:`.manager` - ``JobManager``: queue, lanes, priorities, chains, per-run locks, thread budget, preflight,
  cancel/kill, reattach.
* :mod:`.executors` - ``Executor`` protocol, ``ProcessExecutor``, ``register_executor``.
* :mod:`.worker` - ``python -m sparc.studio.jobs.worker <job_dir>``.
* :mod:`.tailer` - ``JobTailer``: byte-offset cursor over ``events.jsonl`` → projection → hub + SQLite.
* :mod:`.tracker` - the projection reducer ``reduce(state, event)``.
* :mod:`.eta` - cost model, seed rates, calibration.
* :mod:`.resources` - ``ResourceSampler`` and preflight checks.
* :mod:`.testkinds` - ``test.*`` kinds (``SPARC_STUDIO_TEST_KINDS=1``).
* :mod:`.replay` - replay runner (``SPARC_STUDIO_RUNNER=replay:<dir>``).

Importing this package imports nothing heavy.
"""
