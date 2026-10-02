"""The Scenario Lab engine (SPEC §7.3–7.8, api.md §7, §13).

* :mod:`.compile` - ScenarioDoc → core ``Intervention`` arrays, predicted clamping, cost, guardrails;
* :mod:`.preview` - the emulator preview (``POST /preview``), single-flight latest-wins per run;
* :mod:`.host` - the engine host process: an LRU of :class:`sparc.core.session.RunSession` objects serving
  one request at a time over ``multiprocessing.connection``;
* :mod:`.ops` - what the host (and the across-runs job) does per request: open, scenario, batch, sweep, …;
* :mod:`.client` / :mod:`.service` - the server's connection to the host and the per-run engine states;
* :mod:`.executor` - :class:`~.executor.EngineExecutor`, the job executor of the ``engine`` lane;
* :mod:`.store` / :mod:`.stats` - result directories (api.md §12.2), cache keys and result statistics;
* :mod:`.kinds` - the ``engine.*``, ``scenario.across_runs`` and ``export.*_pack`` job kinds.

The server imports :mod:`.kinds`, :mod:`.client`, :mod:`.service`, :mod:`.executor`, :mod:`.compile`,
:mod:`.preview`, :mod:`.stats` and :mod:`.store`, none of which loads torch; the host loads the checkpoints.
"""
