# Run rubric (starter; tune the criteria as the project moves)

Each criterion is graded on its own.

1. The run finished at least one step of `docs/agent/ROADMAP.md`'s current
   phase, or recorded a blocker for it in `docs/agent/STATUS.md` with the
   commands tried and their output.
2. `pytest tests/core/test_identify.py tests/core -m "not slow"` passes on the
   final pushed commit; the run log quotes the pass/fail counts.
3. Every code change in the run comes with a regression test that fails
   without the change.
4. Results are written into the docs the roadmap names for that step, with
   confidence intervals and lab verdicts where the step produces estimates.
   Null or failed results are reported as findings, not omitted.
5. No file under `output/`, no raw download, and no credential is committed.
6. `docs/agent/STATUS.md` is updated: current phase, gate status with
   evidence, next step, blockers, and a run-log entry listing commit hashes.
7. The memory store has a note for this run (what was done, quirks found,
   timings, what to do next).
8. All commits are pushed to `origin/sparc-steward`; nothing was pushed to any
   other branch.
9. No phase was started before the previous phase's gate was recorded as met.
   Every significant decision (method change, rule change, new phase or field)
   has an entry under "Agent decisions" in `docs/agent/STATUS.md` written
   before acting, and any change to a rule that judges results was followed by
   re-running and reporting every affected city, before and after.
