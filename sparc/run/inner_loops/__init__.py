"""Inner-loop reasoning subsystems for the SPARC pipeline.

Each module here implements a *narrow* search problem inside a single stage:
small search space, fast surrogate, full-model verification on the best
candidates. See ``docs/prd/prd-reasoning-engine-spine.md``.
"""
