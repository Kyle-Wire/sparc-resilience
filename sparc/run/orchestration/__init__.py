"""
sparc/run/orchestration/ — Pipeline scheduling, paths, and disk policy.

Exports the public surface of the orchestration layer; everything
stage runners need to construct and inspect a ``RunContext``.

Re-exports (backward compat: callers using ``sparc.run.orchestrator``
continue to work; these names are also available at this shorter path)::

    from sparc.run.orchestration import RunContext, PipelineOrchestrator
    from sparc.run.orchestration import PipelinePaths
    from sparc.run.orchestration import set_disk_writes, get_disk_writes
"""

from sparc.run.orchestrator import RunContext, PipelineOrchestrator
from sparc.run.pipeline_paths import PipelinePaths
from sparc.run.disk_policy import set_disk_writes, disk_writes_enabled

__all__ = [
    "RunContext",
    "PipelineOrchestrator",
    "PipelinePaths",
    "set_disk_writes",
    "disk_writes_enabled",
]
