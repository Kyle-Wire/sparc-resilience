"""run — Pipeline execution scripts.

Sub-packages
------------
orchestration/   RunContext, PipelineOrchestrator, PipelinePaths, disk policy
stages/          One runner per pipeline stage (run_stage0 … run_decision)
cv/              SpatialCVEngine — deep module for spatial cross-validation
training/        run_neural, run_continual, run_transfer

All symbols are also importable from ``sparc.run`` directly for backward
compatibility with code that imports from the flat namespace.
"""

from sparc.run.cv_engine import SpatialCVEngine, CVResult

__all__ = [
    "SpatialCVEngine",
    "CVResult",
]
