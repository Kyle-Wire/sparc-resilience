"""
sparc/run/stages/ — One file per pipeline stage runner.

Each stage runner has the signature ``main(ctx: RunContext, fast_mode: bool) → None``
and is invoked by ``PipelineOrchestrator``::

    from sparc.run.stages import (
        run_stage0,   # Correlogram / spatial autocorrelation
        run_stage1,   # GWEN variable selection
        run_stage2,   # Spatial CV + ensemble
        run_stage3,   # Causal validation
        run_stage4,   # Scenarios
        run_decision, # Intervention ranking
    )

All symbols are thin re-exports of the existing stage runner modules so that
callers importing from ``sparc.run.<stage_name>`` continue to work.
"""

# Stage 0 — spatial autocorrelation / correlograms
from sparc.run.correlogram_analysis import main as run_stage0

# Stage 1 — GWEN variable selection
from sparc.run.gwen_variable_selection import main as run_stage1

# Stage 2 — spatial CV + model ensemble
from sparc.run.enhanced_spatial_cv import main as run_stage2

# Stage 3 — causal validation
from sparc.run.causal_validation import main as run_stage3

# Stage 4 — scenario simulation
from sparc.run.scenarios import main as run_stage4

# Decision stage — intervention ranking
from sparc.run.decision_stage import main as run_decision

__all__ = [
    "run_stage0",
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "run_stage4",
    "run_decision",
]
