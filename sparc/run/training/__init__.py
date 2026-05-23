"""
sparc/run/training/ — Neural and continual training orchestration.

All training runners follow the signature
``main(ctx: RunContext, fast_mode: bool) → None`` and can be invoked
directly or via ``PipelineOrchestrator``::

    from sparc.run.training import run_neural, run_continual, run_transfer

The V2 neural training runner (``run_neural``) is the entry point for
training the ``SPARCMetaLearner`` + physics-informed loss.  It requires
torch and is skipped if torch is unavailable.
"""

# V2 neural training (requires torch)
try:
    from sparc.run.v2_neural_training import main as run_neural
    _NEURAL_AVAILABLE = True
except ImportError:
    run_neural = None  # type: ignore[assignment]
    _NEURAL_AVAILABLE = False

# Continual learning (city-to-city transfer with EWC/replay)
try:
    from sparc.run.continual_training import main as run_continual
    _CONTINUAL_AVAILABLE = True
except ImportError:
    run_continual = None  # type: ignore[assignment]
    _CONTINUAL_AVAILABLE = False

# Transfer learning (cross-city validation)
try:
    from sparc.run.transfer_training import main as run_transfer
    _TRANSFER_AVAILABLE = True
except ImportError:
    run_transfer = None  # type: ignore[assignment]
    _TRANSFER_AVAILABLE = False

__all__ = [
    "run_neural",
    "run_continual",
    "run_transfer",
]
