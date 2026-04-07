"""
training — V2 neural training infrastructure for SPARC
======================================================

Provides the complete training stack for SPARC's end-to-end differentiable
neural meta-learner, including:

- Joint loss function (8-term physics-informed objective)
- Per-component AdamW optimizer with learning rate groups
- 4-stage training curriculum (warmup → physics → joint → SWA)
- CMA-ES hyperparameter optimization

All components are opt-in via ``project.yml`` configuration.  Setting
``models.meta_learner: lightgbm`` bypasses this module entirely and
uses the V1 LightGBM meta-ensemble path.
"""

__all__ = [
    "loss",
    "optimizer",
    "curriculum",
    "cma_es",
]
