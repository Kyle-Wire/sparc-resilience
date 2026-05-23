"""
Intervention Modeling Module for Physics-Anchored Urban Heat Interventions

This module implements a physics-prior based approach to predicting temperature
reductions from urban heat interventions (tree canopy, cool pavements, etc.).

Key Components:
- PhysicsPriors: Literature-based coefficient management
- ScaleHarmonizer: Spatial scale corrections (literature → 30m)
- CoefficientCalibrator: Shrinkage-based local calibration
- OOFExtractor: Out-of-fold spatial intelligence extraction
- ScenarioEngine: Intervention scenario prediction
- PortfolioOptimizer: Budget-constrained intervention selection

Author: SPARC Labs
Date: October 9, 2025
"""

__version__ = "0.1.0"

from .physics_priors import (
    PhysicsPriors,
    ScaleHarmonizer,
    CoefficientCalibrator,
    PriorSensitivityAnalyzer,
    ConservationChecker,
    MultiSourcePriorAggregator,
)

from .config.intervention_config import InterventionConfig

try:
    from .oof_extractor import (
        OOFExtractor,
        OOFExtractionConfig,
        SpatialIntelligence
    )
except ImportError:
    OOFExtractor = None
    OOFExtractionConfig = None
    SpatialIntelligence = None

__all__ = [
    'PhysicsPriors',
    'ScaleHarmonizer',
    'CoefficientCalibrator',
    'PriorSensitivityAnalyzer',
    'ConservationChecker',
    'MultiSourcePriorAggregator',
    'InterventionConfig',
    'OOFExtractor',
    'OOFExtractionConfig',
    'SpatialIntelligence',
    # Causal inputs seam (Candidate 4)
    'CausalInputs',
    'CausalInputLoader',
]

from .causal_inputs import CausalInputs, CausalInputLoader
