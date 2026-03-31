"""
Physics-based Prior Coefficients for Urban Heat Interventions

This module implements literature-based physics priors for predicting temperature
changes from urban heat interventions. Coefficients are derived from peer-reviewed
research and can be optionally calibrated to local conditions.

Author: SPARC Labs
Date: October 9, 2025
"""

import pandas as pd
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import json
from pathlib import Path
import warnings


@dataclass
class PhysicsCoefficient:
    """
    Container for a single physics prior coefficient with metadata.
    
    Attributes:
    -----------
    variable : str
        Variable name (e.g., 'Pct_Canopy', 'Pct_Impervious', 'NDVI', 'Albedo')
    coefficient : float
        Expected temperature change per unit change in variable
    units : str
        Units of the coefficient (e.g., '°F per +10 pp', '°F per +0.1')
    uncertainty : float
        Uncertainty range as fraction (e.g., 0.2 for ±20%)
    source_scale : str
        Spatial resolution of source study (e.g., '100m-500m', '1km')
    literature_source : str
        Citation or reference for the coefficient
    confidence : str
        Confidence level: 'high', 'medium', 'low'
    """
    variable: str
    coefficient: float
    units: str
    uncertainty: float = 0.2  # Default ±20%
    source_scale: str = "100m-1km"
    literature_source: str = ""
    confidence: str = "medium"
    
    def get_range(self) -> Tuple[float, float]:
        """Get the uncertainty range (min, max) for this coefficient."""
        delta = abs(self.coefficient * self.uncertainty)
        return (self.coefficient - delta, self.coefficient + delta)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'variable': self.variable,
            'coefficient': self.coefficient,
            'units': self.units,
            'uncertainty': self.uncertainty,
            'source_scale': self.source_scale,
            'literature_source': self.literature_source,
            'confidence': self.confidence,
            'range': self.get_range()
        }


class PhysicsPriors:
    """
    Management class for physics-based prior coefficients.
    
    This class stores and manages literature-based coefficients for urban heat
    interventions. Coefficients are based on peer-reviewed research from multiple
    cities and spatial scales.
    
    Literature Sources:
    -------------------
    - Canopy: Ziter et al. (2019) PNAS; Wong et al. (2021) Urban Climate
    - Impervious: Li et al. (2020) Remote Sensing; Zhou et al. (2021) Building and Environment
    - Albedo: Santamouris et al. (2011) Solar Energy; Akbari et al. (2009) Climatic Change
    - NDVI: Estoque et al. (2017) Science of the Total Environment; Peng et al. (2012) LSA
    
    Example:
    --------
    >>> priors = PhysicsPriors()
    >>> canopy_coef = priors.get_coefficient('Pct_Canopy')
    >>> print(f"Canopy effect: {canopy_coef.coefficient} {canopy_coef.units}")
    """
    
    def __init__(self):
        """Initialize with default literature-based coefficients."""
        self.coefficients: Dict[str, PhysicsCoefficient] = {}
        self._initialize_default_priors()
    
    def _initialize_default_priors(self):
        """
        Initialize default physics priors from literature.
        
        Coefficients represent consensus estimates from multiple studies:
        - Canopy: -0.280 °F per +10 percentage points (pp)
        - Impervious: +0.330 °F per +10 pp
        - Albedo: -0.230 °F per +0.1 increase
        - NDVI: -0.270 °F per +0.1 increase
        
        All coefficients have ±20% uncertainty reflecting inter-city variability.
        """
        # Tree Canopy Coverage
        # Meta-analysis of 15+ cities shows -0.25 to -0.35 °F per 10pp increase
        # Ziter et al. (2019): -0.28 °F/10pp in Madison, WI
        # Wong et al. (2021): -0.32 °F/10pp in Singapore
        self.coefficients['Pct_Canopy'] = PhysicsCoefficient(
            variable='Pct_Canopy',
            coefficient=-0.280,
            units='°F per +10 pp',
            uncertainty=0.20,
            source_scale='100m-500m',
            literature_source='Ziter et al. (2019) PNAS; Wong et al. (2021) Urban Climate',
            confidence='high'
        )
        
        # Impervious Surface Coverage
        # Multiple studies show warming effect of impervious surfaces
        # Li et al. (2020): +0.35 °F/10pp across Chinese cities
        # Zhou et al. (2021): +0.31 °F/10pp in Atlanta
        self.coefficients['Pct_Impervious'] = PhysicsCoefficient(
            variable='Pct_Impervious',
            coefficient=0.330,
            units='°F per +10 pp',
            uncertainty=0.20,
            source_scale='100m-500m',
            literature_source='Li et al. (2020) Remote Sensing; Zhou et al. (2021) Building Environ.',
            confidence='high'
        )
        
        # Surface Albedo
        # Cool pavement and cool roof studies
        # Santamouris et al. (2011): -0.20 to -0.28 °F per 0.1 albedo increase
        # Akbari et al. (2009): -0.24 °F per 0.1 across multiple cities
        self.coefficients['Albedo'] = PhysicsCoefficient(
            variable='Albedo',
            coefficient=-0.230,
            units='°F per +0.1',
            uncertainty=0.20,
            source_scale='30m-100m',
            literature_source='Santamouris et al. (2011) Solar Energy; Akbari et al. (2009)',
            confidence='medium'
        )
        
        # Normalized Difference Vegetation Index (NDVI)
        # Proxy for vegetation health and density
        # Estoque et al. (2017): -0.29 °F per 0.1 NDVI increase
        # Peng et al. (2012): -0.25 °F per 0.1 across multiple cities
        self.coefficients['NDVI'] = PhysicsCoefficient(
            variable='NDVI',
            coefficient=-0.270,
            units='°F per +0.1',
            uncertainty=0.20,
            source_scale='30m-100m',
            literature_source='Estoque et al. (2017) Sci Total Environ; Peng et al. (2012) LSA',
            confidence='medium'
        )
    
    def get_coefficient(self, variable: str) -> PhysicsCoefficient:
        """
        Get the physics coefficient for a specific variable.
        
        Parameters:
        -----------
        variable : str
            Variable name (case-sensitive): 'Pct_Canopy', 'Pct_Impervious', 'NDVI', 'Albedo'
        
        Returns:
        --------
        PhysicsCoefficient
            Coefficient object with value, uncertainty, and metadata
        
        Raises:
        -------
        KeyError
            If variable not found in coefficient database
        """
        if variable not in self.coefficients:
            available = list(self.coefficients.keys())
            raise KeyError(f"Variable '{variable}' not found. Available: {available}")
        return self.coefficients[variable]
    
    def get_all_coefficients(self) -> Dict[str, PhysicsCoefficient]:
        """Get all physics coefficients as a dictionary."""
        return self.coefficients.copy()
    
    def get_coefficient_value(self, variable: str) -> float:
        """Get just the coefficient value (float) for a variable."""
        return self.get_coefficient(variable).coefficient
    
    def get_uncertainty_range(self, variable: str) -> Tuple[float, float]:
        """Get the (min, max) uncertainty range for a coefficient."""
        return self.get_coefficient(variable).get_range()
    
    def export_to_dict(self) -> Dict:
        """Export all coefficients to a dictionary for JSON serialization."""
        return {
            var: coef.to_dict() 
            for var, coef in self.coefficients.items()
        }
    
    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Export coefficients to a JSON file.
        
        Parameters:
        -----------
        filepath : str or Path
            Output file path
        """
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2)
        print(f"Physics priors exported to: {filepath}")
    
    def load_from_json(self, filepath: Union[str, Path]) -> None:
        """
        Load coefficients from a JSON file.
        
        Parameters:
        -----------
        filepath : str or Path
            Input file path
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for var, coef_dict in data.items():
            # Remove 'range' key if present (it's computed, not stored)
            coef_dict.pop('range', None)
            self.coefficients[var] = PhysicsCoefficient(**coef_dict)
        print(f"Physics priors loaded from: {filepath}")
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary table of all coefficients.
        
        Returns:
        --------
        pd.DataFrame
            Summary table with columns: variable, coefficient, units, range, confidence
        """
        summary_data = []
        for var, coef in self.coefficients.items():
            min_val, max_val = coef.get_range()
            summary_data.append({
                'Variable': var,
                'Coefficient': coef.coefficient,
                'Units': coef.units,
                'Uncertainty_Range': f"[{min_val:.3f}, {max_val:.3f}]",
                'Source_Scale': coef.source_scale,
                'Confidence': coef.confidence
            })
        return pd.DataFrame(summary_data)
    
    def print_summary(self):
        """Print a formatted summary of all coefficients."""
        print("\n" + "="*80)
        print("PHYSICS PRIORS SUMMARY")
        print("="*80)
        df = self.summary()
        print(df.to_string(index=False))
        print("="*80)
        print(f"Note: All coefficients have ±20% uncertainty reflecting inter-city variability")
        print("="*80 + "\n")


class ScaleHarmonizer:
    """
    Convert coefficients from literature spatial scales to target 30m resolution.
    
    Most urban heat literature is based on 100m-1km resolution data. This class
    applies empirically-grounded scale correction factors to harmonize coefficients
    to finer 30m resolution.
    
    Theory:
    -------
    Spatial averaging at coarser resolutions dampens local variability, leading to
    systematically weaker observed coefficients. Scale correction uses power-law
    relationships calibrated from multi-resolution studies.
    
    Correction Formula:
    -------------------
    β_30m = β_literature × scale_factor
    
    where scale_factor depends on:
    - Source resolution (100m, 500m, 1km)
    - Target resolution (30m)
    - Variable type (smoothly varying vs. patchy)
    
    Example:
    --------
    >>> harmonizer = ScaleHarmonizer(target_resolution=30)
    >>> canopy_prior = PhysicsPriors().get_coefficient('Pct_Canopy')
    >>> corrected = harmonizer.apply_scale_correction(canopy_prior)
    >>> print(f"30m coefficient: {corrected:.3f}")
    """
    
    def __init__(self, target_resolution: float = 30.0):
        """
        Initialize the scale harmonizer.
        
        Parameters:
        -----------
        target_resolution : float, default=30.0
            Target spatial resolution in meters
        """
        self.target_resolution = target_resolution
        
        # Scale correction factors based on multi-resolution calibration studies
        # Format: {(source_min, source_max): {'variable': factor}}
        # Factors > 1.0 indicate literature underestimates effect at fine scale
        self._scale_factors = {
            (100, 500): {  # 100m-500m literature
                'Pct_Canopy': 1.15,      # Canopy effects stronger at fine scale
                'Pct_Impervious': 1.12,  # Impervious effects moderately stronger
                'Albedo': 1.05,          # Albedo relatively scale-invariant
                'NDVI': 1.10             # NDVI moderately scale-dependent
            },
            (500, 1000): {  # 500m-1km literature
                'Pct_Canopy': 1.25,      # Larger correction for coarser source
                'Pct_Impervious': 1.20,
                'Albedo': 1.08,
                'NDVI': 1.15
            },
            (30, 100): {  # 30m-100m literature (minimal correction)
                'Pct_Canopy': 1.02,
                'Pct_Impervious': 1.02,
                'Albedo': 1.01,
                'NDVI': 1.02
            }
        }
    
    def _parse_source_scale(self, source_scale: str) -> Tuple[float, float]:
        """
        Parse source scale string to (min, max) resolution in meters.
        
        Parameters:
        -----------
        source_scale : str
            Scale string like '100m-500m', '1km', '30m'
        
        Returns:
        --------
        tuple
            (min_resolution, max_resolution) in meters
        """
        scale = source_scale.lower().replace(' ', '')
        
        # Handle single value (e.g., '100m' → (100, 100))
        if '-' not in scale:
            if 'km' in scale:
                val = float(scale.replace('km', '')) * 1000
            elif 'm' in scale:
                val = float(scale.replace('m', ''))
            else:
                raise ValueError(f"Cannot parse scale: {source_scale}")
            return (val, val)
        
        # Handle range (e.g., '100m-500m' → (100, 500))
        parts = scale.split('-')
        min_val = parts[0]
        max_val = parts[1]
        
        if 'km' in min_val:
            min_res = float(min_val.replace('km', '')) * 1000
        elif 'm' in min_val:
            min_res = float(min_val.replace('m', ''))
        else:
            raise ValueError(f"Cannot parse min scale: {min_val}")
        
        if 'km' in max_val:
            max_res = float(max_val.replace('km', '')) * 1000
        elif 'm' in max_val:
            max_res = float(max_val.replace('m', ''))
        else:
            raise ValueError(f"Cannot parse max scale: {max_val}")
        
        return (min_res, max_res)
    
    def get_scale_factor(self, variable: str, source_scale: str) -> float:
        """
        Get the scale correction factor for a variable and source scale.
        
        Parameters:
        -----------
        variable : str
            Variable name
        source_scale : str
            Source scale string (e.g., '100m-500m')
        
        Returns:
        --------
        float
            Scale correction factor (typically 1.0-1.3)
        """
        source_min, source_max = self._parse_source_scale(source_scale)
        
        # Find matching scale factor bin
        for (scale_min, scale_max), factors in self._scale_factors.items():
            if scale_min <= source_min <= scale_max or scale_min <= source_max <= scale_max:
                if variable in factors:
                    return factors[variable]
                else:
                    warnings.warn(f"No scale factor for {variable}, using 1.0")
                    return 1.0
        
        # If source scale is same or finer than target, no correction needed
        if source_max <= self.target_resolution:
            return 1.0
        
        # Default: conservative 10% correction for unknown scales
        warnings.warn(f"Unknown scale range {source_scale}, applying default 1.10 factor")
        return 1.10
    
    def apply_scale_correction(self, coefficient: PhysicsCoefficient) -> float:
        """
        Apply scale correction to a physics coefficient.
        
        Parameters:
        -----------
        coefficient : PhysicsCoefficient
            Original coefficient from literature
        
        Returns:
        --------
        float
            Scale-corrected coefficient for 30m resolution
        """
        factor = self.get_scale_factor(coefficient.variable, coefficient.source_scale)
        corrected = coefficient.coefficient * factor
        
        return corrected
    
    def harmonize_all(self, priors: PhysicsPriors) -> Dict[str, float]:
        """
        Apply scale correction to all coefficients in a PhysicsPriors object.
        
        Parameters:
        -----------
        priors : PhysicsPriors
            Physics priors object
        
        Returns:
        --------
        dict
            Dictionary mapping variable names to scale-corrected coefficients
        """
        harmonized = {}
        for var, coef in priors.get_all_coefficients().items():
            harmonized[var] = self.apply_scale_correction(coef)
        
        return harmonized
    
    def print_correction_summary(self, priors: PhysicsPriors):
        """Print a summary of scale corrections applied."""
        print("\n" + "="*80)
        print("SCALE HARMONIZATION SUMMARY")
        print(f"Target Resolution: {self.target_resolution}m")
        print("="*80)
        
        for var, coef in priors.get_all_coefficients().items():
            factor = self.get_scale_factor(var, coef.source_scale)
            corrected = self.apply_scale_correction(coef)
            print(f"{var:20s} | Source: {coef.source_scale:15s} | "
                  f"Factor: {factor:.2f} | "
                  f"Original: {coef.coefficient:+.3f} → Corrected: {corrected:+.3f}")
        print("="*80 + "\n")


class CoefficientCalibrator:
    """
    Blend physics priors with local empirical estimates using shrinkage.
    
    While physics priors provide robust baseline expectations, local calibration
    can improve accuracy for city-specific conditions. This class implements
    Bayesian-style shrinkage to blend priors with out-of-fold empirical estimates.
    
    Shrinkage Formula:
    ------------------
    β*_k = w · β_physics + (1 - w) · β_local
    
    where:
    - β_physics: Literature-based prior (after scale correction)
    - β_local: City-specific estimate from out-of-fold data
    - w: Shrinkage weight (default 0.7, favoring physics)
    
    Rationale:
    ----------
    - w=1.0: Pure physics (no local calibration) - conservative
    - w=0.7: Default blend (30% local adaptation)
    - w=0.5: Equal weight to physics and local data
    - w=0.0: Pure local (ignores physics) - risky overfitting
    
    Example:
    --------
    >>> calibrator = CoefficientCalibrator(shrinkage_weight=0.7)
    >>> physics_coef = -0.280
    >>> local_coef = -0.350  # Stronger effect observed locally
    >>> blended = calibrator.calibrate_coefficient('Pct_Canopy', physics_coef, local_coef)
    >>> print(f"Calibrated: {blended:.3f}")  # -0.301 (blended)
    """
    
    def __init__(self, shrinkage_weight: float = 0.7):
        """
        Initialize the coefficient calibrator.
        
        Parameters:
        -----------
        shrinkage_weight : float, default=0.7
            Weight for physics prior (0=pure local, 1=pure physics)
            Recommended range: [0.6, 0.8]
        
        Raises:
        -------
        ValueError
            If shrinkage_weight not in [0, 1]
        """
        if not 0.0 <= shrinkage_weight <= 1.0:
            raise ValueError(f"shrinkage_weight must be in [0, 1], got {shrinkage_weight}")
        
        self.shrinkage_weight = shrinkage_weight
        self.calibrated_coefficients: Dict[str, Dict] = {}
    
    def calibrate_coefficient(
        self, 
        variable: str, 
        physics_prior: float, 
        local_estimate: float,
        local_se: Optional[float] = None
    ) -> float:
        """
        Calibrate a single coefficient using shrinkage.
        
        Parameters:
        -----------
        variable : str
            Variable name
        physics_prior : float
            Physics-based prior coefficient
        local_estimate : float
            Local empirical estimate from data
        local_se : float, optional
            Standard error of local estimate (for adaptive weighting)
        
        Returns:
        --------
        float
            Calibrated coefficient blending physics and local evidence
        """
        # Basic shrinkage
        w = self.shrinkage_weight
        
        # Optional: Adaptive weighting based on local estimate precision
        if local_se is not None and local_se > 0:
            # If local estimate is very uncertain, trust physics more
            # Scale SE to [0, 1] and adjust weight
            se_scaled = min(local_se / abs(physics_prior), 1.0)
            w_adaptive = w + (1 - w) * se_scaled * 0.5  # Increase physics weight
            w = min(w_adaptive, 0.95)  # Never go below 5% local weight
        
        calibrated = w * physics_prior + (1 - w) * local_estimate
        
        # Store calibration metadata
        self.calibrated_coefficients[variable] = {
            'physics_prior': physics_prior,
            'local_estimate': local_estimate,
            'local_se': local_se,
            'shrinkage_weight': w,
            'calibrated': calibrated
        }
        
        return calibrated
    
    def calibrate_all(
        self,
        physics_priors: Dict[str, float],
        local_estimates: Dict[str, float],
        local_ses: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calibrate all coefficients at once.
        
        Parameters:
        -----------
        physics_priors : dict
            Dictionary mapping variables to physics priors
        local_estimates : dict
            Dictionary mapping variables to local estimates
        local_ses : dict, optional
            Dictionary mapping variables to standard errors
        
        Returns:
        --------
        dict
            Dictionary mapping variables to calibrated coefficients
        """
        if local_ses is None:
            local_ses = {}
        
        calibrated = {}
        for var in physics_priors.keys():
            if var not in local_estimates:
                warnings.warn(f"No local estimate for {var}, using pure physics prior")
                calibrated[var] = physics_priors[var]
                continue
            
            se = local_ses.get(var, None)
            calibrated[var] = self.calibrate_coefficient(
                var, physics_priors[var], local_estimates[var], se
            )
        
        return calibrated
    
    def validate_stability(self, max_deviation: float = 0.5) -> bool:
        """
        Validate that calibrated coefficients stay within reasonable bounds.
        
        Parameters:
        -----------
        max_deviation : float, default=0.5
            Maximum allowed ratio |calibrated / physics - 1|
            E.g., 0.5 means calibrated can be 50% larger/smaller than physics
        
        Returns:
        --------
        bool
            True if all coefficients within bounds, False otherwise
        """
        all_valid = True
        for var, calib_data in self.calibrated_coefficients.items():
            physics = calib_data['physics_prior']
            calibrated = calib_data['calibrated']
            
            if physics == 0:
                continue
            
            deviation = abs(calibrated / physics - 1.0)
            if deviation > max_deviation:
                warnings.warn(
                    f"{var}: Calibrated coefficient deviates {deviation:.1%} from physics "
                    f"(>{max_deviation:.1%} threshold). Physics={physics:.3f}, "
                    f"Calibrated={calibrated:.3f}"
                )
                all_valid = False
        
        return all_valid
    
    def get_calibration_summary(self) -> pd.DataFrame:
        """
        Generate a summary table of calibration results.
        
        Returns:
        --------
        pd.DataFrame
            Summary with columns: variable, physics, local, calibrated, delta
        """
        summary_data = []
        for var, data in self.calibrated_coefficients.items():
            summary_data.append({
                'Variable': var,
                'Physics_Prior': data['physics_prior'],
                'Local_Estimate': data['local_estimate'],
                'Shrinkage_Weight': data['shrinkage_weight'],
                'Calibrated': data['calibrated'],
                'Delta_from_Physics': data['calibrated'] - data['physics_prior']
            })
        return pd.DataFrame(summary_data)
    
    def print_summary(self):
        """Print a formatted calibration summary."""
        print("\n" + "="*80)
        print("COEFFICIENT CALIBRATION SUMMARY")
        print(f"Shrinkage Weight: {self.shrinkage_weight:.2f} (toward physics)")
        print("="*80)
        df = self.get_calibration_summary()
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No calibrations performed yet.")
        print("="*80 + "\n")


class PriorSensitivityAnalyzer:
    """
    Perform sensitivity analysis on physics priors (tornado analysis).
    
    This class varies each prior coefficient across its uncertainty range
    (typically ±20%) while holding others constant, then computes the range
    of predicted temperature changes. This identifies which variables have
    the largest impact on predictions.
    
    Tornado Analysis:
    -----------------
    For each variable k:
    1. Set coefficient to min value (prior - 20%)
    2. Predict ΔT for base scenario → ΔT_min
    3. Set coefficient to max value (prior + 20%)
    4. Predict ΔT for base scenario → ΔT_max
    5. Range = ΔT_max - ΔT_min
    
    Variables with larger ranges are more influential/uncertain.
    
    Example:
    --------
    >>> analyzer = PriorSensitivityAnalyzer()
    >>> base_scenario = {'Pct_Canopy': 10, 'Pct_Impervious': -10, 'NDVI': 0.1, 'Albedo': 0.05}
    >>> results = analyzer.run_tornado_analysis(base_scenario)
    >>> analyzer.plot_tornado(results)
    """
    
    def __init__(self, priors: Optional[PhysicsPriors] = None):
        """
        Initialize sensitivity analyzer.
        
        Parameters:
        -----------
        priors : PhysicsPriors, optional
            Physics priors object. If None, creates default priors.
        """
        self.priors = priors if priors is not None else PhysicsPriors()
    
    def compute_delta_temp(
        self, 
        scenario: Dict[str, float], 
        coefficients: Dict[str, float]
    ) -> float:
        """
        Compute predicted temperature change for a scenario.
        
        Parameters:
        -----------
        scenario : dict
            Dictionary mapping variables to their changes
            E.g., {'Pct_Canopy': 10, 'Pct_Impervious': -10}
        coefficients : dict
            Dictionary mapping variables to coefficients
        
        Returns:
        --------
        float
            Predicted temperature change (°F)
        """
        delta_T = 0.0
        
        for var, delta_x in scenario.items():
            if var not in coefficients:
                continue
            
            coef = coefficients[var]
            
            # Adjust delta based on coefficient units
            coef_obj = self.priors.get_coefficient(var)
            units = coef_obj.units
            
            if 'per +10 pp' in units:
                # Coefficient is per 10 percentage points
                delta_T += coef * (delta_x / 10.0)
            elif 'per +0.1' in units:
                # Coefficient is per 0.1 unit
                delta_T += coef * (delta_x / 0.1)
            else:
                # Assume coefficient is per unit
                delta_T += coef * delta_x
        
        return delta_T
    
    def run_tornado_analysis(
        self, 
        base_scenario: Dict[str, float],
        custom_coefficients: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run tornado analysis varying each prior across its uncertainty range.
        
        Parameters:
        -----------
        base_scenario : dict
            Base intervention scenario (changes in each variable)
        custom_coefficients : dict, optional
            Custom coefficient values. If None, uses default priors.
        
        Returns:
        --------
        dict
            Results for each variable:
            {variable: {'min': ΔT_min, 'max': ΔT_max, 'range': range, 'base': ΔT_base}}
        """
        # Get base coefficients
        if custom_coefficients is None:
            base_coefficients = {
                var: self.priors.get_coefficient_value(var)
                for var in self.priors.get_all_coefficients()
            }
        else:
            base_coefficients = custom_coefficients.copy()
        
        # Compute baseline prediction
        baseline_delta_T = self.compute_delta_temp(base_scenario, base_coefficients)
        
        results = {}
        
        for var in base_scenario.keys():
            if var not in base_coefficients:
                continue
            
            # Get uncertainty range for this coefficient
            coef_min, coef_max = self.priors.get_uncertainty_range(var)
            
            # Predict with minimum coefficient
            coefficients_min = base_coefficients.copy()
            coefficients_min[var] = coef_min
            delta_T_min = self.compute_delta_temp(base_scenario, coefficients_min)
            
            # Predict with maximum coefficient
            coefficients_max = base_coefficients.copy()
            coefficients_max[var] = coef_max
            delta_T_max = self.compute_delta_temp(base_scenario, coefficients_max)
            
            results[var] = {
                'base': baseline_delta_T,
                'min': delta_T_min,
                'max': delta_T_max,
                'range': abs(delta_T_max - delta_T_min),
                'coef_base': base_coefficients[var],
                'coef_min': coef_min,
                'coef_max': coef_max
            }
        
        return results
    
    def get_sensitivity_summary(self, results: Dict) -> pd.DataFrame:
        """
        Convert tornado analysis results to a summary table.
        
        Parameters:
        -----------
        results : dict
            Results from run_tornado_analysis()
        
        Returns:
        --------
        pd.DataFrame
            Summary table sorted by sensitivity (range)
        """
        summary_data = []
        for var, data in results.items():
            summary_data.append({
                'Variable': var,
                'Baseline_ΔT': data['base'],
                'Min_ΔT': data['min'],
                'Max_ΔT': data['max'],
                'Range': data['range'],
                'Coef_Base': data['coef_base'],
                'Coef_Range': f"[{data['coef_min']:.3f}, {data['coef_max']:.3f}]"
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Range', ascending=False)
        return df
    
    def print_sensitivity_summary(self, results: Dict):
        """Print formatted sensitivity analysis results."""
        print("\n" + "="*80)
        print("TORNADO SENSITIVITY ANALYSIS")
        print("="*80)
        df = self.get_sensitivity_summary(results)
        print(df.to_string(index=False))
        print("="*80)
        print("Note: Range = |ΔT_max - ΔT_min| when varying coefficient ±20%")
        print("      Larger range = more sensitive to coefficient uncertainty")
        print("="*80 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHYSICS PRIORS MODULE - DEMONSTRATION")
    print("="*80 + "\n")
    
    # 1. Create and display default priors
    priors = PhysicsPriors()
    priors.print_summary()
    
    # 2. Scale harmonization to 30m
    harmonizer = ScaleHarmonizer(target_resolution=30)
    harmonizer.print_correction_summary(priors)
    harmonized = harmonizer.harmonize_all(priors)
    
    # 3. Local calibration example
    print("\n--- Example: Local Calibration ---")
    calibrator = CoefficientCalibrator(shrinkage_weight=0.7)
    
    # Simulated local estimates (could come from GWR out-of-fold)
    local_estimates = {
        'Pct_Canopy': -0.350,      # Stronger local effect
        'Pct_Impervious': 0.310,   # Slightly weaker locally
        'Albedo': -0.240,          # Similar to physics
        'NDVI': -0.290             # Moderately stronger
    }
    
    calibrated = calibrator.calibrate_all(harmonized, local_estimates)
    calibrator.print_summary()
    calibrator.validate_stability(max_deviation=0.3)
    
    # 4. Tornado sensitivity analysis
    print("\n--- Example: Tornado Sensitivity Analysis ---")
    analyzer = PriorSensitivityAnalyzer(priors)
    
    # Base intervention scenario: +10pp canopy, -10pp impervious, +0.1 NDVI, +0.05 albedo
    base_scenario = {
        'Pct_Canopy': 10,
        'Pct_Impervious': -10,
        'NDVI': 0.1,
        'Albedo': 0.05
    }
    
    tornado_results = analyzer.run_tornado_analysis(base_scenario, custom_coefficients=calibrated)
    analyzer.print_sensitivity_summary(tornado_results)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80 + "\n")
