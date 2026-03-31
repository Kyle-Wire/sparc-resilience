"""
Configuration loader for intervention modeling.

This module provides utilities to load and manage configuration files
for physics priors and variable constraints.

Author: SPARC Labs
Date: October 9, 2025
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class InterventionConfig:
    """
    Centralized configuration manager for intervention modeling.
    
    Loads and validates configuration from YAML files:
    - priors.yml: Physics coefficients and scale factors
    - caps.yml: Variable bounds and constraints
    
    Example:
    --------
    >>> config = InterventionConfig()
    >>> canopy_bounds = config.get_variable_bounds('Pct_Canopy')
    >>> print(canopy_bounds)  # {'min': 0.0, 'max': 100.0, ...}
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Parameters:
        -----------
        config_dir : Path, optional
            Directory containing config files. If None, uses default location.
        """
        if config_dir is None:
            # Default to interventions/config/ relative to this file
            # This file is IN config/, so parent is interventions/, parent again to get same dir
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        self.priors_config: Dict[str, Any] = {}
        self.caps_config: Dict[str, Any] = {}
        
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files."""
        # Load priors.yml
        priors_path = self.config_dir / 'priors.yml'
        if priors_path.exists():
            with open(priors_path, 'r') as f:
                self.priors_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"priors.yml not found at: {priors_path}")
        
        # Load caps.yml
        caps_path = self.config_dir / 'caps.yml'
        if caps_path.exists():
            with open(caps_path, 'r') as f:
                self.caps_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"caps.yml not found at: {caps_path}")
    
    # === PRIORS CONFIG ACCESS ===
    
    def get_coefficient_config(self, variable: str) -> Dict[str, Any]:
        """
        Get full coefficient configuration for a variable.
        
        Parameters:
        -----------
        variable : str
            Variable name (e.g., 'Pct_Canopy')
        
        Returns:
        --------
        dict
            Coefficient configuration including value, units, uncertainty, etc.
        """
        coeffs = self.priors_config.get('coefficients', {})
        if variable not in coeffs:
            raise KeyError(f"No coefficient config for: {variable}")
        return coeffs[variable]
    
    def get_all_coefficient_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all coefficient configurations."""
        return self.priors_config.get('coefficients', {})
    
    def get_scale_harmonization_config(self) -> Dict[str, Any]:
        """Get scale harmonization configuration."""
        return self.priors_config.get('scale_harmonization', {})
    
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get calibration settings."""
        return self.priors_config.get('calibration', {})
    
    def get_shrinkage_weight(self) -> float:
        """Get default shrinkage weight for calibration."""
        return self.priors_config.get('calibration', {}).get('shrinkage_weight', 0.7)
    
    # === CAPS CONFIG ACCESS ===
    
    def get_variable_bounds(self, variable: str) -> Dict[str, Any]:
        """
        Get min/max bounds for a variable.
        
        Parameters:
        -----------
        variable : str
            Variable name
        
        Returns:
        --------
        dict
            Bounds configuration with min, max, units, etc.
        """
        bounds = self.caps_config.get('variable_bounds', {})
        if variable not in bounds:
            raise KeyError(f"No bounds defined for: {variable}")
        return bounds[variable]
    
    def get_all_variable_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Get all variable bounds."""
        return self.caps_config.get('variable_bounds', {})
    
    def get_delta_constraints(self, variable: str) -> Dict[str, Any]:
        """
        Get intervention delta constraints for a variable.
        
        Parameters:
        -----------
        variable : str
            Variable name
        
        Returns:
        --------
        dict
            Delta constraints (max_increase, max_decrease, etc.)
        """
        deltas = self.caps_config.get('delta_constraints', {})
        if variable not in deltas:
            raise KeyError(f"No delta constraints for: {variable}")
        return deltas[variable]
    
    def get_monotonicity_rule(self, variable: str) -> Dict[str, Any]:
        """
        Get monotonicity rule for a variable.
        
        Parameters:
        -----------
        variable : str
            Variable name
        
        Returns:
        --------
        dict
            Monotonicity rule (expected_sign, description)
        """
        rules = self.caps_config.get('monotonicity', {})
        if variable not in rules:
            raise KeyError(f"No monotonicity rule for: {variable}")
        return rules[variable]
    
    def get_validation_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds."""
        return self.caps_config.get('validation', {})
    
    # === VALIDATION HELPERS ===
    
    def validate_value(self, variable: str, value: float) -> bool:
        """
        Check if a value is within bounds for a variable.
        
        Parameters:
        -----------
        variable : str
            Variable name
        value : float
            Value to validate
        
        Returns:
        --------
        bool
            True if value within bounds, False otherwise
        """
        bounds = self.get_variable_bounds(variable)
        return bounds['min'] <= value <= bounds['max']
    
    def clip_to_bounds(self, variable: str, value: float) -> float:
        """
        Clip a value to variable bounds.
        
        Parameters:
        -----------
        variable : str
            Variable name
        value : float
            Value to clip
        
        Returns:
        --------
        float
            Clipped value
        """
        bounds = self.get_variable_bounds(variable)
        return max(bounds['min'], min(value, bounds['max']))
    
    def get_capacity_headroom(
        self, 
        variable: str, 
        current_value: float,
        intervention_direction: str = 'increase'
    ) -> float:
        """
        Calculate available capacity headroom for an intervention.
        
        Parameters:
        -----------
        variable : str
            Variable name
        current_value : float
            Current value of the variable
        intervention_direction : str
            'increase' or 'decrease'
        
        Returns:
        --------
        float
            Maximum feasible change
        """
        bounds = self.get_variable_bounds(variable)
        delta_constraints = self.get_delta_constraints(variable)
        
        if intervention_direction == 'increase':
            # Limited by upper bound and max_increase
            headroom_to_bound = bounds['max'] - current_value
            max_increase = delta_constraints.get('max_increase', headroom_to_bound)
            return min(headroom_to_bound, max_increase)
        
        elif intervention_direction == 'decrease':
            # Limited by lower bound and max_decrease
            headroom_to_bound = current_value - bounds['min']
            max_decrease = delta_constraints.get('max_decrease', headroom_to_bound)
            return min(headroom_to_bound, max_decrease)
        
        else:
            raise ValueError(f"Invalid direction: {intervention_direction}")
    
    def print_summary(self):
        """Print a summary of loaded configuration."""
        print("\n" + "="*80)
        print("INTERVENTION CONFIGURATION SUMMARY")
        print("="*80)
        
        print("\nPHYSICS PRIORS:")
        for var, config in self.get_all_coefficient_configs().items():
            print(f"  {var:20s}: {config['value']:+.3f} {config['units']}")
        
        print("\nVARIABLE BOUNDS:")
        for var, bounds in self.get_all_variable_bounds().items():
            print(f"  {var:20s}: [{bounds['min']:.2f}, {bounds['max']:.2f}] {bounds['units']}")
        
        print("\nCALIBRATION:")
        calib = self.get_calibration_config()
        print(f"  Shrinkage weight: {calib.get('shrinkage_weight', 'N/A')}")
        print(f"  Max deviation: {calib.get('max_deviation', 'N/A')}")
        
        print("="*80 + "\n")


# Convenience function for quick access
def load_intervention_config(config_dir: Optional[Path] = None) -> InterventionConfig:
    """
    Load intervention configuration.
    
    Parameters:
    -----------
    config_dir : Path, optional
        Configuration directory path
    
    Returns:
    --------
    InterventionConfig
        Loaded configuration object
    """
    return InterventionConfig(config_dir)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTERVENTION CONFIG LOADER - DEMONSTRATION")
    print("="*80 + "\n")
    
    # Load configuration
    config = InterventionConfig()
    config.print_summary()
    
    # Example: Get specific configs
    print("\n--- Example: Canopy Configuration ---")
    canopy_coef = config.get_coefficient_config('Pct_Canopy')
    print(f"Coefficient: {canopy_coef['value']} {canopy_coef['units']}")
    print(f"Source: {canopy_coef['literature_source']}")
    
    canopy_bounds = config.get_variable_bounds('Pct_Canopy')
    print(f"Bounds: [{canopy_bounds['min']}, {canopy_bounds['max']}] {canopy_bounds['units']}")
    
    # Example: Calculate headroom
    current_canopy = 25.0  # Current: 25%
    headroom = config.get_capacity_headroom('Pct_Canopy', current_canopy, 'increase')
    print(f"Current canopy: {current_canopy}%")
    print(f"Maximum feasible increase: {headroom}pp")
    
    # Example: Validate values
    print("\n--- Example: Value Validation ---")
    test_values = [50.0, 105.0, -5.0]
    for val in test_values:
        valid = config.validate_value('Pct_Canopy', val)
        clipped = config.clip_to_bounds('Pct_Canopy', val)
        print(f"  Value {val:6.1f} → Valid: {str(valid):5s} | Clipped: {clipped:6.1f}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80 + "\n")
