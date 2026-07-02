"""
Hyperspherical Fitness PyMOO Integration Package

This package provides PyMOO-compatible implementations of hyperspherical fitness algorithms
including HF1, HF2, HF3, and the new HF1.Warp.CDF with Beta CDF compensation.
"""

from .algorithm import HypersphericalFitnessAlgorithm, HypersphericalFitnessHF3Algorithm
from .algorithm_warp_cdf import HypersphericalFitnessWarpCDFAlgorithm, create_hf1_warp_cdf_algorithm
from .hf2_algorithm import HypersphericalFitnessHF2Algorithm, create_hf2_algorithm
from .survival import HypersphericalFitnessSurvival, HypersphericalFitnessHF3Survival
from .survival_warp_cdf import HypersphericalFitnessWarpCDFSurvival, HypersphericalFitnessWarpCDFMetrics

__all__ = [
    # Core HF1 Algorithm
    'HypersphericalFitnessAlgorithm',
    
    # HF1.Warp.CDF Algorithm  
    'HypersphericalFitnessWarpCDFAlgorithm',
    'create_hf1_warp_cdf_algorithm',
    'HypersphericalFitnessWarpCDFSurvival',
    'HypersphericalFitnessWarpCDFMetrics',
    
    # HF2 Algorithm
    'HypersphericalFitnessHF2Algorithm', 
    'create_hf2_algorithm',
    
    # HF3 Algorithm
    'HypersphericalFitnessHF3Algorithm',
    
    # Survival Strategies
    'HypersphericalFitnessSurvival',
    'HypersphericalFitnessHF3Survival',
]