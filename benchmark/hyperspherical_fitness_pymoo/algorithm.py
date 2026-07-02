"""
HypersphericalFitnessAlgorithm

Production implementation of the hyperspherical fitness algorithm with proper PyMOO integration.
This algorithm uses angular distance for multi-objective optimization selection.
"""

import numpy as np
import time
from typing import Optional, Dict, Any
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.mating import Mating
try:
    from pymoo.util.display import MultiObjectiveOutput
except ImportError:
    # Fallback for different PyMOO versions
    MultiObjectiveOutput = None

from .survival import HypersphericalFitnessSurvival, HypersphericalFitnessHF3Survival


class HypersphericalFitnessAlgorithm(GeneticAlgorithm):
    """
    Hyperspherical Fitness Algorithm for multi-objective optimization.
    
    This algorithm implements the HF1 approach using angular distance from reference points
    for selection. It inherits from GeneticAlgorithm to leverage PyMOO's existing
    infrastructure while using custom hyperspherical fitness survival.
    
    Key Features:
    - Custom HF1 survival strategy via Rust hff
    - Standard genetic operators (crossover, mutation)
    - Comprehensive logging support
    - Proper PyMOO integration ensuring res.F, res.X, res.opt are set
    
    Attributes:
        generation (int): Current generation counter
        logger: Optional Parquet logger for results
        timing_data (dict): Performance timing information
    """
    
    def __init__(self,
                 pop_size: int = 100,
                 log_file: Optional[str] = None,
                 normalize_objectives: bool = True,
                 north_pole_method: str = "balanced",
                 **kwargs):
        """
        Initialize the HypersphericalFitnessAlgorithm.
        
        Args:
            pop_size: Population size
            log_file: Optional Parquet file for logging results
            normalize_objectives: Whether to normalize objectives in survival
            north_pole_method: HF1 method - "balanced" for BalancedNorth, "truenorth" for TrueNorth
            **kwargs: Additional arguments passed to parent class
        """
        
        # Create survival operator with north_pole_method (paper_spec removed)
        survival = HypersphericalFitnessSurvival(
            normalize_objectives=normalize_objectives,
            north_pole_method=north_pole_method
        )
        
        # Create standard mating exactly like working minimal version
        mating = Mating(
            RandomSelection(),
            SBX(prob=0.9, eta=15),
            PM(eta=20)  # Let PM calculate its own probability
        )
        
        # Initialize with minimal configuration - EXACTLY like the working version
        super().__init__(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            mating=mating,
            survival=survival,
            n_offsprings=pop_size,
            **kwargs
        )
        
        # Algorithm-specific attributes after successful initialization
        self.generation = 0
        self.log_file = log_file
        self.logger = None  # Will be initialized when needed
        self.timing_data = {}
        self.normalize_objectives = normalize_objectives
        self.north_pole_method = north_pole_method
        
        # Statistics tracking
        self.hf1_scores_history = []
        self.population_metrics = []


class HypersphericalFitnessHF3Algorithm(GeneticAlgorithm):
    """
    HF3 (Arctic Circle) Algorithm for multi-objective optimization.
    
    This algorithm implements the HF3 approach using overlapping groups and
    multiple reference points arranged in an arctic circle for selection.
    
    Key Features:
    - HF3 survival strategy via Rust hff batch processing
    - Dynamic group generation based on sqrt(n_objectives)
    - GPU-accelerated normalization and fitness calculation
    - Standard genetic operators (crossover, mutation)
    
    Attributes:
        generation (int): Current generation counter
        overlap_factor (float): Overlap factor for group generation
        random_seed (int): Random seed for reproducible groups
        decrowding (bool): Whether to apply decrowding transformation
    """
    
    def __init__(self,
                 pop_size: int = 100,
                 overlap_factor_ratio: float = None,  # DEPRECATED
                 random_seed: int = 42,
                 decrowding: bool = False,
                 dynamic_groups: bool = False,
                 n_objectives: int = None,
                 overlap_offset: int = 1,  # Controls overlap: overlap = num_groups - overlap_offset
                 **kwargs):
        """
        Initialize the HF3 Algorithm.
        
        Args:
            pop_size: Population size
            overlap_factor_ratio: DEPRECATED - Ignored. Now uses smart formula: overlap = num_groups - 1
            random_seed: Random seed for reproducible groups (default: 42)
            decrowding: Whether to apply decrowding transformation (default: False)
            dynamic_groups: Use random group assignment each generation (default: False)
            n_objectives: Number of objectives (used to calculate overlap_factor)
            **kwargs: Additional arguments passed to parent class
        """
        
        # Create HF3 survival operator with specified parameters
        survival = HypersphericalFitnessHF3Survival(
            overlap_factor_ratio=overlap_factor_ratio,
            random_seed=random_seed,
            decrowding=decrowding,
            dynamic_groups=dynamic_groups,
            overlap_offset=overlap_offset
        )
        
        # Create standard mating
        mating = Mating(
            RandomSelection(),
            SBX(prob=0.9, eta=15),
            PM(eta=20)
        )
        
        # Initialize with standard configuration
        super().__init__(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            mating=mating,
            survival=survival,
            n_offsprings=pop_size,
            **kwargs
        )
        
        # Algorithm-specific attributes
        self.generation = 0
        self.overlap_factor_ratio = overlap_factor_ratio
        self.random_seed = random_seed
        self.decrowding = decrowding
        self.dynamic_groups = dynamic_groups
        self.n_objectives = n_objectives
        self.overlap_offset = overlap_offset
        self.logger = None  # Initialize logger attribute
        
        # Statistics tracking
        self.hf3_scores_history = []
        self.population_metrics = []
    
    @staticmethod
    def create_standard(pop_size: int = 100, **kwargs) -> 'HypersphericalFitnessHF3Algorithm':
        """
        Create a standard HF3 algorithm instance with default parameters.
        
        Args:
            pop_size: Population size
            **kwargs: Additional algorithm parameters
            
        Returns:
            Configured HF3 algorithm instance
        """
        return HypersphericalFitnessHF3Algorithm(
            pop_size=pop_size,
            overlap_factor_ratio=None,  # Uses smart formula now
            random_seed=42,
            decrowding=False,
            **kwargs
        )
    
# Removed _initialize_infill to avoid interference with parent initialization
    
# Removed _advance to avoid interference with parent flow
    
    # Removed _collect_hf1_metrics and _log_generation methods for now
    # These will be added back when we implement proper generation tracking
    
    def get_result(self):
        """
        Get the final result ensuring proper PyMOO format.
        
        This ensures res.F, res.X, and res.opt are properly set,
        addressing the issues we saw with custom survival strategies.
        
        Returns:
            Properly formatted PyMOO result object
        """
        
        # Get the standard result
        result = super().get_result()
        
        # Ensure opt is properly set from population
        if hasattr(self, 'pop') and self.pop is not None:
            # Update result with current population data
            F = self.pop.get("F")
            X = self.pop.get("X")
            
            if F is not None and X is not None:
                # Set the result data
                result.F = F
                result.X = X
                
                # Set opt to the entire final population for multi-objective
                result.opt = self.pop
                
                # Update algorithm reference
                result.algorithm = self
                
        return result
    
    def finalize(self):
        """Finalize the algorithm and close any resources."""
        
        # Finalize logging
        if self.logger is not None:
            try:
                self.logger.finalize()
            except Exception as e:
                print(f"⚠️ Logger finalization failed: {e}")
        
        # Call parent finalization
        super().finalize() if hasattr(super(), 'finalize') else None
    
    def get_algorithm_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm performance metrics.
        
        Returns:
            Dictionary with algorithm performance data
        """
        
        metrics = {
            'generations_completed': self.generation,
            'total_evaluations': getattr(getattr(self, 'evaluator', None), 'n_eval', 0),
            'population_metrics_history': self.population_metrics.copy(),
            'timing_data': self.timing_data.copy()
        }
        
        # Add latest HF1 statistics if available
        if self.hf1_scores_history:
            latest_hf1 = self.hf1_scores_history[-1]
            from .survival import HypersphericalFitnessMetrics
            metrics['latest_hf1_metrics'] = HypersphericalFitnessMetrics.calculate_diversity_metrics(latest_hf1)
        
        return metrics


class HypersphericalFitnessAlgorithmFactory:
    """
    Factory class for creating HypersphericalFitnessAlgorithm instances with various configurations.
    """
    
    @staticmethod
    def create_standard(pop_size: int = 100, north_pole_method: str = "balanced", **kwargs) -> HypersphericalFitnessAlgorithm:
        """Create a standard HF1 algorithm configuration."""
        return HypersphericalFitnessAlgorithm(
            pop_size=pop_size,
            normalize_objectives=True,
            north_pole_method=north_pole_method,
            **kwargs
        )
    
    @staticmethod
    def create_for_many_objectives(n_objectives: int, north_pole_method: str = "balanced", **kwargs) -> HypersphericalFitnessAlgorithm:
        """Create HF1 algorithm optimized for many-objective problems."""
        
        # Increase population size for many objectives
        pop_size = max(100, n_objectives * 10)
        
        return HypersphericalFitnessAlgorithm(
            pop_size=pop_size,
            normalize_objectives=True,
            north_pole_method=north_pole_method,
            **kwargs
        )
    
    @staticmethod
    def create_for_benchmarking(log_file: str, north_pole_method: str = "balanced", **kwargs) -> HypersphericalFitnessAlgorithm:
        """Create HF1 algorithm configured for comprehensive benchmarking."""
        return HypersphericalFitnessAlgorithm(
            log_file=log_file,
            normalize_objectives=True,
            north_pole_method=north_pole_method,
            eliminate_duplicates=True,
            **kwargs
        )