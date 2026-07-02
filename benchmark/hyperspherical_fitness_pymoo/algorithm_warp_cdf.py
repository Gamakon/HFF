#!/usr/bin/env python3
"""
HF1.Warp.CDF Algorithm Implementation

This module implements the HF1.Warp.CDF algorithm using Beta CDF compensation
for concentration of measure in high-dimensional hyperspherical fitness.
"""

import numpy as np
from typing import Optional
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.mating import Mating

from .survival_warp_cdf import HypersphericalFitnessWarpCDFSurvival


class HypersphericalFitnessWarpCDFAlgorithm(GeneticAlgorithm):
    """
    HF1.Warp.CDF Algorithm for multi-objective optimization.
    
    This algorithm implements the HF1.Warp.CDF approach using Beta CDF transformation
    to compensate for concentration of measure in high-dimensional hyperspherical fitness.
    
    Key Features:
    - Beta CDF compensation for dimension-aware fitness
    - Standard genetic operators (crossover, mutation)
    - Built on PyMOO GeneticAlgorithm infrastructure
    - Automatic normalization of objectives
    
    The algorithm transforms raw angular distances to percentile rankings in the
    expected random distribution, providing superior discrimination in high dimensions.
    """
    
    def __init__(self,
                 pop_size: int = 100,
                 normalize_objectives: bool = True,
                 log_file: Optional[str] = None,
                 **kwargs):
        """
        Initialize the HF1.Warp.CDF Algorithm.
        
        Args:
            pop_size: Population size
            normalize_objectives: Whether to normalize objectives in survival
            log_file: Optional Parquet file for logging results
            **kwargs: Additional arguments passed to parent class
        """
        
        # Create HF1.Warp.CDF survival operator
        survival = HypersphericalFitnessWarpCDFSurvival(
            normalize_objectives=normalize_objectives
        )
        
        # Create standard mating operators
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
        self.log_file = log_file
        self.logger = None
        self.normalize_objectives = normalize_objectives
        
        # Performance tracking
        self.warp_fitness_history = []
        self.discrimination_metrics = []
    
    def setup(self, problem, **kwargs):
        """Setup the algorithm with problem-specific configuration."""
        super().setup(problem, **kwargs)
        
        # Initialize GPU if available
        try:
            import hff
            hff.init_gpu()
        except:
            pass  # GPU not available, continue with CPU
        
        # Log configuration
        n_objectives = problem.n_obj
        print(f"HF1.Warp.CDF Algorithm Configuration:")
        print(f"  Problem: {n_objectives} objectives")
        print(f"  Population size: {self.pop_size}")
        print(f"  Normalize objectives: {self.normalize_objectives}")
        print(f"  Beta CDF compensation: Enabled")
    
    def _advance(self, infills=None, **kwargs):
        """Advance one generation of the algorithm."""
        super()._advance(infills, **kwargs)
        self.generation += 1
        
        # Collect performance metrics if population has fitness values
        if hasattr(self.pop, 'fitness') and self.pop.get('fitness') is not None:
            fitness_values = self.pop.get('fitness')
            if fitness_values is not None:
                self.warp_fitness_history.append(fitness_values.copy())
                
                # Calculate discrimination metrics
                from .survival_warp_cdf import HypersphericalFitnessWarpCDFMetrics
                metrics = HypersphericalFitnessWarpCDFMetrics.calculate_discrimination_metrics(fitness_values)
                metrics['generation'] = self.generation
                self.discrimination_metrics.append(metrics)
    
    def get_result(self):
        """Get the final result ensuring proper PyMOO format."""
        result = super().get_result()
        
        # Ensure opt is properly set from population
        if hasattr(self, 'pop') and self.pop is not None:
            F = self.pop.get("F")
            X = self.pop.get("X")
            
            if F is not None and X is not None:
                result.F = F
                result.X = X
                result.opt = self.pop
                result.algorithm = self
                
        return result
    
    def get_discrimination_analysis(self):
        """
        Get comprehensive discrimination analysis for HF1.Warp.CDF.
        
        Returns:
            Dictionary with discrimination analysis data
        """
        if not self.discrimination_metrics:
            return None
            
        analysis = {
            'algorithm': 'HF1.Warp.CDF',
            'generations_analyzed': len(self.discrimination_metrics),
            'final_metrics': self.discrimination_metrics[-1] if self.discrimination_metrics else None,
            'evolution': {
                'mean_fitness': [m['mean'] for m in self.discrimination_metrics],
                'std_fitness': [m['std'] for m in self.discrimination_metrics],
                'fitness_range': [m['range'] for m in self.discrimination_metrics],
                'coefficient_of_variation': [m['coefficient_of_variation'] for m in self.discrimination_metrics]
            }
        }
        
        return analysis


def create_hf1_warp_cdf_algorithm(pop_size: int = 100,
                                  normalize_objectives: bool = True,
                                  **kwargs) -> HypersphericalFitnessWarpCDFAlgorithm:
    """
    Convenience function to create an HF1.Warp.CDF algorithm.
    
    Args:
        pop_size: Population size
        normalize_objectives: Whether to normalize objectives
        **kwargs: Additional arguments
        
    Returns:
        Configured HF1.Warp.CDF algorithm instance
        
    Examples:
        >>> # Basic usage
        >>> algorithm = create_hf1_warp_cdf_algorithm(pop_size=200)
        
        >>> # For high-dimensional problems
        >>> algorithm = create_hf1_warp_cdf_algorithm(
        ...     pop_size=500,
        ...     normalize_objectives=True
        ... )
    """
    return HypersphericalFitnessWarpCDFAlgorithm(
        pop_size=pop_size,
        normalize_objectives=normalize_objectives,
        **kwargs
    )