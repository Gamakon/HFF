#!/usr/bin/env python3
"""
HF1.Warp.CDF Survival Strategy

This module implements survival selection using HF1 with Beta CDF compensation
for concentration of measure in high-dimensional hyperspherical fitness.
"""

import numpy as np
from pymoo.core.survival import Survival
import hff

class HypersphericalFitnessWarpCDFSurvival(Survival):
    """
    HF1.Warp.CDF survival strategy using Beta CDF dimension compensation.
    
    This implements the HF1.Warp.CDF algorithm that applies Beta CDF transformation
    during fitness calculation to compensate for concentration of measure in 
    high-dimensional hyperspherical fitness.
    
    The transformation maps raw angular distances to percentile rankings in the
    expected random distribution, providing better discrimination in high dimensions.
    """
    
    def __init__(self, normalize_objectives: bool = True):
        """
        Initialize HF1.Warp.CDF survival strategy.
        
        Args:
            normalize_objectives: Whether to normalize objectives before fitness calculation
        """
        super().__init__()
        self.normalize_objectives = normalize_objectives
        
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        """
        Perform HF1.Warp.CDF survival selection.
        
        Args:
            problem: Optimization problem
            pop: Population to select from  
            n_survive: Number of individuals to survive
            
        Returns:
            Selected population with HF1.Warp.CDF fitness
        """
        # Get objective values
        F = pop.get("F")
        n_individuals, n_objectives = F.shape
        
        if n_individuals == 0:
            return pop
            
        # Normalize objectives if requested
        if self.normalize_objectives:
            F_normalized = self._normalize_objectives(F)
        else:
            F_normalized = F.copy()
        
        # Calculate HF1.Warp.CDF fitness using Rust implementation
        try:
            fitness_values = hff.calculate_hyperspherical_fitness_hf1_warp_cdf(
                F_normalized
            )
        except Exception as e:
            # Fallback to regular HF1 if Warp.CDF fails
            print(f"Warning: HF1.Warp.CDF failed ({e}), falling back to HF1")
            fitness_values = hff.calculate_hyperspherical_fitness_hf1_enhanced(
                F_normalized, decrowding=False, north_pole_method="balanced"
            )
        
        # Select best individuals (lower fitness is better for minimization)
        indices = np.argsort(fitness_values)[:n_survive]
        
        # Store fitness values in population for analysis
        selected_pop = pop[indices]
        selected_pop.set("fitness", fitness_values[indices])
        
        return selected_pop
    
    def _normalize_objectives(self, F):
        """
        Normalize objectives column-wise to [0,1] range.
        
        Args:
            F: Objective values array of shape (n_individuals, n_objectives)
            
        Returns:
            Normalized objective values
        """
        F_normalized = F.copy()
        
        for j in range(F.shape[1]):
            col = F[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val > min_val:
                F_normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:
                # Handle constant columns
                F_normalized[:, j] = 0.0
                
        return F_normalized


class HypersphericalFitnessWarpCDFMetrics:
    """Utility class for analyzing HF1.Warp.CDF performance metrics."""
    
    @staticmethod
    def calculate_discrimination_metrics(fitness_values):
        """
        Calculate discrimination quality metrics for fitness values.
        
        Args:
            fitness_values: Array of fitness values
            
        Returns:
            Dictionary with discrimination metrics
        """
        return {
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values),
            'min': np.min(fitness_values),
            'max': np.max(fitness_values),
            'range': np.max(fitness_values) - np.min(fitness_values),
            'coefficient_of_variation': np.std(fitness_values) / np.mean(fitness_values) if np.mean(fitness_values) > 0 else 0,
            'percentiles': {
                '25th': np.percentile(fitness_values, 25),
                '50th': np.percentile(fitness_values, 50), 
                '75th': np.percentile(fitness_values, 75),
                '90th': np.percentile(fitness_values, 90),
                '95th': np.percentile(fitness_values, 95)
            }
        }
    
    @staticmethod
    def compare_discrimination(hf1_fitness, warp_fitness):
        """
        Compare discrimination power between HF1 and HF1.Warp.CDF.
        
        Args:
            hf1_fitness: HF1 fitness values
            warp_fitness: HF1.Warp.CDF fitness values
            
        Returns:
            Comparison metrics dictionary
        """
        hf1_metrics = HypersphericalFitnessWarpCDFMetrics.calculate_discrimination_metrics(hf1_fitness)
        warp_metrics = HypersphericalFitnessWarpCDFMetrics.calculate_discrimination_metrics(warp_fitness)
        
        # Calculate improvement metrics
        std_improvement = (warp_metrics['std'] - hf1_metrics['std']) / hf1_metrics['std'] * 100 if hf1_metrics['std'] > 0 else 0
        range_improvement = (warp_metrics['range'] - hf1_metrics['range']) / hf1_metrics['range'] * 100 if hf1_metrics['range'] > 0 else 0
        
        return {
            'hf1_metrics': hf1_metrics,
            'warp_metrics': warp_metrics,
            'improvements': {
                'std_dev_change_percent': std_improvement,
                'range_change_percent': range_improvement,
                'mean_shift': warp_metrics['mean'] - hf1_metrics['mean']
            }
        }