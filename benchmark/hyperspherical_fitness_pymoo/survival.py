"""
HypersphericalFitnessSurvival

Production survival operator using hyperspherical fitness for selection.
Integrates with Rust hff for high-performance angular fitness calculation.
"""

import numpy as np
from pymoo.core.survival import Survival
from typing import Optional, Any
import sys
import os

# Import hff with proper error handling
try:
    import hff
except ImportError as e:
    raise ImportError(f"Failed to import hff: {e}. Please ensure hff is built and installed.") from e


class HypersphericalFitnessSurvival(Survival):
    """
    Selection operator using hyperspherical angular fitness.
    
    This survival operator implements the HF1 (Hyperspherical Fitness 1) algorithm
    for multi-objective optimization. It uses angular distance from reference points
    to assess solution quality and performs selection accordingly.
    
    Key Features:
    - Batch processing via Rust hff for performance
    - Comprehensive input validation with fail-fast error handling
    - Column-wise normalization for stable numerical computation
    - Full compliance with PyMOO survival interface
    
    Attributes:
        normalize_objectives (bool): Whether to apply column normalization
        min_objectives (int): Minimum required number of objectives (default: 2)
    """
    
    def __init__(self, 
                 normalize_objectives: bool = True,
                 min_objectives: int = 2,
                 north_pole_method: str = "balanced",
                 **kwargs):
        """
        Initialize the HypersphericalFitnessSurvival operator.
        
        Args:
            normalize_objectives: Whether to normalize objectives before HF1 calculation
            min_objectives: Minimum number of objectives required (HF1 needs ≥2)
            north_pole_method: HF1 method - "balanced" for BalancedNorth, "truenorth" for TrueNorth
            **kwargs: Additional arguments passed to parent Survival class
        """
        super().__init__(**kwargs)
        self.normalize_objectives = normalize_objectives
        self.min_objectives = min_objectives
        self.north_pole_method = north_pole_method
        
        # Initialize fitness scores storage for logging
        self.last_fitness_scores = None
        
        # Validate hff availability
        self._validate_hff_core()
    
    def _validate_hff_core(self) -> None:
        """Validate that hff enhanced API is available and functional."""
        try:
            # Initialize GPU if not already done
            gpu_ready = hff.init_gpu()
            if not gpu_ready:
                raise RuntimeError("GPU initialization failed - HF1 requires GPU acceleration")
            
            # Test with minimal data using GPU API
            test_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            # Use GPU-accelerated HF1 implementation
            result = hff.calculate_hyperspherical_fitness_hf1_gpu(
                test_data, 
                self.north_pole_method
            )
            if len(result) != 2:
                raise RuntimeError("hff HF1 function returned unexpected result length")
        except Exception as e:
            raise RuntimeError(f"hff enhanced API validation failed: {e}") from e
    
    def _do(self, problem, pop, *args, n_survive: Optional[int] = None, **kwargs) -> Any:
        """
        Perform selection based on hyperspherical fitness.
        
        This method implements the core HF1 selection logic:
        1. Extract and validate objective values from population
        2. Apply normalization if enabled
        3. Call Rust HF1 function for angular fitness calculation
        4. Sort individuals by angular fitness (lower = better)
        5. Return top n_survive individuals
        
        Args:
            problem: The optimization problem instance
            pop: Population to select from
            n_survive: Number of individuals to select (default: all)
            **kwargs: Additional keyword arguments
            
        Returns:
            Selected population subset
            
        Raises:
            ValueError: If population or objectives are invalid
            RuntimeError: If Rust HF1 calculation fails
        """
        
        # Handle default n_survive
        if n_survive is None:
            n_survive = len(pop)
            
        # Return early for trivial cases
        if n_survive <= 0:
            return pop[:0]  # Empty population with correct type
        if len(pop) <= n_survive:
            return pop  # No selection needed
            
        # Extract objectives with comprehensive validation
        F = self._extract_and_validate_objectives(pop)
        
        # Apply normalization if enabled
        if self.normalize_objectives:
            F_processed = self._column_normalize(F)
        else:
            F_processed = F
            
        # Ensure optimal memory layout for Rust
        F_contiguous = np.ascontiguousarray(F_processed, dtype=np.float64)
        
        # Call Rust HF1 function with error handling
        hf1_scores = self._calculate_hf1_scores(F_contiguous)
        
        # Validate HF1 results
        self._validate_hf1_results(hf1_scores, pop)
        
        # Store fitness scores for logging/analysis
        self.last_fitness_scores = hf1_scores.copy()
        
        # Perform selection based on angular fitness
        selected_pop = self._select_by_hf1_scores(pop, hf1_scores, n_survive)
        
        return selected_pop
    
    def _extract_and_validate_objectives(self, pop) -> np.ndarray:
        """
        Extract objective values from population with comprehensive validation.
        
        Args:
            pop: Population to extract objectives from
            
        Returns:
            Validated objective array
            
        Raises:
            ValueError: If objectives are invalid in any way
        """
        
        # Extract objectives
        F = pop.get("F")
        if F is None:
            raise ValueError("Population has no objective values (F attribute is None)")
        
        # Check population size
        if len(F) == 0:
            raise ValueError("Population is empty (no individuals)")
            
        # Validate array structure
        if not isinstance(F, np.ndarray):
            raise ValueError(f"Objectives must be numpy array, got {type(F)}")
            
        if F.ndim != 2:
            raise ValueError(f"Expected 2D objective array (n_individuals × n_objectives), got {F.ndim}D")
            
        n_individuals, n_objectives = F.shape
        
        # Validate objective count
        if n_objectives < self.min_objectives:
            raise ValueError(f"HF1 requires at least {self.min_objectives} objectives, got {n_objectives}")
            
        # Check for invalid values
        if not np.all(np.isfinite(F)):
            nan_count = np.sum(np.isnan(F))
            inf_count = np.sum(np.isinf(F))
            raise ValueError(f"Objectives contain invalid values: {nan_count} NaN, {inf_count} Inf")
        
        # Validate data type
        if F.dtype != np.float64:
            F = F.astype(np.float64)
            
        return F
    
    def _column_normalize(self, F: np.ndarray) -> np.ndarray:
        """
        Apply column-wise min-max normalization to objectives.
        
        This normalization ensures all objectives are in [0, 1] range,
        which helps with numerical stability in angular calculations.
        
        Args:
            F: Objective array to normalize (n_individuals × n_objectives)
            
        Returns:
            Normalized objective array
            
        Raises:
            ValueError: If normalization cannot be applied
        """
        
        # Calculate column-wise min and max
        col_min = np.min(F, axis=0)
        col_max = np.max(F, axis=0)
        col_range = col_max - col_min
        
        # Handle constant objectives (range = 0)
        zero_range_mask = col_range == 0
        if np.any(zero_range_mask):
            # For constant objectives, set to 0.5 (middle of [0,1])
            F_normalized = F.copy()
            F_normalized[:, zero_range_mask] = 0.5
            
            # Normalize non-constant objectives
            non_zero_mask = ~zero_range_mask
            if np.any(non_zero_mask):
                F_normalized[:, non_zero_mask] = (
                    (F[:, non_zero_mask] - col_min[non_zero_mask]) / 
                    col_range[non_zero_mask]
                )
        else:
            # Standard min-max normalization
            F_normalized = (F - col_min) / col_range
            
        # Validate normalization result
        if not np.all((F_normalized >= 0) & (F_normalized <= 1)):
            raise ValueError("Normalization produced values outside [0,1] range")
            
        return F_normalized
    
    def _calculate_hf1_scores(self, F: np.ndarray) -> np.ndarray:
        """
        Calculate HF1 scores using Rust hff enhanced API with north_pole_method.
        
        Args:
            F: Preprocessed objective array (C-contiguous, float64)
            
        Returns:
            Array of HF1 angular fitness scores
            
        Raises:
            RuntimeError: If Rust calculation fails
        """
        
        try:
            # Use GPU-accelerated HF1 implementation with batching optimization
            # Import the batched calculator from the benchmarking system
            try:
                import sys
                from pathlib import Path
                benchmarking_path = Path(__file__).parent.parent.parent / "experiments" / "benchmarking"
                if str(benchmarking_path) not in sys.path:
                    sys.path.insert(0, str(benchmarking_path))
                from gpu_batching_optimizer import batched_calculate_hf1
                hf1_scores = batched_calculate_hf1(F, self.north_pole_method)
            except (ImportError, Exception):
                # Fallback to direct GPU call if batching not available
                hf1_scores = hff.calculate_hyperspherical_fitness_hf1_gpu(
                    F, 
                    self.north_pole_method
                )
            
            return np.array(hf1_scores, dtype=np.float64)
            
        except Exception as e:
            # Provide detailed error context
            error_msg = (
                f"Rust HF1 enhanced calculation failed: {e}\n"
                f"Input shape: {F.shape}\n"
                f"Input dtype: {F.dtype}\n"
                f"Input contiguous: {F.flags['C_CONTIGUOUS']}\n"
                f"Input finite: {np.all(np.isfinite(F))}\n"
                f"North pole method: {self.north_pole_method}"
            )
            raise RuntimeError(error_msg) from e
    
    def _validate_hf1_results(self, hf1_scores: np.ndarray, pop) -> None:
        """
        Validate HF1 calculation results.
        
        Args:
            hf1_scores: HF1 scores from Rust calculation
            pop: Original population
            
        Raises:
            RuntimeError: If HF1 results are invalid
        """
        
        # Check result length
        if len(hf1_scores) != len(pop):
            raise RuntimeError(
                f"HF1 returned {len(hf1_scores)} scores for {len(pop)} individuals"
            )
        
        # Check for invalid values in results
        if not np.all(np.isfinite(hf1_scores)):
            nan_count = np.sum(np.isnan(hf1_scores))
            inf_count = np.sum(np.isinf(hf1_scores))
            raise RuntimeError(
                f"HF1 results contain invalid values: {nan_count} NaN, {inf_count} Inf"
            )
    
    def _select_by_hf1_scores(self, pop, hf1_scores: np.ndarray, n_survive: int):
        """
        Select individuals based on HF1 scores.
        
        Args:
            pop: Population to select from
            hf1_scores: HF1 angular fitness scores (lower = better)
            n_survive: Number of individuals to select
            
        Returns:
            Selected population subset
        """
        
        # Sort by HF1 scores (ascending - lower is better for angular fitness)
        sorted_indices = np.argsort(hf1_scores)
        
        # Select top n_survive individuals
        selected_indices = sorted_indices[:n_survive]
        
        # Return selected population
        return pop[selected_indices]


class HypersphericalFitnessMetrics:
    """
    Utility class for HF1-related metrics and analysis.
    
    Provides additional functionality for analyzing hyperspherical fitness
    results and understanding algorithm behavior.
    """
    
    @staticmethod
    def calculate_diversity_metrics(hf1_scores: np.ndarray) -> dict:
        """
        Calculate diversity metrics for HF1 scores.
        
        Args:
            hf1_scores: Array of HF1 angular fitness scores
            
        Returns:
            Dictionary with diversity metrics
        """
        
        return {
            'mean_hf1': float(np.mean(hf1_scores)),
            'std_hf1': float(np.std(hf1_scores)),
            'min_hf1': float(np.min(hf1_scores)),
            'max_hf1': float(np.max(hf1_scores)),
            'range_hf1': float(np.ptp(hf1_scores)),
            'median_hf1': float(np.median(hf1_scores))
        }
    
    @staticmethod
    def analyze_selection_pressure(hf1_scores: np.ndarray, n_survive: int) -> dict:
        """
        Analyze selection pressure from HF1 scores.
        
        Args:
            hf1_scores: Array of HF1 angular fitness scores
            n_survive: Number of individuals that would survive
            
        Returns:
            Dictionary with selection pressure metrics
        """
        
        sorted_scores = np.sort(hf1_scores)
        
        if n_survive >= len(sorted_scores):
            return {'selection_pressure': 0.0, 'cutoff_score': None}
        
        cutoff_score = sorted_scores[n_survive - 1]
        worst_selected = sorted_scores[n_survive - 1]
        best_rejected = sorted_scores[n_survive] if n_survive < len(sorted_scores) else None
        
        return {
            'selection_pressure': float(np.std(sorted_scores[:n_survive])),
            'cutoff_score': float(cutoff_score),
            'worst_selected': float(worst_selected),
            'best_rejected': float(best_rejected) if best_rejected is not None else None
        }


class HypersphericalFitnessHF3Survival(Survival):
    """
    HF3 (Arctic Circle) Survival operator using overlapping groups and multiple reference points.
    
    This survival operator implements the HF3 algorithm for multi-objective optimization.
    It uses overlapping groups with multiple reference points arranged in an arctic circle
    to assess solution quality and performs selection accordingly.
    
    Key Features:
    - Batch processing via Rust hff with GPU acceleration
    - Dynamic group generation based on sqrt(n_objectives)
    - Overlapping groups for robust fitness calculation
    - Full compliance with PyMOO survival interface
    
    Attributes:
        overlap_factor (float): Overlap factor for group generation
        random_seed (int): Random seed for reproducible groups
        decrowding (bool): Whether to apply decrowding transformation
    """
    
    def __init__(self, 
                 overlap_factor_ratio: float = None,  # DEPRECATED - kept for compatibility
                 random_seed: int = 42,
                 decrowding: bool = False,
                 dynamic_groups: bool = False,
                 overlap_offset: int = 1,  # Controls overlap: overlap = num_groups - overlap_offset
                 **kwargs):
        """
        Initialize the HF3 Survival operator.
        
        Args:
            overlap_factor_ratio: DEPRECATED - Ignored. Now uses smart formula: overlap = num_groups - 1
            random_seed: Random seed for reproducible groups (default: 42)
            decrowding: Whether to apply decrowding transformation (default: False)
            dynamic_groups: Use random group assignment each generation (default: False)
            **kwargs: Additional arguments passed to parent Survival class
        """
        super().__init__(**kwargs)
        self.overlap_factor_ratio = overlap_factor_ratio  # Kept for compatibility but not used
        self.random_seed = random_seed
        self.decrowding = decrowding
        self.overlap_offset = max(0, overlap_offset)  # Ensure non-negative
        self.dynamic_groups = dynamic_groups
        
        # Track generation for dynamic group assignment
        self.current_generation = 0
        
        # Initialize fitness scores storage for logging
        self.last_fitness_scores = None
        
        # Validate hff availability
        self._validate_hff_core()
    
    def _validate_hff_core(self) -> None:
        """Validate that hff batch processing is available and functional."""
        try:
            # Test with minimal data
            test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
            n_objectives = test_data.shape[1]
            num_groups = int(np.floor(np.sqrt(n_objectives)))
            # Smart overlap formula: each objective appears in (num_groups - overlap_offset) groups
            # This ensures maximum diversity while maintaining good coverage
            if num_groups <= 1:
                overlap_factor = 1
            else:
                overlap_factor = max(1, num_groups - self.overlap_offset)
            # Ensure parameters are at least 1
            num_groups = max(1, num_groups)
            overlap_factor = max(1, overlap_factor)
            result = hff.calculate_hff_fitness_batch(
                test_data,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=self.random_seed
            )
            if 'fitness' not in result or len(result['fitness']) != 2:
                raise RuntimeError("hff HF3 batch function returned unexpected result")
        except Exception as e:
            raise RuntimeError(f"hff HF3 validation failed: {e}") from e
    
    def _do(self, problem, pop, *args, n_survive: Optional[int] = None, **kwargs) -> Any:
        """
        Perform selection based on HF3 (Arctic Circle) fitness.
        
        This method implements the core HF3 selection logic:
        1. Extract and validate objective values from population
        2. Calculate dynamic num_groups = floor(sqrt(n_objectives))
        3. Call Rust HF3 batch function with GPU acceleration
        4. Sort individuals by HF3 fitness (lower = better)
        5. Return top n_survive individuals
        
        Args:
            problem: The optimization problem instance
            pop: Population to select from
            n_survive: Number of individuals to select (default: all)
            **kwargs: Additional keyword arguments
            
        Returns:
            Selected population subset
            
        Raises:
            ValueError: If population or objectives are invalid
            RuntimeError: If Rust HF3 calculation fails
        """
        
        # Handle default n_survive
        if n_survive is None:
            n_survive = len(pop)
        
        # Handle edge cases
        if n_survive <= 0:
            raise ValueError(f"n_survive must be positive, got {n_survive}")
        
        if n_survive >= len(pop):
            return pop
        
        # Extract objectives
        F = pop.get("F")
        if F is None:
            raise ValueError("Population has no objective values (F is None)")
        
        # Validate population
        self._validate_population_objectives(F)
        
        # Calculate HF3 fitness using batch processing
        hf3_scores = self._calculate_hf3_fitness_batch(F)
        
        # Validate results
        self._validate_hf3_results(hf3_scores, pop)
        
        # Store scores for logging
        self.last_fitness_scores = hf3_scores
        
        # Select best individuals
        return self._select_by_hf3_scores(pop, hf3_scores, n_survive)
    
    def _validate_population_objectives(self, F: np.ndarray) -> None:
        """Validate population objectives for HF3 calculation."""
        
        if len(F) == 0:
            raise ValueError("Population is empty")
        
        if F.ndim != 2:
            raise ValueError(f"Objectives must be 2D array, got shape {F.shape}")
        
        n_individuals, n_objectives = F.shape
        
        if n_objectives < 2:
            raise ValueError(f"HF3 requires at least 2 objectives, got {n_objectives}")
        
        if not np.all(np.isfinite(F)):
            nan_count = np.sum(np.isnan(F))
            inf_count = np.sum(np.isinf(F))
            raise ValueError(f"Objectives contain invalid values: {nan_count} NaN, {inf_count} Inf")
    
    def _calculate_hf3_fitness_batch(self, F: np.ndarray) -> np.ndarray:
        """Calculate HF3 fitness using Rust batch processing with GPU acceleration."""
        
        try:
            # Ensure proper data format
            F = np.ascontiguousarray(F, dtype=np.float64)
            
            # Calculate dynamic parameters
            n_objectives = F.shape[1]
            num_groups = int(np.floor(np.sqrt(n_objectives)))
            # Smart overlap formula: each objective appears in (num_groups - overlap_offset) groups
            # This ensures maximum diversity while maintaining good coverage
            if num_groups <= 1:
                overlap_factor = 1
            else:
                overlap_factor = max(1, num_groups - self.overlap_offset)
            
            # Ensure num_groups is at least 1
            num_groups = max(1, num_groups)
            # Ensure overlap_factor is at least 1 (but don't cap at num_groups - let Rust handle it)
            overlap_factor = max(1, overlap_factor)
            
            # Determine random seed for group assignment
            if self.dynamic_groups:
                # Use different seed each generation for random group reassignment
                effective_seed = self.random_seed + self.current_generation * 1000
                self.current_generation += 1
            else:
                # Use fixed seed for reproducible group assignment
                effective_seed = self.random_seed
            
            # Call Rust HF3 batch function with dynamic seed support
            if self.dynamic_groups:
                # Pass generation offset for dynamic groups (rHF3)
                result = hff.calculate_hff_fitness_batch(
                    F,
                    algorithm="z3",
                    num_groups=num_groups,
                    overlap_factor=overlap_factor,
                    random_seed=self.random_seed,
                    arctic_generation=self.current_generation  # Use arctic_generation parameter
                    # decrowding parameter handled automatically
                )
            else:
                # Use fixed seed for reproducible groups (HF3)
                result = hff.calculate_hff_fitness_batch(
                    F,
                    algorithm="z3",
                    num_groups=num_groups,
                    overlap_factor=overlap_factor,
                    random_seed=effective_seed
                    # decrowding parameter handled automatically
                )
            
            # Extract fitness scores from result dictionary
            hf3_scores = result['fitness']
            return np.array(hf3_scores, dtype=np.float64)
            
        except Exception as e:
            # Provide detailed error context
            error_msg = (
                f"Rust HF3 batch calculation failed: {e}\n"
                f"Input shape: {F.shape}\n"
                f"Input dtype: {F.dtype}\n"
                f"Input contiguous: {F.flags['C_CONTIGUOUS']}\n"
                f"Input finite: {np.all(np.isfinite(F))}\n"
                f"Num groups: {num_groups}\n"
                f"Overlap factor: {overlap_factor} (ratio: {self.overlap_factor_ratio})\n"
                f"Random seed: {effective_seed} (base: {self.random_seed})\n"
                f"Dynamic groups: {self.dynamic_groups}\n"
                f"Generation: {self.current_generation}"
            )
            raise RuntimeError(error_msg) from e
    
    def _validate_hf3_results(self, hf3_scores: np.ndarray, pop) -> None:
        """Validate HF3 calculation results."""
        
        # Check result length
        if len(hf3_scores) != len(pop):
            raise RuntimeError(
                f"HF3 returned {len(hf3_scores)} scores for {len(pop)} individuals"
            )
        
        # Check for invalid values in results
        if not np.all(np.isfinite(hf3_scores)):
            nan_count = np.sum(np.isnan(hf3_scores))
            inf_count = np.sum(np.isinf(hf3_scores))
            raise RuntimeError(
                f"HF3 results contain invalid values: {nan_count} NaN, {inf_count} Inf"
            )
    
    def _select_by_hf3_scores(self, pop, hf3_scores: np.ndarray, n_survive: int):
        """Select individuals based on HF3 scores."""
        
        # Sort by HF3 scores (ascending - lower is better for angular fitness)
        sorted_indices = np.argsort(hf3_scores)
        
        # Select top n_survive individuals
        selected_indices = sorted_indices[:n_survive]
        
        # Return selected population
        return pop[selected_indices]