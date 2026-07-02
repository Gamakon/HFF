"""
GNBG2Wrapper

C++ FFI wrapper for GNBG2 F1-F24 functions integrated with PyMOO problem interface.
This wrapper provides high-performance evaluation of GNBG2 benchmark functions
for the composable benchmark system.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from pymoo.core.problem import Problem

# Import the existing GNBG FFI implementation
_parent_path = Path(__file__).parent.parent.parent.parent / "demo"
sys.path.append(str(_parent_path))

try:
    from gnbg_ffi import GNBG_FFI
except ImportError as e:
    raise ImportError(f"Failed to import GNBG FFI: {e}. Please ensure gnbg_ffi.py is available.") from e


class GNBG2Problem(Problem):
    """
    PyMOO Problem wrapper for individual GNBG2 functions.
    
    This class wraps a single GNBG2 function (F1-F24) as a PyMOO Problem
    for integration with multi-objective optimization algorithms.
    
    Attributes:
        function_id (int): GNBG2 function number (1-24)
        gnbg_ffi (GNBG_FFI): C++ FFI interface to GNBG function
        data_path (str): Path to GNBG data files
    """
    
    def __init__(self, 
                 function_id: int,
                 data_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize GNBG2 problem wrapper.
        
        Args:
            function_id: GNBG function number (1-24)
            data_path: Path to directory containing f*.txt files (default: auto-detect)
            **kwargs: Additional arguments passed to Problem constructor
        """
        
        # Validate function ID
        if not 1 <= function_id <= 24:
            raise ValueError(f"Invalid function_id {function_id}. Must be between 1 and 24.")
        
        self.function_id = function_id
        
        # Auto-detect data path if not provided
        if data_path is None:
            data_path = str(Path(__file__).parent.parent.parent.parent / "demo" / "gnbg_ffi_compiled")
        
        self.data_path = data_path
        
        # Create GNBG FFI instance
        try:
            self.gnbg_ffi = GNBG_FFI(function_id, data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to create GNBG FFI for F{function_id}: {e}") from e
        
        # Initialize PyMOO Problem with GNBG properties
        super().__init__(
            n_var=self.gnbg_ffi.dimension,
            n_obj=1,  # GNBG functions are single-objective
            xl=self.gnbg_ffi.min_coordinate,
            xu=self.gnbg_ffi.max_coordinate,
            **kwargs
        )
        
        # Cache problem metadata
        self._metadata = {
            'function_id': function_id,
            'dimension': self.gnbg_ffi.dimension,
            'optimum_value': self.gnbg_ffi.optimum_value,
            'bounds': (self.gnbg_ffi.min_coordinate, self.gnbg_ffi.max_coordinate),
            'data_path': data_path
        }
    
    def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray]) -> None:
        """
        Evaluate the GNBG2 function for given input(s).
        
        Args:
            x: Input array of shape (n_samples, n_vars)
            out: Output dictionary to store results
        """
        
        # Validate input dimensions
        if x.shape[1] != self.n_var:
            raise ValueError(f"Input dimension {x.shape[1]} doesn't match problem dimension {self.n_var}")
        
        # Evaluate using FFI
        try:
            fitness_values = self.gnbg_ffi.fitness(x)
            
            # Ensure output is 2D for PyMOO (n_samples, n_objectives)
            if fitness_values.ndim == 0:
                fitness_values = np.array([[fitness_values]])
            elif fitness_values.ndim == 1:
                fitness_values = fitness_values.reshape(-1, 1)
                
            out["F"] = fitness_values
            
        except Exception as e:
            raise RuntimeError(f"GNBG F{self.function_id} evaluation failed: {e}") from e
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get problem metadata and statistics."""
        stats = self.gnbg_ffi.get_stats()
        return {**self._metadata, **stats}
    
    def reset_function_evaluations(self) -> None:
        """Reset the function evaluation counter."""
        self.gnbg_ffi.reset_fe()
    
    @property
    def function_evaluations(self) -> int:
        """Current number of function evaluations."""
        return self.gnbg_ffi.fe_count
    
    @property
    def optimum_value(self) -> float:
        """Known optimum value for this function."""
        return self.gnbg_ffi.optimum_value


class GNBG2Wrapper:
    """
    Factory and utility class for GNBG2 benchmark functions.
    
    This class provides convenient methods to create and manage multiple
    GNBG2 function instances for the composable benchmark system.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize GNBG2 wrapper.
        
        Args:
            data_path: Path to directory containing f*.txt files (default: auto-detect)
        """
        
        # Auto-detect data path if not provided
        if data_path is None:
            data_path = str(Path(__file__).parent.parent.parent.parent / "demo" / "gnbg_ffi_compiled")
        
        self.data_path = data_path
        self._function_cache = {}  # Cache for created function instances
    
    def create_problem(self, function_id: int, **kwargs) -> GNBG2Problem:
        """
        Create a GNBG2 problem instance.
        
        Args:
            function_id: GNBG function number (1-24)
            **kwargs: Additional arguments passed to GNBG2Problem
            
        Returns:
            GNBG2Problem instance
        """
        return GNBG2Problem(function_id, self.data_path, **kwargs)
    
    def create_multiple_problems(self, function_ids: List[int], **kwargs) -> List[GNBG2Problem]:
        """
        Create multiple GNBG2 problem instances.
        
        Args:
            function_ids: List of GNBG function numbers (1-24)
            **kwargs: Additional arguments passed to GNBG2Problem
            
        Returns:
            List of GNBG2Problem instances
        """
        return [self.create_problem(fid, **kwargs) for fid in function_ids]
    
    def get_function_info(self, function_id: int) -> Dict[str, Any]:
        """
        Get information about a specific GNBG function without creating full problem.
        
        Args:
            function_id: GNBG function number (1-24)
            
        Returns:
            Dictionary with function metadata
        """
        if function_id not in self._function_cache:
            try:
                gnbg_ffi = GNBG_FFI(function_id, self.data_path)
                info = {
                    'function_id': function_id,
                    'dimension': gnbg_ffi.dimension,
                    'optimum_value': gnbg_ffi.optimum_value,
                    'bounds': (gnbg_ffi.min_coordinate, gnbg_ffi.max_coordinate)
                }
                self._function_cache[function_id] = info
            except Exception as e:
                raise RuntimeError(f"Failed to get info for F{function_id}: {e}") from e
        
        return self._function_cache[function_id].copy()
    
    def validate_functions(self, function_ids: List[int]) -> Dict[int, bool]:
        """
        Validate that the specified functions can be loaded.
        
        Args:
            function_ids: List of function IDs to validate
            
        Returns:
            Dictionary mapping function_id to success status
        """
        results = {}
        for fid in function_ids:
            try:
                self.get_function_info(fid)
                results[fid] = True
            except Exception:
                results[fid] = False
        
        return results
    
    @staticmethod
    def get_all_function_ids() -> List[int]:
        """Get list of all available GNBG2 function IDs."""
        return list(range(1, 25))  # F1 through F24
    
    @staticmethod
    def get_function_groups() -> Dict[str, List[int]]:
        """
        Get GNBG2 functions organized by characteristics.
        
        Returns:
            Dictionary with function groups
        """
        return {
            'unimodal': [1, 2, 3],  # Functions F1-F3 are typically unimodal
            'multimodal': [4, 5, 6, 7, 8, 9, 10],  # Functions F4-F10 are multimodal
            'hybrid': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # Hybrid functions
            'composition': [21, 22, 23, 24],  # Composition functions
            'low_dimension': [1, 2, 3, 4, 5],  # Commonly used for low-D testing
            'high_complexity': [21, 22, 23, 24],  # Most complex functions
            'all': list(range(1, 25))
        }


class GNBG2MultiObjectiveAdapter:
    """
    Adapter to use multiple GNBG2 functions as objectives in multi-objective problems.
    
    This adapter allows combining multiple GNBG2 functions to create multi-objective
    optimization problems for the composable benchmark system.
    """
    
    def __init__(self, 
                 function_ids: List[int],
                 data_path: Optional[str] = None):
        """
        Initialize multi-objective adapter.
        
        Args:
            function_ids: List of GNBG function numbers to use as objectives
            data_path: Path to directory containing f*.txt files
        """
        
        self.function_ids = function_ids
        self.wrapper = GNBG2Wrapper(data_path)
        
        # Create individual problems
        self.problems = self.wrapper.create_multiple_problems(function_ids)
        
        # Validate compatibility
        dimensions = [p.n_var for p in self.problems]
        xl_bounds = [p.xl for p in self.problems]
        xu_bounds = [p.xu for p in self.problems]
        
        if not all(d == dimensions[0] for d in dimensions):
            raise ValueError(f"All GNBG functions must have same dimension. Got: {dimensions}")
        
        # Check bounds using numpy array comparison
        if not all(np.allclose(xl, xl_bounds[0]) for xl in xl_bounds):
            raise ValueError(f"All GNBG functions must have same lower bounds.")
            
        if not all(np.allclose(xu, xu_bounds[0]) for xu in xu_bounds):
            raise ValueError(f"All GNBG functions must have same upper bounds.")
        
        self.n_var = dimensions[0]
        self.n_obj = len(function_ids)
        self.xl = self.problems[0].xl
        self.xu = self.problems[0].xu
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives for given input(s).
        
        Args:
            x: Input array of shape (n_samples, n_vars)
            
        Returns:
            Objective values of shape (n_samples, n_objectives)
        """
        
        # Evaluate each objective
        objectives = []
        for problem in self.problems:
            out = {}
            problem._evaluate(x, out)
            objectives.append(out["F"])
        
        # Stack objectives horizontally
        return np.hstack(objectives)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for all objectives."""
        return {
            'function_ids': self.function_ids,
            'n_objectives': self.n_obj,
            'n_variables': self.n_var,
            'bounds': (self.xl, self.xu),
            'individual_optima': [p.optimum_value for p in self.problems],
            'individual_metadata': [p.get_metadata() for p in self.problems]
        }


# Factory functions for convenience
def create_gnbg2_problem(function_id: int, **kwargs) -> GNBG2Problem:
    """Convenience function to create a single GNBG2 problem."""
    return GNBG2Problem(function_id, **kwargs)


def create_gnbg2_cascade(function_ids: List[int], **kwargs) -> GNBG2MultiObjectiveAdapter:
    """Convenience function to create multi-objective GNBG2 problem."""
    return GNBG2MultiObjectiveAdapter(function_ids, **kwargs)


# For backward compatibility and direct use
class GNBG2_F1(GNBG2Problem):
    """Direct F1 problem class."""
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class GNBG2_F24(GNBG2Problem):
    """Direct F24 problem class."""
    def __init__(self, **kwargs):
        super().__init__(24, **kwargs)