"""
ComposableBenchmarkProblem

Multi-source problem composition supporting GNBG2, WFG, and DTLZ problems.
This system allows creating hybrid multi-objective problems by combining
objectives from different benchmark sources.
"""

import numpy as np
from pymoo.core.problem import Problem
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Import the GNBG2 wrapper (legacy)
from .gnbg2_wrapper import GNBG2Wrapper, GNBG2MultiObjectiveAdapter

# Import GNBG-II library (new implementation)
try:
    import sys
    sys.path.append('/Users/andrewmorgan/Dev/minkymorgan/GNBG-II/python')
    from gnbg_gpu.multi_objective import GNBGMultiObjectiveProblem
    GNBG_II_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"GNBG-II library not available: {e}. Falling back to legacy GNBG2 wrapper.")
    GNBG_II_AVAILABLE = False
    GNBGMultiObjectiveProblem = None

# Import PyMOO test problems
try:
    from pymoo.problems import get_problem
    # Test if WFG/DTLZ are available
    test_wfg = get_problem('wfg9', n_obj=2, n_var=10)
    test_dtlz = get_problem('dtlz1', n_obj=2, n_var=10)
    PYMOO_AVAILABLE = True
except ImportError:
    warnings.warn("PyMOO WFG/DTLZ problems not available. Only GNBG2 sources will work.")
    PYMOO_AVAILABLE = False


class ComposableBenchmarkProblem(Problem):
    """
    Composable benchmark problem supporting multiple objective sources.
    
    This class allows creating complex multi-objective problems by combining:
    - GNBG2 functions (F1-F24) via C++ FFI
    - WFG test problems (WFG1-9)
    - DTLZ test problems (DTLZ1-7)
    
    Example configurations:
    
    # Pure GNBG2 with cascading objectives
    config = {'gnbg2': [1, 2, 3, 4, 5], 'n_var': 30}
    
    # GNBG2 + WFG hybrid
    config = {
        'gnbg2': [20, 21],
        'wfg': {'problem': 4, 'n_obj': 3},
        'n_var': 30
    }
    
    # Full hybrid with all sources
    config = {
        'gnbg2': [24],
        'wfg': {'problem': 9, 'n_obj': 5},
        'dtlz': {'problem': 2, 'n_obj': 3},
        'n_var': 30
    }
    
    Attributes:
        config (dict): Problem configuration
        gnbg2_adapter: GNBG2 multi-objective adapter (if GNBG2 objectives)
        wfg_problem: PyMOO WFG problem instance (if WFG objectives)
        dtlz_problem: PyMOO DTLZ problem instance (if DTLZ objectives)
        objective_sources (list): List of source types in evaluation order
        objective_counts (list): Number of objectives from each source
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize composable benchmark problem.
        
        Args:
            config: Configuration dictionary specifying objectives sources
                Required keys:
                - At least one of: 'gnbg2', 'wfg', 'dtlz'
                - 'n_var': Number of decision variables
                
                Optional keys for each source:
                - 'gnbg2': List of function IDs [1-24]
                - 'wfg': {'problem': int (1-9), 'n_obj': int}
                - 'dtlz': {'problem': int (1-7), 'n_obj': int}
                - 'bounds': {'xl': float/array, 'xu': float/array}
        
        Raises:
            ValueError: If configuration is invalid
            ImportError: If required PyMOO problems are not available
        """
        
        self.config = config.copy()
        
        # Validate configuration
        self._validate_config()
        
        # Calculate total objectives and variables
        self.total_objectives = self._calculate_total_objectives()
        n_var = self._calculate_required_variables()
        
        # Determine bounds
        xl, xu = self._calculate_bounds()
        
        # Initialize PyMOO Problem
        super().__init__(
            n_var=n_var,
            n_obj=self.total_objectives,
            xl=xl,
            xu=xu
        )
        
        # Initialize subproblems and track objective sources
        self._setup_subproblems()
        
        # Store metadata for analysis
        self._metadata = {
            'sources': list(self.objective_sources),
            'objective_breakdown': dict(zip(self.objective_sources, self.objective_counts)),
            'total_objectives': self.total_objectives,
            'n_variables': n_var,
            'bounds': (self.xl, self.xu)
        }
    
    def _validate_config(self) -> None:
        """Validate the configuration dictionary."""
        
        # Check for at least one objective source
        # NOTE: WFG temporarily disabled for GNBG-II migration - uncomment when needed
        sources = ['gnbg2', 'gnbg_ii', 'dtlz']  # 'wfg' commented out
        if not any(source in self.config for source in sources):
            raise ValueError(f"At least one objective source must be specified: {sources}")
        
        # Check for required n_var
        if 'n_var' not in self.config:
            raise ValueError("Configuration must specify 'n_var' (number of decision variables)")
        
        if self.config['n_var'] <= 0:
            raise ValueError("'n_var' must be positive")
        
        # Validate GNBG2 configuration
        if 'gnbg2' in self.config:
            gnbg2_config = self.config['gnbg2']
            if not isinstance(gnbg2_config, list) or len(gnbg2_config) == 0:
                raise ValueError("'gnbg2' must be a non-empty list of function IDs")
            
            for fid in gnbg2_config:
                if not isinstance(fid, int) or not 1 <= fid <= 24:
                    raise ValueError(f"GNBG2 function ID {fid} must be integer between 1 and 24")
        
        # Validate GNBG-II configuration
        if 'gnbg_ii' in self.config:
            if not GNBG_II_AVAILABLE:
                raise ImportError("GNBG-II library not available")
            
            gnbg_ii_config = self.config['gnbg_ii']
            if not isinstance(gnbg_ii_config, dict):
                raise ValueError("'gnbg_ii' must be a dictionary with 'function' and 'n_obj'")
            
            if 'function' not in gnbg_ii_config or 'n_obj' not in gnbg_ii_config:
                raise ValueError("GNBG-II config must have 'function' and 'n_obj' keys")
            
            function_id = gnbg_ii_config['function']
            n_obj = gnbg_ii_config['n_obj']
            
            if not isinstance(function_id, int) or not 1 <= function_id <= 24:
                raise ValueError(f"GNBG-II function ID {function_id} must be integer between 1 and 24")
            
            if not isinstance(n_obj, int) or n_obj < 1:
                raise ValueError(f"GNBG-II n_obj {n_obj} must be positive integer")
            
            if n_obj > 500:
                warnings.warn(f"GNBG-II with {n_obj} objectives may have performance issues. Consider n_obj ≤ 500.")
        
        # WFG CONFIGURATION TEMPORARILY DISABLED FOR GNBG-II MIGRATION
        # Uncomment when WFG functionality is needed again
        """
        # Validate WFG configuration
        if 'wfg' in self.config:
            if not PYMOO_AVAILABLE:
                raise ImportError("PyMOO WFG problems not available")
            
            wfg_config = self.config['wfg']
            if not isinstance(wfg_config, dict):
                raise ValueError("'wfg' must be a dictionary with 'problem' and 'n_obj'")
            
            if 'problem' not in wfg_config or 'n_obj' not in wfg_config:
                raise ValueError("WFG config must have 'problem' and 'n_obj' keys")
            
            problem_id = wfg_config['problem']
            n_obj = wfg_config['n_obj']
            
            if not isinstance(problem_id, int) or not 1 <= problem_id <= 9:
                raise ValueError("WFG problem ID must be integer between 1 and 9")
            
            if not isinstance(n_obj, int) or n_obj < 2:
                raise ValueError("WFG n_obj must be integer >= 2")
        """
        # DTLZ CONFIGURATION TEMPORARILY DISABLED FOR GNBG-II MIGRATION
        # Uncomment when DTLZ functionality is needed again
        """
        # Validate DTLZ configuration
        if 'dtlz' in self.config:
            if not PYMOO_AVAILABLE:
                raise ImportError("PyMOO DTLZ problems not available")
            
            dtlz_config = self.config['dtlz']
            if not isinstance(dtlz_config, dict):
                raise ValueError("'dtlz' must be a dictionary with 'problem' and 'n_obj'")
            
            if 'problem' not in dtlz_config or 'n_obj' not in dtlz_config:
                raise ValueError("DTLZ config must have 'problem' and 'n_obj' keys")
            
            problem_id = dtlz_config['problem']
            n_obj = dtlz_config['n_obj']
            
            if not isinstance(problem_id, int) or not 1 <= problem_id <= 7:
                raise ValueError("DTLZ problem ID must be integer between 1 and 7")
            
            if not isinstance(n_obj, int) or n_obj < 2:
                raise ValueError("DTLZ n_obj must be integer >= 2")
        """
        
    def _calculate_total_objectives(self) -> int:
        """Calculate total number of objectives from all sources."""
        total = 0
        
        if 'gnbg2' in self.config:
            total += len(self.config['gnbg2'])
            
        if 'gnbg_ii' in self.config:
            total += self.config['gnbg_ii']['n_obj']
            
        # WFG and DTLZ temporarily disabled for GNBG-II migration
        # Uncomment when needed:
        # if 'wfg' in self.config:
        #     total += self.config['wfg']['n_obj']
        #     
        # if 'dtlz' in self.config:
        #     total += self.config['dtlz']['n_obj']
            
        if total == 0:
            raise ValueError("No objective sources specified in configuration")
            
        return total
    
    def _calculate_required_variables(self) -> int:
        """
        Calculate minimum required decision variables.
        
        Uses the specified n_var, with validation against problem requirements.
        """
        n_var = self.config['n_var']
        
        # Validate against minimum requirements for each source
        min_required = 1
        
        # WFG and DTLZ variable requirements temporarily disabled for GNBG-II migration
        # Uncomment when needed:
        # # WFG problems typically need at least n_obj + k variables
        # if 'wfg' in self.config:
        #     wfg_min = self.config['wfg']['n_obj'] + 20  # k=20 is safer for many objectives
        #     min_required = max(min_required, wfg_min)
        # 
        # # DTLZ problems typically need at least n_obj + k variables  
        # if 'dtlz' in self.config:
        #     dtlz_min = self.config['dtlz']['n_obj'] + 5  # k=5 is common default
        #     min_required = max(min_required, dtlz_min)
        
        # GNBG2 problems have fixed dimension of 30
        if 'gnbg2' in self.config:
            min_required = max(min_required, 30)
        
        # GNBG-II problems recommend n_var >= n_obj * 4
        if 'gnbg_ii' in self.config:
            gnbg_ii_min = self.config['gnbg_ii']['n_obj'] * 4
            min_required = max(min_required, gnbg_ii_min)
        
        if n_var < min_required:
            warnings.warn(f"n_var={n_var} may be too small. Minimum recommended: {min_required}")
        
        return n_var
    
    def _calculate_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate variable bounds for the composable problem.
        
        Returns:
            Tuple of (xl, xu) arrays
        """
        n_var = self.config['n_var']
        
        # Check for custom bounds in config
        if 'bounds' in self.config:
            bounds = self.config['bounds']
            xl = np.full(n_var, bounds.get('xl', 0.0))
            xu = np.full(n_var, bounds.get('xu', 1.0))
            return xl, xu
        
        # Default bounds based on sources
        # GNBG2 and GNBG-II use [-100, 100]
        # WFG and DTLZ bounds temporarily disabled for GNBG-II migration
        if ('gnbg2' in self.config or 'gnbg_ii' in self.config):  # and not ('wfg' in self.config or 'dtlz' in self.config):
            xl = np.full(n_var, -100.0)
            xu = np.full(n_var, 100.0)
        else:
            # Default to GNBG bounds during migration
            xl = np.full(n_var, -100.0)
            xu = np.full(n_var, 100.0)
            # # WFG and DTLZ bounds (commented out for migration):
            # # WFG and DTLZ typically use [0, 1] for normalized variables
            # xl = np.full(n_var, 0.0)
            # xu = np.full(n_var, 1.0)
        
        return xl, xu
    
    def _setup_subproblems(self):
        """Initialize all subproblem instances and track evaluation order."""
        
        self.objective_sources = []
        self.objective_counts = []
        
        # Initialize GNBG2 adapter if specified
        if 'gnbg2' in self.config:
            function_ids = self.config['gnbg2']
            self.gnbg2_adapter = GNBG2MultiObjectiveAdapter(function_ids)
            self.objective_sources.append('gnbg2')
            self.objective_counts.append(len(function_ids))
        else:
            self.gnbg2_adapter = None
        
        # Initialize GNBG-II problem if specified
        if 'gnbg_ii' in self.config:
            gnbg_ii_config = self.config['gnbg_ii']
            
            # Create GNBG-II problem using the new library
            self.gnbg_ii_problem = GNBGMultiObjectiveProblem.custom(
                n_var=self.n_var,
                n_obj=gnbg_ii_config['n_obj'],
                name=f"GF{gnbg_ii_config['function']}.{gnbg_ii_config['n_obj']}obj"
            )
            self.objective_sources.append('gnbg_ii')
            self.objective_counts.append(gnbg_ii_config['n_obj'])
        else:
            self.gnbg_ii_problem = None
        
        # WFG and DTLZ problem initialization temporarily disabled for GNBG-II migration
        # Uncomment when needed:
        """
        # Initialize WFG problem if specified
        if 'wfg' in self.config:
            wfg_config = self.config['wfg']
            
            # Create WFG problem using get_problem
            self.wfg_problem = get_problem(
                f"wfg{wfg_config['problem']}",
                n_var=self.n_var,
                n_obj=wfg_config['n_obj']
            )
            self.objective_sources.append('wfg')
            self.objective_counts.append(wfg_config['n_obj'])
        else:
            self.wfg_problem = None
        
        # Initialize DTLZ problem if specified
        if 'dtlz' in self.config:
            dtlz_config = self.config['dtlz']
            
            # Create DTLZ problem using get_problem
            self.dtlz_problem = get_problem(
                f"dtlz{dtlz_config['problem']}",
                n_var=self.n_var,
                n_obj=dtlz_config['n_obj']
            )
            self.objective_sources.append('dtlz')
            self.objective_counts.append(dtlz_config['n_obj'])
        else:
            self.dtlz_problem = None
        """
        # Set to None during migration
        self.wfg_problem = None
        self.dtlz_problem = None
    
        
    def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray]) -> None:
        """
        Evaluate individuals on all configured objective sources.
        
        Args:
            x: Input array of shape (n_samples, n_vars)
            out: Output dictionary to store results
        """
        
        n_samples = x.shape[0]
        all_objectives = []
        
        # Evaluate each source in order
        for source_type in self.objective_sources:
            
            if source_type == 'gnbg2':
                # GNBG2 requires transformation to [-100, 100] bounds if needed
                if np.all(self.xl == 0.0) and np.all(self.xu == 1.0):
                    # Transform from [0,1] to [-100,100] for GNBG2
                    x_gnbg = x * 200.0 - 100.0
                else:
                    x_gnbg = x
                
                gnbg2_objectives = self.gnbg2_adapter.evaluate(x_gnbg)
                all_objectives.append(gnbg2_objectives)
                
            elif source_type == 'gnbg_ii':
                # GNBG-II expects [-100, 100] bounds and float32 data type
                if np.all(self.xl == 0.0) and np.all(self.xu == 1.0):
                    # Transform from [0,1] to [-100,100] for GNBG-II
                    x_gnbg_ii = x * 200.0 - 100.0
                else:
                    x_gnbg_ii = x
                
                # Ensure float32 for GPU acceleration
                x_gnbg_ii = x_gnbg_ii.astype(np.float32)
                
                # Evaluate GNBG-II problem
                gnbg_ii_objectives = self.gnbg_ii_problem.evaluate(x_gnbg_ii)
                all_objectives.append(gnbg_ii_objectives)
                
            # WFG and DTLZ evaluation temporarily disabled for GNBG-II migration
            # Uncomment when needed:
            """
            elif source_type == 'wfg':
                # WFG problems expect [0,1] bounds
                if np.all(self.xl == -100.0) and np.all(self.xu == 100.0):
                    # Transform from [-100,100] to [0,1] for WFG
                    x_wfg = (x + 100.0) / 200.0
                else:
                    x_wfg = x
                
                # Evaluate WFG problem
                wfg_out = {}
                self.wfg_problem._evaluate(x_wfg, wfg_out)
                all_objectives.append(wfg_out["F"])
                
            elif source_type == 'dtlz':
                # DTLZ problems expect [0,1] bounds
                if np.all(self.xl == -100.0) and np.all(self.xu == 100.0):
                    # Transform from [-100,100] to [0,1] for DTLZ
                    x_dtlz = (x + 100.0) / 200.0
                else:
                    x_dtlz = x
                
                # Evaluate DTLZ problem
                dtlz_out = {}
                self.dtlz_problem._evaluate(x_dtlz, dtlz_out)
                all_objectives.append(dtlz_out["F"])
            """
        
        # Combine all objectives horizontally
        combined_objectives = np.hstack(all_objectives)
        out["F"] = combined_objectives
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about the composable problem."""
        
        metadata = self._metadata.copy()
        
        # Add individual subproblem information
        subproblem_info = {}
        
        if self.gnbg2_adapter is not None:
            subproblem_info['gnbg2'] = self.gnbg2_adapter.get_metadata()
        
        # WFG and DTLZ metadata temporarily disabled for GNBG-II migration
        # Uncomment when needed:
        """
        if self.wfg_problem is not None:
            subproblem_info['wfg'] = {
                'problem_id': self.config['wfg']['problem'],
                'n_obj': self.config['wfg']['n_obj'],
                'n_var': self.wfg_problem.n_var,
                'bounds': (self.wfg_problem.xl, self.wfg_problem.xu)
            }
        
        if self.dtlz_problem is not None:
            subproblem_info['dtlz'] = {
                'problem_id': self.config['dtlz']['problem'],
                'n_obj': self.config['dtlz']['n_obj'],
                'n_var': self.dtlz_problem.n_var,
                'bounds': (self.dtlz_problem.xl, self.dtlz_problem.xu)
            }
        """
        
        metadata['subproblem_details'] = subproblem_info
        metadata['configuration'] = self.config
        
        return metadata
    
    def get_objective_breakdown(self, F: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Break down combined objectives back into source-specific objectives.
        
        Args:
            F: Combined objective array of shape (n_samples, total_objectives)
            
        Returns:
            Dictionary mapping source names to their objective arrays
        """
        
        breakdown = {}
        start_idx = 0
        
        for source_type, count in zip(self.objective_sources, self.objective_counts):
            end_idx = start_idx + count
            breakdown[source_type] = F[:, start_idx:end_idx]
            start_idx = end_idx
        
        return breakdown


class ComposableBenchmarkFactory:
    """
    Factory class for creating common composable benchmark configurations.
    
    Provides convenience methods for generating systematic test configurations
    as specified in the design document.
    """
    
    @staticmethod
    def create_gnbg2_cascade(start_f: int, end_f: int, n_var: int = 30) -> ComposableBenchmarkProblem:
        """
        Create cascading GNBG2 problem (F_start to F_end as objectives).
        
        Args:
            start_f: Starting GNBG2 function ID
            end_f: Ending GNBG2 function ID (inclusive)
            n_var: Number of decision variables
            
        Returns:
            ComposableBenchmarkProblem with GNBG2 cascade
        """
        function_ids = list(range(start_f, end_f + 1))
        config = {
            'gnbg2': function_ids,
            'n_var': n_var
        }
        return ComposableBenchmarkProblem(config)
    
    @staticmethod
    def create_wfg_suite(wfg_id: int, n_obj_list: List[int], n_var: int = 30) -> List[ComposableBenchmarkProblem]:
        """
        Create WFG test suite with varying objectives.
        
        Args:
            wfg_id: WFG problem ID (1-9)
            n_obj_list: List of objective counts to test
            n_var: Number of decision variables
            
        Returns:
            List of ComposableBenchmarkProblem instances
        """
        problems = []
        for n_obj in n_obj_list:
            config = {
                'wfg': {'problem': wfg_id, 'n_obj': n_obj},
                'n_var': n_var
            }
            problems.append(ComposableBenchmarkProblem(config))
        return problems
    
    @staticmethod
    def create_dtlz_suite(dtlz_id: int, n_obj_list: List[int], n_var: int = 30) -> List[ComposableBenchmarkProblem]:
        """
        Create DTLZ test suite with varying objectives.
        
        Args:
            dtlz_id: DTLZ problem ID (1-7)
            n_obj_list: List of objective counts to test
            n_var: Number of decision variables
            
        Returns:
            List of ComposableBenchmarkProblem instances
        """
        problems = []
        for n_obj in n_obj_list:
            config = {
                'dtlz': {'problem': dtlz_id, 'n_obj': n_obj},
                'n_var': n_var
            }
            problems.append(ComposableBenchmarkProblem(config))
        return problems
    
    @staticmethod
    def create_hybrid_problem(gnbg2_ids: List[int], 
                            wfg_config: Optional[Dict] = None,
                            dtlz_config: Optional[Dict] = None,
                            n_var: int = 30) -> ComposableBenchmarkProblem:
        """
        Create hybrid problem combining multiple sources.
        
        Args:
            gnbg2_ids: List of GNBG2 function IDs
            wfg_config: Optional WFG configuration {'problem': int, 'n_obj': int}
            dtlz_config: Optional DTLZ configuration {'problem': int, 'n_obj': int}
            n_var: Number of decision variables
            
        Returns:
            ComposableBenchmarkProblem with hybrid configuration
        """
        config = {
            'gnbg2': gnbg2_ids,
            'n_var': n_var
        }
        
        if wfg_config is not None:
            config['wfg'] = wfg_config
        
        if dtlz_config is not None:
            config['dtlz'] = dtlz_config
        
        return ComposableBenchmarkProblem(config)
    
    @staticmethod
    def create_systematic_test_suite() -> List[ComposableBenchmarkProblem]:
        """
        Create the complete systematic test suite (100+ configurations).
        
        Returns:
            List of ComposableBenchmarkProblem instances covering all systematic tests
        """
        problems = []
        
        # Pure GNBG2 cascading tests (1-24 objectives)
        for end_f in range(1, 25):
            problems.append(ComposableBenchmarkFactory.create_gnbg2_cascade(1, end_f))
        
        # WFG test suite (45 configurations: 9 problems × 5 objective counts)
        for wfg_id in range(1, 10):
            problems.extend(ComposableBenchmarkFactory.create_wfg_suite(wfg_id, [2, 3, 5, 8, 10]))
        
        # DTLZ test suite (28 configurations: 7 problems × 4 objective counts)
        for dtlz_id in range(1, 8):
            problems.extend(ComposableBenchmarkFactory.create_dtlz_suite(dtlz_id, [2, 3, 5, 8]))
        
        # Hybrid combinations (selection of interesting combinations)
        hybrid_configs = [
            # GNBG2 + WFG combinations
            {'gnbg2': [24], 'wfg': {'problem': 4, 'n_obj': 3}},
            {'gnbg2': [23, 24], 'wfg': {'problem': 9, 'n_obj': 5}},
            
            # GNBG2 + DTLZ combinations
            {'gnbg2': [21, 22], 'dtlz': {'problem': 2, 'n_obj': 3}},
            {'gnbg2': [20], 'dtlz': {'problem': 7, 'n_obj': 4}},
            
            # All three sources
            {'gnbg2': [24], 'wfg': {'problem': 1, 'n_obj': 2}, 'dtlz': {'problem': 2, 'n_obj': 2}},
        ]
        
        for config in hybrid_configs:
            config['n_var'] = 30
            problems.append(ComposableBenchmarkProblem(config))
        
        return problems


# Convenience functions for direct use
def create_gnbg2_problem(function_ids: List[int], n_var: int = 30) -> ComposableBenchmarkProblem:
    """Create pure GNBG2 multi-objective problem."""
    config = {'gnbg2': function_ids, 'n_var': n_var}
    return ComposableBenchmarkProblem(config)


def create_wfg_problem(problem_id: int, n_obj: int, n_var: int = 30) -> ComposableBenchmarkProblem:
    """Create pure WFG problem."""
    config = {'wfg': {'problem': problem_id, 'n_obj': n_obj}, 'n_var': n_var}
    return ComposableBenchmarkProblem(config)


def create_dtlz_problem(problem_id: int, n_obj: int, n_var: int = 30) -> ComposableBenchmarkProblem:
    """Create pure DTLZ problem."""
    config = {'dtlz': {'problem': problem_id, 'n_obj': n_obj}, 'n_var': n_var}
    return ComposableBenchmarkProblem(config)
