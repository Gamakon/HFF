"""
ParquetLogger

Comprehensive result logging with rich schema for analysis.
This logger captures detailed benchmark results with full objective source breakdown,
population statistics, performance metrics, and metadata for analysis.
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import warnings


class ParquetLogger:
    """
    Production Parquet-based logger for hyperspherical fitness benchmark results.
    
    This logger implements a comprehensive schema for storing:
    - Individual solution data with objective breakdowns
    - Population-level statistics per generation
    - HF1 algorithm-specific metrics
    - Performance timing and metadata
    - Problem configuration and source tracking
    
    Schema Design:
    - Individual-level records (one row per individual per generation)
    - Rich metadata for reproducibility and analysis
    - Efficient columnar storage with compression
    - Schema evolution support for future extensions
    
    Attributes:
        filename (Path): Output Parquet file path
        buffer (list): Buffered records for batch writing
        batch_size (int): Number of records to buffer before writing
        schema (pa.Schema): PyArrow schema definition
        written_batches (int): Number of batches written
        total_records (int): Total records logged
    """
    
    def __init__(self, 
                 filename: Optional[str] = None,
                 batch_size: int = 1000,
                 compression: str = 'snappy'):
        """
        Initialize the Parquet logger with comprehensive schema.
        
        Args:
            filename: Output Parquet file path (auto-generated if None)
            batch_size: Number of records to buffer before writing
            compression: Parquet compression algorithm ('snappy', 'gzip', 'lz4', 'brotli')
        """
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperspherical_fitness_results_{timestamp}.parquet"
            
        self.filename = Path(filename)
        self.batch_size = batch_size
        self.compression = compression
        
        # Initialize buffers and counters
        self.buffer = []
        self.written_batches = 0
        self.total_records = 0
        
        # Define comprehensive schema
        self.schema = self._create_schema()
        
        # Track file state
        self._file_initialized = False
        # Check if file already exists to determine first_write status
        self._first_write = not self.filename.exists()
        
        print(f"📊 ParquetLogger initialized: {self.filename}")
        print(f"   Batch size: {self.batch_size}, Compression: {self.compression}")
        if not self._first_write:
            print(f"   ✅ Appending to existing file")
    
    def _create_schema(self) -> pa.Schema:
        """
        Create comprehensive PyArrow schema for benchmark results.
        
        Returns:
            PyArrow schema with all required fields
        """
        
        return pa.schema([
            # Run identification
            ('run_id', pa.int64()),
            ('experiment_name', pa.string()),
            ('timestamp', pa.timestamp('ns')),
            
            # Algorithm information
            ('algorithm', pa.string()),
            ('algorithm_config', pa.string()),  # JSON serialized
            
            # Problem configuration
            ('problem_type', pa.string()),
            ('problem_config', pa.string()),  # JSON serialized
            ('n_variables', pa.int32()),
            ('n_objectives', pa.int32()),
            ('objective_sources', pa.string()),  # JSON list of sources
            
            # Generation and individual data
            ('generation', pa.int32()),
            ('individual_id', pa.int32()),
            ('population_size', pa.int32()),
            
            # Decision variables (stored as JSON for flexibility)
            ('decision_variables', pa.string()),  # JSON array
            
            # Objective values
            ('objectives', pa.string()),  # JSON array of all objectives
            ('objectives_gnbg2', pa.string()),  # JSON array of GNBG2 objectives
            ('objectives_wfg', pa.string()),    # JSON array of WFG objectives  
            ('objectives_dtlz', pa.string()),   # JSON array of DTLZ objectives
            
            # HF1 algorithm specific metrics
            ('hf1_score', pa.float64()),
            ('hf1_rank', pa.int32()),
            ('angular_fitness', pa.float64()),
            ('normalized_objectives', pa.string()),  # JSON array
            
            # Population statistics (same for all individuals in generation)
            ('pop_mean_hf1', pa.float64()),
            ('pop_std_hf1', pa.float64()),
            ('pop_min_hf1', pa.float64()),
            ('pop_max_hf1', pa.float64()),
            ('pop_diversity_metric', pa.float64()),
            
            # Objective statistics per source
            ('gnbg2_mean', pa.string()),    # JSON array of means
            ('gnbg2_std', pa.string()),     # JSON array of stds
            ('wfg_mean', pa.string()),      # JSON array of means
            ('wfg_std', pa.string()),       # JSON array of stds
            ('dtlz_mean', pa.string()),     # JSON array of means
            ('dtlz_std', pa.string()),      # JSON array of stds
            
            # Performance metrics
            ('generation_time_ms', pa.float64()),
            ('evaluation_time_ms', pa.float64()),
            ('selection_time_ms', pa.float64()),
            ('total_evaluations', pa.int64()),
            
            # Metadata
            ('version', pa.string()),
            ('hostname', pa.string()),
            ('git_commit', pa.string()),
            ('python_version', pa.string()),
            ('numpy_version', pa.string()),
            ('pymoo_version', pa.string()),
        ])
    
    def log_generation(self, 
                      run_id: int,
                      algorithm: str,
                      problem_config: Dict[str, Any],
                      generation: int,
                      population,
                      hf1_scores: Optional[List[float]] = None,
                      timing_data: Optional[Dict[str, float]] = None,
                      experiment_name: str = "benchmark_run",
                      algorithm_config: Optional[Dict[str, Any]] = None,
                      **kwargs):
        """
        Log a complete generation of results with individual-level records.
        
        Args:
            run_id: Unique identifier for this run
            algorithm: Algorithm name (e.g., 'HF1', 'NSGA2')
            problem_config: Problem configuration dictionary
            generation: Generation number
            population: PyMOO population object
            hf1_scores: Optional HF1 scores for each individual
            timing_data: Optional timing measurements
            experiment_name: Human-readable experiment name
            algorithm_config: Algorithm configuration parameters
            **kwargs: Additional metadata
        """
        
        if population is None or len(population) == 0:
            warnings.warn(f"Empty population for generation {generation}")
            return
        
        # Extract population data
        F = population.get("F")  # Objectives
        X = population.get("X")  # Decision variables
        
        if F is None or X is None:
            warnings.warn(f"Missing F or X data for generation {generation}")
            return
        
        n_individuals = len(population)
        timestamp = datetime.now()
        
        # Calculate population statistics
        pop_stats = self._calculate_population_statistics(F, hf1_scores)
        
        # Extract objective breakdowns if available
        objective_breakdown = self._extract_objective_breakdown(problem_config, F)
        
        # Calculate source-specific statistics
        source_stats = self._calculate_source_statistics(objective_breakdown)
        
        # Prepare metadata
        metadata = self._prepare_metadata(algorithm_config, **kwargs)
        
        # Create individual records
        for i in range(n_individuals):
            record = {
                # Run identification
                'run_id': run_id,
                'experiment_name': experiment_name,
                'timestamp': timestamp,
                
                # Algorithm information
                'algorithm': algorithm,
                'algorithm_config': json.dumps(algorithm_config) if algorithm_config else "{}",
                
                # Problem configuration
                'problem_type': problem_config.get('type', 'composable'),
                'problem_config': self._safe_json_dumps(problem_config),
                'n_variables': X.shape[1],
                'n_objectives': F.shape[1],
                'objective_sources': json.dumps(problem_config.get('sources', [])),
                
                # Generation and individual data
                'generation': generation,
                'individual_id': i,
                'population_size': n_individuals,
                
                # Decision variables and objectives
                'decision_variables': json.dumps(X[i].tolist()),
                'objectives': json.dumps(F[i].tolist()),
                'objectives_gnbg2': json.dumps(objective_breakdown.get('gnbg2', [None])[i]),
                'objectives_wfg': json.dumps(objective_breakdown.get('wfg', [None])[i]),
                'objectives_dtlz': json.dumps(objective_breakdown.get('dtlz', [None])[i]),
                
                # HF1 algorithm specific metrics
                'hf1_score': float(hf1_scores[i]) if hf1_scores is not None and i < len(hf1_scores) else None,
                'hf1_rank': None,  # Will be calculated if needed
                'angular_fitness': float(hf1_scores[i]) if hf1_scores is not None and i < len(hf1_scores) else None,
                'normalized_objectives': json.dumps([]),  # TODO: Add if normalization is tracked
                
                # Population statistics (same for all individuals)
                'pop_mean_hf1': pop_stats['mean_hf1'],
                'pop_std_hf1': pop_stats['std_hf1'],
                'pop_min_hf1': pop_stats['min_hf1'],
                'pop_max_hf1': pop_stats['max_hf1'],
                'pop_diversity_metric': pop_stats['diversity'],
                
                # Source-specific statistics
                'gnbg2_mean': json.dumps(source_stats.get('gnbg2_mean', [])),
                'gnbg2_std': json.dumps(source_stats.get('gnbg2_std', [])),
                'wfg_mean': json.dumps(source_stats.get('wfg_mean', [])),
                'wfg_std': json.dumps(source_stats.get('wfg_std', [])),
                'dtlz_mean': json.dumps(source_stats.get('dtlz_mean', [])),
                'dtlz_std': json.dumps(source_stats.get('dtlz_std', [])),
                
                # Performance metrics
                'generation_time_ms': timing_data.get('generation_time_ms', 0.0) if timing_data else 0.0,
                'evaluation_time_ms': timing_data.get('evaluation_time_ms', 0.0) if timing_data else 0.0,
                'selection_time_ms': timing_data.get('selection_time_ms', 0.0) if timing_data else 0.0,
                'total_evaluations': timing_data.get('total_evaluations', 0) if timing_data else 0,
                
                # Metadata
                **metadata
            }
            
            self.buffer.append(record)
        
        # Write batch if buffer is full
        if len(self.buffer) >= self.batch_size:
            self._write_batch()
    
    def _calculate_population_statistics(self, F: np.ndarray, hf1_scores: Optional[List[float]]) -> Dict[str, float]:
        """Calculate population-level statistics."""
        
        stats = {}
        
        if hf1_scores is not None and len(hf1_scores) > 0:
            hf1_array = np.array(hf1_scores)
            stats.update({
                'mean_hf1': float(np.mean(hf1_array)),
                'std_hf1': float(np.std(hf1_array)),
                'min_hf1': float(np.min(hf1_array)),
                'max_hf1': float(np.max(hf1_array)),
                'diversity': float(np.std(hf1_array))  # Simple diversity metric
            })
        else:
            stats.update({
                'mean_hf1': None,
                'std_hf1': None,
                'min_hf1': None,
                'max_hf1': None,
                'diversity': None
            })
        
        return stats
    
    def _extract_objective_breakdown(self, problem_config: Dict[str, Any], F: np.ndarray) -> Dict[str, List]:
        """Extract objectives by source if problem supports it."""
        
        breakdown = {'gnbg2': [], 'wfg': [], 'dtlz': []}
        
        # If problem has objective breakdown method, use it
        if hasattr(problem_config, 'get_objective_breakdown'):
            try:
                breakdown_data = problem_config.get_objective_breakdown(F)
                for source, objectives in breakdown_data.items():
                    if source in breakdown:
                        breakdown[source] = objectives.tolist()
            except Exception:
                # Fall back to empty breakdown
                pass
        
        # Ensure each individual has an entry (even if None)
        n_individuals = F.shape[0]
        for source in breakdown:
            if len(breakdown[source]) == 0:
                breakdown[source] = [None] * n_individuals
        
        return breakdown
    
    def _calculate_source_statistics(self, objective_breakdown: Dict[str, List]) -> Dict[str, List]:
        """Calculate statistics for each objective source."""
        
        stats = {}
        
        for source, objectives_list in objective_breakdown.items():
            if objectives_list and objectives_list[0] is not None:
                try:
                    # Convert to numpy array
                    obj_array = np.array([obj for obj in objectives_list if obj is not None])
                    if obj_array.size > 0 and obj_array.ndim == 2:
                        stats[f'{source}_mean'] = np.mean(obj_array, axis=0).tolist()
                        stats[f'{source}_std'] = np.std(obj_array, axis=0).tolist()
                    else:
                        stats[f'{source}_mean'] = []
                        stats[f'{source}_std'] = []
                except Exception:
                    stats[f'{source}_mean'] = []
                    stats[f'{source}_std'] = []
            else:
                stats[f'{source}_mean'] = []
                stats[f'{source}_std'] = []
        
        return stats
    
    def _prepare_metadata(self, algorithm_config: Optional[Dict], **kwargs) -> Dict[str, str]:
        """Prepare metadata fields."""
        
        import platform
        import sys
        
        try:
            import pymoo
            pymoo_version = pymoo.__version__
        except:
            pymoo_version = "unknown"
        
        metadata = {
            'version': "1.0.0",
            'hostname': platform.node(),
            'git_commit': kwargs.get('git_commit', "unknown"),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'numpy_version': np.__version__,
            'pymoo_version': pymoo_version,
        }
        
        return metadata
    
    def _safe_json_dumps(self, obj: Any) -> str:
        """Safely serialize object to JSON, handling numpy arrays and other non-serializable types."""
        
        def convert_for_json(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, np.integer):
                return int(item)
            elif isinstance(item, np.floating):
                return float(item)
            elif isinstance(item, dict):
                return {k: convert_for_json(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [convert_for_json(i) for i in item]
            else:
                return item
        
        try:
            converted_obj = convert_for_json(obj)
            return json.dumps(converted_obj)
        except Exception:
            # Fall back to string representation
            return str(obj)
    
    def _write_batch(self) -> None:
        """Write buffered records to Parquet file with efficient batching."""
        
        if not self.buffer:
            return
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.buffer)
            
            # Simple approach: read existing data if file exists, then append
            if self.filename.exists() and not self._first_write:
                # Read existing data
                existing_df = pd.read_parquet(self.filename)
                # Combine with new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Write combined data
                combined_df.to_parquet(
                    self.filename,
                    compression=self.compression,
                    index=False,
                    engine='pyarrow'
                )
            else:
                # Write new file
                df.to_parquet(
                    self.filename,
                    compression=self.compression,
                    index=False,
                    engine='pyarrow'
                )
                self._first_write = False
            
            # Update counters
            self.written_batches += 1
            self.total_records += len(self.buffer)
            
            print(f"📊 Wrote batch {self.written_batches}: {len(self.buffer)} records (total: {self.total_records})")
            
            # Clear buffer
            self.buffer.clear()
            
        except Exception as e:
            warnings.warn(f"Failed to write Parquet batch: {e}")
            # Keep buffer for retry
    
    def finalize(self) -> None:
        """Write any remaining buffered data and finalize the file."""
        
        if self.buffer:
            print(f"📊 Finalizing: writing final batch of {len(self.buffer)} records")
            self._write_batch()
        
        print(f"✅ ParquetLogger finalized: {self.total_records} total records in {self.written_batches} batches")
        print(f"   Output file: {self.filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        
        stats = {
            'filename': str(self.filename),
            'total_records': self.total_records,
            'written_batches': self.written_batches,
            'buffered_records': len(self.buffer),
            'batch_size': self.batch_size,
            'compression': self.compression,
            'file_exists': self.filename.exists(),
        }
        
        if self.filename.exists():
            stats['file_size_mb'] = self.filename.stat().st_size / 1024 / 1024
        
        return stats


class ParquetLoggerFactory:
    """Factory for creating ParquetLogger instances with common configurations."""
    
    # Shared database filename
    SHARED_DATABASE = "hf1_benchmark_results.parquet"
    
    @staticmethod
    def get_shared_logger(output_dir: str = "results", **kwargs) -> ParquetLogger:
        """Get the shared ParquetLogger instance (create if doesn't exist)"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Use shared database filename
        filename = output_path / ParquetLoggerFactory.SHARED_DATABASE
        
        return ParquetLogger(filename=str(filename), **kwargs)
    
    @staticmethod
    def create_for_experiment(experiment_name: str, 
                            output_dir: str = "results",
                            **kwargs) -> ParquetLogger:
        """Create logger for a specific experiment - DEPRECATED, use get_shared_logger"""
        
        # Just redirect to shared logger
        return ParquetLoggerFactory.get_shared_logger(output_dir, **kwargs)
    
    @staticmethod
    def create_for_run(run_id: int, 
                      base_dir: str = "results",
                      **kwargs) -> ParquetLogger:
        """Create logger for a specific run ID."""
        
        output_path = Path(base_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"run_{run_id:04d}_{timestamp}.parquet"
        
        return ParquetLogger(filename=str(filename), **kwargs)


# Utility functions
def read_parquet_results(filename: str) -> pd.DataFrame:
    """Read Parquet results file back into DataFrame."""
    return pd.read_parquet(filename)


def analyze_parquet_results(filename: str) -> Dict[str, Any]:
    """Analyze Parquet results and return summary statistics."""
    
    df = pd.read_parquet(filename)
    
    analysis = {
        'total_records': len(df),
        'unique_runs': df['run_id'].nunique(),
        'unique_algorithms': df['algorithm'].unique().tolist(),
        'generation_range': (df['generation'].min(), df['generation'].max()),
        'problem_types': df['problem_type'].unique().tolist(),
        'objective_counts': df['n_objectives'].unique().tolist(),
        'population_sizes': df['population_size'].unique().tolist(),
    }
    
    # Add timing analysis if available
    if 'generation_time_ms' in df.columns:
        analysis['timing'] = {
            'mean_generation_time_ms': df['generation_time_ms'].mean(),
            'total_experiment_time_ms': df['generation_time_ms'].sum(),
        }
    
    return analysis
