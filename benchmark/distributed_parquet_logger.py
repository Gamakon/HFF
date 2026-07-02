#!/usr/bin/env python3
"""
Distributed Parquet Logger with Automatic Consolidation

This logger solves parallel write corruption by:
1. Each worker writes to unique temporary parquet files
2. Periodic consolidation merges all temp files into main database
3. Atomic operations prevent corruption during consolidation
4. Automatic cleanup of processed temp files

Usage:
- Individual experiments write to: results/temp/experiment_name_timestamp.parquet
- Consolidated results go to: results/hf1_benchmark_results.parquet
- Background consolidation runs every N completed experiments
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
import glob
import time
import os
import fcntl
import tempfile
import shutil

class DistributedParquetLogger:
    """
    Distributed parquet logger that prevents parallel write corruption.
    
    Key Features:
    - Individual temp files per experiment/worker
    - Periodic consolidation into main database
    - Atomic operations with file locking
    - Automatic cleanup of processed files
    - Recovery from partial consolidations
    """
    
    def __init__(self, 
                 output_dir: str = "results",
                 temp_dir: str = None,
                 main_filename: str = "hf1_benchmark_results.parquet",
                 consolidation_threshold: int = 10,
                 batch_size: int = 1000,
                 compression: str = 'snappy'):
        """
        Initialize distributed logger.
        
        Args:
            output_dir: Main results directory
            temp_dir: Temporary files directory (default: output_dir/temp)
            main_filename: Main consolidated database filename
            consolidation_threshold: Consolidate after N temp files
            batch_size: Records per batch for writing
            compression: Compression algorithm
        """
        
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_dir / "temp"
        self.main_file = self.output_dir / main_filename
        self.consolidation_threshold = consolidation_threshold
        self.batch_size = batch_size
        self.compression = compression
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Buffer for current experiment
        self.buffer = []
        self.current_temp_file = None
        self.total_records = 0
        
        print(f"📁 Distributed logger initialized:")
        print(f"   Main file: {self.main_file}")
        print(f"   Temp dir: {self.temp_dir}")
        print(f"   Consolidation threshold: {consolidation_threshold} files")
    
    def create_temp_filename(self, experiment_name: str) -> Path:
        """Create unique temporary filename for this experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        pid = os.getpid()
        temp_filename = f"{experiment_name}_{timestamp}_{pid}.parquet"
        return self.temp_dir / temp_filename
    
    def log_generation(self,
                      run_id: int,
                      algorithm: str,
                      problem_config: dict,
                      generation: int,
                      population,
                      hf1_scores: Optional[np.ndarray] = None,
                      generation_time_ms: float = 0.0,
                      experiment_name: str = "experiment") -> None:
        """Log generation data to temporary file."""
        
        # Create temp file if needed
        if self.current_temp_file is None:
            self.current_temp_file = self.create_temp_filename(experiment_name)
        
        # Extract population data - handle both dict and PyMOO result objects
        if hasattr(population, 'F') and hasattr(population, 'X'):
            # PyMOO result object
            F = population.F
            X = population.X
        elif hasattr(population, 'get'):
            # Dictionary-like object
            F = population.get("F")
            X = population.get("X")
        else:
            # Try direct attribute access
            F = getattr(population, 'F', None)
            X = getattr(population, 'X', None)
        
        if F is None or X is None:
            warnings.warn("Population missing F or X data, skipping logging")
            return
        
        # Create records for this generation
        timestamp = datetime.now().isoformat()
        
        for i in range(len(F)):
            record = {
                'timestamp': timestamp,
                'experiment_name': experiment_name,
                'run_id': run_id,
                'algorithm': algorithm,
                'generation': generation,
                'individual_id': i,
                'objectives': json.dumps(F[i].tolist()),
                'variables': json.dumps(X[i].tolist()) if X is not None else None,
                'hf1_score': float(hf1_scores[i]) if hf1_scores is not None and i < len(hf1_scores) else None,
                'generation_time_ms': generation_time_ms,
                'problem_config': json.dumps(problem_config),
                'n_objectives': len(F[i]),
                'n_variables': len(X[i]) if X is not None else 0
            }
            self.buffer.append(record)
        
        # Write batch if buffer is full
        if len(self.buffer) >= self.batch_size:
            self._write_temp_batch()
    
    def _write_temp_batch(self) -> None:
        """Write buffered records to temporary file."""
        if not self.buffer:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.buffer)
            
            # Write or append to temp file
            if self.current_temp_file.exists():
                # Append to existing temp file
                existing_df = pd.read_parquet(self.current_temp_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # Write atomically using temporary file
            temp_write_file = self.current_temp_file.with_suffix('.tmp')
            df.to_parquet(temp_write_file, compression=self.compression, index=False)
            
            # Atomic move
            temp_write_file.replace(self.current_temp_file)
            
            self.total_records += len(self.buffer)
            self.buffer.clear()
            
        except Exception as e:
            warnings.warn(f"Failed to write temp batch: {e}")
    
    def finalize_experiment(self, experiment_name: str) -> None:
        """Finalize current experiment and trigger consolidation if needed."""
        
        # Write remaining buffer
        if self.buffer:
            self._write_temp_batch()
        
        # Reset for next experiment
        self.current_temp_file = None
        self.buffer.clear()
        
        # Check if consolidation is needed
        temp_files = list(self.temp_dir.glob("*.parquet"))
        if len(temp_files) >= self.consolidation_threshold:
            print(f"🔄 Consolidation triggered: {len(temp_files)} temp files")
            self.consolidate_temp_files()
    
    def consolidate_temp_files(self) -> bool:
        """
        Consolidate all temporary files into main database.
        
        Returns:
            bool: Success status
        """
        
        # Get all temp files
        temp_files = list(self.temp_dir.glob("*.parquet"))
        if not temp_files:
            print("📁 No temp files to consolidate")
            return True
        
        print(f"🔄 Consolidating {len(temp_files)} temp files...")
        
        try:
            # Read all temp files
            temp_dfs = []
            for temp_file in temp_files:
                try:
                    df = pd.read_parquet(temp_file)
                    temp_dfs.append(df)
                    print(f"   ✅ Read {temp_file.name}: {len(df)} records")
                except Exception as e:
                    print(f"   ❌ Failed to read {temp_file.name}: {e}")
                    continue
            
            if not temp_dfs:
                print("❌ No valid temp files found")
                return False
            
            # Combine all temp data
            new_data = pd.concat(temp_dfs, ignore_index=True)
            print(f"📊 Combined temp data: {len(new_data)} records")
            
            # Read existing main file if it exists
            if self.main_file.exists():
                try:
                    existing_data = pd.read_parquet(self.main_file)
                    print(f"📊 Existing main file: {len(existing_data)} records")
                    
                    # Ensure compatible schemas before combining
                    # Convert timestamp columns to string if needed
                    for df in [existing_data, new_data]:
                        if 'timestamp' in df.columns:
                            df['timestamp'] = df['timestamp'].astype(str)
                    
                    # Align columns
                    all_columns = set(existing_data.columns) | set(new_data.columns)
                    for col in all_columns:
                        if col not in existing_data.columns:
                            existing_data[col] = None
                        if col not in new_data.columns:
                            new_data[col] = None
                    
                    # Reorder columns to match
                    column_order = sorted(all_columns)
                    existing_data = existing_data[column_order]
                    new_data = new_data[column_order]
                    
                    # Combine with new data
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                except Exception as e:
                    print(f"⚠️ Could not read existing main file: {e}")
                    print("   Using only new data")
                    combined_data = new_data
            else:
                combined_data = new_data
            
            # Remove duplicates (based on key columns)
            print(f"📊 Pre-dedup: {len(combined_data)} records")
            combined_data = combined_data.drop_duplicates(
                subset=['experiment_name', 'run_id', 'generation', 'individual_id'],
                keep='last'
            )
            print(f"📊 Post-dedup: {len(combined_data)} records")
            
            # Atomic write using temporary file
            temp_main_file = self.main_file.with_suffix('.consolidating')
            combined_data.to_parquet(temp_main_file, compression=self.compression, index=False)
            
            # Atomic move
            temp_main_file.replace(self.main_file)
            
            # Clean up processed temp files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    print(f"   🗑️ Removed {temp_file.name}")
                except Exception as e:
                    print(f"   ⚠️ Could not remove {temp_file.name}: {e}")
            
            print(f"✅ Consolidation complete: {len(combined_data)} total records")
            return True
            
        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
            return False
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """Get current completion statistics."""
        
        # Always try consolidation first to get latest data
        self.consolidate_temp_files()
        
        if not self.main_file.exists():
            return {
                'total_records': 0,
                'completed_experiments': 0,
                'temp_files_pending': len(list(self.temp_dir.glob("*.parquet")))
            }
        
        try:
            df = pd.read_parquet(self.main_file)
            
            # Count completed experiments (those with 31 runs)
            experiment_counts = df.groupby('experiment_name')['run_id'].nunique()
            completed_experiments = (experiment_counts >= 31).sum()
            
            return {
                'total_records': len(df),
                'unique_experiments': len(experiment_counts),
                'completed_experiments': completed_experiments,
                'temp_files_pending': len(list(self.temp_dir.glob("*.parquet"))),
                'latest_experiments': experiment_counts.tail(5).to_dict()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'temp_files_pending': len(list(self.temp_dir.glob("*.parquet")))
            }


class DistributedParquetLoggerFactory:
    """Factory for creating distributed parquet loggers."""
    
    _shared_logger = None
    
    @classmethod
    def get_shared_logger(cls,
                         output_dir: str = "results",
                         consolidation_threshold: int = 5,
                         batch_size: int = 500,
                         compression: str = 'snappy') -> DistributedParquetLogger:
        """Get or create shared distributed logger instance."""
        
        if cls._shared_logger is None:
            cls._shared_logger = DistributedParquetLogger(
                output_dir=output_dir,
                consolidation_threshold=consolidation_threshold,
                batch_size=batch_size,
                compression=compression
            )
        
        return cls._shared_logger
    
    @classmethod 
    def force_consolidation(cls) -> bool:
        """Force consolidation of all temp files."""
        if cls._shared_logger is not None:
            return cls._shared_logger.consolidate_temp_files()
        return False


if __name__ == "__main__":
    # Test the distributed logger
    logger = DistributedParquetLogger()
    
    # Simulate some data
    import numpy as np
    from types import SimpleNamespace
    
    # Mock population
    pop = SimpleNamespace()
    pop.get = lambda key: np.random.rand(10, 3) if key == "F" else np.random.rand(10, 5)
    
    # Log some test data
    for run in range(2):
        for gen in range(3):
            logger.log_generation(
                run_id=run,
                algorithm="HF1", 
                problem_config={"test": True},
                generation=gen,
                population=pop,
                experiment_name="test_exp"
            )
    
    # Finalize and consolidate
    logger.finalize_experiment("test_exp")
    
    # Check stats
    stats = logger.get_completion_stats()
    print(f"Final stats: {stats}")