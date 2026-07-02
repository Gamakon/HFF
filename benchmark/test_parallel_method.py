#!/usr/bin/env python3
"""
Test the exact method used by parallel runner
"""

import subprocess
import sys
import time
import os
import tempfile
import re

def create_test_experiment_script():
    """Simulate what parallel runner does"""
    
    # Read the base script
    with open('run_experiment.py', 'r') as f:
        content = f.read()
    
    # Apply the same regex replacements
    exp_name = "f2_wfg9_2obj"
    algorithm = "HF1"
    full_exp_name = f"{exp_name}_{algorithm.lower()}"
    
    content = re.sub(r'EXPERIMENT_NAME = "[^"]*"', f'EXPERIMENT_NAME = "{full_exp_name}"', content)
    content = re.sub(r'ALGORITHM = "[^"]*"', f'ALGORITHM = "{algorithm}"', content)
    content = re.sub(r"SELECTED_PROBLEM = '[^']*'", f"SELECTED_PROBLEM = '{exp_name}'", content)
    
    # Optimize for parallel execution
    content = re.sub(r'VERBOSE_OPTIMIZATION = True', 'VERBOSE_OPTIMIZATION = False', content)
    content = re.sub(r'PRINT_EVERY_N_GENS = \d+', 'PRINT_EVERY_N_GENS = 50', content)
    content = re.sub(r'BATCH_SIZE = \d+', 'BATCH_SIZE = 2000', content)
    
    # Create temp file
    temp_script = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        prefix=f'exp_{full_exp_name}_',
        delete=False
    )
    
    temp_script.write(content)
    temp_script.close()
    
    # Debug: show first 200 chars of the generated script
    print("Generated script preview:")
    with open(temp_script.name, 'r') as f:
        preview = f.read(200)
        print(repr(preview))
    
    return temp_script.name

def test_parallel_method():
    """Test running like parallel runner does"""
    
    print("Creating experiment script like parallel runner...")
    temp_script = create_test_experiment_script()
    
    print(f"Temp script: {temp_script}")
    
    # Set environment like parallel runner
    env = os.environ.copy()
    env['RAYON_NUM_THREADS'] = '4'
    env['OMP_NUM_THREADS'] = '4'
    
    # Run with timeout like parallel runner
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            timeout=60,  # Short timeout for testing
            env=env,
            cwd=os.getcwd()  # Set working directory
        )
        elapsed = time.time() - start_time
        
        print(f"Execution time: {elapsed:.2f}s")
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print("First 500 chars of stdout:")
        print(result.stdout[:500])
        if result.stderr:
            print("First 500 chars of stderr:")
            print(result.stderr[:500])
            
        # Check if parquet file was created
        if os.path.exists('results/hf1_benchmark_results.parquet'):
            print("✅ Parquet file was created!")
            # Check its size
            size = os.path.getsize('results/hf1_benchmark_results.parquet')
            print(f"File size: {size} bytes")
        else:
            print("❌ No parquet file created")
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"❌ Timeout after {elapsed:.2f}s")
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_script):
            os.unlink(temp_script)
    
    return result.returncode == 0

if __name__ == "__main__":
    success = test_parallel_method()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")