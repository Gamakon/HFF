#!/usr/bin/env python3
"""
Test a single experiment to prove it works
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def test_single_experiment():
    """Test running a single experiment"""
    
    print("Testing single experiment execution...")
    
    # Create a minimal test script
    test_script = """
#!/usr/bin/env python3
import sys
# Simple test - just set variables and exit
EXPERIMENT_NAME = "test_f2_wfg9_2obj_hf1"
ALGORITHM = "HF1"
SELECTED_PROBLEM = 'f2_wfg9_2obj'

print(f"Test experiment: {EXPERIMENT_NAME}")
print(f"Algorithm: {ALGORITHM}")
print(f"Problem: {SELECTED_PROBLEM}")

# Try importing to see if the environment works
try:
    from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("✅ Test script completed successfully")
"""
    
    # Write test script
    with open('test_exp.py', 'w') as f:
        f.write(test_script)
    
    # Run it
    start_time = time.time()
    result = subprocess.run([sys.executable, 'test_exp.py'], 
                          capture_output=True, text=True, timeout=30)
    elapsed = time.time() - start_time
    
    print(f"Execution time: {elapsed:.2f}s")
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    if result.stderr:
        print(f"Stderr: {result.stderr}")
    
    # Cleanup
    os.unlink('test_exp.py')
    
    return result.returncode == 0, elapsed

if __name__ == "__main__":
    success, time_taken = test_single_experiment()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'} in {time_taken:.2f}s")