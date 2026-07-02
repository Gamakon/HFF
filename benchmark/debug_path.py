#!/usr/bin/env python3
"""
Debug Python path issue
"""

import tempfile
import re
import subprocess
import sys
import os

# Read the base script
with open('run_experiment.py', 'r') as f:
    content = f.read()

# Find the sys.path.append line
import re
path_line_match = re.search(r'sys\.path\.append\([^)]+\)', content)
if path_line_match:
    print(f"Found path line: {path_line_match.group()}")
else:
    print("❌ No sys.path.append line found!")

# Apply modifications
exp_name = "f2_wfg9_2obj"
algorithm = "HF1"
full_exp_name = f"{exp_name}_{algorithm.lower()}"

content = re.sub(r'EXPERIMENT_NAME = "[^"]*"', f'EXPERIMENT_NAME = "{full_exp_name}"', content)
content = re.sub(r'ALGORITHM = "[^"]*"', f'ALGORITHM = "{algorithm}"', content)
content = re.sub(r"SELECTED_PROBLEM = '[^']*'", f"SELECTED_PROBLEM = '{exp_name}'", content)

# Create temp file to examine the path setup
temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
temp_script.write(content)
temp_script.close()

print(f"Created temp script: {temp_script.name}")

# Show the first 500 chars around the path setup
with open(temp_script.name, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines[:20]):
    if 'sys.path' in line or 'benchmark' in line:
        print(f"Line {i+1}: {line.strip()}")

# Test just the path import part
test_script = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
print("Path appended:", str(Path(__file__).parent))
print("Sys path:", sys.path[-3:])

try:
    from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import os
    print("Current working dir:", os.getcwd())
    print("__file__:", __file__)
    print("Parent dir:", str(Path(__file__).parent))
'''

with open('debug_import.py', 'w') as f:
    f.write(test_script)

print("\nRunning path test:")
result = subprocess.run([sys.executable, 'debug_import.py'], 
                       capture_output=True, text=True, cwd=os.getcwd())
print("Return code:", result.returncode)
print("Stdout:", result.stdout)
if result.stderr:
    print("Stderr:", result.stderr)

# Cleanup
os.unlink(temp_script.name)
os.unlink('debug_import.py')