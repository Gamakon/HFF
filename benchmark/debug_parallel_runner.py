#!/usr/bin/env python3
"""
Debug version of parallel runner to see exactly where it fails
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from run_experiments_parallel import ALGORITHMS
from run_full_experiments import generate_experiments, count_completed_experiments

def debug_parallel_runner():
    """Debug the exact parallel runner logic"""
    
    print("🔍 DEBUGGING PARALLEL RUNNER")
    print("=" * 50)
    
    # Step 1: Generate experiments
    EXPERIMENTS = generate_experiments()
    algorithms = ALGORITHMS
    
    print(f"✅ Generated {len(EXPERIMENTS)} experiments")
    print(f"✅ Using {len(algorithms)} algorithms: {algorithms}")
    
    # Step 2: Calculate total experiments (exact same logic)
    NSGA3_OBJECTIVE_LIMIT = 8
    total_experiments = 0
    for exp_name, _, _ in EXPERIMENTS:
        import re
        obj_match = re.search(r'(\d+)obj', exp_name)
        n_objectives = int(obj_match.group(1)) if obj_match else 10
        
        # Count available algorithms for this experiment
        if n_objectives >= NSGA3_OBJECTIVE_LIMIT:
            valid_algs = [a for a in algorithms if a != "NSGA3"]
            total_experiments += len(valid_algs)
        else:
            total_experiments += len(algorithms)
    
    print(f"✅ Total experiments calculated: {total_experiments}")
    
    # Step 3: Check completion
    completed_count, completed_list = count_completed_experiments()
    remaining = total_experiments - completed_count
    
    print(f"✅ Completed count: {completed_count}")
    print(f"✅ Remaining: {remaining}")
    
    # Step 4: Check exit conditions
    if remaining == 0:
        print("❌ EXIT CONDITION 1: remaining == 0")
        print("❌ This causes fake success exit")
        return False
    
    # Step 5: Group experiments
    from run_experiments_parallel import group_experiments_by_complexity
    groups = group_experiments_by_complexity(EXPERIMENTS, algorithms)
    
    print(f"✅ Groups created:")
    for group_name, exps in groups.items():
        print(f"    {group_name}: {len(exps)} experiments")
    
    total_to_run = sum(len(exps) for exps in groups.values())
    print(f"✅ Total to run: {total_to_run}")
    
    if total_to_run == 0:
        print("❌ EXIT CONDITION 2: total_to_run == 0")
        print("❌ This causes fake success exit")
        return False
    
    print("✅ All checks passed - experiments should run!")
    return True

if __name__ == "__main__":
    success = debug_parallel_runner()
    if success:
        print("\n🎉 Parallel runner should work")
    else:
        print("\n💥 Parallel runner will exit with fake success")