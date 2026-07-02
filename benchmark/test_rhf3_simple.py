#!/usr/bin/env python3
"""
Simple test to see if rHF3 can run to completion
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessHF3Algorithm
from hyperspherical_fitness_pymoo.problems.composable import ComposableBenchmarkProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

def test_rhf3():
    print("🧪 Testing rHF3 algorithm execution...")
    
    # Create simple test problem  
    problem_config = {'wfg': {'problem': 9, 'n_obj': 3}, 'n_var': 30}
    problem = ComposableBenchmarkProblem(problem_config)
    print(f"✅ Problem: {problem.n_var} vars, {problem.n_obj} objs")
    
    # Create rHF3 algorithm
    algorithm = HypersphericalFitnessHF3Algorithm(
        pop_size=20,  # Small population for quick test
        overlap_factor_ratio=None,
        random_seed=42,
        decrowding=False,
        dynamic_groups=True,  # Key: dynamic groups for rHF3
        n_objectives=problem.n_obj
    )
    print("✅ rHF3 algorithm created")
    
    # Run optimization for just a few generations
    print("🚀 Running optimization...")
    
    try:
        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", 3),  # Just 3 generations
            seed=42,
            verbose=False
        )
        
        print(f"✅ Optimization completed!")
        print(f"   Final population: {len(result.F)} solutions")
        print(f"   Objectives shape: {result.F.shape}")
        print(f"   Best objectives: {np.min(result.F, axis=0)}")
        print(f"   Algorithm type: {type(algorithm).__name__}")
        
        # Check if survival operator has fitness scores
        if hasattr(algorithm.survival, 'last_fitness_scores'):
            scores = algorithm.survival.last_fitness_scores
            if scores is not None:
                print(f"   HF3 fitness scores: min={np.min(scores):.6f}, mean={np.mean(scores):.6f}")
            else:
                print("   No fitness scores available")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rhf3()
    if success:
        print("\n🎉 rHF3 test PASSED - algorithm works correctly")
    else:
        print("\n💥 rHF3 test FAILED - algorithm has issues")