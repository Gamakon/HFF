#!/usr/bin/env python3
"""
Test DTLZ2 with our algorithms
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm, HypersphericalFitnessHF3Algorithm
from hyperspherical_fitness_pymoo.problems.composable import ComposableBenchmarkProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

def test_dtlz2_algorithms():
    print("🧪 Testing DTLZ2 with all algorithms (100 objectives)...")
    
    # Create DTLZ2 problem
    config = {'dtlz': {'problem': 2, 'n_obj': 100}, 'n_var': 110}
    problem = ComposableBenchmarkProblem(config)
    print(f"✅ DTLZ2 problem: {problem.n_var} vars, {problem.n_obj} objs")
    
    algorithms_to_test = [
        ("HF1", lambda: HypersphericalFitnessAlgorithm(pop_size=20)),
        ("HF3", lambda: HypersphericalFitnessHF3Algorithm(pop_size=20, dynamic_groups=False)),
        ("rHF3", lambda: HypersphericalFitnessHF3Algorithm(pop_size=20, dynamic_groups=True)),
    ]
    
    for alg_name, alg_factory in algorithms_to_test:
        print(f"\n🔬 Testing {alg_name}...")
        try:
            algorithm = alg_factory()
            
            result = minimize(
                problem,
                algorithm,
                termination=get_termination("n_gen", 2),  # Just 2 generations
                seed=42,
                verbose=False
            )
            
            print(f"✅ {alg_name}: {len(result.F)} solutions, objectives range [{result.F.min():.3f}, {result.F.max():.3f}]")
            
        except Exception as e:
            print(f"❌ {alg_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n🎉 All algorithms work with DTLZ2!")
    return True

if __name__ == "__main__":
    success = test_dtlz2_algorithms()
    if not success:
        exit(1)