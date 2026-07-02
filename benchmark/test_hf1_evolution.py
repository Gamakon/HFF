#!/usr/bin/env python3
"""
Test HF1 evolution from random to optimized objectives.
This validates that HF1 scores evolve from π/2 toward ~0.2 as expected.
"""

import numpy as np
import sys
import os
from benchmark.hf1_unified_ranking import calculate_hyperspherical_fitness_hf1_python
import hff

def test_hf1_evolution():
    """Test that HF1 scores evolve from random toward optimization"""
    np.random.seed(42)
    
    # Test with 3 objectives (same as classification example)
    n_objectives = 3
    n_solutions = 100
    
    print(f"Testing HF1 evolution with {n_objectives} objectives, {n_solutions} solutions")
    print("=" * 60)
    
    # Random population (widely distributed - should start near π/2)
    random_pop = np.random.uniform(0, 100, (n_solutions, n_objectives))
    
    # Optimized population (partially optimized - some directions better)
    optimized_base = np.random.uniform(0, 50, (n_solutions, n_objectives))
    # Make first objective much better (closer to 0)
    optimized_base[:, 0] = np.random.uniform(0, 5, n_solutions)
    optimized_pop = optimized_base
    
    # Perfect population (all objectives very small - should approach 0)
    perfect_pop = np.random.uniform(0.001, 0.1, (n_solutions, n_objectives))
    
    # Test Python implementation
    print("Python HF1 Implementation:")
    python_random = calculate_hyperspherical_fitness_hf1_python(random_pop)
    python_optimized = calculate_hyperspherical_fitness_hf1_python(optimized_pop)
    python_perfect = calculate_hyperspherical_fitness_hf1_python(perfect_pop)
    
    print(f"Random solutions:    mean={np.mean(python_random):.4f}, std={np.std(python_random):.4f}")
    print(f"Optimized solutions: mean={np.mean(python_optimized):.4f}, std={np.std(python_optimized):.4f}")
    print(f"Perfect solutions:   mean={np.mean(python_perfect):.4f}, std={np.std(python_perfect):.4f}")
    
    # Test Rust implementation
    print("\nRust HF1 Implementation:")
    rust_random_result = hff.calculate_hff_fitness_batch(random_pop, algorithm="z1", parallel=True)
    rust_optimized_result = hff.calculate_hff_fitness_batch(optimized_pop, algorithm="z1", parallel=True)
    rust_perfect_result = hff.calculate_hff_fitness_batch(perfect_pop, algorithm="z1", parallel=True)
    
    # Extract fitness values from result dict
    rust_random = rust_random_result["fitness"]
    rust_optimized = rust_optimized_result["fitness"]
    rust_perfect = rust_perfect_result["fitness"]
    
    print(f"Random solutions:    mean={np.mean(rust_random):.4f}, std={np.std(rust_random):.4f}")
    print(f"Optimized solutions: mean={np.mean(rust_optimized):.4f}, std={np.std(rust_optimized):.4f}")
    print(f"Perfect solutions:   mean={np.mean(rust_perfect):.4f}, std={np.std(rust_perfect):.4f}")
    
    # Verify evolution pattern
    print("\nEvolution Analysis:")
    print(f"π/2 ≈ {np.pi/2:.4f}")
    
    # Check that random solutions start reasonably high (balanced solutions should be mid-range)
    random_mean = np.mean(python_random)
    if random_mean > 0.8:  # Should be reasonably high for random/balanced solutions
        print(f"✓ Random solutions start high ({random_mean:.4f})")
    else:
        print(f"✗ Random solutions unexpectedly low ({random_mean:.4f})")
    
    # Check that optimized solutions improve
    optimized_mean = np.mean(python_optimized)
    if optimized_mean < random_mean:
        print(f"✓ Optimized solutions improve ({optimized_mean:.4f} < {random_mean:.4f})")
    else:
        print(f"✗ Optimized solutions don't improve ({optimized_mean:.4f} >= {random_mean:.4f})")
    
    # Check that perfect solutions are very low
    perfect_mean = np.mean(python_perfect)
    if perfect_mean < 0.3:  # Should be much closer to 0
        print(f"✓ Perfect solutions near zero ({perfect_mean:.4f})")
    else:
        print(f"✗ Perfect solutions not near zero ({perfect_mean:.4f})")
    
    # Verify Python and Rust match
    python_rust_diff = np.abs(np.mean(python_random) - np.mean(rust_random))
    if python_rust_diff < 0.001:
        print(f"✓ Python and Rust implementations match (diff: {python_rust_diff:.6f})")
    else:
        print(f"✗ Python and Rust implementations differ (diff: {python_rust_diff:.6f})")
    
    print("\nConclusion:")
    if (random_mean > 0.8 and optimized_mean < random_mean and 
        perfect_mean < 0.3 and python_rust_diff < 0.001):
        print("✓ HF1 implementation correctly evolves from high values toward 0")
        return True
    else:
        print("✗ HF1 implementation has issues")
        return False

if __name__ == "__main__":
    success = test_hf1_evolution()
    sys.exit(0 if success else 1)