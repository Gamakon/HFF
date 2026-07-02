#!/usr/bin/env python3
"""
Debug HF1 implementation with simple test cases
"""

import numpy as np
import sys
from benchmark.hf1_unified_ranking import calculate_hyperspherical_fitness_hf1_python
import hff

def debug_hf1():
    """Debug HF1 with simple test cases"""
    
    # Test 1: Population with diverse points (this is key!)
    print("Test 1: Population with diverse objective values")
    print("=" * 50)
    
    # Create a population with diverse points for proper min-max normalization
    population = np.array([
        [0.1, 0.1],    # Very good on both objectives  
        [1.0, 10.0],   # Good on obj1, bad on obj2
        [10.0, 1.0],   # Bad on obj1, good on obj2
        [10.0, 10.0],  # Bad on both objectives
        [5.0, 5.0]     # Balanced medium solution
    ])
    
    print("Population:")
    for i, point in enumerate(population):
        print(f"  Point {i+1}: {point}")
    
    scores = calculate_hyperspherical_fitness_hf1_python(population)
    
    print("\nHF1 Scores:")
    for i, (point, score) in enumerate(zip(population, scores)):
        print(f"  Point {i+1} {point}: {score:.6f}")
    
    print(f"\nExpected patterns:")
    print(f"- Point 1 (0.1, 0.1) should be best (lowest score)")
    print(f"- Point 4 (10.0, 10.0) should be worst (highest score)")
    print(f"- Balanced point 5 should be in middle range")
    
    # Test 2: Manual calculation for balanced case
    print("\n" + "=" * 50)
    print("Test 2: Manual verification for balanced case")
    
    # For 2D, north pole is (1/√2, 1/√2) ≈ (0.707, 0.707)
    north_pole_2d = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    print(f"North pole (2D): {north_pole_2d}")
    
    # Balanced point (1,1) after min-max normalization should be (1,1) if it's the only point
    # After fractional normalization: (1,1)/2 = (0.5, 0.5)
    # Unit vector: (0.5, 0.5) / ||0.5, 0.5|| = (0.5, 0.5) / (√0.5) ≈ (0.707, 0.707)
    balanced_norm = np.array([0.5, 0.5])
    balanced_unit = balanced_norm / np.linalg.norm(balanced_norm)
    print(f"Balanced (1,1) normalized to unit: {balanced_unit}")
    
    # Angular distance
    dot_product = np.dot(balanced_unit, north_pole_2d)
    angle = np.arccos(np.clip(dot_product, -1, 1))
    print(f"Dot product: {dot_product:.6f}")
    print(f"Manual angle: {angle:.6f} (should be 0 since they're the same!)")
    
    print("\nActual implementation result:")
    single_balanced = np.array([[1.0, 1.0]])
    actual_score = calculate_hyperspherical_fitness_hf1_python(single_balanced)[0]
    print(f"Implementation result: {actual_score:.6f}")

if __name__ == "__main__":
    debug_hf1()