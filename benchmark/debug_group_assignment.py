#!/usr/bin/env python3
"""
Debug why groups are identical regardless of seed
"""

def simulate_group_generation(n_objectives, num_groups, overlap_factor, seed):
    """Simulate the Rust group generation logic."""
    import random
    
    random.seed(seed)
    groups = [[] for _ in range(num_groups)]
    
    for obj_idx in range(n_objectives):
        # Create list of all possible groups and shuffle
        available_groups = list(range(num_groups))
        random.shuffle(available_groups)
        
        # Take min(overlap_factor, num_groups)
        actual_assignments = min(overlap_factor, num_groups)
        
        for i in range(actual_assignments):
            group_idx = available_groups[i]
            groups[group_idx].append(obj_idx)
    
    return groups

def test_group_variations():
    """Test different parameter combinations to see when groups vary."""
    
    print("🔬 Testing group generation with different parameters...\n")
    
    test_cases = [
        # (n_obj, n_groups, overlap, description)
        (8, 2, 5, "Original test case"),
        (8, 2, 1, "Minimal overlap"),
        (8, 4, 2, "More groups, less overlap"),
        (10, 3, 2, "Different configuration"),
        (10, 5, 3, "Even more groups"),
    ]
    
    for n_obj, n_groups, overlap, desc in test_cases:
        print(f"📊 {desc}: {n_obj} objectives, {n_groups} groups, overlap={overlap}")
        
        # Test with two different seeds
        groups1 = simulate_group_generation(n_obj, n_groups, overlap, 42)
        groups2 = simulate_group_generation(n_obj, n_groups, overlap, 1042)
        
        # Check if groups are identical
        identical = groups1 == groups2
        
        print(f"  Groups with seed 42:   {groups1}")
        print(f"  Groups with seed 1042: {groups2}")
        print(f"  Identical: {identical}")
        
        if identical:
            actual_assignments = min(overlap, n_groups)
            if actual_assignments == n_groups:
                print(f"  ⚠️  Every objective goes into ALL {n_groups} groups!")
        else:
            print(f"  ✅ Different seeds produce different groups!")
        
        print()

def analyze_hf3_parameters():
    """Analyze the specific parameters used in HF3."""
    
    print("🔍 Analyzing HF3 parameter calculation...\n")
    
    for n_objectives in [5, 6, 8, 10, 15, 20, 50]:
        import math
        num_groups = int(math.floor(math.sqrt(n_objectives)))
        overlap_factor = int(math.ceil(n_objectives * 0.66))
        
        actual_assignments = min(overlap_factor, num_groups)
        
        print(f"n_objectives={n_objectives:3d}: groups={num_groups:2d}, overlap={overlap_factor:3d}, actual={actual_assignments:2d}", end="")
        
        if actual_assignments == num_groups:
            print(" ❌ All objectives in all groups!")
        else:
            print(" ✅ Groups can vary")

if __name__ == "__main__":
    print("🧪 Debugging why groups don't change with different seeds...\n")
    
    test_group_variations()
    analyze_hf3_parameters()
    
    print("\n💡 CONCLUSION:")
    print("The issue is that with HF3's default parameters, overlap_factor is usually")
    print("much larger than num_groups, so EVERY objective goes into EVERY group!")
    print("This makes the random shuffle irrelevant - groups are always identical.")
    print("\nFor n_objectives < 23, we have num_groups ≤ 4 but overlap ≥ 3,")
    print("so actual_assignments = num_groups, meaning no randomness!")