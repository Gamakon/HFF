#!/usr/bin/env python3
"""
Design a better overlap formula for HF3 that creates meaningful group diversity
"""

import math

def current_formula(n_obj):
    """Current formula that's too aggressive."""
    num_groups = int(math.floor(math.sqrt(n_obj)))
    overlap_factor = int(math.ceil(n_obj * 0.66))
    actual = min(overlap_factor, num_groups)
    return num_groups, overlap_factor, actual

def proposed_formula_v1(n_obj):
    """Proposed: Start with no overlap, gradually increase."""
    num_groups = int(math.floor(math.sqrt(n_obj)))
    
    if n_obj <= 4:
        # No overlap for small problems
        overlap_factor = 1
    elif n_obj <= 9:
        # Minimal overlap
        overlap_factor = 2
    elif n_obj <= 16:
        # Moderate overlap (n_groups - 1)
        overlap_factor = num_groups - 1
    else:
        # For larger problems, use n_groups - 1 to ensure diversity
        overlap_factor = max(2, num_groups - 1)
    
    actual = min(overlap_factor, num_groups)
    return num_groups, overlap_factor, actual

def proposed_formula_v2(n_obj):
    """Proposed v2: Based on your suggestion - each objective in (n_groups - 1) groups."""
    num_groups = int(math.floor(math.sqrt(n_obj)))
    
    if num_groups <= 2:
        # For very small problems, no overlap
        overlap_factor = 1
    else:
        # Each objective appears in (n_groups - 1) groups
        # This ensures each group is missing some objectives
        overlap_factor = num_groups - 1
    
    actual = min(overlap_factor, num_groups)
    return num_groups, overlap_factor, actual

def proposed_formula_v3(n_obj):
    """Proposed v3: More nuanced scaling."""
    num_groups = int(math.floor(math.sqrt(n_obj)))
    
    if n_obj <= 4:
        overlap_factor = 1  # No overlap
    elif n_obj <= 9:
        overlap_factor = min(2, num_groups - 1)  # Light overlap
    elif n_obj <= 25:
        # Medium overlap: about 60% of groups
        overlap_factor = max(2, int(num_groups * 0.6))
    else:
        # Heavy overlap but never all groups
        # Use golden ratio for nice scaling
        overlap_factor = max(3, int(num_groups * 0.618))
    
    # Ensure we never have all objectives in all groups
    overlap_factor = min(overlap_factor, num_groups - 1)
    
    actual = min(overlap_factor, num_groups)
    return num_groups, overlap_factor, actual

def analyze_formulas():
    """Compare different overlap formulas."""
    
    print("Comparing overlap formulas for HF3")
    print("=" * 80)
    print(f"{'n_obj':>6} | {'Current':^20} | {'V1: Stepped':^20} | {'V2: (n-1)':^20} | {'V3: Scaled':^20}")
    print(f"{'':>6} | {'groups overlap actual':^20} | {'groups overlap actual':^20} | {'groups overlap actual':^20} | {'groups overlap actual':^20}")
    print("-" * 80)
    
    test_objectives = [3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25, 30, 50, 100, 200]
    
    for n_obj in test_objectives:
        curr = current_formula(n_obj)
        v1 = proposed_formula_v1(n_obj)
        v2 = proposed_formula_v2(n_obj)
        v3 = proposed_formula_v3(n_obj)
        
        # Format: groups/overlap/actual
        curr_str = f"{curr[0]:2d} / {curr[1]:3d} / {curr[2]:2d}"
        v1_str = f"{v1[0]:2d} / {v1[1]:3d} / {v1[2]:2d}"
        v2_str = f"{v2[0]:2d} / {v2[1]:3d} / {v2[2]:2d}"
        v3_str = f"{v3[0]:2d} / {v3[1]:3d} / {v3[2]:2d}"
        
        # Mark when all objectives go in all groups
        if curr[2] == curr[0]:
            curr_str += " ❌"
        if v1[2] == v1[0]:
            v1_str += " ❌"
        if v2[2] == v2[0]:
            v2_str += " ❌"
        if v3[2] == v3[0]:
            v3_str += " ❌"
            
        print(f"{n_obj:6d} | {curr_str:^20} | {v1_str:^20} | {v2_str:^20} | {v3_str:^20}")
    
    print("\n" + "=" * 80)
    print("Legend: groups / overlap_factor / actual_overlap")
    print("❌ = All objectives in all groups (no diversity)")

def simulate_diversity(n_obj, formula_func):
    """Simulate how many unique group compositions we get."""
    import random
    
    num_groups, overlap_factor, actual = formula_func(n_obj)
    
    # Simulate group generation with different seeds
    unique_patterns = set()
    
    for seed in range(100):  # Test 100 different seeds
        random.seed(seed)
        groups = [[] for _ in range(num_groups)]
        
        for obj_idx in range(n_obj):
            available = list(range(num_groups))
            random.shuffle(available)
            
            for i in range(min(overlap_factor, num_groups)):
                groups[available[i]].append(obj_idx)
        
        # Convert to tuple for hashing
        pattern = tuple(tuple(sorted(g)) for g in groups)
        unique_patterns.add(pattern)
    
    return len(unique_patterns)

def test_diversity():
    """Test how much diversity each formula provides."""
    
    print("\n\nDiversity Analysis (unique patterns from 100 seeds)")
    print("=" * 60)
    print(f"{'n_obj':>6} | {'Current':>10} | {'V1':>10} | {'V2':>10} | {'V3':>10}")
    print("-" * 60)
    
    for n_obj in [6, 10, 16, 25, 50]:
        curr_div = simulate_diversity(n_obj, current_formula)
        v1_div = simulate_diversity(n_obj, proposed_formula_v1)
        v2_div = simulate_diversity(n_obj, proposed_formula_v2)
        v3_div = simulate_diversity(n_obj, proposed_formula_v3)
        
        print(f"{n_obj:6d} | {curr_div:10d} | {v1_div:10d} | {v2_div:10d} | {v3_div:10d}")
    
    print("\nHigher numbers = more diversity = better for rHF3")

if __name__ == "__main__":
    print("🧪 Designing better overlap formula for HF3...\n")
    
    analyze_formulas()
    test_diversity()
    
    print("\n\n💡 RECOMMENDATION:")
    print("Use V2 formula: overlap_factor = num_groups - 1")
    print("This ensures:")
    print("- Each objective appears in (n_groups - 1) groups")
    print("- Each group is missing different objectives")
    print("- Maximum diversity while maintaining good coverage")
    print("- Simple and intuitive rule")