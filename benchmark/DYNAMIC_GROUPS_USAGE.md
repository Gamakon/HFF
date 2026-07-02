# HF3 vs rHF3: Fixed vs Dynamic Group Assignment

## Overview

HF3 Arctic Circle algorithm now comes in two variants for direct comparison:

### **HF3: Fixed Groups (Default)**
- **Algorithm**: `"HF3"`
- Groups are assigned once using fixed random seed (42)
- Same group assignment throughout entire optimization run
- **Reproducible** and **deterministic** behavior
- Good for benchmarking and comparison studies

### **rHF3: Random/Dynamic Groups (New)**
- **Algorithm**: `"rHF3"` 
- Groups are randomly reassigned **each generation**
- Different group assignment every generation using seed = base_seed + generation * 1000
- **Higher exploration** and **diversity**
- Good for complex/noisy optimization landscapes

## Algorithm Selection

### Single Experiment (`run_experiment.py`)
```python
# Line 23: Choose algorithm
ALGORITHM = "HF3"   # Fixed groups
# OR
ALGORITHM = "rHF3"  # Dynamic groups
```

### Parallel Experiments (`run_experiments_parallel.py`)
```python  
# Line 44: Both algorithms included
ALGORITHMS = ["HF1", "NSGA2", "HF3", "rHF3"]  # All four algorithms
```

## New Benchmark Configuration

With rHF3 added, the total experiments are now:
- **Algorithms**: 4 (HF1, NSGA2, HF3, rHF3)  
- **Objectives**: 496 (5-500 range)
- **Total Experiments**: 496 × 4 = **1,984 experiments**

## Technical Details

### Group Assignment Logic
- **Fixed Mode**: `random_seed = 42` (constant)
- **Dynamic Mode**: `random_seed = 42 + generation * 1000` (changes each generation)

### Group Structure (Same for Both Modes)
- **Number of groups**: `floor(√n_objectives)`
- **Overlap factor**: `ceil(n_objectives × 0.66)`
- **Arctic Circle**: Reference points at 0.01 radians from north pole

### Example: 10 objectives
- **Groups**: 3 groups (floor(√10) = 3)
- **Overlap**: 7 factor (ceil(10 × 0.66) = 7)
- **Fixed Mode**: Same 3 groups every generation
- **Dynamic Mode**: New random 3 groups every generation

## When to Use Each Mode

### Use **Fixed Groups** when:
- ✅ Benchmarking against other algorithms
- ✅ Reproducible results needed  
- ✅ Comparative studies
- ✅ Algorithm analysis and debugging

### Use **Dynamic Groups** when:
- ✅ Maximizing exploration in complex landscapes
- ✅ Avoiding local optima in group-dependent problems
- ✅ Testing robustness to group assignment
- ✅ Experimental optimization runs

## Performance Impact

- **Computational**: Minimal overhead (just seed calculation)
- **Memory**: No additional memory usage
- **GPU**: Same GPU acceleration for both modes
- **Convergence**: May require more generations for dynamic mode

## Experimental Results

To compare both modes on your problem:

1. **Run Fixed Mode**: Set `HF3_DYNAMIC_GROUPS = False` and run benchmark
2. **Run Dynamic Mode**: Set `HF3_DYNAMIC_GROUPS = True` and run benchmark  
3. **Compare**: Use `hf1_unified_ranking.py` to analyze results

The dynamic mode may show:
- Higher diversity in intermediate generations
- Different convergence patterns
- Better performance on some problem structures