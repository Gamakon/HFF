# Hyperspherical Fitness Benchmark System - Usage Guide

> **⚠️ Legacy document — predates the migration into this repo.** Some commands
> and paths below (`optimize_gnbg2.py`, `run_single_problem.py`,
> `experiments/…`, `results/…`) refer to files that were not moved and do not
> exist here yet. Treat this as historical reference. For the current state and
> the reproduction plan, see [`README.md`](README.md) and
> [`../docs/PHASE_TWO_BENCHMARK_PLAN.md`](../docs/PHASE_TWO_BENCHMARK_PLAN.md).

## Quick Start

### 1. Basic Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd HFF/benchmark

# Install dependencies
pip install numpy pymoo pandas pyarrow

# Build Rust components (if not already built)
cd ../rust
cargo build --release
cd ../benchmark

# Verify installation
python -c "from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm; print('✅ Installation successful')"
```

### 2. Minimal Working Example

```python
#!/usr/bin/env python3
"""
Minimal example: Optimize GNBG2 F1+F5 using HF1
"""

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
from hyperspherical_fitness_pymoo.problems.composable import ComposableBenchmarkProblem

# Create a 2-objective problem
problem = ComposableBenchmarkProblem({
    'gnbg2': [1, 5],  # F1 and F5 as objectives
    'n_var': 30       # 30 decision variables
})

# Create HF1 algorithm
algorithm = HypersphericalFitnessAlgorithm(pop_size=100)

# Run optimization
result = minimize(
    problem,
    algorithm,
    termination=get_termination('n_gen', 50),
    seed=42,
    verbose=True
)

print(f"Optimization complete! Final population: {len(result.F)} solutions")
```

## Configuration Options

### Algorithm Configuration

```python
from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
from pymoo.util.ref_dirs import get_reference_directions

# Basic configuration
algorithm = HypersphericalFitnessAlgorithm(
    pop_size=100,           # Population size
    n_offsprings=100,       # Offspring per generation
    eliminate_duplicates=True  # Remove duplicate solutions
)

# Advanced configuration
algorithm = HypersphericalFitnessAlgorithm(
    pop_size=200,
    n_offsprings=200,
    # Crossover parameters
    crossover_prob=0.9,     # Crossover probability
    crossover_eta=15,       # SBX distribution index
    # Mutation parameters  
    mutation_eta=20,        # Polynomial mutation index
    # Custom reference directions for many objectives
    ref_dirs=get_reference_directions("energy", 10, 150)
)

# Access survival operator parameters
algorithm.survival.alpha = 2.0  # Angular fitness parameter
algorithm.survival.beta = 1.0   # Distance weighting
```

### Problem Configuration

#### Pure GNBG2 Problems

```python
from hyperspherical_fitness_pymoo.problems.composable import (
    ComposableBenchmarkProblem,
    create_gnbg2_problem,
    create_gnbg2_cascade
)

# Single objective (F1 only)
problem_f1 = ComposableBenchmarkProblem({
    'gnbg2': [1],
    'n_var': 30
})

# Multiple specific functions
problem_multi = ComposableBenchmarkProblem({
    'gnbg2': [5, 10, 15, 20],  # F5, F10, F15, F20
    'n_var': 30
})

# Cascade (F1 through F5)
problem_cascade = ComposableBenchmarkProblem({
    'gnbg2': [1, 2, 3, 4, 5],
    'n_var': 30
})

# Using convenience functions
problem_f12 = create_gnbg2_problem([12], n_var=30)
cascade_f1_f10 = create_gnbg2_cascade([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

#### WFG Problems (if PyMOO has WFG support)

```python
# WFG4 with 3 objectives
problem_wfg = ComposableBenchmarkProblem({
    'wfg': {'problem': 4, 'n_obj': 3},
    'n_var': 30
})

# WFG9 with many objectives
problem_wfg_many = ComposableBenchmarkProblem({
    'wfg': {'problem': 9, 'n_obj': 10},
    'n_var': 50  # More variables for many objectives
})
```

#### DTLZ Problems (if PyMOO has DTLZ support)

```python
# DTLZ2 with 3 objectives
problem_dtlz = ComposableBenchmarkProblem({
    'dtlz': {'problem': 2, 'n_obj': 3},
    'n_var': 30
})

# DTLZ7 with 5 objectives
problem_dtlz7 = ComposableBenchmarkProblem({
    'dtlz': {'problem': 7, 'n_obj': 5},
    'n_var': 30
})
```

#### Hybrid Problems

```python
# GNBG2 + WFG hybrid
hybrid_gw = ComposableBenchmarkProblem({
    'gnbg2': [24],  # F24 (composition function)
    'wfg': {'problem': 4, 'n_obj': 3},
    'n_var': 30
})

# GNBG2 + DTLZ hybrid
hybrid_gd = ComposableBenchmarkProblem({
    'gnbg2': [20, 21],
    'dtlz': {'problem': 2, 'n_obj': 3},
    'n_var': 30
})

# All three sources
hybrid_all = ComposableBenchmarkProblem({
    'gnbg2': [24],
    'wfg': {'problem': 9, 'n_obj': 2},
    'dtlz': {'problem': 2, 'n_obj': 2},
    'n_var': 30
})
```

#### Custom Bounds

```python
# Custom bounds for all variables
problem_custom = ComposableBenchmarkProblem({
    'gnbg2': [1, 5],
    'n_var': 30,
    'bounds': {
        'xl': -50.0,  # Lower bound
        'xu': 50.0    # Upper bound
    }
})

# Note: GNBG2 expects [-100, 100] by default
# WFG/DTLZ expect [0, 1] by default
# The system automatically transforms bounds between sources
```

### Logging Configuration

```python
from hyperspherical_fitness_pymoo.logging.parquet_logger import (
    ParquetLogger,
    ParquetLoggerFactory
)

# Basic logger
logger = ParquetLogger()  # Auto-generated filename

# Custom configuration
logger = ParquetLogger(
    filename="results/my_experiment.parquet",
    batch_size=1000,      # Records before writing
    compression='snappy'  # 'snappy', 'gzip', 'lz4', 'brotli'
)

# Factory methods
logger = ParquetLoggerFactory.create_for_experiment(
    "gnbg2_cascade_study",
    output_dir="results/2024_experiments",
    batch_size=500,
    compression='gzip'
)

# For specific run ID
logger = ParquetLoggerFactory.create_for_run(
    run_id=42,
    base_dir="results/runs",
    batch_size=2000
)
```

## Complete Examples

### Example 1: Single Run with Logging

```python
#!/usr/bin/env python3
"""
Complete example with logging and analysis
"""

import numpy as np
from datetime import datetime
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
from hyperspherical_fitness_pymoo.problems.composable import ComposableBenchmarkProblem
from hyperspherical_fitness_pymoo.logging.parquet_logger import ParquetLoggerFactory

# Configuration
PROBLEM_CONFIG = {
    'gnbg2': [1, 5, 10],  # 3 objectives
    'n_var': 30
}
POPULATION_SIZE = 100
N_GENERATIONS = 100
EXPERIMENT_NAME = "gnbg2_f1_f5_f10_study"

# Create components
problem = ComposableBenchmarkProblem(PROBLEM_CONFIG)
algorithm = HypersphericalFitnessAlgorithm(pop_size=POPULATION_SIZE)
logger = ParquetLoggerFactory.create_for_experiment(EXPERIMENT_NAME)

# Tracking variables
start_time = datetime.now()
generation_times = []

# Callback for logging
def log_callback(algorithm):
    gen_start = datetime.now()
    
    # Get HF1 scores from survival
    hf1_scores = None
    if hasattr(algorithm.survival, 'last_fitness_scores'):
        hf1_scores = algorithm.survival.last_fitness_scores
    
    # Calculate timing
    if len(generation_times) > 0:
        gen_time = (datetime.now() - generation_times[-1]).total_seconds() * 1000
    else:
        gen_time = 0
    
    generation_times.append(datetime.now())
    
    # Log generation
    logger.log_generation(
        run_id=1,
        algorithm='HF1',
        problem_config=problem.get_metadata(),
        generation=algorithm.n_gen,
        population=algorithm.pop,
        hf1_scores=hf1_scores,
        timing_data={
            'generation_time_ms': gen_time,
            'evaluation_time_ms': gen_time * 0.6,  # Estimate
            'selection_time_ms': gen_time * 0.3,   # Estimate
            'total_evaluations': algorithm.evaluator.n_eval
        },
        experiment_name=EXPERIMENT_NAME,
        algorithm_config={
            'pop_size': POPULATION_SIZE,
            'crossover_prob': algorithm.mating.crossover.prob,
            'crossover_eta': algorithm.mating.crossover.eta,
            'mutation_eta': algorithm.mating.mutation.eta
        }
    )

# Run optimization
print(f"🚀 Starting optimization: {EXPERIMENT_NAME}")
print(f"   Problem: {problem.n_obj} objectives, {problem.n_var} variables")
print(f"   Algorithm: HF1 with population size {POPULATION_SIZE}")
print(f"   Termination: {N_GENERATIONS} generations")

result = minimize(
    problem,
    algorithm,
    termination=get_termination('n_gen', N_GENERATIONS),
    callback=log_callback,
    seed=42,
    verbose=True
)

# Finalize logging
logger.finalize()

# Analysis
total_time = (datetime.now() - start_time).total_seconds()
print(f"\n✅ Optimization complete!")
print(f"   Total time: {total_time:.2f} seconds")
print(f"   Evaluations: {algorithm.evaluator.n_eval}")
print(f"   Final population: {len(result.F)} solutions")
print(f"   Best objectives: {np.min(result.F, axis=0)}")
print(f"   Results saved to: {logger.filename}")

# Read and analyze results
from hyperspherical_fitness_pymoo.logging.parquet_logger import analyze_parquet_results
analysis = analyze_parquet_results(str(logger.filename))
print(f"\n📊 Results Analysis:")
print(f"   Total records: {analysis['total_records']}")
print(f"   Generations: {analysis['generation_range']}")
print(f"   Mean generation time: {analysis['timing']['mean_generation_time_ms']:.2f} ms")
```

### Example 2: Benchmark Suite Runner

```python
#!/usr/bin/env python3
"""
Run systematic benchmark suite
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
from hyperspherical_fitness_pymoo.problems.composable import ComposableBenchmarkFactory
from hyperspherical_fitness_pymoo.logging.parquet_logger import ParquetLoggerFactory

class BenchmarkRunner:
    def __init__(self, output_dir="benchmark_results", n_runs=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_runs = n_runs
        self.results_summary = []
        
    def run_problem(self, problem_config, problem_name, run_id):
        """Run single problem instance"""
        
        # Create problem
        problem = ComposableBenchmarkProblem(problem_config)
        
        # Adjust population size based on objectives
        pop_size = max(100, problem.n_obj * 20)
        
        # Create algorithm
        algorithm = HypersphericalFitnessAlgorithm(pop_size=pop_size)
        
        # Create logger
        logger = ParquetLoggerFactory.create_for_experiment(
            f"{problem_name}_run{run_id}",
            output_dir=str(self.output_dir),
            batch_size=1000
        )
        
        # Callback for logging
        def callback(algo):
            hf1_scores = getattr(algo.survival, 'last_fitness_scores', None)
            logger.log_generation(
                run_id=run_id,
                algorithm='HF1',
                problem_config=problem.get_metadata(),
                generation=algo.n_gen,
                population=algo.pop,
                hf1_scores=hf1_scores,
                experiment_name=problem_name
            )
        
        # Run optimization
        start_time = datetime.now()
        
        result = minimize(
            problem,
            algorithm,
            termination=get_termination('n_gen', 50),
            callback=callback,
            seed=run_id * 42,  # Different seed per run
            verbose=False
        )
        
        runtime = (datetime.now() - start_time).total_seconds()
        
        # Finalize logger
        logger.finalize()
        
        # Store summary
        summary = {
            'problem_name': problem_name,
            'run_id': run_id,
            'n_objectives': problem.n_obj,
            'n_variables': problem.n_var,
            'pop_size': pop_size,
            'runtime': runtime,
            'n_evaluations': algorithm.evaluator.n_eval,
            'final_pop_size': len(result.F),
            'min_objectives': np.min(result.F, axis=0).tolist(),
            'mean_objectives': np.mean(result.F, axis=0).tolist(),
            'log_file': str(logger.filename)
        }
        
        return summary
    
    def run_benchmark_suite(self):
        """Run complete benchmark suite"""
        
        # Define test problems
        test_suite = [
            # Low dimensional
            ({'gnbg2': [1, 5], 'n_var': 30}, 'gnbg2_f1_f5'),
            ({'gnbg2': [1, 5, 10], 'n_var': 30}, 'gnbg2_f1_f5_f10'),
            
            # Medium dimensional
            ({'gnbg2': list(range(1, 6)), 'n_var': 30}, 'gnbg2_cascade_5'),
            ({'gnbg2': list(range(1, 9)), 'n_var': 30}, 'gnbg2_cascade_8'),
            
            # Many objectives
            ({'gnbg2': list(range(1, 11)), 'n_var': 30}, 'gnbg2_cascade_10'),
            ({'gnbg2': list(range(1, 16)), 'n_var': 30}, 'gnbg2_cascade_15'),
            
            # High complexity
            ({'gnbg2': [24], 'n_var': 30}, 'gnbg2_f24_composition'),
            ({'gnbg2': [20, 21, 22, 23, 24], 'n_var': 30}, 'gnbg2_hybrid_complex'),
        ]
        
        # Add WFG/DTLZ if available
        try:
            test_suite.extend([
                ({'wfg': {'problem': 4, 'n_obj': 3}, 'n_var': 30}, 'wfg4_3obj'),
                ({'dtlz': {'problem': 2, 'n_obj': 3}, 'n_var': 30}, 'dtlz2_3obj'),
            ])
        except:
            print("⚠️ WFG/DTLZ not available, skipping")
        
        # Run all problems
        total_problems = len(test_suite) * self.n_runs
        completed = 0
        
        print(f"🚀 Running benchmark suite: {len(test_suite)} problems × {self.n_runs} runs")
        
        for problem_config, problem_name in test_suite:
            print(f"\n📊 Problem: {problem_name}")
            
            for run_id in range(1, self.n_runs + 1):
                print(f"   Run {run_id}/{self.n_runs}...", end='', flush=True)
                
                try:
                    summary = self.run_problem(problem_config, problem_name, run_id)
                    self.results_summary.append(summary)
                    print(f" ✅ ({summary['runtime']:.1f}s)")
                except Exception as e:
                    print(f" ❌ Error: {e}")
                
                completed += 1
                
        # Save summary
        df = pd.DataFrame(self.results_summary)
        summary_file = self.output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_file, index=False)
        
        print(f"\n✅ Benchmark complete!")
        print(f"   Total runs: {completed}/{total_problems}")
        print(f"   Summary saved to: {summary_file}")
        
        # Print aggregate statistics
        print(f"\n📊 Aggregate Statistics:")
        for problem_name in df['problem_name'].unique():
            problem_data = df[df['problem_name'] == problem_name]
            print(f"\n   {problem_name}:")
            print(f"      Avg runtime: {problem_data['runtime'].mean():.2f}s ± {problem_data['runtime'].std():.2f}s")
            print(f"      Avg evaluations: {problem_data['n_evaluations'].mean():.0f}")

# Run benchmark
if __name__ == "__main__":
    runner = BenchmarkRunner(
        output_dir="benchmark_results_2024",
        n_runs=5
    )
    runner.run_benchmark_suite()
```

### Example 3: Comparison with Other Algorithms

```python
#!/usr/bin/env python3
"""
Compare HF1 with NSGA-II and NSGA-III
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

from hyperspherical_fitness_pymoo.algorithm import HypersphericalFitnessAlgorithm
from hyperspherical_fitness_pymoo.problems.composable import ComposableBenchmarkProblem

# Problem configuration
problem = ComposableBenchmarkProblem({
    'gnbg2': [1, 5, 10],  # 3 objectives
    'n_var': 30
})

# Common parameters
pop_size = 100
n_gen = 100
seed = 42

# Algorithms to compare
algorithms = {
    'HF1': HypersphericalFitnessAlgorithm(pop_size=pop_size),
    'NSGA-II': NSGA2(pop_size=pop_size),
    'NSGA-III': NSGA3(
        pop_size=pop_size,
        ref_dirs=get_reference_directions("das-dennis", 3, n_partitions=12)
    )
}

# Run comparisons
results = {}
for name, algorithm in algorithms.items():
    print(f"\n🚀 Running {name}...")
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination('n_gen', n_gen),
        seed=seed,
        verbose=True
    )
    
    results[name] = result
    print(f"   Final population: {len(result.F)} solutions")
    print(f"   Best objectives: {np.min(result.F, axis=0)}")

# Analyze results
print("\n📊 Comparison Summary:")
print(f"{'Algorithm':<10} {'Pop Size':<10} {'Min F1':<10} {'Min F5':<10} {'Min F10':<10}")
print("-" * 50)

for name, result in results.items():
    min_objs = np.min(result.F, axis=0)
    print(f"{name:<10} {len(result.F):<10} {min_objs[0]:<10.4f} {min_objs[1]:<10.4f} {min_objs[2]:<10.4f}")

# Plot results (if 2 or 3 objectives)
if problem.n_obj == 2:
    plt.figure(figsize=(12, 4))
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        plt.scatter(result.F[:, 0], result.F[:, 1], alpha=0.6)
        plt.xlabel('F1')
        plt.ylabel('F5')
        plt.title(name)
    plt.tight_layout()
    plt.savefig('algorithm_comparison_2d.png')
    print(f"\n📈 Plots saved to algorithm_comparison_2d.png")
```

### Example 4: Custom Callback with Real-time Monitoring

```python
#!/usr/bin/env python3
"""
Real-time monitoring with custom callback
"""

import numpy as np
from datetime import datetime
from collections import deque

class RealTimeMonitor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.hv_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.diversity_history = deque(maxlen=window_size)
        self.last_time = datetime.now()
        
    def __call__(self, algorithm):
        # Calculate metrics
        current_time = datetime.now()
        gen_time = (current_time - self.last_time).total_seconds()
        self.last_time = current_time
        
        # Get objectives
        F = algorithm.pop.get("F")
        
        # Simple diversity metric (average pairwise distance)
        if len(F) > 1:
            distances = []
            for i in range(len(F)):
                for j in range(i+1, len(F)):
                    distances.append(np.linalg.norm(F[i] - F[j]))
            diversity = np.mean(distances) if distances else 0
        else:
            diversity = 0
        
        # Update histories
        self.time_history.append(gen_time)
        self.diversity_history.append(diversity)
        
        # Print real-time stats
        if algorithm.n_gen % 10 == 0:
            print(f"\n📊 Generation {algorithm.n_gen}")
            print(f"   Population size: {len(F)}")
            print(f"   Min objectives: {np.min(F, axis=0)}")
            print(f"   Diversity: {diversity:.4f}")
            print(f"   Gen time: {gen_time:.2f}s (avg: {np.mean(self.time_history):.2f}s)")
            
            if hasattr(algorithm.survival, 'last_fitness_scores'):
                hf1_scores = algorithm.survival.last_fitness_scores
                print(f"   HF1 scores: min={np.min(hf1_scores):.4f}, "
                      f"max={np.max(hf1_scores):.4f}, "
                      f"mean={np.mean(hf1_scores):.4f}")

# Use the monitor
problem = ComposableBenchmarkProblem({'gnbg2': [1, 5], 'n_var': 30})
algorithm = HypersphericalFitnessAlgorithm(pop_size=100)
monitor = RealTimeMonitor()

result = minimize(
    problem,
    algorithm,
    termination=get_termination('n_gen', 100),
    callback=monitor,
    seed=42,
    verbose=False  # Disable default output
)
```

## Running Scripts

### Command Line Execution

```bash
# Basic run
python optimize_gnbg2.py

# With environment variables
export GNBG2_DATA_PATH=/path/to/gnbg2/data
export RUST_LOG=info
python optimize_gnbg2.py

# Parallel runs with different seeds
for seed in 1 2 3 4 5; do
    python optimize_gnbg2.py --seed $seed &
done
wait

# Run with profiling
python -m cProfile -o profile.stats optimize_gnbg2.py
```

### Batch Processing Script

```bash
#!/bin/bash
# run_benchmarks.sh

# Configuration
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"
N_RUNS=5
PROBLEMS=("gnbg2_f1_f5" "gnbg2_cascade_10" "gnbg2_f24")

# Create output directory
mkdir -p $OUTPUT_DIR

# Run each problem
for problem in "${PROBLEMS[@]}"; do
    echo "Running $problem..."
    for run in $(seq 1 $N_RUNS); do
        python run_single_problem.py \
            --problem $problem \
            --run-id $run \
            --output-dir $OUTPUT_DIR \
            > "$OUTPUT_DIR/${problem}_run${run}.log" 2>&1 &
    done
done

# Wait for all jobs to complete
wait
echo "All benchmarks complete!"

# Generate summary report
python generate_report.py --input-dir $OUTPUT_DIR
```

## Troubleshooting Common Issues

### 1. Module Import Errors

```python
# Add to the beginning of your script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

### 2. GNBG2 Data Path Issues

```python
import os

# Set programmatically
os.environ['GNBG2_DATA_PATH'] = '/path/to/gnbg2/data'

# Or check if set
if 'GNBG2_DATA_PATH' not in os.environ:
    print("Warning: GNBG2_DATA_PATH not set!")
    # Try to auto-detect
    possible_paths = [
        '../data/gnbg2',
        './gnbg_data',
        '/usr/local/share/gnbg2'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['GNBG2_DATA_PATH'] = path
            break
```

### 3. Memory Management for Large Problems

```python
# For many objectives or large populations
import gc

# Run with periodic garbage collection
for generation in range(n_generations):
    # ... optimization step ...
    
    if generation % 10 == 0:
        gc.collect()  # Force garbage collection
```

### 4. Debugging FFI Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test FFI components individually
from hyperspherical_fitness_pymoo.problems.gnbg2_wrapper import GNBG2Wrapper

wrapper = GNBG2Wrapper()
try:
    # Test single evaluation
    result = wrapper.problems[1].evaluate(np.random.rand(30))
    print(f"F1 evaluation successful: {result}")
except Exception as e:
    print(f"FFI Error: {e}")
    import traceback
    traceback.print_exc()
```

## Performance Tips

### 1. Population Size Guidelines

```python
def get_recommended_pop_size(n_objectives):
    """Get recommended population size based on objectives"""
    if n_objectives <= 3:
        return 100
    elif n_objectives <= 5:
        return 150
    elif n_objectives <= 10:
        return 200
    else:
        return min(300, n_objectives * 20)
```

### 2. Efficient Batch Evaluation

```python
# For expensive problems, use vectorized evaluation
class BatchedProblem(ComposableBenchmarkProblem):
    def _evaluate(self, X, out):
        # Evaluate in batches to utilize CPU cache
        batch_size = 100
        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_obj))
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            
            # Call parent evaluation
            batch_out = {}
            super()._evaluate(batch_X, batch_out)
            F[i:batch_end] = batch_out['F']
        
        out['F'] = F
```

### 3. Parallel Evaluation (if supported)

```python
from multiprocessing import Pool

class ParallelProblem(ComposableBenchmarkProblem):
    def __init__(self, config, n_processes=4):
        super().__init__(config)
        self.n_processes = n_processes
        
    def _evaluate(self, X, out):
        with Pool(self.n_processes) as pool:
            # Parallel evaluation
            results = pool.map(self._evaluate_single, X)
        out['F'] = np.array(results)
    
    def _evaluate_single(self, x):
        # Evaluate single solution
        out = {}
        super()._evaluate(x.reshape(1, -1), out)
        return out['F'][0]
```

## Next Steps

1. **Explore Advanced Configurations**: Try different reference directions, population sizes, and operator parameters
2. **Implement Custom Problems**: Extend `ComposableBenchmarkProblem` for your specific use cases
3. **Analyze Results**: Use the Parquet files for detailed performance analysis
4. **Compare Algorithms**: Benchmark against other MOEAs using the same problem suite
5. **Contribute**: Add new problem sources, algorithms, or analysis tools

For more examples and advanced usage, see the `examples/` directory in the repository.