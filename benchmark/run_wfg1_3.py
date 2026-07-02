#!/usr/bin/env python3
"""
WFG1-3 Benchmark (Euclidean IGD)
================================
Runs WFG1-3 problems which have NON-spherical Pareto fronts:
- WFG1: Convex, mixed (Dirichlet sampling on scaled simplex)
- WFG2: Convex, disconnected (same as WFG1)
- WFG3: Linear, degenerate (1D interpolation)

Uses analytical Pareto front sampling from wfg_pareto_fronts.py
which works at any dimension (bypasses pymoo's limitation).

NOTE: HIGD is NOT used for WFG1-3 because their Pareto fronts
are not hyperspheres. Standard Euclidean IGD is calculated.
"""

import sys
import os

# Add demo directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corrected_igd_benchmark import (
    run_benchmark_suite,
    STUDY_OBJECTIVES,
    STUDY_ALGORITHMS,
    STUDY_N_RUNS,
)

# WFG1-3: Non-spherical Pareto fronts (no HIGD)
WFG_TRADITIONAL_PROBLEMS = ['WFG1', 'WFG2', 'WFG3']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run WFG1-3 benchmarks (traditional IGD)')
    parser.add_argument('--problem', type=str, choices=WFG_TRADITIONAL_PROBLEMS,
                        help='Run specific problem only')
    parser.add_argument('--objectives', type=str, default=None,
                        help='Comma-separated objective counts (default: 10-15 - limited due to memory)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str,
                        default='analysis/data/igd_corrected_results.csv',
                        help='Output CSV file')

    args = parser.parse_args()

    # Determine problems
    if args.problem:
        problems = [args.problem]
    else:
        problems = WFG_TRADITIONAL_PROBLEMS

    # Determine objectives
    # With analytical sampling, we can now run at any dimension
    if args.objectives:
        objectives = [int(x.strip()) for x in args.objectives.split(',')]
    else:
        # Full range matching WFG4-9 studies
        objectives = list(range(10, 101, 10))  # 10, 20, ..., 100
        print(f"Running objectives: {objectives}")

    print("=" * 70)
    print("WFG1-3 BENCHMARK (EUCLIDEAN IGD)")
    print("=" * 70)
    print(f"Problems: {problems}")
    print(f"Objectives: {objectives}")
    print(f"Algorithms: {STUDY_ALGORITHMS}")
    print(f"Runs per config: {STUDY_N_RUNS}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print()
    print("Using analytical Pareto front sampling (wfg_pareto_fronts.py)")
    print("NOTE: Euclidean IGD is calculated, NOT HIGD (non-spherical fronts)")
    print()

    # Run benchmarks
    df = run_benchmark_suite(
        problems=problems,
        objectives=objectives,
        algorithms=STUDY_ALGORITHMS,
        n_runs=STUDY_N_RUNS,
        output_file=args.output,
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
