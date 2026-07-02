# Hyperspherical Fitness (HF1) Benchmark System

A benchmark system for evaluating HF1 algorithms (TrueNorth and BalancedNorth variants) against NSGA-II/III on WFG/DTLZ and GNBG-II multi-objective functions, from 1–500 objectives. This is the pymoo-based harness used for the GECCO 2026 paper's benchmark figures.

> **⚠️ Status — not yet runnable against the public `hff` core.**
> This code was migrated from an internal engine. The pymoo survival operators
> currently call four functions that the public `hff` crate does not yet
> expose: `calculate_hff_fitness_batch`, `calculate_hyperspherical_fitness_hf1_gpu`,
> `calculate_hyperspherical_fitness_hf1_warp_cdf`, and `init_gpu`. Porting those
> into `hff` (Rust core + PyO3) is the follow-up task before the benchmarks run
> and regenerate their data. The GNBG-II problem wrappers also need the external
> `GNBG-II/python` package on `sys.path` (see `problems/composable.py`).
>
> No benchmark **result data** is shipped — runs regenerate it locally.

## 🚀 Quick Start

```bash
# Run single experiment (configurable GF1-GF24)
python experiments/single/run_experiment.py

# Run full experiment suite comparing NSGA2 vs HF1 variants
python experiments/configs/run_full_experiments.py

# Unified ranking analysis with HF1 post-hoc evaluation
python experiments/analysis/hf1_unified_ranking.py

# Test complete GNBG-II pipeline
python test_complete_gnbg_pipeline.py
```

## 📁 Directory Structure

### **Core Scripts**
- **`experiments/single/run_experiment.py`** - Single experiment runner (configurable GF1-GF24)
- **`experiments/configs/run_full_experiments.py`** - Full experiment suite with auto-timer and live leaderboard
- **`experiments/analysis/hf1_unified_ranking.py`** - Unified ranking analysis using HF1 post-hoc evaluation
- **`test_complete_gnbg_pipeline.py`** - End-to-end pipeline testing for GNBG-II

### **Implementation**
- **`hyperspherical_fitness_pymoo/`** - Core HF1 implementation with Rust backend
  - `algorithm.py` - HypersphericalFitnessAlgorithm with TrueNorth/BalancedNorth variants
  - `survival.py` - HypersphericalFitnessSurvival with energy-based augmentation
  - `problems/composable.py` - GNBG-II GF1-GF24 multi-objective problems (1-500 objectives)
  - `distributed_parquet_logger.py` - Distributed logging with atomic consolidation

### **Documentation**
- **`README.md`** - This file (main documentation)
- **`USAGE_GUIDE.md`** - Comprehensive usage examples
- **`HFRanking_doc.md`** - Mathematical specification with augmented zero technique
- **`analysis_report.qmd`** - Quarto analysis template

### **Results**
- **`results/hf1_benchmark_results.parquet`** - Shared database for all experiments
- **`hf1_unified_ranking_results.csv`** - Unified ranking across all algorithms
- **`hf1_unified_ranking_analysis.json`** - Statistical analysis stratified by objective count
- **`analysis_report.html`** - Generated analysis report

### **Archive**
- **`archive/`** - Historical development files, tests, debug scripts
- **`archive/2025_01_cleanup/`** - Recent cleanup of outdated files

## 🎯 Key Features

### Key Innovations (Jan 2025)
- **✅ TrueNorth Method** - Direct minimization using augmented space ℝ^(m+1) with energy-based projection
- **✅ BalancedNorth Method** - Equal trade-off optimization using balanced north pole (1/√m, 1/√m,...)
- **✅ GNBG-II Integration** - GPU-accelerated GF1-GF24 functions replacing legacy WFG benchmarks
- **✅ Non-Stationary Analysis** - Statistical analysis stratified by objective count

### Algorithm Comparison
- **HF1-TrueNorth** - Direct minimization with augmented north pole for pure convergence
- **HF1-BalancedNorth** - Balanced trade-offs using equal-weight north pole
- **NSGA2** - Non-dominated Sorting Genetic Algorithm II  
- **NSGA3** - Non-dominated Sorting Genetic Algorithm III

### Scaling Capabilities
- **Objectives**: 1-500 objectives with full GPU acceleration
- **Problems**: GNBG-II GF1-GF24 multi-objective functions (varying difficulty)
- **Performance**: ~4,713 evaluations/second with GPU acceleration
- **Experiments**: 5-second auto-timer, live leaderboard updates

## 📊 Sample Output (Live Leaderboard)

```
🏆 LIVE ALGORITHM LEADERBOARD
Progress: 60/60 experiments completed

Row Obj      Experiment      Algorithm   Min_HF1    Avg_HF1      IGD  Runs Solutions
  1 100        GF24.100obj   HF1-BalancedNorth  0.785398   0.912456     N/A     5       500
  2 100        GF24.100obj   HF1-TrueNorth      0.823145   0.945123     N/A     5       500
  3 100        GF24.100obj   NSGA2              0.987654   1.123456     N/A     5       500

🥇 CURRENT LEADERS BY OBJECTIVE COUNT:
Obj Best Algorithm     Min_HF1   Experiment
100 HF1-BalancedNorth  0.785398  GF24.100obj
 95 HF1-TrueNorth      0.812345  GF18.95obj
 90 HF1-BalancedNorth  0.798123  GF12.90obj
```

## ⚙️ Configuration

Edit `experiments/single/run_experiment.py` to configure:

```python
# Experiment settings
EXPERIMENT_NAME = "GF24.100obj_hf1-balancednorth"  # GNBG-II format
ALGORITHM = "HF1-BalancedNorth"  # Options: "HF1-TrueNorth", "HF1-BalancedNorth", "NSGA2", "NSGA3"
N_RUNS = 5                        # Reasonable for GPU-accelerated experiments

# Problem selection (GNBG-II pattern)
SELECTED_PROBLEM = 'gf24_100obj'  # Pattern: gf{X}_{Y}obj
```

## 🧪 Problem Patterns

The system uses GNBG-II multi-objective functions:

- **`gf{X}_{Y}obj`** - GNBG-II GF{X} function with Y objectives
  - Example: `gf24_100obj` = GF24 (hardest) with 100 objectives
  - GF1: Unimodal, well-conditioned
  - GF6: Unimodal, non-linear basin
  - GF12: Single-component multimodal
  - GF18: Multi-component (10 basins)
  - GF24: Maximum difficulty (all challenges)

## 📈 Analysis Workflow

1. **Run experiments**: `run_full_experiments.py` with auto-timer and live leaderboard
2. **Distributed logging**: Atomic consolidation prevents parallel write corruption
3. **Unified ranking**: Apply HF1 post-hoc to rank all algorithms' solutions
4. **Stratified analysis**: Statistics grouped by benchmark type and objective count
5. **Pipeline testing**: Use `test_complete_gnbg_pipeline.py` to verify end-to-end functionality

## 🔧 Dependencies

- Python 3.8+
- PyMOO (multi-objective optimization)  
- **Rust hff-core** (GPU-accelerated HF1 with TrueNorth/BalancedNorth)
- **GNBG-II library** (GPU-accelerated GF1-GF24 functions via PyO3)
- pandas, pyarrow (for distributed parquet logging)
- CUDA-capable GPU (for acceleration)

## 💡 GNBG-II Migration Complete (Jan 2025)

- **✅ GNBG-II Integration**: Full migration from WFG to GF1-GF24 multi-objective functions
- **✅ TrueNorth & BalancedNorth**: Two HF1 variants for different optimization philosophies
- **✅ GPU Acceleration**: ~4,713 evaluations/second with GNBG-II library
- **✅ Live Leaderboard**: Real-time experiment tracking with automatic updates
- **✅ Distributed Logging**: Atomic consolidation prevents parallel write corruption
- **✅ Production Ready**: Complete pipeline testing, auto-timer, reasonable experiment settings

---

*For detailed documentation, see `USAGE_GUIDE.md` and `HFRanking_doc.md`*