# pymoo benchmark harness

The [pymoo](https://pymoo.org)-based many-objective benchmark harness used for
the GECCO 2026 HFF paper. It runs an NSGA-II loop whose **survival operator
ranks candidates by HFF angular distance** (TrueNorth or BalancedNorth), and
compares against standard NSGA-II/III on WFG/DTLZ and GNBG-II problems from
1–500 objectives.

> **⚠️ Status — migrated, not yet runnable end-to-end.**
>
> This is the harness *source*, de-branded and moved into the repo. Two pieces
> are still needed before it reproduces the paper figures — see
> [`../docs/PHASE_TWO_BENCHMARK_PLAN.md`](../docs/PHASE_TWO_BENCHMARK_PLAN.md):
>
> 1. **The figure runners** (`corrected_igd_benchmark.py`, `run_wfg*.py`,
>    `wfg_pareto_fronts.py`) that drive the sweep and produce figures 6–13 are
>    not yet in this repo.
> 2. **GNBG figures only:** the GNBG-II problem engine (`import gnbg_gpu`, the
>    wgpu crate at [`minkymorgan/GNBG-II`](https://github.com/minkymorgan/GNBG-II))
>    must be installed. The WFG figures need only pymoo + `hff`.
>
> No benchmark **result data** is shipped — runs regenerate it locally.

## What the paper's figures actually use

The published HFF results are computed by an inline `HFFSurvival` that calls
only three `hff` functions, all already in the public core:

- `hff.calculate_fitness_hf1` — HF1 Balanced
- `hff.calculate_fitness_hf1_enhanced(..., north_pole_method="truenorth")` — TrueNorth
- `hff.calculate_higd` / `hff.calculate_angular_igd` — set-level metrics

So **no Rust porting is required** for the WFG figures. The extra machinery in
`hyperspherical_fitness_pymoo/` (batch, GPU, warp-CDF, HF2/HF3) is exploratory
and was **not** on the paper's figure path.

## What's here

```
benchmark/
├── hyperspherical_fitness_pymoo/   pymoo plugin (survival operators, problems, algorithms)
│   ├── survival.py                 HFF angular-distance survival
│   ├── algorithm.py                NSGA2 variants wired to HFF survival
│   ├── problems/composable.py      WFG / DTLZ / GNBG-II problem composition
│   └── problems/gnbg2_wrapper.py   GNBG-II adapter (needs gnbg_gpu)
├── benchmark_gnbg_*.py             GNBG scaling / implementation benchmarks
├── test_*.py, debug_*.py           standalone checks and probes
├── HFRanking_doc.md                math spec (augmented-zero technique)
├── USAGE_GUIDE.md                  legacy usage notes (some paths predate the migration)
└── DYNAMIC_GROUPS_USAGE.md         grouping/decrowding notes
```

## Running the WFG benchmarks (works now)

```bash
maturin develop --release            # build/install hff into your env
cd benchmark
python run_wfg4_9_higd.py --problem WFG4 --objectives 3   # smoke test
python run_wfg4_9_higd.py            # full WFG4–9 sweep, 10–100 objectives
python run_wfg1_3.py                 # WFG1–3 (Euclidean IGD)
```

Each writes a per-experiment CSV; the HFF (TrueNorth / BalancedNorth) rows use
`hff` directly. Figures are produced from those CSVs (analysis scripts still to
be migrated — see the plan doc).

## Dependencies

- Python ≥ 3.9, `pandas`, `pyarrow`, `numpy`
- `hff` (this repo — `maturin develop --release`)
- `pymoo` — **pin to a version compatible with your numpy.** pymoo 0.6.x uses
  `np.row_stack`, removed in numpy ≥ 2.0; on numpy 2.x the NSGA-II/III baselines
  fail with `module 'numpy' has no attribute 'row_stack'` (the HFF runs are
  unaffected). Use numpy < 2.0 or a patched pymoo until this is pinned.
- `gnbg_gpu` — **only** for GNBG-based benchmarks; from
  [`minkymorgan/GNBG-II`](https://github.com/minkymorgan/GNBG-II). Install it, or
  set `GNBG_GPU_PATH` to its `python/` directory. Without it, GNBG benchmarks are
  disabled (a warning is emitted); WFG/DTLZ need none of this.

## Reproducing the paper (planned)

The full sequence — move the figure runners in, wire GNBG-II, verify numerical
parity with the paper's engine, then regenerate figures — is specified in
[`../docs/PHASE_TWO_BENCHMARK_PLAN.md`](../docs/PHASE_TWO_BENCHMARK_PLAN.md).
